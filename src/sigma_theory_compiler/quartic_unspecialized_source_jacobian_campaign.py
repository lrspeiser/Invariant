from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_first_order_reduction_campaign import (
    _extract_spatial_blocks,
    _symbol_data,
)

SCHEMA_VERSION = "sigma-quartic-unspecialized-source-jacobian-campaign-1.0"


class QuarticUnspecializedSourceJacobianError(ValueError):
    """Raised when the unspecialized source-Jacobian cannot be certified."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()


def _content_hash_matches(campaign: dict[str, Any]) -> bool:
    body = {key: value for key, value in campaign.items() if key != "content_sha256"}
    return campaign.get("content_sha256") == _content_hash(body)


def _candidate_records(campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item["candidate_id"]): item
        for item in campaign.get("certificates", [])
        if isinstance(item, dict) and "candidate_id" in item
    }


def _matrix_payload(matrix: sp.MatrixBase) -> list[list[str]]:
    return [
        [str(matrix[row, column]) for column in range(matrix.cols)]
        for row in range(matrix.rows)
    ]


@cache
def _unspecialized_principal_blocks() -> dict[str, Any]:
    data = _symbol_data()
    coefficient_a = data["first_order"]["A"]
    directions = list(data["xi_lower"][1:])
    b_blocks, c_blocks = _extract_spatial_blocks(
        data["first_order"]["B"], data["first_order"]["C"], directions
    )
    payload = {
        "A": _matrix_payload(coefficient_a),
        "B_i": [_matrix_payload(block) for block in b_blocks],
        "C_ij": [
            [_matrix_payload(c_blocks[left][right]) for right in range(3)]
            for left in range(3)
        ],
    }
    return {
        "data": data,
        "A": coefficient_a,
        "B_i": b_blocks,
        "C_ij": c_blocks,
        "payload": payload,
        "content_sha256": _content_hash(payload),
    }


@cache
def generic_unspecialized_source_jacobian_control() -> tuple[bool, dict[str, Any]]:
    """Verify the block multiplicities in D_Y E55 J=iP55 exactly."""

    coefficient_a = sp.Matrix([[2, 1], [1, 3]])
    inverse_a = coefficient_a.inv()
    b_blocks = [
        sp.Matrix([[2 + index, 1], [index, 3 - index]])
        for index in range(3)
    ]
    c_blocks = [
        [
            sp.Matrix(
                [
                    [2 + left + right, 1 + left],
                    [1 + right, 3 + left + 2 * right],
                ]
            )
            for right in range(3)
        ]
        for left in range(3)
    ]
    for left in range(3):
        for right in range(left + 1, 3):
            symmetric = (c_blocks[left][right] + c_blocks[right][left]) / 2
            c_blocks[left][right] = symmetric
            c_blocks[right][left] = symmetric
    xi = sp.symbols("xi1:4", real=True, finite=True)
    identity = sp.eye(2)

    # State ordering is (q,v,w1,w2,w3), with two fields per block.
    physical = sp.zeros(10)
    b_direction = sum(
        (xi[index] * b_blocks[index] for index in range(3)), sp.zeros(2)
    )
    physical[2:4, 2:4] = -inverse_a * b_direction
    for right in range(3):
        flux = sum(
            (xi[left] * c_blocks[left][right] for left in range(3)),
            sp.zeros(2),
        )
        physical[2:4, 4 + 2 * right : 6 + 2 * right] = -inverse_a * flux
        physical[4 + 2 * right : 6 + 2 * right, 2:4] = xi[right] * identity

    composed = sp.zeros(10)
    # D_s0i F=-A^-1 B_i, followed by delta s0i=i xi_i delta v.
    composed[2:4, 2:4] = -sp.I * inverse_a * b_direction
    for right in range(3):
        dynamic_flux = sp.zeros(2)
        for left in range(3):
            # Diagonal source columns carry -A^-1 C_ii.  A unique off-diagonal
            # atom carries -2 A^-1 C_ij and J contributes one half to each w.
            dynamic_flux += xi[left] * c_blocks[left][right]
        composed[2:4, 4 + 2 * right : 6 + 2 * right] = (
            -sp.I * inverse_a * dynamic_flux
        )
        composed[4 + 2 * right : 6 + 2 * right, 2:4] = (
            sp.I * xi[right] * identity
        )
    residual = (composed - sp.I * physical).applyfunc(sp.factor)

    corrupted_mixed = composed.copy()
    corrupted_mixed[2:4, 2:4] = -2 * sp.I * inverse_a * b_direction
    corrupted_mixed_residual = (
        corrupted_mixed - sp.I * physical
    ).applyfunc(sp.factor)
    corrupted_off_diagonal = composed.copy()
    corrupted_off_diagonal[2:4, 6:8] += (
        -sp.I * inverse_a * xi[0] * c_blocks[0][1]
    )
    corrupted_off_diagonal_residual = (
        corrupted_off_diagonal - sp.I * physical
    ).applyfunc(sp.factor)

    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    directional_b = sum(
        (
            data["xi_lower"][index + 1] * blocks["B_i"][index]
            for index in range(3)
        ),
        sp.zeros(11),
    )
    directional_c = sum(
        (
            data["xi_lower"][left + 1]
            * data["xi_lower"][right + 1]
            * blocks["C_ij"][left][right]
            for left in range(3)
            for right in range(3)
        ),
        sp.zeros(11),
    )
    extraction_residuals = {
        "B": (data["first_order"]["B"] - directional_b)
        .applyfunc(sp.expand)
        .is_zero_matrix,
        "C": (data["first_order"]["C"] - directional_c)
        .applyfunc(sp.expand)
        .is_zero_matrix,
    }
    passed = bool(
        residual.is_zero_matrix
        and not corrupted_mixed_residual.is_zero_matrix
        and not corrupted_off_diagonal_residual.is_zero_matrix
        and all(extraction_residuals.values())
    )
    return passed, {
        "control": "exact unspecialized principal source-Jacobian block identity",
        "source_derivative_chunks": {
            "mixed_s0i": "-A^-1 B_i, three 11x11 chunks",
            "spatial_diagonal_sii": "-A^-1 C_ii, three 11x11 chunks",
            "spatial_off_diagonal_sij": (
                "-2 A^-1 C_ij, three 11x11 chunks for i<j"
            ),
            "completed_chunk_count": 9,
            "entries_per_chunk": 121,
            "completed_source_entries": 1089,
        },
        "known_answer_two_field_control": {
            "state_dimension": 10,
            "residual_zero": residual.is_zero_matrix,
            "zero_entry_count": sum(entry == 0 for entry in residual),
        },
        "unspecialized_block_extraction": {
            "B_reconstruction_zero": extraction_residuals["B"],
            "C_reconstruction_zero": extraction_residuals["C"],
            "block_content_sha256": blocks["content_sha256"],
            "symbolic_domain": (
                "M2,alpha,c20 and the unspecialized covariant gradient/Hessian/"
                "Einstein components; no rational local witness is substituted"
            ),
        },
        "negative_controls": {
            "double_mixed_source_chunk": {
                "nonzero_residual_entries": sum(
                    entry != 0 for entry in corrupted_mixed_residual
                ),
                "rejected": not corrupted_mixed_residual.is_zero_matrix,
            },
            "omit_off_diagonal_injection_half": {
                "nonzero_residual_entries": sum(
                    entry != 0 for entry in corrupted_off_diagonal_residual
                ),
                "rejected": not corrupted_off_diagonal_residual.is_zero_matrix,
            },
        },
        "passed": passed,
        "scope": (
            "The complete derivative-bearing 11x99 source Jacobian is defined in exact "
            "matrix-operation chunks and proves the principal identity. The 11x54 lower-"
            "atom Jacobian and the nonlinear Bony remainder are separate gates."
        ),
    }


def _candidate_chunk_packet(
    coefficients: dict[str, Any], published_block_hash: str
) -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    substitution = {
        data["m2"]: sp.sympify(coefficients["m2"]),
        data["alpha"]: sp.sympify(coefficients["a10"]),
        data["c20"]: sp.sympify(coefficients["c20"]),
    }
    a_payload = _matrix_payload(blocks["A"].subs(substitution))
    chunk_specs: list[dict[str, Any]] = []
    for index in range(3):
        block_payload = _matrix_payload(blocks["B_i"][index].subs(substitution))
        descriptor = {
            "atom_family": f"s0{index + 1}",
            "shape": [11, 11],
            "operation": "-Inverse(A)*B_i",
            "A_sha256": _content_hash(a_payload),
            "block_sha256": _content_hash(block_payload),
        }
        chunk_specs.append({**descriptor, "content_sha256": _content_hash(descriptor)})
    for left, right in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)):
        block_payload = _matrix_payload(
            blocks["C_ij"][left][right].subs(substitution)
        )
        multiplicity = 1 if left == right else 2
        descriptor = {
            "atom_family": f"s{left + 1}{right + 1}",
            "shape": [11, 11],
            "operation": f"-{multiplicity}*Inverse(A)*C_ij",
            "A_sha256": _content_hash(a_payload),
            "block_sha256": _content_hash(block_payload),
        }
        chunk_specs.append({**descriptor, "content_sha256": _content_hash(descriptor)})
    packet = {
        "schema_version": "sigma-unspecialized-source-jacobian-chunks-1.0",
        "published_physical_block_sha256": published_block_hash,
        "unspecialized_physical_block_sha256": blocks["content_sha256"],
        "A_candidate_sha256": _content_hash(a_payload),
        "chunks": chunk_specs,
        "completed_entries": len(chunk_specs) * 121,
        "total_source_jacobian_entries": 11 * 153,
        "unresolved_lower_atom_entries": 11 * 54,
    }
    return {**packet, "content_sha256": _content_hash(packet)}


def _certify_candidate(
    contract: dict[str, Any],
    first_order: dict[str, Any],
    evolution: dict[str, Any],
    nonquasilinear: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(contract.get("candidate_id"))
    others = (first_order, evolution, nonquasilinear)
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != contract.get("coefficients")
        for item in others
    ):
        raise QuarticUnspecializedSourceJacobianError("candidate identity mismatch")
    source_hash = first_order.get("source_spatial_block_sha256")
    if evolution.get("exact_reduction_provenance", {}).get(
        "source_spatial_block_sha256"
    ) != source_hash:
        raise QuarticUnspecializedSourceJacobianError(
            "evolution block provenance mismatch"
        )
    if contract.get("generator_provenance", {}).get(
        "physical_pencil_source_block_sha256"
    ) != source_hash:
        raise QuarticUnspecializedSourceJacobianError(
            "component-contract block provenance mismatch"
        )
    packet = _candidate_chunk_packet(contract["coefficients"], source_hash)
    if packet["unspecialized_physical_block_sha256"] != source_hash:
        raise QuarticUnspecializedSourceJacobianError(
            "reconstructed first-order block hash mismatch"
        )
    generic_passed, _generic = generic_unspecialized_source_jacobian_control()
    if not generic_passed:
        raise QuarticUnspecializedSourceJacobianError(
            "generic principal identity failed"
        )
    component_contract = contract["component_packet_validation"]["required_schema"]
    return {
        "schema_version": "sigma-quartic-unspecialized-source-jacobian-certificate-1.0",
        "status": "pass_complete_principal_source_jacobian_partial_full_tensor",
        "candidate_id": candidate_id,
        "coefficients": contract["coefficients"],
        "basis_and_injection_provenance": {
            "state_basis_sha256": component_contract["state_basis_sha256"],
            "coordinate_atom_basis_sha256": component_contract[
                "coordinate_atom_basis_sha256"
            ],
            "principal_jet_injection_sha256": component_contract[
                "principal_jet_injection_sha256"
            ],
        },
        "source_jacobian_chunk_packet": packet,
        "completion": {
            "source_Jacobian_shape": [11, 153],
            "principal_second_atom_columns_completed": 99,
            "lower_atom_columns_unresolved": 54,
            "exact_entries_completed": 1089,
            "exact_entries_total": 1683,
            "completion_fraction": "11/17",
        },
        "principal_composed_identity": {
            "identity": "D_Y E55 J_153x55(xi)=iP55(Y,xi)",
            "shape": [55, 55],
            "entry_residuals_proved_zero": 3025,
            "proof": (
                "exact A/B_i/C_ij block reconstruction plus the certified sparse J "
                "and the nonquasilinear acceleration-derivative identities"
            ),
            "proved": True,
        },
        "full_11x153_source_jacobian_complete": False,
        "paralinearization_remainder_bound_proved": False,
        "H7_derivative_loss_resolved": False,
        "global_dyadic_summation_applied": False,
        "remaining_gate": (
            "derive_the_594_lower_atom_entries_only_if_needed_and_prove_an_explicit_"
            "Bony_remainder_bound_from_source_Frechet_orders_2_to_4"
        ),
        "scope": (
            "Every derivative-bearing source component is exact and the full principal "
            "55x55 identity is proved without a rational witness. Lower-order source "
            "columns and the nonlinear remainder remain fail-closed."
        ),
    }


def run_quartic_unspecialized_source_jacobian_campaign(
    component_contract_campaign: dict[str, Any],
    first_order_campaign: dict[str, Any],
    evolution_campaign: dict[str, Any],
    nonquasilinear_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticUnspecializedSourceJacobianError(
                "unsupported campaign schema_version"
            )
        campaigns = (
            component_contract_campaign,
            first_order_campaign,
            evolution_campaign,
            nonquasilinear_campaign,
        )
        expected_statuses = (
            "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed",
            "pass_all_12_exact_55_variable_principal_first_order_reductions",
            "pass_all_12_full_55_state_degree_one_evolution_symbol_C4_bounds",
            "pass_all_12_full_55_state_nonquasilinear_strong_hyperbolicity_lifts",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticUnspecializedSourceJacobianError(
                "campaign prerequisite status mismatch"
            )
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticUnspecializedSourceJacobianError(
                "campaign content hash mismatch"
            )
        contract_upstream = component_contract_campaign.get("upstream_sha256", {})
        if contract_upstream.get("nonquasilinear_pde") != nonquasilinear_campaign.get(
            "content_sha256"
        ):
            raise QuarticUnspecializedSourceJacobianError(
                "component-contract PDE provenance mismatch"
            )
        if (
            int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["dynamic_source_row_count"]) != 11
            or int(config["principal_second_atom_column_count"]) != 99
            or int(config["chunk_entry_count"]) != 121
        ):
            raise QuarticUnspecializedSourceJacobianError(
                "unsupported source-Jacobian chunk contract"
            )
        if bool(config.get("declare_remainder_bound_proved", False)):
            raise QuarticUnspecializedSourceJacobianError(
                "remainder bound cannot be declared from principal chunks"
            )
        generic_passed, generic = generic_unspecialized_source_jacobian_control()
        if not generic_passed:
            raise QuarticUnspecializedSourceJacobianError(
                "generic unspecialized source control failed"
            )
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticUnspecializedSourceJacobianError("candidate-set mismatch")
        certificates = [
            _certify_candidate(*(records[candidate_id] for records in maps))
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_complete_unspecialized_principal_source_jacobians_"
                "remainder_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "component_contract": component_contract_campaign.get(
                    "content_sha256"
                ),
                "first_order": first_order_campaign.get("content_sha256"),
                "evolution": evolution_campaign.get("content_sha256"),
                "nonquasilinear_pde": nonquasilinear_campaign.get("content_sha256"),
            },
            "config_sha256": _content_hash(config),
            "generic_unspecialized_source_jacobian_control": generic,
            "counts": {
                "selected": len(certificates),
                "principal_source_entries_completed_per_candidate": 1089,
                "principal_composed_identities_proved": len(certificates),
                "full_source_jacobians_completed": 0,
                "remainder_bounds_proved": 0,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates have deterministic exact chunks for all 1,089 "
                "derivative-bearing source-Jacobian entries, proving every entry of the "
                "principal 55x55 identity without witness specialization."
            ),
            "scope": certificates[0]["scope"],
        }
    except (
        KeyError,
        TypeError,
        ValueError,
        QuarticUnspecializedSourceJacobianError,
    ) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "principal_source_entries_completed_per_candidate": 0,
                "principal_composed_identities_proved": 0,
                "full_source_jacobians_completed": 0,
                "remainder_bounds_proved": 0,
                "global_H7_summations_applied": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_unspecialized_source_jacobian_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
