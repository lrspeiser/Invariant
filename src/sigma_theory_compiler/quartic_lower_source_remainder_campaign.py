from __future__ import annotations

import hashlib
import json
from functools import cache
from math import comb, factorial
from pathlib import Path
from typing import Any

import sympy as sp

SCHEMA_VERSION = "sigma-quartic-lower-source-remainder-campaign-1.0"


class QuarticLowerSourceRemainderError(ValueError):
    """Raised when lower-source or component-remainder provenance is invalid."""


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


def _lower_atom_labels() -> list[str]:
    return [
        *[f"q[{field}]" for field in range(10)],
        *[
            f"p{derivative}[{field}]"
            for derivative in range(4)
            for field in range(11)
        ],
    ]


def _missing_lower_column_map() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for column, label in enumerate(_lower_atom_labels()):
        family = "metric_value" if label.startswith("q") else "first_partial"
        records.append(
            {
                "column": column,
                "atom": label,
                "family": family,
                "dynamic_rows": 11,
                "missing_exact_entries": 11,
                "required_identity": (
                    f"D_{label}F=-Inverse(A)*(D_{label}W+(D_{label}A)*F)"
                ),
                "status": "missing_universal_unspecialized_A_W_component_packet",
                "reason": (
                    "the exact gauge-fixed Euler generator is published only at the "
                    "rational local witness; the tube artifact publishes a norm "
                    "majorant, not the unspecialized component values of A and W"
                ),
            }
        )
    return records


@cache
def generic_component_tensor_remainder_control() -> tuple[bool, dict[str, Any]]:
    """Check exact order-2--4 tensor contractions and a norm-only negative."""

    x, y, hx, hy, t = sp.symbols("x y hx hy t", real=True, finite=True)
    variables = (x, y)
    direction = (hx, hy)
    source = sp.Matrix(
        [
            x**2 + 2 * x * y + 3 * y**3 + 5 * x**4,
            7 * x**2 * y + 11 * x * y**2 + 13 * y**4,
        ]
    )
    shifted = source.subs({x: x + t * hx, y: y + t * hy})
    linearized = source + t * source.jacobian(variables) * sp.Matrix(direction)
    exact_remainder = (shifted - linearized).applyfunc(sp.expand)

    tensors: dict[str, list[list[str]]] = {}
    contractions: dict[int, sp.Matrix] = {}
    tensor_entry_count = 0
    for order in range(2, 5):
        entries_by_row: list[list[str]] = []
        contractions_by_row: list[sp.Expr] = []
        for row in range(2):
            entries: list[str] = []
            contraction = sp.Integer(0)
            for flat_index in range(2**order):
                indices = tuple(
                    (flat_index // (2**place)) % 2 for place in range(order)
                )
                derivative = source[row]
                for index in indices:
                    derivative = sp.diff(derivative, variables[index])
                derivative = sp.expand(derivative)
                entries.append(str(derivative))
                contraction += derivative * sp.prod(direction[index] for index in indices)
            entries_by_row.append(entries)
            contractions_by_row.append(sp.expand(contraction))
        tensors[str(order)] = entries_by_row
        contractions[order] = sp.Matrix(contractions_by_row)
        tensor_entry_count += 2 * 2**order

    reconstructed = sum(
        (t**order * contractions[order] / factorial(order) for order in range(2, 5)),
        sp.zeros(2, 1),
    )
    residual = (exact_remainder - reconstructed).applyfunc(sp.expand)

    corrupted = reconstructed.copy()
    corrupted[0] -= t**2 * 2 * hx * hy / 2
    corrupted_residual = (exact_remainder - corrupted).applyfunc(sp.expand)

    left = sp.Matrix([x**2, y**2])
    right = sp.Matrix([y**2, x**2])
    left_hessian = [sp.hessian(item, variables) for item in left]
    right_hessian = [sp.hessian(item, variables) for item in right]
    same_entrywise_absolute_multiset = sorted(
        abs(int(entry)) for matrix in left_hessian for entry in matrix
    ) == sorted(abs(int(entry)) for matrix in right_hessian for entry in matrix)
    probe_difference = sp.Matrix(
        [
            (sp.Matrix(direction).T * matrix * sp.Matrix(direction))[0]
            for matrix in left_hessian
        ]
    ) - sp.Matrix(
        [
            (sp.Matrix(direction).T * matrix * sp.Matrix(direction))[0]
            for matrix in right_hessian
        ]
    )

    passed = bool(
        residual.is_zero_matrix
        and not corrupted_residual.is_zero_matrix
        and same_entrywise_absolute_multiset
        and not probe_difference.is_zero_matrix
        and tensor_entry_count == 56
    )
    return passed, {
        "control": "exact component Fréchet tensors through order four",
        "source_shape": [2, 2],
        "orders": [2, 3, 4],
        "tensor_shapes": {str(order): [2, *([2] * order)] for order in range(2, 5)},
        "exact_component_tensors": tensors,
        "exact_tensor_entry_count": tensor_entry_count,
        "taylor_remainder_identity": (
            "F(z+t*h)-F(z)-t*DF(z)h="
            "sum_{k=2}^4 t^k/k! D^kF(z)[h^k]"
        ),
        "identity_residual_zero": residual.is_zero_matrix,
        "component_majorant_rule": {
            "definition": "M_k=max_row sum_{j1,...,jk}|D^kF_row,j1,...,jk|",
            "bound": "||D^kF[h1,...,hk]||_infinity <= M_k product_l ||hl||_infinity",
            "quadratic_remainder": "||R_2(z,h)||_infinity <= M_2 ||h||_infinity^2/2",
        },
        "negative_controls": {
            "omit_one_mixed_Hessian_permutation": {
                "nonzero_residual": not corrupted_residual.is_zero_matrix,
                "rejected": not corrupted_residual.is_zero_matrix,
            },
            "norm_orientation_inference": {
                "same_absolute_component_multiset": same_entrywise_absolute_multiset,
                "different_directional_contraction": not probe_difference.is_zero_matrix,
                "difference": str(probe_difference),
                "rejected": (
                    same_entrywise_absolute_multiset
                    and not probe_difference.is_zero_matrix
                ),
            },
        },
        "passed": passed,
        "scope": (
            "This is an executable exact component-tensor calculus. It cannot be "
            "applied to a candidate whose order-2--4 component tensors are absent."
        ),
    }


def _certify_candidate(
    unspecialized: dict[str, Any],
    component_contract: dict[str, Any],
    nonlinear: dict[str, Any],
    solved_source: dict[str, Any],
    generic: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(unspecialized.get("candidate_id"))
    others = (component_contract, nonlinear, solved_source)
    if any(
        item.get("candidate_id") != candidate_id
        or item.get("coefficients") != unspecialized.get("coefficients")
        for item in others
    ):
        raise QuarticLowerSourceRemainderError("candidate identity mismatch")
    expected_statuses = (
        "pass_complete_principal_source_jacobian_partial_full_tensor",
        "audit_component_jacobian_packet_missing_fail_closed",
        "pass_exact_local_nonlinear_time_acceleration_elimination",
        "pass_coordinate_atom_C4_solved_source_moser_envelopes",
    )
    if tuple(
        item.get("status")
        for item in (unspecialized, component_contract, nonlinear, solved_source)
    ) != expected_statuses:
        raise QuarticLowerSourceRemainderError("candidate prerequisite status mismatch")

    basis = unspecialized["basis_and_injection_provenance"]
    required = component_contract["component_packet_validation"]["required_schema"]
    if (
        basis["state_basis_sha256"] != required["state_basis_sha256"]
        or basis["coordinate_atom_basis_sha256"]
        != required["coordinate_atom_basis_sha256"]
        or basis["principal_jet_injection_sha256"]
        != required["principal_jet_injection_sha256"]
    ):
        raise QuarticLowerSourceRemainderError("basis or injection provenance mismatch")
    if component_contract["generator_provenance"][
        "evolution_formula_contract_sha256"
    ] != nonlinear["evolution_formula_contract_sha256"]:
        raise QuarticLowerSourceRemainderError("source formula provenance mismatch")

    missing_map = _missing_lower_column_map()
    map_packet = {
        "schema_version": "sigma-lower-source-column-map-1.0",
        "coordinate_atom_basis_sha256": basis["coordinate_atom_basis_sha256"],
        "column_range": [0, 53],
        "columns": missing_map,
        "column_count": len(missing_map),
        "missing_entry_count": sum(item["missing_exact_entries"] for item in missing_map),
    }
    map_packet = {**map_packet, "content_sha256": _content_hash(map_packet)}

    principal = unspecialized["source_jacobian_chunk_packet"]
    frechet = solved_source["solved_source_Frechet_derivatives"]
    return {
        "schema_version": "sigma-quartic-lower-source-remainder-certificate-1.0",
        "status": "audit_lower_source_and_component_remainder_fail_closed",
        "candidate_id": candidate_id,
        "coefficients": unspecialized["coefficients"],
        "provenance": {
            **basis,
            "evolution_formula_contract_sha256": nonlinear[
                "evolution_formula_contract_sha256"
            ],
            "source_geometric_formula_contract_sha256": nonlinear[
                "source_geometric_formula_contract_sha256"
            ],
            "principal_chunk_packet_sha256": principal["content_sha256"],
        },
        "derivable_exact_component_families": {
            "principal_second_atoms": {
                "column_count": 99,
                "entry_count": principal["completed_entries"],
                "chunk_count": len(principal["chunks"]),
                "chunk_hashes": [item["content_sha256"] for item in principal["chunks"]],
                "status": "complete_exact_operational_components",
            },
            "lower_atoms": {
                "column_count": 54,
                "entry_count": 0,
                "status": "unresolved_unspecialized_source_components",
            },
        },
        "lower_source_column_map": map_packet,
        "source_jacobian_completion": {
            "shape": [11, 153],
            "exact_entries_completed": principal["completed_entries"],
            "exact_entries_missing": map_packet["missing_entry_count"],
            "exact_entries_total": 1683,
            "full_component_tensor_complete": False,
        },
        "component_Frechet_tensor_gate": {
            "input_dimension": 153,
            "output_dimension": 11,
            "required_orders": [2, 3, 4],
            "dense_entry_counts": {
                str(order): 11 * 153**order for order in range(2, 5)
            },
            "symmetric_compressed_entry_counts": {
                str(order): 11 * comb(153 + order - 1, order)
                for order in range(2, 5)
            },
            "accepted_encoding": "exact sparse entries or exact expression DAG with indices",
            "existing_orders": frechet["orders"][2:],
            "existing_data_kind": "operator_norm_envelopes_only",
            "actual_component_tensor_orders_available": [],
            "generic_exact_calculus_sha256": _content_hash(generic),
            "component_majorant_applied": False,
        },
        "paralinearization_remainder_bound_proved": False,
        "full_H7_commutator_closed": False,
        "H7_derivative_loss_resolved": False,
        "global_dyadic_summation_applied": False,
        "precise_blocker": (
            "run gauge_fixed_euler_from_state on a universal 153-atom symbolic/DAG "
            "state, split E=A*a+W, emit F=-A^-1W and its exact order-1--4 "
            "component tensors; norm envelopes cannot supply tensor orientation"
        ),
        "scope": (
            "The map is complete about what is missing: all 54 lower columns and 594 "
            "dynamic entries. The exact remainder calculus is executable, but no "
            "candidate remainder is claimed without candidate component tensors."
        ),
    }


def run_quartic_lower_source_remainder_campaign(
    unspecialized_campaign: dict[str, Any],
    component_contract_campaign: dict[str, Any],
    nonlinear_evolution_campaign: dict[str, Any],
    solved_source_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticLowerSourceRemainderError("unsupported campaign schema_version")
        campaigns = (
            unspecialized_campaign,
            component_contract_campaign,
            nonlinear_evolution_campaign,
            solved_source_campaign,
        )
        expected_statuses = (
            "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed",
            "pass_all_12_component_jacobian_schema_audits_packet_missing_fail_closed",
            "pass_all_12_exact_local_nonlinear_time_acceleration_eliminations",
            "pass_all_12_coordinate_atom_C4_solved_source_moser_envelopes",
        )
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticLowerSourceRemainderError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticLowerSourceRemainderError("campaign content hash mismatch")
        upstream = unspecialized_campaign.get("upstream_sha256", {})
        if upstream.get("component_contract") != component_contract_campaign.get(
            "content_sha256"
        ):
            raise QuarticLowerSourceRemainderError("component-contract provenance mismatch")
        contract_upstream = component_contract_campaign.get("upstream_sha256", {})
        if (
            contract_upstream.get("nonlinear_evolution")
            != nonlinear_evolution_campaign.get("content_sha256")
            or contract_upstream.get("solved_source")
            != solved_source_campaign.get("content_sha256")
        ):
            raise QuarticLowerSourceRemainderError("source provenance mismatch")
        if (
            int(config["state_dimension"]) != 55
            or int(config["coordinate_atom_dimension"]) != 153
            or int(config["dynamic_source_row_count"]) != 11
            or int(config["lower_atom_column_count"]) != 54
            or list(config["required_component_Frechet_orders"]) != [2, 3, 4]
        ):
            raise QuarticLowerSourceRemainderError("unsupported component tensor contract")
        if bool(config.get("declare_component_remainder_proved", False)):
            raise QuarticLowerSourceRemainderError(
                "component remainder cannot be declared from norm envelopes"
            )
        generic_passed, generic = generic_component_tensor_remainder_control()
        if not generic_passed:
            raise QuarticLowerSourceRemainderError("generic tensor remainder control failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        expected = int(config.get("expected_candidate_count", 12))
        if len(candidate_ids) != expected or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticLowerSourceRemainderError("candidate-set mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), generic
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "audit_all_12_lower_source_maps_component_remainder_fail_closed",
            "errors": [],
            "upstream_sha256": {
                "unspecialized_source_jacobian": unspecialized_campaign.get(
                    "content_sha256"
                ),
                "component_contract": component_contract_campaign.get("content_sha256"),
                "nonlinear_evolution": nonlinear_evolution_campaign.get("content_sha256"),
                "solved_source": solved_source_campaign.get("content_sha256"),
            },
            "config_sha256": _content_hash(config),
            "generic_component_tensor_remainder_control": generic,
            "counts": {
                "selected": len(certificates),
                "lower_columns_mapped_per_candidate": 54,
                "lower_entries_derived_per_candidate": 0,
                "lower_entries_missing_per_candidate": 594,
                "candidate_component_remainders_proved": 0,
                "H7_derivative_losses_resolved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "All 12 candidates now have an exact column-by-column missing map and "
                "a component-tensor remainder contract; no norm envelope is promoted "
                "to a component or H7 proof."
            ),
            "scope": certificates[0]["scope"],
        }
    except (KeyError, TypeError, ValueError, QuarticLowerSourceRemainderError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "lower_columns_mapped_per_candidate": 0,
                "lower_entries_derived_per_candidate": 0,
                "lower_entries_missing_per_candidate": 0,
                "candidate_component_remainders_proved": 0,
                "H7_derivative_losses_resolved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_lower_source_remainder_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
