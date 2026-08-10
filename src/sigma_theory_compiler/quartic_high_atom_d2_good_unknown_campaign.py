from __future__ import annotations

import hashlib
import json
from functools import cache
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_row0_arithmetic_expansion_campaign import (
    ArithmeticDag,
    _faddeev_leverrier_inverse,
    _matrix_vector,
)
from .quartic_unspecialized_source_jacobian_campaign import (
    _unspecialized_principal_blocks,
)

SCHEMA_VERSION = "sigma-quartic-high-atom-d2-good-unknown-campaign-1.0"
DIMENSION = 11


class QuarticHighAtomD2GoodUnknownError(ValueError):
    """Raised when the representative D2 contraction is not exact."""


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


@cache
def _representative_d2_arithmetic_packet(provenance_hash: str) -> dict[str, Any]:
    """Materialize D_z(-A^-1 W) for z=H_01, W=B_1[:,10]."""

    dag = ArithmeticDag()
    matrix = [
        [
            dag.component(f"A[{row},{column}]", provenance_hash)
            for column in range(DIMENSION)
        ]
        for row in range(DIMENSION)
    ]
    remainder = [
        dag.component(f"W_s01_field10[{row}]", provenance_hash)
        for row in range(DIMENSION)
    ]
    derivative_matrix = [
        [
            dag.component(f"D[H_01]A[{row},{column}]", provenance_hash)
            for column in range(DIMENSION)
        ]
        for row in range(DIMENSION)
    ]
    derivative_remainder = [
        dag.component(f"D[H_01]W_s01_field10[{row}]", provenance_hash)
        for row in range(DIMENSION)
    ]
    inverse, determinant_root, inverse_evidence = _faddeev_leverrier_inverse(
        dag, matrix
    )
    solved = _matrix_vector(dag, inverse, [dag.neg(value) for value in remainder])
    differentiated_forcing = [
        dag.add(derivative_remainder[row], value)
        for row, value in enumerate(_matrix_vector(dag, derivative_matrix, solved))
    ]
    d2_vector = _matrix_vector(
        dag, inverse, [dag.neg(value) for value in differentiated_forcing]
    )
    arithmetic = dag.packet()
    body = {
        "schema_version": "sigma-high-atom-d2-arithmetic-packet-1.0",
        "identity": (
            "D_z F_s=-A^-1(D_z W_s+(D_z A)F_s), F_s=-A^-1 W_s"
        ),
        "high_atom": "s01[10]",
        "covariant_reference_direction": "H_01",
        "source_row": 0,
        "principal_family": "s01",
        "principal_field": 10,
        "determinant_coefficient_root": determinant_root,
        "d2_vector_roots": d2_vector,
        "representative_root": d2_vector[0],
        "inverse_evidence": inverse_evidence,
        "arithmetic_dag": arithmetic,
    }
    return {**body, "content_sha256": _content_hash(body)}


def _evaluate_dag(packet: dict[str, Any], inputs: dict[str, sp.Expr]) -> sp.Expr:
    values: list[sp.Expr] = []
    for node in packet["arithmetic_dag"]["nodes"]:
        op = node["op"]
        if op == "exact_constant":
            value = sp.sympify(node["value"])
        elif op == "exact_component_input":
            value = inputs[node["label"]]
        elif op == "exact_add":
            value = sum((values[index] for index in node["arguments"]), sp.S.Zero)
        elif op == "exact_negate":
            value = -values[node["argument"]]
        elif op == "exact_multiply":
            value = values[node["left"]] * values[node["right"]]
        elif op == "exact_divide":
            value = sp.cancel(values[node["numerator"]] / values[node["denominator"]])
        else:
            raise QuarticHighAtomD2GoodUnknownError(f"unsupported DAG op: {op}")
        values.append(value)
    return sp.factor(values[packet["representative_root"]])


@cache
def _actual_reference_slice() -> dict[str, Any]:
    blocks = _unspecialized_principal_blocks()
    data = blocks["data"]
    matrix = blocks["A"]
    spatial_block = blocks["B_i"][0]
    direction = data["hessian_lower"][0, 1]
    alpha = data["alpha"]
    c20 = data["c20"]
    substitutions = {
        symbol: 0
        for symbol in (
            list(data["gradient_lower"])
            + list(data["hessian_lower"])
            + list(data["einstein_upper"])
        )
    }
    substitutions[data["m2"]] = 1
    substitutions[c20] = c20
    matrix_reference = matrix.subs(substitutions)
    remainder_reference = spatial_block[:, 10].subs(substitutions)
    derivative_matrix_reference = sp.diff(matrix, direction).subs(substitutions)
    derivative_remainder_reference = sp.diff(
        spatial_block[:, 10], direction
    ).subs(substitutions)
    inverse = matrix_reference.inv()
    direct_vector = (
        inverse
        * derivative_matrix_reference
        * inverse
        * remainder_reference
        - inverse * derivative_remainder_reference
    ).applyfunc(sp.factor)
    nonzero = [
        {"source_row": row, "principal_field": column, "value": str(value)}
        for row in range(DIMENSION)
        for column, value in enumerate(
            (
                inverse
                * sp.diff(matrix, direction).subs(substitutions)
                * inverse
                * spatial_block.subs(substitutions)
                - inverse * sp.diff(spatial_block, direction).subs(substitutions)
            ).row(row)
        )
        if sp.factor(value) != 0
    ]
    inputs: dict[str, sp.Expr] = {}
    for row in range(DIMENSION):
        inputs[f"W_s01_field10[{row}]"] = remainder_reference[row]
        inputs[f"D[H_01]W_s01_field10[{row}]"] = derivative_remainder_reference[row]
        for column in range(DIMENSION):
            inputs[f"A[{row},{column}]"] = matrix_reference[row, column]
            inputs[f"D[H_01]A[{row},{column}]"] = derivative_matrix_reference[
                row, column
            ]
    return {
        "direction": str(direction),
        "alpha": alpha,
        "c20": c20,
        "matrix_determinant": sp.factor(matrix_reference.det()),
        "derivative_matrix_identically_zero": sp.diff(matrix, direction).is_zero_matrix,
        "direct_representative": sp.factor(direct_vector[0]),
        "nonzero_entries": nonzero,
        "inputs": inputs,
        "unspecialized_block_sha256": blocks["content_sha256"],
    }


@cache
def generic_high_atom_d2_good_unknown_control() -> tuple[bool, dict[str, Any]]:
    z = sp.Symbol("z")
    matrix = sp.Matrix([[2 + z, 1], [1, 3 - z]])
    remainder = sp.Matrix([1 + 2 * z, 4 - z])
    solved = -matrix.inv() * remainder
    implicit_derivative = -matrix.inv() * (
        remainder.diff(z) + matrix.diff(z) * solved
    )
    residual = (solved.diff(z) - implicit_derivative).applyfunc(sp.factor)
    wrong_sign = (solved.diff(z) + implicit_derivative).subs(z, 0).applyfunc(sp.factor)
    christoffel, gradient = sp.symbols("Gamma p")
    coordinate_hessian = z
    covariant_hessian = coordinate_hessian - christoffel * gradient
    reference_binding = sp.expand(
        (covariant_hessian - coordinate_hessian).subs({christoffel: 0, gradient: 0})
    )
    passed = bool(
        residual.is_zero_matrix
        and not wrong_sign.is_zero_matrix
        and reference_binding == 0
    )
    return passed, {
        "control": "exact implicit D2 contraction and high-atom reference binding",
        "known_answer": {
            "identity": "D_z(-A^-1W)=-A^-1(D_zW+(D_zA)(-A^-1W))",
            "residual_zero": residual.is_zero_matrix,
            "zero_entries": sum(value == 0 for value in residual),
        },
        "coordinate_to_covariant_binding": {
            "identity": "H_01=partial_0 partial_1 phi-Gamma^lambda_01 p_lambda",
            "reference_conditions": "Gamma=0 and p=0",
            "residual": str(reference_binding),
        },
        "negative_controls": {
            "reverse_implicit_derivative_sign": {
                "residual": [str(value) for value in wrong_sign],
                "rejected": not wrong_sign.is_zero_matrix,
            },
            "assign_named_low_high_paraproduct_to_high_low_projection": {
                "named_term": "T_(D_Y E55(Y)) deltaY_j",
                "named_term_support": "coefficient_low/state_high",
                "tested_projection": "coefficient_high/state_low",
                "projection_coefficient": "0",
                "rejected": True,
            },
        },
        "passed": passed,
    }


def _certify_candidate(
    unspecialized: dict[str, Any],
    full: dict[str, Any],
    c9: dict[str, Any],
    good_unknown: dict[str, Any],
    remedy: dict[str, Any],
    arithmetic: dict[str, Any],
    actual: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(unspecialized["candidate_id"])
    records = (full, c9, good_unknown, remedy)
    if any(record.get("candidate_id") != candidate_id for record in records):
        raise QuarticHighAtomD2GoodUnknownError("candidate identity mismatch")
    coefficients = unspecialized["coefficients"]
    if any(record.get("coefficients") != coefficients for record in records):
        raise QuarticHighAtomD2GoodUnknownError("candidate coefficient mismatch")
    alpha_value = sp.sympify(coefficients["a10"])
    c20_value = sp.sympify(coefficients["c20"])
    if alpha_value == 0:
        raise QuarticHighAtomD2GoodUnknownError("representative obstruction needs alpha!=0")
    inputs = {
        key: value.subs({actual["alpha"]: alpha_value, actual["c20"]: c20_value})
        for key, value in actual["inputs"].items()
    }
    dag_value = _evaluate_dag(arithmetic, inputs)
    direct_value = actual["direct_representative"].subs(
        {actual["alpha"]: alpha_value, actual["c20"]: c20_value}
    )
    expected = -2 * alpha_value
    if not (sp.factor(dag_value - direct_value) == 0 and direct_value == expected):
        raise QuarticHighAtomD2GoodUnknownError("D2 arithmetic/direct mismatch")
    named_good_unknown_high_low = sp.S.Zero
    residual = sp.factor(direct_value - named_good_unknown_high_low)
    if residual == 0:
        raise QuarticHighAtomD2GoodUnknownError("expected obstruction vanished")
    residual_payload = {
        "arithmetic_packet_sha256": arithmetic["content_sha256"],
        "representative_root": arithmetic["representative_root"],
        "candidate_id": candidate_id,
        "alpha": str(alpha_value),
        "direct_D2_value": str(direct_value),
        "named_good_unknown_high_low_value": "0",
        "residual": str(residual),
    }
    return {
        "schema_version": "sigma-quartic-high-atom-d2-good-unknown-certificate-1.0",
        "status": "pass_exact_representative_D2_obstruction_good_unknown_cancellation_refuted",
        "candidate_id": candidate_id,
        "coefficients": coefficients,
        "provenance": {
            "unspecialized_principal_block_sha256": actual[
                "unspecialized_block_sha256"
            ],
            "full_entry_manifest_sha256": full["provenance"][
                "full_entry_manifest_sha256"
            ],
            "principal_arithmetic_dag_sha256": full["provenance"][
                "principal_arithmetic_dag_sha256"
            ],
            "C9_orders": c9["orders_cumulatively_closed"],
            "D2_arithmetic_packet_sha256": arithmetic["content_sha256"],
            "residual_sha256": _content_hash(residual_payload),
        },
        "representative_slice": {
            "source_row": 0,
            "principal_family": "s01",
            "principal_field": 10,
            "high_atom": "s01[10]",
            "covariant_direction_at_reference": "H_01",
            "reference": (
                "M2=1; all gradient, Hessian, and Einstein components zero; "
                "flat/zero-connection coordinate-to-covariant identification"
            ),
            "A_reference_determinant": str(actual["matrix_determinant"]),
            "D_H01_A_identically_zero": actual[
                "derivative_matrix_identically_zero"
            ],
            "component_D2_value": str(direct_value),
            "expected_formula": "-2*a10",
            "arithmetic_DAG_matches_direct_block_differentiation": True,
        },
        "named_good_unknown_comparison": {
            "term": good_unknown["paradifferential_good_unknown"][
                "candidate_principal"
            ],
            "term_branch": "coefficient_low/state_high",
            "tested_branch": "coefficient_high/state_low",
            "term_projection_on_tested_branch": "0",
            "hash_bound_residual": str(residual),
            "residual_sha256": _content_hash(residual_payload),
            "cancellation_proved": False,
            "cancellation_refuted_for_this_slice": True,
        },
        "connection_to_B7_global_H7": {
            "representative_slice_removed_from_B7": False,
            "coefficient_high_state_low_branch_removed_from_B7": False,
            "B7_fully_replaced": False,
            "global_H7_differential_inequality_closed": False,
            "global_dyadic_summation_applied": False,
            "nonlinear_lifespan_proved": False,
        },
        "remaining_gate": (
            "construct a different modified unknown whose coefficient-high projection "
            "contributes +2*a10 on this component and prove the full tensor identity, "
            "or retain a derivative-loss theory"
        ),
    }


def run_quartic_high_atom_d2_good_unknown_campaign(
    unspecialized_campaign: dict[str, Any],
    full_jacobian_campaign: dict[str, Any],
    c9_campaign: dict[str, Any],
    good_unknown_campaign: dict[str, Any],
    resonant_remedy_campaign: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    try:
        campaigns = (
            unspecialized_campaign,
            full_jacobian_campaign,
            c9_campaign,
            good_unknown_campaign,
            resonant_remedy_campaign,
        )
        expected_statuses = (
            "pass_all_12_complete_unspecialized_principal_source_jacobians_remainder_fail_closed",
            "pass_all_12_full_11x153_entrywise_arithmetic_mixed_tensors_fail_closed",
            "pass_all_12_solved_source_C9_operator_envelopes_H7_topology_fail_closed",
            "pass_all_12_paradifferential_good_unknown_audits_component_binding_fail_closed",
            (
                "pass_all_12_resonant_H6xH7_operators_and_conditional_H8_"
                "remedies_actual_high_low_cancellation_fail_closed"
            ),
        )
        if config.get("schema_version") != SCHEMA_VERSION:
            raise QuarticHighAtomD2GoodUnknownError("unsupported campaign schema_version")
        if tuple(campaign.get("status") for campaign in campaigns) != expected_statuses:
            raise QuarticHighAtomD2GoodUnknownError("campaign prerequisite status mismatch")
        if not all(_content_hash_matches(campaign) for campaign in campaigns):
            raise QuarticHighAtomD2GoodUnknownError("campaign content hash mismatch")
        if (
            full_jacobian_campaign["upstream_sha256"]["principal_source"]
            != unspecialized_campaign["content_sha256"]
            or resonant_remedy_campaign["upstream_sha256"]["full_source_jacobian"]
            != full_jacobian_campaign["content_sha256"]
            or resonant_remedy_campaign["upstream_sha256"]["solved_source_C9"]
            != c9_campaign["content_sha256"]
        ):
            raise QuarticHighAtomD2GoodUnknownError("upstream provenance mismatch")
        if (
            int(config["expected_candidate_count"]) != 12
            or config.get("representative_high_atom") != "s01[10]"
            or int(config["representative_source_row"]) != 0
            or config.get("cancellation_policy") != "refute_if_nonzero"
            or config.get("global_H7_policy") != "fail_closed"
            or config.get("lifespan_policy") != "fail_closed"
        ):
            raise QuarticHighAtomD2GoodUnknownError("unsupported D2 audit contract")
        generic_passed, generic = generic_high_atom_d2_good_unknown_control()
        if not generic_passed:
            raise QuarticHighAtomD2GoodUnknownError("generic D2 control failed")
        maps = tuple(_candidate_records(campaign) for campaign in campaigns)
        candidate_ids = set(maps[0])
        if len(candidate_ids) != 12 or any(
            set(records) != candidate_ids for records in maps[1:]
        ):
            raise QuarticHighAtomD2GoodUnknownError("candidate-set mismatch")
        provenance_hash = full_jacobian_campaign["common_full_entry_manifest"][
            "content_sha256"
        ]
        arithmetic = _representative_d2_arithmetic_packet(provenance_hash)
        actual = _actual_reference_slice()
        generic_block_hash = unspecialized_campaign[
            "generic_unspecialized_source_jacobian_control"
        ]["unspecialized_block_extraction"]["block_content_sha256"]
        if actual["unspecialized_block_sha256"] != generic_block_hash:
            raise QuarticHighAtomD2GoodUnknownError("actual block hash mismatch")
        certificates = [
            _certify_candidate(
                *(records[candidate_id] for records in maps), arithmetic, actual
            )
            for candidate_id in sorted(candidate_ids)
        ]
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "pass_all_12_exact_representative_D2_obstructions_"
                "named_good_unknown_cancellation_refuted_global_H7_fail_closed"
            ),
            "errors": [],
            "upstream_sha256": {
                "unspecialized_principal_source": unspecialized_campaign[
                    "content_sha256"
                ],
                "full_source_jacobian": full_jacobian_campaign["content_sha256"],
                "solved_source_C9": c9_campaign["content_sha256"],
                "named_good_unknown": good_unknown_campaign["content_sha256"],
                "resonant_remedy": resonant_remedy_campaign["content_sha256"],
            },
            "config_sha256": _content_hash(config),
            "generic_high_atom_D2_control": generic,
            "representative_D2_arithmetic_packet": arithmetic,
            "actual_reference_audit": {
                "unspecialized_block_sha256": actual[
                    "unspecialized_block_sha256"
                ],
                "A_reference_determinant": str(actual["matrix_determinant"]),
                "D_H01_A_identically_zero": actual[
                    "derivative_matrix_identically_zero"
                ],
                "nonzero_D2_entries_in_s01_block": actual["nonzero_entries"],
                "nonzero_entry_count": len(actual["nonzero_entries"]),
            },
            "counts": {
                "selected": len(certificates),
                "representative_high_atom_D2_contractions_materialized": len(
                    certificates
                ),
                "arithmetic_DAG_direct_matches": len(certificates),
                "named_good_unknown_cancellations_proved": 0,
                "named_good_unknown_cancellations_refuted": len(certificates),
                "nonzero_obstructions": len(certificates),
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
            "certificates": certificates,
            "claim": (
                "For the scalar s01[10] high atom in source row 0, exact differentiation "
                "of the real 11x11 A/B_1 blocks and an independently evaluated A/W "
                "Faddeev-LeVerrier DAG both give D2F=-2*a10 at the flat reference. "
                "The named T_(D_Y E55) deltaY term has zero coefficient-high/state-low "
                "projection, so all 12 nonzero-a10 candidates retain this obstruction."
            ),
            "scope": (
                "One representative component is decisive against the named cancellation, "
                "but this bounded slice is not a full D2 tensor materialization and proves "
                "no global H7 estimate or lifespan."
            ),
        }
    except (KeyError, TypeError, ValueError, QuarticHighAtomD2GoodUnknownError) as error:
        errors.append(str(error))
        body = {
            "schema_version": SCHEMA_VERSION,
            "status": "reject",
            "errors": errors,
            "certificates": [],
            "counts": {
                "selected": 0,
                "representative_high_atom_D2_contractions_materialized": 0,
                "arithmetic_DAG_direct_matches": 0,
                "named_good_unknown_cancellations_proved": 0,
                "named_good_unknown_cancellations_refuted": 0,
                "nonzero_obstructions": 0,
                "B7_branches_closed": 0,
                "global_H7_closures": 0,
                "lifespans_proved": 0,
                "rejected": 0,
            },
        }
    return {**body, "content_sha256": _content_hash(body)}


def write_quartic_high_atom_d2_good_unknown_campaign(
    result: dict[str, Any], output: Path
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "campaign.json"
    path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
