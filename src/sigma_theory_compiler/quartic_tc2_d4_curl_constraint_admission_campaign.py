from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_diagonal_third_jet_campaign import _content_hash, _matrix_payload
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import STATE_DIMENSION

SCHEMA = "sigma-quartic-tc2-d4-curl-constraint-admission-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-curl-constraint-admission-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_W_SHA256 = "e44c769b1eaf44c6e0ffc411007d98f9de24c6e8a20bac112d9a0a062e913500"


class QuarticTC2D4CurlConstraintAdmissionError(ValueError):
    """Raised when the curl-constraint admission certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _sparse_matrix(matrix: sp.MatrixBase) -> list[dict[str, Any]]:
    return [
        {"row": row, "column": column, "value": str(sp.factor(matrix[row, column]))}
        for row in range(matrix.rows)
        for column in range(matrix.cols)
        if matrix[row, column] != 0
    ]


def _gradient_lift(
    n1: sp.Symbol, n2: sp.Symbol, n3: sp.Symbol
) -> sp.Matrix:
    lift = sp.zeros(STATE_DIMENSION, 11)
    # Reordered state: z=(q,w2,w3), y=(v,w1).
    lift[11:22, :] = n2 * sp.eye(11)
    lift[22:33, :] = n3 * sp.eye(11)
    lift[44:55, :] = n1 * sp.eye(11)
    return lift


def _constraint_propagation_maps(output: sp.Matrix) -> dict[str, Any]:
    spatial_output = sp.zeros(3, 11)
    spatial_output[0, :] = output[44:55, :].T
    spatial_output[1, :] = output[11:22, :].T
    spatial_output[2, :] = output[22:33, :].T
    reconstructed = sp.zeros(STATE_DIMENSION, 1)
    reconstructed[44:55, :] = spatial_output[0, :].T
    reconstructed[11:22, :] = spatial_output[1, :].T
    reconstructed[22:33, :] = spatial_output[2, :].T
    if reconstructed != output:
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "escape output is not confined to spatial-derivative evolution rows"
        )

    definition_map = sp.zeros(33, 1)
    for spatial in range(3):
        for field in range(11):
            definition_map[11 * spatial + field, 0] = spatial_output[spatial, field]

    # Columns are partial_1 F, partial_2 F, partial_3 F for
    # F=eta(Y) C_12^[10]. Rows are C_12, C_13, C_23 for each field.
    curl_map = sp.zeros(33, 3)
    for field in range(11):
        curl_map[field, 0] = spatial_output[1, field]
        curl_map[field, 1] = -spatial_output[0, field]
        curl_map[11 + field, 0] = spatial_output[2, field]
        curl_map[11 + field, 2] = -spatial_output[0, field]
        curl_map[22 + field, 1] = spatial_output[2, field]
        curl_map[22 + field, 2] = -spatial_output[1, field]

    if (
        sum(value != 0 for value in spatial_output) != 6
        or spatial_output.rank() != 3
        or definition_map.rank() != 1
        or sum(value != 0 for value in definition_map) != 6
        or curl_map.rank() != 3
        or sum(value != 0 for value in curl_map) != 12
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "constraint propagation map rank/count mismatch"
        )
    return {
        "spatial_output_coefficients": {
            "row_order": ["w1", "w2", "w3"],
            "field_count": 11,
            "rank": spatial_output.rank(),
            "nonzero_entries": 6,
            "sparse": _sparse_matrix(spatial_output),
            "sha256": _content_hash(_matrix_payload(spatial_output)),
        },
        "definition_constraint_propagation": {
            "constraint_count": 33,
            "formula": "delta(D_i^a)_t=U_i^a*F, F=eta(Y)*C_12^[10]",
            "map_rank": definition_map.rank(),
            "nonzero_entries": 6,
            "map_sha256": _content_hash(_matrix_payload(definition_map)),
            "homogeneous_in_constraints": True,
        },
        "curl_constraint_propagation": {
            "constraint_count": 33,
            "formula": (
                "delta(C_ij^a)_t=U_j^a*partial_i(F)-U_i^a*partial_j(F)"
            ),
            "source_derivative_count": 3,
            "map_rank": curl_map.rank(),
            "nonzero_entries": 12,
            "map_sha256": _content_hash(_matrix_payload(curl_map)),
            "homogeneous_in_constraints_and_constraint_derivatives": True,
            "variable_coefficient_product_rule_closed": (
                "partial_i(eta*C)=eta*partial_i(C)+partial_i(eta)*C"
            ),
        },
    }


def _quartic_coefficient_jet() -> dict[str, Any]:
    y0, y2, y3, y9 = sp.symbols("Y_0 Y_2 Y_3 Y_9")
    variables = (y0, y2, y3, y9)
    monomial = sp.prod(variables)
    zero = {variable: 0 for variable in variables}
    lower_values: list[sp.Expr] = []
    for order in range(4):
        for derivative_indices in itertools.product(range(4), repeat=order):
            derivative = monomial
            for index in derivative_indices:
                derivative = sp.diff(derivative, variables[index])
            lower_values.append(sp.factor(derivative.subs(zero)))
    fourth_values = []
    for derivative_indices in itertools.product(range(4), repeat=4):
        derivative = monomial
        for index in derivative_indices:
            derivative = sp.diff(derivative, variables[index])
        fourth_values.append(sp.factor(derivative.subs(zero)))
    canonical = sp.diff(monomial, y0, y2, y3, y9).subs(zero)
    if (
        any(value != 0 for value in lower_values)
        or canonical != 1
        or sum(value != 0 for value in fourth_values) != 24
        or any(value not in (0, 1) for value in fourth_values)
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "quartic coefficient jet normalization mismatch"
        )
    return {
        "coefficient_monomial": "Y_0*Y_2*Y_3*Y_9",
        "reference_value_zero": True,
        "ordered_lower_derivatives_checked": len(lower_values),
        "orders_0_through_3_zero": True,
        "ordered_fourth_derivatives_checked": len(fourth_values),
        "nonzero_ordered_fourth_derivatives": 24,
        "canonical_D_0_D_2_D_3_D_9_value": "1",
        "canonical_active_indices": list(ACTIVE_INDICES),
    }


def _exact_admission(minimal_escape: Mapping[str, Any]) -> dict[str, Any]:
    basis = _correction_basis()
    block = basis["block"]
    wedge = basis["wedge"]
    output = block[:, 21]
    e21 = sp.eye(STATE_DIMENSION)[:, 21]
    e54 = sp.eye(STATE_DIMENSION)[:, 54]
    direction_1 = (output * e21.T).applyfunc(sp.factor)
    direction_2 = (-output * e54.T).applyfunc(sp.factor)
    direction_3 = sp.zeros(STATE_DIMENSION)
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3")
    directional = (n1 * direction_1 + n2 * direction_2).applyfunc(sp.factor)
    gradient_lift = _gradient_lift(n1, n2, n3)
    physical_residual = (directional * gradient_lift).applyfunc(sp.factor)
    standalone_residual = (n1 * direction_1 * gradient_lift).applyfunc(sp.factor)
    wrong_sign_residual = (
        (n1 * direction_1 - n2 * direction_2) * gradient_lift
    ).applyfunc(sp.factor)
    field_mismatch = (-output * sp.eye(STATE_DIMENSION)[:, 53].T).applyfunc(
        sp.factor
    )
    field_mismatch_residual = (
        (n1 * direction_1 + n2 * field_mismatch) * gradient_lift
    ).applyfunc(sp.factor)
    if (
        direction_1 != block
        or direction_1.rank() != 1
        or direction_2.rank() != 1
        or sum(value != 0 for value in direction_1) != 6
        or sum(value != 0 for value in direction_2) != 6
        or not physical_residual.is_zero_matrix
        or standalone_residual.is_zero_matrix
        or wrong_sign_residual.is_zero_matrix
        or field_mismatch_residual.is_zero_matrix
        or _content_hash(_matrix_payload(direction_1)) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(wedge)) != EXPECTED_W_SHA256
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "curl companion realization mismatch"
        )

    constraint_maps = _constraint_propagation_maps(output)
    candidate_rows = minimal_escape.get("exact_escape", {}).get(
        "candidate_classification", []
    )
    admitted_candidates = [
        {
            "candidate_id": row["candidate_id"],
            "a10": row["a10"],
            "eta_unique_tuning": row["eta_unique_tuning"],
            "reference_direction_D4_Sylvester_solvable": row[
                "corrected_D4_Sylvester_solvable"
            ],
            "reference_direction_deltaK_sha256": row[
                "corrected_deltaK_sha256"
            ],
            "gauge_fixed_curl_constraint_realization": True,
            "covariant_action_origin": False,
            "all_spatial_directions_checked": False,
        }
        for row in candidate_rows
    ]
    if (
        len(admitted_candidates) != EXPECTED_CANDIDATES
        or len({row["candidate_id"] for row in admitted_candidates})
        != EXPECTED_CANDIDATES
        or any(
            row["reference_direction_D4_Sylvester_solvable"] is not True
            for row in admitted_candidates
        )
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "candidate-specific minimal escape binding mismatch"
        )
    return {
        "gauge_fixed_operator": {
            "name": "single_curl_constraint_completion_of_minimal_V",
            "ordered_state": "z=(q,w2,w3), y=(v,w1)",
            "source_constraint": "C_12^[10]=partial_1(w2[10])-partial_2(w1[10])",
            "output_vector": "u=K55(0)^(-1)*(e16+e28)",
            "operator": "eta(Y)*u*C_12^[10]",
            "directional_symbol": "eta(Y)*u*(n1*e21^T-n2*e54^T)",
            "direction_1_block_equals_V": True,
            "direction_1_block_sha256": EXPECTED_V_SHA256,
            "direction_1_block_rank": 1,
            "direction_1_block_nonzero_entries": 6,
            "direction_2_companion_sha256": _content_hash(
                _matrix_payload(direction_2)
            ),
            "direction_2_companion_rank": 1,
            "direction_2_companion_nonzero_entries": 6,
            "direction_3_block_sha256": _content_hash(
                _matrix_payload(direction_3)
            ),
            "direction_3_block_zero": True,
            "directional_symbol_sha256": _content_hash(
                _matrix_payload(directional)
            ),
            "one_source_curl_constraint": True,
            "minimal_direction_block_count": 2,
        },
        "physical_reduction_equivalence": {
            "gradient_lift": (
                "w1=n1*q, w2=n2*q, w3=n3*q for each of 11 second-order fields"
            ),
            "gradient_lift_sha256": _content_hash(_matrix_payload(gradient_lift)),
            "directional_operator_times_gradient_lift_zero": True,
            "residual_sha256": _content_hash(_matrix_payload(physical_residual)),
            "constraint_surface_operator_zero": True,
            "physical_second_order_solutions_unchanged": True,
        },
        "constraint_propagation": constraint_maps,
        "coefficient_jet": _quartic_coefficient_jet(),
        "reference_D4_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "energy_skew_W_sha256": EXPECTED_W_SHA256,
            "unique_tuning": "eta=-(34816/15)*alpha^5",
            "candidate_specializations": admitted_candidates,
            "candidate_count": len(admitted_candidates),
            "reference_direction_D4_solutions": len(admitted_candidates),
            "other_spatial_directions_checked": False,
            "other_equal_eigenspaces_in_direction_e1_inherited_from_minimal_escape": True,
        },
        "admission_result": {
            "gauge_fixed_constraint_operator_constructed": True,
            "canonical_definition_constraint_surface_invariant": True,
            "canonical_curl_constraint_surface_invariant": True,
            "variable_coefficient_constraint_surface_invariant": True,
            "reference_direction_minimal_escape_physically_equivalent": True,
            "covariant_action_derived": False,
            "spatially_covariant_tensor_completion_proved": False,
            "all_direction_Sylvester_compatibility_proved": False,
        },
        "negative_control_residuals": {
            "omit_direction_2_companion": {
                "residual_nonzero": True,
                "residual_sha256": _content_hash(
                    _matrix_payload(standalone_residual)
                ),
            },
            "wrong_companion_sign": {
                "residual_nonzero": True,
                "residual_sha256": _content_hash(
                    _matrix_payload(wrong_sign_residual)
                ),
            },
            "wrong_companion_field": {
                "used_selector": 53,
                "required_selector": 54,
                "residual_nonzero": True,
                "residual_sha256": _content_hash(
                    _matrix_payload(field_mismatch_residual)
                ),
            },
            "exact_polynomial_witness": {
                "q10": "x1*x2",
                "partial_1_w2_10": "1",
                "partial_2_w1_10": "1",
                "full_curl": "0",
                "standalone_direction_1_term": "1",
            },
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "curl-constraint admission config mismatch"
        )
    for key in ("campaign_source", "campaign_test", "first_order_reduction_source"):
        _check_raw_binding(root, config[key])
    minimal = _load_bound(root, config["minimal_escape"])
    topology = _load_bound(root, config["topology_classification"])
    first_order = _load_bound(root, config["first_order_reduction"])
    if (
        minimal.get("exact_escape", {}).get("correction_ansatz", {}).get("V_sha256")
        != EXPECTED_V_SHA256
        or topology.get("exact_classification", {})
        .get("explicit_TC2_selector_classification", {})
        .get("canonical_capable_indices")
        != [21, 44, 48, 51, 53]
        or topology.get("claims", {}).get(
            "explicit_constraint_row_covariant_origin_constructed"
        )
        is not False
        or first_order.get("counts", {}).get("exact_55_variable_reductions_passed")
        != EXPECTED_CANDIDATES
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "bound predecessor semantic contract mismatch"
        )
    exact = _exact_admission(minimal)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "topology_classification",
                "first_order_reduction",
                "first_order_reduction_source",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "input_selector_direction_1": 21,
            "required_companion_selector_direction_2": 54,
            "V_sha256": EXPECTED_V_SHA256,
            "W_sha256": EXPECTED_W_SHA256,
        },
        "exact_admission": exact,
        "counts": {
            "source_curl_constraints": 1,
            "direction_blocks": 2,
            "output_nonzero_coefficients": 6,
            "definition_constraints_propagated": 33,
            "curl_constraints_propagated": 33,
            "ordered_lower_coefficient_derivatives_checked": 85,
            "ordered_fourth_coefficient_derivatives_checked": 256,
            "candidate_reference_D4_solutions_inherited": 12,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "omit_direction_2_companion": {
                "gradient_subspace_residual_nonzero": True,
                "rejected": True,
            },
            "wrong_companion_sign": {
                "gradient_subspace_residual_nonzero": True,
                "rejected": True,
            },
            "wrong_companion_field": {
                "gradient_subspace_residual_nonzero": True,
                "rejected": True,
            },
            "infer_covariant_action_origin": {
                "constraint_addition_is_action_derived": False,
                "rejected": True,
            },
            "infer_all_direction_compatibility": {
                "other_directional_Sylvester_obligations_checked": False,
                "rejected": True,
            },
            "promote_reference_admission_to_global_closure": {
                "remaining_D4_selector_closed": False,
                "tube_theorem_proved": False,
                "rejected": True,
            },
        },
        "claims": {
            "minimal_V_gauge_fixed_curl_constraint_realized": True,
            "canonical_constraint_surface_invariance_proved": True,
            "candidate_reference_D4_admission_count": 12,
            "covariant_action_origin_constructed": False,
            "spatially_covariant_tensor_completion_proved": False,
            "all_spatial_direction_compatibility_proved": False,
            "corrected_candidate_family_registered": False,
            "remaining_D4_selector_closed": False,
            "full_tube_Sylvester_identity": False,
            "CK1_closed": False,
            "CK3_closed": False,
            "TC2_closed": False,
            "B7_closed": False,
            "global_H7_closed": False,
            "lifespan_proved": False,
        },
        "next_gate": (
            "Promote the fixed-chart curl completion to a spatially covariant gauge-fixed "
            "tensor operator and evaluate its companion-direction/all-eigenspace D4 effects. "
            "Only after those checks may corrected candidates be registered or the remaining "
            "fourth-jet selector be resumed."
        ),
        "scope": (
            "Exact gauge-fixed constraint admission of the minimal rank-one V at the reference "
            "direction. The two-direction curl completion annihilates the canonical gradient "
            "lift and gives a closed homogeneous definition/curl-constraint subsystem, including "
            "variable coefficient eta(Y). It is not derived from a covariant action; spatially "
            "covariant completion, other directions/eigenspaces, corrected registration, the "
            "remaining D4 selector, tube closure, CK1, CK3, TC2, B7, global-H7, and lifespan "
            "remain fail-closed."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "curl-constraint admission content identity mismatch"
        )
    exact = document.get("exact_admission", {})
    operator = exact.get("gauge_fixed_operator", {})
    equivalence = exact.get("physical_reduction_equivalence", {})
    propagation = exact.get("constraint_propagation", {})
    result = exact.get("admission_result", {})
    expected_claims = {
        "minimal_V_gauge_fixed_curl_constraint_realized": True,
        "canonical_constraint_surface_invariance_proved": True,
        "candidate_reference_D4_admission_count": 12,
        "covariant_action_origin_constructed": False,
        "spatially_covariant_tensor_completion_proved": False,
        "all_spatial_direction_compatibility_proved": False,
        "corrected_candidate_family_registered": False,
        "remaining_D4_selector_closed": False,
        "full_tube_Sylvester_identity": False,
        "CK1_closed": False,
        "CK3_closed": False,
        "TC2_closed": False,
        "B7_closed": False,
        "global_H7_closed": False,
        "lifespan_proved": False,
    }
    expected_counts = {
        "source_curl_constraints": 1,
        "direction_blocks": 2,
        "output_nonzero_coefficients": 6,
        "definition_constraints_propagated": 33,
        "curl_constraints_propagated": 33,
        "ordered_lower_coefficient_derivatives_checked": 85,
        "ordered_fourth_coefficient_derivatives_checked": 256,
        "candidate_reference_D4_solutions_inherited": 12,
        "negative_controls": 6,
        "inferred_global_passes": 0,
    }
    if (
        document.get("status")
        != "pass_exact_gauge_fixed_curl_constraint_admission_for_minimal_V"
        or document.get("claims") != expected_claims
        or document.get("counts") != expected_counts
        or operator.get("direction_1_block_equals_V") is not True
        or operator.get("direction_1_block_sha256") != EXPECTED_V_SHA256
        or operator.get("minimal_direction_block_count") != 2
        or equivalence.get("directional_operator_times_gradient_lift_zero") is not True
        or equivalence.get("physical_second_order_solutions_unchanged") is not True
        or propagation.get("definition_constraint_propagation", {}).get("map_rank")
        != 1
        or propagation.get("curl_constraint_propagation", {}).get("map_rank") != 3
        or result.get("gauge_fixed_constraint_operator_constructed") is not True
        or result.get("canonical_definition_constraint_surface_invariant") is not True
        or result.get("canonical_curl_constraint_surface_invariant") is not True
        or result.get("variable_coefficient_constraint_surface_invariant") is not True
        or result.get("covariant_action_derived") is not False
        or result.get("all_direction_Sylvester_compatibility_proved") is not False
        or exact.get("reference_D4_binding", {}).get("candidate_count")
        != EXPECTED_CANDIDATES
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4CurlConstraintAdmissionError(
            "curl-constraint admission exact/fail-closed contract mismatch"
        )


def run_campaign(
    project_root: Path, config_path: Path, output_path: Path
) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Admit the D4 minimal V as a gauge-fixed curl-constraint operator."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_campaign(args.project_root, args.config, args.output)
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "direction_blocks": artifact["counts"]["direction_blocks"],
                "candidate_admissions": artifact["counts"][
                    "candidate_reference_D4_solutions_inherited"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
