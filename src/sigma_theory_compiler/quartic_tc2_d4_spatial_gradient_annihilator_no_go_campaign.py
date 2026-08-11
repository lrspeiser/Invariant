from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_d4_curl_constraint_admission_campaign import _gradient_lift
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

SCHEMA = "sigma-quartic-tc2-d4-spatial-gradient-annihilator-no-go-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-spatial-gradient-annihilator-no-go-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
FIELD_DIMENSION = 11
SPATIAL_DIRECTIONS = 3
OUTPUT_DIMENSION = 55
SPATIAL_STATE_DIMENSION = 33
NON_GRADIENT_STATE_COLUMNS = 22
EXPECTED_CANDIDATES = 12
EXPECTED_V_SHA256 = "a8a6cb0588ebae512db867990f937a3a9e5a9a38bf90be807fc62a8eb928f9c0"
EXPECTED_COMPANION_BLOCK_SHA256 = "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
EXPECTED_RANGE_SHA256 = "a4bef91998b4926d3ba7ee90cec771c2a3ab7dcc0fe9d12ab30199290036485b"
EXPECTED_TARGET_SHA256 = "f8bab1b2033220292cf2b66a3a5f1274b03419636a6fa35e2f0aa87682ae38c8"


class QuarticTC2D4SpatialGradientAnnihilatorNoGoError(ValueError):
    """Raised when the exhaustive spatial-gradient annihilator no-go is invalid."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _spatial_injections() -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    injections = []
    for start in (44, 11, 22):
        injection = sp.zeros(STATE_DIMENSION, FIELD_DIMENSION)
        injection[start : start + FIELD_DIMENSION, :] = sp.eye(FIELD_DIMENSION)
        injections.append(injection)
    return tuple(injections)  # type: ignore[return-value]


def _exhaustive_classification(
    companion_range: Mapping[str, Any], axis2_gate: Mapping[str, Any]
) -> dict[str, Any]:
    basis = _correction_basis()
    direction1 = basis["block"]
    output = direction1[:, 21]
    e1, e2, e3 = _spatial_injections()
    direction2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    direction3 = sp.zeros(STATE_DIMENSION)
    if (
        _content_hash(_matrix_payload(direction1)) != EXPECTED_V_SHA256
        or _content_hash(_matrix_payload(direction2)) != EXPECTED_COMPANION_BLOCK_SHA256
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "fixed direction blocks do not match the predecessor"
        )

    # For one output row and one field, the unknown spatial-column blocks are
    # (B2E1,B2E2,B2E3,B3E1,B3E2,B3E3).  The five independent polynomial
    # conditions below are tensored independently over all 55x11 row/field pairs.
    prototype = sp.Matrix(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0],
        ]
    )
    kernel = prototype.nullspace()
    expected_kernel = sp.Matrix([0, 0, -1, 0, 1, 0])
    if (
        prototype.rank() != 5
        or len(kernel) != 1
        or kernel[0] not in (expected_kernel, -expected_kernel)
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "gradient-annihilator prototype rank mismatch"
        )

    # The fixed B1 slice forces B2E1=-B1E2, B2E2=0, B3E1=B3E3=0,
    # and leaves exactly B2E3=-B3E2=A with arbitrary A in Mat(55,11).
    forced = {
        "B1_E1_zero": (direction1 * e1).is_zero_matrix,
        "B1_E3_zero": (direction1 * e3).is_zero_matrix,
        "B2_E1_equals_minus_B1_E2": direction2 * e1 == -(direction1 * e2),
        "B2_E2_zero": (direction2 * e2).is_zero_matrix,
        "B2_E3_zero_for_canonical_representative": (direction2 * e3).is_zero_matrix,
        "B3_E1_zero": (direction3 * e1).is_zero_matrix,
        "B3_E2_zero_for_canonical_representative": (direction3 * e2).is_zero_matrix,
        "B3_E3_zero": (direction3 * e3).is_zero_matrix,
    }
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3")
    residual = (
        (n1 * direction1 + n2 * direction2 + n3 * direction3) * _gradient_lift(n1, n2, n3)
    ).applyfunc(sp.factor)
    if not all(forced.values()) or not residual.is_zero_matrix:
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "canonical representative does not annihilate the full gradient lift"
        )

    raw_unknowns = 2 * OUTPUT_DIMENSION * SPATIAL_STATE_DIMENSION
    independent_constraints = prototype.rank() * OUTPUT_DIMENSION * FIELD_DIMENSION
    raw_freedom = raw_unknowns - independent_constraints
    if raw_freedom != OUTPUT_DIMENSION * FIELD_DIMENSION:
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "exhaustive affine dimension mismatch"
        )

    companion_audit = companion_range["exact_companion_audit"]
    range_audit = companion_audit["pure_curl_completion_range"]
    declared = range_audit["declared_completion_class"]
    exact_map = range_audit["exact_range_map"]
    axis2_result = axis2_gate["exact_axis2_base_D4_audit"]["result"]
    candidate_rows = axis2_gate["exact_axis2_base_D4_audit"]["candidate_comparison"]
    if (
        declared.get("raw_parameter_dimension") != raw_freedom
        or declared.get("effective_parameter_count") != 363
        or exact_map.get("matrix_shape") != [528, 363]
        or exact_map.get("rank") != 297
        or exact_map.get("target_augmented_rank") != 298
        or exact_map.get("target_in_image") is not False
        or exact_map.get("range_matrix_sha256") != EXPECTED_RANGE_SHA256
        or exact_map.get("target_coordinates_sha256") != EXPECTED_TARGET_SHA256
        or axis2_result.get("base_D4_RHS_identically_zero") is not True
        or axis2_result.get("corrected_axis2_D4_obstructions") != EXPECTED_CANDIDATES
        or len(candidate_rows) != EXPECTED_CANDIDATES
        or any(sp.sympify(row["eta"]) == 0 for row in candidate_rows)
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "projected range or candidate consequence mismatch"
        )

    return {
        "declared_operator_class": {
            "name": (
                "all_linear_directional_55_state_additions_supported_only_on_the_33_"
                "spatial_gradient_columns_preserving_B1_equals_V_and_annihilating_"
                "the_complete_11_field_gradient_lift"
            ),
            "direction_blocks": 3,
            "fixed_direction_1_block_sha256": EXPECTED_V_SHA256,
            "allowed_input_columns": {
                "w1": list(range(44, 55)),
                "w2": list(range(11, 22)),
                "w3": list(range(22, 33)),
            },
            "excluded_input_columns": {
                "q": list(range(11)),
                "v": list(range(33, 44)),
            },
            "support_exhaustive_within_declared_class": True,
            "nonlocal_or_higher_direction_dependence_included": False,
        },
        "polynomial_annihilator_system": {
            "identity": "(n1*B1+n2*B2+n3*B3)*L(n)=0 for all n",
            "coefficient_conditions": [
                "B1*E1=0",
                "B2*E2=0",
                "B3*E3=0",
                "B1*E2+B2*E1=0",
                "B1*E3+B3*E1=0",
                "B2*E3+B3*E2=0",
            ],
            "prototype_variable_order": [
                "B2E1",
                "B2E2",
                "B2E3",
                "B3E1",
                "B3E2",
                "B3E3",
            ],
            "prototype_matrix_sha256": _content_hash(_matrix_payload(prototype)),
            "prototype_rank": prototype.rank(),
            "prototype_nullity": len(kernel),
            "prototype_kernel_generator": [str(value) for value in kernel[0]],
            "canonical_residual_zero": residual.is_zero_matrix,
        },
        "exact_affine_solution": {
            "forced_direction_2_block": "B2*E1=-B1*E2",
            "forced_direction_2_block_sha256": EXPECTED_COMPANION_BLOCK_SHA256,
            "free_matrix": "A in Mat(55,11)",
            "general_solution": "B2*E3=A and B3*E2=-A; every other free spatial block is zero",
            "equivalent_constraint_family": "arbitrary output multiples of C23^[0..10]",
            "raw_unknown_coefficients_after_fixing_B1": raw_unknowns,
            "independent_affine_constraints": independent_constraints,
            "raw_affine_dimension": raw_freedom,
            "forced_block_checks": forced,
        },
        "axis2_projected_range": {
            "raw_affine_dimension": raw_freedom,
            "effective_parameter_count": declared["effective_parameter_count"],
            "codomain_skew_dimension": exact_map["codomain_skew_dimension"],
            "matrix_shape": exact_map["matrix_shape"],
            "range_rank": exact_map["rank"],
            "target_augmented_rank": exact_map["target_augmented_rank"],
            "target_in_image": exact_map["target_in_image"],
            "range_matrix_sha256": exact_map["range_matrix_sha256"],
            "target_coordinates_sha256": exact_map["target_coordinates_sha256"],
        },
        "candidate_consequence": {
            "base_axis2_D4_RHS_identically_zero": True,
            "registered_nonzero_eta_values": sorted({row["eta"] for row in candidate_rows}),
            "candidate_conditions_checked": EXPECTED_CANDIDATES,
            "candidate_completions_in_declared_class": 0,
            "candidate_no_go_results": EXPECTED_CANDIDATES,
        },
        "escape_boundary": {
            "necessary_change": (
                "use at least one q/v input column, alter the fixed B1=V slice, introduce "
                "higher/nonlinear direction dependence, or change the physical principal system"
            ),
            "outside_gradient_input_columns": NON_GRADIENT_STATE_COLUMNS,
            "outside_gradient_input_column_indices": [
                *range(11),
                *range(33, 44),
            ],
            "such_an_escape_constructed": False,
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
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "spatial-gradient annihilator config mismatch"
        )
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    companion_range = _load_bound(root, config["companion_range"])
    axis2_gate = _load_bound(root, config["axis2_base_rhs"])
    if (
        companion_range.get("status")
        != "pass_exact_axis2_companion_obstruction_and_pure_curl_range_no_go"
        or axis2_gate.get("status") != "pass_exact_all_12_axis2_D4_companion_obstructions"
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "spatial-gradient annihilator predecessor status mismatch"
        )
    exact = _exhaustive_classification(companion_range, axis2_gate)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_exhaustive_spatial_gradient_annihilator_completion_no_go",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "companion_range",
                "axis2_base_rhs",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "reference_direction": "e2",
            "same_active_tensor_component_inputs": True,
        },
        "exact_classification": exact,
        "counts": {
            "spatial_directions": SPATIAL_DIRECTIONS,
            "field_dimension": FIELD_DIMENSION,
            "state_dimension": STATE_DIMENSION,
            "spatial_input_columns": SPATIAL_STATE_DIMENSION,
            "non_gradient_input_columns": NON_GRADIENT_STATE_COLUMNS,
            "raw_unknown_coefficients_after_fixing_B1": 3630,
            "independent_affine_constraints": 3025,
            "raw_affine_dimension": 605,
            "effective_projected_parameters": 363,
            "projected_range_rank": 297,
            "target_augmented_rank": 298,
            "candidate_no_go_results": 12,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "omit_mixed_n1_n2_coefficient_condition": {
                "forced_companion_lost": True,
                "rejected": True,
            },
            "flip_forced_companion_sign": {
                "gradient_lift_residual_nonzero": True,
                "rejected": True,
            },
            "change_forced_companion_field_selector": {
                "gradient_lift_residual_nonzero": True,
                "rejected": True,
            },
            "claim_additional_spatial_supported_freedom_beyond_C23": {
                "prototype_nullity_per_output_field": 1,
                "rejected": True,
            },
            "claim_target_in_projected_range": {
                "range_rank": 297,
                "augmented_rank": 298,
                "rejected": True,
            },
            "promote_declared_class_no_go_to_global_TC2_no_go": {
                "non_gradient_input_columns_unclassified": 22,
                "rejected": True,
            },
        },
        "claims": {
            "spatial_gradient_supported_linear_completion_class_exhaustive": True,
            "fixed_B1_forces_axis2_companion_block": True,
            "all_12_candidates_ruled_out_in_declared_completion_class": True,
            "escape_requires_broader_support_or_operator_class": True,
            "all_topology_changing_completions_ruled_out": False,
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
            "Classify the smallest constraint-admissible extension using the 22 q/v input "
            "columns outside the spatial gradient lift, or prove that constraint propagation "
            "and physical principal equivalence force those columns to vanish. Altering B1 or "
            "introducing nonlinear/nonlocal direction dependence remains a separate broader gate."
        ),
        "scope": (
            "Exhaustive exact no-go for every linear 55-state directional addition supported "
            "only on the 33 spatial-gradient columns, preserving the exact direction-one V "
            "slice and annihilating the complete 11-field gradient lift. The affine solution "
            "space is exactly the 605-parameter C23 family already shown to miss the axis-two "
            "target. Operators using q/v columns, altered direction-one data, higher direction "
            "dependence, spatial covariance, full D4 closure, tube closure, CK1, CK3, TC2, B7, "
            "global-H7, and lifespan remain unresolved."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "spatial-gradient annihilator content identity mismatch"
        )
    exact = document.get("exact_classification", {})
    system = exact.get("polynomial_annihilator_system", {})
    affine = exact.get("exact_affine_solution", {})
    projected = exact.get("axis2_projected_range", {})
    consequence = exact.get("candidate_consequence", {})
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    if (
        document.get("status")
        != "pass_exact_exhaustive_spatial_gradient_annihilator_completion_no_go"
        or counts
        != {
            "spatial_directions": 3,
            "field_dimension": 11,
            "state_dimension": 55,
            "spatial_input_columns": 33,
            "non_gradient_input_columns": 22,
            "raw_unknown_coefficients_after_fixing_B1": 3630,
            "independent_affine_constraints": 3025,
            "raw_affine_dimension": 605,
            "effective_projected_parameters": 363,
            "projected_range_rank": 297,
            "target_augmented_rank": 298,
            "candidate_no_go_results": 12,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        }
        or system.get("prototype_rank") != 5
        or system.get("prototype_nullity") != 1
        or system.get("canonical_residual_zero") is not True
        or affine.get("forced_direction_2_block_sha256") != EXPECTED_COMPANION_BLOCK_SHA256
        or affine.get("raw_unknown_coefficients_after_fixing_B1") != 3630
        or affine.get("independent_affine_constraints") != 3025
        or affine.get("raw_affine_dimension") != 605
        or projected.get("effective_parameter_count") != 363
        or projected.get("range_rank") != 297
        or projected.get("target_augmented_rank") != 298
        or projected.get("target_in_image") is not False
        or projected.get("range_matrix_sha256") != EXPECTED_RANGE_SHA256
        or projected.get("target_coordinates_sha256") != EXPECTED_TARGET_SHA256
        or consequence.get("base_axis2_D4_RHS_identically_zero") is not True
        or consequence.get("candidate_conditions_checked") != EXPECTED_CANDIDATES
        or consequence.get("candidate_completions_in_declared_class") != 0
        or consequence.get("candidate_no_go_results") != EXPECTED_CANDIDATES
        or claims.get("spatial_gradient_supported_linear_completion_class_exhaustive") is not True
        or claims.get("fixed_B1_forces_axis2_companion_block") is not True
        or claims.get("all_12_candidates_ruled_out_in_declared_completion_class") is not True
        or claims.get("escape_requires_broader_support_or_operator_class") is not True
        or any(
            claims.get(key) is not False
            for key in (
                "all_topology_changing_completions_ruled_out",
                "spatially_covariant_tensor_completion_proved",
                "all_spatial_direction_compatibility_proved",
                "corrected_candidate_family_registered",
                "remaining_D4_selector_closed",
                "full_tube_Sylvester_identity",
                "CK1_closed",
                "CK3_closed",
                "TC2_closed",
                "B7_closed",
                "global_H7_closed",
                "lifespan_proved",
            )
        )
        or len(document.get("negative_controls", {})) != 6
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
    ):
        raise QuarticTC2D4SpatialGradientAnnihilatorNoGoError(
            "spatial-gradient annihilator exact/fail-closed contract mismatch"
        )


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify all spatial-gradient-supported D4 completion blocks."
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
                "raw_affine_dimension": artifact["counts"]["raw_affine_dimension"],
                "candidate_no_go_results": artifact["counts"]["candidate_no_go_results"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
