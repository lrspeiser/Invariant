from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from .quartic_tc2_d4_axis2_base_rhs_campaign import _axis2_reference
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

SCHEMA = "sigma-quartic-tc2-d4-full-linear-gradient-annihilator-no-go-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-full-linear-gradient-annihilator-no-go-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
FIELD_DIMENSION = 11
ZERO_EIGENSPACE_DIMENSION = 33
EXPECTED_CANDIDATES = 12
QV_INDICES = (*range(11), *range(33, 44))
C23_INDICES = tuple(range(22, 33))
COMBINED_FREE_B2_INDICES = (*range(11), *range(22, 44))
Q_INDICES = tuple(range(11))
V_INDICES = tuple(range(33, 44))
EXPECTED_COMPANION_BLOCK_SHA256 = "9ef0bdb7ea7009ebba9b25ccb1225e1b955351d62a4b16c7989d339508a3b195"
EXPECTED_COMPANION_COMPRESSION_SHA256 = (
    "def5dc985fa3356a9a21b2b06d4ebe0f0365058403e3e762eab161d7fb2822be"
)


class QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(ValueError):
    """Raised when the full linear gradient-annihilator classification is invalid."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _zero_speed_coordinates(
    projector: sp.Matrix, target: sp.Matrix
) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    _, pivot_columns = projector.T.rref()
    covector_basis = projector.T[:, list(pivot_columns)]
    _, pivot_rows = covector_basis.T.rref()
    selected_rows = list(pivot_rows)
    coordinate_map = (
        covector_basis[selected_rows, :].inv() * sp.eye(STATE_DIMENSION)[selected_rows, :]
    )
    target_coordinates = (coordinate_map * target * coordinate_map.T).applyfunc(sp.factor)
    if (
        covector_basis.rank() != ZERO_EIGENSPACE_DIMENSION
        or coordinate_map * covector_basis != sp.eye(ZERO_EIGENSPACE_DIMENSION)
        or covector_basis * target_coordinates * covector_basis.T != target
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "zero-speed coordinate reconstruction mismatch"
        )
    return covector_basis, coordinate_map, target_coordinates


def _subspace_range_classification(
    name: str,
    indices: tuple[int, ...],
    projector: sp.Matrix,
    coordinate_map: sp.Matrix,
    target_coordinates: sp.Matrix,
) -> dict[str, Any]:
    identity = sp.eye(STATE_DIMENSION)
    selector_coordinates = (coordinate_map * projector.T * identity[:, list(indices)]).applyfunc(
        sp.factor
    )
    selector_rank = selector_coordinates.rank()
    selector_basis = sp.Matrix.hstack(*selector_coordinates.columnspace())
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    joint_rank = selector_basis.row_join(target_plane).rank()
    intersection_dimension = selector_rank + target_plane.rank() - joint_rank
    quotient_vectors = selector_coordinates.T.nullspace()
    quotient = sp.Matrix.hstack(*quotient_vectors).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    quotient_dimension = ZERO_EIGENSPACE_DIMENSION - selector_rank
    range_rank = (
        ZERO_EIGENSPACE_DIMENSION * (ZERO_EIGENSPACE_DIMENSION - 1) // 2
        - quotient_dimension * (quotient_dimension - 1) // 2
    )
    target_in_image = quotient_target.is_zero_matrix
    target_augmented_rank = range_rank if target_in_image else range_rank + 1
    if (
        selector_basis.rank() != selector_rank
        or target_plane.rank() != 2
        or quotient.shape != (quotient_dimension, ZERO_EIGENSPACE_DIMENSION)
        or quotient * selector_coordinates != sp.zeros(quotient_dimension, len(indices))
        or quotient_target.rank() != 2
        or target_in_image
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            f"{name} selector-subspace quotient classification mismatch"
        )
    return {
        "name": name,
        "canonical_input_indices": list(indices),
        "canonical_input_count": len(indices),
        "selector_projection_rank": selector_rank,
        "selector_projection_kernel_dimension": len(indices) - selector_rank,
        "zero_speed_quotient_dimension": quotient_dimension,
        "target_plane_dimension": target_plane.rank(),
        "selector_target_plane_intersection_dimension": intersection_dimension,
        "selector_target_joint_rank": joint_rank,
        "raw_output_selector_parameters": STATE_DIMENSION * len(indices),
        "effective_projected_parameters": ZERO_EIGENSPACE_DIMENSION * selector_rank,
        "wedge_range_rank": range_rank,
        "target_augmented_rank": target_augmented_rank,
        "target_in_image": target_in_image,
        "selector_coordinates_sha256": _content_hash(_matrix_payload(selector_coordinates)),
        "quotient_map_sha256": _content_hash(_matrix_payload(quotient)),
        "quotient_target_rank": quotient_target.rank(),
        "quotient_target_nonzero_entries": sum(value != 0 for value in quotient_target),
        "quotient_target_sha256": _content_hash(_matrix_payload(quotient_target)),
    }


def _canonical_qv_partition(
    projector: sp.Matrix, coordinate_map: sp.Matrix, target_coordinates: sp.Matrix
) -> dict[str, Any]:
    identity = sp.eye(STATE_DIMENSION)
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    rows = []
    for index in QV_INDICES:
        projection = (coordinate_map * projector.T * identity[:, index]).applyfunc(sp.factor)
        projection_zero = projection.is_zero_matrix
        capable = (
            not projection_zero and target_plane.row_join(projection).rank() == target_plane.rank()
        )
        rows.append(
            {
                "selector_index": index,
                "state_block": "q" if index in Q_INDICES else "v",
                "projection_zero": projection_zero,
                "induced_rank_one_wedge_map_rank": 0 if projection_zero else 32,
                "target_in_rank_one_image": capable,
                "projected_selector_sha256": _content_hash(_matrix_payload(projection)),
            }
        )
    kernel = [row["selector_index"] for row in rows if row["projection_zero"]]
    capable = [row["selector_index"] for row in rows if row["target_in_rank_one_image"]]
    nonzero_incapable = [
        row["selector_index"]
        for row in rows
        if not row["projection_zero"] and not row["target_in_rank_one_image"]
    ]
    if kernel != list(V_INDICES) or capable or nonzero_incapable != list(Q_INDICES):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "canonical q/v selector partition mismatch"
        )
    return {
        "records": rows,
        "kernel_indices": kernel,
        "nonzero_incapable_indices": nonzero_incapable,
        "capable_indices": capable,
        "counts": {
            "selectors_checked": len(rows),
            "zero_projection_selectors": len(kernel),
            "nonzero_incapable_selectors": len(nonzero_incapable),
            "capable_selectors": len(capable),
        },
    }


def _exact_classification(
    spatial_no_go: Mapping[str, Any], axis2_gate: Mapping[str, Any]
) -> dict[str, Any]:
    reference = _axis2_reference()
    projector = reference["projectors"][sp.S.Zero]
    energy = reference["energy0"]
    identity = sp.eye(STATE_DIMENSION)
    output = _correction_basis()["block"][:, 21]
    companion_block = (-output * identity[:, 54].T).applyfunc(sp.factor)
    companion_skew = (energy * companion_block - companion_block.T * energy).applyfunc(sp.factor)
    target = (projector.T * companion_skew * projector).applyfunc(sp.factor)
    if (
        _content_hash(_matrix_payload(companion_block)) != EXPECTED_COMPANION_BLOCK_SHA256
        or target.rank() != 2
        or _content_hash(_matrix_payload(target)) != EXPECTED_COMPANION_COMPRESSION_SHA256
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "axis-two companion replay mismatch"
        )
    _, coordinate_map, target_coordinates = _zero_speed_coordinates(projector, target)
    qv = _subspace_range_classification(
        "all_q_and_v_input_columns",
        QV_INDICES,
        projector,
        coordinate_map,
        target_coordinates,
    )
    c23 = _subspace_range_classification(
        "all_C23_spatial_freedom_columns",
        C23_INDICES,
        projector,
        coordinate_map,
        target_coordinates,
    )
    combined = _subspace_range_classification(
        "complete_free_B2_input_subspace_under_fixed_B1_gradient_annihilation",
        COMBINED_FREE_B2_INDICES,
        projector,
        coordinate_map,
        target_coordinates,
    )
    spatial_counts = spatial_no_go["counts"]
    axis2_result = axis2_gate["exact_axis2_base_D4_audit"]["result"]
    if (
        qv["selector_projection_rank"] != 11
        or qv["wedge_range_rank"] != 297
        or qv["target_augmented_rank"] != 298
        or c23["selector_projection_rank"] != 11
        or c23["wedge_range_rank"] != 297
        or c23["target_augmented_rank"] != 298
        or combined["selector_projection_rank"] != 22
        or combined["wedge_range_rank"] != 473
        or combined["target_augmented_rank"] != 474
        or spatial_counts.get("raw_affine_dimension") != 605
        or axis2_result.get("base_D4_RHS_identically_zero") is not True
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "full free-selector range contract mismatch"
        )
    return {
        "declared_operator_class": {
            "name": (
                "all_linear_direction_homogeneous_55_state_blocks_B_of_n_equal_sum_ni_Bi_"
                "preserving_B1_equals_V_and_annihilating_the_complete_gradient_lift"
            ),
            "full_state_input_columns": 55,
            "fixed_direction_1_block": True,
            "gradient_annihilator_identity": "B(n)*L(n)=0 for all n",
            "nonlinear_or_nonlocal_direction_dependence_included": False,
            "physical_principal_equivalence_beyond_gradient_annihilation_proved": False,
        },
        "full_affine_dimension": {
            "raw_B2_B3_coefficients": 6050,
            "independent_spatial_gradient_constraints": 3025,
            "affine_dimension": 3025,
            "spatial_C23_freedom": 605,
            "qv_freedom_in_B2": 1210,
            "qv_freedom_in_B3": 1210,
            "axis2_relevant_free_B2_parameters": 1815,
        },
        "canonical_qv_selector_partition": _canonical_qv_partition(
            projector, coordinate_map, target_coordinates
        ),
        "qv_subspace_range": qv,
        "c23_subspace_range": c23,
        "combined_axis2_free_B2_range": combined,
        "target": {
            "companion_block_sha256": EXPECTED_COMPANION_BLOCK_SHA256,
            "compression_sha256": EXPECTED_COMPANION_COMPRESSION_SHA256,
            "compression_rank": target.rank(),
            "compression_nonzero_entries": sum(value != 0 for value in target),
            "target_coordinates_sha256": _content_hash(_matrix_payload(target_coordinates)),
        },
        "candidate_consequence": {
            "base_axis2_D4_RHS_identically_zero": True,
            "candidate_conditions_checked": EXPECTED_CANDIDATES,
            "candidate_linear_gradient_annihilator_completions": 0,
            "candidate_no_go_results": EXPECTED_CANDIDATES,
        },
        "escape_boundary": {
            "all_55_input_columns_classified_within_linear_gradient_annihilator_class": True,
            "remaining_escape_options": [
                "alter_the_fixed_direction_1_B1_equals_V_slice",
                "use_nonlinear_or_higher_degree_direction_dependence",
                "use_nonlocal_or_pseudodifferential_direction_dependence",
                "relax_gradient_lift_annihilation_and_rederive_physical_principal_equivalence",
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
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "full linear gradient-annihilator config mismatch"
        )
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    spatial_no_go = _load_bound(root, config["spatial_gradient_no_go"])
    axis2_gate = _load_bound(root, config["axis2_base_rhs"])
    if (
        spatial_no_go.get("status")
        != "pass_exact_exhaustive_spatial_gradient_annihilator_completion_no_go"
        or axis2_gate.get("status") != "pass_exact_all_12_axis2_D4_companion_obstructions"
    ):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "full linear gradient-annihilator predecessor status mismatch"
        )
    exact = _exact_classification(spatial_no_go, axis2_gate)
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_full_linear_gradient_annihilator_completion_no_go",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "spatial_gradient_no_go",
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
            "canonical_qv_selectors_checked": 22,
            "canonical_qv_kernel_selectors": 11,
            "canonical_qv_nonzero_incapable_selectors": 11,
            "canonical_qv_capable_selectors": 0,
            "qv_selector_subspace_rank": 11,
            "qv_wedge_range_rank": 297,
            "qv_target_augmented_rank": 298,
            "combined_free_B2_selector_subspace_rank": 22,
            "combined_free_B2_wedge_range_rank": 473,
            "combined_target_augmented_rank": 474,
            "candidate_no_go_results": 12,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "treat_v_selector_as_nonzero": {
                "all_11_v_projections_zero": True,
                "rejected": True,
            },
            "treat_canonical_q_selector_as_capable": {
                "all_11_q_selectors_nonzero_but_incapable": True,
                "rejected": True,
            },
            "infer_capable_qv_linear_combination_from_nonzero_q_span": {
                "qv_target_plane_intersection_dimension": 0,
                "rejected": True,
            },
            "analyze_qv_and_C23_ranges_separately_only": {
                "combined_range_rank": 473,
                "combined_augmented_rank": 474,
                "rejected": True,
            },
            "claim_all_operator_classes_ruled_out": {
                "nonlinear_or_nonlocal_direction_dependence_unclassified": True,
                "rejected": True,
            },
            "promote_linear_no_go_to_global_TC2_closure": {
                "remaining_D4_selector_closed": False,
                "rejected": True,
            },
        },
        "claims": {
            "all_22_qv_canonical_selectors_classified": True,
            "qv_selector_subspace_completion_ruled_out": True,
            "combined_qv_and_C23_axis2_completion_ruled_out": True,
            "full_linear_gradient_annihilator_completion_class_ruled_out": True,
            "all_operator_classes_ruled_out": False,
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
            "Alter the fixed direction-one B1=V slice or introduce a rigorously admitted "
            "nonlinear, higher-degree, nonlocal, or pseudodifferential direction dependence. "
            "Every linear direction-homogeneous block on all 55 input columns that preserves "
            "B1 and annihilates the complete gradient lift is now ruled out."
        ),
        "scope": (
            "Exact exhaustive axis-two no-go for the full affine class B(n)=sum_i n_i B_i "
            "on all 55 state-input columns, with B1 fixed to V and B(n)L(n)=0. All 22 q/v "
            "canonical selectors and their complete linear span are classified, then combined "
            "with the full C23 spatial freedom. The combined projected range has rank 473 and "
            "the target raises it to 474. This does not rule out altered B1, nonlinear/higher-"
            "degree/nonlocal direction dependence, relaxed physical equivalence, spatial "
            "covariance, full D4 closure, tube closure, CK1, CK3, TC2, B7, global-H7, or lifespan."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "full linear gradient-annihilator content identity mismatch"
        )
    exact = document.get("exact_classification", {})
    partition = exact.get("canonical_qv_selector_partition", {})
    qv = exact.get("qv_subspace_range", {})
    combined = exact.get("combined_axis2_free_B2_range", {})
    consequence = exact.get("candidate_consequence", {})
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    if (
        document.get("status") != "pass_exact_full_linear_gradient_annihilator_completion_no_go"
        or counts
        != {
            "canonical_qv_selectors_checked": 22,
            "canonical_qv_kernel_selectors": 11,
            "canonical_qv_nonzero_incapable_selectors": 11,
            "canonical_qv_capable_selectors": 0,
            "qv_selector_subspace_rank": 11,
            "qv_wedge_range_rank": 297,
            "qv_target_augmented_rank": 298,
            "combined_free_B2_selector_subspace_rank": 22,
            "combined_free_B2_wedge_range_rank": 473,
            "combined_target_augmented_rank": 474,
            "candidate_no_go_results": 12,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        }
        or partition.get("kernel_indices") != list(V_INDICES)
        or partition.get("nonzero_incapable_indices") != list(Q_INDICES)
        or partition.get("capable_indices") != []
        or len(partition.get("records", [])) != 22
        or qv.get("selector_projection_rank") != 11
        or qv.get("selector_target_plane_intersection_dimension") != 0
        or qv.get("wedge_range_rank") != 297
        or qv.get("target_augmented_rank") != 298
        or qv.get("target_in_image") is not False
        or combined.get("selector_projection_rank") != 22
        or combined.get("selector_target_plane_intersection_dimension") != 0
        or combined.get("wedge_range_rank") != 473
        or combined.get("target_augmented_rank") != 474
        or combined.get("target_in_image") is not False
        or consequence.get("base_axis2_D4_RHS_identically_zero") is not True
        or consequence.get("candidate_linear_gradient_annihilator_completions") != 0
        or consequence.get("candidate_no_go_results") != EXPECTED_CANDIDATES
        or claims.get("all_22_qv_canonical_selectors_classified") is not True
        or claims.get("qv_selector_subspace_completion_ruled_out") is not True
        or claims.get("combined_qv_and_C23_axis2_completion_ruled_out") is not True
        or claims.get("full_linear_gradient_annihilator_completion_class_ruled_out") is not True
        or any(
            claims.get(key) is not False
            for key in (
                "all_operator_classes_ruled_out",
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
        raise QuarticTC2D4FullLinearGradientAnnihilatorNoGoError(
            "full linear gradient-annihilator exact/fail-closed contract mismatch"
        )


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify the full linear fixed-B1 gradient-annihilator escape class."
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
                "qv_capable": artifact["counts"]["canonical_qv_capable_selectors"],
                "combined_range_rank": artifact["counts"]["combined_free_B2_wedge_range_rank"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
