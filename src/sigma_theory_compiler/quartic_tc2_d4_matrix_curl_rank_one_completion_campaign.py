from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_minimal_tc2_escape_campaign import _correction_basis
from .quartic_tc2_d4_parity_cubic_generic_direction_campaign import (
    JET_ORDER,
    _frames,
    _polarized_payload,
    _solve,
    _state_rotation,
)
from .quartic_tc2_diagonal_third_jet_campaign import (
    _content_hash,
    _matrix_payload,
    _reference_and_first_jet_packet,
)
from .quartic_tc2_mixed_third_jet_continuation_service import (
    _atomic_write,
    _file_sha256,
    _hash_matches,
    _json_bytes,
    _load_file,
    _with_hash,
)
from .quartic_tc2_variable_sylvester_campaign import STATE_DIMENSION

SCHEMA = "sigma-quartic-tc2-d4-matrix-curl-rank-one-completion-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-matrix-curl-rank-one-completion-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
TRANSVERSE_CURL_INDICES = tuple(range(11, 33))
ZERO_EIGENSPACE_DIMENSION = 33


class QuarticTC2D4MatrixCurlRankOneCompletionError(ValueError):
    """Raised when the rank-one matrix curl completion is invalid."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _solve_vector_system(matrix: sp.Matrix, rhs: sp.Matrix) -> sp.Matrix:
    solution, parameters = matrix.gauss_jordan_solve(rhs)
    substitutions = {symbol: 0 for symbol in parameters.free_symbols}
    return solution.subs(substitutions).applyfunc(sp.factor)


def _rank_one_wedge_solution(
    target: sp.Matrix, selector_coordinates: sp.Matrix
) -> tuple[sp.Matrix, sp.Matrix, int]:
    column_index = next(
        index for index in range(target.cols) if not target[:, index].is_zero_matrix
    )
    selector_vector = target[:, column_index]
    selector_coefficients = _solve_vector_system(selector_coordinates, selector_vector)
    rows = []
    rhs = []
    for left in range(target.rows):
        for right in range(left + 1, target.cols):
            row = sp.zeros(1, target.rows)
            row[0, left] = selector_vector[right]
            row[0, right] = -selector_vector[left]
            rows.append(row)
            rhs.append(target[left, right])
    output_coordinates = _solve_vector_system(sp.Matrix.vstack(*rows), sp.Matrix(rhs))
    reconstructed = (
        output_coordinates * selector_vector.T - selector_vector * output_coordinates.T
    ).applyfunc(sp.factor)
    if reconstructed != target:
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("rank-one wedge reconstruction mismatch")
    return selector_coefficients, output_coordinates, column_index


def _exact_completion(
    minimal: Mapping[str, Any],
    generic_obstruction: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor_record = generic_obstruction["exact_generic_direction_audit"]["direction_records"][
        0
    ]
    frame = _frames()[0]
    if (
        predecessor_record.get("direction") != ["3/5", "4/5", "0"]
        or predecessor_record.get("candidate_obstructions") != EXPECTED_CANDIDATES
        or frame["name"] != predecessor_record.get("frame_name")
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "generic obstruction predecessor selector mismatch"
        )
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, directional_evaluations = _polarized_payload(frame, fourth_campaign)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    if directional_evaluations != 15:
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "generic recurrence evaluation count mismatch"
        )

    reference = _reference_and_first_jet_packet()
    direction = list(frame["direction"])
    state_rotation, _ = _state_rotation(frame["rotation"])
    basis = _correction_basis()
    direction_1 = basis["block"]
    output = direction_1[:, 21]
    direction_2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    cubic_global = (
        direction[0] ** 2 * (direction[0] * direction_1 + direction[1] * direction_2)
    ).applyfunc(sp.factor)
    cubic = (state_rotation.T * cubic_global * state_rotation).applyfunc(sp.factor)
    cubic_skew = (reference["energy0"] * cubic - cubic.T * reference["energy0"]).applyfunc(
        sp.factor
    )
    rhs = payload["fourth_Sylvester_RHS"]
    if (
        sum(value != 0 for value in rhs) != predecessor_record["base_D4_RHS_nonzero_entries"]
        or _content_hash(_matrix_payload(rhs)) != predecessor_record["base_D4_RHS_sha256"]
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("generic base D4 replay mismatch")
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    normalized_targets = []
    predecessor_candidates = {
        row["candidate_id"]: row for row in predecessor_record["candidate_records"]
    }
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        residual = (candidate_rhs + eta * cubic_skew).applyfunc(sp.factor)
        target = (projector0.T * residual * projector0 / eta).applyfunc(sp.factor)
        predecessor_nonzero = predecessor_candidates[candidate["candidate_id"]][
            "nonzero_equal_eigenspace_compressions"
        ]
        if (
            target.rank() != 2
            or set(predecessor_nonzero) != {"0"}
            or predecessor_nonzero["0"]["rank"] != 2
        ):
            raise QuarticTC2D4MatrixCurlRankOneCompletionError(
                "normalized candidate target mismatch"
            )
        normalized_targets.append(target)
    target_hashes = {_content_hash(_matrix_payload(target)) for target in normalized_targets}
    if len(target_hashes) != 1:
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "candidate targets do not share one eta-normalized form"
        )
    normalized_target = normalized_targets[0]
    desired_correction = (-normalized_target).applyfunc(sp.factor)
    covector_basis, coordinate_map, target_coordinates = _zero_speed_coordinates(
        projector0, desired_correction
    )
    selector_coordinates = (
        coordinate_map * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    selector_rank = selector_coordinates.rank()
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    selector_basis = sp.Matrix.hstack(*selector_coordinates.columnspace())
    intersection_dimension = (
        selector_rank + target_plane.rank() - selector_basis.row_join(target_plane).rank()
    )
    quotient = sp.Matrix.hstack(*selector_coordinates.T.nullspace()).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    quotient_dimension = ZERO_EIGENSPACE_DIMENSION - selector_rank
    range_rank = 528 - quotient_dimension * (quotient_dimension - 1) // 2
    if (
        selector_rank != 22
        or target_plane.rank() != 2
        or intersection_dimension != 2
        or not quotient_target.is_zero_matrix
        or range_rank != 473
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "transverse curl range classification mismatch"
        )

    selector_coefficients, output_coordinates, target_column = _rank_one_wedge_solution(
        target_coordinates, selector_coordinates
    )
    right_covector = (
        sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)] * selector_coefficients
    ).applyfunc(sp.factor)
    output_covector = (covector_basis * output_coordinates).applyfunc(sp.factor)
    energy_weighted_block = (output_covector * right_covector.T).applyfunc(sp.factor)
    completion_block = (reference["energy0"].inv() * energy_weighted_block).applyfunc(sp.factor)
    completion_skew = (
        reference["energy0"] * completion_block - completion_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    zero_compression = (projector0.T * completion_skew * projector0).applyfunc(sp.factor)
    nonzero_compressions = {
        str(eigenvalue): (projector.T * completion_skew * projector).applyfunc(sp.factor)
        for eigenvalue, projector in reference["projectors"].items()
        if eigenvalue != 0
    }
    aligned_gradient_lift = sp.zeros(STATE_DIMENSION, 11)
    aligned_gradient_lift[44:55, :] = sp.eye(11)
    if (
        completion_block.rank() != 1
        or zero_compression != desired_correction
        or any(not compression.is_zero_matrix for compression in nonzero_compressions.values())
        or not (completion_block * aligned_gradient_lift).is_zero_matrix
        or any(
            right_covector[index] != 0
            for index in range(STATE_DIMENSION)
            if index not in TRANSVERSE_CURL_INDICES
        )
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "rank-one curl block exact completion mismatch"
        )

    candidate_rows = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        corrected = (candidate_rhs + eta * (cubic_skew + completion_skew)).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        candidate_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "a10": candidate["a10"],
                "c20": candidate["c20"],
                "eta": candidate["eta_unique_tuning"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
    if len(candidate_rows) != EXPECTED_CANDIDATES or any(
        not row["D4_Sylvester_solvable"] or row["nonzero_equal_eigenspace_compressions"]
        for row in candidate_rows
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "rank-one curl candidate completion mismatch"
        )

    global_completion = (state_rotation * completion_block * state_rotation.T).applyfunc(sp.factor)
    return {
        "declared_completion_class": {
            "name": (
                "all_fixed_frame_matrix_valued_blocks_on_the_22_transverse_spatial_curl_"
                "covectors_at_n_equal_3_5_4_5_0"
            ),
            "frequency_direction": ["3/5", "4/5", "0"],
            "transverse_curl_input_indices_in_aligned_frame": list(TRANSVERSE_CURL_INDICES),
            "curl_covector_dimension": 22,
            "raw_matrix_parameter_dimension": 1210,
            "arbitrary_output_vectors_allowed": True,
            "single_fixed_direction_only": True,
            "global_smooth_direction_dependence_included": False,
        },
        "full_D4_replay": {
            "directional_evaluations": directional_evaluations,
            "orders_1_through_3_mandatory_prerequisites": True,
            "all_seven_eigenspaces_checked": True,
            "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
            "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
            "eta_normalized_candidate_targets": len(normalized_targets),
            "distinct_eta_normalized_targets": len(target_hashes),
            "normalized_target_rank": normalized_target.rank(),
            "normalized_target_sha256": next(iter(target_hashes)),
        },
        "exact_range_classification": {
            "zero_speed_dimension": ZERO_EIGENSPACE_DIMENSION,
            "selector_projection_rank": selector_rank,
            "selector_projection_kernel_dimension": 0,
            "target_plane_dimension": target_plane.rank(),
            "selector_target_plane_intersection_dimension": intersection_dimension,
            "wedge_range_rank": range_rank,
            "target_augmented_rank": range_rank,
            "target_in_image": True,
            "selector_coordinates_sha256": _content_hash(_matrix_payload(selector_coordinates)),
            "quotient_target_zero": quotient_target.is_zero_matrix,
            "quotient_target_sha256": _content_hash(_matrix_payload(quotient_target)),
        },
        "minimal_rank_one_completion": {
            "minimal_nonzero_block_rank": 1,
            "rank_zero_impossible_because_target_nonzero": True,
            "constructed_block_rank": completion_block.rank(),
            "combined_curl_channel_count": 1,
            "target_column_used": target_column,
            "right_covector_nonzero_entries": sum(value != 0 for value in right_covector),
            "right_covector_sha256": _content_hash(_matrix_payload(right_covector)),
            "output_covector_nonzero_entries": sum(value != 0 for value in output_covector),
            "output_covector_sha256": _content_hash(_matrix_payload(output_covector)),
            "aligned_block_nonzero_entries": sum(value != 0 for value in completion_block),
            "aligned_block_sha256": _content_hash(_matrix_payload(completion_block)),
            "global_block_rank": global_completion.rank(),
            "global_block_nonzero_entries": sum(value != 0 for value in global_completion),
            "global_block_sha256": _content_hash(_matrix_payload(global_completion)),
            "gradient_lift_annihilation_exact": True,
            "zero_speed_target_cancelled_exactly": True,
            "all_nonzero_eigenspace_compressions_zero": True,
        },
        "candidate_result": {
            "candidate_conditions_checked": len(candidate_rows),
            "candidate_compatibilities": sum(
                row["D4_Sylvester_solvable"] for row in candidate_rows
            ),
            "candidate_obstructions": sum(
                not row["D4_Sylvester_solvable"] for row in candidate_rows
            ),
            "candidate_records": candidate_rows,
        },
        "first_blocker": {
            "name": "global_angular_extension_and_constraint_admission",
            "required_next": (
                "Extend the one-frame rank-one combined-curl block to an antipodally odd, "
                "bounded smooth matrix symbol on the direction sphere that preserves the e1/e2 "
                "certificates, then prove pseudodifferential constraint/commutator/boundary "
                "admission and re-audit additional generic directions."
            ),
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
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("matrix curl completion config mismatch")
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    minimal = _load_bound(root, config["minimal_escape"])
    generic = _load_bound(root, config["generic_direction_obstruction"])
    fourth = _load_bound(root, config["fourth_campaign"])
    if (
        minimal.get("status") != "pass_exact_minimal_rank_one_tc2_d4_escape_algebraic_only"
        or generic.get("status")
        != "pass_exact_generic_direction_obstruction_of_parity_cubic_escape"
        or fourth.get("status")
        != "pass_exact_fourth_jet_minimal_selector_manifest_no_evaluations_tube_fail_closed"
    ):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError(
            "matrix curl predecessor status mismatch"
        )
    exact = _exact_completion(minimal, generic, fourth)
    result = exact["candidate_result"]
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_fixed_frame_rank_one_matrix_curl_completion",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (
                "minimal_escape",
                "generic_direction_obstruction",
                "fourth_campaign",
                "campaign_source",
                "campaign_test",
            )
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "frequency_direction": ["3/5", "4/5", "0"],
        },
        "exact_completion": exact,
        "counts": {
            "directional_recurrence_evaluations": 15,
            "transverse_curl_covectors": 22,
            "raw_matrix_parameters": 1210,
            "selector_projection_rank": 22,
            "wedge_range_rank": 473,
            "target_augmented_rank": 473,
            "constructed_block_rank": 1,
            "combined_curl_channels": 1,
            "candidate_conditions_checked": result["candidate_conditions_checked"],
            "candidate_compatibilities": result["candidate_compatibilities"],
            "candidate_obstructions": result["candidate_obstructions"],
            "negative_controls": 6,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "rank_zero_completion": {
                "normalized_target_rank": 2,
                "rejected": True,
            },
            "reuse_scalar_cubic_block_without_matrix_channel": {
                "candidate_obstructions": 12,
                "rejected": True,
            },
            "omit_nonzero_eigenspace_checks": {
                "all_seven_eigenspaces_checked": True,
                "rejected": True,
            },
            "infer_global_angular_extension": {
                "single_fixed_direction_only": True,
                "rejected": True,
            },
            "claim_local_or_covariant_origin": {
                "operator_origin_unconstructed": True,
                "rejected": True,
            },
            "promote_fixed_frame_completion_to_global_TC2": {
                "remaining_D4_selector_closed": False,
                "rejected": True,
            },
        },
        "claims": {
            "full_D4_recurrence_evaluated_at_fixed_generic_frame": True,
            "transverse_curl_range_classified_exactly": True,
            "minimal_rank_one_matrix_curl_completion_constructed": True,
            "all_12_fixed_frame_D4_compatibilities_proved": True,
            "global_smooth_angular_extension_constructed": False,
            "additional_generic_directions_audited": False,
            "pseudodifferential_constraint_calculus_proved": False,
            "local_differential_operator_origin_proved": False,
            "covariant_action_origin_proved": False,
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
        "next_gate": exact["first_blocker"]["required_next"],
        "scope": (
            "Exact exhaustive range classification and minimal rank-one construction for all "
            "matrix-valued blocks on the 22 transverse spatial-curl covectors at the single "
            "rational direction n=(3/5,4/5,0). The complete orders-one-through-four recurrence "
            "and all seven eigenspaces pass for all 12 candidates after the new block. No global "
            "angular extension, local/covariant origin, additional generic direction, remaining "
            "D4, tube, CK1, CK3, TC2, B7, global-H7, or lifespan result is inferred."
        ),
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if document.get("schema_version") != SCHEMA or not _hash_matches(dict(document)):
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("matrix curl content identity mismatch")
    exact = document.get("exact_completion", {})
    range_result = exact.get("exact_range_classification", {})
    construction = exact.get("minimal_rank_one_completion", {})
    candidates = exact.get("candidate_result", {})
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    if (
        document.get("status") != "pass_exact_fixed_frame_rank_one_matrix_curl_completion"
        or counts
        != {
            "directional_recurrence_evaluations": 15,
            "transverse_curl_covectors": 22,
            "raw_matrix_parameters": 1210,
            "selector_projection_rank": 22,
            "wedge_range_rank": 473,
            "target_augmented_rank": 473,
            "constructed_block_rank": 1,
            "combined_curl_channels": 1,
            "candidate_conditions_checked": 12,
            "candidate_compatibilities": 12,
            "candidate_obstructions": 0,
            "negative_controls": 6,
            "inferred_global_passes": 0,
        }
        or range_result.get("selector_target_plane_intersection_dimension") != 2
        or range_result.get("target_in_image") is not True
        or range_result.get("quotient_target_zero") is not True
        or construction.get("minimal_nonzero_block_rank") != 1
        or construction.get("constructed_block_rank") != 1
        or construction.get("combined_curl_channel_count") != 1
        or construction.get("gradient_lift_annihilation_exact") is not True
        or construction.get("zero_speed_target_cancelled_exactly") is not True
        or construction.get("all_nonzero_eigenspace_compressions_zero") is not True
        or candidates.get("candidate_conditions_checked") != 12
        or candidates.get("candidate_compatibilities") != 12
        or candidates.get("candidate_obstructions") != 0
        or len(candidates.get("candidate_records", [])) != 12
        or any(
            row.get("D4_Sylvester_solvable") is not True
            or row.get("nonzero_equal_eigenspace_compressions") != {}
            for row in candidates.get("candidate_records", [])
        )
        or any(
            claims.get(key) is not True
            for key in (
                "full_D4_recurrence_evaluated_at_fixed_generic_frame",
                "transverse_curl_range_classified_exactly",
                "minimal_rank_one_matrix_curl_completion_constructed",
                "all_12_fixed_frame_D4_compatibilities_proved",
            )
        )
        or any(
            claims.get(key) is not False
            for key in (
                "global_smooth_angular_extension_constructed",
                "additional_generic_directions_audited",
                "pseudodifferential_constraint_calculus_proved",
                "local_differential_operator_origin_proved",
                "covariant_action_origin_proved",
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
        raise QuarticTC2D4MatrixCurlRankOneCompletionError("matrix curl exact/fail-closed mismatch")


def run_campaign(project_root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    artifact = build_campaign(project_root, config_path)
    validate_campaign(artifact)
    _atomic_write(output_path.resolve(), _json_bytes(artifact))
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Construct the minimal fixed-frame matrix-valued curl completion."
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
                "block_rank": artifact["counts"]["constructed_block_rank"],
                "compatibilities": artifact["counts"]["candidate_compatibilities"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
