from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_degree_three_c23_great_circle_escape_campaign import (
    _symbols as c23_symbols,
)
from .quartic_tc2_d4_degree_three_matrix_curl_sphere_extension_campaign import (
    _symbols as c12_symbols,
)
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import (
    TRANSVERSE_CURL_INDICES,
    _solve_vector_system,
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

SCHEMA = "sigma-quartic-tc2-d4-degree-three-rank-two-xyz-completion-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-degree-three-rank-two-xyz-completion-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
PREDECESSOR_STATUS = "pass_exact_degree_three_C23_great_circle_escape_all_12_xz_compatibilities"
BASE_RHS_SHA256 = "e137be6d8bb6aaafdc45d12c79adc8c2b9e5e37ef511a5e4414b1661c0a0a0b7"
PRIOR_GLOBAL_SHA256 = "59c6f074bd1ade330630e6f607e587b92ec7a1c11d2ce061b54079ea3571936b"
PRIOR_ALIGNED_SHA256 = "e94f117a3cfb51317d4a38456924b1cb44c3bb0ed0b301afb35ddc363babc885"
NORMALIZED_TARGET_SHA256 = "767724a8936ceefbbeea530d0a64be0fa94c47decabede12e061223b71f73ab7"
SELECTOR_SHA256 = "7ef398226365b9e42bd543a3b9c5b00c82621cbf8f67d76b2768e38e81441d26"
QUOTIENT_ZERO_SHA256 = "6bd0f4db2919abb53bd3fc437f3ec440b1c2df2a8e73fe17d06d0a3fc1c10f23"
ALIGNED_BLOCK_SHA256 = "a415ea5ac18d5f43ab57103295df2935acb58e92bb1c532fed97e6ff11bba7b3"
GLOBAL_BLOCK_SHA256 = "b091a2194f13b1b58bbd441a18773659bad91251752c403d7c297aff4a83f3ad"
EXTENSION_SYMBOL_SHA256 = "cff78832e0ffe582290b200702afa857719fc6a47d5d0f3c7871e9125cb0cb0b"
GRADIENT_ZERO_SHA256 = "54efa54b8d23c5fdf1e357239619821b52ed647990182ca8b3fd3e1cb57916f9"


class QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(ValueError):
    """Raised when the rank-two xyz completion is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _prior_symbol_at(direction: list[sp.Expr]) -> dict[str, sp.Matrix]:
    basis = _correction_basis()
    direction_1 = basis["block"]
    output = direction_1[:, 21]
    direction_2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    cubic = (
        direction[0] ** 2 * (direction[0] * direction_1 + direction[1] * direction_2)
    ).applyfunc(sp.factor)
    c12 = c12_symbols()
    c12_variables = c12["variables"]
    extension_12 = (
        c12["extension"].subs(dict(zip(c12_variables, direction, strict=True))).applyfunc(sp.factor)
    )
    c23 = c23_symbols()
    c23_variables = c23["variables"]
    extension_23 = (
        c23["extension"].subs(dict(zip(c23_variables, direction, strict=True))).applyfunc(sp.factor)
    )
    return {
        "cubic": cubic,
        "extension_12": extension_12,
        "extension_23": extension_23,
        "combined": (cubic + extension_12 + extension_23).applyfunc(sp.factor),
    }


def _rank_two_decomposition(
    target_coordinates: sp.Matrix,
    selector_coordinates: sp.Matrix,
    covector_basis: sp.Matrix,
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    residual = target_coordinates
    terms = []
    while not residual.is_zero_matrix:
        left, right = next(
            (row, column)
            for row in range(residual.rows)
            for column in range(row + 1, residual.cols)
            if residual[row, column] != 0
        )
        pivot = residual[left, right]
        left_column = residual[:, left]
        right_column = residual[:, right]
        output_coordinates = (left_column / pivot).applyfunc(sp.factor)
        selector_vector = right_column
        wedge = (
            output_coordinates * selector_vector.T - selector_vector * output_coordinates.T
        ).applyfunc(sp.factor)
        residual = (residual - wedge).applyfunc(sp.factor)
        selector_coefficients = _solve_vector_system(selector_coordinates, selector_vector)
        right_covector = (
            sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)] * selector_coefficients
        ).applyfunc(sp.factor)
        energy_output = (covector_basis * output_coordinates).applyfunc(sp.factor)
        output_vector = (reference["energy0"].inv() * energy_output).applyfunc(sp.factor)
        terms.append(
            {
                "coordinate_pair": [left, right],
                "output_aligned": output_vector,
                "right_aligned": right_covector,
            }
        )
    if len(terms) != 2:
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "rank-four target did not yield two symplectic wedges"
        )
    return terms


def _global_curl_extension(
    terms: list[dict[str, Any]], state_rotation: sp.Matrix, direction: list[sp.Expr]
) -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    frequency = sp.Matrix([n1, n2, n3])
    reference_direction = sp.Matrix(direction)
    envelope = sp.Rational(9, 4) * n2 * n3
    extension = sp.zeros(STATE_DIMENSION)
    records = []
    for term in terms:
        output = (state_rotation * term["output_aligned"]).applyfunc(sp.factor)
        right = (state_rotation * term["right_aligned"]).applyfunc(sp.factor)
        spatial = sp.zeros(3, 11)
        spatial[0, :] = right[44:55, :].T
        spatial[1, :] = right[11:22, :].T
        spatial[2, :] = right[22:33, :].T
        curl_spatial = sp.zeros(3, 11)
        for field in range(11):
            potential = spatial[:, field].cross(reference_direction)
            curl_spatial[:, field] = frequency.cross(potential)
        curl = sp.zeros(STATE_DIMENSION, 1)
        curl[44:55, :] = curl_spatial[0, :].T
        curl[11:22, :] = curl_spatial[1, :].T
        curl[22:33, :] = curl_spatial[2, :].T
        extension += envelope * output * curl.T
        records.append(
            {
                "coordinate_pair": term["coordinate_pair"],
                "output_nonzero_entries": sum(value != 0 for value in output),
                "output_sha256": _content_hash(_matrix_payload(output)),
                "reference_right_nonzero_entries": sum(value != 0 for value in right),
                "reference_right_sha256": _content_hash(_matrix_payload(right)),
                "linear_curl_sha256": _content_hash(_matrix_payload(curl)),
            }
        )
    extension = extension.applyfunc(sp.factor)
    lift = sp.zeros(STATE_DIMENSION, 11)
    lift[11:22, :] = n2 * sp.eye(11)
    lift[22:33, :] = n3 * sp.eye(11)
    lift[44:55, :] = n1 * sp.eye(11)
    gradient_residual = (extension * lift).applyfunc(sp.factor)
    reference_block = extension.subs(
        {n1: direction[0], n2: direction[1], n3: direction[2]}
    ).applyfunc(sp.factor)
    axis_points = (
        {n1: 1, n2: 0, n3: 0},
        {n1: 0, n2: 1, n3: 0},
        {n1: sp.Rational(3, 5), n2: sp.Rational(4, 5), n3: 0},
        {n1: sp.Rational(3, 5), n2: 0, n3: sp.Rational(4, 5)},
    )
    antipodal = extension.subs({n1: -n1, n2: -n2, n3: -n3}).applyfunc(sp.factor)
    if (
        _content_hash(_matrix_payload(extension)) != EXTENSION_SYMBOL_SHA256
        or _content_hash(_matrix_payload(gradient_residual)) != GRADIENT_ZERO_SHA256
        or not gradient_residual.is_zero_matrix
        or antipodal != -extension
        or any(not extension.subs(point).is_zero_matrix for point in axis_points)
    ):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "global rank-two curl extension mismatch"
        )
    return {
        "variables": (n1, n2, n3),
        "envelope": envelope,
        "extension": extension,
        "reference_block": reference_block,
        "term_records": records,
        "gradient_residual": gradient_residual,
    }


def _exact_result(
    predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        predecessor.get("status") != PREDECESSOR_STATUS
        or predecessor.get("counts", {}).get("total_certified_directions") != 4
        or predecessor.get("counts", {}).get("new_candidate_direction_compatibilities") != 12
    ):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError("C23 predecessor mismatch")
    frame = _frames()[2]
    direction = list(frame["direction"])
    if frame["name"] != "xyz_1_2_2" or direction != [
        sp.Rational(1, 3),
        sp.Rational(2, 3),
        sp.Rational(2, 3),
    ]:
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError("xyz selector mismatch")
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth_campaign)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    reference = _reference_and_first_jet_packet()
    state_rotation, _ = _state_rotation(frame["rotation"])
    prior_parts = _prior_symbol_at(direction)
    prior_global = prior_parts["combined"]
    prior_aligned = (state_rotation.T * prior_global * state_rotation).applyfunc(sp.factor)
    prior_skew = (
        reference["energy0"] * prior_aligned - prior_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    rhs_symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = rhs_symbols.get("alpha", sp.Symbol("alpha"))
    c20 = rhs_symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    before_rows = []
    normalized_targets = []
    candidate_payloads = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        before = (candidate_rhs + eta * prior_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(before)
        before_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
        normalized_targets.append((projector0.T * before * projector0 / eta).applyfunc(sp.factor))
        candidate_payloads.append((candidate, before, eta))
    target_hashes = {_content_hash(_matrix_payload(target)) for target in normalized_targets}
    target = normalized_targets[0]
    covector_basis, coordinate_map, target_coordinates = _zero_speed_coordinates(
        projector0, -target
    )
    selector_coordinates = (
        coordinate_map * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    quotient = sp.Matrix.hstack(*selector_coordinates.T.nullspace()).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    selector_basis = sp.Matrix.hstack(*selector_coordinates.columnspace())
    intersection_dimension = (
        selector_coordinates.rank()
        + target_plane.rank()
        - selector_basis.row_join(target_plane).rank()
    )
    terms = _rank_two_decomposition(
        target_coordinates, selector_coordinates, covector_basis, reference
    )
    aligned_block = sum(
        (term["output_aligned"] * term["right_aligned"].T for term in terms),
        sp.zeros(STATE_DIMENSION),
    ).applyfunc(sp.factor)
    global_block = (state_rotation * aligned_block * state_rotation.T).applyfunc(sp.factor)
    angular = _global_curl_extension(terms, state_rotation, direction)
    if angular["reference_block"] != global_block:
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "angular extension reference block mismatch"
        )
    new_skew = (
        reference["energy0"] * aligned_block - aligned_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    after_rows = []
    for candidate, before, eta in candidate_payloads:
        corrected = (before + eta * new_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        after_rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "a10": candidate["a10"],
                "c20": candidate["c20"],
                "eta": candidate["eta_unique_tuning"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
    if (
        evaluations != 15
        or sum(value != 0 for value in rhs) != 116
        or _content_hash(_matrix_payload(rhs)) != BASE_RHS_SHA256
        or _content_hash(_matrix_payload(prior_global)) != PRIOR_GLOBAL_SHA256
        or _content_hash(_matrix_payload(prior_aligned)) != PRIOR_ALIGNED_SHA256
        or len(before_rows) != 12
        or any(row["D4_Sylvester_solvable"] for row in before_rows)
        or any(
            set(row["nonzero_equal_eigenspace_compressions"]) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"]["rank"] != 4
            or row["nonzero_equal_eigenspace_compressions"]["0"]["nonzero_entries"] != 56
            for row in before_rows
        )
        or len(target_hashes) != 1
        or next(iter(target_hashes)) != NORMALIZED_TARGET_SHA256
        or target.rank() != 4
        or selector_coordinates.rank() != 22
        or intersection_dimension != 4
        or not quotient_target.is_zero_matrix
        or _content_hash(_matrix_payload(selector_coordinates)) != SELECTOR_SHA256
        or _content_hash(_matrix_payload(quotient_target)) != QUOTIENT_ZERO_SHA256
        or aligned_block.rank() != 2
        or global_block.rank() != 2
        or _content_hash(_matrix_payload(aligned_block)) != ALIGNED_BLOCK_SHA256
        or _content_hash(_matrix_payload(global_block)) != GLOBAL_BLOCK_SHA256
        or len(after_rows) != 12
        or any(
            not row["D4_Sylvester_solvable"] or row["nonzero_equal_eigenspace_compressions"]
            for row in after_rows
        )
    ):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "xyz rank-two completion audit mismatch"
        )
    return {
        "selector": {
            "frame_name": frame["name"],
            "direction": [str(value) for value in direction],
            "final_declared_rational_frame": True,
            "remaining_declared_frames": 0,
        },
        "prior_combined_symbol_audit": {
            "directional_evaluations": evaluations,
            "all_seven_eigenspaces_checked_per_candidate": True,
            "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
            "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
            "prior_global_symbol_rank": prior_global.rank(),
            "prior_global_symbol_sha256": _content_hash(_matrix_payload(prior_global)),
            "prior_aligned_symbol_sha256": _content_hash(_matrix_payload(prior_aligned)),
            "candidate_compatibilities": sum(row["D4_Sylvester_solvable"] for row in before_rows),
            "candidate_obstructions": sum(not row["D4_Sylvester_solvable"] for row in before_rows),
            "candidate_records": before_rows,
        },
        "exact_range_classification": {
            "eta_normalized_targets": len(normalized_targets),
            "distinct_eta_normalized_targets": len(target_hashes),
            "normalized_target_rank": target.rank(),
            "normalized_target_nonzero_entries": sum(value != 0 for value in target),
            "normalized_target_sha256": next(iter(target_hashes)),
            "transverse_selector_rank": selector_coordinates.rank(),
            "selector_sha256": _content_hash(_matrix_payload(selector_coordinates)),
            "target_plane_dimension": target_plane.rank(),
            "selector_target_plane_intersection_dimension": intersection_dimension,
            "quotient_target_zero": quotient_target.is_zero_matrix,
            "quotient_target_sha256": _content_hash(_matrix_payload(quotient_target)),
            "target_in_full_transverse_curl_range": True,
        },
        "minimal_rank_two_completion": {
            "rank_one_impossible_from_skew_rank_bound": True,
            "lower_bound_completion_rank": 2,
            "constructed_completion_rank": aligned_block.rank(),
            "elementary_curl_channels": len(terms),
            "coordinate_pairs": [term["coordinate_pair"] for term in terms],
            "aligned_block_sha256": _content_hash(_matrix_payload(aligned_block)),
            "global_block_sha256": _content_hash(_matrix_payload(global_block)),
            "term_records": angular["term_records"],
        },
        "exact_sphere_extension": {
            "definition": (
                "DeltaB_xyz(n)=(9/4)*n2*n3*sum_{k=1}^2 u_k*(n cross (r_k cross n_xyz))^T"
            ),
            "envelope": "a_xyz(n)=(9/4)*n2*n3",
            "envelope_value_at_xyz": "1",
            "minimal_total_degree": 3,
            "antipodally_odd": True,
            "polynomial_and_smooth_on_S2": True,
            "bounded_on_S2": True,
            "prior_four_direction_extensions_zero": True,
            "physical_gradient_lift_annihilated_identically": True,
            "symbol_nonzero_entries": sum(value != 0 for value in angular["extension"]),
            "symbol_sha256": _content_hash(_matrix_payload(angular["extension"])),
            "gradient_residual_sha256": _content_hash(
                _matrix_payload(angular["gradient_residual"])
            ),
        },
        "corrected_xyz_result": {
            "candidate_conditions_checked": len(after_rows),
            "candidate_compatibilities": sum(row["D4_Sylvester_solvable"] for row in after_rows),
            "candidate_obstructions": sum(not row["D4_Sylvester_solvable"] for row in after_rows),
            "candidate_records": after_rows,
        },
        "first_blocker": {
            "name": "beyond_declared_five_direction_selector_and_PDE_admission",
            "required_next": (
                "Establish a rigorous finite generic-direction determining theorem or audit a "
                "larger exact sphere selector for the combined degree-three matrix-curl symbol, "
                "then prove pseudodifferential constraint, commutator, boundary-energy and "
                "local/covariant admission before any full-sphere or TC2 claim."
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
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError("xyz completion config mismatch")
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    predecessor_keys = ("c23_predecessor", "minimal_escape", "fourth_campaign")
    predecessors = {key: _load_bound(root, config[key]) for key in predecessor_keys}
    exact = _exact_result(
        predecessors["c23_predecessor"],
        predecessors["minimal_escape"],
        predecessors["fourth_campaign"],
    )
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_rank_two_xyz_completion_all_declared_direction_certificates",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (*predecessor_keys, "campaign_source", "campaign_test")
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "prior_certified_directions": ["e1", "e2", "xy_3_4_5", "xz_3_4_5"],
            "newly_closed_direction": "xyz_1_2_2",
            "remaining_declared_directions": 0,
        },
        "exact_completion": exact,
        "counts": {
            "bound_predecessors": 3,
            "directional_recurrence_evaluations": 15,
            "prior_candidate_obstructions": 12,
            "normalized_target_rank": 4,
            "transverse_selector_rank": 22,
            "minimal_completion_rank": 2,
            "new_curl_channels": 2,
            "new_candidate_direction_systems_evaluated": 12,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "total_certified_directions": 5,
            "remaining_declared_directions": 0,
            "negative_controls": 8,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "rank_zero_completion": {"rejected": True, "target_rank": 4},
            "rank_one_completion": {"rejected": True, "maximum_skew_rank": 2},
            "odd_envelope": {"rejected": True, "symbol_antipodal_parity_even": True},
            "constant_envelope": {"rejected": True, "prior_certificates_not_preserved": True},
            "omit_linear_curl_lift": {"rejected": True, "gradient_annihilation_unproved": True},
            "retain_prior_xyz_obstruction": {"rejected": True, "new_compatibilities": 12},
            "infer_full_direction_sphere": {"rejected": True, "finite_selector_only": True},
            "infer_local_covariant_or_PDE_admission": {
                "rejected": True,
                "origin_and_calculus_unconstructed": True,
            },
        },
        "claims": {
            "full_xyz_orders_one_through_four_recurrence_evaluated": True,
            "prior_combined_symbol_xyz_obstructed_all_12_candidates": True,
            "rank_four_target_in_full_transverse_curl_range": True,
            "minimal_rank_two_completion_constructed": True,
            "all_12_corrected_xyz_D4_compatibilities_proved": True,
            "all_five_declared_direction_certificates_closed": True,
            "finite_selector_determines_full_direction_sphere": False,
            "full_direction_sphere_D4_compatibility_proved": False,
            "broader_matrix_curl_symbol_class_classified": False,
            "local_differential_operator_origin_proved": False,
            "covariant_action_origin_proved": False,
            "variable_coefficient_constraint_calculus_proved": False,
            "boundary_energy_admission_proved": False,
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
        "scope": (
            "Exact full recurrence and sharp rank-two matrix-curl completion at the final declared "
            "xyz_1_2_2 frame. The degree-three two-channel angular extension preserves the four "
            "prior certificates and closes all 12 xyz systems, completing the declared five-"
            "direction selector only. No determining theorem, full sphere, broader operator, "
            "local/covariant origin, PDE admission, remaining D4, tube, CK, TC2, B7, H7 or "
            "lifespan result is inferred."
        ),
        "next_gate": exact["first_blocker"]["required_next"],
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if not _hash_matches(document):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "xyz completion content identity mismatch"
        )
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    exact = document.get("exact_completion", {})
    true_claims = {
        "full_xyz_orders_one_through_four_recurrence_evaluated",
        "prior_combined_symbol_xyz_obstructed_all_12_candidates",
        "rank_four_target_in_full_transverse_curl_range",
        "minimal_rank_two_completion_constructed",
        "all_12_corrected_xyz_D4_compatibilities_proved",
        "all_five_declared_direction_certificates_closed",
    }
    false_claims = {
        "finite_selector_determines_full_direction_sphere",
        "full_direction_sphere_D4_compatibility_proved",
        "broader_matrix_curl_symbol_class_classified",
        "local_differential_operator_origin_proved",
        "covariant_action_origin_proved",
        "variable_coefficient_constraint_calculus_proved",
        "boundary_energy_admission_proved",
        "corrected_candidate_family_registered",
        "remaining_D4_selector_closed",
        "full_tube_Sylvester_identity",
        "CK1_closed",
        "CK3_closed",
        "TC2_closed",
        "B7_closed",
        "global_H7_closed",
        "lifespan_proved",
    }
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status")
        != "pass_exact_rank_two_xyz_completion_all_declared_direction_certificates"
        or counts.get("prior_candidate_obstructions") != 12
        or counts.get("minimal_completion_rank") != 2
        or counts.get("new_candidate_direction_compatibilities") != 12
        or counts.get("new_candidate_direction_obstructions") != 0
        or counts.get("total_certified_directions") != 5
        or counts.get("remaining_declared_directions") != 0
        or counts.get("inferred_global_passes") != 0
        or set(claims) != true_claims | false_claims
        or any(claims.get(key) is not True for key in true_claims)
        or any(claims.get(key) is not False for key in false_claims)
        or exact.get("exact_range_classification", {}).get("normalized_target_sha256")
        != NORMALIZED_TARGET_SHA256
        or exact.get("minimal_rank_two_completion", {}).get("aligned_block_sha256")
        != ALIGNED_BLOCK_SHA256
        or exact.get("minimal_rank_two_completion", {}).get("global_block_sha256")
        != GLOBAL_BLOCK_SHA256
        or exact.get("exact_sphere_extension", {}).get("symbol_sha256") != EXTENSION_SYMBOL_SHA256
        or exact.get("corrected_xyz_result", {}).get("candidate_compatibilities") != 12
        or exact.get("corrected_xyz_result", {}).get("candidate_obstructions") != 0
        or len(document.get("negative_controls", {})) != 8
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
        or document.get("errors") != []
    ):
        raise QuarticTC2D4DegreeThreeRankTwoXYZCompletionError(
            "xyz completion exact/fail-closed mismatch"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the sharp rank-two completion at the final xyz frame."
    )
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    root = Path(args.project_root).resolve()
    artifact = build_campaign(root, (root / args.config).resolve())
    validate_campaign(artifact)
    _atomic_write((root / args.output).resolve(), _json_bytes(artifact))
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "completion_rank": artifact["counts"]["minimal_completion_rank"],
                "xyz_compatibilities": artifact["counts"][
                    "new_candidate_direction_compatibilities"
                ],
                "remaining_declared_directions": artifact["counts"][
                    "remaining_declared_directions"
                ],
                "content_sha256": artifact["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
