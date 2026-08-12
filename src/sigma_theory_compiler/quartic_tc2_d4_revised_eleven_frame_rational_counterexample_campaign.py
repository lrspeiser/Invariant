from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_degree_five_counterexample_escape_campaign as degree_five
from . import quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign as xyz_campaign
from . import quartic_tc2_d4_rational_chart_determining_gate as rational_campaign
from . import quartic_tc2_d4_revised_eight_frame_rational_counterexample_campaign as common
from . import quartic_tc2_d4_revised_ten_frame_rational_counterexample_campaign as prior
from .quartic_tc2_d4_degree_three_sixth_frame_completion_campaign import _solve_vector_system
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import TRANSVERSE_CURL_INDICES
from .quartic_tc2_d4_parity_cubic_generic_direction_campaign import _solve, _state_rotation
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

SCHEMA = "sigma-quartic-tc2-d4-revised-eleven-frame-rational-counterexample-campaign-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-revised-eleven-frame-rational-counterexample-config-1.0"
)
STATUS = "pass_exact_eleven_frame_counterexample_and_bounded_repair_class_exhaustion"
EXPECTED_CANDIDATES = 12
EXPECTED_PREDECESSOR_CONTENT_SHA256 = (
    "fdbcf67e59e2cd9977cab22b1891f6d067183427f0d00b6fb841647638107e2e"
)
EXPECTED_PREDECESSOR_EXACT_SHA256 = (
    "172cf7a3d659fad3158e6ecc0f0685c5d99a091e486fc29f80e7fbb33ad015ad"
)
EXPECTED_A11_EXTENSION_SHA256 = (
    "084d66fd116545bb3d1b83837dc661d44683979631858666a8f762e1a5528900"
)
EXPECTED_A11_TARGET_SHA256 = (
    "6f05c62c3a4f8d90b13901a43db4109acac8865477f433d9437c3fea310edb01"
)
EXPECTED_EXACT_GATE_SHA256 = "9f9eec94db2487b1254189882b28c3d2038ff8dfe45b7f2a5b3149881a7d9e36"

TRUE_CLAIMS = {
    "all_12_candidates_obstructed_at_first_eleven_frame_search_point",
    "bounded_preserving_envelope_class_exhausted",
    "exact_regular_rational_counterexample_proved_for_eleven_frame_symbol",
    "exact_two_chart_SO3_atlas_reused",
    "first_obstruction_is_zero_speed_only",
    "full_orders_one_through_four_recurrence_evaluated",
    "preregistered_signed_height_one_selector_exhausted",
    "revised_eleven_frame_symbol_full_sphere_D4_compatibility_disproved",
}
FALSE_CLAIMS = {
    "B7_closed",
    "CK1_closed",
    "CK3_closed",
    "TC2_closed",
    "boundary_energy_admission_proved",
    "corrected_candidate_family_registered",
    "covariant_action_origin_proved",
    "finite_selector_determines_full_direction_sphere",
    "full_direction_sphere_D4_compatibility_proved",
    "full_tube_Sylvester_identity",
    "global_H7_closed",
    "lifespan_proved",
    "local_differential_operator_origin_proved",
    "revised_twelve_frame_symbol_full_sphere_D4_compatibility_proved",
    "twelfth_local_direction_certificate_constructed",
    "variable_coefficient_constraint_calculus_proved",
}


class RevisedElevenFrameRationalCounterexampleError(ValueError):
    """Raised when the eleven-frame rational certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RevisedElevenFrameRationalCounterexampleError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RevisedElevenFrameRationalCounterexampleError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RevisedElevenFrameRationalCounterexampleError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RevisedElevenFrameRationalCounterexampleError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _known_a11_envelope() -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    envelope = (
        sp.Rational(3, 16) * n1**2 * n2 * n3
        - sp.Rational(81, 32) * n1 * n2**2 * n3
        + sp.Rational(81, 32) * n1 * n2 * n3**2
        + sp.Rational(33, 32) * n2**3 * n3
        + sp.Rational(81, 64) * n2**2 * n3**2
        - sp.Rational(75, 32) * n2 * n3**3
    )
    prior_points = prior._known_a10_envelope()["prior_points"] + (
        prior._known_a10_envelope()["target"],
    )
    return {
        "variables": (n1, n2, n3),
        "envelope": envelope,
        "prior_points": prior_points,
        "target": (sp.Rational(-1, 3), sp.Rational(-2, 3), sp.Rational(2, 3)),
    }


def _reconstruct_extension(
    rotation: sp.Matrix,
    direction: list[sp.Expr],
    payload: Mapping[str, Any],
    extensions: Mapping[str, Mapping[str, Any]],
    envelope: Mapping[str, Any],
    minimal: Mapping[str, Any],
) -> Mapping[str, Any]:
    base = xyz_campaign._prior_symbol_at(direction)["combined"]
    global_symbol = (
        base
        + sum(
            (common._evaluate_extension(value, direction) for value in extensions.values()),
            sp.zeros(STATE_DIMENSION),
        )
    ).applyfunc(sp.factor)
    state_rotation, _ = _state_rotation(rotation)
    aligned = (state_rotation.T * global_symbol * state_rotation).applyfunc(sp.factor)
    reference = _reference_and_first_jet_packet()
    skew = (reference["energy0"] * aligned - aligned.T * reference["energy0"]).applyfunc(
        sp.factor
    )
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    candidate = minimal["exact_escape"]["candidate_classification"][0]
    eta = sp.sympify(candidate["eta_unique_tuning"])
    candidate_rhs = rhs.subs(
        {
            symbols.get("alpha", sp.Symbol("alpha")): sp.sympify(candidate["a10"]),
            symbols.get("c20", sp.Symbol("c20")): sp.sympify(candidate["c20"]),
        }
    ).applyfunc(sp.factor)
    projector0 = reference["projectors"][sp.S.Zero]
    target = (
        projector0.T * (candidate_rhs + eta * skew) * projector0 / eta
    ).applyfunc(sp.factor)
    covectors, coordinates, target_coordinates = _zero_speed_coordinates(projector0, -target)
    selector = (
        coordinates
        * projector0.T
        * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    terms = _bounded_transverse_curl_decompose(
        target_coordinates, selector, covectors, reference
    )
    return degree_five._angular_extension(terms, state_rotation, tuple(direction), envelope)


def _capture_eleven_frame_extensions(
    eleven_frame_predecessor: Mapping[str, Any],
    ten_frame_predecessor: Mapping[str, Any],
    nine_frame_predecessor: Mapping[str, Any],
    revised_predecessor: Mapping[str, Any],
    degree_five_predecessor: Mapping[str, Any],
    rational_predecessor: Mapping[str, Any],
    xyz_predecessor: Mapping[str, Any],
    c23_predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if (
        eleven_frame_predecessor.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or _content_hash(eleven_frame_predecessor.get("exact_gate", {}))
        != EXPECTED_PREDECESSOR_EXACT_SHA256
    ):
        raise RevisedElevenFrameRationalCounterexampleError(
            "eleven-frame predecessor certificate mismatch"
        )
    extensions = prior._capture_ten_frame_extensions(
        ten_frame_predecessor,
        nine_frame_predecessor,
        revised_predecessor,
        degree_five_predecessor,
        rational_predecessor,
        xyz_predecessor,
        c23_predecessor,
        minimal,
        fourth,
    )
    atlas = rational_campaign._atlas()
    rotation, direction, payload, evaluations = common._frame_payload(
        atlas,
        (sp.Integer(-1), sp.Integer(1)),
        "reconstructed_height_one_minus_1_1",
        fourth,
    )
    if direction != [sp.Rational(-1, 3), sp.Rational(-2, 3), sp.Rational(2, 3)] or evaluations != 15:
        raise RevisedElevenFrameRationalCounterexampleError("a11 reconstruction frame mismatch")
    extension = _reconstruct_extension(
        rotation, direction, payload, extensions, _known_a11_envelope(), minimal
    )
    if _content_hash(_matrix_payload(extension["extension"])) != EXPECTED_A11_EXTENSION_SHA256:
        raise RevisedElevenFrameRationalCounterexampleError("a11 extension reconstruction failed")
    extensions["height_one_minus_1_1"] = extension
    return extensions


def _bounded_preserving_envelope(
    target: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> dict[str, Any]:
    prior_points = _known_a11_envelope()["prior_points"] + (_known_a11_envelope()["target"],)
    ranks: dict[int, dict[str, int]] = {}
    for degree in (0, 2, 4):
        monomials = common._homogeneous_monomials(degree)
        matrix = common._evaluation_matrix(prior_points, monomials)
        target_row = common._evaluation_matrix((target,), monomials)
        ranks[degree] = {
            "monomial_dimension": len(monomials),
            "prior_zero_constraint_rank": matrix.rank(),
            "prior_zero_nullity": len(monomials) - matrix.rank(),
            "zero_plus_normalization_rank": matrix.col_join(target_row).rank(),
        }
    monomials = common._homogeneous_monomials(4)
    matrix = common._evaluation_matrix(prior_points, monomials)
    target_row = common._evaluation_matrix((target,), monomials)
    supports_checked: dict[int, int] = {}
    feasible_counts: dict[int, int] = {}
    feasible: list[tuple[tuple[int, ...], sp.Matrix]] = []
    for size in range(1, 15):
        supports = tuple(itertools.combinations(range(len(monomials)), size))
        supports_checked[size] = len(supports)
        rows: list[tuple[tuple[int, ...], sp.Matrix]] = []
        for support in supports:
            restricted = matrix[:, support]
            restricted_target = target_row[:, support]
            for vector in restricted.nullspace():
                value = (restricted_target * vector)[0]
                if value != 0:
                    rows.append((support, (vector / value).applyfunc(sp.factor)))
        feasible_counts[size] = len(rows)
        if rows:
            feasible = rows
            break
    if not feasible:
        return {
            "record": {
                "declared_class": "even_homogeneous_scalar_envelopes_times_transverse_curl_blocks",
                "prior_zero_constraints": 11,
                "target_normalizations": 1,
                "degree_zero": ranks[0],
                "degree_two": ranks[2],
                "degree_four": ranks[4],
                "support_search_maximum": 14,
                "supports_checked_by_size": {
                    str(key): value for key, value in supports_checked.items()
                },
                "feasible_envelopes_by_support_size": {
                    str(key): value for key, value in feasible_counts.items()
                },
                "total_supports_checked": sum(supports_checked.values()),
                "bounded_class_exhausted": True,
                "repair_constructed": False,
                "zero_constraints_cover_all_eleven_predecessor_frames": True,
            },
        }
    support, coefficients = feasible[0]
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    envelope = sp.factor(
        sum(
            coefficient
            * n1 ** monomials[index][0]
            * n2 ** monomials[index][1]
            * n3 ** monomials[index][2]
            for index, coefficient in zip(support, coefficients, strict=True)
        )
    )
    variables = (n1, n2, n3)
    if (
        any(envelope.subs(dict(zip(variables, point, strict=True))) != 0 for point in prior_points)
        or envelope.subs(dict(zip(variables, target, strict=True))) != 1
    ):
        raise RevisedElevenFrameRationalCounterexampleError(
            "deterministic preserving envelope mismatch"
        )
    formula = (
        sp.sstr(sp.expand(envelope))
        .replace("n_1", "n1")
        .replace("n_2", "n2")
        .replace("n_3", "n3")
    )
    return {
        "variables": variables,
        "envelope": envelope,
        "prior_points": prior_points,
        "target": target,
        "record": {
            "declared_class": "even_homogeneous_scalar_envelopes_times_transverse_curl_blocks",
            "prior_zero_constraints": 11,
            "target_normalizations": 1,
            "degree_zero": ranks[0],
            "degree_two": ranks[2],
            "degree_four": ranks[4],
            "minimal_even_homogeneous_degree": 4,
            "degree_four_normalized_affine_dimension": ranks[4]["prior_zero_nullity"] - 1,
            "support_search_maximum": 14,
            "supports_checked_by_size": {str(key): value for key, value in supports_checked.items()},
            "feasible_envelopes_by_support_size": {str(key): value for key, value in feasible_counts.items()},
            "total_supports_checked": sum(supports_checked.values()),
            "sparsest_support_size": len(support),
            "sparsest_support_feasible_envelopes": len(feasible),
            "deterministic_envelope_support_indices": list(support),
            "deterministic_envelope": f"a12(n)={formula}",
            "value_at_counterexample": "1",
            "zero_at_all_eleven_predecessor_frames": True,
        },
    }


def _bounded_transverse_curl_decompose(
    target_coordinates: sp.Matrix,
    selector_coordinates: sp.Matrix,
    covector_basis: sp.Matrix,
    reference: Mapping[str, Any],
) -> list[dict[str, Any]]:
    residual = target_coordinates
    terms: list[dict[str, Any]] = []
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
        residual = (
            residual - output_coordinates * right_column.T + right_column * output_coordinates.T
        ).applyfunc(sp.factor)
        coefficients = _solve_vector_system(selector_coordinates, right_column)
        right_covector = (
            sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)] * coefficients
        ).applyfunc(sp.factor)
        energy_output = (covector_basis * output_coordinates).applyfunc(sp.factor)
        output = (reference["energy0"].inv() * energy_output).applyfunc(sp.factor)
        terms.append(
            {
                "coordinate_pair": [left, right],
                "output_aligned": output,
                "right_aligned": right_covector,
            }
        )
    if not terms:
        raise RevisedElevenFrameRationalCounterexampleError(
            "bounded transverse-curl decomposition was empty"
        )
    return terms


def _exact_result(*artifacts: Mapping[str, Any]) -> dict[str, Any]:
    (
        eleven_frame_predecessor,
        ten_frame_predecessor,
        nine_frame_predecessor,
        revised_predecessor,
        degree_five_predecessor,
        rational_predecessor,
        xyz_predecessor,
        c23_predecessor,
        minimal,
        fourth,
    ) = artifacts
    atlas = rational_campaign._atlas()
    rotation, direction, payload, evaluations = common._frame_payload(
        atlas,
        (sp.Integer(-1), sp.Integer(-1)),
        "primary_stereographic_minus_1_minus_1_revised_eleven_frame",
        fourth,
    )
    if direction != [sp.Rational(-1, 3), sp.Rational(-2, 3), sp.Rational(-2, 3)] or evaluations != 15:
        raise RevisedElevenFrameRationalCounterexampleError("next signed height-one point mismatch")
    extensions = _capture_eleven_frame_extensions(
        eleven_frame_predecessor,
        ten_frame_predecessor,
        nine_frame_predecessor,
        revised_predecessor,
        degree_five_predecessor,
        rational_predecessor,
        xyz_predecessor,
        c23_predecessor,
        minimal,
        fourth,
    )
    base = xyz_campaign._prior_symbol_at(direction)["combined"]
    blocks = {
        key: common._evaluate_extension(value, direction) for key, value in extensions.items()
    }
    global_symbol = (base + sum(blocks.values(), sp.zeros(STATE_DIMENSION))).applyfunc(sp.factor)
    state_rotation, _ = _state_rotation(rotation)
    aligned = (state_rotation.T * global_symbol * state_rotation).applyfunc(sp.factor)
    reference = _reference_and_first_jet_packet()
    skew = (reference["energy0"] * aligned - aligned.T * reference["energy0"]).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    projector0 = reference["projectors"][sp.S.Zero]
    rows, payloads, targets = [], [], []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                symbols.get("alpha", sp.Symbol("alpha")): sp.sympify(candidate["a10"]),
                symbols.get("c20", sp.Symbol("c20")): sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        residual = (candidate_rhs + eta * skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(residual)
        compression = (projector0.T * residual * projector0).applyfunc(sp.factor)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
                "zero_speed_cleared_numerator": common._cleared_compression(compression),
            }
        )
        payloads.append((candidate["candidate_id"], residual, eta))
        targets.append((compression / eta).applyfunc(sp.factor))
    target_hashes = {_content_hash(_matrix_payload(target)) for target in targets}
    if (
        len(rows) != EXPECTED_CANDIDATES
        or any(row["D4_Sylvester_solvable"] for row in rows)
        or any(set(row["nonzero_equal_eigenspace_compressions"]) != {"0"} for row in rows)
        or len(target_hashes) != 1
    ):
        raise RevisedElevenFrameRationalCounterexampleError("eleven-frame obstruction mismatch")
    target = targets[0]
    _covectors, coordinates, target_coordinates = _zero_speed_coordinates(projector0, -target)
    selector = (
        coordinates * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    quotient = sp.Matrix.hstack(*selector.T.nullspace()).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    selector_basis = sp.Matrix.hstack(*selector.columnspace())
    intersection = selector.rank() + target_plane.rank() - selector_basis.row_join(target_plane).rank()
    envelope = _bounded_preserving_envelope(tuple(direction))
    if (
        not quotient_target.is_zero_matrix
        or intersection != target.rank()
        or envelope["record"].get("bounded_class_exhausted") is not True
    ):
        raise RevisedElevenFrameRationalCounterexampleError(
            "bounded eleven-frame repair exhaustion mismatch"
        )
    return {
        "atlas": atlas["record"],
        "search_protocol": {
            "chart_order": ["primary_e1_stereographic", "antipodal_e1_stereographic"],
            "deterministic_point_rule": "evaluate final preregistered signed height-one point",
            "bounded_selector": [
                {"chart": "primary_e1_stereographic", "coordinates": ["-1", "-1"]}
            ],
            "preregistered_points": [
                {"chart": "primary_e1_stereographic", "coordinates": ["-1", "-1"]}
            ],
            "points_evaluated": 1,
            "stopped_at_first_regular_obstruction": True,
            "remaining_bounded_selector_points_after_obstruction": 0,
            "preregistered_signed_height_one_selector_exhausted": True,
            "antipodal_chart_points_evaluated_after_obstruction": 0,
        },
        "first_obstruction": {
            "selector": {
                "chart": "primary_e1_stereographic",
                "chart_coordinates": ["-1", "-1"],
                "chart_denominator_value": "3",
                "direction": [str(value) for value in direction],
                "frame_name": "primary_stereographic_minus_1_minus_1_revised_eleven_frame",
                "regular_real_chart_point": True,
            },
            "full_recurrence": {
                "orders_checked": [1, 2, 3, 4],
                "lower_orders_certified_per_polarization_direction": [1, 2, 3],
                "directional_polarization_evaluations": evaluations,
                "candidate_fourth_order_systems": len(rows),
                "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
                "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
                "base_symbol_sha256": _content_hash(_matrix_payload(base)),
                "extension_block_sha256": {
                    key: _content_hash(_matrix_payload(value)) for key, value in blocks.items()
                },
                "revised_global_symbol_rank": global_symbol.rank(),
                "revised_global_symbol_sha256": _content_hash(_matrix_payload(global_symbol)),
                "revised_aligned_symbol_sha256": _content_hash(_matrix_payload(aligned)),
            },
            "exact_rational_obstruction": {
                "candidate_conditions_checked": len(rows),
                "candidate_compatibilities": 0,
                "candidate_obstructions": len(rows),
                "nonzero_equal_eigenspace_compressions": len(rows),
                "distinct_eta_normalized_targets": len(target_hashes),
                "eta_normalized_target_rank": target.rank(),
                "eta_normalized_target_nonzero_entries": sum(value != 0 for value in target),
                "eta_normalized_target_sha256": next(iter(target_hashes)),
                "candidate_records": rows,
                "revised_eleven_frame_full_sphere_D4_compatibility_disproved": True,
            },
        },
        "bounded_next_escape": {
            "exact_range_classification": {
                "transverse_selector_rank": selector.rank(),
                "selector_sha256": _content_hash(_matrix_payload(selector)),
                "target_plane_dimension": target_plane.rank(),
                "selector_target_plane_intersection_dimension": intersection,
                "quotient_target_zero": quotient_target.is_zero_matrix,
                "quotient_target_sha256": _content_hash(_matrix_payload(quotient_target)),
                "target_in_full_transverse_curl_range": True,
            },
            "minimal_preserving_envelope": envelope["record"],
            "bounded_repair_classification": {
                "transverse_curl_target_range_compatible": True,
                "preserving_envelope_exists_in_declared_class": False,
                "twelfth_local_certificate_constructed": False,
                "prior_eleven_direction_certificates_unchanged": True,
                "total_local_direction_certificates": 11,
            },
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    expected_schedule = [
        {"chart": "primary_e1_stereographic", "coordinates": ["-1", "-1"]}
    ]
    expected_selector = expected_schedule
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("active_indices") != [0, 2, 3, 9]
        or config.get("obligation_offset") != 244
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
        or config.get("search_schedule") != expected_schedule
        or config.get("bounded_selector") != expected_selector
        or config.get("bounded_envelope_degrees") != [0, 2, 4]
        or config.get("bounded_envelope_max_support") != 14
    ):
        raise RevisedElevenFrameRationalCounterexampleError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
        "eleven_frame_predecessor",
        "ten_frame_predecessor",
        "nine_frame_predecessor",
        "revised_predecessor",
        "degree_five_predecessor",
        "rational_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    )
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_result(*(bound[key] for key in bound_keys))
    obstruction = exact["first_obstruction"]["exact_rational_obstruction"]
    envelope = exact["bounded_next_escape"]["minimal_preserving_envelope"]
    repair = exact["bounded_next_escape"]["bounded_repair_classification"]
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key] for key in (*bound_keys, "campaign_source", "campaign_test")
        },
        "counts": {
            "bound_predecessors": len(bound_keys),
            "rational_SO3_charts": 2,
            "preregistered_search_points": 1,
            "bounded_selector_points": 1,
            "regular_search_points_evaluated": 1,
            "directional_polarization_evaluations": 15,
            "recurrence_orders_checked": 4,
            "candidate_conditions_checked": obstruction["candidate_conditions_checked"],
            "candidate_compatibilities": obstruction["candidate_compatibilities"],
            "candidate_obstructions": obstruction["candidate_obstructions"],
            "preserved_direction_constraints": 11,
            "bounded_envelope_degrees_checked": 3,
            "bounded_envelope_supports_checked": envelope["total_supports_checked"],
            "feasible_preserving_envelopes": 0,
            "new_local_direction_certificates": 0,
            "total_local_direction_certificates": repair["total_local_direction_certificates"],
            "negative_controls": 10,
            "inferred_global_passes": 0,
        },
        "exact_gate": exact,
        "claims": {key: True for key in TRUE_CLAIMS} | {key: False for key in FALSE_CLAIMS},
        "negative_controls": {
            "continue_search_after_first_regular_obstruction": {"rejected": True},
            "infer_finite_determining_theorem": {"rejected": True},
            "infer_full_sphere_from_twelve_directions": {"rejected": True},
            "ignore_zero_speed_compression": {"rejected": True},
            "skip_lower_recurrence_orders": {"rejected": True},
            "skip_antipodal_atlas_chart": {"rejected": True},
            "skip_exact_denominator_clearing": {"rejected": True},
            "search_envelopes_beyond_preregistered_bound": {"rejected": True},
            "infer_PDE_or_tube_admission": {"rejected": True},
            "infer_B7_H7_or_lifespan": {"rejected": True},
        },
        "scope": (
            "Exact fail-fast rational audit of the revised eleven-frame symbol at the next signed "
            "height-one point (u,v)=(-1,-1). All 12 candidates obstruct before a bounded degree-"
            "0/2/4, support-at-most-fourteen preserving-envelope class is exhausted, so no "
            "twelfth local certificate is constructed. No full-sphere, finite determining, PDE, "
            "tube, CK, TC2, B7, H7, or "
            "lifespan claim follows."
        ),
        "next_gate": (
            "The preregistered signed height-one selector is exhausted; preregister the next "
            "bounded exact rational selector for the revised twelve-frame symbol or prove a finite "
            "determining theorem before any PDE/global admission."
        ),
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    gate = document.get("exact_gate", {})
    first = gate.get("first_obstruction", {})
    obstruction = first.get("exact_rational_obstruction", {})
    bounded = gate.get("bounded_next_escape", {})
    envelope = bounded.get("minimal_preserving_envelope", {})
    repair = bounded.get("bounded_repair_classification", {})
    rows = obstruction.get("candidate_records", [])
    claims = document.get("claims", {})
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or set(document)
        != {
            "claims", "config_sha256", "content_sha256", "counts", "errors",
            "exact_gate", "negative_controls", "next_gate", "schema_version", "scope",
            "source_bindings", "status",
        }
        or document.get("errors") != []
        or document.get("counts", {}).get("bound_predecessors") != 10
        or document.get("counts", {}).get("bounded_selector_points") != 1
        or document.get("counts", {}).get("preserved_direction_constraints") != 11
        or document.get("counts", {}).get("total_local_direction_certificates") != 11
        or document.get("counts", {}).get("candidate_conditions_checked") != 12
        or document.get("counts", {}).get("candidate_obstructions") != 12
        or document.get("counts", {}).get("candidate_compatibilities") != 0
        or document.get("counts", {}).get("new_local_direction_certificates") != 0
        or document.get("counts", {}).get("feasible_preserving_envelopes") != 0
        or document.get("counts", {}).get("inferred_global_passes") != 0
        or (EXPECTED_EXACT_GATE_SHA256 and _content_hash(gate) != EXPECTED_EXACT_GATE_SHA256)
        or gate.get("atlas", {}).get("union_covers_real_S2") is not True
        or gate.get("search_protocol", {}).get("points_evaluated") != 1
        or gate.get("search_protocol", {}).get("remaining_bounded_selector_points_after_obstruction") != 0
        or gate.get("search_protocol", {}).get("preregistered_signed_height_one_selector_exhausted") is not True
        or first.get("selector", {}).get("chart_coordinates") != ["-1", "-1"]
        or first.get("selector", {}).get("direction") != ["-1/3", "-2/3", "-2/3"]
        or first.get("full_recurrence", {}).get("orders_checked") != [1, 2, 3, 4]
        or first.get("full_recurrence", {}).get("directional_polarization_evaluations") != 15
        or obstruction.get("candidate_compatibilities") != 0
        or obstruction.get("candidate_obstructions") != 12
        or obstruction.get("distinct_eta_normalized_targets") != 1
        or obstruction.get("eta_normalized_target_rank") != 2
        or obstruction.get("eta_normalized_target_nonzero_entries") != 16
        or obstruction.get("eta_normalized_target_sha256")
        != "c6f8ec3483c663fb9619cdd50dec1f61881932c65eb25a72a6bd24ef027a19d4"
        or len(rows) != 12
        or any(
            row.get("D4_Sylvester_solvable") is not False
            or set(row.get("nonzero_equal_eigenspace_compressions", {})) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"].get("rank") <= 0
            or row.get("zero_speed_cleared_numerator", {}).get("numerator_rank") <= 0
            for row in rows
        )
        or bounded.get("exact_range_classification", {}).get("quotient_target_zero") is not True
        or bounded.get("exact_range_classification", {}).get("target_in_full_transverse_curl_range") is not True
        or bounded.get("exact_range_classification", {}).get("target_plane_dimension") != 2
        or bounded.get("exact_range_classification", {}).get("selector_target_plane_intersection_dimension") != 2
        or envelope.get("support_search_maximum") != 14
        or envelope.get("total_supports_checked") != 32766
        or envelope.get("degree_four", {}).get("prior_zero_constraint_rank") != 11
        or envelope.get("degree_four", {}).get("prior_zero_nullity") != 4
        or envelope.get("degree_four", {}).get("zero_plus_normalization_rank") != 11
        or envelope.get("bounded_class_exhausted") is not True
        or envelope.get("repair_constructed") is not False
        or envelope.get("zero_constraints_cover_all_eleven_predecessor_frames") is not True
        or any(value != 0 for value in envelope.get("feasible_envelopes_by_support_size", {}).values())
        or repair.get("transverse_curl_target_range_compatible") is not True
        or repair.get("preserving_envelope_exists_in_declared_class") is not False
        or repair.get("twelfth_local_certificate_constructed") is not False
        or repair.get("prior_eleven_direction_certificates_unchanged") is not True
        or repair.get("total_local_direction_certificates") != 11
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or len(document.get("negative_controls", {})) != 10
    ):
        raise RevisedElevenFrameRationalCounterexampleError(
            "revised eleven-frame rational campaign validation failed"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.project_root.resolve()
    artifact = build_campaign(root, (root / args.config).resolve())
    validate_campaign(artifact)
    _atomic_write((root / args.output).resolve(), _json_bytes(artifact))
    print(json.dumps({"status": artifact["status"], "content_sha256": artifact["content_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
