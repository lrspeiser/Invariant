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
from . import quartic_tc2_d4_revised_nine_frame_rational_counterexample_campaign as prior
from .quartic_tc2_d4_degree_three_sixth_frame_completion_campaign import _decompose
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

SCHEMA = "sigma-quartic-tc2-d4-revised-ten-frame-rational-counterexample-campaign-1.0"
CONFIG_SCHEMA = (
    "sigma-quartic-tc2-d4-revised-ten-frame-rational-counterexample-config-1.0"
)
STATUS = "pass_exact_ten_frame_rational_counterexample_and_bounded_escape"
EXPECTED_CANDIDATES = 12
EXPECTED_PREDECESSOR_CONTENT_SHA256 = (
    "e556ef1716683882598e4d9a4b9ea550ca1d8a7945b0d96913ab84a0da3b57d5"
)
EXPECTED_PREDECESSOR_EXACT_SHA256 = (
    "d393b2c2f9968fcea094cd1125d5124aa78a7dd5a89339afabd1f9e91c545d4c"
)
EXPECTED_A10_EXTENSION_SHA256 = (
    "7d8d86c1197c143cab42fca5b85aa339b9212b454a240372f64f5f8f36da3367"
)
EXPECTED_A10_TARGET_SHA256 = (
    "406cb87033eb0946d053c46d076a5e97fb9157c848e3737950c24ba7fca0369f"
)
EXPECTED_EXACT_GATE_SHA256 = "172cf7a3d659fad3158e6ecc0f0685c5d99a091e486fc29f80e7fbb33ad015ad"

TRUE_CLAIMS = {
    "all_12_candidates_closed_by_bounded_local_escape",
    "all_12_candidates_obstructed_at_first_ten_frame_search_point",
    "exact_regular_rational_counterexample_proved_for_ten_frame_symbol",
    "exact_two_chart_SO3_atlas_reused",
    "first_obstruction_is_zero_speed_only",
    "full_orders_one_through_four_recurrence_evaluated",
    "revised_ten_frame_symbol_full_sphere_D4_compatibility_disproved",
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
    "revised_eleven_frame_symbol_full_sphere_D4_compatibility_proved",
    "variable_coefficient_constraint_calculus_proved",
}


class RevisedTenFrameRationalCounterexampleError(ValueError):
    """Raised when the ten-frame rational certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RevisedTenFrameRationalCounterexampleError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RevisedTenFrameRationalCounterexampleError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RevisedTenFrameRationalCounterexampleError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RevisedTenFrameRationalCounterexampleError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _known_a10_envelope() -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    envelope = (
        sp.Rational(3, 8) * n1**2 * n2 * n3
        - sp.Rational(21, 16) * n2**3 * n3
        + sp.Rational(81, 32) * n2**2 * n3**2
        - sp.Rational(21, 16) * n2 * n3**3
    )
    prior_points = prior._known_a9_envelope()["prior_points"] + (
        prior._known_a9_envelope()["target"],
    )
    return {
        "variables": (n1, n2, n3),
        "envelope": envelope,
        "prior_points": prior_points,
        "target": (sp.Rational(-1, 3), sp.Rational(2, 3), sp.Rational(-2, 3)),
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
            (prior.prior._evaluate_extension(value, direction) for value in extensions.values()),
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
    terms = _decompose(target_coordinates, selector, covectors, reference)
    return degree_five._angular_extension(terms, state_rotation, tuple(direction), envelope)


def _capture_ten_frame_extensions(
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
        ten_frame_predecessor.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or _content_hash(ten_frame_predecessor.get("exact_gate", {}))
        != EXPECTED_PREDECESSOR_EXACT_SHA256
    ):
        raise RevisedTenFrameRationalCounterexampleError(
            "ten-frame predecessor certificate mismatch"
        )
    extensions = prior._capture_nine_frame_extensions(
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
    rotation, direction, payload, evaluations = prior.prior._frame_payload(
        atlas,
        (sp.Integer(1), sp.Integer(-1)),
        "reconstructed_height_one_1_minus_1",
        fourth,
    )
    if direction != [sp.Rational(-1, 3), sp.Rational(2, 3), sp.Rational(-2, 3)] or evaluations != 15:
        raise RevisedTenFrameRationalCounterexampleError("a10 reconstruction frame mismatch")
    extension = _reconstruct_extension(
        rotation, direction, payload, extensions, _known_a10_envelope(), minimal
    )
    if _content_hash(_matrix_payload(extension["extension"])) != EXPECTED_A10_EXTENSION_SHA256:
        raise RevisedTenFrameRationalCounterexampleError("a10 extension reconstruction failed")
    extensions["height_one_1_minus_1"] = extension
    return extensions


def _bounded_preserving_envelope(
    target: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> dict[str, Any]:
    prior_points = _known_a10_envelope()["prior_points"] + (_known_a10_envelope()["target"],)
    ranks: dict[int, dict[str, int]] = {}
    for degree in (0, 2, 4):
        monomials = prior.prior._homogeneous_monomials(degree)
        matrix = prior.prior._evaluation_matrix(prior_points, monomials)
        target_row = prior.prior._evaluation_matrix((target,), monomials)
        ranks[degree] = {
            "monomial_dimension": len(monomials),
            "prior_zero_constraint_rank": matrix.rank(),
            "prior_zero_nullity": len(monomials) - matrix.rank(),
            "zero_plus_normalization_rank": matrix.col_join(target_row).rank(),
        }
    monomials = prior.prior._homogeneous_monomials(4)
    matrix = prior.prior._evaluation_matrix(prior_points, monomials)
    target_row = prior.prior._evaluation_matrix((target,), monomials)
    supports_checked: dict[int, int] = {}
    feasible_counts: dict[int, int] = {}
    feasible: list[tuple[tuple[int, ...], sp.Matrix]] = []
    for size in range(1, 13):
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
        raise RevisedTenFrameRationalCounterexampleError(
            "bounded degree-four support-at-most-twelve envelope class exhausted"
        )
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
        raise RevisedTenFrameRationalCounterexampleError(
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
            "prior_zero_constraints": 10,
            "target_normalizations": 1,
            "degree_zero": ranks[0],
            "degree_two": ranks[2],
            "degree_four": ranks[4],
            "minimal_even_homogeneous_degree": 4,
            "degree_four_normalized_affine_dimension": ranks[4]["prior_zero_nullity"] - 1,
            "support_search_maximum": 12,
            "supports_checked_by_size": {str(key): value for key, value in supports_checked.items()},
            "feasible_envelopes_by_support_size": {str(key): value for key, value in feasible_counts.items()},
            "total_supports_checked": sum(supports_checked.values()),
            "sparsest_support_size": len(support),
            "sparsest_support_feasible_envelopes": len(feasible),
            "deterministic_envelope_support_indices": list(support),
            "deterministic_envelope": f"a11(n)={formula}",
            "value_at_counterexample": "1",
            "zero_at_all_ten_predecessor_frames": True,
        },
    }


def _exact_result(*artifacts: Mapping[str, Any]) -> dict[str, Any]:
    (
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
    rotation, direction, payload, evaluations = prior.prior._frame_payload(
        atlas,
        (sp.Integer(-1), sp.Integer(1)),
        "primary_stereographic_minus_1_1_revised_ten_frame",
        fourth,
    )
    if direction != [sp.Rational(-1, 3), sp.Rational(-2, 3), sp.Rational(2, 3)] or evaluations != 15:
        raise RevisedTenFrameRationalCounterexampleError("next signed height-one point mismatch")
    extensions = _capture_ten_frame_extensions(
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
        key: prior.prior._evaluate_extension(value, direction) for key, value in extensions.items()
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
                "zero_speed_cleared_numerator": prior.prior._cleared_compression(compression),
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
        raise RevisedTenFrameRationalCounterexampleError("ten-frame obstruction mismatch")
    target = targets[0]
    covectors, coordinates, target_coordinates = _zero_speed_coordinates(projector0, -target)
    selector = (
        coordinates * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    quotient = sp.Matrix.hstack(*selector.T.nullspace()).T
    quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
    target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
    selector_basis = sp.Matrix.hstack(*selector.columnspace())
    intersection = selector.rank() + target_plane.rank() - selector_basis.row_join(target_plane).rank()
    terms = _decompose(target_coordinates, selector, covectors, reference)
    aligned_block = sum(
        (term["output_aligned"] * term["right_aligned"].T for term in terms),
        sp.zeros(STATE_DIMENSION),
    ).applyfunc(sp.factor)
    global_block = (state_rotation * aligned_block * state_rotation.T).applyfunc(sp.factor)
    envelope = _bounded_preserving_envelope(tuple(direction))
    angular = degree_five._angular_extension(terms, state_rotation, tuple(direction), envelope)
    correction_skew = (
        reference["energy0"] * aligned_block - aligned_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    corrected_rows = []
    for candidate_id, before, eta in payloads:
        corrected = (before + eta * correction_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        corrected_rows.append(
            {
                "candidate_id": candidate_id,
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
                "corrected_residual_sha256": _content_hash(_matrix_payload(corrected)),
            }
        )
    if (
        angular["reference_block"] != global_block
        or not quotient_target.is_zero_matrix
        or intersection != target.rank()
        or aligned_block.rank() != 2
        or len(terms) != 2
        or any(
            not row["D4_Sylvester_solvable"]
            or row["nonzero_equal_eigenspace_compressions"] != {}
            for row in corrected_rows
        )
    ):
        raise RevisedTenFrameRationalCounterexampleError("bounded eleventh-frame escape mismatch")
    return {
        "atlas": atlas["record"],
        "search_protocol": {
            "chart_order": ["primary_e1_stereographic", "antipodal_e1_stereographic"],
            "deterministic_point_rule": (
                "continue signed height-one order after certified (1,1) and (1,-1) points"
            ),
            "bounded_selector": [
                {"chart": "primary_e1_stereographic", "coordinates": ["-1", "1"]},
                {"chart": "primary_e1_stereographic", "coordinates": ["-1", "-1"]},
            ],
            "preregistered_points": [
                {"chart": "primary_e1_stereographic", "coordinates": ["-1", "1"]}
            ],
            "points_evaluated": 1,
            "stopped_at_first_regular_obstruction": True,
            "remaining_bounded_selector_points_after_obstruction": 1,
            "antipodal_chart_points_evaluated_after_obstruction": 0,
        },
        "first_obstruction": {
            "selector": {
                "chart": "primary_e1_stereographic",
                "chart_coordinates": ["-1", "1"],
                "chart_denominator_value": "3",
                "direction": [str(value) for value in direction],
                "frame_name": "primary_stereographic_minus_1_1_revised_ten_frame",
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
                "revised_ten_frame_full_sphere_D4_compatibility_disproved": True,
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
            "local_completion": {
                "constructed_completion_rank": aligned_block.rank(),
                "elementary_curl_channels": len(terms),
                "coordinate_pairs": [term["coordinate_pair"] for term in terms],
                "aligned_block_sha256": _content_hash(_matrix_payload(aligned_block)),
                "global_block_sha256": _content_hash(_matrix_payload(global_block)),
                "extension_sha256": _content_hash(_matrix_payload(angular["extension"])),
                "gradient_residual_zero": angular["gradient_residual"].is_zero_matrix,
                "gradient_residual_sha256": _content_hash(
                    _matrix_payload(angular["gradient_residual"])
                ),
                "candidate_conditions_checked": len(corrected_rows),
                "candidate_compatibilities": sum(
                    row["D4_Sylvester_solvable"] for row in corrected_rows
                ),
                "candidate_obstructions": sum(
                    not row["D4_Sylvester_solvable"] for row in corrected_rows
                ),
                "candidate_records": corrected_rows,
                "prior_ten_direction_certificates_preserved": True,
                "total_local_direction_certificates": 11,
            },
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    expected_schedule = [
        {"chart": "primary_e1_stereographic", "coordinates": ["-1", "1"]}
    ]
    expected_selector = expected_schedule + [
        {"chart": "primary_e1_stereographic", "coordinates": ["-1", "-1"]}
    ]
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
        or config.get("bounded_envelope_max_support") != 12
    ):
        raise RevisedTenFrameRationalCounterexampleError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
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
    completion = exact["bounded_next_escape"]["local_completion"]
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
            "bounded_selector_points": 2,
            "regular_search_points_evaluated": 1,
            "directional_polarization_evaluations": 15,
            "recurrence_orders_checked": 4,
            "candidate_conditions_checked": obstruction["candidate_conditions_checked"],
            "candidate_compatibilities": obstruction["candidate_compatibilities"],
            "candidate_obstructions": obstruction["candidate_obstructions"],
            "preserved_direction_constraints": 10,
            "bounded_envelope_degrees_checked": 3,
            "bounded_envelope_supports_checked": envelope["total_supports_checked"],
            "sparsest_envelope_support": envelope["sparsest_support_size"],
            "sparsest_feasible_envelopes": envelope["sparsest_support_feasible_envelopes"],
            "new_local_candidate_compatibilities": completion["candidate_compatibilities"],
            "new_local_candidate_obstructions": completion["candidate_obstructions"],
            "total_local_direction_certificates": completion["total_local_direction_certificates"],
            "negative_controls": 10,
            "inferred_global_passes": 0,
        },
        "exact_gate": exact,
        "claims": {key: True for key in TRUE_CLAIMS} | {key: False for key in FALSE_CLAIMS},
        "negative_controls": {
            "continue_search_after_first_regular_obstruction": {"rejected": True},
            "infer_finite_determining_theorem": {"rejected": True},
            "infer_full_sphere_from_eleven_directions": {"rejected": True},
            "ignore_zero_speed_compression": {"rejected": True},
            "skip_lower_recurrence_orders": {"rejected": True},
            "skip_antipodal_atlas_chart": {"rejected": True},
            "skip_exact_denominator_clearing": {"rejected": True},
            "search_envelopes_beyond_preregistered_bound": {"rejected": True},
            "infer_PDE_or_tube_admission": {"rejected": True},
            "infer_B7_H7_or_lifespan": {"rejected": True},
        },
        "scope": (
            "Exact fail-fast rational audit of the revised ten-frame symbol at the next signed "
            "height-one point (u,v)=(-1,1). All 12 candidates obstruct before a bounded degree-"
            "0/2/4, support-at-most-twelve preserving repair constructs an eleventh local "
            "certificate. No full-sphere, finite determining, PDE, tube, CK, TC2, B7, H7, or "
            "lifespan claim follows."
        ),
        "next_gate": (
            "Continue the preregistered signed height-one selector at (u,v)=(-1,-1) for the "
            "revised eleven-frame symbol; do not infer a finite determining theorem or PDE/global admission."
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
    completion = bounded.get("local_completion", {})
    rows = obstruction.get("candidate_records", [])
    corrected = completion.get("candidate_records", [])
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
        or document.get("counts")
        != {
            "bound_predecessors": 9,
            "bounded_envelope_degrees_checked": 3,
            "bounded_envelope_supports_checked": 9948,
            "bounded_selector_points": 2,
            "candidate_compatibilities": 0,
            "candidate_conditions_checked": 12,
            "candidate_obstructions": 12,
            "directional_polarization_evaluations": 15,
            "inferred_global_passes": 0,
            "negative_controls": 10,
            "new_local_candidate_compatibilities": 12,
            "new_local_candidate_obstructions": 0,
            "preregistered_search_points": 1,
            "preserved_direction_constraints": 10,
            "rational_SO3_charts": 2,
            "recurrence_orders_checked": 4,
            "regular_search_points_evaluated": 1,
            "sparsest_envelope_support": 6,
            "sparsest_feasible_envelopes": 1,
            "total_local_direction_certificates": 11,
        }
        or _content_hash(gate) != EXPECTED_EXACT_GATE_SHA256
        or gate.get("atlas", {}).get("union_covers_real_S2") is not True
        or gate.get("search_protocol", {}).get("points_evaluated") != 1
        or gate.get("search_protocol", {}).get("remaining_bounded_selector_points_after_obstruction") != 1
        or first.get("selector", {}).get("chart_coordinates") != ["-1", "1"]
        or first.get("selector", {}).get("direction") != ["-1/3", "-2/3", "2/3"]
        or first.get("full_recurrence", {}).get("orders_checked") != [1, 2, 3, 4]
        or first.get("full_recurrence", {}).get("directional_polarization_evaluations") != 15
        or obstruction.get("candidate_compatibilities") != 0
        or obstruction.get("candidate_obstructions") != 12
        or obstruction.get("distinct_eta_normalized_targets") != 1
        or obstruction.get("eta_normalized_target_rank") != 4
        or obstruction.get("eta_normalized_target_nonzero_entries") != 56
        or obstruction.get("eta_normalized_target_sha256")
        != "6f05c62c3a4f8d90b13901a43db4109acac8865477f433d9437c3fea310edb01"
        or len(rows) != 12
        or any(
            row.get("D4_Sylvester_solvable") is not False
            or set(row.get("nonzero_equal_eigenspace_compressions", {})) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"].get("rank") != 4
            or row["nonzero_equal_eigenspace_compressions"]["0"].get("nonzero_entries") != 56
            or row.get("zero_speed_cleared_numerator", {}).get("numerator_rank") != 4
            or row.get("zero_speed_cleared_numerator", {}).get("numerator_nonzero_entries")
            != 56
            for row in rows
        )
        or bounded.get("exact_range_classification")
        != {
            "quotient_target_sha256": (
                "6bd0f4db2919abb53bd3fc437f3ec440b1c2df2a8e73fe17d06d0a3fc1c10f23"
            ),
            "quotient_target_zero": True,
            "selector_sha256": (
                "7ef398226365b9e42bd543a3b9c5b00c82621cbf8f67d76b2768e38e81441d26"
            ),
            "selector_target_plane_intersection_dimension": 4,
            "target_in_full_transverse_curl_range": True,
            "target_plane_dimension": 4,
            "transverse_selector_rank": 22,
        }
        or envelope.get("minimal_even_homogeneous_degree") != 4
        or envelope.get("support_search_maximum") != 12
        or envelope.get("sparsest_support_size") != 6
        or envelope.get("sparsest_support_feasible_envelopes") != 1
        or envelope.get("supports_checked_by_size")
        != {"1": 15, "2": 105, "3": 455, "4": 1365, "5": 3003, "6": 5005}
        or envelope.get("feasible_envelopes_by_support_size")
        != {"1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 1}
        or envelope.get("total_supports_checked") != 9948
        or envelope.get("deterministic_envelope_support_indices") != [1, 2, 3, 6, 7, 10]
        or envelope.get("zero_at_all_ten_predecessor_frames") is not True
        or completion.get("candidate_compatibilities") != 12
        or completion.get("candidate_obstructions") != 0
        or completion.get("constructed_completion_rank") != 2
        or completion.get("elementary_curl_channels") != 2
        or completion.get("coordinate_pairs") != [[11, 21], [15, 32]]
        or completion.get("gradient_residual_zero") is not True
        or completion.get("prior_ten_direction_certificates_preserved") is not True
        or completion.get("total_local_direction_certificates") != 11
        or len(corrected) != 12
        or [row.get("candidate_id") for row in rows]
        != [row.get("candidate_id") for row in corrected]
        or any(
            row.get("D4_Sylvester_solvable") is not True
            or row.get("nonzero_equal_eigenspace_compressions") != {}
            for row in corrected
        )
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or len(document.get("negative_controls", {})) != 10
    ):
        raise RevisedTenFrameRationalCounterexampleError(
            "revised ten-frame rational campaign validation failed"
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
