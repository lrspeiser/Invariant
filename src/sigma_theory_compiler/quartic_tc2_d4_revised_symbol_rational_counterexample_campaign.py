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
from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_degree_three_sixth_frame_completion_campaign import _decompose
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import (
    TRANSVERSE_CURL_INDICES,
)
from .quartic_tc2_d4_parity_cubic_generic_direction_campaign import (
    JET_ORDER,
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

SCHEMA = "sigma-quartic-tc2-d4-revised-symbol-rational-counterexample-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-revised-symbol-rational-counterexample-config-1.0"
STATUS = "pass_exact_revised_symbol_rational_counterexample_and_bounded_escape"
ACTIVE_INDICES = (0, 2, 3, 9)
OBLIGATION_OFFSET = 244
EXPECTED_CANDIDATES = 12
EXPECTED_PREDECESSOR_CONTENT_SHA256 = (
    "eb46a7b76d5f51a62254ecbe00e91c120103c5d81948443c1b9e4da2004414b5"
)
EXPECTED_PREDECESSOR_EXACT_SHA256 = (
    "2bd3b5fcd272de32cd766c7f68b52cb44e5f80bec1d8199a440dcdda87bf7331"
)
EXPECTED_EXACT_GATE_SHA256 = "181ddfcdf1786c20ff6497e3141969f677a2b08cffeaceb6b1e3ca9e3ab36212"

TRUE_CLAIMS = {
    "all_12_candidates_obstructed_at_first_revised_symbol_search_point",
    "all_12_candidates_closed_by_bounded_e3_local_escape",
    "exact_regular_rational_counterexample_proved_for_revised_symbol",
    "exact_two_chart_SO3_atlas_reused",
    "first_obstruction_is_zero_speed_only",
    "full_orders_one_through_four_recurrence_evaluated",
    "minimal_degree_four_seven_frame_preserving_envelope_classified",
    "revised_seven_frame_symbol_full_sphere_D4_compatibility_disproved",
    "sparsest_degree_four_envelope_support_is_five",
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
    "revised_eight_frame_symbol_full_sphere_D4_compatibility_proved",
    "variable_coefficient_constraint_calculus_proved",
}


class RevisedSymbolRationalCounterexampleError(ValueError):
    """Raised when the revised-symbol rational certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RevisedSymbolRationalCounterexampleError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RevisedSymbolRationalCounterexampleError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RevisedSymbolRationalCounterexampleError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RevisedSymbolRationalCounterexampleError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _capture_revised_extensions(
    degree_five_artifact: Mapping[str, Any],
    rational_predecessor: Mapping[str, Any],
    xyz_predecessor: Mapping[str, Any],
    c23_predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Mapping[str, Any]]:
    if (
        degree_five_artifact.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or _content_hash(degree_five_artifact.get("exact_completion", {}))
        != EXPECTED_PREDECESSOR_EXACT_SHA256
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "degree-five predecessor certificate mismatch"
        )
    captured: dict[str, Mapping[str, Any]] = {}
    original = degree_five._angular_extension

    def capture(*args: Any, **kwargs: Any) -> dict[str, Any]:
        result = original(*args, **kwargs)
        captured["degree_five"] = result
        return result

    degree_five._angular_extension = capture
    try:
        replay = degree_five._exact_result(
            rational_predecessor,
            xyz_predecessor,
            c23_predecessor,
            minimal,
            fourth,
        )
    finally:
        degree_five._angular_extension = original
    if (
        set(captured) != {"degree_five"}
        or _content_hash(replay) != EXPECTED_PREDECESSOR_EXACT_SHA256
        or _content_hash(_matrix_payload(captured["degree_five"]["extension"]))
        != degree_five_artifact["exact_completion"]["degree_five_angular_extension"][
            "symbol_sha256"
        ]
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "degree-five angular extension reconstruction failed"
        )
    captured.update(
        rational_campaign._capture_current_extensions(
            xyz_predecessor, c23_predecessor, minimal, fourth
        )
    )
    if set(captured) != {"xyz", "sixth", "degree_five"}:
        raise RevisedSymbolRationalCounterexampleError(
            "revised combined symbol reconstruction failed"
        )
    return captured


def _homogeneous_monomials(degree: int) -> tuple[tuple[int, int, int], ...]:
    return tuple((i, j, degree - i - j) for i in range(degree + 1) for j in range(degree - i + 1))


def _evaluation_matrix(
    points: tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...],
    monomials: tuple[tuple[int, int, int], ...],
) -> sp.Matrix:
    return sp.Matrix([[x**i * y**j * z**k for i, j, k in monomials] for x, y, z in points])


def _bounded_preserving_envelope() -> dict[str, Any]:
    prior_points = (
        (sp.Integer(1), sp.Integer(0), sp.Integer(0)),
        (sp.Integer(0), sp.Integer(1), sp.Integer(0)),
        (sp.Rational(3, 5), sp.Rational(4, 5), sp.Integer(0)),
        (sp.Rational(3, 5), sp.Integer(0), sp.Rational(4, 5)),
        (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)),
        (sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)),
        (sp.Rational(2, 3), sp.Rational(2, 3), sp.Rational(1, 3)),
    )
    target = (sp.Integer(0), sp.Integer(0), sp.Integer(1))
    ranks: dict[int, dict[str, int]] = {}
    for degree in (0, 2, 4):
        monomials = _homogeneous_monomials(degree)
        prior_matrix = _evaluation_matrix(prior_points, monomials)
        target_row = _evaluation_matrix((target,), monomials)
        ranks[degree] = {
            "monomial_dimension": len(monomials),
            "prior_zero_constraint_rank": prior_matrix.rank(),
            "prior_zero_nullity": len(monomials) - prior_matrix.rank(),
            "zero_plus_normalization_rank": prior_matrix.col_join(target_row).rank(),
        }
    monomials = _homogeneous_monomials(4)
    prior_matrix = _evaluation_matrix(prior_points, monomials)
    target_row = _evaluation_matrix((target,), monomials)
    support_counts: dict[int, int] = {}
    feasible: list[tuple[tuple[int, ...], sp.Matrix]] = []
    supports_checked: dict[int, int] = {}
    for support_size in range(1, 6):
        rows = []
        supports = tuple(itertools.combinations(range(len(monomials)), support_size))
        supports_checked[support_size] = len(supports)
        for support in supports:
            restricted = prior_matrix[:, support]
            restricted_target = target_row[:, support]
            for vector in restricted.nullspace():
                value = (restricted_target * vector)[0]
                if value != 0:
                    rows.append((support, (vector / value).applyfunc(sp.factor)))
        support_counts[support_size] = len(rows)
        if rows:
            feasible = rows
            break
    if (
        ranks
        != {
            0: {
                "monomial_dimension": 1,
                "prior_zero_constraint_rank": 1,
                "prior_zero_nullity": 0,
                "zero_plus_normalization_rank": 1,
            },
            2: {
                "monomial_dimension": 6,
                "prior_zero_constraint_rank": 6,
                "prior_zero_nullity": 0,
                "zero_plus_normalization_rank": 6,
            },
            4: {
                "monomial_dimension": 15,
                "prior_zero_constraint_rank": 7,
                "prior_zero_nullity": 8,
                "zero_plus_normalization_rank": 8,
            },
        }
        or support_counts != {1: 0, 2: 0, 3: 0, 4: 0, 5: 110}
        or supports_checked != {1: 15, 2: 105, 3: 455, 4: 1365, 5: 3003}
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "bounded envelope rank or sparsity classification mismatch"
        )
    support, coefficients = feasible[0]
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    variables = (n1, n2, n3)
    envelope = sp.factor(
        sum(
            coefficient
            * n1 ** monomials[index][0]
            * n2 ** monomials[index][1]
            * n3 ** monomials[index][2]
            for index, coefficient in zip(support, coefficients, strict=True)
        )
    )
    expected = (
        -sp.Rational(1, 18)
        * n3
        * (24 * n1 * n3**2 - 38 * n2**3 + 93 * n2**2 * n3 - 49 * n2 * n3**2 - 18 * n3**3)
    )
    prior_substitutions = [dict(zip(variables, point, strict=True)) for point in prior_points]
    target_substitution = dict(zip(variables, target, strict=True))
    if (
        support != (0, 1, 2, 3, 5)
        or envelope != expected
        or any(envelope.subs(point) != 0 for point in prior_substitutions)
        or envelope.subs(target_substitution) != 1
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "deterministic sparsest preserving envelope mismatch"
        )
    return {
        "variables": variables,
        "envelope": envelope,
        "prior_points": prior_points,
        "target": target,
        "record": {
            "declared_class": ("even_homogeneous_scalar_envelopes_times_transverse_curl_blocks"),
            "prior_zero_constraints": 7,
            "target_normalizations": 1,
            "degree_zero": ranks[0],
            "degree_two": ranks[2],
            "degree_four": ranks[4],
            "minimal_even_homogeneous_degree": 4,
            "degree_four_normalized_affine_dimension": 7,
            "support_search_maximum": 5,
            "supports_checked_by_size": {
                str(key): value for key, value in supports_checked.items()
            },
            "feasible_envelopes_by_support_size": {
                str(key): value for key, value in support_counts.items()
            },
            "total_supports_checked": sum(supports_checked.values()),
            "sparsest_support_size": 5,
            "sparsest_support_feasible_envelopes": len(feasible),
            "deterministic_envelope_support_indices": list(support),
            "deterministic_envelope": (
                "a8(n)=n3^4+(49/18)n2*n3^3-(31/6)n2^2*n3^2+(19/9)n2^3*n3-(4/3)n1*n3^3"
            ),
            "value_at_counterexample": "1",
            "zero_at_all_seven_predecessor_frames": True,
        },
    }


def _evaluate_extension(angular: Mapping[str, Any], direction: list[sp.Expr]) -> sp.Matrix:
    return (
        angular["extension"]
        .subs(dict(zip(angular["variables"], direction, strict=True)))
        .applyfunc(sp.factor)
    )


def _cleared_compression(matrix: sp.Matrix) -> dict[str, Any]:
    nonzero = [sp.together(value) for value in matrix if value != 0]
    denominators = [sp.denom(value) for value in nonzero]
    clearing = sp.lcm(denominators) if denominators else sp.Integer(1)
    numerator = (clearing * matrix).applyfunc(sp.expand)
    if any(sp.denom(value) != 1 for value in numerator):
        raise RevisedSymbolRationalCounterexampleError(
            "point compression denominator clearing failed"
        )
    return {
        "clearing_denominator": str(clearing),
        "numerator_polynomial_total_degree_uv": 0,
        "numerator_nonzero_entries": sum(value != 0 for value in numerator),
        "numerator_rank": numerator.rank(),
        "numerator_sha256": _content_hash(_matrix_payload(numerator)),
    }


def _exact_result(
    degree_five_artifact: Mapping[str, Any],
    rational_predecessor: Mapping[str, Any],
    xyz_predecessor: Mapping[str, Any],
    c23_predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Any]:
    atlas = rational_campaign._atlas()
    u, v = atlas["variables"]
    point = {u: sp.Integer(0), v: sp.Integer(1)}
    rotation3 = atlas["primary"].subs(point)
    direction = list(rotation3[:, 0])
    if direction != [sp.Integer(0), sp.Integer(0), sp.Integer(1)]:
        raise RevisedSymbolRationalCounterexampleError(
            "first bounded rational search point mismatch"
        )
    frame = {
        "name": "primary_stereographic_0_1_revised_seven_frame",
        "rotation": rotation3,
        "direction": tuple(direction),
    }
    extensions = _capture_revised_extensions(
        degree_five_artifact,
        rational_predecessor,
        xyz_predecessor,
        c23_predecessor,
        minimal,
        fourth,
    )
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    if evaluations != 15:
        raise RevisedSymbolRationalCounterexampleError(
            "full fourth-order polarization count mismatch"
        )
    base = xyz_campaign._prior_symbol_at(direction)["combined"]
    xyz_block = _evaluate_extension(extensions["xyz"], direction)
    sixth_block = _evaluate_extension(extensions["sixth"], direction)
    degree_five_block = _evaluate_extension(extensions["degree_five"], direction)
    revised_global = (base + xyz_block + sixth_block + degree_five_block).applyfunc(sp.factor)
    state_rotation, _ = _state_rotation(rotation3)
    revised_aligned = (state_rotation.T * revised_global * state_rotation).applyfunc(sp.factor)
    reference = _reference_and_first_jet_packet()
    revised_skew = (
        reference["energy0"] * revised_aligned - revised_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    rows = []
    payloads = []
    normalized_targets = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {alpha: sp.sympify(candidate["a10"]), c20: sp.sympify(candidate["c20"])}
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        residual = (candidate_rhs + eta * revised_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(residual)
        compression = (projector0.T * residual * projector0).applyfunc(sp.factor)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
                "zero_speed_cleared_numerator": _cleared_compression(compression),
            }
        )
        payloads.append((candidate["candidate_id"], residual, eta))
        normalized_targets.append((compression / eta).applyfunc(sp.factor))
    target_hashes = {_content_hash(_matrix_payload(target)) for target in normalized_targets}
    if (
        len(rows) != EXPECTED_CANDIDATES
        or any(row["D4_Sylvester_solvable"] for row in rows)
        or any(set(row["nonzero_equal_eigenspace_compressions"]) != {"0"} for row in rows)
        or len(target_hashes) != 1
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "first regular revised-symbol obstruction mismatch"
        )
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
    intersection = (
        selector_coordinates.rank()
        + target_plane.rank()
        - selector_basis.row_join(target_plane).rank()
    )
    terms = _decompose(target_coordinates, selector_coordinates, covector_basis, reference)
    aligned_block = sum(
        (term["output_aligned"] * term["right_aligned"].T for term in terms),
        sp.zeros(STATE_DIMENSION),
    ).applyfunc(sp.factor)
    global_block = (state_rotation * aligned_block * state_rotation.T).applyfunc(sp.factor)
    envelope = _bounded_preserving_envelope()
    angular = degree_five._angular_extension(terms, state_rotation, tuple(direction), envelope)
    if angular["reference_block"] != global_block:
        raise RevisedSymbolRationalCounterexampleError("bounded e3 escape reference block mismatch")
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
        not quotient_target.is_zero_matrix
        or intersection != target.rank()
        or not corrected_rows
        or any(
            not row["D4_Sylvester_solvable"] or row["nonzero_equal_eigenspace_compressions"] != {}
            for row in corrected_rows
        )
    ):
        raise RevisedSymbolRationalCounterexampleError("bounded transverse-curl e3 escape mismatch")
    return {
        "atlas": atlas["record"],
        "search_protocol": {
            "chart_order": ["primary_e1_stereographic", "antipodal_e1_stereographic"],
            "preregistered_points": [
                {"chart": "primary_e1_stereographic", "coordinates": ["0", "1"]}
            ],
            "points_evaluated": 1,
            "stopped_at_first_regular_obstruction": True,
            "antipodal_chart_points_evaluated_after_obstruction": 0,
        },
        "first_obstruction": {
            "selector": {
                "chart": "primary_e1_stereographic",
                "chart_coordinates": ["0", "1"],
                "chart_denominator_value": "2",
                "direction": [str(value) for value in direction],
                "frame_name": frame["name"],
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
                "xyz_block_sha256": _content_hash(_matrix_payload(xyz_block)),
                "sixth_block_sha256": _content_hash(_matrix_payload(sixth_block)),
                "degree_five_block_zero_at_e3": degree_five_block.is_zero_matrix,
                "degree_five_block_sha256": _content_hash(_matrix_payload(degree_five_block)),
                "revised_global_symbol_rank": revised_global.rank(),
                "revised_global_symbol_sha256": _content_hash(_matrix_payload(revised_global)),
                "revised_aligned_symbol_sha256": _content_hash(_matrix_payload(revised_aligned)),
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
                "revised_seven_frame_full_sphere_D4_compatibility_disproved": True,
            },
        },
        "bounded_next_escape": {
            "exact_range_classification": {
                "transverse_selector_rank": selector_coordinates.rank(),
                "selector_sha256": _content_hash(_matrix_payload(selector_coordinates)),
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
                "degree_five_extension_sha256": _content_hash(
                    _matrix_payload(angular["extension"])
                ),
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
                "prior_seven_direction_certificates_preserved": True,
                "total_local_direction_certificates": 8,
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
        or tuple(config.get("active_indices", ())) != ACTIVE_INDICES
        or config.get("obligation_offset") != OBLIGATION_OFFSET
        or config.get("expected_candidate_count") != EXPECTED_CANDIDATES
        or config.get("search_schedule")
        != [{"chart": "primary_e1_stereographic", "coordinates": ["0", "1"]}]
        or config.get("bounded_envelope_degrees") != [0, 2, 4]
        or config.get("bounded_envelope_max_support") != 5
    ):
        raise RevisedSymbolRationalCounterexampleError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
        "degree_five_predecessor",
        "rational_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    )
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_result(
        bound["degree_five_predecessor"],
        bound["rational_predecessor"],
        bound["xyz_predecessor"],
        bound["c23_predecessor"],
        bound["minimal_escape"],
        bound["fourth_campaign"],
    )
    obstruction = exact["first_obstruction"]["exact_rational_obstruction"]
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
            "regular_search_points_evaluated": 1,
            "directional_polarization_evaluations": 15,
            "recurrence_orders_checked": 4,
            "candidate_conditions_checked": obstruction["candidate_conditions_checked"],
            "candidate_compatibilities": obstruction["candidate_compatibilities"],
            "candidate_obstructions": obstruction["candidate_obstructions"],
            "preserved_direction_constraints": 7,
            "bounded_envelope_degrees_checked": 3,
            "bounded_envelope_supports_checked": 4943,
            "sparsest_envelope_support": 5,
            "sparsest_feasible_envelopes": 110,
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
            "infer_full_sphere_from_eight_directions": {"rejected": True},
            "ignore_zero_speed_compression": {"rejected": True},
            "skip_lower_recurrence_orders": {"rejected": True},
            "skip_antipodal_atlas_chart": {"rejected": True},
            "skip_exact_denominator_clearing": {"rejected": True},
            "search_envelopes_beyond_preregistered_bound": {"rejected": True},
            "infer_PDE_or_tube_admission": {"rejected": True},
            "infer_B7_H7_or_lifespan": {"rejected": True},
        },
        "scope": (
            "Exact fail-fast two-chart rational search of the revised seven-frame degree-five "
            "symbol at the preregistered primary-chart point (u,v)=(0,1). Full orders one "
            "through four and all 12 candidates obstruct at e3 before a bounded degree-0/2/4, "
            "support-at-most-five preserving-envelope classification constructs an eighth local "
            "certificate. No finite determining, full-sphere, PDE, tube, CK, TC2, B7, H7, or "
            "lifespan claim follows."
        ),
        "next_gate": (
            "Repeat the fail-fast exact rational-chart search for the revised eight-frame symbol; "
            "do not infer a finite determining theorem or any PDE/global admission."
        ),
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    gate = document.get("exact_gate", {})
    first = gate.get("first_obstruction", {})
    recurrence = first.get("full_recurrence", {})
    obstruction = first.get("exact_rational_obstruction", {})
    bounded = gate.get("bounded_next_escape", {})
    range_classification = bounded.get("exact_range_classification", {})
    envelope = gate.get("bounded_next_escape", {}).get("minimal_preserving_envelope", {})
    completion = gate.get("bounded_next_escape", {}).get("local_completion", {})
    claims = document.get("claims", {})
    counts = document.get("counts", {})
    obstruction_rows = obstruction.get("candidate_records", [])
    completion_rows = completion.get("candidate_records", [])
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or set(document)
        != {
            "claims",
            "config_sha256",
            "content_sha256",
            "counts",
            "errors",
            "exact_gate",
            "negative_controls",
            "next_gate",
            "schema_version",
            "scope",
            "source_bindings",
            "status",
        }
        or document.get("errors") != []
        or counts
        != {
            "bound_predecessors": 6,
            "bounded_envelope_degrees_checked": 3,
            "bounded_envelope_supports_checked": 4943,
            "candidate_compatibilities": 0,
            "candidate_conditions_checked": 12,
            "candidate_obstructions": 12,
            "directional_polarization_evaluations": 15,
            "inferred_global_passes": 0,
            "negative_controls": 10,
            "new_local_candidate_compatibilities": 12,
            "new_local_candidate_obstructions": 0,
            "preregistered_search_points": 1,
            "preserved_direction_constraints": 7,
            "rational_SO3_charts": 2,
            "recurrence_orders_checked": 4,
            "regular_search_points_evaluated": 1,
            "sparsest_envelope_support": 5,
            "sparsest_feasible_envelopes": 110,
            "total_local_direction_certificates": 8,
        }
        or _content_hash(gate) != EXPECTED_EXACT_GATE_SHA256
        or gate.get("atlas", {}).get("union_covers_real_S2") is not True
        or gate.get("search_protocol", {}).get("stopped_at_first_regular_obstruction") is not True
        or gate.get("search_protocol", {}).get("points_evaluated") != 1
        or gate.get("search_protocol", {}).get("antipodal_chart_points_evaluated_after_obstruction")
        != 0
        or first.get("selector", {}).get("direction") != ["0", "0", "1"]
        or recurrence.get("orders_checked") != [1, 2, 3, 4]
        or recurrence.get("lower_orders_certified_per_polarization_direction") != [1, 2, 3]
        or recurrence.get("directional_polarization_evaluations") != 15
        or recurrence.get("candidate_fourth_order_systems") != 12
        or recurrence.get("degree_five_block_zero_at_e3") is not True
        or recurrence.get("revised_global_symbol_rank") != 3
        or obstruction.get("candidate_compatibilities") != 0
        or obstruction.get("candidate_obstructions") != 12
        or obstruction.get("distinct_eta_normalized_targets") != 1
        or obstruction.get("eta_normalized_target_rank") != 4
        or obstruction.get("eta_normalized_target_nonzero_entries") != 56
        or len(obstruction_rows) != 12
        or any(
            row.get("D4_Sylvester_solvable") is not False
            or set(row.get("nonzero_equal_eigenspace_compressions", {})) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"].get("rank") != 4
            or row["nonzero_equal_eigenspace_compressions"]["0"].get("nonzero_entries") != 56
            or row.get("zero_speed_cleared_numerator", {}).get("numerator_rank") != 4
            or row.get("zero_speed_cleared_numerator", {}).get("numerator_nonzero_entries") != 56
            for row in obstruction_rows
        )
        or range_classification
        != {
            "quotient_target_sha256": (
                "6bd0f4db2919abb53bd3fc437f3ec440b1c2df2a8e73fe17d06d0a3fc1c10f23"
            ),
            "quotient_target_zero": True,
            "selector_sha256": ("7ef398226365b9e42bd543a3b9c5b00c82621cbf8f67d76b2768e38e81441d26"),
            "selector_target_plane_intersection_dimension": 4,
            "target_in_full_transverse_curl_range": True,
            "target_plane_dimension": 4,
            "transverse_selector_rank": 22,
        }
        or envelope.get("minimal_even_homogeneous_degree") != 4
        or envelope.get("degree_zero", {}).get("prior_zero_nullity") != 0
        or envelope.get("degree_two", {}).get("prior_zero_nullity") != 0
        or envelope.get("degree_four", {}).get("prior_zero_nullity") != 8
        or envelope.get("sparsest_support_size") != 5
        or envelope.get("sparsest_support_feasible_envelopes") != 110
        or envelope.get("total_supports_checked") != 4943
        or envelope.get("supports_checked_by_size")
        != {"1": 15, "2": 105, "3": 455, "4": 1365, "5": 3003}
        or envelope.get("feasible_envelopes_by_support_size")
        != {"1": 0, "2": 0, "3": 0, "4": 0, "5": 110}
        or envelope.get("deterministic_envelope_support_indices") != [0, 1, 2, 3, 5]
        or envelope.get("zero_at_all_seven_predecessor_frames") is not True
        or completion.get("candidate_compatibilities") != 12
        or completion.get("candidate_obstructions") != 0
        or completion.get("candidate_conditions_checked") != 12
        or completion.get("constructed_completion_rank") != 2
        or completion.get("elementary_curl_channels") != 2
        or completion.get("coordinate_pairs") != [[11, 21], [15, 32]]
        or completion.get("gradient_residual_zero") is not True
        or completion.get("prior_seven_direction_certificates_preserved") is not True
        or completion.get("total_local_direction_certificates") != 8
        or len(completion_rows) != 12
        or [row.get("candidate_id") for row in completion_rows]
        != [row.get("candidate_id") for row in obstruction_rows]
        or any(
            row.get("D4_Sylvester_solvable") is not True
            or row.get("nonzero_equal_eigenspace_compressions") != {}
            for row in completion_rows
        )
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or len(document.get("negative_controls", {})) != 10
        or any(
            value != {"rejected": True} for value in document.get("negative_controls", {}).values()
        )
    ):
        raise RevisedSymbolRationalCounterexampleError(
            "revised-symbol rational campaign validation failed"
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
    print(
        json.dumps(
            {
                "status": artifact["status"],
                "content_sha256": artifact["content_sha256"],
                "candidate_obstructions": 12,
                "new_local_compatibilities": 12,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
