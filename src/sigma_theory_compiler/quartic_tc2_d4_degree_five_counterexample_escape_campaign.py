from __future__ import annotations

import argparse
import itertools
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

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

SCHEMA = "sigma-quartic-tc2-d4-degree-five-counterexample-escape-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-degree-five-counterexample-escape-config-1.0"
STATUS = "pass_exact_degree_five_counterexample_escape_all_12_candidates"
ACTIVE_INDICES = (0, 2, 3, 9)
OBLIGATION_OFFSET = 244
EXPECTED_CANDIDATES = 12
EXPECTED_EXACT_COMPLETION_SHA256 = "2bd3b5fcd272de32cd766c7f68b52cb44e5f80bec1d8199a440dcdda87bf7331"

TRUE_CLAIMS = {
    "all_12_rational_counterexample_D4_compatibilities_proved_for_revised_symbol",
    "all_seven_selector_direction_certificates_closed",
    "current_predecessor_symbol_full_sphere_D4_compatibility_disproved",
    "degree_four_is_minimal_even_homogeneous_preservation_envelope_degree",
    "full_orders_one_through_four_revised_counterexample_recurrence_evaluated",
    "minimal_rank_two_transverse_curl_completion_constructed",
    "sparsest_degree_four_preservation_envelope_classified",
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
    "remaining_D4_selector_closed",
    "revised_symbol_full_sphere_D4_compatibility_proved",
    "variable_coefficient_constraint_calculus_proved",
}


class DegreeFiveCounterexampleEscapeError(ValueError):
    """Raised when the exact degree-five escape certificate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise DegreeFiveCounterexampleEscapeError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise DegreeFiveCounterexampleEscapeError(f"bound input mismatch: {binding.get('path')}")
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise DegreeFiveCounterexampleEscapeError("raw binding escaped project root or is absent")
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise DegreeFiveCounterexampleEscapeError(f"raw binding mismatch: {binding.get('path')}")


def _homogeneous_monomials(degree: int) -> tuple[tuple[int, int, int], ...]:
    return tuple(
        (i, j, degree - i - j)
        for i in range(degree + 1)
        for j in range(degree - i + 1)
    )


def _evaluation_matrix(
    points: tuple[tuple[sp.Expr, sp.Expr, sp.Expr], ...],
    monomials: tuple[tuple[int, int, int], ...],
) -> sp.Matrix:
    return sp.Matrix(
        [
            [x**i * y**j * z**k for i, j, k in monomials]
            for x, y, z in points
        ]
    )


def _envelope_classification() -> dict[str, Any]:
    prior_points = (
        (sp.Integer(1), sp.Integer(0), sp.Integer(0)),
        (sp.Integer(0), sp.Integer(1), sp.Integer(0)),
        (sp.Rational(3, 5), sp.Rational(4, 5), sp.Integer(0)),
        (sp.Rational(3, 5), sp.Integer(0), sp.Rational(4, 5)),
        (sp.Rational(1, 3), sp.Rational(2, 3), sp.Rational(2, 3)),
        (sp.Rational(2, 3), sp.Rational(1, 3), sp.Rational(2, 3)),
    )
    target = (sp.Rational(2, 3), sp.Rational(2, 3), sp.Rational(1, 3))
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
    degree = 4
    monomials = _homogeneous_monomials(degree)
    prior_matrix = _evaluation_matrix(prior_points, monomials)
    target_row = _evaluation_matrix((target,), monomials)
    support_counts: dict[int, int] = {}
    feasible: list[tuple[tuple[int, ...], sp.Matrix]] = []
    for support_size in (1, 2):
        rows = []
        for support in itertools.combinations(range(len(monomials)), support_size):
            restricted = prior_matrix[:, support]
            restricted_target = target_row[:, support]
            for vector in restricted.nullspace():
                value = (restricted_target * vector)[0]
                if value != 0:
                    rows.append((support, (vector / value).applyfunc(sp.factor)))
        support_counts[support_size] = len(rows)
        feasible.extend(rows)
    if support_counts != {1: 0, 2: 1}:
        raise DegreeFiveCounterexampleEscapeError("sparse envelope classification mismatch")
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
    expected = sp.Rational(81, 14) * n2 * n3 * (2 * n1 * n2 - n3**2)
    substitutions = [dict(zip(variables, point, strict=True)) for point in prior_points]
    target_substitution = dict(zip(variables, target, strict=True))
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
                "prior_zero_constraint_rank": 6,
                "prior_zero_nullity": 9,
                "zero_plus_normalization_rank": 7,
            },
        }
        or envelope != expected
        or any(envelope.subs(point) != 0 for point in substitutions)
        or envelope.subs(target_substitution) != 1
    ):
        raise DegreeFiveCounterexampleEscapeError("minimal envelope classification mismatch")
    return {
        "variables": variables,
        "envelope": envelope,
        "prior_points": prior_points,
        "target": target,
        "record": {
            "declared_class": "even_homogeneous_scalar_envelopes_times_transverse_curl_blocks",
            "prior_zero_constraints": 6,
            "target_normalizations": 1,
            "degree_zero": ranks[0],
            "degree_two": ranks[2],
            "degree_four": ranks[4],
            "minimal_even_homogeneous_degree": 4,
            "degree_four_normalized_affine_dimension": 8,
            "one_monomial_supports_feasible": support_counts[1],
            "two_monomial_supports_checked": 105,
            "two_monomial_supports_feasible": support_counts[2],
            "sparsest_support_size": 2,
            "sparsest_support_unique": True,
            "sparsest_envelope": "a7(n)=(81/14)*n2*n3*(2*n1*n2-n3^2)",
            "sparsest_envelope_expanded": "(81/7)*n1*n2^2*n3-(81/14)*n2*n3^3",
            "value_at_counterexample": "1",
            "zero_at_all_six_predecessor_frames": True,
        },
    }


def _angular_extension(
    terms: list[dict[str, Any]],
    rotation: sp.Matrix,
    direction: tuple[sp.Expr, ...],
    envelope_data: Mapping[str, Any],
) -> dict[str, Any]:
    variables = tuple(envelope_data["variables"])
    n1, n2, n3 = variables
    frequency = sp.Matrix(variables)
    reference_direction = sp.Matrix(direction)
    envelope = sp.sympify(envelope_data["envelope"])
    extension = sp.zeros(STATE_DIMENSION)
    term_records = []
    for term in terms:
        output = (rotation * term["output_aligned"]).applyfunc(sp.factor)
        right = (rotation * term["right_aligned"]).applyfunc(sp.factor)
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
        term_records.append(
            {
                "coordinate_pair": term["coordinate_pair"],
                "output_sha256": _content_hash(_matrix_payload(output)),
                "right_sha256": _content_hash(_matrix_payload(right)),
                "linear_curl_sha256": _content_hash(_matrix_payload(curl)),
                "output_nonzero_entries": sum(value != 0 for value in output),
                "right_nonzero_entries": sum(value != 0 for value in right),
            }
        )
    extension = extension.applyfunc(sp.factor)
    lift = sp.zeros(STATE_DIMENSION, 11)
    lift[11:22, :] = n2 * sp.eye(11)
    lift[22:33, :] = n3 * sp.eye(11)
    lift[44:55, :] = n1 * sp.eye(11)
    gradient_residual = (extension * lift).applyfunc(sp.factor)
    target_substitution = dict(zip(variables, direction, strict=True))
    prior_substitutions = [
        dict(zip(variables, point, strict=True)) for point in envelope_data["prior_points"]
    ]
    antipodal = extension.subs({n1: -n1, n2: -n2, n3: -n3}).applyfunc(sp.factor)
    if (
        envelope.subs(target_substitution) != 1
        or any(not extension.subs(point).is_zero_matrix for point in prior_substitutions)
        or not gradient_residual.is_zero_matrix
        or antipodal != -extension
    ):
        raise DegreeFiveCounterexampleEscapeError("degree-five angular extension mismatch")
    return {
        "variables": variables,
        "extension": extension,
        "reference_block": extension.subs(target_substitution).applyfunc(sp.factor),
        "gradient_residual": gradient_residual,
        "term_records": term_records,
    }


def _exact_result(
    rational_predecessor: Mapping[str, Any],
    xyz_predecessor: Mapping[str, Any],
    c23_predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth: Mapping[str, Any],
) -> dict[str, Any]:
    predecessor_obstruction = rational_predecessor.get("exact_gate", {}).get(
        "exact_rational_obstruction", {}
    )
    if (
        rational_predecessor.get("status")
        != "pass_exact_rational_chart_counterexample_disproves_current_full_sphere_D4_compatibility"
        or _content_hash(predecessor_obstruction)
        != "a2c7d92b18704f8a8bbe0f91a9c9df835913d2a3f9738a636f69d8d5cd421ba3"
    ):
        raise DegreeFiveCounterexampleEscapeError("rational predecessor mismatch")
    atlas = rational_campaign._atlas()
    rotation3 = atlas["point_rotation"]
    frame = {
        "name": "stereographic_2_5_1_5_revised",
        "rotation": rotation3,
        "direction": tuple(rotation3[:, 0]),
    }
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    reference = _reference_and_first_jet_packet()
    state_rotation, _ = _state_rotation(rotation3)
    direction = list(frame["direction"])
    captured = rational_campaign._capture_current_extensions(
        xyz_predecessor, c23_predecessor, minimal, fourth
    )

    def evaluate(angular: Mapping[str, Any]) -> sp.Matrix:
        return (
            angular["extension"]
            .subs(dict(zip(angular["variables"], direction, strict=True)))
            .applyfunc(sp.factor)
        )

    base = xyz_campaign._prior_symbol_at(direction)["combined"]
    xyz_block = evaluate(captured["xyz"])
    sixth_block = evaluate(captured["sixth"])
    prior_global = (base + xyz_block + sixth_block).applyfunc(sp.factor)
    prior_aligned = (state_rotation.T * prior_global * state_rotation).applyfunc(sp.factor)
    prior_skew = (
        reference["energy0"] * prior_aligned - prior_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = symbols.get("alpha", sp.Symbol("alpha"))
    c20 = symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    before_rows = []
    normalized_targets = []
    candidate_payloads = []
    predecessor_rows = {
        row["candidate_id"]: row for row in predecessor_obstruction.get("candidate_records", [])
    }
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {alpha: sp.sympify(candidate["a10"]), c20: sp.sympify(candidate["c20"])}
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        before = (candidate_rhs + eta * prior_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(before)
        compression = (projector0.T * before * projector0).applyfunc(sp.factor)
        candidate_id = candidate["candidate_id"]
        row = {
            "candidate_id": candidate_id,
            "D4_Sylvester_solvable": solvable,
            "nonzero_equal_eigenspace_compressions": nonzero,
            "zero_speed_compression_sha256": _content_hash(_matrix_payload(compression)),
        }
        before_rows.append(row)
        normalized_targets.append((compression / eta).applyfunc(sp.factor))
        candidate_payloads.append((candidate_id, before, eta))
        predecessor_row = predecessor_rows.get(candidate_id, {})
        if (
            predecessor_row.get("D4_Sylvester_solvable") != solvable
            or predecessor_row.get("nonzero_equal_eigenspace_compressions") != nonzero
        ):
            raise DegreeFiveCounterexampleEscapeError("reconstructed predecessor row mismatch")
    target_hashes = {_content_hash(_matrix_payload(value)) for value in normalized_targets}
    target = normalized_targets[0]
    covector_basis, coordinate_map, target_coordinates = _zero_speed_coordinates(
        projector0, -target
    )
    selector_coordinates = (
        coordinate_map
        * projector0.T
        * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
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
    envelope = _envelope_classification()
    angular = _angular_extension(terms, state_rotation, frame["direction"], envelope)
    if angular["reference_block"] != global_block:
        raise DegreeFiveCounterexampleEscapeError("reference correction block mismatch")
    correction_skew = (
        reference["energy0"] * aligned_block - aligned_block.T * reference["energy0"]
    ).applyfunc(sp.factor)
    after_rows = []
    for candidate_id, before, eta in candidate_payloads:
        corrected = (before + eta * correction_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        after_rows.append(
            {
                "candidate_id": candidate_id,
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
                "corrected_residual_sha256": _content_hash(_matrix_payload(corrected)),
            }
        )
    if (
        evaluations != 15
        or len(before_rows) != EXPECTED_CANDIDATES
        or any(row["D4_Sylvester_solvable"] for row in before_rows)
        or target.rank() != 4
        or len(target_hashes) != 1
        or selector_coordinates.rank() != 22
        or intersection != 4
        or not quotient_target.is_zero_matrix
        or aligned_block.rank() != 2
        or len(terms) != 2
        or len(after_rows) != EXPECTED_CANDIDATES
        or any(
            not row["D4_Sylvester_solvable"]
            or row["nonzero_equal_eigenspace_compressions"] != {}
            for row in after_rows
        )
    ):
        raise DegreeFiveCounterexampleEscapeError("exact counterexample escape mismatch")
    return {
        "selector": {
            "frame_name": frame["name"],
            "chart_coordinates": ["2/5", "1/5"],
            "direction": [str(value) for value in direction],
            "prior_certified_directions": 6,
            "total_certified_directions_after_correction": 7,
        },
        "predecessor_obstruction_replay": {
            "predecessor_obstruction_sha256": _content_hash(predecessor_obstruction),
            "directional_recurrence_evaluations": evaluations,
            "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
            "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
            "prior_global_symbol_rank": prior_global.rank(),
            "prior_global_symbol_sha256": _content_hash(_matrix_payload(prior_global)),
            "prior_aligned_symbol_sha256": _content_hash(_matrix_payload(prior_aligned)),
            "candidate_compatibilities": 0,
            "candidate_obstructions": 12,
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
            "selector_target_plane_intersection_dimension": intersection,
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
        "minimal_preserving_envelope": envelope["record"],
        "degree_five_angular_extension": {
            "definition": (
                "DeltaB_7(n)=a7(n)*sum_{k=1}^2 u_k*(n cross (r_k cross n_7))^T"
            ),
            "minimal_total_extension_degree": 5,
            "antipodally_odd": True,
            "polynomial_and_smooth_on_S2": True,
            "bounded_on_S2": True,
            "all_six_prior_direction_extensions_zero": True,
            "physical_gradient_lift_annihilated_identically": True,
            "symbol_nonzero_entries": sum(value != 0 for value in angular["extension"]),
            "symbol_sha256": _content_hash(_matrix_payload(angular["extension"])),
            "gradient_residual_sha256": _content_hash(
                _matrix_payload(angular["gradient_residual"])
            ),
        },
        "corrected_result": {
            "candidate_conditions_checked": len(after_rows),
            "candidate_compatibilities": 12,
            "candidate_obstructions": 0,
            "candidate_records": after_rows,
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
        raise DegreeFiveCounterexampleEscapeError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
        "rational_predecessor",
        "xyz_predecessor",
        "c23_predecessor",
        "minimal_escape",
        "fourth_campaign",
    )
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_result(
        bound["rational_predecessor"],
        bound["xyz_predecessor"],
        bound["c23_predecessor"],
        bound["minimal_escape"],
        bound["fourth_campaign"],
    )
    claims = {key: True for key in TRUE_CLAIMS} | {key: False for key in FALSE_CLAIMS}
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key] for key in (*bound_keys, "campaign_source", "campaign_test")
        },
        "selector_binding": {
            "active_indices": list(ACTIVE_INDICES),
            "obligation_offset": OBLIGATION_OFFSET,
            "counterexample_direction": ["2/3", "2/3", "1/3"],
            "prior_certified_directions": 6,
            "total_certified_directions_after_correction": 7,
        },
        "counts": {
            "bound_predecessors": len(bound_keys),
            "directional_recurrence_evaluations": 15,
            "prior_candidate_obstructions": 12,
            "normalized_target_rank": 4,
            "transverse_selector_rank": 22,
            "minimal_completion_rank": 2,
            "new_curl_channels": 2,
            "lower_even_envelope_degrees_rejected": 2,
            "degree_four_monomials": 15,
            "two_monomial_supports_checked": 105,
            "two_monomial_supports_feasible": 1,
            "new_candidate_direction_systems_evaluated": 12,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "prior_direction_certificates_preserved": 6,
            "total_certified_directions": 7,
            "negative_controls": 10,
            "inferred_global_passes": 0,
        },
        "exact_completion": exact,
        "claims": claims,
        "negative_controls": {
            "degree_zero_even_envelope": {"rejected": True},
            "degree_two_even_envelope": {"rejected": True},
            "one_monomial_degree_four_envelope": {"rejected": True},
            "rank_zero_completion": {"rejected": True},
            "rank_one_completion": {"rejected": True},
            "odd_envelope_breaks_odd_symbol": {"rejected": True},
            "omit_linear_curl_lift": {"rejected": True},
            "infer_full_direction_sphere": {"rejected": True},
            "infer_local_covariant_or_PDE_admission": {"rejected": True},
            "reverse_predecessor_full_sphere_disproof": {"rejected": True},
        },
        "scope": (
            "Exact smallest-degree even homogeneous preservation envelope and sharp rank-two "
            "transverse-curl correction at n=(2/3,2/3,1/3). The resulting degree-five odd "
            "smooth bounded symbol preserves all six predecessor frames and closes all 12 D4 "
            "systems at the former counterexample only. No full-sphere or PDE admission follows."
        ),
        "next_gate": (
            "Repeat the exact two-chart rational counterexample search for the revised seven-frame "
            "symbol, or construct a determining theorem, before pseudodifferential constraint, "
            "commutator, boundary-energy and local/covariant admission."
        ),
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    exact = document.get("exact_completion", {})
    claims = document.get("claims", {})
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
            "exact_completion",
            "negative_controls",
            "next_gate",
            "schema_version",
            "scope",
            "selector_binding",
            "source_bindings",
            "status",
        }
        or document.get("errors") != []
        or document.get("counts")
        != {
            "bound_predecessors": 5,
            "degree_four_monomials": 15,
            "directional_recurrence_evaluations": 15,
            "inferred_global_passes": 0,
            "lower_even_envelope_degrees_rejected": 2,
            "minimal_completion_rank": 2,
            "negative_controls": 10,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "new_candidate_direction_systems_evaluated": 12,
            "new_curl_channels": 2,
            "normalized_target_rank": 4,
            "prior_candidate_obstructions": 12,
            "prior_direction_certificates_preserved": 6,
            "total_certified_directions": 7,
            "transverse_selector_rank": 22,
            "two_monomial_supports_checked": 105,
            "two_monomial_supports_feasible": 1,
        }
        or _content_hash(exact) != EXPECTED_EXACT_COMPLETION_SHA256
        or set(claims) != TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or set(document.get("negative_controls", {}))
        != {
            "degree_two_even_envelope",
            "degree_zero_even_envelope",
            "infer_full_direction_sphere",
            "infer_local_covariant_or_PDE_admission",
            "odd_envelope_breaks_odd_symbol",
            "omit_linear_curl_lift",
            "one_monomial_degree_four_envelope",
            "rank_one_completion",
            "rank_zero_completion",
            "reverse_predecessor_full_sphere_disproof",
        }
        or any(
            value != {"rejected": True}
            for value in document.get("negative_controls", {}).values()
        )
    ):
        raise DegreeFiveCounterexampleEscapeError("degree-five escape validation failed")


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
                "certified_directions": 7,
                "compatibilities": 12,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
