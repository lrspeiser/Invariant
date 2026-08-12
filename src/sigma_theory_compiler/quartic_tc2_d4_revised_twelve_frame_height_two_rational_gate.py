from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_d4_degree_five_counterexample_escape_campaign as degree_five
from . import quartic_tc2_d4_degree_three_rank_two_xyz_completion_campaign as xyz_campaign
from . import quartic_tc2_d4_rational_chart_determining_gate as rational_campaign
from . import quartic_tc2_d4_revised_eight_frame_rational_counterexample_campaign as common
from . import quartic_tc2_d4_revised_eleven_frame_degree_six_envelope_gate as predecessor
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

SCHEMA = "sigma-quartic-tc2-d4-revised-twelve-frame-height-two-rational-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-revised-twelve-frame-height-two-rational-config-1.0"
STATUS = "pass_exact_second_height_two_point_and_bounded_classification"
EXPECTED_CANDIDATES = 12
EXPECTED_PREDECESSOR_CONTENT_SHA256 = (
    "0c2b8ce7bd797a62111b824a892b3d1f3db075ad87271c3bb1a4b9f58a87418a"
)
EXPECTED_PREDECESSOR_EXACT_SHA256 = (
    "962c4014df740b9ac117a9c344354b73b6d36d74e09fd003625874c63f1f0a0d"
)
EXPECTED_PREDECESSOR_EXTENSION_SHA256 = (
    "5625ccb9e40bc54fd89950f280d9eb72f0c89826b3584d53eae706d648f2529d"
)
EXPECTED_EXACT_GATE_SHA256 = "6edb5a24475fb5e86b6b320da071f7435120cf313c805cbbc9cec0bc09db025a"
EXPECTED_TARGET_SHA256 = "d15ad031088553092f1d6f6901d4c6dfe1fb97265c24e8d8ecb993ff62a7d372"
EXPECTED_EXTENSION_SHA256 = "8bfbede45b170df495d3e5073bc4874277fed1e06e04cd66533383ba791cc6fa"
CONFIG_PATH = (
    "configs/backgrounds/quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate.json"
)
SOURCE_PATH = (
    "src/sigma_theory_compiler/quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate.py"
)
TEST_PATH = "tests/test_quartic_tc2_d4_revised_twelve_frame_height_two_rational_gate.py"

SELECTOR = [{"chart": "primary_e1_stereographic", "coordinates": ["1", "-2"]}]
ENVELOPE_DEGREES = [0, 2, 4, 6]
REPAIR_CLASS = {
    "transverse_curl_state_indices": list(TRANSVERSE_CURL_INDICES),
    "deterministic_decomposition": "lexicographic_first_nonzero_skew_pivot",
    "maximum_elementary_curl_channels": 4,
    "scalar_envelope_degrees": ENVELOPE_DEGREES,
    "degree_six_full_support_ceiling": 28,
}
TRUE_CLAIMS = {
    "bounded_degree_six_envelope_and_transverse_curl_repair_class_preregistered",
    "exact_two_chart_SO3_atlas_reused",
    "full_orders_one_through_four_recurrence_evaluated",
    "second_primitive_height_two_selector_evaluated",
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
    "variable_coefficient_constraint_calculus_proved",
}
EXPECTED_SCOPE = (
    "One exact rational audit at the second preregistered primitive non-axis height-two point, "
    "with only the finite-dimensional even homogeneous degree-0/2/4/6 scalar-envelope and "
    "bounded transverse-curl repair class. No finite determining, full-sphere, covariant, PDE, "
    "tube, CK, TC2, B7, H7, or lifespan claim follows."
)
EXPECTED_NEXT_GATE = (
    "Preregister exactly one further bounded rational selector/class or prove a finite "
    "direction-sphere determining theorem for the revised thirteen-frame symbol."
)
EXPECTED_NEGATIVE_CONTROLS = {
    "evaluate_more_than_one_new_point": {"rejected": True},
    "expand_envelope_above_degree_six": {"rejected": True},
    "expand_transverse_curl_channels": {"rejected": True},
    "infer_B7_H7_or_lifespan": {"rejected": True},
    "infer_PDE_or_tube_admission": {"rejected": True},
    "infer_finite_determining_theorem": {"rejected": True},
    "infer_full_direction_sphere": {"rejected": True},
    "skip_lower_recurrence_orders": {"rejected": True},
}
EXPECTED_DOCUMENT_KEYS = {
    "claims",
    "config_file_sha256",
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


class RevisedTwelveFrameHeightTwoRationalError(ValueError):
    """Raised when the bounded revised-twelve-frame gate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RevisedTwelveFrameHeightTwoRationalError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RevisedTwelveFrameHeightTwoRationalError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RevisedTwelveFrameHeightTwoRationalError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RevisedTwelveFrameHeightTwoRationalError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _predecessor_chain(
    root: Path, degree_six_artifact: Mapping[str, Any]
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    if (
        degree_six_artifact.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or _content_hash(degree_six_artifact.get("exact_gate", {}))
        != EXPECTED_PREDECESSOR_EXACT_SHA256
    ):
        raise RevisedTwelveFrameHeightTwoRationalError("degree-six predecessor mismatch")
    predecessor.validate_campaign(degree_six_artifact)
    bindings = degree_six_artifact.get("source_bindings", {})
    keys = (
        "exhaustion_predecessor",
        "eleven_frame_symbol_predecessor",
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
    artifacts = {key: _load_bound(root, bindings[key]) for key in keys}
    extensions = predecessor.prior._capture_eleven_frame_extensions(
        artifacts["eleven_frame_symbol_predecessor"],
        artifacts["ten_frame_predecessor"],
        artifacts["nine_frame_predecessor"],
        artifacts["revised_predecessor"],
        artifacts["degree_five_predecessor"],
        artifacts["rational_predecessor"],
        artifacts["xyz_predecessor"],
        artifacts["c23_predecessor"],
        artifacts["minimal_escape"],
        artifacts["fourth_campaign"],
    )
    atlas = rational_campaign._atlas()
    rotation, direction, payload, evaluations = common._frame_payload(
        atlas,
        (sp.Integer(1), sp.Integer(2)),
        "reconstructed_height_two_1_2",
        artifacts["fourth_campaign"],
    )
    if direction != [sp.Rational(-2, 3), sp.Rational(1, 3), sp.Rational(2, 3)] or evaluations != 15:
        raise RevisedTwelveFrameHeightTwoRationalError("predecessor repair frame mismatch")
    envelope = predecessor._deterministic_normalized_envelope(tuple(direction))
    extension = predecessor.prior._reconstruct_extension(
        rotation,
        direction,
        payload,
        extensions,
        envelope,
        artifacts["minimal_escape"],
    )
    extension_sha = _content_hash(_matrix_payload(extension["extension"]))
    if extension_sha != EXPECTED_PREDECESSOR_EXTENSION_SHA256:
        raise RevisedTwelveFrameHeightTwoRationalError("predecessor extension mismatch")
    extensions["height_two_1_2"] = extension
    return extensions, artifacts


def _normalized_envelope(target: tuple[sp.Expr, sp.Expr, sp.Expr]) -> dict[str, Any]:
    previous = predecessor._deterministic_normalized_envelope(
        (sp.Rational(-2, 3), sp.Rational(1, 3), sp.Rational(2, 3))
    )
    prior_points = previous["prior_points"] + (previous["target"],)
    records: dict[str, dict[str, Any]] = {}
    selected: tuple[int, tuple[tuple[int, int, int], ...], sp.Matrix] | None = None
    for degree in ENVELOPE_DEGREES:
        monomials = common._homogeneous_monomials(degree)
        zero_matrix = common._evaluation_matrix(prior_points, monomials)
        target_row = common._evaluation_matrix((target,), monomials)
        system = zero_matrix.col_join(target_row)
        rhs = sp.zeros(system.rows, 1)
        rhs[-1] = 1
        augmented = system.row_join(rhs)
        rref, pivots = augmented.rref()
        feasible = not any(
            all(rref[row, column] == 0 for column in range(system.cols)) and rref[row, -1] != 0
            for row in range(rref.rows)
        )
        records[str(degree)] = {
            "monomial_dimension": len(monomials),
            "prior_zero_constraint_rank": zero_matrix.rank(),
            "prior_zero_nullity": len(monomials) - zero_matrix.rank(),
            "zero_plus_normalization_rank": system.rank(),
            "normalized_envelope_exists": feasible,
        }
        if feasible and selected is None:
            coefficients = sp.zeros(len(monomials), 1)
            for row, pivot in enumerate(pivots):
                if pivot < len(monomials):
                    coefficients[pivot] = sp.factor(rref[row, -1])
            selected = degree, monomials, coefficients
    base_record: dict[str, Any] = {
        "declared_degree_ladder": ENVELOPE_DEGREES,
        "degree_records": records,
        "degree_six_full_support_ceiling": 28,
    }
    if selected is None:
        return {
            "variables": sp.symbols("n_1 n_2 n_3", real=True),
            "envelope": None,
            "prior_points": prior_points,
            "target": target,
            "record": base_record
            | {
                "bounded_class_exhausted": True,
                "repair_constructed": False,
                "zero_constraints_cover_all_twelve_predecessor_frames": True,
            },
        }
    degree, monomials, coefficients = selected
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    envelope = sp.factor(
        sum(
            coefficients[index] * n1**a * n2**b * n3**c for index, (a, b, c) in enumerate(monomials)
        )
    )
    variables = (n1, n2, n3)
    if (
        any(envelope.subs(dict(zip(variables, point, strict=True))) != 0 for point in prior_points)
        or envelope.subs(dict(zip(variables, target, strict=True))) != 1
    ):
        raise RevisedTwelveFrameHeightTwoRationalError("normalized envelope mismatch")
    support = [index for index, value in enumerate(coefficients) if value != 0]
    formula = (
        sp.sstr(sp.expand(envelope)).replace("n_1", "n1").replace("n_2", "n2").replace("n_3", "n3")
    )
    return {
        "variables": variables,
        "envelope": envelope,
        "prior_points": prior_points,
        "target": target,
        "record": base_record
        | {
            "minimal_feasible_even_degree": degree,
            "deterministic_solver": "RREF with all free coefficients set to zero",
            "deterministic_support_indices": support,
            "deterministic_support_size": len(support),
            "deterministic_envelope": f"a_next(n)={formula}",
            "value_at_counterexample": "1",
            "zero_at_all_twelve_predecessor_frames": True,
            "bounded_class_exhausted": False,
        },
    }


def _exact_result(root: Path, degree_six_artifact: Mapping[str, Any]) -> dict[str, Any]:
    extensions, artifacts = _predecessor_chain(root, degree_six_artifact)
    atlas = rational_campaign._atlas()
    rotation, direction, payload, evaluations = common._frame_payload(
        atlas,
        (sp.Integer(1), sp.Integer(-2)),
        "primary_stereographic_1_minus_2_revised_twelve_frame",
        artifacts["fourth_campaign"],
    )
    expected_direction = [sp.Rational(-2, 3), sp.Rational(1, 3), sp.Rational(-2, 3)]
    if direction != expected_direction or evaluations != 15:
        raise RevisedTwelveFrameHeightTwoRationalError("second height-two selector mismatch")
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
    rows: list[dict[str, Any]] = []
    payloads: list[tuple[str, sp.Matrix, sp.Expr]] = []
    targets: list[sp.Matrix] = []
    for candidate in artifacts["minimal_escape"]["exact_escape"]["candidate_classification"]:
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
    if len(rows) != EXPECTED_CANDIDATES or len(target_hashes) != 1:
        raise RevisedTwelveFrameHeightTwoRationalError("candidate classification mismatch")
    target = targets[0]
    obstructed = all(not row["D4_Sylvester_solvable"] for row in rows)
    compatible = all(row["D4_Sylvester_solvable"] for row in rows)
    if not (obstructed or compatible):
        raise RevisedTwelveFrameHeightTwoRationalError("mixed candidate verdict outside class")
    envelope = _normalized_envelope(tuple(direction))
    range_record: dict[str, Any] = {"evaluated_after_obstruction": obstructed}
    if compatible:
        repair: dict[str, Any] = {
            "classification": "not_required_all_candidates_compatible",
            "local_certificate_constructed": False,
            "total_local_direction_certificates": 12,
        }
    else:
        covectors, coordinates, target_coordinates = _zero_speed_coordinates(projector0, -target)
        selector = (
            coordinates * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
        ).applyfunc(sp.factor)
        quotient = sp.Matrix.hstack(*selector.T.nullspace()).T
        quotient_target = (quotient * target_coordinates * quotient.T).applyfunc(sp.factor)
        target_plane = sp.Matrix.hstack(*target_coordinates.columnspace())
        selector_basis = sp.Matrix.hstack(*selector.columnspace())
        intersection = (
            selector.rank() + target_plane.rank() - selector_basis.row_join(target_plane).rank()
        )
        range_record = {
            "evaluated_after_obstruction": True,
            "transverse_selector_rank": selector.rank(),
            "target_plane_dimension": target_plane.rank(),
            "selector_target_plane_intersection_dimension": intersection,
            "quotient_target_zero": quotient_target.is_zero_matrix,
        }
        if envelope["envelope"] is None or not quotient_target.is_zero_matrix:
            repair = {
                "classification": "bounded_preregistered_class_exhausted",
                "transverse_curl_target_range_compatible": quotient_target.is_zero_matrix,
                "preserving_envelope_exists_in_declared_class": envelope["envelope"] is not None,
                "local_certificate_constructed": False,
                "total_local_direction_certificates": 12,
            }
        else:
            terms = predecessor.prior._bounded_transverse_curl_decompose(
                target_coordinates, selector, covectors, reference
            )
            if len(terms) > REPAIR_CLASS["maximum_elementary_curl_channels"]:
                repair = {
                    "classification": "bounded_preregistered_class_exhausted",
                    "transverse_curl_target_range_compatible": True,
                    "preserving_envelope_exists_in_declared_class": True,
                    "required_elementary_curl_channels": len(terms),
                    "local_certificate_constructed": False,
                    "total_local_direction_certificates": 12,
                }
            else:
                aligned_block = sum(
                    (term["output_aligned"] * term["right_aligned"].T for term in terms),
                    sp.zeros(STATE_DIMENSION),
                ).applyfunc(sp.factor)
                angular = degree_five._angular_extension(
                    terms, state_rotation, tuple(direction), envelope
                )
                correction_skew = (
                    reference["energy0"] * aligned_block - aligned_block.T * reference["energy0"]
                ).applyfunc(sp.factor)
                corrected = []
                for candidate_id, before, eta in payloads:
                    residual = (before + eta * correction_skew).applyfunc(sp.factor)
                    solvable, nonzero = _solve(residual)
                    corrected.append(
                        {
                            "candidate_id": candidate_id,
                            "D4_Sylvester_solvable": solvable,
                            "nonzero_equal_eigenspace_compressions": nonzero,
                            "corrected_residual_sha256": _content_hash(_matrix_payload(residual)),
                        }
                    )
                if (
                    intersection != target.rank()
                    or angular["gradient_residual"].is_zero_matrix is not True
                    or any(not row["D4_Sylvester_solvable"] for row in corrected)
                ):
                    raise RevisedTwelveFrameHeightTwoRationalError("bounded repair mismatch")
                repair = {
                    "classification": "bounded_local_repair_constructed",
                    "transverse_curl_target_range_compatible": True,
                    "preserving_envelope_exists_in_declared_class": True,
                    "local_certificate_constructed": True,
                    "constructed_completion_rank": aligned_block.rank(),
                    "elementary_curl_channels": len(terms),
                    "coordinate_pairs": [term["coordinate_pair"] for term in terms],
                    "extension_sha256": _content_hash(_matrix_payload(angular["extension"])),
                    "gradient_residual_zero": True,
                    "candidate_records": corrected,
                    "total_local_direction_certificates": 13,
                }
    return {
        "atlas": atlas["record"],
        "search_protocol": {
            "deterministic_selector_rule": (
                "primitive primary-chart integer pairs by Chebyshev height; within absolute "
                "pattern (1,2), positive-before-negative signs"
            ),
            "preregistered_points": SELECTOR,
            "points_evaluated": 1,
            "stopped_after_first_exact_point_and_class": True,
        },
        "selector": {
            "chart": "primary_e1_stereographic",
            "chart_coordinates": ["1", "-2"],
            "chart_denominator_value": "6",
            "direction": [str(value) for value in direction],
            "regular_real_chart_point": True,
        },
        "full_recurrence": {
            "orders_checked": [1, 2, 3, 4],
            "directional_polarization_evaluations": evaluations,
            "candidate_fourth_order_systems": len(rows),
            "revised_global_symbol_rank": global_symbol.rank(),
            "revised_global_symbol_sha256": _content_hash(_matrix_payload(global_symbol)),
            "revised_aligned_symbol_sha256": _content_hash(_matrix_payload(aligned)),
        },
        "exact_rational_classification": {
            "candidate_conditions_checked": len(rows),
            "candidate_compatibilities": sum(row["D4_Sylvester_solvable"] for row in rows),
            "candidate_obstructions": sum(not row["D4_Sylvester_solvable"] for row in rows),
            "distinct_eta_normalized_targets": len(target_hashes),
            "eta_normalized_target_rank": target.rank(),
            "eta_normalized_target_nonzero_entries": sum(value != 0 for value in target),
            "eta_normalized_target_sha256": next(iter(target_hashes)),
            "candidate_records": rows,
        },
        "bounded_classification": {
            "preregistered_repair_class": REPAIR_CLASS,
            "range": range_record,
            "envelope": envelope["record"],
            "repair": repair,
        },
    }


def _counts(exact: Mapping[str, Any]) -> dict[str, int]:
    classification = exact["exact_rational_classification"]
    repair = exact["bounded_classification"]["repair"]
    new_certificate = int(repair.get("local_certificate_constructed") is True)
    return {
        "bound_predecessors": 1,
        "candidate_compatibilities": classification["candidate_compatibilities"],
        "candidate_conditions_checked": classification["candidate_conditions_checked"],
        "candidate_obstructions": classification["candidate_obstructions"],
        "directional_polarization_evaluations": 15,
        "envelope_degrees_checked": 4,
        "inferred_global_passes": 0,
        "inherited_bound_artifacts_verified": 11,
        "negative_controls": 8,
        "new_local_direction_certificates": new_certificate,
        "prior_direction_constraints": 12,
        "rational_SO3_charts": 2,
        "recurrence_orders_checked": 4,
        "regular_search_points_evaluated": 1,
        "total_local_direction_certificates": repair["total_local_direction_certificates"],
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("search_schedule") != SELECTOR
        or config.get("bounded_envelope_degrees") != ENVELOPE_DEGREES
        or config.get("degree_six_full_support_ceiling") != 28
        or config.get("bounded_repair_class") != REPAIR_CLASS
        or set(config)
        != {
            "schema_version",
            "global_claim_policy",
            "search_schedule",
            "bounded_envelope_degrees",
            "degree_six_full_support_ceiling",
            "bounded_repair_class",
            "degree_six_predecessor",
            "campaign_source",
            "campaign_test",
            "content_sha256",
        }
    ):
        raise RevisedTwelveFrameHeightTwoRationalError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    degree_six_artifact = _load_bound(root, config["degree_six_predecessor"])
    exact = _exact_result(root, degree_six_artifact)
    classification = exact["exact_rational_classification"]
    repair = exact["bounded_classification"]["repair"]
    true_claims = set(TRUE_CLAIMS)
    false_claims = set(FALSE_CLAIMS)
    if classification["candidate_obstructions"] == EXPECTED_CANDIDATES:
        true_claims |= {
            "all_12_candidates_obstructed_at_second_height_two_point",
            "revised_twelve_frame_symbol_compatibility_disproved_at_selector",
        }
    else:
        true_claims.add("all_12_candidates_compatible_at_second_height_two_point")
    if repair.get("local_certificate_constructed") is True:
        true_claims.add("bounded_local_repair_constructed")
    else:
        false_claims.add("bounded_local_repair_constructed")
        if repair.get("classification") == "bounded_preregistered_class_exhausted":
            true_claims.add("bounded_preregistered_repair_class_exhausted")
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_file_sha256": _file_sha256(config_path.resolve().read_bytes()),
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key]
            for key in ("degree_six_predecessor", "campaign_source", "campaign_test")
        },
        "counts": _counts(exact),
        "exact_gate": exact,
        "claims": {key: True for key in true_claims} | {key: False for key in false_claims},
        "negative_controls": EXPECTED_NEGATIVE_CONTROLS,
        "scope": EXPECTED_SCOPE,
        "next_gate": EXPECTED_NEXT_GATE,
        "errors": [],
    }
    return _with_hash(artifact)


def _keys(value: object, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def validate_campaign(document: Mapping[str, Any]) -> None:
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / CONFIG_PATH
    config, config_data = _load_file(config_path)
    source_file_sha256 = _file_sha256((project_root / SOURCE_PATH).read_bytes())
    test_file_sha256 = _file_sha256((project_root / TEST_PATH).read_bytes())
    gate = document.get("exact_gate", {})
    search = gate.get("search_protocol", {})
    selector = gate.get("selector", {})
    recurrence = gate.get("full_recurrence", {})
    classification = gate.get("exact_rational_classification", {})
    bounded = gate.get("bounded_classification", {})
    envelope = bounded.get("envelope", {})
    repair = bounded.get("repair", {})
    claims = document.get("claims", {})
    counts = document.get("counts", {})
    bindings = document.get("source_bindings", {})
    predecessor_binding = bindings.get("degree_six_predecessor", {})
    source_binding = bindings.get("campaign_source", {})
    test_binding = bindings.get("campaign_test", {})
    expected_claim_keys = (
        TRUE_CLAIMS
        | FALSE_CLAIMS
        | {
            "all_12_candidates_obstructed_at_second_height_two_point",
            "revised_twelve_frame_symbol_compatibility_disproved_at_selector",
            "bounded_local_repair_constructed",
        }
    )
    expected_counts = {
        "bound_predecessors": 1,
        "candidate_compatibilities": 0,
        "candidate_conditions_checked": 12,
        "candidate_obstructions": 12,
        "directional_polarization_evaluations": 15,
        "envelope_degrees_checked": 4,
        "inferred_global_passes": 0,
        "inherited_bound_artifacts_verified": 11,
        "negative_controls": 8,
        "new_local_direction_certificates": 1,
        "prior_direction_constraints": 12,
        "rational_SO3_charts": 2,
        "recurrence_orders_checked": 4,
        "regular_search_points_evaluated": 1,
        "total_local_direction_certificates": 13,
    }
    candidate_records = classification.get("candidate_records", [])
    corrected_records = repair.get("candidate_records", [])
    predecessor_path = project_root / str(predecessor_binding.get("path", ""))
    predecessor_document: Mapping[str, Any] = {}
    predecessor_data = b""
    if predecessor_path.is_file() and project_root in predecessor_path.resolve().parents:
        predecessor_document, predecessor_data = _load_file(predecessor_path)
    predecessor_validation_failed = False
    try:
        predecessor.validate_campaign(predecessor_document)
    except (KeyError, TypeError, ValueError):
        predecessor_validation_failed = True
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or set(document) != EXPECTED_DOCUMENT_KEYS
        or document.get("errors") != []
        or document.get("scope") != EXPECTED_SCOPE
        or document.get("next_gate") != EXPECTED_NEXT_GATE
        or document.get("negative_controls") != EXPECTED_NEGATIVE_CONTROLS
        or document.get("config_file_sha256") != _file_sha256(config_data)
        or not _hash_matches(config)
        or set(config)
        != {
            "schema_version",
            "global_claim_policy",
            "search_schedule",
            "bounded_envelope_degrees",
            "degree_six_full_support_ceiling",
            "bounded_repair_class",
            "degree_six_predecessor",
            "campaign_source",
            "campaign_test",
            "content_sha256",
        }
        or config.get("schema_version") != CONFIG_SCHEMA
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("search_schedule") != SELECTOR
        or config.get("bounded_envelope_degrees") != ENVELOPE_DEGREES
        or config.get("degree_six_full_support_ceiling") != 28
        or config.get("bounded_repair_class") != REPAIR_CLASS
        or document.get("config_sha256") != config.get("content_sha256")
        or counts != expected_counts
        or set(bindings) != {"degree_six_predecessor", "campaign_source", "campaign_test"}
        or set(predecessor_binding) != {"path", "file_sha256", "content_sha256"}
        or set(source_binding) != {"path", "file_sha256"}
        or set(test_binding) != {"path", "file_sha256"}
        or predecessor_binding.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or predecessor_binding.get("file_sha256")
        != "ad5a6094aa963c8ce855d6954b32ac63ebb89e349128a258102b4e1b3d66a4c7"
        or predecessor_binding.get("path")
        != (
            "runs/physics-language/quartic-tc2-d4-revised-eleven-frame-degree-six-"
            "envelope-gate/campaign.json"
        )
        or _file_sha256(predecessor_data) != predecessor_binding.get("file_sha256")
        or predecessor_document.get("content_sha256") != predecessor_binding.get("content_sha256")
        or not _hash_matches(predecessor_document)
        or predecessor_validation_failed
        or source_binding.get("path") != SOURCE_PATH
        or source_binding.get("file_sha256") != source_file_sha256
        or test_binding.get("path") != TEST_PATH
        or test_binding.get("file_sha256") != test_file_sha256
        or config.get("degree_six_predecessor") != predecessor_binding
        or config.get("campaign_source") != source_binding
        or config.get("campaign_test") != test_binding
        or not _keys(
            gate,
            {
                "atlas",
                "search_protocol",
                "selector",
                "full_recurrence",
                "exact_rational_classification",
                "bounded_classification",
            },
        )
        or search.get("preregistered_points") != SELECTOR
        or search.get("points_evaluated") != 1
        or search.get("stopped_after_first_exact_point_and_class") is not True
        or selector.get("chart_coordinates") != ["1", "-2"]
        or selector.get("direction") != ["-2/3", "1/3", "-2/3"]
        or selector.get("chart_denominator_value") != "6"
        or selector.get("regular_real_chart_point") is not True
        or recurrence.get("orders_checked") != [1, 2, 3, 4]
        or recurrence.get("directional_polarization_evaluations") != 15
        or recurrence.get("candidate_fourth_order_systems") != 12
        or classification.get("candidate_conditions_checked") != 12
        or classification.get("candidate_compatibilities") != 0
        or classification.get("candidate_obstructions") != 12
        or classification.get("distinct_eta_normalized_targets") != 1
        or classification.get("eta_normalized_target_rank") != 4
        or classification.get("eta_normalized_target_nonzero_entries") != 56
        or classification.get("eta_normalized_target_sha256") != EXPECTED_TARGET_SHA256
        or len(candidate_records) != 12
        or any(
            not _keys(
                row,
                {
                    "candidate_id",
                    "D4_Sylvester_solvable",
                    "nonzero_equal_eigenspace_compressions",
                    "zero_speed_cleared_numerator",
                },
            )
            or row.get("D4_Sylvester_solvable") is not False
            for row in candidate_records
        )
        or bounded.get("preregistered_repair_class") != REPAIR_CLASS
        or envelope.get("declared_degree_ladder") != ENVELOPE_DEGREES
        or envelope.get("degree_six_full_support_ceiling") != 28
        or envelope.get("minimal_feasible_even_degree") != 4
        or envelope.get("deterministic_support_indices") != [1, 2, 3, 5, 6, 7, 8, 9, 10, 11]
        or envelope.get("deterministic_support_size") != 10
        or envelope.get("zero_at_all_twelve_predecessor_frames") is not True
        or envelope.get("bounded_class_exhausted") is not False
        or bounded.get("range", {}).get("evaluated_after_obstruction") is not True
        or bounded.get("range", {}).get("quotient_target_zero") is not True
        or bounded.get("range", {}).get("transverse_selector_rank") != 22
        or bounded.get("range", {}).get("target_plane_dimension") != 4
        or bounded.get("range", {}).get("selector_target_plane_intersection_dimension") != 4
        or repair.get("classification") != "bounded_local_repair_constructed"
        or repair.get("transverse_curl_target_range_compatible") is not True
        or repair.get("preserving_envelope_exists_in_declared_class") is not True
        or repair.get("local_certificate_constructed") is not True
        or repair.get("gradient_residual_zero") is not True
        or repair.get("total_local_direction_certificates") != 13
        or repair.get("constructed_completion_rank") != 2
        or repair.get("elementary_curl_channels") != 2
        or repair.get("coordinate_pairs") != [[11, 21], [15, 32]]
        or repair.get("extension_sha256") != EXPECTED_EXTENSION_SHA256
        or len(corrected_records) != 12
        or any(
            not _keys(
                row,
                {
                    "candidate_id",
                    "D4_Sylvester_solvable",
                    "nonzero_equal_eigenspace_compressions",
                    "corrected_residual_sha256",
                },
            )
            or row.get("D4_Sylvester_solvable") is not True
            for row in corrected_records
        )
        or set(claims) != expected_claim_keys
        or any(claims.get(key) is not True for key in expected_claim_keys - FALSE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or _content_hash(gate) != EXPECTED_EXACT_GATE_SHA256
    ):
        raise RevisedTwelveFrameHeightTwoRationalError(
            "revised-twelve-frame height-two gate validation failed"
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
            {"status": artifact["status"], "content_sha256": artifact["content_sha256"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
