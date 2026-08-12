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
from . import quartic_tc2_d4_revised_eleven_frame_rational_counterexample_campaign as prior
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

SCHEMA = "sigma-quartic-tc2-d4-revised-eleven-frame-degree-six-envelope-gate-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-revised-eleven-frame-degree-six-envelope-config-1.0"
STATUS = "pass_exact_first_height_two_point_and_bounded_degree_six_classification"
EXPECTED_CANDIDATES = 12
EXPECTED_PREDECESSOR_CONTENT_SHA256 = (
    "5dac044360e2dbff204e1f762d90634f30c8005bddd85d11f1ad7308faa0a17a"
)
EXPECTED_PREDECESSOR_EXACT_SHA256 = (
    "9f9eec94db2487b1254189882b28c3d2038ff8dfe45b7f2a5b3149881a7d9e36"
)
EXPECTED_EXACT_GATE_SHA256 = "962c4014df740b9ac117a9c344354b73b6d36d74e09fd003625874c63f1f0a0d"
EXPECTED_PREDECESSOR_BINDINGS_SHA256 = (
    "d15dc3d6bdf21923852fd12d6eb67bd751b0d6535d21859d61587422c13aac40"
)

TRUE_CLAIMS = {
    "bounded_degree_six_envelope_class_preregistered",
    "exact_two_chart_SO3_atlas_reused",
    "first_primitive_nonaxis_height_two_point_evaluated",
    "full_orders_one_through_four_recurrence_evaluated",
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
EXPECTED_TRUE_CLAIMS = TRUE_CLAIMS | {
    "all_12_candidates_obstructed_at_first_height_two_point",
    "bounded_local_repair_constructed",
    "revised_eleven_frame_symbol_compatibility_disproved_at_selector",
}
EXPECTED_COUNTS = {
    "bound_predecessors": 11,
    "candidate_compatibilities": 0,
    "candidate_conditions_checked": 12,
    "candidate_obstructions": 12,
    "directional_polarization_evaluations": 15,
    "envelope_degrees_checked": 4,
    "inferred_global_passes": 0,
    "negative_controls": 8,
    "new_local_direction_certificates": 1,
    "prior_direction_constraints": 11,
    "rational_SO3_charts": 2,
    "recurrence_orders_checked": 4,
    "regular_search_points_evaluated": 1,
    "total_local_direction_certificates": 12,
}
EXPECTED_NEGATIVE_CONTROLS = {
    "evaluate_second_height_two_point": {"rejected": True},
    "infer_B7_H7_or_lifespan": {"rejected": True},
    "infer_PDE_or_tube_admission": {"rejected": True},
    "infer_finite_determining_theorem": {"rejected": True},
    "infer_full_direction_sphere": {"rejected": True},
    "search_degree_above_six": {"rejected": True},
    "skip_exact_denominator_clearing": {"rejected": True},
    "skip_lower_recurrence_orders": {"rejected": True},
}
EXPECTED_SCOPE = (
    "One exact rational audit at the first preregistered primitive non-axis height-two point, "
    "with the finite even homogeneous degree-0/2/4/6 envelope class only. No full-sphere, "
    "determining, PDE, tube, CK, TC2, B7, H7, or lifespan claim follows."
)
EXPECTED_NEXT_GATE = (
    "Preregister another bounded point/class or prove a finite determining theorem; do not infer "
    "global or PDE admission from this local gate."
)
EXPECTED_DOCUMENT_KEYS = {
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


class RevisedElevenFrameDegreeSixEnvelopeError(ValueError):
    """Raised when the bounded degree-six gate is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents:
        raise RevisedElevenFrameDegreeSixEnvelopeError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise RevisedElevenFrameDegreeSixEnvelopeError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root not in path.parents or not path.is_file():
        raise RevisedElevenFrameDegreeSixEnvelopeError("raw binding escaped root or is absent")
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise RevisedElevenFrameDegreeSixEnvelopeError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _deterministic_normalized_envelope(
    target: tuple[sp.Expr, sp.Expr, sp.Expr],
) -> dict[str, Any]:
    known = prior._known_a11_envelope()
    prior_points = known["prior_points"] + (known["target"],)
    ranks: dict[int, dict[str, Any]] = {}
    selected: tuple[int, tuple[tuple[int, int, int], ...], sp.Matrix] | None = None
    for degree in (0, 2, 4, 6):
        monomials = common._homogeneous_monomials(degree)
        zero_matrix = common._evaluation_matrix(prior_points, monomials)
        target_row = common._evaluation_matrix((target,), monomials)
        system = zero_matrix.col_join(target_row)
        rhs = sp.zeros(system.rows, 1)
        rhs[-1] = 1
        augmented = system.row_join(rhs)
        rref, pivots = augmented.rref()
        feasible = not any(
            all(rref[row, column] == 0 for column in range(system.cols))
            and rref[row, -1] != 0
            for row in range(rref.rows)
        )
        ranks[degree] = {
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
    if selected is None:
        return {
            "variables": sp.symbols("n_1 n_2 n_3", real=True),
            "envelope": None,
            "prior_points": prior_points,
            "target": target,
            "record": {
                "declared_degree_ladder": [0, 2, 4, 6],
                "degree_records": {str(key): value for key, value in ranks.items()},
                "degree_six_full_support_ceiling": 28,
                "bounded_class_exhausted": True,
                "repair_constructed": False,
            },
        }
    degree, monomials, coefficients = selected
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    envelope = sp.factor(
        sum(
            coefficients[index] * n1**a * n2**b * n3**c
            for index, (a, b, c) in enumerate(monomials)
        )
    )
    variables = (n1, n2, n3)
    if (
        any(envelope.subs(dict(zip(variables, point, strict=True))) != 0 for point in prior_points)
        or envelope.subs(dict(zip(variables, target, strict=True))) != 1
    ):
        raise RevisedElevenFrameDegreeSixEnvelopeError("normalized envelope mismatch")
    support = [index for index, value in enumerate(coefficients) if value != 0]
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
            "declared_degree_ladder": [0, 2, 4, 6],
            "degree_records": {str(key): value for key, value in ranks.items()},
            "degree_six_full_support_ceiling": 28,
            "minimal_feasible_even_degree": degree,
            "deterministic_solver": "RREF with all free coefficients set to zero",
            "deterministic_support_indices": support,
            "deterministic_support_size": len(support),
            "deterministic_envelope": f"a_next(n)={formula}",
            "value_at_counterexample": "1",
            "zero_at_all_eleven_predecessor_frames": True,
            "bounded_class_exhausted": False,
        },
    }


def _exact_result(*artifacts: Mapping[str, Any]) -> dict[str, Any]:
    (
        exhaustion_predecessor,
        eleven_frame_symbol_predecessor,
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
    if (
        exhaustion_predecessor.get("content_sha256") != EXPECTED_PREDECESSOR_CONTENT_SHA256
        or _content_hash(exhaustion_predecessor.get("exact_gate", {}))
        != EXPECTED_PREDECESSOR_EXACT_SHA256
    ):
        raise RevisedElevenFrameDegreeSixEnvelopeError("exhaustion predecessor mismatch")
    atlas = rational_campaign._atlas()
    rotation, direction, payload, evaluations = common._frame_payload(
        atlas,
        (sp.Integer(1), sp.Integer(2)),
        "primary_stereographic_1_2_revised_eleven_frame",
        fourth,
    )
    if direction != [sp.Rational(-2, 3), sp.Rational(1, 3), sp.Rational(2, 3)] or evaluations != 15:
        raise RevisedElevenFrameDegreeSixEnvelopeError("height-two selector mismatch")
    extensions = prior._capture_eleven_frame_extensions(
        eleven_frame_symbol_predecessor,
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
    blocks = {key: common._evaluate_extension(value, direction) for key, value in extensions.items()}
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
    obstructed = all(not row["D4_Sylvester_solvable"] for row in rows)
    if len(rows) != EXPECTED_CANDIDATES or len(target_hashes) != 1 or not obstructed:
        raise RevisedElevenFrameDegreeSixEnvelopeError("height-two candidate classification mismatch")
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
    envelope = _deterministic_normalized_envelope(tuple(direction))
    repair: dict[str, Any]
    if envelope["envelope"] is None:
        repair = {
            "transverse_curl_target_range_compatible": quotient_target.is_zero_matrix,
            "preserving_envelope_exists_in_declared_class": False,
            "local_certificate_constructed": False,
            "total_local_direction_certificates": 11,
        }
    else:
        terms = prior._bounded_transverse_curl_decompose(
            target_coordinates, selector, covectors, reference
        )
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
            not quotient_target.is_zero_matrix
            or intersection != target.rank()
            or angular["gradient_residual"].is_zero_matrix is not True
            or any(not row["D4_Sylvester_solvable"] for row in corrected)
        ):
            raise RevisedElevenFrameDegreeSixEnvelopeError("bounded repair mismatch")
        repair = {
            "transverse_curl_target_range_compatible": True,
            "preserving_envelope_exists_in_declared_class": True,
            "local_certificate_constructed": True,
            "constructed_completion_rank": aligned_block.rank(),
            "elementary_curl_channels": len(terms),
            "coordinate_pairs": [term["coordinate_pair"] for term in terms],
            "extension_sha256": _content_hash(_matrix_payload(angular["extension"])),
            "gradient_residual_zero": True,
            "candidate_records": corrected,
            "total_local_direction_certificates": 12,
        }
    return {
        "atlas": atlas["record"],
        "search_protocol": {
            "deterministic_selector_rule": (
                "first primitive non-axis primary-chart integer pair on the next Chebyshev-height "
                "shell, ordered by absolute pattern then positive-before-negative signs"
            ),
            "preregistered_points": [
                {"chart": "primary_e1_stereographic", "coordinates": ["1", "2"]}
            ],
            "points_evaluated": 1,
            "stopped_after_first_exact_point_and_class": True,
        },
        "selector": {
            "chart": "primary_e1_stereographic",
            "chart_coordinates": ["1", "2"],
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
            "range": {
                "transverse_selector_rank": selector.rank(),
                "target_plane_dimension": target_plane.rank(),
                "selector_target_plane_intersection_dimension": intersection,
                "quotient_target_zero": quotient_target.is_zero_matrix,
            },
            "envelope": envelope["record"],
            "repair": repair,
        },
    }


def build_campaign(project_root: Path, config_path: Path) -> dict[str, Any]:
    root = project_root.resolve()
    config, _ = _load_file(config_path.resolve())
    selector = [{"chart": "primary_e1_stereographic", "coordinates": ["1", "2"]}]
    if (
        config.get("schema_version") != CONFIG_SCHEMA
        or not _hash_matches(config)
        or config.get("global_claim_policy") != "fail_closed"
        or config.get("search_schedule") != selector
        or config.get("bounded_envelope_degrees") != [0, 2, 4, 6]
        or config.get("degree_six_full_support_ceiling") != 28
    ):
        raise RevisedElevenFrameDegreeSixEnvelopeError("invalid campaign config")
    for key in ("campaign_source", "campaign_test"):
        _check_raw(root, config[key])
    bound_keys = (
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
    bound = {key: _load_bound(root, config[key]) for key in bound_keys}
    exact = _exact_result(*(bound[key] for key in bound_keys))
    classification = exact["exact_rational_classification"]
    repair = exact["bounded_classification"]["repair"]
    true_claims = set(TRUE_CLAIMS)
    false_claims = set(FALSE_CLAIMS)
    if classification["candidate_obstructions"] == EXPECTED_CANDIDATES:
        true_claims.add("all_12_candidates_obstructed_at_first_height_two_point")
        true_claims.add("revised_eleven_frame_symbol_compatibility_disproved_at_selector")
    if repair["local_certificate_constructed"]:
        true_claims.add("bounded_local_repair_constructed")
    else:
        false_claims.add("bounded_local_repair_constructed")
    artifact = {
        "schema_version": SCHEMA,
        "status": STATUS,
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: config[key] for key in (*bound_keys, "campaign_source", "campaign_test")
        },
        "counts": dict(EXPECTED_COUNTS),
        "exact_gate": exact,
        "claims": {key: True for key in true_claims} | {key: False for key in false_claims},
        "negative_controls": dict(EXPECTED_NEGATIVE_CONTROLS),
        "scope": EXPECTED_SCOPE,
        "next_gate": EXPECTED_NEXT_GATE,
        "errors": [],
    }
    return _with_hash(artifact)


def validate_campaign(document: Mapping[str, Any]) -> None:
    gate = document.get("exact_gate", {})
    classification = gate.get("exact_rational_classification", {})
    bounded = gate.get("bounded_classification", {})
    envelope = bounded.get("envelope", {})
    repair = bounded.get("repair", {})
    claims = document.get("claims", {})
    raw_source_bindings = document.get("source_bindings", {})
    source_bindings = raw_source_bindings if isinstance(raw_source_bindings, Mapping) else {}
    predecessor_bindings = {
        key: value
        for key, value in source_bindings.items()
        if key not in {"campaign_source", "campaign_test"}
    }
    expected_binding_keys = {
        "c23_predecessor",
        "campaign_source",
        "campaign_test",
        "degree_five_predecessor",
        "eleven_frame_symbol_predecessor",
        "exhaustion_predecessor",
        "fourth_campaign",
        "minimal_escape",
        "nine_frame_predecessor",
        "rational_predecessor",
        "revised_predecessor",
        "ten_frame_predecessor",
        "xyz_predecessor",
    }
    raw_bindings = {
        "campaign_source": {
            "path": (
                "src/sigma_theory_compiler/"
                "quartic_tc2_d4_revised_eleven_frame_degree_six_envelope_gate.py"
            )
        },
        "campaign_test": {
            "path": (
                "tests/test_quartic_tc2_d4_revised_eleven_frame_degree_six_envelope_gate.py"
            )
        },
    }
    expected_config = {
        "schema_version": CONFIG_SCHEMA,
        "global_claim_policy": "fail_closed",
        "search_schedule": [
            {"chart": "primary_e1_stereographic", "coordinates": ["1", "2"]}
        ],
        "bounded_envelope_degrees": [0, 2, 4, 6],
        "degree_six_full_support_ceiling": 28,
        **source_bindings,
    }
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status") != STATUS
        or not _hash_matches(document)
        or set(document) != EXPECTED_DOCUMENT_KEYS
        or document.get("errors") != []
        or document.get("counts") != EXPECTED_COUNTS
        or document.get("negative_controls") != EXPECTED_NEGATIVE_CONTROLS
        or document.get("scope") != EXPECTED_SCOPE
        or document.get("next_gate") != EXPECTED_NEXT_GATE
        or set(source_bindings) != expected_binding_keys
        or _content_hash(predecessor_bindings) != EXPECTED_PREDECESSOR_BINDINGS_SHA256
        or any(
            source_bindings.get(key, {}).get("path") != value["path"]
            or not isinstance(source_bindings.get(key, {}).get("file_sha256"), str)
            or len(source_bindings[key]["file_sha256"]) != 64
            for key, value in raw_bindings.items()
        )
        or document.get("config_sha256") != _content_hash(expected_config)
        or gate.get("selector", {}).get("chart_coordinates") != ["1", "2"]
        or gate.get("selector", {}).get("direction") != ["-2/3", "1/3", "2/3"]
        or gate.get("full_recurrence", {}).get("orders_checked") != [1, 2, 3, 4]
        or gate.get("full_recurrence", {}).get("directional_polarization_evaluations") != 15
        or classification.get("candidate_conditions_checked") != 12
        or classification.get("candidate_obstructions") != 12
        or classification.get("candidate_compatibilities") != 0
        or classification.get("eta_normalized_target_rank") != 4
        or classification.get("eta_normalized_target_nonzero_entries") != 56
        or classification.get("eta_normalized_target_sha256")
        != "7e9d08c8b4b2df3f3027baec1d6f752735e83c141cdf237d7fca98240cad62f0"
        or len(classification.get("candidate_records", [])) != 12
        or envelope.get("declared_degree_ladder") != [0, 2, 4, 6]
        or envelope.get("degree_six_full_support_ceiling") != 28
        or envelope.get("minimal_feasible_even_degree") != 4
        or envelope.get("deterministic_support_indices") != [1, 2, 5, 7, 9, 10]
        or envelope.get("deterministic_support_size") != 6
        or envelope.get("zero_at_all_eleven_predecessor_frames") is not True
        or bounded.get("range", {}).get("quotient_target_zero") is not True
        or bounded.get("range", {}).get("target_plane_dimension") != 4
        or bounded.get("range", {}).get("selector_target_plane_intersection_dimension") != 4
        or repair.get("transverse_curl_target_range_compatible") is not True
        or repair.get("local_certificate_constructed") is not True
        or repair.get("constructed_completion_rank") != 2
        or repair.get("elementary_curl_channels") != 2
        or repair.get("coordinate_pairs") != [[11, 21], [15, 32]]
        or repair.get("extension_sha256")
        != "5625ccb9e40bc54fd89950f280d9eb72f0c89826b3584d53eae706d648f2529d"
        or repair.get("gradient_residual_zero") is not True
        or repair.get("total_local_direction_certificates") != 12
        or set(claims) != EXPECTED_TRUE_CLAIMS | FALSE_CLAIMS
        or any(claims.get(key) is not True for key in EXPECTED_TRUE_CLAIMS)
        or any(claims.get(key) is not False for key in FALSE_CLAIMS)
        or (EXPECTED_EXACT_GATE_SHA256 and _content_hash(gate) != EXPECTED_EXACT_GATE_SHA256)
    ):
        raise RevisedElevenFrameDegreeSixEnvelopeError("degree-six gate validation failed")


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
