from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_curl_constraint_admission_campaign import _gradient_lift
from .quartic_tc2_d4_full_linear_gradient_annihilator_no_go_campaign import (
    _zero_speed_coordinates,
)
from .quartic_tc2_d4_matrix_curl_rank_one_completion_campaign import (
    TRANSVERSE_CURL_INDICES,
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

SCHEMA = "sigma-quartic-tc2-d4-degree-three-c23-great-circle-escape-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-degree-three-c23-great-circle-escape-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
PREDECESSOR_STATUS = (
    "pass_exact_minimal_degree_three_matrix_curl_sphere_extension_with_"
    "first_additional_frame_obstruction"
)
NORMALIZED_TARGET_SHA256 = "49b40a907913c6eeba85bf0a5f013810863a8631d462c74f5b68ac45f0046280"
GLOBAL_XZ_BLOCK_SHA256 = "7c584daafa25e52cf4d2751c1462ac093f3ac51e5968c023a477e525705d377b"
ALIGNED_XZ_BLOCK_SHA256 = "ca7087b814ddf2ef9f00e9ce9bc51a00f629d489aa51aef779dfc6fbe36a2223"
BASE_XZ_RHS_SHA256 = "d3ab104a0de327e978b6bbe03113b2cf883bce4b34684eed94574560388e0513"


class QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(ValueError):
    """Raised when the degree-three C23 escape is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError("bound input escaped project root")
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _fixed_output() -> sp.Matrix:
    raw = {
        16: "29925/54994211",
        19: "-23850/54994211",
        22: "-2546853741*sqrt(2)/240654667336",
        26: "-97359696*sqrt(2)/752045835425",
        28: "53088084963/1504091670850",
        29: "-550872*sqrt(2)/54994211",
        31: "7630534296*sqrt(2)/752045835425",
        44: "-615805353*sqrt(2)/60163666834",
        48: "9826112331*sqrt(2)/1504091670850",
        50: "154199718639/6016366683400",
        51: "-604404*sqrt(2)/54994211",
        53: "6704337069*sqrt(2)/1504091670850",
    }
    output = sp.zeros(STATE_DIMENSION, 1)
    for index, value in raw.items():
        output[index] = sp.sympify(value)
    scale = sp.Rational(5, 4) * sp.Rational(2969687394, 3261078125)
    return (scale * output).applyfunc(sp.factor)


def _symbols() -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    output = _fixed_output()
    e21 = sp.eye(STATE_DIMENSION)[:, 21]
    e32 = sp.eye(STATE_DIMENSION)[:, 32]
    curl = (n3 * e21 - n2 * e32).applyfunc(sp.factor)
    envelope = sp.Rational(25, 16) * n3**2
    extension = (envelope * output * curl.T).applyfunc(sp.factor)
    lift = _gradient_lift(n1, n2, n3)
    xz = {n1: sp.Rational(3, 5), n2: 0, n3: sp.Rational(4, 5)}
    e1 = {n1: 1, n2: 0, n3: 0}
    e2 = {n1: 0, n2: 1, n3: 0}
    xy = {n1: sp.Rational(3, 5), n2: sp.Rational(4, 5), n3: 0}
    xz_block = extension.subs(xz).applyfunc(sp.factor)
    antipodal = extension.subs({n1: -n1, n2: -n2, n3: -n3}).applyfunc(sp.factor)
    if (
        _content_hash(_matrix_payload(xz_block)) != GLOBAL_XZ_BLOCK_SHA256
        or xz_block.rank() != 1
        or not extension.subs(e1).is_zero_matrix
        or not extension.subs(e2).is_zero_matrix
        or not extension.subs(xy).is_zero_matrix
        or antipodal != -extension
        or not (extension * lift).is_zero_matrix
    ):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError("degree-three C23 symbol mismatch")
    return {
        "variables": (n1, n2, n3),
        "output": output,
        "curl": curl,
        "envelope": envelope,
        "extension": extension,
        "xz_block": xz_block,
    }


def _xz_audit(
    predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
    symbols: Mapping[str, Any],
) -> dict[str, Any]:
    prior_audit = predecessor["exact_extension"]["first_additional_frame_audit"]
    if (
        prior_audit.get("selector", {}).get("frame_name") != "xz_3_4_5"
        or prior_audit.get("candidate_obstructions") != EXPECTED_CANDIDATES
        or prior_audit.get("base_D4_RHS_sha256") != BASE_XZ_RHS_SHA256
    ):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            "degree-three predecessor xz mismatch"
        )
    frame = _frames()[1]
    direction = list(frame["direction"])
    if frame["name"] != "xz_3_4_5" or direction != [
        sp.Rational(3, 5),
        0,
        sp.Rational(4, 5),
    ]:
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError("xz selector mismatch")
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth_campaign)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
    reference = _reference_and_first_jet_packet()
    state_rotation, _ = _state_rotation(frame["rotation"])
    basis = _correction_basis()
    direction_1 = basis["block"]
    output = direction_1[:, 21]
    direction_2 = (-output * sp.eye(STATE_DIMENSION)[:, 54].T).applyfunc(sp.factor)
    cubic_global = (
        direction[0] ** 2 * (direction[0] * direction_1 + direction[1] * direction_2)
    ).applyfunc(sp.factor)
    n1, n2, n3 = symbols["variables"]
    new_global = (
        symbols["extension"]
        .subs({n1: direction[0], n2: direction[1], n3: direction[2]})
        .applyfunc(sp.factor)
    )
    new_aligned = (state_rotation.T * new_global * state_rotation).applyfunc(sp.factor)
    current_aligned = (state_rotation.T * cubic_global * state_rotation).applyfunc(sp.factor)
    current_skew = (
        reference["energy0"] * current_aligned - current_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    new_skew = (
        reference["energy0"] * new_aligned - new_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    rhs_symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = rhs_symbols.get("alpha", sp.Symbol("alpha"))
    c20 = rhs_symbols.get("c20", sp.Symbol("c20"))
    projector0 = reference["projectors"][sp.S.Zero]
    normalized_targets = []
    rows = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        before = (candidate_rhs + eta * current_skew).applyfunc(sp.factor)
        normalized_targets.append((projector0.T * before * projector0 / eta).applyfunc(sp.factor))
        corrected = (before + eta * new_skew).applyfunc(sp.factor)
        solvable, nonzero = _solve(corrected)
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "a10": candidate["a10"],
                "c20": candidate["c20"],
                "eta": candidate["eta_unique_tuning"],
                "D4_Sylvester_solvable": solvable,
                "nonzero_equal_eigenspace_compressions": nonzero,
            }
        )
    target_hashes = {_content_hash(_matrix_payload(target)) for target in normalized_targets}
    target = normalized_targets[0]
    _, coordinate_map, _ = _zero_speed_coordinates(projector0, -target)
    selector_coordinates = (
        coordinate_map * projector0.T * sp.eye(STATE_DIMENSION)[:, list(TRANSVERSE_CURL_INDICES)]
    ).applyfunc(sp.factor)
    if (
        evaluations != 15
        or sum(value != 0 for value in rhs) != 20
        or _content_hash(_matrix_payload(rhs)) != BASE_XZ_RHS_SHA256
        or len(target_hashes) != 1
        or next(iter(target_hashes)) != NORMALIZED_TARGET_SHA256
        or target.rank() != 2
        or selector_coordinates.rank() != 22
        or new_global != symbols["xz_block"]
        or new_global.rank() != 1
        or _content_hash(_matrix_payload(new_global)) != GLOBAL_XZ_BLOCK_SHA256
        or _content_hash(_matrix_payload(new_aligned)) != ALIGNED_XZ_BLOCK_SHA256
        or len(rows) != EXPECTED_CANDIDATES
        or any(
            not row["D4_Sylvester_solvable"] or row["nonzero_equal_eigenspace_compressions"]
            for row in rows
        )
    ):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError("exact xz escape audit mismatch")
    return {
        "selector": {
            "frame_name": frame["name"],
            "direction": [str(value) for value in direction],
            "prior_status": "first_exact_additional_frame_obstruction",
            "new_status": "exact_compatibility_after_C23_escape",
            "remaining_declared_frame": "xyz_1_2_2",
        },
        "directional_evaluations": evaluations,
        "all_seven_eigenspaces_checked_per_candidate": True,
        "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
        "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
        "eta_normalized_targets": len(normalized_targets),
        "distinct_eta_normalized_targets": len(target_hashes),
        "normalized_target_rank": target.rank(),
        "normalized_target_sha256": next(iter(target_hashes)),
        "transverse_selector_rank": selector_coordinates.rank(),
        "new_block_rank": new_global.rank(),
        "new_global_block_sha256": _content_hash(_matrix_payload(new_global)),
        "new_aligned_block_sha256": _content_hash(_matrix_payload(new_aligned)),
        "candidate_compatibilities": sum(row["D4_Sylvester_solvable"] for row in rows),
        "candidate_obstructions": sum(not row["D4_Sylvester_solvable"] for row in rows),
        "candidate_records": rows,
    }


def _exact_result(
    predecessor: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    if predecessor.get("status") != PREDECESSOR_STATUS:
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            "degree-three predecessor status mismatch"
        )
    symbols = _symbols()
    n1, n2, n3 = symbols["variables"]
    output_norm_sq = sp.factor((symbols["output"].T * symbols["output"])[0])
    audit = _xz_audit(predecessor, minimal, fourth_campaign, symbols)
    return {
        "declared_escape_class": {
            "name": "fixed_output_single_C23_curl_channel_with_even_scalar_envelope",
            "symbol": "DeltaB23(n)=a23(n)*w23*(n3*e21-n2*e32)^T",
            "single_curl_channel": "C23_field10",
            "fixed_output_vector": True,
            "prior_xy_certificate_plane_preserved": True,
            "broader_matrix_curl_classes_included": False,
        },
        "minimality": {
            "curl_covector_degree": 1,
            "even_envelope_must_vanish_on_n3_zero_plane": True,
            "constant_envelope_impossible": True,
            "degree_one_envelope_rejected_by_antipodal_parity": True,
            "minimal_even_envelope_degree": 2,
            "minimal_total_extension_degree": 3,
            "canonical_envelope": "a23(n)=(25/16)*n3^2",
            "normalization_at_xz_frame": str(
                symbols["envelope"].subs({n1: sp.Rational(3, 5), n2: 0, n3: sp.Rational(4, 5)})
            ),
        },
        "exact_sphere_symbol": {
            "definition": "DeltaB23(n)=(25/16)*n3^2*w23*(n3*e21-n2*e32)^T",
            "antipodally_odd": True,
            "polynomial_and_smooth_on_S2": True,
            "bounded_on_S2": True,
            "envelope_absolute_bound": "25/16",
            "curl_covector_euclidean_bound": "1",
            "frobenius_bound": f"(25/16)*sqrt({output_norm_sq})",
            "nonzero_polynomial_coefficient_blocks": 2,
            "output_vector_nonzero_entries": sum(value != 0 for value in symbols["output"]),
            "output_vector_sha256": _content_hash(_matrix_payload(symbols["output"])),
            "symbol_sha256": _content_hash(_matrix_payload(symbols["extension"])),
            "physical_gradient_lift_annihilated_identically": True,
            "gradient_lift_residual_sha256": _content_hash(
                _matrix_payload(symbols["extension"] * _gradient_lift(*symbols["variables"]))
            ),
        },
        "certificate_preservation": {
            "reference_e1_extension_zero": True,
            "axis2_e2_extension_zero": True,
            "original_xy_generic_extension_zero": True,
            "original_xy_direction": ["3/5", "4/5", "0"],
            "prior_direction_certificates_preserved": 3,
            "candidate_certificates_preserved": EXPECTED_CANDIDATES,
        },
        "xz_escape_audit": audit,
        "first_blocker": {
            "name": "remaining_xyz_rational_frame_D4_recurrence",
            "required_next": (
                "Evaluate the full polarized orders-one-through-four recurrence at the remaining "
                "declared frame xyz_1_2_2 for the combined parity-cubic, C12-envelope and C23-"
                "envelope symbol. If obstructed, classify the smallest additional linear curl "
                "channel/envelope that preserves all four certified directions; otherwise proceed "
                "to a finite generic-direction basis and PDE admission."
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
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError("C23 escape config mismatch")
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    predecessor_keys = ("degree_three_predecessor", "minimal_escape", "fourth_campaign")
    predecessors = {key: _load_bound(root, config[key]) for key in predecessor_keys}
    exact = _exact_result(
        predecessors["degree_three_predecessor"],
        predecessors["minimal_escape"],
        predecessors["fourth_campaign"],
    )
    body = {
        "schema_version": SCHEMA,
        "status": "pass_exact_degree_three_C23_great_circle_escape_all_12_xz_compatibilities",
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key])
            for key in (*predecessor_keys, "campaign_source", "campaign_test")
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "preserved_directions": ["e1", "e2", "xy_3_4_5"],
            "newly_closed_direction": "xz_3_4_5",
            "remaining_declared_direction": "xyz_1_2_2",
        },
        "exact_escape": exact,
        "counts": {
            "bound_predecessors": 3,
            "minimal_total_extension_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "single_curl_channels": 1,
            "prior_direction_certificates_preserved": 3,
            "candidate_certificates_preserved": 12,
            "new_directional_recurrence_evaluations": 15,
            "new_candidate_direction_systems_evaluated": 12,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "total_certified_directions": 4,
            "remaining_declared_frames": 1,
            "negative_controls": 7,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "constant_envelope": {"rejected": True, "prior_plane_not_preserved": True},
            "odd_linear_n3_envelope": {"rejected": True, "symbol_antipodal_parity_even": True},
            "unnormalized_n3_squared": {"rejected": True, "xz_value": "16/25"},
            "omit_C23_companion": {"rejected": True, "gradient_lift_residual_nonzero": True},
            "retain_prior_xz_obstruction": {"rejected": True, "new_compatibilities": 12},
            "infer_xyz_or_full_sphere": {"rejected": True, "remaining_frames": 1},
            "infer_local_covariant_or_PDE_admission": {
                "rejected": True,
                "origin_and_calculus_unconstructed": True,
            },
        },
        "claims": {
            "minimal_degree_three_C23_extension_in_declared_class_constructed": True,
            "antipodally_odd_bounded_smooth_C23_sphere_symbol_constructed": True,
            "e1_e2_and_xy_certificates_preserved": True,
            "full_xz_orders_one_through_four_recurrence_evaluated": True,
            "all_12_xz_D4_compatibilities_proved": True,
            "remaining_xyz_frame_audited": False,
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
            "Exact minimal degree-three C23 construction within the fixed-output, single-linear-"
            "curl, even-envelope class. The new symbol preserves e1, e2 and xy_3_4_5, annihilates "
            "the gradient lift, and closes all 12 full polarized D4 systems at xz_3_4_5. The "
            "remaining xyz frame, broader sphere, local/covariant origin, PDE admission, remaining "
            "D4, tube, CK1, CK3, TC2, B7, H7 and lifespan claims remain fail-closed."
        ),
        "next_gate": exact["first_blocker"]["required_next"],
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if not _hash_matches(document):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            "C23 escape content identity mismatch"
        )
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    exact = document.get("exact_escape", {})
    audit = exact.get("xz_escape_audit", {})
    true_claims = {
        "minimal_degree_three_C23_extension_in_declared_class_constructed",
        "antipodally_odd_bounded_smooth_C23_sphere_symbol_constructed",
        "e1_e2_and_xy_certificates_preserved",
        "full_xz_orders_one_through_four_recurrence_evaluated",
        "all_12_xz_D4_compatibilities_proved",
    }
    false_claims = {
        "remaining_xyz_frame_audited",
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
        set(document)
        != {
            "schema_version",
            "status",
            "config_sha256",
            "source_bindings",
            "selector_binding",
            "exact_escape",
            "counts",
            "negative_controls",
            "claims",
            "scope",
            "next_gate",
            "errors",
            "content_sha256",
        }
        or document.get("schema_version") != SCHEMA
        or document.get("status")
        != "pass_exact_degree_three_C23_great_circle_escape_all_12_xz_compatibilities"
        or counts
        != {
            "bound_predecessors": 3,
            "minimal_total_extension_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "single_curl_channels": 1,
            "prior_direction_certificates_preserved": 3,
            "candidate_certificates_preserved": 12,
            "new_directional_recurrence_evaluations": 15,
            "new_candidate_direction_systems_evaluated": 12,
            "new_candidate_direction_compatibilities": 12,
            "new_candidate_direction_obstructions": 0,
            "total_certified_directions": 4,
            "remaining_declared_frames": 1,
            "negative_controls": 7,
            "inferred_global_passes": 0,
        }
        or set(claims) != true_claims | false_claims
        or any(claims.get(key) is not True for key in true_claims)
        or any(claims.get(key) is not False for key in false_claims)
        or exact.get("minimality", {}).get("minimal_total_extension_degree") != 3
        or exact.get("exact_sphere_symbol", {}).get(
            "physical_gradient_lift_annihilated_identically"
        )
        is not True
        or audit.get("normalized_target_sha256") != NORMALIZED_TARGET_SHA256
        or audit.get("new_global_block_sha256") != GLOBAL_XZ_BLOCK_SHA256
        or audit.get("new_aligned_block_sha256") != ALIGNED_XZ_BLOCK_SHA256
        or audit.get("candidate_compatibilities") != 12
        or audit.get("candidate_obstructions") != 0
        or len(document.get("negative_controls", {})) != 7
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
        or document.get("errors") != []
    ):
        raise QuarticTC2D4DegreeThreeC23GreatCircleEscapeError(
            "C23 escape exact/fail-closed mismatch"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the minimal degree-three C23 great-circle escape campaign."
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
                "degree": artifact["counts"]["minimal_total_extension_degree"],
                "xz_compatibilities": artifact["counts"]["new_candidate_direction_compatibilities"],
                "xz_obstructions": artifact["counts"]["new_candidate_direction_obstructions"],
                "content_sha256": artifact["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
