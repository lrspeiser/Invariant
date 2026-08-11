from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import sympy as sp

from . import quartic_tc2_diagonal_third_jet_campaign as directional_engine
from .quartic_tc2_d4_curl_constraint_admission_campaign import _gradient_lift
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

SCHEMA = "sigma-quartic-tc2-d4-degree-three-matrix-curl-sphere-extension-campaign-1.0"
CONFIG_SCHEMA = "sigma-quartic-tc2-d4-degree-three-matrix-curl-sphere-extension-config-1.0"
OBLIGATION_OFFSET = 244
ACTIVE_INDICES = (0, 2, 3, 9)
EXPECTED_CANDIDATES = 12
FIXED_BLOCK_SHA256 = "006aecdc99032a89a597b56e69ffed9ef35d3c9f1278b20ec96b1b0741dceb3a"
NEXT_BASE_SHA256 = "d3ab104a0de327e978b6bbe03113b2cf883bce4b34684eed94574560388e0513"
NEXT_TOTAL_BLOCK_SHA256 = "8dac2461183b13df9be8d92d60f3bb5926624e75ce72601c864bdddbe99db862"


class QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(ValueError):
    """Raised when the degree-three matrix-curl extension is inconsistent."""


def _load_bound(root: Path, binding: Mapping[str, Any]) -> dict[str, Any]:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents:
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "bound input escaped project root"
        )
    value, data = _load_file(path)
    if (
        _file_sha256(data) != binding.get("file_sha256")
        or value.get("content_sha256") != binding.get("content_sha256")
        or not _hash_matches(value)
    ):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            f"bound input mismatch: {binding.get('path')}"
        )
    return value


def _check_raw_binding(root: Path, binding: Mapping[str, Any]) -> None:
    path = (root / str(binding["path"])).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "raw binding escaped project root or is absent"
        )
    if _file_sha256(path.read_bytes()) != binding.get("file_sha256"):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            f"raw binding mismatch: {binding.get('path')}"
        )


def _fixed_output() -> sp.Matrix:
    raw = {
        11: "-16149063226/1113913340815",
        15: "41433258752/27847833520375",
        16: "64395915876*sqrt(2)/2531621229125",
        18: "33417405568/2531621229125",
        20: "-5982080/407280929",
        28: "234760*sqrt(2)/407280929",
        30: "29680*sqrt(2)/407280929",
        44: "-17202809032/1113913340815",
        48: "266593999064/27847833520375",
        49: "521256889727*sqrt(2)/27847833520375",
        51: "156412040936/27847833520375",
        53: "-6186560/407280929",
    }
    output = sp.zeros(STATE_DIMENSION, 1)
    for index, value in raw.items():
        output[index] = sp.sympify(value)
    scale = sp.Rational(5, 4) * sp.Rational(32989755249, 32610781250) * sp.sqrt(2)
    return (scale * output).applyfunc(sp.factor)


def _symbols() -> dict[str, Any]:
    n1, n2, n3 = sp.symbols("n_1 n_2 n_3", real=True)
    output = _fixed_output()
    e21 = sp.eye(STATE_DIMENSION)[:, 21]
    e54 = sp.eye(STATE_DIMENSION)[:, 54]
    curl = (n1 * e21 - n2 * e54).applyfunc(sp.factor)
    envelope = sp.Rational(25, 12) * n1 * n2
    extension = (envelope * output * curl.T).applyfunc(sp.factor)
    lift = _gradient_lift(n1, n2, n3)
    fixed_direction = {n1: sp.Rational(3, 5), n2: sp.Rational(4, 5), n3: 0}
    e1 = {n1: 1, n2: 0, n3: 0}
    e2 = {n1: 0, n2: 1, n3: 0}
    fixed_block = extension.subs(fixed_direction).applyfunc(sp.factor)
    antipodal = extension.subs({n1: -n1, n2: -n2, n3: -n3}).applyfunc(sp.factor)
    if (
        _content_hash(_matrix_payload(fixed_block)) != FIXED_BLOCK_SHA256
        or fixed_block.rank() != 1
        or not extension.subs(e1).is_zero_matrix
        or not extension.subs(e2).is_zero_matrix
        or antipodal != -extension
        or not (extension * lift).is_zero_matrix
    ):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "degree-three curl extension construction mismatch"
        )
    return {
        "variables": (n1, n2, n3),
        "output": output,
        "curl": curl,
        "envelope": envelope,
        "extension": extension,
        "fixed_block": fixed_block,
    }


def _next_frame_audit(
    minimal: Mapping[str, Any], fourth_campaign: Mapping[str, Any], symbols: Mapping[str, Any]
) -> dict[str, Any]:
    frame = _frames()[1]
    if frame["name"] != "xz_3_4_5" or list(frame["direction"]) != [
        sp.Rational(3, 5),
        0,
        sp.Rational(4, 5),
    ]:
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "next deterministic frame mismatch"
        )
    prior_order = directional_engine.TAYLOR_ORDER
    directional_engine.TAYLOR_ORDER = JET_ORDER
    try:
        payload, evaluations = _polarized_payload(frame, fourth_campaign)
    finally:
        directional_engine.TAYLOR_ORDER = prior_order
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
    n1, n2, n3 = symbols["variables"]
    extension_global = (
        symbols["extension"]
        .subs({n1: direction[0], n2: direction[1], n3: direction[2]})
        .applyfunc(sp.factor)
    )
    total_aligned = (
        state_rotation.T * (cubic_global + extension_global) * state_rotation
    ).applyfunc(sp.factor)
    total_skew = (
        reference["energy0"] * total_aligned - total_aligned.T * reference["energy0"]
    ).applyfunc(sp.factor)
    rhs = payload["fourth_Sylvester_RHS"]
    rhs_symbols = {str(symbol): symbol for symbol in rhs.free_symbols}
    alpha = rhs_symbols.get("alpha", sp.Symbol("alpha"))
    c20 = rhs_symbols.get("c20", sp.Symbol("c20"))
    rows = []
    for candidate in minimal["exact_escape"]["candidate_classification"]:
        candidate_rhs = rhs.subs(
            {
                alpha: sp.sympify(candidate["a10"]),
                c20: sp.sympify(candidate["c20"]),
            }
        ).applyfunc(sp.factor)
        eta = sp.sympify(candidate["eta_unique_tuning"])
        corrected = (candidate_rhs + eta * total_skew).applyfunc(sp.factor)
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
    if (
        evaluations != 15
        or not extension_global.is_zero_matrix
        or sum(value != 0 for value in rhs) != 20
        or _content_hash(_matrix_payload(rhs)) != NEXT_BASE_SHA256
        or total_aligned.rank() != 1
        or _content_hash(_matrix_payload(total_aligned)) != NEXT_TOTAL_BLOCK_SHA256
        or len(rows) != EXPECTED_CANDIDATES
        or any(row["D4_Sylvester_solvable"] for row in rows)
        or any(
            set(row["nonzero_equal_eigenspace_compressions"]) != {"0"}
            or row["nonzero_equal_eigenspace_compressions"]["0"]["rank"] != 2
            or row["nonzero_equal_eigenspace_compressions"]["0"]["nonzero_entries"] != 14
            for row in rows
        )
    ):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "next-frame obstruction audit mismatch"
        )
    return {
        "selector": {
            "frame_name": frame["name"],
            "direction": [str(value) for value in direction],
            "deterministic_position_after_original_generic_frame": 1,
            "stop_reason": "first_exact_additional_frame_obstruction",
            "later_declared_frames_unevaluated": 1,
        },
        "directional_evaluations": evaluations,
        "all_seven_eigenspaces_checked_per_candidate": True,
        "base_D4_RHS_nonzero_entries": sum(value != 0 for value in rhs),
        "base_D4_RHS_sha256": _content_hash(_matrix_payload(rhs)),
        "extension_block_zero_at_frame": extension_global.is_zero_matrix,
        "total_correction_block_rank": total_aligned.rank(),
        "total_correction_block_sha256": _content_hash(_matrix_payload(total_aligned)),
        "candidate_compatibilities": sum(row["D4_Sylvester_solvable"] for row in rows),
        "candidate_obstructions": sum(not row["D4_Sylvester_solvable"] for row in rows),
        "candidate_records": rows,
    }


def _exact_result(
    matrix_completion: Mapping[str, Any],
    parity_cubic: Mapping[str, Any],
    minimal: Mapping[str, Any],
    fourth_campaign: Mapping[str, Any],
) -> dict[str, Any]:
    if (
        matrix_completion.get("status") != "pass_exact_fixed_frame_rank_one_matrix_curl_completion"
        or matrix_completion.get("exact_completion", {})
        .get("minimal_rank_one_completion", {})
        .get("global_block_sha256")
        != FIXED_BLOCK_SHA256
        or parity_cubic.get("status")
        != "pass_exact_minimal_parity_preserving_cubic_angular_two_axis_escape"
    ):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "extension predecessor mismatch"
        )
    symbols = _symbols()
    n1, n2, _ = symbols["variables"]
    output_norm_sq = sp.factor((symbols["output"].T * symbols["output"])[0])
    audit = _next_frame_audit(minimal, fourth_campaign, symbols)
    return {
        "declared_extension_class": {
            "name": "fixed_output_single_C12_curl_channel_with_even_scalar_envelope",
            "symbol": "DeltaB(n)=a(n)*w*(n1*e21-n2*e54)^T",
            "fixed_output_vector": True,
            "single_curl_channel": "C12_field10",
            "even_polynomial_envelope": True,
            "required_axis_values": ["a(e1)=0", "a(e2)=0", "a(3/5,4/5,0)=1"],
            "broader_matrix_symbols_included": False,
        },
        "minimality": {
            "curl_covector_degree": 1,
            "constant_even_envelope_impossible": True,
            "degree_one_envelope_rejected_by_antipodal_parity": True,
            "minimal_even_envelope_degree": 2,
            "minimal_total_extension_degree": 3,
            "canonical_envelope": "a(n)=(25/12)*n1*n2",
            "normalization_at_original_generic_frame": str(
                symbols["envelope"].subs({n1: sp.Rational(3, 5), n2: sp.Rational(4, 5)})
            ),
        },
        "exact_sphere_symbol": {
            "definition": ("DeltaB(n)=(25/12)*n1*n2*w*(n1*e21-n2*e54)^T"),
            "antipodally_odd": True,
            "polynomial_and_smooth_on_S2": True,
            "bounded_on_S2": True,
            "envelope_absolute_bound": "25/24",
            "curl_covector_euclidean_bound": "1",
            "frobenius_bound": f"(25/24)*sqrt({output_norm_sq})",
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
            "minus_e1_extension_zero": True,
            "minus_e2_extension_zero": True,
            "original_generic_direction": ["3/5", "4/5", "0"],
            "original_generic_extension_equals_fixed_block": True,
            "fixed_block_rank": symbols["fixed_block"].rank(),
            "fixed_block_sha256": _content_hash(_matrix_payload(symbols["fixed_block"])),
            "candidate_certificates_preserved": EXPECTED_CANDIDATES,
        },
        "first_additional_frame_audit": audit,
        "first_blocker": {
            "name": "xz_frame_requires_a_nonvanishing_or_different_matrix_curl_channel",
            "required_next": (
                "Enlarge the envelope/channel class so the correction need not vanish on the "
                "n2=0 great circle, while retaining the e1/e2 certificates, antipodal oddness, "
                "gradient-lift annihilation and the original generic-frame block; then re-audit "
                "the xz frame before any all-direction or PDE admission claim."
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
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "degree-three extension config mismatch"
        )
    for key in ("campaign_source", "campaign_test"):
        _check_raw_binding(root, config[key])
    keys = ("matrix_completion", "parity_cubic", "minimal_escape", "fourth_campaign")
    predecessors = {key: _load_bound(root, config[key]) for key in keys}
    exact = _exact_result(
        predecessors["matrix_completion"],
        predecessors["parity_cubic"],
        predecessors["minimal_escape"],
        predecessors["fourth_campaign"],
    )
    body = {
        "schema_version": SCHEMA,
        "status": (
            "pass_exact_minimal_degree_three_matrix_curl_sphere_extension_with_"
            "first_additional_frame_obstruction"
        ),
        "config_sha256": config["content_sha256"],
        "source_bindings": {
            key: dict(config[key]) for key in (*keys, "campaign_source", "campaign_test")
        },
        "selector_binding": {
            "obligation_offset": OBLIGATION_OFFSET,
            "active_indices": list(ACTIVE_INDICES),
            "preserved_directions": ["e1", "e2", "xy_3_4_5"],
            "newly_audited_direction": "xz_3_4_5",
        },
        "exact_extension": exact,
        "counts": {
            "bound_predecessors": 4,
            "minimal_total_extension_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "single_curl_channels": 1,
            "preserved_direction_certificates": 3,
            "candidate_certificates_preserved": 12,
            "additional_frames_evaluated": 1,
            "additional_frames_unevaluated_after_stop": 1,
            "directional_recurrence_evaluations": 15,
            "candidate_direction_systems_evaluated": 12,
            "candidate_direction_compatibilities": 0,
            "candidate_direction_obstructions": 12,
            "negative_controls": 7,
            "inferred_global_passes": 0,
        },
        "negative_controls": {
            "constant_even_envelope": {"rejected": True, "cannot_vanish_at_both_axes": True},
            "odd_linear_envelope": {"rejected": True, "breaks_symbol_antipodal_oddness": True},
            "unnormalized_n1_n2": {"rejected": True, "generic_frame_value": "12/25"},
            "omit_curl_companion": {"rejected": True, "gradient_lift_residual_nonzero": True},
            "claim_xz_compatibility": {"rejected": True, "exact_obstructions": 12},
            "infer_all_direction_completion": {"rejected": True, "unevaluated_frames": 1},
            "infer_local_or_covariant_origin": {"rejected": True, "origin_unconstructed": True},
        },
        "claims": {
            "minimal_degree_three_extension_in_declared_class_constructed": True,
            "antipodally_odd_bounded_smooth_sphere_symbol_constructed": True,
            "e1_e2_and_original_generic_certificates_preserved": True,
            "first_additional_generic_frame_recurrence_evaluated": True,
            "canonical_degree_three_extension_rejected_as_all_direction_completion": True,
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
            "Exact minimal degree-three construction within the fixed-output single-C12-curl "
            "scalar-envelope class. The polynomial symbol is bounded, smooth and antipodally odd "
            "on the direction sphere, annihilates the physical gradient lift, and preserves the "
            "e1, e2 and xy_3_4_5 certificates. Its first additional full orders-one-through-four "
            "audit at xz_3_4_5 obstructs all 12 candidates. No broader all-direction, local, "
            "covariant, PDE-admission, remaining-D4, tube, CK, TC2, B7, H7 or lifespan conclusion "
            "is inferred."
        ),
        "next_gate": exact["first_blocker"]["required_next"],
        "errors": [],
    }
    return _with_hash(body)


def validate_campaign(document: Mapping[str, Any]) -> None:
    if not _hash_matches(document):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "degree-three extension content identity mismatch"
        )
    counts = document.get("counts", {})
    claims = document.get("claims", {})
    expected_claim_keys = {
        "minimal_degree_three_extension_in_declared_class_constructed",
        "antipodally_odd_bounded_smooth_sphere_symbol_constructed",
        "e1_e2_and_original_generic_certificates_preserved",
        "first_additional_generic_frame_recurrence_evaluated",
        "canonical_degree_three_extension_rejected_as_all_direction_completion",
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
    exact = document.get("exact_extension", {})
    audit = exact.get("first_additional_frame_audit", {})
    if (
        document.get("schema_version") != SCHEMA
        or document.get("status")
        != (
            "pass_exact_minimal_degree_three_matrix_curl_sphere_extension_with_"
            "first_additional_frame_obstruction"
        )
        or counts
        != {
            "bound_predecessors": 4,
            "minimal_total_extension_degree": 3,
            "nonzero_polynomial_coefficient_blocks": 2,
            "single_curl_channels": 1,
            "preserved_direction_certificates": 3,
            "candidate_certificates_preserved": 12,
            "additional_frames_evaluated": 1,
            "additional_frames_unevaluated_after_stop": 1,
            "directional_recurrence_evaluations": 15,
            "candidate_direction_systems_evaluated": 12,
            "candidate_direction_compatibilities": 0,
            "candidate_direction_obstructions": 12,
            "negative_controls": 7,
            "inferred_global_passes": 0,
        }
        or set(claims) != expected_claim_keys
        or exact.get("minimality", {}).get("minimal_total_extension_degree") != 3
        or exact.get("exact_sphere_symbol", {}).get(
            "physical_gradient_lift_annihilated_identically"
        )
        is not True
        or exact.get("certificate_preservation", {}).get("fixed_block_sha256") != FIXED_BLOCK_SHA256
        or audit.get("candidate_compatibilities") != 0
        or audit.get("candidate_obstructions") != 12
        or any(
            claims.get(key) is not True
            for key in (
                "minimal_degree_three_extension_in_declared_class_constructed",
                "antipodally_odd_bounded_smooth_sphere_symbol_constructed",
                "e1_e2_and_original_generic_certificates_preserved",
                "first_additional_generic_frame_recurrence_evaluated",
                "canonical_degree_three_extension_rejected_as_all_direction_completion",
            )
        )
        or any(
            claims.get(key) is not False
            for key in (
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
            )
        )
        or len(document.get("negative_controls", {})) != 7
        or any(
            control.get("rejected") is not True
            for control in document.get("negative_controls", {}).values()
        )
        or document.get("errors") != []
    ):
        raise QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError(
            "degree-three extension exact/fail-closed mismatch"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build the minimal degree-three matrix-curl sphere extension audit."
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
                "compatibilities": artifact["counts"]["candidate_direction_compatibilities"],
                "obstructions": artifact["counts"]["candidate_direction_obstructions"],
                "content_sha256": artifact["content_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
