from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_degree_three_matrix_curl_sphere_extension_campaign import (
    QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_d4_degree_three_matrix_curl_sphere_extension_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-degree-three-matrix-curl-sphere-extension-campaign/campaign.json"
)


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()


def _with_hash(body: dict) -> dict:
    result = copy.deepcopy(body)
    result["content_sha256"] = hashlib.sha256(_canonical(result)).hexdigest()
    return result


@pytest.fixture(scope="session")
def artifact() -> dict:
    value = build_campaign(ROOT, CONFIG)
    validate_campaign(value)
    assert value == json.loads(ARTIFACT.read_text(encoding="utf-8"))
    return value


def test_exact_counts(artifact: dict) -> None:
    assert artifact["counts"] == {
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


def test_minimal_degree_three_class(artifact: dict) -> None:
    exact = artifact["exact_extension"]
    declared = exact["declared_extension_class"]
    minimal = exact["minimality"]
    assert declared["single_curl_channel"] == "C12_field10"
    assert declared["broader_matrix_symbols_included"] is False
    assert minimal["curl_covector_degree"] == 1
    assert minimal["minimal_even_envelope_degree"] == 2
    assert minimal["minimal_total_extension_degree"] == 3
    assert minimal["canonical_envelope"] == "a(n)=(25/12)*n1*n2"
    assert minimal["normalization_at_original_generic_frame"] == "1"


def test_exact_sphere_symbol(artifact: dict) -> None:
    symbol = artifact["exact_extension"]["exact_sphere_symbol"]
    assert symbol["antipodally_odd"] is True
    assert symbol["polynomial_and_smooth_on_S2"] is True
    assert symbol["bounded_on_S2"] is True
    assert symbol["envelope_absolute_bound"] == "25/24"
    assert symbol["curl_covector_euclidean_bound"] == "1"
    assert symbol["nonzero_polynomial_coefficient_blocks"] == 2
    assert symbol["physical_gradient_lift_annihilated_identically"] is True


def test_three_certificates_preserved(artifact: dict) -> None:
    preserved = artifact["exact_extension"]["certificate_preservation"]
    assert preserved["reference_e1_extension_zero"] is True
    assert preserved["axis2_e2_extension_zero"] is True
    assert preserved["minus_e1_extension_zero"] is True
    assert preserved["minus_e2_extension_zero"] is True
    assert preserved["original_generic_direction"] == ["3/5", "4/5", "0"]
    assert preserved["original_generic_extension_equals_fixed_block"] is True
    assert preserved["fixed_block_rank"] == 1
    assert (
        preserved["fixed_block_sha256"]
        == "006aecdc99032a89a597b56e69ffed9ef35d3c9f1278b20ec96b1b0741dceb3a"
    )
    assert preserved["candidate_certificates_preserved"] == 12


def test_first_additional_frame_is_exact_obstruction(artifact: dict) -> None:
    audit = artifact["exact_extension"]["first_additional_frame_audit"]
    assert audit["selector"] == {
        "frame_name": "xz_3_4_5",
        "direction": ["3/5", "0", "4/5"],
        "deterministic_position_after_original_generic_frame": 1,
        "stop_reason": "first_exact_additional_frame_obstruction",
        "later_declared_frames_unevaluated": 1,
    }
    assert audit["directional_evaluations"] == 15
    assert audit["all_seven_eigenspaces_checked_per_candidate"] is True
    assert audit["base_D4_RHS_nonzero_entries"] == 20
    assert (
        audit["base_D4_RHS_sha256"]
        == "d3ab104a0de327e978b6bbe03113b2cf883bce4b34684eed94574560388e0513"
    )
    assert audit["extension_block_zero_at_frame"] is True
    assert audit["total_correction_block_rank"] == 1
    assert (
        audit["total_correction_block_sha256"]
        == "8dac2461183b13df9be8d92d60f3bb5926624e75ce72601c864bdddbe99db862"
    )
    assert audit["candidate_compatibilities"] == 0
    assert audit["candidate_obstructions"] == 12


def test_all_candidates_have_only_rank_two_zero_speed_obstruction(artifact: dict) -> None:
    records = artifact["exact_extension"]["first_additional_frame_audit"]["candidate_records"]
    assert len(records) == 12
    assert len({row["candidate_id"] for row in records}) == 12
    for row in records:
        assert row["D4_Sylvester_solvable"] is False
        assert set(row["nonzero_equal_eigenspace_compressions"]) == {"0"}
        zero = row["nonzero_equal_eigenspace_compressions"]["0"]
        assert zero["rank"] == 2
        assert zero["nonzero_entries"] == 14


def test_claims_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["minimal_degree_three_extension_in_declared_class_constructed"] is True
    assert claims["antipodally_odd_bounded_smooth_sphere_symbol_constructed"] is True
    assert claims["e1_e2_and_original_generic_certificates_preserved"] is True
    assert claims["first_additional_generic_frame_recurrence_evaluated"] is True
    assert claims["canonical_degree_three_extension_rejected_as_all_direction_completion"] is True
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
    ):
        assert claims[key] is False


def test_negative_controls(artifact: dict) -> None:
    controls = artifact["negative_controls"]
    assert len(controls) == 7
    assert all(control["rejected"] is True for control in controls.values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "candidate_direction_obstructions"), 11),
        (("counts", "additional_frames_unevaluated_after_stop"), 0),
        (("claims", "full_direction_sphere_D4_compatibility_proved"), True),
        (("claims", "local_differential_operator_origin_proved"), True),
        (("claims", "TC2_closed"), True),
        (("claims", "theory_pass"), True),
        (("exact_extension", "certificate_preservation", "fixed_block_sha256"), "0" * 64),
    ],
)
def test_rehashed_semantic_tamper_rejected(
    artifact: dict, path: tuple[str, ...], value: object
) -> None:
    tampered = copy.deepcopy(artifact)
    target = tampered
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    tampered = _with_hash({key: item for key, item in tampered.items() if key != "content_sha256"})
    with pytest.raises(
        QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError,
        match="exact/fail-closed mismatch",
    ):
        validate_campaign(tampered)


def test_predecessor_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["matrix_completion"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        QuarticTC2D4DegreeThreeMatrixCurlSphereExtensionError,
        match="bound input mismatch",
    ):
        build_campaign(ROOT, path)


def test_raw_source_and_test_bindings() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
