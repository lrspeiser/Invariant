from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_degree_three_c23_great_circle_escape_campaign import (
    QuarticTC2D4DegreeThreeC23GreatCircleEscapeError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/backgrounds/quartic_tc2_d4_degree_three_c23_great_circle_escape_campaign.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-degree-three-c23-great-circle-escape-campaign/campaign.json"
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


def test_minimal_c23_class(artifact: dict) -> None:
    exact = artifact["exact_escape"]
    declared = exact["declared_escape_class"]
    minimal = exact["minimality"]
    assert declared["single_curl_channel"] == "C23_field10"
    assert declared["prior_xy_certificate_plane_preserved"] is True
    assert declared["broader_matrix_curl_classes_included"] is False
    assert minimal["curl_covector_degree"] == 1
    assert minimal["minimal_even_envelope_degree"] == 2
    assert minimal["minimal_total_extension_degree"] == 3
    assert minimal["canonical_envelope"] == "a23(n)=(25/16)*n3^2"
    assert minimal["normalization_at_xz_frame"] == "1"


def test_exact_sphere_symbol(artifact: dict) -> None:
    symbol = artifact["exact_escape"]["exact_sphere_symbol"]
    assert symbol["definition"] == ("DeltaB23(n)=(25/16)*n3^2*w23*(n3*e21-n2*e32)^T")
    assert symbol["antipodally_odd"] is True
    assert symbol["polynomial_and_smooth_on_S2"] is True
    assert symbol["bounded_on_S2"] is True
    assert symbol["envelope_absolute_bound"] == "25/16"
    assert symbol["curl_covector_euclidean_bound"] == "1"
    assert symbol["physical_gradient_lift_annihilated_identically"] is True
    assert symbol["nonzero_polynomial_coefficient_blocks"] == 2


def test_prior_certificates_preserved(artifact: dict) -> None:
    preserved = artifact["exact_escape"]["certificate_preservation"]
    assert preserved == {
        "reference_e1_extension_zero": True,
        "axis2_e2_extension_zero": True,
        "original_xy_generic_extension_zero": True,
        "original_xy_direction": ["3/5", "4/5", "0"],
        "prior_direction_certificates_preserved": 3,
        "candidate_certificates_preserved": 12,
    }


def test_xz_full_recurrence_closes(artifact: dict) -> None:
    audit = artifact["exact_escape"]["xz_escape_audit"]
    assert audit["selector"] == {
        "frame_name": "xz_3_4_5",
        "direction": ["3/5", "0", "4/5"],
        "prior_status": "first_exact_additional_frame_obstruction",
        "new_status": "exact_compatibility_after_C23_escape",
        "remaining_declared_frame": "xyz_1_2_2",
    }
    assert audit["directional_evaluations"] == 15
    assert audit["all_seven_eigenspaces_checked_per_candidate"] is True
    assert audit["base_D4_RHS_nonzero_entries"] == 20
    assert (
        audit["base_D4_RHS_sha256"]
        == "d3ab104a0de327e978b6bbe03113b2cf883bce4b34684eed94574560388e0513"
    )
    assert audit["eta_normalized_targets"] == 12
    assert audit["distinct_eta_normalized_targets"] == 1
    assert audit["normalized_target_rank"] == 2
    assert (
        audit["normalized_target_sha256"]
        == "49b40a907913c6eeba85bf0a5f013810863a8631d462c74f5b68ac45f0046280"
    )
    assert audit["transverse_selector_rank"] == 22
    assert audit["new_block_rank"] == 1
    assert (
        audit["new_global_block_sha256"]
        == "7c584daafa25e52cf4d2751c1462ac093f3ac51e5968c023a477e525705d377b"
    )
    assert (
        audit["new_aligned_block_sha256"]
        == "ca7087b814ddf2ef9f00e9ce9bc51a00f629d489aa51aef779dfc6fbe36a2223"
    )
    assert audit["candidate_compatibilities"] == 12
    assert audit["candidate_obstructions"] == 0


def test_all_candidates_are_exactly_compatible(artifact: dict) -> None:
    records = artifact["exact_escape"]["xz_escape_audit"]["candidate_records"]
    assert len(records) == 12
    assert len({row["candidate_id"] for row in records}) == 12
    assert all(row["D4_Sylvester_solvable"] is True for row in records)
    assert all(row["nonzero_equal_eigenspace_compressions"] == {} for row in records)


def test_claims_are_closed_world_and_fail_closed(artifact: dict) -> None:
    claims = artifact["claims"]
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
    assert set(claims) == true_claims | false_claims
    assert all(claims[key] is True for key in true_claims)
    assert all(claims[key] is False for key in false_claims)


def test_negative_controls(artifact: dict) -> None:
    controls = artifact["negative_controls"]
    assert len(controls) == 7
    assert all(control["rejected"] is True for control in controls.values())


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("counts", "new_candidate_direction_compatibilities"), 11),
        (("counts", "remaining_declared_frames"), 0),
        (("claims", "remaining_xyz_frame_audited"), True),
        (("claims", "TC2_closed"), True),
        (("exact_escape", "xz_escape_audit", "normalized_target_sha256"), "0" * 64),
        (("exact_escape", "xz_escape_audit", "new_global_block_sha256"), "0" * 64),
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
        QuarticTC2D4DegreeThreeC23GreatCircleEscapeError,
        match="exact/fail-closed mismatch",
    ):
        validate_campaign(tampered)


def test_rehashed_unknown_claim_rejected(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["claims"]["theory_pass"] = True
    tampered = _with_hash({key: item for key, item in tampered.items() if key != "content_sha256"})
    with pytest.raises(
        QuarticTC2D4DegreeThreeC23GreatCircleEscapeError,
        match="exact/fail-closed mismatch",
    ):
        validate_campaign(tampered)


def test_predecessor_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["degree_three_predecessor"]["content_sha256"] = "0" * 64
    config = _with_hash({key: value for key, value in config.items() if key != "content_sha256"})
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        QuarticTC2D4DegreeThreeC23GreatCircleEscapeError,
        match="bound input mismatch",
    ):
        build_campaign(ROOT, path)


def test_raw_source_and_test_bindings() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    for key in ("campaign_source", "campaign_test"):
        path = ROOT / config[key]["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == config[key]["file_sha256"]
