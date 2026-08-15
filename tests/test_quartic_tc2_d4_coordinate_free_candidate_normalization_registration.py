from __future__ import annotations

import copy
import json
from fractions import Fraction
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_candidate_normalization_registration import (
    CandidateNormalizationRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/quartic_tc2_d4_coordinate_free_candidate_normalization_registration.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-candidate-normalization-registration/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_largest_complete_derivable_family_is_selected(artifact: dict) -> None:
    audit = {record["input_id"]: record for record in artifact["missing_family_evidence_audit"]}
    assert len(audit) == 6
    assert audit["candidate_normalization_table"]["required_packets"] == 12
    assert audit["candidate_normalization_table"]["exact_complete_packets_found"] == 12
    assert audit["candidate_normalization_table"]["decision"] == (
        "REGISTER_LARGEST_COMPLETE_DERIVABLE_FAMILY"
    )
    for family in (
        "polarized_P55_Taylor_packets",
        "polarized_K55_Taylor_packets",
        "polarized_TC2_Taylor_packets",
        "lower_Sylvester_correction_recurrence",
        "sphere_mode_normal_form_reducer",
    ):
        assert audit[family]["exact_complete_packets_found"] == 0
        assert audit[family]["blocker"]


def test_all_12_candidate_normalizations_have_zero_exact_residual(artifact: dict) -> None:
    packets = artifact["registered_candidate_normalization_packets"]
    assert len(packets) == 12
    assert len({packet["candidate_id"] for packet in packets}) == 12
    for packet in packets:
        a10 = Fraction(packet["a10"])
        eta = Fraction(packet["eta"])
        assert eta + Fraction(34_816, 15) * a10**5 == 0
        assert packet["common_shape_factorization_residual"] == "0"
        assert packet["common_shape_factorization_residual_zero"] is True
        assert packet["content_sha256"] == _content_hash(packet)


def test_manifest_advances_from_6_to_18_and_leaves_286(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["required_symbolic_input_packets"] == 304
    assert counts["predecessor_registered_symbolic_input_packets"] == 6
    assert counts["new_candidate_normalization_packets_registered"] == 12
    assert counts["registered_symbolic_input_packets"] == 18
    assert counts["missing_symbolic_input_packets"] == 286
    manifest = {
        record["input_id"]: record for record in artifact["required_symbolic_input_manifest"]
    }
    normalization = manifest["candidate_normalization_table"]
    assert normalization["required_packets"] == normalization["registered_packets"] == 12
    assert normalization["all_common_shape_factorization_residuals_zero"] is True


def test_remaining_packet_arithmetic_is_exact(artifact: dict) -> None:
    missing = {record["input_id"]: record for record in artifact["remaining_missing_inputs"]}
    assert set(missing) == {
        "polarized_P55_Taylor_packets",
        "polarized_K55_Taylor_packets",
        "polarized_TC2_Taylor_packets",
        "lower_Sylvester_correction_recurrence",
        "sphere_mode_normal_form_reducer",
    }
    assert [missing[name]["missing_packets"] for name in missing] == [75, 75, 75, 60, 1]
    assert sum(record["missing_packets"] for record in missing.values()) == 286


def test_rows_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["bounded_emitter_checkpoint"]["required_output_rows"] == 117_180
    assert artifact["phase_two"] == {
        "decision": "BLOCK",
        "admitted": False,
        "attempted": False,
        "blocker": "286 required symbolic input packets remain unregistered",
    }
    for claim in (
        "matrix_projectors_evaluated",
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "TC2_closed",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
    ):
        assert artifact["claims"][claim] is False


def test_negative_controls_reject_and_artifact_replays(artifact: dict) -> None:
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_symbolic_input_packets"] = 19
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(CandidateNormalizationRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_upstream_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["upstreams"]["canonical_D4_obstruction"]["content_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CandidateNormalizationRegistrationError, match="config upstream seal"):
        build_campaign(ROOT, path)
