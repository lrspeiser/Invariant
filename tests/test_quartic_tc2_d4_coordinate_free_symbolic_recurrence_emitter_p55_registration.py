from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_p55_registration import (
    MISSING_PACKETS,
    REGISTERED_PACKETS,
    REQUIRED_PACKETS,
    P55EmitterRegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_symbolic_recurrence_emitter_p55_registration.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-symbolic-recurrence-emitter-p55-registration/"
    "campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_bound_p55_input_is_native_validated_without_git_history(artifact: dict) -> None:
    binding = artifact["upstream_bindings"]["checkpointable_P55_result"]
    bound = artifact["bound_input_receipt"]
    assert binding["native_exact_validation"] is True
    assert bound["history_independent"] is True
    assert bound["files_verified"] == 4
    assert all(record["current_file_matches_immutable_binding"] for record in bound["files"])


def test_exact_three_p55_packets_are_registered(artifact: dict) -> None:
    packets = artifact["registered_P55_packets"]
    assert [packet["name"] for packet in packets] == ["P_1", "P_2", "P_3"]
    assert [packet["shape"] for packet in packets] == [[55, 55]] * 3
    assert [packet["nonzero_count"] for packet in packets] == [48, 48, 48]
    assert artifact["counts"]["new_P55_sparse_entries_registered"] == 144


def test_manifest_counts_advance_only_by_three(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["required_symbolic_input_packets"] == REQUIRED_PACKETS == 304
    assert counts["registered_symbolic_input_packets"] == REGISTERED_PACKETS == 6
    assert counts["missing_symbolic_input_packets"] == MISSING_PACKETS == 298
    manifest = {
        record["input_id"]: record for record in artifact["required_symbolic_input_manifest"]
    }
    physical = manifest["physical_spatial_pencil_coefficients"]
    assert physical["required_packets"] == physical["registered_packets"] == 3
    assert physical["registered_nonzero_entries"] == 144
    assert physical["exact_linearity_entries_certified"] == 3025
    assert physical["exact_sphere_minimal_polynomial_entries_reduced"] == 3025
    assert physical["exact_sphere_minimal_polynomial_nonzero_remainders"] == 0
    assert manifest["physical_gradient_lift_coefficients"]["registered_packets"] == 3


def test_remaining_manifest_is_exact_and_first_blocker_advances(artifact: dict) -> None:
    missing = {record["input_id"]: record for record in artifact["remaining_missing_inputs"]}
    assert set(missing) == {
        "polarized_P55_Taylor_packets",
        "polarized_K55_Taylor_packets",
        "polarized_TC2_Taylor_packets",
        "lower_Sylvester_correction_recurrence",
        "candidate_normalization_table",
        "sphere_mode_normal_form_reducer",
    }
    assert sum(record["missing_packets"] for record in missing.values()) == 298
    assert artifact["bounded_emitter_checkpoint"]["first_missing_input"] == (
        "polarized_P55_Taylor_packets"
    )


def test_rows_and_phase_two_remain_blocked(artifact: dict) -> None:
    checkpoint = artifact["bounded_emitter_checkpoint"]
    assert checkpoint["emitted_output_rows"] == 0
    assert checkpoint["emitted_rhs_rows"] == 0
    assert checkpoint["required_output_rows"] == 117_180
    assert artifact["phase_two"] == {
        "decision": "BLOCK",
        "admitted": False,
        "attempted": False,
        "blocker": "298 required symbolic input packets remain unregistered",
    }


def test_broad_claims_remain_false(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["exact_three_axis_flat_reference_P55_packets_registered"] is True
    for claim in (
        "matrix_projectors_evaluated",
        "complete_coordinate_free_coefficient_map_emitted",
        "complete_coordinate_free_rhs_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "TC2_closed",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
    ):
        assert claims[claim] is False


def test_negative_controls_all_reject(artifact: dict) -> None:
    assert len(artifact["negative_controls"]) == 6
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())


def test_artifact_replays_deterministically(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["missing_symbolic_input_packets"] = 297
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(P55EmitterRegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_file_binding_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["upstreams"]["checkpointable_P55_result"]["file_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(P55EmitterRegistrationError, match="invalid P55 emitter"):
        build_campaign(ROOT, path)
