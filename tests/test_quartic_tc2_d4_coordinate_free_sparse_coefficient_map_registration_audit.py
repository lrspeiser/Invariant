from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_sparse_coefficient_map_registration_audit import (
    REQUIRED_ROWS,
    CoordinateFreeCoefficientMapAuditError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "backgrounds"
    / "quartic_tc2_d4_coordinate_free_sparse_coefficient_map_registration_audit.json"
)
ARTIFACT = (
    ROOT
    / "runs"
    / "physics-language"
    / "quartic-tc2-d4-coordinate-free-sparse-coefficient-map-registration-audit"
    / "campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_exact_registration_counts_are_fail_closed(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["required_coefficient_rows"] == REQUIRED_ROWS
    assert counts["registered_coefficient_rows"] == 0
    assert counts["missing_coefficient_rows"] == REQUIRED_ROWS
    assert counts["required_rhs_rows"] == REQUIRED_ROWS
    assert counts["registered_rhs_rows"] == 0
    assert counts["missing_rhs_rows"] == REQUIRED_ROWS
    assert counts["registered_sparse_entries"] == 0
    assert counts["global_numerator_polynomials_materialized_upstream"] == 0


def test_checkpoint_is_exact_deterministic_and_incomplete(artifact: dict) -> None:
    checkpoint = artifact["sparse_registration_checkpoint"]
    assert checkpoint["complete"] is False
    assert checkpoint["registration_cursor"] == {
        "next_cokernel_coordinate": 0,
        "next_flat_row_offset": 0,
        "next_odd_sphere_mode": 0,
    }
    assert checkpoint["registered_coefficient_rows"] == []
    assert checkpoint["registered_rhs_rows"] == []
    assert checkpoint["registered_sparse_entries"] == []
    assert checkpoint["counts"]["missing_coefficient_rows"] == REQUIRED_ROWS
    assert checkpoint["content_sha256"] == _content_hash(checkpoint)


def test_point_evidence_is_not_promoted_to_mode_coefficients(artifact: dict) -> None:
    packets = artifact["point_evidence_packets"]
    assert len(packets) == 3
    assert all(packet["registered_coordinate_free_rows"] == 0 for packet in packets)
    by_id = {packet["packet_id"]: packet for packet in packets}
    assert (
        by_id["revised_thirteen_frame_local_certificate_summary"]["local_direction_certificates"]
        == 13
    )
    assert by_id["canonical_e1_D4_obstruction"]["zero_speed_compression_rank"] == 2
    assert (
        by_id["regular_rational_chart_counterexample"]["global_numerator_polynomials_materialized"]
        == 0
    )


def test_phase_two_requires_complete_map_and_does_not_run(artifact: dict) -> None:
    phase = artifact["phase_two"]
    assert phase["decision"] == "BLOCK"
    assert phase["attempted"] is False
    assert phase["admitted"] is False
    assert phase["PASS"] is False
    assert phase["OBSTRUCTED_CLASS"] is False
    assert phase["BLOCK"] is True
    assert phase["admission_requirements"]["require_missing_rows"] == 0
    assert artifact["counts"]["phase_two_solve_attempts"] == 0


def test_resource_caps_are_explicit_and_satisfied(artifact: dict) -> None:
    admission = artifact["resource_admission"]
    assert admission["checkpoint_within_cap"] is True
    assert admission["registered_rows_within_cap"] is True
    assert admission["registered_sparse_entries_within_cap"] is True
    assert admission["checkpoint_bytes"] < admission["caps"]["maximum_checkpoint_bytes"]


def test_broad_claims_remain_false(artifact: dict) -> None:
    claims = artifact["claims"]
    assert claims["deterministic_sparse_checkpoint_materialized"] is True
    assert claims["point_evidence_separated_from_coefficient_registration"] is True
    for claim in (
        "complete_coordinate_free_coefficient_map_registered",
        "complete_exact_rhs_registered",
        "phase_two_exact_solve_admitted",
        "full_direction_sphere_D4_compatibility_proved",
        "complete_D2F_tensor_registered",
        "full_high_atom_identity_proved",
        "TC2_closed",
        "global_H7_closed",
        "full_tube_Sylvester_identity_proved",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
        "theory_candidate_rejected",
    ):
        assert claims[claim] is False


def test_all_negative_controls_reject(artifact: dict) -> None:
    controls = artifact["negative_controls"]
    assert len(controls) == 8
    assert all(control == {"rejected": True} for control in controls.values())


def test_artifact_replays_deterministically(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_coefficient_rows"] = 1
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(CoordinateFreeCoefficientMapAuditError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_upstream_binding_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["upstream_bindings"]["canonical_D4_obstruction_certificate"]["content_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(
        CoordinateFreeCoefficientMapAuditError, match="config upstream hash mismatch"
    ):
        build_campaign(ROOT, path)


def test_admission_cap_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["phase_two_admission"]["require_complete_rhs_rows"] -= 1
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(CoordinateFreeCoefficientMapAuditError, match="invalid coefficient-map"):
        build_campaign(ROOT, path)
