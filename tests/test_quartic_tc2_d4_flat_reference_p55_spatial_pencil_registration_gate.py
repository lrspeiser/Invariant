from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_flat_reference_p55_spatial_pencil_registration_gate import (
    FlatReferenceP55RegistrationError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs/backgrounds/quartic_tc2_d4_flat_reference_p55_spatial_pencil_registration_gate.json"
)
ARTIFACT = (
    ROOT
    / "runs/physics-language/quartic-tc2-d4-flat-reference-p55-spatial-pencil-registration-gate/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_live_constructor_and_predecessor_are_exactly_bound(artifact: dict) -> None:
    assert artifact["predecessor"]["verified"] is True
    live = artifact["live_construction"]
    assert live["functions_present"] is True
    assert live["required_functions"] == [
        "_symbol_data",
        "_extract_spatial_blocks",
        "_full_first_order_pencil",
    ]
    assert live["ordered_state"] == "z=(q,w2,w3), y=(v,w1)"


def test_missing_packet_schema_is_exact_and_fail_closed(artifact: dict) -> None:
    schema = artifact["required_input_packet_schema"]
    assert schema["required_packets"] == 3
    assert schema["required_shape_each"] == [55, 55]
    assert schema["required_axes"] == [1, 2, 3]
    assert schema["expected_nonzero_entries_each_validation_only"] == 48
    assert schema["registered_packets"] == 0
    assert schema["missing_packets"] == 3
    assert artifact["registration"] is None
    assert artifact["phase_two"]["decision"] == "BLOCK"


def test_counts_do_not_masquerade_as_coefficients(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["required_dense_entries"] == 9075
    assert counts["expected_sparse_entries_validation_only"] == 144
    assert counts["registered_sparse_entries"] == 0
    assert counts["linearity_entries_certified"] == 0
    assert counts["minimal_polynomial_entries_reduced"] == 0
    assert counts["minimal_polynomial_nonzero_remainders"] is None
    assert counts["cold_live_construction_attempted"] is False


def test_all_global_claims_remain_false(artifact: dict) -> None:
    for claim in (
        "P55_spatial_pencil_registered",
        "P55_minimal_polynomial_certified",
        "full_direction_sphere_D4_compatibility_proved",
        "complete_coordinate_free_coefficient_map_emitted",
        "TC2_closed",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_negative_controls_reject(artifact: dict) -> None:
    assert len(artifact["negative_controls"]) == 5
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())


def test_artifact_replays_deterministically(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)


def test_semantic_tamper_fails_closed(artifact: dict) -> None:
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_sparse_entries"] = 144
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(FlatReferenceP55RegistrationError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)


def test_embedded_coefficient_packet_is_rejected(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["input_packet"] = {"fabricated": True}
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FlatReferenceP55RegistrationError, match="embedded P55 entries forbidden"):
        build_campaign(ROOT, path)


def test_live_source_hash_tamper_fails_closed(tmp_path: Path) -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["live_construction"]["source_file_sha256"] = "0" * 64
    config["content_sha256"] = _content_hash(config)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(FlatReferenceP55RegistrationError, match="source hash mismatch"):
        build_campaign(ROOT, path)
