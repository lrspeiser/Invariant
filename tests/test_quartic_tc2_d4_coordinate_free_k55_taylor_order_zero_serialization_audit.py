from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_coordinate_free_k55_taylor_order_zero_serialization_audit import (
    K55TaylorOrderZeroSerializationAuditError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/"
    "quartic_tc2_d4_coordinate_free_k55_taylor_order_zero_serialization_audit.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/"
    "quartic-tc2-d4-coordinate-free-k55-taylor-order-zero-serialization-audit/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_strongest_K55_artifacts_contain_no_exact_matrix_packet(artifact: dict) -> None:
    audits = artifact["serialized_K55_evidence_audit"]
    assert set(audits) == {"coordinate_jet_tube", "annular_K55_C6"}
    for audit in audits.values():
        assert audit["JSON_nodes_audited"] > 0
        assert audit["exact_sparse_55x55_records"] == []
        assert audit["exact_sparse_22x22_records"] == []
        assert audit["constructible_K55_order_zero_packets"] == 0
    assert sum(audit["K55_named_paths"] for audit in audits.values()) == 66


def test_minimal_contract_is_one_action_metric_or_direct_K0(artifact: dict) -> None:
    contract = artifact["minimal_missing_serialization_contract"]
    primitive = contract["smallest_sufficient_missing_primitive"]
    assert contract["status"] == "MISSING_REQUIRED_SERIALIZATION"
    assert primitive["packet_id"] == "flat_reference_action_metric_h_plus_0"
    assert primitive["shape"] == [22, 22]
    assert len(contract["already_available_exact_inputs"]) == 3
    assert len(contract["deterministic_construction_after_registration"]) == 4
    assert contract["direct_alternative"] == "serialize one exact symmetric 55x55 K_0 packet"


def test_manifest_does_not_advance_without_exact_K55(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 34
    assert counts["new_K55_Taylor_order_zero_packets_registered"] == 0
    assert counts["registered_symbolic_input_packets"] == 34
    assert counts["missing_symbolic_input_packets"] == 270
    manifest = {row["input_id"]: row for row in artifact["required_symbolic_input_manifest"]}
    assert manifest["polarized_K55_Taylor_packets"]["registered_packets"] == 0


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    for claim in (
        "K55_Taylor_order_zero_packets_registered",
        "complete_coordinate_free_coefficient_map_emitted",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_replay_and_semantic_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    assert all(value == {"rejected": True} for value in artifact["negative_controls"].values())
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["new_K55_Taylor_order_zero_packets_registered"] = 15
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(K55TaylorOrderZeroSerializationAuditError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
