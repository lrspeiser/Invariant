from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_order_one_serialization_frontier_gate import (
    OrderOneSerializationFrontierError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("configs/backgrounds/quartic_tc2_d4_order_one_serialization_frontier_gate.json")
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-order-one-serialization-frontier-gate/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_P55_order_one_frontier_is_exactly_measured(artifact: dict) -> None:
    assert artifact["decision"] == "BLOCK_SERIALIZATION"
    assert len(artifact["required_P55_Taylor_order_one_packets"]) == 15
    assert all(
        row["status"] == "MISSING_REQUIRED_SERIALIZATION"
        for row in artifact["required_P55_Taylor_order_one_packets"]
    )
    contract = artifact["minimal_serialization_contract"]
    assert contract["required_packets"] == 15
    assert contract["registered_packets"] == 0
    assert contract["equivalent_recurrence_inputs"]["identity"] == ("P1=M0^{-1}*(E1-M1*P0)")


def test_reference_deltaK_is_not_promoted_to_coordinate_free_packet(
    artifact: dict,
) -> None:
    audit = artifact["reference_only_deltaK_audit"]
    assert audit["serialized_reference_packets"] == 1
    assert audit["serialized_reference_direction"] == "e1"
    assert audit["serialized_deltaK_entries"] == 24
    assert audit["admissible_coordinate_free_deltaK_packets"] == 0
    assert audit["variable_coefficient_solvability_proofs"] == 0
    assert audit["reference_promotion_rejected"] is True


def test_manifest_remains_exactly_64_of_304(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["predecessor_registered_symbolic_input_packets"] == 64
    assert counts["new_symbolic_input_packets_registered"] == 0
    assert counts["registered_symbolic_input_packets"] == 64
    assert counts["missing_symbolic_input_packets"] == 240
    assert counts["serialized_P55_Taylor_order_one_packets"] == 0
    assert counts["serialized_M1_packets"] == 0
    assert counts["serialized_E1_packets"] == 0
    assert (
        sum(row["registered_packets"] for row in artifact["required_symbolic_input_manifest"]) == 64
    )


def test_phase_two_and_broad_claims_remain_blocked(artifact: dict) -> None:
    assert artifact["counts"]["full_symbol_build_calls"] == 0
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    assert artifact["phase_two"]["decision"] == "BLOCK"
    for claim in (
        "P55_Taylor_order_one_registered",
        "K55_Taylor_order_one_registered",
        "TC2_Taylor_order_one_registered",
        "coordinate_free_deltaK_order_zero_registered",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_exact_replay_and_resealed_semantic_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["serialized_P55_Taylor_order_one_packets"] = 15
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(OrderOneSerializationFrontierError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
