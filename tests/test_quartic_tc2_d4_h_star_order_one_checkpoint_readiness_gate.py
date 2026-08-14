from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_h_star_order_one_checkpoint_readiness_gate import (
    HStarOrderOneReadinessError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / (
    "configs/backgrounds/quartic_tc2_d4_h_star_order_one_checkpoint_readiness_gate.json"
)
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-h-star-order-one-checkpoint-readiness-gate/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_no_exact_H_star_order_one_packet_is_currently_serialized(
    artifact: dict,
) -> None:
    audit = artifact["current_serialization_audit"]
    assert audit["exact_H_star_plus_order_one_packets_found"] == 0
    assert audit["exact_A_star_order_one_matrix_packets_found"] == 0
    assert audit["exact_B_star_order_one_axis_matrix_packets_found"] == 0
    assert audit["scalar_H_star_envelope_records_rejected_as_coefficient_data"] == 15
    assert audit["cold_symbol_build_used_in_audit"] is False


def test_checkpoint_contract_is_minimal_and_closed(artifact: dict) -> None:
    contract = artifact["checkpointable_minimal_source_contract"]
    assert contract["status"] == "READY_NOT_EXECUTED"
    assert [phase["units"] for phase in contract["phases"]] == [1, 4, 15, 1]
    assert contract["phases"][1]["unit_ids"] == ["G_12", "G_01", "H_01", "H_11"]
    counts = contract["expected_counts_after_complete_run"]
    assert counts["basis_A_star_order_one_matrices"] == 4
    assert counts["basis_B_star_order_one_axis_matrices"] == 12
    assert counts["polarized_H_star_plus_order_one_packets"] == 15


def test_exact_source_recipe_uses_action_not_P55_inversion(artifact: dict) -> None:
    recipe = artifact["checkpointable_minimal_source_contract"]["exact_source_recipe"]
    assert "action_symbol" in recipe["action_pencil"]
    assert "D_atom action['A']" in recipe["basis_A_star_order_one"]
    assert "D_atom D_nj action['B']" in recipe["basis_B_star_order_one_axis"]
    assert "H_star_plus_1" in recipe["polarized_metric"]


def test_readiness_does_not_advance_manifest_or_broad_claims(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["registered_symbolic_input_packets"] == 79
    assert counts["missing_symbolic_input_packets"] == 225
    assert counts["registered_polarized_H_star_plus_order_one_packets"] == 0
    assert counts["full_symbol_build_calls"] == 0
    for claim in (
        "H_star_plus_order_one_packets_registered",
        "K55_Taylor_order_one_registered",
        "full_direction_sphere_D4_compatibility_proved",
        "global_H7_closed",
        "nonlinear_PDE_closure_proved",
        "lifespan_proved",
    ):
        assert artifact["claims"][claim] is False


def test_exact_replay_and_resealed_tamper_fail_closed(artifact: dict) -> None:
    assert build_campaign(ROOT, CONFIG) == artifact
    assert artifact["content_sha256"] == _content_hash(artifact)
    tampered = copy.deepcopy(artifact)
    tampered["counts"]["registered_polarized_H_star_plus_order_one_packets"] = 15
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(HStarOrderOneReadinessError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
