from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_k55_taylor_order_one_construction_gate import (
    K55TaylorOrderOneConstructionError,
    _content_hash,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / ("configs/backgrounds/quartic_tc2_d4_k55_taylor_order_one_construction_gate.json")
ARTIFACT = ROOT / (
    "runs/physics-language/quartic-tc2-d4-k55-taylor-order-one-construction-gate/campaign.json"
)


@pytest.fixture(scope="module")
def artifact() -> dict:
    document = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    validate_campaign(document, ROOT)
    return document


def test_all_15_p55_packets_are_block_decomposed_exactly(artifact: dict) -> None:
    counts = artifact["counts"]
    assert counts["P55_Taylor_order_one_packets_validated"] == 15
    assert counts["determined_L1_sparse_entries"] == 248
    assert counts["determined_M22_order_one_sparse_entries"] == 192
    assert len(artifact["determined_order_one_block_census"]) == 15


def test_minimal_H_star_packet_contract_is_closed(artifact: dict) -> None:
    contract = artifact["exact_K55_derivative_boundary"]["missing_input_contract"]
    assert contract["required_packets"] == 15
    assert contract["registered_packets"] == 0
    assert contract["shape_each"] == [22, 22]
    assert contract["Taylor_order"] == 1
    assert "physical_action_A_star_B_star_derivative_provenance" in contract["required_fields"]


def test_exact_nonuniqueness_witness_rejects_P1_only_inference(artifact: dict) -> None:
    witness = artifact["exact_nonuniqueness_witness"]
    assert witness["candidate_A_first_order_symmetrizer_residual_nonzero_entries"] == 0
    assert witness["candidate_B_first_order_symmetrizer_residual_nonzero_entries"] == 0
    assert witness["K1_candidate_A"] != witness["K1_candidate_B"]
    assert witness["candidates_distinct"] is True


def test_manifest_does_not_advance_and_broad_claims_remain_false(
    artifact: dict,
) -> None:
    counts = artifact["counts"]
    assert counts["new_K55_Taylor_order_one_packets_registered"] == 0
    assert counts["registered_symbolic_input_packets"] == 79
    assert counts["missing_symbolic_input_packets"] == 225
    assert artifact["phase_two"]["decision"] == "BLOCK"
    assert artifact["bounded_emitter_checkpoint"]["emitted_output_rows"] == 0
    for claim in (
        "K55_Taylor_order_one_registered",
        "complete_coordinate_free_coefficient_map_emitted",
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
    tampered["counts"]["registered_H_star_plus_order_one_packets"] = 15
    tampered["content_sha256"] = _content_hash(tampered)
    with pytest.raises(K55TaylorOrderOneConstructionError, match="campaign replay mismatch"):
        validate_campaign(tampered, ROOT)
