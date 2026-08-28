from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item15_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_rejects_promotion_retains_hint_and_advances() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == ("REJECT_ITEM15_PROMOTION_RETAIN_TIMESCALE_HINT_ADVANCE_ITEM16")
    claims = receipt["claim_boundaries"]
    assert claims["frozen_timescale_grammars_rejected_for_promotion"] is True
    assert claims["nonpromoted_positive_hint_retained"] is True
    assert claims["direct_hot_gas_cooling_lane_completed"] is True
    assert claims["causal_timescale_mechanism_established"] is False
    assert claims["roadmap_item_15_complete"] is True
    assert claims["roadmap_item_16_authorized_next"] is True
    assert claims["alternative_to_gr_established"] is False


def test_synthesis_preserves_both_positive_but_nonsignificant_results() -> None:
    evidence = synthesis.build_receipt(ROOT)["evidence"]
    galaxy = evidence["galaxy_attempt"]
    cluster = evidence["direct_cooling_cluster_attempt"]
    assert galaxy["gates"] == {"passed": 10, "required": 15}
    assert cluster["gates"] == {"passed": 14, "required": 15}
    assert 0.05 < float(galaxy["primary"]["relative_mse_improvement"]) < 0.06
    assert 0.09 < float(cluster["primary"]["relative_mse_improvement"]) < 0.10
    assert float(galaxy["paired_sign_flip"]["p_value"]) == 0.225
    assert float(cluster["full_selection_permutation"]["p_value"]) == 0.22
    assert cluster["failed_gates"] == ["full_selection_permutation_p_at_most"]


def test_synthesis_preserves_compute_and_access_boundaries() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["counts"]["candidate_formula_cells"] == 524288
    assert receipt["counts"]["candidate_observation_score_evaluations"] == 10082058240
    assert receipt["counts"]["quality_objects"] == 141
    assert receipt["counts"]["confirmation_rows_opened"] == 0
    assert receipt["counts"]["post_response_formula_generation"] == 0
    assert receipt["counts"]["paid_model_calls"] == 0
    cluster = receipt["evidence"]["direct_cooling_cluster_attempt"]
    assert cluster["compute"]["device"] == "NVIDIA GeForce RTX 5090"
    assert cluster["compute"]["candidate_scalar_score_evaluations_with_null"] == 9437184000


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
