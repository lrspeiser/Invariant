from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item6_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_rejects_promotion_but_retains_creative_lead() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM6_TESTED_ACCEPT_THERMODYNAMIC_PROMOTION_RETAIN_COOLING_LEAD_ADVANCE_ITEM7"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["tested_accept_thermodynamic_family_promoted"] is False
    assert boundaries["all_thermodynamic_state_theories_rejected"] is False
    assert boundaries["cooling_state_lead_retained"] is True
    assert boundaries["cooling_state_lead_confirmed"] is False
    assert boundaries["roadmap_item_6_complete"] is True
    assert boundaries["roadmap_item_7_authorized_next"] is True
    assert receipt["nonpromoted_positive_lead"]["label"] == "NONPROMOTED_POSITIVE_LEAD"
    assert receipt["counts"]["confirmation_accesses"] == 0


def test_synthesis_preserves_both_positive_and_failing_evidence() -> None:
    receipt = synthesis.build_receipt(ROOT)
    lead = receipt["nonpromoted_positive_lead"]
    assert float(lead["heldout_r2"]) > 0.4
    assert float(lead["relative_mse_improvement"]) > 0.14
    assert lead["positive_cool_core_strata"] is True
    assert lead["positive_response_error_envelopes"] is True
    assert receipt["evidence"]["permutation_p_value"] == "3.260000000000e-01"
    assert receipt["evidence"]["gates"] == {"passed": 7, "required": 9}


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
