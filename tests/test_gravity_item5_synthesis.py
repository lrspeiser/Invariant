from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item5_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_is_scoped_and_advances_only_to_item6() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM5_TESTED_OBSERVABLE_PRESSURE_SUPPORT_FAMILIES_ADVANCE_ITEM6"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["tested_observable_pressure_support_families_rejected"] is True
    assert boundaries["all_pressure_support_theories_rejected"] is False
    assert boundaries["attempt1_is_conclusive"] is False
    assert boundaries["attempt2_is_clean_scoped_rejection"] is True
    assert boundaries["roadmap_item_5_complete"] is True
    assert boundaries["roadmap_item_6_authorized_next"] is True
    assert receipt["counts"]["confirmation_accesses"] == 0
    assert receipt["counts"]["reserved_confirmation_objects"] == 23


def test_synthesis_binds_both_exact_attempts() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert [row["decision"] for row in receipt["evidence"]] == [
        synthesis.ATTEMPT1_DECISION,
        synthesis.ATTEMPT2_DECISION,
    ]
    assert receipt["evidence"][1]["permutation_p_value"] == "4.300000000000e-01"
    assert receipt["counts"]["permutations"] == 998
    assert receipt["counts"]["paid_model_calls"] == 0


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
