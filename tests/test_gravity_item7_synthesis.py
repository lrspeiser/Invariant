from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item7_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_closes_exact_family_without_overclaiming() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM7_TESTED_GLOBAL_PHASE_COMPOSITION_GENERALIZATION_ADVANCE_ITEM8"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["all_baryonic_composition_theories_rejected"] is False
    assert boundaries["tested_global_phase_family_generalizes"] is False
    assert boundaries["phangs_lead_independently_replicated"] is False
    assert boundaries["phangs_lead_preserved_as_counterexample"] is True
    assert boundaries["roadmap_item_7_complete"] is True
    assert boundaries["roadmap_item_8_authorized_next"] is True
    assert receipt["counts"]["confirmation_accesses"] == 0


def test_synthesis_preserves_positive_origin_and_negative_replay() -> None:
    receipt = synthesis.build_receipt(ROOT)
    lead = receipt["failed_replay_lead"]
    assert lead["label"] == "FAILED_INDEPENDENT_REPLAY_LEAD"
    assert float(lead["origin_relative_mse_improvement"]) > 0.17
    assert float(lead["replay_relative_mse_improvement"]) < -0.23
    assert lead["replay_permutation_p_value"] == "8.820000000000e-01"
    assert lead["replay_unrestricted_qualifying_folds"] == 0
    assert receipt["counts"] == {
        "attempts": 2,
        "exploration_galaxies": 129,
        "quality_passing_galaxies": 126,
        "quality_failures": 3,
        "permutations": 998,
        "reserved_confirmation_galaxies": 45,
        "confirmation_accesses": 0,
        "paid_model_calls": 0,
    }


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt

