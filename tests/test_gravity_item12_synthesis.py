from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item12_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_retains_lead_without_overclaiming() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == "PASS_ITEM12_EXPLORATION_LEAD_RETAINED_ADVANCE_ITEM13"
    claims = receipt["claim_boundaries"]
    assert claims["dynamical_age_cause_established"] is False
    assert claims["retained_family_independently_confirmed"] is False
    assert claims["confirmation_opened"] is False
    assert claims["roadmap_item_12_complete"] is True
    assert claims["roadmap_item_13_authorized_next"] is True
    assert claims["alternative_to_gr_established"] is False
    assert receipt["retained_lead"]["origin_status"] == "COMBINATION"


def test_synthesis_preserves_positive_result_and_clean_confirmation() -> None:
    receipt = synthesis.build_receipt(ROOT)
    attempt = receipt["evidence"]["attempt"]
    assert attempt["gates"] == {"passed": 13, "required": 13}
    assert attempt["quality_passing_galaxies"] == 585
    assert float(attempt["relative_mse_improvement"]) > 0.18
    assert float(attempt["paired_p_value"]) == 0.001
    assert {row["selected_family"] for row in attempt["fold_selections"]} == {
        "spectral_clock_consensus"
    }
    assert {row["modulation"] for row in attempt["fold_selections"]} == {"stellar_surface_density"}
    assert receipt["counts"]["candidate_formula_cells"] == 262144
    assert receipt["counts"]["candidate_galaxy_score_evaluations"] == 3067084800
    assert receipt["counts"]["confirmation_rows_opened"] == 0
    assert receipt["counts"]["post_response_formula_generation"] == 0


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
