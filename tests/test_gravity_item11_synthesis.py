from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item11_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_advances_without_overclaiming_external_field_rejection() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == ("INCONCLUSIVE_ITEM11_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM12")
    claims = receipt["claim_boundaries"]
    assert claims["all_external_field_theories_rejected"] is False
    assert claims["tested_scalar_environment_family_promoted"] is False
    assert claims["confirmation_opened"] is False
    assert claims["roadmap_item_11_complete"] is True
    assert claims["roadmap_item_12_authorized_next"] is True
    assert claims["alternative_to_gr_established"] is False


def test_synthesis_preserves_negative_result_and_clean_confirmation() -> None:
    receipt = synthesis.build_receipt(ROOT)
    attempt = receipt["evidence"]["attempt"]
    assert attempt["gates"] == {"passed": 6, "required": 14}
    assert attempt["quality_passing_galaxies"] == 119
    assert float(attempt["relative_mse_improvement"]) < -0.13
    assert float(attempt["paired_p_value"]) == 0.99
    assert receipt["counts"]["candidate_formula_cells"] == 262144
    assert receipt["counts"]["candidate_galaxy_score_evaluations"] == 623902720
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
