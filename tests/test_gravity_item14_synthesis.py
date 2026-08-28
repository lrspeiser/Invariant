from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item14_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_rejects_mask_coherence_and_advances() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == "REJECT_ITEM14_MASK_COHERENCE_ADVANCE_ITEM15"
    claims = receipt["claim_boundaries"]
    assert claims["static_mask_geometry_family_rejected_in_scope"] is True
    assert claims["resolved_annular_span_pattern_observed"] is True
    assert claims["temporal_resonance_cause_established"] is False
    assert claims["age_dependent_baryonic_mass_error_established"] is False
    assert claims["roadmap_item_14_complete"] is True
    assert claims["roadmap_item_15_authorized_next"] is True
    assert claims["alternative_to_gr_established"] is False


def test_synthesis_preserves_negative_stellar_and_halpha_results() -> None:
    attempt = synthesis.build_receipt(ROOT)["evidence"]["attempt"]
    assert attempt["gates"] == {"passed": 7, "required": 14}
    assert attempt["quality_passing_galaxies"] == 204
    assert float(attempt["primary_relative_mse_improvement"]) < -0.025
    assert float(attempt["paired_sign_flip"]["p_value"]) == 0.827
    assert float(attempt["secondary_halpha"]["relative_mse_improvement"]) < 0
    assert attempt["secondary_halpha"]["candidate_reselection"] is False


def test_synthesis_preserves_descriptive_pattern_and_boundaries() -> None:
    receipt = synthesis.build_receipt(ROOT)
    distribution = receipt["evidence"]["attempt"]["resolved_ratio_distribution"]
    assert float(distribution["stellar_median_outer_to_inner"]) > 1.35
    assert float(distribution["halpha_median_outer_to_inner"]) > 1.23
    assert receipt["counts"]["candidate_formula_cells"] == 262144
    assert receipt["counts"]["candidate_galaxy_score_evaluations"] == 1069547520
    assert receipt["counts"]["confirmation_rows_opened"] == 0
    assert receipt["counts"]["post_response_formula_generation"] == 0
    assert receipt["counts"]["paid_model_calls"] == 0
    assert "not flat circular rotation" in receipt["retained_observation"]["boundary"]


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
