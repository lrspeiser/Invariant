from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item13_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_rejects_disturbance_but_retains_age_replication() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == ("REJECT_ITEM13_DISTURBANCE_RETAIN_AGE_LEAD_ADVANCE_ITEM14")
    claims = receipt["claim_boundaries"]
    assert claims["visual_cas_disturbance_family_rejected_in_scope"] is True
    assert claims["item12_age_family_disjoint_identity_replication"] is True
    assert claims["item12_age_family_cross_source_confirmed"] is False
    assert claims["merger_or_relaxation_cause_established"] is False
    assert claims["roadmap_item_13_complete"] is True
    assert claims["roadmap_item_14_authorized_next"] is True
    assert claims["alternative_to_gr_established"] is False


def test_synthesis_preserves_negative_disturbance_and_positive_age_results() -> None:
    attempt = synthesis.build_receipt(ROOT)["evidence"]["attempt"]
    assert attempt["gates"] == {"passed": 10, "required": 16}
    assert attempt["quality_passing_galaxies"] == 243
    assert float(attempt["disturbance_relative_mse_improvement"]) < 0
    assert float(attempt["paired_sign_flip"]["disturbance_after_age"]["p_value"]) == 0.683
    assert float(attempt["item12_age_replication_relative_mse_improvement"]) > 0.23
    assert float(attempt["paired_sign_flip"]["item12_age_replication"]["p_value"]) == 0.001
    assert float(attempt["item12_age_persistence_relative_mse_improvement"]) > 0.22
    assert (
        float(attempt["paired_sign_flip"]["item12_age_persistence_after_disturbance"]["p_value"])
        == 0.001
    )


def test_synthesis_preserves_boundaries_and_compute_count() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["counts"]["candidate_formula_cells"] == 262144
    assert receipt["counts"]["candidate_galaxy_score_evaluations"] == 1274019840
    assert receipt["counts"]["confirmation_rows_opened"] == 0
    assert receipt["counts"]["post_response_formula_generation"] == 0
    assert receipt["counts"]["paid_model_calls"] == 0
    assert "disjoint-identity replication" in receipt["retained_lead"]["boundary"]
    assert "not cross-survey confirmation" in receipt["retained_lead"]["boundary"]


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
