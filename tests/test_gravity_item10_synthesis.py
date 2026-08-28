from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item10_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_closes_only_the_tested_low_quality_region() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "INCONCLUSIVE_ITEM10_SOURCE_QUALITY_NEGATIVE_DIRECTION_ADVANCE_ITEM11"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["all_baryonic_boundary_theories_rejected"] is False
    assert boundaries["tested_projected_hi_boundary_family_promoted"] is False
    assert boundaries["item10_clean_confirmation_experiment"] is False
    assert boundaries["roadmap_item_10_complete"] is True
    assert boundaries["roadmap_item_11_authorized_next"] is True
    assert boundaries["alternative_to_gr_established"] is False


def test_synthesis_preserves_quality_failure_negative_direction_and_scope_incident() -> None:
    receipt = synthesis.build_receipt(ROOT)
    attempt = receipt["evidence"]["attempt"]
    assert attempt["gates"] == {"passed": 5, "required": 14}
    assert attempt["quality_passing_galaxies"] == 20
    assert float(attempt["relative_mse_improvement"]) < 0
    assert float(attempt["paired_p_value"]) == 0.583
    assert receipt["counts"]["candidate_formula_cells"] == 131072
    assert receipt["counts"]["candidate_point_score_evaluations"] == 720896000
    assert receipt["scope_incident"]["potential_confirmation_rows_transmitted"] == 2
    assert receipt["scope_incident"]["stored_confirmation_rows"] == 0
    assert receipt["scope_incident"]["clean_unambiguous_confirmation_names_remaining_sealed"] == 11


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
