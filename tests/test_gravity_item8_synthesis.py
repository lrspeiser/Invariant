from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item8_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_closes_exact_family_without_overclaiming() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM8_TESTED_PROJECTED_K_LIGHT_FIELD_DERIVATIVES_ADVANCE_ITEM9"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["all_field_curvature_theories_rejected"] is False
    assert boundaries["tested_projected_k_light_derivative_families_promoted"] is False
    assert boundaries["confirmation_opened"] is False
    assert boundaries["roadmap_item_8_complete"] is True
    assert boundaries["roadmap_item_9_authorized_next"] is True
    assert boundaries["alternative_to_gr_established"] is False


def test_synthesis_preserves_negative_result_and_confirmation_boundary() -> None:
    receipt = synthesis.build_receipt(ROOT)
    evidence = receipt["evidence"]
    assert float(evidence["relative_mse_improvement"]) < -0.096
    assert evidence["unrestricted_qualifying_folds"] == 0
    assert evidence["permutation_p_value"] == "7.620000000000e-01"
    assert evidence["gates"] == {"passed": 3, "required": 11}
    assert receipt["counts"] == {
        "attempts": 1,
        "exploration_groups": 98,
        "quality_passing_groups": 98,
        "permutations": 499,
        "preregistered_model_families": 8,
        "qualifying_families": 4,
        "reserved_confirmation_groups": 33,
        "confirmation_accesses": 0,
        "post_response_formula_generation": 0,
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
