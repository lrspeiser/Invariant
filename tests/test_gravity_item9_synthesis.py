from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item9_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_closes_only_the_exact_tested_region() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM9_TESTED_STELLAR_LIGHT_OCCUPANCY_ADVANCE_ITEM10"
    )
    boundaries = receipt["claim_boundaries"]
    assert boundaries["all_interior_exterior_gravity_theories_rejected"] is False
    assert boundaries["tested_stellar_light_occupancy_family_promoted"] is False
    assert boundaries["probes1_confirmation_opened"] is False
    assert boundaries["probes2_alternate_curves_opened"] is False
    assert boundaries["roadmap_item_9_complete"] is True
    assert boundaries["roadmap_item_10_authorized_next"] is True
    assert boundaries["alternative_to_gr_established"] is False


def test_synthesis_preserves_positive_lead_and_failed_transfer() -> None:
    receipt = synthesis.build_receipt(ROOT)
    first = receipt["evidence"]["attempt1"]
    second = receipt["evidence"]["attempt2"]
    assert first["gates"] == {"passed": 11, "required": 12}
    assert float(first["relative_mse_improvement_over_strongest_baseline"]) > 0.13
    assert second["decision"] == "INCONCLUSIVE_ITEM9_PROBES2_QUALITY"
    assert second["gates"] == {"passed": 6, "required": 15}
    assert second["attempt1_cells_beating_fixed_rar"] == 5
    assert float(second["relative_mse_improvement_over_strongest_baseline"]) < -0.66
    assert second["paired_p_value"] == "1.000000000000e+00"
    assert receipt["counts"]["probes1_confirmation_entries_opened"] == 0
    assert receipt["counts"]["probes2_alternate_rotation_entries_opened"] == 0


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
