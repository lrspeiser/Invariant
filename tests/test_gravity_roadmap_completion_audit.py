from __future__ import annotations

from pathlib import Path

from sigma_theory_compiler.gravity_roadmap_completion_audit import build_audit

ROOT = Path(__file__).resolve().parents[1]


def test_all_72_items_have_receipts_without_claiming_72_passes() -> None:
    audit = build_audit(ROOT)
    assert audit["numbered_items"] == 72
    assert audit["items_with_top_level_receipts"] == 72
    assert audit["missing_items"] == []
    assert audit["invalid_json_receipts"] == []
    assert audit["execution_audit_complete"] is True
    assert audit["claims"]["every_numbered_item_passed"] is False
    assert audit["claims"]["single_empirical_counterexample_used_as_universal_veto"] is False
