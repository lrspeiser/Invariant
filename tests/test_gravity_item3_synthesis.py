from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import gravity_item3_synthesis as synthesis

ROOT = Path(__file__).resolve().parents[1]


def test_synthesis_is_scoped_and_advances_only_to_item4() -> None:
    receipt = synthesis.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REJECT_ITEM3_TESTED_LOCAL_SURFACE_VOLUME_DENSITY_FAMILIES_ADVANCE_ITEM4"
    )
    assert receipt["claim_boundaries"]["local_frozen_families_rejected"] is True
    assert receipt["claim_boundaries"]["all_surface_volume_density_theories_rejected"] is False
    assert receipt["claim_boundaries"]["roadmap_item_3_complete"] is True
    assert receipt["claim_boundaries"]["roadmap_item_4_authorized_next"] is True
    assert receipt["counts"]["confirmation_accesses"] == 0


def test_synthesis_hash_and_committed_receipt() -> None:
    receipt = synthesis.build_receipt(ROOT)
    content = dict(receipt)
    content.pop("content_sha256")
    assert receipt["content_sha256"] == synthesis.canonical_sha256(content)
    path = ROOT / synthesis.OUTPUT_PATH
    if path.exists():
        assert json.loads(path.read_text(encoding="utf-8")) == receipt
