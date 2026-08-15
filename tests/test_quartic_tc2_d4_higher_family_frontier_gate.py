from __future__ import annotations

import json
from pathlib import Path

import pytest

from sigma_theory_compiler.quartic_tc2_d4_higher_family_frontier_gate import (
    CONFIG_PATH,
    OUTPUT_PATH,
    HigherFamilyFrontierError,
    build_campaign,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]


def test_frontier_registers_p55_then_blocks_at_first_physical_primitive() -> None:
    artifact = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    validate_campaign(artifact, ROOT)
    assert artifact["counts"]["registered_symbolic_packets"] == 154
    assert artifact["counts"]["P55_higher_packets_registered"] == 45
    assert artifact["counts"]["missing_symbolic_packets"] == 150
    assert artifact["first_missing_primitive"] == {
        "family": "physical_H_star_plus_Taylor_packets",
        "evaluation_id": "subset_0",
        "Taylor_order": 2,
        "shape": [22, 22],
        "factorial_normalization": "1/2!",
        "registered_packets_at_this_order": 0,
        "required_before_K55_order_two": 15,
        "required_orders_before_complete_K55": [2, 3, 4],
        "required_total_packets": 45,
        "reason": "P55 determines the companion and lift derivatives but not the independent physical action inner-product derivative used by K55.",
    }
    assert artifact["downstream_atomicity"]["manifest_advanced_beyond_154"] is False
    assert artifact["counts"]["emitted_output_rows"] == 0


def test_frontier_replay_is_deterministic() -> None:
    assert build_campaign(ROOT, ROOT / CONFIG_PATH) == build_campaign(ROOT, ROOT / CONFIG_PATH)


def test_tampered_frontier_receipt_is_rejected() -> None:
    artifact = json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8"))
    artifact["counts"]["registered_K55_higher_packets"] = 1
    with pytest.raises(HigherFamilyFrontierError, match="content hash|replay mismatch"):
        validate_campaign(artifact, ROOT)
