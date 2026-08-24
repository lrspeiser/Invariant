from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler.erdos_straus_creative_shadow import (
    _mutated_pairs,
    _run_pairs,
    _witness_sample,
    parse_recipe,
    validate_receipt,
)
from sigma_theory_compiler.exponent_diophantine_sweeper import _es_hard_members

EXPERIMENT = {
    "maximum_moduli_per_recipe": 4,
    "maximum_offset": 255,
    "maximum_offsets_per_axis": 6,
    "mutation_radius": 2,
}
ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "runs" / "math" / "erdos-straus-creative-shadow" / "live-runtime.json"


def test_recipe_parser_accepts_typed_schedule_and_rejects_prose():
    recipe = parse_recipe("ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120", EXPERIMENT)
    assert recipe == {
        "basis": "lattice_transform",
        "moduli": [24, 120],
        "t_offsets": [0, 7],
        "x_offsets": [0, 2, 65],
    }
    assert parse_recipe("try a clever lattice", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=magic|x=0|t=0|m=24", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=divisor_pair|x=999|t=0|m=24", EXPERIMENT) is None


def test_exact_pair_schedule_produces_replayable_witnesses():
    members = _es_hard_members(10_000)
    wx, wy, resolved, lane_tests, _ = _run_pairs(__import__("numpy"), members, [(0, 0), (1, 0)])
    assert lane_tests > 0
    assert int(resolved.sum()) > 0
    assert _witness_sample(members, wx, wy, resolved)


def test_mutation_preserves_direct_pairs_and_adds_neighbors():
    pairs = _mutated_pairs([[64, 32]], EXPERIMENT)
    assert (64, 32) in pairs
    assert (62, 30) in pairs
    assert (66, 34) in pairs
    assert len(pairs) == 25


def test_shipped_live_receipt_validates_and_preserves_claim_boundary():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    validate_receipt(receipt, ROOT)
    assert receipt["status"] == "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM"
    assert receipt["accounting"] == {
        "baseline_gpu_lane_tests": 104_839_060,
        "creative_tail_lane_tests": 344_279,
        "denominators_covered": 99_999_999,
        "executable_llm_ideas": 11,
        "llm_ideas_proposed": 12,
        "llm_provider_calls": 4,
        "matched_control_lane_tests": 33_918_680,
        "mutated_parameter_pairs": 1_051,
        "retained_llm_provider_calls": 1,
        "total_exact_modular_lane_tests": 146_588_698,
    }
    assert receipt["hard_tail_funnel"]["creative_tail"]["resolved_from_baseline_tail"] == 173
    assert receipt["hard_tail_funnel"]["creative_tail"]["independent_cpu_exact_verified"] == 173
    rewired = receipt["hard_tail_funnel"]["matched_random_controls"]["pairing_only_rewire"]
    assert rewired["median_resolved"] == "174.000000"
    assert rewired["random_at_least_creative"] == 24
    assert all(value is False for value in receipt["claim_boundary"].values())
