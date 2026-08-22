from __future__ import annotations

import json
from pathlib import Path

from sigma_theory_compiler import declarative_discovery_campaign as C

ROOT = Path(__file__).resolve().parents[1]
RECEIPT = ROOT / C.OUTPUT_PATH


def test_readiness_executes_the_whole_protocol_without_claiming_new_math() -> None:
    value = C.run_readiness(ROOT)
    assert value["counts"] == {
        "behavioral_niches": 10,
        "creativity_operators_executed": 10,
        "dataset_stages_completed": 8,
        "discovery_chain_links": 24,
        "proposals_independently_verified": 10,
    }
    assert value["proof_plan"]["closed"]
    assert value["reachability_qualified_negative"]["status"] == "REAL_NEGATIVE"
    assert value["blind_capability"] == {
        "highest_passed": 3,
        "historical_and_open_capability_established": False,
        "levels_recorded": [1, 2, 3, 4],
    }
    assert value["claims"]["novel_mathematics_established"] is False
    assert value["claims"]["historical_or_open_problem_solved"] is False


def test_sealed_readiness_receipt_replays_byte_for_byte() -> None:
    value = json.loads(RECEIPT.read_text(encoding="utf-8"))
    C.validate_receipt(value, ROOT)
    assert value == C.run_readiness(ROOT)
