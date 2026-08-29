import json
from pathlib import Path

from sigma_theory_compiler.gravity_item40_discrete_network_result import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "runs/gravity/roadmap/item-40-discrete-network-v1.json"


def test_aggregate_result_is_conservative_and_complete() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["decision"] == (
        "NONPROMOTED_ITEM40_DISCRETE_NETWORK_NEGATIVE_BUT_FAMILY_RETAINED"
    )
    assert result["selected_formula"]["candidate_id"] == 255184
    assert result["wallaby_dynamics"]["quality"]["passing_galaxies"] == 8
    assert result["clash_cluster_diagnostic"]["clusters"] == 20
    assert result["claim_boundaries"]["formula_pruned"] is False
    assert result["claim_boundaries"]["formula_family_pruned"] is False
    assert result["claim_boundaries"]["confirmation_remains_sealed"] is True
    assert result["protocol"]["paid_model_calls"] == 0


def test_aggregate_result_replays() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM40_AGGREGATE_REPLAY_VALID"
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
