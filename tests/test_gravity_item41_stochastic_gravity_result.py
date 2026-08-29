import json
from pathlib import Path

from sigma_theory_compiler.gravity_item41_stochastic_gravity_result import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "runs/gravity/roadmap/item-41-stochastic-gravity-v1.json"


def test_aggregate_result_is_conservative_and_complete() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["decision"] == (
        "NONPROMOTED_ITEM41_STOCHASTIC_LAW_NEGATIVE_BUT_FAMILY_RETAINED"
    )
    assert result["selected_formula"]["candidate_id"] == 45024
    assert result["ghasp_paired_side_test"]["quality"]["galaxies"] == 15
    assert result["clash_cluster_diagnostic"]["clusters"] == 20
    assert result["claim_boundaries"]["formula_pruned"] is False
    assert result["claim_boundaries"]["formula_family_pruned"] is False
    assert result["claim_boundaries"]["item28_confirmation_remains_sealed"] is True
    assert result["protocol"]["paid_model_calls"] == 0


def test_aggregate_result_replays() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM41_AGGREGATE_REPLAY_VALID"
    assert replay["confirmation_values_read"] == 0
    assert replay["paid_model_calls"] == 0
