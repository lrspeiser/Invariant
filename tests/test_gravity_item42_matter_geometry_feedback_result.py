import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item42_matter_geometry_feedback_result import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1.json"


def test_aggregate_result_preserves_cross_scale_increment_and_limits() -> None:
    result = json.loads(RESULT.read_text(encoding="utf-8"))
    assert result["decision"] == (
        "NONPROMOTED_ITEM42_CROSS_SCALE_FEEDBACK_INCREMENT_RETAINED"
    )
    assert result["selected_formula"]["candidate_id"] == 170142
    assert result["fresh_wallaby_dynamics"]["quality"]["passing_galaxies"] == 11
    assert result["fresh_wallaby_dynamics"][
        "improvement_vs_matched_no_feedback_percent"
    ] == pytest.approx(1.6027302844363718)
    assert result["clash_cluster_diagnostic"]["convergence"][
        "nonconverged_clusters"
    ] == 1
    assert result["clash_cluster_diagnostic"][
        "improvement_vs_matched_no_feedback_percent"
    ] == pytest.approx(10.545141203201279)
    assert result["claim_boundaries"]["complete_baryonic_inventory"] is False
    assert result["claim_boundaries"]["formula_pruned"] is False
    assert result["claim_boundaries"]["formula_family_pruned"] is False
    assert result["claim_boundaries"]["confirmation_remains_sealed"] is True


def test_aggregate_result_replays() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM42_AGGREGATE_REPLAY_VALID"
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
