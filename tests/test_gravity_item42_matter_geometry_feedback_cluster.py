import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item42_matter_geometry_feedback_cluster import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1-source/clash-transfer-result.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_transfer_is_commit_bound_unchanged_and_not_confirmation(result: dict) -> None:
    assert result["protocol"]["dynamics_commit"] == (
        "e285a831e38f7efcac02bab133996fdf5f1b9653"
    )
    assert result["selected_formula"]["candidate_id"] == 170142
    assert result["protocol"]["selection_use"] is False
    assert result["protocol"]["retuning"] is False
    assert result["protocol"]["post_selection_candidate_cells"] == 0
    assert result["claim_boundaries"]["fresh_confirmation"] is False


def test_feedback_increment_transfers_but_one_cluster_does_not_converge(
    result: dict,
) -> None:
    assert result["data"]["clusters"] == 20
    assert result["data"]["radial_points"] == 84
    assert result["convergence"]["converged_clusters"] == 19
    assert result["convergence"]["nonconverged_clusters"] == 1
    assert result["convergence"]["all_selected_formula_fixed_points_converged"] is False
    assert result["losses"]["candidate"] == pytest.approx(42.5957474394706)
    assert result["losses"]["matched_no_feedback"] == pytest.approx(47.61703054747313)
    assert result["losses"]["mond_RAR"] == pytest.approx(38.73648942103511)
    assert result["improvement_vs_matched_no_feedback_percent"] == pytest.approx(
        10.545141203201279
    )
    assert result["improvement_vs_strongest_percent"] == pytest.approx(
        -9.962849179460735
    )


def test_nonconvergence_and_mismatches_do_not_kill_the_formula(result: dict) -> None:
    report = result["counterexample_policy_report"]
    assessment = result["counterexample_assessment"]
    assert report["numerical_domain_failure_count"] == 1
    assert report["raw_counterexample_count"] == 9
    assert report["uncertainty_resolved_counterexample_count"] == 0
    assert assessment["terminal_rejection_in_tested_scope"] is False
    assert assessment["candidate_pruned_globally"] is False
    assert assessment["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False


def test_clash_transfer_replays_exactly() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM42_CLASH_TRANSFER_REPLAY_VALID"
    assert replay["clusters"] == 20
    assert replay["radial_points"] == 84
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
