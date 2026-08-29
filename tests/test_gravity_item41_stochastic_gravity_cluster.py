import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item41_stochastic_gravity_cluster import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "runs/gravity/roadmap/item-41-stochastic-gravity-v1-source/clash-transfer-result.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_transfer_is_commit_bound_unchanged_and_not_confirmation(result: dict) -> None:
    assert result["protocol"]["dynamics_commit"] == (
        "c0e2bcea03c894209b1d0fca862d2e91cbe731c2"
    )
    assert result["selected_formula"]["candidate_id"] == 45024
    assert result["protocol"]["selection_use"] is False
    assert result["protocol"]["retuning"] is False
    assert result["protocol"]["post_selection_candidate_cells"] == 0
    assert result["claim_boundaries"]["fresh_confirmation"] is False


def test_clash_counts_variance_and_losses_are_frozen(result: dict) -> None:
    assert result["data"]["clusters"] == 20
    assert result["data"]["radial_points"] == 84
    assert result["process_variance_range"]["minimum"] == pytest.approx(0.3136)
    assert result["process_variance_range"]["maximum"] == pytest.approx(0.3136)
    losses = result["losses"]
    assert losses["candidate_on_mond_background"] == pytest.approx(
        1.4045696202258737
    )
    assert losses["mond_ordinary_heteroskedastic_loco"] == pytest.approx(
        0.6443781566078153
    )
    assert losses["candidate_on_baryonic_newton_background"] == pytest.approx(
        5.264900188047875
    )


def test_cluster_mismatches_narrow_but_do_not_kill_formula(result: dict) -> None:
    report = result["counterexample_policy_report"]
    assessment = result["counterexample_assessment"]
    assert report["raw_counterexample_count"] == 18
    assert report["uncertainty_resolved_counterexample_count"] == 17
    assert report["unchanged_independent_replication_failures"] == 0
    assert assessment["terminal_rejection_in_tested_scope"] is False
    assert assessment["candidate_pruned_globally"] is False
    assert assessment["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False


def test_clash_transfer_replays_exactly() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM41_CLASH_TRANSFER_REPLAY_VALID"
    assert replay["clusters"] == 20
    assert replay["radial_points"] == 84
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
