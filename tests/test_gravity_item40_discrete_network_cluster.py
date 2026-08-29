import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item40_discrete_network_cluster import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "runs/gravity/roadmap/item-40-discrete-network-v1-source/clash-transfer-result.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_transfer_is_bound_unchanged_and_not_confirmation(result: dict) -> None:
    assert result["protocol"]["dynamics_commit"] == (
        "41400cebe74fe5e0cf9fe98e2a9b501626ff8ddf"
    )
    assert result["selected_formula"]["candidate_id"] == 255184
    assert result["protocol"]["selection_use"] is False
    assert result["protocol"]["retuning"] is False
    assert result["protocol"]["post_selection_candidate_cells"] == 0
    assert result["claim_boundaries"]["fresh_confirmation"] is False


def test_clash_counts_and_losses_are_frozen(result: dict) -> None:
    assert result["data"]["clusters"] == 20
    assert result["data"]["radial_points"] == 84
    assert result["losses"]["candidate"] == pytest.approx(69.8986358084656)
    assert result["losses"]["baryonic_newton"] == pytest.approx(134.39005516520672)
    assert result["losses"]["mond_RAR"] == pytest.approx(41.209299493402725)
    assert result["improvement_vs_strongest_percent"] == pytest.approx(
        -69.61859742278756
    )


def test_cluster_mismatches_do_not_prune_formula_or_family(result: dict) -> None:
    report = result["counterexample_policy_report"]
    assessment = result["counterexample_assessment"]
    assert report["raw_counterexample_count"] == 20
    assert report["uncertainty_resolved_counterexample_count"] == 0
    assert assessment["terminal_rejection_in_tested_scope"] is False
    assert assessment["candidate_pruned_globally"] is False
    assert assessment["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False


def test_cluster_transfer_replays_exactly() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM40_CLASH_TRANSFER_REPLAY_VALID"
    assert replay["clusters"] == 20
    assert replay["radial_points"] == 84
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
