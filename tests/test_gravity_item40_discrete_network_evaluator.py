import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item40_discrete_network_evaluator import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "runs/gravity/roadmap/item-40-discrete-network-v1-source/compute-manifest.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_search_used_gpu_and_kept_confirmation_sealed(result: dict) -> None:
    search = result["candidate_search"]
    assert search["backend"] == "gpu_cupy"
    assert search["device"] == "NVIDIA GeForce RTX 5090"
    assert search["candidate_point_evaluations"] == 31_034_992
    assert search["cpu_gpu_passed"] is True
    assert result["protocol"]["confirmation_response_rows"] == 0
    assert result["protocol"]["post_response_candidate_cells"] == 0
    assert result["protocol"]["paid_model_calls"] == 0


def test_selected_formula_and_losses_are_frozen(result: dict) -> None:
    selected = result["candidate_search"]["full_exploration_candidate"]
    assert selected["candidate_id"] == 255184
    assert selected["lane"] == "nonlocal_communicability"
    assert selected["parameters"] == {
        "amplitude": 4.0,
        "exponent": 0.4,
        "shape": 0.2,
        "transition_u": 1000.0,
    }
    losses = result["primary_dynamics"]["losses"]
    assert losses["candidate"] == pytest.approx(6.033836303285424)
    assert losses["matched_ordinary_geometry"] == pytest.approx(6.0530621372524545)
    assert losses["mond_RAR"] == pytest.approx(3.8808634865974874)
    assert losses["item39_selected"] == pytest.approx(3.1822944136258644)


def test_counterexamples_are_retained_not_used_as_a_kill_switch(result: dict) -> None:
    primary = result["primary_dynamics"]
    report = primary["counterexample_policy_report"]
    assessment = primary["counterexample_assessment"]
    assert report["raw_counterexample_count"] == 7
    assert report["uncertainty_resolved_counterexample_count"] == 2
    assert assessment["status"] == "ROBUST_SCOPED_NEGATIVE_EVIDENCE"
    assert assessment["terminal_rejection_in_tested_scope"] is False
    assert assessment["candidate_pruned_globally"] is False
    assert assessment["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False


def test_result_replays_scientifically() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM40_DYNAMICS_REPLAY_VALID"
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
