import json
from pathlib import Path

import pytest

from sigma_theory_compiler.gravity_item41_stochastic_gravity_evaluator import check

ROOT = Path(__file__).resolve().parents[1]
RESULT = (
    ROOT
    / "runs/gravity/roadmap/item-41-stochastic-gravity-v1-source/compute-manifest.json"
)


@pytest.fixture(scope="module")
def result() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_search_used_gpu_without_opening_confirmation_data(result: dict) -> None:
    search = result["candidate_search"]
    assert search["backend"] == "gpu_cupy"
    assert search["device"] == "NVIDIA GeForce RTX 5090"
    assert search["candidate_point_evaluations"] == 25_945_530
    assert search["cpu_gpu_passed"] is True
    assert result["protocol"]["confirmation_values_read"] == 0
    assert result["protocol"]["post_response_candidate_cells"] == 0
    assert result["protocol"]["paid_model_calls"] == 0


def test_selected_retrospective_formula_and_controls_are_frozen(result: dict) -> None:
    selected = result["candidate_search"]["full_retrospective_candidate"]
    assert selected["candidate_id"] == 45024
    assert selected["lane"] == "einstein_langevin_white_field"
    assert selected["parameters"] == {
        "acceleration_exponent": 3.0,
        "radial_scale": 0.1,
        "sigma0": 0.56,
        "transition_u": 10000.0,
    }
    joint = result["joint_mean_variance_result"]
    assert joint["losses"]["candidate"] == pytest.approx(1.7493431539511528)
    assert joint["losses"]["homoskedastic"] == pytest.approx(1.3002769679474935)
    assert joint["losses"]["ordinary_heteroskedastic"] == pytest.approx(
        3.043543615339608
    )
    assert joint["strongest_control"] == "homoskedastic"


def test_empirical_mismatches_retain_formula_and_family(result: dict) -> None:
    joint = result["joint_mean_variance_result"]
    report = joint["counterexample_policy_report"]
    assessment = joint["counterexample_assessment"]
    assert report["raw_counterexample_count"] == 5
    assert report["uncertainty_resolved_counterexample_count"] == 3
    assert report["unchanged_independent_replication_failures"] == 0
    assert assessment["terminal_rejection_in_tested_scope"] is False
    assert assessment["candidate_pruned_globally"] is False
    assert assessment["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False


def test_item41_ghasp_result_replays_scientifically() -> None:
    replay = check(ROOT)
    assert replay["status"] == "ITEM41_GHASP_REPLAY_VALID"
    assert replay["confirmation_values_read"] == 0
    assert replay["paid_model_calls"] == 0
