from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import load_config
from sigma_theory_compiler.gravity_item42_matter_geometry_feedback_evaluator import (
    _candidate_log_velocity_batch,
    _candidate_pools,
    check,
)

ROOT = Path(__file__).resolve().parents[1]


def test_predictor_specific_pool_preserves_galaxies_and_removes_only_bad_cells() -> None:
    config = load_config(ROOT)
    candidates, no_feedback, audit = _candidate_pools(ROOT, config)
    assert audit["admitted_candidates"] == 165_242
    assert audit["predictor_specific_nonconvergent_candidates_removed"] == 22_792
    assert audit["predictor_specific_admitted_candidates"] == 142_450
    assert audit["valid_feedback_cells"] == 56
    assert len(candidates["candidate_id"]) == 142_450
    assert len(no_feedback["candidate_id"]) == 2_849
    assert np.all(no_feedback["feedback_index"] == 0)


def test_candidate_prediction_is_finite_and_uses_feedback_coordinate() -> None:
    config = load_config(ROOT)
    candidates, _, _ = _candidate_pools(ROOT, config)
    rows = {key: value[:2] for key, value in candidates.items()}
    arrays = {
        "u": np.asarray([0.01, 1.0, 100.0]),
        "vbar": np.asarray([20.0, 40.0, 80.0]),
        "h": np.full((4, 16, 3), 0.5),
    }
    prediction = _candidate_log_velocity_batch(rows, 0, 2, arrays, config, np)
    assert prediction.shape == (2, 3)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > np.log10(arrays["vbar"])[None, :])


def test_sample_remains_response_blind_with_confirmation_sealed() -> None:
    import json

    path = (
        ROOT
        / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1-source/sample-manifest.json"
    )
    sample = json.loads(path.read_text(encoding="utf-8"))
    assert sample["counts"]["exploration"] == 21
    assert sample["counts"]["reserved_confirmation"] == 5
    assert sample["counts"]["response_rows_read"] == 0
    assert sample["counts"]["confirmation_rows_read"] == 0
    assert sample["claims"]["failed_identity_replacement"] is False


def test_fresh_result_is_partial_and_keeps_confirmation_sealed() -> None:
    import json

    path = (
        ROOT
        / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1-source/compute-manifest.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    search = result["candidate_search"]
    assert search["backend"] == "gpu_cupy"
    assert search["device"] == "NVIDIA GeForce RTX 5090"
    assert search["candidate_point_evaluations"] == 47_293_400
    assert search["cpu_gpu_passed"] is True
    assert search["full_exploration_candidate"]["candidate_id"] == 170142
    assert search["full_exploration_candidate"]["lane"] == (
        "geometry_gradient_reorganization"
    )
    losses = result["primary_dynamics"]["losses"]
    assert losses["candidate"] == pytest.approx(5.415278020616004)
    assert losses["matched_no_feedback"] == pytest.approx(5.503484025796563)
    assert losses["gas_only_mond_RAR"] == pytest.approx(3.3306295999592517)
    assert result["protocol"]["confirmation_response_rows"] == 0
    assert result["protocol"]["post_response_candidate_cells"] == 0
    assert result["protocol"]["post_response_implementation_repair"][
        "response_points_removed"
    ] == 0


def test_counterexamples_are_retained_and_result_replays() -> None:
    import json

    path = (
        ROOT
        / "runs/gravity/roadmap/item-42-matter-geometry-feedback-v1-source/compute-manifest.json"
    )
    result = json.loads(path.read_text(encoding="utf-8"))
    primary = result["primary_dynamics"]
    assert primary["counterexample_policy_report"]["raw_counterexample_count"] == 6
    assert primary["counterexample_policy_report"][
        "uncertainty_resolved_counterexample_count"
    ] == 4
    assert primary["counterexample_assessment"]["terminal_rejection_in_tested_scope"] is False
    assert primary["counterexample_assessment"]["formula_family_pruned"] is False
    assert result["claim_boundaries"]["one_empirical_counterexample_is_veto"] is False
    replay = check(ROOT)
    assert replay["status"] == "ITEM42_DYNAMICS_REPLAY_VALID"
    assert replay["confirmation_response_rows"] == 0
    assert replay["paid_model_calls"] == 0
