from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import load_config
from sigma_theory_compiler.gravity_item42_matter_geometry_feedback_evaluator import (
    _candidate_log_velocity_batch,
    _candidate_pools,
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
