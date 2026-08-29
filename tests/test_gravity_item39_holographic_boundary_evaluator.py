from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    boundary_coordinates,
    generate_raw_candidates,
    load_config,
    predict_multiplier,
)
from sigma_theory_compiler.gravity_item39_holographic_boundary_evaluator import (
    _candidate_log_velocity_batch,
    _paired_sign_flip,
    _ridge_fit,
    _ridge_predict,
    check,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item39_evaluator_candidate_kernel_matches_frozen_equations() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    ids = np.asarray([0, 65536, 131072, 196608])
    candidates = {key: value[ids] for key, value in raw.items()}
    u = np.asarray([1e-4, 1e-2, 1.0])
    fraction = np.asarray([0.1, 0.5, 0.9])
    radius = np.asarray([0.2, 0.6, 1.0])
    slope = np.asarray([0.5, 2.0, 3.0])
    vbar = np.asarray([20.0, 80.0, 150.0])
    arrays = {
        "u": u,
        "vbar": vbar,
        "h": boundary_coordinates(fraction, radius, slope),
    }
    observed = _candidate_log_velocity_batch(candidates, 0, 4, arrays, config, np)
    multiplier = predict_multiplier(candidates, u, fraction, radius, slope, config)
    expected = np.log10(vbar)[None, :] + 0.5 * np.log10(multiplier)
    assert np.allclose(observed, expected, rtol=0.0, atol=1e-14)


def test_item39_weighted_ridge_reproduces_linear_signal() -> None:
    x = np.linspace(-2.0, 2.0, 40)
    design = np.column_stack((x, x**2))
    target = 1.2 + 0.7 * x - 0.3 * x**2
    model = _ridge_fit(design, target, np.ones_like(target), 1e-12)
    prediction = _ridge_predict(model, design)
    assert np.max(np.abs(prediction - target)) < 1e-10


def test_item39_paired_sign_flip_is_seeded_and_nonparametric() -> None:
    config = load_config(ROOT)
    differences = -np.linspace(0.1, 1.0, 20)
    first = _paired_sign_flip(differences, config)
    second = _paired_sign_flip(differences, config)
    assert first == second
    assert 0.0 < first["p_value"] <= 0.05
    assert first["selection_aware"] is False


def test_item39_full_gpu_result_replays_ignoring_only_measured_runtime() -> None:
    result = check(ROOT)
    assert result["status"] == "ITEM39_COMPUTE_REPLAY_VALID"
    assert result["confirmation_response_rows"] == 0
    assert result["paid_model_calls"] == 0
