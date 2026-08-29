from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item41_stochastic_gravity import (
    admissible_candidates,
    build_candidate_manifest,
    build_exposure_manifest,
    decode_candidate,
    generate_raw_candidates,
    load_config,
    stochastic_moments,
)

ROOT = Path(__file__).resolve().parents[1]


def test_config_freezes_retrospective_and_counterexample_boundaries() -> None:
    config = load_config(ROOT)
    assert config["item"] == 41
    assert config["scope"]["fresh_confirmation_claimed"] is False
    assert config["scope"]["confirmation_opening_authorized"] is False
    assert config["discovery_policy"][
        "single_empirical_counterexample_is_not_a_formula_or_family_veto"
    ]
    assert config["candidate_generator"]["post_response_cells"] == 0


def test_raw_grid_has_four_equal_niches() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    assert len(raw["candidate_id"]) == 262_144
    assert [int(np.sum(raw["lane"] == lane)) for lane in range(4)] == [65_536] * 4
    assert decode_candidate(0, config)["lane"] == "einstein_langevin_white_field"
    assert decode_candidate(196_608, config)["lane"] == "two_state_vacuum_telegraph"


def test_stochastic_moments_predict_drift_and_variance() -> None:
    config = load_config(ROOT)
    raw = generate_raw_candidates(config)
    ids = [0, 65_536, 131_072, 196_608]
    rows = {key: value[ids] for key, value in raw.items()}
    drift, variance = stochastic_moments(
        rows,
        np.asarray([1e-3, 1e-3, 1e-3]),
        np.asarray([0.5, 1.0, 2.0]),
        config,
    )
    assert drift.shape == variance.shape == (4, 3)
    assert np.all(variance > 0.0)
    assert np.allclose(drift[0], 0.0)
    assert np.all(drift[1] < 0.0)
    assert np.all(drift[2] >= 0.0)


def test_admission_and_manifests_are_response_safe() -> None:
    config = load_config(ROOT)
    admitted, audit = admissible_candidates(config)
    assert audit["raw_candidates"] == 262_144
    assert audit["admitted_candidates"] > 1_000
    assert len(admitted["candidate_id"]) == audit["admitted_candidates"]
    candidate = build_candidate_manifest(ROOT)
    exposure = build_exposure_manifest(ROOT)
    assert candidate["response_accessed_during_generation"] is False
    assert exposure["counts"]["retrospective_exploration_galaxies"] == 15
    assert exposure["counts"]["paired_radial_points"] == 71
    assert exposure["counts"]["confirmation_values_read"] == 0
