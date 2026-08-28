from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item20_massive_phase import (
    CONFIG_PATH,
    _candidate_digest,
    generate_candidates,
    load_config,
    phase_occupation,
    validate_result,
    verify_sample_freeze,
    verify_science_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_candidates_are_deterministic_equal_niche_and_locally_safe() -> None:
    config = _raw_config()
    first = generate_candidates(config)
    second = generate_candidates(config)
    assert len(first["family"]) == 262_143
    assert _candidate_digest(first) == _candidate_digest(second)
    assert set(first["family"]) == {0, 1, 2}
    raw = int(config["candidate_generator"]["raw_parameter_cells"])
    for family in range(3):
        assert abs(int(np.sum(first["family"] == family)) - raw / 3) <= 3
        for sign in (-1.0, 1.0):
            assert np.sum((first["family"] == family) & (first["sign"] == sign)) > 43_000
    assert np.max(first["maximum_solar_phase_force_fraction"]) <= 1.0e-5


def test_phase_families_are_bounded_resonant_and_target_blind() -> None:
    config = _raw_config()
    z = np.logspace(-12, 12, 20_001)
    for family in range(3):
        value = phase_occupation(family, z, 2.0, 2.0, 0.75)
        assert np.min(value) >= 0.0
        assert np.max(value) <= 1.0
        assert value[0] < 1.0e-20
        assert value[-1] < 1.0e-20
    assert "observed_rotation_frequency_as_phase_trigger" in config["scope"]["forbidden_inputs"]
    assert "baryonic predictors" in config["physics"]["phase_trigger"]
    assert config["candidate_generator"]["historical_novelty_claimed"] is False


def test_two_tracks_do_not_require_the_same_claim() -> None:
    config = _raw_config()
    assert config["scope"]["two_track_evaluation"] is True
    assert config["publication_track_gates"]["requires_full_cross_scale_gravity_solution"] is False
    assert config["publication_track_gates"]["requires_fresh_independent_confirmation_before_paper_claim"] is True


def test_repository_freezes_and_result_replay() -> None:
    config = load_config(ROOT)
    verify_science_freeze(ROOT, config)
    verify_sample_freeze(ROOT, config)
    path = validate_result(ROOT)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["data_source_receipt"]["confirmation_opened"] == 0
    assert result["frozen_boundary"]["post_response_candidate_cells"] == 0
    assert result["historical_novelty_claimed"] is False
    assert result["claim_ceiling"] == config["scope"]["claim_ceiling"]
