from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item21_mixed_modes import (
    CONFIG_PATH,
    _candidate_digest,
    generate_candidates,
    load_config,
    mixing_occupation,
    validate_result,
    verify_sample_freeze,
    verify_science_freeze,
)

ROOT = Path(__file__).resolve().parents[1]


def _raw_config() -> dict:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_raw_niches_begin_equal_and_filtered_candidates_are_replayable() -> None:
    config = _raw_config()
    first = generate_candidates(config)
    second = generate_candidates(config)
    assert len(first["family"]) == 172_784
    assert _candidate_digest(first) == _candidate_digest(second)
    assert set(first["family"]) == {0, 1, 2, 3}
    assert np.min(first["kinetic_determinant"]) >= 0.05
    assert np.max(first["maximum_solar_fractional_force_deviation"]) <= 1.0e-5
    assert int(config["candidate_generator"]["raw_parameter_cells"]) % 8 == 0


def test_nonlinear_mixing_probabilities_are_bounded() -> None:
    log_u = np.linspace(-100.0, 100.0, 100_001)
    for family in (1, 2, 3):
        value = mixing_occupation(family, log_u, 1.0, 0.3)
        assert np.min(value) >= 0.0
        assert np.max(value) <= 1.0 + 1.0e-15
    crossing = mixing_occupation(1, log_u, 1.0, 0.3)
    assert crossing[0] < 1.0e-80 and crossing[-1] < 1.0e-80
    adiabatic = mixing_occupation(2, log_u, 1.0, 0.3)
    assert adiabatic[0] > 0.999 and adiabatic[-1] < 0.001


def test_linear_class_is_labeled_as_item19_equivalent() -> None:
    config = _raw_config()
    assert "Item19" in config["physics"]["linear_equivalence_theorem"]
    label = config["candidate_generator"]["creativity_labels"][
        "healthy_normalized_two_pole"
    ]
    assert label == "KNOWN_FORMULA_EQUIVALENT_TO_ITEM19"
    assert config["candidate_generator"]["historical_novelty_claimed"] is False
    assert config["scope"]["two_track_evaluation"] is True


def test_repository_freezes_and_result_replay() -> None:
    config = load_config(ROOT)
    verify_science_freeze(ROOT, config)
    verify_sample_freeze(ROOT, config)
    path = validate_result(ROOT)
    result = json.loads(path.read_text(encoding="utf-8"))
    assert result["data_source_receipt"]["confirmation_opened"] == 0
    assert result["frozen_boundary"]["post_response_candidate_cells"] == 0
    assert result["linear_equivalence_certificate"]["pass"] is True
    assert result["historical_novelty_claimed"] is False
