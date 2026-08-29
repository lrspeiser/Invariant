from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_g0_experiment import radial_folds
from sigma_theory_compiler.gravity_item56_disk_galaxy_gate import (
    build_aggregate_result,
    build_evaluation_result,
    build_preflight_manifest,
    candidate_velocity,
    load_config,
)

ROOT = Path(__file__).resolve().parents[1]


def test_item56_freezes_one_nonleaking_candidate_and_nonterminal_failures() -> None:
    config = load_config(ROOT, require_bound=False)
    assert config["target_candidate"]["candidate_id"] == 135082
    assert config["target_candidate"]["fitted_gravitational_parameters_on_sparc"] == 0
    assert config["predictor_contract"]["galaxy_identifier_allowed_as_predictor"] is False
    assert config["predictor_contract"]["observed_velocity_allowed_as_predictor"] is False
    assert config["data_boundary"]["confirmation_response_rows_allowed"] == 0
    assert config["counterexample_policy"]["single_counterexample_terminal"] is False
    assert config["counterexample_policy"]["counterexample_count_terminal"] is False
    assert config["claim_policy"]["formula_family_pruned_on_failure"] is False


def test_candidate_is_positive_finite_and_geometry_sensitive_at_fixed_density() -> None:
    config = load_config(ROOT, require_bound=False)
    radius = np.asarray([0.5, 2.0])
    a0 = float(config["predictor_contract"]["acceleration_scale_km2_s2_kpc"])
    vbar2 = radius * a0 * 0.1
    prediction = candidate_velocity(radius, vbar2, 1.0, config["target_candidate"], a0)
    assert np.all(np.isfinite(prediction))
    assert np.all(prediction > 0.0)
    assert not np.isclose(prediction[0] / np.sqrt(vbar2[0]), prediction[1] / np.sqrt(vbar2[1]))


def test_frozen_contiguous_folds_hold_every_radius_once() -> None:
    folds = radial_folds(17, maximum_folds=5, minimum_training_rows=3)
    assert [index for fold in folds for index in fold.holdout] == list(range(17))
    assert all(tuple(sorted(fold.holdout)) == fold.holdout for fold in folds)


def test_recorded_item56_disk_gate_is_exactly_replayable() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    preflight = json.loads(
        (source / config["paths"]["preflight_manifest"]).read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        (source / config["paths"]["evaluation_result"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads((ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8"))
    assert preflight == build_preflight_manifest(ROOT)
    assert evaluation == build_evaluation_result(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert aggregate["decision"] == (
        "ITEM56_DISK_GALAXY_GATE_NOT_PASSED_LEAD_AND_FAILURES_RETAINED"
    )
    assert aggregate["counts"]["exploration_galaxies"] == 139
    assert aggregate["counts"]["exploration_rows"] == 2720
    assert aggregate["counts"]["confirmation_response_rows"] == 0
    assert aggregate["candidate_galaxy_wins_vs_newton"] == 23
    assert aggregate["candidate_galaxy_wins_vs_rar"] == 3
    assert all(value is False for key, value in aggregate["gates"].items() if "zero" not in key)
    assert aggregate["claims"]["formula_family_pruned"] is False
    assert aggregate["claims"]["single_counterexample_used_as_veto"] is False
