"""Hard controls for the frozen G0 SPARC discovery experiment.

G0 is allowed to pass only when the real exploration population, contiguous radial
holdouts, comparator ordering, candidate evaluator, and checked receipt all agree.
These tests deliberately mutate targets and receipts to ensure the controls detect the
failure modes they claim to exclude.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sigma_theory_compiler.gravity_g0_experiment import (
    OUTPUT_PATH,
    GravityG0Error,
    _fit_nfw_fold,
    _galaxy_arrays,
    baseline_replay,
    load_config,
    radial_folds,
    score_predictions,
    throughput_benchmark,
    validate_receipt,
)
from sigma_theory_compiler.sparc_full_sample import assemble

ROOT = Path(__file__).resolve().parents[1]
_CACHE: dict[str, Any] = {}


def _population() -> Any:
    if "population" not in _CACHE:
        _CACHE["population"] = assemble(ROOT)
    return _CACHE["population"]


def _config() -> Any:
    if "config" not in _CACHE:
        _CACHE["config"] = load_config(ROOT)
    return _CACHE["config"]


def _baselines() -> Any:
    if "baselines" not in _CACHE:
        _CACHE["baselines"] = baseline_replay(_population(), _config())
    return _CACHE["baselines"]


def test_contract_is_bound_to_the_real_full_sparc_asset() -> None:
    config = _config()
    population = _population()
    assert config["dataset"]["published_galaxies"] == 175
    assert config["dataset"]["published_points"] == 3391
    assert len(population.split.exploration) == 140
    assert len(population.exploration) == 139
    assert sum(galaxy.count for galaxy in population.exploration) == 2720
    assert len(population.split.confirmation) == 35


@pytest.mark.parametrize("row_count", [4, 5, 6, 17, 91])
def test_contiguous_folds_hold_out_every_ordered_radius_once(row_count: int) -> None:
    folds = radial_folds(row_count)
    assert [index for fold in folds for index in fold.holdout] == list(range(row_count))
    for fold in folds:
        assert list(fold.holdout) == list(
            range(min(fold.holdout), max(fold.holdout) + 1)
        )
        assert set(fold.holdout).isdisjoint(fold.training)
        assert sorted((*fold.holdout, *fold.training)) == list(range(row_count))
        assert len(fold.training) >= 3


def test_fold_control_rejects_too_few_training_rows() -> None:
    with pytest.raises(GravityG0Error, match="cannot leave"):
        radial_folds(3)


def test_nfw_fit_does_not_read_held_out_velocity_targets() -> None:
    galaxy = _population().exploration[0]
    arrays = _galaxy_arrays(galaxy)
    fold = radial_folds(galaxy.count)[0]
    before = _fit_nfw_fold(arrays, fold.training, 64)
    mutated = {key: value.copy() for key, value in arrays.items()}
    mutated["vobs"][list(fold.holdout)] *= 1000.0
    mutated["sigma"][list(fold.holdout)] *= 0.001
    after = _fit_nfw_fold(mutated, fold.training, 64)
    assert after == before


def test_baselines_use_all_exploration_rows_and_no_confirmation_galaxy() -> None:
    replay = _baselines()
    assert replay["evaluated_galaxies"] == 139
    assert replay["evaluated_points"] == 2720
    assert replay["confirmation_evaluator_access_count"] == 0
    assert replay["fold_count"] == sum(
        len(radial_folds(galaxy.count)) for galaxy in _population().exploration
    )


def test_known_baselines_have_the_preregistered_real_data_ordering() -> None:
    aggregate = _baselines()["aggregate"]
    rar = float(aggregate["empirical_rar"]["chi_square"])
    newton = float(aggregate["newtonian_baryons"]["chi_square"])
    wrong = float(aggregate["wrong_high_acceleration_boost"]["chi_square"])
    nfw = float(aggregate["nfw_halo_ceiling"]["chi_square"])
    assert rar < newton
    assert rar < wrong
    assert nfw < newton


def test_score_exact_match_and_prediction_mutation_controls() -> None:
    observed = np.asarray([10.0, 20.0, 30.0])
    sigma = np.asarray([1.0, 2.0, 3.0])
    exact = score_predictions(observed.copy(), observed, sigma)
    shifted = score_predictions(observed + sigma, observed, sigma)
    assert float(exact["chi_square"]) == 0.0
    assert float(exact["coverage_one_sigma"]) == 1.0
    assert float(shifted["chi_square"]) == 3.0
    assert float(shifted["mean_squared_standardized_residual"]) == 1.0


def test_score_refuses_shape_mismatch_and_nonfinite_prediction() -> None:
    with pytest.raises(GravityG0Error, match="different shapes"):
        score_predictions(np.ones(2), np.ones(3), np.ones(3))
    with pytest.raises(GravityG0Error, match="non-finite"):
        score_predictions(np.asarray([np.nan]), np.ones(1), np.ones(1))


def test_actual_candidate_evaluator_runs_on_every_exploration_point_on_cpu() -> None:
    benchmark = throughput_benchmark(_population(), _config(), candidate_count=32, use_gpu=False)
    assert benchmark["candidate_count"] == 32
    assert benchmark["point_count"] == 2720
    assert benchmark["fp32_point_evaluations"] == 32 * 2720
    assert benchmark["uses_actual_sparc_radii_baryons_targets_and_sigmas"] is True
    assert benchmark["fp64_gpu_cpu_decision_mismatches"] == 0
    assert float(benchmark["cpu_fp64_candidate_replays_per_second"]) > 0.0
    assert float(benchmark["cpu_fp64_point_evaluations_per_second"]) > 0.0
    assert benchmark["gpu_memory_pool_peak_increment_bytes"] == 0


def test_checked_pass_receipt_is_bound_to_source_test_config_and_dataset() -> None:
    path = ROOT / OUTPUT_PATH
    assert path.is_file(), "the checked G0 PASS receipt must be generated before G0 is complete"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    validate_receipt(receipt, root=ROOT)
    assert receipt["decision"] == "PASS_G0_EXPERIMENT_FROZEN"
    assert receipt["formula_search_authorized_after_pass"] is True


def test_receipt_control_rejects_a_resealed_false_confirmation_access_claim() -> None:
    path = ROOT / OUTPUT_PATH
    assert path.is_file(), "the checked G0 PASS receipt must be generated before G0 is complete"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(receipt)
    tampered["counts"]["confirmation_evaluator_accesses"] = 1
    from sigma_theory_compiler.sigma_core import canonical_sha256

    tampered.pop("content_sha256")
    tampered["content_sha256"] = canonical_sha256(tampered)
    with pytest.raises(GravityG0Error, match="confirmation evaluator access"):
        validate_receipt(tampered, root=ROOT)
