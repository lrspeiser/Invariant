from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler.gravity_item58_cluster_coefficient_gate import (
    CONFIG_PATH,
    GravityItem58Error,
    _fold_map,
    _metrics,
    _oracle_beta,
    prepare_records,
    validate_config,
)

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict[str, object]:
    return json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def test_unbound_scientific_contract_is_valid_before_freeze() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    assert config["population"]["new_response_rows_allowed"] == 0
    assert config["population"]["direct_lensing_likelihood_evaluations_allowed"] == 0
    assert config["counterexample_policy"]["single_counterexample_terminal"] is False
    assert config["claim_policy"]["failure_prunes_feature_or_formula_family"] is False


def test_response_leakage_tamper_is_rejected() -> None:
    corrupted = deepcopy(_config())
    corrupted["feature_lineage"]["gtot_or_lensing_mass_available_to_model"] = True
    with pytest.raises(GravityItem58Error, match="target leakage"):
        validate_config(ROOT, corrupted, require_bound=False)


def test_single_counterexample_and_family_pruning_tamper_is_rejected() -> None:
    corrupted = deepcopy(_config())
    corrupted["counterexample_policy"]["single_counterexample_terminal"] = True
    with pytest.raises(GravityItem58Error, match="over-pruning"):
        validate_config(ROOT, corrupted, require_bound=False)


def test_oracle_recovers_synthetic_coefficient_exactly() -> None:
    base = np.asarray([0.5, 1.0, 1.5, 2.0])
    component = np.asarray([0.2, 0.4, 0.7, 1.0])
    target = np.log10(base + 0.73 * component)
    beta, loss, at_boundary = _oracle_beta(
        base,
        component,
        target,
        np.ones_like(target),
        np.arange(0.0, 4.0 + 0.005, 0.01),
    )
    assert beta == pytest.approx(0.73)
    assert loss == pytest.approx(0.0, abs=1.0e-24)
    assert at_boundary is False


def test_fold_assignment_is_deterministic_and_balanced() -> None:
    names = [f"cluster-{index:02d}" for index in range(20)]
    first = _fold_map(names, salt="frozen-salt", folds=5)
    second = _fold_map(names, salt="frozen-salt", folds=5)
    assert first == second
    assert sorted(first.values()).count(0) == 4
    assert sorted(first.values()).count(1) == 4
    assert sorted(first.values()).count(2) == 4
    assert sorted(first.values()).count(3) == 4
    assert sorted(first.values()).count(4) == 4


def test_metrics_recognize_predictive_signal() -> None:
    target = np.asarray([0.2, 0.7, 1.3, 1.9, 2.4])
    prediction = target + np.asarray([0.01, -0.02, 0.01, -0.01, 0.02])
    metrics = _metrics(target, prediction)
    assert metrics["r2"] > 0.99
    assert metrics["mean_squared_error"] < 0.001


def test_real_records_replay_frozen_labels_without_feature_leakage() -> None:
    config = _config()
    validate_config(ROOT, config, require_bound=False)
    records = prepare_records(ROOT, config)
    assert len(records) == 20
    assert sum(record["point_count"] for record in records) == 84
    assert [record["name"] for record in records] == config["population"]["sample"]
    assert all(record["item1_beta_replay_delta"] <= 5.0e-12 for record in records)
    allowed = {
        feature
        for family in config["allowed_features"].values()
        for feature in family
    }
    assert all(set(record["features"]) == allowed for record in records)
    assert all(record["beta_at_grid_boundary"] is False for record in records)
