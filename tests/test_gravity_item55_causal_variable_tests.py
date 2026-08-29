from __future__ import annotations

from pathlib import Path

import numpy as np

from sigma_theory_compiler.gravity_item55_causal_variable_tests import (
    _classification_accuracy,
    _common_support,
    _overlap_coefficients,
    load_config,
)


ROOT = Path(__file__).resolve().parents[1]


def test_item55_freezes_observational_not_causal_claims() -> None:
    config = load_config(ROOT, require_bound=False)
    assert config["target_candidate"]["operands"] == ["geometry", "density"]
    assert config["claim_policy"]["observational_ablation_establishes_causality"] is False
    assert config["claim_policy"]["matched_observational_subset_establishes_intervention"] is False
    assert config["claim_policy"]["population_transfer_establishes_universality"] is False
    assert config["claim_policy"]["single_counterexample_terminal"] is False
    assert config["claim_policy"]["formula_family_pruned"] is False


def test_overlap_and_label_diagnostics_detect_separated_synthetic_populations() -> None:
    values = np.vstack((np.zeros((3, 6)), np.ones((3, 6)) * 4.0))
    populations = np.asarray(["S4TM"] * 3 + ["CLASH"] * 3)
    names = np.asarray([f"object-{index}" for index in range(6)])
    overlap = _overlap_coefficients(values, populations)
    accuracy, records = _classification_accuracy(values, populations)
    support = _common_support(values, populations, names, caliper=1.0)
    assert all(value == 0.0 for value in overlap.values())
    assert accuracy == 1.0
    assert all(row["correct"] for row in records)
    assert support["pairs_within_caliper"] == 0
