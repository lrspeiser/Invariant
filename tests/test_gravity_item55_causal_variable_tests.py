from __future__ import annotations

from pathlib import Path
import json

import numpy as np

from sigma_theory_compiler.gravity_item55_causal_variable_tests import (
    _classification_accuracy,
    _common_support,
    _overlap_coefficients,
    build_aggregate_result,
    build_diagnostic_result,
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


def test_recorded_item55_diagnostics_are_exactly_replayable() -> None:
    config = load_config(ROOT)
    source = ROOT / config["paths"]["source_dir"]
    diagnostic = json.loads(
        (source / config["paths"]["diagnostic_result"]).read_text(encoding="utf-8")
    )
    aggregate = json.loads(
        (ROOT / config["paths"]["aggregate_result"]).read_text(encoding="utf-8")
    )
    assert diagnostic == build_diagnostic_result(ROOT)
    assert aggregate == build_aggregate_result(ROOT)
    assert diagnostic["scores"]["item45_universal_interaction"]["balanced_loss"] < diagnostic[
        "scores"
    ]["population_label_only"]["balanced_loss"]
    assert diagnostic["within_population_axis_ablations"]["geometry"][
        "relative_balanced_loss_increase"
    ] > 0.30
    assert diagnostic["object_level_population_overlap"]["density"] == 0.0
    assert diagnostic["population_label_predictability"][
        "leave_one_object_out_nearest_centroid_accuracy"
    ] == 1.0
    assert diagnostic["common_support"]["pairs_within_caliper"] == 0
    assert aggregate["claims"]["causality_established"] is False
    assert aggregate["claims"]["formula_family_pruned"] is False
