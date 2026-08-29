from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_sampler_diagnostic as diagnostic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stored() -> dict[str, object]:
    return json.loads((ROOT / diagnostic.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_stored_diagnostic_rebuilds_and_keeps_cp5_tasks_open(
    stored: dict[str, object],
) -> None:
    diagnostic.validate_receipt(stored, ROOT)
    assert stored["decision"] == (
        "CORRELATION_AWARE_MIXING_IMPROVED_NOT_CONVERGED_REQUIRES_REPARAMETERIZATION_OR_INDEPENDENT_PRIORS"
    )
    assert stored["completed_goal_evidence"] == {}
    assert set(stored["blocked_goal_evidence"]) == {
        "CP5.7",
        "CP5.8",
        "CP5.9",
        "CP5.10",
    }
    assert stored["claims"]["posterior_sampler_converged"] is False
    assert stored["claims"]["development_nuisance_marginalization_complete"] is False
    assert stored["claims"]["CP5_7_through_CP5_10_complete"] is False


def test_diagnostic_progression_records_improvement_without_false_convergence(
    stored: dict[str, object],
) -> None:
    runs = {row["run_id"]: row for row in stored["diagnostic_runs"]}
    assert tuple(runs) == diagnostic.RUN_IDS
    assert runs["SOBOL_PRIOR_IMPORTANCE_4096"]["result"][
        "importance_effective_samples"
    ] == 1.0
    component = runs["COMPONENTWISE_LONG_500"]["result"]
    short = runs["AFFINE_4X36_400_400"]["result"]
    long = runs["AFFINE_4X36_1200_800"]["result"]
    assert component["maximum_rhat"] == pytest.approx(17.37645796502024)
    assert long["minimum_effective_samples"] > short["minimum_effective_samples"] > 50
    assert 1.2 < long["maximum_rhat"] < short["maximum_rhat"]
    assert long["maximum_standardized_between_ensemble_median_spread"] > 0.25
    assert set(long["parameters_above_rhat_threshold"]) == set(
        diagnostic.uncertainty.PARAMETERS
    )


def test_sample_seal_and_evaluation_ledger_are_explicit(stored: dict[str, object]) -> None:
    assert stored["sample_seal"] == {
        "likelihood_split": "development_train",
        "predictive_split": "development_holdout",
        "same_release_confirmation_rows_used": False,
        "independent_source_rows_used": False,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }
    assert stored["counts"] == {
        "diagnostic_runs": 4,
        "candidate_forward_evaluations": 501636,
        "largest_affine_posterior_draws": 115200,
        "nuisance_dimensions": 17,
        "parameters_passing_extended_affine_rhat": 0,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }
    assert stored["reproduction"]["check_reexecutes_expensive_numeric_runs"] is False
    assert stored["reproduction"]["replay_compares_against_frozen_observation"] is True


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["unchanged_completion_thresholds"].__setitem__(
                "maximum_rhat", 2.0
            ),
            "threshold",
        ),
        (
            lambda value: value["sample_seal"].__setitem__("target_rows_opened", 1),
            "sample seal",
        ),
        (
            lambda value: value["adjudication"].__setitem__(
                "CP5_7_through_CP5_10_complete", True
            ),
            "adjudication",
        ),
        (
            lambda value: value["diagnostic_runs"][-1]["result"].__setitem__(
                "converged", True
            ),
            "promoted",
        ),
    ],
)
def test_threshold_access_completion_and_result_mutations_fail_closed(
    mutation: object, match: str
) -> None:
    config = copy.deepcopy(diagnostic.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(diagnostic.GravityClusterNuisanceDiagnosticError, match=match):
        diagnostic.validate_config(config, ROOT)
