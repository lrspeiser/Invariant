from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def stored() -> dict[str, object]:
    return json.loads((ROOT / uncertainty.OUTPUT_PATH).read_text(encoding="utf-8"))


def test_stored_program_rebuilds_and_does_not_overclaim_failed_marginalization(
    stored: dict[str, object],
) -> None:
    uncertainty.validate_receipt(stored, ROOT)
    assert stored["decision"] == (
        "DEVELOPMENT_NUISANCE_SAMPLER_NOT_CONVERGED_SOURCE_COVARIANCE_BLOCKED"
    )
    assert stored["claims"]["development_nuisance_marginalization_complete"] is False
    assert stored["claims"]["candidate_posterior_sampler_converged"] is False
    assert stored["claims"]["full_source_covariance_complete"] is False
    assert stored["claims"]["independent_replication"] is False
    assert stored["sample"]["target_rows_opened"] == 0
    assert set(stored["completed_goal_evidence"]) == {"CP5.12", "CP5.14"}


def test_covariance_and_missingness_sensitivities_are_exhaustive_and_nonselective(
    stored: dict[str, object],
) -> None:
    covariance = stored["covariance_sensitivity"]
    assert covariance["total_scenarios"] == 36
    assert covariance["candidate_beats_nfw_scenarios"] == 36
    assert covariance["full_source_covariance_claimed"] is False
    missingness = stored["missingness_sensitivity"]
    assert len(missingness["scenarios"]) == 12
    assert "never_select_or_exclude" in missingness["status"]
    assert stored["counts"] == {
        "continuous_nuisance_parameters": 17,
        "quasi_monte_carlo_initial_samples_per_family": 1024,
        "posterior_samples_per_family": 256,
        "families_marginalized": 2,
        "forward_evaluations": 21640,
        "covariance_sensitivity_scenarios": 36,
        "missingness_sensitivity_scenarios": 12,
        "completed_CP5_tasks": 2,
        "blocked_CP5_tasks": 12,
        "target_rows_opened": 0,
    }


def test_all_declared_nuisance_causes_are_reported_as_indistinguishable(
    stored: dict[str, object],
) -> None:
    config = uncertainty.load_config(ROOT)
    expected = sorted({row["cause"] for row in config["continuous_priors"]})
    observed = stored["observational_indistinguishability"]
    assert observed["unique_cause_identified"] is False
    assert (
        observed["causes_remaining_indistinguishable_with_current_single_source_diagonal_errors"]
        == expected
    )
    assert observed["merger_state_comparison_complete"] is False


def test_target_access_covariance_overclaim_and_holdout_selection_fail_closed() -> None:
    config = uncertainty.load_config(ROOT)
    opened = copy.deepcopy(config)
    opened["sample_contract"]["target_rows_opened"] = 1
    with pytest.raises(uncertainty.GravityClusterUncertaintyError, match="sample seal"):
        uncertainty.validate_config(opened, ROOT)

    covariance = copy.deepcopy(config)
    covariance["covariance_stress"]["full_source_covariance_claimed"] = True
    with pytest.raises(uncertainty.GravityClusterUncertaintyError, match="covariance"):
        uncertainty.validate_config(covariance, ROOT)

    selected = copy.deepcopy(config)
    selected["quasi_monte_carlo"]["selection_uses_holdout"] = True
    with pytest.raises(uncertainty.GravityClusterUncertaintyError, match="Monte Carlo"):
        uncertainty.validate_config(selected, ROOT)
