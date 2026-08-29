from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_identifiability_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def test_stored_audit_rebuilds_and_closes_more_sampling_branch() -> None:
    stored = json.loads((ROOT / audit.OUTPUT_PATH).read_text(encoding="utf-8"))
    audit.validate_receipt(stored, ROOT)
    assert stored["decision"] == audit.DECISION
    assert stored["completed_goal_evidence"] == {}
    assert set(stored["blocked_goal_evidence"]) == {
        "CP5.7",
        "CP5.8",
        "CP5.9",
        "CP5.10",
    }
    assert stored["claims"]["tempered_smc_mechanics_passed"] is True
    assert stored["claims"]["full_posterior_rejuvenation_completed"] is True
    assert stored["claims"]["more_sampling_alone_supported"] is False
    assert stored["claims"]["posterior_sampler_converged"] is False
    assert stored["claims"]["newtonian_control_run"] is False


def test_evaluation_ledger_and_unchanged_gates_are_explicit() -> None:
    stored = audit.build_receipt(ROOT)
    assert stored["unchanged_completion_thresholds"] == {
        "maximum_rhat": 1.2,
        "minimum_effective_samples": 50,
        "maximum_standardized_between_replicate_median_spread": 0.25,
        "all_17_parameters_must_pass": True,
    }
    assert stored["counts"] == {
        "diagnostic_runs": 2,
        "new_candidate_forward_evaluations": 722944,
        "cumulative_candidate_forward_evaluations_with_predecessor": 1224580,
        "largest_posterior_draws": 131072,
        "nuisance_dimensions": 17,
        "parameters_passing_rejuvenated_rhat": 0,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }
    smc, rejuvenated = stored["diagnostic_runs"]
    assert smc["result"]["maximum_rhat"] < 1.2
    assert smc["result"]["maximum_standardized_between_replicate_median_spread"] > 0.25
    assert rejuvenated["result"]["maximum_rhat"] > 1.2
    assert rejuvenated["result"]["parameters_above_rhat_threshold"] == 17
    assert rejuvenated["result"]["maximum_standardized_source_to_rejuvenated_median_shift"] < 0.25


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["unchanged_completion_thresholds"].__setitem__("maximum_rhat", 1.5),
            "threshold",
        ),
        (
            lambda value: value["sample_seal"].__setitem__("target_rows_opened", 1),
            "sample seal",
        ),
        (
            lambda value: value["diagnostic_runs"][1]["result"].__setitem__("converged", True),
            "promoted",
        ),
        (
            lambda value: value["adjudication"].__setitem__("CP5_7_through_CP5_10_complete", True),
            "adjudication",
        ),
    ],
)
def test_threshold_access_result_and_completion_mutations_fail_closed(
    mutation: object, match: str
) -> None:
    config = copy.deepcopy(audit.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(audit.GravityClusterNuisanceIdentifiabilityError, match=match):
        audit.validate_config(config, ROOT)
