from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_audit as audit

ROOT = Path(__file__).resolve().parents[1]


def test_stored_quotient_rebuilds_without_completing_cp5() -> None:
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
    assert stored["claims"]["maximum_observable_nuisance_dimension"] == 10
    assert stored["claims"]["exact_null_dimensions"] == 7
    assert stored["claims"]["primitive_labels_separately_identified"] is False
    assert stored["claims"]["composite_posterior_converged"] is False
    assert stored["claims"]["CP5_7_through_CP5_10_complete"] is False


def test_exact_pushforward_and_failed_posterior_boundary_are_explicit() -> None:
    receipt = audit.build_receipt(ROOT)
    assert receipt["induced_prior_rule"]["primitive_priors_changed"] is False
    assert receipt["induced_prior_rule"]["stellar_clip_mixture_retained"] is True
    assert [row["coordinate"] for row in receipt["exact_composite_coordinates"]] == list(
        audit.COMPOSITES
    )
    posterior = receipt["observed_results"]["composite_posterior"]
    assert posterior["maximum_rhat"] > 1.2
    assert posterior["minimum_effective_samples"] > 50
    assert posterior["maximum_standardized_between_replicate_median_spread"] > 0.25
    assert posterior["coordinates_above_rhat_threshold"] == 10
    assert receipt["counts"] == {
        "primitive_parameters": 17,
        "exact_composite_coordinates": 10,
        "exact_null_dimensions": 7,
        "rank_anchors": 16,
        "rank_forward_evaluations": 544,
        "frozen_invariance_cases": 88,
        "source_composite_posterior_draws": 131072,
        "target_rows_opened": 0,
        "paid_model_calls": 0,
    }


def test_rank_and_symmetry_replay_passes() -> None:
    result = audit.replay(ROOT)
    assert result["rank"]["ranks"] == [10] * 16
    assert result["rank"]["maximum_first_null_relative_singular_value"] < 1e-8
    assert result["rank"]["minimum_tenth_relative_singular_value"] > 0.001
    assert result["forward_evaluations"] == 633
    assert all(
        row["maximum_absolute_log_prediction_difference"] < 1e-10
        for row in result["forward_invariance"].values()
    )


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["sample_seal"].__setitem__("target_rows_opened", 1),
            "sample seal",
        ),
        (
            lambda value: value["exact_null_structure"].__setitem__("total_null_dimensions", 6),
            "null dimension",
        ),
        (
            lambda value: value["induced_prior_rule"].__setitem__("primitive_priors_changed", True),
            "prior rule",
        ),
        (
            lambda value: value["observed_results"]["composite_posterior"].__setitem__(
                "converged", True
            ),
            "posterior boundary",
        ),
        (
            lambda value: value["adjudication"].__setitem__("CP5_7_through_CP5_10_complete", True),
            "adjudication",
        ),
    ],
)
def test_seal_null_prior_posterior_and_completion_mutations_fail_closed(
    mutation: object, match: str
) -> None:
    config = copy.deepcopy(audit.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(audit.GravityClusterNuisanceQuotientError, match=match):
        audit.validate_config(config, ROOT)
