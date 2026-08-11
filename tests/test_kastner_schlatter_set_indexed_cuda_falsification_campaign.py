from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.kastner_schlatter_set_indexed_cuda_falsification_campaign import (
    _content_sha,
    exact_common_shock_sentinel,
    validate_campaign,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/kastner_schlatter_set_indexed_cuda_falsification_campaign.json"
ARTIFACT = ROOT / "runs/engine/kastner-schlatter-set-indexed-cuda-falsification-campaign.json"


def _artifact() -> dict:
    return json.loads(ARTIFACT.read_text(encoding="utf-8"))


def test_exact_marginal_preserving_common_shock_sentinel() -> None:
    sentinel = exact_common_shock_sentinel()
    assert sentinel["marginal_mean"] == "2"
    assert sentinel["marginal_variance"] == "2"
    assert sentinel["marginal_fano"] == "1"
    assert sentinel["within_group_cross_covariance"] == "1/2"
    assert sentinel["two_cell_union_fano"] == "5/4"
    assert sentinel["alternative_to_null_pgf_ratio"] == "exp(1/8)"


def test_artifact_validates_and_counts_are_honest() -> None:
    result = _artifact()
    validate_campaign(result, CONFIG)
    assert result["counts"]["scenario_cells"] == 48
    assert result["counts"]["gpu_generated_unique_count_values"] == 1887436800
    assert result["counts"]["joint_pgf_terms_evaluated"] == 8053063680
    assert result["counts"]["projection_multiply_adds"] == 322122547200
    assert result["counts"]["prior_poisson_cox_scenario_cells"] == 144
    assert result["evaluator_entry_point"]["writes_artifact"] is False


def test_alternative_preserves_marginals_but_has_joint_signal() -> None:
    result = _artifact()
    assert all(
        row["analytic"]["marginal_fano_null_and_alternative"] == "1"
        and row["analytic"]["within_group_cross_covariance"] != "0"
        for row in result["scenario_results"]
    )
    assert result["gpu_cpu_crosscheck"]["all_heldout_decisions_byte_equal"] is True


@pytest.mark.parametrize(
    "key",
    [
        "observations_opened", "scientific_test_pass", "readiness_advanced", "paper_pass",
        "qed_pass", "theory_pass", "ontology_pass", "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs",
    ],
)
def test_claim_and_data_tamper_fails(key: str) -> None:
    tampered = copy.deepcopy(_artifact())
    tampered[key] = True
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError, match="claim or data seal"):
        validate_campaign(tampered, CONFIG)


def test_hash_and_count_tamper_fail() -> None:
    tampered = copy.deepcopy(_artifact())
    tampered["counts"]["scenario_cells"] = 1
    with pytest.raises(ValueError, match="content hash"):
        validate_campaign(tampered, CONFIG)
    tampered["content_sha256"] = _content_sha(tampered)
    with pytest.raises(ValueError, match="scenario or forbidden count"):
        validate_campaign(tampered, CONFIG)


def test_all_scientific_and_observation_seals_closed() -> None:
    result = _artifact()
    assert result["synthetic_only"] is True
    for key in (
        "observations_opened", "scientific_test_pass", "observational_test_pass", "readiness_advanced",
        "paper_pass", "qed_pass", "theory_pass", "ontology_pass", "dark_matter_or_halo_inputs",
        "redshift_or_cosmology_inputs", "paid_llm_calls",
    ):
        assert result[key] is False
