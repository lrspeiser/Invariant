from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as sbc
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / sbc.CONFIG_PATH
RESULT = ROOT / sbc.RESULT_PATH
RECEIPT = ROOT / sbc.RECEIPT_PATH


def config_hash() -> str:
    return sbc.file_sha256(CONFIG)


def result_summary() -> dict[str, object]:
    archive = np.load(RESULT, allow_pickle=False)
    return json.loads(str(archive["summary"].item()))


def test_config_freezes_exact_prior_scenarios_budget_and_chronology() -> None:
    config = sbc.load_config(CONFIG, config_hash())
    prior = uncertainty.load_config(ROOT)["continuous_priors"]
    assert config["exact_primitive_priors"] == prior
    assert len(prior) == 17
    assert config["scenarios"] == sbc.SCENARIOS
    assert config["gates"] == sbc.GATES
    assert config["seed_lineage"] == sbc.SEED_LINEAGE
    assert config["call_accounting"] == {
        "simulations": 48,
        "maximum_mcmc_synthetic_likelihood_evaluations": 247296,
        "oracle_synthetic_likelihood_evaluations": 786432,
        "maximum_total_synthetic_likelihood_evaluations": 1033728,
        "real_forward_model_evaluations": 0,
    }
    assert config["chronology"] == {
        "config_and_gates_frozen_before_first_result": True,
        "result_written_before_receipt": True,
        "receipt_requires_hash_bound_result": True,
        "failed_result_retained": True,
        "post_result_threshold_changes_forbidden": True,
    }
    with pytest.raises(RuntimeError, match="config hash"):
        sbc.load_config(CONFIG, "0" * 64)


def test_stored_result_is_synthetic_only_failed_and_retained() -> None:
    archive = np.load(RESULT, allow_pickle=False)
    assert set(archive.files) == {
        "truth_units",
        "scenario_indices",
        "candidate_normalized_ranks",
        "oracle_weighted_cdf_ranks",
        "candidate_coverage",
        "oracle_coverage",
        "candidate_tie_counts",
        "oracle_tie_mass",
        "importance_effective_samples",
        "summary",
    }
    assert archive["truth_units"].shape == (48, 17)
    assert archive["candidate_normalized_ranks"].shape == (48, 10)
    assert archive["candidate_coverage"].shape == (48, 10, 3)
    summary = result_summary()
    assert summary["passed"] is False
    assert summary["decision"] == (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_FAILED_RESULT_RETAINED"
    )
    assert summary["data_boundary"] == sbc.DATA_BOUNDARY
    assert summary["claim_boundary"] == sbc.CLAIM_BOUNDARY
    assert summary["call_accounting"] == {
        "actual_mcmc_synthetic_likelihood_evaluations": 97826,
        "oracle_synthetic_likelihood_evaluations": 786432,
        "actual_total_synthetic_likelihood_evaluations": 884258,
        "frozen_maximum_total_synthetic_likelihood_evaluations": 1033728,
        "real_forward_model_evaluations": 0,
    }


def test_failure_reasons_are_explicit_and_thresholds_were_not_relaxed() -> None:
    summary = result_summary()
    moderate, weak = summary["scenario_summaries"]
    assert moderate["passed"] is False
    assert weak["passed"] is False
    assert moderate["maximum_fit_rhat"] == pytest.approx(4.730527265227861)
    assert weak["maximum_fit_rhat"] == pytest.approx(3.565379684939892)
    assert moderate["minimum_importance_effective_samples"] == pytest.approx(
        4.321526188318886
    )
    assert weak["minimum_importance_effective_samples"] == pytest.approx(
        399.6535492044037
    )
    assert moderate["maximum_fit_rhat"] > sbc.GATES[
        "maximum_rank_normalized_split_rhat"
    ]
    assert moderate["minimum_importance_effective_samples"] < sbc.GATES[
        "minimum_importance_effective_samples"
    ]
    assert sbc.GATES["threshold_relaxation_after_result"] is False
    assert sbc.GATES["failed_result_retained"] is True


def test_rank_signal_was_reasonable_but_does_not_override_failures() -> None:
    for scenario in result_summary()["scenario_summaries"]:
        assert scenario["candidate_rank"]["maximum_absolute_mean_rank_z"] < 2.4
        assert scenario[
            "maximum_absolute_candidate_oracle_mean_rank_difference"
        ] < 0.08
        assert scenario["passed"] is False


def test_orbit_control_and_randomized_clipping_ties() -> None:
    orbit = result_summary()["orbit_invariance_control"]
    assert orbit["passed"] is True
    assert orbit["accepted_cases"] == {"stellar": 8, "geometry": 8, "coupled": 8}
    assert orbit["maximum_absolute_composite_difference"] < 1e-12
    draws = np.asarray([[0.4], [0.4], [0.5], [0.6]])
    ranks_a, ties_a = sbc.randomized_integer_ranks(draws, np.asarray([0.4]), 91)
    ranks_b, ties_b = sbc.randomized_integer_ranks(draws, np.asarray([0.4]), 91)
    np.testing.assert_array_equal(ranks_a, ranks_b)
    np.testing.assert_array_equal(ties_a, ties_b)
    assert ties_a.tolist() == [2]
    assert 0.0 < ranks_a[0] < 1.0


def test_synthetic_seed_lineage_and_sobol_starts_are_deterministic() -> None:
    first = sbc.sobol_start_populations(7)
    replay = sbc.sobol_start_populations(7)
    other = sbc.sobol_start_populations(8)
    assert first.shape == (2, 16, 17)
    np.testing.assert_array_equal(first, replay)
    assert not np.array_equal(first, other)
    assert not np.array_equal(first[0], first[1])
    assert np.all((first > 0.0) & (first < 1.0))


def test_weighted_reference_utilities_are_deterministic() -> None:
    values = np.asarray([0.0, 1.0, 2.0, 3.0])
    weights = np.asarray([0.1, 0.2, 0.3, 0.4])
    quantiles = sbc.weighted_quantile(values, weights, np.asarray([0.25, 0.75]))
    np.testing.assert_allclose(quantiles, [0.75, 2.375])
    matrix = values[:, None]
    rank, tie_mass = sbc.weighted_cdf_ranks(matrix, weights, np.asarray([2.0]), 17)
    assert 0.3 <= rank[0] <= 0.6
    assert tie_mass[0] == pytest.approx(0.3)


def test_receipt_is_hash_bound_valid_and_claim_safe() -> None:
    checked = sbc.check(CONFIG, config_hash(), RECEIPT)
    receipt = json.loads(RECEIPT.read_text(encoding="utf-8"))
    assert checked == {
        "valid": True,
        "passed": False,
        "config_sha256": config_hash(),
        "receipt_sha256": sbc.file_sha256(RECEIPT),
        "synthetic_likelihood_evaluations": 884258,
        "real_forward_model_evaluations": 0,
        "candidate_production_runs": 0,
        "scientific_claim_allowed": False,
    }
    assert receipt["status"] == "bounded_synthetic_sbc_failed_result_retained"
    assert receipt["evidence"]["canonical_sampler"] == {
        "path": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_sampler.py",
        "file_sha256": sbc.file_sha256(Path(sampler.__file__)),
    }
    assert receipt["claim_boundary"]["CP5_7_through_CP5_10_complete"] is False
    assert receipt["claim_boundary"]["scientific_claim_allowed"] is False


def test_new_canonical_package_contains_no_development_workspace_binding() -> None:
    for path in (CONFIG, RECEIPT, ROOT / sbc.__file__):
        text = path.read_text(encoding="utf-8").replace("\\", "/").lower()
        assert "work/" not in text
    assert not (ROOT / sbc.ARTIFACT_DIR / "authorized.json").exists()
