from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v2 as v2
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc_v3 as v3
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[1]
CONFIG_SHA256 = "781278eab67272eaec76cece10c3789b22650bec025dacc0296399dc2868cbd3"


def test_v3_preregistration_binds_exact_prior_v1_v2_and_quotient() -> None:
    config = v3.load_config(ROOT / v3.CONFIG_PATH, CONFIG_SHA256)
    assert config["exact_primitive_priors"] == sampler.PRIMITIVE_PRIORS
    assert len(config["exact_primitive_priors"]) == 17
    assert set(config["frozen_evidence_bindings"]) >= {
        "canonical_sampler_source",
        "quotient_audit_receipt",
        "uncertainty_config",
        "v1_result",
        "v2_result",
        "v2_receipt",
        "v2_source",
    }


def test_v3_is_structural_not_more_sweeps_or_weaker_gates() -> None:
    assert v3.GATES == v2.GATES
    for key in (
        "adaptation_sweeps",
        "fixed_kernel_settling_sweeps",
        "retained_sweeps",
        "thin",
        "retained_snapshots_per_particle_chain",
    ):
        assert v3.CANDIDATE_INFERENCE[key] == v2.CANDIDATE_INFERENCE[key]
    change = v3.CANDIDATE_INFERENCE["structural_change_from_v2"]
    assert change["v2_bounded_correlated_random_walk_removed"]
    assert change["gaussianized_prior_reversible_transport_added"]
    assert len(v3.TRANSPORT_BLOCKS) == 6


def test_all_primitives_are_covered_and_exact_pushforward_is_unchanged() -> None:
    covered = sorted(
        {
            int(index)
            for block in v3.TRANSPORT_BLOCKS[:-1]
            for index in block["primitive_indices"]
        }
    )
    assert covered == list(range(17))
    assert v3.TRANSPORT_BLOCKS[-1]["primitive_indices"] == list(range(17))
    assert v3.sampler.composite_values is sampler.composite_values
    assert v3.sampler.PRIMITIVE_PRIORS == sampler.PRIMITIVE_PRIORS


def test_blockwise_detailed_balance_and_posterior_flux_control() -> None:
    control = v3.detailed_balance_control(uncertainty.load_config(ROOT))
    assert control["passed"]
    assert control["maximum_prior_reversibility_log_residual"] <= 5e-11
    assert control["maximum_metropolis_posterior_log_flux_residual"] <= 5e-11
    assert {row["block_id"] for row in control["blocks"]} == {
        row["block_id"] for row in v3.TRANSPORT_BLOCKS
    }


def test_exact_prior_pushforward_and_stellar_atoms_are_invariant() -> None:
    control = v3.prior_invariance_control(uncertainty.load_config(ROOT))
    assert control["passed"]
    assert control["maximum_uniform_marginal_ks"] <= 0.012
    assert control["maximum_composite_two_sample_ks"] <= 0.015
    assert control["maximum_stellar_clip_atom_frequency_difference"] <= 0.005
    assert {row["atom"] for row in control["stellar_clip_atoms"]} == {
        "lower_clip_0p4",
        "upper_clip_2p5",
    }


def test_pcn_never_wraps_or_reflects_out_of_bounds() -> None:
    prior = np.random.default_rng(1).random((2048, 17))
    proposal = v3.pcn_block_proposal(
        prior,
        list(range(17)),
        0.95,
        np.random.default_rng(2),
    )
    assert np.all(proposal > 0.0)
    assert np.all(proposal < 1.0)


def test_v2_independent_reference_and_data_boundary_remain_unchanged() -> None:
    assert v3.REFERENCE == v2.REFERENCE
    assert v3.SCENARIOS == v2.SCENARIOS
    assert v3.DATA_BOUNDARY == {
        "synthetic_data_only": True,
        "real_development_rows_loaded": 0,
        "real_holdout_rows_loaded": 0,
        "real_confirmation_rows_loaded": 0,
        "real_independent_rows_loaded": 0,
        "network_calls": 0,
        "paid_model_calls": 0,
        "candidate_production_runs": 0,
    }
    candidate_source = inspect.getsource(v3.run_candidate_fit)
    assert "active_transition" not in candidate_source
    assert "bounded_correlated_active_proposal" not in candidate_source


def test_config_is_json_and_has_no_work_binding() -> None:
    raw = (ROOT / v3.CONFIG_PATH).read_text(encoding="utf-8")
    config = json.loads(raw)
    assert config["call_accounting"] == v3.maximum_call_accounting()
    assert '"work/' not in raw


def _result_summary() -> dict[str, object]:
    with np.load(ROOT / v3.RESULT_PATH, allow_pickle=False) as archive:
        return json.loads(str(archive["summary"].item()))


def test_single_bounded_result_passes_unchanged_scientific_gates() -> None:
    summary = _result_summary()
    assert summary["decision"] == (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V3_PASSED_NOT_PHYSICS_OR_PRODUCTION"
    )
    assert summary["passed"] is True
    assert summary["kernel_controls"]["passed"] is True
    scenarios = {row["scenario_id"]: row for row in summary["scenario_results"]}
    moderate = scenarios["moderate_correlated_quotient_observation"]
    weak = scenarios["weak_diagonal_quotient_observation"]
    assert moderate["maximum_fit_rhat"] == 1.0071487891371973
    assert moderate["minimum_fit_bulk_ess"] == 2760.235057207742
    assert moderate["minimum_fit_tail_ess"] == 3424.474939363876
    assert weak["maximum_fit_rhat"] == 1.0048844096310794
    assert weak["minimum_fit_bulk_ess"] == 3743.2825872853796
    assert weak["minimum_fit_tail_ess"] == 3749.3951856692984
    assert all(row["passed"] for row in scenarios.values())


def test_v3_reference_is_exactly_the_bound_v2_paired_reference() -> None:
    summary = _result_summary()
    v2_receipt = json.loads(
        (ROOT / "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2.json").read_text(
            encoding="utf-8"
        )
    )
    for v3_row, v2_row in zip(
        summary["scenario_results"], v2_receipt["scenario_results"], strict=True
    ):
        assert v3_row["reference_rank"] == v2_row["reference_rank"]
        assert v3_row["reference_coverage"] == v2_row["reference_coverage"]
        assert v3_row["reference_minimum_stage_conditional_ess_fraction"] == v2_row[
            "reference_minimum_stage_conditional_ess_fraction"
        ]
        assert v3_row["reference_maximum_standardized_replicate_mean_difference"] == (
            v2_row["reference_maximum_standardized_replicate_mean_difference"]
        )


def test_exact_call_accounting_and_zero_real_access() -> None:
    summary = _result_summary()
    assert summary["call_accounting"] == {
        "actual_candidate_synthetic_likelihood_evaluations": 14944256,
        "actual_independent_reference_synthetic_likelihood_evaluations": 9952518,
        "actual_total_synthetic_likelihood_evaluations": 24896774,
        "frozen_maximum_total_synthetic_likelihood_evaluations": 50726912,
        "real_forward_model_evaluations": 0,
    }
    assert all(
        row["transport_out_of_bounds_self_loops"] == 0
        and row["initial_likelihoods_recomputed_fresh"] == 32
        for row in summary["candidate_fit_summaries"]
    )
    assert summary["data_boundary"] == v3.DATA_BOUNDARY
    assert summary["claim_boundary"] == v3.CLAIM_BOUNDARY


def test_receipt_is_hash_bound_and_claim_limited() -> None:
    result = v3.check(
        ROOT / v3.CONFIG_PATH,
        CONFIG_SHA256,
        ROOT / v3.RECEIPT_PATH,
    )
    assert result["valid"] is True
    assert result["passed"] is True
    assert result["synthetic_likelihood_evaluations"] == 24896774
    assert result["real_forward_model_evaluations"] == 0
    assert result["candidate_production_runs"] == 0
    assert result["scientific_claim_allowed"] is False
