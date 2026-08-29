from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sampler as sampler
from sigma_theory_compiler import gravity_cluster_nuisance_quotient_sbc as v1
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sbc_v2.json")
ARTIFACT_DIR = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2"
)
RESULT_PATH = ARTIFACT_DIR / "bounded-synthetic-sbc-v2.npz"
RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v2.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-config-2.0"
RESULT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-result-2.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-receipt-2.0"

SCENARIOS = [
    {
        "scenario_id": "moderate_correlated_quotient_observation",
        "simulations": 32,
        "normalized_noise_sigma": 0.45,
        "ar1_correlation": 0.40,
        "relationship_to_v1": "unchanged_observation_regime_more_simulations",
    },
    {
        "scenario_id": "weak_diagonal_quotient_observation",
        "simulations": 32,
        "normalized_noise_sigma": 0.90,
        "ar1_correlation": 0.0,
        "relationship_to_v1": "unchanged_observation_regime_more_simulations",
    },
]
CANDIDATE_INFERENCE = {
    "replicates": 2,
    "particles_per_replicate": 16,
    "adaptation_sweeps": 64,
    "fixed_kernel_settling_sweeps": 128,
    "retained_sweeps": 1024,
    "thin": 8,
    "retained_snapshots_per_particle_chain": 128,
    "covariance_refresh_during_adaptation": 8,
    "initial_active_scale": 0.35,
    "active_scale_bounds": [0.02, 4.0],
    "target_acceptance": 0.234,
    "adaptation_gain": 0.5,
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "active_kernel": (
        "canonical_symmetric_correlated_gaussian_with_whole_proposal_"
        "out_of_bounds_rejection"
    ),
    "orbit_moves": list(sampler.ORBIT_NAMES),
    "change_from_v1": {
        "adaptation_sweeps_multiplier": 4,
        "settling_sweeps_multiplier": 8,
        "retained_sweeps_multiplier": 8,
        "retained_draws_per_chain_multiplier": 4,
        "reason": "V1 maximum per-fit Rhat exceeded 3.5 in both scenarios",
    },
}
REFERENCE = {
    "algorithm_id": "independent_adaptive_tempering_coordinate_smc",
    "algorithmically_independent_of_candidate_kernel": True,
    "canonical_active_transition_called": False,
    "canonical_orbit_move_called": False,
    "replicates": 2,
    "particles_per_replicate": 1024,
    "adaptive_conditional_ess_fraction": 0.70,
    "maximum_tempering_stages": 16,
    "coordinate_mh_sweeps_per_stage": 17,
    "initial_coordinate_step": 0.15,
    "coordinate_step_bounds": [0.01, 0.40],
    "target_acceptance": 0.44,
    "adaptation_gain": 0.35,
    "systematic_resampling_each_stage": True,
    "terminal_beta": 1.0,
    "change_from_v1": {
        "finite_prior_importance_removed": True,
        "adaptive_tempering_added": True,
        "independent_replicates_added": 2,
        "reason": "V1 finite-Sobol importance ESS fell to 4.32 and 399.65",
    },
}
RANK_PROTOCOL = {
    "coordinates": list(sampler.COMPOSITES),
    "candidate_posterior_draws_per_simulation": 4096,
    "reference_particles_per_simulation": 2048,
    "randomized_tie_ranks_for_stellar_clipping_atoms": True,
    "coverage_levels": [0.5, 0.8, 0.9],
    "rank_histogram_bins": 8,
    "truths_are_exact_independent_17_primitive_prior_draws": True,
}


def finite_simulation_coverage_tolerances(
    simulations: int, z: float = 3.5
) -> dict[str, float]:
    return {
        str(level): z * math.sqrt(level * (1.0 - level) / simulations)
        + 1.0 / simulations
        for level in RANK_PROTOCOL["coverage_levels"]
    }


GATES = {
    "finite_simulation_familywise_z": 3.5,
    "finite_simulation_continuity_allowance": "one_over_simulations_per_scenario",
    "coverage_tolerances_for_32_simulations": finite_simulation_coverage_tolerances(
        32
    ),
    "maximum_absolute_candidate_mean_rank_z": 4.0,
    "maximum_absolute_reference_mean_rank_z": 4.0,
    "maximum_absolute_candidate_reference_mean_rank_difference": 0.20,
    "minimum_fraction_fits_all_coordinates_diagnostic_valid": 0.90,
    "maximum_rank_normalized_split_rhat": 1.20,
    "minimum_bulk_effective_samples_per_valid_coordinate": 50.0,
    "minimum_tail_effective_samples_per_valid_coordinate": 50.0,
    "reference_minimum_stage_conditional_ess_fraction": 0.69,
    "reference_maximum_standardized_replicate_mean_difference": 0.35,
    "reference_minimum_unique_initial_ancestor_fraction": 0.05,
    "reference_all_replicates_must_reach_beta_one": True,
    "maximum_orbit_composite_difference": 1e-12,
    "all_scenarios_must_pass": True,
    "v1_thresholds_retroactively_changed": False,
    "threshold_relaxation_after_v2_result": False,
    "failed_result_retained": True,
}
SEED_LINEAGE = {
    "truth_base": 810000,
    "noise_base": 820000,
    "candidate_sobol_start_base": 830000,
    "candidate_transition_base": 840000,
    "candidate_rank_tie_base": 850000,
    "reference_base": 860000,
    "reference_rank_tie_base": 870000,
    "scenario_stride": 10000,
    "simulation_stride": 10,
    "reference_replicate_stride": 1,
    "no_seed_derived_from_v1_or_v2_observed_result": True,
}
DATA_BOUNDARY = {
    "synthetic_data_only": True,
    "real_development_rows_loaded": 0,
    "real_holdout_rows_loaded": 0,
    "real_confirmation_rows_loaded": 0,
    "real_independent_rows_loaded": 0,
    "network_calls": 0,
    "paid_model_calls": 0,
    "candidate_production_runs": 0,
}
CLAIM_BOUNDARY = {
    "calibrates_synthetic_quotient_inference_only": True,
    "calibrates_full_cluster_forward_model": False,
    "candidate_physics_supported": False,
    "candidate_production_completed": False,
    "CP5_7_through_CP5_10_complete": False,
    "scientific_claim_allowed": False,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def content_sha256(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def maximum_call_accounting() -> dict[str, int]:
    simulations = sum(int(row["simulations"]) for row in SCENARIOS)
    candidate_chains = int(CANDIDATE_INFERENCE["replicates"]) * int(
        CANDIDATE_INFERENCE["particles_per_replicate"]
    )
    candidate_sweeps = (
        int(CANDIDATE_INFERENCE["adaptation_sweeps"])
        + int(CANDIDATE_INFERENCE["fixed_kernel_settling_sweeps"])
        + int(CANDIDATE_INFERENCE["retained_sweeps"])
    )
    candidate = simulations * candidate_chains * (1 + candidate_sweeps)
    reference_per_replicate = int(REFERENCE["particles_per_replicate"]) * (
        1
        + int(REFERENCE["maximum_tempering_stages"])
        * int(REFERENCE["coordinate_mh_sweeps_per_stage"])
    )
    reference = simulations * int(REFERENCE["replicates"]) * reference_per_replicate
    return {
        "simulations": simulations,
        "maximum_candidate_synthetic_likelihood_evaluations": candidate,
        "maximum_reference_synthetic_likelihood_evaluations": reference,
        "maximum_total_synthetic_likelihood_evaluations": candidate + reference,
        "real_forward_model_evaluations": 0,
    }


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if file_sha256(target) != expected_sha256:
        raise RuntimeError("SBC V2 config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    expected_keys = {
        "schema_version",
        "status",
        "purpose",
        "implementation_source",
        "implementation_source_normalized_sha256",
        "v1_evidence_bindings",
        "canonical_sampler_binding",
        "uncertainty_config_binding",
        "exact_primitive_priors",
        "primitive_prior_semantics",
        "composite_scales",
        "scenarios",
        "candidate_inference",
        "independent_reference",
        "rank_protocol",
        "gates",
        "seed_lineage",
        "data_boundary",
        "claim_boundary",
        "call_accounting",
        "chronology",
        "output_paths",
    }
    if not isinstance(config, dict) or set(config) != expected_keys:
        raise RuntimeError("SBC V2 config keys changed")
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "separately_preregistered_before_bounded_v2_run"
    ):
        raise RuntimeError("SBC V2 config identity changed")
    implementation = confined(ROOT / config["implementation_source"])
    if implementation != Path(__file__).resolve() or normalized_sha256(
        implementation
    ) != config["implementation_source_normalized_sha256"]:
        raise RuntimeError("SBC V2 implementation changed after preregistration")
    for label in (
        "canonical_sampler_binding",
        "uncertainty_config_binding",
    ):
        row = config[label]
        bound = confined(ROOT / row["path"])
        if file_sha256(bound) != row["file_sha256"]:
            raise RuntimeError(f"SBC V2 binding changed: {label}")
    for label, row in config["v1_evidence_bindings"].items():
        bound = confined(ROOT / row["path"])
        if file_sha256(bound) != row["file_sha256"]:
            raise RuntimeError(f"SBC V1 evidence changed: {label}")
    uncertainty_config = json.loads(
        (ROOT / config["uncertainty_config_binding"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    if (
        config["exact_primitive_priors"] != uncertainty_config["continuous_priors"]
        or len(config["exact_primitive_priors"]) != 17
    ):
        raise RuntimeError("SBC V2 exact prior changed")
    expected = {
        "primitive_prior_semantics": (
            "17_independent_uniform_primitives_with_clipped_six_factor_"
            "stellar_pushforward_clip_0.4_2.5"
        ),
        "composite_scales": v1.COMPOSITE_SCALES.tolist(),
        "scenarios": SCENARIOS,
        "candidate_inference": CANDIDATE_INFERENCE,
        "independent_reference": REFERENCE,
        "rank_protocol": RANK_PROTOCOL,
        "gates": GATES,
        "seed_lineage": SEED_LINEAGE,
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "call_accounting": maximum_call_accounting(),
        "chronology": {
            "v1_failure_diagnosed_before_v2_design": True,
            "v1_files_unchanged": True,
            "v2_source_config_scenarios_gates_and_budget_frozen_before_first_run": True,
            "v2_result_written_before_receipt": True,
            "failed_v2_result_retained": True,
            "post_result_threshold_changes_forbidden": True,
        },
        "output_paths": {
            "result": RESULT_PATH.as_posix(),
            "receipt": RECEIPT_PATH.as_posix(),
        },
    }
    for name, value in expected.items():
        if config[name] != value:
            raise RuntimeError(f"frozen SBC V2 object changed: {name}")
    config["_config_sha256"] = expected_sha256
    return config


class CandidateLikelihood:
    def __init__(
        self,
        observation: np.ndarray,
        inverse_covariance: np.ndarray,
        prior_config: dict[str, Any],
    ) -> None:
        self.observation = observation
        self.inverse_covariance = inverse_covariance
        self.prior_config = prior_config
        self.calls = 0

    def __call__(self, unit: np.ndarray) -> float:
        self.calls += 1
        return float(
            v1.batch_log_likelihood(
                unit[None, :],
                self.observation,
                self.inverse_covariance,
                self.prior_config,
            )[0]
        )


def candidate_sobol_starts(global_index: int) -> np.ndarray:
    populations = []
    for replicate in range(int(CANDIDATE_INFERENCE["replicates"])):
        seed = (
            int(SEED_LINEAGE["candidate_sobol_start_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        populations.append(qmc.Sobol(d=17, scramble=True, seed=seed).random_base2(m=4))
    return np.stack(populations)


def run_candidate_fit(
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
    global_index: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    particles_by_replicate = candidate_sobol_starts(global_index)
    replicates, particles_count, _ = particles_by_replicate.shape
    evaluator = CandidateLikelihood(observation, inverse_covariance, prior_config)
    likelihood_by_replicate = np.empty((replicates, particles_count))
    for replicate in range(replicates):
        for particle in range(particles_count):
            likelihood_by_replicate[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    retained = int(CANDIDATE_INFERENCE["retained_snapshots_per_particle_chain"])
    traces = np.empty(
        (replicates, particles_count, retained, len(sampler.COMPOSITES))
    )
    active_attempted = 0
    active_evaluated = 0
    active_accepted = 0
    out_of_bounds = 0
    orbit_attempted = 0
    orbit_accepted = 0
    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        likelihood = likelihood_by_replicate[replicate].copy()
        rng = np.random.default_rng(
            int(SEED_LINEAGE["candidate_transition_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        scale = float(CANDIDATE_INFERENCE["initial_active_scale"])
        square_root = sampler.covariance_square_root(
            particles[:, sampler.ACTIVE_INDICES]
        )
        retained_index = 0
        phases = (
            ("adaptation", int(CANDIDATE_INFERENCE["adaptation_sweeps"])),
            (
                "settling",
                int(CANDIDATE_INFERENCE["fixed_kernel_settling_sweeps"]),
            ),
            ("retained", int(CANDIDATE_INFERENCE["retained_sweeps"])),
        )
        for phase, sweeps in phases:
            for sweep in range(sweeps):
                if (
                    phase == "adaptation"
                    and sweep
                    % int(CANDIDATE_INFERENCE["covariance_refresh_during_adaptation"])
                    == 0
                ):
                    square_root = sampler.covariance_square_root(
                        particles[:, sampler.ACTIVE_INDICES]
                    )
                orbit = sampler.orbit_sweep(
                    particles, rng, prior_config, CANDIDATE_INFERENCE
                )
                orbit_attempted += sum(
                    int(orbit[f"{move}_attempted"])
                    for move in sampler.ORBIT_NAMES
                )
                orbit_accepted += sum(
                    int(orbit[f"{move}_accepted"])
                    for move in sampler.ORBIT_NAMES
                )
                transition = sampler.active_transition(
                    particles,
                    likelihood,
                    evaluator,  # type: ignore[arg-type]
                    rng,
                    square_root,
                    scale,
                )
                active_attempted += int(transition["attempted"])
                active_evaluated += int(transition["evaluated"])
                active_accepted += int(transition["accepted"])
                out_of_bounds += int(transition["out_of_bounds_rejected"])
                if phase == "adaptation":
                    acceptance = transition["accepted"] / transition["attempted"]
                    scale *= math.exp(
                        float(CANDIDATE_INFERENCE["adaptation_gain"])
                        * (
                            acceptance
                            - float(CANDIDATE_INFERENCE["target_acceptance"])
                        )
                    )
                    scale = float(
                        np.clip(
                            scale,
                            float(CANDIDATE_INFERENCE["active_scale_bounds"][0]),
                            float(CANDIDATE_INFERENCE["active_scale_bounds"][1]),
                        )
                    )
                elif phase == "retained" and (sweep + 1) % int(
                    CANDIDATE_INFERENCE["thin"]
                ) == 0:
                    traces[replicate, :, retained_index, :] = (
                        sampler.composite_values(particles, prior_config)
                    )
                    retained_index += 1
            if phase == "adaptation":
                square_root = sampler.covariance_square_root(
                    particles[:, sampler.ACTIVE_INDICES]
                )
        if retained_index != retained:
            raise RuntimeError("SBC V2 candidate retained snapshot accounting changed")
    chains = traces.reshape(
        replicates * particles_count, retained, len(sampler.COMPOSITES)
    )
    diagnostics = sampler.rank_split_diagnostics(chains)
    draws = chains.reshape(-1, len(sampler.COMPOSITES))
    valid = diagnostics["valid"]
    finite_rhat = diagnostics["rhat"][valid]
    bulk = diagnostics["bulk_ess"][valid]
    tail = diagnostics["tail_ess"][valid]
    return (
        draws,
        {
            "all_coordinates_diagnostic_valid": bool(np.all(valid)),
            "valid_coordinate_count": int(np.count_nonzero(valid)),
            "maximum_rhat": (
                float(np.max(finite_rhat)) if len(finite_rhat) else None
            ),
            "minimum_bulk_ess": float(np.min(bulk)) if len(bulk) else 0.0,
            "minimum_tail_ess": float(np.min(tail)) if len(tail) else 0.0,
            "active_attempted": active_attempted,
            "active_evaluated": active_evaluated,
            "active_accepted": active_accepted,
            "active_out_of_bounds_self_loops": out_of_bounds,
            "orbit_attempted": orbit_attempted,
            "orbit_accepted": orbit_accepted,
            "initial_likelihoods_recomputed_fresh": replicates * particles_count,
        },
        evaluator.calls,
    )


def effective_sample_size(weights: np.ndarray) -> float:
    normalized = weights / np.sum(weights)
    return float(1.0 / np.sum(normalized**2))


def next_beta(log_likelihood: np.ndarray, beta: float) -> tuple[float, float]:
    target = float(REFERENCE["adaptive_conditional_ess_fraction"]) * len(
        log_likelihood
    )

    def ess_at(proposed: float) -> float:
        log_weights = (proposed - beta) * log_likelihood
        weights = np.exp(log_weights - np.max(log_weights))
        return effective_sample_size(weights)

    if ess_at(1.0) >= target:
        return 1.0, ess_at(1.0)
    low = beta
    high = 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if ess_at(middle) < target:
            high = middle
        else:
            low = middle
    proposed = low
    if proposed <= beta + 1e-12:
        raise RuntimeError("independent SMC tempering schedule became stuck")
    return proposed, ess_at(proposed)


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    normalized = weights / np.sum(weights)
    cumulative = np.cumsum(normalized)
    positions = (float(rng.random()) + np.arange(len(weights))) / len(weights)
    return np.searchsorted(cumulative, positions, side="right")


def run_independent_smc_replicate(
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
    seed: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    rng = np.random.default_rng(seed)
    particle_count = int(REFERENCE["particles_per_replicate"])
    particles = rng.random((particle_count, 17))
    ancestors = np.arange(particle_count)
    log_likelihood = v1.batch_log_likelihood(
        particles, observation, inverse_covariance, prior_config
    )
    calls = particle_count
    beta = 0.0
    step = float(REFERENCE["initial_coordinate_step"])
    stage_rows = []
    for stage in range(int(REFERENCE["maximum_tempering_stages"])):
        proposed_beta, conditional_ess = next_beta(log_likelihood, beta)
        incremental_log_weights = (proposed_beta - beta) * log_likelihood
        weights = np.exp(incremental_log_weights - np.max(incremental_log_weights))
        indices = systematic_resample(weights, rng)
        particles = particles[indices].copy()
        log_likelihood = log_likelihood[indices].copy()
        ancestors = ancestors[indices]
        beta = proposed_beta
        accepted = 0
        attempted = 0
        out_of_bounds = 0
        for sweep in range(int(REFERENCE["coordinate_mh_sweeps_per_stage"])):
            coordinate = sweep % 17
            proposed_coordinate = particles[:, coordinate] + rng.normal(
                scale=step, size=particle_count
            )
            valid = (proposed_coordinate >= 0.0) & (proposed_coordinate <= 1.0)
            valid_indices = np.flatnonzero(valid)
            proposals = particles[valid_indices].copy()
            proposals[:, coordinate] = proposed_coordinate[valid_indices]
            proposed_likelihood = v1.batch_log_likelihood(
                proposals, observation, inverse_covariance, prior_config
            )
            calls += len(valid_indices)
            accept = np.log(
                np.maximum(rng.random(len(valid_indices)), np.finfo(float).tiny)
            ) < np.minimum(
                0.0,
                beta
                * (proposed_likelihood - log_likelihood[valid_indices]),
            )
            accepted_indices = valid_indices[accept]
            particles[accepted_indices] = proposals[accept]
            log_likelihood[accepted_indices] = proposed_likelihood[accept]
            accepted += int(np.count_nonzero(accept))
            attempted += particle_count
            out_of_bounds += particle_count - len(valid_indices)
        acceptance = accepted / attempted
        step *= math.exp(
            float(REFERENCE["adaptation_gain"])
            * (acceptance - float(REFERENCE["target_acceptance"]))
        )
        step = float(
            np.clip(
                step,
                float(REFERENCE["coordinate_step_bounds"][0]),
                float(REFERENCE["coordinate_step_bounds"][1]),
            )
        )
        stage_rows.append(
            {
                "stage": stage + 1,
                "beta": beta,
                "conditional_ess": conditional_ess,
                "conditional_ess_fraction": conditional_ess / particle_count,
                "coordinate_mh_acceptance": acceptance,
                "out_of_bounds_self_loops": out_of_bounds,
                "ending_coordinate_step": step,
                "unique_initial_ancestor_fraction": (
                    len(np.unique(ancestors)) / particle_count
                ),
            }
        )
        if beta >= 1.0 - 1e-14:
            beta = 1.0
            break
    if beta != 1.0:
        raise RuntimeError("independent SMC failed to reach beta one within frozen stages")
    values = sampler.composite_values(particles, prior_config)
    return (
        values,
        {
            "algorithm_id": REFERENCE["algorithm_id"],
            "terminal_beta": beta,
            "stages": len(stage_rows),
            "minimum_stage_conditional_ess_fraction": min(
                row["conditional_ess_fraction"] for row in stage_rows
            ),
            "final_unique_initial_ancestor_fraction": stage_rows[-1][
                "unique_initial_ancestor_fraction"
            ],
            "stage_rows": stage_rows,
            "calls": calls,
        },
        calls,
    )


def independent_reference(
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
    seed_offset: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    replicate_values = []
    replicate_summaries = []
    calls = 0
    for replicate in range(int(REFERENCE["replicates"])):
        seed = (
            int(SEED_LINEAGE["reference_base"])
            + seed_offset
            + replicate * int(SEED_LINEAGE["reference_replicate_stride"])
        )
        values, summary, replicate_calls = run_independent_smc_replicate(
            observation, inverse_covariance, prior_config, seed
        )
        replicate_values.append(values)
        replicate_summaries.append({"replicate": replicate, **summary})
        calls += replicate_calls
    first, second = replicate_values
    pooled = np.concatenate(replicate_values)
    pooled_scale = np.std(pooled, axis=0, ddof=1)
    standardized_difference = np.divide(
        np.abs(np.mean(first, axis=0) - np.mean(second, axis=0)),
        pooled_scale,
        out=np.zeros_like(pooled_scale),
        where=pooled_scale > np.finfo(float).tiny,
    )
    return (
        pooled,
        {
            "replicates": replicate_summaries,
            "maximum_standardized_replicate_mean_difference": float(
                np.max(standardized_difference)
            ),
            "minimum_stage_conditional_ess_fraction": min(
                row["minimum_stage_conditional_ess_fraction"]
                for row in replicate_summaries
            ),
            "minimum_final_unique_initial_ancestor_fraction": min(
                row["final_unique_initial_ancestor_fraction"]
                for row in replicate_summaries
            ),
            "all_replicates_reached_beta_one": all(
                row["terminal_beta"] == 1.0 for row in replicate_summaries
            ),
            "calls": calls,
        },
        calls,
    )


def reference_rank(
    draws: np.ndarray, truth: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.full(len(draws), 1.0 / len(draws))
    return v1.weighted_cdf_ranks(draws, weights, truth, seed)


def reference_coverage(
    draws: np.ndarray, truth: np.ndarray, levels: list[float]
) -> np.ndarray:
    weights = np.full(len(draws), 1.0 / len(draws))
    return v1.coverage_rows(draws, truth, levels, weights)


def scenario_covariance(
    scenario: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    row = {
        "normalized_noise_sigma": scenario["normalized_noise_sigma"],
        "ar1_correlation": scenario["ar1_correlation"],
    }
    return v1.scenario_covariance(row)


def write_result(path: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            **arrays,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    sampler._write_then_publish_no_clobber(path, writer, suffix=".npz.tmp")


def summarize_rank(ranks: np.ndarray) -> dict[str, Any]:
    return v1.rank_summary(ranks)


def summarize_coverage(coverage: np.ndarray) -> dict[str, Any]:
    return v1.coverage_summary(coverage)


def run_bounded(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if confined(output) != ROOT / RESULT_PATH:
        raise RuntimeError("bounded SBC V2 output path changed")
    prior_config = uncertainty.load_config(ROOT)
    orbit_control = v1.orbit_invariance_control(prior_config)
    levels = list(map(float, RANK_PROTOCOL["coverage_levels"]))
    candidate_ranks = []
    reference_ranks = []
    candidate_coverage_rows = []
    reference_coverage_rows = []
    truth_units = []
    scenario_indices = []
    candidate_ties = []
    reference_tie_mass = []
    candidate_summaries = []
    reference_summaries = []
    candidate_calls = 0
    reference_calls = 0
    global_index = 0
    for scenario_index, scenario in enumerate(SCENARIOS):
        covariance, inverse_covariance = scenario_covariance(scenario)
        covariance_root = np.linalg.cholesky(covariance)
        for simulation_index in range(int(scenario["simulations"])):
            seed_offset = (
                scenario_index * int(SEED_LINEAGE["scenario_stride"])
                + simulation_index * int(SEED_LINEAGE["simulation_stride"])
            )
            truth_unit = np.random.default_rng(
                int(SEED_LINEAGE["truth_base"]) + seed_offset
            ).random(17)
            truth = sampler.composite_values(truth_unit, prior_config)
            truth_normalized = truth / v1.COMPOSITE_SCALES
            noise = np.random.default_rng(
                int(SEED_LINEAGE["noise_base"]) + seed_offset
            ).normal(size=len(sampler.COMPOSITES))
            observation = truth_normalized + covariance_root @ noise
            candidate_draws, candidate_summary, calls = run_candidate_fit(
                observation, inverse_covariance, prior_config, global_index
            )
            candidate_calls += calls
            candidate_rank_values, ties = v1.randomized_integer_ranks(
                candidate_draws,
                truth,
                int(SEED_LINEAGE["candidate_rank_tie_base"]) + seed_offset,
            )
            reference_draws, reference_summary, calls = independent_reference(
                observation, inverse_covariance, prior_config, seed_offset
            )
            reference_calls += calls
            reference_rank_values, tie_mass = reference_rank(
                reference_draws,
                truth,
                int(SEED_LINEAGE["reference_rank_tie_base"]) + seed_offset,
            )
            candidate_ranks.append(candidate_rank_values)
            reference_ranks.append(reference_rank_values)
            candidate_coverage_rows.append(
                v1.coverage_rows(candidate_draws, truth, levels)
            )
            reference_coverage_rows.append(
                reference_coverage(reference_draws, truth, levels)
            )
            truth_units.append(truth_unit)
            scenario_indices.append(scenario_index)
            candidate_ties.append(ties)
            reference_tie_mass.append(tie_mass)
            candidate_summaries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "simulation_index": simulation_index,
                    **candidate_summary,
                }
            )
            reference_summaries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "simulation_index": simulation_index,
                    **reference_summary,
                }
            )
            global_index += 1
    candidate_rank_array = np.asarray(candidate_ranks)
    reference_rank_array = np.asarray(reference_ranks)
    candidate_coverage_array = np.asarray(candidate_coverage_rows)
    reference_coverage_array = np.asarray(reference_coverage_rows)
    scenario_index_array = np.asarray(scenario_indices, dtype=int)
    scenario_results = []
    all_scenarios_pass = True
    tolerances = GATES["coverage_tolerances_for_32_simulations"]
    for scenario_index, scenario in enumerate(SCENARIOS):
        selected = scenario_index_array == scenario_index
        candidate_rank_summary = summarize_rank(candidate_rank_array[selected])
        reference_rank_summary = summarize_rank(reference_rank_array[selected])
        candidate_coverage_summary = summarize_coverage(
            candidate_coverage_array[selected]
        )
        reference_coverage_summary = summarize_coverage(
            reference_coverage_array[selected]
        )
        mean_difference = float(
            np.max(
                np.abs(
                    np.mean(candidate_rank_array[selected], axis=0)
                    - np.mean(reference_rank_array[selected], axis=0)
                )
            )
        )
        candidate_rows = [
            row
            for row in candidate_summaries
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        reference_rows = [
            row
            for row in reference_summaries
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        valid_fraction = float(
            np.mean(
                [row["all_coordinates_diagnostic_valid"] for row in candidate_rows]
            )
        )
        finite_rhats = [
            float(row["maximum_rhat"])
            for row in candidate_rows
            if row["maximum_rhat"] is not None
        ]
        candidate_coverage_pass = all(
            candidate_coverage_summary["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        reference_coverage_pass = all(
            reference_coverage_summary["maximum_absolute_error_by_level"][index]
            <= float(tolerances[str(level)])
            for index, level in enumerate(levels)
        )
        scenario_pass = bool(
            candidate_rank_summary["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_candidate_mean_rank_z"])
            and reference_rank_summary["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_reference_mean_rank_z"])
            and mean_difference
            <= float(
                GATES["maximum_absolute_candidate_reference_mean_rank_difference"]
            )
            and candidate_coverage_pass
            and reference_coverage_pass
            and valid_fraction
            >= float(
                GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"]
            )
            and len(finite_rhats) == len(candidate_rows)
            and max(finite_rhats)
            <= float(GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in candidate_rows)
            >= float(GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in candidate_rows)
            >= float(GATES["minimum_tail_effective_samples_per_valid_coordinate"])
            and min(
                row["minimum_stage_conditional_ess_fraction"]
                for row in reference_rows
            )
            >= float(GATES["reference_minimum_stage_conditional_ess_fraction"])
            and max(
                row["maximum_standardized_replicate_mean_difference"]
                for row in reference_rows
            )
            <= float(
                GATES["reference_maximum_standardized_replicate_mean_difference"]
            )
            and min(
                row["minimum_final_unique_initial_ancestor_fraction"]
                for row in reference_rows
            )
            >= float(GATES["reference_minimum_unique_initial_ancestor_fraction"])
            and all(row["all_replicates_reached_beta_one"] for row in reference_rows)
        )
        all_scenarios_pass &= scenario_pass
        scenario_results.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank_summary,
                "reference_rank": reference_rank_summary,
                "candidate_coverage": candidate_coverage_summary,
                "reference_coverage": reference_coverage_summary,
                "maximum_absolute_candidate_reference_mean_rank_difference": (
                    mean_difference
                ),
                "fraction_fits_all_coordinates_diagnostic_valid": valid_fraction,
                "maximum_fit_rhat": max(finite_rhats) if finite_rhats else None,
                "minimum_fit_bulk_ess": min(
                    row["minimum_bulk_ess"] for row in candidate_rows
                ),
                "minimum_fit_tail_ess": min(
                    row["minimum_tail_ess"] for row in candidate_rows
                ),
                "reference_minimum_stage_conditional_ess_fraction": min(
                    row["minimum_stage_conditional_ess_fraction"]
                    for row in reference_rows
                ),
                "reference_maximum_standardized_replicate_mean_difference": max(
                    row["maximum_standardized_replicate_mean_difference"]
                    for row in reference_rows
                ),
                "reference_minimum_unique_initial_ancestor_fraction": min(
                    row["minimum_final_unique_initial_ancestor_fraction"]
                    for row in reference_rows
                ),
                "candidate_coverage_passed": candidate_coverage_pass,
                "reference_coverage_passed": reference_coverage_pass,
                "passed": scenario_pass,
            }
        )
    actual_calls = candidate_calls + reference_calls
    maximum_calls = config["call_accounting"][
        "maximum_total_synthetic_likelihood_evaluations"
    ]
    if actual_calls > maximum_calls:
        raise RuntimeError("SBC V2 exceeded frozen maximum calls")
    passed = bool(all_scenarios_pass and orbit_control["passed"])
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_V2_FAILED_RESULT_RETAINED"
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "config_sha256": config["_config_sha256"],
        "passed": passed,
        "v1_diagnosis": {
            "short_chain_mixing_failure": True,
            "finite_sobol_importance_collapse": True,
            "v1_result_or_thresholds_changed": False,
        },
        "scenario_results": scenario_results,
        "candidate_fit_summaries": candidate_summaries,
        "independent_reference_summaries": reference_summaries,
        "orbit_invariance_control": orbit_control,
        "call_accounting": {
            "actual_candidate_synthetic_likelihood_evaluations": candidate_calls,
            "actual_independent_reference_synthetic_likelihood_evaluations": (
                reference_calls
            ),
            "actual_total_synthetic_likelihood_evaluations": actual_calls,
            "frozen_maximum_total_synthetic_likelihood_evaluations": maximum_calls,
            "real_forward_model_evaluations": 0,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "chronology": config["chronology"],
    }
    write_result(
        output,
        {
            "truth_units": np.asarray(truth_units),
            "scenario_indices": scenario_index_array,
            "candidate_normalized_ranks": candidate_rank_array,
            "reference_normalized_ranks": reference_rank_array,
            "candidate_coverage": candidate_coverage_array,
            "reference_coverage": reference_coverage_array,
            "candidate_tie_counts": np.asarray(candidate_ties),
            "reference_tie_mass": np.asarray(reference_tie_mass),
        },
        summary,
    )
    return summary


def write_receipt(
    config_path: Path,
    expected_config_sha256: str,
    result_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    result_target = confined(result_path)
    if result_target != ROOT / RESULT_PATH:
        raise RuntimeError("SBC V2 result path changed")
    archive = np.load(result_target, allow_pickle=False)
    summary = json.loads(str(archive["summary"].item()))
    if (
        summary["schema_version"] != RESULT_SCHEMA
        or summary["config_sha256"] != expected_config_sha256
        or summary["data_boundary"] != DATA_BOUNDARY
        or summary["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise RuntimeError("SBC V2 result boundary changed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": (
            "bounded_synthetic_sbc_v2_passed_not_candidate_production"
            if summary["passed"]
            else "bounded_synthetic_sbc_v2_failed_result_retained"
        ),
        "decision": summary["decision"],
        "evidence": {
            "config": artifact_binding(config_path),
            "implementation_source": artifact_binding(
                ROOT / config["implementation_source"]
            ),
            "canonical_sampler": config["canonical_sampler_binding"],
            "v1_receipt": config["v1_evidence_bindings"]["receipt"],
            "v1_result": config["v1_evidence_bindings"]["result"],
            "bounded_v2_result": artifact_binding(result_target),
        },
        "v1_diagnosis": summary["v1_diagnosis"],
        "principled_v2_changes": {
            "candidate_chain_lengthening": CANDIDATE_INFERENCE["change_from_v1"],
            "independent_reference_replacement": REFERENCE["change_from_v1"],
            "finite_simulation_threshold_basis": {
                "simulations_per_scenario": 32,
                "familywise_z": GATES["finite_simulation_familywise_z"],
                "continuity_allowance": GATES[
                    "finite_simulation_continuity_allowance"
                ],
                "coverage_tolerances": GATES[
                    "coverage_tolerances_for_32_simulations"
                ],
            },
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_results"],
        "controls": {
            "orbit_invariance": summary["orbit_invariance_control"],
            "reference_algorithm": REFERENCE["algorithm_id"],
            "reference_uses_candidate_transition": False,
            "reference_uses_candidate_orbits": False,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "V2 still calibrates a synthetic Gaussian likelihood in the exact nuisance quotient, not the cluster forward model.",
            "Sixty-four simulations improve but do not provide publication-grade calibration precision.",
            "The independent reference is finite-particle adaptive-tempering SMC and is judged by replicate agreement, ancestry, and conditional ESS gates rather than analytic exactness.",
            "A pass cannot support candidate physics, complete CP5.7-CP5.10, or authorize production; a failure is retained without threshold repair.",
        ],
        "replay": {
            "check": (
                "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sbc_v2 "
                f"check --config {config_path.as_posix()} "
                f"--expected-config-sha256 {expected_config_sha256} "
                f"--receipt {output.as_posix()}"
            ),
            "candidate_production": "NOT_RUN",
        },
    }
    body["content_sha256"] = content_sha256(body)
    sampler.write_json(output, body)
    return body


def check(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    load_config(config_path, expected_config_sha256)
    receipt_target = confined(receipt_path)
    body = json.loads(receipt_target.read_text(encoding="utf-8"))
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256", None)
    if observed != content_sha256(unhashed):
        raise RuntimeError("SBC V2 receipt content hash changed")
    if body["schema_version"] != RECEIPT_SCHEMA:
        raise RuntimeError("SBC V2 receipt schema changed")
    for row in body["evidence"].values():
        target = confined(ROOT / row["path"])
        if file_sha256(target) != row["file_sha256"]:
            raise RuntimeError("SBC V2 receipt evidence changed")
    if body["data_boundary"] != DATA_BOUNDARY or body["claim_boundary"] != CLAIM_BOUNDARY:
        raise RuntimeError("SBC V2 receipt data or claim boundary changed")
    return {
        "valid": True,
        "passed": body["status"]
        == "bounded_synthetic_sbc_v2_passed_not_candidate_production",
        "config_sha256": expected_config_sha256,
        "receipt_sha256": file_sha256(receipt_target),
        "synthetic_likelihood_evaluations": body["counts"][
            "actual_total_synthetic_likelihood_evaluations"
        ],
        "real_forward_model_evaluations": 0,
        "candidate_production_runs": 0,
        "scientific_claim_allowed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_command = subparsers.add_parser("run-bounded")
    run_command.add_argument("--config", type=Path, required=True)
    run_command.add_argument("--expected-config-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    receipt_command = subparsers.add_parser("write-receipt")
    receipt_command.add_argument("--config", type=Path, required=True)
    receipt_command.add_argument("--expected-config-sha256", required=True)
    receipt_command.add_argument("--result", type=Path, required=True)
    receipt_command.add_argument("--output", type=Path, required=True)
    check_command = subparsers.add_parser("check")
    check_command.add_argument("--config", type=Path, required=True)
    check_command.add_argument("--expected-config-sha256", required=True)
    check_command.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run-bounded":
        config = load_config(args.config, args.expected_config_sha256)
        print(json.dumps(run_bounded(config, args.output), sort_keys=True))
        return
    if args.command == "write-receipt":
        print(
            json.dumps(
                write_receipt(
                    args.config,
                    args.expected_config_sha256,
                    args.result,
                    args.output,
                ),
                sort_keys=True,
            )
        )
        return
    print(
        json.dumps(
            check(args.config, args.expected_config_sha256, args.receipt),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
