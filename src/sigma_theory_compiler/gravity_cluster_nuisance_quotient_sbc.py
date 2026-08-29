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
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sbc_v1.json")
ARTIFACT_DIR = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1"
)
RESULT_PATH = ARTIFACT_DIR / "bounded-synthetic-sbc.npz"
RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sbc-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-config-1.0"
RESULT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-result-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sbc-receipt-1.0"

COMPOSITE_SCALES = np.asarray(
    [0.25, 0.75, 2.0, 1.0, 0.14, 0.15, 0.15, 0.20, 0.70, 0.20],
    dtype=float,
)
SCENARIOS = [
    {
        "scenario_id": "moderate_correlated_quotient_observation",
        "simulations": 24,
        "normalized_noise_sigma": 0.45,
        "ar1_correlation": 0.40,
    },
    {
        "scenario_id": "weak_diagonal_quotient_observation",
        "simulations": 24,
        "normalized_noise_sigma": 0.90,
        "ar1_correlation": 0.0,
    },
]
INFERENCE = {
    "replicates": 2,
    "particles_per_replicate": 16,
    "adaptation_sweeps": 16,
    "fixed_kernel_settling_sweeps": 16,
    "retained_sweeps": 128,
    "thin": 4,
    "retained_snapshots_per_particle_chain": 32,
    "covariance_refresh_during_adaptation": 4,
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
}
ORACLE = {
    "engine": "scrambled_sobol_prior_importance_reference",
    "dimensions": 17,
    "power_of_two_exponent": 14,
    "prior_particles_per_simulation": 16384,
    "posterior_resampling": False,
    "weighted_cdf_and_quantiles": True,
}
RANK_PROTOCOL = {
    "coordinates": list(sampler.COMPOSITES),
    "rank_definition": "count_less_plus_discrete_uniform_zero_through_tie_count",
    "normalized_rank_definition": "(integer_rank_plus_0.5)/(posterior_draws_plus_1)",
    "stellar_clipping_atoms_use_randomized_tie_ranks": True,
    "coverage_levels": [0.5, 0.8, 0.9],
    "rank_histogram_bins": 8,
    "posterior_draws_per_simulation": 1024,
    "truths_are_exact_independent_17_primitive_prior_draws": True,
}
GATES = {
    "maximum_absolute_candidate_mean_rank_z": 4.0,
    "maximum_absolute_oracle_mean_rank_z": 4.0,
    "maximum_absolute_candidate_oracle_mean_rank_difference": 0.20,
    "maximum_absolute_candidate_coverage_error": {
        "0.5": 0.30,
        "0.8": 0.25,
        "0.9": 0.20,
    },
    "maximum_absolute_oracle_coverage_error": {
        "0.5": 0.25,
        "0.8": 0.20,
        "0.9": 0.15,
    },
    "minimum_importance_effective_samples": 1000.0,
    "minimum_fraction_fits_all_coordinates_diagnostic_valid": 0.75,
    "maximum_rank_normalized_split_rhat": 1.20,
    "minimum_bulk_effective_samples_per_valid_coordinate": 20.0,
    "minimum_tail_effective_samples_per_valid_coordinate": 20.0,
    "maximum_orbit_composite_difference": 1e-12,
    "all_scenarios_must_pass": True,
    "threshold_relaxation_after_result": False,
    "failed_result_retained": True,
}
SEED_LINEAGE = {
    "truth_base": 710000,
    "noise_base": 720000,
    "sobol_start_base": 730000,
    "transition_base": 740000,
    "rank_tie_base": 750000,
    "oracle_sobol_base": 760000,
    "scenario_stride": 10000,
    "simulation_stride": 10,
    "no_seed_derived_from_observed_result": True,
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


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        raise RuntimeError(f"{label} keys changed")


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def maximum_call_accounting() -> dict[str, int]:
    simulations = sum(int(row["simulations"]) for row in SCENARIOS)
    chains = int(INFERENCE["replicates"]) * int(
        INFERENCE["particles_per_replicate"]
    )
    sweeps = (
        int(INFERENCE["adaptation_sweeps"])
        + int(INFERENCE["fixed_kernel_settling_sweeps"])
        + int(INFERENCE["retained_sweeps"])
    )
    maximum_mcmc = simulations * chains * (1 + sweeps)
    oracle = simulations * int(ORACLE["prior_particles_per_simulation"])
    return {
        "simulations": simulations,
        "maximum_mcmc_synthetic_likelihood_evaluations": maximum_mcmc,
        "oracle_synthetic_likelihood_evaluations": oracle,
        "maximum_total_synthetic_likelihood_evaluations": maximum_mcmc + oracle,
        "real_forward_model_evaluations": 0,
    }


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    config_path = confined(path)
    if file_sha256(config_path) != expected_sha256:
        raise RuntimeError("SBC config hash changed")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "canonical_sampler_binding",
            "uncertainty_config_binding",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "composite_scales",
            "scenarios",
            "inference",
            "oracle",
            "rank_protocol",
            "gates",
            "seed_lineage",
            "data_boundary",
            "claim_boundary",
            "call_accounting",
            "chronology",
            "output_paths",
        },
        "SBC config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_before_bounded_synthetic_sbc"
    ):
        raise RuntimeError("SBC config identity changed")
    implementation = confined(ROOT / config["implementation_source"])
    if implementation != Path(__file__).resolve() or normalized_sha256(
        implementation
    ) != config["implementation_source_normalized_sha256"]:
        raise RuntimeError("SBC implementation changed after freeze")
    sampler_path = confined(ROOT / config["canonical_sampler_binding"]["path"])
    if (
        sampler_path != Path(sampler.__file__).resolve()
        or file_sha256(sampler_path)
        != config["canonical_sampler_binding"]["file_sha256"]
    ):
        raise RuntimeError("canonical sampler binding changed")
    uncertainty_path = confined(ROOT / config["uncertainty_config_binding"]["path"])
    if file_sha256(uncertainty_path) != config["uncertainty_config_binding"][
        "file_sha256"
    ]:
        raise RuntimeError("uncertainty prior binding changed")
    uncertainty_config = json.loads(uncertainty_path.read_text(encoding="utf-8"))
    exact_priors = uncertainty_config["continuous_priors"]
    if config["exact_primitive_priors"] != exact_priors or len(exact_priors) != 17:
        raise RuntimeError("exact 17 primitive priors changed")
    expected = {
        "primitive_prior_semantics": (
            "17_independent_uniform_primitives_with_clipped_six_factor_"
            "stellar_pushforward_clip_0.4_2.5"
        ),
        "composite_scales": COMPOSITE_SCALES.tolist(),
        "scenarios": SCENARIOS,
        "inference": INFERENCE,
        "oracle": ORACLE,
        "rank_protocol": RANK_PROTOCOL,
        "gates": GATES,
        "seed_lineage": SEED_LINEAGE,
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "call_accounting": maximum_call_accounting(),
        "chronology": {
            "config_and_gates_frozen_before_first_result": True,
            "result_written_before_receipt": True,
            "receipt_requires_hash_bound_result": True,
            "failed_result_retained": True,
            "post_result_threshold_changes_forbidden": True,
        },
        "output_paths": {
            "result": RESULT_PATH.as_posix(),
            "receipt": RECEIPT_PATH.as_posix(),
        },
    }
    for name, value in expected.items():
        if config[name] != value:
            raise RuntimeError(f"frozen SBC object changed: {name}")
    config["_config_sha256"] = expected_sha256
    return config


def scenario_covariance(scenario: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    rho = float(scenario["ar1_correlation"])
    sigma = float(scenario["normalized_noise_sigma"])
    indices = np.arange(len(sampler.COMPOSITES))
    correlation = rho ** np.abs(indices[:, None] - indices[None, :])
    covariance = sigma**2 * correlation
    return covariance, np.linalg.inv(covariance)


def normalized_composites(unit: np.ndarray, prior_config: dict[str, Any]) -> np.ndarray:
    return sampler.composite_values(unit, prior_config) / COMPOSITE_SCALES


class SyntheticLikelihood:
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
        residual = normalized_composites(unit, self.prior_config) - self.observation
        return float(-0.5 * residual @ self.inverse_covariance @ residual)


def batch_log_likelihood(
    units: np.ndarray,
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
) -> np.ndarray:
    residual = normalized_composites(units, prior_config) - observation
    return -0.5 * np.einsum(
        "ni,ij,nj->n", residual, inverse_covariance, residual
    )


def randomized_integer_ranks(
    draws: np.ndarray, truth: np.ndarray, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks = np.empty(draws.shape[1], dtype=int)
    ties = np.empty(draws.shape[1], dtype=int)
    for coordinate in range(draws.shape[1]):
        less = int(np.count_nonzero(draws[:, coordinate] < truth[coordinate]))
        equal = int(np.count_nonzero(draws[:, coordinate] == truth[coordinate]))
        ranks[coordinate] = less + int(rng.integers(0, equal + 1))
        ties[coordinate] = equal
    normalized = (ranks + 0.5) / (len(draws) + 1.0)
    return normalized, ties


def weighted_cdf_ranks(
    values: np.ndarray,
    weights: np.ndarray,
    truth: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ranks = np.empty(values.shape[1])
    tie_mass = np.empty(values.shape[1])
    for coordinate in range(values.shape[1]):
        less = float(np.sum(weights[values[:, coordinate] < truth[coordinate]]))
        equal = float(np.sum(weights[values[:, coordinate] == truth[coordinate]]))
        ranks[coordinate] = less + float(rng.random()) * equal
        tie_mass[coordinate] = equal
    return ranks, tie_mass


def weighted_quantile(
    values: np.ndarray, weights: np.ndarray, probabilities: np.ndarray
) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    return np.interp(probabilities, cumulative, sorted_values)


def coverage_rows(
    values: np.ndarray,
    truth: np.ndarray,
    levels: list[float],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    result = np.zeros((values.shape[1], len(levels)), dtype=bool)
    for coordinate in range(values.shape[1]):
        for level_index, level in enumerate(levels):
            tail = (1.0 - level) / 2.0
            if weights is None:
                low, high = np.quantile(values[:, coordinate], [tail, 1.0 - tail])
            else:
                low, high = weighted_quantile(
                    values[:, coordinate], weights, np.asarray([tail, 1.0 - tail])
                )
            result[coordinate, level_index] = bool(
                low <= truth[coordinate] <= high
            )
    return result


def sobol_start_populations(global_index: int) -> np.ndarray:
    populations = []
    for replicate in range(int(INFERENCE["replicates"])):
        seed = (
            int(SEED_LINEAGE["sobol_start_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        populations.append(qmc.Sobol(d=17, scramble=True, seed=seed).random_base2(m=4))
    return np.stack(populations)


def run_mcmc_fit(
    observation: np.ndarray,
    inverse_covariance: np.ndarray,
    prior_config: dict[str, Any],
    global_index: int,
) -> tuple[np.ndarray, dict[str, Any], int]:
    particles_by_replicate = sobol_start_populations(global_index)
    replicates, particle_count, _ = particles_by_replicate.shape
    evaluator = SyntheticLikelihood(observation, inverse_covariance, prior_config)
    log_likelihood = np.empty((replicates, particle_count))
    for replicate in range(replicates):
        for particle in range(particle_count):
            log_likelihood[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    retained = int(INFERENCE["retained_snapshots_per_particle_chain"])
    traces = np.empty(
        (replicates, particle_count, retained, len(sampler.COMPOSITES)), dtype=float
    )
    active_evaluated = 0
    active_out_of_bounds = 0
    active_accepted = 0
    orbit_attempted = 0
    orbit_accepted = 0
    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        likelihood = log_likelihood[replicate].copy()
        rng = np.random.default_rng(
            int(SEED_LINEAGE["transition_base"])
            + global_index * int(SEED_LINEAGE["simulation_stride"])
            + replicate
        )
        scale = float(INFERENCE["initial_active_scale"])
        square_root = sampler.covariance_square_root(
            particles[:, sampler.ACTIVE_INDICES]
        )
        retained_index = 0
        phases = (
            ("adaptation", int(INFERENCE["adaptation_sweeps"])),
            ("settling", int(INFERENCE["fixed_kernel_settling_sweeps"])),
            ("retained", int(INFERENCE["retained_sweeps"])),
        )
        for phase, sweeps in phases:
            for sweep in range(sweeps):
                if (
                    phase == "adaptation"
                    and sweep % int(INFERENCE["covariance_refresh_during_adaptation"])
                    == 0
                ):
                    square_root = sampler.covariance_square_root(
                        particles[:, sampler.ACTIVE_INDICES]
                    )
                orbit = sampler.orbit_sweep(particles, rng, prior_config, INFERENCE)
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
                active_evaluated += int(transition["evaluated"])
                active_out_of_bounds += int(transition["out_of_bounds_rejected"])
                active_accepted += int(transition["accepted"])
                if phase == "adaptation":
                    rate = transition["accepted"] / transition["attempted"]
                    scale *= math.exp(
                        float(INFERENCE["adaptation_gain"])
                        * (rate - float(INFERENCE["target_acceptance"]))
                    )
                    scale = float(
                        np.clip(
                            scale,
                            float(INFERENCE["active_scale_bounds"][0]),
                            float(INFERENCE["active_scale_bounds"][1]),
                        )
                    )
                elif phase == "retained" and (sweep + 1) % int(
                    INFERENCE["thin"]
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
            raise RuntimeError("retained SBC snapshot accounting changed")
    chains = traces.reshape(
        replicates * particle_count, retained, len(sampler.COMPOSITES)
    )
    diagnostics = sampler.rank_split_diagnostics(chains)
    draws = chains.reshape(-1, len(sampler.COMPOSITES))
    summary = {
        "all_coordinates_diagnostic_valid": bool(np.all(diagnostics["valid"])),
        "maximum_rhat": (
            float(np.max(diagnostics["rhat"]))
            if np.all(np.isfinite(diagnostics["rhat"]))
            else None
        ),
        "minimum_bulk_ess": float(np.min(diagnostics["bulk_ess"])),
        "minimum_tail_ess": float(np.min(diagnostics["tail_ess"])),
        "active_evaluated": active_evaluated,
        "active_out_of_bounds_self_loops": active_out_of_bounds,
        "active_accepted": active_accepted,
        "orbit_attempted": orbit_attempted,
        "orbit_accepted": orbit_accepted,
        "initial_likelihoods_recomputed_fresh": replicates * particle_count,
    }
    return draws, summary, evaluator.calls


def orbit_invariance_control(prior_config: dict[str, Any]) -> dict[str, Any]:
    maximum_difference = 0.0
    accepted = {}
    for move_index, move in enumerate(sampler.ORBIT_NAMES):
        rng = np.random.default_rng(780000 + move_index)
        count = 0
        for case in range(8):
            physical = sampler.physical_values(
                np.random.default_rng(781000 + 100 * move_index + case).random(17),
                prior_config,
            )
            before = sampler.composite_values(
                sampler.unit_values(physical, prior_config), prior_config
            )
            for _ in range(10000):
                proposed = physical.copy()
                if sampler.apply_orbit_move(
                    proposed, move, rng, prior_config, 0.08, 0.03, 0.02
                ):
                    after = sampler.composite_values(
                        sampler.unit_values(proposed, prior_config), prior_config
                    )
                    maximum_difference = max(
                        maximum_difference, float(np.max(np.abs(after - before)))
                    )
                    count += 1
                    break
            else:
                raise RuntimeError(f"could not accept synthetic {move} orbit control")
        accepted[move] = count
    return {
        "accepted_cases": accepted,
        "maximum_absolute_composite_difference": maximum_difference,
        "passed": bool(
            all(value == 8 for value in accepted.values())
            and maximum_difference
            <= float(GATES["maximum_orbit_composite_difference"])
        ),
    }


def rank_summary(ranks: np.ndarray) -> dict[str, Any]:
    simulations = ranks.shape[0]
    standard_error = math.sqrt(1.0 / (12.0 * simulations))
    mean = np.mean(ranks, axis=0)
    z = (mean - 0.5) / standard_error
    bins = int(RANK_PROTOCOL["rank_histogram_bins"])
    histograms = [
        np.histogram(ranks[:, index], bins=bins, range=(0.0, 1.0))[0].tolist()
        for index in range(ranks.shape[1])
    ]
    return {
        "mean_normalized_rank": mean.tolist(),
        "mean_rank_z": z.tolist(),
        "maximum_absolute_mean_rank_z": float(np.max(np.abs(z))),
        "histograms": histograms,
        "bins": bins,
    }


def coverage_summary(values: np.ndarray) -> dict[str, Any]:
    levels = list(map(float, RANK_PROTOCOL["coverage_levels"]))
    observed = np.mean(values, axis=0)
    errors = np.abs(observed - np.asarray(levels)[None, :])
    return {
        "levels": levels,
        "observed_by_coordinate": observed.tolist(),
        "maximum_absolute_error_by_level": np.max(errors, axis=0).tolist(),
    }


def write_result(path: Path, arrays: dict[str, np.ndarray], summary: dict[str, Any]) -> None:
    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            **arrays,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    sampler._write_then_publish_no_clobber(path, writer, suffix=".npz.tmp")


def run_bounded(config: dict[str, Any], output: Path) -> dict[str, Any]:
    if confined(output) != ROOT / RESULT_PATH:
        raise RuntimeError("bounded SBC output path changed")
    prior_config = uncertainty.load_config(ROOT)
    orbit_control = orbit_invariance_control(prior_config)
    levels = list(map(float, RANK_PROTOCOL["coverage_levels"]))
    candidate_ranks = []
    oracle_ranks = []
    candidate_coverages = []
    oracle_coverages = []
    candidate_ties = []
    oracle_tie_mass = []
    truth_units = []
    fit_summaries = []
    importance_ess = []
    scenario_indices = []
    mcmc_calls = 0
    oracle_calls = 0
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
            truth_normalized = truth / COMPOSITE_SCALES
            noise = np.random.default_rng(
                int(SEED_LINEAGE["noise_base"]) + seed_offset
            ).normal(size=len(sampler.COMPOSITES))
            observation = truth_normalized + covariance_root @ noise
            draws, fit, calls = run_mcmc_fit(
                observation, inverse_covariance, prior_config, global_index
            )
            mcmc_calls += calls
            rank, ties = randomized_integer_ranks(
                draws,
                truth,
                int(SEED_LINEAGE["rank_tie_base"]) + seed_offset,
            )
            candidate_ranks.append(rank)
            candidate_ties.append(ties)
            candidate_coverages.append(coverage_rows(draws, truth, levels))
            oracle_units = qmc.Sobol(
                d=17,
                scramble=True,
                seed=int(SEED_LINEAGE["oracle_sobol_base"]) + seed_offset,
            ).random_base2(m=int(ORACLE["power_of_two_exponent"]))
            oracle_values = sampler.composite_values(oracle_units, prior_config)
            oracle_log_likelihood = batch_log_likelihood(
                oracle_units,
                observation,
                inverse_covariance,
                prior_config,
            )
            oracle_calls += len(oracle_units)
            weights = np.exp(oracle_log_likelihood - np.max(oracle_log_likelihood))
            weights /= np.sum(weights)
            ess = float(1.0 / np.sum(weights**2))
            oracle_rank, tie_mass = weighted_cdf_ranks(
                oracle_values,
                weights,
                truth,
                int(SEED_LINEAGE["rank_tie_base"]) + 500000 + seed_offset,
            )
            oracle_ranks.append(oracle_rank)
            oracle_tie_mass.append(tie_mass)
            oracle_coverages.append(
                coverage_rows(oracle_values, truth, levels, weights)
            )
            truth_units.append(truth_unit)
            importance_ess.append(ess)
            scenario_indices.append(scenario_index)
            fit_summaries.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "simulation_index": simulation_index,
                    **fit,
                    "importance_effective_samples": ess,
                }
            )
            global_index += 1
    candidate_rank_array = np.asarray(candidate_ranks)
    oracle_rank_array = np.asarray(oracle_ranks)
    candidate_coverage_array = np.asarray(candidate_coverages)
    oracle_coverage_array = np.asarray(oracle_coverages)
    scenario_index_array = np.asarray(scenario_indices, dtype=int)
    scenario_summaries = []
    all_scenarios_pass = True
    for scenario_index, scenario in enumerate(SCENARIOS):
        selected = scenario_index_array == scenario_index
        candidate_rank_metrics = rank_summary(candidate_rank_array[selected])
        oracle_rank_metrics = rank_summary(oracle_rank_array[selected])
        candidate_coverage_metrics = coverage_summary(
            candidate_coverage_array[selected]
        )
        oracle_coverage_metrics = coverage_summary(oracle_coverage_array[selected])
        candidate_oracle_mean_difference = float(
            np.max(
                np.abs(
                    np.mean(candidate_rank_array[selected], axis=0)
                    - np.mean(oracle_rank_array[selected], axis=0)
                )
            )
        )
        scenario_fit_rows = [
            row
            for row in fit_summaries
            if row["scenario_id"] == scenario["scenario_id"]
        ]
        diagnostic_valid_fraction = float(
            np.mean(
                [row["all_coordinates_diagnostic_valid"] for row in scenario_fit_rows]
            )
        )
        finite_rhats = [
            float(row["maximum_rhat"])
            for row in scenario_fit_rows
            if row["maximum_rhat"] is not None
        ]
        candidate_coverage_pass = all(
            candidate_coverage_metrics["maximum_absolute_error_by_level"][index]
            <= float(GATES["maximum_absolute_candidate_coverage_error"][str(level)])
            for index, level in enumerate(levels)
        )
        oracle_coverage_pass = all(
            oracle_coverage_metrics["maximum_absolute_error_by_level"][index]
            <= float(GATES["maximum_absolute_oracle_coverage_error"][str(level)])
            for index, level in enumerate(levels)
        )
        scenario_pass = bool(
            candidate_rank_metrics["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_candidate_mean_rank_z"])
            and oracle_rank_metrics["maximum_absolute_mean_rank_z"]
            <= float(GATES["maximum_absolute_oracle_mean_rank_z"])
            and candidate_oracle_mean_difference
            <= float(
                GATES["maximum_absolute_candidate_oracle_mean_rank_difference"]
            )
            and candidate_coverage_pass
            and oracle_coverage_pass
            and min(
                row["importance_effective_samples"] for row in scenario_fit_rows
            )
            >= float(GATES["minimum_importance_effective_samples"])
            and diagnostic_valid_fraction
            >= float(
                GATES["minimum_fraction_fits_all_coordinates_diagnostic_valid"]
            )
            and len(finite_rhats) == len(scenario_fit_rows)
            and max(finite_rhats)
            <= float(GATES["maximum_rank_normalized_split_rhat"])
            and min(row["minimum_bulk_ess"] for row in scenario_fit_rows)
            >= float(GATES["minimum_bulk_effective_samples_per_valid_coordinate"])
            and min(row["minimum_tail_ess"] for row in scenario_fit_rows)
            >= float(GATES["minimum_tail_effective_samples_per_valid_coordinate"])
        )
        all_scenarios_pass &= scenario_pass
        scenario_summaries.append(
            {
                "scenario_id": scenario["scenario_id"],
                "simulations": int(np.count_nonzero(selected)),
                "candidate_rank": candidate_rank_metrics,
                "oracle_rank": oracle_rank_metrics,
                "candidate_coverage": candidate_coverage_metrics,
                "oracle_coverage": oracle_coverage_metrics,
                "maximum_absolute_candidate_oracle_mean_rank_difference": (
                    candidate_oracle_mean_difference
                ),
                "minimum_importance_effective_samples": min(
                    row["importance_effective_samples"] for row in scenario_fit_rows
                ),
                "fraction_fits_all_coordinates_diagnostic_valid": (
                    diagnostic_valid_fraction
                ),
                "maximum_fit_rhat": max(finite_rhats) if finite_rhats else None,
                "minimum_fit_bulk_ess": min(
                    row["minimum_bulk_ess"] for row in scenario_fit_rows
                ),
                "minimum_fit_tail_ess": min(
                    row["minimum_tail_ess"] for row in scenario_fit_rows
                ),
                "passed": scenario_pass,
            }
        )
    actual_calls = mcmc_calls + oracle_calls
    maximum = config["call_accounting"][
        "maximum_total_synthetic_likelihood_evaluations"
    ]
    if actual_calls > maximum:
        raise RuntimeError("bounded SBC exceeded frozen maximum calls")
    passed = bool(all_scenarios_pass and orbit_control["passed"])
    decision = (
        "BOUNDED_SYNTHETIC_QUOTIENT_SBC_PASSED_NOT_PHYSICS_OR_PRODUCTION"
        if passed
        else "BOUNDED_SYNTHETIC_QUOTIENT_SBC_FAILED_RESULT_RETAINED"
    )
    summary = {
        "schema_version": RESULT_SCHEMA,
        "decision": decision,
        "config_sha256": config["_config_sha256"],
        "passed": passed,
        "scenario_summaries": scenario_summaries,
        "orbit_invariance_control": orbit_control,
        "fit_summaries": fit_summaries,
        "call_accounting": {
            "actual_mcmc_synthetic_likelihood_evaluations": mcmc_calls,
            "oracle_synthetic_likelihood_evaluations": oracle_calls,
            "actual_total_synthetic_likelihood_evaluations": actual_calls,
            "frozen_maximum_total_synthetic_likelihood_evaluations": maximum,
            "real_forward_model_evaluations": 0,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "chronology": config["chronology"],
    }
    arrays = {
        "truth_units": np.asarray(truth_units),
        "scenario_indices": scenario_index_array,
        "candidate_normalized_ranks": candidate_rank_array,
        "oracle_weighted_cdf_ranks": oracle_rank_array,
        "candidate_coverage": candidate_coverage_array,
        "oracle_coverage": oracle_coverage_array,
        "candidate_tie_counts": np.asarray(candidate_ties),
        "oracle_tie_mass": np.asarray(oracle_tie_mass),
        "importance_effective_samples": np.asarray(importance_ess),
    }
    write_result(output, arrays, summary)
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
        raise RuntimeError("SBC result path changed")
    archive = np.load(result_target, allow_pickle=False)
    summary = json.loads(str(archive["summary"].item()))
    if (
        summary["schema_version"] != RESULT_SCHEMA
        or summary["config_sha256"] != expected_config_sha256
        or summary["data_boundary"] != DATA_BOUNDARY
        or summary["claim_boundary"] != CLAIM_BOUNDARY
    ):
        raise RuntimeError("SBC result boundary changed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": (
            "bounded_synthetic_sbc_passed_not_candidate_production"
            if summary["passed"]
            else "bounded_synthetic_sbc_failed_result_retained"
        ),
        "decision": summary["decision"],
        "evidence": {
            "config": artifact_binding(config_path),
            "implementation_source": artifact_binding(
                ROOT / config["implementation_source"]
            ),
            "canonical_sampler": config["canonical_sampler_binding"],
            "bounded_result": artifact_binding(result_target),
        },
        "counts": summary["call_accounting"],
        "scenario_results": summary["scenario_summaries"],
        "controls": {
            "orbit_invariance": summary["orbit_invariance_control"],
            "importance_reference_present": True,
            "randomized_stellar_clipping_tie_ranks": True,
        },
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "limitations": [
            "This calibrates a synthetic Gaussian likelihood in the exact ten-dimensional nuisance quotient, not the real cluster forward model.",
            "Forty-eight simulations provide a bounded control, not publication-grade SBC precision.",
            "The importance reference is a finite scrambled-Sobol approximation, not an analytic posterior.",
            "Passing cannot support the candidate physics, complete CP5.7-CP5.10, or authorize candidate production.",
        ],
        "replay": {
            "check": (
                "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sbc "
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
        raise RuntimeError("SBC receipt content hash changed")
    if body["schema_version"] != RECEIPT_SCHEMA:
        raise RuntimeError("SBC receipt schema changed")
    for row in body["evidence"].values():
        path = confined(ROOT / row["path"])
        if file_sha256(path) != row["file_sha256"]:
            raise RuntimeError("SBC receipt evidence changed")
    if body["data_boundary"] != DATA_BOUNDARY or body["claim_boundary"] != CLAIM_BOUNDARY:
        raise RuntimeError("SBC receipt claim or data boundary changed")
    return {
        "valid": True,
        "passed": body["status"]
        == "bounded_synthetic_sbc_passed_not_candidate_production",
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
