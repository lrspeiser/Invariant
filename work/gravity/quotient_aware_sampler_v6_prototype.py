from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, rankdata

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

SCHEMA = "invariant-gravity-cluster-quotient-sampler-contract-6.0"
COMPOSITES = (
    "outer_nonthermal_fraction",
    "nonthermal_radial_power",
    "outer_pressure_boundary_sigma",
    "density_error_sigma",
    "missing_stellar_to_gas_mass_ratio",
    "clumping_amplitude",
    "spherical_acceleration_scale",
    "projected_gas_geometry_scale",
    "published_stellar_acceleration_scale",
    "temperature_density_calibration_scale",
)
ACTIVE_INDICES = (0, 1, 3, 4, 11, 12, 16, 2, 5, 14)
ORBIT_NAMES = ("stellar", "geometry", "coupled")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_source_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        missing = sorted(expected - set(value))
        extra = sorted(set(value) - expected)
        raise RuntimeError(f"{label} keys changed; missing={missing}, extra={extra}")


def repository_root() -> Path:
    return Path(__file__).resolve().parents[2]


def validate_sampler_settings(
    settings: dict[str, Any], label: str, expected_replicates: int, expected_particles: int
) -> None:
    strict_keys(
        settings,
        {
            "replicates",
            "particles",
            "source_smc_role",
            "adaptation_sweeps",
            "fixed_kernel_settling_sweeps",
            "retained_sweeps",
            "thin",
            "retained_snapshots_per_particle_chain",
            "covariance_refresh_during_adaptation",
            "initial_active_scale",
            "active_scale_bounds",
            "target_acceptance",
            "adaptation_gain",
            "active_kernel",
            "active_primitive_indices",
            "stellar_log_step",
            "geometry_log_step",
            "coupled_log_step",
            "orbit_validation_cases_per_move_per_replicate",
            "seed",
        },
        label,
    )
    if (
        int(settings["replicates"]) != expected_replicates
        or int(settings["particles"]) != expected_particles
        or settings["source_smc_role"]
        != "starting_positions_only_with_all_starting_likelihoods_recomputed"
        or settings["active_kernel"]
        != "symmetric_correlated_gaussian_with_whole_proposal_out_of_bounds_rejection"
        or tuple(settings["active_primitive_indices"]) != ACTIVE_INDICES
        or int(settings["retained_sweeps"]) % int(settings["thin"])
        or int(settings["retained_snapshots_per_particle_chain"])
        != int(settings["retained_sweeps"]) // int(settings["thin"])
        or int(settings["adaptation_sweeps"]) <= 0
        or int(settings["fixed_kernel_settling_sweeps"]) <= 0
        or int(settings["retained_sweeps"]) <= 0
        or int(settings["orbit_validation_cases_per_move_per_replicate"]) <= 0
    ):
        raise RuntimeError(f"{label} violates frozen sampler semantics")


def maximum_forward_calls(settings: dict[str, Any]) -> int:
    replicates = int(settings["replicates"])
    particles = int(settings["particles"])
    validation = (
        2
        * replicates
        * len(ORBIT_NAMES)
        * int(settings["orbit_validation_cases_per_move_per_replicate"])
    )
    proposals = replicates * particles * (
        int(settings["adaptation_sweeps"])
        + int(settings["fixed_kernel_settling_sweeps"])
        + int(settings["retained_sweeps"])
    )
    return validation + replicates * particles + proposals


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    root = repository_root()
    contract_path = path.resolve()
    try:
        contract_path.relative_to(root)
    except ValueError as error:
        raise RuntimeError("contract escaped repository") from error
    observed_contract_sha256 = file_sha256(contract_path)
    if observed_contract_sha256 != expected_sha256:
        raise RuntimeError("execution contract hash differs from the authorized hash")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    strict_keys(
        contract,
        {
            "schema_version",
            "status",
            "purpose",
            "prototype_source",
            "prototype_source_normalized_sha256",
            "source_smc_result",
            "source_smc_sha256",
            "family",
            "likelihood_split",
            "data_seal",
            "primitive_prior_semantics",
            "production_settings",
            "smoke_settings",
            "orbit_validation",
            "diagnostic_validation",
            "prior_recovery_control",
            "completion_thresholds",
            "mechanics_thresholds",
            "call_accounting",
            "adjudication",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA:
        raise RuntimeError("contract schema changed")
    if contract["status"] != "frozen_before_candidate_quotient_sampler_v6":
        raise RuntimeError("contract status is not frozen V6")
    source_path = (root / str(contract["prototype_source"])).resolve()
    if source_path != Path(__file__).resolve():
        raise RuntimeError("contract points to another prototype")
    if normalized_source_sha256(source_path) != str(
        contract["prototype_source_normalized_sha256"]
    ):
        raise RuntimeError("V6 prototype source changed after freeze")
    source_smc = (root / str(contract["source_smc_result"])).resolve()
    try:
        source_smc.relative_to(root)
    except ValueError as error:
        raise RuntimeError("source SMC escaped repository") from error
    if not source_smc.is_file() or file_sha256(source_smc) != str(
        contract["source_smc_sha256"]
    ):
        raise RuntimeError("source SMC changed after freeze")
    if contract["family"] != "cross_scale_boundary":
        raise RuntimeError("candidate family changed")
    if contract["likelihood_split"] != "development_train":
        raise RuntimeError("likelihood split changed")
    seal = contract["data_seal"]
    if seal != {
        "holdout_may_select_sampler_or_settings": False,
        "same_release_confirmation_rows_allowed": False,
        "independent_target_rows_allowed": False,
        "target_rows_opened": 0,
    }:
        raise RuntimeError("development data seal changed")
    if contract["primitive_prior_semantics"] != (
        "unchanged_17_independent_uniform_priors_with_exact_joint_pushforward_to_10_composites"
    ):
        raise RuntimeError("primitive prior semantics changed")
    validate_sampler_settings(contract["production_settings"], "production settings", 4, 512)
    validate_sampler_settings(contract["smoke_settings"], "smoke settings", 2, 32)
    completion = contract["completion_thresholds"]
    if completion != {
        "maximum_rank_normalized_split_rhat": 1.2,
        "minimum_bulk_effective_samples": 50,
        "minimum_tail_effective_samples": 50,
        "maximum_standardized_between_replicate_median_spread": 0.25,
        "all_10_composite_coordinates_must_pass": True,
        "source_start_to_posterior_shift_is_a_gate": False,
    }:
        raise RuntimeError("completion thresholds changed")
    mechanics = contract["mechanics_thresholds"]
    if mechanics != {
        "minimum_retained_active_acceptance": 0.05,
        "maximum_retained_active_acceptance": 0.6,
        "minimum_retained_orbit_acceptance": 0.1,
        "all_replicates_must_pass": True,
    }:
        raise RuntimeError("mechanics thresholds changed")
    production = contract["production_settings"]
    smoke = contract["smoke_settings"]
    expected_accounting = {
        "production_initialization_evaluations": 4 * 512,
        "production_orbit_validation_evaluations": (
            2
            * 4
            * len(ORBIT_NAMES)
            * int(production["orbit_validation_cases_per_move_per_replicate"])
        ),
        "production_maximum_proposal_evaluations": (
            4
            * 512
            * (
                int(production["adaptation_sweeps"])
                + int(production["fixed_kernel_settling_sweeps"])
                + int(production["retained_sweeps"])
            )
        ),
        "production_maximum_total_forward_evaluations": maximum_forward_calls(production),
        "smoke_maximum_total_forward_evaluations": maximum_forward_calls(smoke),
        "out_of_bounds_proposals_require_forward_evaluation": False,
        "actual_calls_must_equal_sum_of_reported_call_categories": True,
    }
    if contract["call_accounting"] != expected_accounting:
        raise RuntimeError("frozen call accounting changed or is inconsistent")
    contract["_execution_contract_sha256"] = observed_contract_sha256
    return contract


def prior_bounds(config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    return lows, highs


def physical_values(unit: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    lows, highs = prior_bounds(config)
    return lows + unit * (highs - lows)


def unit_values(physical: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    lows, highs = prior_bounds(config)
    return (physical - lows) / (highs - lows)


def composite_values(unit: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    shape = unit.shape
    values = physical_values(unit.reshape(-1, shape[-1]), config)
    by_name = {
        name: values[:, index] for index, name in enumerate(uncertainty.PARAMETERS)
    }
    stellar = np.clip(
        by_name["bcg_mass_scale"]
        * by_name["satellite_mass_scale"]
        * (1.0 + by_name["missing_member_fraction"])
        * (1.0 + by_name["intracluster_light_fraction"])
        * by_name["imf_mass_scale"]
        * by_name["mass_to_light_scale"],
        0.4,
        2.5,
    )
    geometry = (1.0 + by_name["centering_radius_shift"]) * by_name[
        "triaxial_radius_scale"
    ]
    result = np.column_stack(
        [
            by_name["outer_nonthermal_fraction"],
            by_name["nonthermal_radial_power"],
            by_name["outer_pressure_boundary_sigma"],
            by_name["density_error_sigma"],
            by_name["missing_stellar_to_gas_mass_ratio"],
            by_name["clumping_amplitude"],
            by_name["spherical_acceleration_scale"],
            by_name["projection_density_scale"] * geometry,
            stellar / geometry**2,
            by_name["xray_temperature_cross_calibration"]
            / by_name["projection_density_scale"],
        ]
    )
    return result.reshape(*shape[:-1], len(COMPOSITES))


def covariance_square_root(values: np.ndarray) -> np.ndarray:
    covariance = np.cov(values, rowvar=False, ddof=1)
    covariance += np.eye(values.shape[1]) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10)))


def stellar_factors(physical: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            physical[5],
            physical[6],
            1.0 + physical[7],
            1.0 + physical[8],
            physical[9],
            physical[10],
        ]
    )


def set_stellar_factors(physical: np.ndarray, factors: np.ndarray) -> None:
    physical[5] = factors[0]
    physical[6] = factors[1]
    physical[7] = factors[2] - 1.0
    physical[8] = factors[3] - 1.0
    physical[9] = factors[4]
    physical[10] = factors[5]


def within(values: np.ndarray, lows: np.ndarray, highs: np.ndarray) -> bool:
    return bool(np.all(values >= lows) and np.all(values <= highs))


def apply_orbit_move(
    physical: np.ndarray,
    move: str,
    rng: np.random.Generator,
    config: dict[str, Any],
    stellar_log_step: float,
    geometry_log_step: float,
    coupled_log_step: float,
) -> bool:
    low, high = prior_bounds(config)
    if move == "stellar":
        factors = stellar_factors(physical)
        partner = int(rng.integers(1, len(factors)))
        delta = float(rng.normal(scale=stellar_log_step))
        proposed = factors.copy()
        proposed[0] *= math.exp(-delta)
        proposed[partner] *= math.exp(delta)
        factor_low = np.asarray([0.75, 0.75, 1.0, 1.0, 0.7, 0.8])
        factor_high = np.asarray([1.25, 1.25, 1.2, 1.3, 1.3, 1.2])
        if within(proposed, factor_low, factor_high):
            set_stellar_factors(physical, proposed)
            return True
        return False
    if move == "geometry":
        factors = np.asarray([1.0 + physical[13], physical[15]])
        delta = float(rng.normal(scale=geometry_log_step))
        proposed = factors * np.asarray([math.exp(delta), math.exp(-delta)])
        if within(proposed, np.asarray([0.95, 0.85]), np.asarray([1.05, 1.15])):
            physical[13] = proposed[0] - 1.0
            physical[15] = proposed[1]
            return True
        return False
    if move == "coupled":
        factors = stellar_factors(physical)
        raw_stellar = float(np.prod(factors))
        delta = float(rng.normal(scale=coupled_log_step))
        proposed = physical.copy()
        proposed[15] *= math.exp(delta)
        proposed[14] *= math.exp(-delta)
        proposed[5] *= math.exp(2.0 * delta)
        proposed[2] *= math.exp(-delta)
        proposed_raw_stellar = raw_stellar * math.exp(2.0 * delta)
        indices = np.asarray([15, 14, 5, 2])
        if (
            0.4 < raw_stellar < 2.5
            and 0.4 < proposed_raw_stellar < 2.5
            and within(proposed[indices], low[indices], high[indices])
            and math.log(max(rng.random(), np.finfo(float).tiny))
            < min(0.0, delta)
        ):
            physical[:] = proposed
            return True
        return False
    raise RuntimeError(f"unknown orbit move: {move}")


def orbit_sweep(
    particles: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, int]:
    physical = physical_values(particles, config)
    counts = {
        f"{move}_{suffix}": 0
        for move in ORBIT_NAMES
        for suffix in ("attempted", "accepted")
    }
    for row in physical:
        for move in ORBIT_NAMES:
            counts[f"{move}_attempted"] += 1
            accepted = apply_orbit_move(
                row,
                move,
                rng,
                config,
                float(settings["stellar_log_step"]),
                float(settings["geometry_log_step"]),
                float(settings["coupled_log_step"]),
            )
            counts[f"{move}_accepted"] += int(accepted)
    particles[:] = unit_values(physical, config)
    if np.any(particles < -1e-12) or np.any(particles > 1.0 + 1e-12):
        raise RuntimeError("orbit move escaped primitive prior")
    particles[:] = np.clip(particles, 0.0, 1.0)
    return counts


class LikelihoodEvaluator:
    def __init__(
        self,
        packets: list[dict[str, Any]],
        family: dict[str, Any],
        config: dict[str, Any],
        config59: dict[str, Any],
    ) -> None:
        self.packets = packets
        self.family = family
        self.config = config
        self.config59 = config59
        self.calls = 0

    def __call__(self, unit: np.ndarray) -> float:
        self.calls += 1
        return float(
            uncertainty._evaluate_unit(
                unit, self.packets, self.family, self.config, self.config59
            )[0]
        )


def bounded_correlated_active_proposal(
    particles: np.ndarray,
    rng: np.random.Generator,
    square_root: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    raw_active = particles[:, ACTIVE_INDICES] + scale * (
        rng.normal(size=(len(particles), len(ACTIVE_INDICES))) @ square_root.T
    )
    in_bounds = np.all((raw_active >= 0.0) & (raw_active <= 1.0), axis=1)
    return raw_active, in_bounds


def active_transition(
    particles: np.ndarray,
    log_likelihood: np.ndarray,
    evaluator: LikelihoodEvaluator,
    rng: np.random.Generator,
    square_root: np.ndarray,
    scale: float,
) -> dict[str, int]:
    raw_active, in_bounds = bounded_correlated_active_proposal(
        particles, rng, square_root, scale
    )
    valid_indices = np.flatnonzero(in_bounds)
    proposal_log_likelihood = np.empty(len(valid_indices))
    proposals = particles[valid_indices].copy()
    proposals[:, ACTIVE_INDICES] = raw_active[valid_indices]
    for local_index, proposal in enumerate(proposals):
        proposal_log_likelihood[local_index] = evaluator(proposal)
    accepted_local = np.log(
        np.maximum(rng.random(len(valid_indices)), np.finfo(float).tiny)
    ) < np.minimum(
        0.0, proposal_log_likelihood - log_likelihood[valid_indices]
    )
    accepted_indices = valid_indices[accepted_local]
    particles[accepted_indices] = proposals[accepted_local]
    log_likelihood[accepted_indices] = proposal_log_likelihood[accepted_local]
    return {
        "attempted": len(particles),
        "out_of_bounds_rejected": int(len(particles) - len(valid_indices)),
        "evaluated": len(valid_indices),
        "accepted": len(accepted_indices),
    }


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def split_chains(chains: np.ndarray) -> np.ndarray:
    if chains.ndim != 3 or chains.shape[1] < 4:
        raise RuntimeError("diagnostics require at least four draws per chain")
    half = chains.shape[1] // 2
    return np.concatenate((chains[:, :half], chains[:, -half:]), axis=0)


def rank_normalize(values: np.ndarray) -> np.ndarray:
    flattened = values.reshape(-1)
    ranks = rankdata(flattened, method="average")
    transformed = norm.ppf((ranks - 3.0 / 8.0) / (len(ranks) + 1.0 / 4.0))
    return transformed.reshape(values.shape)


def classic_rhat(values: np.ndarray) -> float:
    chain_count, draws = values.shape
    if chain_count < 2 or draws < 2:
        return math.inf
    within = float(np.mean(np.var(values, axis=1, ddof=1)))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1))
    if within <= np.finfo(float).tiny:
        return 1.0 if between <= np.finfo(float).tiny else math.inf
    variance = (draws - 1.0) / draws * within + between / draws
    return float(math.sqrt(max(variance / within, 0.0)))


def geyer_ess(values: np.ndarray) -> float:
    chain_count, draws = values.shape
    centered = values - np.mean(values, axis=1, keepdims=True)
    within = float(np.mean(np.sum(centered**2, axis=1) / max(draws - 1, 1)))
    between = float(draws * np.var(np.mean(values, axis=1), ddof=1))
    variance_plus = (draws - 1.0) / draws * within + between / draws
    total = float(chain_count * draws)
    if variance_plus <= np.finfo(float).tiny:
        return total
    rho = [1.0]
    for lag in range(1, draws):
        autocovariance = float(
            np.mean(np.sum(centered[:, : draws - lag] * centered[:, lag:], axis=1))
            / draws
        )
        rho.append(1.0 - (within - autocovariance) / variance_plus)
    pair_sums = []
    for index in range(0, len(rho) - 1, 2):
        pair = rho[index] + rho[index + 1]
        if pair < 0.0:
            break
        pair_sums.append(pair)
    for index in range(1, len(pair_sums)):
        pair_sums[index] = min(pair_sums[index], pair_sums[index - 1])
    tau = max(1.0, -1.0 + 2.0 * sum(pair_sums))
    return float(min(total, total / tau))


def rank_split_diagnostics(chains: np.ndarray) -> dict[str, np.ndarray]:
    split = split_chains(chains)
    dimensions = split.shape[2]
    rhat = np.empty(dimensions)
    bulk_ess = np.empty(dimensions)
    tail_ess = np.empty(dimensions)
    for dimension in range(dimensions):
        raw = split[:, :, dimension]
        ranked = rank_normalize(raw)
        folded = rank_normalize(np.abs(raw - np.median(raw)))
        rhat[dimension] = max(classic_rhat(ranked), classic_rhat(folded))
        bulk_ess[dimension] = geyer_ess(ranked)
        low = float(np.quantile(raw, 0.05))
        high = float(np.quantile(raw, 0.95))
        tail_ess[dimension] = min(
            geyer_ess((raw <= low).astype(float)),
            geyer_ess((raw >= high).astype(float)),
        )
    return {"rhat": rhat, "bulk_ess": bulk_ess, "tail_ess": tail_ess}


def diagnostic_validation_control(settings: dict[str, Any]) -> dict[str, Any]:
    rng = np.random.default_rng(int(settings["seed"]))
    chains = int(settings["chains"])
    draws = int(settings["draws"])
    iid = rng.normal(size=(chains, draws, 1))
    shifted = iid.copy()
    shifted[0, :, 0] += float(settings["shifted_chain_offset"])
    rho = float(settings["ar1_rho"])
    ar1 = np.empty((chains, draws, 1))
    ar1[:, 0, 0] = rng.normal(size=chains)
    innovations = rng.normal(scale=math.sqrt(1.0 - rho**2), size=(chains, draws - 1))
    for draw in range(1, draws):
        ar1[:, draw, 0] = rho * ar1[:, draw - 1, 0] + innovations[:, draw - 1]
    iid_result = rank_split_diagnostics(iid)
    shifted_result = rank_split_diagnostics(shifted)
    ar1_result = rank_split_diagnostics(ar1)
    observed = {
        "iid_rhat": float(iid_result["rhat"][0]),
        "iid_bulk_ess": float(iid_result["bulk_ess"][0]),
        "shifted_rhat": float(shifted_result["rhat"][0]),
        "ar1_rhat": float(ar1_result["rhat"][0]),
        "ar1_bulk_ess": float(ar1_result["bulk_ess"][0]),
    }
    passed = (
        observed["iid_rhat"] <= float(settings["maximum_iid_rhat"])
        and observed["shifted_rhat"] >= float(settings["minimum_shifted_rhat"])
        and observed["ar1_bulk_ess"] < observed["iid_bulk_ess"]
        * float(settings["maximum_ar1_to_iid_ess_ratio"])
        and observed["ar1_bulk_ess"] >= float(settings["minimum_ar1_bulk_ess"])
    )
    return {"passed": bool(passed), "observed": observed}


def maximum_ks_uniform(values: np.ndarray) -> float:
    count = len(values)
    expected_high = np.arange(1, count + 1) / count
    expected_low = np.arange(count) / count
    maximum = 0.0
    for dimension in range(values.shape[1]):
        ordered = np.sort(values[:, dimension])
        maximum = max(
            maximum,
            float(np.max(expected_high - ordered)),
            float(np.max(ordered - expected_low)),
        )
    return maximum


def prior_statistics(values: np.ndarray) -> dict[str, float]:
    correlation = np.corrcoef(values, rowvar=False)
    return {
        "maximum_absolute_mean_error": float(np.max(np.abs(np.mean(values, axis=0) - 0.5))),
        "maximum_absolute_variance_error": float(
            np.max(np.abs(np.var(values, axis=0, ddof=1) - 1.0 / 12.0))
        ),
        "maximum_absolute_offdiagonal_correlation": float(
            np.max(np.abs(correlation - np.eye(values.shape[1])))
        ),
        "maximum_ks_distance_from_uniform": maximum_ks_uniform(values),
    }


def statistics_pass(
    statistics: dict[str, float], thresholds: dict[str, Any]
) -> bool:
    return bool(
        statistics["maximum_absolute_mean_error"]
        <= float(thresholds["maximum_absolute_mean_error"])
        and statistics["maximum_absolute_variance_error"]
        <= float(thresholds["maximum_absolute_variance_error"])
        and statistics["maximum_absolute_offdiagonal_correlation"]
        <= float(thresholds["maximum_absolute_offdiagonal_correlation"])
        and statistics["maximum_ks_distance_from_uniform"]
        <= float(thresholds["maximum_ks_distance_from_uniform"])
    )


def prior_recovery_control(
    settings: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    samples = int(settings["samples"])
    dimensions = len(uncertainty.PARAMETERS)
    active_dimensions = len(ACTIVE_INDICES)
    correlation = float(settings["proposal_ar1_correlation"])
    covariance = correlation ** np.abs(
        np.subtract.outer(np.arange(active_dimensions), np.arange(active_dimensions))
    )
    square_root = np.linalg.cholesky(covariance)
    initial_rng = np.random.default_rng(int(settings["initial_seed"]))
    initial = initial_rng.random((samples, dimensions))
    corrected = initial.copy()
    invalid = initial.copy()
    corrected_rng = np.random.default_rng(int(settings["transition_seed"]))
    invalid_rng = np.random.default_rng(int(settings["transition_seed"]))
    orbit_settings = {
        "stellar_log_step": settings["stellar_log_step"],
        "geometry_log_step": settings["geometry_log_step"],
        "coupled_log_step": settings["coupled_log_step"],
    }
    corrected_oob = 0
    invalid_reflections = 0
    for _sweep in range(int(settings["sweeps"])):
        orbit_sweep(corrected, corrected_rng, config, orbit_settings)
        orbit_sweep(invalid, invalid_rng, config, orbit_settings)
        corrected_raw, corrected_valid = bounded_correlated_active_proposal(
            corrected,
            corrected_rng,
            square_root,
            float(settings["proposal_scale"]),
        )
        corrected_oob += int(np.sum(~corrected_valid))
        corrected[np.ix_(corrected_valid, ACTIVE_INDICES)] = corrected_raw[
            corrected_valid
        ]
        invalid_raw = invalid[:, ACTIVE_INDICES] + float(settings["proposal_scale"]) * (
            invalid_rng.normal(size=(samples, active_dimensions)) @ square_root.T
        )
        invalid_reflections += int(
            np.sum(np.any((invalid_raw < 0.0) | (invalid_raw > 1.0), axis=1))
        )
        wrapped = np.mod(invalid_raw, 2.0)
        invalid[:, ACTIVE_INDICES] = np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)
    corrected_statistics = prior_statistics(corrected)
    invalid_statistics = prior_statistics(invalid)
    thresholds = settings["thresholds"]
    corrected_passed = statistics_pass(corrected_statistics, thresholds)
    invalid_rejected = not statistics_pass(invalid_statistics, thresholds)
    return {
        "passed": bool(corrected_passed and invalid_rejected),
        "corrected_kernel_passed": corrected_passed,
        "invalid_correlated_reflection_rejected": invalid_rejected,
        "corrected_out_of_bounds_self_loops": corrected_oob,
        "invalid_boundary_reflections": invalid_reflections,
        "corrected_statistics": corrected_statistics,
        "invalid_reflection_statistics": invalid_statistics,
    }


def validate_orbits(
    source_particles: np.ndarray,
    evaluator: LikelihoodEvaluator,
    rng: np.random.Generator,
    config: dict[str, Any],
    settings: dict[str, Any],
    cases_per_move_per_replicate: int,
) -> dict[str, Any]:
    maximum_likelihood_difference = 0.0
    maximum_composite_difference = 0.0
    accepted_cases = {move: 0 for move in ORBIT_NAMES}
    calls_before = evaluator.calls
    for replicate_particles in source_particles:
        for move in ORBIT_NAMES:
            for _case in range(cases_per_move_per_replicate):
                index = int(rng.integers(len(replicate_particles)))
                original = replicate_particles[index].copy()
                original_likelihood = evaluator(original)
                proposed_physical = physical_values(original, config)
                accepted = False
                for _attempt in range(1000):
                    proposed_physical = physical_values(original, config)
                    accepted = apply_orbit_move(
                        proposed_physical,
                        move,
                        rng,
                        config,
                        float(settings["stellar_log_step"]),
                        float(settings["geometry_log_step"]),
                        float(settings["coupled_log_step"]),
                    )
                    if accepted:
                        break
                if not accepted:
                    raise RuntimeError(f"could not construct accepted {move} validation move")
                proposed = unit_values(proposed_physical, config)
                if np.any(proposed < 0.0) or np.any(proposed > 1.0):
                    raise RuntimeError("accepted orbit validation move left prior cube")
                proposed_likelihood = evaluator(proposed)
                maximum_likelihood_difference = max(
                    maximum_likelihood_difference,
                    abs(proposed_likelihood - original_likelihood),
                )
                maximum_composite_difference = max(
                    maximum_composite_difference,
                    float(
                        np.max(
                            np.abs(
                                composite_values(proposed, config)
                                - composite_values(original, config)
                            )
                        )
                    ),
                )
                accepted_cases[move] += 1
    return {
        "accepted_cases": accepted_cases,
        "evaluations": evaluator.calls - calls_before,
        "maximum_absolute_training_log_likelihood_difference": maximum_likelihood_difference,
        "maximum_absolute_composite_difference": maximum_composite_difference,
    }


def phase_counter() -> dict[str, int]:
    return {
        "attempted": 0,
        "out_of_bounds_rejected": 0,
        "evaluated": 0,
        "accepted": 0,
    }


def run_sampler(
    contract: dict[str, Any],
    settings: dict[str, Any],
    output: Path,
    smoke: bool,
) -> dict[str, Any]:
    root = repository_root()
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    if family["family_id"] != contract["family"]:
        raise RuntimeError("loaded family disagrees with contract")
    source_path = root / str(contract["source_smc_result"])
    source = np.load(source_path, allow_pickle=False)
    source_particles = np.asarray(source["particles"], dtype=float)
    replicates = int(settings["replicates"])
    particles_count = int(settings["particles"])
    expected_shape = (4, 512, len(uncertainty.PARAMETERS))
    if source_particles.shape != expected_shape:
        raise RuntimeError(f"source SMC shape changed: {source_particles.shape}")
    particles_by_replicate = source_particles[:replicates, :particles_count].copy()
    evaluator = LikelihoodEvaluator(packets, family, config, config59)

    diagnostic_control = diagnostic_validation_control(contract["diagnostic_validation"])
    prior_control = prior_recovery_control(contract["prior_recovery_control"], config)
    if not diagnostic_control["passed"] or not prior_control["passed"]:
        raise RuntimeError("frozen sampler controls failed before forward sampling")

    validation_rng = np.random.default_rng(int(settings["seed"]) + 900_000)
    orbit_validation = validate_orbits(
        particles_by_replicate,
        evaluator,
        validation_rng,
        config,
        settings,
        int(settings["orbit_validation_cases_per_move_per_replicate"]),
    )
    orbit_thresholds = contract["orbit_validation"]
    if (
        orbit_validation["maximum_absolute_training_log_likelihood_difference"]
        > float(orbit_thresholds["maximum_absolute_training_log_likelihood_difference"])
        or orbit_validation["maximum_absolute_composite_difference"]
        > float(orbit_thresholds["maximum_absolute_composite_difference"])
    ):
        raise RuntimeError("orbit validation changed a quotient coordinate or likelihood")

    calls_before_initialization = evaluator.calls
    log_likelihood_by_replicate = np.empty((replicates, particles_count))
    for replicate in range(replicates):
        for particle in range(particles_count):
            log_likelihood_by_replicate[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    initialization_evaluations = evaluator.calls - calls_before_initialization

    retained_count = int(settings["retained_sweeps"]) // int(settings["thin"])
    traces = np.empty(
        (replicates, particles_count, retained_count, len(COMPOSITES)), dtype=float
    )
    ending_particles = np.empty_like(particles_by_replicate)
    ending_log_likelihood = np.empty_like(log_likelihood_by_replicate)
    replicate_summaries = []
    phase_evaluations = {phase: 0 for phase in ("adaptation", "settling", "retained")}

    for replicate in range(replicates):
        particles = particles_by_replicate[replicate].copy()
        log_likelihood = log_likelihood_by_replicate[replicate].copy()
        rng = np.random.default_rng(int(settings["seed"]) + replicate)
        scale = float(settings["initial_active_scale"])
        square_root = covariance_square_root(particles[:, ACTIVE_INDICES])
        active_counts = {
            phase: phase_counter() for phase in ("adaptation", "settling", "retained")
        }
        orbit_counts = {
            phase: {
                f"{move}_{suffix}": 0
                for move in ORBIT_NAMES
                for suffix in ("attempted", "accepted")
            }
            for phase in ("adaptation", "settling", "retained")
        }
        retained_index = 0
        phases = (
            ("adaptation", int(settings["adaptation_sweeps"])),
            ("settling", int(settings["fixed_kernel_settling_sweeps"])),
            ("retained", int(settings["retained_sweeps"])),
        )
        for phase, sweeps in phases:
            for sweep in range(sweeps):
                if phase == "adaptation" and sweep % int(
                    settings["covariance_refresh_during_adaptation"]
                ) == 0:
                    square_root = covariance_square_root(
                        particles[:, ACTIVE_INDICES]
                    )
                orbit_result = orbit_sweep(particles, rng, config, settings)
                add_counts(orbit_counts[phase], orbit_result)
                calls_before = evaluator.calls
                transition = active_transition(
                    particles,
                    log_likelihood,
                    evaluator,
                    rng,
                    square_root,
                    scale,
                )
                add_counts(active_counts[phase], transition)
                phase_evaluations[phase] += evaluator.calls - calls_before
                if phase == "adaptation":
                    rate = transition["accepted"] / transition["attempted"]
                    scale *= math.exp(
                        float(settings["adaptation_gain"])
                        * (rate - float(settings["target_acceptance"]))
                    )
                    scale = float(
                        np.clip(
                            scale,
                            float(settings["active_scale_bounds"][0]),
                            float(settings["active_scale_bounds"][1]),
                        )
                    )
                elif phase == "retained" and (sweep + 1) % int(settings["thin"]) == 0:
                    traces[replicate, :, retained_index, :] = composite_values(
                        particles, config
                    )
                    retained_index += 1
            if phase == "adaptation":
                square_root = covariance_square_root(particles[:, ACTIVE_INDICES])
        if retained_index != retained_count:
            raise RuntimeError("retained snapshot accounting changed")
        ending_particles[replicate] = particles
        ending_log_likelihood[replicate] = log_likelihood
        replicate_summaries.append(
            {
                "replicate": replicate,
                "final_active_scale": scale,
                "active_counts": active_counts,
                "orbit_counts": orbit_counts,
            }
        )

    chains = traces.reshape(replicates * particles_count, retained_count, len(COMPOSITES))
    diagnostics = rank_split_diagnostics(chains)
    pooled = traces.reshape(-1, len(COMPOSITES))
    pooled_standard_deviation = np.std(pooled, axis=0, ddof=1)
    replicate_medians = np.median(traces, axis=(1, 2))
    median_spread = np.divide(
        np.ptp(replicate_medians, axis=0),
        pooled_standard_deviation,
        out=np.zeros_like(pooled_standard_deviation),
        where=pooled_standard_deviation > np.finfo(float).tiny,
    )
    source_start_composites = composite_values(particles_by_replicate, config).reshape(
        -1, len(COMPOSITES)
    )
    descriptive_start_shift = np.divide(
        np.abs(np.median(pooled, axis=0) - np.median(source_start_composites, axis=0)),
        pooled_standard_deviation,
        out=np.zeros_like(pooled_standard_deviation),
        where=pooled_standard_deviation > np.finfo(float).tiny,
    )

    completion = contract["completion_thresholds"]
    mechanics = contract["mechanics_thresholds"]
    coordinate_pass = (
        (diagnostics["rhat"] <= float(completion["maximum_rank_normalized_split_rhat"]))
        & (diagnostics["bulk_ess"] >= float(completion["minimum_bulk_effective_samples"]))
        & (diagnostics["tail_ess"] >= float(completion["minimum_tail_effective_samples"]))
        & (
            median_spread
            <= float(completion["maximum_standardized_between_replicate_median_spread"])
        )
    )
    mechanics_rows = []
    mechanics_pass = True
    for row in replicate_summaries:
        active = row["active_counts"]["retained"]
        active_rate = active["accepted"] / active["attempted"]
        active_pass = (
            float(mechanics["minimum_retained_active_acceptance"])
            <= active_rate
            <= float(mechanics["maximum_retained_active_acceptance"])
        )
        orbit_rates = {}
        orbit_pass = True
        for move in ORBIT_NAMES:
            counts = row["orbit_counts"]["retained"]
            rate = counts[f"{move}_accepted"] / counts[f"{move}_attempted"]
            orbit_rates[move] = rate
            orbit_pass &= rate >= float(mechanics["minimum_retained_orbit_acceptance"])
        mechanics_pass &= active_pass and orbit_pass
        mechanics_rows.append(
            {
                "replicate": row["replicate"],
                "retained_active_acceptance": active_rate,
                "retained_orbit_acceptance": orbit_rates,
                "passed": bool(active_pass and orbit_pass),
            }
        )

    accounting = {
        "orbit_validation_evaluations": int(orbit_validation["evaluations"]),
        "initialization_evaluations": int(initialization_evaluations),
        "adaptation_proposal_evaluations": int(phase_evaluations["adaptation"]),
        "settling_proposal_evaluations": int(phase_evaluations["settling"]),
        "retained_proposal_evaluations": int(phase_evaluations["retained"]),
    }
    accounted_total = sum(accounting.values())
    if accounted_total != evaluator.calls:
        raise RuntimeError(
            f"forward evaluation accounting mismatch: {accounted_total} != {evaluator.calls}"
        )
    maximum_calls = maximum_forward_calls(settings)
    if evaluator.calls > maximum_calls:
        raise RuntimeError("actual forward calls exceeded frozen maximum")
    accounting["total_forward_evaluations"] = int(evaluator.calls)
    accounting["frozen_maximum_forward_evaluations"] = int(maximum_calls)

    production_passed = bool(np.all(coordinate_pass) and mechanics_pass)
    if smoke:
        decision = "SMOKE_ONLY_NOT_PRODUCTION_ADJUDICATION"
    elif production_passed:
        decision = "CANDIDATE_SAMPLER_PASS_DOWNSTREAM_GATES_STILL_REQUIRED"
    else:
        decision = "CANDIDATE_SAMPLER_FAIL_FROZEN_GATES"
    aggregate = {
        "schema_version": "invariant-gravity-cluster-quotient-sampler-result-6.0",
        "mode": "smoke" if smoke else "production",
        "decision": decision,
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "family_id": family["family_id"],
        "source_smc_role": "starting_positions_only_not_posterior_reference",
        "source_smc_sha256": contract["source_smc_sha256"],
        "replicates": replicates,
        "particle_chains_per_replicate": particles_count,
        "retained_snapshots_per_chain": retained_count,
        "posterior_draws": len(pooled),
        "controls": {
            "diagnostic_validation": diagnostic_control,
            "constant_likelihood_prior_recovery": prior_control,
            "orbit_validation": orbit_validation,
        },
        "forward_call_accounting": accounting,
        "maximum_rank_normalized_split_rhat": float(np.max(diagnostics["rhat"])),
        "minimum_bulk_effective_samples": float(np.min(diagnostics["bulk_ess"])),
        "minimum_tail_effective_samples": float(np.min(diagnostics["tail_ess"])),
        "maximum_standardized_between_replicate_median_spread": float(
            np.max(median_spread)
        ),
        "maximum_descriptive_start_to_posterior_median_shift": float(
            np.max(descriptive_start_shift)
        ),
        "descriptive_start_shift_is_a_gate": False,
        "all_coordinate_gates_passed": bool(np.all(coordinate_pass)),
        "all_mechanics_gates_passed": bool(mechanics_pass),
        "production_passed": production_passed if not smoke else False,
        "parameters": [
            {
                "coordinate": name,
                "rank_normalized_split_rhat": float(diagnostics["rhat"][index]),
                "bulk_effective_samples": float(diagnostics["bulk_ess"][index]),
                "tail_effective_samples": float(diagnostics["tail_ess"][index]),
                "standardized_between_replicate_median_spread": float(
                    median_spread[index]
                ),
                "descriptive_start_to_posterior_median_shift": float(
                    descriptive_start_shift[index]
                ),
                "passed": bool(coordinate_pass[index]),
            }
            for index, name in enumerate(COMPOSITES)
        ],
        "mechanics": mechanics_rows,
        "replicate_summaries": replicate_summaries,
        "claim_boundary": contract["adjudication"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        composite_traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=np.asarray(json.dumps(aggregate, sort_keys=True)),
    )
    return aggregate


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("controls", "smoke", "run"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "run":
            command.add_argument("--execute-frozen-production-v6", action="store_true")
    args = parser.parse_args()
    contract = load_contract(args.contract, args.expected_contract_sha256)
    root = repository_root()
    config = uncertainty.load_config(root)
    if args.command == "controls":
        result = {
            "schema_version": "invariant-gravity-cluster-quotient-sampler-controls-6.0",
            "execution_contract_sha256": contract["_execution_contract_sha256"],
            "diagnostic_validation": diagnostic_validation_control(
                contract["diagnostic_validation"]
            ),
            "constant_likelihood_prior_recovery": prior_recovery_control(
                contract["prior_recovery_control"], config
            ),
            "forward_evaluations": 0,
        }
        result["passed"] = bool(
            result["diagnostic_validation"]["passed"]
            and result["constant_likelihood_prior_recovery"]["passed"]
        )
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.command == "smoke":
        result = run_sampler(
            contract, contract["smoke_settings"], args.output, smoke=True
        )
        print(json.dumps(result, sort_keys=True))
        return
    if not args.execute_frozen_production_v6:
        raise RuntimeError(
            "production requires the explicit --execute-frozen-production-v6 sentinel"
        )
    result = run_sampler(
        contract, contract["production_settings"], args.output, smoke=False
    )
    print(json.dumps(result, sort_keys=True))
    if not result["production_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
