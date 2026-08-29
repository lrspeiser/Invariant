from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm, qmc, rankdata

from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path("configs/gravity_cluster_nuisance_quotient_sampler_v1.json")
ARTIFACT_DIR = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sampler-v1"
)
IMPLEMENTATION_RECEIPT = Path(
    "runs/gravity/publication-readiness/nuisance-quotient-sampler-implementation-v1.json"
)
SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sampler-config-1.0"
RESULT_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-sampler-result-1.0"
UNAUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-1.0-unauthorized"
AUTHORIZED_SCHEMA = "invariant-gravity-quotient-sampler-authorization-1.0-authorized"
APPROVAL_SCHEMA = "invariant-gravity-quotient-sampler-external-approval-1.0"
AUTHORIZATION_CONTROLS_SCHEMA = (
    "invariant-gravity-quotient-sampler-authorization-transition-controls-1.0"
)
WRITE_RACE_CONTROLS_SCHEMA = "invariant-gravity-atomic-no-clobber-controls-1.0"
IMPLEMENTATION_RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-nuisance-quotient-sampler-implementation-1.0"
)
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
MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS = 1_575_104

SOURCE_PATHS = {
    "uncertainty_config": "configs/gravity_cluster_uncertainty_program_v1.json",
    "uncertainty_module": "src/sigma_theory_compiler/gravity_cluster_uncertainty_program.py",
    "quotient_config": "configs/gravity_cluster_nuisance_quotient_audit_v1.json",
    "quotient_module": "src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_audit.py",
    "quotient_receipt": "runs/gravity/publication-readiness/nuisance-quotient-audit-v1.json",
    "comparator_module": "src/sigma_theory_compiler/gravity_cluster_comparator_suite.py",
    "comparator_receipt": "runs/gravity/publication-readiness/comparator-suite-v1.json",
    "item59_config": "configs/gravity_item59_xcop_forward_observable_gate_v1.json",
    "item59_module": "src/sigma_theory_compiler/gravity_item59_xcop_forward_observable_gate.py",
    "item59_result": "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1.json",
}
PRIMITIVE_PRIORS = [
    {"parameter": "outer_nonthermal_fraction", "cause": "nonthermal_pressure", "distribution": "uniform", "low": 0.0, "high": 0.5},
    {"parameter": "nonthermal_radial_power", "cause": "nonthermal_pressure", "distribution": "uniform", "low": 0.5, "high": 2.0},
    {"parameter": "xray_temperature_cross_calibration", "cause": "cross_calibration", "distribution": "uniform", "low": 0.85, "high": 1.15},
    {"parameter": "outer_pressure_boundary_sigma", "cause": "boundary_condition", "distribution": "uniform", "low": -2.0, "high": 2.0},
    {"parameter": "density_error_sigma", "cause": "density_measurement", "distribution": "uniform", "low": -1.0, "high": 1.0},
    {"parameter": "bcg_mass_scale", "cause": "BCG_stellar_mass", "distribution": "uniform", "low": 0.75, "high": 1.25},
    {"parameter": "satellite_mass_scale", "cause": "satellite_stellar_mass", "distribution": "uniform", "low": 0.75, "high": 1.25},
    {"parameter": "missing_member_fraction", "cause": "missing_members", "distribution": "uniform", "low": 0.0, "high": 0.2},
    {"parameter": "intracluster_light_fraction", "cause": "intracluster_light", "distribution": "uniform", "low": 0.0, "high": 0.3},
    {"parameter": "imf_mass_scale", "cause": "IMF", "distribution": "uniform", "low": 0.7, "high": 1.3},
    {"parameter": "mass_to_light_scale", "cause": "stellar_mass_to_light", "distribution": "uniform", "low": 0.8, "high": 1.2},
    {"parameter": "missing_stellar_to_gas_mass_ratio", "cause": "unmeasured_stellar_profile", "distribution": "uniform", "low": 0.02, "high": 0.3},
    {"parameter": "clumping_amplitude", "cause": "gas_clumping", "distribution": "uniform", "low": 0.0, "high": 0.3},
    {"parameter": "centering_radius_shift", "cause": "centering", "distribution": "uniform", "low": -0.05, "high": 0.05},
    {"parameter": "projection_density_scale", "cause": "projection", "distribution": "uniform", "low": 0.85, "high": 1.15},
    {"parameter": "triaxial_radius_scale", "cause": "triaxiality", "distribution": "uniform", "low": 0.85, "high": 1.15},
    {"parameter": "spherical_acceleration_scale", "cause": "spherical_approximation", "distribution": "uniform", "low": 0.85, "high": 1.15},
]
PRODUCTION_SETTINGS = {
    "replicates": 4,
    "particles": 512,
    "sobol_start_population_role": "four_independently_scrambled_prior_populations_with_every_likelihood_recomputed",
    "adaptation_sweeps": 128,
    "fixed_kernel_settling_sweeps": 128,
    "retained_sweeps": 512,
    "thin": 2,
    "retained_snapshots_per_particle_chain": 256,
    "covariance_refresh_during_adaptation": 8,
    "initial_active_scale": 0.752622083091612,
    "active_scale_bounds": [0.02, 8.0],
    "target_acceptance": 0.234,
    "adaptation_gain": 0.5,
    "active_kernel": "symmetric_correlated_gaussian_with_whole_proposal_out_of_bounds_rejection",
    "active_primitive_indices": list(ACTIVE_INDICES),
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "orbit_validation_cases_per_move_per_replicate": 8,
    "seed": 597000,
}
SMOKE_SETTINGS = {
    **PRODUCTION_SETTINGS,
    "replicates": 2,
    "particles": 32,
    "adaptation_sweeps": 8,
    "fixed_kernel_settling_sweeps": 8,
    "retained_sweeps": 16,
    "thin": 2,
    "retained_snapshots_per_particle_chain": 8,
    "covariance_refresh_during_adaptation": 4,
    "orbit_validation_cases_per_move_per_replicate": 2,
}
START_GENERATION = {
    "engine": "scipy_qmc_sobol",
    "independent_scrambles": 4,
    "dimensions": 17,
    "samples_per_scramble": 512,
    "power_of_two_exponent": 9,
    "scramble": True,
    "scramble_seeds": [596200, 596201, 596202, 596203],
    "posterior_ancestry": False,
    "stored_likelihoods": False,
}
DATA_SEAL = {
    "runtime_packet_allowed_split": "development_train",
    "runtime_packet_required_rows": 80,
    "holdout_may_select_sampler_or_settings": False,
    "same_release_confirmation_rows_allowed": False,
    "independent_target_rows_allowed": False,
    "target_rows_opened": 0,
}
ORBIT_VALIDATION = {
    "moves_tested_separately": ["stellar", "geometry", "coupled"],
    "accepted_cases_required": True,
    "maximum_absolute_training_log_likelihood_difference": 1e-10,
    "maximum_absolute_composite_difference": 1e-12,
    "production_forward_evaluations": 192,
    "smoke_forward_evaluations": 24,
}
DIAGNOSTIC_VALIDATION = {
    "seed": 598100,
    "chains": 8,
    "draws": 512,
    "shifted_chain_offset": 3.0,
    "ar1_rho": 0.8,
    "maximum_iid_rhat": 1.05,
    "minimum_shifted_rhat": 1.2,
    "maximum_ar1_to_iid_ess_ratio": 0.5,
    "minimum_ar1_bulk_ess": 50,
    "minimum_scaled_within_chain_variance": 1e-14,
    "maximum_arviz_rhat_absolute_difference": 1e-12,
    "maximum_arviz_ess_relative_difference": 0.02,
    "constant_chain_must_fail_validity_gate": True,
}
UNIFORM_TARGET_CONTROL = {
    "role": "uniform_target_kernel_invariance_and_correlated_boundary_negative_control",
    "samples": 32768,
    "sweeps": 8,
    "initial_seed": 598001,
    "transition_seed": 598002,
    "proposal_ar1_correlation": 0.9,
    "proposal_scale": 0.45,
    "stellar_log_step": 0.08,
    "geometry_log_step": 0.03,
    "coupled_log_step": 0.02,
    "negative_control": "coordinatewise_reflection_with_the_same_correlated_gaussian_must_fail",
    "thresholds": {
        "maximum_absolute_mean_error": 0.012,
        "maximum_absolute_variance_error": 0.006,
        "maximum_absolute_offdiagonal_correlation": 0.05,
        "maximum_ks_distance_from_uniform": 0.015,
    },
}
COMPLETION_THRESHOLDS = {
    "maximum_rank_normalized_split_rhat": 1.2,
    "minimum_bulk_effective_samples": 50,
    "minimum_tail_effective_samples": 50,
    "maximum_standardized_between_replicate_median_spread": 0.25,
    "all_10_composite_coordinates_must_pass": True,
    "positive_variance_required_for_every_split_chain": True,
    "sobol_start_to_posterior_shift_is_a_gate": False,
}
MECHANICS_THRESHOLDS = {
    "minimum_retained_active_acceptance": 0.05,
    "maximum_retained_active_acceptance": 0.6,
    "minimum_retained_orbit_acceptance": 0.1,
    "all_replicates_must_pass": True,
}
ADJUDICATION = {
    "candidate_pass_alone_completes_CP5_7_through_CP5_10": False,
    "newtonian_control_required_after_candidate_pass": True,
    "simulation_based_calibration_required_after_candidate_pass": True,
    "source_covariance_required_before_CP5_completion": True,
    "primitive_orbit_labels_may_be_reported_as_separately_identified": False,
    "holdout_selection_allowed": False,
    "threshold_relaxation_after_result": False,
    "failed_run_retained": True,
    "newtonian_control_locked_until_candidate_pass": True,
}
AUTHORIZATION_POLICY = {
    "unauthorized_schema": UNAUTHORIZED_SCHEMA,
    "authorized_schema": AUTHORIZED_SCHEMA,
    "external_approval_schema": APPROVAL_SCHEMA,
    "contract_status": "external_approval_required_before_production",
    "separate_status_and_boundary_validation": True,
    "external_approval_must_bind_all_frozen_artifacts": True,
    "production_authorized_by_default": False,
    "explicit_cli_sentinel_required": True,
    "unauthorized_attempt_fails_before_contract_or_runtime_packet_load": True,
    "all_generated_artifacts_use_atomic_same_filesystem_no_clobber": True,
}


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def strict_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def confined(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return resolved


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_artifact_binding(row: dict[str, Any], label: str) -> Path:
    strict_keys(row, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(row["path"]))
    if not target.is_file() or file_sha256(target) != row["file_sha256"]:
        raise RuntimeError(f"artifact binding missing or tampered: {label}")
    return target


def _publish_complete_temp_no_clobber(
    temporary_path: Path,
    destination: Path,
    *,
    before_link: Callable[[], None] | None = None,
) -> None:
    temporary = confined(temporary_path)
    target = confined(destination)
    if temporary.parent != target.parent:
        raise RuntimeError("atomic publication requires a same-directory temporary file")
    if before_link is not None:
        before_link()
    try:
        os.link(temporary, target)
    except FileExistsError as error:
        raise RuntimeError("atomic no-clobber publication refused an existing target") from error


def _write_then_publish_no_clobber(
    destination: Path,
    writer: Callable[[Any], None],
    *,
    suffix: str,
    before_link: Callable[[], None] | None = None,
) -> None:
    target = confined(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=".quotient-sampler-complete-",
            suffix=suffix,
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            writer(temporary)
            temporary.flush()
            os.fsync(temporary.fileno())
        _publish_complete_temp_no_clobber(Path(temporary_name), target, before_link=before_link)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def write_json(
    path: Path,
    value: dict[str, Any],
    *,
    before_link: Callable[[], None] | None = None,
) -> None:
    payload = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")

    def writer(handle: Any) -> None:
        handle.write(payload)

    _write_then_publish_no_clobber(path, writer, suffix=".json.tmp", before_link=before_link)


def atomic_save_result(
    output: Path,
    *,
    traces: np.ndarray,
    ending_particles: np.ndarray,
    ending_log_likelihood: np.ndarray,
    summary: dict[str, Any],
    before_link: Callable[[], None] | None = None,
) -> None:
    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            composite_traces=traces,
            ending_particles=ending_particles,
            ending_log_likelihood=ending_log_likelihood,
            summary=np.asarray(json.dumps(summary, sort_keys=True, allow_nan=False)),
        )

    _write_then_publish_no_clobber(output, writer, suffix=".npz.tmp", before_link=before_link)


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


def rank_split_diagnostics(
    chains: np.ndarray, variance_floor: float | None = None
) -> dict[str, np.ndarray]:
    floor = (
        float(DIAGNOSTIC_VALIDATION["minimum_scaled_within_chain_variance"])
        if variance_floor is None
        else float(variance_floor)
    )
    split = split_chains(chains)
    dimensions = split.shape[2]
    rhat = np.full(dimensions, np.inf)
    bulk_ess = np.zeros(dimensions)
    tail_ess = np.zeros(dimensions)
    valid = np.zeros(dimensions, dtype=bool)
    minimum_scaled_variance = np.zeros(dimensions)
    for dimension in range(dimensions):
        raw = split[:, :, dimension]
        scale = max(1.0, float(np.max(np.abs(raw))))
        chain_variances = np.var(raw, axis=1, ddof=1) / scale**2
        minimum_scaled_variance[dimension] = float(np.min(chain_variances))
        if (
            np.any(~np.isfinite(raw))
            or np.any(~np.isfinite(chain_variances))
            or np.min(chain_variances) <= floor
        ):
            continue
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
        valid[dimension] = bool(
            np.isfinite(rhat[dimension])
            and np.isfinite(bulk_ess[dimension])
            and np.isfinite(tail_ess[dimension])
        )
    return {
        "rhat": rhat,
        "bulk_ess": bulk_ess,
        "tail_ess": tail_ess,
        "valid": valid,
        "minimum_scaled_within_chain_variance": minimum_scaled_variance,
    }


def _diagnostic_row(chains: np.ndarray, floor: float) -> dict[str, Any]:
    result = rank_split_diagnostics(chains, floor)
    return {
        "valid": bool(result["valid"][0]),
        "rhat": float(result["rhat"][0]) if np.isfinite(result["rhat"][0]) else None,
        "bulk_ess": float(result["bulk_ess"][0]),
        "tail_ess": float(result["tail_ess"][0]),
        "minimum_scaled_within_chain_variance": float(
            result["minimum_scaled_within_chain_variance"][0]
        ),
    }


def diagnostic_validation_control(settings: dict[str, Any]) -> dict[str, Any]:
    if settings != DIAGNOSTIC_VALIDATION:
        raise RuntimeError("diagnostic validation settings changed")
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
    constant = np.full((chains, draws, 1), 0.375)
    floor = float(settings["minimum_scaled_within_chain_variance"])
    observed = {
        "iid": _diagnostic_row(iid, floor),
        "shifted_chain": _diagnostic_row(shifted, floor),
        "ar1": _diagnostic_row(ar1, floor),
        "constant_chain_negative_control": _diagnostic_row(constant, floor),
    }
    arviz_reference = {
        "iid": {"rhat": 0.999686660115622, "bulk_ess": 4311.461357420721, "tail_ess": 4097.780031223812},
        "shifted_chain": {"rhat": 1.2642047647775638, "bulk_ess": 22.498707525317624, "tail_ess": 24.416761439933108},
        "ar1": {"rhat": 1.0158932761075032, "bulk_ess": 410.4947877916216, "tail_ess": 803.4310744749675},
    }
    rhat_differences = []
    ess_relative_differences = []
    total = float(chains * draws)
    for name in ("iid", "shifted_chain", "ar1"):
        own = observed[name]
        reference = arviz_reference[name]
        rhat_differences.append(abs(float(own["rhat"]) - reference["rhat"]))
        for key in ("bulk_ess", "tail_ess"):
            capped_reference = min(reference[key], total)
            ess_relative_differences.append(
                abs(float(own[key]) - capped_reference) / max(capped_reference, 1.0)
            )
    max_rhat = max(rhat_differences)
    max_ess = max(ess_relative_differences)
    constant_result = observed["constant_chain_negative_control"]
    passed = bool(
        observed["iid"]["valid"]
        and float(observed["iid"]["rhat"]) <= float(settings["maximum_iid_rhat"])
        and observed["shifted_chain"]["valid"]
        and float(observed["shifted_chain"]["rhat"])
        >= float(settings["minimum_shifted_rhat"])
        and observed["ar1"]["valid"]
        and float(observed["ar1"]["bulk_ess"])
        < float(observed["iid"]["bulk_ess"])
        * float(settings["maximum_ar1_to_iid_ess_ratio"])
        and float(observed["ar1"]["bulk_ess"])
        >= float(settings["minimum_ar1_bulk_ess"])
        and not constant_result["valid"]
        and float(constant_result["bulk_ess"]) == 0.0
        and float(constant_result["tail_ess"]) == 0.0
        and max_rhat <= float(settings["maximum_arviz_rhat_absolute_difference"])
        and max_ess <= float(settings["maximum_arviz_ess_relative_difference"])
    )
    return {
        "passed": passed,
        "observed": observed,
        "parity": {
            "reference": "arviz_0.22.0_frozen_reference_vectors",
            "reference_values": arviz_reference,
            "ess_reference_cap": chains * draws,
            "maximum_rhat_absolute_difference": max_rhat,
            "maximum_ess_relative_difference": max_ess,
            "rhat_threshold": settings["maximum_arviz_rhat_absolute_difference"],
            "ess_relative_threshold": settings["maximum_arviz_ess_relative_difference"],
        },
    }


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


def uniform_target_invariance_control(
    settings: dict[str, Any], config: dict[str, Any]
) -> dict[str, Any]:
    if settings != UNIFORM_TARGET_CONTROL:
        raise RuntimeError("uniform-target control settings changed")
    result = prior_recovery_control(settings, config)
    result["control_name"] = (
        "uniform_target_kernel_invariance_and_correlated_boundary_negative_control"
    )
    result["likelihood_model"] = "constant_uniform_target_no_forward_model"
    return result


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

PACKET_SCHEMA = "invariant-gravity-cluster-nuisance-quotient-train-packet-1.0"


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def receipt_content_hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
        + b"\n"
    ).hexdigest()


def binding_rows() -> dict[str, dict[str, str]]:
    return {
        name: {"path": path, "file_sha256": file_sha256(ROOT / path)}
        for name, path in SOURCE_PATHS.items()
    }


def validate_train_packet_body(body: dict[str, Any]) -> None:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "source_bindings",
            "clusters",
            "cluster_count",
            "row_count",
            "split_counts",
            "packets",
            "content_sha256",
        },
        "train packet",
    )
    if body["schema_version"] != PACKET_SCHEMA:
        raise RuntimeError("train packet schema changed")
    if body["status"] != "canonical_train_only_no_holdout_confirmation_or_independent_rows":
        raise RuntimeError("train packet status changed")
    validate_source_bindings(body["source_bindings"])
    if int(body["cluster_count"]) != 8 or int(body["row_count"]) != 80:
        raise RuntimeError("train packet cluster or row count changed")
    if body["split_counts"] != {"development_train": 80}:
        raise RuntimeError("train packet split counts changed")
    required_packet_keys = {
        "cluster",
        "density_radius_kpc",
        "ne_cm3",
        "ne_error_low_cm3",
        "ne_error_high_cm3",
        "r500_kpc",
        "anchor",
        "rows",
        "stellar",
    }
    rows = []
    if len(body["packets"]) != 8:
        raise RuntimeError("train packet count changed")
    for index, packet in enumerate(body["packets"]):
        strict_keys(packet, required_packet_keys, f"train packet.packets[{index}]")
        rows.extend(packet["rows"])
    if len(rows) != 80 or any(row.get("split") != "development_train" for row in rows):
        raise RuntimeError("train packet contains a forbidden split")
    unhashed = dict(body)
    observed = unhashed.pop("content_sha256")
    if observed != canonical_hash(unhashed):
        raise RuntimeError("train packet content hash changed")


def build_train_packet_from_train_only_input(input_path: Path, output: Path) -> dict[str, Any]:
    source = json.loads(confined(input_path).read_text(encoding="utf-8"))
    packets = source.get("packets")
    if not isinstance(packets, list):
        raise TypeError("input does not contain packet rows")
    body = {
        "schema_version": PACKET_SCHEMA,
        "status": "canonical_train_only_no_holdout_confirmation_or_independent_rows",
        "source_bindings": binding_rows(),
        "clusters": sorted(str(packet["cluster"]) for packet in packets),
        "cluster_count": len(packets),
        "row_count": sum(len(packet["rows"]) for packet in packets),
        "split_counts": {"development_train": 80},
        "packets": packets,
    }
    body["content_sha256"] = canonical_hash(body)
    validate_train_packet_body(body)
    write_json(output, body)
    return body


def load_train_packet(path: Path, expected_sha256: str) -> list[dict[str, Any]]:
    packet_path = confined(path)
    if not packet_path.is_file() or file_sha256(packet_path) != expected_sha256:
        raise RuntimeError("train-only packet missing or tampered")
    body = json.loads(packet_path.read_text(encoding="utf-8"))
    validate_train_packet_body(body)
    return body["packets"]


def build_sobol_starts(output: Path) -> dict[str, Any]:
    populations = []
    for seed in START_GENERATION["scramble_seeds"]:
        populations.append(
            qmc.Sobol(d=17, scramble=True, seed=int(seed)).random_base2(m=9)
        )
    particles = np.stack(populations)
    if particles.shape != (4, 512, 17) or np.any((particles <= 0) | (particles >= 1)):
        raise RuntimeError("Sobol start population invariant failed")

    def writer(handle: Any) -> None:
        np.savez_compressed(
            handle,
            particles=particles,
            generation=np.asarray(json.dumps(START_GENERATION, sort_keys=True)),
        )

    _write_then_publish_no_clobber(output, writer, suffix=".npz.tmp")
    return {"shape": list(particles.shape), "file_sha256": file_sha256(output)}


def validate_sobol_starts(path: Path, expected_sha256: str) -> None:
    start_path = confined(path)
    if not start_path.is_file() or file_sha256(start_path) != expected_sha256:
        raise RuntimeError("Sobol starts missing or tampered")
    loaded = np.load(start_path, allow_pickle=False)
    strict_keys(
        {name: None for name in loaded.files}, {"particles", "generation"}, "Sobol archive"
    )
    if json.loads(str(loaded["generation"].item())) != START_GENERATION:
        raise RuntimeError("Sobol start-generation contract changed")
    particles = np.asarray(loaded["particles"], dtype=float)
    if particles.shape != (4, 512, 17) or np.any((particles <= 0) | (particles >= 1)):
        raise RuntimeError("Sobol starts left the open prior cube")


def validate_source_bindings(bindings: dict[str, Any]) -> None:
    strict_keys(bindings, set(SOURCE_PATHS), "source_bindings")
    for name, expected_path in SOURCE_PATHS.items():
        row = bindings[name]
        strict_keys(row, {"path", "file_sha256"}, f"source_bindings.{name}")
        if row["path"] != expected_path:
            raise RuntimeError(f"source binding path changed for {name}")
        validate_artifact_binding(row, f"source_bindings.{name}")


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


def expected_call_accounting() -> dict[str, Any]:
    return {
        "production_initialization_evaluations": 2048,
        "production_orbit_validation_evaluations": 192,
        "production_maximum_proposal_evaluations": 1_572_864,
        "production_maximum_total_forward_evaluations": maximum_forward_calls(
            PRODUCTION_SETTINGS
        ),
        "smoke_maximum_total_forward_evaluations": maximum_forward_calls(SMOKE_SETTINGS),
        "out_of_bounds_proposals_require_forward_evaluation": False,
        "actual_calls_must_equal_sum_of_reported_call_categories": True,
    }


def validate_sampler_settings(
    settings: dict[str, Any], label: str, expected_replicates: int, expected_particles: int
) -> None:
    expected = PRODUCTION_SETTINGS if expected_particles == 512 else SMOKE_SETTINGS
    if settings != expected:
        raise RuntimeError(f"{label} differs from frozen settings")
    if (
        int(settings["replicates"]) != expected_replicates
        or int(settings["particles"]) != expected_particles
        or tuple(settings["active_primitive_indices"]) != ACTIVE_INDICES
        or int(settings["retained_sweeps"]) % int(settings["thin"])
        or int(settings["retained_snapshots_per_particle_chain"])
        != int(settings["retained_sweeps"]) // int(settings["thin"])
    ):
        raise RuntimeError(f"{label} violates frozen sampler semantics")


def load_contract(path: Path, expected_sha256: str) -> dict[str, Any]:
    contract_path = confined(path)
    observed_hash = file_sha256(contract_path)
    if observed_hash != expected_sha256:
        raise RuntimeError("contract hash differs from the expected hash")
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    strict_keys(
        contract,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "source_bindings",
            "exact_primitive_priors",
            "primitive_prior_semantics",
            "train_packet",
            "sobol_start_population",
            "start_generation",
            "family",
            "likelihood_split",
            "data_seal",
            "production_settings",
            "smoke_settings",
            "orbit_validation",
            "diagnostic_validation",
            "uniform_target_invariance_control",
            "completion_thresholds",
            "mechanics_thresholds",
            "call_accounting",
            "adjudication",
            "authorization_policy",
        },
        "contract",
    )
    if contract["schema_version"] != SCHEMA:
        raise RuntimeError("contract schema changed")
    if contract["status"] != "external_approval_required_before_production":
        raise RuntimeError("contract status is not external_approval_required")
    if (
        contract["family"] != "cross_scale_boundary"
        or contract["likelihood_split"] != "development_train"
    ):
        raise RuntimeError("family or likelihood split changed")
    implementation = confined(ROOT / str(contract["implementation_source"]))
    if implementation != Path(__file__).resolve():
        raise RuntimeError("contract points to another executable")
    if normalized_sha256(implementation) != contract["implementation_source_normalized_sha256"]:
        raise RuntimeError("canonical executable changed after freeze")
    validate_source_bindings(contract["source_bindings"])
    if contract["exact_primitive_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("exact 17 primitive priors changed")
    if contract["primitive_prior_semantics"] != (
        "17_independent_uniform_primitives_with_clipped_six_factor_stellar_pushforward_clip_0.4_2.5"
    ):
        raise RuntimeError("primitive prior semantics changed")
    uncertainty_config = json.loads(
        (ROOT / SOURCE_PATHS["uncertainty_config"]).read_text(encoding="utf-8")
    )
    if uncertainty_config["continuous_priors"] != PRIMITIVE_PRIORS:
        raise RuntimeError("hash-bound uncertainty prior definitions changed")
    for name in ("train_packet", "sobol_start_population"):
        validate_artifact_binding(contract[name], name)
    load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    validate_sobol_starts(
        ROOT / contract["sobol_start_population"]["path"],
        contract["sobol_start_population"]["file_sha256"],
    )
    expected_nested = {
        "start_generation": START_GENERATION,
        "data_seal": DATA_SEAL,
        "production_settings": PRODUCTION_SETTINGS,
        "smoke_settings": SMOKE_SETTINGS,
        "orbit_validation": ORBIT_VALIDATION,
        "diagnostic_validation": DIAGNOSTIC_VALIDATION,
        "uniform_target_invariance_control": UNIFORM_TARGET_CONTROL,
        "completion_thresholds": COMPLETION_THRESHOLDS,
        "mechanics_thresholds": MECHANICS_THRESHOLDS,
        "call_accounting": expected_call_accounting(),
        "adjudication": ADJUDICATION,
        "authorization_policy": AUTHORIZATION_POLICY,
    }
    for name, expected in expected_nested.items():
        if contract[name] != expected:
            raise RuntimeError(f"frozen nested contract object changed: {name}")
    validate_sampler_settings(contract["production_settings"], "production settings", 4, 512)
    validate_sampler_settings(contract["smoke_settings"], "smoke settings", 2, 32)
    contract["_execution_contract_sha256"] = observed_hash
    return contract


def add_counts(target: dict[str, int], source: dict[str, int]) -> None:
    for key, value in source.items():
        target[key] = target.get(key, 0) + int(value)


def phase_counter() -> dict[str, int]:
    return {
        "attempted": 0,
        "out_of_bounds_rejected": 0,
        "evaluated": 0,
        "accepted": 0,
    }


def run_sampler(
    contract: dict[str, Any], settings: dict[str, Any], output: Path, *, smoke: bool
) -> dict[str, Any]:
    if confined(output).exists():
        raise RuntimeError("refusing to overwrite an existing sampler result")
    packets = load_train_packet(
        ROOT / contract["train_packet"]["path"],
        contract["train_packet"]["file_sha256"],
    )
    config = uncertainty.load_config(ROOT)
    config59 = item59.load_config(ROOT)
    family = config["candidate_and_control_families"][0]
    if family["family_id"] != contract["family"]:
        raise RuntimeError("loaded candidate family disagrees with contract")
    start_archive = np.load(ROOT / contract["sobol_start_population"]["path"], allow_pickle=False)
    all_start_particles = np.asarray(start_archive["particles"], dtype=float)
    replicates = int(settings["replicates"])
    particles_count = int(settings["particles"])
    if all_start_particles.shape != (4, 512, 17):
        raise RuntimeError("Sobol start population shape changed")
    particles_by_replicate = all_start_particles[:replicates, :particles_count].copy()
    evaluator = LikelihoodEvaluator(packets, family, config, config59)

    diagnostic_control = diagnostic_validation_control(contract["diagnostic_validation"])
    uniform_control = uniform_target_invariance_control(
        contract["uniform_target_invariance_control"], config
    )
    if not diagnostic_control["passed"] or not uniform_control["passed"]:
        raise RuntimeError("frozen controls failed before forward sampling")

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
    orbit_validation_passed = bool(
        orbit_validation["maximum_absolute_training_log_likelihood_difference"]
        <= float(orbit_thresholds["maximum_absolute_training_log_likelihood_difference"])
        and orbit_validation["maximum_absolute_composite_difference"]
        <= float(orbit_thresholds["maximum_absolute_composite_difference"])
        and all(int(orbit_validation["accepted_cases"][move]) > 0 for move in ORBIT_NAMES)
    )
    if not orbit_validation_passed:
        raise RuntimeError("orbit validation failed frozen invariance thresholds")

    calls_before_initialization = evaluator.calls
    log_likelihood_by_replicate = np.empty((replicates, particles_count))
    for replicate in range(replicates):
        for particle in range(particles_count):
            log_likelihood_by_replicate[replicate, particle] = evaluator(
                particles_by_replicate[replicate, particle]
            )
    initialization_evaluations = evaluator.calls - calls_before_initialization
    if initialization_evaluations != replicates * particles_count:
        raise RuntimeError("not every Sobol starting likelihood was recomputed fresh")

    retained_count = int(settings["retained_sweeps"]) // int(settings["thin"])
    traces = np.empty((replicates, particles_count, retained_count, len(COMPOSITES)), dtype=float)
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
        active_counts = {phase: phase_counter() for phase in ("adaptation", "settling", "retained")}
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
                if (
                    phase == "adaptation"
                    and sweep % int(settings["covariance_refresh_during_adaptation"]) == 0
                ):
                    square_root = covariance_square_root(particles[:, ACTIVE_INDICES])
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
    start_composites = composite_values(particles_by_replicate, config).reshape(
        -1, len(COMPOSITES)
    )
    descriptive_start_shift = np.divide(
        np.abs(np.median(pooled, axis=0) - np.median(start_composites, axis=0)),
        pooled_standard_deviation,
        out=np.zeros_like(pooled_standard_deviation),
        where=pooled_standard_deviation > np.finfo(float).tiny,
    )

    completion = contract["completion_thresholds"]
    coordinate_pass = (
        diagnostics["valid"]
        & (diagnostics["rhat"] <= float(completion["maximum_rank_normalized_split_rhat"]))
        & (diagnostics["bulk_ess"] >= float(completion["minimum_bulk_effective_samples"]))
        & (diagnostics["tail_ess"] >= float(completion["minimum_tail_effective_samples"]))
        & (
            median_spread
            <= float(completion["maximum_standardized_between_replicate_median_spread"])
        )
    )
    mechanics = contract["mechanics_thresholds"]
    mechanics_rows = []
    mechanics_pass = True
    for row in replicate_summaries:
        active = row["active_counts"]["retained"]
        active_rate = active["accepted"] / active["attempted"]
        active_pass = bool(
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
        passed = bool(active_pass and orbit_pass)
        mechanics_pass &= passed
        mechanics_rows.append(
            {
                "replicate": row["replicate"],
                "retained_active_acceptance": active_rate,
                "retained_orbit_acceptance": orbit_rates,
                "passed": passed,
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

    gate_pass = bool(np.all(coordinate_pass) and mechanics_pass)
    production_passed = bool(gate_pass and not smoke)
    if smoke:
        decision = "SMOKE_ONLY_NOT_PRODUCTION_ADJUDICATION"
    elif production_passed:
        decision = "CANDIDATE_SAMPLER_PASS_DOWNSTREAM_SBC_NEWTONIAN_AND_SOURCE_COVARIANCE_REQUIRED"
    else:
        decision = "CANDIDATE_SAMPLER_FAIL_FROZEN_GATES_RESULT_RETAINED"
    finite_rhat = np.all(np.isfinite(diagnostics["rhat"]))
    aggregate = {
        "schema_version": RESULT_SCHEMA,
        "mode": "smoke" if smoke else "production",
        "decision": decision,
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "family_id": family["family_id"],
        "sobol_start_population": {
            "role": settings["sobol_start_population_role"],
            "generation": START_GENERATION,
            "file_sha256": contract["sobol_start_population"]["file_sha256"],
            "posterior_ancestry": False,
            "stored_likelihoods": False,
            "every_initial_likelihood_recomputed_fresh": True,
        },
        "replicates": replicates,
        "particle_chains_per_replicate": particles_count,
        "retained_snapshots_per_chain": retained_count,
        "posterior_draws": len(pooled),
        "controls": {
            "diagnostic_validation": diagnostic_control,
            "uniform_target_invariance": uniform_control,
            "orbit_validation": orbit_validation,
        },
        "orbit_validation_passed": orbit_validation_passed,
        "forward_call_accounting": accounting,
        "maximum_rank_normalized_split_rhat": (
            float(np.max(diagnostics["rhat"])) if finite_rhat else None
        ),
        "minimum_bulk_effective_samples": float(np.min(diagnostics["bulk_ess"])),
        "minimum_tail_effective_samples": float(np.min(diagnostics["tail_ess"])),
        "maximum_standardized_between_replicate_median_spread": float(np.max(median_spread)),
        "maximum_descriptive_sobol_start_to_posterior_median_shift": float(
            np.max(descriptive_start_shift)
        ),
        "descriptive_sobol_start_to_posterior_shift_is_a_gate": False,
        "all_coordinates_positive_variance": bool(np.all(diagnostics["valid"])),
        "all_coordinate_gates_passed": bool(np.all(coordinate_pass)),
        "all_mechanics_gates_passed": bool(mechanics_pass),
        "production_passed": production_passed,
        "parameters": [
            {
                "coordinate": name,
                "diagnostic_valid_positive_variance": bool(diagnostics["valid"][index]),
                "minimum_scaled_within_chain_variance": float(
                    diagnostics["minimum_scaled_within_chain_variance"][index]
                ),
                "rank_normalized_split_rhat": (
                    float(diagnostics["rhat"][index])
                    if np.isfinite(diagnostics["rhat"][index])
                    else None
                ),
                "bulk_effective_samples": float(diagnostics["bulk_ess"][index]),
                "tail_effective_samples": float(diagnostics["tail_ess"][index]),
                "standardized_between_replicate_median_spread": float(median_spread[index]),
                "descriptive_sobol_start_to_posterior_median_shift": float(
                    descriptive_start_shift[index]
                ),
                "passed": bool(coordinate_pass[index]),
            }
            for index, name in enumerate(COMPOSITES)
        ],
        "mechanics": mechanics_rows,
        "replicate_summaries": replicate_summaries,
        "runtime_data_boundary": {
            "packet_sha256": contract["train_packet"]["file_sha256"],
            "allowed_split": "development_train",
            "rows_loaded": 80,
            "holdout_rows_loaded": 0,
            "confirmation_rows_loaded": 0,
            "independent_rows_loaded": 0,
            "canonical_comparator_packet_builder_called_during_sampling": False,
        },
        "downstream_sequence_if_candidate_passes": [
            "simulation_based_calibration",
            "matched_newtonian_control",
            "source_covariance",
        ],
        "claim_boundary": contract["adjudication"],
    }
    aggregate["publication_semantics"] = {
        "same_filesystem": True,
        "atomic_commit_primitive": "hard_link_complete_temp_to_absent_destination",
        "destination_replacement_allowed": False,
        "temporary_cleanup_required": True,
    }
    atomic_save_result(
        output,
        traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=aggregate,
    )
    return aggregate

def controls_receipt(contract: dict[str, Any]) -> dict[str, Any]:
    config = uncertainty.load_config(ROOT)
    diagnostic = diagnostic_validation_control(contract["diagnostic_validation"])
    uniform = uniform_target_invariance_control(
        contract["uniform_target_invariance_control"], config
    )
    return {
        "schema_version": "invariant-gravity-cluster-nuisance-quotient-sampler-controls-1.0",
        "execution_contract_sha256": contract["_execution_contract_sha256"],
        "implementation_source_normalized_sha256": contract[
            "implementation_source_normalized_sha256"
        ],
        "train_packet_sha256": contract["train_packet"]["file_sha256"],
        "sobol_start_population_sha256": contract["sobol_start_population"][
            "file_sha256"
        ],
        "diagnostic_validation": diagnostic,
        "uniform_target_invariance": uniform,
        "forward_evaluations": 0,
        "artifact_publication": {
            "atomic_same_filesystem_no_clobber": True,
            "destination_replacement_allowed": False,
        },
        "passed": bool(diagnostic["passed"] and uniform["passed"]),
    }


def _concurrent_creator_race(kind: str) -> dict[str, Any]:
    creator_bytes = f"CANONICAL-{kind}-CONCURRENT-CREATOR-WON".encode("ascii")
    ready_for_creator = threading.Event()
    creator_finished = threading.Event()
    creator_error: list[str] = []
    (ROOT / ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f"quotient-sampler-{kind}-race-", dir=ROOT / ARTIFACT_DIR
    ) as temporary_directory:
        directory = Path(temporary_directory)
        destination = directory / ("race.npz" if kind == "npz" else "race.json")

        def creator() -> None:
            try:
                if not ready_for_creator.wait(timeout=10.0):
                    raise RuntimeError("publisher never reached the finalization barrier")
                with destination.open("xb") as handle:
                    handle.write(creator_bytes)
                    handle.flush()
                    os.fsync(handle.fileno())
            except (OSError, RuntimeError) as error:
                creator_error.append(type(error).__name__)
            finally:
                creator_finished.set()

        creator_thread = threading.Thread(target=creator, daemon=True)
        creator_thread.start()

        def before_link() -> None:
            ready_for_creator.set()
            if not creator_finished.wait(timeout=10.0):
                raise RuntimeError("concurrent creator did not finish")

        publication_rejected = False
        try:
            if kind == "npz":
                atomic_save_result(
                    destination,
                    traces=np.zeros((1, 1, 4, 1)),
                    ending_particles=np.zeros((1, 1, 17)),
                    ending_log_likelihood=np.zeros((1, 1)),
                    summary={"schema_version": "canonical-race-control"},
                    before_link=before_link,
                )
            elif kind == "json":
                write_json(
                    destination,
                    {"schema_version": "canonical-race-control"},
                    before_link=before_link,
                )
            else:
                raise RuntimeError(f"unknown race-control kind: {kind}")
        except RuntimeError as error:
            publication_rejected = "no-clobber" in str(error)
        creator_thread.join(timeout=10.0)
        target_preserved = destination.read_bytes() == creator_bytes
        temporary_files = list(directory.glob(".quotient-sampler-complete-*"))
        result = {
            "kind": kind,
            "passed": bool(
                publication_rejected
                and not creator_error
                and target_preserved
                and not temporary_files
            ),
            "creator_ran_after_complete_temp_before_final_link": True,
            "publication_rejected_existing_destination": publication_rejected,
            "concurrent_creator_target_bytes_preserved": target_preserved,
            "temporary_files_remaining": len(temporary_files),
            "creator_errors": creator_error,
        }
    result["temporary_directory_removed"] = not directory.exists()
    result["passed"] = bool(result["passed"] and result["temporary_directory_removed"])
    return result


def write_race_controls() -> dict[str, Any]:
    npz_control = _concurrent_creator_race("npz")
    json_control = _concurrent_creator_race("json")
    return {
        "schema_version": WRITE_RACE_CONTROLS_SCHEMA,
        "passed": bool(npz_control["passed"] and json_control["passed"]),
        "platform": sys.platform,
        "commit_primitive": "os.link_complete_same_directory_temp_to_absent_destination",
        "replacement_fallback": None,
        "npz_concurrent_creator": npz_control,
        "json_concurrent_creator": json_control,
    }


def validate_bound_controls_and_smoke(
    contract_hash: str, controls_path: Path, smoke_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    controls = json.loads(confined(controls_path).read_text(encoding="utf-8"))
    smoke = np.load(confined(smoke_path), allow_pickle=False)
    strict_keys(
        {name: None for name in smoke.files},
        {
            "composite_traces",
            "ending_particles",
            "ending_log_likelihood",
            "summary",
        },
        "smoke archive",
    )
    summary = json.loads(str(smoke["summary"].item()))
    if (
        controls.get("schema_version")
        != "invariant-gravity-cluster-nuisance-quotient-sampler-controls-1.0"
        or not controls.get("passed")
        or controls.get("execution_contract_sha256") != contract_hash
        or summary.get("execution_contract_sha256") != contract_hash
        or summary.get("mode") != "smoke"
        or summary.get("orbit_validation_passed") is not True
        or summary.get("sobol_start_population", {}).get(
            "every_initial_likelihood_recomputed_fresh"
        )
        is not True
    ):
        raise RuntimeError("controls or smoke do not satisfy the canonical contract")
    if summary.get("schema_version") != RESULT_SCHEMA:
        raise RuntimeError("smoke receipt schema changed")
    publication = summary.get("publication_semantics", {})
    if publication != {
        "same_filesystem": True,
        "atomic_commit_primitive": "hard_link_complete_temp_to_absent_destination",
        "destination_replacement_allowed": False,
        "temporary_cleanup_required": True,
    }:
        raise RuntimeError("smoke publication semantics changed")
    return controls, summary


def frozen_artifact_bindings(
    contract: dict[str, Any], contract_path: Path, controls_path: Path, smoke_path: Path
) -> dict[str, dict[str, str]]:
    return {
        "contract": artifact_binding(contract_path),
        "implementation_source": artifact_binding(
            ROOT / contract["implementation_source"]
        ),
        "train_packet": artifact_binding(ROOT / contract["train_packet"]["path"]),
        "sobol_start_population": artifact_binding(
            ROOT / contract["sobol_start_population"]["path"]
        ),
        "controls": artifact_binding(controls_path),
        "smoke": artifact_binding(smoke_path),
    }


def write_unauthorized_manifest(
    contract_path: Path,
    expected_contract_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    output: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path, expected_contract_sha256)
    validate_bound_controls_and_smoke(expected_contract_sha256, controls_path, smoke_path)
    body = {
        "schema_version": UNAUTHORIZED_SCHEMA,
        "status": "external_approval_required_controls_and_smoke_bound",
        "artifact_bindings": frozen_artifact_bindings(
            contract, contract_path, controls_path, smoke_path
        ),
        "production_authorization": {
            "authorized": False,
            "approved_by": None,
            "approval_id": None,
            "maximum_forward_evaluations": 0,
        },
        "claim_boundary": {
            "controls_passed": True,
            "bounded_smoke_executed": True,
            "candidate_production_executed": False,
            "candidate_claim_allowed": False,
            "production_execution_allowed": False,
            "external_approval_required": True,
            "newtonian_control_unlocked": False,
            "simulation_based_calibration_unlocked": False,
        },
    }
    write_json(output, body)
    return body


def validate_frozen_artifact_bindings(bindings: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        bindings,
        {
            "contract",
            "implementation_source",
            "train_packet",
            "sobol_start_population",
            "controls",
            "smoke",
        },
        "authorization.artifact_bindings",
    )
    paths = {
        name: validate_artifact_binding(row, f"authorization.artifact_bindings.{name}")
        for name, row in bindings.items()
    }
    contract = load_contract(paths["contract"], bindings["contract"]["file_sha256"])
    if paths["implementation_source"] != Path(__file__).resolve():
        raise RuntimeError("authorization does not bind this canonical executable")
    validate_bound_controls_and_smoke(
        contract["_execution_contract_sha256"], paths["controls"], paths["smoke"]
    )
    if bindings["train_packet"] != contract["train_packet"]:
        raise RuntimeError("authorization train packet differs from contract")
    if bindings["sobol_start_population"] != contract["sobol_start_population"]:
        raise RuntimeError("authorization Sobol population differs from contract")
    return contract


def validate_unauthorized_body(body: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "production_authorization",
            "claim_boundary",
        },
        "unauthorized manifest",
    )
    if body["schema_version"] != UNAUTHORIZED_SCHEMA:
        raise RuntimeError("unauthorized manifest schema changed")
    if body["status"] != "external_approval_required_controls_and_smoke_bound":
        raise RuntimeError("unauthorized manifest status changed")
    if body["production_authorization"] != {
        "authorized": False,
        "approved_by": None,
        "approval_id": None,
        "maximum_forward_evaluations": 0,
    }:
        raise RuntimeError("unauthorized production fields changed")
    if body["claim_boundary"] != {
        "controls_passed": True,
        "bounded_smoke_executed": True,
        "candidate_production_executed": False,
        "candidate_claim_allowed": False,
        "production_execution_allowed": False,
        "external_approval_required": True,
        "newtonian_control_unlocked": False,
        "simulation_based_calibration_unlocked": False,
    }:
        raise RuntimeError("unauthorized claim boundary changed")
    return validate_frozen_artifact_bindings(body["artifact_bindings"])


def validate_external_approval(
    path: Path, expected_sha256: str, exact_bindings: dict[str, Any]
) -> dict[str, Any]:
    approval_path = confined(path)
    if not approval_path.is_file() or file_sha256(approval_path) != expected_sha256:
        raise RuntimeError("external approval record missing or tampered")
    body = json.loads(approval_path.read_text(encoding="utf-8"))
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
            "artifact_bindings",
        },
        "external approval record",
    )
    if body["schema_version"] != APPROVAL_SCHEMA:
        raise RuntimeError("external approval schema changed")
    if body["status"] != "explicit_external_production_approval":
        raise RuntimeError("external approval status changed")
    if body["approved_by"] != "Henry":
        raise RuntimeError("external approval must be approved_by Henry")
    if not isinstance(body["approval_id"], str) or not body["approval_id"].strip():
        raise RuntimeError("external approval_id must be nonempty")
    if int(body["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS:
        raise RuntimeError("external approval maximum call count changed")
    if body["artifact_bindings"] != exact_bindings:
        raise RuntimeError("external approval does not bind exact frozen artifacts")
    return body


def promote_authorization(
    unauthorized_path: Path,
    expected_unauthorized_sha256: str,
    approval_path: Path,
    expected_approval_sha256: str,
    output: Path,
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("unauthorized manifest hash differs from expected hash")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    approval = validate_external_approval(
        approval_path, expected_approval_sha256, unauthorized["artifact_bindings"]
    )
    body = {
        "schema_version": AUTHORIZED_SCHEMA,
        "status": "production_explicitly_authorized_by_external_approval",
        "artifact_bindings": unauthorized["artifact_bindings"],
        "external_approval_binding": {
            **artifact_binding(approval_path),
            "approval_id": approval["approval_id"],
        },
        "production_authorization": {
            "authorized": True,
            "approved_by": "Henry",
            "approval_id": approval["approval_id"],
            "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        },
        "authorized_execution_boundary": {
            "production_execution_allowed": True,
            "external_approval_satisfied": True,
            "candidate_result_exists": False,
            "candidate_claim_allowed_before_result": False,
            "production_result_must_retain_failed_gates": True,
            "simulation_based_calibration_follows_candidate_pass": True,
            "matched_newtonian_control_follows_sbc": True,
            "source_covariance_follows_newtonian": True,
        },
    }
    write_json(output, body)
    return body


def validate_authorized_body(body: dict[str, Any]) -> dict[str, Any]:
    strict_keys(
        body,
        {
            "schema_version",
            "status",
            "artifact_bindings",
            "external_approval_binding",
            "production_authorization",
            "authorized_execution_boundary",
        },
        "authorized manifest",
    )
    if body["schema_version"] != AUTHORIZED_SCHEMA:
        raise RuntimeError("authorized manifest schema changed")
    if body["status"] != "production_explicitly_authorized_by_external_approval":
        raise RuntimeError("authorized manifest status changed")
    production = body["production_authorization"]
    strict_keys(
        production,
        {
            "authorized",
            "approved_by",
            "approval_id",
            "maximum_forward_evaluations",
        },
        "authorized manifest.production_authorization",
    )
    if (
        production["authorized"] is not True
        or production["approved_by"] != "Henry"
        or not isinstance(production["approval_id"], str)
        or not production["approval_id"].strip()
        or int(production["maximum_forward_evaluations"]) != MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
    ):
        raise RuntimeError("authorized production fields are incomplete")
    expected_boundary = {
        "production_execution_allowed": True,
        "external_approval_satisfied": True,
        "candidate_result_exists": False,
        "candidate_claim_allowed_before_result": False,
        "production_result_must_retain_failed_gates": True,
        "simulation_based_calibration_follows_candidate_pass": True,
        "matched_newtonian_control_follows_sbc": True,
        "source_covariance_follows_newtonian": True,
    }
    if body["authorized_execution_boundary"] != expected_boundary:
        raise RuntimeError("authorized execution boundary changed")
    contract = validate_frozen_artifact_bindings(body["artifact_bindings"])
    approval_binding = body["external_approval_binding"]
    strict_keys(
        approval_binding,
        {"path", "file_sha256", "approval_id"},
        "authorized manifest.external_approval_binding",
    )
    approval = validate_external_approval(
        ROOT / approval_binding["path"],
        approval_binding["file_sha256"],
        body["artifact_bindings"],
    )
    if (
        approval["approval_id"] != approval_binding["approval_id"]
        or approval["approval_id"] != production["approval_id"]
    ):
        raise RuntimeError("authorized approval identifiers disagree")
    return contract


def validate_authorization(
    path: Path, expected_sha256: str, *, require_production: bool
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    authorization_path = confined(path)
    if not authorization_path.is_file() or file_sha256(authorization_path) != expected_sha256:
        raise RuntimeError("authorization manifest hash differs from expected hash")
    body = json.loads(authorization_path.read_text(encoding="utf-8"))
    schema = body.get("schema_version")
    if schema == UNAUTHORIZED_SCHEMA:
        if require_production:
            raise RuntimeError(
                "external approval is required; refusal occurs before contract or packet load"
            )
        return body, validate_unauthorized_body(body)
    if schema == AUTHORIZED_SCHEMA:
        return body, validate_authorized_body(body)
    raise RuntimeError("authorization manifest has no recognized canonical schema")


def injected_approval_body(bindings: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    body = {
        "schema_version": APPROVAL_SCHEMA,
        "status": "explicit_external_production_approval",
        "approved_by": "Henry",
        "approval_id": "CANONICAL-AUTHORIZATION-TRANSITION-CONTROL-ONLY",
        "maximum_forward_evaluations": MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS,
        "artifact_bindings": bindings,
    }
    body.update(overrides)
    return body


def _authorization_transition_once(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    unauthorized_target = confined(unauthorized_path)
    if file_sha256(unauthorized_target) != expected_unauthorized_sha256:
        raise RuntimeError("authorization-control input hash changed")
    unauthorized = json.loads(unauthorized_target.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    negative_results: dict[str, bool] = {}
    temporary_path: Path | None = None
    (ROOT / ARTIFACT_DIR).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="quotient-sampler-authorization-control-", dir=ROOT / ARTIFACT_DIR
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)

        def expect_rejection(name: str, approval: dict[str, Any]) -> None:
            approval_path = temporary_path / f"{name}-approval.json"
            output_path = temporary_path / f"{name}-authorized.json"
            write_json(approval_path, approval)
            try:
                promote_authorization(
                    unauthorized_target,
                    expected_unauthorized_sha256,
                    approval_path,
                    file_sha256(approval_path),
                    output_path,
                )
            except RuntimeError:
                negative_results[name] = not output_path.exists()
                return
            negative_results[name] = False

        bindings = unauthorized["artifact_bindings"]
        expect_rejection(
            "wrong_approver", injected_approval_body(bindings, approved_by="Not Henry")
        )
        expect_rejection("empty_approval_id", injected_approval_body(bindings, approval_id=""))
        expect_rejection(
            "wrong_maximum_calls",
            injected_approval_body(
                bindings,
                maximum_forward_evaluations=MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS - 1,
            ),
        )
        tampered_bindings = json.loads(json.dumps(bindings))
        tampered_bindings["smoke"]["file_sha256"] = "0" * 64
        expect_rejection("wrong_artifact_binding", injected_approval_body(tampered_bindings))

        approval_path = temporary_path / "positive-control-external-approval.json"
        authorized_path = temporary_path / "positive-control-authorized.json"
        write_json(approval_path, injected_approval_body(bindings))
        promote_authorization(
            unauthorized_target,
            expected_unauthorized_sha256,
            approval_path,
            file_sha256(approval_path),
            authorized_path,
        )
        validated, contract = validate_authorization(
            authorized_path,
            file_sha256(authorized_path),
            require_production=True,
        )
        positive_passed = bool(
            validated["schema_version"] == AUTHORIZED_SCHEMA
            and validated["status"] == "production_explicitly_authorized_by_external_approval"
            and validated["production_authorization"]["authorized"] is True
            and contract is not None
        )
    temporary_removed = bool(temporary_path is not None and not temporary_path.exists())
    return {
        "negative_controls": negative_results,
        "positive_disposable_authorized_control": {
            "passed": positive_passed,
            "authorized_schema": AUTHORIZED_SCHEMA,
            "authorized_status": ("production_explicitly_authorized_by_external_approval"),
            "approval_logical_id": (
                "CANONICAL-AUTHORIZATION-TRANSITION-CONTROL-ONLY"
            ),
            "approved_by": "Henry",
            "maximum_forward_evaluations": (MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS),
            "frozen_artifact_binding_count": 6,
            "manifest_disposable": True,
            "production_launched": False,
        },
        "temporary_artifacts_removed": temporary_removed,
        "production_runs": 0,
    }


def authorization_transition_controls(
    unauthorized_path: Path, expected_unauthorized_sha256: str
) -> dict[str, Any]:
    first = _authorization_transition_once(unauthorized_path, expected_unauthorized_sha256)
    second = _authorization_transition_once(unauthorized_path, expected_unauthorized_sha256)
    replay_equal = first == second
    passed = bool(
        all(first["negative_controls"].values())
        and first["positive_disposable_authorized_control"]["passed"]
        and first["temporary_artifacts_removed"]
        and first["production_runs"] == 0
        and replay_equal
    )
    return {
        "schema_version": AUTHORIZATION_CONTROLS_SCHEMA,
        "passed": passed,
        **first,
        "exact_replay_equality": replay_equal,
        "volatile_physical_temp_paths_or_hashes_in_receipt": False,
    }


def write_implementation_receipt(
    config_path: Path,
    expected_config_sha256: str,
    controls_path: Path,
    smoke_path: Path,
    race_controls_path: Path,
    authorization_controls_path: Path,
    unauthorized_manifest_path: Path,
    output: Path,
) -> dict[str, Any]:
    config = load_contract(config_path, expected_config_sha256)
    _controls, smoke = validate_bound_controls_and_smoke(
        expected_config_sha256, controls_path, smoke_path
    )
    race = json.loads(confined(race_controls_path).read_text(encoding="utf-8"))
    if race.get("schema_version") != WRITE_RACE_CONTROLS_SCHEMA or not race.get("passed"):
        raise RuntimeError("atomic no-clobber race controls are not valid and passing")
    authorization_controls = json.loads(
        confined(authorization_controls_path).read_text(encoding="utf-8")
    )
    if (
        authorization_controls.get("schema_version") != AUTHORIZATION_CONTROLS_SCHEMA
        or not authorization_controls.get("passed")
        or authorization_controls.get("production_runs") != 0
        or authorization_controls.get("exact_replay_equality") is not True
    ):
        raise RuntimeError("authorization transition controls are not valid and passing")
    unauthorized_path = confined(unauthorized_manifest_path)
    unauthorized = json.loads(unauthorized_path.read_text(encoding="utf-8"))
    validate_unauthorized_body(unauthorized)
    if unauthorized["production_authorization"]["authorized"] is not False:
        raise RuntimeError("current authorization manifest is not unauthorized")
    if smoke["forward_call_accounting"]["total_forward_evaluations"] != 852:
        raise RuntimeError("bounded smoke call count changed")
    evidence = {
        "config": artifact_binding(config_path),
        "implementation_source": artifact_binding(ROOT / config["implementation_source"]),
        "quotient_predecessor": config["source_bindings"]["quotient_receipt"],
        "train_only_packet": config["train_packet"],
        "sobol_start_population": config["sobol_start_population"],
        "bounded_controls": artifact_binding(controls_path),
        "bounded_smoke": artifact_binding(smoke_path),
        "atomic_no_clobber_controls": artifact_binding(race_controls_path),
        "authorization_transition_controls": artifact_binding(
            authorization_controls_path
        ),
        "current_unauthorized_manifest": artifact_binding(unauthorized_path),
    }
    command_prefix = (
        "python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_sampler"
    )
    body = {
        "schema_version": IMPLEMENTATION_RECEIPT_SCHEMA,
        "status": "canonical_bounded_controls_and_smoke_only_external_approval_required",
        "decision": (
            "CANONICAL_NUISANCE_QUOTIENT_SAMPLER_IMPLEMENTED_CONTROLS_PASS_"
            "PRODUCTION_UNAUTHORIZED"
        ),
        "evidence": evidence,
        "canonicalization": {
            "self_contained_canonical_implementation": True,
            "prototype_generation_2_current_execution_basis": False,
            "prototype_generation_5_current_execution_basis": False,
            "historical_prototype_ledgers_retained": True,
            "canonical_source_or_artifact_binds_development_workspace": False,
        },
        "frozen_mechanics": {
            "primitive_priors": 17,
            "quotient_coordinates": 10,
            "train_only_clusters": 8,
            "train_only_rows": 80,
            "independent_sobol_scrambles": 4,
            "sobol_particles_per_scramble": 512,
            "orbit_moves": list(ORBIT_NAMES),
            "out_of_bounds_policy": "whole_correlated_proposal_rejected_as_self_loop",
            "bounded_smoke_forward_evaluations": 852,
            "maximum_production_forward_evaluations": (
                MAXIMUM_PRODUCTION_FORWARD_EVALUATIONS
            ),
        },
        "authorization_and_execution": {
            "production_authorized": False,
            "authorized_manifests_present": False,
            "production_launches": 0,
            "external_approval_required": True,
        },
        "publication_readiness": {
            "completed_tasks": 59,
            "open_tasks": 63,
            "total_tasks": 122,
            "CP5_status": "PARTIAL",
            "CP5_7_through_CP5_10": "OPEN",
            "implementation_evidence_only": True,
            "scientific_claims_added": False,
            "candidate_production_claim": False,
        },
        "replay_commands": {
            "check": (
                f"{command_prefix} check --config {config_path.as_posix()} "
                f"--expected-config-sha256 {expected_config_sha256} "
                f"--implementation-receipt {output.as_posix()}"
            ),
            "bounded_controls": (
                f"{command_prefix} controls --contract {config_path.as_posix()} "
                f"--expected-contract-sha256 {expected_config_sha256} "
                "--output <new-absent-path>.json"
            ),
            "bounded_smoke": (
                f"{command_prefix} smoke --contract {config_path.as_posix()} "
                f"--expected-contract-sha256 {expected_config_sha256} "
                "--output <new-absent-path>.npz"
            ),
            "production": "NOT_AUTHORIZED_AND_NOT_RUN",
        },
    }
    body["content_sha256"] = receipt_content_hash(body)
    write_json(output, body)
    return body


def check_canonical_implementation(
    config_path: Path, expected_config_sha256: str, implementation_receipt: Path
) -> dict[str, Any]:
    config = load_contract(config_path, expected_config_sha256)
    receipt_path = confined(implementation_receipt)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    strict_keys(
        receipt,
        {
            "schema_version",
            "status",
            "decision",
            "evidence",
            "canonicalization",
            "frozen_mechanics",
            "authorization_and_execution",
            "publication_readiness",
            "replay_commands",
            "content_sha256",
        },
        "implementation receipt",
    )
    if receipt["schema_version"] != IMPLEMENTATION_RECEIPT_SCHEMA:
        raise RuntimeError("implementation receipt schema changed")
    unhashed = dict(receipt)
    observed_content_hash = unhashed.pop("content_sha256")
    if observed_content_hash != receipt_content_hash(unhashed):
        raise RuntimeError("implementation receipt content hash changed")
    for name, row in receipt["evidence"].items():
        validate_artifact_binding(row, f"implementation receipt.evidence.{name}")
    if receipt["evidence"]["config"] != artifact_binding(config_path):
        raise RuntimeError("implementation receipt binds another config")
    if receipt["evidence"]["implementation_source"] != artifact_binding(
        ROOT / config["implementation_source"]
    ):
        raise RuntimeError("implementation receipt binds another executable")
    if receipt["evidence"]["train_only_packet"] != config["train_packet"]:
        raise RuntimeError("implementation receipt train packet changed")
    if receipt["evidence"]["sobol_start_population"] != config["sobol_start_population"]:
        raise RuntimeError("implementation receipt Sobol population changed")
    authorization = receipt["authorization_and_execution"]
    publication = receipt["publication_readiness"]
    if authorization != {
        "production_authorized": False,
        "authorized_manifests_present": False,
        "production_launches": 0,
        "external_approval_required": True,
    }:
        raise RuntimeError("implementation receipt authorization boundary changed")
    if publication != {
        "completed_tasks": 59,
        "open_tasks": 63,
        "total_tasks": 122,
        "CP5_status": "PARTIAL",
        "CP5_7_through_CP5_10": "OPEN",
        "implementation_evidence_only": True,
        "scientific_claims_added": False,
        "candidate_production_claim": False,
    }:
        raise RuntimeError("implementation receipt readiness boundary changed")
    return {
        "valid": True,
        "config_sha256": expected_config_sha256,
        "implementation_receipt_sha256": file_sha256(receipt_path),
        "production_authorized": False,
        "production_launches": 0,
        "completed_tasks": 59,
        "open_tasks": 63,
        "total_tasks": 122,
        "CP5_status": "PARTIAL",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("controls", "smoke"):
        command = subparsers.add_parser(name)
        command.add_argument("--contract", type=Path, required=True)
        command.add_argument("--expected-contract-sha256", required=True)
        command.add_argument("--output", type=Path, required=True)

    packet_command = subparsers.add_parser("build-train-packet")
    packet_command.add_argument("--input-train-packet", type=Path, required=True)
    packet_command.add_argument("--output", type=Path, required=True)

    starts_command = subparsers.add_parser("build-sobol-starts")
    starts_command.add_argument("--output", type=Path, required=True)

    race_command = subparsers.add_parser("write-race-controls")
    race_command.add_argument("--output", type=Path, required=True)

    unauthorized_command = subparsers.add_parser("write-unauthorized")
    unauthorized_command.add_argument("--contract", type=Path, required=True)
    unauthorized_command.add_argument("--expected-contract-sha256", required=True)
    unauthorized_command.add_argument("--controls", type=Path, required=True)
    unauthorized_command.add_argument("--smoke", type=Path, required=True)
    unauthorized_command.add_argument("--output", type=Path, required=True)

    promote_command = subparsers.add_parser("promote-authorization")
    promote_command.add_argument("--unauthorized", type=Path, required=True)
    promote_command.add_argument("--expected-unauthorized-sha256", required=True)
    promote_command.add_argument("--external-approval", type=Path, required=True)
    promote_command.add_argument("--expected-external-approval-sha256", required=True)
    promote_command.add_argument("--output", type=Path, required=True)

    validate_command = subparsers.add_parser("validate-authorization")
    validate_command.add_argument("--authorization", type=Path, required=True)
    validate_command.add_argument("--expected-authorization-sha256", required=True)
    validate_command.add_argument("--require-production", action="store_true")

    transition_command = subparsers.add_parser("authorization-controls")
    transition_command.add_argument("--unauthorized", type=Path, required=True)
    transition_command.add_argument("--expected-unauthorized-sha256", required=True)
    transition_command.add_argument("--output", type=Path, required=True)

    receipt_command = subparsers.add_parser("write-implementation-receipt")
    receipt_command.add_argument("--config", type=Path, required=True)
    receipt_command.add_argument("--expected-config-sha256", required=True)
    receipt_command.add_argument("--controls", type=Path, required=True)
    receipt_command.add_argument("--smoke", type=Path, required=True)
    receipt_command.add_argument("--race-controls", type=Path, required=True)
    receipt_command.add_argument("--authorization-controls", type=Path, required=True)
    receipt_command.add_argument("--unauthorized-manifest", type=Path, required=True)
    receipt_command.add_argument("--output", type=Path, required=True)

    check_command = subparsers.add_parser("check")
    check_command.add_argument("--config", type=Path, required=True)
    check_command.add_argument("--expected-config-sha256", required=True)
    check_command.add_argument("--implementation-receipt", type=Path, required=True)

    run_command = subparsers.add_parser("run")
    run_command.add_argument("--authorization", type=Path, required=True)
    run_command.add_argument("--expected-authorization-sha256", required=True)
    run_command.add_argument("--output", type=Path, required=True)
    run_command.add_argument(
        "--execute-frozen-production-nuisance-quotient-sampler", action="store_true"
    )

    args = parser.parse_args()
    if args.command == "build-train-packet":
        result = build_train_packet_from_train_only_input(
            args.input_train_packet, args.output
        )
        print(
            json.dumps(
                {
                    "cluster_count": result["cluster_count"],
                    "row_count": result["row_count"],
                    "content_sha256": result["content_sha256"],
                    "file_sha256": file_sha256(args.output),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "build-sobol-starts":
        print(json.dumps(build_sobol_starts(args.output), sort_keys=True))
        return
    if args.command in {"controls", "smoke"}:
        contract = load_contract(args.contract, args.expected_contract_sha256)
        if args.command == "controls":
            result = controls_receipt(contract)
            write_json(args.output, result)
            print(json.dumps(result, sort_keys=True))
            if not result["passed"]:
                raise SystemExit(2)
            return
        result = run_sampler(contract, contract["smoke_settings"], args.output, smoke=True)
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "write-race-controls":
        result = write_race_controls()
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.command == "write-unauthorized":
        result = write_unauthorized_manifest(
            args.contract,
            args.expected_contract_sha256,
            args.controls,
            args.smoke,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "promote-authorization":
        result = promote_authorization(
            args.unauthorized,
            args.expected_unauthorized_sha256,
            args.external_approval,
            args.expected_external_approval_sha256,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "validate-authorization":
        body, contract = validate_authorization(
            args.authorization,
            args.expected_authorization_sha256,
            require_production=args.require_production,
        )
        print(
            json.dumps(
                {
                    "valid": True,
                    "schema_version": body["schema_version"],
                    "status": body["status"],
                    "production_authorized": body["production_authorization"]["authorized"],
                    "execution_contract_sha256": (
                        contract["_execution_contract_sha256"] if contract is not None else None
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.command == "authorization-controls":
        result = authorization_transition_controls(
            args.unauthorized, args.expected_unauthorized_sha256
        )
        write_json(args.output, result)
        print(json.dumps(result, sort_keys=True))
        if not result["passed"]:
            raise SystemExit(2)
        return
    if args.command == "write-implementation-receipt":
        result = write_implementation_receipt(
            args.config,
            args.expected_config_sha256,
            args.controls,
            args.smoke,
            args.race_controls,
            args.authorization_controls,
            args.unauthorized_manifest,
            args.output,
        )
        print(json.dumps(result, sort_keys=True))
        return
    if args.command == "check":
        print(
            json.dumps(
                check_canonical_implementation(
                    args.config,
                    args.expected_config_sha256,
                    args.implementation_receipt,
                ),
                sort_keys=True,
            )
        )
        return
    if not args.execute_frozen_production_nuisance_quotient_sampler:
        raise RuntimeError(
            "production requires the explicit canonical production sentinel"
        )
    _authorization, contract = validate_authorization(
        args.authorization,
        args.expected_authorization_sha256,
        require_production=True,
    )
    if contract is None:
        raise RuntimeError("authorized contract was not loaded")
    result = run_sampler(contract, contract["production_settings"], args.output, smoke=False)
    print(json.dumps(result, sort_keys=True))
    if not result["production_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
