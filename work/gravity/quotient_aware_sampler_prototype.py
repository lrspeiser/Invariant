from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reflect_unit(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(values, 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def prior_bounds(config: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    return lows, highs


def physical_values(unit: np.ndarray, config: dict[str, object]) -> np.ndarray:
    lows, highs = prior_bounds(config)
    return lows + unit * (highs - lows)


def unit_values(physical: np.ndarray, config: dict[str, object]) -> np.ndarray:
    lows, highs = prior_bounds(config)
    return (physical - lows) / (highs - lows)


def composite_values(unit: np.ndarray, config: dict[str, object]) -> np.ndarray:
    shape = unit.shape
    values = physical_values(unit.reshape(-1, shape[-1]), config)
    by_name = {
        name: values[:, index]
        for index, name in enumerate(uncertainty.PARAMETERS)
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


def orbit_moves(
    particles: np.ndarray,
    rng: np.random.Generator,
    config: dict[str, object],
    stellar_log_step: float,
    geometry_log_step: float,
    coupled_log_step: float,
) -> dict[str, int]:
    physical = physical_values(particles, config)
    low, high = prior_bounds(config)
    factor_low = np.asarray([0.75, 0.75, 1.0, 1.0, 0.7, 0.8])
    factor_high = np.asarray([1.25, 1.25, 1.2, 1.3, 1.3, 1.2])
    counts = {
        "stellar_attempted": len(particles),
        "stellar_accepted": 0,
        "geometry_attempted": len(particles),
        "geometry_accepted": 0,
        "coupled_attempted": len(particles),
        "coupled_accepted": 0,
    }
    for index, row in enumerate(physical):
        factors = stellar_factors(row)
        partner = int(rng.integers(1, len(factors)))
        delta = float(rng.normal(scale=stellar_log_step))
        proposed_factors = factors.copy()
        proposed_factors[0] *= math.exp(-delta)
        proposed_factors[partner] *= math.exp(delta)
        if within(proposed_factors, factor_low, factor_high):
            set_stellar_factors(row, proposed_factors)
            counts["stellar_accepted"] += 1

        radius_factors = np.asarray([1.0 + row[13], row[15]])
        delta = float(rng.normal(scale=geometry_log_step))
        proposed_radius_factors = radius_factors * np.asarray(
            [math.exp(delta), math.exp(-delta)]
        )
        if within(proposed_radius_factors, np.asarray([0.95, 0.85]), np.asarray([1.05, 1.15])):
            row[13] = proposed_radius_factors[0] - 1.0
            row[15] = proposed_radius_factors[1]
            counts["geometry_accepted"] += 1

        factors = stellar_factors(row)
        raw_stellar = float(np.prod(factors))
        delta = float(rng.normal(scale=coupled_log_step))
        proposed = row.copy()
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
            and math.log(max(rng.random(), np.finfo(float).tiny)) < min(0.0, delta)
        ):
            row[:] = proposed
            counts["coupled_accepted"] += 1
    particles[:] = unit_values(physical, config)
    if np.any(particles < -1e-12) or np.any(particles > 1.0 + 1e-12):
        raise RuntimeError("orbit move escaped primitive prior")
    particles[:] = np.clip(particles, 0.0, 1.0)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--burn-sweeps", type=int, default=128)
    parser.add_argument("--retained-sweeps", type=int, default=512)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--covariance-refresh", type=int, default=8)
    parser.add_argument("--stellar-log-step", type=float, default=0.08)
    parser.add_argument("--geometry-log-step", type=float, default=0.03)
    parser.add_argument("--coupled-log-step", type=float, default=0.02)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256(args.input) != args.expected_input_sha256:
        raise RuntimeError("source SMC hash changed")
    if args.retained_sweeps % args.thin:
        raise RuntimeError("retained sweeps must be divisible by thinning")

    root = Path(__file__).resolve().parents[2]
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    source = np.load(args.input, allow_pickle=False)
    source_particles = np.asarray(source["particles"], dtype=float)
    source_log_likelihood = np.asarray(source["log_likelihood"], dtype=float)
    if source_particles.shape[0] < args.replicates or source_particles.shape[1] < args.particles:
        raise RuntimeError("source SMC does not contain requested population")
    source_particles = source_particles[: args.replicates, : args.particles].copy()
    source_log_likelihood = source_log_likelihood[
        : args.replicates, : args.particles
    ].copy()
    retained_count = args.retained_sweeps // args.thin
    traces = np.empty(
        (args.replicates, args.particles, retained_count, len(COMPOSITES))
    )
    ending_particles = np.empty_like(source_particles)
    ending_log_likelihood = np.empty_like(source_log_likelihood)
    summaries = []

    for replicate in range(args.replicates):
        particles = source_particles[replicate].copy()
        log_likelihood = source_log_likelihood[replicate].copy()
        rng = np.random.default_rng(597_000 + replicate)
        scale = 2.38 / math.sqrt(len(ACTIVE_INDICES))
        square_root = covariance_square_root(particles[:, ACTIVE_INDICES])
        retained_index = 0
        burn_acceptance = []
        retained_acceptance = []
        orbit_counts = {
            "stellar_attempted": 0,
            "stellar_accepted": 0,
            "geometry_attempted": 0,
            "geometry_accepted": 0,
            "coupled_attempted": 0,
            "coupled_accepted": 0,
        }
        orbit_validation_maximum_log_likelihood_difference = 0.0
        orbit_validation_evaluations = 0
        total_sweeps = args.burn_sweeps + args.retained_sweeps
        for sweep in range(total_sweeps):
            if sweep < args.burn_sweeps and sweep % args.covariance_refresh == 0:
                square_root = covariance_square_root(particles[:, ACTIVE_INDICES])
            counts = orbit_moves(
                particles,
                rng,
                config,
                args.stellar_log_step,
                args.geometry_log_step,
                args.coupled_log_step,
            )
            for key, value in counts.items():
                orbit_counts[key] += value
            if sweep == 0:
                for index in range(min(4, args.particles)):
                    validated_log_likelihood = uncertainty._evaluate_unit(
                        particles[index], packets, family, config, config59
                    )[0]
                    orbit_validation_evaluations += 1
                    difference = abs(
                        float(validated_log_likelihood - log_likelihood[index])
                    )
                    orbit_validation_maximum_log_likelihood_difference = max(
                        orbit_validation_maximum_log_likelihood_difference,
                        difference,
                    )
                if orbit_validation_maximum_log_likelihood_difference > 1e-10:
                    raise RuntimeError("orbit move changed the training likelihood")
            proposals = particles.copy()
            noise = rng.normal(size=(args.particles, len(ACTIVE_INDICES)))
            proposals[:, ACTIVE_INDICES] = reflect_unit(
                particles[:, ACTIVE_INDICES] + scale * (noise @ square_root.T)
            )
            proposal_log_likelihood = np.empty(args.particles)
            for index, proposal in enumerate(proposals):
                proposal_log_likelihood[index] = uncertainty._evaluate_unit(
                    proposal, packets, family, config, config59
                )[0]
            accepted = np.log(
                np.maximum(rng.random(args.particles), np.finfo(float).tiny)
            ) < np.minimum(0.0, proposal_log_likelihood - log_likelihood)
            particles[accepted] = proposals[accepted]
            log_likelihood[accepted] = proposal_log_likelihood[accepted]
            rate = float(np.mean(accepted))
            if sweep < args.burn_sweeps:
                burn_acceptance.append(rate)
                scale *= math.exp(0.5 * (rate - 0.234))
                scale = float(np.clip(scale, 0.02, 8.0))
            else:
                retained_acceptance.append(rate)
                retained_sweep = sweep - args.burn_sweeps + 1
                if retained_sweep % args.thin == 0:
                    traces[replicate, :, retained_index, :] = composite_values(
                        particles, config
                    )
                    retained_index += 1
            if (sweep + 1) % 8 == 0 or sweep + 1 == total_sweeps:
                print(
                    json.dumps(
                        {
                            "replicate": replicate,
                            "sweep": sweep + 1,
                            "phase": (
                                "burn" if sweep < args.burn_sweeps else "retained"
                            ),
                            "active_acceptance": rate,
                            "active_scale": scale,
                            "evaluations": (sweep + 1) * args.particles,
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
        ending_particles[replicate] = particles
        ending_log_likelihood[replicate] = log_likelihood
        summaries.append(
            {
                "replicate": replicate,
                "evaluations": total_sweeps * args.particles,
                "orbit_validation_evaluations": orbit_validation_evaluations,
                "orbit_validation_maximum_log_likelihood_difference": (
                    orbit_validation_maximum_log_likelihood_difference
                ),
                "mean_burn_active_acceptance": float(np.mean(burn_acceptance)),
                "mean_retained_active_acceptance": float(
                    np.mean(retained_acceptance)
                ),
                "final_active_scale": scale,
                "orbit_counts": orbit_counts,
            }
        )

    dimensions = len(COMPOSITES)
    chains = traces.reshape(
        args.replicates * args.particles, retained_count, dimensions
    )
    rhat, ess = uncertainty._rhat_and_ess(chains)
    pooled = traces.reshape(-1, dimensions)
    standard_deviation = np.std(pooled, axis=0, ddof=1)
    replicate_medians = np.median(traces, axis=(1, 2))
    median_spread = np.divide(
        np.ptp(replicate_medians, axis=0),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )
    source_composites = composite_values(source_particles, config).reshape(
        -1, dimensions
    )
    source_median = np.median(source_composites, axis=0)
    posterior_median = np.median(pooled, axis=0)
    median_shift = np.divide(
        np.abs(posterior_median - source_median),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )
    aggregate = {
        "family_id": family["family_id"],
        "source_smc_sha256": args.expected_input_sha256,
        "replicates": args.replicates,
        "particle_chains_per_replicate": args.particles,
        "retained_snapshots_per_chain": retained_count,
        "posterior_draws": len(pooled),
        "total_forward_evaluations": sum(
            int(row["evaluations"]) for row in summaries
        ),
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_replicate_median_spread": float(
            np.max(median_spread)
        ),
        "maximum_standardized_source_to_quotient_median_shift": float(
            np.max(median_shift)
        ),
        "parameters": [
            {
                "coordinate": name,
                "rhat": float(rhat[index]),
                "effective_samples": float(ess[index]),
                "standardized_between_replicate_median_spread": float(
                    median_spread[index]
                ),
                "standardized_source_to_quotient_median_shift": float(
                    median_shift[index]
                ),
            }
            for index, name in enumerate(COMPOSITES)
        ],
        "replicate_summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        composite_traces=traces,
        ending_particles=ending_particles,
        ending_log_likelihood=ending_log_likelihood,
        summary=np.asarray(json.dumps(aggregate, sort_keys=True)),
    )
    print(json.dumps({"aggregate": aggregate, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
