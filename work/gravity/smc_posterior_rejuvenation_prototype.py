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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reflect_unit(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(values, 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def covariance_square_root(particles: np.ndarray) -> np.ndarray:
    covariance = np.cov(particles, rowvar=False, ddof=1)
    covariance += np.eye(particles.shape[1]) * 1e-8
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-input-sha256", required=True)
    parser.add_argument("--burn-sweeps", type=int, default=64)
    parser.add_argument("--retained-sweeps", type=int, default=128)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    observed_hash = sha256(args.input)
    if observed_hash != args.expected_input_sha256:
        raise RuntimeError(
            f"source SMC hash mismatch: {observed_hash} != {args.expected_input_sha256}"
        )
    if args.retained_sweeps % args.thin != 0:
        raise RuntimeError("retained sweeps must be divisible by thinning interval")

    source = np.load(args.input, allow_pickle=False)
    initial_particles = np.asarray(source["particles"], dtype=float)
    initial_log_likelihood = np.asarray(source["log_likelihood"], dtype=float)
    source_summary = json.loads(str(source["summary"]))
    if source_summary["family_id"] != "cross_scale_boundary":
        raise RuntimeError("frozen prototype only admits cross_scale_boundary")
    if initial_particles.shape != (4, 512, len(uncertainty.PARAMETERS)):
        raise RuntimeError(f"unexpected source particle shape: {initial_particles.shape}")

    root = Path(__file__).resolve().parents[2]
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    if family["family_id"] != "cross_scale_boundary":
        raise RuntimeError("candidate family ordering changed after the source run")

    replicates, particle_count, dimensions = initial_particles.shape
    retained_count = args.retained_sweeps // args.thin
    traces = np.empty((replicates, particle_count, retained_count, dimensions))
    ending_log_likelihood = np.empty_like(initial_log_likelihood)
    summaries: list[dict[str, object]] = []
    total_sweeps = args.burn_sweeps + args.retained_sweeps

    for replicate in range(replicates):
        particles = initial_particles[replicate].copy()
        log_likelihood = initial_log_likelihood[replicate].copy()
        square_root = covariance_square_root(particles)
        scale = float(
            source_summary["replicate_summaries"][replicate]["final_proposal_scale"]
        )
        rng = np.random.default_rng(595_000 + replicate)
        burn_acceptance: list[float] = []
        retained_acceptance: list[float] = []
        trace_index = 0
        evaluations = 0

        for sweep in range(total_sweeps):
            noise = rng.normal(size=(particle_count, dimensions))
            proposals = reflect_unit(particles + scale * (noise @ square_root.T))
            proposal_log_likelihood = np.empty(particle_count, dtype=float)
            for index, proposal in enumerate(proposals):
                proposal_log_likelihood[index] = uncertainty._evaluate_unit(
                    proposal, packets, family, config, config59
                )[0]
            evaluations += particle_count
            log_acceptance = proposal_log_likelihood - log_likelihood
            accepted = np.log(
                np.maximum(rng.random(particle_count), np.finfo(float).tiny)
            ) < np.minimum(0.0, log_acceptance)
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
                    traces[replicate, :, trace_index, :] = particles
                    trace_index += 1

            print(
                json.dumps(
                    {
                        "replicate": replicate,
                        "sweep": sweep + 1,
                        "phase": "burn" if sweep < args.burn_sweeps else "retained",
                        "acceptance": rate,
                        "proposal_scale": scale,
                        "evaluations": evaluations,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

        ending_log_likelihood[replicate] = log_likelihood
        summaries.append(
            {
                "replicate": replicate,
                "evaluations": evaluations,
                "burn_sweeps": args.burn_sweeps,
                "retained_sweeps": args.retained_sweeps,
                "thinning_sweeps": args.thin,
                "retained_snapshots": retained_count,
                "final_proposal_scale": scale,
                "mean_burn_acceptance": float(np.mean(burn_acceptance)),
                "mean_retained_acceptance": float(np.mean(retained_acceptance)),
                "minimum_retained_sweep_acceptance": min(retained_acceptance),
                "maximum_retained_sweep_acceptance": max(retained_acceptance),
            }
        )

    diagnostic_chains = traces.reshape(
        replicates * particle_count, retained_count, dimensions
    )
    rhat, ess = uncertainty._rhat_and_ess(diagnostic_chains)
    pooled = traces.reshape(-1, dimensions)
    standard_deviation = np.std(pooled, axis=0, ddof=1)
    replicate_medians = np.median(traces, axis=(1, 2))
    standardized_median_spread = np.divide(
        np.ptp(replicate_medians, axis=0),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )
    source_median = np.median(initial_particles.reshape(-1, dimensions), axis=0)
    rejuvenated_median = np.median(pooled, axis=0)
    standardized_source_to_rejuvenated_median_shift = np.divide(
        np.abs(rejuvenated_median - source_median),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )

    aggregate = {
        "family_id": family["family_id"],
        "source_smc_sha256": observed_hash,
        "replicates": replicates,
        "particle_chains_per_replicate": particle_count,
        "retained_snapshots_per_particle_chain": retained_count,
        "posterior_draws": int(len(pooled)),
        "total_evaluations": sum(int(row["evaluations"]) for row in summaries),
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_replicate_median_spread": float(
            np.max(standardized_median_spread)
        ),
        "maximum_standardized_source_to_rejuvenated_median_shift": float(
            np.max(standardized_source_to_rejuvenated_median_shift)
        ),
        "parameters": [
            {
                "parameter": parameter,
                "rhat": float(rhat[index]),
                "effective_samples": float(ess[index]),
                "standardized_between_replicate_median_spread": float(
                    standardized_median_spread[index]
                ),
                "standardized_source_to_rejuvenated_median_shift": float(
                    standardized_source_to_rejuvenated_median_shift[index]
                ),
            }
            for index, parameter in enumerate(uncertainty.PARAMETERS)
        ],
        "replicate_summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        traces=traces,
        ending_log_likelihood=ending_log_likelihood,
        summary=np.asarray(json.dumps(aggregate, sort_keys=True)),
    )
    print(json.dumps({"aggregate": aggregate, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
