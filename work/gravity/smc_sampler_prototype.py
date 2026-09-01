from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59


def reflect_unit(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(values, 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def normalized_weights(log_weights: np.ndarray) -> np.ndarray:
    shifted = np.exp(log_weights - float(np.max(log_weights)))
    return shifted / float(np.sum(shifted))


def effective_samples(weights: np.ndarray) -> float:
    return float(1.0 / np.sum(weights**2))


def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    count = len(weights)
    positions = (rng.random() + np.arange(count)) / count
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    return np.searchsorted(cumulative, positions, side="right")


def next_temperature(beta: float, log_likelihood: np.ndarray, target: float) -> float:
    count = len(log_likelihood)

    def conditional_ess(candidate: float) -> float:
        weights = normalized_weights((candidate - beta) * log_likelihood)
        return effective_samples(weights)

    if conditional_ess(1.0) >= target * count:
        return 1.0
    low = beta
    high = 1.0
    for _ in range(60):
        middle = (low + high) / 2.0
        if conditional_ess(middle) < target * count:
            high = middle
        else:
            low = middle
    if low <= beta + 1e-12:
        raise RuntimeError("temperature schedule stalled")
    return low


def run_replicate(
    replicate: int,
    particles_count: int,
    moves_per_temperature: int,
    target_conditional_ess: float,
    root: Path,
    family_index: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][family_index]
    dimensions = len(uncertainty.PARAMETERS)
    exponent = round(math.log2(particles_count))
    if 2**exponent != particles_count:
        raise RuntimeError("particle count must be a power of two")
    engine = qmc.Sobol(
        d=dimensions,
        scramble=True,
        seed=593_000 + 10_000 * family_index + replicate,
    )
    particles = engine.random_base2(m=exponent)
    evaluations = 0

    def evaluate_many(values: np.ndarray) -> np.ndarray:
        nonlocal evaluations
        result = np.empty(len(values), dtype=float)
        for index, value in enumerate(values):
            result[index] = uncertainty._evaluate_unit(
                value, packets, family, config, config59
            )[0]
        evaluations += len(values)
        return result

    log_likelihood = evaluate_many(particles)
    beta = 0.0
    rng = np.random.default_rng(594_000 + 10_000 * family_index + replicate)
    scale = 2.38 / math.sqrt(dimensions)
    stages = []
    log_evidence = 0.0
    ancestors = np.arange(particles_count)
    while beta < 1.0 - 1e-14:
        new_beta = next_temperature(beta, log_likelihood, target_conditional_ess)
        delta = new_beta - beta
        incremental_log_weights = delta * log_likelihood
        maximum = float(np.max(incremental_log_weights))
        log_evidence += maximum + math.log(
            float(np.mean(np.exp(incremental_log_weights - maximum)))
        )
        weights = normalized_weights(incremental_log_weights)
        conditional_ess = effective_samples(weights)
        indices = systematic_resample(weights, rng)
        particles = particles[indices].copy()
        log_likelihood = log_likelihood[indices].copy()
        ancestors = ancestors[indices]
        covariance = np.cov(particles, rowvar=False, ddof=1)
        covariance += np.eye(dimensions) * 1e-8
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        square_root = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-10)))
        acceptance_rates = []
        for _ in range(moves_per_temperature):
            noise = rng.normal(size=(particles_count, dimensions))
            proposals = reflect_unit(particles + scale * (noise @ square_root.T))
            proposal_log_likelihood = evaluate_many(proposals)
            log_acceptance = new_beta * (proposal_log_likelihood - log_likelihood)
            accepted = np.log(
                np.maximum(rng.random(particles_count), np.finfo(float).tiny)
            ) < np.minimum(0.0, log_acceptance)
            particles[accepted] = proposals[accepted]
            log_likelihood[accepted] = proposal_log_likelihood[accepted]
            rate = float(np.mean(accepted))
            acceptance_rates.append(rate)
            scale *= math.exp(0.5 * (rate - 0.234))
            scale = float(np.clip(scale, 0.02, 8.0))
        beta = new_beta
        stages.append(
            {
                "beta": beta,
                "conditional_ess": conditional_ess,
                "acceptance_rates": acceptance_rates,
                "proposal_scale": scale,
                "unique_ancestors": len(np.unique(ancestors)),
                "log_likelihood_maximum": float(np.max(log_likelihood)),
            }
        )
        print(
            json.dumps(
                {
                    "replicate": replicate,
                    "family": family["family_id"],
                    "stage": len(stages),
                    **stages[-1],
                    "evaluations": evaluations,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    permutation = rng.permutation(particles_count)
    particles = particles[permutation]
    log_likelihood = log_likelihood[permutation]
    summary = {
        "replicate": replicate,
        "family_id": family["family_id"],
        "particles": particles_count,
        "moves_per_temperature": moves_per_temperature,
        "temperature_stages": len(stages),
        "evaluations": evaluations,
        "log_evidence": log_evidence,
        "minimum_stage_acceptance": min(
            rate for stage in stages for rate in stage["acceptance_rates"]
        ),
        "maximum_stage_acceptance": max(
            rate for stage in stages for rate in stage["acceptance_rates"]
        ),
        "final_proposal_scale": scale,
        "final_unique_particles": len(np.unique(particles, axis=0)),
        "final_unique_ancestors": len(np.unique(ancestors)),
        "stages": stages,
    }
    return particles, log_likelihood, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=int, default=512)
    parser.add_argument("--moves", type=int, default=4)
    parser.add_argument("--target-ess", type=float, default=0.8)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--family-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    traces = []
    likelihoods = []
    summaries = []
    for replicate in range(args.replicates):
        particles, log_likelihood, summary = run_replicate(
            replicate,
            args.particles,
            args.moves,
            args.target_ess,
            root,
            args.family_index,
        )
        traces.append(particles)
        likelihoods.append(log_likelihood)
        summaries.append(summary)
    chains = np.asarray(traces)
    rhat, ess = uncertainty._rhat_and_ess(chains)
    flattened = chains.reshape(-1, chains.shape[-1])
    standard_deviation = np.std(flattened, axis=0, ddof=1)
    replicate_medians = np.median(chains, axis=1)
    standardized_median_spread = np.divide(
        np.ptp(replicate_medians, axis=0),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )
    aggregate = {
        "family_id": summaries[0]["family_id"],
        "replicates": args.replicates,
        "particles_per_replicate": args.particles,
        "posterior_particles": int(flattened.shape[0]),
        "total_evaluations": sum(int(summary["evaluations"]) for summary in summaries),
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_replicate_median_spread": float(
            np.max(standardized_median_spread)
        ),
        "log_evidence_range": [
            min(float(summary["log_evidence"]) for summary in summaries),
            max(float(summary["log_evidence"]) for summary in summaries),
        ],
        "parameters": [
            {
                "parameter": parameter,
                "rhat": float(rhat[index]),
                "effective_samples": float(ess[index]),
                "standardized_between_replicate_median_spread": float(
                    standardized_median_spread[index]
                ),
            }
            for index, parameter in enumerate(uncertainty.PARAMETERS)
        ],
        "replicate_summaries": summaries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        particles=chains,
        log_likelihood=np.asarray(likelihoods),
        summary=np.asarray(json.dumps(aggregate, sort_keys=True)),
    )
    print(json.dumps({"aggregate": aggregate, "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
