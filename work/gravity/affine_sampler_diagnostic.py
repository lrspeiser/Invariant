from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import differential_evolution

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_cluster_uncertainty_program as uncertainty
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59


def logit(unit: np.ndarray) -> np.ndarray:
    clipped = np.clip(unit, 1e-12, 1.0 - 1e-12)
    return np.log(clipped) - np.log1p(-clipped)


def logistic(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    dimensions = len(uncertainty.PARAMETERS)
    ensemble_count = 4
    walkers = 36
    burn = 1200
    retained = 800
    stretch = 2.0
    evaluations = 0

    def evaluate_unit(unit: np.ndarray) -> float:
        nonlocal evaluations
        evaluations += 1
        return uncertainty._evaluate_unit(unit, packets, family, config, config59)[0]

    traces = np.empty((ensemble_count, walkers, retained, dimensions), dtype=float)
    acceptance = []
    map_log_likelihoods = []
    for ensemble_index in range(ensemble_count):
        result = differential_evolution(
            lambda unit: -evaluate_unit(np.asarray(unit, dtype=float)),
            bounds=[(1e-8, 1.0 - 1e-8)] * dimensions,
            seed=591_000 + ensemble_index,
            popsize=4,
            maxiter=45,
            tol=1e-7,
            polish=False,
            updating="immediate",
            workers=1,
        )
        order = np.argsort(result.population_energies)
        initial = np.asarray(result.population[order[:walkers]], dtype=float)
        positions = logit(initial)
        log_likelihood = np.asarray([evaluate_unit(unit) for unit in initial])
        units = logistic(positions)
        log_target = log_likelihood + np.sum(np.log(units) + np.log1p(-units), axis=1)
        rng = np.random.default_rng(592_000 + ensemble_index)
        accepted = 0
        proposals = 0
        recorded = 0
        groups = (np.arange(0, walkers, 2), np.arange(1, walkers, 2))
        for sweep in range(burn + retained):
            for active, complement in ((groups[0], groups[1]), (groups[1], groups[0])):
                for walker in active:
                    partner = int(rng.choice(complement))
                    uniform = rng.random()
                    z_scale = (
                        1.0 / math.sqrt(stretch)
                        + uniform
                        * (math.sqrt(stretch) - 1.0 / math.sqrt(stretch))
                    ) ** 2
                    proposal = positions[partner] + z_scale * (
                        positions[walker] - positions[partner]
                    )
                    proposal_unit = logistic(proposal)
                    proposal_log_likelihood = evaluate_unit(proposal_unit)
                    proposal_log_target = proposal_log_likelihood + float(
                        np.sum(np.log(proposal_unit) + np.log1p(-proposal_unit))
                    )
                    log_acceptance = (dimensions - 1) * math.log(z_scale)
                    log_acceptance += proposal_log_target - log_target[walker]
                    proposals += 1
                    if math.log(max(rng.random(), np.finfo(float).tiny)) < min(
                        0.0, log_acceptance
                    ):
                        positions[walker] = proposal
                        log_likelihood[walker] = proposal_log_likelihood
                        log_target[walker] = proposal_log_target
                        accepted += 1
            if sweep >= burn:
                traces[ensemble_index, :, recorded, :] = logistic(positions)
                recorded += 1
        acceptance.append(accepted / proposals)
        map_log_likelihoods.append(float(np.max(log_likelihood)))
        print(
            json.dumps(
                {
                    "progress": f"ensemble_{ensemble_index + 1}_of_{ensemble_count}",
                    "acceptance_rate": acceptance[-1],
                    "map_log_likelihood": map_log_likelihoods[-1],
                    "evaluations": evaluations,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    walker_chains = traces.reshape(ensemble_count * walkers, retained, dimensions)
    rhat, ess = uncertainty._rhat_and_ess(walker_chains)
    flattened = traces.reshape(-1, dimensions)
    posterior_standard_deviation = np.std(flattened, axis=0, ddof=1)
    ensemble_medians = np.median(traces, axis=(1, 2))
    median_spread = np.ptp(ensemble_medians, axis=0)
    standardized_median_spread = np.divide(
        median_spread,
        posterior_standard_deviation,
        out=np.zeros_like(median_spread),
        where=posterior_standard_deviation > np.finfo(float).tiny,
    )
    summary = {
        "engine": "four_independent_logit_affine_invariant_stretch_ensembles",
        "ensemble_count": ensemble_count,
        "walkers_per_ensemble": walkers,
        "burn_sweeps": burn,
        "retained_sweeps": retained,
        "posterior_draws": int(flattened.shape[0]),
        "evaluations": evaluations,
        "acceptance_rates": acceptance,
        "map_log_likelihoods": map_log_likelihoods,
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_ensemble_median_spread": float(
            np.max(standardized_median_spread)
        ),
        "per_parameter": [
            {
                "parameter": parameter,
                "rhat": float(rhat[index]),
                "effective_samples": float(ess[index]),
                "standardized_between_ensemble_median_spread": float(
                    standardized_median_spread[index]
                ),
            }
            for index, parameter in enumerate(uncertainty.PARAMETERS)
        ],
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
