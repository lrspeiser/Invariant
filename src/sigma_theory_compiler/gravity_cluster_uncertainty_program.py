"""Development-data nuisance marginalization and covariance sensitivity for clusters."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import qmc

from sigma_theory_compiler import gravity_cluster_comparator_suite as comparators
from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59

CONFIG_PATH = Path("configs/gravity_cluster_uncertainty_program_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/uncertainty-program-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-uncertainty-program-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-uncertainty-program-receipt-1.0"
PARAMETERS = (
    "outer_nonthermal_fraction",
    "nonthermal_radial_power",
    "xray_temperature_cross_calibration",
    "outer_pressure_boundary_sigma",
    "density_error_sigma",
    "bcg_mass_scale",
    "satellite_mass_scale",
    "missing_member_fraction",
    "intracluster_light_fraction",
    "imf_mass_scale",
    "mass_to_light_scale",
    "missing_stellar_to_gas_mass_ratio",
    "clumping_amplitude",
    "centering_radius_shift",
    "projection_density_scale",
    "triaxial_radius_scale",
    "spherical_acceleration_scale",
)
COMPLETED_TASKS = ("CP5.7", "CP5.8", "CP5.9", "CP5.10", "CP5.12", "CP5.14")
BLOCKED_TASKS = (
    "CP5.1",
    "CP5.2",
    "CP5.3",
    "CP5.4",
    "CP5.5",
    "CP5.6",
    "CP5.11",
    "CP5.13",
)


class GravityClusterUncertaintyError(RuntimeError):
    """Raised when the nuisance, covariance, or sealed-target contract changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterUncertaintyError(f"{label} keys changed")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterUncertaintyError(f"expected JSON object: {path}")
    return value


def _validate_bindings(root: Path, bindings: Mapping[str, Mapping[str, Any]]) -> None:
    if tuple(bindings) != ("item59_config", "item59_result", "comparator_receipt"):
        raise GravityClusterUncertaintyError("uncertainty source binding order changed")
    for label, binding in bindings.items():
        expected = {"path", "file_sha256"} if label == "item59_config" else {
            "path",
            "file_sha256",
            "content_sha256",
        }
        _strict(binding, expected, f"{label} binding")
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterUncertaintyError("source binding escaped root") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterUncertaintyError(f"source binding changed: {label}")
        if "content_sha256" in binding:
            value = _load_json(path)
            if value.get("content_sha256") != binding["content_sha256"]:
                raise GravityClusterUncertaintyError(f"source content changed: {label}")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    validate_config(config, root)
    return config


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "program_id",
            "source_bindings",
            "sample_contract",
            "quasi_monte_carlo",
            "posterior_sampler",
            "continuous_priors",
            "stellar_combination_rule",
            "observable_transform_rules",
            "candidate_and_control_families",
            "covariance_stress",
            "missingness_stress",
            "source_covariance_blockers",
            "claim_boundary",
            "output_path",
        },
        "uncertainty config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"]
        != "frozen_development_nuisance_program_before_independent_target_access"
        or config["program_id"] != "gravity-cluster-uncertainty-program-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterUncertaintyError("uncertainty config identity changed")
    _validate_bindings(root, config["source_bindings"])
    sample = config["sample_contract"]
    if (
        len(sample["clusters"]) != 8
        or sample["posterior_weight_split"] != "development_train"
        or sample["predictive_score_split"] != "development_holdout"
        or sample["xcop_confirmation_rows_used"] is not False
        or sample["independent_source_rows_used"] is not False
        or sample["target_rows_opened"] != 0
    ):
        raise GravityClusterUncertaintyError("uncertainty sample seal changed")
    qmc_config = config["quasi_monte_carlo"]
    if (
        qmc_config["engine"] != "scipy_sobol"
        or qmc_config["role"]
        != "prior_space_initialization_and_log_mean_likelihood_diagnostic"
        or qmc_config["scramble"] is not True
        or qmc_config["samples"] != 2 ** int(qmc_config["power_of_two_exponent"])
        or qmc_config["selection_uses_holdout"] is not False
    ):
        raise GravityClusterUncertaintyError("quasi-Monte Carlo freeze changed")
    sampler = config["posterior_sampler"]
    if (
        sampler["engine"]
        != "deterministic_adaptive_componentwise_random_walk_metropolis"
        or sampler["chains"] != 4
        or sampler["retained_samples_per_chain"] != 64
        or sampler["thinning_sweeps"] != 1
        or sampler["maximum_rhat_for_completion"] > 1.2
        or sampler["minimum_effective_samples_for_completion"] < 50
    ):
        raise GravityClusterUncertaintyError("posterior sampler freeze changed")
    priors = config["continuous_priors"]
    if tuple(row["parameter"] for row in priors) != PARAMETERS:
        raise GravityClusterUncertaintyError("continuous prior inventory changed")
    for row in priors:
        _strict(row, {"parameter", "cause", "distribution", "low", "high"}, "prior")
        if row["distribution"] != "uniform" or not float(row["low"]) < float(row["high"]):
            raise GravityClusterUncertaintyError("continuous prior invalid")
    if set(config["observable_transform_rules"]) != {
        "density_error",
        "clumping",
        "projection",
        "geometry",
        "boundary",
        "spherical_approximation",
    }:
        raise GravityClusterUncertaintyError("observable transformation inventory changed")
    families = config["candidate_and_control_families"]
    if [row["family_id"] for row in families] != [
        "cross_scale_boundary",
        "newtonian_baryons",
    ] or [row["role"] for row in families] != ["frozen_candidate", "nuisance_only_control"]:
        raise GravityClusterUncertaintyError("candidate/control boundary changed")
    covariance = config["covariance_stress"]
    if (
        covariance["status"] != "sensitivity_only_not_source_covariance"
        or covariance["full_source_covariance_claimed"] is not False
        or len(covariance["radial_log_correlation_lengths"]) != 4
        or len(covariance["diagonal_error_inflation"]) != 3
        or len(covariance["shared_temperature_calibration_fraction"]) != 3
    ):
        raise GravityClusterUncertaintyError("covariance sensitivity boundary changed")
    missing = config["missingness_stress"]
    if len(missing["fractions"]) != 3 or len(missing["scenarios"]) != 4 or (
        "never_select_or_exclude" not in missing["claim_use"]
    ):
        raise GravityClusterUncertaintyError("missingness sensitivity changed")
    if len(config["source_covariance_blockers"]) != 8:
        raise GravityClusterUncertaintyError("source covariance blockers changed")
    claims = config["claim_boundary"]
    if claims["development_nuisance_marginalization_complete"] is not True or any(
        claims[key]
        for key in claims
        if key != "development_nuisance_marginalization_complete"
    ):
        raise GravityClusterUncertaintyError("uncertainty claim boundary weakened")


def _samples(config: Mapping[str, Any]) -> tuple[np.ndarray, list[dict[str, float]]]:
    qmc_config = config["quasi_monte_carlo"]
    engine = qmc.Sobol(
        d=len(PARAMETERS),
        scramble=bool(qmc_config["scramble"]),
        seed=int(qmc_config["seed"]),
    )
    unit = engine.random_base2(m=int(qmc_config["power_of_two_exponent"]))
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    values = lows + unit * (highs - lows)
    rows = [
        {name: float(value) for name, value in zip(PARAMETERS, sample, strict=True)}
        for sample in values
    ]
    return values, rows


def _transformed_packets(
    packets: Sequence[Mapping[str, Any]], parameters: Mapping[str, float]
) -> list[dict[str, Any]]:
    transformed = []
    radius_factor = (1.0 + float(parameters["centering_radius_shift"])) * float(
        parameters["triaxial_radius_scale"]
    )
    for original in packets:
        packet = dict(original)
        base_radius = np.asarray(original["density_radius_kpc"], dtype=float)
        r500 = float(original["r500_kpc"])
        ne = np.asarray(original["ne_cm3"], dtype=float).copy()
        shift = float(parameters["density_error_sigma"])
        if shift < 0.0:
            ne += shift * np.asarray(original["ne_error_low_cm3"], dtype=float)
        else:
            ne += shift * np.asarray(original["ne_error_high_cm3"], dtype=float)
        clumping = 1.0 + float(parameters["clumping_amplitude"]) * (base_radius / r500) ** 2
        ne = np.maximum(ne / np.sqrt(clumping), np.finfo(float).tiny)
        ne *= float(parameters["projection_density_scale"])
        packet["density_radius_kpc"] = base_radius * radius_factor
        packet["ne_cm3"] = ne
        packet["r500_kpc"] = r500 * radius_factor
        packet["anchor"] = dict(original["anchor"])
        packet["anchor"]["radius_kpc"] = (
            float(original["anchor"]["radius_kpc"]) * radius_factor
        )
        packet["anchor"]["pressure_kev_cm3"] = max(
            np.finfo(float).tiny,
            float(original["anchor"]["pressure_kev_cm3"])
            + float(parameters["outer_pressure_boundary_sigma"])
            * float(original["anchor"]["error_kev_cm3"]),
        )
        packet["rows"] = [
            {**row, "radius_kpc": float(row["radius_kpc"]) * radius_factor}
            for row in original["rows"]
        ]
        if original["stellar"] is not None:
            packet["stellar"] = {
                **original["stellar"],
                "radius_kpc": np.asarray(original["stellar"]["radius_kpc"], dtype=float)
                * radius_factor,
            }
        transformed.append(packet)
    return transformed


def _variant(
    family_id: str, parameters: Mapping[str, float], sample: Mapping[str, float]
) -> dict[str, Any]:
    stellar_scale = float(
        np.clip(
            float(sample["bcg_mass_scale"])
            * float(sample["satellite_mass_scale"])
            * (1.0 + float(sample["missing_member_fraction"]))
            * (1.0 + float(sample["intracluster_light_fraction"]))
            * float(sample["imf_mass_scale"])
            * float(sample["mass_to_light_scale"]),
            0.4,
            2.5,
        )
    )
    return {
        "family_id": family_id,
        "parameters": dict(parameters),
        "nuisances": {
            "outer_nonthermal_fraction": float(sample["outer_nonthermal_fraction"]),
            "published_stellar_mass_scale": stellar_scale,
            "missing_stellar_to_gas_mass_ratio": float(
                sample["missing_stellar_to_gas_mass_ratio"]
            ),
            "xray_temperature_cross_calibration": float(
                sample["xray_temperature_cross_calibration"]
            ),
        },
    }


def _local_item59_config(
    config59: Mapping[str, Any], sample: Mapping[str, float]
) -> dict[str, Any]:
    local = copy.deepcopy(config59)
    local["nuisance_grid"]["nonthermal_radial_power"] = float(
        sample["nonthermal_radial_power"]
    )
    local["constants"]["gravity_si"] = float(config59["constants"]["gravity_si"]) * float(
        sample["spherical_acceleration_scale"]
    )
    return local


def _log_likelihood(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    config59: Mapping[str, Any],
) -> float:
    floor = float(config59["scoring"]["minimum_fractional_error"])
    value = 0.0
    for row in item59._rows(packets, split):
        sigma = max(float(row["error"]) / float(row["observed"]), floor)
        residual = math.log(float(predictions[str(row["row_id"])]) / float(row["observed"]))
        value += -0.5 * ((residual / sigma) ** 2 + math.log(2.0 * math.pi * sigma**2))
    return float(value)


def _log_mean_exp(values: np.ndarray) -> float:
    maximum = float(np.max(values))
    return maximum + math.log(float(np.mean(np.exp(values - maximum))))


def _decode_unit(unit: np.ndarray, config: Mapping[str, Any]) -> dict[str, float]:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    values = lows + unit * (highs - lows)
    return {
        name: float(value) for name, value in zip(PARAMETERS, values, strict=True)
    }


def _evaluate_unit(
    unit: np.ndarray,
    packets: Sequence[Mapping[str, Any]],
    family: Mapping[str, Any],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> tuple[float, dict[str, float]]:
    sample = _decode_unit(unit, config)
    transformed = _transformed_packets(packets, sample)
    local_config = _local_item59_config(config59, sample)
    variant = _variant(str(family["family_id"]), family["parameters"], sample)
    predictions = item59._variant_predictions(transformed, variant, local_config)
    return (
        _log_likelihood(transformed, predictions, "development_train", local_config),
        predictions,
    )


def _reflect_unit(values: np.ndarray) -> np.ndarray:
    wrapped = np.mod(values, 2.0)
    return np.where(wrapped <= 1.0, wrapped, 2.0 - wrapped)


def _rhat_and_ess(chains: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    chain_count, samples_per_chain, dimensions = chains.shape
    rhat = np.empty(dimensions, dtype=float)
    ess = np.empty(dimensions, dtype=float)
    for dimension in range(dimensions):
        values = chains[:, :, dimension]
        chain_means = np.mean(values, axis=1)
        within = float(np.mean(np.var(values, axis=1, ddof=1)))
        between = float(samples_per_chain * np.var(chain_means, ddof=1))
        if within <= np.finfo(float).tiny:
            rhat[dimension] = 1.0 if between <= np.finfo(float).tiny else math.inf
        else:
            variance = (samples_per_chain - 1.0) / samples_per_chain * within
            variance += between / samples_per_chain
            rhat[dimension] = math.sqrt(max(variance / within, 0.0))

        centered = values - np.mean(values, axis=1, keepdims=True)
        denominator = float(np.sum(centered**2))
        correlations = []
        if denominator > np.finfo(float).tiny:
            for lag in range(1, min(50, samples_per_chain - 1) + 1):
                numerator = float(np.sum(centered[:, :-lag] * centered[:, lag:]))
                rho = numerator / denominator * samples_per_chain / (
                    samples_per_chain - lag
                )
                if rho <= 0.0:
                    break
                correlations.append(rho)
        tau = 1.0 + 2.0 * sum(correlations)
        ess[dimension] = min(chain_count * samples_per_chain, chain_count * samples_per_chain / tau)
    return rhat, ess


def _family_marginalization(
    packets: Sequence[Mapping[str, Any]],
    initial_unit: np.ndarray,
    family: Mapping[str, Any],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    holdout_rows = item59._rows(packets, "development_holdout")
    row_ids = [str(row["row_id"]) for row in holdout_rows]
    initial_log_likelihood = np.empty(len(initial_unit), dtype=float)
    for index, unit in enumerate(initial_unit):
        initial_log_likelihood[index], _predictions = _evaluate_unit(
            unit, packets, family, config, config59
        )

    sampler = config["posterior_sampler"]
    chain_count = int(sampler["chains"])
    retained_per_chain = int(sampler["retained_samples_per_chain"])
    thinning = int(sampler["thinning_sweeps"])
    burn_in = int(sampler["burn_in_sweeps"])
    starts = initial_unit[np.argsort(initial_log_likelihood)[-chain_count:][::-1]]
    chain_values = np.empty((chain_count, retained_per_chain, len(PARAMETERS)))
    chain_log_likelihood = np.empty((chain_count, retained_per_chain))
    chain_holdout_score = np.empty((chain_count, retained_per_chain))
    chain_holdout_predictions = np.empty(
        (chain_count, retained_per_chain, len(row_ids)), dtype=float
    )
    acceptance_rates = []
    final_scales = []
    family_seed_offset = 0 if family["role"] == "frozen_candidate" else 100_000
    proposal_evaluations = len(initial_unit)
    for chain_index in range(chain_count):
        rng = np.random.default_rng(
            int(sampler["seed"]) + family_seed_offset + chain_index
        )
        current = starts[chain_index].copy()
        current_log_likelihood, current_predictions = _evaluate_unit(
            current, packets, family, config, config59
        )
        proposal_evaluations += 1
        scales = np.full(len(PARAMETERS), float(sampler["initial_unit_proposal_scale"]))
        accepted_total = 0
        accepted_interval = np.zeros(len(PARAMETERS), dtype=int)
        retained = 0
        total_sweeps = burn_in + retained_per_chain * thinning
        for sweep in range(total_sweeps):
            for dimension in range(len(PARAMETERS)):
                proposal = current.copy()
                proposal[dimension] = _reflect_unit(
                    np.asarray(
                        [
                            current[dimension]
                            + rng.normal(0.0, scales[dimension])
                        ]
                    )
                )[0]
                proposal_log_likelihood, proposal_predictions = _evaluate_unit(
                    proposal, packets, family, config, config59
                )
                proposal_evaluations += 1
                if math.log(max(rng.random(), np.finfo(float).tiny)) < min(
                    0.0, proposal_log_likelihood - current_log_likelihood
                ):
                    current = proposal
                    current_log_likelihood = proposal_log_likelihood
                    current_predictions = proposal_predictions
                    accepted_total += 1
                    accepted_interval[dimension] += 1
            interval = int(sampler["adapt_interval_during_burn_in_sweeps"])
            if sweep < burn_in and (sweep + 1) % interval == 0:
                rates = accepted_interval / interval
                scales[rates < float(sampler["target_acceptance_low"])] *= 0.7
                scales[rates > float(sampler["target_acceptance_high"])] *= 1.3
                accepted_interval[:] = 0
            if sweep >= burn_in and (sweep - burn_in + 1) % thinning == 0:
                chain_values[chain_index, retained] = current
                chain_log_likelihood[chain_index, retained] = current_log_likelihood
                chain_holdout_predictions[chain_index, retained] = [
                    current_predictions[row_id] for row_id in row_ids
                ]
                chain_holdout_score[chain_index, retained] = float(
                    item59._score_predictions(
                        packets,
                        current_predictions,
                        "development_holdout",
                        config59,
                    )["score"]
                )
                retained += 1
        acceptance_rates.append(
            accepted_total / (total_sweeps * len(PARAMETERS))
        )
        final_scales.append([float(value) for value in scales])
    if np.any(~np.isfinite(chain_values)):
        raise GravityClusterUncertaintyError("posterior chain emitted nonfinite values")
    rhat, ess = _rhat_and_ess(chain_values)
    flattened_unit = chain_values.reshape(-1, len(PARAMETERS))
    holdout_score = chain_holdout_score.reshape(-1)
    holdout_predictions = chain_holdout_predictions.reshape(-1, len(row_ids))
    convergence = bool(
        float(np.max(rhat)) <= float(sampler["maximum_rhat_for_completion"])
        and float(np.min(ess))
        >= float(sampler["minimum_effective_samples_for_completion"])
    )
    median_predictions = {
        row_id: float(np.median(holdout_predictions[:, index]))
        for index, row_id in enumerate(row_ids)
    }
    median_score = item59._score_predictions(
        packets, median_predictions, "development_holdout", config59
    )
    coverage = []
    for index, row in enumerate(holdout_rows):
        low, high = np.quantile(holdout_predictions[:, index], [0.05, 0.95])
        coverage.append(low <= float(row["observed"]) <= high)
    return {
        "family_id": family["family_id"],
        "role": family["role"],
        "initial_prior_space_samples": len(initial_unit),
        "initial_prior_log_mean_likelihood_diagnostic": _log_mean_exp(
            initial_log_likelihood
        ),
        "posterior_samples": len(flattened_unit),
        "posterior_sampler": {
            "converged": convergence,
            "maximum_rhat": float(np.max(rhat)),
            "minimum_effective_samples": float(np.min(ess)),
            "acceptance_rates": acceptance_rates,
            "final_unit_proposal_scales": final_scales,
            "proposal_evaluations": proposal_evaluations,
            "per_parameter": [
                {
                    "parameter": parameter,
                    "rhat": float(rhat[index]),
                    "effective_samples": float(ess[index]),
                }
                for index, parameter in enumerate(PARAMETERS)
            ],
        },
        "posterior_predictive_median_holdout": {
            key: value for key, value in median_score.items() if key != "per_row"
        },
        "posterior_holdout_score": {
            "median": float(np.median(holdout_score)),
            "q05": float(np.quantile(holdout_score, 0.05)),
            "q95": float(np.quantile(holdout_score, 0.95)),
        },
        "nuisance_only_predictive_interval_90pct_row_coverage": float(np.mean(coverage)),
        "posterior_unit_samples": flattened_unit,
        "posterior_log_likelihood": chain_log_likelihood.reshape(-1),
    }


def _posterior_parameters(
    posterior_unit: np.ndarray, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    sample_matrix = lows + posterior_unit * (highs - lows)
    rows = []
    for index, prior in enumerate(config["continuous_priors"]):
        values = sample_matrix[:, index]
        rows.append(
            {
                "parameter": prior["parameter"],
                "cause": prior["cause"],
                "prior_low": float(prior["low"]),
                "prior_high": float(prior["high"]),
                "posterior_q05": float(np.quantile(values, 0.05)),
                "posterior_median": float(np.median(values)),
                "posterior_q95": float(np.quantile(values, 0.95)),
            }
        )
    return rows


def _covariance_score(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    correlation_length: float,
    inflation: float,
    shared_temperature: float,
    jitter: float,
    config59: Mapping[str, Any],
) -> float:
    floor = float(config59["scoring"]["minimum_fractional_error"])
    groups = {}
    for row in item59._rows(packets, split):
        groups.setdefault((str(row["cluster"]), str(row["observable"])), []).append(row)
    scores = []
    for (_cluster, observable), rows in groups.items():
        rows = sorted(rows, key=lambda row: float(row["radius_kpc"]))
        radius = np.asarray([float(row["radius_kpc"]) for row in rows])
        residual = np.asarray(
            [
                math.log(
                    float(predictions[str(row["row_id"])]) / float(row["observed"])
                )
                for row in rows
            ]
        )
        sigma = inflation * np.asarray(
            [max(float(row["error"]) / float(row["observed"]), floor) for row in rows]
        )
        if correlation_length == 0.0:
            correlation = np.eye(len(rows))
        else:
            delta = np.abs(np.log(radius[:, None] / radius[None, :]))
            correlation = np.exp(-delta / correlation_length)
        covariance = sigma[:, None] * correlation * sigma[None, :]
        if observable == "temperature":
            covariance += shared_temperature**2
        covariance += np.eye(len(rows)) * jitter
        scores.append(float(residual @ np.linalg.solve(covariance, residual) / len(rows)))
    return float(np.mean(scores))


def _covariance_stress(
    packets: Sequence[Mapping[str, Any]],
    candidate_predictions: Mapping[str, float],
    nfw_predictions: Mapping[str, float],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    stress = config["covariance_stress"]
    rows = []
    for length in stress["radial_log_correlation_lengths"]:
        for inflation in stress["diagonal_error_inflation"]:
            for shared in stress["shared_temperature_calibration_fraction"]:
                candidate_score = _covariance_score(
                    packets,
                    candidate_predictions,
                    "development_holdout",
                    float(length),
                    float(inflation),
                    float(shared),
                    float(stress["jitter_fraction"]),
                    config59,
                )
                nfw_score = _covariance_score(
                    packets,
                    nfw_predictions,
                    "development_holdout",
                    float(length),
                    float(inflation),
                    float(shared),
                    float(stress["jitter_fraction"]),
                    config59,
                )
                rows.append(
                    {
                        "radial_log_correlation_length": float(length),
                        "diagonal_error_inflation": float(inflation),
                        "shared_temperature_calibration_fraction": float(shared),
                        "candidate_score": candidate_score,
                        "nfw_score": nfw_score,
                        "candidate_beats_nfw": candidate_score < nfw_score,
                    }
                )
    return {
        "status": stress["status"],
        "scenarios": rows,
        "candidate_beats_nfw_scenarios": sum(row["candidate_beats_nfw"] for row in rows),
        "total_scenarios": len(rows),
        "candidate_score_range": [
            min(row["candidate_score"] for row in rows),
            max(row["candidate_score"] for row in rows),
        ],
        "full_source_covariance_claimed": False,
    }


def _missingness_stress(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    config: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    rows = item59._rows(packets, "development_holdout")
    floor = float(config59["scoring"]["minimum_fractional_error"])
    metrics = []
    for row in rows:
        standardized = abs(
            math.log(float(predictions[str(row["row_id"])]) / float(row["observed"]))
        ) / max(float(row["error"]) / float(row["observed"]), floor)
        packet = next(packet for packet in packets if packet["cluster"] == row["cluster"])
        metrics.append(
            {
                "row_id": str(row["row_id"]),
                "standardized": standardized,
                "scaled_radius": float(row["radius_kpc"]) / float(packet["r500_kpc"]),
            }
        )
    results = []
    for fraction in config["missingness_stress"]["fractions"]:
        count = max(1, round(len(metrics) * float(fraction)))
        orders = {
            "drop_largest_absolute_standardized_residuals": sorted(
                metrics, key=lambda row: (-row["standardized"], row["row_id"])
            ),
            "drop_smallest_absolute_standardized_residuals": sorted(
                metrics, key=lambda row: (row["standardized"], row["row_id"])
            ),
            "drop_outermost_radii": sorted(
                metrics, key=lambda row: (-row["scaled_radius"], row["row_id"])
            ),
            "drop_innermost_radii": sorted(
                metrics, key=lambda row: (row["scaled_radius"], row["row_id"])
            ),
        }
        for scenario, ordered in orders.items():
            removed = {row["row_id"] for row in ordered[:count]}
            remaining = [row["standardized"] ** 2 for row in metrics if row["row_id"] not in removed]
            results.append(
                {
                    "scenario": scenario,
                    "fraction": float(fraction),
                    "removed_rows": len(removed),
                    "remaining_row_mean_standardized_square": float(np.mean(remaining)),
                }
            )
    return {
        "status": config["missingness_stress"]["claim_use"],
        "scenarios": results,
        "score_range": [
            min(row["remaining_row_mean_standardized_square"] for row in results),
            max(row["remaining_row_mean_standardized_square"] for row in results),
        ],
    }


def _nominal_predictions(
    packets: Sequence[Mapping[str, Any]], config59: Mapping[str, Any], comparator: Mapping[str, Any]
) -> tuple[dict[str, float], dict[str, float]]:
    candidate_variant = {
        "family_id": "cross_scale_boundary",
        "parameters": {"beta": 1.5},
        "nuisances": {
            "missing_stellar_to_gas_mass_ratio": 0.2,
            "outer_nonthermal_fraction": 0.3,
            "published_stellar_mass_scale": 1.3,
            "xray_temperature_cross_calibration": 1.0,
        },
    }
    candidate = item59._variant_predictions(packets, candidate_variant, config59)
    nfw = comparator["comparators"]["GR_PLUS_NFW"]["selection"]
    nfw_predictions = comparators._gravity_model_predictions(
        packets,
        "GR_PLUS_NFW",
        nfw["parameters"],
        nfw["nuisances"],
        config59,
    )
    return candidate, nfw_predictions


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    if [packet["cluster"] for packet in packets] != config["sample_contract"]["clusters"]:
        raise GravityClusterUncertaintyError("uncertainty cluster population changed")
    initial_unit, samples = _samples(config)
    families = config["candidate_and_control_families"]
    candidate = _family_marginalization(
        packets, initial_unit, families[0], config, config59
    )
    newtonian = _family_marginalization(
        packets, initial_unit, families[1], config, config59
    )
    posterior_parameters = _posterior_parameters(
        candidate["posterior_unit_samples"], config
    )

    comparator = _load_json(root / config["source_bindings"]["comparator_receipt"]["path"])
    nominal_candidate, nominal_nfw = _nominal_predictions(packets, config59, comparator)
    covariance = _covariance_stress(
        packets, nominal_candidate, nominal_nfw, config, config59
    )
    missingness = _missingness_stress(packets, nominal_candidate, config, config59)
    candidate_score = float(
        candidate["posterior_predictive_median_holdout"]["score"]
    )
    newtonian_score = float(
        newtonian["posterior_predictive_median_holdout"]["score"]
    )
    indistinguishable = sorted({str(row["cause"]) for row in config["continuous_priors"]})

    def public_family(value: Mapping[str, Any]) -> dict[str, Any]:
        return {
            key: item
            for key, item in value.items()
            if key not in {"posterior_unit_samples", "posterior_log_likelihood"}
        }

    candidate_sampler_converged = bool(candidate["posterior_sampler"]["converged"])
    newtonian_sampler_converged = bool(newtonian["posterior_sampler"]["converged"])
    marginalization_complete = candidate_sampler_converged
    decision = (
        "DEVELOPMENT_NUISANCE_MARGINALIZATION_COMPLETE_SOURCE_COVARIANCE_BLOCKED"
        if marginalization_complete
        else "DEVELOPMENT_NUISANCE_SAMPLER_NOT_CONVERGED_SOURCE_COVARIANCE_BLOCKED"
    )
    completed_evidence = {
        "CP5.12": "36_covariance_and_12_missingness_sensitivity_scenarios_frozen_and_run",
        "CP5.14": "remaining_observationally_indistinguishable_causes_reported",
    }
    if marginalization_complete:
        completed_evidence.update(
            {
            "CP5.7": "continuous_nonthermal_fraction_and_radial_power_prior_marginalized",
            "CP5.8": "BCG_satellite_missing_member_ICL_IMF_and_mass_to_light_priors_marginalized",
            "CP5.9": "outer_pressure_boundary_prior_marginalized",
            "CP5.10": "clumping_centering_projection_triaxiality_and_spherical_sensitivity_marginalized",
            }
        )

    body = {
        "schema_version": RECEIPT_SCHEMA,
        "program_id": config["program_id"],
        "decision": decision,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "sample": {
            "clusters": config["sample_contract"]["clusters"],
            "training_rows": len(item59._rows(packets, "development_train")),
            "holdout_rows": len(item59._rows(packets, "development_holdout")),
            "xcop_confirmation_rows_used": False,
            "independent_source_rows_used": False,
            "target_rows_opened": 0,
        },
        "marginalization": {
            "candidate": public_family(candidate),
            "newtonian_nuisance_only_control": public_family(newtonian),
            "candidate_improvement_over_newtonian_posterior_predictive_median": 1.0
            - candidate_score / newtonian_score,
            "nuisance_only_newtonian_control_rescues_development_holdout": (
                newtonian_score <= candidate_score
            ),
            "posterior_parameters": posterior_parameters,
        },
        "covariance_sensitivity": covariance,
        "missingness_sensitivity": missingness,
        "observational_indistinguishability": {
            "unique_cause_identified": False,
            "causes_remaining_indistinguishable_with_current_single_source_diagonal_errors": indistinguishable,
            "ordinary_halo_comparison": {
                "source": "bound_CP4_comparator_receipt",
                "candidate_development_holdout_score": comparator["candidate"]["holdout"]["score"],
                "best_NFW_development_holdout_score": comparator["comparators"]["GR_PLUS_NFW"]["holdout"]["score"],
            },
            "merger_state_comparison_complete": False,
        },
        "completed_goal_evidence": completed_evidence,
        "blocked_goal_evidence": {
            "CP5.1": config["source_covariance_blockers"][0],
            "CP5.2": config["source_covariance_blockers"][1],
            "CP5.3": config["source_covariance_blockers"][2],
            "CP5.4": config["source_covariance_blockers"][3],
            "CP5.5": config["source_covariance_blockers"][4],
            "CP5.6": config["source_covariance_blockers"][5],
            "CP5.11": config["source_covariance_blockers"][6],
            "CP5.13": config["source_covariance_blockers"][7],
            **(
                {}
                if marginalization_complete
                else {
                    "CP5.7": "candidate_17_parameter_posterior_sampler_not_converged",
                    "CP5.8": "stellar_component_posterior_sampler_not_converged",
                    "CP5.9": "boundary_posterior_sampler_not_converged",
                    "CP5.10": "geometry_and_clumping_posterior_sampler_not_converged",
                }
            ),
        },
        "counts": {
            "continuous_nuisance_parameters": len(PARAMETERS),
            "quasi_monte_carlo_initial_samples_per_family": len(samples),
            "posterior_samples_per_family": int(
                config["posterior_sampler"]["chains"]
                * config["posterior_sampler"]["retained_samples_per_chain"]
            ),
            "families_marginalized": 2,
            "forward_evaluations": int(
                candidate["posterior_sampler"]["proposal_evaluations"]
                + newtonian["posterior_sampler"]["proposal_evaluations"]
            ),
            "covariance_sensitivity_scenarios": covariance["total_scenarios"],
            "missingness_sensitivity_scenarios": len(missingness["scenarios"]),
            "completed_CP5_tasks": (
                len(COMPLETED_TASKS) if marginalization_complete else 2
            ),
            "blocked_CP5_tasks": (
                len(BLOCKED_TASKS)
                if marginalization_complete
                else len(BLOCKED_TASKS) + len(COMPLETED_TASKS) - 2
            ),
            "target_rows_opened": 0,
        },
        "claims": {
            "development_nuisance_marginalization_complete": marginalization_complete,
            "candidate_posterior_sampler_converged": candidate_sampler_converged,
            "newtonian_control_posterior_sampler_converged": newtonian_sampler_converged,
            "candidate_retained_after_tested_nuisance_marginalization": (
                candidate_score < newtonian_score
            ),
            "full_source_covariance_complete": False,
            "independent_replication": False,
            "unique_physical_cause_identified": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
        },
        "limitations": [
            "The Sobol program assigns bounded continuous development priors; it does not replace independently calibrated source priors.",
            "Covariance matrices are stress models, not released or reconstructed instrument/source covariance.",
            "Morphology, relaxation, cool-core, and merger labels are not yet frozen, so merger-state comparison remains open.",
            "The nuisance-only predictive interval excludes source covariance and model discrepancy and is not a full confidence interval.",
            "All response weights use development-training rows; holdout rows are scored but never used for sample weighting.",
        ],
        "next_action": "Acquire source covariance and frozen morphology labels, then rerun this exact nuisance program before independent target authorization.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterUncertaintyError("uncertainty receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = _load_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "claims": receipt["claims"],
            "marginalization": receipt["marginalization"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
