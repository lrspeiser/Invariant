from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import qmc

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


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def primitive_values(unit: np.ndarray, config: dict[str, object]) -> np.ndarray:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    return lows + unit * (highs - lows)


def unit_values(values: np.ndarray, config: dict[str, object]) -> np.ndarray:
    lows = np.asarray([float(row["low"]) for row in config["continuous_priors"]])
    highs = np.asarray([float(row["high"]) for row in config["continuous_priors"]])
    return (values - lows) / (highs - lows)


def composites(unit: np.ndarray, config: dict[str, object]) -> np.ndarray:
    values = primitive_values(unit, config)
    by_name = dict(zip(uncertainty.PARAMETERS, values, strict=True))
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
    return np.asarray(
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


def transform_many(unit: np.ndarray, config: dict[str, object]) -> np.ndarray:
    flattened = unit.reshape(-1, unit.shape[-1])
    transformed = np.asarray([composites(row, config) for row in flattened])
    return transformed.reshape(*unit.shape[:-1], len(COMPOSITES))


def log_training_predictions(
    unit: np.ndarray,
    packets: list[dict[str, object]],
    family: dict[str, object],
    config: dict[str, object],
    config59: dict[str, object],
    row_ids: list[str],
) -> np.ndarray:
    _likelihood, predictions = uncertainty._evaluate_unit(
        unit, packets, family, config, config59
    )
    return np.log(np.asarray([predictions[row_id] for row_id in row_ids]))


def rank_diagnostic(
    packets: list[dict[str, object]],
    family: dict[str, object],
    config: dict[str, object],
    config59: dict[str, object],
    row_ids: list[str],
) -> dict[str, object]:
    engine = qmc.Sobol(d=len(uncertainty.PARAMETERS), scramble=True, seed=596001)
    anchors = 0.2 + 0.6 * engine.random_base2(m=4)
    rows = []
    evaluations = 0
    for anchor_index, anchor in enumerate(anchors):
        matrix = np.empty((len(row_ids), len(anchor)))
        for dimension in range(len(anchor)):
            low = anchor.copy()
            high = anchor.copy()
            low[dimension] -= 1e-5
            high[dimension] += 1e-5
            matrix[:, dimension] = (
                log_training_predictions(
                    high, packets, family, config, config59, row_ids
                )
                - log_training_predictions(
                    low, packets, family, config, config59, row_ids
                )
            ) / 2e-5
            evaluations += 2
        singular = np.linalg.svd(matrix, compute_uv=False)
        rows.append(
            {
                "anchor": anchor_index,
                "rank_relative_1e_8": int(np.sum(singular > singular[0] * 1e-8)),
                "tenth_relative_singular_value": float(singular[9] / singular[0]),
                "first_null_relative_singular_value": float(
                    singular[10] / singular[0]
                ),
            }
        )
    return {
        "anchors": len(anchors),
        "training_rows_per_anchor": len(row_ids),
        "evaluations": evaluations,
        "ranks": [int(row["rank_relative_1e_8"]) for row in rows],
        "minimum_tenth_relative_singular_value": min(
            float(row["tenth_relative_singular_value"]) for row in rows
        ),
        "maximum_first_null_relative_singular_value": max(
            float(row["first_null_relative_singular_value"]) for row in rows
        ),
        "per_anchor": rows,
    }


def invariance_diagnostic(
    packets: list[dict[str, object]],
    family: dict[str, object],
    config: dict[str, object],
    config59: dict[str, object],
    row_ids: list[str],
) -> dict[str, object]:
    base_values = primitive_values(
        np.full(len(uncertainty.PARAMETERS), 0.5), config
    )
    by_name = dict(zip(uncertainty.PARAMETERS, base_values, strict=True))
    base_unit = unit_values(base_values, config)
    base_predictions = log_training_predictions(
        base_unit, packets, family, config, config59, row_ids
    )
    differences = []
    for scale in np.linspace(0.98, 1.02, 32):
        changed = dict(by_name)
        changed["triaxial_radius_scale"] = scale
        changed["projection_density_scale"] = 1.0 / scale
        changed["bcg_mass_scale"] = scale**2
        changed["xray_temperature_cross_calibration"] = 1.0 / scale
        changed_values = np.asarray(
            [changed[name] for name in uncertainty.PARAMETERS]
        )
        changed_unit = unit_values(changed_values, config)
        changed_predictions = log_training_predictions(
            changed_unit, packets, family, config, config59, row_ids
        )
        differences.append(float(np.max(np.abs(changed_predictions - base_predictions))))
    return {
        "lambda_cases": len(differences),
        "evaluations": 1 + len(differences),
        "maximum_absolute_log_prediction_difference": max(differences),
    }


def posterior_diagnostic(
    smc_path: Path,
    rejuvenated_path: Path,
    config: dict[str, object],
) -> dict[str, object]:
    smc = np.load(smc_path, allow_pickle=False)
    rejuvenated = np.load(rejuvenated_path, allow_pickle=False)
    smc_particles = transform_many(np.asarray(smc["particles"]), config)
    traces = transform_many(np.asarray(rejuvenated["traces"]), config)
    replicates, particles, retained, dimensions = traces.shape
    chains = traces.reshape(replicates * particles, retained, dimensions)
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
    smc_median = np.median(smc_particles.reshape(-1, dimensions), axis=0)
    rejuvenated_median = np.median(pooled, axis=0)
    median_shift = np.divide(
        np.abs(rejuvenated_median - smc_median),
        standard_deviation,
        out=np.zeros_like(standard_deviation),
        where=standard_deviation > np.finfo(float).tiny,
    )
    return {
        "replicates": replicates,
        "particle_chains_per_replicate": particles,
        "retained_snapshots_per_chain": retained,
        "posterior_draws": len(pooled),
        "maximum_rhat": float(np.max(rhat)),
        "minimum_effective_samples": float(np.min(ess)),
        "maximum_standardized_between_replicate_median_spread": float(
            np.max(median_spread)
        ),
        "maximum_standardized_smc_to_rejuvenated_median_shift": float(
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
                "standardized_smc_to_rejuvenated_median_shift": float(
                    median_shift[index]
                ),
            }
            for index, name in enumerate(COMPOSITES)
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smc", type=Path, required=True)
    parser.add_argument("--smc-sha256", required=True)
    parser.add_argument("--rejuvenated", type=Path, required=True)
    parser.add_argument("--rejuvenated-sha256", required=True)
    args = parser.parse_args()
    if sha256(args.smc) != args.smc_sha256:
        raise RuntimeError("SMC source hash changed")
    if sha256(args.rejuvenated) != args.rejuvenated_sha256:
        raise RuntimeError("rejuvenated source hash changed")
    root = Path(__file__).resolve().parents[2]
    config = uncertainty.load_config(root)
    config59 = item59.load_config(root)
    packets = comparators._development_packets(root, config59)
    family = config["candidate_and_control_families"][0]
    row_ids = [
        str(row["row_id"])
        for row in item59._rows(packets, "development_train")
    ]
    result = {
        "composite_coordinates": list(COMPOSITES),
        "rank": rank_diagnostic(
            packets, family, config, config59, row_ids
        ),
        "forward_invariance": invariance_diagnostic(
            packets, family, config, config59, row_ids
        ),
        "posterior": posterior_diagnostic(
            args.smc, args.rejuvenated, config
        ),
        "sample_seal": {
            "likelihood_split": "development_train",
            "holdout_used_for_selection": False,
            "confirmation_rows_used": False,
            "independent_rows_used": False,
            "paid_model_calls": 0,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
