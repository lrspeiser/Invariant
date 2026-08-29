"""Matched-flexibility development comparators for the Item 59 cluster relation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
from scipy.special import gammainc

from sigma_theory_compiler import gravity_item59_xcop_forward_observable_gate as item59
from sigma_theory_compiler.gravity_g4_first_principles_mechanism_search import (
    _kernel_matrix,
    _log_radius_cell_widths,
)

CONFIG_PATH = Path("configs/gravity_cluster_comparator_suite_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/comparator-suite-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-comparator-suite-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-comparator-suite-receipt-1.0"
PARAMETRIC_MODELS = ("GR_PLUS_NFW", "GR_PLUS_EINASTO", "WRONG_REVERSED_NFW")
EMPIRICAL_MODELS = (
    "GNFW_PRESSURE_SHAPE",
    "FLEXIBLE_HSE_MONOTONE_PRESSURE_RECONSTRUCTION",
    "GENERIC_RBF_PRESSURE_CEILING",
)
ABLATIONS = (
    "REMOVE_INTERIOR_KERNEL_CHANNEL",
    "REMOVE_SYMMETRIC_KERNEL_CHANNEL",
    "REMOVE_OCCUPANCY_TRANSITION",
    "REMOVE_BOTH_NONLOCAL_CHANNELS",
)


class GravityClusterComparatorError(RuntimeError):
    """Raised when a comparator, split, or development-only boundary changes."""


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
        raise GravityClusterComparatorError(f"{label} keys changed")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterComparatorError(f"expected JSON object: {path}")
    return value


def _validate_source_bindings(root: Path, config: Mapping[str, Any]) -> None:
    for label, binding in config["source_bindings"].items():
        expected = {"path", "file_sha256", "content_sha256"} if label == "item59_result" else {
            "path",
            "file_sha256",
        }
        _strict(binding, expected, f"{label} binding")
        path = (root / str(binding["path"])).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise GravityClusterComparatorError("source binding escaped root") from error
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterComparatorError(f"bound source changed: {label}")
        if "content_sha256" in binding:
            value = _load_json(path)
            if value.get("content_sha256") != binding["content_sha256"]:
                raise GravityClusterComparatorError(f"bound content changed: {label}")


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
            "suite_id",
            "source_bindings",
            "sample_contract",
            "candidate",
            "shared_nuisance_grid",
            "parametric_gravity_models",
            "empirical_models",
            "candidate_ablations",
            "complexity_accounting",
            "scoring_freeze",
            "claim_boundary",
            "output_path",
        },
        "comparator config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_before_independent_cluster_response_access"
        or config["suite_id"] != "gravity-cluster-comparator-suite-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterComparatorError("comparator config identity changed")
    _validate_source_bindings(root, config)
    sample = config["sample_contract"]
    if (
        len(sample["clusters"]) != 8
        or sample["allowed_splits"] != ["development_train", "development_holdout"]
        or sample["xcop_confirmation_rows_used"] is not False
        or sample["independent_source_rows_used"] is not False
        or sample["inferred_total_mass_used_as_target"] is not False
        or sample["outer_pressure_boundary_scored"] is not False
    ):
        raise GravityClusterComparatorError("development-only sample boundary changed")
    candidate = config["candidate"]
    if (
        candidate["variant_id"] != "cross_scale_boundary:5e945be899287b75"
        or candidate["parameters"] != {"beta": 1.5}
        or candidate["refit"] is not False
        or candidate["explicit_gravity_parameters"] != 1
        or candidate["selected_nuisance_dimensions"] != 4
        or candidate["discrete_variants_originally_screened"] != 2025
    ):
        raise GravityClusterComparatorError("candidate comparator binding changed")
    nuisance = config["shared_nuisance_grid"]
    if (
        nuisance["global_not_per_cluster"] is not True
        or any(len(nuisance[key]) != 3 for key in nuisance if key != "global_not_per_cluster")
    ):
        raise GravityClusterComparatorError("shared nuisance grid changed")
    models = config["parametric_gravity_models"]
    if tuple(row["model_id"] for row in models) != PARAMETRIC_MODELS:
        raise GravityClusterComparatorError("parametric comparator order changed")
    if [row["physical"] for row in models] != [True, True, False]:
        raise GravityClusterComparatorError("physical comparator labels changed")
    empirical = config["empirical_models"]
    if tuple(empirical[key]["model_id"] for key in ("gnfw", "flexible_hydrostatic", "rbf_ceiling")) != EMPIRICAL_MODELS:
        raise GravityClusterComparatorError("empirical comparator inventory changed")
    if tuple(config["candidate_ablations"]) != ABLATIONS:
        raise GravityClusterComparatorError("candidate ablation inventory changed")
    scoring = config["scoring_freeze"]
    if (
        scoring["information_criteria"] != ["AIC", "BIC"]
        or scoring["information_criteria_split"] != "development_train"
        or scoring["predictive_comparison_split"] != "development_holdout"
        or "not_reported" not in scoring["bayesian_evidence"]
    ):
        raise GravityClusterComparatorError("comparator scoring changed")
    claims = config["claim_boundary"]
    if claims["development_comparison_only"] is not True or any(
        claims[key]
        for key in claims
        if key != "development_comparison_only"
    ):
        raise GravityClusterComparatorError("comparator claim boundary weakened")


def _development_packets(root: Path, config59: Mapping[str, Any]) -> list[dict[str, Any]]:
    if not item59._source_receipt_valid(root, config59):
        raise GravityClusterComparatorError("Item 59 source receipt is invalid")
    names = list(map(str, config59["population"]["development_clusters_already_exposed"]))
    packets = [item59._parse_cluster(root, config59, name) for name in names]
    radial = config59["radial_split"]
    minimum_train = int(radial["minimum_train_rows_per_cluster_observable"])
    minimum_holdout = int(radial["minimum_holdout_rows_per_cluster_observable"])
    salt = str(radial["salt"])
    for packet in packets:
        density_radius = np.asarray(packet["density_radius_kpc"])
        pressure_radius = np.asarray(packet["pressure_radius_kpc"])
        usable_pressure = np.flatnonzero(
            (pressure_radius >= density_radius[0]) & (pressure_radius <= density_radius[-1])
        )
        anchor_index = int(usable_pressure[-1])
        anchor_radius = float(pressure_radius[anchor_index])
        packet["anchor"] = {
            "index": anchor_index,
            "radius_kpc": anchor_radius,
            "pressure_kev_cm3": float(packet["pressure_kev_cm3"][anchor_index]),
            "error_kev_cm3": float(packet["pressure_error_kev_cm3"][anchor_index]),
        }
        definitions = {
            "pressure": (
                np.asarray(packet["pressure_radius_kpc"]),
                np.asarray(packet["pressure_kev_cm3"]),
                np.asarray(packet["pressure_error_kev_cm3"]),
                [int(index) for index in usable_pressure[:-1]],
            ),
            "temperature": (
                np.asarray(packet["temperature_radius_kpc"]),
                np.asarray(packet["temperature_kev"]),
                np.asarray(packet["temperature_error_kev"]),
                [
                    int(index)
                    for index, radius in enumerate(packet["temperature_radius_kpc"])
                    if density_radius[0] <= float(radius) < anchor_radius
                ],
            ),
        }
        rows = []
        for observable, (radii, values, errors, indices) in definitions.items():
            ordered = sorted(
                indices,
                key=lambda index: item59._split_order(
                    str(packet["cluster"]), observable, index, salt
                ),
            )
            train_count = round(len(indices) * float(radial["development_train_fraction"]))
            train_count = max(
                minimum_train,
                min(len(indices) - minimum_holdout, train_count),
            )
            training = set(ordered[:train_count])
            for index in indices:
                rows.append(
                    {
                        "row_id": f"{packet['cluster']}:{observable}:{index}",
                        "cluster": str(packet["cluster"]),
                        "observable": observable,
                        "index": index,
                        "radius_kpc": float(radii[index]),
                        "observed": float(values[index]),
                        "error": float(errors[index]),
                        "split": (
                            "development_train"
                            if index in training
                            else "development_holdout"
                        ),
                    }
                )
        packet["rows"] = sorted(rows, key=lambda row: str(row["row_id"]))
    if len(packets) != 8:
        raise GravityClusterComparatorError("development packet count changed")
    return packets


def _nuisance_rows(config: Mapping[str, Any]) -> list[dict[str, float]]:
    grid = config["shared_nuisance_grid"]
    keys = (
        "outer_nonthermal_fraction",
        "published_stellar_mass_scale",
        "missing_stellar_to_gas_mass_ratio",
        "xray_temperature_cross_calibration",
    )
    return [
        {key: float(value) for key, value in zip(keys, values, strict=True)}
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _state(
    packet: Mapping[str, Any], nuisances: Mapping[str, float], config59: Mapping[str, Any]
) -> dict[str, np.ndarray]:
    constants = config59["constants"]
    density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
    ne = np.maximum(np.asarray(packet["ne_cm3"], dtype=float), np.finfo(float).tiny)
    anchor_radius = float(packet["anchor"]["radius_kpc"])
    calc_radius = np.unique(
        np.asarray(
            [
                *density_radius[
                    (density_radius >= density_radius[0]) & (density_radius <= anchor_radius)
                ],
                *[float(row["radius_kpc"]) for row in packet["rows"]],
                anchor_radius,
            ],
            dtype=float,
        )
    )
    calc_ne = np.exp(np.interp(np.log(calc_radius), np.log(density_radius), np.log(ne)))
    radius_m = calc_radius * float(constants["kiloparsec_m"])
    rho = (
        calc_ne
        * 1.0e6
        * float(constants["mean_molecular_weight_per_electron"])
        * float(constants["proton_mass_kg"])
    )
    gas_mass = item59._cumulative_mass(radius_m, rho)
    member_mass = item59._member_mass(
        packet,
        calc_radius,
        gas_mass,
        {"nuisances": nuisances},
        "nominal",
        config59,
    )
    return {
        "calc_radius": calc_radius,
        "density_radius": density_radius,
        "ne": ne,
        "calc_ne": calc_ne,
        "radius_m": radius_m,
        "gas_mass": gas_mass,
        "member_mass": member_mass,
    }


def _predictions_from_acceleration(
    packet: Mapping[str, Any],
    state: Mapping[str, np.ndarray],
    acceleration: np.ndarray,
    nuisances: Mapping[str, float],
    config59: Mapping[str, Any],
) -> dict[str, float]:
    constants = config59["constants"]
    calc_radius = state["calc_radius"]
    radius_m = state["radius_m"]
    thermal_fraction = 1.0 - float(nuisances["outer_nonthermal_fraction"]) * (
        calc_radius / float(packet["r500_kpc"])
    ) ** float(config59["nuisance_grid"]["nonthermal_radial_power"])
    thermal_fraction = np.clip(thermal_fraction, 0.25, 1.0)
    gradient = (
        float(constants["mean_molecular_weight"])
        * float(constants["proton_mass_kg"])
        * state["calc_ne"]
        * 1.0e6
        * acceleration
        * thermal_fraction
    )
    integral = np.zeros_like(radius_m)
    for index in range(len(radius_m) - 2, -1, -1):
        integral[index] = integral[index + 1] + 0.5 * (
            gradient[index + 1] + gradient[index]
        ) * (radius_m[index + 1] - radius_m[index])
    pressure = float(packet["anchor"]["pressure_kev_cm3"]) + integral / float(
        constants["kev_per_cubic_centimeter_j_per_cubic_meter"]
    )
    predictions = {}
    for row in packet["rows"]:
        radius = float(row["radius_kpc"])
        pressure_value = float(np.interp(radius, calc_radius, pressure))
        if row["observable"] == "pressure":
            value = pressure_value
        else:
            ne_value = float(
                np.exp(
                    np.interp(
                        np.log(radius), np.log(state["density_radius"]), np.log(state["ne"])
                    )
                )
            )
            value = pressure_value / ne_value * float(
                nuisances["xray_temperature_cross_calibration"]
            )
        if not math.isfinite(value) or value <= 0.0:
            raise GravityClusterComparatorError("comparator emitted invalid observable")
        predictions[str(row["row_id"])] = value
    return predictions


def _halo_fraction(model_id: str, x: np.ndarray, parameters: Mapping[str, float]) -> np.ndarray:
    c500 = float(parameters["c500"])
    if model_id in {"GR_PLUS_NFW", "WRONG_REVERSED_NFW"}:
        safe_x = np.maximum(x, 1.0e-6)
        if model_id == "WRONG_REVERSED_NFW":
            safe_x = 1.0 / safe_x
        argument = c500 * safe_x
        numerator = np.log1p(argument) - argument / (1.0 + argument)
        denominator = math.log1p(c500) - c500 / (1.0 + c500)
        return numerator / denominator
    alpha = float(parameters["einasto_alpha"])
    s = 3.0 / alpha
    argument = (2.0 / alpha) * np.maximum(c500 * x, 1.0e-12) ** alpha
    normalization = float(gammainc(s, (2.0 / alpha) * c500**alpha))
    return np.asarray(gammainc(s, argument), dtype=float) / normalization


def _gravity_model_predictions(
    packets: Sequence[Mapping[str, Any]],
    model_id: str,
    parameters: Mapping[str, float],
    nuisances: Mapping[str, float],
    config59: Mapping[str, Any],
) -> dict[str, float]:
    predictions: dict[str, float] = {}
    constants = config59["constants"]
    halo_m500 = 10.0 ** float(parameters["log10_halo_m500_solar_mass"]) * float(
        constants["solar_mass_kg"]
    )
    for packet in packets:
        state = _state(packet, nuisances, config59)
        fraction = _halo_fraction(
            model_id,
            state["calc_radius"] / float(packet["r500_kpc"]),
            parameters,
        )
        acceleration = (
            float(constants["gravity_si"])
            * (state["gas_mass"] + state["member_mass"] + halo_m500 * fraction)
            / np.maximum(state["radius_m"] ** 2, np.finfo(float).tiny)
        )
        predictions.update(
            _predictions_from_acceleration(
                packet, state, acceleration, nuisances, config59
            )
        )
    return predictions


def _parameter_rows(model: Mapping[str, Any]) -> list[dict[str, float]]:
    grid = model["parameter_grid"]
    keys = list(grid)
    return [
        {key: float(value) for key, value in zip(keys, values, strict=True)}
        for values in itertools.product(*(grid[key] for key in keys))
    ]


def _score(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    return item59._score_predictions(packets, predictions, split, config59)


def _likelihood_information(
    packets: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    split: str,
    k: float,
    config59: Mapping[str, Any],
) -> dict[str, float | int]:
    rows = item59._rows(packets, split)
    floor = float(config59["scoring"]["minimum_fractional_error"])
    log_likelihood = 0.0
    for row in rows:
        sigma = max(float(row["error"]) / float(row["observed"]), floor)
        residual = math.log(float(predictions[str(row["row_id"])]) / float(row["observed"]))
        log_likelihood += -0.5 * (
            (residual / sigma) ** 2 + math.log(2.0 * math.pi * sigma**2)
        )
    n = len(rows)
    return {
        "rows": n,
        "k": float(k),
        "log_likelihood": float(log_likelihood),
        "AIC": float(2.0 * k - 2.0 * log_likelihood),
        "BIC": float(k * math.log(n) - 2.0 * log_likelihood),
    }


def _public_score(score: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in score.items() if key != "per_row"}


def _select_parametric(
    packets: Sequence[Mapping[str, Any]],
    model: Mapping[str, Any],
    nuisances: Sequence[Mapping[str, float]],
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    best: tuple[float, str, dict[str, Any]] | None = None
    evaluated = 0
    for parameters in _parameter_rows(model):
        for nuisance in nuisances:
            predictions = _gravity_model_predictions(
                packets, str(model["model_id"]), parameters, nuisance, config59
            )
            score = _score(packets, predictions, "development_train", config59)
            payload = {"parameters": parameters, "nuisances": dict(nuisance)}
            tie = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            row = (float(score["score"]), tie, {**payload, "predictions": predictions})
            if best is None or row[:2] < best[:2]:
                best = row
            evaluated += 1
    if best is None:
        raise GravityClusterComparatorError("empty parametric comparator grid")
    selected = best[2]
    selected["evaluated_variants"] = evaluated
    return selected


def _shape_predictions(
    packets: Sequence[Mapping[str, Any]],
    pressure_function: Any,
    cross_calibration: float,
) -> dict[str, float]:
    predictions = {}
    for packet in packets:
        density_radius = np.asarray(packet["density_radius_kpc"], dtype=float)
        ne = np.asarray(packet["ne_cm3"], dtype=float)
        for row in packet["rows"]:
            radius = float(row["radius_kpc"])
            pressure = float(pressure_function(packet, radius))
            if row["observable"] == "pressure":
                value = pressure
            else:
                ne_value = float(
                    np.exp(np.interp(np.log(radius), np.log(density_radius), np.log(ne)))
                )
                value = pressure / ne_value * cross_calibration
            if not math.isfinite(value) or value <= 0.0:
                raise GravityClusterComparatorError("shape comparator emitted invalid value")
            predictions[str(row["row_id"])] = value
    return predictions


def _gnfw_predictions(
    packets: Sequence[Mapping[str, Any]], shape: Mapping[str, float], cross: float
) -> dict[str, float]:
    def pressure(packet: Mapping[str, Any], radius: float) -> float:
        c500 = float(shape["c500"])
        alpha = float(shape["alpha"])
        beta = float(shape["beta"])
        gamma = float(shape["gamma"])

        def profile(value: float) -> float:
            x = max(c500 * value / float(packet["r500_kpc"]), 1.0e-12)
            return x ** (-gamma) * (1.0 + x**alpha) ** (-(beta - gamma) / alpha)

        anchor = float(packet["anchor"]["radius_kpc"])
        return float(packet["anchor"]["pressure_kev_cm3"]) * profile(radius) / profile(
            anchor
        )

    return _shape_predictions(packets, pressure, cross)


def _select_gnfw(
    packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any], config59: Mapping[str, Any]
) -> dict[str, Any]:
    best = None
    evaluated = 0
    for shape in config["empirical_models"]["gnfw"]["shape_grid"]:
        for cross in config["shared_nuisance_grid"]["xray_temperature_cross_calibration"]:
            predictions = _gnfw_predictions(packets, shape, float(cross))
            score = _score(packets, predictions, "development_train", config59)
            tie = json.dumps([shape, cross], sort_keys=True, separators=(",", ":"))
            row = (
                float(score["score"]),
                tie,
                {
                    "parameters": {key: float(value) for key, value in shape.items()},
                    "xray_temperature_cross_calibration": float(cross),
                    "predictions": predictions,
                },
            )
            if best is None or row[:2] < best[:2]:
                best = row
            evaluated += 1
    selected = best[2]
    selected["evaluated_variants"] = evaluated
    return selected


def _hse_pressure_functions(
    packets: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], int, int]:
    functions = {}
    fitted_values = 0
    positive_gradient_violations = 0
    for packet in packets:
        training = sorted(
            [
                row
                for row in packet["rows"]
                if row["split"] == "development_train" and row["observable"] == "pressure"
            ],
            key=lambda row: float(row["radius_kpc"]),
        )
        x = [math.log(float(row["radius_kpc"])) for row in training]
        y = [math.log(float(row["observed"])) for row in training]
        x.append(math.log(float(packet["anchor"]["radius_kpc"])))
        y.append(math.log(float(packet["anchor"]["pressure_kev_cm3"])))
        order = np.argsort(np.asarray(x))
        xx = np.asarray(x, dtype=float)[order]
        yy = np.asarray(y, dtype=float)[order]
        yy = np.minimum.accumulate(yy)
        slopes = np.diff(yy) / np.diff(xx)
        positive_gradient_violations += int(np.sum(slopes > 1.0e-12))
        functions[str(packet["cluster"])] = (xx, yy)
        fitted_values += len(training)
    return functions, fitted_values, positive_gradient_violations


def _select_hse(
    packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any], config59: Mapping[str, Any]
) -> dict[str, Any]:
    functions, fitted_values, violations = _hse_pressure_functions(packets)

    def pressure(packet: Mapping[str, Any], radius: float) -> float:
        x, y = functions[str(packet["cluster"])]
        return float(np.exp(np.interp(math.log(radius), x, y)))

    candidates = []
    for cross in config["shared_nuisance_grid"]["xray_temperature_cross_calibration"]:
        predictions = _shape_predictions(packets, pressure, float(cross))
        score = _score(packets, predictions, "development_train", config59)
        candidates.append((float(score["score"]), float(cross), predictions))
    _, cross, predictions = min(candidates, key=lambda row: (row[0], row[1]))
    return {
        "xray_temperature_cross_calibration": cross,
        "predictions": predictions,
        "fitted_pressure_values": fitted_values,
        "positive_mass_gradient_violations": violations,
        "evaluated_variants": len(candidates),
    }


def _rbf_design(x: np.ndarray, centers: np.ndarray, length: float) -> np.ndarray:
    basis = np.exp(-0.5 * ((x[:, None] - centers[None, :]) / length) ** 2)
    boundary = np.exp(-0.5 * (centers / length) ** 2)
    return basis - boundary[None, :]


def _rbf_pressure(
    packet: Mapping[str, Any],
    radius: float,
    *,
    centers: np.ndarray,
    length: float,
    coefficients: np.ndarray,
) -> float:
    x = np.asarray(
        [math.log(radius / float(packet["anchor"]["radius_kpc"]))], dtype=float
    )
    log_ratio = float(_rbf_design(x, centers, length)[0] @ coefficients)
    return float(packet["anchor"]["pressure_kev_cm3"]) * math.exp(log_ratio)


def _select_rbf(
    packets: Sequence[Mapping[str, Any]], config: Mapping[str, Any], config59: Mapping[str, Any]
) -> dict[str, Any]:
    rbf = config["empirical_models"]["rbf_ceiling"]
    centers = np.asarray(rbf["centers_log_r_over_r_boundary"], dtype=float)
    training = [
        (packet, row)
        for packet in packets
        for row in packet["rows"]
        if row["split"] == "development_train" and row["observable"] == "pressure"
    ]
    counts = {
        str(packet["cluster"]): sum(
            str(other[0]["cluster"]) == str(packet["cluster"]) for other in training
        )
        for packet in packets
    }
    x = np.asarray(
        [
            math.log(float(row["radius_kpc"]) / float(packet["anchor"]["radius_kpc"]))
            for packet, row in training
        ],
        dtype=float,
    )
    target = np.asarray(
        [
            math.log(
                float(row["observed"]) / float(packet["anchor"]["pressure_kev_cm3"])
            )
            for packet, row in training
        ],
        dtype=float,
    )
    weights = np.asarray(
        [1.0 / counts[str(packet["cluster"])] for packet, _row in training], dtype=float
    )
    best = None
    evaluated = 0
    for length in rbf["length_scale_grid"]:
        design = _rbf_design(x, centers, float(length))
        weighted = design * np.sqrt(weights)[:, None]
        response = target * np.sqrt(weights)
        for ridge in rbf["ridge_grid"]:
            normal = weighted.T @ weighted + float(ridge) * np.eye(len(centers))
            coefficients = np.linalg.solve(normal, weighted.T @ response)
            effective_df = float(np.trace(weighted @ np.linalg.solve(normal, weighted.T)))

            pressure = partial(
                _rbf_pressure,
                centers=centers,
                length=float(length),
                coefficients=coefficients.copy(),
            )

            for cross in config["shared_nuisance_grid"][
                "xray_temperature_cross_calibration"
            ]:
                predictions = _shape_predictions(packets, pressure, float(cross))
                score = _score(packets, predictions, "development_train", config59)
                tie = json.dumps([length, ridge, cross], separators=(",", ":"))
                row = (
                    float(score["score"]),
                    tie,
                    {
                        "length_scale": float(length),
                        "ridge": float(ridge),
                        "xray_temperature_cross_calibration": float(cross),
                        "coefficients": [float(value) for value in coefficients],
                        "effective_df": effective_df,
                        "predictions": predictions,
                    },
                )
                if best is None or row[:2] < best[:2]:
                    best = row
                evaluated += 1
    selected = best[2]
    selected["evaluated_variants"] = evaluated
    return selected


def _ablation_predictions(
    packets: Sequence[Mapping[str, Any]],
    ablation: str,
    candidate: Mapping[str, Any],
    config59: Mapping[str, Any],
) -> dict[str, float]:
    predictions = {}
    nuisances = candidate["nuisances"]
    beta = float(candidate["parameters"]["beta"])
    a0 = float(config59["constants"]["transition_acceleration_m_s2"])
    for packet in packets:
        state = _state(packet, nuisances, config59)
        gbar = (
            float(config59["constants"]["gravity_si"])
            * (state["gas_mass"] + state["member_mass"])
            / np.maximum(state["radius_m"] ** 2, np.finfo(float).tiny)
        )
        log_radius = np.log(state["calc_radius"])
        widths = _log_radius_cell_widths(log_radius)
        interior = _kernel_matrix(log_radius, widths, "interior_exponential", 0.25)
        symmetric = _kernel_matrix(log_radius, widths, "symmetric_exponential", 0.25)
        occupancy = (gbar / a0) / (gbar / a0 + 0.1)
        interior_component = gbar * (interior @ occupancy)
        symmetric_component = a0 * (symmetric @ occupancy)
        if ablation == "REMOVE_INTERIOR_KERNEL_CHANNEL":
            acceleration = gbar + beta * symmetric_component
        elif ablation == "REMOVE_SYMMETRIC_KERNEL_CHANNEL":
            acceleration = gbar + beta * interior_component
        elif ablation == "REMOVE_OCCUPANCY_TRANSITION":
            ones = np.ones_like(occupancy)
            acceleration = gbar + beta * (
                gbar * (interior @ ones) + a0 * (symmetric @ ones)
            )
        elif ablation == "REMOVE_BOTH_NONLOCAL_CHANNELS":
            acceleration = gbar
        else:
            raise GravityClusterComparatorError(f"unknown ablation: {ablation}")
        predictions.update(
            _predictions_from_acceleration(
                packet, state, acceleration, nuisances, config59
            )
        )
    return predictions


def _summarize_model(
    model_id: str,
    selection: Mapping[str, Any],
    packets: Sequence[Mapping[str, Any]],
    k: float,
    boundary_count: int,
    config59: Mapping[str, Any],
) -> dict[str, Any]:
    predictions = selection["predictions"]
    parameters = {
        key: value
        for key, value in selection.items()
        if key not in {"predictions"}
    }
    return {
        "model_id": model_id,
        "selection": parameters,
        "training": _public_score(
            _score(packets, predictions, "development_train", config59)
        ),
        "holdout": _public_score(
            _score(packets, predictions, "development_holdout", config59)
        ),
        "complexity": {
            "information_criterion_k": float(k),
            "conditional_boundary_observations": boundary_count,
            "discrete_variants_screened": int(selection["evaluated_variants"]),
        },
        "information_criteria": _likelihood_information(
            packets, predictions, "development_train", k, config59
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    config59 = item59.load_config(root)
    item59_result = _load_json(root / config["source_bindings"]["item59_result"]["path"])
    if item59_result["selection"]["selected_qualifying"]["variant"][
        "variant_id"
    ] != config["candidate"]["variant_id"]:
        raise GravityClusterComparatorError("selected Item 59 candidate changed")
    packets = _development_packets(root, config59)
    if [packet["cluster"] for packet in packets] != config["sample_contract"]["clusters"]:
        raise GravityClusterComparatorError("comparator cluster order changed")
    boundary_count = len(packets)
    nuisance_rows = _nuisance_rows(config)

    candidate_variant = {
        "family_id": config["candidate"]["family_id"],
        "parameters": config["candidate"]["parameters"],
        "nuisances": config["candidate"]["nuisances"],
    }
    candidate_predictions = item59._variant_predictions(
        packets, candidate_variant, config59
    )
    candidate_selection = {
        "parameters": config["candidate"]["parameters"],
        "nuisances": config["candidate"]["nuisances"],
        "refit": False,
        "evaluated_variants": config["candidate"]["discrete_variants_originally_screened"],
        "predictions": candidate_predictions,
    }
    candidate = _summarize_model(
        "ITEM59_CROSS_SCALE_BOUNDARY",
        candidate_selection,
        packets,
        5.0,
        boundary_count,
        config59,
    )

    models = {}
    for model in config["parametric_gravity_models"]:
        selection = _select_parametric(packets, model, nuisance_rows, config59)
        k = float(model["explicit_model_parameters"] + 4)
        models[str(model["model_id"])] = _summarize_model(
            str(model["model_id"]), selection, packets, k, boundary_count, config59
        )
    gnfw = _select_gnfw(packets, config, config59)
    models["GNFW_PRESSURE_SHAPE"] = _summarize_model(
        "GNFW_PRESSURE_SHAPE", gnfw, packets, 5.0, boundary_count, config59
    )
    hse = _select_hse(packets, config, config59)
    hse_k = float(hse["fitted_pressure_values"] + 1)
    models["FLEXIBLE_HSE_MONOTONE_PRESSURE_RECONSTRUCTION"] = _summarize_model(
        "FLEXIBLE_HSE_MONOTONE_PRESSURE_RECONSTRUCTION",
        hse,
        packets,
        hse_k,
        boundary_count,
        config59,
    )
    rbf = _select_rbf(packets, config, config59)
    models["GENERIC_RBF_PRESSURE_CEILING"] = _summarize_model(
        "GENERIC_RBF_PRESSURE_CEILING", rbf, packets, 6.0, boundary_count, config59
    )

    ablations = {}
    for ablation in ABLATIONS:
        predictions = _ablation_predictions(packets, ablation, config["candidate"], config59)
        ablations[ablation] = _summarize_model(
            ablation,
            {
                "refit": False,
                "evaluated_variants": 1,
                "predictions": predictions,
            },
            packets,
            5.0,
            boundary_count,
            config59,
        )

    conventional_ids = ("GR_PLUS_NFW", "GR_PLUS_EINASTO")
    strongest_conventional = min(
        conventional_ids,
        key=lambda name: (float(models[name]["holdout"]["score"]), name),
    )
    generic_ids = (
        "GNFW_PRESSURE_SHAPE",
        "FLEXIBLE_HSE_MONOTONE_PRESSURE_RECONSTRUCTION",
        "GENERIC_RBF_PRESSURE_CEILING",
    )
    strongest_generic = min(
        generic_ids,
        key=lambda name: (float(models[name]["holdout"]["score"]), name),
    )
    candidate_score = float(candidate["holdout"]["score"])
    all_ranked = sorted(
        [("ITEM59_CROSS_SCALE_BOUNDARY", candidate_score)]
        + [(name, float(row["holdout"]["score"])) for name, row in models.items()],
        key=lambda row: (row[1], row[0]),
    )
    candidate_rank = next(
        index for index, row in enumerate(all_ranked, start=1) if row[0] == candidate["model_id"]
    )
    conventional_win = candidate_score < float(models[strongest_conventional]["holdout"]["score"])
    generic_win = candidate_score < float(models[strongest_generic]["holdout"]["score"])
    decision = (
        "CANDIDATE_SURVIVES_MATCHED_DEVELOPMENT_COMPARATORS_NOT_INDEPENDENT"
        if conventional_win
        else "CONVENTIONAL_HALO_OUTPERFORMS_CANDIDATE_ON_DEVELOPMENT_HOLDOUT"
    )
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "suite_id": config["suite_id"],
        "decision": decision,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "sample": {
            "clusters": config["sample_contract"]["clusters"],
            "development_train_rows": len(item59._rows(packets, "development_train")),
            "development_holdout_rows": len(item59._rows(packets, "development_holdout")),
            "conditional_outer_pressure_boundaries": boundary_count,
            "xcop_confirmation_rows_used": False,
            "independent_source_rows_used": False,
            "inferred_total_mass_target_rows": 0,
        },
        "candidate": candidate,
        "comparators": models,
        "ablations": ablations,
        "ranking": {
            "holdout_score_ascending": [
                {"model_id": name, "score": score} for name, score in all_ranked
            ],
            "candidate_rank": candidate_rank,
            "strongest_conventional": strongest_conventional,
            "strongest_generic": strongest_generic,
            "candidate_improvement_over_strongest_conventional": 1.0
            - candidate_score / float(models[strongest_conventional]["holdout"]["score"]),
            "candidate_improvement_over_strongest_generic": 1.0
            - candidate_score / float(models[strongest_generic]["holdout"]["score"]),
        },
        "completed_goal_evidence": {
            "CP4.1": "GR_plus_NFW_forward_pressure_and_temperature",
            "CP4.2": "GR_plus_Einasto_forward_pressure_and_temperature",
            "CP4.3": "monotone_training_only_hydrostatic_mass_reconstruction",
            "CP4.4": "GNFW_empirical_pressure_shape",
            "CP4.5": "Item59_Newtonian_and_RAR_controls_remain_bound",
            "CP4.6": "training_only_regularized_RBF_generic_ceiling",
            "CP4.7": "same_two_gravity_parameter_reversed_NFW_wrong_law_control",
            "CP4.8": "four_exact_candidate_ablations_without_refit",
            "CP4.9": "explicit_k_grid_search_boundary_and_RBF_effective_df_reported",
            "CP4.10": "likelihood_radial_holdout_AIC_BIC_and_no_prior_evidence_rule_frozen",
        },
        "counts": {
            "candidate_models": 1,
            "comparators": len(models),
            "physical_halo_comparators": 2,
            "nonphysical_wrong_law_controls": 1,
            "generic_or_empirical_comparators": 3,
            "ablations": len(ablations),
            "candidate_original_variants_screened": 2025,
            "new_parametric_variants_screened": sum(
                int(models[name]["complexity"]["discrete_variants_screened"])
                for name in PARAMETRIC_MODELS
            ),
            "target_rows_opened": 0,
        },
        "claims": {
            "matched_comparator_suite_complete": True,
            "candidate_beats_strongest_conventional_on_development_holdout": conventional_win,
            "candidate_beats_strongest_generic_on_development_holdout": generic_win,
            "independent_replication": False,
            "full_covariance_used": False,
            "physical_mechanism_established": False,
            "alternative_to_gr_established": False,
            "dark_matter_eliminated": False,
        },
        "limitations": [
            "All comparisons use the eight already exposed X-COP development clusters.",
            "Released diagonal errors with a five-percent floor are used; CP5 covariance work remains open.",
            "Halo grids are spherical phenomenological comparators and do not establish object-level halo truth.",
            "AIC and BIC use nominal fitted-dimension counts; discrete grid-search multiplicity and boundary conditioning are reported separately.",
            "No Bayesian evidence is reported because a common defensible prior measure is not frozen.",
        ],
        "next_action": "Carry this unchanged suite into the eventual independent source after CP5, CP6, and CP7 are complete.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterComparatorError("comparator receipt changed")


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
            "ranking": receipt["ranking"],
            "claims": receipt["claims"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
