"""G0: freeze and benchmark the real SPARC formula-discovery experiment.

This module does not search for a gravity law.  It fixes the experiment that G1 will use:
the admitted exploration population, contiguous radial holdouts, four preregistered
comparators, candidate-admission rules, and the throughput/error of the actual
candidate-by-radius evaluator on the local GPU.

The NFW lane is a performance ceiling only.  Its fitted values never become formula inputs,
features, targets, or scientific truth.  Confirmation galaxies are present in the published
source asset and are validated by the inherited dataset loader, but no confirmation galaxy is
handed to a baseline or candidate evaluator here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from .gpu_baryonic_interpolation_screen import (
    FAMILY_SIZE,
    _digits_from_ordinals,
    decode_ordinal,
    render_candidate,
)
from .real_data_gravity_confrontation import Galaxy, baryonic_v_squared
from .sigma_core import canonical_json_bytes, canonical_sha256
from .sparc_full_sample import FULL_SPLIT_SALT, Population, assemble

SCHEMA = "invariant-gravity-g0-experiment-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-g0-experiment-config-1.0"
CONFIG_PATH = "configs/gravity_g0_experiment.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_g0_experiment.py"
TEST_PATH = "tests/test_gravity_g0_experiment.py"
OUTPUT_PATH = "runs/gravity/g0-experiment/receipt-v1.json"
DATA_PATH = "configs/sparc_rotation_curves_full_v1.json"


class GravityG0Error(ValueError):
    """The frozen G0 experiment, replay, or receipt is inconsistent."""


@dataclass(frozen=True, slots=True)
class RadialFold:
    """One contiguous held-out block and its complementary training rows."""

    fold_id: int
    holdout: tuple[int, ...]
    training: tuple[int, ...]


def _normalized_file_sha256(path: Path) -> str:
    payload = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(payload).hexdigest()


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise GravityG0Error(f"JSON root is not an object: {path}")
    return value


def _metric(value: float) -> str:
    if not math.isfinite(value):
        raise GravityG0Error("non-finite metric at receipt boundary")
    return format(value, ".12e")


def load_config(root: Path) -> Mapping[str, Any]:
    """Load and validate the immutable experiment contract and its dataset binding."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityG0Error("G0 config schema changed")
    dataset = config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise GravityG0Error("G0 config has no dataset binding")
    data_path = root / str(dataset.get("path"))
    if data_path != root / DATA_PATH:
        raise GravityG0Error("G0 dataset path changed")
    if _file_sha256(data_path) != dataset.get("file_sha256"):
        raise GravityG0Error("G0 dataset file seal changed")
    if canonical_sha256(_load_json(data_path)) != dataset.get("content_sha256"):
        raise GravityG0Error("G0 dataset content seal changed")
    if config["population"]["split_salt"] != FULL_SPLIT_SALT:
        raise GravityG0Error("G0 split salt disagrees with the inherited SPARC split")
    if int(config["throughput_benchmark"]["declared_grammar_size"]) != FAMILY_SIZE:
        raise GravityG0Error("G0 throughput grammar size changed")
    return config


def radial_folds(
    row_count: int, *, maximum_folds: int = 5, minimum_training_rows: int = 3
) -> tuple[RadialFold, ...]:
    """Return deterministic contiguous folds that hold out every radius exactly once."""

    if row_count <= minimum_training_rows:
        raise GravityG0Error(
            f"{row_count} rows cannot leave {minimum_training_rows} training rows"
        )
    fold_count = min(maximum_folds, row_count)
    folds: list[RadialFold] = []
    all_rows = tuple(range(row_count))
    for fold_id in range(fold_count):
        start = (fold_id * row_count) // fold_count
        stop = ((fold_id + 1) * row_count) // fold_count
        holdout = tuple(range(start, stop))
        training = tuple(index for index in all_rows if not start <= index < stop)
        if not holdout or len(training) < minimum_training_rows:
            raise GravityG0Error("radial-fold rule produced an invalid fold")
        folds.append(RadialFold(fold_id=fold_id, holdout=holdout, training=training))
    held = [index for fold in folds for index in fold.holdout]
    if held != list(range(row_count)):
        raise GravityG0Error("radial folds do not hold out every ordered row exactly once")
    return tuple(folds)


def _galaxy_arrays(galaxy: Galaxy) -> dict[str, np.ndarray]:
    vbar2 = np.asarray(
        [float(value) for value in baryonic_v_squared(galaxy, Fraction(1, 2), Fraction(7, 10))],
        dtype=np.float64,
    )
    return {
        "radius": np.asarray([float(value) for value in galaxy.radius], dtype=np.float64),
        "vbar2": vbar2,
        "vobs": np.asarray([float(value) for value in galaxy.v_obs], dtype=np.float64),
        "sigma": np.asarray([float(value) for value in galaxy.e_v_obs], dtype=np.float64),
    }


def _newtonian(vbar2: np.ndarray) -> np.ndarray:
    return np.sqrt(vbar2)


def _empirical_rar(radius: np.ndarray, vbar2: np.ndarray, a0: float) -> np.ndarray:
    gbar = vbar2 / radius
    root = np.sqrt(gbar / a0)
    denominator = -np.expm1(-root)
    return np.sqrt(vbar2 / denominator)


def _wrong_high_acceleration(radius: np.ndarray, vbar2: np.ndarray, a0: float) -> np.ndarray:
    gbar = vbar2 / radius
    return np.sqrt(vbar2 * np.sqrt(1.0 + gbar / a0))


def _nfw_shape(radius: np.ndarray, scale_radius: float) -> np.ndarray:
    x = radius / scale_radius
    return (np.log1p(x) - x / (1.0 + x)) / radius


def _fit_nfw_fold(
    arrays: Mapping[str, np.ndarray], training: Sequence[int], scale_grid_size: int
) -> tuple[float, float, float]:
    """Fit a two-parameter NFW-shaped velocity-squared excess on training radii only."""

    radius = arrays["radius"]
    vbar2 = arrays["vbar2"]
    vobs = arrays["vobs"]
    sigma = arrays["sigma"]
    train = np.asarray(training, dtype=np.int64)
    scales = np.geomspace(float(radius.min()) / 3.0, float(radius.max()) * 3.0, scale_grid_size)
    target = vobs[train] ** 2 - vbar2[train]
    sigma_v2 = 2.0 * vobs[train] * sigma[train] + sigma[train] ** 2
    weights = 1.0 / np.maximum(sigma_v2, np.finfo(np.float64).tiny) ** 2
    best: tuple[float, float, float] | None = None
    for scale in scales:
        shape = _nfw_shape(radius[train], float(scale))
        denominator = float(np.sum(weights * shape * shape))
        amplitude = 0.0 if denominator == 0.0 else max(
            0.0, float(np.sum(weights * shape * target) / denominator)
        )
        prediction = np.sqrt(np.maximum(vbar2[train] + amplitude * shape, 0.0))
        chi_square = float(np.sum(((prediction - vobs[train]) / sigma[train]) ** 2))
        key = (chi_square, float(scale), amplitude)
        if best is None or key < best:
            best = key
    if best is None:
        raise GravityG0Error("NFW comparator grid is empty")
    chi_square, scale, amplitude = best
    return amplitude, scale, chi_square


def _nfw_out_of_fold(
    arrays: Mapping[str, np.ndarray], folds: Sequence[RadialFold], scale_grid_size: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    prediction = np.empty_like(arrays["vobs"])
    fits: list[dict[str, Any]] = []
    for fold in folds:
        amplitude, scale, training_chi_square = _fit_nfw_fold(
            arrays, fold.training, scale_grid_size
        )
        held = np.asarray(fold.holdout, dtype=np.int64)
        shape = _nfw_shape(arrays["radius"][held], scale)
        prediction[held] = np.sqrt(
            np.maximum(arrays["vbar2"][held] + amplitude * shape, 0.0)
        )
        fits.append(
            {
                "amplitude": _metric(amplitude),
                "fold_id": fold.fold_id,
                "held_out_indices": list(fold.holdout),
                "scale_radius_kpc": _metric(scale),
                "training_chi_square": _metric(training_chi_square),
            }
        )
    return prediction, fits


def score_predictions(
    prediction: np.ndarray, observed: np.ndarray, sigma: np.ndarray
) -> dict[str, Any]:
    """Frozen conditional-on-published-errors score used by every G0 comparator."""

    if prediction.shape != observed.shape or observed.shape != sigma.shape:
        raise GravityG0Error("score arrays have different shapes")
    if np.any(~np.isfinite(prediction)) or np.any(prediction < 0):
        raise GravityG0Error("prediction is non-finite or negative")
    standardized = (prediction - observed) / sigma
    absolute = np.abs(standardized)
    return {
        "chi_square": _metric(float(np.sum(standardized**2))),
        "coverage_one_sigma": _metric(float(np.mean(absolute <= 1.0))),
        "coverage_three_sigma": _metric(float(np.mean(absolute <= 3.0))),
        "coverage_two_sigma": _metric(float(np.mean(absolute <= 2.0))),
        "maximum_absolute_standardized_residual": _metric(float(np.max(absolute))),
        "mean_squared_standardized_residual": _metric(float(np.mean(standardized**2))),
        "median_absolute_standardized_residual": _metric(float(statistics.median(absolute))),
        "p90_absolute_standardized_residual": _metric(
            float(np.quantile(absolute, 0.9, method="linear"))
        ),
        "row_count": int(observed.size),
    }


def _prediction_digest(values: np.ndarray) -> str:
    rendered = [format(float(value), ".15e") for value in values]
    return canonical_sha256(rendered)


def baseline_replay(population: Population, config: Mapping[str, Any]) -> dict[str, Any]:
    """Run all four baselines on exploration galaxies and no confirmation galaxy."""

    confirmation = set(population.split.confirmation)
    evaluated_names = [galaxy.name for galaxy in population.exploration]
    if confirmation.intersection(evaluated_names):
        raise GravityG0Error("a confirmation galaxy reached the G0 evaluator")
    if len(evaluated_names) != len(set(evaluated_names)):
        raise GravityG0Error("an exploration galaxy appears twice")

    maximum_folds = int(config["radial_holdout"]["maximum_folds"])
    minimum_training = int(config["radial_holdout"]["minimum_training_rows"])
    a0 = float(
        next(item for item in config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    nfw = next(item for item in config["baselines"] if item["id"] == "nfw_halo_ceiling")
    scale_grid_size = int(nfw["scale_radius_grid_size"])

    all_observed: list[np.ndarray] = []
    all_sigma: list[np.ndarray] = []
    predictions: dict[str, list[np.ndarray]] = {
        "newtonian_baryons": [],
        "empirical_rar": [],
        "wrong_high_acceleration_boost": [],
        "nfw_halo_ceiling": [],
    }
    per_galaxy: list[dict[str, Any]] = []
    fold_count = 0
    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        folds = radial_folds(
            galaxy.count,
            maximum_folds=maximum_folds,
            minimum_training_rows=minimum_training,
        )
        fold_count += len(folds)
        galaxy_predictions = {
            "newtonian_baryons": _newtonian(arrays["vbar2"]),
            "empirical_rar": _empirical_rar(arrays["radius"], arrays["vbar2"], a0),
            "wrong_high_acceleration_boost": _wrong_high_acceleration(
                arrays["radius"], arrays["vbar2"], a0
            ),
        }
        halo_prediction, halo_fits = _nfw_out_of_fold(arrays, folds, scale_grid_size)
        galaxy_predictions["nfw_halo_ceiling"] = halo_prediction
        scores = {
            baseline_id: score_predictions(values, arrays["vobs"], arrays["sigma"])
            for baseline_id, values in galaxy_predictions.items()
        }
        per_galaxy.append(
            {
                "baselines": scores,
                "fold_count": len(folds),
                "galaxy": galaxy.name,
                "nfw_training_fits": halo_fits,
                "point_count": galaxy.count,
                "prediction_sha256": {
                    key: _prediction_digest(value) for key, value in galaxy_predictions.items()
                },
            }
        )
        all_observed.append(arrays["vobs"])
        all_sigma.append(arrays["sigma"])
        for baseline_id, values in galaxy_predictions.items():
            predictions[baseline_id].append(values)

    observed = np.concatenate(all_observed)
    sigma = np.concatenate(all_sigma)
    aggregate = {
        baseline_id: {
            **score_predictions(np.concatenate(values), observed, sigma),
            "prediction_sha256": _prediction_digest(np.concatenate(values)),
        }
        for baseline_id, values in predictions.items()
    }
    return {
        "aggregate": aggregate,
        "confirmation_evaluator_access_count": 0,
        "evaluated_galaxies": len(evaluated_names),
        "evaluated_name_root_sha256": canonical_sha256(sorted(evaluated_names)),
        "evaluated_points": int(observed.size),
        "fold_count": fold_count,
        "per_galaxy": per_galaxy,
    }


def _formula_scores(
    xp: Any,
    ordinals: Any,
    *,
    y: Any,
    vbar2: Any,
    vobs: Any,
    sigma: Any,
    dtype: Any,
    point_chunk_size: int,
) -> Any:
    """Evaluate the declared formula grammar on actual SPARC radii and baryons."""

    beta_index, coefficients = _digits_from_ordinals(xp, ordinals)
    score = xp.zeros(ordinals.shape[0], dtype=dtype)
    valid_candidate = xp.ones(ordinals.shape[0], dtype=bool)
    for start in range(0, int(y.shape[0]), point_chunk_size):
        stop = min(start + point_chunk_size, int(y.shape[0]))
        u = y[start:stop].astype(dtype) ** dtype(-0.5)
        power = xp.ones_like(u, dtype=dtype)
        numerator = xp.ones((ordinals.shape[0], stop - start), dtype=dtype)
        denominator = xp.ones_like(numerator)
        for slot in range(5):
            power = power * u
            numerator += coefficients[:, slot, None].astype(dtype) * power[None, :]
            denominator += coefficients[:, 5 + slot, None].astype(dtype) * power[None, :]
        valid = xp.all(xp.isfinite(denominator) & (xp.abs(denominator) > dtype(1e-12)), axis=1)
        ratio = numerator / xp.where(xp.abs(denominator) > dtype(1e-12), denominator, dtype(1))
        valid &= xp.all(xp.isfinite(ratio) & (ratio > dtype(1e-12)), axis=1)
        safe = xp.where(ratio > dtype(1e-12), ratio, dtype(1))
        nu = xp.where(
            beta_index[:, None] == 0,
            xp.cbrt(safe),
            xp.where(
                beta_index[:, None] == 1,
                xp.sqrt(safe),
                xp.where(beta_index[:, None] == 2, safe, safe * safe),
            ),
        )
        prediction = xp.sqrt(xp.maximum(vbar2[start:stop][None, :] * nu, dtype(0)))
        standardized = (
            prediction - vobs[start:stop][None, :]
        ) / sigma[start:stop][None, :]
        standardized = xp.clip(standardized, dtype(-1e12), dtype(1e12))
        score += xp.sum(standardized * standardized, axis=1)
        valid_candidate &= valid & xp.all(xp.isfinite(prediction), axis=1)
    return xp.where(valid_candidate, score, dtype(np.inf))


def _benchmark_arrays(population: Population, a0: float) -> dict[str, np.ndarray]:
    radius_parts: list[np.ndarray] = []
    vbar_parts: list[np.ndarray] = []
    vobs_parts: list[np.ndarray] = []
    sigma_parts: list[np.ndarray] = []
    for galaxy in population.exploration:
        arrays = _galaxy_arrays(galaxy)
        radius_parts.append(arrays["radius"])
        vbar_parts.append(arrays["vbar2"])
        vobs_parts.append(arrays["vobs"])
        sigma_parts.append(arrays["sigma"])
    radius = np.concatenate(radius_parts)
    vbar2 = np.concatenate(vbar_parts)
    return {
        "sigma": np.concatenate(sigma_parts),
        "vbar2": vbar2,
        "vobs": np.concatenate(vobs_parts),
        "y": (vbar2 / radius) / a0,
    }


def throughput_benchmark(
    population: Population,
    config: Mapping[str, Any],
    *,
    candidate_count: int | None = None,
    use_gpu: bool = True,
) -> dict[str, Any]:
    """Measure the declared grammar on all 2,720 exploration rows."""

    benchmark = config["throughput_benchmark"]
    total = int(benchmark["candidate_count"] if candidate_count is None else candidate_count)
    if not 1 <= total <= FAMILY_SIZE:
        raise GravityG0Error("benchmark candidate count is outside the declared grammar")
    batch_size = int(benchmark["candidate_batch_size"])
    point_chunk_size = int(benchmark["point_chunk_size"])
    crosscheck_count = min(int(benchmark["fp64_crosscheck_candidates"]), total)
    a0 = float(
        next(item for item in config["baselines"] if item["id"] == "empirical_rar")[
            "g_dagger_km2_s2_kpc"
        ]
    )
    arrays = _benchmark_arrays(population, a0)
    point_count = int(arrays["y"].size)

    if use_gpu:
        import cupy as xp

        device_properties = xp.cuda.runtime.getDeviceProperties(0)
        device = device_properties["name"].decode()
        memory_pool = xp.get_default_memory_pool()
        initial_pool_bytes = int(memory_pool.total_bytes())
        peak_pool_bytes = initial_pool_bytes
        y = xp.asarray(arrays["y"])
        vbar2 = xp.asarray(arrays["vbar2"])
        vobs = xp.asarray(arrays["vobs"])
        sigma = xp.asarray(arrays["sigma"])
    else:
        xp = np
        device = "cpu-numpy"
        y = arrays["y"]
        vbar2 = arrays["vbar2"]
        vobs = arrays["vobs"]
        sigma = arrays["sigma"]

    best_rows: list[tuple[float, int]] = []
    finite_candidates = 0
    started = time.perf_counter()
    for start in range(0, total, batch_size):
        stop = min(start + batch_size, total)
        ordinals = xp.arange(start, stop, dtype=xp.int64)
        scores = _formula_scores(
            xp,
            ordinals,
            y=y,
            vbar2=vbar2,
            vobs=vobs,
            sigma=sigma,
            dtype=xp.float32,
            point_chunk_size=point_chunk_size,
        )
        host_scores = scores.get() if use_gpu else scores
        if use_gpu:
            peak_pool_bytes = max(peak_pool_bytes, int(memory_pool.total_bytes()))
        finite = np.isfinite(host_scores)
        finite_candidates += int(np.sum(finite))
        if np.any(finite):
            finite_indices = np.flatnonzero(finite)
            take = min(8, finite_indices.size)
            local = finite_indices[
                np.argpartition(host_scores[finite_indices], take - 1)[:take]
            ]
            best_rows.extend((float(host_scores[index]), start + int(index)) for index in local)
            best_rows = sorted(best_rows)[:16]
    if use_gpu:
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started

    sample = np.linspace(0, total - 1, crosscheck_count, dtype=np.int64)
    if use_gpu:
        gpu64 = _formula_scores(
            xp,
            xp.asarray(sample),
            y=y,
            vbar2=vbar2,
            vobs=vobs,
            sigma=sigma,
            dtype=xp.float64,
            point_chunk_size=point_chunk_size,
        ).get()
    else:
        gpu64 = _formula_scores(
            np,
            sample,
            y=arrays["y"],
            vbar2=arrays["vbar2"],
            vobs=arrays["vobs"],
            sigma=arrays["sigma"],
            dtype=np.float64,
            point_chunk_size=point_chunk_size,
        )
    cpu_started = time.perf_counter()
    cpu64 = _formula_scores(
        np,
        sample,
        y=arrays["y"],
        vbar2=arrays["vbar2"],
        vobs=arrays["vobs"],
        sigma=arrays["sigma"],
        dtype=np.float64,
        point_chunk_size=point_chunk_size,
    )
    cpu_elapsed = time.perf_counter() - cpu_started
    finite_agreement = np.isfinite(gpu64) == np.isfinite(cpu64)
    both_finite = np.isfinite(gpu64) & np.isfinite(cpu64)
    relative = np.zeros_like(cpu64)
    relative[both_finite] = np.abs(gpu64[both_finite] - cpu64[both_finite]) / (
        1.0 + np.abs(cpu64[both_finite])
    )
    tolerance = float(benchmark["fp64_gpu_cpu_maximum_relative_error"])
    mismatches = int(np.sum(~finite_agreement | (relative > tolerance)))
    maximum_relative = float(np.max(relative)) if relative.size else 0.0
    point_evaluations = total * point_count
    return {
        "best_fp32_candidates": [
            {
                "candidate": decode_ordinal(ordinal),
                "chi_square": _metric(score),
                "formula": render_candidate(decode_ordinal(ordinal)),
                "ordinal": ordinal,
            }
            for score, ordinal in best_rows
        ],
        "candidate_count": total,
        "candidates_per_second": _metric(total / elapsed),
        "cpu_fp64_candidate_replays_per_second": _metric(crosscheck_count / cpu_elapsed),
        "cpu_fp64_elapsed_seconds": _metric(cpu_elapsed),
        "cpu_fp64_point_evaluations_per_second": _metric(
            (crosscheck_count * point_count) / cpu_elapsed
        ),
        "device": device,
        "domain_finite_candidate_fraction": _metric(finite_candidates / total),
        "elapsed_seconds": _metric(elapsed),
        "finite_candidate_count": finite_candidates,
        "fp32_point_evaluations": point_evaluations,
        "fp32_point_evaluations_per_second": _metric(point_evaluations / elapsed),
        "fp64_gpu_cpu_crosscheck_candidates": crosscheck_count,
        "fp64_gpu_cpu_decision_mismatches": mismatches,
        "fp64_gpu_cpu_maximum_relative_error": _metric(maximum_relative),
        "grammar_size": FAMILY_SIZE,
        "gpu_memory_pool_peak_increment_bytes": (
            peak_pool_bytes - initial_pool_bytes if use_gpu else 0
        ),
        "point_count": point_count,
        "uses_actual_sparc_radii_baryons_targets_and_sigmas": True,
    }


def _fold_audit(population: Population, config: Mapping[str, Any]) -> dict[str, Any]:
    maximum_folds = int(config["radial_holdout"]["maximum_folds"])
    minimum_training = int(config["radial_holdout"]["minimum_training_rows"])
    rows = []
    total_folds = 0
    total_holdouts = 0
    for galaxy in population.exploration:
        folds = radial_folds(
            galaxy.count,
            maximum_folds=maximum_folds,
            minimum_training_rows=minimum_training,
        )
        total_folds += len(folds)
        total_holdouts += sum(len(fold.holdout) for fold in folds)
        rows.append(
            {
                "fold_count": len(folds),
                "galaxy": galaxy.name,
                "point_count": galaxy.count,
                "radial_fold_root_sha256": canonical_sha256(
                    [
                        {
                            "fold_id": fold.fold_id,
                            "holdout": list(fold.holdout),
                            "training": list(fold.training),
                        }
                        for fold in folds
                    ]
                ),
            }
        )
    return {
        "all_rows_held_out_exactly_once": total_holdouts
        == sum(galaxy.count for galaxy in population.exploration),
        "galaxies": rows,
        "galaxy_count": len(rows),
        "holdout_row_assignments": total_holdouts,
        "total_folds": total_folds,
    }


def _binding(root: Path, relative: str) -> dict[str, str]:
    return {
        "normalized_text_sha256": _normalized_file_sha256(root / relative),
        "path": relative,
    }


def build_receipt(
    root: Path, *, candidate_count: int | None = None, use_gpu: bool = True
) -> dict[str, Any]:
    """Build the G0 receipt.  A full PASS requires the declared million-candidate GPU run."""

    root = root.resolve()
    config = load_config(root)
    population = assemble(root)
    expected = config["population"]
    if len(population.exploration) != int(expected["admitted_exploration_galaxies"]):
        raise GravityG0Error("admitted exploration galaxy count changed")
    if sum(galaxy.count for galaxy in population.exploration) != int(
        expected["admitted_exploration_points"]
    ):
        raise GravityG0Error("admitted exploration point count changed")
    if len(population.split.confirmation) != int(expected["confirmation_galaxies"]):
        raise GravityG0Error("confirmation galaxy count changed")

    folds = _fold_audit(population, config)
    baselines = baseline_replay(population, config)
    benchmark = throughput_benchmark(
        population, config, candidate_count=candidate_count, use_gpu=use_gpu
    )
    aggregate = baselines["aggregate"]
    ordering = {
        "empirical_rar_beats_newtonian": float(aggregate["empirical_rar"]["chi_square"])
        < float(aggregate["newtonian_baryons"]["chi_square"]),
        "empirical_rar_beats_wrong_law": float(aggregate["empirical_rar"]["chi_square"])
        < float(aggregate["wrong_high_acceleration_boost"]["chi_square"]),
        "nfw_ceiling_beats_newtonian": float(aggregate["nfw_halo_ceiling"]["chi_square"])
        < float(aggregate["newtonian_baryons"]["chi_square"]),
    }
    declared_count = int(config["throughput_benchmark"]["candidate_count"])
    required_point_evaluations = int(
        config["throughput_benchmark"]["minimum_measured_point_evaluations"]
    )
    required_device = str(config["throughput_benchmark"]["required_device_substring"])
    full_benchmark = benchmark["candidate_count"] == declared_count
    pass_gate = (
        all(ordering.values())
        and folds["all_rows_held_out_exactly_once"]
        and baselines["confirmation_evaluator_access_count"] == 0
        and benchmark["fp64_gpu_cpu_decision_mismatches"] == 0
        and benchmark["fp32_point_evaluations"] >= required_point_evaluations
        and required_device in benchmark["device"]
        and full_benchmark
        and use_gpu
    )
    decision = "PASS_G0_EXPERIMENT_FROZEN" if pass_gate else "BLOCK_G0_PREFLIGHT_OR_BENCHMARK"
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "G0",
        "decision": decision,
        "baseline_replay": baselines,
        "claims": {
            "alternative_to_gr_discovered": False,
            "confirmation_galaxy_evaluated": False,
            "confirmation_payload_present_in_validated_source_asset": True,
            "dark_matter_used_as_candidate_input_or_target": False,
            "formula_atlas_built": False,
            "g0_mechanics_and_comparators_frozen": pass_gate,
            "nfw_used_as_performance_ceiling_only": True,
            "observational_likelihood_complete": False,
            "published_random_errors_only": True,
            "throughput_transfers_to_other_grammars_without_measurement": False,
        },
        "config": {
            "content_sha256": canonical_sha256(config),
            "path": CONFIG_PATH,
        },
        "counts": {
            "admitted_exploration_galaxies": len(population.exploration),
            "admitted_exploration_points": sum(
                galaxy.count for galaxy in population.exploration
            ),
            "confirmation_evaluator_accesses": 0,
            "confirmation_galaxies": len(population.split.confirmation),
            "published_galaxies": population.provenance["galaxy_count"],
            "published_points": population.provenance["point_count"],
        },
        "data_access": {
            "baseline_and_candidate_evaluator_galaxies": len(population.exploration),
            "confirmation_evaluator_access_count": 0,
            "confirmation_payload_is_not_cryptographically_hidden": True,
            "interpretation": (
                "The published source file contains all 175 galaxies and the inherited loader "
                "validates the full asset. G0 hands only the 139 admitted exploration galaxies "
                "to scoring and throughput functions. This is a code-enforced holdout, not a "
                "claim that confirmation bytes are physically inaccessible on this machine."
            ),
        },
        "formula_search_authorized_after_pass": pass_gate,
        "fold_audit": folds,
        "gate_checks": {
            **ordering,
            "confirmation_evaluator_accesses_zero": baselines[
                "confirmation_evaluator_access_count"
            ]
            == 0,
            "declared_gpu_benchmark_completed": full_benchmark and use_gpu,
            "gpu_cpu_decision_mismatches_zero": benchmark[
                "fp64_gpu_cpu_decision_mismatches"
            ]
            == 0,
            "minimum_point_evaluations_reached": benchmark["fp32_point_evaluations"]
            >= required_point_evaluations,
            "required_gpu_present": required_device in benchmark["device"],
        },
        "limitations": [
            "SPARC e_Vobs excludes systematic inclination uncertainty, so these scores are conditional on published random errors and are not a complete observational likelihood.",
            "The NFW-shaped comparator is a cross-validated two-parameter performance ceiling, not a cosmologically constrained halo inference.",
            "The throughput measurement applies to this typed pointwise grammar and evaluator; kernels, gradients, nonlocal formulas, nuisance integration, and covariant forward models require their own benchmarks.",
            "A G0 PASS freezes experiment mechanics. It is not a formula discovery, evidence against dark matter, or evidence for an alternative to GR.",
        ],
        "ordering_controls": ordering,
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "dataset": {
                "file_sha256": _file_sha256(root / DATA_PATH),
                "path": DATA_PATH,
                "semantic_sha256": canonical_sha256(_load_json(root / DATA_PATH)),
            },
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
        "split": {
            "confirmation_count": len(population.split.confirmation),
            "confirmation_name_root_sha256": canonical_sha256(
                sorted(population.split.confirmation)
            ),
            "exploration_count_before_admission": len(population.split.exploration),
            "exploration_name_root_sha256": canonical_sha256(
                sorted(population.split.exploration)
            ),
            "salt": population.split.salt,
            "unit": "whole_galaxy",
        },
        "throughput_benchmark": benchmark,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Fail closed on receipt, source, config, dataset, or test drift."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityG0Error("G0 receipt schema changed")
    supplied = receipt.get("content_sha256")
    unsigned = dict(receipt)
    unsigned.pop("content_sha256", None)
    if supplied != canonical_sha256(unsigned):
        raise GravityG0Error("G0 receipt content seal changed")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityG0Error("G0 receipt config binding changed")
    bindings = receipt.get("source_bindings")
    if not isinstance(bindings, Mapping):
        raise GravityG0Error("G0 receipt has no source bindings")
    for key, relative in (("config", CONFIG_PATH), ("source", SOURCE_PATH), ("test", TEST_PATH)):
        if bindings.get(key) != _binding(root, relative):
            raise GravityG0Error(f"G0 {key} binding changed")
    dataset = bindings.get("dataset")
    if not isinstance(dataset, Mapping):
        raise GravityG0Error("G0 receipt has no dataset binding")
    if dataset.get("file_sha256") != _file_sha256(root / DATA_PATH):
        raise GravityG0Error("G0 receipt dataset file binding changed")
    if dataset.get("semantic_sha256") != canonical_sha256(_load_json(root / DATA_PATH)):
        raise GravityG0Error("G0 receipt dataset semantic binding changed")
    counts = receipt.get("counts", {})
    if counts.get("admitted_exploration_galaxies") != 139:
        raise GravityG0Error("G0 receipt exploration galaxy count changed")
    if counts.get("admitted_exploration_points") != 2720:
        raise GravityG0Error("G0 receipt exploration point count changed")
    if counts.get("confirmation_evaluator_accesses") != 0:
        raise GravityG0Error("G0 receipt reports confirmation evaluator access")
    benchmark = receipt.get("throughput_benchmark", {})
    if benchmark.get("fp64_gpu_cpu_decision_mismatches") != 0:
        raise GravityG0Error("G0 receipt reports GPU/CPU decision mismatch")
    if receipt.get("decision") != "PASS_G0_EXPERIMENT_FROZEN":
        raise GravityG0Error("checked G0 receipt is not a PASS")
    if receipt.get("formula_search_authorized_after_pass") is not True:
        raise GravityG0Error("checked G0 receipt does not authorize G1")


def write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value)
    if path.exists() and path.read_bytes() != payload:
        raise GravityG0Error(f"refusing to overwrite a different G0 receipt: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--candidate-count", type=int)
    parser.add_argument("--cpu-only", action="store_true")
    parser.add_argument("--validate-checked", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if args.validate_checked:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(
        root, candidate_count=args.candidate_count, use_gpu=not args.cpu_only
    )
    write_immutable(output, receipt)
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt["decision"] == "PASS_G0_EXPERIMENT_FROZEN" else 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "GravityG0Error",
    "RadialFold",
    "baseline_replay",
    "build_receipt",
    "load_config",
    "radial_folds",
    "score_predictions",
    "throughput_benchmark",
    "validate_receipt",
]
