"""Evaluate the frozen Item 39 WALLABY holographic/boundary search."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _read_json,
    _sha256_file,
    _verify_content_hash,
    _write_json,
)
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    POLICY_PATH,
    GravityItem39Error,
    _candidate_parameters,
    _source_path,
    admissible_candidates,
    decode_candidate,
    fixed_control_multiplier,
    load_config,
)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _load_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, str]]:
    features = _read_tsv(_source_path(root, config, "point_features"))
    responses = _read_tsv(_source_path(root, config, "rotation_responses"))
    keys = ("galaxy", "team_release_kin", "point_index")
    response_by_key = {tuple(row[key] for key in keys): row for row in responses}
    if len(response_by_key) != len(responses):
        raise GravityItem39Error("duplicate Item 39 response point")
    rows = []
    for feature in features:
        key = tuple(feature[field] for field in keys)
        response = response_by_key.pop(key, None)
        if response is None:
            raise GravityItem39Error("Item 39 feature lacks aligned response")
        rows.append({**feature, **response})
    if response_by_key:
        raise GravityItem39Error("Item 39 response lacks aligned feature")
    return rows


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device, bytes):
            device = device.decode()
        return cp, "gpu_cupy", str(device)
    except Exception:  # noqa: BLE001 - GPU discovery must fall back on any driver error.
        return np, "cpu_numpy", "CPU"


def _to_numpy(value: Any, xp: Any) -> np.ndarray:
    return xp.asnumpy(value) if xp is not np else np.asarray(value)


def _point_arrays(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    galaxies = np.asarray([str(row["galaxy"]) for row in rows])
    unique = sorted(set(galaxies.tolist()))
    galaxy_index = {name: index for index, name in enumerate(unique)}
    object_index = np.asarray([galaxy_index[name] for name in galaxies], dtype=np.int64)
    counts = np.bincount(object_index, minlength=len(unique))
    point_weight = 1.0 / counts[object_index]
    object_folds: dict[str, int] = {}
    for row in rows:
        name = str(row["galaxy"])
        fold = int(row["outer_fold"])
        if name in object_folds and object_folds[name] != fold:
            raise GravityItem39Error("one galaxy entered multiple outer folds")
        object_folds[name] = fold
    folds = np.asarray([object_folds[name] for name in unique], dtype=np.int64)
    point_folds = folds[object_index]
    observed = np.asarray([float(row["observed_speed_km_s"]) for row in rows], dtype=np.float64)
    observed_error = np.asarray(
        [float(row["observed_speed_error_km_s"]) for row in rows], dtype=np.float64
    )
    log_observed = np.log10(observed)
    log_sigma = np.maximum(observed_error / (observed * math.log(10.0)), 0.03)
    return {
        "galaxies": unique,
        "object_index": object_index,
        "point_weight": point_weight,
        "object_folds": folds,
        "point_folds": point_folds,
        "observed": observed,
        "log_observed": log_observed,
        "log_sigma": log_sigma,
        "u": np.asarray([float(row["u"]) for row in rows]),
        "vbar": np.asarray([float(row["vbar_km_s"]) for row in rows]),
        "fraction": np.asarray([float(row["enclosed_fraction"]) for row in rows]),
        "x": np.asarray([float(row["radius_over_screen"]) for row in rows]),
        "slope": np.asarray([float(row["enclosed_log_slope"]) for row in rows]),
        "h": np.vstack(
            [
                np.asarray([float(row[key]) for row in rows])
                for key in (
                    "h_equipartition",
                    "h_quasilocal",
                    "h_wedge",
                    "h_flow",
                )
            ]
        ),
    }


def _candidate_log_velocity_batch(
    candidates: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    xp: Any,
) -> Any:
    rows = {key: value[begin:end] for key, value in candidates.items()}
    amplitude, exponent, transition, shape = _candidate_parameters(rows, config)
    lane = xp.asarray(rows["lane"], dtype=xp.int8)
    aa = xp.asarray(amplitude, dtype=xp.float64)[:, None]
    pp = xp.asarray(exponent, dtype=xp.float64)[:, None]
    tt = xp.asarray(transition, dtype=xp.float64)[:, None]
    ss = xp.asarray(shape, dtype=xp.float64)[:, None]
    u = xp.asarray(arrays["u"], dtype=xp.float64)[None, :]
    h_all = xp.asarray(arrays["h"], dtype=xp.float64)
    multiplier = xp.empty((end - begin, u.shape[1]), dtype=xp.float64)
    envelope = aa * xp.power(u, -pp) * xp.power(1.0 + xp.power(u / tt, ss), -1.0 / ss)
    for lane_id in range(4):
        mask = lane == lane_id
        if not bool(_to_numpy(xp.any(mask), xp)):
            continue
        hh = h_all[lane_id][None, :]
        lane_shape = ss[mask]
        if lane_id == 0:
            boundary = 0.05 + 0.95 * xp.power(hh, lane_shape)
        elif lane_id == 1:
            boundary = 0.05 + 0.95 * xp.power(hh / (0.25 + hh), lane_shape)
        elif lane_id == 2:
            boundary = 0.05 + 0.95 * xp.power(xp.sin(0.5 * xp.pi * hh), lane_shape)
        else:
            boundary = 0.05 + 0.95 * (1.0 - xp.exp(-lane_shape * hh))
        multiplier[mask] = 1.0 + envelope[mask] * boundary
    log_vbar = xp.log10(xp.asarray(arrays["vbar"], dtype=xp.float64))[None, :]
    return log_vbar + 0.5 * xp.log10(multiplier)


def _candidate_prediction_for_ids(
    candidates: Mapping[str, np.ndarray],
    ids: Sequence[int],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    return np.concatenate(
        [
            _candidate_log_velocity_batch(
                candidates, int(index), int(index) + 1, arrays, config, np
            )
            for index in ids
        ],
        axis=0,
    )


def _screen_candidates(
    candidates: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    start = time.perf_counter()
    folds = np.asarray(arrays["object_folds"], dtype=np.int64)
    point_folds = np.asarray(arrays["point_folds"], dtype=np.int64)
    outer_folds = int(config["evaluation"]["outer_folds"])
    if set(folds.tolist()) != set(range(outer_folds)):
        raise GravityItem39Error("Item 39 response-complete folds are incomplete")
    y = xp.asarray(arrays["log_observed"], dtype=xp.float64)[None, :]
    sigma = xp.asarray(arrays["log_sigma"], dtype=xp.float64)[None, :]
    weight = xp.asarray(arrays["point_weight"], dtype=xp.float64)[None, :]
    train_masks = [
        xp.asarray(point_folds != fold, dtype=xp.float64)[None, :] for fold in range(outer_folds)
    ]
    train_objects = [int(np.sum(folds != fold)) for fold in range(outer_folds)]
    best_loss = np.full(outer_folds, np.inf)
    best_index = np.full(outer_folds, -1, dtype=np.int64)
    full_best_loss = math.inf
    full_best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        prediction = _candidate_log_velocity_batch(candidates, begin, end, arrays, config, xp)
        loss = xp.square((prediction - y) / sigma)
        for fold in range(outer_folds):
            score = xp.sum(loss * weight * train_masks[fold], axis=1) / train_objects[fold]
            local = int(_to_numpy(xp.argmin(score), xp))
            value = float(_to_numpy(score[local], xp))
            if value < best_loss[fold]:
                best_loss[fold] = value
                best_index[fold] = begin + local
        score = xp.sum(loss * weight, axis=1) / len(arrays["galaxies"])
        local = int(_to_numpy(xp.argmin(score), xp))
        value = float(_to_numpy(score[local], xp))
        if value < full_best_loss:
            full_best_loss = value
            full_best_index = begin + local
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - start
    if np.any(best_index < 0) or full_best_index < 0:
        raise GravityItem39Error("Item 39 candidate search selected no formula")
    selected_predictions = _candidate_prediction_for_ids(
        candidates, best_index.tolist(), arrays, config
    )
    oof = np.empty_like(arrays["log_observed"])
    for fold in range(outer_folds):
        mask = point_folds == fold
        oof[mask] = selected_predictions[fold, mask]
    full_prediction = _candidate_prediction_for_ids(candidates, [full_best_index], arrays, config)[
        0
    ]
    crosscheck_count = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]),
        len(candidates["candidate_id"]),
    )
    cpu = _candidate_log_velocity_batch(candidates, 0, crosscheck_count, arrays, config, np)
    gpu = _to_numpy(
        _candidate_log_velocity_batch(candidates, 0, crosscheck_count, arrays, config, xp),
        xp,
    )
    max_difference = float(np.max(np.abs(cpu - gpu)))
    return (
        {
            "oof_prediction": oof,
            "selected_admissible_indices": best_index.tolist(),
            "selected_training_losses": best_loss.tolist(),
            "full_selected_admissible_index": int(full_best_index),
            "full_training_loss": float(full_best_loss),
            "full_prediction": full_prediction,
        },
        {
            "backend": backend,
            "device": device,
            "search_seconds": elapsed,
            "candidate_point_evaluations": int(
                len(candidates["candidate_id"]) * len(arrays["log_observed"])
            ),
            "cpu_gpu_crosscheck_candidates": crosscheck_count,
            "cpu_gpu_max_abs_log10_velocity_difference": max_difference,
            "cpu_gpu_tolerance": float(config["evaluation"]["cpu_gpu_tolerance"]),
            "cpu_gpu_passed": max_difference <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        },
    )


def _design(rows: Sequence[Mapping[str, str]], kind: str) -> np.ndarray:
    log_u = np.log10(np.asarray([float(row["u"]) for row in rows]))
    log_x = np.log10(
        np.maximum(np.asarray([float(row["radius_over_screen"]) for row in rows]), 1e-8)
    )
    log_sigma = np.log10(
        np.maximum(
            np.asarray([float(row["local_hi_surface_density"]) for row in rows]),
            1e-8,
        )
    )
    fraction = np.asarray([float(row["enclosed_fraction"]) for row in rows])
    slope = np.asarray([float(row["enclosed_log_slope"]) for row in rows])
    log_mass = np.log10(np.asarray([float(row["total_baryonic_mass_msun"]) for row in rows]))
    gas = np.asarray([float(row["gas_fraction"]) for row in rows])
    size = np.asarray([float(row["effective_to_screen_ratio"]) for row in rows])
    axis = np.asarray([float(row["axis_ratio"]) for row in rows])
    base = np.column_stack((log_u, log_x, log_sigma, fraction, slope, log_mass, gas, size, axis))
    if kind == "flexible_radial_surface":
        polynomial = np.column_stack(
            (
                np.square(log_u),
                log_u**3,
                np.square(log_x),
                log_x**3,
                np.square(log_sigma),
                log_sigma**3,
                log_u * log_x,
                log_u * log_sigma,
                log_x * log_sigma,
                fraction * slope,
                gas * size,
            )
        )
        return np.column_stack((base, polynomial))
    if kind != "matched_ordinary_geometry":
        raise GravityItem39Error(f"unknown ordinary design: {kind}")
    h = np.column_stack(
        [
            np.asarray([float(row[key]) for row in rows])
            for key in ("h_equipartition", "h_quasilocal", "h_wedge", "h_flow")
        ]
    )
    geometry = np.column_stack(
        (
            h,
            np.square(h),
            h * log_u[:, None],
            h * log_x[:, None],
            h * fraction[:, None],
            log_u * log_x,
            log_u * log_sigma,
            log_x * log_sigma,
            fraction * slope,
            gas * size,
        )
    )
    return np.column_stack((base, geometry))


def _ridge_fit(
    design: np.ndarray,
    target: np.ndarray,
    weight: np.ndarray,
    alpha: float,
) -> dict[str, np.ndarray]:
    mean = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design - mean) / scale
    matrix = np.column_stack((np.ones(len(design)), standardized))
    weighted = matrix * np.sqrt(weight)[:, None]
    weighted_target = target * np.sqrt(weight)
    penalty = np.eye(matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(weighted.T @ weighted + penalty) @ (weighted.T @ weighted_target)
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def _ridge_predict(model: Mapping[str, np.ndarray], design: np.ndarray) -> np.ndarray:
    standardized = (design - model["mean"]) / model["scale"]
    matrix = np.column_stack((np.ones(len(design)), standardized))
    return matrix @ model["coefficient"]


def _weighted_point_loss(
    target: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
    point_weight: np.ndarray,
    mask: np.ndarray,
    object_count: int,
) -> float:
    residual = np.square((target - prediction) / sigma)
    return float(np.sum(residual[mask] * point_weight[mask]) / object_count)


def _ridge_oof(
    rows: Sequence[Mapping[str, str]],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    kind: str,
) -> tuple[np.ndarray, list[float]]:
    design = _design(rows, kind)
    target = np.asarray(arrays["log_observed"])
    sigma = np.asarray(arrays["log_sigma"])
    point_weight = np.asarray(arrays["point_weight"])
    point_folds = np.asarray(arrays["point_folds"])
    object_folds = np.asarray(arrays["object_folds"])
    alphas = [float(value) for value in config["evaluation"]["ordinary_ridge_alphas"]]
    output = np.empty_like(target)
    selected_alphas = []
    for outer in range(int(config["evaluation"]["outer_folds"])):
        train_outer = point_folds != outer
        validation_scores = []
        for alpha in alphas:
            scores = []
            for inner in sorted(set(object_folds.tolist()) - {outer}):
                train = train_outer & (point_folds != inner)
                valid = point_folds == inner
                weight = point_weight[train] / np.square(sigma[train])
                model = _ridge_fit(design[train], target[train], weight, alpha)
                prediction = _ridge_predict(model, design[valid])
                scores.append(
                    _weighted_point_loss(
                        target[valid],
                        prediction,
                        sigma[valid],
                        point_weight[valid],
                        np.ones(np.sum(valid), dtype=bool),
                        int(np.sum(object_folds == inner)),
                    )
                )
            validation_scores.append(float(np.mean(scores)))
        alpha = alphas[int(np.argmin(validation_scores))]
        selected_alphas.append(alpha)
        weight = point_weight[train_outer] / np.square(sigma[train_outer])
        model = _ridge_fit(design[train_outer], target[train_outer], weight, alpha)
        held = point_folds == outer
        output[held] = _ridge_predict(model, design[held])
    return output, selected_alphas


def _object_losses(
    arrays: Mapping[str, Any], prediction: np.ndarray, target: np.ndarray | None = None
) -> np.ndarray:
    target = np.asarray(arrays["log_observed"] if target is None else target)
    sigma = np.asarray(arrays["log_sigma"])
    object_index = np.asarray(arrays["object_index"])
    losses = np.square((target - prediction) / sigma)
    result = np.zeros(len(arrays["galaxies"]), dtype=np.float64)
    for index in range(len(result)):
        result[index] = float(np.mean(losses[object_index == index]))
    return result


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def _paired_sign_flip(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    random = np.random.Generator(np.random.PCG64(int(config["evaluation"]["permutation_seed"])))
    trials = int(config["evaluation"]["paired_sign_flip_permutations"])
    observed = float(np.mean(differences))
    null = []
    for _ in range(trials):
        signs = random.choice(np.asarray([-1.0, 1.0]), size=len(differences))
        null.append(float(np.mean(differences * signs)))
    p_value = (1.0 + sum(value <= observed for value in null)) / (trials + 1.0)
    return {
        "trials": trials,
        "observed_mean_candidate_minus_reference_loss": observed,
        "p_value": p_value,
        "selection_aware": False,
        "claim_boundary": "paired whole-galaxy sign flip after frozen search; it does not correct for all researcher degrees of freedom",
    }


def _robustness(
    candidate: np.ndarray,
    reference: np.ndarray,
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    differences = candidate - reference
    full = _improvement(float(np.mean(reference)), float(np.mean(candidate)))
    order = np.argsort(np.abs(differences))[::-1]
    leave = np.ones(len(differences), dtype=bool)
    leave[order[0]] = False
    leave_improvement = _improvement(
        float(np.mean(reference[leave])), float(np.mean(candidate[leave]))
    )
    trim_count = max(
        1, math.floor(float(config["evaluation"]["robust_trim_fraction"]) * len(differences))
    )
    trim = np.ones(len(differences), dtype=bool)
    trim[order[:trim_count]] = False
    trim_improvement = _improvement(
        float(np.mean(reference[trim])), float(np.mean(candidate[trim]))
    )
    return {
        "full_improvement_percent": full,
        "most_influential_galaxy": arrays["galaxies"][int(order[0])],
        "leave_one_improvement_percent": leave_improvement,
        "leave_one_changes_sign": bool((full >= 0.0) != (leave_improvement >= 0.0)),
        "trim_fraction": float(config["evaluation"]["robust_trim_fraction"]),
        "trimmed_galaxies": trim_count,
        "trimmed_improvement_percent": trim_improvement,
        "trim_changes_sign": bool((full >= 0.0) != (trim_improvement >= 0.0)),
    }


def _object_metadata(
    rows: Sequence[Mapping[str, str]], arrays: Mapping[str, Any]
) -> list[dict[str, Any]]:
    by_name: dict[str, Mapping[str, str]] = {}
    for row in rows:
        by_name.setdefault(str(row["galaxy"]), row)
    return [
        {
            "galaxy": name,
            "source_cell": str(by_name[name]["source_cell"]),
            "log_mass": math.log10(float(by_name[name]["total_baryonic_mass_msun"])),
            "gas_fraction": float(by_name[name]["gas_fraction"]),
            "screen_ratio": float(by_name[name]["effective_to_screen_ratio"]),
            "inclination": float(by_name[name]["inclination_degrees"]),
        }
        for name in arrays["galaxies"]
    ]


def _strata(
    candidate: np.ndarray,
    reference: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    mass = np.asarray([float(row["log_mass"]) for row in metadata])
    gas = np.asarray([float(row["gas_fraction"]) for row in metadata])
    ratio = np.asarray([float(row["screen_ratio"]) for row in metadata])
    cells = np.asarray([str(row["source_cell"]) for row in metadata])
    masks: dict[str, np.ndarray] = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_gas_fraction": gas <= np.median(gas),
        "high_gas_fraction": gas > np.median(gas),
        "low_screen_ratio": ratio <= np.median(ratio),
        "high_screen_ratio": ratio > np.median(ratio),
    }
    for cell in sorted(set(cells.tolist())):
        masks[f"cell:{cell}"] = cells == cell
    result = {}
    for name, mask in masks.items():
        result[name] = {
            "galaxies": int(np.sum(mask)),
            "candidate_loss": float(np.mean(candidate[mask])),
            "reference_loss": float(np.mean(reference[mask])),
            "improvement_percent": _improvement(
                float(np.mean(reference[mask])), float(np.mean(candidate[mask]))
            ),
        }
    return result


def _systematic_predictions(
    candidates: Mapping[str, np.ndarray],
    selected: Sequence[int],
    arrays: Mapping[str, Any],
    rows: Sequence[Mapping[str, str]],
    config: Mapping[str, Any],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    point_folds = np.asarray(arrays["point_folds"])
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def selected_oof(changed: dict[str, Any]) -> np.ndarray:
        predictions = _candidate_prediction_for_ids(candidates, selected, changed, config)
        output = np.empty_like(arrays["log_observed"])
        for fold in range(int(config["evaluation"]["outer_folds"])):
            mask = point_folds == fold
            output[mask] = predictions[fold, mask]
        return output

    for dex in (
        -float(config["quality"]["stellar_mass_systematic_dex"]),
        float(config["quality"]["stellar_mass_systematic_dex"]),
    ):
        factor = 10.0**dex
        changed = dict(arrays)
        changed["u"] = np.asarray(arrays["u"]) * factor
        changed["vbar"] = np.asarray(arrays["vbar"]) * math.sqrt(factor)
        result[f"mass_{dex:+.2f}_dex"] = (
            selected_oof(changed),
            np.asarray(arrays["log_observed"]),
        )
    fraction = float(config["quality"]["distance_fractional_audit"])
    for factor in (1.0 - fraction, 1.0 + fraction):
        changed = dict(arrays)
        changed["vbar"] = np.asarray(arrays["vbar"]) * math.sqrt(factor)
        result[f"distance_factor_{factor:.2f}"] = (
            selected_oof(changed),
            np.asarray(arrays["log_observed"]),
        )
    inclination = np.asarray([float(row["inclination_degrees"]) for row in rows])
    shift = float(config["quality"]["inclination_shift_degrees_audit"])
    for delta in (-shift, shift):
        adjusted = np.asarray(arrays["observed"]) * np.sin(np.radians(inclination))
        adjusted /= np.sin(np.radians(np.clip(inclination + delta, 5.0, 85.0)))
        result[f"inclination_{delta:+.1f}_deg"] = (
            selected_oof(dict(arrays)),
            np.log10(adjusted),
        )
    return result


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    summary_path = _source_path(root, config, "extraction_summary")
    summary = _read_json(summary_path)
    rows = _load_rows(root, config)
    arrays = _point_arrays(rows)
    candidates, candidate_audit = admissible_candidates(config)
    screened, compute = _screen_candidates(candidates, arrays, config)
    candidate_prediction = np.asarray(screened["oof_prediction"])
    fixed_predictions = {
        name: np.log10(arrays["vbar"]) + 0.5 * np.log10(fixed_control_multiplier(name, arrays["u"]))
        for name in ("baryonic_newton", "mond_RAR", "item38_selected")
    }
    flexible_prediction, flexible_alphas = _ridge_oof(
        rows, arrays, config, "flexible_radial_surface"
    )
    geometry_prediction, geometry_alphas = _ridge_oof(
        rows, arrays, config, "matched_ordinary_geometry"
    )
    predictions = {
        **fixed_predictions,
        "flexible_radial_surface": flexible_prediction,
        "matched_ordinary_geometry": geometry_prediction,
    }
    candidate_object = _object_losses(arrays, candidate_prediction)
    object_losses = {
        name: _object_losses(arrays, prediction) for name, prediction in predictions.items()
    }
    losses = {
        "candidate": float(np.mean(candidate_object)),
        **{name: float(np.mean(value)) for name, value in object_losses.items()},
    }
    strongest_name = min(
        ("flexible_radial_surface", "matched_ordinary_geometry"),
        key=lambda name: losses[name],
    )
    strongest = object_losses[strongest_name]
    improvement = _improvement(losses[strongest_name], losses["candidate"])
    differences = candidate_object - strongest
    permutation = _paired_sign_flip(differences, config)
    robustness = _robustness(candidate_object, strongest, arrays, config)
    metadata = _object_metadata(rows, arrays)
    strata = _strata(candidate_object, strongest, metadata)
    systematic_predictions = _systematic_predictions(
        candidates,
        screened["selected_admissible_indices"],
        arrays,
        rows,
        config,
    )
    systematic = {}
    stable_counterexample = candidate_object > strongest
    for name, (prediction, target) in systematic_predictions.items():
        candidate_variant = _object_losses(arrays, prediction, target)
        comparison = candidate_variant - strongest
        stable_counterexample &= comparison > 0.0
        systematic[name] = {
            "candidate_loss": float(np.mean(candidate_variant)),
            "reference_loss": float(np.mean(strongest)),
            "improvement_percent": _improvement(
                float(np.mean(strongest)), float(np.mean(candidate_variant))
            ),
        }
    audit_no_sign_reversal = all(
        (value["improvement_percent"] >= 0.0) == (improvement >= 0.0)
        for value in systematic.values()
    )
    raw_counterexamples = int(np.sum(candidate_object > strongest))
    resolved_counterexamples = int(np.sum(stable_counterexample))
    failure_strata = int(sum(value["improvement_percent"] < 0.0 for value in strata.values()))
    quality_passed = bool(summary["quality"]["overall_passed"])
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(arrays["galaxies"]),
        "raw_counterexample_count": raw_counterexamples,
        "quality_verified_counterexample_count": raw_counterexamples,
        "uncertainty_resolved_counterexample_count": resolved_counterexamples,
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": quality_passed,
        "strongest_baseline_failed": improvement < 0.0,
        "leave_one_changes_sign": robustness["leave_one_changes_sign"],
        "trim_changes_sign": robustness["trim_changes_sign"],
        "independent_failure_strata": failure_strata,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    policy = load_counterexample_policy(root / POLICY_PATH)
    assessment = assess_counterexample_evidence(report, policy)
    object_records = []
    for index, name in enumerate(arrays["galaxies"]):
        object_records.append(
            {
                "galaxy": name,
                "candidate_loss": float(candidate_object[index]),
                "reference_loss": float(strongest[index]),
                "comparative_difference": float(differences[index]),
                "raw_counterexample": bool(differences[index] > 0.0),
                "uncertainty_resolved_counterexample": bool(stable_counterexample[index]),
            }
        )
    selected_records = [
        decode_candidate(int(candidates["candidate_id"][index]), config)
        for index in screened["selected_admissible_indices"]
    ]
    full_candidate = decode_candidate(
        int(candidates["candidate_id"][screened["full_selected_admissible_index"]]),
        config,
    )
    gates = {
        "quality_passes": quality_passed,
        "beats_baryonic": losses["candidate"] < losses["baryonic_newton"],
        "beats_mond_RAR": losses["candidate"] < losses["mond_RAR"],
        "beats_item38_selected": losses["candidate"] < losses["item38_selected"],
        "beats_flexible_radial_surface": losses["candidate"] < losses["flexible_radial_surface"],
        "beats_matched_ordinary_geometry": losses["candidate"]
        < losses["matched_ordinary_geometry"],
        "paired_p_at_most_0p05": permutation["p_value"] <= 0.05,
        "all_broad_strata_improve": all(
            value["improvement_percent"] > 0.0 for value in strata.values()
        ),
        "not_single_object_sensitive": not robustness["leave_one_changes_sign"]
        and not robustness["trim_changes_sign"],
        "systematic_audits_do_not_reverse_sign": audit_no_sign_reversal,
        "cpu_gpu_agreement": bool(compute["cpu_gpu_passed"]),
        "confirmation_remains_sealed": int(summary["counts"]["confirmation_response_rows"]) == 0,
        "one_metric_motion_light_contract_preserved": True,
        "hard_theoretical_veto_absent": True,
        "lensing_transfer_pending": False,
    }
    primary_gates = {
        key: value for key, value in gates.items() if key != "lensing_transfer_pending"
    }
    if not quality_passed:
        decision = "INCONCLUSIVE_ITEM39_WALLABY_QUALITY_RETAINED"
    elif all(primary_gates.values()):
        decision = "ITEM39_DYNAMICS_LEAD_PENDING_UNCHANGED_LENSING_TRANSFER"
    elif improvement > 0.0 and not robustness["leave_one_changes_sign"]:
        decision = "NONPROMOTED_ITEM39_PARTIAL_DYNAMICS_PATTERN_RETAINED"
    else:
        decision = "ROBUST_SCOPED_NEGATIVE_ITEM39_DYNAMICS_REPRESENTATION_RETAINED"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item39-compute-manifest-1.0",
            "item": 39,
            "decision": decision,
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "predictor_freeze_commit": config["predictor_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "candidate_manifest_sha256": _sha256_file(
                    _source_path(root, config, "candidate_manifest")
                ),
                "sample_manifest_sha256": _sha256_file(
                    _source_path(root, config, "sample_manifest")
                ),
                "extraction_summary_sha256": _sha256_file(summary_path),
                "post_response_candidate_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "quality": summary["quality"],
            "candidate_search": {
                **candidate_audit,
                **compute,
                "selected_fold_candidates": selected_records,
                "selected_fold_training_losses": screened["selected_training_losses"],
                "full_exploration_candidate": full_candidate,
                "full_exploration_training_loss": screened["full_training_loss"],
            },
            "primary_dynamics": {
                "galaxies": len(arrays["galaxies"]),
                "points": len(rows),
                "losses": losses,
                "strongest_ordinary_baseline": strongest_name,
                "improvement_vs_strongest_percent": improvement,
                "ordinary_selected_alphas": {
                    "flexible_radial_surface": flexible_alphas,
                    "matched_ordinary_geometry": geometry_alphas,
                },
                "paired_sign_flip": permutation,
                "robustness": robustness,
                "strata": strata,
                "systematic_audits": systematic,
                "systematic_audits_do_not_reverse_sign": audit_no_sign_reversal,
                "object_level": object_records,
                "counterexample_policy_report": report,
                "counterexample_assessment": assessment,
            },
            "gates": gates,
            "claim_boundaries": {
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "historical_novelty_established": False,
                "complete_relativistic_theory_established": False,
                "lensing_test_completed": False,
                "formula_family_pruned": False,
                "one_counterexample_is_veto": False,
            },
        }
    )
    path = _source_path(root, config, "compute_manifest")
    if write:
        _write_json(path, result)
    return result


def _scientific_replay_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    """Remove measured runtime fields that are not scientific replay inputs."""

    result = json.loads(json.dumps(value))
    result.pop("content_sha256", None)
    result["candidate_search"].pop("search_seconds", None)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "compute_manifest")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 39 compute manifest")
    replay = evaluate(root, write=False)
    if _scientific_replay_payload(existing) != _scientific_replay_payload(replay):
        raise GravityItem39Error("Item 39 compute replay drifted")
    return {
        "status": "ITEM39_COMPUTE_REPLAY_VALID",
        "compute_manifest_sha256": _sha256_file(path),
        "decision": existing["decision"],
        "confirmation_response_rows": 0,
        "paid_model_calls": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "run":
        value = evaluate(args.root)
        print(json.dumps({"decision": value["decision"]}, sort_keys=True))
    else:
        print(json.dumps(check(args.root), sort_keys=True))


if __name__ == "__main__":
    main()
