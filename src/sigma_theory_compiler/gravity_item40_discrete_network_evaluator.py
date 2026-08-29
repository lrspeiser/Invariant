"""Evaluate the frozen Item 40 discrete/network search on fresh WALLABY responses."""

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
    load_config as load_item39_config,
)
from sigma_theory_compiler.gravity_item40_discrete_network import (
    POLICY_PATH,
    GravityItem40Error,
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
        raise GravityItem40Error("duplicate Item 40 response point")
    rows = []
    for feature in features:
        response = response_by_key.pop(tuple(feature[key] for key in keys), None)
        if response is None:
            raise GravityItem40Error("Item 40 feature lacks aligned response")
        rows.append({**feature, **response})
    if response_by_key:
        raise GravityItem40Error("Item 40 response lacks aligned feature")
    return rows


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device, bytes):
            device = device.decode()
        return cp, "gpu_cupy", str(device)
    except Exception:  # noqa: BLE001 - deterministic CPU fallback is part of the protocol.
        return np, "cpu_numpy", "CPU"


def _to_numpy(value: Any, xp: Any) -> np.ndarray:
    return xp.asnumpy(value) if xp is not np else np.asarray(value)


def _point_arrays(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    point_galaxies = np.asarray([str(row["galaxy"]) for row in rows])
    galaxies = sorted(set(point_galaxies.tolist()))
    galaxy_index = {name: index for index, name in enumerate(galaxies)}
    object_index = np.asarray([galaxy_index[name] for name in point_galaxies], dtype=np.int64)
    counts = np.bincount(object_index, minlength=len(galaxies))
    point_weight = 1.0 / counts[object_index]
    fold_by_name: dict[str, int] = {}
    for row in rows:
        name = str(row["galaxy"])
        fold = int(row["outer_fold"])
        if name in fold_by_name and fold_by_name[name] != fold:
            raise GravityItem40Error("one galaxy entered multiple folds")
        fold_by_name[name] = fold
    object_folds = np.asarray([fold_by_name[name] for name in galaxies], dtype=np.int64)
    point_folds = object_folds[object_index]
    observed = np.asarray([float(row["observed_speed_km_s"]) for row in rows])
    observed_error = np.asarray(
        [float(row["observed_speed_error_km_s"]) for row in rows]
    )
    return {
        "galaxies": galaxies,
        "object_index": object_index,
        "object_folds": object_folds,
        "point_folds": point_folds,
        "point_weight": point_weight,
        "observed": observed,
        "log_observed": np.log10(observed),
        "log_sigma": np.maximum(observed_error / (observed * math.log(10.0)), 0.03),
        "u": np.asarray([float(row["u"]) for row in rows]),
        "vbar": np.asarray([float(row["vbar_km_s"]) for row in rows]),
        "h": np.vstack(
            [
                np.asarray([float(row[key]) for row in rows])
                for key in (
                    "h_spectral",
                    "h_resistance",
                    "h_heat",
                    "h_communicability",
                )
            ]
        ),
        "item39_h": np.vstack(
            [
                np.asarray([float(row[key]) for row in rows])
                for key in (
                    "item39_h_equipartition",
                    "item39_h_quasilocal",
                    "item39_h_wedge",
                    "item39_h_flow",
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
    h = xp.asarray(arrays["h"], dtype=xp.float64)
    multiplier = xp.empty((end - begin, u.shape[1]), dtype=xp.float64)
    envelope = aa * xp.power(u, -pp) * xp.power(
        1.0 + xp.power(u / tt, ss), -1.0 / ss
    )
    for lane_id in range(4):
        mask = lane == lane_id
        if not bool(_to_numpy(xp.any(mask), xp)):
            continue
        boundary = 0.05 + 0.95 * xp.power(h[lane_id][None, :], ss[mask])
        multiplier[mask] = 1.0 + envelope[mask] * boundary
    return xp.log10(xp.asarray(arrays["vbar"], dtype=xp.float64))[None, :] + 0.5 * xp.log10(
        multiplier
    )


def _candidate_prediction_for_indices(
    candidates: Mapping[str, np.ndarray],
    indices: Sequence[int],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
) -> np.ndarray:
    return np.concatenate(
        [
            _candidate_log_velocity_batch(
                candidates, int(index), int(index) + 1, arrays, config, np
            )
            for index in indices
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
    started = time.perf_counter()
    folds = np.asarray(arrays["object_folds"])
    point_folds = np.asarray(arrays["point_folds"])
    unique_folds = sorted(set(folds.tolist()))
    if unique_folds != list(range(int(config["evaluation"]["outer_folds"]))):
        raise GravityItem40Error("response-complete Item 40 folds are incomplete")
    y = xp.asarray(arrays["log_observed"], dtype=xp.float64)[None, :]
    sigma = xp.asarray(arrays["log_sigma"], dtype=xp.float64)[None, :]
    weight = xp.asarray(arrays["point_weight"], dtype=xp.float64)[None, :]
    masks = [xp.asarray(point_folds != fold, dtype=xp.float64)[None, :] for fold in unique_folds]
    train_objects = [int(np.sum(folds != fold)) for fold in unique_folds]
    best_loss = np.full(len(unique_folds), np.inf)
    best_index = np.full(len(unique_folds), -1, dtype=np.int64)
    full_best_loss = math.inf
    full_best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        prediction = _candidate_log_velocity_batch(candidates, begin, end, arrays, config, xp)
        loss = xp.square((prediction - y) / sigma)
        for fold in unique_folds:
            score = xp.sum(loss * weight * masks[fold], axis=1) / train_objects[fold]
            local = int(_to_numpy(xp.argmin(score), xp))
            value = float(_to_numpy(score[local], xp))
            if value < best_loss[fold]:
                best_loss[fold] = value
                best_index[fold] = begin + local
        full_score = xp.sum(loss * weight, axis=1) / len(arrays["galaxies"])
        local = int(_to_numpy(xp.argmin(full_score), xp))
        value = float(_to_numpy(full_score[local], xp))
        if value < full_best_loss:
            full_best_loss = value
            full_best_index = begin + local
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    selected = _candidate_prediction_for_indices(
        candidates, best_index.tolist(), arrays, config
    )
    oof = np.empty_like(arrays["log_observed"])
    for fold in unique_folds:
        mask = point_folds == fold
        oof[mask] = selected[fold, mask]
    full_prediction = _candidate_prediction_for_indices(
        candidates, [full_best_index], arrays, config
    )[0]
    crosscheck = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]),
        len(candidates["candidate_id"]),
    )
    cpu = _candidate_log_velocity_batch(candidates, 0, crosscheck, arrays, config, np)
    device_values = _to_numpy(
        _candidate_log_velocity_batch(candidates, 0, crosscheck, arrays, config, xp), xp
    )
    difference = float(np.max(np.abs(cpu - device_values)))
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
                len(candidates["candidate_id"]) * len(arrays["u"]) * (len(unique_folds) + 1)
            ),
            "cpu_gpu_max_abs_log10_velocity_difference": difference,
            "cpu_gpu_tolerance": float(config["evaluation"]["cpu_gpu_tolerance"]),
            "cpu_gpu_passed": difference
            <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        },
    )


def _design(rows: Sequence[Mapping[str, str]], kind: str) -> np.ndarray:
    log_u = np.log10([float(row["u"]) for row in rows])
    log_x = np.log10([max(float(row["radius_over_screen"]), 1e-8) for row in rows])
    log_sigma = np.log10(
        [max(float(row["local_hi_surface_density"]), 1e-8) for row in rows]
    )
    fraction = np.asarray([float(row["enclosed_fraction"]) for row in rows])
    slope = np.asarray([float(row["enclosed_log_slope"]) for row in rows])
    log_mass = np.log10([float(row["total_baryonic_mass_msun"]) for row in rows])
    gas = np.asarray([float(row["gas_fraction"]) for row in rows])
    log_size = np.log10([float(row["effective_radius_kpc"]) for row in rows])
    base = np.column_stack(
        (log_u, log_x, log_sigma, fraction, slope, log_mass, gas, log_size)
    )
    if kind == "flexible_radial_surface":
        extra = np.column_stack(
            (
                log_u**2,
                log_u**3,
                log_x**2,
                log_x**3,
                log_sigma**2,
                log_sigma**3,
                log_u * log_x,
                log_u * log_sigma,
                log_x * log_sigma,
                fraction * slope,
                gas * log_size,
            )
        )
        return np.column_stack((base, extra))
    if kind != "matched_ordinary_geometry":
        raise GravityItem40Error(f"unknown ordinary design: {kind}")
    extra = np.column_stack(
        (
            base**2,
            log_u * log_x,
            log_u * log_sigma,
            log_x * log_sigma,
            fraction * slope,
            gas * log_size,
        )
    )
    return np.column_stack((base, extra))


def _ridge_fit(
    design: np.ndarray, target: np.ndarray, weight: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    mean = np.mean(design, axis=0)
    scale = np.where(np.std(design, axis=0) > 1e-12, np.std(design, axis=0), 1.0)
    matrix = np.column_stack((np.ones(len(design)), (design - mean) / scale))
    weighted = matrix * np.sqrt(weight)[:, None]
    penalty = np.eye(matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(weighted.T @ weighted + penalty) @ (
        weighted.T @ (target * np.sqrt(weight))
    )
    return {"mean": mean, "scale": scale, "coefficient": coefficient}


def _ridge_predict(model: Mapping[str, np.ndarray], design: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(
        (np.ones(len(design)), (design - model["mean"]) / model["scale"])
    )
    return matrix @ model["coefficient"]


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
    selected_alphas: list[float] = []
    for outer in range(int(config["evaluation"]["outer_folds"])):
        train_outer = point_folds != outer
        scores_by_alpha = []
        inner_folds = sorted(set(object_folds.tolist()) - {outer})
        for alpha in alphas:
            scores = []
            for inner in inner_folds:
                train = train_outer & (point_folds != inner)
                valid = point_folds == inner
                model = _ridge_fit(
                    design[train],
                    target[train],
                    point_weight[train] / np.square(sigma[train]),
                    alpha,
                )
                prediction = _ridge_predict(model, design[valid])
                scores.append(
                    float(
                        np.sum(
                            np.square((target[valid] - prediction) / sigma[valid])
                            * point_weight[valid]
                        )
                        / int(np.sum(object_folds == inner))
                    )
                )
            scores_by_alpha.append(float(np.mean(scores)))
        alpha = alphas[int(np.argmin(scores_by_alpha))]
        selected_alphas.append(alpha)
        model = _ridge_fit(
            design[train_outer],
            target[train_outer],
            point_weight[train_outer] / np.square(sigma[train_outer]),
            alpha,
        )
        held = point_folds == outer
        output[held] = _ridge_predict(model, design[held])
    return output, selected_alphas


def _object_losses(
    arrays: Mapping[str, Any], prediction: np.ndarray
) -> tuple[np.ndarray, list[np.ndarray]]:
    residual = np.square(
        (np.asarray(arrays["log_observed"]) - prediction) / np.asarray(arrays["log_sigma"])
    )
    object_index = np.asarray(arrays["object_index"])
    result = np.zeros(len(arrays["galaxies"]), dtype=np.float64)
    point_values: list[np.ndarray] = []
    for index in range(len(result)):
        values = residual[object_index == index]
        result[index] = float(np.mean(values))
        point_values.append(values)
    return result, point_values


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def _paired_sign_flip(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    random = np.random.Generator(
        np.random.PCG64(int(config["evaluation"]["permutation_seed"]))
    )
    trials = int(config["evaluation"]["paired_sign_flip_permutations"])
    observed = float(np.mean(differences))
    null = [
        float(np.mean(differences * random.choice([-1.0, 1.0], size=len(differences))))
        for _ in range(trials)
    ]
    return {
        "trials": trials,
        "observed_mean_candidate_minus_reference_loss": observed,
        "p_value": (1.0 + sum(value <= observed for value in null)) / (trials + 1.0),
        "selection_aware": False,
    }


def _robustness(
    candidate: np.ndarray, reference: np.ndarray, galaxies: Sequence[str], config: Mapping[str, Any]
) -> dict[str, Any]:
    differences = candidate - reference
    full = _improvement(float(np.mean(reference)), float(np.mean(candidate)))
    order = np.argsort(np.abs(differences))[::-1]
    leave = np.ones(len(differences), dtype=bool)
    leave[order[0]] = False
    leave_value = _improvement(float(np.mean(reference[leave])), float(np.mean(candidate[leave])))
    trim_count = max(1, math.floor(float(config["evaluation"]["robust_trim_fraction"]) * len(differences)))
    trim = np.ones(len(differences), dtype=bool)
    trim[order[:trim_count]] = False
    trim_value = _improvement(float(np.mean(reference[trim])), float(np.mean(candidate[trim])))
    return {
        "full_improvement_percent": full,
        "most_influential_galaxy": galaxies[int(order[0])],
        "leave_one_improvement_percent": leave_value,
        "leave_one_changes_sign": bool((full >= 0.0) != (leave_value >= 0.0)),
        "trim_fraction": float(config["evaluation"]["robust_trim_fraction"]),
        "trimmed_galaxies": trim_count,
        "trimmed_improvement_percent": trim_value,
        "trim_changes_sign": bool((full >= 0.0) != (trim_value >= 0.0)),
    }


def _item39_prediction(arrays: Mapping[str, Any], root: Path) -> np.ndarray:
    config = load_item39_config(root)
    decoded = config["lensing_transfer"]["selected_candidate"]
    u = np.asarray(arrays["u"])
    h = np.asarray(arrays["item39_h"])[2]
    amplitude = float(decoded["amplitude"])
    exponent = float(decoded["exponent"])
    transition = float(decoded["transition_u"])
    shape = float(decoded["shape"])
    envelope = amplitude * u ** (-exponent) * (1.0 + (u / transition) ** shape) ** (
        -1.0 / shape
    )
    direct = 1.0 + envelope * (0.05 + 0.95 * np.sin(0.5 * np.pi * h) ** shape)
    return np.log10(arrays["vbar"]) + 0.5 * np.log10(direct)


def _uncertainty_resolved_counterexamples(
    arrays: Mapping[str, Any], candidate_prediction: np.ndarray, reference_prediction: np.ndarray
) -> np.ndarray:
    target = np.asarray(arrays["log_observed"])
    sigma = np.asarray(arrays["log_sigma"])
    difference = np.square((target - candidate_prediction) / sigma) - np.square(
        (target - reference_prediction) / sigma
    )
    object_index = np.asarray(arrays["object_index"])
    resolved = np.zeros(len(arrays["galaxies"]), dtype=bool)
    for index in range(len(resolved)):
        values = difference[object_index == index]
        standard_error = (
            float(np.std(values, ddof=1)) / math.sqrt(len(values)) if len(values) > 1 else math.inf
        )
        resolved[index] = float(np.mean(values)) - 1.96 * standard_error > 0.0
    return resolved


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

    predictions = {
        name: np.log10(arrays["vbar"])
        + 0.5 * np.log10(fixed_control_multiplier(name, arrays["u"]))
        for name in ("baryonic_newton", "mond_RAR")
    }
    predictions["item39_selected"] = _item39_prediction(arrays, root)
    flexible, flexible_alphas = _ridge_oof(rows, arrays, config, "flexible_radial_surface")
    ordinary, ordinary_alphas = _ridge_oof(rows, arrays, config, "matched_ordinary_geometry")
    predictions["flexible_radial_surface"] = flexible
    predictions["matched_ordinary_geometry"] = ordinary

    candidate_object, _ = _object_losses(arrays, candidate_prediction)
    object_losses = {name: _object_losses(arrays, value)[0] for name, value in predictions.items()}
    losses = {
        "candidate": float(np.mean(candidate_object)),
        **{name: float(np.mean(value)) for name, value in object_losses.items()},
    }
    strongest_name = min(object_losses, key=lambda name: losses[name])
    strongest_object = object_losses[strongest_name]
    strongest_prediction = predictions[strongest_name]
    differences = candidate_object - strongest_object
    improvement = _improvement(losses[strongest_name], losses["candidate"])
    robustness = _robustness(
        candidate_object, strongest_object, arrays["galaxies"], config
    )
    permutation = _paired_sign_flip(differences, config)
    resolved = _uncertainty_resolved_counterexamples(
        arrays, candidate_prediction, strongest_prediction
    )
    raw = candidate_object > strongest_object

    metadata: dict[str, dict[str, float | int]] = {}
    for row in rows:
        metadata.setdefault(
            str(row["galaxy"]),
            {
                "log_mass": math.log10(float(row["total_baryonic_mass_msun"])),
                "gas_fraction": float(row["gas_fraction"]),
                "outer_fold": int(row["outer_fold"]),
            },
        )
    mass = np.asarray([float(metadata[name]["log_mass"]) for name in arrays["galaxies"]])
    gas = np.asarray([float(metadata[name]["gas_fraction"]) for name in arrays["galaxies"]])
    strata_masks = {
        "low_mass": mass <= np.median(mass),
        "high_mass": mass > np.median(mass),
        "low_gas_fraction": gas <= np.median(gas),
        "high_gas_fraction": gas > np.median(gas),
    }
    strata = {
        name: {
            "galaxies": int(np.sum(mask)),
            "improvement_percent": _improvement(
                float(np.mean(strongest_object[mask])), float(np.mean(candidate_object[mask]))
            ),
        }
        for name, mask in strata_masks.items()
        if np.any(mask)
    }
    quality = config["response_quality"]
    quality_passed = (
        int(summary["counts"]["quality_passing_galaxies"])
        >= int(quality["minimum_quality_passing_exploration_galaxies"])
        and int(summary["counts"]["quality_passing_galaxies"])
        / int(summary["counts"]["exploration_response_rows"])
        >= float(quality["minimum_quality_retention_fraction"])
    )
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(arrays["galaxies"]),
        "raw_counterexample_count": int(np.sum(raw)),
        "quality_verified_counterexample_count": int(np.sum(raw)),
        "uncertainty_resolved_counterexample_count": int(np.sum(resolved & raw)),
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": quality_passed,
        "strongest_baseline_failed": improvement < 0.0,
        "leave_one_changes_sign": robustness["leave_one_changes_sign"],
        "trim_changes_sign": robustness["trim_changes_sign"],
        "independent_failure_strata": int(
            sum(value["improvement_percent"] < 0.0 for value in strata.values())
        ),
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        report, load_counterexample_policy(root / POLICY_PATH)
    )
    selected_by_fold = [
        decode_candidate(int(candidates["candidate_id"][index]), config)
        for index in screened["selected_admissible_indices"]
    ]
    full_selected = decode_candidate(
        int(candidates["candidate_id"][screened["full_selected_admissible_index"]]), config
    )
    object_records = [
        {
            "galaxy": name,
            "candidate_loss": float(candidate_object[index]),
            "strongest_reference_loss": float(strongest_object[index]),
            "candidate_minus_reference": float(differences[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(resolved[index] and raw[index]),
        }
        for index, name in enumerate(arrays["galaxies"])
    ]
    gates = {
        "quality_passes": quality_passed,
        "beats_baryonic": losses["candidate"] < losses["baryonic_newton"],
        "beats_mond_RAR": losses["candidate"] < losses["mond_RAR"],
        "beats_item39_selected": losses["candidate"] < losses["item39_selected"],
        "beats_flexible_radial_surface": losses["candidate"]
        < losses["flexible_radial_surface"],
        "beats_matched_ordinary_geometry": losses["candidate"]
        < losses["matched_ordinary_geometry"],
        "paired_p_at_most_0p05": permutation["p_value"] <= 0.05,
        "not_single_object_sensitive": not robustness["leave_one_changes_sign"]
        and not robustness["trim_changes_sign"],
        "confirmation_remains_sealed": int(summary["counts"]["confirmation_response_rows"])
        == 0,
        "cpu_gpu_agreement": bool(compute["cpu_gpu_passed"]),
        "cluster_transfer_pending": True,
    }
    if not quality_passed:
        decision = "INCONCLUSIVE_ITEM40_QUALITY_LIMITED_RETAINED"
    elif improvement > 0.0 and gates["not_single_object_sensitive"]:
        decision = "ITEM40_NETWORK_DYNAMICS_LEAD_PENDING_UNCHANGED_CLUSTER_TRANSFER"
    else:
        decision = "NONPROMOTED_ITEM40_NETWORK_DYNAMICS_RETAINED"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-compute-manifest-1.0",
            "item": 40,
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
                "evaluation_implementation_committed_before_response": False,
                "post_response_candidate_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "quality": {
                "passed": quality_passed,
                "passing_galaxies": summary["counts"]["quality_passing_galaxies"],
                "exploration_galaxies": summary["counts"]["exploration_response_rows"],
                "accepted_points": summary["counts"]["accepted_rotation_points"],
                "failed_records_preserved": summary["galaxies"],
            },
            "candidate_search": {
                **candidate_audit,
                **compute,
                "selected_fold_candidates": selected_by_fold,
                "selected_fold_training_losses": screened["selected_training_losses"],
                "full_exploration_candidate": full_selected,
                "full_exploration_training_loss": screened["full_training_loss"],
            },
            "primary_dynamics": {
                "galaxies": len(arrays["galaxies"]),
                "points": len(rows),
                "losses": losses,
                "strongest_baseline": strongest_name,
                "improvement_vs_strongest_percent": improvement,
                "ordinary_selected_alphas": {
                    "flexible_radial_surface": flexible_alphas,
                    "matched_ordinary_geometry": ordinary_alphas,
                },
                "paired_sign_flip": permutation,
                "robustness": robustness,
                "strata": strata,
                "counterexample_uncertainty_rule": "A raw object-level loss is uncertainty-resolved only when the lower 95% normal bound of its paired point-loss difference remains above zero.",
                "object_level": object_records,
                "counterexample_policy_report": report,
                "counterexample_assessment": assessment,
            },
            "gates": gates,
            "claim_boundaries": {
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "historical_novelty_established": False,
                "covariant_theory_established": False,
                "formula_family_pruned": False,
                "one_empirical_counterexample_is_veto": False,
                "late_evaluator_implementation_forces_diagnostic_status": True,
            },
        }
    )
    if write:
        _write_json(_source_path(root, config, "compute_manifest"), result)
    return result


def _scientific_payload(value: Mapping[str, Any]) -> dict[str, Any]:
    result = json.loads(json.dumps(value))
    result.pop("content_sha256", None)
    result["candidate_search"].pop("search_seconds", None)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "compute_manifest")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 40 compute manifest")
    replay = evaluate(root, write=False)
    if _scientific_payload(existing) != _scientific_payload(replay):
        raise GravityItem40Error("Item 40 scientific replay changed")
    return {
        "status": "ITEM40_DYNAMICS_REPLAY_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "confirmation_response_rows": existing["protocol"]["confirmation_response_rows"],
        "paid_model_calls": existing["protocol"]["paid_model_calls"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    result = evaluate(args.root) if args.command == "run" else check(args.root)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
