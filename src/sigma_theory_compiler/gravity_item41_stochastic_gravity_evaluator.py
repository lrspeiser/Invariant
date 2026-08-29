"""Evaluate Item 41 joint stochastic drift and variance laws on paired GHASP sides."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

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
from sigma_theory_compiler.gravity_item28_periodic_gravity import (
    _load_joined_rows,
    _row_physics,
)
from sigma_theory_compiler.gravity_item28_periodic_gravity import (
    load_config as load_item28_config,
)
from sigma_theory_compiler.gravity_item41_stochastic_gravity import (
    POLICY_PATH,
    GravityItem41Error,
    _candidate_parameters,
    _source_path,
    admissible_candidates,
    decode_candidate,
    load_config,
    stochastic_moments,
)


def _backend() -> tuple[Any, str, str]:
    try:
        import cupy as cp

        device = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device, bytes):
            device = device.decode()
        return cp, "gpu_cupy", str(device)
    except Exception:  # noqa: BLE001 - deterministic CPU fallback is in the protocol.
        return np, "cpu_numpy", "CPU"


def _to_numpy(value: Any, xp: Any) -> np.ndarray:
    return xp.asnumpy(value) if xp is not np else np.asarray(value)


def _arrays(root: Path, config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    item28 = load_item28_config(root)
    rows, prior_quality = _load_joined_rows(root, item28)
    if len(rows) != int(config["data"]["ghasp_expected_paired_points"]):
        raise GravityItem41Error("GHASP paired row count changed")
    names = np.asarray([str(row["identity"]) for row in rows])
    galaxies = sorted(set(names.tolist()))
    if len(galaxies) != int(config["data"]["ghasp_expected_usable_galaxies"]):
        raise GravityItem41Error("GHASP paired galaxy count changed")
    index_by_name = {name: index for index, name in enumerate(galaxies)}
    object_index = np.asarray([index_by_name[name] for name in names], dtype=np.int64)
    counts = np.bincount(object_index, minlength=len(galaxies))
    point_weight = 1.0 / counts[object_index]
    object_folds: dict[str, int] = {}
    for row in rows:
        name = str(row["identity"])
        fold = int(row["fold"])
        if name in object_folds and object_folds[name] != fold:
            raise GravityItem41Error("one GHASP galaxy entered multiple folds")
        object_folds[name] = fold
    folds = np.asarray([object_folds[name] for name in galaxies], dtype=np.int64)
    point_folds = folds[object_index]
    radius = np.asarray([float(row["radius_kpc"]) for row in rows])
    approaching = np.asarray([float(row["approaching_velocity_km_s"]) for row in rows])
    receding = np.asarray([float(row["receding_velocity_km_s"]) for row in rows])
    log_g_approaching = 2.0 * np.log(approaching) - np.log(radius)
    log_g_receding = 2.0 * np.log(receding) - np.log(radius)
    physics = [_row_physics(row, item28) for row in rows]
    gbar = np.asarray([value[1] for value in physics])
    return rows, {
        "galaxies": galaxies,
        "object_index": object_index,
        "object_folds": folds,
        "point_folds": point_folds,
        "point_weight": point_weight,
        "y_mean": 0.5 * (log_g_approaching + log_g_receding),
        "y_difference": (log_g_approaching - log_g_receding) / math.sqrt(2.0),
        "u": gbar / 1.2e-10,
        "x": np.asarray([float(row["radius_disk_scale"]) for row in rows]),
        "prior_quality": prior_quality,
    }


def _design(rows: Sequence[Mapping[str, Any]], root: Path, kind: str) -> np.ndarray:
    item28 = load_item28_config(root)
    physics = [_row_physics(row, item28) for row in rows]
    log_gbar = np.log([value[1] for value in physics])
    log_x = np.log([float(row["radius_disk_scale"]) for row in rows])
    mass = np.asarray([float(row["log_stellar_mass_proxy"]) for row in rows]) - 10.0
    mu0 = np.asarray([float(row["disk_mu0_R"]) for row in rows]) - 21.0
    inclination = np.asarray([float(row["inclination_deg"]) for row in rows]) / 60.0
    morphology = np.asarray([float(row["ttype"]) for row in rows]) / 10.0
    bulge = np.asarray([float(row["bulge_fraction_proxy"]) for row in rows])
    bar = np.asarray([float(row["bar_component_count"]) for row in rows])
    broken = np.asarray([float(row["disk_break_present"]) for row in rows])
    resolution = np.asarray(
        [float(row["seeing_arcsec"]) / float(row["disk_scale_arcsec"]) for row in rows]
    )
    base = np.column_stack(
        (
            log_gbar,
            log_x,
            mass,
            mu0,
            inclination,
            morphology,
            bulge,
            bar,
            broken,
            resolution,
        )
    )
    if kind == "mean":
        return np.column_stack(
            (
                base,
                log_gbar**2,
                log_x**2,
                log_x**3,
                log_gbar * log_x,
                log_x * mass,
                log_x * mu0,
                log_gbar * bulge,
                log_x * resolution,
            )
        )
    if kind == "variance":
        return np.column_stack(
            (
                base,
                log_gbar**2,
                log_x**2,
                log_gbar * log_x,
                log_x * resolution,
                inclination * resolution,
                bar * log_x,
                broken * log_x,
            )
        )
    raise GravityItem41Error(f"unknown design: {kind}")


def _ridge_fit(
    design: np.ndarray, target: np.ndarray, weight: np.ndarray, alpha: float
) -> dict[str, np.ndarray]:
    mean = np.mean(design, axis=0)
    spread = np.std(design, axis=0)
    spread = np.where(spread > 1e-12, spread, 1.0)
    matrix = np.column_stack((np.ones(len(design)), (design - mean) / spread))
    weighted = matrix * np.sqrt(weight)[:, None]
    penalty = np.eye(matrix.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficient = np.linalg.pinv(weighted.T @ weighted + penalty) @ (
        weighted.T @ (target * np.sqrt(weight))
    )
    return {"mean": mean, "spread": spread, "coefficient": coefficient}


def _ridge_predict(model: Mapping[str, np.ndarray], design: np.ndarray) -> np.ndarray:
    matrix = np.column_stack(
        (np.ones(len(design)), (design - model["mean"]) / model["spread"])
    )
    return matrix @ model["coefficient"]


def _mean_models(
    rows: Sequence[Mapping[str, Any]], arrays: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> dict[str, Any]:
    design = _design(rows, root, "mean")
    target = np.asarray(arrays["y_mean"])
    point_weight = np.asarray(arrays["point_weight"])
    point_folds = np.asarray(arrays["point_folds"])
    alpha = float(config["evaluation"]["mean_ridge_alpha"])
    oof = np.empty_like(target)
    train_predictions: list[np.ndarray] = []
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = point_folds != fold
        model = _ridge_fit(design[train], target[train], point_weight[train], alpha)
        prediction = _ridge_predict(model, design)
        oof[~train] = prediction[~train]
        train_predictions.append(prediction)
    full_model = _ridge_fit(design, target, point_weight, alpha)
    return {
        "oof": oof,
        "train_by_fold": train_predictions,
        "full": _ridge_predict(full_model, design),
    }


def _joint_nll(
    y_mean: np.ndarray,
    y_difference: np.ndarray,
    predicted_mean: np.ndarray,
    variance: np.ndarray,
    floor: float,
) -> np.ndarray:
    variance = np.maximum(variance, floor)
    mean_variance = 0.5 * variance
    return 0.5 * (
        np.square(y_mean - predicted_mean) / mean_variance
        + np.log(mean_variance)
        + np.square(y_difference) / variance
        + np.log(variance)
    )


def _stochastic_moments_backend(
    candidates: Mapping[str, np.ndarray],
    begin: int,
    end: int,
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    xp: Any,
) -> tuple[Any, Any]:
    rows = {key: value[begin:end] for key, value in candidates.items()}
    sigma, exponent, transition, radial = _candidate_parameters(rows, config)
    lane = xp.asarray(rows["lane"], dtype=xp.int8)
    sig = xp.asarray(sigma, dtype=xp.float64)[:, None]
    qq = xp.asarray(exponent, dtype=xp.float64)[:, None]
    tt = xp.asarray(transition, dtype=xp.float64)[:, None]
    ll = xp.asarray(radial, dtype=xp.float64)[:, None]
    u = xp.asarray(arrays["u"], dtype=xp.float64)[None, :]
    x = xp.asarray(arrays["x"], dtype=xp.float64)[None, :]
    window = 1.0 / (1.0 + xp.power(u / tt, qq))
    drift = xp.zeros((end - begin, u.shape[1]), dtype=xp.float64)
    variance = xp.zeros_like(drift)
    for lane_id in range(4):
        mask = lane == lane_id
        if not bool(_to_numpy(xp.any(mask), xp)):
            continue
        local_window = window[mask]
        local_sigma = sig[mask]
        if lane_id == 0:
            variance[mask] = xp.square(local_sigma) * local_window
        elif lane_id == 1:
            saturation = 1.0 - xp.exp(-x / ll[mask])
            variance[mask] = xp.square(local_sigma) * local_window * saturation
            drift[mask] = -0.5 * variance[mask]
        elif lane_id == 2:
            variance[mask] = xp.square(local_sigma) * local_window
            drift[mask] = 0.5 * qq[mask] * variance[mask] * (1.0 - local_window)
        else:
            probability = xp.clip(1.0 - xp.exp(-x / ll[mask]), 1e-8, 1.0 - 1e-8)
            drift[mask] = local_sigma * local_window * (2.0 * probability - 1.0)
            variance[mask] = (
                4.0
                * xp.square(local_sigma)
                * xp.square(local_window)
                * probability
                * (1.0 - probability)
            )
    return drift, variance


def _screen_candidates(
    candidates: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    means: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    xp, backend, device = _backend()
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    y_mean = xp.asarray(arrays["y_mean"], dtype=xp.float64)[None, :]
    y_difference = xp.asarray(arrays["y_difference"], dtype=xp.float64)[None, :]
    point_weight = xp.asarray(arrays["point_weight"], dtype=xp.float64)[None, :]
    point_folds = np.asarray(arrays["point_folds"])
    object_folds = np.asarray(arrays["object_folds"])
    floor = float(config["evaluation"]["variance_floor"])
    folds = list(range(int(config["evaluation"]["outer_folds"])))
    if sorted(set(object_folds.tolist())) != folds:
        raise GravityItem41Error("GHASP folds are incomplete")
    masks = [xp.asarray(point_folds != fold, dtype=xp.float64)[None, :] for fold in folds]
    train_objects = [int(np.sum(object_folds != fold)) for fold in folds]
    train_means = [
        xp.asarray(means["train_by_fold"][fold], dtype=xp.float64)[None, :]
        for fold in folds
    ]
    full_mean = xp.asarray(means["full"], dtype=xp.float64)[None, :]
    best_loss = np.full(len(folds), np.inf)
    best_index = np.full(len(folds), -1, dtype=np.int64)
    full_best_loss = math.inf
    full_best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        drift, variance = _stochastic_moments_backend(
            candidates, begin, end, arrays, config, xp
        )
        variance = xp.maximum(variance, floor)
        for fold in folds:
            residual = y_mean - (train_means[fold] + drift)
            nll = 0.5 * (
                xp.square(residual) / (0.5 * variance)
                + xp.log(0.5 * variance)
                + xp.square(y_difference) / variance
                + xp.log(variance)
            )
            score = xp.sum(nll * point_weight * masks[fold], axis=1) / train_objects[fold]
            local = int(_to_numpy(xp.argmin(score), xp))
            value = float(_to_numpy(score[local], xp))
            if value < best_loss[fold]:
                best_loss[fold] = value
                best_index[fold] = begin + local
        residual = y_mean - (full_mean + drift)
        nll = 0.5 * (
            xp.square(residual) / (0.5 * variance)
            + xp.log(0.5 * variance)
            + xp.square(y_difference) / variance
            + xp.log(variance)
        )
        score = xp.sum(nll * point_weight, axis=1) / len(arrays["galaxies"])
        local = int(_to_numpy(xp.argmin(score), xp))
        value = float(_to_numpy(score[local], xp))
        if value < full_best_loss:
            full_best_loss = value
            full_best_index = begin + local
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    elapsed = time.perf_counter() - started
    selected_drift = np.empty(len(arrays["u"]), dtype=np.float64)
    selected_variance = np.empty(len(arrays["u"]), dtype=np.float64)
    for fold in folds:
        index = int(best_index[fold])
        rows = {key: value[index : index + 1] for key, value in candidates.items()}
        drift, variance = stochastic_moments(
            rows, np.asarray(arrays["u"]), np.asarray(arrays["x"]), config
        )
        held = point_folds == fold
        selected_drift[held] = drift[0, held]
        selected_variance[held] = variance[0, held]
    full_rows = {
        key: value[full_best_index : full_best_index + 1] for key, value in candidates.items()
    }
    full_drift, full_variance = stochastic_moments(
        full_rows, np.asarray(arrays["u"]), np.asarray(arrays["x"]), config
    )
    crosscheck = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]),
        len(candidates["candidate_id"]),
    )
    cpu_drift, cpu_variance = _stochastic_moments_backend(
        candidates, 0, crosscheck, arrays, config, np
    )
    device_drift, device_variance = _stochastic_moments_backend(
        candidates, 0, crosscheck, arrays, config, xp
    )
    difference = max(
        float(np.max(np.abs(cpu_drift - _to_numpy(device_drift, xp)))),
        float(np.max(np.abs(cpu_variance - _to_numpy(device_variance, xp)))),
    )
    return (
        {
            "selected_admissible_indices": best_index.tolist(),
            "selected_training_losses": best_loss.tolist(),
            "oof_drift": selected_drift,
            "oof_variance": selected_variance,
            "full_selected_admissible_index": int(full_best_index),
            "full_training_loss": float(full_best_loss),
            "full_drift": full_drift[0],
            "full_variance": full_variance[0],
        },
        {
            "backend": backend,
            "device": device,
            "search_seconds": elapsed,
            "candidate_point_evaluations": int(
                len(candidates["candidate_id"])
                * len(arrays["u"])
                * (int(config["evaluation"]["outer_folds"]) + 1)
            ),
            "cpu_gpu_max_abs_moment_difference": difference,
            "cpu_gpu_tolerance": float(config["evaluation"]["cpu_gpu_tolerance"]),
            "cpu_gpu_passed": difference
            <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        },
    )


def _control_predictions(
    rows: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, Any],
    means: Mapping[str, Any],
    config: Mapping[str, Any],
    root: Path,
) -> dict[str, dict[str, np.ndarray]]:
    point_folds = np.asarray(arrays["point_folds"])
    point_weight = np.asarray(arrays["point_weight"])
    y_mean = np.asarray(arrays["y_mean"])
    y_difference = np.asarray(arrays["y_difference"])
    variance_design = _design(rows, root, "variance")
    floor = float(config["evaluation"]["variance_floor"])
    alpha = float(config["evaluation"]["ordinary_variance_ridge_alpha"])
    homoskedastic = np.empty_like(y_mean)
    heteroskedastic = np.empty_like(y_mean)
    for fold in range(int(config["evaluation"]["outer_folds"])):
        train = point_folds != fold
        held = ~train
        train_mean = np.asarray(means["train_by_fold"][fold])
        energy = np.square(y_mean - train_mean) + 0.5 * np.square(y_difference)
        constant = float(np.average(energy[train], weights=point_weight[train]))
        homoskedastic[held] = max(constant, floor)
        target = np.log(np.maximum(energy[train], floor))
        model = _ridge_fit(
            variance_design[train], target, point_weight[train], alpha
        )
        raw_train = np.exp(_ridge_predict(model, variance_design[train]))
        calibration = float(
            np.average(
                energy[train] / np.maximum(raw_train, floor),
                weights=point_weight[train],
            )
        )
        heteroskedastic[held] = np.maximum(
            np.exp(_ridge_predict(model, variance_design[held])) * calibration,
            floor,
        )
    return {
        "homoskedastic": {"mean": np.asarray(means["oof"]), "variance": homoskedastic},
        "ordinary_heteroskedastic": {
            "mean": np.asarray(means["oof"]),
            "variance": heteroskedastic,
        },
    }


def _object_loss(
    arrays: Mapping[str, Any], point_loss: np.ndarray
) -> np.ndarray:
    object_index = np.asarray(arrays["object_index"])
    result = np.zeros(len(arrays["galaxies"]), dtype=np.float64)
    for index in range(len(result)):
        result[index] = float(np.mean(point_loss[object_index == index]))
    return result


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def _permutation(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    random = np.random.Generator(
        np.random.PCG64(int(config["evaluation"]["permutation_seed"]))
    )
    trials = int(config["evaluation"]["paired_sign_flip_permutations"])
    observed = float(np.mean(differences))
    null = [
        float(np.mean(differences * random.choice([-1.0, 1.0], len(differences))))
        for _ in range(trials)
    ]
    return {
        "trials": trials,
        "observed_mean_candidate_minus_reference_nll": observed,
        "p_value": (1.0 + sum(value <= observed for value in null)) / (trials + 1.0),
        "selection_aware": False,
    }


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    rows, arrays = _arrays(root, config)
    means = _mean_models(rows, arrays, config, root)
    candidates, candidate_audit = admissible_candidates(config)
    screened, compute = _screen_candidates(candidates, arrays, means, config)
    floor = float(config["evaluation"]["variance_floor"])
    candidate_mean = np.asarray(means["oof"]) + np.asarray(screened["oof_drift"])
    candidate_variance = np.maximum(np.asarray(screened["oof_variance"]), floor)
    candidate_point_nll = _joint_nll(
        np.asarray(arrays["y_mean"]),
        np.asarray(arrays["y_difference"]),
        candidate_mean,
        candidate_variance,
        floor,
    )
    controls = _control_predictions(rows, arrays, means, config, root)
    point_nll = {
        name: _joint_nll(
            np.asarray(arrays["y_mean"]),
            np.asarray(arrays["y_difference"]),
            value["mean"],
            value["variance"],
            floor,
        )
        for name, value in controls.items()
    }
    candidate_object = _object_loss(arrays, candidate_point_nll)
    control_object = {name: _object_loss(arrays, value) for name, value in point_nll.items()}
    losses = {
        "candidate": float(np.mean(candidate_object)),
        **{name: float(np.mean(value)) for name, value in control_object.items()},
    }
    strongest_name = min(control_object, key=lambda name: losses[name])
    strongest_object = control_object[strongest_name]
    strongest_point = point_nll[strongest_name]
    improvement = _improvement(losses[strongest_name], losses["candidate"])
    differences = candidate_object - strongest_object
    order = np.argsort(np.abs(differences))[::-1]
    leave = np.ones(len(differences), dtype=bool)
    leave[order[0]] = False
    leave_improvement = _improvement(
        float(np.mean(strongest_object[leave])), float(np.mean(candidate_object[leave]))
    )
    trim_count = max(
        1,
        math.floor(float(config["evaluation"]["robust_trim_fraction"]) * len(differences)),
    )
    trim = np.ones(len(differences), dtype=bool)
    trim[order[:trim_count]] = False
    trim_improvement = _improvement(
        float(np.mean(strongest_object[trim])), float(np.mean(candidate_object[trim]))
    )
    raw = differences > 0.0
    point_difference = candidate_point_nll - strongest_point
    object_index = np.asarray(arrays["object_index"])
    resolved = np.zeros(len(differences), dtype=bool)
    for index in range(len(resolved)):
        values = point_difference[object_index == index]
        standard_error = (
            float(np.std(values, ddof=1)) / math.sqrt(len(values))
            if len(values) > 1
            else math.inf
        )
        resolved[index] = float(np.mean(values)) - 1.96 * standard_error > 0.0
    quality_passed = (
        len(arrays["galaxies"]) >= int(config["gates"]["minimum_quality_galaxies"])
        and len(rows) >= int(config["gates"]["minimum_quality_points"])
    )
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(arrays["galaxies"]),
        "raw_counterexample_count": int(np.sum(raw)),
        "quality_verified_counterexample_count": int(np.sum(raw)),
        "uncertainty_resolved_counterexample_count": int(np.sum(raw & resolved)),
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": quality_passed,
        "strongest_baseline_failed": improvement < 0.0,
        "leave_one_changes_sign": bool((improvement >= 0.0) != (leave_improvement >= 0.0)),
        "trim_changes_sign": bool((improvement >= 0.0) != (trim_improvement >= 0.0)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        report, load_counterexample_policy(root / POLICY_PATH)
    )
    base_mean_mse = float(np.mean(np.square(np.asarray(arrays["y_mean"]) - means["oof"])))
    candidate_mean_mse = float(
        np.mean(np.square(np.asarray(arrays["y_mean"]) - candidate_mean))
    )
    variance_spearman = spearmanr(
        candidate_variance,
        np.square(np.asarray(arrays["y_difference"])),
    )
    selected_by_fold = [
        decode_candidate(int(candidates["candidate_id"][index]), config)
        for index in screened["selected_admissible_indices"]
    ]
    full_selected = decode_candidate(
        int(candidates["candidate_id"][screened["full_selected_admissible_index"]]),
        config,
    )
    permutation = _permutation(differences, config)
    gates = {
        "quality_passes": quality_passed,
        "beats_homoskedastic_nll": losses["candidate"] < losses["homoskedastic"],
        "beats_ordinary_heteroskedastic_nll": losses["candidate"]
        < losses["ordinary_heteroskedastic"],
        "mean_mse_within_tolerance": candidate_mean_mse
        <= base_mean_mse
        * (1.0 + float(config["gates"]["candidate_mean_mse_may_not_worsen_more_than_fraction"])),
        "paired_p_at_most_0p05": permutation["p_value"]
        <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_and_trim_stable": not report["leave_one_changes_sign"]
        and not report["trim_changes_sign"],
        "cpu_gpu_agreement": bool(compute["cpu_gpu_passed"]),
        "confirmation_values_read_zero": True,
        "fresh_confirmation": False,
        "clash_transfer_pending": True,
    }
    lead_gates = {
        key: value
        for key, value in gates.items()
        if key not in {"fresh_confirmation", "clash_transfer_pending"}
    }
    if all(lead_gates.values()):
        decision = "RETROSPECTIVE_ITEM41_STOCHASTIC_LEAD_PENDING_UNCHANGED_CLASH"
    elif losses["candidate"] < losses["homoskedastic"]:
        decision = "NONPROMOTED_ITEM41_PARTIAL_STOCHASTIC_PATTERN_RETAINED"
    else:
        decision = "NONPROMOTED_ITEM41_STOCHASTIC_GHASP_NEGATIVE_RETAINED"
    object_records = [
        {
            "galaxy": name,
            "candidate_nll": float(candidate_object[index]),
            "strongest_reference_nll": float(strongest_object[index]),
            "candidate_minus_reference_nll": float(differences[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(raw[index] and resolved[index]),
        }
        for index, name in enumerate(arrays["galaxies"])
    ]
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item41-compute-manifest-1.0",
            "item": 41,
            "decision": decision,
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "candidate_manifest_sha256": _sha256_file(
                    _source_path(root, config, "candidate_manifest")
                ),
                "exposure_manifest_sha256": _sha256_file(
                    _source_path(root, config, "exposure_manifest")
                ),
                "retrospective_response_source": True,
                "response_values_read_during_candidate_generation": 0,
                "confirmation_values_read": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
            },
            "quality": {
                "passed": quality_passed,
                "galaxies": len(arrays["galaxies"]),
                "paired_points": len(rows),
                "prior_item28_formal_quality_pass": arrays["prior_quality"][
                    "formal_quality_pass"
                ],
                "claim": "Item 41 has a narrower paired-side representation threshold; passing it does not retroactively change Item 28's failed quality gate.",
            },
            "candidate_search": {
                **candidate_audit,
                **compute,
                "selected_fold_candidates": selected_by_fold,
                "selected_fold_training_losses": screened["selected_training_losses"],
                "full_retrospective_candidate": full_selected,
                "full_training_loss": screened["full_training_loss"],
            },
            "joint_mean_variance_result": {
                "losses": losses,
                "strongest_control": strongest_name,
                "improvement_vs_strongest_percent": improvement,
                "base_mean_mse": base_mean_mse,
                "candidate_mean_mse": candidate_mean_mse,
                "variance_vs_squared_side_difference_spearman": {
                    "statistic": float(variance_spearman.statistic),
                    "p_value": float(variance_spearman.pvalue),
                },
                "paired_sign_flip": permutation,
                "robustness": {
                    "most_influential_galaxy": arrays["galaxies"][int(order[0])],
                    "leave_one_improvement_percent": leave_improvement,
                    "leave_one_changes_sign": report["leave_one_changes_sign"],
                    "trimmed_galaxies": trim_count,
                    "trim_improvement_percent": trim_improvement,
                    "trim_changes_sign": report["trim_changes_sign"],
                },
                "object_level": object_records,
                "counterexample_policy_report": report,
                "counterexample_assessment": assessment,
            },
            "gates": gates,
            "claim_boundaries": {
                "fresh_confirmation": False,
                "causal_stochastic_process_established": False,
                "intrinsic_gravity_noise_separated_from_kinematic_asymmetry": False,
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "historical_novelty_established": False,
                "formula_pruned": False,
                "formula_family_pruned": False,
                "one_empirical_counterexample_is_veto": False,
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
    _verify_content_hash(existing, "Item 41 compute manifest")
    replay = evaluate(root, write=False)
    if _scientific_payload(existing) != _scientific_payload(replay):
        raise GravityItem41Error("Item 41 scientific replay changed")
    return {
        "status": "ITEM41_GHASP_REPLAY_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "confirmation_values_read": existing["protocol"]["confirmation_values_read"],
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
