"""Evaluate frozen Item 42 matter-geometry feedback on fresh WALLABY responses."""

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
from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import (
    POLICY_PATH,
    GravityItem42Error,
    _candidate_parameters,
    _source_path,
    admissible_candidates,
    decode_candidate,
    generate_raw_candidates,
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
        raise GravityItem42Error("duplicate Item 42 response point")
    rows = []
    for feature in features:
        response = response_by_key.pop(tuple(feature[key] for key in keys), None)
        if response is None:
            raise GravityItem42Error("Item 42 feature lacks aligned response")
        rows.append({**feature, **response})
    if response_by_key:
        raise GravityItem42Error("Item 42 response lacks aligned feature")
    return rows


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
            raise GravityItem42Error("one galaxy entered multiple folds")
        fold_by_name[name] = fold
    object_folds = np.asarray([fold_by_name[name] for name in galaxies], dtype=np.int64)
    point_folds = object_folds[object_index]
    observed = np.asarray([float(row["observed_speed_km_s"]) for row in rows])
    observed_error = np.asarray(
        [float(row["observed_speed_error_km_s"]) for row in rows]
    )
    h = np.empty((4, 16, len(rows)), dtype=np.float64)
    for lane in range(4):
        for gain in range(16):
            h[lane, gain] = np.asarray(
                [float(row[f"h_lane{lane}_feedback{gain}"]) for row in rows]
            )
    vbar = np.asarray([float(row["vbar_km_s"]) for row in rows])
    if np.any(~np.isfinite(vbar)) or np.any(vbar <= 0.0):
        raise GravityItem42Error("Item 42 baryonic velocity must be finite and positive")
    return {
        "galaxies": galaxies,
        "object_index": object_index,
        "object_folds": object_folds,
        "point_folds": point_folds,
        "point_weight": point_weight,
        "observed": observed,
        "log_observed": np.log10(observed),
        "log_sigma": np.maximum(
            observed_error / (observed * math.log(10.0)),
            0.1,
        ),
        "u": np.asarray([float(row["u"]) for row in rows]),
        "vbar": vbar,
        "h": h,
    }


def _candidate_pools(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    admitted, audit = admissible_candidates(config)
    sample = _read_json(_source_path(root, config, "sample_manifest"))
    valid = np.zeros((4, 16), dtype=bool)
    for row in sample["feedback_cells_valid_across_all_predictors"]:
        valid[int(row["lane"]), int(row["feedback_index"])] = bool(row["valid"])
    keep = valid[
        np.asarray(admitted["lane"], dtype=int),
        np.asarray(admitted["feedback_index"], dtype=int),
    ]
    candidates = {key: value[keep] for key, value in admitted.items()}
    raw = generate_raw_candidates(config)
    admitted_ids = set(np.asarray(admitted["candidate_id"], dtype=np.int64).tolist())
    no_feedback_ids = []
    cells = int(config["candidate_generator"]["cells_per_niche"])
    for local in range(0, cells, 16):
        if local + 1 in admitted_ids:
            no_feedback_ids.append(local)
    indices = np.asarray(no_feedback_ids, dtype=np.int64)
    no_feedback = {key: value[indices] for key, value in raw.items()}
    return candidates, no_feedback, {
        **audit,
        "predictor_specific_nonconvergent_candidates_removed": int(np.sum(~keep)),
        "predictor_specific_admitted_candidates": int(np.sum(keep)),
        "matched_no_feedback_candidates": len(indices),
        "valid_feedback_cells": int(np.sum(valid)),
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
    amplitude, exponent, transition, _ = _candidate_parameters(rows, config)
    lane = xp.asarray(rows["lane"], dtype=xp.int64)
    feedback = xp.asarray(rows["feedback_index"], dtype=xp.int64)
    aa = xp.asarray(amplitude, dtype=xp.float64)[:, None]
    pp = xp.asarray(exponent, dtype=xp.float64)[:, None]
    tt = xp.asarray(transition, dtype=xp.float64)[:, None]
    u = xp.asarray(arrays["u"], dtype=xp.float64)[None, :]
    h_library = xp.asarray(arrays["h"], dtype=xp.float64)
    h = h_library[lane, feedback]
    multiplier = 1.0 + aa * xp.power(u, -pp) / (1.0 + u / tt) * (
        0.05 + 0.95 * h
    )
    return xp.log10(xp.asarray(arrays["vbar"], dtype=xp.float64))[None, :] + (
        0.5 * xp.log10(multiplier)
    )


def _screen_pool(
    candidates: Mapping[str, np.ndarray],
    arrays: Mapping[str, Any],
    config: Mapping[str, Any],
    xp: Any,
) -> dict[str, Any]:
    object_folds = np.asarray(arrays["object_folds"])
    point_folds = np.asarray(arrays["point_folds"])
    folds = list(range(int(config["sample_boundary"]["outer_folds"])))
    if sorted(set(object_folds.tolist())) != folds:
        raise GravityItem42Error("response-complete Item 42 folds are incomplete")
    y = xp.asarray(arrays["log_observed"], dtype=xp.float64)[None, :]
    sigma = xp.asarray(arrays["log_sigma"], dtype=xp.float64)[None, :]
    weight = xp.asarray(arrays["point_weight"], dtype=xp.float64)[None, :]
    masks = [xp.asarray(point_folds != fold, dtype=xp.float64)[None, :] for fold in folds]
    train_objects = [int(np.sum(object_folds != fold)) for fold in folds]
    best_loss = np.full(len(folds), np.inf)
    best_index = np.full(len(folds), -1, dtype=np.int64)
    full_best_loss = math.inf
    full_best_index = -1
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    for begin in range(0, len(candidates["candidate_id"]), batch_size):
        end = min(begin + batch_size, len(candidates["candidate_id"]))
        prediction = _candidate_log_velocity_batch(
            candidates, begin, end, arrays, config, xp
        )
        loss = xp.square((prediction - y) / sigma)
        for fold in folds:
            score = xp.sum(loss * weight * masks[fold], axis=1) / train_objects[fold]
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
    selected_prediction = np.empty(len(arrays["u"]), dtype=np.float64)
    for fold in folds:
        index = int(best_index[fold])
        if index < 0:
            raise GravityItem42Error("no finite Item 42 candidate survived a fold search")
        prediction = _candidate_log_velocity_batch(
            candidates, index, index + 1, arrays, config, np
        )[0]
        held = point_folds == fold
        selected_prediction[held] = prediction[held]
    if full_best_index < 0:
        raise GravityItem42Error("no finite Item 42 candidate survived the full search")
    full_prediction = _candidate_log_velocity_batch(
        candidates, full_best_index, full_best_index + 1, arrays, config, np
    )[0]
    return {
        "selected_admissible_indices": best_index.tolist(),
        "selected_training_losses": best_loss.tolist(),
        "oof_prediction": selected_prediction,
        "full_selected_admissible_index": int(full_best_index),
        "full_training_loss": float(full_best_loss),
        "full_prediction": full_prediction,
    }


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


def _ordinary_design(rows: Sequence[Mapping[str, str]]) -> np.ndarray:
    log_u = np.log10([float(row["u"]) for row in rows])
    log_x = np.log10([float(row["radius_over_source"]) for row in rows])
    log_sigma = np.log10(
        np.maximum([float(row["local_hi_surface_density"]) for row in rows], 1e-6)
    )
    log_mass = np.log10([float(row["total_hi_mass_msun"]) for row in rows]) - 9.0
    concentration = np.asarray(
        [float(row["half_hi_mass_radius_fraction"]) for row in rows]
    )
    return np.column_stack(
        (
            log_u,
            log_x,
            log_sigma,
            log_mass,
            concentration,
            np.square(log_u),
            np.square(log_x),
            log_u * log_x,
            log_x * log_sigma,
            log_x * concentration,
        )
    )


def _ordinary_oof(
    rows: Sequence[Mapping[str, str]], arrays: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, list[float]]:
    design = _ordinary_design(rows)
    target = np.asarray(arrays["log_observed"])
    weight = np.asarray(arrays["point_weight"])
    point_folds = np.asarray(arrays["point_folds"])
    object_index = np.asarray(arrays["object_index"])
    alphas = [float(value) for value in config["evaluation"]["ordinary_ridge_alphas"]]
    prediction = np.empty_like(target)
    selected_alphas = []
    for fold in range(int(config["sample_boundary"]["outer_folds"])):
        train = point_folds != fold
        held = ~train
        train_objects = sorted(set(object_index[train].tolist()))
        scores = []
        for alpha in alphas:
            values = []
            for object_id in train_objects:
                inner_held = train & (object_index == object_id)
                inner_train = train & ~inner_held
                model = _ridge_fit(
                    design[inner_train], target[inner_train], weight[inner_train], alpha
                )
                residual = (
                    _ridge_predict(model, design[inner_held]) - target[inner_held]
                ) / np.asarray(arrays["log_sigma"])[inner_held]
                values.append(float(np.mean(np.square(residual))))
            scores.append(float(np.mean(values)))
        alpha = alphas[int(np.argmin(scores))]
        selected_alphas.append(alpha)
        model = _ridge_fit(design[train], target[train], weight[train], alpha)
        prediction[held] = _ridge_predict(model, design[held])
    return prediction, selected_alphas


def _fixed_predictions(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    u = np.asarray(arrays["u"])
    log_vbar = np.log10(np.asarray(arrays["vbar"]))
    mond = 1.0 / (1.0 - np.exp(-np.sqrt(u)))
    return {
        "gas_only_baryonic_newton": log_vbar,
        "gas_only_mond_RAR": log_vbar + 0.5 * np.log10(mond),
    }


def _object_loss(arrays: Mapping[str, Any], prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    point = np.square(
        (prediction - np.asarray(arrays["log_observed"]))
        / np.asarray(arrays["log_sigma"])
    )
    object_index = np.asarray(arrays["object_index"])
    result = np.asarray(
        [float(np.mean(point[object_index == index])) for index in range(len(arrays["galaxies"]))]
    )
    return point, result


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
        "observed_mean_candidate_minus_reference_loss": observed,
        "p_value": (1.0 + sum(value <= observed for value in null)) / (trials + 1.0),
        "selection_aware": False,
    }


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    rows = _load_rows(root, config)
    arrays = _point_arrays(rows)
    candidates, no_feedback, candidate_audit = _candidate_pools(root, config)
    xp, backend, device = _backend()
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    started = time.perf_counter()
    screened = _screen_pool(candidates, arrays, config, xp)
    no_feedback_screened = _screen_pool(no_feedback, arrays, config, xp)
    if backend == "gpu_cupy":
        xp.cuda.Stream.null.synchronize()
    search_seconds = time.perf_counter() - started
    crosscheck = min(
        int(config["evaluation"]["cpu_crosscheck_candidates"]),
        len(candidates["candidate_id"]),
    )
    cpu = _candidate_log_velocity_batch(candidates, 0, crosscheck, arrays, config, np)
    device_result = _candidate_log_velocity_batch(
        candidates, 0, crosscheck, arrays, config, xp
    )
    difference = float(np.max(np.abs(cpu - _to_numpy(device_result, xp))))
    ordinary, selected_alphas = _ordinary_oof(rows, arrays, config)
    predictions = {
        "candidate": np.asarray(screened["oof_prediction"]),
        "matched_no_feedback": np.asarray(no_feedback_screened["oof_prediction"]),
        "ordinary_radial_hi_ridge": ordinary,
        **_fixed_predictions(arrays),
    }
    point_losses: dict[str, np.ndarray] = {}
    object_losses: dict[str, np.ndarray] = {}
    for name, prediction in predictions.items():
        point_losses[name], object_losses[name] = _object_loss(arrays, prediction)
    losses = {name: float(np.mean(value)) for name, value in object_losses.items()}
    control_names = [name for name in predictions if name != "candidate"]
    strongest_name = min(control_names, key=lambda name: losses[name])
    candidate_object = object_losses["candidate"]
    strongest_object = object_losses[strongest_name]
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
    point_difference = point_losses["candidate"] - point_losses[strongest_name]
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
    extraction = _read_json(_source_path(root, config, "extraction_summary"))
    passing = int(extraction["counts"]["quality_passing_galaxies"])
    requested = int(extraction["counts"]["exploration_response_rows"])
    quality_passed = (
        passing
        >= int(config["response_quality"]["minimum_quality_passing_exploration_galaxies"])
        and passing / requested
        >= float(config["response_quality"]["minimum_quality_retention_fraction"])
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
        "independent_failure_strata": 1,
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
        int(candidates["candidate_id"][screened["full_selected_admissible_index"]]),
        config,
    )
    no_feedback_selected = decode_candidate(
        int(
            no_feedback["candidate_id"][
                no_feedback_screened["full_selected_admissible_index"]
            ]
        ),
        config,
    )
    permutation = _permutation(differences, config)
    gates = {
        "quality_passes": quality_passed,
        "beats_gas_only_newton": losses["candidate"] < losses["gas_only_baryonic_newton"],
        "beats_gas_only_mond": losses["candidate"] < losses["gas_only_mond_RAR"],
        "beats_matched_no_feedback": losses["candidate"] < losses["matched_no_feedback"],
        "beats_ordinary_ridge": losses["candidate"] < losses["ordinary_radial_hi_ridge"],
        "paired_p_at_most_0p05": permutation["p_value"]
        <= float(config["gates"]["paired_p_maximum"]),
        "leave_one_and_trim_stable": not report["leave_one_changes_sign"]
        and not report["trim_changes_sign"],
        "cpu_gpu_agreement": difference <= float(config["evaluation"]["cpu_gpu_tolerance"]),
        "complete_baryonic_inventory": False,
        "confirmation_remains_sealed": True,
        "clash_transfer_pending": True,
    }
    scientific_gates = {
        key: value
        for key, value in gates.items()
        if key
        not in {
            "complete_baryonic_inventory",
            "confirmation_remains_sealed",
            "clash_transfer_pending",
        }
    }
    if all(scientific_gates.values()):
        decision = "ITEM42_FRESH_FEEDBACK_LEAD_WITH_MISSING_STELLAR_LIMIT_PENDING_CLASH"
    elif losses["candidate"] < losses["matched_no_feedback"]:
        decision = "NONPROMOTED_ITEM42_PARTIAL_FEEDBACK_PATTERN_RETAINED"
    else:
        decision = "NONPROMOTED_ITEM42_FEEDBACK_DYNAMICS_NEGATIVE_RETAINED"
    object_records = [
        {
            "galaxy": name,
            "candidate_loss": float(candidate_object[index]),
            "strongest_reference_loss": float(strongest_object[index]),
            "candidate_minus_reference": float(differences[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(raw[index] and resolved[index]),
            "missing_stellar_counterpart": True,
        }
        for index, name in enumerate(arrays["galaxies"])
    ]
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-compute-manifest-1.0",
            "item": 42,
            "decision": decision,
            "protocol": {
                "scientific_freeze_commit": config["scientific_freeze_commit"],
                "sample_freeze_commit": config["sample_freeze_commit"],
                "candidate_manifest_sha256": _sha256_file(
                    _source_path(root, config, "candidate_manifest")
                ),
                "sample_manifest_sha256": _sha256_file(
                    _source_path(root, config, "sample_manifest")
                ),
                "response_source_sha256": _sha256_file(
                    _source_path(root, config, "wallaby_response_source")
                ),
                "response_values_read_during_candidate_generation": 0,
                "confirmation_response_rows": 0,
                "post_response_candidate_cells": 0,
                "paid_model_calls": 0,
                "post_response_implementation_repair": {
                    "count": 1,
                    "reason": "the first evaluator attempt halted because radii inside the first HI annulus received zero enclosed gas",
                    "repair": "use constant central surface density so M_HI(<r) scales as r^2 inside the first annulus",
                    "formula_space_changed": False,
                    "sample_changed": False,
                    "response_points_removed": 0,
                },
            },
            "quality": {
                "passed": quality_passed,
                "requested_exploration_galaxies": requested,
                "passing_galaxies": passing,
                "failing_galaxies": int(
                    extraction["counts"]["quality_failing_galaxies"]
                ),
                "accepted_rotation_points": len(rows),
                "complete_baryonic_inventory": False,
                "missing_stellar_counterparts": True,
            },
            "candidate_search": {
                **candidate_audit,
                "backend": backend,
                "device": device,
                "search_seconds": search_seconds,
                "candidate_point_evaluations": int(
                    len(candidates["candidate_id"])
                    * len(rows)
                    * (int(config["sample_boundary"]["outer_folds"]) + 1)
                ),
                "no_feedback_point_evaluations": int(
                    len(no_feedback["candidate_id"])
                    * len(rows)
                    * (int(config["sample_boundary"]["outer_folds"]) + 1)
                ),
                "cpu_gpu_max_abs_log10_velocity_difference": difference,
                "cpu_gpu_tolerance": float(config["evaluation"]["cpu_gpu_tolerance"]),
                "cpu_gpu_passed": gates["cpu_gpu_agreement"],
                "selected_fold_candidates": selected_by_fold,
                "selected_fold_training_losses": screened["selected_training_losses"],
                "full_exploration_candidate": full_selected,
                "full_exploration_training_loss": screened["full_training_loss"],
                "full_no_feedback_control": no_feedback_selected,
                "full_no_feedback_training_loss": no_feedback_screened["full_training_loss"],
            },
            "primary_dynamics": {
                "galaxies": len(arrays["galaxies"]),
                "losses": losses,
                "ordinary_ridge_selected_alphas": selected_alphas,
                "strongest_baseline": strongest_name,
                "improvement_vs_strongest_percent": improvement,
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
                "fresh_exploration_responses": True,
                "fresh_confirmation": False,
                "complete_baryonic_inventory": False,
                "missing_stars_could_explain_part_of_residual": True,
                "dark_matter_excluded": False,
                "modified_gravity_established": False,
                "covariant_theory_established": False,
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
    _verify_content_hash(existing, "Item 42 compute manifest")
    replay = evaluate(root, write=False)
    if _scientific_payload(existing) != _scientific_payload(replay):
        raise GravityItem42Error("Item 42 scientific replay changed")
    return {
        "status": "ITEM42_DYNAMICS_REPLAY_VALID",
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
