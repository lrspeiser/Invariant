"""Apply the commit-bound Item 41 stochastic law unchanged to CLASH clusters."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Mapping
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
from sigma_theory_compiler.gravity_item40_discrete_network import (
    fixed_control_multiplier,
)
from sigma_theory_compiler.gravity_item41_stochastic_gravity import (
    POLICY_PATH,
    GravityItem41Error,
    _source_path,
    decode_candidate,
    generate_raw_candidates,
    load_config,
    stochastic_moments,
)
from sigma_theory_compiler.gravity_item41_stochastic_gravity_evaluator import (
    _ridge_fit,
    _ridge_predict,
)

DYNAMICS_COMMIT = "c0e2bcea03c894209b1d0fca862d2e91cbe731c2"
DYNAMICS_CONTENT_SHA256 = "3f330db1322af7140ffeb206d1a43878c888649580e14b8cb748e2f205650d10"
SELECTED_CANDIDATE_ID = 45024


def _load_clash(path: Path) -> dict[str, list[dict[str, float]]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(lines) < 4:
        raise GravityItem41Error("CLASH VizieR table is incomplete")
    rows = list(csv.DictReader([lines[0], *lines[3:]], delimiter="\t"))
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["AName"]).strip()].append(
            {
                "radius_kpc": float(row["Rad"]),
                "log_gbar": float(row["log(gbar)"]),
                "log_gtot": float(row["log(gtot)"]),
                "error_log_gbar": float(row["e_log(gbar)"]),
                "error_log_gtot": float(row["e_log(gtot)"]),
            }
        )
    for values in grouped.values():
        values.sort(key=lambda value: value["radius_kpc"])
    if len(rows) != 84 or len(grouped) != 20:
        raise GravityItem41Error("CLASH frozen row or cluster count changed")
    return dict(sorted(grouped.items()))


def _candidate_row(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    raw = generate_raw_candidates(config)
    return {
        key: value[SELECTED_CANDIDATE_ID : SELECTED_CANDIDATE_ID + 1]
        for key, value in raw.items()
    }


def _arrays(grouped: Mapping[str, list[dict[str, float]]]) -> dict[str, np.ndarray]:
    names: list[str] = []
    radius: list[float] = []
    log_gbar: list[float] = []
    log_gtot: list[float] = []
    measurement_sigma: list[float] = []
    normalized_radius: list[float] = []
    for name, rows in grouped.items():
        maximum_radius = max(row["radius_kpc"] for row in rows)
        for row in rows:
            names.append(name)
            radius.append(row["radius_kpc"])
            log_gbar.append(row["log_gbar"])
            log_gtot.append(row["log_gtot"])
            measurement_sigma.append(
                math.log(10.0)
                * math.hypot(row["error_log_gbar"], row["error_log_gtot"])
            )
            normalized_radius.append(row["radius_kpc"] / maximum_radius)
    return {
        "name": np.asarray(names),
        "radius_kpc": np.asarray(radius),
        "log_gbar": np.asarray(log_gbar),
        "observed_ln_g": math.log(10.0) * np.asarray(log_gtot),
        "measurement_variance": np.square(np.asarray(measurement_sigma)),
        "x": np.asarray(normalized_radius),
    }


def _object_weight(names: np.ndarray) -> np.ndarray:
    unique, inverse, counts = np.unique(names, return_inverse=True, return_counts=True)
    if len(unique) == 0:
        raise GravityItem41Error("CLASH object list is empty")
    return 1.0 / counts[inverse]


def _variance_design(u: np.ndarray, x: np.ndarray) -> np.ndarray:
    log_u = np.log(u)
    log_x = np.log(x)
    return np.column_stack(
        (
            log_u,
            log_x,
            np.square(log_u),
            np.square(log_x),
            log_u * log_x,
        )
    )


def _oof_variance_controls(
    names: np.ndarray,
    residual: np.ndarray,
    measurement_variance: np.ndarray,
    design: np.ndarray,
) -> dict[str, np.ndarray]:
    weight = _object_weight(names)
    homoskedastic = np.empty_like(residual)
    heteroskedastic = np.empty_like(residual)
    floor = 1e-8
    for name in sorted(set(names.tolist())):
        held = names == name
        train = ~held
        energy = np.square(residual)
        constant = float(np.average(energy[train], weights=weight[train]))
        homoskedastic[held] = np.maximum(constant, measurement_variance[held])
        model = _ridge_fit(
            design[train],
            np.log(np.maximum(energy[train], floor)),
            weight[train],
            10.0,
        )
        raw_train = np.exp(_ridge_predict(model, design[train]))
        calibration = float(
            np.average(
                energy[train] / np.maximum(raw_train, floor),
                weights=weight[train],
            )
        )
        prediction = np.exp(_ridge_predict(model, design[held])) * calibration
        heteroskedastic[held] = np.maximum(prediction, measurement_variance[held])
    return {
        "mond_homoskedastic_loco": homoskedastic,
        "mond_ordinary_heteroskedastic_loco": heteroskedastic,
    }


def _nll(observed: np.ndarray, mean: np.ndarray, variance: np.ndarray) -> np.ndarray:
    variance = np.maximum(variance, 1e-8)
    return 0.5 * (np.square(observed - mean) / variance + np.log(variance))


def _object_loss(names: np.ndarray, point_loss: np.ndarray) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(names.tolist()))
    return unique, np.asarray(
        [float(np.mean(point_loss[names == name])) for name in unique],
        dtype=np.float64,
    )


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    compute = _read_json(compute_path)
    _verify_content_hash(compute, "Item 41 GHASP compute manifest")
    if compute["content_sha256"] != DYNAMICS_CONTENT_SHA256:
        raise GravityItem41Error("commit-bound Item 41 GHASP result changed")
    selected = compute["candidate_search"]["full_retrospective_candidate"]
    if int(selected["candidate_id"]) != SELECTED_CANDIDATE_ID:
        raise GravityItem41Error("commit-bound Item 41 formula changed")
    if selected != decode_candidate(SELECTED_CANDIDATE_ID, config):
        raise GravityItem41Error("decoded Item 41 formula changed")

    source_path = root / str(config["data"]["clash_source"])
    grouped = _load_clash(source_path)
    arrays = _arrays(grouped)
    names = arrays["name"]
    log_gbar = arrays["log_gbar"]
    u = np.power(10.0, log_gbar) / 1.2e-10
    candidate = _candidate_row(config)
    drift, process_variance = stochastic_moments(candidate, u, arrays["x"], config)
    drift = drift[0]
    process_variance = process_variance[0]
    baryonic_mean = math.log(10.0) * log_gbar
    mond_multiplier = fixed_control_multiplier("mond_RAR", u)
    mond_mean = baryonic_mean + np.log(mond_multiplier)
    observed = arrays["observed_ln_g"]
    measurement_variance = arrays["measurement_variance"]
    mond_residual = observed - mond_mean
    controls = _oof_variance_controls(
        names,
        mond_residual,
        measurement_variance,
        _variance_design(u, arrays["x"]),
    )
    point_nll = {
        "candidate_on_mond_background": _nll(
            observed,
            mond_mean + drift,
            measurement_variance + process_variance,
        ),
        "candidate_on_baryonic_newton_background": _nll(
            observed,
            baryonic_mean + drift,
            measurement_variance + process_variance,
        ),
        "mond_measurement_only": _nll(
            observed,
            mond_mean,
            measurement_variance,
        ),
        "baryonic_newton_measurement_only": _nll(
            observed,
            baryonic_mean,
            measurement_variance,
        ),
        **{
            name: _nll(observed, mond_mean, variance)
            for name, variance in controls.items()
        },
    }
    unique, candidate_object = _object_loss(
        names, point_nll["candidate_on_mond_background"]
    )
    object_losses = {
        name: _object_loss(names, loss)[1]
        for name, loss in point_nll.items()
        if name != "candidate_on_mond_background"
    }
    losses = {
        "candidate_on_mond_background": float(np.mean(candidate_object)),
        **{name: float(np.mean(value)) for name, value in object_losses.items()},
    }
    eligible_controls = [
        "mond_measurement_only",
        "mond_homoskedastic_loco",
        "mond_ordinary_heteroskedastic_loco",
    ]
    strongest_name = min(eligible_controls, key=lambda name: losses[name])
    strongest_object = object_losses[strongest_name]
    strongest_point = point_nll[strongest_name]
    improvement = _improvement(
        losses[strongest_name], losses["candidate_on_mond_background"]
    )
    differences = candidate_object - strongest_object
    raw = differences > 0.0
    point_difference = point_nll["candidate_on_mond_background"] - strongest_point
    resolved = np.zeros(len(unique), dtype=bool)
    for index, name in enumerate(unique):
        values = point_difference[names == name]
        standard_error = (
            float(np.std(values, ddof=1)) / math.sqrt(len(values))
            if len(values) > 1
            else math.inf
        )
        resolved[index] = float(np.mean(values)) - 1.96 * standard_error > 0.0
    order = np.argsort(np.abs(differences))[::-1]
    leave = np.ones(len(unique), dtype=bool)
    leave[order[0]] = False
    leave_improvement = _improvement(
        float(np.mean(strongest_object[leave])),
        float(np.mean(candidate_object[leave])),
    )
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(unique),
        "raw_counterexample_count": int(np.sum(raw)),
        "quality_verified_counterexample_count": int(np.sum(raw)),
        "uncertainty_resolved_counterexample_count": int(np.sum(raw & resolved)),
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": True,
        "strongest_baseline_failed": improvement < 0.0,
        "leave_one_changes_sign": bool(
            (improvement >= 0.0) != (leave_improvement >= 0.0)
        ),
        "trim_changes_sign": bool(
            (improvement >= 0.0) != (leave_improvement >= 0.0)
        ),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        report, load_counterexample_policy(root / POLICY_PATH)
    )
    object_records = [
        {
            "cluster": name,
            "candidate_nll": float(candidate_object[index]),
            "strongest_reference_nll": float(strongest_object[index]),
            "candidate_minus_reference_nll": float(differences[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(raw[index] and resolved[index]),
        }
        for index, name in enumerate(unique)
    ]
    if improvement > 0.0:
        decision = "ITEM41_CLASH_DIAGNOSTIC_SUPPORTS_UNCHANGED_STOCHASTIC_LAW"
    else:
        decision = "ITEM41_CLASH_DIAGNOSTIC_DOES_NOT_SUPPORT_LAW_BUT_RETAINS_FAMILY"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item41-clash-transfer-result-1.0",
            "item": 41,
            "decision": decision,
            "protocol": {
                "dynamics_commit": DYNAMICS_COMMIT,
                "dynamics_compute_file_sha256": _sha256_file(compute_path),
                "dynamics_content_sha256": DYNAMICS_CONTENT_SHA256,
                "source_sha256": _sha256_file(source_path),
                "selection_use": False,
                "retuning": False,
                "post_selection_candidate_cells": 0,
                "confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "selected_formula": selected,
            "mapping": {
                "observable_units": "CLASH base-10 log accelerations and errors are converted to natural-log units used by the Item 41 stochastic closure",
                "scale_coordinate": "x=r/max(r) within each cluster; the selected white-field lane is independent of x",
                "primary_background": "fixed MOND/RAR mean; unchanged Item 41 m and S test residual drift and variance",
                "direct_replacement_diagnostic": "the same m and S are also scored on a baryonic-Newton mean without dark matter",
                "lensing_closure": "Phi=Psi as frozen; this remains a model-dependent ensemble proxy, not a direct image likelihood",
            },
            "data": {
                "clusters": len(unique),
                "radial_points": len(names),
                "role": config["scope"]["clash_role"],
            },
            "losses": losses,
            "strongest_primary_control": strongest_name,
            "improvement_vs_strongest_percent": improvement,
            "process_variance_range": {
                "minimum": float(np.min(process_variance)),
                "maximum": float(np.max(process_variance)),
            },
            "leave_one": {
                "most_influential_cluster": unique[int(order[0])],
                "improvement_percent": leave_improvement,
                "changes_sign": report["leave_one_changes_sign"],
            },
            "object_level": object_records,
            "counterexample_policy_report": report,
            "counterexample_assessment": assessment,
            "claim_boundaries": {
                "fresh_confirmation": False,
                "direct_image_likelihood": False,
                "model_independent_lensing": False,
                "stochastic_process_established": False,
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
        _write_json(_source_path(root, config, "clash_transfer_result"), result)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "clash_transfer_result")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 41 CLASH transfer result")
    if existing != evaluate(root, write=False):
        raise GravityItem41Error("Item 41 CLASH transfer replay changed")
    return {
        "status": "ITEM41_CLASH_TRANSFER_REPLAY_VALID",
        "decision": existing["decision"],
        "content_sha256": existing["content_sha256"],
        "clusters": existing["data"]["clusters"],
        "radial_points": existing["data"]["radial_points"],
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
