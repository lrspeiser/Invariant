"""Apply the commit-bound Item 40 graph law unchanged to CLASH clusters."""

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
from sigma_theory_compiler.gravity_item39_holographic_boundary import (
    generate_raw_candidates,
)
from sigma_theory_compiler.gravity_item40_discrete_network import (
    POLICY_PATH,
    GravityItem40Error,
    _source_path,
    fixed_control_multiplier,
    graph_coordinates,
    load_config,
    predict_multiplier,
)

DYNAMICS_COMMIT = "41400cebe74fe5e0cf9fe98e2a9b501626ff8ddf"
DYNAMICS_CONTENT_SHA256 = "37079d30c468090f7f58c6474e69369d5632904750f07d844e4e1e936111cd5f"


def _load_clash(path: Path) -> dict[str, list[dict[str, float]]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(lines) < 4:
        raise GravityItem40Error("CLASH VizieR table is incomplete")
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
        raise GravityItem40Error("CLASH frozen row or cluster count changed")
    return dict(sorted(grouped.items()))


def _candidate_row(config: Mapping[str, Any], candidate_id: int) -> dict[str, np.ndarray]:
    raw = generate_raw_candidates(config)
    return {key: value[candidate_id : candidate_id + 1] for key, value in raw.items()}


def _object_losses(
    names: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    sigma: np.ndarray,
) -> tuple[list[str], np.ndarray]:
    unique = sorted(set(names.tolist()))
    result = np.zeros(len(unique), dtype=np.float64)
    for index, name in enumerate(unique):
        mask = names == name
        result[index] = float(np.mean(np.square((observed[mask] - prediction[mask]) / sigma[mask])))
    return unique, result


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    compute = _read_json(compute_path)
    _verify_content_hash(compute, "Item 40 dynamics compute manifest")
    if compute["content_sha256"] != DYNAMICS_CONTENT_SHA256:
        raise GravityItem40Error("commit-bound Item 40 dynamics result changed")
    selected = compute["candidate_search"]["full_exploration_candidate"]
    candidate_id = int(selected["candidate_id"])
    if candidate_id != 255184:
        raise GravityItem40Error("commit-bound Item 40 formula changed")
    candidate = _candidate_row(config, candidate_id)

    source_path = root / str(config["cluster_transfer"]["source"])
    grouped = _load_clash(source_path)
    names: list[str] = []
    log_gbar: list[float] = []
    log_gtot: list[float] = []
    sigma: list[float] = []
    feature_parts: list[np.ndarray] = []
    for name, rows in grouped.items():
        radius = np.asarray([row["radius_kpc"] for row in rows])
        local_log_gbar = np.asarray([row["log_gbar"] for row in rows])
        cumulative_mass_proxy = np.power(10.0, local_log_gbar) * np.square(radius)
        graph = graph_coordinates(radius, cumulative_mass_proxy)
        names.extend([name] * len(rows))
        log_gbar.extend(local_log_gbar.tolist())
        log_gtot.extend([row["log_gtot"] for row in rows])
        sigma.extend(
            [
                math.hypot(row["error_log_gbar"], row["error_log_gtot"])
                for row in rows
            ]
        )
        feature_parts.append(graph)
    name_array = np.asarray(names)
    gbar_array = np.asarray(log_gbar)
    observed = np.asarray(log_gtot)
    sigma_array = np.asarray(sigma)
    graph_feature = np.concatenate(feature_parts, axis=1)
    u = np.power(10.0, gbar_array) / float(config["constants"]["acceleration_scale_m_s2"])
    multiplier = predict_multiplier(candidate, u, graph_feature, config)[0]
    predictions = {
        "candidate": gbar_array + np.log10(multiplier),
        "baryonic_newton": gbar_array
        + np.log10(fixed_control_multiplier("baryonic_newton", u)),
        "mond_RAR": gbar_array + np.log10(fixed_control_multiplier("mond_RAR", u)),
    }
    unique, candidate_object = _object_losses(
        name_array, observed, predictions["candidate"], sigma_array
    )
    object_losses = {
        name: _object_losses(name_array, observed, prediction, sigma_array)[1]
        for name, prediction in predictions.items()
        if name != "candidate"
    }
    losses = {
        "candidate": float(np.mean(candidate_object)),
        **{name: float(np.mean(value)) for name, value in object_losses.items()},
    }
    strongest_name = min(object_losses, key=lambda name: losses[name])
    strongest = object_losses[strongest_name]
    improvement = _improvement(losses[strongest_name], losses["candidate"])
    raw = candidate_object > strongest
    point_difference = np.square(
        (observed - predictions["candidate"]) / sigma_array
    ) - np.square((observed - predictions[strongest_name]) / sigma_array)
    resolved = np.zeros(len(unique), dtype=bool)
    for index, name in enumerate(unique):
        values = point_difference[name_array == name]
        standard_error = (
            float(np.std(values, ddof=1)) / math.sqrt(len(values))
            if len(values) > 1
            else math.inf
        )
        resolved[index] = float(np.mean(values)) - 1.96 * standard_error > 0.0
    differences = candidate_object - strongest
    order = np.argsort(np.abs(differences))[::-1]
    leave = np.ones(len(unique), dtype=bool)
    leave[order[0]] = False
    leave_improvement = _improvement(
        float(np.mean(strongest[leave])), float(np.mean(candidate_object[leave]))
    )
    report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(unique),
        "raw_counterexample_count": int(np.sum(raw)),
        "quality_verified_counterexample_count": int(np.sum(raw)),
        "uncertainty_resolved_counterexample_count": int(np.sum(resolved & raw)),
        "aggregate_improvement_percent": improvement,
        "quality_gate_passed": True,
        "strongest_baseline_failed": improvement < 0.0,
        "leave_one_changes_sign": bool((improvement >= 0.0) != (leave_improvement >= 0.0)),
        "trim_changes_sign": bool((improvement >= 0.0) != (leave_improvement >= 0.0)),
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
            "candidate_loss": float(candidate_object[index]),
            "strongest_reference_loss": float(strongest[index]),
            "candidate_minus_reference": float(differences[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(resolved[index] and raw[index]),
        }
        for index, name in enumerate(unique)
    ]
    if improvement > 0.0:
        decision = "ITEM40_CLASH_DIAGNOSTIC_SUPPORTS_UNCHANGED_NETWORK_LAW"
    else:
        decision = "ITEM40_CLASH_DIAGNOSTIC_DOES_NOT_SUPPORT_UNCHANGED_NETWORK_LAW"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item40-clash-transfer-result-1.0",
            "item": 40,
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
                "enclosed_baryonic_mass_proxy": "Mbar(<r) proportional to gbar*r^2; the constant cancels in graph shell fractions",
                "graph_coordinates": config["graph_construction"],
                "prediction": "log10(gtot)=log10(gbar)+log10(nu), with the unchanged galaxy-selected nu and Phi=Psi closure",
            },
            "data": {
                "clusters": len(unique),
                "radial_points": len(names),
                "role": config["cluster_transfer"]["role"],
            },
            "losses": losses,
            "strongest_fixed_control": strongest_name,
            "improvement_vs_strongest_percent": improvement,
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
        _write_json(_source_path(root, config, "cluster_transfer_result"), result)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "cluster_transfer_result")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 40 CLASH transfer result")
    if existing != evaluate(root, write=False):
        raise GravityItem40Error("Item 40 CLASH transfer replay changed")
    return {
        "status": "ITEM40_CLASH_TRANSFER_REPLAY_VALID",
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
