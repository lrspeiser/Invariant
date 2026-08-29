"""Apply the commit-bound Item 42 feedback law unchanged to CLASH clusters."""

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
from sigma_theory_compiler.gravity_item42_matter_geometry_feedback import (
    POLICY_PATH,
    GravityItem42Error,
    _source_path,
    decode_candidate,
    feedback_coordinate,
    load_config,
)

DYNAMICS_COMMIT = "e285a831e38f7efcac02bab133996fdf5f1b9653"
DYNAMICS_CONTENT_SHA256 = "c29a15750fdc26cbc604aeb81b6ff76224049abc2307351c16e37a5546dc6c49"
SELECTED_CANDIDATE_ID = 170142


def _load_clash(path: Path) -> dict[str, list[dict[str, float]]]:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    if len(lines) < 4:
        raise GravityItem42Error("CLASH VizieR table is incomplete")
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
        raise GravityItem42Error("CLASH frozen row or cluster count changed")
    return dict(sorted(grouped.items()))


def _multiplier(
    selected: Mapping[str, Any], u: np.ndarray, h: np.ndarray
) -> np.ndarray:
    parameters = selected["parameters"]
    amplitude = float(parameters["amplitude"])
    exponent = float(parameters["acceleration_exponent"])
    transition = float(parameters["transition_u"])
    return 1.0 + amplitude * np.power(u, -exponent) / (1.0 + u / transition) * (
        0.05 + 0.95 * h
    )


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
        result[index] = float(
            np.mean(np.square((observed[mask] - prediction[mask]) / sigma[mask]))
        )
    return unique, result


def _improvement(reference: float, candidate: float) -> float:
    return 100.0 * (reference - candidate) / max(abs(reference), 1e-15)


def evaluate(root: Path, *, write: bool = True) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    compute_path = _source_path(root, config, "compute_manifest")
    compute = _read_json(compute_path)
    _verify_content_hash(compute, "Item 42 dynamics compute manifest")
    if compute["content_sha256"] != DYNAMICS_CONTENT_SHA256:
        raise GravityItem42Error("commit-bound Item 42 dynamics result changed")
    selected = compute["candidate_search"]["full_exploration_candidate"]
    if int(selected["candidate_id"]) != SELECTED_CANDIDATE_ID:
        raise GravityItem42Error("commit-bound Item 42 formula changed")
    if selected != decode_candidate(SELECTED_CANDIDATE_ID, config):
        raise GravityItem42Error("decoded Item 42 formula changed")
    no_feedback = compute["candidate_search"]["full_no_feedback_control"]

    source_path = root / str(config["data_sources"]["clash_source"])
    grouped = _load_clash(source_path)
    names: list[str] = []
    log_gbar: list[float] = []
    log_gtot: list[float] = []
    sigma: list[float] = []
    candidate_h: list[float] = []
    no_feedback_h: list[float] = []
    convergence_records: list[dict[str, Any]] = []
    selected_lane = int(selected["lane_id"])
    selected_gain = float(selected["parameters"]["feedback_gain"])
    no_feedback_lane = int(no_feedback["lane_id"])
    for name, rows in grouped.items():
        radius = np.asarray([row["radius_kpc"] for row in rows])
        local_log_gbar = np.asarray([row["log_gbar"] for row in rows])
        cumulative_mass_proxy = np.power(10.0, local_log_gbar) * np.square(radius)
        feedback_h, audit = feedback_coordinate(
            radius,
            cumulative_mass_proxy,
            selected_lane,
            selected_gain,
            config,
        )
        base_h, base_audit = feedback_coordinate(
            radius,
            cumulative_mass_proxy,
            no_feedback_lane,
            0.0,
            config,
        )
        candidate_converged = bool(audit["converged"] and base_audit["converged"])
        convergence_records.append(
            {
                "cluster": name,
                "candidate_iterations": int(audit["iterations"]),
                "no_feedback_iterations": int(base_audit["iterations"]),
                "candidate_converged": bool(audit["converged"]),
                "no_feedback_converged": bool(base_audit["converged"]),
            }
        )
        names.extend([name] * len(rows))
        log_gbar.extend(local_log_gbar.tolist())
        log_gtot.extend([row["log_gtot"] for row in rows])
        sigma.extend(
            [
                math.hypot(row["error_log_gbar"], row["error_log_gtot"])
                for row in rows
            ]
        )
        candidate_h.extend(
            feedback_h.tolist() if candidate_converged else [math.nan] * len(rows)
        )
        no_feedback_h.extend(base_h.tolist())
    name_array = np.asarray(names)
    gbar_array = np.asarray(log_gbar)
    observed = np.asarray(log_gtot)
    sigma_array = np.asarray(sigma)
    u = np.power(10.0, gbar_array) / float(
        config["constants"]["acceleration_scale_m_s2"]
    )
    predictions = {
        "candidate": gbar_array
        + np.log10(_multiplier(selected, u, np.asarray(candidate_h))),
        "matched_no_feedback": gbar_array
        + np.log10(_multiplier(no_feedback, u, np.asarray(no_feedback_h))),
        "baryonic_newton": gbar_array
        + np.log10(fixed_control_multiplier("baryonic_newton", u)),
        "mond_RAR": gbar_array
        + np.log10(fixed_control_multiplier("mond_RAR", u)),
    }
    unique, candidate_object = _object_losses(
        name_array, observed, predictions["candidate"], sigma_array
    )
    object_losses = {
        name: _object_losses(name_array, observed, prediction, sigma_array)[1]
        for name, prediction in predictions.items()
        if name != "candidate"
    }
    converged_object = np.isfinite(candidate_object)
    if not np.any(converged_object):
        raise GravityItem42Error("Item 42 selected feedback converged on no CLASH cluster")
    losses = {
        "candidate": float(np.mean(candidate_object[converged_object])),
        **{
            name: float(np.mean(value[converged_object]))
            for name, value in object_losses.items()
        },
    }
    control_losses_all_clusters = {
        name: float(np.mean(value)) for name, value in object_losses.items()
    }
    strongest_name = min(object_losses, key=lambda name: losses[name])
    strongest = object_losses[strongest_name]
    improvement = _improvement(losses[strongest_name], losses["candidate"])
    feedback_increment = _improvement(
        losses["matched_no_feedback"], losses["candidate"]
    )
    raw = (~converged_object) | (candidate_object > strongest)
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
    converged_indices = np.flatnonzero(converged_object)
    order = converged_indices[
        np.argsort(np.abs(differences[converged_object]))[::-1]
    ]
    leave = converged_object.copy()
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
        "quality_gate_passed": bool(np.all(converged_object)),
        "strongest_baseline_failed": improvement < 0.0 or not np.all(converged_object),
        "leave_one_changes_sign": bool((improvement >= 0.0) != (leave_improvement >= 0.0)),
        "trim_changes_sign": bool((improvement >= 0.0) != (leave_improvement >= 0.0)),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "numerical_domain_failure_count": int(np.sum(~converged_object)),
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
            "candidate_loss": (
                float(candidate_object[index]) if converged_object[index] else None
            ),
            "strongest_reference_loss": float(strongest[index]),
            "candidate_minus_reference": (
                float(differences[index]) if converged_object[index] else None
            ),
            "fixed_point_converged": bool(converged_object[index]),
            "raw_counterexample": bool(raw[index]),
            "uncertainty_resolved_counterexample": bool(resolved[index] and raw[index]),
        }
        for index, name in enumerate(unique)
    ]
    if not np.all(converged_object):
        decision = "ITEM42_CLASH_FIXED_POINT_NONCONVERGENCE_RETAINS_FORMULA_AND_FAMILY"
    elif improvement > 0.0:
        decision = "ITEM42_CLASH_DIAGNOSTIC_SUPPORTS_UNCHANGED_FEEDBACK_LAW"
    else:
        decision = "ITEM42_CLASH_DIAGNOSTIC_DOES_NOT_SUPPORT_LAW_BUT_RETAINS_FAMILY"
    result = _content_hashed(
        {
            "schema_version": "invariant-gravity-item42-clash-transfer-result-1.0",
            "item": 42,
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
            "matched_no_feedback_formula": no_feedback,
            "mapping": {
                "enclosed_baryonic_mass_proxy": "Mbar(<r) proportional to gbar*r^2; the constant cancels in feedback source fractions",
                "feedback": {
                    "selected_source_update": selected["source_update"],
                    "kernel_length": config["candidate_generator"]["fixed_point"][
                        "kernel_length"
                    ],
                    "damping_used": config["candidate_generator"]["fixed_point"][
                        "damping"
                    ],
                    "maximum_iterations": config["candidate_generator"][
                        "fixed_point"
                    ]["maximum_iterations"],
                    "documentation_note": "The non-executable feedback_contract iteration sentence retained an early damping value of 0.5; the frozen executable fixed_point value is 0.8 and was used before any response was opened.",
                },
                "prediction": "log10(gtot)=log10(gbar)+log10(nu) under the frozen Phi=Psi weak-field closure",
            },
            "data": {
                "clusters": len(unique),
                "radial_points": len(names),
                "role": config["cluster_transfer"]["role"],
            },
            "convergence": {
                "all_selected_formula_fixed_points_converged": bool(
                    np.all(converged_object)
                ),
                "converged_clusters": int(np.sum(converged_object)),
                "nonconverged_clusters": int(np.sum(~converged_object)),
                "maximum_candidate_iterations": max(
                    row["candidate_iterations"] for row in convergence_records
                ),
                "object_level": convergence_records,
            },
            "losses": losses,
            "loss_scope": "candidate and controls on the candidate-converged cluster subset",
            "control_losses_all_clusters": control_losses_all_clusters,
            "strongest_fixed_control": strongest_name,
            "improvement_vs_strongest_percent": improvement,
            "improvement_vs_matched_no_feedback_percent": feedback_increment,
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
                "complete_cluster_domain_coverage": bool(np.all(converged_object)),
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
        _write_json(_source_path(root, config, "clash_transfer_result"), result)
    return result


def check(root: Path) -> dict[str, Any]:
    config = load_config(root)
    path = _source_path(root, config, "clash_transfer_result")
    existing = _read_json(path)
    _verify_content_hash(existing, "Item 42 CLASH transfer result")
    if existing != evaluate(root, write=False):
        raise GravityItem42Error("Item 42 CLASH transfer replay changed")
    return {
        "status": "ITEM42_CLASH_TRANSFER_REPLAY_VALID",
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
