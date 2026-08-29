"""Item 58: predict CLASH beta_eff from measured baryonic geometry and state."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sigma_theory_compiler.gravity_counterexample_policy import (
    assess_counterexample_evidence,
    load_counterexample_policy,
)
from sigma_theory_compiler.gravity_g4_cluster_lensing_exploration import (
    G_DAGGER,
    VELOCITY2_TO_ACCELERATION,
)
from sigma_theory_compiler.gravity_g4_cluster_lensing_exploration import (
    load_config as load_cluster_config,
)
from sigma_theory_compiler.gravity_g4_cluster_lensing_exploration import (
    prepare_packets as prepare_cluster_packets,
)
from sigma_theory_compiler.gravity_g4_first_principles_mechanism_search import (
    _component_for_spec,
    _packet_context,
    mechanism_specs,
)
from sigma_theory_compiler.gravity_item1_effective_dimension import PARENT_ID
from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _canonical_bytes,
    _content_hashed,
    _read_json,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)

CONFIG_PATH = Path("configs/gravity_item58_cluster_coefficient_gate_v1.json")
GOAL_PATH = Path("docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md")
POLICY_PATH = Path("configs/gravity_empirical_counterexample_policy_v1.json")
ITEM57_PATH = Path("runs/gravity/roadmap/item-57-independent-galaxy-gate-v1.json")
ITEM1_PATH = Path("runs/gravity/roadmap/item-01-effective-dimension-v1.json")
ITEM2_PATH = Path("runs/gravity/roadmap/item-02-shape-anisotropy-v1.json")


class GravityItem58Error(RuntimeError):
    """Raised when the Item 58 freeze, lineage, or replay changes."""


def load_config(root: Path, *, require_bound: bool = True) -> dict[str, Any]:
    config = _read_json(root / CONFIG_PATH)
    validate_config(root, config, require_bound=require_bound)
    return config


def validate_config(
    root: Path, config: Mapping[str, Any], *, require_bound: bool = True
) -> None:
    if (
        config.get("schema_version")
        != "invariant-gravity-item58-cluster-coefficient-gate-config-1.0"
        or int(config.get("item", -1)) != 58
    ):
        raise GravityItem58Error("unexpected Item 58 config")
    if _sha256_file(root / GOAL_PATH) != config["stable_goal_sha256"]:
        raise GravityItem58Error("stable gravity goal changed")
    freeze = str(config["scientific_freeze_commit"])
    if require_bound and re.fullmatch(r"[0-9a-f]{40}", freeze) is None:
        raise GravityItem58Error("Item 58 scientific freeze is not commit-bound")
    if not require_bound and not (
        freeze == "PENDING_FREEZE_COMMIT" or re.fullmatch(r"[0-9a-f]{40}", freeze)
    ):
        raise GravityItem58Error("malformed Item 58 freeze binding")
    for relative, expected in config["scientific_dependencies"].items():
        if _sha256_file(root / str(relative)) != expected:
            raise GravityItem58Error(f"scientific dependency changed: {relative}")
    item57 = _read_json(root / ITEM57_PATH)
    predecessor = config["required_predecessor"]
    if (
        item57["decision"] != predecessor["decision"]
        or bool(item57["claims"]["formula_family_pruned"])
    ):
        raise GravityItem58Error("Item 57 predecessor changed")
    parent = config["diagnostic_parent"]
    grid = parent["coefficient_grid"]
    if (
        parent["candidate_id"] != PARENT_ID
        or float(parent["transition_acceleration_m_s2"]) != G_DAGGER
        or float(parent["log_radius_scale"]) != 0.25
        or (float(grid["minimum"]), float(grid["maximum"]), float(grid["step"]))
        != (0.0, 4.0, 0.01)
    ):
        raise GravityItem58Error("Item 58 diagnostic parent changed")
    population = config["population"]
    if (
        len(population["sample"]) != int(population["clusters"])
        or len(set(population["sample"])) != int(population["clusters"])
        or int(population["radial_points"]) != 84
        or int(population["new_response_rows_allowed"]) != 0
        or int(population["direct_lensing_likelihood_evaluations_allowed"]) != 0
        or int(population["sparc_confirmation_response_rows_allowed"]) != 0
    ):
        raise GravityItem58Error("Item 58 population or sealed boundary changed")
    lineage = config["feature_lineage"]
    forbidden = (
        "feature_builder_target_fields_available",
        "cluster_identifier_available_to_model",
        "gtot_or_lensing_mass_available_to_model",
        "inferred_total_mass_available_to_model",
        "temperature_or_hydrostatic_mass_available_to_model",
        "per_cluster_coefficient_available_to_model",
    )
    if any(bool(lineage[name]) for name in forbidden):
        raise GravityItem58Error("Item 58 permits target leakage or object identity")
    allowed = {
        feature
        for group in config["allowed_features"].values()
        for feature in group
    }
    models = config["model_families"]
    if len(models) != 5 or len({str(model["id"]) for model in models}) != 5:
        raise GravityItem58Error("Item 58 model family changed")
    if any(not set(map(str, model["features"])) <= allowed for model in models):
        raise GravityItem58Error("Item 58 model uses an unapproved feature")
    evaluation = config["evaluation"]
    if (
        int(evaluation["outer_folds"]) != 5
        or int(evaluation["inner_folds"]) != 4
        or int(evaluation["permutations"]) != 1999
        or [float(value) for value in evaluation["ridge_penalties"]]
        != [0.01, 0.1, 1.0, 10.0, 100.0]
    ):
        raise GravityItem58Error("Item 58 evaluation changed")
    counterexample = config["counterexample_policy"]
    if (
        counterexample["single_counterexample_terminal"]
        or counterexample["counterexample_count_alone_terminal"]
        or counterexample["finite_sample_may_prune_formula_family"]
        or not counterexample["model_dependent_target_prevents_gravity_confirmation"]
    ):
        raise GravityItem58Error("Item 58 permits empirical over-pruning or overclaim")


def _contract_digest(config: Mapping[str, Any]) -> str:
    value = json.loads(json.dumps(config))
    value["scientific_freeze_commit"] = "<BOUND_COMMIT>"
    return _sha256_bytes(_canonical_bytes(value))


def _source_path(root: Path, config: Mapping[str, Any], key: str) -> Path:
    return root / str(config["paths"]["source_dir"]) / str(config["paths"][key])


def _parent_spec() -> Mapping[str, Any]:
    matches = [row for row in mechanism_specs() if row["candidate_id"] == PARENT_ID]
    if len(matches) != 1:
        raise GravityItem58Error("diagnostic parent is unavailable")
    spec = matches[0]
    if (
        spec.get("source") != "baryonic_acceleration"
        or float(spec.get("threshold")) != 0.1
        or float(spec.get("log_radius_scale")) != 0.25
        or spec.get("mode") != "permittivity_plus_auxiliary"
    ):
        raise GravityItem58Error("diagnostic parent definition changed")
    return spec


def _grid(config: Mapping[str, Any]) -> np.ndarray:
    definition = config["diagnostic_parent"]["coefficient_grid"]
    lower = float(definition["minimum"])
    upper = float(definition["maximum"])
    step = float(definition["step"])
    return np.linspace(lower, upper, round((upper - lower) / step) + 1)


def _oracle_beta(
    base: np.ndarray,
    component: np.ndarray,
    observed_log: np.ndarray,
    sigma_log: np.ndarray,
    grid: np.ndarray,
) -> tuple[float, float, bool]:
    losses = []
    for beta in grid:
        prediction = base + float(beta) * component
        if np.any(prediction <= 0.0) or np.any(~np.isfinite(prediction)):
            loss = float("inf")
        else:
            residual = (np.log10(prediction) - observed_log) / sigma_log
            loss = float(np.sum(residual**2))
        losses.append(loss)
    index = min(range(len(grid)), key=lambda offset: (losses[offset], abs(float(grid[offset]))))
    return (
        float(grid[index]),
        float(losses[index]),
        bool(index == 0 or index == len(grid) - 1),
    )


def _profile_features(radius: np.ndarray, gbar: np.ndarray) -> dict[str, float]:
    log_radius = np.log10(radius)
    log_gbar = np.log10(gbar)
    log_mass = log_gbar + 2.0 * log_radius
    centered = log_radius - float(np.mean(log_radius))
    denominator = float(centered @ centered)
    if denominator <= 0.0:
        raise GravityItem58Error("degenerate cluster radius profile")
    profile_dimension = float(
        centered @ (log_mass - float(np.mean(log_mass))) / denominator
    )
    local_dimension = np.gradient(log_mass, log_radius)
    return {
        "profile_mass_dimension": profile_dimension,
        "local_dimension_median": float(np.median(local_dimension)),
        "local_dimension_iqr": float(
            np.quantile(local_dimension, 0.75) - np.quantile(local_dimension, 0.25)
        ),
        "median_log10_gbar": float(np.median(log_gbar)),
        "log10_gbar_span": float(np.max(log_gbar) - np.min(log_gbar)),
        "median_log10_radius_kpc": float(np.median(log_radius)),
        "log10_radius_span": float(np.max(log_radius) - np.min(log_radius)),
        "outer_inner_log10_equivalent_baryon_mass_ratio": float(log_mass[-1] - log_mass[0]),
    }


def prepare_records(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    item1 = _read_json(root / ITEM1_PATH)
    item2 = _read_json(root / ITEM2_PATH)
    item1_rows = {
        str(row["name"]): row
        for row in item1["per_object_diagnostics"]
        if row["domain"] == "cluster"
    }
    item2_rows = {
        str(row["name"]): row
        for row in item2["per_object_diagnostics"]
        if row["domain"] == "cluster"
    }
    packets = prepare_cluster_packets(root, load_cluster_config(root))
    packet_by_name = {str(packet["cluster"]): packet for packet in packets}
    expected = set(map(str, config["population"]["sample"]))
    if set(item1_rows) != expected or set(item2_rows) != expected or set(packet_by_name) != expected:
        raise GravityItem58Error("cluster identity sets do not match the frozen sample")
    spec = _parent_spec()
    grid = _grid(config)
    records = []
    for name in sorted(expected):
        packet = packet_by_name[name]
        radius = np.asarray(packet["arrays"]["radius"], dtype=float)
        base = np.asarray(packet["gbar"], dtype=float)
        component_v2 = np.asarray(
            _component_for_spec(spec, packet, _packet_context(packet)), dtype=float
        )
        component = component_v2 * VELOCITY2_TO_ACCELERATION / radius
        observed = np.asarray(packet["log_gtot"], dtype=float)
        sigma = np.asarray(packet["sigma_log_gtot"], dtype=float)
        beta, oracle_loss, boundary = _oracle_beta(base, component, observed, sigma, grid)
        beta_low, _, boundary_low = _oracle_beta(
            base, component, observed - sigma, sigma, grid
        )
        beta_high, _, boundary_high = _oracle_beta(
            base, component, observed + sigma, sigma, grid
        )
        frozen_beta = float(item1_rows[name]["oracle_beta_target_derived"])
        replay_delta = abs(beta - frozen_beta)
        if replay_delta > 5.0e-12 or boundary != bool(
            item1_rows[name]["oracle_beta_at_grid_boundary"]
        ):
            raise GravityItem58Error(f"Item 1 beta label failed exact replay: {name}")
        features = _profile_features(radius, base)
        for key in (
            "profile_mass_dimension",
            "local_dimension_median",
            "local_dimension_iqr",
        ):
            if not np.isclose(
                features[key],
                float(item1_rows[name]["features"][key]),
                rtol=0.0,
                atol=5.0e-12,
            ):
                raise GravityItem58Error(f"Item 1 profile feature failed replay: {name}:{key}")
        shape = item2_rows[name]["features"]
        for key in config["allowed_features"]["xray_morphology_state"]:
            features[str(key)] = float(shape[str(key)])
        features["concentration_x_profile_dimension"] = (
            features["projected_concentration_c20"] * features["profile_mass_dimension"]
        )
        features["ellipticity_x_gbar_span"] = (
            1.0 - features["projected_axis_ratio"]
        ) * features["log10_gbar_span"]
        features["boundary_mass_x_centroid_shift"] = (
            features["outer_inner_log10_equivalent_baryon_mass_ratio"]
            * features["cluster_log_centroid_shift"]
        )
        features["concentration_x_local_dimension_iqr"] = (
            features["projected_concentration_c20"] * features["local_dimension_iqr"]
        )
        allowed = {
            feature for group in config["allowed_features"].values() for feature in group
        }
        if set(features) != allowed or any(not np.isfinite(value) for value in features.values()):
            raise GravityItem58Error(f"cluster feature contract changed: {name}")
        records.append(
            {
                "name": name,
                "point_count": len(radius),
                "features": features,
                "beta": beta,
                "beta_log_gtot_minus_sigma": beta_low,
                "beta_log_gtot_plus_sigma": beta_high,
                "beta_at_grid_boundary": boundary,
                "beta_envelope_at_grid_boundary": boundary_low or boundary_high,
                "item1_beta_replay_delta": replay_delta,
                "oracle_chi_square": oracle_loss,
                "base": base,
                "component": component,
                "observed_log_gtot": observed,
                "sigma_log_gtot": sigma,
            }
        )
    if sum(record["point_count"] for record in records) != int(config["population"]["radial_points"]):
        raise GravityItem58Error("cluster radial point count changed")
    return records


def _feature_table_bytes(records: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> bytes:
    feature_order = [
        *config["allowed_features"]["baryonic_profile"],
        *config["allowed_features"]["xray_morphology_state"],
        *config["allowed_features"]["derived_synthesis"],
    ]
    header = ["cluster", "point_count", *feature_order]
    lines = ["\t".join(header)]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["name"]),
                    str(record["point_count"]),
                    *(format(float(record["features"][feature]), ".12e") for feature in feature_order),
                ]
            )
        )
    return ("\n".join(lines) + "\n").encode()


def build_preflight_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    records = prepare_records(root, config)
    feature_bytes = _feature_table_bytes(records, config)
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item58-preflight-1.0",
            "item": 58,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "config_contract_sha256": _contract_digest(config),
            "clusters": len(records),
            "radial_points": sum(int(record["point_count"]) for record in records),
            "feature_table_sha256": _sha256_bytes(feature_bytes),
            "model_families": config["model_families"],
            "evaluation": config["evaluation"],
            "feature_builder_target_fields": False,
            "new_response_rows": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "sparc_confirmation_response_rows": 0,
            "paid_model_calls": 0,
        }
    )


def write_preflight_and_features(root: Path) -> tuple[Path, Path]:
    config = load_config(root)
    records = prepare_records(root, config)
    preflight_path = _source_path(root, config, "preflight_manifest")
    feature_path = _source_path(root, config, "feature_table")
    _write_json(preflight_path, build_preflight_manifest(root))
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    feature_path.write_bytes(_feature_table_bytes(records, config))
    return preflight_path, feature_path


def _fold_map(names: Sequence[str], *, salt: str, folds: int) -> dict[str, int]:
    ordered = sorted(names, key=lambda name: hashlib.sha256(f"{salt}|{name}".encode()).hexdigest())
    return {name: index % folds for index, name in enumerate(ordered)}


def _matrix(records: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(records), 0), dtype=float)
    matrix = np.asarray(
        [[float(record["features"][feature]) for feature in features] for record in records],
        dtype=float,
    )
    if np.any(~np.isfinite(matrix)):
        raise GravityItem58Error("nonfinite cluster feature matrix")
    return matrix


def _fit_ridge(
    records: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    features: Sequence[str],
    penalty: float,
) -> dict[str, Any]:
    raw = _matrix(records, features)
    means = np.mean(raw, axis=0) if raw.shape[1] else np.empty(0)
    scales = np.std(raw, axis=0) if raw.shape[1] else np.empty(0)
    scales = np.where(scales > 1.0e-12, scales, 1.0)
    standardized = (raw - means) / scales if raw.shape[1] else raw
    design = np.column_stack([np.ones(len(records)), standardized])
    regularizer = np.eye(design.shape[1]) * penalty
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + regularizer, design.T @ target)
    return {
        "features": list(features),
        "penalty": penalty,
        "means": means,
        "scales": scales,
        "coefficients": coefficients,
    }


def _predict_ridge(
    fit: Mapping[str, Any], records: Sequence[Mapping[str, Any]], clip: Sequence[float]
) -> np.ndarray:
    features = list(map(str, fit["features"]))
    raw = _matrix(records, features)
    standardized = (
        (raw - np.asarray(fit["means"])) / np.asarray(fit["scales"])
        if features
        else raw
    )
    design = np.column_stack([np.ones(len(records)), standardized])
    prediction = design @ np.asarray(fit["coefficients"])
    return np.clip(prediction, float(clip[0]), float(clip[1]))


def _inner_select(
    training: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    model: Mapping[str, Any],
    config: Mapping[str, Any],
    outer_fold: int,
) -> tuple[float, float]:
    evaluation = config["evaluation"]
    names = [str(record["name"]) for record in training]
    folds = _fold_map(
        names,
        salt=f"{evaluation['inner_fold_salt']}|outer={outer_fold}",
        folds=int(evaluation["inner_folds"]),
    )
    scores = []
    for penalty in map(float, evaluation["ridge_penalties"]):
        squared = []
        for fold in range(int(evaluation["inner_folds"])):
            train_indices = [index for index, name in enumerate(names) if folds[name] != fold]
            test_indices = [index for index, name in enumerate(names) if folds[name] == fold]
            if not train_indices or not test_indices:
                raise GravityItem58Error("empty Item 58 inner fold")
            fit = _fit_ridge(
                [training[index] for index in train_indices],
                target[train_indices],
                model["features"],
                penalty,
            )
            prediction = _predict_ridge(
                fit,
                [training[index] for index in test_indices],
                evaluation["prediction_clip"],
            )
            squared.extend(np.square(prediction - target[test_indices]))
        scores.append((float(np.mean(squared)), penalty))
    return min(scores, key=lambda row: (row[0], row[1]))


def _cross_validate(
    records: Sequence[Mapping[str, Any]], target: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    names = [str(record["name"]) for record in records]
    outer = _fold_map(
        names,
        salt=str(evaluation["outer_fold_salt"]),
        folds=int(evaluation["outer_folds"]),
    )
    models = config["model_families"]
    model_predictions = {str(model["id"]): np.empty(len(records)) for model in models}
    nested_prediction = np.empty(len(records))
    fold_ledger = []
    for fold in range(int(evaluation["outer_folds"])):
        train_indices = [index for index, name in enumerate(names) if outer[name] != fold]
        test_indices = [index for index, name in enumerate(names) if outer[name] == fold]
        training = [records[index] for index in train_indices]
        heldout = [records[index] for index in test_indices]
        selected = []
        for model in models:
            inner_mse, penalty = _inner_select(
                training, target[train_indices], model, config, fold
            )
            fit = _fit_ridge(
                training, target[train_indices], model["features"], penalty
            )
            prediction = _predict_ridge(
                fit, heldout, evaluation["prediction_clip"]
            )
            model_predictions[str(model["id"])][test_indices] = prediction
            selected.append(
                {
                    "model_id": str(model["id"]),
                    "qualifying": bool(model["qualifying"]),
                    "origin_label": str(model["origin_label"]),
                    "features": list(model["features"]),
                    "inner_mse": inner_mse,
                    "penalty": penalty,
                    "prediction": prediction,
                }
            )
        winner = min(
            selected,
            key=lambda row: (
                float(row["inner_mse"]),
                len(row["features"]),
                str(row["model_id"]),
                float(row["penalty"]),
            ),
        )
        nested_prediction[test_indices] = np.asarray(winner["prediction"])
        fold_ledger.append(
            {
                "fold": fold,
                "training_clusters": [names[index] for index in train_indices],
                "heldout_clusters": [names[index] for index in test_indices],
                "selected_model_id": winner["model_id"],
                "selected_origin_label": winner["origin_label"],
                "selected_qualifying": winner["qualifying"],
                "selected_penalty": winner["penalty"],
                "selected_inner_mse": winner["inner_mse"],
            }
        )
    return {
        "nested_prediction": nested_prediction,
        "model_predictions": model_predictions,
        "fold_ledger": fold_ledger,
    }


def _metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    residual = prediction - target
    sse = float(residual @ residual)
    centered = target - float(np.mean(target))
    sst = float(centered @ centered)
    return {
        "mean_absolute_error": float(np.mean(np.abs(residual))),
        "mean_squared_error": float(np.mean(residual**2)),
        "r2": float(1.0 - sse / sst) if sst > 1.0e-15 else 0.0,
    }


def _observational_score(
    records: Sequence[Mapping[str, Any]], predictions: np.ndarray
) -> dict[str, Any]:
    chi_square = 0.0
    invalid = 0
    per_cluster = {}
    for record, beta in zip(records, predictions, strict=True):
        value = np.asarray(record["base"]) + float(beta) * np.asarray(record["component"])
        bad = int(np.sum(~np.isfinite(value) | (value <= 0.0)))
        safe = np.maximum(value, np.finfo(float).tiny)
        residual = (
            np.log10(safe) - np.asarray(record["observed_log_gtot"])
        ) / np.asarray(record["sigma_log_gtot"])
        score = float(np.sum(residual**2)) + bad * 1.0e24
        chi_square += score
        invalid += bad
        per_cluster[str(record["name"])] = score
    return {
        "chi_square": chi_square,
        "chi_square_per_point": chi_square
        / sum(int(record["point_count"]) for record in records),
        "invalid_predictions": invalid,
        "per_cluster_chi_square": per_cluster,
    }


def _influence(
    target: np.ndarray,
    selected: np.ndarray,
    constant: np.ndarray,
    trim_fraction: float,
) -> dict[str, Any]:
    advantage = np.square(constant - target) - np.square(selected - target)
    full = float(np.mean(advantage))
    leave_values = [float(np.mean(np.delete(advantage, index))) for index in range(len(advantage))]
    count = math.floor(len(advantage) * trim_fraction)
    ordered = np.sort(advantage)
    trimmed = ordered[count:-count] if count and 2 * count < len(advantage) else ordered
    trimmed_mean = float(np.mean(trimmed))
    return {
        "mean_mse_advantage_vs_constant": full,
        "leave_one_minimum_advantage": min(leave_values),
        "leave_one_changes_sign": any(full * value <= 0.0 for value in leave_values),
        "trimmed_each_tail": count,
        "trimmed_mean_advantage": trimmed_mean,
        "trim_changes_sign": full * trimmed_mean <= 0.0,
        "per_cluster_advantage": advantage,
    }


def _permutation_p(
    records: Sequence[Mapping[str, Any]],
    target: np.ndarray,
    observed_improvement: float,
    config: Mapping[str, Any],
) -> tuple[float, list[float]]:
    evaluation = config["evaluation"]
    rng = np.random.default_rng(int(evaluation["permutation_seed"]))
    improvements = []
    for _ in range(int(evaluation["permutations"])):
        shuffled = target[rng.permutation(len(target))]
        result = _cross_validate(records, shuffled, config)
        selected = np.asarray(result["nested_prediction"])
        constant = np.asarray(result["model_predictions"]["constant_training_mean"])
        baseline = float(np.mean(np.square(constant - shuffled)))
        candidate = float(np.mean(np.square(selected - shuffled)))
        improvements.append((baseline - candidate) / baseline if baseline > 0.0 else 0.0)
    extreme = 1 + sum(value >= observed_improvement for value in improvements)
    return extreme / (len(improvements) + 1), improvements


def build_evaluation_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    records = prepare_records(root, config)
    names = [str(record["name"]) for record in records]
    label_keys = {
        "nominal": "beta",
        "log_gtot_minus_sigma": "beta_log_gtot_minus_sigma",
        "log_gtot_plus_sigma": "beta_log_gtot_plus_sigma",
    }
    evaluations = {}
    for label, key in label_keys.items():
        target = np.asarray([float(record[key]) for record in records])
        cv = _cross_validate(records, target, config)
        nested = np.asarray(cv["nested_prediction"])
        constant = np.asarray(cv["model_predictions"]["constant_training_mean"])
        fixed_two = np.full(len(records), float(config["evaluation"]["fixed_beta_two_comparator"]))
        model_metrics = {
            model_id: _metrics(target, np.asarray(prediction))
            for model_id, prediction in cv["model_predictions"].items()
        }
        evaluations[label] = {
            "target": target,
            "nested_prediction": nested,
            "constant_prediction": constant,
            "fixed_two_prediction": fixed_two,
            "fold_ledger": cv["fold_ledger"],
            "metrics": {
                "nested_selector": _metrics(target, nested),
                "constant_training_mean": _metrics(target, constant),
                "fixed_beta_two": _metrics(target, fixed_two),
                "by_model_family": model_metrics,
            },
            "observational_scores": {
                "nested_selector": _observational_score(records, nested),
                "constant_training_mean": _observational_score(records, constant),
                "fixed_beta_two": _observational_score(records, fixed_two),
            },
            "influence": _influence(
                target,
                nested,
                constant,
                float(config["evaluation"]["influence_trim_fraction"]),
            ),
        }
    nominal = evaluations["nominal"]
    constant_mse = float(nominal["metrics"]["constant_training_mean"]["mean_squared_error"])
    nested_mse = float(nominal["metrics"]["nested_selector"]["mean_squared_error"])
    relative_improvement = (constant_mse - nested_mse) / constant_mse
    permutation_p, permutation_improvements = _permutation_p(
        records,
        np.asarray(nominal["target"]),
        relative_improvement,
        config,
    )
    threshold = config["admission"]
    envelope_improvements = {}
    for label in ("log_gtot_minus_sigma", "log_gtot_plus_sigma"):
        row = evaluations[label]
        baseline = float(row["metrics"]["constant_training_mean"]["mean_squared_error"])
        candidate = float(row["metrics"]["nested_selector"]["mean_squared_error"])
        envelope_improvements[label] = (baseline - candidate) / baseline
    qualifying_folds = sum(
        bool(row["selected_qualifying"]) for row in nominal["fold_ledger"]
    )
    gates = {
        "all_20_clusters_evaluable": len(records) == int(config["population"]["clusters"]),
        "no_label_at_grid_boundary": not any(
            bool(record["beta_at_grid_boundary"]) for record in records
        ),
        "nested_selector_qualifying_folds_minimum": qualifying_folds
        >= int(threshold["nested_selector_qualifying_folds_minimum"]),
        "relative_beta_mse_improvement_over_oof_constant_minimum": relative_improvement
        >= float(threshold["relative_beta_mse_improvement_over_oof_constant_minimum"]),
        "oof_beta_r2_minimum": float(nominal["metrics"]["nested_selector"]["r2"])
        > float(threshold["oof_beta_r2_minimum"]),
        "observational_chi_square_below_oof_constant": float(
            nominal["observational_scores"]["nested_selector"]["chi_square"]
        )
        < float(nominal["observational_scores"]["constant_training_mean"]["chi_square"]),
        "observational_chi_square_below_fixed_beta_two": float(
            nominal["observational_scores"]["nested_selector"]["chi_square"]
        )
        < float(nominal["observational_scores"]["fixed_beta_two"]["chi_square"]),
        "permutation_p_maximum": permutation_p <= float(threshold["permutation_p_maximum"]),
        "leave_one_and_trim_preserve_improvement": not bool(
            nominal["influence"]["leave_one_changes_sign"]
        )
        and not bool(nominal["influence"]["trim_changes_sign"]),
        "both_target_error_envelopes_preserve_positive_mse_improvement": all(
            value > 0.0 for value in envelope_improvements.values()
        ),
        "new_response_rows_zero": int(threshold["new_response_rows"]) == 0,
        "direct_lensing_likelihood_evaluations_zero": int(
            threshold["direct_lensing_likelihood_evaluations"]
        )
        == 0,
        "sparc_confirmation_response_rows_zero": int(
            threshold["sparc_confirmation_response_rows"]
        )
        == 0,
    }
    constant_abs = np.abs(
        np.asarray(nominal["constant_prediction"]) - np.asarray(nominal["target"])
    )
    nested_abs = np.abs(
        np.asarray(nominal["nested_prediction"]) - np.asarray(nominal["target"])
    )
    raw_counterexamples = [
        name for name, bad in zip(names, nested_abs > constant_abs, strict=True) if bad
    ]
    stable_counterexamples = []
    for index, name in enumerate(names):
        if name not in raw_counterexamples:
            continue
        stable = True
        for label in ("log_gtot_minus_sigma", "log_gtot_plus_sigma"):
            row = evaluations[label]
            target = np.asarray(row["target"])
            if abs(float(row["nested_prediction"][index]) - target[index]) <= abs(
                float(row["constant_prediction"][index]) - target[index]
            ):
                stable = False
        if stable:
            stable_counterexamples.append(name)
    policy_report = {
        "evidence_kind": "empirical",
        "evaluable_objects": len(records),
        "raw_counterexample_count": len(raw_counterexamples),
        "quality_verified_counterexample_count": len(raw_counterexamples),
        "uncertainty_resolved_counterexample_count": len(stable_counterexamples),
        "aggregate_improvement_percent": 100.0 * relative_improvement,
        "quality_gate_passed": False,
        "strongest_baseline_failed": not gates["observational_chi_square_below_fixed_beta_two"],
        "leave_one_changes_sign": bool(nominal["influence"]["leave_one_changes_sign"]),
        "trim_changes_sign": bool(nominal["influence"]["trim_changes_sign"]),
        "independent_failure_strata": 0,
        "unchanged_independent_replication_failures": 0,
        "object_level_records_preserved": True,
        "missing_quality_limited_records_preserved": True,
        "exclusions_frozen_before_response": True,
    }
    assessment = assess_counterexample_evidence(
        policy_report, load_counterexample_policy(root / POLICY_PATH)
    )
    rendered_evaluations = {}
    for label, row in evaluations.items():
        rendered_evaluations[label] = {
            "fold_ledger": row["fold_ledger"],
            "metrics": row["metrics"],
            "observational_scores": row["observational_scores"],
            "influence": {
                key: value
                for key, value in row["influence"].items()
                if key != "per_cluster_advantage"
            },
            "per_cluster": [
                {
                    "cluster": name,
                    "target_beta": float(row["target"][index]),
                    "nested_oof_beta": float(row["nested_prediction"][index]),
                    "constant_oof_beta": float(row["constant_prediction"][index]),
                    "fixed_beta_two": float(row["fixed_two_prediction"][index]),
                    "nested_mse_advantage_vs_constant": float(
                        row["influence"]["per_cluster_advantage"][index]
                    ),
                }
                for index, name in enumerate(names)
            ],
        }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item58-cluster-coefficient-evaluation-1.0",
            "item": 58,
            "scientific_freeze_commit": config["scientific_freeze_commit"],
            "diagnostic_parent": config["diagnostic_parent"],
            "evaluations": rendered_evaluations,
            "relative_beta_mse_improvement_over_oof_constant": relative_improvement,
            "target_error_envelope_relative_improvements": envelope_improvements,
            "permutation": {
                "count": len(permutation_improvements),
                "seed": int(config["evaluation"]["permutation_seed"]),
                "p_value": permutation_p,
                "null_improvement_mean": float(np.mean(permutation_improvements)),
                "null_improvement_95th_percentile": float(
                    np.quantile(permutation_improvements, 0.95)
                ),
            },
            "qualifying_selected_folds": qualifying_folds,
            "gates": gates,
            "gate_passed": all(gates.values()),
            "raw_counterexamples_vs_oof_constant": raw_counterexamples,
            "counterexamples_stable_across_target_error_envelopes": stable_counterexamples,
            "counterexample_policy_report": policy_report,
            "counterexample_policy_assessment": assessment,
            "counts": {
                "clusters": len(records),
                "radial_points": sum(int(record["point_count"]) for record in records),
                "model_families": len(config["model_families"]),
                "ridge_penalties": len(config["evaluation"]["ridge_penalties"]),
                "nominal_outer_model_fits": int(config["evaluation"]["outer_folds"])
                * len(config["model_families"]),
                "new_response_rows": 0,
                "direct_lensing_likelihood_evaluations": 0,
                "sparc_confirmation_response_rows": 0,
                "paid_model_calls": 0,
            },
            "compute": {
                "backend": "numpy_cpu",
                "gpu_used": False,
                "permutation_full_nested_replays": len(permutation_improvements),
                "paid_api_cost_usd": 0.0,
            },
            "claims": {
                "cluster_coefficient_development_gate_passed": all(gates.values()),
                "causal_variable_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "direct_clash_lensing_completed": False,
                "historical_novelty_established": False,
                "formula_or_feature_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": [
                "The beta_eff label is derived from an already exposed spherical NFW-lensing posterior acceleration diagnostic, not a direct image, shear, magnification, or time-delay likelihood.",
                "The sample has only 20 clusters and 84 radial points; target covariance is unavailable.",
                "X-ray surface brightness is a baryonic-state tracer but is sensitive to gas density squared, projection, disturbance, and reduction choices.",
                "The lower and upper one-sigma target envelopes are sensitivity brackets, not a full correlated nuisance likelihood.",
                "A pass would be development evidence for a measurable predictor, not proof that the variable causes modified gravity.",
            ],
        }
    )


def write_evaluation_result(root: Path) -> Path:
    config = load_config(root)
    path = _source_path(root, config, "evaluation_result")
    _write_json(path, build_evaluation_result(root))
    return path


def build_aggregate_result(root: Path) -> dict[str, Any]:
    config = load_config(root)
    preflight = _read_json(_source_path(root, config, "preflight_manifest"))
    evaluation = _read_json(_source_path(root, config, "evaluation_result"))
    passed = bool(evaluation["gate_passed"])
    bindings = {}
    for label, key in (
        ("preflight", "preflight_manifest"),
        ("feature_table", "feature_table"),
        ("evaluation", "evaluation_result"),
    ):
        path = _source_path(root, config, key)
        bindings[label] = {
            "path": str(path.relative_to(root)).replace("\\", "/"),
            "sha256": _sha256_file(path),
        }
    bindings["config"] = {"path": str(CONFIG_PATH), "sha256": _sha256_file(root / CONFIG_PATH)}
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item58-cluster-coefficient-result-1.0",
            "item": 58,
            "goal": "GRAVITY_ROADMAP_ITEM_58_CLUSTER_COEFFICIENT_GATE",
            "decision": (
                "PASS_ITEM58_CLUSTER_COEFFICIENT_DEVELOPMENT_GATE_NOT_DIRECT_LENSING"
                if passed
                else "ITEM58_CLUSTER_COEFFICIENT_GATE_NOT_PASSED_VARIABLE_FAMILIES_RETAINED"
            ),
            "gates": evaluation["gates"],
            "relative_beta_mse_improvement_over_oof_constant": evaluation[
                "relative_beta_mse_improvement_over_oof_constant"
            ],
            "target_error_envelope_relative_improvements": evaluation[
                "target_error_envelope_relative_improvements"
            ],
            "permutation": evaluation["permutation"],
            "qualifying_selected_folds": evaluation["qualifying_selected_folds"],
            "nominal_evaluation": evaluation["evaluations"]["nominal"],
            "counterexample_policy_assessment": evaluation[
                "counterexample_policy_assessment"
            ],
            "source_bindings": bindings,
            "counts": evaluation["counts"],
            "compute": evaluation["compute"],
            "claims": {
                "roadmap_item_58_execution_complete": True,
                "cluster_coefficient_development_gate_passed": passed,
                "causal_variable_established": False,
                "alternative_to_gr_established": False,
                "dark_matter_eliminated": False,
                "direct_clash_lensing_completed": False,
                "historical_novelty_established": False,
                "formula_or_feature_family_pruned": False,
                "single_counterexample_used_as_veto": False,
            },
            "limitations": evaluation["limitations"],
            "next_action": (
                "Advance to Item 59 X-COP forward-observable testing. Preserve all measured geometry/state families and cluster-level failures; do not treat this model-dependent beta diagnostic as direct lensing or causal proof."
            ),
            "preflight": preflight,
        }
    )


def write_aggregate_result(root: Path) -> Path:
    config = load_config(root)
    path = root / str(config["paths"]["aggregate_result"])
    _write_json(path, build_aggregate_result(root))
    return path


def replay(root: Path) -> dict[str, Any]:
    config = load_config(root)
    records = prepare_records(root, config)
    checks = {
        "preflight": _read_json(_source_path(root, config, "preflight_manifest"))
        == build_preflight_manifest(root),
        "feature_table": _source_path(root, config, "feature_table").read_bytes()
        == _feature_table_bytes(records, config),
        "evaluation": _read_json(_source_path(root, config, "evaluation_result"))
        == build_evaluation_result(root),
        "aggregate": _read_json(root / str(config["paths"]["aggregate_result"]))
        == build_aggregate_result(root),
    }
    return {"ok": all(checks.values()), "checks": checks}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "evaluate", "aggregate", "replay"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "preflight":
        paths = write_preflight_and_features(root)
        result: Any = [str(path) for path in paths]
    elif args.command == "evaluate":
        result = str(write_evaluation_result(root))
    elif args.command == "aggregate":
        result = str(write_aggregate_result(root))
    else:
        result = replay(root)
        if not result["ok"]:
            print(json.dumps(result, sort_keys=True))
            return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
