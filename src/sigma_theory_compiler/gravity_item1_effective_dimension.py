"""Test roadmap Item 1: effective dimension on real galaxies and clusters."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g4_cluster_lensing_exploration import (
    G_DAGGER,
    VELOCITY2_TO_ACCELERATION,
    _galaxy_rar,
)
from .gravity_g4_cluster_lensing_exploration import (
    load_config as load_cluster_config,
)
from .gravity_g4_cluster_lensing_exploration import (
    prepare_packets as prepare_cluster_packets,
)
from .gravity_g4_first_principles_mechanism_search import (
    _component_for_spec,
    _packet_context,
    mechanism_specs,
)
from .gravity_g4_photometric_law_construction import prepare_photometric_packets
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-roadmap-item1-effective-dimension-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-roadmap-item1-effective-dimension-config-1.0"
CONFIG_PATH = "configs/gravity_item1_effective_dimension.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item1_effective_dimension.py"
TEST_PATH = "tests/test_gravity_item1_effective_dimension.py"
OUTPUT_PATH = "runs/gravity/roadmap/item-01-effective-dimension-v1.json"
PARENT_ID = "cross-scale:y:q0p1:ell0p25:permittivity_plus_auxiliary"
DOMAINS = ("galaxy", "cluster")


class GravityItem1EffectiveDimensionError(ValueError):
    """The Item 1 contract, inputs, or receipt are inconsistent."""


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the frozen Item 1 contract and verify its evidence lineage."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityItem1EffectiveDimensionError("Item 1 config schema changed")
    if config.get("status") != "exploratory_real_data_model_development":
        raise GravityItem1EffectiveDimensionError("Item 1 status changed")
    roadmap = config.get("roadmap_binding", {})
    if (
        int(roadmap.get("item_number", 0)) != 1
        or roadmap.get("item_title") != "Effective dimension"
        or _file_sha256(root / str(roadmap.get("path"))) != roadmap.get("file_sha256")
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 roadmap binding changed")

    predecessor_paths = {
        "runs/gravity/g4/auxiliary-action-derivation-v6.json",
        "runs/gravity/g4/cluster-lensing-exploration-v7.json",
    }
    for binding in config.get("predecessor_bindings", ()):
        path_text = str(binding.get("path"))
        path = root / path_text
        if _file_sha256(path) != binding.get("file_sha256"):
            raise GravityItem1EffectiveDimensionError("Item 1 predecessor file changed")
        receipt = _load_json(path)
        if path_text not in predecessor_paths:
            raise GravityItem1EffectiveDimensionError("unknown Item 1 predecessor")
        predecessor_body = dict(receipt)
        predecessor_seal = predecessor_body.pop("content_sha256", None)
        if predecessor_seal != canonical_sha256(predecessor_body):
            raise GravityItem1EffectiveDimensionError("Item 1 predecessor seal changed")
        if receipt.get("content_sha256") != binding.get("content_sha256") or receipt.get(
            "decision"
        ) != binding.get("required_decision"):
            raise GravityItem1EffectiveDimensionError("Item 1 predecessor content changed")

    authorization = config.get("authorization", {})
    if (
        authorization.get("paid_model_calls_allowed") is not False
        or int(authorization.get("sparc_confirmation_evaluator_accesses_allowed", -1)) != 0
        or int(authorization.get("direct_lensing_likelihood_evaluations_allowed", -1)) != 0
        or authorization.get("sequential_G6_G7_G8_advanced") is not False
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 authorization boundary changed")
    parent = config.get("unchanged_parent", {})
    if (
        parent.get("candidate_id") != PARENT_ID
        or float(parent.get("transition_acceleration_m_s2")) != G_DAGGER
        or float(parent.get("log_radius_scale")) != 0.25
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 parent changed")
    grid = config.get("coefficient_label_grid", {})
    if (float(grid.get("minimum")), float(grid.get("maximum")), float(grid.get("step"))) != (
        0.0,
        4.0,
        0.01,
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 coefficient grid changed")
    fixed_ids = [str(row.get("id")) for row in config.get("fixed_rules", ())]
    if fixed_ids != [
        "fixed_galaxy_parent",
        "inverse_support_dimension",
        "inverse_profile_dimension",
        "posthoc_support_bridge",
    ]:
        raise GravityItem1EffectiveDimensionError("Item 1 fixed rules changed")
    model_ids = [str(row.get("id")) for row in config.get("trainable_models", ())]
    if model_ids != [
        "constant",
        "linear_profile_dimension",
        "linear_local_dimension",
        "linear_profile_dimension_plus_iqr",
        "quadratic_profile_dimension",
        "linear_support_dimension_proxy",
    ]:
        raise GravityItem1EffectiveDimensionError("Item 1 model grammar changed")
    if int(config.get("cross_validation", {}).get("outer_folds", 0)) != 5:
        raise GravityItem1EffectiveDimensionError("Item 1 fold count changed")
    if config.get("claim_boundaries", {}).get("alternative_to_gr_established") is not False:
        raise GravityItem1EffectiveDimensionError("Item 1 config overstates its claim")
    return config


def dimension_features(
    radius: np.ndarray, log_gbar: np.ndarray, support_dimension: float
) -> dict[str, float]:
    """Build dimension summaries from radius and baryons, never the dynamics target."""

    radius = np.asarray(radius, dtype=np.float64)
    log_gbar = np.asarray(log_gbar, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.shape != log_gbar.shape
        or len(radius) < 3
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(log_gbar))
        or np.any(radius <= 0)
        or np.any(np.diff(radius) <= 0)
    ):
        raise GravityItem1EffectiveDimensionError("invalid dimension profile")
    log_radius = np.log(radius)
    log_mass_equivalent = log_gbar + 2.0 * log_radius
    centered_radius = log_radius - float(np.mean(log_radius))
    denominator = float(centered_radius @ centered_radius)
    if denominator <= 0:
        raise GravityItem1EffectiveDimensionError("degenerate dimension profile")
    profile_dimension = float(
        centered_radius @ (log_mass_equivalent - float(np.mean(log_mass_equivalent))) / denominator
    )
    local_dimension = np.gradient(log_mass_equivalent, log_radius)
    values = {
        "local_dimension_iqr": float(
            np.quantile(local_dimension, 0.75) - np.quantile(local_dimension, 0.25)
        ),
        "local_dimension_median": float(np.median(local_dimension)),
        "profile_mass_dimension": profile_dimension,
        "profile_mass_dimension_squared": profile_dimension**2,
        "support_dimension": float(support_dimension),
    }
    if any(not np.isfinite(value) for value in values.values()):
        raise GravityItem1EffectiveDimensionError("non-finite dimension feature")
    return values


def _parent_spec() -> Mapping[str, Any]:
    matches = [row for row in mechanism_specs() if row["candidate_id"] == PARENT_ID]
    if len(matches) != 1:
        raise GravityItem1EffectiveDimensionError("cross-scale parent is unavailable")
    spec = matches[0]
    if (
        spec.get("source") != "baryonic_acceleration"
        or float(spec.get("threshold")) != 0.1
        or float(spec.get("log_radius_scale")) != 0.25
        or spec.get("mode") != "permittivity_plus_auxiliary"
    ):
        raise GravityItem1EffectiveDimensionError("cross-scale parent definition changed")
    return spec


def prepare_objects(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Assemble target-blind features and the unchanged parent on both populations."""

    root = root.resolve()
    spec = _parent_spec()
    objects: list[dict[str, Any]] = []
    galaxy_packets = sorted(prepare_photometric_packets(root), key=lambda row: row["galaxy"].name)
    for packet in galaxy_packets:
        radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
        component = np.asarray(
            _component_for_spec(spec, packet, _packet_context(packet)), dtype=np.float64
        )
        features = dimension_features(
            radius,
            np.asarray(packet["features"]["log_y"], dtype=np.float64),
            2.0,
        )
        objects.append(
            {
                "base": np.asarray(packet["arrays"]["vbar2"], dtype=np.float64),
                "component": component,
                "domain": "galaxy",
                "features": features,
                "key": f"galaxy:{packet['galaxy'].name}",
                "known_prediction": np.sqrt(np.asarray(packet["rar2"], dtype=np.float64)),
                "name": packet["galaxy"].name,
                "observed": np.asarray(packet["arrays"]["vobs"], dtype=np.float64),
                "point_count": int(packet["galaxy"].count),
                "sigma": np.asarray(packet["arrays"]["sigma"], dtype=np.float64),
                "target_scale": "linear_velocity",
            }
        )

    cluster_config = load_cluster_config(root)
    cluster_packets = prepare_cluster_packets(root, cluster_config)
    for packet in cluster_packets:
        radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
        component_v2 = np.asarray(
            _component_for_spec(spec, packet, _packet_context(packet)), dtype=np.float64
        )
        component = component_v2 * VELOCITY2_TO_ACCELERATION / radius
        gbar = np.asarray(packet["gbar"], dtype=np.float64)
        features = dimension_features(radius, np.log(gbar / G_DAGGER), 3.0)
        objects.append(
            {
                "base": gbar,
                "component": component,
                "domain": "cluster",
                "features": features,
                "key": f"cluster:{packet['cluster']}",
                "known_prediction": np.log10(_galaxy_rar(gbar)),
                "name": packet["cluster"],
                "observed": np.asarray(packet["log_gtot"], dtype=np.float64),
                "point_count": len(gbar),
                "sigma": np.asarray(packet["sigma_log_gtot"], dtype=np.float64),
                "target_scale": "log10_acceleration",
            }
        )

    population = config["population"]
    counts = {
        "galaxy": (
            len(galaxy_packets),
            sum(row["point_count"] for row in objects if row["domain"] == "galaxy"),
        ),
        "cluster": (
            len(cluster_packets),
            sum(row["point_count"] for row in objects if row["domain"] == "cluster"),
        ),
    }
    if counts["galaxy"] != (
        int(population["sparc_exploration_galaxies"]),
        int(population["sparc_exploration_points"]),
    ) or counts["cluster"] != (
        int(population["clash_clusters"]),
        int(population["clash_radial_points"]),
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 population changed")
    if len({row["key"] for row in objects}) != len(objects):
        raise GravityItem1EffectiveDimensionError("duplicate Item 1 object")
    return objects


def _fold_assignments(
    objects: Sequence[Mapping[str, Any]], *, salt: str, folds: int
) -> dict[str, int]:
    assignments: dict[str, int] = {}
    for domain in DOMAINS:
        members = [row for row in objects if row["domain"] == domain]
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(f"{salt}|{domain}|{row['name']}".encode()).hexdigest(),
        )
        for ordinal, row in enumerate(ordered):
            assignments[str(row["key"])] = ordinal % folds
    if len(assignments) != len(objects):
        raise GravityItem1EffectiveDimensionError("incomplete Item 1 folds")
    return assignments


def _prediction(object_row: Mapping[str, Any], beta: float) -> tuple[np.ndarray, int]:
    value = np.asarray(object_row["base"]) + beta * np.asarray(object_row["component"])
    invalid = int(np.sum(~np.isfinite(value) | (value <= 0)))
    safe = np.maximum(value, np.finfo(np.float64).tiny)
    if object_row["target_scale"] == "linear_velocity":
        return np.sqrt(safe), invalid
    return np.log10(safe), invalid


def _loss(object_row: Mapping[str, Any], beta: float) -> tuple[float, int]:
    prediction, invalid = _prediction(object_row, beta)
    standardized = (prediction - np.asarray(object_row["observed"])) / np.asarray(
        object_row["sigma"]
    )
    penalty = float(invalid) * 1.0e24
    return float(np.sum(standardized**2) + penalty), invalid


def _oracle_labels(
    objects: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    grid_config = config["coefficient_label_grid"]
    lower = float(grid_config["minimum"])
    upper = float(grid_config["maximum"])
    step = float(grid_config["step"])
    grid = np.linspace(lower, upper, round((upper - lower) / step) + 1)
    rows: dict[str, dict[str, Any]] = {}
    for object_row in objects:
        scored = [(*_loss(object_row, float(beta)), float(beta)) for beta in grid]
        loss, invalid, beta = min(scored, key=lambda row: (row[0], abs(row[2])))
        rows[str(object_row["key"])] = {
            "at_grid_boundary": bool(beta == lower or beta == upper),
            "beta": beta,
            "chi_square": loss,
            "invalid_predictions": invalid,
        }
    return rows


def _domain_weights(objects: Sequence[Mapping[str, Any]]) -> np.ndarray:
    counts = {domain: sum(row["domain"] == domain for row in objects) for domain in DOMAINS}
    if any(counts[domain] == 0 for domain in DOMAINS):
        raise GravityItem1EffectiveDimensionError("a model fit is missing a population")
    return np.asarray([0.5 / counts[str(row["domain"])] for row in objects], dtype=np.float64)


def _feature_matrix(objects: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(objects), 0), dtype=np.float64)
    matrix = np.asarray(
        [[float(row["features"][feature]) for feature in features] for row in objects],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(matrix)):
        raise GravityItem1EffectiveDimensionError("non-finite model feature")
    return matrix


def _fit_model(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    features = [str(value) for value in model.get("features", ())]
    raw = _feature_matrix(objects, features)
    weights = _domain_weights(objects)
    if raw.shape[1]:
        means = np.sum(raw * weights[:, None], axis=0) / float(np.sum(weights))
        centered = raw - means
        scales = np.sqrt(np.sum(weights[:, None] * centered**2, axis=0) / float(np.sum(weights)))
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        standardized = centered / scales
    else:
        means = np.empty(0, dtype=np.float64)
        scales = np.empty(0, dtype=np.float64)
        standardized = raw
    design = np.column_stack([np.ones(len(objects), dtype=np.float64), standardized])
    target = np.asarray([float(labels[str(row["key"])]["beta"]) for row in objects])
    weighted_design = design * np.sqrt(weights[:, None])
    weighted_target = target * np.sqrt(weights)
    ridge = float(config["cross_validation"]["ridge"])
    gram = weighted_design.T @ weighted_design
    regularizer = np.eye(gram.shape[0], dtype=np.float64) * ridge
    regularizer[0, 0] = 0.0
    coefficients = np.linalg.solve(gram + regularizer, weighted_design.T @ weighted_target)
    return {
        "coefficients": [_metric(value) for value in coefficients],
        "feature_means": [_metric(value) for value in means],
        "feature_scales": [_metric(value) for value in scales],
        "features": features,
        "model_id": str(model["id"]),
    }


def _predict_model(
    fit: Mapping[str, Any],
    objects: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], int]:
    features = [str(value) for value in fit["features"]]
    raw = _feature_matrix(objects, features)
    means = np.asarray([float(value) for value in fit["feature_means"]])
    scales = np.asarray([float(value) for value in fit["feature_scales"]])
    standardized = (raw - means) / scales if features else raw
    design = np.column_stack([np.ones(len(objects), dtype=np.float64), standardized])
    unbounded = design @ np.asarray([float(value) for value in fit["coefficients"]])
    lower, upper = [float(value) for value in config["cross_validation"]["prediction_clip"]]
    clipped = np.clip(unbounded, lower, upper)
    clip_count = int(np.sum(clipped != unbounded))
    return {
        str(row["key"]): float(value) for row, value in zip(objects, clipped, strict=True)
    }, clip_count


def _beta_metrics(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, float],
) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = {}
    for domain in DOMAINS:
        members = [row for row in objects if row["domain"] == domain]
        target = np.asarray([float(labels[str(row["key"])]["beta"]) for row in members])
        predicted = np.asarray([float(predictions[str(row["key"])]) for row in members])
        residual = predicted - target
        sse = float(residual @ residual)
        centered = target - float(np.mean(target))
        sst = float(centered @ centered)
        by_domain[domain] = {
            "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
            "mean_squared_error": _metric(float(np.mean(residual**2))),
            "objects": len(members),
            "r2": _metric(1.0 - sse / sst) if sst > 1.0e-15 else "0.000000000000e+00",
        }
    return {
        "balanced_mean_squared_error": _metric(
            0.5 * sum(float(by_domain[domain]["mean_squared_error"]) for domain in DOMAINS)
        ),
        "balanced_r2": _metric(0.5 * sum(float(by_domain[domain]["r2"]) for domain in DOMAINS)),
        "by_population": by_domain,
    }


def _observational_score(
    objects: Sequence[Mapping[str, Any]], betas: Mapping[str, float]
) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = {}
    rendered_predictions: list[str] = []
    total_invalid = 0
    for domain in DOMAINS:
        chi_square = 0.0
        invalid = 0
        points = 0
        members = [row for row in objects if row["domain"] == domain]
        for row in members:
            beta = float(betas[str(row["key"])])
            loss, bad = _loss(row, beta)
            prediction, _ = _prediction(row, beta)
            chi_square += loss
            invalid += bad
            points += int(row["point_count"])
            rendered_predictions.extend(format(float(value), ".15e") for value in prediction)
        total_invalid += invalid
        by_domain[domain] = {
            "chi_square": _metric(chi_square),
            "chi_square_per_point": _metric(chi_square / points),
            "invalid_predictions": invalid,
            "objects": len(members),
            "points": points,
        }
    return {
        "balanced_chi_square_per_point": _metric(
            0.5 * sum(float(by_domain[domain]["chi_square_per_point"]) for domain in DOMAINS)
        ),
        "by_population": by_domain,
        "invalid_predictions": total_invalid,
        "prediction_manifest_sha256": canonical_sha256(rendered_predictions),
    }


def _known_rar_score(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_domain: dict[str, dict[str, Any]] = {}
    rendered_predictions: list[str] = []
    for domain in DOMAINS:
        members = [row for row in objects if row["domain"] == domain]
        chi_square = 0.0
        points = 0
        for row in members:
            prediction = np.asarray(row["known_prediction"], dtype=np.float64)
            residual = (prediction - np.asarray(row["observed"])) / np.asarray(row["sigma"])
            chi_square += float(residual @ residual)
            points += int(row["point_count"])
            rendered_predictions.extend(format(float(value), ".15e") for value in prediction)
        by_domain[domain] = {
            "chi_square": _metric(chi_square),
            "chi_square_per_point": _metric(chi_square / points),
            "invalid_predictions": 0,
            "objects": len(members),
            "points": points,
        }
    return {
        "balanced_chi_square_per_point": _metric(
            0.5 * sum(float(by_domain[domain]["chi_square_per_point"]) for domain in DOMAINS)
        ),
        "by_population": by_domain,
        "invalid_predictions": 0,
        "prediction_manifest_sha256": canonical_sha256(rendered_predictions),
    }


def _model_oof(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, int],
    model: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    folds = int(config["cross_validation"]["outer_folds"])
    predictions: dict[str, float] = {}
    ledger: list[dict[str, Any]] = []
    clip_count = 0
    for fold in range(folds):
        training = [row for row in objects if assignments[str(row["key"])] != fold]
        heldout = [row for row in objects if assignments[str(row["key"])] == fold]
        fit = _fit_model(training, labels, model, config)
        values, clipped = _predict_model(fit, heldout, config)
        predictions.update(values)
        clip_count += clipped
        ledger.append(
            {
                "fit": fit,
                "fold": fold,
                "heldout_clusters": sum(row["domain"] == "cluster" for row in heldout),
                "heldout_galaxies": sum(row["domain"] == "galaxy" for row in heldout),
            }
        )
    if len(predictions) != len(objects):
        raise GravityItem1EffectiveDimensionError("incomplete model OOF predictions")
    return predictions, ledger, clip_count


def _nested_model_oof(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    assignments: Mapping[str, int],
    models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], list[dict[str, Any]], int]:
    folds = int(config["cross_validation"]["outer_folds"])
    predictions: dict[str, float] = {}
    ledger: list[dict[str, Any]] = []
    total_clipped = 0
    for outer_fold in range(folds):
        outer_training = [row for row in objects if assignments[str(row["key"])] != outer_fold]
        outer_heldout = [row for row in objects if assignments[str(row["key"])] == outer_fold]
        inner_scores: list[dict[str, Any]] = []
        for model in models:
            inner_predictions: dict[str, float] = {}
            inner_clipped = 0
            for inner_fold in range(folds):
                if inner_fold == outer_fold:
                    continue
                inner_training = [
                    row for row in outer_training if assignments[str(row["key"])] != inner_fold
                ]
                inner_heldout = [
                    row for row in outer_training if assignments[str(row["key"])] == inner_fold
                ]
                fit = _fit_model(inner_training, labels, model, config)
                values, clipped = _predict_model(fit, inner_heldout, config)
                inner_predictions.update(values)
                inner_clipped += clipped
            metrics = _beta_metrics(outer_training, labels, inner_predictions)
            inner_scores.append(
                {
                    "clip_count": inner_clipped,
                    "model_id": str(model["id"]),
                    "score": metrics["balanced_mean_squared_error"],
                }
            )
        selected = min(inner_scores, key=lambda row: (float(row["score"]), row["model_id"]))
        selected_model = next(row for row in models if row["id"] == selected["model_id"])
        fit = _fit_model(outer_training, labels, selected_model, config)
        values, clipped = _predict_model(fit, outer_heldout, config)
        predictions.update(values)
        total_clipped += clipped
        ledger.append(
            {
                "fit": fit,
                "fold": outer_fold,
                "heldout_clusters": sum(row["domain"] == "cluster" for row in outer_heldout),
                "heldout_galaxies": sum(row["domain"] == "galaxy" for row in outer_heldout),
                "inner_scores": sorted(inner_scores, key=lambda row: row["model_id"]),
                "selected_model_id": selected["model_id"],
            }
        )
    if len(predictions) != len(objects):
        raise GravityItem1EffectiveDimensionError("incomplete nested OOF predictions")
    return predictions, ledger, total_clipped


def _fixed_betas(objects: Sequence[Mapping[str, Any]], rule_id: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for row in objects:
        support = float(row["features"]["support_dimension"])
        profile = float(row["features"]["profile_mass_dimension"])
        if rule_id == "fixed_galaxy_parent":
            beta = 0.5
        elif rule_id == "inverse_support_dimension":
            beta = 1.0 / support
        elif rule_id == "inverse_profile_dimension":
            beta = 1.0 / float(np.clip(profile, 0.25, 4.0))
        elif rule_id == "posthoc_support_bridge":
            beta = 0.5 * 4.0 ** (support - 2.0)
        else:
            raise GravityItem1EffectiveDimensionError("unknown fixed Item 1 rule")
        values[str(row["key"])] = float(np.clip(beta, 0.0, 4.0))
    return values


def _public_object(
    row: Mapping[str, Any],
    oracle: Mapping[str, Any],
    nested_beta: float,
    selected_model: str,
) -> dict[str, Any]:
    return {
        "domain": row["domain"],
        "features": {key: _metric(value) for key, value in sorted(row["features"].items())},
        "name": row["name"],
        "nested_oof_beta": _metric(nested_beta),
        "oracle_beta_at_grid_boundary": oracle["at_grid_boundary"],
        "oracle_beta_target_derived": _metric(oracle["beta"]),
        "point_count": row["point_count"],
        "selected_model_id": selected_model,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Run the real-data Item 1 falsification and build its sealed receipt."""

    root = root.resolve()
    config = load_config(root)
    objects = prepare_objects(root, config)
    cv = config["cross_validation"]
    assignments = _fold_assignments(
        objects, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"])
    )
    labels = _oracle_labels(objects, config)
    models = [dict(row) for row in config["trainable_models"]]

    fixed_results: dict[str, Any] = {}
    fixed_predictions: dict[str, dict[str, float]] = {}
    for rule in config["fixed_rules"]:
        rule_id = str(rule["id"])
        betas = _fixed_betas(objects, rule_id)
        fixed_predictions[rule_id] = betas
        fixed_results[rule_id] = {
            "coefficient_prediction": _beta_metrics(objects, labels, betas),
            "origin_label": rule["origin_label"],
            "qualifying_first_principles_rule": rule["qualifying_first_principles_rule"],
            "score": _observational_score(objects, betas),
        }

    model_results: dict[str, Any] = {}
    model_predictions: dict[str, dict[str, float]] = {}
    for model in models:
        predictions, ledger, clipped = _model_oof(objects, labels, assignments, model, config)
        model_id = str(model["id"])
        model_predictions[model_id] = predictions
        model_results[model_id] = {
            "clip_count": clipped,
            "coefficient_prediction": _beta_metrics(objects, labels, predictions),
            "fold_ledger": ledger,
            "qualifying_continuous_dimension_model": model["qualifying_continuous_dimension_model"],
            "score": _observational_score(objects, predictions),
        }

    nested_predictions, nested_ledger, nested_clipped = _nested_model_oof(
        objects, labels, assignments, models, config
    )
    nested_beta_metrics = _beta_metrics(objects, labels, nested_predictions)
    nested_score = _observational_score(objects, nested_predictions)
    constant_score = model_results["constant"]["score"]
    qualifying_models = {
        str(row["id"]) for row in models if row["qualifying_continuous_dimension_model"] is True
    }
    selected_models = [str(row["selected_model_id"]) for row in nested_ledger]

    inverse_score = fixed_results["inverse_support_dimension"]["score"]
    parent_score = fixed_results["fixed_galaxy_parent"]["score"]
    gate_checks = {
        "confirmation_or_direct_lensing_completed": False,
        "inverse_support_dimension_not_worse_than_parent_in_both_populations": all(
            float(inverse_score["by_population"][domain]["chi_square"])
            <= float(parent_score["by_population"][domain]["chi_square"]) + 1.0e-9
            for domain in DOMAINS
        ),
        "nested_model_beats_oof_constant_in_both_populations": all(
            float(nested_score["by_population"][domain]["chi_square"])
            < float(constant_score["by_population"][domain]["chi_square"])
            for domain in DOMAINS
        ),
        "nested_model_beta_r2_positive_in_both_populations": all(
            float(nested_beta_metrics["by_population"][domain]["r2"]) > 0.0 for domain in DOMAINS
        ),
        "qualifying_continuous_dimension_model_selected_in_every_outer_fold": all(
            model_id in qualifying_models for model_id in selected_models
        ),
        "posthoc_support_bridge_excluded_from_admission": (
            fixed_results["posthoc_support_bridge"]["qualifying_first_principles_rule"] is False
        ),
        "target_fields_absent_from_dimension_feature_builder": True,
        "whole_object_target_blind_outer_predictions_complete": (
            len(nested_predictions) == len(objects)
        ),
    }
    development_pass = all(
        gate_checks[key]
        for key in (
            "inverse_support_dimension_not_worse_than_parent_in_both_populations",
            "nested_model_beats_oof_constant_in_both_populations",
            "nested_model_beta_r2_positive_in_both_populations",
            "qualifying_continuous_dimension_model_selected_in_every_outer_fold",
            "posthoc_support_bridge_excluded_from_admission",
            "target_fields_absent_from_dimension_feature_builder",
            "whole_object_target_blind_outer_predictions_complete",
        )
    )
    selected_by_fold = {int(row["fold"]): row["selected_model_id"] for row in nested_ledger}
    per_object = [
        _public_object(
            row,
            labels[str(row["key"])],
            nested_predictions[str(row["key"])],
            str(selected_by_fold[assignments[str(row["key"])]]),
        )
        for row in objects
    ]
    grid = config["coefficient_label_grid"]
    grid_cells = round((float(grid["maximum"]) - float(grid["minimum"])) / float(grid["step"])) + 1
    feature_ranges = {
        domain: {
            feature: {
                "maximum": _metric(
                    max(
                        float(row["features"][feature])
                        for row in objects
                        if row["domain"] == domain
                    )
                ),
                "median": _metric(
                    float(
                        np.median(
                            [
                                float(row["features"][feature])
                                for row in objects
                                if row["domain"] == domain
                            ]
                        )
                    )
                ),
                "minimum": _metric(
                    min(
                        float(row["features"][feature])
                        for row in objects
                        if row["domain"] == domain
                    )
                ),
            }
            for feature in (
                "profile_mass_dimension",
                "local_dimension_median",
                "local_dimension_iqr",
            )
        }
        for domain in DOMAINS
    }

    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "GRAVITY_ROADMAP_ITEM_01_EFFECTIVE_DIMENSION",
        "decision": (
            "PASS_ITEM1_EFFECTIVE_DIMENSION_DEVELOPMENT_GATE"
            if development_pass
            else "INCONCLUSIVE_ITEM1_EFFECTIVE_DIMENSION"
        ),
        "claims": {
            "alternative_to_gr_established": False,
            "binary_support_dimension_is_first_principles": False,
            "continuous_dimension_explains_cross_scale_response": development_pass,
            "direct_lensing_test_completed": False,
            "historical_novelty_established": False,
            "roadmap_item_1_complete": development_pass,
            "sequential_G6_G7_G8_advanced": False,
            "sparc_confirmation_opened": False,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "clash_clusters": sum(row["domain"] == "cluster" for row in objects),
            "clash_points": sum(
                row["point_count"] for row in objects if row["domain"] == "cluster"
            ),
            "coefficient_grid_cells_per_object": grid_cells,
            "direct_lensing_likelihood_evaluations": 0,
            "fixed_rules": len(config["fixed_rules"]),
            "formula_classes": len(config["fixed_rules"]) + len(models) + 1,
            "oracle_candidate_point_evaluations": grid_cells
            * sum(int(row["point_count"]) for row in objects),
            "outer_folds": int(cv["outer_folds"]),
            "paid_model_calls": 0,
            "sparc_confirmation_evaluator_accesses": 0,
            "sparc_exploration_galaxies": sum(row["domain"] == "galaxy" for row in objects),
            "sparc_exploration_points": sum(
                row["point_count"] for row in objects if row["domain"] == "galaxy"
            ),
            "trainable_model_classes": len(models),
            "unique_formula_prediction_classes": len(
                {fixed_results[key]["score"]["prediction_manifest_sha256"] for key in fixed_results}
                | {
                    model_results[key]["score"]["prediction_manifest_sha256"]
                    for key in model_results
                }
                | {nested_score["prediction_manifest_sha256"]}
            ),
        },
        "data_lineage": {
            "clash": config["population"]["clash_target_status"],
            "feature_inputs": "radius_and_published_baryonic_acceleration_only",
            "sparc": "139_exploration_galaxies_only",
            "target_derived_oracle_role": (
                "training_label_and_descriptive_ceiling_never_formula_input"
            ),
        },
        "feature_ranges": feature_ranges,
        "fixed_rule_results": fixed_results,
        "gate_checks": gate_checks,
        "known_rar_control": _known_rar_score(objects),
        "model_results": model_results,
        "nested_model": {
            "clip_count": nested_clipped,
            "coefficient_prediction": nested_beta_metrics,
            "fold_ledger": nested_ledger,
            "score": nested_score,
            "selected_model_counts": {
                model_id: selected_models.count(model_id)
                for model_id in sorted(set(selected_models))
            },
        },
        "per_object_diagnostics": per_object,
        "specific_hypothesis_results": {
            "beta_equals_inverse_support_dimension": (
                "SURVIVES_CURRENT_DIAGNOSTIC"
                if gate_checks[
                    "inverse_support_dimension_not_worse_than_parent_in_both_populations"
                ]
                else "REJECTED_BY_CURRENT_CROSS_SCALE_DIAGNOSTIC"
            ),
            "continuous_baryonic_mass_profile_dimension": (
                "SURVIVES_DEVELOPMENT_GATE"
                if development_pass
                else "NOT_SHOWN_TO_GENERATE_THE_CROSS_SCALE_RESPONSE"
            ),
            "posthoc_support_bridge": (
                "DESCRIPTIVE_REPRODUCTION_ONLY_REQUIRES_NEW_DATA_CONFIRMATION"
            ),
        },
        "limitations": [
            "The CLASH target is reconstructed through spherical NFW posteriors and is not a direct lensing likelihood.",
            "D_support is an assumed disk-versus-sphere class and can act as a population proxy.",
            "M_eq proportional to g_bar*r^2 is a spherical-equivalent profile; for disks its slope is not literal spatial or spacetime dimension.",
            "Per-object oracle coefficients are target-derived training labels and cannot qualify as formulas or first principles.",
            "The posthoc bridge was constructed after both population coefficients were known and has no confirmation force.",
            "No filamentary systems or independently measured three-dimensional baryonic shape profiles are present in this run.",
        ],
        "next_action": (
            "Measure shape and anisotropy continuously from baryonic imaging or gas maps, add at least one intermediate or filamentary geometry population, and freeze the coefficient rule before an independent cluster or galaxy sample."
        ),
        "source_bindings": {
            "config": _binding(root, CONFIG_PATH),
            "predecessor_action": _binding(
                root, "runs/gravity/g4/auxiliary-action-derivation-v6.json"
            ),
            "predecessor_cluster": _binding(
                root, "runs/gravity/g4/cluster-lensing-exploration-v7.json"
            ),
            "roadmap": _binding(root, str(config["roadmap_binding"]["path"])),
            "source": _binding(root, SOURCE_PATH),
            "test": _binding(root, TEST_PATH),
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Fail closed on evidence drift, leakage, or an inflated Item 1 claim."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityItem1EffectiveDimensionError("Item 1 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem1EffectiveDimensionError("Item 1 receipt seal changed")
    counts = receipt.get("counts", {})
    if (
        counts.get("sparc_confirmation_evaluator_accesses") != 0
        or counts.get("direct_lensing_likelihood_evaluations") != 0
        or counts.get("paid_model_calls") != 0
        or counts.get("sparc_exploration_galaxies") != 139
        or counts.get("clash_clusters") != 20
    ):
        raise GravityItem1EffectiveDimensionError("Item 1 accounting changed")
    claims = receipt.get("claims", {})
    for claim in (
        "alternative_to_gr_established",
        "binary_support_dimension_is_first_principles",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
    ):
        if claims.get(claim) is not False:
            raise GravityItem1EffectiveDimensionError("Item 1 overstates its claim")
    if (
        receipt.get("fixed_rule_results", {})
        .get("posthoc_support_bridge", {})
        .get("qualifying_first_principles_rule")
        is not False
    ):
        raise GravityItem1EffectiveDimensionError("posthoc bridge entered admission")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityItem1EffectiveDimensionError("Item 1 config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "predecessor_action": "runs/gravity/g4/auxiliary-action-derivation-v6.json",
        "predecessor_cluster": "runs/gravity/g4/cluster-lensing-exploration-v7.json",
        "roadmap": str(config["roadmap_binding"]["path"]),
        "source": SOURCE_PATH,
        "test": TEST_PATH,
    }
    bindings = receipt.get("source_bindings", {})
    if set(bindings) != set(expected):
        raise GravityItem1EffectiveDimensionError("Item 1 source binding set changed")
    for key, path in expected.items():
        if bindings.get(key) != _binding(root, path):
            raise GravityItem1EffectiveDimensionError(f"Item 1 {key} binding changed")
    passed = receipt.get("decision") == "PASS_ITEM1_EFFECTIVE_DIMENSION_DEVELOPMENT_GATE"
    if claims.get("roadmap_item_1_complete") is not passed:
        raise GravityItem1EffectiveDimensionError("Item 1 decision and claim disagree")
    if claims.get("continuous_dimension_explains_cross_scale_response") is not passed:
        raise GravityItem1EffectiveDimensionError("Item 1 dimension claim changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityItem1EffectiveDimensionError(
                f"refusing to overwrite immutable Item 1 receipt: {path}"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    output = args.output or (root / OUTPUT_PATH)
    if args.check:
        validate_receipt(_load_json(output), root=root)
        return 0
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    _write_immutable(output, receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "CONFIG_PATH",
    "OUTPUT_PATH",
    "GravityItem1EffectiveDimensionError",
    "build_receipt",
    "dimension_features",
    "load_config",
    "main",
    "prepare_objects",
    "validate_receipt",
]
