"""Nested held-out evaluation for gravity Item 3 smooth-density attempt 2."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item3_smooth_density_profiles as source
from .sigma_core import canonical_json_bytes, canonical_sha256

OUTPUT_PATH = "runs/gravity/roadmap/item-03-smooth-density-profiles-v2.json"
FEATURE_PATH = source.RADIAL_FEATURE_PATH
SOURCE_MANIFEST_PATH = source.SOURCE_MANIFEST_PATH


class GravityItem3SmoothExperimentError(RuntimeError):
    """Raised when the frozen evaluation boundary or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def _load_source_manifest(root: Path) -> dict[str, Any]:
    path = root / SOURCE_MANIFEST_PATH
    manifest = json.loads(path.read_text(encoding="utf-8"))
    content = dict(manifest)
    claimed = content.pop("content_sha256")
    if canonical_sha256(content) != claimed:
        raise GravityItem3SmoothExperimentError("source manifest content hash changed")
    if manifest.get("freeze_commit") != "68487eb9d9f00adfb1238c8f904b12e76b9bc9c2":
        raise GravityItem3SmoothExperimentError("freeze commit changed")
    if manifest.get("reserved_confirmation_profiles_opened") != 0:
        raise GravityItem3SmoothExperimentError("confirmation profile was opened")
    return manifest


def _load_rows(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    path = root / FEATURE_PATH
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    text_fields = {"variant", "domain", "name"}
    integer_fields = {"has_stellar_robustness"}
    rows: list[dict[str, Any]] = []
    for index, raw_row in enumerate(raw):
        row: dict[str, Any] = {"row_id": index}
        for key, value in raw_row.items():
            if key in text_fields:
                row[key] = value
            elif key in integer_fields:
                row[key] = int(value)
            else:
                row[key] = float(value)
        row["object_key"] = f"{row['domain']}:{row['name']}"
        rows.append(row)
    primary = [row for row in rows if row["variant"] == "primary"]
    stellar = [row for row in rows if row["variant"] == "stellar_augmented"]
    if not primary or not stellar:
        raise GravityItem3SmoothExperimentError("missing primary or stellar robustness rows")
    for population in (primary, stellar):
        if any(
            not math.isfinite(float(value))
            for row in population
            for key, value in row.items()
            if key
            not in {
                "variant",
                "domain",
                "name",
                "object_key",
                "gbar_with_stars_m_s2",
            }
        ):
            raise GravityItem3SmoothExperimentError("non-finite admitted radial row")
    return primary, stellar


def _models(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    ids: set[str] = set()
    for raw in config["model_families"]:
        model_id = str(raw["id"])
        features = tuple(str(value) for value in raw["features"])
        if model_id in ids or len(features) != len(set(features)):
            raise GravityItem3SmoothExperimentError("duplicate model or feature")
        ids.add(model_id)
        models.append(
            {
                "id": model_id,
                "features": features,
                "qualifying": bool(raw["qualifying"]),
            }
        )
    return models


def _object_folds(
    rows: Sequence[Mapping[str, Any]], *, salt: str, folds: int
) -> dict[str, int]:
    strata: defaultdict[str, set[str]] = defaultdict(set)
    for row in rows:
        strata[str(row["domain"])].add(str(row["object_key"]))
    assignments: dict[str, int] = {}
    for domain, keys in sorted(strata.items()):
        ordered = sorted(
            keys,
            key=lambda key: hashlib.sha256(
                f"{salt}|{domain}|{key}".encode()
            ).hexdigest(),
        )
        if len(ordered) < folds:
            raise GravityItem3SmoothExperimentError(
                f"too few {domain} objects for {folds} folds"
            )
        for index, key in enumerate(ordered):
            assignments[key] = index % folds
    return assignments


def _balanced_weights(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    by_domain: defaultdict[str, defaultdict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    for row in rows:
        by_domain[str(row["domain"])][str(row["object_key"])] += 1
    domains = sorted(by_domain)
    weights = np.empty(len(rows), dtype=np.float64)
    for index, row in enumerate(rows):
        domain = str(row["domain"])
        key = str(row["object_key"])
        weights[index] = (
            1.0
            / len(domains)
            / len(by_domain[domain])
            / by_domain[domain][key]
        )
    weights /= weights.sum()
    return weights


def _matrix(
    rows: Sequence[Mapping[str, Any]], features: Sequence[str]
) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    return np.asarray(
        [[float(row[feature]) for feature in features] for row in rows],
        dtype=np.float64,
    )


def _target(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [float(row["response_log10_ratio"]) for row in rows], dtype=np.float64
    )


def _fit_predict(
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str],
    penalty: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    train_y = _target(train)
    weights = _balanced_weights(train)
    if not features:
        mean = float(np.sum(weights * train_y))
        return np.full(len(test), mean), {
            "intercept": mean,
            "coefficients": {},
        }
    train_x = _matrix(train, features)
    test_x = _matrix(test, features)
    mean_x = np.sum(weights[:, None] * train_x, axis=0)
    variance_x = np.sum(weights[:, None] * (train_x - mean_x) ** 2, axis=0)
    scale_x = np.sqrt(np.maximum(variance_x, 1.0e-12))
    z_train = (train_x - mean_x) / scale_x
    z_test = (test_x - mean_x) / scale_x
    design = np.column_stack([np.ones(len(train)), z_train])
    weighted_design = design * np.sqrt(weights)[:, None]
    weighted_target = train_y * np.sqrt(weights)
    regularizer = np.eye(design.shape[1]) * float(penalty)
    regularizer[0, 0] = 0.0
    coefficient = np.linalg.solve(
        weighted_design.T @ weighted_design + regularizer,
        weighted_design.T @ weighted_target,
    )
    prediction = np.column_stack([np.ones(len(test)), z_test]) @ coefficient
    raw_coefficients = coefficient[1:] / scale_x
    raw_intercept = coefficient[0] - float(np.sum(raw_coefficients * mean_x))
    return prediction, {
        "intercept": float(raw_intercept),
        "coefficients": {
            feature: float(value)
            for feature, value in zip(features, raw_coefficients, strict=True)
        },
    }


def _weighted_mse(
    rows: Sequence[Mapping[str, Any]], prediction: np.ndarray
) -> float:
    residual = _target(rows) - np.asarray(prediction, dtype=np.float64)
    return float(np.sum(_balanced_weights(rows) * residual**2))


def _inner_penalty(
    train: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str],
    penalties: Sequence[float],
    salt: str,
) -> tuple[float, float, int]:
    inner_folds = 4
    assignments = _object_folds(train, salt=salt, folds=inner_folds)
    best: tuple[float, float] | None = None
    fits = 0
    for penalty in penalties:
        losses: list[float] = []
        for fold in range(inner_folds):
            inner_train = [
                row for row in train if assignments[str(row["object_key"])] != fold
            ]
            validation = [
                row for row in train if assignments[str(row["object_key"])] == fold
            ]
            prediction, _ = _fit_predict(
                inner_train,
                validation,
                features=features,
                penalty=float(penalty),
            )
            losses.append(_weighted_mse(validation, prediction))
            fits += 1
        candidate = (float(np.mean(losses)), float(penalty))
        if best is None or candidate < best:
            best = candidate
    if best is None:  # pragma: no cover
        raise GravityItem3SmoothExperimentError("no ridge penalty")
    return best[1], best[0], fits


def _nested_evaluation(
    rows: Sequence[Mapping[str, Any]],
    *,
    models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    fold_assignments: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    folds = int(config["cross_validation"]["outer_folds"])
    salt = str(config["cross_validation"]["fold_salt"])
    assignments = (
        dict(fold_assignments)
        if fold_assignments is not None
        else _object_folds(rows, salt=salt, folds=folds)
    )
    penalties = [float(value) for value in config["cross_validation"]["ridge_penalties"]]
    row_position = {int(row["row_id"]): index for index, row in enumerate(rows)}
    predictions = {
        str(model["id"]): np.full(len(rows), np.nan, dtype=np.float64)
        for model in models
    }
    selected_prediction = np.full(len(rows), np.nan, dtype=np.float64)
    qualifying_prediction = np.full(len(rows), np.nan, dtype=np.float64)
    fold_records: list[dict[str, Any]] = []
    inner_fits = 0
    final_fits = 0
    for outer_fold in range(folds):
        train = [
            row for row in rows if assignments[str(row["object_key"])] != outer_fold
        ]
        test = [
            row for row in rows if assignments[str(row["object_key"])] == outer_fold
        ]
        choices: list[dict[str, Any]] = []
        for model in models:
            penalty, inner_loss, fits = _inner_penalty(
                train,
                features=model["features"],
                penalties=penalties,
                salt=f"{salt}|outer={outer_fold}|model={model['id']}",
            )
            inner_fits += fits
            prediction, parameters = _fit_predict(
                train,
                test,
                features=model["features"],
                penalty=penalty,
            )
            final_fits += 1
            for row, value in zip(test, prediction, strict=True):
                predictions[str(model["id"])][row_position[int(row["row_id"])]] = value
            choices.append(
                {
                    "id": str(model["id"]),
                    "qualifying": bool(model["qualifying"]),
                    "features": list(model["features"]),
                    "penalty": penalty,
                    "inner_loss": inner_loss,
                    "parameters": parameters,
                }
            )
        selected = min(choices, key=lambda choice: (choice["inner_loss"], choice["id"]))
        qualifying = min(
            (choice for choice in choices if choice["qualifying"]),
            key=lambda choice: (choice["inner_loss"], choice["id"]),
        )
        for row in test:
            position = row_position[int(row["row_id"])]
            selected_prediction[position] = predictions[selected["id"]][position]
            qualifying_prediction[position] = predictions[qualifying["id"]][position]
        fold_records.append(
            {
                "fold": outer_fold,
                "test_objects": sorted({str(row["object_key"]) for row in test}),
                "selected_model": selected["id"],
                "selected_qualifying": bool(selected["qualifying"]),
                "selected_penalty": _metric(selected["penalty"]),
                "selected_inner_mse": _metric(selected["inner_loss"]),
                "qualifying_selector_model": qualifying["id"],
                "qualifying_selector_penalty": _metric(qualifying["penalty"]),
                "models": [
                    {
                        "id": choice["id"],
                        "penalty": _metric(choice["penalty"]),
                        "inner_mse": _metric(choice["inner_loss"]),
                        "parameters": {
                            "intercept": _metric(choice["parameters"]["intercept"]),
                            "coefficients": {
                                key: _metric(value)
                                for key, value in choice["parameters"][
                                    "coefficients"
                                ].items()
                            },
                        },
                    }
                    for choice in choices
                ],
            }
        )
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise GravityItem3SmoothExperimentError("incomplete model prediction")
    if np.any(~np.isfinite(selected_prediction)) or np.any(
        ~np.isfinite(qualifying_prediction)
    ):
        raise GravityItem3SmoothExperimentError("incomplete selector prediction")
    return {
        "assignments": assignments,
        "predictions": predictions,
        "selected_prediction": selected_prediction,
        "qualifying_prediction": qualifying_prediction,
        "fold_records": fold_records,
        "compute": {"inner_fits": inner_fits, "final_fits": final_fits},
    }


def _metrics(
    rows: Sequence[Mapping[str, Any]], prediction: np.ndarray
) -> dict[str, Any]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = _target(rows)
    weights = _balanced_weights(rows)
    mean = float(np.sum(weights * target))
    sse = float(np.sum(weights * (target - prediction) ** 2))
    variance = float(np.sum(weights * (target - mean) ** 2))
    result: dict[str, Any] = {
        "r2": 1.0 - sse / variance,
        "mse": sse,
        "rmse_dex": math.sqrt(sse),
        "median_residual_dex": float(np.median(target - prediction)),
        "median_absolute_fractional_acceleration_error": float(
            np.median(np.abs(10.0 ** (prediction - target) - 1.0))
        ),
    }
    if all(str(row["domain"]) == "galaxy" for row in rows):
        result["median_absolute_fractional_speed_error"] = float(
            np.median(np.abs(10.0 ** ((prediction - target) / 2.0) - 1.0))
        )
    return result


def _metrics_by_domain(
    rows: Sequence[Mapping[str, Any]], prediction: np.ndarray
) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {"overall": _metrics(rows, prediction)}
    for domain in sorted({str(row["domain"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["domain"] == domain]
        result[domain] = _metrics(
            [rows[index] for index in indices], prediction[indices]
        )
    return result


DENSITY_BASE_FEATURES = (
    "s",
    "v",
    "m",
    "c",
    "transition_product",
    "transition_balance",
)


def _permuted_density_rows(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    result = [dict(row) for row in rows]
    by_key: defaultdict[str, list[int]] = defaultdict(list)
    by_domain: defaultdict[str, list[str]] = defaultdict(list)
    for index, row in enumerate(rows):
        by_key[str(row["object_key"])].append(index)
    for key in sorted(by_key):
        by_key[key].sort(key=lambda index: float(rows[index]["radius_kpc"]))
        domain = str(rows[by_key[key][0]]["domain"])
        by_domain[domain].append(key)
    for domain, recipient_keys in sorted(by_domain.items()):
        donor_keys = list(recipient_keys)
        rng.shuffle(donor_keys)
        for recipient, donor in zip(recipient_keys, donor_keys, strict=True):
            recipient_indices = by_key[recipient]
            donor_indices = by_key[donor]
            recipient_u = np.linspace(0.0, 1.0, len(recipient_indices))
            donor_u = np.linspace(0.0, 1.0, len(donor_indices))
            for feature in DENSITY_BASE_FEATURES:
                donor_values = np.asarray(
                    [float(rows[index][feature]) for index in donor_indices]
                )
                values = np.interp(recipient_u, donor_u, donor_values)
                for index, value in zip(recipient_indices, values, strict=True):
                    result[index][feature] = float(value)
            for index in recipient_indices:
                result[index]["m_x_c"] = float(result[index]["m"]) * float(
                    result[index]["c"]
                )
                result[index]["a_x_m"] = float(result[index]["a"]) * float(
                    result[index]["m"]
                )
                result[index]["a_x_c"] = float(result[index]["a"]) * float(
                    result[index]["c"]
                )
    return result


def _permutation_test(
    rows: Sequence[Mapping[str, Any]],
    *,
    observed: Mapping[str, Any],
    models: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    qualifying_models = [model for model in models if model["qualifying"]]
    baseline_prediction = observed["predictions"]["gbar_polynomial"]
    baseline_mse = _weighted_mse(rows, baseline_prediction)
    observed_improvement = baseline_mse - _weighted_mse(
        rows, observed["qualifying_prediction"]
    )
    count = int(config["cross_validation"]["stratified_object_permutation_count"])
    salt = str(config["cross_validation"]["permutation_salt"])
    assignments = observed["assignments"]
    null_improvements: list[float] = []
    fits = 0
    for index in range(count):
        seed = int.from_bytes(
            hashlib.sha256(f"{salt}|{index}".encode()).digest()[:8], "big"
        )
        permuted = _permuted_density_rows(rows, seed=seed)
        evaluation = _nested_evaluation(
            permuted,
            models=qualifying_models,
            config=config,
            fold_assignments=assignments,
        )
        fits += int(evaluation["compute"]["inner_fits"]) + int(
            evaluation["compute"]["final_fits"]
        )
        null_improvements.append(
            baseline_mse - _weighted_mse(rows, evaluation["qualifying_prediction"])
        )
    p_value = (1 + sum(value >= observed_improvement for value in null_improvements)) / (
        count + 1
    )
    return {
        "observed_mse_improvement": observed_improvement,
        "p_value": p_value,
        "permutations": count,
        "null_improvement_quantiles": {
            "q05": float(np.quantile(null_improvements, 0.05)),
            "q50": float(np.quantile(null_improvements, 0.50)),
            "q95": float(np.quantile(null_improvements, 0.95)),
        },
        "ridge_fits": fits,
    }


def _string_metrics(metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    return {
        group: {key: _metric(value) for key, value in values.items()}
        for group, values in metrics.items()
    }


def run_experiment(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = source.load_config(root)
    source_manifest = _load_source_manifest(root)
    primary, stellar = _load_rows(root)
    models = _models(config)
    primary_evaluation = _nested_evaluation(primary, models=models, config=config)
    primary_model_metrics = {
        model["id"]: _metrics_by_domain(
            primary, primary_evaluation["predictions"][str(model["id"])]
        )
        for model in models
    }
    selected_metrics = _metrics_by_domain(
        primary, primary_evaluation["selected_prediction"]
    )
    qualifying_metrics = _metrics_by_domain(
        primary, primary_evaluation["qualifying_prediction"]
    )

    galaxy_primary = [row for row in primary if row["domain"] == "galaxy"]
    robustness_rows = galaxy_primary + stellar
    for index, row in enumerate(robustness_rows):
        row["row_id"] = index
    robustness = _nested_evaluation(robustness_rows, models=models, config=config)
    robustness_selected_metrics = _metrics_by_domain(
        robustness_rows, robustness["selected_prediction"]
    )
    robustness_qualifying_metrics = _metrics_by_domain(
        robustness_rows, robustness["qualifying_prediction"]
    )
    robustness_baseline_metrics = _metrics_by_domain(
        robustness_rows, robustness["predictions"]["gbar_polynomial"]
    )

    permutation = _permutation_test(
        primary,
        observed=primary_evaluation,
        models=models,
        config=config,
    )
    admission = config["exploration_admission"]
    baseline = primary_model_metrics["gbar_polynomial"]
    selected_ids = [record["selected_model"] for record in primary_evaluation["fold_records"]]
    qualifying_count = sum(
        bool(record["selected_qualifying"])
        for record in primary_evaluation["fold_records"]
    )
    controls = [
        "gbar_plus_surface",
        "gbar_plus_volume",
        "gbar_plus_scale_contrast",
    ]
    qualifying_improvement_over_baseline = {
        group: float(qualifying_metrics[group]["r2"] - baseline[group]["r2"])
        for group in ("overall", "galaxy", "cluster")
    }
    robustness_improvement = float(
        robustness_qualifying_metrics["overall"]["r2"]
        - robustness_baseline_metrics["overall"]["r2"]
    )
    gates = {
        "quality_contract_passes": bool(source_manifest["extraction"]["quality_pass"]),
        "selected_model_qualifying_in_at_least_folds": qualifying_count
        >= int(admission["selected_model_qualifying_in_at_least_folds"]),
        "selected_density_model_r2_positive_in_each_domain": all(
            float(qualifying_metrics[group]["r2"]) > 0
            for group in ("galaxy", "cluster")
        ),
        "selected_density_model_improvement_over_gbar_baseline_positive_in_each_domain": all(
            qualifying_improvement_over_baseline[group] > 0
            for group in ("galaxy", "cluster")
        ),
        "selected_density_model_overall_r2_improvement_at_least": (
            qualifying_improvement_over_baseline["overall"]
        )
        >= float(admission["selected_density_model_overall_r2_improvement_at_least"]),
        "selected_density_model_beats_each_single_density_and_scale_control_overall": all(
            float(qualifying_metrics["overall"]["r2"])
            > float(primary_model_metrics[control]["overall"]["r2"])
            for control in controls
        ),
        "population_proxy_not_selected_in_any_fold": "gbar_plus_population_proxy"
        not in selected_ids,
        "object_stratified_permutation_p_at_most": float(permutation["p_value"])
        <= float(admission["object_stratified_permutation_p_at_most"]),
        "maximum_absolute_domain_median_residual_dex": all(
            abs(float(qualifying_metrics[group]["median_residual_dex"]))
            <= float(admission["maximum_absolute_domain_median_residual_dex"])
            for group in ("galaxy", "cluster")
        ),
        "stellar_profile_subset_robustness_does_not_reverse_improvement": robustness_improvement
        >= 0,
        "reserved_xcop_confirmation_profiles_untouched": source_manifest[
            "reserved_confirmation_profiles_opened"
        ]
        == 0,
    }
    passed = all(gates.values())
    decision = (
        "PASS_ITEM3_SMOOTH_DENSITY_EXPLORATION_AWAIT_EXPLICIT_CONFIRMATION"
        if passed
        else "REJECT_ITEM3_SMOOTH_DENSITY_CROSSOVER_EXPLORATION"
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-smooth-density-result-2.0",
        "goal": config["goal"],
        "item_number": 3,
        "attempt": 2,
        "decision": decision,
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity_label": config["scientific_contract"]["creativity_label"],
        "rewrite_exclusion": config["profile_conversion"]["rewrite_identity"],
        "freeze_commit": "68487eb9d9f00adfb1238c8f904b12e76b9bc9c2",
        "inputs": {
            "config_path": source.CONFIG_PATH,
            "config_sha256": _sha256_file(root / source.CONFIG_PATH),
            "sample_manifest_path": source.SAMPLE_MANIFEST_PATH,
            "sample_manifest_sha256": _sha256_file(root / source.SAMPLE_MANIFEST_PATH),
            "source_manifest_path": SOURCE_MANIFEST_PATH,
            "source_manifest_sha256": _sha256_file(root / SOURCE_MANIFEST_PATH),
            "source_manifest_content_sha256": source_manifest["content_sha256"],
            "radial_features_path": FEATURE_PATH,
            "radial_features_sha256": _sha256_file(root / FEATURE_PATH),
        },
        "data": {
            "primary_rows": len(primary),
            "stellar_robustness_rows": len(stellar),
            "primary_objects": len({str(row["object_key"]) for row in primary}),
            "valid_galaxies": source_manifest["extraction"]["valid_galaxies"],
            "valid_clusters": source_manifest["extraction"]["valid_clusters"],
            "quality_records": source_manifest["extraction"]["quality_records"],
            "confirmation_profiles_opened": 0,
        },
        "models": [
            {
                "id": model["id"],
                "features": list(model["features"]),
                "qualifying": bool(model["qualifying"]),
            }
            for model in models
        ],
        "primary": {
            "selected_model_counts": dict(sorted(Counter(selected_ids).items())),
            "qualifying_selector_counts": dict(
                sorted(
                    Counter(
                        record["qualifying_selector_model"]
                        for record in primary_evaluation["fold_records"]
                    ).items()
                )
            ),
            "folds": primary_evaluation["fold_records"],
            "model_metrics": {
                model_id: _string_metrics(metrics)
                for model_id, metrics in primary_model_metrics.items()
            },
            "nested_selected_metrics": _string_metrics(selected_metrics),
            "qualifying_selector_metrics": _string_metrics(qualifying_metrics),
            "qualifying_r2_improvement_over_gbar_baseline": {
                group: _metric(value)
                for group, value in qualifying_improvement_over_baseline.items()
            },
        },
        "stellar_baryon_robustness": {
            "objects": len({str(row["object_key"]) for row in stellar}),
            "selected_model_counts": dict(
                sorted(
                    Counter(
                        record["selected_model"]
                        for record in robustness["fold_records"]
                    ).items()
                )
            ),
            "selected_metrics": _string_metrics(robustness_selected_metrics),
            "qualifying_selector_metrics": _string_metrics(
                robustness_qualifying_metrics
            ),
            "gbar_baseline_metrics": _string_metrics(robustness_baseline_metrics),
            "overall_r2_improvement": _metric(robustness_improvement),
        },
        "permutation": {
            "observed_mse_improvement": _metric(
                permutation["observed_mse_improvement"]
            ),
            "p_value": _metric(permutation["p_value"]),
            "permutations": permutation["permutations"],
            "null_improvement_quantiles": {
                key: _metric(value)
                for key, value in permutation["null_improvement_quantiles"].items()
            },
        },
        "gates": gates,
        "gate_counts": {
            "passed": sum(gates.values()),
            "required": len(gates),
        },
        "compute": {
            "primary_inner_ridge_fits": primary_evaluation["compute"]["inner_fits"],
            "primary_final_ridge_fits": primary_evaluation["compute"]["final_fits"],
            "robustness_inner_ridge_fits": robustness["compute"]["inner_fits"],
            "robustness_final_ridge_fits": robustness["compute"]["final_fits"],
            "permutation_ridge_fits": permutation["ridge_fits"],
            "paid_model_calls": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "confirmation_accesses": 0,
        },
        "claim_boundaries": config["claim_boundaries"],
        "next_action": (
            "Do not open confirmation. Request explicit approval for the frozen four-cluster X-COP confirmation and freeze an independent LITTLE THINGS galaxy confirmation."
            if passed
            else "Retain the failed smooth local density-crossover family as an Item 3 counterexample. Decide whether a materially different nonlocal density operator remains before closing Item 3 and advancing to Item 4."
        ),
        "content_sha256": None,
    }
    content = dict(receipt)
    content.pop("content_sha256")
    receipt["content_sha256"] = canonical_sha256(content)
    return receipt


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    path = root / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(run_experiment(root)) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    path = root / OUTPUT_PATH
    existing = json.loads(path.read_text(encoding="utf-8"))
    rebuilt = run_experiment(root)
    if existing != rebuilt:
        raise GravityItem3SmoothExperimentError("Item 3 attempt-2 receipt drifted")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.check:
        check_receipt(args.root)
    else:
        print(write_receipt(args.root))


if __name__ == "__main__":
    main()
