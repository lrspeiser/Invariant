"""Nested held-out experiment for gravity-roadmap Item 2 AXES group attempt 5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import spearmanr

from . import gravity_item2_axes_group_geometry as source
from .sigma_core import canonical_json_bytes, canonical_sha256

OUTPUT_PATH = "runs/gravity/roadmap/item-02-axes-group-geometry-v5.json"

FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "luminosity_size": ("log10_member_luminosity", "log10_rms_radius_kpc"),
    "richness_redshift_environment": (
        "log10_richness",
        "metadata_redshift",
        "d10",
    ),
    "global_shape": (
        "projected_axis_ratio",
        "projected_ellipticity",
        "quadrupole",
        "m3",
        "m4",
    ),
    "radial_nonlocal_shape": (
        "inner_outer_axis_ratio_difference",
        "inner_outer_quadrupole_difference",
        "centroid_shift_profile",
        "axis_twist_inner_outer_sin2",
        "outer_multipole_energy",
    ),
    "graph_filamentarity": (
        "mst_length_per_rms_radius",
        "mst_diameter_efficiency",
        "projected_linearity",
        "angular_gap_entropy",
    ),
}
FEATURE_BLOCKS["all_geometry"] = tuple(
    dict.fromkeys(
        FEATURE_BLOCKS["global_shape"]
        + FEATURE_BLOCKS["radial_nonlocal_shape"]
        + FEATURE_BLOCKS["graph_filamentarity"]
    )
)


class GravityItem2AxesGroupExperimentError(RuntimeError):
    """Raised when the frozen group experiment or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem2AxesGroupExperimentError("non-finite receipt metric")
    return f"{float(value):.12e}"


def _load_feature_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / config["feature_output"]
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    integer_fields = {"group", "richness_bin", "members", "unique_member_redshifts"}
    rows: list[dict[str, Any]] = []
    for input_row in raw:
        row = {
            key: int(value) if key in integer_fields else float(value)
            for key, value in input_row.items()
        }
        rows.append(row)
    if len(rows) != len({row["group"] for row in rows}):
        raise GravityItem2AxesGroupExperimentError("duplicate group in feature table")
    if any(
        not math.isfinite(float(value))
        for row in rows
        for value in row.values()
    ):
        raise GravityItem2AxesGroupExperimentError("non-finite feature-table value")
    return sorted(rows, key=lambda row: int(row["group"]))


def _features_for_model(model: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for block in model["blocks"]:
        if block not in FEATURE_BLOCKS:
            raise GravityItem2AxesGroupExperimentError(f"unknown feature block: {block}")
        values.extend(FEATURE_BLOCKS[str(block)])
    if len(values) != len(set(values)):
        raise GravityItem2AxesGroupExperimentError(
            f"duplicate feature in model {model['id']}"
        )
    return tuple(values)


def fold_assignments(
    rows: Sequence[Mapping[str, Any]], *, salt: str, folds: int
) -> dict[int, int]:
    strata: defaultdict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[int(row["richness_bin"])].append(row)
    assignments: dict[int, int] = {}
    for richness_bin, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{salt}|{richness_bin}|{row['group']}".encode()
            ).hexdigest(),
        )
        for ordinal, row in enumerate(ordered):
            assignments[int(row["group"])] = ordinal % folds
    if len(assignments) != len(rows):
        raise GravityItem2AxesGroupExperimentError("incomplete fold assignment")
    return assignments


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    matrix = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(matrix)):
        raise GravityItem2AxesGroupExperimentError("non-finite model matrix")
    return matrix


def _fit_ridge(
    rows: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> dict[str, Any]:
    y = np.asarray([float(row[response]) for row in rows], dtype=np.float64)
    raw = _matrix(rows, features)
    if raw.shape[1]:
        means = np.mean(raw, axis=0)
        scales = np.std(raw, axis=0)
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        standardized = (raw - means) / scales
    else:
        means = np.empty(0, dtype=np.float64)
        scales = np.empty(0, dtype=np.float64)
        standardized = raw
    design = np.column_stack((np.ones(len(rows)), standardized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return {
        "coefficients": coefficients,
        "features": tuple(features),
        "means": means,
        "scales": scales,
    }


def _predict(fit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    raw = _matrix(rows, fit["features"])
    if raw.shape[1]:
        standardized = (raw - np.asarray(fit["means"])) / np.asarray(fit["scales"])
    else:
        standardized = raw
    design = np.column_stack((np.ones(len(rows)), standardized))
    return design @ np.asarray(fit["coefficients"])


def _inner_score(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[int, int],
    *,
    excluded_outer_fold: int,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> tuple[float, int]:
    losses: list[float] = []
    fits = 0
    folds = sorted(set(assignments.values()) - {excluded_outer_fold})
    for validation_fold in folds:
        training = [
            row
            for row in rows
            if assignments[int(row["group"])] not in {excluded_outer_fold, validation_fold}
        ]
        validation = [
            row
            for row in rows
            if assignments[int(row["group"])] == validation_fold
        ]
        fit = _fit_ridge(training, features=features, response=response, alpha=alpha)
        observed = np.asarray([float(row[response]) for row in validation])
        losses.extend((observed - _predict(fit, validation)) ** 2)
        fits += 1
    return float(np.mean(losses)), fits


def _metrics(
    rows: Sequence[Mapping[str, Any]], predicted: Mapping[int, float], response: str
) -> dict[str, Any]:
    def calculate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        observed = np.asarray([float(row[response]) for row in subset], dtype=np.float64)
        values = np.asarray([float(predicted[int(row["group"])]) for row in subset])
        residual = observed - values
        denominator = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
        correlation = spearmanr(observed, values).statistic
        return {
            "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
            "mean_squared_error": _metric(float(np.mean(residual**2))),
            "objects": len(subset),
            "r2": _metric(r2),
            "spearman": _metric(float(correlation)),
        }

    return {
        "overall": calculate(rows),
        "by_richness_bin": {
            str(richness_bin): calculate(
                [row for row in rows if int(row["richness_bin"]) == richness_bin]
            )
            for richness_bin in range(3)
        },
    }


def _evaluate_response(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str,
    selection_pool: str = "all",
) -> dict[str, Any]:
    cv = config["cross_validation"]
    folds = int(cv["outer_folds"])
    assignments = fold_assignments(rows, salt=str(cv["fold_salt"]), folds=folds)
    models = list(config["model_families"])
    penalties = [float(value) for value in cv["ridge_penalties"]]
    predictions_by_model: dict[str, dict[int, float]] = {
        str(model["id"]): {} for model in models
    }
    selected_predictions: dict[int, float] = {}
    ablation_predictions: dict[int, float] = {}
    fold_ledger: list[dict[str, Any]] = []
    inner_fit_count = 0
    final_fit_count = 0
    for outer_fold in range(folds):
        training = [row for row in rows if assignments[int(row["group"])] != outer_fold]
        heldout = [row for row in rows if assignments[int(row["group"])] == outer_fold]
        candidates: list[dict[str, Any]] = []
        for model_order, model in enumerate(models):
            features = _features_for_model(model)
            model_penalties = penalties if features else penalties[:1]
            model_candidates: list[dict[str, Any]] = []
            for alpha in model_penalties:
                score, fits = _inner_score(
                    rows,
                    assignments,
                    excluded_outer_fold=outer_fold,
                    features=features,
                    response=response,
                    alpha=alpha,
                )
                inner_fit_count += fits
                candidate = {
                    "alpha": alpha,
                    "features": features,
                    "inner_mse": score,
                    "model": model,
                    "model_order": model_order,
                }
                model_candidates.append(candidate)
                candidates.append(candidate)
            best_model_candidate = min(
                model_candidates, key=lambda row: (row["inner_mse"], row["alpha"])
            )
            fit = _fit_ridge(
                training,
                features=features,
                response=response,
                alpha=float(best_model_candidate["alpha"]),
            )
            final_fit_count += 1
            for object_row, prediction in zip(heldout, _predict(fit, heldout), strict=True):
                predictions_by_model[str(model["id"])][int(object_row["group"])] = float(
                    prediction
                )
        selectable = (
            [row for row in candidates if bool(row["model"]["qualifying"])]
            if selection_pool == "qualifying"
            else candidates
        )
        selected = min(
            selectable,
            key=lambda row: (
                row["inner_mse"],
                len(row["features"]),
                row["model_order"],
                row["alpha"],
            ),
        )
        selected_fit = _fit_ridge(
            training,
            features=selected["features"],
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fit_count += 1
        ablation_model = next(
            model for model in models if model["id"] == "richness_environment_nuisance"
        )
        ablation_features = _features_for_model(ablation_model)
        ablation_fit = _fit_ridge(
            training,
            features=ablation_features,
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fit_count += 1
        for object_row, selected_value, ablation_value in zip(
            heldout,
            _predict(selected_fit, heldout),
            _predict(ablation_fit, heldout),
            strict=True,
        ):
            key = int(object_row["group"])
            selected_predictions[key] = float(selected_value)
            ablation_predictions[key] = float(ablation_value)
        fold_ledger.append(
            {
                "alpha": _metric(float(selected["alpha"])),
                "features": list(selected["features"]),
                "fold": outer_fold,
                "heldout_groups": len(heldout),
                "inner_mse": _metric(float(selected["inner_mse"])),
                "model_id": str(selected["model"]["id"]),
                "qualifying": bool(selected["model"]["qualifying"]),
            }
        )
    return {
        "ablation_metrics": _metrics(rows, ablation_predictions, response),
        "compute_counts": {
            "final_ridge_fits": final_fit_count,
            "inner_ridge_fits": inner_fit_count,
        },
        "fold_assignments": assignments,
        "fold_ledger": fold_ledger,
        "model_metrics": {
            model_id: _metrics(rows, predictions, response)
            for model_id, predictions in predictions_by_model.items()
        },
        "predictions": selected_predictions,
        "selected_metrics": _metrics(rows, selected_predictions, response),
    }


def _evaluate_fixed_selection(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str,
    selection: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    cv = config["cross_validation"]
    assignments = fold_assignments(
        rows, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"])
    )
    models = {str(model["id"]): model for model in config["model_families"]}
    predictions: dict[int, float] = {}
    baseline_predictions: dict[int, float] = {}
    baseline = models["richness_environment_nuisance"]
    for fold in range(int(cv["outer_folds"])):
        training = [row for row in rows if assignments[int(row["group"])] != fold]
        heldout = [row for row in rows if assignments[int(row["group"])] == fold]
        chosen = selection[fold]
        model = models[str(chosen["model_id"])]
        alpha = float(chosen["alpha"])
        fit = _fit_ridge(
            training,
            features=_features_for_model(model),
            response=response,
            alpha=alpha,
        )
        baseline_fit = _fit_ridge(
            training,
            features=_features_for_model(baseline),
            response=response,
            alpha=alpha,
        )
        for row, value, baseline_value in zip(
            heldout, _predict(fit, heldout), _predict(baseline_fit, heldout), strict=True
        ):
            predictions[int(row["group"])] = float(value)
            baseline_predictions[int(row["group"])] = float(baseline_value)
    metrics = _metrics(rows, predictions, response)
    baseline_metrics = _metrics(rows, baseline_predictions, response)
    return {
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "passes": (
            float(metrics["overall"]["r2"]) > 0
            and float(metrics["overall"]["spearman"]) > 0
            and float(metrics["overall"]["mean_squared_error"])
            < float(baseline_metrics["overall"]["mean_squared_error"])
        ),
    }


def _permutation_test(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    response = "log10_eta_lum"
    qualifying = _evaluate_response(rows, config, response=response, selection_pool="qualifying")
    baseline_mse = float(
        qualifying["model_metrics"]["richness_environment_nuisance"]["overall"][
            "mean_squared_error"
        ]
    )
    observed_mse = float(qualifying["selected_metrics"]["overall"]["mean_squared_error"])
    observed_improvement = baseline_mse - observed_mse
    count = int(config["cross_validation"]["stratified_permutation_count"])
    salt = str(config["cross_validation"]["permutation_salt"])
    null_improvements: list[float] = []
    for permutation in range(count):
        permuted = [dict(row) for row in rows]
        by_bin: defaultdict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(permuted):
            by_bin[int(row["richness_bin"])].append(index)
        for richness_bin, indices in by_bin.items():
            seed = int.from_bytes(
                hashlib.sha256(f"{salt}|{permutation}|{richness_bin}".encode()).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            values = np.asarray([permuted[index][response] for index in indices])
            shuffled = rng.permutation(values)
            for index, value in zip(indices, shuffled, strict=True):
                permuted[index][response] = float(value)
        result = _evaluate_response(
            permuted, config, response=response, selection_pool="qualifying"
        )
        null_baseline = float(
            result["model_metrics"]["richness_environment_nuisance"]["overall"][
                "mean_squared_error"
            ]
        )
        null_selected = float(result["selected_metrics"]["overall"]["mean_squared_error"])
        null_improvements.append(null_baseline - null_selected)
    exceedances = sum(value >= observed_improvement for value in null_improvements)
    p_value = (1 + exceedances) / (1 + count)
    return {
        "null_improvement_max": _metric(max(null_improvements)),
        "null_improvement_median": _metric(float(np.median(null_improvements))),
        "observed_improvement": _metric(observed_improvement),
        "permutations": count,
        "p_value": _metric(p_value),
        "selection_pool": "qualifying geometry families only",
        "statistic": "richness_environment_baseline_MSE-minus-nested_selected_geometry_MSE",
    }


def _gate_checks(
    primary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    extraction: Mapping[str, Any],
    permutation: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, bool]:
    selected = primary["selected_metrics"]
    model_metrics = primary["model_metrics"]
    luminosity_size = model_metrics["luminosity_size_nuisance"]
    richness_environment = model_metrics["richness_environment_nuisance"]
    ablation = primary["ablation_metrics"]
    return {
        "all_180_exploration_groups_pass_frozen_quality": (
            extraction["decision"] == "PASS_EXPLORATION_REPRESENTATION_QUALITY"
            and int(extraction["counts"]["quality_passing"]) == 180
        ),
        "selected_model_qualifying_in_every_outer_fold": all(
            bool(row["qualifying"]) for row in primary["fold_ledger"]
        ),
        "heldout_r2_positive_overall": float(selected["overall"]["r2"]) > 0,
        "heldout_r2_positive_in_every_richness_bin": all(
            float(selected["by_richness_bin"][str(value)]["r2"]) > 0
            for value in range(3)
        ),
        "selected_model_beats_luminosity_size_nuisance_mse_overall": (
            float(selected["overall"]["mean_squared_error"])
            < float(luminosity_size["overall"]["mean_squared_error"])
        ),
        "selected_model_beats_richness_environment_nuisance_mse_overall": (
            float(selected["overall"]["mean_squared_error"])
            < float(richness_environment["overall"]["mean_squared_error"])
        ),
        "geometry_ablation_improvement_positive_in_every_richness_bin": all(
            float(selected["by_richness_bin"][str(value)]["mean_squared_error"])
            < float(ablation["by_richness_bin"][str(value)]["mean_squared_error"])
            for value in range(3)
        ),
        "stratified_permutation_p_at_most_frozen_threshold": (
            float(permutation["p_value"])
            <= float(config["exploration_admission"]["stratified_permutation_p_must_be_at_most"])
        ),
        "all_response_robustness_controls_pass": all(
            bool(value["passes"]) for value in robustness.values()
        ),
        "reserved_confirmation_untouched": (
            int(extraction["counts"]["reserved_confirmation_target_accesses"]) == 0
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = source.load_config(root)
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    source.validate_sample_manifest(sample, config=config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source.validate_source_manifest(source_manifest, config=config, sample=sample)
    extraction_path = root / config["extraction_summary_output"]
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction_copy = dict(extraction)
    extraction_digest = extraction_copy.pop("content_sha256", None)
    if extraction_digest != canonical_sha256(extraction_copy):
        raise GravityItem2AxesGroupExperimentError("extraction summary hash changed")
    rows = _load_feature_rows(root, config)
    if len(rows) != int(extraction["counts"]["quality_passing"]):
        raise GravityItem2AxesGroupExperimentError("feature/extraction count mismatch")
    primary = _evaluate_response(rows, config, response="log10_eta_lum")
    frozen_selection = {
        int(row["fold"]): {"alpha": row["alpha"], "model_id": row["model_id"]}
        for row in primary["fold_ledger"]
    }
    robustness = {
        response: _evaluate_fixed_selection(
            rows, config, response=response, selection=frozen_selection
        )
        for response in (
            "log10_sigma_gap",
            "log10_eta_half_radius",
            "log10_eta_mad",
        )
    }
    permutation = _permutation_test(rows, config)
    gates = _gate_checks(primary, robustness, extraction, permutation, config)
    decision = (
        "PASS_ITEM2_AXES_GROUP_GEOMETRY_EXPLORATION_REQUIRES_AUTHORIZATION"
        if all(gates.values())
        else (
            "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY_QUALITY_GATE"
            if not gates["all_180_exploration_groups_pass_frozen_quality"]
            else "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY"
        )
    )
    models = list(config["model_families"])
    candidate_cells = sum(
        len(config["cross_validation"]["ridge_penalties"])
        if _features_for_model(model)
        else 1
        for model in models
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item2-axes-group-geometry-receipt-1.0",
        "goal": config["goal"],
        "decision": decision,
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity": {
            "label": config["scientific_contract"]["creativity_label"],
            "known_components": config["scientific_contract"]["known_components"],
            "historical_novelty_established": False,
        },
        "preregistration": {
            "git_commit": source.FREEZE_COMMIT,
            "selected_member_rows_opened_before_commit": 0,
        },
        "data_lineage": {
            "catalog": config["catalog_sources"]["catalog_id"],
            "baryonic_input": "projected positions and r-band luminosities of cleaned group members",
            "response": "line-of-sight dispersion recomputed from exploration member redshifts",
            "published_group_velocity_columns_read": 0,
            "published_r200_read": 0,
            "xray_target_columns_read": 0,
            "lensing_target_used": False,
            "dark_halo_target_used": False,
        },
        "response": {
            "equation": config["response"]["primary_equation"],
            "interpretation": config["response"]["response_interpretation"],
            "primary": {
                "selected_metrics": primary["selected_metrics"],
                "ablation_metrics": primary["ablation_metrics"],
                "model_metrics": primary["model_metrics"],
                "fold_ledger": primary["fold_ledger"],
            },
            "robustness": robustness,
            "permutation_test": permutation,
        },
        "gate_checks": gates,
        "counterexamples": {
            "quality_failures": extraction["failures"],
            "failed_gate_names": [name for name, passed in gates.items() if not passed],
            "selected_nonqualifying_folds": [
                int(row["fold"]) for row in primary["fold_ledger"] if not row["qualifying"]
            ],
        },
        "counts": {
            "candidate_model_families": len(models),
            "candidate_model_ridge_cells": candidate_cells,
            "equivalence_classes": len(models),
            "exploration_selected": 180,
            "exploration_quality_passing": len(rows),
            "exploration_quality_failures": len(extraction["failures"]),
            "reserved_confirmation_groups": 90,
            "reserved_confirmation_target_accesses": 0,
            "outer_folds": int(config["cross_validation"]["outer_folds"]),
            "inner_ridge_fits": int(primary["compute_counts"]["inner_ridge_fits"]),
            "final_ridge_fits": int(primary["compute_counts"]["final_ridge_fits"])
            + 2 * len(robustness) * int(config["cross_validation"]["outer_folds"]),
            "stratified_permutations": int(permutation["permutations"]),
            "paid_model_calls": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "sparc_confirmation_evaluator_accesses": 0,
        },
        "limitations": config["provenance_limitations"],
        "claims": {
            "alternative_to_gr_established": False,
            "complete_baryonic_mass_used": False,
            "cross_scale_beta_explained": False,
            "direct_lensing_test_completed": False,
            "group_finder_independence_established": False,
            "historical_novelty_established": False,
            "reserved_confirmation_opened": False,
            "roadmap_item_2_complete": False,
            "sequential_G6_G7_G8_advanced": False,
        },
        "next_action": (
            "Retain all representation failures and held-out results. Do not open the 90 confirmation groups. Decide whether Item 2 is now sufficiently rejected for the tested projected-shape families and advance to Item 3 surface-versus-volume density, or preregister a genuinely filamentary response with a different source."
        ),
        "source_bindings": {
            "config": {"path": source.CONFIG_PATH, "sha256": _sha256_file(root / source.CONFIG_PATH)},
            "sample_manifest": {
                "path": config["sample_manifest_output"],
                "sha256": _sha256_file(sample_path),
            },
            "source_manifest": {
                "path": config["source_manifest_output"],
                "sha256": _sha256_file(source_path),
            },
            "extraction_summary": {
                "path": config["extraction_summary_output"],
                "sha256": _sha256_file(extraction_path),
            },
            "feature_table": {
                "path": config["feature_output"],
                "sha256": _sha256_file(root / config["feature_output"]),
            },
            "source_code": {
                "path": str(Path(source.__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(source.__file__)),
            },
            "experiment_code": {
                "path": str(Path(__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(__file__)),
            },
            "test": {
                "path": "tests/test_gravity_item2_axes_group_geometry.py",
                "sha256": _sha256_file(root / "tests/test_gravity_item2_axes_group_geometry.py"),
            },
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2AxesGroupExperimentError("receipt content hash changed")
    if value.get("decision") not in {
        "PASS_ITEM2_AXES_GROUP_GEOMETRY_EXPLORATION_REQUIRES_AUTHORIZATION",
        "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY_QUALITY_GATE",
        "INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY",
    }:
        raise GravityItem2AxesGroupExperimentError("unknown experiment decision")
    if int(value["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem2AxesGroupExperimentError("confirmation was accessed")
    if any(bool(claim) for claim in value["claims"].values()):
        raise GravityItem2AxesGroupExperimentError("receipt contains an overclaim")
    if str(value["decision"]).startswith("PASS_") and not all(
        bool(passed) for passed in value["gate_checks"].values()
    ):
        raise GravityItem2AxesGroupExperimentError("PASS decision has a failed gate")
    for binding in value["source_bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = root / path
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem2AxesGroupExperimentError(f"source binding changed: {path}")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / OUTPUT_PATH
    path.write_bytes(canonical_json_bytes(receipt))
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.check:
        path = root / OUTPUT_PATH
        stored = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(stored, root=root)
        if build_receipt(root) != stored:
            raise GravityItem2AxesGroupExperimentError("receipt is not an exact rebuild")
        return 0
    path = write_receipt(root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
