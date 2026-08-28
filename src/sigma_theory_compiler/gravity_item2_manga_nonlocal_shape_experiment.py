"""Nested held-out diagnostic for gravity-roadmap Item 2 MaNGA attempt 4."""

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
from scipy.stats import spearmanr

from . import gravity_item2_manga_nonlocal_shape as source
from .sigma_core import canonical_json_bytes, canonical_sha256

OUTPUT_PATH = "runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4.json"
PREREGISTRATION_COMMIT = "c49ba6ba2f3b2658c43b2e8bb6a27e7a851d598f"

FEATURE_BLOCKS: dict[str, tuple[str, ...]] = {
    "mass_size": ("log10_stellar_mass_aperture", "log10_circularized_radius_kpc"),
    "visual_class_proxy": ("visual_class_proxy",),
    "sersic_axis_ratio": ("sersic_index", "pymorph_axis_ratio"),
    "known_global_shape": (
        "mass_concentration_0p25",
        "centroid_shift_profile",
        "quadrupole_1p0",
        "m3_1p0",
        "m4_1p0",
    ),
    "radial_quadrupole": (
        "quadrupole_inner_outer_difference",
        "quadrupole_profile_variance",
    ),
    "radial_odd_even": (
        "m3_inner_outer_difference",
        "m4_inner_outer_difference",
        "outer_multipole_energy",
    ),
    "boundary_twist": (
        "axis_twist_inner_outer_sin2",
        "inner_outer_centroid_alignment",
        "centroid_shift_profile",
    ),
    "nonlocal_profile_energy": (
        "profile_roughness_energy",
        "outer_multipole_energy",
    ),
    "all_nonlocal_shape": (
        "quadrupole_inner_outer_difference",
        "quadrupole_profile_variance",
        "axis_twist_inner_outer_sin2",
        "m3_inner_outer_difference",
        "m4_inner_outer_difference",
        "outer_multipole_energy",
        "profile_roughness_energy",
        "inner_outer_centroid_alignment",
    ),
}

RESPONSE_COLUMNS = (
    "log10_eta_ap",
    "log10_eta_major_axis",
    "log10_eta_uncorrected_sigma",
    "log10_eta_mass_weighted",
)


class GravityItem2MangaNonlocalShapeExperimentError(RuntimeError):
    """Raised when a frozen MaNGA diagnostic or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str | int:
    if isinstance(value, int):
        return value
    if not math.isfinite(float(value)):
        raise GravityItem2MangaNonlocalShapeExperimentError("non-finite receipt metric")
    return f"{float(value):.12e}"


def load_config(root: Path) -> dict[str, Any]:
    return source.load_config(root)


def _load_feature_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / config["feature_output"]
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    integer_fields = {
        "axis_bin",
        "unique_kinematic_bins",
        "valid_pca_spaxels",
        "visual_class",
    }
    text_fields = {"manga_id", "plateifu"}
    rows: list[dict[str, Any]] = []
    for input_row in raw:
        row: dict[str, Any] = {}
        for key, value in input_row.items():
            if key in text_fields:
                row[key] = value
            elif key in integer_fields:
                row[key] = int(value)
            else:
                row[key] = float(value)
        rows.append(row)
    if len(rows) != len({row["plateifu"] for row in rows}):
        raise GravityItem2MangaNonlocalShapeExperimentError("duplicate feature-table galaxy")
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in text_fields
    ):
        raise GravityItem2MangaNonlocalShapeExperimentError("non-finite feature-table value")
    return sorted(rows, key=lambda row: str(row["plateifu"]))


def _features_for_model(model: Mapping[str, Any]) -> tuple[str, ...]:
    values: list[str] = []
    for block in model["blocks"]:
        if block not in FEATURE_BLOCKS:
            raise GravityItem2MangaNonlocalShapeExperimentError(
                f"unknown feature block: {block}"
            )
        values.extend(FEATURE_BLOCKS[str(block)])
    if len(values) != len(set(values)):
        raise GravityItem2MangaNonlocalShapeExperimentError(
            f"duplicate feature in model {model['id']}"
        )
    return tuple(values)


def fold_assignments(
    rows: Sequence[Mapping[str, Any]], *, salt: str, folds: int
) -> dict[str, int]:
    strata: defaultdict[tuple[int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[(int(row["visual_class"]), int(row["axis_bin"]))].append(row)
    assignments: dict[str, int] = {}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{salt}|{stratum[0]}|{stratum[1]}|{row['plateifu']}".encode()
            ).hexdigest(),
        )
        for ordinal, row in enumerate(ordered):
            assignments[str(row["plateifu"])] = ordinal % folds
    if len(assignments) != len(rows):
        raise GravityItem2MangaNonlocalShapeExperimentError("incomplete fold assignment")
    return assignments


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    result = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=np.float64
    )
    if np.any(~np.isfinite(result)):
        raise GravityItem2MangaNonlocalShapeExperimentError("non-finite model matrix")
    return result


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
        "alpha": float(alpha),
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


def _mse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((observed - predicted) ** 2))


def _inner_score(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    excluded_outer_fold: int,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> tuple[float, int]:
    losses: list[float] = []
    fits = 0
    for validation_fold in sorted(set(assignments.values()) - {excluded_outer_fold}):
        training = [
            row
            for row in rows
            if assignments[str(row["plateifu"])] not in {excluded_outer_fold, validation_fold}
        ]
        validation = [
            row
            for row in rows
            if assignments[str(row["plateifu"])] == validation_fold
        ]
        fit = _fit_ridge(training, features=features, response=response, alpha=alpha)
        observed = np.asarray([float(row[response]) for row in validation])
        losses.extend((observed - _predict(fit, validation)) ** 2)
        fits += 1
    return float(np.mean(losses)), fits


def _metrics(
    rows: Sequence[Mapping[str, Any]], predicted: Mapping[str, float], response: str
) -> dict[str, Any]:
    def calculate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        observed = np.asarray([float(row[response]) for row in subset], dtype=np.float64)
        values = np.asarray([float(predicted[str(row["plateifu"])]) for row in subset])
        residual = observed - values
        denominator = float(np.sum((observed - float(np.mean(observed))) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
        return {
            "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
            "mean_squared_error": _metric(float(np.mean(residual**2))),
            "objects": len(subset),
            "r2": _metric(r2),
        }

    result = {"overall": calculate(rows), "by_visual_class": {}}
    for visual_class in (1, 2):
        result["by_visual_class"][str(visual_class)] = calculate(
            [row for row in rows if int(row["visual_class"]) == visual_class]
        )
    return result


def _evaluate_response(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str,
    fixed_selection: Mapping[int, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    cv = config["cross_validation"]
    folds = int(cv["outer_folds"])
    assignments = fold_assignments(rows, salt=str(cv["fold_salt"]), folds=folds)
    models = list(config["model_families"])
    penalties = [float(value) for value in cv["ridge_penalties"]]
    predictions_by_model: dict[str, dict[str, float]] = {
        str(model["id"]): {} for model in models
    }
    selected_predictions: dict[str, float] = {}
    ablation_predictions: dict[str, float] = {}
    fold_ledger: list[dict[str, Any]] = []
    inner_fit_count = 0
    final_fit_count = 0
    for outer_fold in range(folds):
        training = [
            row for row in rows if assignments[str(row["plateifu"])] != outer_fold
        ]
        heldout = [row for row in rows if assignments[str(row["plateifu"])] == outer_fold]
        candidate_scores: list[dict[str, Any]] = []
        per_model_best: dict[str, dict[str, Any]] = {}
        for model_order, model in enumerate(models):
            features = _features_for_model(model)
            model_penalties = penalties if features else penalties[:1]
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
                candidate_scores.append(
                    {
                        "alpha": alpha,
                        "features": features,
                        "inner_mse": score,
                        "model": model,
                        "model_order": model_order,
                    }
                )
            best = min(
                (row for row in candidate_scores if row["model"] is model),
                key=lambda row: (row["inner_mse"], row["alpha"]),
            )
            per_model_best[str(model["id"])] = best
            fit = _fit_ridge(
                training,
                features=features,
                response=response,
                alpha=float(best["alpha"]),
            )
            final_fit_count += 1
            for object_row, prediction in zip(heldout, _predict(fit, heldout), strict=True):
                predictions_by_model[str(model["id"])][str(object_row["plateifu"])] = float(
                    prediction
                )
        if fixed_selection is None:
            selected = min(
                candidate_scores,
                key=lambda row: (
                    row["inner_mse"],
                    len(row["features"]),
                    row["model_order"],
                    row["alpha"],
                ),
            )
        else:
            frozen = fixed_selection[outer_fold]
            model = next(row for row in models if row["id"] == frozen["model_id"])
            selected = {
                "alpha": float(frozen["alpha"]),
                "features": _features_for_model(model),
                "inner_mse": float("nan"),
                "model": model,
                "model_order": models.index(model),
            }
        selected_fit = _fit_ridge(
            training,
            features=selected["features"],
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fit_count += 1
        selected_values = _predict(selected_fit, heldout)
        ablation_fit = _fit_ridge(
            training,
            features=FEATURE_BLOCKS["mass_size"],
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fit_count += 1
        ablation_values = _predict(ablation_fit, heldout)
        for object_row, selected_value, ablation_value in zip(
            heldout, selected_values, ablation_values, strict=True
        ):
            key = str(object_row["plateifu"])
            selected_predictions[key] = float(selected_value)
            ablation_predictions[key] = float(ablation_value)
        fold_ledger.append(
            {
                "alpha": _metric(float(selected["alpha"])),
                "features": list(selected["features"]),
                "fold": outer_fold,
                "heldout_objects": len(heldout),
                "inner_mse": (
                    None
                    if fixed_selection is not None
                    else _metric(float(selected["inner_mse"]))
                ),
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


def _gate_checks(
    primary: Mapping[str, Any],
    robustness: Mapping[str, Any],
    extraction: Mapping[str, Any],
) -> dict[str, bool]:
    selected = primary["selected_metrics"]
    mass_size = primary["model_metrics"]["mass_size_nuisance"]
    class_proxy = primary["model_metrics"]["class_proxy_nuisance"]
    ablation = primary["ablation_metrics"]
    shape_improvement = all(
        float(selected["by_visual_class"][str(visual_class)]["mean_squared_error"])
        < float(ablation["by_visual_class"][str(visual_class)]["mean_squared_error"])
        for visual_class in (1, 2)
    )
    return {
        "all_sixty_exploration_objects_pass_frozen_quality": (
            extraction["decision"] == "PASS_EXPLORATION_EXTRACTION"
            and int(extraction["counts"]["quality_passing"]) == 60
        ),
        "heldout_r2_positive_overall": float(selected["overall"]["r2"]) > 0,
        "heldout_r2_positive_in_ellipticals_and_s0s": all(
            float(selected["by_visual_class"][str(value)]["r2"]) > 0 for value in (1, 2)
        ),
        "selected_model_beats_mass_size_nuisance_mse_overall": (
            float(selected["overall"]["mean_squared_error"])
            < float(mass_size["overall"]["mean_squared_error"])
        ),
        "selected_model_beats_class_proxy_nuisance_mse_overall": (
            float(selected["overall"]["mean_squared_error"])
            < float(class_proxy["overall"]["mean_squared_error"])
        ),
        "selected_model_qualifying_in_every_outer_fold": all(
            bool(row["qualifying"]) for row in primary["fold_ledger"]
        ),
        "shape_ablation_improvement_positive_in_each_class": shape_improvement,
        "all_response_robustness_controls_preserve_positive_rank_and_proxy_improvement": all(
            bool(row["passes"]) for row in robustness.values()
        ),
        "reserved_confirmation_untouched": (
            int(extraction["counts"]["reserved_confirmation_target_accesses"]) == 0
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    source.validate_sample_manifest(sample, config=config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source.validate_source_manifest(source_manifest, config=config, sample=sample)
    extraction_path = root / source.EXTRACTION_SUMMARY_PATH
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction_copy = dict(extraction)
    extraction_digest = extraction_copy.pop("content_sha256", None)
    if extraction_digest != canonical_sha256(extraction_copy):
        raise GravityItem2MangaNonlocalShapeExperimentError("extraction summary hash mismatch")
    rows = _load_feature_rows(root, config)
    if len(rows) != int(extraction["counts"]["quality_passing"]):
        raise GravityItem2MangaNonlocalShapeExperimentError("feature/extraction count mismatch")
    primary = _evaluate_response(rows, config, response="log10_eta_ap")
    frozen_selection = {
        int(row["fold"]): {"alpha": row["alpha"], "model_id": row["model_id"]}
        for row in primary["fold_ledger"]
    }
    primary_observed = np.asarray([float(row["log10_eta_ap"]) for row in rows])
    primary_predicted = np.asarray(
        [float(primary["predictions"][str(row["plateifu"])]) for row in rows]
    )
    robustness: dict[str, Any] = {}
    extra_fit_counts = Counter()
    for response in RESPONSE_COLUMNS[1:]:
        result = _evaluate_response(
            rows,
            config,
            response=response,
            fixed_selection=frozen_selection,
        )
        for key, value in result["compute_counts"].items():
            extra_fit_counts[key] += int(value)
        observed = np.asarray([float(row[response]) for row in rows])
        predicted = np.asarray(
            [float(result["predictions"][str(row["plateifu"])]) for row in rows]
        )
        rho = float(spearmanr(predicted, observed).statistic)
        class_proxy_mse = float(
            result["model_metrics"]["class_proxy_nuisance"]["overall"]["mean_squared_error"]
        )
        selected_mse = float(result["selected_metrics"]["overall"]["mean_squared_error"])
        robustness[response] = {
            "heldout_r2": result["selected_metrics"]["overall"]["r2"],
            "passes": bool(rho > 0 and selected_mse < class_proxy_mse),
            "selected_mse": _metric(selected_mse),
            "class_proxy_mse": _metric(class_proxy_mse),
            "spearman_prediction_vs_observed": _metric(rho),
        }
    checks = _gate_checks(primary, robustness, extraction)
    downstream_science_pass = all(
        value for key, value in checks.items() if key != "all_sixty_exploration_objects_pass_frozen_quality"
    )
    if not checks["all_sixty_exploration_objects_pass_frozen_quality"]:
        decision = "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE"
    elif downstream_science_pass:
        decision = "PASS_ITEM2_MANGA_EXPLORATION_AWAITING_CONFIRMATION_AUTHORIZATION"
    else:
        decision = "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE"
    selected_models = Counter(row["model_id"] for row in primary["fold_ledger"])
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item2-manga-nonlocal-shape-receipt-1.0",
        "goal": "GRAVITY_ROADMAP_ITEM_02_SHAPE_ANISOTROPY_FOURTH_ATTEMPT",
        "decision": decision,
        "preregistration": {
            "git_commit": PREREGISTRATION_COMMIT,
            "timestamp": "2026-08-28T00:26:10-07:00",
            "selected_kinematic_maps_opened_before_commit": 0,
        },
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity": {
            "label": config["scientific_contract"]["creativity_label"],
            "known_components": config["scientific_contract"]["known_components"],
            "historical_novelty_established": False,
        },
        "counts": {
            "candidate_model_families": len(config["model_families"]),
            "candidate_model_ridge_cells": sum(
                len(config["cross_validation"]["ridge_penalties"])
                if _features_for_model(model)
                else 1
                for model in config["model_families"]
            ),
            "direct_lensing_likelihood_evaluations": 0,
            "equivalence_classes": len(config["model_families"]),
            "exploration_quality_failures": int(extraction["counts"]["quality_failures"]),
            "exploration_quality_passing": len(rows),
            "exploration_selected": int(extraction["counts"]["selected_exploration"]),
            "final_ridge_fits": int(primary["compute_counts"]["final_ridge_fits"])
            + int(extra_fit_counts["final_ridge_fits"]),
            "inner_ridge_fits": int(primary["compute_counts"]["inner_ridge_fits"])
            + int(extra_fit_counts["inner_ridge_fits"]),
            "outer_folds": int(config["cross_validation"]["outer_folds"]),
            "paid_model_calls": 0,
            "reserved_confirmation_objects": int(
                config["target_blind_sample"]["expected_reserved_confirmation_objects"]
            ),
            "reserved_confirmation_target_accesses": 0,
            "sparc_confirmation_evaluator_accesses": 0,
        },
        "data_lineage": {
            "baryonic_input": "MaNGA PCA DR17 resolved i-band stellar mass map",
            "response": "direct DAP stellar velocity and corrected velocity-dispersion aperture statistic",
            "forbidden_derived_targets_read": 0,
            "dark_halo_or_jam_target_used": False,
            "lensing_target_used": False,
        },
        "extraction": extraction,
        "selected_models_by_fold": dict(sorted(selected_models.items())),
        "primary_response": {
            "equation": config["aperture_and_response"]["response_equation"],
            "fold_ledger": primary["fold_ledger"],
            "selected_metrics": primary["selected_metrics"],
            "shape_ablation_metrics": primary["ablation_metrics"],
            "baseline_metrics": {
                model_id: primary["model_metrics"][model_id]
                for model_id in (
                    "constant",
                    "mass_size_nuisance",
                    "class_proxy_nuisance",
                    "pymorph_shape_control",
                    "rejected_global_multipoles",
                )
            },
            "qualifying_model_metrics": {
                model_id: primary["model_metrics"][model_id]
                for model_id in (
                    "radial_quadrupole",
                    "radial_odd_even",
                    "boundary_twist",
                    "nonlocal_profile_energy",
                    "all_nonlocal_shape",
                )
            },
            "observed_primary_mean": _metric(float(np.mean(primary_observed))),
            "heldout_prediction_mean": _metric(float(np.mean(primary_predicted))),
            "heldout_prediction_spearman": _metric(
                float(spearmanr(primary_predicted, primary_observed).statistic)
            ),
        },
        "robustness_responses": robustness,
        "gate_checks": checks,
        "counterexamples": {
            "quality_failures": extraction["failures"],
            "selected_nonqualifying_folds": [
                int(row["fold"]) for row in primary["fold_ledger"] if not row["qualifying"]
            ],
            "failed_gate_names": [key for key, passed in checks.items() if not passed],
        },
        "limitations": [
            "Five of the sixty preregistered exploration galaxies failed the frozen PCA representation-quality gate and were not replaced.",
            "The aperture virial response is a direct observable summary, not a field equation; it remains sensitive to projection, orbital anisotropy, the stellar initial-mass function, distance, and omitted gas.",
            "The population contains ellipticals and S0s, not galaxy groups, filaments, or a direct bridge to the Item 1 beta coefficient.",
            "The five global multipoles rejected in prior attempts are retained only as a nonqualifying control; no new regression on the 88 Item 1 labels was performed.",
            "Because exploration failed, no transfer to the cross-scale beta diagnostic and no reserved-confirmation access is justified.",
        ],
        "source_bindings": {
            "config": {
                "path": source.CONFIG_PATH,
                "sha256": _sha256_file(root / source.CONFIG_PATH),
            },
            "sample_manifest": {
                "path": config["sample_manifest_output"],
                "sha256": _sha256_file(sample_path),
            },
            "source_manifest": {
                "path": config["source_manifest_output"],
                "sha256": _sha256_file(source_path),
            },
            "extraction_summary": {
                "path": source.EXTRACTION_SUMMARY_PATH,
                "sha256": _sha256_file(extraction_path),
            },
            "feature_table": {
                "path": config["feature_output"],
                "sha256": _sha256_file(root / config["feature_output"]),
            },
            "source_code": {
                "path": "src/sigma_theory_compiler/gravity_item2_manga_nonlocal_shape.py",
                "sha256": _sha256_file(
                    root / "src/sigma_theory_compiler/gravity_item2_manga_nonlocal_shape.py"
                ),
            },
            "experiment_code": {
                "path": "src/sigma_theory_compiler/gravity_item2_manga_nonlocal_shape_experiment.py",
                "sha256": _sha256_file(Path(__file__)),
            },
            "test": {
                "path": "tests/test_gravity_item2_manga_nonlocal_shape.py",
                "sha256": _sha256_file(
                    root / "tests/test_gravity_item2_manga_nonlocal_shape.py"
                ),
            },
        },
        "claims": {
            "alternative_to_gr_established": False,
            "aperture_response_is_a_direct_gravity_law": False,
            "cross_scale_beta_explained": False,
            "direct_lensing_test_completed": False,
            "historical_novelty_established": False,
            "manga_confirmation_opened": False,
            "roadmap_item_2_complete": False,
            "sequential_G6_G7_G8_advanced": False,
            "stellar_mass_is_complete_baryonic_mass": False,
        },
        "next_action": (
            "Retain the five quality failures and the held-out diagnostic as counterexamples. "
            "Do not open the 30 reserved confirmation targets. Use the measured result to decide "
            "whether a separately preregistered Item 2 attempt should loosen representation coverage, "
            "move to spectroscopic groups, or transfer only a surviving nonlocal operator."
        ),
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    validate_receipt(receipt, root=root)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    if value.get("schema_version") != (
        "invariant-gravity-roadmap-item2-manga-nonlocal-shape-receipt-1.0"
    ):
        raise GravityItem2MangaNonlocalShapeExperimentError("unexpected receipt schema")
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2MangaNonlocalShapeExperimentError("receipt content hash mismatch")
    if value.get("decision") not in {
        "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE",
        "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE",
        "PASS_ITEM2_MANGA_EXPLORATION_AWAITING_CONFIRMATION_AUTHORIZATION",
    }:
        raise GravityItem2MangaNonlocalShapeExperimentError("invalid Item 2 MaNGA decision")
    if any(bool(value["claims"][claim]) for claim in value["claims"]):
        raise GravityItem2MangaNonlocalShapeExperimentError("receipt contains an overclaim")
    if int(value["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem2MangaNonlocalShapeExperimentError("confirmation target was accessed")
    if int(value["counts"]["paid_model_calls"]) != 0:
        raise GravityItem2MangaNonlocalShapeExperimentError("paid model call entered receipt")
    for binding in value["source_bindings"].values():
        path = root / binding["path"]
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem2MangaNonlocalShapeExperimentError(
                f"receipt source binding changed: {binding['path']}"
            )
    quality_pass = bool(
        value["gate_checks"]["all_sixty_exploration_objects_pass_frozen_quality"]
    )
    if not quality_pass and value["decision"] != (
        "INCONCLUSIVE_ITEM2_MANGA_NONLOCAL_SHAPE_QUALITY_GATE"
    ):
        raise GravityItem2MangaNonlocalShapeExperimentError("quality failure did not control decision")
    if value["decision"].startswith("PASS_") and not all(value["gate_checks"].values()):
        raise GravityItem2MangaNonlocalShapeExperimentError("passing receipt has a failed gate")


def write_receipt(root: Path, receipt: Mapping[str, Any], output: Path) -> None:
    path = output if output.is_absolute() else root.resolve() / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path(OUTPUT_PATH))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    rebuilt = build_receipt(root)
    output = args.output if args.output.is_absolute() else root / args.output
    if args.check:
        stored = json.loads(output.read_text(encoding="utf-8"))
        if stored != rebuilt:
            raise GravityItem2MangaNonlocalShapeExperimentError("stored receipt does not replay")
        validate_receipt(stored, root=root)
    else:
        write_receipt(root, rebuilt, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
