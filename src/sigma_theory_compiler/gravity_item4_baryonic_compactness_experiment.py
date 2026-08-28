"""Nested held-out experiment for gravity roadmap Item 4 baryonic compactness."""

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

from . import gravity_item4_baryonic_compactness as source
from .sigma_core import canonical_json_bytes, canonical_sha256


class GravityItem4CompactnessExperimentError(RuntimeError):
    """Raised when the frozen Item 4 experiment or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem4CompactnessExperimentError("non-finite receipt metric")
    return f"{float(value):.12e}"


def _load_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / config["feature_output"]
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    integer_fields = {"group", "members", "unique_member_redshifts"}
    rows: list[dict[str, Any]] = []
    for input_row in raw:
        row = {
            key: (
                str(value)
                if key == "richness_stratum"
                else int(value)
                if key in integer_fields
                else float(value)
            )
            for key, value in input_row.items()
        }
        rows.append(row)
    if len(rows) != len({int(row["group"]) for row in rows}):
        raise GravityItem4CompactnessExperimentError("duplicate group in feature table")
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key != "richness_stratum"
    ):
        raise GravityItem4CompactnessExperimentError("non-finite feature-table value")
    return sorted(rows, key=lambda row: int(row["group"]))


def _feature_blocks(config: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    return {
        str(block): tuple(str(value) for value in features)
        for block, features in config["feature_blocks"].items()
    }


def _features_for_model(model: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, ...]:
    blocks = _feature_blocks(config)
    values: list[str] = []
    for block in model["blocks"]:
        if str(block) not in blocks:
            raise GravityItem4CompactnessExperimentError(f"unknown feature block: {block}")
        values.extend(blocks[str(block)])
    if len(values) != len(set(values)):
        raise GravityItem4CompactnessExperimentError(f"duplicate feature in model {model['id']}")
    return tuple(values)


def fold_assignments(rows: Sequence[Mapping[str, Any]], *, salt: str, folds: int) -> dict[int, int]:
    strata: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row["richness_stratum"])].append(row)
    assignments: dict[int, int] = {}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(f"{salt}|{stratum}|{row['group']}".encode()).hexdigest(),
        )
        for ordinal, row in enumerate(ordered):
            assignments[int(row["group"])] = ordinal % folds
    if len(assignments) != len(rows):
        raise GravityItem4CompactnessExperimentError("incomplete fold assignment")
    return assignments


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    matrix = np.asarray(
        [[float(row[feature]) for feature in features] for row in rows],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(matrix)):
        raise GravityItem4CompactnessExperimentError("non-finite model matrix")
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
    return np.column_stack((np.ones(len(rows)), standardized)) @ np.asarray(fit["coefficients"])


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
    for validation_fold in sorted(set(assignments.values()) - {excluded_outer_fold}):
        training = [
            row
            for row in rows
            if assignments[int(row["group"])] not in {excluded_outer_fold, validation_fold}
        ]
        validation = [row for row in rows if assignments[int(row["group"])] == validation_fold]
        fit = _fit_ridge(training, features=features, response=response, alpha=alpha)
        observed = np.asarray([float(row[response]) for row in validation])
        losses.extend((observed - _predict(fit, validation)) ** 2)
        fits += 1
    return float(np.mean(losses)), fits


def _calculate_metrics(
    rows: Sequence[Mapping[str, Any]], predicted: Mapping[int, float], response: str
) -> dict[str, Any]:
    def calculate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        observed = np.asarray([float(row[response]) for row in subset], dtype=np.float64)
        values = np.asarray([float(predicted[int(row["group"])]) for row in subset])
        residual = observed - values
        denominator = float(np.sum((observed - np.mean(observed)) ** 2))
        r2 = 1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
        correlation = float(spearmanr(observed, values).statistic)
        if not math.isfinite(correlation):
            correlation = 0.0
        return {
            "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
            "mean_squared_error": _metric(float(np.mean(residual**2))),
            "objects": len(subset),
            "r2": _metric(r2),
            "spearman": _metric(correlation),
        }

    strata = sorted({str(row["richness_stratum"]) for row in rows})
    return {
        "overall": calculate(rows),
        "by_richness_stratum": {
            stratum: calculate([row for row in rows if str(row["richness_stratum"]) == stratum])
            for stratum in strata
        },
    }


def _evaluate_response(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str,
    selection_pool: str,
) -> dict[str, Any]:
    cv = config["cross_validation"]
    folds = int(cv["outer_folds"])
    assignments = fold_assignments(rows, salt=str(cv["fold_salt"]), folds=folds)
    models = list(config["model_families"])
    penalties = [float(value) for value in cv["ridge_penalties"]]
    predictions_by_model: dict[str, dict[int, float]] = {str(model["id"]): {} for model in models}
    selected_predictions: dict[int, float] = {}
    fold_ledger: list[dict[str, Any]] = []
    inner_fits = 0
    final_fits = 0
    for outer_fold in range(folds):
        training = [row for row in rows if assignments[int(row["group"])] != outer_fold]
        heldout = [row for row in rows if assignments[int(row["group"])] == outer_fold]
        candidates: list[dict[str, Any]] = []
        for model_order, model in enumerate(models):
            features = _features_for_model(model, config)
            model_candidates: list[dict[str, Any]] = []
            for alpha in penalties if features else penalties[:1]:
                score, fits = _inner_score(
                    rows,
                    assignments,
                    excluded_outer_fold=outer_fold,
                    features=features,
                    response=response,
                    alpha=alpha,
                )
                inner_fits += fits
                candidate = {
                    "alpha": alpha,
                    "features": features,
                    "inner_mse": score,
                    "model": model,
                    "model_order": model_order,
                }
                candidates.append(candidate)
                model_candidates.append(candidate)
            best = min(
                model_candidates,
                key=lambda row: (row["inner_mse"], len(row["features"]), row["alpha"]),
            )
            fit = _fit_ridge(
                training,
                features=features,
                response=response,
                alpha=float(best["alpha"]),
            )
            final_fits += 1
            for object_row, prediction in zip(heldout, _predict(fit, heldout), strict=True):
                predictions_by_model[str(model["id"])][int(object_row["group"])] = float(prediction)
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
        final_fits += 1
        for object_row, prediction in zip(heldout, _predict(selected_fit, heldout), strict=True):
            selected_predictions[int(object_row["group"])] = float(prediction)
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
        "compute_counts": {
            "final_ridge_fits": final_fits,
            "inner_ridge_fits": inner_fits,
        },
        "fold_assignments": assignments,
        "fold_ledger": fold_ledger,
        "model_metrics": {
            model_id: _calculate_metrics(rows, predictions, response)
            for model_id, predictions in predictions_by_model.items()
        },
        "predictions": selected_predictions,
        "selected_metrics": _calculate_metrics(rows, selected_predictions, response),
        "selection_pool": selection_pool,
    }


def _evaluate_fixed_selection(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    response: str,
    selection: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    cv = config["cross_validation"]
    assignments = fold_assignments(rows, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"]))
    models = {str(model["id"]): model for model in config["model_families"]}
    predictions: dict[int, float] = {}
    baseline_predictions: dict[int, float] = {}
    baseline = models["strongest_nuisance"]
    for fold in range(int(cv["outer_folds"])):
        training = [row for row in rows if assignments[int(row["group"])] != fold]
        heldout = [row for row in rows if assignments[int(row["group"])] == fold]
        chosen = selection[fold]
        model = models[str(chosen["model_id"])]
        alpha = float(chosen["alpha"])
        fit = _fit_ridge(
            training,
            features=_features_for_model(model, config),
            response=response,
            alpha=alpha,
        )
        baseline_fit = _fit_ridge(
            training,
            features=_features_for_model(baseline, config),
            response=response,
            alpha=alpha,
        )
        for row, value, baseline_value in zip(
            heldout,
            _predict(fit, heldout),
            _predict(baseline_fit, heldout),
            strict=True,
        ):
            predictions[int(row["group"])] = float(value)
            baseline_predictions[int(row["group"])] = float(baseline_value)
    metrics = _calculate_metrics(rows, predictions, response)
    baseline_metrics = _calculate_metrics(rows, baseline_predictions, response)
    return {
        "baseline_metrics": baseline_metrics,
        "metrics": metrics,
        "mse_improvement": _metric(
            float(baseline_metrics["overall"]["mean_squared_error"])
            - float(metrics["overall"]["mean_squared_error"])
        ),
    }


def _permuted_rows(
    rows: Sequence[Mapping[str, Any]], *, response: str, salt: str, ordinal: int
) -> list[dict[str, Any]]:
    permuted = [dict(row) for row in rows]
    by_stratum: defaultdict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(permuted):
        by_stratum[str(row["richness_stratum"])].append(index)
    for stratum, indices in sorted(by_stratum.items()):
        seed = int.from_bytes(
            hashlib.sha256(f"{salt}|{ordinal}|{stratum}".encode()).digest()[:8],
            "big",
        )
        rng = np.random.default_rng(seed)
        values = np.asarray([permuted[index][response] for index in indices])
        for index, value in zip(indices, rng.permutation(values), strict=True):
            permuted[index][response] = float(value)
    return permuted


def _permutation_test(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    response = "log10_sigma_gap"
    observed = _evaluate_response(rows, config, response=response, selection_pool="qualifying")
    baseline = float(
        observed["model_metrics"]["strongest_nuisance"]["overall"]["mean_squared_error"]
    )
    selected = float(observed["selected_metrics"]["overall"]["mean_squared_error"])
    observed_improvement = baseline - selected
    count = int(config["cross_validation"]["permutation_count"])
    salt = str(config["cross_validation"]["permutation_salt"])
    null_improvements: list[float] = []
    for ordinal in range(count):
        permuted = _permuted_rows(rows, response=response, salt=salt, ordinal=ordinal)
        result = _evaluate_response(
            permuted, config, response=response, selection_pool="qualifying"
        )
        null_baseline = float(
            result["model_metrics"]["strongest_nuisance"]["overall"]["mean_squared_error"]
        )
        null_selected = float(result["selected_metrics"]["overall"]["mean_squared_error"])
        null_improvements.append(null_baseline - null_selected)
    exceedances = sum(value >= observed_improvement for value in null_improvements)
    return {
        "exceedances": exceedances,
        "null_improvement_max": _metric(max(null_improvements)),
        "null_improvement_median": _metric(float(np.median(null_improvements))),
        "observed_improvement": _metric(observed_improvement),
        "permutations": count,
        "p_value": _metric((1 + exceedances) / (1 + count)),
        "selection_pool": "qualifying potential-structure families only",
        "statistic": "strongest-nuisance MSE minus nested qualifying-structure MSE",
    }


def _eta_dispersion(rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gravity = float(config["constants"]["gravity_kpc_km2_s2_msun"])
    pair = np.asarray([float(row["log10_eta_pair"]) for row in rows])
    mass_size = np.asarray(
        [
            2.0 * float(row["log10_sigma_gap"])
            - math.log10(gravity)
            - float(row["log10_mass"])
            + float(row["log10_r_rms"])
            for row in rows
        ]
    )
    by_stratum: dict[str, Any] = {}
    for stratum in sorted({str(row["richness_stratum"]) for row in rows}):
        indices = [
            index for index, row in enumerate(rows) if str(row["richness_stratum"]) == stratum
        ]
        by_stratum[stratum] = {
            "log_eta_mass_size_std": _metric(float(np.std(mass_size[indices]))),
            "log_eta_pair_std": _metric(float(np.std(pair[indices]))),
        }
    return {
        "by_richness_stratum": by_stratum,
        "log_eta_mass_size_std": _metric(float(np.std(mass_size))),
        "log_eta_pair_std": _metric(float(np.std(pair))),
        "pair_minus_mass_size_std": _metric(float(np.std(pair) - np.std(mass_size))),
    }


def _gate_checks(
    *,
    primary: Mapping[str, Any],
    qualifying: Mapping[str, Any],
    mad: Mapping[str, Any],
    eta: Mapping[str, Any],
    permutation: Mapping[str, Any],
    extraction: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, bool]:
    strata = tuple(str(value["id"]) for value in config["sample"]["strata"])
    selected = primary["selected_metrics"]
    candidate = qualifying["selected_metrics"]
    baseline = qualifying["model_metrics"]["strongest_nuisance"]
    return {
        "all_52_exploration_groups_pass_frozen_quality": (
            extraction["decision"] == "PASS_ITEM4_EXPLORATION_REPRESENTATION_QUALITY"
            and int(extraction["counts"]["quality_passing"]) == 52
        ),
        "selected_model_qualifying_in_every_outer_fold": all(
            bool(row["qualifying"]) for row in primary["fold_ledger"]
        ),
        "primary_r2_positive_in_each_richness_stratum": all(
            float(selected["by_richness_stratum"][stratum]["r2"]) > 0 for stratum in strata
        ),
        "qualifying_model_beats_strongest_nuisance_overall_and_in_each_stratum": (
            float(candidate["overall"]["mean_squared_error"])
            < float(baseline["overall"]["mean_squared_error"])
            and all(
                float(candidate["by_richness_stratum"][stratum]["mean_squared_error"])
                < float(baseline["by_richness_stratum"][stratum]["mean_squared_error"])
                for stratum in strata
            )
        ),
        "permutation_p_at_most_frozen_threshold": (
            float(permutation["p_value"])
            <= float(config["exploration_admission"]["permutation_p_at_most"])
        ),
        "mad_response_improvement_positive": float(mad["mse_improvement"]) > 0,
        "eta_pair_dispersion_decreases_relative_to_mass_size_control": (
            float(eta["log_eta_pair_std"]) < float(eta["log_eta_mass_size_std"])
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
    source.validate_sample_manifest(sample, config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    source.validate_source_manifest(source_manifest, sample=sample)
    extraction_path = root / source.EXTRACTION_SUMMARY_PATH
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction_copy = dict(extraction)
    extraction_digest = extraction_copy.pop("content_sha256", None)
    if extraction_digest != canonical_sha256(extraction_copy):
        raise GravityItem4CompactnessExperimentError("extraction hash changed")
    rows = _load_rows(root, config)
    if len(rows) != int(extraction["counts"]["quality_passing"]):
        raise GravityItem4CompactnessExperimentError("feature/extraction count mismatch")
    primary = _evaluate_response(rows, config, response="log10_sigma_gap", selection_pool="all")
    qualifying = _evaluate_response(
        rows, config, response="log10_sigma_gap", selection_pool="qualifying"
    )
    frozen_qualifying_selection = {
        int(row["fold"]): {"alpha": row["alpha"], "model_id": row["model_id"]}
        for row in qualifying["fold_ledger"]
    }
    mad = _evaluate_fixed_selection(
        rows,
        config,
        response="log10_sigma_mad",
        selection=frozen_qualifying_selection,
    )
    eta = _eta_dispersion(rows, config)
    permutation = _permutation_test(rows, config)
    gates = _gate_checks(
        primary=primary,
        qualifying=qualifying,
        mad=mad,
        eta=eta,
        permutation=permutation,
        extraction=extraction,
        config=config,
    )
    decision = (
        "PASS_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION_REQUIRES_AUTHORIZATION"
        if all(gates.values())
        else (
            "INCONCLUSIVE_ITEM4_BARYONIC_COMPACTNESS_QUALITY_GATE"
            if not gates["all_52_exploration_groups_pass_frozen_quality"]
            else "REJECT_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION"
        )
    )
    models = list(config["model_families"])
    candidate_cells = sum(
        len(config["cross_validation"]["ridge_penalties"])
        if _features_for_model(model, config)
        else 1
        for model in models
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item4-baryonic-compactness-receipt-1.0",
        "goal": config["goal"],
        "decision": decision,
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity": {
            "label": config["scientific_contract"]["creativity_label"],
            "known_components": config["scientific_contract"]["known_components"],
            "nonqualifying_rewrites": config["scientific_contract"]["nonqualifying_rewrites"],
            "historical_novelty_established": False,
        },
        "preregistration": {
            "git_commit": source.FREEZE_COMMIT,
            "member_rows_opened_before_commit": 0,
        },
        "data_lineage": {
            "catalog": config["source"]["catalog_id"],
            "baryonic_input": "projected positions and r-band light of cleaned group members",
            "response": "line-of-sight dispersion recomputed from exploration member redshifts",
            "published_group_velocity_columns_read": 0,
            "published_r200_or_m200_read": 0,
            "lensing_target_used": False,
            "dark_halo_target_used": False,
        },
        "response": {
            "primary_all_family_selector": {
                "selected_metrics": primary["selected_metrics"],
                "fold_ledger": primary["fold_ledger"],
                "model_metrics": primary["model_metrics"],
            },
            "qualifying_only_selector": {
                "selected_metrics": qualifying["selected_metrics"],
                "fold_ledger": qualifying["fold_ledger"],
                "model_metrics": qualifying["model_metrics"],
            },
            "mad_robustness": mad,
            "eta_dispersion": eta,
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
            "exploration_selected": 52,
            "exploration_quality_passing": len(rows),
            "exploration_quality_failures": len(extraction["failures"]),
            "reserved_confirmation_groups": 21,
            "reserved_confirmation_target_accesses": 0,
            "outer_folds": int(config["cross_validation"]["outer_folds"]),
            "inner_ridge_fits_primary": int(primary["compute_counts"]["inner_ridge_fits"]),
            "inner_ridge_fits_qualifying": int(qualifying["compute_counts"]["inner_ridge_fits"]),
            "stratified_permutations": int(permutation["permutations"]),
            "paid_model_calls": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "sparc_confirmation_evaluator_accesses": 0,
        },
        "limitations": config["scientific_contract"]["not_claimed"],
        "claims": {
            "alternative_to_gr_established": False,
            "complete_baryonic_mass_used": False,
            "direct_lensing_test_completed": False,
            "group_finder_independence_established": False,
            "historical_novelty_established": False,
            "reserved_confirmation_opened": False,
            "roadmap_item_4_complete": False,
            "sequential_G6_G7_G8_advanced": False,
        },
        "next_action": (
            "Do not open the 21 confirmation groups without explicit authorization. "
            "If the exploration passes, request authorization; otherwise retain the "
            "counterexamples, synthesize the tested Item 4 families, and advance only "
            "under the stable roadmap."
        ),
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
                "path": str(Path(source.__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(source.__file__)),
            },
            "experiment_code": {
                "path": str(Path(__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(__file__)),
            },
            "test": {
                "path": "tests/test_gravity_item4_baryonic_compactness.py",
                "sha256": _sha256_file(root / "tests/test_gravity_item4_baryonic_compactness.py"),
            },
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem4CompactnessExperimentError("receipt content hash changed")
    if value.get("decision") not in {
        "PASS_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION_REQUIRES_AUTHORIZATION",
        "INCONCLUSIVE_ITEM4_BARYONIC_COMPACTNESS_QUALITY_GATE",
        "REJECT_ITEM4_BARYONIC_COMPACTNESS_EXPLORATION",
    }:
        raise GravityItem4CompactnessExperimentError("unknown decision")
    if int(value["counts"]["reserved_confirmation_target_accesses"]) != 0:
        raise GravityItem4CompactnessExperimentError("confirmation was accessed")
    if any(bool(claim) for claim in value["claims"].values()):
        raise GravityItem4CompactnessExperimentError("receipt contains overclaim")
    if str(value["decision"]).startswith("PASS_") and not all(
        bool(passed) for passed in value["gate_checks"].values()
    ):
        raise GravityItem4CompactnessExperimentError("PASS decision has a failed gate")
    for binding in value["source_bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = root / path
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem4CompactnessExperimentError(f"source binding changed: {path}")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = source.load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / config["output"]
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--check", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    config = source.load_config(root)
    if args.check:
        path = root / config["output"]
        stored = json.loads(path.read_text(encoding="utf-8"))
        validate_receipt(stored, root=root)
        if build_receipt(root) != stored:
            raise GravityItem4CompactnessExperimentError("receipt is not an exact rebuild")
        return 0
    print(write_receipt(root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
