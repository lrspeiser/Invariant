"""Held-out evaluation for gravity-roadmap Item 3 surface/volume density."""

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

from . import gravity_item3_surface_volume_density as source
from .sigma_core import canonical_json_bytes, canonical_sha256

OUTPUT_PATH = "runs/gravity/roadmap/item-03-surface-volume-density-v1.json"
ITEM1_PATH = "runs/gravity/roadmap/item-01-effective-dimension-v1.json"

CROSS_BLOCKS = {
    "surface_amplitude": ("mean_log_surface_source",),
    "item1_dimension": ("local_mass_dimension_median", "local_mass_dimension_iqr"),
    "population_proxy": ("population_proxy",),
    "dual_location": ("log_transition_radius_ratio",),
    "dual_shape": (
        "transition_overlap_cosine",
        "transition_area_asymmetry",
        "log_transition_width_ratio",
    ),
}
CROSS_BLOCKS["all_dual_transition"] = (
    CROSS_BLOCKS["dual_location"] + CROSS_BLOCKS["dual_shape"]
)

GROUP_BASELINE = (
    "log10_total_member_luminosity",
    "log10_r50_kpc",
    "log10_r90_kpc",
    "log10_richness",
    "metadata_redshift",
    "d10",
)
GROUP_DENSITY = (
    "log10_u_surface50",
    "log10_u_volume25_75",
    "surface_volume_log_contrast",
    "log_geometric_mean_sources",
    "source_balance",
    "surface_source_radial_gradient",
    "volume_source_inner_outer_gradient",
)
GROUP_MODELS = (
    {"id": "constant", "features": (), "qualifying": False},
    {"id": "strongest_nuisance_baseline", "features": GROUP_BASELINE, "qualifying": False},
    {
        "id": "surface_volume_density_augmented",
        "features": GROUP_BASELINE + GROUP_DENSITY,
        "qualifying": True,
    },
)


class GravityItem3ExperimentError(RuntimeError):
    """Raised when the frozen Item 3 evaluation or receipt drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem3ExperimentError("non-finite receipt metric")
    return f"{float(value):.12e}"


def _load_tsv(path: Path, *, text_fields: set[str], integer_fields: set[str]) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        raw = list(csv.DictReader(handle, delimiter="\t"))
    rows: list[dict[str, Any]] = []
    for input_row in raw:
        rows.append(
            {
                key: value
                if key in text_fields
                else (int(value) if key in integer_fields else float(value))
                for key, value in input_row.items()
            }
        )
    if any(
        not math.isfinite(float(value))
        for row in rows
        for key, value in row.items()
        if key not in text_fields
    ):
        raise GravityItem3ExperimentError("non-finite feature value")
    return rows


def _load_cross_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _load_tsv(
        root / config["cross_scale_feature_output"],
        text_fields={"domain", "name"},
        integer_fields=set(),
    )
    item1 = json.loads((root / ITEM1_PATH).read_text(encoding="utf-8"))
    labels = {
        (str(row["domain"]), str(row["name"])): float(row["oracle_beta_target_derived"])
        for row in item1["per_object_diagnostics"]
    }
    for row in rows:
        key = (str(row["domain"]), str(row["name"]))
        if key not in labels:
            raise GravityItem3ExperimentError(f"missing Item 1 label: {key}")
        row["beta"] = labels[key]
        row["population_proxy"] = 0.0 if row["domain"] == "galaxy" else 1.0
        row["key"] = f"{row['domain']}:{row['name']}"
    if len({row["key"] for row in rows}) != len(rows):
        raise GravityItem3ExperimentError("duplicate cross-scale object")
    return sorted(rows, key=lambda row: str(row["key"]))


def _load_group_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = _load_tsv(
        root / config["group_feature_output"],
        text_fields=set(),
        integer_fields={"group", "richness_bin", "members", "unique_member_redshifts"},
    )
    for row in rows:
        row["log10_richness"] = math.log10(int(row["members"]))
        row["key"] = str(int(row["group"]))
    if len({row["group"] for row in rows}) != len(rows):
        raise GravityItem3ExperimentError("duplicate fresh group")
    return sorted(rows, key=lambda row: int(row["group"]))


def _cross_models(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    models: list[dict[str, Any]] = []
    for declared in config["cross_scale_development_lane"]["model_families"]:
        features: list[str] = []
        for block in declared["blocks"]:
            features.extend(CROSS_BLOCKS[str(block)])
        if len(features) != len(set(features)):
            raise GravityItem3ExperimentError("duplicate cross-scale feature")
        models.append(
            {
                "id": str(declared["id"]),
                "features": tuple(features),
                "qualifying": bool(declared["qualifying"]),
            }
        )
    return models


def _fold_assignments(
    rows: Sequence[Mapping[str, Any]],
    *,
    key_field: str,
    stratum_field: str,
    salt: str,
    folds: int,
) -> dict[str, int]:
    strata: defaultdict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        strata[str(row[stratum_field])].append(row)
    assignments: dict[str, int] = {}
    for stratum, members in sorted(strata.items()):
        ordered = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{salt}|{stratum}|{row[key_field]}".encode()
            ).hexdigest(),
        )
        for ordinal, row in enumerate(ordered):
            assignments[str(row[key_field])] = ordinal % folds
    return assignments


def _matrix(rows: Sequence[Mapping[str, Any]], features: Sequence[str]) -> np.ndarray:
    if not features:
        return np.empty((len(rows), 0), dtype=np.float64)
    return np.asarray(
        [[float(row[feature]) for feature in features] for row in rows], dtype=np.float64
    )


def _fit(
    rows: Sequence[Mapping[str, Any]],
    *,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> dict[str, Any]:
    raw = _matrix(rows, features)
    y = np.asarray([float(row[response]) for row in rows])
    if raw.shape[1]:
        means = np.mean(raw, axis=0)
        scales = np.std(raw, axis=0)
        scales = np.where(scales > 1.0e-12, scales, 1.0)
        standardized = (raw - means) / scales
    else:
        means = np.empty(0)
        scales = np.empty(0)
        standardized = raw
    design = np.column_stack((np.ones(len(rows)), standardized))
    penalty = np.eye(design.shape[1]) * alpha
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
    standardized = (
        (raw - np.asarray(fit["means"])) / np.asarray(fit["scales"])
        if raw.shape[1]
        else raw
    )
    return np.column_stack((np.ones(len(rows)), standardized)) @ np.asarray(
        fit["coefficients"]
    )


def _inner_score(
    rows: Sequence[Mapping[str, Any]],
    assignments: Mapping[str, int],
    *,
    key_field: str,
    outer_fold: int,
    features: Sequence[str],
    response: str,
    alpha: float,
) -> tuple[float, int]:
    losses: list[float] = []
    fits = 0
    for validation_fold in sorted(set(assignments.values()) - {outer_fold}):
        training = [
            row
            for row in rows
            if assignments[str(row[key_field])] not in {outer_fold, validation_fold}
        ]
        validation = [
            row for row in rows if assignments[str(row[key_field])] == validation_fold
        ]
        fit = _fit(training, features=features, response=response, alpha=alpha)
        observed = np.asarray([float(row[response]) for row in validation])
        losses.extend((observed - _predict(fit, validation)) ** 2)
        fits += 1
    return float(np.mean(losses)), fits


def _metrics(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    *,
    key_field: str,
    stratum_field: str,
    response: str,
) -> dict[str, Any]:
    def calculate(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        observed = np.asarray([float(row[response]) for row in subset])
        predicted = np.asarray([float(predictions[str(row[key_field])]) for row in subset])
        residual = observed - predicted
        denominator = float(np.sum((observed - np.mean(observed)) ** 2))
        correlation = float(spearmanr(observed, predicted).statistic)
        return {
            "mean_squared_error": _metric(float(np.mean(residual**2))),
            "objects": len(subset),
            "r2": _metric(
                1.0 - float(np.sum(residual**2)) / denominator if denominator > 0 else 0.0
            ),
            "spearman": _metric(correlation),
        }

    strata = sorted({str(row[stratum_field]) for row in rows})
    return {
        "overall": calculate(rows),
        "by_stratum": {
            stratum: calculate(
                [row for row in rows if str(row[stratum_field]) == stratum]
            )
            for stratum in strata
        },
    }


def _evaluate(
    rows: Sequence[Mapping[str, Any]],
    models: Sequence[Mapping[str, Any]],
    *,
    key_field: str,
    stratum_field: str,
    response: str,
    salt: str,
    folds: int,
    penalties: Sequence[float],
    selection_pool: str = "all",
) -> dict[str, Any]:
    assignments = _fold_assignments(
        rows,
        key_field=key_field,
        stratum_field=stratum_field,
        salt=salt,
        folds=folds,
    )
    model_predictions = {str(model["id"]): {} for model in models}
    selected_predictions: dict[str, float] = {}
    ledger: list[dict[str, Any]] = []
    inner_fits = 0
    final_fits = 0
    for outer_fold in range(folds):
        training = [
            row for row in rows if assignments[str(row[key_field])] != outer_fold
        ]
        heldout = [row for row in rows if assignments[str(row[key_field])] == outer_fold]
        candidates: list[dict[str, Any]] = []
        for order, model in enumerate(models):
            features = tuple(model["features"])
            model_candidates: list[dict[str, Any]] = []
            for alpha in penalties if features else penalties[:1]:
                score, count = _inner_score(
                    rows,
                    assignments,
                    key_field=key_field,
                    outer_fold=outer_fold,
                    features=features,
                    response=response,
                    alpha=alpha,
                )
                inner_fits += count
                candidate = {
                    "alpha": alpha,
                    "features": features,
                    "inner_mse": score,
                    "model": model,
                    "order": order,
                }
                candidates.append(candidate)
                model_candidates.append(candidate)
            best = min(model_candidates, key=lambda row: (row["inner_mse"], row["alpha"]))
            fit = _fit(
                training,
                features=features,
                response=response,
                alpha=float(best["alpha"]),
            )
            final_fits += 1
            for row, prediction in zip(heldout, _predict(fit, heldout), strict=True):
                model_predictions[str(model["id"])][str(row[key_field])] = float(prediction)
        selectable = (
            [row for row in candidates if row["model"]["qualifying"]]
            if selection_pool == "qualifying"
            else candidates
        )
        selected = min(
            selectable,
            key=lambda row: (
                row["inner_mse"],
                len(row["features"]),
                row["order"],
                row["alpha"],
            ),
        )
        fit = _fit(
            training,
            features=selected["features"],
            response=response,
            alpha=float(selected["alpha"]),
        )
        final_fits += 1
        for row, prediction in zip(heldout, _predict(fit, heldout), strict=True):
            selected_predictions[str(row[key_field])] = float(prediction)
        ledger.append(
            {
                "alpha": _metric(float(selected["alpha"])),
                "fold": outer_fold,
                "heldout_objects": len(heldout),
                "inner_mse": _metric(float(selected["inner_mse"])),
                "model_id": str(selected["model"]["id"]),
                "qualifying": bool(selected["model"]["qualifying"]),
            }
        )
    return {
        "compute": {"final_fits": final_fits, "inner_fits": inner_fits},
        "fold_ledger": ledger,
        "model_metrics": {
            model_id: _metrics(
                rows,
                values,
                key_field=key_field,
                stratum_field=stratum_field,
                response=response,
            )
            for model_id, values in model_predictions.items()
        },
        "model_predictions": model_predictions,
        "predictions": selected_predictions,
        "selected_metrics": _metrics(
            rows,
            selected_predictions,
            key_field=key_field,
            stratum_field=stratum_field,
            response=response,
        ),
    }


def _cross_overlap(
    rows: Sequence[Mapping[str, Any]], result: Mapping[str, Any]
) -> dict[str, Any]:
    galaxy = [float(row["mean_log_surface_source"]) for row in rows if row["domain"] == "galaxy"]
    cluster = [float(row["mean_log_surface_source"]) for row in rows if row["domain"] == "cluster"]
    lower = max(min(galaxy), min(cluster))
    upper = min(max(galaxy), max(cluster))
    subset = [row for row in rows if lower <= float(row["mean_log_surface_source"]) <= upper]

    def mse(predictions: Mapping[str, float], domain: str) -> str:
        selected = [row for row in subset if row["domain"] == domain]
        observed = np.asarray([float(row["beta"]) for row in selected])
        predicted = np.asarray([float(predictions[str(row["key"])]) for row in selected])
        return _metric(float(np.mean((observed - predicted) ** 2)))

    selected_predictions = result["predictions"]
    surface_dimension_predictions = result["model_predictions"][
        "surface_plus_dimension_control"
    ]
    population_predictions = result["model_predictions"]["binary_population_proxy"]
    return {
        "range": [_metric(lower), _metric(upper)],
        "objects": Counter(str(row["domain"]) for row in subset),
        "selected_mse": {
            domain: mse(selected_predictions, domain) for domain in ("galaxy", "cluster")
        },
        "surface_dimension_control_mse": {
            domain: mse(surface_dimension_predictions, domain)
            for domain in ("galaxy", "cluster")
        },
        "population_proxy_mse": {
            domain: mse(population_predictions, domain) for domain in ("galaxy", "cluster")
        },
    }


def _group_permutation(
    rows: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    observed: Mapping[str, Any],
) -> dict[str, Any]:
    baseline_mse = float(
        observed["model_metrics"]["strongest_nuisance_baseline"]["overall"][
            "mean_squared_error"
        ]
    )
    density_mse = float(
        observed["model_metrics"]["surface_volume_density_augmented"]["overall"][
            "mean_squared_error"
        ]
    )
    improvement = baseline_mse - density_mse
    cv = config["cross_validation"]
    count = int(cv["stratified_permutation_count"])
    null: list[float] = []
    for permutation in range(count):
        shuffled = [dict(row) for row in rows]
        by_bin: defaultdict[int, list[int]] = defaultdict(list)
        for index, row in enumerate(shuffled):
            by_bin[int(row["richness_bin"])].append(index)
        for richness_bin, indices in by_bin.items():
            seed = int.from_bytes(
                hashlib.sha256(
                    f"{cv['permutation_salt']}|{permutation}|{richness_bin}".encode()
                ).digest()[:8],
                "big",
            )
            rng = np.random.default_rng(seed)
            values = np.asarray([shuffled[index]["log10_sigma_gap"] for index in indices])
            for index, value in zip(indices, rng.permutation(values), strict=True):
                shuffled[index]["log10_sigma_gap"] = float(value)
        result = _evaluate(
            shuffled,
            GROUP_MODELS,
            key_field="group",
            stratum_field="richness_bin",
            response="log10_sigma_gap",
            salt=str(cv["group_fold_salt"]),
            folds=int(cv["outer_folds"]),
            penalties=[float(value) for value in cv["ridge_penalties"]],
        )
        null.append(
            float(
                result["model_metrics"]["strongest_nuisance_baseline"]["overall"][
                    "mean_squared_error"
                ]
            )
            - float(
                result["model_metrics"]["surface_volume_density_augmented"]["overall"][
                    "mean_squared_error"
                ]
            )
        )
    p_value = (1 + sum(value >= improvement for value in null)) / (1 + count)
    return {
        "null_improvement_max": _metric(max(null)),
        "null_improvement_median": _metric(float(np.median(null))),
        "observed_improvement": _metric(improvement),
        "p_value": _metric(p_value),
        "permutations": count,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = source.load_config(root)
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    source.validate_sample_manifest(sample, config=config)
    source_manifest = json.loads(
        (root / config["source_manifest_output"]).read_text(encoding="utf-8")
    )
    source.validate_source_manifest(source_manifest, sample=sample)
    extraction_path = root / source.EXTRACTION_SUMMARY_PATH
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    extraction_copy = dict(extraction)
    extraction_digest = extraction_copy.pop("content_sha256", None)
    if extraction_digest != canonical_sha256(extraction_copy):
        raise GravityItem3ExperimentError("extraction summary hash changed")
    cross_rows = _load_cross_rows(root, config)
    group_rows = _load_group_rows(root, config)
    cv = config["cross_validation"]
    penalties = [float(value) for value in cv["ridge_penalties"]]
    cross = _evaluate(
        cross_rows,
        _cross_models(config),
        key_field="key",
        stratum_field="domain",
        response="beta",
        salt=str(cv["cross_scale_fold_salt"]),
        folds=int(cv["outer_folds"]),
        penalties=penalties,
    )
    groups = _evaluate(
        group_rows,
        GROUP_MODELS,
        key_field="group",
        stratum_field="richness_bin",
        response="log10_sigma_gap",
        salt=str(cv["group_fold_salt"]),
        folds=int(cv["outer_folds"]),
        penalties=penalties,
    )
    permutation = _group_permutation(group_rows, config, groups)
    robustness = {}
    for response in ("log10_sigma_mad", "log10_eta_r50", "log10_eta_r90"):
        result = _evaluate(
            group_rows,
            GROUP_MODELS,
            key_field="group",
            stratum_field="richness_bin",
            response=response,
            salt=str(cv["group_fold_salt"]),
            folds=int(cv["outer_folds"]),
            penalties=penalties,
        )
        density = result["model_metrics"]["surface_volume_density_augmented"]
        baseline = result["model_metrics"]["strongest_nuisance_baseline"]
        robustness[response] = {
            "baseline_metrics": baseline,
            "density_metrics": density,
            "passes": (
                float(density["overall"]["r2"]) > 0
                and float(density["overall"]["spearman"]) > 0
                and float(density["overall"]["mean_squared_error"])
                < float(baseline["overall"]["mean_squared_error"])
            ),
        }
    cross_selected = cross["selected_metrics"]
    cross_surface_dimension = cross["model_metrics"]["surface_plus_dimension_control"]
    cross_proxy = cross["model_metrics"]["binary_population_proxy"]
    group_selected = groups["selected_metrics"]
    group_baseline = groups["model_metrics"]["strongest_nuisance_baseline"]
    overlap = _cross_overlap(cross_rows, cross)
    gates = {
        "cross_scale_selected_model_qualifying_in_every_fold": all(
            row["qualifying"] for row in cross["fold_ledger"]
        ),
        "cross_scale_beta_r2_positive_in_galaxies_and_clusters": all(
            float(cross_selected["by_stratum"][domain]["r2"]) > 0
            for domain in ("galaxy", "cluster")
        ),
        "cross_scale_beats_surface_dimension_and_population_controls_each_domain": all(
            float(cross_selected["by_stratum"][domain]["mean_squared_error"])
            < min(
                float(cross_surface_dimension["by_stratum"][domain]["mean_squared_error"]),
                float(cross_proxy["by_stratum"][domain]["mean_squared_error"]),
            )
            for domain in ("galaxy", "cluster")
        ),
        "cross_scale_beats_surface_dimension_and_population_controls_in_overlap": all(
            float(overlap["selected_mse"][domain])
            < min(
                float(overlap["surface_dimension_control_mse"][domain]),
                float(overlap["population_proxy_mse"][domain]),
            )
            for domain in ("galaxy", "cluster")
        ),
        "all_120_fresh_groups_pass_quality": (
            int(extraction["counts"]["fresh_group_passing"]) == 120
            and not extraction["fresh_group_failures"]
        ),
        "all_159_cross_scale_objects_pass_quality": (
            int(extraction["counts"]["cross_scale_passing"]) == 159
            and not extraction["cross_scale_failures"]
        ),
        "fresh_group_selected_model_qualifying_in_every_fold": all(
            row["qualifying"] for row in groups["fold_ledger"]
        ),
        "fresh_group_r2_positive_in_every_richness_bin": all(
            float(group_selected["by_stratum"][str(value)]["r2"]) > 0
            for value in range(3)
        ),
        "fresh_group_density_beats_strongest_baseline_overall_and_every_bin": (
            float(group_selected["overall"]["mean_squared_error"])
            < float(group_baseline["overall"]["mean_squared_error"])
            and all(
                float(group_selected["by_stratum"][str(value)]["mean_squared_error"])
                < float(group_baseline["by_stratum"][str(value)]["mean_squared_error"])
                for value in range(3)
            )
        ),
        "fresh_group_permutation_p_at_most_0p05": float(permutation["p_value"]) <= 0.05,
        "all_group_response_robustness_controls_pass": all(
            value["passes"] for value in robustness.values()
        ),
        "confirmation_untouched": (
            int(extraction["counts"]["reserved_confirmation_target_accesses"]) == 0
        ),
    }
    decision = (
        "PASS_ITEM3_SURFACE_VOLUME_DENSITY_EXPLORATION_REQUIRES_AUTHORIZATION"
        if all(gates.values())
        else (
            "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE"
            if not gates["all_120_fresh_groups_pass_quality"]
            or not gates["all_159_cross_scale_objects_pass_quality"]
            else "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY"
        )
    )
    receipt: dict[str, Any] = {
        "schema_version": "invariant-gravity-roadmap-item3-surface-volume-density-receipt-1.0",
        "goal": config["goal"],
        "decision": decision,
        "hypothesis": config["scientific_contract"]["hypothesis"],
        "creativity": {
            "label": config["scientific_contract"]["creativity_label"],
            "exact_rewrite_excluded": config["scientific_contract"]["exact_rewrite_excluded"],
            "historical_novelty_established": False,
        },
        "preregistration": {
            "git_commit": source.FREEZE_COMMIT,
            "item3_features_computed_before_commit": 0,
            "fresh_member_rows_opened_before_commit": 0,
        },
        "data_lineage": {
            "cross_scale": "retrospective SPARC/CLASH development labels from Item 1",
            "fresh_groups": "120 AXES groups disjoint from every Item 2 group role",
            "fresh_group_response": "directly recomputed member-redshift dispersion",
            "published_group_velocity_columns_read": 0,
            "lensing_target_used_in_fresh_lane": False,
        },
        "representation": extraction,
        "cross_scale_result": {
            "selected_metrics": cross_selected,
            "model_metrics": cross["model_metrics"],
            "fold_ledger": cross["fold_ledger"],
            "surface_amplitude_overlap": overlap,
        },
        "fresh_group_result": {
            "selected_metrics": group_selected,
            "model_metrics": groups["model_metrics"],
            "fold_ledger": groups["fold_ledger"],
            "permutation_test": permutation,
            "robustness": robustness,
        },
        "gate_checks": gates,
        "counterexamples": {
            "cross_scale_quality_failures": extraction["cross_scale_failures"],
            "fresh_group_quality_failures": extraction["fresh_group_failures"],
            "failed_gate_names": [key for key, passed in gates.items() if not passed],
        },
        "counts": {
            "cross_scale_quality_passing": len(cross_rows),
            "fresh_group_quality_passing": len(group_rows),
            "fresh_group_exploration_selected": 120,
            "fresh_group_confirmation_groups": 60,
            "fresh_group_confirmation_target_accesses": 0,
            "cross_scale_model_families": len(_cross_models(config)),
            "fresh_group_model_families": len(GROUP_MODELS),
            "stratified_permutations": int(permutation["permutations"]),
            "paid_model_calls": 0,
            "direct_lensing_likelihood_evaluations": 0,
            "sparc_confirmation_evaluator_accesses": 0,
        },
        "limitations": [
            "The cross-scale lane is retrospective because its Item 1 labels were already inspected.",
            "The frozen minimum profile length excludes eleven cross-scale objects, including nine clusters.",
            "Strict luminosity quantile radii exclude thirty-three fresh groups and were not relaxed.",
            "The group density proxy assumes one universal r-band mass-to-light factor and omits gas and diffuse light.",
            "The AXES memberships were created with FoF and Clean procedures that use redshifts and mass-model assumptions.",
            "The fresh-group augmented model is the direct union of the frozen strongest baseline and all seven frozen density features; the config did not enumerate alternative group subfamilies.",
        ],
        "claims": {
            "alternative_to_gr_established": False,
            "cross_scale_lane_is_independent_confirmation": False,
            "historical_novelty_established": False,
            "optical_group_light_is_complete_baryonic_mass": False,
            "reserved_confirmation_opened": False,
            "roadmap_item_3_complete": False,
        },
        "next_action": "Retain every quality and predictive counterexample and keep 60 confirmation groups sealed. Use the result to decide whether to redesign Item 3 around smoother mass profiles or advance only after a scoped synthesis; do not relax this attempt post-target.",
        "source_bindings": {
            "config": {"path": source.CONFIG_PATH, "sha256": _sha256_file(root / source.CONFIG_PATH)},
            "sample": {
                "path": config["sample_manifest_output"],
                "sha256": _sha256_file(root / config["sample_manifest_output"]),
            },
            "source_manifest": {
                "path": config["source_manifest_output"],
                "sha256": _sha256_file(root / config["source_manifest_output"]),
            },
            "extraction": {
                "path": source.EXTRACTION_SUMMARY_PATH,
                "sha256": _sha256_file(extraction_path),
            },
            "cross_features": {
                "path": config["cross_scale_feature_output"],
                "sha256": _sha256_file(root / config["cross_scale_feature_output"]),
            },
            "group_features": {
                "path": config["group_feature_output"],
                "sha256": _sha256_file(root / config["group_feature_output"]),
            },
            "item1_receipt": {"path": ITEM1_PATH, "sha256": _sha256_file(root / ITEM1_PATH)},
            "source_code": {
                "path": str(Path(source.__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(source.__file__)),
            },
            "experiment_code": {
                "path": str(Path(__file__).resolve().relative_to(root)).replace("\\", "/"),
                "sha256": _sha256_file(Path(__file__)),
            },
            "test": {
                "path": "tests/test_gravity_item3_surface_volume_density.py",
                "sha256": _sha256_file(root / "tests/test_gravity_item3_surface_volume_density.py"),
            },
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(value: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem3ExperimentError("receipt content hash changed")
    if value.get("decision") not in {
        "PASS_ITEM3_SURFACE_VOLUME_DENSITY_EXPLORATION_REQUIRES_AUTHORIZATION",
        "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY_QUALITY_GATE",
        "INCONCLUSIVE_ITEM3_SURFACE_VOLUME_DENSITY",
    }:
        raise GravityItem3ExperimentError("unexpected Item 3 decision")
    if int(value["counts"]["fresh_group_confirmation_target_accesses"]) != 0:
        raise GravityItem3ExperimentError("Item 3 confirmation was accessed")
    if any(bool(value["claims"][key]) for key in value["claims"]):
        raise GravityItem3ExperimentError("Item 3 receipt contains an overclaim")
    if str(value["decision"]).startswith("PASS_") and not all(value["gate_checks"].values()):
        raise GravityItem3ExperimentError("PASS decision has a failed gate")
    for binding in value["source_bindings"].values():
        path = Path(binding["path"])
        if not path.is_absolute():
            path = root / path
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem3ExperimentError(f"source binding changed: {path}")


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
        stored = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_receipt(stored, root=root)
        if build_receipt(root) != stored:
            raise GravityItem3ExperimentError("receipt is not an exact rebuild")
        return 0
    path = write_receipt(root)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
