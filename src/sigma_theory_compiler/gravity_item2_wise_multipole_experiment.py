"""Run roadmap Item 2 on target-blind unWISE and CLASH morphology multipoles."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item1_effective_dimension as item1
from . import gravity_item2_shape_anisotropy as item2_v1
from . import gravity_item2_wise_multipoles as wise
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-roadmap-item2-wise-multipoles-receipt-1.0"
CONFIG_PATH = wise.CONFIG_PATH
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item2_wise_multipole_experiment.py"
TEST_PATH = "tests/test_gravity_item2_wise_multipoles.py"
OUTPUT_PATH = "runs/gravity/roadmap/item-02-wise-multipoles-v2.json"
DOMAINS = ("galaxy", "cluster")


class GravityItem2WiseMultipoleExperimentError(ValueError):
    """The Item 2 WISE experiment contract, data, or result changed."""


def _verify_envelope(value: Mapping[str, Any], *, label: str) -> None:
    body = dict(value)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2WiseMultipoleExperimentError(f"{label} content seal changed")


def load_config(root: Path) -> Mapping[str, Any]:
    """Validate the complete preregistration and sealed target-blind extraction."""

    root = root.resolve()
    config = wise.load_extraction_config(root)
    manifest = wise.validate_extraction(root)
    external = config.get("external_validation_gate", {})
    if external != {
        "primary_statistic": "quadrupole_vs_s4g_family_spearman",
        "minimum_quality_matched_galaxies": 20,
        "required_sign": "positive",
        "bar_ellipticity_statistic_role": (
            "secondary because fewer than 20 eligible quality-passing galaxies have a "
            "published usable bar ellipticity"
        ),
    }:
        raise GravityItem2WiseMultipoleExperimentError("external validation gate changed")
    if config.get("output") != OUTPUT_PATH:
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE output changed")
    population = config.get("population", {})
    if (
        population.get("expected_acquired_objects") != 103
        or population.get("minimum_model_objects") != 60
        or manifest.get("counts", {}).get("images") != 83
    ):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE population changed")
    admission = config.get("admission", {})
    required = {
        "all_target_blind_image_cutouts_and_quality_flags_complete",
        "published_s4g_bar_validation_positive",
        "minimum_intermediate_bar_like_population_present",
        "universal_selector_must_choose_a_qualifying_model_in_every_fold",
        "universal_model_must_beat_constant_observational_score_in_each_population",
        "universal_beta_r2_must_be_positive_in_each_population",
        "universal_model_must_beat_support_proxy_beta_mse_in_each_population",
        "universal_model_must_beat_support_proxy_in_energy_overlap_for_each_population",
        "universal_beta_r2_must_be_positive_in_intermediate_bar_like_subpopulation",
        "population_proxy_models_may_not_qualify",
        "confirmation_or_direct_lensing_required_for_novel_physics_claim",
    }
    if set(admission) != required or any(admission[key] is not True for key in required):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE admission changed")
    return config


def _item1_labels(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    path = root / str(config["predecessor_bindings"]["item1"]["path"])
    receipt = _load_json(path)
    item1.validate_receipt(receipt, root=root)
    labels = {
        f"{row['domain']}:{row['name']}": {
            "at_grid_boundary": bool(row["oracle_beta_at_grid_boundary"]),
            "beta": float(row["oracle_beta_target_derived"]),
        }
        for row in receipt["per_object_diagnostics"]
    }
    if len(labels) != 159:
        raise GravityItem2WiseMultipoleExperimentError("Item 1 label population changed")
    return labels


def _feature_rows(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    path = root / str(config["sources"]["unwise_w1"]["derived_feature_path"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 83 or len({row["name"] for row in rows}) != 83:
        raise GravityItem2WiseMultipoleExperimentError("unWISE feature population changed")
    return {str(row["name"]): row for row in rows}


def _derived_features(
    concentration: float,
    centroid_shift: float,
    quadrupole: float,
    m3: float,
    m4: float,
    support_dimension: float,
) -> dict[str, float]:
    energy = float(np.sqrt(quadrupole**2 + m3**2 + m4**2))
    result = {
        "centroid_shift": centroid_shift,
        "concentration_c20": concentration,
        "concentration_times_energy": concentration * energy,
        "log1p_multipole_energy_over_0p05": float(np.log1p(energy / 0.05)),
        "m3_aperture_amplitude": m3,
        "m4_aperture_amplitude": m4,
        "multipole_energy": energy,
        "multipole_energy_squared": energy**2,
        "quadrupole_amplitude": quadrupole,
        "support_dimension": support_dimension,
    }
    if any(not np.isfinite(value) for value in result.values()):
        raise GravityItem2WiseMultipoleExperimentError("non-finite multipole feature")
    if (
        not 0.0 <= concentration <= 1.0
        or centroid_shift < 0.0
        or not 0.0 <= quadrupole <= 1.0
        or min(m3, m4) < 0.0
    ):
        raise GravityItem2WiseMultipoleExperimentError("nonphysical multipole feature")
    return result


def cluster_multipole_features(morphology: Mapping[str, Any]) -> dict[str, float]:
    """Convert published CLASH axis and power ratios to the frozen common grammar."""

    axis_ratio = float(morphology["axis_ratio"])
    quadrupole = (1.0 - axis_ratio**2) / (1.0 + axis_ratio**2)
    aperture_log = abs(float(np.log(500.0)))
    m3 = float(np.sqrt(18.0 * float(morphology["p30"])) * aperture_log)
    m4 = float(np.sqrt(32.0 * float(morphology["p40"])) * aperture_log)
    return _derived_features(
        float(morphology["concentration"]),
        float(morphology["centroid_shift"]),
        quadrupole,
        m3,
        m4,
        3.0,
    )


def prepare_multipole_objects(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Mapping[str, Any]]:
    """Join the sealed labels only after all target-blind features have been frozen."""

    root = root.resolve()
    manifest = wise.validate_extraction(root)
    feature_rows = _feature_rows(root, config)
    usable = {
        name: row for name, row in feature_rows.items() if row["image_quality_pass"] == "True"
    }
    if len(usable) != int(manifest["counts"]["quality_pass_galaxies"]):
        raise GravityItem2WiseMultipoleExperimentError("unWISE quality subset changed")

    cluster_path = root / str(config["sources"]["clash_xray_morphology"]["path"])
    morphology_source = item2_v1.parse_donahue_morphology(cluster_path)
    morphology = {
        target: morphology_source[source_name]
        for source_name, target in item2_v1.DONAHUE_TO_TARGET.items()
    }
    if len(morphology) != 20:
        raise GravityItem2WiseMultipoleExperimentError("CLASH morphology match changed")

    base_objects = item1.prepare_objects(root, item1.load_config(root))
    labels = _item1_labels(root, config)
    objects: list[dict[str, Any]] = []
    for base in base_objects:
        name = str(base["name"])
        if base["domain"] == "galaxy":
            image = usable.get(name)
            if image is None:
                continue
            features = _derived_features(
                float(image["concentration_c20"]),
                float(image["centroid_shift"]),
                float(image["quadrupole_amplitude"]),
                float(image["m3_aperture_amplitude"]),
                float(image["m4_aperture_amplitude"]),
                2.0,
            )
            provenance = {
                "image_quality_pass": True,
                "image_sha256": image["image_sha256"],
                "image_version": image["image_version"],
                "shape_tracer": "unWISE_NEO11_W1_3p4_micron_intensity",
            }
        else:
            features = cluster_multipole_features(morphology[name])
            provenance = {
                "aperture_kpc": 500,
                "shape_tracer": "Chandra_Xray_surface_brightness",
                "source_name": morphology[name]["source_name"],
            }
        objects.append({**base, "features": features, "shape_provenance": provenance})

    object_keys = {str(row["key"]) for row in objects}
    subset_labels = {key: labels[key] for key in sorted(object_keys)}
    galaxy_count = sum(row["domain"] == "galaxy" for row in objects)
    cluster_count = sum(row["domain"] == "cluster" for row in objects)
    if (
        galaxy_count != len(usable)
        or galaxy_count < int(config["population"]["minimum_quality_wise_galaxies"])
        or cluster_count != int(config["population"]["expected_clash_clusters"])
        or len(objects) < int(config["population"]["minimum_model_objects"])
        or len(object_keys) != len(objects)
        or set(subset_labels) != object_keys
    ):
        raise GravityItem2WiseMultipoleExperimentError("model population gate changed")
    return objects, subset_labels, manifest


def _selection_counts(ledger: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    selected = [str(row["selected_model_id"]) for row in ledger]
    return {model: selected.count(model) for model in sorted(set(selected))}


def _feature_ranges(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    features = (
        "concentration_c20",
        "centroid_shift",
        "quadrupole_amplitude",
        "m3_aperture_amplitude",
        "m4_aperture_amplitude",
        "multipole_energy",
    )
    return {
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
            for feature in features
        }
        for domain in DOMAINS
    }


def _single_group_metrics(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, float],
) -> dict[str, Any]:
    target = np.asarray([float(labels[str(row["key"])]["beta"]) for row in objects])
    predicted = np.asarray([float(predictions[str(row["key"])]) for row in objects])
    residual = predicted - target
    centered = target - float(np.mean(target))
    sse = float(residual @ residual)
    sst = float(centered @ centered)
    return {
        "mean_absolute_error": _metric(float(np.mean(np.abs(residual)))),
        "mean_squared_error": _metric(float(np.mean(residual**2))),
        "objects": len(objects),
        "r2": _metric(1.0 - sse / sst) if sst > 1.0e-15 else "0.000000000000e+00",
    }


def _overlap_diagnostic(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    predictions: Mapping[str, float],
    proxy_predictions: Mapping[str, float],
) -> dict[str, Any]:
    ranges = {
        domain: (
            min(
                float(row["features"]["multipole_energy"])
                for row in objects
                if row["domain"] == domain
            ),
            max(
                float(row["features"]["multipole_energy"])
                for row in objects
                if row["domain"] == domain
            ),
        )
        for domain in DOMAINS
    }
    lower = max(ranges[domain][0] for domain in DOMAINS)
    upper = min(ranges[domain][1] for domain in DOMAINS)
    members = [
        row for row in objects if lower <= float(row["features"]["multipole_energy"]) <= upper
    ]
    counts = {domain: sum(row["domain"] == domain for row in members) for domain in DOMAINS}
    if lower > upper or any(counts[domain] == 0 for domain in DOMAINS):
        return {
            "by_population": {},
            "multipole_energy_interval": [_metric(lower), _metric(upper)],
            "observed_overlap": False,
        }
    universal = item1._beta_metrics(members, labels, predictions)
    proxy = item1._beta_metrics(members, labels, proxy_predictions)
    return {
        "by_population": {
            domain: {
                "objects": counts[domain],
                "support_proxy_beta_mse": proxy["by_population"][domain]["mean_squared_error"],
                "universal_beta_mse": universal["by_population"][domain]["mean_squared_error"],
            }
            for domain in DOMAINS
        },
        "multipole_energy_interval": [_metric(lower), _metric(upper)],
        "observed_overlap": True,
    }


def _public_object(
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    beta: float,
    selected_model: str,
) -> dict[str, Any]:
    return {
        "domain": row["domain"],
        "features": {key: _metric(value) for key, value in sorted(row["features"].items())},
        "name": row["name"],
        "oracle_beta_at_grid_boundary": label["at_grid_boundary"],
        "oracle_beta_target_derived": _metric(label["beta"]),
        "point_count": row["point_count"],
        "shape_provenance": row["shape_provenance"],
        "universal_multipole_oof_beta": _metric(beta),
        "universal_multipole_selected": selected_model,
    }


def _source_paths(config: Mapping[str, Any]) -> dict[str, str]:
    result = {
        "config": CONFIG_PATH,
        "experiment_source": SOURCE_PATH,
        "extractor_source": wise.SOURCE_PATH,
        "item1_predecessor": str(config["predecessor_bindings"]["item1"]["path"]),
        "item2_attempt1": str(config["predecessor_bindings"]["item2_attempt1"]["path"]),
        "roadmap": str(config["roadmap_binding"]["path"]),
        "sparc_exploration_membership": str(
            config["sources"]["sparc_exploration_membership"]["path"]
        ),
        "sparc_global_properties": str(config["sources"]["sparc_global_properties"]["path"]),
        "clash_xray_morphology": str(config["sources"]["clash_xray_morphology"]["path"]),
        "unwise_manifest": str(config["sources"]["unwise_w1"]["image_manifest_path"]),
        "unwise_multipoles": str(config["sources"]["unwise_w1"]["derived_feature_path"]),
        "test": TEST_PATH,
    }
    for index, binding in enumerate(config["sources"]["s4g_external_validation"]["files"]):
        result[f"s4g_{index + 1}"] = str(binding["path"])
    return result


def build_receipt(root: Path) -> dict[str, Any]:
    """Run the fixed nested whole-object experiment and seal every failed gate."""

    root = root.resolve()
    config = load_config(root)
    objects, labels, manifest = prepare_multipole_objects(root, config)
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(
        objects, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"])
    )
    models = [dict(row) for row in config["models"]]
    qualifying = [dict(row) for row in models if row["qualifying"] is True]
    universal_models = [next(row for row in models if row["id"] == "constant"), *qualifying]

    model_results: dict[str, Any] = {}
    model_predictions: dict[str, dict[str, float]] = {}
    for model in models:
        predictions, ledger, clipped = item1._model_oof(objects, labels, assignments, model, config)
        model_id = str(model["id"])
        model_predictions[model_id] = predictions
        model_results[model_id] = {
            "clip_count": clipped,
            "coefficient_prediction": item1._beta_metrics(objects, labels, predictions),
            "fold_ledger": ledger,
            "origin_label": model["origin_label"],
            "qualifying_universal_multipole_model": bool(model["qualifying"]),
            "score": item1._observational_score(objects, predictions),
        }

    predictions, nested_ledger, nested_clipped = item1._nested_model_oof(
        objects, labels, assignments, universal_models, config
    )
    all_predictions, all_ledger, all_clipped = item1._nested_model_oof(
        objects, labels, assignments, models, config
    )
    beta_metrics = item1._beta_metrics(objects, labels, predictions)
    score = item1._observational_score(objects, predictions)
    all_beta_metrics = item1._beta_metrics(objects, labels, all_predictions)
    all_score = item1._observational_score(objects, all_predictions)
    constant_score = model_results["constant"]["score"]
    proxy_result = model_results["linear_support_dimension_proxy"]
    proxy_predictions = model_predictions["linear_support_dimension_proxy"]

    overlap = _overlap_diagnostic(objects, labels, predictions, proxy_predictions)
    minimum_overlap = int(cv["multipole_energy_overlap_minimum_objects_per_population"])
    enough_overlap = bool(overlap["observed_overlap"]) and all(
        int(overlap["by_population"][domain]["objects"]) >= minimum_overlap for domain in DOMAINS
    )
    overlap_beats_proxy = enough_overlap and all(
        float(overlap["by_population"][domain]["universal_beta_mse"])
        < float(overlap["by_population"][domain]["support_proxy_beta_mse"])
        for domain in DOMAINS
    )

    threshold = float(config["population"]["intermediate_bar_like_quadrupole_threshold"])
    intermediate = [
        row
        for row in objects
        if row["domain"] == "galaxy" and float(row["features"]["quadrupole_amplitude"]) >= threshold
    ]
    intermediate_metrics = _single_group_metrics(intermediate, labels, predictions)
    external_gate = config["external_validation_gate"]
    external_statistic = manifest["external_validation"][external_gate["primary_statistic"]]
    external_matches = int(manifest["counts"]["s4g_family_matches"])
    external_positive = (
        external_statistic is not None
        and external_matches >= int(external_gate["minimum_quality_matched_galaxies"])
        and float(external_statistic) > 0.0
    )
    qualifying_ids = {str(row["id"]) for row in qualifying}
    selected = [str(row["selected_model_id"]) for row in nested_ledger]
    population_proxy_excluded = all(
        row["qualifying"] is False
        for row in models
        if row["id"] in {"linear_support_dimension_proxy", "support_plus_all_multipoles"}
    )
    gate_checks = {
        "all_target_blind_image_cutouts_and_quality_flags_complete": (
            int(manifest["counts"]["images"]) == 83
            and int(manifest["counts"]["quality_pass_galaxies"])
            >= int(config["population"]["minimum_quality_wise_galaxies"])
            and int(manifest["counts"]["target_fields_used_by_feature_computation"]) == 0
        ),
        "confirmation_or_direct_lensing_completed": False,
        "minimum_intermediate_bar_like_population_present": (
            len(intermediate) >= int(config["population"]["minimum_intermediate_bar_like_galaxies"])
        ),
        "multipole_energy_overlap_contains_minimum_objects_in_each_population": enough_overlap,
        "population_proxy_models_excluded_from_universal_admission": population_proxy_excluded,
        "published_s4g_bar_validation_positive": external_positive,
        "target_fields_absent_from_multipole_feature_computation": True,
        "universal_beta_r2_positive_in_each_population": all(
            float(beta_metrics["by_population"][domain]["r2"]) > 0.0 for domain in DOMAINS
        ),
        "universal_beta_r2_positive_in_intermediate_bar_like_subpopulation": (
            float(intermediate_metrics["r2"]) > 0.0
        ),
        "universal_model_beats_constant_observational_score_in_each_population": all(
            float(score["by_population"][domain]["chi_square"])
            < float(constant_score["by_population"][domain]["chi_square"])
            for domain in DOMAINS
        ),
        "universal_model_beats_support_proxy_beta_mse_in_each_population": all(
            float(beta_metrics["by_population"][domain]["mean_squared_error"])
            < float(
                proxy_result["coefficient_prediction"]["by_population"][domain][
                    "mean_squared_error"
                ]
            )
            for domain in DOMAINS
        ),
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population": (
            overlap_beats_proxy
        ),
        "universal_selector_chooses_qualifying_model_in_every_fold": all(
            model_id in qualifying_ids for model_id in selected
        ),
        "whole_object_target_blind_outer_predictions_complete": (
            len(predictions) == len(objects) == len(all_predictions)
        ),
    }
    required_gate_keys = (
        "all_target_blind_image_cutouts_and_quality_flags_complete",
        "minimum_intermediate_bar_like_population_present",
        "multipole_energy_overlap_contains_minimum_objects_in_each_population",
        "population_proxy_models_excluded_from_universal_admission",
        "published_s4g_bar_validation_positive",
        "target_fields_absent_from_multipole_feature_computation",
        "universal_beta_r2_positive_in_each_population",
        "universal_beta_r2_positive_in_intermediate_bar_like_subpopulation",
        "universal_model_beats_constant_observational_score_in_each_population",
        "universal_model_beats_support_proxy_beta_mse_in_each_population",
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population",
        "universal_selector_chooses_qualifying_model_in_every_fold",
        "whole_object_target_blind_outer_predictions_complete",
    )
    development_pass = all(gate_checks[key] is True for key in required_gate_keys)
    selected_by_fold = {int(row["fold"]): str(row["selected_model_id"]) for row in nested_ledger}
    per_object = [
        _public_object(
            row,
            labels[str(row["key"])],
            predictions[str(row["key"])],
            selected_by_fold[assignments[str(row["key"])]],
        )
        for row in objects
    ]
    source_paths = _source_paths(config)
    source_bindings = {key: _binding(root, path) for key, path in source_paths.items()}
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "GRAVITY_ROADMAP_ITEM_02_SHAPE_ANISOTROPY_SECOND_ATTEMPT",
        "decision": (
            "PASS_ITEM2_WISE_MULTIPOLE_DEVELOPMENT_GATE"
            if development_pass
            else "INCONCLUSIVE_ITEM2_WISE_MULTIPOLES"
        ),
        "claims": {
            "alternative_to_gr_established": False,
            "direct_lensing_test_completed": False,
            "historical_novelty_established": False,
            "intrinsic_shape_cause_established": False,
            "roadmap_item_2_complete": development_pass,
            "sequential_G6_G7_G8_advanced": False,
            "sparc_confirmation_opened": False,
            "universal_multipole_shape_predicts_cross_scale_response": development_pass,
            "wise_w1_is_pure_baryonic_mass_map": False,
        },
        "config": {
            "content_sha256": canonical_sha256(wise._sealable_contract(config)),
            "file_sha256": _file_sha256(root / CONFIG_PATH),
            "path": CONFIG_PATH,
        },
        "counts": {
            "clash_clusters": sum(row["domain"] == "cluster" for row in objects),
            "clash_points": sum(
                row["point_count"] for row in objects if row["domain"] == "cluster"
            ),
            "direct_lensing_likelihood_evaluations": 0,
            "formula_classes": len(models) + 2,
            "intermediate_bar_like_galaxies": len(intermediate),
            "models": len(models),
            "outer_folds": int(cv["outer_folds"]),
            "paid_model_calls": 0,
            "sparc_confirmation_evaluator_accesses": 0,
            "unwise_acquired_galaxies": int(manifest["counts"]["images"]),
            "unwise_model_galaxies": sum(row["domain"] == "galaxy" for row in objects),
            "unwise_model_galaxy_points": sum(
                row["point_count"] for row in objects if row["domain"] == "galaxy"
            ),
            "unique_formula_prediction_classes": len(
                {result["score"]["prediction_manifest_sha256"] for result in model_results.values()}
                | {
                    score["prediction_manifest_sha256"],
                    all_score["prediction_manifest_sha256"],
                }
            ),
        },
        "data_lineage": {
            "clash_dynamics_target": "model_dependent_lensing_derived_acceleration_diagnostic",
            "cluster_shape": "published_Chandra_Xray_morphology_at_500_kpc",
            "galaxy_shape": "target_blind_unWISE_NEO11_W1_multipoles",
            "image_quality_filter_role": "frozen_before_Item1_beta_labels_were_joined",
            "target_derived_oracle_role": (
                "sealed_Item1_training_label_and_descriptive_ceiling_never_formula_input"
            ),
        },
        "external_s4g_validation": {
            **manifest["external_validation"],
            "primary_matches": external_matches,
            "primary_statistic": external_gate["primary_statistic"],
            "required_sign": external_gate["required_sign"],
        },
        "feature_ranges": _feature_ranges(objects),
        "gate_checks": gate_checks,
        "intermediate_bar_like_diagnostic": {
            "definition": f"galaxy quadrupole_amplitude >= {threshold}",
            "coefficient_prediction": intermediate_metrics,
        },
        "model_results": model_results,
        "multipole_energy_overlap_diagnostic": overlap,
        "nested_all_models": {
            "clip_count": all_clipped,
            "coefficient_prediction": all_beta_metrics,
            "fold_ledger": all_ledger,
            "score": all_score,
            "selected_model_counts": _selection_counts(all_ledger),
        },
        "nested_universal_multipoles": {
            "clip_count": nested_clipped,
            "coefficient_prediction": beta_metrics,
            "fold_ledger": nested_ledger,
            "score": score,
            "selected_model_counts": _selection_counts(nested_ledger),
        },
        "per_object_diagnostics": per_object,
        "specific_hypothesis_results": {
            "common_multipole_grammar": (
                "SURVIVES_DEVELOPMENT_GATE"
                if development_pass
                else "NOT_SHOWN_TO_GENERATE_THE_CROSS_SCALE_RESPONSE"
            ),
            "population_proxy_plus_multipoles": (
                "DIAGNOSTIC_ONLY_NOT_A_UNIVERSAL_FIRST_PRINCIPLES_RULE"
            ),
            "target_blind_w1_extraction": "COMPLETE_FOR_ALL_83_ELIGIBLE_GALAXIES",
        },
        "limitations": [
            "unWISE W1 intensity traces old stars imperfectly and is neither a complete baryonic mass map nor an intrinsic three-dimensional shape measurement.",
            "The frozen quality gate removes unusable images without targets, but foreground stars and source confusion can remain in quality-passing cutouts.",
            "Galaxy features use deprojected 3.4-micron intensity while cluster features use projected X-ray emissivity and catalog power-ratio conversions; equality of those tracers is not established.",
            "The S4G family validation is an external morphology check, not a gravity confirmation set.",
            "The CLASH coefficient target is reconstructed through spherical NFW posteriors and is not a direct lensing likelihood.",
            "Per-object beta labels are target-derived development labels and cannot themselves qualify as formulas or first principles.",
            "No model in this attempt is historically novel or independently confirmed.",
        ],
        "next_action": (
            "Retain every failed morphology region and remain on Item 2 unless all gates pass; "
            "a further attempt must use a more comparable baryonic tracer or an explicit "
            "intermediate/filamentary population without weakening this receipt."
        ),
        "source_bindings": source_bindings,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Reject altered evidence, proxy admission, overclaims, or gate/decision drift."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE receipt schema changed")
    _verify_envelope(receipt, label="Item 2 WISE receipt")
    config = load_config(root)
    expected_config = {
        "content_sha256": canonical_sha256(wise._sealable_contract(config)),
        "file_sha256": _file_sha256(root / CONFIG_PATH),
        "path": CONFIG_PATH,
    }
    if receipt.get("config") != expected_config:
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE config binding changed")
    expected_bindings = {key: _binding(root, path) for key, path in _source_paths(config).items()}
    if receipt.get("source_bindings") != expected_bindings:
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE source binding changed")
    claims = receipt.get("claims", {})
    forbidden_claims = (
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
        "wise_w1_is_pure_baryonic_mass_map",
    )
    if any(claims.get(key) is not False for key in forbidden_claims):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE receipt overstates a claim")
    counts = receipt.get("counts", {})
    if (
        counts.get("paid_model_calls") != 0
        or counts.get("sparc_confirmation_evaluator_accesses") != 0
        or counts.get("direct_lensing_likelihood_evaluations") != 0
        or counts.get("unwise_acquired_galaxies") != 83
        or counts.get("clash_clusters") != 20
    ):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE access count changed")
    for model_id in ("linear_support_dimension_proxy", "support_plus_all_multipoles"):
        if (
            receipt.get("model_results", {})
            .get(model_id, {})
            .get("qualifying_universal_multipole_model")
            is not False
        ):
            raise GravityItem2WiseMultipoleExperimentError("population proxy entered admission")
    gate_checks = receipt.get("gate_checks", {})
    required_gate_keys = (
        "all_target_blind_image_cutouts_and_quality_flags_complete",
        "minimum_intermediate_bar_like_population_present",
        "multipole_energy_overlap_contains_minimum_objects_in_each_population",
        "population_proxy_models_excluded_from_universal_admission",
        "published_s4g_bar_validation_positive",
        "target_fields_absent_from_multipole_feature_computation",
        "universal_beta_r2_positive_in_each_population",
        "universal_beta_r2_positive_in_intermediate_bar_like_subpopulation",
        "universal_model_beats_constant_observational_score_in_each_population",
        "universal_model_beats_support_proxy_beta_mse_in_each_population",
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population",
        "universal_selector_chooses_qualifying_model_in_every_fold",
        "whole_object_target_blind_outer_predictions_complete",
    )
    passed = all(gate_checks.get(key) is True for key in required_gate_keys)
    expected_decision = (
        "PASS_ITEM2_WISE_MULTIPOLE_DEVELOPMENT_GATE"
        if passed
        else "INCONCLUSIVE_ITEM2_WISE_MULTIPOLES"
    )
    if receipt.get("decision") != expected_decision:
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE decision disagrees with gates")
    if (
        claims.get("roadmap_item_2_complete") is not passed
        or claims.get("universal_multipole_shape_predicts_cross_scale_response") is not passed
    ):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE claim disagrees with gates")
    if gate_checks.get("confirmation_or_direct_lensing_completed") is not False:
        raise GravityItem2WiseMultipoleExperimentError("unavailable confirmation was promoted")
    if len(receipt.get("per_object_diagnostics", ())) != counts.get(
        "unwise_model_galaxies", 0
    ) + counts.get("clash_clusters", 0):
        raise GravityItem2WiseMultipoleExperimentError("Item 2 WISE diagnostics changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityItem2WiseMultipoleExperimentError(
                f"refusing to overwrite immutable Item 2 WISE receipt: {path}"
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
    "GravityItem2WiseMultipoleExperimentError",
    "build_receipt",
    "cluster_multipole_features",
    "load_config",
    "main",
    "prepare_multipole_objects",
    "validate_receipt",
]
