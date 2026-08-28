"""Run roadmap Item 2 attempt 3 with comparable stellar morphology tracers."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import gravity_item1_effective_dimension as item1
from . import gravity_item2_clash_stellar_multipoles as stellar
from . import gravity_item2_shape_anisotropy as item2_v1
from . import gravity_item2_wise_multipole_experiment as v2_experiment
from .gravity_g1_pilot import _binding, _file_sha256, _load_json
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-roadmap-item2-stellar-multipoles-receipt-1.0"
CONFIG_PATH = stellar.CONFIG_PATH
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item2_stellar_multipole_experiment.py"
TEST_PATH = "tests/test_gravity_item2_stellar_multipoles.py"
OUTPUT_PATH = "runs/gravity/roadmap/item-02-stellar-multipoles-v3.json"
DOMAINS = ("galaxy", "cluster")


class GravityItem2StellarMultipoleExperimentError(ValueError):
    """The third Item 2 experiment contract, data, or result changed."""


def _verify_envelope(value: Mapping[str, Any], *, label: str) -> None:
    body = dict(value)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2StellarMultipoleExperimentError(f"{label} content seal changed")


def _sealable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sealable(child) for key, child in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sealable(child) for child in value]
    if isinstance(value, float):
        return item1._metric(value)
    return value


def load_config(root: Path) -> Mapping[str, Any]:
    """Validate the full frozen contract only after the target-blind manifest passed."""

    root = root.resolve()
    config = stellar.load_config(root)
    manifest = stellar.validate_extraction(root)
    if manifest.get("decision") != "PASS_TARGET_BLIND_REPRESENTATION_GATE":
        raise GravityItem2StellarMultipoleExperimentError("representation gate did not pass")
    for binding in config.get("predecessor_bindings", {}).values():
        path = root / str(binding.get("path"))
        if _file_sha256(path) != binding.get("file_sha256"):
            raise GravityItem2StellarMultipoleExperimentError("predecessor file changed")
        receipt = _load_json(path)
        _verify_envelope(receipt, label="predecessor")
        if receipt.get("content_sha256") != binding.get("content_sha256") or receipt.get(
            "decision"
        ) != binding.get("required_decision"):
            raise GravityItem2StellarMultipoleExperimentError("predecessor content changed")
    expected_models = [
        "constant",
        "linear_concentration",
        "linear_centroid_shift",
        "linear_quadrupole",
        "linear_m3",
        "linear_m4",
        "linear_multipole_energy",
        "quadratic_multipole_energy",
        "log_multipole_energy",
        "concentration_plus_energy",
        "energy_concentration_interaction",
        "all_multipoles",
        "linear_support_dimension_proxy",
        "support_plus_all_multipoles",
    ]
    models = config.get("models", ())
    if [row.get("id") for row in models] != expected_models:
        raise GravityItem2StellarMultipoleExperimentError("model grammar changed")
    if any(
        row.get("qualifying") is not False
        for row in models
        if row.get("id")
        in {"constant", "linear_support_dimension_proxy", "support_plus_all_multipoles"}
    ):
        raise GravityItem2StellarMultipoleExperimentError("proxy entered admission")
    cv = config.get("cross_validation", {})
    if (
        cv.get("outer_folds") != 5
        or cv.get("fold_unit") != "whole_object_stratified_by_population"
        or cv.get("heldout_object_target_available_to_fit_or_selection") is not False
    ):
        raise GravityItem2StellarMultipoleExperimentError("cross-validation changed")
    if config.get("output") != OUTPUT_PATH:
        raise GravityItem2StellarMultipoleExperimentError("output path changed")
    if config.get("claim_boundaries", {}).get("alternative_to_gr_established") is not False:
        raise GravityItem2StellarMultipoleExperimentError("contract overstates its claim")
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
        raise GravityItem2StellarMultipoleExperimentError("Item 1 label population changed")
    return labels


def _galaxy_rows(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    path = root / str(config["sources"]["galaxy_unwise_features"]["path"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    usable = {row["name"]: row for row in rows if row["image_quality_pass"] == "True"}
    if len(rows) != 83 or len(usable) != 68:
        raise GravityItem2StellarMultipoleExperimentError("galaxy morphology population changed")
    return usable


def _cluster_rows(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    manifest = stellar.validate_extraction(root)
    path = root / str(manifest["feature_file"]["path"])
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    primary_power = float(config["target_blind_extraction"]["primary_weight_power"])
    primary = [row for row in rows if float(row["weight_power"]) == primary_power]
    if len(rows) != 60 or len(primary) != 20 or len({row["slug"] for row in primary}) != 20:
        raise GravityItem2StellarMultipoleExperimentError("cluster morphology population changed")
    return {item2_v1.DONAHUE_TO_TARGET[stellar.XRAY_NAME[row["slug"]]]: row for row in primary}


def prepare_objects(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], Mapping[str, Any]]:
    """Join labels only after both stellar feature sources and their gates are sealed."""

    root = root.resolve()
    manifest = stellar.validate_extraction(root)
    galaxies = _galaxy_rows(root, config)
    clusters = _cluster_rows(root, config)
    labels = _item1_labels(root, config)
    base_objects = item1.prepare_objects(root, item1.load_config(root))
    objects = []
    for base in base_objects:
        name = str(base["name"])
        if base["domain"] == "galaxy":
            source = galaxies.get(name)
            if source is None:
                continue
            features = v2_experiment._derived_features(
                float(source["concentration_c20"]),
                float(source["centroid_shift"]),
                float(source["quadrupole_amplitude"]),
                float(source["m3_aperture_amplitude"]),
                float(source["m4_aperture_amplitude"]),
                2.0,
            )
            provenance = {
                "shape_tracer": "unWISE_NEO11_W1_3p4_micron_old_stellar_light",
                "image_sha256": source["image_sha256"],
                "image_quality_pass": True,
            }
        else:
            source = clusters.get(name)
            if source is None:
                raise GravityItem2StellarMultipoleExperimentError(
                    f"missing cluster stellar morphology for {name}"
                )
            features = v2_experiment._derived_features(
                float(source["concentration_c20"]),
                float(source["centroid_shift"]),
                float(source["quadrupole_amplitude"]),
                float(source["m3_aperture_amplitude"]),
                float(source["m4_aperture_amplitude"]),
                3.0,
            )
            provenance = {
                "shape_tracer": "CLASH_HST_photoz_member_non_lensing_corrected_stellar_mass",
                "slug": source["slug"],
                "aperture_kpc": 150,
                "weight_power": 1,
                "member_count_including_bcg": int(float(source["member_count_including_bcg"])),
                "effective_member_count": item1._metric(float(source["effective_member_count"])),
            }
        objects.append({**base, "features": features, "shape_provenance": provenance})
    keys = {str(row["key"]) for row in objects}
    subset_labels = {key: labels[key] for key in sorted(keys)}
    if (
        len(objects) != 88
        or len(keys) != 88
        or sum(row["domain"] == "galaxy" for row in objects) != 68
        or sum(row["domain"] == "cluster" for row in objects) != 20
        or set(subset_labels) != keys
    ):
        raise GravityItem2StellarMultipoleExperimentError("joined population changed")
    return objects, subset_labels, manifest


def _source_paths(config: Mapping[str, Any]) -> dict[str, str]:
    return {
        "config": CONFIG_PATH,
        "experiment_source": SOURCE_PATH,
        "extractor_source": stellar.SOURCE_PATH,
        "test": TEST_PATH,
        "roadmap": str(config["roadmap_binding"]["path"]),
        "item1_predecessor": str(config["predecessor_bindings"]["item1"]["path"]),
        "item2_attempt1": str(config["predecessor_bindings"]["item2_attempt1"]["path"]),
        "item2_attempt2": str(config["predecessor_bindings"]["item2_attempt2"]["path"]),
        "galaxy_unwise_features": str(config["sources"]["galaxy_unwise_features"]["path"]),
        "galaxy_unwise_manifest": str(config["sources"]["galaxy_unwise_features"]["manifest_path"]),
        "clash_stellar_features": stellar.FEATURE_PATH,
        "clash_stellar_manifest": stellar.MANIFEST_PATH,
        "clash_bcg_baryons": str(config["sources"]["clash_bcg_baryons"]["path"]),
        "clash_xray_validation": str(config["sources"]["clash_xray_morphology"]["path"]),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Run the frozen whole-object model comparison and seal every failed gate."""

    root = root.resolve()
    config = load_config(root)
    objects, labels, representation = prepare_objects(root, config)
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(
        objects, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"])
    )
    models = [dict(row) for row in config["models"]]
    qualifying = [dict(row) for row in models if row["qualifying"] is True]
    nested_models = [next(row for row in models if row["id"] == "constant"), *qualifying]

    model_results = {}
    model_predictions = {}
    for model in models:
        predictions, ledger, clipped = item1._model_oof(objects, labels, assignments, model, config)
        model_id = str(model["id"])
        model_predictions[model_id] = predictions
        model_results[model_id] = {
            "clip_count": clipped,
            "coefficient_prediction": item1._beta_metrics(objects, labels, predictions),
            "fold_ledger": ledger,
            "origin_label": model["origin_label"],
            "qualifying_universal_stellar_multipole_model": bool(model["qualifying"]),
            "score": item1._observational_score(objects, predictions),
        }

    predictions, nested_ledger, nested_clipped = item1._nested_model_oof(
        objects, labels, assignments, nested_models, config
    )
    beta_metrics = item1._beta_metrics(objects, labels, predictions)
    score = item1._observational_score(objects, predictions)
    constant_score = model_results["constant"]["score"]
    proxy = model_results["linear_support_dimension_proxy"]
    proxy_predictions = model_predictions["linear_support_dimension_proxy"]
    overlap = v2_experiment._overlap_diagnostic(objects, labels, predictions, proxy_predictions)
    minimum_overlap = int(cv["multipole_energy_overlap_minimum_objects_per_population"])
    enough_overlap = bool(overlap["observed_overlap"]) and all(
        int(overlap["by_population"][domain]["objects"]) >= minimum_overlap for domain in DOMAINS
    )
    overlap_beats_proxy = enough_overlap and all(
        float(overlap["by_population"][domain]["universal_beta_mse"])
        < float(overlap["by_population"][domain]["support_proxy_beta_mse"])
        for domain in DOMAINS
    )
    qualifying_ids = {str(row["id"]) for row in qualifying}
    selected_ids = [str(row["selected_model_id"]) for row in nested_ledger]
    gate_checks = {
        "target_blind_representation_validation_passed": (
            representation["decision"] == "PASS_TARGET_BLIND_REPRESENTATION_GATE"
            and all(representation["checks"].values())
        ),
        "population_proxy_models_excluded_from_universal_admission": all(
            row["qualifying"] is False
            for row in models
            if row["id"] in {"linear_support_dimension_proxy", "support_plus_all_multipoles"}
        ),
        "multipole_energy_overlap_contains_minimum_objects_in_each_population": enough_overlap,
        "universal_selector_chooses_qualifying_model_in_every_fold": all(
            model_id in qualifying_ids for model_id in selected_ids
        ),
        "universal_model_beats_constant_observational_score_in_each_population": all(
            float(score["by_population"][domain]["chi_square"])
            < float(constant_score["by_population"][domain]["chi_square"])
            for domain in DOMAINS
        ),
        "universal_beta_r2_positive_in_each_population": all(
            float(beta_metrics["by_population"][domain]["r2"]) > 0.0 for domain in DOMAINS
        ),
        "universal_model_beats_support_proxy_beta_mse_in_each_population": all(
            float(beta_metrics["by_population"][domain]["mean_squared_error"])
            < float(proxy["coefficient_prediction"]["by_population"][domain]["mean_squared_error"])
            for domain in DOMAINS
        ),
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population": (
            overlap_beats_proxy
        ),
        "whole_object_outer_predictions_complete": len(predictions) == len(objects),
        "confirmation_or_direct_lensing_completed": False,
    }
    required = (
        "target_blind_representation_validation_passed",
        "population_proxy_models_excluded_from_universal_admission",
        "multipole_energy_overlap_contains_minimum_objects_in_each_population",
        "universal_selector_chooses_qualifying_model_in_every_fold",
        "universal_model_beats_constant_observational_score_in_each_population",
        "universal_beta_r2_positive_in_each_population",
        "universal_model_beats_support_proxy_beta_mse_in_each_population",
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population",
        "whole_object_outer_predictions_complete",
    )
    development_pass = all(gate_checks[key] is True for key in required)
    selected_by_fold = {int(row["fold"]): str(row["selected_model_id"]) for row in nested_ledger}
    per_object = [
        v2_experiment._public_object(
            row,
            labels[str(row["key"])],
            predictions[str(row["key"])],
            selected_by_fold[assignments[str(row["key"])]],
        )
        for row in objects
    ]
    source_bindings = {key: _binding(root, path) for key, path in _source_paths(config).items()}
    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "GRAVITY_ROADMAP_ITEM_02_SHAPE_ANISOTROPY_THIRD_ATTEMPT",
        "decision": (
            "PASS_ITEM2_STELLAR_MULTIPOLE_DEVELOPMENT_GATE"
            if development_pass
            else "INCONCLUSIVE_ITEM2_STELLAR_MULTIPOLES"
        ),
        "claims": {
            "alternative_to_gr_established": False,
            "direct_lensing_test_completed": False,
            "historical_novelty_established": False,
            "intrinsic_shape_cause_established": False,
            "roadmap_item_2_complete": development_pass,
            "sequential_G6_G7_G8_advanced": False,
            "sparc_confirmation_opened": False,
            "stellar_multipole_shape_predicts_cross_scale_response": development_pass,
            "stellar_tracer_is_complete_baryonic_mass_map": False,
        },
        "config": {
            "content_sha256": canonical_sha256(_sealable(config)),
            "file_sha256": _file_sha256(root / CONFIG_PATH),
            "path": CONFIG_PATH,
        },
        "counts": {
            "objects": len(objects),
            "galaxies": sum(row["domain"] == "galaxy" for row in objects),
            "clusters": sum(row["domain"] == "cluster" for row in objects),
            "galaxy_points": sum(
                row["point_count"] for row in objects if row["domain"] == "galaxy"
            ),
            "cluster_points": sum(
                row["point_count"] for row in objects if row["domain"] == "cluster"
            ),
            "models": len(models),
            "formula_classes": len(models) + 1,
            "outer_folds": int(cv["outer_folds"]),
            "paid_model_calls": 0,
            "sparc_confirmation_evaluator_accesses": 0,
            "direct_lensing_likelihood_evaluations": 0,
        },
        "data_lineage": {
            "galaxy_shape": "target_blind_unWISE_NEO11_W1_old_stellar_light",
            "cluster_shape": "target_blind_CLASH_photoz_member_raw_stellar_mass_at_150_kpc",
            "cluster_representation_validation": "independent_Chandra_Xray_morphology_at_500_kpc",
            "gravity_response": "sealed_Item1_development_labels_joined_only_after_representation_pass",
            "lensing_corrected_stellar_mass_used": False,
            "direct_lensing_likelihood_used": False,
        },
        "representation_validation": {
            "decision": representation["decision"],
            "checks": representation["checks"],
            "external_xray_validation": representation["external_xray_validation"],
            "counts": representation["counts"],
        },
        "feature_ranges": v2_experiment._feature_ranges(objects),
        "gate_checks": gate_checks,
        "model_results": model_results,
        "multipole_energy_overlap_diagnostic": overlap,
        "nested_universal_stellar_multipoles": {
            "clip_count": nested_clipped,
            "coefficient_prediction": beta_metrics,
            "fold_ledger": nested_ledger,
            "score": score,
            "selected_model_counts": v2_experiment._selection_counts(nested_ledger),
        },
        "per_object_diagnostics": per_object,
        "specific_hypothesis_results": {
            "comparable_stellar_tracer_representation": "PASSED_TARGET_BLIND_EXTERNAL_MORPHOLOGY_GATE",
            "common_stellar_multipole_grammar": (
                "SURVIVES_DEVELOPMENT_GATE"
                if development_pass
                else "NOT_SHOWN_TO_GENERATE_THE_CROSS_SCALE_RESPONSE"
            ),
            "population_proxy": "DIAGNOSTIC_CONTROL_ONLY_NOT_A_FIRST_PRINCIPLES_CAUSE",
        },
        "limitations": [
            "The cluster HST footprint restricts the common member aperture to 150 kpc while the external X-ray morphology is measured at 500 kpc.",
            "Galaxy W1 intensity and cluster catalog stellar mass are closer tracers than W1 and X-ray emissivity, but they are not identical measurement operators.",
            "Photometric-redshift members can contain interlopers, and the stellar map omits hot gas and intracluster light.",
            "The gravity-response labels remain model-dependent development diagnostics reconstructed from prior galaxy and CLASH analyses.",
            "No direct lensing likelihood, independent confirmation set, or historical novelty adjudication was used.",
        ],
        "next_action": (
            "Retain this attempt's excluded equation families. If the development gates fail, remain on Item 2 and add a genuinely intermediate or filamentary population rather than tuning these labels; if all pass, freeze an independent confirmation before any causal claim."
        ),
        "source_bindings": source_bindings,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityItem2StellarMultipoleExperimentError("receipt schema changed")
    _verify_envelope(receipt, label="stellar multipole receipt")
    config = load_config(root)
    expected_config = {
        "content_sha256": canonical_sha256(_sealable(config)),
        "file_sha256": _file_sha256(root / CONFIG_PATH),
        "path": CONFIG_PATH,
    }
    if receipt.get("config") != expected_config:
        raise GravityItem2StellarMultipoleExperimentError("config binding changed")
    expected_sources = {key: _binding(root, path) for key, path in _source_paths(config).items()}
    if receipt.get("source_bindings") != expected_sources:
        raise GravityItem2StellarMultipoleExperimentError("source binding changed")
    claims = receipt.get("claims", {})
    forbidden = (
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
        "stellar_tracer_is_complete_baryonic_mass_map",
    )
    if any(claims.get(key) is not False for key in forbidden):
        raise GravityItem2StellarMultipoleExperimentError("receipt overstates a claim")
    counts = receipt.get("counts", {})
    if (
        counts.get("objects") != 88
        or counts.get("galaxies") != 68
        or counts.get("clusters") != 20
        or counts.get("paid_model_calls") != 0
        or counts.get("sparc_confirmation_evaluator_accesses") != 0
        or counts.get("direct_lensing_likelihood_evaluations") != 0
    ):
        raise GravityItem2StellarMultipoleExperimentError("access or population counts changed")
    for model_id in ("linear_support_dimension_proxy", "support_plus_all_multipoles"):
        if (
            receipt.get("model_results", {})
            .get(model_id, {})
            .get("qualifying_universal_stellar_multipole_model")
            is not False
        ):
            raise GravityItem2StellarMultipoleExperimentError("proxy entered admission")
    gates = receipt.get("gate_checks", {})
    required = (
        "target_blind_representation_validation_passed",
        "population_proxy_models_excluded_from_universal_admission",
        "multipole_energy_overlap_contains_minimum_objects_in_each_population",
        "universal_selector_chooses_qualifying_model_in_every_fold",
        "universal_model_beats_constant_observational_score_in_each_population",
        "universal_beta_r2_positive_in_each_population",
        "universal_model_beats_support_proxy_beta_mse_in_each_population",
        "universal_model_beats_support_proxy_in_energy_overlap_for_each_population",
        "whole_object_outer_predictions_complete",
    )
    passed = all(gates.get(key) is True for key in required)
    expected_decision = (
        "PASS_ITEM2_STELLAR_MULTIPOLE_DEVELOPMENT_GATE"
        if passed
        else "INCONCLUSIVE_ITEM2_STELLAR_MULTIPOLES"
    )
    if receipt.get("decision") != expected_decision:
        raise GravityItem2StellarMultipoleExperimentError("decision disagrees with gates")
    if (
        claims.get("roadmap_item_2_complete") is not passed
        or claims.get("stellar_multipole_shape_predicts_cross_scale_response") is not passed
    ):
        raise GravityItem2StellarMultipoleExperimentError("claim disagrees with gates")
    if gates.get("confirmation_or_direct_lensing_completed") is not False:
        raise GravityItem2StellarMultipoleExperimentError("confirmation was promoted")
    if len(receipt.get("per_object_diagnostics", ())) != 88:
        raise GravityItem2StellarMultipoleExperimentError("diagnostics changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityItem2StellarMultipoleExperimentError(
                f"refusing to overwrite immutable stellar multipole receipt: {path}"
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
