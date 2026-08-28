"""Test roadmap Item 2: projected baryonic shape and anisotropy."""

from __future__ import annotations

import argparse
import csv
import io
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item1_effective_dimension as item1
from .gravity_g1_pilot import _binding, _file_sha256, _load_json, _metric
from .gravity_g4_photometric_law_construction import prepare_photometric_packets
from .sigma_core import canonical_json_bytes, canonical_sha256

SCHEMA = "invariant-gravity-roadmap-item2-shape-anisotropy-receipt-1.0"
CONFIG_SCHEMA = "invariant-gravity-roadmap-item2-shape-anisotropy-config-1.0"
CONFIG_PATH = "configs/gravity_item2_shape_anisotropy.json"
SOURCE_PATH = "src/sigma_theory_compiler/gravity_item2_shape_anisotropy.py"
TEST_PATH = "tests/test_gravity_item2_shape_anisotropy.py"
OUTPUT_PATH = "runs/gravity/roadmap/item-02-shape-anisotropy-v1.json"

DONAHUE_TO_TARGET = {
    "A209": "A209",
    "A383": "A383",
    "0329-02": "MACS0329",
    "0416-24": "MACS0416",
    "0429-02": "MACS0429",
    "0647+70": "MACS0647",
    "0717+37": "MACS0717",
    "0744+39": "MACS0744",
    "A611": "A611",
    "1115+01": "MACS1115",
    "1149+22": "MACS1149",
    "1206-08": "MACS1206",
    "1347-1145": "RXJ1347",
    "1532+30": "RXJ1532",
    "1720+35": "MACS1720",
    "A2261": "A2261",
    "1931-26": "MACS1931",
    "2129+0005": "RXJ2129",
    "MS2137": "MS2137",
    "2248-44": "RXJ2248",
}

VIZIER_ID_TO_TARGET = {
    "A209": "A209",
    "A383": "A383",
    "M0329": "MACS0329",
    "M0416": "MACS0416",
    "M0429": "MACS0429",
    "M0647": "MACS0647",
    "M0717": "MACS0717",
    "M0744": "MACS0744",
    "A611": "A611",
    "M1115": "MACS1115",
    "M1149": "MACS1149",
    "M1206": "MACS1206",
    "RXJ1347": "RXJ1347",
    "M1532": "RXJ1532",
    "M1720": "MACS1720",
    "A2261": "A2261",
    "M1931": "MACS1931",
    "RXJ2129": "RXJ2129",
    "MS2137": "MS2137",
    "RXJ2248": "RXJ2248",
}

DOMAINS = ("galaxy", "cluster")


class GravityItem2ShapeAnisotropyError(ValueError):
    """The Item 2 contract, shape data, or receipt are inconsistent."""


def _verify_envelope(receipt: Mapping[str, Any]) -> None:
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2ShapeAnisotropyError("Item 2 predecessor seal changed")


def load_config(root: Path) -> Mapping[str, Any]:
    """Load the Item 2 contract and verify every frozen input."""

    root = root.resolve()
    config = _load_json(root / CONFIG_PATH)
    if config.get("schema_version") != CONFIG_SCHEMA:
        raise GravityItem2ShapeAnisotropyError("Item 2 config schema changed")
    if config.get("status") != "exploratory_real_data_model_development":
        raise GravityItem2ShapeAnisotropyError("Item 2 status changed")
    roadmap = config.get("roadmap_binding", {})
    if (
        int(roadmap.get("item_number", 0)) != 2
        or roadmap.get("item_title") != "Shape and anisotropy"
        or _file_sha256(root / str(roadmap.get("path"))) != roadmap.get("file_sha256")
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 roadmap binding changed")
    predecessor_binding = config.get("predecessor_binding", {})
    predecessor_path = root / str(predecessor_binding.get("path"))
    if _file_sha256(predecessor_path) != predecessor_binding.get("file_sha256"):
        raise GravityItem2ShapeAnisotropyError("Item 2 predecessor file changed")
    predecessor = _load_json(predecessor_path)
    _verify_envelope(predecessor)
    item1.validate_receipt(predecessor, root=root)
    if predecessor.get("content_sha256") != predecessor_binding.get(
        "content_sha256"
    ) or predecessor.get("decision") != predecessor_binding.get("required_decision"):
        raise GravityItem2ShapeAnisotropyError("Item 2 predecessor content changed")

    source_files: list[Mapping[str, Any]] = []
    sources = config.get("sources", {})
    source_files.extend(sources.get("sparc_global_properties", {}).get("files", ()))
    source_files.append(sources.get("clash_xray_morphology", {}).get("file", {}))
    source_files.extend(sources.get("clash_morphology_crosscheck", {}).get("files", ()))
    if len(source_files) != 5:
        raise GravityItem2ShapeAnisotropyError("Item 2 source binding count changed")
    for binding in source_files:
        if _file_sha256(root / str(binding.get("path"))) != binding.get("file_sha256"):
            raise GravityItem2ShapeAnisotropyError("Item 2 source file changed")

    authorization = config.get("authorization", {})
    if (
        authorization.get("paid_model_calls_allowed") is not False
        or int(authorization.get("sparc_confirmation_evaluator_accesses_allowed", -1)) != 0
        or int(authorization.get("direct_lensing_likelihood_evaluations_allowed", -1)) != 0
        or authorization.get("sequential_G6_G7_G8_advanced") is not False
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 authorization changed")
    model_ids = [str(row.get("id")) for row in config.get("models", ())]
    if model_ids != [
        "constant",
        "linear_projected_axis_ratio",
        "quadratic_projected_axis_ratio",
        "linear_projected_concentration",
        "linear_axis_ratio_plus_concentration",
        "axis_concentration_interaction",
        "linear_support_dimension_proxy",
        "support_plus_shared_shape",
        "domain_specific_shape_bank",
    ]:
        raise GravityItem2ShapeAnisotropyError("Item 2 model grammar changed")
    if int(config.get("cross_validation", {}).get("outer_folds", 0)) != 5:
        raise GravityItem2ShapeAnisotropyError("Item 2 fold count changed")
    if (
        int(config.get("population", {}).get("intermediate_or_filamentary_geometry_systems", -1))
        != 0
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 geometry-population accounting changed")
    admission = config.get("admission", {})
    if (
        admission.get("intrinsic_shape_or_anisotropy_must_be_measured_in_both_populations")
        is not True
        or admission.get("intermediate_or_filamentary_geometry_must_be_included") is not True
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 full-shape admission changed")
    boundaries = config.get("claim_boundaries", {})
    if boundaries.get("alternative_to_gr_established") is not False:
        raise GravityItem2ShapeAnisotropyError("Item 2 config overstates its claim")
    return config


def _vizier_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    try:
        header_index = next(index for index, line in enumerate(lines) if line.startswith("recno\t"))
    except StopIteration as error:
        raise GravityItem2ShapeAnisotropyError("VizieR TSV header is missing") from error
    reader = csv.DictReader(io.StringIO("\n".join(lines[header_index:])), delimiter="\t")
    rows = []
    for raw in reader:
        record = str(raw.get("recno", "")).strip()
        if record.isdigit():
            rows.append({str(key): str(value).strip() for key, value in raw.items()})
    if len({row["recno"] for row in rows}) != len(rows):
        raise GravityItem2ShapeAnisotropyError("duplicate VizieR record")
    return rows


def parse_sparc_properties(path: Path) -> dict[str, dict[str, Any]]:
    """Parse only target-blind SPARC global morphology fields."""

    parsed: dict[str, dict[str, Any]] = {}
    for raw in _vizier_rows(path):
        try:
            row = {
                "effective_radius_kpc": float(raw["Reff"]),
                "effective_surface_brightness": float(raw["SBeff"]),
                "hubble_type": int(raw["Type"]),
                "inclination_deg": float(raw["i"]),
                "inclination_uncertainty_deg": float(raw["e_i"]),
                "name": raw["Name"],
                "scale_length_kpc": float(raw["Rdisk"]),
                "central_disk_surface_brightness": float(raw["SBdisk"]),
            }
        except (KeyError, TypeError, ValueError) as error:
            raise GravityItem2ShapeAnisotropyError("invalid SPARC morphology row") from error
        if (
            not row["name"]
            or not 0.0 < row["inclination_deg"] <= 90.0
            or row["inclination_uncertainty_deg"] <= 0.0
            or row["effective_radius_kpc"] <= 0.0
            or row["scale_length_kpc"] <= 0.0
            or row["effective_surface_brightness"] <= 0.0
            or row["central_disk_surface_brightness"] <= 0.0
        ):
            raise GravityItem2ShapeAnisotropyError("nonphysical SPARC morphology row")
        parsed[str(row["name"])] = row
    if len(parsed) != 175:
        raise GravityItem2ShapeAnisotropyError("SPARC morphology row count changed")
    return parsed


def parse_donahue_morphology(path: Path) -> dict[str, dict[str, Any]]:
    """Parse the hash-bound factual extraction of Donahue et al. Table 3."""

    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    rows: dict[str, dict[str, Any]] = {}
    numeric = (
        "concentration",
        "concentration_unc",
        "centroid_shift",
        "centroid_shift_unc",
        "p30",
        "p30_unc",
        "p40",
        "p40_unc",
        "axis_ratio",
        "axis_ratio_unc",
        "position_angle_deg",
        "position_angle_unc_deg",
    )
    for raw in reader:
        name = str(raw.get("source_name", "")).strip()
        try:
            row = {key: float(str(raw[key]).strip()) for key in numeric}
        except (KeyError, TypeError, ValueError) as error:
            raise GravityItem2ShapeAnisotropyError("invalid Donahue morphology row") from error
        if (
            not name
            or not 0.0 < row["axis_ratio"] <= 1.0
            or not 0.0 < row["concentration"] < 1.0
            or row["centroid_shift"] <= 0.0
            or row["p30"] <= 0.0
            or row["p40"] <= 0.0
            or any(not np.isfinite(value) for value in row.values())
        ):
            raise GravityItem2ShapeAnisotropyError("nonphysical Donahue morphology row")
        rows[name] = {"source_name": name, **row}
    if len(rows) != 25:
        raise GravityItem2ShapeAnisotropyError("Donahue morphology row count changed")
    return rows


def parse_zitrin_crosscheck(path: Path) -> dict[str, dict[str, float]]:
    """Retain only the independent X-ray morphology columns from the CLASH catalog."""

    rows: dict[str, dict[str, float]] = {}
    for raw in _vizier_rows(path):
        target = VIZIER_ID_TO_TARGET.get(raw.get("ID", ""))
        if target is None or not raw.get("ell") or not raw.get("shift"):
            continue
        try:
            ellipticity = float(raw["ell"])
            shift = float(raw["shift"])
        except ValueError as error:
            raise GravityItem2ShapeAnisotropyError("invalid CLASH crosscheck row") from error
        if not 0.0 <= ellipticity < 1.0 or shift <= 0.0:
            raise GravityItem2ShapeAnisotropyError("nonphysical CLASH crosscheck row")
        rows[target] = {
            "axis_ratio_from_ellipticity": 1.0 - ellipticity,
            "centroid_shift": shift,
        }
    return rows


def projected_light_shape(
    radius: np.ndarray, disk: np.ndarray, bulge: np.ndarray
) -> dict[str, float]:
    """Compute aperture-matched C20 and bulge fraction from baryonic light only."""

    radius = np.asarray(radius, dtype=np.float64)
    disk = np.asarray(disk, dtype=np.float64)
    bulge = np.asarray(bulge, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.shape != disk.shape
        or disk.shape != bulge.shape
        or len(radius) < 3
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(disk))
        or np.any(~np.isfinite(bulge))
        or np.any(radius <= 0.0)
        or np.any(np.diff(radius) <= 0.0)
        or np.any(disk < 0.0)
        or np.any(bulge < 0.0)
    ):
        raise GravityItem2ShapeAnisotropyError("invalid projected light profile")
    edges = np.empty(len(radius) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radius[:-1] + radius[1:])
    edges[-1] = radius[-1]
    if np.any(np.diff(edges) <= 0.0):
        raise GravityItem2ShapeAnisotropyError("invalid projected light annuli")
    annular_area = edges[1:] ** 2 - edges[:-1] ** 2
    total_surface = disk + bulge
    total_light = float(total_surface @ annular_area)
    if total_light <= 0.0:
        raise GravityItem2ShapeAnisotropyError("zero projected light")
    cutoff = 0.2 * radius[-1]
    inner_area = np.maximum(
        np.minimum(edges[1:], cutoff) ** 2 - np.minimum(edges[:-1], cutoff) ** 2,
        0.0,
    )
    concentration = float(total_surface @ inner_area) / total_light
    bulge_fraction = float(bulge @ annular_area) / total_light
    if not 0.0 < concentration < 1.0 or not 0.0 <= bulge_fraction <= 1.0:
        raise GravityItem2ShapeAnisotropyError("invalid projected light summary")
    return {
        "bulge_light_fraction": bulge_fraction,
        "concentration_c20": concentration,
    }


def _item1_labels(root: Path, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    receipt = _load_json(root / str(config["predecessor_binding"]["path"]))
    labels = {
        f"{row['domain']}:{row['name']}": {
            "at_grid_boundary": bool(row["oracle_beta_at_grid_boundary"]),
            "beta": float(row["oracle_beta_target_derived"]),
        }
        for row in receipt["per_object_diagnostics"]
    }
    if len(labels) != 159:
        raise GravityItem2ShapeAnisotropyError("Item 1 label count changed")
    return labels


def prepare_shape_objects(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Join measured shape fields to Item 1 objects without exposing targets to features."""

    root = root.resolve()
    base_objects = item1.prepare_objects(root, item1.load_config(root))
    source = config["sources"]
    sparc_path = root / str(source["sparc_global_properties"]["files"][0]["path"])
    donahue_path = root / str(source["clash_xray_morphology"]["file"]["path"])
    crosscheck_path = root / str(source["clash_morphology_crosscheck"]["files"][0]["path"])
    sparc = parse_sparc_properties(sparc_path)
    donahue_source = parse_donahue_morphology(donahue_path)
    donahue = {
        target: donahue_source[source_name] for source_name, target in DONAHUE_TO_TARGET.items()
    }
    if len(donahue) != 20:
        raise GravityItem2ShapeAnisotropyError("CLASH target morphology match changed")
    crosscheck = parse_zitrin_crosscheck(crosscheck_path)
    packets = {packet["galaxy"].name: packet for packet in prepare_photometric_packets(root)}
    labels = _item1_labels(root, config)
    objects: list[dict[str, Any]] = []
    for base in base_objects:
        domain = str(base["domain"])
        name = str(base["name"])
        if domain == "galaxy":
            packet = packets[name]
            properties = sparc[name]
            radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
            disk = np.expm1(np.asarray(packet["features"]["log1p_sb_disk"]))
            bulge = np.expm1(np.asarray(packet["features"]["log1p_sb_bulge"]))
            light = projected_light_shape(radius, disk, bulge)
            axis_ratio = float(np.cos(np.deg2rad(float(properties["inclination_deg"]))))
            concentration = float(light["concentration_c20"])
            support_dimension = 2.0
            galaxy_bulge = float(light["bulge_light_fraction"])
            galaxy_radius_ratio = float(
                properties["effective_radius_kpc"] / properties["scale_length_kpc"]
            )
            cluster_log_shift = 0.0
            cluster_log_p30 = 0.0
            provenance = {
                "hubble_type": int(properties["hubble_type"]),
                "inclination_deg": _metric(properties["inclination_deg"]),
                "inclination_uncertainty_deg": _metric(properties["inclination_uncertainty_deg"]),
                "shape_tracer": "SPARC_inclination_and_3p6_micron_light",
            }
        else:
            morphology = donahue[name]
            axis_ratio = float(morphology["axis_ratio"])
            concentration = float(morphology["concentration"])
            support_dimension = 3.0
            galaxy_bulge = 0.0
            galaxy_radius_ratio = 0.0
            cluster_log_shift = float(np.log10(morphology["centroid_shift"]))
            cluster_log_p30 = float(np.log10(morphology["p30"]))
            provenance = {
                "axis_ratio_uncertainty": _metric(morphology["axis_ratio_unc"]),
                "concentration_uncertainty": _metric(morphology["concentration_unc"]),
                "shape_tracer": "Chandra_Xray_surface_brightness_500_kpc",
                "source_name": morphology["source_name"],
            }
        features = {
            "axis_ratio_times_concentration": axis_ratio * concentration,
            "cluster_log_centroid_shift": cluster_log_shift,
            "cluster_log_p30": cluster_log_p30,
            "galaxy_bulge_light_fraction": galaxy_bulge,
            "galaxy_effective_to_disk_radius": galaxy_radius_ratio,
            "projected_axis_ratio": axis_ratio,
            "projected_axis_ratio_squared": axis_ratio**2,
            "projected_concentration_c20": concentration,
            "support_dimension": support_dimension,
        }
        if any(not np.isfinite(value) for value in features.values()):
            raise GravityItem2ShapeAnisotropyError("non-finite Item 2 shape feature")
        objects.append({**base, "features": features, "shape_provenance": provenance})
    if set(labels) != {str(row["key"]) for row in objects}:
        raise GravityItem2ShapeAnisotropyError("Item 2 object and label sets differ")

    common_crosscheck = sorted(set(crosscheck) & set(donahue))
    if len(common_crosscheck) < 10:
        raise GravityItem2ShapeAnisotropyError("too few CLASH morphology crosschecks")
    current_q = np.asarray([donahue[name]["axis_ratio"] for name in common_crosscheck])
    prior_q = np.asarray(
        [crosscheck[name]["axis_ratio_from_ellipticity"] for name in common_crosscheck]
    )
    correlation = float(np.corrcoef(current_q, prior_q)[0, 1])
    crosscheck_receipt = {
        "axis_ratio_pearson_correlation": _metric(correlation),
        "matched_clusters": len(common_crosscheck),
        "names": common_crosscheck,
        "role": "independent_Xray_morphology_consistency_only",
    }
    return objects, labels, crosscheck_receipt


def _selection_counts(ledger: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    selected = [str(row["selected_model_id"]) for row in ledger]
    return {model: selected.count(model) for model in sorted(set(selected))}


def _feature_ranges(objects: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
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
            for feature in ("projected_axis_ratio", "projected_concentration_c20")
        }
        for domain in DOMAINS
    }


def _overlap_diagnostic(
    objects: Sequence[Mapping[str, Any]],
    labels: Mapping[str, Mapping[str, Any]],
    shape_predictions: Mapping[str, float],
    proxy_predictions: Mapping[str, float],
) -> dict[str, Any]:
    domain_ranges = {
        domain: (
            min(
                float(row["features"]["projected_axis_ratio"])
                for row in objects
                if row["domain"] == domain
            ),
            max(
                float(row["features"]["projected_axis_ratio"])
                for row in objects
                if row["domain"] == domain
            ),
        )
        for domain in DOMAINS
    }
    lower = max(domain_ranges[domain][0] for domain in DOMAINS)
    upper = min(domain_ranges[domain][1] for domain in DOMAINS)
    members = [
        row for row in objects if lower <= float(row["features"]["projected_axis_ratio"]) <= upper
    ]
    member_counts = {domain: sum(row["domain"] == domain for row in members) for domain in DOMAINS}
    if lower > upper or any(member_counts[domain] == 0 for domain in DOMAINS):
        return {
            "axis_ratio_interval": [_metric(lower), _metric(upper)],
            "by_population": {},
            "observed_overlap": False,
        }
    shape_metrics = item1._beta_metrics(members, labels, shape_predictions)
    proxy_metrics = item1._beta_metrics(members, labels, proxy_predictions)
    return {
        "axis_ratio_interval": [_metric(lower), _metric(upper)],
        "by_population": {
            domain: {
                "objects": member_counts[domain],
                "shape_beta_mse": shape_metrics["by_population"][domain]["mean_squared_error"],
                "support_proxy_beta_mse": proxy_metrics["by_population"][domain][
                    "mean_squared_error"
                ],
            }
            for domain in DOMAINS
        },
        "observed_overlap": lower <= upper,
    }


def _public_object(
    row: Mapping[str, Any],
    label: Mapping[str, Any],
    shape_beta: float,
    all_beta: float,
    shape_model: str,
    all_model: str,
) -> dict[str, Any]:
    return {
        "all_model_oof_beta": _metric(all_beta),
        "all_model_selected": all_model,
        "domain": row["domain"],
        "features": {key: _metric(value) for key, value in sorted(row["features"].items())},
        "name": row["name"],
        "oracle_beta_at_grid_boundary": label["at_grid_boundary"],
        "oracle_beta_target_derived": _metric(label["beta"]),
        "point_count": row["point_count"],
        "shape_provenance": row["shape_provenance"],
        "universal_shape_oof_beta": _metric(shape_beta),
        "universal_shape_selected": shape_model,
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Run nested whole-object shape tests and seal their limitations."""

    root = root.resolve()
    config = load_config(root)
    objects, labels, crosscheck = prepare_shape_objects(root, config)
    cv = config["cross_validation"]
    assignments = item1._fold_assignments(
        objects, salt=str(cv["fold_salt"]), folds=int(cv["outer_folds"])
    )
    models = [dict(row) for row in config["models"]]
    universal_models = [
        row
        for row in models
        if row["id"] == "constant" or row["qualifying_universal_shape_model"] is True
    ]

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
            "qualifying_universal_shape_model": model["qualifying_universal_shape_model"],
            "score": item1._observational_score(objects, predictions),
        }

    shape_predictions, shape_ledger, shape_clipped = item1._nested_model_oof(
        objects, labels, assignments, universal_models, config
    )
    all_predictions, all_ledger, all_clipped = item1._nested_model_oof(
        objects, labels, assignments, models, config
    )
    shape_beta_metrics = item1._beta_metrics(objects, labels, shape_predictions)
    shape_score = item1._observational_score(objects, shape_predictions)
    all_beta_metrics = item1._beta_metrics(objects, labels, all_predictions)
    all_score = item1._observational_score(objects, all_predictions)
    constant_score = model_results["constant"]["score"]
    support_result = model_results["linear_support_dimension_proxy"]
    support_predictions = model_predictions["linear_support_dimension_proxy"]
    qualifying_ids = {
        str(row["id"]) for row in models if row["qualifying_universal_shape_model"] is True
    }
    shape_selected = [str(row["selected_model_id"]) for row in shape_ledger]
    overlap = _overlap_diagnostic(objects, labels, shape_predictions, support_predictions)
    minimum_overlap = int(cv["axis_ratio_overlap_minimum_objects_per_population"])
    enough_overlap = overlap["observed_overlap"] and all(
        int(overlap["by_population"][domain]["objects"]) >= minimum_overlap for domain in DOMAINS
    )
    overlap_beats_proxy = enough_overlap and all(
        float(overlap["by_population"][domain]["shape_beta_mse"])
        < float(overlap["by_population"][domain]["support_proxy_beta_mse"])
        for domain in DOMAINS
    )
    gate_checks = {
        "axis_ratio_overlap_contains_minimum_objects_in_each_population": enough_overlap,
        "confirmation_or_direct_lensing_completed": False,
        "intermediate_or_filamentary_geometry_included": (
            int(config["population"]["intermediate_or_filamentary_geometry_systems"]) > 0
        ),
        "intrinsic_shape_or_anisotropy_measured_in_both_populations": False,
        "population_proxy_models_excluded_from_universal_admission": all(
            row["qualifying_universal_shape_model"] is False
            for row in models
            if row["id"]
            in {
                "linear_support_dimension_proxy",
                "support_plus_shared_shape",
                "domain_specific_shape_bank",
            }
        ),
        "shape_sources_match_every_exploration_object": len(objects) == 159,
        "target_fields_absent_from_shape_feature_builder": True,
        "universal_selector_chooses_qualifying_shape_in_every_fold": all(
            model_id in qualifying_ids for model_id in shape_selected
        ),
        "universal_shape_beats_constant_observationally_in_both_populations": all(
            float(shape_score["by_population"][domain]["chi_square"])
            < float(constant_score["by_population"][domain]["chi_square"])
            for domain in DOMAINS
        ),
        "universal_shape_beats_support_proxy_beta_mse_in_both_populations": all(
            float(shape_beta_metrics["by_population"][domain]["mean_squared_error"])
            < float(
                support_result["coefficient_prediction"]["by_population"][domain][
                    "mean_squared_error"
                ]
            )
            for domain in DOMAINS
        ),
        "universal_shape_beats_support_proxy_in_axis_ratio_overlap": overlap_beats_proxy,
        "universal_shape_beta_r2_positive_in_both_populations": all(
            float(shape_beta_metrics["by_population"][domain]["r2"]) > 0.0 for domain in DOMAINS
        ),
        "whole_object_target_blind_outer_predictions_complete": (
            len(shape_predictions) == len(objects) == len(all_predictions)
        ),
    }
    development_pass = all(
        gate_checks[key]
        for key in (
            "axis_ratio_overlap_contains_minimum_objects_in_each_population",
            "intermediate_or_filamentary_geometry_included",
            "intrinsic_shape_or_anisotropy_measured_in_both_populations",
            "population_proxy_models_excluded_from_universal_admission",
            "shape_sources_match_every_exploration_object",
            "target_fields_absent_from_shape_feature_builder",
            "universal_selector_chooses_qualifying_shape_in_every_fold",
            "universal_shape_beats_constant_observationally_in_both_populations",
            "universal_shape_beats_support_proxy_beta_mse_in_both_populations",
            "universal_shape_beats_support_proxy_in_axis_ratio_overlap",
            "universal_shape_beta_r2_positive_in_both_populations",
            "whole_object_target_blind_outer_predictions_complete",
        )
    )
    shape_by_fold = {int(row["fold"]): row["selected_model_id"] for row in shape_ledger}
    all_by_fold = {int(row["fold"]): row["selected_model_id"] for row in all_ledger}
    per_object = [
        _public_object(
            row,
            labels[str(row["key"])],
            shape_predictions[str(row["key"])],
            all_predictions[str(row["key"])],
            str(shape_by_fold[assignments[str(row["key"])]]),
            str(all_by_fold[assignments[str(row["key"])]]),
        )
        for row in objects
    ]
    source_bindings = {
        "config": _binding(root, CONFIG_PATH),
        "predecessor": _binding(root, str(config["predecessor_binding"]["path"])),
        "roadmap": _binding(root, str(config["roadmap_binding"]["path"])),
        "source": _binding(root, SOURCE_PATH),
        "test": _binding(root, TEST_PATH),
    }
    for source_id, binding in (
        ("sparc_table1", config["sources"]["sparc_global_properties"]["files"][0]),
        ("sparc_readme", config["sources"]["sparc_global_properties"]["files"][1]),
        ("donahue_table3", config["sources"]["clash_xray_morphology"]["file"]),
        ("clash_crosscheck", config["sources"]["clash_morphology_crosscheck"]["files"][0]),
        ("clash_readme", config["sources"]["clash_morphology_crosscheck"]["files"][1]),
    ):
        source_bindings[source_id] = _binding(root, str(binding["path"]))

    body: dict[str, Any] = {
        "schema_version": SCHEMA,
        "goal": "GRAVITY_ROADMAP_ITEM_02_SHAPE_ANISOTROPY",
        "decision": (
            "PASS_ITEM2_SHAPE_ANISOTROPY_DEVELOPMENT_GATE"
            if development_pass
            else "INCONCLUSIVE_ITEM2_SHAPE_ANISOTROPY"
        ),
        "claims": {
            "alternative_to_gr_established": False,
            "direct_lensing_test_completed": False,
            "galaxy_intrinsic_axis_ratio_measured": False,
            "historical_novelty_established": False,
            "intrinsic_shape_cause_established": False,
            "roadmap_item_2_complete": development_pass,
            "sequential_G6_G7_G8_advanced": False,
            "sparc_confirmation_opened": False,
            "universal_projected_shape_predicts_cross_scale_response": development_pass,
        },
        "config": {"content_sha256": canonical_sha256(config), "path": CONFIG_PATH},
        "counts": {
            "clash_clusters": sum(row["domain"] == "cluster" for row in objects),
            "clash_points": sum(
                row["point_count"] for row in objects if row["domain"] == "cluster"
            ),
            "direct_lensing_likelihood_evaluations": 0,
            "formula_classes": len(models) + 2,
            "models": len(models),
            "outer_folds": int(cv["outer_folds"]),
            "paid_model_calls": 0,
            "sparc_confirmation_evaluator_accesses": 0,
            "sparc_exploration_galaxies": sum(row["domain"] == "galaxy" for row in objects),
            "sparc_exploration_points": sum(
                row["point_count"] for row in objects if row["domain"] == "galaxy"
            ),
            "unique_formula_prediction_classes": len(
                {result["score"]["prediction_manifest_sha256"] for result in model_results.values()}
                | {
                    shape_score["prediction_manifest_sha256"],
                    all_score["prediction_manifest_sha256"],
                }
            ),
        },
        "data_lineage": {
            "clash_dynamics_target": config["population"]["clash_target_status"],
            "cluster_shape": "Chandra_Xray_surface_brightness_morphology_at_500_kpc",
            "galaxy_shape": "SPARC_inclination_and_3p6_micron_surface_photometry",
            "shape_feature_inputs": "baryonic_imaging_geometry_only",
            "target_derived_oracle_role": (
                "sealed_Item1_training_label_and_descriptive_ceiling_never_formula_input"
            ),
        },
        "feature_ranges": _feature_ranges(objects),
        "gate_checks": gate_checks,
        "model_results": model_results,
        "morphology_crosscheck": crosscheck,
        "overlap_diagnostic": overlap,
        "nested_all_models": {
            "clip_count": all_clipped,
            "coefficient_prediction": all_beta_metrics,
            "fold_ledger": all_ledger,
            "score": all_score,
            "selected_model_counts": _selection_counts(all_ledger),
        },
        "nested_universal_shape": {
            "clip_count": shape_clipped,
            "coefficient_prediction": shape_beta_metrics,
            "fold_ledger": shape_ledger,
            "score": shape_score,
            "selected_model_counts": _selection_counts(shape_ledger),
        },
        "per_object_diagnostics": per_object,
        "specific_hypothesis_results": {
            "projected_axis_ratio_and_concentration": (
                "SURVIVES_DEVELOPMENT_GATE"
                if development_pass
                else "NOT_SHOWN_TO_GENERATE_THE_CROSS_SCALE_RESPONSE"
            ),
            "population_proxy_plus_shape": (
                "DIAGNOSTIC_ONLY_NOT_A_UNIVERSAL_FIRST_PRINCIPLES_RULE"
            ),
            "xray_shape_measurement": "COMPLETE_FOR_ALL_20_CLASH_TARGET_CLUSTERS",
        },
        "limitations": [
            "SPARC inclination is used in rotation-curve deprojection, so a q-dependent residual can reflect inclination systematics rather than gravity.",
            "cos(inclination) is a thin-disk projected-axis proxy, not an intrinsic disk-thickness measurement.",
            "Galaxy concentration uses 3.6-micron stellar light while cluster concentration uses X-ray emissivity proportional approximately to gas density squared.",
            "The CLASH dynamics target is reconstructed through spherical NFW posteriors and is not a direct lensing likelihood.",
            "The current galaxy assets do not provide two-dimensional lopsidedness, bar strength, intrinsic thickness, or position-angle quadrupoles for all 139 objects.",
            "No formula selected in this development run is independently confirmed or historically novel.",
        ],
        "next_action": (
            "If the shared projected-shape gate fails, retain its excluded family and add independent two-dimensional galaxy morphology or intrinsic-thickness data before treating triaxiality or quadrupoles as tested; if it survives, freeze it before any confirmation sample."
        ),
        "source_bindings": source_bindings,
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    """Reject drift, leakage, proxy admission, or inflated claims."""

    root = root.resolve()
    if receipt.get("schema_version") != SCHEMA:
        raise GravityItem2ShapeAnisotropyError("Item 2 receipt schema changed")
    body = dict(receipt)
    supplied = body.pop("content_sha256", None)
    if supplied != canonical_sha256(body):
        raise GravityItem2ShapeAnisotropyError("Item 2 receipt seal changed")
    counts = receipt.get("counts", {})
    if (
        counts.get("sparc_confirmation_evaluator_accesses") != 0
        or counts.get("direct_lensing_likelihood_evaluations") != 0
        or counts.get("paid_model_calls") != 0
        or counts.get("sparc_exploration_galaxies") != 139
        or counts.get("clash_clusters") != 20
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 accounting changed")
    claims = receipt.get("claims", {})
    for claim in (
        "alternative_to_gr_established",
        "direct_lensing_test_completed",
        "galaxy_intrinsic_axis_ratio_measured",
        "historical_novelty_established",
        "intrinsic_shape_cause_established",
        "sequential_G6_G7_G8_advanced",
        "sparc_confirmation_opened",
    ):
        if claims.get(claim) is not False:
            raise GravityItem2ShapeAnisotropyError("Item 2 overstates its claim")
    for model_id in (
        "linear_support_dimension_proxy",
        "support_plus_shared_shape",
        "domain_specific_shape_bank",
    ):
        if (
            receipt.get("model_results", {})
            .get(model_id, {})
            .get("qualifying_universal_shape_model")
            is not False
        ):
            raise GravityItem2ShapeAnisotropyError("population proxy entered admission")
    config = load_config(root)
    if receipt.get("config", {}).get("content_sha256") != canonical_sha256(config):
        raise GravityItem2ShapeAnisotropyError("Item 2 config binding changed")
    expected = {
        "config": CONFIG_PATH,
        "predecessor": str(config["predecessor_binding"]["path"]),
        "roadmap": str(config["roadmap_binding"]["path"]),
        "source": SOURCE_PATH,
        "test": TEST_PATH,
        "sparc_table1": str(config["sources"]["sparc_global_properties"]["files"][0]["path"]),
        "sparc_readme": str(config["sources"]["sparc_global_properties"]["files"][1]["path"]),
        "donahue_table3": str(config["sources"]["clash_xray_morphology"]["file"]["path"]),
        "clash_crosscheck": str(
            config["sources"]["clash_morphology_crosscheck"]["files"][0]["path"]
        ),
        "clash_readme": str(config["sources"]["clash_morphology_crosscheck"]["files"][1]["path"]),
    }
    bindings = receipt.get("source_bindings", {})
    if set(bindings) != set(expected):
        raise GravityItem2ShapeAnisotropyError("Item 2 source binding set changed")
    for key, path in expected.items():
        if bindings.get(key) != _binding(root, path):
            raise GravityItem2ShapeAnisotropyError(f"Item 2 {key} binding changed")
    passed = receipt.get("decision") == "PASS_ITEM2_SHAPE_ANISOTROPY_DEVELOPMENT_GATE"
    gate_checks = receipt.get("gate_checks", {})
    if (
        gate_checks.get("confirmation_or_direct_lensing_completed") is not False
        or gate_checks.get("intermediate_or_filamentary_geometry_included") is not False
        or gate_checks.get("intrinsic_shape_or_anisotropy_measured_in_both_populations")
        is not False
    ):
        raise GravityItem2ShapeAnisotropyError("Item 2 unavailable evidence was promoted")
    required_gate_keys = (
        "axis_ratio_overlap_contains_minimum_objects_in_each_population",
        "intermediate_or_filamentary_geometry_included",
        "intrinsic_shape_or_anisotropy_measured_in_both_populations",
        "population_proxy_models_excluded_from_universal_admission",
        "shape_sources_match_every_exploration_object",
        "target_fields_absent_from_shape_feature_builder",
        "universal_selector_chooses_qualifying_shape_in_every_fold",
        "universal_shape_beats_constant_observationally_in_both_populations",
        "universal_shape_beats_support_proxy_beta_mse_in_both_populations",
        "universal_shape_beats_support_proxy_in_axis_ratio_overlap",
        "universal_shape_beta_r2_positive_in_both_populations",
        "whole_object_target_blind_outer_predictions_complete",
    )
    gates_passed = all(gate_checks.get(key) is True for key in required_gate_keys)
    if passed is not gates_passed:
        raise GravityItem2ShapeAnisotropyError("Item 2 decision and measured gates disagree")
    if claims.get("roadmap_item_2_complete") is not passed:
        raise GravityItem2ShapeAnisotropyError("Item 2 decision and completion disagree")
    if claims.get("universal_projected_shape_predicts_cross_scale_response") is not passed:
        raise GravityItem2ShapeAnisotropyError("Item 2 shape claim changed")


def _write_immutable(path: Path, value: Mapping[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityItem2ShapeAnisotropyError(
                f"refusing to overwrite immutable Item 2 receipt: {path}"
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
    "GravityItem2ShapeAnisotropyError",
    "build_receipt",
    "load_config",
    "main",
    "parse_donahue_morphology",
    "parse_sparc_properties",
    "parse_zitrin_crosscheck",
    "prepare_shape_objects",
    "projected_light_shape",
    "validate_receipt",
]
