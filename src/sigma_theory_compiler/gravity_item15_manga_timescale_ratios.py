"""Frozen fresh-identity timescale-ratio search for gravity roadmap Item 15."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .gravity_item11_neargalcat_external_field import (
    _content_hashed,
    _metric,
    _minimum_separation_arcsec,
    _ridge_fit,
    _ridge_predict,
    _sha256_file,
    _source_rows,
    _validate_content_hash,
    canonical_json_bytes,
)
from .gravity_item14_gz3d_resonance_coherence import (
    GravityItem14CoherenceError,
    _maps_payload,
    derive_radial_response,
)

CONFIG_PATH = Path("configs/gravity_item15_manga_timescale_ratios_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "bfccc9118a637dc5438f60a7f935044170c8a396"
SAMPLE_FREEZE_COMMIT = "73b9dc23f6951c8e0e490b9a4a55394c0b7509fb"


class GravityItem15TimescaleError(RuntimeError):
    """Raised when an Item 15 scientific or response boundary drifts."""


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def _serialize(row: Mapping[str, Any]) -> dict[str, Any]:
    output = {}
    for key, value in row.items():
        if isinstance(value, (np.bool_, bool)):
            output[key] = bool(value)
        elif isinstance(value, (np.integer, int)):
            output[key] = int(value)
        elif isinstance(value, (np.floating, float)):
            output[key] = _metric(float(value))
        else:
            output[key] = value
    return output


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem15TimescaleError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem15TimescaleError("Item 14 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 14 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem15TimescaleError("Item 14 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem15TimescaleError("Item 14 synthesis decision changed")
    predictor_binding = config["sources"]["predictors"]
    predictor_path = root / predictor_binding["path"]
    if _sha256_file(predictor_path) != predictor_binding["file_sha256"]:
        raise GravityItem15TimescaleError("Item 15 predictor source changed")
    predictors = json.loads(predictor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predictors, "Item 15 predictor source")
    if predictors.get("content_sha256") != predictor_binding["content_sha256"]:
        raise GravityItem15TimescaleError("Item 15 predictor content binding changed")
    if len(predictors.get("records", [])) != int(predictor_binding["records"]):
        raise GravityItem15TimescaleError("Item 15 predictor row count changed")
    for section in ("identity_exclusions", "coordinate_exclusions"):
        for entry in config["independence"][section]:
            if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
                raise GravityItem15TimescaleError(
                    f"predecessor exclusion source changed: {entry['path']}"
                )
            if "content_sha256" in entry:
                value = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
                _validate_content_hash(value, "Item 15 identity exclusion")
                if value.get("content_sha256") != entry["content_sha256"]:
                    raise GravityItem15TimescaleError("identity exclusion content changed")
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem15TimescaleError("Item 15 config contains an overclaim")
    authorization = config["authorization"]
    required_false = (
        "paid_model_calls_allowed",
        "response_query_allowed_before_sample_freeze",
        "confirmation_response_query_allowed",
        "post_response_candidate_generation_allowed",
        "kinematic_response_as_timescale_predictor_allowed",
        "object_identity_as_numeric_feature_allowed",
    )
    if any(bool(authorization[key]) for key in required_false):
        raise GravityItem15TimescaleError("Item 15 authorization boundary changed")
    if not bool(authorization["exploration_response_query_allowed_after_sample_freeze"]):
        raise GravityItem15TimescaleError("Item 15 exploration authorization changed")
    if not bool(config["sources"]["response"]["confirmation_query_forbidden"]):
        raise GravityItem15TimescaleError("Item 15 confirmation source boundary changed")
    for key in (
        "fresh_maps_payload_downloads",
        "fresh_maps_header_schema_reads",
        "fresh_maps_pixel_values_read",
        "fresh_resolved_kinematic_response_objects_read",
        "paid_model_calls",
    ):
        if int(config["sources"]["prefreeze_access"][key]) != 0:
            raise GravityItem15TimescaleError("Item 15 prefreeze response boundary changed")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem15TimescaleError("Item 15 candidate count changed")
    if int(config["candidate_generator"]["post_response_cells"]) != 0:
        raise GravityItem15TimescaleError("Item 15 post-response candidate boundary changed")
    sample = config["sample"]
    if int(sample["maximum_total_objects"]) != 320:
        raise GravityItem15TimescaleError("Item 15 sample size changed")
    if int(sample["exploration_objects"]) + int(sample["confirmation_objects"]) != int(
        sample["maximum_total_objects"]
    ):
        raise GravityItem15TimescaleError("Item 15 exploration/confirmation split changed")
    if 4 * int(sample["objects_per_cell"]) != int(sample["maximum_total_objects"]):
        raise GravityItem15TimescaleError("Item 15 sample cell balance changed")
    if 4 * int(sample["exploration_per_cell"]) != int(sample["exploration_objects"]):
        raise GravityItem15TimescaleError("Item 15 exploration cell balance changed")
    audit = sample["prefreeze_predictor_audit"]
    if int(audit["eligible"]) != sum(int(value) for value in audit["cell_counts"].values()):
        raise GravityItem15TimescaleError("Item 15 predictor audit cell accounting changed")
    if int(audit["eligible"]) + sum(
        int(value) for value in audit["failure_counts"].values()
    ) != int(predictor_binding["records"]):
        raise GravityItem15TimescaleError("Item 15 predictor audit total changed")
    if int(audit["response_values_read"]) != 0:
        raise GravityItem15TimescaleError("Item 15 predictor audit opened response")
    if min(int(value) for value in audit["cell_counts"].values()) < int(
        sample["objects_per_cell"]
    ):
        raise GravityItem15TimescaleError("Item 15 audited cells cannot fill the sample")
    constants = config["physical_constants"]
    if not math.isclose(
        float(constants["omega_matter"]) + float(constants["omega_lambda"]),
        1.0,
        abs_tol=1e-12,
    ):
        raise GravityItem15TimescaleError("Item 15 cosmology is not frozen flat")
    if bool(config["timescale_features"]["direct_hot_gas_cooling_time_available"]):
        raise GravityItem15TimescaleError("Item 15 fabricated a direct cooling time")
    if not bool(config["timescale_features"]["direct_hot_gas_cooling_followup_required"]):
        raise GravityItem15TimescaleError("Item 15 cooling follow-up boundary changed")
    quality = config["quality"]
    inner = [float(value) for value in quality["inner_annulus_re"]]
    outer = [float(value) for value in quality["outer_annulus_re"]]
    quantiles = [float(value) for value in quality["velocity_span_quantiles"]]
    if not (
        len(inner) == len(outer) == len(quantiles) == 2
        and 0 <= inner[0] < inner[1] == outer[0] < outer[1]
        and 0 <= quantiles[0] < quantiles[1] <= 1
    ):
        raise GravityItem15TimescaleError("Item 15 annular response definition changed")
    return config


def _require_scientific_freeze() -> None:
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem15TimescaleError("Item 15 scientific freeze is not bound")


def _excluded_identities(root: Path, config: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    plateifus: set[str] = set()
    mangaids: set[str] = set()
    for entry in config["independence"]["identity_exclusions"]:
        source = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        for row in source[entry["objects_key"]]:
            plateifus.add(str(row[entry["plateifu_key"]]).upper())
            mangaids.add(str(row[entry["mangaid_key"]]).upper())
    return plateifus, mangaids


def _coordinates(root: Path, config: Mapping[str, Any]) -> np.ndarray:
    rows = []
    for entry in config["independence"]["coordinate_exclusions"]:
        for row in _source_rows(root, entry):
            try:
                ra = float(row[entry["ra_key"]])
                dec = float(row[entry["dec_key"]])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                rows.append((ra, dec))
    if not rows:
        raise GravityItem15TimescaleError("empty Item 15 predecessor coordinate registry")
    return np.asarray(rows, dtype=np.float64)


def cosmic_age_gyr(redshift: float, config: Mapping[str, Any]) -> float:
    if not math.isfinite(redshift) or redshift < 0:
        raise GravityItem15TimescaleError("invalid Item 15 redshift")
    constants = config["physical_constants"]
    omega_m = float(constants["omega_matter"])
    omega_l = float(constants["omega_lambda"])
    hubble_time = float(constants["hubble_time_gyr"])
    age = 2.0 * hubble_time / (3.0 * math.sqrt(omega_l))
    age *= math.asinh(math.sqrt(omega_l / omega_m) / (1.0 + redshift) ** 1.5)
    if not math.isfinite(age) or age <= 0:
        raise GravityItem15TimescaleError("invalid Item 15 cosmic age")
    return age


def derive_timescale_features(
    predictor: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, float]:
    try:
        log_mass = float(predictor["log_stellar_mass"])
        log_radius = float(predictor["log_half_light_radius"])
        log_specific_sfr = float(predictor["log_specific_sfr"])
        redshift = float(predictor["redshift"])
    except (KeyError, TypeError, ValueError) as exc:
        raise GravityItem15TimescaleError("invalid Item 15 timescale predictor") from exc
    if any(not math.isfinite(value) for value in (log_mass, log_radius, log_specific_sfr)):
        raise GravityItem15TimescaleError("non-finite Item 15 timescale predictor")
    constants = config["physical_constants"]
    mass = 10.0**log_mass
    radius = 10.0**log_radius
    gravitational_constant = float(constants["gravitational_constant_kpc3_msun_gyr2"])
    dynamical = math.sqrt(radius**3 / (gravitational_constant * mass))
    crossing = dynamical
    orbital = 2.0 * math.pi * dynamical
    free_fall = math.pi / math.sqrt(8.0) * dynamical
    mass_doubling = 10.0 ** (-log_specific_sfr - 9.0)
    cosmic = cosmic_age_gyr(redshift, config)
    stellar_count = mass / float(constants["mean_stellar_mass_msun_for_relaxation"])
    relaxation = 0.1 * stellar_count / math.log(stellar_count) * dynamical
    values = {
        "dynamical_time_gyr": dynamical,
        "crossing_time_gyr": crossing,
        "orbital_time_gyr": orbital,
        "free_fall_time_gyr": free_fall,
        "mass_doubling_time_gyr": mass_doubling,
        "cosmic_age_gyr": cosmic,
        "relaxation_time_gyr": relaxation,
        "log_dynamical_time_gyr": math.log10(dynamical),
        "log_mass_doubling_time_gyr": math.log10(mass_doubling),
        "log_cosmic_age_gyr": math.log10(cosmic),
        "log_relaxation_time_gyr": math.log10(relaxation),
        "log_cosmic_to_dynamical": math.log10(cosmic / dynamical),
        "log_growth_to_dynamical": math.log10(mass_doubling / dynamical),
        "log_growth_to_cosmic": math.log10(mass_doubling / cosmic),
        "log_relaxation_to_cosmic": math.log10(relaxation / cosmic),
        "log_relaxation_to_growth": math.log10(relaxation / mass_doubling),
    }
    normalization = config["evaluation"]["fixed_timescale_normalization"]
    normalized_clocks = []
    for key in (
        "log_dynamical_time_gyr",
        "log_mass_doubling_time_gyr",
        "log_cosmic_age_gyr",
        "log_relaxation_time_gyr",
    ):
        center, scale = (float(value) for value in normalization[key])
        normalized_clocks.append((values[key] - center) / scale)
    clock_array = np.asarray(normalized_clocks, dtype=np.float64)
    values["clock_hierarchy_span"] = float(np.max(clock_array) - np.min(clock_array))
    shifted = clock_array - np.max(clock_array)
    weights = np.exp(shifted)
    weights /= np.sum(weights)
    values["clock_entropy"] = float(-np.sum(weights * np.log(weights)) / math.log(len(weights)))
    if any(not math.isfinite(float(value)) or float(value) <= 0 for key, value in values.items() if not key.startswith("log_") and key not in {"clock_hierarchy_span", "clock_entropy"}):
        raise GravityItem15TimescaleError("invalid Item 15 physical timescale")
    if any(not math.isfinite(float(value)) for value in values.values()):
        raise GravityItem15TimescaleError("non-finite Item 15 timescale feature")
    return values


def _eligible_predictors(
    root: Path, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    source = json.loads((root / config["sources"]["predictors"]["path"]).read_text("utf-8"))
    excluded_plates, excluded_mangaids = _excluded_identities(root, config)
    predecessor_coordinates = _coordinates(root, config)
    separation_limit = float(config["independence"]["coordinate_exclusion_arcseconds"])
    sample = config["sample"]
    failures = Counter()
    records_by_mangaid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw in source["records"]:
        plateifu = str(raw["plateifu"]).upper()
        mangaid = str(raw["mangaid"]).upper()
        if plateifu in excluded_plates or mangaid in excluded_mangaids:
            failures["identity"] += 1
            continue
        if _minimum_separation_arcsec(
            float(raw["ra"]), float(raw["dec"]), predecessor_coordinates
        ) <= separation_limit:
            failures["coordinate"] += 1
            continue
        axis_ratio = float(raw["axis_ratio"])
        if not (
            float(sample["minimum_axis_ratio"])
            <= axis_ratio
            <= float(sample["maximum_axis_ratio"])
        ):
            failures["axis_ratio"] += 1
            continue
        log_specific_sfr = float(raw["log_specific_sfr"])
        if not (
            float(sample["minimum_specific_sfr_log10_per_year"])
            <= log_specific_sfr
            <= float(sample["maximum_specific_sfr_log10_per_year"])
        ):
            failures["specific_sfr_bound"] += 1
            continue
        if int(float(raw["drp3qual"])) != 0 or int(float(raw["dapqual"])) != 0:
            failures["catalog_quality"] += 1
            continue
        features = derive_timescale_features(raw, config)
        growth_state = (
            "slow_growth"
            if log_specific_sfr <= float(sample["specific_sfr_threshold_log10_per_year"])
            else "fast_growth"
        )
        mass_state = (
            "higher_mass"
            if float(raw["log_stellar_mass"])
            > float(sample["stellar_mass_threshold_log10"])
            else "lower_mass"
        )
        records_by_mangaid[mangaid].append(
            {
                **raw,
                **features,
                "plateifu": plateifu,
                "mangaid": mangaid,
                "growth_state": growth_state,
                "stellar_mass_state": mass_state,
                "sample_cell": f"{growth_state}|{mass_state}",
            }
        )
    records = []
    for mangaid, rows in records_by_mangaid.items():
        rows.sort(
            key=lambda row: _split_hash(
                f"{mangaid}|{row['plateifu']}", str(sample["split_salt"])
            )
        )
        records.append(rows[0])
        failures["duplicate_predictor_identity"] += len(rows) - 1
    return records, dict(sorted((key, value) for key, value in failures.items() if value))


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    scale_min, scale_max = (float(value) for value in generator["scale_log_uniform"])
    power_min, power_max = (float(value) for value in generator["power_log_uniform"])
    return {
        "family": random.integers(0, len(generator["families"]), count, dtype=np.int8),
        "threshold": random.uniform(*generator["threshold_uniform"], count),
        "scale": np.exp(random.uniform(math.log(scale_min), math.log(scale_max), count)),
        "power": np.exp(random.uniform(math.log(power_min), math.log(power_max), count)),
        "phase": random.uniform(*generator["phase_uniform"], count),
        "modulation": random.integers(0, len(generator["modulations"]), count, dtype=np.int8),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("family", "threshold", "scale", "power", "phase", "modulation"):
        array = np.ascontiguousarray(arrays[key])
        digest.update(key.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def write_prepared_sources(root: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    _require_scientific_freeze()
    config = load_config(root)
    eligible, exclusion_counts = _eligible_predictors(root, config)
    audit = config["sample"]["prefreeze_predictor_audit"]
    cell_counts_eligible = Counter(str(row["sample_cell"]) for row in eligible)
    if len(eligible) != int(audit["eligible"]) or dict(sorted(cell_counts_eligible.items())) != {
        key: int(value) for key, value in sorted(audit["cell_counts"].items())
    }:
        raise GravityItem15TimescaleError("Item 15 response-free predictor audit drifted")
    observed_failures = {key: value for key, value in exclusion_counts.items() if value}
    if observed_failures != {
        key: int(value) for key, value in audit["failure_counts"].items()
    }:
        raise GravityItem15TimescaleError("Item 15 response-free exclusion audit drifted")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in eligible:
        grouped[str(row["sample_cell"])].append(row)
    expected_cells = {
        f"{growth}|{mass}"
        for growth in ("fast_growth", "slow_growth")
        for mass in ("lower_mass", "higher_mass")
    }
    if set(grouped) != expected_cells:
        raise GravityItem15TimescaleError("Item 15 sample cells unavailable")
    sample_config = config["sample"]
    selected_features = []
    sample_objects = []
    per_cell = int(sample_config["objects_per_cell"])
    exploration_per_cell = int(sample_config["exploration_per_cell"])
    split_salt = str(sample_config["split_salt"])
    fold_salt = str(config["evaluation"]["fold_salt"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    for cell in sorted(expected_cells):
        candidates = sorted(
            grouped[cell],
            key=lambda row: _split_hash(
                f"select|{row['mangaid']}|{row['plateifu']}", split_salt
            ),
        )[:per_cell]
        if len(candidates) != per_cell:
            raise GravityItem15TimescaleError("Item 15 sample cell shortfall")
        role_order = sorted(
            candidates,
            key=lambda row: _split_hash(
                f"role|{row['mangaid']}|{row['plateifu']}", split_salt
            ),
        )
        exploration_ids = {
            str(row["plateifu"]) for row in role_order[:exploration_per_cell]
        }
        fold_order = sorted(
            role_order[:exploration_per_cell],
            key=lambda row: _split_hash(
                f"fold|{row['mangaid']}|{row['plateifu']}", fold_salt
            ),
        )
        fold_by_plate = {
            str(row["plateifu"]): ordinal % outer_folds
            for ordinal, row in enumerate(fold_order)
        }
        for row in candidates:
            plateifu = str(row["plateifu"])
            role = "exploration" if plateifu in exploration_ids else "reserved_confirmation"
            outer_fold = fold_by_plate.get(plateifu)
            selected_features.append(_serialize(row))
            sample_objects.append(
                {
                    "plateifu": plateifu,
                    "mangaid": str(row["mangaid"]),
                    "ra": _metric(float(row["ra"])),
                    "dec": _metric(float(row["dec"])),
                    "sample_cell": cell,
                    "growth_state": row["growth_state"],
                    "stellar_mass_state": row["stellar_mass_state"],
                    "role": role,
                    "outer_fold": outer_fold,
                    "selection_digest": _split_hash(
                        f"selected|{row['mangaid']}|{plateifu}", split_salt
                    ),
                }
            )
    role_counts = Counter(row["role"] for row in sample_objects)
    cell_counts = Counter(row["sample_cell"] for row in sample_objects)
    fold_counts = Counter(
        row["outer_fold"] for row in sample_objects if row["role"] == "exploration"
    )
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-timescale-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "objects": sorted(sample_objects, key=lambda row: (row["role"], row["plateifu"])),
            "eligible_cell_counts": dict(sorted(cell_counts_eligible.items())),
            "selected_cell_counts": dict(sorted(cell_counts.items())),
            "fold_counts_exploration": {
                str(key): value for key, value in sorted(fold_counts.items())
            },
            "exclusion_counts": exclusion_counts,
            "counts": {
                "eligible": len(eligible),
                "selected": len(sample_objects),
                "exploration": role_counts["exploration"],
                "reserved_confirmation": role_counts["reserved_confirmation"],
                "response_rows_read": 0,
                "predecessor_selected": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    predictor_source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-timescale-predictors-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "records": sorted(selected_features, key=lambda row: str(row["plateifu"])),
            "counts": {
                "records": len(selected_features),
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {
                "direct_hot_gas_cooling_time_tested": False,
                "causal_timescale_mechanism_established": False,
            },
        }
    )
    arrays = generate_candidates(config)
    families = config["candidate_generator"]["families"]
    family_counts = Counter(families[int(value)]["id"] for value in arrays["family"])
    origin_counts = Counter(
        families[int(value)]["origin_status"] for value in arrays["family"]
    )
    qualifying_counts = Counter(
        "qualifying" if bool(families[int(value)]["qualifying"]) else "control"
        for value in arrays["family"]
    )
    candidates = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-timescale-candidates-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "algorithm": config["candidate_generator"]["algorithm"],
            "seed": config["candidate_generator"]["seed"],
            "candidate_digest_sha256": _candidate_digest(arrays),
            "family_counts": dict(sorted(family_counts.items())),
            "origin_status_counts": dict(sorted(origin_counts.items())),
            "qualifying_counts": dict(sorted(qualifying_counts.items())),
            "equivalence_boundaries": config["candidate_generator"][
                "equivalence_boundaries"
            ],
            "counts": {
                "candidate_cells": len(arrays["family"]),
                "response_rows_read": 0,
                "post_response_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )
    paths = tuple(root / config["outputs"][key] for key in (
        "sample_manifest",
        "predictor_source",
        "candidate_manifest",
    ))
    for path, artifact in zip(paths, (sample, predictor_source, candidates), strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(artifact) + b"\n")
    return paths


def validate_prepared_sources(
    sample: Mapping[str, Any],
    predictors: Mapping[str, Any],
    candidates: Mapping[str, Any],
    root: Path,
) -> None:
    config = load_config(root)
    for value, label in (
        (sample, "Item 15 sample manifest"),
        (predictors, "Item 15 predictor source"),
        (candidates, "Item 15 candidate manifest"),
    ):
        _validate_content_hash(value, label)
        if value["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
            raise GravityItem15TimescaleError(f"{label} scientific binding changed")
    objects = sample["objects"]
    if len(objects) != int(config["sample"]["maximum_total_objects"]):
        raise GravityItem15TimescaleError("Item 15 selected sample count changed")
    ids = [(str(row["plateifu"]), str(row["mangaid"])) for row in objects]
    if len(ids) != len(set(ids)):
        raise GravityItem15TimescaleError("Item 15 selected identity duplicated")
    roles = Counter(str(row["role"]) for row in objects)
    if roles != {
        "exploration": int(config["sample"]["exploration_objects"]),
        "reserved_confirmation": int(config["sample"]["confirmation_objects"]),
    }:
        raise GravityItem15TimescaleError("Item 15 role counts changed")
    if set(sample["selected_cell_counts"].values()) != {
        int(config["sample"]["objects_per_cell"])
    } or len(sample["selected_cell_counts"]) != 4:
        raise GravityItem15TimescaleError("Item 15 cell balance changed")
    if set(sample["fold_counts_exploration"].values()) != {
        int(config["sample"]["exploration_objects"])
        // int(config["evaluation"]["outer_folds"])
    }:
        raise GravityItem15TimescaleError("Item 15 fold balance changed")
    feature_ids = {str(row["plateifu"]) for row in predictors["records"]}
    if feature_ids != {plate for plate, _ in ids} or len(predictors["records"]) != len(ids):
        raise GravityItem15TimescaleError("Item 15 predictor identity set changed")
    if predictors["counts"]["response_rows_read"] != 0:
        raise GravityItem15TimescaleError("response entered Item 15 prepared predictors")
    arrays = generate_candidates(config)
    if candidates["candidate_digest_sha256"] != _candidate_digest(arrays):
        raise GravityItem15TimescaleError("Item 15 candidate digest changed")
    if candidates["counts"]["candidate_cells"] != int(
        config["candidate_generator"]["candidate_cells"]
    ):
        raise GravityItem15TimescaleError("Item 15 candidate manifest count changed")
    if candidates["counts"]["post_response_cells"] != 0:
        raise GravityItem15TimescaleError("post-response candidate entered Item 15")
    if any(bool(value) for value in sample["claims"].values()) or any(
        bool(value) for value in predictors["claims"].values()
    ) or any(bool(value) for value in candidates["claims"].values()):
        raise GravityItem15TimescaleError("Item 15 prepared source contains an overclaim")


def _load_prepared(root: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_config(root)
    values = tuple(
        json.loads((root / config["outputs"][key]).read_text(encoding="utf-8"))
        for key in ("sample_manifest", "predictor_source", "candidate_manifest")
    )
    validate_prepared_sources(*values, root)
    return values


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem15TimescaleError("Item 15 sample freeze is not bound")
    config = load_config(root)
    sample, _, _ = _load_prepared(root)
    exploration = sorted(
        (row for row in sample["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["plateifu"]),
    )
    confirmation = {
        str(row["plateifu"])
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    records = []
    failures = []
    files = []
    for sample_row in exploration:
        plateifu = str(sample_row["plateifu"])
        if plateifu in confirmation:
            raise GravityItem15TimescaleError("Item 15 confirmation entered MAPS acquisition")
        payload, filename, url = _maps_payload(root, sample_row, config)
        file_record = {
            "plateifu": plateifu,
            "file_name": filename,
            "url": url,
            "file_bytes": len(payload),
            "file_sha256": hashlib.sha256(payload).hexdigest(),
        }
        try:
            response = derive_radial_response(payload, sample_row, config)
        except GravityItem14CoherenceError as exc:
            failures.append({**file_record, "reason": str(exc)})
            continue
        records.append({**response, **file_record, "fits_checksum_verified": True})
        files.append({**file_record, "fits_checksum_verified": True})
    observed = {str(row["plateifu"]) for row in [*records, *failures]}
    if confirmation & observed:
        raise GravityItem15TimescaleError("Item 15 confirmation MAPS response entered source")
    source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-manga-maps-response-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "data_model": config["sources"]["response"]["data_model"],
            "files": files,
            "records": sorted(records, key=lambda row: str(row["plateifu"])),
            "failures": sorted(failures, key=lambda row: str(row["plateifu"])),
            "counts": {
                "exploration_response_objects_attempted": len(exploration),
                "exploration_response_objects_parsed": len(records),
                "exploration_response_failures": len(failures),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    path = root / config["outputs"]["response_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(source) + b"\n")
    return path


def validate_response_source(source: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(source, "Item 15 response source")
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 response scientific binding changed")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 response sample binding changed")
    for key in ("confirmation_response_rows", "post_response_formula_cells", "paid_model_calls"):
        if int(source["counts"][key]) != 0:
            raise GravityItem15TimescaleError(f"Item 15 forbidden response count changed: {key}")
    sample, _, _ = _load_prepared(root)
    exploration = {
        str(row["plateifu"])
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    confirmation = {
        str(row["plateifu"])
        for row in sample["objects"]
        if row["role"] == "reserved_confirmation"
    }
    record_ids = [str(row["plateifu"]) for row in source["records"]]
    failure_ids = [str(row["plateifu"]) for row in source["failures"]]
    observed = record_ids + failure_ids
    if len(observed) != len(set(observed)) or set(observed) != exploration:
        raise GravityItem15TimescaleError("Item 15 MAPS response identity set changed")
    if confirmation & set(observed):
        raise GravityItem15TimescaleError("Item 15 confirmation MAPS response opened")
    if int(source["counts"]["exploration_response_objects_attempted"]) != len(exploration):
        raise GravityItem15TimescaleError("Item 15 MAPS attempt count changed")
    if int(source["counts"]["exploration_response_objects_parsed"]) != len(record_ids):
        raise GravityItem15TimescaleError("Item 15 MAPS parsed count changed")
    if int(source["counts"]["exploration_response_failures"]) != len(failure_ids):
        raise GravityItem15TimescaleError("Item 15 MAPS failure count changed")
    file_ids = [str(row["plateifu"]) for row in source["files"]]
    if set(file_ids) != set(record_ids) or len(file_ids) != len(record_ids):
        raise GravityItem15TimescaleError("Item 15 MAPS file receipt set changed")
    if any(not bool(row["fits_checksum_verified"]) for row in source["files"]):
        raise GravityItem15TimescaleError("Item 15 MAPS checksum receipt changed")
    if any(bool(value) for value in source["claims"].values()):
        raise GravityItem15TimescaleError("Item 15 response contains an overclaim")


def extract_rows(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    sample, predictors, _ = _load_prepared(root)
    response = json.loads(
        (root / config["outputs"]["response_source"]).read_text(encoding="utf-8")
    )
    validate_response_source(response, root)
    predictor_by_plate = {str(row["plateifu"]): row for row in predictors["records"]}
    sample_by_plate = {
        str(row["plateifu"]): row
        for row in sample["objects"]
        if row["role"] == "exploration"
    }
    records = []
    failures = [
        {"plateifu": str(row["plateifu"]), "reasons": [str(row["reason"])]}
        for row in response["failures"]
    ]
    quality = config["quality"]
    for raw in response["records"]:
        plateifu = str(raw["plateifu"])
        predictor = predictor_by_plate[plateifu]
        sample_row = sample_by_plate[plateifu]
        reasons = []
        if int(raw["drp3qual"]) != int(quality["required_drp3qual"]):
            reasons.append("drp3qual")
        if int(raw["dapqual"]) != int(quality["required_dapqual"]):
            reasons.append("dapqual")
        minimum_span = float(quality["minimum_annular_velocity_span_km_s"])
        for tracer, maximum in (
            ("stellar", float(quality["maximum_stellar_annular_velocity_span_km_s"])),
            ("halpha", float(quality["maximum_halpha_annular_velocity_span_km_s"])),
        ):
            for annulus in ("inner", "outer"):
                span = float(raw[f"{tracer}_{annulus}_velocity_span_km_s"])
                if not (minimum_span <= span <= maximum):
                    reasons.append(f"{tracer}_{annulus}_span")
            ratio = float(raw[f"{tracer}_outer_to_inner_span_ratio"])
            if not (
                math.isfinite(ratio)
                and float(quality["minimum_outer_to_inner_span_ratio"])
                <= ratio
                <= float(quality["maximum_outer_to_inner_span_ratio"])
            ):
                reasons.append(f"{tracer}_outer_inner_ratio")
        output = {
            **predictor,
            **raw,
            "outer_fold": sample_row["outer_fold"],
            "quality_pass": not reasons,
            "quality_failure_reasons": reasons,
        }
        records.append(_serialize(output))
        if reasons:
            failures.append({"plateifu": plateifu, "reasons": reasons})
    expected = int(config["sample"]["exploration_objects"])
    passing_rows = [row for row in records if row["quality_pass"]]
    passing = len(passing_rows)
    retention = passing / expected
    fold_counts = Counter(int(row["outer_fold"]) for row in passing_rows)
    dynamical_split = float(config["sample"]["dynamical_time_threshold_log10_gyr"])
    cosmic_split = float(config["sample"]["cosmic_age_threshold_log10_gyr"])
    stratum_counts = {
        "fast_growth": sum(row["growth_state"] == "fast_growth" for row in passing_rows),
        "slow_growth": sum(row["growth_state"] == "slow_growth" for row in passing_rows),
        "lower_mass": sum(
            row["stellar_mass_state"] == "lower_mass" for row in passing_rows
        ),
        "higher_mass": sum(
            row["stellar_mass_state"] == "higher_mass" for row in passing_rows
        ),
        "dynamical_low": sum(
            float(row["log_dynamical_time_gyr"]) <= dynamical_split for row in passing_rows
        ),
        "dynamical_high": sum(
            float(row["log_dynamical_time_gyr"]) > dynamical_split for row in passing_rows
        ),
        "cosmic_low": sum(
            float(row["log_cosmic_age_gyr"]) <= cosmic_split for row in passing_rows
        ),
        "cosmic_high": sum(
            float(row["log_cosmic_age_gyr"]) > cosmic_split for row in passing_rows
        ),
    }
    fold_quality = set(fold_counts) == set(range(int(config["evaluation"]["outer_folds"])))
    fold_quality = fold_quality and min(fold_counts.values(), default=0) >= int(
        quality["minimum_quality_passing_per_outer_fold"]
    )
    stratum_quality = min(stratum_counts.values(), default=0) >= int(
        quality["minimum_quality_passing_per_gate_stratum"]
    )
    quality_pass = passing >= int(
        quality["minimum_quality_passing_exploration_galaxies"]
    ) and retention >= float(quality["minimum_quality_retention_fraction"])
    quality_pass = quality_pass and fold_quality and stratum_quality
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-timescale-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": (
                "PASS_ITEM15_TIMESCALE_QUALITY"
                if quality_pass
                else "INCONCLUSIVE_ITEM15_TIMESCALE_QUALITY"
            ),
            "records": sorted(records, key=lambda row: str(row["plateifu"])),
            "failures": failures,
            "quality_fold_counts": {
                str(key): value for key, value in sorted(fold_counts.items())
            },
            "quality_gate_stratum_counts": dict(sorted(stratum_counts.items())),
            "quality_gate_splits": {
                "log_dynamical_time_gyr_median": _metric(dynamical_split),
                "log_cosmic_age_gyr_median": _metric(cosmic_split),
            },
            "counts": {
                "exploration_response_objects_attempted": response["counts"][
                    "exploration_response_objects_attempted"
                ],
                "exploration_response_objects_parsed": len(response["records"]),
                "response_parse_failures": len(response["failures"]),
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": expected - passing,
                "quality_retention_fraction": _metric(retention),
                "predecessor_selected": 0,
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": config["claim_boundaries"],
        }
    )
    path = root / config["outputs"]["extraction_summary"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return path


def _fixed_array(
    rows: Sequence[Mapping[str, Any]], config: Mapping[str, Any], field: str
) -> np.ndarray:
    center, scale = (
        float(value) for value in config["evaluation"]["fixed_timescale_normalization"][field]
    )
    if scale <= 0:
        raise GravityItem15TimescaleError("fixed Item 15 timescale normalization changed")
    return (np.asarray([float(row[field]) for row in rows]) - center) / scale


def _load_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary = json.loads(
        (root / config["outputs"]["extraction_summary"]).read_text(encoding="utf-8")
    )
    _validate_content_hash(summary, "Item 15 extraction summary")
    if summary["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 extraction scientific binding changed")
    if summary["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 extraction sample binding changed")
    for key in ("confirmation_response_rows", "post_response_formula_cells", "paid_model_calls"):
        if int(summary["counts"][key]) != 0:
            raise GravityItem15TimescaleError(f"Item 15 forbidden extraction count: {key}")
    if any(bool(value) for value in summary["claims"].values()):
        raise GravityItem15TimescaleError("Item 15 extraction contains an overclaim")
    _, _, candidate_manifest = _load_prepared(root)
    rows = [row for row in summary["records"] if row["quality_pass"]]
    if not rows:
        raise GravityItem15TimescaleError("no Item 15 quality rows")
    structural = np.column_stack(
        [
            np.asarray([float(row[field]) for row in rows])
            for field in config["evaluation"]["structural_features"]
        ]
    )
    modulation_config = config["evaluation"]["fixed_modulation_normalization"]

    def modulation(field: str, values: np.ndarray) -> np.ndarray:
        center, scale = (float(value) for value in modulation_config[field])
        if scale <= 0:
            raise GravityItem15TimescaleError("fixed Item 15 modulation changed")
        return (values - center) / scale

    stellar_ratio = np.asarray(
        [float(row["stellar_outer_to_inner_span_ratio"]) for row in rows]
    )
    halpha_ratio = np.asarray(
        [float(row["halpha_outer_to_inner_span_ratio"]) for row in rows]
    )
    return {
        "summary": summary,
        "candidate_manifest": candidate_manifest,
        "rows": rows,
        "folds": np.asarray([int(row["outer_fold"]) for row in rows]),
        "y": np.log10(stellar_ratio),
        "y_halpha": np.log10(halpha_ratio),
        "design_control": structural,
        "design_secondary": structural,
        "log_dynamical": _fixed_array(rows, config, "log_dynamical_time_gyr"),
        "log_growth": _fixed_array(rows, config, "log_mass_doubling_time_gyr"),
        "log_cosmic": _fixed_array(rows, config, "log_cosmic_age_gyr"),
        "log_relaxation": _fixed_array(rows, config, "log_relaxation_time_gyr"),
        "cosmic_dynamical": _fixed_array(rows, config, "log_cosmic_to_dynamical"),
        "growth_dynamical": _fixed_array(rows, config, "log_growth_to_dynamical"),
        "growth_cosmic": _fixed_array(rows, config, "log_growth_to_cosmic"),
        "relaxation_cosmic": _fixed_array(rows, config, "log_relaxation_to_cosmic"),
        "relaxation_growth": _fixed_array(rows, config, "log_relaxation_to_growth"),
        "clock_hierarchy": _fixed_array(rows, config, "clock_hierarchy_span"),
        "clock_entropy": _fixed_array(rows, config, "clock_entropy"),
        "surface_modulation": modulation(
            "stellar_surface_density",
            np.asarray([float(row["log_surface_density"]) for row in rows]),
        ),
        "age_modulation": modulation(
            "prior_age_lead",
            np.asarray([float(row["prior_age_lead"]) for row in rows]),
        ),
        "sersic_modulation": modulation(
            "sersic_index", np.asarray([float(row["sersic_index"]) for row in rows])
        ),
        "mass_modulation": modulation(
            "stellar_mass", np.asarray([float(row["log_stellar_mass"]) for row in rows])
        ),
        "redshift_modulation": modulation(
            "redshift", np.asarray([float(row["redshift"]) for row in rows])
        ),
        "growth_state": np.asarray(
            [1.0 if row["growth_state"] == "slow_growth" else -1.0 for row in rows]
        ),
        "mass": np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        "dynamical": np.asarray(
            [float(row["log_dynamical_time_gyr"]) for row in rows]
        ),
        "cosmic": np.asarray([float(row["log_cosmic_age_gyr"]) for row in rows]),
    }


def _candidate_components(
    arrays: Mapping[str, np.ndarray],
    data: Mapping[str, Any],
    begin: int,
    end: int,
    xp: Any,
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.float64)[:, None]
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(
        arrays["modulation"][begin:end], dtype=xp.int32
    )[:, None]

    def value(key: str) -> Any:
        return xp.asarray(data[key], dtype=xp.float64)[None, :]

    cosmic_dynamical = value("cosmic_dynamical")
    growth_dynamical = value("growth_dynamical")
    growth_cosmic = value("growth_cosmic")
    relaxation_cosmic = value("relaxation_cosmic")
    relaxation_growth = value("relaxation_growth")
    hierarchy = value("clock_hierarchy")
    entropy = value("clock_entropy")

    def signed_power(raw: Any) -> Any:
        z = (raw - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    dynamic_cosmic = signed_power(cosmic_dynamical)
    growth_dynamic = signed_power(growth_dynamical)
    growth_cosmic_ratio = signed_power(growth_cosmic)
    relaxation_cosmic_ratio = signed_power(relaxation_cosmic)
    relaxation_growth_ratio = signed_power(relaxation_growth)
    hierarchy_span = signed_power(hierarchy)
    crossover_base = xp.tanh((growth_cosmic - threshold) / scale)
    crossover = xp.sign(crossover_base) * xp.abs(crossover_base) ** power
    resonance = xp.exp(-0.5 * ((growth_dynamical - threshold) / scale) ** 2) ** power
    competing = signed_power(cosmic_dynamical * growth_cosmic)
    entropy_suppression = xp.exp(-xp.abs((entropy - threshold) / scale)) ** power
    phase_lock = xp.tanh(
        (cosmic_dynamical - threshold)
        * xp.cos(phase + power * growth_cosmic)
        / xp.maximum(scale, 1e-12)
    )
    log_periodic = xp.cos(
        phase + power * xp.log1p(xp.abs(growth_dynamical)) / xp.maximum(scale, 1e-12)
    )
    components = xp.where(family == 0, dynamic_cosmic, growth_dynamic)
    components = xp.where(family == 2, growth_cosmic_ratio, components)
    components = xp.where(family == 3, relaxation_cosmic_ratio, components)
    components = xp.where(family == 4, relaxation_growth_ratio, components)
    components = xp.where(family == 5, hierarchy_span, components)
    components = xp.where(family == 6, crossover, components)
    components = xp.where(family == 7, resonance, components)
    components = xp.where(family == 8, competing, components)
    components = xp.where(family == 9, entropy_suppression, components)
    components = xp.where(family == 10, phase_lock, components)
    components = xp.where(family == 11, log_periodic, components)
    modulations = xp.stack(
        (
            xp.ones_like(value("surface_modulation")),
            xp.tanh(value("surface_modulation")),
            xp.tanh(value("age_modulation")),
            xp.tanh(value("sersic_modulation")),
            xp.tanh(value("mass_modulation")),
            xp.tanh(value("redshift_modulation")),
        ),
        axis=0,
    )[:, 0, :]
    selected_modulation = xp.take_along_axis(
        modulations[None, :, :], modulation_index[:, :, None], axis=1
    )[:, 0, :]
    return components * selected_modulation


def _fit_component(
    component: np.ndarray, residual: np.ndarray, ridge: float
) -> tuple[float, float, float]:
    mean = float(np.mean(component))
    scale = max(float(np.std(component)), 1e-12)
    standardized = (component - mean) / scale
    coefficient = float(
        np.sum(standardized * residual) / (np.sum(standardized**2) + ridge)
    )
    return mean, scale, coefficient


def _nested_select(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    started = time.perf_counter()
    try:
        import cupy as xp

        if int(xp.cuda.runtime.getDeviceCount()) < 1:
            raise RuntimeError("no CUDA device")
        backend = "gpu_cupy"
        device = xp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except (ImportError, RuntimeError):
        xp = np
        backend = "cpu_numpy"
        device = None
    arrays = generate_candidates(config)
    candidate_count = len(arrays["family"])
    folds = np.asarray(data["folds"])
    y = np.asarray(data["y"])
    y_halpha = np.asarray(data["y_halpha"])
    control = np.asarray(data["design_control"])
    secondary = np.asarray(data["design_secondary"])
    predictions = {
        key: np.full(len(y), np.nan)
        for key in ("control", "full", "secondary_control", "secondary_full")
    }
    selections = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["timescale_coefficient_ridge"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    component_crosscheck = 0.0
    for outer in range(outer_folds):
        inner_records = []
        for inner in [value for value in range(outer_folds) if value != outer]:
            train = (folds != outer) & (folds != inner)
            validation = folds == inner
            model = _ridge_fit(control[train], y[train], alpha)
            inner_records.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_residual": y[train] - _ridge_predict(model, control[train]),
                    "validation_residual": y[validation]
                    - _ridge_predict(model, control[validation]),
                }
            )
        scores = np.full(candidate_count, np.inf)
        for begin in range(0, candidate_count, batch_size):
            end = min(begin + batch_size, candidate_count)
            components = _candidate_components(arrays, data, begin, end, xp)
            loss = xp.zeros(end - begin, dtype=xp.float64)
            for inner in inner_records:
                train_component = components[:, inner["train"]]
                validation_component = components[:, inner["validation"]]
                mean = xp.mean(train_component, axis=1)
                std = xp.maximum(xp.std(train_component, axis=1), 1e-12)
                standardized = (train_component - mean[:, None]) / std[:, None]
                coefficient = xp.sum(
                    standardized * xp.asarray(inner["train_residual"])[None, :], axis=1
                ) / (xp.sum(standardized**2, axis=1) + coefficient_ridge)
                residual = (
                    xp.asarray(inner["validation_residual"])[None, :]
                    - coefficient[:, None]
                    * (validation_component - mean[:, None])
                    / std[:, None]
                )
                loss += xp.mean(residual**2, axis=1)
            batch_scores = loss / len(inner_records)
            scores[begin:end] = (
                xp.asnumpy(batch_scores) if backend == "gpu_cupy" else batch_scores
            )
            if begin == 0:
                check_count = min(
                    int(config["evaluation"]["cpu_crosscheck_candidates"]), end
                )
                cpu = _candidate_components(arrays, data, 0, check_count, np)
                observed = (
                    xp.asnumpy(components[:check_count])
                    if backend == "gpu_cupy"
                    else components[:check_count]
                )
                component_crosscheck = max(
                    component_crosscheck, float(np.max(np.abs(cpu - observed)))
                )
        selected = int(np.argmin(scores))
        train = folds != outer
        test = folds == outer
        control_model = _ridge_fit(control[train], y[train], alpha)
        control_train = _ridge_predict(control_model, control[train])
        predictions["control"][test] = _ridge_predict(control_model, control[test])
        selected_component = _candidate_components(
            arrays, data, selected, selected + 1, np
        )[0]
        mean, std, coefficient = _fit_component(
            selected_component[train], y[train] - control_train, coefficient_ridge
        )
        predictions["full"][test] = (
            predictions["control"][test]
            + coefficient * (selected_component[test] - mean) / std
        )
        secondary_model = _ridge_fit(secondary[train], y_halpha[train], alpha)
        secondary_train = _ridge_predict(secondary_model, secondary[train])
        predictions["secondary_control"][test] = _ridge_predict(
            secondary_model, secondary[test]
        )
        secondary_mean, secondary_std, secondary_coefficient = _fit_component(
            selected_component[train],
            y_halpha[train] - secondary_train,
            coefficient_ridge,
        )
        predictions["secondary_full"][test] = (
            predictions["secondary_control"][test]
            + secondary_coefficient
            * (selected_component[test] - secondary_mean)
            / secondary_std
        )
        family = config["candidate_generator"]["families"][
            int(arrays["family"][selected])
        ]
        selections.append(
            {
                "outer_fold": outer,
                "selected_ordinal": selected,
                "selected_family": family["id"],
                "origin_status": family["origin_status"],
                "qualifying": bool(family["qualifying"]),
                "threshold": _metric(arrays["threshold"][selected]),
                "scale": _metric(arrays["scale"][selected]),
                "power": _metric(arrays["power"][selected]),
                "phase": _metric(arrays["phase"][selected]),
                "modulation": config["candidate_generator"]["modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_mse": _metric(scores[selected]),
                "fitted_timescale_coefficient": _metric(coefficient),
                "fitted_secondary_coefficient": _metric(secondary_coefficient),
                "test_galaxies": int(np.sum(test)),
            }
        )
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise GravityItem15TimescaleError("Item 15 OOF prediction incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return (
        predictions,
        selections,
        {
            "backend": backend,
            "device": device,
            "cupy_version": getattr(xp, "__version__", None)
            if backend == "gpu_cupy"
            else None,
            "elapsed_seconds": _metric(elapsed),
            "candidate_cells": candidate_count,
            "galaxies": len(y),
            "outer_folds": outer_folds,
            "inner_validation_fits_per_outer": outer_folds - 1,
            "candidate_galaxy_score_evaluations": candidate_count
            * len(y)
            * outer_folds
            * (outer_folds - 1),
            "cpu_crosscheck_candidates": int(
                config["evaluation"]["cpu_crosscheck_candidates"]
            ),
            "cpu_gpu_max_component_difference": _metric(component_crosscheck),
        },
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, str]:
    mse = float(np.mean((y - prediction) ** 2))
    variance = float(np.var(y))
    return {
        "mse": _metric(mse),
        "r2": _metric(1.0 - mse / variance if variance > 0 else 0.0),
    }


def _paired_sign_flip(
    differences: np.ndarray, config: Mapping[str, Any]
) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    seed = int(
        hashlib.sha256(str(config["evaluation"]["permutation_salt"]).encode()).hexdigest()[:16],
        16,
    )
    random = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null = np.asarray(
        [
            np.mean(differences * random.choice([-1.0, 1.0], len(differences)))
            for _ in range(count)
        ]
    )
    return {
        "permutations": count,
        "observed_mean_mse_gain": _metric(observed),
        "p_value": _metric((1 + int(np.sum(null >= observed))) / (count + 1)),
        "null_gain_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.5))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_data(root, config)
    predictions, selections, compute = _nested_select(data, config)
    primary_control = _metrics(data["y"], predictions["control"])
    primary_full = _metrics(data["y"], predictions["full"])
    secondary_control = _metrics(data["y_halpha"], predictions["secondary_control"])
    secondary_full = _metrics(data["y_halpha"], predictions["secondary_full"])
    control_mse = float(primary_control["mse"])
    full_mse = float(primary_full["mse"])
    relative = (control_mse - full_mse) / control_mse
    paired = _paired_sign_flip(
        (data["y"] - predictions["control"]) ** 2
        - (data["y"] - predictions["full"]) ** 2,
        config,
    )
    sign_agreement_folds = sum(
        float(row["fitted_timescale_coefficient"])
        * float(row["fitted_secondary_coefficient"])
        > 0
        for row in selections
    )
    dimensions = {
        "growth_state": (data["growth_state"], 0.0),
        "stellar_mass_half": (
            data["mass"],
            float(config["sample"]["stellar_mass_threshold_log10"]),
        ),
        "dynamical_time_half": (
            data["dynamical"],
            float(config["sample"]["dynamical_time_threshold_log10_gyr"]),
        ),
        "cosmic_age_half": (
            data["cosmic"],
            float(config["sample"]["cosmic_age_threshold_log10_gyr"]),
        ),
    }
    strata = []
    stratum_pass = {}
    for dimension, (values, split) in dimensions.items():
        gains = []
        for label, mask in (("low", values <= split), ("high", values > split)):
            baseline = float(
                np.mean((data["y"][mask] - predictions["control"][mask]) ** 2)
            )
            proposed = float(
                np.mean((data["y"][mask] - predictions["full"][mask]) ** 2)
            )
            gain = baseline - proposed
            gains.append(gain)
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    "galaxies": int(np.sum(mask)),
                    "control_mse": _metric(baseline),
                    "full_model_mse": _metric(proposed),
                    "timescale_mse_gain": _metric(gain),
                }
            )
        stratum_pass[dimension] = all(value > 0 for value in gains)
    summary = data["summary"]
    gates = {
        "quality_count_and_fraction_pass": summary["decision"]
        == "PASS_ITEM15_TIMESCALE_QUALITY",
        "fresh_identity_and_confirmation_boundary_pass": summary["counts"][
            "predecessor_selected"
        ]
        == 0
        and summary["counts"]["confirmation_response_rows"] == 0,
        "candidate_count_exact": compute["candidate_cells"] == 262144,
        "full_model_r2_positive": float(primary_full["r2"]) > 0,
        "timescale_beats_source_variable_baseline": full_mse < control_mse,
        "timescale_relative_mse_improvement_at_least": relative
        >= float(config["admission"]["timescale_relative_mse_improvement_at_least"]),
        "timescale_paired_p_at_most": float(paired["p_value"])
        <= float(config["admission"]["timescale_paired_p_at_most"]),
        "secondary_halpha_transfer_beats_control": float(secondary_full["mse"])
        < float(secondary_control["mse"]),
        "coefficient_sign_agreement_folds_at_least": sign_agreement_folds
        >= int(config["admission"]["coefficient_sign_agreement_folds_at_least"]),
        "gain_positive_in_both_growth_states": stratum_pass["growth_state"],
        "gain_positive_in_both_stellar_mass_halves": stratum_pass["stellar_mass_half"],
        "gain_positive_in_both_dynamical_time_halves": stratum_pass[
            "dynamical_time_half"
        ],
        "gain_positive_in_both_cosmic_age_halves": stratum_pass["cosmic_age_half"],
        "selected_family_is_qualifying_timescale_ratio": all(
            bool(row["qualifying"]) for row in selections
        ),
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM15_MANGA_TIMESCALE_RATIOS_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM15_MANGA_TIMESCALE_RATIOS_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM15_MANGA_TIMESCALE_QUALITY"
    input_keys = (
        "sample_manifest",
        "predictor_source",
        "candidate_manifest",
        "response_source",
        "extraction_summary",
    )
    input_paths = {key: root / config["outputs"][key] for key in input_keys}
    secondary_relative = (
        float(secondary_control["mse"]) - float(secondary_full["mse"])
    ) / float(secondary_control["mse"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item15-manga-timescale-result-1.0",
            "goal": config["goal"],
            "item_number": 15,
            "attempt_number": 1,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "attempt_scope": config["scientific_contract"]["attempt_scope"],
            "response_boundary": config["scientific_contract"]["interpretation_boundary"],
            "counts": {
                "candidate_cells": 262144,
                "quality_passing_galaxies": summary["counts"][
                    "quality_passing_galaxies"
                ],
                "quality_failed_galaxies": summary["counts"]["quality_failed_galaxies"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {
                key + "_sha256": _sha256_file(path) for key, path in input_paths.items()
            },
            "primary_stellar_outer_to_inner_log_span_ratio": {
                "source_variable_control_baseline": primary_control,
                "selected_timescale_full_model": primary_full,
                "relative_mse_improvement": _metric(relative),
                "outer_fold_selections": selections,
            },
            "secondary_halpha_outer_to_inner_log_span_ratio": {
                "source_variable_control_baseline": secondary_control,
                "selected_timescale_full_model": secondary_full,
                "relative_mse_improvement": _metric(secondary_relative),
                "candidate_reselection": False,
                "coefficient_sign_agreement_folds": sign_agreement_folds,
            },
            "resolved_ratio_distribution": {
                "stellar_median_outer_to_inner": _metric(
                    float(np.median(10.0 ** data["y"]))
                ),
                "halpha_median_outer_to_inner": _metric(
                    float(np.median(10.0 ** data["y_halpha"]))
                ),
            },
            "timescale_distribution": {
                key: {
                    "median": _metric(
                        float(np.median([float(row[key]) for row in data["rows"]]))
                    ),
                    "q10": _metric(
                        float(np.quantile([float(row[key]) for row in data["rows"]], 0.1))
                    ),
                    "q90": _metric(
                        float(np.quantile([float(row[key]) for row in data["rows"]], 0.9))
                    ),
                }
                for key in (
                    "dynamical_time_gyr",
                    "mass_doubling_time_gyr",
                    "cosmic_age_gyr",
                    "relaxation_time_gyr",
                )
            },
            "paired_sign_flip": paired,
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "compute": compute,
            "equivalence_boundary": config["timescale_features"]["equivalence_boundary"],
            "limitations": {
                "direct_hot_gas_cooling_time_tested": False,
                "orbital_crossing_freefall_are_independent_laws": False,
                "mass_doubling_time_is_direct_stellar_age": False,
                "two_body_relaxation_is_a_plausible_galaxy_age": False,
                "annular_spans_are_deprojected_circular_speeds": False,
                "same_sdss_manga_survey": True,
                "item15_broad_synthesis_complete": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 15 result")
    config = load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem15TimescaleError("Item 15 result sample binding changed")
    if int(receipt["counts"]["candidate_cells"]) != int(
        config["candidate_generator"]["candidate_cells"]
    ):
        raise GravityItem15TimescaleError("Item 15 result candidate count changed")
    for key in ("confirmation_response_rows", "post_response_formula_cells", "paid_model_calls"):
        if int(receipt["counts"][key]) != 0:
            raise GravityItem15TimescaleError(f"Item 15 forbidden result count: {key}")
    if bool(receipt["limitations"]["direct_hot_gas_cooling_time_tested"]):
        raise GravityItem15TimescaleError("Item 15 result fabricated cooling coverage")
    if bool(receipt["limitations"]["item15_broad_synthesis_complete"]):
        raise GravityItem15TimescaleError("Item 15 attempt overclaims broad completion")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem15TimescaleError("Item 15 result contains an overclaim")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root)
    path = root / config["outputs"]["result"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["outputs"]["result"]).read_text(encoding="utf-8"))
    validate_receipt(stored, root)
    rebuilt = build_receipt(root)
    for value in (stored, rebuilt):
        value.pop("content_sha256", None)
        value["compute"] = dict(value["compute"])
        value["compute"].pop("elapsed_seconds", None)
    if stored != rebuilt:
        raise GravityItem15TimescaleError("Item 15 result receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "responses", "extract", "run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "prepare":
        print("\n".join(str(path) for path in write_prepared_sources(args.root)))
    elif args.command == "responses":
        print(write_response_source(args.root))
    elif args.command == "extract":
        print(extract_rows(args.root))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
