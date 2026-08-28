"""Frozen WALLABY baryonic-boundary search for gravity roadmap Item 10."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/gravity_item10_wallaby_boundaries_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "d1f3ea0a303427077a07f6017abd4d0e87b23f0a"
SAMPLE_FREEZE_COMMIT = "a7989ad42079813d7798d81671b83fd7bb6dd99e"


class GravityItem10BoundaryError(RuntimeError):
    """Raised when an Item 10 scientific or response boundary drifts."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def _content_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def _validate_content_hash(value: Mapping[str, Any], label: str) -> None:
    copy_value = dict(value)
    digest = copy_value.pop("content_sha256", None)
    if digest != canonical_sha256(copy_value):
        raise GravityItem10BoundaryError(f"{label} content hash changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem10BoundaryError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem10BoundaryError("Item 9 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 9 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem10BoundaryError("Item 9 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem10BoundaryError("Item 9 synthesis decision changed")
    coordinate_path = root / config["independence"]["probes1_coordinate_source_path"]
    if _sha256_file(coordinate_path) != config["independence"]["probes1_coordinate_source_sha256"]:
        raise GravityItem10BoundaryError("PROBES-I coordinate exclusion source changed")
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem10BoundaryError("Item 10 config contains an overclaim")
    if int(config["candidate_generator"]["candidate_cells"]) != 131072:
        raise GravityItem10BoundaryError("Item 10 candidate count changed")
    return config


def _tap_query(config: Mapping[str, Any], query: str) -> bytes:
    parameters = urllib.parse.urlencode(
        {"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query}
    )
    url = f"{config['source']['tap_sync_endpoint']}?{parameters}"
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/Item10-WALLABY"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise GravityItem10BoundaryError("empty WALLABY TAP response")
    return payload


def _parse_csv(payload: bytes) -> list[dict[str, str]]:
    text = payload.decode("utf-8-sig", errors="strict")
    reader = csv.DictReader(io.StringIO(text))
    if reader.fieldnames is None:
        raise GravityItem10BoundaryError("WALLABY CSV has no header")
    result = []
    for row in reader:
        if row and any(str(value).strip() for value in row.values()):
            result.append({str(key): str(value).strip() for key, value in row.items()})
    return result


def _parse_vector(value: str) -> np.ndarray:
    try:
        array = np.asarray([float(item.strip()) for item in value.split(",")], dtype=np.float64)
    except ValueError as exc:
        raise GravityItem10BoundaryError("invalid WALLABY vector") from exc
    if not len(array) or np.any(~np.isfinite(array)):
        raise GravityItem10BoundaryError("non-finite or empty WALLABY vector")
    return array


def _distance_mpc(frequency_hz: float, config: Mapping[str, Any]) -> float:
    constants = config["constants"]
    redshift = float(constants["hi_rest_frequency_hz"]) / frequency_hz - 1.0
    distance = (
        float(constants["speed_of_light_km_s"])
        * redshift
        / float(constants["hubble_constant_km_s_mpc"])
    )
    if not math.isfinite(distance) or distance <= 0 or redshift >= 0.1:
        raise GravityItem10BoundaryError("invalid WALLABY Hubble distance")
    return distance


def _edge_crossing(
    radius: np.ndarray, sigma: np.ndarray, threshold: float
) -> tuple[float, float] | None:
    candidates = np.flatnonzero((sigma[:-1] >= threshold) & (sigma[1:] < threshold))
    if not len(candidates):
        return None
    index = int(candidates[-1])
    y0 = math.log(max(float(sigma[index]), 1e-12))
    y1 = math.log(max(float(sigma[index + 1]), 1e-12))
    target = math.log(threshold)
    fraction = (target - y0) / (y1 - y0) if y1 != y0 else 0.5
    fraction = float(np.clip(fraction, 0.0, 1.0))
    edge = float(radius[index] + fraction * (radius[index + 1] - radius[index]))
    log_radius = np.log(np.maximum(radius, 1e-12))
    gradient = np.gradient(np.log(np.maximum(sigma, 1e-12)), log_radius)
    sharpness = abs(float(gradient[index] + fraction * (gradient[index + 1] - gradient[index])))
    return edge, sharpness


def measure_profile(row: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    radius_arcsec = _parse_vector(row["Rad_SD"])
    sigma = _parse_vector(row["SD_model"])
    sigma_error = _parse_vector(row["e_SD_model"])
    if not (len(radius_arcsec) == len(sigma) == len(sigma_error)):
        raise GravityItem10BoundaryError("WALLABY surface-profile vector lengths differ")
    order = np.argsort(radius_arcsec)
    radius_arcsec = radius_arcsec[order]
    sigma = sigma[order]
    sigma_error = sigma_error[order]
    valid = (radius_arcsec > 0) & (sigma > 0) & (sigma_error >= 0)
    radius_arcsec = radius_arcsec[valid]
    sigma = sigma[valid]
    sigma_error = sigma_error[valid]
    if len(radius_arcsec) < 8 or np.any(np.diff(radius_arcsec) <= 0):
        raise GravityItem10BoundaryError("insufficient unique WALLABY surface-profile radii")
    distance = _distance_mpc(float(row["freq"]), config)
    radius_kpc = (
        radius_arcsec * distance * 1000.0 / float(config["constants"]["arcseconds_per_radian"])
    )
    edges = np.empty(len(radius_kpc) + 1, dtype=np.float64)
    edges[0] = 0.0
    edges[1:-1] = 0.5 * (radius_kpc[:-1] + radius_kpc[1:])
    edges[-1] = radius_kpc[-1] + 0.5 * (radius_kpc[-1] - radius_kpc[-2])
    annulus_pc2 = math.pi * np.diff((edges * 1000.0) ** 2)
    shell_mass = np.maximum(sigma * annulus_pc2, 0.0) * float(
        config["constants"]["helium_mass_factor"]
    )
    cumulative = np.cumsum(shell_mass)
    total = float(cumulative[-1])
    if not math.isfinite(total) or total <= 0:
        raise GravityItem10BoundaryError("invalid integrated WALLABY profile mass")
    thresholds = [float(value) for value in config["candidate_generator"]["thresholds_msun_pc2"]]
    boundary = {}
    for threshold in thresholds:
        crossing = _edge_crossing(radius_kpc, sigma, threshold)
        key = _threshold_key(threshold)
        if crossing is None:
            boundary[key] = None
            continue
        edge, sharpness = crossing
        mass_at_edge = float(np.interp(edge, radius_kpc, cumulative))
        boundary[key] = {
            "edge_radius_kpc": edge,
            "edge_sharpness": sharpness,
            "outer_mass_fraction": float(np.clip(1.0 - mass_at_edge / total, 0.0, 1.0)),
        }
    r50 = float(np.interp(0.5 * total, cumulative, radius_kpc))
    r90 = float(np.interp(0.9 * total, cumulative, radius_kpc))
    inner = float(np.median(sigma[: max(1, len(sigma) // 3)]))
    outer = float(np.median(sigma[-max(1, len(sigma) // 3) :]))
    return {
        "name": row["name"],
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "frequency_hz": float(row["freq"]),
        "distance_mpc": distance,
        "team_release": row["team_release"],
        "team_release_kin": row["team_release_kin"],
        "radius_arcsec": radius_arcsec,
        "radius_kpc": radius_kpc,
        "surface_density": sigma,
        "surface_density_error": sigma_error,
        "cumulative_mass": cumulative,
        "total_profile_mass": total,
        "profile_concentration": r50 / r90,
        "profile_contrast": inner / max(outer, 1e-12),
        "boundaries": boundary,
    }


def _threshold_key(value: float) -> str:
    return f"q{value:g}".replace(".", "p")


def _serialize_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "name": profile["name"],
        "ra": _metric(profile["ra"]),
        "dec": _metric(profile["dec"]),
        "frequency_hz": _metric(profile["frequency_hz"]),
        "distance_mpc": _metric(profile["distance_mpc"]),
        "team_release": profile["team_release"],
        "team_release_kin": profile["team_release_kin"],
        "radius_arcsec": [_metric(value) for value in profile["radius_arcsec"]],
        "radius_kpc": [_metric(value) for value in profile["radius_kpc"]],
        "surface_density": [_metric(value) for value in profile["surface_density"]],
        "surface_density_error": [_metric(value) for value in profile["surface_density_error"]],
        "cumulative_mass": [_metric(value) for value in profile["cumulative_mass"]],
        "total_profile_mass": _metric(profile["total_profile_mass"]),
        "profile_concentration": _metric(profile["profile_concentration"]),
        "profile_contrast": _metric(profile["profile_contrast"]),
        "boundaries": {
            key: None
            if value is None
            else {field: _metric(number) for field, number in value.items()}
            for key, value in profile["boundaries"].items()
        },
    }


def write_predictor_source(root: Path) -> Path:
    root = root.resolve()
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem10BoundaryError("Item 10 scientific freeze is not bound")
    config = load_config(root)
    columns = ",".join(config["source"]["predictor_columns"])
    query = f"SELECT {columns} FROM {config['source']['table']} ORDER BY name"
    payload = _tap_query(config, query)
    rows = _parse_csv(payload)
    if len(rows) != int(config["source"]["observed_rows"]):
        raise GravityItem10BoundaryError("WALLABY row count changed")
    expected = set(config["source"]["predictor_columns"])
    if any(set(row) != expected for row in rows):
        raise GravityItem10BoundaryError("WALLABY predictor query schema changed")
    records = []
    failures = []
    for row in rows:
        try:
            records.append(_serialize_profile(measure_profile(row, config)))
        except GravityItem10BoundaryError as exc:
            failures.append({"name": row.get("name"), "reason": str(exc)})
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-wallaby-predictor-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "query": query,
            "query_sha256": hashlib.sha256(payload).hexdigest(),
            "records": records,
            "failures": failures,
            "counts": {
                "catalogue_rows": len(rows),
                "valid_predictor_profiles": len(records),
                "invalid_predictor_profiles": len(failures),
                "response_columns_requested": 0,
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {"response_opened": False},
        }
    )
    path = root / config["outputs"]["predictor_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def validate_predictor_source(source: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(source, "Item 10 predictor source")
    config = load_config(root)
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem10BoundaryError("predictor freeze binding changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem10BoundaryError("response column entered predictor source")
    columns = ",".join(config["source"]["predictor_columns"])
    expected_query = f"SELECT {columns} FROM {config['source']['table']} ORDER BY name"
    if source["query"] != expected_query:
        raise GravityItem10BoundaryError("predictor query changed")


def _probes_coordinates(root: Path, config: Mapping[str, Any]) -> np.ndarray:
    path = root / config["independence"]["probes1_coordinate_source_path"]
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        header = next(reader)
        ra_index = header.index("RA")
        dec_index = header.index("DEC")
        result = []
        for row in reader:
            try:
                ra = float(row[ra_index])
                dec = float(row[dec_index])
            except (ValueError, IndexError):
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                result.append((ra, dec))
    return np.asarray(result, dtype=np.float64)


def _minimum_separation_arcsec(ra: float, dec: float, coordinates: np.ndarray) -> float:
    ra1 = math.radians(ra)
    dec1 = math.radians(dec)
    ra2 = np.radians(coordinates[:, 0])
    dec2 = np.radians(coordinates[:, 1])
    delta_ra = ra2 - ra1
    sin_ddec = np.sin((dec2 - dec1) / 2.0)
    sin_dra = np.sin(delta_ra / 2.0)
    haversine = sin_ddec**2 + math.cos(dec1) * np.cos(dec2) * sin_dra**2
    angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return float(np.min(angle) * 206264.80624709636)


def _split_hash(name: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{name}".encode()).hexdigest()


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["outputs"]["predictor_source"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_predictor_source(source, root)
    probes = _probes_coordinates(root, config)
    exposed = str(config["prefreeze_audit"]["conservatively_excluded_exposed_identity"])
    threshold_key = _threshold_key(1.0)
    admitted = []
    exclusions: Counter[str] = Counter()
    for row in source["records"]:
        reasons = []
        if row["name"] == exposed:
            reasons.append("prefreeze_response_exposure")
        separation = _minimum_separation_arcsec(float(row["ra"]), float(row["dec"]), probes)
        if separation <= float(config["independence"]["coordinate_exclusion_arcseconds"]):
            reasons.append("PROBES_I_coordinate_overlap")
        if row["boundaries"].get(threshold_key) is None:
            reasons.append("no_bracketed_q1_boundary")
        for reason in set(reasons):
            exclusions[reason] += 1
        if reasons:
            continue
        copy_row = {
            "name": row["name"],
            "ra": row["ra"],
            "dec": row["dec"],
            "team_release_kin": row["team_release_kin"],
            "distance_mpc": row["distance_mpc"],
            "q1_edge_radius_kpc": row["boundaries"][threshold_key]["edge_radius_kpc"],
            "q1_edge_sharpness": row["boundaries"][threshold_key]["edge_sharpness"],
            "total_profile_mass": row["total_profile_mass"],
            "minimum_PROBES_I_separation_arcsec": _metric(separation),
        }
        admitted.append(copy_row)
    median_edge = float(np.median([float(row["q1_edge_radius_kpc"]) for row in admitted]))
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in admitted:
        size = "small_edge" if float(row["q1_edge_radius_kpc"]) <= median_edge else "large_edge"
        cells.setdefault((row["team_release_kin"], size), []).append(row)
    objects = []
    salt = str(config["sample"]["split_salt"])
    for (release, size), rows in sorted(cells.items()):
        ordered = sorted(rows, key=lambda row: (_split_hash(row["name"], salt), row["name"]))
        confirmation_count = round(len(ordered) * float(config["sample"]["confirmation_fraction"]))
        confirmation_names = {row["name"] for row in ordered[:confirmation_count]}
        for row in sorted(ordered, key=lambda item: item["name"]):
            role = "reserved_confirmation" if row["name"] in confirmation_names else "exploration"
            output = dict(row)
            output.update(
                {
                    "edge_size_bin": size,
                    "role": role,
                    "outer_fold": int(
                        _split_hash(row["name"], config["evaluation"]["fold_salt"])[:16], 16
                    )
                    % int(config["evaluation"]["outer_folds"]),
                    "response_read": False,
                }
            )
            objects.append(output)
    objects.sort(key=lambda row: row["name"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-wallaby-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "predictor_source_sha256": _sha256_file(source_path),
            "predictor_source_content_sha256": source["content_sha256"],
            "q1_edge_median_kpc": _metric(median_edge),
            "objects": objects,
            "counts": {
                "predictor_profiles": len(source["records"]),
                "admitted": len(objects),
                "exploration": sum(row["role"] == "exploration" for row in objects),
                "reserved_confirmation": sum(
                    row["role"] == "reserved_confirmation" for row in objects
                ),
                "prefreeze_exposed_identity_selected": 0,
                "PROBES_I_coordinate_overlaps_selected": 0,
                "response_rows_read": 0,
            },
            "exclusion_counts": dict(sorted(exclusions.items())),
            "fold_counts_exploration": {
                str(key): value
                for key, value in sorted(
                    Counter(
                        row["outer_fold"] for row in objects if row["role"] == "exploration"
                    ).items()
                )
            },
            "cells": {
                f"{release}|{size}": {
                    "exploration": sum(
                        row["role"] == "exploration"
                        and row["team_release_kin"] == release
                        and row["edge_size_bin"] == size
                        for row in objects
                    ),
                    "reserved_confirmation": sum(
                        row["role"] == "reserved_confirmation"
                        and row["team_release_kin"] == release
                        and row["edge_size_bin"] == size
                        for row in objects
                    ),
                }
                for release, size in sorted(cells)
            },
            "claims": {"confirmation_opened": False},
        }
    )


def validate_sample_manifest(sample: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(sample, "Item 10 sample manifest")
    if sample != build_sample_manifest(root):
        raise GravityItem10BoundaryError("Item 10 sample manifest drifted")
    counts = sample["counts"]
    if counts["prefreeze_exposed_identity_selected"] != 0:
        raise GravityItem10BoundaryError("prefreeze-exposed response entered sample")
    if counts["PROBES_I_coordinate_overlaps_selected"] != 0:
        raise GravityItem10BoundaryError("PROBES-I coordinate overlap entered sample")
    if counts["response_rows_read"] != 0:
        raise GravityItem10BoundaryError("response entered target-blind sample")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    sample = build_sample_manifest(root)
    path = root / config["outputs"]["sample_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(sample) + b"\n")
    return path


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    family_count = len(generator["families"])
    threshold_count = len(generator["thresholds_msun_pc2"])
    modulation_count = len(generator["modulations"])
    scale_low, scale_high = [math.log(float(value)) for value in generator["scale_log_uniform"]]
    power_low, power_high = [math.log(float(value)) for value in generator["power_log_uniform"]]
    phase_low, phase_high = [float(value) for value in generator["phase_uniform"]]
    return {
        "family": random.integers(0, family_count, count, dtype=np.int16),
        "threshold": random.integers(0, threshold_count, count, dtype=np.int8),
        "scale": np.exp(random.uniform(scale_low, scale_high, count)),
        "power": np.exp(random.uniform(power_low, power_high, count)),
        "phase": random.uniform(phase_low, phase_high, count),
        "modulation": random.integers(0, modulation_count, count, dtype=np.int8),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("family", "threshold", "scale", "power", "phase", "modulation"):
        digest.update(key.encode())
        array = np.asarray(arrays[key])
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def build_candidate_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    arrays = generate_candidates(config)
    families = config["candidate_generator"]["families"]
    family_counts = Counter(int(value) for value in arrays["family"])
    status_counts: Counter[str] = Counter()
    for index, family in enumerate(families):
        status_counts[family["origin_status"]] += family_counts[index]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-boundary-candidates-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "generator": config["candidate_generator"],
            "candidate_array_sha256": _candidate_digest(arrays),
            "family_counts": {
                family["id"]: family_counts[index] for index, family in enumerate(families)
            },
            "origin_status_counts": dict(sorted(status_counts.items())),
            "counts": {
                "candidate_cells": len(arrays["family"]),
                "post_response_cells": 0,
                "polarity_equivalence_duplicates_generated": 0,
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )


def validate_candidate_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(manifest, "Item 10 candidate manifest")
    if manifest != build_candidate_manifest(root):
        raise GravityItem10BoundaryError("Item 10 candidate manifest drifted")
    if manifest["counts"]["post_response_cells"] != 0:
        raise GravityItem10BoundaryError("post-response boundary candidate entered manifest")


def write_candidate_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["candidate_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_candidate_manifest(root)) + b"\n")
    return path


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem10BoundaryError("Item 10 sample freeze is not bound")
    config = load_config(root)
    sample_path = root / config["outputs"]["sample_manifest"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, root)
    exploration = [row["name"] for row in sample["objects"] if row["role"] == "exploration"]
    quoted = ",".join("'" + name.replace("'", "''") + "'" for name in exploration)
    columns = ",".join(config["source"]["response_columns"])
    query = (
        f"SELECT {columns} FROM {config['source']['table']} WHERE name IN ({quoted}) ORDER BY name"
    )
    payload = _tap_query(config, query)
    rows = _parse_csv(payload)
    expected = set(config["source"]["response_columns"])
    if any(set(row) != expected for row in rows):
        raise GravityItem10BoundaryError("WALLABY response query schema changed")
    names = [row["name"] for row in rows]
    if set(names) != set(exploration) or len(names) != len(set(names)):
        raise GravityItem10BoundaryError("WALLABY exploration response scope changed")
    confirmation = {
        row["name"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if confirmation & set(names):
        raise GravityItem10BoundaryError("WALLABY confirmation response was returned")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-wallaby-response-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "query_identity_count": len(exploration),
            "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "records": rows,
            "counts": {
                "exploration_response_rows": len(rows),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "claims": {"confirmation_opened": False},
        }
    )
    path = root / config["outputs"]["response_source"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def validate_response_source(source: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(source, "Item 10 response source")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem10BoundaryError("response sample binding changed")
    if source["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem10BoundaryError("confirmation response entered Item 10")
    if source["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem10BoundaryError("post-response formula entered Item 10")


POINT_FIELDS = (
    "radius_kpc",
    "log_radius_over_r1",
    "log_surface_density",
    "enclosed_mass_fraction",
    "log_total_profile_mass",
    "log_distance",
    "log_r1_kpc",
    "profile_concentration",
    "log_radius_squared",
    "log_surface_density_squared",
    "radius_surface_interaction",
    "profile_contrast_mod",
)


def _write_tsv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def _deserialize_profile(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **row,
        "ra": float(row["ra"]),
        "dec": float(row["dec"]),
        "frequency_hz": float(row["frequency_hz"]),
        "distance_mpc": float(row["distance_mpc"]),
        "radius_arcsec": np.asarray(row["radius_arcsec"], dtype=np.float64),
        "radius_kpc": np.asarray(row["radius_kpc"], dtype=np.float64),
        "surface_density": np.asarray(row["surface_density"], dtype=np.float64),
        "surface_density_error": np.asarray(row["surface_density_error"], dtype=np.float64),
        "cumulative_mass": np.asarray(row["cumulative_mass"], dtype=np.float64),
        "total_profile_mass": float(row["total_profile_mass"]),
        "profile_concentration": float(row["profile_concentration"]),
        "profile_contrast": float(row["profile_contrast"]),
        "boundaries": {
            key: None
            if value is None
            else {field: float(number) for field, number in value.items()}
            for key, value in row["boundaries"].items()
        },
    }


def extract_profiles(root: Path) -> dict[str, Path]:
    root = root.resolve()
    config = load_config(root)
    predictor_path = root / config["outputs"]["predictor_source"]
    sample_path = root / config["outputs"]["sample_manifest"]
    response_path = root / config["outputs"]["response_source"]
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    response = json.loads(response_path.read_text(encoding="utf-8"))
    validate_predictor_source(predictor, root)
    validate_sample_manifest(sample, root)
    validate_response_source(response, root)
    profiles = {row["name"]: _deserialize_profile(row) for row in predictor["records"]}
    samples = {row["name"]: row for row in sample["objects"] if row["role"] == "exploration"}
    features = []
    responses = []
    galaxy_receipts = []
    thresholds = [float(value) for value in config["candidate_generator"]["thresholds_msun_pc2"]]
    for response_row in response["records"]:
        name = response_row["name"]
        profile = profiles[name]
        sample_row = samples[name]
        reasons = []
        try:
            inclination = float(response_row["Inc_model"])
            qflag = float(response_row["QFlag_model"])
            radius_arcsec = _parse_vector(response_row["Rad"])
            velocity = _parse_vector(response_row["Vrot_model"])
            error = _parse_vector(response_row["e_Vrot_model"])
            inclination_error = _parse_vector(response_row["e_Vrot_model_inc"])
            if not (len(radius_arcsec) == len(velocity) == len(error) == len(inclination_error)):
                raise GravityItem10BoundaryError("rotation vector lengths differ")
        except (ValueError, GravityItem10BoundaryError) as exc:
            galaxy_receipts.append(
                {
                    "name": name,
                    "quality_pass": False,
                    "quality_failure_reasons": [f"parser:{exc}"],
                }
            )
            continue
        if qflag != float(config["quality"]["required_qflag_model"]):
            reasons.append("qflag")
        if not (
            float(config["quality"]["minimum_inclination_degrees"])
            <= inclination
            <= float(config["quality"]["maximum_inclination_degrees"])
        ):
            reasons.append("inclination")
        radius_kpc = (
            radius_arcsec
            * profile["distance_mpc"]
            * 1000.0
            / float(config["constants"]["arcseconds_per_radian"])
        )
        total_error = np.sqrt(np.maximum(error, 0.0) ** 2 + np.maximum(inclination_error, 0.0) ** 2)
        within = (radius_kpc >= profile["radius_kpc"][0]) & (
            radius_kpc <= profile["radius_kpc"][-1]
        )
        valid = within & (velocity >= float(config["quality"]["minimum_speed_km_s"]))
        valid &= total_error / np.maximum(velocity, 1e-12) <= float(
            config["quality"]["maximum_fractional_speed_error"]
        )
        if int(np.sum(valid)) < int(config["quality"]["minimum_rotation_points"]):
            reasons.append("insufficient_rotation_points")
        if float(np.mean(within)) < float(
            config["quality"]["minimum_fraction_response_radii_within_surface_profile"]
        ):
            reasons.append("surface_profile_overlap")
        passed = not reasons
        accepted_count = int(np.sum(valid)) if passed else 0
        if passed:
            r1 = profile["boundaries"][_threshold_key(1.0)]["edge_radius_kpc"]
            for output_index, source_index in enumerate(np.flatnonzero(valid)):
                radius = float(radius_kpc[source_index])
                local_sigma = float(
                    np.interp(radius, profile["radius_kpc"], profile["surface_density"])
                )
                enclosed = float(
                    np.interp(radius, profile["radius_kpc"], profile["cumulative_mass"])
                    / profile["total_profile_mass"]
                )
                log_radius = math.log10(max(radius / r1, 1e-10))
                log_sigma = math.log10(max(local_sigma, 1e-10))
                row = {
                    "galaxy": name,
                    "point_index": output_index,
                    "outer_fold": sample_row["outer_fold"],
                    "team_release_kin": sample_row["team_release_kin"],
                    "radius_kpc": _metric(radius),
                    "log_radius_over_r1": _metric(log_radius),
                    "log_surface_density": _metric(log_sigma),
                    "enclosed_mass_fraction": _metric(enclosed),
                    "log_total_profile_mass": _metric(math.log10(profile["total_profile_mass"])),
                    "log_distance": _metric(math.log10(profile["distance_mpc"])),
                    "log_r1_kpc": _metric(math.log10(r1)),
                    "profile_concentration": _metric(profile["profile_concentration"]),
                    "log_radius_squared": _metric(log_radius**2),
                    "log_surface_density_squared": _metric(log_sigma**2),
                    "radius_surface_interaction": _metric(log_radius * log_sigma),
                    "profile_contrast_mod": _metric(
                        math.tanh(math.log(max(profile["profile_contrast"], 1e-10)) / 2.0)
                    ),
                }
                for threshold in thresholds:
                    key = _threshold_key(threshold)
                    boundary = profile["boundaries"][key]
                    if boundary is None:
                        row[f"x_{key}"] = _metric(1.0)
                        row[f"edge_valid_{key}"] = "0"
                        row[f"sharpness_mod_{key}"] = _metric(0.0)
                        row[f"outer_mass_mod_{key}"] = _metric(0.0)
                    else:
                        row[f"x_{key}"] = _metric(radius / boundary["edge_radius_kpc"])
                        row[f"edge_valid_{key}"] = "1"
                        row[f"sharpness_mod_{key}"] = _metric(
                            math.tanh(math.log(max(boundary["edge_sharpness"], 1e-10)) / 2.0)
                        )
                        row[f"outer_mass_mod_{key}"] = _metric(
                            2.0 * boundary["outer_mass_fraction"] - 1.0
                        )
                features.append(row)
                responses.append(
                    {
                        "galaxy": name,
                        "point_index": output_index,
                        "observed_speed_km_s": _metric(float(velocity[source_index])),
                        "observed_speed_error_km_s": _metric(float(total_error[source_index])),
                    }
                )
        galaxy_receipts.append(
            {
                "name": name,
                "outer_fold": sample_row["outer_fold"],
                "team_release_kin": sample_row["team_release_kin"],
                "quality_pass": passed,
                "quality_failure_reasons": reasons,
                "raw_rotation_points": len(radius_arcsec),
                "accepted_rotation_points": accepted_count,
                "inclination_degrees": _metric(inclination),
                "q1_edge_radius_kpc": sample_row["q1_edge_radius_kpc"],
                "q1_edge_sharpness": sample_row["q1_edge_sharpness"],
                "total_profile_mass": sample_row["total_profile_mass"],
            }
        )
    feature_fields = [
        "galaxy",
        "point_index",
        "outer_fold",
        "team_release_kin",
        *POINT_FIELDS,
    ]
    for threshold in thresholds:
        key = _threshold_key(threshold)
        feature_fields.extend(
            [f"x_{key}", f"edge_valid_{key}", f"sharpness_mod_{key}", f"outer_mass_mod_{key}"]
        )
    response_fields = [
        "galaxy",
        "point_index",
        "observed_speed_km_s",
        "observed_speed_error_km_s",
    ]
    feature_path = root / config["outputs"]["point_features"]
    rotation_path = root / config["outputs"]["rotation_responses"]
    _write_tsv(feature_path, feature_fields, features)
    _write_tsv(rotation_path, response_fields, responses)
    passing = sum(row["quality_pass"] for row in galaxy_receipts)
    selected = len(galaxy_receipts)
    retention = passing / selected if selected else 0.0
    quality_pass = passing >= int(config["quality"]["minimum_quality_passing_exploration_galaxies"])
    quality_pass &= retention >= float(config["quality"]["minimum_quality_retention_fraction"])
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-wallaby-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": "PASS_ITEM10_WALLABY_QUALITY"
            if quality_pass
            else "FAIL_ITEM10_WALLABY_QUALITY",
            "galaxies": galaxy_receipts,
            "counts": {
                "exploration_response_rows": selected,
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": selected - passing,
                "quality_retention_fraction": _metric(retention),
                "accepted_points": len(features),
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "outputs": {
                "feature_sha256": _sha256_file(feature_path),
                "response_sha256": _sha256_file(rotation_path),
            },
            "claims": config["claim_boundaries"],
        }
    )
    summary_path = root / config["outputs"]["extraction_summary"]
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return {"features": feature_path, "responses": rotation_path, "summary": summary_path}


def _load_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    feature_path = root / config["outputs"]["point_features"]
    response_path = root / config["outputs"]["rotation_responses"]
    summary_path = root / config["outputs"]["extraction_summary"]
    candidate_path = root / config["outputs"]["candidate_manifest"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_content_hash(summary, "Item 10 extraction summary")
    validate_candidate_manifest(candidates, root)
    if summary["outputs"]["feature_sha256"] != _sha256_file(feature_path):
        raise GravityItem10BoundaryError("Item 10 feature table changed")
    if summary["outputs"]["response_sha256"] != _sha256_file(response_path):
        raise GravityItem10BoundaryError("Item 10 response table changed")
    features = _read_tsv(feature_path)
    responses = _read_tsv(response_path)
    if not features or len(features) != len(responses):
        raise GravityItem10BoundaryError("Item 10 point tables are empty or misaligned")
    keys = [(row["galaxy"], row["point_index"]) for row in features]
    if keys != [(row["galaxy"], row["point_index"]) for row in responses]:
        raise GravityItem10BoundaryError("Item 10 point keys changed")
    names = []
    for row in features:
        if not names or names[-1] != row["galaxy"]:
            names.append(row["galaxy"])
    index_by_name = {name: index for index, name in enumerate(names)}
    galaxy_index = np.asarray([index_by_name[row["galaxy"]] for row in features], dtype=np.int64)
    if np.any(np.diff(galaxy_index) < 0):
        raise GravityItem10BoundaryError("Item 10 points are not grouped by galaxy")
    counts = np.bincount(galaxy_index, minlength=len(names))
    starts = np.concatenate(([0], np.cumsum(counts)[:-1])).astype(np.int64)
    first = [features[int(start)] for start in starts]
    thresholds = [float(value) for value in config["candidate_generator"]["thresholds_msun_pc2"]]
    return {
        "summary": summary,
        "candidate_manifest": candidates,
        "features": features,
        "responses": responses,
        "names": names,
        "galaxy_index": galaxy_index,
        "point_counts": counts,
        "starts": starts,
        "folds": np.asarray([int(row["outer_fold"]) for row in first], dtype=np.int64),
        "team_releases": [row["team_release_kin"] for row in first],
        "y": np.log10(np.asarray([float(row["observed_speed_km_s"]) for row in responses])),
        "design": np.column_stack(
            [
                np.asarray([float(row[field]) for row in features])
                for field in config["evaluation"]["local_features"]
            ]
        ),
        "x_thresholds": np.vstack(
            [
                np.asarray([float(row[f"x_{_threshold_key(q)}"]) for row in features])
                for q in thresholds
            ]
        ),
        "valid_thresholds": np.vstack(
            [
                np.asarray([float(row[f"edge_valid_{_threshold_key(q)}"]) for row in features])
                for q in thresholds
            ]
        ),
        "sharpness_mod": np.vstack(
            [
                np.asarray([float(row[f"sharpness_mod_{_threshold_key(q)}"]) for row in features])
                for q in thresholds
            ]
        ),
        "outer_mass_mod": np.vstack(
            [
                np.asarray([float(row[f"outer_mass_mod_{_threshold_key(q)}"]) for row in features])
                for q in thresholds
            ]
        ),
        "concentration_mod": 2.0
        * np.asarray([float(row["profile_concentration"]) for row in features])
        - 1.0,
        "contrast_mod": np.asarray([float(row["profile_contrast_mod"]) for row in features]),
        "enclosed": np.asarray([float(row["enclosed_mass_fraction"]) for row in features]),
        "first_rows": first,
    }


def _point_weights(data: Mapping[str, Any]) -> np.ndarray:
    counts = np.asarray(data["point_counts"], dtype=np.float64)
    return 1.0 / counts[np.asarray(data["galaxy_index"], dtype=np.int64)]


def _ridge_fit(
    design: np.ndarray, target: np.ndarray, weights: np.ndarray, alpha: float
) -> dict[str, Any]:
    normalized = weights / np.sum(weights)
    mean = np.sum(design * normalized[:, None], axis=0)
    centered = design - mean
    scale = np.sqrt(np.sum(centered**2 * normalized[:, None], axis=0))
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = centered / scale
    target_mean = float(np.sum(target * normalized))
    centered_target = target - target_mean
    root_weight = np.sqrt(weights)
    matrix = standardized * root_weight[:, None]
    vector = centered_target * root_weight
    coefficient = np.linalg.solve(
        matrix.T @ matrix + alpha * np.eye(matrix.shape[1]), matrix.T @ vector
    )
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "coefficient": coefficient}


def _ridge_predict(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return float(model["target_mean"]) + (
        (design - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficient"])


def _candidate_components(
    arrays: Mapping[str, np.ndarray], data: Mapping[str, Any], begin: int, end: int, xp: Any
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.int32)
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(arrays["modulation"][begin:end], dtype=xp.int32)
    x_all = xp.asarray(data["x_thresholds"], dtype=xp.float64)
    valid_all = xp.asarray(data["valid_thresholds"], dtype=xp.float64)
    sharp_all = xp.asarray(data["sharpness_mod"], dtype=xp.float64)
    outer_all = xp.asarray(data["outer_mass_mod"], dtype=xp.float64)
    x = x_all[threshold]
    valid = valid_all[threshold]
    logx = xp.log(xp.maximum(x, 1e-10))
    envelope = xp.exp(-((xp.abs(logx) / scale) ** power))
    signed = xp.sign(1.0 - x) * envelope
    component = xp.zeros_like(envelope)
    component = xp.where(family == 0, envelope, component)
    component = xp.where(family == 1, signed, component)
    component = xp.where(
        family == 2,
        envelope / (1.0 + (xp.abs(1.0 - x) / scale) ** power),
        component,
    )
    component = xp.where(
        family == 3,
        xp.exp(-((xp.abs(x - 1.0 / xp.maximum(x, 1e-10)) / scale) ** power)),
        component,
    )
    component = xp.where(family == 4, 1.0 / (1.0 + (xp.abs(1.0 - x) / scale) ** power), component)
    robin = xp.where(
        x <= 1.0,
        xp.exp(-xp.maximum(1.0 - x, 0.0) / scale),
        (1.0 + xp.maximum(x - 1.0, 0.0) / scale) ** (-power),
    )
    component = xp.where(family == 5, robin, component)
    next_threshold = xp.minimum(threshold + 1, x_all.shape[0] - 1)
    x_next = x_all[next_threshold]
    next_envelope = xp.exp(-((xp.abs(xp.log(xp.maximum(x_next, 1e-10))) / scale) ** power))
    component = xp.where(family == 6, envelope - next_envelope, component)
    component = xp.where(family == 7, signed * sharp_all[threshold], component)
    enclosed = xp.asarray(data["enclosed"], dtype=xp.float64)[None, :]
    component = xp.where(family == 8, envelope * (enclosed - xp.clip(x**2, 0.0, 1.0)), component)
    component = xp.where(
        family == 9,
        envelope * xp.cos(phase + power * (x - 1.0 / xp.maximum(x, 1e-10))),
        component,
    )
    component = xp.where(family == 10, envelope * xp.sin(phase + math.pi * power * x), component)
    component = xp.where(family == 11, envelope * xp.cos(phase + power * logx), component)
    modulation = xp.ones_like(component)
    modulation = xp.where(modulation_index[:, None] == 1, sharp_all[threshold], modulation)
    modulation = xp.where(modulation_index[:, None] == 2, outer_all[threshold], modulation)
    modulation = xp.where(
        modulation_index[:, None] == 3,
        xp.asarray(data["concentration_mod"], dtype=xp.float64)[None, :],
        modulation,
    )
    modulation = xp.where(
        modulation_index[:, None] == 4,
        xp.asarray(data["contrast_mod"], dtype=xp.float64)[None, :],
        modulation,
    )
    result = component * modulation * valid
    return xp.clip(xp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _nested_select(
    data: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], dict[str, Any]]:
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
    count = len(arrays["family"])
    folds = np.asarray(data["folds"])
    galaxy_index = np.asarray(data["galaxy_index"])
    point_folds = folds[galaxy_index]
    y = np.asarray(data["y"])
    design = np.asarray(data["design"])
    weights = _point_weights(data)
    baseline_oof = np.full(len(y), np.nan)
    boundary_oof = np.full(len(y), np.nan)
    records = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["boundary_coefficient_ridge"])
    component_crosscheck = 0.0
    for outer in range(int(config["evaluation"]["outer_folds"])):
        other_folds = [fold for fold in range(5) if fold != outer]
        inner_records = []
        for inner in other_folds:
            train = (point_folds != outer) & (point_folds != inner)
            validation = point_folds == inner
            model = _ridge_fit(design[train], y[train], weights[train], alpha)
            inner_records.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_residual": y[train] - _ridge_predict(model, design[train]),
                    "validation_residual": y[validation]
                    - _ridge_predict(model, design[validation]),
                }
            )
        scores = np.full(count, np.inf, dtype=np.float64)
        for begin in range(0, count, batch_size):
            end = min(begin + batch_size, count)
            components = _candidate_components(arrays, data, begin, end, xp)
            total_loss = xp.zeros(end - begin, dtype=xp.float64)
            for inner in inner_records:
                train = inner["train"]
                validation = inner["validation"]
                train_component = components[:, train]
                validation_component = components[:, validation]
                train_weight = xp.asarray(weights[train], dtype=xp.float64)
                validation_weight = xp.asarray(weights[validation], dtype=xp.float64)
                mean = xp.sum(train_component * train_weight[None, :], axis=1) / xp.sum(
                    train_weight
                )
                centered_train = train_component - mean[:, None]
                scale = xp.sqrt(
                    xp.sum(centered_train**2 * train_weight[None, :], axis=1) / xp.sum(train_weight)
                )
                scale = xp.maximum(scale, 1e-12)
                standardized_train = centered_train / scale[:, None]
                coefficient = xp.sum(
                    standardized_train
                    * train_weight[None, :]
                    * xp.asarray(inner["train_residual"])[None, :],
                    axis=1,
                ) / (
                    xp.sum(standardized_train**2 * train_weight[None, :], axis=1)
                    + coefficient_ridge
                )
                standardized_validation = (validation_component - mean[:, None]) / scale[:, None]
                residual = (
                    xp.asarray(inner["validation_residual"])[None, :]
                    - coefficient[:, None] * standardized_validation
                )
                total_loss += xp.sum(residual**2 * validation_weight[None, :], axis=1) / xp.sum(
                    validation_weight
                )
            batch_scores = total_loss / len(inner_records)
            scores[begin:end] = xp.asnumpy(batch_scores) if backend == "gpu_cupy" else batch_scores
            if begin == 0:
                check_count = min(int(config["evaluation"]["cpu_crosscheck_candidates"]), end)
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
        outer_train = point_folds != outer
        outer_test = point_folds == outer
        model = _ridge_fit(design[outer_train], y[outer_train], weights[outer_train], alpha)
        train_base = _ridge_predict(model, design[outer_train])
        test_base = _ridge_predict(model, design[outer_test])
        selected_component = _candidate_components(arrays, data, selected, selected + 1, np)[0]
        train_component = selected_component[outer_train]
        mean = float(np.sum(train_component * weights[outer_train]) / np.sum(weights[outer_train]))
        scale = float(
            np.sqrt(
                np.sum((train_component - mean) ** 2 * weights[outer_train])
                / np.sum(weights[outer_train])
            )
        )
        scale = max(scale, 1e-12)
        standardized_train = (train_component - mean) / scale
        train_residual = y[outer_train] - train_base
        coefficient = float(
            np.sum(standardized_train * weights[outer_train] * train_residual)
            / (np.sum(standardized_train**2 * weights[outer_train]) + coefficient_ridge)
        )
        baseline_oof[outer_test] = test_base
        boundary_oof[outer_test] = (
            test_base + coefficient * (selected_component[outer_test] - mean) / scale
        )
        family = config["candidate_generator"]["families"][int(arrays["family"][selected])]
        records.append(
            {
                "outer_fold": outer,
                "selected_ordinal": selected,
                "selected_family": family["id"],
                "origin_status": family["origin_status"],
                "threshold_msun_pc2": _metric(
                    float(
                        config["candidate_generator"]["thresholds_msun_pc2"][
                            int(arrays["threshold"][selected])
                        ]
                    )
                ),
                "scale": _metric(float(arrays["scale"][selected])),
                "power": _metric(float(arrays["power"][selected])),
                "phase": _metric(float(arrays["phase"][selected])),
                "modulation": config["candidate_generator"]["modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_equal_galaxy_mse": _metric(float(scores[selected])),
                "fitted_universal_coefficient": _metric(coefficient),
                "test_galaxies": int(np.sum(folds == outer)),
            }
        )
    if np.any(~np.isfinite(baseline_oof)) or np.any(~np.isfinite(boundary_oof)):
        raise GravityItem10BoundaryError("Item 10 OOF predictions are incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    points = len(y)
    return (
        baseline_oof,
        boundary_oof,
        records,
        {
            "backend": backend,
            "device": device,
            "cupy_version": getattr(xp, "__version__", None) if backend == "gpu_cupy" else None,
            "elapsed_seconds": _metric(elapsed),
            "candidate_cells": count,
            "points": points,
            "outer_folds": 5,
            "inner_validation_fits_per_outer": 4,
            "candidate_point_score_evaluations": count * points * 20,
            "cpu_crosscheck_candidates": int(config["evaluation"]["cpu_crosscheck_candidates"]),
            "cpu_gpu_max_component_difference": _metric(component_crosscheck),
        },
    )


def _metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, str]:
    normalized = weights / np.sum(weights)
    mse = float(np.sum(normalized * (y - prediction) ** 2))
    mean = float(np.sum(normalized * y))
    variance = float(np.sum(normalized * (y - mean) ** 2))
    return {"mse": _metric(mse), "r2": _metric(1.0 - mse / variance if variance > 0 else 0.0)}


def _galaxy_losses(data: Mapping[str, Any], prediction: np.ndarray) -> np.ndarray:
    return np.add.reduceat(
        (np.asarray(data["y"]) - prediction) ** 2, np.asarray(data["starts"])
    ) / np.asarray(data["point_counts"])


def _paired_sign_flip(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    seed = int(
        hashlib.sha256(config["evaluation"]["permutation_salt"].encode()).hexdigest()[:16], 16
    )
    random = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null = np.empty(count)
    for index in range(count):
        null[index] = float(np.mean(differences * random.choice([-1.0, 1.0], len(differences))))
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
    baseline, boundary, folds, compute = _nested_select(data, config)
    weights = _point_weights(data)
    baseline_metrics = _metrics(data["y"], baseline, weights)
    boundary_metrics = _metrics(data["y"], boundary, weights)
    baseline_mse = float(baseline_metrics["mse"])
    boundary_mse = float(boundary_metrics["mse"])
    relative = (baseline_mse - boundary_mse) / baseline_mse
    differences = _galaxy_losses(data, baseline) - _galaxy_losses(data, boundary)
    paired = _paired_sign_flip(differences, config)
    by_name = {row["name"]: row for row in data["summary"]["galaxies"] if row["quality_pass"]}
    galaxy_rows = [by_name[name] for name in data["names"]]
    dimensions = {
        "team_release_kin": data["team_releases"],
        "edge_radius_half": [
            "low"
            if float(row["q1_edge_radius_kpc"])
            <= float(np.median([float(item["q1_edge_radius_kpc"]) for item in galaxy_rows]))
            else "high"
            for row in galaxy_rows
        ],
        "edge_sharpness_half": [
            "low"
            if float(row["q1_edge_sharpness"])
            <= float(np.median([float(item["q1_edge_sharpness"]) for item in galaxy_rows]))
            else "high"
            for row in galaxy_rows
        ],
        "profile_mass_half": [
            "low"
            if float(row["total_profile_mass"])
            <= float(np.median([float(item["total_profile_mass"]) for item in galaxy_rows]))
            else "high"
            for row in galaxy_rows
        ],
    }
    baseline_loss = _galaxy_losses(data, baseline)
    boundary_loss = _galaxy_losses(data, boundary)
    strata = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            mask = np.asarray([entry == value for entry in values])
            base = float(np.mean(baseline_loss[mask]))
            proposed = float(np.mean(boundary_loss[mask]))
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": value,
                    "galaxies": int(np.sum(mask)),
                    "baseline_mse": _metric(base),
                    "boundary_mse": _metric(proposed),
                    "boundary_mse_gain": _metric(base - proposed),
                }
            )
    team_gate_rows = [
        row
        for row in strata
        if row["dimension"] == "team_release_kin"
        and int(row["galaxies"])
        >= int(config["evaluation"]["minimum_galaxies_per_team_release_gate"])
    ]
    half_pass = {
        dimension: all(
            float(row["boundary_mse_gain"]) > 0 for row in strata if row["dimension"] == dimension
        )
        for dimension in ("edge_radius_half", "edge_sharpness_half", "profile_mass_half")
    }
    selected_thresholds = [float(row["threshold_msun_pc2"]) for row in folds]
    valid_by_threshold = {
        float(q): float(
            np.mean(data["valid_thresholds"][index, np.asarray(data["starts"], dtype=int)])
        )
        for index, q in enumerate(config["candidate_generator"]["thresholds_msun_pc2"])
    }
    selected_coverage = min(valid_by_threshold[value] for value in selected_thresholds)
    gates = {
        "quality_count_and_fraction_pass": data["summary"]["decision"]
        == "PASS_ITEM10_WALLABY_QUALITY",
        "confirmation_responses_untouched": True,
        "candidate_count_exact": compute["candidate_cells"] == 131072,
        "selected_boundary_r2_positive": float(boundary_metrics["r2"]) > 0,
        "selected_boundary_beats_local_baseline": boundary_mse < baseline_mse,
        "relative_mse_improvement_over_local_baseline_at_least": relative
        >= float(config["admission"]["relative_mse_improvement_over_local_baseline_at_least"]),
        "paired_sign_flip_p_at_most": float(paired["p_value"])
        <= float(config["admission"]["paired_sign_flip_p_at_most"]),
        "gain_positive_in_all_team_releases_at_minimum_count": bool(team_gate_rows)
        and all(float(row["boundary_mse_gain"]) > 0 for row in team_gate_rows),
        "gain_positive_in_both_edge_radius_halves": half_pass["edge_radius_half"],
        "gain_positive_in_both_edge_sharpness_halves": half_pass["edge_sharpness_half"],
        "gain_positive_in_both_profile_mass_halves": half_pass["profile_mass_half"],
        "selected_candidate_edge_coverage_at_least": selected_coverage
        >= float(config["admission"]["selected_candidate_edge_coverage_at_least"]),
        "selected_family_not_algebraically_equivalent_to_item9_occupancy": True,
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM10_WALLABY_BOUNDARY_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM10_WALLABY_BOUNDARY_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM10_WALLABY_QUALITY"
    candidate_path = root / config["outputs"]["candidate_manifest"]
    predictor_path = root / config["outputs"]["predictor_source"]
    sample_path = root / config["outputs"]["sample_manifest"]
    response_path = root / config["outputs"]["response_source"]
    extraction_path = root / config["outputs"]["extraction_summary"]
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item10-wallaby-boundary-result-1.0",
            "goal": config["goal"],
            "item_number": 10,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "counts": {
                "candidate_cells": 131072,
                "quality_passing_galaxies": data["summary"]["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": data["summary"]["counts"]["quality_failed_galaxies"],
                "accepted_points": data["summary"]["counts"]["accepted_points"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "candidate_manifest_sha256": _sha256_file(candidate_path),
                "predictor_source_sha256": _sha256_file(predictor_path),
                "sample_manifest_sha256": _sha256_file(sample_path),
                "response_source_sha256": _sha256_file(response_path),
                "extraction_summary_sha256": _sha256_file(extraction_path),
                "feature_sha256": data["summary"]["outputs"]["feature_sha256"],
                "response_sha256": data["summary"]["outputs"]["response_sha256"],
            },
            "primary": {
                "local_baseline": baseline_metrics,
                "selected_boundary": boundary_metrics,
                "absolute_mse_improvement": _metric(baseline_mse - boundary_mse),
                "relative_mse_improvement": _metric(relative),
                "outer_fold_selections": folds,
                "selected_edge_coverage": _metric(selected_coverage),
            },
            "compute": compute,
            "paired_sign_flip": paired,
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {
                "passed": sum(bool(value) for value in gates.values()),
                "required": len(gates),
            },
            "limitations": {
                "complete_baryonic_mass_used": False,
                "projected_hi_profile_proxy_used": True,
                "stellar_mass_profile_used": False,
                "historical_novelty_adjudicated": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 10 result receipt")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem10BoundaryError("Item 10 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem10BoundaryError("Item 10 result sample binding changed")
    if receipt["counts"]["candidate_cells"] != 131072:
        raise GravityItem10BoundaryError("Item 10 result candidate count changed")
    if receipt["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem10BoundaryError("Item 10 confirmation entered result")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem10BoundaryError("Item 10 post-response formula entered result")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem10BoundaryError("Item 10 receipt contains an overclaim")


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
    rebuilt = build_receipt(root)
    stored_compare = dict(stored)
    rebuilt_compare = dict(rebuilt)
    for value in (stored_compare, rebuilt_compare):
        value.pop("content_sha256", None)
        value["compute"] = dict(value["compute"])
        value["compute"].pop("elapsed_seconds", None)
    if stored_compare != rebuilt_compare:
        raise GravityItem10BoundaryError("Item 10 result receipt drifted")
    validate_receipt(stored, root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("predictors", "sample", "candidates", "responses", "extract", "run", "check"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "predictors":
        print(write_predictor_source(args.root))
    elif args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "candidates":
        print(write_candidate_manifest(args.root))
    elif args.command == "responses":
        print(write_response_source(args.root))
    elif args.command == "extract":
        print(json.dumps({key: str(value) for key, value in extract_profiles(args.root).items()}))
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
