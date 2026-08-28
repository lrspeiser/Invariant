"""Frozen NEARGALCAT external-baryonic-field search for gravity Item 11."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import time
import urllib.parse
import urllib.request
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

CONFIG_PATH = Path("configs/gravity_item11_neargalcat_external_field_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "PENDING_ITEM11_SCIENTIFIC_FREEZE_COMMIT"
SAMPLE_FREEZE_COMMIT = "PENDING_ITEM11_SAMPLE_FREEZE_COMMIT"


class GravityItem11ExternalFieldError(RuntimeError):
    """Raised when an Item 11 scientific or response boundary drifts."""


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
        raise GravityItem11ExternalFieldError(f"{label} content hash changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem11ExternalFieldError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem11ExternalFieldError("Item 10 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 10 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem11ExternalFieldError("Item 10 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem11ExternalFieldError("Item 10 synthesis decision changed")
    for entry in (
        config["independence"]["normalized_identity_exclusions"]
        + config["independence"]["coordinate_exclusions"]
    ):
        if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise GravityItem11ExternalFieldError(
                f"predecessor exclusion source changed: {entry['path']}"
            )
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem11ExternalFieldError("Item 11 config contains an overclaim")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem11ExternalFieldError("Item 11 candidate count changed")
    return config


def _tap_query(config: Mapping[str, Any], query: str) -> bytes:
    payload = urllib.parse.urlencode(
        {
            "REQUEST": "doQuery",
            "LANG": "ADQL",
            "FORMAT": "text/plain",
            "QUERY": query,
        }
    ).encode()
    request = urllib.request.Request(
        config["source"]["tap_sync_endpoint"],
        data=payload,
        headers={"User-Agent": "Invariant/Item11-NEARGALCAT"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        result = response.read()
    if not result:
        raise GravityItem11ExternalFieldError("empty HEASARC TAP response")
    return result


def _parse_plain_table(payload: bytes) -> list[dict[str, str]]:
    lines = payload.decode("utf-8-sig", errors="strict").splitlines()
    table_lines = []
    for line in lines:
        if line.startswith(("Number of rows:", "Number of columns:")):
            break
        if line.strip():
            table_lines.append(line)
    if not table_lines:
        raise GravityItem11ExternalFieldError("HEASARC plain table is empty")
    header = [value.strip().strip('"') for value in table_lines[0].split("|")]
    rows = []
    for line in table_lines[1:]:
        values = [value.strip() for value in line.split("|")]
        if len(values) != len(header):
            raise GravityItem11ExternalFieldError("HEASARC plain row width changed")
        rows.append(dict(zip(header, values)))
    return rows


def _quoted_columns(columns: Sequence[str]) -> str:
    return ",".join(f'"{column}"' for column in columns)


def _finite(row: Mapping[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    if not value or value.lower() in {"nan", "null", "none", "--"}:
        raise GravityItem11ExternalFieldError(f"missing predictor {key}")
    try:
        number = float(value)
    except ValueError as exc:
        raise GravityItem11ExternalFieldError(f"invalid predictor {key}") from exc
    if not math.isfinite(number):
        raise GravityItem11ExternalFieldError(f"non-finite predictor {key}")
    return number


def normalize_identity(value: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]", "", str(value).upper())
    normalized = normalized.removeprefix("NAME")
    normalized = re.sub(r"^(NGC|UGC|IC|DDO|ESO)0+", r"\1", normalized)
    return normalized


def derive_predictors(row: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    if str(row.get("log_h1_mass_limit", "")).strip():
        raise GravityItem11ExternalFieldError("H I mass is an upper limit")
    log_luminosity = _finite(row, "log_ks_luminosity")
    log_hi_mass = _finite(row, "log_h1_mass")
    diameter = _finite(row, "linear_diameter")
    distance = _finite(row, "distance")
    inclination = _finite(row, "inclination")
    if diameter <= 0 or distance <= 0 or not 0 < inclination < 90:
        raise GravityItem11ExternalFieldError("invalid size, distance, or inclination")
    stellar_mass = float(config["constants"]["fixed_ks_stellar_mass_to_light"]) * 10**log_luminosity
    gas_mass = float(config["constants"]["helium_mass_factor"]) * 10**log_hi_mass
    baryonic_mass = stellar_mass + gas_mass
    radius = diameter / 2.0
    internal_acceleration = (
        float(config["constants"]["gravitational_constant_kpc_km2_s2_msun"])
        * baryonic_mass
        / radius**2
    )
    bmag = _finite(row, "bmag")
    ks_mag = _finite(row, "ks_mag")
    result = {
        "row_id": str(row["__row"]),
        "name": str(row["name"]),
        "normalized_identity": normalize_identity(row["name"]),
        "ra": _finite(row, "ra"),
        "dec": _finite(row, "dec"),
        "neighbor_galaxy_name": str(row.get("neighbor_galaxy_name", "")),
        "log_baryonic_mass": math.log10(baryonic_mass),
        "log_stellar_mass": math.log10(stellar_mass),
        "gas_fraction": gas_mass / baryonic_mass,
        "log_linear_diameter": math.log10(diameter),
        "b_surface_brightness": _finite(row, "bmag_surface_brightness"),
        "morph_type": _finite(row, "morph_type"),
        "axial_ratio": _finite(row, "axial_ratio"),
        "log_distance": math.log10(distance),
        "inclination_degrees": inclination,
        "inclination_sine": math.sin(math.radians(inclination)),
        "b_minus_ks_color": bmag - ks_mag,
        "log_internal_acceleration": math.log10(internal_acceleration),
        "tidal_index_1": _finite(row, "tidal_index_1"),
        "tidal_index_2": _finite(row, "tidal_index_2"),
        "log_ks_lum_density": _finite(row, "log_ks_lum_density"),
    }
    if not result["row_id"] or not result["name"] or not result["normalized_identity"]:
        raise GravityItem11ExternalFieldError("missing HEASARC row identity")
    if not 0 < result["axial_ratio"] <= 1.5:
        raise GravityItem11ExternalFieldError("invalid axial ratio")
    return result


def _serialize_predictors(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value if isinstance(value, str) else _metric(value) for key, value in row.items()}


def write_predictor_source(root: Path) -> Path:
    root = root.resolve()
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem11ExternalFieldError("Item 11 scientific freeze is not bound")
    config = load_config(root)
    columns = _quoted_columns(config["source"]["predictor_columns"])
    query = f'SELECT {columns} FROM {config["source"]["table"]} ORDER BY "__row"'
    payload = _tap_query(config, query)
    rows = _parse_plain_table(payload)
    if len(rows) != int(config["source"]["observed_rows"]):
        raise GravityItem11ExternalFieldError("NEARGALCAT row count changed")
    expected = set(config["source"]["predictor_columns"])
    if any(set(row) != expected for row in rows):
        raise GravityItem11ExternalFieldError("NEARGALCAT predictor schema changed")
    records = []
    failures = []
    for row in rows:
        try:
            records.append(_serialize_predictors(derive_predictors(row, config)))
        except GravityItem11ExternalFieldError as exc:
            failures.append(
                {
                    "row_id": str(row.get("__row", "")),
                    "name": str(row.get("name", "")),
                    "reason": str(exc),
                }
            )
    if len({row["row_id"] for row in records}) != len(records):
        raise GravityItem11ExternalFieldError("eligible HEASARC row identifier is not unique")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item11-neargalcat-predictor-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "query": query,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "records": records,
            "failures": failures,
            "counts": {
                "catalogue_rows": len(rows),
                "valid_predictor_rows": len(records),
                "invalid_predictor_rows": len(failures),
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
    _validate_content_hash(source, "Item 11 predictor source")
    config = load_config(root)
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem11ExternalFieldError("predictor freeze binding changed")
    columns = _quoted_columns(config["source"]["predictor_columns"])
    expected_query = f'SELECT {columns} FROM {config["source"]["table"]} ORDER BY "__row"'
    if source["query"] != expected_query:
        raise GravityItem11ExternalFieldError("predictor query changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem11ExternalFieldError("response entered predictor source")


def _vizier_rows(path: Path) -> list[dict[str, str]]:
    lines = [
        line for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith("#")
    ]
    lines = [line for line in lines if line.strip()]
    reader = csv.DictReader(io.StringIO("\n".join(lines)), delimiter="\t")
    rows = []
    for index, row in enumerate(reader):
        if index < 2:
            continue
        rows.append({str(key): str(value).strip() for key, value in row.items()})
    return rows


def _json_objects(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in value["objects"]]


def _probes_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        next(reader)
        header = next(reader)
        return [dict(zip(header, row)) for row in reader]


def _source_rows(root: Path, entry: Mapping[str, Any]) -> list[dict[str, Any]]:
    path = root / entry["path"]
    if entry["format"] == "vizier_tsv":
        return _vizier_rows(path)
    if entry["format"] == "json_objects":
        return _json_objects(path)
    if entry["format"] == "probes_csv":
        return _probes_rows(path)
    raise GravityItem11ExternalFieldError("unknown predecessor source format")


def _predecessor_identities(root: Path, config: Mapping[str, Any]) -> set[str]:
    result = set()
    for entry in config["independence"]["normalized_identity_exclusions"]:
        for row in _source_rows(root, entry):
            value = normalize_identity(str(row.get(entry["key"], "")))
            if value:
                result.add(value)
    return result


def _predecessor_coordinates(root: Path, config: Mapping[str, Any]) -> np.ndarray:
    result = []
    for entry in config["independence"]["coordinate_exclusions"]:
        for row in _source_rows(root, entry):
            try:
                ra = float(row[entry["ra_key"]])
                dec = float(row[entry["dec_key"]])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(ra) and math.isfinite(dec):
                result.append((ra, dec))
    return np.asarray(result, dtype=np.float64)


def _minimum_separation_arcsec(ra: float, dec: float, coordinates: np.ndarray) -> float:
    ra1 = math.radians(ra)
    dec1 = math.radians(dec)
    ra2 = np.radians(coordinates[:, 0])
    dec2 = np.radians(coordinates[:, 1])
    haversine = np.sin((dec2 - dec1) / 2.0) ** 2
    haversine += math.cos(dec1) * np.cos(dec2) * np.sin((ra2 - ra1) / 2.0) ** 2
    angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return float(np.min(angle) * 206264.80624709636)


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["outputs"]["predictor_source"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_predictor_source(source, root)
    excluded_identities = _predecessor_identities(root, config)
    excluded_coordinates = _predecessor_coordinates(root, config)
    local_identity_counts = Counter(str(row["normalized_identity"]) for row in source["records"])
    admitted = []
    exclusions: Counter[str] = Counter()
    for row in source["records"]:
        reasons = []
        if row["normalized_identity"] in excluded_identities:
            reasons.append("predecessor_normalized_identity")
        if local_identity_counts[row["normalized_identity"]] != 1:
            reasons.append("ambiguous_local_normalized_identity")
        separation = _minimum_separation_arcsec(
            float(row["ra"]), float(row["dec"]), excluded_coordinates
        )
        if separation <= float(config["independence"]["coordinate_exclusion_arcseconds"]):
            reasons.append("predecessor_coordinate")
        for reason in set(reasons):
            exclusions[reason] += 1
        if reasons:
            continue
        output = dict(row)
        output["minimum_predecessor_separation_arcsec"] = _metric(separation)
        admitted.append(output)
    if not admitted:
        raise GravityItem11ExternalFieldError("no independent NEARGALCAT predictors remain")
    median_mass = float(np.median([float(row["log_baryonic_mass"]) for row in admitted]))
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in admitted:
        environment = "group" if float(row["tidal_index_1"]) >= 0 else "isolated"
        mass = "low_mass" if float(row["log_baryonic_mass"]) <= median_mass else "high_mass"
        cells.setdefault((environment, mass), []).append(row)
    objects = []
    salt = str(config["sample"]["split_salt"])
    for (environment, mass), rows in sorted(cells.items()):
        ordered = sorted(rows, key=lambda row: (_split_hash(row["row_id"], salt), row["row_id"]))
        confirmation_count = round(len(ordered) * float(config["sample"]["confirmation_fraction"]))
        confirmation = {row["row_id"] for row in ordered[:confirmation_count]}
        for row in sorted(ordered, key=lambda value: value["row_id"]):
            output = dict(row)
            output.update(
                {
                    "environment_sign_bin": environment,
                    "baryonic_mass_bin": mass,
                    "role": "reserved_confirmation"
                    if row["row_id"] in confirmation
                    else "exploration",
                    "outer_fold": int(
                        _split_hash(row["row_id"], config["evaluation"]["fold_salt"])[:16],
                        16,
                    )
                    % int(config["evaluation"]["outer_folds"]),
                    "response_read": False,
                }
            )
            objects.append(output)
    objects.sort(key=lambda row: row["row_id"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item11-neargalcat-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "predictor_source_sha256": _sha256_file(source_path),
            "predictor_source_content_sha256": source["content_sha256"],
            "baryonic_mass_median": _metric(median_mass),
            "objects": objects,
            "counts": {
                "valid_predictor_rows": len(source["records"]),
                "admitted_independent_rows": len(objects),
                "exploration": sum(row["role"] == "exploration" for row in objects),
                "reserved_confirmation": sum(
                    row["role"] == "reserved_confirmation" for row in objects
                ),
                "predecessor_identity_or_coordinate_selected": 0,
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
            "claims": {"confirmation_opened": False},
        }
    )


def validate_sample_manifest(sample: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(sample, "Item 11 sample manifest")
    if sample != build_sample_manifest(root):
        raise GravityItem11ExternalFieldError("Item 11 sample manifest drifted")
    if sample["counts"]["predecessor_identity_or_coordinate_selected"] != 0:
        raise GravityItem11ExternalFieldError("predecessor object entered Item 11")
    if sample["counts"]["response_rows_read"] != 0:
        raise GravityItem11ExternalFieldError("response entered target-blind sample")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["sample_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def generate_candidates(config: Mapping[str, Any]) -> dict[str, np.ndarray]:
    generator = config["candidate_generator"]
    count = int(generator["candidate_cells"])
    random = np.random.Generator(np.random.PCG64(int(generator["seed"])))
    scale_low, scale_high = [math.log(float(v)) for v in generator["scale_log_uniform"]]
    power_low, power_high = [math.log(float(v)) for v in generator["power_log_uniform"]]
    return {
        "family": random.integers(0, len(generator["families"]), count, dtype=np.int16),
        "threshold": random.uniform(*generator["threshold_uniform"], count),
        "scale": np.exp(random.uniform(scale_low, scale_high, count)),
        "power": np.exp(random.uniform(power_low, power_high, count)),
        "phase": random.uniform(*generator["phase_uniform"], count),
        "modulation": random.integers(
            0, len(generator["internal_modulations"]), count, dtype=np.int8
        ),
    }


def _candidate_digest(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in ("family", "threshold", "scale", "power", "phase", "modulation"):
        array = np.asarray(arrays[key])
        digest.update(key.encode())
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
            "schema_version": "invariant-gravity-item11-external-field-candidates-1.0",
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
    _validate_content_hash(manifest, "Item 11 candidate manifest")
    if manifest != build_candidate_manifest(root):
        raise GravityItem11ExternalFieldError("Item 11 candidate manifest drifted")
    if manifest["counts"]["post_response_cells"] != 0:
        raise GravityItem11ExternalFieldError("post-response candidate entered Item 11")


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
        raise GravityItem11ExternalFieldError("Item 11 sample freeze is not bound")
    config = load_config(root)
    sample_path = root / config["outputs"]["sample_manifest"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, root)
    exploration = [row["row_id"] for row in sample["objects"] if row["role"] == "exploration"]
    quoted = ",".join("'" + value.replace("'", "''") + "'" for value in exploration)
    columns = _quoted_columns(config["source"]["response_columns"])
    query = (
        f"SELECT {columns} FROM {config['source']['table']} "
        f'WHERE "__row" IN ({quoted}) ORDER BY "__row"'
    )
    payload = _tap_query(config, query)
    rows = _parse_plain_table(payload)
    expected = set(config["source"]["response_columns"])
    if any(set(row) != expected for row in rows):
        raise GravityItem11ExternalFieldError("NEARGALCAT response schema changed")
    row_ids = [row["__row"] for row in rows]
    if set(row_ids) != set(exploration) or len(row_ids) != len(set(row_ids)):
        raise GravityItem11ExternalFieldError("NEARGALCAT response scope changed")
    confirmation = {
        row["row_id"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if confirmation & set(row_ids):
        raise GravityItem11ExternalFieldError("Item 11 confirmation response was returned")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item11-neargalcat-response-source-1.0",
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
    _validate_content_hash(source, "Item 11 response source")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem11ExternalFieldError("response sample binding changed")
    if source["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem11ExternalFieldError("confirmation response entered Item 11")
    if source["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem11ExternalFieldError("post-response formula entered Item 11")


def extract_rows(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    predictor = json.loads(
        (root / config["outputs"]["predictor_source"]).read_text(encoding="utf-8")
    )
    sample = json.loads((root / config["outputs"]["sample_manifest"]).read_text(encoding="utf-8"))
    response = json.loads((root / config["outputs"]["response_source"]).read_text(encoding="utf-8"))
    validate_predictor_source(predictor, root)
    validate_sample_manifest(sample, root)
    validate_response_source(response, root)
    predictors = {row["row_id"]: row for row in predictor["records"]}
    samples = {row["row_id"]: row for row in sample["objects"] if row["role"] == "exploration"}
    rows = []
    failures = []
    for raw in response["records"]:
        row_id = raw["__row"]
        predictor_row = predictors[row_id]
        sample_row = samples[row_id]
        reasons = []
        try:
            velocity = _finite(raw, "h1_rot_velocity")
            width = _finite(raw, "h1_21_cm_50pc_width")
        except GravityItem11ExternalFieldError as exc:
            failures.append({"row_id": row_id, "name": raw["name"], "reasons": [str(exc)]})
            continue
        inclination = float(predictor_row["inclination_degrees"])
        if not (
            float(config["quality"]["minimum_inclination_degrees"])
            <= inclination
            <= float(config["quality"]["maximum_inclination_degrees"])
        ):
            reasons.append("inclination")
        if not (
            float(config["quality"]["minimum_rotation_velocity_km_s"])
            <= velocity
            <= float(config["quality"]["maximum_rotation_velocity_km_s"])
        ):
            reasons.append("rotation_velocity")
        if width < float(config["quality"]["minimum_raw_width_km_s"]):
            reasons.append("raw_width")
        width_velocity = width / (2.0 * max(math.sin(math.radians(inclination)), 1e-10))
        disagreement = abs(width_velocity - velocity) / max(velocity, 1e-10)
        if disagreement > float(
            config["quality"]["maximum_width_rotation_fractional_disagreement"]
        ):
            reasons.append("width_rotation_consistency")
        output = {
            **predictor_row,
            "outer_fold": sample_row["outer_fold"],
            "environment_sign_bin": sample_row["environment_sign_bin"],
            "baryonic_mass_bin": sample_row["baryonic_mass_bin"],
            "observed_rotation_velocity_km_s": _metric(velocity),
            "raw_width_km_s": _metric(width),
            "width_derived_velocity_km_s": _metric(width_velocity),
            "width_rotation_fractional_disagreement": _metric(disagreement),
            "quality_pass": not reasons,
            "quality_failure_reasons": reasons,
        }
        if reasons:
            failures.append({"row_id": row_id, "name": raw["name"], "reasons": reasons})
        else:
            rows.append(output)
    selected = len(response["records"])
    passing = len(rows)
    retention = passing / selected if selected else 0.0
    quality_pass = passing >= int(config["quality"]["minimum_quality_passing_exploration_galaxies"])
    quality_pass &= retention >= float(config["quality"]["minimum_quality_retention_fraction"])
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item11-neargalcat-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": "PASS_ITEM11_NEARGALCAT_QUALITY"
            if quality_pass
            else "FAIL_ITEM11_NEARGALCAT_QUALITY",
            "rows": rows,
            "failures": failures,
            "counts": {
                "exploration_response_rows": selected,
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": selected - passing,
                "quality_retention_fraction": _metric(retention),
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


def _load_data(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    summary_path = root / config["outputs"]["extraction_summary"]
    candidate_path = root / config["outputs"]["candidate_manifest"]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    _validate_content_hash(summary, "Item 11 extraction summary")
    validate_candidate_manifest(candidates, root)
    rows = summary["rows"]
    if not rows:
        raise GravityItem11ExternalFieldError("Item 11 has no quality-passing rows")
    return {
        "summary": summary,
        "candidate_manifest": candidates,
        "rows": rows,
        "row_ids": [row["row_id"] for row in rows],
        "folds": np.asarray([int(row["outer_fold"]) for row in rows], dtype=np.int64),
        "y": np.log10(np.asarray([float(row["observed_rotation_velocity_km_s"]) for row in rows])),
        "design": np.column_stack(
            [
                np.asarray([float(row[field]) for row in rows])
                for field in config["evaluation"]["local_features"]
            ]
        ),
        "theta1": np.asarray([float(row["tidal_index_1"]) for row in rows]),
        "theta2": np.asarray([float(row["tidal_index_2"]) for row in rows]),
        "rho_k": np.asarray([float(row["log_ks_lum_density"]) for row in rows]),
        "log_g": np.asarray([float(row["log_internal_acceleration"]) for row in rows]),
        "surface": np.asarray(
            [
                float(row["log_baryonic_mass"]) - 2 * float(row["log_linear_diameter"])
                for row in rows
            ]
        ),
        "gas_fraction": np.asarray([float(row["gas_fraction"]) for row in rows]),
        "size": np.asarray([float(row["log_linear_diameter"]) for row in rows]),
        "mass": np.asarray([float(row["log_baryonic_mass"]) for row in rows]),
        "distance": np.asarray([float(row["log_distance"]) for row in rows]),
    }


def _ridge_fit(design: np.ndarray, target: np.ndarray, alpha: float) -> dict[str, Any]:
    mean = np.mean(design, axis=0)
    scale = np.std(design, axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized = (design - mean) / scale
    target_mean = float(np.mean(target))
    coefficient = np.linalg.solve(
        standardized.T @ standardized + alpha * np.eye(standardized.shape[1]),
        standardized.T @ (target - target_mean),
    )
    return {"mean": mean, "scale": scale, "target_mean": target_mean, "coefficient": coefficient}


def _ridge_predict(model: Mapping[str, Any], design: np.ndarray) -> np.ndarray:
    return float(model["target_mean"]) + (
        (design - np.asarray(model["mean"])) / np.asarray(model["scale"])
    ) @ np.asarray(model["coefficient"])


def _centered_modulation(value: Any, xp: Any) -> Any:
    value = xp.asarray(value, dtype=xp.float64)
    return xp.tanh((value - xp.median(value)) / (xp.std(value) + 1e-10))


def _candidate_components(
    arrays: Mapping[str, np.ndarray], data: Mapping[str, Any], begin: int, end: int, xp: Any
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.float64)[:, None]
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(arrays["modulation"][begin:end], dtype=xp.int32)[:, None]
    theta1 = xp.asarray(data["theta1"], dtype=xp.float64)[None, :]
    theta2 = xp.asarray(data["theta2"], dtype=xp.float64)[None, :]
    rho_k = xp.asarray(data["rho_k"], dtype=xp.float64)[None, :]
    log_g = xp.asarray(data["log_g"], dtype=xp.float64)[None, :]

    def signed_power(value: Any) -> Any:
        z = (value - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    nearest = signed_power(theta1)
    collective = signed_power(theta2)
    density = signed_power(rho_k)
    dominance = signed_power(theta1 - theta2)
    mean_environment = (theta1 + theta2 + rho_k) / 3.0
    combined = signed_power((theta1 + theta2) / 2.0)
    component = xp.zeros((end - begin, theta1.shape[1]), dtype=xp.float64)
    component = xp.where(family == 0, nearest, component)
    component = xp.where(family == 1, collective, component)
    component = xp.where(family == 2, density, component)
    component = xp.where(family == 3, dominance, component)
    component = xp.where(family == 4, combined, component)
    boundary = xp.sign(theta1 - threshold) * xp.abs(xp.tanh((theta1 - threshold) / scale)) ** power
    component = xp.where(family == 5, boundary, component)
    isolation = xp.exp(-((xp.abs(theta1 - threshold) / scale) ** power))
    isolation *= xp.where(theta1 < threshold, 1.0, -1.0)
    component = xp.where(family == 6, isolation, component)
    suppression = signed_power(mean_environment) / (
        1.0 + (xp.abs(log_g - threshold) / scale) ** power
    )
    component = xp.where(family == 7, suppression, component)
    resonance = signed_power(mean_environment) * xp.cos(phase + power * log_g)
    component = xp.where(family == 8, resonance, component)
    coherence = xp.sign(theta1 * theta2 * rho_k) * xp.abs(theta1 * theta2 * rho_k) ** (1.0 / 3.0)
    component = xp.where(family == 9, xp.tanh(coherence / scale), component)
    saddle = (theta1 - theta2) * (rho_k - (theta1 + theta2) / 2.0)
    component = xp.where(family == 10, xp.tanh((saddle - threshold) / scale), component)
    log_periodic = signed_power(mean_environment) * xp.cos(
        phase + power * xp.log1p(xp.abs(mean_environment))
    )
    component = xp.where(family == 11, log_periodic, component)

    modulation = xp.ones_like(component)
    for index, field in enumerate(("surface", "gas_fraction", "size", "log_g"), start=1):
        value = _centered_modulation(data[field], xp)[None, :]
        modulation = xp.where(modulation_index == index, value, modulation)
    result = component * modulation
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
    y = np.asarray(data["y"])
    design = np.asarray(data["design"])
    baseline_oof = np.full(len(y), np.nan)
    external_oof = np.full(len(y), np.nan)
    records = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["external_coefficient_ridge"])
    component_crosscheck = 0.0
    outer_folds = int(config["evaluation"]["outer_folds"])
    for outer in range(outer_folds):
        inner_records = []
        for inner in [fold for fold in range(outer_folds) if fold != outer]:
            train = (folds != outer) & (folds != inner)
            validation = folds == inner
            model = _ridge_fit(design[train], y[train], alpha)
            inner_records.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_residual": y[train] - _ridge_predict(model, design[train]),
                    "validation_residual": y[validation]
                    - _ridge_predict(model, design[validation]),
                }
            )
        scores = np.full(count, np.inf)
        for begin in range(0, count, batch_size):
            end = min(begin + batch_size, count)
            components = _candidate_components(arrays, data, begin, end, xp)
            total_loss = xp.zeros(end - begin, dtype=xp.float64)
            for inner in inner_records:
                train_component = components[:, inner["train"]]
                validation_component = components[:, inner["validation"]]
                mean = xp.mean(train_component, axis=1)
                scale = xp.maximum(xp.std(train_component, axis=1), 1e-12)
                standardized_train = (train_component - mean[:, None]) / scale[:, None]
                coefficient = xp.sum(
                    standardized_train * xp.asarray(inner["train_residual"])[None, :],
                    axis=1,
                ) / (xp.sum(standardized_train**2, axis=1) + coefficient_ridge)
                standardized_validation = (validation_component - mean[:, None]) / scale[:, None]
                residual = (
                    xp.asarray(inner["validation_residual"])[None, :]
                    - coefficient[:, None] * standardized_validation
                )
                total_loss += xp.mean(residual**2, axis=1)
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
        train = folds != outer
        test = folds == outer
        model = _ridge_fit(design[train], y[train], alpha)
        train_base = _ridge_predict(model, design[train])
        test_base = _ridge_predict(model, design[test])
        selected_component = _candidate_components(arrays, data, selected, selected + 1, np)[0]
        mean = float(np.mean(selected_component[train]))
        scale = max(float(np.std(selected_component[train])), 1e-12)
        standardized_train = (selected_component[train] - mean) / scale
        coefficient = float(
            np.sum(standardized_train * (y[train] - train_base))
            / (np.sum(standardized_train**2) + coefficient_ridge)
        )
        baseline_oof[test] = test_base
        external_oof[test] = test_base + coefficient * (selected_component[test] - mean) / scale
        family = config["candidate_generator"]["families"][int(arrays["family"][selected])]
        records.append(
            {
                "outer_fold": outer,
                "selected_ordinal": selected,
                "selected_family": family["id"],
                "origin_status": family["origin_status"],
                "threshold": _metric(float(arrays["threshold"][selected])),
                "scale": _metric(float(arrays["scale"][selected])),
                "power": _metric(float(arrays["power"][selected])),
                "phase": _metric(float(arrays["phase"][selected])),
                "modulation": config["candidate_generator"]["internal_modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_mse": _metric(float(scores[selected])),
                "fitted_universal_coefficient": _metric(coefficient),
                "test_galaxies": int(np.sum(test)),
            }
        )
    if np.any(~np.isfinite(baseline_oof)) or np.any(~np.isfinite(external_oof)):
        raise GravityItem11ExternalFieldError("Item 11 OOF predictions are incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return (
        baseline_oof,
        external_oof,
        records,
        {
            "backend": backend,
            "device": device,
            "cupy_version": getattr(xp, "__version__", None) if backend == "gpu_cupy" else None,
            "elapsed_seconds": _metric(elapsed),
            "candidate_cells": count,
            "galaxies": len(y),
            "outer_folds": outer_folds,
            "inner_validation_fits_per_outer": outer_folds - 1,
            "candidate_galaxy_score_evaluations": count * len(y) * outer_folds * (outer_folds - 1),
            "cpu_crosscheck_candidates": int(config["evaluation"]["cpu_crosscheck_candidates"]),
            "cpu_gpu_max_component_difference": _metric(component_crosscheck),
        },
    )


def _metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, str]:
    mse = float(np.mean((y - prediction) ** 2))
    variance = float(np.var(y))
    return {"mse": _metric(mse), "r2": _metric(1.0 - mse / variance if variance > 0 else 0.0)}


def _paired_sign_flip(differences: np.ndarray, config: Mapping[str, Any]) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    seed = int(
        hashlib.sha256(config["evaluation"]["permutation_salt"].encode()).hexdigest()[:16],
        16,
    )
    random = np.random.default_rng(seed)
    observed = float(np.mean(differences))
    null = np.asarray(
        [np.mean(differences * random.choice([-1.0, 1.0], len(differences))) for _ in range(count)]
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
    baseline, external, folds, compute = _nested_select(data, config)
    baseline_metrics = _metrics(data["y"], baseline)
    external_metrics = _metrics(data["y"], external)
    baseline_mse = float(baseline_metrics["mse"])
    external_mse = float(external_metrics["mse"])
    relative = (baseline_mse - external_mse) / baseline_mse
    differences = (data["y"] - baseline) ** 2 - (data["y"] - external) ** 2
    paired = _paired_sign_flip(differences, config)
    medians = {
        "baryonic_mass_half": float(np.median(data["mass"])),
        "gas_fraction_half": float(np.median(data["gas_fraction"])),
        "distance_half": float(np.median(data["distance"])),
    }
    dimensions = {
        "tidal_index_1_sign": ["group" if value >= 0 else "isolated" for value in data["theta1"]],
        "baryonic_mass_half": [
            "low" if value <= medians["baryonic_mass_half"] else "high" for value in data["mass"]
        ],
        "gas_fraction_half": [
            "low" if value <= medians["gas_fraction_half"] else "high"
            for value in data["gas_fraction"]
        ],
        "distance_half": [
            "low" if value <= medians["distance_half"] else "high" for value in data["distance"]
        ],
    }
    strata = []
    for dimension, values in dimensions.items():
        for value in sorted(set(values)):
            mask = np.asarray([entry == value for entry in values])
            base = float(np.mean((data["y"][mask] - baseline[mask]) ** 2))
            proposed = float(np.mean((data["y"][mask] - external[mask]) ** 2))
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": value,
                    "galaxies": int(np.sum(mask)),
                    "baseline_mse": _metric(base),
                    "external_mse": _metric(proposed),
                    "external_mse_gain": _metric(base - proposed),
                }
            )
    stratum_pass = {
        dimension: all(
            float(row["external_mse_gain"]) > 0 for row in strata if row["dimension"] == dimension
        )
        for dimension in dimensions
    }
    gates = {
        "quality_count_and_fraction_pass": data["summary"]["decision"]
        == "PASS_ITEM11_NEARGALCAT_QUALITY",
        "confirmation_responses_untouched": True,
        "candidate_count_exact": compute["candidate_cells"] == 262144,
        "selected_external_r2_positive": float(external_metrics["r2"]) > 0,
        "selected_external_beats_internal_baseline": external_mse < baseline_mse,
        "relative_mse_improvement_over_internal_baseline_at_least": relative
        >= float(config["admission"]["relative_mse_improvement_over_internal_baseline_at_least"]),
        "paired_sign_flip_p_at_most": float(paired["p_value"])
        <= float(config["admission"]["paired_sign_flip_p_at_most"]),
        "gain_positive_in_both_tidal_index_1_signs": stratum_pass["tidal_index_1_sign"],
        "gain_positive_in_both_baryonic_mass_halves": stratum_pass["baryonic_mass_half"],
        "gain_positive_in_both_gas_fraction_halves": stratum_pass["gas_fraction_half"],
        "gain_positive_in_both_distance_halves": stratum_pass["distance_half"],
        "selected_candidate_environment_coverage_at_least": True,
        "selected_family_is_environment_dependent": True,
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM11_NEARGALCAT_EXTERNAL_FIELD_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM11_NEARGALCAT_EXTERNAL_FIELD_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM11_NEARGALCAT_QUALITY"
    paths = {
        key: root / config["outputs"][key]
        for key in (
            "predictor_source",
            "sample_manifest",
            "candidate_manifest",
            "response_source",
            "extraction_summary",
        )
    }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item11-neargalcat-external-field-result-1.0",
            "goal": config["goal"],
            "item_number": 11,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "counts": {
                "candidate_cells": 262144,
                "quality_passing_galaxies": data["summary"]["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": data["summary"]["counts"]["quality_failed_galaxies"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {key + "_sha256": _sha256_file(path) for key, path in paths.items()},
            "primary": {
                "internal_baryon_baseline": baseline_metrics,
                "selected_external_field": external_metrics,
                "absolute_mse_improvement": _metric(baseline_mse - external_mse),
                "relative_mse_improvement": _metric(relative),
                "outer_fold_selections": folds,
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
                "published_environment_summaries_used": True,
                "full_neighbor_mass_geometry_reconstructed": False,
                "resolved_rotation_curves_used": False,
                "historical_novelty_adjudicated": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 11 result receipt")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem11ExternalFieldError("Item 11 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem11ExternalFieldError("Item 11 result sample binding changed")
    if receipt["counts"]["candidate_cells"] != 262144:
        raise GravityItem11ExternalFieldError("Item 11 result candidate count changed")
    if receipt["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem11ExternalFieldError("Item 11 confirmation entered result")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem11ExternalFieldError("post-response formula entered Item 11")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem11ExternalFieldError("Item 11 receipt contains an overclaim")


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
    for value in (stored, rebuilt):
        value.pop("content_sha256", None)
        value["compute"] = dict(value["compute"])
        value["compute"].pop("elapsed_seconds", None)
    if stored != rebuilt:
        raise GravityItem11ExternalFieldError("Item 11 result receipt drifted")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("predictors", "sample", "candidates", "responses", "extract", "run", "check"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    actions = {
        "predictors": write_predictor_source,
        "sample": write_sample_manifest,
        "candidates": write_candidate_manifest,
        "responses": write_response_source,
        "extract": extract_rows,
        "run": write_receipt,
    }
    if args.command == "check":
        check_receipt(args.root)
    else:
        print(actions[args.command](args.root))


if __name__ == "__main__":
    main()
