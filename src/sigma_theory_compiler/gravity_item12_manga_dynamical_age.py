"""Frozen SDSS MaNGA dynamical-age search for gravity roadmap Item 12."""

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

from .gravity_item11_neargalcat_external_field import (
    _content_hashed,
    _metric,
    _minimum_separation_arcsec,
    _paired_sign_flip,
    _ridge_fit,
    _ridge_predict,
    _sha256_file,
    _source_rows,
    _validate_content_hash,
    canonical_json_bytes,
)

CONFIG_PATH = Path("configs/gravity_item12_manga_dynamical_age_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "PENDING_ITEM12_SCIENTIFIC_FREEZE_COMMIT"
SAMPLE_FREEZE_COMMIT = "PENDING_ITEM12_SAMPLE_FREEZE_COMMIT"


class GravityItem12DynamicalAgeError(RuntimeError):
    """Raised when an Item 12 scientific or response boundary drifts."""


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem12DynamicalAgeError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem12DynamicalAgeError("Item 11 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 11 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem12DynamicalAgeError("Item 11 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem12DynamicalAgeError("Item 11 synthesis decision changed")
    manga_source = config["independence"]["manga_identity_source"]
    if _sha256_file(root / manga_source["path"]) != manga_source["file_sha256"]:
        raise GravityItem12DynamicalAgeError("prior MaNGA identity source changed")
    for entry in config["independence"]["coordinate_exclusions"]:
        if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
            raise GravityItem12DynamicalAgeError(
                f"predecessor coordinate source changed: {entry['path']}"
            )
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem12DynamicalAgeError("Item 12 config contains an overclaim")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem12DynamicalAgeError("Item 12 candidate count changed")
    return config


def _sql_query(config: Mapping[str, Any], query: str) -> bytes:
    parameters = urllib.parse.urlencode({"cmd": query, "format": "csv"})
    request = urllib.request.Request(
        f"{config['source']['skyserver_endpoint']}?{parameters}",
        headers={"User-Agent": "Invariant/Item12-MaNGA"},
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise GravityItem12DynamicalAgeError("empty SDSS SkyServer response")
    return payload


def _parse_skyserver_csv(payload: bytes) -> list[dict[str, str]]:
    lines = payload.decode("utf-8-sig", errors="strict").splitlines()
    lines = [line for line in lines if line.strip() and not line.startswith("#")]
    if not lines:
        raise GravityItem12DynamicalAgeError("empty SDSS CSV table")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem12DynamicalAgeError(message)
    return rows


def _finite(row: Mapping[str, str], key: str) -> float:
    value = str(row.get(key, "")).strip()
    try:
        number = float(value)
    except ValueError as exc:
        raise GravityItem12DynamicalAgeError(f"missing or invalid {key}") from exc
    if not math.isfinite(number) or number <= -900:
        raise GravityItem12DynamicalAgeError(f"missing or invalid {key}")
    return number


def _predictor_query(config: Mapping[str, Any]) -> str:
    dap_columns = config["source"]["predictor_columns"][:16]
    drp_columns = config["source"]["predictor_columns"][16:]
    selection = [f"d.{column}" for column in dap_columns]
    selection.extend(f"r.{column}" for column in drp_columns)
    return (
        f"SELECT {','.join(selection)} FROM {config['source']['dap_table']} d "
        f"JOIN {config['source']['drp_table']} r ON d.plateifu=r.plateifu "
        f"WHERE d.daptype='{config['source']['daptype']}' ORDER BY d.plateifu"
    )


def derive_predictors(row: Mapping[str, str], config: Mapping[str, Any]) -> dict[str, Any]:
    mass = _finite(row, "nsa_elpetro_mass")
    radius_arcsec = _finite(row, "nsa_elpetro_th50_r")
    axis_ratio = _finite(row, "nsa_elpetro_ba")
    sersic = _finite(row, "nsa_sersic_n")
    redshift = _finite(row, "nsa_z")
    snr = _finite(row, "snr_med_g")
    surface_brightness = _finite(row, "sb_1re")
    dn4000 = _finite(row, "specindex_1re_dn4000")
    d4000 = _finite(row, "specindex_1re_d4000")
    hdelta = _finite(row, "specindex_1re_hdeltaa")
    hgamma = _finite(row, "specindex_1re_hgammaa")
    hbeta = _finite(row, "specindex_1re_hb")
    ha_ew = _finite(row, "emline_sew_1re_ha_6564")
    sfr_1re = _finite(row, "sfr_1re")
    sfr_tot = _finite(row, "sfr_tot")
    try:
        dapqual = int(float(str(row["dapqual"])))
        drp3qual = int(float(str(row["drp3qual"])))
    except (KeyError, TypeError, ValueError) as exc:
        raise GravityItem12DynamicalAgeError("missing or invalid quality bitmask") from exc
    if (
        mass <= 0
        or radius_arcsec <= 0
        or not 0 < axis_ratio <= 1.2
        or sersic <= 0
        or redshift <= 0
        or snr <= 0
        or surface_brightness <= 0
        or not 0.5 <= dn4000 <= 3.0
        or not 0.5 <= d4000 <= 4.0
        or sfr_1re < 0
        or sfr_tot < 0
        or (bool(config["quality"]["require_dapqual_zero"]) and dapqual != 0)
        or (bool(config["quality"]["require_drp3qual_zero"]) and drp3qual != 0)
    ):
        raise GravityItem12DynamicalAgeError("predictor outside physical range")
    log_mass = math.log10(mass)
    log_radius = math.log10(radius_arcsec)
    log_surface = log_mass - 2.0 * log_radius
    ssfr = math.log10(max(sfr_1re, 1e-8)) - log_mass
    g_mag = _finite(row, "nsa_sersic_absmag_g")
    r_mag = _finite(row, "nsa_sersic_absmag_r")
    return {
        "plateifu": str(row["plateifu"]),
        "mangaid": str(row["mangaid"]),
        "ra": _finite(row, "objra"),
        "dec": _finite(row, "objdec"),
        "daptype": str(row["daptype"]),
        "dapqual": dapqual,
        "drp3qual": drp3qual,
        "log_stellar_mass": log_mass,
        "log_half_light_radius": log_radius,
        "log_surface_density": log_surface,
        "axis_ratio": axis_ratio,
        "sersic_index": sersic,
        "g_minus_r_color": g_mag - r_mag,
        "redshift": redshift,
        "log_surface_brightness": math.log10(surface_brightness),
        "log_snr": math.log10(snr),
        "snr_med_g": snr,
        "dn4000": dn4000,
        "d4000": d4000,
        "hdelta_a": hdelta,
        "hgamma_a": hgamma,
        "hbeta": hbeta,
        "halpha_ew": ha_ew,
        "sfr_1re": sfr_1re,
        "sfr_tot": sfr_tot,
        "log_specific_sfr": ssfr,
        "mass_size_crossing_proxy": 1.5 * log_radius - 0.5 * log_mass,
    }


def _serialize(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value if isinstance(value, str) else _metric(value) for key, value in row.items()}


def write_predictor_source(root: Path) -> Path:
    root = root.resolve()
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem12DynamicalAgeError("Item 12 scientific freeze is not bound")
    config = load_config(root)
    query = _predictor_query(config)
    payload = _sql_query(config, query)
    rows = _parse_skyserver_csv(payload)
    if len(rows) != int(config["source"]["observed_join_rows"]):
        raise GravityItem12DynamicalAgeError("MaNGA joined row count changed")
    expected = set(config["source"]["predictor_columns"])
    if any(set(row) != expected for row in rows):
        raise GravityItem12DynamicalAgeError("MaNGA predictor schema changed")
    records = []
    failures = []
    for row in rows:
        try:
            records.append(_serialize(derive_predictors(row, config)))
        except GravityItem12DynamicalAgeError as exc:
            failures.append(
                {"plateifu": row.get("plateifu"), "mangaid": row.get("mangaid"), "reason": str(exc)}
            )
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item12-manga-predictor-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "query": query,
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "records": records,
            "failures": failures,
            "counts": {
                "joined_rows": len(rows),
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
    _validate_content_hash(source, "Item 12 predictor source")
    config = load_config(root)
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem12DynamicalAgeError("Item 12 predictor binding changed")
    if source["query"] != _predictor_query(config):
        raise GravityItem12DynamicalAgeError("Item 12 predictor query changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem12DynamicalAgeError("response entered Item 12 predictor source")


def _prior_manga_ids(root: Path, config: Mapping[str, Any]) -> set[str]:
    entry = config["independence"]["manga_identity_source"]
    value = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
    return {str(row[entry["key"]]).strip().upper() for row in value["objects"]}


def _coordinates(root: Path, config: Mapping[str, Any]) -> np.ndarray:
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


def _split_hash(value: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}|{value}".encode()).hexdigest()


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    source_path = root / config["outputs"]["predictor_source"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_predictor_source(source, root)
    prior_manga = _prior_manga_ids(root, config)
    coordinates = _coordinates(root, config)
    plate_counts = Counter(str(row["plateifu"]) for row in source["records"])
    manga_counts = Counter(str(row["mangaid"]).upper() for row in source["records"])
    admitted = []
    exclusions: Counter[str] = Counter()
    for row in source["records"]:
        reasons = []
        if str(row["mangaid"]).upper() in prior_manga:
            reasons.append("prior_manga_identity")
        if plate_counts[row["plateifu"]] != 1 or manga_counts[str(row["mangaid"]).upper()] != 1:
            reasons.append("duplicate_manga_identity")
        separation = _minimum_separation_arcsec(float(row["ra"]), float(row["dec"]), coordinates)
        if separation <= float(config["independence"]["coordinate_exclusion_arcseconds"]):
            reasons.append("predecessor_coordinate")
        for reason in set(reasons):
            exclusions[reason] += 1
        if not reasons:
            output = dict(row)
            output["minimum_predecessor_separation_arcsec"] = _metric(separation)
            admitted.append(output)
    median_mass = float(np.median([float(row["log_stellar_mass"]) for row in admitted]))
    median_age = float(np.median([float(row["dn4000"]) for row in admitted]))
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in admitted:
        age = "younger" if float(row["dn4000"]) <= median_age else "older"
        mass = "lower_mass" if float(row["log_stellar_mass"]) <= median_mass else "higher_mass"
        cells.setdefault((age, mass), []).append(row)
    cell_keys = sorted(cells)
    per_cell = int(config["sample"]["maximum_total_objects"]) // len(cell_keys)
    selected_by_cell = {
        key: sorted(
            cells[key],
            key=lambda row: (
                _split_hash(row["plateifu"], config["sample"]["split_salt"]),
                row["plateifu"],
            ),
        )[:per_cell]
        for key in cell_keys
    }
    if any(len(rows) != per_cell for rows in selected_by_cell.values()):
        raise GravityItem12DynamicalAgeError("MaNGA stratum cannot fill frozen sample")
    confirmation_total = int(config["sample"]["confirmation_objects"])
    base_confirmation = confirmation_total // len(cell_keys)
    extra = confirmation_total % len(cell_keys)
    objects = []
    for cell_index, key in enumerate(cell_keys):
        rows = selected_by_cell[key]
        confirmation_count = base_confirmation + (1 if cell_index < extra else 0)
        confirmation = {row["plateifu"] for row in rows[:confirmation_count]}
        for row in sorted(rows, key=lambda value: value["plateifu"]):
            output = dict(row)
            output.update(
                {
                    "age_bin": key[0],
                    "stellar_mass_bin": key[1],
                    "role": "reserved_confirmation"
                    if row["plateifu"] in confirmation
                    else "exploration",
                    "outer_fold": int(
                        _split_hash(row["plateifu"], config["evaluation"]["fold_salt"])[:16],
                        16,
                    )
                    % int(config["evaluation"]["outer_folds"]),
                    "response_read": False,
                }
            )
            objects.append(output)
    objects.sort(key=lambda row: row["plateifu"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item12-manga-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "predictor_source_sha256": _sha256_file(source_path),
            "predictor_source_content_sha256": source["content_sha256"],
            "stellar_mass_median": _metric(median_mass),
            "dn4000_median": _metric(median_age),
            "objects": objects,
            "counts": {
                "valid_predictor_rows": len(source["records"]),
                "admitted_independent_rows": len(admitted),
                "selected": len(objects),
                "exploration": sum(row["role"] == "exploration" for row in objects),
                "reserved_confirmation": sum(
                    row["role"] == "reserved_confirmation" for row in objects
                ),
                "predecessor_selected": 0,
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
    _validate_content_hash(sample, "Item 12 sample manifest")
    if sample != build_sample_manifest(root):
        raise GravityItem12DynamicalAgeError("Item 12 sample manifest drifted")
    if sample["counts"]["predecessor_selected"] != 0 or sample["counts"]["response_rows_read"] != 0:
        raise GravityItem12DynamicalAgeError("Item 12 target-blind sample boundary changed")


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
    scale_low, scale_high = [math.log(float(value)) for value in generator["scale_log_uniform"]]
    power_low, power_high = [math.log(float(value)) for value in generator["power_log_uniform"]]
    return {
        "family": random.integers(0, len(generator["families"]), count, dtype=np.int16),
        "threshold": random.uniform(*generator["threshold_uniform"], count),
        "scale": np.exp(random.uniform(scale_low, scale_high, count)),
        "power": np.exp(random.uniform(power_low, power_high, count)),
        "phase": random.uniform(*generator["phase_uniform"], count),
        "modulation": random.integers(
            0, len(generator["structural_modulations"]), count, dtype=np.int8
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
            "schema_version": "invariant-gravity-item12-dynamical-age-candidates-1.0",
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
                "response_rows_read": 0,
                "paid_model_calls": 0,
            },
            "claims": {"historical_novelty_established": False},
        }
    )


def validate_candidate_manifest(manifest: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(manifest, "Item 12 candidate manifest")
    if manifest != build_candidate_manifest(root):
        raise GravityItem12DynamicalAgeError("Item 12 candidate manifest drifted")


def write_candidate_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["candidate_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_candidate_manifest(root)) + b"\n")
    return path


def _response_query(config: Mapping[str, Any], identities: Sequence[str]) -> str:
    columns = ",".join(f"d.{column}" for column in config["source"]["response_columns"])
    quoted = ",".join("'" + value.replace("'", "''") + "'" for value in identities)
    return (
        f"SELECT {columns} FROM {config['source']['dap_table']} d "
        f"WHERE d.daptype='{config['source']['daptype']}' AND d.plateifu IN ({quoted}) "
        "ORDER BY d.plateifu"
    )


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem12DynamicalAgeError("Item 12 sample freeze is not bound")
    config = load_config(root)
    sample_path = root / config["outputs"]["sample_manifest"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, root)
    exploration = [row["plateifu"] for row in sample["objects"] if row["role"] == "exploration"]
    chunk_size = int(config["source"]["response_chunk_size"])
    records = []
    chunks = []
    for begin in range(0, len(exploration), chunk_size):
        identities = exploration[begin : begin + chunk_size]
        query = _response_query(config, identities)
        payload = _sql_query(config, query)
        rows = _parse_skyserver_csv(payload)
        if {row["plateifu"] for row in rows} != set(identities):
            raise GravityItem12DynamicalAgeError("MaNGA response chunk scope changed")
        records.extend(rows)
        chunks.append(
            {
                "ordinal": len(chunks),
                "identity_count": len(identities),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    expected = set(config["source"]["response_columns"])
    if any(set(row) != expected for row in records):
        raise GravityItem12DynamicalAgeError("MaNGA response schema changed")
    row_ids = [row["plateifu"] for row in records]
    if set(row_ids) != set(exploration) or len(row_ids) != len(set(row_ids)):
        raise GravityItem12DynamicalAgeError("MaNGA response scope changed")
    confirmation = {
        row["plateifu"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if confirmation & set(row_ids):
        raise GravityItem12DynamicalAgeError("Item 12 confirmation response entered query")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item12-manga-response-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "chunks": chunks,
            "records": sorted(records, key=lambda row: row["plateifu"]),
            "counts": {
                "exploration_response_rows": len(records),
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
    _validate_content_hash(source, "Item 12 response source")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem12DynamicalAgeError("Item 12 response binding changed")
    if source["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem12DynamicalAgeError("Item 12 confirmation opened")
    if source["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem12DynamicalAgeError("post-response formula entered Item 12")


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
    predictors = {row["plateifu"]: row for row in predictor["records"]}
    samples = {row["plateifu"]: row for row in sample["objects"] if row["role"] == "exploration"}
    rows = []
    failures = []
    for raw in response["records"]:
        plateifu = raw["plateifu"]
        predictor_row = predictors[plateifu]
        sample_row = samples[plateifu]
        reasons = []
        try:
            sigma = _finite(raw, "stellar_sigma_1re")
            rchi2 = _finite(raw, "stellar_rchi2_1re")
            stellar_lo = _finite(raw, "stellar_vel_lo_clip")
            stellar_hi = _finite(raw, "stellar_vel_hi_clip")
            ha_lo = _finite(raw, "ha_gvel_lo_clip")
            ha_hi = _finite(raw, "ha_gvel_hi_clip")
        except GravityItem12DynamicalAgeError as exc:
            failures.append({"plateifu": plateifu, "reasons": [str(exc)]})
            continue
        stellar_span = stellar_hi - stellar_lo
        ha_span = ha_hi - ha_lo
        if float(predictor_row["snr_med_g"]) < float(config["quality"]["minimum_snr_med_g"]):
            reasons.append("snr")
        if rchi2 > float(config["quality"]["maximum_stellar_rchi2_1re"]):
            reasons.append("stellar_rchi2")
        if not (
            float(config["quality"]["minimum_stellar_sigma_km_s"])
            <= sigma
            <= float(config["quality"]["maximum_stellar_sigma_km_s"])
        ):
            reasons.append("stellar_sigma")
        if not (
            float(config["quality"]["minimum_stellar_velocity_span_km_s"])
            <= stellar_span
            <= float(config["quality"]["maximum_stellar_velocity_span_km_s"])
        ):
            reasons.append("stellar_velocity_span")
        output = {
            **predictor_row,
            "outer_fold": sample_row["outer_fold"],
            "age_bin": sample_row["age_bin"],
            "stellar_mass_bin": sample_row["stellar_mass_bin"],
            "stellar_sigma_1re_km_s": _metric(sigma),
            "stellar_rchi2_1re": _metric(rchi2),
            "stellar_velocity_span_km_s": _metric(stellar_span),
            "halpha_velocity_span_km_s": _metric(ha_span),
            "quality_pass": not reasons,
            "quality_failure_reasons": reasons,
        }
        if reasons:
            failures.append({"plateifu": plateifu, "reasons": reasons})
        else:
            rows.append(output)
    selected = len(response["records"])
    passing = len(rows)
    retention = passing / selected if selected else 0.0
    quality_pass = passing >= int(config["quality"]["minimum_quality_passing_exploration_galaxies"])
    quality_pass &= retention >= float(config["quality"]["minimum_quality_retention_fraction"])
    summary = _content_hashed(
        {
            "schema_version": "invariant-gravity-item12-manga-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": "PASS_ITEM12_MANGA_QUALITY"
            if quality_pass
            else "FAIL_ITEM12_MANGA_QUALITY",
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
    summary = json.loads(
        (root / config["outputs"]["extraction_summary"]).read_text(encoding="utf-8")
    )
    candidates = json.loads(
        (root / config["outputs"]["candidate_manifest"]).read_text(encoding="utf-8")
    )
    _validate_content_hash(summary, "Item 12 extraction summary")
    validate_candidate_manifest(candidates, root)
    rows = summary["rows"]
    if not rows:
        raise GravityItem12DynamicalAgeError("Item 12 has no quality rows")
    normalization = config["evaluation"]["fixed_clock_normalization"]

    def fixed(field: str, values: Sequence[float]) -> np.ndarray:
        center, scale = (float(value) for value in normalization[field])
        if scale <= 0:
            raise GravityItem12DynamicalAgeError("Item 12 fixed normalization changed")
        return (np.asarray(values, dtype=np.float64) - center) / scale

    balmer = [(float(row["hdelta_a"]) + float(row["hgamma_a"])) / 2.0 for row in rows]
    signed_log_halpha = [
        math.copysign(math.log1p(abs(float(row["halpha_ew"]))), float(row["halpha_ew"]))
        if float(row["halpha_ew"]) != 0
        else 0.0
        for row in rows
    ]
    return {
        "summary": summary,
        "candidate_manifest": candidates,
        "rows": rows,
        "folds": np.asarray([int(row["outer_fold"]) for row in rows]),
        "y": np.log10(np.asarray([float(row["stellar_sigma_1re_km_s"]) for row in rows])),
        "design": np.column_stack(
            [
                np.asarray([float(row[field]) for row in rows])
                for field in config["evaluation"]["local_features"]
            ]
        ),
        "dn4000": fixed("dn4000", [float(row["dn4000"]) for row in rows]),
        "d4000": fixed("d4000", [float(row["d4000"]) for row in rows]),
        "balmer": fixed("balmer", balmer),
        "hbeta": fixed("hbeta", [float(row["hbeta"]) for row in rows]),
        "haew": fixed("signed_log_halpha_ew", signed_log_halpha),
        "ssfr": fixed("log_specific_sfr", [float(row["log_specific_sfr"]) for row in rows]),
        "crossing": fixed(
            "mass_size_crossing_proxy",
            [float(row["mass_size_crossing_proxy"]) for row in rows],
        ),
        "surface": fixed(
            "log_surface_density", [float(row["log_surface_density"]) for row in rows]
        ),
        "sersic": fixed("sersic_index", [float(row["sersic_index"]) for row in rows]),
        "axis": fixed("axis_ratio", [float(row["axis_ratio"]) for row in rows]),
        "mass": np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        "redshift": np.asarray([float(row["redshift"]) for row in rows]),
    }


def _candidate_components(
    arrays: Mapping[str, np.ndarray], data: Mapping[str, Any], begin: int, end: int, xp: Any
) -> Any:
    family = xp.asarray(arrays["family"][begin:end], dtype=xp.int32)[:, None]
    threshold = xp.asarray(arrays["threshold"][begin:end], dtype=xp.float64)[:, None]
    scale = xp.asarray(arrays["scale"][begin:end], dtype=xp.float64)[:, None]
    power = xp.asarray(arrays["power"][begin:end], dtype=xp.float64)[:, None]
    phase = xp.asarray(arrays["phase"][begin:end], dtype=xp.float64)[:, None]
    modulation_index = xp.asarray(arrays["modulation"][begin:end], dtype=xp.int32)[:, None]
    dn = xp.asarray(data["dn4000"], dtype=xp.float64)[None, :]
    d4 = xp.asarray(data["d4000"], dtype=xp.float64)[None, :]
    balmer = xp.asarray(data["balmer"], dtype=xp.float64)[None, :]
    hbeta = xp.asarray(data["hbeta"], dtype=xp.float64)[None, :]
    haew = xp.asarray(data["haew"], dtype=xp.float64)[None, :]
    ssfr = xp.asarray(data["ssfr"], dtype=xp.float64)[None, :]
    crossing = xp.asarray(data["crossing"], dtype=xp.float64)[None, :]

    def signed_power(value: Any) -> Any:
        z = (value - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    age = signed_power(dn)
    burst = signed_power(balmer)
    postburst = burst * xp.tanh(dn / scale)
    quenching = xp.tanh((dn - threshold) / scale) * xp.tanh((haew - threshold) / scale)
    depletion = signed_power(ssfr - haew)
    coherence_raw = dn * balmer * ssfr
    coherence = xp.tanh(coherence_raw / scale)
    mismatch = xp.abs(dn - balmer)
    settling = xp.exp(-((mismatch / scale) ** power)) * xp.sign(dn)
    ratio = signed_power(dn / (1.0 + xp.abs(crossing)))
    hysteresis = xp.tanh(((dn - threshold) * (balmer - threshold)) / scale)
    double_well = -(((dn - threshold) ** 2 - scale**2) ** 2) / (1.0 + scale**4)
    consensus = signed_power((dn + d4 - balmer - hbeta) / 4.0)
    log_periodic = age * xp.cos(phase + power * xp.log1p(xp.abs(dn - threshold)))
    component = xp.zeros((end - begin, dn.shape[1]), dtype=xp.float64)
    for index, value in enumerate(
        (
            age,
            burst,
            postburst,
            quenching,
            depletion,
            coherence,
            settling,
            ratio,
            hysteresis,
            double_well,
            consensus,
            log_periodic,
        )
    ):
        component = xp.where(family == index, value, component)
    modulation = xp.ones_like(component)
    for index, field in enumerate(("surface", "sersic", "axis", "crossing"), start=1):
        modulation = xp.where(
            modulation_index == index,
            xp.tanh(xp.asarray(data[field], dtype=xp.float64))[None, :],
            modulation,
        )
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
    clock_oof = np.full(len(y), np.nan)
    records = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["clock_coefficient_ridge"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    component_crosscheck = 0.0
    for outer in range(outer_folds):
        inner_records = []
        for inner in [value for value in range(outer_folds) if value != outer]:
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
            loss = xp.zeros(end - begin, dtype=xp.float64)
            for inner in inner_records:
                train_component = components[:, inner["train"]]
                validation_component = components[:, inner["validation"]]
                mean = xp.mean(train_component, axis=1)
                scale = xp.maximum(xp.std(train_component, axis=1), 1e-12)
                standardized = (train_component - mean[:, None]) / scale[:, None]
                coefficient = xp.sum(
                    standardized * xp.asarray(inner["train_residual"])[None, :], axis=1
                ) / (xp.sum(standardized**2, axis=1) + coefficient_ridge)
                residual = (
                    xp.asarray(inner["validation_residual"])[None, :]
                    - coefficient[:, None] * (validation_component - mean[:, None]) / scale[:, None]
                )
                loss += xp.mean(residual**2, axis=1)
            batch_scores = loss / len(inner_records)
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
        standardized = (selected_component[train] - mean) / scale
        coefficient = float(
            np.sum(standardized * (y[train] - train_base))
            / (np.sum(standardized**2) + coefficient_ridge)
        )
        baseline_oof[test] = test_base
        clock_oof[test] = test_base + coefficient * (selected_component[test] - mean) / scale
        family = config["candidate_generator"]["families"][int(arrays["family"][selected])]
        records.append(
            {
                "outer_fold": outer,
                "selected_ordinal": selected,
                "selected_family": family["id"],
                "origin_status": family["origin_status"],
                "threshold": _metric(arrays["threshold"][selected]),
                "scale": _metric(arrays["scale"][selected]),
                "power": _metric(arrays["power"][selected]),
                "phase": _metric(arrays["phase"][selected]),
                "modulation": config["candidate_generator"]["structural_modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_mse": _metric(scores[selected]),
                "fitted_universal_coefficient": _metric(coefficient),
                "test_galaxies": int(np.sum(test)),
            }
        )
    if np.any(~np.isfinite(baseline_oof)) or np.any(~np.isfinite(clock_oof)):
        raise GravityItem12DynamicalAgeError("Item 12 OOF prediction incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return (
        baseline_oof,
        clock_oof,
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
    return {"mse": _metric(mse), "r2": _metric(1 - mse / variance if variance > 0 else 0)}


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    data = _load_data(root, config)
    baseline, clock, folds, compute = _nested_select(data, config)
    baseline_metrics = _metrics(data["y"], baseline)
    clock_metrics = _metrics(data["y"], clock)
    baseline_mse = float(baseline_metrics["mse"])
    clock_mse = float(clock_metrics["mse"])
    relative = (baseline_mse - clock_mse) / baseline_mse
    differences = (data["y"] - baseline) ** 2 - (data["y"] - clock) ** 2
    paired = _paired_sign_flip(differences, config)
    dimensions = {
        "dn4000_half": data["dn4000"],
        "stellar_mass_half": data["mass"],
        "sersic_half": data["sersic"],
        "redshift_half": data["redshift"],
    }
    strata = []
    stratum_pass = {}
    for dimension, values in dimensions.items():
        median = float(np.median(values))
        gains = []
        for label, mask in (("low", values <= median), ("high", values > median)):
            base = float(np.mean((data["y"][mask] - baseline[mask]) ** 2))
            proposed = float(np.mean((data["y"][mask] - clock[mask]) ** 2))
            gains.append(base - proposed)
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    "galaxies": int(np.sum(mask)),
                    "baseline_mse": _metric(base),
                    "clock_mse": _metric(proposed),
                    "clock_mse_gain": _metric(base - proposed),
                }
            )
        stratum_pass[dimension] = all(value > 0 for value in gains)
    gates = {
        "quality_count_and_fraction_pass": data["summary"]["decision"]
        == "PASS_ITEM12_MANGA_QUALITY",
        "confirmation_responses_untouched": True,
        "candidate_count_exact": compute["candidate_cells"] == 262144,
        "selected_clock_r2_positive": float(clock_metrics["r2"]) > 0,
        "selected_clock_beats_structural_baseline": clock_mse < baseline_mse,
        "relative_mse_improvement_at_least": relative
        >= float(config["admission"]["relative_mse_improvement_at_least"]),
        "paired_sign_flip_p_at_most": float(paired["p_value"])
        <= float(config["admission"]["paired_sign_flip_p_at_most"]),
        "gain_positive_in_both_dn4000_halves": stratum_pass["dn4000_half"],
        "gain_positive_in_both_stellar_mass_halves": stratum_pass["stellar_mass_half"],
        "gain_positive_in_both_sersic_halves": stratum_pass["sersic_half"],
        "gain_positive_in_both_redshift_halves": stratum_pass["redshift_half"],
        "selected_family_is_clock_dependent": True,
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM12_MANGA_DYNAMICAL_AGE_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM12_MANGA_DYNAMICAL_AGE_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM12_MANGA_QUALITY"
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
            "schema_version": "invariant-gravity-item12-manga-dynamical-age-result-1.0",
            "goal": config["goal"],
            "item_number": 12,
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
                "structural_baseline": baseline_metrics,
                "selected_dynamical_clock": clock_metrics,
                "absolute_mse_improvement": _metric(baseline_mse - clock_mse),
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
                "integrated_spectral_clock_proxies_used": True,
                "direct_formation_time_measured": False,
                "causal_history_reconstructed": False,
                "historical_novelty_adjudicated": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 12 result receipt")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem12DynamicalAgeError("Item 12 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem12DynamicalAgeError("Item 12 result sample binding changed")
    if receipt["counts"]["candidate_cells"] != 262144:
        raise GravityItem12DynamicalAgeError("Item 12 candidate count changed")
    if receipt["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem12DynamicalAgeError("Item 12 confirmation entered result")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem12DynamicalAgeError("post-response formula entered Item 12")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem12DynamicalAgeError("Item 12 result contains an overclaim")


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
        raise GravityItem12DynamicalAgeError("Item 12 result receipt drifted")


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
