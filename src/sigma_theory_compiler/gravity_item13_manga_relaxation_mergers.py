"""Frozen MaNGA relaxation-and-merger search for gravity roadmap Item 13."""

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
    _ridge_fit,
    _ridge_predict,
    _sha256_file,
    _source_rows,
    _validate_content_hash,
    canonical_json_bytes,
)

CONFIG_PATH = Path("configs/gravity_item13_manga_relaxation_mergers_v1.json")
SCIENTIFIC_FREEZE_COMMIT = "PENDING_ITEM13_SCIENTIFIC_FREEZE_COMMIT"
SAMPLE_FREEZE_COMMIT = "PENDING_ITEM13_SAMPLE_FREEZE_COMMIT"


class GravityItem13RelaxationError(RuntimeError):
    """Raised when an Item 13 scientific or response boundary drifts."""


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem13RelaxationError("stable gravity roadmap changed")
    predecessor_binding = config["predecessor"]
    predecessor_path = root / predecessor_binding["path"]
    if _sha256_file(predecessor_path) != predecessor_binding["file_sha256"]:
        raise GravityItem13RelaxationError("Item 12 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predecessor, "Item 12 synthesis")
    if predecessor.get("content_sha256") != predecessor_binding["content_sha256"]:
        raise GravityItem13RelaxationError("Item 12 synthesis content binding changed")
    if predecessor.get("decision") != predecessor_binding["required_decision"]:
        raise GravityItem13RelaxationError("Item 12 synthesis decision changed")
    predictor_binding = config["sources"]["item12_predictors"]
    predictor_path = root / predictor_binding["path"]
    if _sha256_file(predictor_path) != predictor_binding["file_sha256"]:
        raise GravityItem13RelaxationError("Item 12 predictor source changed")
    predictor = json.loads(predictor_path.read_text(encoding="utf-8"))
    _validate_content_hash(predictor, "Item 12 predictor source")
    if predictor.get("content_sha256") != predictor_binding["content_sha256"]:
        raise GravityItem13RelaxationError("Item 12 predictor content binding changed")
    item12_config_binding = config["sources"]["item12_config"]
    item12_config_path = root / item12_config_binding["path"]
    if _sha256_file(item12_config_path) != item12_config_binding["file_sha256"]:
        raise GravityItem13RelaxationError("Item 12 config changed")
    item12_config = json.loads(item12_config_path.read_text(encoding="utf-8"))
    if (
        config["prior_age_lead"]["fixed_clock_normalization"]
        != item12_config["evaluation"]["fixed_clock_normalization"]
    ):
        raise GravityItem13RelaxationError("Item 12 clock normalization changed")
    for section in ("identity_exclusions", "coordinate_exclusions"):
        for entry in config["independence"][section]:
            if _sha256_file(root / entry["path"]) != entry["file_sha256"]:
                raise GravityItem13RelaxationError(
                    f"predecessor exclusion source changed: {entry['path']}"
                )
    observed_cells = predecessor["evidence"]["attempt"]["fold_selections"]
    configured_cells = config["prior_age_lead"]["cells"]
    if len(observed_cells) != len(configured_cells):
        raise GravityItem13RelaxationError("Item 12 age-cell count changed")
    for observed, configured in zip(observed_cells, configured_cells, strict=True):
        for key in ("ordinal", "modulation"):
            observed_key = "selected_ordinal" if key == "ordinal" else key
            if observed[observed_key] != configured[key]:
                raise GravityItem13RelaxationError("Item 12 age-cell identity changed")
        for key in ("threshold", "scale", "power", "phase"):
            if not math.isclose(
                float(observed[key]), float(configured[key]), rel_tol=0.0, abs_tol=5e-13
            ):
                raise GravityItem13RelaxationError("Item 12 age-cell parameter changed")
    if any(bool(value) for value in config["claim_boundaries"].values()):
        raise GravityItem13RelaxationError("Item 13 config contains an overclaim")
    if int(config["candidate_generator"]["candidate_cells"]) != 262144:
        raise GravityItem13RelaxationError("Item 13 candidate count changed")
    return config


def _download(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/Item13-MaNGA"})
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise GravityItem13RelaxationError("empty remote payload")
    return payload


def _decode(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="strict").strip()
    return str(value).strip()


def _number(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise GravityItem13RelaxationError(f"missing or invalid {label}") from exc
    if not math.isfinite(result) or result <= -900:
        raise GravityItem13RelaxationError(f"missing or invalid {label}")
    return result


def _fixed(value: float, config: Mapping[str, Any], field: str) -> float:
    center, scale = [
        float(item) for item in config["evaluation"]["fixed_morphology_normalization"][field]
    ]
    if scale <= 0:
        raise GravityItem13RelaxationError("fixed morphology normalization changed")
    return (value - center) / scale


def _prior_age_component(row: Mapping[str, Any], config: Mapping[str, Any]) -> float:
    item12_normalization = config["prior_age_lead"]["fixed_clock_normalization"]

    def normalized(field: str, value: float) -> float:
        center, scale = (float(part) for part in item12_normalization[field])
        if scale <= 0:
            raise GravityItem13RelaxationError("fixed Item 12 normalization changed")
        return (value - center) / scale

    dn4000 = normalized("dn4000", float(row["dn4000"]))
    d4000 = normalized("d4000", float(row["d4000"]))
    balmer = normalized("balmer", (float(row["hdelta_a"]) + float(row["hgamma_a"])) / 2.0)
    hbeta = normalized("hbeta", float(row["hbeta"]))
    consensus = (dn4000 + d4000 - balmer - hbeta) / 4.0
    surface = math.tanh(normalized("log_surface_density", float(row["log_surface_density"])))
    components = []
    for cell in config["prior_age_lead"]["cells"]:
        z = (consensus - float(cell["threshold"])) / float(cell["scale"])
        magnitude = abs(z) ** float(cell["power"])
        bounded = math.copysign(magnitude / (1.0 + magnitude), z) if z else 0.0
        components.append(bounded * surface)
    return float(np.mean(components))


def derive_predictors(
    morphology: Mapping[str, Any], item12: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    cas_flag = int(_number(morphology["cas_flag"], "cas_flag"))
    if bool(config["quality"]["require_cas_flag_one"]) and cas_flag != 1:
        raise GravityItem13RelaxationError("unreliable CAS flag")
    concentration = _number(morphology["C"], "concentration")
    concentration_error = _number(morphology["E_C"], "concentration error")
    asymmetry = _number(morphology["A"], "asymmetry")
    asymmetry_error = _number(morphology["E_A"], "asymmetry error")
    clumpiness = _number(morphology["S"], "clumpiness")
    clumpiness_error = _number(morphology["E_S"], "clumpiness error")
    ttype = int(_number(morphology["TType"], "TType"))
    unsure = int(_number(morphology["Unsure"], "Unsure"))
    edge_on = int(_number(morphology["Edge_on"], "Edge_on"))
    tidal = int(_number(morphology["Tidal"], "Tidal"))
    bar = _number(morphology["Bars"], "Bars")
    if unsure not in (0, 1) or edge_on not in (0, 1) or tidal not in (0, 1):
        raise GravityItem13RelaxationError("invalid morphology indicator")
    if not -0.5 <= bar <= 1.0:
        raise GravityItem13RelaxationError("invalid bar strength")
    plateifu = _decode(morphology["plateifu"])
    mangaid = _decode(morphology["MANGAID"])
    if plateifu != str(item12["plateifu"]) or mangaid.upper() != str(item12["mangaid"]).upper():
        raise GravityItem13RelaxationError("morphology and Item 12 identity mismatch")
    result = dict(item12)
    result.update(
        {
            "plateifu": plateifu,
            "mangaid": mangaid,
            "morphology_type": _decode(morphology["Type"]),
            "ttype": ttype,
            "ttype_normalized": _fixed(float(ttype), config, "ttype"),
            "unsure": unsure,
            "bar_strength": bar,
            "bar_normalized": _fixed(bar, config, "bar_strength"),
            "edge_on": edge_on,
            "tidal": tidal,
            "tidal_signed": float(2 * tidal - 1),
            "merger_unclassified": int(ttype == 11),
            "merger_unclassified_signed": float(2 * int(ttype == 11) - 1),
            "concentration": concentration,
            "concentration_error": concentration_error,
            "concentration_normalized": _fixed(concentration, config, "concentration"),
            "asymmetry": asymmetry,
            "asymmetry_error": asymmetry_error,
            "asymmetry_normalized": _fixed(asymmetry, config, "asymmetry"),
            "asymmetry_error_normalized": _fixed(asymmetry_error, config, "asymmetry_error"),
            "clumpiness": clumpiness,
            "clumpiness_error": clumpiness_error,
            "clumpiness_normalized": _fixed(clumpiness, config, "clumpiness"),
            "clumpiness_error_normalized": _fixed(clumpiness_error, config, "clumpiness_error"),
            "cas_flag": cas_flag,
            "prior_age_lead": _prior_age_component(item12, config),
        }
    )
    return result


def _serialize(row: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, (str, bool, int)):
            result[key] = value
        else:
            result[key] = _metric(float(value))
    return result


def write_predictor_source(root: Path) -> Path:
    root = root.resolve()
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem13RelaxationError("Item 13 scientific freeze is not bound")
    config = load_config(root)
    source_config = config["sources"]["morphology"]
    payload = _download(source_config["url"])
    if len(payload) != int(source_config["file_bytes"]):
        raise GravityItem13RelaxationError("morphology payload size changed")
    if hashlib.sha256(payload).hexdigest() != source_config["file_sha256"]:
        raise GravityItem13RelaxationError("morphology payload hash changed")
    official = _download(source_config["official_sha1_url"]).decode("utf-8").strip()
    if official.split()[0] != source_config["official_sha1"]:
        raise GravityItem13RelaxationError("official morphology SHA-1 changed")
    if hashlib.sha1(payload).hexdigest() != source_config["official_sha1"]:
        raise GravityItem13RelaxationError("morphology SHA-1 verification failed")
    raw_path = root / config["outputs"]["morphology_raw"]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_bytes(payload)

    from astropy.io import fits

    with fits.open(io.BytesIO(payload), memmap=False, lazy_load_hdus=False) as hdus:
        hdu = hdus[int(source_config["hdu"])]
        if int(hdu.header["NAXIS2"]) != int(source_config["observed_rows"]):
            raise GravityItem13RelaxationError("morphology row count changed")
        if list(hdu.columns.names) != list(source_config["observed_columns"]):
            raise GravityItem13RelaxationError("morphology schema changed")
        morphology_rows = [{name: row[name] for name in hdu.columns.names} for row in hdu.data]

    item12_binding = config["sources"]["item12_predictors"]
    item12_source = json.loads((root / item12_binding["path"]).read_text(encoding="utf-8"))
    _validate_content_hash(item12_source, "Item 12 predictor source")
    item12_by_plate = {str(row["plateifu"]): row for row in item12_source["records"]}
    records = []
    failures = []
    joined = 0
    for morphology in morphology_rows:
        plateifu = _decode(morphology["plateifu"])
        item12 = item12_by_plate.get(plateifu)
        if item12 is None:
            continue
        joined += 1
        try:
            records.append(_serialize(derive_predictors(morphology, item12, config)))
        except GravityItem13RelaxationError as exc:
            failures.append(
                {
                    "plateifu": plateifu,
                    "mangaid": _decode(morphology["MANGAID"]),
                    "reason": str(exc),
                }
            )
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item13-manga-predictor-source-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "morphology_url": source_config["url"],
            "morphology_file_sha256": hashlib.sha256(payload).hexdigest(),
            "morphology_official_sha1": hashlib.sha1(payload).hexdigest(),
            "morphology_raw_path": config["outputs"]["morphology_raw"],
            "item12_predictor_file_sha256": _sha256_file(root / item12_binding["path"]),
            "records": records,
            "failures": failures,
            "counts": {
                "morphology_rows": len(morphology_rows),
                "item12_predictor_rows": len(item12_source["records"]),
                "joined_rows": joined,
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
    _validate_content_hash(source, "Item 13 predictor source")
    config = load_config(root)
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem13RelaxationError("Item 13 predictor binding changed")
    raw_path = root / config["outputs"]["morphology_raw"]
    if _sha256_file(raw_path) != config["sources"]["morphology"]["file_sha256"]:
        raise GravityItem13RelaxationError("stored morphology payload changed")
    if int(source["counts"]["response_columns_requested"]) != 0:
        raise GravityItem13RelaxationError("response entered Item 13 predictor source")
    if int(source["counts"]["response_rows_read"]) != 0:
        raise GravityItem13RelaxationError("response rows entered Item 13 predictor source")


def _excluded_identities(root: Path, config: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    plates: set[str] = set()
    manga: set[str] = set()
    for entry in config["independence"]["identity_exclusions"]:
        value = json.loads((root / entry["path"]).read_text(encoding="utf-8"))
        for row in value[entry["objects_key"]]:
            plates.add(str(row[entry["plateifu_key"]]).strip().upper())
            manga.add(str(row[entry["mangaid_key"]]).strip().upper())
    return plates, manga


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
    excluded_plates, excluded_manga = _excluded_identities(root, config)
    coordinates = _coordinates(root, config)
    plate_counts = Counter(str(row["plateifu"]).upper() for row in source["records"])
    manga_counts = Counter(str(row["mangaid"]).upper() for row in source["records"])
    admitted = []
    exclusions: Counter[str] = Counter()
    for row in source["records"]:
        plateifu = str(row["plateifu"]).upper()
        mangaid = str(row["mangaid"]).upper()
        reasons = []
        if plateifu in excluded_plates or mangaid in excluded_manga:
            reasons.append("prior_manga_identity")
        if plate_counts[plateifu] != 1 or manga_counts[mangaid] != 1:
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
    cells: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in admitted:
        tidal = "tidal" if int(row["tidal"]) == 1 else "non_tidal"
        mass = "lower_mass" if float(row["log_stellar_mass"]) <= median_mass else "higher_mass"
        cells.setdefault((tidal, mass), []).append(row)
    cell_keys = sorted(cells)
    if len(cell_keys) != 4:
        raise GravityItem13RelaxationError("Item 13 target-blind strata changed")
    per_cell = int(config["sample"]["maximum_total_objects"]) // len(cell_keys)
    selected_by_cell = {
        key: sorted(
            cells[key],
            key=lambda row: (
                _split_hash(str(row["plateifu"]), config["sample"]["split_salt"]),
                row["plateifu"],
            ),
        )[:per_cell]
        for key in cell_keys
    }
    if any(len(rows) != per_cell for rows in selected_by_cell.values()):
        raise GravityItem13RelaxationError("morphology stratum cannot fill frozen sample")
    confirmation_per_cell = int(config["sample"]["confirmation_objects"]) // len(cell_keys)
    objects = []
    for key in cell_keys:
        rows = selected_by_cell[key]
        confirmation = {row["plateifu"] for row in rows[:confirmation_per_cell]}
        exploration = [row for row in rows if row["plateifu"] not in confirmation]
        exploration_fold = {
            row["plateifu"]: ordinal % int(config["evaluation"]["outer_folds"])
            for ordinal, row in enumerate(exploration)
        }
        for row in rows:
            output = dict(row)
            role = "reserved_confirmation" if row["plateifu"] in confirmation else "exploration"
            output.update(
                {
                    "tidal_bin": key[0],
                    "stellar_mass_bin": key[1],
                    "role": role,
                    "outer_fold": exploration_fold.get(
                        row["plateifu"],
                        int(
                            _split_hash(row["plateifu"], config["evaluation"]["fold_salt"])[:16], 16
                        )
                        % int(config["evaluation"]["outer_folds"]),
                    ),
                    "response_read": False,
                }
            )
            objects.append(output)
    objects.sort(key=lambda row: row["plateifu"])
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item13-manga-sample-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "predictor_source_sha256": _sha256_file(source_path),
            "predictor_source_content_sha256": source["content_sha256"],
            "stellar_mass_median": _metric(median_mass),
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
            "cell_counts": {
                "|".join(key): len(rows) for key, rows in sorted(selected_by_cell.items())
            },
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
    _validate_content_hash(sample, "Item 13 sample manifest")
    if sample != build_sample_manifest(root):
        raise GravityItem13RelaxationError("Item 13 sample manifest drifted")
    if sample["counts"]["predecessor_selected"] != 0:
        raise GravityItem13RelaxationError("predecessor entered Item 13 sample")
    if sample["counts"]["response_rows_read"] != 0:
        raise GravityItem13RelaxationError("response entered Item 13 sample")


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
        "modulation": random.integers(0, len(generator["modulations"]), count, dtype=np.int8),
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
            "schema_version": "invariant-gravity-item13-relaxation-candidates-1.0",
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
    _validate_content_hash(manifest, "Item 13 candidate manifest")
    if manifest != build_candidate_manifest(root):
        raise GravityItem13RelaxationError("Item 13 candidate manifest drifted")


def write_candidate_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["outputs"]["candidate_manifest"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_candidate_manifest(root)) + b"\n")
    return path


def _parse_skyserver_csv(payload: bytes) -> list[dict[str, str]]:
    lines = payload.decode("utf-8-sig", errors="strict").splitlines()
    lines = [line for line in lines if line.strip() and not line.startswith("#")]
    if not lines:
        raise GravityItem13RelaxationError("empty SDSS CSV table")
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    rows = [
        {str(key): "" if value is None else str(value).strip() for key, value in row.items()}
        for row in reader
    ]
    if reader.fieldnames == ["error_message"]:
        message = rows[0]["error_message"] if rows else "unknown SkyServer error"
        raise GravityItem13RelaxationError(message)
    return rows


def _sql_query(config: Mapping[str, Any], query: str) -> bytes:
    parameters = urllib.parse.urlencode({"cmd": query, "format": "csv"})
    endpoint = config["sources"]["response"]["skyserver_endpoint"]
    request = urllib.request.Request(
        f"{endpoint}?{parameters}", headers={"User-Agent": "Invariant/Item13-MaNGA"}
    )
    with urllib.request.urlopen(request, timeout=180) as response:
        payload = response.read()
    if not payload:
        raise GravityItem13RelaxationError("empty SkyServer response")
    return payload


def _response_query(config: Mapping[str, Any], identities: Sequence[str]) -> str:
    source = config["sources"]["response"]
    columns = ",".join(f"d.{column}" for column in source["columns"])
    quoted = ",".join("'" + value.replace("'", "''") + "'" for value in identities)
    return (
        f"SELECT {columns} FROM {source['dap_table']} d "
        f"WHERE d.daptype='{source['daptype']}' AND d.plateifu IN ({quoted}) "
        "ORDER BY d.plateifu"
    )


def write_response_source(root: Path) -> Path:
    root = root.resolve()
    if SAMPLE_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem13RelaxationError("Item 13 sample freeze is not bound")
    config = load_config(root)
    sample_path = root / config["outputs"]["sample_manifest"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, root)
    exploration = [row["plateifu"] for row in sample["objects"] if row["role"] == "exploration"]
    chunk_size = int(config["sources"]["response"]["chunk_size"])
    records = []
    chunks = []
    for begin in range(0, len(exploration), chunk_size):
        identities = exploration[begin : begin + chunk_size]
        query = _response_query(config, identities)
        payload = _sql_query(config, query)
        rows = _parse_skyserver_csv(payload)
        if {row["plateifu"] for row in rows} != set(identities):
            raise GravityItem13RelaxationError("MaNGA response chunk scope changed")
        records.extend(rows)
        chunks.append(
            {
                "ordinal": len(chunks),
                "identity_count": len(identities),
                "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                "payload_sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
    expected = set(config["sources"]["response"]["columns"])
    if any(set(row) != expected for row in records):
        raise GravityItem13RelaxationError("MaNGA response schema changed")
    row_ids = [row["plateifu"] for row in records]
    if set(row_ids) != set(exploration) or len(row_ids) != len(set(row_ids)):
        raise GravityItem13RelaxationError("MaNGA response scope changed")
    confirmation = {
        row["plateifu"] for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if confirmation & set(row_ids):
        raise GravityItem13RelaxationError("Item 13 confirmation response entered query")
    manifest = _content_hashed(
        {
            "schema_version": "invariant-gravity-item13-manga-response-source-1.0",
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
    _validate_content_hash(source, "Item 13 response source")
    if source["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem13RelaxationError("Item 13 scientific response binding changed")
    if source["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem13RelaxationError("Item 13 sample response binding changed")
    if source["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem13RelaxationError("Item 13 confirmation opened")
    if source["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem13RelaxationError("post-response formula entered Item 13")


def _finite_response(row: Mapping[str, str], key: str) -> float:
    return _number(str(row.get(key, "")).strip(), key)


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
            sigma = _finite_response(raw, "stellar_sigma_1re")
            rchi2 = _finite_response(raw, "stellar_rchi2_1re")
            stellar_lo = _finite_response(raw, "stellar_vel_lo_clip")
            stellar_hi = _finite_response(raw, "stellar_vel_hi_clip")
            ha_lo = _finite_response(raw, "ha_gvel_lo_clip")
            ha_hi = _finite_response(raw, "ha_gvel_hi_clip")
        except GravityItem13RelaxationError as exc:
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
            "tidal_bin": sample_row["tidal_bin"],
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
            "schema_version": "invariant-gravity-item13-manga-extraction-1.0",
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": "PASS_ITEM13_MANGA_QUALITY"
            if quality_pass
            else "FAIL_ITEM13_MANGA_QUALITY",
            "rows": rows,
            "failures": failures,
            "counts": {
                "exploration_response_rows": selected,
                "quality_passing_galaxies": passing,
                "quality_failed_galaxies": selected - passing,
                "quality_retention_fraction": _metric(retention),
                "predecessor_selected": sample["counts"]["predecessor_selected"],
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
    _validate_content_hash(summary, "Item 13 extraction summary")
    validate_candidate_manifest(candidates, root)
    rows = summary["rows"]
    if not rows:
        raise GravityItem13RelaxationError("Item 13 has no quality rows")
    structural = np.column_stack(
        [
            np.asarray([float(row[field]) for row in rows])
            for field in config["evaluation"]["structural_features"]
        ]
    )
    prior_age = np.asarray([float(row["prior_age_lead"]) for row in rows])
    modulation_normalization = config["evaluation"]["fixed_modulation_normalization"]

    def modulation(field: str, values: np.ndarray) -> np.ndarray:
        center, scale = (float(part) for part in modulation_normalization[field])
        if scale <= 0:
            raise GravityItem13RelaxationError("fixed modulation normalization changed")
        return (values - center) / scale

    return {
        "summary": summary,
        "candidate_manifest": candidates,
        "rows": rows,
        "folds": np.asarray([int(row["outer_fold"]) for row in rows]),
        "y": np.log10(np.asarray([float(row["stellar_sigma_1re_km_s"]) for row in rows])),
        "y_span": np.log10(np.asarray([float(row["stellar_velocity_span_km_s"]) for row in rows])),
        "design_structural": structural,
        "design_age": np.column_stack((structural, prior_age)),
        "prior_age": prior_age,
        "tidal": np.asarray([float(row["tidal_signed"]) for row in rows]),
        "merger": np.asarray([float(row["merger_unclassified_signed"]) for row in rows]),
        "asymmetry": np.asarray([float(row["asymmetry_normalized"]) for row in rows]),
        "asymmetry_error": np.asarray([float(row["asymmetry_error_normalized"]) for row in rows]),
        "clumpiness": np.asarray([float(row["clumpiness_normalized"]) for row in rows]),
        "clumpiness_error": np.asarray([float(row["clumpiness_error_normalized"]) for row in rows]),
        "bar": np.asarray([float(row["bar_normalized"]) for row in rows]),
        "surface": modulation(
            "stellar_surface_density",
            np.asarray([float(row["log_surface_density"]) for row in rows]),
        ),
        "mass_modulation": modulation(
            "stellar_mass",
            np.asarray([float(row["log_stellar_mass"]) for row in rows]),
        ),
        "axis_modulation": modulation(
            "axis_ratio", np.asarray([float(row["axis_ratio"]) for row in rows])
        ),
        "mass": np.asarray([float(row["log_stellar_mass"]) for row in rows]),
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
    tidal = xp.asarray(data["tidal"], dtype=xp.float64)[None, :]
    merger = xp.asarray(data["merger"], dtype=xp.float64)[None, :]
    asymmetry = xp.asarray(data["asymmetry"], dtype=xp.float64)[None, :]
    asymmetry_error = xp.asarray(data["asymmetry_error"], dtype=xp.float64)[None, :]
    clumpiness = xp.asarray(data["clumpiness"], dtype=xp.float64)[None, :]
    bar = xp.asarray(data["bar"], dtype=xp.float64)[None, :]

    def signed_power(value: Any) -> Any:
        z = (value - threshold) / scale
        magnitude = xp.abs(z) ** power
        return xp.sign(z) * magnitude / (1.0 + magnitude)

    tidal_term = signed_power(tidal)
    asymmetry_term = signed_power(asymmetry)
    clumpiness_term = signed_power(clumpiness)
    significance = signed_power(asymmetry / (1.0 + xp.abs(asymmetry_error)))
    coherence = xp.tanh((asymmetry * clumpiness - threshold) / scale)
    tidal_asymmetry = tidal * asymmetry_term
    merger_term = signed_power(merger)
    consensus = signed_power((tidal + merger + asymmetry + clumpiness) / 4.0)
    relaxation = -xp.tanh((asymmetry - threshold) / scale) * xp.exp(
        -((xp.abs(asymmetry - threshold) / scale) ** power)
    )
    hysteresis = xp.tanh(((asymmetry - threshold) * (clumpiness - threshold)) / scale)
    bar_coupling = bar * asymmetry_term
    log_periodic = asymmetry_term * xp.cos(phase + power * xp.log1p(xp.abs(asymmetry - threshold)))
    component = xp.zeros((end - begin, asymmetry.shape[1]), dtype=xp.float64)
    for index, value in enumerate(
        (
            tidal_term,
            asymmetry_term,
            clumpiness_term,
            significance,
            coherence,
            tidal_asymmetry,
            merger_term,
            consensus,
            relaxation,
            hysteresis,
            bar_coupling,
            log_periodic,
        )
    ):
        component = xp.where(family == index, value, component)
    modulations = (
        xp.ones_like(component),
        xp.tanh(xp.asarray(data["surface"], dtype=xp.float64))[None, :],
        xp.tanh(xp.asarray(data["mass_modulation"], dtype=xp.float64))[None, :],
        xp.tanh(xp.asarray(data["axis_modulation"], dtype=xp.float64))[None, :],
        xp.asarray(data["prior_age"], dtype=xp.float64)[None, :],
    )
    modulation = xp.ones_like(component)
    for index, value in enumerate(modulations):
        modulation = xp.where(modulation_index == index, value, modulation)
    result = component * modulation
    return xp.clip(xp.nan_to_num(result, nan=0.0, posinf=1e6, neginf=-1e6), -1e6, 1e6)


def _fit_component(
    component: np.ndarray, residual: np.ndarray, ridge: float
) -> tuple[float, float, float]:
    mean = float(np.mean(component))
    scale = max(float(np.std(component)), 1e-12)
    standardized = (component - mean) / scale
    coefficient = float(np.sum(standardized * residual) / (np.sum(standardized**2) + ridge))
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
    count = len(arrays["family"])
    folds = np.asarray(data["folds"])
    y = np.asarray(data["y"])
    y_span = np.asarray(data["y_span"])
    structural = np.asarray(data["design_structural"])
    age_design = np.asarray(data["design_age"])
    predictions = {
        key: np.full(len(y), np.nan)
        for key in (
            "structural",
            "age",
            "full",
            "disturbance_only",
            "span_age",
            "span_full",
        )
    }
    records = []
    batch_size = int(config["evaluation"]["candidate_batch_size"])
    alpha = float(config["evaluation"]["ridge_alpha"])
    coefficient_ridge = float(config["evaluation"]["disturbance_coefficient_ridge"])
    outer_folds = int(config["evaluation"]["outer_folds"])
    component_crosscheck = 0.0
    for outer in range(outer_folds):
        inner_records = []
        for inner in [value for value in range(outer_folds) if value != outer]:
            train = (folds != outer) & (folds != inner)
            validation = folds == inner
            model = _ridge_fit(age_design[train], y[train], alpha)
            inner_records.append(
                {
                    "train": train,
                    "validation": validation,
                    "train_residual": y[train] - _ridge_predict(model, age_design[train]),
                    "validation_residual": y[validation]
                    - _ridge_predict(model, age_design[validation]),
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
        structural_model = _ridge_fit(structural[train], y[train], alpha)
        age_model = _ridge_fit(age_design[train], y[train], alpha)
        structural_train = _ridge_predict(structural_model, structural[train])
        age_train = _ridge_predict(age_model, age_design[train])
        predictions["structural"][test] = _ridge_predict(structural_model, structural[test])
        predictions["age"][test] = _ridge_predict(age_model, age_design[test])
        selected_component = _candidate_components(arrays, data, selected, selected + 1, np)[0]
        mean_age, scale_age, coefficient_age = _fit_component(
            selected_component[train], y[train] - age_train, coefficient_ridge
        )
        predictions["full"][test] = (
            predictions["age"][test]
            + coefficient_age * (selected_component[test] - mean_age) / scale_age
        )
        mean_structural, scale_structural, coefficient_structural = _fit_component(
            selected_component[train], y[train] - structural_train, coefficient_ridge
        )
        predictions["disturbance_only"][test] = predictions["structural"][test] + (
            coefficient_structural * (selected_component[test] - mean_structural) / scale_structural
        )
        span_model = _ridge_fit(age_design[train], y_span[train], alpha)
        span_train = _ridge_predict(span_model, age_design[train])
        predictions["span_age"][test] = _ridge_predict(span_model, age_design[test])
        mean_span, scale_span, coefficient_span = _fit_component(
            selected_component[train], y_span[train] - span_train, coefficient_ridge
        )
        predictions["span_full"][test] = (
            predictions["span_age"][test]
            + coefficient_span * (selected_component[test] - mean_span) / scale_span
        )
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
                "modulation": config["candidate_generator"]["modulations"][
                    int(arrays["modulation"][selected])
                ],
                "inner_mse": _metric(scores[selected]),
                "fitted_disturbance_coefficient_after_age": _metric(coefficient_age),
                "fitted_disturbance_coefficient_without_age": _metric(coefficient_structural),
                "fitted_span_coefficient": _metric(coefficient_span),
                "test_galaxies": int(np.sum(test)),
            }
        )
    if any(np.any(~np.isfinite(value)) for value in predictions.values()):
        raise GravityItem13RelaxationError("Item 13 OOF prediction incomplete")
    if backend == "gpu_cupy":
        xp.cuda.Device().synchronize()
    elapsed = time.perf_counter() - started
    return (
        predictions,
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


def _paired_sign_flip(
    differences: np.ndarray, config: Mapping[str, Any], salt_key: str
) -> dict[str, Any]:
    count = int(config["evaluation"]["paired_sign_flip_permutations"])
    salt = str(config["evaluation"][salt_key])
    seed = int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
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
    predictions, folds, compute = _nested_select(data, config)
    metrics = {
        key: _metrics(data["y"], value)
        for key, value in predictions.items()
        if not key.startswith("span_")
    }
    span_metrics = {
        key: _metrics(data["y_span"], predictions[key]) for key in ("span_age", "span_full")
    }
    structural_mse = float(metrics["structural"]["mse"])
    age_mse = float(metrics["age"]["mse"])
    full_mse = float(metrics["full"]["mse"])
    disturbance_only_mse = float(metrics["disturbance_only"]["mse"])
    disturbance_relative = (age_mse - full_mse) / age_mse
    age_replication_relative = (structural_mse - age_mse) / structural_mse
    age_persistence_relative = (disturbance_only_mse - full_mse) / disturbance_only_mse
    disturbance_paired = _paired_sign_flip(
        (data["y"] - predictions["age"]) ** 2 - (data["y"] - predictions["full"]) ** 2,
        config,
        "permutation_salt_disturbance",
    )
    age_replication_paired = _paired_sign_flip(
        (data["y"] - predictions["structural"]) ** 2 - (data["y"] - predictions["age"]) ** 2,
        config,
        "permutation_salt_age_replication",
    )
    age_persistence_paired = _paired_sign_flip(
        (data["y"] - predictions["disturbance_only"]) ** 2 - (data["y"] - predictions["full"]) ** 2,
        config,
        "permutation_salt_age_persistence",
    )
    dimensions = {
        "tidal_state": data["tidal"],
        "stellar_mass_half": data["mass"],
        "prior_age_half": data["prior_age"],
    }
    strata = []
    stratum_pass = {}
    for dimension, values in dimensions.items():
        if dimension == "tidal_state":
            split = 0.0
        else:
            split = float(np.median(values))
        gains = []
        for label, mask in (("low", values <= split), ("high", values > split)):
            base = float(np.mean((data["y"][mask] - predictions["age"][mask]) ** 2))
            proposed = float(np.mean((data["y"][mask] - predictions["full"][mask]) ** 2))
            gains.append(base - proposed)
            strata.append(
                {
                    "dimension": dimension,
                    "stratum": label,
                    "galaxies": int(np.sum(mask)),
                    "age_baseline_mse": _metric(base),
                    "full_model_mse": _metric(proposed),
                    "disturbance_mse_gain": _metric(base - proposed),
                }
            )
        stratum_pass[dimension] = all(value > 0 for value in gains)
    summary = data["summary"]
    gates = {
        "quality_count_and_fraction_pass": summary["decision"] == "PASS_ITEM13_MANGA_QUALITY",
        "fresh_identity_and_confirmation_boundary_pass": summary["counts"]["predecessor_selected"]
        == 0
        and summary["counts"]["confirmation_response_rows"] == 0,
        "candidate_count_exact": compute["candidate_cells"] == 262144,
        "full_model_r2_positive": float(metrics["full"]["r2"]) > 0,
        "disturbance_beats_age_baseline": full_mse < age_mse,
        "disturbance_relative_mse_improvement_at_least": disturbance_relative
        >= float(config["admission"]["disturbance_relative_mse_improvement_at_least"]),
        "disturbance_paired_p_at_most": float(disturbance_paired["p_value"])
        <= float(config["admission"]["disturbance_paired_p_at_most"]),
        "disturbance_gain_positive_in_both_tidal_states": stratum_pass["tidal_state"],
        "disturbance_gain_positive_in_both_stellar_mass_halves": stratum_pass["stellar_mass_half"],
        "disturbance_gain_positive_in_both_prior_age_halves": stratum_pass["prior_age_half"],
        "item12_age_lead_beats_structural_baseline": age_mse < structural_mse,
        "item12_age_replication_p_at_most": float(age_replication_paired["p_value"])
        <= float(config["admission"]["item12_age_replication_p_at_most"]),
        "item12_age_persists_after_disturbance": full_mse < disturbance_only_mse,
        "item12_age_persistence_p_at_most": float(age_persistence_paired["p_value"])
        <= float(config["admission"]["item12_age_persistence_p_at_most"]),
        "selected_family_is_disturbance_dependent": True,
        "post_response_formula_generation_zero": True,
    }
    decision = (
        "PASS_ITEM13_MANGA_RELAXATION_AND_MERGERS_EXPLORATION"
        if all(gates.values())
        else "REJECT_ITEM13_MANGA_RELAXATION_AND_MERGERS_EXPLORATION"
    )
    if not gates["quality_count_and_fraction_pass"]:
        decision = "INCONCLUSIVE_ITEM13_MANGA_QUALITY"
    paths = {
        key: root / config["outputs"][key]
        for key in (
            "morphology_raw",
            "predictor_source",
            "sample_manifest",
            "candidate_manifest",
            "response_source",
            "extraction_summary",
        )
    }
    return _content_hashed(
        {
            "schema_version": "invariant-gravity-item13-manga-relaxation-result-1.0",
            "goal": config["goal"],
            "item_number": 13,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "sample_freeze_commit": SAMPLE_FREEZE_COMMIT,
            "decision": decision,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "counts": {
                "candidate_cells": 262144,
                "quality_passing_galaxies": summary["counts"]["quality_passing_galaxies"],
                "quality_failed_galaxies": summary["counts"]["quality_failed_galaxies"],
                "confirmation_response_rows": 0,
                "post_response_formula_cells": 0,
                "paid_model_calls": 0,
            },
            "inputs": {key + "_sha256": _sha256_file(path) for key, path in paths.items()},
            "primary": {
                "structural_baseline": metrics["structural"],
                "item12_age_baseline": metrics["age"],
                "selected_disturbance_full_model": metrics["full"],
                "selected_disturbance_without_age": metrics["disturbance_only"],
                "disturbance_relative_mse_improvement": _metric(disturbance_relative),
                "item12_age_replication_relative_mse_improvement": _metric(
                    age_replication_relative
                ),
                "item12_age_persistence_relative_mse_improvement": _metric(
                    age_persistence_relative
                ),
                "outer_fold_selections": folds,
            },
            "secondary_stellar_velocity_span": {
                "age_baseline": span_metrics["span_age"],
                "selected_disturbance_full_model": span_metrics["span_full"],
                "relative_mse_improvement": _metric(
                    (
                        float(span_metrics["span_age"]["mse"])
                        - float(span_metrics["span_full"]["mse"])
                    )
                    / float(span_metrics["span_age"]["mse"])
                ),
                "candidate_reselection": False,
            },
            "compute": compute,
            "paired_sign_flip": {
                "disturbance_after_age": disturbance_paired,
                "item12_age_replication": age_replication_paired,
                "item12_age_persistence_after_disturbance": age_persistence_paired,
            },
            "strata": strata,
            "gate_checks": gates,
            "gate_counts": {
                "passed": sum(bool(value) for value in gates.values()),
                "required": len(gates),
            },
            "limitations": {
                "imaging_disturbance_is_causal_history": False,
                "visual_tidal_indicator_is_complete_merger_stage": False,
                "item12_age_cells_were_derived_on_a_prior_exploration": True,
                "integrated_stellar_dynamics_only": True,
                "historical_novelty_adjudicated": False,
            },
            "claims": config["claim_boundaries"],
        }
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    _validate_content_hash(receipt, "Item 13 result receipt")
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem13RelaxationError("Item 13 result scientific binding changed")
    if receipt["sample_freeze_commit"] != SAMPLE_FREEZE_COMMIT:
        raise GravityItem13RelaxationError("Item 13 result sample binding changed")
    if receipt["counts"]["candidate_cells"] != 262144:
        raise GravityItem13RelaxationError("Item 13 candidate count changed")
    if receipt["counts"]["confirmation_response_rows"] != 0:
        raise GravityItem13RelaxationError("Item 13 confirmation entered result")
    if receipt["counts"]["post_response_formula_cells"] != 0:
        raise GravityItem13RelaxationError("post-response formula entered Item 13")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem13RelaxationError("Item 13 result contains an overclaim")


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
        raise GravityItem13RelaxationError("Item 13 result receipt drifted")


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
