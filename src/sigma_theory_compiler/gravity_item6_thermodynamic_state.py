"""Frozen ACCEPT/HeCS thermodynamic-state experiment for gravity-roadmap Item 6."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import gravity_item5_pressure_cross_support as cvcore
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item6_thermodynamic_accept_hecs_v1.json"
SCIENTIFIC_FREEZE_COMMIT = "4c318784364b222927a2c2484c1d0c0e195ff94a"
VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class GravityItem6ThermodynamicStateError(RuntimeError):
    """Raised when the frozen Item 6 boundary or result drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem6ThermodynamicStateError("non-finite metric")
    return f"{float(value):.12e}"


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = cvcore._canonicalize_floats(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != ("invariant-gravity-roadmap-item6-thermodynamic-config-1.0"):
        raise GravityItem6ThermodynamicStateError("unexpected Item 6 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem6ThermodynamicStateError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem6ThermodynamicStateError("Item 5 synthesis file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem6ThermodynamicStateError("Item 5 synthesis content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem6ThermodynamicStateError("Item 5 did not authorize Item 6")
    dependency = config["implementation_dependency"]
    if _sha256_file(root / dependency["path"]) != dependency["file_sha256"]:
        raise GravityItem6ThermodynamicStateError("nested-CV dependency changed")

    authorization = config["authorization"]
    forbidden_true = (
        "paid_model_calls_allowed",
        "reserved_confirmation_velocity_responses_allowed",
        "hecs_mass_profiles_allowed",
        "caustic_or_nfw_mass_allowed_as_predictor",
        "lensing_mass_allowed_as_predictor",
    )
    if any(bool(authorization[name]) for name in forbidden_true):
        raise GravityItem6ThermodynamicStateError("Item 6 authorization boundary changed")
    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    if len(exploration) != sample["exploration_count"] or len(set(exploration)) != len(exploration):
        raise GravityItem6ThermodynamicStateError("exploration sample changed")
    if len(confirmation) != sample["reserved_confirmation_count"] or len(set(confirmation)) != len(
        confirmation
    ):
        raise GravityItem6ThermodynamicStateError("confirmation sample changed")
    if set(exploration).intersection(confirmation):
        raise GravityItem6ThermodynamicStateError("sample roles overlap")
    if set(sample["accept_name_map"]) != set(exploration).union(confirmation):
        raise GravityItem6ThermodynamicStateError("ACCEPT name mapping changed")
    if config["prefreeze_audit"]["hecs_velocity_response_values_read"] != 0:
        raise GravityItem6ThermodynamicStateError("prefreeze response boundary changed")
    if config["derivation"]["feature_builder_accepts_velocity_response"]:
        raise GravityItem6ThermodynamicStateError("feature builder cannot accept response")
    return config


def build_sample_manifest(root: Path) -> dict[str, Any]:
    config = load_config(root)
    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    assignments = cvcore.assign_folds(
        exploration,
        salt=str(config["cross_validation"]["fold_salt"]),
        folds=int(config["cross_validation"]["outer_folds"]),
    )
    salt = str(sample["selection_salt"])
    objects: list[dict[str, Any]] = []
    for role, names in (
        ("exploration", exploration),
        ("reserved_confirmation", confirmation),
    ):
        for name in names:
            objects.append(
                {
                    "hecs_name": name,
                    "accept_name": sample["accept_name_map"][name],
                    "role": role,
                    "outer_fold": assignments.get(name),
                    "selection_digest": hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
                }
            )
    objects.sort(key=lambda row: (str(row["role"]), str(row["hecs_name"])))
    return _seal(
        {
            "schema_version": "invariant-gravity-item6-thermodynamic-sample-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM6_TARGET_BLIND_ACCEPT_HECS_SAMPLE",
            "selection": {
                "matching_rule": sample["matching_rule"],
                "quality_rule": sample["quality_rule"],
                "cool_core_rule": sample["cool_core_rule"],
                "salt": salt,
                "selection_used_velocity_response": False,
            },
            "counts": {
                "quality_passing_candidates": sample["quality_passing_candidates"],
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
                "exploration_cool_core": sample["exploration_cool_core_count"],
                "confirmation_cool_core": sample["confirmation_cool_core_count"],
            },
            "objects": objects,
            "prefreeze_boundary": {
                "thermodynamic_predictors_audited_for_candidates": sample[
                    "quality_passing_candidates"
                ],
                "reserved_confirmation_predictors_blinded": False,
                "hecs_velocity_response_values_read": 0,
                "reserved_confirmation_velocity_responses_blinded": True,
                "mass_profile_values_used": 0,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem6ThermodynamicStateError("sample content hash changed")
    expected = {
        "quality_passing_candidates": config["sample"]["quality_passing_candidates"],
        "exploration": config["sample"]["exploration_count"],
        "reserved_confirmation": config["sample"]["reserved_confirmation_count"],
        "exploration_cool_core": config["sample"]["exploration_cool_core_count"],
        "confirmation_cool_core": config["sample"]["confirmation_cool_core_count"],
    }
    if manifest["counts"] != expected:
        raise GravityItem6ThermodynamicStateError("sample counts changed")
    roles = {"exploration": set(), "reserved_confirmation": set()}
    for row in manifest["objects"]:
        roles[str(row["role"])].add(str(row["hecs_name"]))
    if roles["exploration"] != set(config["sample"]["exploration"]):
        raise GravityItem6ThermodynamicStateError("exploration identities changed")
    if roles["reserved_confirmation"] != set(config["sample"]["reserved_confirmation"]):
        raise GravityItem6ThermodynamicStateError("confirmation identities changed")
    boundary = manifest["prefreeze_boundary"]
    if boundary["hecs_velocity_response_values_read"] != 0:
        raise GravityItem6ThermodynamicStateError("sample opened velocity responses")
    if not boundary["reserved_confirmation_velocity_responses_blinded"]:
        raise GravityItem6ThermodynamicStateError("confirmation response boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem6ThermodynamicStateError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def measure_thermodynamic_features(
    *,
    temperature_kev: float,
    core_entropy_kev_cm2: float,
    entropy_100_kev_cm2: float,
    entropy_slope: float,
    xray_luminosity_1e43_erg_s: float,
    redshift: float,
    member_count: int,
) -> dict[str, float]:
    """Build target-blind thermodynamic-state features."""

    values = (
        temperature_kev,
        core_entropy_kev_cm2,
        entropy_100_kev_cm2,
        entropy_slope,
        xray_luminosity_1e43_erg_s,
        redshift,
        float(member_count),
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise GravityItem6ThermodynamicStateError("non-finite thermodynamic observable")
    if (
        temperature_kev <= 0
        or core_entropy_kev_cm2 < 0
        or entropy_100_kev_cm2 <= 0
        or entropy_slope <= 0
        or xray_luminosity_1e43_erg_s <= 0
        or redshift <= 0
        or member_count <= 0
    ):
        raise GravityItem6ThermodynamicStateError("invalid thermodynamic observable")
    log_temperature = math.log10(temperature_kev)
    log_core_entropy = math.log10(1.0 + core_entropy_kev_cm2)
    log_entropy_100 = math.log10(entropy_100_kev_cm2)
    entropy_contrast = log_core_entropy - log_entropy_100
    cooling = 1.5 * log_core_entropy - log_temperature
    pressure = 2.5 * log_temperature - 1.5 * log_core_entropy
    gradient = entropy_slope * (log_entropy_100 - log_core_entropy)
    result = {
        "log_temperature": log_temperature,
        "log_lx": math.log10(xray_luminosity_1e43_erg_s),
        "log1pz": math.log10(1.0 + redshift),
        "log_member_count": math.log10(member_count),
        "log_core_entropy": log_core_entropy,
        "log_entropy_100": log_entropy_100,
        "entropy_slope": entropy_slope,
        "entropy_contrast": entropy_contrast,
        "cooling_time_proxy": cooling,
        "core_pressure_proxy": pressure,
        "entropy_gradient_proxy": gradient,
        "cooling_x_slope": cooling * entropy_slope,
        "pressure_x_slope": pressure * entropy_slope,
        "entropy_contrast_x_temperature": entropy_contrast * log_temperature,
        "gradient_x_temperature": gradient * log_temperature,
        "cooling_squared": cooling**2,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise GravityItem6ThermodynamicStateError("non-finite derived feature")
    return result


FEATURE_NAMES = (
    "log_temperature",
    "log_lx",
    "log1pz",
    "log_member_count",
    "log_core_entropy",
    "log_entropy_100",
    "entropy_slope",
    "entropy_contrast",
    "cooling_time_proxy",
    "core_pressure_proxy",
    "entropy_gradient_proxy",
    "cooling_x_slope",
    "pressure_x_slope",
    "entropy_contrast_x_temperature",
    "gradient_x_temperature",
    "cooling_squared",
)


def _query_url(
    catalog_id: str,
    *,
    columns: Sequence[str],
    constraint_name: str,
    constraint_value: str,
) -> str:
    parameters: list[tuple[str, str]] = [
        ("-source", catalog_id),
        (constraint_name, constraint_value),
    ]
    parameters.extend(("-out", str(column)) for column in columns)
    parameters.append(("-out.max", "100"))
    return f"{VIZIER_ENDPOINT}?{urllib.parse.urlencode(parameters)}"


def _fetch(url: str, *, attempts: int = 3) -> bytes:
    error: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                return response.read()
        except (OSError, TimeoutError, urllib.error.URLError) as exc:  # pragma: no cover
            error = exc
            if attempt + 1 < attempts:
                time.sleep(1.0 + attempt)
    raise GravityItem6ThermodynamicStateError(f"VizieR acquisition failed: {url}") from error


def _tsv_rows(payload: bytes, *, first_field: str, fields: int) -> list[list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem6ThermodynamicStateError("VizieR response is not UTF-8") from exc
    rows = [
        [field.strip() for field in line.split("\t")]
        for line in lines
        if line.split("\t", 1)[0].strip() == first_field
    ]
    if any(len(row) != fields for row in rows):
        raise GravityItem6ThermodynamicStateError("VizieR schema changed")
    return rows


def parse_temperature_payload(payload: bytes, *, accept_name: str) -> list[dict[str, Any]]:
    rows = _tsv_rows(payload, first_field=accept_name, fields=5)
    result = []
    for row in rows:
        if not row[3]:
            continue
        result.append(
            {
                "accept_name": accept_name,
                "obsid": int(row[1]),
                "redshift": float(row[2]),
                "temperature_kev": float(row[3]),
                "notes": row[4],
            }
        )
    if not result:
        raise GravityItem6ThermodynamicStateError("no positive ACCEPT temperature rows")
    return result


def parse_entropy_payload(payload: bytes, *, accept_name: str) -> list[dict[str, Any]]:
    rows = _tsv_rows(payload, first_field=accept_name, fields=10)
    result = []
    for row in rows:
        if row[1] not in {"flat", "extr"} or not all(row[index] for index in range(2, 9)):
            continue
        result.append(
            {
                "accept_name": accept_name,
                "method": row[1],
                "n_bins": int(row[2]),
                "core_entropy_kev_cm2": float(row[3]),
                "core_entropy_error_kev_cm2": float(row[4]),
                "entropy_100_kev_cm2": float(row[5]),
                "entropy_100_error_kev_cm2": float(row[6]),
                "entropy_slope": float(row[7]),
                "entropy_slope_error": float(row[8]),
                "p_value": row[9],
            }
        )
    if not result:
        raise GravityItem6ThermodynamicStateError("no primary ACCEPT entropy rows")
    return result


def parse_hecs_metadata_payload(payload: bytes, *, hecs_name: str) -> dict[str, Any]:
    rows = _tsv_rows(payload, first_field=hecs_name, fields=7)
    if len(rows) != 1:
        raise GravityItem6ThermodynamicStateError("unexpected HeCS metadata rows")
    row = rows[0]
    return {
        "hecs_name": hecs_name,
        "ra_deg": float(row[1]),
        "dec_deg": float(row[2]),
        "redshift": float(row[3]),
        "xray_luminosity_1e43_erg_s": float(row[4]),
        "catalog": row[5],
        "member_count": int(row[6]),
    }


def parse_hecs_response_payload(payload: bytes, *, hecs_name: str) -> dict[str, Any]:
    rows = _tsv_rows(payload, first_field=hecs_name, fields=4)
    if len(rows) != 1:
        raise GravityItem6ThermodynamicStateError("unexpected HeCS response rows")
    row = rows[0]
    return {
        "hecs_name": hecs_name,
        "sigma_km_s": float(row[1]),
        "sigma_upper_error_km_s": float(row[2]),
        "sigma_lower_error_km_s": float(row[3]),
    }


def _retrieval(url: str, payload: bytes) -> dict[str, Any]:
    return {"url": url, "payload_sha256": _sha256_bytes(payload), "bytes": len(payload)}


def acquire_exploration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem6ThermodynamicStateError(
            "scientific freeze commit is not bound; response access forbidden"
        )
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    objects = sorted(
        (row for row in sample["objects"] if row["role"] == "exploration"),
        key=lambda row: str(row["hecs_name"]),
    )
    if {str(row["hecs_name"]) for row in objects} != set(config["sample"]["exploration"]):
        raise GravityItem6ThermodynamicStateError("exploration acquisition scope changed")
    sources = config["sources"]
    records: list[dict[str, Any]] = []
    for obj in objects:
        hecs_name = str(obj["hecs_name"])
        accept_name = str(obj["accept_name"])
        temperature_url = _query_url(
            str(sources["thermodynamic_table"]["catalog_id"]),
            columns=sources["thermodynamic_table"]["allowed_columns"],
            constraint_name="Name",
            constraint_value=accept_name,
        )
        entropy_url = _query_url(
            str(sources["entropy_fit_table"]["catalog_id"]),
            columns=sources["entropy_fit_table"]["allowed_columns"],
            constraint_name="Name",
            constraint_value=accept_name,
        )
        metadata_url = _query_url(
            str(sources["dynamics_table"]["catalog_id"]),
            columns=sources["dynamics_table"]["metadata_columns"],
            constraint_name="Name",
            constraint_value=hecs_name,
        )
        response_url = _query_url(
            str(sources["dynamics_table"]["catalog_id"]),
            columns=sources["dynamics_table"]["response_columns"],
            constraint_name="Name",
            constraint_value=hecs_name,
        )
        temperature_payload = _fetch(temperature_url)
        entropy_payload = _fetch(entropy_url)
        metadata_payload = _fetch(metadata_url)
        response_payload = _fetch(response_url)
        records.append(
            {
                "hecs_name": hecs_name,
                "accept_name": accept_name,
                "temperature_rows": parse_temperature_payload(
                    temperature_payload, accept_name=accept_name
                ),
                "entropy_rows": parse_entropy_payload(entropy_payload, accept_name=accept_name),
                "metadata": parse_hecs_metadata_payload(metadata_payload, hecs_name=hecs_name),
                "primary_response": parse_hecs_response_payload(
                    response_payload, hecs_name=hecs_name
                ),
                "retrievals": {
                    "temperature": _retrieval(temperature_url, temperature_payload),
                    "entropy": _retrieval(entropy_url, entropy_payload),
                    "metadata": _retrieval(metadata_url, metadata_payload),
                    "primary_response": _retrieval(response_url, response_payload),
                },
            }
        )
    return _seal(
        {
            "schema_version": "invariant-gravity-item6-thermodynamic-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM6_EXPLORATION_SOURCE_ACQUISITION",
            "preregistration": {
                "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_path": config["sample_manifest_output"],
                "sample_manifest_sha256": _sha256_file(sample_path),
            },
            "acquisition_history": config["postfreeze_acquisition_audit"],
            "boundary": {
                "successful_exploration_temperature_queries": len(records),
                "cumulative_exploration_temperature_queries": len(records)
                + int(config["postfreeze_acquisition_audit"]["temperature_queries_issued"]),
                "successful_exploration_entropy_queries": len(records),
                "cumulative_exploration_entropy_queries": len(records)
                + int(config["postfreeze_acquisition_audit"]["entropy_queries_issued"]),
                "successful_exploration_metadata_queries": len(records),
                "cumulative_exploration_metadata_queries": len(records)
                + int(config["postfreeze_acquisition_audit"]["metadata_queries_issued"]),
                "successful_exploration_primary_response_queries": len(records),
                "cumulative_exploration_primary_response_queries": len(records)
                + int(config["postfreeze_acquisition_audit"]["primary_response_queries_issued"]),
                "exploration_primary_response_rows": len(records),
                "cumulative_primary_response_rows_returned": len(records)
                + int(config["postfreeze_acquisition_audit"]["primary_response_rows_returned"]),
                "reserved_confirmation_primary_response_queries": 0,
                "hecs_mass_profile_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
            },
            "records": records,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_source_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem6ThermodynamicStateError("source content hash changed")
    if manifest["preregistration"]["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem6ThermodynamicStateError("scientific freeze binding changed")
    if {str(row["hecs_name"]) for row in manifest["records"]} != set(
        config["sample"]["exploration"]
    ):
        raise GravityItem6ThermodynamicStateError("source exploration scope changed")
    boundary = manifest["boundary"]
    if boundary["exploration_primary_response_rows"] != config["sample"]["exploration_count"]:
        raise GravityItem6ThermodynamicStateError("response row count changed")
    if boundary["reserved_confirmation_primary_response_queries"] != 0:
        raise GravityItem6ThermodynamicStateError("confirmation response was queried")
    if boundary["hecs_mass_profile_values_acquired"] != 0:
        raise GravityItem6ThermodynamicStateError("HeCS mass profile was acquired")
    if boundary["lensing_mass_values_acquired"] != 0:
        raise GravityItem6ThermodynamicStateError("lensing mass was acquired")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem6ThermodynamicStateError("source contains an overclaim")


def write_source_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(acquire_exploration(root)) + b"\n")
    return path


def _median(values: Sequence[float]) -> float:
    if not values:
        raise GravityItem6ThermodynamicStateError("cannot aggregate an empty profile")
    return float(np.median(np.asarray(values, dtype=np.float64)))


def _quality_failure(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    metadata = record["metadata"]
    response = record["primary_response"]
    entropy_rows = record["entropy_rows"]
    quality = config["quality"]
    if int(metadata["member_count"]) < int(quality["minimum_member_galaxies"]):
        return "insufficient member galaxies"
    if any(
        int(row["n_bins"]) < int(quality["minimum_primary_entropy_bins"]) for row in entropy_rows
    ):
        return "insufficient primary entropy bins"
    sigma = float(response["sigma_km_s"])
    upper = float(response["sigma_upper_error_km_s"])
    lower = float(response["sigma_lower_error_km_s"])
    if sigma <= 0 or upper <= 0 or lower <= 0 or sigma - lower <= 0:
        return "invalid velocity dispersion uncertainty"
    if max(upper, lower) / sigma > float(quality["maximum_velocity_dispersion_error_fraction"]):
        return "velocity dispersion uncertainty too large"
    return None


def build_feature_rows(
    source: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_source_manifest(source, config)
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for record in source["records"]:
        failure = _quality_failure(record, config)
        if failure is not None:
            failures.append({"cluster": str(record["hecs_name"]), "reason": failure})
            continue
        temperature = _median([float(row["temperature_kev"]) for row in record["temperature_rows"]])
        core_entropy = _median(
            [float(row["core_entropy_kev_cm2"]) for row in record["entropy_rows"]]
        )
        entropy_100 = _median([float(row["entropy_100_kev_cm2"]) for row in record["entropy_rows"]])
        entropy_slope = _median([float(row["entropy_slope"]) for row in record["entropy_rows"]])
        metadata = record["metadata"]
        response = record["primary_response"]
        features = measure_thermodynamic_features(
            temperature_kev=temperature,
            core_entropy_kev_cm2=core_entropy,
            entropy_100_kev_cm2=entropy_100,
            entropy_slope=entropy_slope,
            xray_luminosity_1e43_erg_s=float(metadata["xray_luminosity_1e43_erg_s"]),
            redshift=float(metadata["redshift"]),
            member_count=int(metadata["member_count"]),
        )
        sigma = float(response["sigma_km_s"])
        upper = float(response["sigma_upper_error_km_s"])
        lower = float(response["sigma_lower_error_km_s"])
        rows.append(
            {
                "cluster": str(record["hecs_name"]),
                "accept_name": str(record["accept_name"]),
                "temperature_kev": temperature,
                "core_entropy_kev_cm2": core_entropy,
                "entropy_100_kev_cm2": entropy_100,
                "redshift": float(metadata["redshift"]),
                "member_count": int(metadata["member_count"]),
                "response_log10_sigma": math.log10(sigma),
                "response_upper_log10_sigma": math.log10(sigma + upper),
                "response_lower_log10_sigma": math.log10(sigma - lower),
                "cool_core_stratum": ("cool_core" if core_entropy < 30.0 else "non_cool_core"),
                "temperature_stratum": (
                    "low_temperature" if temperature < 7.0 else "high_temperature"
                ),
                **features,
            }
        )
    rows.sort(key=lambda row: str(row["cluster"]))
    summary = _seal(
        {
            "schema_version": "invariant-gravity-item6-thermodynamic-extraction-1.0",
            "goal": config["goal"],
            "decision": (
                "PASS_ITEM6_THERMODYNAMIC_REPRESENTATION_QUALITY"
                if not failures
                else "FAIL_ITEM6_THERMODYNAMIC_REPRESENTATION_QUALITY"
            ),
            "counts": {
                "exploration_clusters": config["sample"]["exploration_count"],
                "quality_passing_clusters": len(rows),
                "quality_failures": len(failures),
                "reserved_confirmation_response_accesses": 0,
            },
            "failures": failures,
            "source_manifest_content_sha256": source["content_sha256"],
        }
    )
    return rows, summary


def write_extraction(root: Path) -> tuple[Path, Path]:
    root = root.resolve()
    config = load_config(root)
    source = json.loads((root / config["source_manifest_output"]).read_text(encoding="utf-8"))
    rows, summary = build_feature_rows(source, config=config)
    feature_path = root / config["feature_output"]
    summary_path = root / config["extraction_summary_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "cluster",
        "accept_name",
        "temperature_kev",
        "core_entropy_kev_cm2",
        "entropy_100_kev_cm2",
        "redshift",
        "member_count",
        "response_log10_sigma",
        "response_upper_log10_sigma",
        "response_lower_log10_sigma",
        "cool_core_stratum",
        "temperature_stratum",
        *FEATURE_NAMES,
    )
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, summary_path


def load_feature_rows(root: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (root / config["feature_output"]).open(encoding="utf-8", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter="\t"):
            row: dict[str, Any] = {
                "cluster": raw["cluster"],
                "accept_name": raw["accept_name"],
                "cool_core_stratum": raw["cool_core_stratum"],
                "temperature_stratum": raw["temperature_stratum"],
                "member_count": int(raw["member_count"]),
            }
            for field in (
                "temperature_kev",
                "core_entropy_kev_cm2",
                "entropy_100_kev_cm2",
                "redshift",
                "response_log10_sigma",
                "response_upper_log10_sigma",
                "response_lower_log10_sigma",
                *FEATURE_NAMES,
            ):
                row[field] = float(raw[field])
            rows.append(row)
    return rows


def _array(rows: Sequence[Mapping[str, Any]], field: str) -> np.ndarray:
    return np.asarray([float(row[field]) for row in rows])


def _predictions_array(
    rows: Sequence[Mapping[str, Any]], predictions: Mapping[str, float]
) -> np.ndarray:
    return np.asarray([float(predictions[str(row["cluster"])]) for row in rows])


def _metrics_for_response(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    *,
    response: str = "response_log10_sigma",
) -> dict[str, str]:
    observed = _array(rows, response)
    predicted = _predictions_array(rows, predictions)
    return {
        "mse": _metric(cvcore._mse(observed, predicted)),
        "r2": _metric(cvcore._r2(observed, predicted)),
    }


def _permutation_test(
    rows: Sequence[Mapping[str, Any]],
    *,
    baseline_models: Sequence[Mapping[str, Any]],
    qualifying_models: Sequence[Mapping[str, Any]],
    observed_improvement: float,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    count = int(config["cross_validation"]["permutation_count"])
    seed = int.from_bytes(
        hashlib.sha256(str(config["cross_validation"]["permutation_salt"]).encode()).digest()[:8],
        "big",
    )
    rng = np.random.default_rng(seed)
    original = _array(rows, "response_log10_sigma")
    indices_by_stratum = {
        stratum: [index for index, row in enumerate(rows) if row["temperature_stratum"] == stratum]
        for stratum in ("low_temperature", "high_temperature")
    }
    null: list[float] = []
    for _ in range(count):
        shuffled = original.copy()
        for indices in indices_by_stratum.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = [
            dict(row, response_log10_sigma=float(shuffled[index])) for index, row in enumerate(rows)
        ]
        baseline, _ = cvcore._nested_predictions(
            permuted, models=baseline_models, config=config, detailed=False
        )
        qualifying, _ = cvcore._nested_predictions(
            permuted, models=qualifying_models, config=config, detailed=False
        )
        observed = _array(permuted, "response_log10_sigma")
        null.append(
            cvcore._mse(observed, _predictions_array(permuted, baseline))
            - cvcore._mse(observed, _predictions_array(permuted, qualifying))
        )
    p_value = (1 + sum(value >= observed_improvement for value in null)) / (count + 1)
    return {
        "permutations": count,
        "observed_mse_improvement": _metric(observed_improvement),
        "p_value": _metric(p_value),
        "null_improvement_quantiles": {
            "q05": _metric(float(np.quantile(null, 0.05))),
            "q50": _metric(float(np.quantile(null, 0.50))),
            "q95": _metric(float(np.quantile(null, 0.95))),
        },
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    source_path = root / config["source_manifest_output"]
    feature_path = root / config["feature_output"]
    extraction_path = root / config["extraction_summary_output"]
    source = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source, config)
    extraction = json.loads(extraction_path.read_text(encoding="utf-8"))
    rows = load_feature_rows(root, config)
    models = config["model_families"]
    baseline_models = [model for model in models if not model["qualifying"]]
    qualifying_models = [model for model in models if model["qualifying"]]

    unrestricted_predictions, unrestricted_folds = cvcore._nested_predictions(
        rows, models=models, config=config, detailed=True
    )
    baseline_predictions, baseline_folds = cvcore._nested_predictions(
        rows, models=baseline_models, config=config, detailed=True
    )
    qualifying_predictions, qualifying_folds = cvcore._nested_predictions(
        rows, models=qualifying_models, config=config, detailed=True
    )
    observed = _array(rows, "response_log10_sigma")
    baseline_array = _predictions_array(rows, baseline_predictions)
    qualifying_array = _predictions_array(rows, qualifying_predictions)
    baseline_mse = cvcore._mse(observed, baseline_array)
    qualifying_mse = cvcore._mse(observed, qualifying_array)
    improvement = baseline_mse - qualifying_mse
    relative_improvement = improvement / baseline_mse if baseline_mse > 0 else -1.0

    individual_baselines: dict[str, Any] = {}
    beats_each = True
    for model in baseline_models:
        predictions, _ = cvcore._nested_predictions(
            rows, models=[model], config=config, detailed=False
        )
        metrics = _metrics_for_response(rows, predictions)
        gain = float(metrics["mse"]) - qualifying_mse
        individual_baselines[str(model["id"])] = {
            **metrics,
            "qualifying_mse_gain": _metric(gain),
        }
        beats_each = beats_each and gain > 0

    strata: dict[str, Any] = {}
    cool_improvements: list[float] = []
    for field, labels in (
        ("cool_core_stratum", ("cool_core", "non_cool_core")),
        ("temperature_stratum", ("low_temperature", "high_temperature")),
    ):
        for label in labels:
            subset = [row for row in rows if row[field] == label]
            baseline_metrics = _metrics_for_response(subset, baseline_predictions)
            qualifying_metrics = _metrics_for_response(subset, qualifying_predictions)
            gain = float(baseline_metrics["mse"]) - float(qualifying_metrics["mse"])
            strata[f"{field}:{label}"] = {
                "count": len(subset),
                "baseline": baseline_metrics,
                "qualifying": qualifying_metrics,
                "qualifying_mse_gain": _metric(gain),
            }
            if field == "cool_core_stratum":
                cool_improvements.append(gain)

    envelopes: dict[str, Any] = {}
    envelope_improvements: list[float] = []
    for field in ("response_lower_log10_sigma", "response_upper_log10_sigma"):
        baseline_envelope = cvcore._mse(_array(rows, field), baseline_array)
        qualifying_envelope = cvcore._mse(_array(rows, field), qualifying_array)
        gain = baseline_envelope - qualifying_envelope
        envelope_improvements.append(gain)
        envelopes[field] = {
            "baseline_mse": _metric(baseline_envelope),
            "qualifying_mse": _metric(qualifying_envelope),
            "qualifying_mse_gain": _metric(gain),
        }

    permutation = _permutation_test(
        rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        observed_improvement=improvement,
        config=config,
    )
    admission = config["exploration_admission"]
    quality_pass = (
        extraction["decision"] == "PASS_ITEM6_THERMODYNAMIC_REPRESENTATION_QUALITY"
        and len(rows) == config["sample"]["exploration_count"]
    )
    gates = {
        "all_20_exploration_clusters_pass_frozen_quality": quality_pass,
        "unrestricted_selector_qualifying_in_at_least_4_of_5_folds": sum(
            bool(fold["selected_qualifying"]) for fold in unrestricted_folds
        )
        >= int(admission["unrestricted_selector_qualifying_in_at_least_folds"]),
        "qualifying_selector_r2_positive_overall": cvcore._r2(observed, qualifying_array) > 0,
        "qualifying_selector_beats_each_nonqualifying_baseline_overall": beats_each,
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": relative_improvement
        >= float(admission["qualifying_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "qualifying_improvement_positive_in_both_cool_core_strata": all(
            value > 0 for value in cool_improvements
        ),
        "temperature_stratified_permutation_p_at_most_0_05": float(permutation["p_value"])
        <= float(admission["temperature_stratified_permutation_p_at_most"]),
        "upper_and_lower_response_error_envelopes_do_not_reverse_improvement": all(
            value >= 0 for value in envelope_improvements
        ),
        "reserved_confirmation_targets_untouched": source["boundary"][
            "reserved_confirmation_primary_response_queries"
        ]
        == 0,
    }
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM6_THERMODYNAMIC_STATE_QUALITY_GATE"
    elif all(gates.values()):
        decision = "PASS_ITEM6_THERMODYNAMIC_STATE_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
    else:
        decision = "REJECT_ITEM6_THERMODYNAMIC_STATE_EXPLORATION"
    return _seal(
        {
            "schema_version": "invariant-gravity-item6-thermodynamic-result-1.0",
            "goal": config["goal"],
            "item_number": 6,
            "decision": decision,
            "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
            "hypothesis": config["scientific_contract"]["hypothesis"],
            "creativity_label": config["scientific_contract"]["creativity_label"],
            "inputs": {
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_path": config["sample_manifest_output"],
                "sample_manifest_sha256": _sha256_file(sample_path),
                "source_manifest_path": config["source_manifest_output"],
                "source_manifest_sha256": _sha256_file(source_path),
                "source_manifest_content_sha256": source["content_sha256"],
                "feature_path": config["feature_output"],
                "feature_sha256": _sha256_file(feature_path),
                "extraction_summary_path": config["extraction_summary_output"],
                "extraction_summary_sha256": _sha256_file(extraction_path),
            },
            "counts": {
                "exploration_clusters": len(rows),
                "reserved_confirmation_clusters": config["sample"]["reserved_confirmation_count"],
                "reserved_confirmation_target_accesses": 0,
                "paid_model_calls": 0,
                "permutation_nested_cv_runs": config["cross_validation"]["permutation_count"],
            },
            "models": config["model_families"],
            "primary": {
                "unrestricted": {
                    "metrics": _metrics_for_response(rows, unrestricted_predictions),
                    "folds": unrestricted_folds,
                },
                "strongest_nonqualifying_selector": {
                    "metrics": _metrics_for_response(rows, baseline_predictions),
                    "folds": baseline_folds,
                },
                "qualifying_selector": {
                    "metrics": _metrics_for_response(rows, qualifying_predictions),
                    "folds": qualifying_folds,
                    "mse_improvement_over_strongest_baseline": _metric(improvement),
                    "relative_mse_improvement_over_strongest_baseline": _metric(
                        relative_improvement
                    ),
                },
                "individual_nonqualifying_baselines": individual_baselines,
                "strata": strata,
            },
            "response_error_envelopes": envelopes,
            "permutation": permutation,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "claim_boundaries": dict(config["claim_boundaries"]),
            "next_action": (
                "Request explicit authorization before opening any of the eight confirmation dispersions."
                if decision.startswith("PASS_")
                else "Retain the exact thermodynamic-state counterexamples without retuning opened responses and synthesize Item 6 before deciding whether to advance to Item 7 baryonic composition."
            ),
        }
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem6ThermodynamicStateError("receipt content hash changed")
    load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem6ThermodynamicStateError("receipt freeze binding changed")
    gates = receipt["gate_checks"]
    expected = (
        "INCONCLUSIVE_ITEM6_THERMODYNAMIC_STATE_QUALITY_GATE"
        if not gates["all_20_exploration_clusters_pass_frozen_quality"]
        else (
            "PASS_ITEM6_THERMODYNAMIC_STATE_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
            if all(gates.values())
            else "REJECT_ITEM6_THERMODYNAMIC_STATE_EXPLORATION"
        )
    )
    if receipt["decision"] != expected:
        raise GravityItem6ThermodynamicStateError("decision does not follow frozen gates")
    if receipt["counts"]["reserved_confirmation_target_accesses"] != 0:
        raise GravityItem6ThermodynamicStateError("receipt opened confirmation")
    if any(bool(value) for value in receipt["claim_boundaries"].values()):
        raise GravityItem6ThermodynamicStateError("receipt contains an overclaim")
    if receipt["inputs"]["config_sha256"] != _sha256_file(root / CONFIG_PATH):
        raise GravityItem6ThermodynamicStateError("receipt config binding changed")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path = root / config["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def replay(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["output"]).read_text(encoding="utf-8"))
    rebuilt = build_receipt(root)
    if rebuilt != stored:
        raise GravityItem6ThermodynamicStateError("stored receipt does not replay")
    validate_receipt(stored, root=root)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("sample", "acquire", "extract", "experiment", "replay"):
        child = subparsers.add_parser(command)
        child.add_argument("--root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    root = arguments.root.resolve()
    if arguments.command == "sample":
        print(write_sample_manifest(root))
    elif arguments.command == "acquire":
        print(write_source_manifest(root))
    elif arguments.command == "extract":
        print(*write_extraction(root), sep="\n")
    elif arguments.command == "experiment":
        print(write_receipt(root))
    elif arguments.command == "replay":
        replay(root)
        print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
