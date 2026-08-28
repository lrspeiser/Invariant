"""Frozen PHANGS baryonic-composition experiment for gravity-roadmap Item 7."""

from __future__ import annotations

import argparse
import bisect
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

CONFIG_PATH = "configs/gravity_item7_baryonic_composition_phangs_v1.json"
SCIENTIFIC_FREEZE_COMMIT = "7bb8027f7a5e8183f8277c57edec456f5210ad45"
VIZIER_ENDPOINT = "https://vizier.cds.unistra.fr/viz-bin/asu-tsv"


class GravityItem7BaryonicCompositionError(RuntimeError):
    """Raised when the frozen Item 7 boundary or result drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem7BaryonicCompositionError("non-finite metric")
    return f"{float(value):.12e}"


def _seal(value: dict[str, Any]) -> dict[str, Any]:
    result = cvcore._canonicalize_floats(value)
    result.pop("content_sha256", None)
    result["content_sha256"] = canonical_sha256(result)
    return result


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item7-baryonic-composition-config-1.0"
    ):
        raise GravityItem7BaryonicCompositionError("unexpected Item 7 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem7BaryonicCompositionError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem7BaryonicCompositionError("Item 6 synthesis file changed")
    predecessor_receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor_receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem7BaryonicCompositionError("Item 6 synthesis content changed")
    if predecessor_receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem7BaryonicCompositionError("Item 6 did not authorize Item 7")
    dependency = config["implementation_dependency"]
    if _sha256_file(root / dependency["path"]) != dependency["file_sha256"]:
        raise GravityItem7BaryonicCompositionError("nested-CV dependency changed")

    authorization = config["authorization"]
    forbidden_true = (
        "paid_model_calls_allowed",
        "reserved_confirmation_rotation_responses_allowed",
        "dark_matter_or_dynamical_mass_allowed_as_predictor",
        "rotation_speed_or_curve_shape_allowed_as_predictor",
        "lensing_mass_allowed_as_predictor",
    )
    if any(bool(authorization[name]) for name in forbidden_true):
        raise GravityItem7BaryonicCompositionError("Item 7 authorization boundary changed")
    sample = config["sample"]
    exploration = [str(value) for value in sample["exploration"]]
    confirmation = [str(value) for value in sample["reserved_confirmation"]]
    if len(exploration) != sample["exploration_count"] or len(set(exploration)) != len(
        exploration
    ):
        raise GravityItem7BaryonicCompositionError("exploration sample changed")
    if len(confirmation) != sample["reserved_confirmation_count"] or len(set(confirmation)) != len(
        confirmation
    ):
        raise GravityItem7BaryonicCompositionError("confirmation sample changed")
    if set(exploration).intersection(confirmation):
        raise GravityItem7BaryonicCompositionError("sample roles overlap")
    if len(exploration) + len(confirmation) != sample["quality_passing_candidates"]:
        raise GravityItem7BaryonicCompositionError("candidate count changed")
    if config["prefreeze_audit"]["rotation_velocity_values_read"] != 0:
        raise GravityItem7BaryonicCompositionError("prefreeze response boundary changed")
    if config["derivation"]["feature_builder_accepts_rotation_response"]:
        raise GravityItem7BaryonicCompositionError("feature builder cannot accept response")
    for source_name in ("composition_table", "kinematic_metadata_table"):
        source = config["sources"][source_name]
        if set(source["allowed_columns"]).intersection(source.get("forbidden_columns", [])):
            raise GravityItem7BaryonicCompositionError("predictor source admits forbidden columns")
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
    for role, names in (("exploration", exploration), ("reserved_confirmation", confirmation)):
        for name in names:
            objects.append(
                {
                    "galaxy": name,
                    "role": role,
                    "outer_fold": assignments.get(name),
                    "selection_digest": hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
                }
            )
    objects.sort(key=lambda row: (str(row["role"]), str(row["galaxy"])))
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-baryonic-composition-sample-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM7_TARGET_BLIND_PHANGS_COMPOSITION_SAMPLE",
            "selection": {
                "quality_rule": sample["quality_rule"],
                "stratification_rule": sample["stratification_rule"],
                "salt": salt,
                "selection_used_rotation_response": False,
            },
            "counts": {
                "catalog_overlap": config["prefreeze_audit"]["catalog_overlap_galaxies"],
                "base_quality": config["prefreeze_audit"]["base_quality_galaxies"],
                "radius_coverage_quality": sample["quality_passing_candidates"],
                "exploration": len(exploration),
                "reserved_confirmation": len(confirmation),
            },
            "stratification_cells": sample["cell_counts"],
            "objects": objects,
            "prefreeze_boundary": {
                "radius_only_rows_audited": config["prefreeze_audit"][
                    "kinematic_radius_only_rows"
                ],
                "rotation_velocity_values_read": 0,
                "reserved_confirmation_predictors_blinded": False,
                "reserved_confirmation_rotation_responses_blinded": True,
                "forbidden_dynamical_or_dark_mass_values_used": 0,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7BaryonicCompositionError("sample content hash changed")
    roles = {"exploration": set(), "reserved_confirmation": set()}
    for row in manifest["objects"]:
        roles[str(row["role"])].add(str(row["galaxy"]))
    if roles["exploration"] != set(config["sample"]["exploration"]):
        raise GravityItem7BaryonicCompositionError("exploration identities changed")
    if roles["reserved_confirmation"] != set(config["sample"]["reserved_confirmation"]):
        raise GravityItem7BaryonicCompositionError("confirmation identities changed")
    boundary = manifest["prefreeze_boundary"]
    if boundary["rotation_velocity_values_read"] != 0:
        raise GravityItem7BaryonicCompositionError("sample opened rotation responses")
    if not boundary["reserved_confirmation_rotation_responses_blinded"]:
        raise GravityItem7BaryonicCompositionError("confirmation boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem7BaryonicCompositionError("sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def measure_composition_features(
    *,
    log_mstar: float,
    log_mhi: float,
    log_lco: float,
    co_aperture_correction: float,
    stellar_effective_radius_kpc: float,
    stellar_scale_length_kpc: float,
    optical_radius_kpc: float,
    log_sfr: float,
    inclination_deg: float,
    co_covering_fraction: float,
    rco_over_r25: float,
) -> dict[str, float]:
    """Build target-blind stellar/atomic/molecular mixture features."""

    values = (
        log_mstar,
        log_mhi,
        log_lco,
        co_aperture_correction,
        stellar_effective_radius_kpc,
        stellar_scale_length_kpc,
        optical_radius_kpc,
        log_sfr,
        inclination_deg,
        co_covering_fraction,
        rco_over_r25,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise GravityItem7BaryonicCompositionError("non-finite composition observable")
    if (
        co_aperture_correction <= 0
        or stellar_effective_radius_kpc <= 0
        or stellar_scale_length_kpc <= 0
        or optical_radius_kpc <= 0
        or not (0 < inclination_deg < 90)
        or co_covering_fraction <= 0
        or rco_over_r25 <= 0
    ):
        raise GravityItem7BaryonicCompositionError("invalid composition observable")
    mstar = 10.0**log_mstar
    matomic = 1.36 * 10.0**log_mhi
    mmol = (4.35 / 0.65) * 10.0**log_lco * co_aperture_correction
    mbar = mstar + matomic + mmol
    fractions = np.asarray([mstar, matomic, mmol], dtype=np.float64) / mbar
    fstar, fatomic, fmolecular = (float(value) for value in fractions)
    phase_entropy = -sum(value * math.log(value) for value in fractions) / math.log(3.0)
    molecular_to_atomic = math.log10(mmol / matomic)
    gas_to_stars = math.log10((matomic + mmol) / mstar)
    structure = math.log10(stellar_effective_radius_kpc / (1.678 * stellar_scale_length_kpc))
    atomic_molecular_boundary = 4.0 * fatomic * fmolecular
    stellar_gas_boundary = 4.0 * fstar * (fatomic + fmolecular)
    result = {
        "log_mstar": math.log10(mstar),
        "log_matomic": math.log10(matomic),
        "log_mmol": math.log10(mmol),
        "log_mbar": math.log10(mbar),
        "log_lstar": math.log10(stellar_scale_length_kpc),
        "log_re": math.log10(stellar_effective_radius_kpc),
        "log_r25": math.log10(optical_radius_kpc),
        "log_sfr": log_sfr,
        "inclination": inclination_deg,
        "co_covering_fraction": co_covering_fraction,
        "rco_over_r25": rco_over_r25,
        "co_aperture_correction": co_aperture_correction,
        "stellar_fraction": fstar,
        "atomic_fraction": fatomic,
        "molecular_fraction": fmolecular,
        "phase_entropy": phase_entropy,
        "atomic_molecular_boundary": atomic_molecular_boundary,
        "stellar_gas_boundary": stellar_gas_boundary,
        "molecular_to_atomic": molecular_to_atomic,
        "gas_to_stars": gas_to_stars,
        "stellar_structure_mismatch": structure,
        "molecular_atomic_ratio_squared": molecular_to_atomic**2,
        "phase_entropy_squared": phase_entropy**2,
        "phase_entropy_x_structure": phase_entropy * structure,
        "molecular_atomic_x_structure": molecular_to_atomic * structure,
        "gas_stars_x_coverage": gas_to_stars * rco_over_r25,
        "phase_boundary_x_coverage": atomic_molecular_boundary * rco_over_r25,
    }
    if not all(math.isfinite(value) for value in result.values()):
        raise GravityItem7BaryonicCompositionError("non-finite derived feature")
    return result


FEATURE_NAMES = (
    "log_mstar",
    "log_matomic",
    "log_mmol",
    "log_mbar",
    "log_lstar",
    "log_re",
    "log_r25",
    "log_sfr",
    "inclination",
    "co_covering_fraction",
    "rco_over_r25",
    "co_aperture_correction",
    "stellar_fraction",
    "atomic_fraction",
    "molecular_fraction",
    "phase_entropy",
    "atomic_molecular_boundary",
    "stellar_gas_boundary",
    "molecular_to_atomic",
    "gas_to_stars",
    "stellar_structure_mismatch",
    "molecular_atomic_ratio_squared",
    "phase_entropy_squared",
    "phase_entropy_x_structure",
    "molecular_atomic_x_structure",
    "gas_stars_x_coverage",
    "phase_boundary_x_coverage",
)


def _query_url(
    catalog_id: str,
    *,
    columns: Sequence[str],
    constraint_name: str,
    constraint_value: str,
    max_rows: int = 5000,
) -> str:
    parameters: list[tuple[str, str]] = [("-source", catalog_id)]
    parameters.extend(("-out", str(column)) for column in columns)
    parameters.extend(
        ((constraint_name, constraint_value), ("-out.max", str(max_rows)))
    )
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
    raise GravityItem7BaryonicCompositionError(f"VizieR acquisition failed: {url}") from error


def _tsv_rows(payload: bytes, *, first_field: str, fields: int) -> list[list[str]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem7BaryonicCompositionError("VizieR response is not UTF-8") from exc
    rows = [
        [field.strip() for field in line.split("\t")]
        for line in lines
        if line.split("\t", 1)[0].strip().upper() == first_field.upper()
    ]
    if any(len(row) != fields for row in rows):
        raise GravityItem7BaryonicCompositionError("VizieR schema changed")
    return rows


def parse_composition_payload(payload: bytes, *, galaxy: str) -> dict[str, Any]:
    rows = _tsv_rows(payload, first_field=galaxy, fields=9)
    if len(rows) != 1 or not all(rows[0][index] for index in range(1, 9)):
        raise GravityItem7BaryonicCompositionError("unexpected composition row")
    row = rows[0]
    return {
        "galaxy": galaxy,
        "log_mstar": float(row[1]),
        "r25_kpc": float(row[2]),
        "re_kpc": float(row[3]),
        "lstar_kpc": float(row[4]),
        "log_sfr": float(row[5]),
        "log_lco": float(row[6]),
        "co_aperture_correction": float(row[7]),
        "log_mhi": float(row[8]),
    }


def parse_metadata_payload(payload: bytes, *, galaxy: str) -> dict[str, Any]:
    rows = _tsv_rows(payload, first_field=galaxy, fields=6)
    if len(rows) != 1 or not all(rows[0][index] for index in range(1, 6)):
        raise GravityItem7BaryonicCompositionError("unexpected kinematic metadata row")
    row = rows[0]
    return {
        "galaxy": galaxy,
        "inclination_deg": float(row[1]),
        "co_covering_fraction": float(row[2]),
        "rbar_over_rco": float(row[3]),
        "rco_over_r25": float(row[4]),
        "n_rotation_rows": int(row[5]),
    }


def parse_curve_payload(payload: bytes, *, galaxy: str) -> list[dict[str, float]]:
    rows = _tsv_rows(payload, first_field=galaxy, fields=5)
    result = [
        {
            "radius_kpc": float(row[1]),
            "velocity_km_s": float(row[2]),
            "upper_error_km_s": float(row[3]),
            "lower_error_km_s": float(row[4]),
        }
        for row in rows
        if all(row[index] for index in range(1, 5))
    ]
    if not result:
        raise GravityItem7BaryonicCompositionError("no rotation-curve rows")
    result.sort(key=lambda row: row["radius_kpc"])
    return result


def _retrieval(url: str, payload: bytes) -> dict[str, Any]:
    return {"url": url, "payload_sha256": _sha256_bytes(payload), "bytes": len(payload)}


def acquire_exploration(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if SCIENTIFIC_FREEZE_COMMIT.startswith("PENDING_"):
        raise GravityItem7BaryonicCompositionError(
            "scientific freeze commit is not bound; response access forbidden"
        )
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    galaxies = sorted(config["sample"]["exploration"])
    sources = config["sources"]
    records: list[dict[str, Any]] = []
    for galaxy in galaxies:
        composition_url = _query_url(
            sources["composition_table"]["catalog_id"],
            columns=sources["composition_table"]["allowed_columns"],
            constraint_name="Name",
            constraint_value=galaxy,
            max_rows=10,
        )
        metadata_url = _query_url(
            sources["kinematic_metadata_table"]["catalog_id"],
            columns=sources["kinematic_metadata_table"]["allowed_columns"],
            constraint_name="ID",
            constraint_value=galaxy,
            max_rows=10,
        )
        response_url = _query_url(
            sources["rotation_curve_table"]["catalog_id"],
            columns=sources["rotation_curve_table"]["response_columns"],
            constraint_name="ID",
            constraint_value=galaxy,
            max_rows=500,
        )
        composition_payload = _fetch(composition_url)
        metadata_payload = _fetch(metadata_url)
        response_payload = _fetch(response_url)
        records.append(
            {
                "galaxy": galaxy,
                "composition": parse_composition_payload(composition_payload, galaxy=galaxy),
                "metadata": parse_metadata_payload(metadata_payload, galaxy=galaxy),
                "rotation_curve": parse_curve_payload(response_payload, galaxy=galaxy),
                "retrievals": {
                    "composition": _retrieval(composition_url, composition_payload),
                    "metadata": _retrieval(metadata_url, metadata_payload),
                    "primary_response": _retrieval(response_url, response_payload),
                },
            }
        )
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-baryonic-composition-source-1.0",
            "goal": config["goal"],
            "decision": "PASS_ITEM7_EXPLORATION_SOURCE_ACQUISITION",
            "preregistration": {
                "scientific_freeze_commit": SCIENTIFIC_FREEZE_COMMIT,
                "config_path": CONFIG_PATH,
                "config_sha256": _sha256_file(root / CONFIG_PATH),
                "sample_manifest_path": config["sample_manifest_output"],
                "sample_manifest_sha256": _sha256_file(sample_path),
            },
            "boundary": {
                "exploration_composition_queries": len(records),
                "exploration_metadata_queries": len(records),
                "exploration_primary_response_queries": len(records),
                "exploration_rotation_curve_rows": sum(
                    len(row["rotation_curve"]) for row in records
                ),
                "reserved_confirmation_primary_response_queries": 0,
                "dynamical_or_dark_mass_values_acquired": 0,
                "lensing_mass_values_acquired": 0,
                "paid_model_calls": 0,
                "co_intensity_and_velocity_share_phangs_alma_data": True,
            },
            "records": records,
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_source_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7BaryonicCompositionError("source content hash changed")
    if manifest["preregistration"]["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem7BaryonicCompositionError("scientific freeze binding changed")
    if {str(row["galaxy"]) for row in manifest["records"]} != set(
        config["sample"]["exploration"]
    ):
        raise GravityItem7BaryonicCompositionError("source exploration scope changed")
    boundary = manifest["boundary"]
    if boundary["exploration_primary_response_queries"] != config["sample"]["exploration_count"]:
        raise GravityItem7BaryonicCompositionError("response query count changed")
    if boundary["reserved_confirmation_primary_response_queries"] != 0:
        raise GravityItem7BaryonicCompositionError("confirmation response was queried")
    if boundary["dynamical_or_dark_mass_values_acquired"] != 0:
        raise GravityItem7BaryonicCompositionError("forbidden dynamical mass was acquired")
    if boundary["lensing_mass_values_acquired"] != 0:
        raise GravityItem7BaryonicCompositionError("forbidden lensing mass was acquired")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem7BaryonicCompositionError("source contains an overclaim")


def write_source_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(acquire_exploration(root)) + b"\n")
    return path


def _interpolate_curve(
    curve: Sequence[Mapping[str, Any]], *, radius_kpc: float
) -> dict[str, float] | None:
    radii = [float(row["radius_kpc"]) for row in curve]
    index = bisect.bisect_left(radii, radius_kpc)
    if index < len(radii) and math.isclose(radii[index], radius_kpc, abs_tol=1.0e-12):
        row = curve[index]
        return {
            "velocity_km_s": float(row["velocity_km_s"]),
            "upper_error_km_s": float(row["upper_error_km_s"]),
            "lower_error_km_s": float(row["lower_error_km_s"]),
        }
    if index == 0 or index == len(radii):
        return None
    left = curve[index - 1]
    right = curve[index]
    span = float(right["radius_kpc"]) - float(left["radius_kpc"])
    if span <= 0:
        return None
    weight = (radius_kpc - float(left["radius_kpc"])) / span
    return {
        field: (1.0 - weight) * float(left[field]) + weight * float(right[field])
        for field in ("velocity_km_s", "upper_error_km_s", "lower_error_km_s")
    }


def _features_for_record(record: Mapping[str, Any]) -> dict[str, float]:
    composition = record["composition"]
    metadata = record["metadata"]
    return measure_composition_features(
        log_mstar=float(composition["log_mstar"]),
        log_mhi=float(composition["log_mhi"]),
        log_lco=float(composition["log_lco"]),
        co_aperture_correction=float(composition["co_aperture_correction"]),
        stellar_effective_radius_kpc=float(composition["re_kpc"]),
        stellar_scale_length_kpc=float(composition["lstar_kpc"]),
        optical_radius_kpc=float(composition["r25_kpc"]),
        log_sfr=float(composition["log_sfr"]),
        inclination_deg=float(metadata["inclination_deg"]),
        co_covering_fraction=float(metadata["co_covering_fraction"]),
        rco_over_r25=float(metadata["rco_over_r25"]),
    )


def _quality_failure(record: Mapping[str, Any], config: Mapping[str, Any]) -> str | None:
    quality = config["quality"]
    metadata = record["metadata"]
    composition = record["composition"]
    curve = record["rotation_curve"]
    if int(metadata["n_rotation_rows"]) < int(quality["minimum_rotation_curve_rows"]):
        return "insufficient rotation-curve rows"
    inclination = float(metadata["inclination_deg"])
    if not (
        float(quality["minimum_inclination_deg"])
        <= inclination
        <= float(quality["maximum_inclination_deg"])
    ):
        return "inclination outside frozen range"
    target_radius = float(quality["minimum_radius_in_lstar"]) * float(
        composition["lstar_kpc"]
    )
    response = _interpolate_curve(curve, radius_kpc=target_radius)
    if response is None:
        return "rotation curve does not bracket primary radius"
    velocity = float(response["velocity_km_s"])
    upper = float(response["upper_error_km_s"])
    lower = float(response["lower_error_km_s"])
    if velocity <= 0 or upper <= 0 or lower <= 0 or velocity - lower <= 0:
        return "invalid interpolated velocity uncertainty"
    if max(upper, lower) / velocity > float(
        quality["maximum_interpolated_fractional_error"]
    ):
        return "interpolated fractional velocity error too large"
    return None


def build_feature_rows(
    source: Mapping[str, Any], *, config: Mapping[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validate_source_manifest(source, config)
    feature_map = {
        str(record["galaxy"]): _features_for_record(record) for record in source["records"]
    }
    ordered_mass = sorted(
        feature_map, key=lambda name: (feature_map[name]["log_mbar"], name)
    )
    mass_quartile = {
        name: f"q{min(3, (4 * index) // len(ordered_mass)) + 1}"
        for index, name in enumerate(ordered_mass)
    }
    mass_half = {
        name: ("low_mass" if index < len(ordered_mass) / 2 else "high_mass")
        for index, name in enumerate(ordered_mass)
    }
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    shape_count = 0
    for record in source["records"]:
        galaxy = str(record["galaxy"])
        failure = _quality_failure(record, config)
        if failure is not None:
            failures.append({"galaxy": galaxy, "reason": failure})
            continue
        composition = record["composition"]
        lstar = float(composition["lstar_kpc"])
        primary = _interpolate_curve(record["rotation_curve"], radius_kpc=1.5 * lstar)
        radius_alt = _interpolate_curve(record["rotation_curve"], radius_kpc=1.4 * lstar)
        inner = _interpolate_curve(record["rotation_curve"], radius_kpc=0.75 * lstar)
        if primary is None or radius_alt is None:
            raise GravityItem7BaryonicCompositionError("frozen radius coverage drifted")
        velocity = float(primary["velocity_km_s"])
        upper = float(primary["upper_error_km_s"])
        lower = float(primary["lower_error_km_s"])
        curve_shape = ""
        if inner is not None and float(inner["velocity_km_s"]) > 0:
            curve_shape = math.log10(velocity / float(inner["velocity_km_s"]))
            shape_count += 1
        features = feature_map[galaxy]
        rows.append(
            {
                "cluster": galaxy,
                "galaxy": galaxy,
                "response_log10_velocity": math.log10(velocity),
                "response_upper_log10_velocity": math.log10(velocity + upper),
                "response_lower_log10_velocity": math.log10(velocity - lower),
                "response_radius_1_4_log10_velocity": math.log10(
                    float(radius_alt["velocity_km_s"])
                ),
                "curve_shape_log_ratio": curve_shape,
                "mass_quartile": mass_quartile[galaxy],
                "mass_stratum": mass_half[galaxy],
                "phase_stratum": (
                    "molecular_dominant"
                    if features["log_mmol"] > features["log_matomic"]
                    else "atomic_dominant"
                ),
                **features,
            }
        )
    rows.sort(key=lambda row: str(row["galaxy"]))
    summary = _seal(
        {
            "schema_version": "invariant-gravity-item7-baryonic-composition-extraction-1.0",
            "goal": config["goal"],
            "decision": (
                "PASS_ITEM7_BARYONIC_COMPOSITION_REPRESENTATION_QUALITY"
                if not failures
                else "FAIL_ITEM7_BARYONIC_COMPOSITION_REPRESENTATION_QUALITY"
            ),
            "counts": {
                "exploration_galaxies": config["sample"]["exploration_count"],
                "quality_passing_galaxies": len(rows),
                "quality_failures": len(failures),
                "curve_shape_responses": shape_count,
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
        "galaxy",
        "response_log10_velocity",
        "response_upper_log10_velocity",
        "response_lower_log10_velocity",
        "response_radius_1_4_log10_velocity",
        "curve_shape_log_ratio",
        "mass_quartile",
        "mass_stratum",
        "phase_stratum",
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
                "cluster": raw["galaxy"],
                "galaxy": raw["galaxy"],
                "response_log10_sigma": float(raw["response_log10_velocity"]),
                "mass_quartile": raw["mass_quartile"],
                "mass_stratum": raw["mass_stratum"],
                "phase_stratum": raw["phase_stratum"],
                "curve_shape_log_ratio": (
                    float(raw["curve_shape_log_ratio"])
                    if raw["curve_shape_log_ratio"]
                    else None
                ),
            }
            for field in (
                "response_log10_velocity",
                "response_upper_log10_velocity",
                "response_lower_log10_velocity",
                "response_radius_1_4_log10_velocity",
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
    return np.asarray([float(predictions[str(row["galaxy"])]) for row in rows])


def _metrics_for_response(
    rows: Sequence[Mapping[str, Any]],
    predictions: Mapping[str, float],
    *,
    response: str = "response_log10_velocity",
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
    original = _array(rows, "response_log10_velocity")
    indices_by_stratum = {
        stratum: [index for index, row in enumerate(rows) if row["mass_quartile"] == stratum]
        for stratum in ("q1", "q2", "q3", "q4")
    }
    null: list[float] = []
    for _ in range(count):
        shuffled = original.copy()
        for indices in indices_by_stratum.values():
            shuffled[indices] = rng.permutation(shuffled[indices])
        permuted = [
            dict(
                row,
                response_log10_velocity=float(shuffled[index]),
                response_log10_sigma=float(shuffled[index]),
            )
            for index, row in enumerate(rows)
        ]
        baseline, _ = cvcore._nested_predictions(
            permuted, models=baseline_models, config=config, detailed=False
        )
        qualifying, _ = cvcore._nested_predictions(
            permuted, models=qualifying_models, config=config, detailed=False
        )
        observed = _array(permuted, "response_log10_velocity")
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
    observed = _array(rows, "response_log10_velocity")
    baseline_array = _predictions_array(rows, baseline_predictions)
    qualifying_array = _predictions_array(rows, qualifying_predictions)
    baseline_mse = cvcore._mse(observed, baseline_array)
    qualifying_mse = cvcore._mse(observed, qualifying_array)
    improvement = baseline_mse - qualifying_mse
    relative_improvement = improvement / baseline_mse if baseline_mse > 0 else -1.0

    individual_baselines: dict[str, Any] = {}
    beats_each = True
    for model in baseline_models:
        predictions, _ = cvcore._nested_predictions(rows, models=[model], config=config, detailed=False)
        metrics = _metrics_for_response(rows, predictions)
        gain = float(metrics["mse"]) - qualifying_mse
        individual_baselines[str(model["id"])] = {
            **metrics,
            "qualifying_mse_gain": _metric(gain),
        }
        beats_each = beats_each and gain > 0

    strata: dict[str, Any] = {}
    mass_improvements: list[float] = []
    phase_improvements: list[float] = []
    for field, labels in (
        ("mass_stratum", ("low_mass", "high_mass")),
        ("phase_stratum", ("atomic_dominant", "molecular_dominant")),
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
            (mass_improvements if field == "mass_stratum" else phase_improvements).append(gain)

    envelopes: dict[str, Any] = {}
    envelope_improvements: list[float] = []
    for field in (
        "response_lower_log10_velocity",
        "response_upper_log10_velocity",
    ):
        baseline_value = cvcore._mse(_array(rows, field), baseline_array)
        qualifying_value = cvcore._mse(_array(rows, field), qualifying_array)
        gain = baseline_value - qualifying_value
        envelope_improvements.append(gain)
        envelopes[field] = {
            "baseline_mse": _metric(baseline_value),
            "qualifying_mse": _metric(qualifying_value),
            "qualifying_mse_gain": _metric(gain),
        }
    radius_baseline = cvcore._mse(
        _array(rows, "response_radius_1_4_log10_velocity"), baseline_array
    )
    radius_qualifying = cvcore._mse(
        _array(rows, "response_radius_1_4_log10_velocity"), qualifying_array
    )
    radius_gain = radius_baseline - radius_qualifying

    permutation = _permutation_test(
        rows,
        baseline_models=baseline_models,
        qualifying_models=qualifying_models,
        observed_improvement=improvement,
        config=config,
    )
    admission = config["exploration_admission"]
    quality_pass = (
        extraction["decision"] == "PASS_ITEM7_BARYONIC_COMPOSITION_REPRESENTATION_QUALITY"
        and len(rows) == config["sample"]["exploration_count"]
    )
    gates = {
        "all_33_exploration_galaxies_pass_frozen_quality": quality_pass,
        "unrestricted_selector_qualifying_in_at_least_4_of_5_folds": sum(
            bool(fold["selected_qualifying"]) for fold in unrestricted_folds
        )
        >= int(admission["unrestricted_selector_qualifying_in_at_least_folds"]),
        "qualifying_selector_r2_positive_overall": cvcore._r2(observed, qualifying_array) > 0,
        "qualifying_selector_beats_each_nonqualifying_baseline_overall": beats_each,
        "qualifying_relative_mse_improvement_over_strongest_baseline_at_least_0_02": relative_improvement
        >= float(admission["qualifying_relative_mse_improvement_over_strongest_baseline_at_least"]),
        "qualifying_improvement_positive_in_both_mass_strata": all(
            value > 0 for value in mass_improvements
        ),
        "qualifying_improvement_positive_in_atomic_and_molecular_dominance_strata": all(
            value > 0 for value in phase_improvements
        ),
        "mass_stratified_permutation_p_at_most_0_05": float(permutation["p_value"])
        <= float(admission["mass_stratified_permutation_p_at_most"]),
        "velocity_error_envelopes_do_not_reverse_improvement": all(
            value >= 0 for value in envelope_improvements
        ),
        "radius_1_4_robustness_does_not_reverse_improvement": radius_gain >= 0,
        "reserved_confirmation_targets_untouched": source["boundary"][
            "reserved_confirmation_primary_response_queries"
        ]
        == 0,
    }
    if not quality_pass:
        decision = "INCONCLUSIVE_ITEM7_BARYONIC_COMPOSITION_QUALITY_GATE"
    elif all(gates.values()):
        decision = (
            "PASS_ITEM7_BARYONIC_COMPOSITION_EXPLORATION_REQUIRES_CONFIRMATION_AUTHORIZATION"
        )
    else:
        decision = "REJECT_ITEM7_BARYONIC_COMPOSITION_EXPLORATION"
    return _seal(
        {
            "schema_version": "invariant-gravity-item7-baryonic-composition-result-1.0",
            "goal": config["goal"],
            "item_number": 7,
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
                "exploration_galaxies": len(rows),
                "reserved_confirmation_galaxies": config["sample"][
                    "reserved_confirmation_count"
                ],
                "reserved_confirmation_target_accesses": 0,
                "rotation_curve_rows": source["boundary"]["exploration_rotation_curve_rows"],
                "curve_shape_responses": extraction["counts"]["curve_shape_responses"],
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
                    "absolute_mse_improvement_over_strongest_baseline": _metric(improvement),
                    "relative_mse_improvement_over_strongest_baseline": _metric(
                        relative_improvement
                    ),
                },
                "individual_nonqualifying_baselines": individual_baselines,
            },
            "strata": strata,
            "response_error_envelopes": envelopes,
            "radius_robustness": {
                "radius_in_lstar": "1.400000000000e+00",
                "baseline_mse": _metric(radius_baseline),
                "qualifying_mse": _metric(radius_qualifying),
                "qualifying_mse_gain": _metric(radius_gain),
            },
            "permutation": permutation,
            "gate_checks": gates,
            "gate_counts": {"passed": sum(gates.values()), "required": len(gates)},
            "limitations": {
                "same_phangs_co_data_for_luminosity_and_velocity": True,
                "resolved_hi_profile_used": False,
                "resolved_molecular_mass_profile_used": False,
                "plasma_or_ionized_mass_used": False,
                "confirmation_opened": False,
            },
            "claims": dict(config["claim_boundaries"]),
        }
    )


def validate_receipt(receipt: Mapping[str, Any], *, root: Path) -> None:
    copy = dict(receipt)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem7BaryonicCompositionError("result content hash changed")
    load_config(root)
    if receipt["scientific_freeze_commit"] != SCIENTIFIC_FREEZE_COMMIT:
        raise GravityItem7BaryonicCompositionError("result freeze binding changed")
    if receipt["counts"]["reserved_confirmation_target_accesses"] != 0:
        raise GravityItem7BaryonicCompositionError("confirmation target was opened")
    passed = sum(bool(value) for value in receipt["gate_checks"].values())
    if receipt["gate_counts"] != {"passed": passed, "required": len(receipt["gate_checks"])}:
        raise GravityItem7BaryonicCompositionError("gate count changed")
    passing_decision = receipt["decision"].startswith("PASS_ITEM7_")
    if passing_decision != all(bool(value) for value in receipt["gate_checks"].values()):
        raise GravityItem7BaryonicCompositionError("decision does not match frozen gates")
    if any(bool(value) for value in receipt["claims"].values()):
        raise GravityItem7BaryonicCompositionError("result contains an overclaim")
    if receipt != build_receipt(root):
        raise GravityItem7BaryonicCompositionError("result does not replay")


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    receipt = build_receipt(root)
    validate_receipt(receipt, root=root)
    path.write_bytes(canonical_json_bytes(receipt) + b"\n")
    return path


def check_receipt(root: Path) -> None:
    root = root.resolve()
    config = load_config(root)
    stored = json.loads((root / config["output"]).read_text(encoding="utf-8"))
    validate_receipt(stored, root=root)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("sample", "acquire", "extract", "run", "check"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.command == "sample":
        print(write_sample_manifest(args.root))
    elif args.command == "acquire":
        print(write_source_manifest(args.root))
    elif args.command == "extract":
        print(*write_extraction(args.root), sep="\n")
    elif args.command == "run":
        print(write_receipt(args.root))
    else:
        check_receipt(args.root)


if __name__ == "__main__":
    main()
