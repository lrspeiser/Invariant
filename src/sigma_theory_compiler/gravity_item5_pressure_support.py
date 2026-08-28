"""Frozen source and sample boundary for gravity-roadmap Item 5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item5_pressure_support_little_things_v1.json"
FREEZE_COMMIT = "cd78ad6cda26e1251baa3a0ec172a43b0dafe5af"


class GravityItem5PressureSupportError(RuntimeError):
    """Raised when the frozen Item 5 boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _metric(value: float) -> str:
    if not math.isfinite(float(value)):
        raise GravityItem5PressureSupportError("non-finite metric")
    return f"{float(value):.12e}"


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item5-pressure-support-config-1.0"
    ):
        raise GravityItem5PressureSupportError("unexpected Item 5 config schema")
    roadmap = config["roadmap_binding"]
    if _sha256_file(root / roadmap["path"]) != roadmap["file_sha256"]:
        raise GravityItem5PressureSupportError("stable roadmap changed")
    predecessor = config["predecessor"]
    predecessor_path = root / predecessor["path"]
    if _sha256_file(predecessor_path) != predecessor["file_sha256"]:
        raise GravityItem5PressureSupportError("Item 4 synthesis file changed")
    receipt = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if receipt.get("content_sha256") != predecessor["content_sha256"]:
        raise GravityItem5PressureSupportError("Item 4 synthesis content changed")
    if receipt.get("decision") != predecessor["required_decision"]:
        raise GravityItem5PressureSupportError("Item 4 did not authorize Item 5")
    archive = root / config["sources"]["predictor_archive"]["audit_path"]
    if _sha256_file(archive) != config["sources"]["predictor_archive"]["file_sha256"]:
        raise GravityItem5PressureSupportError("predictor archive changed")
    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem5PressureSupportError("paid model calls are forbidden")
    if authorization["reserved_confirmation_archive_members_allowed"]:
        raise GravityItem5PressureSupportError("confirmation predictor access is forbidden")
    if authorization["reserved_confirmation_target_queries_allowed"]:
        raise GravityItem5PressureSupportError("confirmation target access is forbidden")
    return config


def _archive_galaxies(archive: Path, suffix: str) -> dict[str, str]:
    with zipfile.ZipFile(archive) as handle:
        names = handle.namelist()
    result: dict[str, str] = {}
    for member in names:
        path = Path(member)
        if not member.startswith("finalrot/") or not path.name.endswith(suffix):
            continue
        galaxy = path.name[: -len(suffix)].lower()
        if galaxy in result:
            raise GravityItem5PressureSupportError(f"duplicate archive galaxy: {galaxy}")
        result[galaxy] = member
    return result


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample = config["sample"]
    archive = root / config["sources"]["predictor_archive"]["audit_path"]
    archive_names = _archive_galaxies(archive, str(sample["archive_member_suffix"]))
    alternative = archive_names.pop("ddo216b", None)
    if alternative is None or len(archive_names) != int(
        sample["expected_archive_galaxies_excluding_alternative"]
    ):
        raise GravityItem5PressureSupportError("archive galaxy metadata changed")
    excluded = {str(value) for value in sample["excluded"]}
    candidates = [name for name in archive_names if name not in excluded]
    if len(candidates) != int(sample["expected_fresh_candidates"]):
        raise GravityItem5PressureSupportError("fresh candidate count changed")
    salt = str(sample["selection_salt"])
    ordered = sorted(
        candidates,
        key=lambda name: hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
    )
    exploration_count = int(sample["exploration"])
    objects: list[dict[str, Any]] = []
    for ordinal, name in enumerate(ordered):
        role = "exploration" if ordinal < exploration_count else "reserved_confirmation"
        objects.append(
            {
                "galaxy": name,
                "archive_member": archive_names[name],
                "role": role,
                "selection_digest": hashlib.sha256(f"{salt}|{name}".encode()).hexdigest(),
            }
        )
    objects.sort(key=lambda row: (str(row["role"]), str(row["galaxy"])))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item5-pressure-support-sample-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM5_TARGET_BLIND_LITTLE_THINGS_SAMPLE",
        "archive_binding": {
            "path": config["sources"]["predictor_archive"]["audit_path"],
            "file_sha256": _sha256_file(archive),
            "archive_member_contents_read_before_freeze": 0,
        },
        "counts": {
            "archive_galaxies_excluding_alternative": len(archive_names),
            "fresh_candidates": len(candidates),
            "exploration": sum(row["role"] == "exploration" for row in objects),
            "reserved_confirmation": sum(row["role"] == "reserved_confirmation" for row in objects),
        },
        "objects": objects,
        "selection_boundary": {
            "archive_container_downloaded": 1,
            "archive_central_directory_listings": 1,
            "archive_member_contents_read": 0,
            "independent_pipeline_target_rows_read": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "historical_novelty_established": False,
            "roadmap_item_5_complete": False,
            "selection_used_response": False,
        },
        "content_sha256": None,
    }
    content = dict(manifest)
    content.pop("content_sha256")
    manifest["content_sha256"] = canonical_sha256(content)
    validate_sample_manifest(manifest, config)
    return manifest


def validate_sample_manifest(manifest: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureSupportError("sample content hash changed")
    sample = config["sample"]
    expected_counts = {
        "archive_galaxies_excluding_alternative": sample[
            "expected_archive_galaxies_excluding_alternative"
        ],
        "fresh_candidates": sample["expected_fresh_candidates"],
        "exploration": sample["exploration"],
        "reserved_confirmation": sample["reserved_confirmation"],
    }
    if manifest["counts"] != expected_counts:
        raise GravityItem5PressureSupportError("sample counts changed")
    if Counter(str(row["role"]) for row in manifest["objects"]) != {
        "exploration": sample["exploration"],
        "reserved_confirmation": sample["reserved_confirmation"],
    }:
        raise GravityItem5PressureSupportError("sample roles changed")
    if manifest["selection_boundary"] != {
        "archive_container_downloaded": 1,
        "archive_central_directory_listings": 1,
        "archive_member_contents_read": 0,
        "independent_pipeline_target_rows_read": 0,
        "reserved_confirmation_target_accesses": 0,
    }:
        raise GravityItem5PressureSupportError("selection boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem5PressureSupportError("sample contains overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(build_sample_manifest(root)) + b"\n")
    return path


def _load_sample(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config)
    return sample


def _target_name(galaxy: str) -> str:
    if galaxy == "cvidwa":
        return "CVnIdwA"
    if galaxy == "wlm":
        return "WLM"
    prefix = "".join(character for character in galaxy if character.isalpha()).upper()
    number = "".join(character for character in galaxy if character.isdigit())
    if not prefix or not number:
        raise GravityItem5PressureSupportError(f"unknown target identifier: {galaxy}")
    return f"{prefix}_{number}"


def parse_predictor_payload(payload: bytes, *, galaxy: str) -> list[dict[str, float]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem5PressureSupportError("predictor file is not UTF-8") from exc
    rows: list[dict[str, float]] = []
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 12:
            raise GravityItem5PressureSupportError(f"predictor schema changed for {galaxy}")
        try:
            values = [float(value) for value in fields]
        except ValueError as exc:
            raise GravityItem5PressureSupportError(
                f"nonnumeric predictor row for {galaxy}"
            ) from exc
        rows.append(
            {
                "radius_arcsec": values[0],
                "radius_kpc": values[1],
                "vrot_km_s": values[2],
                "vrot_error_km_s": values[3],
                "published_va_km_s": values[4],
                "published_va_error_km_s": values[5],
                "published_vc_km_s": values[6],
                "published_vc_error_km_s": values[7],
                "sigma_km_s": values[8],
                "sigma_error_km_s": values[9],
                "surface_density_msun_pc2": values[10],
                "surface_density_error_msun_pc2": values[11],
            }
        )
    if len(rows) < 2 or any(
        later["radius_kpc"] <= earlier["radius_kpc"] for earlier, later in pairwise(rows)
    ):
        raise GravityItem5PressureSupportError(f"invalid predictor radii: {galaxy}")
    return rows


def parse_target_payload(payload: bytes, *, expected_target_name: str) -> list[dict[str, float]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem5PressureSupportError("target response is not UTF-8") from exc
    header = "Name\tType\tR0.3\tV0.3\tR\tV\te_V"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem5PressureSupportError(
            f"target schema changed: {expected_target_name}"
        ) from exc
    rows: list[dict[str, float]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 7 or fields[1].strip() != "Data":
            continue
        if fields[0].strip() != expected_target_name:
            raise GravityItem5PressureSupportError("target query returned another galaxy")
        try:
            r03, v03, scaled_radius, scaled_velocity, scaled_error = (
                float(value) for value in fields[2:]
            )
        except ValueError:
            continue
        rows.append(
            {
                "radius_kpc": r03 * scaled_radius,
                "target_velocity_km_s": v03 * scaled_velocity,
                "target_error_km_s": v03 * scaled_error,
            }
        )
    rows.sort(key=lambda row: row["radius_kpc"])
    if len(rows) < 2:
        raise GravityItem5PressureSupportError(
            f"target query returned insufficient rows: {expected_target_name}"
        )
    return rows


def _download(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GravityItem5PressureSupportError(f"target query failed: {url}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def acquire_exploration(root: Path, *, cache_dir: Path) -> Path:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation = {
        str(row["galaxy"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    archive_path = root / config["sources"]["predictor_archive"]["audit_path"]
    records: list[dict[str, Any]] = []
    with zipfile.ZipFile(archive_path) as archive:
        for row in exploration:
            galaxy = str(row["galaxy"])
            if galaxy in confirmation:
                raise GravityItem5PressureSupportError("sample roles overlap")
            predictor_payload = archive.read(str(row["archive_member"]))
            parse_predictor_payload(predictor_payload, galaxy=galaxy)
            predictor_path = cache_dir / f"predictor-{galaxy}.txt"
            predictor_path.parent.mkdir(parents=True, exist_ok=True)
            predictor_path.write_bytes(predictor_payload)
            target_name = _target_name(galaxy)
            target_url = str(
                config["sources"]["independent_pipeline_target"]["query_template"]
            ).format(name=urllib.parse.quote(target_name))
            target_path = cache_dir / f"target-{galaxy}.tsv"
            target_payload = _download(target_url, target_path)
            target_rows = parse_target_payload(target_payload, expected_target_name=target_name)
            records.append(
                {
                    "galaxy": galaxy,
                    "predictor": {
                        "archive_member": row["archive_member"],
                        "bytes": len(predictor_payload),
                        "path": str(predictor_path.relative_to(root)).replace("\\", "/"),
                        "sha256": _sha256_bytes(predictor_payload),
                    },
                    "target": {
                        "bytes": len(target_payload),
                        "name": target_name,
                        "path": str(target_path.relative_to(root)).replace("\\", "/"),
                        "rows": len(target_rows),
                        "sha256": _sha256_bytes(target_payload),
                        "url": target_url,
                    },
                }
            )
    records.sort(key=lambda row: str(row["galaxy"]))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item5-pressure-support-source-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM5_EXPLORATION_SOURCE_ACQUISITION",
        "preregistration": {
            "git_commit": FREEZE_COMMIT,
            "archive_member_contents_read_before_commit": 0,
            "target_rows_read_before_commit": 0,
        },
        "sample_binding": {
            "path": config["sample_manifest_output"],
            "file_sha256": _sha256_file(root / config["sample_manifest_output"]),
            "content_sha256": sample["content_sha256"],
        },
        "schema_audit": {
            "exploration_galaxy": "cvidwa",
            "predictor_member_accesses": 1,
            "target_query_accesses": 1,
            "sample_or_scientific_contract_changed": False,
        },
        "boundary": {
            "exploration_galaxies_acquired": len(records),
            "exploration_predictor_member_accesses": len(records) + 1,
            "exploration_target_query_accesses": len(records) + 1,
            "reserved_confirmation_predictor_member_accesses": 0,
            "reserved_confirmation_target_accesses": 0,
            "published_Iorio_Vc_used_as_predictor": False,
        },
        "records": records,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "historical_novelty_established": False,
            "roadmap_item_5_complete": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_source_manifest(manifest, sample=sample)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    return path


def validate_source_manifest(manifest: Mapping[str, Any], *, sample: Mapping[str, Any]) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem5PressureSupportError("source manifest hash changed")
    if manifest.get("decision") != "PASS_ITEM5_EXPLORATION_SOURCE_ACQUISITION":
        raise GravityItem5PressureSupportError("source acquisition did not pass")
    if manifest["preregistration"]["git_commit"] != FREEZE_COMMIT:
        raise GravityItem5PressureSupportError("freeze commit changed")
    expected = {str(row["galaxy"]) for row in sample["objects"] if row["role"] == "exploration"}
    if {str(row["galaxy"]) for row in manifest["records"]} != expected:
        raise GravityItem5PressureSupportError("source IDs differ from exploration")
    boundary = manifest["boundary"]
    if (
        int(boundary["reserved_confirmation_predictor_member_accesses"]) != 0
        or int(boundary["reserved_confirmation_target_accesses"]) != 0
        or bool(boundary["published_Iorio_Vc_used_as_predictor"])
    ):
        raise GravityItem5PressureSupportError("source boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem5PressureSupportError("source manifest contains overclaim")


def _gaussian_slope(log_radius: np.ndarray, log_pressure: np.ndarray) -> np.ndarray:
    bandwidth = 0.35
    slopes = np.empty_like(log_radius)
    for index, center in enumerate(log_radius):
        weights = np.exp(-0.5 * ((log_radius - center) / bandwidth) ** 2)
        design = np.column_stack((np.ones(len(log_radius)), log_radius - center))
        weighted = design.T * weights
        coefficients = np.linalg.solve(
            weighted @ design + np.diag([1.0e-12, 1.0e-12]),
            weighted @ log_pressure,
        )
        slopes[index] = coefficients[1]
    return slopes


def measure_support_only(
    radius_kpc: np.ndarray,
    vrot_km_s: np.ndarray,
    sigma_km_s: np.ndarray,
    surface_density: np.ndarray,
) -> dict[str, np.ndarray]:
    """Build ordered/random support features without accepting either target curve."""

    radius = np.asarray(radius_kpc, dtype=np.float64)
    vrot = np.asarray(vrot_km_s, dtype=np.float64)
    sigma = np.asarray(sigma_km_s, dtype=np.float64)
    density = np.asarray(surface_density, dtype=np.float64)
    if not (len(radius) == len(vrot) == len(sigma) == len(density)):
        raise GravityItem5PressureSupportError("support arrays differ in length")
    valid = (
        np.isfinite(radius)
        & np.isfinite(vrot)
        & np.isfinite(sigma)
        & np.isfinite(density)
        & (radius > 0)
        & (vrot > 0)
        & (sigma > 0)
        & (density > 0)
    )
    if not np.all(valid) or len(radius) < 3 or np.any(np.diff(radius) <= 0):
        raise GravityItem5PressureSupportError("invalid support profile")
    log_radius = np.log(radius)
    log_pressure = np.log(density * sigma**2)
    local_slope = np.gradient(log_pressure, log_radius, edge_order=2)
    curvature = np.gradient(local_slope, log_radius, edge_order=2)
    nonlocal_slope = _gaussian_slope(log_radius, log_pressure)
    memory_slope = np.empty_like(local_slope)
    memory_slope[0] = local_slope[0]
    for index in range(1, len(local_slope)):
        span = log_radius[: index + 1] - log_radius[0]
        denominator = float(span[-1])
        memory_slope[index] = (
            float(np.trapezoid(local_slope[: index + 1], span)) / denominator
            if denominator > 0
            else local_slope[index]
        )
    local_correction = -(sigma**2) * local_slope
    nonlocal_correction = -(sigma**2) * nonlocal_slope
    memory_correction = -(sigma**2) * memory_slope
    v_classical_squared = vrot**2 + local_correction
    if np.any(v_classical_squared <= 0):
        raise GravityItem5PressureSupportError("nonpositive classical support speed")
    result = {
        "log10_radius": np.log10(radius),
        "log10_vrot": np.log10(vrot),
        "log10_v_classical": 0.5 * np.log10(v_classical_squared),
        "local_pressure_fraction": local_correction / vrot**2,
        "pressure_curvature": curvature,
        "nonlocal_pressure_fraction": nonlocal_correction / vrot**2,
        "local_nonlocal_slope_difference": local_slope - nonlocal_slope,
        "memory_pressure_fraction": memory_correction / vrot**2,
        "local_memory_slope_difference": local_slope - memory_slope,
    }
    if any(np.any(~np.isfinite(values)) for values in result.values()):
        raise GravityItem5PressureSupportError("non-finite support feature")
    return result


def extract_features(root: Path, *, cache_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest, sample=sample)
    bindings = {str(row["galaxy"]): row for row in source_manifest["records"]}
    output_rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    galaxy_summaries: list[dict[str, Any]] = []
    for object_row in sample["objects"]:
        if object_row["role"] != "exploration":
            continue
        galaxy = str(object_row["galaxy"])
        binding = bindings[galaxy]
        predictor_path = root / binding["predictor"]["path"]
        target_path = root / binding["target"]["path"]
        if (
            _sha256_file(predictor_path) != binding["predictor"]["sha256"]
            or _sha256_file(target_path) != binding["target"]["sha256"]
        ):
            raise GravityItem5PressureSupportError(f"cached source changed: {galaxy}")
        predictor = parse_predictor_payload(predictor_path.read_bytes(), galaxy=galaxy)
        target = parse_target_payload(
            target_path.read_bytes(), expected_target_name=str(binding["target"]["name"])
        )
        try:
            support = measure_support_only(
                np.asarray([row["radius_kpc"] for row in predictor]),
                np.asarray([row["vrot_km_s"] for row in predictor]),
                np.asarray([row["sigma_km_s"] for row in predictor]),
                np.asarray([row["surface_density_msun_pc2"] for row in predictor]),
            )
            target_radius = np.asarray([row["radius_kpc"] for row in target])
            predictor_radius = np.asarray([row["radius_kpc"] for row in predictor])
            common = (predictor_radius >= target_radius[0]) & (
                predictor_radius <= target_radius[-1]
            )
            if len(predictor) < int(config["quality"]["minimum_predictor_rows"]):
                raise GravityItem5PressureSupportError("insufficient predictor rows")
            if np.count_nonzero(common) < int(config["quality"]["minimum_common_target_rows"]):
                raise GravityItem5PressureSupportError("insufficient common target rows")
            target_velocity = np.interp(
                predictor_radius[common],
                target_radius,
                np.asarray([row["target_velocity_km_s"] for row in target]),
            )
            target_error = np.interp(
                predictor_radius[common],
                target_radius,
                np.asarray([row["target_error_km_s"] for row in target]),
            )
        except GravityItem5PressureSupportError as exc:
            failures.append({"galaxy": galaxy, "reason": str(exc)})
            continue
        pressure_strength = float(np.mean(np.abs(support["local_pressure_fraction"][common])))
        galaxy_summaries.append(
            {
                "galaxy": galaxy,
                "common_rows": int(np.count_nonzero(common)),
                "mean_absolute_local_pressure_fraction": pressure_strength,
            }
        )
        common_indices = np.flatnonzero(common)
        for row_index, target_value, error_value in zip(
            common_indices, target_velocity, target_error, strict=True
        ):
            predictor_row = predictor[int(row_index)]
            output_rows.append(
                {
                    "galaxy": galaxy,
                    "radius_index": int(row_index),
                    "pressure_strength": pressure_strength,
                    **{key: float(values[row_index]) for key, values in support.items()},
                    "vrot_km_s": float(predictor_row["vrot_km_s"]),
                    "sigma_km_s": float(predictor_row["sigma_km_s"]),
                    "surface_density_msun_pc2": float(predictor_row["surface_density_msun_pc2"]),
                    "internal_vc_km_s": float(predictor_row["published_vc_km_s"]),
                    "internal_vc_error_km_s": float(predictor_row["published_vc_error_km_s"]),
                    "target_velocity_km_s": float(target_value),
                    "target_error_km_s": float(error_value),
                    "log10_target_velocity": math.log10(float(target_value)),
                }
            )
    if galaxy_summaries:
        median_strength = float(
            np.median([row["mean_absolute_local_pressure_fraction"] for row in galaxy_summaries])
        )
        for row in output_rows:
            row["pressure_stratum"] = (
                "high" if float(row["pressure_strength"]) >= median_strength else "low"
            )
    else:
        median_strength = 0.0
    fields = [
        "galaxy",
        "radius_index",
        "pressure_stratum",
        "pressure_strength",
        "log10_radius",
        "log10_vrot",
        "log10_v_classical",
        "local_pressure_fraction",
        "pressure_curvature",
        "nonlocal_pressure_fraction",
        "local_nonlocal_slope_difference",
        "memory_pressure_fraction",
        "local_memory_slope_difference",
        "vrot_km_s",
        "sigma_km_s",
        "surface_density_msun_pc2",
        "internal_vc_km_s",
        "internal_vc_error_km_s",
        "target_velocity_km_s",
        "target_error_km_s",
        "log10_target_velocity",
    ]
    output_rows.sort(key=lambda row: (str(row["galaxy"]), int(row["radius_index"])))
    feature_path = root / config["feature_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, dialect="excel-tab")
        writer.writeheader()
        for row in output_rows:
            writer.writerow(
                {
                    key: (
                        str(row[key])
                        if key in {"galaxy", "pressure_stratum"}
                        else int(row[key])
                        if key == "radius_index"
                        else _metric(float(row[key]))
                    )
                    for key in fields
                }
            )
    quality_pass = not failures and len(galaxy_summaries) == int(config["sample"]["exploration"])
    summary: dict[str, Any] = {
        "schema_version": "invariant-gravity-item5-pressure-support-extraction-1.0",
        "goal": config["goal"],
        "decision": (
            "PASS_ITEM5_EXPLORATION_REPRESENTATION_QUALITY"
            if quality_pass
            else "FAIL_ITEM5_EXPLORATION_REPRESENTATION_QUALITY"
        ),
        "counts": {
            "exploration_galaxies": int(config["sample"]["exploration"]),
            "quality_passing_galaxies": len(galaxy_summaries),
            "quality_failures": len(failures),
            "radial_rows": len(output_rows),
            "reserved_confirmation_predictor_member_accesses": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "failures": failures,
        "galaxies": [
            {
                "galaxy": row["galaxy"],
                "common_rows": row["common_rows"],
                "mean_absolute_local_pressure_fraction": _metric(
                    float(row["mean_absolute_local_pressure_fraction"])
                ),
            }
            for row in galaxy_summaries
        ],
        "pressure_stratum_definition": {
            "input_only": True,
            "median_mean_absolute_local_pressure_fraction": _metric(median_strength),
            "threshold_rule": "high if greater than or equal to the exploration median; low otherwise",
        },
        "leakage_boundary": {
            "support_features_finalized_before_target_interpolation": True,
            "support_feature_function_accepts_target": False,
            "Iorio_published_Vc_used_only_as_positive_control_response": True,
            "reserved_confirmation_target_accesses": 0,
        },
        "feature_table": {
            "path": config["feature_output"],
            "rows": len(output_rows),
            "sha256": _sha256_file(feature_path),
        },
    }
    summary["content_sha256"] = canonical_sha256(summary)
    summary_path = root / config["extraction_summary_output"]
    summary_path.write_bytes(canonical_json_bytes(summary) + b"\n")
    return feature_path, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "check-sample", "acquire-exploration", "extract-features"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--cache-dir", type=Path, default=Path("work/item5-pressure-support-v1-raw")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        print(write_sample_manifest(root))
        return 0
    if args.command == "acquire-exploration":
        print(acquire_exploration(root, cache_dir=args.cache_dir))
        return 0
    if args.command == "extract-features":
        for path in extract_features(root, cache_dir=args.cache_dir):
            print(path)
        return 0
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    validate_sample_manifest(stored, config)
    if build_sample_manifest(root) != stored:
        raise GravityItem5PressureSupportError("sample is not an exact rebuild")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
