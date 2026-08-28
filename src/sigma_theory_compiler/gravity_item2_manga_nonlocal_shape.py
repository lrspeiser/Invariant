"""Frozen MaNGA source and feature lane for gravity-roadmap Item 2 attempt 4.

The ``select`` command reads only morphology and photometry catalogs plus HTTP HEAD
responses.  It must not open a selected PCA mass map or DAP kinematic map.  Exploration
acquisition and feature construction are separate commands so the sample is sealed first.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.error
import urllib.request
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM
from astropy.io import fits

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item2_manga_nonlocal_shape.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4-source/"
    "manga-sample-manifest.json"
)
SOURCE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4-source/"
    "manga-exploration-source-manifest.json"
)
FEATURE_PATH = (
    "runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4-source/"
    "manga-exploration-features.tsv"
)


class GravityItem2MangaNonlocalShapeError(RuntimeError):
    """Raised when a frozen source, sample, or target-blind boundary drifts."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8").strip()
    return str(value).strip()


def _metric(value: float) -> str | int:
    if isinstance(value, int):
        return value
    return f"{float(value):.12e}"


def _manifest_value(value: Any) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _metric(value)
    return value


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / CONFIG_PATH
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != (
        "invariant-gravity-roadmap-item2-manga-nonlocal-shape-config-1.0"
    ):
        raise GravityItem2MangaNonlocalShapeError("unexpected Item 2 MaNGA config schema")
    roadmap = root / value["roadmap_binding"]["path"]
    predecessor = root / value["predecessor"]["path"]
    if _sha256_file(roadmap) != value["roadmap_binding"]["file_sha256"]:
        raise GravityItem2MangaNonlocalShapeError("stable roadmap binding changed")
    if _sha256_file(predecessor) != value["predecessor"]["file_sha256"]:
        raise GravityItem2MangaNonlocalShapeError("Item 2 predecessor file changed")
    predecessor_value = json.loads(predecessor.read_text(encoding="utf-8"))
    if predecessor_value.get("content_sha256") != value["predecessor"]["content_sha256"]:
        raise GravityItem2MangaNonlocalShapeError("Item 2 predecessor content changed")
    if predecessor_value.get("decision") != value["predecessor"]["required_decision"]:
        raise GravityItem2MangaNonlocalShapeError("Item 2 predecessor decision changed")
    authorization = value["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem2MangaNonlocalShapeError("paid calls are forbidden in this lane")
    if authorization["reserved_confirmation_kinematic_maps_allowed"]:
        raise GravityItem2MangaNonlocalShapeError("MaNGA confirmation access is not authorized")
    if value["target_blind_sample"]["reserved_confirmation_target_accesses_allowed"] != 0:
        raise GravityItem2MangaNonlocalShapeError("confirmation target access budget must be zero")
    return value


def _download_exact(url: str, path: Path, expected_sha256: str) -> bytes:
    if path.exists():
        payload = path.read_bytes()
    else:
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                payload = response.read()
        except (OSError, urllib.error.URLError) as exc:
            raise GravityItem2MangaNonlocalShapeError(f"catalog download failed: {url}") from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    if _sha256_bytes(payload) != expected_sha256:
        raise GravityItem2MangaNonlocalShapeError(f"catalog hash mismatch: {url}")
    return payload


def _candidate_digest(salt: str, purpose: str, *values: str) -> str:
    return hashlib.sha256("|".join((salt, purpose, *values)).encode("utf-8")).hexdigest()


def _axis_bin(axis_ratio: float, bins: Sequence[Sequence[float]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if float(lower) <= axis_ratio < float(upper):
            return index
    return None


def _eligible_candidates(
    morphology_path: Path,
    pymorph_path: Path,
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    sample = config["target_blind_sample"]
    with fits.open(morphology_path, memmap=False) as handle:
        morphology = handle[1].data
    with fits.open(pymorph_path, memmap=False) as handle:
        pymorph = handle[int(config["catalog_sources"]["pymorph"]["hdu"])].data
    pymorph_by_plateifu = {_text(row["PLATEIFU"]): row for row in pymorph}
    allowed_classes = {int(value) for value in sample["visual_classes"]}
    excluded = {str(value) for value in sample["excluded_schema_audit_plateifus"]}
    lower_z, upper_z = (float(value) for value in sample["redshift_range"])
    lower_q, upper_q = (float(value) for value in sample["axis_ratio_range"])
    minimum_radius = float(sample["minimum_i_band_sersic_half_light_major_axis_arcsec"])
    bins = sample["axis_ratio_bins"]
    salt = str(sample["selection_salt"])
    raw: list[dict[str, Any]] = []
    rejections: defaultdict[str, int] = defaultdict(int)
    for row in morphology:
        plateifu = _text(row["PLATEIFU"])
        if plateifu in excluded:
            rejections["schema_audit_exclusion"] += 1
            continue
        photo = pymorph_by_plateifu.get(plateifu)
        if photo is None:
            rejections["missing_pymorph_join"] += 1
            continue
        visual_class = int(row["Visual_Class"])
        if visual_class not in allowed_classes:
            rejections["visual_class"] += 1
            continue
        if int(row["Visual_Flag"]) != int(sample["visual_flag_required"]):
            rejections["visual_flag"] += 1
            continue
        redshift = float(row["Z"])
        if not math.isfinite(redshift) or not lower_z <= redshift <= upper_z:
            rejections["redshift"] += 1
            continue
        if int(photo["FLAG_FIT"]) == int(sample["pymorph_fit_may_not_equal"]):
            rejections["pymorph_fit"] += 1
            continue
        half_light = float(photo["A_HL_S"])
        axis_ratio = float(photo["BA_S"])
        sersic_index = float(photo["N_S"])
        if not math.isfinite(half_light) or half_light < minimum_radius:
            rejections["half_light_radius"] += 1
            continue
        if not math.isfinite(axis_ratio) or not lower_q <= axis_ratio <= upper_q:
            rejections["axis_ratio"] += 1
            continue
        if not math.isfinite(sersic_index) or sersic_index <= 0:
            rejections["sersic_index"] += 1
            continue
        axis_bin = _axis_bin(axis_ratio, bins)
        if axis_bin is None:
            rejections["axis_ratio_bin"] += 1
            continue
        manga_id = _text(row["MANGA_ID"])
        plate, ifu = plateifu.split("-", maxsplit=1)
        raw.append(
            {
                "axis_bin": axis_bin,
                "axis_ratio": axis_ratio,
                "half_light_major_axis_arcsec": half_light,
                "manga_id": manga_id,
                "plate": plate,
                "ifu": ifu,
                "plateifu": plateifu,
                "redshift": redshift,
                "sersic_index": sersic_index,
                "visual_class": visual_class,
            }
        )
    by_manga: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in raw:
        by_manga[row["manga_id"]].append(row)
    unique: list[dict[str, Any]] = []
    for manga_id, observations in by_manga.items():
        ordered = sorted(
            observations,
            key=lambda row: _candidate_digest(
                salt, "duplicate", manga_id, str(row["plateifu"])
            ),
        )
        unique.append(ordered[0])
        rejections["duplicate_observation"] += len(ordered) - 1
    return unique, dict(sorted(rejections.items()))


def _source_urls(row: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[str, str]:
    sources = config["catalog_sources"]
    values = {
        "ifu": str(row["ifu"]),
        "plate": str(row["plate"]),
        "plateifu": str(row["plateifu"]),
    }
    return (
        str(sources["pca_url_template"]).format(**values),
        str(sources["dap_maps_url_template"]).format(**values),
    )


def build_sample_manifest(
    root: Path,
    *,
    cache_dir: Path,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    catalogs = config["catalog_sources"]
    cache_dir = cache_dir.resolve()
    morphology_path = cache_dir / "manga-morphology-dl-DR17.fits"
    pymorph_path = cache_dir / "manga-pymorph-DR17.fits"
    _download_exact(
        catalogs["morphology"]["url"],
        morphology_path,
        catalogs["morphology"]["file_sha256"],
    )
    _download_exact(
        catalogs["pymorph"]["url"],
        pymorph_path,
        catalogs["pymorph"]["file_sha256"],
    )
    candidates, rejections = _eligible_candidates(morphology_path, pymorph_path, config)
    sample = config["target_blind_sample"]
    salt = str(sample["selection_salt"])
    per_exploration = int(sample["per_class_per_axis_bin"]["exploration"])
    per_confirmation = int(sample["per_class_per_axis_bin"]["reserved_confirmation"])
    strata: defaultdict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in candidates:
        strata[(int(row["visual_class"]), int(row["axis_bin"]))].append(row)
    selected: list[dict[str, Any]] = []
    for visual_class in sorted(int(value) for value in sample["visual_classes"]):
        for axis_bin in range(len(sample["axis_ratio_bins"])):
            ordered = sorted(
                strata[(visual_class, axis_bin)],
                key=lambda row: _candidate_digest(salt, "stratum", str(row["plateifu"])),
            )
            frozen = ordered[: per_exploration + per_confirmation]
            if len(frozen) != per_exploration + per_confirmation:
                raise GravityItem2MangaNonlocalShapeError(
                    f"insufficient catalog-eligible objects in stratum {visual_class}/{axis_bin}"
                )
            available: list[dict[str, Any]] = []
            for row in frozen:
                pca_url, dap_url = _source_urls(row, config)
                candidate = dict(row)
                candidate["pca_url"] = pca_url
                candidate["dap_maps_url"] = dap_url
                candidate["selection_digest"] = _candidate_digest(
                    salt, "stratum", str(row["plateifu"])
                )
                available.append(candidate)
            for ordinal, row in enumerate(available):
                row["role"] = "exploration" if ordinal < per_exploration else "reserved_confirmation"
                row["stratum_ordinal"] = ordinal
                selected.append(row)
    selected.sort(key=lambda row: (str(row["role"]), str(row["plateifu"])))
    role_counts = {
        role: sum(row["role"] == role for row in selected)
        for role in ("exploration", "reserved_confirmation")
    }
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item2-manga-sample-manifest-1.0",
        "goal": "TARGET_BLIND_MANGA_INTERMEDIATE_GEOMETRY_SAMPLE",
        "decision": "PASS_TARGET_BLIND_SAMPLE_SELECTION",
        "selection_boundary": {
            "catalog_columns_only": True,
            "selected_pca_files_opened": 0,
            "selected_dap_maps_opened": 0,
            "selected_kinematic_values_read": 0,
            "source_availability_method": "NOT_QUERIED_DURING_SELECTION",
            "reserved_confirmation_target_accesses": 0,
        },
        "counts": {
            "eligible_unique_objects": len(candidates),
            "source_endpoint_queries": 0,
            **role_counts,
        },
        "catalog_bindings": {
            "morphology": {
                "url": catalogs["morphology"]["url"],
                "sha256": _sha256_file(morphology_path),
            },
            "pymorph": {
                "url": catalogs["pymorph"]["url"],
                "sha256": _sha256_file(pymorph_path),
                "hdu": int(catalogs["pymorph"]["hdu"]),
            },
            "config": {
                "path": CONFIG_PATH,
                "sha256": _sha256_file(root / CONFIG_PATH),
            },
        },
        "rejections_before_source_availability": rejections,
        "objects": [
            {
                **{
                    key: _manifest_value(item)
                    for key, item in row.items()
                    if key not in {"pca_url", "dap_maps_url"}
                },
                "pca_url": row["pca_url"],
                "dap_maps_url": row["dap_maps_url"],
            }
            for row in selected
        ],
        "claims": {
            "confirmation_opened": False,
            "kinematic_response_seen_during_selection": False,
            "roadmap_item_2_complete": False,
            "alternative_to_gr_established": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_sample_manifest(manifest, config=config)
    return manifest


def validate_sample_manifest(value: Mapping[str, Any], *, config: Mapping[str, Any]) -> None:
    if value.get("schema_version") != "invariant-gravity-item2-manga-sample-manifest-1.0":
        raise GravityItem2MangaNonlocalShapeError("unexpected sample manifest schema")
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2MangaNonlocalShapeError("sample manifest content hash mismatch")
    if value.get("decision") != "PASS_TARGET_BLIND_SAMPLE_SELECTION":
        raise GravityItem2MangaNonlocalShapeError("target-blind sample selection did not pass")
    boundary = value["selection_boundary"]
    if not boundary["catalog_columns_only"]:
        raise GravityItem2MangaNonlocalShapeError("sample selection used non-catalog data")
    forbidden_counts = (
        "selected_pca_files_opened",
        "selected_dap_maps_opened",
        "selected_kinematic_values_read",
        "reserved_confirmation_target_accesses",
    )
    if any(int(boundary[key]) != 0 for key in forbidden_counts):
        raise GravityItem2MangaNonlocalShapeError("sample selection crossed a target boundary")
    objects = list(value["objects"])
    if len({row["plateifu"] for row in objects}) != len(objects):
        raise GravityItem2MangaNonlocalShapeError("duplicate plateifu in sample")
    if len({row["manga_id"] for row in objects}) != len(objects):
        raise GravityItem2MangaNonlocalShapeError("duplicate galaxy in sample")
    expected = {
        "exploration": int(config["target_blind_sample"]["expected_exploration_objects"]),
        "reserved_confirmation": int(
            config["target_blind_sample"]["expected_reserved_confirmation_objects"]
        ),
    }
    counts = {role: sum(row["role"] == role for row in objects) for role in expected}
    if counts != expected:
        raise GravityItem2MangaNonlocalShapeError("sample role count drift")
    by_stratum: defaultdict[tuple[int, int, str], int] = defaultdict(int)
    for row in objects:
        by_stratum[(int(row["visual_class"]), int(row["axis_bin"]), str(row["role"]))] += 1
    quotas = config["target_blind_sample"]["per_class_per_axis_bin"]
    for visual_class in (1, 2):
        for axis_bin in range(3):
            if by_stratum[(visual_class, axis_bin, "exploration")] != int(quotas["exploration"]):
                raise GravityItem2MangaNonlocalShapeError("exploration stratum imbalance")
            if by_stratum[(visual_class, axis_bin, "reserved_confirmation")] != int(
                quotas["reserved_confirmation"]
            ):
                raise GravityItem2MangaNonlocalShapeError("confirmation stratum imbalance")
    if any(bool(value["claims"][claim]) for claim in value["claims"]):
        raise GravityItem2MangaNonlocalShapeError("sample manifest contains an overclaim")


def write_sample_manifest(root: Path, manifest: Mapping[str, Any], output: Path) -> None:
    path = output if output.is_absolute() else root.resolve() / output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest) + b"\n")


def _download_source(url: str, path: Path) -> tuple[int, str]:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + ".part")
        request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
        error: Exception | None = None
        for _ in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response, temporary.open(
                    "wb"
                ) as handle:
                    while block := response.read(1024 * 1024):
                        handle.write(block)
                temporary.replace(path)
                error = None
                break
            except (OSError, urllib.error.URLError) as exc:
                error = exc
                temporary.unlink(missing_ok=True)
        if error is not None:
            raise GravityItem2MangaNonlocalShapeError(f"source download failed: {url}") from error
    payload_size = path.stat().st_size
    if payload_size <= 0:
        raise GravityItem2MangaNonlocalShapeError(f"empty source file: {url}")
    return payload_size, _sha256_file(path)


def _source_paths(cache_dir: Path, plateifu: str) -> tuple[Path, Path]:
    directory = cache_dir / plateifu
    return directory / f"mangapca-{plateifu}.fits", directory / f"manga-{plateifu}-MAPS.fits.gz"


def acquire_exploration_sources(
    root: Path,
    *,
    cache_dir: Path,
    workers: int = 6,
) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample_path = root / config["sample_manifest_output"]
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config=config)
    if not config["authorization"]["selected_exploration_kinematic_maps_allowed"]:
        raise GravityItem2MangaNonlocalShapeError("exploration source access is not authorized")
    objects = [row for row in sample["objects"] if row["role"] == "exploration"]
    if any(row["role"] != "exploration" for row in objects):
        raise GravityItem2MangaNonlocalShapeError("confirmation object entered acquisition")
    cache_dir = cache_dir.resolve()

    def acquire(row: Mapping[str, Any]) -> dict[str, Any]:
        plateifu = str(row["plateifu"])
        pca_path, dap_path = _source_paths(cache_dir, plateifu)
        pca_bytes, pca_sha = _download_source(str(row["pca_url"]), pca_path)
        dap_bytes, dap_sha = _download_source(str(row["dap_maps_url"]), dap_path)
        return {
            "plateifu": plateifu,
            "pca": {"bytes": pca_bytes, "sha256": pca_sha, "url": row["pca_url"]},
            "dap_maps": {
                "bytes": dap_bytes,
                "sha256": dap_sha,
                "url": row["dap_maps_url"],
            },
        }

    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        pending = {executor.submit(acquire, row): str(row["plateifu"]) for row in objects}
        for future in as_completed(pending):
            try:
                records.append(future.result())
            except Exception as exc:
                raise GravityItem2MangaNonlocalShapeError(
                    f"failed to acquire exploration source {pending[future]}"
                ) from exc
    records.sort(key=lambda row: row["plateifu"])
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item2-manga-source-manifest-1.0",
        "goal": "MANGA_EXPLORATION_SOURCE_ACQUISITION",
        "decision": "PASS_EXPLORATION_SOURCE_ACQUISITION",
        "sample_binding": {
            "path": config["sample_manifest_output"],
            "file_sha256": _sha256_file(sample_path),
            "content_sha256": sample["content_sha256"],
        },
        "boundary": {
            "roles_acquired": ["exploration"],
            "exploration_objects": len(records),
            "reserved_confirmation_objects_acquired": 0,
            "reserved_confirmation_target_accesses": 0,
            "paid_model_calls": 0,
        },
        "totals": {
            "files": 2 * len(records),
            "bytes": sum(
                int(record[kind]["bytes"])
                for record in records
                for kind in ("pca", "dap_maps")
            ),
        },
        "records": records,
        "claims": {
            "confirmation_opened": False,
            "roadmap_item_2_complete": False,
            "alternative_to_gr_established": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_source_manifest(manifest, config=config, sample=sample)
    return manifest


def validate_source_manifest(
    value: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    sample: Mapping[str, Any],
) -> None:
    if value.get("schema_version") != "invariant-gravity-item2-manga-source-manifest-1.0":
        raise GravityItem2MangaNonlocalShapeError("unexpected exploration source manifest schema")
    copy = dict(value)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2MangaNonlocalShapeError("source manifest content hash mismatch")
    boundary = value["boundary"]
    if boundary["roles_acquired"] != ["exploration"]:
        raise GravityItem2MangaNonlocalShapeError("source manifest crossed the sample role boundary")
    if int(boundary["reserved_confirmation_objects_acquired"]) != 0 or int(
        boundary["reserved_confirmation_target_accesses"]
    ) != 0:
        raise GravityItem2MangaNonlocalShapeError("confirmation sources were accessed")
    records = list(value["records"])
    expected = {
        str(row["plateifu"]) for row in sample["objects"] if row["role"] == "exploration"
    }
    if {str(row["plateifu"]) for row in records} != expected:
        raise GravityItem2MangaNonlocalShapeError("exploration source population drift")
    if len(records) != int(config["target_blind_sample"]["expected_exploration_objects"]):
        raise GravityItem2MangaNonlocalShapeError("exploration source count drift")
    if any(bool(value["claims"][claim]) for claim in value["claims"]):
        raise GravityItem2MangaNonlocalShapeError("source manifest contains an overclaim")


def _weighted_center(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray, mask: np.ndarray
) -> tuple[float, float]:
    total = float(np.sum(weights[mask]))
    if not math.isfinite(total) or total <= 0:
        raise GravityItem2MangaNonlocalShapeError("non-positive map weight")
    return float(np.sum(x[mask] * weights[mask]) / total), float(
        np.sum(y[mask] * weights[mask]) / total
    )


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    ordered_values = values[order]
    ordered_weights = weights[order]
    threshold = 0.5 * float(np.sum(ordered_weights))
    index = int(np.searchsorted(np.cumsum(ordered_weights), threshold, side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _ellipse_geometry(
    mass: np.ndarray,
    valid: np.ndarray,
    half_light_pixels: float,
    axis_ratio: float,
) -> dict[str, Any]:
    ny, nx = mass.shape
    y, x = np.indices((ny, nx), dtype=np.float64)
    initial_x = 0.5 * (nx - 1)
    initial_y = 0.5 * (ny - 1)
    initial_radius = np.hypot(x - initial_x, y - initial_y)
    recenter_mask = valid & (initial_radius <= half_light_pixels)
    if int(np.sum(recenter_mask)) < 5:
        raise GravityItem2MangaNonlocalShapeError("insufficient mass map for recentering")
    center_x, center_y = _weighted_center(x, y, mass, recenter_mask)
    dx = x - center_x
    dy = y - center_y
    orientation_mask = valid & (np.hypot(dx, dy) <= half_light_pixels)
    total = float(np.sum(mass[orientation_mask]))
    covariance = np.asarray(
        [
            [
                np.sum(mass[orientation_mask] * dx[orientation_mask] ** 2),
                np.sum(mass[orientation_mask] * dx[orientation_mask] * dy[orientation_mask]),
            ],
            [
                np.sum(mass[orientation_mask] * dx[orientation_mask] * dy[orientation_mask]),
                np.sum(mass[orientation_mask] * dy[orientation_mask] ** 2),
            ],
        ],
        dtype=np.float64,
    ) / total
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    major = eigenvectors[:, int(np.argmax(eigenvalues))]
    angle = math.atan2(float(major[1]), float(major[0]))
    cosine = math.cos(angle)
    sine = math.sin(angle)
    major_coordinate = cosine * dx + sine * dy
    minor_coordinate = -sine * dx + cosine * dy
    elliptical_radius = np.sqrt(major_coordinate**2 + (minor_coordinate / axis_ratio) ** 2)
    return {
        "angle": angle,
        "center_x": center_x,
        "center_y": center_y,
        "dx": dx,
        "dy": dy,
        "elliptical_radius": elliptical_radius,
        "x": x,
        "y": y,
    }


def _aperture_moments(
    mass: np.ndarray,
    valid: np.ndarray,
    geometry: Mapping[str, Any],
    radius_pixels: float,
) -> dict[str, Any]:
    mask = valid & (np.asarray(geometry["elliptical_radius"]) <= radius_pixels)
    if int(np.sum(mask)) < 3:
        raise GravityItem2MangaNonlocalShapeError("insufficient spaxels in nested aperture")
    x = np.asarray(geometry["x"])
    y = np.asarray(geometry["y"])
    center_x, center_y = _weighted_center(x, y, mass, mask)
    dx = np.asarray(geometry["dx"])[mask]
    dy = np.asarray(geometry["dy"])[mask]
    weights = mass[mask]
    radial = np.hypot(dx, dy)
    complex_position = dx + 1j * dy
    radial2 = float(np.sum(weights * radial**2))
    quadrupole_complex = np.sum(weights * complex_position**2)
    quadrupole = float(abs(quadrupole_complex) / radial2) if radial2 > 0 else 0.0
    total = float(np.sum(weights))
    normalized = complex_position / radius_pixels
    m3 = float(abs(np.sum(weights * normalized**3)) / total)
    m4 = float(abs(np.sum(weights * normalized**4)) / total)
    angle = 0.5 * math.atan2(float(quadrupole_complex.imag), float(quadrupole_complex.real))
    return {
        "center_dx": center_x - float(geometry["center_x"]),
        "center_dy": center_y - float(geometry["center_y"]),
        "m3": m3,
        "m4": m4,
        "mass": total,
        "position_angle": angle,
        "quadrupole": quadrupole,
        "spaxels": int(np.sum(mask)),
    }


def measure_shape_only(
    pca_path: Path,
    object_row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Read only the PCA baryonic product and return target-blind shape features."""

    with fits.open(pca_path, memmap=False) as handle:
        plateifu = str(handle[0].header["PLATEIFU"]).strip()
        if plateifu != str(object_row["plateifu"]):
            raise GravityItem2MangaNonlocalShapeError("PCA plateifu mismatch")
        mass_to_light = np.asarray(handle["MLi"].data[0], dtype=np.float64)
        log_luminosity = np.asarray(handle["LOG_LUM_I"].data, dtype=np.float64)
        snr = np.asarray(handle["SNRMED"].data, dtype=np.float64)
        source_mask = np.asarray(handle["MASK"].data, dtype=np.float64)
        success = np.asarray(handle["SUCCESS"].data, dtype=np.float64)
        good_fraction = np.asarray(handle["GOODFRAC"].data[2], dtype=np.float64)
    finite_luminosity = np.isfinite(log_luminosity)
    luminosity = np.zeros_like(log_luminosity)
    luminosity[finite_luminosity] = 10.0 ** log_luminosity[finite_luminosity]
    valid = (
        finite_luminosity
        & np.isfinite(mass_to_light)
        & (source_mask == 0)
        & (success == 1)
        & (
            good_fraction
            >= float(config["pca_stellar_mass_map"]["minimum_goodfrac_channel_2"])
        )
        & (snr >= float(config["pca_stellar_mass_map"]["minimum_snrmed"]))
    )
    mass = np.zeros_like(luminosity)
    mass[valid] = 10.0 ** (log_luminosity[valid] + mass_to_light[valid])
    half_light_pixels = float(object_row["half_light_major_axis_arcsec"]) / 0.5
    axis_ratio = float(object_row["axis_ratio"])
    geometry = _ellipse_geometry(mass, valid, half_light_pixels, axis_ratio)
    aperture = np.asarray(geometry["elliptical_radius"]) <= half_light_pixels
    valid_aperture = aperture & valid
    valid_spaxels = int(np.sum(valid_aperture))
    if valid_spaxels < int(config["pca_stellar_mass_map"]["minimum_valid_spaxels_in_aperture"]):
        raise GravityItem2MangaNonlocalShapeError("insufficient valid PCA aperture spaxels")
    total_luminosity = float(np.sum(luminosity[aperture & finite_luminosity]))
    valid_luminosity = float(np.sum(luminosity[valid_aperture]))
    coverage = valid_luminosity / total_luminosity if total_luminosity > 0 else 0.0
    if coverage < float(config["pca_stellar_mass_map"]["minimum_valid_mass_fraction_in_aperture"]):
        raise GravityItem2MangaNonlocalShapeError("insufficient PCA aperture coverage")
    fractions = [float(value) for value in config["shape_features"]["aperture_fractions"]]
    moments = [
        _aperture_moments(mass, valid, geometry, fraction * half_light_pixels)
        for fraction in fractions
    ]
    mass_aperture = float(np.sum(mass[valid_aperture]))
    if mass_aperture <= 0:
        raise GravityItem2MangaNonlocalShapeError("non-positive aperture stellar mass")
    q2 = np.asarray([row["quadrupole"] for row in moments], dtype=np.float64)
    m3 = np.asarray([row["m3"] for row in moments], dtype=np.float64)
    m4 = np.asarray([row["m4"] for row in moments], dtype=np.float64)
    centers = np.asarray([[row["center_dx"], row["center_dy"]] for row in moments])
    centroid_shift = float(np.sqrt(np.mean(np.sum(centers**2, axis=1))) / half_light_pixels)
    delta_angle = float(moments[-1]["position_angle"] - moments[1]["position_angle"])
    twist = abs(math.sin(2.0 * delta_angle)) * math.sqrt(float(q2[-1] * q2[1]))
    inner_center = centers[1]
    outer_center = centers[-1]
    center_denominator = float(np.linalg.norm(inner_center) * np.linalg.norm(outer_center))
    center_alignment = (
        float(np.dot(inner_center, outer_center) / center_denominator)
        if center_denominator > 1.0e-12
        else 0.0
    )
    roughness = float(
        np.mean(np.diff(q2) ** 2 + np.diff(m3) ** 2 + np.diff(m4) ** 2)
    )
    cosmology_spec = config["aperture_and_response"]["cosmology"]
    cosmology = FlatLambdaCDM(
        H0=float(cosmology_spec["H0_km_s_Mpc"]),
        Om0=float(cosmology_spec["Omega_m"]),
    )
    scale = float(cosmology.kpc_proper_per_arcmin(float(object_row["redshift"])).value / 60.0)
    circularized_radius = float(object_row["half_light_major_axis_arcsec"]) * math.sqrt(
        axis_ratio
    ) * scale
    features = {
        "axis_twist_inner_outer_sin2": twist,
        "centroid_shift_profile": centroid_shift,
        "inner_outer_centroid_alignment": center_alignment,
        "log10_circularized_radius_kpc": math.log10(circularized_radius),
        "log10_stellar_mass_aperture": math.log10(mass_aperture),
        "m3_1p0": float(m3[-1]),
        "m3_inner_outer_difference": float(m3[-1] - m3[1]),
        "m4_1p0": float(m4[-1]),
        "m4_inner_outer_difference": float(m4[-1] - m4[1]),
        "mass_concentration_0p25": float(moments[0]["mass"] / moments[-1]["mass"]),
        "outer_multipole_energy": float(math.sqrt(q2[-1] ** 2 + m3[-1] ** 2 + m4[-1] ** 2)),
        "profile_roughness_energy": roughness,
        "pymorph_axis_ratio": axis_ratio,
        "quadrupole_1p0": float(q2[-1]),
        "quadrupole_inner_outer_difference": float(q2[-1] - q2[1]),
        "quadrupole_profile_variance": float(np.var(q2)),
        "sersic_index": float(object_row["sersic_index"]),
        "visual_class_proxy": float(int(object_row["visual_class"]) - 1),
    }
    if any(not math.isfinite(value) for value in features.values()):
        raise GravityItem2MangaNonlocalShapeError("non-finite target-blind shape feature")
    context = {
        "aperture": aperture,
        "circularized_radius_kpc": circularized_radius,
        "half_light_major_axis_kpc": float(object_row["half_light_major_axis_arcsec"]) * scale,
        "luminosity": luminosity,
        "mass": mass,
        "mass_aperture": mass_aperture,
        "pca_valid": valid,
        "valid_luminosity_fraction": coverage,
        "valid_spaxels": valid_spaxels,
    }
    return features, context


def measure_response_only(
    dap_path: Path,
    object_row: Mapping[str, Any],
    context: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Read DAP kinematics only after the shape feature vector has been finalized."""

    with fits.open(dap_path, memmap=False) as handle:
        velocity = np.asarray(handle["STELLAR_VEL"].data, dtype=np.float64)
        velocity_ivar = np.asarray(handle["STELLAR_VEL_IVAR"].data, dtype=np.float64)
        velocity_mask = np.asarray(handle["STELLAR_VEL_MASK"].data)
        dispersion = np.asarray(handle["STELLAR_SIGMA"].data, dtype=np.float64)
        dispersion_ivar = np.asarray(handle["STELLAR_SIGMA_IVAR"].data, dtype=np.float64)
        dispersion_mask = np.asarray(handle["STELLAR_SIGMA_MASK"].data)
        correction = np.asarray(handle["STELLAR_SIGMACORR"].data[0], dtype=np.float64)
        binid = np.asarray(handle["BINID"].data[1])
    if velocity.shape != np.asarray(context["mass"]).shape:
        raise GravityItem2MangaNonlocalShapeError("PCA and DAP map shapes differ")
    aperture = np.asarray(context["aperture"], dtype=bool)
    pca_valid = np.asarray(context["pca_valid"], dtype=bool)
    luminosity = np.asarray(context["luminosity"], dtype=np.float64)
    mass = np.asarray(context["mass"], dtype=np.float64)
    valid = (
        aperture
        & pca_valid
        & (velocity_mask == 0)
        & (dispersion_mask == 0)
        & (velocity_ivar > 0)
        & (dispersion_ivar > 0)
        & np.isfinite(velocity)
        & np.isfinite(dispersion)
        & np.isfinite(correction)
        & (luminosity > 0)
    )
    unique_bins = len({int(value) for value in binid[valid] if int(value) >= 0})
    response_config = config["aperture_and_response"]
    if unique_bins < int(response_config["minimum_unique_stellar_kinematic_bins"]):
        raise GravityItem2MangaNonlocalShapeError("insufficient unique stellar kinematic bins")
    usable_fraction = float(np.sum(luminosity[valid]) / np.sum(luminosity[aperture & pca_valid]))
    if usable_fraction < float(response_config["minimum_usable_luminosity_fraction"]):
        raise GravityItem2MangaNonlocalShapeError("insufficient usable kinematic luminosity")
    weights = luminosity[valid]
    velocity_values = velocity[valid]
    zero_point = _weighted_median(velocity_values, weights)
    centered_velocity2 = (velocity_values - zero_point) ** 2
    corrected_dispersion2 = np.maximum(dispersion[valid] ** 2 - correction[valid] ** 2, 0.0)
    raw_dispersion2 = dispersion[valid] ** 2
    mass_weights = mass[valid]
    mean_corrected = float(np.sum(weights * (centered_velocity2 + corrected_dispersion2)) / np.sum(weights))
    mean_raw = float(np.sum(weights * (centered_velocity2 + raw_dispersion2)) / np.sum(weights))
    mean_mass_weighted = float(
        np.sum(mass_weights * (centered_velocity2 + corrected_dispersion2)) / np.sum(mass_weights)
    )
    gravity_constant = float(response_config["gravity_constant_kpc_km2_s2_msun"])
    stellar_mass = float(context["mass_aperture"])

    def eta(radius_kpc: float, velocity2: float) -> float:
        value = radius_kpc * velocity2 / (gravity_constant * stellar_mass)
        if not math.isfinite(value) or value <= 0:
            raise GravityItem2MangaNonlocalShapeError("non-positive aperture response")
        return value

    primary = eta(float(context["circularized_radius_kpc"]), mean_corrected)
    responses = {
        "log10_eta_ap": math.log10(primary),
        "log10_eta_major_axis": math.log10(
            eta(float(context["half_light_major_axis_kpc"]), mean_corrected)
        ),
        "log10_eta_mass_weighted": math.log10(
            eta(float(context["circularized_radius_kpc"]), mean_mass_weighted)
        ),
        "log10_eta_uncorrected_sigma": math.log10(
            eta(float(context["circularized_radius_kpc"]), mean_raw)
        ),
    }
    diagnostics = {
        "unique_kinematic_bins": unique_bins,
        "usable_luminosity_fraction": usable_fraction,
        "velocity_zero_point_km_s": zero_point,
    }
    return responses, diagnostics


FEATURE_COLUMNS = (
    "plateifu",
    "manga_id",
    "visual_class",
    "axis_bin",
    "axis_ratio",
    "redshift",
    "half_light_major_axis_arcsec",
    "valid_pca_spaxels",
    "valid_pca_luminosity_fraction",
    "unique_kinematic_bins",
    "usable_kinematic_luminosity_fraction",
    "log10_stellar_mass_aperture",
    "log10_circularized_radius_kpc",
    "sersic_index",
    "pymorph_axis_ratio",
    "visual_class_proxy",
    "mass_concentration_0p25",
    "centroid_shift_profile",
    "quadrupole_1p0",
    "m3_1p0",
    "m4_1p0",
    "quadrupole_inner_outer_difference",
    "quadrupole_profile_variance",
    "axis_twist_inner_outer_sin2",
    "m3_inner_outer_difference",
    "m4_inner_outer_difference",
    "outer_multipole_energy",
    "profile_roughness_energy",
    "inner_outer_centroid_alignment",
    "log10_eta_ap",
    "log10_eta_major_axis",
    "log10_eta_uncorrected_sigma",
    "log10_eta_mass_weighted",
)


def extract_exploration_features(root: Path, *, cache_dir: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve()
    config = load_config(root)
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config=config)
    source = json.loads((root / config["source_manifest_output"]).read_text(encoding="utf-8"))
    validate_source_manifest(source, config=config, sample=sample)
    source_by_plateifu = {str(row["plateifu"]): row for row in source["records"]}
    objects = [row for row in sample["objects"] if row["role"] == "exploration"]
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for object_row in objects:
        plateifu = str(object_row["plateifu"])
        pca_path, dap_path = _source_paths(cache_dir.resolve(), plateifu)
        binding = source_by_plateifu[plateifu]
        if _sha256_file(pca_path) != binding["pca"]["sha256"]:
            raise GravityItem2MangaNonlocalShapeError(f"PCA source drift: {plateifu}")
        if _sha256_file(dap_path) != binding["dap_maps"]["sha256"]:
            raise GravityItem2MangaNonlocalShapeError(f"DAP source drift: {plateifu}")
        try:
            features, context = measure_shape_only(pca_path, object_row, config)
            responses, diagnostics = measure_response_only(dap_path, object_row, context, config)
        except GravityItem2MangaNonlocalShapeError as exc:
            failures.append({"plateifu": plateifu, "reason": str(exc)})
            continue
        row: dict[str, Any] = {
            "axis_bin": int(object_row["axis_bin"]),
            "axis_ratio": float(object_row["axis_ratio"]),
            "half_light_major_axis_arcsec": float(object_row["half_light_major_axis_arcsec"]),
            "manga_id": str(object_row["manga_id"]),
            "plateifu": plateifu,
            "redshift": float(object_row["redshift"]),
            "unique_kinematic_bins": int(diagnostics["unique_kinematic_bins"]),
            "usable_kinematic_luminosity_fraction": float(
                diagnostics["usable_luminosity_fraction"]
            ),
            "valid_pca_luminosity_fraction": float(context["valid_luminosity_fraction"]),
            "valid_pca_spaxels": int(context["valid_spaxels"]),
            "visual_class": int(object_row["visual_class"]),
            **features,
            **responses,
        }
        rows.append(row)
    rows.sort(key=lambda row: str(row["plateifu"]))
    extraction: dict[str, Any] = {
        "schema_version": "invariant-gravity-item2-manga-extraction-summary-1.0",
        "decision": "PASS_EXPLORATION_EXTRACTION" if not failures else "FAIL_EXPLORATION_QUALITY",
        "counts": {
            "selected_exploration": len(objects),
            "quality_passing": len(rows),
            "quality_failures": len(failures),
            "reserved_confirmation_target_accesses": 0,
        },
        "failures": failures,
        "leakage_boundary": {
            "shape_function_accepts_dap_path": False,
            "shape_features_finalized_before_response_function": True,
            "forbidden_derived_mass_targets_read": 0,
            "reserved_confirmation_target_accesses": 0,
        },
    }
    extraction["content_sha256"] = canonical_sha256(extraction)
    return rows, extraction


def write_feature_table(root: Path, rows: Sequence[Mapping[str, Any]], output: Path) -> None:
    path = output if output.is_absolute() else root.resolve() / output
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FEATURE_COLUMNS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _manifest_value(row[key]) for key in FEATURE_COLUMNS})


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    select = subparsers.add_parser("select", help="seal the target-blind MaNGA sample")
    select.add_argument("--root", type=Path, default=Path.cwd())
    select.add_argument("--cache-dir", type=Path, required=True)
    select.add_argument("--output", type=Path, default=Path(SAMPLE_MANIFEST_PATH))
    check = subparsers.add_parser("check-sample", help="validate the sealed sample manifest")
    check.add_argument("--root", type=Path, default=Path.cwd())
    check.add_argument("--manifest", type=Path, default=Path(SAMPLE_MANIFEST_PATH))
    acquire = subparsers.add_parser(
        "acquire-exploration", help="download only the 60 sealed exploration source pairs"
    )
    acquire.add_argument("--root", type=Path, default=Path.cwd())
    acquire.add_argument("--cache-dir", type=Path, required=True)
    acquire.add_argument("--workers", type=int, default=6)
    acquire.add_argument("--output", type=Path, default=Path(SOURCE_MANIFEST_PATH))
    extract = subparsers.add_parser(
        "extract-exploration", help="derive target-blind shapes, then direct responses"
    )
    extract.add_argument("--root", type=Path, default=Path.cwd())
    extract.add_argument("--cache-dir", type=Path, required=True)
    extract.add_argument("--output", type=Path, default=Path(FEATURE_PATH))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        manifest = build_sample_manifest(root, cache_dir=args.cache_dir)
        write_sample_manifest(root, manifest, args.output)
    elif args.command == "check-sample":
        config = load_config(root)
        path = args.manifest if args.manifest.is_absolute() else root / args.manifest
        manifest = json.loads(path.read_text(encoding="utf-8"))
        validate_sample_manifest(manifest, config=config)
    elif args.command == "acquire-exploration":
        manifest = acquire_exploration_sources(
            root, cache_dir=args.cache_dir, workers=args.workers
        )
        path = args.output if args.output.is_absolute() else root / args.output
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(manifest) + b"\n")
    else:
        rows, extraction = extract_exploration_features(root, cache_dir=args.cache_dir)
        write_feature_table(root, rows, args.output)
        print(json.dumps(extraction, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
