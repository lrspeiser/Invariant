"""Frozen source and dimensionless derivation for gravity-roadmap Item 3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from astropy.cosmology import FlatLambdaCDM

from .gravity_g4_cluster_lensing_exploration import (
    G_DAGGER,
)
from .gravity_g4_cluster_lensing_exploration import (
    load_config as load_cluster_config,
)
from .gravity_g4_cluster_lensing_exploration import (
    prepare_packets as prepare_cluster_packets,
)
from .gravity_g4_photometric_law_construction import prepare_photometric_packets
from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item3_surface_volume_density.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "fresh-group-sample-manifest.json"
)
SOURCE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "fresh-group-exploration-source-manifest.json"
)
GROUP_FEATURE_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "fresh-group-features.tsv"
)
CROSS_SCALE_FEATURE_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "cross-scale-features.tsv"
)
EXTRACTION_SUMMARY_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "extraction-summary.json"
)
FREEZE_COMMIT = "ea75fbb44225811d676444251e4f85bbc064873b"


class GravityItem3SurfaceVolumeDensityError(RuntimeError):
    """Raised when the frozen derivation, sample, or target boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item3-surface-volume-density-config-1.0"
    ):
        raise GravityItem3SurfaceVolumeDensityError("unexpected Item 3 config schema")
    if _sha256_file(root / config["roadmap_binding"]["path"]) != config["roadmap_binding"][
        "file_sha256"
    ]:
        raise GravityItem3SurfaceVolumeDensityError("stable roadmap binding changed")
    predecessor_path = root / config["predecessor"]["path"]
    if _sha256_file(predecessor_path) != config["predecessor"]["file_sha256"]:
        raise GravityItem3SurfaceVolumeDensityError("Item 2 synthesis file changed")
    predecessor = json.loads(predecessor_path.read_text(encoding="utf-8"))
    if predecessor.get("content_sha256") != config["predecessor"]["content_sha256"]:
        raise GravityItem3SurfaceVolumeDensityError("Item 2 synthesis content changed")
    if predecessor.get("decision") != config["predecessor"]["required_decision"]:
        raise GravityItem3SurfaceVolumeDensityError("Item 2 synthesis decision changed")
    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem3SurfaceVolumeDensityError("paid model calls are forbidden")
    if authorization["reserved_confirmation_group_member_rows_allowed"]:
        raise GravityItem3SurfaceVolumeDensityError("confirmation member access is forbidden")
    if config["fresh_group_lane"]["reserved_confirmation_target_accesses_allowed"] != 0:
        raise GravityItem3SurfaceVolumeDensityError("confirmation access budget must be zero")
    group_lane = config["fresh_group_lane"]
    if _sha256_file(root / group_lane["metadata_path"]) != group_lane["metadata_file_sha256"]:
        raise GravityItem3SurfaceVolumeDensityError("metadata-only group source changed")
    exclusion_path = root / group_lane["item2_exclusion_manifest"]
    if _sha256_file(exclusion_path) != group_lane["item2_exclusion_file_sha256"]:
        raise GravityItem3SurfaceVolumeDensityError("Item 2 group exclusion file changed")
    exclusion = json.loads(exclusion_path.read_text(encoding="utf-8"))
    if exclusion.get("content_sha256") != group_lane["item2_exclusion_content_sha256"]:
        raise GravityItem3SurfaceVolumeDensityError("Item 2 group exclusion content changed")
    return config


def surface_volume_profile_features(
    radius: np.ndarray, gbar: np.ndarray, g_dagger: float
) -> dict[str, float]:
    """Build dual Poisson-source transition features without a dynamics target."""

    radius = np.asarray(radius, dtype=np.float64)
    gbar = np.asarray(gbar, dtype=np.float64)
    if (
        radius.ndim != 1
        or radius.shape != gbar.shape
        or radius.size < 5
        or np.any(~np.isfinite(radius))
        or np.any(~np.isfinite(gbar))
        or np.any(radius <= 0)
        or np.any(gbar <= 0)
        or np.any(np.diff(radius) <= 0)
        or not math.isfinite(g_dagger)
        or g_dagger <= 0
    ):
        raise GravityItem3SurfaceVolumeDensityError("invalid density profile")
    log_radius = np.log(radius)
    log_mass_equivalent = np.log(gbar) + 2.0 * log_radius
    local_dimension_raw = np.gradient(log_mass_equivalent, log_radius, edge_order=2)
    local_dimension = np.clip(local_dimension_raw, 0.25, 4.0)
    u_surface = gbar / g_dagger
    u_volume = local_dimension * gbar / (3.0 * g_dagger)
    spacing = np.gradient(log_radius)

    def transition(source: np.ndarray) -> tuple[float, float, float, np.ndarray]:
        kernel = source / (1.0 + source) ** 2
        weighted = kernel * spacing
        area = float(np.sum(weighted))
        if not math.isfinite(area) or area <= 1.0e-12:
            raise GravityItem3SurfaceVolumeDensityError("degenerate density transition")
        center = float(np.sum(weighted * log_radius) / area)
        width = math.sqrt(float(np.sum(weighted * (log_radius - center) ** 2) / area))
        if width <= 0:
            raise GravityItem3SurfaceVolumeDensityError("zero density-transition width")
        return area, center, width, weighted

    surface_area, surface_center, surface_width, surface_weight = transition(u_surface)
    volume_area, volume_center, volume_width, volume_weight = transition(u_volume)
    overlap = float(
        np.sum(np.sqrt(surface_weight * volume_weight))
        / math.sqrt(surface_area * volume_area)
    )
    values = {
        "clipped_local_dimension_fraction": float(
            np.mean(local_dimension != local_dimension_raw)
        ),
        "local_mass_dimension_iqr": float(
            np.quantile(local_dimension, 0.75) - np.quantile(local_dimension, 0.25)
        ),
        "local_mass_dimension_median": float(np.median(local_dimension)),
        "log_transition_radius_ratio": volume_center - surface_center,
        "log_transition_width_ratio": math.log(volume_width / surface_width),
        "mean_log_surface_source": float(np.mean(np.log(u_surface))),
        "mean_log_volume_source": float(np.mean(np.log(u_volume))),
        "surface_volume_log_ratio_median": float(
            np.median(np.log(u_surface / u_volume))
        ),
        "transition_area_asymmetry": (surface_area - volume_area)
        / (surface_area + volume_area),
        "transition_overlap_cosine": overlap,
    }
    if any(not math.isfinite(value) for value in values.values()):
        raise GravityItem3SurfaceVolumeDensityError("non-finite density feature")
    return values


def _read_group_metadata(path: Path) -> list[dict[str, Any]]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = "Group\tNmemb\tzsp\tLR195\tD10"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem3SurfaceVolumeDensityError("group metadata header changed") from exc
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 5 or not fields[0].strip().isdigit():
            continue
        try:
            rows.append(
                {
                    "group": int(fields[0]),
                    "members": int(fields[1]),
                    "redshift": float(fields[2]),
                    "lr195": float(fields[3]),
                    "d10": float(fields[4]),
                }
            )
        except ValueError:
            continue
    return rows


def _richness_bin(members: int, bins: Sequence[Sequence[int]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if int(lower) <= members <= int(upper):
            return index
    return None


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    lane = config["fresh_group_lane"]
    rows = _read_group_metadata(root / lane["metadata_path"])
    lower_z, upper_z = (float(value) for value in lane["redshift_range"])
    eligible: list[dict[str, Any]] = []
    for raw in rows:
        richness_bin = _richness_bin(int(raw["members"]), lane["richness_bins"])
        if richness_bin is None or int(raw["members"]) < int(lane["minimum_members"]):
            continue
        if not lower_z <= float(raw["redshift"]) < upper_z:
            continue
        if float(raw["lr195"]) <= 0 or not math.isfinite(float(raw["d10"])):
            continue
        eligible.append({**raw, "richness_bin": richness_bin})
    if len(eligible) != int(lane["expected_eligible_before_item2_exclusion"]):
        raise GravityItem3SurfaceVolumeDensityError("pre-exclusion eligible count changed")
    exclusion = json.loads(
        (root / lane["item2_exclusion_manifest"]).read_text(encoding="utf-8")
    )
    excluded_ids = {int(row["group"]) for row in exclusion["objects"]}
    remaining = [row for row in eligible if int(row["group"]) not in excluded_ids]
    if len(remaining) != int(lane["expected_eligible_after_item2_exclusion"]):
        raise GravityItem3SurfaceVolumeDensityError("post-exclusion eligible count changed")
    by_stratum: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    salt = str(lane["selection_salt"])
    for row in remaining:
        digest = hashlib.sha256(
            f"{salt}|richness-{row['richness_bin']}|group-{row['group']}".encode()
        ).hexdigest()
        by_stratum[int(row["richness_bin"])].append({**row, "selection_digest": digest})
    if [len(by_stratum[index]) for index in range(3)] != lane[
        "expected_remaining_by_richness_bin"
    ]:
        raise GravityItem3SurfaceVolumeDensityError("remaining richness counts changed")
    quota = lane["per_richness_bin"]
    exploration_count = int(quota["exploration"])
    confirmation_count = int(quota["reserved_confirmation"])
    objects: list[dict[str, Any]] = []
    for richness_bin in range(3):
        ordered = sorted(
            by_stratum[richness_bin],
            key=lambda row: (row["selection_digest"], int(row["group"])),
        )
        for index, row in enumerate(ordered[: exploration_count + confirmation_count]):
            role = "exploration" if index < exploration_count else "reserved_confirmation"
            objects.append(
                {
                    "d10": _metric(row["d10"]),
                    "group": int(row["group"]),
                    "lr195": _metric(row["lr195"]),
                    "members": int(row["members"]),
                    "redshift": _metric(row["redshift"]),
                    "richness_bin": richness_bin,
                    "role": role,
                    "selection_digest": row["selection_digest"],
                }
            )
    objects.sort(key=lambda row: (int(row["richness_bin"]), row["role"], int(row["group"])))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-fresh-group-sample-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM3_TARGET_BLIND_FRESH_GROUP_SAMPLE",
        "source": {
            "catalog_id": lane["catalog_id"],
            "metadata_path": lane["metadata_path"],
            "metadata_file_sha256": _sha256_file(root / lane["metadata_path"]),
            "item2_exclusion_manifest": lane["item2_exclusion_manifest"],
            "item2_exclusion_file_sha256": _sha256_file(
                root / lane["item2_exclusion_manifest"]
            ),
        },
        "selection_boundary": {
            "item2_group_ids_excluded": len(excluded_ids),
            "fresh_member_rows_opened": 0,
            "fresh_member_redshifts_read": 0,
            "reserved_confirmation_target_accesses": 0,
            "published_group_velocity_columns_read": 0,
        },
        "counts": {
            "eligible_before_item2_exclusion": len(eligible),
            "eligible_after_item2_exclusion": len(remaining),
            "exploration_groups": sum(row["role"] == "exploration" for row in objects),
            "reserved_confirmation_groups": sum(
                row["role"] == "reserved_confirmation" for row in objects
            ),
        },
        "strata": {
            str(index): {
                "eligible_after_item2_exclusion": len(by_stratum[index]),
                "exploration": sum(
                    row["richness_bin"] == index and row["role"] == "exploration"
                    for row in objects
                ),
                "reserved_confirmation": sum(
                    row["richness_bin"] == index
                    and row["role"] == "reserved_confirmation"
                    for row in objects
                ),
            }
            for index in range(3)
        },
        "objects": objects,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "item2_groups_reused": False,
            "member_response_seen_during_selection": False,
            "roadmap_item_3_complete": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_sample_manifest(manifest, config=config)
    return manifest


def validate_sample_manifest(
    manifest: Mapping[str, Any], *, config: Mapping[str, Any]
) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem3SurfaceVolumeDensityError("sample content hash changed")
    if manifest.get("decision") != "PASS_ITEM3_TARGET_BLIND_FRESH_GROUP_SAMPLE":
        raise GravityItem3SurfaceVolumeDensityError("fresh group selection did not pass")
    lane = config["fresh_group_lane"]
    expected = {
        "eligible_before_item2_exclusion": lane["expected_eligible_before_item2_exclusion"],
        "eligible_after_item2_exclusion": lane["expected_eligible_after_item2_exclusion"],
        "exploration_groups": lane["expected_exploration_groups"],
        "reserved_confirmation_groups": lane["expected_reserved_confirmation_groups"],
    }
    if manifest["counts"] != expected:
        raise GravityItem3SurfaceVolumeDensityError("fresh sample counts changed")
    objects = manifest["objects"]
    if len({int(row["group"]) for row in objects}) != len(objects):
        raise GravityItem3SurfaceVolumeDensityError("fresh group IDs are not unique")
    roles = Counter(row["role"] for row in objects)
    if roles != {
        "exploration": lane["expected_exploration_groups"],
        "reserved_confirmation": lane["expected_reserved_confirmation_groups"],
    }:
        raise GravityItem3SurfaceVolumeDensityError("fresh role counts changed")
    if manifest["selection_boundary"] != {
        "item2_group_ids_excluded": 270,
        "fresh_member_rows_opened": 0,
        "fresh_member_redshifts_read": 0,
        "reserved_confirmation_target_accesses": 0,
        "published_group_velocity_columns_read": 0,
    }:
        raise GravityItem3SurfaceVolumeDensityError("fresh sample leakage boundary changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem3SurfaceVolumeDensityError("fresh sample contains an overclaim")


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = build_sample_manifest(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def _load_sample(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config=config)
    return sample


def _download_member_query(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GravityItem3SurfaceVolumeDensityError(f"member query failed: {url}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def parse_member_payload(payload: bytes, *, expected_group: int) -> list[dict[str, Any]]:
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise GravityItem3SurfaceVolumeDensityError("member response is not UTF-8") from exc
    header = "Group\tGalID\tSpecObjID\tRAJ2000\tDEJ2000\tzsp\tLr"
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem3SurfaceVolumeDensityError(
            f"member response header changed for group {expected_group}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 7 or not fields[0].strip().isdigit():
            continue
        try:
            row = {
                "group": int(fields[0]),
                "galaxy_id": int(fields[1]),
                "specobjid": int(fields[2]),
                "ra_deg": float(fields[3]),
                "dec_deg": float(fields[4]),
                "member_redshift": float(fields[5]),
                "luminosity": float(fields[6]),
            }
        except ValueError:
            continue
        if row["group"] != expected_group:
            raise GravityItem3SurfaceVolumeDensityError("query returned another group")
        rows.append(row)
    if not rows or len({row["galaxy_id"] for row in rows}) != len(rows):
        raise GravityItem3SurfaceVolumeDensityError(f"invalid member rows: {expected_group}")
    return rows


def _acquire_one(
    row: Mapping[str, Any], *, config: Mapping[str, Any], cache_dir: Path
) -> dict[str, Any]:
    group = int(row["group"])
    url = str(config["fresh_group_lane"]["member_query_template"]).format(group=group)
    path = cache_dir / f"members-{group}.tsv"
    payload = _download_member_query(url, path)
    members = parse_member_payload(payload, expected_group=group)
    if len(members) != int(row["members"]):
        raise GravityItem3SurfaceVolumeDensityError(
            f"member count differs from frozen metadata: {group}"
        )
    return {
        "bytes": len(payload),
        "group": group,
        "member_rows": len(members),
        "sha256": _sha256_bytes(payload),
        "url": url,
    }


def acquire_fresh_exploration(
    root: Path, *, cache_dir: Path, workers: int = 8
) -> Path:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation = {
        int(row["group"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if any(int(row["group"]) in confirmation for row in exploration):
        raise GravityItem3SurfaceVolumeDensityError("fresh sample roles overlap")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_acquire_one, row, config=config, cache_dir=cache_dir): int(row["group"])
            for row in exploration
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - report every immutable source failure
                errors.append(f"{group}: {exc}")
    if errors:
        raise GravityItem3SurfaceVolumeDensityError("; ".join(sorted(errors)))
    records.sort(key=lambda row: int(row["group"]))
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-fresh-group-source-1.0",
        "goal": config["goal"],
        "decision": "PASS_ITEM3_FRESH_EXPLORATION_SOURCE_ACQUISITION",
        "preregistration": {
            "git_commit": FREEZE_COMMIT,
            "fresh_member_rows_opened_before_commit": 0,
        },
        "sample_binding": {
            "path": config["sample_manifest_output"],
            "file_sha256": _sha256_file(root / config["sample_manifest_output"]),
            "content_sha256": sample["content_sha256"],
        },
        "boundary": {
            "fresh_exploration_groups_acquired": len(records),
            "fresh_exploration_target_accesses": len(records),
            "item2_group_target_reuse": 0,
            "published_group_velocity_columns_read": 0,
            "reserved_confirmation_groups_acquired": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "counts": {
            "bytes": sum(int(row["bytes"]) for row in records),
            "groups": len(records),
            "member_rows": sum(int(row["member_rows"]) for row in records),
        },
        "records": records,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "roadmap_item_3_complete": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_source_manifest(manifest, sample=sample)
    path = root / config["source_manifest_output"]
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def validate_source_manifest(
    manifest: Mapping[str, Any], *, sample: Mapping[str, Any]
) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem3SurfaceVolumeDensityError("source manifest hash changed")
    if manifest.get("decision") != "PASS_ITEM3_FRESH_EXPLORATION_SOURCE_ACQUISITION":
        raise GravityItem3SurfaceVolumeDensityError("fresh source acquisition did not pass")
    boundary = manifest["boundary"]
    if (
        boundary["reserved_confirmation_groups_acquired"] != 0
        or boundary["reserved_confirmation_target_accesses"] != 0
        or boundary["item2_group_target_reuse"] != 0
        or boundary["published_group_velocity_columns_read"] != 0
    ):
        raise GravityItem3SurfaceVolumeDensityError("fresh source boundary changed")
    expected = {
        int(row["group"]) for row in sample["objects"] if row["role"] == "exploration"
    }
    if {int(row["group"]) for row in manifest["records"]} != expected:
        raise GravityItem3SurfaceVolumeDensityError("fresh source IDs changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem3SurfaceVolumeDensityError("source manifest contains overclaim")


def _weighted_quantile_radius(
    radius: np.ndarray, luminosity: np.ndarray, quantile: float
) -> float:
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(luminosity[order])
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(radius[order[min(index, radius.size - 1)]])


def measure_group_density_only(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    luminosity: np.ndarray,
    metadata_redshift: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Measure density features without accepting member redshifts or dynamics."""

    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    light = np.asarray(luminosity, dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(light) & (light > 0)
    ra, dec, light = ra[valid], dec[valid], light[valid]
    if ra.size < int(config["fresh_group_lane"]["minimum_members"]):
        raise GravityItem3SurfaceVolumeDensityError("insufficient finite density members")
    angle = np.deg2rad(ra)
    mean_ra = math.atan2(
        float(np.sum(light * np.sin(angle))), float(np.sum(light * np.cos(angle)))
    )
    delta_ra = np.angle(np.exp(1j * (angle - mean_ra)))
    mean_dec = float(np.sum(light * np.deg2rad(dec)) / np.sum(light))
    cosmology = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=2.725)
    distance_kpc = float(cosmology.angular_diameter_distance(metadata_redshift).value) * 1000.0
    x = distance_kpc * math.cos(mean_dec) * delta_ra
    y = distance_kpc * (np.deg2rad(dec) - mean_dec)
    center_x = float(np.sum(light * x) / np.sum(light))
    center_y = float(np.sum(light * y) / np.sum(light))
    radius = np.hypot(x - center_x, y - center_y)
    radii = {
        quantile: _weighted_quantile_radius(radius, light, quantile)
        for quantile in (0.25, 0.5, 0.75, 0.9)
    }
    if not 0 < radii[0.25] < radii[0.5] < radii[0.75] < radii[0.9]:
        raise GravityItem3SurfaceVolumeDensityError("non-strict luminosity quantile radii")
    constants = config["constants"]
    mass_total = (
        float(np.sum(light))
        * float(constants["axes_member_luminosity_unit_lsun"])
        * float(constants["fixed_group_r_band_mass_to_light_msun_per_lsun"])
    )
    gravity = float(constants["gravity_constant_kpc_km2_s2_msun"])
    conversion = float(constants["speed_conversion_km2_s2_per_kpc_to_m_s2"])
    threshold = float(constants["transition_acceleration_m_s2"])

    def surface_source(quantile: float) -> float:
        return (
            gravity
            * quantile
            * mass_total
            / radii[quantile] ** 2
            * conversion
            / threshold
        )

    u_surface25 = surface_source(0.25)
    u_surface50 = surface_source(0.5)
    u_surface75 = surface_source(0.75)
    u_volume = (
        gravity
        * radii[0.5]
        * (0.75 - 0.25)
        * mass_total
        / (radii[0.75] ** 3 - radii[0.25] ** 3)
        * conversion
        / threshold
    )
    u_volume_inner = (
        gravity
        * radii[0.25]
        * 0.5
        * mass_total
        / radii[0.5] ** 3
        * conversion
        / threshold
    )
    outer_radius = math.sqrt(radii[0.5] * radii[0.9])
    u_volume_outer = (
        gravity
        * outer_radius
        * (0.9 - 0.5)
        * mass_total
        / (radii[0.9] ** 3 - radii[0.5] ** 3)
        * conversion
        / threshold
    )
    sources = (
        u_surface25,
        u_surface50,
        u_surface75,
        u_volume,
        u_volume_inner,
        u_volume_outer,
    )
    if any(not math.isfinite(value) or value <= 0 for value in sources):
        raise GravityItem3SurfaceVolumeDensityError("invalid group density source")
    log_surface = math.log10(u_surface50)
    log_volume = math.log10(u_volume)
    return {
        "log10_r25_kpc": math.log10(radii[0.25]),
        "log10_r50_kpc": math.log10(radii[0.5]),
        "log10_r75_kpc": math.log10(radii[0.75]),
        "log10_r90_kpc": math.log10(radii[0.9]),
        "log10_total_member_luminosity": math.log10(mass_total),
        "log10_u_surface50": log_surface,
        "log10_u_volume25_75": log_volume,
        "surface_volume_log_contrast": log_surface - log_volume,
        "log_geometric_mean_sources": 0.5 * (log_surface + log_volume),
        "source_balance": 4.0 * u_surface50 * u_volume / (u_surface50 + u_volume) ** 2,
        "surface_source_radial_gradient": math.log(u_surface75 / u_surface25)
        / math.log(radii[0.75] / radii[0.25]),
        "volume_source_inner_outer_gradient": math.log(u_volume_outer / u_volume_inner)
        / math.log(outer_radius / radii[0.25]),
        "r50_kpc": radii[0.5],
        "r90_kpc": radii[0.9],
        "total_member_luminosity_catalog_units": float(np.sum(light)),
    }


def _gapper_dispersion(velocity: np.ndarray) -> float:
    ordered = np.sort(velocity)
    count = ordered.size
    gaps = np.diff(ordered)
    indices = np.arange(1, count, dtype=np.float64)
    return float(
        math.sqrt(math.pi)
        * np.sum(indices * (count - indices) * gaps)
        / (count * (count - 1))
    )


def measure_group_response_only(
    member_redshift: np.ndarray, density: Mapping[str, float]
) -> dict[str, float]:
    redshift = np.asarray(member_redshift, dtype=np.float64)
    redshift = redshift[np.isfinite(redshift)]
    if redshift.size < 10 or np.unique(redshift).size < 8:
        raise GravityItem3SurfaceVolumeDensityError("insufficient fresh member redshifts")
    median = float(np.median(redshift))
    velocity = 299792.458 * (redshift - median) / (1.0 + median)
    gapper = _gapper_dispersion(velocity)
    mad = 1.4826 * float(np.median(np.abs(velocity - np.median(velocity))))
    if gapper <= 0 or mad <= 0:
        raise GravityItem3SurfaceVolumeDensityError("nonpositive fresh group dispersion")
    light = float(density["total_member_luminosity_catalog_units"])
    return {
        "log10_sigma_gap": math.log10(gapper),
        "log10_sigma_mad": math.log10(mad),
        "log10_eta_r50": math.log10(gapper * gapper * float(density["r50_kpc"]) / light),
        "log10_eta_r90": math.log10(gapper * gapper * float(density["r90_kpc"]) / light),
        "sigma_gap_km_s": gapper,
        "sigma_mad_km_s": mad,
        "unique_member_redshifts": int(np.unique(redshift).size),
    }


def extract_features(root: Path, *, cache_dir: Path) -> tuple[Path, Path, Path]:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    source_manifest = json.loads(
        (root / config["source_manifest_output"]).read_text(encoding="utf-8")
    )
    validate_source_manifest(source_manifest, sample=sample)
    source_by_group = {int(row["group"]): row for row in source_manifest["records"]}
    group_rows: list[dict[str, Any]] = []
    group_failures: list[dict[str, Any]] = []
    for object_row in sample["objects"]:
        if object_row["role"] != "exploration":
            continue
        group = int(object_row["group"])
        path = cache_dir / f"members-{group}.tsv"
        if _sha256_file(path) != source_by_group[group]["sha256"]:
            raise GravityItem3SurfaceVolumeDensityError(f"fresh source hash changed: {group}")
        members = parse_member_payload(path.read_bytes(), expected_group=group)
        try:
            density = measure_group_density_only(
                np.asarray([row["ra_deg"] for row in members]),
                np.asarray([row["dec_deg"] for row in members]),
                np.asarray([row["luminosity"] for row in members]),
                float(object_row["redshift"]),
                config,
            )
            response = measure_group_response_only(
                np.asarray([row["member_redshift"] for row in members]), density
            )
        except GravityItem3SurfaceVolumeDensityError as exc:
            group_failures.append({"group": group, "reason": str(exc)})
            continue
        group_rows.append(
            {
                "group": group,
                "richness_bin": int(object_row["richness_bin"]),
                "members": int(object_row["members"]),
                "metadata_redshift": float(object_row["redshift"]),
                "d10": float(object_row["d10"]),
                **density,
                **response,
            }
        )
    group_rows.sort(key=lambda row: int(row["group"]))
    group_fields = [
        "group",
        "richness_bin",
        "members",
        "metadata_redshift",
        "d10",
        "log10_total_member_luminosity",
        "log10_r25_kpc",
        "log10_r50_kpc",
        "log10_r75_kpc",
        "log10_r90_kpc",
        "log10_u_surface50",
        "log10_u_volume25_75",
        "surface_volume_log_contrast",
        "log_geometric_mean_sources",
        "source_balance",
        "surface_source_radial_gradient",
        "volume_source_inner_outer_gradient",
        "sigma_gap_km_s",
        "sigma_mad_km_s",
        "unique_member_redshifts",
        "log10_sigma_gap",
        "log10_sigma_mad",
        "log10_eta_r50",
        "log10_eta_r90",
    ]
    group_path = root / config["group_feature_output"]
    with group_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=group_fields, dialect="excel-tab")
        writer.writeheader()
        for row in group_rows:
            writer.writerow(
                {
                    key: int(row[key])
                    if key in {"group", "richness_bin", "members", "unique_member_redshifts"}
                    else _metric(float(row[key]))
                    for key in group_fields
                }
            )

    cross_rows: list[dict[str, Any]] = []
    cross_failures: list[dict[str, Any]] = []
    for packet in sorted(prepare_photometric_packets(root), key=lambda row: row["galaxy"].name):
        try:
            radius = np.asarray(packet["arrays"]["radius"], dtype=np.float64)
            gbar = G_DAGGER * np.exp(np.asarray(packet["features"]["log_y"], dtype=np.float64))
            features = surface_volume_profile_features(radius, gbar, G_DAGGER)
        except GravityItem3SurfaceVolumeDensityError as exc:
            cross_failures.append(
                {"domain": "galaxy", "name": packet["galaxy"].name, "reason": str(exc)}
            )
            continue
        cross_rows.append(
            {"domain": "galaxy", "name": packet["galaxy"].name, **features}
        )
    cluster_config = load_cluster_config(root)
    for packet in prepare_cluster_packets(root, cluster_config):
        try:
            features = surface_volume_profile_features(
                np.asarray(packet["arrays"]["radius"], dtype=np.float64),
                np.asarray(packet["gbar"], dtype=np.float64),
                G_DAGGER,
            )
        except GravityItem3SurfaceVolumeDensityError as exc:
            cross_failures.append(
                {"domain": "cluster", "name": packet["cluster"], "reason": str(exc)}
            )
            continue
        cross_rows.append({"domain": "cluster", "name": packet["cluster"], **features})
    cross_rows.sort(key=lambda row: (str(row["domain"]), str(row["name"])))
    cross_fields = [
        "domain",
        "name",
        "mean_log_surface_source",
        "mean_log_volume_source",
        "local_mass_dimension_median",
        "local_mass_dimension_iqr",
        "surface_volume_log_ratio_median",
        "log_transition_radius_ratio",
        "transition_overlap_cosine",
        "transition_area_asymmetry",
        "log_transition_width_ratio",
        "clipped_local_dimension_fraction",
    ]
    cross_path = root / config["cross_scale_feature_output"]
    with cross_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=cross_fields, dialect="excel-tab")
        writer.writeheader()
        for row in cross_rows:
            writer.writerow(
                {
                    key: row[key] if key in {"domain", "name"} else _metric(float(row[key]))
                    for key in cross_fields
                }
            )
    summary: dict[str, Any] = {
        "schema_version": "invariant-gravity-item3-density-extraction-summary-1.0",
        "goal": config["goal"],
        "decision": (
            "PASS_ITEM3_EXPLORATION_REPRESENTATION_QUALITY"
            if not group_failures and not cross_failures and len(group_rows) == 120
            else "FAIL_ITEM3_EXPLORATION_REPRESENTATION_QUALITY"
        ),
        "counts": {
            "cross_scale_failures": len(cross_failures),
            "cross_scale_passing": len(cross_rows),
            "fresh_group_failures": len(group_failures),
            "fresh_group_passing": len(group_rows),
            "reserved_confirmation_target_accesses": 0,
        },
        "cross_scale_failures": cross_failures,
        "fresh_group_failures": group_failures,
        "leakage_boundary": {
            "cross_scale_feature_function_accepts_dynamics_target": False,
            "fresh_group_density_finalized_before_response_function": True,
            "fresh_group_density_function_accepts_member_redshift": False,
            "item2_group_target_reuse": 0,
            "reserved_confirmation_target_accesses": 0,
        },
        "artifacts": {
            "cross_scale_features": {
                "path": config["cross_scale_feature_output"],
                "sha256": _sha256_file(cross_path),
            },
            "group_features": {
                "path": config["group_feature_output"],
                "sha256": _sha256_file(group_path),
            },
        },
    }
    summary["content_sha256"] = canonical_sha256(summary)
    summary_path = root / EXTRACTION_SUMMARY_PATH
    summary_path.write_bytes(canonical_json_bytes(summary))
    return group_path, cross_path, summary_path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "check-sample", "acquire-fresh", "extract-features"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("work/item3-density-v1-raw"))
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        path = write_sample_manifest(root)
        print(path)
        return 0
    if args.command == "acquire-fresh":
        path = acquire_fresh_exploration(root, cache_dir=args.cache_dir, workers=args.workers)
        print(path)
        return 0
    if args.command == "extract-features":
        for path in extract_features(root, cache_dir=args.cache_dir):
            print(path)
        return 0
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    stored = json.loads(path.read_text(encoding="utf-8"))
    validate_sample_manifest(stored, config=config)
    if build_sample_manifest(root) != stored:
        raise GravityItem3SurfaceVolumeDensityError("fresh sample is not an exact rebuild")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
