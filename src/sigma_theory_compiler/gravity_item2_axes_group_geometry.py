"""Frozen AXES-SDSS group source lane for gravity-roadmap Item 2 attempt 5.

Selection reads a committed VizieR response containing only group identifiers, richness,
median redshift, optical luminosity, and environment.  It cannot read member redshifts or
published velocity/radius columns.  Exploration acquisition is a separate command so the
sample and its sealed confirmation role are immutable before any dynamics are opened.
"""

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
from scipy.sparse.csgraph import dijkstra, minimum_spanning_tree
from scipy.spatial.distance import cdist

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item2_axes_group_geometry.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-axes-group-geometry-v5-source/"
    "axes-group-sample-manifest.json"
)
SOURCE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-axes-group-geometry-v5-source/"
    "axes-group-exploration-source-manifest.json"
)
FEATURE_PATH = (
    "runs/gravity/roadmap/item-02-axes-group-geometry-v5-source/"
    "axes-group-exploration-features.tsv"
)
EXTRACTION_SUMMARY_PATH = (
    "runs/gravity/roadmap/item-02-axes-group-geometry-v5-source/"
    "axes-group-exploration-extraction-summary.json"
)
FREEZE_COMMIT = "e138203df70cefb6eb22aee7fd6e93fed2e95fa7"


class GravityItem2AxesGroupError(RuntimeError):
    """Raised when the frozen group source, sample, or leakage boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


def _load_sample(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    sample = json.loads((root / config["sample_manifest_output"]).read_text(encoding="utf-8"))
    validate_sample_manifest(sample, config=config)
    return sample


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = json.loads((root / CONFIG_PATH).read_text(encoding="utf-8"))
    if config.get("schema_version") != (
        "invariant-gravity-roadmap-item2-axes-group-geometry-config-1.0"
    ):
        raise GravityItem2AxesGroupError("unexpected AXES group config schema")
    roadmap = root / config["roadmap_binding"]["path"]
    predecessor = root / config["predecessor"]["path"]
    if _sha256_file(roadmap) != config["roadmap_binding"]["file_sha256"]:
        raise GravityItem2AxesGroupError("stable roadmap binding changed")
    if _sha256_file(predecessor) != config["predecessor"]["file_sha256"]:
        raise GravityItem2AxesGroupError("Item 2 predecessor file changed")
    predecessor_value = json.loads(predecessor.read_text(encoding="utf-8"))
    if predecessor_value.get("content_sha256") != config["predecessor"]["content_sha256"]:
        raise GravityItem2AxesGroupError("Item 2 predecessor content changed")
    if predecessor_value.get("decision") != config["predecessor"]["required_decision"]:
        raise GravityItem2AxesGroupError("Item 2 predecessor decision changed")
    authorization = config["authorization"]
    if authorization["paid_model_calls_allowed"]:
        raise GravityItem2AxesGroupError("paid model calls are forbidden")
    if authorization["reserved_confirmation_member_rows_allowed"]:
        raise GravityItem2AxesGroupError("confirmation member access is not authorized")
    if authorization["published_group_velocity_columns_allowed"]:
        raise GravityItem2AxesGroupError("published group dynamics must remain unopened")
    if config["target_blind_sample"]["reserved_confirmation_target_accesses_allowed"] != 0:
        raise GravityItem2AxesGroupError("confirmation target access budget must be zero")
    metadata = root / config["catalog_sources"]["metadata_path"]
    if _sha256_file(metadata) != config["catalog_sources"]["metadata_file_sha256"]:
        raise GravityItem2AxesGroupError("metadata-only source hash changed")
    if metadata.stat().st_size != config["catalog_sources"]["metadata_bytes"]:
        raise GravityItem2AxesGroupError("metadata-only source size changed")
    return config


def _richness_bin(value: int, bins: Sequence[Sequence[int]]) -> int | None:
    for index, (lower, upper) in enumerate(bins):
        if int(lower) <= value <= int(upper):
            return index
    return None


def read_metadata_only(path: Path, *, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Parse only the exact target-blind VizieR metadata response."""

    text = path.read_text(encoding="utf-8")
    sources = config["catalog_sources"]
    for forbidden in sources["metadata_forbidden_columns"]:
        if f"#Column\t{forbidden}\t" in text:
            raise GravityItem2AxesGroupError(f"forbidden metadata column present: {forbidden}")
    header = "\t".join(sources["metadata_allowed_columns"])
    lines = text.splitlines()
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem2AxesGroupError("metadata-only header changed") from exc
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != 5 or not fields[0].strip().isdigit():
            continue
        try:
            row = {
                "group": int(fields[0]),
                "members": int(fields[1]),
                "redshift": float(fields[2]),
                "lr195": float(fields[3]),
                "d10": float(fields[4]),
            }
        except ValueError:
            continue
        rows.append(row)
    if len(rows) != int(config["target_blind_sample"]["expected_catalog_rows_with_valid_lr195"]):
        raise GravityItem2AxesGroupError("metadata-only parsed row count changed")
    if len({row["group"] for row in rows}) != len(rows):
        raise GravityItem2AxesGroupError("duplicate group in metadata-only source")
    return rows


def eligible_metadata_rows(
    rows: Sequence[Mapping[str, Any]], *, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Apply only preregistered target-blind eligibility rules."""

    sample = config["target_blind_sample"]
    lower_z, upper_z = (float(value) for value in sample["redshift_range"])
    bins = sample["richness_bins"]
    eligible: list[dict[str, Any]] = []
    for source_row in rows:
        row = dict(source_row)
        richness_bin = _richness_bin(int(row["members"]), bins)
        if richness_bin is None or int(row["members"]) < int(sample["minimum_members"]):
            continue
        if not lower_z <= float(row["redshift"]) < upper_z:
            continue
        if sample["positive_lr195_required"] and float(row["lr195"]) <= 0:
            continue
        if sample["finite_d10_required"] and not math.isfinite(float(row["d10"])):
            continue
        row["richness_bin"] = richness_bin
        eligible.append(row)
    if len(eligible) != int(sample["expected_eligible_groups"]):
        raise GravityItem2AxesGroupError("eligible group count changed")
    return eligible


def _selection_digest(salt: str, richness_bin: int, group: int) -> str:
    value = f"{salt}|richness-{richness_bin}|group-{group}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_sample_manifest(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    sample = config["target_blind_sample"]
    source = config["catalog_sources"]
    metadata_path = root / source["metadata_path"]
    rows = eligible_metadata_rows(
        read_metadata_only(metadata_path, config=config), config=config
    )
    by_stratum: defaultdict[int, list[dict[str, Any]]] = defaultdict(list)
    salt = str(sample["selection_salt"])
    for row in rows:
        selected = dict(row)
        selected["selection_digest"] = _selection_digest(
            salt, int(row["richness_bin"]), int(row["group"])
        )
        by_stratum[int(row["richness_bin"])].append(selected)
    objects: list[dict[str, Any]] = []
    quota = sample["per_richness_bin"]
    exploration_count = int(quota["exploration"])
    confirmation_count = int(quota["reserved_confirmation"])
    for richness_bin in range(len(sample["richness_bins"])):
        ordered = sorted(
            by_stratum[richness_bin],
            key=lambda row: (row["selection_digest"], int(row["group"])),
        )
        if len(ordered) < exploration_count + confirmation_count:
            raise GravityItem2AxesGroupError("insufficient groups in richness stratum")
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
        "schema_version": "invariant-gravity-item2-axes-group-sample-1.0",
        "goal": config["goal"],
        "decision": "PASS_TARGET_BLIND_GROUP_SAMPLE_SELECTION",
        "source": {
            "catalog_id": source["catalog_id"],
            "metadata_bytes": metadata_path.stat().st_size,
            "metadata_file_sha256": _sha256_file(metadata_path),
            "metadata_path": source["metadata_path"],
            "metadata_query_url": source["metadata_query_url"],
            "queried_columns": source["metadata_allowed_columns"],
        },
        "selection_boundary": {
            "metadata_endpoint_queries": 1,
            "published_group_velocity_columns_read": 0,
            "selected_member_rows_opened": 0,
            "selected_member_redshifts_read": 0,
            "reserved_confirmation_target_accesses": 0,
            "xray_target_columns_read": 0,
        },
        "counts": {
            "catalog_rows_with_valid_lr195": len(
                read_metadata_only(metadata_path, config=config)
            ),
            "eligible_groups": len(rows),
            "exploration_groups": sum(row["role"] == "exploration" for row in objects),
            "reserved_confirmation_groups": sum(
                row["role"] == "reserved_confirmation" for row in objects
            ),
        },
        "strata": {
            str(index): {
                "members_inclusive": bounds,
                "eligible": len(by_stratum[index]),
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
            for index, bounds in enumerate(sample["richness_bins"])
        },
        "objects": objects,
        "claims": {
            "alternative_to_gr_established": False,
            "confirmation_opened": False,
            "group_finder_independence_established": False,
            "member_response_seen_during_selection": False,
            "roadmap_item_2_complete": False,
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
        raise GravityItem2AxesGroupError("sample manifest content hash changed")
    if manifest.get("decision") != "PASS_TARGET_BLIND_GROUP_SAMPLE_SELECTION":
        raise GravityItem2AxesGroupError("target-blind sample did not pass")
    boundary = manifest["selection_boundary"]
    if boundary != {
        "metadata_endpoint_queries": 1,
        "published_group_velocity_columns_read": 0,
        "selected_member_rows_opened": 0,
        "selected_member_redshifts_read": 0,
        "reserved_confirmation_target_accesses": 0,
        "xray_target_columns_read": 0,
    }:
        raise GravityItem2AxesGroupError("sample selection leakage boundary changed")
    expected_counts = {
        "catalog_rows_with_valid_lr195": config["target_blind_sample"][
            "expected_catalog_rows_with_valid_lr195"
        ],
        "eligible_groups": config["target_blind_sample"]["expected_eligible_groups"],
        "exploration_groups": config["target_blind_sample"][
            "expected_exploration_groups"
        ],
        "reserved_confirmation_groups": config["target_blind_sample"][
            "expected_reserved_confirmation_groups"
        ],
    }
    if manifest["counts"] != expected_counts:
        raise GravityItem2AxesGroupError("sample counts changed")
    objects = manifest["objects"]
    if len({int(row["group"]) for row in objects}) != len(objects):
        raise GravityItem2AxesGroupError("selected group IDs are not unique")
    roles = Counter(str(row["role"]) for row in objects)
    if roles != {
        "exploration": expected_counts["exploration_groups"],
        "reserved_confirmation": expected_counts["reserved_confirmation_groups"],
    }:
        raise GravityItem2AxesGroupError("sample role counts changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem2AxesGroupError("sample manifest contains an overclaim")


def _download_member_query(url: str, path: Path) -> bytes:
    if path.exists():
        return path.read_bytes()
    request = urllib.request.Request(url, headers={"User-Agent": "Invariant/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
    except (OSError, urllib.error.URLError) as exc:
        raise GravityItem2AxesGroupError(f"member query failed: {url}") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def parse_member_payload(
    payload: bytes, *, expected_group: int, config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Parse one selected group from the exact member-only ASU schema."""

    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise GravityItem2AxesGroupError("member response is not UTF-8") from exc
    expected_columns = list(config["catalog_sources"]["member_allowed_columns"])
    header = "\t".join(expected_columns)
    lines = text.splitlines()
    try:
        start = lines.index(header)
    except ValueError as exc:
        raise GravityItem2AxesGroupError(
            f"member response header changed for group {expected_group}"
        ) from exc
    rows: list[dict[str, Any]] = []
    for line in lines[start + 1 :]:
        fields = line.split("\t")
        if len(fields) != len(expected_columns) or not fields[0].strip().isdigit():
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
            raise GravityItem2AxesGroupError("member query returned a different group")
        rows.append(row)
    if not rows:
        raise GravityItem2AxesGroupError(f"member query returned no rows: {expected_group}")
    # GalID is the catalogue's member identifier.  SpecObjID can be zero for more than
    # one otherwise distinct member, so it is provenance metadata rather than a key.
    if len({row["galaxy_id"] for row in rows}) != len(rows):
        raise GravityItem2AxesGroupError(f"duplicate member GalID rows: {expected_group}")
    return rows


def _acquire_one_group(
    row: Mapping[str, Any], *, config: Mapping[str, Any], cache_dir: Path
) -> dict[str, Any]:
    group = int(row["group"])
    url = str(config["catalog_sources"]["member_query_template"]).format(group=group)
    path = cache_dir / f"members-{group}.tsv"
    payload = _download_member_query(url, path)
    members = parse_member_payload(payload, expected_group=group, config=config)
    if len(members) != int(row["members"]):
        raise GravityItem2AxesGroupError(
            f"member count differs from frozen metadata for group {group}"
        )
    return {
        "bytes": len(payload),
        "group": group,
        "member_rows": len(members),
        "sha256": _sha256_bytes(payload),
        "url": url,
    }


def acquire_exploration(root: Path, *, cache_dir: Path, workers: int = 8) -> Path:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    exploration = [row for row in sample["objects"] if row["role"] == "exploration"]
    confirmation_ids = {
        int(row["group"]) for row in sample["objects"] if row["role"] == "reserved_confirmation"
    }
    if any(int(row["group"]) in confirmation_ids for row in exploration):
        raise GravityItem2AxesGroupError("sample roles overlap")
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as pool:
        futures = {
            pool.submit(_acquire_one_group, row, config=config, cache_dir=cache_dir): int(
                row["group"]
            )
            for row in exploration
        }
        for future in as_completed(futures):
            group = futures[future]
            try:
                records.append(future.result())
            except Exception as exc:  # noqa: BLE001 - preserve every source failure in one report
                errors.append(f"{group}: {exc}")
    if errors:
        raise GravityItem2AxesGroupError("; ".join(sorted(errors)))
    records.sort(key=lambda row: int(row["group"]))
    schema_audit = config["catalog_sources"]["member_coordinate_schema_correction"]
    manifest: dict[str, Any] = {
        "schema_version": "invariant-gravity-item2-axes-group-source-1.0",
        "goal": config["goal"],
        "decision": "PASS_EXPLORATION_MEMBER_SOURCE_ACQUISITION",
        "preregistration": {
            "git_commit": FREEZE_COMMIT,
            "selected_member_rows_opened_before_commit": 0,
        },
        "sample_binding": {
            "path": config["sample_manifest_output"],
            "file_sha256": _sha256_file(root / config["sample_manifest_output"]),
            "content_sha256": sample["content_sha256"],
        },
        "schema_correction": {
            "authorized_exploration_group": schema_audit["authorized_exploration_group"],
            "audit_queries": schema_audit["audit_queries"],
            "frozen_readme_labels": schema_audit["frozen_readme_labels"],
            "asu_service_labels": schema_audit["asu_service_labels"],
            "sample_or_scientific_contract_changed": False,
        },
        "boundary": {
            "exploration_groups_acquired": len(records),
            "exploration_member_query_accesses": len(records),
            "schema_audit_target_accesses": int(schema_audit["audit_queries"]),
            "total_exploration_target_accesses": len(records)
            + int(schema_audit["audit_queries"]),
            "reserved_confirmation_groups_acquired": 0,
            "reserved_confirmation_target_accesses": 0,
            "published_group_velocity_columns_read": 0,
            "xray_target_columns_read": 0,
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
            "roadmap_item_2_complete": False,
        },
    }
    manifest["content_sha256"] = canonical_sha256(manifest)
    validate_source_manifest(manifest, config=config, sample=sample)
    path = root / config["source_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def validate_source_manifest(
    manifest: Mapping[str, Any], *, config: Mapping[str, Any], sample: Mapping[str, Any]
) -> None:
    copy = dict(manifest)
    digest = copy.pop("content_sha256", None)
    if digest != canonical_sha256(copy):
        raise GravityItem2AxesGroupError("source manifest content hash changed")
    if manifest.get("decision") != "PASS_EXPLORATION_MEMBER_SOURCE_ACQUISITION":
        raise GravityItem2AxesGroupError("exploration source acquisition did not pass")
    boundary = manifest["boundary"]
    if boundary["reserved_confirmation_groups_acquired"] != 0:
        raise GravityItem2AxesGroupError("confirmation groups were acquired")
    if boundary["reserved_confirmation_target_accesses"] != 0:
        raise GravityItem2AxesGroupError("confirmation member responses were accessed")
    if boundary["published_group_velocity_columns_read"] != 0:
        raise GravityItem2AxesGroupError("published group dynamics were read")
    if boundary["xray_target_columns_read"] != 0:
        raise GravityItem2AxesGroupError("X-ray target columns were read")
    exploration_ids = {
        int(row["group"]) for row in sample["objects"] if row["role"] == "exploration"
    }
    record_ids = {int(row["group"]) for row in manifest["records"]}
    if record_ids != exploration_ids:
        raise GravityItem2AxesGroupError("source records differ from exploration IDs")
    if manifest["counts"]["groups"] != len(exploration_ids):
        raise GravityItem2AxesGroupError("source group count changed")
    if any(bool(value) for value in manifest["claims"].values()):
        raise GravityItem2AxesGroupError("source manifest contains an overclaim")


def _weighted_geometry(
    x: np.ndarray, y: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    total = float(np.sum(weights))
    if total <= 0 or x.size < 3:
        raise GravityItem2AxesGroupError("insufficient weighted positions")
    cx = float(np.sum(weights * x) / total)
    cy = float(np.sum(weights * y) / total)
    dx = x - cx
    dy = y - cy
    covariance = np.array(
        [
            [np.sum(weights * dx * dx) / total, np.sum(weights * dx * dy) / total],
            [np.sum(weights * dx * dy) / total, np.sum(weights * dy * dy) / total],
        ]
    )
    values, vectors = np.linalg.eigh(covariance)
    if values[1] <= 0 or values[0] < -1.0e-12:
        raise GravityItem2AxesGroupError("degenerate projected covariance")
    axis_ratio = math.sqrt(max(float(values[0]), 0.0) / float(values[1]))
    major = vectors[:, 1]
    angle = math.atan2(float(major[1]), float(major[0]))
    radius = np.hypot(dx, dy)
    rms = math.sqrt(float(np.sum(weights * radius * radius) / total))
    if rms <= 0:
        raise GravityItem2AxesGroupError("zero projected size")
    phi = np.arctan2(dy, dx)
    normalized_radius = radius / rms
    moments: dict[str, float] = {}
    for order in (2, 3, 4):
        complex_moment = np.sum(
            weights * normalized_radius**order * np.exp(1j * order * phi)
        ) / np.sum(weights * normalized_radius**order)
        moments[f"m{order}"] = float(abs(complex_moment))
    return {
        "axis_ratio": axis_ratio,
        "center_x": cx,
        "center_y": cy,
        "m2": moments["m2"],
        "m3": moments["m3"],
        "m4": moments["m4"],
        "position_angle": angle,
        "rms_radius": rms,
    }


def _weighted_half_radius(radius: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(radius, kind="stable")
    cumulative = np.cumsum(weights[order])
    index = int(np.searchsorted(cumulative, 0.5 * cumulative[-1], side="left"))
    return float(radius[order[min(index, radius.size - 1)]])


def _graph_features(x: np.ndarray, y: np.ndarray, rms_radius: float) -> dict[str, float]:
    points = np.column_stack((x, y))
    distances = cdist(points, points)
    tree = minimum_spanning_tree(distances).toarray()
    undirected = tree + tree.T
    mst_length = float(np.sum(tree))
    if mst_length <= 0:
        raise GravityItem2AxesGroupError("degenerate member spanning tree")
    path_lengths = dijkstra(undirected, directed=False)
    tree_diameter = float(np.max(path_lengths[np.isfinite(path_lengths)]))
    euclidean_diameter = float(np.max(distances))
    angles = np.mod(np.arctan2(y, x), 2.0 * np.pi)
    ordered = np.sort(angles)
    gaps = np.diff(np.concatenate((ordered, ordered[:1] + 2.0 * np.pi))) / (2.0 * np.pi)
    positive = gaps[gaps > 0]
    entropy = float(-np.sum(positive * np.log(positive)) / math.log(len(gaps)))
    return {
        "angular_gap_entropy": entropy,
        "mst_diameter_efficiency": euclidean_diameter / tree_diameter,
        "mst_length_per_rms_radius": mst_length / (math.sqrt(x.size) * rms_radius),
    }


def measure_geometry_only(
    ra_deg: np.ndarray,
    dec_deg: np.ndarray,
    luminosity: np.ndarray,
    metadata_redshift: float,
    config: Mapping[str, Any],
) -> dict[str, float]:
    """Measure target-blind projected geometry without accepting member redshifts."""

    ra = np.asarray(ra_deg, dtype=np.float64)
    dec = np.asarray(dec_deg, dtype=np.float64)
    weights = np.asarray(luminosity, dtype=np.float64)
    valid = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(weights) & (weights > 0)
    ra = ra[valid]
    dec = dec[valid]
    weights = weights[valid]
    if ra.size < int(config["response"]["minimum_members_after_finite_value_filter"]):
        raise GravityItem2AxesGroupError("insufficient finite member geometry")
    angle = np.deg2rad(ra)
    mean_angle = math.atan2(float(np.sum(weights * np.sin(angle))), float(np.sum(weights * np.cos(angle))))
    delta_ra = np.angle(np.exp(1j * (angle - mean_angle)))
    mean_dec = float(np.sum(weights * np.deg2rad(dec)) / np.sum(weights))
    cosmology_config = config["geometry"]["cosmology"]
    cosmology = FlatLambdaCDM(
        H0=float(cosmology_config["H0_km_s_Mpc"]),
        Om0=float(cosmology_config["Omega_m"]),
        Tcmb0=2.725,
    )
    distance_kpc = float(cosmology.angular_diameter_distance(metadata_redshift).value) * 1000.0
    x = distance_kpc * math.cos(mean_dec) * delta_ra
    y = distance_kpc * (np.deg2rad(dec) - mean_dec)
    global_geometry = _weighted_geometry(x, y, weights)
    cx = global_geometry["center_x"]
    cy = global_geometry["center_y"]
    centered_x = x - cx
    centered_y = y - cy
    radius = np.hypot(centered_x, centered_y)
    half_radius = _weighted_half_radius(radius, weights)
    inner = radius <= half_radius
    outer = ~inner
    if np.count_nonzero(inner) < 3 or np.count_nonzero(outer) < 3:
        raise GravityItem2AxesGroupError("insufficient members in radial geometry split")
    inner_geometry = _weighted_geometry(x[inner], y[inner], weights[inner])
    outer_geometry = _weighted_geometry(x[outer], y[outer], weights[outer])
    centroid_separation = math.hypot(
        inner_geometry["center_x"] - outer_geometry["center_x"],
        inner_geometry["center_y"] - outer_geometry["center_y"],
    )
    graph = _graph_features(centered_x, centered_y, global_geometry["rms_radius"])
    angle_difference = inner_geometry["position_angle"] - outer_geometry["position_angle"]
    total_luminosity = float(np.sum(weights))
    return {
        "angular_gap_entropy": graph["angular_gap_entropy"],
        "axis_twist_inner_outer_sin2": abs(math.sin(2.0 * angle_difference)),
        "centroid_shift_profile": centroid_separation / global_geometry["rms_radius"],
        "inner_outer_axis_ratio_difference": inner_geometry["axis_ratio"]
        - outer_geometry["axis_ratio"],
        "inner_outer_quadrupole_difference": inner_geometry["m2"] - outer_geometry["m2"],
        "luminosity_half_radius_kpc": half_radius,
        "luminosity_total_catalog_units": total_luminosity,
        "m3": global_geometry["m3"],
        "m4": global_geometry["m4"],
        "mst_diameter_efficiency": graph["mst_diameter_efficiency"],
        "mst_length_per_rms_radius": graph["mst_length_per_rms_radius"],
        "outer_multipole_energy": math.hypot(outer_geometry["m3"], outer_geometry["m4"]),
        "projected_axis_ratio": global_geometry["axis_ratio"],
        "projected_ellipticity": 1.0 - global_geometry["axis_ratio"],
        "projected_linearity": 1.0 - global_geometry["axis_ratio"],
        "quadrupole": global_geometry["m2"],
        "rms_radius_kpc": global_geometry["rms_radius"],
    }


def _gapper_dispersion(velocities: np.ndarray) -> float:
    ordered = np.sort(np.asarray(velocities, dtype=np.float64))
    count = ordered.size
    gaps = np.diff(ordered)
    indices = np.arange(1, count, dtype=np.float64)
    weights = indices * (count - indices)
    return float(math.sqrt(math.pi) * np.sum(weights * gaps) / (count * (count - 1)))


def measure_response_only(
    member_redshift: np.ndarray, geometry: Mapping[str, float], config: Mapping[str, Any]
) -> dict[str, float]:
    redshift = np.asarray(member_redshift, dtype=np.float64)
    redshift = redshift[np.isfinite(redshift)]
    if redshift.size < int(config["response"]["minimum_members_after_finite_value_filter"]):
        raise GravityItem2AxesGroupError("insufficient finite member redshifts")
    if np.unique(redshift).size < int(config["response"]["minimum_unique_member_redshifts"]):
        raise GravityItem2AxesGroupError("insufficient unique member redshifts")
    median_redshift = float(np.median(redshift))
    speed_of_light = 299792.458
    velocity = speed_of_light * (redshift - median_redshift) / (1.0 + median_redshift)
    gapper = _gapper_dispersion(velocity)
    mad = 1.4826 * float(np.median(np.abs(velocity - np.median(velocity))))
    if gapper <= 0 or mad <= 0:
        raise GravityItem2AxesGroupError("nonpositive group velocity dispersion")
    luminosity = float(geometry["luminosity_total_catalog_units"])
    rms_radius = float(geometry["rms_radius_kpc"])
    half_radius = float(geometry["luminosity_half_radius_kpc"])
    return {
        "log10_eta_half_radius": math.log10(gapper * gapper * half_radius / luminosity),
        "log10_eta_lum": math.log10(gapper * gapper * rms_radius / luminosity),
        "log10_eta_mad": math.log10(mad * mad * rms_radius / luminosity),
        "log10_sigma_gap": math.log10(gapper),
        "member_median_redshift": median_redshift,
        "sigma_gap_km_s": gapper,
        "sigma_mad_km_s": mad,
        "unique_member_redshifts": int(np.unique(redshift).size),
    }


def _feature_fieldnames(config: Mapping[str, Any]) -> list[str]:
    geometry = config["geometry"]
    return [
        "group",
        "richness_bin",
        "members",
        "metadata_redshift",
        "lr195",
        "d10",
        "log10_member_luminosity",
        "log10_rms_radius_kpc",
        "log10_richness",
        *geometry["global_features"],
        *geometry["radial_nonlocal_features"],
        *geometry["graph_filament_features"],
        "luminosity_half_radius_kpc",
        "rms_radius_kpc",
        "sigma_gap_km_s",
        "sigma_mad_km_s",
        "unique_member_redshifts",
        "log10_eta_lum",
        "log10_sigma_gap",
        "log10_eta_half_radius",
        "log10_eta_mad",
    ]


def extract_exploration(root: Path, *, cache_dir: Path) -> tuple[Path, Path]:
    root = root.resolve()
    cache_dir = cache_dir.resolve()
    config = load_config(root)
    sample = _load_sample(root, config)
    source_path = root / config["source_manifest_output"]
    source_manifest = json.loads(source_path.read_text(encoding="utf-8"))
    validate_source_manifest(source_manifest, config=config, sample=sample)
    source_by_group = {int(row["group"]): row for row in source_manifest["records"]}
    feature_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for object_row in sample["objects"]:
        if object_row["role"] != "exploration":
            continue
        group = int(object_row["group"])
        path = cache_dir / f"members-{group}.tsv"
        binding = source_by_group[group]
        if _sha256_file(path) != binding["sha256"]:
            raise GravityItem2AxesGroupError(f"cached source hash changed: {group}")
        members = parse_member_payload(path.read_bytes(), expected_group=group, config=config)
        try:
            # Geometry is fully finalized before the response function is called.
            geometry = measure_geometry_only(
                np.asarray([row["ra_deg"] for row in members]),
                np.asarray([row["dec_deg"] for row in members]),
                np.asarray([row["luminosity"] for row in members]),
                float(object_row["redshift"]),
                config,
            )
            response = measure_response_only(
                np.asarray([row["member_redshift"] for row in members]), geometry, config
            )
        except GravityItem2AxesGroupError as exc:
            failures.append({"group": group, "reason": str(exc)})
            continue
        values: dict[str, Any] = {
            "d10": float(object_row["d10"]),
            "group": group,
            "log10_member_luminosity": math.log10(
                geometry["luminosity_total_catalog_units"]
            ),
            "log10_richness": math.log10(int(object_row["members"])),
            "log10_rms_radius_kpc": math.log10(geometry["rms_radius_kpc"]),
            "lr195": float(object_row["lr195"]),
            "members": int(object_row["members"]),
            "metadata_redshift": float(object_row["redshift"]),
            "richness_bin": int(object_row["richness_bin"]),
            **geometry,
            **response,
        }
        feature_rows.append(values)
    feature_rows.sort(key=lambda row: int(row["group"]))
    fieldnames = _feature_fieldnames(config)
    feature_path = root / config["feature_output"]
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    with feature_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, dialect="excel-tab")
        writer.writeheader()
        for row in feature_rows:
            writer.writerow(
                {
                    key: int(row[key])
                    if key in {"group", "richness_bin", "members", "unique_member_redshifts"}
                    else _metric(float(row[key]))
                    for key in fieldnames
                }
            )
    summary: dict[str, Any] = {
        "schema_version": "invariant-gravity-item2-axes-group-extraction-summary-1.0",
        "goal": config["goal"],
        "decision": (
            "PASS_EXPLORATION_REPRESENTATION_QUALITY"
            if not failures and len(feature_rows) == 180
            else "FAIL_EXPLORATION_REPRESENTATION_QUALITY"
        ),
        "counts": {
            "quality_failures": len(failures),
            "quality_passing": len(feature_rows),
            "reserved_confirmation_target_accesses": 0,
            "selected_exploration": 180,
        },
        "failures": failures,
        "leakage_boundary": {
            "geometry_finalized_before_response_function": True,
            "geometry_function_accepts_member_redshift": False,
            "published_group_velocity_columns_read": 0,
            "reserved_confirmation_target_accesses": 0,
            "xray_target_columns_read": 0,
        },
        "feature_table": {
            "path": config["feature_output"],
            "rows": len(feature_rows),
            "sha256": _sha256_file(feature_path),
        },
    }
    summary["content_sha256"] = canonical_sha256(summary)
    summary_path = root / config["extraction_summary_output"]
    summary_path.write_bytes(canonical_json_bytes(summary))
    return feature_path, summary_path


def write_sample_manifest(root: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    manifest = build_sample_manifest(root)
    path = root / config["sample_manifest_output"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(manifest))
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("select", "check-sample", "acquire-exploration", "extract-exploration"),
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--cache-dir", type=Path, default=Path("work/item2-axes-groups-v5-raw"))
    parser.add_argument("--workers", type=int, default=8)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        path = write_sample_manifest(root)
        print(path)
        return 0
    if args.command == "acquire-exploration":
        path = acquire_exploration(root, cache_dir=args.cache_dir, workers=args.workers)
        print(path)
        return 0
    if args.command == "extract-exploration":
        feature_path, summary_path = extract_exploration(root, cache_dir=args.cache_dir)
        print(feature_path)
        print(summary_path)
        return 0
    config = load_config(root)
    path = root / config["sample_manifest_output"]
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_sample_manifest(manifest, config=config)
    if build_sample_manifest(root) != manifest:
        raise GravityItem2AxesGroupError("stored sample is not an exact deterministic rebuild")
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
