"""Frozen source and dimensionless derivation for gravity-roadmap Item 3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item3_surface_volume_density.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-03-surface-volume-density-v1-source/"
    "fresh-group-sample-manifest.json"
)


class GravityItem3SurfaceVolumeDensityError(RuntimeError):
    """Raised when the frozen derivation, sample, or target boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "check-sample"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        path = write_sample_manifest(root)
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
