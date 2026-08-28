"""Frozen AXES-SDSS group source lane for gravity-roadmap Item 2 attempt 5.

Selection reads a committed VizieR response containing only group identifiers, richness,
median redshift, optical luminosity, and environment.  It cannot read member redshifts or
published velocity/radius columns.  Exploration acquisition is a separate command so the
sample and its sealed confirmation role are immutable before any dynamics are opened.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item2_axes_group_geometry.json"
SAMPLE_MANIFEST_PATH = (
    "runs/gravity/roadmap/item-02-axes-group-geometry-v5-source/"
    "axes-group-sample-manifest.json"
)


class GravityItem2AxesGroupError(RuntimeError):
    """Raised when the frozen group source, sample, or leakage boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _metric(value: float) -> str:
    return f"{float(value):.12e}"


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
    parser.add_argument("command", choices=("select", "check-sample"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        path = write_sample_manifest(root)
        print(path)
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
