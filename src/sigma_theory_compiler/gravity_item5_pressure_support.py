"""Frozen source and sample boundary for gravity-roadmap Item 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/gravity_item5_pressure_support_little_things_v1.json"


class GravityItem5PressureSupportError(RuntimeError):
    """Raised when the frozen Item 5 boundary drifts."""


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("select", "check-sample"))
    parser.add_argument("--root", type=Path, default=Path("."))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    if args.command == "select":
        print(write_sample_manifest(root))
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
