"""Header-only source registration for the frozen Item 38 KiDS archive."""

from __future__ import annotations

import argparse
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.gravity_item22_polarization_superposition import (
    _content_hashed,
    _git,
    _require_ancestor,
    _sha256_bytes,
    _sha256_file,
    _write_json,
)
from sigma_theory_compiler.gravity_item38_emergent_gravity import (
    CONFIG_PATH,
    MODULE_PATH,
    GravityItem38Error,
    _contract_digest,
    _source_paths,
    load_config,
)

SOURCE_MODULE_PATH = Path("src/sigma_theory_compiler/gravity_item38_emergent_source.py")


def verify_scientific_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["scientific_freeze_commit"] or "")
    if len(commit) != 40:
        raise GravityItem38Error("Item 38 scientific freeze is not bound")
    _require_ancestor(root, commit, "Item 38 scientific freeze")
    frozen_config = json.loads(str(_git(root, "show", f"{commit}:{CONFIG_PATH.as_posix()}")))
    if _contract_digest(frozen_config) != _contract_digest(config):
        raise GravityItem38Error("Item 38 scientific contract differs from frozen commit")
    frozen_module = _git(root, "show", f"{commit}:{MODULE_PATH.as_posix()}", text_mode=False)
    if not isinstance(frozen_module, bytes):
        raise GravityItem38Error("cannot read frozen Item 38 candidate module")
    if _sha256_bytes(frozen_module) != _sha256_file(root / MODULE_PATH):
        raise GravityItem38Error("Item 38 candidate machinery changed after freeze")


def _role_for_member(name: str) -> str | None:
    if name.startswith("._") or not name.endswith(".txt"):
        return None
    if name in {
        "Fig-9_RAR-KiDS-isolated_Massbin-1.txt",
        "Fig-9_RAR-KiDS-isolated_Massbin-2.txt",
        "Fig-9_RAR-KiDS-isolated_Massbin-3.txt",
    }:
        return "exploration"
    if name == "Fig-9_RAR-KiDS-isolated_Massbin-4.txt":
        return "sealed_confirmation"
    if name in {
        "Fig-8_RAR-KiDS-isolated_Colorbin_1.txt",
        "Fig-8_RAR-KiDS-isolated_Colorbin_2.txt",
    }:
        return "unchanged_color_transfer"
    if name in {
        "Fig-9_RAR-KiDS-isolated_Massbins_covmatrix.txt",
        "Fig-8_RAR-KiDS-isolated_Colorbins_covmatrix.txt",
    }:
        return "declared_covariance"
    if name == "README.txt":
        return "metadata_readme"
    return "unused_unopened"


def register_source_headers(
    root: Path, archive: Path, source_output: Path, sample_output: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(root)
    verify_scientific_freeze(root, config)
    if not archive.is_file():
        raise GravityItem38Error(f"KiDS archive is missing: {archive}")

    members: list[dict[str, Any]] = []
    with tarfile.open(archive, mode="r:") as bundle:
        for member in bundle.getmembers():
            if not member.isfile():
                continue
            role = _role_for_member(member.name)
            stored_role = "appledouble_ignored" if role is None else role
            members.append(
                {
                    "name": member.name,
                    "size_bytes": int(member.size),
                    "header_offset": int(member.offset),
                    "payload_offset": int(member.offset_data),
                    "role": stored_role,
                    "member_payload_opened": False,
                }
            )

    role_counts: dict[str, int] = {}
    for row in members:
        role = str(row["role"])
        role_counts[role] = role_counts.get(role, 0) + 1
    expected_counts = {
        "exploration": 3,
        "sealed_confirmation": 1,
        "unchanged_color_transfer": 2,
        "declared_covariance": 2,
        "metadata_readme": 1,
    }
    for role, expected in expected_counts.items():
        if role_counts.get(role) != expected:
            raise GravityItem38Error(
                f"unexpected KiDS header count for {role}: {role_counts.get(role, 0)}"
            )
    selected_names = [
        str(row["name"])
        for row in members
        if row["role"] in expected_counts and row["role"] != "metadata_readme"
    ]
    if len(selected_names) != len(set(selected_names)):
        raise GravityItem38Error("duplicate selected KiDS archive member")

    source = _content_hashed(
        {
            "schema_version": "invariant-gravity-item38-source-metadata-manifest-1.0",
            "item": 38,
            "archive_url": config["data_source"]["archive_url"],
            "archive_sha256": _sha256_file(archive),
            "archive_bytes": int(archive.stat().st_size),
            "header_registration_only": True,
            "semantic_member_payload_bytes_read": 0,
            "archive_hashing_only_is_not_semantic_response_access": True,
            "members": members,
            "role_counts": dict(sorted(role_counts.items())),
            "response_accessed": False,
            "confirmation_accessed": False,
            "paid_api_calls": 0,
        }
    )
    sample = _content_hashed(
        {
            "schema_version": "invariant-gravity-item38-sample-manifest-1.0",
            "item": 38,
            "split_unit": config["sample_boundary"]["split_unit"],
            "selection_used_response_values": False,
            "member_payload_accessed": False,
            "exploration": sorted(
                str(row["name"]) for row in members if row["role"] == "exploration"
            ),
            "sealed_confirmation": sorted(
                str(row["name"])
                for row in members
                if row["role"] == "sealed_confirmation"
            ),
            "unchanged_color_transfer": sorted(
                str(row["name"])
                for row in members
                if row["role"] == "unchanged_color_transfer"
            ),
            "covariance": sorted(
                str(row["name"])
                for row in members
                if row["role"] == "declared_covariance"
            ),
            "unused_unopened_count": role_counts.get("unused_unopened", 0),
            "appledouble_ignored_count": sum(
                str(row["name"]).startswith("._") for row in members
            ),
            "confirmation_access_budget": 0,
            "source_metadata_sha256": source["content_sha256"],
        }
    )
    _write_json(source_output, source)
    _write_json(sample_output, sample)
    return source, sample


def verify_source_metadata_freeze(root: Path, config: Mapping[str, Any]) -> None:
    commit = str(config["source_metadata_freeze_commit"] or "")
    if len(commit) != 40:
        raise GravityItem38Error("Item 38 source metadata freeze is not bound")
    _require_ancestor(root, commit, "Item 38 source metadata freeze")
    paths = _source_paths(root, config)
    for name in ("candidate_manifest", "source_metadata_manifest", "sample_manifest"):
        relative = paths[name].resolve().relative_to(root.resolve()).as_posix()
        frozen = _git(root, "show", f"{commit}:{relative}", text_mode=False)
        if not isinstance(frozen, bytes) or _sha256_bytes(frozen) != _sha256_file(paths[name]):
            raise GravityItem38Error(f"Item 38 {name} changed after source freeze")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register-source-headers")
    register.add_argument("--root", type=Path, default=Path("."))
    register.add_argument("--archive", type=Path, required=True)
    register.add_argument("--source-output", type=Path, required=True)
    register.add_argument("--sample-output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "register-source-headers":
        source, sample = register_source_headers(
            args.root.resolve(),
            args.archive.resolve(),
            args.source_output,
            args.sample_output,
        )
        print(
            json.dumps(
                {
                    "source_content_sha256": source["content_sha256"],
                    "sample_content_sha256": sample["content_sha256"],
                    "role_counts": source["role_counts"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise GravityItem38Error(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
