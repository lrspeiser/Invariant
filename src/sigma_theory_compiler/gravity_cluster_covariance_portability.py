"""Verify portable provenance for the frozen cluster covariance V1 artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_covariance_portability_v1.json")
OUTPUT_PATH = Path(
    "runs/gravity/publication-readiness/covariance-portability-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-covariance-portability-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-covariance-portability-receipt-1.0"
CLUSTERS = (
    "A1644",
    "A1795",
    "A2142",
    "A2255",
    "A2319",
    "A3266",
    "A85",
    "ZW1215",
)
V1_BINDING_IDS = (
    "RECONSTRUCTION_CONFIG",
    "RECONSTRUCTION_IMPLEMENTATION",
    "RECONSTRUCTION_RECEIPT",
    "SCORING_CONFIG",
    "SCORING_IMPLEMENTATION",
    "SCORING_RECEIPT",
)
COVARIANCE_MEMBER_BINDINGS = (
    (
        "A1644",
        "A1644/A1644_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "3577cdab77c6ccde78dc6788564211cafd6498910a0026031bda8801c69e7f0c",
    ),
    (
        "A1795",
        "A1795/A1795_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "b3ed7d352146820b56e75a41fb5dd24ca995073d54e7292623eee968ed65ac22",
    ),
    (
        "A2142",
        "A2142/A2142_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "d7b0128113d9be897700e2b23c9ab0123bb3c9252e75e8c216419acda9d4c851",
    ),
    (
        "A2255",
        "A2255/A2255_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "6b1a4ad6df945213a2189bbfc881cb0c0d916e503cf027c24478d1bf705eb4c2",
    ),
    (
        "A2319",
        "A2319/A2319_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "ed351ab3a38196454a18ac4f66813987233c8c432ebf745a5c6d12b9a4f4c886",
    ),
    (
        "A3266",
        "A3266/A3266_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "99419e88d51499fccda38ee91e68d6cb7848acca4f6ff5b42a5740a2544d5e67",
    ),
    (
        "A85",
        "A85/A85_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "3d3b9fd1301500d46c8e5e8b2557613fccb39bc08f702378e03f22bbe945c328",
    ),
    (
        "ZW1215",
        "ZW1215/ZwCl1215.1+040_Y-PROF-COVMAT_P-PROF-COVMAT.20170830.fits",
        "aa26667e2d54ec1f93d7dd4b011f58fc3820ae019a60420f4187f24130915b97",
    ),
)
PRESSURE_FILE_BINDINGS = (
    (
        "A1644",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A1644/A1644_pressure.fits",
        "93a39cbba8613dc6f1ca8e2b114cf4c512c7ac35ae67c18c6206594665157274",
    ),
    (
        "A1795",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A1795/A1795_pressure.fits",
        "1788e2e4f397f79b27642fda298f7a91a84063a5ad69af990303e7c32c2fbb8d",
    ),
    (
        "A2142",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A2142/A2142_pressure.fits",
        "87bd6fbe0cead2a705beed2004a493197d3a30c5ed291e3c547933c126419bcf",
    ),
    (
        "A2255",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A2255/A2255_pressure.fits",
        "abf963f94b27306ae79ac5c08165e0c5ca948431eae652de1b57d72d90302f25",
    ),
    (
        "A2319",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A2319/A2319_pressure.fits",
        "2bdc28bde58167a6b71bef5754fef9a9c4fffea8a6686ae0fbc1bb23ba7a923e",
    ),
    (
        "A3266",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A3266/A3266_pressure.fits",
        "74b1a7b6b73302ee749c37973244f4a37c62e88ab3a5689f6ecb6d6a1eba2fb3",
    ),
    (
        "A85",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/A85/A85_pressure.fits",
        "8dbf06c943cc9808cd9a06151775a2c0377d0254d7ad9ea79b38dacacdede3eb",
    ),
    (
        "ZW1215",
        "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/ZW1215/ZW1215_pressure.fits",
        "dc92ccd3b4351f27905b469ff765840eb92c1ddd9d81ff7e8b6378a06d6671da",
    ),
)
FROZEN_V1_BINDINGS = (
    (
        "RECONSTRUCTION_CONFIG",
        "configs/gravity_cluster_development_covariance_reconstruction_v1.json",
        "ab5c2964f218bdab38d5f0fdd2bd1931069d595b6dd0759448d53825a7f5ec01",
        None,
    ),
    (
        "RECONSTRUCTION_IMPLEMENTATION",
        "src/sigma_theory_compiler/gravity_cluster_development_covariance_reconstruction.py",
        "846b91f14408f80d8398e738ed9892c4c5c5ad268d535894ec3a1ac123c0c88d",
        None,
    ),
    (
        "RECONSTRUCTION_RECEIPT",
        "runs/gravity/publication-readiness/development-covariance-reconstruction-v1.json",
        "c9e57a36bdb92d27aa1e1250171731ee2b7bbf89bffa20110ef1327ca86ae937",
        "aa1a0f2661918fcb2e84fcf4a451db31d713e9c1ca6310616d91cb7aae2c5284",
    ),
    (
        "SCORING_CONFIG",
        "configs/gravity_cluster_pressure_covariance_scoring_pilot_v1.json",
        "1dd5882fc1dad0beb32c3d628226f8ae08f23582a5919897b6e72376e655f78b",
        None,
    ),
    (
        "SCORING_IMPLEMENTATION",
        "src/sigma_theory_compiler/gravity_cluster_pressure_covariance_scoring_pilot.py",
        "652f8467be4764797e93babe633545aaf6a028d5530f0999e728dc68f7422e56",
        None,
    ),
    (
        "SCORING_RECEIPT",
        "runs/gravity/publication-readiness/pressure-covariance-scoring-pilot-v1.json",
        "da5a61f29ff9366c431ba07503a998603e592b9d465a6eaf37c2faac3c8bd748",
        "a84730f92449a7b78ce9b4bd522a602a88db83662bc714a8fbe240984f401193",
    ),
)


class GravityClusterCovariancePortabilityError(RuntimeError):
    """Raised when portable provenance or an optional archive check fails."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterCovariancePortabilityError(f"expected object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterCovariancePortabilityError(f"{label} keys changed")


def _under(root: Path, relative: str, label: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterCovariancePortabilityError(
            f"{label} escaped repository root"
        ) from error
    return path


def _validate_content_hash(value: Mapping[str, Any], expected: str, label: str) -> None:
    body = dict(value)
    observed = body.pop("content_sha256", None)
    if observed != expected or _sha(body) != expected:
        raise GravityClusterCovariancePortabilityError(f"{label} content changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "verifier_id",
            "purpose",
            "implementation_binding",
            "official_source",
            "external_archive_contract",
            "covariance_members",
            "standalone_pressure_files",
            "frozen_v1_bindings",
            "portable_integrity_mode",
            "optional_full_replay",
            "claim_boundary",
            "output_path",
        },
        "portability config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "portable_integrity_frozen_external_archive_optional"
        or config["verifier_id"] != "gravity-cluster-covariance-portability-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterCovariancePortabilityError("verifier identity changed")

    implementation = config["implementation_binding"]
    if implementation != {
        "path": "src/sigma_theory_compiler/gravity_cluster_covariance_portability.py",
        "file_sha256": implementation.get("file_sha256"),
    } or len(str(implementation.get("file_sha256", ""))) != 64:
        raise GravityClusterCovariancePortabilityError("implementation binding changed")

    official = config["official_source"]
    if official != {
        "project": "XMM Cluster Outskirts Project (X-COP)",
        "release_page_url": "https://dominiqueeckert.wixsite.com/xcop/data",
        "archive_download_url": "https://drive.switch.ch/index.php/s/j3WUOYXWgv9Jbnz/download",
        "license_status": "not_verified_for_redistribution",
        "redistribution_authorized_by_this_contract": False,
    }:
        raise GravityClusterCovariancePortabilityError("official source boundary changed")

    archive = config["external_archive_contract"]
    if archive != {
        "sha256": "0edf5038b419b70d070b73b22f4801e27f318b0854db61eec52142c27c140d94",
        "bytes": 315080566,
        "included_in_portable_package": False,
        "required_for_portable_integrity_check": False,
        "required_for_optional_full_replay": True,
        "local_path_frozen_by_this_verifier": None,
    }:
        raise GravityClusterCovariancePortabilityError("external archive boundary changed")

    members = config["covariance_members"]
    if tuple(
        (row.get("cluster"), row.get("member"), row.get("sha256"))
        for row in members
    ) != COVARIANCE_MEMBER_BINDINGS:
        raise GravityClusterCovariancePortabilityError("covariance member population changed")
    for row in members:
        _strict(row, {"cluster", "member", "sha256"}, "covariance member")
        if len(str(row["sha256"])) != 64 or not str(row["member"]).startswith(
            f"{row['cluster']}/"
        ):
            raise GravityClusterCovariancePortabilityError(
                "covariance member identity changed"
            )

    pressure = config["standalone_pressure_files"]
    if tuple(
        (row.get("cluster"), row.get("path"), row.get("sha256"))
        for row in pressure
    ) != PRESSURE_FILE_BINDINGS:
        raise GravityClusterCovariancePortabilityError("pressure population changed")
    for row in pressure:
        _strict(
            row,
            {"cluster", "path", "sha256", "repository_tracked_at_freeze"},
            "standalone pressure file",
        )
        if (
            not row["repository_tracked_at_freeze"]
            or not str(row["path"]).startswith(
                "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/"
            )
            or len(str(row["sha256"])) != 64
        ):
            raise GravityClusterCovariancePortabilityError(
                "standalone pressure binding changed"
            )

    bindings = config["frozen_v1_bindings"]
    if tuple(
        (
            row.get("binding_id"),
            row.get("path"),
            row.get("file_sha256"),
            row.get("content_sha256"),
        )
        for row in bindings
    ) != FROZEN_V1_BINDINGS:
        raise GravityClusterCovariancePortabilityError("V1 binding inventory changed")
    for row in bindings:
        _strict(
            row,
            {"binding_id", "path", "file_sha256", "content_sha256"},
            "V1 binding",
        )
        if len(str(row["file_sha256"])) != 64 or (
            row["content_sha256"] is not None
            and len(str(row["content_sha256"])) != 64
        ):
            raise GravityClusterCovariancePortabilityError("V1 binding hash changed")

    if config["portable_integrity_mode"] != {
        "archive_file_required": False,
        "work_directory_required": False,
        "git_metadata_required": False,
        "network_required": False,
        "scientific_payload_rows_opened": 0,
        "checks": [
            "verifier_and_config_integrity",
            "six_frozen_v1_file_hashes",
            "two_receipt_content_hashes",
            "receipt_cross_bindings_and_claim_boundaries",
            "eight_tracked_standalone_pressure_file_hashes",
            "eight_external_covariance_member_names_and_expected_hashes",
        ],
    }:
        raise GravityClusterCovariancePortabilityError("portable mode changed")

    replay = config["optional_full_replay"]
    if replay != {
        "archive_preflight_command": "python -m sigma_theory_compiler.gravity_cluster_covariance_portability check-external-archive --archive PATH_TO_XCOP_ARCHIVE",
        "archive_preflight_only": True,
        "archive_preflight_reads_only_named_member_bytes": True,
        "archive_preflight_parses_scientific_rows": False,
        "full_replay_executed_by_this_verifier": False,
        "full_replay_requires_external_archive_at_the_separately_frozen_v1_location": True,
        "full_replay_uses_existing_v1_reconstruction_and_scoring_implementations": True,
        "full_replay_not_claimed_portable_without_external_archive": True,
    }:
        raise GravityClusterCovariancePortabilityError("optional replay boundary changed")

    if config["claim_boundary"] != {
        "portable_integrity_supported": True,
        "archive_redistribution_allowed": False,
        "archive_license_verified": False,
        "portable_package_is_full_replay_complete": False,
        "external_archive_preflight_is_scientific_replay": False,
        "scientific_rescoring_performed": False,
        "scientific_result_changed": False,
        "CP5_status_changed": False,
        "independent_replication": False,
    }:
        raise GravityClusterCovariancePortabilityError("claim boundary changed")


def load_config(root: Path) -> dict[str, Any]:
    config = _read_json(root.resolve() / CONFIG_PATH)
    validate_config(config)
    return config


def _load_bound_files(
    root: Path, config: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    implementation = config["implementation_binding"]
    implementation_path = _under(root, implementation["path"], "implementation")
    if (
        not implementation_path.is_file()
        or _file_sha(implementation_path) != implementation["file_sha256"]
    ):
        raise GravityClusterCovariancePortabilityError("verifier implementation changed")

    paths: dict[str, Any] = {}
    values: dict[str, dict[str, Any]] = {}
    for binding in config["frozen_v1_bindings"]:
        path = _under(root, binding["path"], "V1 binding")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityClusterCovariancePortabilityError(
                f"frozen V1 file changed: {binding['binding_id']}"
            )
        paths[str(binding["binding_id"])] = path
        if binding["content_sha256"] is not None:
            value = _read_json(path)
            _validate_content_hash(
                value,
                str(binding["content_sha256"]),
                str(binding["binding_id"]),
            )
            values[str(binding["binding_id"])] = value
    return paths, values


def _validate_receipt_lineage(
    config: Mapping[str, Any], paths: Mapping[str, Path], receipts: Mapping[str, Any]
) -> dict[str, Any]:
    reconstruction_config = _read_json(paths["RECONSTRUCTION_CONFIG"])
    scoring_config = _read_json(paths["SCORING_CONFIG"])
    reconstruction_receipt = receipts["RECONSTRUCTION_RECEIPT"]
    scoring_receipt = receipts["SCORING_RECEIPT"]
    binding_by_id = {
        str(row["binding_id"]): row for row in config["frozen_v1_bindings"]
    }
    if reconstruction_receipt.get("config_binding") != {
        "path": binding_by_id["RECONSTRUCTION_CONFIG"]["path"],
        "content_sha256": _sha(reconstruction_config),
    }:
        raise GravityClusterCovariancePortabilityError(
            "reconstruction config cross-binding changed"
        )
    if scoring_receipt.get("config_binding") != {
        "path": binding_by_id["SCORING_CONFIG"]["path"],
        "content_sha256": _sha(scoring_config),
    }:
        raise GravityClusterCovariancePortabilityError(
            "scoring config cross-binding changed"
        )
    expected_members = {
        str(row["cluster"]): (str(row["member"]), str(row["sha256"]))
        for row in config["covariance_members"]
    }
    expected_pressure = {
        str(row["cluster"]): (str(row["path"]), str(row["sha256"]))
        for row in config["standalone_pressure_files"]
    }
    reconstructed = reconstruction_receipt.get("pressure_reconstructions", [])
    if tuple(row.get("cluster") for row in reconstructed) != CLUSTERS:
        raise GravityClusterCovariancePortabilityError(
            "reconstruction receipt population changed"
        )
    for row in reconstructed:
        cluster = str(row["cluster"])
        if (
            (row.get("covariance_member"), row.get("covariance_member_sha256"))
            != expected_members[cluster]
            or (
                row.get("standalone_pressure_path"),
                row.get("standalone_pressure_sha256"),
            )
            != expected_pressure[cluster]
            or len(str(row.get("reconstructed_covariance_sha256", ""))) != 64
        ):
            raise GravityClusterCovariancePortabilityError(
                f"reconstruction evidence changed: {cluster}"
            )
    reconstruction_binding = next(
        row
        for row in scoring_receipt.get("source_bindings", [])
        if row.get("source_id") == "RECONSTRUCTION_RECEIPT"
    )
    if reconstruction_binding != {
        "source_id": "RECONSTRUCTION_RECEIPT",
        "path": binding_by_id["RECONSTRUCTION_RECEIPT"]["path"],
        "file_sha256": binding_by_id["RECONSTRUCTION_RECEIPT"]["file_sha256"],
        "content_sha256": binding_by_id["RECONSTRUCTION_RECEIPT"]["content_sha256"],
    }:
        raise GravityClusterCovariancePortabilityError(
            "scoring-to-reconstruction cross-binding changed"
        )
    if (
        reconstruction_receipt.get("decision")
        != "DEVELOPMENT_PRESSURE_COVARIANCE_PILOT_RECONSTRUCTIBLE_CP5_REMAINS_PARTIAL"
        or reconstruction_receipt.get("claims", {}).get(
            "full_source_covariance_complete"
        )
        is not False
        or scoring_receipt.get("decision")
        != "FAIL_FROZEN_PRESSURE_RANKING_ROBUSTNESS"
        or scoring_receipt.get("CP5_1_status")
        != "DEVELOPMENT_PRESSURE_COVARIANCE_SCORED_NOT_COMPONENT_COMPLETE"
        or scoring_receipt.get("claims", {}).get("CP5_1_complete") is not False
        or scoring_receipt.get("access_boundary", {}).get(
            "independent_target_rows_opened"
        )
        != 0
    ):
        raise GravityClusterCovariancePortabilityError("V1 result boundary changed")
    return {
        "reconstruction_decision": reconstruction_receipt["decision"],
        "scoring_decision": scoring_receipt["decision"],
        "CP5_1_status": scoring_receipt["CP5_1_status"],
        "reconstructed_matrices": len(reconstructed),
        "scored_pressure_rows": scoring_receipt["sample_summary"]["pressure_rows"],
    }


def build_portable_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    paths, receipts = _load_bound_files(root, config)
    pressure_checks = []
    for row in config["standalone_pressure_files"]:
        path = _under(root, row["path"], "standalone pressure")
        if not path.is_file() or _file_sha(path) != row["sha256"]:
            raise GravityClusterCovariancePortabilityError(
                f"standalone pressure file changed: {row['cluster']}"
            )
        pressure_checks.append(
            {
                "cluster": row["cluster"],
                "path": row["path"],
                "sha256": row["sha256"],
                "verified": True,
            }
        )
    lineage = _validate_receipt_lineage(config, paths, receipts)
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "verifier_id": config["verifier_id"],
        "decision": "PASS_PORTABLE_INTEGRITY_EXTERNAL_ARCHIVE_NOT_REQUIRED_OR_INCLUDED",
        "config_binding": {"path": CONFIG_PATH.as_posix(), "content_sha256": _sha(config)},
        "implementation_binding": config["implementation_binding"],
        "official_source": config["official_source"],
        "external_archive_contract": config["external_archive_contract"],
        "frozen_v1_bindings": config["frozen_v1_bindings"],
        "standalone_pressure_checks": pressure_checks,
        "external_covariance_member_manifest": config["covariance_members"],
        "lineage": lineage,
        "portable_integrity_mode": config["portable_integrity_mode"],
        "optional_full_replay": config["optional_full_replay"],
        "counts": {
            "frozen_v1_files_verified": len(paths),
            "receipt_content_hashes_verified": len(receipts),
            "tracked_standalone_pressure_files_verified": len(pressure_checks),
            "external_covariance_members_manifested": len(
                config["covariance_members"]
            ),
            "external_archive_files_read": 0,
            "scientific_payload_rows_read": 0,
            "scientific_scores_computed": 0,
            "network_calls": 0,
        },
        "claims": config["claim_boundary"],
        "limitations": [
            "The X-COP archive is not included, and this receipt does not grant or infer redistribution rights.",
            "Portable integrity verifies frozen files, cross-bindings, and tracked standalone pressure products; it is not a scientific replay.",
            "A full numerical replay remains optional and requires the externally obtained archive at the separately frozen V1 location.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def inspect_external_archive(
    archive_path: Path,
    expected_archive_sha256: str,
    expected_archive_bytes: int,
    members: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    path = archive_path.resolve()
    if (
        not path.is_file()
        or path.stat().st_size != expected_archive_bytes
        or _file_sha(path) != expected_archive_sha256
    ):
        raise GravityClusterCovariancePortabilityError(
            "external archive size or hash changed"
        )
    verified = []
    with tarfile.open(path, "r:gz") as archive:
        for row in members:
            member = str(row["member"])
            try:
                stream = archive.extractfile(member)
            except KeyError as error:
                raise GravityClusterCovariancePortabilityError(
                    f"external covariance member missing: {member}"
                ) from error
            if stream is None:
                raise GravityClusterCovariancePortabilityError(
                    f"external covariance member is not a file: {member}"
                )
            observed = _bytes_sha(stream.read())
            if observed != row["sha256"]:
                raise GravityClusterCovariancePortabilityError(
                    f"external covariance member changed: {member}"
                )
            verified.append(
                {
                    "cluster": row["cluster"],
                    "member": member,
                    "sha256": observed,
                }
            )
    return {
        "decision": "PASS_EXTERNAL_ARCHIVE_PREFLIGHT_FULL_REPLAY_NOT_EXECUTED",
        "archive_sha256": expected_archive_sha256,
        "archive_bytes": expected_archive_bytes,
        "verified_members": verified,
        "scientific_rows_parsed": 0,
        "scientific_scores_computed": 0,
        "full_replay_executed": False,
        "redistribution_rights_verified": False,
    }


def check_external_archive(root: Path, archive_path: Path) -> dict[str, Any]:
    config = load_config(root.resolve())
    archive = config["external_archive_contract"]
    return inspect_external_archive(
        archive_path,
        str(archive["sha256"]),
        int(archive["bytes"]),
        config["covariance_members"],
    )


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_portable_receipt(root):
        raise GravityClusterCovariancePortabilityError("portability receipt changed")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("status")
    subparsers.add_parser("check")
    external = subparsers.add_parser("check-external-archive")
    external.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "status":
        receipt = build_portable_receipt(root)
        output: Any = {
            "decision": receipt["decision"],
            "counts": receipt["counts"],
            "claims": receipt["claims"],
        }
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        output = check_external_archive(root, args.archive)
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
