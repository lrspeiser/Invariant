"""Zero-response CAMELS fallback source preflight for dissipative capture."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(
    "configs/open_gravity_dissipative_capture_camels_source_preflight_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/open_gravity_dissipative_capture_camels_source_preflight_v1.py"
)
TEST_PATH = Path(
    "tests/test_open_gravity_dissipative_capture_camels_source_preflight_v1.py"
)
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-dissipative-capture-camels-source-preflight-v1/receipt.json"
)
ARTIFACT_DIR = OUTPUT_PATH.parent / "artifacts"
SOURCE_PATH = OUTPUT_PATH.parent / "sources/Nbody_CV_0_IllustrisTNG_CV_0_snap_033.hdf5"
_CONFIG_RAW_SHA256 = "3de3b3c300f914eb465cf8092bd539340f7d732def6f30bf1294b0d750348b59"
_CONFIG_CONTENT_SHA256 = "79ccd841c1743acb0c121918955d0a2a3128fc7beb367137cca07198366d7610"
_MODULE_SEMANTIC_SHA256 = "b700ef5429fb442ab14d5c36dc45e76dd6091594c25f6a4ac7e52c6dd74fc890"
_TEST_RAW_SHA256 = "04e85bc35c4f3818043f31463a57ebe423481e087cc77ef99aaa91989054c102"
_SCHEMA = "invariant-open-gravity-dissipative-capture-camels-source-preflight-1.0"


class CamelsPreflightError(RuntimeError):
    """Raised when a frozen CAMELS source-preflight invariant fails."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CamelsPreflightError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    marker = '_MODULE_SEMANTIC_SHA256 = "'
    start = text.index(marker) + len(marker)
    end = text.index('"', start)
    normalized = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(normalized.encode()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise CamelsPreflightError(f"invalid JSON: {path}") from error


def snapshot_url(config: Mapping[str, Any], suite: str, snap: int) -> str:
    _require(suite in ("hydro", "nbody"), "invalid suite role")
    base = config["candidate"][f"{suite}_base_url"]
    return f"{base}snapshot_{snap:03d}.hdf5"


def group_url(config: Mapping[str, Any], suite: str, snap: int) -> str:
    _require(suite in ("hydro", "nbody"), "invalid suite role")
    if suite == "hydro":
        return (
            "https://users.flatironinstitute.org/~camels/FOF_Subfind/IllustrisTNG/"
            f"CV/CV_0/groups_{snap:03d}.hdf5"
        )
    return config["http_metadata"]["nbody_group_catalogs"]["url_template"].format(snap=snap)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["status"] == "SOURCE_BLOCKED_ZERO_RESPONSE_PREFLIGHT", "status widened")
    snaps = [row["snap"] for row in config["candidate"]["snapshots"]]
    _require(snaps == list(range(62, 91, 2)), "snapshot grid changed")
    metadata = config["http_metadata"]
    for key in ("hydro_snapshots", "hydro_groups", "nbody_snapshots"):
        _require([row["snap"] for row in metadata[key]] == snaps, f"{key} grid changed")
        _require(all(row["bytes"] > 0 and row["etag"] for row in metadata[key]), f"{key} metadata missing")
    blocked_groups = metadata["nbody_group_catalogs"]
    _require(blocked_groups["snapshots"] == snaps, "blocked group grid changed")
    _require(blocked_groups["head_status"] == 403, "DMO group HEAD status changed")
    _require(blocked_groups["one_byte_range_get_status"] == 403, "DMO group GET status changed")
    _require(blocked_groups["exact_bytes"] is None, "DMO exact bytes invented")
    _require(len(metadata["trees_and_matching"]) == 5, "tree/matching manifest changed")
    acquisitions = metadata["local_acquisitions"]
    _require(len(acquisitions) == 1, "bounded acquisition set changed")
    acquisition = acquisitions[0]
    _require(acquisition["id"] == "Z0_GROUP_MATCHING", "acquisition identity changed")
    _require(acquisition["local_path"] == SOURCE_PATH.as_posix(), "acquisition path changed")
    _require(acquisition["bytes"] == 160096, "acquisition size changed")
    _require(
        acquisition["sha256"] == "bc132b1b07342ab7a173f44006e1109402fe8e37b1e41ef223870047ad160c8a",
        "acquisition hash changed",
    )
    _require(
        acquisition["decode_state"]
        == "DOWNLOADED_AND_HASHED_WITHOUT_OPENING_HDF5_STRUCTURE_OR_ROWS",
        "acquisition decode boundary changed",
    )
    gates = config["eligibility_gates"]
    _require(gates["direct_nbody_groups"] is False, "DMO group gate widened")
    _require(gates["documented_cross_tree_object_matching"] is False, "matching gate widened")
    _require(gates["first_pericenter_timing_equivalent_to_tng100"] is False, "cadence gate widened")
    access = config["access_contract"]
    _require(access["scientific_payload_files_downloaded"] == 1, "bounded download count changed")
    _require(access["scientific_payload_bytes_downloaded"] == 160096, "bounded bytes changed")
    _require(access["scientific_tree_or_group_rows_opened"] == 0, "source rows opened")
    _require(access["hdf5_structures_opened"] == 0, "HDF5 structure opened")
    _require(access["scientific_response_rows_opened"] == 0, "response rows opened")
    _require(access["new_real_data_scores"] == 0, "score computed")
    _require(config["outputs"]["receipt"] == OUTPUT_PATH.as_posix(), "canonical receipt changed")
    _require(config["outputs"]["artifact_directory"] == ARTIFACT_DIR.as_posix(), "artifact path changed")


def load_config() -> dict[str, Any]:
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw hash changed")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantics changed")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw hash changed")
    config = _read_json(CONFIG_PATH)
    _require(type(config) is dict, "config is not an object")
    validate_config(config)
    return config


def _validate_predecessor(config: Mapping[str, Any]) -> None:
    predecessor = config["predecessor"]
    path = Path(predecessor["receipt_path"])
    _require(path.is_file(), "predecessor receipt missing")
    _require(file_sha256(path) == predecessor["receipt_sha256"], "predecessor raw hash changed")
    receipt = _read_json(path)
    _require(receipt["content_sha256"] == predecessor["content_sha256"], "predecessor content changed")


def _validate_acquisition(config: Mapping[str, Any]) -> None:
    acquisition = config["http_metadata"]["local_acquisitions"][0]
    _require(SOURCE_PATH.is_file(), "bounded matching source missing")
    _require(SOURCE_PATH.stat().st_size == acquisition["bytes"], "bounded matching source size changed")
    _require(file_sha256(SOURCE_PATH) == acquisition["sha256"], "bounded matching source hash changed")


def source_manifest(config: Mapping[str, Any]) -> dict[str, Any]:
    metadata = config["http_metadata"]
    files: list[dict[str, Any]] = []
    for role, suite, key in (
        ("HYDRO_SNAPSHOT", "hydro", "hydro_snapshots"),
        ("HYDRO_GROUP", "hydro", "hydro_groups"),
        ("NBODY_SNAPSHOT", "nbody", "nbody_snapshots"),
    ):
        for row in metadata[key]:
            url = (
                group_url(config, suite, row["snap"])
                if role.endswith("GROUP")
                else snapshot_url(config, suite, row["snap"])
            )
            files.append(
                {
                    "role": role,
                    "snap": row["snap"],
                    "url": url,
                    "http_status": 200,
                    "bytes": row["bytes"],
                    "etag": row["etag"],
                    "cryptographic_checksum": None,
                }
            )
    acquisitions = {row["id"]: row for row in metadata["local_acquisitions"]}
    for row in metadata["trees_and_matching"]:
        entry = dict(row)
        acquisition = acquisitions.get(row["id"])
        entry["cryptographic_checksum"] = None
        if acquisition is not None:
            entry["local_acquisition"] = {
                "path": acquisition["local_path"],
                "bytes": acquisition["bytes"],
                "sha256": acquisition["sha256"],
                "decode_state": acquisition["decode_state"],
            }
        files.append(entry)
    blocked = []
    for snap, listing_size in zip(
        metadata["nbody_group_catalogs"]["snapshots"],
        metadata["nbody_group_catalogs"]["public_listing_sizes"],
        strict=True,
    ):
        blocked.append(
            {
                "role": "NBODY_GROUP",
                "snap": snap,
                "url": group_url(config, "nbody", snap),
                "http_status": 403,
                "public_listing_size": listing_size,
                "exact_bytes": None,
                "etag": None,
                "cryptographic_checksum": None,
            }
        )
    return {
        "schema": "invariant-open-gravity-camels-source-manifest-1.0",
        "candidate": config["candidate"],
        "accessible_files": files,
        "blocked_files": blocked,
        "accessible_file_count": len(files),
        "blocked_file_count": len(blocked),
        "accessible_bytes": sum(row["bytes"] for row in files),
        "official_cryptographic_checksum_count": 0,
        "local_sha256_acquisition_count": len(acquisitions),
        "field_mapping": config["field_mapping"],
        "eligibility_gates": config["eligibility_gates"],
        "blockers": config["blockers"],
        "scientific_rows_opened": 0,
        "scientific_scores_computed": 0,
    }


def _artifact_payloads(config: Mapping[str, Any]) -> dict[Path, bytes]:
    manifest = source_manifest(config)
    report = (
        b"# CAMELS same-IC dissipative-capture source preflight\n\n"
        b"Decision: **SOURCE BLOCKED**, not a model failure.\n\n"
        b"The exact CV_0 IllustrisTNG/IllustrisTNG_DM pair has 15 directly accessible paired snapshots "
        b"from z=0.95 to z=0, both SubLink trees, all hydro group catalogs, the required hydro gas "
        b"fields, and one z=0 group-matching file. The 160,096-byte matching file was downloaded and "
        b"SHA-256 hashed without opening its HDF5 structure or rows. However, all 15 DMO group catalogs "
        b"return HTTP 403, so their exact bytes, ETags, and required /IDs membership data cannot be "
        b"receipted through the direct public URL. The inspected official documentation also does not "
        b"establish a per-subhalo cross-tree mapping across the history. The cadence can identify merger "
        b"intervals but makes first pericenter and coalescence interval-censored. No payload exposes an "
        b"official cryptographic checksum.\n\n"
        b"TNG100 remains the preferred source. CAMELS maps or global summaries must not replace merger histories.\n"
    )
    return {
        ARTIFACT_DIR / "camels-source-manifest.json": json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        + b"\n",
        ARTIFACT_DIR / "source-blocked-report.md": report,
    }


def build_receipt(config: Mapping[str, Any], payloads: Mapping[Path, bytes]) -> dict[str, Any]:
    manifest = source_manifest(config)
    artifacts = [
        {"path": path.as_posix(), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}
        for path, payload in sorted(payloads.items(), key=lambda row: row[0].as_posix())
    ]
    receipt = {
        "schema": "invariant-open-gravity-dissipative-capture-camels-source-preflight-receipt-1.0",
        "package_id": config["package_id"],
        "status": "SOURCE_BLOCKED_ZERO_RESPONSE_PREFLIGHT",
        "decision": config["decision"],
        "input_sha256": {"predecessor": config["predecessor"]["receipt_sha256"]},
        "package_bindings": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "summary": {
            "candidate_pair": "IllustrisTNG/IllustrisTNG_DM L25n256 CV CV_0",
            "snapshot_count_z_le_1": len(config["candidate"]["snapshots"]),
            "accessible_candidate_files": manifest["accessible_file_count"],
            "blocked_nbody_group_files": manifest["blocked_file_count"],
            "accessible_candidate_bytes": manifest["accessible_bytes"],
            "official_cryptographic_checksums": 0,
            "locally_sha256_hashed_sources": manifest["local_sha256_acquisition_count"],
            "eligibility_gates": config["eligibility_gates"],
            "blocker_count": len(config["blockers"]),
        },
        "artifacts": artifacts,
        "access_accounting": config["access_contract"],
        "claim_boundary": {
            "camels_source_eligible": False,
            "camels_model_tested": False,
            "tng100_replaced": False,
            "scientific_rows_opened": False,
            "bounded_matching_file_acquired_without_decode": True,
            "scientific_scores_computed": False,
        },
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def build() -> dict[str, Any]:
    """Write only the hard-bound canonical artifact and receipt paths."""
    config = load_config()
    _validate_predecessor(config)
    _validate_acquisition(config)
    payloads = _artifact_payloads(config)
    for path, payload in payloads.items():
        _atomic_write(path, payload)
    receipt = build_receipt(config, payloads)
    _atomic_write(OUTPUT_PATH, json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode() + b"\n")
    return receipt


def check() -> dict[str, Any]:
    """Verify canonical outputs without writing or accepting an alternate path."""
    config = load_config()
    _validate_predecessor(config)
    _validate_acquisition(config)
    payloads = _artifact_payloads(config)
    for path, payload in payloads.items():
        _require(path.is_file(), f"artifact missing: {path}")
        _require(path.read_bytes() == payload, f"artifact changed: {path}")
    expected = build_receipt(config, payloads)
    _require(OUTPUT_PATH.is_file(), "canonical receipt missing")
    _require(_read_json(OUTPUT_PATH) == expected, "canonical receipt changed")
    return expected


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    args = parser.parse_args(argv)
    _require(not inspect.signature(build).parameters, "build path surface widened")
    _require(not inspect.signature(check).parameters, "check path surface widened")
    receipt = build() if args.command == "build" else check()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
