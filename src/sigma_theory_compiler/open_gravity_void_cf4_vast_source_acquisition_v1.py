"""Opaque-byte source receipt for the CF4 x VAST void-path test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_void_cf4_vast_source_acquisition_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_cf4_vast_source_acquisition_v1.py")
TEST_PATH = Path("tests/test_open_gravity_void_cf4_vast_source_acquisition_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-cf4-vast-source-acquisition-v1/receipt.json")
EXPECTED_IDS = (
    "CF4_README",
    "CF4_TABLE3_GROUP_METHODS",
    "CF4_TABLE4_GROUP_DISTANCE_VELOCITY",
    "VAST_README",
    "VAST_TABLE1_VOIDFINDER_MAXIMAL_SPHERES",
    "VAST_TABLE2_VOIDFINDER_ALL_SPHERES",
    "VAST_TABLE3_VIDE_REVOLVER_ELLIPSOIDS",
)


class VoidSourceAcquisitionError(RuntimeError):
    """Raised when the source contract or opaque bytes drift."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidSourceAcquisitionError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return hashlib.sha256(_canonical(body)).hexdigest()


def load_config(root: Path = Path(".")) -> dict[str, Any]:
    path = (root / CONFIG_PATH).resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(config, dict), "config root must be an object")
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema") == "invariant-open-gravity-void-cf4-vast-source-acquisition-1.0",
        "schema drift",
    )
    _require(
        config.get("package_id") == "open-gravity-void-cf4-vast-source-acquisition-v1",
        "package drift",
    )
    _require(
        config.get("status") == "PASS_OPAQUE_PUBLIC_SOURCE_BYTES_HASHED_ROWS_UNOPENED",
        "status drift",
    )
    _require(config.get("output_path") == OUTPUT_PATH.as_posix(), "output path drift")
    files = config.get("files")
    _require(isinstance(files, list), "files must be a list")
    _require(tuple(row.get("id") for row in files) == EXPECTED_IDS, "file inventory drift")
    _require(len({row["local_path"] for row in files}) == len(files), "duplicate local path")
    _require(len({row["sha256"] for row in files}) == len(files), "duplicate byte identity")
    for row in files:
        _require(
            set(row)
            >= {
                "id",
                "role",
                "url",
                "local_path",
                "bytes",
                "sha256",
                "expected_records",
                "scientific_response",
            },
            f"incomplete source row: {row.get('id')}",
        )
        _require(str(row["url"]).startswith("https://cdsarc.cds.unistra.fr/"), "non-CDS source")
        _require(
            str(row["local_path"]).startswith("work/private/open-gravity-void-source-v2/"),
            "source path escaped",
        )
        _require(isinstance(row["bytes"], int) and row["bytes"] > 0, "invalid byte count")
        _require(len(row["sha256"]) == 64, "invalid source hash")
    access = config.get("access_accounting")
    _require(
        access
        == {
            "http_head_requests": 8,
            "http_get_requests": 7,
            "downloaded_files": 7,
            "downloaded_bytes": 8340564,
            "opaque_scientific_files_hashed": 5,
            "scientific_rows_decoded": 0,
            "response_values_inspected": 0,
            "scores_computed": 0,
            "model_calls": 0,
            "paid_calls": 0,
        },
        "access accounting drift",
    )
    boundary = config.get("claim_boundary")
    _require(boundary.get("exact_public_source_bytes_frozen") is True, "source seal removed")
    for key in (
        "scientific_rows_opened",
        "law_repaired",
        "real_data_fit",
        "empirical_signal",
        "publication_ready",
    ):
        _require(boundary.get(key) is False, f"claim widened: {key}")
    future = config.get("future_decode_contract")
    _require(
        "never use the published Vpec" in future.get("primary_response", ""),
        "response contract drift",
    )
    _require("never cz/H0" in future.get("target_distance", ""), "distance contract drift")
    _require(
        "separate successor before" in future.get("no_retuning", ""), "pre-response freeze removed"
    )


def _validate_predecessor_and_origin(config: Mapping[str, Any], root: Path) -> None:
    predecessor = config["predecessor"]
    path = root / predecessor["path"]
    _require(sha256_file(path) == predecessor["raw_sha256"], "predecessor raw drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(
        value.get("content_sha256") == predecessor["content_sha256"], "predecessor content drift"
    )
    origin = config["origin_attachment"]
    _require(sha256_file(Path(origin["path"])) == origin["sha256"], "origin attachment drift")


def validate_opaque_sources(
    config: Mapping[str, Any], root: Path = Path(".")
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in config["files"]:
        path = root / entry["local_path"]
        _require(path.is_file(), f"missing source: {entry['id']}")
        observed_bytes = path.stat().st_size
        observed_sha = sha256_file(path)
        _require(observed_bytes == entry["bytes"], f"byte drift: {entry['id']}")
        _require(observed_sha == entry["sha256"], f"hash drift: {entry['id']}")
        rows.append(
            {
                "id": entry["id"],
                "bytes": observed_bytes,
                "sha256": observed_sha,
                "verification": "OPAQUE_BYTES_ONLY_NO_DECODE",
            }
        )
    return rows


def build_receipt(root: Path = Path(".")) -> dict[str, Any]:
    config = load_config(root)
    _validate_predecessor_and_origin(config, root)
    rows = validate_opaque_sources(config, root)
    ordered_root = hashlib.sha256(b"\n".join(_canonical(row) for row in rows)).hexdigest()
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-cf4-vast-source-acquisition-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "ADVANCE_TO_SEPARATELY_SEALED_LAW_AND_DECODE_CONTRACT_BEFORE_OPENING_ROWS",
        "sources": rows,
        "source_bundle_root_sha256": ordered_root,
        "counts": {
            "files": len(rows),
            "bytes": sum(row["bytes"] for row in rows),
            "opaque_scientific_files": 5,
            "scientific_rows_decoded": 0,
            "response_values_inspected": 0,
            "scores": 0,
        },
        "release_contract": config["release_contract"],
        "future_decode_contract": config["future_decode_contract"],
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "bindings": {
            "config_raw_sha256": sha256_file(root / CONFIG_PATH),
            "module_raw_sha256": sha256_file(root / MODULE_PATH),
            "test_raw_sha256": sha256_file(root / TEST_PATH),
            "predecessor_raw_sha256": config["predecessor"]["raw_sha256"],
            "origin_attachment_sha256": config["origin_attachment"]["sha256"],
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(temporary, path)
    except FileExistsError:
        _require(path.read_bytes() == payload, "concurrent receipt differs")
        return "EXISTING_IDENTICAL"
    finally:
        temporary.unlink(missing_ok=True)
    return "CREATED"


def write_receipt(root: Path = Path(".")) -> str:
    return _atomic_no_clobber(root / OUTPUT_PATH, _pretty(build_receipt(root)))


def check_receipt(root: Path = Path(".")) -> None:
    observed = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
    _require(observed == build_receipt(root), "receipt differs from deterministic rebuild")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.action == "build":
        print(write_receipt())
    elif args.action == "check":
        check_receipt()
        print("VALID")
    else:
        receipt = build_receipt()
        print(receipt["status"])
        print(receipt["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
