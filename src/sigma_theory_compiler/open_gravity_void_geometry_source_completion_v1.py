"""Seal the exact VAST Planck2018 survey mask without decoding science rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickletools
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/open_gravity_void_geometry_source_completion_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_geometry_source_completion_v1.py")
TEST_PATH = Path("tests/test_open_gravity_void_geometry_source_completion_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-geometry-source-completion-v1/receipt.json")


class VoidGeometrySourceError(RuntimeError):
    """Raised when the mask or its frozen structure changes."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidGeometrySourceError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def md5_file(path: Path) -> str:
    return hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return hashlib.sha256(_canonical(body)).hexdigest()


def load_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    _require(
        config.get("schema") == "invariant-open-gravity-void-geometry-source-completion-1.0",
        "schema changed",
    )
    _require(config.get("package_id") == OUTPUT_PATH.parent.name, "package changed")
    _require(
        config.get("status") == "PASS_EXACT_PLANCK2018_SURVEY_MASK_HASHED_ROWS_UNOPENED",
        "status changed",
    )
    _require(config.get("output_path") == OUTPUT_PATH.as_posix(), "output changed")
    mask = config["mask"]
    _require(mask["record"] == "10.5281/zenodo.11043278", "record changed")
    _require(mask["version"] == "1.3.1", "release changed")
    _require(mask["safe_structure"]["mask_shape"] == [360, 180], "shape changed")
    _require(mask["safe_structure"]["true_pixels"] == 9133, "mask population changed")
    _require(config["access_accounting"]["scientific_rows_decoded"] == 0, "rows opened")
    _require(config["access_accounting"]["response_values_inspected"] == 0, "response opened")
    _require(config["claim_boundary"]["real_data_fit"] is False, "fit overclaim")
    _require(config["claim_boundary"]["empirical_executor_frozen"] is False, "executor overclaim")


def safe_pickle_structure(path: Path) -> dict[str, Any]:
    """Inspect pickle opcodes and raw ndarray payload without executing pickle globals."""
    data = path.read_bytes()
    allowed_globals = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }
    strings: list[str] = []
    stack_global_count = 0
    bytes_payloads: list[bytes] = []
    stack_strings: list[str] = []
    integers: list[int] = []
    for opcode, argument, _ in pickletools.genops(data):
        if opcode.name == "SHORT_BINUNICODE":
            strings.append(argument)
            stack_strings.append(argument)
        elif opcode.name == "STACK_GLOBAL":
            stack_global_count += 1
        elif opcode.name in {"BINBYTES", "SHORT_BINBYTES"}:
            bytes_payloads.append(argument)
        elif opcode.name in {"BININT", "BININT1", "BININT2"}:
            integers.append(int(argument))
    boolean_payload = next(payload for payload in bytes_payloads if len(payload) == 64800)
    _require(set(boolean_payload) <= {0, 1}, "mask payload is not boolean")
    _require(stack_global_count == len(allowed_globals), "pickle global count changed")
    _require(
        {"numpy.core.multiarray", "_reconstruct", "numpy", "ndarray", "dtype"}
        <= set(stack_strings),
        "pickle global names changed",
    )
    float_payload = next(payload for payload in bytes_payloads if len(payload) == 8)
    import struct

    dist_limits = list(struct.unpack("<2f", float_payload))
    return {
        "pickle_protocol": data[1],
        "allowed_globals": [list(pair) for pair in sorted(allowed_globals)],
        "mask_shape": [360, 180] if 360 in integers and 180 in integers else [],
        "mask_dtype": "bool" if "b1" in strings else "unknown",
        "true_pixels": sum(boolean_payload),
        "mask_resolution": 1,
        "dist_limits_h_inverse_mpc": dist_limits,
    }


def build_receipt() -> dict[str, Any]:
    config = load_config()
    predecessor = config["predecessor"]
    predecessor_path = Path(predecessor["path"])
    _require(sha256_file(predecessor_path) == predecessor["raw_sha256"], "predecessor changed")
    predecessor_value = json.loads(predecessor_path.read_text(encoding="utf-8"))
    _require(predecessor_value["content_sha256"] == predecessor["content_sha256"], "seal changed")
    _require(
        predecessor_value["source_bundle_root_sha256"] == predecessor["source_bundle_root_sha256"],
        "bundle changed",
    )
    mask = config["mask"]
    mask_path = Path(mask["local_path"])
    _require(mask_path.is_file(), "mask missing")
    _require(mask_path.stat().st_size == mask["bytes"], "mask bytes changed")
    _require(sha256_file(mask_path) == mask["sha256"], "mask SHA changed")
    _require(md5_file(mask_path) == mask["md5"], "published MD5 changed")
    structure = safe_pickle_structure(mask_path)
    _require(structure["mask_shape"] == mask["safe_structure"]["mask_shape"], "shape mismatch")
    _require(structure["true_pixels"] == mask["safe_structure"]["true_pixels"], "pixels mismatch")
    _require(
        structure["dist_limits_h_inverse_mpc"]
        == mask["safe_structure"]["dist_limits_h_inverse_mpc"],
        "distance limits changed",
    )
    geometry_root = hashlib.sha256(
        _canonical(
            {
                "predecessor_source_bundle_root_sha256": predecessor["source_bundle_root_sha256"],
                "mask_sha256": mask["sha256"],
                "mask_structure": structure,
            }
        )
    ).hexdigest()
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-geometry-source-completion-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": "SOURCE_GEOMETRY_COMPLETE_FREEZE_EXECUTOR_BEFORE_SCIENCE_ROW_DECODE",
        "mask_structure": structure,
        "geometry_source_root_sha256": geometry_root,
        "geometry_contract": config["geometry_contract"],
        "access_accounting": config["access_accounting"],
        "claim_boundary": config["claim_boundary"],
        "bindings": {
            "config_raw_sha256": sha256_file(CONFIG_PATH),
            "module_raw_sha256": sha256_file(MODULE_PATH),
            "test_raw_sha256": sha256_file(TEST_PATH),
            "predecessor_raw_sha256": predecessor["raw_sha256"],
            "mask_sha256": mask["sha256"],
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
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "concurrent receipt differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(build_receipt()))


def check_receipt() -> dict[str, Any]:
    _require(OUTPUT_PATH.is_file(), "receipt missing")
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == build_receipt(), "receipt does not reproduce")
    _require(observed["content_sha256"] == _self_hash(observed), "self-hash invalid")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    else:
        receipt = check_receipt()
        print("VALID" if args.command == "check" else json.dumps({"status": receipt["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
