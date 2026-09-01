"""Sealed no-row-access contract for the Lane-9 CF4 x VAST screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import open_gravity_void_geometry_source_completion_v3 as geometry_v3
from . import open_gravity_void_gravitational_load_v4 as law_v4

CONFIG_PATH = Path("configs/open_gravity_void_correlation_executor_contract_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/open_gravity_void_correlation_executor_contract_v1.py")
TEST_PATH = Path("tests/test_open_gravity_void_correlation_executor_contract_v1.py")
OUTPUT_PATH = Path("runs/gravity/open-gravity-void-correlation-executor-contract-v1/receipt.json")
_CONFIG_RAW_SHA256 = "458efb549ededa1673cefe8c207b9b2b3462532c8b578364c93a90c014e57c69"
_CONFIG_CONTENT_SHA256 = "3b01f0e1d266e647003e9f3349e4eda65ba10a4ac6272d0fabe19301030e9fad"
_MODULE_SEMANTIC_SHA256 = "357e2095896604d43577ae6e385a71a8f41ef9e9135b9e3e1bc4c136e73c586a"
_TEST_RAW_SHA256 = "b8c750c971caf60ffece143bf0d4dbe3c260bb415c1e52e5fe8da98e2780d950"


class VoidExecutorContractError(RuntimeError):
    """Raised when the frozen executor contract changes or opens data."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidExecutorContractError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in ("_CONFIG_RAW_SHA256", "_CONFIG_CONTENT_SHA256", "_MODULE_SEMANTIC_SHA256", "_TEST_RAW_SHA256"):
        marker = f'{name} = "'
        start = text.index(marker) + len(marker)
        end = text.index('"', start)
        text = text[:start] + "0" * 64 + text[end:]
    return hashlib.sha256(text.encode()).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test pin drift")


def load_config() -> dict[str, Any]:
    validate_code_pins()
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config semantic drift")
    _require(value["status"] == "FROZEN_CONTRACT_ROWS_UNOPENED_AWAIT_INDEPENDENT_REAUDIT", "status drift")
    _require(value["decision"] == "AWAIT_INDEPENDENT_REAUDIT_BEFORE_ANY_ROW_ACCESS", "access gate drift")
    _require(value["access_accounting"] == {"scientific_rows_decoded": 0, "identifier_rows_decoded": 0, "response_values_inspected": 0, "real_scores": 0}, "access accounting drift")
    return value


def validate_schema(name: str, fields: Sequence[Mapping[str, Any]], record_length: int) -> None:
    seen: set[str] = set()
    last_end = 0
    for field in fields:
        start, end = int(field["start"]), int(field["end"])
        _require(field["name"] not in seen, f"duplicate field: {name}")
        _require(1 <= start <= end <= record_length and start > last_end, f"overlap/order error: {name}")
        _require(field["format"][0] in "AIF", f"unsupported format: {name}")
        seen.add(str(field["name"]))
        last_end = end


def identifier_only_from_synthetic_line(line: bytes, record_length: int = 157) -> int:
    """Exercise the frozen identifier slice on synthetic bytes only."""
    _require(len(line) == record_length, "record length mismatch")
    raw = line[0:7]
    _require(all(byte in b" 0123456789" for byte in raw), "invalid identifier bytes")
    text = raw.decode("ascii").strip()
    _require(text.isdigit() and int(text) > 0, "missing identifier")
    return int(text)


def parse_synthetic_field(raw: bytes, field: Mapping[str, Any]) -> str | int | float | None:
    """Validate field rules using caller-supplied synthetic bytes; never opens a source path."""
    start, end = int(field["start"]) - 1, int(field["end"])
    token = raw[start:end]
    _require(all(byte < 128 for byte in token), "non-ASCII field")
    text = token.decode("ascii")
    if not text.strip():
        _require(not field.get("required", False), "required field missing")
        return None
    kind = str(field["format"])[0]
    if kind == "A":
        return text.rstrip(" ")
    if kind == "I":
        stripped = text.strip()
        _require(stripped.lstrip("-").isdigit(), "invalid integer")
        return int(stripped)
    try:
        result = float(text)
    except ValueError as error:
        raise VoidExecutorContractError("invalid float") from error
    _require(math.isfinite(result), "nonfinite float")
    return result


def split_role(identifier: int) -> tuple[int, str]:
    try:
        return law_v4.v3.split_bucket(identifier)
    except law_v4.v3.VoidLoadV3Error as error:
        raise VoidExecutorContractError(str(error)) from error


def _bind_packages(config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    law = config["law"]
    geometry = config["geometry"]
    _require(file_sha256(Path(law["path"])) == law["raw_sha256"], "law raw drift")
    _require(file_sha256(Path(geometry["path"])) == geometry["raw_sha256"], "geometry raw drift")
    law_value = law_v4.check_receipt()
    geometry_value = geometry_v3.check_receipt()
    _require(law_value["content_sha256"] == law["content_sha256"], "law content drift")
    _require(geometry_value["content_sha256"] == geometry["content_sha256"], "geometry content drift")
    return law_value, geometry_value


def build_receipt() -> dict[str, Any]:
    config = load_config()
    law, geometry = _bind_packages(config)
    for name, fields in config["fixed_width_schemas"].items():
        validate_schema(name, fields, config["inputs"][name]["record_length"])
    for row in config["inputs"].values():
        path = Path(row["path"])
        _require(path.is_file() and file_sha256(path) == row["sha256"], "opaque input drift")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-executor-contract-receipt-1.0",
        "package_id": config["package_id"],
        "status": "PASS_FIXED_CONTRACT_ROWS_UNOPENED_AWAIT_INDEPENDENT_REAUDIT",
        "decision": config["decision"],
        "law_content_sha256": law["content_sha256"],
        "geometry_content_sha256": geometry["content_sha256"],
        "input_hashes": {name: row["sha256"] for name, row in config["inputs"].items()},
        "fixed_width_schemas": config["fixed_width_schemas"],
        "parse_contract": config["parse_contract"],
        "geometry_join": config["geometry_join"],
        "response_likelihood": config["response_likelihood"],
        "flow_nuisance": config["flow_nuisance"],
        "split_and_access": config["split_and_access"],
        "excluded": config["excluded"],
        "access_accounting": config["access_accounting"],
        "bindings": {"config_raw_sha256": file_sha256(CONFIG_PATH), "config_content_sha256": content_sha256(config), "module_raw_sha256": file_sha256(MODULE_PATH), "module_semantic_sha256": module_semantic_sha256(), "test_raw_sha256": file_sha256(TEST_PATH), "law_raw_sha256": config["law"]["raw_sha256"], "geometry_raw_sha256": config["geometry"]["raw_sha256"]},
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing output differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    return _atomic_no_clobber(OUTPUT_PATH, json.dumps(build_receipt(), sort_keys=True, indent=2).encode() + b"\n")


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(observed == build_receipt() and observed["content_sha256"] == _self_hash(observed), "receipt drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    else:
        receipt = check_receipt()
        print("VALID" if args.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
