"""Metadata-only source preflight for the frozen GW170817 propagation test."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(
    "configs/open_gravity_differential_propagation_gw170817_source_preflight_v1.json"
)
MODULE_PATH = Path(
    "src/sigma_theory_compiler/"
    "open_gravity_differential_propagation_gw170817_source_preflight_v1.py"
)
TEST_PATH = Path("tests/test_open_gravity_differential_propagation_gw170817_source_preflight_v1.py")
OUTPUT_PATH = Path(
    "runs/gravity/open-gravity-differential-propagation-gw170817-source-preflight-v1/receipt.json"
)

_SCHEMA = "invariant-open-gravity-differential-propagation-gw170817-source-preflight-1.0"
_RECEIPT_SCHEMA = (
    "invariant-open-gravity-differential-propagation-gw170817-source-preflight-receipt-1.0"
)
_PACKAGE_ID = "open-gravity-differential-propagation-gw170817-source-preflight-v1"
_CONFIG_RAW_SHA256 = "8531007ad3765e04a7514bf022b004d3d7bcb2724590ae6898b8e45573e23f7b"
_CONFIG_CONTENT_SHA256 = "5421fa749fb87052ac57722bd8bb47432d1efc46a59f01fbcaed5d772fcd8bdc"
_MODULE_SEMANTIC_SHA256 = "dc49e366f7b6d48ada8ea29367a1b57e88629843ebc09464a276aa5935caee20"
_TEST_RAW_SHA256 = "fdcefbbce8b6927c919f3ae1899a76595667c4bab5e235b1c593ac9ed87b2993"

_MODULE_PIN_PATTERN = re.compile(rb'(_MODULE_SEMANTIC_SHA256\s*=\s*")[0-9a-f]{64}(")')
_ZERO_HASH = b"0" * 64


class SourcePreflightError(RuntimeError):
    """Raised when the source-only contract changes or fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SourcePreflightError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    normalized, count = _MODULE_PIN_PATTERN.subn(
        rb"\g<1>" + _ZERO_HASH + rb"\g<2>", path.read_bytes()
    )
    _require(count == 1, "module semantic pin count changed")
    return hashlib.sha256(normalized).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SourcePreflightError("invalid JSON") from error
    _require(type(value) is dict, "JSON root must be an object")
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _require(content_sha256(config) == _CONFIG_CONTENT_SHA256, "config semantics changed")
    _require(
        set(config)
        == {
            "schema",
            "package_id",
            "status",
            "purpose",
            "predecessor",
            "official_metadata",
            "products",
            "selection",
            "execution_gate",
            "access",
            "claim_boundary",
            "output",
        },
        "config keys changed",
    )
    _require(config["schema"] == _SCHEMA, "schema changed")
    _require(config["package_id"] == _PACKAGE_ID, "package changed")
    _require(config["output"] == OUTPUT_PATH.as_posix(), "output path changed")
    products = config["products"]
    _require(type(products) is list and len(products) == 3, "product count changed")
    _require([row["detector"] for row in products] == ["H1", "L1", "V1"], "detectors changed")
    _require(len({row["filename"] for row in products}) == 3, "product identity collision")
    for row in products:
        _require(
            set(row)
            == {
                "detector",
                "filename",
                "url",
                "gps_start",
                "duration_seconds",
                "sample_rate_hz",
                "content_length_bytes",
                "published_md5",
                "sha256",
                "payload_opened",
            },
            "product keys changed",
        )
        _require(row["gps_start"] == 1187006834, "GPS start changed")
        _require(row["duration_seconds"] == 4096, "duration changed")
        _require(row["sample_rate_hz"] == 4096, "sample rate changed")
        _require(
            type(row["content_length_bytes"]) is int and row["content_length_bytes"] > 0,
            "byte length invalid",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{32}", row["published_md5"]) is not None, "published MD5 invalid"
        )
        _require(row["sha256"] is None and row["payload_opened"] is False, "payload state changed")
        _require(row["url"].startswith("https://gwosc.org/eventapi/html/"), "source host changed")
    _require(
        sum(row["content_length_bytes"] for row in products) == 378_955_051,
        "total byte ceiling changed",
    )
    gate = config["execution_gate"]
    _require(gate["payload_download_allowed_after_this_receipt"] is False, "download gate changed")
    _require(
        gate["scoring_authority"] is False and gate["real_data_eligible"] is False,
        "authority changed",
    )
    _require(len(gate["blockers"]) == 6, "execution blockers changed")
    access = config["access"]
    _require(access["builder_network_calls"] == 0, "builder network access changed")
    _require(access["builder_payload_files_opened"] == 0, "payload access changed")
    _require(access["builder_payload_rows_opened"] == 0, "payload rows changed")
    _require(access["builder_scores_computed"] == 0, "scores changed")
    claims = config["claim_boundary"]
    _require(
        claims
        == {
            "exact_official_product_metadata_frozen": True,
            "payload_sha256_frozen": False,
            "payload_structure_validated": False,
            "likelihood_frozen": False,
            "observational_fit_tested": False,
            "publication_ready": False,
        },
        "claim boundary changed",
    )


def load_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    validate_config(config)
    return config


def validate_predecessor(config: Mapping[str, Any]) -> dict[str, str]:
    predecessor = config["predecessor"]
    observed: dict[str, str] = {}
    for role in ("config", "module", "test", "receipt"):
        path = Path(predecessor[f"{role}_path"])
        _require(path.is_file(), f"missing predecessor {role}")
        digest = file_sha256(path)
        _require(digest == predecessor[f"{role}_sha256"], f"predecessor {role} changed")
        observed[role] = digest
    return observed


def validate_package_bindings() -> dict[str, str]:
    observed = {
        "config_raw_sha256": file_sha256(CONFIG_PATH),
        "config_content_sha256": content_sha256(_read_json(CONFIG_PATH)),
        "module_semantic_sha256": module_semantic_sha256(),
        "test_raw_sha256": file_sha256(TEST_PATH),
    }
    expected = {
        "config_raw_sha256": _CONFIG_RAW_SHA256,
        "config_content_sha256": _CONFIG_CONTENT_SHA256,
        "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
        "test_raw_sha256": _TEST_RAW_SHA256,
    }
    _require(observed == expected, "package binding changed")
    return observed


def build_receipt(*, validate_files: bool = True) -> dict[str, Any]:
    config = load_config()
    predecessor = validate_predecessor(config)
    package = validate_package_bindings() if validate_files else {}
    products = config["products"]
    receipt: dict[str, Any] = {
        "schema": _RECEIPT_SCHEMA,
        "package_id": _PACKAGE_ID,
        "status": config["status"],
        "decision": "PASS_SOURCE_METADATA_ONLY__BLOCK_PAYLOAD_ACCESS_AND_SCORING",
        "predecessor_sha256": predecessor,
        "package_bindings": package,
        "official_metadata": config["official_metadata"],
        "products": products,
        "product_root_sha256": content_sha256(products),
        "counts": {
            "detectors": 3,
            "products": 3,
            "declared_bytes": sum(row["content_length_bytes"] for row in products),
            "payload_files_opened": 0,
            "payload_rows_opened": 0,
            "scores_computed": 0,
        },
        "selection": config["selection"],
        "execution_gate": config["execution_gate"],
        "access": config["access"],
        "claim_boundary": config["claim_boundary"],
    }
    receipt["content_sha256"] = content_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    observed = receipt.get("content_sha256")
    _require(type(observed) is str, "receipt hash missing")
    body = dict(receipt)
    body.pop("content_sha256")
    _require(content_sha256(body) == observed, "receipt self-hash failed")
    _require(dict(receipt) == build_receipt(), "receipt differs from exact rebuild")


def _receipt_bytes(receipt: Mapping[str, Any]) -> bytes:
    return (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing receipt differs")
        return "EXISTING_IDENTICAL"
    descriptor, temporary_name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            _require(path.read_bytes() == payload, "receipt race differs")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def write_receipt() -> str:
    return _atomic_no_clobber(OUTPUT_PATH, _receipt_bytes(build_receipt()))


def check_receipt() -> None:
    _require(OUTPUT_PATH.is_file(), "receipt missing")
    validate_receipt(_read_json(OUTPUT_PATH))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    command = _parser().parse_args(argv).command
    if command == "build":
        print(write_receipt())
    elif command == "check":
        check_receipt()
        print("VALID")
    else:
        print(build_receipt()["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
