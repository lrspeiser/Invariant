"""Append-only test-correction successor for the frozen GW synthetic matrix v1."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from sigma_theory_compiler import (
    open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v1 as v1,
)
from sigma_theory_compiler.sigma_core import SchemaViolation

CONFIG_PATH = Path(
    "configs/open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v2.json"
)
TEST_PATH = Path(
    "tests/test_open_gravity_gw_timeseries_source_anchored_synthetic_injection_matrix_v2.py"
)
OUTPUT_DIR = Path(
    "runs/gravity/open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-v2"
)
RECEIPT_PATH = OUTPUT_DIR / "receipt.json"
_ROOT = Path(__file__).resolve().parents[2]
_EXPECTED_CONFIG_RAW_SHA256 = "6121321f4f8532f68420199a92118f63c85126e9de5e21762fa8fc42413c1757"
_EXPECTED_CONFIG_CONTENT_SHA256 = "ba741fef0995ea325a82aa6d548bac6c8f176506efd7093698db2a77e85e46ac"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaViolation(message)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_bytes(value: Any, *, indent: int | None = None) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":") if indent is None else None,
            indent=indent,
            allow_nan=False,
        )
        + ("\n" if indent is not None else "")
    ).encode("utf-8")


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _repo_path(value: str | Path) -> Path:
    parsed = PurePosixPath(str(value).replace("\\", "/"))
    if parsed.is_absolute() or any(part in {"", ".", ".."} for part in parsed.parts):
        raise SchemaViolation("GW v2 path escaped repository")
    result = (_ROOT / parsed.as_posix()).resolve()
    if not result.is_relative_to(_ROOT):
        raise SchemaViolation("GW v2 path escaped repository")
    return result


def load_config() -> dict[str, Any]:
    return json.loads((_ROOT / CONFIG_PATH).read_text(encoding="utf-8"))


def validate_config(config: Mapping[str, Any], *, verify_hashes: bool = True) -> None:
    if verify_hashes:
        _require(
            _file_sha256(_ROOT / CONFIG_PATH) == _EXPECTED_CONFIG_RAW_SHA256,
            "GW v2 config bytes changed",
        )
    _require(_json_sha256(config) == _EXPECTED_CONFIG_CONTENT_SHA256, "GW v2 config changed")
    _require(
        config["status"] == "FROZEN_SYNTHETIC_ONLY_TEST_CORRECTION_PRE_AUDIT"
        and config["claim_class"] == "SYNTHETIC_DIRECTIONAL_SIGNAL",
        "GW v2 claim boundary changed",
    )
    correction = config["correction"]
    _require(
        correction
        == {
            "blocked_v1_assertion": "RESULT_HASH_COUNT_INCORRECTLY_EXCLUDED_RETAINED_NUMERICAL_INVALID_EXECUTIONS",
            "expected_result_hash_count": 12_240,
            "numerical_invalid_result_hash_count": 680,
            "scored_result_hash_count": 11_560,
            "scope": "TEST_EXPECTATION_ONLY_NO_SCIENTIFIC_OR_MATRIX_SEMANTICS_CHANGED",
        },
        "GW v2 correction scope changed",
    )
    _require(
        all(value == 0 for value in config["access_contract"].values()),
        "GW v2 response seal changed",
    )
    predecessor = config["predecessor"]
    receipt_path = _repo_path(predecessor["receipt_path"])
    if verify_hashes:
        _require(
            receipt_path.is_file()
            and _file_sha256(receipt_path) == predecessor["receipt_raw_sha256"],
            "GW v1 receipt bytes changed",
        )
        _require(
            _file_sha256(_ROOT / v1.CONFIG_PATH) == predecessor["config_raw_sha256"]
            and _file_sha256(Path(v1.__file__)) == predecessor["module_raw_sha256"]
            and _file_sha256(_ROOT / v1.TEST_PATH) == predecessor["test_raw_sha256"],
            "GW v1 package source changed",
        )
        for name, expected in predecessor["artifact_sha256"].items():
            _require(
                _file_sha256(_ROOT / v1.OUTPUT_DIR / name) == expected,
                f"GW v1 artifact changed: {name}",
            )


def derive_receipt() -> dict[str, Any]:
    config = load_config()
    validate_config(config)
    predecessor = config["predecessor"]
    rebuilt = v1.check()
    _require(
        rebuilt["content_sha256"] == predecessor["receipt_content_sha256"],
        "GW v1 exact rebuild content changed",
    )
    ledger = json.loads((_ROOT / v1.LEDGER_PATH).read_text(encoding="utf-8"))
    entries = ledger["entries"]
    execution_eligible = sum(entry["status"] == "ELIGIBLE_NOT_RUN" for entry in entries)
    scored = sum(
        entry["status"]
        in {
            "UNDERPOWERED",
            "AMBIGUOUS_WITH_COMPARATOR",
            "PROMISING_DISTINCT_SIGNATURE",
        }
        for entry in entries
    )
    numerical_invalid = sum(entry["status"] == "NUMERICAL_INVALID" for entry in entries)
    source_blocked = sum(entry["status"] == "SOURCE_BLOCKED" for entry in entries)
    unadapted = sum(entry["status"] == "UNADAPTED" for entry in entries)
    result_hash_count = sum(entry["result_sha256"] is not None for entry in entries)
    _require(
        (
            len(entries),
            execution_eligible,
            scored,
            numerical_invalid,
            source_blocked,
            unadapted,
        )
        == (28_560, 12_240, 11_560, 680, 2_720, 1_360),
        "GW v1 replay status arithmetic changed",
    )
    _require(
        result_hash_count == scored + numerical_invalid == 12_240,
        "GW retained result-hash arithmetic changed",
    )
    receipt_body = {
        "schema": "open-gravity-gw-timeseries-source-anchored-synthetic-injection-matrix-test-correction-receipt-2.0",
        "package_id": config["package_id"],
        "version": config["version"],
        "status": "FROZEN_SYNTHETIC_ONLY_TEST_CORRECTION_COMPLETE_AWAITING_DISTINCT_AUDIT",
        "claim_class": config["claim_class"],
        "scientific_claim": "NONE_SYNTHETIC_ONLY_NOT_SUPPORT_OR_REJECTION",
        "independent_audit_completed": False,
        "distinct_independent_audit_required": True,
        "correction": config["correction"],
        "exact_rebuild": {
            "scenario_count": rebuilt["scenario_count"],
            "attempted_matrix_cell_count": rebuilt["attempted_matrix_cell_count"],
            "scored_matrix_cell_count": rebuilt["scored_matrix_cell_count"],
            "numerical_invalid_cell_count": rebuilt["numerical_invalid_cell_count"],
            "source_blocked_cell_count": rebuilt["source_blocked_cell_count"],
            "unadapted_cell_count": rebuilt["unadapted_cell_count"],
            "replay_entry_count": rebuilt["replay_entry_count"],
            "result_hash_count": result_hash_count,
            "invariance_gates": rebuilt["invariance_gates"],
        },
        "predecessor": predecessor,
        "package_hashes": {
            "config_raw_sha256": _file_sha256(_ROOT / CONFIG_PATH),
            "module_raw_sha256": _file_sha256(Path(__file__)),
            "test_raw_sha256": _file_sha256(_ROOT / TEST_PATH),
        },
        "access_accounting": config["access_contract"],
    }
    return {**receipt_body, "content_sha256": _json_sha256(receipt_body)}


def _write_once(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, f"refusing changed GW v2 artifact: {path}")
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
            _require(path.read_bytes() == payload, "concurrent changed GW v2 receipt")
            return "EXISTING_IDENTICAL"
        return "CREATED"
    finally:
        temporary.unlink(missing_ok=True)


def freeze() -> str:
    return _write_once(_ROOT / RECEIPT_PATH, _json_bytes(derive_receipt(), indent=2))


def check() -> dict[str, Any]:
    receipt = derive_receipt()
    expected = _json_bytes(receipt, indent=2)
    _require(
        (_ROOT / RECEIPT_PATH).is_file() and (_ROOT / RECEIPT_PATH).read_bytes() == expected,
        "stored GW v2 receipt changed",
    )
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "check"))
    args = parser.parse_args(argv)
    if args.command == "freeze":
        print(freeze())
    else:
        print(json.dumps(check(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
