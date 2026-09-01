"""Identifier-only CF4 partition gate for Lane 9.

Only CF4 table4 bytes 1..7 are parsed.  The remaining CF4 payload and every
VAST/Pantheon byte are outside this module's semantic access surface.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import open_gravity_void_correlation_executor_contract_v3 as executor_v3

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_ids_partition_v1.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_ids_partition_v1.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_ids_partition_v1.py"
OUTPUT_ROOT = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-ids-partition-v1"
LEDGER_PATH = OUTPUT_ROOT / "artifacts/identifier-ledger.jsonl"
FAILURES_PATH = OUTPUT_ROOT / "artifacts/failures.json"
SUMMARY_PATH = OUTPUT_ROOT / "artifacts/partition-summary.json"
RECEIPT_PATH = OUTPUT_ROOT / "receipt.json"
_CONFIG_RAW_SHA256 = "acd8d33b2790d7de6c94d5773cc71c7d3d6d530211a03c9ec3b5ad2f1c41579a"
_CONFIG_CONTENT_SHA256 = "a16ff0d95515f45c1a4d3dda2ab58877569e8ce2ad9e4f5e85a418235249ff7e"
_MODULE_SEMANTIC_SHA256 = "7f2cac92b28a33703553c29d7c9c8c4ea6440e0285bffc091c110a7d05d52f17"
_TEST_RAW_SHA256 = "d213dbb44f38c2f323a538ddad3fe3dc1ce88c99331afac65274227de31c6a40"

ID_START = 0
ID_STOP = 7
PAYLOAD_LENGTH = 157
_I7_TOKEN = re.compile(rb" *[+-]?[0-9]+ *\Z", re.ASCII)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_PIN_NAMES = ("_CONFIG_RAW_SHA256", "_CONFIG_CONTENT_SHA256", "_MODULE_SEMANTIC_SHA256", "_TEST_RAW_SHA256")
_ROLE_BY_BUCKET = {
    0: "development",
    1: "development",
    2: "development",
    3: "development",
    4: "development",
    5: "development",
    6: "validation",
    7: "validation",
    8: "confirmation",
    9: "confirmation",
}


class VoidIdsPartitionError(RuntimeError):
    """Raised when the identifier-only release or frozen package fails closed."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise VoidIdsPartitionError(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    text = path.read_text(encoding="utf-8")
    for name in _PIN_NAMES:
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


def canonical_bound_path(relative: str) -> Path:
    _require(isinstance(relative, str) and relative != "", "empty bound path")
    _require("\\" not in relative and ":" not in relative, "non-POSIX or drive-qualified path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and pure.drive == "" and pure.root == "", "absolute bound path")
    _require(all(part not in ("", ".", "..") for part in pure.parts), "noncanonical path component")
    _require(pure.as_posix() == relative, "noncanonical path spelling")
    cursor = REPO_ROOT
    for part in pure.parts:
        cursor = cursor / part
        _require(not cursor.is_symlink(), "symlink in bound path")
    _require(cursor.is_file(), "bound path is not a regular file")
    resolved = cursor.resolve(strict=True)
    try:
        resolved.relative_to(REPO_ROOT)
    except ValueError as error:
        raise VoidIdsPartitionError("bound path escapes repository") from error
    _require(resolved == cursor.absolute(), "bound path aliases target")
    return resolved


def load_config() -> dict[str, Any]:
    validate_code_pins()
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config content drift")
    _require(value["status"] == "AUTHORIZED_IDS_PARTITION_ONLY_NOT_YET_BUILT", "config status drift")
    _require(value["decision"] == "IDS_ONLY_AUTHORIZED_NO_DEVELOPMENT_RELEASE", "decision drift")
    _require(value["source"]["identifier_slice_zero_based_half_open"] == [ID_START, ID_STOP], "identifier slice drift")
    _require(value["source"]["payload_bytes_per_record"] == PAYLOAD_LENGTH, "record length drift")
    _require(set(value["access_barrier"]["implemented_commands"]) == {"build", "check", "status"}, "command surface drift")
    return value


def _validate_json_receipt(path: Path, raw_sha256: str, content_hash: str, status: str) -> dict[str, Any]:
    _require(file_sha256(path) == raw_sha256, "bound receipt raw drift")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == content_hash, "bound receipt content pin drift")
    _require(value.get("content_sha256") == _self_hash(value), "bound receipt self-hash drift")
    _require(value.get("status") == status, "bound receipt status drift")
    return value


def validate_release_gate(config: Mapping[str, Any]) -> dict[str, Any]:
    executor = config["executor_v3"]
    config_path = canonical_bound_path(executor["config"]["path"])
    _require(file_sha256(config_path) == executor["config"]["raw_sha256"], "v3 config raw drift")
    executor_config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(content_sha256(executor_config) == executor["config"]["content_sha256"], "v3 config content drift")

    module_path = canonical_bound_path(executor["module"]["path"])
    _require(file_sha256(module_path) == executor["module"]["raw_sha256"], "v3 module raw drift")
    _require(executor_v3.module_semantic_sha256(module_path) == executor["module"]["semantic_sha256"], "v3 module semantic drift")
    _require(file_sha256(canonical_bound_path(executor["test"]["path"])) == executor["test"]["raw_sha256"], "v3 test raw drift")

    executor_receipt = _validate_json_receipt(
        canonical_bound_path(executor["receipt"]["path"]),
        executor["receipt"]["raw_sha256"],
        executor["receipt"]["content_sha256"],
        executor["receipt"]["status"],
    )
    audit = config["independent_audit_release"]
    audit_receipt = _validate_json_receipt(
        canonical_bound_path(audit["path"]), audit["raw_sha256"], audit["content_sha256"], audit["status"]
    )
    _require(audit_receipt.get("decision") == audit["decision"], "audit decision drift")
    _require(audit_receipt.get("authorization_scope", {}).get("allowed_next_stage") == audit["allowed_next_stage"], "audit stage release drift")
    _require(audit_receipt.get("audit_independence", {}).get("scientific_source_rows_opened") is False, "audit opened source rows")
    for name in ("config", "module", "test", "receipt"):
        _require(audit_receipt["audited_artifacts"][name]["raw_sha256"] == executor[name]["raw_sha256"], f"audit/v3 {name} mismatch")
    return {
        "executor_receipt_content_sha256": executor_receipt["content_sha256"],
        "audit_receipt_content_sha256": audit_receipt["content_sha256"],
    }


def parse_i7_identifier(token: bytes) -> tuple[int, str]:
    """Parse exactly the authorized seven-byte identifier slice."""
    _require(isinstance(token, bytes) and len(token) == ID_STOP - ID_START, "identifier width mismatch")
    _require(_I7_TOKEN.fullmatch(token) is not None, "invalid I7 identifier grammar")
    identifier = int(token.decode("ascii"))
    _require(identifier > 0, "identifier is not positive")
    canonical = str(identifier)
    _require(canonical.encode("ascii") == executor_v3.canonical_identifier(identifier), "canonical identifier drift")
    return identifier, canonical


def split_bucket_role(identifier: int) -> tuple[int, str]:
    canonical = str(identifier).encode("ascii")
    bucket = int.from_bytes(hashlib.sha256(canonical).digest()[:8], "big", signed=False) % 10
    role = _ROLE_BY_BUCKET[bucket]
    _require(executor_v3.split_role(identifier) == (bucket, role), "executor split drift")
    return bucket, role


def _leaf_hash(entry_without_leaf: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(entry_without_leaf)).hexdigest()


def _digest_root(entries: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(b"".join(bytes.fromhex(str(entry["leaf_sha256"])) for entry in entries)).hexdigest()


def scan_identifier_lines(
    lines: Iterable[bytes],
    *,
    expected_records: int,
    payload_length: int = PAYLOAD_LENGTH,
) -> dict[str, Any]:
    """Scan a decompressed byte-line iterator under the identifier-only barrier."""
    entries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    offset = 0
    framed_rows = 0
    for source_index, framed in enumerate(lines):
        _require(isinstance(framed, bytes), "decompressed record is not bytes")
        framed_rows += 1
        framed_start = offset
        framed_end = framed_start + len(framed)
        offset = framed_end
        base = {
            "source_index": source_index,
            "framed_start": framed_start,
            "framed_end_exclusive": framed_end,
            "framed_raw_sha256": hashlib.sha256(framed).hexdigest(),
        }
        try:
            _require(framed.endswith(b"\n"), "missing terminal LF")
            payload = framed[:-1]
            line_ending_bytes = 1
            if payload.endswith(b"\r"):
                payload = payload[:-1]
                line_ending_bytes = 2
            _require(b"\r" not in payload and b"\n" not in payload, "embedded or repeated line ending")
            _require(len(payload) == payload_length, "payload length mismatch")
            identifier_token = payload[ID_START:ID_STOP]
            identifier, canonical = parse_i7_identifier(identifier_token)
            bucket, role = split_bucket_role(identifier)
            entry_without_leaf: dict[str, Any] = {
                **base,
                "payload_start": framed_start,
                "payload_end_exclusive": framed_start + payload_length,
                "line_ending_bytes": line_ending_bytes,
                "payload_raw_sha256": hashlib.sha256(payload).hexdigest(),
                "identifier_field_raw_sha256": hashlib.sha256(identifier_token).hexdigest(),
                "opaque_tail_raw_sha256": hashlib.sha256(payload[ID_STOP:]).hexdigest(),
                "identifier": identifier,
                "canonical_identifier": canonical,
                "bucket": bucket,
                "role": role,
            }
            entry = dict(entry_without_leaf)
            entry["leaf_sha256"] = _leaf_hash(entry_without_leaf)
            entries.append(entry)
        except VoidIdsPartitionError as error:
            failures.append({**base, "failure": str(error)})
    if framed_rows != expected_records:
        failures.append({"failure": "record count mismatch", "expected": expected_records, "observed": framed_rows})
    return {
        "entries": entries,
        "failures": failures,
        "framed_rows": framed_rows,
        "decompressed_bytes": offset,
    }


def _partition_summary(scan: Mapping[str, Any], config: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    entries = list(scan["entries"])
    failures = list(scan["failures"])
    indexes_by_id: dict[int, list[int]] = defaultdict(list)
    for entry in entries:
        indexes_by_id[int(entry["identifier"])].append(int(entry["source_index"]))
    duplicates = [
        {"identifier": identifier, "source_indexes": indexes}
        for identifier, indexes in sorted(indexes_by_id.items())
        if len(indexes) > 1
    ]
    bucket_counts = {str(bucket): sum(int(entry["bucket"]) == bucket for entry in entries) for bucket in range(10)}
    role_counts = {role: sum(entry["role"] == role for entry in entries) for role in ("development", "validation", "confirmation")}
    minima = config["staged_minima"]
    minimum_failures = [
        {"role": role, "minimum": int(minima[role]), "observed": role_counts[role]}
        for role in role_counts
        if role_counts[role] < int(minima[role])
    ]
    source_order = sorted(entries, key=lambda entry: int(entry["source_index"]))
    canonical_order = sorted(entries, key=lambda entry: (int(entry["identifier"]), int(entry["source_index"])))
    role_roots = {
        role: _digest_root([entry for entry in canonical_order if entry["role"] == role])
        for role in ("development", "validation", "confirmation")
    }
    bucket_roots = {
        str(bucket): _digest_root([entry for entry in canonical_order if int(entry["bucket"]) == bucket])
        for bucket in range(10)
    }
    passed = not failures and not duplicates and not minimum_failures and len(entries) == int(config["source"]["expected_records"])
    failure_ledger = {
        "schema": "invariant-open-gravity-void-correlation-ids-partition-failures-1.0",
        "status": "PASS_NO_FAILURES" if passed else config["block_status"],
        "parse_or_framing_failures": failures,
        "duplicate_identifiers": duplicates,
        "minimum_failures": minimum_failures,
        "scientific_failures": [],
        "scientific_failure_note": "No scientific field was authorized or decoded, so no scientific failure or score exists at this gate.",
    }
    summary = {
        "schema": "invariant-open-gravity-void-correlation-ids-partition-summary-1.0",
        "package_id": config["package_id"],
        "status": config["success_status"] if passed else config["block_status"],
        "source": {
            "path": config["source"]["path"],
            "raw_sha256": config["source"]["raw_sha256"],
            "compressed_bytes": config["source"]["raw_bytes"],
            "decompressed_bytes": int(scan["decompressed_bytes"]),
            "framed_rows": int(scan["framed_rows"]),
            "valid_identifier_rows": len(entries),
        },
        "bucket_counts": bucket_counts,
        "role_counts": role_counts,
        "roots": {
            "source_order_root": _digest_root(source_order),
            "canonical_id_root": _digest_root(canonical_order),
            "role_roots": role_roots,
            "bucket_roots": bucket_roots,
        },
        "duplicate_count": len(duplicates),
        "failure_count": len(failures),
        "minimum_failure_count": len(minimum_failures),
        "staged_minima": config["staged_minima"],
        "next_gate": config["next_gate"],
    }
    return summary, failure_ledger


def _serialize_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False).encode() + b"\n"


def _serialize_ledger(entries: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(entry) + b"\n" for entry in entries)


def _scan_bound_source(config: Mapping[str, Any]) -> dict[str, Any]:
    source = config["source"]
    path = canonical_bound_path(source["path"])
    _require(path.stat().st_size == int(source["raw_bytes"]), "source byte-count drift")
    _require(file_sha256(path) == source["raw_sha256"], "source raw hash drift")
    with gzip.open(path, "rb") as handle:
        return scan_identifier_lines(
            handle,
            expected_records=int(source["expected_records"]),
            payload_length=int(source["payload_bytes_per_record"]),
        )


def _access_accounting(summary: Mapping[str, Any]) -> dict[str, Any]:
    rows = int(summary["source"]["valid_identifier_rows"])
    role_counts = summary["role_counts"]
    return {
        "cf4_source_files_hashed": 1,
        "cf4_source_files_decompressed": 1,
        "cf4_decompression_passes": 1,
        "cf4_decompressed_bytes": int(summary["source"]["decompressed_bytes"]),
        "cf4_full_rows_hashed": int(summary["source"]["framed_rows"]),
        "identifier_rows_decoded": rows,
        "identifier_bytes_decoded": rows * (ID_STOP - ID_START),
        "development_identifier_rows_decoded": int(role_counts["development"]),
        "validation_identifier_rows_decoded": int(role_counts["validation"]),
        "confirmation_identifier_rows_decoded": int(role_counts["confirmation"]),
        "nonidentifier_cf4_bytes_semantically_decoded": 0,
        "scientific_rows_decoded": 0,
        "VAST_files_opened": 0,
        "VAST_fields_decoded": 0,
        "Pantheon_files_opened": 0,
        "Pantheon_values_decoded": 0,
        "response_values_inspected": 0,
        "real_scores": 0,
    }


def _build_receipt_from_artifacts(
    config: Mapping[str, Any],
    release: Mapping[str, Any],
    ledger_bytes: bytes,
    failures_bytes: bytes,
    summary_bytes: bytes,
    summary: Mapping[str, Any],
) -> dict[str, Any]:
    ledger_values = [json.loads(line) for line in ledger_bytes.splitlines()]
    failures = json.loads(failures_bytes)
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-ids-partition-receipt-1.0",
        "package_id": config["package_id"],
        "status": summary["status"],
        "decision": config["decision"],
        "release_bindings": {
            **release,
            "executor_v3_config_raw_sha256": config["executor_v3"]["config"]["raw_sha256"],
            "executor_v3_module_raw_sha256": config["executor_v3"]["module"]["raw_sha256"],
            "executor_v3_test_raw_sha256": config["executor_v3"]["test"]["raw_sha256"],
            "executor_v3_receipt_raw_sha256": config["executor_v3"]["receipt"]["raw_sha256"],
            "independent_audit_raw_sha256": config["independent_audit_release"]["raw_sha256"],
        },
        "source": summary["source"],
        "identifier_grammar": config["identifier_grammar"],
        "record_framing": config["record_framing"],
        "split": config["split"],
        "bucket_counts": summary["bucket_counts"],
        "role_counts": summary["role_counts"],
        "roots": summary["roots"],
        "staged_minima": summary["staged_minima"],
        "failures": {
            "status": failures["status"],
            "parse_or_framing": len(failures["parse_or_framing_failures"]),
            "duplicates": len(failures["duplicate_identifiers"]),
            "minimums": len(failures["minimum_failures"]),
        },
        "artifacts": {
            "identifier_ledger": {
                "path": config["artifacts"]["identifier_ledger"],
                "raw_sha256": hashlib.sha256(ledger_bytes).hexdigest(),
                "content_sha256": content_sha256(ledger_values),
                "rows": len(ledger_values),
            },
            "failure_ledger": {
                "path": config["artifacts"]["failure_ledger"],
                "raw_sha256": hashlib.sha256(failures_bytes).hexdigest(),
                "content_sha256": content_sha256(failures),
            },
            "summary": {
                "path": config["artifacts"]["summary"],
                "raw_sha256": hashlib.sha256(summary_bytes).hexdigest(),
                "content_sha256": content_sha256(summary),
            },
        },
        "access_barrier": config["access_barrier"],
        "access_accounting": _access_accounting(summary),
        "next_gate": config["next_gate"],
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def _make_new_package() -> dict[Path, bytes]:
    config = load_config()
    release = validate_release_gate(config)
    scan = _scan_bound_source(config)
    summary, failures = _partition_summary(scan, config)
    ledger_bytes = _serialize_ledger(scan["entries"])
    failures_bytes = _serialize_json(failures)
    summary_bytes = _serialize_json(summary)
    receipt = _build_receipt_from_artifacts(config, release, ledger_bytes, failures_bytes, summary_bytes, summary)
    return {
        LEDGER_PATH: ledger_bytes,
        FAILURES_PATH: failures_bytes,
        SUMMARY_PATH: summary_bytes,
        RECEIPT_PATH: _serialize_json(receipt),
    }


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.read_bytes() == payload, "existing artifact differs")
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


def _read_ledger_strict(path: Path) -> tuple[bytes, list[dict[str, Any]]]:
    raw = path.read_bytes()
    _require(raw.endswith(b"\n"), "ledger lacks terminal LF")
    values: list[dict[str, Any]] = []
    for line in raw.splitlines(keepends=True):
        _require(line.endswith(b"\n") and not line.endswith(b"\r\n"), "noncanonical ledger line ending")
        value = json.loads(line[:-1])
        _require(_canonical(value) + b"\n" == line, "noncanonical ledger JSON")
        values.append(value)
    return raw, values


def _validate_frozen_entries(entries: Sequence[Mapping[str, Any]], expected_records: int) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    prior_end = 0
    for expected_index, entry in enumerate(entries):
        _require(int(entry["source_index"]) == expected_index, "ledger source-index gap")
        _require(int(entry["framed_start"]) == prior_end, "ledger offset gap")
        _require(int(entry["payload_start"]) == int(entry["framed_start"]), "payload-start drift")
        _require(int(entry["payload_end_exclusive"]) - int(entry["payload_start"]) == PAYLOAD_LENGTH, "payload-length drift")
        _require(int(entry["framed_end_exclusive"]) - int(entry["payload_end_exclusive"]) == int(entry["line_ending_bytes"]), "line-ending offset drift")
        _require(int(entry["line_ending_bytes"]) in (1, 2), "line-ending width drift")
        prior_end = int(entry["framed_end_exclusive"])
        identifier = int(entry["identifier"])
        _require(entry["canonical_identifier"] == str(identifier) and identifier > 0, "identifier canonicalization drift")
        bucket, role = split_bucket_role(identifier)
        _require((int(entry["bucket"]), entry["role"]) == (bucket, role), "split role drift")
        for key in ("framed_raw_sha256", "payload_raw_sha256", "identifier_field_raw_sha256", "opaque_tail_raw_sha256"):
            _require(_HEX64.fullmatch(str(entry[key])) is not None, f"invalid row hash: {key}")
        body = dict(entry)
        leaf = str(body.pop("leaf_sha256"))
        _require(_HEX64.fullmatch(leaf) is not None and leaf == _leaf_hash(body), "row leaf drift")
    if len(entries) != expected_records:
        failures.append({"failure": "valid ledger row count mismatch", "expected": expected_records, "observed": len(entries)})
    scan = {"entries": list(entries), "failures": failures, "framed_rows": len(entries), "decompressed_bytes": prior_end}
    return scan


def check_package() -> dict[str, Any]:
    config = load_config()
    release = validate_release_gate(config)
    source_path = canonical_bound_path(config["source"]["path"])
    _require(source_path.stat().st_size == int(config["source"]["raw_bytes"]), "source byte-count drift")
    _require(file_sha256(source_path) == config["source"]["raw_sha256"], "source raw hash drift")

    ledger_bytes, entries = _read_ledger_strict(LEDGER_PATH)
    failures_bytes = FAILURES_PATH.read_bytes()
    summary_bytes = SUMMARY_PATH.read_bytes()
    failures = json.loads(failures_bytes)
    summary = json.loads(summary_bytes)
    _require(_serialize_json(failures) == failures_bytes, "failure ledger canonical drift")
    _require(_serialize_json(summary) == summary_bytes, "summary canonical drift")
    scan = _validate_frozen_entries(entries, int(config["source"]["expected_records"]))
    recomputed_summary, recomputed_failures = _partition_summary(scan, config)
    _require(recomputed_summary == summary, "partition summary drift")
    _require(recomputed_failures == failures, "failure ledger drift")
    expected_receipt = _build_receipt_from_artifacts(config, release, ledger_bytes, failures_bytes, summary_bytes, summary)
    observed_receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    _require(observed_receipt == expected_receipt, "receipt drift")
    _require(observed_receipt["content_sha256"] == _self_hash(observed_receipt), "receipt self-hash drift")
    return observed_receipt


def write_package() -> str:
    paths = (LEDGER_PATH, FAILURES_PATH, SUMMARY_PATH, RECEIPT_PATH)
    existing = [path.exists() for path in paths]
    if any(existing):
        _require(all(existing), "partial existing IDS package")
        check_package()
        return "EXISTING_VALID_NO_SOURCE_DECOMPRESSION"
    payloads = _make_new_package()
    results = [_atomic_no_clobber(path, payload) for path, payload in payloads.items()]
    _require(all(result == "CREATED" for result in results), "unexpected creation result")
    check_package()
    return "CREATED_ONE_CF4_DECOMPRESSION_PASS"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_package())
    else:
        receipt = check_package()
        print("VALID_NO_SOURCE_DECOMPRESSION" if args.command == "check" else receipt["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
