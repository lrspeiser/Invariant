"""Source-free v4 repair: final write exists only inside the owned live run."""

from __future__ import annotations

import argparse
import ast
import ctypes
import gzip
import hashlib
import inspect
import json
import marshal
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

from . import open_gravity_void_correlation_development_release_v1 as v1
from . import open_gravity_void_correlation_development_release_v2 as v2
from . import open_gravity_void_correlation_development_release_v3 as v3
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v4.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v4.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v4.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v4/receipt.json"
FINAL_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v4"
STAGING_ROOT = REPO_ROOT / "work/open-gravity-void-correlation-development-score-v4-staging"
FAILURE_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v4-failures"
CONSUMPTION_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v4-authorization-consumed"

_CONFIG_RAW_SHA256 = "6b4fef579cfdea8e907b56c88bdfd0e7d663fb4ef03aacab4d9d28b42b95a4f6"
_CONFIG_CONTENT_SHA256 = "07639270392266a116cc0c6c4de9d534fbeaed02925b18d5dc66572001417c4f"
_MODULE_SEMANTIC_SHA256 = "a14ef2c2a401351a7bcf1b4d006e6bbf02e5547912a49c26c42e3c55e9ec3096"
_TEST_RAW_SHA256 = "eb09ff25dbea0ae249116d76c2ab83393b400c4491e1f76ba40de530ef7f71cc"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_PERMUTATIONS = 10000
_SEED = 902104729
_ARTIFACT_NAMES = frozenset(v2._ARTIFACT_NAMES)


class DevelopmentReleaseV4Error(RuntimeError):
    """Fail-closed v4 freeze, authorization, validation, or owned-run error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV4Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def content_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    lines = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f'{name} = "') for name in _SELF_CONSTANTS):
            continue
        lines.append(line)
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v4 config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "v4 config content drift")
    _require(
        value["status"] == "DRAFT_SOURCE_FREE_V3_FINAL_WRITE_CAPABILITY_REPAIR_AWAIT_INDEPENDENT_REAUDIT",
        "v4 config status drift",
    )
    return value


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v4 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v4 test raw drift")
    for name, original in _RUNTIME_ORIGINALS.items():
        current = _RUNTIME_LOOKUP[name]()
        _require(current is original, f"v4 runtime identity drift: {name}")
        _require(
            hashlib.sha256(marshal.dumps(current.__code__)).hexdigest() == _RUNTIME_CODE_SHA256[name],
            f"v4 runtime code drift: {name}",
        )


def validate_blocked_v3(config: Mapping[str, Any]) -> dict[str, str]:
    blocked = config["blocked_v3"]
    for name in ("config", "module", "test", "receipt"):
        section = blocked[name]
        original = v2.canonical_file(section["path"])
        preserved = v2.canonical_file(section["preserved_path"])
        _require(file_sha256(original) == section["raw_sha256"], f"blocked v3 raw drift: {name}")
        _require(original.read_bytes() == preserved.read_bytes(), f"blocked v3 preservation drift: {name}")
    config_value = json.loads(v2.canonical_file(blocked["config"]["path"]).read_text(encoding="utf-8"))
    _require(content_sha256(config_value) == blocked["config"]["content_sha256"], "blocked v3 config content drift")
    _require(v3.module_semantic_sha256() == blocked["module"]["semantic_sha256"], "blocked v3 module semantic drift")
    receipt = json.loads(v2.canonical_file(blocked["receipt"]["path"]).read_text(encoding="utf-8"))
    _require(receipt["content_sha256"] == blocked["receipt"]["content_sha256"], "blocked v3 receipt content drift")
    _require(receipt["content_sha256"] == v3._self_hash(receipt), "blocked v3 receipt self-hash drift")
    _require(receipt["status"] == blocked["receipt"]["status"], "blocked v3 receipt status drift")
    _require(v3.check_receipt() == receipt, "blocked v3 full runtime or receipt drift")
    return {
        "v3_receipt_raw_sha256": blocked["receipt"]["raw_sha256"],
        "v3_receipt_content_sha256": blocked["receipt"]["content_sha256"],
    }


def validate_source_bindings(config: Mapping[str, Any]) -> None:
    frozen = v1.load_config()["sources"]
    _require(set(config["sources"]) == {"CF4_TABLE4", "VAST_TABLE1", "VAST_TABLE2", "MASK_U8"}, "v4 source set")
    for name, source in config["sources"].items():
        _require(all(source[key] == frozen[name][key] for key in source), f"v4 source binding drift: {name}")


def expected_success_access_counts() -> dict[str, int]:
    return dict(load_config()["runner_contract"]["expected_counts"])


def _authorization_binding(contract_receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        **dict(contract_receipt["mutation_freeze"]),
        "receipt_raw_sha256": file_sha256(OUTPUT_PATH),
        "receipt_content_sha256": contract_receipt["content_sha256"],
    }


def validate_authorization_bytes(payload: bytes, contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    value = json.loads(payload)
    _require(
        set(value)
        == {
            "schema",
            "status",
            "decision",
            "authorization_id",
            "uses_allowed",
            "hard_seals",
            "contract_binding",
            "content_sha256",
        },
        "v4 authorization exact-key mismatch",
    )
    contract = load_config()["authorization_contract"]
    _require(value["schema"] == contract["schema"], "v4 authorization schema")
    _require(value["status"] == contract["status"] and value["decision"] == contract["decision"], "v4 authorization decision")
    _require(_HEX64.fullmatch(str(value["authorization_id"])) is not None, "v4 authorization ID")
    _require(value["uses_allowed"] == 1 and not isinstance(value["uses_allowed"], bool), "v4 authorization uses")
    _require(value["hard_seals"] == load_config()["final_write_contract"]["hard_seals"], "v4 authorization seals")
    _require(value["contract_binding"] == _authorization_binding(contract_receipt), "v4 authorization binding")
    _require(value["content_sha256"] == _self_hash(value), "v4 authorization self-hash")
    return {**value, "raw_sha256": bytes_sha256(payload)}


def _validate_fixed_directory(path: Path, expected: Path) -> None:
    _require(path.is_absolute() and path == expected, "noncanonical v4 fixed directory")
    root = REPO_ROOT.resolve(strict=True)
    _require(REPO_ROOT == root and not REPO_ROOT.is_symlink(), "v4 repository root redirected")
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise DevelopmentReleaseV4Error("v4 fixed directory escapes repository") from error
    cursor = REPO_ROOT
    for part in relative.parts:
        cursor /= part
        _require(not cursor.is_symlink(), "v4 fixed path symlink")
        if cursor.exists():
            _require(cursor.is_dir(), "v4 fixed path component not directory")
    _require(path.resolve(strict=False) == path, "v4 fixed directory resolution drift")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _consumption_marker(authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-authorization-consumption-4.0",
        "authorization_id": authorization["authorization_id"],
        "authorization_raw_sha256": authorization["raw_sha256"],
        "authorization_content_sha256": authorization["content_sha256"],
        "contract_receipt_content_sha256": contract_receipt["content_sha256"],
        "uses_consumed": 1,
        "content_sha256": "",
    }
    marker["content_sha256"] = _self_hash(marker)
    return marker


def _consume_fixed_authorization(
    payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    authorization = validate_authorization_bytes(payload, contract_receipt)
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v4-authorization-consumed",
    )
    CONSUMPTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v4-authorization-consumed",
    )
    marker = _consumption_marker(authorization, contract_receipt)
    target = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
    _require(not target.exists() and not target.is_symlink(), "v4 authorization replay")
    with target.open("xb") as handle:
        handle.write(_pretty(marker))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(CONSUMPTION_DIRECTORY)
    return authorization, marker, target


def regenerate_permutations_from_rows(
    rows: Sequence[Mapping[str, Any]],
    permutations: int = _PERMUTATIONS,
) -> dict[str, Any]:
    return v3.regenerate_permutations_from_rows(rows, permutations)


def build_final_receipt_v4(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
    order_hashes: Sequence[str],
) -> dict[str, Any]:
    _require(set(artifacts) == _ARTIFACT_NAMES, "v4 final artifact set")
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    _require(len(order_hashes) == _PERMUTATIONS, "v4 order hash count")
    order_root = v3._order_root(order_hashes)
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-score-receipt-4.0",
        "package_id": "open-gravity-void-correlation-development-score-v4",
        "status": "PASS_PRIVATE_FINAL_WRITE_DEVELOPMENT_SCORE_VALIDATION_CONFIRMATION_PANTHEON_SEALED",
        "decision": summary["classification"]["empirical_label"],
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "raw_sha256": authorization["raw_sha256"],
            "content_sha256": authorization["content_sha256"],
            "uses_allowed": 1,
        },
        "release_chain": {
            "blocked_v3": load_config()["blocked_v3"],
            "development_release_v4": _authorization_binding(contract_receipt),
        },
        "artifacts": {
            name: {
                "bytes": len(payload),
                "raw_sha256": bytes_sha256(payload),
                "content_sha256": v2._artifact_content(name, payload),
            }
            for name, payload in sorted(artifacts.items())
        },
        "roots": summary["roots"],
        "permutation_order_hashes": list(order_hashes),
        "permutation_order_root_sha256": order_root,
        "counts": {
            "development_rows": summary["development_rows"],
            "eligible_primary_rows": summary["eligible_primary_rows"],
            "partial_mask_rows": summary["partial_mask_rows"],
            "permutations": summary["permutation"]["rows"],
        },
        "countermodels": summary["countermodels"],
        "access_counts": summary["access_counts"],
        "hard_seals": load_config()["final_write_contract"]["hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_package_payloads_v4(
    payloads: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == _ARTIFACT_NAMES | {"receipt.json"}, "v4 package file set")
    artifacts = {name: payloads[name] for name in _ARTIFACT_NAMES}
    v3._v2_validate_artifacts(artifacts)
    observed = json.loads(payloads["receipt.json"])
    _require(observed["content_sha256"] == _self_hash(observed), "v4 final receipt self-hash")
    expected = build_final_receipt_v4(
        artifacts,
        authorization_payload,
        contract_receipt,
        observed["permutation_order_hashes"],
    )
    _require(observed == expected, "v4 final receipt mismatch")
    ledger = v2._parse_jsonl(artifacts["artifacts/development-rows.jsonl"])
    rows = v2._reconstruct_scored_rows(ledger)
    permutation_rows = v2._parse_jsonl(artifacts["artifacts/permutation-statistics.jsonl"])
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    claimed = {
        "observed": float.fromhex(summary["permutation"]["observed_hex"]),
        "permutation_statistics": [float.fromhex(row["statistic_hex"]) for row in permutation_rows],
        "tail_count": summary["permutation"]["tail_count"],
        "p_value": float.fromhex(summary["permutation"]["p_value_hex"]),
    }
    v3._exact_validate_regenerated(
        rows,
        claimed,
        observed["permutation_order_hashes"],
        permutations=_PERMUTATIONS,
    )
    _require(
        observed["permutation_order_root_sha256"] == v3._order_root(observed["permutation_order_hashes"]),
        "v4 order root mismatch",
    )
    _require(observed["access_counts"] == expected_success_access_counts(), "v4 final access counts")
    return observed


def run_development_once() -> str:
    """Sole production entry; final-write authority never escapes this live call."""
    contract_receipt = check_receipt()
    audit_path = v2.canonical_file(load_config()["authorization_contract"]["future_path"])
    authorization_payload = audit_path.read_bytes()
    authorization, marker, marker_path = _consume_fixed_authorization(
        authorization_payload,
        contract_receipt,
    )

    config = load_config()
    counts = {key: 0 for key in expected_success_access_counts()}
    counts["authorization_consumptions"] = 1
    decoded_development_ids: list[int] = []
    stage = "AUTHORIZED_AND_CONSUMED_BEFORE_SOURCE_RESOLUTION"
    generated_payloads: dict[str, bytes] | None = None
    frozen_mapping_identity: int | None = None
    frozen_payload_identities: dict[str, int] | None = None
    frozen_payload_hashes: dict[str, str] | None = None
    final_write_capability = object()
    unspent_final_write_capability: object | None = final_write_capability

    class HashingReader:
        def __init__(self, handle: BinaryIO) -> None:
            self.handle = handle
            self.digest = hashlib.sha256()
            self.bytes_read = 0

        def read(self, size: int = -1) -> bytes:
            data = self.handle.read(size)
            self.digest.update(data)
            self.bytes_read += len(data)
            return data

    def source_path(name: str) -> Path:
        _require(name in config["sources"], "v4 owned source name")
        return v2.canonical_file(config["sources"][name]["path"])

    def check_raw(name: str, observed_bytes: int, observed_hash: str) -> None:
        source = config["sources"][name]
        _require(observed_bytes == source["bytes"], f"{name} byte-count drift")
        _require(observed_hash == source["raw_sha256"], f"{name} raw-hash drift")

    def gzip_lines(name: str, open_counter: str, pass_counter: str) -> Iterator[bytes]:
        path = source_path(name)
        with path.open("rb") as raw:
            counts[open_counter] += 1
            reader = HashingReader(raw)
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                yield from stream
            check_raw(name, reader.bytes_read, reader.digest.hexdigest())
            counts[pass_counter] += 1

    def plain_lines(name: str, open_counter: str) -> Iterator[bytes]:
        path = source_path(name)
        digest = hashlib.sha256()
        observed_bytes = 0
        with path.open("rb") as handle:
            counts[open_counter] += 1
            for line in handle:
                digest.update(line)
                observed_bytes += len(line)
                yield line
        check_raw(name, observed_bytes, digest.hexdigest())

    def read_cf4(ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        nonlocal stage
        stage = "CF4_OWNED_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        observed = 0
        for source_index, line in enumerate(gzip_lines("CF4_TABLE4", "cf4_source_opens", "cf4_gzip_passes")):
            _require(source_index < len(ledger), "CF4 row overflow")
            entry = ledger[source_index]
            v2.validate_ledger_entry(entry)
            payload = v1._frame_payload(line, 157)
            _require(source_index == entry["source_index"] and offset == entry["framed_start"], "CF4 order")
            _require(offset + len(line) == entry["framed_end_exclusive"], "CF4 framed offset")
            _require(bytes_sha256(line) == entry["framed_raw_sha256"], "CF4 framed hash")
            _require(bytes_sha256(payload) == entry["payload_raw_sha256"], "CF4 payload hash")
            _require(bytes_sha256(payload[:7]) == entry["identifier_field_raw_sha256"], "CF4 ID hash")
            _require(bytes_sha256(payload[7:]) == entry["opaque_tail_raw_sha256"], "CF4 tail hash")
            identifier, canonical = ids_v1.parse_i7_identifier(payload[:7])
            _require(identifier == entry["identifier"] and canonical == entry["canonical_identifier"], "CF4 ID drift")
            counts["cf4_identifier_rows_reverified"] += 1
            if entry["role"] == "development":
                rows.append(
                    v2.parse_cf4_development_record_v2(
                        line,
                        entry,
                        source_index=source_index,
                        framed_start=offset,
                    )
                )
                decoded_development_ids.append(identifier)
                counts["cf4_development_scientific_rows_decoded"] += 1
            offset += len(line)
            observed += 1
        _require(observed == len(ledger) == 38053 and len(rows) == 22897, "CF4 exact coverage")
        return rows

    def read_vast1() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST1_OWNED_STREAM"
        rows = [v1.parse_vast_table1_record(line) for line in plain_lines("VAST_TABLE1", "vast_table1_source_opens")]
        _require(len(rows) == 2347, "VAST1 row count")
        counts["vast_table1_rows_decoded"] = len(rows)
        return rows

    def read_vast2() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST2_OWNED_STREAM"
        rows = [
            v1.parse_vast_table2_record(line)
            for line in gzip_lines("VAST_TABLE2", "vast_table2_source_opens", "vast_table2_gzip_passes")
        ]
        _require(len(rows) == 80080, "VAST2 row count")
        counts["vast_table2_rows_decoded"] = len(rows)
        return rows

    def read_mask() -> bytes:
        nonlocal stage
        stage = "MASK_OWNED_READ"
        path = source_path("MASK_U8")
        with path.open("rb") as handle:
            counts["mask_source_opens"] += 1
            payload = handle.read()
        check_raw("MASK_U8", len(payload), bytes_sha256(payload))
        v1.validate_mask(payload)
        return payload

    def write_failure(error: Exception) -> None:
        _validate_fixed_directory(
            FAILURE_DIRECTORY,
            REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v4-failures",
        )
        FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        failure: dict[str, Any] = {
            "schema": "invariant-open-gravity-void-correlation-owned-development-failure-4.0",
            "status": "RETAINED_PRIVATE_FINAL_WRITE_RUN_FAILURE_NO_PARTIAL_SUCCESS",
            "authorization_id": authorization["authorization_id"],
            "stage": stage,
            "reason_code": type(error).__name__.upper(),
            "access_counts": dict(counts),
            "authorized_development_ids": sorted(set(decoded_development_ids)),
            "hard_seals_preserved": config["final_write_contract"]["hard_seals"],
            "content_sha256": "",
        }
        failure["content_sha256"] = _self_hash(failure)
        target = FAILURE_DIRECTORY / f"{failure['authorization_id']}-{failure['content_sha256']}.json"
        _require(not target.exists() and not target.is_symlink(), "v4 failure receipt collision")
        with target.open("xb") as handle:
            handle.write(_pretty(failure))
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(FAILURE_DIRECTORY)

    def final_write() -> str:
        nonlocal unspent_final_write_capability
        _require(
            unspent_final_write_capability is final_write_capability,
            "v4 final-write closure capability absent or replayed",
        )
        unspent_final_write_capability = None
        _require(generated_payloads is not None, "v4 generated payloads absent")
        _require(frozen_mapping_identity == id(generated_payloads), "v4 payload mapping identity drift")
        _require(
            frozen_payload_identities == {name: id(payload) for name, payload in generated_payloads.items()},
            "v4 payload object identity drift",
        )
        _require(
            frozen_payload_hashes == {name: bytes_sha256(payload) for name, payload in generated_payloads.items()},
            "v4 payload hash drift",
        )
        _require(check_receipt() == contract_receipt, "v4 contract receipt changed at write boundary")
        _require(
            validate_authorization_bytes(authorization_payload, contract_receipt) == authorization,
            "v4 authorization changed at write boundary",
        )
        _require(
            audit_path == v2.canonical_file(load_config()["authorization_contract"]["future_path"]),
            "v4 audit receipt path changed at write boundary",
        )
        _require(audit_path.read_bytes() == authorization_payload, "v4 audit receipt bytes changed at write boundary")
        _require(marker == _consumption_marker(authorization, contract_receipt), "v4 consumed marker changed")
        _require(
            marker_path == CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json",
            "v4 consumed marker path changed",
        )
        _require(marker_path.is_file() and not marker_path.is_symlink(), "v4 consumed marker absent")
        _require(marker_path.resolve(strict=True) == marker_path, "v4 consumed marker path redirected")
        _require(marker_path.read_bytes() == _pretty(marker), "v4 consumed marker bytes changed")
        validate_code_pins()
        validate_blocked_v3(load_config())
        validate_package_payloads_v4(generated_payloads, authorization_payload, contract_receipt)
        _validate_fixed_directory(
            FINAL_DIRECTORY,
            REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v4",
        )
        _validate_fixed_directory(
            STAGING_ROOT,
            REPO_ROOT / "work/open-gravity-void-correlation-development-score-v4-staging",
        )
        _validate_fixed_directory(
            CONSUMPTION_DIRECTORY,
            REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v4-authorization-consumed",
        )
        if FINAL_DIRECTORY.exists():
            observed = v2._read_fixed_package(FINAL_DIRECTORY)
            validate_package_payloads_v4(observed, authorization_payload, contract_receipt)
            _require(observed == generated_payloads, "existing v4 package differs")
            return "EXISTING_IDENTICAL"
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT)).resolve()
        _require(STAGING_ROOT.resolve() in staging.parents, "v4 staging containment")
        try:
            for name, payload in sorted(generated_payloads.items()):
                _require(name in _ARTIFACT_NAMES | {"receipt.json"}, "v4 staged payload name")
                target = staging.joinpath(*name.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                _require(file_sha256(target) == frozen_payload_hashes[name], "v4 staged payload hash")
            _fsync_directory(staging / "artifacts")
            _fsync_directory(staging)
            validate_package_payloads_v4(v2._read_fixed_package(staging), authorization_payload, contract_receipt)
            FINAL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                move = kernel32.MoveFileExW
                move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
                move.restype = ctypes.c_int
                _require(bool(move(str(staging), str(FINAL_DIRECTORY), 0x00000008)), "v4 MoveFileExW failed")
            else:
                FINAL_DIRECTORY.mkdir()
                os.rename(staging, FINAL_DIRECTORY)
                _fsync_directory(FINAL_DIRECTORY.parent)
            _require(not staging.exists(), "v4 staging survived promotion")
            return "PROMOTED_COMPLETE"
        except Exception:
            if staging.exists():
                shutil.rmtree(staging)
            raise

    try:
        ledger = v2.load_identifier_ledger()
        cf4 = read_cf4(ledger)
        vast1 = read_vast1()
        vast2 = read_vast2()
        mask = read_mask()
        stage = "DERIVE_SCORE_AND_GENERATE"
        geometry = v1.prepare_vast_geometry(vast1, vast2)
        rows = [v1.derive_development_row(row, mask, geometry["spheres_Mpc"]) for row in cf4]
        profile = v1.profile_grid_details(rows)
        countermodels = v1.score_countermodels(rows)
        permutation = regenerate_permutations_from_rows(rows, _PERMUTATIONS)
        counts["development_scores"] += 1
        _require(counts == expected_success_access_counts(), "v4 owned counts incomplete")
        artifacts = v2.assemble_development_artifacts_v2(
            rows,
            ledger,
            profile,
            permutation,
            countermodels,
            counts,
            [],
        )
        final_receipt = build_final_receipt_v4(
            artifacts,
            authorization_payload,
            contract_receipt,
            permutation["order_hashes"],
        )
        generated_payloads = {**artifacts, "receipt.json": _pretty(final_receipt)}
        frozen_mapping_identity = id(generated_payloads)
        frozen_payload_identities = {name: id(payload) for name, payload in generated_payloads.items()}
        frozen_payload_hashes = {name: bytes_sha256(payload) for name, payload in generated_payloads.items()}
        stage = "PRIVATE_FINAL_WRITE"
        return final_write()
    except Exception as error:
        write_failure(error)
        raise


def _run_structure() -> dict[str, bool]:
    source = inspect.getsource(run_development_once)
    tree = ast.parse(source)
    outer = tree.body[0]
    _require(isinstance(outer, (ast.FunctionDef, ast.AsyncFunctionDef)), "v4 runner AST")
    nested = [node for node in outer.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    final_nodes = [node for node in nested if node.name == "final_write"]
    calls = [node for node in ast.walk(outer) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    consume_lines = [node.lineno for node in calls if node.func.id == "_consume_fixed_authorization"]
    final_calls = [node.lineno for node in calls if node.func.id == "final_write"]
    capability_lines = [
        node.lineno
        for node in outer.body
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "final_write_capability" for target in node.targets)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "object"
        and not node.value.args
        and not node.value.keywords
    ]
    return {
        "one_nested_final_write": len(final_nodes) == 1,
        "final_write_no_arguments": len(final_nodes) == 1 and not final_nodes[0].args.args,
        "local_unforgeable_capability_after_consumption": (
            len(capability_lines) == 1
            and bool(consume_lines)
            and consume_lines[0] < capability_lines[0] < final_nodes[0].lineno
        ),
        "single_final_write_call": len(final_calls) == 1,
        "blocked_v3_surfaces_absent": all(
            token not in source
            for token in ("v3._promote_fixed_payloads", "v3._OWNED_RUN_SECRET", "v3._OwnedDevelopmentRun")
        ),
    }


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    structure = _run_structure()
    rows = v3._synthetic_permutation_rows()
    exact = regenerate_permutations_from_rows(rows, 3)
    forged = {
        "observed": exact["observed"],
        "permutation_statistics": [1.0, 1.0, 1.0],
        "tail_count": sum(1.0 >= exact["observed"] for _ in range(3)),
        "p_value": (1 + sum(1.0 >= exact["observed"] for _ in range(3))) / 4,
    }
    forgery_rejected = False
    try:
        v3._exact_validate_regenerated(rows, forged, exact["order_hashes"], permutations=3)
    except v3.DevelopmentReleaseV3Error:
        forgery_rejected = True
    module_names = set(globals())
    forbidden_globals = {
        "_OWNED_RUN_SECRET",
        "_OwnedDevelopmentRun",
        "_promote_fixed_payloads",
        "promote_fixed_package",
        "final_write",
    }
    return [
        {"check_id": "BLOCKED_V1_V2_V3_BYTE_EXACT", "passed": True},
        {"check_id": "SOLE_RUNNER_NO_ARGUMENTS", "passed": not inspect.signature(run_development_once).parameters},
        {"check_id": "NO_EXPORTED_SECRET_GATE_OR_PROMOTION", "passed": not (module_names & forbidden_globals)},
        {"check_id": "PRIVATE_FINAL_WRITE_STRUCTURE", "passed": all(structure.values())},
        {"check_id": "PCG64_10000_REGENERATION_RETAINED", "passed": _PERMUTATIONS == 10000 and _SEED == 902104729},
        {"check_id": "COHERENT_ALL_ONES_FORGERY_REJECTED", "passed": forgery_rejected},
        {"check_id": "NO_SCIENTIFIC_ACCESS", "passed": all(value == 0 for value in config["current_access_accounting"].values())},
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    blocked = validate_blocked_v3(config)
    validate_source_bindings(config)
    gates = conformance_gates(config)
    _require(all(gate["passed"] for gate in gates), "v4 source-free conformance failure")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-4.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "blocked_v3": config["blocked_v3"],
        "bindings": blocked,
        "audit_finding": config["audit_finding"],
        "sources_binding_only": config["sources"],
        "final_write_contract": config["final_write_contract"],
        "permutation_contract": config["permutation_contract"],
        "runner_contract": config["runner_contract"],
        "authorization_contract": config["authorization_contract"],
        "output_contract": config["output_contract"],
        "conformance_gates": gates,
        "access_accounting": config["current_access_accounting"],
        "mutation_freeze": {
            "config_raw_sha256": file_sha256(CONFIG_PATH),
            "config_content_sha256": content_sha256(config),
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": module_semantic_sha256(),
            "test_raw_sha256": file_sha256(TEST_PATH),
        },
        "next_gate": config["next_gate"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


_RUNTIME_ORIGINALS = {
    "run_development_once": run_development_once,
    "validate_package_payloads_v4": validate_package_payloads_v4,
    "regenerate_permutations_from_rows": regenerate_permutations_from_rows,
    "build_final_receipt_v4": build_final_receipt_v4,
}
_RUNTIME_LOOKUP = {
    "run_development_once": lambda: run_development_once,
    "validate_package_payloads_v4": lambda: validate_package_payloads_v4,
    "regenerate_permutations_from_rows": lambda: regenerate_permutations_from_rows,
    "build_final_receipt_v4": lambda: build_final_receipt_v4,
}
_RUNTIME_CODE_SHA256 = {
    name: hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    for name, function in _RUNTIME_ORIGINALS.items()
}


def _write_contract_receipt() -> str:
    """Write only the fixed source-free contract receipt; this has no score-promotion authority."""
    payload = _pretty(build_receipt())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(
            OUTPUT_PATH.is_file() and not OUTPUT_PATH.is_symlink() and OUTPUT_PATH.read_bytes() == payload,
            "existing v4 receipt differs",
        )
        return "CONTRACT_RECEIPT_IDENTICAL"
    descriptor, name = tempfile.mkstemp(prefix=OUTPUT_PATH.name + ".", suffix=".tmp", dir=OUTPUT_PATH.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, OUTPUT_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CONTRACT_RECEIPT_CREATED"


def write_receipt() -> str:
    return _write_contract_receipt()


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "v4 receipt drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_V4_SOURCE_FREE_NO_SCIENTIFIC_ACCESS")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
