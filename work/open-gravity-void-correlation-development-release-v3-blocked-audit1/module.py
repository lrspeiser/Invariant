"""Source-free v3 repair freezing the internally owned Lane-9 development run."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import inspect
import json
import marshal
import math
import os
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from . import open_gravity_void_correlation_development_release_v1 as v1
from . import open_gravity_void_correlation_development_release_v2 as v2
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v3.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v3.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v3.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v3/receipt.json"
FINAL_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v3"
STAGING_ROOT = REPO_ROOT / "work/open-gravity-void-correlation-development-score-v3-staging"
FAILURE_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v3-failures"
CONSUMPTION_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v3-authorization-consumed"

_CONFIG_RAW_SHA256 = "22a50141d2d434c4b5c24c2c3ba42c761dd0c720df41aa82a83ea19d4e13cf7a"
_CONFIG_CONTENT_SHA256 = "ce3527d40482d46b6d155cd42c7cda2d00ec2163598bd5b9f958fa15f5a42be7"
_MODULE_SEMANTIC_SHA256 = "c9283cd2417366dc3f8a65fc9ad8c79cb1d7030b65f355c83a0d8475e8265283"
_TEST_RAW_SHA256 = "d8a4cf0a1977863ef963b43020d8f4024f84b673bba22fb4fcf58edfcf81e92a"
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


class DevelopmentReleaseV3Error(RuntimeError):
    """Fail-closed v3 release or owned-run violation."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV3Error(message)


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
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v3 config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "v3 config content drift")
    _require(
        value["status"] == "DRAFT_SOURCE_FREE_V2_AUDIT_BLOCK_REPAIR_AWAIT_INDEPENDENT_REAUDIT",
        "v3 config status drift",
    )
    return value


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v3 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v3 test raw drift")
    for name, original in _OWNED_RUNTIME_ORIGINALS.items():
        current = _OWNED_RUNTIME_LOOKUP[name]()
        _require(current is original, f"v3 owned runtime identity drift: {name}")
        _require(
            hashlib.sha256(marshal.dumps(current.__code__)).hexdigest() == _OWNED_RUNTIME_CODE_SHA256[name],
            f"v3 owned runtime code drift: {name}",
        )


def validate_blocked_v2(config: Mapping[str, Any]) -> dict[str, str]:
    blocked = config["blocked_v2"]
    for name in ("config", "module", "test", "receipt"):
        section = blocked[name]
        original = v2.canonical_file(section["path"])
        preserved = v2.canonical_file(section["preserved_path"])
        _require(file_sha256(original) == section["raw_sha256"], f"blocked v2 raw drift: {name}")
        _require(original.read_bytes() == preserved.read_bytes(), f"blocked v2 preservation drift: {name}")
    config_value = json.loads(v2.canonical_file(blocked["config"]["path"]).read_text(encoding="utf-8"))
    _require(content_sha256(config_value) == blocked["config"]["content_sha256"], "blocked v2 config content drift")
    _require(v2.module_semantic_sha256() == blocked["module"]["semantic_sha256"], "blocked v2 module semantic drift")
    receipt = json.loads(v2.canonical_file(blocked["receipt"]["path"]).read_text(encoding="utf-8"))
    _require(receipt["content_sha256"] == blocked["receipt"]["content_sha256"], "blocked v2 receipt content drift")
    _require(receipt["content_sha256"] == v2._self_hash(receipt), "blocked v2 receipt self-hash drift")
    _require(receipt["status"] == blocked["receipt"]["status"], "blocked v2 receipt status drift")
    v2.validate_blocked_v1(v2.load_config())
    _require(v2.check_receipt() == receipt, "blocked v2 full runtime or receipt drift")
    return {
        "v2_receipt_raw_sha256": blocked["receipt"]["raw_sha256"],
        "v2_receipt_content_sha256": blocked["receipt"]["content_sha256"],
    }


def expected_success_access_counts() -> dict[str, int]:
    return dict(load_config()["runner_contract"]["expected_counts"])


def validate_source_bindings(config: Mapping[str, Any]) -> None:
    frozen = v1.load_config()["sources"]
    _require(set(config["sources"]) == {"CF4_TABLE4", "VAST_TABLE1", "VAST_TABLE2", "MASK_U8"}, "v3 source set")
    for name, source in config["sources"].items():
        expected = frozen[name]
        _require(
            all(source[key] == expected[key] for key in source),
            f"v3 source binding differs from frozen v1: {name}",
        )
    _require("PANTHEON_PLUS_DATA" not in config["sources"] and "PANTHEON_PLUS_COVARIANCE" not in config["sources"], "Pantheon source exposed")


def _authorization_binding(contract_receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        **dict(contract_receipt["mutation_freeze"]),
        "receipt_raw_sha256": file_sha256(OUTPUT_PATH),
        "receipt_content_sha256": contract_receipt["content_sha256"],
    }


def validate_authorization_bytes(payload: bytes, contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    authorization = json.loads(payload)
    _require(
        set(authorization)
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
        "v3 authorization exact-key mismatch",
    )
    contract = load_config()["authorization_contract"]
    _require(authorization["schema"] == contract["schema"], "v3 authorization schema drift")
    _require(
        authorization["status"] == contract["status"] and authorization["decision"] == contract["decision"],
        "v3 authorization decision drift",
    )
    _require(_HEX64.fullmatch(str(authorization["authorization_id"])) is not None, "authorization ID syntax")
    _require(authorization["uses_allowed"] == 1 and not isinstance(authorization["uses_allowed"], bool), "authorization uses")
    _require(authorization["hard_seals"] == load_config()["runner_contract"]["hard_seals"], "authorization seals")
    _require(authorization["contract_binding"] == _authorization_binding(contract_receipt), "authorization binding")
    _require(authorization["content_sha256"] == _self_hash(authorization), "authorization self-hash")
    return {**authorization, "raw_sha256": bytes_sha256(payload)}


def _validate_fixed_directory(path: Path, expected: Path) -> None:
    _require(path.is_absolute() and path == expected, "noncanonical fixed directory")
    root = REPO_ROOT.resolve(strict=True)
    _require(REPO_ROOT == root and not REPO_ROOT.is_symlink(), "repository root redirected")
    try:
        relative = path.relative_to(REPO_ROOT)
    except ValueError as error:
        raise DevelopmentReleaseV3Error("fixed directory escapes repository") from error
    cursor = REPO_ROOT
    for part in relative.parts:
        cursor /= part
        _require(not cursor.is_symlink(), "fixed path symlink")
        if cursor.exists():
            _require(cursor.is_dir(), "fixed path component not directory")
    _require(path.resolve(strict=False) == path, "fixed path resolution drift")


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
        "schema": "invariant-open-gravity-void-correlation-development-authorization-consumption-3.0",
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
) -> tuple[dict[str, Any], dict[str, Any]]:
    authorization = validate_authorization_bytes(payload, contract_receipt)
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v3-authorization-consumed",
    )
    CONSUMPTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    _validate_fixed_directory(
        CONSUMPTION_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v3-authorization-consumed",
    )
    marker = _consumption_marker(authorization, contract_receipt)
    target = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
    _require(not target.exists() and not target.is_symlink(), "authorization replay")
    with target.open("xb") as handle:
        handle.write(_pretty(marker))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(CONSUMPTION_DIRECTORY)
    return authorization, marker


def _permutation_order_sha256(orders: Sequence[Sequence[int]]) -> str:
    _require(len(orders) == 10, "permutation order stratum count")
    _require(all(all(isinstance(value, int) and not isinstance(value, bool) for value in order) for order in orders), "order type")
    return bytes_sha256(_canonical([list(order) for order in orders]))


def _order_root(order_hashes: Sequence[str]) -> str:
    _require(all(_HEX64.fullmatch(value) is not None for value in order_hashes), "order hash syntax")
    return bytes_sha256(b"".join(bytes.fromhex(value) for value in order_hashes))


def regenerate_permutations_from_rows(
    rows: Sequence[Mapping[str, Any]],
    permutations: int = _PERMUTATIONS,
) -> dict[str, Any]:
    """Regenerate the frozen PCG64 orders and statistics from scored development rows."""
    _require(np.__version__ == "2.2.6", "NumPy version drift")
    _require(isinstance(permutations, int) and not isinstance(permutations, bool) and permutations > 0, "permutation count")
    y, sigma, luminosity, path, directions, exposure, identifiers = v1._primary_arrays(rows, "L_void_Mpc")
    labels = v1.executor_v3.distance_strata(luminosity, identifiers, 10)
    index_groups = [
        sorted(
            (index for index, label in enumerate(labels) if label == stratum),
            key=lambda index: int(identifiers[index]),
        )
        for stratum in range(10)
    ]
    _require(all(index_groups) and sum(map(len, index_groups)) == len(identifiers), "permutation strata coverage")
    generator = np.random.Generator(np.random.PCG64(_SEED))
    observed = float(v1.executor_v3.profile_grid(y, sigma, path, directions, exposure, identifiers)["one_sided_statistic"])
    statistics: list[float] = []
    order_hashes: list[str] = []
    for _ in range(permutations):
        orders = v1.executor_v3._pcg64_permutation_orders(generator, [len(group) for group in index_groups])
        order_hashes.append(_permutation_order_sha256(orders))
        permuted = [float(value) for value in exposure]
        for stratum, indexes in enumerate(index_groups):
            values = [float(exposure[index]) for index in indexes]
            for target_position, source_position in enumerate(orders[stratum]):
                permuted[indexes[target_position]] = values[source_position]
        statistic = float(
            v1.executor_v3.profile_grid(y, sigma, path, directions, permuted, identifiers)["one_sided_statistic"]
        )
        _require(math.isfinite(statistic), "nonfinite regenerated statistic")
        statistics.append(statistic)
    tail = sum(value >= observed for value in statistics)
    return {
        "observed": observed,
        "permutation_statistics": statistics,
        "tail_count": tail,
        "p_value": (1 + tail) / (permutations + 1),
        "order_hashes": order_hashes,
        "order_root_sha256": _order_root(order_hashes),
    }


def _exact_validate_regenerated(
    rows: Sequence[Mapping[str, Any]],
    claimed: Mapping[str, Any],
    claimed_order_hashes: Sequence[str],
    *,
    permutations: int = _PERMUTATIONS,
) -> dict[str, Any]:
    regenerated = regenerate_permutations_from_rows(rows, permutations)
    _require(float(claimed["observed"]) == regenerated["observed"], "regenerated observed mismatch")
    claimed_statistics = [float(value) for value in claimed["permutation_statistics"]]
    _require(len(claimed_statistics) == permutations, "regenerated permutation count mismatch")
    _require(claimed_statistics == regenerated["permutation_statistics"], "regenerated statistic mismatch")
    _require(list(claimed_order_hashes) == regenerated["order_hashes"], "regenerated order-hash mismatch")
    _require(_order_root(claimed_order_hashes) == regenerated["order_root_sha256"], "regenerated order-root mismatch")
    _require(claimed["tail_count"] == regenerated["tail_count"], "regenerated inclusive tail mismatch")
    _require(float(claimed["p_value"]) == regenerated["p_value"], "regenerated plus-one p mismatch")
    return regenerated


def _shadow_v2_authorization(contract_receipt: Mapping[str, Any]) -> bytes:
    config = v2.load_config()
    value: dict[str, Any] = {
        "schema": config["authorization_contract"]["schema"],
        "status": config["authorization_contract"]["status"],
        "decision": config["authorization_contract"]["decision"],
        "authorization_id": bytes_sha256(b"v3-internal-v2-artifact-validator"),
        "uses_allowed": 1,
        "hard_seals": config["authorization_contract"]["required_hard_seals"],
        "contract_binding": v2._authorization_binding(contract_receipt),
        "content_sha256": "",
    }
    value["content_sha256"] = v2._self_hash(value)
    return v2._pretty(value)


def _v2_validate_artifacts(artifacts: Mapping[str, bytes]) -> None:
    contract = v2.check_receipt()
    authorization = _shadow_v2_authorization(contract)
    shadow = v2.finalize_development_artifacts(artifacts, authorization, contract)
    v2.validate_package_payloads(shadow, authorization, contract)


def build_final_receipt_v3(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
    order_hashes: Sequence[str],
) -> dict[str, Any]:
    _require(set(artifacts) == _ARTIFACT_NAMES, "v3 artifact exact-set mismatch")
    authorization = validate_authorization_bytes(authorization_payload, contract_receipt)
    _require(len(order_hashes) == _PERMUTATIONS, "order hash count")
    order_root_sha256 = _order_root(order_hashes)
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-score-receipt-3.0",
        "package_id": "open-gravity-void-correlation-development-score-v3",
        "status": "PASS_INTERNALLY_OWNED_DEVELOPMENT_SCORE_VALIDATION_CONFIRMATION_PANTHEON_SEALED",
        "decision": summary["classification"]["empirical_label"],
        "authorization": {
            "authorization_id": authorization["authorization_id"],
            "raw_sha256": authorization["raw_sha256"],
            "content_sha256": authorization["content_sha256"],
            "uses_allowed": 1,
        },
        "release_chain": {
            "blocked_v2": load_config()["blocked_v2"],
            "development_release_v3": _authorization_binding(contract_receipt),
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
        "permutation_order_root_sha256": order_root_sha256,
        "counts": {
            "development_rows": summary["development_rows"],
            "eligible_primary_rows": summary["eligible_primary_rows"],
            "partial_mask_rows": summary["partial_mask_rows"],
            "permutations": summary["permutation"]["rows"],
        },
        "countermodels": summary["countermodels"],
        "access_counts": summary["access_counts"],
        "hard_seals": load_config()["runner_contract"]["hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def finalize_artifacts_v3(
    artifacts: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
    order_hashes: Sequence[str],
) -> dict[str, bytes]:
    receipt = build_final_receipt_v3(artifacts, authorization_payload, contract_receipt, order_hashes)
    return {**artifacts, "receipt.json": _pretty(receipt)}


def validate_package_payloads_v3(
    payloads: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == _ARTIFACT_NAMES | {"receipt.json"}, "v3 package file-set mismatch")
    artifacts = {name: payloads[name] for name in _ARTIFACT_NAMES}
    _v2_validate_artifacts(artifacts)
    observed = json.loads(payloads["receipt.json"])
    _require(observed["content_sha256"] == _self_hash(observed), "v3 final receipt self-hash mismatch")
    expected = build_final_receipt_v3(
        artifacts,
        authorization_payload,
        contract_receipt,
        observed["permutation_order_hashes"],
    )
    _require(observed == expected, "v3 final receipt mismatch")
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
    _exact_validate_regenerated(
        rows,
        claimed,
        observed["permutation_order_hashes"],
        permutations=_PERMUTATIONS,
    )
    _require(
        observed["permutation_order_root_sha256"] == _order_root(observed["permutation_order_hashes"]),
        "v3 receipt permutation order root mismatch",
    )
    _require(observed["access_counts"] == expected_success_access_counts(), "v3 final access counts")
    return observed


class _HashingReader:
    """Read-only wrapper hashing the exact compressed bytes consumed by GzipFile."""

    def __init__(self, handle: BinaryIO) -> None:
        self._handle = handle
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        data = self._handle.read(size)
        self.digest.update(data)
        self.bytes_read += len(data)
        return data


class _OwnedDevelopmentRun:
    """Private state whose counters can advance only through owned I/O and scoring."""

    def __init__(
        self,
        secret: object,
        authorization_payload: bytes,
        authorization: Mapping[str, Any],
        contract_receipt: Mapping[str, Any],
        marker: Mapping[str, Any],
    ) -> None:
        _require(secret is _OWNED_RUN_SECRET, "owned runner constructor is not authorized")
        _require(marker == _consumption_marker(authorization, contract_receipt), "owned runner marker drift")
        marker_path = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
        _require(marker_path.is_file() and not marker_path.is_symlink(), "owned runner marker absent")
        _require(json.loads(marker_path.read_text(encoding="utf-8")) == marker, "owned runner marker file drift")
        self._authorization_payload = authorization_payload
        self._authorization = dict(authorization)
        self._contract_receipt = dict(contract_receipt)
        self._config = load_config()
        self._counts = {key: 0 for key in expected_success_access_counts()}
        self._counts["authorization_consumptions"] = 1
        self._decoded_development_ids: list[int] = []
        self._stage = "AUTHORIZED_BEFORE_SOURCE_RESOLUTION"

    def _source_path(self, name: str) -> Path:
        _require(name in {"CF4_TABLE4", "VAST_TABLE1", "VAST_TABLE2", "MASK_U8"}, "owned source name")
        return v2.canonical_file(self._config["sources"][name]["path"])

    def _check_raw(self, name: str, observed_bytes: int, observed_hash: str) -> None:
        source = self._config["sources"][name]
        _require(observed_bytes == source["bytes"], f"{name} byte-count drift")
        _require(observed_hash == source["raw_sha256"], f"{name} raw hash drift")

    def _gzip_lines(self, name: str, open_counter: str, pass_counter: str) -> Iterator[bytes]:
        path = self._source_path(name)
        with path.open("rb") as raw:
            self._counts[open_counter] += 1
            reader = _HashingReader(raw)
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                yield from stream
            self._check_raw(name, reader.bytes_read, reader.digest.hexdigest())
            self._counts[pass_counter] += 1

    def _plain_lines(self, name: str, open_counter: str) -> Iterator[bytes]:
        path = self._source_path(name)
        digest = hashlib.sha256()
        observed_bytes = 0
        with path.open("rb") as handle:
            self._counts[open_counter] += 1
            for line in handle:
                digest.update(line)
                observed_bytes += len(line)
                yield line
        self._check_raw(name, observed_bytes, digest.hexdigest())

    def _read_cf4(self, ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        self._stage = "CF4_OWNED_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        observed = 0
        for source_index, line in enumerate(
            self._gzip_lines("CF4_TABLE4", "cf4_source_opens", "cf4_gzip_passes")
        ):
            _require(source_index < len(ledger), "CF4 row overflow")
            entry = ledger[source_index]
            v2.validate_ledger_entry(entry)
            payload = v1._frame_payload(line, 157)
            _require(source_index == entry["source_index"] and offset == entry["framed_start"], "CF4 ledger order")
            _require(offset + len(line) == entry["framed_end_exclusive"], "CF4 framed offset")
            _require(bytes_sha256(line) == entry["framed_raw_sha256"], "CF4 framed hash")
            _require(bytes_sha256(payload) == entry["payload_raw_sha256"], "CF4 payload hash")
            _require(bytes_sha256(payload[:7]) == entry["identifier_field_raw_sha256"], "CF4 ID hash")
            _require(bytes_sha256(payload[7:]) == entry["opaque_tail_raw_sha256"], "CF4 tail hash")
            identifier, canonical = ids_v1.parse_i7_identifier(payload[:7])
            _require(identifier == entry["identifier"] and canonical == entry["canonical_identifier"], "CF4 ID drift")
            self._counts["cf4_identifier_rows_reverified"] += 1
            if entry["role"] == "development":
                parsed = v2.parse_cf4_development_record_v2(
                    line,
                    entry,
                    source_index=source_index,
                    framed_start=offset,
                )
                rows.append(parsed)
                self._decoded_development_ids.append(identifier)
                self._counts["cf4_development_scientific_rows_decoded"] += 1
            offset += len(line)
            observed += 1
        _require(observed == len(ledger) == 38053, "CF4 exact coverage")
        _require(len(rows) == 22897, "CF4 exact development decode count")
        return rows

    def _read_vast1(self) -> list[dict[str, Any]]:
        self._stage = "VAST1_OWNED_STREAM"
        rows = [v1.parse_vast_table1_record(line) for line in self._plain_lines("VAST_TABLE1", "vast_table1_source_opens")]
        _require(len(rows) == 2347, "VAST table1 row count")
        self._counts["vast_table1_rows_decoded"] = len(rows)
        return rows

    def _read_vast2(self) -> list[dict[str, Any]]:
        self._stage = "VAST2_OWNED_STREAM"
        rows = [
            v1.parse_vast_table2_record(line)
            for line in self._gzip_lines("VAST_TABLE2", "vast_table2_source_opens", "vast_table2_gzip_passes")
        ]
        _require(len(rows) == 80080, "VAST table2 row count")
        self._counts["vast_table2_rows_decoded"] = len(rows)
        return rows

    def _read_mask(self) -> bytes:
        self._stage = "MASK_OWNED_READ"
        path = self._source_path("MASK_U8")
        with path.open("rb") as handle:
            self._counts["mask_source_opens"] += 1
            payload = handle.read()
        self._check_raw("MASK_U8", len(payload), bytes_sha256(payload))
        v1.validate_mask(payload)
        return payload

    def execute(self) -> str:
        ledger = v2.load_identifier_ledger()
        cf4 = self._read_cf4(ledger)
        vast1 = self._read_vast1()
        vast2 = self._read_vast2()
        mask = self._read_mask()
        self._stage = "DERIVE_AND_SCORE"
        geometry = v1.prepare_vast_geometry(vast1, vast2)
        rows = [v1.derive_development_row(row, mask, geometry["spheres_Mpc"]) for row in cf4]
        profile = v1.profile_grid_details(rows)
        countermodels = v1.score_countermodels(rows)
        permutation = regenerate_permutations_from_rows(rows, _PERMUTATIONS)
        self._counts["development_scores"] += 1
        _require(self._counts == expected_success_access_counts(), "owned operation counts incomplete")
        artifacts = v2.assemble_development_artifacts_v2(
            rows,
            ledger,
            profile,
            permutation,
            countermodels,
            self._counts,
            [],
        )
        payloads = finalize_artifacts_v3(
            artifacts,
            self._authorization_payload,
            self._contract_receipt,
            permutation["order_hashes"],
        )
        validate_package_payloads_v3(payloads, self._authorization_payload, self._contract_receipt)
        self._stage = "PROMOTION"
        return _promote_fixed_payloads(payloads, self._authorization_payload, self._contract_receipt)


def _write_owned_failure(run: _OwnedDevelopmentRun, error: Exception) -> None:
    _validate_fixed_directory(
        FAILURE_DIRECTORY,
        REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v3-failures",
    )
    FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    failure: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-owned-development-failure-3.0",
        "status": "RETAINED_OWNED_DEVELOPMENT_FAILURE_NO_PARTIAL_SUCCESS",
        "authorization_id": run._authorization["authorization_id"],
        "stage": run._stage,
        "reason_code": type(error).__name__.upper(),
        "access_counts": dict(run._counts),
        "authorized_development_ids": sorted(set(run._decoded_development_ids)),
        "hard_seals_preserved": load_config()["runner_contract"]["hard_seals"],
        "content_sha256": "",
    }
    failure["content_sha256"] = _self_hash(failure)
    target = FAILURE_DIRECTORY / f"{failure['authorization_id']}-{failure['content_sha256']}.json"
    _require(not target.exists() and not target.is_symlink(), "owned failure receipt collision")
    with target.open("xb") as handle:
        handle.write(_pretty(failure))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(FAILURE_DIRECTORY)


def _promote_fixed_payloads(
    payloads: Mapping[str, bytes],
    authorization_payload: bytes,
    contract_receipt: Mapping[str, Any],
) -> str:
    _validate_fixed_directory(FINAL_DIRECTORY, REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v3")
    _validate_fixed_directory(STAGING_ROOT, REPO_ROOT / "work/open-gravity-void-correlation-development-score-v3-staging")
    if FINAL_DIRECTORY.exists():
        observed = v2._read_fixed_package(FINAL_DIRECTORY)
        validate_package_payloads_v3(observed, authorization_payload, contract_receipt)
        _require(observed == payloads, "existing v3 package differs")
        return "EXISTING_IDENTICAL"
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT)).resolve()
    _require(STAGING_ROOT.resolve() in staging.parents, "v3 staging containment")
    try:
        v2._write_payload_tree(staging, payloads)
        validate_package_payloads_v3(v2._read_fixed_package(staging), authorization_payload, contract_receipt)
        FINAL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
        v2._atomic_directory_promote(staging, FINAL_DIRECTORY)
        _require(not staging.exists(), "v3 staging survived promotion")
        return "PROMOTED_COMPLETE"
    except Exception:
        if staging.exists():
            import shutil

            shutil.rmtree(staging)
        raise


def run_development_once() -> str:
    """The sole production entry; it accepts no caller inputs or reported operations."""
    contract_receipt = check_receipt()
    authorization_path = v2.canonical_file(load_config()["authorization_contract"]["future_path"])
    authorization_payload = authorization_path.read_bytes()
    authorization, marker = _consume_fixed_authorization(authorization_payload, contract_receipt)
    run = _OwnedDevelopmentRun(
        _OWNED_RUN_SECRET,
        authorization_payload,
        authorization,
        contract_receipt,
        marker,
    )
    try:
        return run.execute()
    except Exception as error:
        _write_owned_failure(run, error)
        raise


_OWNED_RUN_SECRET = object()
_OWNED_RUNTIME_ORIGINALS = {
    "run_development_once": run_development_once,
    "runner.execute": _OwnedDevelopmentRun.execute,
    "runner._read_cf4": _OwnedDevelopmentRun._read_cf4,
    "runner._read_vast1": _OwnedDevelopmentRun._read_vast1,
    "runner._read_vast2": _OwnedDevelopmentRun._read_vast2,
    "runner._read_mask": _OwnedDevelopmentRun._read_mask,
    "regenerate_permutations_from_rows": regenerate_permutations_from_rows,
    "_exact_validate_regenerated": _exact_validate_regenerated,
    "validate_package_payloads_v3": validate_package_payloads_v3,
    "_promote_fixed_payloads": _promote_fixed_payloads,
}
_OWNED_RUNTIME_LOOKUP = {
    "run_development_once": lambda: run_development_once,
    "runner.execute": lambda: _OwnedDevelopmentRun.execute,
    "runner._read_cf4": lambda: _OwnedDevelopmentRun._read_cf4,
    "runner._read_vast1": lambda: _OwnedDevelopmentRun._read_vast1,
    "runner._read_vast2": lambda: _OwnedDevelopmentRun._read_vast2,
    "runner._read_mask": lambda: _OwnedDevelopmentRun._read_mask,
    "regenerate_permutations_from_rows": lambda: regenerate_permutations_from_rows,
    "_exact_validate_regenerated": lambda: _exact_validate_regenerated,
    "validate_package_payloads_v3": lambda: validate_package_payloads_v3,
    "_promote_fixed_payloads": lambda: _promote_fixed_payloads,
}
_OWNED_RUNTIME_CODE_SHA256 = {
    name: hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
    for name, function in _OWNED_RUNTIME_ORIGINALS.items()
}


def _synthetic_permutation_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index in range(20):
        distance = 10.0 + index
        direction = (1.0, 0.05 * (index + 1), 0.03 * ((index % 3) + 1))
        exposure = 1.0 + float((7 * index) % 11)
        rows.append(
            {
                "identifier": index + 1,
                "source_index": index,
                "bucket": v1.executor_v3.split_role(index + 1)[0],
                "role": "development",
                "eligible_primary": True,
                "reason_codes": [],
                "cf4": {"Dist": distance},
                "D_path_Mpc": distance,
                "direction": direction,
                "L_void_Mpc": exposure,
                "y": (4.0 * exposure + 0.1 * index) / 299792.458,
                "sigma_s": 0.001,
            }
        )
    return rows


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    signature = inspect.signature(run_development_once)
    small = regenerate_permutations_from_rows(_synthetic_permutation_rows(), 3)
    forged = {
        "observed": small["observed"],
        "permutation_statistics": [1.0, 1.0, 1.0],
        "tail_count": sum(1.0 >= small["observed"] for _ in range(3)),
        "p_value": (1 + sum(1.0 >= small["observed"] for _ in range(3))) / 4,
    }
    forgery_rejected = False
    try:
        _exact_validate_regenerated(
            _synthetic_permutation_rows(),
            forged,
            small["order_hashes"],
            permutations=3,
        )
    except DevelopmentReleaseV3Error:
        forgery_rejected = True
    runner_names = set(run_development_once.__code__.co_names) | set(_OwnedDevelopmentRun.execute.__code__.co_names)
    return [
        {"check_id": "BLOCKED_V1_V2_BYTE_EXACT", "passed": True},
        {"check_id": "SOLE_RUNNER_ACCEPTS_NO_ARGUMENTS", "passed": not signature.parameters},
        {
            "check_id": "ASSERTION_ONLY_V2_GATE_NOT_AUTHORITY",
            "passed": not ({"OneShotDevelopmentGate", "source_open", "cf4_row"} & runner_names),
        },
        {
            "check_id": "PCG64_ORDER_AND_STATISTIC_REGENERATION",
            "passed": len(small["order_hashes"]) == 3 and len(set(small["order_hashes"])) == 3,
        },
        {"check_id": "COHERENT_ALL_ONES_FORGERY_REJECTED", "passed": forgery_rejected},
        {
            "check_id": "NO_SCIENTIFIC_ACCESS",
            "passed": all(value == 0 for value in config["current_access_accounting"].values()),
        },
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    blocked = validate_blocked_v2(config)
    validate_source_bindings(config)
    gates = conformance_gates(config)
    _require(all(gate["passed"] for gate in gates), "v3 source-free conformance failure")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-3.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "blocked_v2": config["blocked_v2"],
        "bindings": blocked,
        "audit_findings": config["audit_findings"],
        "sources_binding_only": config["sources"],
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


def _atomic_no_clobber(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload, "existing v3 receipt differs")
        return "EXISTING_IDENTICAL"
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()
    return "CREATED"


def write_receipt() -> str:
    return _atomic_no_clobber(OUTPUT_PATH, _pretty(build_receipt()))


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "v3 receipt drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_V3_SOURCE_FREE_NO_SCIENTIFIC_ACCESS")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
