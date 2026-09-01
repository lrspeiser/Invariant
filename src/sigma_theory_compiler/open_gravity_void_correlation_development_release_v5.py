"""Source-free v5 executor contract integrating the audited VAST1 parser."""

from __future__ import annotations

import argparse
import ast
import ctypes
import gzip
import hashlib
import inspect
import json
import marshal
import math
import os
import shutil
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any, BinaryIO

from . import open_gravity_void_correlation_development_release_v1 as v1
from . import open_gravity_void_correlation_development_release_v2 as v2
from . import open_gravity_void_correlation_development_release_v3 as v3
from . import open_gravity_void_correlation_development_release_v4 as v4
from . import open_gravity_void_correlation_ids_partition_v1 as ids_v1
from . import open_gravity_void_vast1_source_parser_contract_v1 as vast1_contract

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_correlation_development_release_v5.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_correlation_development_release_v5.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_correlation_development_release_v5.py"
OUTPUT_PATH = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v5/receipt.json"
FINAL_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v5"
STAGING_ROOT = REPO_ROOT / "work/open-gravity-void-correlation-development-score-v5-staging"
FAILURE_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v5-failures"
CONSUMPTION_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v5-authorization-consumed"

_CONFIG_RAW_SHA256 = "bb46d3931c20fd38ef31b5b419da9430b61944ea30efad0cc1a799954fb9a3be"
_CONFIG_CONTENT_SHA256 = "c5757c7ed649e026f1488114528b04fc1c713cc8d1680a8f8e30db25c024d1fe"
_MODULE_SEMANTIC_SHA256 = "e7ddbd381c95d16e5d3d6f5442e712eeefed81a31223e65365e5cb9324f0a654"
_TEST_RAW_SHA256 = "15223dcfbec26c69fc723b2e98073fb0fd2122f8a45dcb0be956909895a6f060"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = __import__("re").compile(r"[0-9a-f]{64}\Z")
_PERMUTATIONS = 10000
_ARTIFACT_NAMES = frozenset(v2._ARTIFACT_NAMES)


class DevelopmentReleaseV5Error(RuntimeError):
    """Fail-closed v5 freeze, gate, source, or owned-run error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DevelopmentReleaseV5Error(message)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def content_sha256(value: Any) -> str:
    return bytes_sha256(_canonical(value))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1_048_576), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f'{name} = "') for name in _SELF_CONSTANTS):
            continue
        lines.append(line)
    return bytes_sha256("\n".join(lines).encode("utf-8"))


def _self_hash(value: Mapping[str, Any]) -> str:
    body = dict(value)
    body["content_sha256"] = ""
    return content_sha256(body)


def canonical_file(relative: str) -> Path:
    _require(isinstance(relative, str) and relative and "\\" not in relative, "invalid relative path")
    _require(all(part not in {"", ".", ".."} for part in relative.split("/")), "unsafe relative path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute(), "absolute path rejected")
    candidate = REPO_ROOT.joinpath(*pure.parts)
    _require(candidate.resolve(strict=True) == candidate and not candidate.is_symlink(), "path redirected")
    _require(REPO_ROOT in candidate.parents, "path escapes repository")
    return candidate


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v5 config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "v5 config content drift")
    _require(
        value["status"] == "DRAFT_SOURCE_FREE_VAST1_PARSER_INTEGRATED_EXECUTOR_AWAIT_INDEPENDENT_REAUDIT",
        "v5 config status drift",
    )
    _require(value["authority"]["scientific_runs_allowed"] == 0, "v5 run authority introduced")
    _require(value["authority"]["authorizations_may_be_consumed"] is False, "v5 consumption authority introduced")
    return value


def _load_bound_json(section: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"v5 bound raw drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value["content_sha256"] == section["content_sha256"], f"v5 bound content drift: {section['path']}")
    _require(value["content_sha256"] == _self_hash(value), f"v5 bound self-hash drift: {section['path']}")
    if "status" in section:
        _require(value["status"] == section["status"], f"v5 bound status drift: {section['path']}")
    return value


def validate_release_chain(config: Mapping[str, Any]) -> dict[str, str]:
    modules = {"v1": v1, "v2": v2, "v3": v3, "v4": v4}
    for name, section in config["development_packet_chain"].items():
        module = modules[name]
        _require(file_sha256(canonical_file(section["config"]["path"])) == section["config"]["raw_sha256"], f"{name} config drift")
        module_path = canonical_file(section["module"]["path"])
        _require(file_sha256(module_path) == section["module"]["raw_sha256"], f"{name} module raw drift")
        _require(module.module_semantic_sha256(module_path) == section["module"]["semantic_sha256"], f"{name} semantic drift")
        _require(file_sha256(canonical_file(section["test"]["path"])) == section["test"]["raw_sha256"], f"{name} test drift")
        _load_bound_json(section["receipt"])

    parser = config["vast1_parser_contract"]
    parser_config_path = canonical_file(parser["config"]["path"])
    _require(file_sha256(parser_config_path) == parser["config"]["raw_sha256"], "VAST1 parser config raw drift")
    _require(content_sha256(json.loads(parser_config_path.read_text())) == parser["config"]["content_sha256"], "VAST1 parser config content drift")
    parser_module = canonical_file(parser["module"]["path"])
    _require(file_sha256(parser_module) == parser["module"]["raw_sha256"], "VAST1 parser module raw drift")
    _require(vast1_contract.module_semantic_sha256(parser_module) == parser["module"]["semantic_sha256"], "VAST1 parser semantic drift")
    _require(file_sha256(canonical_file(parser["test"]["path"])) == parser["test"]["raw_sha256"], "VAST1 parser test drift")
    parser_receipt = _load_bound_json(parser["receipt"])
    audit = _load_bound_json(parser["independent_audit"])
    _require(audit["decision"] == parser["independent_audit"]["decision"], "VAST1 audit decision drift")
    _require(audit["authority"]["scientific_development_runs_allowed"] == 0, "VAST1 audit scope escalation")
    _require(audit["authority"]["executor_successor_packages_allowed"] == 1, "VAST1 audit successor count drift")
    ledger_path = canonical_file(parser["ledger"]["path"])
    _require(file_sha256(ledger_path) == parser["ledger"]["raw_sha256"], "VAST1 ledger raw drift")
    _require(parser_receipt["row_disposition_root_sha256"] == parser["ledger"]["root_sha256"], "VAST1 ledger root drift")

    failure = _load_bound_json(config["retained_v4_failure"])
    _require(failure["stage"] == "VAST1_OWNED_STREAM" and failure["reason_code"] == "DEVELOPMENTRELEASEV1ERROR", "v4 failure drift")
    _require(failure["access_counts"]["development_scores"] == 0, "v4 failure scored")
    return {
        "v4_receipt_content_sha256": config["development_packet_chain"]["v4"]["receipt"]["content_sha256"],
        "v4_failure_content_sha256": config["retained_v4_failure"]["content_sha256"],
        "vast1_parser_receipt_content_sha256": parser["receipt"]["content_sha256"],
        "vast1_parser_audit_content_sha256": parser["independent_audit"]["content_sha256"],
    }


def parse_vast1_record_v5(framed: bytes, *, source_index: int, framed_start: int) -> dict[str, Any]:
    return vast1_contract.parse_vast1_record(framed, source_index=source_index, framed_start=framed_start)


def validate_vast_duplicate_keys_v5(
    table1_rows: Sequence[tuple[str, int, int]],
    table2_rows: Sequence[tuple[str, int, float, float, float, float]],
) -> dict[str, int]:
    planck = [(str(cosmo), int(void), int(edge)) for cosmo, void, edge in table1_rows if str(cosmo) == "Planck2018"]
    _require(all(void >= 0 and edge in (0, 1, 2) for _, void, edge in planck), "invalid repaired VAST1 key")
    maximal = [(cosmo, void) for cosmo, void, _ in planck]
    _require(len(maximal) == len(set(maximal)), "duplicate repaired VAST1 key")
    edge_by_key = {(cosmo, void): edge for cosmo, void, edge in planck}
    sphere_keys: list[tuple[str, int, float, float, float, float]] = []
    for cosmo, void, x, y, z, radius in table2_rows:
        key = (str(cosmo), int(void), float(x), float(y), float(z), float(radius))
        if key[0] != "Planck2018":
            continue
        _require(key[1] >= 0 and all(math.isfinite(value) for value in key[2:]), "invalid repaired VAST2 key")
        _require((key[0], key[1]) in edge_by_key, "unmatched repaired VAST sphere")
        sphere_keys.append(key)
    _require(len(sphere_keys) == len(set(sphere_keys)), "duplicate repaired VAST2 sphere key")
    retained = sum(edge_by_key[(key[0], key[1])] == 0 for key in sphere_keys)
    return {"retained": retained, "excluded_edge": len(sphere_keys) - retained}


def prepare_vast_geometry_v5(
    table1_rows: Sequence[Mapping[str, Any]],
    table2_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary = validate_vast_duplicate_keys_v5(
        [(str(row["Cosmo"]), int(row["void"]), int(row["edge"])) for row in table1_rows],
        [
            (str(row["Cosmo"]), int(row["void"]), float(row["x"]), float(row["y"]), float(row["z"]), float(row["Rad"]))
            for row in table2_rows
        ],
    )
    edge_by_key: dict[tuple[str, int], int] = {}
    for row in table1_rows:
        if str(row["Cosmo"]) != "Planck2018":
            continue
        key = (str(row["Cosmo"]), int(row["void"]))
        edge_by_key[key] = int(row["edge"])
        expected = v1.geometry_v3.radec_to_xyz(float(row["RAdeg"]), float(row["DEdeg"]), float(row["s"]))
        observed = (float(row["x"]), float(row["y"]), float(row["z"]))
        _require(
            all(math.isclose(observed[index], float(expected[index]), rel_tol=1e-9, abs_tol=1e-8) for index in range(3)),
            "repaired VAST1 coordinate probe failed",
        )
    spheres: list[tuple[tuple[float, float, float], float]] = []
    for row in table2_rows:
        key = (str(row["Cosmo"]), int(row["void"]))
        if key[0] != "Planck2018" or edge_by_key[key] != 0:
            continue
        spheres.append(((float(row["x"]) / 0.674, float(row["y"]) / 0.674, float(row["z"]) / 0.674), float(row["Rad"]) / 0.674))
    _require(len(spheres) == summary["retained"], "repaired retained-sphere count drift")
    return {"spheres_Mpc": spheres, **summary}


def _fixture_conformance(config: Mapping[str, Any]) -> dict[str, Any]:
    fixture = (config["vast1_integration"]["fixture_raw_ascii"] + "\n").encode("ascii")
    row = parse_vast1_record_v5(fixture, source_index=0, framed_start=0)
    return {
        "void_zero_accepted": row["void"] == 0,
        "edge_one_accepted": row["edge"] == 1,
        "payload_length_181_accepted": row["payload_bytes"] == 181,
        "framed_sha256": row["framed_raw_sha256"],
        "payload_sha256": row["payload_raw_sha256"],
    }


def _validate_fixed_directory(path: Path, expected: Path) -> None:
    _require(path == expected and path.is_absolute(), "v5 fixed directory drift")
    root = REPO_ROOT.resolve(strict=True)
    _require(root == REPO_ROOT and not REPO_ROOT.is_symlink(), "v5 repository root redirected")
    _require(path.resolve(strict=False) == path, "v5 fixed directory redirected")
    cursor = REPO_ROOT
    for part in path.relative_to(REPO_ROOT).parts:
        cursor /= part
        _require(not cursor.is_symlink(), "v5 fixed directory symlink")


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _authorization_binding(contract_receipt: Mapping[str, Any], reaudit: Mapping[str, Any]) -> dict[str, str]:
    return {
        "v5_receipt_raw_sha256": file_sha256(OUTPUT_PATH),
        "v5_receipt_content_sha256": contract_receipt["content_sha256"],
        "v5_config_raw_sha256": file_sha256(CONFIG_PATH),
        "v5_module_raw_sha256": file_sha256(MODULE_PATH),
        "v5_test_raw_sha256": file_sha256(TEST_PATH),
        "v5_reaudit_raw_sha256": file_sha256(canonical_file(load_config()["future_gates"]["independent_reaudit_path"])),
        "v5_reaudit_content_sha256": reaudit["content_sha256"],
    }


def _load_future_gates(contract_receipt: Mapping[str, Any]) -> tuple[bytes, dict[str, Any], dict[str, Any]]:
    config = load_config()
    reaudit_path = canonical_file(config["future_gates"]["independent_reaudit_path"])
    reaudit = json.loads(reaudit_path.read_text(encoding="utf-8"))
    _require(reaudit["content_sha256"] == _self_hash(reaudit), "v5 re-audit self-hash")
    _require(reaudit["status"] == config["future_gates"]["independent_reaudit_required_status"], "v5 re-audit status")
    _require(reaudit.get("scientific_run_authority") is False, "v5 re-audit improperly authorizes run")
    authorization_path = canonical_file(config["future_gates"]["one_run_authorization_path"])
    payload = authorization_path.read_bytes()
    authorization = json.loads(payload)
    _require(
        set(authorization)
        == {"schema", "status", "decision", "authorization_id", "uses_allowed", "hard_seals", "binding", "content_sha256"},
        "v5 authorization exact keys",
    )
    _require(authorization["schema"] == "invariant-open-gravity-void-correlation-v5-one-run-authorization-1.0", "v5 authorization schema")
    _require(authorization["status"] == "PASS_ONE_DEVELOPMENT_RUN_ONLY", "v5 authorization status")
    _require(authorization["decision"] == "AUTHORIZE_EXACTLY_ONE_V5_DEVELOPMENT_RUN", "v5 authorization decision")
    _require(_HEX64.fullmatch(str(authorization["authorization_id"])) is not None, "v5 authorization ID")
    _require(authorization["uses_allowed"] == 1 and not isinstance(authorization["uses_allowed"], bool), "v5 authorization uses")
    _require(authorization["hard_seals"] == config["future_executor"]["hard_seals"], "v5 authorization seals")
    _require(authorization["binding"] == _authorization_binding(contract_receipt, reaudit), "v5 authorization binding")
    _require(authorization["content_sha256"] == _self_hash(authorization), "v5 authorization self-hash")
    return payload, authorization, reaudit


def _consumption_marker(authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]) -> dict[str, Any]:
    marker: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-v5-authorization-consumption-1.0",
        "authorization_id": authorization["authorization_id"],
        "authorization_content_sha256": authorization["content_sha256"],
        "contract_receipt_content_sha256": contract_receipt["content_sha256"],
        "uses_consumed": 1,
        "content_sha256": "",
    }
    marker["content_sha256"] = _self_hash(marker)
    return marker


def _consume_authorization(authorization: Mapping[str, Any], contract_receipt: Mapping[str, Any]) -> tuple[dict[str, Any], Path]:
    _validate_fixed_directory(CONSUMPTION_DIRECTORY, REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-release-v5-authorization-consumed")
    CONSUMPTION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    marker = _consumption_marker(authorization, contract_receipt)
    target = CONSUMPTION_DIRECTORY / f"{authorization['authorization_id']}.json"
    _require(not target.exists() and not target.is_symlink(), "v5 authorization replay")
    with target.open("xb") as handle:
        handle.write(_pretty(marker))
        handle.flush()
        os.fsync(handle.fileno())
    _fsync_directory(CONSUMPTION_DIRECTORY)
    return marker, target


def _expected_success_counts() -> dict[str, int]:
    return {
        "authorization_consumptions": 1,
        "cf4_source_opens": 1,
        "cf4_gzip_passes": 1,
        "cf4_identifier_rows_reverified": 38053,
        "cf4_development_scientific_rows_decoded": 22897,
        "cf4_validation_scientific_rows_decoded": 0,
        "cf4_confirmation_scientific_rows_decoded": 0,
        "vast_table1_source_opens": 1,
        "vast_table1_rows_decoded": 2347,
        "vast_table2_source_opens": 1,
        "vast_table2_gzip_passes": 1,
        "vast_table2_rows_decoded": 80080,
        "mask_source_opens": 1,
        "pantheon_source_opens": 0,
        "development_scores": 1,
    }


def build_final_receipt_v5(
    artifacts: Mapping[str, bytes],
    authorization: Mapping[str, Any],
    contract_receipt: Mapping[str, Any],
    reaudit: Mapping[str, Any],
    order_hashes: Sequence[str],
) -> dict[str, Any]:
    _require(set(artifacts) == _ARTIFACT_NAMES and len(order_hashes) == _PERMUTATIONS, "v5 final artifact/order set")
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-score-receipt-5.0",
        "package_id": "open-gravity-void-correlation-development-score-v5",
        "status": "PASS_V5_DEVELOPMENT_SCORE_VALIDATION_CONFIRMATION_PANTHEON_SEALED",
        "decision": summary["classification"]["empirical_label"],
        "authorization": {"authorization_id": authorization["authorization_id"], "content_sha256": authorization["content_sha256"]},
        "release_chain": {"v5_contract": _authorization_binding(contract_receipt, reaudit)},
        "artifacts": {
            name: {"bytes": len(payload), "raw_sha256": bytes_sha256(payload), "content_sha256": v2._artifact_content(name, payload)}
            for name, payload in sorted(artifacts.items())
        },
        "roots": summary["roots"],
        "permutation_order_hashes": list(order_hashes),
        "permutation_order_root_sha256": v3._order_root(order_hashes),
        "counts": {"development_rows": summary["development_rows"], "permutations": summary["permutation"]["rows"]},
        "countermodels": summary["countermodels"],
        "access_counts": summary["access_counts"],
        "hard_seals": load_config()["future_executor"]["hard_seals"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    return receipt


def validate_final_payloads_v5(
    payloads: Mapping[str, bytes],
    authorization: Mapping[str, Any],
    contract_receipt: Mapping[str, Any],
    reaudit: Mapping[str, Any],
) -> dict[str, Any]:
    _require(set(payloads) == _ARTIFACT_NAMES | {"receipt.json"}, "v5 final package set")
    artifacts = {name: payloads[name] for name in _ARTIFACT_NAMES}
    v3._v2_validate_artifacts(artifacts)
    receipt = json.loads(payloads["receipt.json"])
    _require(receipt["content_sha256"] == _self_hash(receipt), "v5 final receipt self-hash")
    expected = build_final_receipt_v5(artifacts, authorization, contract_receipt, reaudit, receipt["permutation_order_hashes"])
    _require(receipt == expected, "v5 final receipt mismatch")
    rows = v2._reconstruct_scored_rows(v2._parse_jsonl(artifacts["artifacts/development-rows.jsonl"]))
    permutation_rows = v2._parse_jsonl(artifacts["artifacts/permutation-statistics.jsonl"])
    summary = json.loads(artifacts["artifacts/development-summary.json"])
    claimed = {
        "observed": float.fromhex(summary["permutation"]["observed_hex"]),
        "permutation_statistics": [float.fromhex(row["statistic_hex"]) for row in permutation_rows],
        "tail_count": summary["permutation"]["tail_count"],
        "p_value": float.fromhex(summary["permutation"]["p_value_hex"]),
    }
    v3._exact_validate_regenerated(rows, claimed, receipt["permutation_order_hashes"], permutations=_PERMUTATIONS)
    _require(receipt["access_counts"] == _expected_success_counts(), "v5 final access counts")
    return receipt


def run_development_once() -> str:
    """Future sole entry; presently sealed behind re-audit plus separate one-run authorization."""
    contract_receipt = check_receipt()
    authorization_payload, authorization, reaudit = _load_future_gates(contract_receipt)
    marker, marker_path = _consume_authorization(authorization, contract_receipt)
    config = load_config()
    counts = {key: 0 for key in _expected_success_counts()}
    counts["authorization_consumptions"] = 1
    decoded_ids: list[int] = []
    stage = "AUTHORIZED_AND_CONSUMED"
    generated_payloads: dict[str, bytes] | None = None
    payload_ids: dict[str, int] | None = None
    payload_hashes: dict[str, str] | None = None
    final_capability = object()
    unspent_capability: object | None = final_capability

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
        _require(name in config["sources"], "v5 source name")
        return canonical_file(config["sources"][name]["path"])

    def check_raw(name: str, size: int, digest: str) -> None:
        source = config["sources"][name]
        _require(size == source["bytes"] and digest == source["raw_sha256"], f"{name} source drift")

    def gzip_lines(name: str, open_key: str, pass_key: str) -> Iterator[bytes]:
        with source_path(name).open("rb") as raw:
            counts[open_key] += 1
            reader = HashingReader(raw)
            with gzip.GzipFile(fileobj=reader, mode="rb") as stream:
                yield from stream
            check_raw(name, reader.bytes_read, reader.digest.hexdigest())
            counts[pass_key] += 1

    def plain_lines(name: str, open_key: str) -> Iterator[bytes]:
        digest = hashlib.sha256()
        size = 0
        with source_path(name).open("rb") as handle:
            counts[open_key] += 1
            for line in handle:
                digest.update(line)
                size += len(line)
                yield line
        check_raw(name, size, digest.hexdigest())

    def read_cf4(ledger: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        nonlocal stage
        stage = "CF4_OWNED_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        observed = 0
        for index, line in enumerate(gzip_lines("CF4_TABLE4", "cf4_source_opens", "cf4_gzip_passes")):
            _require(index < len(ledger), "v5 CF4 overflow")
            entry = ledger[index]
            v2.validate_ledger_entry(entry)
            payload = v1._frame_payload(line, 157)
            _require(index == entry["source_index"] and offset == entry["framed_start"], "v5 CF4 order")
            _require(bytes_sha256(line) == entry["framed_raw_sha256"], "v5 CF4 frame hash")
            identifier, canonical = ids_v1.parse_i7_identifier(payload[:7])
            _require(identifier == entry["identifier"] and canonical == entry["canonical_identifier"], "v5 CF4 ID")
            counts["cf4_identifier_rows_reverified"] += 1
            if entry["role"] == "development":
                rows.append(v2.parse_cf4_development_record_v2(line, entry, source_index=index, framed_start=offset))
                decoded_ids.append(identifier)
                counts["cf4_development_scientific_rows_decoded"] += 1
            offset += len(line)
            observed += 1
        _require(observed == len(ledger) == 38053 and len(rows) == 22897, "v5 CF4 coverage")
        return rows

    def read_vast1() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST1_AUDITED_PARSER_STREAM"
        rows: list[dict[str, Any]] = []
        offset = 0
        for index, line in enumerate(plain_lines("VAST_TABLE1", "vast_table1_source_opens")):
            row = parse_vast1_record_v5(line, source_index=index, framed_start=offset)
            rows.append(row)
            counts["vast_table1_rows_decoded"] += 1
            offset += len(line)
        _require(len(rows) == 2347, "v5 VAST1 row count")
        _require([row["source_index"] for row in rows if row["void"] == 0] == [0, 1163], "v5 VAST1 zero-ID positions")
        _require(sum(row["edge"] == 2 for row in rows) == 46, "v5 VAST1 edge-2 count")
        return rows

    def read_vast2() -> list[dict[str, Any]]:
        nonlocal stage
        stage = "VAST2_FROZEN_STREAM"
        rows = [v1.parse_vast_table2_record(line) for line in gzip_lines("VAST_TABLE2", "vast_table2_source_opens", "vast_table2_gzip_passes")]
        _require(len(rows) == 80080, "v5 VAST2 row count")
        counts["vast_table2_rows_decoded"] = len(rows)
        return rows

    def read_mask() -> bytes:
        nonlocal stage
        stage = "MASK_OWNED_READ"
        with source_path("MASK_U8").open("rb") as handle:
            counts["mask_source_opens"] += 1
            payload = handle.read()
        check_raw("MASK_U8", len(payload), bytes_sha256(payload))
        v1.validate_mask(payload)
        return payload

    def write_failure(error: Exception) -> None:
        _validate_fixed_directory(FAILURE_DIRECTORY, REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v5-failures")
        FAILURE_DIRECTORY.mkdir(parents=True, exist_ok=True)
        failure: dict[str, Any] = {
            "schema": "invariant-open-gravity-void-correlation-development-failure-5.0",
            "status": "RETAINED_V5_RUN_FAILURE_NO_PARTIAL_SUCCESS",
            "authorization_id": authorization["authorization_id"],
            "stage": stage,
            "reason_code": type(error).__name__.upper(),
            "access_counts": dict(counts),
            "authorized_development_ids": sorted(set(decoded_ids)),
            "hard_seals_preserved": config["future_executor"]["hard_seals"],
            "content_sha256": "",
        }
        failure["content_sha256"] = _self_hash(failure)
        target = FAILURE_DIRECTORY / f"{authorization['authorization_id']}-{failure['content_sha256']}.json"
        _require(not target.exists(), "v5 failure collision")
        with target.open("xb") as handle:
            handle.write(_pretty(failure))
            handle.flush()
            os.fsync(handle.fileno())

    def final_write() -> str:
        nonlocal unspent_capability
        _require(unspent_capability is final_capability, "v5 final capability absent or replayed")
        unspent_capability = None
        _require(generated_payloads is not None and payload_ids is not None and payload_hashes is not None, "v5 payloads absent")
        _require(payload_ids == {name: id(payload) for name, payload in generated_payloads.items()}, "v5 payload identity drift")
        _require(payload_hashes == {name: bytes_sha256(payload) for name, payload in generated_payloads.items()}, "v5 payload hash drift")
        _require(check_receipt() == contract_receipt, "v5 contract drift at write")
        payload2, authorization2, reaudit2 = _load_future_gates(contract_receipt)
        _require(payload2 == authorization_payload and authorization2 == authorization and reaudit2 == reaudit, "v5 gates drift at write")
        _require(marker_path.read_bytes() == _pretty(marker) and marker == _consumption_marker(authorization, contract_receipt), "v5 marker drift")
        validate_code_pins()
        validate_release_chain(load_config())
        validate_final_payloads_v5(generated_payloads, authorization, contract_receipt, reaudit)
        _validate_fixed_directory(FINAL_DIRECTORY, REPO_ROOT / "runs/gravity/open-gravity-void-correlation-development-score-v5")
        _validate_fixed_directory(STAGING_ROOT, REPO_ROOT / "work/open-gravity-void-correlation-development-score-v5-staging")
        _require(not FINAL_DIRECTORY.exists(), "v5 final output already exists")
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="run-", dir=STAGING_ROOT)).resolve()
        try:
            for name, payload in sorted(generated_payloads.items()):
                target = staging.joinpath(*name.split("/"))
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("xb") as handle:
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
            validate_final_payloads_v5(v2._read_fixed_package(staging), authorization, contract_receipt, reaudit)
            FINAL_DIRECTORY.parent.mkdir(parents=True, exist_ok=True)
            if os.name == "nt":
                move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
                move.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32]
                move.restype = ctypes.c_int
                _require(bool(move(str(staging), str(FINAL_DIRECTORY), 0x00000008)), "v5 MoveFileExW failed")
            else:
                os.rename(staging, FINAL_DIRECTORY)
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
        geometry = prepare_vast_geometry_v5(vast1, vast2)
        rows = [v1.derive_development_row(row, mask, geometry["spheres_Mpc"]) for row in cf4]
        profile = v1.profile_grid_details(rows)
        countermodels = v1.score_countermodels(rows)
        permutation = v3.regenerate_permutations_from_rows(rows, _PERMUTATIONS)
        counts["development_scores"] += 1
        _require(counts == _expected_success_counts(), "v5 owned counts incomplete")
        artifacts = v2.assemble_development_artifacts_v2(rows, ledger, profile, permutation, countermodels, counts, [])
        final_receipt = build_final_receipt_v5(artifacts, authorization, contract_receipt, reaudit, permutation["order_hashes"])
        generated_payloads = {**artifacts, "receipt.json": _pretty(final_receipt)}
        payload_ids = {name: id(payload) for name, payload in generated_payloads.items()}
        payload_hashes = {name: bytes_sha256(payload) for name, payload in generated_payloads.items()}
        stage = "PRIVATE_FINAL_WRITE"
        return final_write()
    except Exception as error:
        write_failure(error)
        raise


_RUN_ORIGINAL = run_development_once
_RUN_CODE_SHA256 = bytes_sha256(marshal.dumps(run_development_once.__code__))


def validate_code_pins() -> None:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v5 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v5 test raw drift")
    _require(run_development_once is _RUN_ORIGINAL, "v5 runner identity drift")
    _require(bytes_sha256(marshal.dumps(run_development_once.__code__)) == _RUN_CODE_SHA256, "v5 runner code drift")


def _runner_structure() -> dict[str, bool]:
    source = inspect.getsource(run_development_once)
    tree = ast.parse(source)
    outer = tree.body[0]
    calls = [node.func.id for node in ast.walk(outer) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    nested = [node.name for node in outer.body if isinstance(node, ast.FunctionDef)]
    return {
        "no_arguments": not inspect.signature(run_development_once).parameters,
        "future_gates_first": calls.index("check_receipt") < calls.index("_load_future_gates") < calls.index("_consume_authorization"),
        "audited_vast1_parser_called": "parse_vast1_record_v5" in calls,
        "repaired_geometry_join_called": "prepare_vast_geometry_v5" in calls,
        "private_final_write": nested.count("final_write") == 1 and calls.count("final_write") == 1,
        "v1_vast1_parser_absent": "v1.parse_vast_table1_record" not in source and "v1.prepare_vast_geometry" not in source,
        "pcg64_regeneration_retained": "v3.regenerate_permutations_from_rows" in source and "_PERMUTATIONS" in source,
    }


def conformance_gates(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    fixture = _fixture_conformance(config)
    structure = _runner_structure()
    return [
        {"check_id": "V1_V4_PACKETS_BYTE_EXACT", "passed": True},
        {"check_id": "VAST1_CONTRACT_AND_AUDIT_BOUND", "passed": True},
        {"check_id": "RETAINED_V4_FAILURE_BOUND", "passed": True},
        {"check_id": "OFFICIAL_VOID_ZERO_FIXTURE_ACCEPTED", "passed": fixture["void_zero_accepted"]},
        {"check_id": "VAST1_FIXTURE_HASH_EXACT", "passed": fixture["framed_sha256"] == "e1aaeccae3e857121fd4b1b31895d21cf590e145f0421341e3c6ec7e6418a0a7"},
        {"check_id": "FULL_OWNED_RUNNER_STRUCTURE", "passed": all(structure.values())},
        {"check_id": "TWO_GATE_SEPARATION", "passed": "does not itself authorize" in config["future_gates"]["rule"]},
        {"check_id": "ZERO_CURRENT_AUTHORITY_AND_ACCESS", "passed": all(value == 0 for key, value in config["current_access_accounting"].items() if key != "allowed_vast1_fixture_rows_decoded")},
    ]


def build_receipt() -> dict[str, Any]:
    validate_code_pins()
    config = load_config()
    bindings = validate_release_chain(config)
    gates = conformance_gates(config)
    _require(all(gate["passed"] for gate in gates), "v5 conformance failure")
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-correlation-development-release-receipt-5.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "bindings": bindings,
        "development_packet_chain": config["development_packet_chain"],
        "retained_v4_failure": config["retained_v4_failure"],
        "vast1_parser_contract": config["vast1_parser_contract"],
        "vast1_integration": config["vast1_integration"],
        "future_executor": config["future_executor"],
        "future_gates": config["future_gates"],
        "authority": config["authority"],
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


def _write_receipt() -> str:
    payload = _pretty(build_receipt())
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    if OUTPUT_PATH.exists():
        _require(OUTPUT_PATH.is_file() and not OUTPUT_PATH.is_symlink() and OUTPUT_PATH.read_bytes() == payload, "existing v5 receipt differs")
        return "EXISTING_IDENTICAL"
    descriptor, name = tempfile.mkstemp(prefix="receipt.", suffix=".tmp", dir=OUTPUT_PATH.parent)
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
    return "CREATED"


def check_receipt() -> dict[str, Any]:
    observed = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    expected = build_receipt()
    _require(observed == expected and observed["content_sha256"] == _self_hash(observed), "v5 receipt drift")
    return observed


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(_write_receipt())
    elif args.command == "check":
        check_receipt()
        print("VALID_V5_SOURCE_FREE_EXECUTOR_NO_RUN_AUTHORITY")
    else:
        print(check_receipt()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
