"""Append-only source/parser contract for the retained Lane 9 VAST1 failure."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import open_gravity_void_correlation_development_release_v1 as failed_v1

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_vast1_source_parser_contract_v1.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_vast1_source_parser_contract_v1.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_vast1_source_parser_contract_v1.py"
OUTPUT_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-vast1-source-parser-contract-v1"
OUTPUT_PATH = OUTPUT_DIRECTORY / "receipt.json"
LEDGER_PATH = OUTPUT_DIRECTORY / "artifacts/vast1-row-dispositions.jsonl"
SUMMARY_PATH = OUTPUT_DIRECTORY / "artifacts/source-summary.json"

_CONFIG_RAW_SHA256 = "a46cb305d6927bc154b95a3721040d6b598b488a63e554fcf9ed41187a942fad"
_CONFIG_CONTENT_SHA256 = "8723f467da298473f4c7a8e0b18071bb66ff291552caa5a5db436ec6b57591c2"
_MODULE_SEMANTIC_SHA256 = "d2b6c5c70eabe1f74a173d17733b9bb7445bde7c45693c952b28b4dfdeefb0d4"
_TEST_RAW_SHA256 = "5ba4542064bb0a76f0576607b294aaf4caac8cc47c8834262f9d26ae8a20491f"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT_TOKEN = re.compile(rb" *[+-]?(?:[0-9]+\.[0-9]+|\.[0-9]+) *\Z")
_UINT_TOKEN = re.compile(rb" *[0-9]+ *\Z")
_REFF_TOKEN = re.compile(rb"[0-9]{2}\.[0-9]{12,15}\Z")
_SEPARATORS = (10, 31, 54, 76, 95, 100, 102, 122, 141, 162)
_FIELD_SPANS = {
    "x": (11, 31),
    "y": (32, 54),
    "z": (55, 76),
    "Rad": (77, 95),
    "void": (96, 100),
    "edge": (101, 102),
    "s": (103, 122),
    "RAdeg": (123, 141),
    "DEdeg": (142, 162),
}


class Vast1SourceParserContractError(RuntimeError):
    """Fail-closed source, grammar, binding, or artifact error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Vast1SourceParserContractError(message)


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
    _require(all(part not in {"", ".", ".."} for part in relative.split("/")), "unsafe path")
    pure = PurePosixPath(relative)
    _require(not pure.is_absolute() and all(part not in {"", ".", ".."} for part in pure.parts), "unsafe path")
    root = REPO_ROOT.resolve(strict=True)
    _require(root == REPO_ROOT and not REPO_ROOT.is_symlink(), "repository root redirected")
    candidate = REPO_ROOT.joinpath(*pure.parts)
    _require(candidate.resolve(strict=True) == candidate and not candidate.is_symlink(), "path redirected")
    _require(root in candidate.parents, "path escapes repository")
    return candidate


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "config raw drift")
    _require(content_sha256(value) == _CONFIG_CONTENT_SHA256, "config content drift")
    _require(value["status"] == "SOURCE_CONTRACT_ONLY_AWAIT_INDEPENDENT_AUDIT", "config status drift")
    _require(value["authority"]["scoring_authority"] is False, "scoring authority introduced")
    _require(value["authority"]["development_run_authority"] is False, "development authority introduced")
    _require(value["authority"]["may_mint_or_consume_authorization"] is False, "authorization authority introduced")
    return value


def _load_bound_json(section: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"bound raw drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == section["content_sha256"], f"bound content drift: {section['path']}")
    _require(value["content_sha256"] == _self_hash(value), f"bound self-hash drift: {section['path']}")
    if "status" in section:
        _require(value.get("status") == section["status"], f"bound status drift: {section['path']}")
    return value


def validate_frozen_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    failed = config["retained_v4_failure"]
    failure = _load_bound_json(failed)
    _require(failure["stage"] == failed["stage"] and failure["reason_code"] == failed["reason_code"], "failure reason drift")
    _require(failure["authorization_id"] == failed["authorization_id"], "failure authorization drift")
    identifiers = failure["authorized_development_ids"]
    _require(
        len(identifiers) == len(set(identifiers)) == failed["authorized_development_ids"],
        "failure development-ID coverage drift",
    )
    _require(failure["access_counts"]["development_scores"] == 0, "failed run unexpectedly scored")
    _require(failure["access_counts"]["cf4_validation_scientific_rows_decoded"] == 0, "validation seal drift")
    _require(failure["access_counts"]["cf4_confirmation_scientific_rows_decoded"] == 0, "confirmation seal drift")
    _require(failure["access_counts"]["pantheon_source_opens"] == 0, "Pantheon seal drift")

    chain_values = {name: _load_bound_json(section) for name, section in config["v4_chain"].items()}
    _require(chain_values["consumption_marker"]["uses_consumed"] == 1, "v4 authorization consumption drift")
    _require(
        chain_values["consumption_marker"]["authorization_id"] == failed["authorization_id"],
        "marker/failure authorization mismatch",
    )
    parser = config["frozen_failed_parser"]
    parser_config_path = canonical_file(parser["config"]["path"])
    _require(file_sha256(parser_config_path) == parser["config"]["raw_sha256"], "failed parser config raw drift")
    parser_config = json.loads(parser_config_path.read_text(encoding="utf-8"))
    _require(content_sha256(parser_config) == parser["config"]["content_sha256"], "failed parser config content drift")
    parser_module = canonical_file(parser["module"]["path"])
    _require(file_sha256(parser_module) == parser["module"]["raw_sha256"], "failed parser module raw drift")
    _require(failed_v1.module_semantic_sha256(parser_module) == parser["module"]["semantic_sha256"], "failed parser semantic drift")
    _require(file_sha256(canonical_file(parser["test"]["path"])) == parser["test"]["raw_sha256"], "failed parser test drift")
    return {
        "failure_raw_sha256": failed["raw_sha256"],
        "failure_content_sha256": failed["content_sha256"],
        "authorization_id": failed["authorization_id"],
        "consumption_marker_content_sha256": config["v4_chain"]["consumption_marker"]["content_sha256"],
    }


def _parse_float(token: bytes, name: str, domain: Sequence[float]) -> float:
    _require(_FLOAT_TOKEN.fullmatch(token) is not None, f"invalid {name} decimal token")
    value = float(token.decode("ascii"))
    _require(math.isfinite(value), f"nonfinite {name}")
    _require(float(domain[0]) <= value <= float(domain[1]), f"{name} outside documented domain")
    return value


def _parse_uint(token: bytes, name: str, domain: Sequence[int]) -> int:
    _require(_UINT_TOKEN.fullmatch(token) is not None, f"invalid {name} integer token")
    value = int(token.decode("ascii"))
    _require(int(domain[0]) <= value <= int(domain[1]), f"{name} outside documented domain")
    return value


def parse_vast1_record(
    framed: bytes,
    *,
    source_index: int,
    framed_start: int,
) -> dict[str, Any]:
    """Parse one official VAST table-1 line under the repaired exact grammar."""
    _require(isinstance(framed, bytes) and isinstance(source_index, int) and isinstance(framed_start, int), "invalid frame inputs")
    _require(source_index >= 0 and framed_start >= 0, "negative frame coordinate")
    _require(framed.endswith(b"\n") and not framed.endswith(b"\r\n"), "record must end in exactly one LF")
    payload = framed[:-1]
    _require(b"\r" not in payload and b"\n" not in payload, "embedded line ending")
    _require(len(payload) in (178, 179, 180, 181), "payload length outside official observed range")
    _require(all(byte < 128 for byte in payload), "non-ASCII payload")
    _require(all(payload[index : index + 1] == b" " for index in _SEPARATORS), "separator byte drift")

    cosmo_token = payload[:10]
    _require(cosmo_token in (b"Planck2018", b"WMAP5     "), "invalid exact Cosmo token")
    cosmo = cosmo_token.decode("ascii").rstrip(" ")
    grammar = load_config()["field_grammar"]["fields"]
    domains = {str(field["name"]): field["domain"] for field in grammar if "domain" in field}
    row: dict[str, Any] = {"Cosmo": cosmo}
    for name in ("x", "y", "z", "Rad"):
        start, end = _FIELD_SPANS[name]
        row[name] = _parse_float(payload[start:end], name, domains[name])
    row["void"] = _parse_uint(payload[96:100], "void", domains["void"])
    row["edge"] = _parse_uint(payload[101:102], "edge", domains["edge"])
    for name in ("s", "RAdeg", "DEdeg"):
        start, end = _FIELD_SPANS[name]
        row[name] = _parse_float(payload[start:end], name, domains[name])
    reff_token = payload[163:]
    _require(_REFF_TOKEN.fullmatch(reff_token) is not None, "invalid variable-width Reff token")
    row["Reff"] = _parse_float(reff_token, "Reff", domains["Reff"])
    row.update(
        {
            "source_index": source_index,
            "line_number": source_index + 1,
            "framed_start": framed_start,
            "framed_end_exclusive": framed_start + len(framed),
            "framed_bytes": len(framed),
            "payload_bytes": len(payload),
            "framed_raw_sha256": bytes_sha256(framed),
            "payload_raw_sha256": bytes_sha256(payload),
            "raw_ascii": payload.decode("ascii"),
        }
    )
    return row


def _disposition_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    tags: list[str] = []
    if row["void"] == 0:
        tags.append("VALID_ZERO_BASED_VOID_IDENTIFIER")
    if row["edge"] == 2:
        tags.append("VALID_DOCUMENTED_EDGE_CLASS_2")
    if row["payload_bytes"] < 181:
        tags.append("VALID_VARIABLE_WIDTH_FINAL_REFF")
    entry: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast1-row-disposition-1.0",
        "source_index": row["source_index"],
        "line_number": row["line_number"],
        "framed_start": row["framed_start"],
        "framed_end_exclusive": row["framed_end_exclusive"],
        "framed_bytes": row["framed_bytes"],
        "payload_bytes": row["payload_bytes"],
        "framed_raw_sha256": row["framed_raw_sha256"],
        "payload_raw_sha256": row["payload_raw_sha256"],
        "Cosmo": row["Cosmo"],
        "void": row["void"],
        "edge": row["edge"],
        "parser_disposition": "ACCEPT_OFFICIAL_VAST1_ROW",
        "primary_geometry_disposition": (
            "PRIMARY_INTERIOR_CANDIDATE" if row["edge"] == 0 else "EXCLUDE_DOCUMENTED_EDGE_FLAG"
        ),
        "special_tags": tags,
        "content_sha256": "",
    }
    entry["content_sha256"] = _self_hash(entry)
    return entry


def _ledger_root(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        value = str(entry["content_sha256"])
        _require(_HEX64.fullmatch(value) is not None, "invalid row disposition hash")
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _witness(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: row[key]
        for key in (
            "source_index",
            "line_number",
            "framed_start",
            "framed_end_exclusive",
            "framed_bytes",
            "payload_bytes",
            "framed_raw_sha256",
            "payload_raw_sha256",
            "raw_ascii",
            "Cosmo",
            "Rad",
            "void",
            "edge",
            "Reff",
        )
    }


def audit_vast1_source(config: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    readme_path = canonical_file(config["documentation"]["path"])
    readme = readme_path.read_bytes()
    _require(len(readme) == config["documentation"]["bytes"], "ReadMe byte-count drift")
    _require(bytes_sha256(readme) == config["documentation"]["raw_sha256"], "ReadMe raw drift")
    readme_lines = readme.splitlines(keepends=True)
    start = int(config["documentation"]["table1_block_line_start"]) - 1
    end = int(config["documentation"]["table1_block_line_end"])
    block = b"".join(readme_lines[start:end])
    _require(len(block) == config["documentation"]["table1_block_bytes"], "ReadMe table1 block size drift")
    _require(bytes_sha256(block) == config["documentation"]["table1_block_raw_sha256"], "ReadMe table1 block drift")

    source_path = canonical_file(config["source"]["path"])
    source = source_path.read_bytes()
    _require(len(source) == config["source"]["bytes"], "VAST1 byte-count drift")
    _require(bytes_sha256(source) == config["source"]["raw_sha256"], "VAST1 raw drift")
    frames = source.splitlines(keepends=True)
    _require(len(frames) == config["source"]["records"], "VAST1 record-count drift")
    _require(source.endswith(b"\n") and all(frame.endswith(b"\n") for frame in frames), "VAST1 line framing drift")
    rows: list[dict[str, Any]] = []
    entries: list[dict[str, Any]] = []
    offset = 0
    for index, frame in enumerate(frames):
        row = parse_vast1_record(frame, source_index=index, framed_start=offset)
        rows.append(row)
        entries.append(_disposition_entry(row))
        offset += len(frame)
    _require(offset == len(source), "VAST1 framed coverage drift")

    expected = config["expected_source_disposition"]
    cosmo_counts = Counter(str(row["Cosmo"]) for row in rows)
    edge_counts = Counter(str(row["edge"]) for row in rows)
    length_counts = Counter(str(row["payload_bytes"]) for row in rows)
    zeros = [int(row["source_index"]) for row in rows if row["void"] == 0]
    _require(dict(cosmo_counts) == expected["cosmology_counts"], "cosmology counts drift")
    _require(dict(sorted(edge_counts.items())) == expected["edge_counts"], "edge counts drift")
    _require(dict(sorted(length_counts.items())) == expected["payload_length_counts"], "payload-length counts drift")
    _require(zeros == expected["zero_identifier_source_indices"], "zero identifier positions drift")
    for cosmo, contract in expected["identifier_contract"].items():
        identifiers = [int(row["void"]) for row in rows if row["Cosmo"] == cosmo]
        _require(len(identifiers) == contract["unique"] == len(set(identifiers)), f"{cosmo} ID uniqueness drift")
        _require(min(identifiers) == contract["min"] and max(identifiers) == contract["max"], f"{cosmo} ID range drift")
        _require(sorted(identifiers) == list(range(contract["max"] + 1)), f"{cosmo} IDs not zero-based contiguous")

    first_edge2 = next(index for index, row in enumerate(rows) if row["edge"] == 2)
    first_short = next(index for index, row in enumerate(rows) if row["payload_bytes"] == 178)
    summary: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast1-source-summary-1.0",
        "status": "PASS_ALL_OFFICIAL_ROWS_ACCEPTED_FROZEN_V4_PARSER_WRONG_SOURCE_NOT_CORRUPT",
        "source_raw_sha256": config["source"]["raw_sha256"],
        "documentation_raw_sha256": config["documentation"]["raw_sha256"],
        "documentation_table1_block_raw_sha256": config["documentation"]["table1_block_raw_sha256"],
        "accepted_rows": len(rows),
        "rejected_rows": 0,
        "cosmology_counts": dict(cosmo_counts),
        "edge_counts": dict(sorted(edge_counts.items())),
        "payload_length_counts": dict(sorted(length_counts.items())),
        "zero_identifier_source_indices": zeros,
        "identifier_contract": expected["identifier_contract"],
        "row_disposition_root_sha256": _ledger_root(entries),
        "diagnosis": config["diagnosis"],
        "witnesses": {
            "frozen_failure_row": _witness(rows[0]),
            "neighbor_next": _witness(rows[1]),
            "neighbor_next_short_record": _witness(rows[2]),
            "second_cosmology_zero_identifier": _witness(rows[1163]),
            "first_edge_class_2": _witness(rows[first_edge2]),
            "first_178_byte_payload": _witness(rows[first_short]),
        },
        "access_accounting": config["access_accounting"],
        "claim_boundary": {
            "source_corrupt": False,
            "documented_sentinel_or_missing_value": False,
            "zero_is_valid_zero_based_identifier": True,
            "frozen_parser_wrong": True,
            "scores_computed": 0,
        },
        "content_sha256": "",
    }
    summary["content_sha256"] = _self_hash(summary)
    return entries, summary, rows


def _old_parser_conformance(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    gates: list[dict[str, Any]] = []
    try:
        failed_v1.parse_vast_table1_record((str(rows[0]["raw_ascii"]) + "\n").encode("ascii"))
        zero_rejected = False
    except failed_v1.DevelopmentReleaseV1Error as error:
        zero_rejected = str(error) == "invalid VAST table1 radius or void"
    gates.append({"check_id": "FROZEN_V1_REJECTS_DOCUMENTED_ZERO_ID", "passed": zero_rejected})
    try:
        failed_v1.parse_vast_table1_record((str(rows[2]["raw_ascii"]) + "\n").encode("ascii"))
        short_rejected = False
    except failed_v1.DevelopmentReleaseV1Error as error:
        short_rejected = str(error) == "record length mismatch"
    gates.append({"check_id": "FROZEN_V1_REJECTS_OFFICIAL_SHORT_FINAL_REFF", "passed": short_rejected})
    gates.extend(
        [
            {"check_id": "SUCCESSOR_ACCEPTS_ALL_2347_ROWS", "passed": len(rows) == 2347},
            {"check_id": "ZERO_IS_NOT_SENTINEL", "passed": rows[0]["void"] == 0 and rows[1163]["void"] == 0},
            {"check_id": "EDGE_CLASS_2_DOCUMENTED_AND_ACCEPTED", "passed": sum(row["edge"] == 2 for row in rows) == 46},
            {"check_id": "ZERO_SCORING_AUTHORITY", "passed": load_config()["authority"]["scoring_authority"] is False},
        ]
    )
    return gates


def _ledger_bytes(entries: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(entry) + b"\n" for entry in entries)


def _parse_ledger(payload: bytes) -> list[dict[str, Any]]:
    _require(payload.endswith(b"\n"), "ledger terminal LF missing")
    values = [json.loads(line) for line in payload.splitlines()]
    _require(len(values) == 2347, "ledger row count")
    for index, value in enumerate(values):
        _require(value["source_index"] == index and value["line_number"] == index + 1, "ledger order drift")
        _require(value["content_sha256"] == _self_hash(value), "ledger entry self-hash drift")
        _require(value["parser_disposition"] == "ACCEPT_OFFICIAL_VAST1_ROW", "ledger disposition drift")
    return values


def build_artifacts() -> tuple[dict[str, bytes], dict[str, Any]]:
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw drift")
    config = load_config()
    bindings = validate_frozen_bindings(config)
    entries, summary, rows = audit_vast1_source(config)
    gates = _old_parser_conformance(rows)
    _require(all(gate["passed"] for gate in gates), "source/parser conformance failure")
    artifacts = {
        "artifacts/vast1-row-dispositions.jsonl": _ledger_bytes(entries),
        "artifacts/source-summary.json": _pretty(summary),
    }
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast1-source-parser-contract-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "source": config["source"],
        "documentation": config["documentation"],
        "retained_v4_failure": config["retained_v4_failure"],
        "bindings": bindings,
        "field_grammar": config["field_grammar"],
        "source_disposition": config["expected_source_disposition"],
        "diagnosis": config["diagnosis"],
        "authority": config["authority"],
        "conformance_gates": gates,
        "artifacts": {
            name: {
                "bytes": len(payload),
                "raw_sha256": bytes_sha256(payload),
                "content_sha256": (
                    content_sha256(_parse_ledger(payload))
                    if name.endswith(".jsonl")
                    else json.loads(payload)["content_sha256"]
                ),
            }
            for name, payload in sorted(artifacts.items())
        },
        "row_disposition_root_sha256": summary["row_disposition_root_sha256"],
        "access_accounting": config["access_accounting"],
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
    return artifacts, receipt


def validate_package_payloads(payloads: Mapping[str, bytes]) -> dict[str, Any]:
    expected_names = {"artifacts/vast1-row-dispositions.jsonl", "artifacts/source-summary.json", "receipt.json"}
    _require(set(payloads) == expected_names, "package file set drift")
    entries = _parse_ledger(payloads["artifacts/vast1-row-dispositions.jsonl"])
    summary = json.loads(payloads["artifacts/source-summary.json"])
    _require(summary["content_sha256"] == _self_hash(summary), "summary self-hash drift")
    _require(summary["row_disposition_root_sha256"] == _ledger_root(entries), "summary ledger root drift")
    receipt = json.loads(payloads["receipt.json"])
    _require(receipt["content_sha256"] == _self_hash(receipt), "receipt self-hash drift")
    for name in expected_names - {"receipt.json"}:
        _require(receipt["artifacts"][name]["raw_sha256"] == bytes_sha256(payloads[name]), f"artifact raw drift: {name}")
    _require(receipt["row_disposition_root_sha256"] == _ledger_root(entries), "receipt ledger root drift")
    _require(receipt["access_accounting"]["scores_computed"] == 0, "score count introduced")
    return receipt


def _read_fixed_package() -> dict[str, bytes]:
    _require(OUTPUT_DIRECTORY.is_dir() and not OUTPUT_DIRECTORY.is_symlink(), "output directory absent or redirected")
    expected = {LEDGER_PATH, SUMMARY_PATH, OUTPUT_PATH}
    observed = {path for path in OUTPUT_DIRECTORY.rglob("*") if path.is_file()}
    _require(observed == expected and all(not path.is_symlink() for path in expected), "output file set drift")
    return {
        "artifacts/vast1-row-dispositions.jsonl": LEDGER_PATH.read_bytes(),
        "artifacts/source-summary.json": SUMMARY_PATH.read_bytes(),
        "receipt.json": OUTPUT_PATH.read_bytes(),
    }


def _write_no_clobber(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        _require(path.is_file() and not path.is_symlink() and path.read_bytes() == payload, f"existing output differs: {path.name}")
        return
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


def write_package() -> str:
    artifacts, receipt = build_artifacts()
    payloads = {**artifacts, "receipt.json": _pretty(receipt)}
    existed = OUTPUT_PATH.exists()
    for name, payload in sorted(payloads.items(), key=lambda item: item[0] == "receipt.json"):
        path = OUTPUT_DIRECTORY.joinpath(*name.split("/"))
        _write_no_clobber(path, payload)
    validate_package_payloads(_read_fixed_package())
    return "EXISTING_IDENTICAL" if existed else "CREATED"


def check_package() -> dict[str, Any]:
    artifacts, receipt = build_artifacts()
    expected = {**artifacts, "receipt.json": _pretty(receipt)}
    observed = _read_fixed_package()
    _require(observed == expected, "frozen source/parser package drift")
    return validate_package_payloads(observed)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    if args.command == "build":
        print(write_package())
    elif args.command == "check":
        check_package()
        print("VALID_VAST1_SOURCE_PARSER_CONTRACT_ZERO_SCORING_AUTHORITY")
    else:
        print(check_package()["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
