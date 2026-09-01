"""Append-only, source-only parser contract for frozen Lane 9 VAST table 2."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from . import open_gravity_void_correlation_development_release_v1 as frozen_v1

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_vast2_source_parser_contract_v1.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_vast2_source_parser_contract_v1.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_vast2_source_parser_contract_v1.py"
OUTPUT_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-vast2-source-parser-contract-v1"
OUTPUT_PATH = OUTPUT_DIRECTORY / "receipt.json"
LEDGER_PATH = OUTPUT_DIRECTORY / "artifacts/vast2-row-dispositions.jsonl"
SUMMARY_PATH = OUTPUT_DIRECTORY / "artifacts/source-summary.json"

_CONFIG_RAW_SHA256 = "0a1bbce3a0c2cd60d846d0fdafc32d6985320b65c44befb51bb47f49a99128b5"
_CONFIG_CONTENT_SHA256 = "0c3d1222ee3fd221717380f2407a012c32773bc6e64557b8a99c3ec41f40d76c"
_MODULE_SEMANTIC_SHA256 = "d56c3c29e78f84722a4b2a4dff4973a46bcd02af2dfb856b0ba9b8bf14ef1786"
_TEST_RAW_SHA256 = "7cc5b5760c3618f41e0acf68d8dd7c23bf37ddcaf762b839291091cf79749ef8"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_V5_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_FLOAT_TOKEN = re.compile(rb" *-?[0-9]+\.[0-9]+ *\Z")
_UINT_TOKEN = re.compile(rb" *[0-9]+\Z")
_SEPARATORS = (10, 31, 57, 80, 100)
_FIELD_SPANS = {
    "x": (11, 31),
    "y": (32, 57),
    "z": (58, 80),
    "Rad": (81, 100),
    "void": (101, 105),
}
_DOMAINS: dict[str, tuple[float, float]] = {
    "x": (-328.4, -13.9),
    "y": (-295.2, 271.5),
    "z": (-15.1, 300.1),
    "Rad": (3.64, 22.6),
    "void": (0.0, 1183.0),
}


class Vast2SourceParserContractError(RuntimeError):
    """Fail-closed source, grammar, binding, or append-only artifact error."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Vast2SourceParserContractError(message)


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


def _semantic_module_sha256(path: Path, constants: set[str]) -> str:
    lines: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if any(stripped.startswith(f'{name} = "') for name in constants):
            continue
        lines.append(line)
    return bytes_sha256("\n".join(lines).encode("utf-8"))


def module_semantic_sha256(path: Path = MODULE_PATH) -> str:
    return _semantic_module_sha256(path, _SELF_CONSTANTS)


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
    _require(value["status"] == "SOURCE_CONTRACT_ONLY_AWAIT_DISTINCT_INDEPENDENT_AUDIT", "status drift")
    authority = value["authority"]
    _require(authority["scoring_authority"] is False, "scoring authority introduced")
    _require(authority["development_run_authority"] is False, "run authority introduced")
    _require(authority["reauthorization_authority"] is False, "reauthorization authority introduced")
    _require(authority["may_mint_or_consume_authorization"] is False, "authorization authority introduced")
    return value


def _load_self_hashed_json(section: Mapping[str, Any]) -> dict[str, Any]:
    path = canonical_file(str(section["path"]))
    _require(file_sha256(path) == section["raw_sha256"], f"bound raw drift: {section['path']}")
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(value.get("content_sha256") == section["content_sha256"], f"bound content drift: {section['path']}")
    _require(value["content_sha256"] == _self_hash(value), f"bound self-hash drift: {section['path']}")
    if "status" in section:
        _require(value.get("status") == section["status"], f"bound status drift: {section['path']}")
    return value


def validate_frozen_v5_bindings(config: Mapping[str, Any]) -> dict[str, Any]:
    subject = config["frozen_v5_parser_subject"]
    config_path = canonical_file(subject["config"]["path"])
    _require(file_sha256(config_path) == subject["config"]["raw_sha256"], "v5 config raw drift")
    config_value = json.loads(config_path.read_text(encoding="utf-8"))
    _require(content_sha256(config_value) == subject["config"]["content_sha256"], "v5 config content drift")
    module_path = canonical_file(subject["module"]["path"])
    _require(file_sha256(module_path) == subject["module"]["raw_sha256"], "v5 module raw drift")
    _require(
        _semantic_module_sha256(module_path, _V5_SELF_CONSTANTS) == subject["module"]["semantic_sha256"],
        "v5 module semantic drift",
    )
    module_source = module_path.read_text(encoding="utf-8")
    _require("def read_vast2()" in module_source, "v5 VAST2 reader absent")
    _require("v1.parse_vast_table2_record(line)" in module_source, "v5 frozen VAST2 dispatch drift")
    _require(file_sha256(canonical_file(subject["test"]["path"])) == subject["test"]["raw_sha256"], "v5 test drift")
    receipt = _load_self_hashed_json(subject["receipt"])
    audit = _load_self_hashed_json(subject["independent_audit"])
    attestation = audit["source_access_attestation"]
    _require(attestation["vast2_source_opened"] == 0 and attestation["vast2_verified"] is False, "v5 audit scope drift")
    _require(audit["scientific_run_authority"] is False, "v5 audit run authority drift")
    return {
        "v5_config_raw_sha256": subject["config"]["raw_sha256"],
        "v5_module_raw_sha256": subject["module"]["raw_sha256"],
        "v5_test_raw_sha256": subject["test"]["raw_sha256"],
        "v5_receipt_raw_sha256": subject["receipt"]["raw_sha256"],
        "v5_receipt_content_sha256": receipt["content_sha256"],
        "v5_audit_raw_sha256": subject["independent_audit"]["raw_sha256"],
        "v5_audit_content_sha256": audit["content_sha256"],
        "v5_audit_vast2_opened": attestation["vast2_source_opened"],
        "v5_audit_vast2_verified": attestation["vast2_verified"],
    }


def _parse_float(token: bytes, name: str) -> float:
    _require(_FLOAT_TOKEN.fullmatch(token) is not None, f"invalid {name} decimal token")
    value = float(token.decode("ascii"))
    _require(math.isfinite(value), f"nonfinite {name}")
    low, high = _DOMAINS[name]
    _require(low <= value <= high, f"{name} outside documented domain")
    return value


def _parse_void(token: bytes) -> int:
    _require(_UINT_TOKEN.fullmatch(token) is not None, "invalid void integer token")
    value = int(token.decode("ascii"))
    low, high = _DOMAINS["void"]
    _require(int(low) <= value <= int(high), "void outside documented domain")
    return value


def parse_vast2_record(framed: bytes, *, source_index: int, framed_start: int) -> dict[str, Any]:
    """Parse one official frozen VAST table-2 frame under the exact source grammar."""
    _require(isinstance(framed, bytes), "record must be bytes")
    _require(isinstance(source_index, int) and isinstance(framed_start, int), "invalid frame coordinates")
    _require(source_index >= 0 and framed_start >= 0, "negative frame coordinate")
    _require(framed.endswith(b"\n") and not framed.endswith(b"\r\n"), "record must end in exactly one LF")
    payload = framed[:-1]
    _require(b"\r" not in payload and b"\n" not in payload, "embedded line ending")
    _require(len(payload) == 105, "payload length must be exactly 105")
    _require(all(byte < 128 for byte in payload), "non-ASCII payload")
    _require(all(payload[index : index + 1] == b" " for index in _SEPARATORS), "separator byte drift")
    cosmo_token = payload[:10]
    _require(cosmo_token in (b"Planck2018", b"WMAP5     "), "invalid exact Cosmo token")
    row: dict[str, Any] = {"Cosmo": cosmo_token.decode("ascii").rstrip(" ")}
    for name in ("x", "y", "z", "Rad"):
        start, end = _FIELD_SPANS[name]
        row[name] = _parse_float(payload[start:end], name)
    row["void"] = _parse_void(payload[101:105])
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


def semantic_sphere_key(row: Mapping[str, Any]) -> tuple[str, int, float, float, float, float]:
    return (
        str(row["Cosmo"]),
        int(row["void"]),
        float(row["x"]),
        float(row["y"]),
        float(row["z"]),
        float(row["Rad"]),
    )


def validate_no_semantic_sphere_duplicates(rows: Sequence[Mapping[str, Any]]) -> int:
    keys = [semantic_sphere_key(row) for row in rows]
    _require(len(keys) == len(set(keys)), "duplicate VAST2 semantic sphere key")
    return len(keys)


def _stable_float(value: float) -> str:
    return "0x0.0p+0" if value == 0.0 else value.hex()


def _semantic_key_sha256(row: Mapping[str, Any]) -> str:
    key = semantic_sphere_key(row)
    return content_sha256([key[0], key[1], *(_stable_float(value) for value in key[2:])])


def _disposition_entry(row: Mapping[str, Any]) -> dict[str, Any]:
    tags = ["VALID_ZERO_BASED_VOID_IDENTIFIER"] if row["void"] == 0 else []
    entry: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast2-row-disposition-1.0",
        "source_index": row["source_index"],
        "line_number": row["line_number"],
        "framed_start": row["framed_start"],
        "framed_end_exclusive": row["framed_end_exclusive"],
        "framed_bytes": row["framed_bytes"],
        "payload_bytes": row["payload_bytes"],
        "framed_raw_sha256": row["framed_raw_sha256"],
        "payload_raw_sha256": row["payload_raw_sha256"],
        "semantic_sphere_key_sha256": _semantic_key_sha256(row),
        "Cosmo": row["Cosmo"],
        "void": row["void"],
        "parser_disposition": "ACCEPT_OFFICIAL_FROZEN_VAST2_ROW",
        "grouping_disposition": "SPHERE_MEMBER_OF_DOCUMENTED_VOID_UNION",
        "special_tags": tags,
        "content_sha256": "",
    }
    entry["content_sha256"] = _self_hash(entry)
    return entry


def _ledger_root(entries: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for entry in entries:
        value = str(entry["content_sha256"])
        _require(_HEX64.fullmatch(value) is not None, "invalid disposition hash")
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
            "x",
            "y",
            "z",
            "Rad",
            "void",
        )
    }


def _lexical_observation(
    payload: bytes,
) -> tuple[dict[str, int], dict[str, int], dict[str, int], int]:
    decimals: dict[str, int] = {}
    leading: dict[str, int] = {}
    trailing: dict[str, int] = {}
    signed_zeros = 0
    for name in ("x", "y", "z", "Rad"):
        start, end = _FIELD_SPANS[name]
        token = payload[start:end]
        leading[name] = len(token) - len(token.lstrip(b" "))
        trailing[name] = len(token) - len(token.rstrip(b" "))
        stripped = token.strip()
        decimals[name] = len(stripped) - stripped.index(b".") - 1
        if stripped.startswith(b"-") and float(stripped.decode("ascii")) == 0.0:
            signed_zeros += 1
    token = payload[101:105]
    leading["void"] = len(token) - len(token.lstrip(b" "))
    trailing["void"] = len(token) - len(token.rstrip(b" "))
    return decimals, leading, trailing, signed_zeros


def audit_vast2_source(
    config: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    readme_path = canonical_file(config["documentation"]["path"])
    readme = readme_path.read_bytes()
    _require(len(readme) == config["documentation"]["bytes"], "ReadMe byte-count drift")
    _require(bytes_sha256(readme) == config["documentation"]["raw_sha256"], "ReadMe raw drift")
    readme_lines = readme.splitlines(keepends=True)
    start = int(config["documentation"]["table2_block_line_start"]) - 1
    end = int(config["documentation"]["table2_block_line_end"])
    block = b"".join(readme_lines[start:end])
    _require(len(block) == config["documentation"]["table2_block_bytes"], "ReadMe table2 block size drift")
    _require(bytes_sha256(block) == config["documentation"]["table2_block_raw_sha256"], "ReadMe table2 block drift")

    source_path = canonical_file(config["source"]["path"])
    compressed = source_path.read_bytes()
    _require(len(compressed) == config["source"]["bytes"], "VAST2 compressed byte-count drift")
    _require(bytes_sha256(compressed) == config["source"]["raw_sha256"], "VAST2 compressed raw drift")
    try:
        source = gzip.decompress(compressed)
    except (EOFError, OSError) as error:
        raise Vast2SourceParserContractError("VAST2 gzip decode failed") from error
    _require(len(source) == config["source"]["uncompressed_bytes"], "VAST2 uncompressed byte-count drift")
    _require(bytes_sha256(source) == config["source"]["uncompressed_sha256"], "VAST2 uncompressed raw drift")
    frames = source.splitlines(keepends=True)
    _require(len(frames) == config["source"]["records"], "VAST2 record-count drift")
    _require(source.endswith(b"\n") and all(frame.endswith(b"\n") for frame in frames), "VAST2 framing drift")

    entries: list[dict[str, Any]] = []
    payloads: set[bytes] = set()
    semantic_keys: set[tuple[str, int, float, float, float, float]] = set()
    group_counts: Counter[tuple[str, int]] = Counter()
    cosmo_counts: Counter[str] = Counter()
    identifiers: dict[str, set[int]] = defaultdict(set)
    zero_counts: Counter[str] = Counter()
    payload_lengths: Counter[str] = Counter()
    decimal_counts: dict[str, Counter[str]] = {name: Counter() for name in ("x", "y", "z", "Rad")}
    leading_counts: dict[str, Counter[str]] = {name: Counter() for name in ("x", "y", "z", "Rad", "void")}
    trailing_counts: dict[str, Counter[str]] = {name: Counter() for name in ("x", "y", "z", "Rad", "void")}
    observed_ranges = {name: [math.inf, -math.inf] for name in ("x", "y", "z", "Rad")}
    duplicate_bytes = 0
    duplicate_semantic = 0
    missing_void_tokens = 0
    separator_violations = 0
    signed_zero_tokens = 0
    offset = 0
    first_row: dict[str, Any] | None = None
    first_by_cosmo: dict[str, dict[str, Any]] = {}
    range_witnesses: dict[str, dict[str, dict[str, Any]]] = {
        name: {} for name in ("x", "y", "z", "Rad")
    }
    for index, frame in enumerate(frames):
        payload = frame[:-1]
        payload_lengths[str(len(payload))] += 1
        missing_void_tokens += payload[101:105] == b"    " if len(payload) >= 105 else 0
        separator_violations += (
            sum(payload[position : position + 1] != b" " for position in _SEPARATORS) if len(payload) >= 105 else 1
        )
        row = parse_vast2_record(frame, source_index=index, framed_start=offset)
        if first_row is None:
            first_row = row
        first_by_cosmo.setdefault(str(row["Cosmo"]), row)
        entries.append(_disposition_entry(row))
        cosmo = str(row["Cosmo"])
        void = int(row["void"])
        cosmo_counts[cosmo] += 1
        identifiers[cosmo].add(void)
        group_counts[(cosmo, void)] += 1
        zero_counts[cosmo] += void == 0
        if payload in payloads:
            duplicate_bytes += 1
        payloads.add(payload)
        semantic_key = semantic_sphere_key(row)
        if semantic_key in semantic_keys:
            duplicate_semantic += 1
        semantic_keys.add(semantic_key)
        decimals, leading, trailing, signed_zeros = _lexical_observation(payload)
        signed_zero_tokens += signed_zeros
        for name, count in decimals.items():
            decimal_counts[name][str(count)] += 1
        for name, count in leading.items():
            leading_counts[name][str(count)] += 1
        for name, count in trailing.items():
            trailing_counts[name][str(count)] += 1
        for name in ("x", "y", "z", "Rad"):
            value = float(row[name])
            if value < observed_ranges[name][0]:
                observed_ranges[name][0] = value
                range_witnesses[name]["min"] = row
            if value > observed_ranges[name][1]:
                observed_ranges[name][1] = value
                range_witnesses[name]["max"] = row
        offset += len(frame)
    _require(first_row is not None and offset == len(source), "VAST2 framed coverage drift")

    expected = config["expected_source_disposition"]
    _require(dict(cosmo_counts) == expected["cosmology_counts"], "cosmology counts drift")
    _require(dict(sorted(payload_lengths.items())) == expected["payload_length_counts"], "payload length drift")
    identifier_contract: dict[str, Any] = {}
    for cosmo, contract in expected["identifier_contract"].items():
        values = identifiers[cosmo]
        _require(min(values) == contract["min"] and max(values) == contract["max"], f"{cosmo} ID range drift")
        _require(len(values) == contract["unique"], f"{cosmo} ID uniqueness drift")
        _require(values == set(range(contract["max"] + 1)), f"{cosmo} IDs not zero-based contiguous")
        identifier_contract[cosmo] = dict(contract)
    memberships = list(group_counts.values())
    _require(len(group_counts) == expected["unique_group_keys"], "group-key count drift")
    _require(min(memberships) == expected["group_membership_min"], "group minimum drift")
    _require(max(memberships) == expected["group_membership_max"], "group maximum drift")
    _require(sum(value == 1 for value in memberships) == expected["single_sphere_groups"], "single group drift")
    _require(sum(memberships) - len(memberships) == expected["repeated_group_membership_rows"], "group repeats drift")
    _require(sum(zero_counts.values()) == expected["zero_identifier_sphere_rows"], "zero-ID sphere count drift")
    _require(duplicate_bytes == expected["byte_duplicate_rows"], "byte duplicate drift")
    _require(duplicate_semantic == expected["semantic_duplicate_sphere_keys"], "semantic duplicate drift")
    _require(missing_void_tokens == expected["missing_void_tokens"], "missing void drift")
    _require(separator_violations == expected["separator_violations"], "separator drift")
    _require(signed_zero_tokens == expected["signed_zero_tokens"], "signed-zero drift")
    observed_decimal_digits = {
        name: [min(map(int, counts)), max(map(int, counts))] for name, counts in decimal_counts.items()
    }
    _require(observed_decimal_digits == expected["observed_decimal_digits"], "decimal profile drift")

    summary: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast2-source-summary-1.0",
        "status": "PASS_ALL_OFFICIAL_FROZEN_ROWS_ACCEPTED_V5_PARSER_WRONG_SOURCE_NOT_CORRUPT",
        "compressed_raw_sha256": config["source"]["raw_sha256"],
        "compressed_bytes": len(compressed),
        "uncompressed_raw_sha256": config["source"]["uncompressed_sha256"],
        "uncompressed_bytes": len(source),
        "documentation_raw_sha256": config["documentation"]["raw_sha256"],
        "documentation_table2_block_raw_sha256": config["documentation"]["table2_block_raw_sha256"],
        "accepted_rows": len(entries),
        "rejected_rows": 0,
        "payload_length_counts": dict(sorted(payload_lengths.items())),
        "cosmology_counts": dict(cosmo_counts),
        "identifier_contract": identifier_contract,
        "zero_identifier_sphere_rows": sum(zero_counts.values()),
        "zero_identifier_sphere_rows_by_cosmology": dict(zero_counts),
        "unique_group_keys": len(group_counts),
        "group_membership_min": min(memberships),
        "group_membership_max": max(memberships),
        "single_sphere_groups": sum(value == 1 for value in memberships),
        "repeated_group_membership_rows": sum(memberships) - len(memberships),
        "byte_duplicate_rows": duplicate_bytes,
        "semantic_duplicate_sphere_keys": duplicate_semantic,
        "missing_void_tokens": missing_void_tokens,
        "separator_violations": separator_violations,
        "signed_zero_tokens": signed_zero_tokens,
        "lexical_profiles": {
            "decimal_digits_counts": {name: dict(sorted(counts.items(), key=lambda pair: int(pair[0]))) for name, counts in decimal_counts.items()},
            "leading_space_counts": {name: dict(sorted(counts.items(), key=lambda pair: int(pair[0]))) for name, counts in leading_counts.items()},
            "trailing_space_counts": {name: dict(sorted(counts.items(), key=lambda pair: int(pair[0]))) for name, counts in trailing_counts.items()},
            "observed_decimal_digits": observed_decimal_digits,
        },
        "observed_numeric_ranges": observed_ranges,
        "row_disposition_root_sha256": _ledger_root(entries),
        "duplicate_interpretation": {
            "full_semantic_sphere_key_duplicates": 0,
            "byte_duplicates": 0,
            "repeated_Cosmo_void_memberships": expected["repeated_group_membership_rows"],
            "repeated_Cosmo_void_is_documented_union_membership_not_duplicate": True,
        },
        "witnesses": {
            "first_frozen_parser_failure": _witness(first_row),
            "first_by_cosmology": {name: _witness(row) for name, row in first_by_cosmo.items()},
            "numeric_extrema": {
                name: {kind: _witness(row) for kind, row in witnesses.items()}
                for name, witnesses in range_witnesses.items()
            },
        },
        "diagnosis": config["diagnosis"],
        "build_access_accounting": config["build_access_accounting"],
        "claim_boundary": {
            "source_corrupt": False,
            "zero_is_valid_zero_based_identifier": True,
            "missing_void_tokens_observed": 0,
            "frozen_v5_parser_wrong": True,
            "executor_runs": 0,
            "scores_computed": 0,
            "reauthorizations_granted": 0,
        },
        "content_sha256": "",
    }
    summary["content_sha256"] = _self_hash(summary)
    return entries, summary, first_row


def _synthetic_frame(*, cosmo: bytes = b"Planck2018", void: int = 0) -> bytes:
    _require(len(cosmo) == 10 and 0 <= void <= 9999, "invalid synthetic fixture request")
    fields = (
        cosmo,
        f"{-100.0:20.15f}".encode("ascii"),
        f"{0.0:25.20f}".encode("ascii"),
        f"{0.0:22.18f}".encode("ascii"),
        f"{10.0:19.16f}".encode("ascii"),
        f"{void:4d}".encode("ascii"),
    )
    frame = b" ".join(fields) + b"\n"
    _require(len(frame) == 106, "synthetic fixture width")
    return frame


def _frozen_parser_conformance(first_row: Mapping[str, Any]) -> list[dict[str, Any]]:
    first_frame = (str(first_row["raw_ascii"]) + "\n").encode("ascii")
    try:
        frozen_v1.parse_vast_table2_record(first_frame)
        zero_rejected = False
        error_text = ""
    except frozen_v1.DevelopmentReleaseV1Error as error:
        zero_rejected = True
        error_text = str(error)
    valid_one = _synthetic_frame(void=1)
    bad_separator = valid_one[:10] + b"X" + valid_one[11:]
    arbitrary_cosmo = _synthetic_frame(cosmo=b"Other     ", void=1)
    out_of_domain = _synthetic_frame(void=9999)
    frozen_accepts_bad_separator = frozen_v1.parse_vast_table2_record(bad_separator)["void"] == 1
    frozen_accepts_arbitrary_cosmo = frozen_v1.parse_vast_table2_record(arbitrary_cosmo)["Cosmo"] == "Other"
    frozen_accepts_crlf = frozen_v1.parse_vast_table2_record(valid_one[:-1] + b"\r\n")["void"] == 1
    frozen_accepts_out_of_domain = frozen_v1.parse_vast_table2_record(out_of_domain)["void"] == 9999
    return [
        {
            "check_id": "FROZEN_V5_DISPATCH_REJECTS_FIRST_DOCUMENTED_ZERO_ID",
            "passed": zero_rejected and error_text == "invalid VAST table2 radius or void",
        },
        {"check_id": "SUCCESSOR_ACCEPTS_ALL_80080_ROWS", "passed": True},
        {"check_id": "FROZEN_PARSER_ACCEPTS_NONSPACE_SEPARATOR", "passed": frozen_accepts_bad_separator},
        {"check_id": "FROZEN_PARSER_ACCEPTS_ARBITRARY_COSMO", "passed": frozen_accepts_arbitrary_cosmo},
        {"check_id": "FROZEN_PARSER_ACCEPTS_NONFROZEN_CRLF", "passed": frozen_accepts_crlf},
        {"check_id": "FROZEN_PARSER_ACCEPTS_VOID_ABOVE_DOCUMENTED_DOMAIN", "passed": frozen_accepts_out_of_domain},
        {"check_id": "SOURCE_HAS_ZERO_BYTE_DUPLICATES", "passed": True},
        {"check_id": "SOURCE_HAS_ZERO_SEMANTIC_SPHERE_DUPLICATES", "passed": True},
        {"check_id": "ZERO_EXECUTOR_OR_REAUTHORIZATION_AUTHORITY", "passed": True},
    ]


def _ledger_bytes(entries: Sequence[Mapping[str, Any]]) -> bytes:
    return b"".join(_canonical(entry) + b"\n" for entry in entries)


def _parse_ledger(payload: bytes) -> list[dict[str, Any]]:
    _require(payload.endswith(b"\n"), "ledger terminal LF missing")
    values = [json.loads(line) for line in payload.splitlines()]
    _require(len(values) == 80080, "ledger row count drift")
    for index, value in enumerate(values):
        _require(value["source_index"] == index and value["line_number"] == index + 1, "ledger order drift")
        _require(value["content_sha256"] == _self_hash(value), "ledger self-hash drift")
    return values


def _write_once(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError as error:
        raise Vast2SourceParserContractError(f"append-only target already exists: {path.relative_to(REPO_ROOT)}") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_package() -> dict[str, Any]:
    _require(not any(path.exists() for path in (LEDGER_PATH, SUMMARY_PATH, OUTPUT_PATH)), "append-only package already exists")
    config = load_config()
    bindings = validate_frozen_v5_bindings(config)
    entries, summary, first_row = audit_vast2_source(config)
    gates = _frozen_parser_conformance(first_row)
    _require(all(gate["passed"] for gate in gates), "conformance gate failed")
    ledger_payload = _ledger_bytes(entries)
    summary_payload = _pretty(summary)
    artifacts = {
        "artifacts/vast2-row-dispositions.jsonl": {
            "bytes": len(ledger_payload),
            "raw_sha256": bytes_sha256(ledger_payload),
            "content_sha256": content_sha256(entries),
        },
        "artifacts/source-summary.json": {
            "bytes": len(summary_payload),
            "raw_sha256": bytes_sha256(summary_payload),
            "content_sha256": summary["content_sha256"],
        },
    }
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast2-source-parser-contract-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "source": config["source"],
        "documentation": config["documentation"],
        "field_grammar": config["field_grammar"],
        "source_disposition": {
            key: summary[key]
            for key in (
                "accepted_rows",
                "rejected_rows",
                "payload_length_counts",
                "cosmology_counts",
                "identifier_contract",
                "zero_identifier_sphere_rows",
                "zero_identifier_sphere_rows_by_cosmology",
                "unique_group_keys",
                "group_membership_min",
                "group_membership_max",
                "single_sphere_groups",
                "repeated_group_membership_rows",
                "byte_duplicate_rows",
                "semantic_duplicate_sphere_keys",
                "missing_void_tokens",
                "separator_violations",
                "signed_zero_tokens",
                "lexical_profiles",
                "observed_numeric_ranges",
                "duplicate_interpretation",
            )
        },
        "diagnosis": config["diagnosis"],
        "frozen_v5_bindings": bindings,
        "conformance_gates": gates,
        "row_disposition_root_sha256": summary["row_disposition_root_sha256"],
        "artifacts": artifacts,
        "mutation_freeze": {
            "config_raw_sha256": _CONFIG_RAW_SHA256,
            "config_content_sha256": _CONFIG_CONTENT_SHA256,
            "module_raw_sha256": file_sha256(MODULE_PATH),
            "module_semantic_sha256": _MODULE_SEMANTIC_SHA256,
            "test_raw_sha256": _TEST_RAW_SHA256,
        },
        "build_access_accounting": config["build_access_accounting"],
        "authority": config["authority"],
        "claim_boundary": {
            "successor_built": True,
            "successor_independently_audited": False,
            "self_review_claimed_independent": False,
            "executor_run_or_authorized": False,
            "Lane9_reauthorized": False,
        },
        "next_gate": config["next_gate"],
        "content_sha256": "",
    }
    receipt["content_sha256"] = _self_hash(receipt)
    receipt_payload = _pretty(receipt)
    _write_once(LEDGER_PATH, ledger_payload)
    _write_once(SUMMARY_PATH, summary_payload)
    _write_once(OUTPUT_PATH, receipt_payload)
    return receipt


def check_package() -> dict[str, Any]:
    config = load_config()
    _require(file_sha256(MODULE_PATH) == json.loads(OUTPUT_PATH.read_text())["mutation_freeze"]["module_raw_sha256"], "module raw drift")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "test raw drift")
    receipt = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
    _require(receipt["content_sha256"] == _self_hash(receipt), "receipt self-hash drift")
    _require(receipt["status"] == config["success_status"], "receipt status drift")
    _require(receipt["decision"] == config["decision"], "receipt decision drift")
    ledger_payload = LEDGER_PATH.read_bytes()
    entries = _parse_ledger(ledger_payload)
    summary_payload = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_payload)
    _require(summary["content_sha256"] == _self_hash(summary), "summary self-hash drift")
    for relative, payload, content in (
        ("artifacts/vast2-row-dispositions.jsonl", ledger_payload, content_sha256(entries)),
        ("artifacts/source-summary.json", summary_payload, summary["content_sha256"]),
    ):
        binding = receipt["artifacts"][relative]
        _require(len(payload) == binding["bytes"], f"artifact size drift: {relative}")
        _require(bytes_sha256(payload) == binding["raw_sha256"], f"artifact raw drift: {relative}")
        _require(content == binding["content_sha256"], f"artifact content drift: {relative}")
    _require(_ledger_root(entries) == receipt["row_disposition_root_sha256"], "ledger root drift")
    _require(summary["row_disposition_root_sha256"] == receipt["row_disposition_root_sha256"], "summary root drift")
    _require(all(gate["passed"] for gate in receipt["conformance_gates"]), "receipt gate drift")
    _require(receipt["claim_boundary"]["successor_independently_audited"] is False, "false audit claim")
    return receipt


def status() -> dict[str, Any]:
    config = load_config()
    return {
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": config["decision"],
        "output_exists": OUTPUT_PATH.is_file(),
        "scientific_run_authority": config["authority"]["development_run_authority"],
        "reauthorization_authority": config["authority"]["reauthorization_authority"],
        "next_gate": config["next_gate"],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check", "status"))
    args = parser.parse_args(argv)
    value = build_package() if args.command == "build" else check_package() if args.command == "check" else status()
    print(json.dumps(value, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
