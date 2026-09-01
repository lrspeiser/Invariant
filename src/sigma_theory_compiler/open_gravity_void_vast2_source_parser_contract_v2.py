"""Binary-write successor for the append-only Lane 9 VAST2 parser contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import open_gravity_void_vast2_source_parser_contract_v1 as v1

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs/open_gravity_void_vast2_source_parser_contract_v2.json"
MODULE_PATH = REPO_ROOT / "src/sigma_theory_compiler/open_gravity_void_vast2_source_parser_contract_v2.py"
TEST_PATH = REPO_ROOT / "tests/test_open_gravity_void_vast2_source_parser_contract_v2.py"
OUTPUT_DIRECTORY = REPO_ROOT / "runs/gravity/open-gravity-void-vast2-source-parser-contract-v2"
OUTPUT_PATH = OUTPUT_DIRECTORY / "receipt.json"
LEDGER_PATH = OUTPUT_DIRECTORY / "artifacts/vast2-row-dispositions.jsonl"
SUMMARY_PATH = OUTPUT_DIRECTORY / "artifacts/source-summary.json"

_CONFIG_RAW_SHA256 = "58fcbf43da65d7fe1a6d370ec44f7722b6a341c09d960feae5c86e49373d5464"
_CONFIG_CONTENT_SHA256 = "875c418f692e20cd3b8521c897e97e16f9803be4bb0d616d762e7cb454ea5123"
_MODULE_SEMANTIC_SHA256 = "d35fe341ec1ebdec0886d907d9d10e21cad5c827cf47e2374d9eaf9bcb82e38d"
_TEST_RAW_SHA256 = "e1c45067832c41a244fc8fe5be39b6d00cabd1a24c144061171d2e34a5c3f524"
_SELF_CONSTANTS = {
    "_CONFIG_RAW_SHA256",
    "_CONFIG_CONTENT_SHA256",
    "_MODULE_SEMANTIC_SHA256",
    "_TEST_RAW_SHA256",
}

Vast2SourceParserContractV2Error = v1.Vast2SourceParserContractError
parse_vast2_record = v1.parse_vast2_record
semantic_sphere_key = v1.semantic_sphere_key
validate_no_semantic_sphere_duplicates = v1.validate_no_semantic_sphere_duplicates


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise Vast2SourceParserContractV2Error(message)


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
    return v1.bytes_sha256("\n".join(lines).encode("utf-8"))


def load_config() -> dict[str, Any]:
    value = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    _require(file_sha256(CONFIG_PATH) == _CONFIG_RAW_SHA256, "v2 config raw drift")
    _require(v1.content_sha256(value) == _CONFIG_CONTENT_SHA256, "v2 config content drift")
    _require(
        value["status"] == "SOURCE_CONTRACT_BINARY_WRITE_SUCCESSOR_AWAIT_DISTINCT_INDEPENDENT_AUDIT",
        "v2 status drift",
    )
    authority = value["authority"]
    _require(authority["scoring_authority"] is False, "v2 scoring authority introduced")
    _require(authority["development_run_authority"] is False, "v2 run authority introduced")
    _require(authority["reauthorization_authority"] is False, "v2 reauthorization authority introduced")
    _require(authority["may_mint_or_consume_authorization"] is False, "v2 authorization authority introduced")
    return value


def validate_frozen_v1_and_failure(config: Mapping[str, Any]) -> dict[str, Any]:
    binding = config["parser_contract_v1"]
    config_path = v1.canonical_file(binding["config"]["path"])
    _require(file_sha256(config_path) == binding["config"]["raw_sha256"], "v1 config raw drift")
    v1_config = json.loads(config_path.read_text(encoding="utf-8"))
    _require(v1.content_sha256(v1_config) == binding["config"]["content_sha256"], "v1 config content drift")
    module_path = v1.canonical_file(binding["module"]["path"])
    _require(file_sha256(module_path) == binding["module"]["raw_sha256"], "v1 module raw drift")
    _require(v1.module_semantic_sha256(module_path) == binding["module"]["semantic_sha256"], "v1 module semantic drift")
    _require(file_sha256(v1.canonical_file(binding["test"]["path"])) == binding["test"]["raw_sha256"], "v1 test drift")
    receipt_path = v1.canonical_file(binding["frozen_failed_build_receipt"]["path"])
    _require(file_sha256(receipt_path) == binding["frozen_failed_build_receipt"]["raw_sha256"], "v1 receipt raw drift")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    _require(receipt["content_sha256"] == binding["frozen_failed_build_receipt"]["content_sha256"], "v1 receipt content drift")
    _require(receipt["content_sha256"] == v1._self_hash(receipt), "v1 receipt self-hash drift")
    failure = config["superseded_v1_build_failure"]
    ledger_path = v1.LEDGER_PATH
    summary_path = v1.SUMMARY_PATH
    _require(ledger_path.stat().st_size == failure["actual_ledger_bytes"], "v1 failed ledger size drift")
    _require(file_sha256(ledger_path) == failure["actual_ledger_raw_sha256"], "v1 failed ledger raw drift")
    _require(summary_path.stat().st_size == failure["actual_summary_bytes"], "v1 failed summary size drift")
    _require(file_sha256(summary_path) == failure["actual_summary_raw_sha256"], "v1 failed summary raw drift")
    _require(
        receipt["artifacts"]["artifacts/vast2-row-dispositions.jsonl"]["bytes"]
        == failure["declared_ledger_bytes"],
        "v1 declared ledger size drift",
    )
    _require(
        receipt["artifacts"]["artifacts/source-summary.json"]["bytes"] == failure["declared_summary_bytes"],
        "v1 declared summary size drift",
    )
    _require(
        failure["actual_ledger_bytes"] - failure["declared_ledger_bytes"] == failure["bytes_added_to_ledger"],
        "v1 CRLF expansion diagnosis drift",
    )
    return {
        "v1_config_raw_sha256": binding["config"]["raw_sha256"],
        "v1_module_raw_sha256": binding["module"]["raw_sha256"],
        "v1_test_raw_sha256": binding["test"]["raw_sha256"],
        "v1_failed_receipt_raw_sha256": binding["frozen_failed_build_receipt"]["raw_sha256"],
        "v1_failed_receipt_content_sha256": receipt["content_sha256"],
        "v1_failure_classification": failure["classification"],
        "v1_preserved": True,
    }


def _write_once_binary(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(path, flags, 0o644)
    except FileExistsError as error:
        raise Vast2SourceParserContractV2Error(
            f"append-only v2 target already exists: {path.relative_to(REPO_ROOT)}"
        ) from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            _require(written > 0, "zero-byte artifact write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require(path.stat().st_size == len(payload), "binary artifact size mismatch after write")
    _require(file_sha256(path) == v1.bytes_sha256(payload), "binary artifact hash mismatch after write")


def build_package() -> dict[str, Any]:
    _require(not any(path.exists() for path in (LEDGER_PATH, SUMMARY_PATH, OUTPUT_PATH)), "append-only v2 package already exists")
    config = load_config()
    v1_preservation = validate_frozen_v1_and_failure(config)
    base_config = v1.load_config()
    frozen_v5_bindings = v1.validate_frozen_v5_bindings(base_config)
    entries, summary, first_row = v1.audit_vast2_source(base_config)
    gates = v1._frozen_parser_conformance(first_row)
    gates.extend(
        [
            {"check_id": "V1_FAILED_BUILD_PRESERVED", "passed": v1_preservation["v1_preserved"]},
            {
                "check_id": "V2_BINARY_APPEND_ONLY_WRITER",
                "passed": os.name != "nt" or getattr(os, "O_BINARY", 0) != 0,
            },
            {"check_id": "V2_ZERO_INDEPENDENT_AUDIT_CLAIM", "passed": True},
        ]
    )
    _require(all(gate["passed"] for gate in gates), "v2 conformance gate failed")
    ledger_payload = v1._ledger_bytes(entries)
    summary_payload = v1._pretty(summary)
    artifacts = {
        "artifacts/vast2-row-dispositions.jsonl": {
            "bytes": len(ledger_payload),
            "raw_sha256": v1.bytes_sha256(ledger_payload),
            "content_sha256": v1.content_sha256(entries),
        },
        "artifacts/source-summary.json": {
            "bytes": len(summary_payload),
            "raw_sha256": v1.bytes_sha256(summary_payload),
            "content_sha256": summary["content_sha256"],
        },
    }
    receipt: dict[str, Any] = {
        "schema": "invariant-open-gravity-void-vast2-source-parser-contract-v2-receipt-1.0",
        "package_id": config["package_id"],
        "status": config["success_status"],
        "decision": config["decision"],
        "source": base_config["source"],
        "documentation": base_config["documentation"],
        "field_grammar": base_config["field_grammar"],
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
        "diagnosis": base_config["diagnosis"],
        "frozen_v5_bindings": frozen_v5_bindings,
        "v1_preservation_and_failure_binding": v1_preservation,
        "v1_build_failure": config["superseded_v1_build_failure"],
        "repair_scope": config["repair_scope"],
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
        "build_access_accounting": base_config["build_access_accounting"],
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
    receipt["content_sha256"] = v1._self_hash(receipt)
    receipt_payload = v1._pretty(receipt)
    _write_once_binary(LEDGER_PATH, ledger_payload)
    _write_once_binary(SUMMARY_PATH, summary_payload)
    _write_once_binary(OUTPUT_PATH, receipt_payload)
    return receipt


def check_package() -> dict[str, Any]:
    config = load_config()
    receipt_payload = OUTPUT_PATH.read_bytes()
    receipt = json.loads(receipt_payload)
    _require(receipt["content_sha256"] == v1._self_hash(receipt), "v2 receipt self-hash drift")
    _require(receipt["status"] == config["success_status"], "v2 receipt status drift")
    _require(receipt["decision"] == config["decision"], "v2 receipt decision drift")
    _require(file_sha256(MODULE_PATH) == receipt["mutation_freeze"]["module_raw_sha256"], "v2 module raw drift")
    _require(module_semantic_sha256() == _MODULE_SEMANTIC_SHA256, "v2 module semantic drift")
    _require(file_sha256(TEST_PATH) == _TEST_RAW_SHA256, "v2 test raw drift")
    ledger_payload = LEDGER_PATH.read_bytes()
    entries = v1._parse_ledger(ledger_payload)
    summary_payload = SUMMARY_PATH.read_bytes()
    summary = json.loads(summary_payload)
    _require(summary["content_sha256"] == v1._self_hash(summary), "v2 summary self-hash drift")
    for relative, payload, content in (
        ("artifacts/vast2-row-dispositions.jsonl", ledger_payload, v1.content_sha256(entries)),
        ("artifacts/source-summary.json", summary_payload, summary["content_sha256"]),
    ):
        binding = receipt["artifacts"][relative]
        _require(len(payload) == binding["bytes"], f"v2 artifact size drift: {relative}")
        _require(v1.bytes_sha256(payload) == binding["raw_sha256"], f"v2 artifact raw drift: {relative}")
        _require(content == binding["content_sha256"], f"v2 artifact content drift: {relative}")
    _require(v1._ledger_root(entries) == receipt["row_disposition_root_sha256"], "v2 ledger root drift")
    _require(summary["row_disposition_root_sha256"] == receipt["row_disposition_root_sha256"], "v2 summary root drift")
    _require(all(gate["passed"] for gate in receipt["conformance_gates"]), "v2 gate drift")
    _require(receipt["claim_boundary"]["successor_independently_audited"] is False, "false v2 audit claim")
    return receipt


def status() -> dict[str, Any]:
    config = load_config()
    return {
        "package_id": config["package_id"],
        "status": config["status"],
        "decision": config["decision"],
        "output_exists": OUTPUT_PATH.is_file(),
        "scientific_run_authority": False,
        "reauthorization_authority": False,
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
