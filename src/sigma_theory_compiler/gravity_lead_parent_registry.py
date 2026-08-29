"""Hash-bound, metadata-only registry for the five retained gravity lead programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_lead_parent_registry_v1.json")
IMPLEMENTATION_PATH = Path("src/sigma_theory_compiler/gravity_lead_parent_registry.py")
OUTPUT_PATH = Path("runs/gravity/lead-programs/gravity-lead-parent-registry-v1.json")
CONFIG_SCHEMA = "invariant-gravity-lead-parent-registry-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-lead-parent-registry-receipt-1.0"
LEAD_IDS = (
    "nonlocal_boundary_response",
    "baryonic_transition_variable",
    "dynamical_age_spectral_clock",
    "massive_field_orbital_resonance",
    "emergent_gravity_transition",
)
EVIDENCE_KINDS = {
    "candidate_interface",
    "implementation",
    "lead_summary",
    "machine_result",
    "result_summary",
    "scientific_config",
    "source_receipt",
}
FORBIDDEN_COMMAND_TOKENS = {
    "aggregate",
    "evaluate",
    "extract",
    "fetch-responses",
    "open-exploration-response",
    "prepare",
    "run",
}
_SHA256 = re.compile(r"[0-9a-f]{64}")


class GravityLeadParentRegistryError(RuntimeError):
    """Raised when the registry contract or any bound evidence changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def _content_hashed(value: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["content_sha256"] = hashlib.sha256(_canonical_bytes(result)).hexdigest()
    return result


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityLeadParentRegistryError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityLeadParentRegistryError(f"{label} keys changed")


def _resolve_registered_path(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise GravityLeadParentRegistryError("evidence path must use a relative POSIX path")
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityLeadParentRegistryError("registered evidence escaped repository root") from error
    return path


def _validate_safe_command(command: str) -> None:
    if not command.startswith("python -m "):
        raise GravityLeadParentRegistryError("safe command must start with 'python -m '")
    if any(character in command for character in ";&|><\n\r"):
        raise GravityLeadParentRegistryError("safe command contains shell control syntax")
    tokens = set(command.casefold().split())
    if tokens & FORBIDDEN_COMMAND_TOKENS:
        raise GravityLeadParentRegistryError("safe command requests a mutating or production action")
    if not ({"check", "validate", "replay", "pytest"} & tokens):
        raise GravityLeadParentRegistryError("safe command lacks a check/replay verb")


def validate_config(config: Mapping[str, Any], root: Path) -> None:
    root = root.resolve()
    _strict(
        config,
        {
            "schema_version",
            "status",
            "registry_id",
            "purpose",
            "safety_contract",
            "lead_programs",
            "output_path",
        },
        "registry config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only_parent_registry"
        or config["registry_id"] != "gravity-lead-parent-registry-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityLeadParentRegistryError("registry identity changed")

    safety = config["safety_contract"]
    _strict(
        safety,
        {
            "reads_only_registered_metadata_evidence",
            "raw_payload_extensions_forbidden",
            "sealed_target_rows_opened",
            "network_calls_allowed",
            "gpu_production_allowed",
            "paid_model_calls_allowed",
            "safe_commands_are_descriptive_not_executed",
        },
        "safety contract",
    )
    expected_extensions = [".csv", ".fits", ".npz", ".sqlite", ".tsv", ".zip"]
    if (
        safety["reads_only_registered_metadata_evidence"] is not True
        or safety["raw_payload_extensions_forbidden"] != expected_extensions
        or safety["sealed_target_rows_opened"] != 0
        or safety["network_calls_allowed"] is not False
        or safety["gpu_production_allowed"] is not False
        or safety["paid_model_calls_allowed"] is not False
        or safety["safe_commands_are_descriptive_not_executed"] is not True
    ):
        raise GravityLeadParentRegistryError("registry safety boundary changed")

    leads = config["lead_programs"]
    if not isinstance(leads, list) or tuple(row.get("lead_id") for row in leads) != LEAD_IDS:
        raise GravityLeadParentRegistryError("five-lead inventory or ranking changed")
    if [row.get("rank") for row in leads] != [1, 2, 3, 4, 5]:
        raise GravityLeadParentRegistryError("lead ranking changed")

    seen_paths: set[str] = set()
    forbidden_extensions = set(expected_extensions)
    for lead in leads:
        lead_id = str(lead["lead_id"])
        _strict(
            lead,
            {
                "lead_id",
                "rank",
                "title",
                "empirical_role",
                "output_interface",
                "data_target_boundary",
                "claim_status",
                "bounded_local_rerun",
                "safe_commands",
                "evidence",
                "known_limitations",
            },
            f"lead {lead_id}",
        )
        interface = lead["output_interface"]
        _strict(interface, {"input_type", "output_type", "interface_type"}, f"{lead_id} interface")
        boundary = lead["data_target_boundary"]
        _strict(
            boundary,
            {"source_scope", "target_scope", "target_rows_opened_by_registry"},
            f"{lead_id} boundary",
        )
        if boundary["target_rows_opened_by_registry"] != 0:
            raise GravityLeadParentRegistryError(f"{lead_id} authorizes target access")
        rerun = lead["bounded_local_rerun"]
        _strict(rerun, {"possible", "scope", "reason"}, f"{lead_id} rerun")
        if not isinstance(rerun["possible"], bool):
            raise GravityLeadParentRegistryError(f"{lead_id} rerun flag is not boolean")
        if not isinstance(lead["safe_commands"], list) or not lead["safe_commands"]:
            raise GravityLeadParentRegistryError(f"{lead_id} lacks a safe check command")
        for command in lead["safe_commands"]:
            _validate_safe_command(str(command))
        if not isinstance(lead["known_limitations"], list) or not lead["known_limitations"]:
            raise GravityLeadParentRegistryError(f"{lead_id} lacks limitations")

        evidence = lead["evidence"]
        if not isinstance(evidence, list) or not evidence:
            raise GravityLeadParentRegistryError(f"{lead_id} lacks evidence")
        kinds = {str(row.get("kind")) for row in evidence}
        required = {
            "implementation",
            "lead_summary",
            "scientific_config",
            "source_receipt",
            "machine_result",
        }
        if not required <= kinds:
            raise GravityLeadParentRegistryError(f"{lead_id} evidence roles are incomplete")
        for item in evidence:
            _strict(item, {"kind", "path", "sha256"}, f"{lead_id} evidence")
            kind = str(item["kind"])
            relative = str(item["path"])
            digest = str(item["sha256"])
            if kind not in EVIDENCE_KINDS or _SHA256.fullmatch(digest) is None:
                raise GravityLeadParentRegistryError(f"{lead_id} evidence metadata is invalid")
            if relative in seen_paths:
                raise GravityLeadParentRegistryError(f"duplicate registered evidence: {relative}")
            seen_paths.add(relative)
            path = _resolve_registered_path(root, relative)
            if path.suffix.casefold() in forbidden_extensions or "/raw/" in f"/{relative.casefold()}/":
                raise GravityLeadParentRegistryError(f"raw payload registered as metadata: {relative}")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = _read_json(root / CONFIG_PATH)
    validate_config(config, root)
    return config


def verify_evidence(root: Path, item: Mapping[str, Any]) -> dict[str, Any]:
    root = root.resolve()
    path = _resolve_registered_path(root, str(item["path"]))
    if not path.is_file():
        raise GravityLeadParentRegistryError(f"registered evidence is missing: {item['path']}")
    observed = _file_sha256(path)
    expected = str(item["sha256"])
    if observed != expected:
        raise GravityLeadParentRegistryError(f"registered evidence changed: {item['path']}")
    return {"kind": str(item["kind"]), "path": str(item["path"]), "sha256": observed}


def build_receipt(root: Path, lead_id: str | None = None) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    if lead_id is not None and lead_id not in LEAD_IDS:
        raise GravityLeadParentRegistryError(f"unknown lead: {lead_id}")
    selected = [
        row for row in config["lead_programs"] if lead_id is None or row["lead_id"] == lead_id
    ]
    lead_receipts = []
    all_evidence = []
    for lead in selected:
        checked = [verify_evidence(root, item) for item in lead["evidence"]]
        all_evidence.extend(checked)
        counts = {
            kind: sum(item["kind"] == kind for item in checked)
            for kind in sorted(EVIDENCE_KINDS)
            if any(item["kind"] == kind for item in checked)
        }
        lead_receipts.append(
            {
                "lead_id": lead["lead_id"],
                "rank": lead["rank"],
                "empirical_role": lead["empirical_role"],
                "output_interface": lead["output_interface"],
                "data_target_boundary": lead["data_target_boundary"],
                "claim_status": lead["claim_status"],
                "bounded_local_rerun": lead["bounded_local_rerun"],
                "safe_commands": lead["safe_commands"],
                "known_limitations": lead["known_limitations"],
                "evidence_counts": counts,
                "evidence_set_sha256": hashlib.sha256(_canonical_bytes(checked)).hexdigest(),
                "registry_status": "REGISTERED_EVIDENCE_INTACT",
            }
        )
    return _content_hashed(
        {
            "schema_version": RECEIPT_SCHEMA,
            "registry_id": config["registry_id"],
            "registry_config_sha256": _file_sha256(root / CONFIG_PATH),
            "registry_implementation_sha256": _file_sha256(root / IMPLEMENTATION_PATH),
            "scope": "all_five_leads" if lead_id is None else lead_id,
            "decision": (
                "PASS_ALL_FIVE_PARENTS_REGISTERED_EVIDENCE_INTACT"
                if lead_id is None
                else "PASS_SELECTED_PARENT_REGISTERED_EVIDENCE_INTACT"
            ),
            "lead_count": len(lead_receipts),
            "registered_evidence_files": len(all_evidence),
            "safety": {
                "metadata_only": True,
                "raw_payloads_opened": 0,
                "sealed_target_rows_opened": 0,
                "network_calls": 0,
                "gpu_production_runs": 0,
                "paid_model_calls": 0,
            },
            "lead_programs": lead_receipts,
            "claim_boundary": {
                "registry_pass_establishes_empirical_replication": False,
                "registry_pass_establishes_physical_mechanism": False,
                "registry_pass_establishes_alternative_to_gr": False,
                "registry_pass_establishes_historical_novelty": False,
                "registry_pass_only_establishes_metadata_integrity": True,
            },
        }
    )


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_receipt(root)
    output = (root / OUTPUT_PATH).resolve()
    try:
        output.relative_to(root / "runs" / "gravity" / "lead-programs")
    except ValueError as error:
        raise GravityLeadParentRegistryError("registry output escaped lead-program directory") from error
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(_canonical_bytes(receipt))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "write-receipt"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--lead", choices=LEAD_IDS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    try:
        if args.command == "write-receipt":
            if args.lead is not None:
                raise GravityLeadParentRegistryError("stored receipt must cover all five leads")
            path = write_receipt(root)
            result: Any = {"output": path.relative_to(root).as_posix(), "receipt": _read_json(path)}
        else:
            result = build_receipt(root, args.lead)
    except GravityLeadParentRegistryError as error:
        print(json.dumps({"ok": False, "error": str(error)}, sort_keys=True))
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
