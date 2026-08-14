"""Authorization-first System 11 closure for the frozen G2 Solar one-shot."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sigma_theory_compiler.sigma_core import canonical_json_bytes, canonical_sha256

CONFIG_PATH = "configs/system11_g2_solar_authorization_closure.json"
OUTPUT_PATH = "runs/math/system11-g2-solar-authorization-closure/receipt.json"
CONFIG_SCHEMA = "sigma-system11-g2-solar-authorization-closure-config-1.0"
RESULT_SCHEMA = "sigma-system11-g2-solar-authorization-closure-result-1.0"
OPENING_SCHEMA = "sigma-system11-g2-solar-opening-packet-1.0"
AUTH_RECEIPT_SCHEMA = "sigma-system11-g2-solar-opening-authorization-receipt-1.0"
CAMPAIGN_ID = "system11-g2-solar-authorization-closure-001"
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_CANDIDATES = (
    (
        "G3A-2f8983c88f504150381064f2",
        "19f36a7c814ca11ace6de1270802a542872c35c27c7e64542eea672e16cbae88",
    ),
    (
        "G3A-58e59412e5fe77cd54caf863",
        "9457ba1ff99ecfdabc08200dda3ff15b8656b025d106fe2c2cd4abd77a01c3b5",
    ),
)
_OBLIGATIONS = (
    "source_branch_domain_instantiation_sha256",
    "held_out_split_commitment_sha256",
    "selected_primary_record_roots_sha256",
    "observation_opening_authorization_sha256",
)
_AUTHORITY_FLAGS = {
    "candidate_use_authorized",
    "formula_search_authorized",
    "observation_opening_authorized",
    "observational_authorization",
}
_OPEN_FLAGS = {
    "data_opened",
    "observational_data_opened",
    "observational_dataset_opened",
    "target_values_accessed",
}


class System11SolarAuthorizationError(ValueError):
    """A sealed source or authorization-first boundary changed."""


def _load_json(path: Path, *, size_limit: int = 2_000_000) -> dict[str, Any]:
    try:
        if not path.is_file() or path.stat().st_size > size_limit:
            raise System11SolarAuthorizationError(
                f"JSON source missing or oversized: {path.name}"
            )
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise System11SolarAuthorizationError(f"cannot read JSON: {path.name}") from error
    if not isinstance(value, dict):
        raise System11SolarAuthorizationError("JSON root must be an object")
    return value


def _resolve(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise System11SolarAuthorizationError("path must be portable and relative")
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise System11SolarAuthorizationError("path escapes project root") from error
    return path


def _semantic_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha(value: object, label: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise System11SolarAuthorizationError(f"invalid SHA-256: {label}")
    return value


def _validate_config(config: Mapping[str, Any]) -> None:
    expected = {
        "schema_version",
        "campaign_id",
        "source_bindings",
        "candidates",
        "opening_obligations",
        "launch_contract",
    }
    if set(config) != expected:
        raise System11SolarAuthorizationError("config keys changed")
    if config["schema_version"] != CONFIG_SCHEMA or config["campaign_id"] != CAMPAIGN_ID:
        raise System11SolarAuthorizationError("config identity changed")
    if config["candidates"] != [
        {"candidate_id": candidate_id, "action_sha256": action_sha}
        for candidate_id, action_sha in _CANDIDATES
    ]:
        raise System11SolarAuthorizationError("candidate/action freeze changed")
    if config["opening_obligations"] != list(_OBLIGATIONS):
        raise System11SolarAuthorizationError("opening obligations changed")
    launch = config["launch_contract"]
    if not isinstance(launch, Mapping) or launch != {
        "opening_packet_schema": OPENING_SCHEMA,
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
        "consume_packet_once": True,
        "target_data_read_by_preflight": False,
        "command": (
            "python -m sigma_theory_compiler.system11_g2_solar_authorization_closure "
            "--root . --opening-packet <opening-packet.json> "
            "--consume-once-output <authorization-receipt.json>"
        ),
    }:
        raise System11SolarAuthorizationError("launch contract changed")
    bindings = config["source_bindings"]
    if not isinstance(bindings, Mapping) or len(bindings) != 10:
        raise System11SolarAuthorizationError("source-binding inventory changed")
    for role, descriptor in bindings.items():
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "path",
            "semantic_sha256",
        }:
            raise System11SolarAuthorizationError(f"binding changed: {role}")
        _require_sha(descriptor["semantic_sha256"], f"binding {role}")


def _load_bound_values(
    root: Path, config: Mapping[str, Any]
) -> dict[str, dict[str, Any]]:
    values: dict[str, dict[str, Any]] = {}
    for role, descriptor in sorted(config["source_bindings"].items()):
        value = _load_json(_resolve(root, descriptor["path"]))
        if _semantic_sha256(value) != descriptor["semantic_sha256"]:
            raise System11SolarAuthorizationError(f"bound semantic source changed: {role}")
        values[role] = value
    return values


def _true_flags(value: object, path: str = "$") -> list[str]:
    found: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in (_AUTHORITY_FLAGS | _OPEN_FLAGS) and item is True:
                found.append(child)
            found.extend(_true_flags(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_true_flags(item, f"{path}[{index}]"))
    return found


def _validate_sealed_boundary(values: Mapping[str, Mapping[str, Any]]) -> None:
    receipt = values["one_shot_receipt"]
    transfer = values["transfer_registration"]
    parser = values["parser_readiness"]
    calibration = values["calibration_readiness"]
    registration = values["cassini_registration"]
    cassini_audit = values["cassini_audit"]
    protocol = values["solar_protocol"]
    protocol_audit = values["solar_protocol_audit"]
    policy = values["evidence_policy"]
    if (
        receipt.get("decision") != "block"
        or receipt.get("missing_opening_fields") != sorted(_OBLIGATIONS)
        or receipt.get("counts", {}).get("real_data_evaluations") != 0
        or receipt.get("data_boundary", {}).get("observational_data_opened") is not False
        or transfer.get("observational_authorization") is not False
        or transfer.get("observational_data_opened") is not False
        or transfer.get("primary_record_access_count") != 0
        or parser.get("observational_authorization") is not False
        or parser.get("metadata_selection", {}).get("primary_record_access_count") != 0
        or calibration.get("observational_authorization") is not False
        or calibration.get("primary_record_access_count") != 0
    ):
        raise System11SolarAuthorizationError("G2 one-shot sealed boundary changed")
    if (
        registration.get("data_opened") is not False
        or registration.get("candidate_use_authorized") is not False
        or registration.get("readiness", {}).get("primary_files_downloaded") is not False
        or cassini_audit.get("candidate_use_authorized") is not False
        or cassini_audit.get("observational_dataset_opened") is not False
        or protocol.get("data_opened") is not False
        or protocol_audit.get("observational_dataset_opened") is not False
        or protocol_audit.get("formula_search_authorized") is not False
        or policy.get("status") != "frozen"
    ):
        raise System11SolarAuthorizationError("Cassini policy boundary changed")
    if any(_true_flags(value) for value in values.values()):
        raise System11SolarAuthorizationError("unexpected local observational authorization")


def _opening_packet_contract() -> dict[str, Any]:
    return {
        "schema_version": OPENING_SCHEMA,
        "exact_top_level_fields": [
            "schema_version",
            "campaign_id",
            "candidate_source_domains",
            "held_out_split_commitment",
            "selected_primary_record_roots",
            "independent_authorization",
        ],
        "candidate_source_domains": [
            {
                "candidate_id": candidate_id,
                "action_sha256": action_sha,
                "document_descriptor": ["path", "semantic_sha256"],
            }
            for candidate_id, action_sha in _CANDIDATES
        ],
        "shared_document_descriptors": {
            "held_out_split_commitment": ["path", "semantic_sha256"],
            "selected_primary_record_roots": ["path", "semantic_sha256"],
            "independent_authorization": ["path", "semantic_sha256"],
        },
        "independent_authorization_must_bind": [
            "both exact candidate/action pairs",
            "both source-domain semantic roots",
            "held-out split semantic root",
            "selected primary-record semantic root",
            "one atomic opening",
            "two candidate evaluations",
            "zero refits and zero promotion actions",
        ],
    }


def build_receipt(root: Path) -> dict[str, Any]:
    """Audit bound metadata without reading any target or primary record payload."""
    config = _load_json(root / CONFIG_PATH)
    _validate_config(config)
    values = _load_bound_values(root, config)
    _validate_sealed_boundary(values)
    result: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "block",
        "first_blocker": "independent_observation_opening_authorization_absent",
        "candidate_ids": [candidate_id for candidate_id, _ in _CANDIDATES],
        "missing_opening_fields": list(_OBLIGATIONS),
        "missing_obligations_by_candidate": {
            candidate_id: list(_OBLIGATIONS) for candidate_id, _ in _CANDIDATES
        },
        "authorization_inventory": {
            "bound_metadata_artifacts_audited": len(values),
            "already_authorized_observational_artifacts": 0,
            "true_authority_or_open_flags": [],
            "other_authorized_observational_evidence_in_scope": [],
        },
        "data_boundary": {
            "target_data_read": False,
            "primary_record_payloads_read": False,
            "held_out_target_access_count": 0,
            "primary_record_access_count": 0,
            "real_data_evaluation_count": 0,
        },
        "counts": {
            "candidates": 2,
            "bound_metadata_artifacts_audited": len(values),
            "unique_missing_external_opening_fields": len(_OBLIGATIONS),
            "missing_external_opening_obligations": len(_CANDIDATES)
            * len(_OBLIGATIONS),
            "already_authorized_observational_artifacts": 0,
            "primary_record_accesses": 0,
            "held_out_target_accesses": 0,
            "real_data_evaluations": 0,
        },
        "one_shot_budget": {
            "atomic_open_batches": 1,
            "candidate_evaluations": 2,
            "refits_after_open": 0,
            "promotion_actions": 0,
        },
        "opening_packet_contract": _opening_packet_contract(),
        "launch_contract": config["launch_contract"],
        "execution_implementation_audit": {
            "PDS_label_and_record_parser": "implementation_ready_primary_records_sealed",
            "direct_signal_calibration": "implementation_ready_primary_records_sealed",
            "registered_evaluator_role": "synthetic_GR_known_answer_control_only",
            "action_bound_real_record_likelihood_executor": "block_not_registered",
            "first_post_authorization_blocker": (
                "registered_action_bound_direct_signal_likelihood_executor_sha256"
            ),
            "preflight_command_executes_observational_evaluation": False,
        },
        "claims": {
            "opening_authorized": False,
            "one_shot_executed": False,
            "candidate_supported_by_data": False,
            "candidate_rejected_by_data": False,
            "promotion_authorized": False,
        },
        "source_bindings": config["source_bindings"],
    }
    result["content_sha256"] = canonical_sha256(result)
    return result


def _descriptor_document(
    root: Path, descriptor: object, label: str
) -> tuple[dict[str, Any], str]:
    if not isinstance(descriptor, Mapping) or set(descriptor) != {
        "path",
        "semantic_sha256",
    }:
        raise System11SolarAuthorizationError(f"opening descriptor changed: {label}")
    expected = _require_sha(descriptor["semantic_sha256"], label)
    value = _load_json(_resolve(root, descriptor["path"]))
    actual = _semantic_sha256(value)
    if actual != expected:
        raise System11SolarAuthorizationError(f"opening document changed: {label}")
    return value, actual


def validate_and_consume_opening_packet(
    root: Path, packet_path: Path, output_path: Path
) -> dict[str, Any]:
    """Validate external authority metadata once; never open referenced record payloads."""
    if output_path.exists():
        raise System11SolarAuthorizationError("consume-once output already exists")
    packet = _load_json(packet_path)
    expected_keys = {
        "schema_version",
        "campaign_id",
        "candidate_source_domains",
        "held_out_split_commitment",
        "selected_primary_record_roots",
        "independent_authorization",
    }
    if set(packet) != expected_keys or packet.get("schema_version") != OPENING_SCHEMA:
        raise System11SolarAuthorizationError("opening packet schema changed")
    if packet.get("campaign_id") != CAMPAIGN_ID:
        raise System11SolarAuthorizationError("opening packet campaign changed")
    domains = packet["candidate_source_domains"]
    if not isinstance(domains, list) or len(domains) != 2:
        raise System11SolarAuthorizationError("source-domain inventory changed")
    domain_hashes: dict[str, str] = {}
    expected_candidates = dict(_CANDIDATES)
    for entry in domains:
        if not isinstance(entry, Mapping) or set(entry) != {
            "candidate_id",
            "action_sha256",
            "document",
        }:
            raise System11SolarAuthorizationError("source-domain entry changed")
        candidate_id = entry["candidate_id"]
        if expected_candidates.get(candidate_id) != entry["action_sha256"]:
            raise System11SolarAuthorizationError("candidate/action authorization changed")
        document, semantic = _descriptor_document(
            root, entry["document"], f"source domain {candidate_id}"
        )
        if (
            document.get("candidate_id") != candidate_id
            or document.get("action_sha256") != entry["action_sha256"]
            or document.get("status") != "instantiated_before_target_access"
            or document.get("target_values_accessed") is not False
        ):
            raise System11SolarAuthorizationError("source-domain document is not admissible")
        domain_hashes[candidate_id] = semantic
    if set(domain_hashes) != set(expected_candidates):
        raise System11SolarAuthorizationError("candidate source domains incomplete")
    split, split_hash = _descriptor_document(
        root, packet["held_out_split_commitment"], "held-out split"
    )
    roots, roots_hash = _descriptor_document(
        root, packet["selected_primary_record_roots"], "primary roots"
    )
    authorization, authorization_hash = _descriptor_document(
        root, packet["independent_authorization"], "independent authorization"
    )
    if (
        split.get("status") != "committed_before_target_access"
        or split.get("target_values_accessed") is not False
        or split.get("candidate_ids") != list(expected_candidates)
    ):
        raise System11SolarAuthorizationError("held-out split is not admissible")
    record_roots = roots.get("record_roots_sha256")
    if (
        roots.get("status") != "selected_and_hash_bound_before_target_access"
        or roots.get("target_values_accessed") is not False
        or roots.get("primary_record_payloads_opened") is not False
        or not isinstance(record_roots, list)
        or not record_roots
        or any(_SHA256.fullmatch(str(item)) is None for item in record_roots)
    ):
        raise System11SolarAuthorizationError("primary-root manifest is not admissible")
    required_authority = {
        "status": "authorized",
        "independent_of_candidate_generation": True,
        "campaign_id": CAMPAIGN_ID,
        "candidate_actions": [
            {"candidate_id": candidate_id, "action_sha256": action_sha}
            for candidate_id, action_sha in _CANDIDATES
        ],
        "candidate_source_domain_sha256": domain_hashes,
        "held_out_split_commitment_sha256": split_hash,
        "selected_primary_record_roots_sha256": roots_hash,
        "atomic_open_batches": 1,
        "candidate_evaluations": 2,
        "refits_after_open": 0,
        "promotion_actions": 0,
    }
    for key, expected in required_authority.items():
        if authorization.get(key) != expected:
            raise System11SolarAuthorizationError(f"authorization binding changed: {key}")
    result: dict[str, Any] = {
        "schema_version": AUTH_RECEIPT_SCHEMA,
        "campaign_id": CAMPAIGN_ID,
        "decision": "pass_authorization_preflight_only",
        "opening_packet_semantic_sha256": _semantic_sha256(packet),
        "source_branch_domain_instantiation_sha256": domain_hashes,
        "held_out_split_commitment_sha256": split_hash,
        "selected_primary_record_roots_sha256": roots_hash,
        "observation_opening_authorization_sha256": authorization_hash,
        "one_shot_budget": {
            "atomic_open_batches": 1,
            "candidate_evaluations": 2,
            "refits_after_open": 0,
            "promotion_actions": 0,
        },
        "data_boundary": {
            "target_data_read": False,
            "primary_record_payloads_read": False,
            "authorization_metadata_consumptions": 1,
        },
        "claims": {
            "authorization_preflight_passed": True,
            "one_shot_executed": False,
            "observational_result_exists": False,
            "candidate_supported_by_data": False,
            "candidate_rejected_by_data": False,
        },
    }
    result["content_sha256"] = canonical_sha256(result)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with output_path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, sort_keys=True, separators=(",", ":")))
            handle.write("\n")
    except FileExistsError as error:
        raise System11SolarAuthorizationError("consume-once output already exists") from error
    return result


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--opening-packet", type=Path)
    parser.add_argument("--consume-once-output", type=Path)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.opening_packet is not None:
        if args.consume_once_output is None or args.output is not None:
            raise System11SolarAuthorizationError(
                "opening packet requires only --consume-once-output"
            )
        validate_and_consume_opening_packet(
            root,
            args.opening_packet.resolve(),
            args.consume_once_output.resolve(),
        )
        return 0
    if args.consume_once_output is not None:
        raise System11SolarAuthorizationError("consume-once output requires opening packet")
    _write_json(args.output or root / OUTPUT_PATH, build_receipt(root))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
