"""Guard one externally authorized X-CLASS identity-only catalog acquisition."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

CONFIG_PATH = Path("configs/gravity_group_scale_xclass_identity_executor_v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_group_scale_xclass_identity_executor_v1.py")
TEST_PATH = Path("tests/test_gravity_group_scale_xclass_identity_executor_v1.py")
PREFLIGHT_PATH = Path(
    "runs/gravity/publication-readiness/group-scale-xclass-identity-executor-v1-preflight.json"
)
EXECUTOR_DIRECTORY = Path(
    "runs/gravity/publication-readiness/group-scale-xclass-identity-executor-v1"
)
UNAUTHORIZED_PATH = EXECUTOR_DIRECTORY / "authorization-current-unauthorized.json"
APPROVED_AUTHORIZATION_PATH = EXECUTOR_DIRECTORY / "authorization-approved.json"
ACCESS_INTENT_PATH = EXECUTOR_DIRECTORY / "access-intent-run-001.json"
GET_ATTEMPT_PATH = EXECUTOR_DIRECTORY / "get-attempt-run-001.json"
RESULT_PATH = EXECUTOR_DIRECTORY / "identity-sanitized-v1.json"
TEMPORARY_PARENT = EXECUTOR_DIRECTORY / "private-temporary"

CONFIG_FILE_SHA256 = "bd68d4bcf33cb72044866fd894edbc927ab5ada6bb62daa597a23e5637a493f1"
TEST_FILE_SHA256 = "fa5d252d5dd3c2a5de72fddc846db92e82981477837df45144fe4784b94a5bf6"
UNAUTHORIZED_FILE_SHA256 = "b0e59667b93132cf89a151ae2c7c9fba0d44b63fe618b2cc46e910e21e5a7827"
CONFIG_SCHEMA = "invariant-gravity-group-scale-xclass-identity-executor-config-1.0"
PREFLIGHT_SCHEMA = "invariant-gravity-group-scale-xclass-identity-preflight-receipt-1.0"
AUTHORIZATION_SCHEMA = "invariant-gravity-group-scale-xclass-identity-authorization-1.0"
AUTHORIZED_STATUS = "authorized_for_exactly_one_xclass_identity_get"
INTENT_SCHEMA = "invariant-gravity-group-scale-xclass-identity-access-intent-1.0"
GET_ATTEMPT_SCHEMA = "invariant-gravity-group-scale-xclass-identity-get-attempt-1.0"
RESULT_SCHEMA = "invariant-gravity-group-scale-xclass-identity-result-1.0"
RUN_ID = "XCLASS_IDENTITY_V1_RUN_001"
SOURCE_URL = "https://cdsarc.cds.unistra.fr/ftp/J/A+A/710/A77/tabled1.dat"
EXPECTED_BYTES = 16895
EXPECTED_ROWS = 155
RECORD_BYTES = 109
RECORD_CONTENT_BYTES = 108
EXECUTE_SENTINEL = "EXECUTE_XCLASS_IDENTITY_V1_SINGLE_GET"
AUTHORIZATION_PHRASE = (
    "AUTHORIZE RUN XCLASS_IDENTITY_V1_RUN_001: DOWNLOAD EXACTLY ONE 16,895-BYTE "
    "155-ROW X-CLASS TABLED1.DAT MIXED SCIENTIFIC CATALOG AS OPAQUE PRIVATE "
    "TEMPORARY BYTES; DECODE ONLY XCLASS, RADEG, DEDEG, AND Z AT FROZEN "
    "FIXED-WIDTH BYTES; DELETE RAW BYTES BEFORE SANITIZED OUTPUT; NO REDIRECTS, "
    "RETRIES, HEAD, OBSID GUESSING, X-COP OVERLAP, SCIENCE SCORING, MODEL, OR "
    "PAID CALLS."
)
XCLASS_SLICE = slice(0, 5)
RA_SLICE = slice(6, 14)
DEC_SLICE = slice(15, 22)
REDSHIFT_SLICE = slice(23, 28)
DELIMITER_OFFSETS = (5, 14, 22)
XCLASS_FIELD = re.compile(rb"[ ]{0,4}[0-9]{1,5}")
RA_FIELD = re.compile(rb"[ ]{0,2}[0-9]{1,3}\.[0-9]{4}")
DEC_FIELD = re.compile(rb"[ ]*[+-]?[0-9]{1,2}\.[0-9]{4}")
REDSHIFT_FIELD = re.compile(rb"[0-9]\.[0-9]{3}")
PARENT_PATH_KEYS = ("config_path", "module_path", "test_path", "receipt_path")
PARENT_SHA_KEYS = (
    "config_file_sha256",
    "module_file_sha256",
    "test_file_sha256",
    "receipt_file_sha256",
)
SECTION_BINDINGS = {
    "parent_v3_binding": "dc6f653582885c1b2b5393622efa1aba755c2e5bbeaf007db9f9e32f240a69f4",
    "source_contract": "4570f8b903af239b08ea48ad00ea69e3a162072a39f0a9c965435b6d90309e3a",
    "network_contract": "94b1d5da1f59d4f2d23d75170df9eecf04faaeed22edd0ae7701659a9bf9529b",
    "column_contract": "5248daa4c77ac41d87d4424f5a4de100dcce42a33eee2bcfd758d0a3e41f9ad2",
    "private_payload_contract": "e6ba581f3061cf4639c495ab7e5e8af9a9a769b02235f80cdae0143b576463ef",
    "authorization_contract": "49da5cf9020bf5701b6870fd9ed79a49643132f509987610b592d19e6f29c77f",
    "obsid_contract": "47de40f1fb218c6923114c78cf30a15de50fe283e8143b331937fbc8fc832b40",
    "xcop_overlap_contract": "ff8835af1150b2762cfe68af27df2e2070ddff20d8a3fb26dc347120832088f6",
    "output_contract": "1948f9bb1ff6c2d264b71899a573e8e3b920e7cac1a323b610f4fc9f7b3a2d34",
    "execution_accounting_at_freeze": "4a5583fdf94549f2713b61ecb949e707b201e992c437502aa56e1a46767ae7a5",
    "claim_boundary": "108f8cc1f33af67997f38fbbac8dcbab23c5d60d1a56f4d28aaefbce276fece7",
    "publication_contract": "f50e0af2672b00fcb64b4d995c6ffd70f466895f7d5cf7445fbde43b34a0c2db",
}


class GravityGroupScaleXclassIdentityExecutorV1Error(RuntimeError):
    """Raised when a frozen contract or privacy boundary fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityGroupScaleXclassIdentityExecutorV1Error(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise GravityGroupScaleXclassIdentityExecutorV1Error(f"{label} keys changed")


def _confined(root: Path, relative: Path) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("path escapes repository root")
    return candidate


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "executor_id",
            "frozen_at",
            "purpose",
            "implementation_evidence",
            "parent_v3_binding",
            "source_contract",
            "network_contract",
            "column_contract",
            "private_payload_contract",
            "authorization_contract",
            "obsid_contract",
            "xcop_overlap_contract",
            "output_contract",
            "execution_accounting_at_freeze",
            "claim_boundary",
            "decision",
            "publication_contract",
            "preflight_receipt_path",
        },
        "executor config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_executor_preflight_external_authorization_required_unrun"
        or config["executor_id"] != "gravity-group-scale-xclass-identity-executor-v1"
        or config["frozen_at"] != "2026-08-29"
        or config["preflight_receipt_path"] != PREFLIGHT_PATH.as_posix()
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("executor identity changed")
    if config["implementation_evidence"] != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
        "test_binding_required": True,
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("implementation evidence changed")

    parent = config["parent_v3_binding"]
    if (
        parent["audit_id"] != "gravity-group-scale-source-audit-v3"
        or parent["git_commit"] != "31b229da"
        or parent["receipt_content_sha256"]
        != "91f4013f57c6b5c85b7cd80b02f5585b85435edd8f449a07f6cdf2f9dfd9dd96"
        or parent["required_parent_decision"]
        != "METADATA_SOURCE_AUDIT_V3_SEALED_ZERO_READY_LANES_XCLASS_PREFERRED_"
        "EFEDS_BACKUP_NO_ACQUISITION_AUTHORIZED"
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("parent V3 binding changed")

    source = config["source_contract"]
    if (
        source["url"] != SOURCE_URL
        or source["schema_source_url"]
        != "https://cdsarc.cds.unistra.fr/viz-bin/ReadMe/J/A%2BA/710/A77?format=html&tex=true"
        or source["schema_retrieved_at"] != "2026-08-29"
        or source["scheme"] != "https"
        or source["host"] != "cdsarc.cds.unistra.fr"
        or source["port"] != 443
        or source["request_method"] != "GET"
        or source["expected_network_bytes"] != EXPECTED_BYTES
        or source["expected_rows"] != EXPECTED_ROWS
        or source["wire_record_bytes_including_lf"] != RECORD_BYTES
        or source["wire_record_content_bytes"] != RECORD_CONTENT_BYTES
        or source["line_ending_hex"] != "0a"
        or source["publisher_checksum_available"] is not False
        or source["expected_source_sha256_before_get"] is not None
        or source["scientific_mixed_row_warning"] is not True
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("source contract changed")

    network = config["network_contract"]
    if (
        network["head_calls"] != 0
        or network["get_calls"] != 1
        or network["redirect_calls"] != 0
        or network["retry_calls"] != 0
        or network["maximum_http_attempts"] != 1
        or network["maximum_network_bytes"] != EXPECTED_BYTES
        or network["required_status"] != 200
        or network["required_content_length"] != EXPECTED_BYTES
        or network["transfer_encoding_allowed"] is not False
        or network["response_final_url_must_equal_request_url"] is not True
        or network["no_redirect_handler_required"] is not True
        or network["no_retry_path_exists"] is not True
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("network contract changed")

    columns = config["column_contract"]
    if (
        columns["wire_order_source_names"] != ["XClass", "RAdeg", "DEdeg", "z"]
        or columns["decode_allowlist"] != ["XClass", "RAdeg", "DEdeg", "z"]
        or columns["sanitized_output_names"]
        != ["source_object_id", "ra_deg", "dec_deg", "redshift"]
        or columns["fixed_width_slices_1_based_inclusive"]
        != {
            "XClass": [1, 5],
            "RAdeg": [7, 14],
            "DEdeg": [16, 22],
            "z": [24, 28],
            "opaque_suffix": [29, 108],
        }
        or columns["fixed_width_slices_zero_based_half_open"]
        != {
            "XClass": [0, 5],
            "RAdeg": [6, 14],
            "DEdeg": [15, 22],
            "z": [23, 28],
            "opaque_suffix": [28, 108],
        }
        or columns["required_single_space_delimiters_1_based"] != [6, 15, 23]
        or columns["required_single_space_delimiter_hex"] != "20"
        or columns["field_format_ascii"]
        != {
            "XClass": "I5 right-aligned decimal integer in exactly 5 bytes",
            "RAdeg": "F8.4 in exactly 8 bytes",
            "DEdeg": "F7.4 in exactly 7 bytes",
            "z": "F5.3 in exactly 5 bytes",
        }
        or columns["opaque_suffix_required"] is not True
        or columns["opaque_suffix_exact_bytes"] != 80
        or columns["source_object_ids_unique"] is not True
        or columns["coordinates_must_be_finite"] is not True
        or columns["scientific_values_instantiated"] is not False
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("column contract changed")

    private = config["private_payload_contract"]
    if (
        private["temporary_parent"] != TEMPORARY_PARENT.as_posix()
        or private["temporary_directory_mode_octal"] != "0700"
        or private["temporary_file_mode_octal"] != "0600"
        or private["path_components_user_controlled"] is not False
        or private["archive_extraction_performed"] is not False
        or private["symlinks_allowed"] is not False
        or private["raw_payload_deleted_before_output_publish"] is not True
        or private["temporary_directory_removed_on_success_or_exception"] is not True
        or private["raw_bytes_or_rows_may_appear_in_logs_exceptions_receipts_or_results"]
        is not False
        or private["sanitized_records_built_from_allowlisted_captures_only"] is not True
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("private payload contract changed")

    authorization = config["authorization_contract"]
    if (
        authorization["authorization_schema"] != AUTHORIZATION_SCHEMA
        or authorization["authorized_status"] != AUTHORIZED_STATUS
        or authorization["run_id"] != RUN_ID
        or authorization["required_authorization_phrase"] != AUTHORIZATION_PHRASE
        or authorization["authorization_manifest_path"] != APPROVED_AUTHORIZATION_PATH.as_posix()
        or authorization["current_unauthorized_manifest_path"] != UNAUTHORIZED_PATH.as_posix()
        or authorization["current_unauthorized_manifest_sha256"] != UNAUTHORIZED_FILE_SHA256
        or authorization["cli_execute_sentinel"] != EXECUTE_SENTINEL
        or authorization["authorization_checked_before_temporary_directory_or_network"] is not True
        or authorization["authorized_manifest_present_at_freeze"] is not False
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization contract changed")

    obsid = config["obsid_contract"]
    if (
        obsid["tabled1_obsid_field_available"] is not False
        or obsid["obsid_mapping_executed"] is not False
        or obsid["obsid_guessing_allowed"] is not False
        or obsid["short_name_inference_allowed"] is not False
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("ObsID contract changed")
    overlap = config["xcop_overlap_contract"]
    if (
        overlap["coordinate_ledger_bound"] is not False
        or overlap["overlap_executed"] is not False
        or overlap["overlap_count"] is not None
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("X-COP overlap contract changed")

    output = config["output_contract"]
    if (
        output["access_intent_path"] != ACCESS_INTENT_PATH.as_posix()
        or output["access_intent_schema"] != INTENT_SCHEMA
        or output["access_intent_present_at_freeze"] is not False
        or output["access_intent_published_after_authorization_before_temporary_or_network"]
        is not True
        or output["access_intent_retained_on_success_failure_or_interruption"] is not True
        or output["get_attempt_marker_path"] != GET_ATTEMPT_PATH.as_posix()
        or output["get_attempt_marker_schema"] != GET_ATTEMPT_SCHEMA
        or output["get_attempt_marker_present_at_freeze"] is not False
        or output["get_attempt_marker_published_immediately_before_opener_call"] is not True
        or output["get_attempt_marker_retained_on_success_failure_or_interruption"] is not True
        or output["result_path"] != RESULT_PATH.as_posix()
        or output["result_schema"] != RESULT_SCHEMA
        or output["record_count"] != EXPECTED_ROWS
        or output["atomic_no_clobber"] is not True
        or output["same_directory_hard_link_no_replace"] is not True
        or output["result_present_at_freeze"] is not False
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("output contract changed")
    if set(output["record_keys"]) != {"source_object_id", "ra_deg", "dec_deg", "redshift"}:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("output record schema changed")

    accounting = config["execution_accounting_at_freeze"]
    if any(value != 0 for value in accounting.values()):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("freeze accounting changed")
    claims = config["claim_boundary"]
    allowed_true = {"guarded_executor_implemented", "authorization_template_frozen"}
    for key, value in claims.items():
        if key in allowed_true and value is not True:
            raise GravityGroupScaleXclassIdentityExecutorV1Error(f"claim boundary changed: {key}")
        if key not in allowed_true and value is not False:
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                f"claim boundary overstated: {key}"
            )
    if config["decision"] != (
        "GUARDED_XCLASS_IDENTITY_EXECUTOR_V1_FROZEN_EXTERNAL_AUTHORIZATION_REQUIRED_ZERO_RUNS"
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("decision changed")
    publication = config["publication_contract"]
    if (
        publication["publish_primitive"] != "SAME_DIRECTORY_HARD_LINK_NO_REPLACE"
        or publication["staging_file_fsync_before_publish"] is not True
        or publication["existing_different_preflight_action"] != "FAIL_CLOSED"
        or publication["race_loser_action"] != "VERIFY_IDENTICAL_OR_FAIL_CLOSED"
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("publication contract changed")
    for section, expected in SECTION_BINDINGS.items():
        if _sha(config[section]) != expected:
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                f"frozen nested section changed: {section}"
            )


def _validate_unauthorized_manifest(value: Mapping[str, Any]) -> None:
    _strict(
        value,
        {
            "schema_version",
            "status",
            "authorized",
            "run_id",
            "authorization_phrase",
            "approved_by",
            "approved_at",
            "contract_path",
            "source_url",
            "maximum_get_calls",
            "maximum_network_bytes",
            "scientific_payload_exposure_acknowledged",
            "bindings",
            "note",
        },
        "unauthorized manifest",
    )
    if value != {
        "schema_version": AUTHORIZATION_SCHEMA,
        "status": "not_authorized",
        "authorized": False,
        "run_id": RUN_ID,
        "authorization_phrase": None,
        "approved_by": None,
        "approved_at": None,
        "contract_path": CONFIG_PATH.as_posix(),
        "source_url": SOURCE_URL,
        "maximum_get_calls": 1,
        "maximum_network_bytes": EXPECTED_BYTES,
        "scientific_payload_exposure_acknowledged": False,
        "bindings": None,
        "note": (
            "This file is an explicit refusal state, not an approval template or executable "
            "authorization."
        ),
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("unauthorized manifest changed")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = _confined(root, CONFIG_PATH)
    if not config_path.is_file() or _file_sha(config_path) != CONFIG_FILE_SHA256:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("executor config hash changed")
    config = _read_json(config_path)
    validate_config(config)
    if _file_sha(_confined(root, TEST_PATH)) != TEST_FILE_SHA256:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("executor test binding changed")
    parent = config["parent_v3_binding"]
    for path_key, sha_key in zip(PARENT_PATH_KEYS, PARENT_SHA_KEYS, strict=True):
        path = _confined(root, Path(parent[path_key]))
        if not path.is_file() or _file_sha(path) != parent[sha_key]:
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                f"parent V3 binding changed: {path_key}"
            )
    parent_receipt = _read_json(_confined(root, Path(parent["receipt_path"])))
    if (
        parent_receipt.get("content_sha256") != parent["receipt_content_sha256"]
        or parent_receipt.get("decision") != parent["required_parent_decision"]
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("parent V3 receipt changed")
    unauthorized = _confined(root, UNAUTHORIZED_PATH)
    if not unauthorized.is_file() or _file_sha(unauthorized) != UNAUTHORIZED_FILE_SHA256:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("unauthorized manifest seal changed")
    _validate_unauthorized_manifest(_read_json(unauthorized))
    return config


def build_preflight_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "executor_id": config["executor_id"],
        "status": config["status"],
        "decision": config["decision"],
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(_confined(root, CONFIG_PATH)),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "module_path": MODULE_PATH.as_posix(),
            "module_file_sha256": _file_sha(_confined(root, MODULE_PATH)),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(_confined(root, TEST_PATH)),
        },
        "parent_v3_binding": config["parent_v3_binding"],
        "unauthorized_manifest_binding": {
            "path": UNAUTHORIZED_PATH.as_posix(),
            "file_sha256": _file_sha(_confined(root, UNAUTHORIZED_PATH)),
            "status": "not_authorized",
            "authorized": False,
        },
        "source_contract": config["source_contract"],
        "network_contract": config["network_contract"],
        "column_contract": config["column_contract"],
        "private_payload_contract": config["private_payload_contract"],
        "authorization_contract": config["authorization_contract"],
        "obsid_contract": config["obsid_contract"],
        "xcop_overlap_contract": config["xcop_overlap_contract"],
        "output_contract": config["output_contract"],
        "execution_accounting": config["execution_accounting_at_freeze"],
        "claims": config["claim_boundary"],
        "claim_limit": (
            "Guarded executor mechanics only. No authorization, network access, identity rows, "
            "ObsID mapping, X-COP overlap, reduction, score, CP10 completion, or publication claim."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_preflight_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    content_sha = body.pop("content_sha256", None)
    if content_sha != _sha(body):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("preflight content hash changed")
    if receipt != build_preflight_receipt(root):
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "preflight differs from frozen package"
        )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_no_replace(
    path: Path, payload: bytes, *, identical_existing_allowed: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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
            if identical_existing_allowed and path.read_bytes() == payload:
                return
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                f"refusing to replace or reuse existing artifact: {path.name}"
            ) from None
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_preflight_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_preflight_receipt(root)
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    path = _confined(root, PREFLIGHT_PATH)
    _atomic_publish_no_replace(path, payload, identical_existing_allowed=True)
    validate_preflight_receipt(_read_json(path), root)
    return path


def _validate_stored_preflight(root: Path) -> dict[str, Any]:
    path = _confined(root, PREFLIGHT_PATH)
    if not path.is_file():
        raise GravityGroupScaleXclassIdentityExecutorV1Error("preflight receipt missing")
    receipt = _read_json(path)
    validate_preflight_receipt(receipt, root)
    return receipt


def expected_authorization_bindings(root: Path) -> dict[str, str]:
    root = root.resolve()
    preflight = _validate_stored_preflight(root)
    parent = load_config(root)["parent_v3_binding"]
    return {
        "config_file_sha256": _file_sha(_confined(root, CONFIG_PATH)),
        "module_file_sha256": _file_sha(_confined(root, MODULE_PATH)),
        "test_file_sha256": _file_sha(_confined(root, TEST_PATH)),
        "preflight_receipt_file_sha256": _file_sha(_confined(root, PREFLIGHT_PATH)),
        "preflight_receipt_content_sha256": preflight["content_sha256"],
        "parent_v3_receipt_file_sha256": parent["receipt_file_sha256"],
        "parent_v3_receipt_content_sha256": parent["receipt_content_sha256"],
    }


def validate_authorization(
    root: Path, authorization_path: Path, expected_authorization_sha256: str
) -> dict[str, Any]:
    root = root.resolve()
    nominal_supplied = (
        authorization_path if authorization_path.is_absolute() else root / authorization_path
    )
    nominal_supplied = nominal_supplied.absolute()
    if nominal_supplied.is_symlink():
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization symlink prohibited")
    supplied = nominal_supplied.resolve()
    unauthorized = _confined(root, UNAUTHORIZED_PATH)
    approved = _confined(root, APPROVED_AUTHORIZATION_PATH)
    if supplied == unauthorized:
        if _file_sha(supplied) != UNAUTHORIZED_FILE_SHA256:
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                "authorization hash changed before refusal"
            )
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization is not authorized")
    if supplied != approved or not supplied.is_relative_to(root):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization path is not approved")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_authorization_sha256):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization hash format changed")
    if not supplied.is_file() or _file_sha(supplied) != expected_authorization_sha256:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization hash mismatch")
    authorization = _read_json(supplied)
    expected_keys = {
        "schema_version",
        "status",
        "authorized",
        "run_id",
        "authorization_phrase",
        "approved_by",
        "approved_at",
        "scientific_payload_exposure_acknowledged",
        "source_url",
        "maximum_get_calls",
        "maximum_network_bytes",
        "bindings",
    }
    _strict(authorization, expected_keys, "approved authorization")
    if (
        authorization["schema_version"] != AUTHORIZATION_SCHEMA
        or authorization["status"] != AUTHORIZED_STATUS
        or authorization["authorized"] is not True
        or authorization["run_id"] != RUN_ID
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization is not authorized")
    if authorization["authorization_phrase"] != AUTHORIZATION_PHRASE:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization phrase mismatch")
    if authorization["scientific_payload_exposure_acknowledged"] is not True:
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "scientific payload exposure was not acknowledged"
        )
    if authorization["source_url"] != SOURCE_URL:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization source mismatch")
    if (
        authorization["maximum_get_calls"] != 1
        or authorization["maximum_network_bytes"] != EXPECTED_BYTES
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization network bounds changed")
    if (
        not isinstance(authorization["approved_by"], str)
        or not authorization["approved_by"].strip()
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization approver missing")
    if not isinstance(authorization["approved_at"], str):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization timestamp missing")
    try:
        approved_at = datetime.fromisoformat(authorization["approved_at"])
    except ValueError as error:
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "authorization timestamp invalid"
        ) from error
    if approved_at.utcoffset() != UTC.utcoffset(None):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization timestamp must be UTC")
    if authorization["bindings"] != expected_authorization_bindings(root):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("authorization bindings mismatch")
    return authorization


def _intent_body(authorization: Mapping[str, Any], authorization_sha256: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": INTENT_SCHEMA,
        "status": "AUTHORIZED_ACCESS_INTENT_RETAINED_ATTEMPT_NOT_YET_PROVEN",
        "run_id": RUN_ID,
        "source_url": SOURCE_URL,
        "maximum_get_calls": 1,
        "maximum_network_bytes": EXPECTED_BYTES,
        "authorization_file_sha256": authorization_sha256,
        "approved_by": authorization["approved_by"],
        "approved_at": authorization["approved_at"],
        "config_file_sha256": CONFIG_FILE_SHA256,
        "privacy_boundary": "MIXED_BYTES_PRIVATE_ALLOWLISTED_IDENTITY_ONLY",
    }
    return {**body, "content_sha256": _sha(body)}


def _publish_access_intent(
    root: Path, authorization: Mapping[str, Any], authorization_sha256: str
) -> None:
    intent = _intent_body(authorization, authorization_sha256)
    payload = (
        json.dumps(intent, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    _atomic_publish_no_replace(
        _confined(root, ACCESS_INTENT_PATH), payload, identical_existing_allowed=False
    )


def _validate_access_intent(
    root: Path, authorization: Mapping[str, Any], authorization_sha256: str
) -> None:
    path = _confined(root, ACCESS_INTENT_PATH)
    if not path.is_file() or path.is_symlink():
        raise GravityGroupScaleXclassIdentityExecutorV1Error("access intent missing or unsafe")
    intent = _read_json(path)
    if intent != _intent_body(authorization, authorization_sha256):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("access intent binding changed")


def _get_attempt_body(authorization_sha256: str) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": GET_ATTEMPT_SCHEMA,
        "status": "ONE_GET_ATTEMPT_COUNTED_CONSERVATIVELY_COMPLETION_NOT_YET_PROVEN",
        "run_id": RUN_ID,
        "source_url": SOURCE_URL,
        "authorization_file_sha256": authorization_sha256,
        "get_attempts_conservative": 1,
        "get_completions": None,
        "network_bytes_completed": None,
        "retry_calls_allowed": 0,
    }
    return {**body, "content_sha256": _sha(body)}


def _publish_get_attempt(root: Path, authorization_sha256: str) -> None:
    marker = _get_attempt_body(authorization_sha256)
    payload = (
        json.dumps(marker, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    _atomic_publish_no_replace(
        _confined(root, GET_ATTEMPT_PATH), payload, identical_existing_allowed=False
    )


def _validate_get_attempt(root: Path, authorization_sha256: str) -> None:
    path = _confined(root, GET_ATTEMPT_PATH)
    if not path.is_file() or path.is_symlink():
        raise GravityGroupScaleXclassIdentityExecutorV1Error("GET attempt marker missing or unsafe")
    if _read_json(path) != _get_attempt_body(authorization_sha256):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("GET attempt marker binding changed")


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        raise HTTPError(request.full_url, code, "redirect prohibited", headers, file_pointer)


def _default_opener() -> Any:
    return build_opener(_NoRedirect())


def _ensure_private_parent(root: Path) -> Path:
    root = root.resolve()
    current = root
    for part in TEMPORARY_PARENT.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                "private temporary path contains a symlink"
            )
    parent = root / TEMPORARY_PARENT
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if parent.is_symlink() or parent.resolve() != _confined(root, TEMPORARY_PARENT):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("private temporary path changed")
    try:
        parent.chmod(0o700)
    except OSError:
        pass
    return parent


def _download_once(opener: Any, destination: Path, config: Mapping[str, Any]) -> str:
    network = config["network_contract"]
    request = Request(SOURCE_URL, headers=network["request_headers"], method="GET")
    try:
        response_context = opener.open(request, timeout=network["timeout_seconds"])
        with response_context as response:
            status = response.getcode()
            if status != 200:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    "single GET returned prohibited status"
                )
            if response.geturl() != SOURCE_URL:
                raise GravityGroupScaleXclassIdentityExecutorV1Error("single GET final URL changed")
            if response.headers.get("Transfer-Encoding") is not None:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    "single GET transfer encoding prohibited"
                )
            content_encoding = response.headers.get("Content-Encoding")
            if content_encoding not in (None, "identity"):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    "single GET content encoding prohibited"
                )
            content_length = response.headers.get("Content-Length")
            if (
                content_length is None
                or not content_length.isascii()
                or not content_length.isdigit()
            ):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    "single GET Content-Length missing or invalid"
                )
            if int(content_length) != EXPECTED_BYTES:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    "single GET Content-Length changed"
                )
            payload = response.read(EXPECTED_BYTES)
    except GravityGroupScaleXclassIdentityExecutorV1Error:
        raise
    except OSError:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("single GET failed") from None
    if len(payload) != EXPECTED_BYTES:
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "single GET response byte count changed"
        )
    source_sha = hashlib.sha256(payload).hexdigest()
    try:
        with destination.open("xb") as handle:
            try:
                destination.chmod(0o600)
            except OSError:
                pass
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        payload = b""
    return source_sha


def _parse_allowlisted_records(path: Path, config: Mapping[str, Any]) -> list[dict[str, Any]]:
    bounds = config["column_contract"]["numeric_bounds"]
    records: list[dict[str, Any]] = []
    identifiers: set[str] = set()
    if path.stat().st_size != EXPECTED_BYTES:
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "private payload byte count changed before fixed-width parsing"
        )
    with path.open("rb") as handle:
        for row_number in range(1, EXPECTED_ROWS + 1):
            line = handle.read(RECORD_BYTES)
            if len(line) != RECORD_BYTES or line[RECORD_CONTENT_BYTES:] != b"\n":
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"wire record structure changed at row {row_number}"
                )
            prefix = line[:28]
            if any(prefix[offset : offset + 1] != b" " for offset in DELIMITER_OFFSETS):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"fixed-width delimiter changed at row {row_number}"
                )
            xclass_raw = prefix[XCLASS_SLICE]
            ra_raw = prefix[RA_SLICE]
            dec_raw = prefix[DEC_SLICE]
            redshift_raw = prefix[REDSHIFT_SLICE]
            if (
                XCLASS_FIELD.fullmatch(xclass_raw) is None
                or RA_FIELD.fullmatch(ra_raw) is None
                or DEC_FIELD.fullmatch(dec_raw) is None
                or REDSHIFT_FIELD.fullmatch(redshift_raw) is None
            ):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"allowlisted fixed-width field changed at row {row_number}"
                )
            try:
                source_object_id = xclass_raw.decode("ascii").strip()
                ra_deg = float(ra_raw.decode("ascii"))
                dec_deg = float(dec_raw.decode("ascii"))
                redshift = float(redshift_raw.decode("ascii"))
            except (UnicodeDecodeError, ValueError):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"allowlisted prefix value invalid at row {row_number}"
                ) from None
            if source_object_id in identifiers:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"duplicate source identity at row {row_number}"
                )
            if not all(math.isfinite(value) for value in (ra_deg, dec_deg, redshift)):
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"nonfinite allowlisted value at row {row_number}"
                )
            if not bounds["ra_deg_min_inclusive"] <= ra_deg < bounds["ra_deg_max_exclusive"]:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"RA outside frozen bound at row {row_number}"
                )
            if not bounds["dec_deg_min_inclusive"] <= dec_deg <= bounds["dec_deg_max_inclusive"]:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"Dec outside frozen bound at row {row_number}"
                )
            if not bounds["redshift_min_exclusive"] < redshift < bounds["redshift_max_exclusive"]:
                raise GravityGroupScaleXclassIdentityExecutorV1Error(
                    f"redshift outside frozen bound at row {row_number}"
                )
            identifiers.add(source_object_id)
            records.append(
                {
                    "source_object_id": source_object_id,
                    "ra_deg": ra_deg,
                    "dec_deg": dec_deg,
                    "redshift": redshift,
                }
            )
    if len(records) != EXPECTED_ROWS:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("catalog row count changed")
    return records


def _build_result(
    root: Path,
    authorization: Mapping[str, Any],
    authorization_sha256: str,
    source_sha256: str,
    records: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "run_id": RUN_ID,
        "status": "IDENTITY_ONLY_OBSID_MAPPING_BLOCKED_SEPARATE_AUTHORITATIVE_SOURCE_REQUIRED",
        "authorization_receipt": {
            "path": APPROVED_AUTHORIZATION_PATH.as_posix(),
            "file_sha256": authorization_sha256,
            "approved_by": authorization["approved_by"],
            "approved_at": authorization["approved_at"],
        },
        "source_receipt": {
            "url": SOURCE_URL,
            "network_bytes": EXPECTED_BYTES,
            "row_count": EXPECTED_ROWS,
            "wire_record_bytes_including_lf": RECORD_BYTES,
            "source_sha256": source_sha256,
            "license": config["source_contract"]["license"],
        },
        "records": records,
        "obsid_mapping": {
            "executed": False,
            "obsids_published": 0,
            "status": "BLOCKED_SEPARATE_REGISTERED_XCLASS_DATABASE_OR_AUTHORITATIVE_MAPPING_REQUIRED",
            "guessing_used": False,
        },
        "xcop_overlap": {
            "executed": False,
            "count": None,
            "status": "BLOCKED_NO_INDEPENDENTLY_FROZEN_XCOP_COORDINATE_LEDGER",
        },
        "accounting": {
            "head_calls": 0,
            "get_attempts": 1,
            "get_completions": 1,
            "redirect_calls": 0,
            "retry_calls": 0,
            "network_bytes": EXPECTED_BYTES,
            "identity_rows_decoded": EXPECTED_ROWS,
            "scientific_values_decoded": 0,
            "obsid_mappings": 0,
            "xcop_overlap_runs": 0,
            "scores_computed": 0,
            "model_or_paid_calls": 0,
        },
        "claim_boundary": {
            "identity_only_output": True,
            "source_sha256_recorded_after_authorized_get": True,
            "scientific_values_decoded": False,
            "obsid_mapping_available": False,
            "xcop_overlap_known": False,
            "five_object_pilot_unlocked": False,
            "group_bridge_ready": False,
            "candidate_tested_on_groups": False,
            "CP10_1_complete": False,
            "CP10_2_complete": False,
            "publication_claim_supported": False,
        },
    }
    result = {**body, "content_sha256": _sha(body)}
    validate_result(result, root)
    return result


def validate_result(result: Mapping[str, Any], root: Path) -> None:
    config = load_config(root)
    output = config["output_contract"]
    if set(result) != set(output["exact_top_level_keys"]):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result top-level schema changed")
    body = dict(result)
    content_sha = body.pop("content_sha256", None)
    if content_sha != _sha(body):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result content hash changed")
    if (
        result["schema_version"] != RESULT_SCHEMA
        or result["run_id"] != RUN_ID
        or result["status"]
        != "IDENTITY_ONLY_OBSID_MAPPING_BLOCKED_SEPARATE_AUTHORITATIVE_SOURCE_REQUIRED"
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result identity changed")
    authorization = result["authorization_receipt"]
    _strict(authorization, set(output["authorization_receipt_keys"]), "authorization receipt")
    if authorization["path"] != APPROVED_AUTHORIZATION_PATH.as_posix():
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result authorization path changed")
    path = _confined(root, APPROVED_AUTHORIZATION_PATH)
    if not path.is_file() or path.is_symlink() or _file_sha(path) != authorization["file_sha256"]:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result authorization binding changed")
    approved = validate_authorization(root, path, authorization["file_sha256"])
    if (
        authorization["approved_by"] != approved["approved_by"]
        or authorization["approved_at"] != approved["approved_at"]
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "result authorization identity changed"
        )
    _validate_access_intent(root, approved, authorization["file_sha256"])
    _validate_get_attempt(root, authorization["file_sha256"])
    source = result["source_receipt"]
    _strict(source, set(output["source_receipt_keys"]), "source receipt")
    if (
        source["url"] != SOURCE_URL
        or source["network_bytes"] != EXPECTED_BYTES
        or source["row_count"] != EXPECTED_ROWS
        or source["wire_record_bytes_including_lf"] != RECORD_BYTES
        or not re.fullmatch(r"[0-9a-f]{64}", source["source_sha256"])
        or source["license"] != config["source_contract"]["license"]
    ):
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result source receipt changed")
    records = result["records"]
    if not isinstance(records, list) or len(records) != EXPECTED_ROWS:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result record count changed")
    identifiers: set[str] = set()
    for record in records:
        _strict(record, set(output["record_keys"]), "sanitized record")
        if not isinstance(record["source_object_id"], str) or not re.fullmatch(
            r"[0-9]{1,5}", record["source_object_id"]
        ):
            raise GravityGroupScaleXclassIdentityExecutorV1Error("result source identity invalid")
        if record["source_object_id"] in identifiers:
            raise GravityGroupScaleXclassIdentityExecutorV1Error("result duplicate identity")
        identifiers.add(record["source_object_id"])
        if not all(
            type(record[key]) is float and math.isfinite(record[key])
            for key in ("ra_deg", "dec_deg", "redshift")
        ):
            raise GravityGroupScaleXclassIdentityExecutorV1Error("result coordinate invalid")
        if not (
            0.0 <= record["ra_deg"] < 360.0
            and -90.0 <= record["dec_deg"] <= 90.0
            and 0.0 < record["redshift"] < 1.0
        ):
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                "result coordinate outside frozen bounds"
            )
    if result["obsid_mapping"] != {
        "executed": False,
        "obsids_published": 0,
        "status": "BLOCKED_SEPARATE_REGISTERED_XCLASS_DATABASE_OR_AUTHORITATIVE_MAPPING_REQUIRED",
        "guessing_used": False,
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result ObsID boundary changed")
    if result["xcop_overlap"] != {
        "executed": False,
        "count": None,
        "status": "BLOCKED_NO_INDEPENDENTLY_FROZEN_XCOP_COORDINATE_LEDGER",
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result X-COP boundary changed")
    if result["accounting"] != {
        "head_calls": 0,
        "get_attempts": 1,
        "get_completions": 1,
        "redirect_calls": 0,
        "retry_calls": 0,
        "network_bytes": EXPECTED_BYTES,
        "identity_rows_decoded": EXPECTED_ROWS,
        "scientific_values_decoded": 0,
        "obsid_mappings": 0,
        "xcop_overlap_runs": 0,
        "scores_computed": 0,
        "model_or_paid_calls": 0,
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result accounting changed")
    claims = result["claim_boundary"]
    if claims != {
        "identity_only_output": True,
        "source_sha256_recorded_after_authorized_get": True,
        "scientific_values_decoded": False,
        "obsid_mapping_available": False,
        "xcop_overlap_known": False,
        "five_object_pilot_unlocked": False,
        "group_bridge_ready": False,
        "candidate_tested_on_groups": False,
        "CP10_1_complete": False,
        "CP10_2_complete": False,
        "publication_claim_supported": False,
    }:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("result claim boundary changed")


def execute(
    root: Path,
    authorization_path: Path,
    expected_authorization_sha256: str,
    execute_sentinel: str,
    *,
    opener: Any | None = None,
) -> Path:
    root = root.resolve()
    config = load_config(root)
    _validate_stored_preflight(root)
    if execute_sentinel != EXECUTE_SENTINEL:
        raise GravityGroupScaleXclassIdentityExecutorV1Error("execute sentinel mismatch")
    authorization = validate_authorization(root, authorization_path, expected_authorization_sha256)
    result_path = _confined(root, RESULT_PATH)
    if result_path.exists():
        raise GravityGroupScaleXclassIdentityExecutorV1Error(
            "result already exists; replay refused"
        )
    _publish_access_intent(root, authorization, expected_authorization_sha256)
    temporary_parent = _ensure_private_parent(root)
    source_sha256: str
    records: list[dict[str, Any]]
    with tempfile.TemporaryDirectory(prefix="xclass-identity-v1-", dir=temporary_parent) as name:
        private_directory = Path(name)
        try:
            private_directory.chmod(0o700)
        except OSError:
            pass
        raw_path = private_directory / config["private_payload_contract"]["raw_filename"]
        _publish_get_attempt(root, expected_authorization_sha256)
        source_sha256 = _download_once(opener or _default_opener(), raw_path, config)
        records = _parse_allowlisted_records(raw_path, config)
        raw_path.unlink()
        _fsync_directory(private_directory)
        if raw_path.exists():
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                "private raw payload deletion not confirmed"
            )
    result = _build_result(
        root,
        authorization,
        expected_authorization_sha256,
        source_sha256,
        records,
        config,
    )
    payload = (
        json.dumps(result, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
    _atomic_publish_no_replace(result_path, payload, identical_existing_allowed=False)
    validate_result(_read_json(result_path), root)
    return result_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action", choices=("write-preflight", "check-preflight", "run", "check-result")
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--expected-authorization-sha256")
    parser.add_argument("--execute-sentinel")
    args = parser.parse_args(argv)
    if args.action == "write-preflight":
        print(write_preflight_receipt(args.root))
    elif args.action == "check-preflight":
        receipt = _validate_stored_preflight(args.root.resolve())
        print(json.dumps({"status": "PASS", "content_sha256": receipt["content_sha256"]}))
    elif args.action == "run":
        if (
            args.authorization is None
            or args.expected_authorization_sha256 is None
            or args.execute_sentinel is None
        ):
            raise GravityGroupScaleXclassIdentityExecutorV1Error(
                "run requires authorization, expected hash, and execute sentinel"
            )
        print(
            execute(
                args.root,
                args.authorization,
                args.expected_authorization_sha256,
                args.execute_sentinel,
            )
        )
    else:
        result = _read_json(_confined(args.root.resolve(), RESULT_PATH))
        validate_result(result, args.root)
        print(json.dumps({"status": "PASS", "content_sha256": result["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
