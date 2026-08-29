"""Prepare and gate the ACT DR6 x eRASS1 catalog-overlap executor."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tarfile
import tempfile
import unicodedata
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any, BinaryIO

CONFIG_PATH = Path("configs/gravity_cluster_act_dr6_erass1_overlap_executor_v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_cluster_act_dr6_erass1_overlap_executor.py")
TEST_PATH = Path("tests/test_gravity_cluster_act_dr6_erass1_overlap_executor.py")
CURRENT_AUTH_PATH = Path(
    "runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2/"
    "authorization-current-unauthorized.json"
)
APPROVED_AUTH_PATH = Path(
    "runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2/"
    "authorization-approved.json"
)
PREFLIGHT_PATH = Path(
    "runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2-preflight.json"
)
RESULT_DIR = Path("runs/gravity/publication-readiness/act-dr6-erass1-overlap-executor-v2-result")
CONFIG_FILE_SHA256 = "04d2cc80e21efd688707c036f54da266d20561f24fc34f3fe4bd912fe9387b3a"
TEST_FILE_SHA256 = "4ed1f3e71dabd7c14135743007681f12d047c9444824fd04f8dbcde2390e3b78"
CURRENT_AUTH_FILE_SHA256 = "22154a9521d680317251e9315be548469357c4c75887ae2382930b7d689404b9"
CONFIG_SCHEMA = "invariant-gravity-cluster-act-dr6-erass1-overlap-executor-config-2.0"
PREFLIGHT_SCHEMA = "invariant-gravity-cluster-act-dr6-erass1-overlap-executor-preflight-2.0"
AUTH_SCHEMA = "invariant-act-dr6-erass1-overlap-executor-authorization-2.0"
RESULT_SCHEMA = "invariant-act-dr6-erass1-overlap-count-receipt-2.0"
RUN_ID = "ACT_DR6_ERASS1_OVERLAP_V2_RUN_001"
AUTHORIZATION_PHRASE = (
    "AUTHORIZE RUN ACT_DR6_ERASS1_OVERLAP_V2_RUN_001: ACT/eRASS CATALOG-ONLY OVERLAP V2; "
    "DOWNLOAD 2 FULL CATALOG FILES AS OPAQUE TEMPORARY BYTES (EXACTLY 2 GET CALLS, "
    "33,469,695 NETWORK BYTES, 15,994 CATALOG ROWS), DECODE ONLY THE FROZEN ALLOWLIST, "
    "AND MAKE NO HEAD, REDIRECT, OR RETRY CALLS AND NO MAP, PROFILE, THERMODYNAMIC, LENSING, "
    "INFERRED-MASS, SCORING, MODEL, OR PAID CALLS."
)

CATALOG_IDS = ("ACT_DR6_LEGACY_V1_0", "ERASS1_PRIMARY_V3_2")
ACT_COLUMNS = (
    "name",
    "RADeg",
    "decDeg",
    "fixed_SNR",
    "flags",
    "footprint_eROSITADe",
    "footprint_Legacy",
    "eRASS1CL",
    "redshift",
    "redshiftErr",
    "redshiftType",
    "opt_RADeg",
    "opt_decDeg",
    "opt_positionSource",
    "warnings",
)
ERASS_COLUMNS = (
    "DETUID",
    "NAME",
    "RA",
    "DEC",
    "RA_XFIT",
    "DEC_XFIT",
    "EXT_LIKE",
    "DET_LIKE_0",
    "BEST_Z",
    "BEST_ZERR",
    "BEST_Z_TYPE",
    "PCONT",
    "MATCH_NAME",
)
PROJECTION_TFORM_CONTRACT = {
    "ACT_DR6_LEGACY_V1_0": {
        "name": "19A",
        "RADeg": "D",
        "decDeg": "D",
        "fixed_SNR": "D",
        "flags": "K",
        "footprint_eROSITADe": "L",
        "footprint_Legacy": "L",
        "eRASS1CL": "L",
        "redshift": "D",
        "redshiftErr": "D",
        "redshiftType": "1000A",
        "opt_RADeg": "D",
        "opt_decDeg": "D",
        "opt_positionSource": "11A",
        "warnings": "93A",
    },
    "ERASS1_PRIMARY_V3_2": {
        "DETUID": "32A",
        "NAME": "23A",
        "RA": "D",
        "DEC": "D",
        "RA_XFIT": "D",
        "DEC_XFIT": "D",
        "EXT_LIKE": "E",
        "DET_LIKE_0": "E",
        "BEST_Z": "D",
        "BEST_ZERR": "D",
        "BEST_Z_TYPE": "16A",
        "PCONT": "D",
        "MATCH_NAME": "240A",
    },
}
XCOP_OBJECTS = (
    "A1644",
    "A1795",
    "A2029",
    "A2142",
    "A2255",
    "A2319",
    "A3158",
    "A3266",
    "A644",
    "A85",
    "RXC1825",
    "ZW1215",
)
PAIR_COMPARISON_CEILING = 3747 * 12247
LEDGER_FIELDS = (
    "canonical_join_id",
    "act_name",
    "erass_detuid",
    "erass_name",
    "match_state",
    "match_method",
    "candidate_ids",
    "angular_separation_arcmin",
    "absolute_redshift_difference",
    "xcop_identity",
    "xcop_excluded",
    "eligible_catalog_overlap",
    "quarantine_reason",
)
AUTH_ACCESS_STATE_ZERO = {
    "network_calls": 0,
    "files_downloaded": 0,
    "network_bytes_downloaded": 0,
    "catalog_rows_opened": 0,
    "forbidden_values_decoded_or_logged": 0,
    "sanitized_ledger_rows_emitted": 0,
    "scores_computed": 0,
    "model_or_paid_calls": 0,
    "execution_started": False,
}
SECTION_BINDINGS = {
    "parent_binding": "6f4b0790054528aeef2ecb1ed5244bf63fecbb9d3d31d2768afab63f9652b04d",
    "implementation_evidence": "947c8b579c14bed4c3de1f24e392c0aeaaade731553622068cefab5f2e5a8c91",
    "current_authorization_binding": "613f4d7cfca01f7c7646c7b51ffddd6fe9c9741724648a01ea64e4a6ce12b096",
    "future_execution_contract": "94b7f99236bd4de5b03e2304b71923369d09637432f2ae3db174c949d78dbda5",
    "catalog_assets": "7a923a09ae31d1d2dedfc4991c06f565287b2552ac9bf153856b499e8ca6e914",
    "projection_contract": "179f9622306222ac746293020a7a463d689f76ad8b4926e0b6685cb8cb70d14b",
    "selection_match_and_exclusion_contract": "61e91437617e07faeb0e00df6ff51319c3c1bf7b593d008881e5900c50fa546a",
    "sanitized_output_contract": "1e307a833e4d680927c71192d828954834fc60aa3a80aa2bb3b9b7e30bc060b6",
    "failure_and_publication_contract": "693261a05f9f1a7f50adda4ac58f215941538c15db0a8fa7610c2a51847d4af8",
    "preflight_access_state": "8201ab7ce65e6292529c3090d627dc29181baa5c5f306a6b6fa02e205514f76e",
    "claim_boundary": "419841de3366e88dd01f43874f599bc8de7497d0affdaee57a6e2968b84788e5",
}


class GravityClusterActErassOverlapExecutorError(RuntimeError):
    """Raised when the frozen executor contract or an execution gate fails."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode() + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterActErassOverlapExecutorError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterActErassOverlapExecutorError(f"{label} keys changed")


def _asset_map(config: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {item["catalog_id"]: item for item in config["catalog_assets"]}


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "executor_id",
            "audit_cutoff",
            "purpose",
            "parent_binding",
            "implementation_evidence",
            "current_authorization_binding",
            "future_execution_contract",
            "catalog_assets",
            "projection_contract",
            "selection_match_and_exclusion_contract",
            "sanitized_output_contract",
            "failure_and_publication_contract",
            "preflight_access_state",
            "claim_boundary",
            "output_path",
        },
        "executor config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_unauthorized_executor_not_run"
        or config["executor_id"] != "gravity-cluster-act-dr6-erass1-overlap-executor-v2"
        or config["audit_cutoff"] != "2026-08-29"
        or config["output_path"] != PREFLIGHT_PATH.as_posix()
    ):
        raise GravityClusterActErassOverlapExecutorError("executor identity changed")
    if config["implementation_evidence"] != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
    }:
        raise GravityClusterActErassOverlapExecutorError("implementation evidence changed")
    if config["current_authorization_binding"] != {
        "path": CURRENT_AUTH_PATH.as_posix(),
        "file_sha256": CURRENT_AUTH_FILE_SHA256,
        "required_status": "UNAUTHORIZED_EXECUTOR_NOT_RUN",
        "authorization": False,
    }:
        raise GravityClusterActErassOverlapExecutorError("current authorization binding changed")
    parent = config["parent_binding"]
    if (
        parent["commit"] != "1ce8c666ecb45093738cf80a1fcde90f1f11f1ba"
        or parent["receipt_content_sha256"]
        != "5b982eed48caacd55a5fefbcda72b1fd7d597c6ae5c1c543d00667d4b5bc06b7"
    ):
        raise GravityClusterActErassOverlapExecutorError("parent identity changed")
    execution = config["future_execution_contract"]
    if (
        execution["approved_authorization_path"] != APPROVED_AUTH_PATH.as_posix()
        or execution["required_run_id"] != RUN_ID
        or execution["required_authorization_phrase"] != AUTHORIZATION_PHRASE
        or execution["output_directory"] != RESULT_DIR.as_posix()
        or execution["preflight_receipt_path"] != PREFLIGHT_PATH.as_posix()
        or execution["network_get_call_ceiling"] != 2
        or execution["network_head_call_ceiling"] != 0
        or execution["network_redirect_call_ceiling"] != 0
        or execution["network_retry_call_ceiling"] != 0
        or execution["network_byte_ceiling"] != 33469695
        or execution["catalog_row_ceiling"] != 15994
        or execution["scores_or_model_call_ceiling"] != 0
        or execution["authorization_gate_order"][-1]
        != "only_then_import_network_client_and_issue_GET"
    ):
        raise GravityClusterActErassOverlapExecutorError("execution ceiling or gate changed")
    assets = config["catalog_assets"]
    if tuple(item["catalog_id"] for item in assets) != CATALOG_IDS:
        raise GravityClusterActErassOverlapExecutorError("catalog asset inventory changed")
    by_id = _asset_map(config)
    if (
        by_id[CATALOG_IDS[0]]["expected_network_bytes"] != 9705600
        or by_id[CATALOG_IDS[0]]["expected_rows"] != 3747
        or by_id[CATALOG_IDS[1]]["expected_network_bytes"] != 23764095
        or by_id[CATALOG_IDS[1]]["expected_rows"] != 12247
        or by_id[CATALOG_IDS[1]]["expected_fits_member"] != "erass1cl_main_v3.2.fits"
        or sum(item["network_calls"] for item in assets) != 2
        or sum(item["expected_network_bytes"] for item in assets) != 33469695
    ):
        raise GravityClusterActErassOverlapExecutorError("catalog byte/row contract changed")
    projection = config["projection_contract"]
    if (
        tuple(projection[CATALOG_IDS[0]]) != ACT_COLUMNS
        or tuple(projection[CATALOG_IDS[1]]) != ERASS_COLUMNS
        or projection["column_tform_contract"] != PROJECTION_TFORM_CONTRACT
        or projection["scaling_and_null_rule"]
        != "Reject TSCALn, TZEROn, TNULLn and TDIMn for every allowlisted column; no scaling, integer-null sentinel, or array-shape semantics are frozen."
        or projection["row_layout_rule"]
        != "The sum of every declared TFORM width must equal NAXIS1 exactly; implicit or trailing row padding is forbidden."
    ):
        raise GravityClusterActErassOverlapExecutorError("projection allowlist changed")
    schema_sources = projection["schema_metadata_sources"]
    if not isinstance(schema_sources, list) or len(schema_sources) != 2:
        raise GravityClusterActErassOverlapExecutorError("schema metadata sources changed")
    for source in schema_sources:
        _strict(
            source,
            {
                "catalog_id",
                "retrieved_date_utc",
                "method",
                "url",
                "local_capture_path",
                "local_capture_sha256",
                "runtime_mismatch_action",
                "applicability",
            },
            "schema metadata source",
        )
    if (
        tuple(item["catalog_id"] for item in schema_sources) != CATALOG_IDS
        or any(item["retrieved_date_utc"] != "2026-08-29" for item in schema_sources)
        or any(
            item["method"] != "official_schema_metadata_page_only_no_catalog_rows_or_payload"
            for item in schema_sources
        )
        or any(item["local_capture_path"] is not None for item in schema_sources)
        or any(item["local_capture_sha256"] is not None for item in schema_sources)
        or any(
            item["runtime_mismatch_action"]
            != "FAIL_CLOSED_BEFORE_ROW_DECODE_IF_ANY_FROZEN_TFORM_OR_SEMANTIC_RULE_DIFFERS"
            for item in schema_sources
        )
        or tuple(item["url"] for item in schema_sources)
        != (
            "https://lambda.gsfc.nasa.gov/cgi-bin/fitsheader.cgi?fitsfile=/data/suborbital/ACT/actadv_dr6_cluster_cat/DR6_cluster-catalog_v1.0.fits",
            "https://erosita.mpe.mpg.de/dr1/AllSkySurveyData_dr1/Catalogues_dr1/BulbulE_DR1/erass1cl_primary_v3.2.html",
        )
    ):
        raise GravityClusterActErassOverlapExecutorError("schema metadata provenance changed")
    sanitized = config["sanitized_output_contract"]
    if (
        tuple(sanitized["ledger_fields"]) != LEDGER_FIELDS
        or sanitized["ledger_filename"] != "sanitized-overlap-ledger.jsonl"
        or sanitized["count_receipt_filename"] != "count-receipt.json"
        or sanitized["completion_marker_filename"] != "COMPLETE.json"
    ):
        raise GravityClusterActErassOverlapExecutorError("sanitized output contract changed")
    match = config["selection_match_and_exclusion_contract"]
    if (
        tuple(match["canonical_xcop_objects"]) != XCOP_OBJECTS
        or match["population_gate"]["confirmatory_target_clusters"] != 192
        or match["population_gate"]["underpowered_execution_floor_clusters"] != 120
        or match["cosmology"]["distance_integrator"] != "deterministic_Simpson_4096_even_intervals"
        or match["matching_engine"]
        != "exhaustive_all_pairs_exact_spherical_haversine_no_flat_RA_Dec_grid"
        or match["pair_comparison_ceiling"] != PAIR_COMPARISON_CEILING
        or match["global_erass_identity_rule"]
        != "In a first pass before projected-system, overlap-selection, unique-match, or ordinary ACT eligibility exits, every X-COP-named or X-COP-like ACT row with usable selected coordinates and 0.01<=z<=0.6 globally taints every DETUID in its complete spherical candidate set, including multiple-candidate and alias-ambiguous rows; only after that pass group ordinary unique matches by DETUID, propagate taint/exclusion, quarantine ambiguous-set taint and reuse, and count only untainted distinct DETUIDs."
        or match["xcop_candidate_taint_rule"]
        != "QUARANTINE_ALL_CANDIDATE_IDS before any ACT selection exit or unique-match adjudication; projected warnings or failed eROSITA overlap flags on a usable X-COP source row do not erase its candidate taint, and no later ordinary unique reuse of a tainted DETUID is eligible."
    ):
        raise GravityClusterActErassOverlapExecutorError("match or population contract changed")
    if any(config["preflight_access_state"].values()):
        raise GravityClusterActErassOverlapExecutorError("preflight access state changed")
    claims = config["claim_boundary"]
    if claims["executor_contract_frozen"] is not True or any(
        value for key, value in claims.items() if key != "executor_contract_frozen"
    ):
        raise GravityClusterActErassOverlapExecutorError("claim boundary overstated")
    for section, expected in SECTION_BINDINGS.items():
        if _sha(config[section]) != expected:
            raise GravityClusterActErassOverlapExecutorError(
                f"canonical nested section changed: {section}"
            )


def _validate_current_unauthorized(auth: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    _validate_authorization_shape(auth)
    if (
        auth["schema_version"] != AUTH_SCHEMA
        or auth["status"] != "UNAUTHORIZED_EXECUTOR_NOT_RUN"
        or auth["authorization_id"] != "act-dr6-erass1-overlap-executor-v2"
        or auth["authorization"] is not False
        or auth["run_id"] is not None
        or auth["authorized_by"] is not None
        or auth["approved_at_utc"] is not None
        or auth["approval_phrase"] is not None
    ):
        raise GravityClusterActErassOverlapExecutorError("current authorization identity changed")
    if any(
        auth["package_binding"][key] is not None
        for key in (
            "config_file_sha256",
            "module_file_sha256",
            "test_file_sha256",
            "preflight_receipt_file_sha256",
            "preflight_receipt_content_sha256",
        )
    ):
        raise GravityClusterActErassOverlapExecutorError("unauthorized package hashes populated")
    if auth["package_binding"]["parent_commit"] != config["parent_binding"]["commit"]:
        raise GravityClusterActErassOverlapExecutorError("unauthorized parent commit changed")
    if any(item["authorized"] for item in auth["catalog_authorizations"]):
        raise GravityClusterActErassOverlapExecutorError("catalog authorization became true")
    if auth["access_state"] != AUTH_ACCESS_STATE_ZERO:
        raise GravityClusterActErassOverlapExecutorError("unauthorized access state changed")
    if any(auth["claim_boundary"].values()):
        raise GravityClusterActErassOverlapExecutorError("unauthorized claim overstated")
    _validate_catalog_authorization_specs(auth, config, authorized=False)


def _validate_authorization_shape(auth: Mapping[str, Any]) -> None:
    _strict(
        auth,
        {
            "schema_version",
            "status",
            "authorization_id",
            "authorization",
            "run_id",
            "authorized_by",
            "approved_at_utc",
            "approval_phrase",
            "package_binding",
            "catalog_authorizations",
            "network_and_output_scope",
            "access_state",
            "future_authorized_state_requirements",
            "claim_boundary",
        },
        "authorization manifest",
    )
    _strict(
        auth["package_binding"],
        {
            "parent_commit",
            "config_file_sha256",
            "module_file_sha256",
            "test_file_sha256",
            "preflight_receipt_file_sha256",
            "preflight_receipt_content_sha256",
        },
        "authorization package binding",
    )
    _strict(
        auth["access_state"],
        set(AUTH_ACCESS_STATE_ZERO),
        "authorization access state",
    )
    if not isinstance(auth["catalog_authorizations"], list):
        raise GravityClusterActErassOverlapExecutorError("authorization catalogs changed")
    for item in auth["catalog_authorizations"]:
        _strict(
            item,
            {
                "catalog_id",
                "url",
                "expected_network_bytes",
                "maximum_rows",
                "permitted_columns",
                "authorized",
            },
            "catalog authorization",
        )


def _validate_catalog_authorization_specs(
    auth: Mapping[str, Any], config: Mapping[str, Any], *, authorized: bool
) -> None:
    catalogs = auth["catalog_authorizations"]
    if tuple(item["catalog_id"] for item in catalogs) != CATALOG_IDS:
        raise GravityClusterActErassOverlapExecutorError("authorization catalog inventory changed")
    by_id = _asset_map(config)
    for item in catalogs:
        catalog_id = item["catalog_id"]
        asset = by_id[catalog_id]
        expected_columns = config["projection_contract"][catalog_id]
        if item != {
            "catalog_id": catalog_id,
            "url": asset["url"],
            "expected_network_bytes": asset["expected_network_bytes"],
            "maximum_rows": asset["expected_rows"],
            "permitted_columns": expected_columns,
            "authorized": authorized,
        }:
            raise GravityClusterActErassOverlapExecutorError(
                f"authorization catalog specification changed: {catalog_id}"
            )


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _file_sha(root / CONFIG_PATH) != CONFIG_FILE_SHA256:
        raise GravityClusterActErassOverlapExecutorError("executor config hash changed")
    config = _read_json(root / CONFIG_PATH)
    validate_config(config)
    if _file_sha(root / TEST_PATH) != TEST_FILE_SHA256:
        raise GravityClusterActErassOverlapExecutorError("executor test binding changed")
    if _file_sha(root / CURRENT_AUTH_PATH) != CURRENT_AUTH_FILE_SHA256:
        raise GravityClusterActErassOverlapExecutorError("current authorization file changed")
    _validate_current_unauthorized(_read_json(root / CURRENT_AUTH_PATH), config)
    parent = config["parent_binding"]
    for path_key, sha_key in (
        ("config_path", "config_file_sha256"),
        ("module_path", "module_file_sha256"),
        ("test_path", "test_file_sha256"),
        ("authorization_path", "authorization_file_sha256"),
        ("receipt_path", "receipt_file_sha256"),
    ):
        if _file_sha(root / parent[path_key]) != parent[sha_key]:
            raise GravityClusterActErassOverlapExecutorError(
                f"committed parent binding changed: {path_key}"
            )
    parent_receipt = _read_json(root / parent["receipt_path"])
    if parent_receipt.get("content_sha256") != parent["receipt_content_sha256"]:
        raise GravityClusterActErassOverlapExecutorError("parent receipt content changed")
    return config


def build_preflight_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    body: dict[str, Any] = {
        "schema_version": PREFLIGHT_SCHEMA,
        "executor_id": config["executor_id"],
        "audit_cutoff": config["audit_cutoff"],
        "status": config["status"],
        "decision": "PREPARED_NOT_AUTHORIZED_NOT_EXECUTED",
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "module_path": MODULE_PATH.as_posix(),
            "module_file_sha256": _file_sha(root / MODULE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(root / TEST_PATH),
        },
        "parent_binding": config["parent_binding"],
        "current_authorization_binding": config["current_authorization_binding"],
        "future_execution_contract": config["future_execution_contract"],
        "catalog_assets": config["catalog_assets"],
        "projection_contract": config["projection_contract"],
        "selection_match_and_exclusion_contract": config["selection_match_and_exclusion_contract"],
        "sanitized_output_contract": config["sanitized_output_contract"],
        "failure_and_publication_contract": config["failure_and_publication_contract"],
        "access_state": config["preflight_access_state"],
        "claims": config["claim_boundary"],
        "counts": {
            "network_calls": 0,
            "files_downloaded": 0,
            "network_bytes_downloaded": 0,
            "catalog_rows_opened": 0,
            "forbidden_values_decoded_or_logged": 0,
            "sanitized_ledger_rows_emitted": 0,
            "scores_computed": 0,
            "model_or_paid_calls": 0,
        },
        "limitations": [
            "This receipt freezes an executor and authorization schema; it is not authorization and no catalog request has occurred.",
            "The raw publisher files contain scientific columns as opaque bytes, but a future authorized run keeps them temporary, never decodes forbidden values, and deletes them before publication.",
            "X-COP exclusion is exact-name/alias based; coordinate-only exclusion remains unavailable without a separately frozen X-COP coordinate ledger.",
            "Even a post-X-COP count at or above 192 is only a catalog population upper bound and does not establish complete thermodynamic/profile packets.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_preflight_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    content_sha = body.pop("content_sha256", None)
    if content_sha != _sha(body):
        raise GravityClusterActErassOverlapExecutorError("preflight receipt content hash changed")
    if receipt != build_preflight_receipt(root):
        raise GravityClusterActErassOverlapExecutorError(
            "preflight receipt differs from frozen executor"
        )


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_tree_verified(path: Path, *, label: str) -> None:
    if not path.exists():
        return
    try:
        shutil.rmtree(path)
    except Exception as exc:
        raise GravityClusterActErassOverlapExecutorError(
            f"cleanup failed for {label}; path may remain"
        ) from exc
    if path.exists():
        raise GravityClusterActErassOverlapExecutorError(
            f"cleanup postcondition failed for {label}; path remains"
        )


def _atomic_write_no_replace(path: Path, payload: bytes) -> None:
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
        except FileExistsError as exc:
            if path.read_bytes() != payload:
                raise GravityClusterActErassOverlapExecutorError(
                    f"refusing to replace existing output file: {path.name}"
                ) from exc
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_preflight_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_preflight_receipt(root)
    payload = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    path = root / PREFLIGHT_PATH
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityClusterActErassOverlapExecutorError(
                "refusing to replace a different executor preflight receipt"
            )
    else:
        _atomic_write_no_replace(path, payload)
    validate_preflight_receipt(_read_json(path), root)
    return path


def _authorized_manifest_expected_package(root: Path, receipt: Mapping[str, Any]) -> dict[str, str]:
    return {
        "parent_commit": "1ce8c666ecb45093738cf80a1fcde90f1f11f1ba",
        "config_file_sha256": _file_sha(root / CONFIG_PATH),
        "module_file_sha256": _file_sha(root / MODULE_PATH),
        "test_file_sha256": _file_sha(root / TEST_PATH),
        "preflight_receipt_file_sha256": _file_sha(root / PREFLIGHT_PATH),
        "preflight_receipt_content_sha256": receipt["content_sha256"],
    }


def validate_authorized_manifest(
    auth: Mapping[str, Any], config: Mapping[str, Any], root: Path
) -> None:
    _validate_authorization_shape(auth)
    receipt = _read_json(root / PREFLIGHT_PATH)
    validate_preflight_receipt(receipt, root)
    if (
        auth["schema_version"] != AUTH_SCHEMA
        or auth["status"] != "AUTHORIZED_CATALOG_ONLY_EXECUTION"
        or auth["authorization_id"] != "act-dr6-erass1-overlap-executor-v2"
        or auth["authorization"] is not True
        or auth["run_id"] != RUN_ID
        or not isinstance(auth["authorized_by"], str)
        or not auth["authorized_by"].strip()
        or not isinstance(auth["approved_at_utc"], str)
        or auth["approval_phrase"] != AUTHORIZATION_PHRASE
        or re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", auth["approved_at_utc"]) is None
    ):
        raise GravityClusterActErassOverlapExecutorError("exact authorized identity is absent")
    try:
        datetime.fromisoformat(auth["approved_at_utc"])
    except ValueError as exc:
        raise GravityClusterActErassOverlapExecutorError(
            "authorized approval timestamp is invalid"
        ) from exc
    if auth["package_binding"] != _authorized_manifest_expected_package(root, receipt):
        raise GravityClusterActErassOverlapExecutorError("authorized package binding changed")
    _validate_catalog_authorization_specs(auth, config, authorized=True)
    current = _read_json(root / CURRENT_AUTH_PATH)
    if auth["network_and_output_scope"] != current["network_and_output_scope"]:
        raise GravityClusterActErassOverlapExecutorError("authorized network/output scope changed")
    if auth["access_state"] != AUTH_ACCESS_STATE_ZERO:
        raise GravityClusterActErassOverlapExecutorError(
            "authorized chronology is not zero at entry"
        )
    if (
        auth["future_authorized_state_requirements"]
        != current["future_authorized_state_requirements"]
    ):
        raise GravityClusterActErassOverlapExecutorError("authorized requirements changed")
    expected_claims = dict(current["claim_boundary"])
    expected_claims["authorized_successor_ready_to_execute"] = True
    if auth["claim_boundary"] != expected_claims:
        raise GravityClusterActErassOverlapExecutorError("authorized claim boundary changed")


def _parse_fits_value(raw: str) -> Any:
    text = raw.strip()
    if text.startswith("'"):
        end = text.rfind("'")
        return text[1:end].replace("''", "'").rstrip()
    text = text.split("/", 1)[0].strip()
    if text == "T":
        return True
    if text == "F":
        return False
    if text == "":
        return None
    try:
        if any(token in text for token in (".", "E", "D")):
            return float(text.replace("D", "E"))
        return int(text)
    except ValueError:
        return text


def _read_fits_header(handle: BinaryIO) -> tuple[dict[str, Any], int]:
    cards: list[bytes] = []
    blocks = 0
    while True:
        block = handle.read(2880)
        if len(block) != 2880:
            raise GravityClusterActErassOverlapExecutorError("truncated FITS header")
        blocks += 1
        block_cards = [block[index : index + 80] for index in range(0, 2880, 80)]
        cards.extend(block_cards)
        if any(card[:8].decode("ascii", "strict").strip() == "END" for card in block_cards):
            break
    header: dict[str, Any] = {}
    structural_keys: set[str] = set()
    for card in cards:
        key = card[:8].decode("ascii", "strict").strip()
        if key == "END":
            break
        if card[8:10] == b"= " and key:
            is_structural = (
                key
                in {
                    "SIMPLE",
                    "XTENSION",
                    "BITPIX",
                    "NAXIS",
                    "PCOUNT",
                    "GCOUNT",
                    "TFIELDS",
                    "THEAP",
                }
                or re.fullmatch(r"NAXIS\d+", key) is not None
                or re.fullmatch(
                    r"(?:TTYPE|TFORM|TUNIT|TNULL|TSCAL|TZERO|TDISP|TDIM|TBCOL)\d+",
                    key,
                )
                is not None
            )
            if is_structural and key in structural_keys:
                raise GravityClusterActErassOverlapExecutorError(
                    f"duplicate FITS structural card: {key}"
                )
            if is_structural:
                structural_keys.add(key)
            header[key] = _parse_fits_value(card[10:].decode("ascii", "strict"))
    return header, blocks * 2880


def _hdu_data_bytes(header: Mapping[str, Any]) -> int:
    naxis = int(header.get("NAXIS", 0))
    if header.get("XTENSION") == "BINTABLE":
        return int(header["NAXIS1"]) * int(header["NAXIS2"]) + int(header.get("PCOUNT", 0))
    if naxis == 0:
        return 0
    elements = 1
    for axis in range(1, naxis + 1):
        elements *= int(header[f"NAXIS{axis}"])
    return abs(int(header["BITPIX"])) // 8 * elements * int(header.get("GCOUNT", 1)) + int(
        header.get("PCOUNT", 0)
    )


def _find_binary_table(handle: BinaryIO) -> tuple[dict[str, Any], int]:
    while True:
        header, _ = _read_fits_header(handle)
        data_start = handle.tell()
        if header.get("XTENSION") == "BINTABLE":
            return header, data_start
        size = _hdu_data_bytes(header)
        handle.seek(((size + 2879) // 2880) * 2880, os.SEEK_CUR)


def _tform_layout(tform: str) -> tuple[int, str, int]:
    fixed = re.fullmatch(r"\s*(\d*)([LXBIJKAEDCM])\s*", tform)
    variable = re.fullmatch(r"\s*(\d*)([PQ])([LXBIJKAEDCM])(?:\(\d+\))?\s*", tform)
    match = fixed or variable
    if match is None:
        raise GravityClusterActErassOverlapExecutorError(
            "malformed or unsupported FITS column format"
        )
    repeat = int(match.group(1) or "1")
    if repeat <= 0:
        raise GravityClusterActErassOverlapExecutorError(
            "malformed or unsupported FITS column format"
        )
    code = match.group(2)
    unit_bytes = {
        "L": 1,
        "B": 1,
        "I": 2,
        "J": 4,
        "K": 8,
        "A": 1,
        "E": 4,
        "D": 8,
        "C": 8,
        "M": 16,
        "P": 8,
        "Q": 16,
    }
    width = (repeat + 7) // 8 if code == "X" else repeat * unit_bytes[code]
    return repeat, code, width


def _decode_fits_cell(
    raw: bytes,
    *,
    repeat: int,
    code: str,
    scale: float,
    zero: float,
    null: int | None,
) -> Any:
    if code == "A":
        return raw.decode("ascii", "replace").rstrip(" \x00")
    if code == "L":
        logical_values: list[bool] = []
        for byte in raw[:repeat]:
            if byte == ord("T"):
                logical_values.append(True)
            elif byte == ord("F"):
                logical_values.append(False)
            else:
                raise GravityClusterActErassOverlapExecutorError("invalid FITS logical flag")
        values = tuple(logical_values)
    elif code == "B":
        values = tuple(raw[:repeat])
    elif code in "IJKED":
        fmt = {"I": "h", "J": "i", "K": "q", "E": "f", "D": "d"}[code]
        values = struct.unpack(f">{repeat}{fmt}", raw)
    else:
        raise GravityClusterActErassOverlapExecutorError(
            "allowlisted FITS column uses an unsupported decoded format"
        )
    converted: list[Any] = []
    for value in values:
        if null is not None and isinstance(value, int) and value == null:
            converted.append(None)
        elif isinstance(value, (int, float)) and code != "L":
            converted.append(value * scale + zero)
        else:
            converted.append(value)
    return converted[0] if repeat == 1 else tuple(converted)


def read_fits_projection(
    path: Path,
    allowed_columns: Sequence[str],
    *,
    expected_tforms: Mapping[str, str],
    expected_rows: int,
    maximum_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    try:
        return _read_fits_projection_checked(
            path,
            allowed_columns,
            expected_tforms=expected_tforms,
            expected_rows=expected_rows,
            maximum_rows=maximum_rows,
        )
    except GravityClusterActErassOverlapExecutorError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError, struct.error):
        raise GravityClusterActErassOverlapExecutorError(
            "FITS schema/type/coercion validation failed without exposing values"
        )


def _read_fits_projection_checked(
    path: Path,
    allowed_columns: Sequence[str],
    *,
    expected_tforms: Mapping[str, str],
    expected_rows: int,
    maximum_rows: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if set(expected_tforms) != set(allowed_columns):
        raise GravityClusterActErassOverlapExecutorError(
            "FITS projection TFORM contract keys changed"
        )
    if expected_rows > maximum_rows:
        raise GravityClusterActErassOverlapExecutorError("expected FITS rows exceed ceiling")
    with path.open("rb") as handle:
        header, data_start = _find_binary_table(handle)
        rows = int(header["NAXIS2"])
        row_bytes = int(header["NAXIS1"])
        fields = int(header["TFIELDS"])
        if rows != expected_rows or rows > maximum_rows:
            raise GravityClusterActErassOverlapExecutorError("FITS row count changed")
        layout: dict[str, dict[str, Any]] = {}
        offset = 0
        for index in range(1, fields + 1):
            name = str(header[f"TTYPE{index}"])
            repeat, code, width = _tform_layout(str(header[f"TFORM{index}"]))
            folded_name = name.casefold()
            if folded_name in layout:
                raise GravityClusterActErassOverlapExecutorError(
                    "duplicate case-folded FITS column name"
                )
            null = header.get(f"TNULL{index}")
            if null is not None and (code not in {"B", "I", "J", "K"} or type(null) is not int):
                raise GravityClusterActErassOverlapExecutorError(
                    "invalid FITS integer-null declaration"
                )
            layout[folded_name] = {
                "name": name,
                "offset": offset,
                "repeat": repeat,
                "code": code,
                "width": width,
                "raw_tform": str(header[f"TFORM{index}"]).strip().upper(),
                "scale_card_present": f"TSCAL{index}" in header,
                "zero_card_present": f"TZERO{index}" in header,
                "null_card_present": f"TNULL{index}" in header,
                "dim_card_present": f"TDIM{index}" in header,
                "null": null,
            }
            offset += width
        if offset != row_bytes:
            raise GravityClusterActErassOverlapExecutorError(
                "FITS column layout does not exactly equal row width"
            )
        selected: list[dict[str, Any]] = []
        for column in allowed_columns:
            entry = layout.get(column.casefold())
            if entry is None:
                raise GravityClusterActErassOverlapExecutorError(
                    f"required FITS column missing: {column}"
                )
            if entry["name"] != column:
                raise GravityClusterActErassOverlapExecutorError(
                    f"required FITS column case changed: {column}"
                )
            expected_tform = expected_tforms[column].strip().upper()
            expected_repeat, expected_code, _ = _tform_layout(expected_tform)
            if (
                entry["raw_tform"] != expected_tform
                or entry["repeat"] != expected_repeat
                or entry["code"] != expected_code
            ):
                raise GravityClusterActErassOverlapExecutorError(
                    f"allowlisted FITS column TFORM changed: {column}"
                )
            if any(
                entry[key]
                for key in (
                    "scale_card_present",
                    "zero_card_present",
                    "null_card_present",
                    "dim_card_present",
                )
            ):
                raise GravityClusterActErassOverlapExecutorError(
                    f"allowlisted FITS column scaling/null semantics changed: {column}"
                )
            if entry["code"] in {"X", "C", "M", "P", "Q"}:
                raise GravityClusterActErassOverlapExecutorError(
                    f"allowlisted FITS column has unsupported format: {column}"
                )
            selected.append(entry)
        projected_rows: list[dict[str, Any]] = []
        for row_index in range(rows):
            row: dict[str, Any] = {}
            for canonical_name, entry in zip(allowed_columns, selected, strict=True):
                handle.seek(data_start + row_index * row_bytes + entry["offset"])
                raw = handle.read(entry["width"])
                if len(raw) != entry["width"]:
                    raise GravityClusterActErassOverlapExecutorError(
                        f"truncated FITS cell at row {row_index}, column {canonical_name}"
                    )
                row[canonical_name] = _decode_fits_cell(
                    raw,
                    repeat=entry["repeat"],
                    code=entry["code"],
                    scale=1.0,
                    zero=0.0,
                    null=entry["null"],
                )
            projected_rows.append(row)
    return projected_rows, {
        "rows": rows,
        "row_bytes": row_bytes,
        "fields_in_source_schema": fields,
        "fields_decoded": len(allowed_columns),
    }


def extract_single_fits_member(
    archive_path: Path, destination: Path, *, expected_basename: str, maximum_bytes: int
) -> dict[str, Any]:
    with tarfile.open(archive_path, mode="r:gz") as archive:
        regular = []
        for member in archive.getmembers():
            posix_path = PurePosixPath(member.name)
            windows_path = PureWindowsPath(member.name)
            if (
                not member.name
                or "\x00" in member.name
                or posix_path.is_absolute()
                or windows_path.is_absolute()
                or bool(windows_path.drive)
                or ".." in posix_path.parts
                or ".." in windows_path.parts
            ):
                raise GravityClusterActErassOverlapExecutorError(
                    "archive member path safety failed"
                )
            if member.isdir():
                continue
            if not member.isfile():
                raise GravityClusterActErassOverlapExecutorError(
                    "archive contains a prohibited special member"
                )
            regular.append(member)
        if len(regular) != 1 or Path(regular[0].name).name != expected_basename:
            raise GravityClusterActErassOverlapExecutorError("archive member inventory changed")
        member = regular[0]
        if member.size > maximum_bytes:
            raise GravityClusterActErassOverlapExecutorError("archive member safety ceiling failed")
        source = archive.extractfile(member)
        if source is None:
            raise GravityClusterActErassOverlapExecutorError("archive member is unreadable")
        digest = hashlib.sha256()
        written = 0
        with destination.open("xb") as output:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > maximum_bytes:
                    raise GravityClusterActErassOverlapExecutorError(
                        "archive decompressed byte ceiling exceeded"
                    )
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != member.size:
            destination.unlink(missing_ok=True)
            raise GravityClusterActErassOverlapExecutorError("archive member size changed")
    return {"member": expected_basename, "bytes": written, "sha256": digest.hexdigest()}


class _GetBudget:
    def __init__(self, ceiling: int) -> None:
        self.ceiling = ceiling
        self.calls = 0

    def claim(self) -> None:
        if self.calls >= self.ceiling:
            raise GravityClusterActErassOverlapExecutorError("GET call ceiling exceeded")
        self.calls += 1

    def verify_exact(self) -> None:
        if self.calls != self.ceiling:
            raise GravityClusterActErassOverlapExecutorError("exact GET call count not reached")


def _make_no_redirect_handler() -> Any:
    from urllib.error import HTTPError
    from urllib.request import HTTPRedirectHandler

    class _RejectRedirects(HTTPRedirectHandler):
        def redirect_request(self, request, file_pointer, code, message, headers, new_url):
            del file_pointer, headers, new_url
            raise HTTPError(request.full_url, code, f"redirect prohibited: {message}", None, None)

    return _RejectRedirects()


def _download_exact(
    asset: Mapping[str, Any],
    destination: Path,
    *,
    opener: Callable[..., Any] | None = None,
    budget: _GetBudget | None = None,
) -> dict[str, Any]:
    if budget is None:
        budget = _GetBudget(1)
    if opener is None:
        from urllib.request import Request, build_opener

        request: Any = Request(asset["url"], headers={"User-Agent": "Invariant/ACT-eRASS-v2"})
        opener = build_opener(_make_no_redirect_handler()).open
    else:
        request = asset["url"]
    response = None
    digest = hashlib.sha256()
    written = 0
    try:
        budget.claim()
        response = opener(request, timeout=120)
        if hasattr(response, "getcode") and response.getcode() != 200:
            raise GravityClusterActErassOverlapExecutorError("download HTTP status changed")
        final_url = response.geturl()
        if final_url != asset["url"]:
            raise GravityClusterActErassOverlapExecutorError("download redirect changed exact URL")
        content_length = response.headers.get("Content-Length")
        if content_length is not None and int(content_length) != asset["expected_network_bytes"]:
            raise GravityClusterActErassOverlapExecutorError("download Content-Length changed")
        with destination.open("xb") as output:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                written += len(chunk)
                if written > asset["expected_network_bytes"]:
                    raise GravityClusterActErassOverlapExecutorError(
                        "download byte ceiling exceeded"
                    )
                output.write(chunk)
                digest.update(chunk)
            output.flush()
            os.fsync(output.fileno())
        if written != asset["expected_network_bytes"]:
            raise GravityClusterActErassOverlapExecutorError("download byte count changed")
        return {"bytes": written, "sha256": digest.hexdigest(), "url": asset["url"]}
    except Exception:
        destination.unlink(missing_ok=True)
        raise
    finally:
        if response is not None:
            response.close()


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _truth(value: Any) -> bool:
    return value is True or value == 1 or str(value).strip().upper() in {"T", "TRUE", "YES"}


def _select_position(row: Mapping[str, Any], catalog: str) -> tuple[float, float] | None:
    choices = (
        (("opt_RADeg", "opt_decDeg"), ("RADeg", "decDeg"))
        if catalog == "ACT"
        else (("RA_XFIT", "DEC_XFIT"), ("RA", "DEC"))
    )
    for ra_key, dec_key in choices:
        ra, dec = row.get(ra_key), row.get(dec_key)
        if _finite(ra) and _finite(dec) and not (float(ra) == 0.0 and float(dec) == 0.0):
            return float(ra) % 360.0, float(dec)
    return None


def _canonical_name(value: Any) -> str:
    text = unicodedata.normalize("NFKD", str(value or "")).encode("ascii", "ignore").decode()
    token = re.sub(r"[^A-Z0-9]", "", text.upper())
    match = re.fullmatch(r"(?:ABELL|ACO)(\d+)", token)
    if match:
        return f"A{int(match.group(1))}"
    if token.startswith("ZWCL1215") or token == "ZW1215":
        return "ZW1215"
    if token.startswith(("RXCJ1825", "RXC1825")):
        return "RXC1825"
    return token


def _name_tokens(*values: Any) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        for part in re.split(r"[,;/|]", str(value or "")):
            token = _canonical_name(part)
            if token:
                tokens.add(token)
    return tokens


def _xcop_identity(*values: Any) -> tuple[str | None, bool]:
    tokens = _name_tokens(*values)
    matches = sorted(set(XCOP_OBJECTS).intersection(tokens))
    if len(matches) == 1:
        return matches[0], False
    if len(matches) > 1:
        return None, True
    partial = [
        token for token in tokens if any(item in token or token in item for item in XCOP_OBJECTS)
    ]
    return (None, True) if partial else (None, False)


def _angular_separation_arcmin(a: tuple[float, float], b: tuple[float, float]) -> float:
    ra1, dec1 = map(math.radians, a)
    ra2, dec2 = map(math.radians, b)
    delta_ra = (ra2 - ra1 + math.pi) % (2.0 * math.pi) - math.pi
    delta_dec = dec2 - dec1
    haversine = (
        math.sin(delta_dec / 2.0) ** 2
        + math.cos(dec1) * math.cos(dec2) * math.sin(delta_ra / 2.0) ** 2
    )
    angle = 2.0 * math.asin(math.sqrt(max(0.0, min(1.0, haversine))))
    return math.degrees(angle) * 60.0


def _angular_diameter_distance_mpc(z: float) -> float:
    intervals = 4096
    step = z / intervals
    total = 0.0
    for index in range(intervals + 1):
        sample = index * step
        inverse_e = 1.0 / math.sqrt(0.3 * (1.0 + sample) ** 3 + 0.7)
        weight = 1 if index in {0, intervals} else (4 if index % 2 else 2)
        total += weight * inverse_e
    comoving = (299792.458 / 70.0) * step * total / 3.0
    return comoving / (1.0 + z)


def _match_radius_arcmin(z: float) -> float:
    distance = _angular_diameter_distance_mpc(z)
    return max(1.22, math.degrees(0.5 / distance) * 60.0)


def _empty_ledger(act_name: str) -> dict[str, Any]:
    return {
        "canonical_join_id": hashlib.sha256(f"ACT|{act_name}".encode()).hexdigest()[:24],
        "act_name": act_name,
        "erass_detuid": None,
        "erass_name": None,
        "match_state": "QUARANTINED",
        "match_method": None,
        "candidate_ids": [],
        "angular_separation_arcmin": None,
        "absolute_redshift_difference": None,
        "xcop_identity": None,
        "xcop_excluded": False,
        "eligible_catalog_overlap": False,
        "quarantine_reason": None,
    }


def match_catalogs(
    act_rows: Sequence[Mapping[str, Any]], erass_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    try:
        return _match_catalogs_checked(act_rows, erass_rows)
    except GravityClusterActErassOverlapExecutorError:
        raise
    except (KeyError, TypeError, ValueError, OverflowError, UnicodeError):
        raise GravityClusterActErassOverlapExecutorError(
            "private catalog row failed frozen type/coercion validation without exposing values"
        )


def _match_catalogs_checked(
    act_rows: Sequence[Mapping[str, Any]], erass_rows: Sequence[Mapping[str, Any]]
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    erass_eligible: list[tuple[Mapping[str, Any], tuple[float, float], float]] = []
    erass_rejected = 0
    for row in erass_rows:
        position = _select_position(row, "ERASS")
        redshift = row.get("BEST_Z")
        if (
            not str(row.get("DETUID") or "").strip()
            or not str(row.get("NAME") or "").strip()
            or position is None
            or not _finite(row.get("EXT_LIKE"))
            or float(row["EXT_LIKE"]) <= 3.0
            or not _finite(redshift)
            or not 0.01 <= float(redshift) <= 0.6
        ):
            erass_rejected += 1
            continue
        erass_eligible.append((row, position, float(redshift)))
    ledger: list[dict[str, Any]] = []
    erass_match_groups: dict[str, list[int]] = defaultdict(list)
    xcop_evidence: dict[int, tuple[str | None, bool]] = {}
    xcop_candidate_taint: dict[str, dict[str, Any]] = {}
    selection_rejected = 0
    projected = 0
    pair_comparisons = 0
    xcop_candidate_source_act_rows = 0

    def scan_candidates(position: tuple[float, float], redshift: float) -> list[tuple[int, float]]:
        nonlocal pair_comparisons
        radius = _match_radius_arcmin(redshift)
        candidates: list[tuple[int, float]] = []
        for index, (_, erass_position, _) in enumerate(erass_eligible):
            pair_comparisons += 1
            if pair_comparisons > PAIR_COMPARISON_CEILING:
                raise GravityClusterActErassOverlapExecutorError(
                    "frozen spherical pair-comparison ceiling exceeded"
                )
            separation = _angular_separation_arcmin(position, erass_position)
            if separation <= radius:
                candidates.append((index, separation))
        candidates.sort(key=lambda value: (value[1], str(erass_eligible[value[0]][0]["DETUID"])))
        return candidates

    # This pass is deliberately earlier than every ACT selection or projected-system
    # exit.  A recognized X-COP row with usable matching coordinates/redshift taints
    # its complete candidate set even when that source row is later quarantined.
    xcop_candidates_by_act_index: dict[int, list[tuple[int, float]]] = {}
    xcop_identity_by_act_index: dict[int, tuple[str | None, bool]] = {}
    for act_index, act in enumerate(act_rows):
        act_xcop, act_xcop_ambiguous = _xcop_identity(act.get("name"))
        if act_xcop is None and not act_xcop_ambiguous:
            continue
        position = _select_position(act, "ACT")
        redshift = act.get("redshift")
        if position is None or not _finite(redshift) or not 0.01 <= float(redshift) <= 0.6:
            continue
        candidates = scan_candidates(position, float(redshift))
        xcop_candidates_by_act_index[act_index] = candidates
        xcop_identity_by_act_index[act_index] = (act_xcop, act_xcop_ambiguous)
        if not candidates:
            continue
        xcop_candidate_source_act_rows += 1
        for candidate_index, _ in candidates:
            candidate_id = str(erass_eligible[candidate_index][0]["DETUID"])
            taint = xcop_candidate_taint.setdefault(
                candidate_id,
                {
                    "identities": set(),
                    "alias_ambiguous": False,
                    "ambiguous_candidate_source": False,
                    "source_rows": 0,
                },
            )
            if act_xcop is not None:
                taint["identities"].add(act_xcop)
            taint["alias_ambiguous"] = taint["alias_ambiguous"] or act_xcop_ambiguous
            taint["ambiguous_candidate_source"] = (
                taint["ambiguous_candidate_source"] or len(candidates) != 1
            )
            taint["source_rows"] += 1

    for act_index, act in enumerate(act_rows):
        act_name = str(act.get("name") or "").strip()
        item = _empty_ledger(act_name)
        xcop_candidates = xcop_candidates_by_act_index.get(act_index)
        if xcop_candidates:
            act_xcop, _ = xcop_identity_by_act_index[act_index]
            item["candidate_ids"] = [
                str(erass_eligible[index][0]["DETUID"]) for index, _ in xcop_candidates
            ]
            item["xcop_identity"] = act_xcop
            item["xcop_excluded"] = True
        if (
            not _finite(act.get("fixed_SNR"))
            or float(act["fixed_SNR"]) <= 5.5
            or int(act.get("flags") or 0) != 0
            or not _truth(act.get("footprint_Legacy"))
        ):
            raise GravityClusterActErassOverlapExecutorError(
                "ACT Legacy base-selection invariant changed"
            )
        position = _select_position(act, "ACT")
        redshift = act.get("redshift")
        if (
            not act_name
            or not _truth(act.get("footprint_eROSITADe"))
            or not _truth(act.get("eRASS1CL"))
            or position is None
            or not _finite(redshift)
            or not 0.01 <= float(redshift) <= 0.6
        ):
            selection_rejected += 1
            item["quarantine_reason"] = "ACT_SELECTION_NOT_ELIGIBLE"
            ledger.append(item)
            continue
        if "possible projected system" in str(act.get("warnings") or "").casefold():
            projected += 1
            item["quarantine_reason"] = "POSSIBLE_PROJECTED_SYSTEM"
            ledger.append(item)
            continue
        candidates = (
            xcop_candidates
            if xcop_candidates is not None
            else scan_candidates(position, float(redshift))
        )
        item["candidate_ids"] = [str(erass_eligible[index][0]["DETUID"]) for index, _ in candidates]
        if not candidates:
            item["quarantine_reason"] = "NO_IN_RADIUS_ERASS_CANDIDATE"
            ledger.append(item)
            continue
        if len(candidates) != 1:
            item["quarantine_reason"] = "MULTIPLE_IN_RADIUS_ERASS_CANDIDATES"
            ledger.append(item)
            continue
        erass_index, separation = candidates[0]
        erass, _, erass_z = erass_eligible[erass_index]
        erass_id = str(erass["DETUID"])
        item.update(
            {
                "canonical_join_id": hashlib.sha256(
                    f"ACT|{act_name}|ERASS|{erass_id}".encode()
                ).hexdigest()[:24],
                "erass_detuid": erass_id,
                "erass_name": str(erass["NAME"]),
                "match_state": "MATCHED",
                "match_method": "UNIQUE_POSITIONAL_IN_PUBLISHER_RADIUS",
                "angular_separation_arcmin": round(separation, 12),
                "absolute_redshift_difference": round(abs(float(redshift) - erass_z), 12),
                "quarantine_reason": None,
            }
        )
        ledger_index = len(ledger)
        xcop_evidence[ledger_index] = _xcop_identity(
            act_name, erass["NAME"], erass.get("MATCH_NAME")
        )
        erass_match_groups[erass_id].append(ledger_index)
        ledger.append(item)

    for indices in erass_match_groups.values():
        erass_id = str(ledger[indices[0]]["erass_detuid"])
        taint = xcop_candidate_taint.get(erass_id)
        identities = {
            xcop_evidence[index][0] for index in indices if xcop_evidence[index][0] is not None
        }
        if taint is not None:
            identities.update(taint["identities"])
        ambiguous = (
            any(xcop_evidence[index][1] for index in indices)
            or bool(taint and taint["alias_ambiguous"])
            or len(identities) > 1
        )
        propagated_xcop = next(iter(identities)) if len(identities) == 1 else None
        if ambiguous:
            for index in indices:
                ledger[index]["xcop_identity"] = propagated_xcop
                ledger[index]["xcop_excluded"] = taint is not None or propagated_xcop is not None
                ledger[index]["match_state"] = "QUARANTINED"
                ledger[index]["eligible_catalog_overlap"] = False
                ledger[index]["quarantine_reason"] = "AMBIGUOUS_XCOP_CANDIDATE_TAINT"
            continue
        if propagated_xcop is not None:
            for index in indices:
                ledger[index]["xcop_identity"] = propagated_xcop
                ledger[index]["xcop_excluded"] = True
                ledger[index]["eligible_catalog_overlap"] = False
                if taint is not None and taint["ambiguous_candidate_source"]:
                    ledger[index]["match_state"] = "QUARANTINED"
                    ledger[index]["quarantine_reason"] = (
                        "XCOP_CANDIDATE_TAINT_FROM_AMBIGUOUS_ACT_ROW"
                    )
                elif len(indices) == 1:
                    ledger[index]["match_state"] = "MATCHED_XCOP_EXCLUDED"
                else:
                    ledger[index]["match_state"] = "QUARANTINED"
                    ledger[index]["quarantine_reason"] = (
                        "ERASS_CANDIDATE_REUSED_BY_MULTIPLE_ACT_ROWS"
                    )
            continue
        if len(indices) > 1:
            for index in indices:
                ledger[index]["match_state"] = "QUARANTINED"
                ledger[index]["eligible_catalog_overlap"] = False
                ledger[index]["quarantine_reason"] = "ERASS_CANDIDATE_REUSED_BY_MULTIPLE_ACT_ROWS"
            continue
        ledger[indices[0]]["eligible_catalog_overlap"] = True
    for item in ledger:
        if tuple(item) != LEDGER_FIELDS:
            raise GravityClusterActErassOverlapExecutorError("sanitized ledger schema changed")
    eligible_distinct = sum(
        any(ledger[index]["eligible_catalog_overlap"] for index in indices)
        for indices in erass_match_groups.values()
    )
    xcop_excluded_distinct_ids = set(xcop_candidate_taint)
    xcop_excluded_distinct_ids.update(
        erass_id
        for erass_id, indices in erass_match_groups.items()
        if any(ledger[index]["xcop_excluded"] for index in indices)
    )
    counts = {
        "act_rows": len(act_rows),
        "erass_rows": len(erass_rows),
        "spherical_pair_comparisons": pair_comparisons,
        "act_selection_rejected": selection_rejected,
        "projected_system_quarantines": projected,
        "erass_selection_rejected": erass_rejected,
        "unique_positional_match_rows": sum(
            len(indices) for indices in erass_match_groups.values()
        ),
        "distinct_matched_erass_objects": len(erass_match_groups),
        "reused_erass_groups": sum(len(indices) > 1 for indices in erass_match_groups.values()),
        "reused_act_rows": sum(
            len(indices) for indices in erass_match_groups.values() if len(indices) > 1
        ),
        "matched_rows": sum(item["match_state"].startswith("MATCHED") for item in ledger),
        "quarantined_rows": sum(item["match_state"] == "QUARANTINED" for item in ledger),
        "xcop_excluded_rows": sum(item["xcop_excluded"] for item in ledger),
        "xcop_excluded_distinct_erass_objects": len(xcop_excluded_distinct_ids),
        "xcop_candidate_taint_source_act_rows": xcop_candidate_source_act_rows,
        "xcop_candidate_taint_edges": sum(
            int(item["source_rows"]) for item in xcop_candidate_taint.values()
        ),
        "xcop_candidate_tainted_distinct_erass_objects": len(xcop_candidate_taint),
        "xcop_candidate_tainted_by_ambiguous_sets": sum(
            bool(item["ambiguous_candidate_source"]) for item in xcop_candidate_taint.values()
        ),
        "eligible_distinct_erass_objects": eligible_distinct,
        "post_xcop_catalog_upper_bound": eligible_distinct,
    }
    return ledger, counts


def _population_classification(count: int) -> str:
    if count >= 192:
        return "CATALOG_UPPER_BOUND_SUFFICIENT_PROFILE_READINESS_NOT_ESTABLISHED"
    if count >= 120:
        return "UNDERPOWERED_EXPLORATORY_ONLY"
    return "PRIMARY_TRIAL_NOT_OPENED"


def _write_execution_outputs(
    output_dir: Path,
    *,
    source_receipts: Mapping[str, Any],
    schema_receipts: Mapping[str, Any],
    ledger: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    authorization_sha256: str,
) -> None:
    ledger_payload = b"".join(_canonical_bytes(item) for item in ledger)
    ledger_path = output_dir / "sanitized-overlap-ledger.jsonl"
    _atomic_write_no_replace(ledger_path, ledger_payload)
    receipt_body: dict[str, Any] = {
        "schema_version": RESULT_SCHEMA,
        "status": "CATALOG_OVERLAP_EXECUTED_NO_SCIENTIFIC_SCORING",
        "authorization_file_sha256": authorization_sha256,
        "source_file_receipts": source_receipts,
        "schema_receipts": schema_receipts,
        "counts": counts,
        "post_xcop_catalog_upper_bound": counts["post_xcop_catalog_upper_bound"],
        "population_gate_classification": _population_classification(
            counts["post_xcop_catalog_upper_bound"]
        ),
        "ledger_binding": {
            "filename": ledger_path.name,
            "rows": len(ledger),
            "file_sha256": _file_sha(ledger_path),
        },
        "access_accounting": {
            "network_calls": 2,
            "network_bytes": 33469695,
            "catalog_rows_opened": 15994,
            "forbidden_values_decoded_or_logged": 0,
            "scores_computed": 0,
            "model_or_paid_calls": 0,
            "raw_source_files_retained": 0,
        },
        "claim_ceiling": "Version-bound catalog-overlap upper bound and exact-name/alias X-COP exclusion audit only; not profile completeness, independent replication readiness, physical-model evidence, or CP3/CP7 completion.",
        "limitations": [
            "X-COP coordinate-only exclusion was not performed because no separate frozen X-COP coordinate ledger is bound.",
            "At least 192 catalog matches is necessary but insufficient for a 192-object complete-profile confirmation packet.",
            "No map, profile, thermodynamic, lensing, inferred-mass, residual, candidate prediction, or score value was decoded or emitted.",
        ],
    }
    receipt = {**receipt_body, "content_sha256": _sha(receipt_body)}
    receipt_payload = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    receipt_path = output_dir / "count-receipt.json"
    _atomic_write_no_replace(receipt_path, receipt_payload)
    complete_body = {
        "status": "COMPLETE",
        "ledger_file_sha256": _file_sha(ledger_path),
        "count_receipt_file_sha256": _file_sha(receipt_path),
        "count_receipt_content_sha256": receipt["content_sha256"],
        "raw_source_files_retained": 0,
    }
    _atomic_write_no_replace(
        output_dir / "COMPLETE.json",
        json.dumps(complete_body, indent=2, sort_keys=True).encode() + b"\n",
    )


def execute_authorized(root: Path, authorization_path: Path, output_dir: Path) -> Path:
    root = root.resolve()
    config = load_config(root)
    if authorization_path.resolve() != (root / APPROVED_AUTH_PATH).resolve():
        raise GravityClusterActErassOverlapExecutorError(
            "authorization path differs from frozen command"
        )
    if output_dir.resolve() != (root / RESULT_DIR).resolve():
        raise GravityClusterActErassOverlapExecutorError("output path differs from frozen command")
    auth = _read_json(authorization_path.resolve())
    validate_authorized_manifest(auth, config, root)
    output = output_dir.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.mkdir(output)
    except FileExistsError as exc:
        raise GravityClusterActErassOverlapExecutorError(
            "result directory already exists; refusing before network"
        ) from exc
    raw_dir = output / ".private-raw-staging"
    try:
        raw_dir.mkdir()
        assets = _asset_map(config)
        get_budget = _GetBudget(2)
        source_receipts: dict[str, Any] = {}
        act_raw = raw_dir / str(assets[CATALOG_IDS[0]]["filename"])
        source_receipts[CATALOG_IDS[0]] = _download_exact(
            assets[CATALOG_IDS[0]], act_raw, budget=get_budget
        )
        erass_archive = raw_dir / str(assets[CATALOG_IDS[1]]["filename"])
        source_receipts[CATALOG_IDS[1]] = _download_exact(
            assets[CATALOG_IDS[1]], erass_archive, budget=get_budget
        )
        get_budget.verify_exact()
        erass_fits = raw_dir / str(assets[CATALOG_IDS[1]]["expected_fits_member"])
        source_receipts[CATALOG_IDS[1]]["extracted_member"] = extract_single_fits_member(
            erass_archive,
            erass_fits,
            expected_basename=str(assets[CATALOG_IDS[1]]["expected_fits_member"]),
            maximum_bytes=int(assets[CATALOG_IDS[1]]["maximum_decompressed_member_bytes"]),
        )
        act_rows, act_schema = read_fits_projection(
            act_raw,
            ACT_COLUMNS,
            expected_tforms=PROJECTION_TFORM_CONTRACT[CATALOG_IDS[0]],
            expected_rows=int(assets[CATALOG_IDS[0]]["expected_rows"]),
            maximum_rows=int(assets[CATALOG_IDS[0]]["expected_rows"]),
        )
        erass_rows, erass_schema = read_fits_projection(
            erass_fits,
            ERASS_COLUMNS,
            expected_tforms=PROJECTION_TFORM_CONTRACT[CATALOG_IDS[1]],
            expected_rows=int(assets[CATALOG_IDS[1]]["expected_rows"]),
            maximum_rows=int(assets[CATALOG_IDS[1]]["expected_rows"]),
        )
        ledger, counts = match_catalogs(act_rows, erass_rows)
        _remove_tree_verified(raw_dir, label="private raw staging after successful parsing")
        if raw_dir.exists():
            raise GravityClusterActErassOverlapExecutorError(
                "private raw staging remains before sanitized publication"
            )
        _write_execution_outputs(
            output,
            source_receipts=source_receipts,
            schema_receipts={CATALOG_IDS[0]: act_schema, CATALOG_IDS[1]: erass_schema},
            ledger=ledger,
            counts=counts,
            authorization_sha256=_file_sha(authorization_path.resolve()),
        )
        _fsync_directory(output)
        return output
    except Exception as execution_error:
        try:
            _remove_tree_verified(output, label="owned reserved result after execution failure")
        except GravityClusterActErassOverlapExecutorError as cleanup_error:
            raise GravityClusterActErassOverlapExecutorError(
                "execution failed and cleanup failed loudly"
            ) from cleanup_error
        if output.exists() or raw_dir.exists():
            raise GravityClusterActErassOverlapExecutorError(
                "execution cleanup postcondition failed"
            ) from execution_error
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    preflight = subparsers.add_parser("build-preflight")
    preflight.add_argument("--root", type=Path, default=Path.cwd())
    check = subparsers.add_parser("check")
    check.add_argument("--root", type=Path, default=Path.cwd())
    execute = subparsers.add_parser("execute")
    execute.add_argument("--root", type=Path, default=Path.cwd())
    execute.add_argument("--authorization", type=Path, required=True)
    execute.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.action == "build-preflight":
        print(write_preflight_receipt(args.root))
    elif args.action == "check":
        receipt = _read_json(args.root.resolve() / PREFLIGHT_PATH)
        validate_preflight_receipt(receipt, args.root)
        print(json.dumps({"status": "PASS", "content_sha256": receipt["content_sha256"]}))
    else:
        print(execute_authorized(args.root, args.authorization, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
