"""Verify the metadata-only ACT DR6 x eRASS1 overlap preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_act_dr6_erass1_overlap_preflight_v1.json")
AUTH_PATH = Path(
    "runs/gravity/publication-readiness/act-dr6-erass1-overlap-preflight-v1/"
    "authorization-current-unauthorized.json"
)
OUTPUT_PATH = Path("runs/gravity/publication-readiness/act-dr6-erass1-overlap-preflight-v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_cluster_act_dr6_erass1_overlap_preflight.py")
TEST_PATH = Path("tests/test_gravity_cluster_act_dr6_erass1_overlap_preflight.py")
CONFIG_FILE_SHA256 = "8ea9fdcf0b7aa36ba277a32d931363b81ed8bc93ecc8034261e0954abe937735"
TEST_FILE_SHA256 = "f393a3c237d6f58787bbd1581d9d163938fa6b01ff8d126a53cff9500f047565"
AUTH_FILE_SHA256 = "de29475463092498194b42b9825fafe1cbc87ab960dd6c80324f66425acb4060"
CONFIG_SCHEMA = "invariant-gravity-cluster-act-dr6-erass1-overlap-preflight-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-act-dr6-erass1-overlap-preflight-receipt-1.0"

FACT_FIELDS = (
    "source_id",
    "source_type",
    "url",
    "doi",
    "observed_release",
    "audited_fact",
    "observation_method",
    "retrieved_at",
    "evidence_locator",
)
SOURCE_IDS = (
    "ACT_DR6_CATALOG_RELEASE",
    "ACT_DR6_FITS_HEADERS",
    "ACT_DR6_PRIMARY_PAPER",
    "ACT_DR6_02_MAP_RELEASE",
    "ERASS1_CLUSTER_RELEASE",
    "ERASS1_PRIMARY_V3_2_DATA_MODEL",
    "ERASS1_DR1_RIGHTS_AND_ACKNOWLEDGEMENT",
)
ASSET_IDS = (
    "ACT_DR6_FULL_V1_0",
    "ACT_DR6_LEGACY_V1_0",
    "ERASS1_PRIMARY_V3_2",
    "ACT_DR6_02_MAP_FAMILY",
)
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
SECTION_BINDINGS = {
    "parent_bindings": "24cf1c7d121a7000298102dd692c9c23e93bfcab3680d65ac92cba83a4885075",
    "fact_provenance_contract": "1851158e3b37b49720cd96b97eff2a0af06d4a8abf576a5bc49733bb68644af9",
    "authoritative_sources": "e55e1153f30a37ec2f1568682972579eab37f225c129a8ef838e0ab20b282920",
    "file_manifest": "f6cfee5dfd60c9e49ec7fc14243b9017d57be6a92e0e1d5ad7ee8f0fc03f4f01",
    "catalog_projection_contract": "287e7958bffab1e4d4ef551eb99c1c12448ff8cdf440344b118549290a20b45d",
    "selection_and_quality_contract": "71d391326eb3ba0ebae7be1a7e3f20df22d6ddc717b0ccf6acfa0ba5bbe7d357",
    "act_erass_match_contract": "6587dd6bd0726f51bca430b1bb8be11bb5e67be18e7582aa6e9fa045aa0b4561",
    "xcop_exclusion_ledger_contract": "8b9a12d66b50767eda63e2ef106d86fefde3e80cdd31355162e95bb17b98f750",
    "population_gate": "afb99fbaf26e80e4af2dd4f15891f47210d24541fd39dbe77eb761f12496d14e",
    "license_and_terms": "131e1065cc495ceda0d356523a954d623c85d87294b38a7f5bad744aaae46cb2",
    "access_and_decision": "c7c9a61f79d9822bb8994379094b9ee32045f217a7dd169a0a3eb84350dfb200",
    "claim_boundary": "63f3542d79dd9c53863e45051aa8af0d9a8486088c0eb88dcd230d7fb564e609",
    "publication_contract": "6c70a5c55e35fe73dd3d2e89ee7074a4504c56fffd06cfa045d101f8dbf06c2c",
}


class GravityClusterActErassOverlapPreflightError(RuntimeError):
    """Raised when the frozen overlap preflight changes."""


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
        raise GravityClusterActErassOverlapPreflightError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterActErassOverlapPreflightError(f"{label} keys changed")


def _fact_material(source: Mapping[str, Any]) -> dict[str, Any]:
    return {field: source[field] for field in FACT_FIELDS}


def _validate_authorization(auth: Mapping[str, Any]) -> None:
    if (
        auth["schema_version"] != "invariant-act-dr6-erass1-catalog-access-authorization-1.0"
        or auth["status"] != "UNAUTHORIZED_CATALOG_ROWS_NOT_OPENED"
        or auth["authorization_id"] != "act-dr6-erass1-overlap-catalog-access-v1"
    ):
        raise GravityClusterActErassOverlapPreflightError("authorization identity changed")
    if tuple(item["catalog_id"] for item in auth["catalogs"]) != (
        "ACT_DR6_LEGACY_V1_0",
        "ERASS1_PRIMARY_V3_2",
    ):
        raise GravityClusterActErassOverlapPreflightError("authorization catalog list changed")
    if any(item["authorized"] is not False for item in auth["catalogs"]):
        raise GravityClusterActErassOverlapPreflightError("catalog authorization became true")
    access = auth["access_state"]
    if access["authorization"] is not False or any(
        access[key] != 0
        for key in (
            "catalog_files_downloaded",
            "catalog_bytes_downloaded",
            "catalog_rows_opened",
            "alias_rows_opened",
            "scientific_target_rows_opened",
            "map_or_profile_bytes_downloaded",
            "scores_computed",
            "model_or_paid_calls",
        )
    ):
        raise GravityClusterActErassOverlapPreflightError("authorization access state changed")
    if any(auth["claim_boundary"].values()):
        raise GravityClusterActErassOverlapPreflightError("authorization claim overstated")
    future = auth["future_exact_approval_schema"]
    if (
        future["required_status"] != "AUTHORIZED_CATALOG_ONLY_ACCESS"
        or future["maximum_rows"] != {"ACT_DR6_LEGACY_V1_0": 3747, "ERASS1_PRIMARY_V3_2": 12247}
        or future["no_map_profile_target_or_scoring_authorization"] is not True
    ):
        raise GravityClusterActErassOverlapPreflightError("future approval schema changed")


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "preflight_id",
            "audit_cutoff",
            "purpose",
            "implementation_evidence",
            "parent_bindings",
            "authorization_manifest",
            "fact_provenance_contract",
            "authoritative_sources",
            "file_manifest",
            "catalog_projection_contract",
            "selection_and_quality_contract",
            "act_erass_match_contract",
            "xcop_exclusion_ledger_contract",
            "population_gate",
            "license_and_terms",
            "access_and_decision",
            "claim_boundary",
            "publication_contract",
            "output_path",
        },
        "overlap preflight config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only_unauthorized_catalog_rows_required"
        or config["preflight_id"] != "gravity-cluster-act-dr6-erass1-overlap-preflight-v1"
        or config["audit_cutoff"] != "2026-08-29"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterActErassOverlapPreflightError("preflight identity changed")
    if config["implementation_evidence"] != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
        "test_binding_required": True,
    }:
        raise GravityClusterActErassOverlapPreflightError("implementation evidence changed")
    auth_binding = config["authorization_manifest"]
    if auth_binding != {
        "path": AUTH_PATH.as_posix(),
        "file_sha256": AUTH_FILE_SHA256,
        "required_status": "UNAUTHORIZED_CATALOG_ROWS_NOT_OPENED",
        "authorization": False,
    }:
        raise GravityClusterActErassOverlapPreflightError("authorization binding changed")
    sources = config["authoritative_sources"]
    if tuple(item["source_id"] for item in sources) != SOURCE_IDS:
        raise GravityClusterActErassOverlapPreflightError("source inventory changed")
    for source in sources:
        _strict(source, set(FACT_FIELDS) | {"fact_binding_sha256"}, source["source_id"])
        if source["retrieved_at"] != "2026-08-29":
            raise GravityClusterActErassOverlapPreflightError("source date changed")
        if source["fact_binding_sha256"] != _sha(_fact_material(source)):
            raise GravityClusterActErassOverlapPreflightError(
                f"source fact binding changed: {source['source_id']}"
            )
    assets = config["file_manifest"]
    if tuple(item["asset_id"] for item in assets) != ASSET_IDS:
        raise GravityClusterActErassOverlapPreflightError("asset inventory changed")
    by_id = {item["asset_id"]: item for item in assets}
    expected_assets = {
        "ACT_DR6_FULL_V1_0": (25971840, 10040, '"18c4c80-645ab2531aa80"'),
        "ACT_DR6_LEGACY_V1_0": (9705600, 3747, '"941880-645ab26ec2bc0"'),
        "ERASS1_PRIMARY_V3_2": (23764095, 12247, '"16a9c7f-610f2c2a175aa"'),
    }
    for asset_id, (size, rows, etag) in expected_assets.items():
        asset = by_id[asset_id]
        if (
            asset["bytes"] != size
            or asset["rows"] != rows
            or asset["etag_noncryptographic"] != etag
            or asset["sha256"] is not None
            or asset["downloaded"] is not False
            or asset["rows_opened"] != 0
        ):
            raise GravityClusterActErassOverlapPreflightError(f"asset state changed: {asset_id}")
    if by_id["ACT_DR6_02_MAP_FAMILY"]["downloaded"] is not False:
        raise GravityClusterActErassOverlapPreflightError("map boundary changed")
    projections = config["catalog_projection_contract"]
    if any(
        "M500" in column.upper()
        for projection in projections.values()
        if isinstance(projection, dict)
        for column in projection.get("allowed_columns", [])
    ):
        raise GravityClusterActErassOverlapPreflightError("mass field entered projection")
    match = config["act_erass_match_contract"]
    if (
        match["coordinate_radius"]
        != "theta_match(z)=max(1.22 arcmin, angular size of 0.5 Mpc at the ACT redshift under the frozen cosmology), matching the ACT primary-paper automated catalog rule."
        or match["overlap_count_state"] != "NOT_COMPUTED_CATALOG_ROWS_UNAUTHORIZED"
        or match["ambiguous_or_missing_action"] != "QUARANTINE_AND_DO_NOT_COUNT"
    ):
        raise GravityClusterActErassOverlapPreflightError("match contract changed")
    xcop = config["xcop_exclusion_ledger_contract"]
    if (
        tuple(xcop["canonical_xcop_objects"]) != XCOP_OBJECTS
        or xcop["executed"] is not False
        or xcop["input_rows"] != 0
        or xcop["resolved_overlaps"] != 0
    ):
        raise GravityClusterActErassOverlapPreflightError("X-COP exclusion boundary changed")
    gate = config["population_gate"]
    if (
        gate["confirmatory_target_clusters"] != 192
        or gate["underpowered_execution_floor_clusters"] != 120
        or gate["catalog_overlap_count"] is not None
        or gate["post_xcop_catalog_upper_bound"] is not None
        or gate["rule_evaluated"] is not False
    ):
        raise GravityClusterActErassOverlapPreflightError("population gate changed")
    if any(
        item["redistribution_authorized"]
        for item in config["license_and_terms"].values()
        if isinstance(item, dict)
    ):
        raise GravityClusterActErassOverlapPreflightError("license boundary overstated")
    access = config["access_and_decision"]
    if (
        access["current_catalog_access_authorized"] is not False
        or access["catalog_row_access_required_for_overlap_count"] is not True
        or access["decision"]
        != "BLOCKED_EXACT_CATALOG_ONLY_AUTHORIZATION_REQUIRED_BEFORE_OVERLAP_COUNT"
        or any(
            access[key] != 0
            for key in (
                "catalog_files_downloaded",
                "catalog_bytes_downloaded",
                "catalog_rows_opened",
                "alias_rows_opened",
                "profile_thermodynamic_lensing_target_rows_opened",
                "map_bytes_downloaded",
                "scores_computed",
                "model_or_paid_calls",
            )
        )
    ):
        raise GravityClusterActErassOverlapPreflightError("access boundary changed")
    claims = config["claim_boundary"]
    if claims["metadata_preflight_complete"] is not True:
        raise GravityClusterActErassOverlapPreflightError("metadata claim changed")
    if any(value for key, value in claims.items() if key != "metadata_preflight_complete"):
        raise GravityClusterActErassOverlapPreflightError("claim boundary overstated")
    publication = config["publication_contract"]
    if (
        publication["publish_primitive"] != "SAME_DIRECTORY_HARD_LINK_NO_REPLACE"
        or publication["staging_file_fsync_before_publish"] is not True
        or publication["existing_different_receipt_action"] != "FAIL_CLOSED"
    ):
        raise GravityClusterActErassOverlapPreflightError("publication contract weakened")
    for section, expected in SECTION_BINDINGS.items():
        if _sha(config[section]) != expected:
            raise GravityClusterActErassOverlapPreflightError(
                f"canonical nested section changed: {section}"
            )


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    if _file_sha(root / CONFIG_PATH) != CONFIG_FILE_SHA256:
        raise GravityClusterActErassOverlapPreflightError("config hash changed")
    config = _read_json(root / CONFIG_PATH)
    validate_config(config)
    if _file_sha(root / TEST_PATH) != TEST_FILE_SHA256:
        raise GravityClusterActErassOverlapPreflightError("test binding changed")
    auth_path = root / AUTH_PATH
    if _file_sha(auth_path) != AUTH_FILE_SHA256:
        raise GravityClusterActErassOverlapPreflightError("authorization file changed")
    auth = _read_json(auth_path)
    _validate_authorization(auth)
    for item in auth["catalogs"]:
        if (
            item["permitted_columns"]
            != config["catalog_projection_contract"][item["catalog_id"]]["allowed_columns"]
        ):
            raise GravityClusterActErassOverlapPreflightError(
                "authorization projection differs from config"
            )
    for parent in config["parent_bindings"]:
        for path_key, sha_key in (
            ("config_path", "config_file_sha256"),
            ("module_path", "module_file_sha256"),
            ("test_path", "test_file_sha256"),
            ("receipt_path", "receipt_file_sha256"),
        ):
            if _file_sha(root / parent[path_key]) != parent[sha_key]:
                raise GravityClusterActErassOverlapPreflightError(
                    f"parent binding changed: {path_key}"
                )
        receipt = _read_json(root / parent["receipt_path"])
        if receipt.get("content_sha256") != parent["receipt_content_sha256"]:
            raise GravityClusterActErassOverlapPreflightError("parent receipt content changed")
    return config


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "preflight_id": config["preflight_id"],
        "audit_cutoff": config["audit_cutoff"],
        "status": config["status"],
        "decision": config["access_and_decision"]["decision"],
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
        "authorization_binding": {
            "path": AUTH_PATH.as_posix(),
            "file_sha256": _file_sha(root / AUTH_PATH),
            "status": "UNAUTHORIZED_CATALOG_ROWS_NOT_OPENED",
            "authorization": False,
        },
        "parent_bindings": config["parent_bindings"],
        "authoritative_sources": config["authoritative_sources"],
        "file_manifest": config["file_manifest"],
        "catalog_projection_contract": config["catalog_projection_contract"],
        "selection_and_quality_contract": config["selection_and_quality_contract"],
        "act_erass_match_contract": config["act_erass_match_contract"],
        "xcop_exclusion_ledger_contract": config["xcop_exclusion_ledger_contract"],
        "population_gate": config["population_gate"],
        "license_and_terms": config["license_and_terms"],
        "access_and_decision": config["access_and_decision"],
        "claims": config["claim_boundary"],
        "counts": {
            "authoritative_metadata_sources": len(config["authoritative_sources"]),
            "file_manifest_records": len(config["file_manifest"]),
            "catalog_files_downloaded": 0,
            "catalog_bytes_downloaded": 0,
            "catalog_rows_opened": 0,
            "profile_thermodynamic_lensing_target_rows_opened": 0,
            "map_bytes_downloaded": 0,
            "overlap_rows": 0,
            "scores_computed": 0,
            "network_calls_by_receipt_builder": 0,
            "model_or_paid_calls": 0,
            "ready_lanes": 0,
        },
        "limitations": [
            "Official metadata pages cannot produce the ACT DR6 x eRASS1 overlap count; ACT eRASS1CL and independent identity fields are catalog rows.",
            "No prior receipt authorizes ACT or eRASS1 catalog rows, so no catalog was downloaded or opened.",
            "The current Legacy v1.0 FITS header contains 3,747 rows, while paper-era parent metadata reports 3,758; the discrepancy is preserved and unresolved without a file receipt.",
            "Remote SHA-256 checksums and explicit per-file redistribution grants were not found in the audited official metadata.",
            "The 192-object rule is unevaluated, and even a passing catalog upper bound would not establish profile completeness or replication readiness.",
        ],
        "next_action": config["access_and_decision"]["next_action"],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    content_sha = body.pop("content_sha256", None)
    if content_sha != _sha(body):
        raise GravityClusterActErassOverlapPreflightError("receipt content hash changed")
    if receipt != build_receipt(root):
        raise GravityClusterActErassOverlapPreflightError("receipt differs from frozen preflight")


def _fsync_directory(directory: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_no_replace(path: Path, payload: bytes) -> None:
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
            if path.read_bytes() != payload:
                raise GravityClusterActErassOverlapPreflightError(
                    "refusing to replace a different overlap receipt"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_receipt(root)
    payload = json.dumps(receipt, indent=2, sort_keys=True).encode() + b"\n"
    path = root / OUTPUT_PATH
    _atomic_publish_no_replace(path, payload)
    validate_receipt(_read_json(path), root)
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("build-receipt", "check"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    if args.action == "build-receipt":
        print(write_receipt(args.root))
    else:
        receipt = _read_json(args.root.resolve() / OUTPUT_PATH)
        validate_receipt(receipt, args.root)
        print(json.dumps({"status": "PASS", "content_sha256": receipt["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
