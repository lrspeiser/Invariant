"""Seal the append-only metadata-only group-scale source audit V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_group_scale_source_audit_v3.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_group_scale_source_audit_v3.py")
TEST_PATH = Path("tests/test_gravity_group_scale_source_audit_v3.py")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/group-scale-source-audit-v3.json")
CONFIG_FILE_SHA256 = "df36fd618328e07977d44d47a286581702a22ecd8870567ae41321dc3288247e"
TEST_FILE_SHA256 = "2e3187f9deddde6a4e764f5e0f7c676cf860c563e2e1131898a576ee5c19b7ee"
CONFIG_SCHEMA = "invariant-gravity-group-scale-source-audit-config-3.0"
RECEIPT_SCHEMA = "invariant-gravity-group-scale-source-audit-receipt-3.0"
PREDECESSOR_PATH_KEYS = ("config_path", "module_path", "test_path", "receipt_path")
PREDECESSOR_SHA_KEYS = (
    "config_file_sha256",
    "module_file_sha256",
    "test_file_sha256",
    "receipt_file_sha256",
)
SECTION_BINDINGS = {
    "predecessor_bindings": "e174b8f6313a8666b9fdafd7d871f0514e0aa85d3acd0bc2f376950bb2088884",
    "audit_method": "f529abd416e3f1fb515cdc23235312a321f384a2e920b5ca810835020ee37cdc",
    "authoritative_source_facts": "26849aa8699573ae066b7c3e7d8ac8d060f73ea8ee34d08b131b355585cd34e1",
    "lane_readiness": "3fbb92fe34d281ab3a20375acc0db85621f162008843bd14b6c045eaac2932a0",
    "xcop_overlap_contract": "db4acdd83d93772c748dc0c9e666208585bccfcf96c1c4d416d6cf437a6b2526",
    "interactive_audit_disclosure": "090a844f0c2019e7eaecbb9fe57a51f95ffa9812c9a6525fb9da5401f482c629",
    "future_identity_obsid_acquisition": "a7fb06be794d3607be3c89a62b25142bbe7420ceb53c3a9e9a5024853d5cd2bf",
    "future_xclass_five_object_pilot": "ee8cc43c571a3150e1f28d1c9e5afa9bacaebe7b8cfde8a290bb837e5b431faa",
    "access_chronology": "f3225373fae08f73fa956d982eb337b90f1ff51390c5e7f16a766dc0b44baa96",
    "claim_boundary": "463091bf4e20e6d17043cb5bc40bfc4accf7c0d922a998e23825f7420a225094",
    "publication_contract": "6c70a5c55e35fe73dd3d2e89ee7074a4504c56fffd06cfa045d101f8dbf06c2c",
}


class GravityGroupScaleSourceAuditV3Error(RuntimeError):
    """Raised when the frozen V3 source-audit package changes."""


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
        raise GravityGroupScaleSourceAuditV3Error(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise GravityGroupScaleSourceAuditV3Error(f"{label} keys changed")


def _lane(config: Mapping[str, Any], lane_id: str) -> Mapping[str, Any]:
    matches = [row for row in config["lane_readiness"] if row["lane_id"] == lane_id]
    if len(matches) != 1:
        raise GravityGroupScaleSourceAuditV3Error(f"lane inventory changed: {lane_id}")
    return matches[0]


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "audit_id",
            "audit_cutoff",
            "purpose",
            "implementation_evidence",
            "predecessor_bindings",
            "audit_method",
            "authoritative_source_facts",
            "lane_readiness",
            "xcop_overlap_contract",
            "interactive_audit_disclosure",
            "future_identity_obsid_acquisition",
            "future_xclass_five_object_pilot",
            "access_chronology",
            "claim_boundary",
            "decision",
            "publication_contract",
            "output_path",
        },
        "V3 config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only_audit_zero_ready_science_lanes"
        or config["audit_id"] != "gravity-group-scale-source-audit-v3"
        or config["audit_cutoff"] != "2026-08-29"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityGroupScaleSourceAuditV3Error("V3 identity changed")
    if config["implementation_evidence"] != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
        "test_binding_required": True,
    }:
        raise GravityGroupScaleSourceAuditV3Error("implementation evidence changed")

    predecessors = config["predecessor_bindings"]
    if (
        not isinstance(predecessors, list)
        or [row.get("audit_id") for row in predecessors]
        != ["gravity-group-scale-source-audit-v1", "gravity-group-scale-bridge-acquisition-v2"]
        or _sha(predecessors) != SECTION_BINDINGS["predecessor_bindings"]
    ):
        raise GravityGroupScaleSourceAuditV3Error("predecessor bindings changed")

    method = config["audit_method"]
    if (
        method["retrieved_at"] != "2026-08-29"
        or method["remote_content_archived"] is not False
        or method["search_scope_is_not_proof_of_absence"] is not True
        or method["receipt_builder_has_network_code"] is not False
        or "download or open scientific catalog data rows"
        not in method["forbidden_remote_operations"]
    ):
        raise GravityGroupScaleSourceAuditV3Error("audit method boundary changed")

    sources = config["authoritative_source_facts"]
    if not isinstance(sources, list) or len(sources) != 14:
        raise GravityGroupScaleSourceAuditV3Error("authoritative source inventory changed")
    source_ids = [row.get("source_id") for row in sources]
    if len(set(source_ids)) != len(source_ids):
        raise GravityGroupScaleSourceAuditV3Error("duplicate source identity")
    required_source_keys = {
        "source_id",
        "lane_id",
        "source_type",
        "url",
        "doi",
        "retrieved_at",
        "observation_method",
        "audited_fact",
        "license",
        "assets",
    }
    required_asset_keys = {"url", "bytes", "rows", "payload_opened"}
    for source in sources:
        _strict(source, required_source_keys, f"source {source.get('source_id')}")
        if source["retrieved_at"] != "2026-08-29" or not source["url"].startswith("https://"):
            raise GravityGroupScaleSourceAuditV3Error("source provenance changed")
        if not isinstance(source["assets"], list):
            raise GravityGroupScaleSourceAuditV3Error("source asset schema changed")
        for asset in source["assets"]:
            _strict(asset, required_asset_keys, "source asset")
            if asset["payload_opened"] is not False:
                raise GravityGroupScaleSourceAuditV3Error("source payload boundary changed")

    lanes = config["lane_readiness"]
    if not isinstance(lanes, list) or len(lanes) != 11:
        raise GravityGroupScaleSourceAuditV3Error("lane inventory changed")
    if any(row.get("decision") == "READY" for row in lanes):
        raise GravityGroupScaleSourceAuditV3Error("ready lane claim is forbidden")
    if (
        _lane(config, "XCLASS_LOWZ_155")["role"] != "PREFERRED_RAW_REDUCTION_COHORT"
        or _lane(config, "XCLASS_LOWZ_155")["decision"] != "PARTIAL"
        or _lane(config, "EFEDS_542_RAW_REDUCTION")["role"] != "BACKUP_COMMON_INSTRUMENT_COHORT"
        or _lane(config, "XGAP_49_XMM")["decision"] != "BLOCKED"
        or _lane(config, "ERASS1_2MRS_619")["license_state"]
        != "CATALOG_IDENTIFIER_AND_TERMS_UNRESOLVED"
    ):
        raise GravityGroupScaleSourceAuditV3Error("lane priority or state changed")
    accept = _lane(config, "ACCEPT_239")
    accept_source = next(source for source in sources if source["source_id"] == "ACCEPT_OFFICIAL")
    if (
        accept["documented_objects"] is not None
        or accept["reported_counts"]
        != {
            "author_project_overview_sample": 239,
            "current_heasarc_one_row_per_cluster_table": 240,
        }
        or accept["population_count_state"]
        != "UNRESOLVED_239_AUTHOR_SAMPLE_VS_240_CURRENT_HEASARC_ROWS"
        or accept["decision"] != "BLOCKED"
        or "original sample of 239" not in accept_source["audited_fact"]
        or "reports 240 rows" not in accept_source["audited_fact"]
    ):
        raise GravityGroupScaleSourceAuditV3Error("ACCEPT count reconciliation changed")

    overlap = config["xcop_overlap_contract"]
    if (
        overlap["executed"] is not False
        or overlap["overlap_count"] is not None
        or len(overlap["frozen_xcop_names"]) != 12
        or overlap["development_count"] != 8
        or overlap["formerly_exposed_same_release_holdout_count"] != 4
        or overlap["mass_range_inference_is_not_overlap_proof"] is not True
    ):
        raise GravityGroupScaleSourceAuditV3Error("X-COP overlap boundary changed")

    incident = config["interactive_audit_disclosure"]
    if (
        incident["incident_id"] != "ACCEPT_OFFICIAL_PAGE_EMBEDDED_ROW_RENDER_2026_08_29"
        or incident["incident_scope"] != "INTERACTIVE_RESEARCH_SESSION_OUTSIDE_ARTIFACT_BUILDER"
        or incident["webpage_rendered_embedded_scientific_table_rows"] is not True
        or incident["rendered_row_count"] is not None
        or incident["query_executed"] is not False
        or incident["file_downloaded"] is not False
        or incident["rows_persisted"] is not False
        or incident["rows_used_for_selection_overlap_scoring_or_facts"] is not False
        or incident["receipt_builder_involved"] is not False
    ):
        raise GravityGroupScaleSourceAuditV3Error("ACCEPT disclosure changed")

    acquisition = config["future_identity_obsid_acquisition"]
    if (
        acquisition["defined"] is not True
        or acquisition["executed"] is not False
        or acquisition["authorized"] is not False
        or acquisition["preferred_lane"] != "XCLASS_LOWZ_155"
        or acquisition["backup_lane"] != "EFEDS_542_RAW_REDUCTION"
        or "temperature" not in acquisition["forbidden_fields"]
        or "mission_observation_id" not in acquisition["allowed_fields"]
    ):
        raise GravityGroupScaleSourceAuditV3Error("future acquisition lock changed")
    pilot = config["future_xclass_five_object_pilot"]
    if (
        pilot["defined"] is not True
        or pilot["executed"] is not False
        or pilot["authorized"] is not False
        or pilot["object_count"] != 5
        or pilot["object_identities_selected"] is not False
        or pilot["selection_must_be_target_blind"] is not True
    ):
        raise GravityGroupScaleSourceAuditV3Error("future pilot lock changed")

    access = config["access_chronology"]
    if access["scope"] != "ARTIFACT_BUILDER_ONLY" or any(
        value != 0 for key, value in access.items() if key != "scope"
    ):
        raise GravityGroupScaleSourceAuditV3Error("access chronology changed")
    claims = config["claim_boundary"]
    allowed_true = {
        "metadata_source_audit_complete",
        "future_identity_protocol_defined",
        "future_pilot_protocol_defined",
    }
    if claims["ready_science_lanes"] != 0:
        raise GravityGroupScaleSourceAuditV3Error("claim boundary ready count changed")
    for key, value in claims.items():
        if key in allowed_true and value is not True:
            raise GravityGroupScaleSourceAuditV3Error(f"claim boundary understated: {key}")
        if key not in allowed_true | {"ready_science_lanes"} and value is not False:
            raise GravityGroupScaleSourceAuditV3Error(f"claim boundary overstated: {key}")
    if config["decision"] != (
        "METADATA_SOURCE_AUDIT_V3_SEALED_ZERO_READY_LANES_XCLASS_PREFERRED_"
        "EFEDS_BACKUP_NO_ACQUISITION_AUTHORIZED"
    ):
        raise GravityGroupScaleSourceAuditV3Error("decision changed")
    publication = config["publication_contract"]
    if (
        publication["publish_primitive"] != "SAME_DIRECTORY_HARD_LINK_NO_REPLACE"
        or publication["staging_file_fsync_before_publish"] is not True
        or publication["existing_different_receipt_action"] != "FAIL_CLOSED"
        or publication["race_loser_action"] != "VERIFY_IDENTICAL_OR_FAIL_CLOSED"
    ):
        raise GravityGroupScaleSourceAuditV3Error("publication contract weakened")

    for section, expected in SECTION_BINDINGS.items():
        if _sha(config[section]) != expected:
            label = "predecessor" if section == "predecessor_bindings" else "section"
            raise GravityGroupScaleSourceAuditV3Error(f"frozen {label} changed: {section}")


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / CONFIG_PATH
    if not config_path.is_file() or _file_sha(config_path) != CONFIG_FILE_SHA256:
        raise GravityGroupScaleSourceAuditV3Error("V3 config hash changed")
    config = _read_json(config_path)
    validate_config(config)
    if _file_sha(root / TEST_PATH) != TEST_FILE_SHA256:
        raise GravityGroupScaleSourceAuditV3Error("V3 test binding changed")
    for predecessor in config["predecessor_bindings"]:
        for path_key, sha_key in zip(PREDECESSOR_PATH_KEYS, PREDECESSOR_SHA_KEYS, strict=True):
            path = root / predecessor[path_key]
            if not path.is_file() or _file_sha(path) != predecessor[sha_key]:
                raise GravityGroupScaleSourceAuditV3Error(
                    f"predecessor binding changed: {predecessor['audit_id']} {path_key}"
                )
        receipt = _read_json(root / predecessor["receipt_path"])
        if receipt.get("content_sha256") != predecessor["receipt_content_sha256"]:
            raise GravityGroupScaleSourceAuditV3Error(
                f"predecessor receipt content changed: {predecessor['audit_id']}"
            )
    return config


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": config["audit_id"],
        "audit_cutoff": config["audit_cutoff"],
        "status": config["status"],
        "decision": config["decision"],
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
        "predecessor_bindings": config["predecessor_bindings"],
        "audit_method": config["audit_method"],
        "authoritative_source_facts": config["authoritative_source_facts"],
        "lane_readiness": config["lane_readiness"],
        "xcop_overlap_contract": config["xcop_overlap_contract"],
        "interactive_audit_disclosure": config["interactive_audit_disclosure"],
        "future_identity_obsid_acquisition": config["future_identity_obsid_acquisition"],
        "future_xclass_five_object_pilot": config["future_xclass_five_object_pilot"],
        "access_chronology": config["access_chronology"],
        "claims": config["claim_boundary"],
        "counts": {
            "authoritative_source_records": len(config["authoritative_source_facts"]),
            "remote_asset_metadata_records": sum(
                len(source["assets"]) for source in config["authoritative_source_facts"]
            ),
            "lane_records": len(config["lane_readiness"]),
            "ready_science_lanes": sum(
                row["decision"] == "READY" for row in config["lane_readiness"]
            ),
            "partial_lanes": sum(row["decision"] == "PARTIAL" for row in config["lane_readiness"]),
            "blocked_lanes": sum(row["decision"] == "BLOCKED" for row in config["lane_readiness"]),
            "network_calls_by_receipt_builder": 0,
            "catalog_payload_downloads_by_receipt_builder": 0,
            "scientific_rows_opened_by_receipt_builder": 0,
            "scores_computed": 0,
            "model_or_paid_calls": 0,
            "future_acquisition_runs": 0,
            "future_pilot_runs": 0,
        },
        "claim_limit": (
            "Metadata-only source triage. No group sample, overlap, scientific packet, raw reduction, "
            "formula score, CP10 completion, or publication claim is produced."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    content_sha = body.pop("content_sha256", None)
    if content_sha != _sha(body):
        raise GravityGroupScaleSourceAuditV3Error("receipt content hash changed")
    if receipt != build_receipt(root):
        raise GravityGroupScaleSourceAuditV3Error("receipt differs from frozen V3 audit")


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
                raise GravityGroupScaleSourceAuditV3Error(
                    "refusing to replace a different V3 source-audit receipt"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    path = (root / OUTPUT_PATH).resolve()
    if not path.is_relative_to(root):
        raise GravityGroupScaleSourceAuditV3Error("output path escapes root")
    receipt = build_receipt(root)
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
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
