"""Verify the metadata-only CP10 group-scale bridge acquisition contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_group_scale_bridge_acquisition_v2.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/group-scale-bridge-acquisition-v2.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_group_scale_bridge_acquisition.py")
TEST_PATH = Path("tests/test_gravity_group_scale_bridge_acquisition.py")
SOURCE_PATH = Path(
    "runs/gravity/publication-readiness/group-scale-bridge-acquisition-v2-sources/"
    "axes2mrs-cds-readme-2024-10-08.txt"
)
CONFIG_FILE_SHA256 = "421992a178d89ced8123958c015fb40b7a403fc21e97c1e1c18b27aee9e40567"
TEST_FILE_SHA256 = "5b0b737b7e02b9807ed0402178d6c0e10d968fbd9ab0f2f31c1e238279b79ede"
SOURCE_FILE_SHA256 = "136a74a7218f8913113e8f86b9a9a529391a9cfe461e9d940156549e9486bcd2"
SOURCE_FILE_BYTES = 6278
CONFIG_SCHEMA = "invariant-gravity-group-scale-bridge-acquisition-config-2.0"
RECEIPT_SCHEMA = "invariant-gravity-group-scale-bridge-acquisition-receipt-2.0"

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
    "XGAP_SAMPLE_PAGE_LIVE_REDIRECT",
    "XGAP_MASTER_ATTACHMENT",
    "XGAP_PRIMARY_2024",
    "XMM_LARGE_PROGRAMME_090389",
    "AXES2MRS_CDS_README",
    "AXES2MRS_PRIMARY_2024",
)
HISTORICAL_SOURCE_IDS = ("XGAP_SAMPLE_SELECTION_LEGACY_CACHE",)
ASSET_IDS = (
    "XGAP_SAMPLE_PAGE",
    "XGAP_MASTER_V1_1",
    "AXES2MRS_README",
    "AXES2MRS_DATA",
)
SECTION_BINDINGS = {
    "fact_provenance_contract": "8b2a9d866400b21ebb4437ea33fdf538114031adedec0c33b6e3ea4ca17205b5",
    "source_discovery_scope": "c347532e21aaae98bbd309e59840bee5fe81fd6b4573b0998373b11b87f54f42",
    "historical_non_authoritative_evidence": "8b82a39ff1daed14b6e7587b040a4707162c78b4668eb892c5a5dae46277603d",
    "authoritative_sources": "274fc1b60af3abcbda069ad7f0542e8deaee4ff06b466df286ac10a2cc752775",
    "remote_assets": "f2febbfa8059facbf9221f9a09961efd2432c2a929f40c983974fcf128e41542",
    "sample_and_alias_state": "94e80aa041d4c7b1d76f24434ac3ffe0b3f3157aa1c5cd4cf7455fa6afd1714e",
    "xcop_overlap_contract": "1a2d3f4f40bc72876ba56dac7463f8d1743d73d9ff8f1b648b2b0f5c180054d4",
    "endpoint_and_covariance_state": "948279b2b7b11fd470496093c8b8e50f47058967f9f7a876009cd742cf970226",
    "license_and_redistribution_state": "75a69820fb838998d30994cc4f5bc3d9f19b3f359fdfed3be599e4f19bd68965",
    "access_chronology": "67a01a59811be007c8e33a5e0c28e259c45d05238611bbf6775f964b6bea7ca2",
    "claim_boundary": "2f977bebd9237253736cd41e54c5faf93c06db85c2d1029771fd6c006a3a6389",
    "publication_contract": "6c70a5c55e35fe73dd3d2e89ee7074a4504c56fffd06cfa045d101f8dbf06c2c",
}


class GravityGroupScaleBridgeAcquisitionError(RuntimeError):
    """Raised when the frozen metadata-only acquisition package changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityGroupScaleBridgeAcquisitionError(f"{label} keys changed")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityGroupScaleBridgeAcquisitionError(f"expected JSON object: {path}")
    return value


def _fact_material(source: Mapping[str, Any]) -> dict[str, Any]:
    return {field: source[field] for field in FACT_FIELDS}


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
            "parent_source_audit_binding",
            "fact_provenance_contract",
            "source_discovery_scope",
            "historical_non_authoritative_evidence",
            "authoritative_sources",
            "remote_assets",
            "sample_and_alias_state",
            "xcop_overlap_contract",
            "endpoint_and_covariance_state",
            "license_and_redistribution_state",
            "access_chronology",
            "claim_boundary",
            "decision",
            "next_action",
            "publication_contract",
            "output_path",
        },
        "bridge acquisition config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_manifest_only_bridge_blocked"
        or config["audit_id"] != "gravity-group-scale-bridge-acquisition-v2"
        or config["audit_cutoff"] != "2026-08-29"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityGroupScaleBridgeAcquisitionError("acquisition identity changed")

    implementation = config["implementation_evidence"]
    if implementation != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
        "test_binding_required": True,
    }:
        raise GravityGroupScaleBridgeAcquisitionError("implementation evidence changed")

    parent = config["parent_source_audit_binding"]
    if parent != {
        "config_path": "configs/gravity_group_scale_source_audit_v1.json",
        "config_file_sha256": "65b37105b66d0d97db4bc02d3c3eeeb1beafa429350429b7aa899943ffb2aaf9",
        "module_path": "src/sigma_theory_compiler/gravity_group_scale_source_audit.py",
        "module_file_sha256": "b506e5dad5d6de1899a4e179814f9651b71d57c8c709948524eced271392a93d",
        "test_path": "tests/test_gravity_group_scale_source_audit.py",
        "test_file_sha256": "3f5689428cd0b9e58bec8b46894d32d3bc4903bb08eee73b2fb478d3f669a069",
        "receipt_path": "runs/gravity/publication-readiness/group-scale-source-audit-v1.json",
        "receipt_file_sha256": "7707606dfdeae6a4bbeef06b0c7553074f8f5496a7f30d45a427fce10075f73e",
        "receipt_content_sha256": "d76a308369cf1175e418790cf2799905c9dc2484bb13ff1a1322c1f45160d3af",
    }:
        raise GravityGroupScaleBridgeAcquisitionError("parent source-audit binding changed")

    provenance = config["fact_provenance_contract"]
    if provenance["binding_algorithm"] != "SHA256_CANONICAL_JSON" or provenance[
        "binding_fields"
    ] != list(FACT_FIELDS):
        raise GravityGroupScaleBridgeAcquisitionError("fact provenance contract changed")
    if (
        provenance["retrieved_at_precision"] != "UTC_DAY"
        or provenance["search_scope_is_not_proof_of_absence"] is not True
        or provenance["remote_content_archived"] is not False
        or provenance["historical_legacy_excerpt_archived"] is not False
    ):
        raise GravityGroupScaleBridgeAcquisitionError("source-audit caveat weakened")

    discovery = config["source_discovery_scope"]
    if (
        discovery["preferred_lane"] != "XGAP_49_XMM"
        or discovery["scientific_payload_search_or_download_authorized"] is not False
        or discovery["fallback_order"][-1] != "AXES2MRS_CDS_REPLACEMENT"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("source discovery order or boundary changed")

    historical = config["historical_non_authoritative_evidence"]
    if (
        not isinstance(historical, list)
        or tuple(item["source_id"] for item in historical) != HISTORICAL_SOURCE_IDS
    ):
        raise GravityGroupScaleBridgeAcquisitionError("historical evidence inventory changed")
    sources = config["authoritative_sources"]
    if not isinstance(sources, list) or tuple(item["source_id"] for item in sources) != SOURCE_IDS:
        raise GravityGroupScaleBridgeAcquisitionError("authoritative source inventory changed")
    for source in [*historical, *sources]:
        _strict(source, set(FACT_FIELDS) | {"fact_binding_sha256"}, source["source_id"])
        if source["retrieved_at"] != "2026-08-29":
            raise GravityGroupScaleBridgeAcquisitionError("source retrieval date changed")
        if source["fact_binding_sha256"] != _sha(_fact_material(source)):
            raise GravityGroupScaleBridgeAcquisitionError(
                f"source fact binding changed: {source['source_id']}"
            )
    source_by_id = {item["source_id"]: item for item in sources}
    if source_by_id["XGAP_MASTER_ATTACHMENT"]["url"] != (
        "https://www.astro.unige.ch/xgap/sites/default/files/xgap_master_v1.1.fits"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("X-GAP attachment URL changed")
    if source_by_id["AXES2MRS_CDS_README"]["doi"] != "10.26093/cds/vizier.36900212":
        raise GravityGroupScaleBridgeAcquisitionError("AXES-2MRS catalogue identity changed")

    assets = config["remote_assets"]
    if not isinstance(assets, list) or tuple(item["asset_id"] for item in assets) != ASSET_IDS:
        raise GravityGroupScaleBridgeAcquisitionError("remote asset inventory changed")
    asset_by_id = {item["asset_id"]: item for item in assets}
    sample_page = asset_by_id["XGAP_SAMPLE_PAGE"]
    if (
        sample_page["observed_http_status"] != 301
        or sample_page["observed_content_length_bytes"] != 162
        or sample_page["observed_redirect"] != "https://www.unige.ch/sciences/astro/en"
        or sample_page["downloaded"] is not False
        or sample_page["payload_rows_opened"] != 0
        or sample_page["state"] != "BLOCKED_LIVE_PAGE_REDIRECT"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("X-GAP live-page finding changed")
    xgap = asset_by_id["XGAP_MASTER_V1_1"]
    if (
        xgap["observed_http_status"] != 301
        or xgap["observed_redirect"] != "https://www.unige.ch/sciences/astro/en"
        or xgap["downloaded"] is not False
        or xgap["file_sha256"] is not None
        or xgap["payload_rows_opened"] != 0
        or xgap["state"] != "BLOCKED_DEAD_REDIRECT"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("X-GAP dead-attachment finding changed")
    readme = asset_by_id["AXES2MRS_README"]
    if (
        readme["local_path"] != SOURCE_PATH.as_posix()
        or readme["observed_content_length_bytes"] != SOURCE_FILE_BYTES
        or readme["file_sha256"] != SOURCE_FILE_SHA256
        or readme["downloaded"] is not True
        or readme["payload_rows_opened"] != 0
        or readme["state"] != "FROZEN_METADATA_ONLY_README_NO_SCIENTIFIC_ROWS"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("frozen ReadMe metadata changed")
    data = asset_by_id["AXES2MRS_DATA"]
    if (
        data["url"] != "https://cdsarc.cds.unistra.fr/ftp/J/A+A/690/A212/axes2mrs.dat"
        or data["observed_content_length_bytes"] != 82584
        or data["downloaded"] is not False
        or data["local_path"] is not None
        or data["file_sha256"] is not None
        or data["payload_rows_opened"] != 0
        or data["state"] != "PUBLIC_METADATA_VERIFIED_PAYLOAD_UNOPENED"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("AXES-2MRS payload boundary changed")

    sample = config["sample_and_alias_state"]
    if sample["xgap"] != {
        "documented_final_object_count": 49,
        "primary_selection_redshift_rule": "z < 0.06",
        "primary_actual_redshift_range": "0.025 < z < 0.06",
        "legacy_selection_redshift_rule": "z < 0.05",
        "redshift_discrepancy_resolved": False,
        "alias_rows_opened": 0,
        "canonical_alias_inventory_frozen": False,
    }:
        raise GravityGroupScaleBridgeAcquisitionError("X-GAP sample state changed")
    if (
        sample["axes2mrs"]["documented_catalogue_record_count"] != 558
        or sample["axes2mrs"]["catalogue_payload_rows_opened"] != 0
        or sample["axes2mrs"]["catalogue_aliases_frozen"] is not False
        or sample["axes2mrs"]["mass_range_1e13_to_1e14_subset_frozen"] is not False
    ):
        raise GravityGroupScaleBridgeAcquisitionError("AXES-2MRS sample boundary changed")

    overlap = config["xcop_overlap_contract"]
    if (
        overlap["executed"] is not False
        or overlap["overlap_count"] is not None
        or overlap["input_alias_rows"] != 0
        or len(overlap["procedure"]) != 5
        or "Do not auto-match on sky position" not in overlap["procedure"][2 + 1]
    ):
        raise GravityGroupScaleBridgeAcquisitionError("X-COP overlap boundary changed")

    endpoint = config["endpoint_and_covariance_state"]
    for key, value in endpoint.items():
        if key != "reason" and value is not False:
            raise GravityGroupScaleBridgeAcquisitionError(
                "endpoint/covariance readiness overstated"
            )

    license_state = config["license_and_redistribution_state"]
    if (
        license_state["xgap_master_fits_per_file_license_verified"] is not False
        or license_state["axes2mrs_catalogue_per_file_license_verified"] is not False
        or license_state["redistribution_claim"] is not False
    ):
        raise GravityGroupScaleBridgeAcquisitionError("license boundary overstated")

    access = config["access_chronology"]
    required_counts = {
        "clean_metadata_manifest_downloads": 1,
        "clean_metadata_manifest_bytes": SOURCE_FILE_BYTES,
        "scientific_payload_downloads": 0,
        "scientific_payload_bytes": 0,
        "sample_alias_rows_opened": 0,
        "thermodynamic_rows_opened": 0,
        "stellar_baryon_rows_opened": 0,
        "inferred_mass_rows_opened": 0,
        "lensing_rows_opened": 0,
        "target_rows_opened": 0,
        "scores_computed": 0,
        "model_or_paid_calls": 0,
    }
    if any(access[key] != value for key, value in required_counts.items()):
        raise GravityGroupScaleBridgeAcquisitionError("access chronology changed")
    if (
        access["metadata_head_checks_performed"] is not True
        or access["interactive_metadata_call_count_exhaustive"] is not False
        or access["authorization"] is not False
    ):
        raise GravityGroupScaleBridgeAcquisitionError("access disclosure or authorization changed")

    claims = config["claim_boundary"]
    if claims["replacement_metadata_manifest_frozen"] is not True:
        raise GravityGroupScaleBridgeAcquisitionError("metadata manifest claim changed")
    for key, value in claims.items():
        if key != "replacement_metadata_manifest_frozen" and value is not False:
            raise GravityGroupScaleBridgeAcquisitionError(f"claim boundary overstated: {key}")
    if config["decision"] != (
        "PARTIAL_METADATA_MANIFEST_FROZEN_BRIDGE_BLOCKED_XGAP_ALIASES_ENDPOINTS_"
        "COVARIANCE_LICENSE_AND_OVERLAP"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("decision changed")

    publication = config["publication_contract"]
    if (
        publication["publish_primitive"] != "SAME_DIRECTORY_HARD_LINK_NO_REPLACE"
        or publication["staging_file_fsync_before_publish"] is not True
        or publication["existing_different_receipt_action"] != "FAIL_CLOSED"
        or publication["race_loser_action"] != "VERIFY_IDENTICAL_OR_FAIL_CLOSED"
    ):
        raise GravityGroupScaleBridgeAcquisitionError("publication contract weakened")
    for section, expected_sha in SECTION_BINDINGS.items():
        if _sha(config[section]) != expected_sha:
            raise GravityGroupScaleBridgeAcquisitionError(
                f"canonical nested section changed: {section}"
            )


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config_path = root / CONFIG_PATH
    if not config_path.is_file() or _file_sha(config_path) != CONFIG_FILE_SHA256:
        raise GravityGroupScaleBridgeAcquisitionError("bridge acquisition config hash changed")
    config = _read_json(config_path)
    validate_config(config)
    if _file_sha(root / TEST_PATH) != config["implementation_evidence"]["test_file_sha256"]:
        raise GravityGroupScaleBridgeAcquisitionError("bridge acquisition test binding changed")
    parent = config["parent_source_audit_binding"]
    for path_key, sha_key in (
        ("config_path", "config_file_sha256"),
        ("module_path", "module_file_sha256"),
        ("test_path", "test_file_sha256"),
        ("receipt_path", "receipt_file_sha256"),
    ):
        path = root / parent[path_key]
        if not path.is_file() or _file_sha(path) != parent[sha_key]:
            raise GravityGroupScaleBridgeAcquisitionError(f"parent binding changed: {path_key}")
    parent_receipt = _read_json(root / parent["receipt_path"])
    if parent_receipt.get("content_sha256") != parent["receipt_content_sha256"]:
        raise GravityGroupScaleBridgeAcquisitionError("parent receipt content binding changed")
    source_path = root / SOURCE_PATH
    if (
        not source_path.is_file()
        or source_path.stat().st_size != SOURCE_FILE_BYTES
        or _file_sha(source_path) != SOURCE_FILE_SHA256
    ):
        raise GravityGroupScaleBridgeAcquisitionError("frozen CDS ReadMe changed")
    return config


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    parent = config["parent_source_audit_binding"]
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
        "parent_source_audit_binding": parent,
        "source_manifest_binding": {
            "path": SOURCE_PATH.as_posix(),
            "file_sha256": _file_sha(root / SOURCE_PATH),
            "bytes": (root / SOURCE_PATH).stat().st_size,
            "content_class": "CDS_README_SCHEMA_AND_FILE_MANIFEST_NO_OBJECT_ROWS",
        },
        "authoritative_sources": config["authoritative_sources"],
        "historical_non_authoritative_evidence": config["historical_non_authoritative_evidence"],
        "remote_assets": config["remote_assets"],
        "sample_and_alias_state": config["sample_and_alias_state"],
        "xcop_overlap_contract": config["xcop_overlap_contract"],
        "endpoint_and_covariance_state": config["endpoint_and_covariance_state"],
        "license_and_redistribution_state": config["license_and_redistribution_state"],
        "access_chronology": config["access_chronology"],
        "claims": config["claim_boundary"],
        "counts": {
            "authoritative_metadata_sources": len(config["authoritative_sources"]),
            "historical_non_authoritative_evidence_records": len(
                config["historical_non_authoritative_evidence"]
            ),
            "remote_asset_records": len(config["remote_assets"]),
            "metadata_manifests_frozen": 1,
            "metadata_manifest_bytes_frozen": SOURCE_FILE_BYTES,
            "scientific_payload_downloads": 0,
            "scientific_payload_bytes": 0,
            "scientific_payload_rows_opened": 0,
            "sample_alias_rows_opened": 0,
            "target_rows_opened": 0,
            "scores_computed": 0,
            "network_calls_by_receipt_builder": 0,
            "model_or_paid_calls": 0,
            "ready_science_lanes": 0,
        },
        "limitations": [
            "The live X-GAP sample-selection page and preferred v1.1 FITS attachment both redirect to the generic UNIGE astronomy page; the FITS bytes, aliases, checksum, and per-file license remain unavailable.",
            "Exact legacy project-page selection wording is retained only from a non-archived cached search excerpt and the byte-bound parent V1 audit, not attributed to the current live page.",
            "The frozen 6278-byte CDS ReadMe is a schema/file manifest for the 558-record AXES-2MRS catalogue; axes2mrs.dat was not downloaded or opened.",
            "The ReadMe does not freeze aliases, a direct 1e13-1e14 subset, radial density/pressure/temperature/stellar-baryon endpoints, or covariance.",
            "The X-GAP legacy z<0.05 and primary z<0.06/actual 0.025-0.06 statements remain unreconciled against a canonical object manifest.",
            "Public access and article-level licenses are not treated as per-file redistribution terms for scientific payloads.",
            "Named repository searches are not proof that no other release exists, and mutable remote pages are not archived by this package.",
        ],
        "next_action": config["next_action"],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    actual = dict(receipt)
    content_sha = actual.pop("content_sha256", None)
    if content_sha != _sha(actual):
        raise GravityGroupScaleBridgeAcquisitionError("receipt content hash changed")
    expected = build_receipt(root)
    if receipt != expected:
        raise GravityGroupScaleBridgeAcquisitionError("receipt differs from frozen acquisition")


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
                raise GravityGroupScaleBridgeAcquisitionError(
                    "refusing to replace a different acquisition receipt"
                )
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    receipt = build_receipt(root)
    payload = (
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True).encode("utf-8") + b"\n"
    )
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
        path = write_receipt(args.root)
        print(path)
    else:
        receipt = _read_json(args.root.resolve() / OUTPUT_PATH)
        validate_receipt(receipt, args.root)
        print(json.dumps({"status": "PASS", "content_sha256": receipt["content_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
