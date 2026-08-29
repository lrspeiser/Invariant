"""Build the metadata-only, target-blind CP10 group-scale source audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_group_scale_source_audit_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/group-scale-source-audit-v1.json")
MODULE_PATH = Path("src/sigma_theory_compiler/gravity_group_scale_source_audit.py")
TEST_PATH = Path("tests/test_gravity_group_scale_source_audit.py")
CONFIG_FILE_SHA256 = "65b37105b66d0d97db4bc02d3c3eeeb1beafa429350429b7aa899943ffb2aaf9"
TEST_FILE_SHA256 = "3f5689428cd0b9e58bec8b46894d32d3bc4903bb08eee73b2fb478d3f669a069"
CONFIG_SCHEMA = "invariant-gravity-group-scale-source-audit-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-group-scale-source-audit-receipt-1.0"

SOURCE_IDS = (
    "XGAP_SAMPLE_SELECTION",
    "XGAP_MASTER_ATTACHMENT",
    "XGAP_OBSERVATIONAL_STRATEGY",
    "XGAP_PRIMARY_PAPER",
    "XGAP_2026_SCALING_RELATIONS",
    "XGAP_2026_FORWARD_MODEL",
    "XMM_SCIENCE_ARCHIVE",
    "XMM_CALIBRATION_DOCUMENTATION",
    "SUN09_PRIMARY_PAPER",
    "ERASS1_DR1_ARCHIVE",
    "ERASS1_GROUP_CATALOG_RELEASE",
    "ERASS1_GROUP_THERMODYNAMICS_PAPER",
)
SOURCE_FACT_BINDINGS = {
    "XGAP_SAMPLE_SELECTION": "d83b627744e610390c7bf00ab3e03821c4b5c6de8b448fd4b0e8c24286f3c12f",
    "XGAP_MASTER_ATTACHMENT": "180c26f03f6e54c566332b7af15f1efed09df7c9d2396e019a6efdd86f713946",
    "XGAP_OBSERVATIONAL_STRATEGY": "36bc40c1f614dd5cc3cbb21379be3ef147cdd0dd3ac898fb4c30bb5091c7c00f",
    "XGAP_PRIMARY_PAPER": "8f5eed2b354ffe1d6c82bad7f4408db267392a11b873bcc1f017ba4c5c985b35",
    "XGAP_2026_SCALING_RELATIONS": "6eb9ce7ee0d4bccf6486af4014987f1933fbcbea4258bf39340804a5c56bed41",
    "XGAP_2026_FORWARD_MODEL": "21f923dd5077934dda17cfba25f1733eaa908e5fb33b06446df89e055c3f5ff8",
    "XMM_SCIENCE_ARCHIVE": "1c2009fde3e472d45e02afefeb8433217e90d66e2681fea794f57f1c4c8a66dd",
    "XMM_CALIBRATION_DOCUMENTATION": "01ab21a841f29c33bb45326a692c2328d2e87b0557fc3f36eef1f2e90f51e113",
    "SUN09_PRIMARY_PAPER": "2596de13876af09fd98c8b727cb40c265907112009f287b75846c319249938c7",
    "ERASS1_DR1_ARCHIVE": "12b3ace1fea8b2f11788e7bb42bed95d29cf991c20367a5a55a5bfa459063e15",
    "ERASS1_GROUP_CATALOG_RELEASE": "cbbdd13fce5165c90507341c8cd4b25f175c2fa60165ad10173432612c8733e8",
    "ERASS1_GROUP_THERMODYNAMICS_PAPER": "31156a3aef66025a47f4a636a24304353e8b222e060d9ca763e61fed9f98a826",
}
LANE_IDS = (
    "XGAP_49_XMM",
    "SUN09_CHANDRA_43",
    "ERASS1_STACKED_GROUP_THERMODYNAMICS",
)
MANIFEST_IDS = (
    "XGAP_MASTER_V1_1",
    "XGAP_FULL_SAMPLE_THERMODYNAMIC_PRODUCTS",
    "XGAP_STELLAR_BARYON_PRODUCTS",
    "ERASS1_PUBLIC_CATALOG_AND_EVENTS",
)
READINESS_FIELDS = (
    "public_sample_scope_documented",
    "machine_readable_sample_manifest_documented",
    "machine_readable_sample_manifest_currently_retrievable",
    "stable_alias_map_frozen",
    "mass_scope_1e13_1e14_documented",
    "density_endpoint_files_verified",
    "pressure_endpoint_files_verified",
    "temperature_endpoint_files_verified",
    "stellar_baryon_endpoint_files_verified",
    "within_endpoint_covariance_verified",
    "cross_endpoint_covariance_verified",
    "calibration_and_background_roles_verified",
    "per_file_reuse_terms_verified",
    "xcop_overlap_audited",
    "target_blind_whole_group_split_committable",
    "no_inferred_mass_or_lensing_target_dependency",
    "payload_commitment_frozen",
)
ENDPOINT_IDS = (
    "ELECTRON_DENSITY_RADIAL",
    "ELECTRON_PRESSURE_RADIAL",
    "SPECTROSCOPIC_TEMPERATURE_RADIAL",
    "STELLAR_BARYON_CUMULATIVE",
)
COVARIANCE_ROLES = (
    "density_cross_radius_deprojection_covariance",
    "temperature_cross_radius_spectral_covariance",
    "pressure_cross_radius_covariance_or_joint_density_temperature_draws",
    "stellar_profile_membership_photometry_IMF_and_diffuse_light_covariance",
    "shared_Xray_background_abundance_and_calibration_covariance",
    "PSF_ARF_RMF_and_deprojection_model_covariance",
    "cross_endpoint_density_temperature_pressure_covariance",
)
CALIBRATION_ROLES = (
    "observation_ids_dates_instruments_modes_filters_and_exposures",
    "pipeline_software_and_exact_calibration_database_revision",
    "flare_filtering_masks_point_source_and_background_provenance",
    "ARF_RMF_PSF_mixing_and_spectral_abundance_model",
    "surface_brightness_emissivity_deprojection_and_geometry",
    "optical_IR_catalog_release_zeropoint_extinction_membership_and_IMF",
    "radius_cosmology_redshift_and_unit_transformations",
)


class GravityGroupScaleSourceAuditError(RuntimeError):
    """Raised when the metadata-only group-scale contract changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_fact_material(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: source[key]
        for key in (
            "source_id",
            "lane_id",
            "source_type",
            "url",
            "doi",
            "observed_release",
            "audited_fact",
            "observation_method",
            "retrieved_at",
            "evidence_locator",
        )
    }


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityGroupScaleSourceAuditError(f"{label} keys changed")


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityGroupScaleSourceAuditError(f"expected JSON object: {path}")
    return value


def load_config(root: Path) -> dict[str, Any]:
    root = root.resolve()
    path = root / CONFIG_PATH
    if _file_sha(path) != CONFIG_FILE_SHA256:
        raise GravityGroupScaleSourceAuditError("group-scale source-audit config hash changed")
    value = _read_json(path)
    validate_config(value)
    evidence = value["implementation_evidence"]
    test_path = root / evidence["test_path"]
    if not test_path.is_file() or _file_sha(test_path) != evidence["test_file_sha256"]:
        raise GravityGroupScaleSourceAuditError("group-scale source-audit test binding changed")
    return value


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
            "fact_provenance_contract",
            "interactive_audit_disclosure",
            "xgap_redshift_reconciliation",
            "publication_contract",
            "mass_scope",
            "authoritative_sources",
            "file_manifest_audit",
            "readiness_fields",
            "candidate_lanes",
            "direct_endpoint_contract",
            "required_covariance_roles",
            "required_calibration_roles",
            "alias_overlap_contract",
            "whole_group_split_contract",
            "leakage_and_access_contract",
            "selection_state",
            "claim_boundary",
            "output_path",
        },
        "group-scale source audit",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only_zero_ready_lanes_with_disclosed_excerpt"
        or config["audit_id"] != "gravity-group-scale-source-audit-v1"
        or config["audit_cutoff"] != "2026-08-29"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityGroupScaleSourceAuditError("group-scale audit identity changed")

    evidence = config["implementation_evidence"]
    if evidence != {
        "module_path": MODULE_PATH.as_posix(),
        "test_path": TEST_PATH.as_posix(),
        "test_file_sha256": TEST_FILE_SHA256,
        "hash_mode": "SHA256_RAW_BYTES",
        "test_binding_required": True,
    }:
        raise GravityGroupScaleSourceAuditError("implementation evidence changed")
    provenance = config["fact_provenance_contract"]
    if provenance != {
        "binding_algorithm": "SHA256_CANONICAL_JSON",
        "binding_fields": [
            "source_id",
            "lane_id",
            "source_type",
            "url",
            "doi",
            "observed_release",
            "audited_fact",
            "observation_method",
            "retrieved_at",
            "evidence_locator",
        ],
        "retrieved_at_precision": "UTC_DAY",
        "remote_content_archived": False,
        "remote_content_sha256_available": False,
        "claim_limit": (
            "The fact binding detects local factual-field mutation but does not prove that "
            "mutable remote pages retain their observed content."
        ),
    }:
        raise GravityGroupScaleSourceAuditError("fact provenance contract changed")
    disclosure = config["interactive_audit_disclosure"]
    if disclosure != {
        "incident_id": "XGAP_2026_FORWARD_MODEL_WEB_FIND_EXCERPT_2026_08_29",
        "disclosed": True,
        "source_id": "XGAP_2026_FORWARD_MODEL",
        "observation_method": "web_find_excerpt_returned_with_metadata_query",
        "retrieved_at": "2026-08-29",
        "public_appendix_measurement_rows_rendered": 1,
        "downloaded": False,
        "persisted_in_package": False,
        "used_for_fact_or_lane_readiness": False,
        "scored": False,
        "receipt_builder_involved": False,
        "consequence": ("INTERACTIVE_ZERO_ROW_PURITY_FALSE_BUILDER_ZERO_ROW_COUNTS_REMAIN_TRUE"),
    }:
        raise GravityGroupScaleSourceAuditError("interactive audit disclosure changed")
    redshift = config["xgap_redshift_reconciliation"]
    if redshift != {
        "legacy_project_page_source_id": "XGAP_SAMPLE_SELECTION",
        "legacy_project_selection_rule": "z < 0.05",
        "primary_2024_source_id": "XGAP_PRIMARY_PAPER",
        "primary_2024_program_selection_rule": "z < 0.06",
        "primary_2024_reported_actual_range": "0.025 < z < 0.06",
        "discrepancy_resolved": False,
        "payload_selection_rule_chosen": None,
        "required_resolution": (
            "Reconcile the retrievable canonical 49-object manifest against the 2024 "
            "primary selection and actual-range statements before aliases, eligibility, "
            "or splits are frozen."
        ),
        "failure_action": "BLOCK_SAMPLE_ASSEMBLY_AND_SPLIT",
    }:
        raise GravityGroupScaleSourceAuditError("X-GAP redshift reconciliation changed")
    publication = config["publication_contract"]
    if publication != {
        "staging_location": "OUTPUT_SAME_DIRECTORY",
        "staging_file_fsync_before_publish": True,
        "publish_primitive": "SAME_DIRECTORY_HARD_LINK_NO_REPLACE",
        "directory_fsync": "POSIX_WHEN_SUPPORTED_AFTER_PUBLISH_AND_TEMP_CLEANUP",
        "existing_identical_receipt_action": "RETURN_WITHOUT_REWRITE",
        "existing_different_receipt_action": "FAIL_CLOSED",
        "race_loser_action": "VERIFY_IDENTICAL_OR_FAIL_CLOSED",
    }:
        raise GravityGroupScaleSourceAuditError("publication contract changed")

    mass = config["mass_scope"]
    if mass != {
        "lower_m500_msun": 10_000_000_000_000.0,
        "upper_m500_msun": 100_000_000_000_000.0,
        "selection_role_only": True,
        "mass_as_response_or_candidate_input": False,
        "rule": (
            "A source may document membership in the approximate group-scale interval, "
            "but source-inferred hydrostatic, lensing, dynamical, richness, luminosity-scaling, "
            "or count-rate-scaling mass may not be loaded as a response or candidate input."
        ),
    }:
        raise GravityGroupScaleSourceAuditError("group mass boundary changed")

    sources = config["authoritative_sources"]
    if tuple(row.get("source_id") for row in sources) != SOURCE_IDS:
        raise GravityGroupScaleSourceAuditError("authoritative source order changed")
    if {row.get("lane_id") for row in sources} != set(LANE_IDS):
        raise GravityGroupScaleSourceAuditError("authoritative source lanes changed")
    for source in sources:
        _strict(
            source,
            {
                "source_id",
                "lane_id",
                "source_type",
                "url",
                "doi",
                "observed_release",
                "audited_fact",
                "observation_method",
                "retrieved_at",
                "evidence_locator",
                "fact_binding_sha256",
            },
            "authoritative source",
        )
        if (
            source["source_type"]
            not in {
                "primary_paper",
                "official_project_page",
                "official_project_manifest",
                "official_archive",
                "official_calibration",
            }
            or not str(source["url"]).startswith("https://")
            or not source["observed_release"]
            or not source["audited_fact"]
            or not source["observation_method"]
            or source["retrieved_at"] != "2026-08-29"
            or not source["evidence_locator"]
            or source["fact_binding_sha256"] != _sha(_source_fact_material(source))
            or source["fact_binding_sha256"] != SOURCE_FACT_BINDINGS[source["source_id"]]
        ):
            raise GravityGroupScaleSourceAuditError("source authority weakened")

    manifests = config["file_manifest_audit"]
    if tuple(row.get("manifest_id") for row in manifests) != MANIFEST_IDS:
        raise GravityGroupScaleSourceAuditError("file manifest order changed")
    for manifest in manifests:
        _strict(
            manifest,
            {
                "manifest_id",
                "lane_id",
                "evidence_source_ids",
                "role",
                "url",
                "reported_filename",
                "reported_size",
                "file_sha256",
                "license_for_file_verified",
                "current_retrieval_state",
                "payload_opened",
            },
            "file manifest",
        )
        if (
            manifest["lane_id"] not in LANE_IDS
            or not manifest["evidence_source_ids"]
            or not set(manifest["evidence_source_ids"]).issubset(SOURCE_IDS)
            or manifest["file_sha256"] is not None
            or manifest["license_for_file_verified"] is not False
            or manifest["payload_opened"] is not False
            or not manifest["current_retrieval_state"]
            or (manifest["url"] is not None and not str(manifest["url"]).startswith("https://"))
        ):
            raise GravityGroupScaleSourceAuditError("file manifest overclaimed readiness")
    xgap_manifest = manifests[0]
    if (
        xgap_manifest["reported_filename"] != "xgap_master_v1.1.fits"
        or xgap_manifest["reported_size"] != "25.31 KB"
        or xgap_manifest["current_retrieval_state"]
        != "BLOCKED_LEGACY_ATTACHMENT_REDIRECTS_TO_GENERIC_DEPARTMENT_PAGE"
    ):
        raise GravityGroupScaleSourceAuditError("X-GAP manifest observation changed")

    if tuple(config["readiness_fields"]) != READINESS_FIELDS:
        raise GravityGroupScaleSourceAuditError("readiness field order changed")
    lanes = config["candidate_lanes"]
    if tuple(lane.get("lane_id") for lane in lanes) != LANE_IDS:
        raise GravityGroupScaleSourceAuditError("candidate lane order changed")
    for lane in lanes:
        _strict(
            lane,
            {
                "lane_id",
                "sample_scope",
                "instrument_family",
                "payload_opened",
                "selected_role",
                "readiness",
                "audit_details",
                "decision",
            },
            "candidate lane",
        )
        readiness = lane["readiness"]
        audit = lane["audit_details"]
        if (
            lane["payload_opened"] is not False
            or lane["selected_role"] is not None
            or tuple(readiness) != READINESS_FIELDS
            or not all(isinstance(value, bool) for value in readiness.values())
            or all(readiness.values())
            or "BLOCKED" not in lane["decision"]
        ):
            raise GravityGroupScaleSourceAuditError("candidate lane seal changed")
        _strict(
            audit,
            {
                "observed_availability",
                "evidence_source_ids",
                "exact_missing_fields",
                "alias_and_overlap_blocker",
                "endpoint_blocker",
                "covariance_blocker",
                "calibration_blocker",
                "licensing_blocker",
                "population_limitation",
                "payload_commitment",
            },
            "lane audit details",
        )
        missing = [field for field, ready in readiness.items() if not ready]
        if (
            audit["exact_missing_fields"] != missing
            or not audit["evidence_source_ids"]
            or not set(audit["evidence_source_ids"]).issubset(SOURCE_IDS)
            or not audit["observed_availability"]
            or not all(isinstance(item, str) and item for item in audit["observed_availability"])
            or not all(
                audit[key]
                for key in (
                    "alias_and_overlap_blocker",
                    "endpoint_blocker",
                    "covariance_blocker",
                    "calibration_blocker",
                    "licensing_blocker",
                    "population_limitation",
                )
            )
            or audit["payload_commitment"] is not None
        ):
            raise GravityGroupScaleSourceAuditError("lane blockers are incomplete")
    xgap, sun09, erass = lanes
    if (
        "49" not in xgap["sample_scope"]
        or "1e13 < M500 < 1e14" not in xgap["sample_scope"]
        or "legacy page says z < 0.05" not in xgap["sample_scope"]
        or "2024 primary says z < 0.06" not in xgap["sample_scope"]
        or "actual 0.025 < z < 0.06" not in xgap["sample_scope"]
        or "redshift discrepancy is unresolved"
        not in xgap["audit_details"]["alias_and_overlap_blocker"]
        or xgap["readiness"]["machine_readable_sample_manifest_currently_retrievable"] is not False
        or "43" not in sun09["sample_scope"]
        or "1178" not in erass["sample_scope"]
        or "271" not in erass["sample_scope"]
        or erass["readiness"]["machine_readable_sample_manifest_documented"] is not False
        or erass["readiness"]["machine_readable_sample_manifest_currently_retrievable"] is not False
        or erass["readiness"]["no_inferred_mass_or_lensing_target_dependency"] is not False
    ):
        raise GravityGroupScaleSourceAuditError("group lane factual boundary changed")

    endpoints = config["direct_endpoint_contract"]
    if tuple(row.get("endpoint_id") for row in endpoints) != ENDPOINT_IDS:
        raise GravityGroupScaleSourceAuditError("direct endpoint order changed")
    for endpoint in endpoints:
        _strict(
            endpoint,
            {
                "endpoint_id",
                "quantity",
                "required_source",
                "allowed_derivation",
                "forbidden_substitutes",
            },
            "direct endpoint",
        )
        if (
            not endpoint["quantity"]
            or not endpoint["required_source"]
            or not endpoint["allowed_derivation"]
            or not endpoint["forbidden_substitutes"]
        ):
            raise GravityGroupScaleSourceAuditError("direct endpoint weakened")
    if "density-temperature cross-covariance" not in endpoints[1]["allowed_derivation"]:
        raise GravityGroupScaleSourceAuditError("pressure covariance rule changed")
    if tuple(config["required_covariance_roles"]) != COVARIANCE_ROLES:
        raise GravityGroupScaleSourceAuditError("covariance roles changed")
    if tuple(config["required_calibration_roles"]) != CALIBRATION_ROLES:
        raise GravityGroupScaleSourceAuditError("calibration roles changed")

    alias = config["alias_overlap_contract"]
    _strict(
        alias,
        {
            "canonical_alias_fields",
            "normalization_order",
            "xcop_overlap_reference",
            "xcop_overlap_audited",
            "cross_lane_overlap_audited",
            "missing_alias_action",
            "post_response_alias_reassignment_allowed",
        },
        "alias overlap contract",
    )
    if (
        tuple(alias["canonical_alias_fields"])
        != ("lane_id", "source_object_id", "ra_deg", "dec_deg", "redshift")
        or len(alias["normalization_order"]) != 5
        or alias["xcop_overlap_audited"] is not False
        or alias["cross_lane_overlap_audited"] is not False
        or alias["missing_alias_action"] != "FAIL_BEFORE_SPLIT_OR_PAYLOAD_ACCESS"
        or alias["post_response_alias_reassignment_allowed"] is not False
    ):
        raise GravityGroupScaleSourceAuditError("alias or overlap seal changed")

    split = config["whole_group_split_contract"]
    _strict(
        split,
        {
            "algorithm",
            "namespace",
            "canonical_key",
            "bucket_derivation",
            "assignments",
            "pre_split_exclusions",
            "response_or_target_fields_used",
            "whole_group_rule",
            "post_response_movement_allowed",
            "split_execution_authorized",
        },
        "whole-group split contract",
    )
    if (
        split["algorithm"] != "SHA256_UTF8_PREFIX_BUCKET"
        or split["namespace"] != "invariant-group-scale-split-v1|"
        or split["assignments"]
        != {
            "development_train": [0, 1, 2, 3, 4, 5],
            "development_holdout": [6, 7],
            "confirmation": [8, 9],
        }
        or sorted(bucket for values in split["assignments"].values() for bucket in values)
        != list(range(10))
        or split["response_or_target_fields_used"] != []
        or split["post_response_movement_allowed"] is not False
        or split["split_execution_authorized"] is not False
    ):
        raise GravityGroupScaleSourceAuditError("target-blind split rule changed")

    leakage = config["leakage_and_access_contract"]
    _strict(
        leakage,
        {
            "allowed_before_authorization",
            "forbidden_before_authorization",
            "forbidden_candidate_inputs",
            "failure_action",
            "paid_or_model_calls_allowed",
        },
        "leakage and access contract",
    )
    if (
        "thermodynamic_response_rows" not in leakage["forbidden_before_authorization"]
        or "lensing_shear_or_mass_rows" not in leakage["forbidden_before_authorization"]
        or "hydrostatic_mass" not in leakage["forbidden_candidate_inputs"]
        or leakage["failure_action"] != "FAIL_CLOSED_BEFORE_PAYLOAD_LOAD"
        or leakage["paid_or_model_calls_allowed"] is not False
    ):
        raise GravityGroupScaleSourceAuditError("payload or leakage boundary weakened")

    selection = config["selection_state"]
    if selection != {
        "selected_lane": None,
        "counts_scope": "RECEIPT_BUILDER_ONLY",
        "sample_alias_inventory_frozen": False,
        "payload_manifest_committed": False,
        "split_executed": False,
        "observational_authorization": False,
        "payload_rows_opened": 0,
        "thermodynamic_rows_opened": 0,
        "stellar_baryon_rows_opened": 0,
        "inferred_mass_rows_opened": 0,
        "lensing_rows_opened": 0,
        "scientific_scores_computed": 0,
        "paid_or_model_calls": 0,
        "downloads": 0,
        "downloaded_bytes": 0,
    }:
        raise GravityGroupScaleSourceAuditError("selection or access seal changed")
    claims = config["claim_boundary"]
    if claims != {
        "CP10_1_complete": False,
        "CP10_2_complete": False,
        "source_metadata_audit_complete": True,
        "interactive_audit_zero_row_purity": False,
        "receipt_builder_zero_row_purity": True,
        "direct_endpoint_contract_frozen": True,
        "target_blind_split_algorithm_frozen": True,
        "public_lane_ready": False,
        "group_sample_assembled": False,
        "group_splits_frozen": False,
        "scientific_data_ready": False,
        "candidate_tested_on_groups": False,
        "domain_boundary_measured": False,
        "publication_claim_supported": False,
        "central_readiness_changed": False,
    }:
        raise GravityGroupScaleSourceAuditError("claim boundary changed")


def split_bucket(canonical_group_key: str) -> tuple[int, str]:
    """Return the frozen metadata-key bucket and split without reading response values."""

    if not canonical_group_key or canonical_group_key.count("|") != 4:
        raise GravityGroupScaleSourceAuditError("canonical group key is incomplete")
    digest = hashlib.sha256(
        f"invariant-group-scale-split-v1|{canonical_group_key}".encode()
    ).digest()
    bucket = int.from_bytes(digest[:8], "big", signed=False) % 10
    if bucket <= 5:
        split = "development_train"
    elif bucket <= 7:
        split = "development_holdout"
    else:
        split = "confirmation"
    return bucket, split


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    lanes = config["candidate_lanes"]
    selection = config["selection_state"]
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": config["audit_id"],
        "decision": (
            "METADATA_AUDIT_SEALED_DISCLOSED_EXCERPT_ZERO_READY_LANES_CP10_1_CP10_2_REMAIN_OPEN"
        ),
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
            "test_config_binding_matches": (
                _file_sha(root / TEST_PATH) == config["implementation_evidence"]["test_file_sha256"]
            ),
        },
        "fact_provenance_contract": config["fact_provenance_contract"],
        "authoritative_source_facts": config["authoritative_sources"],
        "interactive_audit_disclosure": config["interactive_audit_disclosure"],
        "xgap_redshift_reconciliation": config["xgap_redshift_reconciliation"],
        "publication_contract": config["publication_contract"],
        "mass_scope": config["mass_scope"],
        "source_audit": [
            {
                "lane_id": lane["lane_id"],
                "decision": lane["decision"],
                "verified_readiness_fields": [
                    field for field, ready in lane["readiness"].items() if ready
                ],
                "blocking_readiness_fields": [
                    field for field, ready in lane["readiness"].items() if not ready
                ],
                "audit_details": lane["audit_details"],
            }
            for lane in lanes
        ],
        "file_manifest_audit": config["file_manifest_audit"],
        "direct_endpoint_contract": [
            {
                "endpoint_id": endpoint["endpoint_id"],
                "quantity": endpoint["quantity"],
                "allowed_derivation": endpoint["allowed_derivation"],
            }
            for endpoint in config["direct_endpoint_contract"]
        ],
        "covariance_and_calibration_gate": {
            "required_covariance_roles": config["required_covariance_roles"],
            "required_calibration_roles": config["required_calibration_roles"],
            "all_roles_verified_for_any_lane": False,
        },
        "alias_and_split_preflight": {
            "canonical_alias_fields": config["alias_overlap_contract"]["canonical_alias_fields"],
            "xcop_overlap_audited": config["alias_overlap_contract"]["xcop_overlap_audited"],
            "cross_lane_overlap_audited": config["alias_overlap_contract"][
                "cross_lane_overlap_audited"
            ],
            "split_algorithm": config["whole_group_split_contract"]["algorithm"],
            "split_assignments": config["whole_group_split_contract"]["assignments"],
            "response_or_target_fields_used": [],
            "split_executed": False,
            "reason_not_executed": "sample aliases, overlap, endpoint roles, and payload manifest are not frozen",
        },
        "CP10_status": {
            "CP10.1": "ADVANCED_METADATA_SOURCE_AUDIT_ONLY_NOT_COMPLETE",
            "CP10.2": "ADVANCED_ENDPOINT_AND_SPLIT_CONTRACT_ONLY_NOT_COMPLETE",
        },
        "counts": {
            "scope": selection["counts_scope"],
            "authoritative_metadata_sources": len(config["authoritative_sources"]),
            "candidate_lanes": len(lanes),
            "ready_lanes": sum(all(lane["readiness"].values()) for lane in lanes),
            "file_manifest_records": len(config["file_manifest_audit"]),
            "direct_endpoint_roles": len(config["direct_endpoint_contract"]),
            "covariance_roles": len(config["required_covariance_roles"]),
            "calibration_roles": len(config["required_calibration_roles"]),
            "payload_rows_opened": selection["payload_rows_opened"],
            "thermodynamic_rows_opened": selection["thermodynamic_rows_opened"],
            "stellar_baryon_rows_opened": selection["stellar_baryon_rows_opened"],
            "inferred_mass_rows_opened": selection["inferred_mass_rows_opened"],
            "lensing_rows_opened": selection["lensing_rows_opened"],
            "scientific_scores_computed": selection["scientific_scores_computed"],
            "downloads": selection["downloads"],
            "downloaded_bytes": selection["downloaded_bytes"],
            "network_calls_by_receipt_builder": 0,
            "paid_or_model_calls": selection["paid_or_model_calls"],
        },
        "claims": {
            **config["claim_boundary"],
            "observational_authorization": selection["observational_authorization"],
            "payload_accessed": False,
            "scientific_result_emitted": False,
        },
        "limitations": [
            "This package records official metadata observed by 2026-08-29; it does not cache or redistribute source payloads.",
            "One public appendix measurement row was rendered incidentally by an interactive web-find excerpt; it was not downloaded, persisted, used for readiness, or scored, so interactive zero-row purity is false while receipt-builder counts remain zero.",
            "The legacy project page states z < 0.05, while the 2024 primary states z < 0.06 and reports 0.025 < z < 0.06; no eligibility rule is chosen until a canonical manifest reconciles the discrepancy.",
            "The legacy X-GAP master filename and reported size are documented, but current bytes and a file hash are unavailable because the attachment redirects.",
            "The receipt builder opened no aliases, thermodynamic rows, stellar-baryon rows, inferred-mass rows, lensing rows, or candidate scores.",
            "A frozen split algorithm is not a frozen split assignment until canonical aliases and overlap exclusions are sealed.",
            "Public raw observations and article-level licenses do not establish a complete licensed derived-product and covariance packet.",
        ],
        "next_action": (
            "Obtain an official retrievable X-GAP alias manifest or replacement receipt, freeze its hash and "
            "per-file terms, reconcile X-COP/cross-lane aliases without response values, and require a complete "
            "density-pressure-temperature-stellar packet with covariance and calibration roles before executing "
            "the frozen whole-group split or requesting payload authorization."
        ),
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityGroupScaleSourceAuditError("group-scale source-audit receipt changed")


def _fsync_directory(directory: Path) -> None:
    """Flush directory metadata where the platform exposes directory descriptors."""

    if os.name == "nt":
        return
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_publish_no_replace(path: Path, payload: bytes) -> None:
    """Publish complete fsynced bytes atomically without replacing an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        temporary_path.unlink(missing_ok=True)
        _fsync_directory(path.parent)


def write_receipt(root: Path) -> Path:
    root = root.resolve()
    path = root / OUTPUT_PATH
    payload = _canonical_bytes(build_receipt(root))
    if path.exists():
        if path.read_bytes() != payload:
            raise GravityGroupScaleSourceAuditError("refusing to overwrite changed receipt")
        return path
    try:
        _atomic_publish_no_replace(path, payload)
    except FileExistsError:
        if not path.is_file() or path.read_bytes() != payload:
            raise GravityGroupScaleSourceAuditError(
                "receipt publication race found different existing bytes"
            ) from None
    return path


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        output: Any = str(write_receipt(root))
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "CP10_status": receipt["CP10_status"],
            "claims": receipt["claims"],
            "next_action": receipt["next_action"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
