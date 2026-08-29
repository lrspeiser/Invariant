"""Metadata-only independent cluster source audit and frozen transformation contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_independent_data_contract_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/independent-data-contract-v1.json")
CONFIG_SCHEMA = "invariant-gravity-cluster-independent-data-contract-1.0"
RECEIPT_SCHEMA = "invariant-gravity-cluster-independent-data-contract-receipt-1.0"
LANE_IDS = (
    "CHEX_MATE_XMM_THERMODYNAMICS",
    "LOCUSS_XRAY_AND_WEAK_LENSING",
    "INDEPENDENT_CHANDRA_THERMODYNAMICS",
    "ACT_OR_SPT_RADIAL_SZ",
    "CHEX_MATE_A302_PRESSURE_SUBSAMPLE",
    "ACT_DR6_ERASS1_PROSPECTIVE_REDUCTION",
)
SOURCE_IDS = (
    "CHEX_MATE_TEMPERATURE_METHOD",
    "CHEX_MATE_XMM_ARCHIVE",
    "LOCUSS_XRAY_ANALYSIS",
    "ACCEPT_HEASARC",
    "ACT_LAMBDA_RELEASES",
    "SPT_LAMBDA_RELEASES",
    "CHEX_MATE_A302_PRESSURE_PAPER",
    "ACT_DR6_CLUSTER_PAPER",
    "ACT_DR6_CLUSTER_RELEASE",
    "ACT_DR6_MAP_RELEASE",
    "ERASS1_CLUSTER_RELEASE",
)
READINESS_FIELDS = (
    "public_object_inventory",
    "gas_density_profile_files_verified",
    "stellar_baryon_profile_files_verified",
    "sz_pressure_profile_files_verified",
    "xray_temperature_profile_files_verified",
    "calibration_role_files_verified",
    "full_covariance_files_verified",
    "license_per_file_verified",
    "xcop_overlap_audited",
)
CALIBRATION_ROLES = (
    "background_provenance",
    "detector_or_instrument_calibration",
    "point_spread_or_beam_response",
    "selection_or_mask_provenance",
    "spectral_or_bandpass_response",
)
COVARIANCE_ROLES = (
    "calibration_covariance",
    "measurement_uncertainty_or_covariance",
)
TRANSFORMATION_IDS = (
    "ANGULAR_RADIUS_TO_KPC",
    "XRAY_SPECTRUM_TO_PROJECTED_TEMPERATURE",
    "XRAY_SURFACE_BRIGHTNESS_TO_ELECTRON_DENSITY",
    "SZ_MAP_TO_ELECTRON_PRESSURE",
    "MEMBER_LIGHT_TO_STELLAR_BARYON_PROFILE",
)


class GravityClusterDataContractError(RuntimeError):
    """Raised when the independent-source or sealed-target contract changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterDataContractError(f"{label} keys changed")


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root.resolve() / CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterDataContractError("data contract must be an object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "contract_id",
            "audit_cutoff",
            "purpose",
            "metadata_sources",
            "candidate_lanes",
            "source_manifest_required_fields",
            "role_requirements",
            "unit_and_constant_freeze",
            "cosmology_and_redshift_freeze",
            "transformations",
            "leakage_and_exclusion_freeze",
            "selection_state",
            "seals",
            "output_path",
        },
        "independent data contract",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_metadata_only"
        or config["contract_id"] != "gravity-cluster-independent-data-contract-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterDataContractError("independent data contract identity changed")

    sources = config["metadata_sources"]
    if (
        tuple(row["source_id"] for row in sources) != SOURCE_IDS
        or {row["lane_id"] for row in sources} != set(LANE_IDS)
    ):
        raise GravityClusterDataContractError("source metadata inventory changed")
    for source in sources:
        _strict(
            source,
            {"source_id", "lane_id", "source_type", "url", "doi", "audited_fact"},
            "source metadata",
        )
        if (
            source["source_type"] not in {"primary_paper", "official_archive"}
            or not str(source["url"]).startswith("https://")
            or not source["audited_fact"]
        ):
            raise GravityClusterDataContractError("source metadata is not authoritative")

    lanes = config["candidate_lanes"]
    if tuple(lane["lane_id"] for lane in lanes) != LANE_IDS:
        raise GravityClusterDataContractError("candidate lane order changed")
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
        if (
            lane["payload_opened"] is not False
            or lane["selected_role"] is not None
            or tuple(lane["readiness"]) != READINESS_FIELDS
            or not all(isinstance(value, bool) for value in lane["readiness"].values())
            or all(lane["readiness"].values())
            or "BLOCKED" not in lane["decision"]
        ):
            raise GravityClusterDataContractError("candidate lane seal or status changed")
        audit = lane["audit_details"]
        _strict(
            audit,
            {
                "observed_availability",
                "exact_missing_fields",
                "population_and_power_limitations",
                "licensing_blocker",
                "overlap_blocker",
                "covariance_blocker",
                "payload_commitment",
            },
            "candidate lane audit details",
        )
        missing_fields = [
            key for key, value in lane["readiness"].items() if not value
        ]
        if (
            not audit["observed_availability"]
            or not all(isinstance(item, str) and item for item in audit["observed_availability"])
            or audit["exact_missing_fields"] != missing_fields
            or not audit["population_and_power_limitations"]
            or not audit["licensing_blocker"]
            or not audit["overlap_blocker"]
            or not audit["covariance_blocker"]
            or audit["payload_commitment"] is not None
        ):
            raise GravityClusterDataContractError("candidate lane audit is incomplete")

    required_manifest = list(map(str, config["source_manifest_required_fields"]))
    if len(required_manifest) != 18 or len(set(required_manifest)) != 18:
        raise GravityClusterDataContractError("source manifest fields changed")
    roles = config["role_requirements"]
    if (
        tuple(roles["calibration_roles"]) != CALIBRATION_ROLES
        or tuple(roles["covariance_roles"]) != COVARIANCE_ROLES
        or roles["missing_role_action"] != "FAIL_PACKET_BEFORE_TARGET_ACCESS"
    ):
        raise GravityClusterDataContractError("source packet roles weakened")

    units = config["unit_and_constant_freeze"]
    if (
        set(units["internal_units"])
        != {
            "radius",
            "mass",
            "acceleration",
            "electron_number_density",
            "electron_pressure",
            "spectroscopic_temperature",
            "angle",
        }
        or len(units["constants"]) != 6
        or not units["source_native_scaled_profile_rule"]
        or not units["uncertainty_rule"]
    ):
        raise GravityClusterDataContractError("unit freeze changed")
    cosmology = config["cosmology_and_redshift_freeze"]
    if (
        cosmology["reference_cosmology"]
        != {
            "model": "flat_LCDM",
            "H0_km_s-1_Mpc-1": 70.0,
            "Omega_m": 0.3,
            "Omega_lambda": 0.7,
        }
        or len(cosmology["allowed_redshift_uses"]) != 5
        or len(cosmology["prohibited_redshift_uses"]) != 5
        or "candidate_formula_input" not in cosmology["prohibited_redshift_uses"]
    ):
        raise GravityClusterDataContractError("cosmology or redshift boundary changed")

    transformations = config["transformations"]
    if tuple(row["transformation_id"] for row in transformations) != TRANSFORMATION_IDS:
        raise GravityClusterDataContractError("transformation inventory changed")
    for row in transformations:
        _strict(
            row,
            {
                "transformation_id",
                "inputs",
                "outputs",
                "required_uncertainty",
                "forbidden_outputs",
            },
            "transformation",
        )
        if not row["inputs"] or not row["outputs"] or not row["forbidden_outputs"]:
            raise GravityClusterDataContractError("transformation boundary is incomplete")

    leakage = config["leakage_and_exclusion_freeze"]
    if (
        leakage["post_response_exclusions_allowed"] is not False
        or leakage["target_derived_predictors_allowed"] is not False
        or leakage["halo_labels_allowed"] is not False
        or leakage["derived_mass_truth_allowed"] is not False
        or len(leakage["forbidden_predictor_or_selection_fields"]) != 10
        or leakage["failure_action"]
        != "FAIL_CLOSED_AND_PRESERVE_ROW_FOR_DATA_QUALITY_AUDIT"
    ):
        raise GravityClusterDataContractError("leakage boundary weakened")
    selection = config["selection_state"]
    if selection != {
        "selected_primary_lane": None,
        "selected_secondary_lane": None,
        "authorized_target_packets": [],
        "independent_target_rows_opened": 0,
        "observational_authorization": False,
    }:
        raise GravityClusterDataContractError("independent target seal changed")
    if any(config["seals"].values()):
        raise GravityClusterDataContractError("metadata-only seal changed")


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    lanes = config["candidate_lanes"]
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "contract_id": config["contract_id"],
        "decision": "SOURCE_AUDIT_COMPLETE_SELECTION_BLOCKED_TARGETS_SEALED",
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "source_audit": [
            {
                "lane_id": lane["lane_id"],
                "decision": lane["decision"],
                "verified_readiness_fields": [
                    key for key, value in lane["readiness"].items() if value
                ],
                "blocking_readiness_fields": [
                    key for key, value in lane["readiness"].items() if not value
                ],
                "audit_details": lane["audit_details"],
            }
            for lane in lanes
        ],
        "completed_goal_evidence": {
            "CP3.7": "units_cosmology_distance_transformations_and_redshift_uses_frozen",
            "CP3.8": "target_derived_halo_mass_and_post_response_leakage_fail_closed",
            "CP7.1": "six_candidate_source_lanes_audited_from_primary_or_official_metadata",
        },
        "blocked_goal_evidence": {
            "CP3.5": "no_selected_independent_lane_or_file_level_payload_manifest",
            "CP3.6": "no_real_source_packet_with_all_calibration_and_covariance_roles",
            "CP7.2": "no_lane_satisfies_all_required_direct_observable_and_covariance_fields",
            "CP7.3": "xcop_overlap_not_audited_for_any_candidate_lane",
            "CP7.9": "no_source_lane_has_a_metadata_only_payload_commitment_or_file_receipt",
        },
        "gate_status": {"CP3": "PARTIAL", "CP7": "PARTIAL"},
        "counts": {
            "metadata_sources": len(config["metadata_sources"]),
            "candidate_lanes": len(lanes),
            "fully_ready_lanes": sum(all(lane["readiness"].values()) for lane in lanes),
            "selected_lanes": sum(lane["selected_role"] is not None for lane in lanes),
            "payloads_opened": sum(lane["payload_opened"] for lane in lanes),
            "target_rows_opened": config["selection_state"]["independent_target_rows_opened"],
            "transformations": len(config["transformations"]),
            "manifest_required_fields": len(config["source_manifest_required_fields"]),
        },
        "claims": {
            "source_metadata_audit_complete": True,
            "independent_source_selected": False,
            "independent_data_ready": False,
            "observational_authorization": False,
            "payload_accessed": False,
            "target_rows_accessed": False,
            "scientific_result_emitted": False,
        },
        "next_action": "Resolve file-level direct-observable releases or frozen reduction manifests, eligible population and power, full covariance, per-file licenses, and X-COP overlap before selecting any lane or committing a payload.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterDataContractError("independent data receipt changed")


def write_receipt(root: Path) -> Path:
    path = root.resolve() / OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_bytes(build_receipt(root)))
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
        receipt = json.loads((root / OUTPUT_PATH).read_text(encoding="utf-8"))
        validate_receipt(receipt, root)
        output = {"status": "PASS", "content_sha256": receipt["content_sha256"]}
    else:
        receipt = build_receipt(root)
        output = {
            "decision": receipt["decision"],
            "claims": receipt["claims"],
            "next_action": receipt["next_action"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
