"""Validate the metadata-only A1795 covariance source-feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_a1795_covariance_source_feasibility_v1.json")
OUTPUT_PATH = Path(
    "runs/gravity/publication-readiness/a1795-covariance-source-feasibility-v1.json"
)
IMPLEMENTATION_PATH = Path(
    "src/sigma_theory_compiler/gravity_cluster_a1795_covariance_source_feasibility.py"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-a1795-covariance-source-feasibility-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-a1795-covariance-source-feasibility-receipt-1.0"
)
AUDIT_ID = "gravity-cluster-a1795-covariance-source-feasibility-v1"
DECISION = "BLOCKED_NO_COMPLETE_PUBLIC_COVARIANCE_SOURCE_PACKET"
OBSERVATION_IDS = (
    "0097820101",
    "0109070201",
    "0205190101",
    "0205190201",
    "0744412001",
    "0744412101",
)
PPS_VERSION = "21.51_20241115_1113"
SAS_VERSION = "xmmsas_20241108_1150-21.0.0"
CP_STATUS = {
    "CP5.2": "BLOCKED_TEMPERATURE_COVARIANCE_NOT_RECONSTRUCTIBLE_FROM_PUBLIC_PACKET",
    "CP5.3": "BLOCKED_DENSITY_COVARIANCE_NOT_RECONSTRUCTIBLE_FROM_PUBLIC_PACKET",
    "CP5.4": "BLOCKED_SHARED_CALIBRATION_COVARIANCE_NOT_PUBLICLY_SPECIFIED",
    "CP5.5": "BLOCKED_MATCHED_BACKGROUND_BEAM_SIMULATION_ENSEMBLE_NOT_PUBLIC",
    "CP5.6": "BLOCKED_JOINT_XRAY_SZ_CALIBRATION_COVARIANCE_NOT_PUBLIC",
}
PLANCK_PRODUCTS = {
    "COM_CompMap_Compton-SZMap_R2.02.tgz": 12252877791,
    "HFI_RIMO_R2.00.fits": 17199360,
    "HFI_RIMO_Beams-075pc_R2.00.fits": 19097280,
    "HFI_RIMO_Beams-100pc_R2.00.fits": 19097280,
    "COM_Mask_Compton-SZMap_2048_R2.00.fits": 1006643520,
}
REQUIRED_MISSING_IDS = {
    "A1795_XRAY_TEMPERATURE_LIKELIHOOD_OR_CHAINS",
    "A1795_XRAY_DENSITY_DEPROJECTION_REALIZATIONS",
    "A1795_SHARED_XMM_CALIBRATION_NUISANCE_ENSEMBLE",
    "A1795_XCOP_LOCAL_MILCA_7ARCMIN_YMAP",
    "A1795_XCOP_MATCHED_NOISE_REALIZATION_MAPS",
    "A1795_XCOP_PRESSURE_DECONVOLUTION_DEPROJECTION_OPERATOR",
    "A1795_JOINT_XRAY_SZ_CALIBRATION_COVARIANCE",
}


class GravityClusterA1795CovarianceSourceFeasibilityError(RuntimeError):
    """Raised when the frozen source-feasibility evidence changes."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8") + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            f"expected JSON object: {path}"
        )
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            f"{label} keys changed"
        )


def _under(root: Path, relative: Path, label: str) -> Path:
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            f"{label} escaped repository root"
        ) from error
    return path


def validate_config(config: Mapping[str, Any]) -> None:
    """Fail closed if the metadata manifest or its claim ceiling changes."""

    _strict(
        config,
        {
            "schema_version",
            "status",
            "audit_id",
            "as_of_utc",
            "purpose",
            "implementation_binding",
            "scope",
            "source_references",
            "xmm_observations",
            "xmm_source_packet",
            "planck_source_packet",
            "required_missing_assets",
            "cp5_adjudication",
            "authorization",
            "claim_boundary",
            "output_path",
        },
        "source-feasibility config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "metadata_audit_blocked"
        or config["audit_id"] != AUDIT_ID
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "audit identity changed"
        )

    implementation = config["implementation_binding"]
    _strict(implementation, {"path", "file_sha256"}, "implementation binding")
    if (
        implementation["path"] != IMPLEMENTATION_PATH.as_posix()
        or len(str(implementation["file_sha256"])) != 64
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "implementation binding changed"
        )

    scope = config["scope"]
    if scope != {
        "cluster": "A1795",
        "sample_role": "already_exposed_development_object",
        "scientific_payload_rows_opened": 0,
        "confirmation_rows_opened": 0,
        "independent_rows_opened": 0,
        "hidden_answers_opened": 0,
        "large_files_downloaded": 0,
        "downloaded_bytes": 0,
        "paid_or_model_calls": 0,
    }:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "metadata-only scope changed"
        )

    observations = config["xmm_observations"]
    if not isinstance(observations, list) or len(observations) != 6:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "A1795 observation manifest changed"
        )
    if tuple(row.get("observation_id") for row in observations) != OBSERVATION_IDS:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "A1795 observation IDs changed"
        )
    if any(
        row.get("public") is not True
        or row.get("pps_version") != PPS_VERSION
        or row.get("sas_version") != SAS_VERSION
        for row in observations
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "XSA release evidence changed"
        )

    xmm = config["xmm_source_packet"]
    _strict(
        xmm,
        {
            "archive_endpoint",
            "license",
            "observation_archives",
            "component_dispositions",
            "bounded_packet_status",
        },
        "XMM source packet",
    )
    archives = xmm["observation_archives"]
    if not isinstance(archives, list) or len(archives) != 6:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "XMM archive manifest changed"
        )
    for row, observation_id in zip(archives, OBSERVATION_IDS, strict=True):
        expected_base = (
            "https://nxsa.esac.esa.int/nxsa-sl/servlet/data-action-aio?obsno="
            f"{observation_id}"
        )
        if row != {
            "observation_id": observation_id,
            "odf_url": f"{expected_base}&level=ODF",
            "pps_event_url": f"{expected_base}&level=PPS&name=EVENLI",
            "odf_head_status": 200,
            "pps_event_head_status": 200,
            "odf_expected_bytes": None,
            "pps_event_expected_bytes": None,
            "size_status": "NOT_REPORTED_BY_XSA_HEAD_OR_TAP",
            "download_authorized": False,
        }:
            raise GravityClusterA1795CovarianceSourceFeasibilityError(
                f"XMM archive row changed: {observation_id}"
            )
    if xmm["bounded_packet_status"] != (
        "BLOCKED_ARCHIVE_SIZES_UNREPORTED_AND_DERIVED_COVARIANCE_INPUTS_ABSENT"
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "XMM packet conclusion changed"
        )

    planck = config["planck_source_packet"]
    products = planck.get("public_products")
    if not isinstance(products, list) or len(products) != len(PLANCK_PRODUCTS):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "Planck product manifest changed"
        )
    observed_products = {row.get("file"): row.get("head_content_length") for row in products}
    if observed_products != PLANCK_PRODUCTS:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "Planck product sizes changed"
        )
    if any(row.get("head_status") != 200 for row in products):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "Planck public availability changed"
        )
    if planck.get("exact_reconstruction_status") != (
        "BLOCKED_PUBLIC_PR2_PRODUCTS_DO_NOT_INCLUDE_XCOP_LOCAL_MAP_OR_MATCHED_SIMULATIONS"
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "Planck packet conclusion changed"
        )

    missing = config["required_missing_assets"]
    if (
        not isinstance(missing, list)
        or {row.get("asset_id") for row in missing} != REQUIRED_MISSING_IDS
        or any(row.get("publicly_located") is not False for row in missing)
    ):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "required missing-asset set changed"
        )

    adjudication = config["cp5_adjudication"]
    if adjudication != {
        "decision": DECISION,
        "statuses": CP_STATUS,
        "new_reduction_feasible": True,
        "exact_xcop_covariance_reconstruction_feasible": False,
        "component_complete_covariance_feasible": False,
        "smallest_next_action": (
            "obtain explicit authorization and storage budget for a new A1795 ODF/CCF/Planck "
            "reduction, then freeze archive hashes, CCF snapshot, regions, background model, "
            "response generation, deprojection operator, seeds, and nuisance priors before opening rows"
        ),
    }:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "CP5 adjudication changed"
        )

    authorization = config["authorization"]
    if authorization != {
        "authorized": False,
        "downloads_authorized": False,
        "payload_access_authorized": False,
        "reduction_authorized": False,
        "scoring_authorized": False,
        "reason": "SOURCE_PACKET_INCOMPLETE_AND_NO_PAYLOAD_COMMITMENT_GRANTED",
    }:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "authorization changed"
        )

    claims = config["claim_boundary"]
    if claims != {
        "public_inputs_exist_for_a_new_a1795_reduction": True,
        "complete_bounded_source_packet_frozen": False,
        "original_xcop_temperature_covariance_reconstructible": False,
        "original_xcop_density_covariance_reconstructible": False,
        "original_xcop_pressure_covariance_reconstructible_from_public_raw_assets": False,
        "shared_calibration_covariance_reconstructible": False,
        "joint_xray_sz_covariance_reconstructible": False,
        "CP5_2_through_CP5_6_complete": False,
        "scientific_reanalysis_performed": False,
        "scientific_result_changed": False,
        "publication_claim_supported": False,
    }:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "claim boundary changed"
        )


def load_config(root: Path) -> dict[str, Any]:
    """Load and validate the frozen metadata manifest."""

    config_path = _under(root, CONFIG_PATH, "config")
    config = _read_json(config_path)
    validate_config(config)
    implementation_path = _under(root, IMPLEMENTATION_PATH, "implementation")
    if _file_sha(implementation_path) != config["implementation_binding"]["file_sha256"]:
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "implementation hash changed"
        )
    return config


def build_receipt(root: Path) -> dict[str, Any]:
    """Build the deterministic offline blocker receipt without network or payload access."""

    config = load_config(root)
    config_body = dict(config)
    receipt = {
        "schema_version": RECEIPT_SCHEMA,
        "audit_id": AUDIT_ID,
        "decision": DECISION,
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config_body),
        },
        "implementation_binding": dict(config["implementation_binding"]),
        "cluster": "A1795",
        "observation_ids": list(OBSERVATION_IDS),
        "xsa_release": {
            "pps_version": PPS_VERSION,
            "sas_version": SAS_VERSION,
            "public_observations": 6,
            "archive_size_fields_reported": 0,
        },
        "public_packet": {
            "xmm_observation_archives_located": 6,
            "xmm_event_endpoints_located": 6,
            "planck_products_located": len(PLANCK_PRODUCTS),
            "planck_public_bytes_manifested": sum(PLANCK_PRODUCTS.values()),
            "new_reduction_feasible": True,
            "exact_xcop_covariance_reconstruction_feasible": False,
        },
        "missing_asset_ids": sorted(REQUIRED_MISSING_IDS),
        "cp5_statuses": dict(CP_STATUS),
        "authorization": dict(config["authorization"]),
        "claims": dict(config["claim_boundary"]),
        "counts": {
            "metadata_source_references": len(config["source_references"]),
            "metadata_network_calls_during_receipt_build": 0,
            "scientific_payload_rows_read": 0,
            "confirmation_rows_read": 0,
            "independent_rows_read": 0,
            "hidden_answers_read": 0,
            "large_files_downloaded": 0,
            "downloaded_bytes": 0,
            "scientific_scores_computed": 0,
            "paid_or_model_calls": 0,
        },
        "limitations": [
            "XSA exposes all six public ODF/PPS endpoints but does not report archive byte sizes in the observed HEAD or TAP metadata.",
            "Public SAS/CCF/ESAS assets permit a new reduction, not an exact replay of the X-COP extraction, background, likelihood, deprojection, or nuisance ensemble.",
            "The public Planck PR2 package supplies full-sky maps, weights, half splits, noise summaries, beams, and masks, but not the X-COP local 7-arcmin MILCA map or its matched noise-realization ensemble.",
            "No payload was opened and no download, reduction, scoring, or publication claim is authorized by this receipt.",
        ],
    }
    receipt["content_sha256"] = _sha(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    """Validate a stored receipt and all locally checkable bindings."""

    body = dict(receipt)
    observed = body.pop("content_sha256", None)
    if observed != _sha(body):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "receipt content hash changed"
        )
    if dict(receipt) != build_receipt(root):
        raise GravityClusterA1795CovarianceSourceFeasibilityError(
            "stored receipt no longer rebuilds exactly"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("build-receipt")
    subparsers.add_parser("check")
    subparsers.add_parser("status")
    return parser


def main() -> int:
    args = _parser().parse_args()
    root = args.root.resolve()
    if args.command == "build-receipt":
        receipt = build_receipt(root)
        output_path = _under(root, OUTPUT_PATH, "output")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(_canonical_bytes(receipt))
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if args.command == "check":
        receipt = _read_json(_under(root, OUTPUT_PATH, "output"))
        validate_receipt(receipt, root)
        print(DECISION)
        return 0
    if args.command == "status":
        receipt = build_receipt(root)
        print(
            json.dumps(
                {
                    "decision": receipt["decision"],
                    "cp5_statuses": receipt["cp5_statuses"],
                    "authorization": receipt["authorization"],
                },
                sort_keys=True,
            )
        )
        return 0
    raise AssertionError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
