from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sigma_theory_compiler import (
    gravity_cluster_a1795_covariance_source_feasibility as feasibility,
)

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = Path(
    "configs/gravity_cluster_a1795_covariance_source_feasibility_adjudicator_v1.json"
)
RECEIPT_PATH = Path(
    "runs/gravity/publication-readiness/a1795-covariance-source-feasibility-adjudicator-v1.json"
)
TEST_PATH = Path("tests/test_gravity_cluster_a1795_covariance_source_feasibility_adjudicator.py")

CONFIG_SCHEMA = (
    "invariant-gravity-cluster-a1795-covariance-source-feasibility-adjudicator-config-1.0"
)
RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-a1795-covariance-source-feasibility-adjudicator-receipt-1.0"
)
STATUS = "strictly_verified_source_packet_incomplete_cp5_2_through_cp5_6_blocked"
DECISION = "BLOCKED_NO_COMPLETE_PUBLIC_COVARIANCE_SOURCE_PACKET"

EXPECTED_ORIGINAL_ARTIFACTS = {
    "source": {
        "path": (
            "src/sigma_theory_compiler/gravity_cluster_a1795_covariance_source_feasibility.py"
        ),
        "file_sha256": "a6401da1245041da6443b6fd309186ef95b9b23ae4e51a47498b6c5805737ac7",
    },
    "config": {
        "path": "configs/gravity_cluster_a1795_covariance_source_feasibility_v1.json",
        "file_sha256": "88634eed11f039084ae8b4c71b23ea11960f965ed2aa5d9bf6c785fdebd2efb2",
    },
    "test": {
        "path": "tests/test_gravity_cluster_a1795_covariance_source_feasibility.py",
        "file_sha256": "b50b8f33208ce5bdaae669ddd8aaed73ed35e1f7ed343ce2eec9bda95b1b3ef0",
    },
    "receipt": {
        "path": ("runs/gravity/publication-readiness/a1795-covariance-source-feasibility-v1.json"),
        "file_sha256": "ddaa05d1d22343f386c53c4e872af8ce75292330b2a8801aba3063d216e09898",
    },
}

EXPECTED_ITEM59_BINDINGS = {
    "source_receipt": {
        "path": (
            "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/"
            "source-receipt.json"
        ),
        "file_sha256": "819360acdd40c58254cc43b18f9d6613b2c5f908122e134b7fa4b3f346b69aac",
    },
    "preflight_manifest": {
        "path": (
            "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/"
            "preflight-manifest.json"
        ),
        "file_sha256": "152af8f8bb3da718e023ef56497d85747f6ddeefcb198b76cc3473f23792e484",
    },
}

EXPECTED_TEST_SHA256 = "bd19fed2e3bd77ca2d9f834af20e79b2a7ab91f2898a2f26410fd622aa7548dc"

FROZEN_SECTION_SHA256 = {
    "scope": "dde6360c652b2b2896bcd4758b12dc3926c00872a421634f57bc6ed83cd580d4",
    "source_references": ("45536e684b1f3a874e3a4e8c66107f223831f3701384912e76e5a826d7442164"),
    "xmm_observations": ("4091d58140b0b40dd2128f77cd9bf54ac3a6a1d954572886a1ef30eea309b3bd"),
    "xmm_source_packet": ("90bd747b01270562317424070efa3cc358e61205b42ff0c931082cf0616a2bc8"),
    "planck_source_packet": ("a50a038dfc3fa1e7ed0b37aac5842a5f09b55e38960a016bb16fe9fab2f2cece"),
    "required_missing_assets": ("1ad6bc44ff519d8525b31161f18f289759d0e8ae1d46c349a5dddbc25a2c9a05"),
    "cp5_adjudication": ("ff046d8a04236b68cb7208ad09462a751d8c176d86413c9c27b69376072ab0eb"),
    "authorization": ("52591e574576655e52f4896b23b36dba67a17727d3243f5dcb2e0d0e6e4c5237"),
    "claim_boundary": ("1433ea4ae50295630674ef8b567cd834248fd9a6c431b76ad8356bd61099acd4"),
}

CP5_STATUSES = {
    "CP5.2": "BLOCKED_TEMPERATURE_COVARIANCE_NOT_RECONSTRUCTIBLE_FROM_PUBLIC_PACKET",
    "CP5.3": "BLOCKED_DENSITY_COVARIANCE_NOT_RECONSTRUCTIBLE_FROM_PUBLIC_PACKET",
    "CP5.4": "BLOCKED_SHARED_CALIBRATION_COVARIANCE_NOT_PUBLICLY_SPECIFIED",
    "CP5.5": "BLOCKED_MATCHED_BACKGROUND_BEAM_SIMULATION_ENSEMBLE_NOT_PUBLIC",
    "CP5.6": "BLOCKED_JOINT_XRAY_SZ_CALIBRATION_COVARIANCE_NOT_PUBLIC",
}

DATA_BOUNDARY = {
    "network_calls": 0,
    "downloads": 0,
    "downloaded_bytes": 0,
    "scientific_payload_rows_read": 0,
    "confirmation_rows_read": 0,
    "independent_rows_read": 0,
    "hidden_answers_read": 0,
    "scientific_scores_computed": 0,
    "paid_or_model_calls": 0,
}

CLAIM_BOUNDARY = {
    "strict_artifact_integrity_verified": True,
    "public_inputs_exist_for_a_new_a1795_reduction": True,
    "complete_bounded_source_packet_frozen": False,
    "CP5_2_through_CP5_6_complete": False,
    "scientific_reanalysis_performed": False,
    "scientific_result_changed": False,
    "publication_claim_supported": False,
    "downloads_authorized": False,
    "payload_access_authorized": False,
}

FACTUAL_QUALIFICATIONS = {
    "xcop_release_page": {
        "corrected_fact": (
            "The current X-COP project page distributes both basic and high-level "
            "thermodynamic, mass, gas-mass, and gas-fraction products."
        ),
        "original_wording_status": (
            "The original manifest phrase 'only basic products are currently available' "
            "is stale and is not adopted by this adjudicator."
        ),
        "covariance_consequence": (
            "Neither the corrected page wording nor the released high-level profiles "
            "supply the exact X-ray likelihood chains, shared calibration ensemble, "
            "local 7-arcmin MILCA map, or matched simulations required by CP5.2-CP5.6."
        ),
    },
    "xmm_license": {
        "verified_scope": "proposal_074441_only_from_the_bound_primary_rights_page",
        "unverified_scope": ["009782", "010907", "020519"],
        "qualification": (
            "Public XSA access is verified for all six observations, but this packet "
            "does not extend the proposal-074441 CC BY-NC 3.0 IGO citation to the "
            "other three proposal identifiers without broader primary rights evidence."
        ),
    },
    "xcop_archive_provenance": {
        "archive_sha256": ("0edf5038b419b70d070b73b22f4801e27f318b0854db61eec52142c27c140d94"),
        "archive_bytes": 315_080_566,
        "hash_source": EXPECTED_ITEM59_BINDINGS["source_receipt"],
        "byte_count_source": EXPECTED_ITEM59_BINDINGS["preflight_manifest"],
        "current_adjudicator_downloaded_archive": False,
        "current_adjudicator_opened_archive_rows": False,
    },
    "planck_release_label": {
        "product": "COM_CompMap_Compton-SZMap_R2.02.tgz",
        "product_label": "R2.02",
        "storage_path_contains": "/Planck/release_3/",
        "qualification": (
            "The R2.02 product name and PR2-facing documentation are retained together "
            "with the release_3 storage path; this adjudicator does not resolve or "
            "reinterpret that archive-label ambiguity."
        ),
    },
}

TOP_LEVEL_CONFIG_KEYS = {
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
}


def canonical_sha256(value: Any) -> str:
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def normalized_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def confined(path: Path) -> Path:
    target = path.resolve()
    try:
        target.relative_to(ROOT)
    except ValueError as error:
        raise RuntimeError(f"path escaped repository: {path}") from error
    return target


def strict_keys(value: Any, expected: set[str], label: str) -> None:
    if not isinstance(value, dict) or set(value) != expected:
        actual = set(value) if isinstance(value, dict) else set()
        raise RuntimeError(
            f"{label} keys changed; missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def artifact_binding(path: Path) -> dict[str, str]:
    target = confined(path)
    return {
        "path": target.relative_to(ROOT).as_posix(),
        "file_sha256": file_sha256(target),
    }


def validate_binding(binding: Mapping[str, Any], label: str) -> Path:
    strict_keys(binding, {"path", "file_sha256"}, label)
    target = confined(ROOT / str(binding["path"]))
    if not target.is_file() or file_sha256(target) != binding["file_sha256"]:
        raise RuntimeError(f"{label} missing, changed, or swapped")
    return target


def read_bound_json(path: Path, binding: Mapping[str, Any], label: str) -> dict[str, Any]:
    expected = validate_binding(binding, label)
    target = confined(path)
    if target != expected:
        raise RuntimeError(f"{label} path changed or swapped")
    value = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{label} is not a JSON object")
    return value


def _validate_source_schemas(config: Mapping[str, Any]) -> None:
    references = config["source_references"]
    if not isinstance(references, list) or len(references) != 18:
        raise RuntimeError("source reference count changed")
    for index, row in enumerate(references):
        strict_keys(row, {"source_id", "url", "role"}, f"source reference {index}")

    observations = config["xmm_observations"]
    if not isinstance(observations, list) or len(observations) != 6:
        raise RuntimeError("XMM observation count changed")
    observation_keys = {
        "observation_id",
        "mosaic_role",
        "target",
        "ra_deg",
        "dec_deg",
        "duration_s",
        "odf_version",
        "pps_version",
        "sas_version",
        "pps_proc_date",
        "public",
        "science_exposures",
    }
    for index, row in enumerate(observations):
        strict_keys(row, observation_keys, f"XMM observation {index}")
        if not isinstance(row["science_exposures"], list):
            raise TypeError(f"XMM observation {index} exposure schema changed")

    xmm = config["xmm_source_packet"]
    strict_keys(
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
    strict_keys(
        xmm["license"],
        {"status", "rights_url", "redistribution_by_this_manifest"},
        "XMM license",
    )
    archives = xmm["observation_archives"]
    if not isinstance(archives, list) or len(archives) != 6:
        raise RuntimeError("XMM observation archive count changed")
    archive_keys = {
        "observation_id",
        "odf_url",
        "pps_event_url",
        "odf_head_status",
        "pps_event_head_status",
        "odf_expected_bytes",
        "pps_event_expected_bytes",
        "size_status",
        "download_authorized",
    }
    for index, row in enumerate(archives):
        strict_keys(row, archive_keys, f"XMM archive row {index}")
    dispositions = xmm["component_dispositions"]
    if not isinstance(dispositions, list) or len(dispositions) != 7:
        raise RuntimeError("XMM component disposition count changed")
    for index, row in enumerate(dispositions):
        strict_keys(
            row,
            {"component", "availability", "exact_xcop_product", "blocker"},
            f"XMM component disposition {index}",
        )

    planck = config["planck_source_packet"]
    strict_keys(
        planck,
        {
            "release",
            "xcop_basic_archive",
            "public_products",
            "license",
            "simulation_pointer",
            "exact_reconstruction_status",
            "blocker",
        },
        "Planck source packet",
    )
    strict_keys(
        planck["xcop_basic_archive"],
        {
            "release_page_url",
            "archive_download_url",
            "observed_archive_bytes",
            "observed_archive_sha256",
            "redistribution_license_status",
            "contains_basic_profile_and_covariance_products",
            "contains_xcop_local_milca_map_or_matched_noise_realizations",
            "download_authorized",
        },
        "X-COP archive provenance",
    )
    products = planck["public_products"]
    if not isinstance(products, list) or len(products) != 5:
        raise RuntimeError("Planck public product count changed")
    for index, row in enumerate(products):
        strict_keys(
            row,
            {
                "file",
                "url",
                "contents",
                "head_status",
                "head_content_length",
                "last_modified",
                "download_authorized",
            },
            f"Planck product {index}",
        )
    strict_keys(
        planck["license"],
        {
            "public_access_verified",
            "machine_readable_redistribution_license_located",
            "redistribution_by_this_manifest",
        },
        "Planck license",
    )
    strict_keys(
        planck["simulation_pointer"],
        {
            "url",
            "observed_resolution",
            "file_manifest_or_sizes_verified",
            "is_xcop_local_milca_noise_ensemble",
        },
        "Planck simulation pointer",
    )

    missing = config["required_missing_assets"]
    if not isinstance(missing, list) or len(missing) != 7:
        raise RuntimeError("required missing-asset count changed")
    for index, row in enumerate(missing):
        strict_keys(
            row,
            {"asset_id", "needed_for", "minimum_receipt", "publicly_located"},
            f"required missing asset {index}",
        )


def validate_frozen_feasibility_config(config: Mapping[str, Any]) -> None:
    strict_keys(config, TOP_LEVEL_CONFIG_KEYS, "original feasibility config")
    _validate_source_schemas(config)
    for section, expected_sha256 in FROZEN_SECTION_SHA256.items():
        if canonical_sha256(config[section]) != expected_sha256:
            raise RuntimeError(f"original feasibility {section} changed")

    planck_total = sum(
        int(row["head_content_length"]) for row in config["planck_source_packet"]["public_products"]
    )
    if planck_total != 13_314_915_231:
        raise RuntimeError("Planck public byte total changed")
    if config["cp5_adjudication"]["decision"] != DECISION:
        raise RuntimeError("feasibility decision changed")
    if config["cp5_adjudication"]["statuses"] != CP5_STATUSES:
        raise RuntimeError("CP5.2-CP5.6 statuses changed")


def validate_item59_provenance(config: Mapping[str, Any]) -> None:
    source_binding = config["item59_bindings"]["source_receipt"]
    source = read_bound_json(
        ROOT / str(source_binding["path"]), source_binding, "Item59 source receipt"
    )
    if (
        source.get("schema_version") != "invariant-gravity-item59-source-receipt-1.0"
        or source.get("content_sha256")
        != "da4b00265cbb3a5fe09192bed21564d8a752b6fa32bdf5b5fad84564d512c113"
        or source.get("archive_sha256")
        != FACTUAL_QUALIFICATIONS["xcop_archive_provenance"]["archive_sha256"]
    ):
        raise RuntimeError("Item59 source-receipt archive provenance changed")

    preflight_binding = config["item59_bindings"]["preflight_manifest"]
    preflight = read_bound_json(
        ROOT / str(preflight_binding["path"]),
        preflight_binding,
        "Item59 preflight manifest",
    )
    expected_archive = {
        "path": "work/item3-density-v2-audit/xcop-allfiles.tar.gz",
        "sha256": FACTUAL_QUALIFICATIONS["xcop_archive_provenance"]["archive_sha256"],
        "bytes": FACTUAL_QUALIFICATIONS["xcop_archive_provenance"]["archive_bytes"],
    }
    if (
        preflight.get("schema_version") != "invariant-gravity-item59-preflight-1.0"
        or preflight.get("content_sha256")
        != "acc02410dd10203ecde419f0811383a23eef01045f821d0282ec469e35e41cd2"
        or preflight.get("archive") != expected_archive
    ):
        raise RuntimeError("Item59 preflight archive provenance changed")


def load_config(path: Path, expected_sha256: str) -> dict[str, Any]:
    target = confined(path)
    if target != ROOT / CONFIG_PATH:
        raise RuntimeError("strict A1795 adjudicator config path changed")
    if not target.is_file() or file_sha256(target) != expected_sha256:
        raise RuntimeError("strict A1795 adjudicator config hash changed")
    config = json.loads(target.read_text(encoding="utf-8"))
    strict_keys(
        config,
        {
            "schema_version",
            "status",
            "purpose",
            "implementation_source",
            "implementation_source_normalized_sha256",
            "verifier_test",
            "original_artifacts",
            "item59_bindings",
            "frozen_section_sha256",
            "cp5_statuses",
            "data_boundary",
            "claim_boundary",
            "factual_qualifications",
            "receipt_path",
        },
        "strict A1795 adjudicator config",
    )
    expected_test = {"path": TEST_PATH.as_posix(), "file_sha256": EXPECTED_TEST_SHA256}
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_append_only_strict_a1795_feasibility_adjudication"
        or config["verifier_test"] != expected_test
        or config["original_artifacts"] != EXPECTED_ORIGINAL_ARTIFACTS
        or config["item59_bindings"] != EXPECTED_ITEM59_BINDINGS
        or config["frozen_section_sha256"] != FROZEN_SECTION_SHA256
        or config["cp5_statuses"] != CP5_STATUSES
        or config["data_boundary"] != DATA_BOUNDARY
        or config["claim_boundary"] != CLAIM_BOUNDARY
        or config["factual_qualifications"] != FACTUAL_QUALIFICATIONS
        or config["receipt_path"] != RECEIPT_PATH.as_posix()
    ):
        raise RuntimeError("strict A1795 adjudicator frozen contract changed")
    source = confined(ROOT / str(config["implementation_source"]))
    if (
        source != Path(__file__).resolve()
        or normalized_sha256(source) != config["implementation_source_normalized_sha256"]
    ):
        raise RuntimeError("strict A1795 adjudicator source changed")
    validate_binding(config["verifier_test"], "verifier test")
    for name, binding in config["original_artifacts"].items():
        validate_binding(binding, f"original artifact {name}")
    for name, binding in config["item59_bindings"].items():
        validate_binding(binding, f"Item59 binding {name}")
    validate_item59_provenance(config)
    config["_config_sha256"] = expected_sha256
    return config


def adjudicate(config: Mapping[str, Any]) -> dict[str, Any]:
    original_config_binding = config["original_artifacts"]["config"]
    original_config = read_bound_json(
        ROOT / str(original_config_binding["path"]),
        original_config_binding,
        "original feasibility config",
    )
    validate_frozen_feasibility_config(original_config)

    original_receipt_binding = config["original_artifacts"]["receipt"]
    original_receipt = read_bound_json(
        ROOT / str(original_receipt_binding["path"]),
        original_receipt_binding,
        "original feasibility receipt",
    )
    feasibility.validate_receipt(original_receipt, ROOT)
    if (
        original_receipt.get("decision") != DECISION
        or original_receipt.get("cp5_statuses") != CP5_STATUSES
        or original_receipt.get("counts")
        != {
            "metadata_source_references": 18,
            "metadata_network_calls_during_receipt_build": 0,
            "scientific_payload_rows_read": 0,
            "confirmation_rows_read": 0,
            "independent_rows_read": 0,
            "hidden_answers_read": 0,
            "large_files_downloaded": 0,
            "downloaded_bytes": 0,
            "scientific_scores_computed": 0,
            "paid_or_model_calls": 0,
        }
    ):
        raise RuntimeError("original feasibility receipt decision or access boundary changed")

    return {
        "strict_verifier_passed": True,
        "original_receipt_reconstructed_exactly": True,
        "nested_factual_sections_exact": True,
        "item59_archive_provenance_exact": True,
        "observation_count": 6,
        "observation_ids": list(feasibility.OBSERVATION_IDS),
        "planck_product_count": 5,
        "planck_public_bytes_manifested": 13_314_915_231,
        "complete_public_covariance_source_packet": False,
        "CP5_2_through_CP5_6_complete": False,
        "downloads_authorized": False,
        "payload_access_authorized": False,
        "decision": DECISION,
    }


def expected_receipt(config: Mapping[str, Any]) -> dict[str, Any]:
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "status": STATUS,
        "decision": DECISION,
        "evidence": {
            "config": artifact_binding(ROOT / CONFIG_PATH),
            "implementation_source": artifact_binding(ROOT / str(config["implementation_source"])),
            "verifier_test": config["verifier_test"],
            "original_artifacts": config["original_artifacts"],
            "item59_bindings": config["item59_bindings"],
            "frozen_section_sha256": config["frozen_section_sha256"],
        },
        "adjudication": adjudicate(config),
        "cp5_statuses": CP5_STATUSES,
        "data_boundary": DATA_BOUNDARY,
        "claim_boundary": CLAIM_BOUNDARY,
        "factual_qualifications": FACTUAL_QUALIFICATIONS,
        "limitations": [
            "This receipt verifies metadata and frozen evidence only; it does not open scientific payload rows.",
            "Public raw and high-level products permit a new A1795 reduction but not an exact reconstruction of the missing X-COP covariance and nuisance ensemble.",
            "Historical Item59 archive bytes and hash are provenance evidence, not downloads performed by this adjudicator.",
            "The strict-verifier pass preserves the source-feasibility BLOCK decision and does not complete CP5.2-CP5.6.",
        ],
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def require_exact_receipt(
    actual: Mapping[str, Any], expected: Mapping[str, Any], label: str
) -> None:
    strict_keys(actual, set(expected), label)
    if dict(actual) != dict(expected):
        raise RuntimeError(f"{label} does not exactly reconstruct from frozen evidence")


def _write_json_no_clobber(path: Path, value: Mapping[str, Any]) -> None:
    target = confined(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{target.name}.", suffix=".tmp", dir=target.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, target)
        except FileExistsError as error:
            raise RuntimeError("atomic no-clobber publication refused existing receipt") from error
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_receipt(config_path: Path, expected_config_sha256: str, output: Path) -> dict[str, Any]:
    target = confined(output)
    if target != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict A1795 adjudicator receipt path changed")
    config = load_config(config_path, expected_config_sha256)
    receipt = expected_receipt(config)
    _write_json_no_clobber(target, receipt)
    return receipt


def check_receipt(
    config_path: Path, expected_config_sha256: str, receipt_path: Path
) -> dict[str, Any]:
    config = load_config(config_path, expected_config_sha256)
    target = confined(receipt_path)
    if target != ROOT / RECEIPT_PATH:
        raise RuntimeError("strict A1795 adjudicator receipt path changed")
    actual = json.loads(target.read_text(encoding="utf-8"))
    require_exact_receipt(actual, expected_receipt(config), "strict A1795 adjudicator receipt")
    return {
        "valid": True,
        "strict_verifier_passed": True,
        "decision": DECISION,
        "complete_public_covariance_source_packet": False,
        "CP5_2_through_CP5_6_complete": False,
        "downloads_authorized": False,
        "payload_access_authorized": False,
        "scientific_claim_allowed": False,
        "receipt_sha256": file_sha256(target),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    write = commands.add_parser("write-receipt")
    write.add_argument("--config", type=Path, required=True)
    write.add_argument("--expected-config-sha256", required=True)
    write.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check")
    check.add_argument("--config", type=Path, required=True)
    check.add_argument("--expected-config-sha256", required=True)
    check.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "write-receipt":
        result = write_receipt(args.config, args.expected_config_sha256, args.output)
    else:
        result = check_receipt(args.config, args.expected_config_sha256, args.receipt)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
