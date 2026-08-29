"""Freeze the target-blind missing-variable predictor contract without scoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_missing_variable_preflight_v1.json")
OUTPUT_PATH = Path("runs/gravity/publication-readiness/missing-variable-preflight-v1.json")
SOURCE_PATH = Path("src/sigma_theory_compiler/gravity_missing_variable_preflight.py")
TEST_PATH = Path("tests/test_gravity_missing_variable_preflight.py")
CONFIG_SCHEMA = "invariant-gravity-missing-variable-preflight-config-1.0"
RECEIPT_SCHEMA = "invariant-gravity-missing-variable-preflight-receipt-1.0"
EXPECTED_CONFIG_FILE_SHA256 = "5bd3655e12aa972a1d52d0483b652e51577d72bcce087132585189c4d81fbf6f"
EXPECTED_CONFIG_CONTENT_SHA256 = "30fdc8f51efd223e1a058023cc2864d40e7bbb62f410d6bdcfdbae0c0bbf2cf4"
SOURCE_IDS = (
    "cluster_predictor_strata_receipt",
    "cluster_alternative_cause_matrix",
    "cluster_public_predictor_source_receipt",
    "group_scale_source_audit",
    "shared_ben_real_preflight_v2",
)
LANE_IDS = (
    "SPARC_DEVELOPMENT_ONLY",
    "XCOP_EIGHT_EXPOSED_DEVELOPMENT",
    "GROUP_SOURCE_AUDIT_LANES",
)
VARIABLE_IDS = (
    "geometry_3d",
    "nonthermal_pressure",
    "calibration",
    "clumping",
    "boundary_conditions",
    "environment",
    "assembly_history",
)
SOURCE_CATALOG_IDS = (
    "SPARC_MASSMODELS_TABLE",
    "SPARC_PHOTOMETRY_HI_MAPS",
    "SPARC_RESOLVED_GAS_KINEMATICS",
    "SPARC_CALIBRATION_ENVIRONMENT_HISTORY",
    "XCOP_MORPHOLOGY_AND_ASSEMBLY_PROXY",
    "XCOP_JOINT_3D_GEOMETRY",
    "XCOP_GAS_MOTION_NONTHERMAL",
    "XCOP_CALIBRATION_CLUMPING_BOUNDARY",
    "XCOP_ENVIRONMENT_AND_TIMELINE",
    "GROUP_ALIAS_DIRECT_ENDPOINT_PACKET",
    "GROUP_MISSING_VARIABLE_PRODUCTS",
)


class GravityMissingVariablePreflightError(RuntimeError):
    """Raised when a frozen contract, source, access boundary, or receipt changes."""


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        + b"\n"
    )


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityMissingVariablePreflightError(f"expected JSON object: {path}")
    return value


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityMissingVariablePreflightError(f"{label} keys changed")


def _under(root: Path, relative: str | Path, label: str) -> Path:
    target = (root.resolve() / Path(relative)).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as error:
        raise GravityMissingVariablePreflightError(f"{label} escaped repository") from error
    return target


def _declared_content_sha(value: Mapping[str, Any]) -> str:
    body = dict(value)
    expected = body.pop("content_sha256", None)
    with_newline = _sha(body)
    compact = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()
    if expected not in {with_newline, compact}:
        raise GravityMissingVariablePreflightError("bound source content hash changed")
    return str(expected)


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "preflight_id",
            "status",
            "purpose",
            "source_bindings",
            "lane_contract",
            "public_source_catalog",
            "variable_registry",
            "chronology_and_access_contract",
            "claim_boundary",
            "next_actionable_acquisition",
            "output_path",
        },
        "missing-variable config",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["preflight_id"] != "gravity-missing-variable-preflight-v1"
        or config["status"] != "frozen_target_blind_predictor_contract_no_scoring"
        or config["output_path"] != OUTPUT_PATH.as_posix()
        or _sha(dict(config)) != EXPECTED_CONFIG_CONTENT_SHA256
    ):
        raise GravityMissingVariablePreflightError("frozen config content changed")
    if tuple(row.get("source_id") for row in config["source_bindings"]) != SOURCE_IDS:
        raise GravityMissingVariablePreflightError("source binding inventory changed")
    if tuple(config["lane_contract"].get("lane_ids", ())) != LANE_IDS:
        raise GravityMissingVariablePreflightError("lane contract changed")
    if tuple(row.get("source_id") for row in config["public_source_catalog"]) != (
        SOURCE_CATALOG_IDS
    ):
        raise GravityMissingVariablePreflightError("public source catalog changed")
    if tuple(row.get("variable_id") for row in config["variable_registry"]) != VARIABLE_IDS:
        raise GravityMissingVariablePreflightError("variable registry changed")
    source_ids = set(SOURCE_CATALOG_IDS)
    for row in config["variable_registry"]:
        _strict(
            row,
            {
                "variable_id",
                "label",
                "definition_status",
                "dimensionless_predictor",
                "physical_provenance",
                "covariance_nuisance_treatment",
                "missingness_rule",
                "forbidden_response_derived_substitutes",
                "domain_applicability",
                "prospective_test",
            },
            "variable registry row",
        )
        if row["dimensionless_predictor"].get("units") != "1":
            raise GravityMissingVariablePreflightError("predictor is not dimensionless")
        lane_rows = row["domain_applicability"]
        if tuple(item.get("lane_id") for item in lane_rows) != LANE_IDS:
            raise GravityMissingVariablePreflightError("variable lane inventory changed")
        for item in lane_rows:
            if not set(item["required_source_ids"]).issubset(source_ids):
                raise GravityMissingVariablePreflightError("unknown required public source")
        if row["prospective_test"] != {
            **row["prospective_test"],
            "multiplicity_family": "missing_variable_primary_seven",
            "run_status": "NOT_RUN_NO_RESPONSE_ACCESS",
        }:
            raise GravityMissingVariablePreflightError("prospective test freeze changed")
    if config["lane_contract"] != {
        "lane_ids": list(LANE_IDS),
        "lane_ids_are_metadata_not_predictors": True,
        "object_identity_predictor_forbidden": True,
        "domain_identity_predictor_forbidden": True,
        "whole_object_split_required": True,
        "post_response_variable_selection_forbidden": True,
        "response_derived_normalization_forbidden": True,
    }:
        raise GravityMissingVariablePreflightError("leakage boundary changed")
    access = config["chronology_and_access_contract"]
    if (
        access["config_and_sources_validate_before_any_predictor_loader"] is not True
        or access["predecessor_public_predictor_rows_read"] != 8
        or access["new_predictor_source_payload_rows_opened"] != 0
        or any(
            value != 0
            for key, value in access.items()
            if key
            not in {
                "config_and_sources_validate_before_any_predictor_loader",
                "predecessor_public_predictor_rows_read",
            }
        )
    ):
        raise GravityMissingVariablePreflightError("access chronology changed")
    if config["claim_boundary"] != {
        "defined_proxy_contracts_frozen": True,
        "all_variable_predictor_contracts_frozen": False,
        "source_definition_blockers_present": True,
        "proxy_executable_is_measurement_ready": False,
        "continuous_missing_variables_measured": False,
        "scientific_scoring_executed": False,
        "cause_identified": False,
        "candidate_supported_or_refuted": False,
        "cross_domain_law_supported": False,
        "publication_readiness_changed": False,
        "scientific_claim_allowed": False,
    }:
        raise GravityMissingVariablePreflightError("claim boundary changed")


def load_config(root: Path) -> dict[str, Any]:
    path = _under(root, CONFIG_PATH, "config")
    if _file_sha(path) != EXPECTED_CONFIG_FILE_SHA256:
        raise GravityMissingVariablePreflightError("config file changed")
    config = _read_json(path)
    validate_config(config)
    return config


def _load_sources(root: Path, bindings: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for binding in bindings:
        _strict(
            binding,
            {
                "source_id",
                "path",
                "file_sha256",
                "content_sha256",
                "schema_version",
                "semantic_field",
                "semantic_value",
            },
            "source binding",
        )
        path = _under(root, str(binding["path"]), "source binding")
        if not path.is_file() or _file_sha(path) != binding["file_sha256"]:
            raise GravityMissingVariablePreflightError(
                f"bound source file changed: {binding['source_id']}"
            )
        value = _read_json(path)
        if (
            _declared_content_sha(value) != binding["content_sha256"]
            or value.get("schema_version") != binding["schema_version"]
            or value.get(str(binding["semantic_field"])) != binding["semantic_value"]
        ):
            raise GravityMissingVariablePreflightError(
                f"bound source semantics changed: {binding['source_id']}"
            )
        result[str(binding["source_id"])] = value
    _validate_source_semantics(result)
    return result


def _validate_source_semantics(sources: Mapping[str, Mapping[str, Any]]) -> None:
    strata = sources["cluster_predictor_strata_receipt"]
    matrix = sources["cluster_alternative_cause_matrix"]
    predictor_source = sources["cluster_public_predictor_source_receipt"]
    group = sources["group_scale_source_audit"]
    ben_v2 = sources["shared_ben_real_preflight_v2"]
    if (
        strata["readiness"]["CP5_11_predictor_definition_and_labels_ready"] is not True
        or strata["readiness"]["CP5_13_task_complete"] is not False
        or strata["data_boundary"]["target_or_response_rows_loaded"] != 0
        or strata["data_boundary"]["holdout_rows_loaded"] != 0
        or strata["data_boundary"]["confirmation_rows_loaded"] != 0
        or strata["data_boundary"]["independent_rows_loaded"] != 0
    ):
        raise GravityMissingVariablePreflightError("cluster strata boundary changed")
    expected_causes = {
        "nonthermal_pressure",
        "extra_member_baryons",
        "calibration_shift",
        "gas_clumping",
        "geometry",
        "merger_or_assembly_state",
        "boundary_error",
    }
    if (
        {row["cause_id"] for row in matrix["cause_rows"]} != expected_causes
        or matrix["summary"]["causes"] != 7
        or matrix["summary"]["scientific_comparisons_complete"] != 0
        or matrix["data_boundary"]["target_or_response_rows_loaded"] != 0
        or matrix["data_boundary"]["target_scoring_calls"] != 0
    ):
        raise GravityMissingVariablePreflightError("alternative-cause matrix changed")
    if (
        predictor_source["counts"]["cluster_rows"] != 8
        or predictor_source["counts"]["target_or_response_rows"] != 0
        or predictor_source["data_boundary"]["target_or_response_rows_loaded"] != 0
        or predictor_source["data_boundary"]["target_scoring_calls"] != 0
    ):
        raise GravityMissingVariablePreflightError("public predictor source changed")
    if (
        group["counts"]["candidate_lanes"] != 3
        or group["counts"]["ready_lanes"] != 0
        or group["counts"]["payload_rows_opened"] != 0
        or group["counts"]["scientific_scores_computed"] != 0
        or group["claims"]["CP10_1_complete"] is not False
        or group["claims"]["CP10_2_complete"] is not False
    ):
        raise GravityMissingVariablePreflightError("group source boundary changed")
    if (
        ben_v2["status"] != "v2_pre_score_contract_blocked_no_payload_access"
        or ben_v2["claims"]["all_local_sparc_rows_development_only_for_descendant"] is not True
        or ben_v2["claims"]["local_sparc_confirmation_claim_survives"] is not False
        or ben_v2["claims"]["real_scoring_executed"] is not False
        or ben_v2["mapping_decision"]["blocked_before_payload_load"] is not True
        or ben_v2["production_gate"]["payload_loader_present_in_v2"] is not False
        or any(
            value != 0
            for key, value in ben_v2["zero_access_chronology"].items()
            if key != "v2_contract_frozen_before_payload_access"
        )
    ):
        raise GravityMissingVariablePreflightError("SPARC/X-COP preflight boundary changed")


def _registry_summary(registry: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lane_rows = [
        (str(row["variable_id"]), item) for row in registry for item in row["domain_applicability"]
    ]
    executable_proxy = [
        {"variable_id": variable_id, "lane_id": item["lane_id"]}
        for variable_id, item in lane_rows
        if item["current_execution_status"] == "EXECUTABLE_PROXY_ONLY"
    ]
    source_blocked = [
        {"variable_id": variable_id, "lane_id": item["lane_id"]}
        for variable_id, item in lane_rows
        if item["current_execution_status"] == "SOURCE_BLOCKED"
    ]
    not_applicable = [
        {"variable_id": variable_id, "lane_id": item["lane_id"]}
        for variable_id, item in lane_rows
        if item["current_execution_status"] == "NOT_APPLICABLE"
    ]
    source_definition_blocked = [
        str(row["variable_id"])
        for row in registry
        if str(row["definition_status"]).startswith("SOURCE_DEFINITION_BLOCKED")
    ]
    continuous_definition_frozen = [
        str(row["variable_id"])
        for row in registry
        if row["definition_status"] == "CONTINUOUS_DEFINITION_FROZEN_SOURCE_DATA_BLOCKED"
    ]
    return {
        "executable_proxy_only": executable_proxy,
        "source_blocked": source_blocked,
        "not_applicable": not_applicable,
        "continuous_measurement_ready": [],
        "source_definition_blocked_variables": source_definition_blocked,
        "continuous_definition_frozen_source_data_blocked_variables": (
            continuous_definition_frozen
        ),
    }


def build_receipt(root: Path) -> dict[str, Any]:
    root = root.resolve()
    config = load_config(root)
    _load_sources(root, config["source_bindings"])
    summary = _registry_summary(config["variable_registry"])
    if (
        len(summary["executable_proxy_only"]) != 4
        or len(summary["source_blocked"]) != 16
        or len(summary["not_applicable"]) != 1
        or summary["source_definition_blocked_variables"] != ["nonthermal_pressure", "calibration"]
        or len(summary["continuous_definition_frozen_source_data_blocked_variables"]) != 5
    ):
        raise GravityMissingVariablePreflightError("registry readiness counts changed")
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "preflight_id": config["preflight_id"],
        "status": "target_blind_defined_proxies_frozen_continuous_sources_or_definitions_blocked",
        "decision": "FOUR_DEFINED_XCOP_PROXIES_EXECUTABLE_TWO_CONTINUOUS_VARIABLES_SOURCE_DEFINITION_BLOCKED_ZERO_MEASUREMENTS_SIXTEEN_APPLICABLE_ROWS_SOURCE_BLOCKED_NO_SCORE",
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "file_sha256": _file_sha(root / CONFIG_PATH),
            "content_sha256": _sha(config),
        },
        "implementation_binding": {
            "path": SOURCE_PATH.as_posix(),
            "file_sha256": _file_sha(root / SOURCE_PATH),
            "test_path": TEST_PATH.as_posix(),
            "test_file_sha256": _file_sha(root / TEST_PATH),
        },
        "source_bindings": config["source_bindings"],
        "lane_contract": config["lane_contract"],
        "public_source_catalog": config["public_source_catalog"],
        "variable_registry": config["variable_registry"],
        "registry_summary": summary,
        "chronology_and_access": config["chronology_and_access_contract"],
        "claim_boundary": config["claim_boundary"],
        "counts": {
            "bound_source_receipts_or_artifacts": len(config["source_bindings"]),
            "public_source_requirements": len(config["public_source_catalog"]),
            "variable_families": len(config["variable_registry"]),
            "lane_applicability_rows": 21,
            "executable_proxy_only_rows": len(summary["executable_proxy_only"]),
            "source_blocked_applicable_rows": len(summary["source_blocked"]),
            "not_applicable_rows": len(summary["not_applicable"]),
            "continuous_measurement_ready_rows": 0,
            "defined_proxy_contracts": 4,
            "source_definition_blocked_variables": len(
                summary["source_definition_blocked_variables"]
            ),
            "continuous_definition_frozen_source_data_blocked_variables": len(
                summary["continuous_definition_frozen_source_data_blocked_variables"]
            ),
            "predecessor_public_predictor_rows_read": 8,
            "new_predictor_source_payload_rows_opened": 0,
            "response_or_target_rows_opened": 0,
            "scientific_scores_computed": 0,
        },
        "publication_contract": {
            "publish_primitive": "SAME_DIRECTORY_HARD_LINK_NO_REPLACE",
            "staging_file_fsync_before_publish": True,
            "existing_identical_receipt_action": "RETURN_WITHOUT_REWRITE",
            "existing_different_receipt_action": "FAIL_CLOSED",
            "race_loser_action": "VERIFY_IDENTICAL_OR_FAIL_CLOSED",
            "directory_fsync": "POSIX_WHEN_SUPPORTED_AFTER_PUBLISH_AND_TEMP_CLEANUP",
        },
        "next_actionable_acquisition": config["next_actionable_acquisition"],
        "limitations": [
            "Four X-COP lane rows are executable only as previously frozen categorical or projected proxies; none is a continuous missing-variable measurement.",
            "All locally accessible SPARC rows remain development-only for this descendant, and the mixed local packet cannot supply a confirmation claim.",
            "The group audit has zero ready lanes; public archive existence is not an alias-bound direct-endpoint and covariance packet.",
            "No variable was selected using a response, no formula was scored, and no cause or cross-domain law is supported or refuted.",
        ],
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected = body.pop("content_sha256", None)
    if expected != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityMissingVariablePreflightError("missing-variable receipt changed")


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_no_clobber(root: Path, relative: Path, payload: bytes) -> tuple[Path, str]:
    target = _under(root, relative, "output")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=target.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temp_path, target)
            disposition = "CREATED"
        except FileExistsError as error:
            if target.read_bytes() != payload:
                raise GravityMissingVariablePreflightError(
                    "refused to overwrite different existing output"
                ) from error
            disposition = "EXISTING_IDENTICAL"
        _fsync_directory(target.parent)
        return target, disposition
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_receipt(root: Path) -> tuple[Path, str]:
    receipt = build_receipt(root)
    return _atomic_no_clobber(root.resolve(), OUTPUT_PATH, _canonical_bytes(receipt))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "check", "status"))
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "write":
        path, disposition = write_receipt(root)
        output: Any = {"path": str(path), "disposition": disposition}
    elif args.command == "check":
        receipt = _read_json(root / OUTPUT_PATH)
        validate_receipt(receipt, root)
        output = {
            "status": "PASS",
            "decision": receipt["decision"],
            "content_sha256": receipt["content_sha256"],
            "response_or_target_rows_opened": 0,
            "scientific_scores_computed": 0,
        }
    else:
        receipt = build_receipt(root)
        output = {
            "status": receipt["status"],
            "decision": receipt["decision"],
            "counts": receipt["counts"],
            "next_actionable_acquisition": receipt["next_actionable_acquisition"],
        }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
