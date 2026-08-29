"""Fail-closed preselection protocol for independent cluster replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

CONFIG_PATH = Path("configs/gravity_cluster_independent_replication_protocol_v1.json")
OUTPUT_PATH = Path(
    "runs/gravity/publication-readiness/independent-replication-protocol-v1.json"
)
CONFIG_SCHEMA = "invariant-gravity-cluster-independent-replication-protocol-1.0"
RECEIPT_SCHEMA = (
    "invariant-gravity-cluster-independent-replication-protocol-receipt-1.0"
)
XCOP_IDS = (
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


class GravityClusterReplicationProtocolError(RuntimeError):
    """Raised when the preregistered independent-replication protocol weakens."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode(
        "utf-8"
    ) + b"\n"


def _sha(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _strict(value: Mapping[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise GravityClusterReplicationProtocolError(f"{label} keys changed")


def load_config(root: Path) -> dict[str, Any]:
    value = json.loads((root.resolve() / CONFIG_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise GravityClusterReplicationProtocolError("replication protocol must be an object")
    validate_config(value)
    return value


def validate_config(config: Mapping[str, Any]) -> None:
    _strict(
        config,
        {
            "schema_version",
            "status",
            "protocol_id",
            "purpose",
            "source_eligibility",
            "population_and_quality_freeze",
            "xcop_exclusion_freeze",
            "identity_and_duplicate_freeze",
            "predictor_blind_split_freeze",
            "primary_decision_freeze",
            "sample_size_and_stopping_freeze",
            "missing_data_and_exclusion_freeze",
            "payload_commitment_template",
            "authorization_freeze",
            "seals",
            "output_path",
        },
        "replication protocol",
    )
    if (
        config["schema_version"] != CONFIG_SCHEMA
        or config["status"] != "frozen_preselection_metadata_only"
        or config["protocol_id"] != "gravity-cluster-independent-replication-protocol-v1"
        or config["output_path"] != OUTPUT_PATH.as_posix()
    ):
        raise GravityClusterReplicationProtocolError("replication protocol identity changed")

    source = config["source_eligibility"]
    if (
        source["current_selected_primary_lane"] is not None
        or source["current_selected_secondary_lane"] is not None
        or len(source["primary_lane_requirements"]) != 7
        or len(source["secondary_lane_requirements"]) != 4
        or "If no lane qualifies, select neither" not in source["source_selection_rule"]
    ):
        raise GravityClusterReplicationProtocolError("source selection seal weakened")

    population = config["population_and_quality_freeze"]
    if (
        population["redshift_range"] != [0.01, 0.6]
        or population["radial_range_R500"] != [0.1, 2.0]
        or population["minimum_radial_bins_per_observable"] != 5
        or population["minimum_predictor_coverage_fraction"] != 0.8
        or set(population["required_primary_observables"])
        != {"electron_pressure_keV_cm-3", "spectroscopic_temperature_keV"}
        or set(population["required_predictors"])
        != {"electron_density_cm-3", "stellar_baryon_profile_solar_mass"}
        or not any("do not exclude" in row for row in population["quality_rules"])
    ):
        raise GravityClusterReplicationProtocolError("population or quality freeze changed")

    xcop = config["xcop_exclusion_freeze"]
    if (
        tuple(xcop["canonical_development_objects"]) != XCOP_IDS
        or xcop["overlaps_resolved"] != 0
        or xcop["source_inventory_available"] is not False
        or not xcop["action"].startswith("Exclude any independent-source object")
    ):
        raise GravityClusterReplicationProtocolError("X-COP exclusion seal changed")

    identity = config["identity_and_duplicate_freeze"]
    if (
        identity["coordinate_match_max_arcmin"] != 2.0
        or identity["redshift_match_max_absolute"] != 0.005
        or identity["ambiguous_match_action"] != "QUARANTINE_OBJECT_BEFORE_TARGET_ACCESS"
        or identity["alias_and_duplicate_ledger_required"] is not True
    ):
        raise GravityClusterReplicationProtocolError("identity or duplicate rule changed")

    split = config["predictor_blind_split_freeze"]
    if (
        split["unit"] != "whole_canonical_cluster_identity"
        or split["hash"] != "sha256"
        or split["allowed_split_inputs"] != ["canonical_object_id"]
        or len(split["forbidden_split_inputs"]) != 9
        or "pressure" not in split["forbidden_split_inputs"]
        or "One authorized opening" not in split["confirmation_access"]
    ):
        raise GravityClusterReplicationProtocolError("predictor-blind split weakened")

    decision = config["primary_decision_freeze"]
    absolute = decision["absolute_accuracy"]
    if (
        not math.isclose(
            absolute["joint_median_absolute_log_residual_max"], math.log(1.25), abs_tol=1e-15
        )
        or not math.isclose(
            absolute["pressure_median_absolute_log_residual_max"],
            math.log(1.3),
            abs_tol=1e-15,
        )
        or not math.isclose(
            absolute["temperature_median_absolute_log_residual_max"],
            math.log(1.4),
            abs_tol=1e-15,
        )
        or decision["minimum_relative_score_improvement_over_each_comparator"] != 0.2
        or decision["maximum_catastrophic_cluster_fraction"] != 0.1
        or len(decision["comparators"]) != 3
        or "No candidate" not in decision["no_repair_rule"]
    ):
        raise GravityClusterReplicationProtocolError("primary decision threshold changed")

    size = config["sample_size_and_stopping_freeze"]
    if (
        size["confirmatory_target_clusters"] != 192
        or size["confirmatory_power_target"] != 0.9
        or size["underpowered_execution_floor_clusters"] != 120
        or size["projected_power_at_floor"] != 0.727
        or "underpowered exploratory" not in size["classification_rule"]
        or "Do not stop early" not in size["stopping_rule"]
    ):
        raise GravityClusterReplicationProtocolError("sample-size or stopping rule weakened")

    missing = config["missing_data_and_exclusion_freeze"]
    if (
        missing["post_access_exclusions_allowed"] is not False
        or len(missing["pre_access_exclusions"]) != 7
        or "never silently clip" not in missing["catastrophic_predictions"]
        or "Do not score outside" not in missing["extrapolation"]
    ):
        raise GravityClusterReplicationProtocolError("missing-data rule weakened")

    commitments = config["payload_commitment_template"]
    if (
        len(commitments["required_before_selection"]) != 11
        or len(commitments["target_manifest_may_expose_before_authorization"]) != 6
        or len(commitments["target_manifest_may_not_expose_before_authorization"]) != 5
        or commitments["current_commitments"] != []
    ):
        raise GravityClusterReplicationProtocolError("payload commitment seal changed")

    authorization = config["authorization_freeze"]
    if (
        authorization["observational_authorization"] is not False
        or authorization["authorized_target_packets"] != []
        or authorization["independent_target_rows_opened"] != 0
        or len(authorization["required_authorization_fields"]) != 11
        or authorization["failure_action"] != "REFUSE_TARGET_ACCESS"
        or any(config["seals"].values())
    ):
        raise GravityClusterReplicationProtocolError("authorization or access seal changed")


def split_identity(config: Mapping[str, Any], canonical_object_id: str, rank: int, n: int) -> str:
    """Return the frozen rank-based split after the caller sorts by :func:`split_key`."""
    if not canonical_object_id or rank < 0 or n < 0 or rank >= n:
        raise GravityClusterReplicationProtocolError("invalid split identity or rank")
    development_count = min(24, n // 5)
    return "infrastructure_development" if rank < development_count else "untouched_confirmation"


def split_key(config: Mapping[str, Any], canonical_object_id: str) -> str:
    if not canonical_object_id:
        raise GravityClusterReplicationProtocolError("empty canonical object identity")
    salt = config["predictor_blind_split_freeze"]["salt"]
    return hashlib.sha256(f"{salt}:{canonical_object_id}".encode("utf-8")).hexdigest()


def build_receipt(root: Path) -> dict[str, Any]:
    config = load_config(root.resolve())
    body = {
        "schema_version": RECEIPT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "decision": "REPLICATION_PROTOCOL_FROZEN_SOURCE_SELECTION_BLOCKED_TARGETS_SEALED",
        "config_binding": {
            "path": CONFIG_PATH.as_posix(),
            "content_sha256": _sha(config),
        },
        "completed_goal_evidence": {
            "CP7.4": "population_selection_quality_radial_and_information_rules_frozen",
            "CP7.5": "whole_cluster_predictor_blind_split_and_use_boundary_frozen",
            "CP7.6": "alias_resolution_duplicate_detection_and_ambiguous_match_rule_frozen",
            "CP7.7": "primary_score_accuracy_comparator_catastrophic_and_per_observable_thresholds_frozen",
            "CP7.8": "preaccess_missingness_exclusions_and_zero_postaccess_exclusion_rule_frozen",
            "CP7.10": "explicit_hash_bound_authorization_required_before_any_target_access",
        },
        "blocked_goal_evidence": {
            "CP7.2": "zero_source_lanes_currently_satisfy_the_frozen_primary_packet_contract",
            "CP7.3": "source_object_inventory_not_selected_so_actual_xcop_overlap_ledger_is_unmaterialized",
            "CP7.9": "payload_commitment_schema_exists_but_no_selected_lane_commitments_exist",
        },
        "frozen_decision_summary": {
            "confirmatory_target_clusters": config["sample_size_and_stopping_freeze"][
                "confirmatory_target_clusters"
            ],
            "underpowered_execution_floor_clusters": config[
                "sample_size_and_stopping_freeze"
            ]["underpowered_execution_floor_clusters"],
            "joint_median_absolute_log_residual_max": config["primary_decision_freeze"][
                "absolute_accuracy"
            ]["joint_median_absolute_log_residual_max"],
            "minimum_relative_score_improvement_over_each_comparator": config[
                "primary_decision_freeze"
            ]["minimum_relative_score_improvement_over_each_comparator"],
            "maximum_catastrophic_cluster_fraction": config["primary_decision_freeze"][
                "maximum_catastrophic_cluster_fraction"
            ],
            "post_access_exclusions_allowed": config[
                "missing_data_and_exclusion_freeze"
            ]["post_access_exclusions_allowed"],
        },
        "counts": {
            "xcop_development_identities": len(XCOP_IDS),
            "primary_lane_requirements": len(
                config["source_eligibility"]["primary_lane_requirements"]
            ),
            "required_payload_commitments": len(
                config["payload_commitment_template"]["required_before_selection"]
            ),
            "materialized_payload_commitments": len(
                config["payload_commitment_template"]["current_commitments"]
            ),
            "selected_source_lanes": 0,
            "resolved_xcop_overlaps": 0,
            "independent_target_rows_opened": 0,
        },
        "claims": {
            "population_and_quality_rules_frozen": True,
            "predictor_blind_whole_cluster_split_frozen": True,
            "identity_and_duplicate_rules_frozen": True,
            "primary_decision_and_stopping_rules_frozen": True,
            "missingness_and_exclusion_rules_frozen": True,
            "explicit_authorization_required": True,
            "source_selected": False,
            "payload_commitment_materialized": False,
            "observational_authorization": False,
            "payload_accessed": False,
            "target_rows_accessed": False,
            "independent_replication_result": False,
        },
        "next_action": "Resolve one complete file-level source packet, materialize its object and target commitments without values, remove and record X-COP overlaps, and only then select primary and secondary lanes.",
    }
    return {**body, "content_sha256": _sha(body)}


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    body = dict(receipt)
    expected_hash = body.pop("content_sha256", None)
    if expected_hash != _sha(body) or dict(receipt) != build_receipt(root):
        raise GravityClusterReplicationProtocolError("replication protocol receipt changed")


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
