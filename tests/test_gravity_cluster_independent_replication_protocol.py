from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler import gravity_cluster_independent_replication_protocol as protocol

ROOT = Path(__file__).resolve().parents[1]


def test_protocol_freezes_six_cp7_tasks_without_claiming_source_selection() -> None:
    receipt = protocol.build_receipt(ROOT)
    assert receipt["decision"] == (
        "REPLICATION_PROTOCOL_FROZEN_SOURCE_SELECTION_BLOCKED_TARGETS_SEALED"
    )
    assert set(receipt["completed_goal_evidence"]) == {
        "CP7.4",
        "CP7.5",
        "CP7.6",
        "CP7.7",
        "CP7.8",
        "CP7.10",
    }
    assert set(receipt["blocked_goal_evidence"]) == {"CP7.2", "CP7.3", "CP7.9"}
    assert receipt["counts"]["selected_source_lanes"] == 0
    assert receipt["counts"]["independent_target_rows_opened"] == 0
    assert receipt["claims"]["independent_replication_result"] is False


def test_power_thresholds_and_underpowered_classification_are_explicit() -> None:
    receipt = protocol.build_receipt(ROOT)
    frozen = receipt["frozen_decision_summary"]
    assert frozen["confirmatory_target_clusters"] == 192
    assert frozen["underpowered_execution_floor_clusters"] == 120
    assert frozen["minimum_relative_score_improvement_over_each_comparator"] == 0.2
    assert frozen["maximum_catastrophic_cluster_fraction"] == 0.1
    assert frozen["post_access_exclusions_allowed"] is False


def test_whole_cluster_split_is_deterministic_and_response_blind() -> None:
    config = protocol.load_config(ROOT)
    identities = [f"CLUSTER-{index:03d}" for index in range(150)]
    ordered = sorted(identities, key=lambda identity: protocol.split_key(config, identity))
    assignments = {
        identity: protocol.split_identity(config, identity, rank, len(ordered))
        for rank, identity in enumerate(ordered)
    }
    assert list(assignments.values()).count("infrastructure_development") == 24
    assert list(assignments.values()).count("untouched_confirmation") == 126
    assert assignments == {
        identity: protocol.split_identity(config, identity, rank, len(ordered))
        for rank, identity in enumerate(ordered)
    }
    assert config["predictor_blind_split_freeze"]["allowed_split_inputs"] == [
        "canonical_object_id"
    ]


@pytest.mark.parametrize(
    "mutation,match",
    [
        (
            lambda value: value["sample_size_and_stopping_freeze"].__setitem__(
                "confirmatory_target_clusters", 118
            ),
            "sample-size",
        ),
        (
            lambda value: value["primary_decision_freeze"].__setitem__(
                "minimum_relative_score_improvement_over_each_comparator", 0.0
            ),
            "primary decision",
        ),
        (
            lambda value: value["missing_data_and_exclusion_freeze"].__setitem__(
                "post_access_exclusions_allowed", True
            ),
            "missing-data",
        ),
        (
            lambda value: value["authorization_freeze"].__setitem__(
                "observational_authorization", True
            ),
            "authorization",
        ),
        (
            lambda value: value["seals"].__setitem__("target_rows_accessed", True),
            "authorization",
        ),
    ],
)
def test_weakened_thresholds_and_access_mutations_fail_closed(mutation: object, match: str) -> None:
    config = copy.deepcopy(protocol.load_config(ROOT))
    mutation(config)  # type: ignore[operator]
    with pytest.raises(protocol.GravityClusterReplicationProtocolError, match=match):
        protocol.validate_config(config)


def test_stored_receipt_rebuilds_exactly() -> None:
    stored = json.loads((ROOT / protocol.OUTPUT_PATH).read_text(encoding="utf-8"))
    protocol.validate_receipt(stored, ROOT)
    assert stored == protocol.build_receipt(ROOT)
