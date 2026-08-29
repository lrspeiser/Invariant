from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import pytest

from sigma_theory_compiler import research_publication_readiness as readiness

ROOT = Path(__file__).resolve().parents[1]


def test_current_cluster_result_is_retained_but_not_data_or_paper_ready() -> None:
    receipt = readiness.build_receipt(ROOT)
    assert receipt["decision"] == "DEVELOPMENT_CLUSTER_RESULT_RETAINED_BLOCKED_NOT_DATA_READY"
    assert receipt["claim_tracks"]["bounded_empirical_publication"] == {
        "status": "DEVELOPMENT_RESULT_RETAINED_NOT_PUBLICATION_READY",
        "ready": False,
        "missing_requirements": [
            "full_uncertainty_and_covariance",
            "independent_replication",
            "nuisance_and_alternative_cause_robustness",
            "prior_art_and_reproducibility",
        ],
    }
    assert receipt["automatic_findings"]["bounded_cluster_result_retained"] is True
    assert receipt["automatic_findings"]["galaxy_failure_erased_cluster_result"] is False
    assert receipt["automatic_findings"]["galaxy_failure_blocks_universal_promotion"] is True
    assert receipt["readiness"]["independent_cluster_data"]["next_gate"] == "CP3"
    assert receipt["readiness"]["independent_cluster_data"]["ready"] is False
    assert receipt["readiness"]["observational_authorization"] is False
    assert receipt["readiness"]["independent_target_rows_opened"] == 0
    assert receipt["counts"]["completed_tasks"] == 33
    assert receipt["counts"]["open_tasks"] == 89
    cp3 = next(gate for gate in receipt["gate_ledger"] if gate["gate_id"] == "CP3")
    assert cp3["completed_task_ids"] == [
        "CP3.1",
        "CP3.2",
        "CP3.3",
        "CP3.4",
        "CP3.7",
        "CP3.8",
    ]
    assert cp3["open_task_ids"] == ["CP3.5", "CP3.6"]


def test_adjacent_domain_failure_does_not_veto_a_complete_bounded_claim() -> None:
    policy = readiness.load_policy(ROOT)
    outcome = {field: True for field in policy["outcome_fields"]}
    outcome["adjacent_domain_failures"] = True
    tracks = readiness.classify_claim_tracks(outcome, policy)
    assert tracks["bounded_empirical_publication"]["ready"] is True
    assert tracks["physical_mechanism"]["ready"] is True
    assert tracks["universal_theory"]["ready"] is False
    assert tracks["universal_theory"]["missing_requirements"] == [
        "no_adjacent_domain_failures"
    ]


def test_same_release_confirmation_does_not_substitute_for_independent_replication() -> None:
    policy = readiness.load_policy(ROOT)
    outcome = {field: True for field in policy["outcome_fields"]}
    outcome["same_release_confirmation"] = True
    outcome["independent_replication"] = False
    tracks = readiness.classify_claim_tracks(outcome, policy)
    assert tracks["bounded_empirical_publication"]["ready"] is False
    assert tracks["bounded_empirical_publication"]["missing_requirements"] == [
        "independent_replication"
    ]


def test_policy_rejects_erasure_and_weakened_replication_rules() -> None:
    erased = copy.deepcopy(readiness.load_policy(ROOT))
    erased["core_rules"]["adjacent_domain_failure_erases_bounded_result"] = True
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="weakened"):
        readiness.validate_policy(erased)

    same_release = copy.deepcopy(readiness.load_policy(ROOT))
    same_release["core_rules"][
        "same_release_confirmation_counts_as_independent_replication"
    ] = True
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="weakened"):
        readiness.validate_policy(same_release)


def test_goal_document_and_machine_task_inventory_are_exactly_aligned() -> None:
    policy = readiness.load_policy(ROOT)
    project = readiness.load_project(ROOT, policy)
    document = (ROOT / project["goal_document_binding"]["path"]).read_text(
        encoding="utf-8"
    )
    documented = set(re.findall(r"\*\*(CP(?:[0-9]|1[0-2])\.[0-9]+)\*\*", document))
    configured = {
        task_id for gate in project["gates"] for task_id in gate["task_ids"]
    }
    assert documented == configured
    assert len(configured) == 122
    assert [gate["gate_id"] for gate in project["gates"]] == list(readiness.GATE_ORDER)


def test_goal_checkbox_progress_and_gate_status_fail_closed() -> None:
    policy = readiness.load_policy(ROOT)
    project = readiness.load_project(ROOT, policy)
    path = ROOT / project["goal_document_binding"]["path"]
    partial = copy.deepcopy(project["gates"])
    partial[0]["status"] = "PARTIAL"
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="mixed task"):
        readiness._goal_task_progress(path, partial)

    not_started = copy.deepcopy(project["gates"])
    not_started[0]["status"] = "NOT_STARTED"
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="completed goal"):
        readiness._goal_task_progress(path, not_started)


def test_independent_target_seal_and_evidence_bindings_fail_closed() -> None:
    policy = readiness.load_policy(ROOT)
    opened = copy.deepcopy(readiness.load_project(ROOT, policy))
    opened["independent_data_program"]["independent_target_rows_opened"] = 1
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="target seal"):
        readiness.validate_project(opened, policy, ROOT)

    project = readiness.load_project(ROOT, policy)
    bindings = copy.deepcopy(project["evidence_bindings"])
    bindings[0]["file_sha256"] = "0" * 64
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="evidence file"):
        readiness._load_evidence(ROOT, bindings)


def test_stored_receipt_rebuilds_exactly_and_is_content_bound() -> None:
    stored = json.loads((ROOT / readiness.OUTPUT_PATH).read_text(encoding="utf-8"))
    readiness.validate_receipt(stored, ROOT)
    assert stored == readiness.build_receipt(ROOT)
    assert stored["counts"] == {
        "claim_tracks": 3,
        "gates": 13,
        "tasks": 122,
        "completed_tasks": 33,
        "open_tasks": 89,
        "pass_gates": 3,
        "partial_gates": 4,
        "blocked_gates": 2,
        "not_started_gates": 4,
        "bound_evidence_receipts": 6,
        "independent_target_rows_opened": 0,
    }


def test_semantically_resealed_overclaim_still_fails_against_bound_evidence() -> None:
    stored = readiness.build_receipt(ROOT)
    stored["claims"]["universal_gravity_theory_ready"] = True
    body = {key: value for key, value in stored.items() if key != "content_sha256"}
    stored["content_sha256"] = readiness._sha(body)
    with pytest.raises(readiness.ResearchPublicationReadinessError, match="evidence changed"):
        readiness.validate_receipt(stored, ROOT)
