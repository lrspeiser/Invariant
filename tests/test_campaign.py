from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from sigma_theory_compiler.campaign import CampaignStore
from sigma_theory_compiler.campaign_engine import (
    CampaignEngine,
    initialize_campaign,
    validate_proposal,
)
from sigma_theory_compiler.equation_universe import SCHEMA_VERSION, build_equation_universe

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _contract():
    return {
        "prohibited_evidence_patterns": [
            "dark matter",
            "nfw",
            "redshift-derived distance",
            "supernova distance",
        ]
    }


def _config(tmp_path, artifacts=None):
    return {
        "name": "test campaign",
        "project_root": str(tmp_path),
        "output_root": str(tmp_path / "output"),
        "seed": {"candidate_limit": 1, "include_gr_control": False},
        "budget": {
            "duration_days": 1,
            "max_tasks": 100,
            "max_failures": 10,
            "max_cycles": 0,
        },
        "runtime": {"lease_seconds": 2, "research_cycle_seconds": 3600},
        "llm": {"command": []},
        "existing_artifacts": artifacts or {},
        "scientific_contract": _contract(),
    }


def _campaign(store, tmp_path):
    store.initialize()
    return store.create_campaign(_config(tmp_path))


def test_expired_lease_is_recovered_and_task_resumes(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    task_id = store.add_task(
        campaign_id,
        "test",
        stage=0,
        payload={},
        max_attempts=3,
    )
    task = store.claim_task(campaign_id, "crashed-worker", lease_seconds=-1)
    assert task and task.task_id == task_id
    recovered = store.recover_expired_leases(campaign_id)
    assert recovered == {"recovered": 1, "failed": 0}
    resumed = store.claim_task(campaign_id, "replacement-worker", lease_seconds=30)
    assert resumed and resumed.attempt == 2
    store.finish_task(resumed, "replacement-worker", "succeeded", {"ok": True})
    assert store.task_counts(campaign_id) == {"succeeded": 1}


def test_worker_lane_only_claims_allowed_task_types(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    cpu_task = store.add_task(
        campaign_id,
        "symbolic_proxy",
        stage=0,
        payload={"lane": "cpu"},
        priority=1,
    )
    llm_task = store.add_task(
        campaign_id,
        "llm_research",
        stage=0,
        payload={"lane": "llm"},
        priority=100,
    )
    claimed = store.claim_task(
        campaign_id,
        "cpu-worker",
        lease_seconds=30,
        allowed_task_types={"symbolic_proxy"},
    )
    assert claimed and claimed.task_id == cpu_task
    assert claimed.task_id != llm_task
    store.finish_task(claimed, "cpu-worker", "succeeded", {"ok": True})
    assert (
        store.claim_task(
            campaign_id,
            "empty-worker",
            lease_seconds=30,
            allowed_task_types=set(),
        )
        is None
    )
    llm_claimed = store.claim_task(
        campaign_id,
        "llm-worker",
        lease_seconds=30,
        allowed_task_types={"llm_research"},
    )
    assert llm_claimed and llm_claimed.task_id == llm_task


def test_candidate_and_task_registration_are_idempotent(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    first = store.add_candidate(
        campaign_id, kind="test", expression="q+z", canonical={"expression": "q+z"}
    )
    second = store.add_candidate(
        campaign_id, kind="test", expression="q+z", canonical={"expression": "q+z"}
    )
    assert first == second
    task_one = store.add_task(campaign_id, "test", stage=0, payload={"candidate": first})
    task_two = store.add_task(campaign_id, "test", stage=0, payload={"candidate": first})
    assert task_one == task_two
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 1


def test_hard_gate_rejection_is_terminal_but_dossier_survives(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    candidate_id = store.add_candidate(
        campaign_id,
        kind="test",
        expression="q",
        canonical={"expression": "q"},
    )
    store.add_task(
        campaign_id,
        "constraint_analysis",
        stage=3,
        payload={},
        candidate_id=candidate_id,
    )
    store.add_task(
        campaign_id,
        "candidate_dossier",
        stage=80,
        payload={},
        candidate_id=candidate_id,
    )
    store.record_evidence(
        campaign_id,
        candidate_id,
        None,
        {"gate_id": "ghost", "stage": 3, "is_hard": True, "outcome": "reject"},
    )
    assert store.candidate(candidate_id)["status"] == "rejected"
    assert store.task_counts(campaign_id) == {"cancelled": 1, "queued": 1}


def test_dossier_records_known_equation_overlap_without_novelty_claim(tmp_path):
    seed = {
        "schema_version": SCHEMA_VERSION,
        "sources": [
            {
                "source_id": "SRC-TEST",
                "title": "Test baseline",
                "url": "https://example.invalid/test-baseline",
                "authors": ["test"],
                "source_kind": "test",
                "license_id": "CC0-1.0",
                "ingestion_mode": "full",
                "policy_reason": "test fixture",
            }
        ],
        "equations": [
            {
                "equation_id": "EQ-BASELINE-Q",
                "name": "Baseline q correction",
                "domain": "sigma_reduced_candidate",
                "representation": "scalar_sympy",
                "expression": "F = q",
                "variables": [
                    {"symbol": "F", "meaning": "correction", "dimension": {}},
                    {"symbol": "q", "meaning": "state", "dimension": {}},
                ],
                "source_id": "SRC-TEST",
                "independently_encoded": True,
            }
        ],
        "derivations": [],
    }
    seed_path = tmp_path / "equations.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")
    database = tmp_path / "equations.sqlite"
    build_equation_universe(seed_path, database, tmp_path / "build.json")

    store = CampaignStore(tmp_path / "campaign.sqlite")
    config = _config(tmp_path)
    config["prior_art"] = {
        "equation_universe_database": str(database),
        "nearest_limit": 3,
    }
    store.initialize()
    campaign_id = store.create_campaign(config)
    candidate_id = store.add_candidate(
        campaign_id,
        kind="test",
        expression="q",
        canonical={"correction_function": "q"},
    )
    store.add_task(
        campaign_id,
        "candidate_dossier",
        stage=80,
        payload={},
        candidate_id=candidate_id,
    )

    engine = CampaignEngine(store, campaign_id, "test-worker")
    action_screen = engine._equation_prior_art(
        "ACTION-TEST",
        "S = integral sqrt(-g) R",
        representation="tensor_dsl",
    )
    assert action_screen["status"] == "screened"
    assert action_screen["classification"] == "not_found_in_corpus"
    assert action_screen["novelty_claim_allowed"] is False
    result = engine.run(max_tasks=1)
    assert result["processed_tasks"] == 1
    dossier_path = tmp_path / "output" / campaign_id / "dossiers" / f"{candidate_id}.json"
    dossier = json.loads(dossier_path.read_text(encoding="utf-8"))
    prior_art = dossier["equation_prior_art"]
    assert prior_art["classification"] == "known_semantic_equivalent"
    assert prior_art["semantic_matches"][0]["equation_id"] == "EQ-BASELINE-Q"
    assert prior_art["novelty_claim_allowed"] is False
    with store.connect() as connection:
        evidence = connection.execute(
            "SELECT outcome,payload_json FROM evidence WHERE gate_id='equation_prior_art_screen'"
        ).fetchone()
    assert evidence["outcome"] == "pass"
    assert json.loads(evidence["payload_json"])["novelty_claim_allowed"] is False


def test_campaign_fail_closed_rejects_legacy_baryonic_z_lift(tmp_path):
    artifact_payloads = {
        "survivor_audit": {"all_checks_pass": True},
        "dense_gpu_report": {"accounting_pass": True},
        "dense_crosscheck": {"all_cpu_gpu_samples_agree": True},
        "priority_queue": {"schema_version": "sigma-generated-priority-1.0"},
    }
    artifact_paths = {}
    for name, payload in artifact_payloads.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        artifact_paths[name] = str(path)
    queue = {
        "work_queue": [
            {
                "family_id": "GF-TEST",
                "ordinal": 7,
                "term_ids": [3, 19],
                "sign_mask": 3,
                "correction_expression": "+(q)+(sqrt(1+(x*z))-1)",
                "pareto_front": 1,
                "mechanism_tags": ["gradient_state", "measured_state"],
            }
        ]
    }
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(json.dumps(queue), encoding="utf-8")
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = initialize_campaign(store, _config(tmp_path, artifact_paths), queue_path)
    engine = CampaignEngine(store, campaign_id, "test-worker")
    result = engine.run(max_tasks=100)
    assert result["retries_scheduled"] == 0
    status = store.status(campaign_id)
    assert status["candidate_counts"] == {"rejected": 1}
    assert status["hard_gate_evidence"]["reject"] == 1
    assert status["task_counts"]["deferred"] == 1  # offline LLM packet only
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE kind='covariant_lift'"
        ).fetchone()[0] == 0
        evidence = connection.execute(
            "SELECT outcome,payload_json FROM evidence WHERE gate_id='universal_minimal_matter_coupling'"
        ).fetchone()
    assert evidence and evidence["outcome"] == "reject"
    assert json.loads(evidence["payload_json"])["invariant"] == "z_b"
    assert store.integrity_check() == "ok"


def test_proposal_validation_rejects_prohibited_evidence_and_unbounded_search():
    proposal = {
        "proposal_type": "bad",
        "action": "fit an NFW dark matter halo",
        "fields": ["g"],
        "symmetries": ["diffeomorphism"],
        "universal_constants": [],
        "derivative_order": 1,
        "degeneracy_conditions": [],
        "matter_metric": "object_specific",
        "claimed_static_limit": "unknown",
        "expected_dof": "unknown",
        "evasion_rationale": [],
        "falsification_tests": [],
        "literature_overlap": [],
        "bounded_grammar": {"basis": ["R"], "max_terms": 100, "coefficient_alphabet": [1]},
    }
    result = validate_proposal(proposal, _contract())
    assert not result["valid"]
    assert "dark matter" in result["prohibited_matches"]
    assert "nfw" in result["prohibited_matches"]


def test_higher_derivative_proposal_requires_structured_degeneracy_relation() -> None:
    proposal = json.loads((PROJECT_ROOT / "configs" / "proposal_example.json").read_text())
    proposal["derivative_order"] = 2
    proposal["degeneracy_conditions"] = []
    missing = validate_proposal(proposal, _contract())
    assert not missing["valid"]
    assert any("machine-readable degeneracy_conditions" in item for item in missing["errors"])

    proposal["degeneracy_conditions"] = [
        {
            "id": "quadratic_dhost_a1_a2",
            "expression": "a1 + a2",
            "equals": 0,
            "variables": ["a1", "a2"],
            "status": "declared_unverified",
        }
    ]
    declared = validate_proposal(proposal, _contract())
    assert declared["valid"]


def test_pause_and_resume_prevent_and_restore_leasing(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    store.add_task(campaign_id, "test", stage=0, payload={})
    store.set_campaign_state(campaign_id, "paused", "test")
    assert store.claim_task(campaign_id, "worker", 30) is None
    store.set_campaign_state(campaign_id, "active")
    assert store.claim_task(campaign_id, "worker", 30) is not None


def test_heartbeat_extends_active_lease(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    store.add_task(campaign_id, "test", stage=0, payload={})
    task = store.claim_task(campaign_id, "worker", 1)
    assert task
    before = datetime.now(UTC) + timedelta(seconds=20)
    assert store.heartbeat(task.task_id, "worker", 30)
    with store.connect() as connection:
        expiry = connection.execute(
            "SELECT lease_expires_utc FROM tasks WHERE task_id=?", (task.task_id,)
        ).fetchone()[0]
    assert datetime.fromisoformat(expiry) > before


def test_llm_budget_reservations_are_atomic_and_fail_closed(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    status = store.configure_llm_budget(campaign_id, total_budget_usd=5.0, max_calls=2)
    assert status["remaining_usd"] == 5.0

    first = store.reserve_llm_call(
        campaign_id,
        task_id=None,
        provider="test",
        model="test-model",
        max_cost_usd=2.0,
    )
    second = store.reserve_llm_call(
        campaign_id,
        task_id=None,
        provider="test",
        model="test-model",
        max_cost_usd=3.0,
    )
    assert store.llm_budget_status(campaign_id)["reserved_usd"] == 5.0

    with pytest.raises(ValueError, match="call-count budget exhausted"):
        store.reserve_llm_call(
            campaign_id,
            task_id=None,
            provider="test",
            model="test-model",
            max_cost_usd=0.1,
        )

    store.settle_llm_call(first, actual_cost_usd=0.4, status="succeeded")
    # Missing provider metering pessimistically consumes the full $3 reservation.
    final = store.settle_llm_call(second, actual_cost_usd=None, status="failed_unmetered")
    assert final["spent_usd"] == 3.4
    assert final["reserved_usd"] == 0.0
    assert final["remaining_usd"] == 1.6
    assert final["calls_completed"] == 2


def test_llm_budget_cannot_be_raised_after_usage_starts(tmp_path):
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    store.configure_llm_budget(campaign_id, total_budget_usd=5.0, max_calls=2)
    store.reserve_llm_call(
        campaign_id,
        task_id=None,
        provider="test",
        model="test-model",
        max_cost_usd=1.0,
    )
    with pytest.raises(ValueError, match="Refusing to raise"):
        store.configure_llm_budget(campaign_id, total_budget_usd=6.0, max_calls=3)
