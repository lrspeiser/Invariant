from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.action_ir import compile_action_spec, load_action_grammar
from sigma_theory_compiler.campaign import CampaignStore, stable_id
from sigma_theory_compiler.campaign_engine import CampaignEngine
from sigma_theory_compiler.compiler_receipt_registry_bridge import (
    TASK_TYPE,
    BridgeConfig,
    CallbackBinding,
    CompilerReceiptRegistryBridge,
    RegisteredCallback,
    ReviewedRegistryCallbacks,
    build_readiness_artifact,
)
from sigma_theory_compiler.equation_universe import EquationUniverse
from sigma_theory_compiler.formal_backend import load_field_contract
from sigma_theory_compiler.llm_formula_proposal_adapter import canonical_sha256, sha256_file
from sigma_theory_compiler.typed_dsl_campaign_admission import (
    ADMISSION_TASK_TYPE,
    COMPILER_QUEUE_TASK_TYPE,
)

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = ROOT / "configs/covariant_action_grammar.json"
CONTRACT = ROOT / "configs/covariant_field_contract.json"
CONFIG = ROOT / "configs/compiler_receipt_registry_bridge.json"
ARTIFACT = ROOT / "runs/engine/compiler-receipt-registry-bridge-readiness.json"


def _campaign(store: CampaignStore, tmp_path: Path) -> str:
    store.initialize()
    return store.create_campaign(
        {
            "name": "registry bridge fixture",
            "project_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            "budget": {"duration_days": 1, "max_tasks": 100, "max_failures": 10, "max_cycles": 0},
            "runtime": {"lease_seconds": 2},
            "scientific_contract": {
                "observations_authorized": False,
                "dark_matter_or_halo_inputs": False,
                "redshift_distance_inputs": False,
                "prohibited_evidence_patterns": ["dark matter", "halo", "redshift"],
            },
        }
    )


def _spec(**updates):
    value = {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu"],
        "matter_metric": "g_mu_nu",
        "terms": ["EH_R"],
        "coefficients": {},
        "universal_constants": ["M_Pl"],
        "parameter_domain": {"positive": ["M_Pl"]},
        "static_dictionary_status": "derived",
    }
    value.update(updates)
    return value


def _binding(kind: str, callback_id: str) -> CallbackBinding:
    return CallbackBinding(
        callback_id,
        kind,
        canonical_sha256({"contract": callback_id}),
        sha256_file(Path(__file__)),
    )


def _finish(store, campaign_id, task_id, result):
    with store.connect() as connection:
        task_type = connection.execute(
            "SELECT task_type FROM tasks WHERE task_id=?", (task_id,)
        ).fetchone()[0]
    task = store.claim_task(
        campaign_id, f"finish-{task_id}", 30, allowed_task_types={task_type}
    )
    assert task and task.task_id == task_id
    store.finish_task(task, f"finish-{task_id}", "succeeded", result)


def _compiler_receipt(store, campaign_id, proposal_sha: str, source_binding: str) -> str:
    grammar = load_action_grammar(GRAMMAR)
    contract = load_field_contract(CONTRACT)
    action_sha = compile_action_spec(_spec(), grammar, contract)["content_sha256"]
    output_sha = canonical_sha256({"proposal": proposal_sha})
    quarantine_sha = canonical_sha256({"quarantine": proposal_sha})
    validation_sha = "4" * 64
    candidate_id = stable_id(
        "CANDLLM",
        proposal_sha,
        action_sha,
        sha256_file(GRAMMAR),
        sha256_file(CONTRACT),
        source_binding,
    )
    work_sha = canonical_sha256(
        {
            "action_ir_sha256": action_sha,
            "candidate_id": candidate_id,
            "compiler_callback_binding_sha256": source_binding,
            "output_sha256": output_sha,
            "proposal_sha256": proposal_sha,
            "quarantine_lineage_sha256": quarantine_sha,
            "validation_artifact_sha256": validation_sha,
        }
    )
    admission = store.add_task(
        campaign_id,
        ADMISSION_TASK_TYPE,
        stage=92,
        payload={
            "compiler_callback_binding_sha256": source_binding,
            "field_contract_sha256": sha256_file(CONTRACT),
            "grammar_sha256": sha256_file(GRAMMAR),
            "output_sha256": output_sha,
            "proposal_sha256": proposal_sha,
            "quarantine_lineage_sha256": quarantine_sha,
            "request_id": f"proposal:fixture:{proposal_sha[:8]}",
        },
    )
    _finish(store, campaign_id, admission, {"decision": "pass"})
    payload = {
        "action_ir_sha256": action_sha,
        "candidate_id": candidate_id,
        "compiler_callback_binding_sha256": source_binding,
        "field_contract_sha256": sha256_file(CONTRACT),
        "grammar_sha256": sha256_file(GRAMMAR),
        "output_sha256": output_sha,
        "proposal_sha256": proposal_sha,
        "validation_artifact_sha256": validation_sha,
        "work_lineage_sha256": work_sha,
    }
    compiler = store.add_task(campaign_id, COMPILER_QUEUE_TASK_TYPE, stage=93, payload=payload)
    _finish(store, campaign_id, compiler, {**payload, "decision": "pass"})
    return compiler


def _bridge(tmp_path, resolved_spec, *, next_stage=True):
    universe = tmp_path / "universe.sqlite"
    if not universe.exists():
        EquationUniverse(universe).initialize()
    source_binding = "3" * 64
    resolver_binding = _binding("action_receipt_resolver", "fixture_action_resolver")
    next_binding = _binding("next_stage_adapter", "fixture_next_stage")

    def resolver(locator):
        return {
            "schema_version": "sigma-action-receipt-resolution-1.0",
            "proposal_sha256": locator["proposal_sha256"],
            "validation_artifact_sha256": "4" * 64,
            "action_spec": resolved_spec(),
        }

    def adapter(candidate_id, metadata):
        assert metadata["novelty_claim_allowed"] is False
        return {
            "schema_version": "sigma-reviewed-next-stage-adapter-1.0",
            "candidate_id": candidate_id,
            "task_type": "policy_validate",
            "payload": {"candidate_id": candidate_id},
            "adapter_artifact_sha256": "5" * 64,
        }

    callbacks = [RegisteredCallback(resolver_binding, resolver)]
    if next_stage:
        callbacks.append(RegisteredCallback(next_binding, adapter))
    config = BridgeConfig(
        True,
        sha256_file(GRAMMAR),
        sha256_file(CONTRACT),
        sha256_file(universe),
        source_binding,
        resolver_binding.callback_id,
        resolver_binding.binding_sha256,
        next_binding.callback_id if next_stage else None,
        next_binding.binding_sha256 if next_stage else None,
        3,
    )
    return (
        CompilerReceiptRegistryBridge(
            config=config,
            grammar_path=GRAMMAR,
            field_contract_path=CONTRACT,
            equation_universe_path=universe,
            callbacks=ReviewedRegistryCallbacks(tuple(callbacks)),
        ),
        source_binding,
    )


def _run(store, campaign_id, bridge, task_id):
    engine = CampaignEngine(store, campaign_id, "registry", {TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    return row["status"], json.loads(row["result_json"])


def test_pass_registers_hash_only_candidate_and_enqueues_policy(tmp_path: Path):
    bridge, source = _bridge(tmp_path, _spec)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign = _campaign(store, tmp_path)
    compiler = _compiler_receipt(store, campaign, "a" * 64, source)
    task_id = bridge.enqueue_completed_receipt(store, campaign, compiler)
    crashed = store.claim_task(campaign, "crashed", -1, {TASK_TYPE})
    assert crashed and store.recover_expired_leases(campaign) == {"recovered": 1, "failed": 0}
    status, result = _run(store, campaign, bridge, task_id)
    assert (status, result["decision"], result["next_stage_tasks_enqueued"]) == (
        "succeeded", "pass", 1
    )
    with store.connect() as connection:
        candidate = connection.execute("SELECT * FROM candidates").fetchone()
        policy_count = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_type='policy_validate'"
        ).fetchone()[0]
    assert candidate["expression"].startswith("action-ir-sha256:")
    assert result["novelty_claim_allowed"] is False
    assert policy_count == 1
    with sqlite3.connect(store.database) as connection:
        dump = "\n".join(connection.iterdump())
    assert "sqrt(-g)" not in dump and "mock-secret" not in dump


def test_missing_next_adapter_blocks_before_registry_mutation(tmp_path: Path):
    bridge, source = _bridge(tmp_path, _spec, next_stage=False)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign = _campaign(store, tmp_path)
    compiler = _compiler_receipt(store, campaign, "b" * 64, source)
    task_id = bridge.enqueue_completed_receipt(store, campaign, compiler)
    status, result = _run(store, campaign, bridge, task_id)
    assert status == "deferred" and result["decision"] == "block"
    assert result["reason"] == "missing_reviewed_next_stage_adapter"
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 0


def test_second_equivalent_action_is_deduplicated_without_novelty(tmp_path: Path):
    bridge, source = _bridge(tmp_path, _spec)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign = _campaign(store, tmp_path)
    first = _compiler_receipt(store, campaign, "c" * 64, source)
    first_task = bridge.enqueue_completed_receipt(store, campaign, first)
    assert _run(store, campaign, bridge, first_task)[1]["decision"] == "pass"
    second = _compiler_receipt(store, campaign, "d" * 64, source)
    second_task = bridge.enqueue_completed_receipt(store, campaign, second)
    _, result = _run(store, campaign, bridge, second_task)
    assert result["decision"] == "dedup"
    assert result["reason"] == "exact_action_ir_already_registered"
    assert result["novelty_claim_allowed"] is False
    with store.connect() as connection:
        assert connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0] == 1


@pytest.mark.parametrize(
    "mutator",
    [
        lambda value: {**value, "matter_metric": "g_tilde"},
        lambda value: {**value, "terms": ["EH_R", "z_b"]},
        lambda value: {**value, "note": "dark-matter"},
        lambda value: {**value, "note": "halo"},
        lambda value: {**value, "note": "redshift"},
        lambda value: {**value, "coefficients": {"EH_R": "2*M_Pl^2"}},
    ],
)
def test_unsupported_or_forbidden_resolved_actions_reject(tmp_path: Path, mutator):
    bridge, source = _bridge(tmp_path, lambda: mutator(_spec()))
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign = _campaign(store, tmp_path)
    compiler = _compiler_receipt(store, campaign, "e" * 64, source)
    task_id = bridge.enqueue_completed_receipt(store, campaign, compiler)
    _, result = _run(store, campaign, bridge, task_id)
    assert result["decision"] == "reject"
    assert result["novelty_claim_allowed"] is False


def test_registry_replay_tamper_rejects(tmp_path: Path):
    bridge, source = _bridge(tmp_path, _spec)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign = _campaign(store, tmp_path)
    compiler = _compiler_receipt(store, campaign, "f" * 64, source)
    task_id = bridge.enqueue_completed_receipt(store, campaign, compiler)
    with store.connect() as connection:
        payload = json.loads(connection.execute("SELECT payload_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0])
        payload["work_lineage_sha256"] = "9" * 64
        connection.execute("UPDATE tasks SET payload_json=? WHERE task_id=?", (json.dumps(payload), task_id))
    _, result = _run(store, campaign, bridge, task_id)
    assert result["decision"] == "reject"
    assert result["reason"] == "registry_admission_receipt_replay_tamper"


def test_checked_in_readiness_matches():
    built = build_readiness_artifact(ROOT, CONFIG)
    assert built["fixture_expected_counts"] == {
        "block": 1, "dedup": 1, "enqueue": 1, "pass": 1, "reject": 7
    }
    assert json.loads(ARTIFACT.read_text(encoding="utf-8")) == built
