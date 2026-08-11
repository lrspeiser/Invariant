from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.campaign import CampaignStore
from sigma_theory_compiler.campaign_engine import CampaignEngine
from sigma_theory_compiler.llm_formula_proposal_adapter import (
    canonical_sha256,
    sha256_file,
    validate_proposal_output,
)
from sigma_theory_compiler.typed_dsl_campaign_admission import (
    ADMISSION_TASK_TYPE,
    COMPILER_QUEUE_TASK_TYPE,
    AdmissionConfig,
    CallbackBinding,
    RegisteredCallback,
    ReviewedAdmissionRegistry,
    TypedDSLCampaignAdmission,
    build_admission_readiness_artifact,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
GRAMMAR_PATH = REPO_ROOT / "configs/covariant_action_grammar.json"
CONTRACT_PATH = REPO_ROOT / "configs/covariant_field_contract.json"
CONFIG_PATH = REPO_ROOT / "configs/typed_dsl_campaign_admission.json"
ARTIFACT_PATH = REPO_ROOT / "runs/engine/typed-dsl-campaign-admission-readiness.json"


def _campaign(store: CampaignStore, tmp_path: Path) -> str:
    store.initialize()
    return store.create_campaign(
        {
            "name": "typed DSL admission fixture",
            "project_root": str(tmp_path),
            "output_root": str(tmp_path / "output"),
            "budget": {
                "duration_days": 1,
                "max_tasks": 100,
                "max_failures": 10,
                "max_cycles": 0,
            },
            "runtime": {"lease_seconds": 2},
            "scientific_contract": {
                "observations_authorized": False,
                "dark_matter_or_halo_inputs": False,
                "redshift_distance_inputs": False,
            },
        }
    )


def _output(
    expression: str = "covariant_action(EH_R)",
    *,
    tags: list[str] | None = None,
    proposal_id: str = "candidate:typed:0001",
) -> dict[str, object]:
    return {
        "schema_version": "sigma-formula-proposals-1.0",
        "proposals": [
            {
                "proposal_id": proposal_id,
                "expression": expression,
                "parameters": ["M_Pl"],
                "concept_tags": tags or ["covariant"],
            }
        ],
    }


def _receipt(output: dict[str, object], request_id: str = "proposal:typed:0001"):
    normalized = validate_proposal_output(output)
    return {
        "decision": "quarantined",
        "lineage_sha256": "1" * 64,
        "output_sha256": canonical_sha256(normalized),
        "proposal_sha256": [canonical_sha256(normalized["proposals"][0])],
        "request_id": request_id,
    }


def _binding(kind: str, callback_id: str) -> CallbackBinding:
    return CallbackBinding(
        callback_id=callback_id,
        kind=kind,
        contract_sha256=canonical_sha256({"callback": callback_id, "contract": "fixture-v1"}),
        source_sha256=sha256_file(Path(__file__)),
    )


def _action_spec(terms: list[str], *, matter_metric: str = "g_mu_nu") -> dict[str, object]:
    return {
        "schema_version": "sigma-action-spec-1.0",
        "role": "candidate",
        "fields": ["g_mu_nu"],
        "matter_metric": matter_metric,
        "terms": terms,
        "coefficients": {},
        "universal_constants": ["M_Pl"],
        "parameter_domain": {"positive": ["M_Pl"]},
        "static_dictionary_status": "derived",
    }


def _bridge(
    outputs: dict[str, dict[str, object]],
    *,
    compiler_present: bool = True,
    nonuniversal_compiler: bool = False,
) -> TypedDSLCampaignAdmission:
    resolver_binding = _binding("quarantine_resolver", "fixture_quarantine_resolver")
    compiler_binding = _binding("covariant_compiler", "fixture_covariant_compiler")

    def resolver(locator):
        return outputs[locator["output_sha256"]]

    def compiler(typed_packet, lineage):
        assert lineage["typed_packet_sha256"] == canonical_sha256(typed_packet)
        return {
            "schema_version": "sigma-reviewed-covariant-compiler-callback-1.0",
            "proposal_sha256": typed_packet["proposal_sha256"],
            "validation_artifact_sha256": "2" * 64,
            "action_spec": _action_spec(
                typed_packet["terms"],
                matter_metric="g_tilde" if nonuniversal_compiler else "g_mu_nu",
            ),
        }

    callbacks = [RegisteredCallback(resolver_binding, resolver)]
    if compiler_present:
        callbacks.append(RegisteredCallback(compiler_binding, compiler))
    config = AdmissionConfig(
        execution_enabled=True,
        grammar_sha256=sha256_file(GRAMMAR_PATH),
        field_contract_sha256=sha256_file(CONTRACT_PATH),
        quarantine_resolver_id=resolver_binding.callback_id,
        quarantine_resolver_binding_sha256=resolver_binding.binding_sha256,
        compiler_callback_id=compiler_binding.callback_id,
        compiler_callback_binding_sha256=compiler_binding.binding_sha256,
        maximum_task_attempts=3,
        maximum_admissions=64,
    )
    return TypedDSLCampaignAdmission(
        config=config,
        grammar_path=GRAMMAR_PATH,
        field_contract_path=CONTRACT_PATH,
        registry=ReviewedAdmissionRegistry(tuple(callbacks)),
    )


def _enqueue(
    bridge: TypedDSLCampaignAdmission,
    store: CampaignStore,
    campaign_id: str,
    output: dict[str, object],
    request_id: str = "proposal:typed:0001",
) -> str:
    receipt = _receipt(output, request_id)
    return bridge.enqueue_quarantine(
        store,
        campaign_id,
        quarantine_receipt=receipt,
        raw_output=output,
        selected_proposal_sha256=receipt["proposal_sha256"][0],
    )


def _task_result(store: CampaignStore, task_id: str) -> tuple[str, dict[str, object], int]:
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    return row["status"], json.loads(row["result_json"]), row["attempt"]


def _dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_valid_quarantine_resumes_then_enters_separate_compiler_queue_hash_only(
    tmp_path: Path,
) -> None:
    output = _output()
    receipt = _receipt(output)
    outputs = {receipt["output_sha256"]: output}
    bridge = _bridge(outputs)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    admission_id = _enqueue(bridge, store, campaign_id, output)

    crashed = store.claim_task(campaign_id, "crashed", lease_seconds=-1, allowed_task_types={ADMISSION_TASK_TYPE})
    assert crashed and crashed.task_id == admission_id
    assert store.recover_expired_leases(campaign_id) == {"recovered": 1, "failed": 0}

    admission_engine = CampaignEngine(store, campaign_id, "admit", {ADMISSION_TASK_TYPE})
    bridge.install(admission_engine)
    assert admission_engine.run(max_tasks=1)["processed_tasks"] == 1
    status, admission, attempt = _task_result(store, admission_id)
    assert (status, admission["decision"], admission["enqueued_count"], attempt) == (
        "succeeded",
        "pass",
        1,
        2,
    )
    compiler_id = admission["compiler_queue_task_id"]
    compiler_engine = CampaignEngine(store, campaign_id, "compile", {COMPILER_QUEUE_TASK_TYPE})
    bridge.install(compiler_engine)
    assert compiler_engine.run(max_tasks=1)["processed_tasks"] == 1
    compiler_status, compiler, _ = _task_result(store, compiler_id)
    assert compiler_status == "succeeded"
    assert compiler["decision"] == "pass"
    assert compiler["candidate_id"] == admission["candidate_id"]
    assert compiler["work_lineage_sha256"] == admission["work_lineage_sha256"]

    assert _enqueue(bridge, store, campaign_id, output) == admission_id
    with store.connect() as connection:
        counts = dict(
            connection.execute(
                "SELECT task_type,COUNT(*) FROM tasks GROUP BY task_type"
            ).fetchall()
        )
    assert counts == {ADMISSION_TASK_TYPE: 1, COMPILER_QUEUE_TASK_TYPE: 1}
    persisted = _dump(store.database)
    for body in ("covariant_action(EH_R)", '"terms":["EH_R"]', "mock-secret"):
        assert body not in persisted


def test_missing_reviewed_compiler_blocks_without_enqueue(tmp_path: Path) -> None:
    output = _output()
    receipt = _receipt(output)
    bridge = _bridge({receipt["output_sha256"]: output}, compiler_present=False)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    task_id = _enqueue(bridge, store, campaign_id, output)
    engine = CampaignEngine(store, campaign_id, "blocked", {ADMISSION_TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    status, result, _ = _task_result(store, task_id)
    assert status == "deferred"
    assert result["decision"] == "block"
    assert result["reason"] == "missing_reviewed_covariant_compiler_callback"
    assert result["compiler_tasks_enqueued"] == 0


@pytest.mark.parametrize(
    ("expression", "tags", "reason"),
    [
        ("EH_R+SCALAR_X", ["covariant"], "unsupported_typed_dsl_operator"),
        ("covariant_action(EH_R,ALIEN_TERM)", ["covariant"], "unsupported_covariant_term"),
        (
            "covariant_action(EH_R,z_b)",
            ["covariant"],
            "forbidden_input_or_nonuniversal_matter_seal",
        ),
        (
            "covariant_action(EH_R)",
            ["dark-matter"],
            "forbidden_input_or_nonuniversal_matter_seal",
        ),
        (
            "covariant_action(EH_R)",
            ["halo-fit"],
            "forbidden_input_or_nonuniversal_matter_seal",
        ),
        (
            "covariant_action(EH_R)",
            ["redshift"],
            "forbidden_input_or_nonuniversal_matter_seal",
        ),
    ],
)
def test_unsupported_and_forbidden_proposals_reject_without_enqueue(
    tmp_path: Path, expression: str, tags: list[str], reason: str
) -> None:
    output = _output(expression, tags=tags)
    receipt = _receipt(output)
    bridge = _bridge({receipt["output_sha256"]: output})
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    task_id = _enqueue(bridge, store, campaign_id, output)
    engine = CampaignEngine(store, campaign_id, "reject", {ADMISSION_TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    status, result, _ = _task_result(store, task_id)
    assert status == "succeeded"
    assert result["decision"] == "reject"
    assert result["reason"] == reason
    assert store.task_counts(campaign_id) == {"succeeded": 1}


def test_nonuniversal_compiler_action_is_rejected(tmp_path: Path) -> None:
    output = _output()
    receipt = _receipt(output)
    bridge = _bridge(
        {receipt["output_sha256"]: output},
        nonuniversal_compiler=True,
    )
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    task_id = _enqueue(bridge, store, campaign_id, output)
    engine = CampaignEngine(store, campaign_id, "reject", {ADMISSION_TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    _, result, _ = _task_result(store, task_id)
    assert result["decision"] == "reject"
    assert result["reason"] == "covariant_action_grammar_or_matter_contract_rejected"


def test_quarantine_hash_and_compiler_replay_tamper_reject(tmp_path: Path) -> None:
    output = _output()
    receipt = _receipt(output)
    outputs = {receipt["output_sha256"]: output, "9" * 64: output}
    bridge = _bridge(outputs)
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    task_id = _enqueue(bridge, store, campaign_id, output)
    with store.connect() as connection:
        payload = json.loads(
            connection.execute("SELECT payload_json FROM tasks WHERE task_id=?", (task_id,)).fetchone()[0]
        )
        payload["output_sha256"] = "9" * 64
        connection.execute(
            "UPDATE tasks SET payload_json=? WHERE task_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), task_id),
        )
    engine = CampaignEngine(store, campaign_id, "tamper", {ADMISSION_TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    _, result, _ = _task_result(store, task_id)
    assert result["decision"] == "reject"
    assert result["reason"] == "quarantine_output_hash_mismatch"

    clean_store = CampaignStore(tmp_path / "clean.sqlite")
    clean_campaign = _campaign(clean_store, tmp_path)
    clean_id = _enqueue(bridge, clean_store, clean_campaign, output)
    admission_engine = CampaignEngine(
        clean_store, clean_campaign, "admit", {ADMISSION_TASK_TYPE}
    )
    bridge.install(admission_engine)
    admission_engine.run(max_tasks=1)
    _, admitted, _ = _task_result(clean_store, clean_id)
    compiler_id = admitted["compiler_queue_task_id"]
    with clean_store.connect() as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload_json FROM tasks WHERE task_id=?", (compiler_id,)
            ).fetchone()[0]
        )
        payload["action_ir_sha256"] = "8" * 64
        connection.execute(
            "UPDATE tasks SET payload_json=? WHERE task_id=?",
            (json.dumps(payload, sort_keys=True, separators=(",", ":")), compiler_id),
        )
    compiler_engine = CampaignEngine(
        clean_store, clean_campaign, "compile", {COMPILER_QUEUE_TASK_TYPE}
    )
    bridge.install(compiler_engine)
    compiler_engine.run(max_tasks=1)
    _, replay_result, _ = _task_result(clean_store, compiler_id)
    assert replay_result["decision"] == "reject"
    assert replay_result["reason"] == "compiler_queue_replay_tamper"


def test_checked_in_readiness_is_disabled_and_matches_artifact() -> None:
    built = build_admission_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert built == build_admission_readiness_artifact(REPO_ROOT, CONFIG_PATH)
    assert built["default_execution_enabled"] is False
    assert built["formula_body_persistence"] is False
    assert built["paid_spend_usd"] == "0.000000"
    assert built["fixture_expected_counts"] == {
        "block": 1,
        "enqueue": 1,
        "pass": 1,
        "reject": 9,
    }
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")) == built
