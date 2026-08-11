from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sigma_theory_compiler.campaign import CampaignStore
from sigma_theory_compiler.campaign_engine import CampaignEngine
from sigma_theory_compiler.campaign_llm_proposal_bridge import (
    TASK_TYPE,
    BridgeConfig,
    CallbackBinding,
    CampaignFormulaProposalBridge,
    CampaignProposalBridgeError,
    RegisteredCallback,
    ReviewedCallbackRegistry,
    build_bridge_readiness_artifact,
)
from sigma_theory_compiler.llm_formula_proposal_adapter import canonical_sha256, sha256_file

REPO_ROOT = Path(__file__).resolve().parents[1]
BRIDGE_CONFIG_PATH = REPO_ROOT / "configs/campaign_llm_proposal_bridge.json"
ARTIFACT_PATH = REPO_ROOT / "runs/engine/campaign-llm-proposal-bridge-readiness.json"
SECRET = "mock-provider-secret-never-persist"
PRIVATE_PROMPT = "Propose a bounded curvature and coherence formula from formal artifacts"
RAW_EXPRESSION = "R + alpha*coherence(phi)"


def _campaign(store: CampaignStore, tmp_path: Path) -> str:
    store.initialize()
    return store.create_campaign(
        {
            "name": "bounded llm proposal bridge test",
            "project_root": str(tmp_path),
            "output_root": str(tmp_path / "output"),
            "budget": {
                "duration_days": 1,
                "max_tasks": 20,
                "max_failures": 5,
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


def _output() -> dict[str, object]:
    return {
        "schema_version": "sigma-formula-proposals-1.0",
        "proposals": [
            {
                "proposal_id": "candidate:mock:0001",
                "expression": RAW_EXPRESSION,
                "parameters": ["alpha"],
                "concept_tags": ["curvature", "coherence"],
            }
        ],
    }


def _binding(kind: str, callback_id: str) -> CallbackBinding:
    return CallbackBinding(
        callback_id=callback_id,
        kind=kind,
        contract_sha256=canonical_sha256({"callback": callback_id, "contract": "test-v1"}),
        source_sha256=sha256_file(Path(__file__)),
    )


def _adapter_config(tmp_path: Path, *, enabled: bool) -> Path:
    path = tmp_path / f"adapter-{enabled}.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "sigma-llm-formula-proposal-adapter-config-1.0",
                "adapter": {
                    "allowed_data_classes": ["formal_artifact", "formula_dsl"],
                    "api_key_env_var": "SIGMA_FORMULA_LLM_API_KEY",
                    "maximum_attempts": 3,
                    "maximum_call_usd": "5.000000",
                    "maximum_total_usd": "500.000000",
                    "paid_calls_enabled": enabled,
                    "provider_id": "deterministic_mock_provider",
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


def _bridge(
    tmp_path: Path,
    provider,
    *,
    execution_enabled: bool = True,
    admission=None,
) -> CampaignFormulaProposalBridge:
    adapter_path = _adapter_config(tmp_path, enabled=execution_enabled)
    provider_binding = _binding("provider", "deterministic_mock_provider")
    resolver_binding = _binding("prompt_resolver", "deterministic_prompt_resolver")
    registered = [
        RegisteredCallback(provider_binding, provider),
        RegisteredCallback(resolver_binding, lambda spec: PRIVATE_PROMPT),
    ]
    admission_binding = None
    if admission is not None:
        admission_binding = _binding("dsl_admission", "reviewed_dsl_admission")
        registered.append(RegisteredCallback(admission_binding, admission))
    config = BridgeConfig(
        execution_enabled=execution_enabled,
        adapter_config_sha256=sha256_file(adapter_path),
        provider_callback_id=provider_binding.callback_id,
        provider_binding_sha256=provider_binding.binding_sha256,
        prompt_resolver_callback_id=resolver_binding.callback_id,
        prompt_resolver_binding_sha256=resolver_binding.binding_sha256,
        admission_callback_id=None if admission_binding is None else admission_binding.callback_id,
        admission_binding_sha256=(
            None if admission_binding is None else admission_binding.binding_sha256
        ),
        maximum_task_attempts=3,
    )
    return CampaignFormulaProposalBridge(
        bridge_config=config,
        adapter_config_path=adapter_path,
        spend_ledger_path=tmp_path / "spend.sqlite",
        registry=ReviewedCallbackRegistry(tuple(registered)),
    )


def _enqueue(bridge: CampaignFormulaProposalBridge, store: CampaignStore, campaign_id: str) -> str:
    return bridge.enqueue(
        store,
        campaign_id,
        request_id="proposal:campaign:0001",
        prompt_spec={"prompt_packet_id": "reviewed-curvature-coherence-v1"},
        prompt_template_sha256="1" * 64,
        context_packets=({"content_sha256": "2" * 64, "data_class": "formal_artifact"},),
        dsl_version="sigma-gravity-dsl-3",
        deterministic_seed=17,
        maximum_call_usd="1.250000",
    )


def _dump(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        return "\n".join(connection.iterdump())


def test_mock_provider_campaign_e2e_is_quarantined_and_secret_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    calls: list[str] = []

    def provider(request, secret):
        assert secret == SECRET
        calls.append(request["idempotency_key"])
        return {"billed_usd": "0.500000", "output": _output()}

    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    bridge = _bridge(tmp_path, provider)
    task_id = _enqueue(bridge, store, campaign_id)
    engine = CampaignEngine(store, campaign_id, "proposal-worker", {TASK_TYPE})
    bridge.install(engine)
    report = engine.run(max_tasks=1)

    assert report["processed_tasks"] == 1
    assert calls == ["proposal:campaign:0001"]
    with store.connect() as connection:
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
        compiler_count = connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_type='proposal_compile'"
        ).fetchone()[0]
    result = json.loads(task["result_json"])
    assert task["status"] == "succeeded"
    assert result["decision"] == "quarantined"
    assert result["quarantine_manifest_complete"] is True
    assert len(result["proposal_sha256"]) == 1
    assert result["compiler_tasks_enqueued"] == 0
    assert compiler_count == 0
    for forbidden in (SECRET, PRIVATE_PROMPT, RAW_EXPRESSION, "transient body"):
        assert forbidden not in _dump(store.database)
        assert forbidden not in _dump(tmp_path / "spend.sqlite")


def test_provider_retry_and_campaign_crash_recovery_are_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    calls: list[str] = []

    def provider(request, _secret):
        calls.append(request["idempotency_key"])
        if len(calls) == 1:
            raise RuntimeError("transient body that must not persist")
        return {"billed_usd": "0.750000", "output": _output()}

    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    bridge = _bridge(tmp_path, provider)
    task_id = _enqueue(bridge, store, campaign_id)

    crashed_task = store.claim_task(campaign_id, "crashed-worker", lease_seconds=-1)
    assert crashed_task and crashed_task.task_id == task_id
    first_outcome = bridge.handle_task(crashed_task)
    assert first_outcome.result["quarantine_manifest_complete"] is True
    assert store.recover_expired_leases(campaign_id) == {"recovered": 1, "failed": 0}

    engine = CampaignEngine(store, campaign_id, "replacement-worker", {TASK_TYPE})
    bridge.install(engine)
    assert engine.run(max_tasks=1)["processed_tasks"] == 1
    with store.connect() as connection:
        task = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    replayed = json.loads(task["result_json"])
    assert task["attempt"] == 2
    assert replayed["replayed"] is True
    assert replayed["quarantine_manifest_complete"] is False
    assert replayed["output_sha256"] == first_outcome.result["output_sha256"]
    assert calls == ["proposal:campaign:0001", "proposal:campaign:0001"]
    assert bridge.ledger.telemetry()["settled_usd"] == "0.750000"


def test_disabled_and_unregistered_callback_paths_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)
    called = 0

    def provider(_request, _secret):
        nonlocal called
        called += 1
        return {"billed_usd": "1.000000", "output": _output()}

    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    disabled = _bridge(tmp_path, provider, execution_enabled=False)
    _enqueue(disabled, store, campaign_id)
    engine = CampaignEngine(store, campaign_id, "disabled-worker", {TASK_TYPE})
    disabled.install(engine)
    engine.run(max_tasks=1)
    assert store.task_counts(campaign_id) == {"deferred": 1}
    assert called == 0
    assert disabled.ledger.telemetry()["settled_usd"] == "0.000000"

    adapter_path = _adapter_config(tmp_path, enabled=True)
    resolver_binding = _binding("prompt_resolver", "deterministic_prompt_resolver")
    provider_binding = _binding("provider", "deterministic_mock_provider")
    missing_provider_config = BridgeConfig(
        execution_enabled=True,
        adapter_config_sha256=sha256_file(adapter_path),
        provider_callback_id=provider_binding.callback_id,
        provider_binding_sha256=provider_binding.binding_sha256,
        prompt_resolver_callback_id=resolver_binding.callback_id,
        prompt_resolver_binding_sha256=resolver_binding.binding_sha256,
        admission_callback_id=None,
        admission_binding_sha256=None,
        maximum_task_attempts=3,
    )
    missing_provider = CampaignFormulaProposalBridge(
        bridge_config=missing_provider_config,
        adapter_config_path=adapter_path,
        spend_ledger_path=tmp_path / "missing-provider-spend.sqlite",
        registry=ReviewedCallbackRegistry(
            (RegisteredCallback(resolver_binding, lambda spec: PRIVATE_PROMPT),)
        ),
    )
    second_store = CampaignStore(tmp_path / "missing-provider-campaign.sqlite")
    second_campaign = _campaign(second_store, tmp_path)
    _enqueue(missing_provider, second_store, second_campaign)
    second_engine = CampaignEngine(
        second_store, second_campaign, "missing-provider-worker", {TASK_TYPE}
    )
    missing_provider.install(second_engine)
    report = second_engine.run(max_tasks=1)
    assert report["retries_scheduled"] == 1
    assert missing_provider.ledger.status("proposal:campaign:0001") is None
    assert SECRET not in _dump(second_store.database)


def test_prompt_body_cannot_be_smuggled_into_durable_task_payload(tmp_path: Path) -> None:
    bridge = _bridge(tmp_path, lambda _request, _secret: {})
    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    with pytest.raises(CampaignProposalBridgeError, match="locators or hashes"):
        bridge.enqueue(
            store,
            campaign_id,
            request_id="proposal:campaign:unsafe",
            prompt_spec={"prompt": PRIVATE_PROMPT},
            prompt_template_sha256="1" * 64,
            context_packets=(
                {"content_sha256": "2" * 64, "data_class": "formal_artifact"},
            ),
            dsl_version="sigma-gravity-dsl-3",
            deterministic_seed=17,
            maximum_call_usd="1.250000",
        )
    assert store.task_counts(campaign_id) == {}

    engine = CampaignEngine(store, campaign_id, "unsafe-seal-worker", {TASK_TYPE})
    engine.config["scientific_contract"]["observations_authorized"] = True
    with pytest.raises(CampaignProposalBridgeError, match="data seals"):
        bridge.install(engine)


def test_reviewed_admission_is_separate_and_still_does_not_enqueue(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SIGMA_FORMULA_LLM_API_KEY", SECRET)

    def provider(_request, _secret):
        return {"billed_usd": "0.100000", "output": _output()}

    def admission(_normalized, _output_sha256):
        return {"decision": "admit", "validation_artifact_sha256": "5" * 64}

    store = CampaignStore(tmp_path / "campaign.sqlite")
    campaign_id = _campaign(store, tmp_path)
    bridge = _bridge(tmp_path, provider, admission=admission)
    _enqueue(bridge, store, campaign_id)
    engine = CampaignEngine(store, campaign_id, "proposal-worker", {TASK_TYPE})
    bridge.install(engine)
    engine.run(max_tasks=1)
    receipt = bridge.review_admission_in_memory(
        request_id="proposal:campaign:0001", raw_output=_output()
    )
    assert receipt["decision"] == "admit"
    assert receipt["compiler_tasks_enqueued"] == 0
    with store.connect() as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM tasks WHERE task_type='proposal_compile'"
        ).fetchone()[0] == 0


def test_checked_in_readiness_is_disabled_zero_spend_and_deterministic() -> None:
    built = build_bridge_readiness_artifact(REPO_ROOT, BRIDGE_CONFIG_PATH)
    assert built == build_bridge_readiness_artifact(REPO_ROOT, BRIDGE_CONFIG_PATH)
    assert built["default_execution_enabled"] is False
    assert built["maximum_total_usd"] == "500.000000"
    assert built["paid_spend_usd"] == "0.000000"
    assert built["network_calls_made"] == 0
    assert built["compiler_tasks_enqueued"] == 0
    assert json.loads(ARTIFACT_PATH.read_text(encoding="utf-8")) == built
