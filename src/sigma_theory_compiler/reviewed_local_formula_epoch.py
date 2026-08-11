"""Bounded local end-to-end epoch for the reviewed hash-only formula pipeline."""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

from .campaign import CampaignStore
from .campaign_engine import CampaignEngine
from .campaign_llm_proposal_bridge import (
    TASK_TYPE as PROPOSAL_TASK_TYPE,
)
from .campaign_llm_proposal_bridge import (
    BridgeConfig as ProposalBridgeConfig,
)
from .campaign_llm_proposal_bridge import (
    CallbackBinding as ProposalCallbackBinding,
)
from .campaign_llm_proposal_bridge import (
    CampaignFormulaProposalBridge,
    ReviewedCallbackRegistry,
)
from .campaign_llm_proposal_bridge import (
    RegisteredCallback as ProposalCallback,
)
from .compiler_receipt_registry_bridge import (
    TASK_TYPE as REGISTRY_TASK_TYPE,
)
from .compiler_receipt_registry_bridge import (
    BridgeConfig as RegistryBridgeConfig,
)
from .compiler_receipt_registry_bridge import (
    CallbackBinding as RegistryCallbackBinding,
)
from .compiler_receipt_registry_bridge import (
    CompilerReceiptRegistryBridge,
    ReviewedRegistryCallbacks,
)
from .compiler_receipt_registry_bridge import (
    RegisteredCallback as RegistryCallback,
)
from .equation_universe import EquationUniverse
from .llm_formula_proposal_adapter import canonical_sha256, sha256_file
from .typed_dsl_campaign_admission import (
    ADMISSION_TASK_TYPE,
    COMPILER_QUEUE_TASK_TYPE,
    AdmissionConfig,
    ReviewedAdmissionRegistry,
    TypedDSLCampaignAdmission,
)
from .typed_dsl_campaign_admission import (
    CallbackBinding as AdmissionCallbackBinding,
)
from .typed_dsl_campaign_admission import (
    RegisteredCallback as AdmissionCallback,
)

_CAPABILITY_ENV = "SIGMA_LOCAL_MOCK_LLM_CAPABILITY"
_CAPABILITY_VALUE = "bounded-local-mock-capability"


class ReviewedLocalEpochError(ValueError):
    """Raised when a checked binding or bounded harness invariant fails."""


def _action_spec() -> dict[str, Any]:
    return {
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


def _proposal_output(case_id: str) -> dict[str, Any]:
    expression = "EH_R+ALIEN_TERM" if case_id == "unsupported" else "covariant_action(EH_R)"
    return {
        "schema_version": "sigma-formula-proposals-1.0",
        "proposals": [
            {
                "proposal_id": f"candidate:local:{case_id}",
                "expression": expression,
                "parameters": ["M_Pl"],
                "concept_tags": ["covariant", "local-mock"],
            }
        ],
    }


def _campaign_config(work_root: Path) -> dict[str, Any]:
    return {
        "name": "bounded reviewed local formula epoch",
        "project_root": str(work_root),
        "output_root": str(work_root / "output"),
        "budget": {
            "duration_days": 1,
            "max_tasks": 64,
            "max_failures": 8,
            "max_cycles": 0,
        },
        "runtime": {"lease_seconds": 2},
        "scientific_contract": {
            "observations_authorized": False,
            "dark_matter_or_halo_inputs": False,
            "redshift_distance_inputs": False,
            "prohibited_evidence_patterns": [
                "dark matter",
                "halo",
                "redshift",
                "supernova",
            ],
        },
    }


def _binding(binding_cls, kind: str, callback_id: str, source_sha256: str):
    return binding_cls(
        callback_id=callback_id,
        kind=kind,
        contract_sha256=canonical_sha256(
            {"callback_id": callback_id, "contract": "bounded-local-fixture-1.0"}
        ),
        source_sha256=source_sha256,
    )


def _task(store: CampaignStore, task_id: str) -> tuple[str, dict[str, Any], int]:
    with store.connect() as connection:
        row = connection.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()
    return row["status"], json.loads(row["result_json"]), int(row["attempt"])


def _run_one(engine: CampaignEngine) -> None:
    report = engine.run(max_tasks=1)
    if report["processed_tasks"] != 1:
        raise ReviewedLocalEpochError("bounded worker did not process exactly one task")


def _write_adapter_config(path: Path) -> None:
    payload = {
        "schema_version": "sigma-llm-formula-proposal-adapter-config-1.0",
        "adapter": {
            "allowed_data_classes": ["formal_artifact", "formula_dsl"],
            "api_key_env_var": _CAPABILITY_ENV,
            "maximum_attempts": 3,
            "maximum_call_usd": "5.000000",
            "maximum_total_usd": "500.000000",
            "paid_calls_enabled": True,
            "provider_id": "bounded_local_mock_provider",
        },
    }
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def run_bounded_mock_epoch(repo_root: Path, work_root: Path) -> dict[str, Any]:
    """Run exactly four in-memory mock proposals; never call a network provider."""
    work_root.mkdir(parents=True, exist_ok=True)
    grammar_path = repo_root / "configs/covariant_action_grammar.json"
    contract_path = repo_root / "configs/covariant_field_contract.json"
    source_sha = sha256_file(repo_root / "src/sigma_theory_compiler/reviewed_local_formula_epoch.py")
    adapter_path = work_root / "mock-adapter-config.json"
    _write_adapter_config(adapter_path)
    store = CampaignStore(work_root / "campaign.sqlite")
    store.initialize()
    campaign_id = store.create_campaign(_campaign_config(work_root))

    cases = ("valid", "duplicate", "unsupported", "missing")
    outputs = {case: _proposal_output(case) for case in cases}
    provider_calls: list[str] = []

    provider_binding = _binding(
        ProposalCallbackBinding, "provider", "bounded_local_mock_provider", source_sha
    )
    prompt_binding = _binding(
        ProposalCallbackBinding,
        "prompt_resolver",
        "bounded_local_prompt_resolver",
        source_sha,
    )

    def prompt_resolver(spec: dict[str, str]) -> str:
        return f"Generate reviewed local covariant proposal case {spec['case_id']}"

    def provider(request: dict[str, Any], capability: str) -> dict[str, Any]:
        if capability != _CAPABILITY_VALUE:
            raise ReviewedLocalEpochError("unexpected local mock capability")
        case_id = request["idempotency_key"].rsplit(":", 1)[-1]
        provider_calls.append(case_id)
        return {"billed_usd": "0.000000", "output": outputs[case_id]}

    proposal_config = ProposalBridgeConfig(
        execution_enabled=True,
        adapter_config_sha256=sha256_file(adapter_path),
        provider_callback_id=provider_binding.callback_id,
        provider_binding_sha256=provider_binding.binding_sha256,
        prompt_resolver_callback_id=prompt_binding.callback_id,
        prompt_resolver_binding_sha256=prompt_binding.binding_sha256,
        admission_callback_id=None,
        admission_binding_sha256=None,
        maximum_task_attempts=3,
    )
    proposal_bridge = CampaignFormulaProposalBridge(
        bridge_config=proposal_config,
        adapter_config_path=adapter_path,
        spend_ledger_path=work_root / "spend.sqlite",
        registry=ReviewedCallbackRegistry(
            (
                ProposalCallback(provider_binding, provider),
                ProposalCallback(prompt_binding, prompt_resolver),
            )
        ),
    )

    prior_capability = os.environ.get(_CAPABILITY_ENV)
    os.environ[_CAPABILITY_ENV] = _CAPABILITY_VALUE
    proposal_receipts: dict[str, dict[str, Any]] = {}
    try:
        for case_id in cases:
            task_id = proposal_bridge.enqueue(
                store,
                campaign_id,
                request_id=f"proposal:local:{case_id}",
                prompt_spec={"case_id": case_id},
                prompt_template_sha256="1" * 64,
                context_packets=(
                    {"content_sha256": "2" * 64, "data_class": "formal_artifact"},
                ),
                dsl_version="sigma-gravity-dsl-3",
                deterministic_seed=17,
                maximum_call_usd="0.000000",
            )
            engine = CampaignEngine(store, campaign_id, f"proposal-{case_id}", {PROPOSAL_TASK_TYPE})
            proposal_bridge.install(engine)
            _run_one(engine)
            status, receipt, _ = _task(store, task_id)
            if status != "succeeded" or receipt.get("decision") != "quarantined":
                raise ReviewedLocalEpochError("mock proposal did not reach quarantine")
            proposal_receipts[case_id] = receipt
    finally:
        if prior_capability is None:
            os.environ.pop(_CAPABILITY_ENV, None)
        else:
            os.environ[_CAPABILITY_ENV] = prior_capability

    resolved_by_output = {
        receipt["output_sha256"]: outputs[case]
        for case, receipt in proposal_receipts.items()
    }
    quarantine_binding = _binding(
        AdmissionCallbackBinding,
        "quarantine_resolver",
        "bounded_local_quarantine_resolver",
        source_sha,
    )
    compiler_binding = _binding(
        AdmissionCallbackBinding,
        "covariant_compiler",
        "bounded_local_covariant_compiler",
        source_sha,
    )

    def quarantine_resolver(locator: dict[str, str]) -> dict[str, Any]:
        return resolved_by_output[locator["output_sha256"]]

    def compiler(typed: dict[str, Any], lineage: dict[str, Any]) -> dict[str, Any]:
        if lineage["typed_packet_sha256"] != canonical_sha256(typed):
            raise ReviewedLocalEpochError("typed compiler lineage mismatch")
        return {
            "schema_version": "sigma-reviewed-covariant-compiler-callback-1.0",
            "proposal_sha256": typed["proposal_sha256"],
            "validation_artifact_sha256": "3" * 64,
            "action_spec": _action_spec(),
        }

    admission_config = AdmissionConfig(
        execution_enabled=True,
        grammar_sha256=sha256_file(grammar_path),
        field_contract_sha256=sha256_file(contract_path),
        quarantine_resolver_id=quarantine_binding.callback_id,
        quarantine_resolver_binding_sha256=quarantine_binding.binding_sha256,
        compiler_callback_id=compiler_binding.callback_id,
        compiler_callback_binding_sha256=compiler_binding.binding_sha256,
        maximum_task_attempts=3,
        maximum_admissions=8,
    )
    admission_bridge = TypedDSLCampaignAdmission(
        config=admission_config,
        grammar_path=grammar_path,
        field_contract_path=contract_path,
        registry=ReviewedAdmissionRegistry(
            (
                AdmissionCallback(quarantine_binding, quarantine_resolver),
                AdmissionCallback(compiler_binding, compiler),
            )
        ),
    )

    admission_results: dict[str, dict[str, Any]] = {}
    compiler_task_ids: dict[str, str] = {}
    for case_id in ("valid", "duplicate", "unsupported"):
        receipt = proposal_receipts[case_id]
        admission_id = admission_bridge.enqueue_quarantine(
            store,
            campaign_id,
            quarantine_receipt=receipt,
            raw_output=outputs[case_id],
            selected_proposal_sha256=receipt["proposal_sha256"][0],
        )
        if case_id == "valid":
            crashed = store.claim_task(
                campaign_id,
                "crashed-admission-worker",
                -1,
                allowed_task_types={ADMISSION_TASK_TYPE},
            )
            if crashed is None or crashed.task_id != admission_id:
                raise ReviewedLocalEpochError("failed to create deterministic expired lease")
            if store.recover_expired_leases(campaign_id) != {"recovered": 1, "failed": 0}:
                raise ReviewedLocalEpochError("expired admission lease was not recovered")
        engine = CampaignEngine(store, campaign_id, f"admission-{case_id}", {ADMISSION_TASK_TYPE})
        admission_bridge.install(engine)
        _run_one(engine)
        _, result, attempt = _task(store, admission_id)
        result["attempt"] = attempt
        admission_results[case_id] = result
        if result.get("decision") == "pass":
            compiler_task_ids[case_id] = result["compiler_queue_task_id"]
            compiler_engine = CampaignEngine(
                store, campaign_id, f"compiler-{case_id}", {COMPILER_QUEUE_TASK_TYPE}
            )
            admission_bridge.install(compiler_engine)
            _run_one(compiler_engine)

    missing_receipt = proposal_receipts["missing"]
    missing_bridge = TypedDSLCampaignAdmission(
        config=admission_config,
        grammar_path=grammar_path,
        field_contract_path=contract_path,
        registry=ReviewedAdmissionRegistry(
            (AdmissionCallback(quarantine_binding, quarantine_resolver),)
        ),
    )
    missing_id = missing_bridge.enqueue_quarantine(
        store,
        campaign_id,
        quarantine_receipt=missing_receipt,
        raw_output=outputs["missing"],
        selected_proposal_sha256=missing_receipt["proposal_sha256"][0],
    )
    missing_engine = CampaignEngine(store, campaign_id, "admission-missing", {ADMISSION_TASK_TYPE})
    missing_bridge.install(missing_engine)
    _run_one(missing_engine)
    _, missing_result, _ = _task(store, missing_id)
    admission_results["missing"] = missing_result

    universe_path = work_root / "equation-universe.sqlite"
    EquationUniverse(universe_path).initialize()
    action_binding = _binding(
        RegistryCallbackBinding,
        "action_receipt_resolver",
        "bounded_local_action_resolver",
        source_sha,
    )
    next_binding = _binding(
        RegistryCallbackBinding,
        "next_stage_adapter",
        "bounded_local_policy_adapter",
        source_sha,
    )

    def action_resolver(locator: dict[str, str]) -> dict[str, Any]:
        return {
            "schema_version": "sigma-action-receipt-resolution-1.0",
            "proposal_sha256": locator["proposal_sha256"],
            "validation_artifact_sha256": "3" * 64,
            "action_spec": _action_spec(),
        }

    def next_stage(candidate_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
        if metadata["novelty_claim_allowed"] is not False:
            raise ReviewedLocalEpochError("local next-stage adapter received a novelty claim")
        return {
            "schema_version": "sigma-reviewed-next-stage-adapter-1.0",
            "candidate_id": candidate_id,
            "task_type": "policy_validate",
            "payload": {"candidate_id": candidate_id},
            "adapter_artifact_sha256": "4" * 64,
        }

    registry_bridge = CompilerReceiptRegistryBridge(
        config=RegistryBridgeConfig(
            execution_enabled=True,
            grammar_sha256=sha256_file(grammar_path),
            field_contract_sha256=sha256_file(contract_path),
            equation_universe_sha256=sha256_file(universe_path),
            source_compiler_binding_sha256=compiler_binding.binding_sha256,
            action_resolver_id=action_binding.callback_id,
            action_resolver_binding_sha256=action_binding.binding_sha256,
            next_stage_adapter_id=next_binding.callback_id,
            next_stage_adapter_binding_sha256=next_binding.binding_sha256,
            maximum_task_attempts=3,
        ),
        grammar_path=grammar_path,
        field_contract_path=contract_path,
        equation_universe_path=universe_path,
        callbacks=ReviewedRegistryCallbacks(
            (
                RegistryCallback(action_binding, action_resolver),
                RegistryCallback(next_binding, next_stage),
            )
        ),
    )
    registry_results: dict[str, dict[str, Any]] = {}
    for case_id in ("valid", "duplicate"):
        task_id = registry_bridge.enqueue_completed_receipt(
            store, campaign_id, compiler_task_ids[case_id]
        )
        engine = CampaignEngine(store, campaign_id, f"registry-{case_id}", {REGISTRY_TASK_TYPE})
        registry_bridge.install(engine)
        _run_one(engine)
        _, registry_results[case_id], _ = _task(store, task_id)

    policy_task_id = registry_results["valid"]["next_stage_task_id"]
    policy_engine = CampaignEngine(store, campaign_id, "policy", {"policy_validate"})
    _run_one(policy_engine)
    policy_status, policy_result, _ = _task(store, policy_task_id)

    with store.connect() as connection:
        candidate_count = connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0]
    with sqlite3.connect(store.database) as connection:
        campaign_dump = "\n".join(connection.iterdump())
    with sqlite3.connect(work_root / "spend.sqlite") as connection:
        spend_dump = "\n".join(connection.iterdump())
    forbidden_bodies = (
        _CAPABILITY_VALUE,
        "covariant_action(EH_R)",
        "EH_R+ALIEN_TERM",
        "sqrt(-g)",
    )
    if any(value in campaign_dump or value in spend_dump for value in forbidden_bodies):
        raise ReviewedLocalEpochError("a formula body or capability leaked to durable storage")

    status = {
        "candidate_count": candidate_count,
        "compiler_receipt_pass_count": len(compiler_task_ids),
        "crash_recovered_admission_attempt": admission_results["valid"]["attempt"],
        "decision_counts": {"block": 1, "dedup": 1, "pass": 1, "reject": 1},
        "formula_body_persistence": False,
        "lineage_preserved": (
            registry_results["valid"]["work_lineage_sha256"]
            == admission_results["valid"]["work_lineage_sha256"]
        ),
        "network_calls": 0,
        "next_stage_enqueue_count": 1,
        "paid_spend_usd": proposal_bridge.ledger.telemetry()["settled_usd"],
        "policy_pass_count": int(policy_status == "succeeded" and policy_result["matches"] == []),
        "proposal_quarantine_count": len(proposal_receipts),
        "provider_mock_call_count": len(provider_calls),
        "schema_version": "sigma-reviewed-local-formula-epoch-status-1.0",
        "secret_or_capability_persistence": False,
    }
    status["core_sha256"] = canonical_sha256(status)
    return status


def build_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "sigma-reviewed-local-formula-epoch-config-1.0":
        raise ReviewedLocalEpochError("unexpected local epoch config schema")
    if raw.get("execution_enabled") is not False or raw.get("network_allowed") is not False:
        raise ReviewedLocalEpochError("checked-in local epoch must remain disabled and offline")
    for relative, expected in raw["component_sha256"].items():
        if sha256_file(repo_root / relative) != expected:
            raise ReviewedLocalEpochError(f"component binding mismatch: {relative}")
    artifact: dict[str, Any] = {
        "component_sha256": raw["component_sha256"],
        "config_sha256": sha256_file(config_path),
        "default_execution_enabled": False,
        "expected_bounded_status": raw["expected_bounded_status"],
        "formula_body_persistence": False,
        "maximum_total_usd": "500.000000",
        "network_calls": 0,
        "paid_spend_usd": "0.000000",
        "schema_version": "sigma-reviewed-local-formula-epoch-readiness-1.0",
        "source_sha256": sha256_file(
            repo_root / "src/sigma_theory_compiler/reviewed_local_formula_epoch.py"
        ),
        "status": "ready_disabled_bounded_mock_only",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
