"""Durable campaign bridge for secret-safe, quarantined LLM formula proposals.

The campaign database receives hashes and bounded status fields only. Prompt and
provider-output bodies exist only in process memory. This bridge deliberately has
no compiler-enqueue operation; admission is a separate reviewed callback step.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .campaign import CampaignStore, ClaimedTask, stable_id
from .campaign_engine import CampaignEngine, WorkerOutcome
from .llm_formula_proposal_adapter import (
    AdapterConfig,
    FormulaProposalAdapter,
    ProposalAdapterError,
    ProposalRequest,
    SpendLedger,
    canonical_sha256,
    sha256_bytes,
    sha256_file,
    validate_proposal_output,
)

TASK_TYPE = "reviewed_llm_formula_proposal"
_SHA_RE = re.compile(r"^[0-9a-f]{64}$")


class CampaignProposalBridgeError(ValueError):
    """Raised for an unreviewed callback or invalid hash-bound task."""


@dataclass(frozen=True)
class CallbackBinding:
    callback_id: str
    kind: str
    contract_sha256: str
    source_sha256: str
    version: str = "1.0"

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_.:-]{2,127}", self.callback_id):
            raise CampaignProposalBridgeError("invalid callback ID")
        if self.kind not in {"provider", "prompt_resolver", "dsl_admission"}:
            raise CampaignProposalBridgeError("invalid callback kind")
        if not _SHA_RE.fullmatch(self.contract_sha256) or not _SHA_RE.fullmatch(
            self.source_sha256
        ):
            raise CampaignProposalBridgeError("callback binding is not hash-bound")

    @property
    def binding_sha256(self) -> str:
        self.validate()
        return canonical_sha256(
            {
                "callback_id": self.callback_id,
                "contract_sha256": self.contract_sha256,
                "kind": self.kind,
                "source_sha256": self.source_sha256,
                "version": self.version,
            }
        )


@dataclass(frozen=True)
class RegisteredCallback:
    binding: CallbackBinding
    callback: Callable[..., Any]


class ReviewedCallbackRegistry:
    """In-memory allowlist; callable objects and secrets are never serialized."""

    def __init__(self, callbacks: tuple[RegisteredCallback, ...] = ()) -> None:
        self._callbacks: dict[tuple[str, str], RegisteredCallback] = {}
        for registered in callbacks:
            registered.binding.validate()
            key = (registered.binding.kind, registered.binding.callback_id)
            if key in self._callbacks:
                raise CampaignProposalBridgeError("duplicate callback registration")
            self._callbacks[key] = registered

    def resolve(self, kind: str, callback_id: str, expected_binding_sha256: str) -> Callable[..., Any]:
        registered = self._callbacks.get((kind, callback_id))
        if registered is None:
            raise CampaignProposalBridgeError("reviewed callback is not registered")
        if registered.binding.binding_sha256 != expected_binding_sha256:
            raise CampaignProposalBridgeError("reviewed callback binding hash mismatch")
        return registered.callback


@dataclass(frozen=True)
class BridgeConfig:
    execution_enabled: bool
    adapter_config_sha256: str
    provider_callback_id: str
    provider_binding_sha256: str
    prompt_resolver_callback_id: str
    prompt_resolver_binding_sha256: str
    admission_callback_id: str | None
    admission_binding_sha256: str | None
    maximum_task_attempts: int

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> BridgeConfig:
        required = {
            "adapter_config_sha256",
            "admission_binding_sha256",
            "admission_callback_id",
            "execution_enabled",
            "maximum_task_attempts",
            "prompt_resolver_binding_sha256",
            "prompt_resolver_callback_id",
            "provider_binding_sha256",
            "provider_callback_id",
        }
        if set(raw) != required:
            raise CampaignProposalBridgeError("bridge config shape mismatch")
        if not isinstance(raw["execution_enabled"], bool):
            raise CampaignProposalBridgeError("execution_enabled must be boolean")
        hashes = (
            raw["adapter_config_sha256"],
            raw["provider_binding_sha256"],
            raw["prompt_resolver_binding_sha256"],
        )
        if any(not _SHA_RE.fullmatch(str(value)) for value in hashes):
            raise CampaignProposalBridgeError("bridge config contains an invalid hash")
        admission_id = raw["admission_callback_id"]
        admission_sha = raw["admission_binding_sha256"]
        if (admission_id is None) != (admission_sha is None):
            raise CampaignProposalBridgeError("admission callback ID and hash must be paired")
        if admission_sha is not None and not _SHA_RE.fullmatch(str(admission_sha)):
            raise CampaignProposalBridgeError("admission callback hash is invalid")
        maximum_task_attempts = int(raw["maximum_task_attempts"])
        if not 1 <= maximum_task_attempts <= 8:
            raise CampaignProposalBridgeError("maximum task attempts must be in [1, 8]")
        return cls(
            execution_enabled=bool(raw["execution_enabled"]),
            adapter_config_sha256=str(raw["adapter_config_sha256"]),
            provider_callback_id=str(raw["provider_callback_id"]),
            provider_binding_sha256=str(raw["provider_binding_sha256"]),
            prompt_resolver_callback_id=str(raw["prompt_resolver_callback_id"]),
            prompt_resolver_binding_sha256=str(raw["prompt_resolver_binding_sha256"]),
            admission_callback_id=None if admission_id is None else str(admission_id),
            admission_binding_sha256=None if admission_sha is None else str(admission_sha),
            maximum_task_attempts=maximum_task_attempts,
        )


class CampaignFormulaProposalBridge:
    """Installs one reviewed proposal handler into an existing campaign worker."""

    def __init__(
        self,
        *,
        bridge_config: BridgeConfig,
        adapter_config_path: Path,
        spend_ledger_path: Path,
        registry: ReviewedCallbackRegistry,
    ) -> None:
        if sha256_file(adapter_config_path) != bridge_config.adapter_config_sha256:
            raise CampaignProposalBridgeError("adapter config file hash mismatch")
        raw = json.loads(adapter_config_path.read_text(encoding="utf-8"))
        self.adapter_config = AdapterConfig.from_mapping(raw["adapter"])
        if self.adapter_config.provider_id != bridge_config.provider_callback_id:
            raise CampaignProposalBridgeError("adapter provider ID is not the reviewed callback")
        if bridge_config.execution_enabled and not self.adapter_config.paid_calls_enabled:
            raise CampaignProposalBridgeError("execution cannot open while paid calls are disabled")
        self.config = bridge_config
        self.registry = registry
        self.ledger = SpendLedger(spend_ledger_path, self.adapter_config)

    def install(self, engine: CampaignEngine) -> None:
        if engine.allowed_task_types is not None and TASK_TYPE not in engine.allowed_task_types:
            raise CampaignProposalBridgeError("worker lane does not allow proposal task type")
        contract = engine.config.get("scientific_contract", {})
        required_seals = {
            "dark_matter_or_halo_inputs": False,
            "observations_authorized": False,
            "redshift_distance_inputs": False,
        }
        if any(contract.get(key) is not expected for key, expected in required_seals.items()):
            raise CampaignProposalBridgeError("campaign scientific data seals are not closed")
        engine.handlers[TASK_TYPE] = self.handle_task

    def enqueue(
        self,
        store: CampaignStore,
        campaign_id: str,
        *,
        request_id: str,
        prompt_spec: Mapping[str, Any],
        prompt_template_sha256: str,
        context_packets: tuple[Mapping[str, str], ...],
        dsl_version: str,
        deterministic_seed: int,
        maximum_call_usd: str,
    ) -> str:
        prompt = self._resolve_prompt(prompt_spec)
        request = ProposalRequest(
            request_id=request_id,
            prompt=prompt,
            prompt_template_sha256=prompt_template_sha256,
            context_packets=context_packets,
            dsl_version=dsl_version,
            deterministic_seed=deterministic_seed,
            maximum_call_usd=maximum_call_usd,
        )
        validated = request.validate(self.adapter_config)
        safe_payload = {
            "context_packets": [dict(packet) for packet in context_packets],
            "deterministic_seed": deterministic_seed,
            "dsl_version": dsl_version,
            "maximum_call_usd": maximum_call_usd,
            "prompt_sha256": validated["prompt_sha256"],
            "prompt_spec": dict(prompt_spec),
            "prompt_spec_sha256": canonical_sha256(prompt_spec),
            "prompt_template_sha256": prompt_template_sha256,
            "request_id": request_id,
        }
        self._validate_prompt_spec(prompt_spec)
        return store.add_task(
            campaign_id,
            TASK_TYPE,
            stage=91,
            payload=safe_payload,
            max_attempts=self.config.maximum_task_attempts,
            diversity_bucket="reviewed-llm-proposal",
            idempotency_key=stable_id("LLMREQ", campaign_id, request_id, validated["lineage_sha256"]),
        )

    @staticmethod
    def _validate_prompt_spec(value: Mapping[str, Any]) -> None:
        """Permit locators and hashes, never free-form prompt fragments."""
        if not value or len(value) > 16:
            raise CampaignProposalBridgeError("prompt spec is empty or too large")
        for key, item in value.items():
            name = str(key)
            if not re.fullmatch(r"[a-z][a-z0-9_]{0,54}_(?:id|sha256)", name):
                raise CampaignProposalBridgeError("prompt spec keys must be locators or hashes")
            if (
                not isinstance(item, str)
                or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,128}", item)
            ):
                raise CampaignProposalBridgeError("prompt spec value is not a bounded locator")
            if name.endswith("_sha256") and not _SHA_RE.fullmatch(item):
                raise CampaignProposalBridgeError("prompt spec digest is invalid")

    def _resolve_prompt(self, prompt_spec: Mapping[str, Any]) -> str:
        try:
            resolver = self.registry.resolve(
                "prompt_resolver",
                self.config.prompt_resolver_callback_id,
                self.config.prompt_resolver_binding_sha256,
            )
            prompt = resolver(dict(prompt_spec))
        except CampaignProposalBridgeError:
            raise
        except Exception:  # noqa: BLE001 - callback errors must be sanitized before persistence
            raise CampaignProposalBridgeError("prompt resolver failed") from None
        if not isinstance(prompt, str):
            raise CampaignProposalBridgeError("prompt resolver did not return text")
        return prompt

    def handle_task(self, task: ClaimedTask) -> WorkerOutcome:
        if not self.config.execution_enabled:
            return WorkerOutcome(
                status="deferred",
                result=self._safe_result(task, "blocked", "campaign_proposal_execution_disabled"),
            )
        try:
            provider = self.registry.resolve(
                "provider",
                self.config.provider_callback_id,
                self.config.provider_binding_sha256,
            )
            payload = self._validate_task_payload(task.payload)
            self._validate_prompt_spec(payload["prompt_spec"])
            prompt = self._resolve_prompt(payload["prompt_spec"])
            if sha256_bytes(prompt.encode("utf-8")) != payload["prompt_sha256"]:
                raise CampaignProposalBridgeError("resolved prompt hash mismatch")
            request = ProposalRequest(
                request_id=payload["request_id"],
                prompt=prompt,
                prompt_template_sha256=payload["prompt_template_sha256"],
                context_packets=tuple(payload["context_packets"]),
                dsl_version=payload["dsl_version"],
                deterministic_seed=payload["deterministic_seed"],
                maximum_call_usd=payload["maximum_call_usd"],
            )
            validated_request = request.validate(self.adapter_config)
            result = FormulaProposalAdapter(self.adapter_config, self.ledger, provider).propose(request)
        except CampaignProposalBridgeError:
            raise
        except ProposalAdapterError:
            raise CampaignProposalBridgeError("proposal adapter rejected the request") from None
        except Exception:  # noqa: BLE001 - provider boundary failures must persist no bodies
            raise CampaignProposalBridgeError("proposal adapter execution failed") from None

        decision = str(result["decision"])
        if decision == "blocked":
            return WorkerOutcome(
                status="deferred",
                result=self._safe_result(task, decision, str(result["reason"])),
            )
        proposal_hashes: list[str] = []
        if "output" in result:
            output = validate_proposal_output(result["output"])
            proposal_hashes = [canonical_sha256(item) for item in output["proposals"]]
        safe = {
            "admission_status": "not_requested",
            "compiler_tasks_enqueued": 0,
            "decision": decision,
            "downstream_validation_required": decision == "quarantined",
            "lineage_sha256": validated_request["lineage_sha256"],
            "output_sha256": result.get("output_sha256"),
            "proposal_sha256": proposal_hashes,
            "quarantine_manifest_complete": bool(proposal_hashes),
            "request_id": request.request_id,
            "replayed": bool(result.get("replayed", False)),
            "settled_usd": result.get("settled_usd", "0.000000"),
            "task_id": task.task_id,
        }
        status = "succeeded" if decision == "quarantined" else "deferred"
        return WorkerOutcome(status=status, result=safe)

    @staticmethod
    def _validate_task_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
        required = {
            "context_packets",
            "deterministic_seed",
            "dsl_version",
            "maximum_call_usd",
            "prompt_sha256",
            "prompt_spec",
            "prompt_spec_sha256",
            "prompt_template_sha256",
            "request_id",
        }
        if set(payload) != required:
            raise CampaignProposalBridgeError("proposal task payload shape mismatch")
        if canonical_sha256(payload["prompt_spec"]) != payload["prompt_spec_sha256"]:
            raise CampaignProposalBridgeError("prompt spec hash mismatch")
        if not _SHA_RE.fullmatch(str(payload["prompt_sha256"])):
            raise CampaignProposalBridgeError("prompt hash is invalid")
        return dict(payload)

    @staticmethod
    def _safe_result(task: ClaimedTask, decision: str, reason: str) -> dict[str, Any]:
        return {
            "admission_status": "not_requested",
            "compiler_tasks_enqueued": 0,
            "decision": decision,
            "reason": reason,
            "request_id": task.payload.get("request_id"),
            "task_id": task.task_id,
        }

    def review_admission_in_memory(
        self,
        *,
        request_id: str,
        raw_output: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Validate a quarantined body in memory; this never enqueues compiler work."""
        if self.config.admission_callback_id is None or self.config.admission_binding_sha256 is None:
            raise CampaignProposalBridgeError("no reviewed DSL admission callback configured")
        callback = self.registry.resolve(
            "dsl_admission",
            self.config.admission_callback_id,
            self.config.admission_binding_sha256,
        )
        normalized = validate_proposal_output(raw_output)
        output_sha256 = canonical_sha256(normalized)
        ledger_row = self.ledger.status(request_id)
        if ledger_row is None or ledger_row["status"] != "settled":
            raise CampaignProposalBridgeError("proposal request has no settled quarantine")
        if ledger_row["output_sha256"] != output_sha256:
            raise CampaignProposalBridgeError("admission output hash differs from quarantine")
        try:
            reviewed = callback(normalized, output_sha256)
        except Exception:  # noqa: BLE001 - callback errors must be sanitized before persistence
            raise CampaignProposalBridgeError("reviewed DSL admission callback failed") from None
        if not isinstance(reviewed, Mapping) or set(reviewed) != {
            "decision",
            "validation_artifact_sha256",
        }:
            raise CampaignProposalBridgeError("admission callback receipt shape mismatch")
        if reviewed["decision"] not in {"admit", "reject"} or not _SHA_RE.fullmatch(
            str(reviewed["validation_artifact_sha256"])
        ):
            raise CampaignProposalBridgeError("admission callback receipt is invalid")
        return {
            "compiler_tasks_enqueued": 0,
            "decision": reviewed["decision"],
            "output_sha256": output_sha256,
            "request_id": request_id,
            "validation_artifact_sha256": str(reviewed["validation_artifact_sha256"]),
        }


def build_bridge_readiness_artifact(repo_root: Path, config_path: Path) -> dict[str, Any]:
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "sigma-campaign-llm-proposal-bridge-config-1.0":
        raise CampaignProposalBridgeError("unexpected bridge config schema")
    config = BridgeConfig.from_mapping(raw["bridge"])
    if config.execution_enabled:
        raise CampaignProposalBridgeError("checked-in campaign proposal bridge must be disabled")
    adapter_path = repo_root / "configs/llm_formula_proposal_adapter.json"
    if sha256_file(adapter_path) != config.adapter_config_sha256:
        raise CampaignProposalBridgeError("checked-in adapter config binding mismatch")
    eligibility = raw.get("data_eligibility", {})
    if set(eligibility.values()) != {False}:
        raise CampaignProposalBridgeError("checked-in data eligibility seals must all be closed")
    artifact: dict[str, Any] = {
        "adapter_config_sha256": config.adapter_config_sha256,
        "admission_callback_configured": config.admission_callback_id is not None,
        "campaign_task_type": TASK_TYPE,
        "compiler_tasks_enqueued": 0,
        "config_sha256": sha256_file(config_path),
        "data_eligibility": raw["data_eligibility"],
        "default_execution_enabled": False,
        "maximum_total_usd": "500.000000",
        "network_calls_made": 0,
        "paid_spend_usd": "0.000000",
        "prompt_resolver_binding_sha256": config.prompt_resolver_binding_sha256,
        "provider_callback_registered": False,
        "provider_binding_sha256": config.provider_binding_sha256,
        "prompt_resolver_callback_registered": False,
        "raw_body_persistence": False,
        "schema_version": "sigma-campaign-llm-proposal-bridge-readiness-1.0",
        "source_sha256": sha256_file(
            repo_root / "src/sigma_theory_compiler/campaign_llm_proposal_bridge.py"
        ),
        "status": "ready_disabled_quarantine_only",
    }
    artifact["content_sha256"] = canonical_sha256(artifact)
    return artifact
