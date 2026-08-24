"""Run one authenticated, context-bound Claude call through the core application path.

The health check is deliberately cheaper than the eight-call discovery campaign. It proves that
the current core can discover a credential, authenticate the configured model, inject the current
first-principles context, parse a structured creative response, and remove an injected credential.
It does not score creativity or establish a formula, proof, correctness, or novelty claim.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .claude_creativity_api import (
    ANTHROPIC_VERSION,
    ClaudeAPIConfig,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeRole,
    Transport,
    urllib_transport,
)
from .core_creative_discovery import (
    CLAUDE_API_SOURCE_PATH,
    PROMPT_CONTEXT_SOURCE_PATH,
    _load_bound_receipts,
    _load_config,
    _normalized_file_sha256,
)
from .core_creative_discovery import (
    CONFIG_PATH as CORE_CONFIG_PATH,
)
from .core_creative_discovery import (
    OUTPUT_PATH as CORE_RECEIPT_PATH,
)
from .core_creative_discovery import (
    SOURCE_PATH as CORE_SOURCE_PATH,
)
from .core_creative_discovery import (
    validate_receipt as validate_core_receipt,
)
from .core_creative_prompt_context import (
    FirstPrinciplesContextTransport,
    build_creative_prompt_context,
)
from .core_credential import CredentialActivationError, activated_credential
from .sigma_core import canonical_sha256

OUTPUT_PATH = "runs/math/core-creative-discovery/live-llm-health.json"
SOURCE_PATH = "src/sigma_theory_compiler/core_llm_health.py"
SCHEMA_VERSION = "invariant-core-live-llm-health-1.0"
BENCHMARK_ID = "core-health.current-main"
MAXIMUM_CALLS = 1
MAXIMUM_TOTAL_TOKENS = 8_192
MAXIMUM_OUTPUT_TOKENS = 2_048
TIMEOUT_SECONDS = 90
EFFORT = "high"

_HEX_64 = re.compile(r"[0-9a-f]{64}\Z")
_RESPONSE_ID = re.compile(r"[A-Za-z0-9_-]{10,180}\Z")
_ORIGINS = {
    "known_rewrite",
    "cross_domain_synthesis",
    "proposed_new_construction",
    "uncertain",
}


class CoreLLMHealthError(ValueError):
    """The live core LLM health path failed closed."""


def _strict(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise CoreLLMHealthError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or _HEX_64.fullmatch(value) is None:
        raise CoreLLMHealthError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _utc(value: str | None) -> str:
    if value is None:
        return datetime.now(UTC).isoformat().replace("+00:00", "Z")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise CoreLLMHealthError("health retrieval time is not ISO-8601") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CoreLLMHealthError("health retrieval time lacks an offset")
    return parsed.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise CoreLLMHealthError(f"{label} is not readable JSON") from error
    if not isinstance(value, dict):
        raise CoreLLMHealthError(f"{label} is not a JSON object")
    return value


def _load_core_state(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = _load_config(root)
    receipts = _load_bound_receipts(root, config)
    context = build_creative_prompt_context(
        receipts[6],
        receipts[11],
        receipts[12],
        receipts[13],
        receipts[2],
        receipts[7],
    )
    core_path = root / CORE_RECEIPT_PATH
    core_receipt = _read_json(core_path, "core live receipt")
    validate_core_receipt(core_receipt, root)
    if (
        core_receipt.get("llm_prompt_context", {}).get("content_sha256")
        != context["content_sha256"]
    ):
        raise CoreLLMHealthError("core receipt and current prompt context diverged")
    return config, context, core_receipt


def _source_bindings(
    root: Path,
    context: Mapping[str, Any],
    core_receipt: Mapping[str, Any],
) -> dict[str, Any]:
    core_receipt_path = root / CORE_RECEIPT_PATH
    return {
        "claude_api_source": {
            "path": CLAUDE_API_SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / CLAUDE_API_SOURCE_PATH),
        },
        "core_config": {
            "path": CORE_CONFIG_PATH,
            "sha256": _normalized_file_sha256(root / CORE_CONFIG_PATH),
        },
        "core_live_receipt": {
            "content_sha256": core_receipt["content_sha256"],
            "file_sha256": _normalized_file_sha256(core_receipt_path),
            "path": CORE_RECEIPT_PATH,
        },
        "core_source": {
            "path": CORE_SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / CORE_SOURCE_PATH),
        },
        "health_source": {
            "path": SOURCE_PATH,
            "sha256": _normalized_file_sha256(root / SOURCE_PATH),
        },
        "prompt_context": {
            "content_sha256": context["content_sha256"],
            "source_path": PROMPT_CONTEXT_SOURCE_PATH,
            "source_sha256": _normalized_file_sha256(root / PROMPT_CONTEXT_SOURCE_PATH),
        },
    }


def _call_evidence(
    result: Any,
    transport: FirstPrinciplesContextTransport,
) -> dict[str, Any]:
    if result.status is not ClaudeCallStatus.COMPLETED or result.output is None:
        raise CoreLLMHealthError("core health call did not complete")
    if (
        len(result.output.hypotheses) != 1
        or result.output.rejected_hypotheses != 0
        or result.output.steering_actions
        or result.output.rejected_steering_actions != 0
    ):
        raise CoreLLMHealthError("core health structured output was not exactly one valid idea")
    client = dict(result.evidence)
    response_id = client.get("api_response_id")
    if not isinstance(response_id, str) or _RESPONSE_ID.fullmatch(response_id) is None:
        raise CoreLLMHealthError("core health response ID changed")
    provider = dict(transport.evidence_for(response_id))
    hypothesis = result.output.hypotheses[0]
    model_evidence = client.get("model_evidence", {})
    if not isinstance(model_evidence, Mapping):
        raise CoreLLMHealthError("core health model evidence changed")
    return {
        "accepted_hypotheses": 1,
        "anthropic_version": client.get("anthropic_version"),
        "api_response_id": response_id,
        "client_output_sha256": client.get("output_sha256"),
        "client_prompt_sha256": client.get("prompt_sha256"),
        "client_raw_output_sha256": client.get("raw_output_sha256"),
        "client_request_schema_sha256": client.get("request_schema_sha256"),
        "creative_context_injected": provider.get("creative_context_injected"),
        "creative_context_sha256": provider.get("creative_context_sha256"),
        "credential_persisted": client.get("credential_persisted"),
        "header_names": client.get("header_names"),
        "model": client.get("model"),
        "model_capabilities_sha256": model_evidence.get("capabilities_sha256"),
        "network_calls": client.get("network_calls"),
        "origin_assessment": hypothesis.llm_origin_assessment,
        "provider_header_names": provider.get("provider_header_names"),
        "provider_prompt_sha256": provider.get("provider_prompt_sha256"),
        "provider_raw_output_sha256": provider.get("provider_raw_output_sha256"),
        "provider_request_schema_sha256": provider.get("provider_request_schema_sha256"),
        "quarantined_hypotheses": result.output.rejected_hypotheses,
        "representation": hypothesis.representation,
        "role": result.role.value,
        "status": result.status.value,
        "structured_outputs_supported": model_evidence.get("structured_outputs_supported"),
        "usage": dict(client.get("usage", {})),
        "wire_contract_adapter_used": provider.get("wire_contract_adapter_used"),
    }


def run_live_health(
    root: Path,
    *,
    credential_file: Path | None = None,
    transport: Transport = urllib_transport,
    retrieved_utc: str | None = None,
) -> dict[str, Any]:
    """Execute one bounded provider call and return credential-free evidence."""

    root = root.resolve()
    config, context, core_receipt = _load_core_state(root)
    env_var = config["claude"]["credential_env_var"]
    source_environment = dict(os.environ)
    if credential_file is not None:
        source_environment["INVARIANT_ENV_FILE"] = str(credential_file.resolve())
    credential_present_before = bool(os.environ.get(env_var, "").strip())
    wrapped_transport = FirstPrinciplesContextTransport(context, transport)
    client = ClaudeCreativityClient(
        ClaudeAPIConfig(
            model=config["claude"]["required_model"],
            credential_env_var=env_var,
            execution_enabled=True,
            maximum_calls=MAXIMUM_CALLS,
            maximum_total_tokens=MAXIMUM_TOTAL_TOKENS,
            maximum_output_tokens=MAXIMUM_OUTPUT_TOKENS,
            timeout_seconds=TIMEOUT_SECONDS,
            effort=EFFORT,
        ),
        wrapped_transport,
    )
    public_payload = {
        "health_check": "current core authenticated structured creativity path",
        "instructions": (
            "Propose exactly one typed explanation of the visible integer pattern. Self-label "
            "its origin and name a falsifier. This checks transport, not mathematical novelty."
        ),
        "visible_indices": [0, 1, 2, 3, 4],
        "visible_values": [0, 1, 3, 6, 10],
    }
    try:
        with activated_credential(
            project_root=root,
            env_var=env_var,
            environment=source_environment,
        ) as activation:
            result = client.run(
                ClaudeRole.PROPOSER,
                BENCHMARK_ID,
                public_payload,
                hypothesis_slots=1,
            )
    except (CredentialActivationError, ClaudeCreativityError) as error:
        raise CoreLLMHealthError(str(error)) from error
    credential_present_after = bool(os.environ.get(env_var, "").strip())
    call = _call_evidence(result, wrapped_transport)
    body: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "retrieved_utc": _utc(retrieved_utc),
        "source_bindings": _source_bindings(root, context, core_receipt),
        "credential_activation": activation.to_evidence(),
        "environment_boundary": {
            "credential_present_after_run": credential_present_after,
            "credential_present_before_run": credential_present_before,
            "credential_restored_to_pre_run_state": (
                credential_present_after == credential_present_before
            ),
            "default_machine_credential_discovered": (
                activation.source_kind == "user_invariant_env_file"
                and activation.injected_into_process
                and not credential_present_before
            ),
        },
        "execution_policy": {
            "effort": EFFORT,
            "maximum_calls": MAXIMUM_CALLS,
            "maximum_output_tokens": MAXIMUM_OUTPUT_TOKENS,
            "maximum_total_tokens": MAXIMUM_TOTAL_TOKENS,
            "model": config["claude"]["required_model"],
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "call": call,
        "release_gate": {
            "health_check_passed": True,
            "serious_claim_released": False,
            "status": "PASS_LIVE_CORE_LLM_HEALTH_ONLY",
        },
        "claims": {
            "credential_authenticated_provider_request_completed": True,
            "credential_material_persisted": False,
            "health_check_establishes_creativity_advantage": False,
            "health_check_establishes_formula_correctness": False,
            "health_check_establishes_literature_novelty": False,
            "health_check_solves_open_problem": False,
            "llm_is_verifier_authority": False,
        },
    }
    body["content_sha256"] = canonical_sha256(body)
    validate_health_receipt(body, root)
    return body


def validate_health_receipt(value: Mapping[str, Any], root: Path | None = None) -> None:
    _strict(
        value,
        {
            "call",
            "claims",
            "content_sha256",
            "credential_activation",
            "environment_boundary",
            "execution_policy",
            "release_gate",
            "retrieved_utc",
            "schema_version",
            "source_bindings",
        },
        "core LLM health receipt",
    )
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    if value.get("schema_version") != SCHEMA_VERSION or value.get(
        "content_sha256"
    ) != canonical_sha256(body):
        raise CoreLLMHealthError("core LLM health content seal changed")
    if _utc(value.get("retrieved_utc")) != value.get("retrieved_utc"):
        raise CoreLLMHealthError("core LLM health time is not normalized UTC")
    expected_claims = {
        "credential_authenticated_provider_request_completed": True,
        "credential_material_persisted": False,
        "health_check_establishes_creativity_advantage": False,
        "health_check_establishes_formula_correctness": False,
        "health_check_establishes_literature_novelty": False,
        "health_check_solves_open_problem": False,
        "llm_is_verifier_authority": False,
    }
    expected_release = {
        "health_check_passed": True,
        "serious_claim_released": False,
        "status": "PASS_LIVE_CORE_LLM_HEALTH_ONLY",
    }
    if value.get("claims") != expected_claims or value.get("release_gate") != expected_release:
        raise CoreLLMHealthError("core LLM health opened a scientific release gate")
    credential = value.get("credential_activation", {})
    _strict(
        credential,
        {
            "credential_env_var",
            "credential_persisted",
            "credential_value_recorded",
            "injected_into_process",
            "source_kind",
            "source_locator_sha256",
        },
        "health credential activation",
    )
    if (
        credential["credential_env_var"] != "ANTHROPIC_API_KEY"
        or credential["credential_persisted"] is not False
        or credential["credential_value_recorded"] is not False
        or not isinstance(credential["injected_into_process"], bool)
        or credential["source_kind"]
        not in {
            "explicit_env_file",
            "process_environment",
            "project_env_file",
            "user_invariant_env_file",
        }
    ):
        raise CoreLLMHealthError("core LLM health credential boundary changed")
    _sha(credential["source_locator_sha256"], "credential source locator hash")
    environment = value.get("environment_boundary", {})
    _strict(
        environment,
        {
            "credential_present_after_run",
            "credential_present_before_run",
            "credential_restored_to_pre_run_state",
            "default_machine_credential_discovered",
        },
        "health environment boundary",
    )
    if environment["credential_restored_to_pre_run_state"] is not True:
        raise CoreLLMHealthError("core LLM health did not restore the credential environment")
    if any(
        not isinstance(environment[key], bool)
        for key in (
            "credential_present_after_run",
            "credential_present_before_run",
            "credential_restored_to_pre_run_state",
            "default_machine_credential_discovered",
        )
    ):
        raise CoreLLMHealthError("core LLM health environment evidence changed")
    expected_default = (
        credential["source_kind"] == "user_invariant_env_file"
        and credential["injected_into_process"] is True
        and environment["credential_present_before_run"] is False
    )
    if environment["default_machine_credential_discovered"] is not expected_default:
        raise CoreLLMHealthError("core LLM default credential evidence changed")
    policy = value.get("execution_policy", {})
    if policy != {
        "effort": EFFORT,
        "maximum_calls": MAXIMUM_CALLS,
        "maximum_output_tokens": MAXIMUM_OUTPUT_TOKENS,
        "maximum_total_tokens": MAXIMUM_TOTAL_TOKENS,
        "model": "claude-opus-4-6",
        "timeout_seconds": TIMEOUT_SECONDS,
    }:
        raise CoreLLMHealthError("core LLM health execution policy changed")
    call = value.get("call", {})
    _strict(
        call,
        {
            "accepted_hypotheses",
            "anthropic_version",
            "api_response_id",
            "client_output_sha256",
            "client_prompt_sha256",
            "client_raw_output_sha256",
            "client_request_schema_sha256",
            "creative_context_injected",
            "creative_context_sha256",
            "credential_persisted",
            "header_names",
            "model",
            "model_capabilities_sha256",
            "network_calls",
            "origin_assessment",
            "provider_header_names",
            "provider_prompt_sha256",
            "provider_raw_output_sha256",
            "provider_request_schema_sha256",
            "quarantined_hypotheses",
            "representation",
            "role",
            "status",
            "structured_outputs_supported",
            "usage",
            "wire_contract_adapter_used",
        },
        "core LLM health call",
    )
    if (
        not isinstance(call["api_response_id"], str)
        or _RESPONSE_ID.fullmatch(call["api_response_id"]) is None
        or call["accepted_hypotheses"] != 1
        or call["quarantined_hypotheses"] != 0
        or call["anthropic_version"] != ANTHROPIC_VERSION
        or call["creative_context_injected"] is not True
        or call["credential_persisted"] is not False
        or call["model"] != "claude-opus-4-6"
        or call["network_calls"] != 2
        or call["origin_assessment"] not in _ORIGINS
        or call["role"] != "proposer"
        or call["status"] != "completed"
        or call["structured_outputs_supported"] is not True
        or call["wire_contract_adapter_used"] is not False
        or not isinstance(call["header_names"], list)
        or not isinstance(call["provider_header_names"], list)
        or "x-api-key" not in call["header_names"]
        or "x-api-key" not in call["provider_header_names"]
    ):
        raise CoreLLMHealthError("core LLM health call evidence changed")
    for key in (
        "client_output_sha256",
        "client_prompt_sha256",
        "client_raw_output_sha256",
        "client_request_schema_sha256",
        "creative_context_sha256",
        "model_capabilities_sha256",
        "provider_prompt_sha256",
        "provider_raw_output_sha256",
        "provider_request_schema_sha256",
    ):
        _sha(call[key], f"health call {key}")
    usage = call["usage"]
    _strict(usage, {"input_tokens", "output_tokens"}, "health call usage")
    if (
        any(isinstance(usage[key], bool) or not isinstance(usage[key], int) for key in usage)
        or min(usage.values()) < 0
        or sum(usage.values()) <= 0
        or sum(usage.values()) > MAXIMUM_TOTAL_TOKENS
    ):
        raise CoreLLMHealthError("core LLM health usage changed")
    if root is None:
        return
    root = root.resolve()
    _, context, core_receipt = _load_core_state(root)
    bindings = value.get("source_bindings", {})
    _strict(
        bindings,
        {
            "claude_api_source",
            "core_config",
            "core_live_receipt",
            "core_source",
            "health_source",
            "prompt_context",
        },
        "core LLM health source bindings",
    )
    if value.get("source_bindings") != _source_bindings(root, context, core_receipt):
        raise CoreLLMHealthError("core LLM health source bindings changed")
    if call["creative_context_sha256"] != context["content_sha256"]:
        raise CoreLLMHealthError("core LLM health prompt context binding changed")


__all__ = [
    "OUTPUT_PATH",
    "CoreLLMHealthError",
    "run_live_health",
    "validate_health_receipt",
]
