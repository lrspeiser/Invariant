from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from sigma_theory_compiler.claude_creativity_api import (
    CLAUDE_OUTPUT_SCHEMA_VERSION,
    ClaudeAPIConfig,
    ClaudeCreativityClient,
    ClaudeCreativityError,
    ClaudeRole,
)
from sigma_theory_compiler.external_claude_transport import (
    CLIENT_USER_AGENT,
    ProviderCompatibleClaudeTransport,
)

MODEL = "claude-opus-4-6"


class ProviderFixture:
    def __init__(self, *, omit_last_action: bool = False) -> None:
        self.omit_last_action = omit_last_action
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        parsed = None if body is None else json.loads(body)
        self.requests.append(
            {
                "body": parsed,
                "headers": dict(headers),
                "method": method,
                "timeout": timeout,
                "url": url,
            }
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        assert parsed is not None
        prompt = json.loads(parsed["messages"][0]["content"])
        schema = parsed["output_config"]["format"]["schema"]
        actions = schema["properties"]["steering_actions"]
        candidate_ids = actions["items"]["properties"]["candidate_id"]["enum"]
        assert actions["type"] == "array"
        assert "exactly one steering action" in prompt["instruction"]
        if self.omit_last_action:
            candidate_ids = candidate_ids[:-1]
        output = {
            "benchmark_id": "external.sum-squares",
            "hypotheses": {},
            "role": "critic",
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": [
                {
                    "blocker_kind": "train_residual",
                    "candidate_id": candidate_id,
                    "distance_denominator": 1,
                    "distance_numerator": 1,
                    "repair": "Retain and branch through another representation.",
                    "verdict": "repair",
                }
                for candidate_id in candidate_ids
            ],
        }
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": "msg_provider_fixture",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }


class WideProposerFixture:
    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        hypotheses = [
            {
                "expression": f"n + {index}",
                "falsifiers": ["sealed holdout"],
                "family": "wide_search",
                "hypothesis_id": f"hypothesis.wide.{index}",
                "invariants": ["translation"],
                "known_analogues": ["affine sequence"],
                "llm_origin_assessment": "uncertain",
                "proof_plan": ["test boundary cases"],
                "rationale": "Retain a distinct bounded branch.",
                "representation": "sympy_expression",
                "source_idea_domains": ["algebra"],
                "synthesis_note": "A wide-output adapter control.",
            }
            for index in range(24)
        ]
        output = {
            "benchmark_id": "external.sum-squares",
            "hypotheses": hypotheses,
            "role": "proposer",
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": {},
        }
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": "msg_wide_provider_fixture",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 500},
        }


def _run(monkeypatch, provider: ProviderFixture):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    transport = ProviderCompatibleClaudeTransport(provider)
    client = ClaudeCreativityClient(
        ClaudeAPIConfig(model=MODEL, execution_enabled=True), transport
    )
    result = client.run(
        ClaudeRole.CRITIC,
        "external.sum-squares",
        {
            "benchmark_id": "external.sum-squares",
            "observations": [{"inputs": {"n": "1"}, "output": "1"}],
            "variables": ["n"],
        },
        candidate_summaries=[
            {"candidate_id": "candidate.one"},
            {"candidate_id": "candidate.two"},
        ],
    )
    return transport, result


def test_provider_adapter_compacts_critic_schema_and_preserves_exact_coverage(
    monkeypatch,
) -> None:
    provider = ProviderFixture()
    transport, result = _run(monkeypatch, provider)

    assert result.output is not None
    assert [item.candidate_id for item in result.output.steering_actions] == [
        "candidate.one",
        "candidate.two",
    ]
    assert all(request["headers"]["user-agent"] == CLIENT_USER_AGENT for request in provider.requests)
    evidence = transport.evidence_for("msg_provider_fixture")
    assert evidence["wire_contract_adapter_used"] is True
    assert evidence["provider_header_names"] == [
        "anthropic-version",
        "content-type",
        "user-agent",
        "x-api-key",
    ]
    assert all(len(evidence[key]) == 64 for key in (
        "provider_prompt_sha256",
        "provider_raw_output_sha256",
        "provider_request_schema_sha256",
    ))
    assert "fixture-secret" not in json.dumps(evidence, sort_keys=True)


def test_provider_adapter_rejects_missing_candidate_coverage(monkeypatch) -> None:
    with pytest.raises(ClaudeCreativityError, match="candidate coverage"):
        _run(monkeypatch, ProviderFixture(omit_last_action=True))


def test_provider_adapter_stages_every_proposer_branch_beyond_legacy_parser_limit(
    monkeypatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    transport = ProviderCompatibleClaudeTransport(WideProposerFixture())
    client = ClaudeCreativityClient(
        ClaudeAPIConfig(model=MODEL, execution_enabled=True), transport
    )
    result = client.run(
        ClaudeRole.PROPOSER,
        "external.sum-squares",
        {
            "benchmark_id": "external.sum-squares",
            "observations": [{"inputs": {"n": "1"}, "output": "1"}],
            "variables": ["n"],
        },
    )
    assert result.output is not None and len(result.output.hypotheses) == 16
    overflow = transport.hypothesis_overflow_for("msg_wide_provider_fixture")
    assert len(overflow) == 8
    assert [item["hypothesis_id"] for item in overflow] == [
        f"hypothesis.wide.{index}" for index in range(16, 24)
    ]
    evidence = transport.evidence_for("msg_wide_provider_fixture")
    assert evidence["hypothesis_overflow_adapter_used"] is True
    assert evidence["overflow_hypotheses_retained"] == 8
    assert evidence["wire_contract_adapter_used"] is False
