from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest

from sigma_theory_compiler import claude_creativity_api as C

MODEL = "claude-opus-4-6"


def config(*, enabled: bool = True, calls: int = 8, tokens: int = 64_000) -> C.ClaudeAPIConfig:
    return C.ClaudeAPIConfig(
        model=MODEL,
        execution_enabled=enabled,
        maximum_calls=calls,
        maximum_total_tokens=tokens,
    )


def proposer_output(benchmark_id: str) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "hypotheses": [
            {
                "expression": "n*(n+1)*(2*n+1)/6",
                "falsifiers": ["test sealed indices outside interpolation range"],
                "family": "generating_function",
                "hypothesis_id": "claude.sum-squares.1",
                "invariants": ["third finite difference is constant"],
                "known_analogues": ["Faulhaber polynomial for sums of powers"],
                "llm_origin_assessment": "known_rewrite",
                "proof_plan": ["derive recurrence", "induct on n"],
                "rationale": "The finite-difference signature suggests a cubic polynomial.",
                "representation": "sympy_expression",
                "source_idea_domains": ["finite differences", "generating functions"],
                "synthesis_note": "This is a familiar identity recovered through a different lens.",
            }
        ],
        "role": "proposer",
        "schema_version": C.CLAUDE_OUTPUT_SCHEMA_VERSION,
        "steering_actions": [],
    }


def critic_output(benchmark_id: str) -> dict[str, Any]:
    return {
        "benchmark_id": benchmark_id,
        "hypotheses": [],
        "role": "critic",
        "schema_version": C.CLAUDE_OUTPUT_SCHEMA_VERSION,
        "steering_actions": [
            {
                "blocker_kind": "holdout_counterexample",
                "candidate_id": "candidate.polynomial.1",
                "distance_denominator": 8,
                "distance_numerator": 1,
                "repair": "Switch from interpolation to a recurrence representation.",
                "verdict": "repair",
            }
        ],
    }


class FakeTransport:
    def __init__(
        self, outputs: list[dict[str, Any]], *, usage: tuple[int, int] = (101, 53)
    ) -> None:
        self.outputs = outputs
        self.usage = usage
        self.requests: list[dict[str, Any]] = []

    def __call__(
        self,
        method: str,
        url: str,
        headers: Mapping[str, str],
        body: bytes | None,
        timeout: float,
    ) -> tuple[int, Mapping[str, Any]]:
        self.requests.append(
            {
                "body": None if body is None else json.loads(body),
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
        output = self.outputs.pop(0)
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg_test_{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": self.usage[0], "output_tokens": self.usage[1]},
        }


def public_benchmark() -> dict[str, Any]:
    return {
        "benchmark_id": "external.sum-squares",
        "observations": [
            {"inputs": {"n": "1"}, "output": "1"},
            {"inputs": {"n": "2"}, "output": "5"},
        ],
        "variables": ["n"],
    }


def test_disabled_and_missing_credentials_fail_closed_without_network(monkeypatch) -> None:
    transport = FakeTransport([])
    disabled = C.ClaudeCreativityClient(config(enabled=False), transport)
    result = disabled.run(C.ClaudeRole.PROPOSER, "external.sum-squares", public_benchmark())
    assert result.status is C.ClaudeCallStatus.BLOCKED_DISABLED
    assert result.evidence["network_calls"] == 0
    assert transport.requests == []

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    missing = C.ClaudeCreativityClient(config(), transport)
    result = missing.run(C.ClaudeRole.PROPOSER, "external.sum-squares", public_benchmark())
    assert result.status is C.ClaudeCallStatus.BLOCKED_MISSING_CREDENTIAL
    assert result.evidence["credential_persisted"] is False
    assert transport.requests == []


def test_creative_roles_are_open_but_remain_structured_and_non_authoritative() -> None:
    assert {role.value for role in C.ClaudeRole} == {
        "analogue_scout",
        "critic",
        "dataset_explainer",
        "proof_strategist",
        "proposer",
        "recombiner",
        "representation_inventor",
    }
    for role in C.ClaudeRole:
        schema = C._structured_output_schema(role, "external.sum-squares")
        assert schema["properties"]["role"]["const"] == role.value
        assert "llm_origin_assessment" in schema["properties"]["hypotheses"]["items"][
            "required"
        ]

def test_live_contract_uses_model_capability_check_and_structured_messages(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-never-persisted")
    transport = FakeTransport(
        [proposer_output("external.sum-squares"), critic_output("external.sum-squares")]
    )
    client = C.ClaudeCreativityClient(config(), transport)

    proposal = client.run(C.ClaudeRole.PROPOSER, "external.sum-squares", public_benchmark())
    critique = client.run(
        C.ClaudeRole.CRITIC,
        "external.sum-squares",
        public_benchmark(),
        candidate_summaries=[
            {
                "candidate_id": "candidate.polynomial.1",
                "expression": "n**3",
                "train_loss": "1/8",
            }
        ],
    )

    assert proposal.status is C.ClaudeCallStatus.COMPLETED
    assert proposal.output is not None and proposal.output.hypotheses
    assert critique.output is not None and critique.output.steering_actions
    assert client.budget.to_dict() == {
        "calls": 2,
        "input_tokens": 202,
        "output_tokens": 106,
        "total_tokens": 308,
    }
    assert [request["method"] for request in transport.requests] == ["GET", "POST", "POST"]
    message_request = transport.requests[1]
    assert message_request["url"] == C.MESSAGES_ENDPOINT
    assert message_request["headers"]["anthropic-version"] == C.ANTHROPIC_VERSION
    assert message_request["headers"]["x-api-key"] == "test-secret-never-persisted"
    assert "output_config" in message_request["body"]
    assert "output_format" not in message_request["body"]
    assert message_request["body"]["output_config"]["format"]["type"] == "json_schema"
    serialized = json.dumps([proposal.to_dict(), critique.to_dict()], sort_keys=True)
    assert "test-secret-never-persisted" not in serialized
    assert proposal.evidence["credential_persisted"] is False
    assert proposal.evidence["model_evidence"]["structured_outputs_supported"] is True


def test_sealed_material_and_role_smuggling_are_rejected(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    client = C.ClaudeCreativityClient(config(), FakeTransport([]))
    tainted = public_benchmark() | {"nested": {"target": "n**3"}}
    with pytest.raises(C.ClaudeCreativityError, match="sealed target"):
        client.run(C.ClaudeRole.PROPOSER, "external.sum-squares", tainted)
    with pytest.raises(C.ClaudeCreativityError, match="post-generation"):
        client.run(
            C.ClaudeRole.PROPOSER,
            "external.sum-squares",
            public_benchmark(),
            candidate_summaries=[{"candidate_id": "leak"}],
        )
    with pytest.raises(C.ClaudeCreativityError, match="requires candidate summaries"):
        client.run(C.ClaudeRole.CRITIC, "external.sum-squares", public_benchmark())


def test_invalid_structured_output_and_token_overrun_fail_closed(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "secret")
    wrong_role = proposer_output("external.sum-squares") | {"role": "critic"}
    with pytest.raises(C.ClaudeCreativityError, match="crossed its requested role"):
        C.ClaudeCreativityClient(config(), FakeTransport([wrong_role])).run(
            C.ClaudeRole.PROPOSER, "external.sum-squares", public_benchmark()
        )

    with pytest.raises(C.ClaudeCreativityError, match="total token budget"):
        C.ClaudeCreativityClient(
            config(tokens=1_024),
            FakeTransport([proposer_output("external.sum-squares")], usage=(900, 200)),
        ).run(C.ClaudeRole.PROPOSER, "external.sum-squares", public_benchmark())


def test_config_and_output_shapes_are_strict() -> None:
    with pytest.raises(C.ClaudeCreativityError, match="keys changed"):
        C.ClaudeAPIConfig.from_mapping(config().to_dict() | {"api_key": "forbidden"})
    invalid = proposer_output("external.sum-squares")
    invalid["hypotheses"][0]["expression"] = "x" * 513
    output = C.ClaudeStructuredOutput.from_mapping(invalid)
    assert output.hypotheses == ()
    assert output.rejected_hypotheses == 1
