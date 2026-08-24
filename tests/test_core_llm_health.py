from __future__ import annotations

import copy
import json
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import core_llm_health as H
from sigma_theory_compiler.core_credential import CredentialActivationError
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"
SECRET = "test-anthropic-health-secret-never-persisted"


def proposer_output() -> dict[str, Any]:
    return {
        "benchmark_id": H.BENCHMARK_ID,
        "hypotheses": {
            "idea_1": {
                "expression": "sum(k,k=1..n)",
                "falsifiers": ["test the next visible index"],
                "family": "triangular_sequence",
                "hypothesis_id": "idea.health-control",
                "invariants": ["first differences increase by one"],
                "known_analogues": ["triangular numbers"],
                "llm_origin_assessment": "known_rewrite",
                "proof_plan": ["induct on n"],
                "rationale": "A bounded health-check proposal.",
                "representation": "finite_sum",
                "source_idea_domains": ["finite sums"],
                "synthesis_note": "Known control for transport health.",
            }
        },
        "role": "proposer",
        "schema_version": "invariant-claude-creativity-output-2.0",
        "steering_actions": {},
    }


class FakeTransport:
    def __init__(self) -> None:
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
        return 200, {
            "content": [{"text": json.dumps(proposer_output()), "type": "text"}],
            "id": "msg_health_test_0001",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 200},
        }


def run_fake_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[dict, FakeTransport]:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    credential = tmp_path / "explicit.env"
    credential.write_text(f"ANTHROPIC_API_KEY={SECRET}\n", encoding="utf-8")
    transport = FakeTransport()
    receipt = H.run_live_health(
        ROOT,
        credential_file=credential,
        transport=transport,
        retrieved_utc="2026-08-24T12:00:00+00:00",
    )
    return receipt, transport


@pytest.fixture(scope="module")
def fake_health(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[dict, FakeTransport]]:
    monkeypatch = pytest.MonkeyPatch()
    try:
        yield run_fake_health(tmp_path_factory.mktemp("health"), monkeypatch)
    finally:
        monkeypatch.undo()


def test_live_health_uses_core_context_and_never_persists_credential(
    fake_health: tuple[dict, FakeTransport],
) -> None:
    receipt, transport = fake_health

    assert "ANTHROPIC_API_KEY" not in os.environ
    assert [request["method"] for request in transport.requests] == ["GET", "POST"]
    assert all(request["headers"]["x-api-key"] == SECRET for request in transport.requests)
    prompt = json.loads(transport.requests[1]["body"]["messages"][0]["content"])
    context = prompt["creative_context"]
    assert context["content_sha256"] == receipt["call"]["creative_context_sha256"]
    assert prompt["benchmark"]["visible_values"] == [0, 1, 3, 6, 10]
    schema = transport.requests[1]["body"]["output_config"]["format"]["schema"]
    assert schema["properties"]["hypotheses"]["required"] == ["idea_1"]
    assert receipt["credential_activation"]["source_kind"] == "explicit_env_file"
    assert receipt["credential_activation"]["credential_value_recorded"] is False
    assert receipt["environment_boundary"]["credential_restored_to_pre_run_state"] is True
    assert receipt["claims"]["health_check_establishes_creativity_advantage"] is False
    assert SECRET not in json.dumps(receipt, sort_keys=True)
    H.validate_health_receipt(receipt, ROOT)


def test_health_fails_before_network_when_credential_activation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport()

    @contextmanager
    def missing_credential(**_: Any):
        raise CredentialActivationError("credential deliberately absent")
        yield

    monkeypatch.setattr(H, "activated_credential", missing_credential)
    with pytest.raises(H.CoreLLMHealthError, match="deliberately absent"):
        H.run_live_health(ROOT, transport=transport)
    assert transport.requests == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda value: value["claims"].__setitem__(
            "health_check_establishes_creativity_advantage", True
        ),
        lambda value: value["call"].__setitem__("creative_context_sha256", "0" * 64),
        lambda value: value["call"].__setitem__("api_response_id", "bad"),
        lambda value: value["source_bindings"]["health_source"].__setitem__("sha256", "0" * 64),
    ],
)
def test_resealed_claim_context_response_or_source_tamper_fails_closed(
    fake_health: tuple[dict, FakeTransport],
    mutate: Any,
) -> None:
    receipt, _ = fake_health
    tampered = copy.deepcopy(receipt)
    mutate(tampered)
    tampered["content_sha256"] = canonical_sha256(
        {key: item for key, item in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(H.CoreLLMHealthError):
        H.validate_health_receipt(tampered, ROOT)


def test_committed_live_health_receipt_validates_offline() -> None:
    path = ROOT / H.OUTPUT_PATH
    assert path.is_file()
    H.validate_health_receipt(json.loads(path.read_text(encoding="utf-8")), ROOT)


def test_health_rebind_preserves_live_call_and_repairs_only_source_bindings() -> None:
    path = ROOT / H.OUTPUT_PATH
    original = json.loads(path.read_text(encoding="utf-8"))
    stale = copy.deepcopy(original)
    stale["source_bindings"]["core_live_receipt"]["content_sha256"] = "0" * 64
    stale["content_sha256"] = canonical_sha256(
        {key: item for key, item in stale.items() if key != "content_sha256"}
    )

    H.validate_health_receipt(stale)
    with pytest.raises(H.CoreLLMHealthError, match="source bindings"):
        H.validate_health_receipt(stale, ROOT)
    rebound = H.rebind_health_receipt(ROOT, stale)

    assert rebound["call"] == original["call"]
    assert rebound["credential_activation"] == original["credential_activation"]
    assert rebound["retrieved_utc"] == original["retrieved_utc"]
    assert rebound["source_bindings"] == original["source_bindings"]
    H.validate_health_receipt(rebound, ROOT)
