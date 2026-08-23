from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import creativity_component_knockout_generation as G
from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION
from sigma_theory_compiler.durable_llm_attempt_journal import DurableAttemptJournal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"
EXPERIMENTS = (
    "creativity-knockout-minus-expanded-grammar-001",
    "creativity-knockout-minus-independent-proof-recombination-001",
    "creativity-knockout-minus-lineage-labels-001",
    "creativity-knockout-minus-non-pruning-001",
)


class KnockoutTransport:
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
        benchmark_id = parsed["output_config"]["format"]["schema"]["properties"][
            "benchmark_id"
        ]["const"]
        if prompt["role"] == "proposer":
            legacy = "legacy algebraic representation slice" in prompt["instruction"]
            output = self._proposal(benchmark_id, len(self.requests), legacy=legacy)
        else:
            output = self._critic(benchmark_id, prompt["candidate_summaries"])
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg.knockout.{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    @staticmethod
    def _proposal(
        benchmark_id: str, call_number: int, *, legacy: bool
    ) -> dict[str, Any]:
        representations = (
            ("sympy_expression",) * 3
            if legacy
            else ("linear_recurrence", "generating_function", "modular_relation")
        )
        origins = ("known_rewrite", "cross_domain_synthesis", "uncertain")
        return {
            "benchmark_id": benchmark_id,
            "hypotheses": [
                {
                    "expression": f"knockout_relation_{call_number}_{index}(n)",
                    "falsifiers": [f"separating index {index}"],
                    "family": f"knockout_family_{index}",
                    "hypothesis_id": f"hypothesis.{call_number}.{index}",
                    "invariants": [f"invariant {index}"],
                    "known_analogues": [f"analogue {index}"],
                    "llm_origin_assessment": origins[index],
                    "proof_plan": [f"derive route {index}", "test boundary cases"],
                    "rationale": "A bounded fake response for the knockout contract.",
                    "representation": representations[index],
                    "source_idea_domains": [f"domain {index}", f"bridge {index}"],
                    "synthesis_note": "Prior-art review remains separate.",
                }
                for index in range(3)
            ],
            "role": "proposer",
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": [],
        }

    @staticmethod
    def _critic(benchmark_id: str, summaries: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "benchmark_id": benchmark_id,
            "hypotheses": [],
            "role": "critic",
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": [
                {
                    "blocker_kind": f"typed_blocker_{index}",
                    "candidate_id": summary["candidate_id"],
                    "distance_denominator": 4,
                    "distance_numerator": index + 1,
                    "repair": "Retain or prune exactly as the registered arm requires.",
                    "verdict": ("reject", "repair", "retain")[index],
                }
                for index, summary in enumerate(summaries)
            ],
        }


def _authorization(
    experiment_id: str, *, journal_path: Path | None = None, approved: bool = True
) -> dict[str, Any]:
    if journal_path is None:
        journal_path = ROOT / "work" / "test-authorization" / "attempts.jsonl"
    value = G.authorization_template(ROOT, experiment_id, journal_path)
    value.update(
        {
            "authorization_nonce": "a" * 64,
            "authorized_at_utc": "2026-08-23T15:30:00Z",
            "authorized_by": "test-authority",
            "paid_execution_authorized": approved,
        }
    )
    return G.seal_authorization(value)


def _journal(
    tmp_path: Path, experiment_id: str, key: bytes = b"k" * 32
) -> DurableAttemptJournal:
    return DurableAttemptJournal.create(
        tmp_path / "work" / experiment_id / "attempts.jsonl",
        experiment_id=experiment_id,
        source_bindings=G._source_bindings(ROOT),
        unblinding_key=key,
    )


def _run(
    tmp_path: Path, experiment_id: str
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    DurableAttemptJournal,
    KnockoutTransport,
]:
    journal = _journal(tmp_path, experiment_id)
    transport = KnockoutTransport()
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-knockout-secret-never-persisted"
    try:
        review, public, coordinator = G.run_generation(
            ROOT,
            authorization=_authorization(experiment_id, journal_path=journal.path),
            journal=journal,
            transport=transport,
        )
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original
    return review, public, coordinator, journal, transport


def _outputs_for_arm(
    review: Mapping[str, Any], coordinator: Mapping[str, Any], arm: str
) -> list[dict[str, Any]]:
    ids = {
        item["blinded_output_id"] for item in coordinator["mapping"] if item["arm"] == arm
    }
    return [item for item in review["blinded_outputs"] if item["blinded_output_id"] in ids]


def test_authorization_is_sealed_scoped_and_required(tmp_path: Path) -> None:
    experiment_id = EXPERIMENTS[0]
    approved = _authorization(experiment_id)
    experiment = G.validate_authorization(approved, ROOT)
    assert experiment["experiment_id"] == experiment_id
    assert approved["maximum_provider_calls"] == 96
    assert approved["maximum_total_tokens"] == 400_000

    denied = _authorization(experiment_id, approved=False)
    with pytest.raises(G.ComponentKnockoutGenerationError, match="not authorized"):
        G.validate_authorization(denied, ROOT)

    tampered = dict(approved)
    tampered["maximum_provider_calls"] = 97
    with pytest.raises(G.ComponentKnockoutGenerationError, match="seal changed"):
        G.validate_authorization(tampered, ROOT)

    journal = _journal(tmp_path, experiment_id)
    denied = _authorization(experiment_id, journal_path=journal.path, approved=False)

    def forbidden_transport(*_args, **_kwargs):
        raise AssertionError("provider transport must not run without authorization")

    with pytest.raises(G.ComponentKnockoutGenerationError, match="journal path"):
        G.run_generation(
            ROOT,
            authorization=approved,
            journal=journal,
            transport=forbidden_transport,
        )

    with pytest.raises(G.ComponentKnockoutGenerationError, match="not authorized"):
        G.run_generation(
            ROOT,
            authorization=denied,
            journal=journal,
            transport=forbidden_transport,
        )


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_each_component_knockout_runs_once_per_slot_and_stays_blinded(
    tmp_path: Path, experiment_id: str
) -> None:
    review, public, coordinator, journal, transport = _run(tmp_path, experiment_id)
    posts = [item for item in transport.requests if item["method"] == "POST"]
    assert len(posts) == 96
    assert public["attempt_accounting"] == {
        "attempt_journal_content_sha256": journal.content_sha256,
        "balanced_provider_message_attempts": True,
        "contract_outcome_counts": {"contract_pass": 96},
        "provider_message_attempts": 96,
        "replacement_calls": 0,
        "scheduled_slots": 96,
        "transient_retries": 0,
    }
    assert public["release_gate"]["generation_eligible"] is True
    assert public["claude_runtime"]["authenticated_messages_api_working"] is True
    assert len(review["blinded_outputs"]) == 48
    assert '"arm"' not in json.dumps(review, sort_keys=True)
    assert "test-knockout-secret-never-persisted" not in journal.path.read_text(
        encoding="utf-8"
    )
    G.validate_public(review, public, ROOT)
    G.validate_coordinator(coordinator, review, public, journal)


@pytest.mark.parametrize("experiment_id", EXPERIMENTS)
def test_registered_intervention_changes_only_its_targeted_behavior(
    tmp_path: Path, experiment_id: str
) -> None:
    review, public, coordinator, _, _ = _run(tmp_path, experiment_id)
    knockout_arm = public["intervention"]["knockout_arm"]
    reference = _outputs_for_arm(review, coordinator, "full_creativity_first")
    knockout = _outputs_for_arm(review, coordinator, knockout_arm)
    assert len(reference) == len(knockout) == 24

    if knockout_arm == "minus_expanded_grammar":
        hypothesis_rows = [
            branch
            for output in knockout
            for branch in output["branches"]
            if branch["branch_kind"] == "hypothesis_proof_route"
        ]
        assert {row["representation"] for row in hypothesis_rows} == {"sympy_expression"}
        assert all(output["typed_usable_ideas"] == 3 for output in knockout)
    elif knockout_arm == "minus_independent_proof_recombination":
        assert {
            branch["proof_mechanism"]
            for output in knockout
            for branch in output["branches"]
        } == {"llm_declared_plan"}
        assert all(
            branch["branch_kind"] != "cross_idea_recombination"
            for output in knockout
            for branch in output["branches"]
        )
        assert any(
            branch["branch_kind"] == "cross_idea_recombination"
            for output in reference
            for branch in output["branches"]
        )
    elif knockout_arm == "minus_lineage_labels":
        assert all(
            branch["llm_origin_assessment"] == "uncertain"
            and branch["known_analogues"] == []
            and branch["source_domains"] == []
            for output in knockout
            for branch in output["branches"]
        )
        assert any(
            branch["llm_origin_assessment"] != "uncertain"
            for output in reference
            for branch in output["branches"]
        )
    else:
        assert knockout_arm == "minus_non_pruning"
        assert all(output["typed_usable_ideas"] == 2 for output in knockout)
        assert all(output["typed_usable_ideas"] == 3 for output in reference)


def test_out_of_slice_representation_is_counted_without_replacement(
    tmp_path: Path,
) -> None:
    experiment_id = EXPERIMENTS[0]

    class ViolatingTransport(KnockoutTransport):
        @staticmethod
        def _proposal(
            benchmark_id: str, call_number: int, *, legacy: bool
        ) -> dict[str, Any]:
            return KnockoutTransport._proposal(benchmark_id, call_number, legacy=False)

    journal = _journal(tmp_path, experiment_id)
    transport = ViolatingTransport()
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-knockout-secret-never-persisted"
    try:
        review, public, _ = G.run_generation(
            ROOT,
            authorization=_authorization(experiment_id, journal_path=journal.path),
            journal=journal,
            transport=transport,
        )
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original
    assert len([item for item in transport.requests if item["method"] == "POST"]) == 96
    assert public["attempt_accounting"]["contract_outcome_counts"] == {
        "contract_failure": 24,
        "contract_pass": 72,
    }
    assert public["release_gate"]["generation_eligible"] is False
    assert sum(output["typed_usable_ideas"] == 0 for output in review["blinded_outputs"]) == 24
    assert all(output["branches"] for output in review["blinded_outputs"])


def test_resume_after_durable_response_never_redispatches_the_slot(
    monkeypatch, tmp_path: Path
) -> None:
    experiment_id = EXPERIMENTS[1]
    journal = _journal(tmp_path, experiment_id, key=b"r" * 32)
    first = KnockoutTransport()
    original_contract = G.confirmatory._contract_errors
    calls = 0

    def interrupt(result, role, config, summaries):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt("simulated crash after durable response")
        return original_contract(result, role, config, summaries)

    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-knockout-secret-never-persisted"
    monkeypatch.setattr(G.confirmatory, "_contract_errors", interrupt)
    try:
        with pytest.raises(KeyboardInterrupt, match="simulated crash"):
            G.run_generation(
                ROOT,
                authorization=_authorization(experiment_id, journal_path=journal.path),
                journal=journal,
                transport=first,
            )
        monkeypatch.setattr(G.confirmatory, "_contract_errors", original_contract)
        resumed = KnockoutTransport()
        _, public, _ = G.run_generation(
            ROOT,
            authorization=_authorization(experiment_id, journal_path=journal.path),
            journal=journal,
            transport=resumed,
        )
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original

    first_posts = [item for item in first.requests if item["method"] == "POST"]
    resumed_posts = [item for item in resumed.requests if item["method"] == "POST"]
    assert len(first_posts) + len(resumed_posts) == 96
    assert public["attempt_accounting"]["provider_message_attempts"] == 96
    assert public["attempt_accounting"]["contract_outcome_counts"] == {
        "contract_pass": 95,
        "indeterminate_after_dispatch": 1,
    }
    assert public["release_gate"]["generation_eligible"] is False
