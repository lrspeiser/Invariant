from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import creativity_confirmatory_generation as C
from sigma_theory_compiler import creativity_confirmatory_recovery as R
from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION
from sigma_theory_compiler.durable_llm_attempt_journal import DurableAttemptJournal

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"


class ConfirmatoryTransport:
    def __init__(self, *, incomplete_critic_once: bool = False) -> None:
        self.requests: list[dict[str, Any]] = []
        self.incomplete_critic_once = incomplete_critic_once
        self._fault_used = False

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
        role = prompt["role"]
        if role == "proposer":
            output = self._proposal(benchmark_id, len(self.requests))
        else:
            summaries = prompt["candidate_summaries"]
            if self.incomplete_critic_once and not self._fault_used:
                summaries = summaries[:-1]
                self._fault_used = True
            output = self._critic(benchmark_id, summaries)
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg.confirmatory.{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }

    @staticmethod
    def _proposal(benchmark_id: str, call_number: int) -> dict[str, Any]:
        representations = ("linear_recurrence", "generating_function", "modular_relation")
        origins = ("known_rewrite", "cross_domain_synthesis", "uncertain")
        hypotheses = []
        for index in range(3):
            hypotheses.append(
                {
                    "expression": f"confirmatory_relation_{call_number}_{index}(n)",
                    "falsifiers": [f"separating index {index}"],
                    "family": f"confirmatory_family_{index}",
                    "hypothesis_id": f"hypothesis.{call_number}.{index}",
                    "invariants": [f"invariant {index}"],
                    "known_analogues": [f"analogue {index}"],
                    "llm_origin_assessment": origins[index],
                    "proof_plan": [f"derive route {index}", "test boundary cases"],
                    "rationale": "A bounded fake response for the no-replacement contract.",
                    "representation": representations[index],
                    "source_idea_domains": [f"domain {index}", f"bridge {index}"],
                    "synthesis_note": "Prior-art review remains separate.",
                }
            )
        return {
            "benchmark_id": benchmark_id,
            "hypotheses": hypotheses,
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
                    "repair": "Retain the lineage and change representation.",
                    "verdict": ("reject", "repair", "retain")[index],
                }
                for index, summary in enumerate(summaries)
            ],
        }


def _new_journal(tmp_path: Path, key: bytes = b"k" * 32) -> DurableAttemptJournal:
    return DurableAttemptJournal.create(
        tmp_path / "work" / "confirmatory-attempts.jsonl",
        experiment_id=C.load_config(ROOT)["experiment_id"],
        source_bindings=C._source_bindings(ROOT),
        unblinding_key=key,
    )


def _run_with_environment(
    journal: DurableAttemptJournal, transport: ConfirmatoryTransport
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-secret-never-persisted"
    try:
        return C.run_generation(ROOT, journal=journal, transport=transport)
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_clean_confirmatory_run_has_exactly_one_dispatch_per_slot(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path)
    transport = ConfirmatoryTransport()
    review, public, coordinator = _run_with_environment(journal, transport)

    post_requests = [item for item in transport.requests if item["method"] == "POST"]
    assert len(post_requests) == 96
    assert public["attempt_accounting"] == {
        "attempt_journal_content_sha256": journal.content_sha256,
        "balanced_provider_message_attempts": True,
        "contract_outcome_counts": {"contract_pass": 96},
        "provider_message_attempts": 96,
        "replacement_calls": 0,
        "scheduled_slots": 96,
        "transient_retries": 0,
    }
    assert public["release_gate"]["confirmatory_generation_eligible"] is True
    assert public["claude_runtime"]["authenticated_messages_api_working"] is True
    assert len(review["blinded_outputs"]) == 48
    assert '"arm"' not in json.dumps(review, sort_keys=True)
    assert "test-secret-never-persisted" not in journal.path.read_text(encoding="utf-8")
    C.validate_public(review, public, ROOT)
    C.validate_coordinator(coordinator, review, public, journal)


def test_historical_source_compatibility_is_exact_and_fail_closed() -> None:
    review = json.loads(
        (ROOT / "runs/math/creativity-confirmatory/paired-review-packet.json").read_text(
            encoding="utf-8"
        )
    )
    public = json.loads(
        (ROOT / "runs/math/creativity-confirmatory/paired-generation-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    C.validate_public(review, public, ROOT)

    tampered = json.loads(json.dumps(public))
    tampered["source_bindings"]["claude_adapter"]["sha256"] = "0" * 64
    R._reseal(tampered)
    with pytest.raises(C.ConfirmatoryGenerationError, match="source bindings changed"):
        C.validate_public(review, tampered, ROOT)


def test_contract_failure_is_retained_and_never_replaced(tmp_path: Path) -> None:
    journal = _new_journal(tmp_path)
    transport = ConfirmatoryTransport(incomplete_critic_once=True)
    review, public, _ = _run_with_environment(journal, transport)

    assert len([item for item in transport.requests if item["method"] == "POST"]) == 96
    assert public["attempt_accounting"]["provider_message_attempts"] == 96
    assert public["attempt_accounting"]["contract_outcome_counts"] == {
        "contract_failure": 1,
        "contract_pass": 95,
    }
    assert public["release_gate"]["confirmatory_generation_eligible"] is False
    assert any(
        branch["generation_contract_status"] == "failed"
        for output in review["blinded_outputs"]
        for branch in output["branches"]
    )
    assert sum(
        item["event_kind"] == "message_dispatch" for item in journal.events
    ) == 96


def test_zero_retained_ideas_get_a_scored_placeholder_without_new_calls(
    tmp_path: Path,
) -> None:
    journal = _new_journal(tmp_path)
    transport = ConfirmatoryTransport()
    review, public, coordinator = _run_with_environment(journal, transport)
    target = review["blinded_outputs"][0]
    target["branches"] = []
    target["typed_usable_ideas"] = 0
    R._reseal(review)
    public["blinding"]["review_packet_content_sha256"] = review["content_sha256"]
    R._reseal(public)
    coordinator["review_packet_content_sha256"] = review["content_sha256"]
    coordinator["public_receipt_content_sha256"] = public["content_sha256"]
    R._reseal(coordinator)
    event_count = len(journal.events)

    recovered_review, recovered_public, recovered_coordinator = (
        R.materialize_rejection_placeholders(
            ROOT, review, public, coordinator, journal
        )
    )

    placeholders = [
        (output, branch)
        for output in recovered_review["blinded_outputs"]
        for branch in output["branches"]
        if branch["branch_kind"] == "all_proposals_rejected_outcome"
    ]
    assert len(placeholders) == 1
    assert placeholders[0][0]["typed_usable_ideas"] == 0
    assert recovered_public["post_generation_deviation"]["empty_output_count"] == 1
    assert recovered_public["post_generation_deviation"]["llm_calls_repeated"] is False
    assert len(journal.events) == event_count
    C.validate_public(recovered_review, recovered_public, ROOT)
    C.validate_coordinator(
        recovered_coordinator, recovered_review, recovered_public, journal
    )
    R.validate_recovery(
        ROOT,
        recovered_review,
        recovered_public,
        recovered_coordinator,
        journal,
    )
    R.validate_recovery_public(ROOT, recovered_review, recovered_public)


def test_resume_after_response_does_not_redispatch_indeterminate_slot(
    monkeypatch, tmp_path: Path
) -> None:
    journal = _new_journal(tmp_path, b"r" * 32)
    first = ConfirmatoryTransport()
    original_contract = C._contract_errors
    calls = 0

    def interrupt(result, role, config, summaries):
        nonlocal calls
        calls += 1
        if calls == 5:
            raise KeyboardInterrupt("simulated crash after durable response")
        return original_contract(result, role, config, summaries)

    monkeypatch.setattr(C, "_contract_errors", interrupt)
    with pytest.raises(KeyboardInterrupt, match="simulated crash"):
        _run_with_environment(journal, first)
    assert len([item for item in first.requests if item["method"] == "POST"]) == 5

    monkeypatch.setattr(C, "_contract_errors", original_contract)
    resumed = ConfirmatoryTransport()
    review, public, _ = _run_with_environment(journal, resumed)
    assert len([item for item in resumed.requests if item["method"] == "POST"]) == 91
    assert public["attempt_accounting"]["provider_message_attempts"] == 96
    assert public["attempt_accounting"]["contract_outcome_counts"] == {
        "contract_pass": 95,
        "indeterminate_after_dispatch": 1,
    }
    assert public["release_gate"]["confirmatory_generation_eligible"] is False
    assert len(review["blinded_outputs"]) == 48


def test_budget_blocked_slots_are_counted_without_provider_replacement(
    monkeypatch, tmp_path: Path
) -> None:
    journal = _new_journal(tmp_path)
    original_load = C.load_config

    def tiny_budget(root: Path) -> dict[str, Any]:
        config = original_load(root)
        config["claude"] = dict(config["claude"])
        config["claude"]["maximum_total_tokens_per_arm"] = 1
        config["matched_resource_budget"] = dict(config["matched_resource_budget"])
        config["matched_resource_budget"]["tokens_per_arm"] = 1
        return config

    monkeypatch.setattr(C, "load_config", tiny_budget)
    transport = ConfirmatoryTransport()
    review, public, _ = _run_with_environment(journal, transport)
    assert len([item for item in transport.requests if item["method"] == "POST"]) == 2
    assert public["attempt_accounting"]["contract_outcome_counts"] == {
        "budget_blocked": 94,
        "contract_pass": 2,
    }
    assert public["release_gate"]["confirmatory_generation_eligible"] is False
    assert len(review["blinded_outputs"]) == 48
