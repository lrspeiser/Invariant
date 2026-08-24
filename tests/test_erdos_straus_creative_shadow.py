from __future__ import annotations

import json
import os
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION
from sigma_theory_compiler.durable_llm_attempt_journal import DurableAttemptJournal
from sigma_theory_compiler.erdos_straus_creative_shadow import (
    _creative_calls,
    _mutated_pairs,
    _run_pairs,
    _witness_sample,
    parse_recipe,
    validate_receipt,
)
from sigma_theory_compiler.exponent_diophantine_sweeper import _es_hard_members

EXPERIMENT = {
    "maximum_moduli_per_recipe": 4,
    "maximum_offset": 255,
    "maximum_offsets_per_axis": 6,
    "mutation_radius": 2,
}
ROOT = Path(__file__).resolve().parents[1]
RECEIPT_PATH = ROOT / "runs" / "math" / "erdos-straus-creative-shadow" / "live-runtime.json"
CONFIG_PATH = ROOT / "configs" / "erdos_straus_creative_shadow.json"
MODEL = "claude-opus-4-6"


class SimulatedCrash(RuntimeError):
    pass


class _ExplodingResponse(dict):
    def get(self, key, default=None):
        if key == "type":
            raise SimulatedCrash("process stopped after the durable response append")
        return super().get(key, default)


class ShadowTransport:
    def __init__(self, *, crash_after_first_response: bool = False, invalid_last_critic: bool = False):
        self.requests: list[dict[str, Any]] = []
        self.crash_after_first_response = crash_after_first_response
        self.invalid_last_critic = invalid_last_critic
        self.critic_calls = 0

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
        prompt = json.loads(parsed["messages"][0]["content"])
        benchmark_id = parsed["output_config"]["format"]["schema"]["properties"][
            "benchmark_id"
        ]["const"]
        role = prompt["role"]
        if role == "critic":
            self.critic_calls += 1
            if self.invalid_last_critic and self.critic_calls == 2:
                output = {}
            else:
                output = {
                    "benchmark_id": benchmark_id,
                    "hypotheses": {},
                    "role": role,
                    "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                    "steering_actions": {
                        item["candidate_id"]: {
                            "blocker_kind": "bounded_test_blocker",
                            "distance_denominator": 2,
                            "distance_numerator": 1,
                            "repair": "Retain and recombine with a different offset scale.",
                            "verdict": "repair",
                        }
                        for item in prompt["candidate_summaries"]
                    },
                }
        else:
            requested = 2 if self.invalid_last_critic else 1
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": [self._hypothesis(index) for index in range(requested)],
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": {},
            }
        response: Mapping[str, Any] = {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg.shadow.{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }
        if self.crash_after_first_response:
            self.crash_after_first_response = False
            response = _ExplodingResponse(response)
        return 200, response

    @staticmethod
    def _hypothesis(index: int) -> dict[str, Any]:
        return {
            "expression": f"ESDSL1|basis=lattice_transform|x=0,{index + 1}|t=0,{index + 2}|m=24",
            "falsifiers": ["Fails an exact modular lane test."],
            "family": f"shadow_family_{index}",
            "hypothesis_id": f"shadow.{index}",
            "invariants": ["Exact integer divisibility only."],
            "known_analogues": ["lattice parameterization"],
            "llm_origin_assessment": "cross_domain_synthesis",
            "proof_plan": ["Derive the divisor identity independently."],
            "rationale": "A bounded fake response for durable recovery testing.",
            "representation": "modular_relation",
            "source_idea_domains": ["lattices", "Egyptian fractions"],
            "synthesis_note": "Novelty remains unestablished pending prior-art review.",
        }


class NoCallTransport:
    def __init__(self):
        self.calls = 0

    def __call__(self, *_args):
        self.calls += 1
        raise AssertionError("a completed scheduled slot was redispatched")


def _test_config(*, critics: bool) -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config = deepcopy(config)
    config["campaign_id"] = f"erdos-straus-journal-test-{int(critics)}"
    config["experiment"]["creative_roles"] = ["proposer"]
    config["experiment"]["requested_ideas_per_creative_call"] = 2 if critics else 1
    config["experiment"]["llm_critic_batch_size"] = 1 if critics else 0
    return config


def _journal(tmp_path: Path, config: Mapping[str, Any]) -> DurableAttemptJournal:
    return DurableAttemptJournal.create(
        tmp_path / "private" / "attempts.jsonl",
        experiment_id=config["campaign_id"],
        source_bindings={"test_source_sha256": "a" * 64},
        unblinding_key=b"j" * 32,
    )


def _with_test_credential(callback):
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "secret-never-persisted"
    try:
        return callback()
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original


def test_recipe_parser_accepts_typed_schedule_and_rejects_prose():
    recipe = parse_recipe("ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120", EXPERIMENT)
    assert recipe == {
        "basis": "lattice_transform",
        "moduli": [24, 120],
        "t_offsets": [0, 7],
        "x_offsets": [0, 2, 65],
    }
    assert parse_recipe("try a clever lattice", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=magic|x=0|t=0|m=24", EXPERIMENT) is None
    assert parse_recipe("ESDSL1|basis=divisor_pair|x=999|t=0|m=24", EXPERIMENT) is None


def test_exact_pair_schedule_produces_replayable_witnesses():
    members = _es_hard_members(10_000)
    wx, wy, resolved, lane_tests, _ = _run_pairs(__import__("numpy"), members, [(0, 0), (1, 0)])
    assert lane_tests > 0
    assert int(resolved.sum()) > 0
    assert _witness_sample(members, wx, wy, resolved)


def test_mutation_preserves_direct_pairs_and_adds_neighbors():
    pairs = _mutated_pairs([[64, 32]], EXPERIMENT)
    assert (64, 32) in pairs
    assert (62, 30) in pairs
    assert (66, 34) in pairs
    assert len(pairs) == 25


def test_response_survives_process_stop_and_replays_without_redispatch(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    first = ShadowTransport(crash_after_first_response=True)
    with pytest.raises(SimulatedCrash, match="durable response"):
        _with_test_credential(
            lambda: _creative_calls(config, journal, base_transport=first)
        )
    loaded = DurableAttemptJournal.load(journal.path)
    assert loaded.event_counts() == {
        "journal_header": 1,
        "message_dispatch": 1,
        "message_response": 1,
        "model_probe_dispatch": 1,
        "model_probe_response": 1,
    }
    resumed_transport = NoCallTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, loaded, base_transport=resumed_transport)
    )
    assert resumed_transport.calls == 0
    assert len(ideas) == 1
    assert evidence["budget"] == {
        "calls": 1,
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
    }
    assert evidence["attempts"][0]["status"] == "completed"
    assert "secret-never-persisted" not in journal.path.read_text(encoding="utf-8")


def test_missing_credential_does_not_permanently_consume_a_slot(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    original = os.environ.pop("ANTHROPIC_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="no journaled Claude idea"):
            _creative_calls(config, journal, base_transport=NoCallTransport())
    finally:
        if original is not None:
            os.environ["ANTHROPIC_API_KEY"] = original
    assert journal.event_counts() == {"journal_header": 1}
    transport = ShadowTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=transport)
    )
    assert len(ideas) == 1
    assert evidence["budget"]["calls"] == 1


def test_restart_accepts_repeated_identical_model_probes_and_recovers_critic(tmp_path: Path):
    config = _test_config(critics=False)
    journal = _journal(tmp_path, config)
    _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=ShadowTransport())
    )
    config["experiment"]["llm_critic_batch_size"] = 1
    with pytest.raises(SimulatedCrash, match="durable response"):
        _with_test_credential(
            lambda: _creative_calls(
                config,
                DurableAttemptJournal.load(journal.path),
                base_transport=ShadowTransport(crash_after_first_response=True),
            )
        )
    loaded = DurableAttemptJournal.load(journal.path)
    assert loaded.event_counts()["model_probe_response"] == 2
    no_calls = NoCallTransport()
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, loaded, base_transport=no_calls)
    )
    assert no_calls.calls == 0
    assert evidence["budget"]["calls"] == 2
    assert ideas[0]["critic_source"] == "journaled_llm_critic_retained_without_pruning"


def test_later_bad_critic_is_retained_and_all_slots_resume_without_calls(tmp_path: Path):
    config = _test_config(critics=True)
    journal = _journal(tmp_path, config)
    first = ShadowTransport(invalid_last_critic=True)
    ideas, evidence = _with_test_credential(
        lambda: _creative_calls(config, journal, base_transport=first)
    )
    assert len([item for item in first.requests if item["method"] == "POST"]) == 3
    assert evidence["budget"]["calls"] == 3
    assert [item["status"] for item in evidence["attempts"]] == [
        "completed",
        "completed",
        "client_or_contract_failure",
    ]
    assert ideas[0]["critic_source"] == "journaled_llm_critic_retained_without_pruning"
    assert ideas[1]["critic_source"] == "deterministic_machine_admission_not_llm_critique"
    assert all(item["critic"]["verdict"] != "reject" for item in ideas)
    resumed_transport = NoCallTransport()
    resumed_ideas, resumed_evidence = _with_test_credential(
        lambda: _creative_calls(
            config,
            DurableAttemptJournal.load(journal.path),
            base_transport=resumed_transport,
        )
    )
    assert resumed_transport.calls == 0
    assert resumed_ideas == ideas
    assert resumed_evidence == evidence


def test_shipped_live_receipt_validates_and_preserves_claim_boundary():
    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    validate_receipt(receipt, ROOT)
    assert receipt["status"] == "PASS_BOUNDED_CREATIVE_SHADOW_NO_OPEN_PROBLEM_CLAIM"
    assert receipt["accounting"] == {
        "baseline_gpu_lane_tests": 104_839_060,
        "creative_tail_lane_tests": 344_279,
        "denominators_covered": 99_999_999,
        "executable_llm_ideas": 11,
        "llm_ideas_proposed": 12,
        "llm_provider_calls": 4,
        "matched_control_lane_tests": 33_918_680,
        "mutated_parameter_pairs": 1_051,
        "retained_llm_provider_calls": 1,
        "total_exact_modular_lane_tests": 146_588_698,
    }
    assert receipt["hard_tail_funnel"]["creative_tail"]["resolved_from_baseline_tail"] == 173
    assert receipt["hard_tail_funnel"]["creative_tail"]["independent_cpu_exact_verified"] == 173
    rewired = receipt["hard_tail_funnel"]["matched_random_controls"]["pairing_only_rewire"]
    assert rewired["median_resolved"] == "174.000000"
    assert rewired["random_at_least_creative"] == 24
    assert all(value is False for value in receipt["claim_boundary"].values())
