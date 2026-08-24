from __future__ import annotations

import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import creativity_tournament_generation as T
from sigma_theory_compiler.claude_creativity_api import (
    CLAUDE_OUTPUT_SCHEMA_VERSION,
    ClaudeCreativityError,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"


class TournamentTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.arm_requests: list[tuple[str, str, str]] = []

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
        arm = (
            "full_creativity_first"
            if "creative component" in parsed["system"]
            else "baseline"
        )
        self.arm_requests.append((benchmark_id, role, arm))
        if role == "proposer":
            output = self._proposal(benchmark_id, len(self.arm_requests))
        else:
            output = self._critique(benchmark_id, prompt["candidate_summaries"])
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg.tournament.{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 250, "output_tokens": 175},
        }

    @staticmethod
    def _proposal(benchmark_id: str, call_number: int) -> dict[str, Any]:
        hypotheses = []
        representations = ("linear_recurrence", "generating_function", "modular_relation")
        origins = ("known_rewrite", "cross_domain_synthesis", "uncertain")
        for index in range(3):
            hypotheses.append(
                {
                    "expression": f"relation_{call_number}_{index}(n)",
                    "falsifiers": [f"a separating index for route {index}"],
                    "family": f"structural_family_{index}",
                    "hypothesis_id": f"hypothesis.{call_number}.{index}",
                    "invariants": [f"invariant {index}"],
                    "known_analogues": [f"analogue {index}"],
                    "llm_origin_assessment": origins[index],
                    "proof_plan": [f"derive route {index}", "test boundary cases"],
                    "rationale": "A bounded synthetic response for the paired-run contract test.",
                    "representation": representations[index],
                    "source_idea_domains": [f"domain {index}", f"bridge {index}"],
                    "synthesis_note": "Lineage is uncertain until claim-specific prior-art review.",
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
    def _critique(
        benchmark_id: str, summaries: list[dict[str, Any]]
    ) -> dict[str, Any]:
        verdicts = ("reject", "repair", "retain")
        actions = []
        for index, summary in enumerate(summaries):
            actions.append(
                {
                    "blocker_kind": f"typed_blocker_{index}",
                    "candidate_id": summary["candidate_id"],
                    "distance_denominator": 4,
                    "distance_numerator": index + 1,
                    "repair": "Change representation while preserving the original lineage.",
                    "verdict": verdicts[index],
                }
            )
        return {
            "benchmark_id": benchmark_id,
            "hypotheses": [],
            "role": "critic",
            "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": actions,
        }


@pytest.fixture(scope="module")
def paired_run() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], TournamentTransport]:
    transport = TournamentTransport()
    original = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = "test-secret-never-persisted"
    try:
        review, public, coordinator = T.run_generation(
            ROOT, unblinding_key=b"k" * 32, transport=transport
        )
    finally:
        if original is None:
            os.environ.pop("ANTHROPIC_API_KEY", None)
        else:
            os.environ["ANTHROPIC_API_KEY"] = original
    return review, public, coordinator, transport


def test_paired_run_is_complete_matched_and_publicly_blinded(paired_run) -> None:
    review, public, coordinator, transport = paired_run
    assert len(review["blinded_outputs"]) == 48
    assert public["claude_runtime"]["completed_calls"] == 96
    assert len(coordinator["arm_calls"]["baseline"]) == 48
    assert len(coordinator["arm_calls"]["full_creativity_first"]) == 48
    assert len([request for request in transport.requests if request["method"] == "GET"]) == 2
    assert len([request for request in transport.requests if request["method"] == "POST"]) == 96
    assert all(
        output["resource_budget"] == review["blinded_outputs"][0]["resource_budget"]
        for output in review["blinded_outputs"]
    )
    serialized = json.dumps(review, sort_keys=True)
    assert '"arm"' not in serialized
    assert '"baseline"' not in serialized
    assert '"full_creativity_first"' not in serialized
    assert "test-secret-never-persisted" not in json.dumps(
        [review, public, coordinator], sort_keys=True
    )
    assert public["credential_activation"]["credential_persisted"] is False
    assert public["release_gate"]["tournament_scored"] is False
    assert all(event["target_reads"] == 0 for event in public["chronology"])


def test_treatment_retains_failed_lineage_and_expands_routes(paired_run) -> None:
    review, _, coordinator, _ = paired_run
    outputs = {item["blinded_output_id"]: item for item in review["blinded_outputs"]}
    branch_counts = {"baseline": [], "full_creativity_first": []}
    statuses = {"baseline": set(), "full_creativity_first": set()}
    for binding in coordinator["mapping"]:
        output = outputs[binding["blinded_output_id"]]
        branch_counts[binding["arm"]].append(len(output["branches"]))
        statuses[binding["arm"]].update(
            branch["initial_check_status"] for branch in output["branches"]
        )
    assert set(branch_counts["baseline"]) == {2}
    assert set(branch_counts["full_creativity_first"]) == {12}
    assert "failed" not in statuses["baseline"]
    assert "failed" in statuses["full_creativity_first"]


def test_tournament_seals_detect_public_and_private_tampering(paired_run) -> None:
    review, public, coordinator, _ = paired_run
    changed_review = copy.deepcopy(review)
    changed_review["blinded_outputs"][0]["branches"] = []
    with pytest.raises(T.TournamentGenerationError, match="content seal"):
        T.validate_public_generation(changed_review, public)

    changed_coordinator = copy.deepcopy(coordinator)
    changed_coordinator["mapping"][0]["arm"] = "full_creativity_first"
    changed_coordinator["content_sha256"] = canonical_sha256(
        {key: value for key, value in changed_coordinator.items() if key != "content_sha256"}
    )
    with pytest.raises(T.TournamentGenerationError, match="unblinding map"):
        T.validate_coordinator(changed_coordinator, review, public)


def test_transient_transport_errors_retry_but_contract_errors_do_not(monkeypatch) -> None:
    calls = 0

    def flaky(*_args):
        nonlocal calls
        calls += 1
        if calls < 3:
            raise ClaudeCreativityError("Claude API returned HTTP 429: rate_limit")
        return 200, {"ok": True}

    monkeypatch.setattr(T.time, "sleep", lambda _seconds: None)
    assert T._retrying_transport(flaky)("GET", "https://example.test", {}, None, 1) == (
        200,
        {"ok": True},
    )
    assert calls == 3

    def invalid(*_args):
        raise ClaudeCreativityError("invalid structured response")

    with pytest.raises(ClaudeCreativityError, match="invalid structured"):
        T._retrying_transport(invalid)("GET", "https://example.test", {}, None, 1)


def test_generation_resumes_from_a_private_task_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-never-persisted")
    first = TournamentTransport()
    post_calls = 0

    def interrupted(method, url, headers, body, timeout):
        nonlocal post_calls
        if method == "POST":
            post_calls += 1
            if post_calls == 10:
                raise ClaudeCreativityError("intentional non-transient interruption")
        return first(method, url, headers, body, timeout)

    checkpoint = tmp_path / "work" / "paired-private.json"
    with pytest.raises(ClaudeCreativityError, match="intentional"):
        T.run_generation(
            ROOT,
            unblinding_key=b"r" * 32,
            transport=interrupted,
            checkpoint_path=checkpoint,
        )
    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["schema_version"] == T.CHECKPOINT_SCHEMA
    assert len(saved["completed_task_ids"]) == 2

    resumed = TournamentTransport()
    review, public, coordinator = T.run_generation(
        ROOT,
        unblinding_key=b"r" * 32,
        transport=resumed,
        checkpoint_path=checkpoint,
    )
    assert public["claude_runtime"]["completed_calls"] == 96
    assert len([item for item in resumed.requests if item["method"] == "POST"]) == 88
    T.validate_public_generation(review, public, ROOT)
    T.validate_coordinator(coordinator, review, public)


def test_cli_private_state_must_stay_under_ignored_work(tmp_path: Path) -> None:
    root = tmp_path.resolve()
    assert T._private_output_path(root, root / "work" / "private.json").parent.name == "work"
    with pytest.raises(T.TournamentGenerationError, match="ignored work"):
        T._private_output_path(root, root / "runs" / "private.json")


def test_historical_source_compatibility_is_exact_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    review = json.loads(
        (ROOT / "runs/math/creativity-tournament/paired-review-packet.json").read_text(
            encoding="utf-8"
        )
    )
    public = json.loads(
        (ROOT / "runs/math/creativity-tournament/paired-generation-receipt.json").read_text(
            encoding="utf-8"
        )
    )
    T.validate_public_generation(review, public, ROOT)
    monkeypatch.chdir(ROOT)
    T.validate_public_generation(review, public, Path(""))

    tampered = copy.deepcopy(public)
    tampered["source_bindings"]["claude_adapter"]["sha256"] = "0" * 64
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(T.TournamentGenerationError, match="source binding changed"):
        T.validate_public_generation(review, tampered, ROOT)


def test_live_pilot_deviation_blocks_confirmatory_claims() -> None:
    path = ROOT / "runs/math/creativity-tournament/pilot-deviation.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    body = {key: item for key, item in value.items() if key != "content_sha256"}
    assert value["content_sha256"] == canonical_sha256(body)
    assert value["retained_packet"]["completed_messages_api_calls"] == 96
    assert value["disposition"]["blinded_pilot_review_eligible"] is True
    assert value["disposition"]["confirmatory_decision_eligible"] is False
    assert not any(value["claims"].values())


def test_generation_loader_rejects_target_bearing_bindings(tmp_path: Path) -> None:
    config = T.load_config(ROOT)
    config["generation_packet"] = dict(config["generation_packet"])
    config["generation_packet"]["path"] = "runs/private-targets.json"
    (tmp_path / "configs").mkdir()
    (tmp_path / T.CONFIG_PATH).write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(T.TournamentGenerationError, match="target-bearing"):
        T.load_config(tmp_path)
