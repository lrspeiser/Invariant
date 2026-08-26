from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import broken_arxiv_task2 as task2
from sigma_theory_compiler import claude_creativity_api as claude

ROOT = Path(__file__).resolve().parents[1]


def _authorization() -> dict:
    return task2.build_authorization(
        ROOT, now=datetime(2026, 8, 26, 12, 0, tzinfo=UTC)
    )


def _catalog(*, future: bool) -> list[dict]:
    rows = [
        {
            "id": "MathArena/brokenarxiv-0626",
            "lastModified": "2026-06-11T00:00:00Z",
            "sha": "6" * 40,
        },
        {
            "id": "MathArena/brokenarxiv-0626_outputs",
            "lastModified": "2026-06-12T00:00:00Z",
            "sha": "a" * 40,
        },
    ]
    if future:
        rows.extend(
            [
                {
                    "id": "MathArena/brokenarxiv-0826",
                    "lastModified": "2026-08-30T00:00:00Z",
                    "sha": "8" * 40,
                },
                {
                    "id": "MathArena/brokenarxiv-0726",
                    "lastModified": "2026-08-29T00:00:00Z",
                    "sha": "7" * 40,
                },
            ]
        )
    return rows


def _projected_packet(config: dict, checked: dict, items: list[dict]) -> dict:
    selected = checked["selected_release"]

    def metadata_fetcher(_: str) -> dict:
        return {
            "id": selected["dataset_id"],
            "sha": selected["revision"],
            "siblings": [{"rfilename": "data/train-00000-of-00001.parquet"}],
        }

    def table_projector(_: str, columns: list[str]) -> list[dict]:
        assert columns == ["problem_idx", "problem"]
        return items

    return task2.fetch_projected_release_packet(
        checked,
        config,
        metadata_fetcher=metadata_fetcher,
        table_projector=table_projector,
    )


def _staged_chain() -> tuple[dict, dict, dict, dict]:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    packet = _projected_packet(
        config,
        checked,
        [
            {"problem_idx": "p3", "problem": "False statement three."},
            {"problem_idx": "p1", "problem": "False statement one."},
            {"problem_idx": "p2", "problem": "False statement two."},
        ],
    )
    staged = task2.stage_problem(authorization, checked, config, packet)
    task2.validate_staged_problem(staged, authorization, checked, config)
    return config, authorization, checked, staged


def _hypothesis(arm: str, slot: int) -> dict:
    return {
        "expression": f"counterexample attempt {arm} {slot}",
        "falsifiers": ["substitute into the statement"],
        "family": "counterexample_repair",
        "hypothesis_id": f"candidate.{arm}.{slot}",
        "invariants": ["statement domain"],
        "known_analogues": ["boundary testing"],
        "llm_origin_assessment": "uncertain",
        "proof_plan": [
            "Failed assumption: candidate premise.",
            "Repaired theorem: restricted statement.",
            "Repair verification: exact substitution.",
        ],
        "rationale": "A bounded candidate for independent evaluation.",
        "representation": "proof_plan",
        "source_idea_domains": ["logic"],
        "synthesis_note": "No novelty claim.",
    }


def _generation() -> tuple[dict, dict, dict, dict, dict]:
    config, _, _, staged = _staged_chain()
    candidates = []
    for items in task2.build_arm_specs(staged, config).values():
        for spec in items:
            hypothesis = _hypothesis(spec["arm"], spec["slot_index"])
            candidates.append(
                {
                    "arm": spec["arm"],
                    "call": {
                        "benchmark_id": "task2.fixture",
                        "evidence": {"credential_persisted": False},
                        "output": {"hypotheses": [hypothesis]},
                        "role": spec["role"],
                        "status": "completed",
                    },
                    "falsifier_family": spec["falsifier_family"],
                    "hypothesis": hypothesis,
                    "role": spec["role"],
                    "slot_index": spec["slot_index"],
                }
            )
    activation = {
        "credential_env_var": "ANTHROPIC_API_KEY",
        "credential_persisted": False,
        "credential_value_recorded": False,
        "injected_into_process": False,
        "source_kind": "test_fixture",
        "source_locator_sha256": "f" * 64,
    }
    public, receipt, coordinator = task2.compile_generation(
        staged,
        config,
        candidates,
        unblinding_key=hashlib.sha256(b"task-2-test-key").digest(),
        credential_activation=activation,
    )
    return config, staged, public, receipt, coordinator


def _evaluation_packet(
    public: dict,
    coordinator: dict,
    *,
    creative_slot: int = 1,
    old_slot: int = 5,
    random_slot: int = 4,
    shared_repair: bool = False,
) -> dict:
    valid_slots = {
        ("creativity_first_llm", creative_slot),
        ("old_failure_first_llm", old_slot),
        ("matched_random_falsifier", random_slot),
    }
    private = {row["submission_id"]: row for row in coordinator["mapping"]}
    evaluations = []
    for submission in public["submissions"]:
        submission_id = submission["submission_id"]
        mapped = private[submission_id]
        valid = (mapped["arm"], mapped["slot_index"]) in valid_slots
        repair_label = "shared" if shared_repair else mapped["arm"]
        evaluations.append(
            {
                "canonical_counterexample_sha256": "c" * 64 if valid else None,
                "canonical_repair_graph_sha256": (
                    hashlib.sha256(repair_label.encode()).hexdigest() if valid else None
                ),
                "counterexample_or_rejection": "x = 0 gives a contradiction" if valid else "none",
                "exact_counterexample_valid": valid,
                "failed_assumption": "nonzero input is required" if valid else "not established",
                "false_as_written": valid,
                "independent_external_rejection_valid": False,
                "notes": "independent fixture adjudication",
                "repair_nonvacuous_valid": valid,
                "repair_proof_or_external_acceptance_valid": valid,
                "repaired_statement": "The statement holds for x != 0." if valid else "none",
                "smallest_failed_assumption_valid": valid,
                "submission_id": submission_id,
                "verifier_invocations": 1,
            }
        )
    return task2.seal_evaluation_packet(
        {
            "task_id": "task2.creative-falsification.future-broken-arxiv",
            "public_submissions_content_sha256": public["content_sha256"],
            "reference_material_opened_after_submissions_sealed": True,
            "evaluator": {
                "counterexample_canonicalizer": "exact normalized witness serialization v1",
                "evidence_uri": "https://example.invalid/task2-evidence",
                "independent_from_generator": True,
                "name": "Exact fixture evaluator",
                "named_human_reviewers": [],
                "organization": "External test authority",
                "proof_graph_canonicalizer": "typed lemma dependency DAG v1",
                "signed_artifact_sha256": "e" * 64,
                "verifier_kind": "exact_executable_verifier",
            },
            "evaluations": evaluations,
        }
    )


class _DynamicClaudeTransport:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.post_calls = 0

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
            {"body": parsed, "headers": dict(headers), "method": method, "url": url}
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": "claude-opus-4-6",
                "type": "model",
            }
        self.post_calls += 1
        prompt = json.loads(parsed["messages"][0]["content"])
        hypothesis = _hypothesis(prompt["role"], self.post_calls)
        output = {
            "benchmark_id": prompt["benchmark"].get("benchmark_id", "task2.fixture"),
            "hypotheses": {"idea_1": hypothesis},
            "role": prompt["role"],
            "schema_version": claude.CLAUDE_OUTPUT_SCHEMA_VERSION,
            "steering_actions": {},
        }
        schema_benchmark = parsed["output_config"]["format"]["schema"]["properties"][
            "benchmark_id"
        ]["const"]
        output["benchmark_id"] = schema_benchmark
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg_task2_{self.post_calls}",
            "model": "claude-opus-4-6",
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 100, "output_tokens": 50},
        }


def test_authorization_binds_implementation_selector_and_zero_target_reads() -> None:
    authorization = _authorization()
    task2.validate_authorization(authorization, ROOT)
    assert authorization["source_cutoff"]["problem_rows_read"] == 0
    assert authorization["source_cutoff"]["reference_answers_read"] == 0
    assert len(authorization["implementation_bindings"]) == 4
    assert len(authorization["selector_commitment"]) == 64


def test_catalog_check_blocks_without_reading_current_problem_rows() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=False))
    task2.validate_source_check(checked, authorization, config)
    assert checked["status"] == "BLOCKED_FUTURE_RELEASE_NOT_PUBLISHED"
    assert checked["selected_release"] is None
    assert checked["catalog_query"]["problem_rows_read"] == 0


def test_first_future_release_is_forced_even_if_later_release_is_listed_first() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    task2.validate_source_check(checked, authorization, config)
    assert checked["status"] == "READY_FIRST_ELIGIBLE_RELEASE_METADATA_ONLY"
    assert checked["selected_release"]["dataset_id"] == "MathArena/brokenarxiv-0726"


def test_backdated_future_release_cannot_cross_the_authorization_boundary() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    catalog = _catalog(future=False) + [
        {
            "id": "MathArena/brokenarxiv-0726",
            "lastModified": "2026-08-25T23:59:59Z",
            "sha": "7" * 40,
        }
    ]
    checked = task2.evaluate_catalog(authorization, config, catalog)
    task2.validate_source_check(checked, authorization, config)
    assert checked["selected_release"] is None


def test_problem_selection_is_deterministic_and_reference_blind() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    raw_items = [
        {"problem_idx": "p3", "problem": "False statement three."},
        {"problem_idx": "p1", "problem": "False statement one."},
        {"problem_idx": "p2", "problem": "False statement two."},
    ]
    packet = _projected_packet(config, checked, raw_items)
    first = task2.stage_problem(authorization, checked, config, packet)
    second = task2.stage_problem(
        authorization, checked, config, _projected_packet(config, checked, list(reversed(raw_items)))
    )
    assert first == second
    assert first["blindness"]["reference_answers_read"] == 0
    selected = first["selection"]
    assert selected["statement_sha256"] == hashlib.sha256(
        selected["statement"].encode()
    ).hexdigest()


def test_reference_material_and_manual_resealing_fail_closed() -> None:
    config = task2.load_config(ROOT)
    authorization = _authorization()
    checked = task2.evaluate_catalog(authorization, config, _catalog(future=True))
    selected = checked["selected_release"]

    def metadata_fetcher(_: str) -> dict:
        return {
            "id": selected["dataset_id"],
            "sha": selected["revision"],
            "siblings": [{"rfilename": "data/train-00000-of-00001.parquet"}],
        }

    def tainted_projector(_: str, __: list[str]) -> list[dict]:
        return [
            {
                "original_problem": "Reference repair.",
                "problem": "False.",
                "problem_idx": 1,
            }
        ]

    with pytest.raises(task2.BrokenArxivTask2Error, match="forbidden columns"):
        task2.fetch_projected_release_packet(
            checked,
            config,
            metadata_fetcher=metadata_fetcher,
            table_projector=tainted_projector,
        )
    changed = deepcopy(checked)
    changed["catalog_query"]["problem_rows_read"] = 1
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="metadata-only chronology"):
        task2.validate_source_check(changed, authorization, config)


def test_three_arm_schedule_is_exactly_resource_matched_and_randomized() -> None:
    config, _, _, staged = _staged_chain()
    specs = task2.build_arm_specs(staged, config)
    assert set(specs) == {
        "old_failure_first_llm",
        "creativity_first_llm",
        "matched_random_falsifier",
    }
    assert {len(items) for items in specs.values()} == {12}
    assert {item["role"] for item in specs["old_failure_first_llm"]} == {"proposer"}
    assert len({item["role"] for item in specs["creativity_first_llm"]}) >= 5
    random_families = [item["falsifier_family"] for item in specs["matched_random_falsifier"]]
    assert set(random_families) == set(config["trial"]["random_falsifier_families"])
    assert random_families != config["trial"]["random_falsifier_families"]


def test_generation_blinds_arm_identity_and_preserves_equal_budgets() -> None:
    config, staged, public, receipt, coordinator = _generation()
    task2.validate_generation(public, receipt, coordinator, staged, config)
    assert len(public["submissions"]) == 36
    assert all("arm" not in row for row in public["submissions"])
    assert len({task2.canonical_sha256(row["resource_budget"]) for row in public["submissions"]}) == 1
    assert receipt["claims"]["candidate_correctness_established"] is False
    assert coordinator["claims"]["safe_to_publish_before_independent_scoring"] is False

    changed = deepcopy(public)
    changed["submissions"][0]["hypothesis"]["expression"] = "resealed replacement"
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="generation receipt changed"):
        task2.validate_generation(changed, receipt, coordinator, staged, config)


def test_live_generation_orchestrates_all_36_structured_calls(monkeypatch) -> None:
    config, _, _, staged = _staged_chain()
    transport = _DynamicClaudeTransport()
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-secret-never-persisted")
    public, receipt, coordinator = task2.run_generation(
        staged,
        config,
        root=ROOT,
        unblinding_key=hashlib.sha256(b"live-task2-fixture").digest(),
        transport=transport,
    )
    task2.validate_generation(public, receipt, coordinator, staged, config)
    assert transport.post_calls == 36
    assert len(transport.requests) == 37
    assert receipt["generation"]["calls"] == 36
    serialized = json.dumps([public, receipt, coordinator], sort_keys=True)
    assert "test-secret-never-persisted" not in serialized


def test_independent_adjudication_passes_only_for_valid_creative_advantage() -> None:
    config, staged, public, receipt, coordinator = _generation()
    evaluation = _evaluation_packet(public, coordinator)
    result = task2.build_adjudication(
        public, receipt, coordinator, staged, config, evaluation
    )
    assert result["decision"] == "PASS"
    assert result["claims"]["task_2_completed"] is True
    assert result["claims"]["historical_novelty_established"] is False
    assert result["comparison"]["creative_lower_search_cost_than_both_controls"] is True
    assert result["comparison"]["creative_distinct_valid_repair_missed_by_both_controls"] is True
    changed = deepcopy(result)
    changed["arm_results"]["creativity_first_llm"]["decisive_valid_results"] = 12
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="adjudication replay changed"):
        task2.validate_adjudication(
            changed, public, receipt, coordinator, staged, config, evaluation
        )


def test_correct_creative_candidate_without_comparative_advantage_is_rejected() -> None:
    config, staged, public, receipt, coordinator = _generation()
    evaluation = _evaluation_packet(
        public,
        coordinator,
        creative_slot=9,
        old_slot=0,
        random_slot=1,
        shared_repair=True,
    )
    result = task2.build_adjudication(
        public, receipt, coordinator, staged, config, evaluation
    )
    assert result["decision"] == "REJECT"
    assert result["claims"]["candidate_correctness_established_by_independent_evaluator"] is True
    assert result["claims"]["creative_method_advantage_established_on_this_problem"] is False
    assert result["claims"]["task_2_completed"] is False


def test_incomplete_or_self_controlled_evaluation_fails_closed() -> None:
    config, _, public, _, coordinator = _generation()
    evaluation = _evaluation_packet(public, coordinator)
    changed = deepcopy(evaluation)
    changed["evaluator"]["independent_from_generator"] = False
    body = {key: value for key, value in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="independently admissible"):
        task2.validate_evaluation_packet(changed, public, config)
    incomplete = deepcopy(evaluation)
    incomplete["evaluations"].pop()
    body = {key: value for key, value in incomplete.items() if key != "content_sha256"}
    incomplete["content_sha256"] = task2.canonical_sha256(body)
    with pytest.raises(task2.BrokenArxivTask2Error, match="score every submission"):
        task2.validate_evaluation_packet(incomplete, public, config)
