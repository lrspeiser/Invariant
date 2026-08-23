from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from sigma_theory_compiler import external_creativity_validation as E
from sigma_theory_compiler.claude_creativity_api import CLAUDE_OUTPUT_SCHEMA_VERSION

ROOT = Path(__file__).resolve().parents[1]
MODEL = "claude-opus-4-6"


class CampaignClaudeTransport:
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
            {"body": parsed, "headers": dict(headers), "method": method, "url": url}
        )
        if method == "GET":
            return 200, {
                "capabilities": {"structured_outputs": {"supported": True}},
                "id": MODEL,
                "type": "model",
            }
        assert parsed is not None
        prompt = json.loads(parsed["messages"][0]["content"])
        benchmark_id = prompt["benchmark"]["blind_id"]
        role = prompt["role"]
        if role == "proposer":
            hypothesis = {
                "expression": "x0",
                "falsifiers": ["sealed holdout"],
                "family": "analogy_transfer",
                "hypothesis_id": f"hypothesis.{len(self.requests)}",
                "invariants": ["identity_scaling"],
                "known_analogues": ["identity map"],
                "llm_origin_assessment": "known_rewrite",
                "proof_plan": ["test base cases", "induct"],
                "rationale": "A deliberately simple typed control hypothesis.",
                "representation": "sympy_expression",
                "source_idea_domains": ["algebra", "recurrences"],
                "synthesis_note": "A control recovered through recurrence language.",
            }
            hypothesis_schema = parsed["output_config"]["format"]["schema"]["properties"][
                "hypotheses"
            ]
            if hypothesis_schema["type"] == "object":
                hypotheses = {
                    name: hypothesis | {"hypothesis_id": f"{hypothesis['hypothesis_id']}.{name}"}
                    for name in hypothesis_schema["required"]
                }
            else:
                hypotheses = [hypothesis]
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": hypotheses,
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": [],
            }
        else:
            candidate_id = prompt["candidate_summaries"][0]["candidate_id"]
            output = {
                "benchmark_id": benchmark_id,
                "hypotheses": [],
                "role": role,
                "schema_version": CLAUDE_OUTPUT_SCHEMA_VERSION,
                "steering_actions": [
                    {
                        "blocker_kind": "train_residual",
                        "candidate_id": candidate_id,
                        "distance_denominator": 1,
                        "distance_numerator": 1,
                        "repair": "Change representation and test another invariant.",
                        "verdict": "repair",
                    }
                ],
            }
        return 200, {
            "content": [{"text": json.dumps(output), "type": "text"}],
            "id": f"msg_campaign_{len(self.requests)}",
            "model": MODEL,
            "role": "assistant",
            "stop_reason": "end_turn",
            "type": "message",
            "usage": {"input_tokens": 200, "output_tokens": 100},
        }


@pytest.fixture(scope="module")
def dry_receipt() -> dict[str, Any]:
    return E.run_campaign(ROOT)


def test_external_authorship_is_distinct_and_generation_view_is_anonymous() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    assert public["generator_principal_id"] == "invariant.discovery-engine"
    assert len(benchmarks) == 4
    assert [item.capability_level for item in benchmarks].count(4) == 2
    assert [item.capability_level for item in benchmarks].count(5) == 2
    for benchmark in benchmarks:
        assert benchmark.source.authoring_principal_id.startswith("external.")
        assert benchmark.source.authoring_principal_id != public["generator_principal_id"]
        generation = benchmark.generation_view()
        serialized = json.dumps(generation, sort_keys=True)
        assert benchmark.benchmark_id not in serialized
        assert benchmark.source.source_uri not in serialized
        assert "target" not in serialized
        assert "holdout" not in serialized
        assert list(generation["variables"]) == [
            f"x{index}" for index in range(len(benchmark.variables))
        ]


def test_targets_open_commitments_and_bounded_unknown_has_no_formula() -> None:
    public, benchmarks = E.load_public_benchmarks(ROOT)
    targets = E.unseal_targets(ROOT, public, benchmarks)
    assert len(targets) == 4
    for target in targets:
        benchmark = next(item for item in benchmarks if item.benchmark_id == target.benchmark_id)
        assert target.commitment == benchmark.target_commitment
        if target.target_kind == "bounded_unknown":
            assert target.reference_formula is None
        else:
            assert target.reference_formula is not None


def test_target_commitment_tamper_fails_closed(tmp_path: Path) -> None:
    public = json.loads((ROOT / E.PUBLIC_CONFIG_PATH).read_text(encoding="utf-8"))
    targets = json.loads(
        (ROOT / "configs/external_sealed_creativity_targets.json").read_text(encoding="utf-8")
    )
    targets["targets"][0]["holdout_records"][0]["output"] = "101"
    (tmp_path / "configs").mkdir()
    (tmp_path / E.PUBLIC_CONFIG_PATH).write_text(json.dumps(public), encoding="utf-8")
    (tmp_path / public["sealed_targets_path"]).write_text(json.dumps(targets), encoding="utf-8")
    _, benchmarks = E.load_public_benchmarks(tmp_path)
    with pytest.raises(E.ExternalCreativityError, match="does not open"):
        E.unseal_targets(tmp_path, public, benchmarks)


def test_source_bindings_normalize_git_checkout_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(b"first\nsecond\n")
    crlf.write_bytes(b"first\r\nsecond\r\n")
    assert E._file_sha256(lf) == E._file_sha256(crlf)


def test_known_and_bounded_unknown_campaign_is_honest(dry_receipt: dict[str, Any]) -> None:
    assert dry_receipt["schema_version"] == E.RECEIPT_SCHEMA
    assert dry_receipt["claims"] == {
        "claude_used_throughout": False,
        "externally_authored_sealed_benchmarks_executed": True,
        "novel_formula_established": False,
        "open_problem_attempted": False,
        "open_problem_solved": False,
    }
    known = [item for item in dry_receipt["benchmarks"] if item["target_kind"] == "known_formula"]
    unknown = [
        item for item in dry_receipt["benchmarks"] if item["target_kind"] == "bounded_unknown"
    ]
    assert len(known) == len(unknown) == 2
    assert all(item["claims"]["known_formula_rediscovered"] for item in known)
    assert all(item["ranked_candidates"][0]["holdout_loss"] == "0" for item in known)
    assert all(not item["claims"]["novel_formula_established"] for item in unknown)
    assert all(item["ranked_candidates"][0]["holdout_loss"] != "0" for item in unknown)
    assert all(not item["bounded_unknown_process_pass"] for item in unknown)


def test_every_family_has_matched_random_and_leave_one_out_controls(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        metrics = benchmark["family_metrics"]
        ablations = benchmark["family_ablation"]
        assert [item["family"] for item in metrics] == list(E.FAMILY_IDS)
        assert [item["family"] for item in ablations] == list(E.FAMILY_IDS)
        assert all(item["candidate_budget"] == item["matched_random_budget"] for item in metrics)
        assert all(item["candidate_budget"] > 0 for item in metrics)
        assert all(item["unique_behaviors"] > 0 for item in metrics)
        assert len(benchmark["random_controls"]) == len(E.FAMILY_IDS)


def test_exact_cas_smt_interval_pass_but_kernel_and_release_fail_closed(
    dry_receipt: dict[str, Any],
) -> None:
    known = [item for item in dry_receipt["benchmarks"] if item["target_kind"] == "known_formula"]
    for benchmark in known:
        assert benchmark["formal_verification"]["backends"] == {
            "cas": True,
            "exact_arithmetic": True,
            "interval": True,
            "lean": False,
            "smt": True,
        }
        assert not benchmark["formal_verification"]["serious_claim_eligible"]
        assert (
            benchmark["prior_art"]["external_literature_index"]["status"]
            == "AUTHORITATIVE_PRIOR_ART_RECORD_FOUND_NOT_NOVELTY_CLEARED"
        )
        assert benchmark["prior_art"]["human_review"]["status"] == "NOT_PERFORMED"
        assert not benchmark["claims"]["serious_claim_released"]
    assert dry_receipt["serious_claim_policy"]["released_claims"] == 0


def test_dataset_pipeline_contains_units_groups_residuals_interventions_and_ood(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        evidence = benchmark["dataset_evidence"]
        assert evidence["dimension_basis"] == ["mass", "length", "time"]
        assert evidence["dimension_matrix_rank"] >= 0
        assert "dimension_solution_set" in evidence
        assert "dimensionless_group_basis" in evidence
        assert evidence["unit_normalization"]["inputs"]
        assert evidence["symmetry_groups"]["declared_coordinates"]
        assert evidence["residual_channels"]["declared"]
        assert evidence["causal_interventions"]["declared"]
        assert (
            evidence["causal_interventions"]["execution_status"]
            == "DECLARED_REQUIRES_INTERVENTION_DATA"
        )
        assert not evidence["causal_interventions"]["observational_rows_mislabelled_as_interventions"]
        assert evidence["ood_split_rule"]
        assert evidence["holdout_opened_last"]


def test_proof_plan_search_and_independent_exact_implementation_are_recorded(
    dry_receipt: dict[str, Any],
) -> None:
    for benchmark in dry_receipt["benchmarks"]:
        proof_search = benchmark["proof_plan_search"]
        assert proof_search["selected_route"][0] == "exact_row_replay"
        assert any(item["plan"] == "lean_kernel_bridge" for item in proof_search["plans"])
        reproduction = benchmark["independent_exact_reproduction"]
        assert reproduction["implementation"] == "python_stdlib_fraction_ast_v1"
        assert reproduction["match"]
        assert not reproduction["shared_symbolic_runtime"]


def test_target_is_opened_after_claude_and_proposal_seals(dry_receipt: dict[str, Any]) -> None:
    events = dry_receipt["blind_chronology"]
    assert [item["sequence"] for item in events] == list(range(len(events)))
    target_open = next(item for item in events if item["target_reads"] == 1)
    proposal_seal = next(
        item for item in events if item["event"] == "proposal_roots_and_train_evidence_sealed"
    )
    critique = next(
        item for item in events if item["event"] == "claude_blind_critique_completed_or_blocked"
    )
    assert proposal_seal["sequence"] < critique["sequence"] < target_open["sequence"]
    assert all(item["target_reads"] == 0 for item in events[: target_open["sequence"]])


def test_live_claude_fixture_proposes_and_steers_without_verifying(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "fixture-secret")
    transport = CampaignClaudeTransport()
    receipt = E.run_campaign(ROOT, live_claude=True, claude_transport=transport)
    assert receipt["claude"]["status"] == "PASS"
    assert receipt["claude"]["completed_calls"] == receipt["claude"]["required_calls"] == 8
    assert receipt["claude"]["proposer_hypotheses"] == 4
    assert receipt["claude"]["steering_actions"] == 4
    assert receipt["claims"]["claude_used_throughout"]
    assert all(
        item["proposer_admission"]
        == {
            "admitted_executable_hypotheses": 1,
            "non_executable_typed_hypotheses": 0,
            "proposed_hypotheses": 1,
        }
        for item in receipt["benchmarks"]
    )
    assert receipt["serious_claim_policy"]["released_claims"] == 0
    assert [item["method"] for item in transport.requests].count("GET") == 1
    assert [item["method"] for item in transport.requests].count("POST") == 8
    assert "fixture-secret" not in json.dumps(receipt, sort_keys=True)


def test_open_problem_spend_and_reproduction_remain_blocked(dry_receipt: dict[str, Any]) -> None:
    gate = dry_receipt["open_problem_gate"]
    assert gate["level5_process_passes"] == 0
    assert gate["minimum_independent_level5_passes"] == 3
    assert not gate["authorized"]
    assert gate["public_failure_receipt_required"]
    assert gate["success_criteria"]["sealed_holdout_loss"] == "0"
    reproduction = dry_receipt["independent_reproduction"]
    assert reproduction["status"] == "IMPLEMENTATIONS_PASS_MACHINE_PENDING"
    assert reproduction["received_machines"] < reproduction["minimum_machines"]
    assert reproduction["received_implementations"] == reproduction["minimum_implementations"]


def test_dry_receipt_replays_byte_for_byte(dry_receipt: dict[str, Any]) -> None:
    assert E.run_campaign(ROOT) == dry_receipt
    assert json.loads((ROOT / E.OUTPUT_PATH).read_text(encoding="utf-8")) == dry_receipt
    assert set(dry_receipt["config"]) == {
        "campaign_sha256",
        "claude_source_sha256",
        "claude_transport_source_sha256",
        "independent_evaluator_sha256",
        "lean_source_sha256",
        "public_benchmarks_sha256",
        "sealed_targets_sha256",
        "source_sha256",
        "test_sha256",
    }
    assert dry_receipt["content_sha256"] == E.canonical_sha256(
        {key: value for key, value in dry_receipt.items() if key != "content_sha256"}
    )
