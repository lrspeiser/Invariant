from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from sigma_theory_compiler import core_creative_discovery as C
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


def _fake_campaign() -> dict[str, object]:
    calls = []
    for index in range(8):
        role = "proposer" if index < 4 else "critic"
        benchmark_id = f"blind.benchmark-{index % 4}"
        output = {
            "benchmark_id": benchmark_id,
            "hypotheses": (
                [
                    {
                        "expression": f"x{index} + 1",
                        "falsifiers": ["test shifted samples"],
                        "family": "cross_domain_trial",
                        "hypothesis_id": f"idea.test-{index}",
                        "invariants": ["translation response"],
                        "known_analogues": ["finite-difference identity"],
                        "llm_origin_assessment": (
                            "cross_domain_synthesis" if index % 2 == 0 else "uncertain"
                        ),
                        "proof_plan": ["seek an induction invariant"],
                        "rationale": "Keep the structural branch alive while testing it.",
                        "representation": "other_typed_relation",
                        "source_idea_domains": ["finite differences", "dynamical systems"],
                        "synthesis_note": "Transfers translation structure into a recurrence lens.",
                    }
                ]
                if role == "proposer"
                else []
            ),
            "quarantine": {
                "rejected_hypotheses": 0,
                "rejected_steering_actions": 0,
            },
            "role": role,
            "schema_version": "invariant-claude-creativity-output-2.0",
            "steering_actions": (
                []
                if role == "proposer"
                else [
                    {
                        "blocker_kind": "execution_grammar_gap",
                        "candidate_id": f"candidate.test-{index}",
                        "distance_denominator": 4,
                        "distance_numerator": 1,
                        "repair": "Retain and try a generating-function transform.",
                        "verdict": "repair",
                    }
                ]
            ),
        }
        calls.append(
            {
                "benchmark_id": benchmark_id,
                "evidence": {
                    "api_response_id": f"msg_test_{index}",
                    "capabilities_sha256": "a" * 64,
                    "credential_persisted": False,
                    "model": "claude-opus-4-6",
                    "model_evidence": {
                        "capabilities_sha256": "b" * 64,
                        "model": "claude-opus-4-6",
                        "structured_outputs_supported": True,
                    },
                    "output_sha256": "c" * 64,
                    "prompt_sha256": "d" * 64,
                    "raw_output_sha256": "e" * 64,
                    "request_schema_sha256": "f" * 64,
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "wire_contract_adapter_used": role == "critic",
                },
                "output": output,
                "role": role,
                "status": "completed",
            }
        )
    body: dict[str, object] = {
        "schema_version": "invariant-external-creativity-validation-result-1.0",
        "benchmarks": [
            {
                "benchmark_id": "external.test",
                "capability_level": 5,
                "claims": {
                    "known_formula_rediscovered": False,
                    "serious_claim_released": False,
                },
                "prior_art": {"human_review": {"status": "NOT_PERFORMED"}},
                "ranked_candidates": [
                    {
                        "candidate_id": "candidate.test",
                        "expression": "x0",
                        "holdout_loss": "1",
                        "proposer": "claude_api",
                    }
                ],
                "target_kind": "bounded_unknown",
                "unique_behaviors": 1,
                "unique_proof_mechanisms": 1,
            }
        ],
        "claims": {
            "claude_used_throughout": True,
            "novel_formula_established": False,
            "open_problem_solved": False,
        },
        "claude": {
            "budget": {
                "calls": 8,
                "input_tokens": 80,
                "output_tokens": 40,
                "total_tokens": 120,
            },
            "calls": calls,
            "completed_calls": 8,
            "required_calls": 8,
            "proposer_hypotheses": 1,
            "status": "PASS",
            "steering_actions": 1,
        },
        "open_problem_gate": {"authorized": False, "level5_process_passes": 0},
    }
    body["content_sha256"] = canonical_sha256(body)
    return body


def test_core_run_requires_and_sanitizes_live_claude(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    credential_file = tmp_path / "secret.env"
    secret = "test-secret-that-must-not-survive"
    credential_file.write_text(f"ANTHROPIC_API_KEY={secret}\n", encoding="utf-8")
    operational = json.loads(
        (ROOT / "runs/math/declarative-discovery-platform/operational-v2.json").read_text()
    )
    multi_host = json.loads(
        (ROOT / "runs/math/external-creativity-validation/multi-host-reproduction.json").read_text()
    )
    bound_receipts = C._load_bound_receipts(ROOT, C._load_config(ROOT))
    expanded_grammar = bound_receipts[2]
    dataset_challenges = bound_receipts[3]
    external_dataset_challenges = bound_receipts[4]
    external_structured_benchmarks = bound_receipts[5]
    symmetry_dimension_derivation = bound_receipts[6]
    proof_plan_search = bound_receipts[7]
    serious_claim_ladder = bound_receipts[8]
    component_knockout = bound_receipts[9]
    monkeypatch.setattr(
        C,
        "_load_bound_receipts",
        lambda *_: (
            operational,
            multi_host,
            expanded_grammar,
            dataset_challenges,
            external_dataset_challenges,
            external_structured_benchmarks,
            symmetry_dimension_derivation,
            proof_plan_search,
            serious_claim_ladder,
            component_knockout,
        ),
    )
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    def runner(_: Path) -> dict[str, object]:
        assert os.environ["ANTHROPIC_API_KEY"] == secret
        return _fake_campaign()

    receipt = C.run_core(ROOT, credential_file=credential_file, campaign_runner=runner)
    C.validate_receipt(receipt, ROOT)
    assert receipt["claude_runtime"]["authenticated_messages_api_working"]
    assert receipt["claude_runtime"]["completed_calls"] == 8
    assert receipt["credential_activation"]["source_kind"] == "explicit_env_file"
    assert receipt["idea_lineage_archive"]["summary"]["ideas_received"] == 4
    assert receipt["idea_lineage_archive"]["summary"]["ideas_retained"] == 4
    assert receipt["discovery_runtime"]["typed_grammar_controls_passed"] == 7
    assert "variational_functional" in receipt["discovery_runtime"]["typed_formula_kinds"]
    assert receipt["discovery_runtime"]["dataset_positive_controls_passed"] == 4
    assert receipt["discovery_runtime"]["dataset_mutation_controls_rejected"] == 4
    assert receipt["discovery_runtime"]["external_dataset_challenges_passed"] == 4
    assert receipt["discovery_runtime"]["external_dataset_mutation_controls_rejected"] == 4
    assert receipt["external_dataset_challenges"]["release_gate"]["level5_eligible"] is False
    assert receipt["discovery_runtime"]["external_structured_benchmark_tasks"] == 8
    assert receipt["discovery_runtime"]["external_structured_benchmark_families"] == [
        "tensor_identity",
        "variational_functional",
    ]
    assert (
        receipt["external_structured_benchmarks"]["release_gate"]["level5_eligible"]
        is False
    )
    assert receipt["discovery_runtime"]["first_principles_d4_controls_passed"] == 4
    assert receipt["discovery_runtime"]["first_principles_d4_invariant_coordinates"] == 4
    assert (
        receipt["discovery_runtime"]["first_principles_d4_dimension_mutations_rejected"]
        == 4
    )
    assert (
        receipt["discovery_runtime"]["first_principles_d4_symmetry_mutations_rejected"]
        == 4
    )
    assert receipt["symmetry_dimension_derivation"]["claims"]["specific_law_discovered"] is False
    assert receipt["discovery_runtime"]["independent_proof_plan_routes_closed"] == 6
    assert receipt["discovery_runtime"]["independent_proof_plan_mutations_rejected"] == 6
    assert receipt["discovery_runtime"]["component_knockout_experiments_preflighted"] == 4
    assert receipt["discovery_runtime"]["component_knockout_scheduled_slots"] == 384
    assert receipt["discovery_runtime"]["component_knockout_live_runs_complete"] is False
    assert receipt["verification"]["serious_claim_required_stage_order"] == [
        "exact_arithmetic",
        "cas",
        "smt",
        "interval",
        "lean",
    ]
    assert receipt["verification"]["serious_claims_released_by_ladder"] == 0
    assert receipt["verification"]["serious_claim_backend_mutations_rejected"] == 10
    assert receipt["verification"]["serious_claim_lean_mutation_artifact_bound"] is True
    assert all(
        idea["retention_status"] == "RETAINED_ACTIVE"
        for idea in receipt["idea_lineage_archive"]["ideas"]
    )
    assert "ANTHROPIC_API_KEY" not in os.environ
    assert secret not in json.dumps(receipt, sort_keys=True)
    assert "test-secret" not in json.dumps(receipt, sort_keys=True)


def test_core_receipt_fails_when_claude_health_claim_is_removed() -> None:
    path = ROOT / C.OUTPUT_PATH
    if not path.is_file():
        pytest.skip("live core receipt is generated by the authenticated integration run")
    receipt = json.loads(path.read_text(encoding="utf-8"))
    changed = json.loads(json.dumps(receipt))
    changed["claude_runtime"]["authenticated_messages_api_working"] = False
    body = {key: item for key, item in changed.items() if key != "content_sha256"}
    changed["content_sha256"] = canonical_sha256(body)
    with pytest.raises(C.CoreCreativeDiscoveryError, match="health"):
        C.validate_receipt(changed)


def test_stored_live_core_receipt_validates_against_current_bound_sources() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    C.validate_receipt(receipt, ROOT)


def test_deterministic_rebind_preserves_authenticated_llm_evidence() -> None:
    receipt = json.loads((ROOT / C.OUTPUT_PATH).read_text(encoding="utf-8"))
    rebound = C.rebind_core_receipt(ROOT, receipt)
    assert rebound == receipt
    assert rebound["claude_runtime"] == receipt["claude_runtime"]
    assert rebound["idea_lineage_archive"] == receipt["idea_lineage_archive"]
    assert rebound["verification"]["serious_claim_backend_mutations_rejected"] == 10
    assert rebound["verification"]["serious_claim_lean_mutation_artifact_bound"] is True
