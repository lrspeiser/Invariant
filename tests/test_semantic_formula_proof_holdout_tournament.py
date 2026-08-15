from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sigma_theory_compiler.candidate_knowledge_graph import (
    CandidateKnowledgeGraph,
    HoldoutCut,
)
from sigma_theory_compiler.semantic_formula_proof_holdout_tournament import (
    CLAIMS,
    CONFIG_PATH,
    FAMILIES,
    OUTPUT_PATH,
    WORLD_ROWS,
    build_campaign,
    validate_campaign,
)
from sigma_theory_compiler.sigma_core import canonical_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def result() -> dict[str, object]:
    return build_campaign(ROOT)


def _reseal(value: dict[str, object]) -> None:
    value["content_sha256"] = canonical_sha256(
        {key: child for key, child in value.items() if key != "content_sha256"}
    )


def test_three_semantic_worlds_are_preregistered_without_finite_labels() -> None:
    config = json.loads((ROOT / CONFIG_PATH).read_text(encoding="utf-8"))
    assert [
        (row["world_id"], row["public_seed"], row["target_commitment_sha256"])
        for row in config["worlds"]
    ] == list(WORLD_ROWS)
    assert config["generator_families"] == list(FAMILIES)
    assert config["policies"] == {
        "finite_label_selection": "forbidden",
        "generator_target_access": "forbidden",
        "live_sqlite_access": "forbidden",
        "network_access": "forbidden",
        "post_unseal_tuning": "forbidden",
        "proof_required_for_pass": True,
        "target_records_per_unseal": 3,
        "target_unseal_batches": 1,
    }
    assert config["budgets"]["maximum_absolute_coefficient"] == 1_000_003
    assert "hypothesis_inventory_size" not in config["budgets"]


def test_all_structured_generation_precedes_one_atomic_unseal(
    result: dict[str, object],
) -> None:
    ledger = result["phase_ledger"]
    assert ledger["generation_events_before_unseal"] == 21
    assert ledger["pre_unseal_target_access_count"] == 0
    assert ledger["atomic_unseal_batches"] == 1
    assert ledger["target_records_unsealed"] == 3
    assert ledger["post_unseal_generation_count"] == 0
    assert ledger["post_unseal_tuning_events"] == 0
    assert [row["event"] for row in ledger["events"]] == ["generated"] * 21 + ["targets_unsealed"]


def test_candidates_are_structured_expressions_with_exact_terminal_evidence(
    result: dict[str, object],
) -> None:
    for world in result["world_results"]:
        assert tuple(row["family"] for row in world["family_bindings"]) == FAMILIES
        assert set(world["native_generator_receipts"]) == set(FAMILIES)
        candidates = {row["artifact_id"]: row for row in world["candidates"]}
        for binding in world["family_bindings"]:
            representation = candidates[binding["candidate"]["artifact_id"]]["representation"]
            assert representation["schema"] == "exact_sympy_rational_expression_v1"
            assert representation["target_fields_read"] == []
            assert representation["float_atoms"] == 0
            assert representation["node_count"] <= 32
            assert representation["expression_class"] in {
                "cubic_polynomial_over_integers",
                "quadratic_over_quadratic_rational_function",
                "cubic_integer_index_closed_form",
            }
        for family, assessment in world["assessments"].items():
            assert family in FAMILIES
            if assessment["status"] == "pass":
                assert assessment["proof_certificate"] is not None
                assert assessment["counterexample"] is None
            elif assessment["status"] == "reject":
                assert assessment["proof_certificate"] is None
                assert assessment["counterexample"]["difference_nonzero"] is True
            else:
                assert assessment["status"] == "block"
        for evaluation in world["evaluations"].values():
            assert evaluation["counts"]["registered_steps"] == 2
            assert evaluation["gate_outcomes"][0]["status"] == "pass"
            assert evaluation["status"] in {"pass", "reject", "block"}


def test_reference_proofs_and_knowledge_holdouts_are_closed(
    result: dict[str, object],
) -> None:
    kinds = []
    for world in result["world_results"]:
        proof = world["reference_proof_certificate"]
        assert proof["content_sha256"]
        kinds.append(proof["certificate_kind"])
        graph = CandidateKnowledgeGraph.from_dict(world["holdout"]["graph"])
        cut = HoldoutCut.from_dict(world["holdout"]["cut"])
        cut.validate_against(graph)
        assert len(cut.visible_node_ids) == 1
        assert len(cut.forbidden_node_ids) == 2
    assert kinds == [
        "exact_rational_identity",
        "exact_rational_identity",
        "exact_first_order_induction",
    ]
    assert result["counts"]["reference_proof_certificates"] == 3


def test_result_is_honest_deterministic_and_committed(result: dict[str, object]) -> None:
    assert result["counts"] == {
        "worlds": 3,
        "generator_families": 7,
        "structured_candidates": 21,
        "candidate_passes": 0,
        "candidate_rejects": 21,
        "candidate_blocks": 0,
        "world_passes": 0,
        "world_rejects": 3,
        "world_blocks": 0,
        "reference_proof_certificates": 3,
        "exact_counterexamples": 21,
        "pareto_eligible_candidates": 0,
    }
    assert all(
        world["decision"] == "reject_fixed_budget_with_exact_counterexamples"
        and world["pareto"] is None
        and world["metric_receipts"] == []
        for world in result["world_results"]
    )
    assert result["claims"] == CLAIMS
    assert build_campaign(ROOT) == result
    validate_campaign(result, ROOT)
    assert json.loads((ROOT / OUTPUT_PATH).read_text(encoding="utf-8")) == result


def test_resealed_expression_counterexample_proof_and_claim_tampers_fail_closed(
    result: dict[str, object],
) -> None:
    expression = copy.deepcopy(result)
    expression["world_results"][0]["candidates"][0]["representation"]["target_fields_read"] = [
        "right"
    ]
    _reseal(expression)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_campaign(expression, ROOT)

    counterexample = copy.deepcopy(result)
    counterexample["world_results"][0]["assessments"]["bayesian"]["counterexample"][
        "difference_nonzero"
    ] = False
    _reseal(counterexample)
    with pytest.raises(ValueError, match="exact replay mismatch"):
        validate_campaign(counterexample, ROOT)

    promoted = copy.deepcopy(result)
    promoted["claims"]["promotion_authorized"] = True
    _reseal(promoted)
    with pytest.raises(ValueError, match="claim boundary changed"):
        validate_campaign(promoted, ROOT)
