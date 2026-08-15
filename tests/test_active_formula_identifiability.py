from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.active_formula_identifiability import (
    ActiveIdentifiabilityError,
    build_query_answer,
    resume_active_identifiability,
    run_active_identifiability,
    target_commitment,
    validate_initial_result,
    validate_proposed_query,
    validate_resumed_result,
)
from sigma_theory_compiler.sigma_core import CandidateArtifact, canonical_sha256

NONCE = "target-isolated-nonce-001"


def _q(value: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": value, "denominator": denominator}


def _limits() -> dict[str, int]:
    return {
        "max_expression_nodes": 64,
        "max_hypotheses": 8,
        "max_integer_bits": 64,
        "max_observations": 8,
        "max_query_space": 8,
    }


def _problem(*, query_budget: int = 2, query_space: list[int] | None = None) -> dict[str, object]:
    session_id = "active.example"
    return {
        "schema_version": "sigma-active-formula-identifiability-problem-1.0",
        "session_id": session_id,
        "variable": "x",
        "variable_domain": "integer",
        "hypotheses": [
            {"hypothesis_id": "formula.linear", "expression": "x"},
            {
                "hypothesis_id": "formula.quadratic",
                "expression": "x + x*(x - 1)",
            },
        ],
        "observations": [{"point": _q(0), "value": _q(0)}],
        "query_space": [_q(value) for value in (query_space or [1, 2])],
        "query_budget": query_budget,
        "target_commitment": target_commitment(session_id, "formula.quadratic", NONCE),
        "limits": _limits(),
    }


def test_initial_ambiguous_data_blocks_with_exact_witness_and_separating_query() -> None:
    problem = _problem()
    result = run_active_identifiability(problem)
    validate_initial_result(result, problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["ambiguous_public_data_query_proposed"]
    assert result["surviving_hypothesis_ids"] == ["formula.linear", "formula.quadratic"]
    witness = result["ambiguity_witness"]
    assert witness["equivalence_class_size"] == 2
    assert witness["indistinguishable_pair"] == ["formula.linear", "formula.quadratic"]
    assert witness["all_survivors_match_every_public_observation"] is True
    query = result["query"]
    assert query["point"] == _q(2)
    assert query["partition_count"] == 2
    assert query["worst_case_remaining"] == 1
    assert query["prediction_partition"] == [
        {"value": _q(2), "hypothesis_ids": ["formula.linear"]},
        {"value": _q(4), "hypothesis_ids": ["formula.quadratic"]},
    ]
    assert result["claims"]["target_opened_during_initial_query_selection"] is False


def test_preregistered_target_isolated_answer_resumes_to_exact_candidate_and_proof() -> None:
    problem = _problem()
    initial = run_active_identifiability(problem)
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.quadratic",
        nonce=NONCE,
    )
    assert answer["value"] == _q(4)
    resumed = resume_active_identifiability(problem, initial, answer)
    validate_resumed_result(resumed, problem, initial, answer)
    assert resumed["decision"] == "PASS"
    assert resumed["surviving_hypothesis_ids"] == ["formula.quadratic"]
    candidate = CandidateArtifact.from_dict(resumed["candidate"])
    candidate.validate()
    assert candidate.representation["hypothesis_id"] == "formula.quadratic"
    assert candidate.representation["expression"] == "x**2"
    assert candidate.representation["identification_evidence_sha256"] == initial["content_sha256"]
    proof = resumed["proof_certificate"]
    assert proof["decision"] == "proved_unique_within_declared_family_after_bound_answer"
    assert proof["remaining_equivalence_class_size"] == 1
    assert len(proof["exact_checks"]) == 2
    assert all(row["residual_zero"] for row in proof["exact_checks"])


def test_illegal_repeated_and_uninformative_queries_are_rejected() -> None:
    problem = _problem(query_budget=3, query_space=[0, 1, 2])
    initial = run_active_identifiability(problem)
    with pytest.raises(ActiveIdentifiabilityError, match="outside the legal"):
        validate_proposed_query(problem, initial, _q(3))
    with pytest.raises(ActiveIdentifiabilityError, match="repeats a public"):
        validate_proposed_query(problem, initial, _q(0))
    with pytest.raises(ActiveIdentifiabilityError, match="does not separate"):
        validate_proposed_query(problem, initial, _q(1))
    validate_proposed_query(problem, initial, _q(2))


def test_answer_hash_provenance_opening_and_prediction_tamper_are_rejected() -> None:
    problem = _problem()
    initial = run_active_identifiability(problem)
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.quadratic",
        nonce=NONCE,
    )

    tampered = copy.deepcopy(answer)
    tampered["value"] = _q(2)
    with pytest.raises(ActiveIdentifiabilityError, match="answer content hash changed"):
        resume_active_identifiability(problem, initial, tampered)

    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ActiveIdentifiabilityError, match="opened target prediction"):
        resume_active_identifiability(problem, initial, tampered)

    wrong_opening = copy.deepcopy(answer)
    wrong_opening["opening"]["nonce"] = "different-preregistered-nonce"
    wrong_opening["content_sha256"] = canonical_sha256(
        {key: value for key, value in wrong_opening.items() if key != "content_sha256"}
    )
    with pytest.raises(ActiveIdentifiabilityError, match="commitment opening mismatch"):
        resume_active_identifiability(problem, initial, wrong_opening)


def test_candidate_and_resealed_result_tamper_fail_replay() -> None:
    problem = _problem()
    initial = run_active_identifiability(problem)
    answer = build_query_answer(
        problem,
        initial,
        target_hypothesis_id="formula.quadratic",
        nonce=NONCE,
    )
    resumed = resume_active_identifiability(problem, initial, answer)
    tampered = copy.deepcopy(resumed)
    tampered["candidate"]["representation"]["expression"] = "x"
    tampered["candidate"]["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered["candidate"].items() if key != "content_sha256"}
    )
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(ActiveIdentifiabilityError):
        validate_resumed_result(tampered, problem, initial, answer)


def test_initial_and_resumed_runs_are_deterministic() -> None:
    problem = _problem()
    first = run_active_identifiability(problem)
    second = run_active_identifiability(copy.deepcopy(problem))
    assert first == second
    answer = build_query_answer(
        problem,
        first,
        target_hypothesis_id="formula.quadratic",
        nonce=NONCE,
    )
    assert resume_active_identifiability(problem, first, answer) == resume_active_identifiability(
        copy.deepcopy(problem), copy.deepcopy(first), copy.deepcopy(answer)
    )


def test_query_search_budget_stops_before_unopened_informative_query() -> None:
    problem = _problem(query_budget=1)
    result = run_active_identifiability(problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["query_search_budget_exhausted"]
    assert result["query"] is None
    assert result["counts"]["query_points_evaluated"] == 1
    assert result["counts"]["informative_queries"] == 0


def test_no_identifiable_query_is_an_honest_terminal_block() -> None:
    problem = _problem(query_budget=1, query_space=[1])
    result = run_active_identifiability(problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["no_identifiable_legal_query"]
    assert result["ambiguity_witness"]["equivalence_class_size"] == 2
    assert result["query"] is None


def test_malformed_unsafe_duplicate_and_over_budget_problems_fail_closed() -> None:
    malformed = _problem()
    malformed["extra"] = True
    assert run_active_identifiability(malformed)["reason_codes"] == ["malformed_problem"]

    unsafe = _problem()
    unsafe["hypotheses"][0]["expression"] = "__import__('os').getcwd()"
    assert run_active_identifiability(unsafe)["reason_codes"] == ["unsupported_problem"]

    duplicate = _problem()
    duplicate["hypotheses"][1]["expression"] = "x + 0"
    assert run_active_identifiability(duplicate)["reason_codes"] == [
        "duplicate_equivalent_hypotheses"
    ]

    budget = _problem()
    budget["limits"]["max_hypotheses"] = 1
    assert run_active_identifiability(budget)["reason_codes"] == ["malformed_problem"]
