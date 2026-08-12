from __future__ import annotations

from dataclasses import replace

import pytest

from sigma_theory_compiler.math_expression_ir import Equation, Recurrence, call, literal, symbol
from sigma_theory_compiler.math_proof_strategy import (
    ProofAttemptStatus,
    ProofStrategy,
    ProofStrategyBoundaryError,
    ProofStrategyProposal,
    ProposalOrigin,
    deterministic_strategy_proposals,
    execute_strategy,
)
from sigma_theory_compiler.sigma_core import ArtifactRef, SourceBinding

ARTIFACT = ArtifactRef("sig-proof-strategy", "a" * 64)


def identity() -> Equation:
    n = symbol("n")
    return Equation(n * (n + 1), n**2 + n)


def recurrence() -> Recurrence:
    n = symbol("n")
    return Recurrence(
        "S",
        n,
        1,
        Equation(call("S", n + 1), call("S", n) + n + 1),
        ((0, literal(0)),),
    )


def test_deterministic_proposals_are_suggestions_not_proof() -> None:
    proposals = deterministic_strategy_proposals(ARTIFACT, identity())
    assert tuple(item.strategy for item in proposals) == (
        ProofStrategy.EXACT_ALGEBRA,
        ProofStrategy.FACTORIZATION,
    )
    assert all(item.origin is ProposalOrigin.DETERMINISTIC for item in proposals)
    assert all("status" not in item.to_dict() for item in proposals)


def test_exact_executor_proves_identity_with_bound_certificate() -> None:
    proposal = deterministic_strategy_proposals(ARTIFACT, identity())[0]
    attempt = execute_strategy(proposal, identity())
    assert attempt.status is ProofAttemptStatus.PROVED
    assert attempt.certificate_sha256 is not None
    assert attempt.reason_codes == ()


def test_exact_executor_refutes_false_identity() -> None:
    n = symbol("n")
    false = Equation(n * (n + 1), n**2)
    proposal = ProofStrategyProposal.create(
        "false-exact",
        ARTIFACT,
        ProofStrategy.EXACT_ALGEBRA,
        ProposalOrigin.HUMAN,
        {"declared": True},
    )
    attempt = execute_strategy(proposal, false)
    assert attempt.status is ProofAttemptStatus.REFUTED
    assert attempt.reason_codes == ("exact_identity_failed",)


def test_induction_executor_proves_natural_sum_closed_form() -> None:
    n = symbol("n")
    proposal = deterministic_strategy_proposals(ARTIFACT, recurrence())[0]
    assert proposal.strategy is ProofStrategy.INDUCTION
    statement = Equation(call("S", n), n * (n + 1) / 2)
    attempt = execute_strategy(proposal, statement, recurrence=recurrence(), base_index=0)
    assert attempt.status is ProofAttemptStatus.PROVED
    assert attempt.certificate_sha256 is not None


def test_incomplete_induction_and_unregistered_executor_block() -> None:
    induction = deterministic_strategy_proposals(ARTIFACT, recurrence())[0]
    n = symbol("n")
    statement = Equation(call("S", n), n * (n + 1) / 2)
    assert execute_strategy(induction, statement).status is ProofAttemptStatus.BLOCKED
    factorization = deterministic_strategy_proposals(ARTIFACT, identity())[1]
    attempt = execute_strategy(factorization, identity())
    assert attempt.status is ProofAttemptStatus.BLOCKED
    assert attempt.reason_codes == ("strategy_executor_not_registered",)


def test_llm_proposal_requires_hashed_source_and_cannot_self_promote() -> None:
    with pytest.raises(ProofStrategyBoundaryError, match="source"):
        ProofStrategyProposal.create(
            "llm-induction",
            ARTIFACT,
            ProofStrategy.INDUCTION,
            ProposalOrigin.LLM,
            {"temperature": "0"},
        )
    proposal = ProofStrategyProposal.create(
        "llm-induction",
        ARTIFACT,
        ProofStrategy.INDUCTION,
        ProposalOrigin.LLM,
        {"temperature": "0"},
        source=SourceBinding("proposal_receipt", "evidence/llm-proposal.json", "b" * 64),
    )
    assert "status" not in proposal.to_dict()


def test_resealed_proposal_and_attempt_tampering_rejects() -> None:
    proposal = deterministic_strategy_proposals(ARTIFACT, identity())[0]
    with pytest.raises(ProofStrategyBoundaryError, match="canonical"):
        replace(proposal, parameters_sha256="c" * 64)
    attempt = execute_strategy(proposal, identity())
    with pytest.raises(ProofStrategyBoundaryError, match="canonical"):
        replace(attempt, executor_id="forged-executor")


def test_closed_strategy_enum_contains_requested_search_families() -> None:
    assert {item.value for item in ProofStrategy} == {
        "exact_algebra",
        "induction",
        "contradiction",
        "substitution",
        "factorization",
        "generating_function",
        "bijection",
        "invariant",
        "symmetry",
        "extremal",
        "descent",
        "change_of_variables",
    }
