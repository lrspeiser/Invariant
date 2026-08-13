from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.formula_discovery_job import (
    PROBLEM_SCHEMA,
    RESULT_SCHEMA,
    SYSTEM_CAPS,
    FormulaDiscoveryValidationError,
    run_formula_discovery_job,
    validate_formula_discovery_result,
)
from sigma_theory_compiler.sigma_core import CandidateArtifact, canonical_sha256


def _q(value: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": value, "denominator": denominator}


def _limits() -> dict[str, int]:
    return {
        "max_basis_terms": 8,
        "max_constraint_rows": 32,
        "max_expression_nodes": 128,
        "max_integer_bits": 128,
        "max_validation_rows": 16,
    }


def _polynomial_problem() -> dict[str, object]:
    def target(x: int) -> int:
        return x**4 + 2 * x**3 - x - 30

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "example.quartic",
        "variable": "x",
        "variable_domain": "rational",
        "solver": {
            "kind": "exact_linear_basis_v1",
            "basis": ["1", "x", "x**2", "x**3", "x**4"],
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [{"point": _q(x), "value": _q(target(x))} for x in range(5)],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(x), "value": _q(target(x))} for x in (-3, 7, 11)],
        },
        "proof": {"kind": "none"},
        "limits": _limits(),
    }


def _recurrence_problem() -> dict[str, object]:
    def target(n: int) -> int:
        return 2 * n**3 + 2 * n**2 + n + 7

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "example.recurrence",
        "variable": "n",
        "variable_domain": "integer",
        "solver": {
            "kind": "exact_linear_basis_v1",
            "basis": ["1", "n", "n**2", "n**3"],
        },
        "constraints": {
            "kind": "first_order_recurrence",
            "sequence": "u",
            "base": {"index": 0, "value": _q(7)},
            "successor_increment": "6*n**2 + 10*n + 5",
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(n), "value": _q(target(n))} for n in (1, 4, 9)],
        },
        "proof": {"kind": "induction"},
        "limits": _limits(),
    }


def test_polynomial_job_recovers_sigma_candidate_and_checks_disjoint_holdout() -> None:
    problem = _polynomial_problem()
    result = run_formula_discovery_job(problem)
    validate_formula_discovery_result(result, problem)
    assert result["schema_version"] == RESULT_SCHEMA
    assert result["decision"] == "PASS"
    assert result["synthesis"]["expression"] == "x**4 + 2*x**3 - x - 30"
    assert result["synthesis"]["rank"] == result["synthesis"]["column_count"] == 5
    assert result["synthesis"]["problem_sha256"] == canonical_sha256(problem)
    assert result["validation"]["status"] == "PASS"
    assert result["validation"]["checked_rows"] == 3
    assert result["validation"]["validation_input_sha256"] == canonical_sha256(
        problem["validation"]
    )
    assert result["proof_certificate"] is None
    candidate = CandidateArtifact.from_dict(result["candidate"])
    candidate.validate()
    assert candidate.kind.value == "formula"
    assert candidate.representation["problem_sha256"] == canonical_sha256(problem)
    assert candidate.provenance.domain_pack.pack_id == "formula.discovery.job"


def test_recurrence_job_recovers_closed_form_and_checks_induction() -> None:
    problem = _recurrence_problem()
    result = run_formula_discovery_job(problem)
    validate_formula_discovery_result(result, problem)
    assert result["decision"] == "PASS"
    assert result["synthesis"]["expression"] == "2*n**3 + 2*n**2 + n + 7"
    assert result["proof_certificate"]["decision"] == (
        "proved_by_base_and_symbolic_successor_identity"
    )
    assert result["counts"] == {
        "candidates_emitted": 1,
        "counterexamples_found": 0,
        "proof_certificates": 1,
        "synthesis_rows": 4,
        "validation_rows_checked": 3,
    }


def test_independent_holdout_returns_exact_counterexample() -> None:
    problem = _polynomial_problem()
    problem["validation"]["rows"][1]["value"] = _q(999)
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "REJECT"
    assert result["reason_codes"] == ["heldout_counterexample"]
    witness = result["validation"]["counterexample"]
    assert witness["row_index"] == 1
    assert witness["point"] == _q(7)
    assert witness["expected"] == _q(999)
    assert witness["observed"] == _q(3050)
    assert witness["residual"] == _q(2051)
    assert result["counts"]["counterexamples_found"] == 1


def test_underdetermined_constraints_block_with_exact_ranks() -> None:
    problem = _polynomial_problem()
    problem["constraints"]["rows"] = problem["constraints"]["rows"][:2]
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["underdetermined_exact_constraints"]
    assert result["synthesis"]["rank"] == 2
    assert result["synthesis"]["augmented_rank"] == 2
    assert result["candidate"] is None


def test_inconsistent_constraints_reject_with_exact_ranks() -> None:
    problem = _polynomial_problem()
    problem["constraints"]["rows"].append({"point": _q(0), "value": _q(-29)})
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "REJECT"
    assert result["reason_codes"] == ["inconsistent_exact_constraints"]
    assert result["synthesis"]["augmented_rank"] > result["synthesis"]["rank"]
    assert result["candidate"] is None


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (lambda problem: problem.update({"extra": True}), "malformed_problem"),
        (lambda problem: problem["solver"]["basis"].append({"not": "source"}), "malformed_problem"),
        (
            lambda problem: problem["solver"].update({"kind": "neural_oracle_v1"}),
            "unsupported_problem",
        ),
        (
            lambda problem: problem["limits"].update(
                {"max_basis_terms": SYSTEM_CAPS["max_basis_terms"] + 1}
            ),
            "budget_exceeded",
        ),
    ],
)
def test_malformed_unsupported_and_over_budget_jobs_fail_closed(mutation, reason: str) -> None:
    problem = _polynomial_problem()
    mutation(problem)
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "BLOCK"
    assert result["problem_schema_valid"] is False
    assert result["reason_codes"] == [reason]
    assert result["candidate"] is None
    validate_formula_discovery_result(result, problem)


def test_expression_parser_does_not_execute_calls() -> None:
    problem = _polynomial_problem()
    problem["solver"]["basis"][0] = "__import__('os').getcwd()"
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["unsupported_problem"]


def test_training_and_validation_overlap_is_malformed_not_evidence() -> None:
    problem = _polynomial_problem()
    problem["validation"]["rows"][0] = copy.deepcopy(problem["constraints"]["rows"][0])
    result = run_formula_discovery_job(problem)
    assert result["decision"] == "BLOCK"
    assert result["reason_codes"] == ["malformed_problem"]
    assert result["candidate"] is None


def test_result_is_deterministic_and_tamper_or_reseal_cannot_replay() -> None:
    problem = _recurrence_problem()
    first = run_formula_discovery_job(problem)
    second = run_formula_discovery_job(copy.deepcopy(problem))
    assert first == second

    tampered = copy.deepcopy(first)
    tampered["decision"] = "REJECT"
    with pytest.raises(FormulaDiscoveryValidationError, match="content hash changed"):
        validate_formula_discovery_result(tampered, problem)

    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(FormulaDiscoveryValidationError, match="exact replay changed"):
        validate_formula_discovery_result(tampered, problem)


def test_closed_result_schema_and_claim_boundary() -> None:
    result = run_formula_discovery_job(_polynomial_problem())
    assert set(result) == {
        "candidate",
        "claims",
        "content_sha256",
        "counts",
        "decision",
        "job_id",
        "problem_schema_valid",
        "problem_sha256",
        "proof_certificate",
        "reason_codes",
        "requested_limits",
        "schema_version",
        "scope",
        "synthesis",
        "system_caps",
        "validation",
    }
    assert result["claims"] == {
        "candidate_is_scientific_law": False,
        "novelty_established": False,
        "promotion_authorized": False,
        "synthesis_constraints_are_proof": False,
    }
