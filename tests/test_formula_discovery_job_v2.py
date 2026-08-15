from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from sigma_theory_compiler.formula_discovery_job_v2 import (
    PROBLEM_SCHEMA,
    RESULT_SCHEMA,
    FormulaDiscoveryV2ValidationError,
    run_formula_discovery_job_v2,
    validate_formula_discovery_result_v2,
)
from sigma_theory_compiler.sigma_core import CandidateArtifact, canonical_sha256


def _q(value: int, denominator: int = 1) -> dict[str, int]:
    return {"numerator": value, "denominator": denominator}


def _ratio(numerator: int, denominator: int) -> dict[str, int]:
    divisor = abs(math.gcd(numerator, denominator))
    numerator, denominator = numerator // divisor, denominator // divisor
    if denominator < 0:
        numerator, denominator = -numerator, -denominator
    return _q(numerator, denominator)


def _limits() -> dict[str, int]:
    return {
        "max_basis_terms": 12,
        "max_constraint_rows": 32,
        "max_expression_nodes": 128,
        "max_integer_bits": 128,
        "max_parameter_combinations": 128,
        "max_parameters": 4,
        "max_recurrence_order": 4,
        "max_validation_rows": 16,
        "max_variables": 4,
    }


def _point(**values: int) -> dict[str, dict[str, int]]:
    return {name: _q(value) for name, value in values.items()}


def _multivariate_problem() -> dict[str, object]:
    def target(x: int, y: int) -> int:
        return x * x + 2 * x * y + 3 * y + 1

    training = ((0, 0), (1, 0), (0, 1), (1, 1), (2, -1))
    holdout = ((-2, 3), (4, 2), (3, -4))
    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "v2.multivariate",
        "variables": [
            {"name": "x", "domain": "rational"},
            {"name": "y", "domain": "rational"},
        ],
        "premises": [],
        "solver": {
            "kind": "exact_polynomial_basis_v2",
            "basis": ["1", "x", "y", "x**2", "x*y"],
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [{"point": _point(x=x, y=y), "value": _q(target(x, y))} for x, y in training],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _point(x=x, y=y), "value": _q(target(x, y))} for x, y in holdout],
        },
        "proof": {"kind": "none"},
        "limits": _limits(),
    }


def _rational_problem() -> dict[str, object]:
    def target(x: int) -> tuple[int, int]:
        return x + 1, x - 1

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "v2.rational",
        "variables": [{"name": "x", "domain": "rational"}],
        "premises": [{"kind": "nonzero", "expression": "x + 2"}],
        "solver": {
            "kind": "exact_rational_basis_v2",
            "numerator_basis": ["1", "x"],
            "denominator_basis": ["1", "x"],
            "denominator_anchor": 1,
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [{"point": _point(x=x), "value": _ratio(*target(x))} for x in (0, 2, 3)],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _point(x=x), "value": _ratio(*target(x))} for x in (4, 5, -3)],
        },
        "proof": {"kind": "none"},
        "limits": _limits(),
    }


def _nonlinear_problem() -> dict[str, object]:
    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "v2.nonlinear",
        "variables": [{"name": "x", "domain": "rational"}],
        "premises": [],
        "solver": {
            "kind": "exact_nonlinear_parameter_grid_v2",
            "expression": "a**2*x + b",
            "parameters": [
                {"name": "a", "values": [_q(-2), _q(-1), _q(1), _q(2)]},
                {"name": "b", "values": [_q(2), _q(3), _q(4)]},
            ],
            "parameter_relations": ["a**2 - 4"],
        },
        "constraints": {
            "kind": "evaluations",
            "rows": [
                {"point": _point(x=0), "value": _q(3)},
                {"point": _point(x=1), "value": _q(7)},
            ],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [
                {"point": _point(x=-3), "value": _q(-9)},
                {"point": _point(x=5), "value": _q(23)},
            ],
        },
        "proof": {"kind": "none"},
        "limits": _limits(),
    }


def _recurrence_problem() -> dict[str, object]:
    def target(n: int) -> int:
        return n * n + 1

    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": "v2.recurrence.order2",
        "variables": [{"name": "n", "domain": "integer"}],
        "premises": [],
        "solver": {
            "kind": "exact_polynomial_basis_v2",
            "basis": ["1", "n", "n**2"],
        },
        "constraints": {
            "kind": "linear_recurrence",
            "sequence": "u",
            "order": 2,
            "coefficients": ["-1", "2"],
            "forcing": "2",
            "initial_values": [
                {"index": 0, "value": _q(1)},
                {"index": 1, "value": _q(2)},
            ],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _point(n=n), "value": _q(target(n))} for n in (3, 5, 11)],
        },
        "proof": {"kind": "recurrence_identity"},
        "limits": _limits(),
    }


@pytest.mark.parametrize(
    ("problem_factory", "class_id", "expression"),
    [
        (_multivariate_problem, "multivariate_polynomial", "x**2 + 2*x*y + 3*y + 1"),
        (_rational_problem, "rational_function_with_domain", "(x + 1)/(x - 1)"),
        (_nonlinear_problem, "nonlinear_algebraic_parameterization", "4*x + 3"),
        (_recurrence_problem, "higher_order_recurrence", "n**2 + 1"),
    ],
)
def test_v2_class_adapter_passes_exact_holdout_and_replays(
    problem_factory, class_id: str, expression: str
) -> None:
    problem = problem_factory()
    result = run_formula_discovery_job_v2(problem)
    validate_formula_discovery_result_v2(result, problem)
    assert result["schema_version"] == RESULT_SCHEMA
    assert result["decision"] == "PASS"
    assert result["class_id"] == class_id
    assert result["synthesis"]["expression"] == expression
    assert result["validation"]["status"] == "PASS"
    candidate = CandidateArtifact.from_dict(result["candidate"])
    candidate.validate()
    assert candidate.provenance.domain_pack.pack_version == "2.0.0"
    assert candidate.representation["class_id"] == class_id


def test_rational_candidate_carries_generated_nonzero_domain_premise() -> None:
    result = run_formula_discovery_job_v2(_rational_problem())
    representation = result["candidate"]["representation"]
    assert representation["caller_domain_premises"] == [{"kind": "nonzero", "expression": "x + 2"}]
    assert representation["generated_domain_premises"] == [
        {"kind": "nonzero", "expression": "x - 1"}
    ]
    assert result["synthesis"]["details"]["generated_domain_premise"] == "x - 1 != 0"


def test_nonlinear_implicit_relation_preserves_equivalent_parameter_assignments() -> None:
    result = run_formula_discovery_job_v2(_nonlinear_problem())
    details = result["synthesis"]["details"]
    assert details["parameter_combinations"] == 12
    assert details["relation_admissible_assignments"] == 6
    assert details["constraint_matching_assignments"] == 2
    assert details["distinct_matching_expressions"] == 1
    assert details["parameter_relations"] == ["a**2 - 4"]


def test_higher_order_recurrence_emits_exact_certificate() -> None:
    result = run_formula_discovery_job_v2(_recurrence_problem())
    proof = result["proof_certificate"]
    assert proof["decision"] == "proved_exact_higher_order_recurrence_identity"
    assert proof["order"] == 2
    assert proof["recurrence_residual_numerator"] == "0"
    assert all(row["residual_zero"] for row in proof["initial_checks"])


def test_independent_validation_rejects_with_multivariate_counterexample() -> None:
    problem = _multivariate_problem()
    problem["validation"]["rows"][1]["value"] = _q(999)
    result = run_formula_discovery_job_v2(problem)
    assert result["decision"] == "REJECT"
    assert result["reason_codes"] == ["heldout_counterexample"]
    witness = result["validation"]["counterexample"]
    assert witness["point"] == {"x": _q(4), "y": _q(2)}
    assert witness["observed"] == _q(39)
    assert witness["residual"] == _q(-960)


def test_rational_undefined_holdout_is_an_exact_reject_not_a_pass() -> None:
    problem = _rational_problem()
    problem["validation"]["rows"][0] = {"point": _point(x=1), "value": _q(0)}
    result = run_formula_discovery_job_v2(problem)
    assert result["decision"] == "REJECT"
    assert result["validation"]["counterexample"]["reason"] == (
        "candidate_undefined_or_nonrational"
    )


def test_grid_zero_match_rejects_and_multiple_expressions_block() -> None:
    inconsistent = _nonlinear_problem()
    inconsistent["constraints"]["rows"][0]["value"] = _q(99)
    rejected = run_formula_discovery_job_v2(inconsistent)
    assert rejected["decision"] == "REJECT"
    assert rejected["reason_codes"] == ["inconsistent_exact_parameter_grid"]

    ambiguous = _nonlinear_problem()
    ambiguous["solver"]["parameter_relations"] = []
    ambiguous["constraints"]["rows"] = [{"point": _point(x=0), "value": _q(3)}]
    blocked = run_formula_discovery_job_v2(ambiguous)
    assert blocked["decision"] == "BLOCK"
    assert blocked["reason_codes"] == ["underdetermined_exact_parameter_grid"]


def test_closed_schema_budget_and_unsafe_expression_fail_closed() -> None:
    malformed = _multivariate_problem()
    malformed["unknown"] = True
    assert run_formula_discovery_job_v2(malformed)["reason_codes"] == ["malformed_problem"]

    unsafe = _multivariate_problem()
    unsafe["solver"]["basis"][0] = "__import__('os').getcwd()"
    assert run_formula_discovery_job_v2(unsafe)["reason_codes"] == ["unsupported_problem"]

    over_budget = _nonlinear_problem()
    over_budget["limits"]["max_parameter_combinations"] = 2
    assert run_formula_discovery_job_v2(over_budget)["reason_codes"] == ["budget_exceeded"]


def test_v2_is_deterministic_and_resealed_tamper_fails_replay() -> None:
    problem = _recurrence_problem()
    first = run_formula_discovery_job_v2(problem)
    assert first == run_formula_discovery_job_v2(copy.deepcopy(problem))

    tampered = copy.deepcopy(first)
    tampered["decision"] = "REJECT"
    with pytest.raises(FormulaDiscoveryV2ValidationError, match="content hash changed"):
        validate_formula_discovery_result_v2(tampered, problem)
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(FormulaDiscoveryV2ValidationError, match="exact replay changed"):
        validate_formula_discovery_result_v2(tampered, problem)


def test_all_public_v2_example_problems_pass_and_replay() -> None:
    paths = sorted(Path("examples/formula-discovery-v2").glob("*.json"))
    assert [path.name for path in paths] == [
        "higher-order-recurrence.json",
        "multivariate-polynomial.json",
        "nonlinear-implicit-grid.json",
        "rational-domain.json",
    ]
    for path in paths:
        problem = json.loads(path.read_text(encoding="utf-8"))
        result = run_formula_discovery_job_v2(problem)
        assert result["decision"] == "PASS", (path, result["reason_codes"])
        validate_formula_discovery_result_v2(result, problem)
