from __future__ import annotations

import copy

import pytest

from sigma_theory_compiler.formula_discovery_job import PROBLEM_SCHEMA
from sigma_theory_compiler.formula_discovery_orchestration import (
    BATCH_SCHEMA,
    CLAIMS,
    FormulaDiscoveryOrchestrationError,
    build_formula_discovery_orchestration,
    validate_formula_discovery_orchestration,
)
from sigma_theory_compiler.sigma_core import CandidateArtifact, canonical_sha256


def _q(value: int) -> dict[str, int]:
    return {"numerator": value, "denominator": 1}


def _problem(job_id: str, *, wrong_holdout: bool = False) -> dict[str, object]:
    validation = 5 if wrong_holdout else 7
    return {
        "schema_version": PROBLEM_SCHEMA,
        "job_id": job_id,
        "variable": "x",
        "variable_domain": "rational",
        "solver": {"kind": "exact_linear_basis_v1", "basis": ["1", "x"]},
        "constraints": {
            "kind": "evaluations",
            "rows": [
                {"point": _q(0), "value": _q(1)},
                {"point": _q(1), "value": _q(3)},
            ],
        },
        "validation": {
            "kind": "evaluations",
            "rows": [{"point": _q(3), "value": _q(validation)}],
        },
        "proof": {"kind": "none"},
        "limits": {
            "max_basis_terms": 4,
            "max_constraint_rows": 8,
            "max_expression_nodes": 32,
            "max_integer_bits": 64,
            "max_validation_rows": 4,
        },
    }


@pytest.fixture(scope="module")
def problems() -> tuple[dict[str, object], dict[str, object]]:
    return (_problem("caller.pass"), _problem("caller.reject", wrong_holdout=True))


@pytest.fixture(scope="module")
def report(problems) -> dict[str, object]:
    value = build_formula_discovery_orchestration(problems, batch_id="test.formula.batch")
    validate_formula_discovery_orchestration(value, problems)
    return value


def test_two_caller_jobs_create_sigma_candidates_and_bound_outcomes(report) -> None:
    assert report["schema_version"] == BATCH_SCHEMA
    assert report["counts"] == {
        "caller_jobs": 2,
        "job_passes": 1,
        "job_rejects": 1,
        "job_blocks": 0,
        "candidates": 2,
        "evaluation_passes": 1,
        "evaluation_rejects": 1,
        "evaluation_blocks": 0,
        "hard_gate_eligible": 1,
        "pareto_fronts": 1,
    }
    assert [row["result"]["decision"] for row in report["jobs"]] == ["PASS", "REJECT"]
    for row in report["candidates"]:
        candidate = CandidateArtifact.from_dict(row)
        candidate.validate()
        assert len(candidate.provenance.inputs) == 1
        assert candidate.representation["source_candidate"] == (
            candidate.provenance.inputs[0].to_dict()
        )
    assert [row["status"] for row in report["evaluations"]] == ["pass", "reject"]
    assert all(len(row["gate_outcomes"]) == 2 for row in report["evaluations"])


def test_only_all_hard_gate_pass_candidate_enters_pareto(report) -> None:
    explanations = report["pareto"]["explanations"]
    eligible = [row for row in explanations if row["soft_metric_eligible"]]
    excluded = [row for row in explanations if not row["soft_metric_eligible"]]
    assert len(eligible) == len(excluded) == 1
    assert eligible[0]["pareto_front"] == 1
    assert excluded[0]["pareto_front"] is None
    assert [row["status"] for row in excluded[0]["hard_gate_outcomes"]] == [
        "pass",
        "reject",
    ]
    assert report["pareto"]["pareto_fronts"] == [[eligible[0]["candidate"]]]
    assert report["claims"] == CLAIMS


def test_metrics_are_exact_complete_and_bound_to_discovery_results(report) -> None:
    receipts = report["metric_receipts"]
    assert len(receipts) == 4
    assert {row["metric_id"] for row in receipts} == {
        "coefficient_count",
        "expression_bytes",
    }
    assert all(row["value"]["denominator"] == 1 for row in receipts)
    assert all(row["evidence_sha256"] for row in receipts)


def test_input_order_is_canonical_and_report_replays(problems) -> None:
    first = build_formula_discovery_orchestration(problems, batch_id="test.formula.batch")
    reversed_report = build_formula_discovery_orchestration(
        tuple(reversed(problems)), batch_id="test.formula.batch"
    )
    assert first == reversed_report
    validate_formula_discovery_orchestration(first, problems)


def test_resealed_semantic_tamper_fails_exact_replay(report, problems) -> None:
    tampered = copy.deepcopy(report)
    tampered["claims"]["promotion_authorized"] = True
    tampered["content_sha256"] = canonical_sha256(
        {key: value for key, value in tampered.items() if key != "content_sha256"}
    )
    with pytest.raises(FormulaDiscoveryOrchestrationError, match="exact replay changed"):
        validate_formula_discovery_orchestration(tampered, problems)


def test_duplicate_ids_and_candidate_free_batch_fail_closed() -> None:
    duplicate = (_problem("same.job"), _problem("same.job", wrong_holdout=True))
    with pytest.raises(FormulaDiscoveryOrchestrationError, match="unique and bound"):
        build_formula_discovery_orchestration(duplicate)

    blocked = _problem("caller.blocked")
    blocked["constraints"]["rows"] = blocked["constraints"]["rows"][:1]
    with pytest.raises(FormulaDiscoveryOrchestrationError, match="no caller job emitted"):
        build_formula_discovery_orchestration((blocked,))
