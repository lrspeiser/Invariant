"""Exact active identifiability for a bounded public formula hypothesis family."""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import sympy as sp

from .sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackDescriptor,
    ProvenanceRecord,
    SchemaViolation,
    StageDefinition,
    canonical_json_bytes,
    canonical_sha256,
)

PROBLEM_SCHEMA = "sigma-active-formula-identifiability-problem-1.0"
INITIAL_RESULT_SCHEMA = "sigma-active-formula-identifiability-initial-result-1.0"
AMBIGUITY_SCHEMA = "sigma-active-formula-ambiguity-witness-1.0"
QUERY_SCHEMA = "sigma-active-formula-separating-query-1.0"
ANSWER_SCHEMA = "sigma-active-formula-query-answer-1.0"
OPENING_SCHEMA = "sigma-active-formula-target-opening-1.0"
RESUMED_RESULT_SCHEMA = "sigma-active-formula-identifiability-resumed-result-1.0"
PROOF_SCHEMA = "sigma-active-formula-identification-proof-1.0"

SYSTEM_CAPS = {
    "max_expression_nodes": 256,
    "max_hypotheses": 64,
    "max_integer_bits": 256,
    "max_observations": 64,
    "max_problem_bytes": 65_536,
    "max_query_space": 128,
}

_PROBLEM_KEYS = {
    "hypotheses",
    "limits",
    "observations",
    "query_budget",
    "query_space",
    "schema_version",
    "session_id",
    "target_commitment",
    "variable",
    "variable_domain",
}
_LIMIT_KEYS = set(SYSTEM_CAPS) - {"max_problem_bytes"}
_INITIAL_KEYS = {
    "ambiguity_witness",
    "candidate",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "problem_schema_valid",
    "problem_sha256",
    "query",
    "reason_codes",
    "schema_version",
    "scope",
    "session_id",
    "surviving_hypothesis_ids",
    "system_caps",
}
_ANSWER_KEYS = {
    "ambiguity_result_sha256",
    "content_sha256",
    "opening",
    "problem_sha256",
    "query_content_sha256",
    "schema_version",
    "value",
}
_RESUMED_KEYS = {
    "answer_content_sha256",
    "candidate",
    "claims",
    "content_sha256",
    "counts",
    "decision",
    "initial_result_sha256",
    "problem_sha256",
    "proof_certificate",
    "reason_codes",
    "schema_version",
    "scope",
    "session_id",
    "surviving_hypothesis_ids",
}
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_HASH = re.compile(r"^[0-9a-f]{64}$")


class ActiveIdentifiabilityError(ValueError):
    """A problem, query answer, or sealed result crossed a closed boundary."""


class _ProblemError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _Hypothesis:
    hypothesis_id: str
    expression: sp.Expr
    expression_source: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class _ParsedProblem:
    value: Mapping[str, Any]
    problem_sha256: str
    variable: sp.Symbol
    hypotheses: tuple[_Hypothesis, ...]
    observations: tuple[tuple[sp.Rational, sp.Rational], ...]
    query_space: tuple[sp.Rational, ...]
    query_budget: int
    limits: Mapping[str, int]


def _exact_keys(value: Any, expected: set[str]) -> bool:
    return isinstance(value, Mapping) and set(value) == expected


def _seal(value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body["content_sha256"] = canonical_sha256(body)
    return body


def _rational(value: Any, *, max_bits: int) -> sp.Rational:
    if not _exact_keys(value, {"denominator", "numerator"}):
        raise _ProblemError("malformed_problem")
    numerator, denominator = value["numerator"], value["denominator"]
    if (
        isinstance(numerator, bool)
        or isinstance(denominator, bool)
        or not isinstance(numerator, int)
        or not isinstance(denominator, int)
        or denominator <= 0
        or math.gcd(numerator, denominator) != 1
        or max(abs(numerator).bit_length(), denominator.bit_length()) > max_bits
    ):
        raise _ProblemError("malformed_problem")
    return sp.Rational(numerator, denominator)


def _rational_data(value: sp.Rational) -> dict[str, int]:
    exact = sp.Rational(value)
    return {"numerator": int(exact.p), "denominator": int(exact.q)}


def _parse_expression(source: Any, variable: sp.Symbol, limits: Mapping[str, int]) -> sp.Expr:
    if not isinstance(source, str) or not source or len(source) > 512:
        raise _ProblemError("malformed_problem")
    try:
        root = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if sum(1 for _ in ast.walk(root)) > limits["max_expression_nodes"]:
        raise _ProblemError("budget_exceeded")

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise _ProblemError("unsupported_problem")
            if abs(node.value).bit_length() > limits["max_integer_bits"]:
                raise _ProblemError("budget_exceeded")
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            if node.id != str(variable):
                raise _ProblemError("malformed_problem")
            return variable
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = convert(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left, right = convert(node.left), convert(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if right == 0:
                    raise _ProblemError("malformed_problem")
                return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_Integer or abs(int(right)) > 16:
                    raise _ProblemError("unsupported_problem")
                return left ** int(right)
        raise _ProblemError("unsupported_problem")

    expression = sp.cancel(convert(root.body))
    if expression.atoms(sp.Float) or expression.free_symbols - {variable}:
        raise _ProblemError("malformed_problem")
    if sum(1 for _ in sp.preorder_traversal(expression)) > limits["max_expression_nodes"]:
        raise _ProblemError("budget_exceeded")
    return expression


def _parse_limits(value: Any) -> dict[str, int]:
    if not _exact_keys(value, _LIMIT_KEYS):
        raise _ProblemError("malformed_problem")
    result: dict[str, int] = {}
    for key in sorted(_LIMIT_KEYS):
        item = value[key]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
            or item > SYSTEM_CAPS[key]
        ):
            raise _ProblemError("budget_exceeded")
        result[key] = item
    return result


def _evaluate(expression: sp.Expr, variable: sp.Symbol, point: sp.Rational) -> sp.Rational | None:
    value = sp.cancel(expression.subs(variable, point))
    if value.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not value.is_Rational:
        return None
    return sp.Rational(value)


def _parse_problem(problem: Mapping[str, Any]) -> _ParsedProblem:
    try:
        payload = canonical_json_bytes(problem)
    except (SchemaViolation, TypeError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if len(payload) > SYSTEM_CAPS["max_problem_bytes"]:
        raise _ProblemError("budget_exceeded")
    if not _exact_keys(problem, _PROBLEM_KEYS) or problem.get("schema_version") != PROBLEM_SCHEMA:
        raise _ProblemError("malformed_problem")
    session_id, variable_name = problem["session_id"], problem["variable"]
    if (
        not isinstance(session_id, str)
        or _IDENTIFIER.fullmatch(session_id) is None
        or not isinstance(variable_name, str)
        or _SYMBOL.fullmatch(variable_name) is None
        or variable_name.startswith("_")
        or problem["variable_domain"] not in {"integer", "rational"}
        or not isinstance(problem["target_commitment"], str)
        or _HASH.fullmatch(problem["target_commitment"]) is None
    ):
        raise _ProblemError("malformed_problem")
    limits = _parse_limits(problem["limits"])
    variable = sp.Symbol(variable_name, integer=problem["variable_domain"] == "integer")
    hypothesis_rows = problem["hypotheses"]
    if (
        not isinstance(hypothesis_rows, list)
        or not 2 <= len(hypothesis_rows) <= limits["max_hypotheses"]
    ):
        raise _ProblemError("malformed_problem")
    hypotheses: list[_Hypothesis] = []
    ids: set[str] = set()
    for row in hypothesis_rows:
        if not _exact_keys(row, {"expression", "hypothesis_id"}):
            raise _ProblemError("malformed_problem")
        hypothesis_id = row["hypothesis_id"]
        if (
            not isinstance(hypothesis_id, str)
            or _IDENTIFIER.fullmatch(hypothesis_id) is None
            or hypothesis_id in ids
        ):
            raise _ProblemError("malformed_problem")
        ids.add(hypothesis_id)
        expression = _parse_expression(row["expression"], variable, limits)
        content = canonical_sha256(
            {
                "expression": sp.sstr(expression),
                "hypothesis_id": hypothesis_id,
                "schema_version": "sigma-active-formula-hypothesis-1.0",
            }
        )
        hypotheses.append(_Hypothesis(hypothesis_id, expression, row["expression"], content))
    for left_index, left in enumerate(hypotheses):
        for right in hypotheses[left_index + 1 :]:
            if sp.cancel(left.expression - right.expression) == 0:
                raise _ProblemError("duplicate_equivalent_hypotheses")
    observations_value = problem["observations"]
    if (
        not isinstance(observations_value, list)
        or not 1 <= len(observations_value) <= limits["max_observations"]
    ):
        raise _ProblemError("malformed_problem")
    observations: list[tuple[sp.Rational, sp.Rational]] = []
    observed_points: set[sp.Rational] = set()
    for row in observations_value:
        if not _exact_keys(row, {"point", "value"}):
            raise _ProblemError("malformed_problem")
        point = _rational(row["point"], max_bits=limits["max_integer_bits"])
        value = _rational(row["value"], max_bits=limits["max_integer_bits"])
        if point in observed_points:
            raise _ProblemError("malformed_problem")
        if problem["variable_domain"] == "integer" and point.q != 1:
            raise _ProblemError("malformed_problem")
        observed_points.add(point)
        observations.append((point, value))
    query_value = problem["query_space"]
    if not isinstance(query_value, list) or not 1 <= len(query_value) <= limits["max_query_space"]:
        raise _ProblemError("malformed_problem")
    query_space = tuple(
        _rational(item, max_bits=limits["max_integer_bits"]) for item in query_value
    )
    if len(set(query_space)) != len(query_space):
        raise _ProblemError("malformed_problem")
    if problem["variable_domain"] == "integer" and any(point.q != 1 for point in query_space):
        raise _ProblemError("malformed_problem")
    query_budget = problem["query_budget"]
    if (
        isinstance(query_budget, bool)
        or not isinstance(query_budget, int)
        or not 1 <= query_budget <= len(query_space)
    ):
        raise _ProblemError("budget_exceeded")
    return _ParsedProblem(
        problem,
        canonical_sha256(problem),
        variable,
        tuple(hypotheses),
        tuple(observations),
        query_space,
        query_budget,
        limits,
    )


def target_commitment(session_id: str, target_hypothesis_id: str, nonce: str) -> str:
    """Create the pre-registration hash later opened by a query answer."""

    if (
        not isinstance(session_id, str)
        or _IDENTIFIER.fullmatch(session_id) is None
        or not isinstance(target_hypothesis_id, str)
        or _IDENTIFIER.fullmatch(target_hypothesis_id) is None
        or not isinstance(nonce, str)
        or not 8 <= len(nonce) <= 128
    ):
        raise ActiveIdentifiabilityError("target opening fields are not canonical")
    return canonical_sha256(
        {
            "schema_version": OPENING_SCHEMA,
            "session_id": session_id,
            "target_hypothesis_id": target_hypothesis_id,
            "nonce": nonce,
        }
    )


def _survivors(parsed: _ParsedProblem) -> tuple[_Hypothesis, ...]:
    return tuple(
        hypothesis
        for hypothesis in parsed.hypotheses
        if all(
            _evaluate(hypothesis.expression, parsed.variable, point) == value
            for point, value in parsed.observations
        )
    )


def _ambiguity_witness(parsed: _ParsedProblem, survivors: Sequence[_Hypothesis]) -> dict[str, Any]:
    signatures = []
    for hypothesis in survivors:
        signatures.append(
            {
                "hypothesis_id": hypothesis.hypothesis_id,
                "predictions": [
                    {
                        "point": _rational_data(point),
                        "value": _rational_data(value),
                    }
                    for point, value in parsed.observations
                ],
            }
        )
    return _seal(
        {
            "schema_version": AMBIGUITY_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "equivalence_class": [item.hypothesis_id for item in survivors],
            "equivalence_class_size": len(survivors),
            "indistinguishable_pair": [
                survivors[0].hypothesis_id,
                survivors[1].hypothesis_id,
            ],
            "public_observation_signatures": signatures,
            "all_survivors_match_every_public_observation": True,
        }
    )


def _partition_at(
    parsed: _ParsedProblem,
    survivors: Sequence[_Hypothesis],
    point: sp.Rational,
) -> dict[sp.Rational, list[str]] | None:
    partition: dict[sp.Rational, list[str]] = {}
    for hypothesis in survivors:
        value = _evaluate(hypothesis.expression, parsed.variable, point)
        if value is None:
            return None
        partition.setdefault(value, []).append(hypothesis.hypothesis_id)
    return partition


def validate_proposed_query(
    problem: Mapping[str, Any], initial_result: Mapping[str, Any], point: Mapping[str, Any]
) -> None:
    """Reject a caller query unless it is legal, new, and exactly separating."""

    validate_initial_result(initial_result, problem)
    parsed = _parse_problem(problem)
    query_point = _rational(point, max_bits=parsed.limits["max_integer_bits"])
    if query_point not in parsed.query_space:
        raise ActiveIdentifiabilityError("query point is outside the legal query space")
    if query_point in {row[0] for row in parsed.observations}:
        raise ActiveIdentifiabilityError("query point repeats a public observation")
    by_id = {item.hypothesis_id: item for item in parsed.hypotheses}
    survivors = [by_id[item] for item in initial_result["surviving_hypothesis_ids"]]
    partition = _partition_at(parsed, survivors, query_point)
    if partition is None or len(partition) < 2:
        raise ActiveIdentifiabilityError("query does not separate the surviving equivalence class")


def _select_query(
    parsed: _ParsedProblem,
    survivors: Sequence[_Hypothesis],
    ambiguity_sha256: str,
) -> tuple[dict[str, Any] | None, dict[str, int]]:
    observed = {row[0] for row in parsed.observations}
    candidates: list[
        tuple[tuple[int, int, int, int], sp.Rational, dict[sp.Rational, list[str]]]
    ] = []
    evaluated = 0
    legal = 0
    informative = 0
    for order, point in enumerate(parsed.query_space[: parsed.query_budget]):
        evaluated += 1
        if point in observed:
            continue
        partition = _partition_at(parsed, survivors, point)
        if partition is None:
            continue
        legal += 1
        if len(partition) < 2:
            continue
        informative += 1
        worst = max(len(group) for group in partition.values())
        score = (worst, -len(partition), order, int(point.p) * 1_000_003 + int(point.q))
        candidates.append((score, point, partition))
    counts = {
        "query_points_evaluated": evaluated,
        "legal_defined_queries": legal,
        "informative_queries": informative,
    }
    if not candidates:
        return None, counts
    _, point, partition = min(candidates, key=lambda item: item[0])
    groups = [
        {
            "value": _rational_data(value),
            "hypothesis_ids": sorted(ids),
        }
        for value, ids in sorted(partition.items(), key=lambda item: (item[0].p, item[0].q))
    ]
    left = groups[0]
    right = groups[1]
    query = _seal(
        {
            "schema_version": QUERY_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "ambiguity_witness_sha256": ambiguity_sha256,
            "point": _rational_data(point),
            "prediction_partition": groups,
            "partition_count": len(groups),
            "worst_case_remaining": max(len(group["hypothesis_ids"]) for group in groups),
            "separating_pair": {
                "left_hypothesis_id": left["hypothesis_ids"][0],
                "left_value": left["value"],
                "right_hypothesis_id": right["hypothesis_ids"][0],
                "right_value": right["value"],
            },
            "selection_policy": (
                "minimize_worst_case_then_maximize_partitions_then_caller_query_order"
            ),
        }
    )
    return query, counts


def _descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        "formula.active-identifiability",
        "1.0.0",
        (ArtifactKind.FORMULA,),
        (StageDefinition("identified", 0, (ArtifactKind.FORMULA,)),),
        (),
    )


def _candidate(
    parsed: _ParsedProblem,
    hypothesis: _Hypothesis,
    *,
    identification_evidence_sha256: str,
    answer_sha256: str | None,
) -> CandidateArtifact:
    provenance = ProvenanceRecord.create(
        _descriptor().ref,
        {
            "answer_sha256": answer_sha256,
            "hypothesis_content_sha256": hypothesis.content_sha256,
            "identification_evidence_sha256": identification_evidence_sha256,
            "problem_sha256": parsed.problem_sha256,
        },
    )
    representation = {
        "schema_version": "sigma-active-identified-formula-candidate-1.0",
        "hypothesis_id": hypothesis.hypothesis_id,
        "hypothesis_content_sha256": hypothesis.content_sha256,
        "expression": sp.sstr(hypothesis.expression),
        "expression_srepr": sp.srepr(hypothesis.expression),
        "variable": str(parsed.variable),
        "variable_domain": parsed.value["variable_domain"],
        "problem_sha256": parsed.problem_sha256,
        "identification_evidence_sha256": identification_evidence_sha256,
        "answer_sha256": answer_sha256,
    }
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"actively identified formula {hypothesis.hypothesis_id}",
        representation,
        provenance,
        assumptions=("closed public hypothesis family", "exact public constraints"),
        claims=("uniquely_identified_within_declared_family",),
    )


def _initial_result(
    *,
    session_id: str,
    problem_sha256: str | None,
    valid: bool,
    decision: str,
    reason_codes: Sequence[str],
    survivors: Sequence[str] = (),
    ambiguity: Mapping[str, Any] | None = None,
    query: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    base_counts = {
        "declared_hypotheses": 0,
        "surviving_hypotheses": len(survivors),
        "public_observations": 0,
        "query_budget": 0,
        "query_points_evaluated": 0,
        "legal_defined_queries": 0,
        "informative_queries": 0,
    }
    if counts is not None:
        base_counts.update(counts)
    return _seal(
        {
            "schema_version": INITIAL_RESULT_SCHEMA,
            "session_id": session_id,
            "problem_sha256": problem_sha256,
            "problem_schema_valid": valid,
            "decision": decision,
            "reason_codes": list(reason_codes),
            "surviving_hypothesis_ids": list(survivors),
            "ambiguity_witness": None if ambiguity is None else dict(ambiguity),
            "query": None if query is None else dict(query),
            "candidate": None if candidate is None else dict(candidate),
            "counts": base_counts,
            "system_caps": dict(SYSTEM_CAPS),
            "claims": {
                "ambiguous_data_treated_as_identification": False,
                "candidate_unique_outside_declared_family": False,
                "novelty_established": False,
                "target_opened_during_initial_query_selection": False,
            },
            "scope": (
                "exact identifiability within one caller-declared finite formula family; "
                "ambiguous public data BLOCK and may emit one bounded separating query"
            ),
        }
    )


def run_active_identifiability(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Classify current data and emit a deterministic exact separating query when possible."""

    raw_session = problem.get("session_id") if isinstance(problem, Mapping) else None
    session_id = (
        raw_session
        if isinstance(raw_session, str) and _IDENTIFIER.fullmatch(raw_session)
        else "unbound"
    )
    try:
        problem_sha256 = canonical_sha256(problem)
    except (SchemaViolation, TypeError, ValueError):
        problem_sha256 = None
    try:
        parsed = _parse_problem(problem)
    except _ProblemError as error:
        return _initial_result(
            session_id=session_id,
            problem_sha256=problem_sha256,
            valid=False,
            decision="BLOCK",
            reason_codes=(error.code,),
        )
    survivors = _survivors(parsed)
    common_counts = {
        "declared_hypotheses": len(parsed.hypotheses),
        "surviving_hypotheses": len(survivors),
        "public_observations": len(parsed.observations),
        "query_budget": parsed.query_budget,
    }
    if not survivors:
        return _initial_result(
            session_id=parsed.value["session_id"],
            problem_sha256=parsed.problem_sha256,
            valid=True,
            decision="REJECT",
            reason_codes=("public_data_inconsistent_with_hypothesis_family",),
            counts=common_counts,
        )
    if len(survivors) == 1:
        provisional_sha = canonical_sha256(
            {
                "problem_sha256": parsed.problem_sha256,
                "survivor": survivors[0].hypothesis_id,
                "status": "already_identified",
            }
        )
        candidate = _candidate(
            parsed,
            survivors[0],
            identification_evidence_sha256=provisional_sha,
            answer_sha256=None,
        )
        return _initial_result(
            session_id=parsed.value["session_id"],
            problem_sha256=parsed.problem_sha256,
            valid=True,
            decision="PASS",
            reason_codes=("already_uniquely_identified",),
            survivors=(survivors[0].hypothesis_id,),
            candidate=candidate.to_dict(),
            counts=common_counts,
        )
    ambiguity = _ambiguity_witness(parsed, survivors)
    query, query_counts = _select_query(parsed, survivors, ambiguity["content_sha256"])
    counts = {**common_counts, **query_counts}
    if query is None:
        exhausted = parsed.query_budget < len(parsed.query_space)
        reason = "query_search_budget_exhausted" if exhausted else "no_identifiable_legal_query"
        return _initial_result(
            session_id=parsed.value["session_id"],
            problem_sha256=parsed.problem_sha256,
            valid=True,
            decision="BLOCK",
            reason_codes=(reason,),
            survivors=tuple(item.hypothesis_id for item in survivors),
            ambiguity=ambiguity,
            counts=counts,
        )
    return _initial_result(
        session_id=parsed.value["session_id"],
        problem_sha256=parsed.problem_sha256,
        valid=True,
        decision="BLOCK",
        reason_codes=("ambiguous_public_data_query_proposed",),
        survivors=tuple(item.hypothesis_id for item in survivors),
        ambiguity=ambiguity,
        query=query,
        counts=counts,
    )


def validate_initial_result(result: Mapping[str, Any], problem: Mapping[str, Any]) -> None:
    if (
        not _exact_keys(result, _INITIAL_KEYS)
        or result.get("schema_version") != INITIAL_RESULT_SCHEMA
    ):
        raise ActiveIdentifiabilityError("initial result schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != canonical_sha256(body):
        raise ActiveIdentifiabilityError("initial result content hash changed")
    candidate = result.get("candidate")
    if candidate is not None:
        try:
            CandidateArtifact.from_dict(candidate).validate()
        except (SchemaViolation, TypeError, ValueError) as error:
            raise ActiveIdentifiabilityError("initial Sigma candidate changed") from error
    if dict(result) != run_active_identifiability(problem):
        raise ActiveIdentifiabilityError("initial result exact replay changed")


def build_query_answer(
    problem: Mapping[str, Any],
    initial_result: Mapping[str, Any],
    *,
    target_hypothesis_id: str,
    nonce: str,
) -> dict[str, Any]:
    """Open the preregistered target and bind its exact response to the proposed query."""

    validate_initial_result(initial_result, problem)
    parsed = _parse_problem(problem)
    query = initial_result.get("query")
    if not isinstance(query, Mapping):
        raise ActiveIdentifiabilityError("initial result has no query to answer")
    if (
        target_commitment(parsed.value["session_id"], target_hypothesis_id, nonce)
        != parsed.value["target_commitment"]
    ):
        raise ActiveIdentifiabilityError("target commitment opening mismatch")
    by_id = {item.hypothesis_id: item for item in parsed.hypotheses}
    try:
        target = by_id[target_hypothesis_id]
    except KeyError as error:
        raise ActiveIdentifiabilityError("opened target is not a registered hypothesis") from error
    point = _rational(query["point"], max_bits=parsed.limits["max_integer_bits"])
    value = _evaluate(target.expression, parsed.variable, point)
    if value is None:
        raise ActiveIdentifiabilityError("opened target is undefined at the proposed query")
    body = {
        "schema_version": ANSWER_SCHEMA,
        "problem_sha256": parsed.problem_sha256,
        "ambiguity_result_sha256": initial_result["content_sha256"],
        "query_content_sha256": query["content_sha256"],
        "value": _rational_data(value),
        "opening": {
            "schema_version": OPENING_SCHEMA,
            "session_id": parsed.value["session_id"],
            "target_hypothesis_id": target_hypothesis_id,
            "nonce": nonce,
        },
    }
    return _seal(body)


def _load_answer(
    answer: Mapping[str, Any], parsed: _ParsedProblem, initial_result: Mapping[str, Any]
) -> tuple[sp.Rational, str]:
    if not _exact_keys(answer, _ANSWER_KEYS) or answer.get("schema_version") != ANSWER_SCHEMA:
        raise ActiveIdentifiabilityError("answer schema changed")
    body = {key: value for key, value in answer.items() if key != "content_sha256"}
    if answer.get("content_sha256") != canonical_sha256(body):
        raise ActiveIdentifiabilityError("answer content hash changed")
    query = initial_result.get("query")
    if (
        not isinstance(query, Mapping)
        or answer["problem_sha256"] != parsed.problem_sha256
        or answer["ambiguity_result_sha256"] != initial_result["content_sha256"]
        or answer["query_content_sha256"] != query.get("content_sha256")
    ):
        raise ActiveIdentifiabilityError("answer provenance binding changed")
    opening = answer["opening"]
    if (
        not _exact_keys(opening, {"nonce", "schema_version", "session_id", "target_hypothesis_id"})
        or opening.get("schema_version") != OPENING_SCHEMA
    ):
        raise ActiveIdentifiabilityError("target opening schema changed")
    target_id = opening["target_hypothesis_id"]
    if (
        not isinstance(target_id, str)
        or target_commitment(opening["session_id"], target_id, opening["nonce"])
        != parsed.value["target_commitment"]
        or opening["session_id"] != parsed.value["session_id"]
    ):
        raise ActiveIdentifiabilityError("target commitment opening mismatch")
    value = _rational(answer["value"], max_bits=parsed.limits["max_integer_bits"])
    by_id = {item.hypothesis_id: item for item in parsed.hypotheses}
    try:
        target = by_id[target_id]
    except KeyError as error:
        raise ActiveIdentifiabilityError("opened target is not a registered hypothesis") from error
    point = _rational(query["point"], max_bits=parsed.limits["max_integer_bits"])
    if _evaluate(target.expression, parsed.variable, point) != value:
        raise ActiveIdentifiabilityError("answer does not equal the opened target prediction")
    return value, target_id


def _proof(
    parsed: _ParsedProblem,
    initial_result: Mapping[str, Any],
    answer: Mapping[str, Any],
    hypothesis: _Hypothesis,
    candidate: CandidateArtifact,
) -> dict[str, Any]:
    query = initial_result["query"]
    point = _rational(query["point"], max_bits=parsed.limits["max_integer_bits"])
    answer_value = _rational(answer["value"], max_bits=parsed.limits["max_integer_bits"])
    checks = []
    for observation_point, expected in parsed.observations:
        observed = _evaluate(hypothesis.expression, parsed.variable, observation_point)
        if observed is None or observed != expected:
            raise ActiveIdentifiabilityError("identified hypothesis failed public replay")
        checks.append(
            {
                "kind": "public_observation",
                "point": _rational_data(observation_point),
                "expected": _rational_data(expected),
                "observed": _rational_data(observed),
                "residual_zero": observed == expected,
            }
        )
    active_observed = _evaluate(hypothesis.expression, parsed.variable, point)
    if active_observed is None or active_observed != answer_value:
        raise ActiveIdentifiabilityError("identified hypothesis failed active-answer replay")
    checks.append(
        {
            "kind": "active_query_answer",
            "point": _rational_data(point),
            "expected": _rational_data(answer_value),
            "observed": _rational_data(active_observed),
            "residual_zero": active_observed == answer_value,
        }
    )
    return _seal(
        {
            "schema_version": PROOF_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "initial_result_sha256": initial_result["content_sha256"],
            "answer_content_sha256": answer["content_sha256"],
            "candidate_content_sha256": candidate.content_sha256,
            "hypothesis_id": hypothesis.hypothesis_id,
            "hypothesis_content_sha256": hypothesis.content_sha256,
            "decision": "proved_unique_within_declared_family_after_bound_answer",
            "exact_checks": checks,
            "remaining_equivalence_class_size": 1,
            "claim_boundary": "not uniqueness outside the caller-declared hypothesis family",
        }
    )


def resume_active_identifiability(
    problem: Mapping[str, Any],
    initial_result: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> dict[str, Any]:
    """Consume one bound answer and return PASS only for a uniquely identified target."""

    validate_initial_result(initial_result, problem)
    parsed = _parse_problem(problem)
    answer_value, target_id = _load_answer(answer, parsed, initial_result)
    query = initial_result.get("query")
    if not isinstance(query, Mapping):
        raise ActiveIdentifiabilityError("initial result has no query")
    point = _rational(query["point"], max_bits=parsed.limits["max_integer_bits"])
    by_id = {item.hypothesis_id: item for item in parsed.hypotheses}
    initial_survivors = [by_id[item] for item in initial_result["surviving_hypothesis_ids"]]
    survivors = tuple(
        item
        for item in initial_survivors
        if _evaluate(item.expression, parsed.variable, point) == answer_value
    )
    candidate = None
    proof = None
    if not survivors:
        decision, reasons = "REJECT", ["answer_inconsistent_with_surviving_hypotheses"]
    elif target_id not in {item.hypothesis_id for item in survivors}:
        decision, reasons = "REJECT", ["opened_target_inconsistent_with_answer"]
    elif len(survivors) > 1:
        decision, reasons = "BLOCK", ["answer_did_not_identify_unique_formula"]
    else:
        decision, reasons = "PASS", ["uniquely_identified_after_bound_answer"]
        candidate_object = _candidate(
            parsed,
            survivors[0],
            identification_evidence_sha256=initial_result["content_sha256"],
            answer_sha256=answer["content_sha256"],
        )
        candidate = candidate_object.to_dict()
        proof = _proof(parsed, initial_result, answer, survivors[0], candidate_object)
    body = {
        "schema_version": RESUMED_RESULT_SCHEMA,
        "session_id": parsed.value["session_id"],
        "problem_sha256": parsed.problem_sha256,
        "initial_result_sha256": initial_result["content_sha256"],
        "answer_content_sha256": answer["content_sha256"],
        "decision": decision,
        "reason_codes": reasons,
        "surviving_hypothesis_ids": [item.hypothesis_id for item in survivors],
        "candidate": candidate,
        "proof_certificate": proof,
        "counts": {
            "survivors_before_answer": len(initial_survivors),
            "survivors_after_answer": len(survivors),
            "candidates_emitted": int(candidate is not None),
            "proof_certificates": int(proof is not None),
        },
        "claims": {
            "answer_bound_to_initial_ambiguity_and_query": True,
            "candidate_unique_outside_declared_family": False,
            "novelty_established": False,
            "target_commitment_opened": True,
        },
        "scope": (
            "one exact active-query update over the sealed initial equivalence class; PASS proves "
            "uniqueness only within the caller-declared finite formula family"
        ),
    }
    return _seal(body)


def validate_resumed_result(
    result: Mapping[str, Any],
    problem: Mapping[str, Any],
    initial_result: Mapping[str, Any],
    answer: Mapping[str, Any],
) -> None:
    if (
        not _exact_keys(result, _RESUMED_KEYS)
        or result.get("schema_version") != RESUMED_RESULT_SCHEMA
    ):
        raise ActiveIdentifiabilityError("resumed result schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != canonical_sha256(body):
        raise ActiveIdentifiabilityError("resumed result content hash changed")
    candidate = result.get("candidate")
    if candidate is not None:
        try:
            CandidateArtifact.from_dict(candidate).validate()
        except (SchemaViolation, TypeError, ValueError) as error:
            raise ActiveIdentifiabilityError("resumed Sigma candidate changed") from error
    if dict(result) != resume_active_identifiability(problem, initial_result, answer):
        raise ActiveIdentifiabilityError("resumed result exact replay changed")


__all__ = [
    "ANSWER_SCHEMA",
    "INITIAL_RESULT_SCHEMA",
    "PROBLEM_SCHEMA",
    "RESUMED_RESULT_SCHEMA",
    "SYSTEM_CAPS",
    "ActiveIdentifiabilityError",
    "build_query_answer",
    "resume_active_identifiability",
    "run_active_identifiability",
    "target_commitment",
    "validate_initial_result",
    "validate_proposed_query",
    "validate_resumed_result",
]
