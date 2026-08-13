"""Domain-neutral, bounded exact formula discovery jobs.

The job boundary intentionally separates synthesis constraints from public holdout
validation.  A unique interpolation result is only a candidate; it passes only after
independent exact checks (and, when requested, a checked induction certificate).
"""

from __future__ import annotations

import ast
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import sympy as sp

from .math_expression_ir import Equation, Recurrence, call, literal, symbol
from .math_proof import (
    ProofFailure,
    UnsupportedProof,
    prove_induction,
    validate_induction_certificate,
)
from .math_types import INTEGER
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
from .symbolic_candidate_generator import SymbolicGeneratorError, _sympy_to_ir

PROBLEM_SCHEMA = "sigma-formula-discovery-problem-1.0"
RESULT_SCHEMA = "sigma-formula-discovery-result-1.0"
SYNTHESIS_SCHEMA = "sigma-formula-discovery-synthesis-1.0"
VALIDATION_SCHEMA = "sigma-formula-discovery-validation-1.0"

SYSTEM_CAPS = {
    "max_basis_terms": 16,
    "max_constraint_rows": 64,
    "max_expression_nodes": 256,
    "max_integer_bits": 256,
    "max_problem_bytes": 65_536,
    "max_validation_rows": 128,
}

_PROBLEM_KEYS = {
    "constraints",
    "job_id",
    "limits",
    "proof",
    "schema_version",
    "solver",
    "validation",
    "variable",
    "variable_domain",
}
_RESULT_KEYS = {
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
_LIMIT_KEYS = {
    "max_basis_terms",
    "max_constraint_rows",
    "max_expression_nodes",
    "max_integer_bits",
    "max_validation_rows",
}
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


class FormulaDiscoveryValidationError(ValueError):
    """Raised when a persisted result does not replay exactly."""


class _ProblemError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _ParsedProblem:
    value: Mapping[str, Any]
    problem_sha256: str
    variable: sp.Symbol
    basis: tuple[sp.Expr, ...]
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


def _parse_expression(source: Any, variable: sp.Symbol, *, max_bits: int) -> sp.Expr:
    """Parse a deliberately tiny arithmetic language without Python evaluation."""

    if not isinstance(source, str) or not source or len(source) > 512:
        raise _ProblemError("malformed_problem")
    try:
        root = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if sum(1 for _ in ast.walk(root)) > SYSTEM_CAPS["max_expression_nodes"]:
        raise _ProblemError("budget_exceeded")

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise _ProblemError("unsupported_problem")
            if abs(node.value).bit_length() > max_bits:
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
    if sum(1 for _ in sp.preorder_traversal(expression)) > SYSTEM_CAPS["max_expression_nodes"]:
        raise _ProblemError("budget_exceeded")
    return expression


def _parse_limits(value: Any) -> dict[str, int]:
    if not _exact_keys(value, _LIMIT_KEYS):
        raise _ProblemError("malformed_problem")
    limits: dict[str, int] = {}
    for key in sorted(_LIMIT_KEYS):
        item = value[key]
        if (
            isinstance(item, bool)
            or not isinstance(item, int)
            or item < 1
            or item > SYSTEM_CAPS[key]
        ):
            raise _ProblemError("budget_exceeded")
        limits[key] = item
    return limits


def _validate_rows(value: Any, *, limits: Mapping[str, int], maximum: int) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise _ProblemError("budget_exceeded" if isinstance(value, list) else "malformed_problem")
    rows = []
    for row in value:
        if not _exact_keys(row, {"point", "value"}):
            raise _ProblemError("malformed_problem")
        point = _rational(row["point"], max_bits=limits["max_integer_bits"])
        target = _rational(row["value"], max_bits=limits["max_integer_bits"])
        rows.append({"point": point, "value": target})
    return rows


def _parse_problem(problem: Mapping[str, Any]) -> _ParsedProblem:
    try:
        problem_bytes = canonical_json_bytes(problem)
    except (SchemaViolation, TypeError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if len(problem_bytes) > SYSTEM_CAPS["max_problem_bytes"]:
        raise _ProblemError("budget_exceeded")
    if not _exact_keys(problem, _PROBLEM_KEYS) or problem.get("schema_version") != PROBLEM_SCHEMA:
        raise _ProblemError("malformed_problem")
    job_id = problem["job_id"]
    variable_name = problem["variable"]
    if (
        not isinstance(job_id, str)
        or _IDENTIFIER.fullmatch(job_id) is None
        or not isinstance(variable_name, str)
        or not variable_name.isidentifier()
        or variable_name.startswith("_")
    ):
        raise _ProblemError("malformed_problem")
    if problem["variable_domain"] not in {"integer", "rational"}:
        raise _ProblemError("unsupported_problem")
    limits = _parse_limits(problem["limits"])
    solver = problem["solver"]
    if not _exact_keys(solver, {"basis", "kind"}):
        raise _ProblemError("malformed_problem")
    if solver["kind"] != "exact_linear_basis_v1":
        raise _ProblemError("unsupported_problem")
    basis_sources = solver["basis"]
    if not isinstance(basis_sources, list) or not basis_sources:
        raise _ProblemError("malformed_problem")
    if len(basis_sources) > limits["max_basis_terms"]:
        raise _ProblemError("budget_exceeded")
    if not all(isinstance(item, str) for item in basis_sources) or len(set(basis_sources)) != len(
        basis_sources
    ):
        raise _ProblemError("malformed_problem")
    variable = sp.Symbol(variable_name, integer=problem["variable_domain"] == "integer")
    basis = tuple(
        _parse_expression(item, variable, max_bits=limits["max_integer_bits"])
        for item in basis_sources
    )
    if not _exact_keys(problem["proof"], {"kind"}):
        raise _ProblemError("malformed_problem")
    if problem["proof"]["kind"] not in {"none", "induction"}:
        raise _ProblemError("unsupported_problem")
    validation = problem["validation"]
    if not _exact_keys(validation, {"kind", "rows"}) or validation["kind"] != "evaluations":
        raise _ProblemError("unsupported_problem")
    _validate_rows(validation["rows"], limits=limits, maximum=limits["max_validation_rows"])
    constraints = problem["constraints"]
    if not isinstance(constraints, Mapping) or constraints.get("kind") not in {
        "evaluations",
        "first_order_recurrence",
    }:
        raise _ProblemError("unsupported_problem")
    if constraints["kind"] == "evaluations":
        if not _exact_keys(constraints, {"kind", "rows"}):
            raise _ProblemError("malformed_problem")
        _validate_rows(constraints["rows"], limits=limits, maximum=limits["max_constraint_rows"])
        if problem["proof"]["kind"] != "none":
            raise _ProblemError("unsupported_problem")
    else:
        if not _exact_keys(constraints, {"base", "kind", "sequence", "successor_increment"}):
            raise _ProblemError("malformed_problem")
        if (
            problem["variable_domain"] != "integer"
            or not isinstance(constraints["sequence"], str)
            or not constraints["sequence"].isidentifier()
        ):
            raise _ProblemError("unsupported_problem")
        base = constraints["base"]
        if (
            not _exact_keys(base, {"index", "value"})
            or isinstance(base["index"], bool)
            or not isinstance(base["index"], int)
        ):
            raise _ProblemError("malformed_problem")
        _rational(base["value"], max_bits=limits["max_integer_bits"])
        _parse_expression(
            constraints["successor_increment"],
            variable,
            max_bits=limits["max_integer_bits"],
        )
    return _ParsedProblem(problem, canonical_sha256(problem), variable, basis, limits)


def _linear_system(parsed: _ParsedProblem) -> tuple[list[sp.Expr], int]:
    problem, variable = parsed.value, parsed.variable
    coefficients = sp.symbols(f"coefficient_0:{len(parsed.basis)}")
    candidate = sum(
        coefficient * basis for coefficient, basis in zip(coefficients, parsed.basis, strict=True)
    )
    constraints = problem["constraints"]
    equations: list[sp.Expr] = []
    if constraints["kind"] == "evaluations":
        rows = _validate_rows(
            constraints["rows"],
            limits=parsed.limits,
            maximum=parsed.limits["max_constraint_rows"],
        )
        equations.extend(
            sp.cancel(candidate.subs(variable, row["point"]) - row["value"]) for row in rows
        )
    else:
        increment = _parse_expression(
            constraints["successor_increment"],
            variable,
            max_bits=parsed.limits["max_integer_bits"],
        )
        residual = sp.together(candidate.subs(variable, variable + 1) - candidate - increment)
        numerator, denominator = sp.fraction(residual)
        if denominator == 0 or denominator.free_symbols & set(coefficients):
            raise _ProblemError("unsupported_problem")
        try:
            polynomial = sp.Poly(sp.expand(numerator), variable)
        except sp.PolynomialError as error:
            raise _ProblemError("unsupported_problem") from error
        equations.extend(polynomial.all_coeffs())
        base = constraints["base"]
        equations.append(
            sp.cancel(
                candidate.subs(variable, base["index"])
                - _rational(base["value"], max_bits=parsed.limits["max_integer_bits"])
            )
        )
    if len(equations) > parsed.limits["max_constraint_rows"]:
        raise _ProblemError("budget_exceeded")
    return equations, len(coefficients)


def _synthesize(parsed: _ParsedProblem) -> tuple[dict[str, Any], sp.Expr | None]:
    equations, column_count = _linear_system(parsed)
    coefficients = sp.symbols(f"coefficient_0:{column_count}")
    try:
        matrix, vector = sp.linear_eq_to_matrix(equations, coefficients)
    except (sp.PolynomialError, ValueError) as error:
        raise _ProblemError("unsupported_problem") from error
    rank = int(matrix.rank())
    augmented_rank = int(matrix.row_join(vector).rank())
    outcome, reason = "CANDIDATE", "unique_exact_solution"
    solution: tuple[sp.Rational, ...] = ()
    expression: sp.Expr | None = None
    if augmented_rank > rank:
        outcome, reason = "REJECT", "inconsistent_exact_constraints"
    elif rank < column_count:
        outcome, reason = "BLOCK", "underdetermined_exact_constraints"
    else:
        solution_set = sp.linsolve((matrix, vector), coefficients)
        if solution_set is sp.EmptySet or len(solution_set) != 1:
            raise _ProblemError("unsupported_problem")
        solution = tuple(sp.Rational(item) for item in next(iter(solution_set)))
        expression = sp.cancel(
            sum(value * basis for value, basis in zip(solution, parsed.basis, strict=True))
        )
        if (
            sum(1 for _ in sp.preorder_traversal(expression))
            > parsed.limits["max_expression_nodes"]
        ):
            raise _ProblemError("budget_exceeded")
    receipt = _seal(
        {
            "schema_version": SYNTHESIS_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "outcome": outcome,
            "reason": reason,
            "rank": rank,
            "augmented_rank": augmented_rank,
            "row_count": matrix.rows,
            "column_count": matrix.cols,
            "row_order": list(range(matrix.rows)),
            "coefficients": [_rational_data(item) for item in solution],
            "expression": None if expression is None else sp.sstr(expression),
        }
    )
    return receipt, expression


def _domain_descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        "formula.discovery.job",
        "1.0.0",
        (ArtifactKind.FORMULA,),
        (StageDefinition("generated", 0, (ArtifactKind.FORMULA,)),),
        (),
    )


def _candidate(
    parsed: _ParsedProblem, synthesis: Mapping[str, Any], expression: sp.Expr
) -> CandidateArtifact:
    problem = parsed.value
    provenance = ProvenanceRecord.create(
        _domain_descriptor().ref,
        {
            "job_id": problem["job_id"],
            "problem_sha256": parsed.problem_sha256,
            "solver": "exact_linear_basis_v1",
            "synthesis_sha256": synthesis["content_sha256"],
        },
    )
    representation = {
        "schema_version": "sigma-formula-discovery-candidate-1.0",
        "basis": list(problem["solver"]["basis"]),
        "coefficients": synthesis["coefficients"],
        "expression": sp.sstr(expression),
        "expression_srepr": sp.srepr(expression),
        "problem_sha256": parsed.problem_sha256,
        "solver_receipt_sha256": synthesis["content_sha256"],
        "variable": problem["variable"],
        "variable_domain": problem["variable_domain"],
    }
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"exact linear-basis candidate for {problem['job_id']}",
        representation,
        provenance,
        assumptions=("caller-supplied public exact constraints",),
        claims=("generated_candidate",),
    )


def _validate_candidate(
    parsed: _ParsedProblem, candidate: CandidateArtifact, expression: sp.Expr
) -> dict[str, Any]:
    rows = _validate_rows(
        parsed.value["validation"]["rows"],
        limits=parsed.limits,
        maximum=parsed.limits["max_validation_rows"],
    )
    if parsed.value["constraints"]["kind"] == "evaluations":
        training = {
            _rational(row["point"], max_bits=parsed.limits["max_integer_bits"])
            for row in parsed.value["constraints"]["rows"]
        }
        if any(row["point"] in training for row in rows):
            raise _ProblemError("malformed_problem")
    counterexample = None
    checked = 0
    for index, row in enumerate(rows):
        checked += 1
        observed = sp.cancel(expression.subs(parsed.variable, row["point"]))
        if observed.has(sp.zoo, sp.nan, sp.oo, -sp.oo) or not observed.is_Rational:
            counterexample = {
                "row_index": index,
                "point": _rational_data(row["point"]),
                "expected": _rational_data(row["value"]),
                "observed": None,
                "residual": None,
                "reason": "candidate_undefined_or_nonrational",
            }
            break
        residual = sp.Rational(observed - row["value"])
        if residual != 0:
            counterexample = {
                "row_index": index,
                "point": _rational_data(row["point"]),
                "expected": _rational_data(row["value"]),
                "observed": _rational_data(sp.Rational(observed)),
                "residual": _rational_data(residual),
                "reason": "exact_heldout_mismatch",
            }
            break
    return _seal(
        {
            "schema_version": VALIDATION_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "validation_input_sha256": canonical_sha256(parsed.value["validation"]),
            "candidate_content_sha256": candidate.content_sha256,
            "status": "PASS" if counterexample is None else "REJECT",
            "checked_rows": checked,
            "counterexample": counterexample,
        }
    )


def _induction_certificate(parsed: _ParsedProblem, expression: sp.Expr) -> dict[str, Any] | None:
    if parsed.value["proof"]["kind"] == "none":
        return None
    constraints = parsed.value["constraints"]
    if constraints["kind"] != "first_order_recurrence":
        raise _ProblemError("unsupported_problem")
    try:
        index = symbol(parsed.value["variable"], INTEGER)
        sequence = constraints["sequence"]
        increment = _parse_expression(
            constraints["successor_increment"],
            parsed.variable,
            max_bits=parsed.limits["max_integer_bits"],
        )
        increment_ir = _sympy_to_ir(increment, {parsed.value["variable"]: INTEGER})
        expression_ir = _sympy_to_ir(expression, {parsed.value["variable"]: INTEGER})
        base_value = _rational(
            constraints["base"]["value"], max_bits=parsed.limits["max_integer_bits"]
        )
        recurrence = Recurrence(
            sequence,
            index,
            1,
            Equation(call(sequence, index + 1), call(sequence, index) + increment_ir),
            (
                (
                    constraints["base"]["index"],
                    literal(Fraction(int(base_value.p), int(base_value.q))),
                ),
            ),
        )
        statement = Equation(call(sequence, index), expression_ir)
        certificate = prove_induction(
            statement, recurrence, base_index=constraints["base"]["index"]
        )
        validate_induction_certificate(certificate, statement, recurrence)
        return certificate
    except (ProofFailure, UnsupportedProof, SymbolicGeneratorError, TypeError, ValueError) as error:
        raise _ProblemError("proof_failed") from error


def _base_result(
    *,
    job_id: str,
    problem_sha256: str | None,
    problem_schema_valid: bool,
    requested_limits: Mapping[str, int] | None,
    decision: str,
    reason_codes: Sequence[str],
    synthesis: Mapping[str, Any] | None = None,
    candidate: Mapping[str, Any] | None = None,
    validation: Mapping[str, Any] | None = None,
    proof_certificate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return _seal(
        {
            "schema_version": RESULT_SCHEMA,
            "job_id": job_id,
            "problem_sha256": problem_sha256,
            "problem_schema_valid": problem_schema_valid,
            "decision": decision,
            "reason_codes": list(reason_codes),
            "requested_limits": None if requested_limits is None else dict(requested_limits),
            "system_caps": dict(SYSTEM_CAPS),
            "synthesis": None if synthesis is None else dict(synthesis),
            "candidate": None if candidate is None else dict(candidate),
            "validation": None if validation is None else dict(validation),
            "proof_certificate": (None if proof_certificate is None else dict(proof_certificate)),
            "counts": {
                "candidates_emitted": int(candidate is not None),
                "counterexamples_found": int(
                    validation is not None and validation.get("counterexample") is not None
                ),
                "proof_certificates": int(proof_certificate is not None),
                "synthesis_rows": 0 if synthesis is None else synthesis["row_count"],
                "validation_rows_checked": (
                    0 if validation is None else validation["checked_rows"]
                ),
            },
            "claims": {
                "candidate_is_scientific_law": False,
                "novelty_established": False,
                "promotion_authorized": False,
                "synthesis_constraints_are_proof": False,
            },
            "scope": (
                "bounded exact linear-basis synthesis from caller-supplied public constraints; "
                "PASS requires disjoint public holdout checks and any requested induction proof"
            ),
        }
    )


def run_formula_discovery_job(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Run one deterministic job, returning BLOCK/REJECT/PASS instead of failing open."""

    raw_job_id = problem.get("job_id") if isinstance(problem, Mapping) else None
    job_id = (
        raw_job_id
        if isinstance(raw_job_id, str) and _IDENTIFIER.fullmatch(raw_job_id)
        else "unbound"
    )
    try:
        problem_sha256 = canonical_sha256(problem)
    except (SchemaViolation, TypeError, ValueError):
        problem_sha256 = None
    try:
        parsed = _parse_problem(problem)
    except _ProblemError as error:
        return _base_result(
            job_id=job_id,
            problem_sha256=problem_sha256,
            problem_schema_valid=False,
            requested_limits=None,
            decision="BLOCK",
            reason_codes=(error.code,),
        )
    try:
        synthesis, expression = _synthesize(parsed)
        if synthesis["outcome"] != "CANDIDATE" or expression is None:
            decision = "REJECT" if synthesis["outcome"] == "REJECT" else "BLOCK"
            return _base_result(
                job_id=problem["job_id"],
                problem_sha256=parsed.problem_sha256,
                problem_schema_valid=True,
                requested_limits=parsed.limits,
                decision=decision,
                reason_codes=(synthesis["reason"],),
                synthesis=synthesis,
            )
        candidate = _candidate(parsed, synthesis, expression)
        validation = _validate_candidate(parsed, candidate, expression)
        if validation["status"] != "PASS":
            return _base_result(
                job_id=problem["job_id"],
                problem_sha256=parsed.problem_sha256,
                problem_schema_valid=True,
                requested_limits=parsed.limits,
                decision="REJECT",
                reason_codes=("heldout_counterexample",),
                synthesis=synthesis,
                candidate=candidate.to_dict(),
                validation=validation,
            )
        try:
            proof = _induction_certificate(parsed, expression)
        except _ProblemError as error:
            return _base_result(
                job_id=problem["job_id"],
                problem_sha256=parsed.problem_sha256,
                problem_schema_valid=True,
                requested_limits=parsed.limits,
                decision="REJECT",
                reason_codes=(error.code,),
                synthesis=synthesis,
                candidate=candidate.to_dict(),
                validation=validation,
            )
        return _base_result(
            job_id=problem["job_id"],
            problem_sha256=parsed.problem_sha256,
            problem_schema_valid=True,
            requested_limits=parsed.limits,
            decision="PASS",
            reason_codes=("exact_holdout_validated",),
            synthesis=synthesis,
            candidate=candidate.to_dict(),
            validation=validation,
            proof_certificate=proof,
        )
    except _ProblemError as error:
        return _base_result(
            job_id=problem["job_id"],
            problem_sha256=parsed.problem_sha256,
            problem_schema_valid=True,
            requested_limits=parsed.limits,
            decision="BLOCK",
            reason_codes=(error.code,),
        )


def validate_formula_discovery_result(
    result: Mapping[str, Any], problem: Mapping[str, Any]
) -> None:
    """Validate the closed result schema, all seals, Sigma candidate, and exact replay."""

    if not _exact_keys(result, _RESULT_KEYS) or result.get("schema_version") != RESULT_SCHEMA:
        raise FormulaDiscoveryValidationError("result schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != canonical_sha256(body):
        raise FormulaDiscoveryValidationError("result content hash changed")
    candidate = result.get("candidate")
    if candidate is not None:
        try:
            CandidateArtifact.from_dict(candidate).validate()
        except (SchemaViolation, TypeError, ValueError) as error:
            raise FormulaDiscoveryValidationError("Sigma Core candidate changed") from error
    if dict(result) != run_formula_discovery_job(problem):
        raise FormulaDiscoveryValidationError("result exact replay changed")


__all__ = [
    "PROBLEM_SCHEMA",
    "RESULT_SCHEMA",
    "SYSTEM_CAPS",
    "FormulaDiscoveryValidationError",
    "run_formula_discovery_job",
    "validate_formula_discovery_result",
]
