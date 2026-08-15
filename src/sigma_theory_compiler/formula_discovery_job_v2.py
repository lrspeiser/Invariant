"""Formula Discovery Job v2: bounded exact multi-class formula synthesis.

The v2 boundary deliberately supports a small set of mathematically distinct adapters
without pretending to search an unbounded formula space.  All adapters use exact
integer/rational arithmetic, closed schemas, independent public holdout rows, and
deterministic replay.
"""

from __future__ import annotations

import ast
import itertools
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

PROBLEM_SCHEMA = "sigma-formula-discovery-problem-2.0"
RESULT_SCHEMA = "sigma-formula-discovery-result-2.0"
SYNTHESIS_SCHEMA = "sigma-formula-discovery-synthesis-2.0"
VALIDATION_SCHEMA = "sigma-formula-discovery-validation-2.0"
RECURRENCE_PROOF_SCHEMA = "sigma-formula-discovery-recurrence-proof-2.0"

SYSTEM_CAPS = {
    "max_basis_terms": 24,
    "max_constraint_rows": 96,
    "max_expression_nodes": 512,
    "max_integer_bits": 256,
    "max_parameter_combinations": 4096,
    "max_parameters": 6,
    "max_problem_bytes": 131_072,
    "max_recurrence_order": 6,
    "max_validation_rows": 192,
    "max_variables": 6,
}

_PROBLEM_KEYS = {
    "constraints",
    "job_id",
    "limits",
    "premises",
    "proof",
    "schema_version",
    "solver",
    "validation",
    "variables",
}
_RESULT_KEYS = {
    "candidate",
    "claims",
    "class_id",
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
_LIMIT_KEYS = set(SYSTEM_CAPS) - {"max_problem_bytes"}
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SYMBOL_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
_SOLVER_KINDS = {
    "exact_polynomial_basis_v2",
    "exact_rational_basis_v2",
    "exact_nonlinear_parameter_grid_v2",
}


class FormulaDiscoveryV2ValidationError(ValueError):
    """A persisted v2 result failed its closed replay boundary."""


class _ProblemError(ValueError):
    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class _ParsedProblem:
    value: Mapping[str, Any]
    problem_sha256: str
    variables: tuple[sp.Symbol, ...]
    variable_domains: Mapping[str, str]
    premises: tuple[sp.Expr, ...]
    limits: Mapping[str, int]
    solver_kind: str
    solver_data: Mapping[str, Any]
    class_id: str


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


def _parse_expression(
    source: Any,
    symbols: Mapping[str, sp.Symbol],
    *,
    max_bits: int,
    max_nodes: int,
) -> sp.Expr:
    """Parse a closed arithmetic language without evaluation or function calls."""

    if not isinstance(source, str) or not source or len(source) > 1024:
        raise _ProblemError("malformed_problem")
    try:
        root = ast.parse(source, mode="eval")
    except (SyntaxError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if sum(1 for _ in ast.walk(root)) > max_nodes:
        raise _ProblemError("budget_exceeded")

    def convert(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, int):
                raise _ProblemError("unsupported_problem")
            if abs(node.value).bit_length() > max_bits:
                raise _ProblemError("budget_exceeded")
            return sp.Integer(node.value)
        if isinstance(node, ast.Name):
            try:
                return symbols[node.id]
            except KeyError as error:
                raise _ProblemError("malformed_problem") from error
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
    if expression.atoms(sp.Float) or expression.free_symbols - set(symbols.values()):
        raise _ProblemError("malformed_problem")
    if sum(1 for _ in sp.preorder_traversal(expression)) > max_nodes:
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


def _parse_variables(
    value: Any, limits: Mapping[str, int]
) -> tuple[tuple[sp.Symbol, ...], dict[str, str]]:
    if not isinstance(value, list) or not 1 <= len(value) <= limits["max_variables"]:
        raise _ProblemError("malformed_problem")
    variables: list[sp.Symbol] = []
    domains: dict[str, str] = {}
    for row in value:
        if not _exact_keys(row, {"domain", "name"}):
            raise _ProblemError("malformed_problem")
        name, domain = row["name"], row["domain"]
        if (
            not isinstance(name, str)
            or _SYMBOL_NAME.fullmatch(name) is None
            or name.startswith("_")
            or domain not in {"integer", "rational"}
            or name in domains
        ):
            raise _ProblemError("malformed_problem")
        domains[name] = domain
        variables.append(sp.Symbol(name, integer=domain == "integer"))
    return tuple(variables), domains


def _symbol_map(variables: Sequence[sp.Symbol]) -> dict[str, sp.Symbol]:
    return {str(item): item for item in variables}


def _parse_point(
    value: Any, variables: Sequence[sp.Symbol], *, max_bits: int
) -> dict[sp.Symbol, sp.Rational]:
    names = {str(item) for item in variables}
    if not isinstance(value, Mapping) or set(value) != names:
        raise _ProblemError("malformed_problem")
    return {item: _rational(value[str(item)], max_bits=max_bits) for item in variables}


def _point_data(point: Mapping[sp.Symbol, sp.Rational]) -> dict[str, dict[str, int]]:
    return {str(key): _rational_data(point[key]) for key in sorted(point, key=str)}


def _point_key(point: Mapping[sp.Symbol, sp.Rational]) -> tuple[tuple[str, int, int], ...]:
    return tuple((str(key), int(point[key].p), int(point[key].q)) for key in sorted(point, key=str))


def _parse_rows(
    value: Any,
    variables: Sequence[sp.Symbol],
    *,
    max_bits: int,
    maximum: int,
) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value or len(value) > maximum:
        raise _ProblemError("budget_exceeded" if isinstance(value, list) else "malformed_problem")
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, int, int], ...]] = set()
    for row in value:
        if not _exact_keys(row, {"point", "value"}):
            raise _ProblemError("malformed_problem")
        point = _parse_point(row["point"], variables, max_bits=max_bits)
        key = _point_key(point)
        if key in seen:
            raise _ProblemError("malformed_problem")
        seen.add(key)
        rows.append({"point": point, "value": _rational(row["value"], max_bits=max_bits)})
    return rows


def _premises_hold(premises: Sequence[sp.Expr], point: Mapping[sp.Symbol, sp.Rational]) -> bool:
    for premise in premises:
        value = sp.cancel(premise.subs(point))
        if not value.is_Rational or value == 0:
            return False
    return True


def _parse_solver(
    solver: Any,
    variables: Sequence[sp.Symbol],
    limits: Mapping[str, int],
) -> tuple[str, dict[str, Any]]:
    if not isinstance(solver, Mapping) or solver.get("kind") not in _SOLVER_KINDS:
        raise _ProblemError("unsupported_problem")
    kind = solver["kind"]
    symbols = _symbol_map(variables)
    parse = lambda source, table=symbols: _parse_expression(
        source,
        table,
        max_bits=limits["max_integer_bits"],
        max_nodes=limits["max_expression_nodes"],
    )
    if kind == "exact_polynomial_basis_v2":
        if not _exact_keys(solver, {"basis", "kind"}):
            raise _ProblemError("malformed_problem")
        sources = solver["basis"]
        if (
            not isinstance(sources, list)
            or not sources
            or len(sources) > limits["max_basis_terms"]
            or not all(isinstance(item, str) for item in sources)
            or len(set(sources)) != len(sources)
        ):
            raise _ProblemError("malformed_problem")
        basis = tuple(parse(item) for item in sources)
        try:
            for item in basis:
                sp.Poly(item, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.CoercionFailed) as error:
            raise _ProblemError("unsupported_problem") from error
        return kind, {"basis": basis, "basis_sources": tuple(sources)}
    if kind == "exact_rational_basis_v2":
        expected = {"denominator_anchor", "denominator_basis", "kind", "numerator_basis"}
        if not _exact_keys(solver, expected):
            raise _ProblemError("malformed_problem")
        numerator_sources, denominator_sources = (
            solver["numerator_basis"],
            solver["denominator_basis"],
        )
        anchor = solver["denominator_anchor"]
        if (
            not isinstance(numerator_sources, list)
            or not numerator_sources
            or not isinstance(denominator_sources, list)
            or not denominator_sources
            or not all(isinstance(item, str) for item in [*numerator_sources, *denominator_sources])
            or len(numerator_sources) + len(denominator_sources) - 1 > limits["max_basis_terms"]
            or isinstance(anchor, bool)
            or not isinstance(anchor, int)
            or not 0 <= anchor < len(denominator_sources)
        ):
            raise _ProblemError("malformed_problem")
        numerator = tuple(parse(item) for item in numerator_sources)
        denominator = tuple(parse(item) for item in denominator_sources)
        try:
            for item in (*numerator, *denominator):
                sp.Poly(item, *variables, domain=sp.QQ)
        except (sp.PolynomialError, sp.CoercionFailed) as error:
            raise _ProblemError("unsupported_problem") from error
        return kind, {
            "numerator_basis": numerator,
            "numerator_sources": tuple(numerator_sources),
            "denominator_basis": denominator,
            "denominator_sources": tuple(denominator_sources),
            "denominator_anchor": anchor,
        }
    expected = {"expression", "kind", "parameter_relations", "parameters"}
    if not _exact_keys(solver, expected):
        raise _ProblemError("malformed_problem")
    parameters = solver["parameters"]
    if not isinstance(parameters, list) or not 1 <= len(parameters) <= limits["max_parameters"]:
        raise _ProblemError("malformed_problem")
    parameter_symbols: list[sp.Symbol] = []
    value_grid: list[tuple[sp.Rational, ...]] = []
    parameter_names: set[str] = set()
    for row in parameters:
        if not _exact_keys(row, {"name", "values"}):
            raise _ProblemError("malformed_problem")
        name, values = row["name"], row["values"]
        if (
            not isinstance(name, str)
            or _SYMBOL_NAME.fullmatch(name) is None
            or name in symbols
            or name in parameter_names
            or not isinstance(values, list)
            or not values
        ):
            raise _ProblemError("malformed_problem")
        parameter_names.add(name)
        parameter_symbols.append(sp.Symbol(name))
        parsed_values = tuple(
            _rational(item, max_bits=limits["max_integer_bits"]) for item in values
        )
        if len(set(parsed_values)) != len(parsed_values):
            raise _ProblemError("malformed_problem")
        value_grid.append(parsed_values)
    combinations = math.prod(len(values) for values in value_grid)
    if combinations > limits["max_parameter_combinations"]:
        raise _ProblemError("budget_exceeded")
    all_symbols = {**symbols, **{str(item): item for item in parameter_symbols}}
    expression = _parse_expression(
        solver["expression"],
        all_symbols,
        max_bits=limits["max_integer_bits"],
        max_nodes=limits["max_expression_nodes"],
    )
    relations = solver["parameter_relations"]
    if not isinstance(relations, list) or not all(isinstance(item, str) for item in relations):
        raise _ProblemError("malformed_problem")
    parsed_relations = tuple(
        _parse_expression(
            item,
            {str(symbol): symbol for symbol in parameter_symbols},
            max_bits=limits["max_integer_bits"],
            max_nodes=limits["max_expression_nodes"],
        )
        for item in relations
    )
    return kind, {
        "expression": expression,
        "expression_source": solver["expression"],
        "parameters": tuple(parameter_symbols),
        "value_grid": tuple(value_grid),
        "parameter_relations": parsed_relations,
        "parameter_relation_sources": tuple(relations),
        "parameter_combinations": combinations,
    }


def _parse_recurrence(
    constraints: Mapping[str, Any],
    variable: sp.Symbol,
    limits: Mapping[str, int],
) -> dict[str, Any]:
    expected = {
        "coefficients",
        "forcing",
        "initial_values",
        "kind",
        "order",
        "sequence",
    }
    if not _exact_keys(constraints, expected):
        raise _ProblemError("malformed_problem")
    order, sequence = constraints["order"], constraints["sequence"]
    if (
        isinstance(order, bool)
        or not isinstance(order, int)
        or not 2 <= order <= limits["max_recurrence_order"]
        or not isinstance(sequence, str)
        or _SYMBOL_NAME.fullmatch(sequence) is None
    ):
        raise _ProblemError("unsupported_problem")
    coefficients = constraints["coefficients"]
    initial_values = constraints["initial_values"]
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != order
        or not all(isinstance(item, str) for item in coefficients)
        or not isinstance(initial_values, list)
        or len(initial_values) != order
    ):
        raise _ProblemError("malformed_problem")
    symbols = {str(variable): variable}
    parse = lambda source: _parse_expression(
        source,
        symbols,
        max_bits=limits["max_integer_bits"],
        max_nodes=limits["max_expression_nodes"],
    )
    parsed_initials = []
    for expected_index, row in enumerate(initial_values):
        if (
            not _exact_keys(row, {"index", "value"})
            or row["index"] != expected_index
            or isinstance(row["index"], bool)
        ):
            raise _ProblemError("malformed_problem")
        parsed_initials.append(
            {
                "index": expected_index,
                "value": _rational(row["value"], max_bits=limits["max_integer_bits"]),
            }
        )
    return {
        "order": order,
        "sequence": sequence,
        "coefficients": tuple(parse(item) for item in coefficients),
        "coefficient_sources": tuple(coefficients),
        "forcing": parse(constraints["forcing"]),
        "forcing_source": constraints["forcing"],
        "initial_values": tuple(parsed_initials),
    }


def _parse_problem(problem: Mapping[str, Any]) -> _ParsedProblem:
    try:
        payload = canonical_json_bytes(problem)
    except (SchemaViolation, TypeError, ValueError) as error:
        raise _ProblemError("malformed_problem") from error
    if len(payload) > SYSTEM_CAPS["max_problem_bytes"]:
        raise _ProblemError("budget_exceeded")
    if not _exact_keys(problem, _PROBLEM_KEYS) or problem.get("schema_version") != PROBLEM_SCHEMA:
        raise _ProblemError("malformed_problem")
    if not isinstance(problem["job_id"], str) or _IDENTIFIER.fullmatch(problem["job_id"]) is None:
        raise _ProblemError("malformed_problem")
    limits = _parse_limits(problem["limits"])
    variables, domains = _parse_variables(problem["variables"], limits)
    symbols = _symbol_map(variables)
    premises_value = problem["premises"]
    if not isinstance(premises_value, list):
        raise _ProblemError("malformed_problem")
    premises = []
    for row in premises_value:
        if not _exact_keys(row, {"expression", "kind"}) or row["kind"] != "nonzero":
            raise _ProblemError("unsupported_problem")
        premises.append(
            _parse_expression(
                row["expression"],
                symbols,
                max_bits=limits["max_integer_bits"],
                max_nodes=limits["max_expression_nodes"],
            )
        )
    solver_kind, solver_data = _parse_solver(problem["solver"], variables, limits)
    validation = problem["validation"]
    if not _exact_keys(validation, {"kind", "rows"}) or validation["kind"] != "evaluations":
        raise _ProblemError("unsupported_problem")
    validation_rows = _parse_rows(
        validation["rows"],
        variables,
        max_bits=limits["max_integer_bits"],
        maximum=limits["max_validation_rows"],
    )
    constraints = problem["constraints"]
    if not isinstance(constraints, Mapping) or constraints.get("kind") not in {
        "evaluations",
        "linear_recurrence",
    }:
        raise _ProblemError("unsupported_problem")
    if constraints["kind"] == "evaluations":
        if not _exact_keys(constraints, {"kind", "rows"}):
            raise _ProblemError("malformed_problem")
        training_rows = _parse_rows(
            constraints["rows"],
            variables,
            max_bits=limits["max_integer_bits"],
            maximum=limits["max_constraint_rows"],
        )
        if {_point_key(row["point"]) for row in training_rows} & {
            _point_key(row["point"]) for row in validation_rows
        }:
            raise _ProblemError("malformed_problem")
        class_id = {
            "exact_polynomial_basis_v2": "multivariate_polynomial",
            "exact_rational_basis_v2": "rational_function_with_domain",
            "exact_nonlinear_parameter_grid_v2": "nonlinear_algebraic_parameterization",
        }[solver_kind]
    else:
        if solver_kind != "exact_polynomial_basis_v2" or len(variables) != 1:
            raise _ProblemError("unsupported_problem")
        if domains[str(variables[0])] != "integer":
            raise _ProblemError("unsupported_problem")
        recurrence = _parse_recurrence(constraints, variables[0], limits)
        if any(row["point"][variables[0]].q != 1 for row in validation_rows):
            raise _ProblemError("malformed_problem")
        if {int(row["point"][variables[0]]) for row in validation_rows} & set(
            range(recurrence["order"])
        ):
            raise _ProblemError("malformed_problem")
        class_id = "higher_order_recurrence"
    for row in validation_rows:
        if not _premises_hold(premises, row["point"]):
            raise _ProblemError("malformed_problem")
    if constraints["kind"] == "evaluations":
        for row in training_rows:
            if not _premises_hold(premises, row["point"]):
                raise _ProblemError("malformed_problem")
    proof = problem["proof"]
    if not _exact_keys(proof, {"kind"}) or proof["kind"] not in {"none", "recurrence_identity"}:
        raise _ProblemError("unsupported_problem")
    if (constraints["kind"] == "linear_recurrence") != (proof["kind"] == "recurrence_identity"):
        raise _ProblemError("unsupported_problem")
    return _ParsedProblem(
        problem,
        canonical_sha256(problem),
        variables,
        domains,
        tuple(premises),
        limits,
        solver_kind,
        solver_data,
        class_id,
    )


def _evaluations(parsed: _ParsedProblem, which: str) -> list[dict[str, Any]]:
    return _parse_rows(
        parsed.value[which]["rows"],
        parsed.variables,
        max_bits=parsed.limits["max_integer_bits"],
        maximum=parsed.limits[
            "max_constraint_rows" if which == "constraints" else "max_validation_rows"
        ],
    )


def _linear_equations(parsed: _ParsedProblem, candidate: sp.Expr) -> list[sp.Expr]:
    constraints = parsed.value["constraints"]
    if constraints["kind"] == "evaluations":
        return [
            sp.cancel(candidate.subs(row["point"]) - row["value"])
            for row in _evaluations(parsed, "constraints")
        ]
    recurrence = _parse_recurrence(constraints, parsed.variables[0], parsed.limits)
    index = parsed.variables[0]
    residual = candidate.subs(index, index + recurrence["order"]) - recurrence["forcing"]
    for offset, coefficient in enumerate(recurrence["coefficients"]):
        residual -= coefficient * candidate.subs(index, index + offset)
    numerator, denominator = sp.fraction(sp.together(residual))
    if denominator == 0:
        raise _ProblemError("unsupported_problem")
    try:
        equations = list(sp.Poly(sp.expand(numerator), index).all_coeffs())
    except sp.PolynomialError as error:
        raise _ProblemError("unsupported_problem") from error
    equations.extend(
        sp.cancel(candidate.subs(index, row["index"]) - row["value"])
        for row in recurrence["initial_values"]
    )
    if len(equations) > parsed.limits["max_constraint_rows"]:
        raise _ProblemError("budget_exceeded")
    return equations


def _solve_linear(
    parsed: _ParsedProblem,
    candidate: sp.Expr,
    coefficients: Sequence[sp.Symbol],
    details: Mapping[str, Any],
) -> tuple[dict[str, Any], sp.Expr | None, tuple[sp.Rational, ...]]:
    equations = _linear_equations(parsed, candidate)
    try:
        matrix, vector = sp.linear_eq_to_matrix(equations, coefficients)
    except (sp.PolynomialError, ValueError) as error:
        raise _ProblemError("unsupported_problem") from error
    rank, augmented_rank = int(matrix.rank()), int(matrix.row_join(vector).rank())
    outcome, reason = "CANDIDATE", "unique_exact_solution"
    solution: tuple[sp.Rational, ...] = ()
    expression: sp.Expr | None = None
    if augmented_rank > rank:
        outcome, reason = "REJECT", "inconsistent_exact_constraints"
    elif rank < len(coefficients):
        outcome, reason = "BLOCK", "underdetermined_exact_constraints"
    else:
        solution_set = sp.linsolve((matrix, vector), coefficients)
        if solution_set is sp.EmptySet or len(solution_set) != 1:
            raise _ProblemError("unsupported_problem")
        try:
            solution = tuple(sp.Rational(item) for item in next(iter(solution_set)))
        except (TypeError, ValueError) as error:
            raise _ProblemError("unsupported_problem") from error
        expression = sp.cancel(candidate.subs(dict(zip(coefficients, solution, strict=True))))
        if (
            sum(1 for _ in sp.preorder_traversal(expression))
            > parsed.limits["max_expression_nodes"]
        ):
            raise _ProblemError("budget_exceeded")
        if parsed.solver_kind == "exact_rational_basis_v2":
            denominator = sp.fraction(expression)[1]
            if any(
                sp.cancel(denominator.subs(row["point"])) == 0
                for row in _evaluations(parsed, "constraints")
            ):
                outcome, reason, expression = (
                    "REJECT",
                    "candidate_denominator_zero_on_constraint",
                    None,
                )
                solution = ()
    receipt = _seal(
        {
            "schema_version": SYNTHESIS_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "adapter": parsed.solver_kind,
            "class_id": parsed.class_id,
            "outcome": outcome,
            "reason": reason,
            "rank": rank,
            "augmented_rank": augmented_rank,
            "row_count": matrix.rows,
            "column_count": matrix.cols,
            "row_order": list(range(matrix.rows)),
            "coefficients": [_rational_data(item) for item in solution],
            "expression": None if expression is None else sp.sstr(expression),
            "details": dict(details),
        }
    )
    return receipt, expression, solution


def _synthesize_polynomial(parsed: _ParsedProblem) -> tuple[dict[str, Any], sp.Expr | None]:
    basis = parsed.solver_data["basis"]
    coefficients = sp.symbols(f"coefficient_0:{len(basis)}")
    candidate = sum(
        coefficient * term for coefficient, term in zip(coefficients, basis, strict=True)
    )
    receipt, expression, _ = _solve_linear(
        parsed,
        candidate,
        coefficients,
        {"basis": list(parsed.solver_data["basis_sources"])},
    )
    return receipt, expression


def _synthesize_rational(parsed: _ParsedProblem) -> tuple[dict[str, Any], sp.Expr | None]:
    numerator_basis = parsed.solver_data["numerator_basis"]
    denominator_basis = parsed.solver_data["denominator_basis"]
    anchor = parsed.solver_data["denominator_anchor"]
    numerator_coefficients = sp.symbols(f"numerator_coefficient_0:{len(numerator_basis)}")
    denominator_indices = [index for index in range(len(denominator_basis)) if index != anchor]
    denominator_coefficients = sp.symbols(f"denominator_coefficient_0:{len(denominator_indices)}")
    numerator = sum(
        coefficient * term
        for coefficient, term in zip(numerator_coefficients, numerator_basis, strict=True)
    )
    denominator = denominator_basis[anchor] + sum(
        coefficient * denominator_basis[index]
        for coefficient, index in zip(denominator_coefficients, denominator_indices, strict=True)
    )
    coefficients = (*numerator_coefficients, *denominator_coefficients)
    rows = _evaluations(parsed, "constraints")
    equations = [
        sp.cancel(numerator.subs(row["point"]) - row["value"] * denominator.subs(row["point"]))
        for row in rows
    ]
    # Reuse the linear classifier with an equivalent candidate whose evaluation residuals are
    # supplied directly through a temporary parsed-independent solve.
    try:
        matrix, vector = sp.linear_eq_to_matrix(equations, coefficients)
    except (sp.PolynomialError, ValueError) as error:
        raise _ProblemError("unsupported_problem") from error
    rank, augmented_rank = int(matrix.rank()), int(matrix.row_join(vector).rank())
    outcome, reason = "CANDIDATE", "unique_exact_normalized_rational_solution"
    solution: tuple[sp.Rational, ...] = ()
    expression: sp.Expr | None = None
    if augmented_rank > rank:
        outcome, reason = "REJECT", "inconsistent_exact_constraints"
    elif rank < len(coefficients):
        outcome, reason = "BLOCK", "underdetermined_exact_constraints"
    else:
        solution_set = sp.linsolve((matrix, vector), coefficients)
        if solution_set is sp.EmptySet or len(solution_set) != 1:
            raise _ProblemError("unsupported_problem")
        solution = tuple(sp.Rational(item) for item in next(iter(solution_set)))
        substitution = dict(zip(coefficients, solution, strict=True))
        solved_denominator = sp.cancel(denominator.subs(substitution))
        if solved_denominator == 0 or any(
            sp.cancel(solved_denominator.subs(row["point"])) == 0 for row in rows
        ):
            outcome, reason, solution = "REJECT", "candidate_denominator_zero_on_constraint", ()
        else:
            expression = sp.cancel(numerator.subs(substitution) / solved_denominator)
    receipt = _seal(
        {
            "schema_version": SYNTHESIS_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "adapter": parsed.solver_kind,
            "class_id": parsed.class_id,
            "outcome": outcome,
            "reason": reason,
            "rank": rank,
            "augmented_rank": augmented_rank,
            "row_count": matrix.rows,
            "column_count": matrix.cols,
            "row_order": list(range(matrix.rows)),
            "coefficients": [_rational_data(item) for item in solution],
            "expression": None if expression is None else sp.sstr(expression),
            "details": {
                "numerator_basis": list(parsed.solver_data["numerator_sources"]),
                "denominator_basis": list(parsed.solver_data["denominator_sources"]),
                "denominator_anchor": anchor,
                "normalization": "selected denominator coefficient fixed to one",
                "generated_domain_premise": (
                    None if expression is None else f"{sp.sstr(sp.fraction(expression)[1])} != 0"
                ),
            },
        }
    )
    return receipt, expression


def _assignment_data(
    parameters: Sequence[sp.Symbol], values: Sequence[sp.Rational]
) -> dict[str, dict[str, int]]:
    return {
        str(parameter): _rational_data(value)
        for parameter, value in zip(parameters, values, strict=True)
    }


def _synthesize_parameter_grid(parsed: _ParsedProblem) -> tuple[dict[str, Any], sp.Expr | None]:
    data = parsed.solver_data
    rows = _evaluations(parsed, "constraints")
    matches: list[tuple[tuple[sp.Rational, ...], sp.Expr]] = []
    relation_admissible = 0
    for values in itertools.product(*data["value_grid"]):
        substitution = dict(zip(data["parameters"], values, strict=True))
        if any(
            sp.cancel(relation.subs(substitution)) != 0 for relation in data["parameter_relations"]
        ):
            continue
        relation_admissible += 1
        expression = sp.cancel(data["expression"].subs(substitution))
        valid = True
        for row in rows:
            observed = sp.cancel(expression.subs(row["point"]))
            if not observed.is_Rational or observed != row["value"]:
                valid = False
                break
        if valid:
            matches.append((tuple(values), expression))
    by_expression: dict[str, tuple[sp.Expr, list[tuple[sp.Rational, ...]]]] = {}
    for values, expression in matches:
        key = sp.srepr(expression)
        if key not in by_expression:
            by_expression[key] = (expression, [])
        by_expression[key][1].append(values)
    outcome, reason, expression = "CANDIDATE", "unique_exact_grid_expression", None
    selected: tuple[sp.Rational, ...] = ()
    if not by_expression:
        outcome, reason = "REJECT", "inconsistent_exact_parameter_grid"
    elif len(by_expression) > 1:
        outcome, reason = "BLOCK", "underdetermined_exact_parameter_grid"
    else:
        expression, assignments = next(iter(by_expression.values()))
        selected = min(assignments, key=lambda values: tuple((int(v.p), int(v.q)) for v in values))
    receipt = _seal(
        {
            "schema_version": SYNTHESIS_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "adapter": parsed.solver_kind,
            "class_id": parsed.class_id,
            "outcome": outcome,
            "reason": reason,
            "rank": None,
            "augmented_rank": None,
            "row_count": len(rows),
            "column_count": len(data["parameters"]),
            "row_order": list(range(len(rows))),
            "coefficients": [_rational_data(item) for item in selected],
            "expression": None if expression is None else sp.sstr(expression),
            "details": {
                "parameter_names": [str(item) for item in data["parameters"]],
                "parameter_combinations": data["parameter_combinations"],
                "relation_admissible_assignments": relation_admissible,
                "constraint_matching_assignments": len(matches),
                "distinct_matching_expressions": len(by_expression),
                "selected_assignment": (
                    None if not selected else _assignment_data(data["parameters"], selected)
                ),
                "parameter_relations": list(data["parameter_relation_sources"]),
            },
        }
    )
    return receipt, expression


def _synthesize(parsed: _ParsedProblem) -> tuple[dict[str, Any], sp.Expr | None]:
    if parsed.solver_kind == "exact_polynomial_basis_v2":
        return _synthesize_polynomial(parsed)
    if parsed.solver_kind == "exact_rational_basis_v2":
        return _synthesize_rational(parsed)
    return _synthesize_parameter_grid(parsed)


def _domain_descriptor() -> DomainPackDescriptor:
    return DomainPackDescriptor(
        "formula.discovery.job",
        "2.0.0",
        (ArtifactKind.FORMULA,),
        (StageDefinition("generated", 0, (ArtifactKind.FORMULA,)),),
        (),
    )


def _candidate(
    parsed: _ParsedProblem, synthesis: Mapping[str, Any], expression: sp.Expr
) -> CandidateArtifact:
    denominator = sp.fraction(expression)[1]
    generated_premises = (
        [] if denominator == 1 else [{"kind": "nonzero", "expression": sp.sstr(denominator)}]
    )
    caller_premises = [{"kind": "nonzero", "expression": sp.sstr(item)} for item in parsed.premises]
    provenance = ProvenanceRecord.create(
        _domain_descriptor().ref,
        {
            "adapter": parsed.solver_kind,
            "class_id": parsed.class_id,
            "job_id": parsed.value["job_id"],
            "problem_sha256": parsed.problem_sha256,
            "synthesis_sha256": synthesis["content_sha256"],
        },
    )
    representation = {
        "schema_version": "sigma-formula-discovery-candidate-2.0",
        "adapter": parsed.solver_kind,
        "class_id": parsed.class_id,
        "coefficients": synthesis["coefficients"],
        "expression": sp.sstr(expression),
        "expression_srepr": sp.srepr(expression),
        "variables": list(parsed.value["variables"]),
        "caller_domain_premises": caller_premises,
        "generated_domain_premises": generated_premises,
        "problem_sha256": parsed.problem_sha256,
        "solver_receipt_sha256": synthesis["content_sha256"],
    }
    return CandidateArtifact.create(
        ArtifactKind.FORMULA,
        f"exact {parsed.class_id} candidate for {parsed.value['job_id']}",
        representation,
        provenance,
        assumptions=("caller-supplied public exact constraints", "declared nonzero premises"),
        claims=("generated_candidate",),
    )


def _validate_candidate(
    parsed: _ParsedProblem, candidate: CandidateArtifact, expression: sp.Expr
) -> dict[str, Any]:
    rows = _evaluations(parsed, "validation")
    counterexample = None
    checked = 0
    denominator = sp.fraction(expression)[1]
    for index, row in enumerate(rows):
        checked += 1
        denominator_value = sp.cancel(denominator.subs(row["point"]))
        observed = sp.cancel(expression.subs(row["point"]))
        if (
            denominator_value == 0
            or observed.has(sp.zoo, sp.nan, sp.oo, -sp.oo)
            or not observed.is_Rational
        ):
            counterexample = {
                "row_index": index,
                "point": _point_data(row["point"]),
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
                "point": _point_data(row["point"]),
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


def _recurrence_certificate(
    parsed: _ParsedProblem, candidate: CandidateArtifact, expression: sp.Expr
) -> dict[str, Any] | None:
    if parsed.value["proof"]["kind"] == "none":
        return None
    recurrence = _parse_recurrence(parsed.value["constraints"], parsed.variables[0], parsed.limits)
    index = parsed.variables[0]
    residual = expression.subs(index, index + recurrence["order"]) - recurrence["forcing"]
    for offset, coefficient in enumerate(recurrence["coefficients"]):
        residual -= coefficient * expression.subs(index, index + offset)
    numerator, denominator = sp.fraction(sp.cancel(sp.together(residual)))
    initial_checks = []
    for row in recurrence["initial_values"]:
        observed = sp.cancel(expression.subs(index, row["index"]))
        initial_checks.append(
            {
                "index": row["index"],
                "expected": _rational_data(row["value"]),
                "observed": _rational_data(sp.Rational(observed)),
                "residual_zero": observed == row["value"],
            }
        )
    if numerator != 0 or not all(row["residual_zero"] for row in initial_checks):
        raise _ProblemError("proof_failed")
    return _seal(
        {
            "schema_version": RECURRENCE_PROOF_SCHEMA,
            "problem_sha256": parsed.problem_sha256,
            "candidate_content_sha256": candidate.content_sha256,
            "decision": "proved_exact_higher_order_recurrence_identity",
            "order": recurrence["order"],
            "sequence": recurrence["sequence"],
            "recurrence_residual_numerator": sp.sstr(numerator),
            "recurrence_residual_denominator": sp.sstr(denominator),
            "initial_checks": initial_checks,
        }
    )


def _base_result(
    *,
    job_id: str,
    problem_sha256: str | None,
    problem_schema_valid: bool,
    requested_limits: Mapping[str, int] | None,
    class_id: str | None,
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
            "class_id": class_id,
            "decision": decision,
            "reason_codes": list(reason_codes),
            "requested_limits": None if requested_limits is None else dict(requested_limits),
            "system_caps": dict(SYSTEM_CAPS),
            "synthesis": None if synthesis is None else dict(synthesis),
            "candidate": None if candidate is None else dict(candidate),
            "validation": None if validation is None else dict(validation),
            "proof_certificate": None if proof_certificate is None else dict(proof_certificate),
            "counts": {
                "candidates_emitted": int(candidate is not None),
                "counterexamples_found": int(
                    validation is not None and validation.get("counterexample") is not None
                ),
                "proof_certificates": int(proof_certificate is not None),
                "synthesis_rows": 0 if synthesis is None else synthesis["row_count"],
                "validation_rows_checked": 0 if validation is None else validation["checked_rows"],
            },
            "claims": {
                "candidate_is_scientific_law": False,
                "formula_space_exhausted_beyond_declared_adapter": False,
                "novelty_established": False,
                "promotion_authorized": False,
                "synthesis_constraints_are_proof": False,
            },
            "scope": (
                "bounded exact Formula Discovery Job v2 over declared polynomial, normalized "
                "rational, nonlinear parameter-grid, or higher-order recurrence adapters; PASS "
                "requires independent public holdout validation and any requested recurrence proof"
            ),
        }
    )


def run_formula_discovery_job_v2(problem: Mapping[str, Any]) -> dict[str, Any]:
    """Run one deterministic v2 job and fail closed to BLOCK/REJECT/PASS."""

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
            class_id=None,
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
                class_id=parsed.class_id,
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
                class_id=parsed.class_id,
                decision="REJECT",
                reason_codes=("heldout_counterexample",),
                synthesis=synthesis,
                candidate=candidate.to_dict(),
                validation=validation,
            )
        proof = _recurrence_certificate(parsed, candidate, expression)
        return _base_result(
            job_id=problem["job_id"],
            problem_sha256=parsed.problem_sha256,
            problem_schema_valid=True,
            requested_limits=parsed.limits,
            class_id=parsed.class_id,
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
            class_id=parsed.class_id,
            decision="BLOCK",
            reason_codes=(error.code,),
        )


def validate_formula_discovery_result_v2(
    result: Mapping[str, Any], problem: Mapping[str, Any]
) -> None:
    """Validate the closed v2 result, Sigma candidate, seal, and exact replay."""

    if not _exact_keys(result, _RESULT_KEYS) or result.get("schema_version") != RESULT_SCHEMA:
        raise FormulaDiscoveryV2ValidationError("result schema changed")
    body = {key: value for key, value in result.items() if key != "content_sha256"}
    if result.get("content_sha256") != canonical_sha256(body):
        raise FormulaDiscoveryV2ValidationError("result content hash changed")
    candidate = result.get("candidate")
    if candidate is not None:
        try:
            CandidateArtifact.from_dict(candidate).validate()
        except (SchemaViolation, TypeError, ValueError) as error:
            raise FormulaDiscoveryV2ValidationError("Sigma Core candidate changed") from error
    if dict(result) != run_formula_discovery_job_v2(problem):
        raise FormulaDiscoveryV2ValidationError("result exact replay changed")


__all__ = [
    "PROBLEM_SCHEMA",
    "RESULT_SCHEMA",
    "SYSTEM_CAPS",
    "FormulaDiscoveryV2ValidationError",
    "run_formula_discovery_job_v2",
    "validate_formula_discovery_result_v2",
]
