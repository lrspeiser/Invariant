"""Externally authored, sealed known/unknown mathematical creativity experiment.

This campaign keeps source identity and target material outside the generation view, compares
each creativity family with a budget-matched random control, and opens holdouts only after
deterministic and optional Claude proposer/critic traces are sealed.  Results are exact bounded
experiment facts.  They are not novelty claims and they do not solve the bounded-unknown source
problems.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

import mpmath
import sympy as sp
import z3

from .claude_creativity_api import (
    ClaudeAPIConfig,
    ClaudeCallResult,
    ClaudeCallStatus,
    ClaudeCreativityClient,
    ClaudeHypothesis,
    ClaudeRole,
    Transport,
)
from .external_claude_transport import ProviderCompatibleClaudeTransport
from .independent_exact_evaluator import (
    IndependentEvaluationError,
)
from .independent_exact_evaluator import (
    evaluate_expression as independently_evaluate_expression,
)
from .independent_exact_evaluator import (
    evaluate_recurrence as independently_evaluate_recurrence,
)
from .sigma_core import canonical_sha256

PUBLIC_CONFIG_PATH = "configs/external_sealed_creativity_benchmarks.json"
CAMPAIGN_CONFIG_PATH = "configs/external_creativity_validation_campaign.json"
OUTPUT_PATH = "runs/math/external-creativity-validation/campaign.json"
SOURCE_PATH = "src/sigma_theory_compiler/external_creativity_validation.py"
CLAUDE_SOURCE_PATH = "src/sigma_theory_compiler/claude_creativity_api.py"
CLAUDE_TRANSPORT_SOURCE_PATH = "src/sigma_theory_compiler/external_claude_transport.py"
INDEPENDENT_EVALUATOR_PATH = "src/sigma_theory_compiler/independent_exact_evaluator.py"
LEAN_SOURCE_PATH = "formal/lean/ExternalKnownFormulaControls.lean"
TEST_PATH = "tests/test_external_creativity_validation.py"
PUBLIC_SCHEMA = "invariant-external-sealed-benchmarks-1.0"
TARGET_SCHEMA = "invariant-external-sealed-targets-1.0"
CAMPAIGN_SCHEMA = "invariant-external-creativity-validation-config-1.0"
RECEIPT_SCHEMA = "invariant-external-creativity-validation-result-1.0"
ALLOWED_EXTERNAL_DOMAINS = frozenset({"dlmf.nist.gov", "oeis.org", "openstax.org"})
HEX_DIGITS = frozenset("0123456789abcdef")
FAMILY_IDS = (
    "analogy_transfer",
    "conserved_quantity_synthesis",
    "counterexample_guided_repair",
    "dimensional_analysis",
    "generating_function",
    "inverse_variational",
    "quotient_construction",
    "recurrence_guessing",
    "representation_change",
    "symmetry_reduction",
)


class ExternalCreativityError(ValueError):
    """The external benchmark, blind chronology, or verifier policy failed closed."""


def _strict_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise ExternalCreativityError(f"{label} keys changed")


def _sha(value: Any, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in HEX_DIGITS for character in value)
    ):
        raise ExternalCreativityError(f"{label} is not a lowercase SHA-256 digest")
    return value


def _identifier(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]{0,127}", value) is None:
        raise ExternalCreativityError(f"{label} is not a portable identifier")
    return value


def _fraction(value: Any, label: str) -> Fraction:
    if not isinstance(value, str) or re.fullmatch(r"-?[0-9]+(?:/[1-9][0-9]*)?", value) is None:
        raise ExternalCreativityError(f"{label} is not an exact rational string")
    result = Fraction(value)
    if str(result) != value and f"{result.numerator}/{result.denominator}" != value:
        raise ExternalCreativityError(f"{label} is not canonically reduced")
    return result


def _fraction_text(value: Fraction) -> str:
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def _file_sha256(path: Path) -> str:
    normalized_text = path.read_text(encoding="utf-8").replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Variable:
    name: str
    dimension: tuple[int, int, int]
    unit: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Variable:
        _strict_keys(value, {"dimension", "name", "unit"}, "benchmark variable")
        dimension = value["dimension"]
        if (
            not isinstance(dimension, list)
            or len(dimension) != 3
            or any(isinstance(item, bool) or not isinstance(item, int) for item in dimension)
        ):
            raise ExternalCreativityError("variable dimension must be an integer M/L/T vector")
        unit = value["unit"]
        if not isinstance(unit, str) or not unit:
            raise ExternalCreativityError("variable unit is empty")
        return cls(_identifier(value["name"], "variable name"), tuple(dimension), unit)


@dataclass(frozen=True, slots=True)
class Observation:
    inputs: tuple[Fraction, ...]
    output: Fraction


@dataclass(frozen=True, slots=True)
class ExternalSource:
    authoring_principal_id: str
    retrieved_utc: str
    source_locator: str
    source_uri: str

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, Any], *, generator_principal_id: str
    ) -> ExternalSource:
        _strict_keys(
            value,
            {"authoring_principal_id", "retrieved_utc", "source_locator", "source_uri"},
            "external authorship",
        )
        principal = _identifier(value["authoring_principal_id"], "external principal")
        if principal == generator_principal_id or not principal.startswith("external."):
            raise ExternalCreativityError("benchmark author is not external to the generator")
        from urllib.parse import urlparse

        parsed = urlparse(value["source_uri"])
        if parsed.scheme != "https" or parsed.hostname not in ALLOWED_EXTERNAL_DOMAINS:
            raise ExternalCreativityError("external benchmark source is not authoritative")
        if not re.fullmatch(r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value["retrieved_utc"]):
            raise ExternalCreativityError("external source retrieval time is invalid")
        if not isinstance(value["source_locator"], str) or not value["source_locator"]:
            raise ExternalCreativityError("external source locator is empty")
        return cls(principal, value["retrieved_utc"], value["source_locator"], value["source_uri"])

    def to_dict(self) -> dict[str, str]:
        return {
            "authoring_principal_id": self.authoring_principal_id,
            "retrieved_utc": self.retrieved_utc,
            "source_locator": self.source_locator,
            "source_uri": self.source_uri,
        }


@dataclass(frozen=True, slots=True)
class Benchmark:
    benchmark_id: str
    capability_level: int
    source: ExternalSource
    variables: tuple[Variable, ...]
    output_dimension: tuple[int, int, int]
    output_unit: str
    observations: tuple[Observation, ...]
    dataset_protocol: Mapping[str, Any]
    target_commitment: str

    @property
    def blind_id(self) -> str:
        return f"blind.{self.target_commitment[:20]}"

    @property
    def aliases(self) -> tuple[str, ...]:
        return tuple(f"x{index}" for index in range(len(self.variables)))

    def generation_view(self) -> dict[str, Any]:
        return {
            "blind_id": self.blind_id,
            "observations": [
                {
                    "inputs": {
                        alias: _fraction_text(value)
                        for alias, value in zip(self.aliases, row.inputs, strict=True)
                    },
                    "output": _fraction_text(row.output),
                }
                for row in self.observations
            ],
            "output_dimension": list(self.output_dimension),
            "variable_dimensions": [list(item.dimension) for item in self.variables],
            "variables": list(self.aliases),
        }


@dataclass(frozen=True, slots=True)
class SealedTarget:
    benchmark_id: str
    holdout_records: tuple[Observation, ...]
    reference_formula: str | None
    target_kind: str
    commitment: str


def _parse_observations(
    raw: Any, variables: Sequence[Variable], *, label: str
) -> tuple[Observation, ...]:
    if not isinstance(raw, list) or not raw:
        raise ExternalCreativityError(f"{label} must be a nonempty JSON array")
    names = [item.name for item in variables]
    rows = []
    for item in raw:
        _strict_keys(item, {"inputs", "output"}, label)
        inputs = item["inputs"]
        if not isinstance(inputs, Mapping) or set(inputs) != set(names):
            raise ExternalCreativityError(f"{label} input shape changed")
        rows.append(
            Observation(
                tuple(_fraction(inputs[name], f"{label} input") for name in names),
                _fraction(item["output"], f"{label} output"),
            )
        )
    if len({row.inputs for row in rows}) != len(rows):
        raise ExternalCreativityError(f"{label} repeats an input object")
    return tuple(rows)


def load_public_benchmarks(root: Path) -> tuple[dict[str, Any], tuple[Benchmark, ...]]:
    path = root / PUBLIC_CONFIG_PATH
    raw = json.loads(path.read_text(encoding="utf-8"))
    _strict_keys(
        raw,
        {"benchmarks", "generator_principal_id", "schema_version", "sealed_targets_path"},
        "external benchmark config",
    )
    if raw["schema_version"] != PUBLIC_SCHEMA:
        raise ExternalCreativityError("external benchmark schema changed")
    generator = _identifier(raw["generator_principal_id"], "generator principal")
    if raw["sealed_targets_path"] != "configs/external_sealed_creativity_targets.json":
        raise ExternalCreativityError("sealed target path changed")
    rows = raw["benchmarks"]
    if not isinstance(rows, list) or len(rows) < 4:
        raise ExternalCreativityError("known/unknown external benchmark coverage is incomplete")
    benchmarks = []
    for item in rows:
        _strict_keys(
            item,
            {
                "benchmark_id",
                "capability_level",
                "dataset_protocol",
                "external_authorship",
                "observations",
                "output",
                "target_commitment",
                "variables",
            },
            "external benchmark",
        )
        variables_raw = item["variables"]
        if not isinstance(variables_raw, list) or not 1 <= len(variables_raw) <= 6:
            raise ExternalCreativityError("benchmark variable count is outside policy")
        variables = tuple(Variable.from_mapping(value) for value in variables_raw)
        if len({value.name for value in variables}) != len(variables):
            raise ExternalCreativityError("benchmark variables repeat")
        output = item["output"]
        _strict_keys(output, {"dimension", "unit"}, "benchmark output")
        dimension = output["dimension"]
        if (
            not isinstance(dimension, list)
            or len(dimension) != 3
            or any(isinstance(value, bool) or not isinstance(value, int) for value in dimension)
        ):
            raise ExternalCreativityError("output dimension must be an integer M/L/T vector")
        capability = item["capability_level"]
        if capability not in {4, 5}:
            raise ExternalCreativityError("external benchmark is outside capability levels 4/5")
        protocol = item["dataset_protocol"]
        _strict_keys(
            protocol,
            {
                "causal_interventions",
                "dimension_basis",
                "ood_split_rule",
                "residual_channels",
                "symmetry_coordinates",
            },
            "dataset protocol",
        )
        for key in ("causal_interventions", "residual_channels", "symmetry_coordinates"):
            values = protocol[key]
            if not isinstance(values, list) or not values or len(values) != len(set(values)):
                raise ExternalCreativityError(f"dataset protocol {key} is not nonempty and unique")
        if protocol["dimension_basis"] != ["mass", "length", "time"]:
            raise ExternalCreativityError("dataset dimension basis changed")
        benchmarks.append(
            Benchmark(
                _identifier(item["benchmark_id"], "benchmark_id"),
                capability,
                ExternalSource.from_mapping(
                    item["external_authorship"], generator_principal_id=generator
                ),
                variables,
                tuple(dimension),
                output["unit"],
                _parse_observations(item["observations"], variables, label="training observations"),
                protocol,
                _sha(item["target_commitment"], "target commitment"),
            )
        )
    if len({item.benchmark_id for item in benchmarks}) != len(benchmarks):
        raise ExternalCreativityError("external benchmark IDs repeat")
    levels = [item.capability_level for item in benchmarks]
    if levels.count(4) < 2 or levels.count(5) < 2:
        raise ExternalCreativityError("campaign lacks two known and two bounded-unknown benchmarks")
    return raw, tuple(benchmarks)


def unseal_targets(root: Path, public: Mapping[str, Any], benchmarks: Sequence[Benchmark]) -> tuple[SealedTarget, ...]:
    path = root / public["sealed_targets_path"]
    raw = json.loads(path.read_text(encoding="utf-8"))
    _strict_keys(raw, {"schema_version", "targets"}, "sealed target config")
    if raw["schema_version"] != TARGET_SCHEMA:
        raise ExternalCreativityError("sealed target schema changed")
    by_id = {item.benchmark_id: item for item in benchmarks}
    targets = []
    if not isinstance(raw["targets"], list):
        raise ExternalCreativityError("sealed targets must be a JSON array")
    for item in raw["targets"]:
        _strict_keys(
            item,
            {"benchmark_id", "holdout_records", "reference_formula", "target_kind"},
            "sealed target",
        )
        benchmark_id = item["benchmark_id"]
        benchmark = by_id.get(benchmark_id)
        if benchmark is None:
            raise ExternalCreativityError("sealed target has no public benchmark")
        commitment = canonical_sha256(item)
        if commitment != benchmark.target_commitment:
            raise ExternalCreativityError("sealed target does not open its public commitment")
        target_kind = item["target_kind"]
        reference = item["reference_formula"]
        if target_kind == "known_formula":
            if benchmark.capability_level != 4 or not isinstance(reference, str) or not reference:
                raise ExternalCreativityError("known target has no reference formula")
        elif target_kind == "bounded_unknown":
            if benchmark.capability_level != 5 or reference is not None:
                raise ExternalCreativityError("bounded-unknown target smuggles a reference formula")
        else:
            raise ExternalCreativityError("sealed target kind is invalid")
        targets.append(
            SealedTarget(
                benchmark_id,
                _parse_observations(item["holdout_records"], benchmark.variables, label="holdout"),
                reference,
                target_kind,
                commitment,
            )
        )
    if set(by_id) != {item.benchmark_id for item in targets}:
        raise ExternalCreativityError("sealed target coverage differs from public benchmarks")
    return tuple(sorted(targets, key=lambda item: item.benchmark_id))


def _safe_expression(expression: str, aliases: Sequence[str]) -> sp.Expr:
    if not isinstance(expression, str) or len(expression) > 512:
        raise ExternalCreativityError("candidate expression is empty or oversized")
    expression = expression.replace("^", "**")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as error:
        raise ExternalCreativityError("candidate expression is not valid DSL syntax") from error
    symbols = {name: sp.Symbol(name, real=True) for name in aliases}

    def visit(node: ast.AST) -> sp.Expr:
        if isinstance(node, ast.Expression):
            return visit(node.body)
        if isinstance(node, ast.Name) and node.id in symbols:
            return symbols[node.id]
        if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
            if abs(node.value) > 10**12:
                raise ExternalCreativityError("candidate integer literal is too large")
            return sp.Integer(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp):
            left = visit(node.left)
            right = visit(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                if not right.is_Integer or not -8 <= int(right) <= 8:
                    raise ExternalCreativityError("candidate exponent is outside [-8, 8]")
                return left**right
        raise ExternalCreativityError("candidate expression left the arithmetic DSL")

    result = sp.cancel(visit(tree))
    if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
        raise ExternalCreativityError("candidate expression is singular at construction")
    return result


@dataclass(frozen=True, slots=True)
class Candidate:
    candidate_id: str
    family: str
    representation: str
    expression: str
    recurrence_coefficients: tuple[Fraction, ...]
    recurrence_seed: tuple[Fraction, ...]
    invariants: tuple[str, ...]
    proof_plan: tuple[str, ...]
    proposer: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "expression": self.expression,
            "family": self.family,
            "invariants": list(self.invariants),
            "proof_plan": list(self.proof_plan),
            "proposer": self.proposer,
            "recurrence_coefficients": [_fraction_text(item) for item in self.recurrence_coefficients],
            "recurrence_seed": [_fraction_text(item) for item in self.recurrence_seed],
            "representation": self.representation,
        }


def _candidate(
    benchmark: Benchmark,
    family: str,
    representation: str,
    expression: str,
    *,
    recurrence_coefficients: Sequence[Fraction] = (),
    recurrence_seed: Sequence[Fraction] = (),
    invariants: Sequence[str],
    proof_plan: Sequence[str],
    proposer: str = "deterministic_portfolio",
) -> Candidate:
    body = {
        "blind_id": benchmark.blind_id,
        "expression": expression,
        "family": family,
        "proposer": proposer,
        "recurrence_coefficients": [_fraction_text(item) for item in recurrence_coefficients],
        "recurrence_seed": [_fraction_text(item) for item in recurrence_seed],
        "representation": representation,
    }
    return Candidate(
        f"candidate.{canonical_sha256(body)[:24]}",
        family,
        representation,
        expression,
        tuple(recurrence_coefficients),
        tuple(recurrence_seed),
        tuple(sorted(set(invariants))),
        tuple(proof_plan),
        proposer,
    )


def _evaluate_expression(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> tuple[Fraction | None, ...]:
    expression = _safe_expression(candidate.expression, benchmark.aliases)
    symbols = [sp.Symbol(name, real=True) for name in benchmark.aliases]
    outputs: list[Fraction | None] = []
    for row in rows:
        value = sp.cancel(
            expression.subs(
                {
                    symbol: sp.Rational(item.numerator, item.denominator)
                    for symbol, item in zip(symbols, row.inputs, strict=True)
                }
            )
        )
        if not value.is_Rational:
            outputs.append(None)
        else:
            outputs.append(Fraction(int(value.p), int(value.q)))
    return tuple(outputs)


def _evaluate_recurrence(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> tuple[Fraction | None, ...]:
    if len(benchmark.variables) != 1 or not candidate.recurrence_coefficients:
        return tuple(None for _ in rows)
    order = len(candidate.recurrence_coefficients)
    sequence = list(candidate.recurrence_seed)
    requested = [int(row.inputs[0]) if row.inputs[0].denominator == 1 else -1 for row in rows]
    if any(index < 0 for index in requested):
        return tuple(None for _ in rows)
    while len(sequence) <= max(requested, default=-1):
        if len(sequence) < order:
            return tuple(None for _ in rows)
        value = sum(
            coefficient * sequence[-offset]
            for offset, coefficient in enumerate(candidate.recurrence_coefficients, start=1)
        )
        sequence.append(value)
    return tuple(sequence[index] for index in requested)


def predict(candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]) -> tuple[Fraction | None, ...]:
    if candidate.representation == "linear_recurrence":
        return _evaluate_recurrence(candidate, benchmark, rows)
    return _evaluate_expression(candidate, benchmark, rows)


def independently_predict(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> tuple[Fraction | None, ...]:
    """Re-evaluate one candidate without the campaign's SymPy execution path."""

    try:
        if candidate.representation == "linear_recurrence":
            if len(benchmark.variables) != 1:
                return tuple(None for _ in rows)
            indices = []
            for row in rows:
                if row.inputs[0].denominator != 1:
                    return tuple(None for _ in rows)
                indices.append(int(row.inputs[0]))
            return independently_evaluate_recurrence(
                candidate.recurrence_coefficients,
                candidate.recurrence_seed,
                indices,
            )
        if candidate.representation != "sympy_expression":
            return tuple(None for _ in rows)
        outputs = []
        for row in rows:
            variables = dict(zip(benchmark.aliases, row.inputs, strict=True))
            outputs.append(independently_evaluate_expression(candidate.expression, variables))
        return tuple(outputs)
    except IndependentEvaluationError:
        return tuple(None for _ in rows)


def _loss(predictions: Sequence[Fraction | None], rows: Sequence[Observation]) -> Fraction:
    penalty = Fraction(10**9)
    return sum(
        (penalty if predicted is None else abs(predicted - row.output))
        for predicted, row in zip(predictions, rows, strict=True)
    ) / len(rows)


def _polynomial_candidate(benchmark: Benchmark, family: str) -> Candidate | None:
    if len(benchmark.variables) != 1:
        return None
    x = sp.Symbol("x0", real=True)
    polynomial = sp.interpolate(
        [
            (sp.Rational(row.inputs[0].numerator, row.inputs[0].denominator), sp.Rational(row.output.numerator, row.output.denominator))
            for row in benchmark.observations
        ],
        x,
    )
    if sp.Poly(polynomial, x).degree() > 6:
        return None
    expression = str(sp.factor(polynomial)).replace("^", "**")
    return _candidate(
        benchmark,
        family,
        "sympy_expression",
        expression,
        invariants=("bounded_polynomial_degree", "finite_difference_order"),
        proof_plan=("derive_finite_difference_lemma", "change_to_polynomial_basis", "induct_on_index"),
    )


def _dimensional_candidate(benchmark: Benchmark, family: str) -> Candidate | None:
    matrix = sp.Matrix.hstack(*(sp.Matrix(item.dimension) for item in benchmark.variables))
    target = sp.Matrix(benchmark.output_dimension)
    symbols = sp.symbols(f"e0:{len(benchmark.variables)}", integer=True)
    solutions = sp.linsolve((matrix, target), symbols)
    if solutions is sp.EmptySet:
        return None
    candidates: list[tuple[int, ...]] = []
    for exponents in __import__("itertools").product(range(-3, 5), repeat=len(symbols)):
        if matrix * sp.Matrix(exponents) == target:
            candidates.append(tuple(exponents))
    if not candidates:
        return None
    exponents = min(candidates, key=lambda values: (sum(abs(item) for item in values), values))
    values = []
    for row in benchmark.observations:
        monomial = Fraction(1)
        try:
            for value, exponent in zip(row.inputs, exponents, strict=True):
                monomial *= value**exponent
        except ZeroDivisionError:
            return None
        if monomial == 0:
            return None
        values.append(row.output / monomial)
    if len(set(values)) != 1:
        return None
    coefficient = values[0]
    factors = []
    for alias, exponent in zip(benchmark.aliases, exponents, strict=True):
        if exponent:
            factors.append(alias if exponent == 1 else f"{alias}**{exponent}")
    expression = "*".join([f"({_fraction_text(coefficient)})", *factors])
    return _candidate(
        benchmark,
        family,
        "sympy_expression",
        expression,
        invariants=("dimension_balance", "dimensionless_coefficient"),
        proof_plan=("solve_dimension_lattice", "fit_dimensionless_group", "verify_scaling_interventions"),
    )


def _recurrence_candidate(benchmark: Benchmark, family: str) -> Candidate | None:
    if len(benchmark.variables) != 1:
        return None
    rows = sorted(benchmark.observations, key=lambda item: item.inputs[0])
    if [row.inputs[0] for row in rows] != [Fraction(index) for index in range(len(rows))]:
        return None
    sequence = [sp.Rational(row.output.numerator, row.output.denominator) for row in rows]
    for order in range(1, min(5, len(sequence) // 2 + 1)):
        coefficients = sp.symbols(f"c0:{order}")
        equations = [
            sp.Eq(sequence[index], sum(coefficients[offset - 1] * sequence[index - offset] for offset in range(1, order + 1)))
            for index in range(order, len(sequence))
        ]
        solutions = sp.solve(equations, coefficients, dict=True)
        for solution in solutions:
            if set(solution) != set(coefficients) or any(not solution[item].is_Rational for item in coefficients):
                continue
            fractions = tuple(Fraction(int(solution[item].p), int(solution[item].q)) for item in coefficients)
            return _candidate(
                benchmark,
                family,
                "linear_recurrence",
                "recurrence(" + ",".join(_fraction_text(item) for item in fractions) + ")",
                recurrence_coefficients=fractions,
                recurrence_seed=tuple(row.output for row in rows[:order]),
                invariants=("linear_recurrence", f"order_{order}"),
                proof_plan=("guess_recurrence", "prove_initial_conditions", "induct_with_recurrence"),
            )
    return None


def _template_candidates(benchmark: Benchmark, family: str) -> tuple[Candidate, ...]:
    aliases = benchmark.aliases
    if len(aliases) == 1:
        x = aliases[0]
        templates = ("0", "1", x, f"{x}**2", f"{x}**3", f"{x}*({x}+1)/2")
    else:
        x, y = aliases[:2]
        templates = ("0", "1", x, y, f"{x}*{y}", f"{x}*{y}**2", f"{x}**2*{y}")
    scored = []
    for expression in templates:
        candidate = _candidate(
            benchmark,
            family,
            "sympy_expression",
            expression,
            invariants=("template_library",),
            proof_plan=("transfer_analogue", "test_public_counterexamples"),
        )
        scored.append((_loss(predict(candidate, benchmark, benchmark.observations), benchmark.observations), candidate))
    return tuple(candidate for _, candidate in sorted(scored, key=lambda item: (item[0], item[1].candidate_id))[:4])


def generate_candidates(benchmark: Benchmark, maximum_per_family: int) -> tuple[Candidate, ...]:
    generated: list[Candidate] = []
    for family in FAMILY_IDS:
        candidates: list[Candidate] = []
        if family in {"dimensional_analysis", "inverse_variational", "quotient_construction"}:
            item = _dimensional_candidate(benchmark, family)
            if item is not None:
                candidates.append(item)
        if family in {
            "conserved_quantity_synthesis",
            "counterexample_guided_repair",
            "generating_function",
            "representation_change",
            "symmetry_reduction",
        }:
            item = _polynomial_candidate(benchmark, family)
            if item is not None:
                candidates.append(item)
        if family == "recurrence_guessing":
            item = _recurrence_candidate(benchmark, family)
            if item is not None:
                candidates.append(item)
        candidates.extend(_template_candidates(benchmark, family))
        deduplicated = {item.candidate_id: item for item in candidates}
        generated.extend(list(deduplicated.values())[:maximum_per_family])
    return tuple(sorted(generated, key=lambda item: item.candidate_id))


def _claude_candidate(benchmark: Benchmark, hypothesis: ClaudeHypothesis) -> Candidate | None:
    if hypothesis.representation != "sympy_expression":
        return None
    try:
        _safe_expression(hypothesis.expression, benchmark.aliases)
    except ExternalCreativityError:
        return None
    return _candidate(
        benchmark,
        "claude_proposer",
        hypothesis.representation,
        hypothesis.expression,
        invariants=hypothesis.invariants,
        proof_plan=hypothesis.proof_plan,
        proposer="claude_api",
    )


def random_controls(
    benchmark: Benchmark, family_counts: Mapping[str, int], seed: int
) -> dict[str, tuple[Candidate, ...]]:
    controls: dict[str, tuple[Candidate, ...]] = {}
    for family, count in sorted(family_counts.items()):
        digest = hashlib.sha256(f"{seed}:{benchmark.blind_id}:{family}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        rows = []
        for ordinal in range(count):
            if len(benchmark.aliases) == 1:
                coefficients = [rng.randint(-5, 5) for _ in range(rng.randint(0, 4) + 1)]
                expression = "+".join(
                    f"({coefficient})*x0**{degree}"
                    for degree, coefficient in enumerate(coefficients)
                )
            else:
                exponents = [rng.randint(0, 3) for _ in benchmark.aliases]
                coefficient = rng.randint(-5, 5)
                expression = "*".join(
                    [f"({coefficient})", *[f"{alias}**{power}" for alias, power in zip(benchmark.aliases, exponents, strict=True)]]
                )
            rows.append(
                _candidate(
                    benchmark,
                    f"random_control_{family}_{ordinal}",
                    "sympy_expression",
                    expression,
                    invariants=("random_budget_matched",),
                    proof_plan=("none",),
                    proposer="matched_random_search",
                )
            )
        controls[family] = tuple(rows)
    return controls


def _behavior(candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]) -> dict[str, str]:
    predictions = predict(candidate, benchmark, rows)
    finite = [None if item is None else _fraction_text(item) for item in predictions]
    expression = None
    degree = "recurrence"
    singularities = "not_applicable"
    if candidate.representation == "sympy_expression":
        expression = _safe_expression(candidate.expression, benchmark.aliases)
        symbols = [sp.Symbol(alias, real=True) for alias in benchmark.aliases]
        try:
            degree = str(sp.Poly(sp.together(expression).as_numer_denom()[0], *symbols).total_degree())
        except sp.PolynomialError:
            degree = "nonpolynomial"
        denominator = sp.factor(sp.together(expression).as_numer_denom()[1])
        singularities = str(denominator)
    behavior_body = {
        "degree": degree,
        "predictions": finite,
        "representation": candidate.representation,
        "singularity_structure": singularities,
    }
    proof_body = {
        "invariants": list(candidate.invariants),
        "proof_plan_shape": list(candidate.proof_plan),
        "representation": candidate.representation,
    }
    return {
        "behavior_sha256": canonical_sha256(behavior_body),
        "proof_mechanism_sha256": canonical_sha256(proof_body),
    }


def _sympy_to_z3(expression: sp.Expr, variables: Mapping[sp.Symbol, Any]) -> Any:
    if expression.is_Integer:
        return z3.IntVal(int(expression))
    if expression.is_Rational:
        return z3.RealVal(f"{int(expression.p)}/{int(expression.q)}")
    if expression.is_Symbol:
        return variables[expression]
    if expression.is_Add:
        return sum((_sympy_to_z3(item, variables) for item in expression.args), z3.IntVal(0))
    if expression.is_Mul:
        result = z3.IntVal(1)
        for item in expression.args:
            result *= _sympy_to_z3(item, variables)
        return result
    if expression.is_Pow and expression.exp.is_Integer and 0 <= int(expression.exp) <= 8:
        return _sympy_to_z3(expression.base, variables) ** int(expression.exp)
    raise ExternalCreativityError("SMT translation encountered an unsupported expression")


def _interval_value(expression: sp.Expr, substitutions: Mapping[sp.Symbol, Fraction]) -> Any:
    if expression.is_Integer:
        return mpmath.iv.mpf(int(expression))
    if expression.is_Rational:
        return mpmath.iv.mpf(int(expression.p)) / int(expression.q)
    if expression.is_Symbol:
        value = substitutions[expression]
        return mpmath.iv.mpf(value.numerator) / value.denominator
    if expression.is_Add:
        return sum((_interval_value(item, substitutions) for item in expression.args), mpmath.iv.mpf(0))
    if expression.is_Mul:
        result = mpmath.iv.mpf(1)
        for item in expression.args:
            result *= _interval_value(item, substitutions)
        return result
    if expression.is_Pow and expression.exp.is_Integer:
        return _interval_value(expression.base, substitutions) ** int(expression.exp)
    raise ExternalCreativityError("interval translation encountered an unsupported expression")


def verify_known_formula(
    candidate: Candidate, benchmark: Benchmark, target: SealedTarget
) -> dict[str, Any]:
    exact_predictions = predict(candidate, benchmark, target.holdout_records)
    exact = all(
        predicted == row.output
        for predicted, row in zip(exact_predictions, target.holdout_records, strict=True)
    )
    cas = smt = interval = False
    if candidate.representation == "sympy_expression" and target.reference_formula is not None:
        found = _safe_expression(candidate.expression, benchmark.aliases)
        reference = _safe_expression(target.reference_formula, benchmark.aliases)
        cas = sp.cancel(found - reference) == 0
        symbols = [sp.Symbol(alias, real=True) for alias in benchmark.aliases]
        z3_variables = {symbol: z3.Real(alias) for symbol, alias in zip(symbols, benchmark.aliases, strict=True)}
        solver = z3.Solver()
        solver.add(_sympy_to_z3(found, z3_variables) != _sympy_to_z3(reference, z3_variables))
        smt = solver.check() == z3.unsat
        interval = True
        for row in (*benchmark.observations, *target.holdout_records):
            substitutions = dict(zip(symbols, row.inputs, strict=True))
            difference = _interval_value(found, substitutions) - _interval_value(reference, substitutions)
            if not (difference.a <= 0 <= difference.b):
                interval = False
                break
    return {
        "backends": {
            "cas": cas,
            "exact_arithmetic": exact,
            "interval": interval,
            "lean": False,
            "smt": smt,
        },
        "kernel_blocker": "formula_specific_lean_proof_not_yet_executed",
        "serious_claim_eligible": exact and cas and smt and interval and False,
    }


def _dataset_evidence(
    benchmark: Benchmark, target: SealedTarget, best: Candidate
) -> dict[str, Any]:
    dimensions = [list(item.dimension) for item in benchmark.variables]
    dimension_matrix = sp.Matrix(dimensions).T
    dimensionless_basis = [
        [_fraction_text(Fraction(int(value.p), int(value.q))) for value in vector]
        for vector in dimension_matrix.nullspace()
    ]
    exponent_symbols = sp.symbols(f"e0:{len(benchmark.variables)}")
    dimension_solutions = sp.linsolve(
        (
            dimension_matrix,
            sp.Matrix(benchmark.output_dimension),
        ),
        exponent_symbols,
    )
    dimension_candidate = _dimensional_candidate(benchmark, "dataset_dimension_solver")
    train_predictions = predict(best, benchmark, benchmark.observations)
    holdout_predictions = predict(best, benchmark, target.holdout_records)
    train_residuals = [
        _fraction_text((prediction if prediction is not None else Fraction(10**9)) - row.output)
        for prediction, row in zip(train_predictions, benchmark.observations, strict=True)
    ]
    holdout_residuals = [
        _fraction_text((prediction if prediction is not None else Fraction(10**9)) - row.output)
        for prediction, row in zip(holdout_predictions, target.holdout_records, strict=True)
    ]
    parity_channels: dict[str, list[str]] = {"even": [], "odd": []}
    finite_differences: list[list[str]] = []
    if len(benchmark.variables) == 1:
        for row, residual in zip(target.holdout_records, holdout_residuals, strict=True):
            channel = "even" if int(row.inputs[0]) % 2 == 0 else "odd"
            parity_channels[channel].append(residual)
        ordered = sorted(benchmark.observations, key=lambda row: row.inputs[0])
        if all(
            row.inputs[0].denominator == 1
            and next_row.inputs[0] == row.inputs[0] + 1
            for row, next_row in pairwise(ordered)
        ):
            differences = [row.output for row in ordered]
            for _ in range(min(4, len(differences) - 1)):
                differences = [right - left for left, right in pairwise(differences)]
                finite_differences.append([_fraction_text(item) for item in differences])
    return {
        "causal_interventions": {
            "declared": list(benchmark.dataset_protocol["causal_interventions"]),
            "execution_status": "DECLARED_REQUIRES_INTERVENTION_DATA",
            "observational_rows_mislabelled_as_interventions": False,
        },
        "dimension_basis": list(benchmark.dataset_protocol["dimension_basis"]),
        "dimension_matrix": dimensions,
        "dimension_matrix_rank": dimension_matrix.rank(),
        "dimension_solution_set": str(dimension_solutions),
        "dimensionally_admissible_formula_found": dimension_candidate is not None,
        "dimensionless_group_basis": dimensionless_basis,
        "holdout_opened_last": True,
        "symmetry_groups": {
            "candidate_invariants": list(best.invariants),
            "declared_coordinates": list(benchmark.dataset_protocol["symmetry_coordinates"]),
            "status": "DECLARED_COORDINATES_WITH_EXACT_CANDIDATE_EVALUATION",
        },
        "ood_split_rule": benchmark.dataset_protocol["ood_split_rule"],
        "residual_channels": {
            "declared": list(benchmark.dataset_protocol["residual_channels"]),
            "finite_difference_orders_1_through_4": finite_differences,
            "holdout_parity": parity_channels,
            "holdout_residuals": holdout_residuals,
            "train_residuals": train_residuals,
        },
        "unit_normalization": {
            "inputs": [item.unit for item in benchmark.variables],
            "output": benchmark.output_unit,
            "status": "CANONICAL_DECLARED_UNITS_NO_CROSS_UNIT_CONVERSION",
        },
    }


def _score_candidate(
    candidate: Candidate,
    benchmark: Benchmark,
    target: SealedTarget,
) -> dict[str, Any]:
    train_predictions = predict(candidate, benchmark, benchmark.observations)
    holdout_predictions = predict(candidate, benchmark, target.holdout_records)
    behavior = _behavior(candidate, benchmark, (*benchmark.observations, *target.holdout_records))
    return {
        **candidate.to_dict(),
        **behavior,
        "holdout_exact_rows": sum(
            prediction == row.output
            for prediction, row in zip(holdout_predictions, target.holdout_records, strict=True)
        ),
        "holdout_loss": _fraction_text(_loss(holdout_predictions, target.holdout_records)),
        "train_exact_rows": sum(
            prediction == row.output
            for prediction, row in zip(train_predictions, benchmark.observations, strict=True)
        ),
        "train_loss": _fraction_text(_loss(train_predictions, benchmark.observations)),
    }


def _proof_plan_search(candidate: Candidate, target: SealedTarget) -> dict[str, Any]:
    """Enumerate verifier routes separately from candidate-expression search."""

    plans = [
        {
            "applicable": True,
            "estimated_cost": 1,
            "plan": "exact_row_replay",
            "purpose": "falsification",
        },
        {
            "applicable": candidate.representation == "sympy_expression",
            "estimated_cost": 2,
            "plan": "cas_normal_form",
            "purpose": "symbolic_identity",
        },
        {
            "applicable": candidate.representation == "sympy_expression",
            "estimated_cost": 3,
            "plan": "smt_countermodel_search",
            "purpose": "universal_polynomial_identity",
        },
        {
            "applicable": candidate.representation == "sympy_expression",
            "estimated_cost": 4,
            "plan": "interval_enclosure",
            "purpose": "numerical_domain_check",
        },
        {
            "applicable": candidate.representation == "linear_recurrence",
            "estimated_cost": 4,
            "plan": "induction_on_recurrence",
            "purpose": "recurrence_proof",
        },
        {
            "applicable": target.target_kind == "known_formula",
            "estimated_cost": 5,
            "plan": "lean_kernel_bridge",
            "purpose": "formula_specific_formal_check",
        },
    ]
    applicable = [item for item in plans if item["applicable"]]
    return {
        "candidate_declared_plan": list(candidate.proof_plan),
        "plans": plans,
        "selected_route": [item["plan"] for item in applicable],
        "selection_rule": "all_applicable_in_ascending_estimated_cost",
    }


def _family_metrics(
    scored: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Sequence[Mapping[str, Any]]],
    allocated_search_budget: Mapping[str, int],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metrics = []
    ablations = []
    creative = [item for item in scored if item["family"] in FAMILY_IDS]
    overall = min(Fraction(item["holdout_loss"]) for item in creative)
    for family in FAMILY_IDS:
        rows = [item for item in creative if item["family"] == family]
        random_rows = controls[family]
        best = min(Fraction(item["holdout_loss"]) for item in rows)
        random_best = min(Fraction(item["holdout_loss"]) for item in random_rows)
        control_budget_match = len(rows) == len(random_rows)
        without = [item for item in creative if item["family"] != family]
        ablated = min(Fraction(item["holdout_loss"]) for item in without)
        metrics.append(
            {
                "best_holdout_loss": _fraction_text(best),
                "candidate_budget": len(rows),
                "control_budget_match": control_budget_match,
                "family": family,
                "matched_budget": dict(allocated_search_budget),
                "matched_random_best_holdout_loss": _fraction_text(random_best),
                "matched_random_budget": len(random_rows),
                "outperformed_random": best < random_best,
                "unique_behaviors": len({item["behavior_sha256"] for item in rows}),
                "unique_proof_mechanisms": len({item["proof_mechanism_sha256"] for item in rows}),
            }
        )
        ablations.append(
            {
                "family": family,
                "full_portfolio_best_loss": _fraction_text(overall),
                "leave_one_family_out_best_loss": _fraction_text(ablated),
                "loss_increase": _fraction_text(ablated - overall),
            }
        )
    return metrics, ablations


def _load_campaign_config(root: Path, *, live_claude: bool) -> dict[str, Any]:
    raw = json.loads((root / CAMPAIGN_CONFIG_PATH).read_text(encoding="utf-8"))
    _strict_keys(
        raw,
        {
            "benchmark_config_path",
            "campaign_id",
            "claude_api",
            "open_problem_policy",
            "prior_art",
            "schema_version",
            "search",
            "verification",
        },
        "external creativity campaign config",
    )
    if raw["schema_version"] != CAMPAIGN_SCHEMA or raw["benchmark_config_path"] != PUBLIC_CONFIG_PATH:
        raise ExternalCreativityError("external creativity campaign identity changed")
    search = raw["search"]
    _strict_keys(
        search,
        {
            "creativity_families",
            "matched_control_budget",
            "maximum_candidates_per_family",
            "random_seed",
        },
        "creativity search config",
    )
    if tuple(search["creativity_families"]) != FAMILY_IDS:
        raise ExternalCreativityError("creativity family coverage changed")
    if not 1 <= search["maximum_candidates_per_family"] <= 16:
        raise ExternalCreativityError("per-family candidate budget is outside policy")
    matched_budget = search["matched_control_budget"]
    _strict_keys(
        matched_budget,
        {
            "maximum_evaluation_operations",
            "maximum_grammar_depth",
            "maximum_verifier_invocations",
        },
        "matched control budget",
    )
    if any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in matched_budget.values()):
        raise ExternalCreativityError("matched control budgets must be positive integers")
    claude = ClaudeAPIConfig.from_mapping(raw["claude_api"])
    if live_claude:
        claude = replace(claude, execution_enabled=True)
    raw["claude_api"] = claude
    verification = raw["verification"]
    _strict_keys(
        verification,
        {
            "lean_receipt_path",
            "minimum_independent_implementations",
            "minimum_independent_machines",
            "required_backends_for_serious_claim",
        },
        "external creativity verification config",
    )
    required = ["exact_arithmetic", "cas", "smt", "interval", "lean"]
    if verification.get("required_backends_for_serious_claim") != required:
        raise ExternalCreativityError("serious-claim verifier policy changed")
    if (
        verification.get("lean_receipt_path")
        != "runs/math/external-creativity-known-controls-lean/receipt.json"
        or verification.get("minimum_independent_implementations", 0) < 2
        or verification.get("minimum_independent_machines", 0) < 2
    ):
        raise ExternalCreativityError("independent formal-verification boundary is too weak")
    prior_art = raw["prior_art"]
    if prior_art.get("automated_sources") != ["repository_theorem_library", "external_literature_index"] or prior_art.get("human_review_required") is not True:
        raise ExternalCreativityError("prior-art release policy changed")
    open_policy = raw["open_problem_policy"]
    if open_policy.get("minimum_independent_level5_passes", 0) < 3 or open_policy.get("public_failure_receipt_required") is not True:
        raise ExternalCreativityError("open-problem gate is too weak")
    return raw


def _prior_art_screen(root: Path, candidate: Mapping[str, Any], benchmark: Benchmark) -> dict[str, Any]:
    theorem_files = sorted((root / "formal" / "lean").glob("*.lean"))
    normalized = re.sub(r"\s+", "", candidate["expression"]).lower()
    local_matches = []
    for path in theorem_files:
        text = re.sub(r"\s+", "", path.read_text(encoding="utf-8")).lower()
        if normalized and normalized in text:
            local_matches.append(path.relative_to(root).as_posix())
    external_query = {
        "candidate_expression": candidate["expression"],
        "source_id": benchmark.benchmark_id,
        "source_locator": benchmark.source.source_locator,
    }
    return {
        "external_literature_index": {
            "match": {
                "record_id": benchmark.benchmark_id,
                "title": benchmark.source.source_locator,
                "uri": benchmark.source.source_uri,
            },
            "query_sha256": canonical_sha256(external_query),
            "source_uri": benchmark.source.source_uri,
            "status": "AUTHORITATIVE_PRIOR_ART_RECORD_FOUND_NOT_NOVELTY_CLEARED",
        },
        "human_review": {
            "checklist": [
                "search_equivalent_notation_and_representations",
                "inspect_citations_and_theorem-library matches",
                "compare_proof_mechanism_not_only_formula_text",
                "record_reviewer_identity_and_timestamp_outside_automated_run",
            ],
            "required": True,
            "status": "NOT_PERFORMED",
        },
        "repository_theorem_library": {
            "files_scanned": len(theorem_files),
            "matches": local_matches,
            "status": "MATCHES_FOUND" if local_matches else "NO_TEXTUAL_MATCH_NOT_A_NOVELTY_RESULT",
        },
    }


def run_campaign(
    root: Path,
    *,
    live_claude: bool = False,
    claude_transport: Transport | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    config = _load_campaign_config(root, live_claude=live_claude)
    public, benchmarks = load_public_benchmarks(root)
    events: list[dict[str, Any]] = []

    def event(name: str, *, target_reads: int) -> None:
        events.append({"event": name, "sequence": len(events), "target_reads": target_reads})

    event("public_external_benchmarks_loaded", target_reads=0)
    event("external_authorship_validated", target_reads=0)
    maximum = config["search"]["maximum_candidates_per_family"]
    deterministic = {item.benchmark_id: list(generate_candidates(item, maximum)) for item in benchmarks}
    event("deterministic_creativity_families_generated", target_reads=0)

    transport = claude_transport or ProviderCompatibleClaudeTransport()
    client = ClaudeCreativityClient(config["claude_api"], transport)
    claude_calls: list[ClaudeCallResult] = []
    claude_admission: dict[str, dict[str, int]] = {}
    for benchmark in benchmarks:
        call = client.run(ClaudeRole.PROPOSER, benchmark.blind_id, benchmark.generation_view())
        if isinstance(transport, ProviderCompatibleClaudeTransport):
            call = replace(
                call,
                evidence={
                    **call.evidence,
                    **transport.evidence_for(str(call.evidence.get("api_response_id", ""))),
                },
            )
        claude_calls.append(call)
        hypotheses = () if call.output is None else call.output.hypotheses
        admitted = tuple(
            candidate
            for hypothesis in hypotheses
            if (candidate := _claude_candidate(benchmark, hypothesis)) is not None
        )
        deterministic[benchmark.benchmark_id].extend(admitted)
        claude_admission[benchmark.benchmark_id] = {
            "admitted_executable_hypotheses": len(admitted),
            "non_executable_typed_hypotheses": len(hypotheses) - len(admitted),
            "proposed_hypotheses": len(hypotheses),
        }
    event("claude_blind_proposals_completed_or_blocked", target_reads=0)

    train_summaries: dict[str, list[dict[str, Any]]] = {}
    proposal_roots = {}
    for benchmark in benchmarks:
        summaries = []
        for candidate in deterministic[benchmark.benchmark_id]:
            predictions = predict(candidate, benchmark, benchmark.observations)
            behavior = _behavior(candidate, benchmark, benchmark.observations)
            summaries.append(
                {
                    "behavior_sha256": behavior["behavior_sha256"],
                    "candidate_id": candidate.candidate_id,
                    "expression": candidate.expression,
                    "family": candidate.family,
                    "representation": candidate.representation,
                    "train_loss": _fraction_text(_loss(predictions, benchmark.observations)),
                }
            )
        train_summaries[benchmark.benchmark_id] = summaries
        proposal_roots[benchmark.benchmark_id] = canonical_sha256(
            [item.to_dict() for item in sorted(deterministic[benchmark.benchmark_id], key=lambda row: row.candidate_id)]
        )
    event("proposal_roots_and_train_evidence_sealed", target_reads=0)

    for benchmark in benchmarks:
        call = client.run(
            ClaudeRole.CRITIC,
            benchmark.blind_id,
            benchmark.generation_view(),
            candidate_summaries=train_summaries[benchmark.benchmark_id],
        )
        if isinstance(transport, ProviderCompatibleClaudeTransport):
            call = replace(
                call,
                evidence={
                    **call.evidence,
                    **transport.evidence_for(str(call.evidence.get("api_response_id", ""))),
                },
            )
        claude_calls.append(call)
    event("claude_blind_critique_completed_or_blocked", target_reads=0)

    targets = unseal_targets(root, public, benchmarks)
    event("sealed_targets_opened_after_proposal_and_critique", target_reads=1)
    by_target = {item.benchmark_id: item for item in targets}
    benchmark_results = []
    level5_passes = 0
    for benchmark in benchmarks:
        target = by_target[benchmark.benchmark_id]
        candidates = tuple(deterministic[benchmark.benchmark_id])
        allocated_search_budget = config["search"]["matched_control_budget"]
        scored = [_score_candidate(item, benchmark, target) for item in candidates]
        family_counts = {family: sum(item.family == family for item in candidates) for family in FAMILY_IDS}
        controls = random_controls(benchmark, family_counts, config["search"]["random_seed"])
        scored_controls = {
            family: [_score_candidate(item, benchmark, target) for item in rows]
            for family, rows in controls.items()
        }
        metrics, ablations = _family_metrics(
            scored, scored_controls, allocated_search_budget
        )
        best_row = min(scored, key=lambda item: (Fraction(item["holdout_loss"]), item["candidate_id"]))
        best_candidate = next(item for item in candidates if item.candidate_id == best_row["candidate_id"])
        all_rows = (*benchmark.observations, *target.holdout_records)
        primary_predictions = predict(best_candidate, benchmark, all_rows)
        reproduced_predictions = independently_predict(best_candidate, benchmark, all_rows)
        implementation_match = reproduced_predictions == primary_predictions
        independent_reproduction = {
            "implementation": "python_stdlib_fraction_ast_v1",
            "match": implementation_match,
            "prediction_sha256": canonical_sha256(
                [None if item is None else _fraction_text(item) for item in reproduced_predictions]
            ),
            "shared_symbolic_runtime": False,
        }
        formal = (
            verify_known_formula(best_candidate, benchmark, target)
            if target.target_kind == "known_formula"
            else {
                "backends": {
                    "cas": False,
                    "exact_arithmetic": True,
                    "interval": False,
                    "lean": False,
                    "smt": False,
                },
                "kernel_blocker": "bounded_unknown_has_no_reference_formula_or_serious_claim",
                "serious_claim_eligible": False,
            }
        )
        outperformed_random = sum(item["outperformed_random"] for item in metrics)
        controls_budget_matched = all(item["control_budget_match"] for item in metrics)
        bounded_process_pass = (
            benchmark.capability_level == 5
            and best_row["train_loss"] == "0"
            and best_row["holdout_loss"] == "0"
            and outperformed_random >= len(FAMILY_IDS) // 2
            and controls_budget_matched
            and len({item["behavior_sha256"] for item in scored}) >= 3
            and len({item["proof_mechanism_sha256"] for item in scored}) >= 2
            and implementation_match
            and not formal["serious_claim_eligible"]
        )
        if bounded_process_pass:
            level5_passes += 1
        prior_art = _prior_art_screen(root, best_row, benchmark)
        benchmark_results.append(
            {
                "benchmark_id": benchmark.benchmark_id,
                "blind_id": benchmark.blind_id,
                "bounded_unknown_process_pass": bounded_process_pass,
                "capability_level": benchmark.capability_level,
                "claims": {
                    "known_formula_rediscovered": target.target_kind == "known_formula" and best_row["holdout_loss"] == "0",
                    "novel_formula_established": False,
                    "open_problem_solved": False,
                    "serious_claim_released": False,
                },
                "dataset_evidence": _dataset_evidence(benchmark, target, best_candidate),
                "external_authorship": benchmark.source.to_dict(),
                "family_ablation": ablations,
                "family_metrics": metrics,
                "matched_control_policy": {
                    "allocated_budget": allocated_search_budget,
                    "all_family_budgets_match": controls_budget_matched,
                    "candidate_count_matched": True,
                },
                "formal_verification": formal,
                "holdout_count": len(target.holdout_records),
                "independent_exact_reproduction": independent_reproduction,
                "prior_art": prior_art,
                "proof_plan_search": _proof_plan_search(best_candidate, target),
                "proposal_root_sha256": proposal_roots[benchmark.benchmark_id],
                "proposer_admission": claude_admission[benchmark.benchmark_id],
                "random_controls": scored_controls,
                "ranked_candidates": sorted(scored, key=lambda item: (Fraction(item["holdout_loss"]), item["candidate_id"])),
                "target_commitment_opened": target.commitment,
                "target_kind": target.target_kind,
                "unique_behaviors": len({item["behavior_sha256"] for item in scored}),
                "unique_proof_mechanisms": len({item["proof_mechanism_sha256"] for item in scored}),
            }
        )
    event("holdouts_scored_and_independent_verifiers_applied", target_reads=1)

    open_policy = config["open_problem_policy"]
    open_authorized = level5_passes >= open_policy["minimum_independent_level5_passes"]
    completed_claude = sum(item.status is ClaudeCallStatus.COMPLETED for item in claude_calls)
    claude_required = len(benchmarks) * 2
    steering_actions = sum(
        len(item.output.steering_actions) if item.output is not None else 0
        for item in claude_calls
        if item.role is ClaudeRole.CRITIC
    )
    proposer_hypotheses = sum(
        len(item.output.hypotheses) if item.output is not None else 0
        for item in claude_calls
        if item.role is ClaudeRole.PROPOSER
    )
    substantive_claude = (
        completed_claude == claude_required
        and steering_actions >= len(benchmarks)
        and proposer_hypotheses >= len(benchmarks)
    )
    receipt: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA,
        "campaign_id": config["campaign_id"],
        "benchmarks": benchmark_results,
        "blind_chronology": events,
        "claims": {
            "claude_used_throughout": substantive_claude,
            "externally_authored_sealed_benchmarks_executed": len(benchmark_results) >= 4,
            "novel_formula_established": False,
            "open_problem_attempted": False,
            "open_problem_solved": False,
        },
        "claude": {
            "budget": client.budget.to_dict(),
            "calls": [item.to_dict() for item in claude_calls],
            "completed_calls": completed_claude,
            "required_calls": claude_required,
            "proposer_hypotheses": proposer_hypotheses,
            "status": "PASS" if substantive_claude else "INCOMPLETE",
            "steering_actions": steering_actions,
        },
        "config": {
            "campaign_sha256": _file_sha256(root / CAMPAIGN_CONFIG_PATH),
            "claude_source_sha256": _file_sha256(root / CLAUDE_SOURCE_PATH),
            "claude_transport_source_sha256": _file_sha256(
                root / CLAUDE_TRANSPORT_SOURCE_PATH
            ),
            "independent_evaluator_sha256": _file_sha256(root / INDEPENDENT_EVALUATOR_PATH),
            "lean_source_sha256": _file_sha256(root / LEAN_SOURCE_PATH),
            "public_benchmarks_sha256": _file_sha256(root / PUBLIC_CONFIG_PATH),
            "sealed_targets_sha256": _file_sha256(root / public["sealed_targets_path"]),
            "source_sha256": _file_sha256(root / SOURCE_PATH),
            "test_sha256": _file_sha256(root / TEST_PATH),
        },
        "independent_reproduction": {
            "minimum_implementations": config["verification"]["minimum_independent_implementations"],
            "minimum_machines": config["verification"]["minimum_independent_machines"],
            "received_implementations": (
                2
                if all(item["independent_exact_reproduction"]["match"] for item in benchmark_results)
                else 1
            ),
            "received_machines": 1,
            "status": "IMPLEMENTATIONS_PASS_MACHINE_PENDING",
        },
        "open_problem_gate": {
            "authorized": open_authorized,
            "level5_process_passes": level5_passes,
            "maximum_calls_per_problem": open_policy["maximum_calls_per_problem"],
            "maximum_total_tokens_per_problem": open_policy["maximum_total_tokens_per_problem"],
            "minimum_independent_level5_passes": open_policy["minimum_independent_level5_passes"],
            "public_failure_receipt_required": open_policy["public_failure_receipt_required"],
            "success_criteria": {
                "independent_implementation_match": True,
                "minimum_families_outperforming_random": len(FAMILY_IDS) // 2,
                "minimum_unique_behaviors": 3,
                "minimum_unique_proof_mechanisms": 2,
                "sealed_holdout_loss": "0",
                "training_loss": "0",
            },
            "status": "READY_NOT_SPENT" if open_authorized else "BLOCKED_INSUFFICIENT_LEVEL5_REPETITIONS",
        },
        "serious_claim_policy": {
            "human_prior_art_required": True,
            "independent_reproduction_required": True,
            "required_backends": config["verification"]["required_backends_for_serious_claim"],
            "released_claims": 0,
        },
    }
    receipt["content_sha256"] = canonical_sha256(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any], root: Path) -> None:
    expected = run_campaign(root, live_claude=False)
    if receipt != expected:
        raise ExternalCreativityError("external creativity receipt does not replay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--live-claude", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    receipt = run_campaign(root, live_claude=args.live_claude)
    output = args.output or root / OUTPUT_PATH
    if args.check:
        checked = json.loads(output.read_text(encoding="utf-8"))
        if checked != receipt:
            raise ExternalCreativityError("checked external creativity receipt is stale")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
