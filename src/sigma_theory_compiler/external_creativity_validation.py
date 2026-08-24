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
from math import prod
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
    ClaudeCreativityError,
    ClaudeHypothesis,
    ClaudeRole,
    Transport,
)
from .expanded_math_independent_evaluator import (
    ExpandedIndependentEvaluationError,
)
from .expanded_math_independent_evaluator import evaluate as independently_evaluate_expanded
from .expanded_math_primary_evaluator import ExpandedPrimaryEvaluationError
from .expanded_math_primary_evaluator import evaluate as primarily_evaluate_expanded
from .external_claude_transport import ProviderCompatibleClaudeTransport
from .independent_exact_evaluator import (
    IndependentEvaluationError,
)
from .independent_exact_evaluator import (
    evaluate_comparison as independently_evaluate_comparison,
)
from .independent_exact_evaluator import (
    evaluate_expression as independently_evaluate_expression,
)
from .independent_exact_evaluator import (
    evaluate_recurrence as independently_evaluate_recurrence,
)
from .math_expression_ir import (
    Expression,
    ExpressionIRError,
    TensorIdentity,
    VariationalFunctional,
)
from .math_expression_ir import (
    add as ir_add,
)
from .math_expression_ir import (
    literal as ir_literal,
)
from .math_expression_ir import (
    multiply as ir_multiply,
)
from .math_expression_ir import (
    power as ir_power,
)
from .math_expression_ir import (
    symbol as ir_symbol,
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
RECEIPT_SCHEMA = "invariant-external-creativity-validation-result-1.1"
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
EXECUTABLE_CLAUDE_REPRESENTATIONS = frozenset(
    {
        "finite_product",
        "finite_sum",
        "generating_function",
        "invariant_relation",
        "linear_recurrence",
        "modular_relation",
        "piecewise_relation",
        "sympy_expression",
        "tensor_identity",
        "transform_relation",
        "variational_principle",
    }
)
MAXIMUM_TYPED_TERMS = 64
MAXIMUM_RECURRENCE_ORDER = 8
MAXIMUM_PIECEWISE_BRANCHES = 8
MAXIMUM_TENSOR_COMPONENTS = 256
MAXIMUM_TENSOR_RANK = 4
MAXIMUM_TRANSFORM_TERMS = 16
EXECUTABLE_PROPOSER_INSTRUCTION = (
    "Propose structurally distinct mathematical hypotheses and proof plans. For "
    "each idea, self-assess whether it is a known rewrite, cross-domain synthesis, "
    "proposed new construction, or uncertain; name analogues and source domains. "
    "Uncertainty is welcome and no idea is pruned by this label. To request bounded "
    "execution, use exactly one admitted expression contract: sympy_expression uses "
    "arithmetic over the public x aliases only; invariant_relation uses 'output = "
    "<arithmetic>'; linear_recurrence uses JSON with coefficients and equally long "
    "seed arrays; finite_sum or finite_product uses JSON with body, index, lower, and "
    "upper; generating_function uses JSON with numerator and denominator coefficient "
    "arrays plus an index alias; modular_relation uses JSON with expression and integer "
    "modulus; piecewise_relation uses JSON with one to eight ordered branches, each "
    "containing condition {left, comparator, right} plus expression, and a mandatory "
    "default_expression; comparators are lt, le, eq, ne, ge, or gt; transform_relation "
    "uses JSON with transform_kind='linear_shift_stencil', "
    "a public index alias, source_expression, claimed_transform, and one to sixteen "
    "stencil terms containing exact rational coefficients and integer offsets; "
    "tensor_identity uses JSON with tensor_name, shape, variance, complete "
    "left_components and right_components arrays, symmetries, and an output_component; "
    "variational_principle uses JSON with field, coordinate, first_derivative, "
    "second_derivative, integrand, claimed_euler_lagrange, and bindings from all four "
    "formal symbols to arithmetic over public aliases. Put explanation only in rationale. "
    "Other typed ideas are retained for later compilers."
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
    return (
        str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"
    )


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
        if not re.fullmatch(
            r"20[0-9]{2}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}Z", value["retrieved_utc"]
        ):
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


def unseal_targets(
    root: Path, public: Mapping[str, Any], benchmarks: Sequence[Benchmark]
) -> tuple[SealedTarget, ...]:
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
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
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


def _arithmetic_tree_stats(expression: str, aliases: Sequence[str]) -> tuple[int, int]:
    """Return exact syntax-tree depth and node count for the admitted arithmetic DSL."""

    normalized = expression.replace("^", "**")
    _safe_expression(normalized, aliases)
    tree = ast.parse(normalized, mode="eval")

    def stats(node: ast.AST) -> tuple[int, int]:
        if isinstance(node, (ast.Constant, ast.Name)):
            return 1, 1
        if isinstance(node, ast.UnaryOp):
            depth, count = stats(node.operand)
            return depth + 1, count + 1
        if isinstance(node, ast.BinOp):
            left_depth, left_count = stats(node.left)
            right_depth, right_count = stats(node.right)
            return (
                1 + max(left_depth, right_depth),
                1 + left_count + right_count,
            )
        raise ExternalCreativityError("arithmetic syntax profiler left the admitted DSL")

    return stats(tree.body)


def _normalize_claude_arithmetic(expression: str, aliases: Sequence[str]) -> tuple[str, str]:
    """Normalize a small, declared set of harmless model notations.

    This is intentionally not a prose parser. It accepts an optional output assignment,
    maps the conventional single sequence variable n or x onto the public alias, and accepts
    an equality only when both sides independently parse and are algebraically identical.
    """

    text = expression.strip().replace("^", "**")
    normalization = "exact_dsl"
    assignment = re.fullmatch(r"output\s*=\s*(.+)", text, flags=re.DOTALL)
    if assignment is not None:
        text = assignment.group(1).strip()
        normalization = "output_assignment"
    if len(aliases) == 1:
        alias = aliases[0]
        for conventional in ("n", "x"):
            if conventional != alias and re.search(rf"\b{conventional}\b", text):
                text = re.sub(rf"\b{conventional}\b", alias, text)
                normalization = (
                    f"{normalization}+single_variable_alias"
                    if normalization != "exact_dsl"
                    else "single_variable_alias"
                )
    try:
        parsed = _safe_expression(text, aliases)
    except ExternalCreativityError as direct_error:
        if text.count("=") != 1:
            raise
        left_text, right_text = (item.strip() for item in text.split("=", 1))
        try:
            left = _safe_expression(left_text, aliases)
            right = _safe_expression(right_text, aliases)
        except ExternalCreativityError:
            raise direct_error from None
        if sp.cancel(left - right) != 0:
            raise ExternalCreativityError(
                "candidate equality sides are not algebraically identical"
            )
        parsed = left
        normalization = (
            f"{normalization}+equivalent_equality"
            if normalization != "exact_dsl"
            else "equivalent_equality"
        )
    canonical = str(sp.factor(parsed))
    _safe_expression(canonical, aliases)
    return canonical, normalization


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
            "recurrence_coefficients": [
                _fraction_text(item) for item in self.recurrence_coefficients
            ],
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


def _typed_json(expression: str, expected: set[str], label: str) -> dict[str, Any]:
    try:
        value = json.loads(expression)
    except json.JSONDecodeError as error:
        raise ExternalCreativityError(f"{label} must be canonical JSON") from error
    _strict_keys(value, expected, label)
    return dict(value)


def _bounded_fraction(value: Any, label: str) -> Fraction:
    result = _fraction(value, label)
    if abs(result.numerator) > 10**6 or result.denominator > 10**6:
        raise ExternalCreativityError(f"{label} exceeds the exact coefficient budget")
    return result


def _parse_recurrence_spec(
    expression: str,
) -> tuple[str, tuple[Fraction, ...], tuple[Fraction, ...]]:
    value = _typed_json(expression, {"coefficients", "seed"}, "recurrence specification")
    coefficients_raw = value["coefficients"]
    seed_raw = value["seed"]
    if (
        not isinstance(coefficients_raw, list)
        or not isinstance(seed_raw, list)
        or not 1 <= len(coefficients_raw) <= MAXIMUM_RECURRENCE_ORDER
        or len(seed_raw) != len(coefficients_raw)
    ):
        raise ExternalCreativityError("recurrence order or seed coverage changed")
    coefficients = tuple(
        _bounded_fraction(item, f"recurrence coefficient {index}")
        for index, item in enumerate(coefficients_raw)
    )
    seed = tuple(
        _bounded_fraction(item, f"recurrence seed {index}") for index, item in enumerate(seed_raw)
    )
    canonical = json.dumps(
        {
            "coefficients": [_fraction_text(item) for item in coefficients],
            "seed": [_fraction_text(item) for item in seed],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return canonical, coefficients, seed


def _parse_bound(value: Any, aliases: Sequence[str], label: str) -> str:
    if not isinstance(value, str):
        raise ExternalCreativityError(f"{label} must be an integer or public alias string")
    if value in aliases:
        return value
    parsed = _bounded_fraction(value, label)
    if parsed.denominator != 1:
        raise ExternalCreativityError(f"{label} is not an integer")
    return str(parsed.numerator)


def _parse_aggregate_spec(
    expression: str, aliases: Sequence[str], label: str
) -> tuple[str, dict[str, str]]:
    value = _typed_json(
        expression,
        {"body", "index", "lower", "upper"},
        f"{label} specification",
    )
    index = value["index"]
    body = value["body"]
    if (
        not isinstance(index, str)
        or re.fullmatch(r"[a-z][a-z0-9_]{0,15}", index) is None
        or index in aliases
        or not isinstance(body, str)
    ):
        raise ExternalCreativityError(f"{label} index or body is invalid")
    _safe_expression(body, (*aliases, index))
    spec = {
        "body": body.replace("^", "**"),
        "index": index,
        "lower": _parse_bound(value["lower"], aliases, f"{label} lower bound"),
        "upper": _parse_bound(value["upper"], aliases, f"{label} upper bound"),
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec


def _parse_generating_function_spec(
    expression: str, aliases: Sequence[str]
) -> tuple[str, dict[str, Any]]:
    value = _typed_json(
        expression,
        {"denominator", "index", "numerator"},
        "generating-function specification",
    )
    index = value["index"]
    numerator_raw = value["numerator"]
    denominator_raw = value["denominator"]
    if (
        not isinstance(index, str)
        or index not in aliases
        or not isinstance(numerator_raw, list)
        or not isinstance(denominator_raw, list)
        or not 1 <= len(numerator_raw) <= MAXIMUM_TYPED_TERMS
        or not 1 <= len(denominator_raw) <= MAXIMUM_TYPED_TERMS
    ):
        raise ExternalCreativityError("generating-function index or coefficients are invalid")
    numerator = tuple(
        _bounded_fraction(item, f"generating-function numerator {position}")
        for position, item in enumerate(numerator_raw)
    )
    denominator = tuple(
        _bounded_fraction(item, f"generating-function denominator {position}")
        for position, item in enumerate(denominator_raw)
    )
    if denominator[0] == 0:
        raise ExternalCreativityError("generating-function denominator has zero constant term")
    spec = {
        "denominator": [_fraction_text(item) for item in denominator],
        "index": index,
        "numerator": [_fraction_text(item) for item in numerator],
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec


def _parse_modular_spec(expression: str, aliases: Sequence[str]) -> tuple[str, dict[str, Any]]:
    value = _typed_json(expression, {"expression", "modulus"}, "modular specification")
    inner = value["expression"]
    modulus = value["modulus"]
    if (
        not isinstance(inner, str)
        or isinstance(modulus, bool)
        or not isinstance(modulus, int)
        or not 2 <= modulus <= 10**6
    ):
        raise ExternalCreativityError("modular expression or modulus is invalid")
    normalized = inner.strip().replace("^", "**")
    _safe_expression(normalized, aliases)
    spec = {"expression": normalized, "modulus": modulus}
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec


def _parse_piecewise_spec(expression: str, aliases: Sequence[str]) -> tuple[str, dict[str, Any]]:
    value = _typed_json(
        expression,
        {"branches", "default_expression"},
        "piecewise specification",
    )
    branches_raw = value["branches"]
    if (
        not isinstance(branches_raw, list)
        or not 1 <= len(branches_raw) <= MAXIMUM_PIECEWISE_BRANCHES
    ):
        raise ExternalCreativityError("piecewise branch count is outside the budget")
    branches: list[dict[str, Any]] = []
    comparators = {"eq", "ge", "gt", "le", "lt", "ne"}
    for position, branch in enumerate(branches_raw):
        _strict_keys(branch, {"condition", "expression"}, f"piecewise branch {position}")
        condition = branch["condition"]
        _strict_keys(
            condition,
            {"comparator", "left", "right"},
            f"piecewise branch {position} condition",
        )
        comparator = condition["comparator"]
        if comparator not in comparators:
            raise ExternalCreativityError("piecewise comparator is unsupported")
        branches.append(
            {
                "condition": {
                    "comparator": comparator,
                    "left": _canonical_typed_arithmetic(
                        condition["left"], aliases, f"piecewise branch {position} left operand"
                    ),
                    "right": _canonical_typed_arithmetic(
                        condition["right"], aliases, f"piecewise branch {position} right operand"
                    ),
                },
                "expression": _canonical_typed_arithmetic(
                    branch["expression"], aliases, f"piecewise branch {position} expression"
                ),
            }
        )
    spec = {
        "branches": branches,
        "default_expression": _canonical_typed_arithmetic(
            value["default_expression"], aliases, "piecewise default expression"
        ),
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec


def _typed_symbol(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,63}", value) is None:
        raise ExternalCreativityError(f"{label} is not a typed symbol")
    return value


def _canonical_typed_arithmetic(expression: Any, aliases: Sequence[str], label: str) -> str:
    if not isinstance(expression, str) or not expression.strip():
        raise ExternalCreativityError(f"{label} is not an arithmetic string")
    normalized = expression.strip().replace("^", "**")
    _safe_expression(normalized, aliases)
    try:
        return ast.unparse(ast.parse(normalized, mode="eval").body)
    except SyntaxError as error:  # pragma: no cover - _safe_expression already guards this
        raise ExternalCreativityError(f"{label} is not valid arithmetic") from error


def _parse_transform_spec(expression: str, aliases: Sequence[str]) -> tuple[str, dict[str, Any]]:
    value = _typed_json(
        expression,
        {
            "claimed_transform",
            "index",
            "source_expression",
            "stencil",
            "transform_kind",
        },
        "transform specification",
    )
    if value["transform_kind"] != "linear_shift_stencil":
        raise ExternalCreativityError("transform kind is not executable")
    index = value["index"]
    if not isinstance(index, str):
        raise ExternalCreativityError("transform index is not a string")
    if index in aliases:
        source = _canonical_typed_arithmetic(
            value["source_expression"], aliases, "transform source expression"
        )
        claimed = _canonical_typed_arithmetic(
            value["claimed_transform"], aliases, "claimed transform expression"
        )
    elif len(aliases) == 1 and index in {"n", "x"}:
        source, _ = _normalize_claude_arithmetic(value["source_expression"], aliases)
        claimed, _ = _normalize_claude_arithmetic(value["claimed_transform"], aliases)
        index = aliases[0]
    else:
        raise ExternalCreativityError("transform index is not a public or normalized alias")
    stencil_raw = value["stencil"]
    if not isinstance(stencil_raw, list) or not 1 <= len(stencil_raw) <= MAXIMUM_TRANSFORM_TERMS:
        raise ExternalCreativityError("transform stencil is outside the term budget")
    stencil: list[dict[str, Any]] = []
    offsets: set[int] = set()
    for position, term in enumerate(stencil_raw):
        _strict_keys(term, {"coefficient", "offset"}, f"transform stencil term {position}")
        offset = term["offset"]
        if (
            isinstance(offset, bool)
            or not isinstance(offset, int)
            or not -16 <= offset <= 16
            or offset in offsets
        ):
            raise ExternalCreativityError("transform stencil offsets are invalid or duplicated")
        coefficient = _bounded_fraction(
            term["coefficient"], f"transform stencil coefficient {position}"
        )
        if coefficient == 0:
            raise ExternalCreativityError("transform stencil coefficient cannot be zero")
        offsets.add(offset)
        stencil.append({"coefficient": _fraction_text(coefficient), "offset": offset})
    stencil.sort(key=lambda item: item["offset"])
    spec = {
        "claimed_transform": claimed,
        "index": index,
        "source_expression": source,
        "stencil": stencil,
        "transform_kind": "linear_shift_stencil",
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec


def _sympy_expression_ir(expression: sp.Expr) -> Expression:
    if expression.is_Integer:
        return ir_literal(int(expression))
    if expression.is_Rational:
        return ir_literal(Fraction(int(expression.p), int(expression.q)))
    if expression.is_Symbol:
        return ir_symbol(str(expression))
    if expression.is_Add:
        return ir_add(*(_sympy_expression_ir(item) for item in expression.args))
    if expression.is_Mul:
        return ir_multiply(*(_sympy_expression_ir(item) for item in expression.args))
    if expression.is_Pow and expression.exp.is_Integer:
        exponent = int(expression.exp)
        if not -8 <= exponent <= 8:
            raise ExternalCreativityError("typed arithmetic exponent is outside [-8, 8]")
        return ir_power(_sympy_expression_ir(expression.base), exponent)
    raise ExternalCreativityError("typed arithmetic cannot be represented in the exact IR")


def _expression_ir_from_text(expression: str, aliases: Sequence[str]) -> Expression:
    return _sympy_expression_ir(_safe_expression(expression, aliases))


def _parse_tensor_spec(
    expression: str, aliases: Sequence[str]
) -> tuple[str, dict[str, Any], TensorIdentity]:
    value = _typed_json(
        expression,
        {
            "left_components",
            "output_component",
            "right_components",
            "shape",
            "symmetries",
            "tensor_name",
            "variance",
        },
        "tensor specification",
    )
    tensor_name = _typed_symbol(value["tensor_name"], "tensor name")
    shape_raw = value["shape"]
    variance_raw = value["variance"]
    left_raw = value["left_components"]
    right_raw = value["right_components"]
    if (
        not isinstance(shape_raw, list)
        or not 1 <= len(shape_raw) <= MAXIMUM_TENSOR_RANK
        or any(
            isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 8
            for size in shape_raw
        )
    ):
        raise ExternalCreativityError("tensor shape is invalid")
    shape = tuple(shape_raw)
    component_count = prod(shape)
    if component_count > MAXIMUM_TENSOR_COMPONENTS:
        raise ExternalCreativityError("tensor exceeds the component budget")
    if (
        not isinstance(variance_raw, list)
        or len(variance_raw) != len(shape)
        or any(item not in {"covariant", "contravariant"} for item in variance_raw)
    ):
        raise ExternalCreativityError("tensor variance does not match its rank")
    if (
        not isinstance(left_raw, list)
        or not isinstance(right_raw, list)
        or len(left_raw) != component_count
        or len(right_raw) != component_count
    ):
        raise ExternalCreativityError("tensor components do not cover the declared shape")
    left = [
        _canonical_typed_arithmetic(item, aliases, f"tensor left component {index}")
        for index, item in enumerate(left_raw)
    ]
    right = [
        _canonical_typed_arithmetic(item, aliases, f"tensor right component {index}")
        for index, item in enumerate(right_raw)
    ]
    symmetries_raw = value["symmetries"]
    if not isinstance(symmetries_raw, list):
        raise ExternalCreativityError("tensor symmetries must be a list")
    symmetries: list[tuple[int, int, int]] = []
    for index, item in enumerate(symmetries_raw):
        _strict_keys(
            item,
            {"left_axis", "right_axis", "sign"},
            f"tensor symmetry {index}",
        )
        left_axis = item["left_axis"]
        right_axis = item["right_axis"]
        sign = item["sign"]
        if (
            isinstance(left_axis, bool)
            or isinstance(right_axis, bool)
            or not isinstance(left_axis, int)
            or not isinstance(right_axis, int)
            or isinstance(sign, bool)
            or sign not in {-1, 1}
        ):
            raise ExternalCreativityError("tensor symmetry is malformed")
        symmetries.append((left_axis, right_axis, sign))
    symmetry_tuples = tuple(sorted(set(symmetries)))
    output = value["output_component"]
    _strict_keys(output, {"flat_index", "side"}, "tensor output component")
    flat_index = output["flat_index"]
    side = output["side"]
    if (
        side not in {"left", "right"}
        or isinstance(flat_index, bool)
        or not isinstance(flat_index, int)
        or not 0 <= flat_index < component_count
    ):
        raise ExternalCreativityError("tensor output component is invalid")
    try:
        formula = TensorIdentity(
            tensor_name,
            shape,
            tuple(variance_raw),
            tuple(_expression_ir_from_text(item, aliases) for item in left),
            tuple(_expression_ir_from_text(item, aliases) for item in right),
            symmetry_tuples,
        )
    except ExpressionIRError as error:
        raise ExternalCreativityError(str(error)) from error
    spec = {
        "left_components": left,
        "output_component": {"flat_index": flat_index, "side": side},
        "right_components": right,
        "shape": list(shape),
        "symmetries": [
            {"left_axis": left_axis, "right_axis": right_axis, "sign": sign}
            for left_axis, right_axis, sign in symmetry_tuples
        ],
        "tensor_name": tensor_name,
        "variance": list(variance_raw),
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec, formula


def _parse_variational_spec(
    expression: str, aliases: Sequence[str]
) -> tuple[str, dict[str, Any], VariationalFunctional]:
    try:
        value = json.loads(expression)
    except json.JSONDecodeError as error:
        raise ExternalCreativityError("variational specification must be canonical JSON") from error
    required = {
        "bindings",
        "claimed_euler_lagrange",
        "coordinate",
        "field",
        "first_derivative",
        "integrand",
        "second_derivative",
    }
    if not isinstance(value, Mapping) or frozenset(value) not in {
        frozenset(required),
        frozenset((*required, "output_expression")),
    }:
        raise ExternalCreativityError("variational specification keys changed")
    field = _typed_symbol(value["field"], "variational field")
    coordinate = _typed_symbol(value["coordinate"], "variational coordinate")
    first_derivative = _typed_symbol(value["first_derivative"], "variational first derivative")
    second_derivative = _typed_symbol(value["second_derivative"], "variational second derivative")
    formal_symbols = (field, coordinate, first_derivative, second_derivative)
    if len(set(formal_symbols)) != 4 or set(formal_symbols) & set(aliases):
        raise ExternalCreativityError(
            "variational formal symbols must be distinct from public aliases"
        )
    allowed_symbols = (*formal_symbols, *aliases)
    integrand = _canonical_typed_arithmetic(
        value["integrand"], allowed_symbols, "variational integrand"
    )
    claimed = _canonical_typed_arithmetic(
        value["claimed_euler_lagrange"],
        allowed_symbols,
        "claimed Euler-Lagrange expression",
    )
    symbolic_variables = [sp.Symbol(name, real=True) for name in allowed_symbols]
    for label, item in (("integrand", integrand), ("claim", claimed)):
        try:
            polynomial = sp.Poly(_safe_expression(item, allowed_symbols), *symbolic_variables)
        except sp.PolynomialError as error:
            raise ExternalCreativityError(
                f"variational {label} leaves the polynomial evaluator"
            ) from error
        if any(degree > 8 for degree in polynomial.degree_list()):
            raise ExternalCreativityError(
                f"variational {label} exceeds the polynomial power budget"
            )
    bindings_raw = value["bindings"]
    if not isinstance(bindings_raw, Mapping) or set(bindings_raw) != set(formal_symbols):
        raise ExternalCreativityError("variational bindings do not cover every formal symbol")
    bindings = {
        name: _canonical_typed_arithmetic(
            bindings_raw[name], aliases, f"variational binding {name}"
        )
        for name in formal_symbols
    }
    claimed_expression = _safe_expression(claimed, allowed_symbols)
    substitutions = {
        sp.Symbol(name, real=True): _safe_expression(bindings[name], aliases)
        for name in formal_symbols
    }
    output = sp.cancel(claimed_expression.subs(substitutions, simultaneous=True))
    alias_symbols = {sp.Symbol(name, real=True) for name in aliases}
    if output.free_symbols - alias_symbols:
        raise ExternalCreativityError("variational output retains an unbound formal symbol")
    output_expression = str(sp.factor(output))
    _safe_expression(output_expression, aliases)
    supplied_output = value.get("output_expression")
    if supplied_output is not None:
        supplied = _safe_expression(
            _canonical_typed_arithmetic(supplied_output, aliases, "variational output expression"),
            aliases,
        )
        if sp.cancel(supplied - output) != 0:
            raise ExternalCreativityError(
                "variational output expression is not induced by the declared bindings"
            )
    try:
        formula = VariationalFunctional(
            field,
            coordinate,
            first_derivative,
            second_derivative,
            _expression_ir_from_text(integrand, allowed_symbols),
            _expression_ir_from_text(claimed, allowed_symbols),
        )
    except ExpressionIRError as error:
        raise ExternalCreativityError(str(error)) from error
    spec = {
        "bindings": bindings,
        "claimed_euler_lagrange": claimed,
        "coordinate": coordinate,
        "field": field,
        "first_derivative": first_derivative,
        "integrand": integrand,
        "output_expression": output_expression,
        "second_derivative": second_derivative,
    }
    return json.dumps(spec, sort_keys=True, separators=(",", ":")), spec, formula


def _resolve_integer_bound(value: str, aliases: Mapping[str, Fraction]) -> int | None:
    if value in aliases:
        resolved = aliases[value]
        return int(resolved) if resolved.denominator == 1 else None
    try:
        return int(value)
    except ValueError:
        return None


def _evaluate_aggregate(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    product: bool,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec = _parse_aggregate_spec(
        candidate.expression,
        benchmark.aliases,
        "finite product" if product else "finite sum",
    )
    body = spec["body"]
    index = spec["index"]
    sympy_body = None if independent else _safe_expression(body, (*benchmark.aliases, index))
    sympy_symbols = (
        {}
        if independent
        else {name: sp.Symbol(name, real=True) for name in (*benchmark.aliases, index)}
    )
    outputs: list[Fraction | None] = []
    for row in rows:
        aliases = dict(zip(benchmark.aliases, row.inputs, strict=True))
        lower = _resolve_integer_bound(spec["lower"], aliases)
        upper = _resolve_integer_bound(spec["upper"], aliases)
        if lower is None or upper is None or upper - lower + 1 > MAXIMUM_TYPED_TERMS:
            outputs.append(None)
            continue
        accumulator = Fraction(1 if product else 0)
        valid = True
        for term_index in range(lower, upper + 1):
            variables = {**aliases, index: Fraction(term_index)}
            try:
                if independent:
                    value = independently_evaluate_expression(body, variables)
                else:
                    assert sympy_body is not None
                    evaluated = sp.cancel(
                        sympy_body.subs(
                            {
                                sympy_symbols[name]: sp.Rational(item.numerator, item.denominator)
                                for name, item in variables.items()
                            }
                        )
                    )
                    if not evaluated.is_Rational:
                        valid = False
                        break
                    value = Fraction(int(evaluated.p), int(evaluated.q))
            except (IndependentEvaluationError, ZeroDivisionError):
                valid = False
                break
            accumulator = accumulator * value if product else accumulator + value
        outputs.append(accumulator if valid else None)
    return tuple(outputs)


def _evaluate_modular(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec = _parse_modular_spec(candidate.expression, benchmark.aliases)
    expression = spec["expression"]
    modulus = spec["modulus"]
    if independent:
        values = []
        for row in rows:
            try:
                value = independently_evaluate_expression(
                    expression, dict(zip(benchmark.aliases, row.inputs, strict=True))
                )
            except IndependentEvaluationError:
                values.append(None)
                continue
            values.append(Fraction(value.numerator % modulus) if value.denominator == 1 else None)
        return tuple(values)
    base = _evaluate_expression(
        replace(candidate, representation="sympy_expression", expression=expression),
        benchmark,
        rows,
    )
    return tuple(
        Fraction(value.numerator % modulus)
        if value is not None and value.denominator == 1
        else None
        for value in base
    )


def _evaluate_piecewise(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec = _parse_piecewise_spec(candidate.expression, benchmark.aliases)
    outputs: list[Fraction | None] = []
    if independent:
        for row in rows:
            assignment = dict(zip(benchmark.aliases, row.inputs, strict=True))
            selected = spec["default_expression"]
            try:
                for branch in spec["branches"]:
                    condition = branch["condition"]
                    if independently_evaluate_comparison(
                        condition["left"],
                        condition["comparator"],
                        condition["right"],
                        assignment,
                    ):
                        selected = branch["expression"]
                        break
                outputs.append(independently_evaluate_expression(selected, assignment))
            except (IndependentEvaluationError, ZeroDivisionError):
                outputs.append(None)
        return tuple(outputs)

    symbols = {name: sp.Symbol(name, real=True) for name in benchmark.aliases}
    parsed_branches = [
        (
            _safe_expression(branch["condition"]["left"], benchmark.aliases),
            branch["condition"]["comparator"],
            _safe_expression(branch["condition"]["right"], benchmark.aliases),
            _safe_expression(branch["expression"], benchmark.aliases),
        )
        for branch in spec["branches"]
    ]
    default_expression = _safe_expression(spec["default_expression"], benchmark.aliases)
    for row in rows:
        substitutions = {
            symbols[name]: sp.Rational(value.numerator, value.denominator)
            for name, value in zip(benchmark.aliases, row.inputs, strict=True)
        }
        selected = default_expression
        valid = True
        for left, comparator, right, branch_expression in parsed_branches:
            left_value = sp.cancel(left.subs(substitutions))
            right_value = sp.cancel(right.subs(substitutions))
            if not left_value.is_Rational or not right_value.is_Rational:
                valid = False
                break
            comparisons = {
                "eq": left_value == right_value,
                "ge": left_value >= right_value,
                "gt": left_value > right_value,
                "le": left_value <= right_value,
                "lt": left_value < right_value,
                "ne": left_value != right_value,
            }
            if comparisons[comparator]:
                selected = branch_expression
                break
        if not valid:
            outputs.append(None)
            continue
        result = sp.cancel(selected.subs(substitutions))
        outputs.append(Fraction(int(result.p), int(result.q)) if result.is_Rational else None)
    return tuple(outputs)


def _generating_function_coefficients(
    numerator: Sequence[Fraction],
    denominator: Sequence[Fraction],
    maximum_index: int,
) -> tuple[Fraction, ...]:
    coefficients: list[Fraction] = []
    for index in range(maximum_index + 1):
        numerator_term = numerator[index] if index < len(numerator) else Fraction(0)
        feedback = sum(
            denominator[lag] * coefficients[index - lag]
            for lag in range(1, min(index, len(denominator) - 1) + 1)
        )
        coefficients.append((numerator_term - feedback) / denominator[0])
    return tuple(coefficients)


def _evaluate_generating_function(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec = _parse_generating_function_spec(candidate.expression, benchmark.aliases)
    index_alias = spec["index"]
    index_position = benchmark.aliases.index(index_alias)
    requested = [row.inputs[index_position] for row in rows]
    if any(value.denominator != 1 or value < 0 for value in requested):
        return tuple(None for _ in rows)
    indices = [int(value) for value in requested]
    maximum_index = max(indices, default=-1)
    if maximum_index >= MAXIMUM_TYPED_TERMS:
        return tuple(None for _ in rows)
    numerator = tuple(Fraction(item) for item in spec["numerator"])
    denominator = tuple(Fraction(item) for item in spec["denominator"])
    if independent:
        coefficients = _generating_function_coefficients(numerator, denominator, maximum_index)
    else:
        z = sp.Symbol("z")
        numerator_expression = sum(
            sp.Rational(value.numerator, value.denominator) * z**degree
            for degree, value in enumerate(numerator)
        )
        denominator_expression = sum(
            sp.Rational(value.numerator, value.denominator) * z**degree
            for degree, value in enumerate(denominator)
        )
        series = sp.series(
            numerator_expression / denominator_expression,
            z,
            0,
            maximum_index + 1,
        ).removeO()
        coefficients = tuple(
            Fraction(int(value.p), int(value.q))
            for index in range(maximum_index + 1)
            if (value := sp.cancel(series.coeff(z, index))).is_Rational
        )
        if len(coefficients) != maximum_index + 1:
            return tuple(None for _ in rows)
    return tuple(coefficients[index] for index in indices)


def _evaluate_transform(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec = _parse_transform_spec(candidate.expression, benchmark.aliases)
    source = spec["source_expression"]
    claimed = spec["claimed_transform"]
    index = spec["index"]
    stencil = tuple((Fraction(term["coefficient"]), term["offset"]) for term in spec["stencil"])
    outputs: list[Fraction | None] = []
    if independent:
        for row in rows:
            assignment = dict(zip(benchmark.aliases, row.inputs, strict=True))
            try:
                transformed = sum(
                    (
                        coefficient
                        * independently_evaluate_expression(
                            source,
                            {**assignment, index: assignment[index] + offset},
                        )
                        for coefficient, offset in stencil
                    ),
                    Fraction(0),
                )
                claimed_value = independently_evaluate_expression(claimed, assignment)
            except (IndependentEvaluationError, ZeroDivisionError):
                outputs.append(None)
                continue
            outputs.append(claimed_value if transformed == claimed_value else None)
        return tuple(outputs)

    source_expression = _safe_expression(source, benchmark.aliases)
    claimed_expression = _safe_expression(claimed, benchmark.aliases)
    symbols = {name: sp.Symbol(name, real=True) for name in benchmark.aliases}
    for row in rows:
        assignment = dict(zip(benchmark.aliases, row.inputs, strict=True))
        transformed = sp.Integer(0)
        for coefficient, offset in stencil:
            shifted = {**assignment, index: assignment[index] + offset}
            transformed += sp.Rational(coefficient.numerator, coefficient.denominator) * sp.cancel(
                source_expression.subs(
                    {
                        symbols[name]: sp.Rational(value.numerator, value.denominator)
                        for name, value in shifted.items()
                    }
                )
            )
        claimed_value = sp.cancel(
            claimed_expression.subs(
                {
                    symbols[name]: sp.Rational(value.numerator, value.denominator)
                    for name, value in assignment.items()
                }
            )
        )
        transformed = sp.cancel(transformed)
        if transformed.is_Rational and claimed_value.is_Rational and transformed == claimed_value:
            outputs.append(Fraction(int(claimed_value.p), int(claimed_value.q)))
        else:
            outputs.append(None)
    return tuple(outputs)


def _evaluate_expanded_formula_output(
    formula: TensorIdentity | VariationalFunctional,
    output_expression: str,
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    if independent:
        predicted: list[Fraction | None] = []
        for row in rows:
            assignment = dict(zip(benchmark.aliases, row.inputs, strict=True))
            try:
                identity_holds = independently_evaluate_expanded(formula, assignment)
                value = independently_evaluate_expression(output_expression, assignment)
            except (
                ExpandedIndependentEvaluationError,
                IndependentEvaluationError,
                ZeroDivisionError,
            ):
                predicted.append(None)
                continue
            predicted.append(value if identity_holds else None)
        return tuple(predicted)
    base = _evaluate_expression(
        replace(
            candidate,
            representation="sympy_expression",
            expression=output_expression,
        ),
        benchmark,
        rows,
    )
    predicted = []
    for row, value in zip(rows, base, strict=True):
        assignment = dict(zip(benchmark.aliases, row.inputs, strict=True))
        try:
            identity_holds = primarily_evaluate_expanded(formula, assignment)
        except (ExpandedPrimaryEvaluationError, ZeroDivisionError):
            identity_holds = False
        predicted.append(value if identity_holds else None)
    return tuple(predicted)


def _evaluate_tensor(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec, formula = _parse_tensor_spec(candidate.expression, benchmark.aliases)
    output = spec["output_component"]
    components = spec[f"{output['side']}_components"]
    return _evaluate_expanded_formula_output(
        formula,
        components[output["flat_index"]],
        candidate,
        benchmark,
        rows,
        independent=independent,
    )


def _evaluate_variational(
    candidate: Candidate,
    benchmark: Benchmark,
    rows: Sequence[Observation],
    *,
    independent: bool,
) -> tuple[Fraction | None, ...]:
    _, spec, formula = _parse_variational_spec(candidate.expression, benchmark.aliases)
    return _evaluate_expanded_formula_output(
        formula,
        spec["output_expression"],
        candidate,
        benchmark,
        rows,
        independent=independent,
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


def predict(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> tuple[Fraction | None, ...]:
    if candidate.representation == "linear_recurrence":
        return _evaluate_recurrence(candidate, benchmark, rows)
    if candidate.representation == "generating_function":
        return _evaluate_generating_function(candidate, benchmark, rows, independent=False)
    if candidate.representation == "finite_sum":
        return _evaluate_aggregate(candidate, benchmark, rows, product=False, independent=False)
    if candidate.representation == "finite_product":
        return _evaluate_aggregate(candidate, benchmark, rows, product=True, independent=False)
    if candidate.representation == "modular_relation":
        return _evaluate_modular(candidate, benchmark, rows, independent=False)
    if candidate.representation == "piecewise_relation":
        return _evaluate_piecewise(candidate, benchmark, rows, independent=False)
    if candidate.representation == "transform_relation":
        return _evaluate_transform(candidate, benchmark, rows, independent=False)
    if candidate.representation == "tensor_identity":
        return _evaluate_tensor(candidate, benchmark, rows, independent=False)
    if candidate.representation == "variational_functional":
        return _evaluate_variational(candidate, benchmark, rows, independent=False)
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
        if candidate.representation == "finite_sum":
            return _evaluate_aggregate(candidate, benchmark, rows, product=False, independent=True)
        if candidate.representation == "finite_product":
            return _evaluate_aggregate(candidate, benchmark, rows, product=True, independent=True)
        if candidate.representation == "modular_relation":
            return _evaluate_modular(candidate, benchmark, rows, independent=True)
        if candidate.representation == "piecewise_relation":
            return _evaluate_piecewise(candidate, benchmark, rows, independent=True)
        if candidate.representation == "generating_function":
            return _evaluate_generating_function(candidate, benchmark, rows, independent=True)
        if candidate.representation == "transform_relation":
            return _evaluate_transform(candidate, benchmark, rows, independent=True)
        if candidate.representation == "tensor_identity":
            return _evaluate_tensor(candidate, benchmark, rows, independent=True)
        if candidate.representation == "variational_functional":
            return _evaluate_variational(candidate, benchmark, rows, independent=True)
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
            (
                sp.Rational(row.inputs[0].numerator, row.inputs[0].denominator),
                sp.Rational(row.output.numerator, row.output.denominator),
            )
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
        proof_plan=(
            "derive_finite_difference_lemma",
            "change_to_polynomial_basis",
            "induct_on_index",
        ),
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
        proof_plan=(
            "solve_dimension_lattice",
            "fit_dimensionless_group",
            "verify_scaling_interventions",
        ),
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
            sp.Eq(
                sequence[index],
                sum(
                    coefficients[offset - 1] * sequence[index - offset]
                    for offset in range(1, order + 1)
                ),
            )
            for index in range(order, len(sequence))
        ]
        solutions = sp.solve(equations, coefficients, dict=True)
        for solution in solutions:
            if set(solution) != set(coefficients) or any(
                not solution[item].is_Rational for item in coefficients
            ):
                continue
            fractions = tuple(
                Fraction(int(solution[item].p), int(solution[item].q)) for item in coefficients
            )
            return _candidate(
                benchmark,
                family,
                "linear_recurrence",
                "recurrence(" + ",".join(_fraction_text(item) for item in fractions) + ")",
                recurrence_coefficients=fractions,
                recurrence_seed=tuple(row.output for row in rows[:order]),
                invariants=("linear_recurrence", f"order_{order}"),
                proof_plan=(
                    "guess_recurrence",
                    "prove_initial_conditions",
                    "induct_with_recurrence",
                ),
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
        scored.append(
            (
                _loss(
                    predict(candidate, benchmark, benchmark.observations), benchmark.observations
                ),
                candidate,
            )
        )
    return tuple(
        candidate
        for _, candidate in sorted(scored, key=lambda item: (item[0], item[1].candidate_id))[:4]
    )


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


def _claude_candidate(
    benchmark: Benchmark, hypothesis: ClaudeHypothesis
) -> tuple[Candidate | None, dict[str, Any]]:
    record: dict[str, Any] = {
        "candidate_id": None,
        "diagnostic": None,
        "executable_representation": None,
        "hypothesis_id": hypothesis.hypothesis_id,
        "llm_self_assessed_origin": hypothesis.llm_origin_assessment,
        "normalization": None,
        "reason": None,
        "source_representation": hypothesis.representation,
        "status": "RETAINED_NON_EXECUTABLE",
    }
    if hypothesis.representation not in EXECUTABLE_CLAUDE_REPRESENTATIONS:
        record["reason"] = "representation_not_yet_executable"
        return None, record
    representation = hypothesis.representation
    expression = hypothesis.expression
    coefficients: tuple[Fraction, ...] = ()
    seed: tuple[Fraction, ...] = ()
    try:
        if representation == "sympy_expression":
            expression, normalization = _normalize_claude_arithmetic(expression, benchmark.aliases)
        elif representation == "invariant_relation":
            if re.fullmatch(r"output\s*=\s*.+", expression.strip(), flags=re.DOTALL) is None:
                record["reason"] = "invariant_relation_missing_output_assignment"
                return None, record
            expression, normalization = _normalize_claude_arithmetic(expression, benchmark.aliases)
            representation = "sympy_expression"
        elif representation == "linear_recurrence":
            expression, coefficients, seed = _parse_recurrence_spec(expression)
            normalization = "canonical_typed_json"
        elif representation in {"finite_product", "finite_sum"}:
            expression, _ = _parse_aggregate_spec(
                expression,
                benchmark.aliases,
                "finite product" if representation == "finite_product" else "finite sum",
            )
            normalization = "canonical_typed_json"
        elif representation == "generating_function":
            expression, _ = _parse_generating_function_spec(expression, benchmark.aliases)
            normalization = "canonical_typed_json"
        elif representation == "modular_relation":
            expression, _ = _parse_modular_spec(expression, benchmark.aliases)
            normalization = "canonical_typed_json"
        elif representation == "piecewise_relation":
            expression, _ = _parse_piecewise_spec(expression, benchmark.aliases)
            normalization = "canonical_typed_json+ordered_exact_predicates"
        elif representation == "transform_relation":
            expression, _ = _parse_transform_spec(expression, benchmark.aliases)
            normalization = "canonical_typed_json+exact_shift_stencil"
        elif representation == "tensor_identity":
            expression, _, _ = _parse_tensor_spec(expression, benchmark.aliases)
            normalization = "canonical_typed_json"
        elif representation == "variational_principle":
            expression, _, _ = _parse_variational_spec(expression, benchmark.aliases)
            representation = "variational_functional"
            normalization = "canonical_typed_json+derived_output_expression"
        else:  # pragma: no cover - guarded by the executable representation set
            raise ExternalCreativityError("executable representation compiler is missing")
    except ExternalCreativityError as error:
        record["reason"] = "typed_expression_failed_validation"
        record["diagnostic"] = str(error)
        return None, record
    candidate = _candidate(
        benchmark,
        "claude_proposer",
        representation,
        expression,
        recurrence_coefficients=coefficients,
        recurrence_seed=seed,
        invariants=hypothesis.invariants,
        proof_plan=hypothesis.proof_plan,
        proposer="claude_api",
    )
    record.update(
        {
            "candidate_id": candidate.candidate_id,
            "executable_representation": candidate.representation,
            "normalization": normalization,
            "reason": "typed_expression_admitted",
            "status": "ADMITTED_EXECUTABLE",
        }
    )
    return candidate, record


def _candidate_syntax_profile(candidate: Candidate, benchmark: Benchmark) -> tuple[int, int]:
    if candidate.representation == "sympy_expression":
        return _arithmetic_tree_stats(candidate.expression, benchmark.aliases)
    if candidate.representation == "generating_function":
        _, spec = _parse_generating_function_spec(candidate.expression, benchmark.aliases)
        width = max(len(spec["numerator"]), len(spec["denominator"]))
        return 2 + width.bit_length(), 2 + len(spec["numerator"]) + len(spec["denominator"])
    if candidate.representation == "linear_recurrence":
        order = len(candidate.recurrence_coefficients)
        return 2 + order.bit_length(), 1 + len(candidate.recurrence_coefficients) + len(
            candidate.recurrence_seed
        )
    if candidate.representation in {"finite_product", "finite_sum"}:
        _, spec = _parse_aggregate_spec(
            candidate.expression,
            benchmark.aliases,
            "finite product" if candidate.representation == "finite_product" else "finite sum",
        )
        depth, nodes = _arithmetic_tree_stats(spec["body"], (*benchmark.aliases, spec["index"]))
        return depth + 1, nodes + 4
    if candidate.representation == "modular_relation":
        _, spec = _parse_modular_spec(candidate.expression, benchmark.aliases)
        depth, nodes = _arithmetic_tree_stats(spec["expression"], benchmark.aliases)
        return depth + 1, nodes + 1
    if candidate.representation == "piecewise_relation":
        _, spec = _parse_piecewise_spec(candidate.expression, benchmark.aliases)
        expression_stats = [
            *(
                stats
                for branch in spec["branches"]
                for stats in (
                    _arithmetic_tree_stats(branch["condition"]["left"], benchmark.aliases),
                    _arithmetic_tree_stats(branch["condition"]["right"], benchmark.aliases),
                    _arithmetic_tree_stats(branch["expression"], benchmark.aliases),
                )
            ),
            _arithmetic_tree_stats(spec["default_expression"], benchmark.aliases),
        ]
        return (
            3 + max(depth for depth, _ in expression_stats),
            2 + len(spec["branches"]) + sum(nodes for _, nodes in expression_stats),
        )
    if candidate.representation == "transform_relation":
        _, spec = _parse_transform_spec(candidate.expression, benchmark.aliases)
        source_depth, source_nodes = _arithmetic_tree_stats(
            spec["source_expression"], benchmark.aliases
        )
        claimed_depth, claimed_nodes = _arithmetic_tree_stats(
            spec["claimed_transform"], benchmark.aliases
        )
        return (
            3 + max(source_depth, claimed_depth),
            5 + source_nodes + claimed_nodes + 2 * len(spec["stencil"]),
        )
    if candidate.representation == "tensor_identity":
        _, spec, _ = _parse_tensor_spec(candidate.expression, benchmark.aliases)
        component_stats = [
            _arithmetic_tree_stats(item, benchmark.aliases)
            for item in (*spec["left_components"], *spec["right_components"])
        ]
        metadata_nodes = len(spec["shape"]) + 3 * len(spec["symmetries"]) + 2
        return (
            2 + len(spec["shape"]) + max(depth for depth, _ in component_stats),
            metadata_nodes + sum(nodes for _, nodes in component_stats),
        )
    if candidate.representation == "variational_functional":
        _, spec, _ = _parse_variational_spec(candidate.expression, benchmark.aliases)
        formal_symbols = (
            spec["field"],
            spec["coordinate"],
            spec["first_derivative"],
            spec["second_derivative"],
        )
        expression_stats = [
            _arithmetic_tree_stats(spec["integrand"], (*formal_symbols, *benchmark.aliases)),
            _arithmetic_tree_stats(
                spec["claimed_euler_lagrange"],
                (*formal_symbols, *benchmark.aliases),
            ),
            *(
                _arithmetic_tree_stats(item, benchmark.aliases)
                for item in spec["bindings"].values()
            ),
            _arithmetic_tree_stats(spec["output_expression"], benchmark.aliases),
        ]
        return (
            3 + max(depth for depth, _ in expression_stats),
            8 + sum(nodes for _, nodes in expression_stats),
        )
    raise ExternalCreativityError("candidate has no executable syntax profile")


def _evaluation_operations(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> int:
    _, nodes = _candidate_syntax_profile(candidate, benchmark)
    if candidate.representation == "linear_recurrence":
        requested = [
            int(row.inputs[0])
            for row in rows
            if len(row.inputs) == 1 and row.inputs[0].denominator == 1
        ]
        if len(requested) != len(rows) or any(index < 0 for index in requested):
            return 10**12
        generated = max(0, max(requested, default=-1) + 1 - len(candidate.recurrence_seed))
        return max(1, generated * len(candidate.recurrence_coefficients))
    if candidate.representation == "generating_function":
        _, spec = _parse_generating_function_spec(candidate.expression, benchmark.aliases)
        position = benchmark.aliases.index(spec["index"])
        requested = [row.inputs[position] for row in rows]
        if any(value.denominator != 1 or value < 0 for value in requested):
            return 10**12
        maximum_index = max((int(value) for value in requested), default=-1)
        if maximum_index >= MAXIMUM_TYPED_TERMS:
            return 10**12
        recurrence_width = max(1, len(spec["denominator"]) - 1)
        return max(1, (maximum_index + 1) * recurrence_width)
    if candidate.representation in {"finite_product", "finite_sum"}:
        _, spec = _parse_aggregate_spec(
            candidate.expression,
            benchmark.aliases,
            "finite product" if candidate.representation == "finite_product" else "finite sum",
        )
        body_nodes = _arithmetic_tree_stats(spec["body"], (*benchmark.aliases, spec["index"]))[1]
        operations = 0
        for row in rows:
            aliases = dict(zip(benchmark.aliases, row.inputs, strict=True))
            lower = _resolve_integer_bound(spec["lower"], aliases)
            upper = _resolve_integer_bound(spec["upper"], aliases)
            if lower is None or upper is None:
                return 10**12
            terms = max(0, upper - lower + 1)
            if terms > MAXIMUM_TYPED_TERMS:
                return 10**12
            operations += terms * (body_nodes + 1)
        return max(1, operations)
    return max(1, nodes * len(rows))


def _candidate_resource_profile(
    candidate: Candidate,
    benchmark: Benchmark,
    target: SealedTarget,
    allocated_budget: Mapping[str, int],
) -> dict[str, int]:
    rows = (*benchmark.observations, *target.holdout_records)
    grammar_depth, _ = _candidate_syntax_profile(candidate, benchmark)
    evaluation_operations = _evaluation_operations(candidate, benchmark, rows)
    verifier_invocations = len(
        _proof_plan_search(
            candidate,
            target,
            allocated_budget["maximum_verifier_invocations"],
        )["selected_route"]
    )
    profile = {
        "evaluation_runtime_budget_units": evaluation_operations,
        "grammar_depth": grammar_depth,
        "verifier_invocation_budget": verifier_invocations,
    }
    if (
        grammar_depth > allocated_budget["maximum_grammar_depth"]
        or evaluation_operations > allocated_budget["maximum_evaluation_operations"]
        or verifier_invocations > allocated_budget["maximum_verifier_invocations"]
    ):
        raise ExternalCreativityError("candidate exceeds the matched-control resource budget")
    return profile


def _mutate_arithmetic_same_shape(
    expression: str, aliases: Sequence[str], rng: random.Random
) -> str:
    original = ast.parse(expression.replace("^", "**"), mode="eval")

    def mutate(node: ast.AST, *, exponent: bool = False) -> ast.AST:
        if isinstance(node, ast.Expression):
            return ast.Expression(mutate(node.body))
        if isinstance(node, ast.Name):
            if exponent:
                return ast.Constant(rng.randint(0, 4))
            choices: list[ast.expr] = [ast.Name(id=name, ctx=ast.Load()) for name in aliases]
            choices.extend(ast.Constant(value) for value in range(6))
            return rng.choice(choices)
        if isinstance(node, ast.Constant):
            return ast.Constant(rng.randint(0, 4) if exponent else rng.randint(0, 5))
        if isinstance(node, ast.UnaryOp):
            return ast.UnaryOp(op=type(node.op)(), operand=mutate(node.operand, exponent=exponent))
        if isinstance(node, ast.BinOp):
            if isinstance(node.op, ast.Pow):
                return ast.BinOp(
                    left=mutate(node.left),
                    op=ast.Pow(),
                    right=mutate(node.right, exponent=True),
                )
            operators = (ast.Add, ast.Sub, ast.Mult)
            operator = rng.choice(operators)()
            return ast.BinOp(left=mutate(node.left), op=operator, right=mutate(node.right))
        raise ExternalCreativityError("random-control mutation left the arithmetic DSL")

    expected = _arithmetic_tree_stats(expression, aliases)
    last_attempt: tuple[str, tuple[int, int] | None] | None = None
    for _ in range(64):
        candidate_tree = ast.fix_missing_locations(mutate(original))
        candidate_expression = ast.unparse(candidate_tree)
        try:
            _safe_expression(candidate_expression, aliases)
        except ExternalCreativityError:
            last_attempt = (candidate_expression, None)
            continue
        observed = _arithmetic_tree_stats(candidate_expression, aliases)
        last_attempt = (candidate_expression, observed)
        if observed == expected:
            return candidate_expression
    raise ExternalCreativityError(
        "could not construct a same-shape random control: "
        f"expected {expected}, last attempt {last_attempt!r}"
    )


def _matched_random_candidate(
    benchmark: Benchmark,
    source: Candidate,
    family: str,
    ordinal: int,
    rng: random.Random,
) -> Candidate:
    expression = source.expression
    coefficients: tuple[Fraction, ...] = ()
    seed: tuple[Fraction, ...] = ()
    if source.representation == "sympy_expression":
        try:
            expression = _mutate_arithmetic_same_shape(expression, benchmark.aliases, rng)
        except ExternalCreativityError as error:
            raise ExternalCreativityError(
                f"same-shape mutation failed for {source.expression!r}"
            ) from error
    elif source.representation == "generating_function":
        _, spec = _parse_generating_function_spec(expression, benchmark.aliases)
        numerator = [_fraction_text(Fraction(rng.randint(-5, 5))) for _ in spec["numerator"]]
        denominator = [_fraction_text(Fraction(rng.randint(-5, 5))) for _ in spec["denominator"]]
        if denominator[0] == "0":
            denominator[0] = "1"
        expression = json.dumps(
            {
                "denominator": denominator,
                "index": spec["index"],
                "numerator": numerator,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    elif source.representation == "linear_recurrence":
        coefficients = tuple(Fraction(rng.randint(-3, 3)) for _ in source.recurrence_coefficients)
        seed = tuple(Fraction(rng.randint(-5, 5)) for _ in source.recurrence_seed)
        expression = json.dumps(
            {
                "coefficients": [_fraction_text(item) for item in coefficients],
                "seed": [_fraction_text(item) for item in seed],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    elif source.representation in {"finite_product", "finite_sum"}:
        _, spec = _parse_aggregate_spec(
            expression,
            benchmark.aliases,
            "finite product" if source.representation == "finite_product" else "finite sum",
        )
        spec["body"] = _mutate_arithmetic_same_shape(
            spec["body"], (*benchmark.aliases, spec["index"]), rng
        )
        expression = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    elif source.representation == "modular_relation":
        _, spec = _parse_modular_spec(expression, benchmark.aliases)
        spec["expression"] = _mutate_arithmetic_same_shape(
            spec["expression"], benchmark.aliases, rng
        )
        expression = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    elif source.representation == "piecewise_relation":
        _, spec = _parse_piecewise_spec(expression, benchmark.aliases)
        spec["branches"] = [
            {
                "condition": {
                    "comparator": branch["condition"]["comparator"],
                    "left": _mutate_arithmetic_same_shape(
                        branch["condition"]["left"], benchmark.aliases, rng
                    ),
                    "right": _mutate_arithmetic_same_shape(
                        branch["condition"]["right"], benchmark.aliases, rng
                    ),
                },
                "expression": _mutate_arithmetic_same_shape(
                    branch["expression"], benchmark.aliases, rng
                ),
            }
            for branch in spec["branches"]
        ]
        spec["default_expression"] = _mutate_arithmetic_same_shape(
            spec["default_expression"], benchmark.aliases, rng
        )
        expression, _ = _parse_piecewise_spec(
            json.dumps(spec, sort_keys=True, separators=(",", ":")),
            benchmark.aliases,
        )
    elif source.representation == "transform_relation":
        _, spec = _parse_transform_spec(expression, benchmark.aliases)
        spec["source_expression"] = _mutate_arithmetic_same_shape(
            spec["source_expression"], benchmark.aliases, rng
        )
        spec["claimed_transform"] = _mutate_arithmetic_same_shape(
            spec["claimed_transform"], benchmark.aliases, rng
        )
        coefficient_choices = tuple(value for value in range(-5, 6) if value)
        spec["stencil"] = [
            {
                "coefficient": str(rng.choice(coefficient_choices)),
                "offset": term["offset"],
            }
            for term in spec["stencil"]
        ]
        expression, _ = _parse_transform_spec(
            json.dumps(spec, sort_keys=True, separators=(",", ":")),
            benchmark.aliases,
        )
    elif source.representation == "tensor_identity":
        _, spec, _ = _parse_tensor_spec(expression, benchmark.aliases)
        spec["left_components"] = [
            _mutate_arithmetic_same_shape(item, benchmark.aliases, rng)
            for item in spec["left_components"]
        ]
        spec["right_components"] = [
            _mutate_arithmetic_same_shape(item, benchmark.aliases, rng)
            for item in spec["right_components"]
        ]
        expression, _, _ = _parse_tensor_spec(
            json.dumps(spec, sort_keys=True, separators=(",", ":")),
            benchmark.aliases,
        )
    elif source.representation == "variational_functional":
        _, spec, _ = _parse_variational_spec(expression, benchmark.aliases)
        formal_symbols = (
            spec["field"],
            spec["coordinate"],
            spec["first_derivative"],
            spec["second_derivative"],
        )
        allowed_symbols = (*formal_symbols, *benchmark.aliases)
        for _ in range(64):
            mutated = {key: value for key, value in spec.items() if key != "output_expression"}
            mutated["integrand"] = _mutate_arithmetic_same_shape(
                spec["integrand"], allowed_symbols, rng
            )
            mutated["claimed_euler_lagrange"] = _mutate_arithmetic_same_shape(
                spec["claimed_euler_lagrange"], allowed_symbols, rng
            )
            mutated["bindings"] = {
                name: _mutate_arithmetic_same_shape(value, benchmark.aliases, rng)
                for name, value in spec["bindings"].items()
            }
            try:
                expression, _, _ = _parse_variational_spec(
                    json.dumps(mutated, sort_keys=True, separators=(",", ":")),
                    benchmark.aliases,
                )
            except ExternalCreativityError:
                continue
            break
        else:
            raise ExternalCreativityError(
                "could not construct an executable variational random control"
            )
    else:
        raise ExternalCreativityError("matched control has an unsupported representation")
    return _candidate(
        benchmark,
        f"random_control_{family}_{ordinal}",
        source.representation,
        expression,
        recurrence_coefficients=coefficients,
        recurrence_seed=seed,
        invariants=("random_budget_matched",),
        proof_plan=source.proof_plan,
        proposer="matched_random_search",
    )


def random_controls(
    benchmark: Benchmark,
    family_candidates: Mapping[str, Sequence[Candidate]],
    seed: int,
) -> dict[str, tuple[Candidate, ...]]:
    controls: dict[str, tuple[Candidate, ...]] = {}
    for family, candidates in sorted(family_candidates.items()):
        digest = hashlib.sha256(f"{seed}:{benchmark.blind_id}:{family}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        rows: list[Candidate] = []
        used_signatures: set[tuple[str, str]] = set()
        for ordinal, candidate in enumerate(candidates):
            source_signature = (candidate.representation, candidate.expression)
            source_syntax_profile = _candidate_syntax_profile(candidate, benchmark)
            for _ in range(128):
                control = _matched_random_candidate(benchmark, candidate, family, ordinal, rng)
                signature = (control.representation, control.expression)
                if (
                    signature != source_signature
                    and signature not in used_signatures
                    and _candidate_syntax_profile(control, benchmark) == source_syntax_profile
                ):
                    used_signatures.add(signature)
                    rows.append(control)
                    break
            else:
                raise ExternalCreativityError(
                    "could not construct a distinct matched random control for "
                    f"{family} ordinal {ordinal} ({candidate.representation})"
                )
        controls[family] = tuple(rows)
    return controls


def _behavior(
    candidate: Candidate, benchmark: Benchmark, rows: Sequence[Observation]
) -> dict[str, str]:
    predictions = predict(candidate, benchmark, rows)
    finite = [None if item is None else _fraction_text(item) for item in predictions]
    expression = None
    degree = "recurrence"
    singularities = "not_applicable"
    structure: Any = None
    if candidate.representation == "sympy_expression":
        expression = _safe_expression(candidate.expression, benchmark.aliases)
        symbols = [sp.Symbol(alias, real=True) for alias in benchmark.aliases]
        try:
            degree = str(
                sp.Poly(sp.together(expression).as_numer_denom()[0], *symbols).total_degree()
            )
        except sp.PolynomialError:
            degree = "nonpolynomial"
        denominator = sp.factor(sp.together(expression).as_numer_denom()[1])
        singularities = str(denominator)
        structure = str(sp.factor(expression))
    elif candidate.representation == "generating_function":
        _, spec = _parse_generating_function_spec(candidate.expression, benchmark.aliases)
        degree = "rational_generating_function"
        singularities = canonical_sha256(spec["denominator"])
        structure = {
            "denominator_degree": len(spec["denominator"]) - 1,
            "numerator_degree": len(spec["numerator"]) - 1,
        }
    elif candidate.representation in {"finite_product", "finite_sum"}:
        _, spec = _parse_aggregate_spec(
            candidate.expression,
            benchmark.aliases,
            "finite product" if candidate.representation == "finite_product" else "finite sum",
        )
        degree = "finite_aggregate"
        structure = {
            "body": str(
                sp.factor(_safe_expression(spec["body"], (*benchmark.aliases, spec["index"])))
            ),
            "bounds": [spec["lower"], spec["upper"]],
        }
    elif candidate.representation == "modular_relation":
        _, spec = _parse_modular_spec(candidate.expression, benchmark.aliases)
        degree = "modular"
        structure = {
            "expression": str(sp.factor(_safe_expression(spec["expression"], benchmark.aliases))),
            "modulus": spec["modulus"],
        }
    elif candidate.representation == "piecewise_relation":
        _, spec = _parse_piecewise_spec(candidate.expression, benchmark.aliases)
        degree = "piecewise"
        expressions = [
            *(
                item
                for branch in spec["branches"]
                for item in (
                    branch["condition"]["left"],
                    branch["condition"]["right"],
                    branch["expression"],
                )
            ),
            spec["default_expression"],
        ]
        singularities = canonical_sha256(
            [
                str(
                    sp.factor(
                        sp.together(_safe_expression(item, benchmark.aliases)).as_numer_denom()[1]
                    )
                )
                for item in expressions
            ]
        )
        structure = spec
    elif candidate.representation == "linear_recurrence":
        structure = {
            "order": len(candidate.recurrence_coefficients),
            "coefficients": [_fraction_text(item) for item in candidate.recurrence_coefficients],
        }
    elif candidate.representation == "tensor_identity":
        _, spec, _ = _parse_tensor_spec(candidate.expression, benchmark.aliases)
        degree = "tensor_identity"
        structure = {
            "output_component": spec["output_component"],
            "shape": spec["shape"],
            "symmetries": spec["symmetries"],
            "variance": spec["variance"],
        }
    elif candidate.representation == "transform_relation":
        _, spec = _parse_transform_spec(candidate.expression, benchmark.aliases)
        degree = "linear_shift_stencil"
        source = _safe_expression(spec["source_expression"], benchmark.aliases)
        claimed = _safe_expression(spec["claimed_transform"], benchmark.aliases)
        singularities = canonical_sha256(
            [
                str(sp.factor(sp.together(source).as_numer_denom()[1])),
                str(sp.factor(sp.together(claimed).as_numer_denom()[1])),
            ]
        )
        structure = {
            "claimed_transform": str(sp.factor(claimed)),
            "index": spec["index"],
            "source_expression": str(sp.factor(source)),
            "stencil": spec["stencil"],
            "transform_kind": spec["transform_kind"],
        }
    elif candidate.representation == "variational_functional":
        _, spec, _ = _parse_variational_spec(candidate.expression, benchmark.aliases)
        degree = "variational_functional"
        structure = {
            "claimed_euler_lagrange": spec["claimed_euler_lagrange"],
            "integrand": spec["integrand"],
            "output_expression": spec["output_expression"],
        }
    behavior_body = {
        "degree": degree,
        "predictions": finite,
        "representation": candidate.representation,
        "singularity_structure": singularities,
        "typed_structure": structure,
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
        return sum(
            (_interval_value(item, substitutions) for item in expression.args), mpmath.iv.mpf(0)
        )
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
        z3_variables = {
            symbol: z3.Real(alias) for symbol, alias in zip(symbols, benchmark.aliases, strict=True)
        }
        solver = z3.Solver()
        solver.add(_sympy_to_z3(found, z3_variables) != _sympy_to_z3(reference, z3_variables))
        smt = solver.check() == z3.unsat
        interval = True
        for row in (*benchmark.observations, *target.holdout_records):
            substitutions = dict(zip(symbols, row.inputs, strict=True))
            difference = _interval_value(found, substitutions) - _interval_value(
                reference, substitutions
            )
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
            row.inputs[0].denominator == 1 and next_row.inputs[0] == row.inputs[0] + 1
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
    allocated_budget: Mapping[str, int],
) -> dict[str, Any]:
    train_predictions = predict(candidate, benchmark, benchmark.observations)
    holdout_predictions = predict(candidate, benchmark, target.holdout_records)
    behavior = _behavior(candidate, benchmark, (*benchmark.observations, *target.holdout_records))
    return {
        **candidate.to_dict(),
        **behavior,
        "resource_profile": _candidate_resource_profile(
            candidate, benchmark, target, allocated_budget
        ),
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


def _proof_plan_search(
    candidate: Candidate,
    target: SealedTarget,
    maximum_invocations: int = 5,
) -> dict[str, Any]:
    """Enumerate verifier routes separately from candidate-expression search."""

    algebraic = candidate.representation == "sympy_expression"
    aggregate = candidate.representation in {"finite_product", "finite_sum"}
    recurrence = candidate.representation == "linear_recurrence"
    generating = candidate.representation == "generating_function"
    modular = candidate.representation == "modular_relation"
    piecewise = candidate.representation == "piecewise_relation"
    plans = [
        {
            "applicable": True,
            "estimated_cost": 1,
            "plan": "exact_row_replay",
            "purpose": "falsification",
        },
        {
            "applicable": algebraic or generating,
            "estimated_cost": 2,
            "plan": "cas_normal_form",
            "purpose": "symbolic_identity",
        },
        {
            "applicable": algebraic,
            "estimated_cost": 3,
            "plan": "smt_countermodel_search",
            "purpose": "universal_polynomial_identity",
        },
        {
            "applicable": algebraic or generating,
            "estimated_cost": 4,
            "plan": "interval_enclosure",
            "purpose": "numerical_domain_check",
        },
        {
            "applicable": recurrence or aggregate,
            "estimated_cost": 4,
            "plan": "induction_on_recurrence",
            "purpose": "choose_and_test_an_induction_variable",
        },
        {
            "applicable": recurrence or modular,
            "estimated_cost": 4,
            "plan": "invariant_strengthening",
            "purpose": "search_for_a_preserved_auxiliary_statement",
        },
        {
            "applicable": aggregate,
            "estimated_cost": 4,
            "plan": "bijection_construction",
            "purpose": "replace_algebra_with_structure_preserving_counting",
        },
        {
            "applicable": recurrence or modular,
            "estimated_cost": 4,
            "plan": "minimal_counterexample_descent",
            "purpose": "search_for_a_strictly_smaller_counterexample",
        },
        {
            "applicable": recurrence or generating,
            "estimated_cost": 4,
            "plan": "transform_domain_identity",
            "purpose": "move_between_sequence_and_generating_function_domains",
        },
        {
            "applicable": modular,
            "estimated_cost": 4,
            "plan": "contradiction_via_modular_obstruction",
            "purpose": "derive_an_incompatible_residue_class",
        },
        {
            "applicable": piecewise,
            "estimated_cost": 4,
            "plan": "boundary_partition_analysis",
            "purpose": "check_coverage_overlap_and_boundary_consistency",
        },
        {
            "applicable": target.target_kind == "known_formula",
            "estimated_cost": 5,
            "plan": "lean_kernel_bridge",
            "purpose": "formula_specific_formal_check",
        },
    ]
    applicable = [item for item in plans if item["applicable"]]
    selected = applicable[:maximum_invocations]
    lean = next((item for item in applicable if item["plan"] == "lean_kernel_bridge"), None)
    if lean is not None and lean not in selected:
        selected[-1] = lean
    return {
        "candidate_declared_plan": list(candidate.proof_plan),
        "plans": plans,
        "selected_route": [item["plan"] for item in selected],
        "selection_rule": "applicable_routes_in_cost_order_capped_by_verifier_budget_with_lean_reserved",
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
        candidate_count_match = len(rows) == len(random_rows)
        source_by_id = {item["candidate_id"]: item for item in rows}

        def profile_matches(
            field: str,
            *,
            count_match: bool = candidate_count_match,
            sources: Mapping[str, Mapping[str, Any]] = source_by_id,
            control_rows: Sequence[Mapping[str, Any]] = random_rows,
        ) -> bool:
            return count_match and all(
                control.get("matched_candidate_id") in sources
                and control["resource_profile"][field]
                == sources[control["matched_candidate_id"]]["resource_profile"][field]
                for control in control_rows
            )

        grammar_depth_match = profile_matches("grammar_depth")
        evaluation_runtime_budget_match = profile_matches("evaluation_runtime_budget_units")
        verifier_budget_match = profile_matches("verifier_invocation_budget")
        control_budget_match = (
            candidate_count_match
            and grammar_depth_match
            and evaluation_runtime_budget_match
            and verifier_budget_match
        )
        without = [item for item in creative if item["family"] != family]
        ablated = min(Fraction(item["holdout_loss"]) for item in without)
        metrics.append(
            {
                "best_holdout_loss": _fraction_text(best),
                "candidate_budget": len(rows),
                "candidate_count_match": candidate_count_match,
                "control_budget_match": control_budget_match,
                "evaluation_runtime_budget_match": evaluation_runtime_budget_match,
                "family": family,
                "grammar_depth_match": grammar_depth_match,
                "matched_budget": dict(allocated_search_budget),
                "matched_random_best_holdout_loss": _fraction_text(random_best),
                "matched_random_budget": len(random_rows),
                "outperformed_random": best < random_best,
                "unique_behaviors": len({item["behavior_sha256"] for item in rows}),
                "unique_proof_mechanisms": len({item["proof_mechanism_sha256"] for item in rows}),
                "verifier_budget_match": verifier_budget_match,
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


def _claude_contribution(
    scored: Sequence[Mapping[str, Any]],
    controls: Sequence[Mapping[str, Any]],
    admission_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    claude_rows = [item for item in scored if item["family"] == "claude_proposer"]
    deterministic_rows = [item for item in scored if item["family"] in FAMILY_IDS]
    source_by_id = {item["candidate_id"]: item for item in claude_rows}

    def profile_matches(field: str) -> bool | None:
        if not claude_rows:
            return None
        return len(claude_rows) == len(controls) and all(
            control.get("matched_candidate_id") in source_by_id
            and control["resource_profile"][field]
            == source_by_id[control["matched_candidate_id"]]["resource_profile"][field]
            for control in controls
        )

    origin_counts: dict[str, int] = {}
    executable_origin_counts: dict[str, int] = {}
    for record in admission_records:
        origin = str(record["llm_self_assessed_origin"])
        origin_counts[origin] = origin_counts.get(origin, 0) + 1
        if record["status"] == "ADMITTED_EXECUTABLE":
            executable_origin_counts[origin] = executable_origin_counts.get(origin, 0) + 1
    deterministic_behaviors = {item["behavior_sha256"] for item in deterministic_rows}
    deterministic_proofs = {item["proof_mechanism_sha256"] for item in deterministic_rows}
    best = min(Fraction(item["holdout_loss"]) for item in claude_rows) if claude_rows else None
    control_best = min(Fraction(item["holdout_loss"]) for item in controls) if controls else None
    status = "NO_CLAUDE_PROPOSALS"
    if admission_records:
        status = (
            "MEASURED_EXECUTABLE_CLAUDE_CONTRIBUTION"
            if claude_rows
            else "RETAINED_NON_EXECUTABLE_CLAUDE_PROPOSALS"
        )
    return {
        "admitted_executable_hypotheses": sum(
            record["status"] == "ADMITTED_EXECUTABLE" for record in admission_records
        ),
        "behavior_novelty_against_deterministic_count": len(
            {
                item["behavior_sha256"]
                for item in claude_rows
                if item["behavior_sha256"] not in deterministic_behaviors
            }
        ),
        "best_holdout_loss": None if best is None else _fraction_text(best),
        "claim_boundary": {
            "behavioral_novelty_is_literature_novelty": False,
            "llm_self_assessment_is_prior_art_authority": False,
            "proof_mechanism_novelty_is_mathematical_correctness": False,
        },
        "evaluation_runtime_budget_match": profile_matches("evaluation_runtime_budget_units"),
        "executable_llm_self_assessed_origin_counts": dict(
            sorted(executable_origin_counts.items())
        ),
        "grammar_depth_match": profile_matches("grammar_depth"),
        "llm_self_assessed_origin_counts": dict(sorted(origin_counts.items())),
        "matched_control_best_holdout_loss": (
            None if control_best is None else _fraction_text(control_best)
        ),
        "matched_control_count": len(controls),
        "outperformed_matched_random": (
            None if best is None or control_best is None else best < control_best
        ),
        "proof_mechanism_novelty_against_deterministic_count": len(
            {
                item["proof_mechanism_sha256"]
                for item in claude_rows
                if item["proof_mechanism_sha256"] not in deterministic_proofs
            }
        ),
        "proposed_hypotheses": len(admission_records),
        "scored_executable_candidates": len(claude_rows),
        "status": status,
        "verifier_budget_match": profile_matches("verifier_invocation_budget"),
    }


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
    if (
        raw["schema_version"] != CAMPAIGN_SCHEMA
        or raw["benchmark_config_path"] != PUBLIC_CONFIG_PATH
    ):
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
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in matched_budget.values()
    ):
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
    if (
        prior_art.get("automated_sources")
        != ["repository_theorem_library", "external_literature_index"]
        or prior_art.get("human_review_required") is not True
    ):
        raise ExternalCreativityError("prior-art release policy changed")
    open_policy = raw["open_problem_policy"]
    if (
        open_policy.get("minimum_independent_level5_passes", 0) < 3
        or open_policy.get("public_failure_receipt_required") is not True
    ):
        raise ExternalCreativityError("open-problem gate is too weak")
    return raw


def _prior_art_screen(
    root: Path, candidate: Mapping[str, Any], benchmark: Benchmark
) -> dict[str, Any]:
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
    deterministic = {
        item.benchmark_id: list(generate_candidates(item, maximum)) for item in benchmarks
    }
    event("deterministic_creativity_families_generated", target_reads=0)

    transport = claude_transport or ProviderCompatibleClaudeTransport()
    client = ClaudeCreativityClient(config["claude_api"], transport)
    claude_calls: list[ClaudeCallResult] = []
    claude_admission: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        call = client.run(
            ClaudeRole.PROPOSER,
            benchmark.blind_id,
            benchmark.generation_view(),
            instruction_override=EXECUTABLE_PROPOSER_INSTRUCTION,
        )
        if isinstance(transport, ProviderCompatibleClaudeTransport):
            response_id = str(call.evidence.get("api_response_id", ""))
            overflow = []
            rejected_overflow = 0
            for item in transport.hypothesis_overflow_for(response_id):
                try:
                    overflow.append(ClaudeHypothesis.from_mapping(item))
                except ClaudeCreativityError:
                    rejected_overflow += 1
            if overflow or rejected_overflow:
                if call.output is None:
                    raise ExternalCreativityError("provider overflow has no proposer output")
                call = replace(
                    call,
                    output=replace(
                        call.output,
                        hypotheses=(*call.output.hypotheses, *overflow),
                        rejected_hypotheses=(call.output.rejected_hypotheses + rejected_overflow),
                    ),
                )
            call = replace(
                call,
                evidence={
                    **call.evidence,
                    **transport.evidence_for(response_id),
                },
            )
        claude_calls.append(call)
        hypotheses = () if call.output is None else call.output.hypotheses
        admitted_rows = [_claude_candidate(benchmark, hypothesis) for hypothesis in hypotheses]
        admitted = tuple(candidate for candidate, _ in admitted_rows if candidate is not None)
        admission_records = [record for _, record in admitted_rows]
        deterministic[benchmark.benchmark_id].extend(admitted)
        deterministic[benchmark.benchmark_id] = list(
            {
                candidate.candidate_id: candidate
                for candidate in deterministic[benchmark.benchmark_id]
            }.values()
        )
        claude_admission[benchmark.benchmark_id] = {
            "admitted_executable_hypotheses": len(admitted),
            "non_executable_typed_hypotheses": len(hypotheses) - len(admitted),
            "proposed_hypotheses": len(hypotheses),
            "records": admission_records,
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
            [
                item.to_dict()
                for item in sorted(
                    deterministic[benchmark.benchmark_id], key=lambda row: row.candidate_id
                )
            ]
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

    family_candidates_by_benchmark: dict[str, dict[str, tuple[Candidate, ...]]] = {}
    control_candidates_by_benchmark: dict[str, dict[str, tuple[Candidate, ...]]] = {}
    for benchmark in benchmarks:
        candidates = deterministic[benchmark.benchmark_id]
        family_candidates = {
            family: tuple(item for item in candidates if item.family == family)
            for family in FAMILY_IDS
        }
        claude_candidates = tuple(item for item in candidates if item.family == "claude_proposer")
        if claude_candidates:
            family_candidates["claude_proposer"] = claude_candidates
        family_candidates_by_benchmark[benchmark.benchmark_id] = family_candidates
        control_candidates_by_benchmark[benchmark.benchmark_id] = random_controls(
            benchmark,
            family_candidates,
            config["search"]["random_seed"],
        )
    event("matched_random_controls_sealed", target_reads=0)

    targets = unseal_targets(root, public, benchmarks)
    event("sealed_targets_opened_after_proposal_and_critique", target_reads=1)
    by_target = {item.benchmark_id: item for item in targets}
    benchmark_results = []
    level5_passes = 0
    for benchmark in benchmarks:
        target = by_target[benchmark.benchmark_id]
        candidates = tuple(deterministic[benchmark.benchmark_id])
        allocated_search_budget = config["search"]["matched_control_budget"]
        scored = [
            _score_candidate(item, benchmark, target, allocated_search_budget)
            for item in candidates
        ]
        source_candidates = family_candidates_by_benchmark[benchmark.benchmark_id]
        controls = control_candidates_by_benchmark[benchmark.benchmark_id]
        scored_controls = {
            family: [
                {
                    **_score_candidate(control, benchmark, target, allocated_search_budget),
                    "matched_candidate_id": source.candidate_id,
                }
                for source, control in zip(source_candidates[family], rows, strict=True)
            ]
            for family, rows in controls.items()
        }
        metrics, ablations = _family_metrics(scored, scored_controls, allocated_search_budget)
        best_row = min(
            scored, key=lambda item: (Fraction(item["holdout_loss"]), item["candidate_id"])
        )
        best_candidate = next(
            item for item in candidates if item.candidate_id == best_row["candidate_id"]
        )
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
        candidate_counts_matched = all(item["candidate_count_match"] for item in metrics)
        grammar_depths_matched = all(item["grammar_depth_match"] for item in metrics)
        evaluation_runtime_budgets_matched = all(
            item["evaluation_runtime_budget_match"] for item in metrics
        )
        verifier_budgets_matched = all(item["verifier_budget_match"] for item in metrics)
        claude_controls = scored_controls.get("claude_proposer", [])
        claude_contribution = _claude_contribution(
            scored,
            claude_controls,
            claude_admission[benchmark.benchmark_id]["records"],
        )
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
                    "known_formula_rediscovered": target.target_kind == "known_formula"
                    and best_row["holdout_loss"] == "0",
                    "novel_formula_established": False,
                    "open_problem_solved": False,
                    "serious_claim_released": False,
                },
                "claude_contribution": claude_contribution,
                "claude_matched_controls": claude_controls,
                "dataset_evidence": _dataset_evidence(benchmark, target, best_candidate),
                "external_authorship": benchmark.source.to_dict(),
                "family_ablation": ablations,
                "family_metrics": metrics,
                "matched_control_policy": {
                    "allocated_budget": allocated_search_budget,
                    "all_family_budgets_match": controls_budget_matched,
                    "candidate_count_matched": candidate_counts_matched,
                    "deterministic_operation_budget_used": True,
                    "evaluation_runtime_budget_matched": evaluation_runtime_budgets_matched,
                    "grammar_depth_matched": grammar_depths_matched,
                    "verifier_budget_matched": verifier_budgets_matched,
                    "wall_clock_runtime_claimed_matched": False,
                },
                "formal_verification": formal,
                "holdout_count": len(target.holdout_records),
                "independent_exact_reproduction": independent_reproduction,
                "prior_art": prior_art,
                "proof_plan_search": _proof_plan_search(
                    best_candidate,
                    target,
                    allocated_search_budget["maximum_verifier_invocations"],
                ),
                "proposal_root_sha256": proposal_roots[benchmark.benchmark_id],
                "proposer_admission": claude_admission[benchmark.benchmark_id],
                "random_controls": {family: scored_controls[family] for family in FAMILY_IDS},
                "ranked_candidates": sorted(
                    scored, key=lambda item: (Fraction(item["holdout_loss"]), item["candidate_id"])
                ),
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
            "claude_transport_source_sha256": _file_sha256(root / CLAUDE_TRANSPORT_SOURCE_PATH),
            "independent_evaluator_sha256": _file_sha256(root / INDEPENDENT_EVALUATOR_PATH),
            "lean_source_sha256": _file_sha256(root / LEAN_SOURCE_PATH),
            "public_benchmarks_sha256": _file_sha256(root / PUBLIC_CONFIG_PATH),
            "sealed_targets_sha256": _file_sha256(root / public["sealed_targets_path"]),
            "source_sha256": _file_sha256(root / SOURCE_PATH),
            "test_sha256": _file_sha256(root / TEST_PATH),
        },
        "independent_reproduction": {
            "minimum_implementations": config["verification"][
                "minimum_independent_implementations"
            ],
            "minimum_machines": config["verification"]["minimum_independent_machines"],
            "received_implementations": (
                2
                if all(
                    item["independent_exact_reproduction"]["match"] for item in benchmark_results
                )
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
            "status": "READY_NOT_SPENT"
            if open_authorized
            else "BLOCKED_INSUFFICIENT_LEVEL5_REPETITIONS",
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
