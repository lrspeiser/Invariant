"""Bounded exact symbolic generation into Sigma Core candidate artifacts.

Templates are caller-constructed SymPy trees, never parsed source strings.  Algebraic deduplication
is exactly the Math Pack canonicalizer boundary; it is not a proof, truth, or promotion decision.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import product
from typing import Any

import sympy as sp

from sigma_theory_compiler.math_canonicalizer import (
    CanonicalizationError,
    canonical_data,
    canonicalize_expression,
)
from sigma_theory_compiler.math_canonicalizer import (
    canonical_sha256 as expression_sha256,
)
from sigma_theory_compiler.math_expression_ir import (
    Expression,
    add,
    call,
    literal,
    multiply,
    power,
    symbol,
)
from sigma_theory_compiler.math_types import IntegerType, MathType, RationalType
from sigma_theory_compiler.sigma_core import (
    ArtifactKind,
    CandidateArtifact,
    DomainPackRef,
    ProvenanceRecord,
    SchemaViolation,
    canonical_sha256,
)

TEMPLATE_SCHEMA = "sigma-symbolic-template-1.0"
BATCH_SCHEMA = "sigma-symbolic-candidate-batch-1.0"
ORIGIN_SCHEMA = "sigma-symbolic-origin-1.0"
SCOPE = (
    "Bounded exact symbolic generation and Math Pack canonical deduplication only; canonical "
    "equivalence does not imply proof, truth, scientific validity, gate passage, or promotion."
)
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_SAFE_FUNCTIONS = frozenset({"Abs", "cos", "exp", "log", "sin"})
_FUNCTION_IR_NAMES = {"Abs": "abs", "cos": "cos", "exp": "exp", "log": "log", "sin": "sin"}
_MATH_TYPES = (IntegerType, RationalType)
_MAX_TEMPLATES = 64
_MAX_COEFFICIENTS_PER_AXIS = 64
_MAX_CANDIDATES = 10_000
_MAX_WORK_ITEMS = 100_000
_MAX_TEMPLATE_NODES = 256
_MAX_EXPONENT_MAGNITUDE = 16
_MAX_INTEGER_BITS = 256
_SHA256_LENGTH = 64


class SymbolicGeneratorError(ValueError):
    """A symbolic template, budget, or sealed receipt is invalid."""


def _exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise SymbolicGeneratorError(f"{label} keys changed")


def _identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise SymbolicGeneratorError(f"{label} is not a canonical identifier")
    return value


def _sha256(value: str, label: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != _SHA256_LENGTH
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise SymbolicGeneratorError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _bounded_positive(value: int, maximum: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
        raise SymbolicGeneratorError(f"{label} must be an integer in [1, {maximum}]")
    return value


@dataclass(frozen=True, slots=True, order=True)
class ExactRational:
    numerator: int
    denominator: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.numerator, bool)
            or isinstance(self.denominator, bool)
            or not isinstance(self.numerator, int)
            or not isinstance(self.denominator, int)
        ):
            raise SymbolicGeneratorError("exact rational components must be integers")
        if self.denominator <= 0 or math.gcd(self.numerator, self.denominator) != 1:
            raise SymbolicGeneratorError(
                "exact rational must have a positive denominator and be reduced"
            )
        if max(abs(self.numerator).bit_length(), self.denominator.bit_length()) > _MAX_INTEGER_BITS:
            raise SymbolicGeneratorError("exact rational exceeds the coefficient bit budget")

    @classmethod
    def create(cls, value: int | Fraction) -> ExactRational:
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            raise SymbolicGeneratorError("coefficients must be exact int or Fraction values")
        fraction = Fraction(value)
        return cls(fraction.numerator, fraction.denominator)

    @property
    def fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def to_dict(self) -> dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExactRational:
        _exact_keys(value, {"numerator", "denominator"}, "exact rational")
        return cls(value["numerator"], value["denominator"])


@dataclass(frozen=True, slots=True)
class SymbolicVariable:
    name: str
    math_type: MathType

    def __post_init__(self) -> None:
        _identifier(self.name, "variable name")
        if not isinstance(self.math_type, _MATH_TYPES):
            raise SymbolicGeneratorError("symbolic variables require IntegerType or RationalType")

    def to_dict(self) -> dict[str, str]:
        return {"name": self.name, "math_type": repr(self.math_type)}


def _validate_expression(
    expression: sp.Expr,
    *,
    variable_names: set[str],
    coefficient_names: set[str],
    allowed_functions: set[str],
) -> None:
    if not isinstance(expression, sp.Expr):
        raise SymbolicGeneratorError("template expression must be a SymPy Expr")
    if expression.atoms(sp.Float):
        raise SymbolicGeneratorError("floating-point SymPy atoms are forbidden")
    if sum(1 for _ in sp.preorder_traversal(expression)) > _MAX_TEMPLATE_NODES:
        raise SymbolicGeneratorError("template exceeds the node cap")
    free_names = {str(item) for item in expression.free_symbols}
    declared = variable_names | coefficient_names
    unknown = free_names - declared
    if unknown:
        raise SymbolicGeneratorError(f"template contains unknown symbols: {sorted(unknown)}")
    missing_coefficients = coefficient_names - free_names
    if missing_coefficients:
        raise SymbolicGeneratorError(
            f"template omits coefficient symbols: {sorted(missing_coefficients)}"
        )
    used_functions = {item.func.__name__ for item in expression.atoms(sp.Function)}
    unsafe = used_functions - allowed_functions
    if unsafe:
        raise SymbolicGeneratorError(f"template contains unsafe functions: {sorted(unsafe)}")
    for node in sp.preorder_traversal(expression):
        if isinstance(node, sp.Pow):
            exponent = node.exp
            if not exponent.is_Rational:
                raise SymbolicGeneratorError("symbolic exponents are forbidden")
            exact = Fraction(int(exponent.p), int(exponent.q))
            if abs(exact) > _MAX_EXPONENT_MAGNITUDE:
                raise SymbolicGeneratorError("template exponent exceeds the magnitude cap")
        elif isinstance(node, (sp.Symbol, sp.Rational, sp.Function, sp.Add, sp.Mul)):
            continue
        else:
            raise SymbolicGeneratorError(f"unsupported SymPy node: {type(node).__name__}")


@dataclass(frozen=True, slots=True)
class SymbolicTemplate:
    template_id: str
    expression: sp.Expr
    variables: tuple[SymbolicVariable, ...]
    coefficient_symbols: tuple[str, ...]
    allowed_functions: tuple[str, ...]
    content_sha256: str
    schema_version: str = TEMPLATE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != TEMPLATE_SCHEMA:
            raise SymbolicGeneratorError("template schema_version changed")
        _identifier(self.template_id, "template_id")
        variable_names = tuple(item.name for item in self.variables)
        if variable_names != tuple(sorted(set(variable_names))):
            raise SymbolicGeneratorError("template variables must be unique and sorted")
        if self.coefficient_symbols != tuple(sorted(set(self.coefficient_symbols))):
            raise SymbolicGeneratorError("coefficient symbols must be unique and sorted")
        for name in self.coefficient_symbols:
            _identifier(name, "coefficient symbol")
        if set(variable_names) & set(self.coefficient_symbols):
            raise SymbolicGeneratorError("variables and coefficient symbols must be disjoint")
        if self.allowed_functions != tuple(sorted(set(self.allowed_functions))):
            raise SymbolicGeneratorError("allowed functions must be unique and sorted")
        if not set(self.allowed_functions) <= _SAFE_FUNCTIONS:
            raise SymbolicGeneratorError("template requested a function outside the safe set")
        _validate_expression(
            self.expression,
            variable_names=set(variable_names),
            coefficient_names=set(self.coefficient_symbols),
            allowed_functions=set(self.allowed_functions),
        )
        _sha256(self.content_sha256, "template content_sha256")
        if self.content_sha256 != canonical_sha256(self._descriptor_body()):
            raise SymbolicGeneratorError("template canonical hash changed")

    def _descriptor_body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "template_id": self.template_id,
            "expression_srepr": sp.srepr(self.expression),
            "variables": [item.to_dict() for item in self.variables],
            "coefficient_symbols": list(self.coefficient_symbols),
            "allowed_functions": list(self.allowed_functions),
        }

    @classmethod
    def create(
        cls,
        template_id: str,
        expression: sp.Expr,
        *,
        variables: Sequence[SymbolicVariable],
        coefficient_symbols: Sequence[str],
        allowed_functions: Sequence[str] = (),
    ) -> SymbolicTemplate:
        ordered_variables = tuple(sorted(variables, key=lambda item: item.name))
        ordered_coefficients = tuple(sorted(coefficient_symbols))
        ordered_functions = tuple(sorted(allowed_functions))
        body = {
            "schema_version": TEMPLATE_SCHEMA,
            "template_id": template_id,
            "expression_srepr": sp.srepr(expression),
            "variables": [item.to_dict() for item in ordered_variables],
            "coefficient_symbols": list(ordered_coefficients),
            "allowed_functions": list(ordered_functions),
        }
        return cls(
            template_id=template_id,
            expression=expression,
            variables=ordered_variables,
            coefficient_symbols=ordered_coefficients,
            allowed_functions=ordered_functions,
            content_sha256=canonical_sha256(body),
        )

    def descriptor(self) -> dict[str, Any]:
        return {**self._descriptor_body(), "content_sha256": self.content_sha256}


def _validate_template_descriptor(value: Mapping[str, Any]) -> dict[str, Any]:
    _exact_keys(
        value,
        {
            "schema_version",
            "template_id",
            "expression_srepr",
            "variables",
            "coefficient_symbols",
            "allowed_functions",
            "content_sha256",
        },
        "template descriptor",
    )
    if value["schema_version"] != TEMPLATE_SCHEMA:
        raise SymbolicGeneratorError("template descriptor schema changed")
    _identifier(value["template_id"], "template descriptor ID")
    for key in ("variables", "coefficient_symbols", "allowed_functions"):
        if not isinstance(value[key], list):
            raise SymbolicGeneratorError(f"template descriptor {key} must be an array")
    variable_names = []
    for variable in value["variables"]:
        _exact_keys(variable, {"name", "math_type"}, "template descriptor variable")
        variable_names.append(_identifier(variable["name"], "template descriptor variable name"))
        if not isinstance(variable["math_type"], str) or not variable["math_type"].startswith(
            ("IntegerType(", "RationalType(")
        ):
            raise SymbolicGeneratorError("template descriptor variable type changed")
    if variable_names != sorted(set(variable_names)):
        raise SymbolicGeneratorError("template descriptor variables must be unique and sorted")
    coefficient_symbols = value["coefficient_symbols"]
    if coefficient_symbols != sorted(set(coefficient_symbols)):
        raise SymbolicGeneratorError("template descriptor coefficients must be unique and sorted")
    for name in coefficient_symbols:
        _identifier(name, "template descriptor coefficient")
    if set(variable_names) & set(coefficient_symbols):
        raise SymbolicGeneratorError("template descriptor symbols overlap")
    if (
        value["allowed_functions"] != sorted(set(value["allowed_functions"]))
        or not set(value["allowed_functions"]) <= _SAFE_FUNCTIONS
    ):
        raise SymbolicGeneratorError("template descriptor safe-function boundary changed")
    if not isinstance(value["expression_srepr"], str) or not value["expression_srepr"]:
        raise SymbolicGeneratorError("template descriptor expression_srepr is invalid")
    body = {key: child for key, child in value.items() if key != "content_sha256"}
    _sha256(value["content_sha256"], "template descriptor content_sha256")
    if canonical_sha256(body) != value["content_sha256"]:
        raise SymbolicGeneratorError("template descriptor canonical hash changed")
    return dict(value)


@dataclass(frozen=True, slots=True)
class SymbolicGeneratorBudget:
    max_templates: int
    max_coefficients_per_axis: int
    max_candidates: int
    max_work_items: int

    def __post_init__(self) -> None:
        _bounded_positive(self.max_templates, _MAX_TEMPLATES, "max_templates")
        _bounded_positive(
            self.max_coefficients_per_axis,
            _MAX_COEFFICIENTS_PER_AXIS,
            "max_coefficients_per_axis",
        )
        _bounded_positive(self.max_candidates, _MAX_CANDIDATES, "max_candidates")
        _bounded_positive(self.max_work_items, _MAX_WORK_ITEMS, "max_work_items")

    def to_dict(self) -> dict[str, int]:
        return {
            "max_templates": self.max_templates,
            "max_coefficients_per_axis": self.max_coefficients_per_axis,
            "max_candidates": self.max_candidates,
            "max_work_items": self.max_work_items,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolicGeneratorBudget:
        _exact_keys(
            value,
            {
                "max_templates",
                "max_coefficients_per_axis",
                "max_candidates",
                "max_work_items",
            },
            "symbolic generator budget",
        )
        return cls(
            max_templates=value["max_templates"],
            max_coefficients_per_axis=value["max_coefficients_per_axis"],
            max_candidates=value["max_candidates"],
            max_work_items=value["max_work_items"],
        )


def _sympy_to_ir(value: sp.Expr, variable_types: Mapping[str, MathType]) -> Expression:
    if value.is_Integer:
        return literal(int(value))
    if value.is_Rational:
        return literal(Fraction(int(value.p), int(value.q)))
    if value.is_Symbol:
        name = str(value)
        if name not in variable_types:
            raise SymbolicGeneratorError(f"instantiated expression has unknown symbol: {name}")
        return symbol(name, variable_types[name])
    if value.is_Add:
        return add(*(_sympy_to_ir(item, variable_types) for item in value.args))
    if value.is_Mul:
        return multiply(*(_sympy_to_ir(item, variable_types) for item in value.args))
    if value.is_Pow:
        return power(*(_sympy_to_ir(item, variable_types) for item in value.args))
    if value.is_Function and value.func.__name__ in _SAFE_FUNCTIONS:
        return call(
            _FUNCTION_IR_NAMES[value.func.__name__],
            *(_sympy_to_ir(item, variable_types) for item in value.args),
        )
    raise SymbolicGeneratorError(f"instantiated SymPy node is unsupported: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class SymbolicOriginRecord:
    work_index: int
    template_id: str
    template_sha256: str
    assignment: tuple[tuple[str, ExactRational], ...]
    canonical_expression_sha256: str
    candidate_artifact_id: str
    disposition: str
    content_sha256: str
    schema_version: str = ORIGIN_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != ORIGIN_SCHEMA:
            raise SymbolicGeneratorError("origin schema_version changed")
        if (
            isinstance(self.work_index, bool)
            or not isinstance(self.work_index, int)
            or self.work_index < 0
        ):
            raise SymbolicGeneratorError("origin work_index must be nonnegative")
        _identifier(self.template_id, "origin template_id")
        _sha256(self.template_sha256, "origin template_sha256")
        names = tuple(name for name, _ in self.assignment)
        if names != tuple(sorted(set(names))):
            raise SymbolicGeneratorError("origin assignment must be unique and sorted")
        _sha256(self.canonical_expression_sha256, "canonical_expression_sha256")
        if not isinstance(
            self.candidate_artifact_id, str
        ) or not self.candidate_artifact_id.startswith("sig-"):
            raise SymbolicGeneratorError("origin candidate_artifact_id is invalid")
        if self.disposition not in {"representative", "deduplicated_equivalent"}:
            raise SymbolicGeneratorError("origin disposition changed")
        _sha256(self.content_sha256, "origin content_sha256")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise SymbolicGeneratorError("origin canonical hash changed")

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "work_index": self.work_index,
            "template_id": self.template_id,
            "template_sha256": self.template_sha256,
            "assignment": [
                {"symbol": name, "value": coefficient.to_dict()}
                for name, coefficient in self.assignment
            ],
            "canonical_expression_sha256": self.canonical_expression_sha256,
            "candidate_artifact_id": self.candidate_artifact_id,
            "disposition": self.disposition,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolicOriginRecord:
        _exact_keys(
            value,
            {
                "schema_version",
                "work_index",
                "template_id",
                "template_sha256",
                "assignment",
                "canonical_expression_sha256",
                "candidate_artifact_id",
                "disposition",
                "content_sha256",
            },
            "symbolic origin",
        )
        if not isinstance(value["assignment"], list):
            raise SymbolicGeneratorError("origin assignment must be an array")
        assignment = []
        for item in value["assignment"]:
            _exact_keys(item, {"symbol", "value"}, "origin assignment item")
            assignment.append((str(item["symbol"]), ExactRational.from_dict(item["value"])))
        return cls(
            work_index=value["work_index"],
            template_id=str(value["template_id"]),
            template_sha256=str(value["template_sha256"]),
            assignment=tuple(assignment),
            canonical_expression_sha256=str(value["canonical_expression_sha256"]),
            candidate_artifact_id=str(value["candidate_artifact_id"]),
            disposition=str(value["disposition"]),
            content_sha256=str(value["content_sha256"]),
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True, slots=True)
class SymbolicGenerationBatch:
    domain_pack: DomainPackRef
    budget: SymbolicGeneratorBudget
    template_descriptors: tuple[Mapping[str, Any], ...]
    coefficient_values: tuple[ExactRational, ...]
    candidates: tuple[CandidateArtifact, ...]
    origins: tuple[SymbolicOriginRecord, ...]
    generated_before_deduplication: int
    duplicates_removed: int
    lineage_sha256: str
    content_sha256: str
    scope: str = SCOPE
    schema_version: str = BATCH_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != BATCH_SCHEMA or self.scope != SCOPE:
            raise SymbolicGeneratorError("symbolic batch schema or scope changed")
        descriptors = tuple(
            _validate_template_descriptor(item) for item in self.template_descriptors
        )
        object.__setattr__(self, "template_descriptors", descriptors)
        template_ids = tuple(item["template_id"] for item in descriptors)
        if template_ids != tuple(sorted(set(template_ids))):
            raise SymbolicGeneratorError("template descriptors must be unique and sorted")
        if not descriptors or len(descriptors) > self.budget.max_templates:
            raise SymbolicGeneratorError("batch template budget violated")
        if not self.coefficient_values or self.coefficient_values != tuple(
            sorted(set(self.coefficient_values), key=lambda item: item.fraction)
        ):
            raise SymbolicGeneratorError("coefficient values must be unique and sorted")
        if len(self.coefficient_values) > self.budget.max_coefficients_per_axis:
            raise SymbolicGeneratorError("batch coefficient budget violated")
        candidate_ids = tuple(item.artifact_id for item in self.candidates)
        if candidate_ids != tuple(sorted(set(candidate_ids))):
            raise SymbolicGeneratorError("batch candidates must be unique and sorted")
        for candidate in self.candidates:
            try:
                candidate.validate()
            except SchemaViolation as error:
                raise SymbolicGeneratorError(
                    "candidate artifact failed canonical validation"
                ) from error
            if (
                candidate.kind is not ArtifactKind.FORMULA
                or candidate.claims != ("generated_candidate",)
                or candidate.assumptions != ("bounded exact symbolic template scope",)
                or set(candidate.representation)
                != {
                    "generator",
                    "canonicalizer",
                    "canonical_expression_sha256",
                    "math_expression_sha256",
                    "canonical_expression",
                    "typed_variables",
                    "origin_count",
                }
                or candidate.representation["generator"] != "bounded_exact_symbolic"
                or candidate.representation["canonicalizer"] != "math-pack-exact-rational-1.0"
            ):
                raise SymbolicGeneratorError("candidate symbolic scope changed")
        if len(self.candidates) > self.budget.max_candidates:
            raise SymbolicGeneratorError("batch candidate budget violated")
        if self.generated_before_deduplication != len(self.origins):
            raise SymbolicGeneratorError("batch generated count changed")
        if self.generated_before_deduplication > self.budget.max_work_items:
            raise SymbolicGeneratorError("batch work budget violated")
        if self.duplicates_removed != len(self.origins) - len(self.candidates):
            raise SymbolicGeneratorError("batch duplicate count changed")
        if tuple(item.work_index for item in self.origins) != tuple(range(len(self.origins))):
            raise SymbolicGeneratorError("origin work indices must be contiguous and ordered")
        template_receipts = {item["template_id"]: item["content_sha256"] for item in descriptors}
        if any(
            template_receipts.get(item.template_id) != item.template_sha256 for item in self.origins
        ):
            raise SymbolicGeneratorError("origin template binding changed")
        by_candidate = {item.artifact_id: item for item in self.candidates}
        if {item.candidate_artifact_id for item in self.origins} != set(by_candidate):
            raise SymbolicGeneratorError("origin candidate references changed")
        expression_candidates = {
            item.representation["canonical_expression_sha256"]: item.artifact_id
            for item in self.candidates
        }
        if len(expression_candidates) != len(self.candidates) or any(
            expression_candidates.get(item.canonical_expression_sha256)
            != item.candidate_artifact_id
            for item in self.origins
        ):
            raise SymbolicGeneratorError("origin canonical candidate binding changed")
        first_by_hash: dict[str, int] = {}
        for origin in self.origins:
            first_by_hash.setdefault(origin.canonical_expression_sha256, origin.work_index)
            expected = (
                "representative"
                if first_by_hash[origin.canonical_expression_sha256] == origin.work_index
                else "deduplicated_equivalent"
            )
            if origin.disposition != expected:
                raise SymbolicGeneratorError("origin disposition does not match canonical dedup")
        _sha256(self.lineage_sha256, "batch lineage_sha256")
        _sha256(self.content_sha256, "batch content_sha256")
        if self.lineage_sha256 != canonical_sha256(self._lineage_body()):
            raise SymbolicGeneratorError("batch lineage hash changed")
        if self.content_sha256 != canonical_sha256(self._body()):
            raise SymbolicGeneratorError("batch canonical hash changed")

    def _lineage_body(self) -> dict[str, Any]:
        return {
            "schema_version": BATCH_SCHEMA,
            "domain_pack": self.domain_pack.to_dict(),
            "budget": self.budget.to_dict(),
            "template_receipts": [item["content_sha256"] for item in self.template_descriptors],
            "coefficient_values": [item.to_dict() for item in self.coefficient_values],
            "candidate_refs": [item.ref.to_dict() for item in self.candidates],
            "origin_receipts": [item.content_sha256 for item in self.origins],
        }

    def _body(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scope": self.scope,
            "domain_pack": self.domain_pack.to_dict(),
            "budget": self.budget.to_dict(),
            "template_descriptors": list(self.template_descriptors),
            "coefficient_values": [item.to_dict() for item in self.coefficient_values],
            "candidates": [item.to_dict() for item in self.candidates],
            "origins": [item.to_dict() for item in self.origins],
            "generated_before_deduplication": self.generated_before_deduplication,
            "duplicates_removed": self.duplicates_removed,
            "lineage_sha256": self.lineage_sha256,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._body(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SymbolicGenerationBatch:
        _exact_keys(
            value,
            {
                "schema_version",
                "scope",
                "domain_pack",
                "budget",
                "template_descriptors",
                "coefficient_values",
                "candidates",
                "origins",
                "generated_before_deduplication",
                "duplicates_removed",
                "lineage_sha256",
                "content_sha256",
            },
            "symbolic generation batch",
        )
        for key in ("template_descriptors", "coefficient_values", "candidates", "origins"):
            if not isinstance(value[key], list):
                raise SymbolicGeneratorError(f"batch {key} must be an array")
        try:
            domain_pack = DomainPackRef.from_dict(value["domain_pack"])
            candidates = tuple(CandidateArtifact.from_dict(item) for item in value["candidates"])
        except (SchemaViolation, TypeError, ValueError) as error:
            raise SymbolicGeneratorError("batch Sigma Core binding failed validation") from error
        return cls(
            domain_pack=domain_pack,
            budget=SymbolicGeneratorBudget.from_dict(value["budget"]),
            template_descriptors=tuple(value["template_descriptors"]),
            coefficient_values=tuple(
                ExactRational.from_dict(item) for item in value["coefficient_values"]
            ),
            candidates=candidates,
            origins=tuple(SymbolicOriginRecord.from_dict(item) for item in value["origins"]),
            generated_before_deduplication=value["generated_before_deduplication"],
            duplicates_removed=value["duplicates_removed"],
            lineage_sha256=str(value["lineage_sha256"]),
            content_sha256=str(value["content_sha256"]),
            scope=str(value["scope"]),
            schema_version=str(value["schema_version"]),
        )


class SymbolicCandidateGenerator:
    """Enumerate a finite exact coefficient grid and emit canonical Sigma candidates."""

    @staticmethod
    def generate(
        templates: Sequence[SymbolicTemplate],
        coefficient_values: Sequence[int | Fraction],
        *,
        domain_pack: DomainPackRef,
        budget: SymbolicGeneratorBudget,
    ) -> SymbolicGenerationBatch:
        if not templates:
            raise SymbolicGeneratorError("at least one symbolic template is required")
        if len(templates) > budget.max_templates:
            raise SymbolicGeneratorError("templates exceed max_templates")
        if not isinstance(domain_pack, DomainPackRef):
            raise SymbolicGeneratorError("domain_pack must be a Sigma Core DomainPackRef")
        template_ids = tuple(item.template_id for item in templates)
        if len(set(template_ids)) != len(template_ids):
            raise SymbolicGeneratorError("template IDs contain duplicates")
        ordered_templates = tuple(sorted(templates, key=lambda item: item.template_id))
        exact_values = tuple(
            sorted(
                {ExactRational.create(value) for value in coefficient_values},
                key=lambda x: x.fraction,
            )
        )
        if not exact_values:
            raise SymbolicGeneratorError("coefficient grid must be nonempty")
        if len(exact_values) > budget.max_coefficients_per_axis:
            raise SymbolicGeneratorError("coefficient grid exceeds max_coefficients_per_axis")
        work_count = 0
        for template in ordered_templates:
            work_count += len(exact_values) ** len(template.coefficient_symbols)
            if work_count > budget.max_work_items:
                raise SymbolicGeneratorError("coefficient expansion exceeds max_work_items")

        raw: list[dict[str, Any]] = []
        work_index = 0
        for template in ordered_templates:
            variable_types = {item.name: item.math_type for item in template.variables}
            for values in product(exact_values, repeat=len(template.coefficient_symbols)):
                assignment = tuple(zip(template.coefficient_symbols, values, strict=True))
                substitutions = {
                    sp.Symbol(name): sp.Rational(value.numerator, value.denominator)
                    for name, value in assignment
                }
                instantiated = template.expression.xreplace(substitutions)
                if not {str(item) for item in instantiated.free_symbols} <= set(variable_types):
                    raise SymbolicGeneratorError(
                        "instantiated expression contains a symbol outside the typed variables"
                    )
                try:
                    canonical = canonicalize_expression(_sympy_to_ir(instantiated, variable_types))
                    representation = canonical_data(canonical)
                    math_expression_hash = expression_sha256(canonical)
                    canonical_expression_hash = canonical_sha256(
                        {
                            "canonical_expression": representation,
                            "typed_variables": [item.to_dict() for item in template.variables],
                        }
                    )
                except (CanonicalizationError, TypeError, ValueError) as error:
                    raise SymbolicGeneratorError(
                        "Math Pack canonicalization failed closed"
                    ) from error
                origin_body = {
                    "work_index": work_index,
                    "template_id": template.template_id,
                    "template_sha256": template.content_sha256,
                    "assignment": [
                        {"symbol": name, "value": value.to_dict()} for name, value in assignment
                    ],
                }
                raw.append(
                    {
                        **origin_body,
                        "origin_sha256": canonical_sha256(origin_body),
                        "canonical_expression_sha256": canonical_expression_hash,
                        "math_expression_sha256": math_expression_hash,
                        "canonical_expression": representation,
                        "variables": [item.to_dict() for item in template.variables],
                        "assignment_values": assignment,
                    }
                )
                work_index += 1

        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in raw:
            grouped.setdefault(item["canonical_expression_sha256"], []).append(item)
        if len(grouped) > budget.max_candidates:
            raise SymbolicGeneratorError("unique canonical expressions exceed max_candidates")
        template_root = canonical_sha256([item.descriptor() for item in ordered_templates])
        candidate_by_expression: dict[str, CandidateArtifact] = {}
        for canonical_hash, origins in sorted(grouped.items()):
            first = origins[0]
            origin_descriptors = [
                {
                    "work_index": item["work_index"],
                    "template_id": item["template_id"],
                    "template_sha256": item["template_sha256"],
                    "assignment": [
                        {"symbol": name, "value": value.to_dict()}
                        for name, value in item["assignment_values"]
                    ],
                    "origin_sha256": item["origin_sha256"],
                }
                for item in origins
            ]
            provenance = ProvenanceRecord.create(
                domain_pack,
                {
                    "generator_schema": BATCH_SCHEMA,
                    "template_registry_sha256": template_root,
                    "canonical_expression_sha256": canonical_hash,
                    "math_expression_sha256": first["math_expression_sha256"],
                    "origins": origin_descriptors,
                },
            )
            candidate_by_expression[canonical_hash] = CandidateArtifact.create(
                ArtifactKind.FORMULA,
                f"bounded symbolic candidate {canonical_hash[:24]}",
                {
                    "generator": "bounded_exact_symbolic",
                    "canonicalizer": "math-pack-exact-rational-1.0",
                    "canonical_expression_sha256": canonical_hash,
                    "math_expression_sha256": first["math_expression_sha256"],
                    "canonical_expression": first["canonical_expression"],
                    "typed_variables": first["variables"],
                    "origin_count": len(origins),
                },
                provenance,
                assumptions=("bounded exact symbolic template scope",),
                claims=("generated_candidate",),
            )
        origins = []
        seen: set[str] = set()
        for item in raw:
            canonical_hash = item["canonical_expression_sha256"]
            candidate = candidate_by_expression[canonical_hash]
            disposition = "deduplicated_equivalent" if canonical_hash in seen else "representative"
            seen.add(canonical_hash)
            body = {
                "schema_version": ORIGIN_SCHEMA,
                "work_index": item["work_index"],
                "template_id": item["template_id"],
                "template_sha256": item["template_sha256"],
                "assignment": [
                    {"symbol": name, "value": value.to_dict()}
                    for name, value in item["assignment_values"]
                ],
                "canonical_expression_sha256": canonical_hash,
                "candidate_artifact_id": candidate.artifact_id,
                "disposition": disposition,
            }
            origins.append(
                SymbolicOriginRecord(
                    work_index=item["work_index"],
                    template_id=item["template_id"],
                    template_sha256=item["template_sha256"],
                    assignment=item["assignment_values"],
                    canonical_expression_sha256=canonical_hash,
                    candidate_artifact_id=candidate.artifact_id,
                    disposition=disposition,
                    content_sha256=canonical_sha256(body),
                )
            )
        candidates = tuple(
            sorted(candidate_by_expression.values(), key=lambda item: item.artifact_id)
        )
        descriptors = tuple(item.descriptor() for item in ordered_templates)
        provisional = object.__new__(SymbolicGenerationBatch)
        fields = {
            "domain_pack": domain_pack,
            "budget": budget,
            "template_descriptors": descriptors,
            "coefficient_values": exact_values,
            "candidates": candidates,
            "origins": tuple(origins),
            "generated_before_deduplication": len(raw),
            "duplicates_removed": len(raw) - len(candidates),
            "scope": SCOPE,
            "schema_version": BATCH_SCHEMA,
        }
        for name, value in fields.items():
            object.__setattr__(provisional, name, value)
        lineage = canonical_sha256(provisional._lineage_body())
        object.__setattr__(provisional, "lineage_sha256", lineage)
        content = canonical_sha256(provisional._body())
        return SymbolicGenerationBatch(**fields, lineage_sha256=lineage, content_sha256=content)
