"""Closed, immutable mathematical type descriptors for Math Pack v1."""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Any


class MathTypeError(ValueError):
    """Raised when a type descriptor or typed value is invalid."""


@dataclass(frozen=True, slots=True)
class ExactComplex:
    """A complex number whose real and imaginary parts remain rational."""

    real: Fraction = Fraction(0)
    imaginary: Fraction = Fraction(0)

    def __init__(
        self,
        real: int | Fraction = Fraction(0),
        imaginary: int | Fraction = Fraction(0),
    ) -> None:
        object.__setattr__(self, "real", Fraction(real))
        object.__setattr__(self, "imaginary", Fraction(imaginary))


@dataclass(frozen=True, slots=True)
class IntegerType:
    minimum: int | None = None
    maximum: int | None = None

    def __post_init__(self) -> None:
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise MathTypeError("integer minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class RationalType:
    minimum: Fraction | None = None
    maximum: Fraction | None = None

    def __post_init__(self) -> None:
        if self.minimum is not None:
            object.__setattr__(self, "minimum", Fraction(self.minimum))
        if self.maximum is not None:
            object.__setattr__(self, "maximum", Fraction(self.maximum))
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise MathTypeError("rational minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class RealType:
    minimum: int | Fraction | float | None = None
    maximum: int | Fraction | float | None = None

    def __post_init__(self) -> None:
        for label, value in (("minimum", self.minimum), ("maximum", self.maximum)):
            if isinstance(value, bool) or (isinstance(value, float) and not math.isfinite(value)):
                raise MathTypeError(f"real {label} must be finite")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise MathTypeError("real minimum exceeds maximum")


@dataclass(frozen=True, slots=True)
class ComplexType:
    pass


@dataclass(frozen=True, slots=True)
class SequenceType:
    element_type: MathType
    length: int | None = None

    def __post_init__(self) -> None:
        if self.length is not None and self.length < 0:
            raise MathTypeError("sequence length cannot be negative")


@dataclass(frozen=True, slots=True)
class SetType:
    element_type: MathType
    maximum_cardinality: int | None = None

    def __post_init__(self) -> None:
        if self.maximum_cardinality is not None and self.maximum_cardinality < 0:
            raise MathTypeError("set maximum cardinality cannot be negative")


@dataclass(frozen=True, slots=True)
class MatrixType:
    element_type: MathType
    rows: int
    columns: int

    def __post_init__(self) -> None:
        if self.rows < 0 or self.columns < 0:
            raise MathTypeError("matrix dimensions cannot be negative")


@dataclass(frozen=True, slots=True)
class PolynomialType:
    coefficient_type: MathType
    variables: tuple[str, ...]
    maximum_total_degree: int | None = None

    def __post_init__(self) -> None:
        if not self.variables or any(not name.isidentifier() for name in self.variables):
            raise MathTypeError("polynomial variables must be nonempty identifiers")
        if len(set(self.variables)) != len(self.variables):
            raise MathTypeError("polynomial variables must be unique")
        if self.maximum_total_degree is not None and self.maximum_total_degree < 0:
            raise MathTypeError("polynomial maximum degree cannot be negative")


@dataclass(frozen=True, slots=True)
class FunctionType:
    domain: tuple[MathType, ...]
    codomain: MathType


@dataclass(frozen=True, slots=True)
class GraphType:
    vertex_type: MathType
    edge_type: MathType | None = None
    directed: bool = False
    allow_self_loops: bool = False


MathType = (
    IntegerType
    | RationalType
    | RealType
    | ComplexType
    | SequenceType
    | SetType
    | MatrixType
    | PolynomialType
    | FunctionType
    | GraphType
)


@dataclass(frozen=True, slots=True)
class PolynomialValue:
    """Sparse multivariate polynomial as ``(exponents, coefficient)`` terms."""

    terms: tuple[tuple[tuple[int, ...], Any], ...]


@dataclass(frozen=True, slots=True)
class FunctionValue:
    """Finite function table; unlike a callable, every represented value is auditable."""

    entries: tuple[tuple[tuple[Any, ...], Any], ...]


@dataclass(frozen=True, slots=True)
class GraphValue:
    """Finite graph with edges represented as ``(source, target, label)`` triples."""

    vertices: frozenset[Any]
    edges: frozenset[tuple[Any, Any, Any | None]]


INTEGER = IntegerType()
RATIONAL = RationalType()
REAL = RealType()
COMPLEX = ComplexType()


def _within(value: Any, minimum: Any | None, maximum: Any | None) -> bool:
    return (minimum is None or value >= minimum) and (maximum is None or value <= maximum)


def validate_value(type_spec: MathType, value: Any) -> bool:
    """Return whether *value* exactly inhabits *type_spec*.

    Composite validation is deliberately closed: functions must be finite ``FunctionValue``
    tables and polynomials/graphs must use their explicit immutable value representations.
    """

    if isinstance(type_spec, IntegerType):
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and _within(value, type_spec.minimum, type_spec.maximum)
        )
    if isinstance(type_spec, RationalType):
        if isinstance(value, bool) or not isinstance(value, (int, Fraction)):
            return False
        return _within(Fraction(value), type_spec.minimum, type_spec.maximum)
    if isinstance(type_spec, RealType):
        if isinstance(value, bool) or not isinstance(value, (int, Fraction, float)):
            return False
        if isinstance(value, float) and not math.isfinite(value):
            return False
        return _within(value, type_spec.minimum, type_spec.maximum)
    if isinstance(type_spec, ComplexType):
        if isinstance(value, bool):
            return False
        if isinstance(value, (int, Fraction, ExactComplex)):
            return True
        return isinstance(value, (float, complex)) and (
            math.isfinite(value)
            if isinstance(value, float)
            else math.isfinite(value.real) and math.isfinite(value.imag)
        )
    if isinstance(type_spec, SequenceType):
        return (
            isinstance(value, (tuple, list))
            and (type_spec.length is None or len(value) == type_spec.length)
            and all(validate_value(type_spec.element_type, item) for item in value)
        )
    if isinstance(type_spec, SetType):
        return (
            isinstance(value, (set, frozenset))
            and (
                type_spec.maximum_cardinality is None or len(value) <= type_spec.maximum_cardinality
            )
            and all(validate_value(type_spec.element_type, item) for item in value)
        )
    if isinstance(type_spec, MatrixType):
        return (
            isinstance(value, (tuple, list))
            and len(value) == type_spec.rows
            and all(
                isinstance(row, (tuple, list))
                and len(row) == type_spec.columns
                and all(validate_value(type_spec.element_type, item) for item in row)
                for row in value
            )
        )
    if isinstance(type_spec, PolynomialType):
        if not isinstance(value, PolynomialValue):
            return False
        seen: set[tuple[int, ...]] = set()
        for exponents, coefficient in value.terms:
            if (
                len(exponents) != len(type_spec.variables)
                or any(power < 0 for power in exponents)
                or exponents in seen
                or not validate_value(type_spec.coefficient_type, coefficient)
                or (
                    type_spec.maximum_total_degree is not None
                    and sum(exponents) > type_spec.maximum_total_degree
                )
            ):
                return False
            seen.add(exponents)
        return True
    if isinstance(type_spec, FunctionType):
        if not isinstance(value, FunctionValue):
            return False
        seen_arguments: set[tuple[Any, ...]] = set()
        for arguments, result in value.entries:
            if (
                len(arguments) != len(type_spec.domain)
                or arguments in seen_arguments
                or not all(
                    validate_value(argument_type, argument)
                    for argument_type, argument in zip(type_spec.domain, arguments, strict=True)
                )
                or not validate_value(type_spec.codomain, result)
            ):
                return False
            seen_arguments.add(arguments)
        return True
    if isinstance(type_spec, GraphType):
        if not isinstance(value, GraphValue) or not all(
            validate_value(type_spec.vertex_type, vertex) for vertex in value.vertices
        ):
            return False
        for source, target, label in value.edges:
            if source not in value.vertices or target not in value.vertices:
                return False
            if not type_spec.allow_self_loops and source == target:
                return False
            if type_spec.edge_type is None:
                if label is not None:
                    return False
            elif not validate_value(type_spec.edge_type, label):
                return False
            if not type_spec.directed and (target, source, label) not in value.edges:
                return False
        return True
    raise TypeError(f"unsupported mathematical type descriptor: {type(type_spec).__name__}")
