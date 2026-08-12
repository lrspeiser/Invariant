from __future__ import annotations

from fractions import Fraction

import pytest

from sigma_theory_compiler.math_types import (
    COMPLEX,
    INTEGER,
    RATIONAL,
    REAL,
    ExactComplex,
    FunctionType,
    FunctionValue,
    GraphType,
    GraphValue,
    IntegerType,
    MathTypeError,
    MatrixType,
    PolynomialType,
    PolynomialValue,
    RationalType,
    SequenceType,
    SetType,
    validate_value,
)


def test_scalar_types_are_exact_and_bounded() -> None:
    assert validate_value(IntegerType(-2, 2), 2)
    assert not validate_value(INTEGER, True)
    assert validate_value(RationalType(Fraction(-1, 2), Fraction(1, 2)), Fraction(1, 3))
    assert not validate_value(RATIONAL, 0.5)
    assert validate_value(REAL, Fraction(1, 3))
    assert validate_value(COMPLEX, ExactComplex(Fraction(1, 3), Fraction(2, 5)))
    assert not validate_value(COMPLEX, complex(float("inf"), 0))
    with pytest.raises(MathTypeError):
        IntegerType(3, 2)


def test_sequence_set_and_matrix_types_validate_shape_and_elements() -> None:
    assert validate_value(SequenceType(INTEGER, 3), (1, 2, 3))
    assert not validate_value(SequenceType(INTEGER, 2), (1, 2, 3))
    assert validate_value(SetType(INTEGER, 2), frozenset({1, 2}))
    assert not validate_value(SetType(INTEGER, 1), {1, 2})
    assert validate_value(MatrixType(RATIONAL, 2, 2), [[1, Fraction(1, 2)], [0, -1]])
    assert not validate_value(MatrixType(INTEGER, 2, 2), [[1, 2], [3]])


def test_polynomial_function_and_graph_values_are_closed_world() -> None:
    polynomial_type = PolynomialType(RATIONAL, ("x", "y"), maximum_total_degree=2)
    polynomial = PolynomialValue((((2, 0), Fraction(1, 2)), ((0, 1), 3)))
    assert validate_value(polynomial_type, polynomial)
    assert not validate_value(polynomial_type, PolynomialValue((((2, 1), 1),)))

    function_type = FunctionType((INTEGER,), RATIONAL)
    assert validate_value(
        function_type,
        FunctionValue((((0,), Fraction(1, 2)), ((1,), Fraction(3, 2)))),
    )
    assert not validate_value(function_type, lambda value: value)

    graph_type = GraphType(INTEGER, directed=False)
    graph = GraphValue(frozenset({1, 2}), frozenset({(1, 2, None), (2, 1, None)}))
    assert validate_value(graph_type, graph)
    assert not validate_value(
        graph_type,
        GraphValue(frozenset({1, 2}), frozenset({(1, 2, None)})),
    )
