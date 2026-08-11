import sympy as sp

from sigma_theory_compiler.quartic_tc2_fourth_jet_parallel_kernel import (
    _combine_directions,
    _direction_key,
)


def test_direction_aggregation_is_exact_and_canonical() -> None:
    directions = (
        {"x": sp.Integer(1), "y": sp.Integer(2)},
        {"x": sp.Integer(-1), "z": sp.Rational(1, 2)},
    )
    combined = _combine_directions(directions, (0, 1))
    assert combined == {"y": 2, "z": sp.Rational(1, 2)}
    assert _direction_key(combined) == (("y", "2"), ("z", "1/2"))
