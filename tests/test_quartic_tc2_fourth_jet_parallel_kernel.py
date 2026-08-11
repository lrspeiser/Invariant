import pytest
import sympy as sp

from sigma_theory_compiler.quartic_tc2_fourth_jet_parallel_kernel import (
    QuarticTC2FourthJetParallelKernelError,
    _combine_directions,
    _direction_key,
    _directional_fourth_payload,
)


def test_direction_aggregation_is_exact_and_canonical() -> None:
    directions = (
        {"x": sp.Integer(1), "y": sp.Integer(2)},
        {"x": sp.Integer(-1), "z": sp.Rational(1, 2)},
    )
    combined = _combine_directions(directions, (0, 1))
    assert combined == {"y": 2, "z": sp.Rational(1, 2)}
    assert _direction_key(combined) == (("y", "2"), ("z", "1/2"))


def _fake_packet(*, failed_order: int | None) -> dict:
    zero = sp.zeros(1)
    orders = [
        {
            "order": order,
            "solvable": order != failed_order,
            "residual_zero": order != failed_order,
            "rhs": sp.Matrix([[order]]),
        }
        for order in range(1, 5)
    ]
    return {
        "orders": orders,
        "physical": [zero for _ in range(5)],
        "energy": [zero for _ in range(5)],
        "block": [zero for _ in range(5)],
    }


def test_order_four_incompatibility_is_retained_as_rhs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sigma_theory_compiler.quartic_tc2_fourth_jet_parallel_kernel."
        "directional_engine._directional_taylor_packet",
        lambda _: _fake_packet(failed_order=4),
    )
    payload = _directional_fourth_payload({"x": sp.Integer(1)})
    assert payload["fourth_Sylvester_RHS"] == sp.Matrix([[96]])


def test_lower_order_failure_remains_infrastructure_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "sigma_theory_compiler.quartic_tc2_fourth_jet_parallel_kernel."
        "directional_engine._directional_taylor_packet",
        lambda _: _fake_packet(failed_order=3),
    )
    with pytest.raises(
        QuarticTC2FourthJetParallelKernelError,
        match="mandatory orders one through three",
    ):
        _directional_fourth_payload({"x": sp.Integer(1)})
