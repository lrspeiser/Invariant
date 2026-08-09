from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable

import sympy as sp

from .compiler import TheoryCompiler
from .dimensions import Dimension, assert_dimensionless_invariants, normalized_invariant_dimensions
from .gates import algebraic_gates, sampled_static_convexity
from .grammar import Q, X, Z, enumerate_expressions


def _check(name: str, claim: str, function: Callable[[], tuple[bool, dict]]) -> dict:
    try:
        passed, evidence = function()
        return {"name": name, "claim": claim, "status": "pass" if passed else "fail", "evidence": evidence}
    except Exception as error:  # Validation must report a failure rather than hide it.
        return {
            "name": name,
            "claim": claim,
            "status": "fail",
            "evidence": {"exception": type(error).__name__, "message": str(error)},
        }


def _dimensions() -> tuple[bool, dict]:
    good = normalized_invariant_dimensions()
    assert_dimensionless_invariants(good)
    deliberate_rejection = False
    try:
        assert_dimensionless_invariants({"bad": Dimension(length=1)})
    except ValueError:
        deliberate_rejection = True
    return deliberate_rejection, {
        "normalized": {name: value.as_dict() for name, value in good.items()},
        "dimensionful_control_rejected": deliberate_rejection,
    }


def _known_enumeration() -> tuple[bool, dict]:
    expressions, counts = enumerate_expressions(["x", "q"], [], ["add"], 3)
    actual = {str(item.expression) for item in expressions}
    expected = {"x", "q", "2*x", "q + x", "2*q"}
    return actual == expected and counts["unique"] == 5, {
        "expected": sorted(expected),
        "actual": sorted(actual),
        "counts": counts,
    }


def _determinism() -> tuple[bool, dict]:
    arguments = (["x", "q", "z"], ["saturate"], ["add", "multiply"], 4)
    first, first_counts = enumerate_expressions(*arguments)
    second, second_counts = enumerate_expressions(*arguments)
    first_keys = [item.canonical for item in first]
    second_keys = [item.canonical for item in second]
    return first_keys == second_keys and first_counts == second_counts, {
        "first_sha256": __import__("hashlib").sha256("\n".join(first_keys).encode()).hexdigest(),
        "second_sha256": __import__("hashlib").sha256("\n".join(second_keys).encode()).hexdigest(),
        "counts": first_counts,
    }


def _gate_status(expression: sp.Expr, name: str) -> str:
    return next(gate.status for gate in algebraic_gates(expression, 4, 5) if gate.name == name)


def _screening_controls() -> tuple[bool, dict]:
    passing = _gate_status(Q, "high_field_newtonian_limit")
    failing = _gate_status(X * Q, "high_field_newtonian_limit")
    return passing == "pass" and failing == "reject", {
        "q_control": passing,
        "x_times_q_control": failing,
    }


def _state_controls() -> tuple[bool, dict]:
    passing = _gate_status(Q, "new_spatial_state_information")
    failing = _gate_status(X, "new_spatial_state_information")
    return passing == "pass" and failing == "reject", {
        "q_control": passing,
        "x_only_control": failing,
    }


def _convexity_controls() -> tuple[bool, dict]:
    samples = {"d": [0.1, 1.0], "p": [0.0, 1.0], "state": [0.0]}
    positive = sampled_static_convexity(Q, 0.1, samples, 1e-9)
    negative = sampled_static_convexity(Q, -0.1, samples, 1e-9)
    return positive.status == "pass" and negative.status == "reject", {
        "positive_elasticity": positive.as_dict(),
        "negative_elasticity": negative.as_dict(),
    }


def _euler_lagrange_control() -> tuple[bool, dict]:
    equation = TheoryCompiler.field_equation(Q, 0.1)
    radius = sp.symbols("r", real=True)
    displacement = sp.Function("D")(radius)
    a_sigma, length_sigma, z_zero = sp.symbols(
        "a_sigma L_sigma Z_0", positive=True, finite=True
    )
    expected = displacement - sp.Float(0.2) * length_sigma**2 * sp.diff(
        displacement, radius, 2
    )
    residual = sp.simplify(equation.rhs - expected)
    return residual == 0, {
        "derived_rhs": str(equation.rhs),
        "hand_derived_rhs": str(expected),
        "symbolic_residual": str(residual),
        "unused_scale_sentinels": [str(a_sigma), str(z_zero)],
    }


def _newton_reference_recovery() -> tuple[bool, dict]:
    equation = TheoryCompiler.field_equation(sp.Integer(0), 0.0)
    radius = sp.symbols("r", real=True)
    displacement = sp.Function("D")(radius)
    residual = sp.simplify(equation.rhs - displacement)
    return residual == 0, {
        "base_action": "H = D^2/2",
        "expected_constitutive_equation": "dW_dr = D(r)",
        "derived_equation": str(equation),
        "symbolic_residual": str(residual),
        "interpretation": "Together with div D = 4*pi*G*rho_b, this is the frozen Newtonian static control.",
    }


def run_validation() -> dict:
    checks = [
        _check("dimensions", "Normalized atoms pass and a dimensionful control is rejected.", _dimensions),
        _check("known_enumeration", "A hand-countable grammar yields exactly its five known forms.", _known_enumeration),
        _check("determinism", "Two independent runs produce identical canonical keys and counts.", _determinism),
        _check("screening_controls", "q screens at fixed state while x*q is correctly rejected.", _screening_controls),
        _check("state_controls", "q carries new state information while flux-only x is rejected.", _state_controls),
        _check("convexity_controls", "Positive elasticity passes and negative elasticity fails.", _convexity_controls),
        _check("euler_lagrange", "The symbolic field equation matches a hand-derived q action.", _euler_lagrange_control),
        _check("newton_reference", "The base action exactly recovers the Newtonian constitutive law.", _newton_reference_recovery),
    ]
    passed = sum(check["status"] == "pass" for check in checks)
    return {
        "schema_version": "sigma-theory-compiler-validation-1.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "counts": {"total": len(checks), "passed": passed, "failed": len(checks) - passed},
        "interpretation": (
            "These known-answer checks validate compiler mechanics only. They do not validate any gravity candidate against nature."
        ),
        "checks": checks,
    }


def write_validation(report: dict, output_directory: str | Path) -> Path:
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "validation.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
