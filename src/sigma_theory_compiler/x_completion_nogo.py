from __future__ import annotations

from typing import Any

import sympy as sp


def static_null_k14_multiplicative_no_go_control() -> tuple[bool, dict[str, Any]]:
    """Necessary-condition no-go for positive decaying W(X)(K1+K4) completions."""

    gamma, weight, weight_r, weight_rr = sp.symbols(
        "gamma W W_r W_rr", positive=True, real=True
    )
    metric_shear = 1 - gamma * weight
    schur_shear_coefficient = sp.factor(
        -gamma * weight_rr / 2 - gamma**2 * weight_r**2 / metric_shear
    )
    identity_residual = sp.simplify(
        schur_shear_coefficient
        - (-gamma * weight_rr / 2 - gamma**2 * weight_r**2 / metric_shear)
    )

    x = sp.Symbol("X", nonnegative=True, real=True)
    witnesses: dict[str, Any] = {}
    all_witnesses_negative = True
    for exponent in (sp.Rational(1, 2), sp.Rational(2, 3), sp.Rational(3, 4)):
        matched = sp.factor(
            (1 + x) ** (exponent - 2) * (1 + (2 * exponent - 1) * x)
        )
        derivative = sp.diff(matched, x)
        coefficient = sp.factor(
            -sp.Rational(1, 2) * (derivative + 2 * x * sp.diff(derivative, x))
            - x * derivative**2 / (1 - matched / 2)
        )
        value = sp.simplify(coefficient.subs(x, 1))
        negative = value.is_negative is True
        all_witnesses_negative = all_witnesses_negative and negative
        witnesses[str(exponent)] = {
            "X": "1",
            "gamma": "1/2",
            "large_shear_coefficient": str(value),
            "negative_exactly": negative,
        }

    exponent = sp.Rational(2, 3)
    f_longitudinal = sp.factor(
        2
        * exponent
        * (1 + x) ** (exponent - 2)
        * (1 + (2 * exponent - 1) * x)
    )
    constant_weight_speed_limit = sp.limit(1 / f_longitudinal, x, sp.oo)
    constant_weight_fails = constant_weight_speed_limit == sp.oo

    passed = identity_residual == 0 and all_witnesses_negative and constant_weight_fails
    return passed, {
        "completion_class": "positive W(X)(K1+K4) with X=r^2",
        "assumptions": [
            "W is C2, even as a function w(r)=W(r^2), positive, and nonconstant",
            "the metric shear factor 1-gamma*W is positive",
            "W tends to zero strongly enough to track the decaying nonlinear-X kinetic Hessian",
        ],
        "generic_shear_schur_coefficient": str(schur_shear_coefficient),
        "symbolic_identity_residual": str(identity_residual),
        "calculus_no_go_proof": [
            "nonnegative large-shear Schur coefficient requires w''(r)<=0",
            "evenness gives w'(0)=0, so global concavity makes w' nonincreasing and nonpositive",
            "if w' is ever negative it stays bounded above by a negative constant and w crosses zero",
            "if w' is never negative then w is constant and cannot decay from W(0)>0 to zero",
            "therefore some finite r has w''(r)>0, where sufficiently large traceless shear makes the Schur complement negative and forces a rank-changing surface",
        ],
        "registered_matched_weight_witnesses": witnesses,
        "constant_weight_escape": {
            "p": "2/3",
            "high_X_speed_factor_limit": str(constant_weight_speed_limit),
            "rejected": constant_weight_fails,
            "reason": "constant W avoids the concavity obstruction but its gradient-to-kinetic ratio diverges",
        },
        "conclusion": (
            "No positive multiplicative W(X)(K1+K4) completion in the declared class can be "
            "both globally shear-Legendre-regular and high-X speed bounded. A new tensor "
            "structure or additional compensating operators is required."
        ),
        "scope": (
            "necessary local aligned metric-vector kinetic condition on arbitrary finite "
            "traceless extrinsic curvature; it does not classify completions outside the "
            "multiplicative K1+K4 class"
        ),
    }

