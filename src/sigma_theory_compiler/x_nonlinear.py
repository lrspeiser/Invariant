from __future__ import annotations

from typing import Any

import sympy as sp


def nonlinear_aether_acceleration_convexity_control() -> tuple[bool, dict[str, Any]]:
    """Global velocity-Hessian and high-field control for F_p(X)=(1+X)^p-1."""

    x, exponent = sp.symbols("X p", nonnegative=True, real=True)
    transverse = 2 * exponent * (1 + x) ** (exponent - 1)
    longitudinal = sp.factor(
        2 * exponent * (1 + x) ** (exponent - 2) * (1 + (2 * exponent - 1) * x)
    )
    field = (1 + x) ** exponent - 1
    transverse_residual = sp.simplify(transverse - 2 * sp.diff(field, x))
    longitudinal_residual = sp.simplify(
        longitudinal - (2 * sp.diff(field, x) + 4 * x * sp.diff(field, x, 2))
    )
    high_field_ratio = sp.limit(field.subs(exponent, sp.Rational(2, 3)) / x, x, sp.oo)
    negative_factor = sp.factor(
        (1 + (2 * exponent - 1) * x).subs(
            {exponent: sp.Rational(1, 3), x: 4}
        )
    )
    negative_rejected = bool(negative_factor < 0)
    endpoint_factors = {
        "p=1/2": str(sp.factor((1 + (2 * exponent - 1) * x).subs(exponent, sp.Rational(1, 2)))),
        "p=1": str(sp.factor((1 + (2 * exponent - 1) * x).subs(exponent, 1))),
    }
    passed = (
        transverse_residual == 0
        and longitudinal_residual == 0
        and high_field_ratio == 0
        and negative_rejected
        and endpoint_factors == {"p=1/2": "1", "p=1": "X + 1"}
    )
    return passed, {
        "family": "F_p(X)=(1+X)^p-1",
        "declared_domain": "X>=0 and 1/2<=p<1",
        "transverse_velocity_hessian_eigenvalue_without_scale": str(transverse),
        "longitudinal_velocity_hessian_eigenvalue_without_scale": str(longitudinal),
        "transverse_derivation_residual": str(transverse_residual),
        "longitudinal_derivation_residual": str(longitudinal_residual),
        "positivity_proof": (
            "2p>0, (1+X)^(p-2)>0, and 1+(2p-1)X>=1 on the declared domain"
        ),
        "high_field_F_over_X_limit_at_p_2_3": str(high_field_ratio),
        "endpoint_longitudinal_factors": endpoint_factors,
        "negative_control": {
            "p": "1/3",
            "X": "4",
            "longitudinal_bracket": str(negative_factor),
            "rejected": negative_rejected,
        },
        "claim_limit": (
            "exact local acceleration-velocity convexity and high-field scaling only; full "
            "metric-Aether ADM/Dirac closure and nonlinear energy remain separate"
        ),
    }
