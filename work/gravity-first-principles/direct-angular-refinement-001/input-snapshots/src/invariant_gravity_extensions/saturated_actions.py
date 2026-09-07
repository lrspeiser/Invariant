"""Versioned bounded-action successors; no old card or expression is modified.

The bounded excess is an explicit structural ansatz, not a uniquely derived
physical principle. The two- and three-potential variational frameworks are
known QUMOND/TRIMOND; no historical novelty or relativistic health is claimed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from hashlib import sha256
from typing import Any

import numpy as np
import sympy as sp

from .actions import EPS, ActionSpec, Q, X, Y, Z


def saturated_q(x: sp.Expr, epsilon: sp.Expr, shape: sp.Rational) -> sp.Expr:
    """Q=x+(4/3)[S(x+eps^2)-S(eps^2)]; S(u)=[u^m/(1+u^m)]^(3/(4m))."""
    u = x + epsilon**2

    def saturation(v):
        return v**sp.Rational(3, 4) / (1 + v**shape)**(sp.Rational(3, 4) / shape)

    return x + sp.Rational(4, 3) * (saturation(u) - saturation(epsilon**2))


@dataclass(frozen=True)
class SaturatedActionSpec(ActionSpec):
    shape: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.family not in {"qumond", "trimond_alignment"}:
            raise ValueError("saturation successor supports only first-gradient QUMOND/TRIMOND")
        if self.shape not in {0.5, 1.0, 2.0}:
            raise ValueError("frozen saturation grammar has shape 0.5, 1 or 2")

    def expression(self) -> sp.Expr:
        eps = sp.Rational(str(self.epsilon))
        new_q = saturated_q(X, eps, sp.Rational(str(self.shape)))
        return super().expression() - Q.subs(EPS, eps) + new_q

    def delta_nu(self, y: np.ndarray | float) -> np.ndarray:
        """Spherical anomalous response, computed separately to avoid 1+tiny-1."""
        y = np.asarray(y, dtype=float)
        if np.any(~np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("positive finite g_N/a0 required")
        log_u = 2 * np.log(np.hypot(y, self.epsilon))
        return np.exp(-0.25 * log_u - (1 + 0.75 / self.shape) *
                      np.logaddexp(0, self.shape * log_u))

    def card(self) -> dict[str, Any]:
        record = super().card()
        record.pop("content_sha256")
        record["schema"] = "invariant-saturated-action-1"
        record["parameters"]["shape"] = self.shape
        record["kernel"] = "bounded_excess_action_v1"
        record["high_acceleration_delta_nu_power"] = 2 * (1 + self.shape)
        record["physical_motivation"] = (
            "low-gradient MOND homogeneity plus bounded high-gradient excess action; "
            "interpolation and auxiliary couplings are ansatz choices, not derived microphysics")
        record["prior_art_scope"] = (
            "known QUMOND/TRIMOND framework; this bounded-kernel ansatz has not received "
            "a comprehensive algebraic or literature novelty check")
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**record, "content_sha256": sha256(raw.encode()).hexdigest()}


def saturated_certificates(shape: float) -> dict[str, Any]:
    spec = SaturatedActionSpec("qumond", shape=shape)
    m = sp.Rational(str(shape))
    positive_x = sp.symbols("u", positive=True)
    q = saturated_q(positive_x, sp.Integer(0), m)
    delta = positive_x**(-sp.Rational(1, 4)) * (1 + positive_x**m)**(-1 - sp.Rational(3, 4) / m)
    derivative_residual = sp.simplify(sp.diff(q, positive_x) - 1 - delta)
    deep = sp.limit((q - positive_x) / positive_x**sp.Rational(3, 4), positive_x, 0)
    high = sp.limit(delta * positive_x**(1 + m), positive_x, sp.oo)
    excess_limit = sp.limit(q - positive_x, positive_x, sp.oo)
    trimond = SaturatedActionSpec("trimond_alignment", mixing=0.75, beta=2,
                                 power=2, shape=shape)
    f = trimond.expression()
    s = sp.Rational(3, 4) / (1 + X)**2
    substitutions = {Y: s*s*X, Z: 2*s*X}
    aux = sp.simplify((sp.diff(f, Y)*s + sp.diff(f, Z)).subs(substitutions))
    flux = sp.simplify((sp.diff(f, X)+s*sp.diff(f, Z)).subs(substitutions)-sp.diff(spec.expression(), X))
    return {
        "shape": shape, "derivative_residual": str(derivative_residual),
        "unregularized_deep_Q_excess_coefficient": str(deep),
        "unregularized_high_delta_nu_coefficient": str(high),
        "unregularized_high_action_excess_limit": str(excess_limit),
        "high_delta_nu_power_in_gN_over_a0": 2 * (1 + shape),
        "collinear_auxiliary_residual": str(aux), "collinear_physical_residual": str(flux),
        "all_pass": bool(derivative_residual == 0 and deep == sp.Rational(4, 3)
                         and high == 1 and excess_limit == sp.Rational(4, 3)
                         and aux == 0 and flux == 0),
        "claim_ceiling": "static_symbolic_limits_and_variations_not_global_stability",
    }


def generate_saturated_specs(grammar: dict) -> list[SaturatedActionSpec]:
    result = []
    for shape in grammar["shapes"]:
        result.append(SaturatedActionSpec("qumond", shape=shape, epsilon=grammar["epsilon"]))
        for mix in grammar["mixing"]:
            if mix == 0:
                continue
            for beta in grammar["beta"]:
                for power in grammar["powers"]:
                    result.append(SaturatedActionSpec("trimond_alignment", mixing=mix, beta=beta,
                                                      power=power, shape=shape, epsilon=grammar["epsilon"]))
    return list({s.card()["content_sha256"]: s for s in result}.values())
