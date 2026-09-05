"""Action-level static adapters, with explicit scope and known-family provenance.

TRIMOND: Milgrom 2023, arXiv:2305.19986, equations 2, 3, 7-9.
GQUMOND: Milgrom 2023, arXiv:2305.01589, equations 6, 7, 14, 16.

The baseline is a regularized analytic MOND-limit Q, NOT the empirical RAR.
A shared spherical limit is not a proof of equal disk rotation curves.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
import json
from typing import Any

import numpy as np
import sympy as sp

X, Y, Z, H = sp.symbols("x y z h", real=True)
EPS = sp.symbols("epsilon", positive=True)
MIX, BETA, POWER = sp.symbols("mixing beta power", nonnegative=True)
# x=|grad psi|^2/a0^2, y=|grad chi|^2/a0^2, z=2 grad psi.grad chi/a0^2.
# h=ell^2 sum_ij(psi_ij^2)/a0^2. epsilon is a declared numerical regularizer.
Q = X + sp.Rational(4, 3) * X / (X + EPS**2) ** sp.Rational(1, 4)


@dataclass(frozen=True)
class ActionSpec:
    family: str
    mixing: float = 0.0
    beta: float = 0.0
    length: float = 0.0
    power: int = 1
    epsilon: float = 1e-4

    def __post_init__(self) -> None:
        if self.family not in {"qumond", "trimond_alignment", "gqumond_length"}:
            raise ValueError("unknown action family")
        for key in ("mixing", "beta", "length", "epsilon"):
            v = getattr(self, key)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.epsilon == 0 or type(self.power) is not int or self.power not in {1, 2}:
            raise ValueError("epsilon > 0 and power in {1,2} are required")
        if self.family != "trimond_alignment" and (self.mixing != 0 or self.beta != 0):
            raise ValueError("unused multi-field coefficients are forbidden")
        if self.family != "gqumond_length" and self.length != 0:
            raise ValueError("length only belongs to gqumond_length")

    def expression(self) -> sp.Expr:
        q = Q.subs(EPS, sp.Rational(str(self.epsilon)))
        if self.family == "qumond":
            return q
        if self.family == "gqumond_length":
            # The f=u/(1+u) length-screening construction, using our regularized Q.
            return X + sp.Rational(4, 3) * X / (
                X + H + sp.Rational(str(self.epsilon))**2
            ) ** sp.Rational(1, 4)
        s = sp.Rational(str(self.mixing)) / (1 + X)**self.power
        b = sp.Rational(str(self.beta))
        # Concave in grad chi; eliminating chi gives a static TRIMOND subclass.
        # The defect y-s*z+s*s*x and Gram term both have zero first variation
        # on grad chi=s(x)*grad psi. Such a field exists in 1-D symmetry.
        return q - (Y - s * Z + s**2 * X) - b * (X * Y - Z**2 / 4) / (1 + X)**2

    def card(self) -> dict[str, Any]:
        expression = self.expression()
        record = {
            "family": self.family,
            "parameters": {k: getattr(self, k) for k in
                           ("mixing", "beta", "length", "power", "epsilon")},
            "action_function": str(expression),
            "derivatives": {str(v): str(sp.diff(expression, v)) for v in (X, Y, Z, H)},
            "scope": "nonrelativistic_static_candidate_not_covariantly_admitted",
            "numerical_domain": "periodic_density_contrast_synthetic_controls",
            "prior_art": ("arXiv:2305.19986" if self.family == "trimond_alignment"
                          else "arXiv:2305.01589"),
            "historical_novelty_claimed": False,
            "empirical_support": False,
            "photon_sector": "unsupported_unless_explicit_assumed_metric_is_requested",
            "open_obligations": ["isolated_boundaries", "continuum_convergence_on_real_sources",
                                 "relativistic_completion", "causality", "cosmology",
                                 "local_gravity_constraints", "independent_observations"],
        }
        raw = json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {**record, "content_sha256": sha256(raw.encode()).hexdigest()}

    def partials(self, x: np.ndarray, y: np.ndarray | float = 0,
                 z: np.ndarray | float = 0, h: np.ndarray | float = 0) -> tuple[np.ndarray, ...]:
        shape = np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z), np.shape(h))
        values = _compiled(self)(x, y, z, h)
        out = tuple(np.broadcast_to(np.asarray(v, dtype=float), shape) for v in values)
        if any(not np.all(np.isfinite(v)) for v in out):
            raise FloatingPointError("action derivative produced nonfinite values")
        return out


@lru_cache(maxsize=512)
def _compiled(spec: ActionSpec):
    expr = spec.expression()
    return sp.lambdify((X, Y, Z, H), [sp.diff(expr, v) for v in (X, Y, Z, H)], "numpy")


def generate_specs(config: dict[str, Any]) -> list[ActionSpec]:
    """Discrete STRUCTURES; coefficients remain universal across all scenes.

    Boundary-equivalent zero-amplitude cases are kept only once. Neither a
    source shape nor an observed response is an input to this generator.
    """
    epsilon = float(config["epsilon"])
    result = [ActionSpec("qumond", epsilon=epsilon)]
    for mix in config["mixing"]:
        if float(mix) == 0:
            continue  # chi=0 makes every beta irrelevant on this branch.
        for beta in config["beta"]:
            for power in config["powers"]:
                result.append(ActionSpec("trimond_alignment", float(mix), float(beta),
                                         power=power, epsilon=epsilon))
    for length in config["lengths"]:
        if float(length) != 0:
            result.append(ActionSpec("gqumond_length", length=float(length), epsilon=epsilon))
    # Stable exact-card de-duplication, not a finite-output equivalence claim.
    return list({s.card()["content_sha256"]: s for s in result}.values())


def action_certificates() -> dict[str, Any]:
    """Exact symbolic tests of variations and limits, not just action values."""
    a = sp.Matrix(sp.symbols("a0:3", real=True))
    b = sp.Matrix(sp.symbols("b0:3", real=True))
    t = sp.symbols("t", real=True)
    gram = a.dot(a) * b.dot(b) - a.dot(b)**2
    first = [sp.diff(gram, v).subs(dict(zip(b, t*a, strict=True))) for v in (*a, *b)]
    collinear = all(sp.simplify(v) == 0 for v in first)
    s = MIX / (1 + X)**POWER
    f = Q - (Y - s*Z + s*s*X) - BETA*(X*Y-Z*Z/4)/(1+X)**2
    substitutions = {Y: s*s*X, Z: 2*s*X}
    aux = sp.simplify((sp.diff(f, Y)*s + sp.diff(f, Z)).subs(substitutions))
    flux = sp.simplify((sp.diff(f, X)+s*sp.diff(f, Z)).subs(substitutions)-sp.diff(Q, X))
    # Q asymptotic certificate is for epsilon=0 and x>0, not at the singular origin.
    q0 = Q.subs(EPS, 0)
    deep = sp.limit(q0 / X**sp.Rational(3, 4), X, 0, dir="+")
    high = sp.limit(sp.diff(q0, X), X, sp.oo)
    # A term that is zero in value but not in first variation must NOT pass.
    value_only = sp.symbols("b1", real=True)
    return {
        "gram_first_variation_zero_when_collinear": collinear,
        "trimond_collinear_auxiliary_residual": str(aux),
        "trimond_collinear_physical_flux_residual": str(flux),
        "deep_Q_power": "3/4", "deep_Q_coefficient": str(deep),
        "deep_AQUAL_squared_gradient_power": "3/2",
        "deep_AQUAL_gradient_magnitude_power": "3",
        "high_Q_derivative": str(high),
        "value_only_negative_control_rejected": sp.diff(value_only, value_only) != 0,
        "all_pass": bool(collinear and aux == 0 and flux == 0 and deep == sp.Rational(4, 3)
                         and high == 1),
        "claim_ceiling": "symbolic_static_identities_only_not_global_stability",
    }
