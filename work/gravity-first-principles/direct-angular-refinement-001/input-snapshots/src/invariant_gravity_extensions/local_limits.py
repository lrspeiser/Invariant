"""Isolated spherical orbital diagnostics, not a full Solar System likelihood.

The tested monopole is derived from Q_x, with zero anomalous integration mass
and the regular collinear TRIMOND auxiliary branch. Galactic external fields,
planet interactions, photon propagation and a covariant completion are absent.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp

from .actions import ActionSpec

DeltaNu = Callable[[np.ndarray], np.ndarray]
RAD_TO_MAS = 180 / np.pi * 3600 * 1000


@dataclass(frozen=True)
class Orbit:
    semimajor_m: float
    eccentricity: float
    gm_m3_s2: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.semimajor_m) or self.semimajor_m <= 0:
            raise ValueError("positive finite semimajor axis required")
        if not np.isfinite(self.gm_m3_s2) or self.gm_m3_s2 <= 0:
            raise ValueError("positive finite GM required")
        if not np.isfinite(self.eccentricity) or not 0 < self.eccentricity < 1:
            raise ValueError("an eccentric bound orbit requires 0 < e < 1")

    @property
    def period_s(self) -> float:
        return float(2 * np.pi * np.sqrt(self.semimajor_m**3 / self.gm_m3_s2))


def _positive(values: np.ndarray | float, name: str) -> np.ndarray:
    out = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"{name} must be finite and positive")
    return out


def baseline_delta_nu(y: np.ndarray | float, epsilon: float = 1e-4) -> np.ndarray:
    """Q_x(y^2)-1 without subtracting nearly equal numbers.

    Q=x+(4/3)*x/(x+epsilon^2)^(1/4).
    Delta nu=(x+4*epsilon^2/3)/(x+epsilon^2)^(5/4).
    Hypotenuse form avoids both squaring huge y and overflowing epsilon/y.
    """
    y = _positive(y, "g_N/a0")
    _positive(epsilon, "epsilon")
    norm = np.hypot(y, epsilon)
    return (1 + (epsilon / norm)**2 / 3) / np.sqrt(norm)


def monopole_delta_nu(spec: ActionSpec, y: np.ndarray | float) -> np.ndarray:
    if spec.family not in {"qumond", "trimond_alignment"}:
        raise NotImplementedError(
            "length-sensitive action needs its spherical higher-derivative solution "
            "and a dimensionful length; Q_x alone is not its force")
    return baseline_delta_nu(y, spec.epsilon)


def power_tail(y: np.ndarray | float, exponent: float, coefficient: float = 1) -> np.ndarray:
    """Asymptotic diagnostic only; this does not define a complete MOND law."""
    y = _positive(y, "g_N/a0")
    _positive(exponent, "exponent")
    _positive(coefficient, "coefficient")
    return coefficient * y**(-exponent)


def perihelion_first_order(
    orbit: Orbit, a0: float, delta_nu: DeltaNu, *, nodes: int = 64,
) -> float:
    """Radians per orbit from the radial Gauss equation, inward anomaly positive.

    Delta omega = 1/e * integral_0^(2pi) delta_nu(g_N/a0)*cos(f) df.
    Pair near/far anomalies on [0,pi/2] to cancel a fitted constant GM shift
    analytically, without interpreting such a shift as perihelion precession.
    This is first order in the perturbation, with osculating Kepler elements.
    """
    _positive(a0, "a0")
    if type(nodes) is not int or nodes < 16:
        raise ValueError("at least 16 quadrature nodes required")
    abscissa, weights = leggauss(nodes)
    angle = (abscissa + 1) * np.pi / 4
    cos = np.cos(angle)
    e = orbit.eccentricity
    p = orbit.semimajor_m * (1 - e**2)
    near = p / (1 + e * cos)
    far = p / (1 - e * cos)
    yn = orbit.gm_m3_s2 / (a0 * near**2)
    yf = orbit.gm_m3_s2 / (a0 * far**2)
    dn, df = np.asarray(delta_nu(yn)), np.asarray(delta_nu(yf))
    if dn.shape != yn.shape or df.shape != yf.shape or not np.all(np.isfinite([dn, df])):
        raise ValueError("delta_nu must return finite arrays with matching shapes")
    return float(2 / e * np.pi / 4 * np.dot(weights, cos * (dn - df)))


def logarithmic_precession(orbit: Orbit, a0: float) -> float:
    """Closed first-order answer for inward delta g=sqrt(GM*a0)/r."""
    _positive(a0, "a0")
    s = np.sqrt(1 - orbit.eccentricity**2)
    return float(-2 * np.pi * orbit.semimajor_m *
                 np.sqrt(a0 / orbit.gm_m3_s2) * s / (1 + s))


def binet_precession(orbit: Orbit, a0: float, *, rtol: float = 2e-12) -> float:
    """Independent nonperturbative orbit integration for the unregularized log tail.

    Dimensionless u=a/r, J=1-e^2, with initial Kepler perihelion and angular
    momentum. Integrate u''+u=(1+delta/u)/J until the next perihelion. The initial
    elements are osculating, not the perturbed orbit's turning-point elements.
    """
    _positive(a0, "a0")
    _positive(rtol, "rtol")
    delta = orbit.semimajor_m * np.sqrt(a0 / orbit.gm_m3_s2)
    j = 1 - orbit.eccentricity**2

    def rhs(_angle, state):
        u, derivative = state
        if u <= 0:
            raise RuntimeError("orbit left the positive-radius domain")
        return [derivative, (1 + delta / u) / j - u]

    def perihelion(_angle, state):
        return state[1]

    perihelion.direction = -1
    solution = solve_ivp(rhs, (0, 8 * np.pi),
                         [1 / (1 - orbit.eccentricity), 0], method="DOP853",
                         events=perihelion, rtol=rtol, atol=rtol * 0.01,
                         max_step=0.1)
    events = solution.t_events[0]
    events = events[events > 1e-6]
    if not solution.success or len(events) < 1:
        raise RuntimeError("independent orbit integration did not reach next perihelion")
    return float(events[0] - 2 * np.pi)


def mas_per_century(radians_per_orbit: float, orbit: Orbit, century_s: float) -> float:
    _positive(century_s, "century_s")
    return float(radians_per_orbit * RAD_TO_MAS * century_s / orbit.period_s)
