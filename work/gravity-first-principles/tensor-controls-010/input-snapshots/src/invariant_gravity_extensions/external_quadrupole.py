"""QUMOND point-source quadrupole in a constant external field.

Equation 12 of Hees et al. arXiv:1510.01369 and an independent induced-source
Green-function Hessian integral. These equations do not cover TRIMOND.
"""
from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.optimize import brentq

from .saturated_actions import SaturatedActionSpec

Response = Callable[[np.ndarray], np.ndarray]


def saturated_nu_derivative(spec: SaturatedActionSpec, y: np.ndarray) -> np.ndarray:
    """Derivative d nu / d y, evaluated without forming nu-1 by subtraction."""
    y = np.asarray(y, dtype=float)
    delta = spec.delta_nu(y)
    norm = np.hypot(y, spec.epsilon)
    log_u = 2 * np.log(norm)
    sigmoid = np.exp(-np.logaddexp(0, -spec.shape * log_u))
    return -delta * (y / norm) / norm * (0.5 + (2 * spec.shape + 1.5) * sigmoid)


def reference_nu_delta(y: np.ndarray, alpha: int) -> np.ndarray:
    """Known nu_alpha family, Hees et al. eq. 5a; reference control only."""
    y = np.asarray(y, dtype=float)
    if alpha not in {2, 4, 8} or np.any(y <= 0) or np.any(~np.isfinite(y)):
        raise ValueError("reference requires alpha=2,4,8 and positive finite y")
    a = y**(-alpha)
    root = np.sqrt(1 + 4*a)
    return np.expm1(np.log1p(2*a/(root+1))/alpha)


def reference_nu_derivative(y: np.ndarray, alpha: int) -> np.ndarray:
    delta = reference_nu_delta(y, alpha)
    a = y**(-alpha)
    root = np.sqrt(1 + 4*a)
    return -(1+delta)*a/(y*root*((1+root)/2))


def newtonian_external_ratio(eta: float, delta_nu: Response) -> float:
    """Solve eta_N*nu(eta_N)=eta under the declared standard boundary mapping.

    This assumes the positive monotone force branch of the supplied response;
    it does not reconstruct the actual Galactic mass distribution.
    """
    if not np.isfinite(eta) or eta < 0:
        raise ValueError("physical external field magnitude must be nonnegative")
    if eta == 0:
        return 0.0

    def residual(x):
        d = float(delta_nu(np.asarray(x)))
        if not np.isfinite(d) or d < 0:
            raise ValueError("nonnegative finite response required")
        return x*(1+d)-eta

    return float(brentq(residual, eta*1e-12, eta, xtol=1e-14, rtol=1e-13))


@lru_cache(maxsize=8)
def _nodes(n: int):
    if type(n) is not int or not 32 <= n <= 2048:
        raise ValueError("use 32 to 2048 Gaussian nodes")
    return leggauss(n)


def quadrupole_integrals(
    eta_newtonian: float, delta_nu: Response, nu_derivative: Response,
    *, nodes: int = 256,
) -> dict[str, float]:
    """Return two independently derived representations of dimensionless q.

    v=R_M/r. Split at the saddle v=sqrt(eta_N); map the second interval to
    infinity, without radial truncation. An angle-independent subtraction in
    the first integrand cancels analytically and reduces roundoff in the tails.
    The second uses the induced Poisson source and a Green-function Hessian.
    Refinement is empirical numerical evidence, not a rigorous error bound.
    """
    if not np.isfinite(eta_newtonian) or eta_newtonian < 0:
        raise ValueError("Newtonian external ratio must be nonnegative")
    abscissa, weights = _nodes(nodes)
    if eta_newtonian == 0:
        return {"q_milgrom": 0.0, "q_source_hessian": 0.0, "absolute_agreement": 0.0}
    eta = eta_newtonian
    xi = abscissa[None, :]
    t, tw = (abscissa+1)/2, weights/2
    saddle = np.sqrt(eta)
    intervals = [(saddle*t, saddle*tw), (saddle+t/(1-t), tw/(1-t)**2)]
    values = np.zeros(2)
    for radial, radial_weights in intervals:
        v = radial[:, None]
        # This spelling avoids cancellation exactly at the Newtonian saddle.
        w2 = (v*v-eta)**2+2*eta*v*v*(1+xi)
        w = np.sqrt(w2)
        d = delta_nu(w)-delta_nu(np.sqrt(eta*eta+v**4))
        bracket = eta*(3*xi-5*xi**3)+v*v*(1-3*xi**2)
        first = 1.5*d*bracket
        # For psi=-1/r-eta*z, p.H(psi).p = r^-3*(|p|^2-3(p.n)^2).
        # xi=-cos(theta), so |p|=w and p.n=v^2+eta*xi.
        second = (-0.5*v*v*nu_derivative(w)/w *
                  (w2-3*(v*v+eta*xi)**2)*(3*xi**2-1))
        if not np.all(np.isfinite(first)) or not np.all(np.isfinite(second)):
            raise FloatingPointError("nonfinite quadrupole integrand")
        values += [radial_weights@(first@weights), radial_weights@(second@weights)]
    return {"q_milgrom": float(values[0]), "q_source_hessian": float(values[1]),
            "absolute_agreement": float(abs(values[0]-values[1]))}


def q_to_Q2(q: float, a0: float, gm: float) -> float:
    """Phi_an=-Q2/2*r_i*r_j*(e_i*e_j-delta_ij/3), in SI units."""
    if not all(np.isfinite(x) for x in (q, a0, gm)) or a0 <= 0 or gm <= 0:
        raise ValueError("finite q and positive finite a0, GM required")
    return float(-1.5*q*a0**1.5/np.sqrt(gm))


def scalar_quadrupole(
    spec: SaturatedActionSpec, physical_external_m_s2: float, a0_m_s2: float,
    gm_m3_s2: float, *, nodes: int = 256,
) -> dict:
    if not isinstance(spec, SaturatedActionSpec) or spec.family != "qumond":
        raise NotImplementedError("TRIMOND and higher-derivative external fields need their own solver")
    if not np.isfinite(a0_m_s2) or a0_m_s2 <= 0:
        raise ValueError("positive finite a0 required")
    eta = physical_external_m_s2/a0_m_s2
    eta_n = newtonian_external_ratio(eta, spec.delta_nu)
    values = quadrupole_integrals(eta_n, spec.delta_nu,
                                 lambda y: saturated_nu_derivative(spec, y), nodes=nodes)
    return {**values, "eta_physical": eta, "eta_newtonian": eta_n,
            "Q2_s_minus2": q_to_Q2(values["q_milgrom"], a0_m_s2, gm_m3_s2),
            "scope": "QUMOND_point_source_constant_external_field_only",
            "boundary_mapping": "eta_N*nu(eta_N)=eta; not a fitted Galactic model",
            "full_solar_system_pass": False}


def acceleration_tensor(Q2: float, direction: np.ndarray) -> np.ndarray:
    """Anomalous acceleration A*r; a symmetric traceless tensor."""
    axis = np.asarray(direction, dtype=float)
    if axis.shape != (3,) or not np.all(np.isfinite(axis)) or np.linalg.norm(axis) == 0 or not np.isfinite(Q2):
        raise ValueError("finite Q2 and nonzero finite three-vector required")
    axis = axis/np.linalg.norm(axis)
    return Q2*(np.outer(axis, axis)-np.eye(3)/3)
