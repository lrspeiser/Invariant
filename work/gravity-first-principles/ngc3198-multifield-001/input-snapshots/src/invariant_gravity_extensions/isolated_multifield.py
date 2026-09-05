"""Isolated bounded TRIMOND auxiliary fields on a supplied axisymmetric source.

Coordinates may have any consistent length unit. p=grad(psi)/a0 and the solved
q=grad(chi)/a0 are dimensionless. The physical extra acceleration is a0 times
the gradient of the resulting flux potential, with the force's minus sign.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .external_multifield import FluxPoissonSolver, auxiliary_anisotropy, physical_auxiliary_flux
from .isolated_axisymmetric import MultipolePotential


def gradient_on_flux_grid(potential, solver):
    """Evaluate the same spectral potential on different angular quadrature nodes.

    The density may have been integrated on a plane-focused grid. Changing the
    angles used to evaluate its existing coefficients is exact evaluation of
    that potential, not density rebinning or interpolation between mass models.
    """
    for key in ("r_min", "r_max", "radial_nodes", "l_max"):
        if getattr(potential.grid, key) != getattr(solver.grid, key):
            raise ValueError("matching radial grid and polynomial degree required")
    equivalent = MultipolePotential(solver.grid, potential.spline, potential.source_coefficients)
    return solver.gradient(equivalent)


@dataclass
class IsolatedAuxiliarySolution:
    potential: MultipolePotential
    physical_flux_potential: MultipolePotential
    q: np.ndarray
    history: list[dict]
    relative_equation_residual: float
    maximum_equation_residual: float


def solve_isolated_auxiliary(solver: FluxPoissonSolver, p, beta, power, *,
                             tolerance=1e-9, max_iterations=100):
    """Solve the unit-mixing branch; finite real mixing scales q and flux exactly.

    Background q is zero. A final independent application of the fixed-point
    map checks the returned iterate, separately from its update history.
    """
    p = solver.validate_flux(p)
    if (not np.isfinite(beta) or not 0 <= beta <= 2 or type(power) is not int or power not in (1, 2) or
            not np.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 2):
        raise ValueError("beta in [0,2], power 1 or 2, positive tolerance and at least two iterations required")
    x = np.sum(p*p, axis=0)
    s = 1/(1+x)**power
    rhs = s*p
    q = np.zeros_like(p)
    history = []
    for iteration in range(1, max_iterations+1):
        potential = solver.solve(rhs-auxiliary_anisotropy(p, q, beta))
        new_q = solver.gradient(potential)
        relative = solver.energy_norm(new_q-q)/max(solver.energy_norm(new_q), 1e-300)
        maximum = float(np.max(np.linalg.norm(new_q-q, axis=0)))
        history.append({"iteration": iteration, "relative_update": relative, "maximum_update": maximum})
        q = new_q
        if relative < tolerance and maximum < tolerance:
            check = solver.gradient(solver.solve(rhs-auxiliary_anisotropy(p, q, beta)))
            residual = solver.energy_norm(check-q)/max(solver.energy_norm(q), 1e-300)
            max_residual = float(np.max(np.linalg.norm(check-q, axis=0)))
            if residual < tolerance and max_residual < tolerance:
                flux = physical_auxiliary_flux(p, q, 1, beta, power)
                return IsolatedAuxiliarySolution(potential, solver.solve(flux), q,
                                                 history, residual, max_residual)
    raise RuntimeError(f"isolated auxiliary convergence unresolved: {history[-1]}")


def normalized_newtonian_gradient(fields, radius, mu, a0):
    """Cylindrical to radial/theta gradient, in units of a0."""
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError("positive finite acceleration scale required")
    gradient = np.asarray(fields["gradient"])
    sine = np.sqrt(1-mu*mu)
    shape = np.broadcast_shapes(np.shape(radius[:, None]), np.shape(mu))
    if gradient.shape != (2,)+shape:
        raise ValueError("gradient must cover the radial/angular nodes")
    return np.array([gradient[0]*sine+gradient[1]*mu,
                     gradient[0]*mu-gradient[1]*sine])/a0
