"""Axisymmetric flux Green solver and bounded TRIMOND external auxiliary field.

Dimensionless GM=a0=1. Only a point source plus an idealized constant,
collinear background is supported. This is not a relativistic or Galactic
source solution. Finite shells zero-extend the flux; surface effects must be
checked when approximating an infinite-domain problem.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import legder, legval, legvander
from scipy.integrate import simpson
from scipy.interpolate import CubicHermiteSpline, CubicSpline

from .isolated_axisymmetric import MultipoleGrid, MultipolePotential, _decaying_integral


class FluxPoissonSolver:
    """Solve lap(u)=div(J) using integrated radial/tangential flux moments."""

    def __init__(self, grid: MultipoleGrid):
        if grid.plane_scale is not None:
            raise NotImplementedError("flux solver currently requires ordinary Gaussian angles")
        self.grid = grid
        self.radius, self.mu, self.weights = grid.nodes()
        self.sine = np.sqrt(1-self.mu**2)
        self.t = np.log(self.radius)
        self.orders = np.arange(grid.l_max+1)
        self.P = legvander(self.mu[0], grid.l_max)
        self.dP = np.array([legval(self.mu[0], legder(row)) for row in np.eye(grid.l_max+1)]).T
        factors = (2*self.orders+1)/2
        self.radial_projection = self.weights.T*self.P*factors
        self.angular_projection = self.weights.T*self.sine.T*self.dP*factors

    @property
    def shape(self):
        return (2, self.grid.radial_nodes, self.grid.angular_nodes)

    def validate_flux(self, flux):
        f = np.asarray(flux, dtype=float)
        if f.shape != self.shape or not np.all(np.isfinite(f)):
            raise ValueError("finite radial/theta flux on the full native grid required")
        return f

    def solve(self, flux):
        f = self.validate_flux(flux)
        radial = f[0]@self.radial_projection
        angular = f[1]@self.angular_projection
        r = self.radius[:, None]
        orders = self.orders[None, :]
        inner = _decaying_integral(self.t, r*(orders*radial-angular), self.orders+1)
        outer = _decaying_integral(-self.t[::-1],
                                  (r*((orders+1)*radial+angular))[::-1], self.orders)[::-1]
        potential = (inner-outer)/(2*orders+1)
        derivative = r*radial-((orders+1)*inner+self.orders*outer)/(2*orders+1)
        source = (2*radial+CubicSpline(self.t, radial)(self.t, 1)+angular)/r
        spline = CubicHermiteSpline(self.t, potential, derivative, axis=0, extrapolate=False)
        return MultipolePotential(self.grid, spline, source)

    def gradient(self, solution: MultipolePotential):
        if solution.grid != self.grid:
            raise ValueError("solution must use the identical native grid")
        r = self.radius[:, None]
        return np.array([solution.spline(self.t, 1)@self.P.T/r,
                         -(solution.spline(self.t)@self.dP.T)*self.sine/r])

    def quadrupole(self, flux):
        """Interior Q2 from the full flux and finite-shell surface correction.

        Phi_an=-Q2*r^2*P2/3. bulk is the zero-extended-flux result; volume
        additionally removes its artificial boundary sheets, so corresponds
        to div(J) restricted to the shell. Both require tail/inner convergence
        before being interpreted as an infinite-domain physical quadrupole.
        """
        f = self.validate_flux(flux)
        p2 = (3*self.mu**2-1)/2
        a = np.sum(f[0]*p2*self.weights, axis=1)
        b = np.sum(f[1]*self.mu*self.sine*self.weights, axis=1)
        bulk = float(4.5*simpson((a+b)/self.radius, x=self.t))
        surface = float(1.5*(a[-1]/self.radius[-1]-a[0]/self.radius[0]))
        return {"Q2_bulk": bulk, "Q2_surface": surface, "Q2_volume": bulk+surface}

    def energy_norm(self, vector):
        f = self.validate_flux(vector)
        return float(np.sqrt(simpson(self.radius**3*np.sum(np.sum(f*f, axis=0)*self.weights, axis=1), x=self.t)))


def point_external_gradient(solver: FluxPoissonSolver, eta_newtonian: float):
    if not np.isfinite(eta_newtonian) or eta_newtonian < 0:
        raise ValueError("finite nonnegative Newtonian external field required")
    radial = 1/solver.radius[:, None]**2-eta_newtonian*solver.mu
    theta = np.broadcast_to(eta_newtonian*solver.sine, radial.shape)
    return np.array([radial, theta])


def _couplings(p, mixing, beta, power):
    # Signed mixing is permitted here as an analytic symmetry control. The
    # frozen candidate grammar itself still contains only nonnegative mixing.
    if (not np.isfinite(mixing) or not np.isfinite(beta) or not 0 <= beta <= 2 or
            type(power) is not int or power not in (1, 2)):
        raise ValueError("finite mixing, beta in [0,2], power 1 or 2 required")
    x = np.sum(p*p, axis=0)
    s = mixing/(1+x)**power
    return x, s, -power*s/(1+x), beta/(1+x)**2


def auxiliary_anisotropy(p, q, beta):
    x = np.sum(p*p, axis=0)
    cross = p[0]*q[1]-p[1]*q[0]
    return beta/(1+x)**2*cross*np.array([-p[1], p[0]])


def physical_auxiliary_flux(p, q, mixing, beta, power):
    """Half the p-variation of F-Q at fixed q; includes all reaction terms."""
    if np.shape(p) != np.shape(q) or np.ndim(p) != 3 or np.shape(p)[0] != 2:
        raise ValueError("matching radial/theta arrays required")
    x, s, ds, w = _couplings(p, mixing, beta, power)
    defect = q-s*p
    cross = p[0]*q[1]-p[1]*q[0]
    result = (s*defect+2*ds*np.sum(p*defect, axis=0)*p+
              2*w/(1+x)*cross**2*p-w*cross*np.array([q[1], -q[0]]))
    if not np.all(np.isfinite(result)):
        raise FloatingPointError("nonfinite physical auxiliary flux")
    return result


@dataclass
class ExternalAuxiliarySolution:
    solver: FluxPoissonSolver
    potential: MultipolePotential
    p: np.ndarray
    q: np.ndarray
    background_q: np.ndarray
    iterations: int
    relative_update_energy: float
    max_absolute_update: float
    history: list[dict]


def solve_external_auxiliary(solver, eta_newtonian, mixing, beta, power,
                             *, tolerance=1e-9, max_iterations=100):
    """Poisson-preconditioned contraction for div(A q)=div(s p).

    A=I+w(x I-p p^T); eigenvalues in [1,1+beta/4]. The returned iterate
    has an explicitly checked fixed-point residual. Elliptic static solvability
    does not establish dynamical health or a relativistic completion.
    """
    if not np.isfinite(tolerance) or tolerance <= 0 or type(max_iterations) is not int or max_iterations < 2:
        raise ValueError("positive tolerance and at least two iterations required")
    p = point_external_gradient(solver, eta_newtonian)
    _, s, _, _ = _couplings(p, mixing, beta, power)
    external_s = mixing/(1+eta_newtonian**2)**power
    background = np.broadcast_to(np.array([-eta_newtonian*solver.mu, eta_newtonian*solver.sine]), p.shape)*external_s
    base_flux = s*p-background
    q = background.copy()
    history = []
    for iteration in range(1, max_iterations+1):
        potential = solver.solve(base_flux-auxiliary_anisotropy(p, q, beta))
        next_q = background+solver.gradient(potential)
        difference = next_q-q
        relative = solver.energy_norm(difference)/max(solver.energy_norm(next_q-background), 1e-300)
        absolute = float(np.max(np.linalg.norm(difference, axis=0)))
        history.append({"iteration": iteration, "relative_update_energy": relative, "max_absolute_update": absolute})
        q = next_q
        if relative < tolerance and absolute < tolerance*max(1, abs(mixing)):
            return ExternalAuxiliarySolution(solver, potential, p, q, background,
                                             iteration, relative, absolute, history)
    raise RuntimeError(f"auxiliary fixed-point convergence unresolved: {history[-1]}")


def beta_zero_source(eta_newtonian, mixing, power, R, z):
    """Analytic div(s grad psi) away from the point; s at the point is zero."""
    r = np.hypot(R, z)
    mu = z/r
    pr = 1/r**2-eta_newtonian*mu
    pt = eta_newtonian*np.sqrt(np.maximum(0, 1-mu**2))
    x = pr**2+pt**2
    ds = -power*mixing/(1+x)**(power+1)
    return 2*ds*(x-3*pr**2)/r**3
