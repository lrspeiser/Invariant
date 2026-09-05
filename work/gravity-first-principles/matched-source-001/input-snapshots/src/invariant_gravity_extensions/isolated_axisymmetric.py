"""Isolated scalar fields for named, analytic axisymmetric mass components.

The Green-function multipole boundary is nonperiodic. Finite radial truncation
and angular resolution must be checked for each source. This is not a general
3-D source adapter, TRIMOND solver, equilibrium model or photon prescription.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import legder, leggauss, legval, legvander
from scipy.interpolate import CubicHermiteSpline, CubicSpline
from scipy.special import gammainc

from .saturated_actions import SaturatedActionSpec


@dataclass(frozen=True)
class MassComponent:
    """Miyamoto-Nagai potential; a=0 is Plummer. All units must be consistent.

    gm is G times mass, not mass or a fitted acceleration. Components share one
    axis, with optional centers along it. Nonnegative masses and finite positive
    thickness give smooth, nonnegative densities. No razor-thin sheet adapter.
    """
    name: str
    gm: float
    a: float
    b: float
    z_center: float = 0.0

    def __post_init__(self):
        if (not isinstance(self.name, str) or not self.name or
                not all(np.isfinite(v) for v in (self.gm, self.a, self.b, self.z_center)) or
                self.gm <= 0 or self.a < 0 or self.b <= 0):
            raise ValueError("named positive mass and thickness, nonnegative radial scale required")

    def fields(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        if np.any(R < 0) or not np.all(np.isfinite(R)) or not np.all(np.isfinite(z)):
            raise ValueError("finite cylindrical coordinates with R>=0 required")
        z = z-self.z_center
        B = np.hypot(z, self.b)
        A = self.a+B
        D = np.hypot(R, A)
        inv3 = self.gm/D**3
        t = z/B
        gradient = np.array([inv3*R, inv3*A*t])
        hessian = np.array([[inv3*(1-3*(R/D)**2), -3*inv3*R*A*t/D**2],
                            [-3*inv3*R*A*t/D**2,
                             inv3*(t*t+A*self.b**2/B**3-3*(A*t/D)**2)]])
        laplacian = self.gm*self.b**2*(self.a*R*R+(self.a+3*B)*A*A)/(B**3*D**5)
        return {"potential": -self.gm/D, "gradient": gradient,
                "hessian": hessian, "laplacian": laplacian}


def total_newtonian(components: tuple[MassComponent, ...], R, z):
    if not components or len({c.name for c in components}) != len(components):
        raise ValueError("nonempty uniquely named components required")
    values = [c.fields(R, z) for c in components]
    return {key: np.sum([v[key] for v in values], axis=0) for key in values[0]}


def anomalous_source(components, spec, a0, R, z):
    """div[(nu-1) grad psi], after summing all Newtonian source fields."""
    if not isinstance(spec, SaturatedActionSpec) or spec.family != "qumond":
        raise NotImplementedError("only saturated scalar QUMOND has an isolated adapter")
    if not np.isfinite(a0) or a0 <= 0:
        raise ValueError("positive finite a0 required")
    fields = total_newtonian(components, R, z)
    p = fields["gradient"]
    y = np.sqrt(np.sum(p*p, axis=0))/a0
    delta = spec.delta_nu(np.maximum(y, np.finfo(float).tiny))
    norm = np.hypot(y, spec.epsilon)
    sigmoid = np.exp(-np.logaddexp(0, -2*spec.shape*np.log(norm)))
    # nu'(y)/(a0^2*y), extended continuously to a zero-gradient saddle.
    coefficient = -delta/(a0*norm)**2*(.5+(2*spec.shape+1.5)*sigmoid)
    directional = np.einsum("i...,ij...,j...->...", p, fields["hessian"], p)
    return delta*fields["laplacian"]+coefficient*directional


@dataclass(frozen=True)
class MultipoleGrid:
    r_min: float
    r_max: float
    radial_nodes: int = 1025
    angular_nodes: int = 128
    l_max: int = 32
    plane_scale: float | None = None

    def __post_init__(self):
        if (not np.isfinite(self.r_min) or not np.isfinite(self.r_max) or
                not 0 < self.r_min < self.r_max):
            raise ValueError("positive ordered radial bounds required")
        if (type(self.radial_nodes) is not int or self.radial_nodes < 33 or
                type(self.angular_nodes) is not int or self.angular_nodes < 8 or
                type(self.l_max) is not int or not 0 <= self.l_max < self.angular_nodes):
            raise ValueError("invalid radial/angular resolution or multipole order")
        if self.plane_scale is not None and (not np.isfinite(self.plane_scale) or self.plane_scale <= 0):
            raise ValueError("positive finite angular plane sampling scale required")

    def nodes(self):
        radius = np.geomspace(self.r_min, self.r_max, self.radial_nodes)
        mu, weights = leggauss(self.angular_nodes)
        if self.plane_scale is None:
            return radius, mu[None, :], weights[None, :]
        k = np.arcsinh(radius[:, None]/self.plane_scale)
        # mu=sinh(k*t)/sinh(k) resolves a fixed physical thickness near z=0.
        mapped = np.sinh(k*mu[None, :])/np.sinh(k)
        jacobian = k*np.cosh(k*mu[None, :])/np.sinh(k)
        return radius, mapped, weights[None, :]*jacobian


@dataclass
class MultipolePotential:
    grid: MultipoleGrid
    spline: CubicHermiteSpline
    source_coefficients: np.ndarray

    def evaluate(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, dtype=float), np.asarray(z, dtype=float))
        radius = np.hypot(R, z)
        if (np.any(R < 0) or not np.all(np.isfinite(radius)) or
                np.any(radius < self.grid.r_min) or np.any(radius > self.grid.r_max)):
            raise ValueError("evaluation must be inside declared radial domain, R>=0")
        mu, sintheta = z/radius, R/radius
        coefficients = self.spline(np.log(radius))
        radial_derivatives = self.spline(np.log(radius), 1)
        polynomials = legvander(mu, self.grid.l_max)
        angular_derivatives = np.array([legval(mu, legder(np.eye(self.grid.l_max+1)[l]))
                                        for l in range(self.grid.l_max+1)])
        angular_derivatives = np.moveaxis(angular_derivatives, 0, -1)
        potential = np.sum(coefficients*polynomials, axis=-1)
        ar = -np.sum(radial_derivatives*polynomials, axis=-1)/radius
        at = sintheta*np.sum(coefficients*angular_derivatives, axis=-1)/radius
        return {"potential": potential,
                "acceleration": np.array([ar*sintheta+at*mu, ar*mu-at*sintheta])}


def _decaying_integral(t, f, decay):
    """Integrate f(u)*exp[-decay*(t-u)] using exact cubic segment moments.

    This avoids huge r^l factors and large-exponent quadrature error. f is
    interpolated in log radius; refinement still controls interpolation error.
    """
    spline = CubicSpline(t, f, axis=0)
    h = np.diff(t)[:, None]
    a, b, c, _ = spline.c
    endpoint_coefficients = [f[1:], -(3*a*h*h+2*b*h+c), 3*a*h+b, -a]
    safe_decay = np.where(decay == 0, 1, decay)[None, :]
    increment = np.zeros_like(f[1:])
    for j, factorial in enumerate((1, 1, 2, 6)):
        moment = factorial*gammainc(j+1, decay[None, :]*h)/safe_decay**(j+1)
        moment = np.where(decay[None, :] == 0, h**(j+1)/(j+1), moment)
        increment += endpoint_coefficients[j]*moment
    factor = np.exp(-h*decay[None, :])
    integral = np.zeros_like(f)
    for i in range(len(t)-1):
        integral[i+1] = factor[i]*integral[i]+increment[i]
    return integral


def solve_poisson(grid: MultipoleGrid, source) -> MultipolePotential:
    """Solve lap(Phi)=source with the isolated Green kernel on a finite shell.

    Source is a callable of cylindrical R,z. Material outside the radial bounds
    is omitted explicitly. Spherical exterior shells change only the potential
    zero inside. Nonspherical omitted tails require boundary-refinement checks.
    No homogeneous density subtraction, images or periodic zero mode occur.
    """
    radius, mu, weights = grid.nodes()
    R = radius[:, None]*np.sqrt(1-mu*mu)
    z = radius[:, None]*mu
    values = np.asarray(source(R, z), dtype=float)
    if values.shape != R.shape or not np.all(np.isfinite(values)):
        raise ValueError("finite source array on the requested grid required")
    orders = np.arange(grid.l_max+1)
    # Recurrence avoids allocating an nr*nangle*lmax polynomial tensor for
    # radius-dependent angular mapping, retaining arbitrary source symmetry.
    coefficients = np.empty((grid.radial_nodes, grid.l_max+1))
    previous, polynomial = np.zeros_like(mu), np.ones_like(mu)
    for order in orders:
        coefficients[:, order] = np.sum(values*weights*polynomial, axis=1)*(2*order+1)/2
        previous, polynomial = polynomial, ((2*order+1)*mu*polynomial-order*previous)/(order+1)
    log_radius = np.log(radius)
    f = coefficients*radius[:, None]**2
    inner = _decaying_integral(log_radius, f, orders+1)
    outer = _decaying_integral(-log_radius[::-1], f[::-1], orders)[::-1]
    potential = -(inner+outer)/(2*orders+1)[None, :]
    derivative = ((orders+1)[None, :]*inner-orders[None, :]*outer)/(2*orders+1)[None, :]
    if not np.all(np.isfinite(potential)) or not np.all(np.isfinite(derivative)):
        raise FloatingPointError("nonfinite multipole integral")
    spline = CubicHermiteSpline(log_radius, potential, derivative, axis=0, extrapolate=False)
    return MultipolePotential(grid, spline, coefficients)


@dataclass
class IsolatedScalarSolution:
    components: tuple[MassComponent, ...]
    spec: SaturatedActionSpec
    a0: float
    anomaly: MultipolePotential

    def evaluate(self, R, z):
        n = total_newtonian(self.components, R, z)
        correction = self.anomaly.evaluate(R, z)
        return {"potential": n["potential"]+correction["potential"],
                "acceleration": -n["gradient"]+correction["acceleration"],
                "newtonian_acceleration": -n["gradient"],
                "anomalous_acceleration": correction["acceleration"]}


def solve_isolated(components, spec, a0, grid):
    """Solve the anomaly jointly and add the exact analytic Newtonian field."""
    components = tuple(components)
    anomaly = solve_poisson(grid, lambda R, z: anomalous_source(components, spec, a0, R, z))
    return IsolatedScalarSolution(components, spec, a0, anomaly)
