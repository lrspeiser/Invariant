"""THEORY_BENCHMARK_ONLY: static scalar weak-field, Born/thin-lens operator.

SI convention (-+++): g_tt=-(1+2 Phi/c^2), g_ij=(1-2 Psi/c^2) delta_ij.
alpha_hat = integral grad_perp(Phi+Psi) dell / c^2;
beta(theta) = theta - (D_ls/D_s) alpha_hat(D_l theta).
The potentials are inputs, not solutions of any gravity field equations.
See the frozen config for primary equations and explicit closure limitations.
"""

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable

import numpy as np
from scipy.special import roots_legendre

G_SI = 6.67430e-11
C_SI = 299792458.0
KPC_M = 3.085677581491367e19
MPC_M = 1000.0 * KPC_M
MSUN_KG = 1.98847e30
DISPOSITION = "THEORY_BENCHMARK_ONLY"


def _positive(value, name, allow_zero=False):
    value = float(value)
    if not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return value


def _vector(value, size, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (size,) or not np.all(np.isfinite(value)):
        raise ValueError(f"{name} must be a finite {size}-vector")
    return value


@dataclass(frozen=True)
class AngularDistances:
    """Three independently supplied angular-diameter distances, in meters.

    No redshift, cosmological model, or D_s-D_l approximation is used here.
    D_ls=0 is permitted as a manufactured zero-deflection geometry control.
    """

    D_l_m: float
    D_s_m: float
    D_ls_m: float

    def __post_init__(self):
        _positive(self.D_l_m, "D_l_m")
        _positive(self.D_s_m, "D_s_m")
        _positive(self.D_ls_m, "D_ls_m", allow_zero=True)

    @property
    def efficiency(self):
        return self.D_ls_m / self.D_s_m


@dataclass(frozen=True)
class ScalarMetric:
    """Vectorized Phi/Psi callables f(x,y,ell), each in m^2/s^2.

    A caller must declare the physical closure and register any point
    singularities. The guard is not a proof that an arbitrary field is regular.
    """

    phi: Callable
    psi: Callable
    closure: str
    point_centers_m: tuple = ()

    def __post_init__(self):
        if not callable(self.phi) or not callable(self.psi) or not self.closure.strip():
            raise ValueError("two callable potentials and an explicit closure label are required")
        for center in self.point_centers_m:
            _vector(center, 3, "point center")


@dataclass(frozen=True)
class SphericalComponent:
    """Synthetic -GM/sqrt(|r-center|^2+a^2); a=0 is a point mass."""

    mass_kg: float
    scale_m: float = 0.0
    center_m: tuple = (0.0, 0.0, 0.0)

    def __post_init__(self):
        _positive(self.mass_kg, "mass_kg", allow_zero=True)
        _positive(self.scale_m, "scale_m", allow_zero=True)
        _vector(self.center_m, 3, "center_m")


def manufactured_metric(components, *, eta, G=G_SI):
    """Explicit Psi=eta Phi test fixture; no candidate-theory closure claim."""
    components = tuple(components)
    _positive(G, "G")
    if not np.isfinite(eta):
        raise ValueError("eta must be finite")

    def phi(x, y, ell):
        x, y, ell = np.broadcast_arrays(x, y, ell)
        result = np.zeros_like(x, dtype=float)
        for component in components:
            if component.mass_kg == 0:
                continue
            cx, cy, cz = component.center_m
            radius = np.sqrt((x-cx)**2 + (y-cy)**2 + (ell-cz)**2 + component.scale_m**2)
            with np.errstate(divide="ignore", invalid="ignore"):
                result -= G * component.mass_kg / radius
        return result

    def psi(x, y, ell):
        return eta * phi(x, y, ell)

    points = tuple(c.center_m for c in components if c.scale_m == 0 and c.mass_kg > 0)
    label = "EQUAL_POTENTIALS_BENCHMARK_ONLY" if eta == 1 else f"MANUFACTURED_ETA_{eta:g}_ONLY"
    return ScalarMetric(phi, psi, label, points)


@lru_cache(maxsize=32)
def _legendre(order):
    nodes, weights = roots_legendre(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


def line_quadrature(*, order=256, scale_m=KPC_M, half_depth_m=None, ell_origin_m=0.0):
    """Gauss-Legendre under ell=origin+scale*tan(t); no tail correction.

    half_depth_m=None denotes an ideal infinite domain, not finite data support.
    Finite domains are [origin-half_depth, origin+half_depth].
    """
    if isinstance(order, bool) or int(order) != order or not 2 <= order <= 4096:
        raise ValueError("quadrature order must be an integer from 2 through 4096")
    _positive(scale_m, "scale_m")
    if not np.isfinite(ell_origin_m):
        raise ValueError("ell_origin_m must be finite")
    tmax = np.pi / 2 if half_depth_m is None else np.arctan(_positive(half_depth_m, "half_depth_m") / scale_m)
    nodes, weights = _legendre(int(order))
    t = tmax * nodes
    ell = ell_origin_m + scale_m * np.tan(t)
    d_ell_weights = tmax * weights * scale_m / np.cos(t)**2
    return ell, d_ell_weights


def transverse_gradient(metric, xy_m, ell_m, *, step_m, c=C_SI, weak_limit=1e-3):
    """Numerical gradient of Phi+Psi, with fourth-order centered stencils.

    Tests individual potentials on the stencil for finite values/weak amplitude.
    This sampled check does not establish global weak-field or Born validity.
    """
    xy_m = _vector(xy_m, 2, "xy_m")
    ell_m = np.asarray(ell_m, dtype=float)
    if not np.all(np.isfinite(ell_m)):
        raise ValueError("ell_m must be finite")
    _positive(step_m, "step_m")
    _positive(c, "c")
    _positive(weak_limit, "weak_limit")

    def W(xy):
        phi, psi = (np.asarray(fn(xy[0], xy[1], ell_m), dtype=float) for fn in (metric.phi, metric.psi))
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(psi)):
            raise ValueError("potential is nonfinite on the differentiation stencil")
        if max(float(np.max(np.abs(phi))), float(np.max(np.abs(psi)))) / c**2 > weak_limit:
            raise ValueError("sampled potential violates the declared weak-field limit")
        return phi + psi

    gradient = []
    for axis in range(2):
        hvec = np.zeros(2)
        hvec[axis] = step_m
        gradient.append((W(xy_m-2*hvec) - W(xy_m+2*hvec) + 8*(W(xy_m+hvec)-W(xy_m-hvec))) / (12*step_m))
    return np.stack(np.broadcast_arrays(*gradient), axis=-1)


def deflection(metric, xy_m, *, step_m=1e-4*KPC_M, order=256, scale_m=KPC_M,
               half_depth_m=None, ell_origin_m=0.0, c=C_SI, weak_limit=1e-3,
               singular_guard_steps=4.0):
    """Return alpha_hat in radians. It points outward for negative point Phi.

    The actual propagation-direction change has the opposite sign. Phi and
    Psi remain distinct inputs: no hidden factor two or assumed slip closure.
    """
    xy_m = _vector(xy_m, 2, "xy_m")
    _positive(step_m, "step_m")
    _positive(singular_guard_steps, "singular_guard_steps")
    for center in metric.point_centers_m:
        if np.linalg.norm(xy_m-np.asarray(center[:2])) <= singular_guard_steps * step_m:
            raise ValueError("ray/stencil too close to a registered point singularity")
    ell, weights = line_quadrature(order=order, scale_m=scale_m, half_depth_m=half_depth_m, ell_origin_m=ell_origin_m)
    gradient = transverse_gradient(metric, xy_m, ell, step_m=step_m, c=c, weak_limit=weak_limit)
    # Constant callable potentials also broadcast across the quadrature nodes.
    gradient = np.broadcast_to(gradient, (len(ell), 2))
    return weights @ gradient / c**2


def lens_map(metric, theta_rad, distances, **numerics):
    theta = _vector(theta_rad, 2, "theta_rad")
    return theta - distances.efficiency * deflection(metric, distances.D_l_m * theta, **numerics)


def lens_jacobian(metric, theta_rad, distances, *, angular_step_rad, **numerics):
    theta = _vector(theta_rad, 2, "theta_rad")
    _positive(angular_step_rad, "angular_step_rad")
    result = np.empty((2, 2))
    for axis in range(2):
        shift = np.zeros(2)
        shift[axis] = angular_step_rad
        result[:, axis] = (lens_map(metric, theta-2*shift, distances, **numerics)
                           - lens_map(metric, theta+2*shift, distances, **numerics)
                           + 8*(lens_map(metric, theta+shift, distances, **numerics)
                                - lens_map(metric, theta-shift, distances, **numerics))) / (12*angular_step_rad)
    return result


def signed_magnification(jacobian):
    jacobian = np.asarray(jacobian, dtype=float)
    if jacobian.shape != (2, 2) or not np.all(np.isfinite(jacobian)):
        raise ValueError("Jacobian must be finite and 2 by 2")
    determinant = float(np.linalg.det(jacobian))
    if abs(determinant) <= np.finfo(float).eps:
        raise ValueError("critical or numerically unresolved Jacobian determinant")
    return 1.0 / determinant
