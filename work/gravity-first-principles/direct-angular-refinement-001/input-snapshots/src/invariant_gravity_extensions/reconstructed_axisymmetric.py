"""Isolated scalar fields for an explicitly assumed axisymmetric source density.

An observed surface map plus a vertical prescription defines a conditional
source, not an observed three-dimensional mass distribution. Numerical source,
field, boundary, and angular truncations require independent controls.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import PchipInterpolator

from .isolated_axisymmetric import MultipoleGrid, MultipolePotential, solve_poisson


def multipole_fields(potential: MultipolePotential, R, z, *, batch_size=8192):
    """Differentiate one shared potential through second order in cylindrical R,z.

    Radial derivatives are of the log-radius Hermite spline. Angular first and
    second derivatives use Legendre recurrences, including the axis limits.
    The returned laplacian is reconstructed from the potential, not supplied
    independently from density. Batching bounds memory on full source meshes.
    """
    R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
    shape = R.shape
    radius = np.hypot(R, z)
    grid = potential.grid
    if (np.any(R < 0) or np.any(~np.isfinite(radius)) or
            np.any(radius < grid.r_min*(1-1e-12)) or np.any(radius > grid.r_max*(1+1e-12))):
        raise ValueError("finite coordinates inside the declared multipole domain required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive integer batch size required")
    radius = np.clip(radius, grid.r_min, grid.r_max).ravel()
    rr, zz = R.ravel(), z.ravel()
    values = np.empty((8, len(radius)))
    for start in range(0, len(radius), batch_size):
        end = min(start+batch_size, len(radius))
        r = radius[start:end]
        mu, s = zz[start:end]/r, rr[start:end]/r
        f = potential.spline(np.log(r))
        ft = potential.spline(np.log(r), 1)
        ftt = potential.spline(np.log(r), 2)
        psi = np.zeros_like(r)
        radial, radial2, angular, mixed, angular2 = [np.zeros_like(r) for _ in range(5)]
        p, dp, ddp = np.ones_like(r), np.zeros_like(r), np.zeros_like(r)
        previous, dprevious, ddprevious = [np.zeros_like(r) for _ in range(3)]
        for order in range(grid.l_max+1):
            psi += f[:, order]*p
            radial += ft[:, order]*p
            radial2 += (ftt[:, order]-ft[:, order])*p
            angular += f[:, order]*dp
            mixed += (ft[:, order]-f[:, order])*dp
            angular2 += f[:, order]*(s*s*ddp-mu*dp)
            next_p = ((2*order+1)*mu*p-order*previous)/(order+1)
            next_dp = ((2*order+1)*(p+mu*dp)-order*dprevious)/(order+1)
            next_ddp = ((2*order+1)*(2*dp+mu*ddp)-order*ddprevious)/(order+1)
            previous, p = p, next_p
            dprevious, dp = dp, next_dp
            ddprevious, ddp = ddp, next_ddp
        pr, pt = radial/r, -s*angular/r
        hrr, hrt, htt = radial2/r**2, -s*mixed/r**2, (radial+angular2)/r**2
        hpp = (radial-mu*angular)/r**2
        values[:, start:end] = [psi, s*pr+mu*pt, mu*pr-s*pt,
                                s*s*hrr+2*s*mu*hrt+mu*mu*htt,
                                s*mu*(hrr-htt)+(mu*mu-s*s)*hrt,
                                mu*mu*hrr-2*s*mu*hrt+s*s*htt, hrr+htt+hpp, hpp]
    a = values.reshape((8,)+shape)
    return {"potential": a[0], "gradient": a[1:3],
            "hessian": np.array([[a[3], a[4]], [a[4], a[5]]]), "laplacian": a[6]}


@dataclass
class ReconstructedNewtonianSource:
    """Named source with numerical Newtonian field and a supplied Poisson source.

    poisson_source must return 4*pi*G*rho in the same units as the potential.
    The scalar anomaly uses that physical source for lap(psi); its difference
    from the numerically reconstructed laplacian is a convergence diagnostic.
    """
    name: str
    poisson_source: object
    potential: MultipolePotential

    @classmethod
    def build(cls, name, poisson_source, grid: MultipoleGrid):
        if not isinstance(name, str) or not name or not callable(poisson_source):
            raise ValueError("named callable physical Poisson source required")
        return cls(name, poisson_source, solve_poisson(grid, poisson_source))

    def fields(self, R, z):
        result = multipole_fields(self.potential, R, z)
        physical = np.asarray(self.poisson_source(R, z), float)
        if physical.shape != result["potential"].shape or np.any(~np.isfinite(physical)) or np.any(physical < 0):
            raise ValueError("finite nonnegative physical mass density required")
        result["laplacian"] = physical
        return result


@dataclass
class SurfaceDensityDisk:
    """Nonnegative axisymmetric surface profile with a normalized sech² lift.

    Input surface density and all lengths must use one consistent unit system.
    The fixed cosine taper is part of the source hypothesis, not a data cut.
    """
    radius: np.ndarray
    surface_density: np.ndarray
    height: float
    outer_radius: float
    taper_width: float

    def __post_init__(self):
        self.radius = np.asarray(self.radius, float)
        self.surface_density = np.asarray(self.surface_density, float)
        if (self.radius.ndim != 1 or len(self.radius) < 3 or self.radius.shape != self.surface_density.shape or
                np.any(~np.isfinite(self.radius)) or np.any(~np.isfinite(self.surface_density)) or
                np.any(self.radius < 0) or np.any(np.diff(self.radius) <= 0) or np.any(self.surface_density < 0) or
                not all(np.isfinite(x) for x in [self.height, self.outer_radius, self.taper_width]) or
                not self.height > 0 or not 0 < self.taper_width < self.outer_radius or
                self.outer_radius > self.radius[-1]):
            raise ValueError("finite nonnegative measured surface profile and valid lift required")
        self.interpolator = PchipInterpolator(self.radius, self.surface_density, extrapolate=False)

    def surface(self, R):
        R = np.asarray(R, float)
        if np.any(~np.isfinite(R)) or np.any(R < 0):
            raise ValueError("finite nonnegative cylindrical radius required")
        raw = self.interpolator(np.clip(R, self.radius[0], self.radius[-1]))
        phase = np.clip((R-(self.outer_radius-self.taper_width))/self.taper_width, 0, 1)
        return np.maximum(raw, 0)*(.5+.5*np.cos(np.pi*phase))

    def density(self, R, z):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        if np.any(~np.isfinite(z)):
            raise ValueError("finite vertical coordinate required")
        t = np.exp(-2*abs(z)/self.height)
        return self.surface(R)*(2*t/(1+t)**2)/self.height
