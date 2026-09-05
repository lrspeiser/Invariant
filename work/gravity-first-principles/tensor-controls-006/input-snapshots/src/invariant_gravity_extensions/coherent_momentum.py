"""Internal-force audit of the historical coherent-monopole base operator.

Smooth analytic sources only, G=1. This is an audit, not a replacement force
law, observation adapter or claim about all possible coherence theories.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import simpson

from .isolated_axisymmetric import MultipoleGrid, total_newtonian


def _plummer_components(components):
    parts = tuple(components)
    if not parts or any(c.a != 0 for c in parts):
        raise ValueError("nonempty smooth Plummer components required")
    if len({c.name for c in parts}) != len(parts):
        raise ValueError("component names must be unique")
    return parts


def barycenter(components):
    parts = _plummer_components(components)
    return sum(c.gm*c.z_center for c in parts)/sum(c.gm for c in parts)


def plummer_shell_acceleration(components, radius):
    """Exact mean inward Newtonian field on spheres about the entire COM.

    For a Plummer source displaced by d, the angular-mean potential is
    -2*GM/(sqrt((r+d)^2+b^2)+sqrt((r-d)^2+b^2)). Differentiate in r.
    This closed form is independent of the numerical angular density integral.
    """
    parts = _plummer_components(components)
    r = np.asarray(radius, dtype=float)
    if np.any(r <= 0) or not np.all(np.isfinite(r)):
        raise ValueError("finite positive shell radius required")
    center = barycenter(parts)
    field = np.zeros_like(r)
    for part in parts:
        d = part.z_center-center
        plus, minus = np.hypot(r+d, part.b), np.hypot(r-d, part.b)
        field += 2*part.gm*((r+d)/plus+(r-d)/minus)/(plus+minus)**2
    if np.any(field < 0):
        raise FloatingPointError("positive smooth sources must have inward mean field")
    return field


def coherent_excess(g0, a0):
    """Simple-MOND minus Newtonian, evaluated without subtractive cancellation."""
    values = np.asarray(g0, dtype=float)
    if (not np.isfinite(a0) or a0 < 0 or np.any(values < 0) or
            not np.all(np.isfinite(values))):
        raise ValueError("finite nonnegative accelerations required")
    denominator = np.sqrt(values)*np.sqrt(values+4*a0)+values
    return np.divide(2*a0*values, denominator, out=np.zeros_like(values), where=denominator > 0)


def integrate_axisymmetric(grid: MultipoleGrid, values):
    """Volume integral 2*pi*int d(log r) r^3 int dmu values, full azimuth."""
    radius, mu, weights = grid.nodes()
    values = np.asarray(values, dtype=float)
    expected = np.broadcast_shapes((len(radius), 1), mu.shape)
    if values.shape != expected or not np.all(np.isfinite(values)):
        raise ValueError("finite full radial/angular array required")
    return float(2*np.pi*simpson(radius**3*np.sum(values*weights, axis=1), x=np.log(radius)))


def audit_scene(components, a0, grid: MultipoleGrid):
    parts = _plummer_components(components)
    center = barycenter(parts)
    radius, mu, weights = grid.nodes()
    sine = np.sqrt(1-mu*mu)
    R, z = radius[:, None]*sine, radius[:, None]*mu+center
    fields = total_newtonian(parts, R, z)
    rho, newton = fields["laplacian"]/(4*np.pi), -fields["gradient"]
    g0 = plummer_shell_acceleration(parts, radius)
    excess = coherent_excess(g0, a0)
    force_z = -excess[:, None]*mu
    mass = integrate_axisymmetric(grid, rho)
    scale = integrate_axisymmetric(grid, rho*np.linalg.norm(newton, axis=0))
    net_newton = integrate_axisymmetric(grid, rho*newton[1])
    net_excess = integrate_axisymmetric(grid, rho*force_z)
    direct_shell = np.sum(-(newton[0]*sine+newton[1]*mu)*weights, axis=1)/2
    probes = radius >= .01
    full_mass = sum(c.gm for c in parts)
    return {
        "mass_in_domain": mass, "analytic_total_mass": full_mass,
        "relative_mass_deficit": (full_mass-mass)/full_mass,
        "exact_barycenter_z": center,
        "finite_domain_barycenter_offset": integrate_axisymmetric(grid, rho*(z-center))/mass,
        "force_normalizer": scale,
        "newtonian_net_force_z": net_newton,
        "correction_net_force_z": net_excess,
        "total_net_force_z": net_newton+net_excess,
        "center_of_mass_acceleration_z": (net_newton+net_excess)/mass,
        "normalized_newtonian_net_force": abs(net_newton)/scale,
        "normalized_correction_net_force": abs(net_excess)/scale,
        "normalized_total_net_force": abs(net_newton+net_excess)/scale,
        "max_relative_shell_gauss_error_for_r_at_least_0_01": float(np.max(abs(direct_shell[probes]/g0[probes]-1))),
        "profile": {"radius": radius.tolist(), "shell_gN_exact": g0.tolist(),
                    "shell_gN_quadrature": direct_shell.tolist(), "coherent_excess": excess.tolist(),
                    "d_correction_force_z_d_log_r": (-2*np.pi*radius**3*excess*np.sum(rho*mu*weights, axis=1)).tolist()},
        "empirical_evidence": False,
    }
