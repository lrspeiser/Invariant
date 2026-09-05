"""Positive log-radius mollification of measured spherical source profiles.

Convolve dM/dlog(r) with a normalized Gaussian, then interpolate its logarithm
and integrate that same positive density. The width is a source-reconstruction
nuisance, never a fitted gravity constant. Outer continuation is explicit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import log_ndtr, logsumexp


def log_normal_interval(lower, upper):
    """Log probability of a normal interval, including far same-sign tails."""
    lower, upper = np.broadcast_arrays(lower, upper)
    if np.any(lower >= upper):
        raise ValueError('ordered normal integration bounds required')
    positive = lower > 0
    high = log_ndtr(np.where(positive, -lower, upper))
    low = log_ndtr(np.where(positive, -upper, lower))
    return high+np.log(-np.expm1(low-high))


@dataclass(frozen=True)
class LogMassSegment:
    """D(t)=D_reference exp[k(t-reference)] on [lower,upper]."""
    lower: float
    upper: float
    reference: float
    log_D_reference: float
    exponent: float


@dataclass
class SmoothedMass:
    log_density_spline: CubicSpline
    cumulative_mass: np.ndarray
    reference_radius: float
    width: float
    expected_total_mass: float
    metadata: dict

    @classmethod
    def build(cls, segments, reference_radius, inner_mass_at_reference, minimum_radius,
              maximum_radius, total_mass, *, width=.01, nodes=4097, metadata=None):
        if (not segments or not 0 < width <= .2 or type(nodes) is not int or nodes < 129
                or not 0 < minimum_radius < maximum_radius or reference_radius <= 0
                or inner_mass_at_reference <= 0 or total_mass <= 0):
            raise ValueError('valid positive source, width and integration grid required')
        t = np.linspace(np.log(minimum_radius/reference_radius), np.log(maximum_radius/reference_radius), nodes)
        contributions = []
        for segment in segments:
            k = segment.exponent
            lower = (segment.lower-t-k*width**2)/width
            upper = (segment.upper-t-k*width**2)/width
            contributions.append(segment.log_D_reference+k*(t-segment.reference)+.5*k*k*width**2+
                                 log_normal_interval(lower, upper))
        log_D = logsumexp(contributions, axis=0)
        if np.any(~np.isfinite(log_D)):
            raise FloatingPointError('nonfinite positive-density representation')
        spline = CubicSpline(t, log_D, extrapolate=False)
        q, w = leggauss(12)
        dt = np.diff(t)
        locations = t[:-1,None]+dt[:,None]*(q+1)/2
        increments = np.sum(np.exp(spline(locations))*w, axis=1)*dt/2
        # Far inside the declared constant-density core, the Gaussian moment
        # of M proportional to r^3 is exact up to an explicitly remote tail.
        initial = inner_mass_at_reference*np.exp(3*t[0]+4.5*width**2)
        masses = np.r_[initial, initial+np.cumsum(increments)]
        if np.any(~np.isfinite(masses)) or np.any(np.diff(masses) < 0):
            raise FloatingPointError('invalid integrated source mass')
        return cls(spline, masses, reference_radius, width, total_mass, metadata or {})

    def evaluate(self, radius):
        r = np.asarray(radius, dtype=float)
        if np.any(~np.isfinite(r)) or np.any(r <= 0):
            raise ValueError('positive finite radii required')
        t = np.log(r/self.reference_radius)
        knots = self.log_density_spline.x
        if np.any(t < knots[0]) or np.any(t > knots[-1]):
            raise ValueError('outside declared smooth-source integration domain')
        index = np.clip(np.searchsorted(knots, t, side='right')-1, 0, len(knots)-2)
        q, w = leggauss(12)
        delta = t-knots[index]
        points = knots[index][...,None]+delta[...,None]*(q+1)/2
        increment = np.sum(np.exp(self.log_density_spline(points))*w, axis=-1)*delta/2
        mass = self.cumulative_mass[index]+increment
        D = np.exp(self.log_density_spline(t))
        slope = self.log_density_spline(t, 1)
        density = D/(4*np.pi*r**3)
        gradient = density*(slope-3)/r
        return {'mass': mass, 'density': density, 'density_gradient': gradient}


def smooth_power_density(radius, density, *, width=.01, nodes=4097, outer_factor=2.0, outer_slope='last'):
    """Smooth a piecewise power-law gas density plus a finite outer closure."""
    from .cluster_pressure import PowerLawDensity

    profile = PowerLawDensity(radius, density)
    if not np.isfinite(outer_factor) or outer_factor <= 1 or outer_slope not in {'last','flat'}:
        raise ValueError('explicit finite outer continuation required')
    r = profile.radius
    rho = profile.density
    scale = r[0]
    t = np.log(r/scale)
    D = 4*np.pi*r**3*rho
    parts = [LogMassSegment(-np.inf, 0, 0, float(np.log(D[0])), 3)]
    for i, slope in enumerate(profile.slopes):
        parts.append(LogMassSegment(t[i], t[i+1], t[i], float(np.log(D[i])), slope+3))
    slope = profile.slopes[-1] if outer_slope == 'last' else 0.0
    k = slope+3
    extension = np.log(outer_factor)
    shell = D[-1]*extension if abs(k) < 1e-12 else D[-1]*np.expm1(k*extension)/k
    total = profile.mass_at_knots[-1]+shell
    parts.append(LogMassSegment(t[-1], t[-1]+extension, t[-1], float(np.log(D[-1])), k))
    return SmoothedMass.build(parts, scale, profile.mass_at_knots[0], scale/100,
                              r[-1]*outer_factor*2, total, width=width, nodes=nodes,
                              metadata={'kind':'gas_power_law_mollification','outer_factor':outer_factor,
                                        'outer_slope':outer_slope,'unmeasured_exterior_is_a_declared_closure':True})


def smooth_cumulative_mass(radius, mass, *, width=.01, nodes=4097, maximum_radius=None):
    """Smooth the exact old monotone linear-in-radius stellar mass source.

    The inherited monotonic correction is recorded. No gravity observable is
    used to select or weight a source knot. Total mass remains unchanged.
    """
    radius, mass = np.asarray(radius, float), np.asarray(mass, float)
    if (radius.ndim != 1 or radius.shape != mass.shape or len(radius) < 2 or np.any(radius <= 0)
            or np.any(mass <= 0) or np.any(np.diff(radius) <= 0) or np.any(~np.isfinite([radius,mass]))):
        raise ValueError('finite positive ordered cumulative source required')
    monotone = np.maximum.accumulate(mass)
    scale = radius[0]
    t = np.log(radius/scale)
    parts = [LogMassSegment(-np.inf, 0, 0, float(np.log(3*monotone[0])), 3)]
    slopes = np.diff(monotone)/np.diff(radius)
    for i, slope in enumerate(slopes):
        if slope > 0:
            parts.append(LogMassSegment(t[i], t[i+1], t[i], float(np.log(radius[i]*slope)), 1))
    outer = max(radius[-1]*2, maximum_radius or radius[-1]*2)
    return SmoothedMass.build(parts, scale, monotone[0], scale/100, outer,
                              monotone[-1], width=width, nodes=nodes,
                              metadata={'kind':'stellar_cumulative_mollification',
                                        'inherited_monotonic_corrections':int(np.count_nonzero(monotone != mass)),
                                        'maximum_inherited_monotonic_fraction':float(np.max(monotone/mass-1))})


def spherical_acceleration_derivatives(radius, mass, density, density_gradient, G):
    """Derivatives of one source: g=GM/r^2, with no independent interpolation."""
    r, mass, rho, drho = np.broadcast_arrays(radius, mass, density, density_gradient)
    if np.any(r <= 0) or np.any(mass <= 0) or np.any(rho < 0) or np.any(~np.isfinite([r,mass,rho,drho])):
        raise ValueError('finite positive-radius nonnegative source required')
    g = G*mass/r**2
    first = 4*np.pi*G*rho-2*g/r
    second = 4*np.pi*G*drho-2*first/r+2*g/r**2
    return g, first, second


def spherical_length_anomaly(spec, radius, g, first, second, length, a0):
    """Full radial action variation for an extended spherical source."""
    r, g, first, second = np.broadcast_arrays(radius, g, first, second)
    if (np.any(~np.isfinite([r,g,first,second])) or np.any(r <= 0) or np.any(g <= 0)
            or not np.isfinite(length) or length < 0 or not np.isfinite(a0) or a0 <= 0):
        raise ValueError('physical finite radial field and units required')
    x = g*g/a0**2
    h = length**2*(first*first+2*(g/r)**2)/a0**2
    px, ph, k1, k2, fraction = spec.partials(x,h)
    if length == 0:
        return px*g
    dx = 2*g*first/a0**2
    dh = length**2*(2*first*second+4*g/r*(first/r-g/r**2))/a0**2
    dph = ((k1+fraction*k2)*dx+fraction*k2*dh)/(x+h)
    return px*g-length**2*(dph*first+ph*second+2*ph*(first-g/r)/r)


def build_cluster_sources(packet, *, width=.01, nodes=4097, density_shift=0, outer_factor=2, outer_slope='last'):
    """Build from already exposed source columns; no pressure enters this step."""
    from .cluster_pressure import GM_SUN, KPC, MU_E, PROTON_MASS, G

    ne = np.asarray(packet['ne_cm3'],float).copy()
    errors = np.asarray(packet['ne_high_error' if density_shift > 0 else 'ne_low_error'],float)
    ne += density_shift*errors
    if np.any(ne <= 0):
        raise ValueError('source uncertainty produces nonpositive density; no clipping')
    radius = np.asarray(packet['density_radius_kpc'])*KPC
    gas = smooth_power_density(radius,ne*1e6*MU_E*PROTON_MASS,width=width,nodes=nodes,
                               outer_factor=outer_factor,outer_slope=outer_slope)
    stellar = None
    if packet['stellar'] is not None:
        s = packet['stellar']
        stellar = smooth_cumulative_mass(np.asarray(s['radius_kpc'])*KPC,np.asarray(s['mass_msun'])*GM_SUN/G,
                                         width=width,nodes=nodes,maximum_radius=radius[-1]*outer_factor*2)
    return {'gas':gas,'stellar':stellar,'input_ne_cm3':ne,'source_radius_m':radius,
            'width':width,'density_shift':density_shift,'outer_factor':outer_factor,'outer_slope':outer_slope}


def cluster_source_fields(sources, radius_nominal_m, nuisance):
    """One homologous distance transformation for masses and their derivatives."""
    from .cluster_pressure import G

    distance = nuisance.get('distance_scale',1)
    stellar_scale = nuisance.get('stellar_scale',1)
    if distance <= 0 or stellar_scale < 0:
        raise ValueError('positive distance and nonnegative stellar scaling required')
    radius = np.asarray(radius_nominal_m)*distance
    gas = sources['gas'].evaluate(radius_nominal_m)
    stellar = sources['stellar']
    exponents = {'mass':2.5,'density':-.5,'density_gradient':-1.5}
    total = {name:value*distance**exponents[name] for name,value in gas.items()}
    gas_density = total['density'].copy()
    if stellar is None:
        factor = nuisance.get('missing_stellar_gas_ratio',.1)*stellar_scale
        total = {name:value*(1+factor) for name,value in total.items()}
    else:
        star = stellar.evaluate(radius_nominal_m)
        powers = {'mass':2,'density':-1,'density_gradient':-2}
        total = {name:value+star[name]*distance**powers[name]*stellar_scale for name,value in total.items()}
    g, first, second = spherical_acceleration_derivatives(radius,total['mass'],total['density'],total['density_gradient'],G)
    return {'radius_m':radius,'gas_density':gas_density,'gbar':g,'gbar_first':first,'gbar_second':second,**total}
