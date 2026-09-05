"""Higher-precision cancellation arithmetic for the fixed radial tail source.

The PCHIP coefficients, core coefficient and transform are the same stored
binary numbers. The radial Gauss rule is refined at unchanged order. No
physical source or high-k cutoff is changed.
"""
from __future__ import annotations

import math
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.special import j0, j1, roots_legendre


@lru_cache(maxsize=8)
def accurate_legendre(nodes, digits):
    """Refine the same Gauss rule, including weights, beyond double precision."""
    c = mp.mp.clone()
    c.dps = digits

    def polynomial(x):
        previous, value = c.mpf(1), x
        for n in range(1, nodes):
            previous, value = value, ((2*n+1)*x*value-n*previous)/(n+1)
        return value, nodes*(x*value-previous)/(x*x-1)

    x, w = [], []
    for seed in roots_legendre(nodes)[0]:
        root = c.mpf(float(seed))
        for _ in range(10):
            value, derivative = polynomial(root)
            step = value/derivative
            root -= step
            if abs(step) < 4*c.eps:
                break
        _, derivative = polynomial(root)
        x.append(root)
        w.append(2/((1-root*root)*derivative**2))
    return tuple(x), tuple(w)


class PreciseLogGreen:
    """Evaluate the declared source polynomial and its log integrals accurately."""

    def __init__(self, disk, nodes=128, digits=50):
        self.ctx = mp.mp.clone()
        self.ctx.dps = digits
        c = self.ctx
        self.disk = disk
        self.x, self.w = accurate_legendre(nodes, digits)
        self.edges = np.unique(np.r_[0., disk.radius[disk.radius < disk.outer_radius],
            disk.outer_radius-disk.taper_width, disk.outer_radius])
        self.r0 = c.mpf(float(disk.radius[0]))
        self.coefficient = c.mpf(float(disk.core_coefficient))
        self.amplitude = c.mpf(float(disk.surface_density[0]))*c.exp(-self.coefficient*self.r0**2)
        self.coefficients = [[c.mpf(float(v)) for v in row] for row in disk.interpolator.c]
        self.mass = [c.mpf(0)]
        self.log_mass = [c.mpf(0)]
        for a, b in zip(self.edges[:-1], self.edges[1:], strict=True):
            mass, log = self.interval(c.mpf(float(a)), c.mpf(float(b)))
            self.mass.append(self.mass[-1]+mass)
            self.log_mass.append(self.log_mass[-1]+log)

    def core(self, radius):
        c = self.ctx
        if not radius:
            return c.mpf(0), c.mpf(0)
        u = radius**2
        y = self.coefficient*u
        if abs(y) > 10:
            raise ValueError('core series requires abs(c R^2)<=10')
        factor, mass, log = c.mpf(1), c.mpf(0), c.mpf(0)
        for n in range(200):
            mass += factor/(n+1)
            log += factor*(c.log(u)/(n+1)-c.mpf(1)/(n+1)**2)
            factor *= y/(n+1)
            if n > 20 and abs(factor) < c.eps:
                break
        return self.amplitude*u*mass/2, self.amplitude*u*log/4

    def surface(self, radius):
        c, d = self.ctx, self.disk
        if radius < self.r0:
            return self.amplitude*c.exp(self.coefficient*radius**2)
        if radius >= d.outer_radius:
            return c.mpf(0)
        cell = np.clip(np.searchsorted(d.radius, float(radius), side='right')-1, 0, len(d.radius)-2)
        t = radius-c.mpf(float(d.radius[cell]))
        a = [row[cell] for row in self.coefficients]
        value = ((a[0]*t+a[1])*t+a[2])*t+a[3]
        edge = c.mpf(float(d.outer_radius))-c.mpf(float(d.taper_width))
        if radius > edge:
            phase = (radius-edge)/c.mpf(float(d.taper_width))
            value *= (1+c.cos(c.pi*phase))/2
        return value

    def interval(self, a, b):
        c = self.ctx
        if b <= a:
            return c.mpf(0), c.mpf(0)
        if b <= self.r0:
            m1, l1 = self.core(b)
            m0, l0 = self.core(a)
            return m1-m0, l1-l0
        half, center = (b-a)/2, (a+b)/2
        radii = [center+half*x for x in self.x]
        measure = [half*w*r*self.surface(r) for r, w in zip(radii, self.w, strict=True)]
        return c.fsum(measure), c.fsum(v*c.log(r) for v, r in zip(measure, radii, strict=True))

    def evaluate(self, radius):
        c = self.ctx
        r = c.mpf(float(radius))
        if not r:
            return c.mpf(0), self.log_mass[-1]
        if r >= self.disk.outer_radius:
            return self.mass[-1], self.mass[-1]*c.log(r)
        cell = np.clip(np.searchsorted(self.edges, float(r), side='right')-1, 0, len(self.edges)-2)
        m, log = self.interval(c.mpf(float(self.edges[cell])), r)
        mass = self.mass[cell]+m
        return mass, c.log(r)*mass+self.log_mass[-1]-self.log_mass[cell]-log


@lru_cache(maxsize=4)
def accurate_bessel_band(radius, wavenumber, digits):
    """Cache shared source-independent values; retain their actual mp precision."""
    c = mp.mp.clone()
    c.dps = digits
    k = [c.mpf(v) for v in wavenumber]
    return tuple(tuple((c.besselj(0, c.mpf(r)*q), c.besselj(1, c.mpf(r)*q)) for q in k) for r in radius)


def precise_radial_integrals(disk, k, weights, transform, radius, cutoff, *, nodes=128, digits=50, low_k_limit=8.):
    if type(digits) is not int or digits < 25 or not np.isfinite(low_k_limit) or low_k_limit <= 0:
        raise ValueError('at least 25 decimal digits and a positive finite accurate band required')
    green = PreciseLogGreen(disk, nodes, digits)
    c = green.ctx
    low = k < low_k_limit
    band = accurate_bessel_band(tuple(map(float, radius)), tuple(map(float, k[low])), digits)
    factor = [c.mpf(float(w))*c.mpf(float(s))/c.mpf(float(q)) for w, s, q in zip(weights[low], transform[low], k[low], strict=True)]
    p_factor = [c.mpf(float(w))*c.mpf(float(s)) for w, s in zip(weights[low], transform[low], strict=True)]
    constant = c.log(c.mpf(2)/float(cutoff))-c.euler+c.fsum(c.mpf(float(w))/c.mpf(float(q)) for w, q in zip(weights, k, strict=True))
    value, first = [], []
    high_factor = weights[~low]*transform[~low]/k[~low]
    high_p_factor = weights[~low]*transform[~low]
    for r, row in zip(radius, band, strict=True):
        mass, log = green.evaluate(r)
        sum_low = c.fsum(a*j[0] for a, j in zip(factor, row, strict=True))
        sum_high = math.fsum(high_factor*j0(k[~low]*r))
        value.append(float(green.mass[-1]*constant-log-sum_low-c.mpf(sum_high)))
        p_low = c.fsum(a*j[1] for a, j in zip(p_factor, row, strict=True))
        p_high = math.fsum(high_p_factor*j1(k[~low]*r))
        first.append(float(-mass/c.mpf(float(r))+p_low+c.mpf(p_high)) if r else 0.)
    return np.array(value), np.array(first)
