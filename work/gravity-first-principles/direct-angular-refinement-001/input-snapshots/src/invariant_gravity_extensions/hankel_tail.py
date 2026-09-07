"""Leading high-wavenumber potential completion for a separable disk.

Z(k,z)=2 f(z)/k+O(k^-3) gives a tail potential -4 pi G f(z) A_K(R),
where A_K is the high-pass two-dimensional logarithmic Green potential.
All corrections below are derivatives of that scalar potential. The residual
source trace and its gradient must still be tested; neither is overwritten.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.special import j0, j1, roots_legendre

from .potential_join import cartesian_tensors, pack_cartesian


class DiskLogGreen:
    """Radial mass and logarithmic Green integral of the declared disk source."""

    def __init__(self, disk, *, nodes=128):
        if type(nodes) is not int or nodes < 8:
            raise ValueError('at least eight radial quadrature nodes required')
        self.disk = disk
        self.x, self.w = roots_legendre(nodes)
        self.edges = np.unique(np.r_[0., disk.radius[disk.radius < disk.outer_radius],
                                     disk.outer_radius-disk.taper_width, disk.outer_radius])
        masses, logs = [], []
        for a, b in zip(self.edges[:-1], self.edges[1:], strict=True):
            m, log = self.interval(a, b)
            masses.append(m)
            logs.append(log)
        self.prefix_mass = np.r_[np.longdouble(0), np.cumsum(np.array(masses, dtype=np.longdouble))]
        self.prefix_log = np.r_[np.longdouble(0), np.cumsum(np.array(logs, dtype=np.longdouble))]

    def core(self, radius):
        """Convergent analytic series for the regular quadratic-log core."""
        if radius == 0:
            return np.longdouble(0), np.longdouble(0)
        c = np.longdouble(self.disk.core_coefficient)
        r0 = np.longdouble(self.disk.radius[0])
        amplitude = np.longdouble(self.disk.surface_density[0])*np.exp(-c*r0*r0)
        u = np.longdouble(radius)**2
        y = c*u
        if abs(y) > 10:
            raise ValueError('core series currently admitted only for abs(c R^2)<=10')
        factor = np.longdouble(1)
        mass, log = np.longdouble(0), np.longdouble(0)
        for n in range(100):
            mass += factor/(n+1)
            log += factor*(np.log(u)/(n+1)-1/np.longdouble(n+1)**2)
            factor *= y/(n+1)
            if n > 20 and abs(factor) < np.longdouble('1e-25'):
                break
        return amplitude*u*mass/2, amplitude*u*log/4

    def interval(self, a, b):
        if b <= a:
            return np.longdouble(0), np.longdouble(0)
        if b <= self.disk.radius[0]:
            m1, l1 = self.core(b)
            m0, l0 = self.core(a)
            return m1-m0, l1-l0
        r = (a+b)/2+(b-a)/2*self.x
        measure = np.array((b-a)/2*self.w*r*self.disk.surface(r), dtype=np.longdouble)
        return np.sum(measure), np.sum(measure*np.log(r.astype(np.longdouble)))

    def evaluate(self, radius):
        radius = np.asarray(radius, float)
        mass, log_green = np.empty(radius.shape, np.longdouble), np.empty(radius.shape, np.longdouble)
        for index in np.ndindex(radius.shape):
            r = radius[index]
            if r == 0:
                mass[index], log_green[index] = 0, self.prefix_log[-1]
                continue
            cell = np.clip(np.searchsorted(self.edges, r, side='right')-1, 0, len(self.edges)-2)
            m, log = self.interval(self.edges[cell], min(r, self.disk.outer_radius))
            mass[index] = self.prefix_mass[cell]+m
            log_green[index] = (np.log(np.longdouble(r))*mass[index]
                                +self.prefix_log[-1]-self.prefix_log[cell]-log)
        return mass, log_green


def radial_tail_jet(disk, k, weights, transform, radius, cutoff, *, log_nodes=128, batch_size=32, precision=None):
    """A_K and its radial derivatives/axis quotients from one log potential.

    A_K=(log(2/K)-gamma) M-log_source-integral_0^K [S J0-M]/k.
    A_K'=-m(<R)/R+integral_0^K S J1. The radial Poisson identity supplies
    higher exact derivatives of this same integral, with its physical source.
    Small-R even series avoids dividing a near-zero cancellation by R twice.
    """
    k, weights, transform, radius = (np.asarray(v, float) for v in (k, weights, transform, radius))
    if (k.ndim != 1 or weights.shape != k.shape or transform.shape != k.shape or radius.ndim != 1 or
            np.any(k <= 0) or np.any(k >= cutoff) or np.any(radius < 0) or
            not np.isfinite(cutoff) or cutoff <= 0 or any(np.any(~np.isfinite(v)) for v in [k, weights, transform, radius])):
        raise ValueError('finite aligned positive wavenumbers inside cutoff and nonnegative radii required')
    green = DiskLogGreen(disk, nodes=log_nodes)
    mass, log_source = green.evaluate(radius)
    total = green.prefix_mass[-1]
    result = np.empty((6, len(radius)))
    small = radius*cutoff < .1
    for start in range(0, len(radius), batch_size):
        stop = min(start+batch_size, len(radius))
        r = radius[start:stop]
        x = r[:, None]*k
        J0, J1 = j0(x), j1(x)
        integral = np.sum(((transform*J0).astype(np.longdouble)-total)*
            (weights/k).astype(np.longdouble), axis=1)
        A = (np.log(np.longdouble(2)/cutoff)-np.longdouble(np.euler_gamma))*total-log_source[start:stop]-integral
        sigma, sigma_prime = disk.surface_and_derivative(r)
        B = sigma-J0@(k*weights*transform)
        Bp = sigma_prime+J1@(k*k*weights*transform)
        p = np.divide(-mass[start:stop], r, out=np.zeros_like(r, dtype=np.longdouble), where=r > 0)
        p += np.sum((J1*transform*weights).astype(np.longdouble), axis=1)
        p = np.array(p, float)
        p_over_r = np.divide(p, r, out=-B/2, where=r > 0)
        h = -B-p_over_r
        t = -Bp-np.divide(h-p_over_r, r, out=np.zeros_like(r), where=r > 0)
        t_pp = np.divide(h-p_over_r, r, out=np.zeros_like(r), where=r > 0)
        result[:, start:stop] = [A, p, h, t, p_over_r, t_pp]
    if precision is not None:
        from .precise_tail import precise_radial_integrals

        A, p = precise_radial_integrals(disk, k, weights, transform, radius, cutoff, nodes=log_nodes, **precision)
        delta_p = p-result[1]
        delta_pp = np.divide(delta_p, radius, out=np.zeros_like(radius), where=radius > 0)
        result[0], result[1] = A, p
        result[2] -= delta_pp
        result[3] += np.divide(2*delta_pp, radius, out=np.zeros_like(radius), where=radius > 0)
        result[4] += delta_pp
        result[5] -= np.divide(2*delta_pp, radius, out=np.zeros_like(radius), where=radius > 0)
    if np.any(small):
        r = radius[small]
        central = disk.surface_density[0]*np.exp(-disk.core_coefficient*disk.radius[0]**2)
        coeff = np.array([central*disk.core_coefficient**n/math.factorial(n)
            -(-1)**n*np.sum((weights*transform*k**(2*n+1)).astype(np.longdouble))/
                (4**n*math.factorial(n)**2) for n in range(8)], float)
        result[1, small] = -sum(coeff[n]*r**(2*n+1)/(2*n+2) for n in range(8))
        result[2, small] = -sum((2*n+1)*coeff[n]*r**(2*n)/(2*n+2) for n in range(8))
        result[3, small] = -sum(2*n*(2*n+1)*coeff[n]*r**(2*n-1)/(2*n+2) for n in range(1, 8))
        result[4, small] = -sum(coeff[n]*r**(2*n)/(2*n+2) for n in range(8))
        result[5, small] = -sum(2*n*coeff[n]*r**(2*n-1)/(2*n+2) for n in range(1, 8))
    return {'potential': result[0], 'first': result[1], 'second': result[2], 'third': result[3],
        'first_over_radius': result[4], 'derivative_first_over_radius': result[5],
        'radial_mass': np.array(mass, float), 'total_radial_mass': float(total), 'log_nodes': log_nodes}


def sech2_source_jet(height, z):
    z = np.asarray(z, float)
    q = np.exp(-2*abs(z)/height)
    f = 2*q/(1+q)**2/height
    t = np.tanh(z/height)
    return np.array([f, -2*t*f/height, (6*t*t-2)*f/height**2, (16*t-24*t**3)*f/height**3])


def complete_leading_tail(near, disks, components, k, weights, transforms, radius, height, G, cutoff, *, log_nodes=128, precision=None):
    """Add -4 pi G sum_c f_c(z) A_K,c(R), with all Cartesian derivatives."""
    radius, height = np.asarray(radius), np.asarray(height)
    shape = (len(radius), len(height))
    correction = np.zeros((13,)+shape)
    radial_records = []
    for name, S in zip(components, transforms, strict=True):
        d = disks[name]
        a = radial_tail_jet(d, k, weights, S, radius, cutoff, log_nodes=log_nodes, precision=precision)
        f, fp, fpp, fppp = sech2_source_jet(d.height, height)
        A, p, h, t, pp, tpp = [a[key][:, None] for key in ['potential', 'first', 'second', 'third', 'first_over_radius', 'derivative_first_over_radius']]
        correction += -4*np.pi*G*np.array([A*f, p*f, A*fp, h*f, p*fp, A*fpp, pp*f,
                                         t*f, h*fp, p*fpp, A*fppp, tpp*f, pp*fp])
        radial_records.append({'component': name, **a})
    temporary = {'potential': correction[0], 'gradient_R_z': correction[1:3],
        'hessian_RR_Rz_zz_pp': correction[3:7], 'third_RRR_RRz_Rzz_zzz_Rpp_zpp': correction[7:13]}
    n = cartesian_tensors(near)
    c = cartesian_tensors(temporary)
    corrected = pack_cartesian(*[a+b for a, b in zip(n, c, strict=True)], radius, height)
    return corrected, {'radial_records': radial_records, 'correction': temporary}
