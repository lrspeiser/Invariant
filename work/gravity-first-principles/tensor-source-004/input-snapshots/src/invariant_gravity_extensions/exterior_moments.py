"""Compact-source exterior moments and derivative-tail bounds.

The physical disk has infinite vertical tails. Here an explicit height cutoff
defines a positive compact numerical source. The omitted physical tail mass is
bounded and the resulting exterior field must be compared with the untruncated
Hankel reference. No bound here silently covers that separate source change.
"""
from __future__ import annotations

import math
from fractions import Fraction

import numpy as np

from .hankel_midplane import piecewise_gauss


def combine_even_moments(radial, vertical, maximum_order):
    """Integrate solid zonal harmonics from separable normalized moments.

    radial[k] = integral R Sigma(R) (R/scale)^(2k) dR;
    vertical[n] = integral f(z) (z/scale)^(2n) dz.
    Return M_l/scale^l, in mass units. Odd modes vanish by exact reflection.
    """
    radial, vertical = np.asarray(radial, float), np.asarray(vertical, float)
    if (type(maximum_order) is not int or maximum_order < 0 or maximum_order % 2 or
            radial.shape != (maximum_order//2+1,) or vertical.shape != radial.shape or
            np.any(~np.isfinite(radial)) or np.any(~np.isfinite(vertical))):
        raise ValueError('finite aligned moments through a nonnegative even order required')
    result = np.zeros(maximum_order+1)
    for order in range(0, maximum_order+1, 2):
        terms = []
        for j in range(order//2+1):
            coefficient = (-1)**j*math.comb(order, 2*j)*math.comb(2*j, j)/4**j
            terms.append(coefficient*radial[j]*vertical[order//2-j])
        result[order] = 2*np.pi*math.fsum(terms)
    return result


def disk_exterior_moments(disks, *, maximum_order=64, scale=36., radial_nodes=64,
                          vertical_nodes=16, vertical_extent=24., vertical_interval=.5):
    if (not np.isfinite(scale) or scale <= 0 or not np.isfinite(vertical_extent) or vertical_extent <= 0 or
            not np.isfinite(vertical_interval) or vertical_interval <= 0 or
            type(maximum_order) is not int or maximum_order < 0 or maximum_order % 2):
        raise ValueError('finite positive scales and nonnegative even maximum order required')
    intervals = round(vertical_extent/vertical_interval)
    if intervals < 1 or not np.isclose(intervals*vertical_interval, vertical_extent, atol=1e-12, rtol=0):
        raise ValueError('vertical interval must divide the declared height cutoff')
    u, wu = piecewise_gauss(np.arange(intervals+1)*vertical_interval, vertical_nodes)
    t = np.exp(-2*u)
    # Two hemispheres times the dimensionless f=sech²(u)/2.
    vertical_measure = 4*t/(1+t)**2*wu
    powers = np.arange(maximum_order//2+1)
    components = []
    for name, disk in disks.items():
        edges = np.unique(np.r_[0., disk.radius[disk.radius < disk.outer_radius],
                                disk.outer_radius-disk.taper_width, disk.outer_radius])
        R, wR = piecewise_gauss(edges, radial_nodes)
        radial = ((R[:, None]/scale)**(2*powers)*(wR*R*disk.surface(R))[:, None]).sum(axis=0)
        vertical = ((disk.height*u[:, None]/scale)**(2*powers)*vertical_measure[:, None]).sum(axis=0)
        moments = combine_even_moments(radial, vertical, maximum_order)
        components.append({'component': name, 'height': disk.height, 'outer_radius': disk.outer_radius,
            'radial_moments': radial, 'vertical_moments': vertical, 'scaled_multipole_moments': moments,
            'uncut_radial_mass': 2*np.pi*radial[0]})
    moments = sum(x['scaled_multipole_moments'] for x in components)
    cutoff_mass_fraction_bound = 2*np.exp(-2*vertical_extent)/(1+np.exp(-2*vertical_extent))
    support_squared = max(Fraction(float(d.outer_radius))**2+(Fraction(float(vertical_extent))*Fraction(float(d.height)))**2 for d in disks.values())
    support = math.sqrt(float(support_squared))
    while Fraction(support)**2 < support_squared:
        support = np.nextafter(support, np.inf)
    return {'maximum_order': maximum_order, 'scale': scale, 'radial_nodes': radial_nodes,
            'vertical_nodes': vertical_nodes, 'vertical_extent': vertical_extent, 'vertical_interval': vertical_interval,
            'components': components, 'scaled_multipole_moments': moments,
            'compact_source_mass': moments[0], 'physical_vertical_tail_mass_fraction': cutoff_mass_fraction_bound,
            'support_radius': support}


def derivative_tail_bounds(ratio, maximum_order):
    """Uniform exterior series-tail bounds in monopole units, orders 0..3.

    For a positive source inside radius s and r>s, |M_l| <= M*s^l.
    Potential, gradient, Hessian and third-tensor Frobenius norms are bounded
    termwise by (l+3)^p GM*s^l/r^(l+1+n), with p=0,2,4,6, n=0..3.
    Summing all l (including vanishing odd modes) is conservative. Eulerian
    generating functions sum the infinite polynomial-geometric tails exactly
    as rational numbers for the supplied floating ratio; round the final answer
    upward. Numerical moments and the omitted physical source are separate.
    """
    if not np.isfinite(ratio) or not 0 <= ratio < 1 or type(maximum_order) is not int or maximum_order < 0:
        raise ValueError('exterior radius ratio in [0,1) and nonnegative order required')
    if ratio == 0:
        return dict.fromkeys(['potential', 'gradient', 'hessian', 'third_tensor'], 0.)
    q = Fraction(float(ratio))
    eulerian = [[1], [1], [1, 1], [1, 4, 1], [1, 11, 11, 1],
                [1, 26, 66, 26, 1], [1, 57, 302, 302, 57, 1]]
    sums = [1/(1-q)]
    for j in range(1, 7):
        sums.append(q*sum(coefficient*q**i for i, coefficient in enumerate(eulerian[j]))/(1-q)**(j+1))
    bounds = []
    for power in [0, 2, 4, 6]:
        value = q**(maximum_order+1)*sum(math.comb(power, j)*(maximum_order+4)**(power-j)*sums[j] for j in range(power+1))
        bounds.append(float(np.nextafter(float(value), np.inf)))
    return dict(zip(['potential', 'gradient', 'hessian', 'third_tensor'], bounds, strict=True))


class ExteriorMomentField:
    """Analytic exterior potential and Cartesian jets from its finite series."""

    def __init__(self, moments, G, *, maximum_order=None, minimum_radius=None):
        self.scale = float(moments['scale'])
        self.support_radius = float(moments['support_radius'])
        self.maximum_order = moments['maximum_order'] if maximum_order is None else maximum_order
        self.minimum_radius = self.support_radius if minimum_radius is None else float(minimum_radius)
        if (type(self.maximum_order) is not int or not 0 <= self.maximum_order <= moments['maximum_order'] or
                self.maximum_order % 2 or not np.isfinite(G) or G <= 0 or
                not np.isfinite(self.minimum_radius) or self.minimum_radius < self.support_radius):
            raise ValueError('positive G, implemented even order and exterior domain required')
        self.G = G
        self.moments = np.array(moments['scaled_multipole_moments'][:self.maximum_order+1])

    def fields(self, R, z, *, batch_size=4096):
        R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
        radii = np.hypot(R, z)
        if (np.any(~np.isfinite(radii)) or np.any(R < 0) or np.any(radii <= self.support_radius) or
                np.any(radii < self.minimum_radius*(1-1e-12)) or type(batch_size) is not int or batch_size < 1):
            raise ValueError('finite coordinates strictly outside the compact source and in the declared domain required')
        flat_R, flat_z, flat_r = R.ravel(), z.ravel(), radii.ravel()
        values = np.empty((13, len(flat_r)))
        for start in range(0, len(flat_r), batch_size):
            end = min(start+batch_size, len(flat_r))
            r = flat_r[start:end]
            s, mu = flat_R[start:end]/r, flat_z[start:end]/r
            sums = np.zeros((10, len(r)))
            P, dP, ddP, dddP = np.ones_like(r), np.zeros_like(r), np.zeros_like(r), np.zeros_like(r)
            old, dold, ddold, dddold = [np.zeros_like(r) for _ in range(4)]
            factor = -self.G/r
            for l in range(self.maximum_order+1):
                f = self.moments[l]*factor
                a = l+1
                sums += np.array([f*P, -a*f*P, a*a*f*P, -a**3*f*P,
                    f*dP, -a*f*dP, a*a*f*dP, f*ddP, -a*f*ddP, f*dddP])
                k = 2*l+1
                next_P = (k*mu*P-l*old)/(l+1)
                next_dP = (k*(P+mu*dP)-l*dold)/(l+1)
                next_ddP = (k*(2*dP+mu*ddP)-l*ddold)/(l+1)
                next_dddP = (k*(3*ddP+mu*dddP)-l*dddold)/(l+1)
                old, P, dold, dP, ddold, ddP, dddold, dddP = P, next_P, dP, next_dP, ddP, next_ddP, dddP, next_dddP
                factor *= self.scale/r
            psi, B, C, D, E, F, Q, H, I, J = sums
            hrr, hrt = (C-B)/r**2, -s*(F-E)/r**2
            htt, hpp = (B+s*s*H-mu*E)/r**2, (B-mu*E)/r**2
            # Covariant third derivatives in the orthonormal spherical basis.
            trrr = (D-3*C+2*B)/r**3
            trrt = -s*(Q-3*F+2*E)/r**3
            trtt = (C-2*B+s*s*(I-2*H)-mu*(F-2*E))/r**3
            tttt = -s*(F+s*s*J-3*mu*H-E)/r**3+2*hrt/r
            trpp = (C-2*B-mu*(F-2*E))/r**3
            ttpp = -s*(F-E-mu*H)/r**3
            # Rotate all three tensor indices into the cylindrical basis.
            pR, pz = (s*B-mu*s*E)/r, (mu*B+s*s*E)/r
            hRR = s*s*hrr+2*s*mu*hrt+mu*mu*htt
            hRz = s*mu*(hrr-htt)+(mu*mu-s*s)*hrt
            hzz = mu*mu*hrr-2*s*mu*hrt+s*s*htt
            tRRR = s**3*trrr+3*s*s*mu*trrt+3*s*mu*mu*trtt+mu**3*tttt
            tRRz = s*s*mu*trrr+(2*s*mu*mu-s**3)*trrt+(mu**3-2*s*s*mu)*trtt-s*mu*mu*tttt
            tRzz = s*mu*mu*trrr+(mu**3-2*s*s*mu)*trrt+(s**3-2*s*mu*mu)*trtt+mu*s*s*tttt
            tzzz = mu**3*trrr-3*s*mu*mu*trrt+3*s*s*mu*trtt-s**3*tttt
            tRpp, tzpp = s*trpp+mu*ttpp, mu*trpp-s*ttpp
            values[:, start:end] = [psi, pR, pz, hRR, hRz, hzz, hpp, tRRR, tRRz, tRzz, tzzz, tRpp, tzpp]
        fields = values.reshape((13,)+R.shape)
        psi, pR, pz, hRR, hRz, hzz, hpp, tRRR, tRRz, tRzz, tzzz, tRpp, tzpp = fields
        tensor = fields[7:13]
        return {'radius': R, 'height': z, 'potential': psi, 'gradient_R_z': fields[1:3],
            'hessian_RR_Rz_zz_pp': fields[3:7], 'third_RRR_RRz_Rzz_zzz_Rpp_zpp': tensor,
            'hessian_norm': hRR*hRR+2*hRz*hRz+hzz*hzz+hpp*hpp,
            'third_tensor_norm': np.sqrt(np.einsum('i,i...,i...->...', [1, 3, 3, 1, 3, 3], tensor, tensor)),
            'laplacian': hRR+hzz+hpp, 'gradient_laplacian_R_z': np.array([tRRR+tRzz+tRpp, tRRz+tzzz+tzpp]),
            'gradient_hessian_norm_R_z': 2*np.array([hRR*tRRR+2*hRz*tRRz+hzz*tRzz+hpp*tRpp,
                                                    hRR*tRRz+2*hRz*tRzz+hzz*tzzz+hpp*tzpp])}
