"""Independent direct spatial Newtonian quadrature for exterior field checks.

Cartesian inverse-distance derivatives are summed over positive source weights.
This deliberately expensive reference is not a production near-source solver.
"""
from __future__ import annotations

import numpy as np
from scipy.special import roots_laguerre

from .hankel_midplane import piecewise_gauss


def point_jet_sum(R, z, source_x, source_y, source_z, gm):
    R, z, source_x, source_y, source_z, gm = (np.asarray(x, float) for x in (R, z, source_x, source_y, source_z, gm))
    if (R.ndim != 1 or z.shape != R.shape or any(x.shape != gm.shape for x in [source_x, source_y, source_z]) or gm.ndim != 1 or
            any(np.any(~np.isfinite(x)) for x in [R, z, source_x, source_y, source_z, gm]) or np.any(gm < 0)):
        raise ValueError('aligned finite point sources with nonnegative weights required')
    dx, dy, dz = R[:, None]-source_x, -source_y[None, :], z[:, None]-source_z
    squared = dx*dx+dy*dy+dz*dz
    if np.any(squared == 0):
        raise ValueError('direct point quadrature may not sample a singular coincidence')
    inv = 1/np.sqrt(squared)
    a3 = gm*inv**3
    a5 = a3*inv**2
    a7 = a5*inv**2
    return np.array([-np.sum(gm*inv, axis=1), np.sum(a3*dx, axis=1), np.sum(a3*dz, axis=1),
        np.sum(a3-3*a5*dx*dx, axis=1), np.sum(-3*a5*dx*dz, axis=1),
        np.sum(a3-3*a5*dz*dz, axis=1), np.sum(a3-3*a5*dy*dy, axis=1),
        np.sum(-9*a5*dx+15*a7*dx**3, axis=1), np.sum(-3*a5*dz+15*a7*dx*dx*dz, axis=1),
        np.sum(-3*a5*dx+15*a7*dx*dz*dz, axis=1), np.sum(-9*a5*dz+15*a7*dz**3, axis=1),
        np.sum(-3*a5*dx+15*a7*dx*dy*dy, axis=1), np.sum(-3*a5*dz+15*a7*dz*dy*dy, axis=1),
        np.sum(a3*dy, axis=1)])


def unpack_point_jet(values):
    psi, _pR, _pz, hRR, hRz, hzz, hpp, tRRR, tRRz, tRzz, tzzz, tRpp, tzpp, pphi = values
    T = values[7:13]
    return {'potential': psi, 'gradient_R_z': values[1:3], 'hessian_RR_Rz_zz_pp': values[3:7],
        'third_RRR_RRz_Rzz_zzz_Rpp_zpp': T, 'azimuthal_gradient': pphi,
        'hessian_norm': hRR*hRR+2*hRz*hRz+hzz*hzz+hpp*hpp,
        'third_tensor_norm': np.sqrt(np.einsum('i,i...,i...->...', [1, 3, 3, 1, 3, 3], T, T)),
        'laplacian': hRR+hzz+hpp, 'gradient_laplacian_R_z': np.array([tRRR+tRzz+tRpp, tRRz+tzzz+tzpp]),
        'gradient_hessian_norm_R_z': 2*np.array([hRR*tRRR+2*hRz*tRRz+hzz*tRzz+hpp*tRpp,
                                               hRR*tRRz+2*hRz*tRzz+hzz*tzzz+hpp*tzpp])}


def direct_disk_fields(disks, R, z, G, *, radial_nodes=8, vertical_nodes=64, azimuth_nodes=64, radial_batch=4):
    """Integrate the physical infinite sech-squared lift with Laguerre nodes.

    v=2*abs(z')/h maps f(z') dz' on each hemisphere to
    exp(-v)/(1+exp(-v))² dv. The quadrature has no finite vertical cutoff.
    Periodic trapezoidal nodes integrate azimuth. Split radial integration at
    every physical profile knot, core join and taper edge.
    """
    R, z = np.asarray(R, float), np.asarray(z, float)
    if (R.ndim != 1 or z.shape != R.shape or np.any(R < 0) or not np.isfinite(G) or G <= 0 or
            type(vertical_nodes) is not int or vertical_nodes < 8 or type(azimuth_nodes) is not int or
            azimuth_nodes < 8 or azimuth_nodes % 2 or type(radial_batch) is not int or radial_batch < 1):
        raise ValueError('registered point coordinates, positive G and positive even angular quadrature required')
    v, wv = roots_laguerre(vertical_nodes)
    vertical_weight = np.tile(wv/(1+np.exp(-v))**2, 2)
    signed_v = np.r_[v, -v]
    angle = np.arange(azimuth_nodes)*2*np.pi/azimuth_nodes
    cosine, sine = np.cos(angle), np.sin(angle)
    total = np.zeros((14, len(R)))
    masses = []
    for name, disk in disks.items():
        edges = np.unique(np.r_[0., disk.radius[disk.radius < disk.outer_radius],
                                disk.outer_radius-disk.taper_width, disk.outer_radius])
        radial, wr = piecewise_gauss(edges, radial_nodes)
        radial_mass = wr*radial*disk.surface(radial)
        masses.append({'component': name, 'mass_quadrature': 2*np.pi*radial_mass.sum()*vertical_weight.sum()})
        for start in range(0, len(radial), radial_batch):
            stop = min(start+radial_batch, len(radial))
            shape = (stop-start, len(signed_v), azimuth_nodes)
            x = np.broadcast_to(radial[start:stop, None, None]*cosine, shape).ravel()
            y = np.broadcast_to(radial[start:stop, None, None]*sine, shape).ravel()
            zz = np.broadcast_to(disk.height*signed_v[None, :, None]/2, shape).ravel()
            mass = np.broadcast_to(radial_mass[start:stop, None, None]*vertical_weight[None, :, None]*2*np.pi/azimuth_nodes, shape).ravel()
            total += point_jet_sum(R, z, x, y, zz, G*mass)
    return {**unpack_point_jet(total), 'radius': R, 'height': z, 'source_mass_quadrature': masses,
            'vertical_mass_fraction_quadrature': float(vertical_weight.sum())}
