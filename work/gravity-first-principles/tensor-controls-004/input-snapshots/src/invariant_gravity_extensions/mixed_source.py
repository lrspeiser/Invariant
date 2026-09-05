"""Tensor-product partials of the same Hankel and omitted-tail potentials.

The interpolation table needs i,j=0..3 separately, including total order six.
These are not additional physical fields or independent fitted derivatives.
Their quadrature convergence must be qualified before source interpolation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jv

from .hankel_tail import radial_tail_jet, sech2_source_jet


def hankel_mixed_jet(k, weights, transforms, vertical_jets, radius, height, G, *, batch_size=32):
    """Return physical partials [R order, z order, R index, z index]."""
    k, weights, transforms, vertical_jets, radius, height = (np.asarray(x, float) for x in
        (k, weights, transforms, vertical_jets, radius, height))
    if (k.ndim != 1 or weights.shape != k.shape or transforms.ndim != 2 or transforms.shape[1] != len(k) or
            radius.ndim != 1 or height.ndim != 1 or vertical_jets.shape != (len(transforms), 4, len(height), len(k)) or
            np.any(k < 0) or np.any(radius < 0) or not np.isfinite(G) or G <= 0 or
            type(batch_size) is not int or batch_size < 1 or
            any(np.any(~np.isfinite(x)) for x in [k, weights, transforms, vertical_jets, radius, height])):
        raise ValueError('aligned finite transforms and coordinates with positive G required')
    vertical = np.einsum('ck,cdzk->dzk', transforms, vertical_jets)*(2*np.pi*G*weights)
    result = np.empty((4, 4, len(radius), len(height)))
    for start in range(0, len(radius), batch_size):
        stop = min(start+batch_size, len(radius))
        x = radius[start:stop, None]*k
        J0, J1 = j0(x), j1(x)
        radial = [-J0, k*J1, k*k*.5*(J0-jv(2, x)), k**3*.25*(jv(3, x)-3*J1)]
        for i, kernel in enumerate(radial):
            for j in range(4):
                result[i, j, start:stop] = kernel@vertical[j].T
    return result


def leading_tail_mixed_jet(disks, components, k, weights, transforms, radius, height, G, cutoff,
                           *, log_nodes=128, precision=None):
    """Return all 16 partials of -4 pi G sum A_K(R) f(z), plus radial records."""
    radius, height = np.asarray(radius, float), np.asarray(height, float)
    if (radius.ndim != 1 or height.ndim != 1 or np.any(~np.isfinite(height)) or
            not np.isfinite(G) or G <= 0):
        raise ValueError('finite one-dimensional coordinates and positive G required')
    result = np.zeros((4, 4, len(radius), len(height)))
    records = []
    for name, transform in zip(components, transforms, strict=True):
        disk = disks[name]
        radial = radial_tail_jet(disk, k, weights, transform, radius, cutoff, log_nodes=log_nodes, precision=precision)
        a = np.array([radial[key] for key in ['potential', 'first', 'second', 'third']])
        f = sech2_source_jet(disk.height, height)
        result += -4*np.pi*G*a[:, None, :, None]*f[None, :, None, :]
        records.append({'component': name, **radial})
    return result, records
