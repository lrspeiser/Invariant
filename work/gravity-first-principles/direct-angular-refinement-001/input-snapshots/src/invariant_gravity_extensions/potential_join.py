"""C3 radial joining of two potentials, retaining every product-rule term.

This is a numerical representation, not a gravity modification. Both providers
must use the same potential zero. A joined field can introduce spurious source
density; its trace and trace gradient must be audited, never set by hand.
"""
from __future__ import annotations

from itertools import permutations

import numpy as np


def radial_weight_jet(R, z, inner, outer):
    """Septic step and its Cartesian derivatives through order three."""
    R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
    r = np.hypot(R, z)
    if (np.any(~np.isfinite(r)) or np.any(R < 0) or not np.isfinite(inner) or
            not np.isfinite(outer) or not 0 < inner < outer):
        raise ValueError('finite coordinates and ordered positive join radii required')
    t = np.clip((r-inner)/(outer-inner), 0., 1.)
    # Symmetric evaluation avoids cancellation near the upper endpoint.
    q = np.minimum(t, 1-t)
    w = q**4*(35+q*(-84+q*(70-20*q)))
    w = np.where(t <= .5, w, 1-w)
    span = outer-inner
    d1 = 140*t**3*(1-t)**3/span
    d2 = 420*t**2*(1-t)**2*(1-2*t)/span**2
    d3 = 840*t*(1-t)*(1-5*t+5*t*t)/span**3
    invr = np.divide(1., r, out=np.zeros_like(r), where=r > 0)
    u = np.array([R*invr, z*invr, np.zeros_like(r)])
    eye = np.eye(3).reshape((3, 3)+(1,)*r.ndim)
    uu = np.einsum('i...,j...->ij...', u, u)
    p = d1*u
    h = (d2-d1*invr)*uu+d1*invr*eye
    sym = (np.einsum('ij...,k...->ijk...', eye, u)+np.einsum('ik...,j...->ijk...', eye, u)
           +np.einsum('jk...,i...->ijk...', eye, u))
    third = ((d3-3*d2*invr+3*d1*invr**2)*np.einsum('i...,j...,k...->ijk...', u, u, u)
             +(d2*invr-d1*invr**2)*sym)
    return w, p, h, third


def cartesian_tensors(fields):
    """Expand axisymmetric jets in the orthonormal (R,z,phi) basis."""
    psi = np.asarray(fields['potential'], float)
    p = np.zeros((3,)+psi.shape)
    h = np.zeros((3, 3)+psi.shape)
    t = np.zeros((3, 3, 3)+psi.shape)
    p[:2] = fields['gradient_R_z']
    h[0, 0], h[0, 1], h[1, 1], h[2, 2] = fields['hessian_RR_Rz_zz_pp']
    h[1, 0] = h[0, 1]
    for index, value in zip([(0, 0, 0), (0, 0, 1), (0, 1, 1), (1, 1, 1), (0, 2, 2), (1, 2, 2)],
                            fields['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], strict=True):
        for order in set(permutations(index)):
            t[order] = value
    if any(np.any(~np.isfinite(v)) for v in (psi, p, h, t)):
        raise ValueError('finite potential derivatives required')
    return psi, p, h, t


def pack_cartesian(psi, p, h, t, R, z):
    """Pack one scalar potential jet and derive its invariants by contraction."""
    return {'radius': np.asarray(R), 'height': np.asarray(z), 'potential': psi,
        'gradient_R_z': p[:2], 'hessian_RR_Rz_zz_pp': np.array([h[0, 0], h[0, 1], h[1, 1], h[2, 2]]),
        'third_RRR_RRz_Rzz_zzz_Rpp_zpp': np.array([t[0, 0, 0], t[0, 0, 1], t[0, 1, 1], t[1, 1, 1], t[0, 2, 2], t[1, 2, 2]]),
        'hessian_norm': np.einsum('ij...,ij...->...', h, h),
        'third_tensor_norm': np.sqrt(np.einsum('ijk...,ijk...->...', t, t)),
        'laplacian': np.einsum('ii...->...', h),
        'gradient_laplacian_R_z': np.einsum('iik...->k...', t)[:2],
        'gradient_hessian_norm_R_z': 2*np.einsum('ij...,ijk...->k...', h, t)[:2]}


def blend_potential_jets(near, far, R, z, *, inner, outer):
    """Return derivatives of (1-w)*near_potential + w*far_potential.

    Input jets must already have the broadcast coordinate shape. This routine
    cannot make an inaccurate input field valid. Its nonzero product terms are
    retained even when that worsens the numerical source residual.
    """
    R, z = np.broadcast_arrays(np.asarray(R, float), np.asarray(z, float))
    if np.shape(near['potential']) != R.shape or np.shape(far['potential']) != R.shape:
        raise ValueError('potential shape must match broadcast coordinate shape')
    w, wp, wh, wt = radial_weight_jet(R, z, inner, outer)
    npsi, np_, nh, nt = cartesian_tensors(near)
    fpsi, fp, fh, ft = cartesian_tensors(far)
    dpsi, dp, dh, dt = fpsi-npsi, fp-np_, fh-nh, ft-nt
    psi = npsi+w*dpsi
    p = np_+w*dp+wp*dpsi
    h = nh+w*dh+wh*dpsi+np.einsum('i...,j...->ij...', wp, dp)+np.einsum('j...,i...->ij...', wp, dp)
    t = (nt+w*dt+wt*dpsi+np.einsum('i...,jk...->ijk...', wp, dh)
         +np.einsum('j...,ik...->ijk...', wp, dh)+np.einsum('k...,ij...->ijk...', wp, dh)
         +np.einsum('ij...,k...->ijk...', wh, dp)+np.einsum('ik...,j...->ijk...', wh, dp)
         +np.einsum('jk...,i...->ijk...', wh, dp))
    return pack_cartesian(psi, p, h, t, R, z)
