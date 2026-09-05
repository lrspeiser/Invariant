"""Full Newtonian cylindrical jets from separable-source Hankel transforms.

Coordinates form a tensor-product (R,z) mesh. Every partial derivative is from
the same integral. The physical source is used only for external error checks.
"""
from __future__ import annotations

import numpy as np
from scipy.special import j0, j1, jv


def cylindrical_jet(k, weights, transforms, vertical_jets, radius, height, G, *, batch_size=32):
    """Potential and its first three Cartesian derivatives in cylindrical basis.

    vertical_jets has shape (component, derivative=0..3, z, k), with physical
    z derivatives already scaled. Return coordinate order (R,z,phi). The six
    independent nonzero third entries are RRR,RRz,Rzz,zzz,Rphiphi,zphiphi.
    """
    k, weights, transforms, vertical_jets, radius, height = (np.asarray(x, float) for x in
        (k, weights, transforms, vertical_jets, radius, height))
    if (k.ndim != 1 or weights.shape != k.shape or transforms.ndim != 2 or transforms.shape[1] != len(k) or
            radius.ndim != 1 or height.ndim != 1 or vertical_jets.shape != (len(transforms), 4, len(height), len(k)) or
            np.any(k < 0) or np.any(radius < 0) or not np.isfinite(G) or G <= 0 or
            type(batch_size) is not int or batch_size < 1 or
            any(np.any(~np.isfinite(x)) for x in [k, weights, transforms, vertical_jets, radius, height])):
        raise ValueError('aligned finite transforms and coordinates with positive G required')
    A = np.einsum('ck,cdzk->dzk', transforms, vertical_jets)*(2*np.pi*G*weights)
    fields = np.empty((10, len(radius), len(height)))
    for start in range(0, len(radius), batch_size):
        stop = min(start+batch_size, len(radius))
        x = radius[start:stop, None]*k
        J0, J1 = j0(x), j1(x)
        J1p, J1pp = .5*(J0-jv(2, x)), .25*(jv(3, x)-3*J1)
        fields[:, start:stop] = [-J0@A[0].T, (k*J1)@A[0].T, -J0@A[1].T,
            (k*k*J1p)@A[0].T, (k*J1)@A[1].T, -J0@A[2].T,
            (k**3*J1pp)@A[0].T, (k*k*J1p)@A[1].T, (k*J1)@A[2].T, -J0@A[3].T]
    psi, pR, pz, hRR, hRz, hzz, tRRR, tRRz, tRzz, tzzz = fields
    R = radius[:, None]
    hpp = np.divide(pR, R, out=hRR.copy(), where=R > 0)
    tRpp = np.divide(hRR-hpp, R, out=np.zeros_like(hRR), where=R > 0)
    tzpp = np.divide(hRz, R, out=tRRz.copy(), where=R > 0)
    hnorm = hRR*hRR+2*hRz*hRz+hzz*hzz+hpp*hpp
    grad_norm = 2*np.array([hRR*tRRR+2*hRz*tRRz+hzz*tRzz+hpp*tRpp,
                           hRR*tRRz+2*hRz*tRzz+hzz*tzzz+hpp*tzpp])
    third = np.array([tRRR, tRRz, tRzz, tzzz, tRpp, tzpp])
    third_norm = np.sqrt(np.einsum('i,irz,irz->rz', [1, 3, 3, 1, 3, 3], third, third))
    return {'radius': radius, 'height': height, 'potential': psi, 'gradient_R_z': np.array([pR, pz]),
            'hessian_RR_Rz_zz_pp': np.array([hRR, hRz, hzz, hpp]),
            'third_RRR_RRz_Rzz_zzz_Rpp_zpp': third, 'third_tensor_norm': third_norm,
            'laplacian': hRR+hzz+hpp, 'gradient_laplacian_R_z': np.array([tRRR+tRzz+tRpp, tRRz+tzzz+tzpp]),
            'hessian_norm': hnorm, 'gradient_hessian_norm_R_z': grad_norm}
