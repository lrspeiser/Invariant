"""Independent Newtonian midplane jets for separable axisymmetric sources.

This is an integral reference at z=0, not a full modified-gravity field solver.
All derivatives belong to the same finite-k potential. In particular, its trace
and trace gradient are not replaced by the exact untruncated physical density.
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma, j0, j1, jv, roots_legendre


def piecewise_gauss(edges, nodes):
    edges = np.asarray(edges, float)
    if (edges.ndim != 1 or len(edges) < 2 or np.any(~np.isfinite(edges)) or
            np.any(np.diff(edges) <= 0) or type(nodes) is not int or nodes < 4):
        raise ValueError('increasing finite edges and at least four nodes required')
    x, w = roots_legendre(nodes)
    half = np.diff(edges)/2
    return (((edges[:-1]+half)[:, None]+half[:, None]*x).ravel(), (half[:, None]*w).ravel())


def sech2_midplane_laplace(a):
    """Integral_0^infinity sech²(u) exp(-a*u) du, including the a=0 limit.

    Integration by parts and the alternating reciprocal series give the digamma
    expression. Large-a terms are the even Taylor coefficients of sech²(u)
    times their factorials; this avoids cancellation of two nearly equal ones.
    """
    a = np.asarray(a, float)
    if np.any(~np.isfinite(a)) or np.any(a < 0):
        raise ValueError('finite nonnegative dimensionless wavenumbers required')
    result = np.empty_like(a)
    small = a < 32
    t = a[small]
    result[small] = 1-t/2*(digamma(t/4+1)-digamma(t/4+.5))
    t = a[~small]
    result[~small] = np.polynomial.polynomial.polyval(t**-2, [1, -2, 16, -272, 7936, -353792, 22368256])/t
    return result


def radial_transform(radius, weights, surface, k, *, batch_size=256):
    """S_j(k) = integral R Sigma_j(R) J0(kR) dR for named source rows."""
    radius, weights, surface, k = (np.asarray(x, float) for x in (radius, weights, surface, k))
    if (radius.ndim != 1 or weights.shape != radius.shape or surface.ndim != 2 or
            surface.shape[1] != len(radius) or k.ndim != 1 or np.any(radius < 0) or np.any(k < 0) or
            type(batch_size) is not int or batch_size < 1 or
            any(np.any(~np.isfinite(x)) for x in [radius, weights, surface, k])):
        raise ValueError('finite aligned source and wavenumber quadratures required')
    measured = surface*(radius*weights)[None, :]
    result = np.empty((len(surface), len(k)))
    for start in range(0, len(k), batch_size):
        stop = min(start+batch_size, len(k))
        result[:, start:stop] = measured@j0(k[start:stop, None]*radius).T
    return result


def disk_transforms(disks, k, radial_nodes):
    """Resolve every measured-knot, central and taper interval separately."""
    names = list(disks)
    edges = np.unique(np.concatenate([np.r_[0., d.radius[d.radius < d.outer_radius],
                                           d.outer_radius-d.taper_width, d.outer_radius]
                                     for d in disks.values()]))
    radius, weights = piecewise_gauss(edges, radial_nodes)
    surface = np.array([disks[n].surface(radius) for n in names])
    return {'components': names, 'radial_edges': edges, 'radial_nodes_per_interval': radial_nodes,
            'radial_node_count': len(radius), 'k': np.asarray(k),
            'surface_hankel': radial_transform(radius, weights, surface, k),
            'component_mass': 2*np.pi*(surface@(radius*weights))}


def midplane_jet(k, weights, transforms, vertical_laplace, vertical_center, radius, G):
    """Potential, force gradient, Hessian and radial third derivatives at z=0.

    Coordinate order is orthonormal cylindrical (R,z,phi). Off-diagonal Hessian
    entries and z gradients of reflection-even invariants vanish in this plane.
    Z''(k,0)=k² Z(k,0)-2 k f(0) supplies the vertical contact contribution.
    """
    k, weights, transforms, vertical_laplace, vertical_center, radius = (np.asarray(x, float) for x in
        (k, weights, transforms, vertical_laplace, vertical_center, radius))
    if (k.ndim != 1 or weights.shape != k.shape or transforms.ndim != 2 or
            transforms.shape != vertical_laplace.shape or transforms.shape[1] != len(k) or
            vertical_center.shape != (len(transforms),) or radius.ndim != 1 or
            np.any(k < 0) or np.any(radius < 0) or not np.isfinite(G) or G <= 0 or
            any(np.any(~np.isfinite(x)) for x in [k, weights, transforms, vertical_laplace, vertical_center, radius])):
        raise ValueError('finite aligned transforms, nonnegative radii and positive G required')
    A = np.sum(transforms*vertical_laplace, axis=0)
    B = np.sum(transforms*(2*vertical_center[:, None]-k*vertical_laplace), axis=0)
    x = radius[:, None]*k
    J0, J1 = j0(x), j1(x)
    J1p, J1pp = .5*(J0-jv(2, x)), .25*(jv(3, x)-3*J1)
    measure = 2*np.pi*G*weights
    potential = -J0@(measure*A)
    force = J1@(measure*k*A)
    hrr = J1p@(measure*k*k*A)
    hzz = J0@(measure*k*B)
    hpp = np.divide(force, radius, out=hrr.copy(), where=radius > 0)
    drr = J1pp@(measure*k**3*A)
    dzz = -J1@(measure*k*k*B)
    dpp = np.divide(hrr-hpp, radius, out=np.zeros_like(hrr), where=radius > 0)
    H, dH = np.array([hrr, hzz, hpp]), np.array([drr, dzz, dpp])
    return {'radius': radius, 'potential': potential, 'radial_gradient': force,
            'hessian_RR_ZZ_PP': H, 'radial_derivative_hessian_RR_ZZ_PP': dH,
            'laplacian': H.sum(axis=0), 'radial_gradient_laplacian': dH.sum(axis=0),
            'hessian_norm': np.sum(H*H, axis=0), 'radial_gradient_hessian_norm': 2*np.sum(H*dH, axis=0)}
