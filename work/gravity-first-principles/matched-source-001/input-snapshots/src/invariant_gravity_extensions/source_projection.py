"""Source-only angular resolution audit for reflection-symmetric disks."""
from __future__ import annotations

import numpy as np
from scipy.special import roots_legendre


def half_gauss(nodes):
    if type(nodes) is not int or nodes < 8:
        raise ValueError('at least eight positive-half Gauss nodes required')
    mu, weights = roots_legendre(nodes)
    return (mu+1)/2, weights/2


def project_even_source(disks, radii, maximum_order, nodes):
    """Project density and its radial derivative at fixed spherical angle.

    Reflection symmetry is supplied by each disk's sech² vertical profile.
    Integrate both hemispheres using an explicit Gauss rule on mu in [0,1].
    No potential, velocity, gravity constant or response data enters this step.
    """
    if type(maximum_order) is not int or maximum_order < 0 or maximum_order % 2:
        raise ValueError('nonnegative even maximum Legendre order required')
    radii = np.asarray(radii, float)
    if radii.ndim != 1 or np.any(~np.isfinite(radii)) or np.any(radii <= 0):
        raise ValueError('finite positive shell radii required')
    mu, weights = half_gauss(nodes)
    sine = np.sqrt(1-mu*mu)
    R, z = radii[:, None]*sine, radii[:, None]*mu
    values = [disk.density_and_gradient(R, z) for disk in disks.values()]
    density = sum(v[0] for v in values)
    gradient = sum(v[1] for v in values)
    radial = sine*gradient[0]+mu*gradient[1]
    coefficients = np.zeros((len(radii), maximum_order+1))
    derivative_coefficients = np.zeros_like(coefficients)
    polynomial, previous = np.ones_like(mu), np.zeros_like(mu)
    for order in range(maximum_order+1):
        if order % 2 == 0:
            measure = weights*polynomial
            coefficients[:, order] = (2*order+1)*(density@measure)
            derivative_coefficients[:, order] = (2*order+1)*(radial@measure)
        previous, polynomial = polynomial, ((2*order+1)*mu*polynomial-order*previous)/(order+1)
    return {'radius_kpc': radii, 'coefficients': coefficients, 'radial_derivative_coefficients': derivative_coefficients,
            'source_half_gauss_nodes': nodes, 'maximum_order': maximum_order}


def projection_metrics(disks, projection, orders, nodes):
    """Evaluate density and gradient errors on a separate angular quadrature."""
    mu, weights = half_gauss(nodes)
    sine = np.sqrt(1-mu*mu)
    radii = np.asarray(projection['radius_kpc'])
    R, z = radii[:, None]*sine, radii[:, None]*mu
    values = [d.density_and_gradient(R, z) for d in disks.values()]
    density = sum(v[0] for v in values)
    grad = sum(v[1] for v in values)
    true_radial, true_theta = sine*grad[0]+mu*grad[1], mu*grad[0]-sine*grad[1]
    shell_mass = density@weights
    gradient_norm = np.hypot(true_radial, true_theta)@weights
    if np.any(shell_mass <= 0) or np.any(gradient_norm <= 0):
        raise ValueError('nonzero physical source and gradient on every registered shell required')
    polynomial, previous = np.ones_like(mu), np.zeros_like(mu)
    dp, dprevious = np.zeros_like(mu), np.zeros_like(mu)
    reconstructed = np.zeros_like(density)
    derivative_r, derivative_mu = np.zeros_like(density), np.zeros_like(density)
    records = []
    for order in range(max(orders)+1):
        if order % 2 == 0:
            coefficient = projection['coefficients'][:, order, None]
            reconstructed += coefficient*polynomial
            derivative_r += projection['radial_derivative_coefficients'][:, order, None]*polynomial
            derivative_mu += coefficient*dp
        if order in orders:
            derivative_theta = -sine*derivative_mu/radii[:, None]
            records.append({'maximum_order': order, 'evaluation_half_gauss_nodes': nodes,
                'radius_kpc': radii,
                'negative_density_fraction': (np.maximum(-reconstructed, 0)@weights)/shell_mass,
                'density_L1_fraction_error': (abs(reconstructed-density)@weights)/shell_mass,
                'density_peak_scaled_maximum_error': np.max(abs(reconstructed-density), axis=1)/np.max(density, axis=1),
                'gradient_L1_fraction_error': (np.hypot(derivative_r-true_radial, derivative_theta-true_theta)@weights)/gradient_norm,
                'relative_shell_mass_error': (reconstructed@weights)/shell_mass-1,
                'physical_hemisphere_density_integral': shell_mass,
                'physical_hemisphere_gradient_norm_integral': gradient_norm})
        k = 2*order+1
        nxt = (k*mu*polynomial-order*previous)/(order+1)
        dnxt = (k*(polynomial+mu*dp)-order*dprevious)/(order+1)
        previous, polynomial, dprevious, dp = polynomial, nxt, dp, dnxt
    return records
