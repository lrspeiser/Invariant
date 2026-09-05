"""Known harmonic moments, exact point-source tensors and exterior tail bounds."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_legendre

from invariant_gravity_extensions.exterior_moments import (
    ExteriorMomentField,
    combine_even_moments,
    derivative_tail_bounds,
    disk_exterior_moments,
)
from invariant_gravity_extensions.length_axisymmetric import RegularSurfaceDensityDisk


def test_ring_and_gaussian_solid_harmonic_moments():
    mass, radius, scale, order = 2.7, .8, 1.3, 64
    powers = np.arange(order//2+1)
    radial = mass/(2*np.pi)*(radius/scale)**(2*powers)
    vertical = np.zeros_like(radial)
    vertical[0] = 1.
    got = combine_even_moments(radial, vertical, order)
    expected = mass*(radius/scale)**np.arange(order+1)*eval_legendre(np.arange(order+1), 0.)
    np.testing.assert_allclose(got, expected, atol=2e-15, rtol=3e-14)
    # The mean of every nonconstant solid harmonic in an isotropic Gaussian
    # vanishes. These are analytic radial and vertical Gaussian moments.
    a, scale, order = .7, 3., 12
    radial = np.array([mass/(2*np.pi)*(2*a*a/scale**2)**j*math.factorial(j) for j in range(order//2+1)])
    vertical = np.array([(a/scale)**(2*j)*math.factorial(2*j)/(2**j*math.factorial(j)) for j in range(order//2+1)])
    got = combine_even_moments(radial, vertical, order)
    assert got[0] == pytest.approx(mass)
    np.testing.assert_allclose(got[1:], 0., atol=2e-14)


def test_disk_mass_quadrupole_and_explicit_tail_fraction():
    density, width, outer, height, scale = 3., .2, 1., .12, 1.4
    disk = RegularSurfaceDensityDisk(np.array([.1, .5, 1., 1.5]), np.full(4, density), height, outer, width)
    cutoff = 4.
    got = disk_exterior_moments({'test': disk}, maximum_order=8, scale=scale, radial_nodes=32, vertical_nodes=16, vertical_extent=cutoff)
    radial_mass = np.pi*density*(outer**2-outer*width+width*width/2)-2*density*width*width/np.pi
    fraction = np.tanh(cutoff)
    assert got['compact_source_mass'] == pytest.approx(radial_mass*fraction, rel=3e-14)
    assert got['physical_vertical_tail_mass_fraction'] == pytest.approx(1-fraction, rel=3e-13)
    z2 = quad(lambda u: (height*u)**2/np.cosh(u)**2, 0, cutoff, epsabs=1e-14)[0]
    radial2 = quad(lambda R: 2*np.pi*R**3*disk.surface(R), 0, outer, points=[outer-width], epsabs=1e-13)[0]
    expected = (radial_mass*z2-.5*radial2*fraction)/scale**2
    assert got['scaled_multipole_moments'][2] == pytest.approx(expected, rel=2e-13)
    assert got['support_radius'] >= np.hypot(outer, cutoff*height)


def paired_point_fields(R, z, separation, mass, G):
    potential = np.zeros_like(R)
    p = np.zeros((3, len(R)))
    H = np.zeros((3, 3, len(R)))
    T = np.zeros((3, 3, 3, len(R)))
    for center in [-separation, separation]:
        x = np.array([R, z-center, np.zeros_like(R)])
        r = np.linalg.norm(x, axis=0)
        gm = G*mass/2
        potential -= gm/r
        p += gm*x/r**3
        for i in range(3):
            for j in range(3):
                H[i, j] += gm*((i == j)/r**3-3*x[i]*x[j]/r**5)
                for k in range(3):
                    T[i, j, k] += gm*(-3*((i == j)*x[k]+(i == k)*x[j]+(j == k)*x[i])/r**5+15*x[i]*x[j]*x[k]/r**7)
    return potential, p, H, T


def test_exterior_jets_against_exact_two_point_sources_and_tail_bound():
    separation, mass, G, scale = .7, 2.3, 1.2, 1.1
    moments = np.zeros(65)
    moments[::2] = mass*(separation/scale)**np.arange(0, 65, 2)
    definition = {'scale': scale, 'support_radius': separation, 'maximum_order': 64, 'scaled_multipole_moments': moments}
    R = np.array([0., 0., 2., 3., 5., 1.3, 7.])
    z = np.array([3., -3., 0., .4, -.2, 2., 6.])
    result = ExteriorMomentField(definition, G).fields(R, z, batch_size=2)
    psi, p, H, T = paired_point_fields(R, z, separation, mass, G)
    expected_H = np.array([H[0, 0], H[0, 1], H[1, 1], H[2, 2]])
    expected_T = np.array([T[0, 0, 0], T[0, 0, 1], T[0, 1, 1], T[1, 1, 1], T[0, 2, 2], T[1, 2, 2]])
    np.testing.assert_allclose(result['potential'], psi, atol=3e-14, rtol=2e-12)
    np.testing.assert_allclose(result['gradient_R_z'], p[:2], atol=3e-14, rtol=2e-12)
    np.testing.assert_allclose(result['hessian_RR_Rz_zz_pp'], expected_H, atol=3e-14, rtol=2e-12)
    np.testing.assert_allclose(result['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], expected_T, atol=5e-14, rtol=3e-11)
    np.testing.assert_allclose(result['gradient_hessian_norm_R_z'], 2*np.einsum('ijr,ijkr->kr', H, T)[:2], atol=5e-14, rtol=3e-11)
    np.testing.assert_allclose(result['third_tensor_norm'], np.sqrt(np.sum(T*T, axis=(0, 1, 2))), atol=5e-14, rtol=3e-11)
    np.testing.assert_allclose(result['laplacian'], 0., atol=3e-15)
    np.testing.assert_allclose(result['gradient_laplacian_R_z'], 0., atol=5e-15)
    low = ExteriorMomentField(definition, G, maximum_order=8).fields(R, z)
    radius = np.hypot(R, z)
    delta = [abs(low['potential']-psi), np.linalg.norm(low['gradient_R_z']-p[:2], axis=0),
             np.sqrt(np.einsum('i,ir,ir->r', [1, 2, 1, 1], low['hessian_RR_Rz_zz_pp']-expected_H, low['hessian_RR_Rz_zz_pp']-expected_H)),
             np.sqrt(np.einsum('i,ir,ir->r', [1, 3, 3, 1, 3, 3], low['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-expected_T, low['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-expected_T))]
    for i, r in enumerate(radius):
        bounds = derivative_tail_bounds(separation/r, 8)
        for n, key in enumerate(['potential', 'gradient', 'hessian', 'third_tensor']):
            assert delta[n][i]/(G*mass/r**(n+1)) < bounds[key]


def test_exact_rational_tail_matches_long_positive_series():
    for ratio in [.1, .5, .8]:
        for order in [0, 12, 64]:
            bounds = derivative_tail_bounds(ratio, order)
            for power, value in zip([0, 2, 4, 6], bounds.values(), strict=True):
                reference = math.fsum((l+3)**power*ratio**l for l in range(order+1, 4000))
                assert value == pytest.approx(reference, rel=3e-14)
    assert derivative_tail_bounds(0., 64) == dict.fromkeys(['potential', 'gradient', 'hessian', 'third_tensor'], 0.)


def test_exterior_input_domain_checks():
    with pytest.raises(ValueError):
        derivative_tail_bounds(1., 20)
    with pytest.raises(ValueError):
        combine_even_moments([1], [1], 1)
    field = ExteriorMomentField({'scale': 1., 'support_radius': 1., 'maximum_order': 0, 'scaled_multipole_moments': [1.]}, 1.)
    with pytest.raises(ValueError):
        field.fields([.5], [0.])
