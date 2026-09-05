"""Independent integral and spherical-Gaussian controls, without observations."""
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import erf, erfcx

from invariant_gravity_extensions.hankel_midplane import (
    midplane_jet,
    piecewise_gauss,
    radial_transform,
    sech2_midplane_laplace,
)


def test_sech_laplace_against_adaptive_direct_integral():
    for a in [0., 1e-10, .001, .1, 1., 10., 31., 32., 33., 50., 100., 1e3, 1e6]:
        scale = max(1., a)

        def integrand(v, scale=scale, a=a):
            t = np.exp(-2*v/scale)
            return 4*t/(1+t)**2*np.exp(-a*v/scale)/scale

        reference, _ = quad(integrand, 0, np.inf, epsabs=1e-18, epsrel=5e-13)
        assert sech2_midplane_laplace(a) == pytest.approx(reference, rel=1e-10, abs=1e-17)


def test_radial_hankel_gaussian_against_closed_form_and_partition():
    r, w = piecewise_gauss(np.linspace(0, 12, 49), 24)
    k = np.r_[0., np.geomspace(.01, 15., 50)]
    surface = np.exp(-r*r/2)/(2*np.pi)
    whole = radial_transform(r, w, surface[None, :], k)[0]
    expected = np.exp(-k*k/2)/(2*np.pi)
    np.testing.assert_allclose(whole, expected, atol=8e-16, rtol=2e-12)
    split = radial_transform(r, w, np.array([.3*surface, .7*surface]), k, batch_size=7).sum(axis=0)
    np.testing.assert_allclose(split, whole, atol=3e-16, rtol=2e-12)


def test_midplane_jet_against_three_dimensional_spherical_gaussian():
    # Exact enclosed mass supplies an independent Cartesian tensor and its
    # radial derivative; this does not use the cylindrical Bessel identities.
    a, mass, G = 1.3, 2.7, .8
    r = np.array([0., .05, .2, 1., 2., 5., 10.])
    k, w = piecewise_gauss(np.linspace(0, 16, 65), 24)
    S = mass/(2*np.pi)*np.exp(-.5*a*a*k*k)
    Z = erfcx(a*k/np.sqrt(2))
    f0 = 1/(np.sqrt(2*np.pi)*a)
    got = midplane_jet(k, w, S[None, :], Z[None, :], [f0], r, G)
    rho = mass/(2*np.pi*a*a)**1.5*np.exp(-.5*(r/a)**2)
    drho = -r/a**2*rho
    enclosed = mass*(erf(r/(np.sqrt(2)*a))-np.sqrt(2/np.pi)*r/a*np.exp(-.5*(r/a)**2))
    force = np.divide(G*enclosed, r*r, out=np.zeros_like(r), where=r > 0)
    tangential = np.divide(force, r, out=4*np.pi*G*rho/3, where=r > 0)
    radial = 4*np.pi*G*rho-2*tangential
    dt = np.divide(radial-tangential, r, out=np.zeros_like(r), where=r > 0)
    dr = 4*np.pi*G*drho-2*dt
    H, dH = np.array([radial, tangential, tangential]), np.array([dr, dt, dt])
    np.testing.assert_allclose(got['radial_gradient'], force, atol=3e-14, rtol=1e-11)
    np.testing.assert_allclose(got['hessian_RR_ZZ_PP'], H, atol=5e-13, rtol=1e-10)
    np.testing.assert_allclose(got['radial_derivative_hessian_RR_ZZ_PP'], dH, atol=8e-12, rtol=1e-9)
    np.testing.assert_allclose(got['laplacian'], 4*np.pi*G*rho, atol=2e-14, rtol=1e-10)
    np.testing.assert_allclose(got['radial_gradient_laplacian'], 4*np.pi*G*drho, atol=2e-14, rtol=1e-10)
    np.testing.assert_allclose(got['radial_gradient_hessian_norm'], 2*np.sum(H*dH, axis=0), atol=2e-12, rtol=1e-9)
    potential = np.divide(-G*mass*erf(r/(np.sqrt(2)*a)), r,
                          out=np.full_like(r, -G*mass*np.sqrt(2/np.pi)/a), where=r > 0)
    np.testing.assert_allclose(got['potential'], potential, atol=2e-14, rtol=1e-12)


def test_contact_term_is_not_replaced_by_exact_density():
    k, w = piecewise_gauss([0., .05, .1], 16)
    r = np.array([.5, 1.])
    S, Z = np.exp(-k*k/2)/(2*np.pi), erfcx(k/np.sqrt(2))
    f0 = 1/np.sqrt(2*np.pi)
    got = midplane_jet(k, w, S[None, :], Z[None, :], [f0], r, 1.)
    physical = 4*np.pi*np.exp(-r*r/2)/(2*np.pi)**1.5
    assert np.max(abs(got['laplacian']/physical-1)) > .9
    assert np.array_equal(got['laplacian'], got['hessian_RR_ZZ_PP'].sum(axis=0))


def test_hankel_invalid_inputs_fail():
    with pytest.raises(ValueError):
        sech2_midplane_laplace(-1.)
    with pytest.raises(ValueError):
        piecewise_gauss([0, 1, .5], 16)
    with pytest.raises(ValueError):
        radial_transform([1.], [1.], [[1.]], [-1.])
