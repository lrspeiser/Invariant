"""Independent derivative, variational, symmetry and quadrupole controls."""
import mpmath as mp
import numpy as np
import pytest

from invariant_gravity_extensions.external_quadrupole import (
    quadrupole_integrals,
    saturated_nu_derivative,
)
from invariant_gravity_extensions.fields import PeriodicGrid
from invariant_gravity_extensions.length_screening import (
    LengthScreening,
    anomalous_flux,
    point_external_flux,
    point_monopole_delta,
    point_quadrupole,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


@pytest.mark.parametrize('shape', [.5, 1, 2])
def test_kernel_against_high_precision_derivatives(shape):
    spec = LengthScreening(shape, 1e-4)
    with mp.workdps(90):
        m, eps = mp.mpf(shape), mp.mpf('1e-4')

        def excess(u):
            def S(v):
                return v**mp.mpf('.75')/(1+v**m)**(mp.mpf('.75')/m)
            return mp.mpf(4)/3*(S(u+eps**2)-S(eps**2))

        for value in ['1e-30', '1e-10', '1e-8', '1e-4', '1', '1e12']:
            u = mp.mpf(value)
            K = lambda t: excess(t)/t
            expected = [K(u), u*mp.diff(K, u), u*u*mp.diff(K, u, 2), mp.diff(excess, u)]
            np.testing.assert_allclose(spec.kernel(float(u)), np.array(expected, dtype=float), rtol=2e-10, atol=0)
        assert spec.kernel(0)[0] == pytest.approx(float(mp.diff(excess, 0)), rel=2e-13)


@pytest.mark.parametrize('shape', [.5, 1, 2])
def test_zero_length_monopole_recovers_exact_scalar(shape):
    spec = LengthScreening(shape)
    y = np.logspace(-10, 12, 201)
    scalar = SaturatedActionSpec('qumond', shape=shape, epsilon=spec.epsilon)
    np.testing.assert_allclose(point_monopole_delta(spec, y, 0), scalar.delta_nu(y), rtol=2e-13, atol=0)


def test_cartesian_and_polar_point_flux_agree_in_physical_units():
    r = np.logspace(-3, 3, 37)
    mu = np.linspace(-.93, .91, r.size)
    n = np.array([np.sqrt(1-mu*mu), np.zeros_like(mu), mu])
    tangent = np.array([mu, np.zeros_like(mu), -np.sqrt(1-mu*mu)])
    eta, ell, spec = 1.4, .7, LengthScreening(2)
    p = n/r**2-np.array([0, 0, eta])[:, None]
    H = (np.eye(3)[:, :, None]-3*np.einsum('i...,j...->ij...', n, n))/r**3
    dH2 = -36*n/r**7
    # Scale all derivatives to physical acceleration units, leaving the same
    # coordinate unit. A factor other than one catches missing a0 powers.
    a0 = .31
    generic = anomalous_flux(spec, a0*p, a0*H, a0*a0*dH2, np.zeros_like(p), ell, a0)/a0
    polar = point_external_flux(spec, r, mu, eta, ell)
    expected = polar['radial']*n+polar['polar']*tangent
    np.testing.assert_allclose(generic, expected, rtol=2e-10, atol=3e-15)


def test_bounded_high_gradient_point_limit_keeps_outward_sign():
    r, ell, spec = np.array([.001, .003, .01]), 3, LengthScreening(1)
    eps = spec.epsilon
    C = (4/3)*(1-eps**1.5/(1+eps**2)**.75)
    expected = -2*C*ell**2*r**6*(30*ell**2+r**2)/(6*ell**2+r**2)**3
    actual = point_monopole_delta(spec, 1/r**2, ell)
    assert np.all(actual < 0)
    np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=0)


@pytest.mark.parametrize('length', [.1, 1, 10])
def test_quadrupole_flux_and_action_integrals_agree(length):
    result = point_quadrupole(LengthScreening(1), 1.4, length, quadrature_nodes=256)
    assert result['absolute_agreement'] < 2e-7


@pytest.mark.parametrize('shape', [.5, 1, 2])
def test_zero_length_quadrupole_against_independent_scalar_integral(shape):
    scalar = SaturatedActionSpec('qumond', shape=shape, epsilon=1e-6)
    ref = quadrupole_integrals(1.4, scalar.delta_nu,
                              lambda y: saturated_nu_derivative(scalar, y), nodes=512)
    result = point_quadrupole(LengthScreening(shape), 1.4, 0, quadrature_nodes=256)
    assert result['Q2_flux'] == pytest.approx(-1.5*ref['q_milgrom'], abs=3e-7)
    assert result['absolute_agreement'] < 1e-14


def test_periodic_action_variation_and_internal_momentum():
    grid = PeriodicGrid(33, 2*np.pi)
    x, y, z = grid.coordinates()
    psi = .15*np.cos(x)+.08*np.sin(y+z)+.04*np.sin(x-2*z)
    background = np.array([.7, .4, -.3])[:, None, None, None]
    gradient, H = grid.gradient(psi)+background, grid.hessian(psi)
    h2 = (H*H).sum(axis=(0, 1))
    spec, ell = LengthScreening(1), .3
    flux = anomalous_flux(spec, gradient, H, grid.gradient(h2), grid.gradient(grid.laplacian(psi)), ell)
    anomaly = grid.poisson(grid.divergence(flux))
    acceleration = -grid.gradient(anomaly)
    rho = grid.laplacian(psi)  # four_pi_G=1; periodic density contrast control
    total = (rho*acceleration).mean(axis=(1, 2, 3))
    normalizer = np.mean(abs(rho)*np.linalg.norm(acceleration, axis=0))
    assert np.linalg.norm(total)/normalizer < 1e-10
    delta = np.cos(x+y)+.3*np.sin(2*z-x)
    expected = -2*np.mean(delta*grid.divergence(gradient+flux))

    def energy(amount):
        perturbed = psi+amount*delta
        grad, Hnew = grid.gradient(perturbed)+background, grid.hessian(perturbed)
        return np.mean(spec.value(np.sum(grad*grad, axis=0), ell**2*np.sum(Hnew*Hnew, axis=(0, 1))))

    measured = (energy(1e-5)-energy(-1e-5))/(2e-5)
    assert measured == pytest.approx(expected, rel=2e-8, abs=2e-11)


def test_zero_field_and_constant_background_are_regular():
    spec = LengthScreening()
    zero = np.zeros((3, 2))
    H = np.zeros((3, 3, 2))
    np.testing.assert_array_equal(anomalous_flux(spec, zero, H, zero, zero, 1), 0)
    p = np.ones_like(zero)
    expected = spec.excess_derivatives(np.sum(p*p, axis=0))[0]*p
    np.testing.assert_allclose(anomalous_flux(spec, p, H, zero, zero, 100), expected)
    assert point_quadrupole(spec, 0, 1)['Q2_flux'] == 0


def test_unsupported_or_nonfinite_inputs_fail_explicitly():
    with pytest.raises(ValueError):
        LengthScreening(3)
    with pytest.raises(ValueError):
        LengthScreening().kernel(-1)
    with pytest.raises(ValueError):
        point_external_flux(LengthScreening(), 1, 1.1, 1, 1)
    with pytest.raises(ValueError):
        anomalous_flux(LengthScreening(), np.zeros(2), np.zeros((2, 2)), np.zeros(2), np.zeros(2), 1)
