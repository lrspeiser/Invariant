"""Analytic flux/source controls and variational checks, without observations."""
import numpy as np
import pytest

from invariant_gravity_extensions.external_multifield import (
    FluxPoissonSolver,
    beta_zero_source,
    physical_auxiliary_flux,
    solve_external_auxiliary,
)
from invariant_gravity_extensions.isolated_axisymmetric import MultipoleGrid, solve_poisson
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def solver():
    return FluxPoissonSolver(MultipoleGrid(1e-4, 100, 513, 64, 24))


def test_manufactured_gradient_flux_recovers_potential_gradient():
    s = solver()
    r, mu = s.radius[:, None], s.mu
    e = np.exp(-r*r)
    # Phi=exp(-r^2)*(1+.3*z); exact analytic smooth field.
    flux = np.array([e*(-2*r*(1+.3*r*mu)+.3*mu), -.3*e*s.sine+np.zeros_like(r)])
    recovered = s.gradient(s.solve(flux))
    select = (s.radius > .03) & (s.radius < 3)
    assert np.linalg.norm((recovered-flux)[:, select])/np.linalg.norm(flux[:, select]) < 1e-5


def test_quadrupole_of_vanishing_inner_radial_flux():
    s = solver()
    r, mu = s.radius[:, None], s.mu
    flux = np.array([r**3*np.exp(-r*r)*(3*mu*mu-1)/2, np.zeros((len(r), mu.size))])
    # div(J)_l2=(5*r^2-2*r^4)*exp(-r^2). Q2=3/5 integral S_l2/r dr=.9.
    assert s.quadrupole(flux)["Q2_volume"] == pytest.approx(.9, abs=1e-6)


def test_linear_inner_flux_requires_quadrupole_surface_term():
    s = solver()
    r, mu = s.radius[:, None], s.mu
    e = np.exp(-r*r)
    flux = np.array([(2*r-2*r**3)*e*(3*mu*mu-1)/2, -3*r*e*mu*s.sine])
    q2 = s.quadrupole(flux)
    # Phi=r^2*P2*exp(-r^2). Its Q2=-3; dropping the inner sheet gives -1.8.
    assert q2["Q2_volume"] == pytest.approx(-3, abs=2e-6)
    assert q2["Q2_bulk"] == pytest.approx(-1.8, abs=2e-6)


def test_beta_zero_against_analytic_source_green_solution():
    s = solver()
    aux = solve_external_auxiliary(s, 1.4, 1, 0, 1)
    independent = solve_poisson(s.grid, lambda R, z: beta_zero_source(1.4, 1, 1, R, z))
    R, z = np.array([.1, .3, .8, 1.4, 3]), np.array([.03, -.1, .2, -.5, 1])
    a = aux.potential.evaluate(R, z)["acceleration"]
    b = independent.evaluate(R, z)["acceleration"]
    assert np.linalg.norm(a-b)/np.linalg.norm(b) < 1e-4


def test_auxiliary_linear_scaling_and_physical_quadratic_scaling():
    s = solver()
    a = solve_external_auxiliary(s, 1.4, 1, 2, 1)
    b = solve_external_auxiliary(s, 1.4, -.5, 2, 1)
    np.testing.assert_allclose(b.q, -.5*a.q, atol=1e-12, rtol=1e-10)
    fa = physical_auxiliary_flux(a.p, a.q, 1, 2, 1)
    fb = physical_auxiliary_flux(b.p, b.q, -.5, 2, 1)
    np.testing.assert_allclose(fb, .25*fa, atol=1e-12, rtol=1e-10)


def test_physical_flux_matches_independent_symbolic_action_variation():
    rng = np.random.default_rng(1973)
    p, q = rng.normal(size=(2, 3, 7)), rng.normal(size=(2, 3, 7))
    x, y, z = (p*p).sum(axis=0), (q*q).sum(axis=0), 2*(p*q).sum(axis=0)
    spec = SaturatedActionSpec("trimond_alignment", mixing=.75, beta=2, power=2)
    fx, _, fz, _ = spec.partials(x, y, z)
    expected = (fx-1-spec.delta_nu(np.sqrt(x)))*p+fz*q
    np.testing.assert_allclose(physical_auxiliary_flux(p, q, .75, 2, 2), expected, atol=1e-13, rtol=1e-12)


def test_collinear_physical_flux_zero_and_zero_mixing_solution():
    s = solver()
    a = solve_external_auxiliary(s, 1.4, 0, 2, 1)
    np.testing.assert_array_equal(a.q, 0)
    p = a.p
    q = .75/(1+(p*p).sum(axis=0))*p
    assert np.max(abs(physical_auxiliary_flux(p, q, .75, 2, 1))) < 1e-13


def test_explicit_unsupported_and_nonconvergence_errors():
    with pytest.raises(NotImplementedError):
        FluxPoissonSolver(MultipoleGrid(1e-4, 100, 129, 32, 8, plane_scale=.5))
    with pytest.raises(ValueError):
        solve_external_auxiliary(solver(), 1, 1, 3, 1)
    with pytest.raises(RuntimeError):
        solve_external_auxiliary(solver(), 1.4, 1, 2, 1, max_iterations=2)
