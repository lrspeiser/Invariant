"""Analytic, mutation and independent-integrator tests of monopole diagnostics."""
import numpy as np
import pytest

from invariant_gravity_extensions.actions import ActionSpec
from invariant_gravity_extensions.local_limits import (
    Orbit,
    baseline_delta_nu,
    binet_precession,
    logarithmic_precession,
    monopole_delta_nu,
    perihelion_first_order,
    power_tail,
)


@pytest.mark.parametrize("e", [0.01, 0.20563593, 0.7])
def test_gauss_integral_matches_independent_log_formula(e):
    orbit = Orbit(1, e, 1)
    for a0 in (1e-12, 1e-8):
        predicted = perihelion_first_order(orbit, a0, baseline_delta_nu)
        assert predicted == pytest.approx(logarithmic_precession(orbit, a0), rel=1e-11)
        assert predicted < 0  # attractive logarithmic tail causes retrograde precession


def test_constant_gm_change_is_not_an_anomalous_precession():
    orbit = Orbit(1, 0.2, 1)
    assert perihelion_first_order(orbit, 1e-10, lambda y: np.full_like(y, 0.3)) == 0


def test_constant_radial_acceleration_has_known_gauss_average():
    orbit = Orbit(1, 0.4, 1)
    a0 = 1e-8
    expected = -2 * np.pi * a0 * np.sqrt(1 - orbit.eccentricity**2)
    result = perihelion_first_order(orbit, a0, lambda y: power_tail(y, 1))
    assert result == pytest.approx(expected, rel=1e-12)


@pytest.mark.parametrize("epsilon", [1e-4, 0.03])
def test_stable_monopole_matches_action_derivative(epsilon):
    y = np.logspace(-3, 4, 100)
    derivative = ActionSpec("qumond", epsilon=epsilon).partials(y**2)[0]
    np.testing.assert_allclose(baseline_delta_nu(y, epsilon), derivative - 1,
                               rtol=1e-12, atol=1e-14)


def test_tiny_high_acceleration_tail_is_not_rounded_to_zero():
    assert baseline_delta_nu(1e40) == pytest.approx(1e-20, rel=1e-14, abs=0)
    assert baseline_delta_nu(1e300) == pytest.approx(1e-150, rel=1e-14, abs=0)
    assert baseline_delta_nu(1e-300) == pytest.approx(4 / 3 / np.sqrt(1e-4), rel=1e-14)


def test_exact_orbit_converges_to_first_order_with_expected_error():
    orbit = Orbit(1, 0.2, 1)
    errors = []
    for a0 in (1e-6, 2.5e-7):
        approx = logarithmic_precession(orbit, a0)
        direct = binet_precession(orbit, a0)
        assert direct < 0
        errors.append(abs(direct - approx))
    # Halving perturbation gives a quadratic first-order truncation error.
    assert 3.8 < errors[0] / errors[1] < 4.2


def test_no_tail_gives_no_precession_and_sign_mutation_is_visible():
    orbit = Orbit(1, 0.2, 1)
    assert perihelion_first_order(orbit, 1e-10, np.zeros_like) == 0
    correct = perihelion_first_order(orbit, 1e-10, baseline_delta_nu)
    wrong = perihelion_first_order(orbit, 1e-10, lambda y: -baseline_delta_nu(y))
    assert wrong == -correct > 0


def test_length_sensitive_family_cannot_borrow_qumond_monopole():
    with pytest.raises(NotImplementedError, match="higher-derivative"):
        monopole_delta_nu(ActionSpec("gqumond_length", length=0.25), np.array([1e8]))


@pytest.mark.parametrize("e", [-0.1, 0, 1, np.nan])
def test_invalid_orbit_is_refused(e):
    with pytest.raises(ValueError):
        Orbit(1, e, 1)


def test_quadrature_and_input_failures_are_refused():
    orbit = Orbit(1, 0.2, 1)
    with pytest.raises(ValueError):
        perihelion_first_order(orbit, 0, baseline_delta_nu)
    with pytest.raises(ValueError):
        perihelion_first_order(orbit, 1, baseline_delta_nu, nodes=8)
    with pytest.raises(ValueError):
        perihelion_first_order(orbit, 1, lambda y: y * np.nan)
