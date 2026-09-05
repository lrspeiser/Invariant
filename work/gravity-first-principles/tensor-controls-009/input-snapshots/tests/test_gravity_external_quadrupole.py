import numpy as np
import pytest

from invariant_gravity_extensions.external_quadrupole import (
    acceleration_tensor,
    newtonian_external_ratio,
    q_to_Q2,
    quadrupole_integrals,
    reference_nu_delta,
    reference_nu_derivative,
    saturated_nu_derivative,
    scalar_quadrupole,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


@pytest.mark.parametrize("alpha,eta,rounded,tolerance", [
    (2, 1, .088, .0005), (2, 1.5, .11, .005), (2, 2, .12, .005), (2, 3, .12, .005),
    (4, 1, .074, .0005), (4, 1.5, .057, .0005), (4, 2, .038, .0005), (4, 3, .018, .0005),
    (8, 1, .061, .0005), (8, 1.5, .022, .0005), (8, 2, .0086, .00005), (8, 3, .0021, .00005),
])
def test_published_reference_magnitudes_and_sign(alpha, eta, rounded, tolerance):
    # Hees et al. 1510.01369, table B1: positive displayed magnitudes. Equation
    # 10 and the source-Hessian derivation fix the signed q<0, Q2>0 convention.
    delta = lambda y: reference_nu_delta(y, alpha)
    derivative = lambda y: reference_nu_derivative(y, alpha)
    en = newtonian_external_ratio(eta, delta)
    row = quadrupole_integrals(en, delta, derivative, nodes=256)
    assert row["q_milgrom"] < 0
    assert abs(-row["q_milgrom"]-rounded) < tolerance
    assert row["absolute_agreement"] < 3e-6


@pytest.mark.parametrize("shape", [.5, 1, 2])
def test_action_derivative_and_independent_source_agree(shape):
    spec = SaturatedActionSpec("qumond", shape=shape)
    y = np.logspace(-3, 3, 100)
    h = 1e-5
    finite = (spec.delta_nu(y*(1+h))-spec.delta_nu(y*(1-h)))/(2*h*y)
    np.testing.assert_allclose(saturated_nu_derivative(spec, y), finite, rtol=1e-8, atol=1e-15)
    low = scalar_quadrupole(spec, 1.9e-10, 1.2e-10, 1.32712440041279419e20, nodes=128)
    high = scalar_quadrupole(spec, 1.9e-10, 1.2e-10, 1.32712440041279419e20, nodes=512)
    assert abs(high["q_milgrom"]-low["q_milgrom"]) < 1e-7
    assert high["absolute_agreement"] < 3e-7
    assert high["full_solar_system_pass"] is False


def test_no_external_field_and_newtonian_response_give_zero():
    spec = SaturatedActionSpec("qumond")
    assert scalar_quadrupole(spec, 0, 1e-10, 1e20)["Q2_s_minus2"] == 0
    row = quadrupole_integrals(2, np.zeros_like, np.zeros_like)
    assert row["q_milgrom"] == row["q_source_hessian"] == 0


def test_external_mapping_is_not_silently_identified_with_physical_field():
    spec = SaturatedActionSpec("qumond", shape=1)
    eta = 1.9/1.2
    en = newtonian_external_ratio(eta, spec.delta_nu)
    assert en*(1+spec.delta_nu(en)) == pytest.approx(eta, rel=1e-12)
    assert eta-en > .1


def test_tensor_orientation_sign_trace_and_dimensions():
    tensor = acceleration_tensor(3e-27, np.array([0, 0, 1]))
    np.testing.assert_allclose(np.diag(tensor), [-1e-27, -1e-27, 2e-27], atol=1e-42)
    assert abs(np.trace(tensor)) < 1e-42
    np.testing.assert_allclose(tensor, acceleration_tensor(3e-27, np.array([0, 0, -1])), atol=0)
    assert q_to_Q2(-.1, 4e-10, 4e20) / q_to_Q2(-.1, 1e-10, 1e20) == pytest.approx(4)


def test_multifield_card_cannot_borrow_scalar_external_result():
    with pytest.raises(NotImplementedError, match="own solver"):
        scalar_quadrupole(SaturatedActionSpec("trimond_alignment", .75, 2), 2e-10, 1e-10, 1e20)


@pytest.mark.parametrize("eta", [-1, np.nan, np.inf])
def test_invalid_external_field_rejected(eta):
    with pytest.raises(ValueError):
        newtonian_external_ratio(eta, np.zeros_like)
