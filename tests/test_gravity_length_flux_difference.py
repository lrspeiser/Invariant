"""Independent high-precision and covariance checks of the flux difference."""
import runpy
from pathlib import Path

import numpy as np
import pytest

from invariant_gravity_extensions.length_flux_difference import length_flux_difference
from invariant_gravity_extensions.length_screening import LengthScreening

PRECISE = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/diagnose_gravity_length_cancellation.py'))['precise']


def fields():
    return (np.array([.696, 1.672, 0.]), np.array([[2.32, .48, 0.], [.48, 4.18, 0.], [0., 0., 2.32]]),
            np.array([13.104, 18.5344, 0.]), np.array([1.2, 3.2, 0.]))


@pytest.mark.parametrize('shape', [.5, 1., 2.])
@pytest.mark.parametrize('gradient_scale', [1., 1e-8])
@pytest.mark.parametrize('length', [1e-8, 1e-4, .01, 1., 100.])
def test_difference_against_independent_80_digit_action(shape, gradient_scale, length):
    p, H, dH2, dlap = fields()
    p *= gradient_scale
    spec = LengthScreening(shape)
    expected = PRECISE(shape, spec.epsilon, p, H, dH2, dlap, length)
    actual = length_flux_difference(spec, p, H, dH2, dlap, length)
    assert np.linalg.norm(actual-expected)/np.linalg.norm(expected) < 1e-9


def test_batch_rotation_and_physical_acceleration_scaling():
    p, H, dH2, dlap = fields()
    q, _ = np.linalg.qr(np.random.default_rng(539).normal(size=(3,3)))
    spec = LengthScreening(1.)
    expected = length_flux_difference(spec, p, H, dH2, dlap, 1e-5)
    a0 = 3700.
    rotated = length_flux_difference(spec, a0*q@p, a0*q@H@q.T, a0*a0*q@dH2, a0*q@dlap, 1e-5, a0)
    np.testing.assert_allclose(rotated, a0*q@expected, rtol=1e-11, atol=1e-24)
    batch = length_flux_difference(spec, np.stack([p,p],axis=-1), np.stack([H,H],axis=-1),
        np.stack([dH2,dH2],axis=-1), np.stack([dlap,dlap],axis=-1), 1e-5)
    np.testing.assert_allclose(batch, np.stack([expected,expected],axis=-1), rtol=1e-12, atol=1e-24)


def test_zero_length_and_stationary_point():
    p, H, dH2, dlap = fields()
    spec = LengthScreening(.5)
    np.testing.assert_array_equal(length_flux_difference(spec,p,H,dH2,dlap,0.), np.zeros(3))
    np.testing.assert_array_equal(length_flux_difference(spec,np.zeros(3),H,dH2,dlap,.01), np.zeros(3))
    np.testing.assert_array_equal(length_flux_difference(spec,np.zeros(3),np.zeros((3,3)),dH2,dlap,.01), np.zeros(3))
