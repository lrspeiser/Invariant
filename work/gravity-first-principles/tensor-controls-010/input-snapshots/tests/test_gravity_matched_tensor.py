"""Gauge restoration and active-domain dispatch for the matched interpolator."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential
from invariant_gravity_extensions.potential_join import pack_cartesian
from invariant_gravity_extensions.tensor_potential import C3TensorPotential


class QuadraticExterior:
    minimum_radius = 2.

    def fields(self, radius, height):
        R, z = np.broadcast_arrays(radius, height)
        assert np.all(np.hypot(R, z) >= self.minimum_radius)
        return exact(R, z)


def exact(R, z):
    R, z = np.broadcast_arrays(R, z)
    H = np.eye(3).reshape((3, 3)+(1,)*R.ndim)*np.ones(R.shape)*2
    return pack_cartesian(17+R*R+z*z, np.array([2*R, 2*z, np.zeros_like(R)]), H,
        np.zeros((3, 3, 3)+R.shape), R, z)


def provider():
    grid = np.array([0., 1., 2., 4.])
    R, z = np.meshgrid(grid, grid, indexing='ij')
    data = np.zeros((4, 4, 4, 4))
    data[0, 0], data[1, 0], data[0, 1] = 17+R*R+z*z, 2*R, 2*z
    data[2, 0], data[0, 2] = 2., 2.
    return MatchedTensorPotential(grid, grid, data, QuadraticExterior(), inner=2., outer=4.)


def test_one_potential_gauge_through_join_and_beyond_tensor_domain():
    source = provider()
    R = np.array([0., 1., 2.5, 3., 4., 5., 0.])
    z = np.array([0., .5, -1., 0., 1., .7, -5.])
    actual, expected = source.fields(R, z), exact(R, z)
    for key in actual:
        np.testing.assert_allclose(actual[key], expected[key], rtol=2e-12, atol=3e-11, err_msg=key)
    assert source.gauge == 17.
    assert source.fields(0., 0.)['potential'] == 17.
    assert source.fields([], [])['potential'].shape == (0,)


def test_matched_tensor_rejects_invalid_coordinates():
    source = provider()
    for R, z in [(-1., 0.), (0., np.nan), (np.inf, 0.)]:
        with pytest.raises(ValueError, match='finite'):
            source.fields(R, z)


def test_independent_derivatives_respect_c3_interface_and_cancel_center_gauge():
    audit = runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/audit_gravity_matched_tensor.py'))
    grid = np.array([0.,1.,2.])
    u = np.maximum(grid-1,0)[:,None]
    data = np.zeros((4,4,3,2))
    data[0,0],data[1,0],data[2,0],data[3,0] = 17+u**4,4*u**3,12*u*u,24*u
    source = C3TensorPotential(grid,[0.,1.],data)
    wrapped = SimpleNamespace(fields=source.fields,near=source,outer=2.)
    R,z = np.array([[1.,0.]]),np.array([[.2,0.]])
    scale = {key:np.ones_like(R) for key in ['force','hessian','third']}
    central = audit['derivative_check'](wrapped,R,z,.001,scale)
    assert central['third'][0,0] > 1e-4
    assert central['force'][0,1] == 0.
    for stencil in ['left','right']:
        errors = audit['derivative_check'](wrapped,R,z,.001,scale,radial_stencil=stencil)
        assert max(np.max(v) for v in errors.values()) < 1e-7
