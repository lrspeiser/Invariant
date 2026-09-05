"""Active-domain and exact-provider checks for matched potential assembly."""
import numpy as np
import pytest

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.matched_axisymmetric import matched_grid


def test_matched_grid_exact_plateaus_and_inactive_provider():
    moment = {'scale': 1., 'support_radius': .1, 'maximum_order': 0, 'scaled_multipole_moments': [1.]}
    near_provider = ExteriorMomentField(moment, 1.)
    far_provider = ExteriorMomentField(moment, 1., minimum_radius=2.)
    R, z = np.array([.5, 1., 2., 3., 4., 6.]), np.array([-2., 0., 2.])
    RR, ZZ = np.meshgrid(R, z, indexing='ij')
    near = near_provider.fields(RR, ZZ)
    near['radius'], near['height'] = R, z
    baseline = {k: np.array(v, copy=True) for k, v in near.items()}
    outside = np.hypot(RR, ZZ) >= 4.
    for k in near.keys()-{'radius', 'height'}:
        near[k][..., outside] = np.nan
    value = matched_grid(near, far_provider, R, z, inner=2., outer=4.)
    for k in value.keys()-{'radius', 'height'}:
        np.testing.assert_allclose(value[k], baseline[k], atol=3e-14, rtol=2e-14, err_msg=k)
    # A potential-zero mismatch must still create product terms in transition.
    baseline['potential'] = baseline['potential']+.2
    value = matched_grid(baseline, far_provider, R, z, inner=2., outer=4.)
    assert abs(value['laplacian'][3, 1]) > .01
    np.testing.assert_array_equal(value['potential'][outside], near_provider.fields(RR[outside], ZZ[outside])['potential'])


def test_matched_grid_rejects_invalid_active_domain():
    moment = {'scale': 1., 'support_radius': .1, 'maximum_order': 0, 'scaled_multipole_moments': [1.]}
    provider = ExteriorMomentField(moment, 1., minimum_radius=2.)
    R, z = np.array([2.]), np.array([0.])
    near = provider.fields(R[:, None], z[None, :])
    with pytest.raises(ValueError, match='admitted'):
        matched_grid(near, provider, R, z, inner=1., outer=4.)
    near['potential'][0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        matched_grid(near, provider, R, z, inner=2., outer=4.)
