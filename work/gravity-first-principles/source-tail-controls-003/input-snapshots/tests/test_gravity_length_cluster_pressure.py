"""End-to-end analytic and units controls for the new pressure adapter."""
import numpy as np
import pytest

from invariant_gravity_extensions.cluster_pressure import KPC, MU, MU_E, PROTON_MASS, G
from invariant_gravity_extensions.length_cluster_pressure import (
    array_packet,
    predict_from_context,
    pressure_context,
    score_prediction,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec
from invariant_gravity_extensions.smooth_spherical_source import (
    build_cluster_sources,
    cluster_source_fields,
)


def packet(stars=False):
    r = np.arange(1., 9.)
    return array_packet({'cluster': 'synthetic', 'density_radius_kpc': [1., 1000.], 'ne_cm3': [1., 1.],
                         'ne_low_error': [.1, .1], 'ne_high_error': [.1, .1],
                         'pressure_radius_kpc': r, 'pressure': r**-1, 'pressure_error': r**-1/10,
                         'covariance': np.diag(r**-2/100), 'native_scaled_covariance': np.diag(r**-2/100),
                         'stellar': {'radius_kpc': [1., 1000.], 'mass_msun': [1e8, 1e12]} if stars else None})


def test_uniform_sphere_pressure_matches_analytic_integral():
    data = packet()
    width = .01
    source = build_cluster_sources(data, width=width, nodes=4097)
    nuisance = {'outer_nonthermal_fraction': 0., 'missing_stellar_gas_ratio': .1,
                'distance_scale': 1.1, 'pressure_calibration': .9}
    context = pressure_context(data, source, nuisance)
    answer = predict_from_context(context, {'family': 'newtonian'})
    # Gaussian convolution multiplies D proportional to exp(3t) by exp(9 sigma^2/2).
    ne = 1e6*np.exp(4.5*width**2)*1.1**-.5
    rho = ne*MU_E*PROTON_MASS
    radii = data['pressure_radius_kpc'][answer['indices']]*1.1*KPC
    anchor = data['pressure_radius_kpc'][answer['anchor']]*1.1*KPC
    pscale = 1.602176634e-10
    expected = (context['boundary_si']+MU*PROTON_MASS*ne*(4*np.pi*G*rho*1.1/3)*(anchor**2-radii**2)/2)/pscale
    np.testing.assert_allclose(answer['prediction'], expected, rtol=2e-10)
    assert answer['anchor'] not in answer['indices']
    scored = score_prediction(data, answer)
    assert np.isfinite(list(scored['whitened_mean_squared_residual'].values())).all()


def test_zero_length_recovers_saturated_scalar_on_same_joint_source():
    data = packet(stars=True)
    sources = build_cluster_sources(data, width=.02, nodes=4097)
    context = pressure_context(data, sources, {})
    for shape in [.5, 1., 2.]:
        model = {'family': 'length_screening', 'shape': shape, 'epsilon': 1e-6, 'length_pc': 0., 'a0_m_s2': 1.2e-10}
        answer = predict_from_context(context, model)
        g = answer['source_acceleration_m_s2']
        expected = g*(1+SaturatedActionSpec('qumond', shape=shape, epsilon=1e-6).delta_nu(g/1.2e-10))
        np.testing.assert_allclose(answer['predicted_acceleration_m_s2'], expected, rtol=3e-14)


def test_gas_and_stellar_distance_scaling_apply_to_same_source_derivatives():
    data = packet(stars=True)
    source = build_cluster_sources(data, width=.02, nodes=4097)
    radii = np.geomspace(2, 500, 15)*KPC
    d, scale = 1.1, 1.3
    result = cluster_source_fields(source, radii, {'distance_scale': d, 'stellar_scale': scale})
    gas = source['gas'].evaluate(radii)
    star = source['stellar'].evaluate(radii)
    for key, gpower, spower in [('mass', 2.5, 2), ('density', -.5, -1), ('density_gradient', -1.5, -2)]:
        np.testing.assert_allclose(result[key], gas[key]*d**gpower+star[key]*scale*d**spower, rtol=1e-14)
    np.testing.assert_allclose(result['gbar'], G*result['mass']/(d*radii)**2, rtol=1e-14)


def test_invalid_force_is_retained_and_invalid_nuisance_errors():
    data = packet()
    source = build_cluster_sources(data, width=.02, nodes=4097)
    context = pressure_context(data, source, {})
    context['fields']['gbar'][3] = -1.
    answer = predict_from_context(context, {'family': 'newtonian'})
    assert answer['status'].endswith('UNSCORED')
    assert 'prediction' not in answer
    assert answer['bad_force_m_s2'] == [-1.]
    with pytest.raises(ValueError):
        pressure_context(data, source, {'outer_nonthermal_fraction': 1.})
