"""Spherical, smooth-source pressure predictions for the full length action."""
from __future__ import annotations

import numpy as np

from .cluster_pressure import (
    KPC,
    MU_E,
    PRESSURE_SI_PER_KEV_CM3,
    PROTON_MASS,
    boundary_residual_covariance,
    covariance_loss,
    integrate_electron_pressure,
    pressure_indices,
)
from .length_screening import LengthScreening
from .smooth_spherical_source import cluster_source_fields, spherical_length_anomaly


def array_packet(packet):
    """Restore the existing derived packet's numeric columns, without raw reads."""
    keys = ['density_radius_kpc', 'ne_cm3', 'ne_low_error', 'ne_high_error',
            'pressure_radius_kpc', 'pressure', 'pressure_error', 'covariance', 'native_scaled_covariance']
    answer = {**packet, **{key: np.asarray(packet[key], float) for key in keys}}
    if packet['stellar'] is not None:
        answer['stellar'] = {key: np.asarray(value, float) for key, value in packet['stellar'].items()}
    return answer


def pressure_context(packet, sources, nuisance, *, nodes=4097):
    """Model-independent source and pressure boundary on a declared radial grid.

    Only the first target through the boundary is needed for the inward HSE
    integral. Retain original gas and stellar knots in that interval. The
    response values never choose the grid or smooth-source parameters.
    """
    if type(nodes) is not int or nodes < 129:
        raise ValueError('resolved radial integration grid required')
    ids, anchor, dispositions = pressure_indices(packet)
    targets = packet['pressure_radius_kpc'][ids]*KPC
    outer = packet['pressure_radius_kpc'][anchor]*KPC
    knots = [sources['source_radius_m'], targets, [outer]]
    if packet['stellar'] is not None:
        knots.append(packet['stellar']['radius_kpc']*KPC)
    support = np.concatenate(knots)
    support = support[(support >= targets[0]) & (support <= outer)]
    nominal_r = np.unique(np.r_[np.geomspace(targets[0], outer, nodes), support])
    fields = cluster_source_fields(sources, nominal_r, nuisance)
    distance = nuisance.get('distance_scale', 1.)
    calibration = nuisance.get('pressure_calibration', 1.)
    fout = nuisance.get('outer_nonthermal_fraction', .15)
    if not np.isfinite([distance, calibration, fout]).all() or distance <= 0 or calibration <= 0 or not 0 <= fout < 1:
        raise ValueError('physical distance, pressure calibration and nonthermal fraction required')
    return {'fields': fields, 'nominal_radius_m': nominal_r, 'target_locations': np.searchsorted(nominal_r, targets),
            'indices': ids, 'anchor': anchor, 'dispositions': dispositions,
            'electron_density_m3': fields['gas_density']/(MU_E*PROTON_MASS),
            'fraction': fout*nominal_r/outer,
            'boundary_si': packet['pressure'][anchor]*calibration/distance*PRESSURE_SI_PER_KEV_CM3,
            'observed': packet['pressure'][ids]*calibration/distance, 'pressure_scale': calibration/distance}


def predict_from_context(context, model):
    fields = context['fields']
    gbar = fields['gbar']
    if model['family'] == 'newtonian':
        acceleration = gbar.copy()
    elif model['family'] == 'rar_comparator':
        acceleration = gbar/(-np.expm1(-np.sqrt(gbar/model['a0_m_s2'])))
    elif model['family'] == 'length_screening':
        spec = LengthScreening(model['shape'], model['epsilon'])
        delta = spherical_length_anomaly(spec, fields['radius_m'], gbar, fields['gbar_first'], fields['gbar_second'],
                                         model['length_pc']*KPC/1000, model['a0_m_s2'])
        acceleration = gbar+delta
    else:
        raise NotImplementedError('no registered spherical response for this family')
    bad = ~np.isfinite(acceleration) | (acceleration <= 0)
    if np.any(bad):
        return {'status': 'NONPOSITIVE_OR_NONFINITE_FORCE_RETAINED_UNSCORED',
                'bad_radius_m': fields['radius_m'][bad],
                'bad_force_m_s2': [float(g) if np.isfinite(g) else None for g in acceleration[bad]]}
    p, k = integrate_electron_pressure(fields['radius_m'], context['electron_density_m3'], acceleration,
                                      context['fraction'], context['boundary_si'])
    loc = context['target_locations']
    base = {key: context[key] for key in ['indices', 'anchor', 'dispositions', 'observed', 'pressure_scale']}
    return {**base, 'status': 'PREDICTED_PENDING_NUMERICAL_ADMISSION',
            'prediction': p[loc]/PRESSURE_SI_PER_KEV_CM3, 'boundary_coefficients': k[loc],
            'source_acceleration_m_s2': gbar[loc], 'predicted_acceleration_m_s2': acceleration[loc],
            'minimum_integration_force_m_s2': float(acceleration.min()),
            'maximum_fractional_force_change': float(np.max(abs(acceleration/gbar-1))),
            'integration_nodes': len(p)}


def score_prediction(packet, answer):
    """Called only after the whole card passes its predeclared numerical gates."""
    residual = answer['prediction']-answer['observed']
    log_ratio = np.log10(answer['prediction']/answer['observed'])
    choices = {'transferred_correlation': packet['covariance'], 'native_scaled': packet['native_scaled_covariance'],
               'diagonal_quoted': np.diag(packet['pressure_error']**2)}
    covariance_scores, standardized = {}, {}
    for name, covariance in choices.items():
        transformed = boundary_residual_covariance(covariance*answer['pressure_scale']**2, answer['indices'],
                                                   answer['anchor'], answer['boundary_coefficients'])
        covariance_scores[name] = covariance_loss(residual, transformed)
        standardized[name] = residual/np.sqrt(np.diag(transformed))
    return {'cluster': packet['cluster'], 'stellar_profile_present': packet['stellar'] is not None,
            **answer, 'status': 'NUMERICALLY_ADMITTED_QUALITY_LIMITED', 'residual': residual, 'log10_ratio': log_ratio,
            'mse_log10_ratio': float(np.mean(log_ratio**2)), 'mean_absolute_dex': float(np.mean(abs(log_ratio))),
            'median_pressure_ratio': float(np.median(answer['prediction']/answer['observed'])),
            'whitened_mean_squared_residual': covariance_scores,
            'marginal_standardized_residual_conditional_not_significance': standardized}
