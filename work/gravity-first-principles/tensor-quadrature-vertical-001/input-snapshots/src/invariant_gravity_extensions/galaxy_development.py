"""Conditional scalar galaxy predictions and explicitly descriptive diagnostics."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .isolated_axisymmetric import MultipoleGrid, solve_isolated
from .reconstructed_axisymmetric import ReconstructedNewtonianSource, SurfaceDensityDisk
from .saturated_actions import SaturatedActionSpec

KPC_M = 3.085677581491367e19
SI_ACCELERATION_TO_KMS2_KPC = KPC_M / 1e6


class GridCachedSource:
    """Reuse derivatives only on the exact coordinates on which they were built."""

    def __init__(self, source, grid):
        self.name = source.name
        self.source = source
        radius, mu, _ = grid.nodes()
        self.R = radius[:, None]*np.sqrt(1-mu*mu)
        self.z = radius[:, None]*mu
        self.values = source.fields(self.R, self.z)

    def fields(self, R, z):
        if np.array_equal(R, self.R) and np.array_equal(z, self.z):
            return self.values
        return self.source.fields(R, z)


def source_disks(profile, variant, *, outer_kpc=36.0):
    """Named, one-at-a-time source variations, without target-based fitting."""
    radii = np.asarray(profile['radius_kpc'])
    height = profile['stellar_half_mass_radius_kpc']/(1.678*7.3)
    factors = {'stellar_fixed': variant.get('stellar_factor', 1.0),
               'hi': variant.get('hi_factor', 1.0), 'co': variant.get('co_factor', 1.0)}
    disks = {}
    for name in ['stellar_fixed', 'hi', 'co']:
        surface = np.asarray(profile['components'][name]['surface_density_msun_pc2'])*1e6*factors[name]
        disks[name] = SurfaceDensityDisk(radii, surface,
                                        (height if name == 'stellar_fixed' else .2)*variant.get('height_factor', 1.0),
                                        variant.get('outer_kpc', outer_kpc), 2.0)
    return disks


def candidate_id(shape, a0):
    return f'SAT_m{shape:g}_a0_{a0:.1e}'


def field_family(profile, variant, definition, candidate_config, radii, G):
    """One joint density and isolated Newtonian field, then every fixed candidate."""
    disks = source_disks(profile, variant)
    grid = MultipoleGrid(**definition, plane_scale=min(d.height for d in disks.values()))
    source = ReconstructedNewtonianSource.build(variant['id'],
                lambda R, z: 4*np.pi*G*sum(d.density(R, z) for d in disks.values()), grid)
    cached = GridCachedSource(source, grid)
    g_n = source.fields(radii, 0)['gradient'][0]
    rar_a0 = candidate_config['RAR_a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC
    predictions = {'NEWTON_BARYONS': g_n,
                   'RAR_2016_ALGEBRAIC': g_n/(-np.expm1(-np.sqrt(np.maximum(g_n, 0)/rar_a0)))}
    vertical = {}
    for shape in candidate_config['shapes']:
        spec = SaturatedActionSpec('qumond', shape=shape, epsilon=candidate_config['epsilon'])
        for a0 in candidate_config['a0_m_s2']:
            solution = solve_isolated((cached,), spec, a0*SI_ACCELERATION_TO_KMS2_KPC, grid)
            fields = solution.evaluate(radii, np.zeros_like(radii))
            name = candidate_id(shape, a0)
            predictions[name] = -fields['acceleration'][0]
            vertical[name] = float(np.max(abs(fields['acceleration'][1])/np.maximum(abs(predictions[name]), 1e-30)))
    return {'predictions': predictions, 'vertical_relative_max': vertical,
            'grid': definition, 'variant': variant,
            'units': 'inward force in (km/s)^2/kpc', 'radii_kpc': radii}


def refinement(reference, alternative):
    """Report every comparison; a failure cannot remove an unfavorable radius."""
    return {key: {'fractional_change': alternative['predictions'][key]/value-1,
                  'maximum_absolute': float(np.max(abs(alternative['predictions'][key]/value-1)))}
            for key, value in reference['predictions'].items()}


@dataclass(frozen=True)
class Geometry:
    published_distance: float
    nominal_distance: float
    published_inclination: float
    nominal_inclination: float

    def radii(self, published):
        return np.asarray(published)*self.nominal_distance/self.published_distance

    def velocity_factor(self, inclination_offset=0.0):
        inclination = self.nominal_inclination+inclination_offset
        if not 0 < inclination < 90:
            raise ValueError('physical disk inclination required')
        return np.sin(np.deg2rad(self.published_inclination))/np.sin(np.deg2rad(inclination))

    def distance_speed_factor(self, offset):
        if self.nominal_distance+offset <= 0:
            raise ValueError('positive distance required')
        return np.sqrt((self.nominal_distance+offset)/self.nominal_distance)


def losses(predicted, observed, errors, inclination_deg, inclination_error_deg):
    """Three diagnostics, not calibrated likelihoods or independent data products."""
    predicted, observed, errors = np.broadcast_arrays(predicted, observed, errors)
    if (predicted.ndim != 1 or len(predicted) < 1 or not np.all(np.isfinite([predicted, observed, errors]))
            or np.any(predicted <= 0) or np.any(observed <= 0) or np.any(errors <= 0)):
        raise ValueError('finite positive one-dimensional speeds and random errors required')
    residual = predicted-observed
    standardized = residual/errors
    correlated_shift = observed/np.tan(np.deg2rad(inclination_deg))*np.deg2rad(inclination_error_deg)
    covariance = np.diag(errors**2)+np.outer(correlated_shift, correlated_shift)
    return {'random_error_loss': float(np.mean(standardized**2)),
            'inclination_covariance_loss': float(residual@np.linalg.solve(covariance, residual)/len(residual)),
            'five_kms_floor_loss': float(np.mean(residual**2/(errors**2+25))),
            'velocity_RMS_kms': float(np.sqrt(np.mean(residual**2))),
            'median_predicted_observed_ratio': float(np.median(predicted/observed)),
            'residual_kms': residual, 'standardized_residual': standardized}


def paired_influence(candidate_z, comparator_z, trim_fraction):
    """Keep the primary loss; companion diagnostics delete one or trim both tails."""
    delta = np.asarray(candidate_z)**2-np.asarray(comparator_z)**2
    if delta.ndim != 1 or len(delta) < 3 or not np.all(np.isfinite(delta)) or not 0 <= trim_fraction < .5:
        raise ValueError('at least three finite paired residuals and a small trim required')
    primary = float(np.mean(delta))
    most = int(np.argmax(abs(delta)))
    leave_one = float(np.mean(np.delete(delta, most)))
    count = int(np.floor(len(delta)*trim_fraction))
    sorted_delta = np.sort(delta)
    trimmed = float(np.mean(sorted_delta[count:len(delta)-count]))
    return {'candidate_minus_RAR_loss': primary, 'most_influential_row_position': most,
            'drop_one_radial_loss_difference': leave_one, 'trim_each_tail_count': count,
            'trimmed_radial_loss_difference': trimmed,
            'drop_one_sign_change': bool(np.sign(primary) != np.sign(leave_one)),
            'trim_sign_change': bool(np.sign(primary) != np.sign(trimmed)),
            'single_object_removal': None, 'single_object_removal_reason': 'only one development galaxy'}
