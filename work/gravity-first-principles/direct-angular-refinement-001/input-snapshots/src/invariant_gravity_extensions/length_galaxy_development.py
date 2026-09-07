"""Joint source, exact global-length distance transfer and numerical gates."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.integrate import quad

from .external_multifield import FluxPoissonSolver
from .galaxy_development import SI_ACCELERATION_TO_KMS2_KPC, source_disks
from .isolated_axisymmetric import MultipoleGrid
from .length_axisymmetric import C3MultipolePotential, RegularSurfaceDensityDisk, full_length_flux
from .length_screening import LengthScreening


def regular_disks(profile, variant):
    old = source_disks(profile, variant)
    disks = {name: RegularSurfaceDensityDisk(d.radius, d.surface_density, d.height, d.outer_radius, d.taper_width)
             for name, d in old.items()}
    return old, disks


def core_fidelity(profile, variant):
    old, new = regular_disks(profile, variant)
    rows = []
    for name, disk in new.items():
        r0 = disk.radius[0]
        old_core = np.pi*old[name].surface(r0)*r0**2
        new_core = quad(lambda R, d=disk: 2*np.pi*R*d.surface(R), 0, r0, epsabs=1e-5, epsrel=1e-11)[0]
        breaks = np.unique(np.r_[disk.radius[disk.radius < disk.outer_radius], disk.outer_radius-disk.taper_width])
        total = quad(lambda R, d=disk: 2*np.pi*R*d.surface(R), 0, disk.outer_radius,
                     points=breaks, epsabs=1e-3, epsrel=1e-9, limit=300)[0]
        measured = disk.surface(disk.radius)
        prior = old[name].surface(disk.radius)
        rows.append({'component': name, 'core_coefficient_kpc_minus2': disk.core_coefficient,
                     'central_surface_over_first_measured': float(disk.surface(0)/disk.surface_density[0]),
                     'old_core_mass_msun': old_core, 'new_core_mass_msun': new_core, 'total_mass_msun': total,
                     'total_mass_fraction_change': (new_core-old_core)/(total-new_core+old_core),
                     'maximum_measured_surface_change_over_peak': float(np.max(abs(measured-prior))/np.max(prior))})
    return {'variant': variant, 'components': rows}


def field_family(profile, variant, definition, cards, distances, radii, G, *, workers=3):
    _, disks = regular_disks(profile, variant)
    source_grid = MultipoleGrid(**definition, plane_scale=min(d.height for d in disks.values()))

    def density(R, z):
        return 4*np.pi*G*sum(d.density(R, z) for d in disks.values())

    source = C3MultipolePotential.build(source_grid, density)
    grid = MultipoleGrid(**definition)
    solver = FluxPoissonSolver(grid)
    r, mu, sine = solver.radius[:, None], solver.mu, solver.sine
    fields = source.fields(r*sine, r*mu)
    target_newton = source.fields(radii, np.zeros_like(radii))
    newton = target_newton['gradient_r_theta'][0]
    vertical = -target_newton['gradient_r_theta'][1]
    if np.any(newton <= 0) or np.any(~np.isfinite(newton)):
        raise ValueError('nonpositive joint Newtonian target source; no RAR square-root clipping')
    a0_rar = 1.2e-10*SI_ACCELERATION_TO_KMS2_KPC
    rar_factor = 1/(-np.expm1(-np.sqrt(newton/a0_rar)))
    predictions = [{'model': name, 'distance_scale': d, 'inward_force': force, 'vertical_gradient': vz}
                   for d in distances for name, force, vz in [('NEWTON_BARYONS', newton, vertical),
                    ('RAR_2016_ALGEBRAIC', newton*rar_factor, vertical*rar_factor)]]
    groups = list(dict.fromkeys((c['shape'], c['a0_m_s2'], c['epsilon']) for c in cards))

    def group(shape, a0_si, epsilon):
        spec = LengthScreening(shape, epsilon)
        a0 = a0_si*SI_ACCELERATION_TO_KMS2_KPC
        results, cache = [], {}
        for card in cards:
            if (card['shape'], card['a0_m_s2'], card['epsilon']) != (shape, a0_si, epsilon):
                continue
            for distance in distances:
                effective_length = card['length_pc']/1000/distance
                if effective_length not in cache:
                    flux = full_length_flux(fields, spec, effective_length, a0)
                    solved = solver.solve(flux)
                    correction = -solved.evaluate(radii, np.zeros_like(radii))['acceleration']
                    cache[effective_length] = correction
                correction = cache[effective_length]
                results.append({'model': card['id'], 'distance_scale': distance,
                                'physical_length_pc': card['length_pc'], 'nominal_coordinate_length_kpc': effective_length,
                                'inward_force': newton+correction[0], 'vertical_gradient': vertical+correction[1]})
        print(f"  Completed {variant['id']} m={shape:g} a0={a0_si:.1e}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=workers) as pool:
        tasks = [pool.submit(group, *values) for values in groups]
        for task in tasks:
            predictions.extend(task.result())
    # These diagnostics describe spectral source error; they never replace the
    # declared positive physical density or remove unfavorable spatial points.
    physical = density(r*sine, r*mu)
    projected = fields['laplacian']
    weights = r**3*solver.weights
    from scipy.integrate import simpson

    positive_mass = simpson(np.sum(physical*weights, axis=1), x=solver.t)
    negative_mass = simpson(np.sum(np.maximum(-projected, 0)*weights, axis=1), x=solver.t)
    source_l1 = simpson(np.sum(abs(projected-physical)*weights, axis=1), x=solver.t)/positive_mass
    return {'variant': variant, 'grid': definition, 'source_plane_scale_kpc': source_grid.plane_scale,
            'radii_kpc': radii, 'predictions': predictions,
            'projected_source_negative_mass_fraction': float(negative_mass/positive_mass),
            'projected_source_L1_fraction_error': float(source_l1),
            'projection_scope': 'Finite angular/radial numerical approximation to the declared positive source; no physical negative matter is added. Pointwise higher-derivative convergence is not implied by target-force convergence.',
            'units': 'forces in (km/s)^2/kpc at homologous nominal coordinates',
            'distance_rule': 'Keep physical ell and a0 fixed. Compute at ell/d on nominal source; actual circular speed is sqrt(d*r_nominal*g).'}


def numerical_admission(families, model_ids, variants, distances, gates):
    lookup = {key: {(row['model'], row['distance_scale']): row for row in value['predictions']}
              for key, value in families.items()}
    pairs = [(v+'/fine', v+'/coarse', 'resolution', gates['maximum_relative_force_refinement']) for v in variants]
    pairs += [('primary/fine', name, name, gates['maximum_relative_force_'+suffix])
              for name, suffix in [('boundary', 'boundary_change'), ('map', 'map_refinement')]]
    answers = {}
    for model in model_ids:
        comparisons, failures, branches = [], [], []
        for key, entries in lookup.items():
            for distance in distances:
                row = entries[(model, distance)]
                force, vertical = np.asarray(row['inward_force']), np.asarray(row['vertical_gradient'])
                scale = np.maximum(abs(force), entries[('NEWTON_BARYONS', distance)]['inward_force'])
                if np.any(~np.isfinite(force)) or np.any(~np.isfinite(vertical)):
                    failures.append({'family': key, 'distance_scale': distance, 'reason': 'nonfinite force'})
                symmetry = float(np.max(abs(vertical)/scale))
                if symmetry > gates['reflection_relative_vertical_force_maximum']:
                    failures.append({'family': key, 'distance_scale': distance, 'reason': 'reflection symmetry', 'maximum': symmetry})
                if np.any(force <= 0):
                    branches.append({'family': key, 'distance_scale': distance, 'positions': np.flatnonzero(force <= 0),
                                     'inward_force': force[force <= 0]})
        for reference, alternative, kind, limit in pairs:
            for distance in distances:
                a = np.asarray(lookup[reference][(model, distance)]['inward_force'])
                b = np.asarray(lookup[alternative][(model, distance)]['inward_force'])
                delta = (b-a)/np.maximum(abs(a), lookup[reference][('NEWTON_BARYONS', distance)]['inward_force'])
                maximum = float(np.max(abs(delta)))
                comparisons.append({'reference': reference, 'alternative': alternative, 'kind': kind,
                                    'distance_scale': distance, 'normalized_difference': delta, 'maximum': maximum, 'limit': limit})
                if maximum > limit:
                    failures.append({'kind': kind, 'alternative': alternative, 'distance_scale': distance, 'maximum': maximum, 'limit': limit})
        answers[model] = {'numerical_pass': not failures, 'complete_inward_branch': not branches,
                          'failures': failures, 'nonpositive_branch_records': branches, 'comparisons': comparisons,
                          'family_pruned': False}
    return answers
