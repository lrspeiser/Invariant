"""Fixed bounded-TRIMOND transfer to the already exposed NGC3198 source."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
from invariant_gravity_extensions.coherent_momentum import integrate_axisymmetric
from invariant_gravity_extensions.external_multifield import FluxPoissonSolver
from invariant_gravity_extensions.galaxy_development import (
    SI_ACCELERATION_TO_KMS2_KPC,
    Geometry,
    candidate_id,
    losses,
    paired_influence,
    source_disks,
)
from invariant_gravity_extensions.isolated_axisymmetric import (
    MassComponent,
    MultipoleGrid,
    solve_poisson,
    total_newtonian,
)
from invariant_gravity_extensions.isolated_multifield import (
    beta_zero_density_source,
    gradient_on_flux_grid,
    normalized_newtonian_gradient,
    solve_isolated_auxiliary,
)
from invariant_gravity_extensions.reconstructed_axisymmetric import (
    ReconstructedNewtonianSource,
    multipole_fields,
)
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def controls(config):
    c = config['theory_controls']
    grid = MultipoleGrid(**c['grid'])
    solver = FluxPoissonSolver(grid)
    r, mu, _ = grid.nodes()
    R, z = r[:, None]*np.sqrt(1-mu*mu), r[:, None]*mu
    parts = tuple(MassComponent(**p) for p in c['components'])
    fields = total_newtonian(parts, R, z)
    p = normalized_newtonian_gradient(fields, r, mu, c['a0'])
    rho = fields['laplacian']/(4*np.pi)
    normalizer = integrate_axisymmetric(grid, rho*np.linalg.norm(fields['gradient'], axis=0))
    rows = []
    for power in c['powers']:
        source_values = beta_zero_density_source(fields, c['a0'], power)
        independent = solve_poisson(grid, lambda R, z, values=source_values: values)
        for beta in c['beta']:
            solved = solve_isolated_auxiliary(solver, p, beta, power, **config['solver'])
            gradient = solver.gradient(solved.physical_flux_potential)*c['a0']
            az = -gradient[0]*mu+gradient[1]*solver.sine
            net = integrate_axisymmetric(grid, rho*az)/normalizer
            disagreement = None if beta != 0 else float(solver.energy_norm(solved.q-solver.gradient(independent))/solver.energy_norm(solved.q))
            rows.append({'beta': beta, 'power': power, 'normalized_internal_force': net,
                         'beta_zero_relative_field_disagreement': disagreement,
                         'equation_residual': solved.relative_equation_residual})
    mapped = MultipoleGrid(**c['grid'], plane_scale=.5)
    potential = solve_poisson(mapped, lambda R, z: total_newtonian(parts, R, z)['laplacian'])
    direct = normalized_newtonian_gradient(multipole_fields(potential, R, z), r, mu, 1)
    replay = gradient_on_flux_grid(potential, solver)
    replay_error = float(np.linalg.norm(replay-direct)/np.linalg.norm(direct))
    passes = (all(abs(row['normalized_internal_force']) < c['maximum_normalized_internal_force'] for row in rows) and
              all(row['beta_zero_relative_field_disagreement'] < c['maximum_beta_zero_field_disagreement'] for row in rows if row['beta'] == 0) and
              replay_error < c['maximum_gradient_angle_replay_error'])
    return {'rows': rows, 'gradient_angle_replay_error': replay_error, 'passes': passes}


def candidates(config, bound_data, local_data):
    rows = []
    for shape in config['shapes']:
        for beta in config['beta']:
            for power in config['powers']:
                bound = next(r for r in bound_data['rows'] if r['shape'] == shape and r['a0_m_s2'] == config['a0_m_s2'] and r['beta'] == beta and r['power'] == power)
                for mixing in config['mixing_by_power'][str(power)]:
                    if bound['intersection_empty'] or mixing > bound['absolute_mixing_interval'][1]:
                        raise ValueError('coupling outside declared conditional local bound')
                    spec = SaturatedActionSpec('trimond_alignment', shape=shape, mixing=mixing, beta=beta, power=power)
                    local_rows = [r for r in local_data['rows'] if r['shape'] == shape and r['a0_m_s2'] == config['a0_m_s2'] and r['beta'] == beta and r['power'] == power and r['mixing'] == .25]
                    if len(local_rows) != 2 or not all(r['numerical_controls_pass'] for r in local_rows):
                        raise ValueError('exact two-background local solutions required')
                    q2 = [r['scalar_Q2_s_minus2']+r['auxiliary_Q2_s_minus2']*(mixing/r['mixing'])**2 for r in local_rows]
                    rows.append({'id': f'TRI_m{shape:g}_b{beta:g}_p{power}_lambda{mixing:g}',
                                 'shape': shape, 'beta': beta, 'power': power, 'mixing': mixing,
                                 'a0_m_s2': config['a0_m_s2'], 'card_sha256': spec.card()['content_sha256'],
                                 'conditional_local_Q2_s_minus2': q2,
                                 'full_solar_system_pass': False})
    return rows


def family(config, profile, variant, definition, scalar_fields, G):
    disks = source_disks(profile, variant)
    source_grid = MultipoleGrid(**definition, plane_scale=min(d.height for d in disks.values()))
    source = ReconstructedNewtonianSource.build(variant['id'],
             lambda R, z: 4*np.pi*G*sum(d.density(R, z) for d in disks.values()), source_grid)
    solver = FluxPoissonSolver(MultipoleGrid(**definition))
    a0 = config['a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC
    p = gradient_on_flux_grid(source.potential, solver)/a0
    radii = np.asarray(scalar_fields['radii_kpc'])
    old_n = np.asarray(scalar_fields['predictions']['NEWTON_BARYONS'])
    new_n = source.fields(radii, 0)['gradient'][0]
    replay = float(np.max(abs(new_n/old_n-1)))
    if replay > config['numerical_controls']['maximum_relative_source_replay_error']:
        raise RuntimeError(f'Newtonian source replay does not match predecessor: {replay}')

    def solve(beta, power):
        a = solve_isolated_auxiliary(solver, p, beta, power, **config['solver'])
        force = -a.physical_flux_potential.evaluate(radii, np.zeros_like(radii))['acceleration']*a0
        return {'beta': beta, 'power': power, 'inward_unit_force': force[0], 'vertical_unit_gradient': force[1],
                'iterations': len(a.history), 'history': a.history,
                'relative_equation_residual': a.relative_equation_residual,
                'maximum_equation_residual': a.maximum_equation_residual}

    with ThreadPoolExecutor(max_workers=3) as pool:
        jobs = [pool.submit(solve, beta, power) for beta in config['beta'] for power in config['powers']]
        unit = [job.result() for job in jobs]
    return {'source_variant': variant, 'grid': definition, 'source_plane_scale_kpc': source_grid.plane_scale,
            'radii_kpc': radii, 'source_replay_maximum_relative_error': replay,
            'units': 'inward force in (km/s)^2/kpc per mixing squared', 'unit_fields': unit}


def forces(config, card, auxiliary, scalar):
    a = next(r for r in auxiliary['unit_fields'] if r['beta'] == card['beta'] and r['power'] == card['power'])
    name = candidate_id(card['shape'], config['a0_m_s2'])
    base = np.asarray(scalar['predictions'][name])
    total = base+card['mixing']**2*np.asarray(a['inward_unit_force'])
    vertical_bound = abs(card['mixing']**2*np.asarray(a['vertical_unit_gradient']))+scalar['vertical_relative_max'][name]*abs(base)
    return total, vertical_bound


def admission(config, cards, auxiliary, scalar):
    limits = config['numerical_controls']
    results = {}
    variants = [key.removesuffix('/fine') for key in auxiliary if key.endswith('/fine')]
    for card in cards:
        fields = {key: forces(config, card, value, scalar[key]) for key, value in auxiliary.items()}
        failures, branches, comparisons = [], [], []
        for key, (force, vertical) in fields.items():
            norm = np.maximum(abs(force), scalar[key]['predictions']['NEWTON_BARYONS'])
            if np.any(~np.isfinite(force)) or np.any(~np.isfinite(vertical)):
                failures.append({'run': key, 'reason': 'nonfinite field'})
            v = float(np.max(vertical/norm))
            if v > limits['maximum_relative_vertical_force']:
                failures.append({'run': key, 'reason': 'reflection symmetry', 'maximum': v})
            if np.any(force <= 0):
                branches.append({'run': key, 'radius_positions': np.flatnonzero(force <= 0).tolist(),
                                 'inward_forces': force[force <= 0].tolist()})
        pairs = [(v+'/fine', v+'/coarse', 'resolution', limits['maximum_resolution_difference']) for v in variants]
        pairs += [('primary/fine', name, name, limits['maximum_'+name+'_difference']) for name in ['boundary', 'map']]
        for fine, alternative, kind, limit in pairs:
            f, g = fields[fine][0], fields[alternative][0]
            norm = np.maximum(abs(f), scalar[fine]['predictions']['NEWTON_BARYONS'])
            delta = (g-f)/norm
            maximum = float(np.max(abs(delta)))
            comparisons.append({'fine': fine, 'alternative': alternative, 'kind': kind,
                                'normalized_difference': delta, 'maximum': maximum, 'limit': limit})
            if maximum > limit:
                failures.append({'comparison': kind+'/'+alternative, 'reason': 'numerical refinement', 'maximum': maximum, 'limit': limit})
        results[card['id']] = {'numerical_pass': not failures, 'numerical_failures': failures,
                               'no_inward_circular_branch': branches, 'comparisons': comparisons,
                               'numerical_failure_is_not_physical_rejection': True}
    return results


def score(config, old_config, prior, maps, cards, auxiliary, scalar, gates):
    geometry = Geometry(old_config['geometry']['published_distance_mpc'], maps['metadata']['distance_mpc'],
                        old_config['geometry']['published_inclination_deg'], maps['metadata']['inclination_deg'])
    primary = prior['primary_rows']
    radii = np.asarray([r['nominal_radius_kpc'] for r in primary])
    gate_radii = np.asarray(scalar['primary/fine']['radii_kpc'])
    positions = np.searchsorted(gate_radii, radii)
    if not np.array_equal(gate_radii[positions], radii):
        raise RuntimeError('response radii changed')
    published = np.asarray([r['published_velocity_kms'] for r in primary])
    random = np.asarray([r['published_random_error_kms'] for r in primary])
    scenarios = []
    for old in prior['scenarios']:
        variant = old['source_variant']
        offset, inclination = old['distance_offset_mpc'], old['inclination_offset_deg']
        factor = geometry.velocity_factor(inclination)
        observed, errors = published*factor, random*factor
        rar = old['candidate_results']['RAR_2016_ALGEBRAIC']
        # Independent check that the reused data geometry exactly matches the predecessor.
        baseline = losses(np.asarray(rar['predicted_velocity_kms']), observed, errors,
                          geometry.nominal_inclination+inclination, old_config['geometry']['published_inclination_error_deg'])
        if abs(baseline['random_error_loss']-rar['random_error_loss']) > 1e-9:
            raise RuntimeError('predecessor baseline or response geometry mismatch')
        entries = {}
        for card in cards:
            force = forces(config, card, auxiliary[variant+'/fine'], scalar[variant+'/fine'])[0][positions]
            branch = next((b for b in gates[card['id']]['no_inward_circular_branch'] if b['run'] == variant+'/fine'), None)
            record = {'inward_force': force, 'predicted_velocity_kms': None,
                      'complete_inward_circular_branch_on_gate_radii': branch is None,
                      'nonpositive_gate_radii': branch}
            if not gates[card['id']]['numerical_pass']:
                record['status'] = 'NUMERICAL_BRIDGE_UNRESOLVED_RETAINED'
            elif np.any(force <= 0):
                record.update(status='NO_INWARD_CIRCULAR_BRANCH_FOR_DECLARED_SOURCE',
                              nonpositive_response_positions=np.flatnonzero(force <= 0))
            else:
                velocity = np.sqrt(radii*force)*geometry.distance_speed_factor(offset)
                record.update(losses(velocity, observed, errors, geometry.nominal_inclination+inclination,
                                     old_config['geometry']['published_inclination_error_deg']))
                record['predicted_velocity_kms'] = velocity
                for name, keep in [('inner', radii < old_config['radial_selection']['outer_stratum_minimum_kpc']),
                                   ('outer', radii >= old_config['radial_selection']['outer_stratum_minimum_kpc'])]:
                    record[name+'_random_error_loss'] = float(np.mean(record['standardized_residual'][keep]**2))
                record['influence'] = paired_influence(record['standardized_residual'], rar['standardized_residual'],
                                  old_config['scoring']['symmetric_radial_influence_trim_fraction_each_tail'])
                record['status'] = 'QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED'
            entries[card['id']] = record
        scenarios.append({'source_variant': variant, 'distance_offset_mpc': offset, 'inclination_offset_deg': inclination,
                          'observed_velocity_kms': observed, 'random_error_kms': errors,
                          'RAR_comparator': rar, 'candidate_results': entries})
    nominal = next(s for s in scenarios if s['source_variant'] == 'primary' and s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0)
    summary = {}
    for card in cards:
        comparable = [(s['candidate_results'][card['id']], s['RAR_comparator']) for s in scenarios
                      if s['candidate_results'][card['id']]['status'] == 'QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED']
        summary[card['id']] = {'nominal': nominal['candidate_results'][card['id']],
                              'scenarios_comparable': len(comparable), 'scenarios_total': len(scenarios),
                              'scenarios_worse_than_RAR': {metric: sum(a[metric] > b[metric] for a, b in comparable)
                                     for metric in ['random_error_loss', 'inclination_covariance_loss', 'five_kms_floor_loss']},
                              'quality_verified_counterexample_galaxies': 0,
                              'uncertainty_resolved_counterexample_galaxies': 0,
                              'raw_nominal_worse_than_RAR_galaxies': None if nominal['candidate_results'][card['id']]['predicted_velocity_kms'] is None else
                              int(nominal['candidate_results'][card['id']]['random_error_loss'] > nominal['RAR_comparator']['random_error_loss']),
                              'single_object_removal': None, 'single_object_removal_reason': 'one development galaxy only'}
    return {'scenarios': scenarios, 'summary': summary, 'primary_rows': primary,
            'held_out_loss': None, 'held_out_loss_reason': 'previously exposed development pilot',
            'galaxies': 1, 'family_pruning': False, 'discovery_claim': False,
            'spherical_cluster_predictions_unchanged': True, 'status': 'QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_ngc3198_multifield_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--controls-only', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    old_config = json.loads((ROOT/config['scalar_config']).read_bytes())
    scalar_paths = sorted((ROOT/config['scalar_run']).glob('fields_*.json'))
    paths = [Path(__file__).resolve(), args.config.resolve(), ROOT/config['scalar_config'],
             ROOT/config['scalar_run']/'result.json', ROOT/config['source_run']/'result.json',
             ROOT/config['source_run']/'source_profiles.json', ROOT/config['local_run']/'result.json',
             ROOT/config['coupling_bounds'], *scalar_paths,
             *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, obj):
        with (args.output/name).open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(serial(obj), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    before = hashes()
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    provenance = {'input_hashes': before, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'config': config, 'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'controls_only': args.controls_only, 'raw_observations_accessed': False, 'thread_workers': 3}
    write('started.json', provenance)
    try:
        for name, key in [('scalar_run', 'scalar_result_sha256'), ('local_run', 'local_result_sha256')]:
            if sha256((ROOT/config[name]/'result.json').read_bytes()).hexdigest() != config[key]:
                raise ValueError('predecessor hash mismatch')
        checks = controls(config)
        write('controls.json', checks)
        print(json.dumps(checks), flush=True)
        if not checks['passes']:
            raise RuntimeError('Independent theoretical controls unresolved; galaxy calculation not started')
        result = {'controls_pass': True}
        if not args.controls_only:
            maps = json.loads((ROOT/config['source_run']/'source_profiles.json').read_bytes())
            source = json.loads((ROOT/config['source_run']/'result.json').read_bytes())
            prior = json.loads((ROOT/config['scalar_run']/'result.json').read_bytes())
            bound_data = json.loads((ROOT/config['coupling_bounds']).read_bytes())
            local_data = json.loads((ROOT/config['local_run']/'result.json').read_bytes())
            cards = candidates(config, bound_data, local_data)
            write('candidate_cards.json', cards)
            auxiliary, scalar = {}, {}
            for variant in old_config['source_variants']:
                for resolution in ['coarse', 'fine']:
                    key = variant['id']+'/'+resolution
                    print(f"Observed source: {key}", flush=True)
                    sf = json.loads((ROOT/config['scalar_run']/('fields_'+key.replace('/', '_')+'.json')).read_bytes())
                    scalar[key] = sf
                    auxiliary[key] = family(config, maps['profiles'][-1], variant, old_config['field_grids'][resolution], sf, source['config']['units']['G_kpc_kms2_msun'])
                    write('fields_'+key.replace('/', '_')+'.json', auxiliary[key])
            for name, profile, definition in [('boundary', maps['profiles'][-1], old_config['field_grids']['boundary']),
                                               ('map', maps['profiles'][0], old_config['field_grids']['fine'])]:
                print('Primary '+name+' check', flush=True)
                sf = json.loads((ROOT/config['scalar_run']/('fields_'+name+'.json')).read_bytes())
                scalar[name] = sf
                auxiliary[name] = family(config, profile, {'id': name}, definition, sf, source['config']['units']['G_kpc_kms2_msun'])
                write('fields_'+name+'.json', auxiliary[name])
            gates = admission(config, cards, auxiliary, scalar)
            write('numerical_admission.json', gates)
            print(f"Numerically admitted {sum(r['numerical_pass'] for r in gates.values())}/{len(cards)} cards", flush=True)
            result = {**score(config, old_config, prior, maps, cards, auxiliary, scalar, gates),
                      'cards': cards, 'numerical_admission': gates}
        if hashes() != before:
            raise RuntimeError('Input changed during run')
        write('result.json', {**provenance, 'controls': checks, **result})
        write('receipt.json', {'status': result.get('status', 'THEORY_CONTROLS_PASS'),
                               'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest(),
                               'finished_utc': datetime.now(UTC).isoformat()})
        print(json.dumps({'status': result.get('status', 'THEORY_CONTROLS_PASS'),
                          'cards': len(result.get('cards', [])), 'scenarios': len(result.get('scenarios', []))}), flush=True)
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_OR_NUMERICAL_FAILURE_RETAINED', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
