"""Frozen same-card galaxy transfer after derivative-consistent source controls."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.galaxy_development import Geometry, losses, paired_influence
from invariant_gravity_extensions.length_galaxy_development import (
    core_fidelity,
    field_family,
    numerical_admission,
)
from invariant_gravity_extensions.length_screening import LengthScreening


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(v) for key, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    return value


def score(old_config, prior, maps, families, gates, model_ids, geometry):
    primary = prior['primary_rows']
    radii = np.asarray([r['nominal_radius_kpc'] for r in primary])
    field_radii = np.asarray(families['primary/fine']['radii_kpc'])
    positions = np.searchsorted(field_radii, radii)
    if not np.array_equal(field_radii[positions], radii):
        raise ValueError('Response radii changed')
    published = np.asarray([r['published_velocity_kms'] for r in primary])
    errors = np.asarray([r['published_random_error_kms'] for r in primary])
    lookup = {key: {(r['model'], r['distance_scale']): r for r in family['predictions']} for key, family in families.items()}
    scenarios = []
    for variant in old_config['source_variants']:
        for offset in old_config['geometry']['distance_offsets_mpc']:
            distance = (geometry.nominal_distance+offset)/geometry.nominal_distance
            for inclination in old_config['geometry']['inclination_offsets_deg']:
                factor = geometry.velocity_factor(inclination)
                observed, error = published*factor, errors*factor
                rows = {}
                for model in model_ids:
                    gate = gates[model]
                    force = np.asarray(lookup[variant['id']+'/fine'][(model, distance)]['inward_force'])[positions]
                    status = ('NUMERICAL_UNRESOLVED_RETAINED_UNSCORED' if not gate['numerical_pass'] else
                              'NO_COMPLETE_INWARD_CIRCULAR_BRANCH_RETAINED_UNSCORED' if not gate['complete_inward_branch'] else
                              'QUALITY_LIMITED_DEVELOPMENT_RETAINED')
                    row = {'status': status, 'inward_force': force, 'predicted_velocity_kms': None}
                    if status == 'QUALITY_LIMITED_DEVELOPMENT_RETAINED':
                        velocity = np.sqrt(distance*radii*force)
                        row.update(losses(velocity, observed, error, geometry.nominal_inclination+inclination,
                                          old_config['geometry']['published_inclination_error_deg']))
                        row['predicted_velocity_kms'] = velocity
                        for name, keep in [('inner', radii < old_config['radial_selection']['outer_stratum_minimum_kpc']),
                                           ('outer', radii >= old_config['radial_selection']['outer_stratum_minimum_kpc'])]:
                            row[name+'_random_error_loss'] = float(np.mean(row['standardized_residual'][keep]**2))
                    rows[model] = row
                comparator = rows['RAR_2016_ALGEBRAIC']
                for row in rows.values():
                    if 'standardized_residual' in row and 'standardized_residual' in comparator:
                        row['influence'] = paired_influence(row['standardized_residual'], comparator['standardized_residual'],
                            old_config['scoring']['symmetric_radial_influence_trim_fraction_each_tail'])
                scenarios.append({'source_variant': variant['id'], 'distance_offset_mpc': offset,
                                  'distance_scale': distance, 'inclination_offset_deg': inclination,
                                  'observed_velocity_kms': observed, 'random_error_kms': error, 'candidate_results': rows})
    nominal = next(s for s in scenarios if s['source_variant'] == 'primary' and s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0)
    summary = {}
    for model in model_ids:
        pairs = [(s['candidate_results'][model], s['candidate_results']['RAR_2016_ALGEBRAIC']) for s in scenarios]
        comparisons = {}
        for metric in ['random_error_loss', 'inclination_covariance_loss', 'five_kms_floor_loss']:
            differences = [a[metric]-b[metric] for a, b in pairs if metric in a and metric in b]
            comparisons[metric] = {'comparable_scenarios': len(differences), 'lower_loss_count': sum(v < 0 for v in differences),
                'higher_loss_count': sum(v > 0 for v in differences), 'tie_count': sum(v == 0 for v in differences),
                'minimum_matched_difference': min(differences) if differences else None,
                'maximum_matched_difference': max(differences) if differences else None}
        summary[model] = {'nominal': nominal['candidate_results'][model], 'matched_comparisons_with_RAR': comparisons,
                          'raw_galaxy_count': 1, 'quality_verified_counterexamples': 0,
                          'uncertainty_resolved_counterexamples': 0, 'independent_replications': 0, 'family_pruned': False}
    return {'primary_rows': primary, 'scenarios': scenarios, 'summary': summary}


def campaign(config, write):
    for key in ['local_result', 'cluster_result', 'galaxy_predecessor', 'source_result']:
        if sha256((ROOT/config[key]).read_bytes()).hexdigest() != config[key+'_sha256']:
            raise ValueError('Parent digest mismatch: '+key)
    local = json.loads((ROOT/config['local_result']).read_bytes())
    cluster = json.loads((ROOT/config['cluster_result']).read_bytes())
    old = json.loads((ROOT/config['inherited_config']).read_bytes())
    maps = json.loads((ROOT/config['source_profiles']).read_bytes())
    source_result = json.loads((ROOT/config['source_result']).read_bytes())
    selection = json.loads((ROOT/config['radial_selection']).read_bytes())
    if old['object_id'] != 'NGC3198' or not source_result['controls']['passes']:
        raise ValueError('Only the admitted NGC3198 development source is supported')
    cards = []
    for item in local['rows']:
        card = item['card']
        expected = LengthScreening(card['shape'], card['epsilon']).card(card['length_pc'], card['a0_m_s2'])
        if {k: v for k, v in card.items() if k != 'id'} != expected:
            raise ValueError('Action card changed since local audit')
        if not any(m.get('card_sha256') == card['card_sha256'] for m in cluster['models']):
            raise ValueError('Action card missing from cluster run')
        cards.append({**card, 'prior_local_status': item['status']})
    if len(cards) != 54:
        raise ValueError('All 54 cards required')
    model_ids = ['NEWTON_BARYONS', 'RAR_2016_ALGEBRAIC', *[c['id'] for c in cards]]
    geometry = Geometry(old['geometry']['published_distance_mpc'], maps['metadata']['distance_mpc'],
                        old['geometry']['published_inclination_deg'], maps['metadata']['inclination_deg'])
    distances = [(geometry.nominal_distance+offset)/geometry.nominal_distance for offset in old['geometry']['distance_offsets_mpc']]
    radii = np.asarray(selection['gate_radii_kpc'])
    write('registry.json', {'cards': cards, 'model_ids': model_ids, 'distance_scales': distances,
                          'radii_kpc': radii, 'source_variants': old['source_variants'], 'new_velocity_scoring_started': False})
    core = [core_fidelity(maps['profiles'][-1], variant) for variant in old['source_variants']]
    write('core_source_fidelity.json', {'variants': core, 'gravity_prediction_started': False})
    if any(abs(c['total_mass_fraction_change']) > config['source_regularization']['maximum_component_total_mass_fraction_change']
           or c['maximum_measured_surface_change_over_peak'] > config['source_regularization']['maximum_measured_surface_change_over_peak']
           for row in core for c in row['components']):
        raise RuntimeError('Fixed inner source fidelity gate failed before gravity calculation')
    families = {}
    G = source_result['config']['units']['G_kpc_kms2_msun']
    cases = [(v['id']+'/'+res, maps['profiles'][-1], v, old['field_grids'][res])
             for v in old['source_variants'] for res in ['coarse', 'fine']]
    cases.extend([('boundary', maps['profiles'][-1], {'id': 'boundary'}, old['field_grids']['boundary']),
                  ('map', maps['profiles'][0], {'id': 'map'}, old['field_grids']['fine'])])
    for name, profile, variant, grid in cases:
        print('Length-action joint field '+name, flush=True)
        family = field_family(profile, variant, grid, cards, distances, radii, G, workers=config['workers'])
        families[name] = family
        write('fields_'+name.replace('/', '_')+'.json', family)
    gates = numerical_admission(families, model_ids, [v['id'] for v in old['source_variants']], distances, old['field_grids'])
    write('numerical_admission.json', gates)
    # Only now form new velocity residuals, using the already-exposed predecessor packet.
    prior = json.loads((ROOT/config['galaxy_predecessor']).read_bytes())
    results = score(old, prior, maps, families, gates, model_ids, geometry)
    statuses = Counter(row['nominal']['status'] for row in results['summary'].values())
    return {'cards': cards, 'model_ids': model_ids, 'numerical_admission': gates, **results,
            'status_counts': dict(statuses), 'source_core_fidelity': core,
            'source_projection_diagnostics': [{key: family[key] for key in ['variant', 'grid', 'projected_source_negative_mass_fraction', 'projected_source_L1_fraction_error', 'projection_scope']}
                                               for family in families.values()],
            'field_record_names': ['fields_'+key.replace('/', '_')+'.json' for key in families],
            'scoped_observables': {'galaxies': 1, 'velocity_targets': len(prior['primary_rows']), 'scenarios': len(results['scenarios']),
                                   'model_configurations': len(model_ids), 'direct_outer_star_targets': 0,
                                   'new_cluster_or_local_observations': 0, 'lensing_observations': 0, 'new_raw_or_reserved_data': False},
            'new_formula_selected': False, 'full_solar_system_pass': False, 'discovery_claim': False,
            'family_pruning_authorized': False, 'inherited_limitations': old['unresolved'],
            'additional_limitations': ['Unmeasured smooth central source continuation',
                'Pointwise spectral density and higher derivative accuracy is not certified by target-force refinement',
                'No kernel/constant derivation from microscopic principles, covariant photon sector or dynamical stability proof']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_length_ngc3198_development_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(),
             *[ROOT/config[key] for key in ['local_result', 'cluster_result', 'galaxy_predecessor', 'inherited_config',
                                            'source_profiles', 'source_result', 'radial_selection']],
             *[ROOT/path for path in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    before = hashes()
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    provenance = {'config': config, 'input_hashes': before, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    write('started.json', provenance)
    try:
        test = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
                              env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'},
                              capture_output=True, text=True, check=False)
        write('controls.json', {'command': test.args, 'exit_code': test.returncode, 'stdout': test.stdout, 'stderr': test.stderr})
        if test.returncode:
            raise RuntimeError('Analytic controls failed before galaxy prediction')
        result = campaign(config, write)
        if hashes() != before:
            raise RuntimeError('Input changed during registered calculation')
        write('result.json', {**provenance, **result})
        write('receipt.json', {'status': 'COMPLETED_AT_DECLARED_DEVELOPMENT_SCOPE',
                               'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps({'status_counts': result['status_counts'], 'scope': result['scoped_observables']}))
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_FAILURE_RETAINED_NOT_PHYSICS_REJECTION', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
