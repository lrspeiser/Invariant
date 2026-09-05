"""Append-only, fixed-parameter NGC3198 scalar gravity development experiment."""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.galaxy_development import (
    Geometry,
    field_family,
    losses,
    paired_influence,
    refinement,
)


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


def numerical_pass(config, records, comparisons):
    gates = config['field_grids']
    limits = {'resolution': gates['maximum_relative_force_refinement'],
              'boundary': gates['maximum_relative_force_boundary_change'],
              'map': gates['maximum_relative_force_map_refinement']}
    failures = []
    for name, family in records.items():
        for candidate, force in family['predictions'].items():
            if np.any(~np.isfinite(force)) or np.any(force <= 0):
                failures.append({'run': name, 'candidate': candidate, 'reason': 'nonpositive or nonfinite force'})
        for candidate, value in family['vertical_relative_max'].items():
            if value > gates['reflection_relative_vertical_force_maximum']:
                failures.append({'run': name, 'candidate': candidate, 'reason': 'reflection symmetry', 'value': value})
    for name, comparison in comparisons.items():
        limit = limits[name.split('/')[0]]
        for candidate, row in comparison.items():
            if row['maximum_absolute'] > limit:
                failures.append({'comparison': name, 'candidate': candidate, 'maximum': row['maximum_absolute'], 'limit': limit})
    return {'passes': not failures, 'failures': failures, 'comparisons': comparisons}


def score(config, geometry, galaxy, indices, selected_radii, fields, positions):
    # This is the first conversion of individual velocity values in this runner.
    values = np.asarray([[float(galaxy['rows'][i][j]) for j in [1, 2]] for i in indices])
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError('invalid velocity or random error retained; no automatic row removal')
    scenarios, primary_rows = [], []
    for variant in config['source_variants']:
        family = fields[variant['id']+'/fine']
        for distance_offset in config['geometry']['distance_offsets_mpc']:
            for inclination_offset in config['geometry']['inclination_offsets_deg']:
                factor = geometry.velocity_factor(inclination_offset)
                observed, errors = values[:, 0]*factor, values[:, 1]*factor
                entries = {}
                for candidate, force in family['predictions'].items():
                    velocity = np.sqrt(selected_radii*force[positions])*geometry.distance_speed_factor(distance_offset)
                    diagnostics = losses(velocity, observed, errors,
                                         geometry.nominal_inclination+inclination_offset,
                                         config['geometry']['published_inclination_error_deg'])
                    diagnostics['predicted_velocity_kms'] = velocity
                    for name, keep in [('inner', selected_radii < config['radial_selection']['outer_stratum_minimum_kpc']),
                                       ('outer', selected_radii >= config['radial_selection']['outer_stratum_minimum_kpc'])]:
                        diagnostics[name+'_random_error_loss'] = float(np.mean(diagnostics['standardized_residual'][keep]**2)) if keep.any() else None
                    entries[candidate] = diagnostics
                baseline = entries['RAR_2016_ALGEBRAIC']['standardized_residual']
                for candidate, row in entries.items():
                    row['influence'] = paired_influence(row['standardized_residual'], baseline,
                                          config['scoring']['symmetric_radial_influence_trim_fraction_each_tail'])
                scenarios.append({'source_variant': variant['id'], 'distance_offset_mpc': distance_offset,
                                  'inclination_offset_deg': inclination_offset, 'candidate_results': entries})
                if variant['id'] == 'primary' and distance_offset == 0 and inclination_offset == 0:
                    primary_rows = [{'published_row_index': int(index), 'nominal_radius_kpc': float(radius),
                                     'published_radius_kpc': float(galaxy['rows'][index][0]),
                                     'published_velocity_kms': values[j, 0], 'published_random_error_kms': values[j, 1],
                                     'geometry_corrected_velocity_kms': observed[j], 'geometry_corrected_error_kms': errors[j]}
                                    for j, (index, radius) in enumerate(zip(indices, selected_radii, strict=True))]
    nominal = next(s for s in scenarios if s['source_variant'] == 'primary' and s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0)
    summary = {}
    for candidate in nominal['candidate_results']:
        differences = [s['candidate_results'][candidate]['random_error_loss']-s['candidate_results']['RAR_2016_ALGEBRAIC']['random_error_loss'] for s in scenarios]
        summary[candidate] = {'nominal': nominal['candidate_results'][candidate],
                              'scenarios_worse_than_RAR': sum(d > 0 for d in differences),
                              'scenarios_better_than_RAR': sum(d < 0 for d in differences),
                              'scenario_count': len(scenarios),
                              'minimum_matched_loss_difference': min(differences),
                              'maximum_matched_loss_difference': max(differences),
                              'raw_nominal_worse_than_RAR_galaxies': int(nominal['candidate_results'][candidate]['random_error_loss'] > nominal['candidate_results']['RAR_2016_ALGEBRAIC']['random_error_loss']),
                              'raw_galaxy_count': 1,
                              'quality_verified_counterexample_galaxies': 0,
                              'uncertainty_resolved_counterexample_galaxies': 0}
    return {'primary_rows': primary_rows, 'scenarios': scenarios, 'summary': summary,
            'held_out_loss': None, 'held_out_loss_reason': 'previously exposed development pilot',
            'status': 'QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED',
            'individual_velocity_values_converted': int(len(indices)*2),
            'rotation_velocity_rows_scored': len(indices), 'galaxies_scored': 1,
            'population_replication': False, 'family_pruning': False, 'discovery_claim': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_ngc3198_scalar_development_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config_path = args.config.resolve()
    if not config_path.is_relative_to(ROOT):
        raise ValueError('config must be inside repository')
    config = json.loads(config_path.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    paths = [Path(__file__), config_path, *(ROOT/config[k] for k in ['response_asset', 'metadata_receipt', 'split_contract']),
             *(ROOT/config['source_run']/name for name in ['result.json', 'receipt.json', 'source_profiles.json']),
             *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        snapshot = args.output/'input-snapshots'/path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        with snapshot.open('xb') as handle:
            handle.write(path.read_bytes())
    provenance = {'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'whole_previously_exposed_response_container_read': True,
                  'access_class': 'development', 'config': config}
    write('started.json', provenance)
    try:
        source = json.loads((ROOT/config['source_run']/'result.json').read_bytes())
        if sha256((ROOT/config['source_run']/'result.json').read_bytes()).hexdigest() != config['source_result_sha256']:
            raise ValueError('source result hash mismatch')
        if source['status'] != 'SOURCE_AND_NUMERICAL_BRIDGE_RETAINED' or not source['controls']['passes']:
            raise ValueError('source and independent controls not admitted')
        split = json.loads((ROOT/config['split_contract']).read_bytes())
        if config['object_id'] != 'NGC3198' or split['assignment'].get(config['object_id']) != 'train':
            raise PermissionError('only the registered NGC3198 development galaxy is admitted')
        maps = json.loads((ROOT/config['source_run']/'source_profiles.json').read_bytes())
        metadata = json.loads((ROOT/config['metadata_receipt']).read_bytes())
        for key in ['published_distance_mpc', 'published_inclination_deg']:
            if config['geometry'][key] != metadata[key.removeprefix('published_')]:
                raise ValueError('published geometry mismatch')
        geometry = Geometry(config['geometry']['published_distance_mpc'], maps['metadata']['distance_mpc'],
                            config['geometry']['published_inclination_deg'], maps['metadata']['inclination_deg'])
        asset = json.loads((ROOT/config['response_asset']).read_bytes())
        galaxy = next(g for g in asset['galaxies'] if g['name'] == config['response_name'])
        row_payload = {key: galaxy[key] for key in ['name', 'point_count', 'rows']}
        actual_rows_hash = sha256(json.dumps(row_payload, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(',', ':')).encode('utf8')).hexdigest()
        if galaxy['provenance']['rows_sha256'] != config['response_rows_sha256'] or actual_rows_hash != config['response_rows_sha256']:
            raise ValueError('response rows identity mismatch')
        published_radii = np.asarray([float(row[0]) for row in galaxy['rows']])
        radial = config['radial_selection']
        all_radii = geometry.radii(published_radii)
        keep = (all_radii >= radial['minimum_at_nominal_source_distance_kpc']) & (all_radii <= radial['maximum_at_nominal_source_distance_kpc'])
        indices = np.flatnonzero(keep)
        selected = all_radii[keep]
        if len(indices) < config['scoring']['minimum_selected_rows']:
            raise ValueError('insufficient geometrically selected rows')
        radii = np.unique(np.r_[selected, np.linspace(radial['minimum_at_nominal_source_distance_kpc'],
                              radial['maximum_at_nominal_source_distance_kpc'], radial['supplementary_uniform_control_radii_count'])])
        positions = np.searchsorted(radii, selected)
        write('radial_selection.json', {'all_nominal_radii_kpc': all_radii, 'selected_indices': indices,
               'excluded_indices': np.flatnonzero(~keep), 'gate_radii_kpc': radii,
               'individual_velocity_values_converted': 0, 'whole_response_container_previously_exposed': True})
        families, comparisons = {}, {}
        for variant in config['source_variants']:
            for resolution in ['coarse', 'fine']:
                print(f"Joint scalar source: {variant['id']}, {resolution}", flush=True)
                key = variant['id']+'/'+resolution
                families[key] = field_family(maps['profiles'][-1], variant, config['field_grids'][resolution],
                                               config['candidates'], radii, source['config']['units']['G_kpc_kms2_msun'])
                write('fields_'+key.replace('/', '_')+'.json', families[key])
            comparisons['resolution/'+variant['id']] = refinement(families[variant['id']+'/fine'], families[variant['id']+'/coarse'])
        for name, profile, grid in [('boundary', maps['profiles'][-1], config['field_grids']['boundary']),
                                    ('map', maps['profiles'][0], config['field_grids']['fine'])]:
            print('Primary '+name+' check', flush=True)
            families[name] = field_family(profile, {'id': name}, grid, config['candidates'], radii,
                                           source['config']['units']['G_kpc_kms2_msun'])
            write('fields_'+name+'.json', families[name])
            comparisons[name+'/primary'] = refinement(families['primary/fine'], families[name])
        gates = numerical_pass(config, families, comparisons)
        write('numerical_gates.json', gates)
        if not gates['passes']:
            result = {**provenance, 'status': 'NUMERICAL_BRIDGE_UNRESOLVED_RETAINED',
                      'rotation_velocity_rows_scored': 0, 'individual_velocity_values_converted': 0,
                      'failures': gates['failures'], 'discovery_claim': False}
        else:
            print('Numerical gates pass; opening selected development velocities', flush=True)
            result = {**provenance, **score(config, geometry, galaxy, indices, selected, families, positions)}
        if hashes != {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}:
            raise RuntimeError('input changed during run')
        write('result.json', result)
        write('receipt.json', {'status': result['status'], 'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest(),
                              'finished_utc': datetime.now(UTC).isoformat()})
        print(json.dumps({'status': result['status'], 'rows_scored': result['rotation_velocity_rows_scored']}))
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_FAILURE_NOT_PHYSICAL_REJECTION', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
