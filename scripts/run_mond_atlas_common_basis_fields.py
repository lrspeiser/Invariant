"""Validate both newly fitted source representations without motion scoring."""
from __future__ import annotations
import argparse, gc, io, shutil, unittest
from pathlib import Path
import numpy as np
from mond_atlas_common import ROOT, read_json, write_json, write_csv, digest
from run_mond_atlas_blocked_refinement import execute
from check_mond_atlas_field_pattern import forces, vector_difference


def checked_bindings(summary):
    bindings = {}
    for group in ('code_hashes', 'source_bindings', 'input_bindings'):
        bindings.update(summary.get(group, {}))
    for item in summary.get('products', []):
        bindings[item['path']] = item['sha256']
    for path, expected in bindings.items():
        if digest(ROOT/path) != expected:
            raise ValueError('bound input changed: '+path)
    return bindings


def run(config_path, output, private):
    config = read_json(config_path)
    if config['admission_disposition'] != 'SOURCE_BLOCKED':
        raise ValueError('this package is restricted to source/numerical diagnostics')
    source = read_json(ROOT/config['source_summary'])
    solver = read_json(ROOT/config['solver_summary'])
    bindings = checked_bindings(source)
    bindings.update(checked_bindings(solver))
    if not source['all_optimizers_converged'] or not solver['replay_pass']:
        raise ValueError('source optimizer or numerical replay not verified')
    known = {p['path'].replace('\\', '/'): p['sha256'] for p in source['products']}
    specs = [(c['source'], c['vertical_components']) for c in config['stellar_cases']]
    specs += [(config['gas_cases'][key], [[1., config['gas_cases']['height_kpc']]])
              for key in ('atomic_helium', 'co21')]
    for path, layers in specs:
        if digest(ROOT/path) != known[path]:
            raise ValueError('unbound common-basis source: '+path)
        with np.load(ROOT/path) as packet:
            if not np.array_equal(packet['vertical_layers'], layers):
                raise ValueError('image and field vertical layers differ')
            axis = packet['axis']; surface = packet['intrinsic_effective_surface']
            if surface.shape != (len(axis), len(axis)) or not np.all(np.diff(axis) > 0):
                raise ValueError('invalid source grid')
            if np.any(surface[[0, -1], :]) or np.any(surface[:, [0, -1]]):
                raise ValueError('nonzero edge nodes invalidate the full hat integral')
    storage = read_json(ROOT/config['storage_protocol'])
    storage['workspace_disk_reserve_bytes'] = config['workspace_disk_reserve_bytes']
    gravity = read_json(ROOT/config['gravity_protocol'])
    if output.exists() or private.exists():
        raise FileExistsError('immutable outputs; use unused paths')
    estimates = []
    for grid in config['grids']:
        shape = np.array([round(2*grid['half_width_kpc']/h)+1 for h in grid['spacing_kpc']], dtype=np.int64)
        estimates.append(int(8*(3*np.prod(shape)+np.prod(shape-2)))+1024*1024)
    required = len(config['stellar_cases'])*sum(estimates)
    available = shutil.disk_usage(ROOT).free
    if available < required+config['workspace_disk_reserve_bytes']:
        raise OSError(f'declared ensemble needs {required} bytes plus reserve; only {available} free')
    output.mkdir(parents=True); private.mkdir(parents=True)
    paths = [config_path, ROOT/config['source_summary'], ROOT/config['solver_summary'],
             ROOT/config['storage_protocol'], ROOT/config['gravity_protocol'], Path(__file__),
             ROOT/'scripts/run_mond_atlas_blocked_refinement.py', ROOT/'scripts/mond_atlas_blocked_fields.py',
             ROOT/'scripts/run_mond_atlas_ngc2903_fields.py', ROOT/'scripts/run_mond_atlas_reprojected_fields.py',
             ROOT/'scripts/mond_atlas_fields.py', ROOT/'scripts/mond_atlas_rectangular_fields.py',
             ROOT/'scripts/check_mond_atlas_field_pattern.py']
    bindings.update({str(p.relative_to(ROOT)): digest(p) for p in paths})
    write_json(output/'prospective-bindings.json', dict(admission_disposition='SOURCE_BLOCKED',
        config=config, config_sha256=digest(config_path), bindings=bindings,
        declared_field_runs=len(config['stellar_cases'])*len(config['grids']),
        disk_estimate_bytes=required, free_disk_before_bytes=available,
        response_files_opened=[], kinematic_response_scores_computed=0))
    suite = unittest.defaultTestLoader.discover(str(ROOT/'tests'), pattern='test_mond_atlas*.py')
    log = io.StringIO(); tests = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    (output/'validation.log').write_text(log.getvalue(), encoding='utf-8', newline='\n')
    if not tests.wasSuccessful():
        raise RuntimeError(log.getvalue())
    all_results = []; checks = []
    for case in config['stellar_cases']:
        for grid in config['grids']:
            label = case['id']+'_'+grid['id']
            result = execute(case, config, gravity, storage, grid['half_width_kpc'],
                             grid['spacing_kpc'], output, private, label)
            all_results.append(result)
            if grid['id'] != 'base':
                values = vector_difference(forces(output, case['id']+'_base'), forces(output, label))
                passed = all(v < (config['maximum_ring_vector_rms_gate'] if 'maximum_ring' in k
                                  else config['vector_relative_rms_gate']) for k, v in values.items())
                check = dict(case=case['id'], perturbation=grid['id'], **values, gates_pass=passed)
                checks.append(check); write_csv(output/'numerical-checks.csv', checks)
                print(check, flush=True)
            gc.collect()
    integrity = []
    for result in all_results:
        residuals = {key: result['numerical'][key]['relative_pde_residual'] for key in ('newton', 'mond')}
        mass_error = abs(sum(result['source']['component_mass_msun'].values()) /
                         result['source']['finite_grid_total_mass_msun']-1)
        passed = max(residuals.values()) < config['relative_pde_residual_gate'] and mass_error < config['source_integral_relative_error_gate']
        integrity.append(dict(id=result['id'], newton_pde_relative_residual=residuals['newton'],
            mond_pde_relative_residual=residuals['mond'], source_mass_relative_error=mass_error, gates_pass=passed))
    write_csv(output/'field-integrity.csv', integrity)
    write_json(output/'summary.json', dict(status='COMMON_BASIS_FIELDS_NUMERICAL_DIAGNOSTIC',
        admission_disposition='SOURCE_BLOCKED', config=config, config_sha256=digest(config_path),
        bindings=bindings, source_cases=2, new_full_field_runs=len(all_results),
        numerical_benchmark_tests=tests.testsRun, checks=checks,
        numerical_gates_pass=all(c['gates_pass'] for c in checks+integrity),
        case_numerical_gates_pass={c['id']:all(r['gates_pass'] for r in checks if r['case']==c['id']) for c in config['stellar_cases']},
        response_files_opened=[], kinematic_response_scores_computed=0,
        admitted_galaxy_cube_predictions=0, goal_complete=False))
    print(dict(field_runs=len(all_results), numerical_gates_pass=all(c['gates_pass'] for c in checks+integrity),
               goal_complete=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/mond_atlas_common_basis_fields_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--private', type=Path, required=True)
    args = parser.parse_args()
    run(args.config.resolve(), args.output.resolve(), args.private.resolve())
