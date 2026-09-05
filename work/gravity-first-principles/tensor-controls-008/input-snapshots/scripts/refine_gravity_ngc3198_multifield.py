"""Resolve inherited p=2, lambda=6 numerical exceptions without relaxing gates."""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from run_gravity_ngc3198_multifield import forces, score, serial

from invariant_gravity_extensions.external_multifield import FluxPoissonSolver
from invariant_gravity_extensions.galaxy_development import (
    SI_ACCELERATION_TO_KMS2_KPC,
    candidate_id,
    source_disks,
)
from invariant_gravity_extensions.isolated_axisymmetric import MultipoleGrid
from invariant_gravity_extensions.isolated_multifield import (
    gradient_on_flux_grid,
    solve_isolated_auxiliary,
)
from invariant_gravity_extensions.reconstructed_axisymmetric import ReconstructedNewtonianSource


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_ngc3198_multifield_refinement_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    parent_path = ROOT/config['parent']/'result.json'
    if sha256(parent_path.read_bytes()).hexdigest() != config['parent_sha256']:
        raise ValueError('parent result changed')
    parent = json.loads(parent_path.read_bytes())
    c = json.loads((ROOT/config['parent_config']).read_bytes())
    old = json.loads((ROOT/c['scalar_config']).read_bytes())
    prior = json.loads((ROOT/c['scalar_run']/'result.json').read_bytes())
    maps = json.loads((ROOT/c['source_run']/'source_profiles.json').read_bytes())
    source_record = json.loads((ROOT/c['source_run']/'result.json').read_bytes())
    paths = [Path(__file__), args.config, ROOT/config['parent_config'], ROOT/'scripts/run_gravity_ngc3198_multifield.py',
             parent_path, ROOT/c['scalar_config'], ROOT/c['scalar_run']/'result.json',
             ROOT/c['source_run']/'source_profiles.json', ROOT/c['source_run']/'result.json',
             *sorted((ROOT/config['parent']).glob('fields_*.json')),
             *sorted((ROOT/c['scalar_run']).glob('fields_*.json')),
             *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]

    def hashes():
        return {p.resolve().relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    before = hashes()
    # Large parents are already immutable tracked artifacts. Seal their hashes;
    # copy new executing code/config exactly, without duplicating their payloads.
    for p in paths[:4]:
        target = args.output/'input-snapshots'/p.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    provenance = {'config': config, 'input_hashes': before, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'large_parent_artifacts_referenced_by_hash': True, 'new_observations_accessed': False}
    write('started.json', provenance)
    try:
        scalar, auxiliary = {}, {}
        for variant in old['source_variants']:
            for resolution in ['fine', 'coarse']:
                key = variant['id']+'/'+resolution
                name = 'fields_'+key.replace('/', '_')+'.json'
                scalar[key] = json.loads((ROOT/c['scalar_run']/name).read_bytes())
                auxiliary[key] = json.loads((ROOT/config['parent']/name).read_bytes())
        for key in ['boundary', 'map']:
            scalar[key] = json.loads((ROOT/c['scalar_run']/f'fields_{key}.json').read_bytes())
            auxiliary[key] = json.loads((ROOT/config['parent']/f'fields_{key}.json').read_bytes())
        previous_auxiliary = copy.deepcopy(auxiliary)
        selected = [card for card in parent['cards'] if card['power'] == config['power'] and card['mixing'] == config['mixing']]
        diagnostics = {}
        for name in config['refined_source_variants']:
            print('Refining '+name+' at 4097 x 640, l=160', flush=True)
            variant = next(v for v in old['source_variants'] if v['id'] == name)
            disks = source_disks(maps['profiles'][-1], variant)
            density_grid = MultipoleGrid(**config['grid'], plane_scale=min(d.height for d in disks.values()))
            G = source_record['config']['units']['G_kpc_kms2_msun']
            components = tuple(disks.values())
            newtonian = ReconstructedNewtonianSource.build(
                name, lambda R, z, grav=G, parts=components:
                4*np.pi*grav*sum(d.density(R, z) for d in parts), density_grid)
            solver = FluxPoissonSolver(MultipoleGrid(**config['grid']))
            a0 = c['a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC
            p = gradient_on_flux_grid(newtonian.potential, solver)/a0
            radii = np.asarray(scalar[name+'/fine']['radii_kpc'])
            old_n = np.asarray(scalar[name+'/fine']['predictions']['NEWTON_BARYONS'])
            new_n = newtonian.fields(radii, 0)['gradient'][0]
            n_error = float(np.max(abs(new_n/old_n-1)))

            def solve(beta, field_solver=solver, gradient=p, probe_radii=radii, acceleration_unit=a0):
                a = solve_isolated_auxiliary(field_solver, gradient, beta, config['power'], **c['solver'])
                field = -a.physical_flux_potential.evaluate(probe_radii, np.zeros_like(probe_radii))['acceleration']*acceleration_unit
                return {'beta': beta, 'power': config['power'], 'inward_unit_force': field[0],
                        'vertical_unit_gradient': field[1], 'iterations': len(a.history),
                        'relative_equation_residual': a.relative_equation_residual,
                        'maximum_equation_residual': a.maximum_equation_residual}

            with ThreadPoolExecutor(max_workers=2) as pool:
                jobs = [pool.submit(solve, b) for b in c['beta']]
                fields = [job.result() for job in jobs]
            updated = {**auxiliary[name+'/fine'], 'grid': config['grid'], 'unit_fields': fields}
            auxiliary[name+'/fine'] = updated
            diagnostics[name] = {'newtonian_refinement_maximum': n_error, 'prior_newtonian': old_n, 'refined_newtonian': new_n}
            write('fields_'+name+'_refined.json', updated)
        gates = {card['id']: copy.deepcopy(parent['numerical_admission'][card['id']]) for card in selected}
        for card in selected:
            gate = gates[card['id']]
            replaced = {'resolution/'+name+'/coarse' for name in config['refined_source_variants']}
            gate['numerical_failures'] = [f for f in gate['numerical_failures'] if f.get('comparison') not in replaced]
            gate['refinement_followup'] = []
            for name in config['refined_source_variants']:
                key = name+'/fine'
                high, vertical = forces(c, card, auxiliary[key], scalar[key])
                low = forces(c, card, previous_auxiliary[key], scalar[key])[0]
                scalar_name = candidate_id(card['shape'], c['a0_m_s2'])
                scalar_error = abs(np.asarray(scalar[key]['predictions'][scalar_name])-np.asarray(scalar[name+'/coarse']['predictions'][scalar_name]))
                norm = np.maximum(abs(high), scalar[key]['predictions']['NEWTON_BARYONS'])
                bound = (abs(high-low)+scalar_error)/norm
                maximum = float(np.max(bound))
                gate['refinement_followup'].append({'variant': name, 'combined_discrepancy': bound, 'maximum': maximum,
                                                     'meaning': 'sum of observed numerical changes; not a rigorous error bound'})
                if maximum > config['combined_discrepancy_limit'] or diagnostics[name]['newtonian_refinement_maximum'] > config['maximum_newtonian_refinement_change']:
                    gate['numerical_failures'].append({'comparison': 'refined/'+name, 'reason': 'refined numerical discrepancy', 'maximum': maximum})
                if np.max(vertical/norm) > c['numerical_controls']['maximum_relative_vertical_force']:
                    gate['numerical_failures'].append({'comparison': 'refined/'+name, 'reason': 'reflection symmetry'})
                gate['no_inward_circular_branch'] = [b for b in gate['no_inward_circular_branch'] if b['run'] != key]
                if np.any(high <= 0):
                    gate['no_inward_circular_branch'].append({'run': key, 'radius_positions': np.flatnonzero(high <= 0), 'inward_forces': high[high <= 0]})
            gate['numerical_pass'] = not gate['numerical_failures']
        write('numerical_admission.json', gates)
        result = score(c, old, prior, maps, selected, auxiliary, scalar, gates)
        if hashes() != before:
            raise RuntimeError('input changed during refinement')
        write('result.json', {**provenance, **result, 'cards': selected, 'numerical_admission': gates,
                              'source_diagnostics': diagnostics, 'scoped_supersession': 'lambda=6,p=2 only; all other parent outcomes retained'})
        write('receipt.json', {'status': result['status'], 'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps({'admitted': sum(v['numerical_pass'] for v in gates.values()), 'cards': len(selected),
                          'newtonian_refinement_changes': {k:v['newtonian_refinement_maximum'] for k,v in diagnostics.items()}}), flush=True)
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_OR_NUMERICAL_FAILURE_RETAINED', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
