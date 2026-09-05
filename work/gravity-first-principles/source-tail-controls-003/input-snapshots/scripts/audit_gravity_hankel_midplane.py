"""Independent Newtonian source/derivative audit, with no response scoring."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.hankel_midplane import (
    disk_transforms,
    midplane_jet,
    piecewise_gauss,
    sech2_midplane_laplace,
)
from invariant_gravity_extensions.isolated_axisymmetric import MultipoleGrid
from invariant_gravity_extensions.length_axisymmetric import C3MultipolePotential
from invariant_gravity_extensions.length_galaxy_development import regular_disks


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: serial(v) for key, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def jet_difference(value, reference):
    r = np.asarray(reference['radius'])
    H = np.asarray(reference['hessian_RR_ZZ_PP'])
    dH = np.asarray(reference['radial_derivative_hessian_RR_ZZ_PP'])
    hscale = np.linalg.norm(H, axis=0)
    tscale = np.maximum(np.linalg.norm(dH, axis=0), hscale/r)
    return {'maximum_force_fraction_change': float(np.max(abs(value['radial_gradient']/reference['radial_gradient']-1))),
            'maximum_hessian_norm_scaled_change': float(np.max(np.linalg.norm(value['hessian_RR_ZZ_PP']-H, axis=0)/hscale)),
            'maximum_third_derivative_scaled_change': float(np.max(np.linalg.norm(value['radial_derivative_hessian_RR_ZZ_PP']-dH, axis=0)/tscale))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_hankel_midplane_audit_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(), *[ROOT/config[k] for k in ['source_profiles', 'source_result']],
             *[ROOT/p for p in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for key in ['source_result', 'predecessor_result']:
        if sha256((ROOT/config[key]).read_bytes()).hexdigest() != config[key+'_sha256']:
            raise ValueError(f'{key} changed')
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'new_response_scoring': False, 'new_modified_gravity_predictions': False,
                  'new_raw_or_reserved_data': False, 'quality_verified_observational_tests': 0}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Independent integral controls failed')
        maps = json.loads((ROOT/config['source_profiles']).read_bytes())
        source_result = json.loads((ROOT/config['source_result']).read_bytes())
        G = source_result['config']['units']['G_kpc_kms2_msun']
        variants = [(v, regular_disks(maps['profiles'][-1], v)[1]) for v in config['variants']]
        r = np.array(config['midplane_radii_kpc'])
        cutoffs = config['wavenumber_cutoffs_kpc_inverse']
        step = config['wavenumber_interval_kpc_inverse']
        edges = np.arange(round(max(cutoffs)/step)+1)*step
        records = []
        for k_nodes in config['wavenumber_nodes_per_interval']:
            k, weights = piecewise_gauss(edges, k_nodes)
            for radial_nodes in config['radial_nodes_per_source_interval']:
                print(f'Hankel transform radial={radial_nodes}, k rule={k_nodes}, cutoff={max(cutoffs)}', flush=True)
                transform = disk_transforms(variants[0][1], k, radial_nodes)
                # This sharing is valid only because the registered variants
                # differ by height. Enforce exact radial-source equality.
                for _, disks in variants[1:]:
                    for name, primary_disk in variants[0][1].items():
                        other = disks[name]
                        if not (np.array_equal(primary_disk.radius, other.radius) and
                                np.array_equal(primary_disk.surface_density, other.surface_density) and
                                primary_disk.outer_radius == other.outer_radius and primary_disk.taper_width == other.taper_width):
                            raise ValueError('Radial transform sharing requires identical source profiles')
                file = f'transform_r{radial_nodes}_k{k_nodes}.json'
                write(file, {**transform, 'wavenumber_weights': weights, 'wavenumber_nodes_per_interval': k_nodes})
                file_hash = sha256((args.output/file).read_bytes()).hexdigest()
                for variant, disks in variants:
                    heights = np.array([disks[n].height for n in transform['components']])
                    vertical = sech2_midplane_laplace(heights[:, None]*k)
                    for cutoff in cutoffs:
                        mask = k < cutoff
                        jet = midplane_jet(k[mask], weights[mask], transform['surface_hankel'][:, mask],
                            vertical[:, mask], .5/heights, r, G)
                        rho, drho = np.zeros_like(r), np.zeros_like(r)
                        for disk in disks.values():
                            density, grad = disk.density_and_gradient(r, np.zeros_like(r))
                            rho += density
                            drho += grad[0]
                        q, dq = 4*np.pi*G*rho, 4*np.pi*G*drho
                        scale = np.maximum(abs(dq), np.linalg.norm(jet['hessian_RR_ZZ_PP'], axis=0)/r)
                        records.append({'variant': variant, 'radial_nodes_per_interval': radial_nodes,
                            'wavenumber_nodes_per_interval': k_nodes, 'wavenumber_cutoff_kpc_inverse': cutoff,
                            'transform_file': file, 'transform_sha256': file_hash, 'jet': jet,
                            'physical_laplacian': q, 'physical_radial_gradient_laplacian': dq,
                            'physical_density_fraction_error': abs(jet['laplacian']/q-1),
                            'physical_density_gradient_scaled_error': abs(jet['radial_gradient_laplacian']-dq)/scale})
        summaries, old = [], []
        target = config['numerical_targets']
        for variant, disks in variants:
            rows = [x for x in records if x['variant'] == variant]
            reference = next(x for x in rows if x['radial_nodes_per_interval'] == max(config['radial_nodes_per_source_interval'])
                and x['wavenumber_nodes_per_interval'] == max(config['wavenumber_nodes_per_interval'])
                and x['wavenumber_cutoff_kpc_inverse'] == max(cutoffs))
            comparisons = []
            for row in rows:
                keys = ['radial_nodes_per_interval', 'wavenumber_nodes_per_interval', 'wavenumber_cutoff_kpc_inverse']
                changed = [key for key in keys if row[key] != reference[key]]
                if len(changed) == 1 and row['wavenumber_cutoff_kpc_inverse'] != min(cutoffs):
                    delta = jet_difference(row['jet'], reference['jet'])
                    comparisons.append({'changed': changed[0], 'configuration': {k: row[k] for k in keys},
                        'differences': delta, 'within_targets': all(delta[k] < target[k] for k in delta)})
            density_error = float(np.max(reference['physical_density_fraction_error']))
            gradient_error = float(np.max(reference['physical_density_gradient_scaled_error']))
            source_pass = (density_error < target['maximum_physical_density_fraction_error'] and
                           gradient_error < target['maximum_physical_density_gradient_scaled_error'])
            if len(comparisons) != 3:
                raise ValueError('All three independent refinement axes must be measured')
            summaries.append({'variant': variant, 'refinements': comparisons,
                'maximum_physical_density_fraction_error': density_error,
                'maximum_physical_density_gradient_scaled_error': gradient_error,
                'within_registered_midplane_targets': source_pass and all(x['within_targets'] for x in comparisons)})
            for definition in config['comparison_multipole_grids']:
                print(f"Multipole comparison {variant['id']} {definition['id']}", flush=True)
                grid = MultipoleGrid(**{k: v for k, v in definition.items() if k != 'id'},
                                     plane_scale=min(d.height for d in disks.values()))

                def physical_source(R, z, disks=disks):
                    return 4*np.pi*G*sum(d.density(R, z) for d in disks.values())

                potential = C3MultipolePotential.build(grid, physical_source)
                fields = potential.fields(r, np.zeros_like(r))
                ref = reference['jet']
                hscale = np.linalg.norm(ref['hessian_RR_ZZ_PP'], axis=0)
                H = fields['hessian_rr_rt_tt_pp'][[0, 2, 3]]
                old.append({'variant': variant, 'grid': definition, 'fields': fields,
                    'force_fraction_difference': abs(fields['gradient_r_theta'][0]/ref['radial_gradient']-1),
                    'hessian_norm_scaled_difference': np.linalg.norm(H-ref['hessian_RR_ZZ_PP'], axis=0)/hscale,
                    'hessian_invariant_gradient_scaled_difference': abs(fields['gradient_hessian_norm_r_theta'][0]-ref['radial_gradient_hessian_norm'])/
                        np.maximum(abs(ref['radial_gradient_hessian_norm']), ref['hessian_norm']/r),
                    'physical_density_fraction_error': abs(fields['laplacian']/reference['physical_laplacian']-1),
                    'physical_density_gradient_scaled_error': abs(fields['gradient_laplacian_r_theta'][0]-reference['physical_radial_gradient_laplacian'])/
                        np.maximum(abs(reference['physical_radial_gradient_laplacian']), hscale/r)})
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != h for p, h in hashes.items()):
            raise RuntimeError('Input changed during audit')
        write('result.json', {**provenance, 'G_kpc_kms2_msun': G, 'records': records, 'summary': summaries,
            'multipole_comparisons': old, 'physical_gravity_rejection': False, 'full_field_validation': False})
        write('receipt.json', {'status': 'CONDITIONAL_MIDPLANE_REFERENCE_RETAINED',
            'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps(serial(summaries)))
    except Exception as exc:
        write('failure.json', {'status': 'MIDPLANE_AUDIT_FAILURE_RETAINED', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
