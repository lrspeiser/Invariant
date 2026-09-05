"""Frozen off-plane Newtonian source audit; no response or gravity-card scores."""
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
from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def weighted_norm(values, weights):
    return np.sqrt(np.einsum('i,i...,i...->...', weights, values, values))


def normalizers(fields, minimum_height, acceleration, half_radius):
    spherical = np.hypot(fields['radius'][:, None], fields['height'][None, :])
    H = np.sqrt(fields['hessian_norm'])
    return {'force': np.maximum(np.linalg.norm(fields['gradient_R_z'], axis=0), 1e-10*acceleration),
            'hessian': np.maximum(H, 1e-10*acceleration/half_radius),
            'third': np.maximum(fields['third_tensor_norm'], H/(spherical+minimum_height)),
            'density_gradient_floor': H/(spherical+minimum_height),
            'invariant_gradient': np.maximum(np.linalg.norm(fields['gradient_hessian_norm_R_z'], axis=0),
                                             fields['hessian_norm']/(spherical+minimum_height))}


def differences(value, reference, scale):
    return {'force_scaled_change': np.linalg.norm(value['gradient_R_z']-reference['gradient_R_z'], axis=0)/scale['force'],
            'hessian_scaled_change': weighted_norm(value['hessian_RR_Rz_zz_pp']-reference['hessian_RR_Rz_zz_pp'], [1, 2, 1, 1])/scale['hessian'],
            'third_tensor_scaled_change': weighted_norm(value['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-reference['third_RRR_RRz_Rzz_zzz_Rpp_zpp'],
                                                       [1, 3, 3, 1, 3, 3])/scale['third']}


def reflection(fields, scale):
    rows = []
    for key, parity, denominator in [('gradient_R_z', [1, -1], scale['force']),
            ('hessian_RR_Rz_zz_pp', [1, -1, 1, 1], scale['hessian']),
            ('third_RRR_RRz_Rzz_zzz_Rpp_zpp', [1, -1, 1, -1, 1, -1], scale['third']),
            ('gradient_laplacian_R_z', [1, -1], scale['third']),
            ('gradient_hessian_norm_R_z', [1, -1], scale['invariant_gradient'])]:
        error = abs(fields[key][..., ::-1]-np.array(parity)[:, None, None]*fields[key])/denominator
        rows.append({'field': key, 'maximum_scaled_error': float(np.max(error))})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_hankel_offplane_audit_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(), *[ROOT/config[k] for k in ['source_profiles', 'source_result', 'predecessor_result']],
             *[ROOT/t['path'] for t in config['transforms']], *[ROOT/t for t in config['control_tests']],
             *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for key in ['source_result', 'predecessor_result']:
        if hashes[config[key]] != config[key+'_sha256']:
            raise ValueError(f'{key} changed')
    for record in config['transforms']:
        if hashes[record['path']] != record['sha256']:
            raise ValueError('Registered radial transform changed')
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_response_scoring': False, 'new_gravity_cards': False, 'new_raw_or_reserved_data': False,
        'new_quality_verified_observational_tests': 0}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Off-plane analytic controls failed')
        maps = json.loads((ROOT/config['source_profiles']).read_bytes())
        source_result = json.loads((ROOT/config['source_result']).read_bytes())
        predecessor = json.loads((ROOT/config['predecessor_result']).read_bytes())
        transforms = {(t['radial_nodes'], t['wavenumber_nodes']): json.loads((ROOT/t['path']).read_bytes()) for t in config['transforms']}
        for t in transforms.values():
            for key in ['k', 'wavenumber_weights', 'surface_hankel']:
                t[key] = np.array(t[key])
        G = source_result['config']['units']['G_kpc_kms2_msun']
        R, z = np.array(config['radii_kpc']), np.array(config['heights_kpc'])
        if not np.array_equal(z, -z[::-1]):
            raise ValueError('All registered reflection pairs must be present')
        source_profile = maps['profiles'][-1]
        half_radius = source_profile['stellar_half_mass_radius_kpc']
        acceleration = G*sum(transforms[(128, 32)]['component_mass'])/half_radius**2
        vertical_sources = {v['id']: Sech2VerticalGreen(**{k: val for k, val in v.items() if k != 'id'}) for v in config['vertical_sources']}
        vertical_source_checks = []
        for name, source in vertical_sources.items():
            probe = np.linspace(0, source.extent, 4*(len(source.nodes)-1)+1)
            f, fp = source.source(probe)
            t = np.exp(-2*probe)
            exact = 2*t/(1+t)**2
            exact_prime = -2*np.tanh(probe)*exact
            vertical_source_checks.append({'definition': name, 'unnormalized_mass': source.unnormalized_mass,
                'normalization_residual': 2*(source.spline.integrate(0, source.extent)+source.tail_density/source.tail_rate)-1,
                'maximum_peak_scaled_density_error': float(np.max(abs(f-exact))/.5),
                'maximum_peak_scaled_density_derivative_error': float(np.max(abs(fp-exact_prime))/.5),
                'minimum_sampled_density': float(np.min(f)), 'weak_third_derivative_jump': source.third_derivative_jump})
            write(f'vertical_source_{name}.json', {'definition': name, 'nodes': source.nodes,
                'cubic_coefficients': source.spline.c, 'tail_density': source.tail_density, 'tail_rate': source.tail_rate,
                'unnormalized_mass': source.unnormalized_mass, 'weak_third_derivative_jump': source.third_derivative_jump})
        records, summaries = [], []
        for variant in config['variants']:
            _, disks = regular_disks(source_profile, variant)
            minimum_height = min(d.height for d in disks.values())
            rho = np.zeros((len(R), len(z)))
            grad_rho = np.zeros((2, len(R), len(z)))
            for d in disks.values():
                density, gradient = d.density_and_gradient(R[:, None], z[None, :])
                rho += density
                grad_rho += gradient
            physical_q, physical_grad_q = 4*np.pi*G*rho, 4*np.pi*G*grad_rho
            vertical_cache = {}
            by_case = {}
            for case in config['cases']:
                print(f"Off-plane {variant['id']} {case['id']}", flush=True)
                transform = transforms[(case['radial_nodes'], case['wavenumber_nodes'])]
                k, w = transform['k'], transform['wavenumber_weights']
                heights = np.array([disks[n].height for n in transform['components']])
                vertical = []
                for height in heights:
                    key = (case['wavenumber_nodes'], case['vertical_source'], height)
                    if key not in vertical_cache:
                        vertical_cache[key] = vertical_sources[case['vertical_source']].jet(k*height, z/height)/height**np.arange(4)[:, None, None]
                    vertical.append(vertical_cache[key])
                vertical = np.array(vertical)
                mask = k < case['cutoff_kpc_inverse']
                fields = cylindrical_jet(k[mask], w[mask], transform['surface_hankel'][:, mask], vertical[..., mask], R, z, G)
                scale = normalizers(fields, minimum_height, acceleration, half_radius)
                source_errors = {'physical_density_scaled_error': abs(fields['laplacian']-physical_q)/np.maximum(abs(physical_q), scale['hessian']),
                    'physical_density_gradient_scaled_error': np.linalg.norm(fields['gradient_laplacian_R_z']-physical_grad_q, axis=0)/
                        np.maximum(np.linalg.norm(physical_grad_q, axis=0), scale['density_gradient_floor'])}
                row = {'variant': variant, 'case': case, 'fields': fields, 'source_errors': source_errors,
                    'vertical_kernel_le_float64_sha256': sha256(vertical.astype('<f8').tobytes()).hexdigest(),
                    'reflection': reflection(fields, scale)}
                records.append(row)
                by_case[case['id']] = row
            reference = by_case['reference']
            scale = normalizers(reference['fields'], minimum_height, acceleration, half_radius)
            comparisons = []
            target = config['numerical_targets']
            for case in config['cases']:
                if case['id'] == 'reference':
                    continue
                diff = differences(by_case[case['id']]['fields'], reference['fields'], scale)
                maxima = {'maximum_'+key: float(np.max(value)) for key, value in diff.items()}
                comparisons.append({'case': case['id'], 'role': case['role'], 'errors_by_probe': diff, 'maximum_errors': maxima,
                                    'within_registered_targets': all(value < target[key] for key, value in maxima.items())})
            mid = next(row for row in predecessor['records'] if row['variant'] == variant and row['radial_nodes_per_interval'] == 128
                and row['wavenumber_nodes_per_interval'] == 32 and row['wavenumber_cutoff_kpc_inverse'] == 400.)['jet']
            ri = [R.tolist().index(r) for r in mid['radius']]
            zi = z.tolist().index(0.)
            f = reference['fields']
            oldH = np.array(mid['hessian_RR_ZZ_PP'])
            oldT = np.array(mid['radial_derivative_hessian_RR_ZZ_PP'])
            midplane = {'force': float(np.max(abs(f['gradient_R_z'][0, ri, zi]/np.array(mid['radial_gradient'])-1))),
                'hessian': float(np.max(np.linalg.norm(f['hessian_RR_Rz_zz_pp'][[0, 2, 3]][:, ri, zi]-oldH, axis=0)/np.linalg.norm(oldH, axis=0))),
                'radial_hessian_derivative': float(np.max(np.linalg.norm(f['third_RRR_RRz_Rzz_zzz_Rpp_zpp'][[0, 2, 4]][:, ri, zi]-oldT, axis=0)/
                    np.maximum(np.linalg.norm(oldT, axis=0), np.linalg.norm(oldH, axis=0)/np.array(mid['radius']))))}
            source_max = {'maximum_'+key: float(np.max(value)) for key, value in reference['source_errors'].items()}
            mirror = max(r['maximum_scaled_error'] for record in by_case.values() for r in record['reflection'])
            passed = (all(c['within_registered_targets'] for c in comparisons if c['role'] == 'refinement') and
                all(v < target[k] for k, v in source_max.items()) and mirror < target['maximum_reflection_scaled_error'] and
                max(midplane.values()) < target['maximum_midplane_reference_scaled_difference'])
            summaries.append({'variant': variant, 'comparisons': comparisons, 'reference_source_errors': source_max,
                'maximum_reflection_scaled_error': mirror, 'midplane_reference_differences': midplane,
                'within_all_registered_offplane_targets': passed})
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError('Input changed during off-plane audit')
        write('result.json', {**provenance, 'G_kpc_kms2_msun': G, 'vertical_source_checks': vertical_source_checks,
            'records': records, 'summary': summaries, 'gravity_rejection': False, 'full_unbounded_field_validation': False})
        write('receipt.json', {'status': 'CONDITIONAL_OFFPLANE_NEWTONIAN_REFERENCE_RETAINED',
            'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps(serial([{k: v for k, v in row.items() if k != 'comparisons'} for row in summaries])))
    except Exception as exc:
        write('failure.json', {'status': 'OFFPLANE_AUDIT_EXECUTION_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
