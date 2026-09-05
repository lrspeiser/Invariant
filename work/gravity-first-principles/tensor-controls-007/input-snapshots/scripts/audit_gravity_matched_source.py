"""Audit the actual piecewise C3 Newtonian potential at every registered point."""
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
from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.hankel_axisymmetric import cylindrical_jet
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_axisymmetric import matched_grid
from invariant_gravity_extensions.vertical_green import Sech2VerticalGreen


def serial(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serial(x) for x in v]
    return v


def weighted_norm(v, weights):
    return np.sqrt(np.einsum('i,i...,i...->...', weights, v, v))


def differences(v, f, scale):
    return {'force': np.linalg.norm(v['gradient_R_z']-f['gradient_R_z'], axis=0)/scale['force'],
        'hessian': weighted_norm(v['hessian_RR_Rz_zz_pp']-f['hessian_RR_Rz_zz_pp'], [1, 2, 1, 1])/scale['hessian'],
        'third': weighted_norm(v['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-f['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], [1, 3, 3, 1, 3, 3])/scale['third']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_matched_source_audit_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(serial(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    inputs = [Path(__file__), args.config.resolve(), *[ROOT/p for p in config['input_files']],
        *[ROOT/p for p in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs}
    for path, digest in config['input_files'].items():
        if hashes[path] != digest:
            raise ValueError(f'Input changed: {path}')
    for p in inputs:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_response_scoring': False, 'new_gravity_parameters': False, 'new_raw_or_reserved_data': False,
        'quality_verified_observational_tests': 0, 'complete_nonlinear_action_solver': False}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Matched potential controls failed')
        exterior_result = json.loads((ROOT/config['exterior_result']).read_bytes())
        if (not all(r['within_registered_exterior_targets'] for r in exterior_result['summary']) or
                config['inner_join_kpc'] < exterior_result['config']['canonical_minimum_radius_kpc']):
            raise ValueError('Exterior provider not admitted for this transition')
        profile = json.loads((ROOT/config['source_profiles']).read_bytes())['profiles'][-1]
        G = json.loads((ROOT/config['source_result']).read_bytes())['config']['units']['G_kpc_kms2_msun']
        transforms = {}
        for t in config['transforms']:
            value = json.loads((ROOT/t['path']).read_bytes())
            transforms[(t['radial_nodes'], t['wavenumber_nodes'])] = {k: np.array(v) if k in
                ['k', 'wavenumber_weights', 'surface_hankel'] else v for k, v in value.items()}
        R, z = np.array(config['radii_kpc']), np.array(config['heights_kpc'])
        RR, ZZ = np.meshgrid(R, z, indexing='ij')
        r = np.hypot(RR, ZZ)
        join = (r >= config['inner_join_kpc']) & (r <= config['outer_join_kpc'])
        rows, summary = [], []
        for variant in config['variants']:
            _, disks = regular_disks(profile, variant)
            moments = json.loads((ROOT/config['moment_files'][variant['id']]).read_bytes())
            exterior = ExteriorMomentField(moments, G, minimum_radius=config['inner_join_kpc'])
            gm = G*moments['compact_source_mass']
            physical = [d.density_and_gradient(RR, ZZ) for d in disks.values()]
            q = 4*np.pi*G*sum(v[0] for v in physical)
            grad_q = 4*np.pi*G*sum(v[1] for v in physical)
            cases = {}
            for case in config['cases']:
                print(f"Matched source {variant['id']} {case['id']}", flush=True)
                transform = transforms[(case['radial_nodes'], case['wavenumber_nodes'])]
                mask = transform['k'] < case['cutoff']
                k, w, S = transform['k'][mask], transform['wavenumber_weights'][mask], transform['surface_hankel'][:, mask]
                vertical_source = Sech2VerticalGreen(intervals=case['vertical_intervals'], extent=case['vertical_extent'])
                cache = {h: vertical_source.jet(k*h, z/h)/h**np.arange(4)[:, None, None] for h in {d.height for d in disks.values()}}
                near = cylindrical_jet(k, w, S, np.array([cache[disks[n].height] for n in transform['components']]), R, z, G)
                del cache
                matched = matched_grid(near, exterior, R, z, inner=config['inner_join_kpc'], outer=config['outer_join_kpc'])
                row = {'case': case, 'near_fields': near, 'matched_fields': matched}
                write(f"fields_{variant['id']}_{case['id']}.json", row)
                cases[case['id']] = row
            ref = cases['reference']['matched_fields']
            H = np.sqrt(ref['hessian_norm'])
            half = profile['stellar_half_mass_radius_kpc']
            height = min(d.height for d in disks.values())
            scale = {'force': np.maximum(np.linalg.norm(ref['gradient_R_z'], axis=0), 1e-10*gm/half**2),
                'hessian': np.maximum(H, 1e-10*gm/half**3), 'third': np.maximum(ref['third_tensor_norm'], H/(r+height))}
            qscale = np.maximum(abs(q), H)
            grad_qscale = np.maximum(np.linalg.norm(grad_q, axis=0), H/(r+height))
            comparisons = []
            for name, value in cases.items():
                if name == 'reference':
                    continue
                changes = differences(value['matched_fields'], ref, scale)
                comparisons.append({'case': name, 'errors_by_point': changes, 'maximum_errors': {k: float(np.max(v)) for k, v in changes.items()}})
            errors = {'density': abs(ref['laplacian']-q)/qscale,
                'density_gradient': np.linalg.norm(ref['gradient_laplacian_R_z']-grad_q, axis=0)/grad_qscale}
            near = cases['reference']['near_fields']
            diagnostic = {'standalone_near_density_gradient': np.linalg.norm(near['gradient_laplacian_R_z']-grad_q, axis=0)/grad_qscale}
            near_overlap = {k: v[..., join] for k, v in near.items() if k not in ['radius', 'height']}
            far_overlap = exterior.fields(RR[join], ZZ[join])
            cross = differences(near_overlap, far_overlap, {k: v[join] for k, v in scale.items()})
            potential_cross = abs(near_overlap['potential']-far_overlap['potential'])/(gm/r[join])
            worst = {}
            for key, error in errors.items():
                index = np.unravel_index(np.argmax(error), r.shape)
                worst[key] = {'value': float(error[index]), 'R_kpc': float(RR[index]), 'z_kpc': float(ZZ[index])}
            target = config['numerical_targets']
            passed = (all(v < target[k] for row in comparisons for k, v in row['maximum_errors'].items())
                and all(row['value'] < target[k] for k, row in worst.items())
                and float(np.max(potential_cross)) < target['potential'])
            compact = {'variant': variant, 'all_grid_points': r.size, 'transition_points': int(join.sum()),
                'maximum_refinement_changes': {k: max(c['maximum_errors'][k] for c in comparisons) for k in scale},
                'worst_source_identity_errors': worst,
                'maximum_near_far_diagnostic_difference': {k: float(np.max(v)) for k, v in cross.items()},
                'maximum_potential_difference_monopole_scaled': float(np.max(potential_cross)),
                'maximum_unused_near_density_gradient_diagnostic': float(np.max(diagnostic['standalone_near_density_gradient'][r >= config['outer_join_kpc']])),
                'within_all_registered_matched_targets': passed}
            rows.append({'variant': variant, 'comparisons': comparisons, 'source_identity_errors': errors,
                'standalone_provider_diagnostics': diagnostic, 'overlap_errors': cross, 'potential_overlap_error': potential_cross, 'scales': scale})
            summary.append(compact)
            print(json.dumps(serial(compact)), flush=True)
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError('Input changed during matched source audit')
        write('result.json', {**provenance, 'G_kpc_kms2_msun': G, 'records': rows, 'summary': summary,
            'production_interpolant_validated': False, 'new_physical_gravity_rejections': 0})
        write('receipt.json', {'status': 'MATCHED_SOURCE_AUDIT_RETAINED', 'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'status': 'MATCHED_SOURCE_EXECUTION_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
