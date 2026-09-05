"""Complete the leading omitted potential tail on the retained source grid."""
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
from invariant_gravity_extensions.hankel_tail import complete_leading_tail
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_axisymmetric import matched_grid


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


def norm(v, weights):
    return np.sqrt(np.einsum('i,i...,i...->...', weights, v, v))


def differences(v, f, scale):
    return {'force': np.linalg.norm(v['gradient_R_z']-f['gradient_R_z'], axis=0)/scale['force'],
        'hessian': norm(v['hessian_RR_Rz_zz_pp']-f['hessian_RR_Rz_zz_pp'], [1, 2, 1, 1])/scale['hessian'],
        'third': norm(v['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-f['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], [1, 3, 3, 1, 3, 3])/scale['third']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_source_tail_audit_v1.json')
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
    for p, digest in config['input_files'].items():
        if hashes[p] != digest:
            raise ValueError(f'Input changed: {p}')
    for p in inputs:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'longdouble_mantissa_bits': np.finfo(np.longdouble).nmant,
        'new_response_scoring': False, 'new_gravity_parameters': False, 'new_raw_or_reserved_data': False,
        'changed_physical_source': False, 'quality_verified_observational_tests': 0, 'complete_nonlinear_action_solver': False}
    write('started.json', provenance)
    try:
        controls = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'exit_code': controls.returncode, 'stdout': controls.stdout, 'stderr': controls.stderr})
        if controls.returncode:
            raise RuntimeError('Tail completion controls failed')
        old = json.loads((ROOT/config['base_result']).read_bytes())
        source = old['config']
        R, z = np.array(source['radii_kpc']), np.array(source['heights_kpc'])
        RR, ZZ = np.meshgrid(R, z, indexing='ij')
        r = np.hypot(RR, ZZ)
        G = old['G_kpc_kms2_msun']
        profile = json.loads((ROOT/source['source_profiles']).read_bytes())['profiles'][-1]
        transforms = {}
        for t in source['transforms']:
            raw = json.loads((ROOT/t['path']).read_bytes())
            transforms[(t['radial_nodes'], t['wavenumber_nodes'])] = {k: np.array(v) if k in ['k', 'wavenumber_weights', 'surface_hankel'] else v for k, v in raw.items()}
        rows, summary = [], []
        for variant in source['variants']:
            _, disks = regular_disks(profile, variant)
            moments = json.loads((ROOT/source['moment_files'][variant['id']]).read_bytes())
            exterior = ExteriorMomentField(moments, G, minimum_radius=source['inner_join_kpc'])
            physical = [d.density_and_gradient(RR, ZZ) for d in disks.values()]
            q = 4*np.pi*G*sum(v[0] for v in physical)
            grad_q = 4*np.pi*G*sum(v[1] for v in physical)
            cases = {}
            overlap_gauge = {}
            overlap = (r >= source['inner_join_kpc']) & (r <= source['outer_join_kpc'])
            exterior_overlap = exterior.fields(RR[overlap], ZZ[overlap])
            for case in config['cases']:
                print(f"Tail completed source {variant['id']} {case['id']}", flush=True)
                base = next(x for x in source['cases'] if x['id'] == case['base_case'])
                p = ROOT/config['base_directory']/f"fields_{variant['id']}_{case['base_case']}.json"
                near = {k: np.array(v) for k, v in json.loads(p.read_bytes())['near_fields'].items()}
                transform = transforms[(base['radial_nodes'], base['wavenumber_nodes'])]
                mask = transform['k'] < base['cutoff']
                k, w, S = transform['k'][mask], transform['wavenumber_weights'][mask], transform['surface_hankel'][:, mask]
                corrected, details = complete_leading_tail(near, disks, transform['components'], k, w, S, R, z, G, base['cutoff'], log_nodes=case['log_nodes'])
                overlap_gauge[case['id']] = float(np.max(abs(corrected['potential'][overlap]-exterior_overlap['potential'])/
                    (G*moments['compact_source_mass']/r[overlap])))
                matched = matched_grid(corrected, exterior, R, z, inner=source['inner_join_kpc'], outer=source['outer_join_kpc'])
                cases[case['id']] = matched
                write(f"fields_{variant['id']}_{case['id']}.json", {'case': case, 'matched_fields': matched,
                    'radial_tail_records': details['radial_records'],
                    'overlap_potential_difference_monopole_scaled': overlap_gauge[case['id']],
                    'maximum_raw_potential_correction': float(np.max(abs(details['correction']['potential'])))})
            ref = cases['reference']
            H = np.sqrt(ref['hessian_norm'])
            gm = G*moments['compact_source_mass']
            half = profile['stellar_half_mass_radius_kpc']
            height = min(d.height for d in disks.values())
            scale = {'force': np.maximum(np.linalg.norm(ref['gradient_R_z'], axis=0), 1e-10*gm/half**2),
                'hessian': np.maximum(H, 1e-10*gm/half**3), 'third': np.maximum(ref['third_tensor_norm'], H/(r+height))}
            changes = []
            for name, fields in cases.items():
                if name == 'reference':
                    continue
                errors = differences(fields, ref, scale)
                changes.append({'case': name, 'errors_by_point': errors, 'maximum_errors': {k: float(np.max(v)) for k, v in errors.items()}})
            source_errors = {'density': abs(ref['laplacian']-q)/np.maximum(abs(q), H),
                'density_gradient': np.linalg.norm(ref['gradient_laplacian_R_z']-grad_q, axis=0)/np.maximum(np.linalg.norm(grad_q, axis=0), H/(r+height))}
            worst = {}
            for name, errors in source_errors.items():
                index = np.unravel_index(np.argmax(errors), r.shape)
                worst[name] = {'value': float(errors[index]), 'R_kpc': float(RR[index]), 'z_kpc': float(ZZ[index])}
            reference_old = json.loads((ROOT/config['base_directory']/f"fields_{variant['id']}_reference.json").read_bytes())['matched_fields']
            delta = differences(ref, {k: np.array(v) for k, v in reference_old.items()}, scale)
            targets = source['numerical_targets']
            passed = (all(v < targets[k] for c in changes for k, v in c['maximum_errors'].items())
                      and all(row['value'] < targets[k] for k, row in worst.items())
                      and overlap_gauge['reference'] < targets['potential'])
            compact = {'variant': variant, 'all_grid_points': r.size, 'maximum_refinement_changes': {k: max(c['maximum_errors'][k] for c in changes) for k in scale},
                'worst_source_identity_errors': worst, 'maximum_correction_effect': {k: float(np.max(v)) for k, v in delta.items()},
                'reference_overlap_potential_difference_monopole_scaled': overlap_gauge['reference'],
                'within_all_registered_tail_completion_targets': passed}
            summary.append(compact)
            rows.append({'variant': variant, 'comparisons': changes, 'source_identity_errors': source_errors,
                'correction_effect': delta, 'scales': scale})
            print(json.dumps(serial(compact)), flush=True)
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError('Input changed during tail completion audit')
        write('result.json', {**provenance, 'source_config': source, 'G_kpc_kms2_msun': G, 'records': rows, 'summary': summary,
            'production_interpolant_validated': False, 'new_physical_gravity_rejections': 0})
        write('receipt.json', {'status': 'SOURCE_TAIL_COMPLETION_RETAINED', 'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'status': 'SOURCE_TAIL_COMPLETION_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
