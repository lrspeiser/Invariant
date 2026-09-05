"""Registered near/far potential join audit, without gravity-card scoring."""
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
from invariant_gravity_extensions.hankel_midplane import disk_transforms, piecewise_gauss
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.potential_join import blend_potential_jets
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


def norm(v, weights):
    return np.sqrt(np.einsum('i,i...,i...->...', weights, v, v))


def difference(v, f, scale):
    return {'force': np.linalg.norm(v['gradient_R_z']-f['gradient_R_z'], axis=0)/scale['force'],
        'hessian': norm(v['hessian_RR_Rz_zz_pp']-f['hessian_RR_Rz_zz_pp'], [1, 2, 1, 1])/scale['hessian'],
        'third': norm(v['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-f['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], [1, 3, 3, 1, 3, 3])/scale['third']}


def select(fields, mask):
    return {k: np.array(v)[..., mask] for k, v in fields.items() if k not in ['radius', 'height']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_potential_join_audit_v1.json')
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
    for p, expected in config['input_files'].items():
        if hashes[p] != expected:
            raise ValueError(f'Input changed: {p}')
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
        controls = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'exit_code': controls.returncode, 'stdout': controls.stdout, 'stderr': controls.stderr})
        if controls.returncode:
            raise RuntimeError('Potential join analytic controls failed')
        profile = json.loads((ROOT/config['source_profiles']).read_bytes())['profiles'][-1]
        G = json.loads((ROOT/config['source_result']).read_bytes())['config']['units']['G_kpc_kms2_msun']
        _, primary_disks = regular_disks(profile, {'id': 'primary'})
        transforms = {}
        for row in config['transforms']:
            key = (row['radial_nodes'], row['wavenumber_nodes'])
            if 'path' in row:
                transform = json.loads((ROOT/row['path']).read_bytes())
                transform = {k: np.array(v) if k in ['k', 'wavenumber_weights', 'surface_hankel'] else v for k, v in transform.items()}
            else:
                print(f'New radial transform {key}', flush=True)
                k, w = piecewise_gauss(np.arange(0, config['cutoff']+.25, .5), row['wavenumber_nodes'])
                transform = {**disk_transforms(primary_disks, k, row['radial_nodes']), 'wavenumber_weights': w}
                write(f'transform_r{key[0]}_k{key[1]}.json', transform)
            transforms[key] = transform
        R, z = np.array(config['radii_kpc']), np.array(config['heights_kpc'])
        RR, ZZ = np.meshgrid(R, z, indexing='ij')
        r = np.hypot(RR, ZZ)
        inside = r <= config['outer_join_kpc']
        join = (r >= config['inner_join_kpc']) & inside
        rows, summary = [], []
        for variant in config['variants']:
            _, disks = regular_disks(profile, variant)
            moment = json.loads((ROOT/config['moment_files'][variant['id']]).read_bytes())
            gm = G*moment['compact_source_mass']
            far = ExteriorMomentField(moment, G, minimum_radius=config['inner_join_kpc']).fields(RR[join], ZZ[join])
            physical = [d.density_and_gradient(RR, ZZ) for d in disks.values()]
            q = 4*np.pi*G*sum(v[0] for v in physical)
            grad_q = 4*np.pi*G*sum(v[1] for v in physical)
            cases = {}
            for case in config['cases']:
                print(f"Near/far {variant['id']} {case['id']}", flush=True)
                transform = transforms[(case['radial_nodes'], case['wavenumber_nodes'])]
                mask = transform['k'] < case['cutoff']
                k, w, S = transform['k'][mask], transform['wavenumber_weights'][mask], transform['surface_hankel'][:, mask]
                vertical_source = Sech2VerticalGreen(intervals=case['vertical_intervals'], extent=case['vertical_extent'])
                cache = {}
                for h in {d.height for d in disks.values()}:
                    cache[h] = vertical_source.jet(k*h, z/h)/h**np.arange(4)[:, None, None]
                vertical = np.array([cache[disks[name].height] for name in transform['components']])
                fields = cylindrical_jet(k, w, S, vertical, R, z, G)
                del vertical, cache
                blended = blend_potential_jets(select(fields, join), far, RR[join], ZZ[join],
                    inner=config['inner_join_kpc'], outer=config['outer_join_kpc'])
                cases[case['id']] = {'case': case, 'near_fields': fields, 'joined_fields': blended}
                write(f"fields_{variant['id']}_{case['id']}.json", cases[case['id']])
            ref = cases['reference']['near_fields']
            hnorm = np.sqrt(ref['hessian_norm'])
            half = profile['stellar_half_mass_radius_kpc']
            height = min(d.height for d in disks.values())
            scale = {'force': np.maximum(np.linalg.norm(ref['gradient_R_z'], axis=0), 1e-10*gm/half**2),
                'hessian': np.maximum(hnorm, 1e-10*gm/half**3),
                'third': np.maximum(ref['third_tensor_norm'], hnorm/(r+height))}
            qscale = np.maximum(abs(q), hnorm)
            grad_qscale = np.maximum(np.linalg.norm(grad_q, axis=0), hnorm/(r+height))
            comparisons = []
            for name, value in cases.items():
                if name == 'reference':
                    continue
                near_diff = difference(value['near_fields'], ref, scale)
                join_diff = difference(value['joined_fields'], cases['reference']['joined_fields'], {k: v[join] for k, v in scale.items()})
                comparisons.append({'case': name, 'near_errors': near_diff, 'joined_errors': join_diff,
                    'maximum_near_changes': {k: float(np.max(v[inside])) for k, v in near_diff.items()},
                    'maximum_joined_changes': {k: float(np.max(v)) for k, v in join_diff.items()}})
            f = cases['reference']['joined_fields']
            errors = {'near_density': abs(ref['laplacian']-q)/qscale,
                'near_density_gradient': np.linalg.norm(ref['gradient_laplacian_R_z']-grad_q, axis=0)/grad_qscale,
                'joined_density': abs(f['laplacian']-q[join])/qscale[join],
                'joined_density_gradient': np.linalg.norm(f['gradient_laplacian_R_z']-grad_q[:, join], axis=0)/grad_qscale[join]}
            maxima = {k: float(np.max(v[inside] if k.startswith('near') else v)) for k, v in errors.items()}
            cross = difference(select(ref, join), far, {k: v[join] for k, v in scale.items()})
            # Direct comparison of potentials matters: a gauge mismatch can
            # create a force in the join despite perfectly equal input forces.
            potential_cross = abs(select(ref, join)['potential']-far['potential'])/(gm/r[join])
            target = config['numerical_targets']
            passed = (all(c[m][k] < target[k] for c in comparisons for m in ['maximum_near_changes', 'maximum_joined_changes'] for k in scale)
                and all(v < target['density_gradient' if k.endswith('gradient') else 'density'] for k, v in maxima.items())
                and all(float(np.max(v)) < target[k] for k, v in cross.items())
                and float(np.max(potential_cross)) < target['potential'])
            compact = {'variant': variant, 'near_points': int(inside.sum()), 'join_points': int(join.sum()),
                'maximum_source_identity_errors': maxima, 'maximum_near_far_difference': {k: float(np.max(v)) for k, v in cross.items()},
                'maximum_potential_difference_monopole_scaled': float(np.max(potential_cross)),
                'maximum_refinement_changes': {k: max(c[m][k] for c in comparisons for m in ['maximum_near_changes', 'maximum_joined_changes']) for k in scale},
                'within_all_registered_join_targets': passed}
            summary.append(compact)
            rows.append({'variant': variant, 'comparisons': comparisons, 'source_identity_errors': errors,
                'near_far_errors': cross, 'potential_errors': potential_cross, 'scales': scale})
            print(json.dumps(compact), flush=True)
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError('Input changed during potential join audit')
        write('result.json', {**provenance, 'G_kpc_kms2_msun': G, 'records': rows, 'summary': summary,
            'new_physical_gravity_rejections': 0, 'production_interpolant_validated': False})
        write('receipt.json', {'status': 'POTENTIAL_JOIN_AUDIT_RETAINED', 'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'status': 'POTENTIAL_JOIN_EXECUTION_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
