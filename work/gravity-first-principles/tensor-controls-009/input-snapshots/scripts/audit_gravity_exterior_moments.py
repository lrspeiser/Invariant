"""Registered exterior source audit with direct infinite-height spatial checks."""
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
from invariant_gravity_extensions.direct_exterior import direct_disk_fields
from invariant_gravity_extensions.exterior_moments import (
    ExteriorMomentField,
    derivative_tail_bounds,
    disk_exterior_moments,
)
from invariant_gravity_extensions.length_galaxy_development import regular_disks


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


def compare(value, reference, radii, gm):
    h = value['hessian_RR_Rz_zz_pp']-reference['hessian_RR_Rz_zz_pp']
    t = value['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-reference['third_RRR_RRz_Rzz_zzz_Rpp_zpp']
    return {'potential': abs(value['potential']-reference['potential'])/(gm/radii),
        'force': np.linalg.norm(value['gradient_R_z']-reference['gradient_R_z'], axis=0)/(gm/radii**2),
        'hessian': np.sqrt(np.einsum('i,i...,i...->...', [1, 2, 1, 1], h, h))/(gm/radii**3),
        'third_tensor': np.sqrt(np.einsum('i,i...,i...->...', [1, 3, 3, 1, 3, 3], t, t))/(gm/radii**4)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_exterior_moment_audit_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(), *[ROOT/config[k] for k in ['source_profiles', 'source_result', 'predecessor_result']],
        *[ROOT/p for p in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for key in ['source_result', 'predecessor_result']:
        if hashes[config[key]] != config[key+'_sha256']:
            raise ValueError(f'{key} changed')
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(serial(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    provenance = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'new_gravity_parameters': False, 'new_response_scoring': False, 'new_raw_or_reserved_data': False,
        'quality_verified_astronomical_tests': 0}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
            env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'}, capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Exterior analytic controls failed')
        profile = json.loads((ROOT/config['source_profiles']).read_bytes())['profiles'][-1]
        source_result = json.loads((ROOT/config['source_result']).read_bytes())
        predecessor = json.loads((ROOT/config['predecessor_result']).read_bytes())
        G = source_result['config']['units']['G_kpc_kms2_msun']
        radii, mu = np.meshgrid(config['shell_radii_kpc'], config['shell_mu'], indexing='ij')
        radii, mu = radii.ravel(), mu.ravel()
        R, z = radii*np.sqrt(1-mu*mu), radii*mu
        canonical = radii >= config['canonical_minimum_radius_kpc']
        direct_positions = np.array(config['direct_probes_R_z_kpc'])
        direct_radius = np.linalg.norm(direct_positions, axis=1)
        direct_canonical = direct_radius >= config['canonical_minimum_radius_kpc']
        records, summaries = [], []
        for variant in config['variants']:
            print(f"Exterior moments {variant['id']}", flush=True)
            _, disks = regular_disks(profile, variant)
            moments = {}
            for case in config['moment_cases']:
                moments[case['id']] = disk_exterior_moments(disks, maximum_order=max(config['multipole_orders']),
                    scale=config['moment_scale_kpc'], vertical_interval=config['vertical_interval'],
                    **{k: v for k, v in case.items() if k != 'id'})
                write(f"moments_{variant['id']}_{case['id']}.json", moments[case['id']])
            primary = moments['reference']
            gm = G*primary['compact_source_mass']
            fields = {}
            bounds = []
            for order in config['multipole_orders']:
                fields[order] = ExteriorMomentField(primary, G, maximum_order=order).fields(R, z)
                for radius in config['shell_radii_kpc']:
                    ratio = float(np.nextafter(primary['support_radius']/radius, np.inf))
                    bounds.append({'order': order, 'radius_kpc': radius, 'source_radius_ratio_upper': ratio,
                        'monopole_scaled_bounds': derivative_tail_bounds(ratio, order)})
            reference = fields[max(config['multipole_orders'])]
            moment_changes = []
            for case in config['moment_cases']:
                if case['id'] == 'reference':
                    continue
                case_fields = ExteriorMomentField(moments[case['id']], G).fields(R, z)
                delta = compare(case_fields, reference, radii, gm)
                moment_changes.append({'case': case['id'], 'errors_by_point': delta,
                    'maximum_canonical_errors': {k: float(np.max(v[canonical])) for k, v in delta.items()},
                    'maximum_all_point_errors': {k: float(np.max(v)) for k, v in delta.items()}})
            order_changes = []
            for order, value in fields.items():
                delta = compare(value, reference, radii, gm)
                order_changes.append({'order': order, 'errors_by_point': delta,
                    'maximum_canonical_errors': {k: float(np.max(v[canonical])) for k, v in delta.items()}})
            vacuum = max(float(np.max(abs(f['laplacian'][canonical])/(gm/radii[canonical]**3))) for f in fields.values())
            vacuum_gradient = max(float(np.max(np.linalg.norm(f['gradient_laplacian_R_z'][:, canonical], axis=0)/(gm/radii[canonical]**4))) for f in fields.values())
            direct = {}
            for case in config['direct_cases']:
                print(f"Direct spatial integration {variant['id']} {case['id']}", flush=True)
                direct[case['id']] = direct_disk_fields(disks, direct_positions[:, 0], direct_positions[:, 1], G,
                    **{k: v for k, v in case.items() if k != 'id'})
                write(f"direct_{variant['id']}_{case['id']}.json", direct[case['id']])
                print(f"  Completed direct {variant['id']} {case['id']}", flush=True)
            direct_changes = []
            for case in config['direct_cases']:
                if case['id'] == 'reference':
                    continue
                delta = compare(direct[case['id']], direct['reference'], direct_radius, gm)
                direct_changes.append({'case': case['id'], 'errors_by_point': delta,
                    'maximum_canonical_errors': {k: float(np.max(v[direct_canonical])) for k, v in delta.items()},
                    'maximum_all_point_errors': {k: float(np.max(v)) for k, v in delta.items()}})
            direct_exterior = ExteriorMomentField(primary, G).fields(direct_positions[:, 0], direct_positions[:, 1])
            cross = compare(direct_exterior, direct['reference'], direct_radius, gm)
            cross_max = {k: float(np.max(v[direct_canonical])) for k, v in cross.items()}
            # Previously computed Hankel data at four smaller, overlapping radii
            # are stress comparisons only. No new Hankel or velocity fit occurs.
            old = next(row['fields'] for row in predecessor['records'] if row['variant'] == variant and row['case']['id'] == 'reference')
            positions = np.array(config['hankel_overlap_R_z_kpc'])
            ri = [old['radius'].index(r) for r in positions[:, 0]]
            zi = [old['height'].index(z) for z in positions[:, 1]]
            old_field = {k: np.array(old[k])[..., ri, zi] for k in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp', 'third_RRR_RRz_Rzz_zzz_Rpp_zpp']}
            overlap_field = ExteriorMomentField(primary, G).fields(positions[:, 0], positions[:, 1])
            overlap = compare(overlap_field, old_field, np.linalg.norm(positions, axis=1), gm)
            target = config['numerical_targets']
            max_moment = max(v for c in moment_changes for v in c['maximum_canonical_errors'].values())
            bound = next(b['monopole_scaled_bounds'] for b in bounds if b['order'] == max(config['multipole_orders']) and b['radius_kpc'] == config['canonical_minimum_radius_kpc'])
            direct_pass = all(row['maximum_canonical_errors'][key] < target['maximum_direct_'+key+'_scaled_change']
                for row in direct_changes for key in ['force', 'hessian', 'third_tensor'])
            cross_pass = all(cross_max[key] < target['maximum_cross_method_'+key+'_scaled_difference'] for key in ['force', 'hessian', 'third_tensor'])
            passed = (max_moment < target['maximum_moment_refinement_scaled_error'] and direct_pass and cross_pass and
                      bound['third_tensor'] < target['maximum_order64_uniform_third_tail_bound'] and
                      max(vacuum, vacuum_gradient) < target['maximum_vacuum_identity_scaled_error'])
            summary = {'variant': variant, 'compact_support_radius_kpc': primary['support_radius'],
                'physical_vertical_tail_mass_fraction': primary['physical_vertical_tail_mass_fraction'],
                'maximum_moment_case_canonical_change': max_moment, 'uniform_series_tail_bound_at_admission_radius': bound,
                'maximum_direct_refinement_changes': {key: max(c['maximum_canonical_errors'][key] for c in direct_changes) for key in cross_max},
                'maximum_canonical_cross_method_difference': cross_max, 'maximum_all_direct_probe_cross_method_difference': {k: float(np.max(v)) for k, v in cross.items()},
                'maximum_hankel_stress_difference': {k: float(np.max(v)) for k, v in overlap.items()},
                'maximum_vacuum_trace_error': vacuum, 'maximum_vacuum_gradient_error': vacuum_gradient,
                'within_registered_exterior_targets': passed}
            summaries.append(summary)
            records.append({'variant': variant, 'shell_radius': radii, 'shell_mu': mu, 'fields_by_order': fields,
                'bounds': bounds, 'moment_changes': moment_changes, 'order_changes': order_changes,
                'direct_changes': direct_changes, 'cross_method_errors': cross, 'hankel_overlap_errors': overlap})
            print(json.dumps(serial(summary)), flush=True)
        if any(sha256((ROOT/p).read_bytes()).hexdigest() != digest for p, digest in hashes.items()):
            raise RuntimeError('Input changed during exterior audit')
        write('result.json', {**provenance, 'G_kpc_kms2_msun': G, 'records': records, 'summary': summaries,
            'physical_gravity_rejection': False, 'complete_isolated_source_or_nonlinear_solver': False})
        write('receipt.json', {'status': 'EXTERIOR_REFERENCE_AND_BOUND_RETAINED',
            'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'status': 'EXTERIOR_AUDIT_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
