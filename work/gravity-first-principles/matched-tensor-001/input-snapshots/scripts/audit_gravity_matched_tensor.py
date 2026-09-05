"""Joined interpolation, physical-source identities and independent derivatives."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.length_galaxy_development import regular_disks
from invariant_gravity_extensions.matched_axisymmetric import matched_grid
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential


def serial(v):
    if isinstance(v, np.ndarray):
        return serial(v.tolist())
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [serial(x) for x in v]
    return v


def norm(values, weights):
    return np.sqrt(np.einsum('i,i...,i...->...', weights, values, values))


def scales(f, R, z, h, acceleration, half_radius):
    H = np.sqrt(f['hessian_norm'])
    return {'force': np.maximum(np.linalg.norm(f['gradient_R_z'], axis=0), 1e-10*acceleration),
        'hessian': np.maximum(H, 1e-10*acceleration/half_radius),
        'third': np.maximum(f['third_tensor_norm'], H/(np.hypot(R, z)+h))}


def source_errors(f, disks, R, z, G, h):
    physical = [d.density_and_gradient(R, z) for d in disks.values()]
    q, dq = 4*np.pi*G*sum(v[0] for v in physical), 4*np.pi*G*sum(v[1] for v in physical)
    H = np.sqrt(f['hessian_norm'])
    return {'density': abs(f['laplacian']-q)/np.maximum(abs(q), H),
        'density_gradient': np.linalg.norm(f['gradient_laplacian_R_z']-dq, axis=0)/
            np.maximum(np.linalg.norm(dq, axis=0), H/(np.hypot(R, z)+h))}


def maxima(errors, R, z):
    rows = {}
    for key, values in errors.items():
        index = np.unravel_index(np.argmax(values), values.shape)
        rows[key] = {'value': float(values[index]), 'R_kpc': float(R[index]), 'z_kpc': float(z[index])}
    return rows


def derivative_check(provider, R, z, step, scale):
    base = provider.fields(R, z)
    derivatives = []
    for axis in range(2):
        accum = {key: np.zeros_like(base[key]) for key in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp']}
        for offset, weight in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
            x, zz = (R+offset*step, z) if axis == 0 else (R, z+offset*step)
            f = provider.fields(abs(x), zz)
            # Extend along signed Cartesian x across the cylindrical axis.
            sign = np.where(x < 0, -1., 1.)
            f['gradient_R_z'][0] *= sign
            f['hessian_RR_Rz_zz_pp'][1] *= sign
            for key in accum:
                accum[key] += weight*f[key]/(12*step)
        derivatives.append(accum)
    dx, dz = derivatives
    p = np.array([dx['potential'], dz['potential']])
    H = np.array([dx['gradient_R_z'][0], dz['gradient_R_z'][0], dz['gradient_R_z'][1]])
    T = np.array([dx['hessian_RR_Rz_zz_pp'][0], dz['hessian_RR_Rz_zz_pp'][0],
        dx['hessian_RR_Rz_zz_pp'][2], dz['hessian_RR_Rz_zz_pp'][2],
        dx['hessian_RR_Rz_zz_pp'][3], dz['hessian_RR_Rz_zz_pp'][3]])
    return {'force': np.linalg.norm(p-base['gradient_R_z'], axis=0)/scale['force'],
        'hessian': norm(H-base['hessian_RR_Rz_zz_pp'][:3], [1, 2, 1])/scale['hessian'],
        'third': norm(T-base['third_RRR_RRz_Rzz_zzz_Rpp_zpp'], [1, 3, 3, 1, 3, 3])/scale['third']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    root = ROOT/'work/gravity-first-principles'
    variants = [('primary', 'tensor-source-003', '41f956064f275472550a03a3a663b792a7171b9d25f68aa08b8e331f6a17d9f0'),
                ('height_half', 'tensor-source-004', '44965a3659efa9dc8f2129452736a2be7b63c00f1804d080e7a076608ac72e62')]
    config = {'inner_kpc': 60., 'outer_kpc': 80., 'targets': {'force': 1e-4, 'hessian': .002,
        'third': .01, 'density': .002, 'density_gradient': .01}, 'fine_derivative_target': 1e-4,
        'derivative_steps_kpc': [.001, .0005],
        'derivative_radii_kpc': [0., .125, .25, .75, 1.25, 12.25, 30., 32.75, 34., 35.75, 36., 36.0625,
            40., 59.999, 60., 60.001, 65., 77., 79.999, 80., 80.001, 90.],
        'derivative_heights_kpc': [-.1, 0., .025, .1, .5, 1., 32., 60., 80.],
        'scope': 'Retained 1364 source probes plus fixed 198 derivative/source probes per thickness; no quadrature refinement or Poisson solve',
        'derivative_scope': 'Scalar-to-meridional-gradient, gradient-to-meridional-Hessian, and Hessian-to-all-six-third-components; symmetry extension at axis',
        'new_observational_scores': 0, 'full_source_admitted': False}
    profile_path, units_path = root/'map-source-003/source_profiles.json', root/'map-source-003/result.json'
    tail_config = ROOT/'configs/gravity_source_tail_audit_v2.json'
    prior = json.loads(tail_config.read_bytes())
    paths = [Path(__file__), profile_path, units_path, tail_config,
        ROOT/'tests/test_gravity_matched_tensor.py', *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    for variant, folder, expected in variants:
        if sha256((root/folder/'result.json').read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen source pilot changed')
        paths.extend([root/folder/name for name in ['result.json','mixed_table.json','direct_reference.json']])
        paths.append(root/f'exterior-moment-002/moments_{variant}_reference.json')
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    for p in [profile_path, units_path, *[root/f'exterior-moment-002/moments_{v}_reference.json' for v, _, _ in variants]]:
        if hashes[p.relative_to(ROOT).as_posix()] != prior['input_files'][p.relative_to(ROOT).as_posix()]:
            raise ValueError('Frozen physical source or exterior changed')
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(serial(value), handle, indent=2, allow_nan=False)
            handle.write('\n')

    write('started.json', {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat()})
    try:
        profile = json.loads(profile_path.read_bytes())['profiles'][-1]
        G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
        summaries = []
        for variant, folder, _ in variants:
            print(f'Joined tensor source: {variant}', flush=True)
            pilot = json.loads((root/folder/'result.json').read_bytes())
            source_config = pilot['config']
            _, disks = regular_disks(profile, source_config['variant'])
            minimum_height = min(d.height for d in disks.values())
            moments = json.loads((root/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            exterior = ExteriorMomentField(moments, G, minimum_radius=60.)
            half_radius = profile['stellar_half_mass_radius_kpc']
            acceleration = G*moments['compact_source_mass']/half_radius**2
            table = json.loads((root/folder/'mixed_table.json').read_bytes())
            tr, tz, mixed = [np.array(table[key]) for key in ['radius_kpc','height_kpc','mixed']]
            direct = {key: np.array(v) for key, v in json.loads((root/folder/'direct_reference.json').read_bytes()).items()}
            r, z = direct['radius'], direct['height']
            R, Z = np.meshgrid(r, z, indexing='ij')
            reference = matched_grid(direct, exterior, r, z, inner=60., outer=80.)
            scale = scales(reference, R, Z, minimum_height, acceleration, half_radius)
            rows = []
            for grid in ['coarse','fine']:
                gr, gz = np.array(source_config['coarse_radii_kpc']), np.array(source_config['coarse_heights_kpc'])
                if grid == 'fine':
                    gr, gz = [np.sort(np.r_[a, a[:-1]+np.diff(a)/2]) for a in [gr, gz]]
                ri, zi = np.searchsorted(tr, gr), np.searchsorted(tz, gz)
                assert np.array_equal(tr[ri], gr) and np.array_equal(tz[zi], gz)
                provider = MatchedTensorPotential(gr, gz, mixed[:, :, ri][:, :, :, zi], exterior)
                value = provider.fields(R, Z)
                errors = {'force': np.linalg.norm(value['gradient_R_z']-reference['gradient_R_z'],axis=0)/scale['force'],
                    'hessian': norm(value['hessian_RR_Rz_zz_pp']-reference['hessian_RR_Rz_zz_pp'],[1,2,1,1])/scale['hessian'],
                    'third': norm(value['third_RRR_RRz_Rzz_zzz_Rpp_zpp']-reference['third_RRR_RRz_Rzz_zzz_Rpp_zpp'],[1,3,3,1,3,3])/scale['third']}
                identities = source_errors(value, disks, R, Z, G, minimum_height)
                row = {'grid': grid, 'interpolation': maxima(errors,R,Z), 'source_identities': maxima(identities,R,Z)}
                row['within_interpolation_and_source_targets'] = all(v['value'] < config['targets'][k]
                    for group in [row['interpolation'],row['source_identities']] for k,v in group.items())
                write(f'joined_{variant}_{grid}.json', {'fields':value,'interpolation_errors':errors,'source_errors':identities})
                if grid == 'fine':
                    DR, DZ = np.meshgrid(config['derivative_radii_kpc'],config['derivative_heights_kpc'],indexing='ij')
                    df = provider.fields(DR,DZ)
                    ds = scales(df,DR,DZ,minimum_height,acceleration,half_radius)
                    derivative_rows = []
                    for step in config['derivative_steps_kpc']:
                        errors = derivative_check(provider,DR,DZ,step,ds)
                        derivative_rows.append({'step_kpc':step,'errors':maxima(errors,DR,DZ)})
                        write(f'derivatives_{variant}_{step}.json',{'R_kpc':DR,'z_kpc':DZ,'errors':errors})
                    row['independent_derivatives'] = derivative_rows
                    row['fine_derivative_checks_pass'] = all(v['value'] < config['fine_derivative_target']
                        for v in derivative_rows[-1]['errors'].values())
                    row['additional_source_probes'] = maxima(source_errors(df,disks,DR,DZ,G,minimum_height),DR,DZ)
                rows.append(row)
            summaries.append({'variant':variant,'rows':rows,'direct_reference_source_identities':
                maxima(source_errors(reference,disks,R,Z,G,minimum_height),R,Z)})
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json',{'config':config,'summaries':summaries,'completed_utc':datetime.now(UTC).isoformat(),
            'full_source_admitted':False,'quadrature_refinement':'pending','new_observational_scores':0})
        print(json.dumps(summaries,indent=2),flush=True)
    except Exception as exc:
        write('failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
