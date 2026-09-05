"""Snapshot-based derivative verification of the active tail correction."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def serial(v):
    if isinstance(v, np.ndarray):
        return serial(v.tolist())
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, dict):
        return {k: serial(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [serial(x) for x in v]
    return v


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'work/gravity-first-principles/source-tail-002')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verifier.py').write_bytes(Path(__file__).read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(serial(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    result = json.loads((args.run/'result.json').read_bytes())
    digest = sha256((args.run/'result.json').read_bytes()).hexdigest()
    registration = {'run_result_sha256': digest, 'step_sizes_kpc': [.001, .0005],
        'maximum_fine_scaled_derivative_error': .0001,
        'method': 'Fourth-order finite differences of the active correction potential, its gradient and Hessian, using the preserved execution code. All registered coordinates retained. The correction is multiplied by the exact complementary join weight, so its contribution is zero where the exterior provider is active.',
        'interface_treatment': 'At measured radial knots, core join and taper endpoints, use both left and right fourth-order one-sided stencils: the source is C1 there and a central high-order stencil would straddle a higher-derivative jump. At the axis use signed Cartesian x with the proper component parity. Other points use central stencils.',
        'stencil_sampling': 'Evaluate the union of all required central and interface one-sided stencil coordinates; unused third and fourth offsets away from interfaces need not be evaluated. All original validation points and stencils are retained.',
        'scope': 'Numerical derivative consistency of the newly added active potential correction, normalized by the full matched field. Inherited Hankel and exterior derivatives have separate controls. This is not a full nonlinear action solve or new astronomical validation.'}
    write('started.json', registration)
    try:
        assert digest == json.loads((args.run/'receipt.json').read_bytes())['result_sha256']
        snapshots = args.run/'input-snapshots'
        for p, expected in result['input_hashes'].items():
            assert sha256((snapshots/p).read_bytes()).hexdigest() == expected, p
        package = snapshots/'src/invariant_gravity_extensions'
        alias = '_verified_source_tail_execution'
        spec = importlib.util.spec_from_file_location(alias, package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        complete = importlib.import_module(alias+'.hankel_tail').complete_leading_tail
        blend = importlib.import_module(alias+'.potential_join').blend_potential_jets
        regular_disks = importlib.import_module(alias+'.length_galaxy_development').regular_disks
        source = result['source_config']
        profile = json.loads((snapshots/source['source_profiles']).read_bytes())['profiles'][-1]
        row = next(t for t in source['transforms'] if t['radial_nodes'] == 128 and t['wavenumber_nodes'] == 64)
        transform = json.loads((snapshots/row['path']).read_bytes())
        k, w, S = [np.array(transform[key]) for key in ['k', 'wavenumber_weights', 'surface_hankel']]
        R, z = np.array(source['radii_kpc']), np.array(source['heights_kpc'])
        G = result['G_kpc_kms2_msun']
        precision = result['config'].get('precision')
        precision_options = {'precision': precision} if precision is not None else {}
        rows = []
        stencils = {'central': (np.array([-2, -1, 0, 1, 2]), np.array([1., -8., 0., 8., -1.])/12),
            'right': (np.arange(5), np.array([-25., 48., -36., 16., -3.])/12),
            'left': (-np.arange(5), -np.array([-25., 48., -36., 16., -3.])/12)}
        for variant in source['variants']:
            _, disks = regular_disks(profile, variant)
            edges = np.unique(np.r_[*[d.radius for d in disks.values()], *[d.outer_radius for d in disks.values()],
                                    *[d.outer_radius-d.taper_width for d in disks.values()]])
            interfaces = np.any(np.isclose(R[:, None], edges, rtol=0, atol=1e-12), axis=1)
            scales = next(row['scales'] for row in result['records'] if row['variant'] == variant)
            scales = {k: np.array(v) for k, v in scales.items()}
            for step in registration['step_sizes_kpc']:
                print(f"Correction derivative verification {variant['id']} h={step}", flush=True)
                signed = np.r_[(R[:, None]+step*np.arange(-2, 3)).ravel(),
                    (R[interfaces, None]+step*np.array([-4, -3, 3, 4])).ravel()]
                heights = z[:, None]+step*np.arange(-2, 3)
                rr, zz = np.unique(abs(signed)), np.unique(heights)
                rbase, zbase = np.searchsorted(rr, R), np.searchsorted(zz, z)
                shape = (len(rr), len(zz))
                zero = {'potential': np.zeros(shape), 'gradient_R_z': np.zeros((2,)+shape),
                    'hessian_RR_Rz_zz_pp': np.zeros((4,)+shape), 'third_RRR_RRz_Rzz_zzz_Rpp_zpp': np.zeros((6,)+shape)}
                correction, _ = complete(zero, disks, transform['components'], k, w, S, rr, zz, G, 400., log_nodes=128, **precision_options)
                XR, XZ = np.meshgrid(rr, zz, indexing='ij')
                field = blend(correction, zero, XR, XZ, inner=source['inner_join_kpc'], outer=source['outer_join_kpc'])
                base = {key: value[..., rbase[:, None], zbase[None, :]] for key, value in field.items() if key not in ['radius', 'height']}
                tests = []
                for axis in [0, 1]:
                    for name, (offsets, coefficients) in stencils.items():
                        if axis == 1 and name != 'central':
                            continue
                        active_r = (~interfaces if name == 'central' else interfaces) if axis == 0 else np.ones(len(R), bool)
                        if not np.any(active_r):
                            continue
                        derivatives = {key: np.zeros_like(base[key]) for key in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp']}
                        for offset, coefficient in zip(offsets, coefficients, strict=True):
                            if coefficient == 0:
                                continue
                            xr = R+step*offset if axis == 0 else R
                            hz = z+step*offset if axis == 1 else z
                            ri, zi = np.searchsorted(rr, abs(xr)), np.searchsorted(zz, hz)
                            for key in derivatives:
                                values = field[key][..., ri[:, None], zi[None, :]].copy()
                                if axis == 0:
                                    if key == 'gradient_R_z':
                                        values[0] *= np.sign(xr)[:, None]
                                    elif key == 'hessian_RR_Rz_zz_pp':
                                        values[1] *= np.sign(xr)[:, None]
                                derivatives[key] += coefficient*values/step
                        H, T = base['hessian_RR_Rz_zz_pp'], base['third_RRR_RRz_Rzz_zzz_Rpp_zpp']
                        expected = [base['gradient_R_z'][axis], H[[0, 1] if axis == 0 else [1, 2]],
                            T[[0, 1, 2, 4] if axis == 0 else [1, 2, 3, 5]]]
                        errors = [abs(derivatives['potential']-expected[0])/scales['force'],
                            np.linalg.norm(derivatives['gradient_R_z']-expected[1], axis=0)/scales['hessian'],
                            np.sqrt(np.einsum('i,irz,irz->rz', [1, 2, 1, 1], derivatives['hessian_RR_Rz_zz_pp']-expected[2],
                                derivatives['hessian_RR_Rz_zz_pp']-expected[2]))/scales['third']]
                        maxima = {}
                        for label, e in zip(['gradient', 'hessian', 'third'], errors, strict=True):
                            masked = np.where(active_r[:, None], e, -1.)
                            index = np.unravel_index(np.argmax(masked), e.shape)
                            maxima[label] = {'value': float(e[index]), 'R_kpc': float(R[index[0]]), 'z_kpc': float(z[index[1]])}
                        tests.append({'direction': 'R' if axis == 0 else 'z', 'stencil': name,
                            'points': int(active_r.sum()*len(z)), 'maximum_errors': maxima})
                rows.append({'variant': variant, 'step_kpc': step, 'checks': tests})
                print(json.dumps(serial(rows[-1])), flush=True)
        finest = [row for row in rows if row['step_kpc'] == min(registration['step_sizes_kpc'])]
        passed = all(item['value'] < registration['maximum_fine_scaled_derivative_error'] for row in finest for c in row['checks'] for item in c['maximum_errors'].values())
        write('result.json', {**registration, 'rows': rows, 'verified_input_snapshots': len(result['input_hashes']),
            'all_registered_fine_checks_pass': passed, 'status': 'SOURCE_TAIL_DERIVATIVE_VERIFICATION_RETAINED'})
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
