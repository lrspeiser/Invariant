"""Snapshot-based finite-difference verification of every registered field point.

This checks consistency of the numerical potential derivatives. It is not an
independent astrophysical prediction or validation of unregistered space.
"""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'work/gravity-first-principles/hankel-offplane-001')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verifier.py').write_bytes(Path(__file__).read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(serial(value), f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    digest = sha256((args.run/'result.json').read_bytes()).hexdigest()
    result = json.loads((args.run/'result.json').read_bytes())
    registration = {'run_result_sha256': digest, 'method': 'Fourth-order central differences of Hessian, trace and tensor norm at every registered R,z point; source and field code loaded from checked execution snapshots.',
        'step_sizes_kpc': [.001, .0005], 'maximum_fine_normalized_derivative_difference': .0001,
        'axis_extension': 'Use signed Cartesian x through R=0. Diagonal tensor components are even in x; H_xz is odd. Scalar invariants are even. Both sides use the same absolute cylindrical radius.',
        'scope': 'Derivative consistency of the executed finite numerical potential; not independent source data, unbounded-field admission, or gravity-law validation.'}
    write('started.json', registration)
    try:
        assert digest == json.loads((args.run/'receipt.json').read_bytes())['result_sha256']
        snapshots = args.run/'input-snapshots'
        for relative, expected in result['input_hashes'].items():
            assert sha256((snapshots/relative).read_bytes()).hexdigest() == expected, relative
        # Load the preserved package bytes under an isolated name. This avoids
        # depending on current checkout versions or Windows newline conversion.
        package = snapshots/'src/invariant_gravity_extensions'
        alias = '_verified_offplane_execution'
        spec = importlib.util.spec_from_file_location(alias, package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        cylindrical_jet = importlib.import_module(alias+'.hankel_axisymmetric').cylindrical_jet
        Vertical = importlib.import_module(alias+'.vertical_green').Sech2VerticalGreen
        regular_disks = importlib.import_module(alias+'.length_galaxy_development').regular_disks
        config = result['config']
        profile = json.loads((snapshots/config['source_profiles']).read_bytes())['profiles'][-1]
        transform_path = next(t['path'] for t in config['transforms'] if t['radial_nodes'] == 128 and t['wavenumber_nodes'] == 32)
        transform = json.loads((snapshots/transform_path).read_bytes())
        k, w, S = [np.array(transform[key]) for key in ['k', 'wavenumber_weights', 'surface_hankel']]
        G = result['G_kpc_kms2_msun']
        R, z = np.array(config['radii_kpc']), np.array(config['heights_kpc'])
        vertical_config = next(x for x in config['vertical_sources'] if x['id'] == 'fine')
        vertical_source = Vertical(**{k: v for k, v in vertical_config.items() if k != 'id'})
        rows = []
        coefficients = np.array([1., -8., 0., 8., -1.])
        offsets = np.arange(-2, 3)
        for variant in config['variants']:
            _, disks = regular_disks(profile, variant)
            reference = next(r['fields'] for r in result['records'] if r['variant'] == variant and r['case']['id'] == 'reference')
            reference = {key: np.array(value) for key, value in reference.items()}
            scale = np.maximum(reference['third_tensor_norm'], np.sqrt(reference['hessian_norm'])/
                (np.hypot(R[:, None], z[None, :])+min(d.height for d in disks.values())))
            invariant_scale = np.maximum(np.linalg.norm(reference['gradient_hessian_norm_R_z'], axis=0), reference['hessian_norm']/
                (np.hypot(R[:, None], z[None, :])+min(d.height for d in disks.values())))
            for step in registration['step_sizes_kpc']:
                print(f"Finite differences {variant['id']} step={step}", flush=True)
                signed_r = R[:, None]+step*offsets
                stencil_z = z[:, None]+step*offsets
                refined_r, refined_z = np.unique(abs(signed_r)), np.unique(stencil_z)
                r_indices = np.searchsorted(refined_r, abs(signed_r))
                z_indices = np.searchsorted(refined_z, stencil_z)
                r_base, z_base = r_indices[:, 2], z_indices[:, 2]
                vertical = []
                cache = {}
                for name in transform['components']:
                    height = disks[name].height
                    if height not in cache:
                        cache[height] = vertical_source.jet(k*height, refined_z/height)/height**np.arange(4)[:, None, None]
                    vertical.append(cache[height])
                fields = cylindrical_jet(k, w, S, np.array(vertical), refined_r, refined_z, G)
                dR = np.zeros((4, len(R), len(z)))
                dz = np.zeros_like(dR)
                dnorm = np.zeros((2, len(R), len(z)))
                dlap = np.zeros_like(dnorm)
                for j, coefficient in enumerate(coefficients):
                    if coefficient == 0:
                        continue
                    H_r = fields['hessian_RR_Rz_zz_pp'][:, r_indices[:, j]][:, :, z_base].copy()
                    H_r[1] *= np.sign(signed_r[:, j])[:, None]
                    H_z = fields['hessian_RR_Rz_zz_pp'][:, r_base][:, :, z_indices[:, j]]
                    dR += coefficient*H_r/(12*step)
                    dz += coefficient*H_z/(12*step)
                    for destination, key in [(dnorm, 'hessian_norm'), (dlap, 'laplacian')]:
                        destination[0] += coefficient*fields[key][r_indices[:, j]][:, z_base]/(12*step)
                        destination[1] += coefficient*fields[key][r_base][:, z_indices[:, j]]/(12*step)
                T = reference['third_RRR_RRz_Rzz_zzz_Rpp_zpp']
                errors = {'radial_hessian_derivative': np.sqrt(np.einsum('i,irz,irz->rz', [1, 2, 1, 1], dR-T[[0, 1, 2, 4]], dR-T[[0, 1, 2, 4]]))/scale,
                    'vertical_hessian_derivative': np.sqrt(np.einsum('i,irz,irz->rz', [1, 2, 1, 1], dz-T[[1, 2, 3, 5]], dz-T[[1, 2, 3, 5]]))/scale,
                    'trace_gradient': np.linalg.norm(dlap-reference['gradient_laplacian_R_z'], axis=0)/scale,
                    'tensor_norm_gradient': np.linalg.norm(dnorm-reference['gradient_hessian_norm_R_z'], axis=0)/invariant_scale}
                maximum = {key: float(np.max(value)) for key, value in errors.items()}
                rows.append({'variant': variant, 'step_kpc': step, 'points': len(R)*len(z), 'errors_by_probe': errors,
                    'maximum_errors': maximum, 'within_registered_derivative_target': max(maximum.values()) < registration['maximum_fine_normalized_derivative_difference']})
                print(json.dumps(maximum), flush=True)
        fine = [r for r in rows if r['step_kpc'] == min(registration['step_sizes_kpc'])]
        passed = all(r['within_registered_derivative_target'] for r in fine)
        write('result.json', {**registration, 'verified_input_snapshots': len(result['input_hashes']), 'rows': rows,
            'all_fine_stencils_within_target': passed, 'status': 'DERIVATIVE_STENCILS_VERIFIED' if passed else 'DERIVATIVE_STENCIL_DISAGREEMENT_RETAINED'})
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        if not passed:
            raise RuntimeError('Fine stencil disagreement is retained; investigate before promotion')
    except Exception as exc:
        write('failure.json', {'error': repr(exc), 'status': 'OFFPLANE_VERIFICATION_FAILURE_RETAINED'})
        raise


if __name__ == '__main__':
    main()
