"""Verify immutable audit inputs and independent adaptive integral spot checks."""
from __future__ import annotations

import argparse
import json
import sys
from hashlib import sha256
from pathlib import Path

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import eval_legendre, j0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.length_galaxy_development import regular_disks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    snapshot = args.output/'verifier.py'
    snapshot.write_bytes(Path(__file__).read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    directories = [ROOT/'work/gravity-first-principles'/x for x in ['angular-source-resolution-002', 'hankel-midplane-001']]
    results = [json.loads((p/'result.json').read_bytes()) for p in directories]
    started = {'method': 'SHA-256 snapshot integrity, adaptive angular-source and radial-Hankel integrals independent of production Gauss rules',
               'audit_result_hashes': {p.name: sha256((p/'result.json').read_bytes()).hexdigest() for p in directories},
               'maximum_normalized_adaptive_disagreement': 1e-7,
               'projection_shells_kpc': [1., 16., 35.], 'projection_orders': [0, 80, 1280, 2560],
               'transform_k_indices': [0, 1600, 6400, 12800, 25599],
               'modified_gravity_test': False, 'full_field_validation': False}
    write('started.json', started)
    try:
        n = 0
        for path, result in zip(directories, results, strict=True):
            receipt = json.loads((path/'receipt.json').read_bytes())
            assert sha256((path/'result.json').read_bytes()).hexdigest() == receipt['result_sha256']
            for relative, digest in result['input_hashes'].items():
                assert sha256((path/'input-snapshots'/relative).read_bytes()).hexdigest() == digest
                n += 1
        # Source construction must still be exactly the executed definition.
        for file in ['length_galaxy_development.py', 'length_axisymmetric.py', 'galaxy_development.py', 'reconstructed_axisymmetric.py']:
            relative = 'src/invariant_gravity_extensions/'+file
            assert sha256((ROOT/relative).read_bytes()).hexdigest() == results[1]['input_hashes'][relative]
        maps = json.loads((directories[1]/'input-snapshots'/results[1]['config']['source_profiles']).read_bytes())
        projection_checks, transform_checks = [], []
        orders = np.array(started['projection_orders'])
        for variant in results[0]['config']['variants']:
            _, disks = regular_disks(maps['profiles'][-1], variant)
            coefficients = json.loads((directories[0]/f"coefficients_{variant['id']}_4096.json").read_bytes())
            for radius in started['projection_shells_kpc']:
                i = coefficients['radius_kpc'].index(radius)
                q0 = coefficients['coefficients'][i][0]

                def integrand(mu, radius=radius, disks=disks, q0=q0):
                    sine = np.sqrt(1-mu*mu)
                    values = [d.density_and_gradient(radius*sine, radius*mu) for d in disks.values()]
                    rho = sum(v[0] for v in values)
                    grad = sum(v[1] for v in values)
                    derivative = sine*grad[0]+mu*grad[1]
                    P = eval_legendre(orders, mu)
                    return np.r_[P*rho/q0, P*derivative/(q0/radius)]

                points = sorted({0., 1., *[min(1., factor*d.height/radius) for factor in [1, 4, 12] for d in disks.values()]})
                value, error = quad_vec(integrand, 0., 1., points=points, epsabs=1e-9, epsrel=1e-9, limit=6000)
                expected = np.r_[np.array(coefficients['coefficients'][i])[orders]/(2*orders+1)/q0,
                                  np.array(coefficients['radial_derivative_coefficients'][i])[orders]/(2*orders+1)/(q0/radius)]
                discrepancy = float(np.max(abs(value-expected)))
                projection_checks.append({'variant': variant, 'radius_kpc': radius, 'orders': orders.tolist(),
                    'adaptive_values': value.tolist(), 'recorded_values': expected.tolist(),
                    'maximum_normalized_difference': discrepancy, 'adaptive_error_estimate': float(error)})
                print(f"Adaptive angular check {variant['id']} R={radius}: {discrepancy:.3g}", flush=True)
                assert discrepancy < started['maximum_normalized_adaptive_disagreement']
        transform_path = directories[1]/'transform_r128_k32.json'
        transform = json.loads(transform_path.read_bytes())
        digest = sha256(transform_path.read_bytes()).hexdigest()
        for record in results[1]['records']:
            assert sha256((directories[1]/record['transform_file']).read_bytes()).hexdigest() == record['transform_sha256']
        _, disks = regular_disks(maps['profiles'][-1], {'id': 'primary'})
        for index in started['transform_k_indices']:
            k = transform['k'][index]
            for j, name in enumerate(transform['components']):
                disk = disks[name]
                scale = transform['component_mass'][j]/(2*np.pi)
                edges = np.array(transform['radial_edges'])

                def integrand(R, disk=disk, k=k, scale=scale):
                    return R*disk.surface(R)*j0(k*R)/scale

                value, error = quad(integrand, 0, disk.outer_radius, points=edges[1:-1], epsabs=2e-10, epsrel=1e-9, limit=6000)
                expected = transform['surface_hankel'][j][index]/scale
                discrepancy = abs(value-expected)
                transform_checks.append({'component': name, 'k_index': index, 'k_kpc_inverse': k,
                    'adaptive_mass_scaled_transform': value, 'recorded_mass_scaled_transform': expected,
                    'normalized_difference': discrepancy, 'adaptive_error_estimate': error})
                assert discrepancy < started['maximum_normalized_adaptive_disagreement']
            print(f'Adaptive radial transform check k={k:.5g}', flush=True)
        write('result.json', {**started, 'verified_input_snapshots': n,
            'projection_coefficient_integrals': 8*len(projection_checks), 'projection_checks': projection_checks,
            'transform_checks': transform_checks, 'finest_transform_sha256': digest,
            'maximum_projection_disagreement': max(r['maximum_normalized_difference'] for r in projection_checks),
            'maximum_transform_disagreement': max(r['normalized_difference'] for r in transform_checks),
            'status': 'HASHES_AND_ADAPTIVE_INTEGRALS_VERIFIED'})
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'status': 'VERIFICATION_FAILURE_RETAINED', 'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
