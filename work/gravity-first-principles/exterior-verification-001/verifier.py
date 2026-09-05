"""Verify exterior evidence from immutable execution snapshots."""
from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import sys
from hashlib import sha256
from pathlib import Path

import mpmath as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'work/gravity-first-principles/exterior-moment-001')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/'verifier.py').write_bytes(Path(__file__).read_bytes())

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    digest = sha256((args.run/'result.json').read_bytes()).hexdigest()
    result = json.loads((args.run/'result.json').read_bytes())
    registration = {'source_result_sha256': digest, 'fractional_stencil_steps': [.0005, .00025],
        'maximum_fine_stencil_monopole_scaled_error': 1e-6,
        'maximum_high_precision_moment_third_error_bound_at_80_kpc': 1e-10,
        'scope': 'Execution-byte checks, 80-digit recombination of stored source moment integrals, and fourth-order differences of field derivatives at every exterior shell point. This is not new source quadrature or an independent astronomical test.'}
    write('started.json', registration)
    try:
        assert digest == json.loads((args.run/'receipt.json').read_bytes())['result_sha256']
        snapshots = args.run/'input-snapshots'
        for relative, expected in result['input_hashes'].items():
            assert sha256((snapshots/relative).read_bytes()).hexdigest() == expected, relative
        package = snapshots/'src/invariant_gravity_extensions'
        alias = '_verified_exterior_execution'
        spec = importlib.util.spec_from_file_location(alias, package/'__init__.py', submodule_search_locations=[str(package)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[alias] = module
        spec.loader.exec_module(module)
        Field = importlib.import_module(alias+'.exterior_moments').ExteriorMomentField
        mp.mp.dps = 80
        rows, moment_rows = [], []
        for record in result['records']:
            name = record['variant']['id']
            moments = json.loads((args.run/f'moments_{name}_reference.json').read_bytes())
            recombined = []
            for l in range(0, 65):
                value = mp.mpf(0)
                if l % 2 == 0:
                    for c in moments['components']:
                        value += 2*mp.pi*mp.fsum([(-1)**j*mp.mpf(math.comb(l, 2*j))*math.comb(2*j, j)/mp.mpf(4)**j
                            *mp.mpf(c['radial_moments'][j])*mp.mpf(c['vertical_moments'][l//2-j]) for j in range(l//2+1)])
                recombined.append(value)
            bound = mp.fsum([abs(v-mp.mpf(moments['scaled_multipole_moments'][l]))
                *(mp.mpf(moments['scale'])/80)**l*(l+3)**6 for l, v in enumerate(recombined)])/mp.mpf(moments['compact_source_mass'])
            moment_rows.append({'variant': name, 'precision_digits': 80, 'third_tensor_rounding_bound_at_80_kpc': float(bound)})
            r, mu = np.array(record['shell_radius']), np.array(record['shell_mu'])
            R, z = r*np.sqrt(1-mu*mu), r*mu
            field = Field(moments, result['G_kpc_kms2_msun'])
            base = field.fields(R, z)
            gm = field.G*moments['compact_source_mass']
            for fraction in registration['fractional_stencil_steps']:
                step = r*fraction
                derivatives = []
                for axis in [0, 1]:
                    accum = {k: np.zeros_like(base[k]) for k in ['potential', 'gradient_R_z', 'hessian_RR_Rz_zz_pp', 'hessian_norm']}
                    for offset, coefficient in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
                        signed_R = R+offset*step if axis == 0 else R
                        value = field.fields(abs(signed_R), z+offset*step if axis == 1 else z)
                        value['gradient_R_z'][0] *= np.sign(signed_R)
                        value['hessian_RR_Rz_zz_pp'][1] *= np.sign(signed_R)
                        for k in accum:
                            accum[k] += coefficient*value[k]/(12*step)
                    derivatives.append(accum)
                h = base['hessian_RR_Rz_zz_pp']
                t = base['third_RRR_RRz_Rzz_zzz_Rpp_zpp']
                expected_h = [h[[0, 1]], h[[1, 2]]]
                expected_t = [t[[0, 1, 2, 4]], t[[1, 2, 3, 5]]]
                errors = {'gradient': max(float(np.max(abs(d['potential']-base['gradient_R_z'][i])/(gm/r**2))) for i, d in enumerate(derivatives)),
                    'hessian': max(float(np.max(np.linalg.norm(d['gradient_R_z']-expected_h[i], axis=0)/(gm/r**3))) for i, d in enumerate(derivatives)),
                    'third': max(float(np.max(np.linalg.norm(d['hessian_RR_Rz_zz_pp']-expected_t[i], axis=0)/(gm/r**4))) for i, d in enumerate(derivatives)),
                    'gradient_hessian_norm': max(float(np.max(abs(d['hessian_norm']-base['gradient_hessian_norm_R_z'][i])/(gm*gm/r**7))) for i, d in enumerate(derivatives))}
                rows.append({'variant': name, 'fractional_step': fraction, 'points': len(r), 'maximum_errors': errors})
                print(json.dumps(rows[-1]), flush=True)
        fine = [r for r in rows if r['fractional_step'] == min(registration['fractional_stencil_steps'])]
        passed = (all(v < registration['maximum_fine_stencil_monopole_scaled_error'] for r in fine for v in r['maximum_errors'].values())
            and all(r['third_tensor_rounding_bound_at_80_kpc'] < registration['maximum_high_precision_moment_third_error_bound_at_80_kpc'] for r in moment_rows))
        write('result.json', {**registration, 'rows': rows, 'high_precision_moments': moment_rows,
            'verified_input_snapshots': len(result['input_hashes']), 'all_registered_checks_pass': passed,
            'status': 'EXTERIOR_SNAPSHOT_DERIVATIVE_VERIFICATION_RETAINED'})
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
