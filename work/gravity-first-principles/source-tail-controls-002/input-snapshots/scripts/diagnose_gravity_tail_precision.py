"""Separate cancellation error from quadrature error at an exterior tail point."""
from __future__ import annotations

import argparse
import json
import math
import sys
from hashlib import sha256
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy.special import j0

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.hankel_tail import DiskLogGreen, radial_tail_jet
from invariant_gravity_extensions.length_galaxy_development import regular_disks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    transform_path = ROOT/'work/gravity-first-principles/potential-join-001/transform_r128_k64.json'
    profile_path = ROOT/'work/gravity-first-principles/map-source-003/source_profiles.json'
    inputs = [Path(__file__), transform_path, profile_path, *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs}
    registration = {'radius_kpc': 66.5, 'step_kpc': .0005, 'precision_digits': 50,
        'accurate_bessel_cutoffs': [2., 8.], 'input_hashes': hashes,
        'scope': 'Diagnostic at an already exposed failing point outside the physical radial source. Same fixed transform and total mass. Higher precision in low-k Bessel values and cancellation arithmetic, compensated high-k sum. No source change, no admission gate or observational scoring.'}

    def write(name, value):
        (args.output/name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n', encoding='utf8', newline='\n')

    write('started.json', registration)
    for p in inputs:
        destination = args.output/'input-snapshots'/p.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(p.read_bytes())
    try:
        mp.mp.dps = registration['precision_digits']
        d = json.loads(transform_path.read_bytes())
        k, w, S = [np.array(d[key]) for key in ['k', 'wavenumber_weights', 'surface_hankel']]
        _, disks = regular_disks(json.loads(profile_path.read_bytes())['profiles'][-1], {'id': 'primary'})
        radius, step = registration['radius_kpc'], registration['step_kpc']
        rr = radius+step*np.arange(-2, 3)
        rows = []
        for name, transform in zip(d['components'], S, strict=True):
            disk = disks[name]
            assert rr.min() > disk.outer_radius
            ordinary = radial_tail_jet(disk, k, w, transform, rr, 400.)
            total = mp.mpf(float(DiskLogGreen(disk).prefix_mass[-1]))
            # This form is exact for R beyond the radial source. Numerical
            # inputs remain the recorded binary floats, embedded exactly in mp.
            constant = mp.log(mp.mpf(2)/400)-mp.euler+mp.fsum(mp.mpf(float(a))/float(b) for a, b in zip(w, k, strict=True))
            row = {'component': name, 'analytic_first': float(ordinary['first'][2]),
                'ordinary_potential': ordinary['potential'].tolist(), 'precision_cases': []}
            row['ordinary_finite_first'] = float(np.dot([1, -8, 0, 8, -1], ordinary['potential'])/(12*step))
            for limit in registration['accurate_bessel_cutoffs']:
                low = k < limit
                potentials = []
                for r in rr:
                    high_sum = math.fsum(w[~low]*transform[~low]*j0(k[~low]*r)/k[~low])
                    low_sum = mp.fsum(mp.mpf(float(a))*float(b)*mp.besselj(0, mp.mpf(float(c))*float(r))/float(c)
                        for a, b, c in zip(w[low], transform[low], k[low], strict=True))
                    potentials.append(total*(constant-mp.log(float(r)))-low_sum-mp.mpf(high_sum))
                first = mp.fsum(c*p for c, p in zip([1, -8, 0, 8, -1], potentials, strict=True))/(12*mp.mpf(step))
                row['precision_cases'].append({'low_k_accurate_limit': limit, 'finite_first': float(first),
                    'finite_first_minus_analytic': float(first)-row['analytic_first'],
                    'potential': [float(p) for p in potentials]})
            rows.append(row)
            print(json.dumps(row), flush=True)
        assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
        write('result.json', {**registration, 'rows': rows})
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
    except Exception as exc:
        write('failure.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
