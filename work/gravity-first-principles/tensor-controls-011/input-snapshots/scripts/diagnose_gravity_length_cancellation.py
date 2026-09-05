"""Synthetic high-precision control for a cancellation-resistant flux difference.

Prototype only; does not change the production action or running galaxy solve.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import mpmath as mp
import numpy as np
from numpy.polynomial.legendre import leggauss

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux


def prototype(spec, p, H, dH2, dlap, length, order):
    """Integral identity for delta P_x, with a0=1 in this control."""
    x, h = np.dot(p, p), length**2*np.sum(H*H)
    abscissa, weights = leggauss(order)
    u = x+h*(abscissa+1)/2
    _, k1, k2, _ = spec.kernel(u)
    delta_px = h*np.dot(weights/2, k1/u+x*k2/u**2)
    _, ph, k1, k2, fraction = spec.partials(x, h)
    dx, dh = 2*H@p, length**2*dH2
    dph = ((k1+fraction*k2)*dx+fraction*k2*dh)/(x+h)
    return delta_px*p-length**2*(H@dph+ph*dlap)


def precise(shape, epsilon, p, H, dH2, dlap, length):
    ctx = mp.mp.clone()
    ctx.dps = 80
    convert = lambda a: ctx.mpf(float(a))
    p, H, dH2, dlap = ctx.matrix(list(map(convert, p))), ctx.matrix([[convert(x) for x in row] for row in H]), ctx.matrix(list(map(convert, dH2))), ctx.matrix(list(map(convert, dlap)))
    ell, eps, m = convert(length), convert(epsilon), convert(shape)
    x, h = sum(v*v for v in p), ell**2*sum(v*v for v in H)

    def kernel(u):
        def s(v):
            return (1+v**(-m))**(-ctx.mpf(3)/(4*m))
        return ctx.mpf(4)/3*(s(u+eps**2)-s(eps**2))/u

    u = x+h
    kp, kpp = ctx.diff(kernel, u), ctx.diff(kernel, u, 2)
    delta_px = kernel(u)+x*kp-kernel(x)-x*ctx.diff(kernel, x)
    ph = x*kp
    dx, dh = 2*H*p, ell**2*dH2
    dph = (kp+x*kpp)*dx+x*kpp*dh
    return np.array([float(v) for v in delta_px*p-ell**2*(H*dph+ph*dlap)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), ROOT/'src/invariant_gravity_extensions/length_screening.py']
    hashes = {p.relative_to(ROOT).as_posix():sha256(p.read_bytes()).hexdigest() for p in paths}
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())

    def write(name, data):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write('\n')

    config = {'scope':'Synthetic polynomial potential, a0=1, nonzero gradients; no axis/saddle/uniform-domain or galaxy admission.',
        'shapes':[.5, 1., 2.], 'lengths':[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, .001, .01],
        'points':[[.3, .4], [1.2, -.7]], 'precision_digits':80,
        'identity':'delta P_x = integral_0^h (K_prime(x+t) + x K_second(x+t)) dt',
        'prototype_orders':[16, 32], 'relative_vector_target':1e-9,
        'not_production_change':True, 'new_observational_scores':0}
    write('started.json', {'config':config, 'input_hashes':hashes, 'started_utc':datetime.now(UTC).isoformat()})
    rows = []
    for R, z in config['points']:
        # psi=R^2+2*z^2+R^2*z^2, Cartesian basis (R,z,phi).
        p = np.array([2*R+2*R*z*z, 4*z+2*R*R*z, 0.])
        H = np.array([[2+2*z*z, 4*R*z, 0.], [4*R*z, 4+2*R*R, 0.], [0., 0., 2+2*z*z]])
        dH2 = np.array([64*R*z*z+8*R*(4+2*R*R), 16*z*(2+2*z*z)+64*R*R*z, 0.])
        dlap = np.array([4*R, 8*z, 0.])
        for shape in config['shapes']:
            spec = LengthScreening(shape, 1e-6)
            zero = anomalous_flux(spec, p, H, dH2, dlap, 0.)
            for length in config['lengths']:
                ref = precise(shape, spec.epsilon, p, H, dH2, dlap, length)
                direct = anomalous_flux(spec, p, H, dH2, dlap, length)-zero
                scale = np.linalg.norm(ref)
                errors = {str(order):float(np.linalg.norm(prototype(spec,p,H,dH2,dlap,length,order)-ref)/scale) for order in [16,32]}
                rows.append({'R':R, 'z':z, 'shape':shape, 'length':length, 'reference':ref.tolist(),
                    'direct_subtraction_relative_error':float(np.linalg.norm(direct-ref)/scale), 'prototype_relative_errors':errors})
    write('result.json', {'config':config, 'rows':rows,
        'prototype_all_sampled_checks_pass':all(e<1e-9 for r in rows for e in r['prototype_relative_errors'].values()),
        'production_changed':False, 'new_observational_scores':0})
    print(json.dumps({'cases':len(rows), 'worst_prototype_relative_error':max(e for r in rows for e in r['prototype_relative_errors'].values()),
        'worst_direct_subtraction_relative_error':max(r['direct_subtraction_relative_error'] for r in rows)}, indent=2))


if __name__ == '__main__':
    main()
