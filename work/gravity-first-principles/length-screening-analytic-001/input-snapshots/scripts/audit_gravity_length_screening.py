"""Derive spherical higher-derivative forces before choosing physical lengths."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import sympy as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.saturated_actions import saturated_q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_length_screening_audit_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    files = [Path(__file__), args.config, ROOT/'src/invariant_gravity_extensions/actions.py',
             ROOT/'src/invariant_gravity_extensions/saturated_actions.py']
    hashes = {}
    for path in files:
        relative = path.resolve().relative_to(ROOT)
        data = path.read_bytes()
        hashes[relative.as_posix()] = sha256(data).hexdigest()
        target = args.output/'input-snapshots'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    provenance = {'started_utc': datetime.now(UTC).isoformat(), 'input_hashes': hashes,
                  'config': json.loads(args.config.read_bytes()), 'sympy': sp.__version__, 'observations_accessed': False}
    (args.output/'started.json').write_text(json.dumps(provenance, indent=2)+'\n', newline='\n')
    try:
        r, ell, gm, a0, x, h, u, eps, C = sp.symbols('r ell GM a0 x h u epsilon C', positive=True)
        g = sp.Function('g')(r)
        gp = sp.diff(g, r)
        substitution = {x: g**2/a0**2, h: ell**2*(gp**2+2*(g/r)**2)/a0**2}
        P = sp.Function('P')(x, h)
        px = sp.diff(P, x).subs(substitution)
        ph = sp.diff(P, h).subs(substitution)
        radial_flux = px*g-ell**2*(sp.diff(ph*gp, r)+2*ph*(gp-g/r)/r)
        radial_action = r**2*a0**2*P.subs(substitution)/2
        # Radial Euler-Lagrange variation is independent of the tensor-divergence
        # reduction used for radial_flux above.
        varied = (sp.diff(radial_action, g)-sp.diff(sp.diff(radial_action, gp), r))/r**2
        variation_identity = sp.simplify(varied-radial_flux)
        checks = {'radial_tensor_flux_matches_reduced_action': variation_identity == 0}
        point_g = gm/r**2
        point_substitution = {x: point_g**2/a0**2, h: 6*ell**2*point_g**2/(a0**2*r**2)}

        def point_anomaly(action):
            dpx = (sp.diff(action, x)-1).subs(point_substitution)
            dph = sp.diff(action, h).subs(point_substitution)
            dg = sp.diff(point_g, r)
            return sp.factor(dpx*point_g-ell**2*(sp.diff(dph*dg, r)+2*dph*(dg-point_g/r)/r))

        old_action = x+sp.Rational(4, 3)*x/(x+h)**sp.Rational(1, 4)
        old_force = point_anomaly(old_action)
        expected_old = sp.sqrt(gm*a0)*(102*ell**4+40*ell**2*r**2+3*r**4)/(3*sp.sqrt(r)*(6*ell**2+r**2)**sp.Rational(9, 4))
        checks['original_unregularized_point_formula'] = sp.simplify(old_force-expected_old) == 0
        checks['original_zero_length_limit'] = sp.simplify(sp.limit(old_force, ell, 0)-sp.sqrt(gm*a0)/r) == 0
        small_system_coefficient = sp.simplify(sp.limit(old_force*sp.sqrt(ell*r)/sp.sqrt(gm*a0), ell, sp.oo))
        checks['original_compact_system_coefficient'] = sp.simplify(small_system_coefficient-sp.Rational(17, 108)*6**sp.Rational(3, 4)) == 0
        checks['newtonian_flux'] = point_anomaly(x) == 0
        rows = []
        for m in [sp.Rational(1, 2), sp.Integer(1), sp.Integer(2)]:
            q = saturated_q(u, eps, m)
            excess = q-u
            K = excess/u
            proposed = x+x*K.subs(u, x+h)
            zero_h = sp.simplify(proposed.subs(h, 0)-saturated_q(x, eps, m))
            K0 = sp.simplify(sp.diff(excess, u).subs(u, 0))
            expected_K0 = eps**(-sp.Rational(1, 2))*(1+eps**(2*m))**(-1-sp.Rational(3, 4)/m)
            checks[f'shape_{m}_zero_h_recovers_scalar'] = zero_h == 0
            checks[f'shape_{m}_removable_K_origin'] = sp.simplify(K0-expected_K0) == 0
            rows.append({'shape': str(m), 'proposed_action': str(proposed), 'K_origin': str(K0),
                         'bounded_excess_high_u_limit': str(sp.Rational(4, 3)*(1-eps**sp.Rational(3, 2)/(1+eps**(2*m))**(sp.Rational(3, 4)/m)))})
        # The bounded scalar excess tends to C. Multiplying by x/(x+h)
        # makes that formerly constant action term spatially varying.
        leading = x+C*x/(x+h)
        bounded_ratio = sp.factor(point_anomaly(leading)/point_g)
        expected_ratio = -2*C*ell**2*a0**2*r**6*(30*ell**2+r**2)/(gm**2*(6*ell**2+r**2)**3)
        checks['bounded_leading_high_gradient_force'] = sp.simplify(bounded_ratio-expected_ratio) == 0
        checks['bounded_leading_sign_outward'] = bool(sp.ask(sp.Q.negative(expected_ratio)))
        # All signs in this last statement are limited to the specified positive
        # point-source domain and leading high-gradient action term.
        result = {**provenance, 'checks': checks, 'passes': all(checks.values()),
                  'radial_flux': 'P_x*g-ell^2*((P_h*g_prime)_prime+2*P_h*(g_prime-g/r)/r)',
                  'original_unregularized_point_anomaly': str(old_force),
                  'original_compact_system_coefficient': str(small_system_coefficient),
                  'proposed_cards': rows, 'bounded_leading_high_gradient_fractional_anomaly': str(bounded_ratio),
                  'note': 'The leading bounded term is outward in this limit; the full finite-u force is not assigned a sign.',
                  'empirical_scores': None, 'physical_length_selected': False,
                  'status': 'ANALYTIC_CONTROLS_PASS_NOT_OBSERVATIONAL_ADMISSION' if all(checks.values()) else 'ANALYTIC_FAILURE_RETAINED'}
        for name, digest in hashes.items():
            if sha256((ROOT/name).read_bytes()).hexdigest() != digest:
                raise RuntimeError('Input changed during audit')
        blob = (json.dumps(result, indent=2, sort_keys=True)+'\n').encode()
        (args.output/'result.json').write_bytes(blob)
        (args.output/'receipt.json').write_text(json.dumps({'result_sha256': sha256(blob).hexdigest(), 'status': result['status']})+'\n', newline='\n')
        print(json.dumps({'checks': len(checks), 'passes': result['passes'], 'status': result['status']}))
        if not result['passes']:
            raise RuntimeError('Analytic controls failed')
    except Exception as exc:
        (args.output/'failure.json').write_text(json.dumps({'error': str(exc), 'status': 'FAILURE_RETAINED'})+'\n', newline='\n')
        raise


if __name__ == '__main__':
    main()
