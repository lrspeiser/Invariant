"""Audit the user's overlapping-range and elastic ideas before data fitting."""
from __future__ import annotations

import argparse
import json
import math
import subprocess
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import sympy as sp
from scipy.integrate import quad
from scipy.special import exp1, gammainc

ROOT = Path(__file__).resolve().parents[1]


def continuum_boost(radius, lo, hi):
    """Integral_lo^hi [1-(1+r/lambda) exp(-r/lambda)] d lambda, L=1."""
    radius = np.asarray(radius, float)
    value = -hi*np.expm1(-radius/hi)+lo*np.expm1(-radius/lo)
    small = radius/lo < .1
    u = radius[small]/lo
    value[small] = lo*sum((-1)**n*u**n*(1-(lo/hi)**(n-1))/math.factorial(n) for n in range(2, 22))
    return value


def continuum_potential(radius, lo, hi):
    """Unit-mass pair potential, zero at infinity, from the same spectrum."""
    radius = np.asarray(radius, float)
    integral = continuum_boost(radius, lo, hi)+radius*(exp1(radius/hi)-exp1(radius/lo))
    return -(1+integral)/radius


def source_force(positions, masses, lo, hi):
    force = np.zeros_like(positions)
    for i in range(len(masses)):
        for j in range(i):
            d = positions[j]-positions[i]
            r = np.linalg.norm(d)
            f = masses[i]*masses[j]*(1+continuum_boost(np.array([r]), lo, hi)[0])*d/r**3
            force[i] += f
            force[j] -= f
    return force


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_overlap_elastic_audit_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    inputs = [Path(__file__), args.config.resolve(), *[ROOT/p for p in config['historical_documents']]]
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in inputs}

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as f:
            json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
            f.write('\n')

    for p in inputs:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())
    registration = {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
    write('started.json', registration)
    try:
        r, lam = sp.symbols('r lambda', positive=True)
        powers = []
        for p in config['relative_power_exponents']:
            potential_extra = sp.log(r) if p == 1 else r**(p-1)/(p-1)
            assert sp.simplify(sp.diff(potential_extra, r)-r**(p-2)) == 0
            speed_squared = 1/r+r**(p-1)
            slope = sp.limit(r*sp.diff(speed_squared, r)/(2*speed_squared), r, sp.oo)
            powers.append({'relative_boost_power': p, 'extra_force_radial_power': p-2,
                'extra_potential_in_units_GM_over_L': str(potential_extra),
                'asymptotic_speed_log_slope': float(slope),
                'extra_dominated_speed_factor_for_doubled_radius': float(2**slope)})
        B = 1-(1+r/lam)*sp.exp(-r/lam)
        psi = -(1-sp.exp(-r/lam))/r
        assert sp.simplify(sp.diff(psi, r)-B/r**2) == 0
        assert sp.simplify(sp.diff(B, r)-r*sp.exp(-r/lam)/lam**2) == 0
        lo, hi = config['continuous_range_limits_in_L']
        radii = np.array(config['quadrature_comparison_radii'])
        analytic = continuum_boost(radii, lo, hi)
        numerical = np.array([quad(lambda t, radius=radius: np.exp(t)*gammainc(2., radius/np.exp(t)),
            np.log(lo), np.log(hi), epsabs=1e-18, epsrel=1e-11)[0] for radius in radii])
        quad_errors = abs(analytic-numerical)/np.maximum(abs(numerical), 1e-30)
        assert quad_errors.max() < config['quad_relative_tolerance']
        probes = np.array(config['potential_derivative_radii'])
        step = config['finite_difference_fractional_step']*probes
        derivative = sum(c*continuum_potential(probes+offset*step, lo, hi) for offset, c in [(-2, 1), (-1, -8), (1, 8), (2, -1)])/(12*step)
        force = (1+continuum_boost(probes, lo, hi))/probes**2
        derivative_errors = abs(derivative-force)/force
        assert derivative_errors.max() < config['potential_force_relative_tolerance']
        positions = np.array(config['synthetic_source']['positions'])
        masses = np.array(config['synthetic_source']['masses'])
        forces = source_force(positions, masses, lo, hi)
        force_scale = np.linalg.norm(forces, axis=1).sum()
        net_force = np.linalg.norm(forces.sum(axis=0))/force_scale
        net_torque = np.linalg.norm(np.cross(positions, forces).sum(axis=0))/(force_scale*np.linalg.norm(positions, axis=1).max())
        assert max(net_force, net_torque) < config['momentum_torque_scaled_tolerance']

        def energy(pos):
            return sum(masses[i]*masses[j]*continuum_potential(np.array([np.linalg.norm(pos[i]-pos[j])]), lo, hi)[0]
                       for i in range(len(masses)) for j in range(i))

        derivatives = np.zeros_like(positions)
        for i in range(len(masses)):
            for j in range(3):
                for offset, coefficient in [(-2, 1), (-1, -8), (1, 8), (2, -1)]:
                    pos = positions.copy()
                    pos[i, j] += offset*1e-4
                    derivatives[i, j] += coefficient*energy(pos)/(12e-4)
        energy_force_error = np.linalg.norm(derivatives+forces)/force_scale
        assert energy_force_error < config['potential_force_relative_tolerance']
        # A fixed linear kernel is invariant under splitting a source element.
        strength = lambda m: m*(1+continuum_boost(np.array([3.]), lo, hi)[0])/9
        split_error = abs(strength(1.)-2*strength(.5))/strength(1.)
        assert split_error == 0.
        square_root_split_ratio = float(2*sp.sqrt(sp.Rational(1, 2)))
        result = {**registration, 'power_responses': powers,
            'continuum': {'formula_extra_boost': 'lambda_max*(1-exp(-r/lambda_max))-lambda_min*(1-exp(-r/lambda_min)), divided by L',
                'small_radius_coefficient': .5*(1/lo-1/hi), 'far_boost_plateau': hi-lo,
                'radii_in_L': radii.tolist(), 'extra_boost': analytic.tolist(), 'independent_quadrature_relative_errors': quad_errors.tolist(),
                'potential_force_probe_radii': probes.tolist(), 'potential_force_relative_errors': derivative_errors.tolist(),
                'interpretation': 'Finite continuum of known subtracted-Yukawa responses; approximately 1/r extra force between its cutoffs, returns to inverse square beyond the largest range.'},
            'conservative_pair_controls': {'net_force_scaled': float(net_force), 'net_torque_scaled': float(net_torque),
                'energy_gradient_force_error_scaled': float(energy_force_error), 'source_split_relative_error': float(split_error),
                'scope': 'Instantaneous nonrelativistic central pair potential. No propagating-field stability, relativistic causality or photon law checked.'},
            'mass_scaling': {'linear_kernel_speed_exponent_in_flat_compact_source_regime': .5,
                'approximately_observed_BTF_speed_exponent': .25,
                'sixteen_times_mass_linear_speed_factor': 4., 'sixteen_times_mass_quarter_power_speed_factor': 2.,
                'required_one_over_r_length_for_quarter_power_mass_scaling': 'L=sqrt(G M/a0); must emerge from a universally coupled field, not a per-object fit.',
                'scope': 'Constraint on a universal mass-linear kernel in a genuinely flat compact-source regime, not a likelihood or a rejection of arbitrary nonlinear multiscale gravity.'},
            'naive_square_root_source_split_witness': {'field_ratio_after_two_equal_colocated_source_pieces': square_root_split_ratio,
                'scope': 'Rejects only assigning an independently square-rooted mass contribution to each arbitrarily chosen source piece. A nonlinear field of the total source is a different construction.'},
            'all_registered_analytic_and_numerical_controls_pass': True,
            'source_tail_precision_repair_status': 'PENDING_UNCHANGED_NO_NEW_SOURCE_TAIL_RUN',
            'new_observational_scores': 0, 'validated_universal_gravity_laws': 0,
            'admitted_as_novel_first_principles_theory': False}
        assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
        write('result.json', result)
        write('receipt.json', {'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps({'controls_pass': True, 'quadrature_error': float(quad_errors.max()),
            'potential_force_error': float(derivative_errors.max()), 'energy_force_error': float(energy_force_error),
            'net_force': float(net_force), 'net_torque': float(net_torque), 'square_root_source_split_ratio': square_root_split_ratio}), flush=True)
    except Exception as exc:
        write('failure.json', {'error': repr(exc)})
        raise


if __name__ == '__main__':
    main()
