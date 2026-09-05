"""Frozen physical-length scan against previously exposed local summaries."""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.external_quadrupole import newtonian_external_ratio
from invariant_gravity_extensions.length_screening import LengthScreening, point_monopole_delta, point_quadrupole
from invariant_gravity_extensions.local_limits import Orbit, mas_per_century, perihelion_first_order
from invariant_gravity_extensions.saturated_actions import SaturatedActionSpec


def classification(value, low, high, numerical_ok, spread):
    if not numerical_ok:
        return 'NUMERICALLY_UNRESOLVED'
    if min(abs(value-low), abs(value-high)) <= spread:
        return 'NEAR_SCREEN_EDGE_REQUIRES_REFINEMENT'
    return 'WITHIN_DECLARED_SUMMARY_SCREEN' if low <= value <= high else 'OUTSIDE_DECLARED_SUMMARY_SCREEN'


def local_card(card, config, monopole, external):
    spec = LengthScreening(card['shape'], config['epsilon'])
    a0, length_pc = card['a0_m_s2'], card['length_pc']
    gm = external['gm_sun_m3_s2']
    radius_mond = np.sqrt(gm/a0)
    length = length_pc*config['parsec_m']/radius_mond
    records = []
    for planet in monopole['planets']:
        orbit = Orbit(planet['a_au']*monopole['au_m'], planet['e'], gm)

        def prediction(model, count):
            delta = lambda y: point_monopole_delta(model, y, length)
            return mas_per_century(perihelion_first_order(orbit, a0, delta, nodes=count), orbit, monopole['century_s'])

        estimates = [{'nodes': count, 'precession_mas_century': prediction(spec, count)} for count in config['monopole_nodes']]
        epsilons = [{'epsilon': e, 'precession_mas_century': prediction(LengthScreening(card['shape'], e), config['monopole_nodes'][-1])}
                    for e in config['epsilon_sensitivity']]
        anomaly = estimates[-1]['precession_mas_century']
        refinement = abs(anomaly-estimates[-2]['precession_mas_century'])
        sensitivity = max(abs(anomaly-e['precession_mas_century']) for e in epsilons)
        phase = np.linspace(0, 2*np.pi, 257)
        r = orbit.semimajor_m*(1-orbit.eccentricity**2)/(1+orbit.eccentricity*np.cos(phase))
        fraction = point_monopole_delta(spec, gm/(a0*r*r), length)
        maximum = float(np.max(abs(fraction)))
        numerical = (max(refinement, sensitivity) < config['maximum_monopole_mas_century_change'] and
                     maximum < config['monopole_perturbation_maximum_fraction'] and np.all(1+fraction > 0))
        low = planet['interval_center_mas_cy']-planet['interval_halfwidth_mas_cy']
        high = planet['interval_center_mas_cy']+planet['interval_halfwidth_mas_cy']
        records.append({'planet': planet['name'], 'precession_mas_century': anomaly,
                        'interval_mas_century': [low, high], 'quadrature': estimates, 'epsilon_sensitivity': epsilons,
                        'last_refinement_change': refinement, 'epsilon_change': sensitivity,
                        'maximum_sampled_fractional_anomaly': maximum, 'numerical_controls_pass': bool(numerical),
                        'status': classification(anomaly, low, high, numerical, max(refinement, sensitivity))})
    qrecords = []
    scalar = SaturatedActionSpec('qumond', shape=card['shape'], epsilon=config['epsilon'])
    conversion = a0**1.5/np.sqrt(gm)
    cassini = external['cassini_summary']
    low = cassini['mean_Q2_s_minus2']-cassini['one_sigma_s_minus2']*cassini['screen_sigma_multiplier']
    high = cassini['mean_Q2_s_minus2']+cassini['one_sigma_s_minus2']*cassini['screen_sigma_multiplier']
    for background in external['physical_external_m_s2']:
        eta = background/a0
        eta_n = newtonian_external_ratio(eta, scalar.delta_nu)
        estimates = []

        def evaluate(count):
            return {'nodes': count, **point_quadrupole(spec, eta_n, length, quadrature_nodes=count)}

        for count in config['quadrature_nodes']:
            estimates.append(evaluate(count))

        def grid_ok():
            return (abs(estimates[-1]['Q2_flux']-estimates[-2]['Q2_flux']) < config['maximum_dimensionless_Q2_change'] and
                    estimates[-1]['absolute_agreement'] < config['maximum_action_flux_disagreement'])

        for count in config['automatic_numerical_refinements']:
            if grid_ok():
                break
            estimates.append(evaluate(count))
        final = estimates[-1]
        sensitivities = []
        for e in config['epsilon_sensitivity']:
            alternative = LengthScreening(card['shape'], e)
            boundary = SaturatedActionSpec('qumond', shape=card['shape'], epsilon=e)
            eta_check = newtonian_external_ratio(eta, boundary.delta_nu)
            sensitivities.append({'epsilon': e, 'eta_newtonian': eta_check,
                                  **point_quadrupole(alternative, eta_check, length, quadrature_nodes=final['nodes'])})
        refinement = abs(final['Q2_flux']-estimates[-2]['Q2_flux'])
        sensitivity = max(abs(v['Q2_flux']-final['Q2_flux']) for v in sensitivities)
        numerical = grid_ok() and sensitivity < config['maximum_epsilon_Q2_change']
        q2 = conversion*final['Q2_flux']
        spread = conversion*max(refinement, sensitivity, final['absolute_agreement'])
        qrecords.append({'physical_external_m_s2': background, 'eta_physical': eta, 'eta_newtonian': eta_n,
                         'Q2_s_minus2': q2, 'Q2_dimensionless': final['Q2_flux'], 'quadrature': estimates,
                         'epsilon_sensitivity': sensitivities, 'last_refinement_change': refinement,
                         'epsilon_change': sensitivity, 'empirical_Q2_spread_not_error_bound': spread,
                         'numerical_controls_pass': bool(numerical), 'interval_Q2_s_minus2': [low, high],
                         'status': classification(q2, low, high, numerical, spread)})
    statuses = [r['status'] for r in records+qrecords]
    if 'NUMERICALLY_UNRESOLVED' in statuses:
        combined = 'NUMERICALLY_UNRESOLVED'
    elif 'OUTSIDE_DECLARED_SUMMARY_SCREEN' in statuses:
        combined = 'OUTSIDE_COMBINED_DECLARED_SUMMARY_SCREEN'
    elif 'NEAR_SCREEN_EDGE_REQUIRES_REFINEMENT' in statuses:
        combined = 'NEAR_SCREEN_EDGE_REQUIRES_REFINEMENT'
    else:
        combined = 'WITHIN_COMBINED_DECLARED_SUMMARY_SCREEN'
    return {'card': card, 'mond_radius_m': float(radius_mond), 'dimensionless_length': float(length),
            'monopole': records, 'external_quadrupole': qrecords, 'status': combined,
            'full_solar_system_pass': False, 'galaxy_cluster_lensing_pass': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_length_screening_local_v1.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    external = json.loads((ROOT/config['historical_quadrupole_config']).read_bytes())
    monopole = json.loads((ROOT/config['historical_monopole_config']).read_bytes())
    paths = [Path(__file__), args.config, ROOT/config['historical_quadrupole_config'], ROOT/config['historical_monopole_config'],
             ROOT/'tests/test_gravity_length_screening.py', *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]

    def hashes():
        return {p.resolve().relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open('x', encoding='utf-8', newline='\n') as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    before = hashes()
    for path in paths:
        target = args.output/'input-snapshots'/path.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    provenance = {'input_hashes': before, 'config': config, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__,
                  'observations_accessed': 'previously exposed historical summaries only', 'new_raw_or_reserved_data': False}
    write('started.json', provenance)
    try:
        environment = {**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
        control = subprocess.run([sys.executable, '-m', 'pytest', 'tests/test_gravity_length_screening.py', '-q'],
                                 cwd=ROOT, env=environment, capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Independent theory controls failed before local summary scoring')
        rows = []
        for shape in config['shapes']:
            for a0 in config['a0_m_s2']:
                for length in config['length_pc']:
                    card = LengthScreening(shape, config['epsilon']).card(length, a0)
                    card['id'] = f'GQL_m{shape:g}_a0_{a0:.1e}_ell_{length:g}pc'
                    value = local_card(card, config, monopole, external)
                    write('card_'+card['id']+'.json', value)
                    rows.append(value)
                    print(card['id']+' '+value['status'], flush=True)
        if hashes() != before:
            raise RuntimeError('Input changed during scan')
        result = {**provenance, 'rows': rows, 'counts': dict(Counter(r['status'] for r in rows)),
                  'cards': len(rows), 'monopole_observables': sum(len(r['monopole']) for r in rows),
                  'quadrupole_observables': sum(len(r['external_quadrupole']) for r in rows),
                  'full_solar_system_pass': False, 'discovery_claim': False, 'family_pruning': False}
        write('result.json', result)
        write('receipt.json', {'status': 'CONDITIONAL_LOCAL_SUMMARY_SCAN_RETAINED',
                               'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest(),
                               'finished_utc': datetime.now(UTC).isoformat()})
        print(json.dumps({'cards': len(rows), 'counts': result['counts']}))
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_FAILURE_RETAINED_NOT_THEORY_REJECTION', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
