"""Propagate source interpolation discrepancies through all 54 fixed actions.

This checks flux inputs only. It does not solve Poisson's equation or score
observations, and a small full-flux discrepancy does not resolve a tiny anomaly.
"""
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
from invariant_gravity_extensions.galaxy_development import SI_ACCELERATION_TO_KMS2_KPC
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux
from invariant_gravity_extensions.potential_join import cartesian_tensors


def full_flux(fields, card):
    _, p, H, _ = cartesian_tensors(fields)
    dnorm, dlap = np.zeros_like(p), np.zeros_like(p)
    dnorm[:2], dlap[:2] = fields['gradient_hessian_norm_R_z'], fields['gradient_laplacian_R_z']
    anomaly = anomalous_flux(LengthScreening(card['shape'], card['epsilon']), p, H, dnorm, dlap,
        card['length_pc']/1000, card['a0_m_s2']*SI_ACCELERATION_TO_KMS2_KPC)
    return p+anomaly, anomaly


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pilot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    cards_path = ROOT/'work/gravity-first-principles/length-screening-local-001/result.json'
    paths = [Path(__file__), cards_path, args.pilot/'result.json', args.pilot/'direct_reference.json',
        args.pilot/'interpolated_coarse.json', args.pilot/'interpolated_fine.json',
        *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]
    hashes = {p.resolve().relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}
    if hashes[cards_path.relative_to(ROOT).as_posix()] != '66ff601b1012da7cbc555a27d8836723a2c6e7b23f393ead530da64e6e938a77':
        raise ValueError('Registered action cards changed')
    for p in paths:
        target = args.output/'input-snapshots'/p.resolve().relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, data):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(data, handle, indent=2, allow_nan=False)
            handle.write('\n')

    write('started.json', {'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat(),
        'full_flux_scaled_target': .01, 'target_scope': 'numerical diagnostic only, before separate Poisson solve',
        'scaling': 'max(norm(reference full flux), 1e-10 times median Newtonian gradient norm)',
        'new_action_cards': False, 'new_observational_scores': False})
    cards = [row['card'] for row in json.loads(cards_path.read_bytes())['rows']]
    assert len(cards) == 54 and len({c['id'] for c in cards}) == 54
    reference = {k: np.array(v) for k, v in json.loads((args.pilot/'direct_reference.json').read_bytes()).items()}
    rows = []
    floor = 1e-10*np.median(np.linalg.norm(reference['gradient_R_z'], axis=0))
    for grid in ['coarse', 'fine']:
        fields = {k: np.array(v) for k, v in json.loads((args.pilot/f'interpolated_{grid}.json').read_bytes())['fields'].items()}
        for card in cards:
            direct, direct_anomaly = full_flux(reference, card)
            value, anomaly = full_flux(fields, card)
            scale = np.maximum(np.linalg.norm(direct, axis=0), floor)
            errors = np.linalg.norm(value-direct, axis=0)/scale
            anomaly_errors = np.linalg.norm(anomaly-direct_anomaly, axis=0)/scale
            rows.append({'grid': grid, 'card': card['id'], 'maximum_full_flux_scaled_error': float(np.max(errors)),
                'maximum_anomalous_flux_error_scaled_by_full_flux': float(np.max(anomaly_errors)),
                'worst_index': [int(i) for i in np.unravel_index(np.argmax(errors), errors.shape)],
                'within_flux_diagnostic_target': bool(np.max(errors) < .01)})
    assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    summary = [{'grid': grid, 'cards': 54, 'within_target': sum(r['within_flux_diagnostic_target'] for r in rows if r['grid']==grid),
        'maximum_full_flux_scaled_error': max(r['maximum_full_flux_scaled_error'] for r in rows if r['grid']==grid)}
        for grid in ['coarse', 'fine']]
    write('result.json', {'summary': summary, 'rows': rows, 'completed_utc': datetime.now(UTC).isoformat(),
        'separate_poisson_solve': False, 'full_source_admitted': False, 'new_observational_scores': 0})
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
