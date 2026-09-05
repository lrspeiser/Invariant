"""Full action development solve from the qualified matched tensor source.

No observational scoring. Keep all cards, distances and both qualified sources.
Finite-shell flux boundaries and numerical truncation remain explicit.
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
from audit_gravity_tensor_flux import full_flux

from invariant_gravity_extensions.exterior_moments import ExteriorMomentField
from invariant_gravity_extensions.external_multifield import FluxPoissonSolver
from invariant_gravity_extensions.isolated_axisymmetric import MultipoleGrid
from invariant_gravity_extensions.matched_tensor import MatchedTensorPotential


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    source = base/'radial-tensor-replay-001'
    inherited_path = ROOT/'configs/gravity_ngc3198_scalar_development_v1.json'
    inherited = json.loads(inherited_path.read_bytes())
    definitions = inherited['field_grids']
    grids = {k: definitions[k] for k in ['coarse', 'fine', 'boundary']}
    grids['radial_only'] = {**grids['coarse'], 'radial_nodes': grids['fine']['radial_nodes']}
    grids['angular_only'] = {**grids['coarse'], 'angular_nodes': grids['fine']['angular_nodes'],
                             'l_max': grids['fine']['l_max']}
    cards_path = base/'length-screening-local-001/result.json'
    selection_path = base/'ngc3198-scalar-001/radial_selection.json'
    units_path = base/'map-source-003/result.json'
    variants = ['primary', 'height_half']
    distances = [1+x/inherited['geometry']['published_distance_mpc']
                 for x in inherited['geometry']['distance_offsets_mpc']]
    config = {'registration': 'Frozen before this full-action run; all 54 existing cards, three distances, both qualified thicknesses.',
        'grids': grids, 'distances': distances, 'variants': variants,
        'force_refinement_target': .02, 'boundary_target': .005, 'reflection_target': 1e-8,
        'scope': 'Numerical development only. Other source scenarios, map uncertainty and observational scores remain pending.',
        'length_signal_rule': 'Record length minus zero-length force for every matching shape/a0/distance; full-field convergence alone does not resolve it.',
        'boundary_rule': 'Existing Green solver zero-extends anomalous flux outside each finite radial shell.',
        'source_qualification': 'Sampled source gate only, not a uniform error bound.'}
    paths = {Path(__file__), ROOT/'scripts/audit_gravity_tensor_flux.py', inherited_path,
        cards_path, selection_path, units_path, source/'result.json',
        *list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
    for variant in variants:
        paths.add(source/f'mixed_canonical_{variant}.json')
        paths.add(base/f'exterior-moment-002/moments_{variant}_reference.json')
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    assert hashes[cards_path.relative_to(ROOT).as_posix()] == '66ff601b1012da7cbc555a27d8836723a2c6e7b23f393ead530da64e6e938a77'
    assert hashes[(source/'result.json').relative_to(ROOT).as_posix()] == '9136c90030a114b89b350f816beffa41ce24ad7958d6eeafaad107483e5c74c2'
    assert json.loads((source/'result.json').read_bytes())['ready_for_development_field_solve']
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, data):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(data, handle, indent=2, default=serial, allow_nan=False)
            handle.write('\n')

    write('started.json', {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat()})
    cards = [row['card'] for row in json.loads(cards_path.read_bytes())['rows']]
    assert len(cards) == len({c['id'] for c in cards}) == 54
    radii = np.array(json.loads(selection_path.read_bytes())['gate_radii_kpc'])
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    families = {}
    try:
        for variant in variants:
            raw = json.loads((source/f'mixed_canonical_{variant}.json').read_bytes())
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            provider = MatchedTensorPotential(np.array(raw['radius_kpc']), np.array(raw['height_kpc']),
                np.array(raw['mixed']), ExteriorMomentField(moments, G, minimum_radius=60.))
            newton = provider.fields(radii, np.zeros_like(radii))['gradient_R_z']
            for name, definition in grids.items():
                solver = FluxPoissonSolver(MultipoleGrid(**definition))
                chunks = []
                for start in range(0, len(solver.radius), 32):
                    r = solver.radius[start:start+32, None]
                    chunks.append(provider.fields(r*solver.sine, r*solver.mu))
                fields = {k: np.concatenate([c[k] for c in chunks], axis=-2) for k in chunks[0]}
                del chunks
                print(f'{variant}/{name}: source evaluated on {definition["radial_nodes"]} x {definition["angular_nodes"]}', flush=True)
                predictions = []
                for distance in distances:
                    zero = {}
                    for card in sorted(cards, key=lambda c: c['length_pc']):
                        effective = {**card, 'length_pc': card['length_pc']/distance}
                        _, anomaly = full_flux(fields, effective)
                        flux = np.array([solver.sine*anomaly[0]+solver.mu*anomaly[1],
                                         solver.mu*anomaly[0]-solver.sine*anomaly[1]])
                        correction = -solver.solve(flux).evaluate(radii, np.zeros_like(radii))['acceleration']
                        total = newton+correction
                        key = (card['shape'], card['a0_m_s2'], card['epsilon'])
                        if card['length_pc'] == 0:
                            zero[key] = total.copy()
                        predictions.append({'card': card['id'], 'distance': distance, 'gradient_R_z': total,
                            'length_minus_zero_gradient_R_z': total-zero[key],
                            'positive_radial': bool(np.all(total[0] > 0)),
                            'reflection_error': float(np.max(abs(total[1])/np.maximum(abs(total[0]), abs(newton[0]))))})
                    print(f'{variant}/{name}: all 54 cards at distance {distance:.6g}', flush=True)
                family = {'variant': variant, 'grid': definition, 'radii_kpc': radii,
                    'newton_gradient_R_z': newton, 'predictions': predictions}
                families[variant+'/'+name] = family
                write(f'family_{variant}_{name}.json', family)
                del fields
        comparisons = []
        for variant in variants:
            reference = families[variant+'/fine']
            for name in ['coarse', 'boundary', 'radial_only', 'angular_only']:
                for a, b in zip(reference['predictions'], families[variant+'/'+name]['predictions'], strict=True):
                    assert (a['card'], a['distance']) == (b['card'], b['distance'])
                    scale = np.maximum(abs(a['gradient_R_z'][0]), abs(reference['newton_gradient_R_z'][0]))
                    err = np.linalg.norm(a['gradient_R_z']-b['gradient_R_z'], axis=0)/scale
                    signal_err = np.linalg.norm(a['length_minus_zero_gradient_R_z']-b['length_minus_zero_gradient_R_z'], axis=0)/scale
                    limit = .005 if name == 'boundary' else .02
                    comparisons.append({'variant': variant, 'alternative': name, 'card': a['card'], 'distance': a['distance'],
                        'maximum_scaled_force_change': float(np.max(err)), 'force_target': limit,
                        'within_force_target': bool(np.max(err) < limit),
                        'maximum_scaled_length_signal_change': float(np.max(signal_err)),
                        'maximum_scaled_length_signal': float(np.max(np.linalg.norm(a['length_minus_zero_gradient_R_z'], axis=0)/scale))})
        assert all(sha256((ROOT/p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
        write('result.json', {'config': config, 'comparisons': comparisons,
            'all_cases_complete': True, 'all_force_refinements_pass': all(c['within_force_target'] for c in comparisons),
            'all_reflections_pass': all(p['reflection_error'] < 1e-8 for f in families.values() for p in f['predictions']),
            'all_radial_forces_positive': all(p['positive_radial'] for f in families.values() for p in f['predictions']),
            'new_observational_scores': 0, 'physical_exclusions': 0, 'completed_utc': datetime.now(UTC).isoformat()})
    except Exception as exc:
        write('failure.json', {'type': type(exc).__name__, 'message': str(exc)})
        raise


if __name__ == '__main__':
    main()
