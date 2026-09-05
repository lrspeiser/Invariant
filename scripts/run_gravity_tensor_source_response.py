"""Propagate every retained source quadrature variation through the full action."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
from run_gravity_tensor_poisson import (
    ROOT,
    ExteriorMomentField,
    FluxPoissonSolver,
    MatchedTensorPotential,
    MultipoleGrid,
    full_flux,
    serial,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    base = ROOT/'work/gravity-first-principles'
    source, reference = base/'radial-tensor-replay-001', base/'tensor-poisson-001'
    prior = json.loads((reference/'result.json').read_bytes())
    cases = ['radial_coarse', 'wavenumber_coarse', 'cutoff_200', 'vertical_coarse', 'tail_extent']
    variants, distances = prior['config']['variants'], prior['config']['distances']
    definition = prior['config']['grids']['fine']
    cards_path, units_path = base/'length-screening-local-001/result.json', base/'map-source-003/result.json'
    paths = {Path(__file__), ROOT/'scripts/run_gravity_tensor_poisson.py', ROOT/'scripts/audit_gravity_tensor_flux.py',
        source/'result.json', reference/'result.json', cards_path, units_path,
        *list((ROOT/'src/invariant_gravity_extensions').glob('*.py'))}
    for variant in variants:
        paths.add(reference/f'family_{variant}_fine.json')
        paths.add(base/f'exterior-moment-002/moments_{variant}_reference.json')
        paths.update(source/f'mixed_{case}_{variant}.json' for case in cases)
    hashes = {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}
    for path, expected in [(reference/'result.json', '5c5d19ea954df993f7e4e2104d6495257eb7b7995c0bb9cf56fb12f0ad656306'),
        (source/'result.json', '9136c90030a114b89b350f816beffa41ce24ad7958d6eeafaad107483e5c74c2'),
        (cards_path, '66ff601b1012da7cbc555a27d8836723a2c6e7b23f393ead530da64e6e938a77')]:
        assert hashes[path.relative_to(ROOT).as_posix()] == expected
    for p in paths:
        target = args.output/'input-snapshots'/p.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(p.read_bytes())

    def write(name, data):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(data, handle, indent=2, default=serial, allow_nan=False)
            handle.write('\n')

    config = {'registration': 'All five retained source variations, both thicknesses, all 54 cards and three distances; frozen before response evaluation.',
        'cases': cases, 'variants': variants, 'distances': distances, 'grid': definition,
        'full_force_diagnostic_target': .002,
        'target_scope': 'Numerical source sensitivity below one tenth of inherited 2 percent full-force refinement target; not empirical admission.',
        'small_signal_rule': 'Store source-induced change in length-minus-zero response separately; no small effect is admitted by a total-force tolerance.',
        'scope': 'Numerical source representation only; physical source uncertainty and other source scenarios remain pending.',
        'new_observational_scores': 0}
    write('started.json', {'config': config, 'input_hashes': hashes, 'started_utc': datetime.now(UTC).isoformat()})
    cards = [r['card'] for r in json.loads(cards_path.read_bytes())['rows']]
    assert len(cards) == len({c['id'] for c in cards}) == 54
    G = json.loads(units_path.read_bytes())['config']['units']['G_kpc_kms2_msun']
    comparisons = []
    try:
        solver = FluxPoissonSolver(MultipoleGrid(**definition))
        for variant in variants:
            ref = json.loads((reference/f'family_{variant}_fine.json').read_bytes())
            lookup = {(p['card'], p['distance']):p for p in ref['predictions']}
            radii = np.array(ref['radii_kpc'])
            moments = json.loads((base/f'exterior-moment-002/moments_{variant}_reference.json').read_bytes())
            exterior = ExteriorMomentField(moments, G, minimum_radius=60.)
            for case in cases:
                raw = json.loads((source/f'mixed_{case}_{variant}.json').read_bytes())
                provider = MatchedTensorPotential(np.array(raw['radius_kpc']), np.array(raw['height_kpc']), np.array(raw['mixed']), exterior)
                newton = provider.fields(radii, np.zeros_like(radii))['gradient_R_z']
                chunks = []
                for start in range(0, len(solver.radius), 32):
                    r = solver.radius[start:start+32, None]
                    chunks.append(provider.fields(r*solver.sine, r*solver.mu))
                fields = {k: np.concatenate([c[k] for c in chunks], axis=-2) for k in chunks[0]}
                del chunks
                print(f'{variant}/{case}: source evaluated', flush=True)
                predictions = []
                for distance in distances:
                    zero = {}
                    for card in sorted(cards, key=lambda c:c['length_pc']):
                        _, anomaly = full_flux(fields, {**card, 'length_pc':card['length_pc']/distance})
                        flux = np.array([solver.sine*anomaly[0]+solver.mu*anomaly[1], solver.mu*anomaly[0]-solver.sine*anomaly[1]])
                        total = newton-solver.solve(flux).evaluate(radii, np.zeros_like(radii))['acceleration']
                        key = (card['shape'], card['a0_m_s2'], card['epsilon'])
                        if card['length_pc'] == 0:
                            zero[key] = total.copy()
                        signal = total-zero[key]
                        original = lookup[(card['id'], distance)]
                        ref_total, ref_signal = np.array(original['gradient_R_z']), np.array(original['length_minus_zero_gradient_R_z'])
                        scale = np.maximum(abs(ref_total[0]), abs(np.array(ref['newton_gradient_R_z'])[0]))
                        err = np.linalg.norm(total-ref_total, axis=0)/scale
                        serr = np.linalg.norm(signal-ref_signal, axis=0)/scale
                        comparisons.append({'variant':variant, 'case':case, 'card':card['id'], 'distance':distance,
                            'maximum_scaled_force_change':float(max(err)), 'within_full_force_target':bool(max(err)<.002),
                            'maximum_scaled_signal_change':float(max(serr)),
                            'maximum_scaled_reference_signal':float(max(np.linalg.norm(ref_signal, axis=0)/scale)),
                            'reflection_error':float(max(abs(total[1])/np.maximum(abs(total[0]),abs(newton[0])))),
                            'positive_radial':bool(np.all(total[0]>0))})
                        predictions.append({'card':card['id'], 'distance':distance, 'gradient_R_z':total,
                            'length_minus_zero_gradient_R_z':signal})
                    print(f'{variant}/{case}: all 54 cards at distance {distance:.6g}', flush=True)
                write(f'family_{variant}_{case}.json', {'variant':variant, 'case':case, 'radii_kpc':radii,
                    'newton_gradient_R_z':newton, 'predictions':predictions})
                del fields
        assert len(comparisons) == 1620
        assert all(sha256((ROOT/p).read_bytes()).hexdigest()==digest for p,digest in hashes.items())
        write('result.json', {'config':config, 'comparisons':comparisons, 'all_cases_complete':True,
            'all_full_force_diagnostics_pass':all(c['within_full_force_target'] for c in comparisons),
            'all_reflections_pass':all(c['reflection_error']<1e-8 for c in comparisons),
            'all_radial_forces_positive':all(c['positive_radial'] for c in comparisons),
            'small_effects_admitted':False, 'new_observational_scores':0, 'physical_exclusions':0,
            'completed_utc':datetime.now(UTC).isoformat()})
    except Exception as exc:
        write('failure.json', {'type':type(exc).__name__, 'message':str(exc)})
        raise


if __name__ == '__main__':
    main()
