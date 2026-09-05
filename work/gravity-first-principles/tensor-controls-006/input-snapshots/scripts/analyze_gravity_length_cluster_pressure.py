"""Record cross-regime descriptive counts without selecting a new action card."""
import argparse
import json
from hashlib import sha256
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    paths = {'pressure': 'work/gravity-first-principles/length-cluster-pressure-001/result.json',
             'local': 'work/gravity-first-principles/length-screening-local-001/result.json',
             'old_pressure': 'work/gravity-first-principles/xcop-pressure-002/result.json',
             'source_preflight': 'work/gravity-first-principles/length-cluster-pressure-001/source_preflight.json',
             'verification': 'work/gravity-first-principles/length-cluster-pressure-verification-001/result.json'}
    hashes = {path: sha256((ROOT/path).read_bytes()).hexdigest() for path in paths.values()}
    hashes[Path(__file__).relative_to(ROOT).as_posix()] = sha256(Path(__file__).read_bytes()).hexdigest()
    (args.output/'analysis-script.py').write_bytes(Path(__file__).read_bytes())
    data = {key: json.loads((ROOT/path).read_bytes()) for key, path in paths.items()}
    r = data['pressure']
    nominal = {e['model']: e for e in r['entries'] if e['scenario'] == 'nominal'}
    entries = {(e['model'], e['scenario']): e for e in r['entries']}
    comparator = 'empirical_RAR_a0_1.2e-10'
    base = nominal[comparator]['summary']
    local = {e['card']['id']: e for e in data['local']['rows']}
    records, max_pressure, max_force = [], 0., 0.
    for model in r['models']:
        if model['family'] != 'length_screening':
            continue
        key = model['id']
        comp = next(c for c in r['nominal_comparisons'] if c['model'] == key and c['baseline'] == comparator)
        sens = next(s for s in r['global_sensitivities'] if s['model'] == key)
        matched = next(c for c in sens['matched_comparisons'] if c['baseline'] == comparator)
        wins = {name: 0 for name in base['equal_cluster_whitened_mean_squared_residual']}
        for scenario in r['scenarios']:
            a, b = entries[(key, scenario['id'])]['summary'], entries[(comparator, scenario['id'])]['summary']
            for name in wins:
                wins[name] += int(a['equal_cluster_whitened_mean_squared_residual'][name] < b['equal_cluster_whitened_mean_squared_residual'][name])
        zero = key.split('_ell_')[0]+'_ell_0pc'
        pressure_change = force_change = 0.
        for a, b in zip(nominal[key]['rows'], nominal[zero]['rows'], strict=True):
            pressure_change = max(pressure_change, float(np.max(abs(np.asarray(a['prediction'])/b['prediction']-1))))
            force_change = max(force_change, float(np.max(abs(np.asarray(a['predicted_acceleration_m_s2'])/b['predicted_acceleration_m_s2']-1))))
        max_pressure, max_force = max(max_pressure, pressure_change), max(max_force, force_change)
        records.append({'model': model, 'within_local_screens': local[key]['status'].startswith('WITHIN_'),
                        'summary': nominal[key]['summary'], 'nominal_comparison_with_RAR': comp,
                        'scenarios_with_lower_primary_loss': matched['scenarios_with_lower_loss'],
                        'scenarios_with_lower_covariance_loss': wins,
                        'local_Q2_range_s_minus2': [min(q['Q2_s_minus2'] for q in local[key]['external_quadrupole']),
                                                  max(q['Q2_s_minus2'] for q in local[key]['external_quadrupole'])],
                        'maximum_nominal_pressure_change_from_zero_length': pressure_change,
                        'maximum_nominal_force_change_from_zero_length': force_change})
    within = [x for x in records if x['within_local_screens']]
    improving = [x for x in within if x['nominal_comparison_with_RAR']['mean_difference'] < 0]
    bridge = {}
    for name in r['config']['comparators']:
        old = next(e for e in data['old_pressure']['entries'] if e['model'] == name and e['scenario'] == 'nominal')
        bridge[name] = max(float(np.max(abs(np.asarray(a['prediction'])/b['prediction']-1)))
                           for a, b in zip(nominal[name]['rows'], old['rows'], strict=True))
    stats = {'input_hashes': hashes, 'records': records, 'cards_within_local_screens': len(within),
             'within_local_and_lower_nominal_cluster_loss_than_RAR': len(improving),
             'these_cards_lower_in_all_global_scenarios': all(x['scenarios_with_lower_primary_loss'] == len(r['scenarios']) for x in improving),
             'these_cards_lower_in_all_covariance_treatments_and_scenarios': all(all(n == len(r['scenarios']) for n in x['scenarios_with_lower_covariance_loss'].values()) for x in improving),
             'these_cards_lower_after_influence_removal_and_trim': all(x['nominal_comparison_with_RAR']['leave_most_influential_out']['mean_difference'] < 0 and x['nominal_comparison_with_RAR']['symmetric_trim']['mean_difference'] < 0 for x in improving),
             'these_cards_lower_in_all_eight_nominal_objects': all(x['nominal_comparison_with_RAR']['raw_comparative_win_count'] == 8 for x in improving),
             'maximum_nominal_pressure_change_from_zero_length': max_pressure,
             'maximum_nominal_target_force_change_from_zero_length': max_force,
             'comparator_bridge_maximum_relative_pressure_change': bridge,
             'source_fidelity_flagged_sensitivities': [{'cluster': x['cluster'], 'source_key': x['source_key']}
                 for x in data['source_preflight']['rows'] if not x['within_primary_source_limits']],
             'quality_verified_counterexamples': 0, 'uncertainty_resolved_counterexamples': 0,
             'independent_validation_count': 0, 'new_formula_selected': False, 'family_pruned': False,
             'interpretation': 'Same-constant local/cluster tension can be reduced within this finite ansatz grid; no joint galaxy or lensing success, first-principles derivation or full local validation follows.'}
    with (args.output/'result.json').open('x', encoding='utf8', newline='\n') as f:
        json.dump(stats, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')
    print(json.dumps({key: value for key, value in stats.items() if key not in ['input_hashes', 'records']}))


if __name__ == '__main__':
    main()
