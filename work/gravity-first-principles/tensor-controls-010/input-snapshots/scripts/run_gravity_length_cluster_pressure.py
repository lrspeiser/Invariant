"""Frozen same-constant cluster transfer of the 54 length-action local cards."""
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
from run_gravity_xcop_pressure import aggregate, comparison, definitions, jsonable

from invariant_gravity_extensions.cluster_pressure import (
    DEVELOPMENT_CLUSTERS,
    GM_SUN,
    KPC,
    MU_E,
    PROTON_MASS,
    G,
)
from invariant_gravity_extensions.length_cluster_pressure import (
    array_packet,
    predict_from_context,
    pressure_context,
    score_prediction,
)
from invariant_gravity_extensions.length_screening import LengthScreening
from invariant_gravity_extensions.smooth_spherical_source import build_cluster_sources


def source_fidelity(packet, source, config):
    """Compare reconstructed source with its fixed input, before gravity scoring."""
    gas = source['gas'].evaluate(packet['density_radius_kpc']*KPC)['density']/(1e6*MU_E*PROTON_MASS)
    deviation = gas-source['input_ne_cm3']
    errors = np.where(deviation >= 0, packet['ne_high_error'], packet['ne_low_error'])
    sigma = deviation/errors
    star = None
    if source['stellar'] is not None:
        old = packet['stellar']
        mass = source['stellar'].evaluate(old['radius_kpc']*KPC)['mass']*G/GM_SUN
        star = {'mass_msun': mass, 'fraction_from_monotone': mass/np.maximum.accumulate(old['mass_msun'])-1,
                'fraction_from_raw': mass/old['mass_msun']-1, 'metadata': source['stellar'].metadata}
    mass_errors = {key: abs(source[key].cumulative_mass[-1]/source[key].expected_total_mass-1)
                   for key in ['gas', 'stellar'] if source[key] is not None}
    return {'gas_smoothing_shift_quoted_errors': sigma, 'maximum_gas_smoothing_shift': float(np.max(abs(sigma))),
            'stellar': star, 'total_mass_relative_errors': mass_errors,
            'within_primary_source_limits': bool(np.max(abs(sigma)) < config['maximum_gas_smoothing_shift_quoted_errors']
                and (star is None or np.max(abs(star['fraction_from_monotone'])) < config['maximum_stellar_mass_fraction_change'])),
            'total_mass_numerical_pass': max(mass_errors.values()) < config['maximum_total_mass_error']}


def source_key(scenario):
    s = scenario['source']
    return (s['width'], s['outer_factor'], s['outer_slope'], scenario['values']['density_error_shift'])


def relative_change(first, second):
    if 'prediction' not in first or 'prediction' not in second:
        return None
    return float(np.max(abs(first['prediction']/second['prediction']-1)))


def campaign(config, write):
    for key in ['local_result', 'source_audit', 'source_packet']:
        if sha256((ROOT/config[key]).read_bytes()).hexdigest() != config[key+'_sha256']:
            raise ValueError('Registered parent digest changed: '+key)
    local = json.loads((ROOT/config['local_result']).read_bytes())
    audit = json.loads((ROOT/config['source_audit']).read_bytes())
    if not audit['all_primary_numerical_pass'] or not audit['all_primary_fidelity_pass']:
        raise ValueError('Primary source audit did not pass')
    if audit['config']['primary_log_radius_width'] != config['primary_source']['width']:
        raise ValueError('Primary smoothing width changed')
    packets = [array_packet(p) for p in json.loads((ROOT/config['source_packet']).read_bytes())['packets']]
    if len(packets) != 8 or {p['cluster'] for p in packets} != DEVELOPMENT_CLUSTERS:
        raise ValueError('Entire registered development population required')
    models = [{'id': 'newtonian_baryons', 'family': 'newtonian'},
              {'id': 'empirical_RAR_a0_1.2e-10', 'family': 'rar_comparator', 'a0_m_s2': 1.2e-10}]
    for row in local['rows']:
        card = row['card']
        expected = LengthScreening(card['shape'], card['epsilon']).card(card['length_pc'], card['a0_m_s2'])
        if {key: value for key, value in card.items() if key != 'id'} != expected:
            raise ValueError('Local action card changed')
        models.append({**card, 'family': 'length_screening', 'prior_local_status': row['status']})
    if len(models) != 56:
        raise ValueError('All 54 cards and two comparators required')
    inherited = json.loads((ROOT/config['inherited_scenario_config']).read_bytes())
    _, scenarios = definitions(inherited)
    for scenario in scenarios:
        scenario['source'] = config['primary_source'].copy()
    for source in config['source_sensitivities_at_nominal_nuisance']:
        scenarios.append({'id': source['id'], 'values': inherited['nominal_nuisance'].copy(),
                          'source': {key: value for key, value in source.items() if key != 'id'}})
    write('registry.json', {'models': models, 'scenarios': scenarios, 'scoring_started': False})
    # Build all sources and inspect source fidelity before any pressure prediction.
    source_cache, preflight = {}, []
    for packet in packets:
        for key in dict.fromkeys(source_key(s) for s in scenarios):
            width, outer, slope, shift = key
            print(f"Source {packet['cluster']} width={width:g} outer={outer:g}/{slope} density={shift:g}", flush=True)
            pair = [build_cluster_sources(packet, width=width, outer_factor=outer, outer_slope=slope, density_shift=shift, nodes=n)
                    for n in [config['source_control']['coarse_nodes'], config['source_control']['fine_nodes']]]
            record = {'cluster': packet['cluster'], 'source_key': key, **source_fidelity(packet, pair[1], config['source_control'])}
            source_cache[(packet['cluster'], key)] = pair
            preflight.append(record)
    write('source_preflight.json', {'rows': preflight, 'pressure_predicted': False})
    primary_width = config['primary_source']['width']
    if any(not r['total_mass_numerical_pass'] or (r['source_key'][0] == primary_width and not r['within_primary_source_limits']) for r in preflight):
        raise RuntimeError('Source preflight unresolved; no reduced-population pressure scoring')
    context_cache = {}
    for packet in packets:
        for scenario in scenarios:
            coarse_source, fine_source = source_cache[(packet['cluster'], source_key(scenario))]
            kwargs = {'packet': packet, 'nuisance': scenario['values']}
            context_cache[(packet['cluster'], scenario['id'])] = {
                'fine': pressure_context(sources=fine_source, nodes=config['pressure_control']['fine_nodes'], **kwargs),
                'pressure_coarse': pressure_context(sources=fine_source, nodes=config['pressure_control']['coarse_nodes'], **kwargs),
                'source_coarse': pressure_context(sources=coarse_source, nodes=config['pressure_control']['fine_nodes'], **kwargs)}
    entries, dispositions, numerical = [], [], []
    for model in models:
        print('Pressure prediction '+model['id'], flush=True)
        raw, controls = [], []
        for scenario in scenarios:
            rows = []
            for packet in packets:
                contexts = context_cache[(packet['cluster'], scenario['id'])]
                answers = {key: predict_from_context(context, model) for key, context in contexts.items()}
                changes = {key: relative_change(answers[key], answers['fine']) for key in ['pressure_coarse', 'source_coarse']}
                passed = all(value is not None and value <= config['pressure_control']['maximum_relative_pressure_change'] for value in changes.values())
                control = {'model': model['id'], 'scenario': scenario['id'], 'cluster': packet['cluster'],
                           'maximum_relative_pressure_change': changes, 'numerical_pass': passed,
                           'comparison_predictions': {key: a.get('prediction') for key, a in answers.items()},
                           'failures': {key: a for key, a in answers.items() if 'prediction' not in a}}
                controls.append(control)
                rows.append({'cluster': packet['cluster'], **answers['fine']})
            raw.append({'model': model['id'], 'scenario': scenario['id'], 'rows': rows})
        admitted = all(c['numerical_pass'] for c in controls)
        disposition = {'model': model['id'], 'numerically_admitted': admitted,
                       'status': 'QUALITY_LIMITED_DEVELOPMENT_RETAINED' if admitted else 'NUMERICAL_OR_FORCE_UNRESOLVED_RETAINED_UNSCORED',
                       'failed_cases': sum(not c['numerical_pass'] for c in controls), 'family_pruned': False}
        dispositions.append(disposition)
        numerical.extend(controls)
        scored = []
        if admitted:
            for entry in raw:
                rows = [score_prediction(packet, row) for packet, row in zip(packets, entry['rows'], strict=True)]
                scored.append({**entry, 'rows': rows, 'summary': aggregate(rows)})
            entries.extend(scored)
        write('model_'+model['id']+'.json', {'model': model, 'disposition': disposition, 'numerical_controls': controls,
                                            'predictions_retained': raw, 'scored_entries': scored if admitted else None})
    nominal = {e['model']: e for e in entries if e['scenario'] == 'nominal'}
    comparisons, sensitivities = [], []
    for model in models:
        if model['id'] not in nominal:
            continue
        subset = [e for e in entries if e['model'] == model['id']]
        matched = []
        for baseline in config['comparators']:
            if baseline == model['id'] or baseline not in nominal:
                continue
            comparisons.append({'model': model['id'], 'baseline': baseline,
                                **comparison(nominal[model['id']]['rows'], nominal[baseline]['rows'])})
            lookup = {e['scenario']: e for e in entries if e['model'] == baseline}
            differences = [{'scenario': e['scenario'], 'difference': e['summary']['equal_cluster_mse_log10_ratio']-
                            lookup[e['scenario']]['summary']['equal_cluster_mse_log10_ratio']} for e in subset]
            matched.append({'baseline': baseline, 'scenario_differences': differences,
                            'scenarios_with_lower_loss': sum(d['difference'] < 0 for d in differences),
                            'minimum_matched_difference': min(d['difference'] for d in differences),
                            'maximum_matched_difference': max(d['difference'] for d in differences)})
        sensitivities.append({'model': model['id'], 'matched_comparisons': matched,
                              'minimum_mse_dex2': min(e['summary']['equal_cluster_mse_log10_ratio'] for e in subset),
                              'maximum_mse_dex2': max(e['summary']['equal_cluster_mse_log10_ratio'] for e in subset)})
    return {'models': models, 'scenarios': scenarios, 'entries': entries, 'dispositions': dispositions,
            'numerical_controls': numerical, 'nominal_comparisons': comparisons, 'global_sensitivities': sensitivities,
            'summary': {'formula_cards': 54, 'comparators': 2, 'clusters': 8, 'scenarios': len(scenarios),
                        'profile_predictions': len(models)*len(scenarios)*len(packets),
                        'numerically_admitted_models': sum(d['numerically_admitted'] for d in dispositions),
                        'disposition_counts': dict(Counter(d['status'] for d in dispositions)),
                        'maximum_relative_pressure_change': {key: max(c['maximum_relative_pressure_change'][key]
                            for c in numerical if c['maximum_relative_pressure_change'][key] is not None)
                            for key in ['pressure_coarse', 'source_coarse']},
                        'nominal_models': {key: e['summary'] for key, e in nominal.items()}},
            'status': 'QUALITY_LIMITED_SPHERICAL_DEVELOPMENT', 'quality_verified_counterexample_count': 0,
            'uncertainty_resolved_counterexample_count': 0, 'independent_replication_count': 0,
            'full_solar_system_pass': False, 'galaxy_profiles_scored': 0, 'lensing_profiles_scored': 0,
            'reserved_clusters_accessed': 0, 'new_raw_data_accessed': False, 'family_pruning_authorized': False,
            'discovery_claim': False,
            'limitations': ['Spherical reconstruction and conditional mass-distance scalings',
                'Unresolved native-to-high-level covariance mapping and joint density/stellar/response covariance',
                'Gas clumping, nonthermal radial shape, time dependence and triaxiality',
                'Finite source reconstruction and unmeasured outer closure; wider-width sensitivities exceed primary stellar fidelity',
                'Local comparison uses historical point-source summary screens, not a full Solar System fit',
                'No galaxy, photon, covariant, dynamical stability or independent confirmation calculation for these cards']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/gravity_length_cluster_pressure_v1.json')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    config = json.loads(args.config.read_bytes())
    paths = [Path(__file__), args.config.resolve(), ROOT/'scripts/run_gravity_xcop_pressure.py',
             *[ROOT/config[k] for k in ['local_result', 'source_audit', 'source_packet', 'inherited_scenario_config']],
             *[ROOT/p for p in config['control_tests']], *sorted((ROOT/'src/invariant_gravity_extensions').glob('*.py'))]

    def hashes():
        return {p.relative_to(ROOT).as_posix(): sha256(p.read_bytes()).hexdigest() for p in paths}

    def write(name, value):
        with (args.output/name).open('x', encoding='utf8', newline='\n') as handle:
            json.dump(jsonable(value), handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')

    before = hashes()
    for path in paths:
        target = args.output/'input-snapshots'/path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
    provenance = {'config': config, 'input_hashes': before, 'started_utc': datetime.now(UTC).isoformat(),
                  'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                  'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__}
    write('started.json', provenance)
    try:
        control = subprocess.run([sys.executable, '-m', 'pytest', *config['control_tests'], '-q'], cwd=ROOT,
                                 env={**os.environ, 'PYTHONPATH': str(ROOT/'src'), 'OPENBLAS_NUM_THREADS': '1'},
                                 capture_output=True, text=True, check=False)
        write('controls.json', {'command': control.args, 'exit_code': control.returncode, 'stdout': control.stdout, 'stderr': control.stderr})
        if control.returncode:
            raise RuntimeError('Synthetic controls failed before pressure prediction')
        result = campaign(config, write)
        if hashes() != before:
            raise RuntimeError('Registered input changed during campaign')
        write('result.json', {**provenance, **result})
        write('receipt.json', {'status': 'COMPLETED_AT_DECLARED_SCOPE',
                               'result_sha256': sha256((args.output/'result.json').read_bytes()).hexdigest()})
        print(json.dumps({key: value for key, value in result['summary'].items() if key != 'nominal_models'}))
    except Exception as exc:
        write('failure.json', {'status': 'EXECUTION_FAILURE_RETAINED_NOT_PHYSICS_REJECTION', 'error': str(exc)})
        raise


if __name__ == '__main__':
    main()
