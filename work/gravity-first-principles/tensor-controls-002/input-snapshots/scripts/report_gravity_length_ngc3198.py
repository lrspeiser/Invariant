"""Independent velocity/covariance replay and cross-regime development report."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def write(path, data):
    with path.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(data, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def replay(result, run):
    old = json.loads((run/'input-snapshots'/result['config']['inherited_config']).read_bytes())
    maps = json.loads((run/'input-snapshots'/result['config']['source_profiles']).read_bytes())
    fields = {name.removeprefix('fields_').removesuffix('.json'): json.loads((run/name).read_bytes())
              for name in result['field_record_names']}
    lookup = {key: {(row['model'], row['distance_scale']): row for row in family['predictions']} for key, family in fields.items()}
    primary = result['primary_rows']
    radius = np.array([row['nominal_radius_kpc'] for row in primary])
    published = np.array([row['published_velocity_kms'] for row in primary])
    quoted = np.array([row['published_random_error_kms'] for row in primary])
    positions = np.searchsorted(fields['primary_fine']['radii_kpc'], radius)
    np.testing.assert_array_equal(np.asarray(fields['primary_fine']['radii_kpc'])[positions], radius)
    max_velocity = max_loss = 0.
    scored = unscored = influence = count_length = 0
    for scenario in result['scenarios']:
        distance = (maps['metadata']['distance_mpc']+scenario['distance_offset_mpc'])/maps['metadata']['distance_mpc']
        assert distance == scenario['distance_scale']
        inc = maps['metadata']['inclination_deg']+scenario['inclination_offset_deg']
        conversion = np.sin(np.deg2rad(old['geometry']['published_inclination_deg']))/np.sin(np.deg2rad(inc))
        observed, errors = published*conversion, quoted*conversion
        np.testing.assert_allclose(observed, scenario['observed_velocity_kms'], rtol=1e-14)
        np.testing.assert_allclose(errors, scenario['random_error_kms'], rtol=1e-14)
        rar = scenario['candidate_results']['RAR_2016_ALGEBRAIC']
        for name, row in scenario['candidate_results'].items():
            source = lookup[scenario['source_variant']+'_fine'][(name, distance)]
            if 'physical_length_pc' in source:
                np.testing.assert_allclose(source['nominal_coordinate_length_kpc']*distance*1000,
                                           source['physical_length_pc'], rtol=1e-14, atol=1e-15)
                count_length += 1
            gate = result['numerical_admission'][name]
            if row['status'] != 'QUALITY_LIMITED_DEVELOPMENT_RETAINED':
                assert row['predicted_velocity_kms'] is None
                assert not gate['numerical_pass'] or not gate['complete_inward_branch']
                unscored += 1
                continue
            assert gate['numerical_pass'] and gate['complete_inward_branch']
            force = np.asarray(source['inward_force'])[positions]
            speed = np.sqrt(distance*radius*force)
            max_velocity = max(max_velocity, float(np.max(abs(speed-np.asarray(row['predicted_velocity_kms'])))))
            np.testing.assert_allclose(speed, row['predicted_velocity_kms'], rtol=1e-13, atol=1e-12)
            residual, z = speed-observed, (speed-observed)/errors
            u = observed/np.tan(np.deg2rad(inc))*np.deg2rad(old['geometry']['published_inclination_error_deg'])/errors
            # Sherman-Morrison replay, independent of the campaign's dense covariance solve.
            scores = {'random_error_loss': float(np.mean(z*z)),
                      'inclination_covariance_loss': float((z@z-(z@u)**2/(1+u@u))/len(z)),
                      'five_kms_floor_loss': float(np.mean(residual**2/(errors**2+25))),
                      'velocity_RMS_kms': float(np.sqrt(np.mean(residual*residual))),
                      'median_predicted_observed_ratio': float(np.median(speed/observed))}
            for metric, value in scores.items():
                max_loss = max(max_loss, abs(value-row[metric])/max(1., abs(value)))
                np.testing.assert_allclose(value, row[metric], rtol=3e-11, atol=1e-10)
            if 'standardized_residual' in rar:
                delta = z*z-np.asarray(rar['standardized_residual'])**2
                omitted = int(np.argmax(abs(delta)))
                trimmed = np.sort(delta)
                n = int(np.floor(len(delta)*old['scoring']['symmetric_radial_influence_trim_fraction_each_tail']))
                saved = row['influence']
                assert saved['most_influential_row_position'] == omitted
                np.testing.assert_allclose(np.delete(delta, omitted).mean(), saved['drop_one_radial_loss_difference'], atol=1e-10)
                np.testing.assert_allclose(trimmed[n:len(delta)-n].mean(), saved['trimmed_radial_loss_difference'], atol=1e-10)
                influence += 1
            scored += 1
    comparison_replays = 0
    for model, gate in result['numerical_admission'].items():
        for comp in gate['comparisons']:
            d = comp['distance_scale']
            a = lookup[comp['reference'].replace('/', '_')][(model, d)]
            b = lookup[comp['alternative'].replace('/', '_')][(model, d)]
            newton = lookup[comp['reference'].replace('/', '_')][('NEWTON_BARYONS', d)]
            delta = (np.asarray(b['inward_force'])-a['inward_force'])/np.maximum(abs(np.asarray(a['inward_force'])), newton['inward_force'])
            np.testing.assert_allclose(delta, comp['normalized_difference'], rtol=1e-14, atol=1e-15)
            assert abs(float(np.max(abs(delta)))-comp['maximum']) < 1e-14
            comparison_replays += 1
    return {'scored_scenario_replays': scored, 'unscored_scenario_records_verified': unscored,
            'fixed_physical_length_records_verified': count_length, 'radial_influence_replays': influence,
            'numerical_comparison_replays': comparison_replays, 'maximum_velocity_difference_kms': max_velocity,
            'maximum_scaled_loss_difference': max_loss, 'source_approximation_independently_validated': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--verification', type=Path, required=True)
    parser.add_argument('--outputs', type=Path, required=True)
    args = parser.parse_args()
    args.verification.mkdir(parents=True, exist_ok=False)
    result = json.loads((args.run/'result.json').read_bytes())
    digest = sha256((args.run/'result.json').read_bytes()).hexdigest()
    if json.loads((args.run/'receipt.json').read_bytes())['result_sha256'] != digest:
        raise ValueError('Result digest mismatch')
    for path, expected in result['input_hashes'].items():
        if sha256((args.run/'input-snapshots'/path).read_bytes()).hexdigest() != expected:
            raise ValueError('Input snapshot digest changed: '+path)
    field_hashes = {name: sha256((args.run/name).read_bytes()).hexdigest() for name in result['field_record_names']}
    provenance = {'started_utc': datetime.now(UTC).isoformat(), 'run_result_sha256': digest,
                  'input_snapshots_verified': len(result['input_hashes']), 'field_hashes': field_hashes,
                  'script_sha256': sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.verification/'report-script.py').write_bytes(Path(__file__).read_bytes())
    write(args.verification/'started.json', provenance)
    verified = {**provenance, **replay(result, args.run)}
    write(args.verification/'result.json', verified)
    write(args.verification/'receipt.json', {'status': 'REPLAY_VERIFIED_AT_DECLARED_SCOPE',
            'result_sha256': sha256((args.verification/'result.json').read_bytes()).hexdigest()})
    cluster = json.loads((args.run/'input-snapshots'/result['config']['cluster_result']).read_bytes())
    cluster_comp = {row['model']: row for row in cluster['nominal_comparisons'] if row['baseline'] == 'empirical_RAR_a0_1.2e-10'}
    summaries = []
    for card in result['cards']:
        summary = result['summary'][card['id']]
        summaries.append({'card': card, 'galaxy': summary, 'cluster_comparison_with_RAR': cluster_comp[card['id']],
                          'within_local_screens': card['prior_local_status'].startswith('WITHIN_')})
    open_cluster_local = [r for r in summaries if r['within_local_screens'] and r['cluster_comparison_with_RAR']['mean_difference'] < 0]
    cross_regime = {'local_compatible_and_cluster_improving_cards': len(open_cluster_local),
                   'galaxy_numeric_admitted_among_them': sum(r['galaxy']['nominal']['status'] == 'QUALITY_LIMITED_DEVELOPMENT_RETAINED' for r in open_cluster_local),
                   'nominal_galaxy_better_than_RAR_counts_by_metric': {},
                   'full_three_regime_validated_cards': 0}
    baseline = result['summary']['RAR_2016_ALGEBRAIC']['nominal']
    for metric in ['random_error_loss', 'inclination_covariance_loss', 'five_kms_floor_loss']:
        cross_regime['nominal_galaxy_better_than_RAR_counts_by_metric'][metric] = sum(
            metric in r['galaxy']['nominal'] and metric in baseline and r['galaxy']['nominal'][metric] < baseline[metric] for r in open_cluster_local)
    output = {'run_sha256': digest, 'verification_sha256': sha256((args.verification/'result.json').read_bytes()).hexdigest(),
              'cross_regime': cross_regime, 'status_counts': result['status_counts'], 'cards': summaries,
              'comparators': {name: result['summary'][name] for name in ['NEWTON_BARYONS', 'RAR_2016_ALGEBRAIC']},
              'source_projection_diagnostics': result['source_projection_diagnostics'],
              'quality_verified_counterexamples': 0, 'uncertainty_resolved_counterexamples': 0,
              'first_principles_derivation': False, 'discovery_claim': False}
    args.outputs.mkdir(parents=True, exist_ok=True)
    write(args.outputs/'Gravity-length-NGC3198-summary.json', output)
    nominal = next(s for s in result['scenarios'] if s['source_variant'] == 'primary' and s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0)
    radius = np.array([r['nominal_radius_kpc'] for r in result['primary_rows']])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.9), sharey=True)
    for ax, shape in zip(axes, [.5, 1., 2.], strict=True):
        ax.errorbar(radius, nominal['observed_velocity_kms'], yerr=nominal['random_error_kms'],
                    color='0.2', marker='o', markersize=3, linewidth=.8, linestyle='none', label='Gas-traced observations')
        for a0, color in zip([5e-11, 1.2e-10, 2e-10], ['#6a51a3', '#2878b5', '#e07b23'], strict=True):
            card = next(c for c in result['cards'] if c['shape'] == shape and c['a0_m_s2'] == a0 and c['length_pc'] == 1.)
            row = nominal['candidate_results'][card['id']]
            if row['predicted_velocity_kms'] is not None:
                ax.plot(radius, row['predicted_velocity_kms'], color=color, label=f'a₀={a0:.1e} m/s²')
        row = nominal['candidate_results']['RAR_2016_ALGEBRAIC']
        if row['predicted_velocity_kms'] is not None:
            ax.plot(radius, row['predicted_velocity_kms'], color='black', linestyle='--', linewidth=1.2, label='RAR comparator')
        ax.set_title(f'm={shape:g}, universal ℓ=1 pc')
        ax.set_xlabel('Radius at nominal distance (kpc)')
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Circular speed (km/s)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .915), ncol=5, fontsize=9)
    fig.suptitle('NGC3198: same gravity constants, conditional galaxy-source reconstruction', fontsize=13)
    fig.text(.5, .025, 'Nominal development scenario. Angular source ringing remains unresolved; these curves are not physical validation.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .06, 1, .85])
    for ext in ['png', 'svg']:
        fig.savefig(args.outputs/f'Gravity-length-NGC3198-comparison.{ext}', dpi=180)
    plt.close(fig)
    print(json.dumps({'cross_regime': cross_regime, 'verification': {key: value for key, value in verified.items() if key != 'field_hashes'}}))


if __name__ == '__main__':
    main()
