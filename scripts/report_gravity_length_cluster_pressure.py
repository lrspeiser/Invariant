"""Independent loss and Cartesian-force/Simpson replay of cluster transfer."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import matplotlib
import numpy as np
from scipy.integrate import cumulative_simpson

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.cluster_pressure import KPC, MU, MU_E, PROTON_MASS
from invariant_gravity_extensions.length_cluster_pressure import array_packet, pressure_context
from invariant_gravity_extensions.length_screening import LengthScreening, anomalous_flux
from invariant_gravity_extensions.smooth_spherical_source import build_cluster_sources


def write_json(path, value):
    with path.open('x', encoding='utf8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def replay_losses(result, packets):
    by_cluster = {p['cluster']: p for p in packets}
    lookup, count, covariance_count = {}, 0, 0
    max_error = 0.
    for entry in result['entries']:
        losses = []
        for row in entry['rows']:
            p = by_cluster[row['cluster']]
            predicted, observed = np.asarray(row['prediction']), np.asarray(row['observed'])
            residual = predicted-observed
            ids, anchor = row['indices'], row['anchor']
            expected_observed = p['pressure'][ids]*row['pressure_scale']
            np.testing.assert_allclose(observed, expected_observed, rtol=1e-14)
            loss = float(np.mean(np.log10(predicted/observed)**2))
            np.testing.assert_allclose(loss, row['mse_log10_ratio'], rtol=1e-12)
            losses.append(loss)
            T = np.eye(len(p['pressure']))[ids]
            T[:, anchor] -= np.asarray(row['boundary_coefficients'])
            covs = {'transferred_correlation': p['covariance'], 'native_scaled': p['native_scaled_covariance'],
                    'diagonal_quoted': np.diag(p['pressure_error']**2)}
            for name, C in covs.items():
                C = T@C@T.T*row['pressure_scale']**2
                # Direct dense solve is independent of the campaign's standardized Cholesky whitening.
                value = float(residual@np.linalg.solve(C, residual)/len(residual))
                expected = row['whitened_mean_squared_residual'][name]
                error = abs(value-expected)/max(1., abs(expected))
                max_error = max(error, max_error)
                np.testing.assert_allclose(value, expected, rtol=1e-10, atol=1e-10)
                covariance_count += 1
            count += 1
        np.testing.assert_allclose(np.mean(losses), entry['summary']['equal_cluster_mse_log10_ratio'], rtol=1e-12)
        lookup[(entry['model'], entry['scenario'])] = dict(zip([r['cluster'] for r in entry['rows']], losses, strict=True))
    influence_count = 0
    for record in result['nominal_comparisons']:
        a, b = lookup[(record['model'], 'nominal')], lookup[(record['baseline'], 'nominal')]
        names = list(a)
        delta = np.array([a[k]-b[k] for k in names])
        omit = int(np.argmax(abs(delta)))
        assert record['leave_most_influential_out']['omitted'] == names[omit]
        np.testing.assert_allclose(np.delete(delta, omit).mean(), record['leave_most_influential_out']['mean_difference'], atol=1e-14)
        np.testing.assert_allclose(np.sort(delta)[1:-1].mean(), record['symmetric_trim']['mean_difference'], atol=1e-14)
        assert record['raw_comparative_win_count'] == int(np.count_nonzero(delta < 0))
        assert record['raw_comparative_loss_count'] == int(np.count_nonzero(delta > 0))
        assert record['quality_verified_counterexample_count'] == record['uncertainty_resolved_counterexample_count'] == 0
        influence_count += 1
    return {'profile_loss_replays': count, 'covariance_loss_replays': covariance_count,
            'maximum_scaled_covariance_loss_difference': max_error, 'object_influence_replays': influence_count}


def cartesian_simpson_replay(result, packets):
    """Full 3D Cartesian flux on the radial axis; separate pressure quadrature."""
    config = result['config']
    scenario = next(s for s in result['scenarios'] if s['id'] == 'nominal')
    nominal = {e['model']: e for e in result['entries'] if e['scenario'] == 'nominal'}
    records = []
    for packet in packets:
        print('Independent Cartesian pressure replay '+packet['cluster'], flush=True)
        source = build_cluster_sources(packet, **scenario['source'], nodes=config['source_control']['fine_nodes'])
        context = pressure_context(packet, source, scenario['values'], nodes=8193)
        fields = context['fields']
        r, g, gp, gpp = (fields[key] for key in ['radius_m', 'gbar', 'gbar_first', 'gbar_second'])
        p = np.array([g, np.zeros_like(g), np.zeros_like(g)])
        H = np.zeros((3, 3, len(r)))
        H[0, 0], H[1, 1], H[2, 2] = gp, g/r, g/r
        grad_hessian_norm = np.zeros_like(p)
        grad_hessian_norm[0] = 2*gp*gpp+4*g/r*(gp/r-g/r**2)
        grad_laplacian = np.zeros_like(p)
        # Poisson gives grad(laplacian psi)=4*pi*G*grad(rho); use source rho' directly.
        grad_laplacian[0] = 4*np.pi*6.67430e-11*fields['density_gradient']
        ne = fields['gas_density']/(MU_E*PROTON_MASS)
        for model in result['models']:
            if model['id'] not in nominal:
                continue
            if model['family'] == 'newtonian':
                acceleration = g
            elif model['family'] == 'rar_comparator':
                acceleration = g/(-np.expm1(-np.sqrt(g/model['a0_m_s2'])))
            else:
                acceleration = g+anomalous_flux(LengthScreening(model['shape'], model['epsilon']), p, H,
                    grad_hessian_norm, grad_laplacian, model['length_pc']*KPC/1000, model['a0_m_s2'])[0]
            integral = cumulative_simpson(MU*PROTON_MASS*ne*acceleration, x=r, initial=0.)
            fraction = context['fraction']
            pressure = (1-fraction)*(context['boundary_si']/(1-fraction[-1])+integral[-1]-integral)/1.602176634e-10
            loc = context['target_locations']
            row = next(x for x in nominal[model['id']]['rows'] if x['cluster'] == packet['cluster'])
            relative = float(np.max(abs(pressure[loc]/row['prediction']-1)))
            force = float(np.max(abs(acceleration[loc]/row['predicted_acceleration_m_s2']-1)))
            records.append({'model': model['id'], 'cluster': packet['cluster'], 'pressure': pressure[loc].tolist(),
                            'maximum_relative_pressure_change': relative, 'maximum_relative_force_difference': force})
    maximum = max(r['maximum_relative_pressure_change'] for r in records)
    if maximum > config['pressure_control']['maximum_relative_pressure_change']:
        raise RuntimeError('Independent Simpson replay exceeds frozen pressure tolerance')
    if max(r['maximum_relative_force_difference'] for r in records) > 1e-10:
        raise RuntimeError('Cartesian force replay disagreement')
    return {'rows': records, 'maximum_relative_pressure_change': maximum,
            'maximum_relative_force_difference': max(r['maximum_relative_force_difference'] for r in records)}


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
        raise ValueError('Result digest changed')
    for relative, expected in result['input_hashes'].items():
        if sha256((args.run/'input-snapshots'/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Input snapshot changed: '+relative)
    (args.verification/'report-script.py').write_bytes(Path(__file__).read_bytes())
    write_json(args.verification/'started.json', {'started_utc': datetime.now(UTC).isoformat(), 'result_sha256': digest,
        'report_script_sha256': sha256(Path(__file__).read_bytes()).hexdigest(),
        'independent_replay': '8193 radial nodes; Cartesian action flux and Simpson pressure quadrature for all nominal models',
        'tolerance': result['config']['pressure_control']['maximum_relative_pressure_change']})
    # The replay uses current source code only if byte-identical to the executed snapshot.
    for relative, expected in result['input_hashes'].items():
        if relative.startswith('src/') and sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Current replay module differs from frozen implementation: '+relative)
    parent = args.run/'input-snapshots'/result['config']['source_packet']
    packets = [array_packet(p) for p in json.loads(parent.read_bytes())['packets']]
    verification = {'result_sha256': digest, 'verified_input_snapshots': len(result['input_hashes']),
                    **replay_losses(result, packets), 'independent_nominal_replay': cartesian_simpson_replay(result, packets)}
    write_json(args.verification/'result.json', verification)
    write_json(args.verification/'receipt.json', {'status': 'VERIFIED_AT_DECLARED_SCOPE',
                'result_sha256': sha256((args.verification/'result.json').read_bytes()).hexdigest()})
    nominal = result['summary']['nominal_models']
    comparator = 'empirical_RAR_a0_1.2e-10'
    candidates = [m for m in result['models'] if m['family'] == 'length_screening' and m['id'] in nominal]
    rows = []
    for m in candidates:
        comp = next(c for c in result['nominal_comparisons'] if c['model'] == m['id'] and c['baseline'] == comparator)
        sensitivity = next(s for s in result['global_sensitivities'] if s['model'] == m['id'])
        match = next(c for c in sensitivity['matched_comparisons'] if c['baseline'] == comparator)
        rows.append({**m, **nominal[m['id']], 'comparison_with_RAR': comp, 'matched_sensitivity_with_RAR': match})
    summary = {'run_result_sha256': digest, 'verification_sha256': sha256((args.verification/'result.json').read_bytes()).hexdigest(),
               'campaign': {key: value for key, value in result['summary'].items() if key != 'nominal_models'},
               'comparators': {key: nominal[key] for key in result['config']['comparators'] if key in nominal},
               'cards': rows, 'scope': result['status'], 'limitations': result['limitations'],
               'quality_verified_counterexamples': 0, 'uncertainty_resolved_counterexamples': 0,
               'discovery_claim': False}
    args.outputs.mkdir(parents=True, exist_ok=True)
    write_json(args.outputs/'Gravity-length-cluster-pressure-summary.json', summary)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.7), sharey=True)
    for ax, shape in zip(axes, [.5, 1., 2.], strict=True):
        for a0, color in zip([5e-11, 1.2e-10, 2e-10], ['#6a51a3', '#2878b5', '#e07b23'], strict=True):
            selected = sorted([r for r in rows if r['shape'] == shape and r['a0_m_s2'] == a0], key=lambda r: r['length_pc'])
            x = np.arange(len(selected))
            y = [r['median_cluster_median_pressure_ratio'] for r in selected]
            ax.plot(x, y, color=color, alpha=.6, label=f'a₀={a0:.1e} m/s²')
            for ix, r in enumerate(selected):
                within = r['prior_local_status'].startswith('WITHIN_')
                ax.scatter(ix, y[ix], color=color if within else 'white', edgecolor=color, s=52, zorder=3)
        ax.axhline(1., color='0.6', linewidth=.8)
        ax.axhline(nominal[comparator]['median_cluster_median_pressure_ratio'], color='black', linestyle='--', label='RAR comparator')
        ax.set_xticks(range(6), ['0', '.001', '.01', '.1', '1', '10'])
        ax.set_xlabel('Universal length ℓ (pc)')
        ax.set_title(f'Shape m={shape:g}')
        ax.grid(axis='y', alpha=.2)
    axes[0].set_ylabel('Median cluster predicted / observed pressure')
    axes[0].legend(fontsize=8, loc='lower right')
    fig.suptitle('Same constants: cluster pressure versus local gravity screening', fontsize=13)
    fig.text(.5, .025, 'Filled: within both historical local screens. Hollow: outside. Eight development clusters; conditional spherical sources.', ha='center', fontsize=9)
    fig.tight_layout(rect=[0, .07, 1, .94])
    for extension in ['png', 'svg']:
        fig.savefig(args.outputs/f'Gravity-length-cluster-pressure-comparison.{extension}', dpi=180)
    plt.close(fig)
    print(json.dumps({'cards': len(rows), 'verification': {key: value for key, value in verification.items() if key != 'independent_nominal_replay'},
                      'independent_max_pressure_change': verification['independent_nominal_replay']['maximum_relative_pressure_change']}))


if __name__ == '__main__':
    main()
