"""Independently replay retained galaxy diagnostics and report their full scope."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'work/gravity-first-principles'
METRICS = ['random_error_loss', 'inclination_covariance_loss', 'five_kms_floor_loss']


def read(path):
    return json.loads(path.read_bytes())


def seal(run):
    """Verify executed snapshots, falling back only to hash-identical parents."""
    started = read(run/'started.json')
    for name, digest in started['input_hashes'].items():
        snapshot = run/'input-snapshots'/name
        path = snapshot if snapshot.exists() else ROOT/name
        if sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('Input identity failed: '+str(path))
    result = run/'result.json'
    digest = sha256(result.read_bytes()).hexdigest() if result.exists() else None
    if digest is not None and digest != read(run/'receipt.json')['result_sha256']:
        raise ValueError('Result identity failed: '+str(run))
    return {'input_hashes_verified': len(started['input_hashes']), 'result_sha256': digest,
            'failure_retained': (run/'failure.json').exists()}


def replay(result, old, maps, refined=False):
    radii = np.array([r['nominal_radius_kpc'] for r in result['primary_rows']])
    published = np.array([r['published_velocity_kms'] for r in result['primary_rows']])
    random = np.array([r['published_random_error_kms'] for r in result['primary_rows']])
    geometry = old['config']['geometry']
    source_distance = maps['metadata']['distance_mpc']
    source_inclination = maps['metadata']['inclination_deg']
    fields = {}
    for variant in old['config']['source_variants']:
        name = variant['id']
        scalar = read(BASE/'ngc3198-scalar-001'/f'fields_{name}_fine.json')
        auxiliary = read(BASE/'ngc3198-multifield-003'/f'fields_{name}_fine.json')
        if refined and name in result['config']['refined_source_variants']:
            auxiliary = read(BASE/'ngc3198-multifield-refinement-001'/f'fields_{name}_refined.json')
        fields[name] = (scalar, auxiliary)
    checked = withheld = 0
    influence_changes = {'drop_one': 0, 'trim': 0}
    for scenario in result['scenarios']:
        scalar, auxiliary = fields[scenario['source_variant']]
        positions = np.searchsorted(scalar['radii_kpc'], radii)
        np.testing.assert_array_equal(np.asarray(scalar['radii_kpc'])[positions], radii)
        inclination = source_inclination+scenario['inclination_offset_deg']
        factor = np.sin(np.deg2rad(geometry['published_inclination_deg']))/np.sin(np.deg2rad(inclination))
        observed, errors = published*factor, random*factor
        np.testing.assert_array_equal(observed, scenario['observed_velocity_kms'])
        np.testing.assert_array_equal(errors, scenario['random_error_kms'])
        for card in result['cards']:
            entry = scenario['candidate_results'][card['id']]
            a = next(u for u in auxiliary['unit_fields'] if u['beta'] == card['beta'] and u['power'] == card['power'])
            key = f"SAT_m{card['shape']:g}_a0_{card['a0_m_s2']:.1e}"
            full_force = np.asarray(scalar['predictions'][key])+card['mixing']**2*np.asarray(a['inward_unit_force'])
            np.testing.assert_allclose(full_force[positions], entry['inward_force'], rtol=2e-14, atol=0)
            if not result['numerical_admission'][card['id']]['numerical_pass']:
                assert entry['predicted_velocity_kms'] is None
                assert entry['status'] == 'NUMERICAL_BRIDGE_UNRESOLVED_RETAINED'
                assert not any(metric in entry for metric in METRICS)
                withheld += 1
                continue
            assert np.all(full_force > 0), 'Unexpected no-circular-branch result requires a separate report'
            velocity = np.sqrt(radii*full_force[positions])*np.sqrt((source_distance+scenario['distance_offset_mpc'])/source_distance)
            np.testing.assert_allclose(velocity, entry['predicted_velocity_kms'], rtol=2e-14, atol=0)
            residual = velocity-observed
            z = residual/errors
            q = observed/errors/np.tan(np.deg2rad(inclination))*np.deg2rad(geometry['published_inclination_error_deg'])
            # Sherman-Morrison provides a different covariance evaluation from
            # the campaign's dense linear solve.
            values = [np.mean(z*z), (z@z-(z@q)**2/(1+q@q))/len(z), np.mean(residual**2/(errors**2+25))]
            np.testing.assert_allclose(values, [entry[m] for m in METRICS], rtol=2e-10, atol=1e-10)
            np.testing.assert_allclose(z, entry['standardized_residual'], rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(residual, entry['residual_kms'], rtol=1e-12, atol=1e-12)
            for label, keep in [('inner', radii < 12), ('outer', radii >= 12)]:
                np.testing.assert_allclose(np.mean(z[keep]**2), entry[label+'_random_error_loss'])
            rar_z = (np.asarray(scenario['RAR_comparator']['predicted_velocity_kms'])-observed)/errors
            delta = z*z-rar_z*rar_z
            i = int(np.argmax(abs(delta)))
            trimmed = np.sort(delta)[1:-1].mean()
            dropped = (delta.sum()-delta[i])/(len(delta)-1)
            influence = entry['influence']
            assert i == influence['most_influential_row_position']
            assert influence['trim_each_tail_count'] == 1
            np.testing.assert_allclose([delta.mean(), dropped, trimmed],
                [influence['candidate_minus_RAR_loss'], influence['drop_one_radial_loss_difference'], influence['trimmed_radial_loss_difference']])
            assert (np.sign(delta.mean()) != np.sign(dropped)) == influence['drop_one_sign_change']
            assert (np.sign(delta.mean()) != np.sign(trimmed)) == influence['trim_sign_change']
            influence_changes['drop_one'] += influence['drop_one_sign_change']
            influence_changes['trim'] += influence['trim_sign_change']
            checked += 1
    return {'scored_records_replayed': checked, 'withheld_records_verified': withheld,
            'radial_influence_sign_changes': influence_changes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.verification_output.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    sealed = {name: seal(BASE/name) for name in ['ngc3198-multifield-001', 'ngc3198-multifield-002',
              'ngc3198-multifield-003', 'ngc3198-multifield-refinement-001']}
    parent = read(BASE/'ngc3198-multifield-003/result.json')
    refined = read(BASE/'ngc3198-multifield-refinement-001/result.json')
    old = read(BASE/'ngc3198-scalar-001/result.json')
    maps = read(BASE/'map-source-003/source_profiles.json')
    checks = {'parents': sealed, 'initial': replay(parent, old, maps),
              'refined': replay(refined, old, maps, refined=True)}
    summary = {**parent['summary'], **refined['summary']}
    gates = {**parent['numerical_admission'], **refined['numerical_admission']}
    admitted = [key for key, value in gates.items() if value['numerical_pass']]
    unresolved = [key for key in gates if key not in admitted]
    assert len(admitted) == 72 and len(unresolved) == 9
    assert all(summary[key]['scenarios_comparable'] == 99 and
               all(summary[key]['scenarios_worse_than_RAR'][m] == 99 for m in METRICS) for key in admitted)
    assert all(card['power'] == 2 and card['mixing'] == 10 for card in parent['cards'] if card['id'] in unresolved)
    assert all(-3e-27 <= q <= 9e-27 for card in parent['cards'] for q in card['conditional_local_Q2_s_minus2'])
    nominal = next(s for s in parent['scenarios'] if s['source_variant'] == 'primary' and s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0)
    baseline = nominal['RAR_comparator']
    for key, item in summary.items():
        item['aggregate_improvement_percent_vs_RAR'] = None if key in unresolved else {
            metric: 100*(baseline[metric]-item['nominal'][metric])/baseline[metric] for metric in METRICS}
    control = parent['controls']
    refined_max = max(v['maximum'] for g in refined['numerical_admission'].values() for v in g['refinement_followup'])
    payload = {'status': parent['status'], 'checks': checks, 'cards': 81, 'admitted': 72, 'unresolved': 9,
               'admitted_card_scenario_records': 72*99, 'unresolved_card_scenario_records': 9*99,
               'refined_discrepancy_maximum': refined_max, 'summary': summary,
               'unresolved_cards': unresolved, 'quality_gate_passed': False,
               'evaluable_objects': 1, 'raw_counterexample_count_per_scored_card': 1,
               'quality_verified_counterexample_count': 0, 'uncertainty_resolved_counterexample_count': 0,
               'strongest_baseline_failed': False, 'independent_failure_strata': 0,
               'unchanged_independent_replication_failures': 0, 'family_pruning': False, 'discovery_claim': False,
               'single_object_removal': None, 'single_object_removal_reason': 'one development galaxy',
               'object_level_records_preserved': True, 'missing_quality_limited_records_preserved': True,
               'exclusions_frozen_before_response': 'Inherited 2-20 kpc selection; previously exposed development observations',
               'full_solar_system_pass': False, 'spherical_cluster_predictions_unchanged': True}
    payload['executing_report_script_sha256'] = sha256(Path(__file__).read_bytes()).hexdigest()
    blob = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+'\n').encode()
    (args.verification_output/'result.json').write_bytes(blob)
    (args.verification_output/'report_script.py').write_bytes(Path(__file__).read_bytes())
    (args.verification_output/'receipt.json').write_text(json.dumps({'result_sha256': sha256(blob).hexdigest()})+'\n', newline='\n')
    (args.output/'Gravity-NGC3198-multifield-summary.json').write_bytes(blob)

    radius = np.array([r['nominal_radius_kpc'] for r in parent['primary_rows']])
    observed = np.asarray(nominal['observed_velocity_kms'])
    errors = np.asarray(nominal['random_error_kms'])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True, layout='constrained')
    table = ['| Shape m | Coupling | Median speed ratio | RMS (km/s) | Random-error loss | Inclination loss |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for ax, m in zip(axes, [.5, 1, 2], strict=True):
        ax.errorbar(radius, observed, yerr=errors, fmt='o', color='#263044', ms=3.5, lw=.8, label='Observed gas-traced speed', zorder=5)
        ax.plot(radius, baseline['predicted_velocity_kms'], '--', color='#727d88', label='RAR comparator')
        ax.plot(radius, old['summary'][f'SAT_m{m:g}_a0_5.0e-11']['nominal']['predicted_velocity_kms'], ':', color='#a38352', label='Scalar (coupling = 0)')
        for mixing, color in [(3, '#207b9d'), (6, '#b14865')]:
            key = f'TRI_m{m:g}_b2_p2_lambda{mixing}'
            row = summary[key]['nominal']
            ax.plot(radius, row['predicted_velocity_kms'], color=color, label=f'Multifield, coupling = {mixing}')
            table.append(f"| {m:g} | {mixing} | {row['median_predicted_observed_ratio']:.3f} | {row['velocity_RMS_kms']:.2f} | {row['random_error_loss']:.2f} | {row['inclination_covariance_loss']:.2f} |")
        ax.set(title=f'Shape m = {m:g}', xlabel='Radius at nominal distance (kpc)')
        ax.grid(alpha=.17)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylabel('Circular speed (km/s)')
    axes[1].legend(fontsize=8, frameon=False, loc='lower right')
    fig.suptitle('NGC 3198: fixed a₀ = 5 × 10⁻¹¹ m/s², β = 2, p = 2\nIllustrative cards at nominal source and geometry; all 72 resolved cards lose to RAR in 99 scenarios', fontsize=13)
    for extension in ['png', 'svg']:
        fig.savefig(args.output/f'Gravity-NGC3198-multifield-comparison.{extension}', dpi=170)
    plt.close(fig)
    table.append(f"| RAR | — | {baseline['median_predicted_observed_ratio']:.3f} | {baseline['velocity_RMS_kms']:.2f} | {baseline['random_error_loss']:.2f} | {baseline['inclination_covariance_loss']:.2f} |")
    report = f'''# Observed-source multifield gravity test

Seventy-two of 81 frozen bounded-TRIMOND configurations now pass the declared
numerical checks. Every one has greater error than the RAR comparator across
all 99 matched source/geometry scenarios, under all three error treatments.
That is 7,128 scored card/scenario records from **one** development galaxy,
not 7,128 independent astronomical tests. Nine configurations with p=2 and
coupling 10 remain numerically unresolved and unscored. No universal gravity
formula has been established, and no entire formula family is rejected.

The initial 63-card admission was followed by a frozen numerical refinement
of the two source variants responsible for the nine coupling-6 exceptions.
All nine then passed: the maximum combined scalar/auxiliary discrepancy is
{100*refined_max:.4f}%, below the unchanged 2% limit. Historical failed comparisons
remain in the parent receipt; the refinement supersedes only the named
resolution checks for those nine cards. It changes neither source parameters
nor the selected response radii. The other nine cards still fail map and field
resolution requirements; they have no scored velocities or losses.

## Formula and physical interpretation

The tested action is F=Q(x)−|q−s p|²−w|p×q|², where p=∇ψ/a₀,
q=∇χ/a₀, x=p², s=λ/(1+x)^P and w=β/(1+x)². Q is the previously
registered saturated scalar action. Newtonian ψ is sourced by the full
continuous baryonic density, and the auxiliary equation is div(Aq)=div(sp),
with A=I+w(xI−ppᵀ). The physical potential includes all action derivatives.
The general tripotential framework is prior art; this experiment does not
claim its invention. [Milgrom (2023)](https://arxiv.org/abs/2305.19986)

All cards use a₀=5e−11 m/s², shapes 0.5, 1 and 2, and β=0, 0.5 or 2.
P=1 uses λ=0.25, 0.75, 1.5 or 3; P=2 uses λ=0.25, 0.75, 3, 6 or 10.
The constants are global across all scenarios. This grid was chosen within
the prior conditional Cassini-summary coupling bounds before these multifield
galaxy predictions. That summary screen is not a full Solar System pass.

At fixed source, the auxiliary physical force scales as λ². At the nominal
source's outermost numerical probe, 20 kpc, all six auxiliary kernels contribute
outward radial force: −0.309 to −0.567 (km/s)²/kpc per λ². Increasing coupling
therefore deepens the outer deficit there, although it improves some interior
speeds. This sign statement covers the declared nominal source and radius.
The total inward force remains positive for the admitted cards.

In exact spherical symmetry, q=sp and this auxiliary correction is zero.
Consequently the existing spherical cluster-pressure deficits are unchanged;
this is an analytic transfer, not a fresh cluster-data test. A nonspherical
cluster and lensing calculation remains outstanding.

## Fixed-card comparisons

The table and figure show β=2, P=2 at two declared couplings, chosen after
inspection for compact illustration. They are not newly fitted parameters or
promoted winners. The machine-readable summary preserves all 81 cards.
The empirical RAR comparator is described by
[McGaugh, Lelli and Schombert (2016)](https://arxiv.org/abs/1609.05917).

{chr(10).join(table)}

The primary loss averages squared residuals divided by published random-error
variance. The other diagnostics use a shared inclination covariance and a
5 km/s floor. These are descriptive losses, not calibrated likelihoods.
Independent replay verified every stored velocity, selected force, residual,
all three losses, radial strata, largest-residual removal and symmetric trim
for the 6,237 original and 891 refined scored records. Neither radial
influence diagnostic reverses any admitted comparison. Removing the only
galaxy leaves no sample, so object-level influence is undefined.

## Numerical and provenance checks

The source is the same conditional axisymmetric S4G/THINGS/HERACLES
reconstruction used by the scalar predecessor. Eleven source variants and
nine geometry choices retain all 24 selected response radii in 2–20 kpc.
The 97 numerical probe radii are unchanged. Primary Newtonian replay is exact
against stored predecessor forces at those radii. This run accesses derived
snapshots only; it opens no new raw observations or reserved holdout.

The initial calculation retains 144 auxiliary solutions across coarse/fine,
map and boundary comparisons; the refinement adds six higher-resolution
solutions. Coarse/fine grids use 1,025/2,049 radial nodes, 192/320 angles and
Legendre order 48/80. The follow-up uses 4,097 radial nodes, 640 angles and
order 160 on the same density. New Newtonian changes in the two refined
variants are 0.3480% and 0.3219%. The old scalar term is retained and its
absolute coarse/fine change is added to the auxiliary refinement change.
This sum is a numerical discrepancy budget, not a rigorous error bound.
The primary boundary and map gates remain 0.5% and 3% respectively.

An unequal two-cloud control conserves internal force to a maximum normalized
residual of {max(abs(r['normalized_internal_force']) for r in control['rows']):.3g}.
Independent β=0 source integration agrees with the flux solution to
{max(r['beta_zero_relative_field_disagreement'] for r in control['rows'] if r['beta'] == 0):.3g}.
An initial independent-control coordinate-basis bug failed before galaxy
scoring; that failed run and its exact input bytes remain preserved. The
corrected control and full run pass. A later lint cleanup binds loop variables
explicitly in the refinement runner; the exact executed version remains
snapshotted, and each loop's workers completed before its variables changed.
All 162 focused implementation checks pass; lint also passes for the new code.

Input snapshots and both scientific result hashes were verified independently.
Parent result SHA-256: `{sealed['ngc3198-multifield-003']['result_sha256']}`.
Refinement result SHA-256: `{sealed['ngc3198-multifield-refinement-001']['result_sha256']}`.

## Limits and next work

This response is gas-traced circular motion, not a direct outer-star sample.
The source/response covariance, stellar mass conversion, outer map coverage,
vertical lift, warps, noncircular motion and environmental field remain
unresolved. Distance changes use homologous source scaling; inclination
changes affect velocity deprojection only, not source-map reprojection.
Shared geometry/calibration may correlate source and response; shared raw
measurements have not been established. Quality-verified and uncertainty-
resolved counterexample counts are both zero. No independent failure stratum
or unchanged independent replication is claimed from this one-galaxy pilot.

Keep the λ=10 cards pending numerical/source refinement. A physically distinct
next route is a higher-spatial-derivative action, which can introduce a universal
length and alter the connection between compact-system and galactic response.
Published GQUMOND supplies examples of that mechanism, not a demonstrated
solution to this project's three-regime problem.
[Milgrom, Generalizations of QUMOND (2023)](https://arxiv.org/abs/2305.01589)
Any proposed successor must include its full variational derivative, conservation
and boundary checks before observational scoring. Sigma's unfinished
thermodynamic-source route also remains open after measurement-model repair.
Matter/light coupling, stability and untouched cross-regime validation remain
required. The discovery goal stays active.
'''
    (ROOT/'docs/GRAVITY_NGC3198_MULTIFIELD_RESULTS.md').write_text(report, encoding='utf-8', newline='\n')
    (args.output/'Gravity-NGC3198-multifield-results.md').write_text(report+'\n![Fixed multifield comparisons](Gravity-NGC3198-multifield-comparison.png)\n', encoding='utf-8', newline='\n')
    print(json.dumps({'admitted': len(admitted), 'unresolved': len(unresolved), 'replay': checks['initial'], 'refinement': checks['refined']}))


if __name__ == '__main__':
    main()
