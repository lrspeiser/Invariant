"""Verify retained local evidence, physical units and every summary decision."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from invariant_gravity_extensions.length_screening import (
    LengthScreening,
    anomalous_flux,
    point_monopole_delta,
)


def read(path):
    return json.loads(path.read_bytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verification-output', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.verification_output.mkdir(parents=True, exist_ok=False)
    args.output.mkdir(parents=True, exist_ok=True)
    run = ROOT/'work/gravity-first-principles/length-screening-local-001'
    result, receipt = read(run/'result.json'), read(run/'receipt.json')
    digest = sha256((run/'result.json').read_bytes()).hexdigest()
    assert digest == receipt['result_sha256']
    for name, expected in result['input_hashes'].items():
        assert sha256((run/'input-snapshots'/name).read_bytes()).hexdigest() == expected
    source = 'src/invariant_gravity_extensions/length_screening.py'
    assert sha256((ROOT/source).read_bytes()).hexdigest() == result['input_hashes'][source]
    config = result['config']
    old = read(run/'input-snapshots'/config['historical_monopole_config'])
    gm, au = old['gm_sun_m3_s2'], old['au_m']
    assert read(run/'controls.json')['exit_code'] == 0
    physical_errors = []
    for row in result['rows']:
        card = row['card']
        spec = LengthScreening(card['shape'], card['epsilon'])
        assert spec.card(card['length_pc'], card['a0_m_s2']) == {k:v for k,v in card.items() if k != 'id'}
        assert read(run/('card_'+card['id']+'.json')) == row
        ell = card['length_pc']*config['parsec_m']
        a0 = card['a0_m_s2']
        assert row['dimensionless_length'] == ell/np.sqrt(gm/a0)
        # Physical SI Cartesian derivatives versus the dimensionless monopole
        # route. Periapse and apoapse of every orbit/card are checked.
        r = np.array([p['a_au']*au*(1+sign*p['e']) for p in old['planets'] for sign in [-1,1]])
        g = gm/r**2
        gradient = np.array([g, np.zeros_like(g), np.zeros_like(g)])
        H = np.zeros((3, 3, len(r)))
        H[0,0], H[1,1], H[2,2] = -2*g/r, g/r, g/r
        dH2 = np.array([-36*gm**2/r**7, np.zeros_like(r), np.zeros_like(r)])
        physical = anomalous_flux(spec, gradient, H, dH2, np.zeros_like(gradient), ell, a0)[0]/g
        normalized = point_monopole_delta(spec, g/a0, row['dimensionless_length'])
        np.testing.assert_allclose(physical, normalized, rtol=3e-10, atol=0)
        physical_errors.append(float(np.max(abs((physical-normalized)/normalized))))
        outcomes = []
        for observation in row['monopole']+row['external_quadrupole']:
            assert observation['numerical_controls_pass']
            if 'precession_mas_century' in observation:
                value = observation['precession_mas_century']
                low, high = observation['interval_mas_century']
            else:
                value = observation['Q2_s_minus2']
                low, high = observation['interval_Q2_s_minus2']
                np.testing.assert_allclose(value, observation['Q2_dimensionless']*a0**1.5/np.sqrt(gm), rtol=1e-15)
            status = 'WITHIN_DECLARED_SUMMARY_SCREEN' if low <= value <= high else 'OUTSIDE_DECLARED_SUMMARY_SCREEN'
            assert observation['status'] == status  # no unresolved or near-edge rows in this run
            outcomes.append(low <= value <= high)
        assert row['status'] == ('WITHIN_COMBINED_DECLARED_SUMMARY_SCREEN' if all(outcomes) else 'OUTSIDE_COMBINED_DECLARED_SUMMARY_SCREEN')
    q = [x for row in result['rows'] for x in row['external_quadrupole']]
    monopoles = [x for row in result['rows'] for x in row['monopole']]
    assert all(x['status'] == 'WITHIN_DECLARED_SUMMARY_SCREEN' for x in monopoles)
    summary = {'scientific_result_sha256': digest, 'counts': dict(Counter(r['status'] for r in result['rows'])),
               'all_108_quadrupole_numerical_checks_pass': all(x['numerical_controls_pass'] for x in q),
               'all_324_monopole_screens_within': True, 'physical_SI_points_checked': len(result['rows'])*12,
               'maximum_SI_vs_dimensionless_fractional_disagreement': max(physical_errors),
               'maximum_quadrupole_refinement': max(x['last_refinement_change'] for x in q),
               'maximum_action_flux_disagreement': max(x['quadrature'][-1]['absolute_agreement'] for x in q),
               'maximum_epsilon_quadrupole_change': max(x['epsilon_change'] for x in q),
               'monopole_mas_century_range': [min(x['precession_mas_century'] for x in monopoles), max(x['precession_mas_century'] for x in monopoles)],
               'maximum_sampled_fractional_perturbation': max(x['maximum_sampled_fractional_anomaly'] for x in monopoles),
               'full_solar_system_pass': False, 'galaxy_cluster_lensing_pass': False,
               'verified_input_snapshots': len(result['input_hashes']), 'discovery_claim': False}
    blob = (json.dumps(summary, indent=2, sort_keys=True)+'\n').encode()
    (args.verification_output/'result.json').write_bytes(blob)
    (args.verification_output/'script.py').write_bytes(Path(__file__).read_bytes())
    (args.verification_output/'receipt.json').write_text(json.dumps({'result_sha256':sha256(blob).hexdigest(),
        'script_sha256':sha256(Path(__file__).read_bytes()).hexdigest()})+'\n', newline='\n')
    (args.output/'Gravity-length-screening-local-summary.json').write_bytes(blob)
    (args.output/'Gravity-length-screening-local-results.json').write_bytes((run/'result.json').read_bytes())
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5), layout='constrained', sharey=True)
    for ax, a0 in zip(axes, config['a0_m_s2'], strict=True):
        ax.axhspan(-3, 9, color='#b8d9c4', alpha=.55, label='Historical summary screen')
        for shape, color in [(.5, '#267f96'), (1, '#af5369'), (2, '#9b7529')]:
            rows = [r for r in result['rows'] if r['card']['shape']==shape and r['card']['a0_m_s2']==a0]
            lengths = [r['card']['length_pc'] for r in rows]
            for background, style in [(0,'-'),(1,'--')]:
                ax.plot(lengths, [r['external_quadrupole'][background]['Q2_s_minus2']/1e-27 for r in rows],
                        style, color=color, marker='o', ms=3.5, label=f'm = {shape:g}' if background==0 else None)
        ax.set_xscale('symlog', linthresh=.001)
        ax.set_xticks([0,.001,.01,.1,1,10], ['0','.001','.01','.1','1','10'])
        ax.set(title=f'a₀ = {a0:.1e} m/s²', xlabel='Universal length ℓ (pc)')
        ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    axes[0].set_ylabel('External quadrupole Q₂ (10⁻²⁷ s⁻²)')
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle('Length-dependent action: conditional Solar System diagnostic\nSolid / dashed: two fixed Galactic backgrounds; no galaxy, cluster or full ephemeris pass claimed', fontsize=13)
    for ext in ['png','svg']:
        fig.savefig(args.output/f'Gravity-length-screening-local-comparison.{ext}', dpi=170)
    plt.close(fig)
    table = ['| Shape | a₀ (m/s²) | Q₂ at ℓ=0 | Q₂ at ℓ=0.1 pc | Q₂ at ℓ=1 pc | Sampled lengths within both screens (pc) |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for shape in config['shapes']:
        for a0 in config['a0_m_s2']:
            rows = [r for r in result['rows'] if r['card']['shape']==shape and r['card']['a0_m_s2']==a0]
            ranges = []
            for length in [0,.1,1]:
                values = [x['Q2_s_minus2']/1e-27 for r in rows if r['card']['length_pc']==length for x in r['external_quadrupole']]
                ranges.append(f'{min(values):.3f}–{max(values):.3f}')
            within = ', '.join(f"{r['card']['length_pc']:g}" for r in rows if r['status']=='WITHIN_COMBINED_DECLARED_SUMMARY_SCREEN')
            table.append(f'| {shape:g} | {a0:.1e} | '+ ' | '.join(ranges)+f' | {within} |')
    report = f'''# Length-dependent gravity: conditional local results

**34 of 54 fixed configurations lie within the declared historical local
screens; 20 lie outside.** All 54 pass numerical controls. All 324 isolated
monopole precession predictions lie within their six published intervals.
The distinguishing result comes from the external-field quadrupole: 69 of 108
background/configuration rows lie within its interval, and 39 outside.
Each complete card must satisfy both backgrounds, not select the favorable one.

This is a material change from the first-gradient candidates: higher sampled
acceleration scales can now satisfy these local diagnostics. No galaxy,
cluster, lensing, full ephemeris, relativistic or independent-validation pass
has been established. The discovery goal remains active.

## Fixed action and constants

The action is P=x+x K_m(x+h), K_m(u)=[Q_m(u)−u]/u, with the removable origin
defined by Q_m'(0)−1. Q_m is the previously registered bounded scalar action.
Here x=|∇ψ|²/a₀² and h=ℓ² Hψ:Hψ/a₀². Its full physical flux is
J_i=P_x ψ_i−ℓ² ∂_j(P_h ψ_ij). Including that derivative is essential.
The general higher-derivative framework and length-screening construction
are prior art; combining them with this bounded kernel is an explicit
effective-action ansatz, not a uniquely derived microscopic principle.
[Milgrom (2023)](https://arxiv.org/html/2305.01589v2)

Before physical predictions, the scan froze shapes 0.5, 1 and 2; a₀ values
5e−11, 1.2e−10 and 2e−10 m/s²; and ℓ=0, 0.001, 0.01, 0.1, 1 and 10 pc.
The finite regularizer is 1e−6, with 1e−7 and 1e−8 sensitivity checks.
Every card uses one length and acceleration scale across all six orbits and
both background assumptions. This grid is not a confidence interval.

The following quadrupole ranges span the two fixed backgrounds, in units of
1e−27 s⁻². The declared quadrupole screen is [−3,9] in those units. Listed
lengths are sampled values only, not continuous allowed intervals.

{chr(10).join(table)}

One pc and ten pc lie within both local screens for all nine shape/a₀ groups.
At 0.1 pc the highest-a₀ cases with shapes 1 and 2 do not satisfy both
backgrounds. Smaller nonzero lengths can initially increase the quadrupole;
screening is not a simple monotone multiplier on the scalar result.

## What was calculated and verified

The external solution uses ψ=−1/r−η_N z in units GM=a₀=1, and ℓ is converted
from pc to units of sqrt(GM/a₀). The asymptotic mapping
η_N ν(η_N)=g_external/a₀ follows the action's exact h=0 scalar limit.
The physical backgrounds, 1.9e−10 and 2.4e−10 m/s², are inherited published
scenarios, not a reconstructed Galactic field or a measured uncertainty range.

Two infinite-domain quadrupole representations are evaluated independently:
one integrates the full flux against the harmonic Green kernel, while the
other integrates the higher-derivative action term twice by parts. The latter
requires the full three-dimensional Hessian. Their maximum dimensionless
disagreement is {summary['maximum_action_flux_disagreement']:.3g}; maximum
128→256-node change is {summary['maximum_quadrupole_refinement']:.3g}; maximum
regularizer change is {summary['maximum_epsilon_quadrupole_change']:.3g}.
No automatic 512/1024-node follow-up was needed. These changes are numerical
diagnostics, not certified error bounds or statistical uncertainty.

Seventeen new implementation tests cover high-precision kernel derivatives,
origin regularity, exact scalar recovery, independent scalar quadrupoles,
Cartesian versus polar flux, the bounded point-source asymptote, periodic
action variation and internal momentum conservation. SI Cartesian and
dimensionless radial calculations agree at all 648 periapse/apoapse probes;
maximum fractional disagreement is {max(physical_errors):.3g}.
Input snapshots, all 54 card hashes and every stored classification were
independently verified. A post-run lint cleanup binds sequential loop values
explicitly; the exact executed runner remains in the frozen input snapshots.

The first-order isolated precession range is
[{summary['monopole_mas_century_range'][0]:.6g}, {summary['monopole_mas_century_range'][1]:.6g}]
mas/century. The maximum sampled fractional perturbation is
{summary['maximum_sampled_fractional_perturbation']:.3g}. The external quadrupole
and isolated monopole are separate leading diagnostics, not a joint planetary
orbit and light-propagation fit.

## Evidence limits and next transfer

The quadrupole comparison uses the previously exposed Cassini result
Q₂=(3±3)e−27 s⁻² and the predeclared two-standard-deviation summary interval.
It is not a fresh analysis of Cassini observations or necessarily the latest
constraint. [Hees et al. (2014)](https://arxiv.org/abs/1402.6950)
The monopole uses INPOP10a supplementary-precession sensitivity intervals;
these are postfit-residual criteria, not Gaussian errors or a candidate-specific
likelihood. [Fienga et al. (2011)](https://arxiv.org/abs/1108.5546)
No new raw observations or reserved outcomes were opened.

The next test must use these same constants on observed source distributions.
Existing scalar galaxy and cluster scores cannot simply be attached to the
new cards: the new force depends on density derivatives. In particular, the
cluster adapter's piecewise linear stellar enclosed mass has slope jumps,
which imply discontinuous stellar density. Gas log-linear slopes and the
galaxy potential's cubic-Hermite derivative representation also need a
regularity audit. Source interpolation and its uncertainty must be declared
and tested before scoring the higher-derivative response. Lensing requires a
derived matter/photon coupling, which this static action does not supply.

Reproduce with `scripts/run_gravity_length_screening_local.py --output
<unused-directory>`. Evidence is in `work/gravity-first-principles/length-screening-local-001/`.
Result SHA-256: `{digest}`.
'''
    (ROOT/'docs/GRAVITY_LENGTH_SCREENING_LOCAL_RESULTS.md').write_text(report, encoding='utf-8', newline='\n')
    (args.output/'Gravity-length-screening-local-results.md').write_text(report+'\n![Local comparison](Gravity-length-screening-local-comparison.png)\n', encoding='utf-8', newline='\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
