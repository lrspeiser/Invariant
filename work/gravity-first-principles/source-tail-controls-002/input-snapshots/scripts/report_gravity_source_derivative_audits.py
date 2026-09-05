"""Create a reproducible source-resolution report and exportable figure."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['angular-source-resolution-002', 'hankel-midplane-001', 'source-derivative-verification-001']
    values, hashes = [], {}
    for name in names:
        p = ROOT/'work/gravity-first-principles'/name
        digest = sha256((p/'result.json').read_bytes()).hexdigest()
        if digest != json.loads((p/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError(f'{name} hash mismatch')
        hashes[name] = digest
        values.append(json.loads((p/'result.json').read_bytes()))
    angular, hankel, verification = values
    radii = np.array(hankel['config']['midplane_radii_kpc'])
    summary = {'result_hashes': hashes, 'angular_source_summary': angular['summary'], 'midplane_summary': hankel['summary'],
        'multipole_summary': [], 'verification': {k: verification[k] for k in ['verified_input_snapshots',
        'projection_coefficient_integrals', 'maximum_projection_disagreement', 'maximum_transform_disagreement']},
        'new_observational_scores': 0, 'new_physical_rejections': 0, 'quality_verified_cross_regime_candidates': 0,
        'scope': 'Conditional source and Newtonian midplane numerical diagnostics; no full two-dimensional derivative or physical-law validation.'}
    for row in hankel['multipole_comparisons']:
        summary['multipole_summary'].append({'variant': row['variant']['id'], 'grid': row['grid']['id'],
            'maximum_over_registered_probes': {key: max(row[key]) for key in row if key.endswith(('difference', 'error'))},
            'maximum_over_registered_probes_to_20kpc': {key: float(np.max(np.array(row[key])[radii <= 20]))
                for key in row if key.endswith(('difference', 'error'))}})
    (args.output/'Gravity-source-derivative-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf8')
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.2), layout='constrained')
    colors = {'primary': '#1464a0', 'height_half': '#b44423'}
    labels = {'primary': 'Nominal disk', 'height_half': 'Half-height disk'}
    for variant, color in colors.items():
        rows = [r for r in angular['summary'] if r['variant']['id'] == variant]
        orders = [r['maximum_order'] for r in rows]
        for ax, key in [(axes[0, 0], 'density_L1_fraction_error'), (axes[1, 0], 'gradient_L1_fraction_error')]:
            ax.loglog(orders, [r['maximum_errors'][key] for r in rows], marker='o', color=color, label=labels[variant])
        row = next(x for x in hankel['multipole_comparisons'] if x['variant']['id'] == variant and x['grid']['id'] == 'inherited_fine')
        for ax, key in [(axes[0, 1], 'force_fraction_difference'), (axes[1, 1], 'hessian_norm_scaled_difference')]:
            ax.semilogy(radii, row[key], marker='o', color=color, label=labels[variant])
    for ax, threshold, title in [(axes[0, 0], .01, 'Density reconstruction'), (axes[1, 0], .05, 'Source-gradient reconstruction')]:
        ax.axhline(threshold, color='.35', ls='--', label=f'Registered target: {threshold:g}')
        ax.set(title=title, xlabel='Maximum Legendre order', ylabel='Worst shell fractional L1 error')
        ax.legend(fontsize=8, loc='lower left')
        ax.grid(alpha=.18)
    axes[0, 1].set(title='Order-80 radial force vs Hankel reference', ylabel='Absolute fractional force difference')
    axes[1, 1].set(title='Order-80 Hessian vs Hankel reference', ylabel='Difference / reference tensor norm')
    for ax in axes[:, 1]:
        ax.set_xlabel('Midplane radius (kpc)')
        ax.grid(alpha=.18)
        ax.legend(fontsize=8)
    fig.suptitle('Accurate radial force can coexist with inaccurate source derivatives\nFixed NGC3198 source assumptions; numerical diagnostics, no new velocity fits', fontsize=13)
    fig.savefig(args.output/'Gravity-source-derivative-comparison.png', dpi=170)
    fig.savefig(args.output/'Gravity-source-derivative-comparison.svg')
    plt.close(fig)
    angular_table = '\n'.join(f"| {labels[r['variant']['id']]} | {r['maximum_order']} | {r['maximum_errors']['density_L1_fraction_error']:.6g} | {r['maximum_errors']['gradient_L1_fraction_error']:.6g} | {r['maximum_errors']['negative_density_fraction']:.6g} | {'yes' if r['within_registered_source_targets'] else 'no'} |"
        for r in angular['summary'] if r['maximum_order'] in [80, 640, 1280, 2560])
    tensor_table = '\n'.join(f"| {labels[r['variant']]} | {r['grid']} | {100*r['maximum_over_registered_probes_to_20kpc']['force_fraction_difference']:.4f}% | {100*r['maximum_over_registered_probes_to_20kpc']['hessian_norm_scaled_difference']:.2f}% | {100*r['maximum_over_registered_probes_to_20kpc']['hessian_invariant_gradient_scaled_difference']:.2f}% |"
        for r in summary['multipole_summary'])
    max_refinement = {key: max(ref['differences'][key] for row in hankel['summary'] for ref in row['refinements'])
                      for key in hankel['summary'][0]['refinements'][0]['differences']}
    text = f'''# Gravity source and derivative audit

The inherited order-80 disk representation is unsuitable as a verified derivative
source for extending the length-action scan. An independent Newtonian midplane
integral passes the registered refinement and density-consistency targets. It
finds small radial-force errors alongside much larger Hessian errors. No gravity
family is rejected and no new observational score is added by this audit.

## Angular source resolution

The two fixed sources are the same positive regular-core surface profiles and
sech-squared vertical lifts used previously: nominal and half-height. We project
only even Legendre modes, as required by their exact assumed reflection symmetry.
Projection uses 2,048/4,096 Gauss nodes in the positive hemisphere; separate
evaluation uses 4,096/8,192. The twelve registered spherical radii cover 0.25–35 kpc.
Reported errors are the maximum across these shells, not a volume-integrated
negative-mass fraction and not an observational residual.

| Conditional source | Order | Density L1 fraction | Gradient L1 fraction | Negative density fraction | Within all targets |
|---|---:|---:|---:|---:|---|
{angular_table}

The first sampled order meeting every target is 1,280 for the nominal disk and
2,560 for the half-height disk. Density and gradient L1 targets are 1% and 5%;
negative projected density must integrate to less than 0.5% of the physical
hemispheric density, and quadrature changes must remain below 0.001 in these
dimensionless metrics. These finite targets neither prove pointwise positivity
nor validate a full gravitational field. At order 80, the worst density shell is
35 kpc for both sources. The small integration changes identify angular
truncation as the dominant measured error.

Initial run 001 failed a synthetic gradient tolerance: 2.35e-12 versus 1e-12.
Small floating-point projections of a constant into analytically absent high
modes are amplified by differentiation at the smallest test radius. The retained
correction uses 1e-11 for that synthetic projection test and separately checks
the exact supplied polynomial coefficients to 1e-14. No source-audit target was
loosened, and run 001 remains intact.

## Independent Newtonian midplane reference

The reference integrates the source in cylindrical coordinates. The general
separable-source Green representation follows
[Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
It does not use a spherical multipole expansion. Potential, force, Hessian and
third derivatives are differentiated from the same finite-wavenumber integral.
Its trace is compared with physical density; physical density is never substituted
into a truncated Hessian to force that comparison to pass.

There are 24 retained jets: two heights, three cutoffs (100, 200, 400 per kpc),
two radial source rules (64, 128 nodes per interval), and two wavenumber rules
(16, 32 nodes per 0.5/kpc interval). All measured surface-profile knots, the core
join and the outer taper boundaries are integration edges. Radial transforms
are shared only after checking that the two sources differ solely in height.
All component fields are summed before constructing nonlinear tensor invariants.

Both sources pass all three separate refinements. Maximum changes across them:

- Radial force: {max_refinement['maximum_force_fraction_change']:.7g} of the reference force.
- Hessian: {max_refinement['maximum_hessian_norm_scaled_change']:.7g} of its tensor norm.
- Radial Hessian derivative: {max_refinement['maximum_third_derivative_scaled_change']:.7g} of the registered scale.
- Physical density consistency: {max(r['maximum_physical_density_fraction_error'] for r in hankel['summary']):.7g} fractional error.
- Physical density-gradient consistency: {max(r['maximum_physical_density_gradient_scaled_error'] for r in hankel['summary']):.7g} of the registered scale.

Third-derivative comparisons use max(norm(dH/dR), norm(H)/R); density-gradient
consistency uses max(abs(4*pi*G*density_R), norm(H)/R). This retains zero-crossing
probes. These are engineering accuracy criteria for the fixed assumed source,
not uncertainty bounds on the source inferred from observations.

## What the earlier force gate missed

Maximum discrepancies over registered midplane probes at or within 20 kpc:

| Source | Earlier grid | Radial force | Hessian norm | Gradient of H:H |
|---|---|---:|---:|---:|
{tensor_table}

The last column is scaled by max(abs(gradient(H:H)), (H:H)/R). Across all twelve
probes, the fine nominal/thinner Hessian discrepancies reach 63.0%/80.4%; fine
force discrepancies reach 2.18%/3.10% at 35 kpc. A nearly converged radial force
therefore does not establish the derivative accuracy required by this action.
The earlier conditional galaxy scores remain recorded, but cannot promote a
physical card. The result strengthens the existing numerical limitation; it is
not evidence against all length-dependent laws.

## Verification and next step

Eight new synthetic tests cover finite-polynomial projection, unclipped negative
reconstruction, adaptive sech-squared integrals, exact Gaussian Hankel transforms,
source partition, a three-dimensional spherical Gaussian potential and its
derivatives, retained finite-cutoff density error, and invalid inputs. All 209
focused tests and the updated workflow's lint command pass locally. The separate
verifier checked {verification['verified_input_snapshots']} exact input snapshots,
{verification['projection_coefficient_integrals']} source coefficient integrals,
and 15 radial-transform spot integrals with adaptive quadrature. Maximum normalized
disagreements are {verification['maximum_projection_disagreement']:.6g} and
{verification['maximum_transform_disagreement']:.6g}, respectively. Normalizations
and individual values are retained; these checks do not certify unsampled space.

Next, extend the independent integral to off-plane Newtonian derivatives and
validate their source identity and convergence on the full domain feeding the
nonlinear field equation. Then validate that equation's separate Poisson solve.
A midplane action flux alone is not the physical modified disk acceleration.
Only after those checks should a new, globally fixed length grid be registered
and evaluated again in the local, cluster and galaxy regimes.

Source uncertainties, nonaxisymmetric structure, external environment, direct
outer-star observables, photon dynamics, stability and untouched confirmation
remain open. New response scores, physical rejections and quality-verified
cross-regime candidates added here are all zero. The discovery goal stays active.

## Evidence

- Angular result: `{hashes[names[0]]}`.
- Midplane result: `{hashes[names[1]]}`.
- Adaptive verification: `{hashes[names[2]]}`.

Run directories are under `work/gravity-first-principles/`. Each has its executed
configuration, inputs or verifier snapshot, raw diagnostics and receipt. The
standalone JSON summary and PNG/SVG figure accompany this report.
'''
    (args.output/'Gravity-source-derivative-results.md').write_text(text, encoding='utf8')
    print(json.dumps({'hashes': hashes, 'maximum_refinement_changes': max_refinement}))


if __name__ == '__main__':
    main()
