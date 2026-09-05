"""Export the off-plane reference result without creating observational scores."""
from __future__ import annotations

import argparse
import json
from hashlib import sha256
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    names = ['hankel-offplane-001', 'hankel-offplane-verification-001']
    values, hashes = [], {}
    for name in names:
        directory = ROOT/'work/gravity-first-principles'/name
        digest = sha256((directory/'result.json').read_bytes()).hexdigest()
        if digest != json.loads((directory/'receipt.json').read_bytes())['result_sha256']:
            raise ValueError(f'{name} changed')
        hashes[name] = digest
        values.append(json.loads((directory/'result.json').read_bytes()))
    result, verification = values
    summary = {'result_hashes': hashes, 'source_cases': [], 'verification': {k: verification[k] for k in
        ['verified_input_snapshots', 'all_fine_stencils_within_target', 'status']},
        'new_observational_scores': 0, 'new_physical_rejections': 0, 'validated_universal_gravity_laws': 0,
        'scope': result['config']['scope']}
    table = []
    for row in result['summary']:
        compact = {k: v for k, v in row.items() if k != 'comparisons'}
        compact['comparisons'] = [{k: v for k, v in x.items() if k != 'errors_by_probe'} for x in row['comparisons']]
        summary['source_cases'].append(compact)
        for comparison in compact['comparisons']:
            e = comparison['maximum_errors']
            table.append(f"| {row['variant']['id']} | {comparison['case']} | {comparison['role']} | {e['maximum_force_scaled_change']:.6g} | {e['maximum_hessian_scaled_change']:.6g} | {e['maximum_third_tensor_scaled_change']:.6g} |")
    fine = [row for row in verification['rows'] if row['step_kpc'] == min(verification['step_sizes_kpc'])]
    maximum_stencil = max(max(row['maximum_errors'].values()) for row in fine)
    summary['maximum_fine_stencil_error'] = maximum_stencil
    summary['refinement_field_evaluations'] = len(result['records'])*len(result['config']['radii_kpc'])*len(result['config']['heights_kpc'])
    (args.output/'Gravity-offplane-reference-summary.json').write_text(json.dumps(summary, indent=2, sort_keys=True)+'\n', encoding='utf8')
    R, z = np.array(result['config']['radii_kpc']), np.array(result['config']['heights_kpc'])
    X, Y = np.meshgrid(R, z, indexing='ij')
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), layout='constrained', sharex=True, sharey=True)
    normalization = LogNorm(vmin=1e-12, vmax=.01)
    for col, row in enumerate(result['summary']):
        variant = row['variant']['id']
        third = np.maximum.reduce([np.array(x['errors_by_probe']['third_tensor_scaled_change'])
            for x in row['comparisons'] if x['role'] == 'refinement'])
        physical = next(x['source_errors']['physical_density_gradient_scaled_error'] for x in result['records']
            if x['variant']['id'] == variant and x['case']['id'] == 'reference')
        for ax, data in zip(axes[:, col], [third, np.array(physical)], strict=True):
            plot = ax.scatter(X.ravel(), Y.ravel(), c=np.maximum(data.ravel(), 1e-12), s=36, marker='s',
                              cmap='magma', norm=normalization, edgecolors='none')
            ax.set_yscale('symlog', linthresh=.03)
            ax.set_yticks([-32, -8, -1, -.1, 0, .1, 1, 8, 32], labels=['−32', '−8', '−1', '−0.1', '0', '0.1', '1', '8', '32'])
            ax.set_xscale('symlog', linthresh=1.)
            ax.set_xticks([0, .5, 1, 4, 12, 35], labels=['0', '0.5', '1', '4', '12', '35'])
            ax.set_xlim(-.08, 40)
            ax.set_ylim(-45, 45)
            ax.grid(alpha=.12)
        axes[0, col].set_title('Nominal source' if variant == 'primary' else 'Half-height source')
        axes[1, col].set_xlabel('Cylindrical radius (kpc, symmetric log)')
    axes[0, 0].set_ylabel('Height (kpc, symmetric log)\nThird-derivative refinement error')
    axes[1, 0].set_ylabel('Height (kpc, symmetric log)\nPhysical density-gradient error')
    fig.colorbar(plot, ax=axes, label='Scaled error (registered target 0.01)', shrink=.85)
    fig.suptitle('Off-plane Newtonian reference: every registered location retained\nFive refinement axes; fixed source assumptions; no new gravity or velocity fits', fontsize=12)
    fig.savefig(args.output/'Gravity-offplane-reference-comparison.png', dpi=170)
    fig.savefig(args.output/'Gravity-offplane-reference-comparison.svg')
    plt.close(fig)
    text = f'''# Off-plane Newtonian reference for the gravity search

Both fixed galaxy-source variants meet every registered off-plane numerical
target. The implementation now evaluates the potential, both force components,
the full axisymmetric Hessian, all six independent nonzero Cartesian third
derivatives, and the gradients of its trace and tensor norm. This removes a
specific derivative-implementation obstacle on the sampled domain. It does not
establish a successful gravity law or validate the full isolated nonlinear solve.

## Test scope and result

The 273 registered locations combine 13 cylindrical radii (0–35 kpc) with 21
heights (−32 to +32 kpc). They include the axis, midplane and reflection pairs.
The sources are the same nominal and half-height NGC3198 assumptions as the prior
audit. Fourteen field configurations retain {summary['refinement_field_evaluations']:,}
evaluations. The radial transforms are exact byte-checked predecessor artifacts;
no velocity, new gravity parameter, raw source spectrum or reserved observation
was opened. These are numerical checks of a conditional source.

The reference uses 128 nodes per radial source interval, 32 nodes per 0.5/kpc
wavenumber interval, a cutoff of 400/kpc, and 2,400 vertical spline intervals
over 0–24 scale heights. Five separate refinements change radial quadrature,
wavenumber quadrature, cutoff, vertical interpolation and the height at which an
infinite exponential tail is attached. Cutoff 100/kpc is a retained stress case.

| Source | Changed setting | Role | Force error | Hessian error | Third tensor error |
|---|---|---|---:|---:|---:|
{chr(10).join(table)}

The force, Hessian and third-tensor targets are 1e-4, 0.002 and 0.01. Every
mandatory refinement passes for both sources. The largest third-tensor change is
0.00077314, driven by the cutoff refinement; the 100/kpc stress case reaches
0.00232584. The largest reference density-identity discrepancy is 5.2470e-7.
The largest density-gradient discrepancy is 0.00077610, at R=0.25 kpc,
z=−0.025 kpc for the nominal source. The corresponding half-height maximum is
0.00053288 at R=0.25, z=0. These are errors in representing the assumed source,
not residuals against an astronomical measurement.

Force errors use the reference vector norm with a tiny fixed characteristic
floor at its zero. Tensor errors use the full Cartesian Frobenius norms. Third
derivatives use max(norm(T), norm(H)/(spherical radius + minimum source height)).
Density identity errors use max(abs(4*pi*G*rho),norm(H)); gradient errors use
max(norm(4*pi*G*grad rho), norm(H)/(spherical radius + minimum height)). Thus
points with vanishing physical density far from the plane remain in the checks.
The precise normalizations and all per-point values are in the frozen record.

Reflection errors stay below 1.172e-15. At the midplane, the new calculation
agrees with the preceding independent reference within 3.281e-11 on the stated
force, Hessian and radial-derivative scales.

## One source and one potential for all derivatives

The disk Green representation follows
[Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
The implementation extends the previous midplane specialization by calculating
the height-dependent Green convolution and its derivatives from one explicit
vertical source. A cubic spline approximates the normalized sech-squared lift
on its positive half; an exponential matched in value and first derivative
continues to infinity. Exact reflection supplies its other half. The complete
source is normalized to unit mass before use.

Directly subtracting k-squared times the potential kernel from a large local
source term can lose precision in high derivatives. Instead, exact exponential
moments integrate each polynomial source derivative. The third derivative includes
the weak contribution from the small second-derivative jump at the spline/tail
join. The code therefore differentiates one potential throughout, including the
tail. It does not replace a failed numerical trace by the physical density.

The fine vertical source differs from the physical lift by at most 4.168e-10
of peak density and 1.251e-7 of peak density per scale height for its first
derivative on the retained check grid. The independently supplied physical density
and gradient are still used to test the final three-dimensional Poisson identity.

## Verification

Six new synthetic tests cover exact exponential moments, equal decay rates,
adaptive direct Green integrals, the splice contribution, reflection, normalized
mass, the exact sech-squared midplane limit, full Cartesian Gaussian tensors,
source partition, distance homology and invalid inputs. The Gaussian tensor
control is derived from enclosed mass rather than the cylindrical formulas.
All 215 focused tests and the updated workflow's lint command pass locally.

The separate verifier loads the executed package from
{verification['verified_input_snapshots']} byte-checked snapshots under an isolated
module name. It evaluates fourth-order finite differences of the Hessian, its
trace and its squared norm at all 273 locations for both sources and two step
sizes, 0.001 and 0.0005 kpc. Through the axis it uses a signed Cartesian extension
with the correct even/odd tensor parity. No axis points are dropped.
The finest differences agree within {maximum_stencil:.7g} on the registered
scales. This verifies derivative consistency; it does not independently establish
the true astrophysical mass distribution or an unsampled global error bound.

## Next work toward the universal law

Use the validated integral as a reference while establishing an accurate source
provider over the full isolated domain, including the outer boundary and any
interpolation between points. Then propagate source errors through the full
action flux and its separate Poisson solve. A local action flux is not itself
the physical modified disk acceleration.

Only after that validation should a wider length grid be registered and carried
through Solar System, cluster and galaxy tests with the same constants.
The earlier 54-card comparisons remain conditional; no source or theory is
discarded on the basis of this audit. Source uncertainty, outer-star observables,
photon coupling, dynamics and stability, and untouched confirmation remain open.
The full discovery goal remains active. Added observational scores, physical
rejections and validated universal laws are all zero.

## Evidence

- Off-plane result: `{hashes[names[0]]}`.
- Derivative verification: `{hashes[names[1]]}`.

Full evidence is in `work/gravity-first-principles/{names[0]}/` and
`work/gravity-first-principles/{names[1]}/`. The JSON summary and exportable PNG/SVG
figure accompany this report. The figure colors errors below 1e-12 at that floor;
the underlying numerical values are retained unchanged.
'''
    (args.output/'Gravity-offplane-reference-results.md').write_text(text, encoding='utf8')
    print(json.dumps({'hashes': hashes, 'maximum_fine_stencil_error': maximum_stencil}))


if __name__ == '__main__':
    main()
