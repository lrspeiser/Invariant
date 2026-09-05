"""Present retained development evidence without selecting a preferred nuisance."""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

repo = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
run = repo / 'work/gravity-first-principles/ngc3198-scalar-001'
output = args.output
output.mkdir(parents=True, exist_ok=True)
result = json.loads((run / 'result.json').read_bytes())
if result['status'] != 'QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED':
    raise RuntimeError('No scored result; do not create an empirical curve plot')
rows = result['primary_rows']
r = np.array([row['nominal_radius_kpc'] for row in rows])
observed = np.array([row['geometry_corrected_velocity_kms'] for row in rows])
errors = np.array([row['geometry_corrected_error_kms'] for row in rows])
summary = result['summary']
fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3), sharey=True, layout='constrained')
scales = result['config']['candidates']['a0_m_s2']
for ax, a0 in zip(axes, scales, strict=True):
    ax.errorbar(r, observed, yerr=errors, fmt='o', color='#222c38', ms=3.5, lw=.9, label='Observed gas-traced circular speed', zorder=10)
    for m, color in [(0.5, '#9f437c'), (1, '#1b8290'), (2, '#c38427')]:
        key = f'SAT_m{m:g}_a0_{a0:.1e}'
        primary = np.array(summary[key]['nominal']['predicted_velocity_kms'])
        variants = np.array([s['candidate_results'][key]['predicted_velocity_kms'] for s in result['scenarios']
                             if s['distance_offset_mpc'] == s['inclination_offset_deg'] == 0])
        ax.fill_between(r, variants.min(axis=0), variants.max(axis=0), color=color, alpha=.10)
        ax.plot(r, primary, color=color, label=f'Saturated scalar, m = {m:g}')
    for key, label, style in [('NEWTON_BARYONS', 'Newtonian baryons', ':'), ('RAR_2016_ALGEBRAIC', 'RAR comparator', '--')]:
        ax.plot(r, summary[key]['nominal']['predicted_velocity_kms'], style, color='#636d7e', lw=1.2, label=label)
    ax.set(title=f'a₀ = {a0:.1e} m/s²', xlabel='Radius at the source distance (kpc)')
    ax.grid(alpha=.18)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_ylabel('Circular speed (km/s)')
axes[1].legend(fontsize=8, loc='lower right', frameon=False)
fig.suptitle('NGC 3198 • same fixed gravity parameters as the cluster and Solar System tests\nBands: declared source variants only; development evidence with unresolved joint uncertainties', fontsize=13)
for extension in ['png', 'svg']:
    fig.savefig(output / f'Gravity-NGC3198-scalar-comparison.{extension}', dpi=170)

gates = json.loads((run / 'numerical_gates.json').read_bytes())
maxima = {kind: max(row['maximum_absolute'] for name, comparison in gates['comparisons'].items() if name.startswith(kind+'/')
                    for row in comparison.values()) for kind in ['resolution', 'boundary', 'map']}
receipt = json.loads((run / 'receipt.json').read_bytes())
table = ['| Shape m | a₀ (m/s²) | Median predicted / observed speed | Random-error loss | Matched scenarios worse than RAR |',
         '| --- | --- | ---: | ---: | ---: |']
for a0 in scales:
    for m in [.5, 1, 2]:
        key = f'SAT_m{m:g}_a0_{a0:.1e}'
        row = summary[key]
        nominal = row['nominal']
        table.append(f"| {m:g} | {a0:.1e} | {nominal['median_predicted_observed_ratio']:.4f} | {nominal['random_error_loss']:.2f} | {row['scenarios_worse_than_RAR']}/{row['scenario_count']} |")
for key in ['NEWTON_BARYONS', 'RAR_2016_ALGEBRAIC']:
    row = summary[key]['nominal']
    table.append(f"| {key} | — | {row['median_predicted_observed_ratio']:.4f} | {row['random_error_loss']:.2f} | — |")
report = f'''# NGC 3198 scalar gravity development test

The same nine scalar candidates used for the cluster-pressure and Solar System
audits have now been compared with 24 gas-traced circular-velocity measurements
between 2 and 20 kpc in NGC 3198. This is a single previously exposed development
galaxy. There is no fitted galaxy-specific gravity constant, no untouched
confirmation claim and no new-law promotion.

All three candidates at a₀ = 5e-11 m/s²—the sampled scale that remained inside
the earlier historical Cassini-summary screen—lose to the RAR comparator in
all 99 matched scenarios and all three error diagnostics. Their nominal median
speed ratios are 0.663–0.695. This adds galaxy evidence to the previously
recorded cluster-pressure deficit at that same scale; neither result resolves
the missing source/systematic covariance.

The m = 2, a₀ = 1.2e-10 candidate has a lower nominal random-error loss than RAR
(10.34 versus 17.10), but greater velocity RMS error (8.23 versus 7.58 km/s)
and greater loss with correlated inclination uncertainty (7.40 versus 3.98).
It wins 66/99 random-error scenarios and 0/99 inclination-covariance scenarios.
Its apparent preference depends on how uncertainty is represented. The higher
sampled acceleration scales exceeded the earlier Cassini-summary screen.
No tested parameter set has demonstrated success across all three regimes.

## Fixed predictions and comparisons

Every candidate uses the same observed-map source within each source scenario.
The scalar QUMOND field is solved jointly from the full conditional disk source,
not by algebraically modifying separately fitted gas and stellar velocities.
The three shapes and three acceleration scales are unchanged from the cluster
and local experiments. The RAR curve is an empirical acceleration comparator,
not a separately derived field equation. [RAR source](https://arxiv.org/abs/1609.05917)

{chr(10).join(table)}

Loss is the mean squared residual divided by the published random velocity error.
It is a descriptive diagnostic, not a calibrated likelihood or discovery
significance. All curves, all 99 matched source/geometry scenarios, and every
selected radial residual are retained. The source bands shown in the figure
span eleven declared mass/thickness/aperture choices at nominal geometry;
they are stress envelopes, not confidence intervals. No nuisance winner is
selected.

## Source and numerical checks

The seven S4G/THINGS/HERACLES maps, conversions, vertical lift, masking, partial
coverage and source-only refinement failures are documented in the companion
galaxy source report. The primary source uses photometric inclination 71.923
degrees and distance 13.987 Mpc. The SPARC catalog supplies 73 ± 3 degrees and
13.8 ± 1.4 Mpc. Radii scale with distance; published speeds and errors are
transformed by sin(73 degrees)/sin(71.923 degrees).
[SPARC data and metadata](https://astroweb.cwru.edu/SPARC/)

The largest force changes across the frozen 97 numerical probe radii are
{100*maxima['resolution']:.4f}% for coarse/fine fields across all eleven source
variants, {100*maxima['boundary']:.4f}% for the primary inner/outer boundary
extension, and {100*maxima['map']:.4f}% for primary 1024/2048 map refinement.
All declared numerical gates passed before individual velocity conversion and
scoring. The campaign retains 216 scalar field solves and Newton/RAR controls
on 24 source/grid configurations. Four additional implementation tests cover
exact-grid caching, geometry/covariance, influence sensitivity and a manufactured
end-to-end curve; together with the prior 135 controls, 139 tests pass.

The 43 published radii were inspected for geometric selection. Nineteen rows
outside the fixed 2–20 kpc range were not scored or numerically converted to
velocities by this runner. The primary includes all 24 eligible rows; no
residual-based exclusion changes that set. This does not validate the entire
published curve or its farthest outer points.

## Measurement and influence limitations

The published response combines gas kinematics, with original references
[Daigle et al. (2006)](https://arxiv.org/abs/astro-ph/0601376) and
[Begeman et al. (1991)](https://doi.org/10.1093/mnras/249.3.523), plus Begeman's
1987 thesis. It is not a direct outer-star velocity sample. The SPARC quality
flag is 1, but that catalog label does not establish the joint source/response
quality needed for a new gravity claim. The published random errors omit
inclination systematics.

Distance stress tests rescale all source lengths homologously, including the
assumed vertical heights, and hence scale speed by sqrt(D/D_nominal). Inclination
stress tests change kinematic deprojection only. Full source reprojection,
warps, noncircular streaming, alternate HI weighting, missing outer CO/stellar
material, exact beam response, external fields and joint covariance remain
unresolved. Inclination rank-one covariance and an extra 5 km/s error floor
are retained as separate diagnostics. The config's statement that source maps
and velocities share measurements is a conservative unverified assumption;
common raw observations have not been established for these distinct catalog
products. Shared geometry and calibration may still correlate them.

The result records raw worse-than-RAR counts for this one galaxy; quality-verified
and uncertainty-resolved counterexample counts are zero because the audit is
incomplete. Removing the only galaxy leaves no sample, so that object-level
influence diagnostic is explicitly undefined. The companion radial diagnostics
remove the single largest comparative residual and trim one point from each
tail of the 24 paired loss differences, without altering the primary result.
Inner/outer strata are frozen at 12 kpc. They are not independent galaxy
replications.

Eight of nine scalar candidates are raw nominal worse-than-RAR cases under the
primary diagnostic; all nine are worse under the inclination-covariance
diagnostic. None is a quality-verified or uncertainty-resolved counterexample.
No nominal comparison changes sign under either radial influence diagnostic.
Across nuisance scenarios, the m = 2, a₀ = 1.2e-10 candidate changes sign in
7/99 single-radial-removal cases and 2/99 symmetric-trim cases. Those sensitivities
are retained alongside its primary results, not used to discard observations.

## Access accounting and reproduction

The full SPARC response container was already exposed in historical work and
was parsed again here for schema, provenance and radii. An obsolete fixed-byte
header caused the catalog's aggregate flat speed and uncertainty to appear
during metadata extraction; this is recorded in `map-response-metadata-001`.
The response contract was frozen after that exposure and before implementation
or individual velocity scoring. This is documented development evidence.
No new reserved cluster or lensing payload was opened. Other galaxies were not
scored; the already exposed full SPARC container and the public summary catalog
were read as containers, as disclosed above.

Run `scripts/run_gravity_ngc3198_development.py` with the versioned config and
an unused output directory. Exact code/config/source/response input bytes are
snapshotted in the run. The original live checkout is unchanged. Result SHA-256:
`{receipt['result_sha256']}`.

The result remains `QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED`. One galaxy
cannot establish a universal formula. The cluster/local comparison, full
population and far-outer-radius tests, matter/light coupling, external boundary,
stability and independent validation remain required.
'''
(repo / 'docs/GRAVITY_NGC3198_SCALAR_2026_09_05.md').write_text(report, encoding='utf8', newline='\n')
(output / 'Gravity-NGC3198-scalar-results.md').write_text(report+'\n![Fixed scalar predictions](Gravity-NGC3198-scalar-comparison.png)\n', encoding='utf8', newline='\n')
(output / 'Gravity-NGC3198-scalar-summary.json').write_text(json.dumps({'result_sha256': receipt['result_sha256'], 'numerical_maxima': maxima, 'summary': summary}, indent=2)+'\n', encoding='utf8', newline='\n')
(output / 'Gravity-NGC3198-scalar-results.json').write_bytes((run / 'result.json').read_bytes())
