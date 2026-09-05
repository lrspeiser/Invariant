# NGC 3198 scalar gravity development test

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

| Shape m | a₀ (m/s²) | Median predicted / observed speed | Random-error loss | Matched scenarios worse than RAR |
| --- | --- | ---: | ---: | ---: |
| 0.5 | 5.0e-11 | 0.6628 | 540.76 | 99/99 |
| 1 | 5.0e-11 | 0.6890 | 404.54 | 99/99 |
| 2 | 5.0e-11 | 0.6947 | 333.22 | 99/99 |
| 0.5 | 1.2e-10 | 0.7702 | 200.08 | 99/99 |
| 1 | 1.2e-10 | 0.8802 | 47.20 | 98/99 |
| 2 | 1.2e-10 | 0.9485 | 10.34 | 33/99 |
| 0.5 | 2.0e-10 | 0.8751 | 54.08 | 99/99 |
| 1 | 2.0e-10 | 1.0038 | 24.38 | 54/99 |
| 2 | 2.0e-10 | 1.0500 | 35.01 | 57/99 |
| NEWTON_BARYONS | — | 0.6097 | 836.84 | — |
| RAR_2016_ALGEBRAIC | — | 0.9405 | 17.10 | — |

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
0.9989% for coarse/fine fields across all eleven source
variants, 0.0017% for the primary inner/outer boundary
extension, and 1.1066% for primary 1024/2048 map refinement.
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
`2892912141e39808ae1b30917d718fa9cc5d0b1970b650ab2a7419f6ede6db1e`.

The result remains `QUALITY_LIMITED_DEVELOPMENT_EVIDENCE_RETAINED`. One galaxy
cannot establish a universal formula. The cluster/local comparison, full
population and far-outer-radius tests, matter/light coupling, external boundary,
stability and independent validation remain required.
