# MOND observation atlas and gravity pattern system: active handoff

Authoritative current parent receipt: `work/gravity-first-principles/mond-atlas-execution-018/`.
The overall unbudgeted goal remains ACTIVE. The current increment completes pressure and light-propagation theory controls;
it does not complete the full system or discover a new gravity law.
Prior handoffs are preserved exactly in each execution package.

## User authorization and environment

The user authorized building/executing the system, separate tasks, using the
RTX 5090, and publishing validated milestones to GitHub main regularly.
Repository: lrspeiser/Invariant. Current checkout is the research branch;
publish with an ordinary fast-forward to main after fetching remote changes.
AGENTS.md requires intended-file staging and raw arrays outside Git. No force
push, blanket add, deletion of concurrent work, or repeated permission request.
The former filesystem/network restrictions are resolved.

Python313: C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe.
CuPy 13.5.1 works on the RTX 5090 with CUDA 12.9. PyTorch remains CPU-only.
Use bounded GPU allocations and one CPU numerical thread; leave other processes
alone. Source-resolution-001 used a maximum 19,091,968-byte default CuPy pool.

## What has actually been executed

- 13525 identity groups, not certified distinct galaxies; 175 radial galaxies;
  126 development-learning galaxies; 12 resolved seeds.
- The first CUDA learning pass on 126 galaxies found a small, split-sensitive
  combined-feature gain, not a stable structural correction. Galaxy folds do
  not supply physical-group/survey holdouts or pristine confirmation.
- Two galaxies have conditional source-grid work: NGC2903 and NGC2976.
  There are 70 source-fit executions (34 earlier + 36 refinement fits), including
  reruns and alternative parameterizations of the same observations. These are
  not 70 independent models, galaxies, measured depths, or validation samples.
- 29 prior conditional field runs remain for NGC2903 only.
- ZERO admitted observed full-field cube likelihoods and ZERO admitted lensing
  likelihoods. Target remains 10–20 validated development pilots, then an
  eligible 100–300 resolved sample and broader population tiers.

## New result: fixed-image source refinement

`mond-atlas-source-resolution-001/run-001`, findings-001, and verification.json.
Source-only SOURCE_BLOCKED. Exact observed NGC2976 cells, coverage, radius masks
and weights are held fixed while latent bilinear source spacing changes from
125 to 62.5 to 31.25 pc. Heights 0/.1/.2/.4 kpc are illustrative alternatives.
All 36 CuPy fits converge. Eight independent benchmark tests pass before the
source packet is opened; all 36 saved projections, stationary optimality
receipts and unchanged observation weights replay on CPU. No response read.

Thin-sheet stellar RMS falls 5.228% -> .1675% -> .0807%. The old discrepancy
was strongly affected by source representation. HI falls .1253% -> .0128% ->
.0102%. CO falls 9.8533% -> 9.2343% -> 9.2343%, reaching the unavoidable
9.234263% nonnegative prediction floor from 411 negative signed measurements.
Finite-height alternatives retain residuals; at .1 kpc the finest stellar/HI/CO
RMS is 14.57/.579/10.77 percent. Do not convert this into measured thickness:
native beams differ, source covariance/calibration/depth remain uncertain, and
latent refinement adds freedom without adding observations. Zero height is a
sheet limit and cannot be loaded as finite volume density by a 3D field solver.
All old 12 NGC2976 height fits remain unchanged.

## Existing task milestones and current live follow-ups

T2: Recover independent baryonic mass inputs
Thread 01a077c5-6c38-7831-9701-c97dafed68b3. Completed bounded metadata increment.
Current follow-up RUNNING: turn 01a07800-bcf4-7132-b9d6-be1212b9020b,
NGC3198 generic-source-002 with the unchanged source adapter and fresh frozen
geometry/conversion sensitivities. No new source package is counted yet.
Original S4G tables, nine geometries (six uncertain), three missing. Four seeds
have relative stellar-transfer and prior absolute Gaia support: NGC2903/2976/
3198/3521. NGC4214 lacks absolute support and has a 9.11% local mismatch. P5
STELLAR_MASS_MAP products are flux, not observed stellar mass. Historical misuse
of an uncleaned color relation did not alter current fixed M/L=.6 fields.
Use run_mond_atlas_stellar_transfer_checked.py and fresh private/public paths;
do not reuse original frozen runner output paths.

T3: Validate native gas cube selection
Thread 01a077c5-6e58-7a21-8e3b-185a99065e49. Completed conditional native pilot.
864 actual-background and 2304 simulated injections. Exact publisher mask,
clean background support, spectral response and observed covariance unresolved.
Current follow-up RUNNING: turn 01a07804-0d7d-74f0-97d3-f5c7a828be03,
native-covariance-001 training/validation on spatial background blocks.
It remains SOURCE_BLOCKED and is not part of the current publication.
The native NGC2976 header/geometry check is not a full remote-cube rehash.

T5: Acquire a direct-observable lensing pilot
Thread 01a077c5-7055-7f80-a7d1-7ddbbe69cb4e. Three SLACS systems and native HST
SCI/ERR/DQ acquired and replayed, 29 principal assets / 650187552 bytes.
PSF, foreground/arc separation, full noise model and independent mass calibration
remain. Auger SPS mass uses velocity-dispersion-conditioned metallicity: ancillary,
not independent input mass. Legacy reserved table rows were incidentally exposed;
do not claim that old reserved sample is unseen. No established HI-seed match.
Light follow-up COMPLETE and task idle: turn 01a077f6-c6ab-7cb0-8dc8-add09ac5d379.
light-projection-001 passes 203 required checks and 23 tests in each of two runs,
with all 251 records identical, including 27 retained coarse diagnostic failures.
Separate Phi/Psi callables, point/Plummer/asymmetric surface-density references,
angular distances, image roots and signed magnifications pass. Equal potentials
remain a benchmark closure only. Candidate field/relativistic/source/instrument
admission is still missing; no observed likelihood. Parent replays all 244
mathematical records, checks seven metadata/hash/test gates, and rehashes receipts.

T6: Build resolved galaxy motion controls
Thread 01a077cf-96a5-75c3-b0e3-db86f91e6eef. Synthetic mechanics and correlated-noise
increments completed: 25 mechanics controls, 30 statistical controls, 96 fits /
192 starts across four independent noise realizations per class. All noise and
prediction arrays replay. Correct conditional channel forecasts can look good
while a wrong motion model fails on fresh noise. Combined circular fresh q/N
1.517 versus expanded 1.013; zero extra motion slightly worsens with more freedom.
Known synthetic masks/covariance and imposed velocities are not observed truth.
Pressure follow-up COMPLETE and task idle: turn 01a077f6-c647-79f1-875b-0f1d950e6c66.
pressure-support-001/run-002 passes 42 controls and 21 tests, with 24 noisy plus
six noiseless fits and one retained impossible equilibrium. Pressure-blind
harmonic fit gets force amplitude 400 instead of 625 (-36%) while reproducing
speeds exactly. Known pressure resolves force only under the supplied closure;
fresh speeds alone cannot break that degeneracy. Varying-pressure fresh q/N
improves 3.666 -> 1.037. These are radial fluid benchmarks; only the separate
flaring control closes both 3D radial and vertical Euler equations. No observed
pressure, general stress/transport/energy or new cube likelihood is supplied.
Parent replays 24 noise arrays and 30 predictions exactly. Both new theory
modules pass 44 relevant parent tests; previous modules/results remain unchanged.

## Next executable work

1. Continue the running NGC3198 source and NGC2976 background-covariance tasks;
   review and publish each executed package. T5/T6 theory increments are complete.
   Keep theory controls distinct from observational admission.
2. Build signed source-noise/beam controls and validate actual native cube
   covariance/selection before interpreting source residuals or gas speeds.
3. Expand the generic registered-source adapter to eligible additional galaxies,
   carrying mass-conversion, calibration, geometry and depth alternatives.
   NGC2976 refined sources need finite-volume compatibility and numerical field
   controls before adding conditional forces; no response tuning.
4. Evaluate structure additions beyond motion/instrument/source uncertainty,
   with whole galaxies and physical groups/surveys withheld from selection.
   Translate only reproducible effects into candidate physical formulas.

Read docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md before new operators.
HI is not all ordinary matter. Unknown 3D structure is not observed truth.
Inferred lens/halo/dynamical masses are not independent baryonic labels.
Nonrelativistic MOND or a new acceleration formula alone does not define photons.
The system currently supports exploratory diagnostics and theory controls; a
new first-principles gravity formula has not been established.

## Execution-019: adaptive sparse formula search

Published a runnable greedy expression-search loop on the RTX 5090, with training-only transforms, nested whole-galaxy complexity selection, exact formula replay and shuffled controls. On 126 previously exposed galaxies, additions from 30 candidate expressions made outer MSE worse by 3.62%, 4.55% and 0.90% across three splits. Seven of fifteen fits selected no additional term. No expression advances as a validated gravity law. Five pre-access tests and all 110 saved formula replays passed; full first-seed CPU selection matches GPU. See `work/gravity-first-principles/mond-atlas-formula-search-001/README.md`.

This milestone adds no admitted observed full-field or lensing likelihood. The separate 16 NGC2976 conditional field calculations are saved locally, but grid refinement failed accuracy gates and their review is not included here. NGC3198 and native covariance supporting tasks stopped at usage limits; intermediate outputs remain available. Overall research goal is unfinished.

## Execution-020: measured background uncertainty transfer

Completed review of the interrupted native covariance run and executed a new spatial-aperture test on its 29 western and 27 eastern NGC2976 background cores. Seventeen pre-access tests pass. Western channel models and ranking replay; independent aperture aggregation and inverse scoring reproduce six covariances and all 324 core scores. Independent-pixel variance underpredicts eastern 4x4-pixel fluctuation power by 8.13x. A western aperture model predicts eastern total power within 3.1% at all six sizes, but its full channel q/N gate fails at sides 12 and 24. Failures retained; no shrinkage retuning, source-region admission or new gravity score. See `work/gravity-first-principles/mond-atlas-aperture-noise-001/README.md`.

T3 now has reviewed native covariance and spatial-aggregation diagnostics. Its historical task stopped at a usage limit; the coordinator completed this review locally. Source-region selection/noise and a coherent joint spatial likelihood remain unresolved. NGC3198 correction and NGC2976 field-grid convergence remain separate unfinished work. Overall research goal is unfinished.

## Execution-021: fixed CMB axes and sky-position tests

Tested two published Planck SMICA KQ-corrected CMB axes, their bisector and Galactic/ecliptic/equatorial directions on 86 directly matched PROBES/SPARC development galaxies. Forty lack a direct coordinate/metadata match. Axial partial correlations are -0.042 and -0.020; adding the CMB direction terms worsens all random-galaxy and Galactic-octant prediction comparisons. Equatorial declination has the largest adjusted association (-0.276); all-sky terms offer small exploratory improvements, without a causal interpretation. Coverage is strongly uneven (45 of 86 galaxies in one octant). Four pre-access tests pass; independent sklearn replay verifies 132 nested choices and 2064 predictions. Reports, every galaxy-axis angle, exclusions and reviewed sky map are in `work/gravity-first-principles/mond-atlas-sky-alignment-001/`. This is line-of-sight analysis, not disk-spin alignment or a new gravity law. Next: coordinate completion and actual observing-reference/instrument controls. Goal unfinished.

## Execution-022: parallel relay and published halo tests

Completed three parallel agent investigations plus coordinator halo and geometry
tests: absorption/redirection, distributed secondary sources, finite memory and
feedback, and cumulative return shapes. Twenty tests pass. All 525 selected fit
rows from 175 SPARC galaxies match source tables; 504 pilot vectors independently
match galpy within 1.69e-12. This is fitted-halo calibration, not observational
validation. A manufactured distributed disk becomes nearly spherical far out,
but its inner strength differs from the point-calibrated halo. Absorption alone
weakens the tested attraction; strong feedback lengthens memory and can become
unstable. Scalar disk boosts fail off-plane vector geometry. Two finite-return
shapes interpolate NFW/Burkert well but degrade in outer extrapolation. All
failures and scope limits are retained. See
`work/gravity-first-principles/mond-atlas-relay-001/README.md`.

Next priority: a conservative distributed response with ordinary-matter rules
for strength, scale and finite extent, then real-source and held-galaxy tests.
No new observed full-field, cluster, Solar System or lensing score is admitted.
The existing data/noise and field-convergence blockers remain. Goal unfinished.

## Execution-023: real radial clock/relay formula comparisons

Ran the frozen 713-candidate comparison on the RTX 5090 using 102 eligible
galaxies and 2212 radii from the 139 historically exposed identities. Reserved
archive members were not parsed by the fit runners. Independent replay checks
all 180 selections and 79,632 initial held-family predictions. Global parameters
are selected with training velocities; only predictor inputs are source-only.
Adjusted algebraic MOND wins every initial training selection; absorption selects
zero opacity. Original clock potential has excessive inner attraction. Separate,
explicitly post-hoc mass-scale and central-core repairs are recorded with frozen
grids and no claims of fresh confirmation. See the complete results and limits:
`work/gravity-first-principles/mond-atlas-clock-relay-001/README.md`.

These are source-backed radial empirical tests, not an observed 3D operator or
evidence that time supplies energy. Energy exchange, source histories, actual
distributed 3D source fields, cluster/Solar System transfer and lensing remain
unresolved. No parameter is inferred independently of training responses merely
because its formula uses photometry. Overall research goal remains unfinished.
