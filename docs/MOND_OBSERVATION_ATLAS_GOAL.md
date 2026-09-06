# MOND observation atlas and gravity pattern system: active handoff

Authoritative current parent receipt: `work/gravity-first-principles/mond-atlas-execution-017/`.
The overall unbudgeted goal remains ACTIVE. This is a completed source-resolution
increment, not completion of the full system or discovery of a new gravity law.
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
The native NGC2976 header/geometry check is not a full remote-cube rehash.

T5: Acquire a direct-observable lensing pilot
Thread 01a077c5-7055-7f80-a7d1-7ddbbe69cb4e. Three SLACS systems and native HST
SCI/ERR/DQ acquired and replayed, 29 principal assets / 650187552 bytes.
PSF, foreground/arc separation, full noise model and independent mass calibration
remain. Auger SPS mass uses velocity-dispersion-conditioned metallicity: ancillary,
not independent input mass. Legacy reserved table rows were incidentally exposed;
do not claim that old reserved sample is unseen. No established HI-seed match.
Current bounded follow-up RUNNING: turn 01a077f6-c6ab-7cb0-8dc8-add09ac5d379,
THEORY_BENCHMARK_ONLY light-potential/deflection controls. Its unfinished files
are not part of execution-017. Query this exact app handle before changing status.

T6: Build resolved galaxy motion controls
Thread 01a077cf-96a5-75c3-b0e3-db86f91e6eef. Synthetic mechanics and correlated-noise
increments completed: 25 mechanics controls, 30 statistical controls, 96 fits /
192 starts across four independent noise realizations per class. All noise and
prediction arrays replay. Correct conditional channel forecasts can look good
while a wrong motion model fails on fresh noise. Combined circular fresh q/N
1.517 versus expanded 1.013; zero extra motion slightly worsens with more freedom.
Known synthetic masks/covariance and imposed velocities are not observed truth.
Current bounded follow-up RUNNING: turn 01a077f6-c647-79f1-875b-0f1d950e6c66,
THEORY_BENCHMARK_ONLY pressure/force-balance controls. Its unfinished files are
not part of execution-017. Query this exact app handle before changing status.

## Next executable work

1. Finish, independently review and publish the existing pressure and light
   follow-ups. Keep their theory status distinct from observational admission.
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
