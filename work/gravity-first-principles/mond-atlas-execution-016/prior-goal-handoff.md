# Active goal: gravity pattern system and MOND observation atlas

The user requested tasks and execution on 2026-09-06. The goal is active and
unfinished. Read work/gravity-first-principles/mond-atlas-execution-015/README.md
and execution-status.json, then docs/GRAVITY_PATTERN_SYSTEM_TASKS.md.
Prior handoff/task documents are preserved byte-for-byte in execution-015.

Filesystem/network/Git access works. The latest previously published milestone
is 25045db1463ded44e641fa0e7756dbeb04ba1a8a (execution-014). Execution-015 is the
next NGC2976 source package; inspect Git to confirm its publication.
Do not repeat obsolete permission blockers. Coordinator owns publication;
fetch/integrate concurrent changes and push ordinary fast-forward milestones
to main. Keep all raw arrays/private data outside Git.

Python C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe has
NumPy2.2.6, SciPy1.16.1, sklearn1.7.1, Astropy7.1.1 and working CuPy13.5.1 CUDA
on RTX5090. Torch2.7.1 here is CPU-only. No existing process was stopped or
environment replaced. CuPy learning passed CPU/independent reference checks.

First learning run: mond-atlas-pattern-learning-001, report pattern-findings-001.
126 previously exposed SPARC galaxies, 8 comparisons, 3 five-fold seeds,
3024 OOF rows and 16 structure shuffles. Combined nonlinear features yield
2.26% MSE gain, with -5.33% to +10.29% across splits and uncertainty including
zero. No stable structure correction; gas-only features do not help here.
All outcomes retained. No causal, unique-3D or independently confirmed law claim.

Execution-014 parent audit: 152 tests pass, zero failures/errors/skips; 609 prior
manifest entries and 787 unique input/artifact files rehashed. Raw files remain
required for reproduction. verify_mond_atlas_execution014.py is the entry point;
it requires a fresh report output and checks archived prior mutable context.

COMPLETED FIRST TASK MILESTONES (T6 has a new active follow-up, described below):

T2: Recover independent baryonic mass inputs
  Thread 01a077c5-6c38-7831-9701-c97dafed68b3, baryon-recovery-001.
  Original S4G tables recovered; nine geometries, six uncertain orientations,
  three missing. Current fixed M/L=.6 fields unaffected by historical misuse of
  an uncleaned integrated-color relation on cleaned P5 images. Five historical
  STELLAR_MASS_MAP products are flux images, not stellar-mass observations.
T3: Validate native gas cube selection
  Thread 01a077c5-6e58-7a21-8e3b-185a99065e49, native-selection-001/run-001.
  864 actual-background and 2304 conditional simulated injections. Publisher
  mask, spectral response, clean line-free support and covariance unresolved.
T5: Acquire a direct-observable lensing pilot
  Thread 01a077c5-7055-7f80-a7d1-7ddbbe69cb4e, lensing-pilot-001/replay-002.
  Three SLACS systems, measured redshifts/aperture dispersions, photometric
  alternatives and three native HST SCI/ERR/DQ images. 29 principal downloads,
  650187552 bytes rehashed. 21 tests. PSF, foreground/arc masks, image covariance,
  independent mass calibration and light-propagation closure remain missing.
  Auger SPS mass conditions on velocity dispersion: ancillary, not independent
  input mass. Grillo photometric alternatives retain their assumptions.
  Legacy confirmation rows incidentally exposed: old reserved sample is not
  certified unseen. No crossmatch to the nearby HI pilots is established.
T6: Build resolved galaxy motion controls
  Thread 01a077cf-96a5-75c3-b0e3-db86f91e6eef, motion-controls-001/run-002.
  THEORY_BENCHMARK_ONLY. 25 numerical controls and six synthetic cases, known
  source/instrument/diagonal covariance. Warps, streaming and emission asymmetry
  improve held-out predictions; zero-amplitude case slightly worsens. Face-on
  motions remain unidentified despite predictive pass; retain that failure.
  No pressure support, force balance or observed galaxy likelihood is supplied.

Additional parent source milestone: stellar-transfer-001 and -002, summary in
stellar-transfer-findings-001. Five P5 sums reconstructed against P1 and measured
translations checked on disjoint blocks. Four pass relative transfer plus prior
absolute Gaia support: NGC2903/2976/3198/3521. NGC4214 lacks absolute support and
has a 9.11% local quadrant mismatch. Both split passes are sensitivity on the
same data, not independent observations or posterior samples. Old fields intact.

The original stellar-transfer runner is frozen by executed hashes. Use
run_mond_atlas_stellar_transfer_checked.py for future work with a copied config,
fresh private_directory and fresh public output. The checked wrapper prevents
private sample overwrite; all ten previous sample packets remain unchanged.

NEXT WORK:

1. NGC2976 generic registered source maps now exist in generic-source-001/run-002,
   report findings-002. Extend them through common-basis source reprojection over
   plausible vertical profiles, retaining beam/calibration/depth alternatives.
   Use build_mond_atlas_registered_source.py; do not reuse the old hardcoded
   NGC2903 adapter. No new field or observed-motion score exists yet.
2. Extend motion controls to correlated channel noise and selection, and add
   pressure-support closure before interpreting speeds as gravitational force.
   Native noise/selection checks remain conditional, not observed truth.
3. Validate the lensing instrument/foreground model and explicit relativistic
   light closure. Inferred total/lens/halo masses are not baryonic truth labels.
4. Expand eligible resolved systems and run galaxy/group/survey transfer tests
   before extracting or promoting candidate gravity formulas.

Scientific scale: 13525 identity groups (not certified distinct), 175 radial
galaxies, 126 learning galaxies, 12 resolved seeds, 22 source-image fits and 29
conditional field runs for one galaxy. ZERO admitted full-field cube likelihoods
and ZERO admitted lensing likelihoods. Target remains 10–20 validated development
pilots then 100–300 eligible resolved galaxies plus broader population tiers.

Read docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md before new physical
builders/operators. Existing radial reanalysis is exploratory; full-field source
admission remains SOURCE_BLOCKED. Preserve failed/prior controls and uncertainty.
HI is not total ordinary matter. Unknown depth must be represented by constrained
alternatives, never invented observed 3D truth. A nonrelativistic acceleration
formula alone does not specify light bending. The overall goal remains active.

EXECUTION-015 UPDATE:
NGC2976 generic source: 12 cases, 36 tracer grids, 72 conditional mass rows;
161 publication-subset tests pass. Stellar/HI/CO coverage inside nominal 6 kpc
is 99.53/78.50/37.54 percent. Nominal conditional mass 2.394e9 Msun; conversion
alternatives 1.619–3.243e9; distance-scatter alternatives 1.549–3.424e9. These
are not confidence intervals or directly measured total masses. Orientation
changes redistribute tracer coordinates while total mass changes less than
0.6 percent. All missing phases/source likelihood/3D admission remain open.
Nine new tests and independent real-header Jacobian/translation checks pass.
Initial cropped-test assumption and FITS-header failure preserved with original
code snapshots. Corrected report legend is findings-002; both numeric reports agree.
There are now two galaxies with conditional source-grid work, still only one
with the 29 old conditional fields; ZERO admitted observed full-field likelihoods.

T6 follow-up ACTIVE: same thread 01a077cf-96a5-75c3-b0e3-db86f91e6eef, new turn
01a077e1-2fe4-7fe2-8a22-03aac0682bda, last confirmed live via wait_threads.
Owns only new motion_covariance scripts/config/tests and
mond-atlas-motion-covariance-001 public/private paths. Known correlated channel
noise, proper marginal/conditional covariance, fresh-noise and truth controls.
30 statistical and 25 imported numerical controls pass; zero-amplitude cases
complete, remaining study active at handoff. Do not publish before delivery/review.
No source data/observed covariance admission or pressure closure supplied by this
THEORY_BENCHMARK_ONLY follow-up. Parent owns Git. Preserve its concurrent files.
