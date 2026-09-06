# Active goal: gravity pattern system and MOND observation atlas

The user requested tasks and execution on 2026-09-06. The goal is active and
unfinished. Read work/gravity-first-principles/mond-atlas-execution-016/README.md
and execution-status.json, then docs/GRAVITY_PATTERN_SYSTEM_TASKS.md.
Prior handoff/task documents are preserved byte-for-byte in execution-016.

Filesystem/network/Git access works. The latest previously published milestone
is 0d3aace45309ffa2792c2d78953a624b3fb5c4d5 (execution-015). Execution-016 is the
next source-closure/correlated-noise package; inspect Git to confirm publication.
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

COMPLETED TASK MILESTONES (all app tasks idle; T6 follow-up now completed below):

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

1. NGC2976 source maps and twelve common-basis height fits now exist. Investigate
   source resolution, beam and signed-data noise before a physical height or
   gravity claim. Use registered-source and registered-projection adapters;
   retain the earlier geometry/mass alternatives. No new field/observed score.
2. Correlated-noise motion controls now pass. Validate actual observed channel
   and spatial covariance/selection and add pressure support before interpreting
   speeds as gravitational force. Known synthetic covariance is not observed truth.
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

T6 correlated-noise follow-up COMPLETE: thread 01a077cf-96a5-75c3-b0e3-db86f91e6eef,
turn 01a077e1-2fe4-7fe2-8a22-03aac0682bda completed; authoritative app status idle.
No live job remains. Package motion-covariance-001/run-001, 96 fits/192 starts,
30 statistical +25 mechanics controls, 18 new tests. All 36 noise draws and
96 predictions replay exactly. Three injection classes, four independent noise
realizations each, two overlapping folds. Correct covariance and deliberate
diagonal approximation, circular and expanded mean models. Combined circular
conditional q/N 1.084 looks close to noise but fresh q/N 1.517; expanded fresh 1.013.
Zero-extra-amplitude models slightly worsen with added motion freedom. All
separate marginal/conditional/fresh/noiseless metrics retained. Fixed masks and
known AR(1) channel noise with independent spatial pixels remain synthetic.
No pressure, observed covariance, source uncertainty or gravity admission.

EXECUTION-016 UPDATE: 179 tests pass. NGC2976 common-basis vertical projection
in mond-atlas-ngc2976-projection-001/run-001 and findings-001 executes 12 fits,
all converge, ten above the 5% descriptive flag. HI thin/.1kpc RMS .13/1.50%; stars
thin 5.23%. CO negative measurements impose a 9.23% nonnegative-prediction RMS floor;
the thin fit has 9.85%, so 5% is not a physical rejection threshold. All negative values
are retained. Heights 0/0.1/0.2/0.4 kpc are model alternatives, not measured/posterior
heights. Zero height is a sheet limit, not a finite-volume field input. Existing
nodal operator unchanged; five independent controls replay. Next needs signed
source likelihood, finer source/beam checks and additional galaxies before
new observed-gravity scoring. Total source-image fits now 34 (prior 22 + new 12),
conditionalfields still 29 for one galaxy, conditional source work for two galaxies.
ZERO admitted observed full-field cube or lensing likelihoods. Goal remains active.
