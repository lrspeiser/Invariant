# Active goal: MOND observation atlas

User authorization: “Make the above our goal and complete the work,” 2026-09-06.
The active Codex goal contains the full objective. This file is its repository handoff.

Build the largest trustworthy, deduplicated observation catalog available, then
predict motion from independently constrained baryonic matter using Newtonian
gravity and full-field MOND. Model observationally allowed 3D structures as
ensembles; never call a spectral velocity axis spatial depth. Compare to the
observations through instrument response, validated masks, and correlated noise.
No dark-halo term is part of this experiment.

## Executed milestone

- Design: `work/gravity-first-principles/mond-atlas-design-001/README.md`.
- Protocol: `configs/mond_atlas_v1.json`.
- Current catalog: `work/gravity-first-principles/mond-atlas-catalog-004/`.
- Current radial baseline: `work/gravity-first-principles/mond-atlas-radial-002/`.
- Numerical controls: `work/gravity-first-principles/mond-atlas-numerics-001/`.
- Radial report/viewer: `work/gravity-first-principles/mond-atlas-execution-002/`.
- Current readiness report: `work/gravity-first-principles/mond-atlas-execution-003/`.
- Current conditional-field report: `work/gravity-first-principles/mond-atlas-execution-006/`.
- Current source reprojection correction: `work/gravity-first-principles/mond-atlas-execution-007/`.
- Source-image inverse/mixture runs: `work/gravity-first-principles/mond-atlas-projection-001/` and `mond-atlas-projection-002/`.
- Reprojected-source fields and retained mixed-model failure: `work/gravity-first-principles/mond-atlas-field-003/` and `mond-atlas-field-004/`.
- NGC2903 conditional source: `work/gravity-first-principles/mond-atlas-source-001/`.
- Full-field runs and stricter vector controls: `work/gravity-first-principles/mond-atlas-field-001/` and `mond-atlas-field-002/`.
- Object identity overlay: `work/gravity-first-principles/mond-atlas-identity-001/`.
- Stellar footprint audit: `work/gravity-first-principles/mond-atlas-astrometry-001/`.
- Current real background covariance audit: `work/gravity-first-principles/mond-atlas-noise-002/`.

The base catalog contains 13,530 identity groups. Five exact same-release NSA
identifier matches reduce the current grouping overlay to 13,525 groups, with
90 proximity pairs and 58 missing coordinates still needing resolution. These are not 13,525 certified
distinct galaxies or 3D models. All 137 resolved-seed assets were rehashed.
The SPARC archive matches all 3,391 stored decimal-string rows. Radial comparisons
were executed for 175 galaxies; 126 pass the declared descriptive-analysis cuts.
The new full-field and cube building blocks pass analytic and covariance controls.

## Required work that remains

The newest source check supersedes treating the old nominal thickening as
observationally consistent. Stretching the measured stellar image into a plane
and adding a 0.4 kpc vertical thickness smears the image again: its projected
source-image mismatch is 22.87% in the declared coverage-weighted diagnostic.
Nonnegative source inversion reduces the 0.4 kpc case to 8.38%, still above the
5% gross-mismatch flag. This threshold is not a calibrated noise likelihood.
Single 0.1/0.2 kpc layers and thin-plus-thick mixtures can reproduce the image
more closely while remaining distinct in depth. No unique height or posterior
admission follows. Read execution-007 before using any earlier source result.

Read `mond-atlas-execution-006/admission-audit.json` first. The first conditional
NGC2903 source builder did not complete the required admission record before
the published rotation comparison was opened. That comparison is retained as
exploratory, excluded from admitted scientific scores, and must not be described
as preregistered validation. All future motion scoring must complete the package
in `docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md` prospectively.

1. Resolve identity ambiguities and acquire per-object resolved data from the
   verified registry. Preserve nondetections, source selection and prior exposure.
2. Complete stellar-map transfer and co-spatial total baryon coverage for the
   pilot. The footprint audit corrected a previous test error: 283 Gaia positions
   fell outside finite mosaic coverage. Eleven raw/P1 stellar images now pass
   the original strict coordinate thresholds; NGC4214 lacks enough reference
   stars. Only NGC2903 has a validated P1-to-cleaned-P5 transfer, and its shared
   matter coverage is restricted. Seven other seed sources are raw IRAC flux
   products, requiring stellar/dust separation or explicit uncertainty modeling.
3. Construct source-only 3D ensembles, including stellar/HI/molecular conversions,
   thickness, flaring, warp/bulge alternatives, exterior-field and missing-phase
   uncertainties. Do not infer those inputs from the target motion and then call
   the prediction independent.
4. Complete astrophysical boundaries and distinct AQUAL controls. Eleven
   conditional NGC2903 Newtonian/QUMOND field runs now exist, using observed
   stellar/HI/CO maps with illustrative missingness, conversion and depth choices.
   The initial mean radial-force convergence passed but the full-vector check
   failed. A further refinement reduced vector differences to 0.58% Newtonian
   and 0.51% QUMOND, with every ring below 0.9%. Box and vertical checks also pass
   for the tested interior range. Isolation and MOND monopole boundaries remain
   conditional; no observed external field or full mass posterior is established.
5. Connect the observer-volume integrator to validated coordinate transforms,
   matter-to-motion models, gas pressure/streaming/lag, measured beam/channel
   responses, real spatial/channel covariance and independently validated masks.
   The guarded, quadrant-balanced background covariance audit passes preliminary
   moment checks for 11/12 galaxies; NGC3198 fails the residual channel-lag gate.
   Ten galaxies clear both the stellar-flux astrometry and background-noise
   checks. These are not ten validated galaxy likelihoods. Split robustness,
   nonstationarity, separability and the galaxy selection mask remain unresolved.
6. Execute 10–20 development pilots, expand toward 100–300 eligible resolved
   galaxies as coverage permits, and test transfer by galaxy/group/survey.
   Thousands of population records alone do not satisfy this stage.
7. Publish each validated milestone to main while preserving remote changes and
   unrelated local work. Raw observations remain outside Git.

The goal is **not complete**. There are no newly validated full-field galaxy cube
predictions in this milestone. Do not relabel the algebraic radial baseline as a
full QUMOND/AQUAL disk prediction.

The real-source field diagnostic preserves the bar: sideways force is about
15.5%, 7.5% and 5.1% of mean inward force at 2, 5 and 10 kpc, respectively, on
the 0.25 kpc grid. Circular averaging the same mass removes most of that sideways
force while leaving mean force-equivalent speeds nearly unchanged at these
radii. This is a conditional force pattern, not an observed streaming detection
or evidence for a new gravity law. Mass-conversion sensitivity is larger than
the tested thickness effect. Thirty-six atlas tests pass.

That earlier bar statement describes the old conditional source only; its
nominal stellar lift now fails the source-image check above. The latest phase
executed 18 source-image fits and 8 additional fields (19 field runs in total,
still only one galaxy), with 43 unit tests passing. The new thin-source model
passes the unchanged numerical gates. The mixed-source model fails lateral
convergence at 3.53% Newtonian and 3.14% QUMOND vector RMS (3% limit), despite
passing its vertical and box tests. Retain this failure and refine the grid;
do not inherit the thin model's certificate or relax the threshold. A memory-
bounded solver may be appropriate for the next lateral refinement. No new
kinematic response comparison was made in this source-only phase.

The original raw S4G geometry files referenced by the stored configuration are
absent from the current workspace and the checked original checkout. Their
derived record is present and bound, but raw-record revalidation remains
incomplete. This is additional to the existing source noise, absolute flux,
mass conversion, missing phases, exterior fields and 3D geometry requirements.

## Current execution access

Bundled Python 3.12/NumPy runs on CPU. The old CUDA virtualenv cannot start under
the current permissions. Direct shell downloads are denied. Git's linked
worktree metadata is outside the writable root, so fetch/commit/publication are
not available in this session. No permission bypass or remote publication was
attempted. Preserve the local milestone until ordinary access is restored.

Historical catalog attempts 001/002 stopped at source checks; catalog 003 was
superseded by 004's corrected source citation and code hashes. Radial 001 was
superseded by 002's added robust summary statistics and code hashes; scientific
predictions are unchanged. Execution report 001 failed at final input binding
and is superseded by 002. Retain these statuses; do not publish an incomplete
attempt as a successful atlas run.

Noise run 001 is also retained: its checkerboard starved some validation
quadrants. Noise run 002 repaired the split using geometry alone, preserving
guards and noise thresholds. A channel-correlation failure changes from NGC2841
to NGC3198 between these splits, so neither should be labeled intrinsically
problematic without further robustness work. Current source/noise code passes
28 tests, including held-region mutation and synthetic nonstationary-noise
controls. Eight new field/source tests bring the current total to 36.
Use the execution-007 publication manifest for the current combined
milestone; earlier publication manifests describe older local snapshots.

Report attempt 004 stopped on an asset-schema mismatch. Report 005 completed
the numerical/sensitivity account; 006 adds the required admission correction
without recalculating or fitting response data. Preserve the failed attempt
and the supersession history. The last published commit is still afc721a1;
the latest ordinary fetch again failed because linked Git metadata is outside
the writable root. No changes have been committed or pushed in this session.

Projection-001 froze SOURCE_BLOCKED explicitly before source-image testing and
does not perform motion scoring. Six independent projection/inverse controls
passed before real-source application; a seventh new vertical-cell control was
kept in a separate test file to preserve the earlier frozen test hashes.
Execution-007 includes the verified source figure, artifact bindings and all
current numerical failures. Do not run the retired exploratory response report
as a way around the admission policy.
