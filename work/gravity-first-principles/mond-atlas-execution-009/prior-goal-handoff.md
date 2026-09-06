# Active goal: MOND observation atlas

User authorization: “Make the above our goal and complete the work,” 2026-09-06.
The active Codex goal remains unfinished. Build the catalog and execute ordinary-
matter Newtonian/full-field MOND predictions from independently constrained 3D
ensembles, then compare through validated instruments, masks and noise. The
experiment contains no dark-halo term. A spectral velocity axis is not depth.

## Current result

Read `work/gravity-first-principles/mond-atlas-execution-008/README.md` and its
`execution-status.json`, `verification.json`, and `publication-manifest.json`.
This supersedes execution-007's current status while preserving every earlier
scientific artifact and failure. The previous handoff is archived with the report.

- Identity overlay: 13,525 groups, not certified distinct galaxies; 90 proximity
  pairs and 58 missing coordinates remain unresolved.
- Radial baseline: 175 galaxies, 126 passing the declared descriptive cuts.
- Source basis corrected: image inversion and gravity interpolation now share
  bilinear source nodes. Revised stellar fits have 1.21% and 4.80% image RMS;
  they are not measured depths or a calibrated source posterior.
- Previous mixed-source convergence failure resolved: 1.06% Newtonian and 0.95%
  QUMOND at the next lateral refinement. The former failure remains retained.
- Both corrected coefficient sets now have separate lateral, vertical and box
  checks: midplane full-vector gates pass = True.
- Above-plane three-component checks at 0.25, 0.5 and 1 kpc: full-vector gates
  pass = True. Read component errors and
  individual failures rather than interpreting one Boolean as universal accuracy.
- Conditional pattern: at radius 5 kpc the two models change QUMOND's mean
  force-equivalent speed by 1.7%, but its downward pull at height 0.25 kpc by
  32%. Model total mass differs by only 0.50%. This is joint deprojection
  sensitivity, not an observed anomaly or an observational confidence interval.
- Noise: 192 partition checks. Nine galaxies pass every declared split;
  NGC2841, NGC2903 and NGC3198 fail some. Eight galaxies pass both raw/P1
  astrometry and all background partitions. These are not valid galaxy likelihoods.
- Five files historically named STELLAR_MASS_MAP are explicitly classified as
  cleaned stellar flux in MJy/sr by the new role overlay, not preconverted mass.
- Totals: 22 source-image fits, 29 conditional field runs for one galaxy,
  57 passing unit tests, zero admitted full-field galaxy cube predictions.

## Admission and remaining work

Read `docs/OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md` before any new
source/solver or motion scoring. New packages prospectively declare SOURCE_BLOCKED.
The older motion comparison remains exploratory and nonadmitted; no retrospective
preregistration is claimed. Keep all previously used seed galaxies development-exposed.

1. Model source noise, native beams/pixel footprints, absolute calibration,
   source conversions and missing mass phases; retain nondetections and missingness.
2. Recover both missing original S4G geometry tables; the derived record is bound
   but raw-source revalidation is incomplete. Validate P1-to-P5 transfer beyond
   NGC2903 and account for dust in the seven raw IRAC stellar inputs.
3. Constrain geometry/depth/exterior-field ensembles independently of target
   motions; resolve remaining numerical failures without weakening gates.
4. Improve and validate covariance across spatial splits and within the galaxy;
   establish selection-mask validity. Do not choose favorable background splits.
5. Add distinct AQUAL controls and an instrument-aware motion model for pressure,
   warps, streaming and other permitted motions; execute true cube likelihoods.
6. Complete 10–20 development pilots and expand toward 100–300 eligible resolved
   systems, then evaluate galaxy/group/survey transfer. Population catalog rows
   do not substitute for executed resolved predictions.
7. Publish the combined verified milestone when a write route is permitted,
   with a fresh ancestry check and a non-forced ref update preserving remote work.

## Access and preservation

The working Python 3.12/NumPy CPU runtime is bundled under the user's
`.cache/codex-runtimes`. The previous CUDA virtualenv cannot start. Direct shell
downloads are denied. Linked Git metadata is outside the writable root; the
latest ordinary fetch failed. The connected GitHub integration can access the
repository; its preparation check found main unchanged at
afc721a13782acec4ebc94ad8f6d97ed71be7152. Its first blob write was blocked because
approval is required but the session policy is never. Nothing was published.
Do not alter the restricted local Git metadata or bypass the connector rejection.
Raw observations and large field arrays remain outside Git. Preserve unrelated
untracked work. The current report manifest contains the intended publication set.

Key immutable runs: catalog-004, identity-001, radial-002, astrometry-001,
noise-002 and noise-robustness-001, source-basis-001, field-005/006, offplane-001
(all prefixed `mond-atlas-` under `work/gravity-first-principles`). Earlier
execution reports preserve the admission correction, source representation
failures and numerical counterexamples. The goal remains active.
