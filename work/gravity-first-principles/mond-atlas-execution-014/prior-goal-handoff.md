# Active goal: gravity pattern system and MOND observation atlas

The user explicitly requested tasks and execution on 2026-09-06. The goal is
active and unfinished. Read work/gravity-first-principles/mond-atlas-execution-013/
README.md and execution-status.json, then docs/GRAVITY_PATTERN_SYSTEM_TASKS.md.
The previous detailed handoff is archived in execution-013/prior-goal-handoff.md.

NEW: permissions now allow filesystem and network access. Git fetch/push work.
Prior validated atlas milestone is on main at 34b156ac95e9b03a8fc27a82bb99e3727331a756.
No longer repeat the old approval/filesystem publication blocker. Preserve
concurrent changes; coordinator owns Git and publishes ordinary fast-forward
updates to main, with exact intended files and raw observations outside Git.

Python C:/Users/henry/AppData/Local/Programs/Python/Python313/python.exe has NumPy,
SciPy, sklearn, Astropy and working CuPy13.5.1 CUDA on RTX5090. Its Torch2.7.1
is CPU-only. Actual GPU nonlinear learning now passes CPU/independent reference
controls. Do not infer CUDA readiness from import alone or stop other processes.

First learning run: mond-atlas-pattern-learning-001, report pattern-findings-001.
126 exposed SPARC galaxies, 8 comparisons, 3 five-fold seeds, 3024 OOF rows,
16 structure shuffles. Combined nonlinear structure yields only 2.26% mean
squared-error gain, with -5.33% to +10.29% across split seeds and uncertainty
including zero. No stable structural correction. Gas alone does not help here.
All results retained; no causal, unique-3D or independently confirmed law claim.
Published subset tests: 112 pass after source/native integration. More tests may exist from concurrent tasks.

TASK STATUS:
T2 metadata recovery complete: 01a077c5-6c38-7831-9701-c97dafed68b3
  mond-atlas-baryon-recovery-001. Both originals recovered and verified; nine
  source geometries, six uncertain orientations, three missing. Current fixed
  M/L=.6 fields unaffected by historical cleaned/global-color applicability bug.
T3 conditional native injections complete: 01a077c5-6e58-7a21-8e3b-185a99065e49
  mond-atlas-native-selection-001/run-001. 864 actual-background and 2304
  simulated injections; mask is not certified publisher mask, covariance and
  online spectral response remain unresolved. Source packages reviewed in013.
T5 lensing pilot ACTIVE: 01a077c5-7055-7f80-a7d1-7ddbbe69cb4e
T6 synthetic motion controls ACTIVE: 01a077cf-96a5-75c3-b0e3-db86f91e6eef
  T6 is THEORY_BENCHMARK_ONLY or SOURCE_BLOCKED; no actual galaxy scoring.

Each active task owns separate new paths and does not write Git/common modules.
Lensing task disclosed incidental exposure of some reserved SLACS table rows;
do not claim the whole old reserved sample remains unseen. Review and publish
its completed package and motion controls after delivery, not before.

Next: review and publish completed task milestones; advance T6 resolved source
ensembles and motion controls using their findings; extend the actual learning
harness to resolved observables and eligible additional systems. T7 tests
physical-group/survey transfer and candidate formulas only after valid inputs.

Scientific scale unchanged: 13525 identity groups (not certified distinct),
175 radial galaxies, 12 resolved seeds, 22 source-image fits, 29 conditional
field runs for one galaxy, ZERO admitted full-field cube likelihoods. The goal
remains 10-20 validated development pilots, expansion toward 100-300 eligible
resolved galaxies and larger population tiers, with real independent tests.

Read OPEN_GRAVITY_BUILDER_SOLVER_ADMISSION_POLICY_V1.md before new builders or
operators. Existing radial reanalysis is exploratory; full-field admission is
still SOURCE_BLOCKED. Preserve all failed and prior controls. HI is not total
mass; five historical STELLAR_MASS_MAP assets are cleaned flux, not mass truth.
3D sources require constrained alternatives. Lensing needs an explicit metric
or light-propagation model and direct observables, not halo-derived mass labels.
