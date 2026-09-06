# Active goal: gravity pattern system and MOND observation atlas

The user explicitly requested tasks and execution on 2026-09-06. The goal is
active and unfinished. Read work/gravity-first-principles/mond-atlas-execution-012/
README.md and execution-status.json, then docs/GRAVITY_PATTERN_SYSTEM_TASKS.md.
The previous detailed handoff is archived in execution-012/prior-goal-handoff.md.

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
Published subset tests: 89 pass. More tests may exist from concurrent tasks.

ACTIVE APP TASKS (inspect status/results and review before integrating):
T2 baryonic input recovery: 01a077c5-6c38-7831-9701-c97dafed68b3
T3 native gas selection: 01a077c5-6e58-7a21-8e3b-185a99065e49
T5 direct lensing pilot: 01a077c5-7055-7f80-a7d1-7ddbbe69cb4e
Each owns new matching scripts/config/tests plus private and report directories.
They must not edit common modules or Git. Source tables reportedly recovered
with old hashes; await executed reconciliation. A historical stellar-color
relation applicability mismatch is under audit. Do not mutate old receipts.
Lensing task disclosed incidental exposure of some reserved SLACS table rows;
do not claim the whole old reserved sample remains unseen.

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
