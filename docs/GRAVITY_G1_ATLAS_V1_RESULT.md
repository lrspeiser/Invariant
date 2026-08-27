# G1 production atlas v1 result

The sealed G1-v1 production campaign completed on 2026-08-27 with the decision
`BLOCK_G1_ATLAS_INCOMPLETE`.

## Measured result

- 139 admitted SPARC exploration galaxies and 2,720 rotation-curve tracers were evaluated.
- Each galaxy received exactly 100,000,000 new formula trials: 13,900,000,000
  candidate-galaxy trials in total.
- 138/139 galaxies retained at least one CPU-FP64-admitted formula.
- NGC2955 retained no admitted formula and is the sole counterexample.
- The 35-galaxy confirmation partition was not evaluated; the receipt records zero
  confirmation-evaluator accesses.
- The immutable atlas is
  `runs/gravity/g1-atlas/galaxy-formula-atlas-v1.json`, with content seal
  `cffa857882d8890cafcbffb3421296508d482049ae11503c9a097981e6c9cdf5`.

The assembled atlas is 153 MB, above GitHub's single-file limit, so it is deliberately not
tracked. All 139 content-sealed checkpoint JSON files are tracked, and the bound atlas module
reconstructs and validates the identical assembled file from them without rerunning search.

The search exhausted both declared 33,550,336-formula feature-pair spaces and evaluated the
declared 32,899,328-formula creative prefix for every galaxy. GPU screening was followed by
authoritative CPU-FP64 replay. Every whole-galaxy checkpoint is bound to the config, source,
test, and passed G1 pilot receipt.

## What the counterexample says

The best NGC2955 candidate has aggregate held-out chi-square
`3.905651530364e+01`, passes all five aggregate checks, and passes 24 of 25 fold-level
obligations. It misses only the empirical-RAR ceiling on fold 3: its chi-square is
`4.321958590573e+00`, while the frozen five-point RAR ceiling is
`6.517210311664e-01`.

This rules out a simple lack of aggregate fit quality. Scale-invariant ridge tests did not
repair the fold. Continuous refinement of the winning coarse RBF cell also failed, showing
that resolution within the current Newtonian-base feature-pair family is not the main gap.
The next G1 version must introduce and predeclare a distinct family while retaining NGC2955
and the full v1 failure ledger as its predecessor evidence.

## Claim boundary

This result is a per-galaxy diagnostic search, not a universal galaxy law, an alternative to
general relativity, evidence against dark matter, or a historical novelty result. A local
formula can contain two coefficients fitted separately inside each galaxy's training folds.
G2 remains locked until a separately sealed G1 repair achieves 139/139 without weakening any
v1 threshold or opening confirmation data.
