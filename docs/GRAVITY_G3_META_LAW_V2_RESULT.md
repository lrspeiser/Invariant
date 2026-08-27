# G3 fixed-shrinkage meta-law result

Date: **2026-08-27**

## Result

G3-v2 passed its exploration gate. The sealed receipt is
`runs/gravity/g3/galaxy-formula-meta-law-v2.json`.

- Decision: `PASS_G3_FIXED_SHRINKAGE_META_LAW`
- Whole-galaxy predictions: **139/139 galaxies, 2,720/2,720 points**
- Invalid direct predictions: **0**
- Confirmation evaluator accesses: **0**
- Projected G2 classes used: **98**
- Exact matches to each held-out galaxy's evaluator-only best G1 class: **0**
- Projected meta-law chi-square: **123,472.313**
- Direct ensemble chi-square: **122,436.722**
- Frozen empirical RAR chi-square: **130,714.689**
- Newtonian-baryon chi-square: **1,697,326.398**

The projected formula improves aggregate chi-square by **5.54%** relative to the empirical
RAR comparator, **8.56%** relative to the constant-residual comparator, **89.35%** relative
to the nearest-galaxy formula transfer, and **92.73%** relative to Newtonian baryons. Every
gain exceeds the predeclared 0.5% G3 threshold.

## What passed

Five folds held out entire galaxies. The model received 52 baryonic local and galaxy-summary
features, but never galaxy identity, held-out velocity, velocity error, held-out G1 class, or
held-out fit coefficients. Each learned curve was projected into a G2 structural class that
had at least one member in that fold's training galaxies; projection coefficients were fitted
to the model prediction rather than to observed velocity.

This establishes a bounded result: baryonic structure carries enough transferable information
to improve a target-blind prediction of the exploration galaxies' rotation curves and local
formula behavior.

## What did not pass

This result is not independent validation. Fixed shrinkage `0.3` was chosen after inspecting
the failed G3-v1 outer-fold diagnostics, and the same exploration folds were reused. The model
is an Extra Trees ensemble around the already known RAR relation. It is neither a compact
first-principles formula nor evidence of a historically novel law. Its generated G2 formulas
still contain two coefficients per galaxy, so they are not the zero-local-constant G4 law.

G3-v2 authorizes G4 construction only. The 35 confirmation galaxies remain unopened.
