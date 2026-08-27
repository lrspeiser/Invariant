# G1 atlas repair v2 result

The sealed G1 repair-v2 campaign completed on 2026-08-27 with the decision
`BLOCK_G1_REPAIR`.

## Measured result

- The sole v1 counterexample, NGC2955, received exactly 100,000,000 additional formula
  trials on the NVIDIA GeForce RTX 5090.
- The repair exhaustively tested 33,550,336 symmetric feature-RBF pairs, exhaustively tested
  33,550,336 skew feature-RBF pairs, and tested the declared 32,899,328-formula creative
  prefix.
- No GPU-slack survivor existed, so no candidate could reach authoritative CPU admission.
- The result leaves the cumulative G1 search at 14,000,000,000 candidate-galaxy trials,
  with union coverage still 138/139 and zero confirmation-evaluator accesses.
- The immutable receipt is `runs/gravity/g1-atlas/repair-v2.json`, with content seal
  `a691cdd66fbf72ddb36bd9c814fb567364fa569050e4e540d1a84d37fba30b42`.

## Excluded family

Within the exact declared grids, NGC2955 cannot be repaired by

```text
V_pred^2 = V_RAR^2 + r*(A*phi_1(feature_1) + B*phi_2(feature_2))
```

when each `phi` is a single-feature member of the v3 RBF, skew-RBF, or creative transition
grammar. `A` and `B` were the only local coefficients and were fitted on training radii only.
This is a finite-grammar exclusion, not a proof about every possible RAR correction.

The RAR base is a known empirical/MOND-family construction. The residual shell is labeled
`new_combination_of_known_ideas`; no historical-novelty claim is made.

## Next counterexample-driven family

The nearest creative miss passed folds 0 through 3 but failed the outer fold. A subsequent
exploration diagnostic found that target-blind cross-feature interactions can express the
missing regime separation without adding a third fitted coefficient. That distinct grammar
must be separately declared, tested, and sealed; the repair-v2 result is not altered.
