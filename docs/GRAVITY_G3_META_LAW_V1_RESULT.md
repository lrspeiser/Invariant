# G3 whole-galaxy meta-law v1 result

The sealed G3-v1 experiment completed on 2026-08-27 with `BLOCK_G3_META_LAW`.

## Measured result

- All 139 exploration galaxies and 2,720 rotation-curve tracers received exactly one
  outer-fold prediction.
- No held-out galaxy velocity, uncertainty, name, ID hash, G1 class, or fitted coefficient
  entered the model or formula projection.
- The projected meta-law selected 66 training-only G2 classes and generated two coefficients
  for every held-out galaxy.
- It achieved aggregate chi-square `1.305932102208e+05`, versus
  `1.307146893155e+05` for empirical RAR: a gain of only `0.0929%`, below the declared
  `0.5%` minimum.
- It beat the constant-residual baseline by `3.28%`, the nearest-galaxy baseline by `88.74%`,
  and Newtonian baryons by `92.31%`.
- Exact recovery of the evaluator-only best G1 class was 0/139. This is consistent with G2's
  finding that the local atlas contains thousands of degenerate or near-degenerate solution
  classes; it is not treated as a successful class-recovery result.
- Confirmation-evaluator access remained zero.
- The sealed receipt is `runs/gravity/g3/galaxy-formula-meta-law-v1.json`, content seal
  `83510b7a44d84a0e21be2fe3dddfce243bbf07873b2fd59a59ec8dac552f1c7f`.

## Why it blocked

Nested whole-galaxy selection chose residual shrinkage `alpha=0` in three of five outer
folds and `alpha=0.3` in two. The inner folds therefore judged the learned residual too
unstable to use for most outer predictions. Formula projection slightly improved the final
score, but not enough to clear the frozen RAR gain.

The largest individual regression was UGC03580, where projected chi-square increased by
about 1,276 relative to RAR. This and the two outer folds that worsened are retained as G3
counterexamples.

## Next declared repair

Post-result diagnostics on the same exploration folds show that fixed-shrinkage models are
more stable: ExtraTrees with `alpha=0.3` reaches unprojected chi-square near `1.2293e5`, and a
ridge residual model also improves RAR. Any v2 run must disclose that these choices were
informed by the v1 outer-fold results. It can be a reproducibility/model-development result,
not independent confirmation; the 35 confirmation galaxies remain sealed.

## Claim boundary

The v1 model is a phenomenological residual learner around a known RAR relation, not a compact
first-principles law. It does not authorize G4, does not establish novelty, and is not an
alternative to GR.
