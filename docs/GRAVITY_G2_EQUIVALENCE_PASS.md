# G2 equivalence-collapse PASS

G2 reached `PASS_G2_EQUIVALENCE_COLLAPSE` on 2026-08-27.

## Measured result

- Input: all 8,615 retained G1 survivors across 139 exploration galaxies.
- Exact semantic canonicalization produced 8,609 structural classes, collapsing six duplicate
  representations.
- Adversarial coefficient-span analysis produced 6,326 behavioral classes, collapsing 2,283
  further distinctions that do not change the two-function linear span on the declared design.
- The largest behavioral classes contain 16 retained formulas.
- All assignments are unique and confirmation-evaluator access remains zero.
- The sealed receipt is
  `runs/gravity/g2/gravity-formula-equivalence-classes-v1.json`, content seal
  `5b9683bf9526c62a8eef1640e779d194ed58823e0609d85b1cf49488146ef83c`.

The detector compares the projection of eight deterministic probes onto each formula's
two-component span on 257 adversarial feature points. It is invariant to component order,
nonzero unit rescaling, coefficient renaming, and invertible two-column basis changes while
keeping different base laws and feature normalizations separate.

## Mutation controls

Positive-equivalence maximum errors were approximately:

- component swap: `8.88e-16`
- unequal nonzero rescaling: `1.55e-15`
- invertible triangular change of basis: `1.48e-15`

Negative controls remained clearly distinct:

- a `1e-3` orthogonal component mutation: projection error `4.05e-4`
- changing one feature identity: projection error `1.57`

The largest within-class projection disagreement was `3.20e-14`, below the declared `1e-8`
equivalence tolerance.

## Novelty result

This stage found **no formula that it can defend as structurally unmatched**. Of the 8,609
structural classes, 5,385 are bounded as `KNOWN_FAMILY` and 3,224 as `COMBINATION`. The latter
means a combination of known components, not historical novelty. No external prior-art corpus
was searched, so the receipt cannot promote a class to a novelty claim.

## Claim boundary

G2 proves that the pipeline can avoid counting many cosmetic or numerically redundant formulas
as separate ideas, and its mutation controls show that nearby non-equivalent ideas remain
separate. Behavioral equivalence is bounded to the declared adversarial design and local
two-coefficient semantics. G2 does not yield a universal galaxy law or an alternative to GR.
It authorizes G3 whole-galaxy meta-law testing while the 35 confirmation galaxies remain sealed.
