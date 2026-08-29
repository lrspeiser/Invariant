# Gravity roadmap Item 53: diversity preservation result

## Outcome

**PASS: MORE MECHANISM DIVERSITY WITH NO PREDICTIVE LOSS IN THIS TEST.**

Invariant built two equally sized 64-formula archives from the 878 unique best
representatives in the Item 52 failure-space database:

- a score-only archive containing the 64 lowest full-data training losses; and
- a diversity-preserving archive seeded across operators, ordered source pairs, and
  operator-by-source niches before filling remaining slots by score.

The diverse archive preserved substantially more mechanism niches and produced exactly the
same grouped cross-validated predictions as the score-only archive. The source database was
not modified: all 1,000 region records and all 878 unique representatives remain available.

## Diversity result

| Coverage in 64 slots | Score-only | Diversity preserving |
|---|---:|---:|
| Binary operators | 5 | **8** |
| Ordered source-item pairs | 16 | **16** |
| Operator-by-source niches | 36 | **64** |
| Transform pairs | 33 | **35** |
| Distinct primitives | 97 | **100** |
| Outer-parameter cells | 11 | 11 |

Only 36 formulas appeared in both archives. The diversity view therefore protected 28
different alternatives while still retaining the strongest representatives used by the
predictor.

## Real-data comparison

On the same 28 S4TM exploration galaxy lenses and 20 CLASH clusters, each archive selected
one formula using the training objects in each fold and applied it unchanged to the held-out
objects.

| Archive | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Score-only, 64 slots | 1.15351 | 0.18545 | 2.12156 |
| Diversity preserving, 64 slots | **1.15351** | **0.18545** | **2.12156** |

The predictions were identical: loss ratio `1.000`, improvement `0.0%`, paired sign-flip
`p = 1.0`. In every fold, the best candidate selected from the broader diversity archive
was also present in the score-only archive. This is useful because the additional protected
niches cost nothing on the current predictive diagnostic.

The shared prediction is still the Item 51 result and remains worse cross-scale than Item
45. Item 53 validates archive behavior; it does not improve the gravity law.

## Interpretation

In lay terms, we can keep a much wider gene pool of ideas without lowering the performance
of the current team. If later data or a new constraint makes today's winner less attractive,
the archive still contains alternatives involving operators and source combinations that a
pure leaderboard would have dropped from its 64 visible slots.

This is a protected view, not a deletion mechanism. A low score, one counterexample, or a
counterexample count cannot remove a source candidate or globally prune a niche.

The result is retrospective and categorical diversity is not proof of genuinely independent
physics or historical novelty. No alternative to general relativity or removal of dark
matter is claimed.

## Reproduction

- `runs/gravity/roadmap/item-53-diversity-preservation-v1.json`
- `runs/gravity/roadmap/item-53-diversity-preservation-v1-source/archive-manifest.json`
- `runs/gravity/roadmap/item-53-diversity-preservation-v1-source/archive-evaluation-result.json`

```powershell
python -m sigma_theory_compiler.gravity_item53_diversity_preservation replay
```

The next task is Item 54: detect algebraic, rescaling, renaming, and behavioral equivalence
without erasing distinct lineage or protected archive niches.
