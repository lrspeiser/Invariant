# Gravity roadmap Item 51: GPU screening result

## Outcome

**OPERATIONAL PASS; CROSS-SCALE SCIENTIFIC LEAD NOT DEMONSTRATED.**

Invariant measured the real end-to-end evaluator rate on the local RTX 5090 and then
executed the complete frozen 67,108,864-ordinal schedule. The GPU screen was fast,
deterministic, response-blind at candidate construction, and numerically consistent with
the CPU. Its selected formula did not beat the strongest earlier Item 45 formula across
galaxy lenses and galaxy clusters.

The selected formula and every mismatch remain in the evidence archive. This retrospective
screen does not prune that formula, its ingredients, or any wider equation family.

## What was actually searched

- Full addressable Item 49 grammar: **6,496,138,035,200 ordinals**.
- Frozen Item 51 schedule: **67,108,864 raw ordinals**.
- Fraction of the grammar scheduled: about **0.00103%**.
- Physically admitted and outcome-scored: **5,505,024 candidates**.
- Exact canonical symbolic classes: **5,505,024**.
- Useful candidate-point-fold evaluations: **3,082,813,440**.
- Sealed confirmation rows: zero.
- Paid model calls: zero.

The affine schedule is a collision-free prefix of a full permutation because its stride is
coprime to the grammar size. It spans the ordinal range but is not a cryptographic or
uniformly random sampler. The candidate manifest was generated and committed before the
outcome screen.

This is 32 times Item 49's combined raw schedule. It is still not a trillion-formula run,
and it does not exhaust the 6.496-trillion addressable grammar.

## Measured RTX 5090 throughput

The synchronized benchmark included ordinal generation, physical filtering, formula
evaluation on 112 predictor rows, and six selection-loss columns.

| Raw batch | Admitted | Median seconds | Raw ordinals/s | Useful point-fold evaluations/s |
|---:|---:|---:|---:|---:|
| 65,536 | 5,376 | 0.02908 | 2.25 million | 0.104 billion |
| 262,144 | 21,504 | 0.01798 | 14.58 million | 0.670 billion |
| 1,048,576 | 86,016 | 0.03372 | **31.10 million** | **1.428 billion** |

The complete campaign ran in **2.393 seconds**, averaging 28.04 million raw ordinals per
second and 1.288 billion useful candidate-point-fold evaluations per second. These are
local measurements for this compact 112-row evaluator, not a general sustained performance
claim for arbitrary field equations, image-plane lensing, simulations, or trillion-scale
searches.

## Empirical result

Lower balanced loss is better. The score gives equal weight to 28 S4TM galaxy lenses and
20 CLASH clusters.

| Method | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Item 45 universal interaction | **0.76148** | 0.18782 | **1.33514** |
| Item 47 operator generator | 0.77891 | **0.16351** | 1.39431 |
| Item 46 dimensionless generator | 0.87009 | 0.24753 | 1.49265 |
| Item 44 scale hierarchy | 0.91598 | 0.16644 | 1.66552 |
| Item 49 pseudorandom program | 1.10118 | 0.17467 | 2.02769 |
| **Item 51 GPU stream** | **1.15351** | **0.18545** | **2.12156** |
| Ordinary ridge | 1.87782 | 0.26047 | 3.49517 |
| Baryonic Newton | 67.65046 | 0.91086 | 134.39006 |

The GPU winner was 51.48% worse than Item 45 in balanced cross-validation and 4.75% worse
than Item 49. It was about 1.26% better than Item 45 on S4TM alone, but about 58.9% worse on
CLASH. That is an interesting galaxy-scale niche, not evidence of one cross-scale law.

The paired sign-flip result was `p = 0.02905`, but the significant direction favored Item
45. Leave-one-object, trimmed, and all four baryonic-mass shift tests did not reverse the
cross-scale conclusion. Four of five folds selected the same top-level operator, so the
weighted-difference motif is structurally repeatable even though its prediction is not
better overall.

There were 31 raw object-level counterexamples relative to Item 45, 26 of which retained
their direction under the relevant frozen mass shift. The executable policy classified
this as `QUALITY_LIMITED_EVIDENCE_RETAINED`: no terminal rejection and no family pruning.

## What the selected formula says

The full-data winner combines:

- a dimensionless variable involving baryonic size, cosmic expansion, an auxiliary scale,
  acceleration, and a time proxy; with
- an oscillatory transform of local acceleration.

It subtracts four times the oscillatory local term from the dimensionless scale term, then
applies the same universal low-acceleration response envelope used by the frozen grammar.
In plain language, it proposes that a galaxy's response may depend on an interaction
between its scale/history and a repeating acceleration pattern.

That expression is a generated cross-mechanism combination. It has not been derived from
an action, adjudicated as historically novel, or shown to be a viable theory of gravity.
Its weak transfer to clusters is the main current scientific limitation.

## What Item 51 establishes

Item 51 establishes that the 5090 can screen millions of compact formula programs rapidly,
with exact raw and symbolic counts and CPU cross-checks. It also shows that simply searching
32 times more of this grammar does not automatically produce a better universal law. Search
coverage, mechanism diversity, validation design, and realistic observables matter more
than raw candidate count alone.

The result does not establish an alternative to general relativity, eliminate dark matter,
or independently confirm any formula.

## Reproduction

Recorded receipts:

- `runs/gravity/roadmap/item-51-gpu-screening-v1.json`
- `runs/gravity/roadmap/item-51-gpu-screening-v1-source/preflight-manifest.json`
- `runs/gravity/roadmap/item-51-gpu-screening-v1-source/candidate-manifest.json`
- `runs/gravity/roadmap/item-51-gpu-screening-v1-source/throughput-benchmark.json`
- `runs/gravity/roadmap/item-51-gpu-screening-v1-source/gpu-screen-result.json`
- `runs/gravity/roadmap/item-51-gpu-screening-v1-source/joint-evaluation-result.json`

Replay all deterministic transformations and verify the timed GPU receipts without
rerunning the campaign:

```powershell
python -m sigma_theory_compiler.gravity_item51_gpu_screening replay
```

The next task is Item 52: turn exact failed regions and their failure causes into a
queryable failure-space database while preserving uncertainty and avoiding global family
pruning from finite empirical data.
