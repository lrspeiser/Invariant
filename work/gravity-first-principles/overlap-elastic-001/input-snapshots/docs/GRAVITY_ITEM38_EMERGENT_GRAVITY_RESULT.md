# Gravity Item 38 emergent-gravity result

Date: **2026-08-28**

## Decision

`NONPROMOTED_ITEM38_EMERGENT_GRAVITY_LEAD`

The fresh KiDS-1000 exploration produces a bounded positive signal, but neither formal track is
promoted. A Verlinde-like elastic-strain formula with a declared transition extension improves
the full-covariance loss by **14.09%** over the strongest ordinary exploration control and wins on
all three whole mass-bin profiles. The comparison is stable to leaving out any one profile and to
the frozen 5% profile trim.

The same unchanged formula is **1.67% worse** than the flexible ordinary control on the two
predeclared color-bin transfer profiles. On the already exposed, model-dependent CLASH cluster
acceleration diagnostic it is **55.67% worse** than fixed MOND RAR. The exact three-profile sign
test has `p=0.2222`, which is also the smallest possible value under the frozen finite-sample
calculation, so the significance gate cannot pass on this exploration design. The formula is
retained as a lead for a genuinely independent unchanged test; it is not evidence that emergent
gravity, Verlinde's theory, or a new theory has been established.

One empirical mismatch never prunes a formula or formula family. The complete machine-readable
result is `runs/gravity/roadmap/item-38-emergent-gravity-v1.json`.

## Frozen source and sample

The official Brouwer et al. KiDS-1000 radial-acceleration-relation high-level product was selected
because every SPARC curve had already been exposed in repository work. The archive was frozen at
SHA-256 `73adc2d4e848fa0a6a43187f7b5447e749b92eaec4a795b9902b0beaf8a78733` before any member
payload was read. Whole published profiles define every role:

- exploration: Figure 9 isolated stellar-mass bins 1, 2, and 3;
- sealed confirmation: Figure 9 isolated stellar-mass bin 4;
- unchanged transfer: Figure 8 isolated color bins 1 and 2; and
- declared covariance: the corresponding full published covariance matrices.

All five allowed profiles pass the frozen quality rules. The three exploration bins contain 44
positive, log-evaluable points. Mass bin 1 also contains one published negative excess-surface-
density row; it is preserved in the raw receipt and omitted only from logarithmic likelihood under
the predeclared positive-domain rule. The two color profiles contain 30 positive transfer points.
The fourth mass-bin payload remains sealed, with zero accesses.

The first 11 printed rows of the five allowed profiles were inspected after the formula and source
freezes but before the evaluator freeze to determine the file format. No formula was generated or
scored after that inspection. This is disclosed protocol history and is one reason the run remains
exploration rather than independent confirmation.

## Formula space and provenance

Exactly 262,144 raw cells assign 65,536 cells to each mechanism niche:

1. Verlinde elastic strain, a known formula plus a declared transition extension;
2. generalized entropy susceptibility, a combination of known entropy families;
3. a Clausius/entanglement crossover, labeled only as a potentially new synthesis of known
   derivations; and
4. collective information compression, labeled only as a potentially new synthesis whose prior
   art is not excluded.

Finite, positive, bounded, material, monotone, and high-acceleration local-limit gates admit
128,411 cells. They collapse to 122,106 behavioral equivalence classes on the frozen probe grid.
The search also includes fixed baryonic Newton, exact point-mass Verlinde, MOND RAR, and a
cross-fitted cubic B-spline as controls.

The selected full-exploration cell is candidate `54595` in the Verlinde lane. With
`u = g_bar/a0`, it predicts

`g_pred/g_bar = 1 + 2.75 u^(-0.45) [1 + (u/0.01)^(0.5)]^(-1)`.

This is explicitly **not** classified as historically new. Fitting a macroscopic susceptibility
also is not a microscopic derivation.

## Fresh KiDS exploration result

The primary score is a full-covariance standardized log-acceleration loss with equal weight per
whole profile and leave-one-profile-out formula selection.

| Model | Exploration loss | Selected candidate improvement |
|---|---:|---:|
| Baryonic Newton | 481.0918 | +99.70% |
| Fixed point-mass Verlinde | 36.9085 | +96.08% |
| Fixed MOND RAR | 4.5350 | +68.10% |
| Flexible ordinary spline | **1.6838** | **+14.09%** |
| Selected transition formula | **1.4466** | — |

The selected family beats the flexible control on all three profiles. There are zero raw
profile-level counterexamples, leaving any one profile out does not reverse the sign, and the
frozen 5% trim still gives **+10.09%**. The exact paired-profile sign-permutation value is
`p=2/9=0.2222`; with three profiles, stronger finite-sample evidence is impossible under this
test. That is a limitation of the frozen design, not a post-result reason to relax the gate.

## Unchanged color transfer

No parameter, lane, formula, scale, or flexible-control rule is reselected on the two color bins.

| Model | Color-transfer loss | Selected candidate improvement |
|---|---:|---:|
| Baryonic Newton | 900.5636 | +99.66% |
| Fixed point-mass Verlinde | 73.2340 | +95.77% |
| Fixed MOND RAR | 10.5325 | +70.62% |
| Flexible ordinary spline | **3.0435** | **-1.67%** |
| Unchanged transition formula | **3.0944** | — |

The small transfer loss is a negative generalization result, not a global veto. The color stacks
overlap the mass-selected population and are not independent galaxies or a sealed confirmation.

## Unchanged CLASH cluster diagnostic

The formula is also evaluated without reselection on 84 radial points from 20 previously exposed
CLASH clusters. This is a model-dependent acceleration table reconstructed with an NFW-based
analysis; it is not fresh confirmation and is not a direct shear, arc, image, magnification, or
time-delay likelihood.

The candidate's standardized MSE is `65.7629`, compared with `42.2453` for fixed MOND RAR. It is
therefore **55.67% worse** than the strongest fixed cluster comparator, although it remains better
than baryonic Newton, the fixed point-mass Verlinde formula, and the galaxy-trained flexible
spline on this diagnostic. The cluster result blocks promotion but does not kill the formula.

## Compute and controls

The NVIDIA GeForce RTX 5090 evaluated 16,950,252 admitted candidate-point combinations. The main
GPU calculation took 0.927 seconds, and the frozen CPU/GPU comparison has zero absolute
difference. Quality, no-hard-veto, confirmation-seal, broad-exploration, robustness, and all four
exploration-baseline gates pass. The exact-p, unchanged-color-transfer, and cluster-diagnostic
gates fail. No paid API calls or confirmation computations are used.

## Claim boundary and next action

This result says that a simple, predeclared transition extension of a known emergent-gravity
shape describes three fresh KiDS mass stacks better than a cross-fitted smooth ordinary curve.
It does not show a microscopic origin, historical novelty, object-level galaxy prediction, a
universal galaxy/cluster law, or a dark-matter-free alternative to general relativity.

Preserve all four Item 38 formula regions, 122,106 behavioral classes, the selected candidate,
the positive mass-stack result, the color and cluster failures, and the negative ESD row. Keep the
fourth KiDS mass profile sealed unless a separately frozen gate authorizes it. Advance the ordered
roadmap to Item 39, holographic or boundary gravity, on a new response while keeping this exact
candidate eligible for an unchanged independent replication.
