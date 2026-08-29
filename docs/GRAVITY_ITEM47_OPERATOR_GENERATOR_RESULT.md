# Gravity roadmap Item 47: operator generator

## Outcome

Item 47 is complete as a **retrospective, non-promoted exterior-profile clue**. A response-blind typed generator compared local, differential, interior-integral, exterior-nonlocal, tensor-derived, and causal-history scalar operators with equal search capacity. Every outer fold selected the same exterior potential kernel and the same radial scale.

The candidate improved S4TM galaxy-lens loss by 12.94% relative to Item 45, but worsened CLASH cluster loss by 4.43%, leaving the balanced score 2.29% worse. Its paired sign-flip result was `p=0.8137`. The decision is:

`NONPROMOTED_ITEM47_OPERATOR_RESULT_RETAINED`

The result is worth preserving because the operator structure repeats across all folds and beats pointwise, scale-free, Item 44, and Item 46 controls. It is not confirmation, a complete nonlocal theory, an alternative to general relativity, a dark-matter-free explanation, or a historically novel operator.

## Frozen operator grammar

Six classes received exactly 16 recipes and 65,536 raw formula cells each:

1. pointwise local operators;
2. Gaussian log-radius differential operators;
3. interior cumulative integrals;
4. exterior nonlocal kernels;
5. axis-ratio tensor-amplitude scalars;
6. causal exponential history kernels under constant-current-state closure.

The frozen run contains:

- scientific-freeze commit: `118824fb123ff40bed6d7bd37c15cee8af745b44`;
- 96 symbolic operator recipes, all behaviorally distinct on the development predictors;
- 393,216 raw formulas and 305,472 admitted formulas;
- 50,912 admitted formulas in every operator class;
- 28 S4TM galaxy lenses and 84 radial points from 20 CLASH clusters;
- 0 of 7 reserved S4TM confirmations opened;
- 0 post-response formulas and 0 paid model calls;
- 201,000,576 RTX 5090 candidate-point-fold evaluations;
- CPU/GPU selected-loss difference `1.22e-15`.

The feature receipt contains no response, target, or uncertainty fields. Replacing all observed values leaves it unchanged. Executable support tests also show that changing only the outer profile cannot alter interior-integral coordinates, while changing only inner structure without changing current enclosed mass cannot alter exterior coordinates.

## Profile and operator boundary

S4TM supplies one observed aperture per lens, so its response-blind radial profile is the analytic projected de Vaucouleurs stellar profile on 128 nodes from `0.001 R_b` to `100 R_b`. CLASH uses its published `g_bar` radial points converted to a monotone enclosed-mass proxy, with no extrapolation past the last published point.

The tensor class uses only

`e = (1-axis_ratio)/(1+axis_ratio)`

from S4TM light shapes and published CLASH X-ray shapes. This is a rotation/reflection-invariant quadrupole amplitude, not an orientation-resolved tensor lensing calculation.

The history class evaluates a causal exponential convolution only under `S(t')=S(t)`. There is no measured baryonic history in these data. Therefore the run tests an age response of a kernel projection, not gravitational memory. No history recipe was selected.

## Selected operator and formula

For a measurement at radius `R`, define the exterior potential-like operator

`I_out(R) = [R/M(<R)] sum_{r'>R} exp(-|ln(r'/R)|/0.6) Delta M(r')/r'`.

Its bounded coordinate is

`H_out = 0.5 + 0.5 I_out/(1+|I_out|)`.

The full-data descriptive formula is

`nu = 1 + 6 u^-0.5/(1+u/10) * (0.05+0.95 H_out)`,

where `u=g_bar/a0`.

In plain language, the rule asks whether baryonic mass outside the measurement radius contributes an additional, distance-weighted potential effect. Nearby outer mass counts more than very distant outer mass. This resembles known nonlocal-kernel ideas and prior repository exterior-profile searches; it is explicitly labeled a known-family combination, not a new theory.

All five folds chose `kernel_outer_potential` with log-radius width `0.6`. Four chose `p=0.55`, `A=6`, `u_t=10`; one chose the neighboring `p=0.5` cell. That is stronger structural stability than Item 46 achieved.

## Comparative results

| Model | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Item 45 universal interaction | **0.76148** | 0.18782 | **1.33514** |
| Item 47 operator generator | 0.77891 | **0.16351** | 1.39431 |
| Item 46 dimensionless generator | 0.87009 | 0.24753 | 1.49265 |
| Item 44 scale hierarchy | 0.91598 | 0.16644 | 1.66552 |
| Matched scale-free search | 1.00812 | 0.18094 | 1.83531 |
| Matched local-operator search | 1.04449 | 0.18953 | 1.89945 |
| Ordinary ridge | 1.87782 | 0.26047 | 3.49517 |

Item 47 improves 25.43% over the matched local-operator search, 22.74% over scale-free search, and 14.96% over Item 44. This is evidence that radial support matters in this grammar. Item 45 remains the strongest balanced result because the exterior operator's galaxy improvement does not transfer fully to clusters.

Under frozen mass-scale alternatives, Item 47 beats Item 45 for S4TM `-0.25 dex` (+3.02%), S4TM `+0.25 dex` (+5.59%), and CLASH `-0.10 dex` (+8.33%). It loses for CLASH `+0.10 dex` (-19.39%). The sensitivity reinforces the need for better baryonic and lens-model uncertainty, not formula deletion.

## Counterexamples and influence

Relative to Item 45, 23 of 48 objects are raw counterexamples: 12 S4TM lenses and 11 CLASH clusters. Seven CLASH clusters and no S4TM lenses remain mismatches under all four mass-scale alternatives.

The ordinary equal-object mean is close enough to zero that removing CLASH cluster `RXJ2129`, the largest unfavorable object for Item 47, changes its sign. The frozen population-balanced score still favors Item 45, and no object is deleted. The policy correctly marks this as single-object-sensitive and quality-limited rather than using `RXJ2129` as a veto or using its removal to promote the operator.

One mismatch is not enough to kill a formula, and neither is a raw count. These data have profile truncation, analytic S4TM profile assumptions, model-derived lens quantities, incomplete baryons, and limited mass-systematic variants. All 96 operators and all mismatch records remain available for later unchanged tests.

## Reproducibility

- Aggregate content SHA-256: `192a5157121fd04894866f175fbb790c513e9069c8265ad675179210808d247e`
- Aggregate file SHA-256: `c32789b85e9adf5db6f93c9d8ab2a639c61fa4054125cce55b08cb12c5ec7193`
- Response-blind feature file SHA-256: `921321784f3a8c725a00a4bdb8772d17d260efbe5ef3186e638d4cdc7e068c33`

Replay with:

`python -m sigma_theory_compiler.gravity_item47_operator_generator replay`
