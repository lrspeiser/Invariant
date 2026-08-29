# Item 42: matter-geometry feedback result

## Bottom line

Item 42 completed a fresh, response-blind search for nonlinear feedback between baryonic
organization and an effective weak-field geometry. It generated 262,144 fixed-point formulas in
four equally sized mechanism niches. After generic physical checks and real-profile convergence
checks, 142,450 formulas were evaluated over 47,293,400 candidate-point-fold combinations on an
RTX 5090. Five confirmation galaxies remain sealed.

The result is not a gravity discovery, but it contains a retained cross-scale clue. The active
feedback term improved over a matched no-feedback formula on both fresh galaxy rotation profiles
and an unchanged cluster-lensing diagnostic. The improvements were about 1.6% and 10.5%.
Nevertheless, the formula lost to MOND, the galaxy increment was not statistically persuasive, the
galaxies lack measured stellar counterparts, and the fixed point failed to converge for one cluster.

## What was tested

Response-blind HI shell fractions `b_i` source a radial geometry field through

`K_ij proportional to exp(-|x_i-x_j|/0.2)`.

The search gave equal capacity to four nonlinear source updates: positive reinforcement, screening,
gradient-seeking reorganization, and competing reinforcement/screening. The selected formula came
from the gradient niche. In compact form its fixed point is

`w_i proportional to b_i exp[5 tanh(0.2 |dH/dx|_i)]`

`H = normalize(K w)`

and its motion law is

`u = gbar/a0`

`nu = 1 + u^-0.6 / (1 + u/3) (0.05 + 0.95 H)`

`v_pred = sqrt(nu) vbar`.

The iteration used the executable frozen damping value 0.8. An early non-executable prose sentence
in the config still says 0.5; the result records that documentation inconsistency explicitly. It did
not change the formula used before or after response access.

## Fresh WALLABY galaxy test

Twenty-six real HI profiles with no accepted Legacy DR10 optical match were selected before any
rotation response was read. Twenty-one were exploration objects and five were sealed confirmation
objects. Eleven exploration galaxies passed the frozen kinematic-quality rules, contributing 83
radial points. All ten failures remain recorded and none was replaced.

Seven response radii lay inside the first published HI annulus. The initial evaluator run halted
because the interpolation assigned them zero enclosed gas. The implementation was repaired using
constant central surface density, so `M_HI(<r)` scales as `r^2` inside that annulus. No formula,
galaxy, or response point changed or was removed.

Equal-galaxy held-out losses were:

| Model | Loss | Reading |
|---|---:|---|
| Gas-only MOND/RAR | 3.3306 | best tested control |
| Item 42 feedback candidate | 5.4153 | worse than MOND |
| Matched no-feedback law | 5.5035 | feedback is about 1.6% better |
| Ordinary radial/HI ridge | 6.8630 | feedback is better |
| Gas-only baryonic Newton | 22.9527 | feedback is much better |

The paired sign-flip probability against the strongest control was 0.871, so there is no persuasive
evidence that the candidate beats MOND. Six of eleven galaxies were raw counterexamples relative to
MOND; four remained uncertainty-resolved. Removing the most influential galaxy did not reverse the
negative conclusion.

These galaxies have no accepted optical counterpart in the frozen source. The calculation therefore
uses gas plus helium but not a measured stellar mass. Missing stars may explain some residuals, so
this cannot support a complete-baryon or no-dark-matter claim.

## Unchanged CLASH cluster diagnostic

The galaxy-selected formula was committed and then applied unchanged to 20 CLASH clusters and 84
published radial acceleration points. Enclosed baryonic source weights were reconstructed up to a
cancelling constant from `Mbar(<r) proportional to gbar r^2`.

The high-gain fixed point converged for 19 clusters but not for A2261. On the 19-cluster converged
subset, equal-cluster losses were:

| Model | Loss |
|---|---:|
| MOND/RAR | 38.7365 |
| Item 42 feedback candidate | 42.5957 |
| Matched no-feedback law | 47.6170 |
| Baryonic Newton | 126.6551 |

Feedback improved about 10.5% over no feedback, the same direction as the galaxy result, but remained
about 10.0% worse than MOND. A2261 is preserved as a numerical domain failure, not silently excluded.
Because the exact formula is undefined there, the cluster gate fails regardless of its subset score.
CLASH is also an already exposed, model-dependent proxy rather than fresh confirmation.

## What may be interesting

The same-direction increment over a matched no-feedback law in two very different regimes is the
most interesting Item 42 observation. It is worth a dedicated replication using complete stellar
and gas inventories, a convergence-stable feedback equation, and fresh lensing data. It is not yet
a paper-level finding: the galaxy effect is small and nonsignificant, cluster selection is incomplete,
and the candidate still loses to MOND.

## Why counterexamples do not kill the idea

All galaxy mismatches and the A2261 nonconvergence remain in the evidence record. The executable
policy forbids using one empirical mismatch—or a count by itself—as a kill switch. Uncertainty,
influence, data completeness, numerical domain coverage, and independent unchanged replication all
matter. The exact high-gain formula is not promoted, but a finite test cannot prune the entire
reinforcement, screening, gradient, or competing-feedback families.

## Claim limits

- No dark-matter explanation was excluded.
- No modification of gravity or covariant field equation was established.
- The galaxy baryonic inventory is incomplete because stellar counterparts are missing.
- Historical novelty was not established; curvature-matter coupling and feedback have prior art.
- The cluster transfer is not a direct image likelihood or fresh confirmation.
- Five WALLABY confirmation galaxies remain sealed.
- No paid model calls were made.
