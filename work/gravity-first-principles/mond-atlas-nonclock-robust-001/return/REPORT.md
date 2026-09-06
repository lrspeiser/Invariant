# Broader continuous return-family test

This post-hoc development run expands the earlier coarse grids. It does not
exhaust all formulas or turn already examined galaxies into fresh confirmation.
All 102 historical eligible galaxies and 2,212 radii were retained. No reserved
archive member bodies were opened. The fit used the original three seeds and
five whole-galaxy folds, plus two population-transfer directions.

## What changed

Global stellar mass factor now ranges continuously from 0.8 to 2.0. The extra
field length can interpolate continuously between disk scale Rd and
`rM=sqrt(GM/A0)`. Strength, core mixture and outer cutoff were also fitted within
the frozen bounds. Every training fit used three starts; extra-field models
also included explicit zero-amplitude candidates optimized over mass factor.
The adjusted MOND control received continuous mass and acceleration-scale fits.

The finite_mix and truncated_point_kernel families each have five global fitted
parameters; adjusted MOND has two. Finite_flat_bridge has four and is a
generalized cored rational force shape overlapping the cored clock family.
It is an outer-transition repair/control, not a separate demonstrated physical
mechanism. Static radial observations cannot distinguish different stories
attached to the same potential. These are radial models, not full 3D convolution.

## Results across whole-galaxy folds

Primary metric is equal-galaxy mean squared log10-speed error. Percentages below
are MSE improvements; positive means better. They are not significance levels.

| Formula | RMSE dex | Gain vs fixed MOND | Gain vs continuously adjusted MOND |
|---|---:|---:|---:|
| Fixed MOND | 0.10667 | 0% | -7.97% |
| Adjusted MOND | 0.10265 | 7.39% | 0% |
| Finite mixture | 0.10240 | 7.85% | 0.50% |
| Truncated point kernel | 0.10566 | 1.88% | -5.94% |
| Finite flat bridge / clock-overlap control | 0.10031 | 11.56% | 4.51% |

The finite mixture's small advantage is not consistent across seeds: better in
two, worse in one. Its flexibility largely catches up with the adjusted control.
The bridge improves all three seeds, but its additional shape flexibility and
post-hoc origin remain relevant. A winning development fit is not a causal result.

## Population transfer is more revealing

The gas-rich proxy uses fixed nominal HI and luminosity, not fitted mass factors:
`1.33*MHI/(1.33*MHI+0.5*L3.6)>=0.5`. This gives 44 gas-rich and 58 stellar-rich
galaxies. A family is fitted only on its donor population and predicts the other.

| Formula | Train gas-rich, predict stellar-rich: RMSE dex | Train stellar-rich, predict gas-rich: RMSE dex |
|---|---:|---:|
| Fixed MOND | 0.09502 | 0.12031 |
| Adjusted MOND | 0.09359 | 0.11187 |
| Finite mixture | 0.10305 | 0.11205 |
| Truncated point kernel | 0.09941 | 0.11401 |
| Finite flat bridge / clock-overlap control | 0.09341 | 0.10852 |

Finite mixture becomes about 21% worse in MSE than adjusted MOND when trained
on gas-rich systems and transferred to stellar-rich systems. In the reverse
direction it is approximately tied, but slightly worse. This argues against
claiming a robust new universal return rule from its small reshuffled-fold gain.
The bridge is approximately tied on gas-to-stellar transfer and improves about
5.90% on stellar-to-gas transfer. The point kernel loses to the adjusted control
in both directions.

## Patterns worth understanding

- **A softer center:** finite-mixture q ranges 0.949–0.999 across folds, close to
  the cored `x/(1+x)^3` component. Its inner signed speed overprediction falls
  to about +0.018 dex, compared with +0.044 dex for adjusted MOND and +0.064 dex
  for fixed MOND. The bridge similarly reaches about +0.020 dex. This suggests
  investigating inner radial shape, but mass, bulge and kinematic uncertainties
  can also produce this pattern.
- **A mixed length scale:** finite-mixture t ranges 0.625–0.718, favoring a length
  depending on both Rd and rM rather than simply choosing either. This is an
  empirical global relation under the current mass proxy, not proof of a relay.
- **The point kernel wants more amplitude:** A reaches its upper bound of 100
  in all 15 folds. The experiment is bounded and does not rule out every larger
  amplitude/length combination. It shows this tested range does not beat the
  adjusted control.
- **The cutoff is not measured:** for the point kernel, all catalog radii lie
  below 6.4% of the selected cutoff distance. Different cutoff values therefore
  yield the same sampled branch. Do not interpret its selected C as detected
  halo extent. The bridge pushes C to its upper bound of 100 in all folds,
  delaying the outer decline; a finite termination is not established here.

The radial-bias table retains inner and outer results for every seed and family.
Smaller mean bias does not automatically mean lower scatter at all radii.

## Checks and limits

Four pre-access tests passed: zero-amplitude behavior and units, independent
density integration, potential-gradient/positive-enclosed-mass/outer-limit
checks, and planted fitting with training-only isolation. A sharp-cutoff
quadrature failure and its pre-access repair are preserved separately. No
scientific threshold or formula was changed in that repair.

All 315 fold optimizer attempts completed successfully; all population-transfer
attempts also succeeded. Starts, selected parameters, boundary flags, full held
predictions, per-galaxy metrics and source bindings are saved. Independent direct
formula replay matched 33,180 fold predictions to 2.78e-15 dex, 11,060 transfer
predictions to 1.78e-15 dex, and every recorded training loss to 1.39e-17. Selection
was verified against the minimum successful training loss, not the held result.

The current data use fixed published distance and inclination, a simplified
stellar mass proxy that does not separately model bulge luminosity, no explicit
molecular component, and no radial covariance model. No time-energy exchange,
absorption, 3D mass response, lensing, cluster or Solar System conclusion follows.

Next directions supported by this run: investigate the softer central response
with improved baryonic/kinematic controls; explain the gas-to-stellar transfer
failure before expanding model claims; and keep the bridge as an overlapping
shape control while deriving a genuinely predictive source-based rule. More
parameter freedom alone has not identified the proposed mechanism.
