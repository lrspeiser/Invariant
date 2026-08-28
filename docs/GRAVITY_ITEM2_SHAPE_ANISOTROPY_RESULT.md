# Gravity roadmap Item 2: shape-and-anisotropy result

## Outcome

The first Item 2 attempt is **inconclusive**. Continuously measured projected shape is a
better cross-scale separator than one global constant, but it does not explain variation
within galaxies or within clusters and is inferior to the binary disk/cluster proxy retained
from Item 1.

The sealed evidence is
`runs/gravity/roadmap/item-02-shape-anisotropy-v1.json` with decision
`INCONCLUSIVE_ITEM2_SHAPE_ANISOTROPY`. This does not establish a new gravity law, a cause,
historical novelty, or an alternative to GR. Item 2 remains open.

## Real-data test

The attempt kept the Item 1 parent unchanged:

```text
g = g_bar + beta (g_bar chi + g_dagger psi).
```

It tested whether target-blind baryonic imaging summaries predict each object's sealed
best-fit development coefficient `beta`:

- 139 SPARC exploration galaxies and 2,720 rotation-curve points;
- 20 CLASH clusters and 84 model-dependent baryon-plus-lensing acceleration points;
- five whole-object outer folds, with 4 clusters and 27 or 28 galaxies held out per fold;
- nine fixed model classes plus target-blind nested selectors, yielding 11 evaluated
  formula classes and 9 distinct prediction classes;
- no SPARC confirmation access, direct-lensing likelihood evaluation, paid model call, or
  G6–G8 advance.

For galaxies, projected axis ratio was `q=cos(i)` from the published SPARC inclination and
concentration was the measured 3.6-micron disk-plus-bulge light inside `0.2 R_last` divided
by the light inside `R_last`. For clusters, `q` was the Chandra X-ray surface-brightness axis
ratio within 500 kpc and concentration was X-ray brightness inside 100 kpc divided by that
inside 500 kpc, from Donahue et al. (2016). An older independent X-ray catalog overlaps 15
targets and correlates with the adopted axis ratios at `r=0.924`.

These are deliberately limited common projections. Galaxy `cos(i)` is orientation-dominated,
not intrinsic thickness, and stellar light is not the same tracer as X-ray emissivity.

## Measured result

The universal shape selector chose `quadratic_projected_axis_ratio` in all five folds. It
beats a single global constant in observational chi-square, but fails the causal guardrails:

| Predictor | Galaxy beta MSE / R² | Cluster beta MSE / R² | Galaxy / cluster chi-square |
|---|---:|---:|---:|
| Global constant | 0.656 / -8.200 | 0.764 / -3.565 | 1,729,099.626 / 589.932 |
| Quadratic projected axis ratio | 0.336 / -3.715 | 0.324 / -0.937 | 488,182.944 / 213.519 |
| Binary support/population proxy | 0.072 / -0.009 | 0.194 / -0.161 | 135,364.543 / 153.348 |
| Support proxy plus shared shape | 0.073 / -0.024 | 0.207 / -0.239 | 139,827.612 / 154.411 |

The negative within-population `R^2` values mean each model is worse than predicting that
population's held-out mean coefficient. Shape also does not improve the population proxy.

The strongest counterexample is the common projected-axis-ratio interval `0.750–0.940`,
which contains 24 galaxies and 16 clusters. There, the shape selector's coefficient MSE is
`1.417` for galaxies and `0.328` for clusters, versus `0.109` and `0.157` for the population
proxy. Thus the apparent full-sample improvement is primarily population separation, not a
shared shape law.

## What was learned

1. **Excluded current family:** global linear/quadratic combinations of projected axis ratio
   and aperture concentration do not generate the required cross-scale coefficient.
2. **Proxy warning reproduced:** geometry can look predictive because the available galaxy
   and cluster shape ranges differ; overlap and within-population tests prevent promotion.
3. **Useful data control:** two independent published CLASH X-ray morphology measurements
   agree strongly enough that simple catalog transcription is not the likely failure cause.
4. **Not yet tested:** intrinsic disk thickness, bars, lopsidedness, two-dimensional stellar
   quadrupoles, pressure anisotropy, and comparable intermediate or filamentary geometries.
5. **Pruning rule:** retain non-global and tensorial shape mechanisms; exclude only the tested
   projected-summary family and its measured regions, not all anisotropic gravity ideas.

## Required next test

Remain on Item 2. Add a target-blind galaxy source with intrinsic-thickness or two-dimensional
bar/lopsidedness/quadrupole measurements, retain the cluster morphology and disturbance
features, and add an intermediate or filamentary geometry population. Preregister a model
that predicts coefficient variation within every population and in feature-overlap regions.
Only after it passes should its unchanged predictions be opened on independent galaxies or a
direct CLASH/X-COP observable likelihood.
