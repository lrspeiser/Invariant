# Gravity roadmap Item 1: effective-dimension result

## Outcome

Item 1 is **inconclusive**, with one useful falsification and one useful warning:

- the existing `beta=1/D_support` closure is rejected by the current galaxy-to-cluster
  diagnostic;
- continuous dimension summaries computed from measured baryonic radial profiles do not
  predict the required response coefficient;
- a nested model selects the binary disk-versus-spherical-cluster support label in all
  five folds, but that label only learns the two population means and has negative
  within-population predictive power;
- the formula `beta=(1/2) 4^(D_support-2)` reproduces `beta=1/2` for disks and `beta=2`
  for clusters, but it was constructed after those answers were known and is retained only
  as a post-target synthesis requiring new-data confirmation.

The sealed evidence is
`runs/gravity/roadmap/item-01-effective-dimension-v1.json` with decision
`INCONCLUSIVE_ITEM1_EFFECTIVE_DIMENSION`. Item 1 is not complete, no alternative to GR is
claimed, and the roadmap advances to measured shape and anisotropy only because that is the
missing dependency needed to make “dimension” more than a two-class label.

## Real-data test

The run used the unchanged cross-scale parent

```text
g = g_bar + beta (g_bar chi + g_dagger psi)
```

on:

- 139 admitted SPARC exploration galaxies and 2,720 rotation-curve points;
- 20 CLASH clusters and 84 published baryon-plus-lensing acceleration points;
- five whole-object outer folds, stratified so every fold held out galaxies and clusters;
- 401 nonnegative coefficient cells per object and 1,124,404 coefficient–data-point
  evaluations;
- six target-blind coefficient model classes, four fixed rules, and one nested selector.

No SPARC confirmation galaxy, direct lensing likelihood, paid model call, or sequential
G6–G8 gate was opened. The CLASH target remains a model-dependent acceleration diagnostic
reconstructed from spherical NFW lensing posteriors, not a direct image or shear likelihood.

For both populations the continuous feature was built without the dynamics target:

```text
M_eq(r) proportional to g_bar(r) r^2
D_M(r) = d log(M_eq) / d log(r)
       = 2 + d log(g_bar) / d log(r).
```

The experiment used the profile slope, local median, and local interquartile range of
`D_M`. For disks this is explicitly a spherical-equivalent baryonic mass slope, not a claim
that a disk literally has that spatial or spacetime dimension.

## Measured result

Smaller chi-square is better. Galaxy and cluster scores use their existing frozen error
contracts, so they are shown separately.

| Rule | Galaxy chi-square | Cluster chi-square | Interpretation |
|---|---:|---:|---|
| Fixed galaxy parent, `beta=0.5` | 138,636.609 | 3,127.772 | Existing development reference |
| `beta=1/D_support` | 138,636.609 | 4,648.493 | Same disk prediction; substantially worse on clusters |
| Nested binary support proxy | 139,209.978 | 153.456 | Transfers the population shift, but not within-population variation |
| Post-target bridge | 138,636.609 | 144.151 | Descriptive reproduction by construction; excluded from admission |
| Known galaxy RAR control | 130,714.689 | 3,642.536 | Known-family comparator, not a candidate derivation |

The measurable profile dimension does not separate the populations: its median is `1.518`
for galaxies and `1.557` for clusters, with strongly overlapping ranges. The strongest of
the tested continuous feature combinations still has balanced coefficient `R^2=-2.839`;
none has positive predictive power in both populations.

The nested selector chooses `linear_support_dimension_proxy` in all five outer folds. It
predicts galaxy coefficients from `0.443` to `0.468` and cluster coefficients from `1.899`
to `2.051`, close to the population means. But its coefficient `R^2` is `-0.010` within
galaxies and `-0.143` within clusters. In plain language: knowing “disk or cluster” tells the
system which average bucket to use, but does not explain why individual systems need their
specific coefficients.

## What was learned

The run narrows the search rather than producing a gravity law:

1. **Rejected family:** inverse support dimension, `beta=1/D`, moves in the wrong direction
   at cluster scale.
2. **Unsupported family:** the radial growth rate of spherical-equivalent baryonic mass is
   not the hidden variable in these data.
3. **Retained clue:** geometry class tracks the cross-scale shift, so real flattening,
   triaxiality, boundary geometry, pressure anisotropy, or another correlated variable is
   worth testing.
4. **Pruning protection:** the exact two-class bridge remains in the idea database, labeled
   post-target rather than discarded or promoted.
5. **Key counterexample:** a formula can look excellent across held-out objects while merely
   learning a population proxy; positive within-population prediction is now a required
   guardrail.

## Required next test

Item 2 must replace the binary label with continuously measured baryonic shape and
anisotropy: disk thickness/flattening and bulge fraction for galaxies; gas and stellar axis
ratios, ellipticity, triaxiality, disturbance, and pressure anisotropy for clusters. The test
must include an intermediate or filamentary geometry population, predict variation within
each population, and freeze its coefficient rule before an independent galaxy or cluster
sample is opened.
