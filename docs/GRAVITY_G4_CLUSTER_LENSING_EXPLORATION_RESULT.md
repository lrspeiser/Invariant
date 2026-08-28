# G4 cluster-lensing exploration result

## Outcome

The exploratory cross-scale test is sealed in
`runs/gravity/g4/cluster-lensing-exploration-v7.json`. It gives a useful split result:

- the unchanged spherical `D=3` projection of the v6 action is rejected as a useful description of
  this cluster-lensing diagnostic;
- the earlier cross-scale mechanism **shape** transfers unusually well and is selected in every
  whole-cluster holdout fold;
- the required coefficient is `beta=2`, four times the frozen galaxy-parent value `beta=0.5`, so
  this is not one universal galaxy-to-cluster law;
- no direct lensing, covariant-theory, historical-novelty, or alternative-to-GR claim passes.

The result is therefore
`BLOCK_CROSS_SCALE_ACTION_CLUSTER_LENSING_EXPLORATION`, with the structural transfer retained as a
specific lead rather than pruned.

## Real source and evidence boundary

The source is the 84-point, 20-cluster table from Tian et al.,
[The Radial Acceleration Relation in CLASH Galaxy Clusters](https://arxiv.org/abs/2001.08340),
published as [VizieR J/ApJ/896/70](https://vizier.cds.unistra.fr/viz-bin/VizieR-3?-source=J/ApJ/896/70/fig2).
It combines X-ray baryonic profiles with strong-lensing, weak-lensing shear, and magnification
information from CLASH.

This is real cluster-lensing evidence, but it is not a direct lensing likelihood. The paper converts
the lensing measurements to spherical total-acceleration profiles using NFW posteriors. The table
also applies empirical stellar corrections to the gas baryons and does not publish the full radial
covariance. Consequently, v7 is explicitly a model-dependent diagnostic. The frozen direct-image
and direct-shear readiness gates remain unchanged and closed.

The exploratory branch used a declared no-slip closure: photons were assumed to respond to the same
effective radial potential as nonrelativistic tracers. The v6 action does not derive that closure.

## What was tested

The run evaluated:

- 20 entire clusters and 84 radial points;
- the unchanged `D=3`, `beta=1/3`, `ell=1/6` v6 action with zero cluster-fitted parameters;
- the unchanged v5 galaxy parent at `beta=0.5`, `ell=0.25`;
- 184 creative mechanisms from nine source-compatible v5 lanes plus one exact known-RAR pipeline
  control;
- 13 universal coefficient values per creative mechanism, or 2,392 cells per selection;
- five whole-cluster outer folds plus one disclosed all-data descriptive selection;
- 1,205,568 candidate-point score evaluations.

Cluster identity and held-out cluster targets were unavailable to each formula. Positivity was
checked from baryonic inputs across all points before target scoring. The exact known RAR control was
recovered to a maximum acceleration error of `2.58e-26 m/s^2`.

## Measured results

All scores below use the published uncertainty in `log10(gtot)`. Smaller is better. The source does
not provide full covariance, and the primary score does not propagate `gbar` uncertainty, so the
absolute chi-square values are descriptive rather than formal likelihoods.

| Prediction | Cluster parameters fit | Chi-square | RMSE (dex) | Within 2 sigma |
|---|---:|---:|---:|---:|
| Newtonian baryons | 0 | 12,037.259 | 0.884 | 0/84 |
| Galaxy RAR, `g_dagger=1.2e-10 m/s^2` | 0 | 3,642.536 | 0.508 | 8/84 |
| Frozen v5 galaxy parent, `beta=0.5` | 0 | 3,127.772 | 0.465 | 11/84 |
| Fixed spherical v6 action, `D=3` | 0 | 4,663.872 | 0.563 | 4/84 |
| Whole-cluster-held-out mechanism transfer | one universal training-fold selection | **144.151** | **0.116** | **73/84** |
| Published cluster low-acceleration relation | published same-sample comparator | 137.290 | 0.110 | 74/84 |

The observed lensing-inferred acceleration is a median **3.58 times** the fixed `D=3` action
prediction. The action improves on Newtonian baryons but is worse than the ordinary galaxy RAR. The
post-v5 rule `beta=1/D`, which helped explain the disk coefficient internally, therefore moves in
the wrong direction at cluster scale.

## The transferred structural lead

Every outer fold selected the same source-compatible construction:

```text
q = (g_bar/g_dagger) / ((g_bar/g_dagger) + 0.1)
chi = I_in,ell=0.25[q]
psi = I_sym,ell=0.25[q]

g = g_bar + beta (g_bar chi + g_dagger psi).
```

The all-data descriptive selection is the same cell. The galaxy parent has `beta=0.5`; every
cluster fold selects `beta=2`. Thus the cluster result preserves the same occupancy transition,
interior-directed response, symmetric screened response, and radial scale, while demanding four
times the response strength.

That is a much narrower and more useful clue than “modified gravity works”: the structural operator
may be worth deriving further, but its current dimensional closure is falsified by this diagnostic.
Possible explanations such as support dimension, pressure geometry, boundary conditions, or a
universal environment field must predict the factor without seeing the lensing target. Treating
`beta=2` as a new cluster constant would merely hide a per-phenomenon patch and is prohibited.

## What this does and does not establish

It establishes that the engine can transfer a pre-existing, baryon-only formula grammar to whole
held-out real clusters, retain a failed fixed theory, and isolate the exact structural part that
survives versus the coefficient that does not.

It does not establish that the selected construction is historically new, that it explains direct
lensing images or shear, or that it is an alternative to GR. The target profile inherits an NFW
conversion, the same dataset selected the cluster coefficient, and the action lacks a covariant
metric completion and gravitational-slip prediction.

## Next falsifiable steps

1. Replace `beta=1/D` with candidate coefficient rules derived from the action, pressure support,
   geometry, or boundary conditions. A rule must predict both `0.5` for disks and the cluster
   response without cluster target fitting.
2. Use the public X-COP density, temperature, and SZ-pressure profiles to forward-predict direct
   thermodynamic observables. Hydrostatic acceleration may be a diagnostic, never target truth.
3. Derive two metric potentials from a covariant completion. Only then score the same unchanged
   field equations against CLASH image pixels, shapes, arcs, parities, and time delays.
4. Freeze the resulting coefficient rule before opening a separate cluster sample. A repeated
   `beta=2` on CLASH alone is development evidence, not confirmation.
