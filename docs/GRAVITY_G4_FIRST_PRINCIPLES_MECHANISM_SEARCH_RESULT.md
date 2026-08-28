# G4 first-principles mechanism search result

Date: **2026-08-27**

## Decision

`BLOCK_G4_FIRST_PRINCIPLES_MECHANISM_SEARCH`

The sealed receipt is `runs/gravity/g4/first-principles-mechanism-search-v5.json`. The search used
all **139 SPARC exploration galaxies and 2,720 published rotation-curve points** under five outer
whole-galaxy folds. For each fold, every candidate and its universal coefficient were selected
using only the other galaxies. Every reported galaxy was therefore absent from mechanism selection
for its fold. Confirmation galaxies, clusters, and lensing evaluators were not accessed.

## The ten implemented lanes

The compiler generated typed velocity-squared consequences from:

1. inverse effective-action discovery;
2. baryonic gravitational permittivity;
3. a screened auxiliary focusing field;
4. geometry-directed radial tensor response;
5. occupied-interior/empty-exterior boundary fields;
6. causal gravitational memory;
7. baryon-defined orbital eigenmodes;
8. nonlocal modified inertia;
9. multiscale running gravity; and
10. cross-scale action synthesis.

The frozen grammar contained **280 proposed mechanisms** plus one exact empirical-RAR known-family
control. Thirteen universal coefficient values produced **3,653 cells per selection**. Five outer
training selections plus one final all-exploration selection executed **49,680,800 grid-scoring
point evaluations**. There were no galaxy-specific fitted gravitational constants.

Every structure has an equation IR, velocity-squared dimensional type, target-blind baryonic
source, origin label, and explicit completion status. Poisoning the observed velocities and
uncertainties leaves all 281 mechanism components exactly unchanged.

## Positive control

The compiler recovered the exact empirical RAR rewrite with coefficient **1.0** and chi-square
**130,714.689**. It is labeled `known_family_instance`, cannot count as a creative survivor, and
demonstrates that the grammar and scorer can recover a known baryonic relation without treating it
as a discovery.

## Genuine whole-galaxy holdout result

When the selector was allowed to choose among all 280 proposed mechanisms, it chose four different
cells across the five outer folds. Its concatenated unseen-galaxy chi-square was **193,904.989**,
which is **48.34% worse than RAR**. It improved 63 of 139 galaxies, short of a majority, and several
predefined population strata regressed materially. This is selection instability: attractive
training-population mechanisms did not transfer reliably to unseen galaxies.

The best single stable lane was cross-scale action synthesis. All five folds independently chose

```text
q(r) = (g_bar(r)/g_dagger) / ((g_bar(r)/g_dagger) + 0.1)

V_pred^2(r) = V_bar^2(r)
  + 0.5 [V_bar^2(r) I_in,0.25[q](r)
         + r g_dagger I_sym,0.25[q](r)].
```

It reached chi-square **138,636.609**, only **6.06% worse than RAR**, and was structurally identical
in all five folds. This makes it a reproducible parent for further derivation, not an accepted law.

The complete lane ranking was:

| Lane | Nested unseen-galaxy chi-square | Relative to RAR |
|---|---:|---:|
| Cross-scale action synthesis | 138,636.609 | 6.06% worse |
| Baryonic gravitational permittivity | 155,179.707 | 18.72% worse |
| Auxiliary focusing field | 190,985.904 | 46.11% worse |
| Multiscale running gravity | 198,143.004 | 51.58% worse |
| Vacuum boundary field | 203,391.374 | 55.60% worse |
| Geometry-directed gravity | 301,933.318 | 130.99% worse |
| Nonlocal modified inertia | 694,994.997 | 431.69% worse |
| Inverse action discovery | 843,749.561 | 545.49% worse |
| Orbital-mode resonance | 993,809.486 | 660.29% worse |
| Causal gravitational memory | 1,376,717.118 | 953.22% worse |

The static resonance and memory constructions therefore receive strong counterexamples in this
specific grammar. They remain retained as ideas, but the data do not justify prioritizing these
forms over the stable cross-scale parent.

## Why this is not yet first principles

The implemented equations are effective radial weak-field projections. Dimensional typing and
target blindness pass, but covariance, conservation identities, positive-energy/stability,
causal initial-value formulations, lensing, clusters, Solar-System limits, and gravitational-wave
limits remain pending or locked. The NFW-plus-slack ceiling is **33,458.807**; the overall nested
mechanism pipeline exceeds it by **160,446.182**.

No alternative to GR, completed first-principles theory, or historical novelty was discovered.
The next bounded derivation should begin from the fold-stable cross-scale parent and ask whether a
single auxiliary-field action can derive both its permittivity and screened-source terms. That
derivation must predict the coefficient and scale rather than fitting or enumerating them, and it
must supply conservation and lensing equations before any downstream data are opened.
