# Run BS — baryons and gravity on the same cells, and a ninth artefact caught

Lane `work/wellnet-2026-09/clusterjoint/`.

## What was built

The first dataset in this programme where a direction-dependent law could be
distinguished from one that is not.

**52 clusters** from the OPEN half of the gold-cluster holdout that carry both
channels. Each is cut into **5 radial bins × 8 azimuthal sectors**, and every
cell gets baryons on one side and gravity on the other, at the same place on the
sky:

- **gravity** — tangential reduced shear from DECADE per-source shapes, with its
  own metacalibration response per cell, from `clustershear/extract.py` unchanged
- **baryons** — Chandra ACIS photons, 91 observations, 3,116 ks, surface
  brightness and hardness, both with their cluster's own radial median removed

2,080 rows; **1,201 usable** after requiring shear, Chandra coverage and a
measurable asymmetry. Weighted mean g_t = 0.0099 across them.

Nothing is azimuthally averaged anywhere in the chain. That is the point: every
previous cluster input to this programme was a radial profile or a stacked mean,
and both are azimuthally symmetric by construction, so the question could not be
posed at all.

## The question, and the answer

> Does knowing how much brighter or hotter one **direction** is than the average
> at the same radius improve the prediction of the lensing signal there, beyond
> what radius alone already tells you?

Two nested gradient-boosted models, cross-validated **by cluster** — never by
row, because cells from one cluster are not independent and a row-wise split
would let the model memorise a cluster in training and be graded on it in test.

| model | features | R² |
|---|---|---|
| radial | R, z, kT | −0.0085 |
| full | + sb_asym, hardness_asym | +0.0061 |
| | **gain** | **+0.0146** |

**The null that matters** permutes the azimuthal index *within each cluster and
radial bin*. Every radial quantity is untouched, every marginal is identical, and
only the association between a direction's baryons and that direction's gravity
is destroyed. A purely radial model scores identically under it; only a genuinely
directional signal beats it.

**Result: gain +0.0146 against a null of −0.0091 ± 0.0144 → z = +1.64, p = 0.055.
No directional information detectable at this depth.**

## The ninth artefact, and it was mine

The first run used **40** permutations and reported:

    null -0.0165 +- 0.0095   ->   z = +2.67,  p = 0.000
    VERDICT: the lensing cells DO see the baryon asymmetry

Raising it to **200** permutations gave:

    null -0.0091 +- 0.0144   ->   z = +1.64,  p = 0.055
    VERDICT: no directional information detectable at this depth

Nothing about the data changed. The null distribution is simply wider than 40
draws could resolve, and the undersized null manufactured a 2.7σ detection out of
a 1.6σ fluctuation. Had it been written up at 40 permutations it would have been
this programme's ninth shared-denominator artefact, and it would have looked as
convincing as the first eight.

The standing rule that caught it is the programme's own: **size your own test
before trusting it.**

## Two controls that did pass, and are worth keeping

Both were run before the permutation count was raised, and both behaved
correctly, which is why the failure above is attributable to the null size alone
and not to the pipeline.

| control | expectation | result |
|---|---|---|
| `n_phot` alone as the "asymmetry" feature | it is a brightness/exposure proxy, not an asymmetry — should carry nothing | gain −0.0002, z = +0.63, p = 0.23 ✓ |
| target swapped to `g_x` | lensing cannot produce a parity-odd signal — must be null | gain −0.0146, z = −0.28, p = 0.63 ✓ |

`n_phot` is excluded from the feature set as a result, so the measured gain
cannot be attributed to it. The `g_x` control rules out the cell geometry or a
shear systematic producing a spurious association.

## What this does and does not say

**It does not say clusters are azimuthally symmetric.** They visibly are not —
see `clusterxray/cluster_xray_maps.png`. It says the *lensing cells at this
depth* cannot resolve the association between baryon asymmetry and gravitational
asymmetry.

The reason is signal-to-noise, and it is measurable: the **median S/N per cell is
0.78**. Each cell holds a few hundred background galaxies, and shape noise
swamps the per-cell shear. The asymmetry is a second-order effect being asked for
from a first-order-marginal measurement.

**What would change it, in order of leverage:**

1. **More sources per cell.** DECADE gives 5,000–20,000 sources per *cluster*;
   split 40 ways that is a few hundred per cell. Deeper shear over the same
   clusters is the single biggest lever.
2. **Fewer, larger cells.** 5×8 was chosen before seeing any result. 3×4 would
   quadruple the per-cell counts at the cost of angular resolution — but that
   choice must now be pre-registered, because it has become a fork the data can
   see.
3. **The 304 sealed clusters**, once and only once, if a directional law is ever
   pre-registered. They are untouched: 0 tokens spent.

## Provenance

Sealed half never queried. `loader.assert_not_sealed` checks the target list in
`shear_cells.py` and again in `learn.py`. 304 sealed, 0 tokens spent, seal tests
13/13.
