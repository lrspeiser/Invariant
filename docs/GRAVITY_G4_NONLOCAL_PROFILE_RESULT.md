# G4 nonlocal radial-profile construction result

Date: **2026-08-27**

## Decision

`BLOCK_G4_NONLOCAL_PROFILE_CONSTRUCTION`

The sealed receipt is
`runs/gravity/g4/universal-galaxy-law-construction-v3-nonlocal-profile.json`. All **139 SPARC
exploration galaxies and 2,720 published points** were evaluated under whole-galaxy folds. The 35
confirmation galaxies were not accessed and remain locked.

## The tested construction

The formula family augments the frozen empirical RAR with one or two scale-free nonlocal profile
features:

```text
V²(r) = V_RAR²(r) + r g† [C0 + Σm Cm Φm(r)].
```

Each `Φ` is a row-normalized integral in `log r` over a measured baryonic source field. The finite
grammar uses six sources (`log_y`, total surface brightness, gas fraction, bulge fraction, and the
baryonic and brightness log slopes), five symmetric/directional kernel shapes, six log-radius
scales, and both a weighted mean and mean-minus-local representation. Because it depends on
`log(r_j/r_i)`, a common observer-distance rescaling cancels. It uses zero galaxy IDs, halo inputs,
target-derived features, or per-galaxy gravitational parameters.

The complete cascade evaluated all **360 univariate features**, retained the best 24 under the
declared exploration cascade, exhausted their **276 unordered pairs**, and tried 19 shrinkages per
structure: **12,084 formula cells, 1,679,676 candidate-galaxy evaluations, and 32,868,480
candidate-point evaluations**.

## Best formula and measured improvement

The selected three-constant formula combines:

- the exterior exponential (`ell = 0.5`) mean-minus-local transform of total log surface
  brightness; and
- the interior exponential (`ell = 1`) weighted mean of gas fraction.

Its final all-exploration coefficients are `C0 = -0.00563994`, `C1 = -0.0482179`, and
`C2 = -0.0571318`; the selection shrinkage was `0.75`. Its whole-galaxy projected chi-square was
**120,016.785**, compared with **130,714.689** for RAR: an **8.18% improvement**. This is the best
compact zero-local-parameter exploration result in the current G4 sequence. A direct refit to all
exploration galaxies reached 113,324.843, but that in-sample diagnostic is not the gate score.

The worst target-blind population stratum regressed **7.37%** relative to RAR, inside the allowed
10% limit. Every prediction was positive and finite. The formula therefore passed every G4
obligation except one.

## Why G4 remains blocked

The unchanged NFW-shaped two-parameter-per-galaxy performance ceiling plus slack is
**33,458.807**. The nonlocal formula exceeds it by **86,557.978**. The gap is far larger than the
incremental gains from G4 v1, photometric v2, or adding another nearby feature to this cascade.
Opening confirmation would therefore violate the preregistered protocol.

The origin status is `COMBINATION`: nonlocal kernels, scale-space smoothing, radial profile
summaries, and an RAR correction are known ideas. This run tests a specific mix; it does not prove
historical novelty, derive a three-dimensional causal Green function, or discover an alternative
to general relativity.

## Consequence for the goal ladder

G4 is the terminal blocker for this run. G5's independent-galaxy replication requires a frozen G4
law, G6/G7 require a compatible cluster law and shared weak-field operator, and G8-G12 require
those upstream survivors. None is authorized by this result. The next scientifically distinct G4
attempt must introduce a derived operator or new target-blind observable with a credible mechanism
for closing an order-`10^5` chi-square gap; more terms from the same residual-correction grammar do
not qualify.
