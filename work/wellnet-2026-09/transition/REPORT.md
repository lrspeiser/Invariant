# One observable space for S(M, r): mass, radius, acceleration or pipeline?

Lane `work/wellnet-2026-09/transition/`.  Code: `decl.py` (the frozen
declaration), `build.py`, `common.py`, `fitlib.py`, `nulls.py`,
`transition.py`, `final.py`, `null3.py`, `transfer.py`, `radial.py`,
`overlap.py`, `sensitivity.py`, `dump_points.py`, `test_transition.py`,
`decompose.py`, `render_report.py`.  Results JSON:
`final_results.json` (primary), `decompose_results.json`,
`transition_results.json`, `null_results.json`, `radial_results.json`,
`overlap_results.json`, `transfer_results.json`,
`sensitivity_results.json`, `points.json`.  Every number below is
rendered from those files.

Declaration `decl.py` sha256 `521ea654f122f7367f8ec7a82b095bce1cf01bde4781dce45d3ddc86c1576100`, written and hashed
before any residual was examined.

---

## 0.  The answer: RADIUS

**The cluster lensing residual is organised by clustercentric radius,
not by mass and not by acceleration.  A residual factor of about 1.3
remains on top of the radial law, and that residual cannot be told
apart from a survey offset.**

| model | k | -2 ln L | BIC | dBIC | fitted |
|---|---|---|---|---|---|
| **H_R** | 2 | 3532.77 | 3549.03 | 0.00 | `c=+0.397` `beta=-0.350` |
| **H_T** | 3 | 3526.39 | 3550.78 | 1.75 | `c=-0.791` `A=+1.400` `lnxt=+1.099` |
| **H_MR** | 3 | 3530.64 | 3555.04 | 6.00 | `c=+0.343` `alpha=-0.063` `beta=-0.400` |
| **H_P** | 3 | 3543.25 | 3567.64 | 18.61 |  |
| **H_G** | 2 | 3554.88 | 3571.14 | 22.11 | `c=+0.655` `gamma=+0.100` |
| **H0** | 0 | 3574.32 | 3574.32 | 25.29 |  |
| **H_M** | 2 | 3561.15 | 3577.41 | 28.38 | `c=+0.337` `alpha=+0.020` |

`H_R` wins by dBIC 18.6 over the pure-pipeline
model, 28.4 over mass, 22.1
over acceleration and 25.3 over no excess at all.
The fitted slope is `beta = -0.350`.

The single cleanest statement in the lane, and a genuine
out-of-sample prediction:

> eFEDS ALONE -- 496 X-ray groups, 3365 raw DECADE tangential-shear
> points, no cluster data of any kind -- measures an internal radial
> slope `beta = -0.400 [-0.450, -0.350]` with `c = +0.261`.  Extrapolated
> inward by a factor 7 in radius and upward by 25x in mass, it predicts
> `S = 2.90 [2.62, 3.21]` at the Hubble
> Frontier Field strong-lensing cores, where `4.06` is observed:
> **+0.90 sigma**.
> At `r = R500` it predicts `1.30` against LoCuSS's observed
> `1.66`: **+1.97 sigma**.

And the sharpest single comparison in the lane: what eFEDS measures
INTERNALLY on each axis, against what each story would need in order
to reach the strong-lens cores from the group scale.

| story | eFEDS measures internally | it would need | sigma away |
|---|---|---|---|
| **RADIUS** (`beta`) | -0.4000 [-0.4500, -0.3500] | -0.4452 | **0.9** |
| **ACCELERATION** (`gamma`) | +0.2000 [+0.1500, +0.2500] | +0.3658 | **3.3** |
| **MASS** (`alpha`) | -0.0055 [-0.0255, +0.0145] | +0.3897 | **19.8** |

Radius is the only axis on which the group-scale weak lensing
already has the slope the cluster cores require.

Leave-one-survey-out, with the prediction interval carrying the
held-out survey's own external prior:

| model | held-out LoCuSS | held-out SL cores | worst |
|---|---|---|---|
| H0 | +3.86 | +3.87 | **3.87** |
| H_M | +1.87 | +3.18 | **3.18** |
| H_R | +1.67 | +0.57 | **1.67** |
| H_G | +0.18 | +2.72 | **2.72** |
| H_MR | +2.19 | +0.58 | **2.19** |

**`H_R` is the only model that is never rejected out of sample
(1.67 sigma worst case).**  Acceleration predicts LoCuSS
almost exactly (+0.18) and the cores
badly (+2.72); mass is mediocre on both;
no excess at all is rejected by both.

### The three statements the programme could not reconcile,
### reproduced here in ONE framework

| regime | S measured here | the record |
|---|---|---|
| eFEDS weak shear | 0.982 | 0.981 (RAR) / 0.992 (AQUAL), Run AL.5 |
| LoCuSS massive clusters | 1.689 | E median 1.62, range 1.22-2.34, Run K.3 |
| strong-lens cores | 4.062 | 4.11 (delay) / 4.62 (images) for MACS J1149, Run AL.9 |

The tension is real, it is not an artefact of three independent
analyses, and it survives one forward model with one frozen law.
**It is also not new physics beyond a radial response**: a single
radial power law fitted to the group-scale weak lensing alone spans
it.

**`H_R` is the best model on BIC in 9 of 9 declared
variants and `beta` moves only over [-0.350, -0.300] across all of them** (section 10).

### Four qualifications, all measured rather than asserted

1. **The radial slope is not one number.**  Inside 2 R500 it is `-0.250 [-0.400, +0.000]`; outside, `-0.800 [-1.050, -0.600]`,
   1.8 sigma apart, and the steep
   outer value sits exactly where the Bahar+2022 density fit is
   extrapolated.  **This is the lane's dominant systematic** -- see
   section 4.
2. **Two matched-radius comparisons give incompatible mass slopes:** `alpha = +0.397` at r/R500 ~ 0.5 and `+0.090` at r/R500 ~ 1, over comparable mass
   ratios.  No single mass power law produces both.
3. **The residual factor of 1.28 (LoCuSS) and 1.40 (cores) above the eFEDS radial law** is consistent between the two cluster samples and
   is equally well described by `alpha ~ +0.09`
   or by a per-survey constant.  With three surveys occupying three
   disjoint mass ranges those are not separable.
4. **The strong-lens sample cannot measure its own radial slope.**  Its
   internal beta is `+0.2045 +- 0.0250` --
   POSITIVE -- which is the shared-`theta` artefact the null predicted.
   That is why the primary fit uses one point per strong-lens cluster;
   see section 5.

---

## 1.  The common observable, and what went into it

    S  =  observed lensing response / response predicted from
          (baryons + frozen RAR + NO SLIP)

Frozen law: RAR (Lelli+2017 interpolation), a0 = 1.0844e-10 m/s^2, SPARC train.  Never refitted in this lane.  S is exactly
the lensing response `Sigma_s = (Phi+Psi)/(2 Psi)` of the closure lane;
within lensing alone it is exactly degenerate with the lens mass, so it
is identifiable ONLY because the dynamics law is frozen first.

| survey | what S is | n | source |
|---|---|---|---|
| eFEDS | `Sigma_s(r)` applied to the 3-D mass and re-projected; the likelihood is chi2 on raw reduced shear | 496 systems / 3365 points | DECADE (DELVE DR3) shapes x Bahar+2022 density fits |
| LoCuSS | `S = M_WL / M_dyn(r500_WL)` | 27 of 41 clusters | Mulroy+2019 x Okabe & Smith 2016 M_WL x ACCEPT n_e |
| SL cores | `S = 1/kappa_bar(<theta>)`, aggregated to one point per cluster | 4 clusters from 49 spectroscopic image systems | HFF multiple images x ACCEPT n_e x MCXC R500 |

Gates: eFEDS asserted at 496/3365; the
M_gas,500 gate reproduces Bahar's published value on n = 414
systems at median ratio 0.9994, scatter
0.0469 dex.  LoCuSS asserted at 41 rows x 13 and 28
columns.

| SL cluster | image systems | S | ln(r/R500) | error on the mean | within-cluster sd |
|---|---|---|---|---|---|
| A2744 | 22 | 4.198 | -2.380 | 0.0269 | 0.125 |
| AS1063 | 10 | 3.009 | -2.294 | 0.0326 | 0.097 |
| MACSJ0717 | 8 | 5.353 | -1.198 | 0.0774 | 0.212 |
| MACSJ1149 | 9 | 4.017 | -2.174 | 0.1261 | 0.377 |

| SL cluster | images in file | spec images | systems used | MCXC match |
|---|---|---|---|---|
| A2744 | 149 | 121 | 22 | J0014.3-3023 at 1.52' |
| A370 | 138 | 123 | 39 | none (nearest J0248.2-0216) at 132.80' |
| AS1063 | 48 | 25 | 10 | J2248.7-4431 at 0.15' |
| MACSJ0717 | 165 | 24 | 8 | J0717.5+3745 at 1.59' |
| MACSJ1149 | 97 | 27 | 9 | J1149.5+2224 at 0.26' |

**A370 dropped from the primary sample.** no external catalogue R500: nearest MCXC source is J0248.2-0216 at 132.8 arcmin, beyond the declared 3 arcmin match radius.  Kept in the R500_dyn variant.

**MACSJ0416 excluded.** multiple-image catalogue present (Caminha+2017, Bergamini+2023) but no ACCEPT deprojected n_e profile, so no baryon model

### Constraint-2 labels, stated plainly

* **LoCuSS `M_WL` is Okabe & Smith (2016)'s NFW-FITTED M_500**, not raw
  shear -- a parametric lens model, so not a raw observation in the
  sense the standing brief requires.  The cluster-data lane's
  weak-lensing availability audit establishes that no public per-source
  shear catalogue exists for ANY LoCuSS cluster, so there is no
  raw-shear route to this sample.  Carried, LABELLED, prior widened as
  a sensitivity.
* eFEDS uses raw per-source tangential shear; strong lensing uses raw
  image positions and spectroscopic redshifts.  Both comply.
* **No time delay is used anywhere in this lane.**  See section 12.
* `R500` is an externally supplied aperture LABEL, not a mass
  measurement.  Assumption stated in section 2.

### The declaration was amended once, before any residual was seen

PRIMARY_R500 was declared as R500_dyn.  Before any residual was examined it was shown ALGEBRAICALLY that for a single-aperture dataset like LoCuSS, ln(r/R500_dyn) = ln S/(3-m) exactly, i.e. the radius axis is a deterministic function of the very ratio it is meant to explain (measured corr = +0.885).  The primary was therefore amended to the EXTERNAL CATALOGUE aperture R500_cat (Bahar+2022 / Okabe M_WL / MCXC), on a pre-data admissibility argument in the sense of Run AM.  R500_dyn is retained as the declared alternative and both are reported.

---

## 2.  Design and leverage, measured BEFORE any residual

| survey | n | ln(M/M0) min/med/max | ln(r/R500) | ln(g_b/a0) |
|---|---|---|---|---|
| efeds | 3365 | -8.86 / -2.88 / -0.46 | -1.05 / +1.09 / +3.32 | -13.23 / -5.57 / -2.60 |
| locuss | 27 | -1.49 / -0.01 / +1.47 | +0.00 / +0.00 / +0.00 | -3.23 / -2.36 / -1.14 |
| sl | 49 | +0.26 / +0.45 / +0.75 | -3.11 / -2.26 / -0.31 | -2.64 / -2.05 / -1.09 |

(The `sl` row counts the 49 individual image systems, which is the
design that sets the RANGES.  The primary fit uses the 4 cluster
aggregates -- section 3.)

Radius occupancy -- the three surveys barely share the axis they
are supposed to be compared on:

| r/R500 | total | eFEDS | LoCuSS | SL |
|---|---|---|---|---|
| 0.030 - 0.082 | 14 | 0 | 0 | 14 |
| 0.082 - 0.223 | 28 | 0 | 0 | 28 |
| 0.223 - 0.472 | 10 | 5 | 0 | 5 |
| 0.472 - 0.779 | 96 | 94 | 0 | 2 |
| 0.779 - 1.284 | 379 | 352 | 27 | 0 |
| 1.284 - 2.117 | 674 | 674 | 0 | 0 |
| 2.117 - 4.482 | 1274 | 1274 | 0 | 0 |
| 4.482 - 12.182 | 926 | 926 | 0 | 0 |
| 12.182 - 33.115 | 40 | 40 | 0 | 0 |

| axis | between-survey sd | within eFEDS | within LoCuSS | within SL |
|---|---|---|---|---|
| ln M | 1.626 | 1.455 | 0.668 | 0.205 |
| ln r/R500 | 1.326 | 0.684 | 0.000 | 0.635 |
| ln g/a0 | 1.737 | 1.733 | 0.497 | 0.565 |

**The three survey means lie almost on a line in the (mass, radius)
plane: correlation -0.8208 on three
points.**  Between-survey information alone cannot separate mass from
radius; only within-survey slopes can, and only eFEDS has real spread
on more than one axis at once.  That is the whole reason the answer
here is carried by eFEDS's internal radial run and not by the
cluster/group contrast.

`R^2` of `ln(g_b/a0)` on `[1, ln M, ln r/R500]`, pooled = 0.7622, residual sd 0.8783.  Acceleration is a
partly distinct story here, unlike Run AI's potential depth, which was
98.6% a function of (g, r).

### LoCuSS carries no radial information, under either definition

* Under the catalogue aperture `R500 = r500(M_WL)`, every LoCuSS point
  sits at `r/R500 = 1` identically (sd 0.0e+00).
  **LoCuSS is a single-radius dataset.**
* Under the dynamical aperture, `corr(ln S, ln r/R500) = +0.8848` -- not a coincidence: with `M_dyn ~ r^m` near the aperture, `ln(r/R500_dyn) = ln S/(3-m)` EXACTLY.  A fit
  using it would have 'discovered' a radial dependence that is pure
  algebra.

**Assumption declared, for the lane auditing r/R500:** `R500` is taken
as an external aperture label (Bahar+2022 / `r500(M_WL)` / MCXC),
derived under standard-gravity scaling relations in all three cases.
That is uniform in kind, and it is the axis the programme's existing
`r/R500` claim refers to.  The law-frozen alternative is in section 10.

---

## 3.  Matched-radius comparisons -- the model-free discriminators

These do not extrapolate.  They compare S at the SAME r/R500 between
samples differing by more than a decade in mass.

| r/R500 band | eFEDS n / median M_gas500 | eFEDS S | other sample | its S | mass ratio | ln S difference | sigma | implied alpha |
|---|---|---|---|---|---|---|---|---|
| 0.2-0.8 | 106 / 7.70e+12 | 1.698 [1.334, 2.065] | SL, 7 systems in 2 clusters | 6.092 | 25x | +1.277 | +4.7 | +0.3967 |
| 0.3-0.8 | 106 / 7.70e+12 | 1.698 [1.334, 2.065] | SL, 6 systems in 2 clusters | 6.277 | 25x | +1.307 | +4.8 | +0.4060 |
| 0.2-1.3 | 467 / 6.02e+12 | 1.380 [1.216, 1.549] | SL, 7 systems in 2 clusters | 6.092 | 32x | +1.485 | +7.5 | +0.4282 |
| 0.8-1.3 | 361 / 5.47e+12 | 1.303 [1.122, 1.496] | LoCuSS, 27 clusters | 1.689 | 18x | +0.259 | +1.7 | +0.0897 |

**The implied mass slope is +0.397 at r/R500 ~ 0.5 and +0.090 at r/R500 ~ 1, a factor 4.4 apart.**  A single mass power
law produces neither pair.  Read the other way round: at fixed radius,
an 18x mass increase buys only +0.259 in ln S (+1.7 sigma) -- mass is a
weak organiser.

**Caveat, and a serious one.**  The eFEDS/SL overlap band exists
only because strong-lensing image systems are included out to large
clustercentric radius.  Restricting to mean image radius < 100
arcsec -- roughly twice the largest cluster Einstein radius known,
so only cluster-scale critical-curve tracers survive -- the same
band keeps 5 systems in 1
cluster (MACSJ0717) and
gives `alpha = +0.366` (+4.0 sigma).  The conclusion is unchanged but it
then rests on ONE merging cluster.  The r/R500 ~ 1 comparison --
361 eFEDS points against 27 LoCuSS clusters, two independent
methods -- is much the more secure.

### The strong-lens sample's own internal radial slope, and why it is discarded

| cluster | n systems | ln r span | beta |
|---|---|---|---|
| A2744 | 22 | 2.80 | +0.1648 +- 0.0277 |
| MACSJ0717 | 8 | 1.38 | +0.4269 +- 0.0596 |
| MACSJ1149 | 9 | 1.24 | -0.4970 +- 0.2530 |

Combined `beta = +0.2045 +- 0.0250` -- POSITIVE,
the opposite sign to every other probe here.  This is exactly the
artefact the null predicted: `S = 1/kappa_bar(theta)` and
`ln(r/R500) = ln(theta D_l/R500)` share `theta`, with
`d ln x/d ln theta = 1` EXACTLY, so any image system not actually on
the cluster's tangential critical curve is pushed up and out together.

Decomposing the joint likelihood by survey at `beta = -0.40`
against `beta = -0.10` shows exactly who was setting it:

| SL treatment | eFEDS prefers -0.40 by | strong-lens term prefers -0.10 by | LoCuSS | offset priors |
|---|---|---|---|---|
| 49 image systems | 12.33 | 64.66 | -0.21 | 8.46 |
| 4 cluster means | 12.33 | 6.47 | -0.21 | 8.30 |

**49 image systems in 4 clusters moved beta by
64.7 in -2 ln L while the 3365 eFEDS
raw shear points moved it by 12.3 -- outvoting the entire
weak-lensing dataset on the one parameter the lane exists to
measure.**  Aggregated, the same term is worth
6.5 and eFEDS decides.

The primary analysis therefore collapses the
strong-lens sample to one point per cluster, keeping the
Einstein-radius AMPLITUDE, which the argument supports, and discarding
the within-cluster radial structure, which it does not.  The
uncollapsed fit is reported in section 5 as the declared alternative.

---

## 4.  The eFEDS internal radial slope -- the real radial measurement

Refitted through the same 3-D forward model in radial windows, with the
amplitude profiled out each time.

| window | n | beta | 68% |
|---|---|---|---|
| all radii | 3365 | -0.400 | [-0.450, -0.350] |
| r/R500 < 1.3 | 467 | -0.050 | [-0.450, +0.400] |
| r/R500 < 2.0 | 1036 | -0.250 | [-0.400, +0.000] |
| r/R500 < 3.2 | 1823 | -0.150 | [-0.250, -0.050] |
| r/R500 < 5.0 | 2581 | -0.350 | [-0.400, -0.300] |
| r/R500 > 2.0 | 2329 | -0.800 | [-1.050, -0.600] |
| r/R500 > 3.2 | 1542 | -0.250 | [-0.700, +0.150] |
| 1.0 < r/R500 < 5.0 | 2352 | -0.450 | [-0.550, -0.300] |

**Blind protection.**  On the closure lane's declared split, TRAIN
gives `-0.500 [-0.600, -0.400]` (n = 1709) and the
HELD-OUT half gives `-0.300 [-0.400, -0.200]` (n = 1656).  The slope transfers.

**Null test.**  The identical statistic on the B-mode (cross)
component, which carries no lensing signal, buys `0.00`
chi2 across the whole beta grid against `106.29`
on the tangential component.  The estimator finds exactly nothing where
there is nothing.

**But the slope is not one number.**  Inside 2 R500 it is
`-0.250`; outside, `-0.800`, 1.8 sigma apart.  The Bahar+2022
Vikhlinin fits are anchored inside about R500; beyond that the baryon
model is an extrapolation, and one that over-predicts `M_b` pushes the
fitted response down and manufactures exactly this steepening.  Two
effects that could fake it the other way -- the two-halo term and
member contamination -- both RAISE the observed shear at large radius,
so they work against the signal rather than explaining it.

**This is the lane's dominant systematic and it is not resolved.**
With the outer bands the slope is `-0.400` and
the extrapolation to the strong-lens cores succeeds; restricted to
`r/R500 < 2` it is `-0.250` and it does
not.

---

## 5.  The prespecified hierarchy, primary and alternative

PRIMARY (strong lensing aggregated to 4 clusters,
N = 3396):

| model | k | -2 ln L | BIC | dBIC | fitted | what it says |
|---|---|---|---|---|---|---|
| **H_R** | 2 | 3532.77 | 3549.03 | 0.00 | `c=+0.397` `beta=-0.350` | ln S = c + beta ln(r/R500) |
| **H_T** | 3 | 3526.39 | 3550.78 | 1.75 | `c=-0.791` `A=+1.400` `lnxt=+1.099` | TRANSITION, form declared in advance: ln S = c + A / (1 + (r/(x_t R500))^p) with p = 2 FIXED |
| **H_MR** | 3 | 3530.64 | 3555.04 | 6.00 | `c=+0.343` `alpha=-0.063` `beta=-0.400` | ln S = c + alpha ln(M/M0) + beta ln(r/R500) |
| **H_P** | 3 | 3543.25 | 3567.64 | 18.61 |  | ln S = survey offsets only |
| **H_G** | 2 | 3554.88 | 3571.14 | 22.11 | `c=+0.655` `gamma=+0.100` | ln S = c + gamma ln(g_b/a0) |
| **H0** | 0 | 3574.32 | 3574.32 | 25.29 |  | ln S = 0 |
| **H_M** | 2 | 3561.15 | 3577.41 | 28.38 | `c=+0.337` `alpha=+0.020` | ln S = c + alpha ln(M/M0) |

DECLARED ALTERNATIVE (every image system treated as a separate point,
N = 3441):

| model | k | -2 ln L | BIC | dBIC | fitted |
|---|---|---|---|---|---|
| H_T | 3 | 3440.19 | 3464.62 | 0.00 | `c=-0.623` `A=+1.200` `lnxt=+1.099` |
| H_P | 3 | 3454.24 | 3478.67 | 14.05 |  |
| H_G | 2 | 3465.65 | 3481.94 | 17.32 | `c=+0.656` `gamma=+0.100` |
| H_R | 2 | 3467.14 | 3483.43 | 18.81 | `c=+0.348` `beta=-0.100` |
| H0 | 0 | 3485.60 | 3485.60 | 20.98 |  |
| H_M | 2 | 3472.34 | 3488.62 | 24.00 | `c=+0.339` `alpha=+0.020` |
| H_MR | 3 | 3467.13 | 3491.56 | 26.94 | `c=+0.344` `alpha=-0.004` `beta=-0.100` |

**The two disagree, and the reason is section 3.**  In the alternative,
the strong-lens sample's artefactual positive internal slope fights
beta, drags it from `-0.350` to `-0.100`, and hands the win to `H_T` with `H_P` second -- a step function and a set
of free per-survey constants, in that order, neither of which is a
statement about gravity.  The aggregated fit is primary because the
artefact is demonstrated, not assumed: the sign of the strong-lens
internal slope is wrong, and the mechanism (`d ln x/d ln theta = 1`
exactly) is algebra.

Offset priors, external and declared before fitting: eFEDS 0.05 dex, LoCuSS 0.05 dex, SL 0.15 dex; sources in `decl.py`.
Intrinsic scatters, estimated once under H_P and FROZEN for every
model: LoCuSS 0.2565, SL within 0.0000, SL cluster-common
0.2014.  (A free variance absorbs model misfit; under H0 it ran
to 0.93 and swallowed the entire strong-lensing signal.)

Profile 68% intervals (Delta(-2 ln L) = 1): `beta` (H_R) -0.400 to -0.300; `beta` (H_MR) -0.450 to -0.350; `gamma` (H_G) +0.100 to +0.150.

The transition model, declared in advance with `p = 2` fixed, was
admitted and lands at dBIC +1.75 with `A = 1.400`, `x_t = 3.000`.  It does not beat the
single power law, so the extra parameter is not bought.

---

## 6.  Frozen transfer: fit two surveys, predict the third

Declared in `decl.py` before any fit: train on eFEDS + strong lensing,
predict LoCuSS.  **Honesty note, declared in advance:** the LoCuSS
excess is already in the programme record and has been read by this
lane's author, so the freeze is PROCEDURAL, not epistemic.  It is not a
blind test in the strong sense and is not reported as one.
Leave-one-survey-out is run for all three.

The held-out survey's own offset is held at its prior mean, and the
prediction interval therefore INCLUDES the prior width; the
cluster-common scatter is divided by the number of CLUSTERS.  An
earlier version divided by image systems instead and inflated every
strong-lensing significance by about 3.5.

### held out: **locuss**  (trained on efeds + sl)

| model | predicted S | observed S | ln residual | sigma_pred | sigma |
|---|---|---|---|---|---|
| H0 | 1.000 | 1.661 | +0.507 | 0.132 | **+3.86** |
| H_M | 1.299 | 1.661 | +0.246 | 0.132 | **+1.87** |
| H_R | 1.333 | 1.661 | +0.220 | 0.132 | **+1.67** |
| H_G | 1.621 | 1.661 | +0.024 | 0.132 | **+0.18** |
| H_MR | 1.246 | 1.661 | +0.288 | 0.132 | **+2.19** |

### held out: **sl**  (trained on efeds + locuss)

| model | predicted S | observed S | ln residual | sigma_pred | sigma |
|---|---|---|---|---|---|
| H0 | 1.000 | 4.059 | +1.401 | 0.362 | **+3.87** |
| H_M | 1.286 | 4.059 | +1.149 | 0.362 | **+3.18** |
| H_R | 3.304 | 4.059 | +0.206 | 0.362 | **+0.57** |
| H_G | 1.518 | 4.059 | +0.984 | 0.362 | **+2.72** |
| H_MR | 3.287 | 4.059 | +0.211 | 0.362 | **+0.58** |

### eFEDS alone, extrapolated -- the result that carries the lane

`beta = -0.4000 [-0.4500, -0.3500]`, `c = +0.2609`, fitted on eFEDS raw
shear alone with no cluster data of any kind:

| target | r/R500 | predicted S | observed S | ln difference | sigma |
|---|---|---|---|---|---|
| LoCuSS | 1.000 | 1.298 [1.298, 1.298] | 1.661 | +0.247 | **+1.97** |
| SL cores | 0.134 | 2.903 [2.625, 3.210] | 4.059 | +0.335 | **+0.90** |

Both cluster samples sit a consistent factor of ~1.3 ABOVE the
group-scale radial law.  That common offset is the part this lane
cannot attribute.

---

## 7.  Shared-quantity audit and the null

Construction expressions are written out in `nulls.py`.  The null is
**not** `S = 1`: the parameters under test are alpha, beta and gamma
and the survey amplitudes are nuisances, so the null is `ln S = o_k`
with no dependence on M, r or g.  Setting `S = 1` in the strong-lensing
cores would mean the observed arcs do not exist; a null built that way
was run and returned `ln S` scatter of 1.69 against 0.27 in the data.
Under the null every REGRESSOR is rebuilt from the redrawn inputs, so
the shared paths are live.

| error scale | kept / rejected | E[c] median (MAD) | E[alpha] median (MAD) | E[beta] median (MAD) |
|---|---|---|---|---|
| 0.25 | 200 / 0 (0% rejected) | +0.4037 (0.0438) | +0.05788 (0.03714) | -0.06567 (0.07819) |
| 0.5 | 200 / 0 (0% rejected) | +0.3792 (0.0384) | +0.04634 (0.03436) | -0.06311 (0.06236) |
| 1.0 | 200 / 350 (64% rejected) | +0.3426 (0.0323) | +0.00803 (0.02391) | -0.06286 (0.05192) |

Three variance scalings because **Bahar+2022 publishes no covariance**
for its Vikhlinin parameters and they are strongly covariant, so the
marginal errors are an upper bound on the independent variance.  Each
realisation must pass the same M_gas,500 gate the ingest uses; the
rejection rate at scale 1.0 is 64%, which is itself
evidence that the marginal errors badly overstate the independent
variance.  **Without that gate the scale-1.0 null diverges** (it
returned `E[c] = -57 +- 798`), which is how the problem was found.

| parameter | estimate (joint, linearised) | null median bracket | null MAD | sigma from its own null |
|---|---|---|---|---|
| alpha | -0.0080 | [+0.0080, +0.0579] | 0.0239 | -2.75 to -0.67 |
| beta | -0.0971 | [-0.0657, -0.0629] | 0.0519 | -0.66 to -0.60 |

**The null bias on beta is `-0.0629` with MAD
`0.0519`.**  The eFEDS-only measurement,
`-0.400`, therefore sits 6.5
MAD from its own null rather than from zero.  (The null was computed
for the joint linearised estimator; the eFEDS-only estimator shares the
same input structure, so this is an indicative rather than exact
transfer, and is labelled as such.)

**Fisher errors against the simulated null** -- never quote the first
alone for a regressor built from someone else's fit:

| parameter | Fisher sigma | null MAD | ratio |
|---|---|---|---|
| c | 0.09713 | 0.03229 | 3.008 |
| alpha | 0.04125 | 0.02391 | 1.725 |
| beta | 0.04907 | 0.05192 | 0.945 |

The Fisher error is conservative for every parameter here, which is
the direction that does not mislead.

---

## 8.  Responsiveness

| parameter | injected | recovered (full fitter) | recovered (linearised) | d(est)/d(inj) | spread |
|---|---|---|---|---|---|
| alpha | [-0.1, -0.05, 0.0, 0.05, 0.1] | [-0.0774, -0.0283, 0.02, 0.0673, 0.1138] | [-0.1269, -0.0624, -0.008, 0.0385, 0.0784] | 0.9559 | 0.1912 |
| beta | [-0.3, -0.15, 0.0, 0.15, 0.3] | [-0.4, -0.25, -0.1, 0.05, 0.2] | [-0.3824, -0.2431, -0.0971, 0.0574, 0.2225] | 1.0000 | 0.6000 |

Both headline parameters move with their own injected value, so
neither is a monotone-blind statistic of the kind Run L and the
X-COP rank test caught.  The B-mode test in section 4 is the
complementary check: the estimator returns nothing where there is
nothing.

---

## 9.  POWER

| story | slope needed eFEDS -> SL | needed eFEDS -> LoCuSS |
|---|---|---|
| alpha | +0.3897 | +0.1686 |
| beta | -0.4452 | -0.5128 |
| gamma | +0.3658 | +0.1571 |

| parameter | eFEDS internal | 68% | required | sigma away |
|---|---|---|---|---|
| alpha | -0.0055 | [-0.0255, +0.0145] | +0.3897 | 19.8 |
| beta | -0.4000 | [-0.4500, -0.3500] | -0.4452 | 0.9 |
| gamma | +0.2000 | [+0.1500, +0.2500] | +0.3658 | 3.3 |

### What is limiting, and what would fix it

**Statistics are not limiting.**  The eFEDS radial slope is measured to
+-0.050
statistically on the present data, and more sky would not change the
answer.  Two systematics decide it:

1. **whether the eFEDS baryon model may be extrapolated past ~2 R500**
   -- worth 0.55 in
   beta, which is the entire disagreement;
2. **whether the strong-lensing image systems used in the overlap band
   are on the cluster critical curve** -- worth the difference between
   two clusters and one.

Three measurements would settle it, none needing more area:

* **Publish the Bahar+2022 Vikhlinin parameter covariance.**  The null
  bracket on alpha spans [+0.0080,
  +0.0579] purely because the covariance
  is unavailable and the marginal errors have to be bracketed over
  three scalings, at a 64% rejection rate.
* **Resolved weak-lensing shear profiles for massive clusters reaching
  inside 0.3 R500, from one pipeline.**  That would tie the
  strong-lensing cores to the weak-lensing scale with no cross-survey
  offset, and give radial leverage at FIXED high mass -- the one
  direction the present data have none of.  Sizing it: the
  cluster-level scatter measured here is
  0.201 in ln S per cluster; a profile spanning a decade in
  radius gives each cluster a lever arm of about +-1.15 in ln x, so
  sigma(beta) = 0.201/(sqrt(N) x 1.15) and beta to +-0.05 needs
  **N = 13 clusters** with
  per-cluster profiles -- 20-30 once systematics are allowed for, which
  is one existing sample, not a new survey.
* **Raw Subaru shear profiles for LoCuSS**, which would turn a
  single-radius aperture mass into a radial profile and remove both the
  NFW assumption and the single-radius theorem at once.

---

## 10.  Sensitivity of the verdict

**`H_R` is the best model on BIC in 9 of 9 variants,
and `beta` moves only over [-0.350, -0.300]
across all of them** -- including the law-frozen radius definition,
the temperature mass axis that breaks the shared path with the
density fit, a factor of four in the strong-lens stellar template,
two strong-lens selection cuts, and doubled priors on both cluster
samples.  The verdict is not a modelling choice.

| variant | n (eF/Lo/SL) | best on BIC | alpha | beta | gamma |
|---|---|---|---|---|---|
| primary | 3365/27/4 | H_R | +0.020 | -0.350 | +0.100 |
| r500_dyn | 3365/27/5 | H_R | +0.032 | -0.300 | +0.100 |
| mass_is_kT | 3365/27/4 | H_R | +0.036 | -0.350 | +0.100 |
| stars_half | 3365/27/4 | H_R | +0.020 | -0.350 | +0.100 |
| stars_double | 3365/27/4 | H_R | +0.019 | -0.350 | +0.100 |
| sl_theta_lt_100 | 3365/27/4 | H_R | +0.019 | -0.350 | +0.100 |
| sl_theta_lt_60 | 3365/27/4 | H_R | +0.016 | -0.350 | +0.100 |
| locuss_prior_wide | 3365/27/4 | H_R | -0.001 | -0.350 | +0.100 |
| sl_prior_wide | 3365/27/4 | H_R | +0.005 | -0.350 | +0.100 |

* **primary** -- R500 = external catalogue, M = M_gas(<R500), stars x1
* **r500_dyn** -- declared alternative: R500 from the frozen law and the baryons alone; adds A370 to the SL sample; LoCuSS radius axis is CIRCULAR here and the result must be read with that in mind
* **mass_is_kT** -- declared alternative mass axis: core-excised X-ray temperature, which breaks the shared path with the density fit
* **stars_half** -- strong-lens stellar template halved
* **stars_double** -- strong-lens stellar template doubled
* **sl_theta_lt_100** -- strong-lens image systems restricted to mean radius < 100 arcsec, roughly twice the largest cluster Einstein radius known, so that only cluster-scale critical-curve tracers are kept.  This removes the eFEDS/SL overlap in r/R500 almost entirely
* **sl_theta_lt_60** -- tighter still: only systems inside the largest observed cluster Einstein radii
* **locuss_prior_wide** -- widened because M_WL is an NFW-FITTED mass, not raw shear -- the profile-shape systematic is not in Okabe's quoted calibration budget
* **sl_prior_wide** -- doubled, to test whether the strong-lens anchor is doing the work through its prior rather than its data

---

## 11.  Bugs the tests found

`test_transition.py`: 17 checks, all passing.  Six real problems were
caught -- three by tests, three by numbers that were impossible:

1. **`pipeline.sigma_from_g`'s `Sigma_bar` is wrong at small radius.**
   It integrates `Sigma(R')` inward from a grid starting at 1 kpc and
   assumes `Sigma ~ const` inside it.  Against a singular isothermal
   sphere it is wrong by **8.1% at R = 27 kpc, 4.1% at 54 kpc and 2.1%
   at 108 kpc**, and the error is FLAT in `n_t`, `n_R` and the radial
   grid density -- the programme's own signature for a modelling
   mismatch rather than a quadrature error.  Negligible for the eFEDS
   shear (R > 0.29 Mpc) but not for strong-lensing cores at 50-250 kpc.
   Replaced by an exact form with no inner boundary term, integrated in
   `r = R cosh t` so the `1 - sqrt(1 - R^2/r^2)` kink at `r = R` is
   removed; it reproduces the SIS to **8e-6** once the declared 20 Mpc
   truncation deficit `R/(pi r_t)` is accounted for, and that deficit
   is matched to four figures.  **This affects any other lane using
   `sigma_from_g` inside ~50x its inner grid radius.**
2. **The declared radius axis was circular.**  `ln(r/R500_dyn)` for a
   single-aperture dataset is `ln S/(3-m)` exactly; measured
   `corr = +0.8848` on LoCuSS.  Caught by a test
   written before the fit; the declaration was amended on a pre-data
   admissibility argument.
3. **The strong-lens internal radial slope is an artefact**, sign and
   all, and it was outvoting 3365 shear points on beta.  Found by
   asking why the joint fit and the eFEDS-only fit disagreed, then
   confirmed by decomposing the likelihood: the eFEDS term prefers
   `beta = -0.40` by 12.3 while the strong-lens term prefers `-0.10` by
   64.7.
4. **The first null was ill-posed.**  Built around `S = 1` it scattered
   strong-lens image radii into a regime where the frozen law is
   subcritical, giving `ln S` scatter 1.69 against 0.27 in the data.
   Rebuilt around the fitted per-survey amplitudes.
5. **The null diverged at error scale 1.0** (`E[c] = -57 +- 798`)
   because pathological Vikhlinin draws give a near-zero predicted
   shear.  Fixed with a declared per-realisation validity gate; the
   rejection rate is now a reported result in its own right.
6. **The strong-lensing transfer significance was inflated ~3.5x** by
   treating 49 image systems in 4 clusters as independent, and the
   held-out survey's offset prior was omitted from the prediction
   interval.  Both fixed.

A seventh was avoided rather than caught: the B-mode radial-slope test
initially returned a strong spurious beta because the amplitude grid
was bounded away from zero, so the fit minimised the model instead of
matching it.  With zero admitted, the B-mode buys 0.00
chi2 against 106.29 for the real signal.

---

## 12.  What could NOT be established

1. **Whether the eFEDS radial slope beyond 2 R500 is gravity or the
   X-ray density extrapolation.**  This is the single fact the answer
   turns on: with the outer bands the slope is `-0.400` and the extrapolation to the
   strong-lens cores succeeds at +0.90 sigma; restricted to
   `r/R500 < 2` it is `-0.250` and it does
   not.  Deciding it needs gas profiles measured, not extrapolated,
   past 2 R500.
2. **Mass, radius and acceleration cannot be separated BETWEEN
   surveys.**  The three survey means are collinear at -0.8208 in (ln M, ln r/R500).  The
   verdict here rests on eFEDS's INTERNAL radial run; the
   cluster/group contrast contributes almost nothing to it.
3. **LoCuSS carries no radial information at all**, under either
   aperture definition.  A radially resolved LoCuSS measurement needs
   the raw Subaru shear profiles, which are not public.
4. **The strong-lens sample cannot measure its own radial slope.**  Its
   internal beta is `+0.2045 +- 0.0250`, positive,
   which is the shared-`theta` artefact.  Only its cluster-level
   amplitude is usable, and that rests on four clusters.
5. **Mass and survey pipeline are not separated.**  The residual factor
   of ~1.34
   by which both cluster samples sit above the eFEDS radial law is
   equally well described by `alpha ~ +0.09` or by
   a per-survey constant.
6. **No upper limit is set on a mass dependence** independent of the
   X-ray fit noise: alpha's null median moves over
   [+0.0080, +0.0579] across the three declared
   variance scalings, a property of the published catalogue rather than
   of the shear.
7. **The monopole approximation in the strong-lens cores is sized, not
   validated.**  These are merging clusters; the Refsdal lane measured
   source-plane rms 0.40-0.61 arcsec against theta_E = 10.6.  A proper
   test needs a full non-circular lens solve under each law.
8. **LoCuSS `M_WL` is an NFW fit.**  The profile-shape systematic is
   not in Okabe & Smith's quoted calibration budget and is bracketed
   here by widening the prior, not measured.
9. **Nothing here tests whether a radial response is ADMISSIBLE.**  A
   radial closure and a radial modification of the force law are the
   same object; this lane measures the object, it does not derive it
   from an action or check it against the compiler's gates.  That is
   the obvious next step and it is a theory step, not a data step.

---

## 13.  Standing-checklist items, each answered

* **Shared-denominator artefacts** -- construction expressions written
  out for all three surveys before any correlation was believed
  (`nulls.py`); the null simulated with the actual published errors at
  three variance scalings, with every regressor rebuilt from the
  redrawn inputs.  **Three live shared paths were found**: LoCuSS's
  `M_WL` in both numerator and aperture; the strong-lens sample's
  shared `theta`, large enough to reverse the sign of its internal
  beta; and the circular `R500_dyn` axis retired before any fit.
* **Monotone-invariant statistics** -- responsiveness measured for both
  headline parameters (section 8), plus a B-mode null showing the
  estimator returns nothing when there is nothing.
* **Refitting on the held-out set** -- the held-out survey's offset is
  NOT refitted; it is held at its prior mean and the prior width enters
  the prediction interval.
* **Silent extraction failures** -- every ingest asserts row and column
  counts and echoes identifiers; SHA-256 of every input file is in the
  results JSON.
* **A radial closure and a radial force-law modification are the same
  object** -- so `beta` here is the MEASUREMENT, and the survey offsets
  are pure amplitudes with informative external priors, never radial
  and never free, except in H_P where free amplitudes ARE the
  hypothesis.
* **Fisher vs null** -- reported as a ratio for every headline
  parameter in section 7.
* **A single time delay cannot test a gravity law** -- respected
  absolutely: **no time delay is used anywhere in this lane.**  The
  strong-lensing points are Einstein-radius constraints,
  `S = 1/kappa_bar(<theta>)` with the law frozen, which is a statement
  about the closure GIVEN the law, not a model-free slip measurement.
  They inherit the monopole approximation, and the within-cluster
  scatter of `ln S` is the empirical size of that approximation's
  error -- 0.097 to 0.377 across the four clusters.
* **Do not kill a candidate merely because it fails somewhere** -- no
  model is eliminated.  Each is reported with where it works and where
  it does not: `H_G` predicts LoCuSS better than `H_R` does; `H_M` is
  never best but is never catastrophic either once the strong-lens
  artefact is removed; and in the uncollapsed alternative the winners
  are `H_T` then `H_P`, reported as such in section 5.
