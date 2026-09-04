# lead01-ablation — the four named weaknesses of the potential-depth transfer

Lane: `work/wellnet-2026-09/lead01-ablation/`.
Input reused unchanged: `work/wellnet-2026-09/potential-depth/potential_depth_ladder.csv`
(4,150 rows x 20 columns, 317 systems, sha256
`0aa416205a6e9392317d72e1b6e8e33f50d2985863d808459e11f7dadbd8d3f8`). The ladder
was not rebuilt. Every published Run R number is reproduced **bit-exactly** as a
gate before anything new is computed: 252 systems in the window, transfer
0.10663923247024337 (M1) / 0.09536049731045002 (M3) / 0.2917146880783851 (M0) /
0.12520136775322022 (M2) dex, `beta` 0.17188370232387992 and 0.1686638467066086.

Pre-registration: `PREREGISTRATION.md`, sealed 2026-09-04T12:05:06Z, sha256
`40a22ae2f8f38eca63c1d145e887f7f18ede68428dad7a0f0871e16e6e35414d`, written
before any of the four tests was computed and declaring the ablation split, the
bootstrap statistic, the primary boundary rule and the fresh-sample protocol.

---

## VERDICT — the paired object bootstrap (item 2, first, as instructed)

**The potential law and the class step are statistically indistinguishable on the
held-out clusters, and no number of additional clusters can separate them.**

```
RMS(M1 potential depth) = 0.1066 dex      RMS(M3 class step) = 0.0954 dex
paired  dRMS = RMS(M1) - RMS(M3)          = +0.01128 dex

bootstrap, 20,000 draws, coefficients FROZEN, held-out systems resampled
   +0.01115 +- 0.00575     95% [-0.00007, +0.02246]    P(M1 better) = 0.026
nested bootstrap, training set resampled as well
   +0.01241 +- 0.01347     95% [-0.01883, +0.03709]    P(M1 better) = 0.145
percentiles 1/5/25/50/75/95/99 (frozen)
   -0.0020  +0.0016  +0.0073  +0.0111  +0.0150  +0.0207  +0.0245

per object: M1 is the closer prediction on 24 of 52 clusters
   sign test p = 0.678        paired t on squared error = +1.81

against its own shared-denominator null, simulated with the actual error
covariance:  null dRMS = -0.00049 +- 0.00667   ->   observed z = +1.76
```

Four independent readings agree. The frozen 95% interval grazes zero. The
object-by-object sign test is 24 against 28 — a coin flip, so the RMS gap is
not systematic superiority, it is a handful of objects. Including coefficient
uncertainty the one-sided probability is 0.145. And the difference sits 1.76
sigma from where it would sit if potential depth did nothing at all.

**The difference is at a hard ceiling.** Decomposing the bootstrap variance:

| source | contribution to sd(dRMS) | shrinks with more clusters? |
|---|---:|---|
| resampling the 52 held-out clusters | 0.00575 dex | yes, as 1/sqrt(n_test) |
| the 200-system **training** set | **0.01217 dex** | **no** |
| total | 0.01347 dex | |

```
z with an INFINITE held-out cluster sample          0.93 sigma
held-out clusters needed for 2 or 3 sigma           unreachable at any n
training systems needed for 3 sigma                 ~2,100   (currently 200)
```

The 1/sqrt(n_test) scaling was verified empirically (bootstrap sd x sqrt(n/52) =
0.0053, 0.0059, 0.0058 at n = 13, 26, 52). Reporting "0.0954 against 0.1066" as
a ranking over-read a difference the data cannot support in either direction.

**Can the two be distinguished at all, by construction? Barely.**

```
correlation of the two frozen predictions over all 252 systems     +0.9227
rms | pred(M1) - pred(M3) |, all 252 systems                        0.0626 dex
rms | pred(M1) - pred(M3) |, the 52 held-out clusters               0.0461 dex
max | pred(M1) - pred(M3) |, the held-out clusters                  0.1224 dex
median measurement error on a held-out cluster's deviation          0.0500 dex
clusters where the gap exceeds that object's OWN error              20 of 52
```

Inside the held-out clusters `log|Phi_b|` spans 1.484 dex, so M1's prediction
swings 0.255 dex across them while M3 is flat by construction — that swing is
the only place the two models say different things. Per rung, the mean
| pred(M1) - pred(M3) | is 0.055 (galaxies), 0.206 (the single small group),
0.055 (poor groups), 0.039 (rich groups), 0.041 (low-mass clusters), 0.024
(massive clusters). **The disagreement is largest exactly where the data are
thinnest.** The five held-out clusters with the largest disagreement are
Sun+2009 A262, RXJ1206-0744, NGC6338, RBS461 and A744 — all low-mass, all at the
shallow end — and in all five the observation lands closer to the step.

**A methodological finding from the responsiveness gate.** The headline transfer
statistic is *exactly invariant* to the effect it is supposed to measure.
Injecting `(q/2) log|Phi_b|` into the response and refitting M1 gives, at every
injected `q` from 0 to 0.8:

```
d(beta)/d(q)                       = 0.5000 exactly   (estimator unbiased)
d(rms of M1 on the holdout)/d(q)   = 0.0000 exactly
```

because M1's own coefficient absorbs the injection algebraically. "0.1066 dex, a
63% reduction" would read 0.1066 dex whether the true `q` were 0, 0.37 or 0.8.
The statistic that *does* respond is the paired difference: `d(dRMS)/dq` runs
between -0.044 and +0.113 over `q` in [-0.4, +0.8], with dRMS spanning 0.054 dex.
Read against that scale, the observed dRMS = +0.01128 is the displacement a true
`q` of about **0.18** would produce — against the `q = 0.371` the cluster excess
requires. This is the same family as the monotone-invariant rank statistic this
programme already retracted, found this time in the headline number of the
strongest live lead.

---

## 1. THE ABLATION

Three training sets; the identical 52 held-out cluster systems in every arm;
coefficients frozen before the held-out set is touched.

### 1.1 The three arms

Frozen-coefficient RMS of `log10(nu_obs/nu_RAR)`, in dex.

| arm | train | test | M0 RAR only | M1 potential | M2 radius | M3 class step |
|---|---:|---|---:|---:|---:|---:|
| **A** galaxies only | 164 | 52 clusters | 0.3579 | **0.2007** | 0.2515 | *not estimable* |
| **A** galaxies only | 164 | 36 groups | 0.4018 | **0.2694** | 0.2999 | *not estimable* |
| **B** groups only | 36 | 52 clusters | 0.0871 | 0.3331 | 0.1661 | *not estimable* |
| **C** galaxies+groups | 200 | 52 clusters | 0.2917 | 0.1066 | 0.1252 | **0.0954** |

The held-out cluster deviations are `+0.2628 +- 0.0993` dex; the groups are
`+0.2344 +- 0.1199`; the galaxies are `-0.0581 +- 0.1938`.

### 1.2 What each arm actually does

| training set | n | `beta` | `q` | mean prediction / observed cluster offset |
|---|---:|---:|---:|---:|
| galaxies only, M1 | 164 | +0.0900 | +0.180 | 33.5% |
| groups only, M1 | 36 | **-0.3189** | -0.638 | 22.9%, wrong sign, RMS 0.3331 |
| galaxies + groups, M1 | 200 | +0.1719 | +0.344 | 100.6% |
| galaxies + groups, M3 | 200 | step +0.376 dex | — | 111.7% |
| **groups only, constant offset** | 36 | *no parameter at all* | — | **89.2%** |

* **Arm A does not predict the group and cluster trend.** Fitted on 164 galaxies
  alone, `beta = +0.0900` — half the published value. It recovers 33.5% of the
  cluster offset and **-0.4% of the group offset**, i.e. essentially none of it.
  It is unambiguously better than the RAR alone, which predicts a *negative*
  offset (-31.7%): paired P(M1 better than M0) = 1.0000, M1 closer on 52 of 52
  clusters and 36 of 36 groups. But recovering a third of an effect while
  missing the intermediate population entirely is not "predicting the trend".
* **Arm B is a failure, not an extrapolation that works.** Fitted on the 36
  groups alone, `beta = -0.3189` — the wrong sign — and the transfer RMS is
  0.3331 dex, worse than the RAR alone, worse than the frozen training mean
  (0.1033), worse than doing nothing. The groups span 0.231 dex in `log|Phi_b|`
  and the clusters sit 1.53 training sd beyond their median. **The group
  population cannot constrain `beta` at all.**
* So the published `beta = +0.1719` is neither a group measurement nor a galaxy
  measurement. It is fitted across the 2.66-dex galaxy-to-group lever arm, which
  Run R already showed is 86% class label. Its job in arm C is to carry a
  training set that is 82% galaxies, mean deviation -0.058 dex, up to the
  group/cluster level of +0.26 dex. **That is a bridging coefficient for a class
  offset — which is what the class step is, written differently.**

### 1.3 The transferable content is one number

The strongest robust competitor found anywhere in the exercise carries **no
potential-depth term and no class label**: the frozen mean deviation of the 36
groups, `+0.2344 dex`, a single constant.

| model | trained on | point RMS on the 52 clusters | 95% under training resampling |
|---|---|---:|---|
| **constant group offset** | 36 groups | **0.1033** | **[0.0993, 0.1200]** |
| M0 linear in `log g_bar` | 36 groups | 0.0869 | [0.0851, 0.1769] |
| M0 quadratic in `log g_bar` | 36 groups | 0.0871 | [0.0823, 0.9488] |
| M1 potential depth (published) | 200 | 0.1066 | [0.1045, 0.1391] |
| M3 class step (published) | 200 | 0.0954 | [0.0872, 0.1387] |
| M0 RAR only (published arm) | 200 | 0.2917 | [0.2645, 0.3227] |

The groups-only *quadratic* point estimate of 0.0871 dex — nominally the best
number in the table — **does not survive resampling its own training set**
(median 0.175, 95% up to 0.95). It extrapolates a quadratic fitted over 0.470
dex in `log g_bar` to clusters reaching 0.459 dex beyond that range. It is
reported and then discarded; that is why the constant is the honest headline.

Paired object bootstrap of the constant against the published models, frozen
coefficients, same 52 clusters:

```
constant (+0.2344 dex) vs published M1 : dRMS -0.0033 [-0.0208, +0.0140]  P = 0.659, closer on 33/52
constant (+0.2344 dex) vs published M3 : dRMS +0.0079 [-0.0124, +0.0282]  P = 0.225, closer on 27/52
constant (+0.2344 dex) vs published M0 : dRMS -0.1884 [-0.2082, -0.1673]  P = 1.000, closer on 50/52
```

**A single frozen number is statistically indistinguishable from both the
potential-depth model and the class step, and beats the RAR alone by the same
margin they do.**

### 1.4 The class step is not estimable without both classes

A structural point the ablation exposes that was not previously on the record.
In arm A the step dummy is identically 0 on the training set; in arm B it is
identically 1. Both designs are rank-deficient, and in both arms M3's frozen
predictions collapse onto M0's *exactly* — 0.3579 and 0.0871 dex, digit for
digit. **The class step is estimable only when the training set straddles the
galaxy/non-galaxy boundary; the potential-depth model is not so restricted.**
That is a real advantage for M1 over M3, and it is the only one the ablation
found.

Fair treatment of the null also requires checking whether the step is fragile to
*where* the boundary is drawn. It is not:

| class-step definition | transfer |
|---|---:|
| as published, step = 1 for rank > 1 | 0.0954 |
| the SDSS small groups reclassified as galaxies | 0.0944 |
| small and poor groups reclassified as galaxies | 0.1023 |
| "is it X-ray selected?" (rungs 3-6) | 0.0944 |
| "is it a cluster?" (rungs 5-6) | 0.2917 — not estimable in training |

### 1.5 Leave-one-rung-out of the published training set

| dropped from training | n train | M0 | M1 | M3 | `beta` | step |
|---|---:|---:|---:|---:|---:|---:|
| — (full) | 200 | 0.2917 | 0.1066 | 0.0954 | +0.1719 | +0.3760 |
| rung 2, small groups (1) | 199 | 0.2937 | 0.1063 | 0.0947 | +0.1701 | +0.3738 |
| rung 3, poor groups (6) | 194 | 0.2954 | 0.1071 | 0.0936 | +0.1698 | +0.3684 |
| rung 4, rich groups (29) | 171 | 0.3281 | 0.1089 | **0.1384** | +0.1569 | +0.4505 |
| rung 1, all galaxies (164) | 36 | 0.0871 | 0.3331 | 0.0871 | -0.3189 | n/a |

Dropping the 29 rich groups barely moves M1 (0.1066 -> 0.1089) but degrades the
step by 45% (0.0954 -> 0.1384). Of the two, the *step* is the more
group-dependent.

### 1.6 The gradient inside each population — the decisive table

The transfer test asks whether a frozen model lands the held-out clusters at the
right **level**. It never asks whether the model tracks the right **gradient**.
Fitting `beta` inside each population separately — one class, one instrument,
one pipeline, no label to lean on:

| population | n | span of `log|Phi_b|` | `beta` | 95% | `q` | P(`beta` > 0) |
|---|---:|---:|---:|---|---:|---:|
| galaxies (rung 1) | 164 | 2.20 | +0.0900 | [+0.0032, +0.1830] | +0.180 | 0.978 |
| groups (rungs 2-4) | 36 | 0.91 | -0.3189 | [-0.5740, -0.1779] | -0.638 | 0.000 |
| clusters (rungs 5-6) | 52 | 1.48 | -0.1807 | [-0.3008, -0.0786] | -0.361 | 0.001 |
| groups + clusters | 88 | 1.93 | -0.1987 | [-0.2976, -0.1327] | -0.397 | 0.000 |
| **all 252 pooled** | 252 | 3.68 | **+0.1687** | [+0.1406, +0.1969] | +0.337 | 1.000 |

The same table under the primary boundary rule BARY: +0.0791, -0.3065, -0.1643,
-0.1897, +0.1662. **Every within-class slope is consistent with zero or
negative; only the pooled fit is positive.** This is a Simpson's-paradox
structure: the published `beta` is entirely a between-class quantity.

And inside the held-out set itself, with no control at all — split the 52
clusters at their median `log|Phi_b| = 11.062`:

```
low  half  n=26  median log|Phi_b| 10.908   mean deviation +0.2556 +- 0.0834
high half  n=26  median log|Phi_b| 11.605   mean deviation +0.2699 +- 0.1125
observed slope  +0.0143 / +0.6972 = +0.0205    ->   q = +0.041
the FROZEN beta = +0.1719 predicts +0.1198 dex across that span; observed +0.0143
```

**M1 passes the transfer test by getting the level right, not the gradient.**
The gradient it was fitted to is not present inside the cluster population.

*Caveat, stated because Run Z requires it:* the negative within-group and
within-cluster `beta` must **not** be read as a measurement of gravity. For
hydrostatic systems the shape factor `S` and the observable share the density
log-slope; Run Z measured -0.400 on eFEDS and watched it flip to +0.463 when the
log-slope was controlled. The safe reading is only that **no within-class
measurement supports a positive `beta` of the published size**, and that the
galaxy value — which is not hydrostatic — is the one clean number and it is
+0.090 [+0.003, +0.183].

### 1.7 How the claim should be stated

Not a galaxy-to-cluster prediction; not a group-to-cluster extrapolation either:

> Groups and clusters share a common ~+0.25 dex offset from the RAR. A single
> frozen number — the group offset — predicts the held-out clusters as well as
> the potential-depth model does, and the potential-depth gradient is absent
> inside every individual population. The role of `beta log|Phi_b|` in the
> published fit is to carry a training set that is 82% galaxies across that
> class offset.

Recorded in the hypothesis's favour, because the brief requires it: the
galaxies-only `beta = +0.0900` is a genuinely **label-free, single-class,
single-instrument, non-hydrostatic** estimate; it has the right sign; it is
2.0 sigma from zero; it beats the RAR alone on all 88 held-out non-galaxies; and
it is consistent with Run N's within-galaxy bound `|q| <= 0.29`. It is also a
third of what the cluster excess needs.

**Robustness.** A linear rather than quadratic baseline in `log g_bar` changes
nothing qualitative: arm A 0.2007 -> 0.1973, arm B M0 0.0871 -> 0.0869, arm C M1
0.1066 -> 0.1057, M3 0.0954 -> 0.0879.

---

## 2. THE PAIRED BOOTSTRAP IN FULL

All pairwise comparisons, same 52 held-out clusters, coefficients frozen:

| comparison | RMS A | RMS B | dRMS | 95% | P(A better) |
|---|---:|---:|---:|---|---:|
| M0 vs M1 | 0.2917 | 0.1066 | +0.1851 | [+0.1527, +0.2186] | 0.000 |
| M0 vs M3 | 0.2917 | 0.0954 | +0.1964 | [+0.1590, +0.2341] | 0.000 |
| M1 vs M2 | 0.1066 | 0.1252 | -0.0186 | [-0.0355, -0.0024] | 0.989 |
| **M1 vs M3** | **0.1066** | **0.0954** | **+0.0113** | **[-0.0003, +0.0226]** | **0.028** |
| M2 vs M3 | 0.1252 | 0.0954 | +0.0298 | [+0.0088, +0.0504] | 0.003 |

Both M1 and M3 beat the RAR alone decisively and by an almost identical margin.
The only comparison in the table that is *not* decisive is the one the programme
has been treating as decisive. M1 does beat radius (M2) at 98.9%, which is worth
recording: at fixed `g_bar` the potential is 92% radius, but the residual 8%
points the right way.

---

## 3. AN OPERATIONAL DEFINITION OF POTENTIAL DEPTH

`|Phi_b|` was replaced by the potential difference
`DeltaPhi_b(r; r_ref) = Int_r^r_ref g_b(s) ds`. Each system's baryonic profile
was reconstructed from the ladder in the three forms `ladder.py` actually used
(187 resolved, 43 two-radius, 87 single-radius; single-radius systems use the
median two-radius power-law index k = 1.3067, a declared assumption).

**Reconstruction gate:** the TAIL rule reproduces the published `log|Phi_b|`
column to **3.6e-15 dex** over all 4,150 rows.

**PRIMARY rule declared in advance:** **BARY**, `r_ref = 10 x r_half,b`.

| rule | `r_ref` | usable rows | rows with `r_ref` < `r` | `beta` | `q` | transfer M1 | transfer M3 | dRMS |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **BARY** *(primary)* | `10 r_half,b` | 4141 | 26 | **+0.1731** | +0.346 | **0.1057** | 0.0953 | +0.0103 |
| PHYS | 2000 kpc | 4150 | 0 | +0.1779 | +0.356 | 0.0990 | 0.0954 | +0.0036 |
| OVER | `r_200b`, baryonic | 4133 | 362 | +0.1408 | +0.282 | 0.1614 | 0.0918 | +0.0696 |
| TAIL | infinity + point mass | 4150 | 0 | +0.1719 | +0.344 | 0.1066 | 0.0954 | +0.0113 |

```
beta spans        +0.1408 ... +0.1779    range 0.0372 = 22.4% of the mean
implied q spans   +0.2815 ... +0.3559    (q required for the cluster excess = 0.371)
M1 transfer       0.0990 ...  0.1614     range 0.0624 dex
dRMS(M1 - M3)     +0.0036 ... +0.0696
```

**The boundary rule moves `beta` twelve times more than the refit offered as
evidence of stability.** The published "1.9% shift" is `beta` moving 0.00322 when
the model is refitted on everything; changing the reference rule moves it 0.0372.

| rule | `q` / `q_required` | M1 error reduction vs the RAR | M3 error reduction | in-sample dBIC(M1 - M3) |
|---|---:|---:|---:|---:|
| BARY | 0.93 | 63.8% | 67.3% | +20.7 |
| PHYS | 0.96 | 66.1% | 67.3% | +14.6 |
| OVER | 0.76 | 44.5% | 68.5% | +69.2 |
| TAIL | 0.93 | 63.4% | 67.3% | +17.6 |

**Under every one of the four rules the class step wins on both the error
reduction and the BIC.** The primary rule does not rescue the potential law, and
the existing convention was not a lucky choice — it was a middling one.

### Three corrections to how the Run Z identity has been read

1. `corr(residual log|DeltaPhi_b|, residual log S) = +1.000000` under **all four
   rules**. That is because `S` is *defined* as `|DeltaPhi_b| / (g_bar r)`, so
   `log|DeltaPhi_b| - log g_bar - log r` **is** `log S` identically, for any
   boundary rule whatever. The +1.0000 is a tautology, not a property of the
   boundary condition, and no choice of rule can escape it.
2. What the rule genuinely changes is the information content of `S`:

   | rule | sd(log S) | min | max | within-`g_bar`-bin leverage | R2 on quadratic(`log g_bar`, `log r`) | partial corr with `log r` |
   |---|---:|---:|---:|---:|---:|---:|
   | BARY | 0.4612 | -1.662 | 3.228 | 0.7796 | 0.8745 | +0.9494 |
   | PHYS | 0.3959 | -0.750 | 3.266 | 0.6916 | 0.8870 | +0.9720 |
   | OVER | 0.4564 | -1.662 | 2.544 | 0.6788 | 0.8729 | +0.9050 |
   | TAIL | 0.3867 | -0.057 | 3.267 | 0.7656 | 0.9068 | +0.9740 |

   **Under every rule, potential depth at fixed acceleration is still radius**
   (+0.905 to +0.974). Run R's collinearity verdict survives the reformulation
   intact.
3. Run R's theorem `S >= 1` is a property of the **point-mass-tail convention**,
   not of gravity. With a finite reference radius `log S` reaches -1.66, and
   26 (BARY) / 362 (OVER) rows have `r_ref` inside `r` altogether. Those counts
   are reported, not silently dropped; 4 systems fall out of the OVER analysis
   entirely (248 instead of 252).

Median `r_ref` by rung, in kpc — the rules are genuinely different variables,
not reparametrisations:

| rung | BARY | PHYS | OVER | TAIL |
|---|---:|---:|---:|---:|
| 1 field galaxies | 52.0 | 2000 | 54.1 | inf |
| 2 small groups | 1676.6 | 2000 | 50.7 | inf |
| 3 poor groups | 2448.3 | 2000 | 103.2 | inf |
| 4 rich groups | 3147.6 | 2000 | 240.8 | inf |
| 5 low-mass clusters | 3843.3 | 2000 | 356.9 | inf |
| 6 massive clusters | 9009.2 | 2000 | 679.5 | inf |

The ablation repeated under the primary rule tells the same story: galaxies-only
`beta = +0.0791`, RMS 0.2165; groups-only `beta = -0.3065`, RMS 0.3376;
groups-only M0 unchanged at 0.0871 because it does not use the variable at all.

---

## 4. THE CLUSTERS ARE NOW VALIDATION DATA — A FRESH SEALED SAMPLE

### 4.1 What was sealed, and when

**Sample: Babyk, McNamara, Nulsen, Hogan, Vantyghem, Russell, Pulido & Edge
2018, ApJ 857, 32 (`2018ApJ...857...32B`, arXiv:1803.00020) — 94 early-type
galaxies with Chandra hydrostatic masses at 5 r_e**, plus 2MASS K-band total
magnitudes for the stellar mass.

* **Never used anywhere in this programme.** It does not appear in Runs A–AB and
  is not in the cluster-audit acquisition set.
* **Acquired by a process that computed no residual, no boost and no
  acceleration ratio.** The acquisition was carried out under instructions that
  forbade any analysis; only masses, radii, temperatures, distances and
  magnitudes were extracted.
* **The model was frozen and sealed first.** `fresh_seal.json`, written
  2026-09-04T12:18:16Z, sha256 `ee2fa831df5f77f2…`, contains the frozen
  coefficients, the response definition, the window and every cut, and was
  written before the sample was read. One amendment is recorded, timestamped
  12:27:35Z, before any residual was computed — see 4.2.
* Because four models have already been ranked on the 52 ladder clusters, they
  were used for **training** here: all 252 ladder systems train, and the fresh
  sample is the only holdout.

**Why this sample and not more clusters.** These objects are *galaxies*, so the
class step predicts no offset for them, while potential depth predicts an offset
set by a potential depth that sits between spirals and groups. It is the one
configuration in which the two models disagree by more than their measurement
error. Their median `log|Phi_b|` is 10.27, against 9.54 for the ladder's field
galaxies and ~10.6–11.8 for its groups and clusters.

Frozen coefficients, response `log10(nu_obs/nu_RAR)`, free quadratic in
`log10 g_bar`, trained on all 252 ladder systems:

```
M1_BARY   -7.336815  -0.941937  -0.038282  +0.166224   (train rms 0.1714 dex)
M1_TAIL   -8.916062  -1.191354  -0.048276  +0.168664   (train rms 0.1703 dex)
M0       +17.998893  +3.371839  +0.157970               (train rms 0.2212 dex)
M3       -15.755991  -2.990811  -0.142051  +0.354918    (train rms 0.1645 dex)
```

### 4.2 Ingest gates, and one amendment

**Row counts asserted:** 94 / 94 / 94 across Babyk's three tables, 94 K-band
rows, identical name lists. The acquisition caught three silent-extraction traps
this programme already knows about — all three of Babyk's tables are split
across two `table*` environments (57 + 37 rows), so a naive parser would have
returned 57 of 94 with no error.

**GATE: reproduce the published `M_gas(<5 r_e)`** by integrating the tabulated
beta-model `(rho_0, r_c, beta)`. **Median ratio 0.9976** — the formula and the
units are right — but the scatter is 0.53 dex with outliers from 0.007 to 47.6,
and `corr(|log ratio|, log(r_5re/r_c)) = +0.545`. The cause is that `r_c` is
printed to 0.01 kpc and 17 of 94 objects have `r_c <= 0.10 kpc`, so `r_c` itself
carries a factor-of-two rounding error and the beta-model normalisation carries
up to 2 dex with it. **Amendment, recorded 12:27:35Z before any residual was
computed: the gas NORMALISATION is taken from Babyk's own tabulated
`M_gas(<5 r_e)` and the beta-model supplies only the SHAPE.** Without this gate
the sample would have carried 2-dex gas-mass errors invisibly.

**Baryons.** `M_b(<5 r_e)` = the tabulated `M_gas` plus a Hernquist stellar
component with `a = r_e/1.8153` and `M_* = Upsilon_K L_K`, `Upsilon_K = 0.75`
declared in advance, `M_K,sun = 3.27`. Median `M_b = 1.5e11 Msun`
(`M_* = 1.34e11`, `M_gas = 3.4e10`; the median gas fraction is 0.316). K-band:
92 of 94 from 2MASS XSC `K.ext`, one from HyperLeda, one from 2MRS; median match
separation 0.30 arcsec; ATLAS-3D photometric cross-check on 35 overlapping
objects gives a median offset of -0.011 mag and rms 0.008 mag — a consistency
check, since ATLAS-3D's `KMAG` is itself built on 2MASS.

**Cuts, all declared in the seal.** 94 acquired -> 93 usable
(`M_tot_err_stat/M_tot <= 0.5`, finite everything). 89 of 93 fall inside the
fitted acceleration window. Primary subset = the 67 that are inside the window
**and** carry neither the BCG nor the cD flag, so they are unambiguously
individual galaxies rather than the central object of a group halo. The 24
BCG/cD objects are reported separately and never pooled.

### 4.3 The one evaluation

67 primary objects. `log g_bar` median -10.76, median radius 35 kpc,
`kT` 0.20–2.05 keV, `z` 0.0007–0.0273.

**Observed `log10(nu_obs/nu_RAR)` = +0.3418 dex, 95% [+0.2766, +0.4089], sd
0.2786. 60 of 67 objects are positive; 46 exceed +0.2 dex.**

For scale, the same quantity across the existing ladder: field galaxies -0.058,
groups +0.234, clusters +0.263. **These are galaxies, and they deviate from the
RAR by MORE than clusters do**, at a potential depth 0.8 dex shallower than the
clusters.

| frozen model | prediction is | RMS | bias | scatter | mean prediction |
|---|---|---:|---:|---:|---:|
| M0 RAR only | — | 0.4283 | +0.3290 | 0.2742 | +0.0128 |
| **M1 potential depth (BARY, primary)** | pre-declared | **0.4202** | +0.3005 | 0.2937 | +0.0413 |
| M1 potential depth (TAIL) | pre-declared | 0.4121 | +0.2946 | 0.2881 | +0.0472 |
| **M3 class step, step = 0** | pre-declared | **0.4668** | +0.3688 | 0.2862 | -0.0270 |
| *M3 class step, step = 1* | *post-hoc relabel* | *0.2865* | *+0.0139* | *0.2862* | *+0.3279* |

Paired object bootstrap, 20,000 draws, all coefficients frozen:

```
M1(BARY) vs M3(step = 0, the pre-declared reading)
   dRMS -0.0466  95% [-0.0616, -0.0319]   P(M1 better) = 1.0000, M1 closer on 54 of 67
M1(BARY) vs M3(step = 1, the post-hoc relabel)
   dRMS +0.1337  95% [+0.0808, +0.1813]   P(M1 better) = 0.0000, M1 closer on 20 of 67
```

### 4.4 What this says

1. **Both frozen models fail on genuinely fresh data.** M1 predicts +0.041 dex
   where +0.342 is observed — it recovers **12%** of the offset. M3, honestly
   applied, predicts -0.027 dex — it recovers **-8%**, i.e. it points the wrong
   way. The RAR alone predicts +0.013. None of the three is close.
2. **On the pre-declared comparison, potential depth beats the class step
   decisively — the opposite of the ladder verdict.** P(M1 better) = 1.0000,
   M1 closer on 54 of 67 objects. This is the strongest result in the lane in
   the hypothesis's favour, and it is the only one obtained on a sample that had
   never been looked at.
3. **The class step's apparent success depends on a label chosen after the
   answer is known.** Reclassifying these galaxies as non-galaxies takes M3 from
   0.4668 to 0.2865 dex and from a +0.37 dex bias to +0.01. The model contains
   no rule that would have told anyone to do that in advance — the ladder's own
   `class_rank` is `field_galaxy` for every rotation-curve galaxy, and these are
   galaxies. **That is exactly the objection the class step was supposed to
   answer, and it is now demonstrated rather than asserted.**
4. **A monotone `A(|Phi_b|)` cannot fit the combined set.** These objects sit
   0.73 dex deeper than the ladder's field galaxies and 0.8 dex shallower than
   its clusters, yet their deviation (+0.342) exceeds the clusters' (+0.263).
   Run R found the rungs non-monotone in `|Phi_b|` inside the ladder; the fresh
   sample makes the non-monotonicity larger and puts it on a completely
   different instrument.

### 4.5 Systematics on the fresh sample, stated plainly

* **Stellar mass-to-light ratio.** `Upsilon_K = 0.60 / 0.75 / 1.00` gives an
  observed mean deviation of +0.381 / +0.342 / +0.287 dex. The result is not an
  `Upsilon_K` artefact. Quantitatively, `d(deviation)/d(log M_b) = -0.60` at
  this acceleration, so erasing +0.342 dex would need **0.57 dex more baryons —
  a factor 3.7** — which no plausible `Upsilon_K` or gas correction supplies.
* **Hydrostatic bias is the real exposure.** `d(deviation)/d(log M_tot) = +1`
  exactly, so erasing the offset needs `M_tot` lower by a factor 2.2. Babyk's
  masses come from an *isothermal beta-model* plus a *single-temperature*
  spectral fit, and the hot haloes of early-type galaxies at 5 r_e are the
  regime where hydrostatic equilibrium is least secure (rotation, AGN-driven
  bulk motion, sloshing). A factor-2.2 hydrostatic bias is larger than is
  usually claimed but is not excluded by this lane's own evidence. **This lane
  cannot distinguish "gravity deviates for X-ray ETGs" from "X-ray ETG
  hydrostatic masses are biased high by a factor ~2".**
* Where the surface-brightness profile does not reach 5 r_e, Babyk extrapolated
  using the slope of the last 20 points. K magnitudes are not extinction-
  corrected for 93 of 94 objects (`A_K` is 0.01–0.1 mag, at most 0.04 dex in
  `L_K`). Redshifts are all below 0.028, so no `E(z)` correction matters.
* The scatter is large — 0.279 dex, three times the held-out clusters' 0.099 —
  so the per-object power is low. The 3.4-sigma-per-object-mean detection of a
  nonzero offset is solid; per-object model discrimination on this sample is not.

---

## 5. FAILURE MODES CHECKED, EXPLICITLY

* **Shared-denominator artefacts.** Simulated under H0 with the actual error
  covariance — a coherent per-system baryonic-mass error moves `log g_bar` and
  `log|Phi_b|` by `+delta` and `log nu` by `-delta`; a distance error moves
  `log g_bar` by `-2 eps` and `log|Phi_b|` by `-eps`; 4,000 draws. The null
  expectation of `beta` is **-0.0174 +- 0.0283**, not zero, consistent with the
  existing lane's -0.0078 +- 0.0192. The null expectation of the transfer
  statistic `dRMS(M1 - M3)` is **-0.00049 +- 0.00667**, so the observed +0.01128
  sits at z = +1.76 against its own null. Reported, not assumed.
  Both axes of the fresh sample were checked the same way: `M_tot` and `M_b`
  come from different measurements (hydrostatic mass versus K-band light plus
  tabulated gas mass), so they do not share a denominator — but they DO share
  the radius and the distance, which is why the `Upsilon_K` and `M_tot`
  sensitivities are quoted as derivatives above.
* **Monotone-invariant statistics.** `d(beta)/d(q) = 0.5000` at every step over
  `q` in [0, 0.8], estimator spanning 0.400 — the gate passes for `beta`. It
  **fails for the headline transfer statistic**, where `d(rms M1)/dq = 0.0000`
  exactly. The discriminating statistic `d(dRMS)/dq` is non-zero (-0.044 to
  +0.113) and was used instead. This is a new entry for the register.
* **Refitting on the held-out set.** Never. Every arm fits on its declared
  training rows, freezes, and evaluates once. The fresh sample's coefficients
  were written to a timestamped, hashed seal before the sample was read.
* **Silent extraction failures.** Row and column counts asserted on every ingest
  (ladder 4,150 x 20 with 317 systems; Babyk 94/94/94 with identical name lists;
  K-band 94/94). The published Run R numbers are reproduced to machine precision
  before any new computation. The `M_gas` reconstruction gate caught a 2-dex
  normalisation problem that would otherwise have been invisible.
* **Non-monotone `M_b(<r)`.** 74 of 187 resolved ladder systems have a
  non-monotone `M_eff(<r) = g_bar r^2/G`, worst local decrease 15.7% — SPARC
  disk rows, cause already identified in Run R. The half-mass radius the primary
  boundary rule needs is defined as the first upward crossing; **0 systems** have
  a second crossing, so it is unambiguous everywhere.
* **Sealed holdouts.** KiDS and wide binaries: every distinct value of system,
  class, source and probe in the ladder was string-scanned, **0 matches**. The
  acquisition instructions forbade searching for either and none was searched
  for. Sources present: SPARC, X-COP, Sun+2009, Lovisari+2015, Gonzalez+2013,
  `J/A+A/690/A52`, and now Babyk+2018 with 2MASS/2MRS/HyperLeda photometry.
* **Dark-matter-dependent inputs.** None. Babyk's masses are hydrostatic.
  Humphrey+2006 was considered and **rejected** because its only tabulated total
  masses come from a parametric NFW+stars fit; the rejection is recorded with
  its raw provenance. ATLAS-3D was used for photometry only, never for its
  dynamical M/L.
* **Weak lensing.** No shear and no lensing mass enters this lane at any point.
* **Two VizieR traps found during acquisition, worth propagating.** (i) The ASU
  error marker is `#INFO` + TAB + `Error=` — testing for the string with a space
  never matches, so a missing catalogue reads as success. (ii) The
  `/ReadMe/<ID>` JSON endpoint returns "ReadMe is not found" for table-level IDs
  that exist and serve data, so it is not authoritative for existence.
  (iii) `J/A+A/601/A95` is **not** O'Sullivan's CLoGS — it is Calabro+ 2017 —
  so an existence test on that ID returns a false positive.

---

## 6. WHAT COULD NOT BE ESTABLISHED

* **Whether potential depth or the class label is the right variable, from the
  ladder.** The ceiling calculation settles this: 0.93 sigma with infinitely
  many validation clusters, ~2,100 training systems needed for 3 sigma. It is a
  structural limit of the 200-system training set, not a shortage of holdout
  data. The fresh sample breaks the tie in M1's favour, but on a sample where
  both models are badly biased.
* **A label-free measurement of `q` at the required size.** The only clean
  label-free estimate is the galaxies-only `beta = +0.0900` [+0.0032, +0.1830]
  -> `q = 0.18`, half of `q_required = 0.371` and inside Run N's within-galaxy
  bound. Nothing in this lane raises it. Every other within-class estimate is
  negative and contaminated by the Run Z shared-shape artefact.
* **Whether the group/cluster/ETG offset is gravity at all.** Run R's
  class-level systematic budget (0.199 dex, forging `q = 0.192`) is untouched,
  and section 1.3 makes it more pressing rather than less: the entire
  transferable content of the ladder result is one number — the offset of X-ray
  systems from the RAR — which is precisely the number a class-level systematic
  forges. The fresh sample adds a *third* instrument showing the same sign, at a
  larger amplitude, which is evidence for a common physical cause; it is equally
  consistent with a common systematic in X-ray hydrostatic mass estimation,
  which is the one thing every non-galaxy rung in this programme shares.

---

## FILES

| file | contents |
|---|---|
| `PREREGISTRATION.md` + `prereg_seal.json` | declarations, sealed 12:05:06Z |
| `fresh_seal.json` | frozen coefficients + protocol, sealed 12:18:16Z, one amendment 12:27:35Z |
| `ablation.json` | items 1 and 2: arms, cross-arm scoreboard, paired bootstraps, power ceiling, within-class `beta`, step-definition sensitivity, nulls, gates |
| `boundary_sensitivity.json` | item 3: four rules, spread, information content, BIC |
| `fresh_result.json` | item 4: the single evaluation, per object |
| `code/common.py` | ladder loader, models, freeze-and-evaluate, reproduction gate |
| `code/ablation.py`, `ablation2.py`, `armb.py`, `power.py`, `influence.py`, `withinclass.py`, `stepdef.py` | items 1 and 2 |
| `code/boundary.py`, `boundary2.py` | item 3 |
| `code/freeze.py`, `score_fresh.py` | item 4 |
| `data/fresh/` | Babyk+2018 tables, 2MASS K-band, raw tarballs, 11 manifests, extractors |
| `*_output.txt` | the console log of every run |
