# Potential-depth lane — can a galaxies-to-clusters ladder decide `F(g_bar)` vs `F(g_bar, |Phi_b|)`?

**Verdict up front.** The ladder delivers the leverage asked for: **0.766 dex**
median spread in `log|Phi_b|` inside narrow `g_bar` bins against SPARC's
**0.309 dex** — a 2.5x gain — with **3.63 dex** of range inside a single 0.25-dex
`g_bar` bin containing all six rungs, and **12,224 matched system pairs** at
`|dlog g_bar| <= 0.1` with `dlog|Phi_b| >= 1 dex`. Statistically the experiment is
easy: it detects `q = 0.115` at 3 sigma where the cluster anomaly needs
`q = 0.371`, and the fitted coefficient is `q = +0.337 +- 0.028`, 9.3 sigma from
its own shared-denominator null.

**And it still cannot decide the question.** 85.8% of those pairs are
cross-class; removing the bare class label takes the leverage from 0.768 dex to
0.286 dex, *below* what SPARC already had; a one-parameter step knowing only "is
this a galaxy?" beats the potential-depth model by dBIC = 17.6 and also wins the
frozen transfer test onto held-out clusters (0.0954 vs 0.1066 dex); and the
class-level systematic budget alone forges `q = 0.192`, only 1.9x below the
effect being tested. **The dataset is systematics-limited and label-degenerate:
it measures the galaxy/cluster boundary, not potential depth.**

## 1. What was assembled

4,150 rows, 317 systems, six rungs.

| rung | class | rows | systems | source | `g_obs` from | `M_b` from |
|---|---|---:|---:|---|---|---|
| 1 | field galaxies | 3389 | 175 | SPARC | `V_obs^2/r` | 3.6 um at Ups*=0.5/0.7 + HI x1.33 |
| 2 | small groups | 40 | 40 | SDSS `J/A+A/690/A52` | `2 sigma^2/r`, 10-14 members | **stars only**, r-band |
| 3 | poor groups | 22 | 19 | Sun+2009, Lovisari+2015, SDSS | X-ray hydrostatic | `M_gas` + calibrated stars |
| 4 | rich groups | 53 | 31 | Sun+2009, Lovisari+2015 | X-ray hydrostatic | `M_gas` + calibrated stars |
| 5 | low-mass clusters | 51 | 33 | Sun, Lovisari, Gonzalez+2013 | X-ray hydrostatic | `M_gas` + stars |
| 6 | massive clusters | 595 | 19 | X-COP (resolved), Gonzalez+2013 | hydrostatic, XMM+Planck | deprojected `n_e` + measured `M_*` |

Rungs cut on `kT`: `<1`, `1-2`, `2-4`, `>=4` keV.

**Acquisition notes and corrected premises**

- **Sun+2009 and Lovisari+2015 are not in VizieR.** `J/ApJ/693/1142` and
  `J/A+A/573/A118` both return `#INFO Error=Table or Catalog not found` at HTTP
  200. Route is the arXiv LaTeX (`0805.2320`, `1409.3845`). Also confirmed
  missing: Gonzalez+2013 `J/ApJ/778/14`, GEMS `J/MNRAS/350/1511`,
  Gastaldello+2007, Humphrey+2006, Vikhlinin+2006, Kettula+2015, Leauthaud+2010,
  Ettori+2019, Umetsu+2020, Mulroy+2019, Herbonnet+2020. Confirmed present:
  eFEDS `J/A+A/661/A7`, eRASS1 `J/A+A/685/A106`, Mantz+2015 `J/MNRAS/449/199`,
  Maughan+2008 `J/ApJS/174/117`, XXL DR2 `IX/52`.
- **The CDS FTP 200/404 existence test is broken** (anti-bot wall). The reliable
  test is the `#INFO Error=` line in the ASU response body.
- `vizier.cds.unistra.fr` was unusable during this run — sibling lanes saturated
  it and requests hung past a 20 s socket timeout. The **CfA mirror**
  `vizier.cfa.harvard.edu/viz-bin/asu-tsv` answered every probe in <= 0.5 s.
- **Three silent LaTeX extraction failures were caught by row-count assertions**,
  each of which would have passed unnoticed: Sun+2009 Table 6 is a plain
  `tabular`, not a `deluxetable`, so an AASTeX `startdata/enddata` parser returns
  the entire file; splitting the data block on `\hline` and skipping fragments
  containing it silently eats the **first row after every rule** (2 of 15
  Gonzalez rows); and testing `line.startswith("%")` before stripping comments
  eats the first data row, glued to a `%cluster z T ...` legend. Final asserted
  counts: Sun 43, Lovisari 20, Gonzalez 15 (12 with photometry, 3 blank **by
  design**), optical groups 52, resolved profiles 3,977.
- **Umetsu+2020 XXL weak lensing was deliberately excluded.** It is the only
  group-scale total mass from photons alone, and it is exactly what the standing
  brief forbids: an **NFW-fitted mass**. Named here so the omission is a
  decision, not an oversight.
- **KiDS and wide binaries were dropped by probe name before any value was
  read** — 66 rows of the 4,093-row source table.

**Stellar masses.** The only *measured* stellar masses are Gonzalez+2013's twelve
deprojected BCG+ICL+satellite masses and X-COP's seven photometric profiles.
Everything else uses a calibration fitted here on those twelve:
`log10(M*/M_gas) = +7.598 - 0.620 log10(M_gas/Msun)`, rms 0.064 dex, r = -0.949,
valid over `log10 M_gas = 12.82-13.83` and **extrapolated below it** for groups.

## 2. The identity that governs everything

With the same boundary condition `phi_rank.py` uses,
`Phi_b(r) = -[Int_r^Rmax g_bar dr' + g_bar(Rmax) Rmax]`, and
`S(r) = |Phi_b(r)| / (g_bar(r) r)`:

> **For spherical `M_b(<r)`, `S(r) >= 1` exactly**, because `M_b(<r') >= M_b(<r)`
> for `r' >= r`, so `|Phi_b(r)| = Int_r^inf G M_b(<r')/r'^2 dr' >= G M_b(<r)/r`.
> Equality iff no baryonic mass outside `r`.

Two consequences:

1. **Every single-radius row is a strict lower bound on `|Phi_b|`, not an
   estimate** — the Gonzalez systems, the optical groups, and the 20 Sun groups
   with only `r2500` all carry `S = 1` by construction and are labelled so.
2. **`log|Phi_b| = log g_bar + log r + log S`.** `|Phi_b|` carries no information
   beyond `(g_bar, r)` except the bounded shape factor. Across the ladder `log S`
   spans -0.06 to +3.27 with sd **0.387 dex**; the large values are inner disk
   points where `g_bar -> 0` while `|Phi_b|` stays finite. 316 SPARC rows have
   `S < 1` (min 0.877): these are *disk* rows, where `g_bar` is the razor-thin
   disk field rather than `GM(<r)/r^2` and the signed `V_gas` (central HI holes)
   makes `V_b^2 r/G` non-monotone. **No spherical row violates the theorem.**

## 3. Leverage

Eight equal-count `log g_bar` bins, the same binning logic as `phi_rank.py`:

| sample | median within-bin sd | max within-bin range |
|---|---:|---:|
| **full ladder** | **0.766 dex** | **3.96 dex** |
| tier 1 only | 0.814 dex | 3.96 dex |
| SPARC only, recomputed here | 0.354 dex | 2.14 dex |
| *SPARC reference, `phi_rank.json`* | *0.309 dex* | — |

Where the rungs overlap:

| `log g_bar` bin | rungs | systems | sd | range |
|---|---:|---:|---:|---:|
| -11.75 ... -11.50 | 1,3,4,5 | 89 | 0.659 | 2.64 |
| **-11.50 ... -11.25** | **1,2,3,4,5,6** | **148** | **0.785** | **3.63** |
| -11.25 ... -11.00 | 1,4,5,6 | 151 | 1.036 | 3.56 |
| -11.00 ... -10.75 | 1,5,6 | 115 | 1.041 | 3.49 |
| -10.75 ... -10.50 | 1,6 | 90 | 0.902 | 3.40 |

**Achieved dynamic range at fixed `g_bar`: 3.63 dex**, and 2.08 dex as the median
galaxy-to-massive-cluster contrast inside matched 0.1-dex `g_bar` bins (stable at
1.88-2.23 dex across ten bins).

**Matched pairs** (system-level representative points, never self-paired):

| `log g_bar` | all pairs | `dlog|Phi_b| > 1 dex` | cross-class |
|---:|---:|---:|---:|
| -12.5 | 115 | 9 | 9 |
| -12.0 | 312 | 83 | 68 |
| **-11.5** | **12,288** | **5,644** | 5,311 |
| **-11.0** | **15,227** | **5,468** | 4,578 |
| -10.5 | 6,039 | 926 | 526 |
| -10.0 | 3,201 | 94 | 0 |
| **total** | **38,785** | **12,224** | **10,492 (85.8%)** |

Pairs exist in bulk, concentrated exactly where they should be —
`g_bar/a0 ~ 0.03-0.1`, where dwarf-galaxy outskirts and cluster interiors
coincide. **85.8% straddle a class boundary.**

## 4. Collinearity — no new direction was bought

`R^2` of `log|Phi_b|` on a quadratic in `(log g_bar, log r)`, system-weighted:

| sample | `R^2` linear | **`R^2` quadratic** | residual sd |
|---|---:|---:|---:|
| full ladder | 0.875 | **0.9147** | 0.247 dex |
| tier 1 only | 0.862 | 0.9033 | 0.270 dex |
| groups + clusters only | 0.891 | 0.9188 | 0.168 dex |
| X-ray hydrostatic only | 0.952 | 0.9665 | 0.155 dex |
| *SPARC alone (`phi_rank.json`)* | *0.901* | *0.9322* | *0.218 dex* |

**The ladder's quadratic `R^2` (0.915) is no lower than SPARC's own (0.932), and
its residual scatter no larger (0.247 vs 0.218 dex).** Four decades of system
mass and three of radius bought no new direction, and by section 2 never could.

**The decisive collinearity is with radius:**
`partial corr(log|Phi_b|, log r | log g_bar) = +0.9217`, VIF = 6.6. Potential
depth at fixed acceleration *is* radius at fixed acceleration, to within the
0.387 dex `log S` provides.

## 5. The confound

| regression of `log|Phi_b|` on | `R^2` |
|---|---:|
| quadratic in `log g_bar` | 0.175 |
| `g_bar` + **class dummies** | **0.885** |
| `g_bar` + `log r` | 0.876 |

sd of `log|Phi_b|` after removing `g_bar`: **0.768 dex**; after removing `g_bar`
**and the class label**: **0.286 dex**.

Within-class leverage — one instrument, one pipeline, one systematic:

| rung | n | median within-`g_bar`-bin sd | sd after removing `g_bar` |
|---|---:|---:|---:|
| 1 field galaxies | 3389 | 0.367 | **0.356** |
| 2 small groups | 40 | 0.144 | 0.098 |
| 3 poor groups | 22 | 0.208 | 0.160 |
| 4 rich groups | 53 | 0.132 | 0.169 |
| 5 low-mass clusters | 51 | 0.150 | 0.158 |
| 6 massive clusters | 595 | 0.088 | 0.114 |

**The largest within-class leverage anywhere is SPARC's own 0.356 dex. Every
group and cluster rung has *less* internal `|Phi_b|` variation at fixed `g_bar`
than SPARC already had.** The group regime was prioritised as instructed, and it
is where the two decouple *least*: X-ray groups are observed at two overdensity
radii a fixed ratio apart, so `r` and hence `|Phi_b|` barely move at fixed
`g_bar` within the rung.

**Label control** — replace `log|Phi_b|` by a bare class index 1...6:

| second variable | `R^2` | partial corr with the `g_bar`-residual |
|---|---:|---:|
| `log|Phi_b|` | 0.8748 | +0.5270 |
| **bare class index 1...6** | **0.8663** | **+0.4778** |
| `log r` | 0.8744 | +0.5249 |

A variable carrying no physics reproduces 99% of the fit quality and 91% of the
partial correlation — the same signature as the six false positives this
programme has already retracted.

## 6. Shared-denominator null, and power

A coherent rescale `M_b -> (1+delta) M_b` moves `log g_bar` and `log|Phi_b|` by
`+delta` and `log nu_obs` by `-delta`: the error vector lies **exactly along the
degeneracy**, so controlling for `log g_bar` removes it to first order. It is
*not* removed for distance errors, which move `log g_bar` by `-2 eps` and
`log|Phi_b|` by only `-eps`.

Null simulated under H0 (truth = `quadratic(log g_bar)` with 0.250 dex
per-system intrinsic scatter drawn **independently of `|Phi_b|`**), then observed
quantities generated with the actual error covariance, 2,000 draws:

    NULL EXPECTATION of beta = -0.0078      (the naive assumption is 0)
    null sd                  =  0.0192
    null 95% interval        = [-0.0468, +0.0289]

    OBSERVED beta = +0.1716, system bootstrap [+0.1429, +0.2000]
                 -> q = 2 beta = +0.343 [+0.286, +0.400], z = +9.34 vs its null
      + class dummies: beta = +0.0914  (null -0.039 +- 0.047, z = +2.77)
      + log r:         beta = +0.0935

**Monotone-invariance gate:** injection-recovery over `q = 0...0.8` gives
`d(beta)/d(q) = 0.5000` at every step, exactly the unbiased value, estimator
spanning 0.400 in `beta`.

**Power**

| quantity | value |
|---|---:|
| `sd(beta)` under H0 / system bootstrap | 0.0192 / 0.0141 |
| **`q` detectable at 3 sigma (statistics only)** | **0.115** |
| `q` detectable at 3 sigma, class dummies also fitted | 0.282 |
| `q` required to explain the 2.43x cluster excess over 2.08 dex | **0.371** |

But the contrast is carried by the class boundary, so any **class-level**
systematic in `log nu` forges the same signal:

| systematic (one side of the boundary only) | dex |
|---|---:|
| hydrostatic mass bias, X-ray rungs vs galaxies (10-30%) | 0.080 |
| SPARC Ups* = 0.5 vs dynamical (0.4-0.55 dex disagreement measured here) | 0.150 |
| group stellar masses (Gonzalez relation extrapolated below its range) | 0.080 |
| gas clumping in clusters (measured `P_X/P_SZ`, median 6%) | 0.030 |
| non-thermal pressure support | 0.060 |
| **quadrature sum** | **0.199** |

`spurious q from systematics = 2 x 0.199 / 2.08 = 0.192` against
`q_required = 0.371`, ratio **1.93**. **The experiment is systematics-limited at
`q ~ 0.19`, not statistics-limited at `q ~ 0.12`.**

## 7. What the data actually say

`nu_obs/nu_RAR` per rung inside `0.021 < g_bar/a0 < 0.32`:

| rung | n | systems | median `log|Phi_b|` | `nu_obs/nu_RAR` | dex |
|---|---:|---:|---:|---:|---:|
| 1 field galaxies | 1948 | 164 | 9.54 | 0.940 | -0.027 |
| 2 small groups | 1 | 1 | 10.06 | 2.147 | +0.332 |
| 3 poor groups | 7 | 6 | 10.35 | 1.999 | +0.301 |
| 4 rich groups | 45 | 29 | 10.75 | 1.723 | +0.236 |
| 5 low-mass clusters | 50 | 33 | 10.96 | 1.721 | +0.236 |
| 6 massive clusters | 595 | 19 | 11.78 | 2.381 | +0.377 |

The rungs are **not monotone in `|Phi_b|`**: the boost jumps by ~2x between
galaxies and the first group rung, *falls* through the group regime, then rises
at the cluster end. A monotone `A(|Phi_b|)` cannot produce that shape. (Rungs 2-3
hold 1 and 6 systems here and should not be over-read; rungs 4-6, with 29/33/19
systems, carry the non-monotonicity on their own.)

**Model comparison**, one row per system, 252 systems, every model carrying a
free quadratic in `log g_bar`:

| model | k | rms (dex) | `R^2` | dBIC | frozen transfer to held-out clusters |
|---|---:|---:|---:|---:|---:|
| M0 RAR only | 3 | 0.2212 | 0.028 | +143.9 | 0.2917 |
| M1 `+ beta log|Phi_b|` | 4 | 0.1703 | 0.424 | +17.6 | 0.1066 |
| M2 `+ gamma log r` | 4 | 0.1735 | 0.402 | +26.9 | 0.1252 |
| **M3 `+` step: is it a galaxy?** | 4 | **0.1645** | **0.463** | **0.0** | **0.0954** |
| M4 `+` full class dummies | 8 | 0.1638 | 0.467 | +20.0 | — |

**A one-parameter step knowing only "galaxy or not" beats the potential-depth
model by dBIC = 17.6 at equal parameter count, and wins the frozen transfer
test.** `beta` was fitted on rungs 1-4 (200 systems), frozen, evaluated once on
rungs 5-6 (52 systems).

One point in favour of the hypothesis, recorded because the brief requires it:
`beta` fitted on galaxies and groups alone is **+0.1719**, on everything
**+0.1687** — the group rungs extrapolate to the clusters without adjustment, and
the implied `q = 0.34` is within 10% of the `q = 0.371` the cluster excess
requires. The potential-depth law is *not* falsified. It is simply not
distinguishable from a step at the dataset boundary.

## 8. Explicit verdict

**This dataset cannot decide the question, and the reason is structural rather
than a shortage of data.**

1. It *does* provide the leverage: 0.766 dex at fixed `g_bar` against SPARC's
   0.309, 3.63 dex of range, 12,224 matched pairs, ample statistical power
   (`q_3sigma = 0.115` vs `q_required = 0.371`).
2. But `log|Phi_b| = log g_bar + log r + log S` with `sd(log S) = 0.387 dex`, so
   at fixed `g_bar` the variable is radius:
   `partial corr(log|Phi_b|, log r | log g_bar) = +0.922`.
3. 86% of the at-fixed-`g_bar` variance is the class label; removing it leaves
   0.286 dex, less than SPARC alone. **No single rung offers more than 0.36 dex,
   and the group rungs — the priority — offer 0.10-0.17 dex, the least of any.**
4. A physics-free galaxy/not-a-galaxy step outperforms the hypothesis on BIC and
   on frozen transfer.
5. The class-level systematic budget (0.199 dex) forges `q = 0.192`, within a
   factor 1.93 of the effect being tested.

**What would decide it** — `|Phi_b|` varying by >= 1 dex at fixed `g_bar`
**within one class, one instrument, one pipeline**:

- **Resolved X-ray profiles for groups, not two overdensity radii.** The 2-radius
  tables cap within-class leverage at ~0.17 dex. Bahar+2022 eFEDS
  (`J/A+A/661/A7`) ships Vikhlinin-form `n_e(r)` parameters for **542** systems —
  already on disk — giving continuous `g_bar(r)` and a genuine `S(r)`. Its
  `M_tot` is from a scaling relation, so it needs an independent `g_obs`; pairing
  with X-GAP or CLoGS hydrostatic profiles is the obvious next lane.
- **Kill the boundary systematic instead of the boundary.** The 0.199 dex budget
  is dominated by two *measurable* terms: SPARC's Ups* (0.150) and the HSE bias
  (0.080). Pinning either to 0.03 dex halves `q_sys`.
- **A within-galaxies test of `S`.** SPARC's 0.356 dex is the largest
  within-class leverage in the ladder and is genuinely shape-driven (compact HSB
  disks vs extended LSBs at the same `g_bar`). It is small but label-free, and
  `phi_rank.py` already bounds it: partial correlation +0.018, CI
  [-0.118, +0.145] — corresponding to `|q| <= 0.29` with no class boundary
  anywhere in it. That is the cleanest constraint this programme currently holds
  on the potential-depth hypothesis, and it is already consistent with zero.

**Failure modes checked:** shared-denominator artefacts (simulated under H0 with
the actual covariance; null expectation -0.0078 +- 0.0192, not zero);
monotone-invariant statistics (`d(beta)/d(q) = 0.5000` over `q = 0...0.8`);
refitting on the held-out set (fit on rungs 1-4, freeze, evaluate once); silent
extraction failures (three found and fixed; row and column counts asserted on
every ingest); non-monotonic `M(r)` (316 `S < 1` rows found, all SPARC disk rows,
cause identified, theorem restricted to spherical systems); sealed holdouts
(KiDS and wide binaries dropped by probe name before any value was read);
dark-matter-dependent inputs (Umetsu+2020's NFW-fitted WL masses excluded).
