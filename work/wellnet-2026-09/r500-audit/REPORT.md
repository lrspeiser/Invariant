# R500 tautology audit — lane `r500-audit`

**Date** 2026-09-04 · **Repo** `Invariant-main-integration` · **Lane** `work/wellnet-2026-09/r500-audit/`

All numbers below are rendered programmatically from `results.json`. KiDS and the wide binaries were not loaded, referenced or scored at any point in this lane.

## Verdict in one paragraph

The reviewer is right that both axes contain the same quantity: the X-COP `R500` is the hydrostatic-equilibrium R500, identical to `(4/3)π·500·ρ_c(z)·R500³ = M500` to 0.03%, and the excess numerator is the hydrostatic acceleration reconstructed from the same n_e and T. The naive significance of the r/R500 organisation duly evaporates: against a permutation null it sits at percentile 5.81 (correlation) and 4.18 (collapse), but that test has a false-positive rate of 0.53 and 0.70 at a nominal 5%, and the **correctly calibrated p-values are 0.58 and 0.65**. This is the eighth shared-quantity artefact, and it lands almost exactly where the seventh did (the retracted ρ_p = −0.304 sat at p = 0.563).

But the tautology is **not** the source of −0.788. Two structural facts stop it. First, R500 cancels identically from the numerator (the round trip `T_X × T500` at `RW_X × R500` returns the observed physical temperature for any R500). Second, a per-cluster normalisation is a monotone map, so it cannot change a within-cluster rank statistic at all — and 90.3% of the residual variance is within-cluster. A forward null with no true radial dependence of the excess returns -0.1417 ± 0.0403; the observed -0.7884 sits at z = -16.1. The radial trend is real.

What the audit does destroy is the phrase **"organised by r/R500"**. Physical radius with no normalisation at all gives -0.7790 against -0.7884 — the whole normalisation is worth 0.0095 in Spearman. And it cannot be worth more: `log(r/R500_i) = log r − log R500_i`, and `log R500_i` is constant within a cluster, so the two design matrices span the same column space (rank 13 = 13, residual 5.2e-14). The one well-posed version of the question — two global-parameter hypotheses, no per-object parameter — separates by **0.68σ**. There is no detection of self-similarity here, and none was possible.

## Provenance table — which mass enters which axis

| dataset | numerator (the excess) | radius on the x-axis | shared? |
|---|---|---|---|
| X-COP (12 clusters, 588 points) — **the −0.788 sample** | `g_obs` = hydrostatic `−(kT/μm_p)·dln(n_e T)/dln r / r`, from the X-COP `*_density_L1.fits` n_e and `*_temperature.fits` T | `R500` from the FITS header, comment *"Hydrostatic-equilibrium R500"* | **yes** — same X-ray data, same hydrostatic assumption |
| X-COP denominator | `g_bar` = `G(M_gas + M_star)/r²`, M_gas from the same n_e, M_star from `*_mstar.fits` HDU2 (physical kpc) | — | shares n_e with the numerator |
| CLASH (Tian+2020, `J/ApJ/896/70/fig2`) | `g_tot` from Umetsu+2016 CLASH SL+WL+magnification mass profiles | the bench uses a **fixed 1500 kpc**, not R500; lane 12's binned table used Umetsu+16 M500c | yes, for the binned table |
| Herbonnet+2020 (`tbl:masses`) | not used as a numerator here | `R500_ap`, deprojected-aperture **weak-lensing** R500 | **no** — independent of the X-ray hydrostatic mass |
| baryon-only radii (this lane) | — | `R_b,gas` (mean enclosed *gas* density = 500 ρ_c f_b) and `R_b,ne` (n_e = 1e−4 cm⁻³) | **no total mass anywhere** |

Three facts pin the X-COP entanglement exactly:

1. `M500_hdr / [(4/3)π·500·ρ_c(z)·R500³] = 1.0003` for all 12 clusters, and `ERR_M500/M500 = 3 × ERR_R500/R500` to five figures. **M500 and R500 are one number, not two.**
2. The published temperature profile is stored in scaled units — `RW_X` in `R/R500`, `T_X` in `T/T500` — so the physical temperature is only recoverable through R500 and T500. The bench's `kT500 = G M500 μ m_p/(2R500)` is verified to be X-COP's own T500: `P500_hdr / kT500` implies n_e,500 = 500 f_b ρ_c/(μ_e m_p) to 2.6%, the same constant for all 12.
3. **The cancellation lemma.** Because the header R500 *is* the R500 used to scale, `T_X × T500(R500) at RW_X × R500` returns the observed physical temperature for any R500. Verified in `tests.py` T2: scaling R500 by 0.55× and 2.30× changes `g_obs` by 1.6e−13 relative. **R500 enters the x-axis only.**

So the entanglement is *estimator-level*, not *pipeline-level*: R500 is a monotone function of the same hydrostatic mass whose excess is on the y-axis, so an upward mass fluctuation raises the excess and raises R500 together. That channel is sign-definite negative — which is why it had to be simulated rather than argued away.

The channel is directly visible in the real data: across the 12 clusters, corr(per-cluster mean residual, ln R500) = **+0.5000** (Spearman +0.4126). Positive, as the tautology requires. It is not significant at n = 12, but the sign is right and the size is what the simulation predicts — which is exactly why the naive permutation null is wrong.

A second, weaker shared quantity is present and worth recording: the numerator's `dln n_e/dln r` and the denominator's `M_gas` both come from the same n_e. That channel is included in the forward null (a common n_e realisation drives both), and its sign is the opposite of R500's — an over-estimated gas density raises `g_bar`, lowering the residual, while also raising `R_b,gas` and lowering `r/R_b`.

## Job 1 — the synthetic null

Reproduction first: the bench's own statistic on the real data gives Spearman(r/R500, RAR residual) = **-0.7884** over 588 points in 12 clusters, which is the record's −0.788 exactly.

### 1a. The forward null — no true dependence of the excess on scaled radius

Clusters are built with `y_true` a per-cluster constant, anchored so `R500_true` equals the published R500. A temperature profile is integrated from hydrostatic equilibrium, n_e and T are observed with the catalogue's own errors (plus a bin-correlated component for the L1 deprojection), M500/R500 are re-inferred from the noisy hydrostatic profile, the profiles are republished in R500/T500 units, and the bench reads them back.

| | null mean | null sd | null 5–95% | observed | percentile | z |
|---|---|---|---|---|---|---|
| S1 = Spearman(r/R500, y) | -0.1417 | 0.0403 | -0.2059 to -0.0723 | **-0.7884** | 0.00 | -16.07 |
| S1 = Spearman(r, y) | -0.1181 | 0.0407 | -0.1820 to -0.0492 | **-0.7790** | 0.00 | -16.23 |
| S2 = collapse rms | +0.1883 | 0.0183 | +0.1613 to +0.2232 | **+0.1734** | 19.00 | -0.81 |
| S3 = slope beyond 0.25 R500 | -0.1287 | 0.0478 | -0.2053 to -0.0455 | **-0.4803** | 0.00 | -7.36 |

**The number that matters.** The null mean is -0.1417, not zero — the tautology is real and it is negative, exactly as the reviewer predicted. But it accounts for only 18% of the observed correlation, and −0.788 sits 16.1σ beyond it (percentile 0.00 of 400 realisations).

With **all measurement noise switched off** the same flat truth still returns S1 = -0.2067 and a slope of -0.1394 dex/dex. That is pure deterministic bias of the analysis pipeline — see §Bugs — and it is 29% of the observed slope -0.4803.

Null sensitivity (150 realisations each). The null never approaches the observed value under any noise model tried:

| variation | S1 null mean ± sd | percentile of observed |
|---|---|---|
| white noise only | -0.1336 ± 0.0283 | 0.00 |
| fully correlated bins | -0.1633 ± 0.0550 | 0.00 |
| 2x quoted errors | -0.1281 ± 0.0419 | 0.00 |
| 4x quoted errors | -0.1619 ± 0.0536 | 0.00 |
| R500 smeared 10% extra | -0.1415 ± 0.0454 | 0.00 |
| R500 smeared 25% extra | -0.1347 ± 0.0484 | 0.00 |
| R500 from raw pointwise crossing | -0.1319 ± 0.0417 | 0.00 |
| T calibration 10% per cluster | -0.1462 ± 0.0474 | 0.00 |

### 1b. The R500-scrambling null, and why the naive version is anti-conservative

Permuting the published R500 across the 12 clusters leaves the within-cluster rank structure untouched, so it isolates exactly what R500 contributes. Over 20000 permutations the null is **-0.7724 ± 0.0102** and the observed -0.7884 sits at percentile 5.8. On the collapse statistic the null is 0.1769 ± 0.0020 and the observed 0.1734 sits at percentile 4.3.

Taken naively those read as p ≈ 0.06 and p ≈ 0.04. **They are not p-values.** Under a flat truth the same test rejects at the nominal 5% level in 53% (S1) and 70% (S2) of realisations, because the inferred R500 is correlated with the cluster's own excess. Calibrating the percentile against its own flat-truth distribution (7.5 ± 8.8 for S1, 8.1 ± 15.8 for S2) gives

> **calibrated p = 0.580 (S1) and 0.653 (S2).**

### 1c. Responsiveness

| injected slope | corr injected | corr measured | slope measured |
|---|---|---|---|
| +0.000 | -0.1188 | -0.1498 ± 0.0395 | -0.1394 ± 0.0463 |
| -0.250 | -0.9241 | -0.5894 ± 0.0308 | -0.3715 ± 0.0449 |
| -0.500 | -0.9785 | -0.8207 ± 0.0179 | -0.6048 ± 0.0438 |
| -0.750 | -0.9900 | -0.9138 ± 0.0093 | -0.8372 ± 0.0429 |
| -1.000 | -0.9943 | -0.9515 ± 0.0055 | -1.0709 ± 0.0423 |
| -1.354 | -0.9968 | -0.9734 ± 0.0032 | -1.4013 ± 0.0405 |
| -2.000 | -0.9985 | -0.9856 ± 0.0019 | -1.9991 ± 0.0396 |

`d(corr_measured)/d(corr_injected)` = **0.867** over the full range and 0.757 near the null; `d(slope_measured)/d(slope_injected)` = **0.930**. The detector is responsive, so the negative results below are real limits and not blindness.

But note the shape of that table: injecting a slope of only −0.25 already drives the injected correlation to -0.9241. **The correlation coefficient saturates and is a poor summary of this relation; the slope is the responsive statistic.** Any future claim should be quoted as a slope with an error, not as a correlation.

## Job 2 — four radial definitions on the real data

| cluster | z | R500,X (kpc) | R500,WL (kpc) | R_b,gas (kpc) | R_b,ne (kpc) |
|---|---|---|---|---|---|
| A1644 | 0.0473 | 1054 ± 20 | — | 876 | 1306 |
| A1795 | 0.0622 | 1153 ± 12 | 1450 | 1067 | 1242 |
| A2029 | 0.0766 | 1414 ± 16 | 1520 | 1317 | 1620 |
| A2142 | 0.0900 | 1424 ± 14 | 1540 | 1403 | 1849 |
| A2255 | 0.0809 | 1196 ± 26 | — | 1148 | 1555 |
| A2319 | 0.0557 | 1368 ± 17 | — | 1504 | 2208 |
| A3158 | 0.0590 | 1123 ± 16 | — | 1055 | 1353 |
| A3266 | 0.0589 | 1430 ± 31 | — | 1293 | 1452 |
| A644 | 0.0704 | 1250 ± 36 | — | 1113 | 1340 |
| A85 | 0.0555 | 1235 ± 13 | 1350 | 1180 | 1425 |
| RXC1825 | 0.0650 | 1105 ± 12 | — | 975 | 1335 |
| ZW1215 | 0.0766 | 1358 ± 31 | — | 1089 | 1309 |

R500,WL/R500,X median 1.087 (4/12 clusters, scatter 0.033 dex); R_b,gas/R500,X median 0.929 (scatter 0.035 dex); R_b,ne/R500,X median 1.179 (scatter 0.059 dex).

### The relation, all 12 clusters, 588 points

| normalising radius | S1 Spearman | scramble null | pct | S2 collapse | S3 slope, all points |
|---|---|---|---|---|---|
| `r_physical` | **-0.7790** | -0.7790 ± 0.0000 | 100.0 | 0.1755 | -0.4996 |
| `r_over_R500_X` | **-0.7884** | -0.7723 ± 0.0102 | 5.8 | 0.1734 | -0.4974 |
| `r_over_Rb_gas` | **-0.7837** | -0.7655 ± 0.0145 | 11.6 | 0.1745 | -0.4875 |
| `r_over_Rb_ne` | **-0.7764** | -0.7629 ± 0.0159 | 21.0 | 0.1770 | -0.4841 |

**Every definition gives the same answer.** The slope ranges -0.4996 to -0.4841 dex/dex across all four; the Spearman ranges -0.7884 to -0.7764. In particular the trend survives under `R_b,gas` and `R_b,ne`, which contain **no total mass of any kind** and therefore cannot be tautological. A transition visible only under the mass-derived radius would have been the suspect case; that is not what the data show.

The weak-lensing subset, where an independent mass is available:

| normalising radius | S1 | S2 | S3 |
|---|---|---|---|
| `r_physical` | -0.8684 | 0.0854 | -0.4391 |
| `r_over_R500_X` | -0.8752 | 0.0843 | -0.4399 |
| `r_over_R500_WL` | -0.8713 | 0.0852 | -0.4390 |
| `r_over_Rb_gas` | -0.8710 | 0.0850 | -0.4374 |
| `r_over_Rb_ne` | -0.8678 | 0.0868 | -0.4305 |

only 4 of 12 X-COP clusters appear in Herbonnet+2020; with 4 objects the permutation null has 24 states, so this comparison has almost no resolving power and is reported for completeness — A1795, A2029, A2142, A85, 168 points.

### Why no radial definition can be distinguished from any other here

`log(r/R_i) = log r − log R_i`, and `log R_i` is constant within cluster i, so it lies in the span of the cluster indicators. On the real X-COP design:

- rank[ indicators | log r ] = 13 of 13 columns
- rank[ indicators | log(r/R500) ] = 13
- rank[ indicators | log r | log(r/R500) ] = 13 of 14 columns — **0 new directions**
- residual of log(r/R500) on that span: max |e| = 5.2e-14, rms 4.9e-15
- the two fixed-effects fits have identical RSS (16.632046) and identical slope (-0.4817)

The variance decomposition says the same thing in physical terms: of the total residual variance 0.05595, 9.7% is between clusters and 90.3% is within. R500 can only act on the between part, and it explains r² = 0.250 of it — at most **2.43% of the total residual variance**. And the leverage is tiny: R500 spans 1054–1430 kpc, a factor 1.36, so sd(ln R500) = 0.1086 against sd(ln r) = 0.7231 inside one cluster — a ratio of 0.150.

### The well-posed version, and its power

Self-similarity is a claim that the levels are *not* free. So two global-parameter hypotheses were built — two constants each, no per-object parameter anywhere — that genuinely differ, tuned to the same S1 and the same median R500:

| hypothesis | global level | global slope | S1 | S2 | Drel = S2(r/R500)/S2(r) − 1 |
|---|---|---|---|---|---|
| H_scaled: same excess at the same r/R500 | +0.1889 | -0.5637 | -0.7925 ± 0.0492 | 0.1773 ± 0.0196 | -0.01062 ± 0.01713 |
| H_phys: same excess at the same physical r | +0.2423 | -0.5657 | -0.7898 ± 0.0493 | 0.1767 ± 0.0188 | +0.00123 ± 0.01786 |
| **observed** | | | **-0.7884** | **0.1734** | **-0.01192** |

The observed Drel sits at percentile 48.3 under H_scaled and 20.0 under H_phys. **The two hypotheses are separated by 0.68σ.** That is the entire discriminating power of the lane-12 self-similarity claim on this sample, and it is the number the verdict has to be read against.

Power of the scrambling test, for completeness (fraction of realisations in which the true R500 assignment lands at or below the 5th percentile of its own permutation null):

| truth | S1 percentile | S1 rejection rate | S2 percentile | S2 rejection rate |
|---|---|---|---|---|
| flat truth (null) | 7.9 ± 10.0 | 0.57 | 6.5 ± 8.6 | 0.65 |
| true r/R500 organisation, s=-0.5 | 1.5 ± 2.5 | 0.92 | 0.3 ± 0.6 | 1.00 |
| true r/R500 organisation, s=-1.0 | 0.4 ± 1.1 | 0.98 | 0.1 ± 0.3 | 1.00 |
| true r/R500 organisation, s=-1.354 | 0.3 ± 0.7 | 1.00 | 0.1 ± 0.3 | 1.00 |
| true PHYSICAL-radius organisation, s=-0.472 | 1.7 ± 2.8 | 0.90 | 0.2 ± 0.5 | 1.00 |

Read the first row: under a **flat** truth the test already rejects 57% / 65% of the time. And it rejects just as often under a truth organised by *physical* radius (1.00) as under one organised by r/R500 (1.00). The test measures "is R500 correlated with the excess", which is the tautology, not "is the excess organised by r/R500".

### Absolute radius reproduces everything

Injecting an excess organised purely by **physical** radius at -0.4720 dex/dex gives S1(r/R500) = -0.8006 ± 0.0199 and S1(r) = -0.7824 ± 0.0203. The advantage of normalising, S1(r/R500) − S1(r), comes out at -0.01817 ± 0.00702 — while the **observed** advantage is -0.00946. The real data show *less* advantage for r/R500 than a truth organised by physical radius does.

## Job 3 — `t = r/r_a0`, revisited

### 3a. Is `t` subject to the same tautology?

Structurally, yes — and in exactly the same way, which is worth stating because it is not the way the record framed it. `r_a0` is a per-cluster constant, so `log t = log r − log r_a0` lives in the span of the cluster indicators: rank[indicators | log r] = 13, rank[indicators | log r | log t] = 13, residual 1.3e-14. `t` adds 0 directions.

Its *shared-quantity* channel, however, has the **opposite sign**. `r_a0` is built from M_b, which also sets `g_bar`; perturbing M_b moves the residual and log t together at +1.148 dex/dex, i.e. it induces a *positive* correlation. A baryon error therefore **cannot manufacture** a negative correlation. On this axis `t` is conservative where R500 is not.

On the data `t` performs worse than raw radius: Spearman(t, y) = -0.7045 against -0.7790 for plain r and -0.7884 for r/R500.

### 3b. Bounding the extrapolation

The record flagged that the g_bar = a0 crossing is extrapolated inward past the innermost data point. It can be bounded, and the bound is wide.

- The crossing is **directly measured in only 2/12 X-COP clusters** even with no radial cut applied (max g_bar/a0 over the whole sample = 2.812).
- In 5/12 clusters the measured inner logarithmic mass slope is ≥ 2, so g_bar **turns over inward** and under a continuation of that slope `r_a0` does not exist at all.
- Freezing M_b at its innermost measured value makes g_bar rise as fast as it possibly can inward, so it is a strict upper bound on r_a0 for the measured baryons. It puts the crossing at a median of 0.50× the innermost measured radius — inside the data, and inside the BCG.
- Over the defensible family (bare gas, plus a BCG of 0.5, 1 and 2 × 10¹² M☉) the spread of log10 r_a0 is **0.87 dex (range 0.53–1.49)**, i.e. a factor 7.

So the answer to "can the extrapolation be bounded rather than merely flagged" is **yes, to about a factor 7 per cluster**, and the width is set entirely by the unmeasured BCG stellar mass. That is the same acquisition the record already identified — baryonic profiles inside ~30 kpc of cluster cores — and this lane now puts a number on what it would buy.

## Bugs found

**Bug 1 — in the pipeline under audit.** `invariant_bench._cluster_profile` interpolates the published `T/T500` profile onto the finer n_e grid with `np.interp`, which **clamps** beyond the last measured temperature bin. There `dln kT/dln r` is forced to zero and kT is held flat, so the hydrostatic g is wrong. No warning is emitted.

- **93 of 588 points (15.8%)** are affected, and they are *all* at the outer end — exactly where the claimed trend lives.
- The measured temperature grid ends at r/R500 = 0.79–1.10 (median 0.91), but the relation is quoted out to r/R500 = 1.52.
- Mean residual at the clamped points +0.0770 dex against +0.3887 dex at the clean points — a 0.31 dex difference pulling the trend down.
- Removing them moves S1 from -0.7884 to -0.7335 and the slope from -0.4803 to -0.4092, i.e. 15% of the slope.

This matters beyond this lane: the record's *"falls to a factor 1.4 at R200"* and *"extrapolated crossing at r/R500 = 1.9–2.5"* are, for X-COP, beyond any measured temperature.

**Bug 2 — in this lane's own first implementation.** `make_truth` solved for the excess normalisation and R500 jointly by fixed-point iteration. The map is near-identity, so it is neutrally stable, and mixing `np.interp` (linear in r) for the anchor with a log-linear crossing let interpolation error accumulate: R500_true drifted up to 5.8% from the value it was supposed to equal. Caught by test T4. Replaced with a single non-iterative solve using one consistent log-log interpolation; the residual is now 8.1e−04. This is the same shape as the record's note that undamped Picard never converges on the AQUAL equation.

**Bug 3 — in this lane's first test design.** The R500-scrambling permutation test looked like a clean null and is not: its false-positive rate under a flat truth is 0.53 (S1) and 0.70 (S2) at a nominal 0.05. Reporting the naive percentile would have turned a null into a 2σ detection in the wrong direction. Fixed by double calibration.

**Bug 4 — in this lane's first discriminator.** Job 2C tuned "organised by r/R500" and "organised by physical r" separately and got bit-identical output. That was not a coding error but the identity above, discovered by the test rather than by inspection. It is why `identity.py` exists.

Failure modes from the standing brief explicitly checked: shared-denominator artefacts (the whole lane); monotone-invariant statistics (T3 — the within-cluster Spearman is bit-identical across a 10× range of R500, which is the point, not a defect; and T7 confirms dS/dθ ≠ 0 for the pooled statistic); silent extraction failures (row/column/identifier assertions on every ingest; the Herbonnet table is split over two `table*` environments and is asserted at 100 rows with ordinals 1..100); clipped outer slopes (Bug 1).

## What I could NOT establish

1. **Whether the excess is organised by r/R500 rather than by physical radius.** Not "the answer is no" — the question is not answerable on this sample. The two hypotheses separate by 0.68σ, and once each cluster is allowed its own level they are algebraically the same model (rank 13 = 13). Deciding it needs clusters spanning far more than the factor 1.36 in R500 that X-COP covers — groups at 10¹³ M☉ alongside 10¹⁵ M☉ clusters, measured the same way.

2. **Whether the within-cluster radial trend is physics or hydrostatic bias.** This audit shows the trend is not manufactured by the R500 normalisation, and that 29% of the slope is manufactured by the interpolation pipeline. The remaining 71% is either real modified gravity or the outward-rising non-thermal pressure the brief already flags. Nothing here separates those two, and X-ray data alone cannot.

3. **The weak-lensing cross-check is underpowered.** Only 4 of 12 X-COP clusters appear in Herbonnet+2020 by name. With 4 objects the permutation null has 24 states. The record's *"X-COP gas × Herbonnet/LC² WL (n=7)"* must have drawn the other three from LC²/Sereno rather than Herbonnet directly; that catalogue is not in this repo and I did not fetch it.

4. **CLASH was not re-audited on its own terms.** The bench's CLASH loader discards the cluster name and normalises by a fixed 1500 kpc, not by R500, so the −0.788 statistic has no CLASH analogue. Lane 12's *binned* table did use Umetsu+16 M500c against a numerator derived from the same lensing profiles — that configuration **is** the reviewer's tautology in its pure form and it is untested here. It also inherits the Run AL.3 provenance failure. Testing it needs the Umetsu+16 per-cluster masses, which are not in this repo.

5. **The absolute size of the excess is not audited.** This lane is about the *organising variable*, not the amplitude. The factor 4.07× at matched acceleration is untouched by anything here.

6. **The forward null's noise model is mine, not X-COP's.** I used the catalogue's own NE_LOW/NE_HIGH and eT_X with an assumed bin-correlation structure. The true covariance of an L1-penalised deprojection is not published. The sensitivity table shows the conclusion does not move across the range tried, but that is a sensitivity check, not the real covariance.

## Files

| file | what |
|---|---|
| `ingest.py` | X-COP + Herbonnet ingest, assertions, baryon-only radii |
| `nullsim.py` | forward null: truth, HSE integration, observation, publication, analysis |
| `tests.py` | 9 self-tests including the cancellation lemma and monotone-invariance |
| `run_job1.py` | synthetic null, sensitivity, responsiveness |
| `run_job2.py` | four radial definitions, provenance table, scramble power |
| `run_job2b.py` | double-calibrated permutation test, discriminator |
| `run_job2c.py` | amplitude-matched discriminator (found the identity) |
| `run_job2d.py` | well-posed self-similarity test, clamped-temperature bug |
| `run_job3.py` | t = r/r_a0 tautology and extrapolation bound |
| `identity.py` | the rank computation and variance decomposition |
| `report.py` | renders this file from the JSON |
| `results.json` | every number above |

