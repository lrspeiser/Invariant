# CLASH R500 tautology audit — the open half of Run AT

Every number below is rendered from `results.json` by `report.py`; nothing
is typed by hand.  KiDS and the wide binaries were never loaded.

Reproduce with `python run_all.py`.  Code: `ingest.py`, `stats.py`,
`nullsim.py`, `run_provenance.py`, `run_cancellation.py`,
`run_structure.py`, `run_diagnostics.py`, `run_null.py`,
`run_sensitivity.py`, `run_truthcheck.py`, `tests.py`, `report.py`.
Results: `results.json` (merged) plus the seven per-job JSONs;
`ACQUISITION.json` has every source URL, arXiv version and sha256; `raw/`
holds the acquired tables; `*.log` the run transcripts.

## 0. POWER, stated before any verdict

| contrast | n | \|rho\| detectable at 80% power | power at rho = 0.5 | power at rho = 0.3 |
|---|---|---|---|---|
| 100 kpc | 18 | 0.62 | 0.57 | 0.22 |
| 200 kpc | 20 | 0.59 | 0.62 | 0.25 |
| 400 kpc | 15 | 0.67 | 0.48 | 0.19 |
| 600 kpc | 11 | 0.76 | 0.34 | 0.14 |
| between_cluster_all | 20 | 0.59 | 0.62 | 0.25 |

The sample is 84 rows over 20 clusters, and the whole lensing chain carries
only **40 free
numbers** (20 M200c + 20 c200c).  With a per-cluster level free, the radial
trend has **1
degree of freedom per cluster**, so its effective df is 20, not 64.

R500 is never reached: the outermost datum sits at
**0.357 R500** (median; range
0.144–0.593), every CLASH point is at
r/R500 <= 0.593, and R500 lies
**2.80x beyond the last
measurement**.

## 1. Job 1 — acquisition

Umetsu+2016 (ApJ 821, 116) has **no VizieR catalogue**.  Verified two ways,
both with positive controls:

```
asu-tsv -source=J/ApJ/821/116&-out.all=1   HTTP 200,
        #INFO Error=Table or Catalog not found: J/ApJ/821/116
        (identifier echoed back; no CatalogsExamined= fallback)
METAcat title=*CLASH*                      14 catalogues, Umetsu+2016 absent;
        positive control J/ApJ/896/70 (Tian+2020) IS in the list
```

The masses were obtained instead from the **arXiv e-print source** of
arXiv:1507.04385v4, whose `table2.tex` and `table3.tex` are the journal
tables verbatim.  Also acquired: Donahue+2014 CLASH-X (arXiv:1405.7876v3),
which supplies a Chandra hydrostatic r500 that is **independent of the
lensing** — the control Run AT could not build for X-COP.

| file | rows | contents |
|---|---|---|
| `raw/umetsu2016_table1.tex` | 20 | z, kT_X (Postman+2012) |
| `raw/umetsu2016_table2.tex` | 20 | M200c, c200c, r_-2 |
| `raw/umetsu2016_table3.tex` | 20 | M2500c … M200m, **M500c** |
| `raw/donahue2014_chandra_hse.tex` | 25 (20 matched) | Chandra JACO r500 |

Ingest asserts, on every table: the catalogue identifier echoed from the
`#Name:` line, the absence of `CatalogsExamined=`, the full column list
against the ReadMe, the row count, and finiteness of every value column.
`tests.py` verifies that a wrong identifier and a truncated column list are
both rejected.

## 2. The provenance table

| quantity | source | derived from | root measurement | assumes GR | assumes a halo model |
|---|---|---|---|---|---|
| excess numerator  g_obs(r) | Tian+2020 fig2.dat col log(gtot) | `G M_NFW(<r \| M200_i, c200_i) / r^2` | Umetsu+2016 joint SL+WL+magnification kappa, spherical NFW fit | yes -- kappa from GR deflection, no slip | yes -- NFW, 2 parameters |
| excess denominator  g_bar(r) | Tian+2020 fig2.dat col log(gbar) | `G [M_gas(<r) + M_star(<r)] / r^2` | Donahue+2014 Chandra X-ray gas + Cooke+2016 BCG stellar mass + Chiu+2018 satellite stars | no | no |
| x-axis radius  r | Tian+2020 fig2.dat col Rad | `fixed grid 14-30 kpc (BCG) and 100/200/400/600 kpc` | none -- a chosen radial grid | no | no |
| x-axis normaliser  R500_i | Umetsu+2016 Table 3 M500c -> overdensity definition | `M_NFW(<R500 \| M200_i, c200_i) = (4/3)pi 500 rho_c R500^3` | THE SAME Umetsu+2016 NFW fit as the numerator | yes | yes -- the same NFW |
| independent X-ray normaliser  R500_X,i | Donahue+2014 CLASH-X Chandra JACO hydrostatic r500 | `X-ray hydrostatic mass profile` | Chandra spectroscopy; no lensing anywhere | no (Newtonian HSE) | partly -- JACO parametric v_circ model, but not tied to the lensing fit |

**The numerator's lensing profile and the M500c are not merely the same
measurement — they are two functionals of the same two-parameter fit.**
Tian+2020 §2.1: *"we use these posterior distributions of the NFW
parameters to obtain well-characterized inference of M_tot(<r|M200,c200)"*.
Umetsu+2016 Table 3 caption: *"Cluster mass estimates M_3D(<r) from single
spherical NFW fits to individual surface mass density profiles"*.

Proved rather than asserted.  Regenerating Tian's published `log(gtot)` from
Umetsu's (M200c, c200c) **alone** reproduces all 84 rows to
mean +0.0035 dex, sd 0.0084 dex, max
0.0289 dex — against a published uncertainty whose
median is 0.085 dex, i.e. the
regeneration residual is **10x
smaller than the quoted error**.

And R500 is the same fit: R500 from the published M500c agrees with R500
solved directly on M_NFW(<r|M200,c200) to
0.35% (median), 1.40% (max).

## 3. Is there a cancellation lemma?  **No.**

Run AT's X-COP lemma has a precondition — the numerator must be tabulated
in R500-scaled units so the R500 that scales and the R500 that unscales are
the same number.  CLASH does not meet it: Tian tabulates absolute m/s^2
against absolute kpc.

* **Table level.** Substituting a different R500 on the x-axis moves the
  tabulated numerator by exactly 0e+00 dex.  That is a
  property of the table, not of the measurement: R500 is never an input to
  the tabulated numerator, so there is nothing to cancel — and therefore no
  lemma bounding the numerator when the underlying mass moves.
* **Estimator level.** Move the lensing mass that *generates* R500 and the
  numerator moves with it, one for one:

| mass scaled by | d log10 g_obs | d log10 R500 |
|---|---|---|
| 0.55 | -0.2596 | -0.1321 |
| 0.70 | -0.1549 | -0.0779 |
| 0.85 | -0.0706 | -0.0352 |
| 1.00 | +0.0000 | +0.0000 |
| 1.20 | +0.0792 | +0.0389 |
| 1.50 | +0.1761 | +0.0856 |
| 2.30 | +0.3617 | +0.1729 |

    d log10 g_obs / d log10 R500 = +2.021
    X-COP, same test             =  1.6e-13
    ratio                        =  6.2e+12

**There is no cancellation lemma, and the induced slope is sign-definite
negative — the sign the claim requires:**

    induced d(y)/d log10(r/R500)      = -2.021
    induced d(log a0)/d log10(r/R500) = -4.043

The dangerous parameter is the concentration, not the mass:

| parameter | d log g_obs / d log par | d log R500 / d log par | ratio |
|---|---|---|---|
| M200 | +0.4431 | +0.3333 | +1.329 |
| c200 | +0.8480 | +0.0737 | +11.508 |

Leverage: sd(log10 R500) across the 20 clusters is
0.0578 dex (span factor
1.69), while the mean *quoted* uncertainty on
log10 R500 is 0.0371 dex.  So
**64% of the R500 spread is measurement
error, and that error is shared with the numerator.**

## 4. The rank identity — it applies, and the design also isolates the tautology

CLASH's binned table **does** carry per-cluster levels: `fig2.dat` has an
`AName` column, 84 rows over 20 named clusters.  (`invariant_bench._clash()`
reads `q[2],q[3],q[4]` and discards `q[1]=AName`, which is why the record
says CLASH has no object identity.  The identity is in the file.)  So Run
AT's identity applies unchanged:

| subset | n | clusters | rank[ind\|log r] | rank[ind\|log(r/R500)] | rank[ind\|both] | R500 adds |
|---|---|---|---|---|---|---|
| all_84 | 84 | 20 | 21 | 21 | 21 of 22 | 0 |
| cluster_scale_64 | 64 | 20 | 21 | 21 | 21 of 22 | 0 |

But CLASH's radial grid is **common across clusters** — exactly 100, 200,
400 and 600 kpc, plus one per-cluster BCG radius.  The design is therefore
crossed, and it separates the two contrasts perfectly:

* **within cluster, across levels**: r varies, R500_i fixed → r and r/R500
  are the same regressor (the identity above), and a per-cluster normaliser
  cannot move a within-cluster statistic.
* **between clusters, at one level**: r is *fixed*, so log(r/R500_i) varies
  **only** through R500_i.  Any correlation of the excess with r/R500 at
  fixed r *is* a correlation with -log R500_i.

CLASH is the only sample in this programme that isolates the second
contrast, and it is the reviewer's tautology in its purest available form.

## 5. Within- versus between-cluster variance

| subset / statistic | total | between | within | between % |
|---|---|---|---|---|
| all_84/y | 0.01469 | 0.00517 | 0.00952 | **35.2%** |
| all_84/a0 | 0.04805 | 0.01977 | 0.02828 | **41.1%** |
| cluster_scale_64/y | 0.01050 | 0.00696 | 0.00354 | **66.3%** |
| cluster_scale_64/a0 | 0.04320 | 0.02541 | 0.01778 | **58.8%** |

X-COP was **90.3% within**.  On the cluster-scale CLASH points — where the
lane-12 claim lives — CLASH is
**66.3% between** (RAR
residual) and
58.8% between (a0
statistic).  **The monotone-invariance protection is not merely weaker than
X-COP's, it is inverted.**  Run AT's second structural protection is absent.

## 6. Slopes under the radial definitions

Slopes, not correlations (AT.6).  `pooled` is the OLS slope over all points;
`within (FE)` gives each cluster its own level.

**all_84, statistic = RAR residual**

| radial definition | pooled slope | within (FE) slope | Spearman | n |
|---|---|---|---|---|
| r_physical | +0.0215 | +0.0222 | -0.0457 | 84 |
| r_over_R500_lens | +0.0196 | +0.0222 | -0.0497 | 84 |
| r_over_R500_xray | +0.0193 | +0.0222 | -0.0328 | 84 |
| r_over_R500_TX | +0.0186 | +0.0222 | -0.0715 | 84 |
| r_over_Rb_gas | +0.0242 | +0.0222 | -0.0101 | 84 |
| r_over_Rb_M | +0.0136 | +0.0222 | -0.1018 | 84 |
| r_over_Rb_g | +0.0189 | +0.0222 | +0.2391 | 84 |

**all_84, statistic = log10(a0_eff/a0_can)**

| radial definition | pooled slope | within (FE) slope | Spearman | n |
|---|---|---|---|---|
| r_physical | -0.0745 | -0.0728 | -0.3110 | 84 |
| r_over_R500_lens | -0.0792 | -0.0728 | -0.3095 | 84 |
| r_over_R500_xray | -0.0747 | -0.0728 | -0.2961 | 84 |
| r_over_R500_TX | -0.0806 | -0.0728 | -0.3360 | 84 |
| r_over_Rb_gas | -0.0705 | -0.0728 | -0.2821 | 84 |
| r_over_Rb_M | -0.0781 | -0.0728 | -0.3290 | 84 |
| r_over_Rb_g | -0.0077 | -0.0728 | -0.0589 | 84 |

**cluster_scale_64, statistic = RAR residual**

| radial definition | pooled slope | within (FE) slope | Spearman | n |
|---|---|---|---|---|
| r_physical | -0.1593 | -0.1604 | -0.4537 | 64 |
| r_over_R500_lens | -0.1744 | -0.1604 | -0.4709 | 64 |
| r_over_R500_xray | -0.1394 | -0.1604 | -0.3855 | 64 |
| r_over_R500_TX | -0.1636 | -0.1604 | -0.4804 | 64 |
| r_over_Rb_gas | -0.1364 | -0.1604 | -0.3608 | 64 |
| r_over_Rb_M | -0.1817 | -0.1604 | -0.5472 | 64 |
| r_over_Rb_g | -0.0080 | -0.1604 | -0.0424 | 64 |

**cluster_scale_64, statistic = log10(a0_eff/a0_can)**

| radial definition | pooled slope | within (FE) slope | Spearman | n |
|---|---|---|---|---|
| r_physical | -0.4266 | -0.4273 | -0.5978 | 64 |
| r_over_R500_lens | -0.4592 | -0.4273 | -0.6175 | 64 |
| r_over_R500_xray | -0.3870 | -0.4273 | -0.5364 | 64 |
| r_over_R500_TX | -0.4394 | -0.4273 | -0.6266 | 64 |
| r_over_Rb_gas | -0.3933 | -0.4273 | -0.5277 | 64 |
| r_over_Rb_M | -0.4275 | -0.4273 | -0.6369 | 64 |
| r_over_Rb_g | -0.0398 | -0.4273 | -0.2298 | 64 |

Two things to read off these tables.

**(a) The within-cluster slope is bit-identical under every normaliser.**
That is the rank identity, and `tests.py` confirms it survives even a random
per-cluster normaliser spanning two decades.  Only the *pooled* slope can
differ, and it differs only through the between-cluster part.

**(b) The trend survives radii containing no total mass.**  On the a0
statistic: physical r -0.4266, r/R500_lens -0.4592, r/R500_Xray -0.3870,
baryon-only r/R_b,gas -0.3933.  As in Run AT §AT.4, the effect is not
visible only under the mass-derived radius.

Caveats on the baryon radii: R_b,gas requires extrapolating past the last
measured point for 13/20 clusters and R_b,g for 4/20;
R_b,M needs none.  And a baryon-only normaliser is **not automatically a
clean control** — see §9.

### 6b. The pure tautology contrast: fixed r, R500 varying

At fixed physical radius, log(r/R500) is exactly `const - log R500`.

| statistic | r | n | corr(excess, log R500_lens) | slope | corr(excess, log R500_Xray) |
|---|---|---|---|---|---|
| y | 100kpc | 18 | **+0.0681** | +0.126 | +0.0825 |
| y | 200kpc | 20 | **+0.1457** | +0.198 | -0.1093 |
| y | 400kpc | 15 | **+0.5274** | +0.677 | -0.4290 |
| y | 600kpc | 11 | **+0.7142** | +1.003 | -0.1597 |
| a0 | 100kpc | 18 | **+0.0692** | +0.221 | +0.0904 |
| a0 | 200kpc | 20 | **+0.2090** | +0.521 | -0.0965 |
| a0 | 400kpc | 15 | **+0.6260** | +1.557 | -0.3682 |
| a0 | 600kpc | 11 | **+0.7749** | +2.171 | -0.0681 |

The correlation with the **shared** lensing R500 rises steadily outward and
reaches +0.71 (y) and +0.77 (a0) at 600 kpc; the correlation with the
**independent** Chandra R500 does not follow it and changes sign.  That is
the shape a live tautology makes.  §7 shows it is also the shape a flat
truth makes, so it is not by itself evidence of one.

Not a selection effect: the 11 clusters reaching
600 kpc have mean log10 R500 22.6272
against 22.5902 for the rest,
and restricted to those same clusters the correlation still climbs with
radius: 
100 kpc +0.3076, 
200 kpc +0.4606, 
400 kpc +0.6370, 
then +0.7142 at 600 kpc.

The independent control is blunt, and this bounds how much it can settle:
corr(log R500_lens, log R500_Xray) = +0.2359 over 20
(+0.4252 dropping MACS0416, whose Donahue fit is
unconstrained — r_s is `nodata` with only a `<8 Mpc` limit).  Median ratio
1.089, sd of the log ratio
0.099 dex.  Dropping MACS0416 changes the pooled
X-ray-radius slope from -0.3870 to
-0.3908, so nothing here rests on it.

## 7. The forward synthetic null

Clusters are built with **no true radial dependence of the excess at all**,
projected (Abel, converged to 2e-5 against the analytic NFW Sigma), given
noise, fitted with a **spherical NFW over R <= 2.9 Mpc** — Umetsu's model and
fit range — and then published exactly as the papers publish: g_tot at the
tabulated radii and M500c → R500 from the same fit.

The noise is calibrated against **both** of Umetsu's quoted uncertainties
simultaneously: coherent amplitude 0.15, radial
tilt 0.10, independent 0.03,
giving e_M500/M500 = 0.232 (Umetsu
0.241) and e_c200/c200 =
0.291 (Umetsu
0.301).

### 7a. The pipeline manufactures the trend with no noise at all

| statistic / subset | S1 pooled slope | S2 Spearman | S3 within slope | S4 between corr |
|---|---|---|---|---|
| y/all_84 | +0.0465 | +0.0472 | +0.0594 | +0.3016 |
| y/cluster_scale_64 | -0.1440 | -0.4050 | -0.1023 | +0.3045 |
| a0/all_84 | -0.0237 | -0.1800 | +0.0017 | +0.2838 |
| a0/cluster_scale_64 | -0.3913 | -0.5578 | -0.3095 | +0.3141 |

The mechanism, measured directly — `log10(g_published / g_true)` by radius:

| radius | NFW-template misfit |
|---|---|
| BCG | -0.0991 dex |
| 100kpc | +0.0369 dex |
| 200kpc | +0.0466 dex |
| 400kpc | +0.0008 dex |
| 600kpc | -0.0528 dex |

The NFW template overstates g at 100–200 kpc and understates it at 600 kpc,
which is a manufactured negative slope.  A **flat truth with no noise**
returns -0.1440 against an observed -0.1744 on the RAR residual
(**83%**) and -0.3913 against -0.4592 on the a0 statistic
(**85%**).

Run AT found 29% of the X-COP slope was pipeline.  **For CLASH it is
83–85%.**

That figure depends on what the truth is assumed to do beyond the data, and
nothing measures that region:

| truth follows the flat-excess law out to | noise-free S1 | fraction of observed |
|---|---|---|
| 0.6 Mpc | -0.1440 | 83% |
| 1.0 Mpc | -0.1043 | 60% |
| 1.5 Mpc | -0.0620 | 36% |

### 7b. The observed values against the null

| subset / statistic | statistic | observed | null mean +- sd | z | percentile |
|---|---|---|---|---|---|
| y/all_84 | S1 | +0.0196 | +0.0321 +- 0.0146 | **-0.86** | 19.2 |
| y/all_84 | S2 | -0.0497 | -0.0286 +- 0.0571 | **-0.37** | 36.7 |
| y/all_84 | S3 | +0.0222 | +0.0560 +- 0.0113 | **-3.00** | 0.2 |
| y/all_84 | S4 | +0.0654 | +0.4644 +- 0.1544 | **-2.58** | 1.5 |
| y/all_84 | S6_600 | +0.7142 | +0.6542 +- 0.1413 | **+0.42** | 61.8 |
| y/all_84 | S7_600 | -0.1597 | -0.1060 +- 0.2355 | **-0.23** | 41.8 |
| y/all_84 | S8_slope_vs_xray_R500 | +0.0193 | +0.0441 +- 0.0137 | **-1.81** | 4.0 |
| y/cluster_scale_64 | S1 | -0.1744 | -0.1823 +- 0.0348 | **+0.23** | 58.1 |
| y/cluster_scale_64 | S2 | -0.4709 | -0.3660 +- 0.0579 | **-1.81** | 3.4 |
| y/cluster_scale_64 | S3 | -0.1604 | -0.1064 +- 0.0206 | **-2.61** | 0.6 |
| y/cluster_scale_64 | S4 | +0.1399 | +0.4781 +- 0.1454 | **-2.33** | 2.4 |
| y/cluster_scale_64 | S6_600 | +0.7142 | +0.6542 +- 0.1413 | **+0.42** | 61.8 |
| y/cluster_scale_64 | S7_600 | -0.1597 | -0.1060 +- 0.2355 | **-0.23** | 41.8 |
| y/cluster_scale_64 | S8_slope_vs_xray_R500 | -0.1394 | -0.1126 +- 0.0319 | **-0.84** | 19.4 |
| a0/all_84 | S1 | -0.0792 | -0.0519 +- 0.0288 | **-0.95** | 16.4 |
| a0/all_84 | S2 | -0.3095 | -0.1959 +- 0.0542 | **-2.10** | 1.1 |
| a0/all_84 | S3 | -0.0728 | -0.0051 +- 0.0226 | **-3.00** | 0.2 |
| a0/all_84 | S4 | +0.0599 | +0.4566 +- 0.1559 | **-2.54** | 1.5 |
| a0/all_84 | S6_600 | +0.7749 | +0.7054 +- 0.1273 | **+0.55** | 67.2 |
| a0/all_84 | S7_600 | -0.0681 | -0.0303 +- 0.2368 | **-0.16** | 44.1 |
| a0/all_84 | S8_slope_vs_xray_R500 | -0.0747 | -0.0252 +- 0.0274 | **-1.81** | 4.0 |
| a0/cluster_scale_64 | S1 | -0.4592 | -0.4661 +- 0.0683 | **+0.10** | 52.8 |
| a0/cluster_scale_64 | S2 | -0.6175 | -0.4633 +- 0.0580 | **-2.66** | 0.5 |
| a0/cluster_scale_64 | S3 | -0.4273 | -0.3179 +- 0.0413 | **-2.65** | 0.5 |
| a0/cluster_scale_64 | S4 | +0.1595 | +0.4836 +- 0.1450 | **-2.23** | 2.7 |
| a0/cluster_scale_64 | S6_600 | +0.7749 | +0.7054 +- 0.1273 | **+0.55** | 67.2 |
| a0/cluster_scale_64 | S7_600 | -0.0681 | -0.0303 +- 0.2368 | **-0.16** | 44.1 |
| a0/cluster_scale_64 | S8_slope_vs_xray_R500 | -0.3870 | -0.3317 +- 0.0638 | **-0.87** | 18.5 |

Statistic key: S1 pooled slope vs log(r/R500); S2 Spearman; S3 within-cluster
slope; S4 between-cluster corr(mean excess, log R500); S6_600 corr(excess,
log R500_lens) at r = 600 kpc; S7_600 the same against the independent
Chandra R500; S8 pooled slope against the independent X-ray radius.

### 7c. Responsiveness

| injected slope | measured pooled slope S1 | measured within slope S3 |
|---|---|---|
| +0.00 | -0.1835 | -0.1089 |
| -0.10 | -0.2017 | -0.1295 |
| -0.20 | -0.2210 | -0.1508 |
| -0.40 | -0.2618 | -0.1947 |
| -0.60 | -0.3023 | -0.2361 |

    d(S1 measured)/d(injected) = 0.199
    d(S3 measured)/d(injected) = 0.213

**The CLASH pipeline attenuates a true radial trend by a factor
5.0 in the pooled slope and 4.7 in the within-cluster
slope.**  Inverting each against its own null:

| statistic | observed | null mean | implied TRUE slope |
|---|---|---|---|
| S1 pooled | -0.1744 | -0.1823 | **+0.040 +- 0.175** |
| S3 within | -0.1604 | -0.1064 | **-0.253 +- 0.097** |

The pooled slope is **consistent with zero**; a 95% interval is
[-0.30, +0.38], so **no upper limit
tighter than |s| < 0.38 dex/dex has been set** by this
sample.  The within-cluster slope is the only statistic that is not
consistent with zero, at 2.6 sigma before the
systematic in the next subsection is applied.

### 7d. What the verdict depends on: the unmeasured outer truth

The null's **centre**, not its width, moves with how far out the flat-excess
truth is imposed before it is handed to the published NFW.  Nothing in CLASH
measures beyond 600 kpc, so this is scanned rather than chosen.

| flat truth imposed out to | S1 null mean +- sd | S1 z (y) | S1 z (a0) | S3 z (y) | S3 z (a0) |
|---|---|---|---|---|---|
| 0.6 Mpc | -0.1801 +- 0.0326 | **+0.17** | +0.04 | -2.79 | -2.83 |
| 0.8 Mpc | -0.1587 +- 0.0327 | **-0.48** | -0.67 | -4.37 | -4.40 |
| 1.0 Mpc | -0.1389 +- 0.0332 | **-1.07** | -1.31 | -5.92 | -5.96 |
| 1.5 Mpc | -0.0982 +- 0.0342 | **-2.23** | -2.53 | -9.40 | -9.45 |
| 2.5 Mpc | -0.0346 +- 0.0373 | **-3.75** | -4.11 | -14.82 | -14.87 |

The conservative choice is the data edge, 0.6 Mpc: it grants the published
NFW everywhere nothing is measured and asks only whether a flat excess
*inside* the measurements comes back out sloped.  Larger values assert the
flat-excess law over a region nothing constrains, i.e. assume part of the
hypothesis under test.  Across the whole scan the pooled slope moves from
z = -3.75 to +0.17 and the within-cluster
slope from z = -14.82 to -2.79.

Left there, that would be an undecidable systematic.  It is not — see 7e.

### 7e. Which candidate truths the data itself allows

Tian tabulates g_tot only to 600 kpc, but **Umetsu+2016 measured the
projected profile out to R = 2.9 Mpc**.  A flat-excess truth imposed past
~1 Mpc predicts a Sigma(R) that Umetsu did not observe, and is excluded by
that data whatever it does to the null.  Amplitude profiled out, so this
tests shape only:

| flat truth imposed out to | chi2/dof of its Sigma vs the published NFW | median max \|dlnSigma\| | |
|---|---|---|---|
| 0.6 Mpc | 0.39 | 0.198 | **allowed** |
| 0.8 Mpc | 0.99 | 0.288 | **allowed** |
| 1.0 Mpc | 1.80 | 0.368 | **allowed** |
| 1.5 Mpc | 5.00 | 0.617 | excluded |
| 2.5 Mpc | 16.80 | 0.948 | excluded |

**The admissible nulls are exactly the ones that put the observation inside
them.**  Over r_break <= 1.0 Mpc the pooled slope sits at
z = -1.07 to +0.17; the r_break = 2.5 Mpc case that gave
z = -3.75 predicts a lensing profile off by
0.95 in ln Sigma and is
ruled out by Umetsu's own measurement.

### 7f. What the surviving within-cluster discrepancy actually is

The published g_tot **is** `M_NFW(<r|M200,c200)` exactly (§2), so with a
per-cluster level free the within-cluster radial run of the excess is a
function of **c200_i and the baryon profile alone**.  The S3 discrepancy is
therefore, exactly and only, the statement that the published concentrations
differ from those an NFW fit to a flat-excess cluster would return:

| flat truth out to | mean c200 published | mean c200 from the null | log10 ratio | sigma |
|---|---|---|---|---|
| 0.6 Mpc | 3.87 | 3.60 | +0.0292 +- 0.0083 | 3.5 |
| 0.8 Mpc | 3.87 | 3.39 | +0.0542 +- 0.0096 | 5.7 |
| 1.0 Mpc | 3.87 | 3.20 | +0.0790 +- 0.0106 | 7.5 |

At the admissible r_break of 0.6 Mpc the whole effect is
an offset of **+0.0292 dex in log10 c200** — 8% in the
concentration, accumulated over 20 clusters.  Umetsu quotes
e_c200/c200 = 0.30 per cluster,
i.e. 0.131 dex, so the offset is
**0.22 of ONE
cluster's own quoted concentration uncertainty**.

That is a statement about NFW concentrations inside a dark-matter halo
model.  CLASH concentrations are known to carry selection and triaxiality
systematics of order this size; nothing about it is a measurement of
gravity, and it is not the kind of quantity this programme can admit.

### 7g. False-positive rates of the obvious tests

| test | nominal | measured FPR under a flat truth |
|---|---|---|
| R500_label_permutation | 0.05 | **0.855** |
| naive_OLS_t_test | 0.05 | **0.970** |

Run AT measured 0.53-0.70 for the X-COP permutation test and called it
"not a test".  The CLASH versions are worse: the R500-label permutation
rejects 86% of the time and
the naive OLS t-test
97% of the time when the truth is
flat.  Neither carries any information.

## 8. Job 3 — the dark-matter-presupposition check

* numerator = `G * M_NFW(<r | M200_i, c200_i) / r^2`
* raw shear in the repo: **False**
* reconstructible from raw shear: **False**

Three separate model layers stand between the CLASH shear catalogues and the published g_tot, and NONE of them is invertible from the table: (i) the joint SL+WL+magnification reconstruction of kappa assumes GR light bending with no gravitational slip, so kappa is a GR-derived convergence map; (ii) the deprojection to M_3D(<r) assumes spherical symmetry; (iii) the profile is a TWO-PARAMETER NFW FIT, i.e. a parametric mass model of exactly the kind standing constraint 2 excludes.  The published table contains only the output of (iii).  Recovering a raw-shear numerator would require the CLASH shear catalogues and the Umetsu+2016 kappa reconstruction, neither of which is in this repo.

Every lensing product CLASH publishes, checked table by table:

| published product | source | what it is | admissible |
|---|---|---|---|
| M200c, c200c per cluster | Umetsu+2016 Table 2 | two-parameter spherical NFW fit | **no** |
| M2500c..M200m, M(<1.5Mpc) | Umetsu+2016 Table 3 | the same NFW fit, evaluated at overdensity radii | **no** |
| M_2D(<theta), theta = 10-40 arcsec | Umetsu+2016 Table 1 | Zitrin+2015 parametric strong-lensing models, mass tied to light by construction | **no** |
| S/N of g_+ and n_mu per cluster | Umetsu+2014 Table 5 | a summary statistic, not a profile | **no** |
| non-parametric kappa(R) profiles | Umetsu+2016 Figure 'kappa' | shown in figures, NOT tabulated anywhere in the e-print; and still a GR-derived convergence map | **no** |

None of the CLASH lensing papers is on VizieR: J/ApJ/821/116 (Umetsu+2016), J/ApJ/795/163 (Umetsu+2014), J/ApJ/755/56 (Umetsu+2012) all return "Table or Catalog not found", and METAcat title=*Umetsu* returns only J/ApJ/890/148 (XXL, Umetsu+2020).
Positive control: J/ApJ/896/70 (Tian+2020) resolves and is the table this lane uses.

Quotes, from the acquired sources:

> we use the CLASH lensing constraints on the total mass profile M_tot(<r) of each individual CLASH cluster assuming a spherical NFW profile  — Tian+2020 §2.1

> Cluster parameters derived from single spherical NFW fits to individual surface mass density profiles reconstructed from combined strong-lensing, weak-lensing shear and magnification measurements  — Umetsu+2016 Table 2 caption

And the repo already says so, in a file written before this audit:

> runs/gravity/g4/cluster-lensing-exploration-v7.json, data_lineage.lensing: 'strong-lensing, weak-lensing shear, and magnification constraints converted by the source paper to spherical NFW Mtot posteriors and then gtot'; data_lineage.gr_model_independent_target = false

**Verdict: INADMISSIBLE under standing constraint 2 as currently sourced.**  Run AL.3 rejected an amplitude selected
against *interpolated* published lensing mass profiles.  CLASH is a stronger
case of the same failure: the numerator is not interpolated from a fitted
mass, it **is** a fitted mass — a two-parameter NFW whose only inputs are a
GR-derived convergence map and the assumption of spherical symmetry.  Under
standing constraint 2 that is not a raw observation.

## 9. Bugs the tests found

`tests.py`: 18/18 pass.  Six defects were found and fixed on the way, all in
this lane's own first implementation.

1. **Abel truncation.** The truth density was truncated at 5 Mpc; the
   projection then lost 6.5% of Sigma at 1.5 Mpc and **21% at the 2.9 Mpc
   outer fit radius**, biasing the fitted NFW and manufacturing a radial
   effect inside the null itself.  Converged to 2e-5 only past ~100 Mpc.
2. **Non-monotone M(<r).** The inner-sphere mass was written into `M[0]`
   after the cumulative integral instead of added as an offset, so `M[0]`
   exceeded `M[1]` by ~180x.
3. **The null's R500 population was wrong.** Imposing the constant-excess law
   all the way to 1.5 Mpc gave R500 values 1.2–1.7x the real ones and
   corr(excess, log R500) = +0.74 against +0.14 in the data.  Fixed by making
   the truth agree with the published NFW outside the measured range, where
   nothing is measured anyway.
4. **The null's c200 uncertainty was 4x too small.** A coherent amplitude
   term reproduces e_M500/M500 = 0.224 but leaves e_c200/c200 at 0.074
   against Umetsu's 0.301, because rescaling Sigma barely moves its shape.
   The width of the *within-cluster* slope null is set by exactly that shape
   uncertainty, so S3's z was ~4x too large before a radial tilt was added.

6. **A shadowed accumulator in the reporting layer.**  `report.py` built the
   document in a list named `L` and later reused `L` as a loop variable over
   radius labels; `w = L.append` kept writing to the original list while the
   final `join(L)` joined the string `"400"`.  REPORT.md came out 6 bytes
   long.  Caught only because the file size was checked -- a silent
   truncation that no assertion in the analysis would have found.

And one substantive error of reasoning, caught twice by the same check:

5. **A baryon-only normaliser is not automatically a clean control.**  The
   excess carries g_bar in its denominator, so a normaliser whose
   between-cluster variation tracks the baryon amplitude puts the same
   quantity on both axes.  I first predicted the contaminated one was R_b,g
   and used a pooled diagnostic; both were wrong.  Measured between clusters
   at fixed radius, corr(log R_norm, log g_bar) is R500_lens +0.17, R500_xray +0.05, R500_TX +0.26, Rb_gas +0.48, Rb_M -0.99, Rb_g +0.53.
   **R_b,M** — the radius enclosing a fixed baryonic mass — is
   an almost exact inverse of the baryon amplitude and must not be quoted as
   an independent control.  This is the eighth instance of the
   shared-denominator pattern, and the first found in a *control* rather than
   in a measurement.

## 10. Verdict

**The tautology is live on CLASH, and unlike X-COP nothing structural stops
it.**

| protection | X-COP (Run AT) | CLASH |
|---|---|---|
| cancellation lemma | numerator moves 1.6e-13 | **absent**; moves +2.02 dex per dex of R500 |
| monotone invariance | 90.3% within-cluster | **inverted**; 66% between-cluster |
| independent radius | none available | Donahue+2014 Chandra r500, but corr with the lensing radius is only +0.24 |
| pipeline share of the slope | 29% | **83–85%** |

**But the audit does not end where Run AT's did.**  On X-COP the trend
survived a forward null at 16 sigma with responsiveness 0.87.  On CLASH the
pooled slope does not survive at all — it lands on the null's median:

* RAR residual, pooled slope: observed -0.1744, forward null (**no true
  radial dependence at all**) -0.1823 +- 0.0348,
  **z = +0.23, percentile 58.1**.
* a0 statistic, pooled slope: observed -0.4592, forward null (**no true
  radial dependence at all**) -0.4661 +- 0.0683,
  **z = +0.10, percentile 52.8**.

The pure tautology contrast goes the same way.  The +0.71 correlation of the
excess with log R500 at 600 kpc, which looks like a smoking gun in §6b, sits
at z = +0.42 against the flat-truth
null (which produces +0.6542
+- 0.141).  It
is real, it is the tautology, and it is **also** exactly what a genuine
excess would produce — a cluster with a larger true excess has a larger true
mass and hence a larger true R500.  The two hypotheses predict the same sign
and, on this sample, the same size.  That is why the contrast cannot decide.

The one statistic that is not consistent with the null is the
**within-cluster** slope, the one Run AT's monotone-invariance argument
protects from R500 entirely:

* RAR residual, within-cluster slope: observed -0.1604, null -0.1064 +- 0.0206, **z = -2.61**.
* a0 statistic, within-cluster slope: observed -0.4273, null -0.3179 +- 0.0413, **z = -2.65**.

Across the r_break scan that runs from z = -14.82 to
-2.79, and only the small-r_break end is allowed by
Umetsu's own measured Sigma (§7e), so the honest figure is the ~2.8 sigma
at the data edge.  With responsiveness 0.21 the implied true
within-cluster slope is -0.253 +- 0.097 dex/dex.

But §7f says what that 2.8 sigma **is**: the published NFW concentrations
sit 8% (+0.0292 dex) above those an NFW fit to a flat-excess
cluster with the same baryons would return — 0.22 of ONE cluster's own quoted concentration
uncertainty, accumulated over 20.  **The whole surviving CLASH signal is a
8% offset in NFW
concentration.**  It is a statement inside a dark-matter
halo model, at a level that CLASH's known selection and triaxiality
systematics reach, and it is not a measurement of gravity.

Following standing constraint 4, none of this kills the candidate.  It
removes CLASH from the evidence for it.  The surviving cluster statement is
Run AT's X-COP one, unchanged; **CLASH adds nothing to it and should not be
quoted alongside it**, for two independent reasons, either of which is
sufficient on its own:

1. **Admissibility.** The numerator is a two-parameter NFW fit to a
   GR-derived convergence map.  Standing constraint 2 excludes it, and this
   is not a matter of degree: there is no version of the CLASH numerator that
   is not a fitted halo mass.
2. **Statistics.** Even taken at face value, the pooled radial trend sits at
   z = +0.10 (percentile
   53) against a null containing no
   radial dependence whatsoever, and the tests that made it look significant
   have false-positive rates of 0.85
   and 0.97 against a nominal 0.05.

**What would change this.**  Not more CLASH clusters — the limitation is
structural, not statistical.  It needs a numerator that is not a fitted halo
mass: the CLASH shear catalogues themselves, scored the way Run AL.5 scored
raw eFEDS/HSC shear, with the law predicting the shear rather than being
compared against somebody's NFW posterior.  Until then CLASH is a
consistency check on Umetsu+2016's NFW fits, not a measurement of gravity.

### Corrections owed to the record

* The record's within-CLASH number reproduces exactly:
  -0.3495 +- 0.0498 over n = 11,
  10/11 negative, against the quoted
  "a0 falls by -0.347 +- 0.057 dex, 10 of 11 negative".  **The number is right; the inference from it is
  not.**
* Lane 12's two "CLASH fig2" rows use **one pooled R500 for all of CLASH**
  — the quoted r/R500 of 0.073 and
  0.291 at 100 and 400 kpc imply
  1370 and 1375 kpc.  The
  per-cluster R500 values span
  1012–1715 kpc.  Under a single
  global normaliser r/R500 and r are the same variable up to a constant, so
  those rows carry no R500 information at all — which is *conservative*, and
  worth stating rather than leaving implicit.
* **Every CLASH point lies at r/R500 <= 0.59.**  The record's Lane-12 table
  places CLASH at r/R500 = 0.073 and 0.291 and then extrapolates the same
  sequence past R500; R500 itself sits
  2.8x beyond the outermost
  CLASH measurement and is a property of the NFW fit, not of the data.
* `invariant_bench._clash()` discards the `AName` column.  The record's note
  that "CLASH has no object identity in the bench" is a bench defect, not a
  data limitation — 20 named clusters are in the file.
* On the full 84 points the RAR-residual slope against log(r/R500) is
  +0.0196 — **flat**.  The CLASH radial trend exists only after the BCG
  points are dropped, or when the a0 parametrisation is used.  Whichever is
  quoted, that choice should be stated.

