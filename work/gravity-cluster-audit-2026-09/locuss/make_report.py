#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Emit REPORT.md for the LoCuSS pressure-gravity test."""
import os

HERE = os.path.dirname(os.path.abspath(__file__))

REPORT = r'''# LoCuSS test of pressure-sourced gravity, `rho_eff = rho_b + 3 kappa P / c^2`

**Sample:** 41 LoCuSS clusters (Mulroy et al. 2019, MNRAS 484, 60), 40 used.
**Gravity measurement:** Subaru weak lensing (Okabe & Smith 2016), **not** gas hydrostatics.
**Verdict:** the model as written is **refuted** on this sample. Details in the final section.

Everything below is produced by `locuss_test.py`; every number is in `locuss_results.json`.

---

## 1. PRE-REGISTRATION

Fixed before any residual was computed. Not changed afterwards.

### 1.1 Stellar mass-to-light ratio

**`Upsilon_K = 0.73 M_sun / L_sun,K`**, a single **global** value applied to every cluster.
Rest-frame K band, Vega, consistent with Mulroy's own `M_K,sun = 3.39`.

Source: old passive cluster-galaxy population on a Chabrier IMF. Bell et al. (2003)
give `log10(M/L_K) = -0.206 + 0.135 (B-V)` on a diet-Salpeter scale; a red-sequence
`B-V = 0.90` gives `M/L_K = 0.82`, and `-0.05` dex to Chabrier gives 0.73. This sits
inside the range of ATLAS-3D dynamical `M/L` for early types translated to K band.

Declared secondary sensitivity grid: `Upsilon_K in {0.5, 0.6, 0.73, 0.9, 1.1}`.
This is a *nuisance*, never a per-cluster free parameter.

### 1.2 Quality cuts

1. Require `M_WL`, `kT_X_ce`, `M_gas` and `L_K_tot` all present.
2. **No dynamical-state / merger cut.** Weak lensing does not assume equilibrium; that
   is the entire reason for using this sample instead of a hydrostatic one.
3. **No weak-lensing S/N cut in the primary.** A cut at S/N = 2.0 would land exactly on
   ZwCl0857.9+2107 (S/N = 2.000) and would be a knife-edge decision.

### 1.3 Missing columns and how they are handled

| Cluster | Missing | Handling |
|---|---|---|
| Abell2697 | `L_K_tot`, `L_K_BCG` -- no UKIRT/WFCAM NIR data at all | **Dropped from the primary sample**, giving n = 40. Retained in declared secondary (a) with stellar mass imputed at the sample-median `M_star/M_gas`. |

`Y_SZA` (11 missing), `lambda` (8), `Y_X` (2) are incomplete but **not used anywhere in
this analysis**, so they impose no further loss.

### 1.4 Primary statistic -- ONE

> **Partial Spearman rank correlation `rho_p( E^2 - 1 , kT_X_ce | M_WL )`.**
> **Threshold: two-sided `p < 0.05`, p from a >= 20,000-draw permutation null.**

Chosen because `kT` and cluster mass are strongly correlated *and* `E` is built from
`M_WL`, so the raw `E`-`kT` correlation is confounded by construction. The partial
correlation at fixed `M_WL` is the one number that asks whether temperature carries
information about the gravity excess *beyond what mass already carries*.

Everything else reported below is a required supporting check, not a second primary.

### 1.5 Declared secondary analyses

(a) n = 41 with Abell2697 imputed; (b) `M_WL` S/N >= 2.5 subsample; (c) the
`Upsilon_K` grid; (d) drop the 4 largest-absolute-residual clusters; (e) the physically
derived isothermal form alongside the mandated compressed form. Plus `Omega_m = 0.27`
as a cosmology check.

---

## 2. Method -- the mass/acceleration conversion, stated exactly

This is where errors hide, so it is spelled out step by step.

1. **Aperture.** `r_500 = [ 3 M_WL / (4 pi * 500 * rho_c(z)) ]^(1/3)`, with
   `rho_c(z) = 3 H0^2 E(z)^2 / (8 pi G)`, flat LCDM, `Om = 0.3`, `H0 = 70`. This is
   *exactly* Mulroy's `r_500,WL`: they adopt the weak-lensing radius as the aperture for
   every other observable, so `M_gas` and `L_K` are the baryons inside this same sphere.
2. **Baryons.** `M_b(<r_500) = M_gas + Upsilon_K * L_K,tot`.
3. **Mass to acceleration.** `g_bar(r_500) = G M_b(<r_500) / r_500^2`. Spherical
   Newtonian. This is legitimate here because `M_WL` is itself an NFW *spherical
   enclosed* mass, so both sides of the comparison use the same geometry.
4. **RAR.** `g_obs = nu(x) * g_bar`, `x = g_bar / a0`, `nu(x) = 1/(1 - exp(-sqrt(x)))`,
   `a0 = 1.2e-10 m/s^2`.
5. **Acceleration back to mass.** `M_pred(<r_500) = g_obs * r_500^2 / G = nu(x) * M_b`.
   The `r_500^2 / G` factors cancel, so `M_pred / M_b = nu` exactly -- **but `r_500` is
   still needed to evaluate `x`**, and that is the only route by which the aperture
   enters the prediction. Check 5 quantifies it.
6. `E = M_WL / M_pred`; `t = 3 kT / (mu m_p c^2)` with `mu = 0.6`; `Y = E^2 - 1`.

### Sanity of the derived quantities

| Quantity | Range over the 40 clusters |
|---|---|
| `r_500` | 0.83 - 1.64 Mpc |
| `g_bar / a0` | 0.042 - 0.119 (deep-MOND regime throughout) |
| `nu` | 3.43 - 5.38 |
| `M_b / M_WL` | median 0.143 (cosmic value 0.157) |
| `kT_X_ce` | 3.95 - 13.38 keV |
| `E` | median **1.617**, 16-84 per cent 1.397-1.930, full 1.216 - 2.340 |

`E ~ 1.6-2` is the long-known MOND cluster residual mass discrepancy, here reproduced
with **lensing** masses rather than hydrostatic ones. The pipeline is behaving.

---

## 3. RESULTS

### Check 1 -- fit with a free intercept

`E^2 - 1 = a + kappa * t`, n = 40, OLS with 10,000-resample cluster bootstrap.

| | value | bootstrap 95% | bootstrap 68% |
|---|---|---|---|
| **intercept `a`** | **+1.218** (SE 0.539, 2.26 sigma) | **[+0.136, +2.352]** | [+0.680, +1.793] |
| **`kappa`** | **1.59e4** (SE 1.27e4) | [-1.03e4, +4.32e4] | [+2.0e3, +2.91e4] |
| residual sd | 0.957 | | |

Bootstrap **99%** interval on the intercept: `[-0.257, +2.694]`.

**The intercept is not consistent with zero at the 95 per cent level** (it excludes zero,
though not at 99 per cent). The slope is fully consistent with zero. The data want a
**large constant excess and essentially no temperature term** -- the opposite of the
model's structure.

One-sided upper limits on `kappa`: **95 per cent: 3.85e4**, 99 per cent: 4.87e4.

### Check 2 -- comparison with the X-COP value `kappa = 1.36e5`

| fit | `kappa` | ratio to X-COP | sigma from X-COP | 1.36e5 inside boot95? |
|---|---|---|---|---|
| free intercept | 1.59e4 | **0.117** | **-9.4** | **no** |
| zero intercept (model as written) | 4.35e4 | **0.320** | **-24.6** | **no** (boot95 [3.66e4, 5.14e4]) |

**Not compatible, under either fit.** The lensing-based `kappa` is 3-9x smaller than the
hydrostatic X-COP value, and the discrepancy is far outside both bootstrap intervals.

### Check 3 -- partialling out `M_WL` (the primary statistic)

The confound is real and large:

| pair | Spearman | p |
|---|---|---|
| `kT` vs `M_WL` | **+0.557** | 1.9e-4 |
| `E^2-1` vs `M_WL` | **+0.676** | 1.7e-6 |
| `E^2-1` vs `kT` (raw) | +0.190 | 0.239 |

**Primary statistic:**

> **`rho_p(E^2-1, kT | M_WL) = -0.304`**, bootstrap 95 per cent **[-0.588, +0.046]**,
> bootstrap 68 per cent [-0.446, -0.123], permutation p (two-sided) = **0.062**.

It is **negative** -- the wrong sign for the model -- and marginal.

As an interpretable effect size, `ln E = c + p ln kT + q ln M_WL` gives

- `p` (kT exponent at fixed mass) = **-0.155**, bootstrap 95 per cent **[-0.276, +0.001]**
- `q` (M_WL exponent) = +0.339 [+0.261, +0.413]
- the compressed model at `kappa = 1.36e5` **requires `p = +0.419`**.

The 95 per cent upper bound on the observed exponent (+0.001) is nowhere near the
required +0.419.

### Check 4 -- leave-one-cluster-out (all 40)

| statistic | LOO range | driven by |
|---|---|---|
| intercept | +0.950 to +1.576 | min Abell0907, max Abell0611 |
| `kappa` (free intercept) | 5.5e3 to 2.09e4 | min Abell0611, max Abell0907 |
| `kappa` (zero intercept) | 4.19e4 to 4.47e4 | min Abell0611, max Abell2219 |
| `rho_p` (primary) | **-0.382 to -0.242** (always negative) | min ZwCl0949.6+5207, max Abell0750 |
| p of `rho_p` | 0.018 to 0.143 | |

**No single cluster controls any headline number.** The intercept stays positive, `kappa`
stays an order of magnitude below 1.36e5, and `rho_p` stays negative, for every one of the
40 subsamples of 39. The primary statistic's *significance* does wander across p = 0.05,
so the marginal negative correlation is not itself a robust detection -- but its sign never
flips, and nothing ever produces a positive result.

### Check 5 -- THE APERTURE CAVEAT

All LoCuSS observables are measured inside `r_500,WL`, and `r_500,WL` scales as
`M_WL^(1/3)`, so a weak-lensing mass error `delta` drags `M_gas`, `L_K` and `kT` with it:

```
dln r500  = delta/3
dln M_gas = alpha_g * delta/3
dln L_K   = alpha_s * delta/3
dln kT    = beta_T  * delta/3
```

Mulroy quote the radial uncertainty directly: `delta_r ~ 50-130 kpc`, i.e. **4-15 per cent
of `r_500,WL`**, consistent with the 15-25 per cent `M_WL` errors in their table.

**(5a) Analytic propagation.** With `dln nu/dln x = -s u / (2(1-u))`, `s = sqrt(x)`,
`u = exp(-s)`, the response is `dln E/d delta ~ 0.44 - 0.56` across the plausible
`alpha` grid, while `dln kT/d delta = beta_T/3`. The induced correlation is

| `beta_T` | -0.4 | -0.2 | 0.0 | +0.2 |
|---|---|---|---|---|
| induced `corr(ln E, ln kT)` | -0.077 | -0.039 | 0.000 | +0.039 |

The aperture supplies **40-64 per cent of the variance in `ln E`** but only
**0.03-1.1 per cent of the variance in `ln kT`** -- the temperature channel is the
bottleneck and it is tiny. The sign of the induced correlation is the sign of `beta_T`,
and since cluster temperature profiles *decline* outward, `beta_T < 0`: **the aperture
pushes toward a negative correlation, i.e. away from the model, not toward it.**

**(5b) Three-arm Monte Carlo** (2,000 draws per configuration). A genuine null world is
constructed in which `E` carries no `kT` information at fixed `M_WL` (the observed
`E`-`M_WL` relation and scatter are preserved; the residuals are permuted; the baryons
are solved backwards so the RAR reproduces that `E` exactly). Three arms on identical
draws:

| config (`alpha_g`, `alpha_s`, `beta_T`) | CLEAN (no WL error) | OFF (WL error, no drag) | ON (WL error + drag) | **dilution** OFF-CLEAN | **aperture** ON-OFF | frac. of null reaching `rho_p <= -0.304` |
|---|---|---|---|---|---|---|
| 1.2, 1.0, -0.2 | +0.006 | -0.132 | -0.086 | -0.138 | **+0.046** | 8.5% |
| 1.2, 1.0, 0.0 | +0.006 | -0.132 | -0.075 | -0.138 | **+0.058** | 7.7% |
| 1.0, 1.0, -0.4 | +0.006 | -0.132 | -0.111 | -0.138 | **+0.022** | 11.3% |
| 1.5, 1.4, +0.2 | +0.006 | -0.132 | -0.045 | -0.138 | **+0.088** | 5.3% |
| 1.2, 1.0, -0.6 | +0.006 | -0.132 | -0.109 | -0.138 | **+0.024** | 11.0% |

**The aperture effect is bounded, and it is bounded small.** CLEAN = +0.006 validates the
null construction. The shared aperture manufactures at most **`Delta rho_p = +0.088`**
(95 per cent upper +0.229 in the most extreme configuration) and at most
**`Delta kappa` of order 1.6e4** -- nowhere near enough to fake `kappa = 1.36e5`, and of
the *wrong sign* to fake the observed negative `rho_p`.

**Two further findings from this simulation, both important:**

1. **Regression dilution dominates the aperture effect, and it is negative.** Conditioning
   on the *measured* `M_WL` rather than the true one biases `rho_p` by **-0.138**. That is
   roughly 45 per cent of the observed -0.304. So the marginally-negative primary result is
   substantially an artefact of noise in the control variable, not evidence of a genuine
   anti-correlation. Between 5.3 and 11.3 per cent of null realisations reach the observed
   value.
2. **The raw slope carries no evidence whatsoever.** In the null world -- where `E` has
   *zero* true temperature dependence by construction -- the fitted `kappa` comes out at
   **2.05e4** (CLEAN) and **1.46e4 - 2.13e4** (aperture ON). The **observed** value is
   **1.59e4**, sitting squarely inside that range. The entire measured slope is reproduced
   by the mass-temperature correlation alone. This is the confound the brief warned about,
   now quantified: it means the marginal fit in Check 1 must be read only through the
   partial statistic, which is exactly why the partial statistic was pre-registered.

### Check 6 -- permutation null, 20,000 draws

Shuffling `kT` across clusters and re-running everything.

**Null distribution of `rho_p`** (observed = **-0.3035**): mean -0.0003, sd 0.1625.

| percentile | 0.5 | 1 | 2.5 | 5 | 10 | 16 | 25 | 50 | 75 | 84 | 90 | 95 | 97.5 | 99 | 99.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `rho_p` | -0.403 | -0.369 | -0.320 | -0.270 | -0.211 | -0.165 | -0.112 | -0.001 | +0.113 | +0.164 | +0.209 | +0.266 | +0.312 | +0.372 | +0.410 |

Full null histogram (20,000 draws, bin centre / count / bar):

```
 -0.57 |                                            5
 -0.48 | #                                         16
 -0.41 | ##                                        62
 -0.35 | ####                                     155
 -0.32 | #######                                  257   <-- OBSERVED -0.3035
 -0.26 | #############                            482
 -0.19 | ######################                   830
 -0.13 | ###############################         1143
 -0.07 | ####################################    1336
 -0.00 | ########################################1494
 +0.06 | #######################################  1451
 +0.12 | ###############################         1151
 +0.19 | ######################                   834
 +0.25 | ##############                           506
 +0.31 | #######                                  265
 +0.38 | ###                                      100
 +0.44 | #                                         44
 +0.57 |                                            2
```

**Null distribution of `kappa`** (observed = **+1.59e4**): mean +50, sd 1.28e4;
2.5 / 50 / 97.5 percentiles -2.45e4 / -0.01e4 / +2.51e4. The observed value sits at
roughly the 89th percentile of its own null -- unremarkable.

| test | permutation p (two-sided) |
|---|---|
| `rho_p` (primary), plain shuffle | **0.062** |
| `rho_p`, **block-restricted** shuffle (preserves the `kT`-`M_WL` relation) | 0.035 |
| `kappa` (free intercept) | 0.216 |
| raw Spearman `E^2-1` vs `kT` | 0.237 |

One-sided p for a **positive** `rho_p` (the direction the model predicts): **0.968**.

The permutation sd of `rho_p` (0.1625) matches the analytic `1/sqrt(n-3)` = 0.1644 to
1 per cent, which validates using the analytic t-approximation inside the power simulation.

### Check 7 -- power

Type-I rates at `kappa = 0`: slope test 0.057, primary statistic 0.034. Both near the
nominal 0.05, so both null baselines are sound (the primary is slightly conservative,
which if anything *understates* the power below).

| injected `kappa` | power, slope test | power, primary statistic |
|---|---|---|
| 0 | 0.057 | 0.034 |
| 5.0e3 | 0.084 | 0.036 |
| 1.0e4 | 0.149 | 0.087 |
| 2.0e4 | 0.365 | 0.309 |
| 3.0e4 | 0.681 | 0.606 |
| **5.0e4** | **0.978** | **0.975** |
| 7.5e4 | 1.000 | 1.000 |
| 1.0e5 | 1.000 | 1.000 |
| **1.36e5 (X-COP)** | **1.000** | **1.000** |
| 2.0e5 - 5.0e5 | 1.000 | 1.000 |

**`kappa` detectable at 80 per cent power: 3.80e4 (slope test), 4.05e4 (primary
statistic).** Analytic cross-check for the slope test: 3.57e4.

**The null is informative.** 40 LoCuSS clusters detect `kappa >= ~4e4` at 80 per cent, and
would detect the X-COP value of 1.36e5 essentially every time. The absence of a signal is a
real constraint, not a shrug.

### Error-model calibration

Required before any chi-squared-like statement. Propagating the published asymmetric
errors on `M_WL`, `M_gas`, `L_K_tot` and `kT` (log-normal, as Mulroy assume) through the
full chain gives median `sigma_Y = 0.826` per cluster, against an observed residual sd of
**0.957** -- a ratio of **1.16**. The error model *is* approximately calibrated; the
measurement errors account for the bulk of the residual spread, leaving little room for
intrinsic scatter. The measurement-error-only `kappa` sd (1.40e4) also agrees with the
cluster bootstrap SE (1.27e4).

Even so, **no chi-squared, AIC or BIC is quoted anywhere in this analysis**; all intervals
above and below are bootstrap or permutation intervals.

### The "single universal kappa, no per-cluster coefficient" requirement

Solving the compressed form cluster by cluster:

- implied `kappa_i` spans **1.51e4** (RXJ2129.6+0005) to **1.41e5** (Abell0907) --
  a factor of **9.3**, sd of `ln kappa_i` = 0.55 (0.24 dex);
- `Spearman(kappa_i, kT) = -0.317` (p = 0.047);
- `Spearman(kappa_i, M_WL) = +0.341` (p = 0.031).

The coefficient the model requires to be universal instead tracks mass upward and
temperature downward.

### Sensitivity (all declared in advance)

| variant | n | intercept | `kappa` | `rho_p` |
|---|---|---|---|---|
| **primary** (`Upsilon_K` = 0.73) | 40 | +1.218 | 1.59e4 | -0.304 |
| `Upsilon_K` = 0.5 | 40 | +1.336 | 1.59e4 | -0.312 |
| `Upsilon_K` = 0.6 | 40 | +1.283 | 1.59e4 | -0.302 |
| `Upsilon_K` = 0.9 | 40 | +1.139 | 1.59e4 | -0.304 |
| `Upsilon_K` = 1.1 | 40 | +1.052 | 1.58e4 | -0.269 |
| (a) impute Abell2697 | 41 | +1.206 | 1.61e4 | -0.295 |
| (b) `M_WL` S/N >= 2.5 | 39 | +1.397 | 1.22e4 | -0.321 |
| (d) drop 4 largest residuals | 36 | +1.375 | 8.4e3 | -0.327 |
| `Omega_m` = 0.27 | 40 | +1.199 | 1.56e4 | -- |

Nothing moves. `kappa` is remarkably insensitive to `Upsilon_K` because stars are only
~10 per cent of the baryons; the intercept moves by ~0.28 across the whole grid and stays
positive throughout.

---

## 4. HOW MUCH COULD THE COMPRESSED FORM BIAS `kappa`?

The brief is right that

```
E^2 - 1 = kappa * 3 kT / (mu m_p c^2)
```

is an approximation and not the field prediction. Two distinct discrepancies, quantified.

### (i) Form -- a factor of ~3, and it is not even a constant rescaling

For genuinely isothermal gas the pressure integral is exact:
`Int_0^r 4 pi r'^2 P dr' = (kT / mu m_p) * M_gas(<r)`. Substituting into the field
equation gives

```
M_eff / M_b - 1 = kappa * t * (M_gas / M_b)
```

that is, the response is **`E - 1`, not `E^2 - 1`**, and it carries an extra factor
`f_gas = M_gas/M_b` (median 0.895, range 0.792-0.943 here). The ratio of the two
responses is `(E+1)/f_gas`:

- median **2.98**, range **2.39 - 3.82**.

So the compressed form **inflates `kappa` by roughly a factor 3** -- and because the factor
depends on `E`, it varies systematically across the sample rather than acting as a clean
constant rescaling.

Fitting the physically derived form directly:

| `E - 1 = kappa * t * f_gas` | value |
|---|---|
| `kappa`, zero intercept | 1.72e4, boot95 [1.50e4, 2.00e4] |
| intercept, if free | **+0.528 +/- 0.151** |
| `kappa`, if free intercept | 3.93e3 +/- 3.98e3 (consistent with zero) |

Same verdict: a large positive constant, no temperature term.

### (ii) Weighting -- 5 to 25 per cent

`kT_X_ce` is a *core-excised spectroscopic* temperature in `[0.15-1] r_500`, whereas the
integral needs the *gas-mass-weighted* temperature over the same sphere. For observed
cluster `T(r)` profiles the mass-weighted mean is roughly 0.80-0.95 of the core-excised
spectroscopic value, so the compressed form overstates the pressure integral and therefore
**understates `kappa` by a factor 1.05-1.25**. Small next to (i).

### Net bias, and what it does and does not affect

Combining, the compressed-form `kappa` is high by a factor of roughly **3 to 3.7** relative
to a true isothermal field-equation `kappa`. That is a large amplitude bias.

**But it cannot rescue the model**, for two reasons:

1. **A multiplicative rescaling of `kappa` cannot make a nonzero intercept vanish.** The
   intercept is the primary failure, and it is invariant to this issue.
2. **The primary statistic is invariant to the approximation.** `E^2-1` is a strictly
   monotone function of `E-1` (rank correlation exactly **1.000**), and the `kT` ranks
   agree with the `t*f_gas` ranks at **0.994**. Recomputing the pre-registered statistic in
   the exact isothermal form gives `rho_p(E-1, t*f_gas | M_WL) = -0.336` (p = 0.036),
   against -0.304 (p = 0.060) in the compressed form. The sign and the conclusion are
   unchanged.

### What this sample genuinely cannot do

**LoCuSS publishes one temperature per cluster. There is no `P(r)` and no `T(r)`.** The
isothermal-plus-proportional-density-profiles assumption underlying *both* forms above
cannot be tested here at all. **The exact test -- a radially weighted
`Int 4 pi r'^2 P(r') dr'` evaluated on radii matched to the lensing aperture -- requires
resolved pressure profiles that this dataset does not provide.** What is settled here is
the *scaling*: whether the lensing gravity excess grows with temperature the way any
version of this model requires. What is not settled is the precise value of `kappa` in a
correct radial treatment.

---

## 5. VERDICT

**The pressure model, as written, is REFUTED on this sample.**

The test demanded three things simultaneously. All three fail.

| requirement | result |
|---|---|
| **zero intercept** | **FAILS.** `a = +1.218`, bootstrap 95 per cent [+0.136, +2.352], excludes zero at 95 per cent (not at 99). LOO range +0.95 to +1.58 -- never near zero. The brief states plainly that a nonzero intercept falsifies the model as written. |
| **a single universal `kappa`** | **FAILS.** `kappa = 1.59e4` (free intercept) or 4.35e4 (zero intercept), versus the X-COP value 1.36e5: a factor 3-9 discrepancy at 9.4 and 24.6 sigma, with 1.36e5 outside both bootstrap intervals. |
| **no per-cluster coefficient** | **FAILS.** Implied `kappa_i` spans a factor 9.3, correlating negatively with `kT` (-0.317, p = 0.047) and positively with `M_WL` (+0.341, p = 0.031). |

And the temperature term the model rests on is simply **absent**:

- primary statistic `rho_p(E^2-1, kT | M_WL) = -0.304` -- **the wrong sign**, permutation
  p = 0.062, and one-sided p = 0.968 against the predicted positive direction;
- the `kT` exponent at fixed mass is `-0.155` with a 95 per cent upper bound of `+0.001`,
  against the `+0.419` the model requires at the X-COP `kappa`;
- the LOO range on `rho_p` is -0.382 to -0.242 -- negative in all 40 subsamples.

**The null is informative, not a shrug.** These 40 clusters have 80 per cent power at
`kappa ~ 4e4` and essentially 100 per cent power at the X-COP value of 1.36e5. A `kappa` of
the claimed size would have been seen. It is not there. The 95 per cent one-sided upper
limit is `kappa < 3.85e4`, an order of magnitude below the value the earlier hydrostatic
analysis reported.

**The confound is now measured rather than assumed.** In a simulated world where `E` has
*zero* true temperature dependence, the mass-temperature correlation alone produces
`kappa = 1.5-2.1e4` -- bracketing the observed 1.59e4. The raw slope contains no
information. This is precisely the pathology that made the X-COP version uninterpretable,
and it reappears here in the marginal fit; only the pre-registered partial statistic avoids
it. **This is a strong reason to distrust the X-COP `kappa = 1.36e5` itself**, which was
derived with temperature on both sides of the relation.

**The aperture caveat is bounded, and it does not rescue the model.** The shared
`r_500,WL` aperture can manufacture at most `Delta rho_p = +0.088` and `Delta kappa` of
order 1.6e4. It cannot fake a `kappa` of 1.36e5, and it works in the negative direction
for any realistic declining temperature profile. A larger, separately quantified artefact
-- regression dilution from conditioning on a noisy `M_WL`, worth **-0.138** in `rho_p` --
accounts for roughly 45 per cent of the observed negative correlation, which is why I do
*not* claim a genuine anti-correlation: 5-11 per cent of null realisations reach the
observed value. The honest reading of the primary statistic is **no temperature
dependence**, not **negative temperature dependence**.

### What *is* real here, and what it means

There is a large, robust gravity excess over baryons-plus-RAR in these clusters:
**`E` median 1.62, range 1.22-2.34**, measured with weak lensing and therefore free of
the hydrostatic circularity. That is the classic MOND cluster missing-mass problem,
independently confirmed. **It just does not scale with temperature.** Whatever sources the
cluster excess, `3 kappa P / c^2` with a universal `kappa` is not a description of it: the
excess behaves like a roughly constant multiplicative offset (plus a mild *mass*
dependence, `dlnE/dlnM_WL = +0.34`), not like a term proportional to `kT`.

### The one honest limitation

This refutes the **compressed** form, and it refutes the temperature *scaling* that any
version of the model implies. It does not evaluate the **exact** field prediction, which
needs `P(r)` integrated over radii matched to the lensing aperture. Because the primary
statistic is rank-based and the compressed-to-exact transformation is very nearly monotone
(rank correlations 1.000 and 0.994), the *qualitative* conclusion -- no temperature trend
-- carries over to the exact form. The *quantitative* value of `kappa` does not: the exact
treatment would move it by a factor of ~3-3.7, which is why the amplitude comparison
against X-COP should be read as "an order of magnitude too small" rather than as a precise
ratio.

To go further, the required dataset is resolved `P(r)` -- i.e. joint X-ray/SZ pressure
profiles -- on clusters with independent weak-lensing masses, evaluated on matched radii.
LoCuSS gives the lensing but not the profiles; X-COP gives the profiles but not an
independent gravity measurement. Neither sample alone can close it.
'''

with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8", newline="\n") as fh:
    fh.write(REPORT)
print("wrote REPORT.md, %d chars" % len(REPORT))
