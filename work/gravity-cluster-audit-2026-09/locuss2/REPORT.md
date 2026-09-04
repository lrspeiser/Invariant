# LoCuSS test of the source law `rho_eff = rho_b + 3 kappa P / c^2`, computed exactly

**Sample:** 41 LoCuSS clusters (Mulroy et al. 2019, MNRAS 484, 60), 40 used.
**Gravity measurement:** Subaru weak lensing (Okabe & Smith 2016) -- not gas hydrostatics.
**Verdict:** the ORIGINAL SOURCE LAW is **REFUTED** on this sample, and by a larger margin
than the compressed proxy showed. Details in section 9.

Everything below is produced by `locuss2_test.py`; every number is in `locuss2_results.json`.
This run supersedes `../locuss/` and does **not** reuse its `E - 1` branch.

---

## 0. What changed relative to the previous run

| | previous run | this run |
|---|---|---|
| response | `E^2 - 1 = kappa t` (compressed) | `E_pred = F(g_N_eff)/F(g_N_b)` (exact) |
| `M_gas/M_b` factor | **omitted** | carried: `delta = kappa t f_gas` |
| interpolation | deep-MOND limit for the prediction | **full RAR `nu` on both sides** |
| the `E-1` branch | explored | **discarded** -- it is an additive post-RAR acceleration, a different theory |
| primary statistic | partial Spearman conditioned on a noisy `M_WL` | **errors-in-variables regression** propagating the `M_WL` uncertainty |
| power | compressed proxy, nominal size | exact chain, **size-corrected** |

**Pipeline validation.** Running the previous run's compressed form through this code
reproduces it to the last digit: `kappa` (zero intercept) `4.35e4`, `kappa` (free intercept)
`1.59e4`, intercept `+1.218`, residual sd `0.957`, naive `rho_p = -0.3035`, `E_obs` median
`1.617`, `nu_b` range `3.43-5.38`, `r500` range `0.83-1.64` Mpc. The ingest, cosmology and
aperture handling are the validated ones, unchanged.

---

## 1. THE CHAIN, STEP BY STEP

Nothing is compressed. Per cluster:

```
1  aperture      r500 = [3 M_WL / (4 pi 500 rho_c(z))]^(1/3)      <- LENSING mass
2  stars         M_star   = Upsilon_K L_K,tot ,  Upsilon_K = 0.73
3  baryons       M_b      = M_gas + M_star
4  thermal       t        = 3 kT / (mu m_p c^2) ,  mu = 0.6
5  pressure mass DM_P     = (3 kappa/c^2) Int_0^r 4 pi r'^2 P dr' = kappa t M_gas
6  Newtonian     g_N_b    = G M_b / r500^2 ;  g_N_eff = G (M_b + DM_P) / r500^2
7  acceleration  g        = F(g_N) = nu(g_N/a0) g_N ,  nu(x) = 1/(1-exp(-sqrt(x)))
8  PREDICTION    E_pred   = F(g_N_eff) / F(g_N_b)
9  OBSERVATION   E_obs    = M_WL / (nu(x_b) M_b) = g_WL / F(g_N_b)
```

### 1.1 Step 5 is the only algebraic substitution, and it is exact

For ideal isothermal gas `P = rho_g kT/(mu m_p)`, so
`Int_0^r 4 pi r'^2 P dr' = (kT/mu m_p) M_gas(<r)` **independently of the density profile**.
Verified numerically on a beta-model (`r_c = 0.2` Mpc, `beta = 0.65`, 6e5-point grid):
ratio = **1.000000000000**. No profile-shape assumption is smuggled in.

What the substitution *does* assume is isothermality, and that `kT_X_ce` stands in for the
gas-mass-weighted temperature. For Vikhlinin-like declining profiles the exact integral is
`T_mw/T_ce` times the substituted value, with **`T_mw/T_ce = 0.88 - 0.94`**. That is a
multiplicative amplitude effect only; it cannot change a sign or a per-cluster ordering, and
section 8.4 shows it is nowhere near large enough to matter.

### 1.2 Where the clusters sit in `x` -- and why the full `nu` matters

| | min | median | max |
|---|---:|---:|---:|
| `x_b = g_N_b/a0` | 0.0423 | 0.0725 | 0.1185 |
| `nu(x_b)` | 3.43 | 4.19 | 5.38 |
| `delta = DM_P/M_b` at `kappa = 1.36e5` | 2.27 | 4.70 | 8.68 |
| **`x_eff = g_N_eff/a0`** at `1.36e5` | **0.175** | **0.371** | **0.914** |
| `nu(x_eff)` at `1.36e5` | 1.62 | 2.32 | 2.92 |

**The baryons alone sit deep in the MOND regime, but the pressure term does not leave them
there.** At the X-COP `kappa` it multiplies the source by 3.3-9.7, pushing `x_eff` into the
RAR transition region where `nu` is between 1.6 and 2.9. The local logarithmic slope of `F`
is therefore steeper than the deep-MOND `1/2`, so:

> **`E_pred` median = 2.848 with the full `nu`, against 2.387 in the deep-MOND limit.**
> Using the exact interpolation makes the model over-predict by a *further 19 per cent*.

This is the first place the corrected calculation moves against the model rather than for it.

---

## 2. PER-CLUSTER CHAIN TABLE

Masses in `1e14 Msun`, `kT` in keV, sorted by temperature. `sig` is the propagated
1-sigma uncertainty on `ln(E_obs/E_pred)`. `E_pred` and `DM_P` are at the fixed X-COP
`kappa = 1.36e5`.
| cluster | z | M_b | M_gas | f_gas | kT | DM_P | g_Nb/a0 | g_Neff/a0 | nu_b | E_pred | E_obs | ln(Eo/Ep) | sig |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Abell2204 | 0.152 | 1.37 | 1.23 | 0.895 | 13.38 | 11.93 | 0.0766 | 0.7420 | 4.14 | 4.055 | 1.746 | -0.842 | 0.149 |
| Abell0611 | 0.288 | 0.79 | 0.69 | 0.874 | 11.96 | 5.98 | 0.0514 | 0.4405 | 4.93 | 3.586 | 2.340 | -0.427 | 0.188 |
| Abell0697 | 0.282 | 1.32 | 1.22 | 0.927 | 11.06 | 9.78 | 0.0953 | 0.8033 | 3.77 | 3.783 | 1.556 | -0.888 | 0.199 |
| Abell2390 | 0.233 | 1.79 | 1.66 | 0.929 | 10.79 | 12.98 | 0.1014 | 0.8380 | 3.67 | 3.758 | 1.607 | -0.850 | 0.128 |
| Abell1835 | 0.253 | 1.59 | 1.43 | 0.901 | 10.16 | 10.53 | 0.0889 | 0.6787 | 3.88 | 3.508 | 1.783 | -0.677 | 0.118 |
| Abell2219 | 0.228 | 1.84 | 1.68 | 0.914 | 10.13 | 12.33 | 0.1185 | 0.9137 | 3.43 | 3.648 | 1.370 | -0.979 | 0.138 |
| Abell1914 | 0.171 | 1.21 | 1.11 | 0.919 | 10.06 | 8.09 | 0.0799 | 0.6154 | 4.06 | 3.488 | 1.597 | -0.782 | 0.156 |
| Abell2537 | 0.297 | 0.97 | 0.83 | 0.854 | 9.93 | 5.97 | 0.0663 | 0.4738 | 4.40 | 3.260 | 2.001 | -0.488 | 0.249 |
| Abell1689 | 0.183 | 1.48 | 1.31 | 0.886 | 9.71 | 9.22 | 0.0720 | 0.5206 | 4.25 | 3.312 | 2.000 | -0.504 | 0.098 |
| Abell2552 | 0.300 | 1.14 | 1.00 | 0.875 | 9.69 | 7.02 | 0.0881 | 0.6294 | 3.89 | 3.351 | 1.609 | -0.733 | 0.249 |
| RXCJ1504.1-0248 | 0.215 | 1.14 | 1.06 | 0.934 | 9.55 | 7.34 | 0.0874 | 0.6521 | 3.91 | 3.447 | 1.474 | -0.849 | 0.208 |
| ZwCl1021.0+0426 | 0.291 | 1.02 | 0.95 | 0.934 | 9.04 | 6.22 | 0.0950 | 0.6758 | 3.77 | 3.367 | 1.386 | -0.887 | 0.155 |
| Abell1451 | 0.199 | 1.16 | 1.02 | 0.882 | 8.87 | 6.56 | 0.0759 | 0.5061 | 4.15 | 3.154 | 1.700 | -0.618 | 0.127 |
| Abell1423 | 0.213 | 0.69 | 0.62 | 0.896 | 8.20 | 3.68 | 0.0689 | 0.4354 | 4.33 | 3.021 | 1.480 | -0.713 | 0.181 |
| Abell0267 | 0.230 | 0.79 | 0.70 | 0.883 | 8.03 | 4.07 | 0.0684 | 0.4198 | 4.35 | 2.962 | 1.625 | -0.600 | 0.198 |
| Abell1763 | 0.228 | 1.77 | 1.61 | 0.910 | 7.67 | 8.95 | 0.0763 | 0.4624 | 4.14 | 2.964 | 2.155 | -0.318 | 0.149 |
| Abell0068 | 0.255 | 0.89 | 0.80 | 0.898 | 7.66 | 4.44 | 0.0686 | 0.4106 | 4.34 | 2.915 | 1.764 | -0.502 | 0.131 |
| Abell0773 | 0.217 | 1.26 | 1.10 | 0.872 | 7.50 | 5.98 | 0.0729 | 0.4184 | 4.23 | 2.852 | 1.890 | -0.412 | 0.127 |
| Abell2261 | 0.224 | 1.42 | 1.23 | 0.864 | 7.50 | 6.69 | 0.0792 | 0.4510 | 4.08 | 2.856 | 1.851 | -0.433 | 0.127 |
| Abell2009 | 0.153 | 0.76 | 0.69 | 0.909 | 7.44 | 3.72 | 0.0568 | 0.3352 | 4.71 | 2.847 | 1.785 | -0.467 | 0.191 |
| ZwCl0949.6+5207 | 0.214 | 0.46 | 0.40 | 0.874 | 7.31 | 2.12 | 0.0423 | 0.2380 | 5.38 | 2.710 | 2.018 | -0.295 | 0.185 |
| Abell2111 | 0.229 | 0.79 | 0.68 | 0.859 | 7.21 | 3.55 | 0.0727 | 0.3992 | 4.23 | 2.770 | 1.520 | -0.600 | 0.248 |
| RXJ1720.1+2638 | 0.164 | 0.78 | 0.71 | 0.909 | 7.14 | 3.67 | 0.0699 | 0.3988 | 4.30 | 2.830 | 1.469 | -0.656 | 0.215 |
| Abell2631 | 0.278 | 1.07 | 0.97 | 0.906 | 6.91 | 4.86 | 0.0955 | 0.5290 | 3.76 | 2.849 | 1.393 | -0.715 | 0.271 |
| Abell0521 | 0.247 | 1.18 | 1.08 | 0.913 | 6.72 | 5.26 | 0.1061 | 0.5775 | 3.60 | 2.843 | 1.266 | -0.809 | 0.153 |
| ZwCl1231.4+1007 | 0.229 | 0.76 | 0.69 | 0.909 | 6.56 | 3.28 | 0.0745 | 0.3964 | 4.19 | 2.721 | 1.451 | -0.629 | 0.280 |
| Abell0963 | 0.205 | 0.91 | 0.80 | 0.881 | 6.53 | 3.79 | 0.0666 | 0.3441 | 4.40 | 2.649 | 1.743 | -0.419 | 0.129 |
| Abell1682 | 0.226 | 0.98 | 0.84 | 0.855 | 6.46 | 3.93 | 0.0639 | 0.3196 | 4.48 | 2.587 | 1.936 | -0.290 | 0.186 |
| Abell0209 | 0.206 | 1.59 | 1.44 | 0.906 | 6.39 | 6.67 | 0.0796 | 0.4135 | 4.07 | 2.692 | 1.908 | -0.344 | 0.125 |
| RXJ2129.6+0005 | 0.235 | 0.72 | 0.67 | 0.924 | 5.94 | 2.88 | 0.0865 | 0.4308 | 3.92 | 2.636 | 1.216 | -0.774 | 0.297 |
| Abell0115 | 0.197 | 0.98 | 0.87 | 0.890 | 5.93 | 3.74 | 0.0845 | 0.4077 | 3.96 | 2.579 | 1.391 | -0.618 | 0.251 |
| Abell0781 | 0.298 | 0.86 | 0.74 | 0.859 | 5.92 | 3.17 | 0.0872 | 0.4086 | 3.91 | 2.537 | 1.410 | -0.587 | 0.361 |
| Abell0907 | 0.167 | 1.03 | 0.93 | 0.902 | 5.66 | 3.81 | 0.0526 | 0.2471 | 4.88 | 2.459 | 2.290 | -0.071 | 0.129 |
| Abell0586 | 0.171 | 0.86 | 0.73 | 0.845 | 5.56 | 2.94 | 0.0604 | 0.2660 | 4.59 | 2.382 | 1.819 | -0.270 | 0.180 |
| RXCJ2102.1-2431 | 0.188 | 0.52 | 0.46 | 0.890 | 5.32 | 1.77 | 0.0569 | 0.2523 | 4.71 | 2.383 | 1.524 | -0.447 | 0.183 |
| Abell0141 | 0.230 | 0.71 | 0.60 | 0.845 | 4.78 | 2.08 | 0.0702 | 0.2758 | 4.30 | 2.238 | 1.496 | -0.403 | 0.184 |
| ZwCl1454.8+2233 | 0.258 | 0.59 | 0.54 | 0.918 | 4.74 | 1.86 | 0.0678 | 0.2815 | 4.36 | 2.312 | 1.457 | -0.462 | 0.320 |
| Abell0291 | 0.196 | 0.53 | 0.47 | 0.892 | 4.03 | 1.37 | 0.0516 | 0.1861 | 4.92 | 2.091 | 1.721 | -0.195 | 0.175 |
| ZwCl0857.9+2107 | 0.235 | 0.36 | 0.34 | 0.943 | 3.97 | 0.98 | 0.0606 | 0.2250 | 4.58 | 2.146 | 1.253 | -0.538 | 0.439 |
| Abell0750 | 0.163 | 0.69 | 0.55 | 0.792 | 3.95 | 1.57 | 0.0536 | 0.1753 | 4.84 | 1.975 | 1.832 | -0.075 | 0.196 |

`M_star`, `r500`, `t`, `nu_eff`, per-cluster `kappa_i`, deep-limit `E_pred` and both sigma
columns are in `locuss2_results.json:per_cluster_chain`.

**Read the last two numeric columns.** `E_pred` exceeds `E_obs` for **all 40 clusters**, and
the residual is largest for the hottest clusters -- the opposite of what a temperature-sourced
term needs.

---

## 3. THE FIXED-`kappa` TEST -- `kappa = 1.36e5`, NO FITTED INTERCEPT

Full uncertainty propagation: 20,000 log-normal draws per cluster on `M_WL`, `M_gas`,
`L_K_tot` and `kT` pushed through the entire chain, so the correlation induced by `M_gas`
appearing on **both** sides (in `M_b` and in `DM_P`) is carried correctly.

| quantity | value |
|---|---|
| `E_obs` median | **1.617** (16-84%: 1.397-1.930; full 1.216-2.340) |
| `E_pred` median | **2.848** (full 1.975-4.055) |
| `E_pred/E_obs` | median **1.756**, range 1.074-2.662 |
| mean residual `ln(E_obs/E_pred)` | **-0.554** |
| weighted mean residual | **-0.5436**, bootstrap 95% [-0.627, -0.463] |
| bootstrap SE | 0.0417 -> **-13.0 sigma** |
| residual sd | 0.226 |
| mean pull, sd of pull | **-3.15**, 1.67 |
| clusters within 2 sigma | **10 of 40** |
| clusters over-predicted | **40 of 40** |
| in `E^2-1` space | mean observed **1.87** vs mean predicted **7.73**; median ratio **4.73** |

Under the alternative (aperture-drag) error model of section 7.1 the same test gives
**-0.548 at -12.6 sigma**. The verdict does not depend on the error model.

> **At the fixed X-COP `kappa` the source law over-predicts the lensing excess by a factor
> 1.76 in `E`, i.e. 4.7 in `E^2-1`, for every cluster in the sample, at 13 sigma.**

The `M_gas/M_b` factor the previous run omitted **worsens** this: `f_gas` has median 0.895,
so restoring it *lowers* `DM_P` by 10.5 per cent, but the full `nu` raises the prediction by
19 per cent, and the net of the two corrections is that the exact chain over-predicts by more
than the compressed proxy did.

---

## 4. FREE FITS

Weights are `1/sigma^2(ln E_obs)` and are **`kappa`-independent by construction**, so no
parameter-dependent-weight pathology enters. `kappa` is profiled on a grid whose lower edge
(-1.32e4) keeps `1 + delta > 0.05` for every cluster; every optimum below is interior.

| model | `kappa` | bootstrap 95% | ratio to X-COP | 1.36e5 inside 95%? |
|---|---:|---|---:|---|
| **no intercept** (the law as written) | **4.075e4** | [3.375e4, 4.850e4] | **0.300** | **no** (-25.2 sigma) |
| + free constant in `ln E` | **-2.0e3** | [-7.4e3, +7.0e3] | -0.015 | **no** |
| + free constant + free `ln M_WL` slope | **-6.1e3** | [-9.6e3, -1.30e3] | -0.045 | **no** |

Nuisance coefficients: the free constant is `+0.603` in `ln E` (a constant factor 1.83);
with the mass term the constant is `+0.647` and the `ln M_WL` slope is `+0.313`.

**The two channels disagree with each other.** Forced through the zero intercept the fit
returns `4.08e4`, driven entirely by matching the mean amplitude. Given a free constant, the
preferred `kappa` collapses to zero or below. A single parameter that has to explain both the
amplitude and the cluster-to-cluster variation cannot do both here.

### 4.1 Deep-limit cross-check, and the size of the missing factor

Fitting the deep-MOND form `E^2-1 = kappa t f_gas` linearly:

| | `kappa`, zero intercept | `kappa`, free intercept | intercept | residual sd |
|---|---:|---:|---:|---:|
| **with `f_gas` (correct)** | **4.815e4** | 1.32e4 | +1.386 | 0.966 |
| without `f_gas` (previous run) | 4.350e4 | 1.59e4 | +1.218 | 0.957 |

The omitted `M_gas/M_b` factor was worth exactly `1/0.895 = 1.107` in `kappa`. It is a
**10.7 per cent** correction. It does not rescue anything: `4.8e4` is still 0.354 of `1.36e5`.

### 4.2 The universality requirement

Solving `E_pred_i(kappa_i) = E_obs_i` exactly, cluster by cluster:

- `kappa_i` spans **1.38e4** (RXJ2129.6+0005) to **1.17e5** (Abell0907) -- a factor **8.45**;
  sd of `ln kappa_i` = 0.552 (0.24 dex);
- `Spearman(kappa_i, kT) = -0.342` (p = 0.031);
- `Spearman(kappa_i, M_WL) = +0.312` (p = 0.050).

The coefficient required to be universal instead runs *down* with temperature and *up* with
mass -- the same signature the compressed analysis found, unchanged by the exact treatment.

---

## 5. STATISTICS: ERRORS-IN-VARIABLES INSTEAD OF A PARTIAL RANK CORRELATION

### 5.1 Why the old statistic had to go

`E_obs` is **built from** `M_WL`. Propagating the published errors through the chain, the
measurement-error correlation between `ln E_obs` and `ln M_WL` is

> **median +0.960** (range +0.922 to +0.971).

Conditioning on the *measured* `M_WL` therefore does not condition on the true one, and any
partial statistic is biased. Quantified below: the naive estimator's expectation under a true
null is not zero.

### 5.2 The model

Per cluster, `y_i = (ln E_obs, ln M_WL, ln kT)`, latents `xi_i`, and

```
measurement   y_i = xi_i + e_i ,  e_i ~ N(0, C_i)   C_i KNOWN and non-diagonal,
                                                    measured by MC through the exact chain
structure     ln E_true = c + p tau + q m + eps ,   eps ~ N(0, s^2)
population    (m, tau) ~ N(mu, Sigma)
marginal      y_i ~ N(mean, B Sigma B' + s^2 e1 e1' + C_i)
```

fitted by maximum likelihood (which constrains the covariance to stay valid), with a
closed-form method-of-moments solution as a diagnostic. Inference: cluster bootstrap plus a
parametric null. **The target parameter `p = d ln E / d ln kT` at fixed mass is exactly the
quantity the source law makes a prediction about.**

### 5.3 Estimator validation

Simulated with the real `C_i` and the fitted population, sweeping the true `p`:

| true `p` | naive OLS returns | **EIV MLE returns** |
|---:|---:|---:|
| -0.300 | -0.320 | **-0.275** |
| -0.159 | -0.224 | **-0.148** |
| 0.000 | **-0.126** | **-0.005** |
| +0.159 | -0.017 | **+0.178** |
| +0.300 | +0.069 | **+0.308** |
| +0.580 | +0.258 | **+0.596** |

> **The naive estimator carries a bias of about `-0.12` and an attenuation of about 0.66.
> The EIV estimator is unbiased across the whole range.**

Under the parametric null (`p = 0`, 3,000 draws) the naive median is **-0.118** and the EIV
MLE median is **+0.010**. This is the same pathology the previous run measured as `-0.138` on
`rho_p`, now measured on an interpretable coefficient and *removed* rather than merely bounded.

### 5.4 Result

| estimator | `p` | 95% interval | p-value vs its own null |
|---|---:|---|---:|
| naive OLS (`M_WL` treated as exact) | -0.155 | [-0.277, -0.006] | **0.563** |
| naive partial Spearman `rho_p` (previous primary) | -0.3035 | -- | -- |
| **EIV MLE (primary)** | **-0.166** | **[-0.356, +0.228]** | **0.143** |
| EIV MLE, aperture-drag error model | -0.221 | [-0.776, +0.059] | 0.102 |
| leave-one-out range on the EIV `p` | **-0.205 to -0.093** | always negative | |

**The naive `-0.155` sits at p = 0.563 against its own biased null -- completely consistent
with no signal. The previous run's marginal `rho_p = -0.304` was regression dilution,
essentially in full.** The EIV value `-0.166` is likewise consistent with zero (p = 0.143).
The honest reading of the temperature channel is **no dependence**, not negative dependence.

### 5.5 What the model requires, in the same statistic

Generating `ln E` **directly from the exact chain** at `kappa = 1.36e5` with a free
normalisation (fair to the model -- the amplitude mismatch is given away), the observed
scatter and the real measurement errors, then running the *same* EIV estimator:

> model predicts `p = +0.592`, with 95 per cent of realisations in **[+0.347, +0.889]**
> observed `p = -0.166`
> **separation: 5.4 sigma below the median of the model's own predictive distribution.**

Comparing instead against the analytic requirement `p_req = +0.581` using the bootstrap sd of
the observed estimate gives 3.8 sigma. Either way, decisive.

### 5.6 `kappa` from the shape channel alone

Inverting the (strictly monotone, verified) map `kappa -> p`:

| `kappa` | 0 | 1e4 | 1.7e4 | 3e4 | 4.08e4 | 1e5 | 1.36e5 |
|---|---:|---:|---:|---:|---:|---:|---:|
| required `p` | 0.000 | 0.158 | 0.229 | 0.321 | 0.374 | 0.530 | 0.581 |

- `kappa_EIV = -6.1e3`; one-sided 95 per cent upper **+4.3e3**; two-sided 97.5 per cent upper
  **+1.69e4**.
- The amplitude channel wants `4.08e4`; the shape channel excludes `4.08e4` at **2.7 sigma**
  and `1.36e5` at 3.8-5.4 sigma. **The two channels are mutually inconsistent by a factor
  2.4**, which is itself evidence that a pressure term is not what produces the excess.

---

## 6. POWER -- INJECTION AND RECOVERY THROUGH THE EXACT FORWARD MODEL

Synthetic `ln E` is generated by the **exact chain** at each injected `kappa`, with an
identical generator at every `kappa`, so the `kappa = 0` arm and the `kappa > 0` arms are
exchangeable by construction. Test statistic
`T = sign(kappa_hat) * [SSR(kappa=0) - SSR(kappa_hat)]`, one-sided in the model's direction.

**Recovery is unbiased at every point:** the median fitted `kappa` equals the injected
`kappa` at all twelve grid values in all three arms.

### 6.1 Size check and size correction -- required

| arm | nominal false-positive rate at `kappa = 0` | corrected critical value | rate after correction |
|---|---:|---:|---:|
| A amplitude-inclusive (law as written) | **0.0022** | 1.264 | **0.050** |
| B shape only, free constant | **0.0015** | 1.202 | **0.050** |
| C shape only at fixed mass | **0.0018** | 1.157 | **0.050** |

**The nominal size is NOT 0.05 -- it is about 0.002, i.e. the naive test is roughly 25x
over-conservative**, because the propagated measurement errors used as weights are larger
than the actual scatter (section 7.1). The critical value was therefore taken from the
`kappa = 0` null of the same generator, making the size exactly 0.05 by construction. Under
the conservative generator (full measurement scatter) the nominal rate is 0.024, still wrong,
and was corrected the same way. **All power numbers below are size-corrected.**

### 6.2 Power curve (size-corrected)

| injected `kappa` | arm A | arm B | arm C |
|---:|---:|---:|---:|
| 0 | 0.050 | 0.050 | 0.050 |
| 5.0e3 | 1.000 | 0.563 | 0.529 |
| 1.0e4 | 1.000 | 0.929 | 0.907 |
| 2.0e4 | 1.000 | 1.000 | 1.000 |
| 3.0e4 | 1.000 | 1.000 | 1.000 |
| 5.0e4 | 1.000 | 1.000 | 1.000 |
| **1.36e5 (X-COP)** | **1.000** | **1.000** | **1.000** |
| 5.0e5 | 1.000 | 1.000 | 1.000 |

**Detectable floor at 80 per cent power:**

| generator | arm A | arm B | arm C |
|---|---:|---:|---:|
| matched to the observed scatter | 3.9e3 | 8.2e3 | **8.6e3** |
| conservative (full measurement scatter) | 3.9e3 | 1.45e4 | **1.55e4** |

> **Power at `kappa = 1.36e5` is 1.000 in every arm under every noise assumption. The
> detectable floor is 4e3-1.6e4, a factor 9-35 below the value being tested.**

The null is not a shrug. This sample would have found a coupling of the claimed size every
time. The exact forward model gives *more* power than the compressed proxy did (previous
80 per cent floor about 4e4), because the response is steeper and because `ln E` is the space
in which the scatter is small.

---

## 7. THE APERTURE CAVEAT, AND WHETHER `M_gas/M_b` CHANGES IT

All LoCuSS observables are measured inside `r500` set by the **lensing** mass, and `kT_X_ce`
is measured over about 0.15-1 `r500,WL`. A WL mass error `d` drags everything:
`dln r500 = d/3`, `dln M_gas = a_g d/3`, `dln L_K = a_s d/3`, `dln kT = b_T d/3`.

### 7.1 First: the error model is over-stated when errors are treated as independent

| | variance of `ln E` |
|---|---:|
| observed across the 40 clusters | 0.02628 |
| propagated from published errors, treated as **independent** | **0.03004** |
| propagated with shared-aperture drag `(1.2, 1.0, -0.2)` | 0.01548 |

**Treated as independent, the published errors predict more scatter in `ln E` than actually
exists** -- the implied intrinsic variance is negative, and the moment-form EIV estimator is
undefined (the estimated latent covariance has a negative eigenvalue, -0.0053). Adding the
aperture drag, which must be present, removes the inconsistency. This is independent evidence
that the shared aperture is real. Both error models are carried through the whole analysis;
the independent one is the conservative choice and is used for every headline number.

### 7.2 Does the `M_gas/M_b` factor change the aperture contribution?

**Yes, slightly, and in the direction of *more* aperture leverage, not less.**

Because `DM_P` is proportional to `M_gas` while the denominator is `M_b`, the aperture drags
numerator and denominator together, so `delta = kappa t f_gas` is nearly aperture-invariant:

| response to a WL mass error, averaged over the `(a_g, a_s, b_T)` grid | |
|---|---:|
| `d ln E_obs / d d` | **+0.501** |
| `d ln E_pred / d d` (at the X-COP `kappa`) | **-0.019 to -0.125** |
| `d ln residual / d d` | **+0.558** |

The residual responds **11 per cent more** than the observation alone -- the opposite of the
cancellation one might expect. The previous run's aperture bound therefore has to be inflated
by about 11 per cent, which changes nothing.

### 7.3 Size of the effect

- The aperture supplies **49 per cent** of the variance in `ln E_obs`, but only
  **0.0-2.4 per cent** of the variance in `ln kT`. The temperature channel is the bottleneck
  and it is tiny.
- Induced `corr(ln E_obs, ln kT)` over the whole grid: **[-0.125, +0.042]**; on the residual:
  **[-0.153, +0.041]**. The sign is the sign of `b_T`, and since cluster temperature profiles
  decline outward, `b_T < 0`: **the aperture pushes away from the model, not toward it.**
- Three-arm Monte Carlo on a genuine null world (`E` carries no `kT` information at fixed
  `M_WL`), 2,000 draws per configuration, all three arms on identical draws:

| config `(a_g, a_s, b_T)` | CLEAN | OFF | ON | dilution OFF-CLEAN | **aperture ON-OFF** |
|---|---:|---:|---:|---:|---:|
| 1.2, 1.0, -0.2 | +0.001 | -0.072 | -0.039 | -0.073 | **+0.033** |
| 1.2, 1.0, 0.0 | -0.000 | -0.077 | -0.038 | -0.077 | **+0.039** |
| 1.0, 1.0, -0.4 | -0.002 | -0.077 | -0.055 | -0.075 | **+0.022** |
| 1.5, 1.4, +0.2 | -0.001 | -0.078 | -0.024 | -0.077 | **+0.054** |
| 1.2, 1.0, -0.6 | -0.001 | -0.077 | -0.051 | -0.077 | **+0.026** |

`CLEAN ~ 0.000` validates the null construction. The shared aperture manufactures at most
**`Delta p = +0.054`** -- against a required `p` of `+0.59`. It cannot fake the signal, and it
is of the wrong sign for realistic temperature profiles.

---

## 8. THE CLUSTER EXCLUDED, ROBUSTNESS, AND WHAT WOULD HAVE TO BE TRUE

### 8.1 41 -> 40: **Abell2697**, verified

**Criterion, stated explicitly:** retain a cluster iff all four quantities that enter the
forward chain are published -- `M_WL` (the gravity measurement), `kT_X_ce` (sets the
pressure), `M_gas` (which now enters **twice**, in `M_b` and in `DM_P`), and `L_K_tot`
(stellar mass). No dynamical-state cut, no weak-lensing S/N cut.

Programmatic audit: exactly one cluster fails, **Abell2697**, missing **`L_K_tot`** (and
`L_K_BCG`) -- it has no UKIRT/WFCAM near-infrared data. The belief stated in the brief is
**confirmed**. `Y_SZA` (11 missing), `lambda` (8) and `Y_X` (2) are incomplete but enter
nothing. Imputing Abell2697's stellar mass at the sample-median `M_star/M_gas` and running
n = 41 changes the fixed-`kappa` mean residual from -0.5542 to -0.5543.

### 8.2 Leave-one-cluster-out, all 40

| statistic | LOO range | driven by |
|---|---|---|
| weighted mean residual at `1.36e5` | **-0.564 to -0.527** | Abell0907 / Abell2219 |
| `kappa`, no intercept | **3.95e4 to 4.23e4** | Abell0907 / Abell2219 |
| `kappa`, intercept + mass | **-7.5e3 to -5.3e3** (always negative) | Abell0611 / Abell2219 |
| `p` (EIV) | **-0.205 to -0.093** (always negative) | Abell2204 / Abell0907 |

No single cluster controls anything. `kappa` never approaches `1.36e5`; `p` never approaches
`+0.59`.

### 8.3 Sensitivity

| variant | `kappa` (no intercept) | mean residual at `1.36e5` |
|---|---:|---:|
| `Upsilon_K = 0.50` | 4.175e4 | -0.550 |
| `Upsilon_K = 0.73` (primary) | 4.075e4 | -0.554 |
| `Upsilon_K = 1.10` | 3.925e4 | -0.561 |
| n = 41, Abell2697 imputed | -- | -0.554 |
| `M_WL` S/N >= 2.5 (n = 39) | -- | -0.555 |
| `Omega_m = 0.27` | -- | -0.559 |
| aperture-drag error model | 4.050e4 | -0.548 |

Nothing moves. Stars are about 10 per cent of the baryons, so `Upsilon_K` is nearly
irrelevant.

### 8.4 What would have to be true for `kappa = 1.36e5` to survive

The fitted amplitude is **0.300** of the X-COP value. To close that gap the pressure integral
would have to be 30 per cent of what the isothermal identity gives with `kT_X_ce` -- i.e. an
effective gas-mass-weighted temperature **30 per cent** of the published core-excised value.
Realistic declining profiles give **0.88-0.94** (section 1.1). Non-thermal pressure would push
the required factor the *wrong* way, since it adds pressure. There is no room here.

### 8.5 Statistics licence

The mean normalised squared residual per cluster at the fixed X-COP `kappa` is **12.6**,
nowhere near 1. **No `chi^2`, `lnL`, AIC or BIC is quoted anywhere in this analysis.** Every
interval above is a cluster bootstrap or a Monte-Carlo interval, and every significance is
either a bootstrap ratio or a simulation-calibrated tail probability.

### 8.6 The monotone-invariance trap, checked numerically

Required by the project's recorded lesson. Each statistic evaluated on data generated by the
exact chain across `kappa = 1e3` to `1e6` -- three decades:

| statistic | spread over three decades of `kappa` | moves? |
|---|---:|---|
| `Spearman(kappa t, t)` -- the previous run's compressed response | **0.000000** | **NO -- exactly blind** |
| `Spearman(delta, t f_gas)` | **0.000000** | **NO -- exactly blind** |
| `Spearman(E_pred, t)` | 0.024 | nearly blind |
| `Spearman(E_pred^2-1, t f_gas)` | 0.025 | nearly blind |
| mean `ln E_pred` | 2.367 | yes |
| sd of `ln E_pred` | 0.269 | yes |
| weighted mean residual against the real data | 2.422 | yes |
| `p`, the EIV target parameter | **0.891** (0.000 -> 0.891) | yes |
| `kappa` recovered, no intercept | 1e3 -> 1e6, exact | yes |
| `kappa` recovered, intercept + mass | 1e3 -> 1e6, exact | yes |

The two exactly-blind entries are the trap, reproduced deliberately: a rank correlation
between a prediction and its own monotone input is bit-identical at every `kappa`. **Every
statistic used for any conclusion in this report is in the lower block and moves by orders of
magnitude.**

---

## 9. VERDICT

> **The original source law `rho_eff = rho_b + 3 kappa P/c^2`, computed exactly and evaluated
> against LoCuSS weak-lensing masses, is REFUTED at `kappa = 1.36e5`, and the corrected
> calculation refutes it by a LARGER margin than the compressed proxy did.**

The exact chain does not rescue the model. It hurts it, for a reason that only appears once
the chain is computed properly: at the X-COP coupling the pressure term multiplies the source
by 3.3-9.7, which lifts `g_N_eff/a0` from 0.04-0.12 up to 0.18-0.91. That is out of the
deep-MOND regime and into the RAR transition, where `F` is steeper than `sqrt`, so the
prediction is **19 per cent higher** than the deep-MOND formula suggests. The `M_gas/M_b`
factor that the previous run omitted pulls the other way, but it is worth only
`1/0.895 = 1.107` in `kappa`. The two corrections do not cancel; the net is against the model.

Three independent requirements, all failed:

| requirement | result |
|---|---|
| **the stated amplitude, `kappa = 1.36e5`, zero intercept** | **FAILS.** `E_pred/E_obs` median 1.756; all 40 clusters over-predicted; weighted mean `ln` residual -0.544 at **-13.0 sigma**; only 10 of 40 within 2 sigma. The free amplitude fit gives `4.08e4` = **0.300** of X-COP, with `1.36e5` far outside the bootstrap interval. |
| **a temperature dependence at fixed mass** | **FAILS.** EIV `p = -0.166` [-0.356, +0.228], consistent with zero (p = 0.143). The model's own predictive distribution for that same statistic is `+0.592` [+0.347, +0.889]. **Separation 5.4 sigma.** LOO range -0.205 to -0.093, never positive. |
| **a single universal `kappa`** | **FAILS.** Per-cluster `kappa_i` spans a factor **8.45**, running down with `kT` (-0.342, p = 0.031) and up with `M_WL` (+0.312, p = 0.050). |

And a fourth failure, internal to the model: **the amplitude channel and the shape channel
demand different couplings.** Matching the mean excess needs `4.08e4`; the temperature shape
excludes anything above about `1.7e4`. A one-parameter source term cannot be both.

**The null is informative.** With the exact forward model, size-corrected to a true 0.05
false-positive rate, this sample has **power 1.000 at `kappa = 1.36e5`** and an 80 per cent
detection floor of **4e3-1.6e4**, a factor 9-35 below the tested value. A coupling of the
claimed size would have been seen every time. It is not there.

**The previously-reported negative correlation was an artefact, and is retired.** The naive
`rho_p = -0.304` (reproduced here exactly) and the naive `p = -0.155` both sit at p = 0.56
against their own biased nulls: the naive estimator's expectation under a true null is
`-0.12`, because `ln E_obs` and `ln M_WL` have measurement errors correlated at +0.96. The
EIV treatment removes that bias (null median +0.010) and returns a value consistent with zero.
**There is no anti-correlation with temperature. There is no correlation at all.**

### What is real, and what this does not settle

A large, robust gravity excess over baryons-plus-RAR is present: `E` median **1.62**, range
1.22-2.34, measured with **weak lensing** and therefore free of hydrostatic circularity. It
behaves as a roughly constant multiplicative offset (free constant `+0.603` in `ln E`, a
factor 1.83) with a mild **mass** dependence (`d ln E/d ln M_WL = +0.31`) and **no temperature
dependence**. That is the classic MOND cluster missing-mass problem, independently confirmed
with lensing; it is simply not shaped like `3 kappa P/c^2`.

Honest limits on the scope of this refutation:

1. **Isothermality is assumed** -- LoCuSS publishes one temperature per cluster, so `P(r)` is
   unavailable. The reduction of the integral is exact *given* isothermality (verified to
   1e-12, profile-independently), and a realistic declining `T(r)` moves the amplitude by
   0.88-0.94. Reconciliation would need 0.30. The assumption is not load-bearing.
2. **`a0 = 1.2e-10` and the RAR `nu` are held fixed.** A source law combined with a different
   interpolation function or a different `a0` is a different theory and is untested here.
3. **No gravitational slip** is assumed: lensing measures the same `g` as dynamics.
4. **Non-thermal pressure is unmeasured** in this sample; including it would raise `DM_P` and
   make the over-prediction worse.
5. **The radially resolved test is not performed.** The genuinely complete test needs joint
   X-ray/SZ pressure profiles on clusters with independent weak-lensing masses, evaluated on
   matched radii. LoCuSS has the lensing but not the profiles; X-COP has the profiles but no
   independent gravity measurement. What is settled here is the *enclosed-mass* form of the
   exact chain, at one radius, with a real gravity measurement -- and it fails on amplitude,
   on shape, and on universality simultaneously.

### On the X-COP `kappa = 1.36e5` itself

This analysis gives an independent reason to distrust it. The X-COP value was derived with
temperature on both sides of the relation. Here, with temperature on only one side, the
amplitude channel returns `4.08e4` and the shape channel returns a value consistent with zero
-- and the two disagree with each other. A coupling inferred from a confounded relation, then
found to be a factor 3.3 too large *and* internally inconsistent when tested against an
independent gravity probe, is most simply read as a fit to the confound.
