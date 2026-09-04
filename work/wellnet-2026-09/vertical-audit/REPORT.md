# Vertical-audit lane — is `d log10 B_z / d log10 Sigma_0 = -0.346` real?

Lane: `work/wellnet-2026-09/vertical-audit/`
Audited: `work/gravity-cluster-audit-2026-09/adyn/` — the DiskMass vertical-field
result `-0.346 +- 0.173`, galaxy bootstrap `p(slope >= 0) = 0.0095`.

**Sealed holdouts: KiDS and wide binaries were never loaded, listed, queried or
referenced by this lane.** No mass map, NFW fit or parametric lens model was
used. No parameter is fitted per galaxy anywhere below.

---

## VERDICT ON ITEM 2 (the decisive one), first

**The shared-denominator structure is real and exact. Its effect is not.**

`Sigma_L0` *is* in the denominator of `B_z` with a measured logarithmic
coefficient of **-0.994**, and it *is* the abscissa of the headline regression,
so a photometric error does move a galaxy right in *x* and down in *y*, exactly
as charged, and no label shuffle can see it. But when the complete observational
covariance is pushed through the real pipeline with a true Newtonian `B_z = 1`,
the artefact displaces the slope by only

> **-0.012 to -0.018 dex per dex, i.e. 3.5-5.2% of -0.346.**

In **400 Newtonian universes per scenario across five covariance scenarios, the
recovered slope never once reached -0.346** (minimum over 2,000 trials:
**-0.325**):

> **P(slope <= -0.346 | Newton, full covariance) = 0, 95% upper bound 0.0075.**

To manufacture the observed slope from the shared denominator alone, the
photometric error would have to be `e_mu0_K,i` ~ **0.70 mag, ten times** the
tabulated median of 0.070 mag. The retracted `rho_p = -0.304` cluster case is not
repeating itself: there the two axes shared an error correlated at +0.96; here
the shared error is 0.028 dex against an abscissa spread of 0.336 dex.

**But the result still must not be promoted as a gravitational measurement**, for
two reasons that have nothing to do with the shared denominator:

1. **The published error bar is wrong** (item 3). `+- 0.173` is the raw standard
   deviation of a distribution with skewness **+26** and excess kurtosis
   **+901**, produced by an unguarded `sigma_z^2` floor in the forward model. It
   does not converge: 0.066 at 200 draws, 0.316 at 1,600, 0.328 at 6,400. The
   defensible headline is **`-0.346 +- 0.117`**, one-sided `p = 0.010`.
2. **The winning model is not a model of gravity** (item 4). With the model
   selection calibrated against synthetic universes under **both** hypotheses,
   **Newton and RAR are both rejected**: the data carry a `Sigma_0`-dependent
   amplitude that Newton does not have, and lack the radial gradient RAR
   requires. What fits is a **radially flat** factor `B_z ~ Sigma_0^-0.35`, and
   that is identical **to 5e-15 dex** — machine precision — to Newton with a
   K-band mass-to-light ratio scaling as `Sigma_0^-0.35`. The DiskMass vertical
   data cannot distinguish those two even in principle.

---

## 0. Reproduction gate

`vaudit_core.py` re-instantiates the pipeline — every physics function imported
from `adyn_model.py`, the three glue functions copied character-for-character out
of the `adyn_run.py` script body — and reproduces:

| quantity | this lane | `adyn_results.json` |
|---|---:|---:|
| galaxies retained | 28 | 28 |
| `d log10 B_z / d log10 Sigma_0` | -0.3459 | -0.34592 |
| 68% interval | [-0.4159, -0.2761] | [-0.41591, -0.27613] |
| raw sd over 800 nuisance draws | 0.1287 | 0.1287 (= 0.1735 / 1.3485) |
| RAR predicted slope | -0.288 | -0.291 |
| AQUAL predicted slope | -0.260 | -0.264 |
| isotropic tensor predicted slope | -0.048 | -0.055 |

The three predicted-slope differences are the semi-analytic tensor
`B_z = 1/mu_z` versus the original's 2-D solve, which the original itself gates
at 0.6%.

---

## 1. The exact formula for B_z — full document in `bz_formula.md`

```
B_z(galaxy) = [ sigma_LOS_0(observed) / sigma_LOS_0(model, Newton) ]^2
```

The **numerator** is one measured number per galaxy (DiskMass VI table 6). The
**denominator** is not a closed form: it is the amplitude of an exponential
fitted, over the same window, to a forward-modelled `sigma_LOS(R)` built from

```
sigma_z^2(R) = max( 2 pi G h_z [Sigma_*(R) A_ss(k) + Sigma_g(R) A_sg(k,f_hzg)]
                    - L_s(k) h_z^2 (1/R) dV_c^2/dR ,  1e-30 )
Sigma_*(R)   = Upsilon_K * Sigma_L0 * exp(-R/h_R)
Sigma_L0     = 10^(0.4 (M_K,sun + 21.572 - mu0_K,i))         <== ALSO THE X-AXIS
```

**Measured:** `sigma_LOS_0`, `mu0_K,i`, `h_R`, `i`, `D`.
**Inferred:** `h_z`, from `h_R` via Bershady+2010b, exponent 0.643.
**Priors only:** `Upsilon_K`, `k`, `alpha`, `f_gas`, `f_hg`, `f_hzg`, the window.
**In more than one place:** `Sigma_L0` (denominator **and** abscissa); `h_R`
(via `h_z`, via the thickness factor `T`, via the radial grid and window).

Measured logarithmic sensitivities, finite difference through the real code:

| input | `d log10 B_z / d log10 x` | prose in `adyn/REPORT.md` |
|---|---:|---:|
| `sigma_LOS_0` | **+2.0000** | +2 |
| **`Sigma_L0`** | **-0.9943** | not mentioned |
| `Upsilon_K` | -1.0000 | -1 |
| `h_z` | **-0.6667** | -1 |
| `k` | **-0.7965** | -1 |
| `h_R` (at fixed `h_z`, `Sigma_L0`) | -0.3048 | not mentioned |
| `alpha` | +0.7559 | not mentioned |
| `f_gas` | -0.0774 | not mentioned |
| inclination | -0.4402 | not mentioned |
| distance | -0.3239 | not mentioned |

Two consequences.

* **`h_z` carries 0.667, not 1.** The original's headline "the Bershady relation
  would have to be wrong in SLOPE by 1.99" assumes a coefficient of 1. Its own
  numerical scan two lines above (-0.366 at `d = -0.30`, -0.316 at `d = +0.30`)
  implies **+0.083 per unit `d`**, so removing -0.340 needs **`d` = 4.1**. The
  conclusion is unchanged and in fact strengthened; the number is wrong by 2x.
* **A defect.** `newton_chain` floors `sigma_z^2` at `1e-30`. When a nuisance
  draw puts `h_z` far above the tabulated value for a small, thick galaxy
  (UGC 6918: 0.21 / 1.2 kpc; UGC 1862: 0.24 / 1.4), the leakage term exceeds the
  vertical gravity, `sigma_z^2` is clipped, and that galaxy's `log B_z` diverges
  — realised values to **+41 dex**. It happens **inside the fit window in 11 of
  1,600 draws** and is the sole cause of the `+- 0.173` error bar.

---

## 2. The shared-denominator audit

### 2a. Injection and recovery — `inject_recover.py`

Each trial: (a) `B_z = 1` exactly; (b) latent `Upsilon_K`, `f_gas`, `k`, `alpha`,
gas geometry and window drawn from the pipeline's own priors, latent photometry =
the catalogue; (c) the full observational covariance applied as a latent ->
observed map so the **same** realisation feeds *x* and *y*, including the
correlated `h_z = C h_R^0.643` channel and the `mu0_K`/`h_R` decomposition
degeneracy; (d) reconstruction through the **real** `Bench` — same
`newton_chain`, `to_los`, aperture operator, exponential fit and nuisance
marginalisation that produce -0.346; (e) 400 trials per scenario.

| scenario | null slope (median) | 95% | worst of 400 | P(<= -0.346) |
|---|---:|---|---:|---:|
| A `mu0_K`, `h_R` errors independent | -0.0009 | [-0.138, +0.152] | -0.206 | 0 |
| B fixed-`L_tot` degeneracy, `rho = -1` | +0.0002 | [-0.146, +0.145] | -0.214 | 0 |
| C = B + inclination error | +0.0071 | [-0.121, +0.154] | -0.179 | 0 |
| D = C + **all photometric errors x3** | -0.0684 | [-0.228, +0.109] | **-0.325** | 0 |
| E = C + galaxy resampling | +0.0002 | [-0.187, +0.203] | -0.307 | 0 |

`rho = -1` is the physical worst case: for an exponential disk
`L = 2 pi Sigma_0 h_R^2`, so a well-constrained total magnitude forces
`d log Sigma_0 = -2 d log h_R`, the strongest possible anticorrelation between
the abscissa error and the `h_z` channel.

**Is the null correctly sized?** Both checks the promotion criterion requires:

* *scatter* — the Newtonian null's residual rms about the `Sigma_0` fit is
  **0.1705 dex** (68% [0.153, 0.184]) against the observed **0.1669 dex**, ratio
  **0.98**. The null is not an under-dispersed forward model.
* *slope* — injecting a genuine `B_z ~ Sigma_0^p` and recovering it through the
  same machinery: bias `<= 0.012` for `p` from -0.60 to +0.30
  (-0.600 -> -0.588; -0.346 -> -0.339; -0.150 -> -0.152; 0.000 -> -0.007;
  +0.300 -> +0.307).

### 2b. Inverting the question

| `e_mu0_K,i` inflation | median error | null slope | P(<= -0.346) |
|---:|---:|---:|---:|
| x1 (as tabulated) | 0.07 mag | -0.015 +- 0.068 | 0.000 |
| x3 | 0.21 mag | -0.074 +- 0.084 | 0.000 |
| x5 | 0.35 mag | -0.172 +- 0.075 | 0.010 |
| x7 | 0.49 mag | -0.266 +- 0.068 | 0.110 |
| **x10** | **0.70 mag** | **-0.377** | 0.650 |

DiskMass VI quotes `e_mu0_K,i` between 0.03 and 0.18 mag.

### 2c. The clean form, fitted directly

`log Sigma_dyn = a + s log Sigma_b`, latent variables on both axes, per-galaxy
2x2 error covariance measured by pushing the real errors through the real chain
(2,000 realisations), free intrinsic scatter:

| | `rho(mu0,h_R) = 0` | `rho = -1` |
|---|---:|---:|
| OLS, ratio form (`log B_z` on `log Sigma_0`) | -0.3473 | -0.3473 |
| OLS, clean form | +0.6527 | +0.6527 |
| **EIV clean form, `s`** | **+0.649 +- 0.066** | **+0.646 +- 0.064** |
| fitted intrinsic scatter | 0.071 dex | 0.073 dex |
| test `s = 1` | 5.35 sigma | 5.55 sigma |

The two forms are the same fit shifted by 1 — verified numerically, not asserted.
The errors-in-variables correction moves `s` by **+0.012**
(`var(eps_x)/var(x) = 0.018`), which is the entire shared-denominator
displacement. The EIV error bar is model-conditional; the galaxy bootstrap gives
0.117 because two galaxies carry most of the `Sigma_0` leverage. **Use the
bootstrap.**

### 2d. Not the forward model either

The closed-form DiskMass estimator, which never touches the forward chain —
`Sigma_dyn = sigma_z,0^2/(pi G k h_z)`, `Sigma_b = Upsilon_K Sigma_L0 (1+f_gas)`
— gives `s = 0.647` on `Sigma_b` and `0.668` on `Sigma_L0`, an implied
`d log B_z/d log Sigma_0` of **-0.332** against the forward chain's -0.346.
**The signal is in the raw catalogue numbers.**

### 2e. What the signal actually is, and what systematic would be needed

| piece | value | contribution to the slope |
|---|---:|---:|
| `2 x d log sigma_LOS,0 / d log Sigma_0` | +0.2716 | +0.543 |
| `-0.667 x d log h_z / d log Sigma_0` | -0.1172 | +0.078 |
| `-0.15 x d(B-K)/d log Sigma_0` (the `Upsilon` colour term) | +0.324 mag/dex | -0.049 |
| `Sigma_L0` itself, coefficient -0.994 | | -0.994 |
| **sum** | | **-0.42** (pipeline -0.346; the rest is gas, leakage, per-galaxy thickness) |

To leading order the headline is **twice the `sigma_LOS,0`-`Sigma_0` scaling
minus one** — a vertical Fundamental-Plane-like relation restated as a ratio.
Newton with a fixed `Upsilon_K` needs `d log sigma/d log Sigma_0 = +0.486`; the
data give **+0.272**.

The systematics that would have to exist instead, each quantified:

* **`Upsilon_K`** anticorrelating with `Sigma_0` at **-0.395 dex/dex** beyond the
  modelled colour term — a factor **4.1** across the sample's 1.544 dex, with the
  *low* surface-brightness galaxies carrying the *higher* K-band M/L. The
  observed `(B-K)` colours imply the opposite sign (+0.049 dex/dex).
* **`h_z`** falling with `Sigma_0` at `-0.636` dex/dex instead of the measured
  `-0.117`, so `h_z/h_R` would vary by a factor **5.2** across the sample where
  the Bershady relation gives **1.22**; equivalently the Bershady exponent would
  be 3.7 (linear estimate) or **4.1** (the original's own numerical scan) instead
  of 0.643.

Neither is plausible on its own. That is the strongest positive statement this
lane can make for the signal.

---

## 3. Reconciling `-0.346 +- 0.173` with `p = 0.0095`

**They are different statistics, and the first is broken.**

**What `+- 0.173` is:** `std(slope over nuisance draws) x 1.3485`, the 1.3485
being the published `chi2/dof = 1` inflation. The underlying distribution is
pathological:

| estimator of the nuisance-only sd | value | x1.3485 |
|---|---:|---:|
| raw `np.std` (as published) | 0.2567 at 3,000 draws | 0.346 |
| from the 16-84 range | 0.0699 | 0.094 |
| after a 1% two-sided trim | 0.0654 | 0.088 |
| 1.4826 x MAD | 0.0696 | 0.094 |

skewness **+26.4**, excess kurtosis **+901**, 16 of 3,000 draws beyond 5 robust
sigma, extreme slope **+10.16**. And the raw sd does not converge:

| draws | 200 | 400 | 800 | 1,600 | 3,200 | 6,400 |
|---|---:|---:|---:|---:|---:|---:|
| raw sd | 0.066 | 0.166 | **0.129** | 0.316 | 0.260 | 0.328 |
| robust sd | 0.067 | 0.064 | 0.070 | 0.071 | 0.070 | 0.070 |

The published 0.1287 is simply what 800 draws happened to give. The cause is the
`sigma_z^2` floor of section 1.

**What `0.0095` is:** a **one-sided** tail fraction `P(slope >= 0)` of a
**galaxy** bootstrap — resampling galaxies with one random nuisance draw each —
where `0.173` resamples nuisances at a fixed galaxy set. Reproduced here as
**0.0106** with 20,000 resamples. That distribution is right-skewed too
(skewness +37), so the percentile is the right estimator and a normal
approximation to it would be wrong.

**Made commensurate:**

| pairing | sigma | two-sided p |
|---|---:|---:|
| nuisance-only, raw sd, as published | 0.96 | 0.338 |
| nuisance-only, robust sd | 3.67 | 0.0002 |
| galaxy bootstrap, robust sd | 2.96 | 0.0031 |
| galaxy bootstrap, percentile | — | 0.021 |

Both published numbers are individually correct; **printing them side by side is
not**, because the reader divides one by the other and 0.346/0.173 = 2.00 is
meaningless. The defensible headline is

> `d log10 B_z / d log10 Sigma_0 = -0.346 +- 0.117` (galaxy bootstrap, robust),
> one-sided `p = 0.010`, i.e. **2.3 to 3.0 sigma** depending on estimator.

---

## 4. Local versus global

All models run through the identical chain, scored on **both** DiskMass
observables (`sigma_LOS,0` and `h_sigma_LOS`, 28 x 2 = 56 numbers) with a free
common-mode amplitude offset and a **free intrinsic scatter on each observable**,
so `chi2/dof = 1` by construction — which removes the original report's stated
reason for refusing `lnL`/AIC/BIC. Global parameters only.

**Boundary rule for the potential, declared in the code before any fit:** primary
`dPhi_b(R) = Phi_b(R) - Phi_b(4 h_R)` for the isolated baryonic disk with
`Phi -> 0` at infinity, the reference at 4 `h_R` lying **outside** the 0.3-2.0
`h_R` window; secondaries at 8 `h_R` and at infinity. Response
`(1 + |dPhi|/Phi_*)^q`, `Phi_* = (100 km/s)^2` declared — not a bare power law,
because the reference rules put a zero of `dPhi` at `R_ref`.

| model | k | AIC | dAIC | dBIC | `s_amp` | `s_h` | fitted shape |
|---|---:|---:|---:|---:|---:|---:|---|
| **M_global** `B_z ~ Sigma_0^p` | 4 | -192.43 | **0.00** | **0.00** | **0.052** | **0.149** | `p = -0.350` |
| M_local_RAR (`a0` frozen from SPARC) | 3 | -183.62 | 8.80 | 6.78 | 0.053 | 0.184 | — |
| M_local_AQUAL (`a0` frozen) | 3 | -183.03 | 9.40 | 7.38 | 0.055 | 0.183 | — |
| M_local_RAR, `a0` free | 4 | -181.73 | 10.70 | 10.70 | 0.055 | 0.180 | `log a0 = -10.10` |
| M_local_pow `B_z ~ \|g\|^-m` | 4 | -181.52 | 10.91 | 10.91 | 0.056 | 0.176 | `m = +0.25` |
| M_Phi, reference at infinity | 4 | -175.64 | 16.79 | 16.79 | 0.071 | 0.160 | `q = -0.20` |
| M_Newton = M_tensor_aniso | 3 | -174.53 | 17.90 | 15.87 | 0.080 | 0.149 | — |
| M_Phi, reference 8 `h_R` | 4 | -173.68 | 18.75 | 18.75 | 0.071 | 0.166 | `q = -0.20` |
| M_Phi, reference 4 `h_R` (**primary**) | 4 | -172.92 | 19.51 | 19.51 | 0.072 | 0.168 | `q = -0.20` |
| M_tensor_iso | 3 | -133.98 | 58.45 | 56.42 | 0.074 | 0.369 | — |

**The mechanism is in the two intrinsic scatters, and it is the whole result.**
Every model that helps cuts the amplitude scatter from Newton's 0.080 dex to
about 0.052-0.056. Only `M_global` does so **without paying for it on the scale
length**: it leaves `s_h` at Newton's 0.149 while every local law inflates it to
0.176-0.184, because a local response must vary radially as the local
acceleration does and the data say it does not. The fitted `p = -0.350` recovers
the headline slope from a completely different estimator.

Ranking under draws of the full nuisance prior (`dAIC` against the best model in
each draw):

| model | median dAIC | p16 | p84 | draws won | fitted shape p16-p84 |
|---|---:|---:|---:|---:|---|
| M_global | 0.00 | 0.00 | 0.00 | 20 | -0.350 to -0.302 |
| M_local_RAR | 8.19 | 7.05 | 9.66 | 0 | -- |
| M_local_AQUAL | 8.86 | 7.55 | 9.91 | 0 | -- |
| M_local_RAR_a0 | 9.84 | 8.80 | 10.80 | 0 | -10.396 to -10.000 |
| M_local_pow | 10.23 | 8.34 | 10.81 | 0 | +0.200 to +0.250 |
| M_Phi_inf | 15.23 | 12.61 | 16.44 | 0 | -0.200 to -0.200 |
| M_Newton | 16.04 | 10.67 | 17.84 | 0 | -- |
| M_tensor_aniso | 16.04 | 10.67 | 17.84 | 0 | -- |
| M_Phi_ref8 | 17.58 | 12.66 | 18.84 | 0 | -0.200 to +0.000 |
| M_Phi_ref4 | 18.00 | 12.67 | 19.45 | 0 | -0.200 to +0.000 |
| M_tensor_iso | 56.68 | 47.31 | 63.20 | 0 | -- |

**Why local and global are nearly inseparable here.** For an exponential disk the
Freeman formula makes `g_b` at a fixed multiple of `h_R` proportional to
`Sigma_0` times a pure constant:

| state variable | within-galaxy range, 0.3-2.0 `h_R` | between-galaxy sd |
|---|---:|---:|
| `g_b(R)` | -0.051 dex | 0.353 dex |
| `Sigma_0` | 0.000 dex | 0.336 dex |
| `\|dPhi_b\|`, ref 4 `h_R` | +0.392 dex | 0.348 dex |
| `\|Phi_b\|`, ref infinity | +0.211 dex | 0.348 dex |

"Local in `g_b`" and "global in `Sigma_0`" are *the same between-galaxy
predictor* on DiskMass — the env-data lane found the identical degeneracy at
`r = 0.9955`. **Only the radial shape separates them**, and `g_b` varies by just
0.05 dex across the window. What actually distinguishes RAR/AQUAL is not `g_b(R)`
but the `z`-structure: `nu` and `mu` take `|grad Phi| = sqrt(g_R^2 + K_z^2)`,
which falls outward with `Sigma(R)`, giving `B_z(0.3 h_R)/B_z(2 h_R) = 0.848`.
That radial signature is what the scale lengths reject.

The potential-depth models are **worse than Newton on BIC under all three
declared boundary rules**: `|dPhi_b|` has *more* within-galaxy structure than
`g_b` does, so it is penalised on the scale lengths for the same reason.

### The finding that decides the interpretation — `degeneracy_check.py`

`M_global` multiplies `K_z` by `(Sigma_0/<Sigma_0>)^p` at every radius. "Newton
with `Upsilon_K ~ Sigma_0^p`" multiplies `Sigma_*(R)`, and therefore `K_z`, by
the same factor at every radius. Because `sigma_z^2`, `g_R` and the leakage term
are **all** exactly linear in `Sigma_*0`, the two predictions are identical:

| `p` | max \|d log amplitude\| | max \|d log scale length\| |
|---:|---:|---:|
| -0.60 | 5.5e-15 | 1.1e-14 |
| -0.35 | 3.9e-15 | 7.1e-15 |
| +0.20 | 5.5e-15 | 1.1e-14 |

**Machine precision.** Against measurement errors of 0.028 dex (amplitude) and
0.058 dex (scale length). `M_global` is not a gravitational model that the data
prefer; it is the statement "a radially flat per-galaxy factor that tracks
surface brightness", and gravity and mass-to-light are the same object inside it.

### The model selection, CALIBRATED — `model_null.py`

A `dAIC` is not evidence until its null distribution is known. The programme's
standing objection ("detector calibration is conditional, not general") applies
here as much as to the tensor detector. `model_null.py` generates 100 synthetic
universes under **each** hypothesis, with the same full observational covariance
as the injection test, and scores the same three models. A gate asserts that its
scoring reproduces `model_compare.json`'s AIC values on the real data to
**0.000**.

**Statistic 1: `dAIC(Newton - global)`, observed +17.90** — does the data
demand *anything* `Sigma_0`-dependent?

| null | median | 95% | [min, max] | P(>= observed) |
|---|---:|---|---|---:|
| newton-truth | -1.4 | [-2.0, +3.0] | [-2.0, +6.0] | 0.000 |
| rar-truth | +22.6 | [+11.7, +36.6] | [+6.7, +47.0] | 0.710 |

**Statistic 2: `dAIC(RAR - global)`, observed +8.80** — is what it demands
*local*?

| null | median | 95% | [min, max] | P(>= observed) |
|---|---:|---|---|---:|
| newton-truth | +31.2 | [-6.8, +55.7] | [-24.2, +59.3] | 0.820 |
| rar-truth | -22.8 | [-35.5, +1.2] | [-41.1, +9.1] | 0.010 |

**Both hypotheses are rejected, each by the statistic the other passes.**
Statistic 1 excludes Newton and is entirely comfortable under RAR-truth.
Statistic 2 excludes RAR and is entirely comfortable under Newton-truth. The
observed data are simultaneously too `Sigma_0`-dependent for Newton and too
radially flat for RAR.

This also **retires a naive reading of the raw `dAIC = 8.8`**: under Newton-truth
the median `dAIC(RAR - global)` is already +31.2,
so 8.8 on its own is unremarkable and must never be quoted as "RAR is disfavoured
by 8.8 units". It is only meaningful against the RAR-truth null, where 1 of 100
realisations reaches it (max +9.1).

### What item 4 does and does not establish

**Establishes:** the vertical enhancement these data demand has a
`Sigma_0`-dependent amplitude and **no** accompanying radial gradient. Newton is
rejected (`P = 0.000`,
95% bound 0.030); RAR is rejected
(`P = 0.010`);
the isotropic tensor is dead at `dAIC = 58`; the anisotropic tensor is
numerically identical to Newton (`mu_z = 1` means exactly that); the
potential-depth models are worse than Newton on BIC under all three declared
boundary rules.

**Does not establish:** that this is gravity. The surviving description — a
radially flat factor tracking surface brightness — is exactly degenerate with a
calibration systematic. What the comparison removes from the table is MOND:
whatever produces the trend, it is not a local function of `g_b`, and it is not a
function of baryonic potential depth under any declared boundary rule.

That is the resolution of the original report's unresolved tension, and it is
worth more than the slope. The between-galaxy trend and the within-galaxy radial
trend are not in conflict with each other — they are jointly in conflict with
every law in which the vertical response is a local function of the field.

---

## 5. Law predictions through the identical pipeline

Verified rather than assumed: same photometry, same 2.7" fibre convolved with a
1.5" PSF, same 0.3-2.0 `h_R` window, same exponential-fit operator, same
inclination and ellipsoid projection as the data. Amplitudes and scale lengths of
the fitted model profile, not pointwise `B_z`.

| model | amp pred | amp obs | `h_sigma` pred (") | obs | `<log B_z>` | `d log B_z/d log Sigma_0` | published |
|---|---:|---:|---:|---:|---:|---:|---:|
| Newton | 59.45 | 51.05 | 30.80 | 28.65 | 0.000 | 0.000 | 0.000 |
| RAR | 71.16 | 51.05 | 35.09 | 28.65 | +0.183 | **-0.288** | -0.291 |
| AQUAL | 71.41 | 51.05 | 34.85 | 28.65 | +0.184 | **-0.260** | -0.264 |
| tensor aniso | 59.45 | 51.05 | 30.80 | 28.65 | 0.000 | 0.000 | -0.020 |
| tensor iso | 58.30 | 51.05 | 51.33 | 28.65 | -0.011 | -0.048 | -0.055 |
| **observed** | | 51.05 | | 28.65 | -0.146 | **-0.346** | |

Median per-galaxy `h_sigma / h_R`, the form an external sample can be compared
against:

| | observed | Newton | RAR | AQUAL | tensor iso |
|---|---:|---:|---:|---:|---:|
| `h_sigma / h_R` | **2.086** | 2.499 | 2.896 | 2.876 | 4.278 |

The observed vertical dispersion profile is **steeper** than Newton; RAR needs it
**flatter**. That is the within-galaxy half of the tension, per galaxy.

---

## 6. Independent check of the sign — resolved MaNGA face-on disks

`manga_check.py`, on Run V's `env-data` product
(`manga_faceon_sigma_profiles.csv`, 240 near-face-on MaNGA DR17 late types, 1,671
radial points, manifest present; row and galaxy counts asserted on ingest).
Deprojected with the same `alpha = 0.60`, `beta = 0.70` ellipsoid; `h_z` imported
from the same Bershady exponent, which is stated at every step and is **not**
independent.

| tier | galaxies | median `sigma_LOS` | `d log sigma_z,0/d log Sigma_b` | Newton needs | `d log B_z/d log Sigma_b` | `P(>= 0)` |
|---|---:|---:|---:|---:|---|---:|
| all points > 70 km/s | 42 | 101 | +0.016 | +0.431 | **-0.830** [-1.035, -0.685] | 0.0000 |
| points > 70 km/s | 74 | 96 | +0.071 | +0.421 | **-0.700** [-0.806, -0.604] | 0.0000 |
| all points > 50 km/s | 139 | 77 | +0.055 | +0.414 | **-0.719** [-0.806, -0.632] | 0.0000 |
| all points > 90 km/s | 14 | 141 | +0.037 | +0.416 | -0.757 [-0.878, -0.647] | 0.0000 |
| all points > 110 km/s | 8 | 154 | +0.056 | +0.399 | -0.687 [-1.143, -0.564] | 0.0003 |

**The sign replicates** on different galaxies, different photometry, different
stellar masses (NSA SED, Chabrier) and a different spectrograph, and is stable
along the instrumental-floor ladder from 70 to 110 km/s (median 154 km/s in the
last tier, more than twice the floor), so it is not a truncation effect. Bulge
contamination works *against* the signal — bulges raise `sigma` where `Sigma_b`
is high — so it cannot be the cause either.

**Four reasons it is a weaker test than DiskMass, not a stronger one:**

1. Its own shared-denominator floor is far larger. `Sigma_b = M_b/(2 pi R_d^2)`,
   so `e(log M_*) = 0.10` and `e(log R_d) = 0.043` already give **-0.15**; at
   0.20 / 0.087 it is **-0.60**, comparable to the signal. MaNGA cannot support
   the injection test DiskMass can, because its input errors are not tabulated.
2. `sigma_LOS` is not `sigma_z`, and no scale height is measured anywhere.
3. Median `sigma` is 77-154 km/s against DiskMass's 24-45. These are massive
   bulge-bearing disks, not the thin submaximal disks DiskMass selected.
4. Its radial statistic has no power: `h_sigma/R_d = 2.81`, 68% [1.75, 6.21],
   against a Newton-to-RAR separation of 2.50 to 2.90 — and beam smearing
   flattens MaNGA profiles in exactly the direction that matters.

**An independent confirmation of the sign, with a steeper amplitude and much
weaker control.** Consistent with the DiskMass slope being real; equally
consistent with both samples sharing a surface-brightness-correlated
mass-to-light systematic. That is the point.

---

## 7. Promotion criterion, stated in advance, evaluated

| gate | result |
|---|---|
| Newtonian injection null correctly sized | **PASS** — residual rms 0.171 vs 0.167 observed (0.98); recovery bias `<= 0.012` over injected slopes -0.60 to +0.30 |
| slope survives the shared-denominator covariance | **PASS** — artefact contributes -0.012 to -0.018 of -0.346; `P(<= -0.346 | Newton) = 0` in 2,000 trials, 95% bound 0.0075 |
| latent `Sigma_dyn`-vs-`Sigma_b` regression agrees | **PASS** — `s = 0.646 +- 0.064` against `s - 1 = -0.346 => s = 0.654`; identity verified numerically |
| law predictions through the same pipeline | **PASS** — RAR -0.288 vs published -0.291, AQUAL -0.260 vs -0.264 |

**All four gates pass. The result is not a tautology and it is not an artefact of
the shared denominator.** Two things nonetheless block promotion as a
*gravitational* result:

* the quoted significance is wrong — `+- 0.173` is a non-convergent moment of a
  distribution corrupted by a code defect; the correct headline is
  `-0.346 +- 0.117`, one-sided `p = 0.010`;
* the calibrated model selection rejects **both** Newton and RAR, and the model
  that survives is degenerate **to machine precision** with a `Sigma_0`-dependent
  `Upsilon_K`.

**Recommended status:** promote as a **measurement of the vertical
`Sigma_dyn`-`Sigma_b` relation**, `s = 0.65 +- 0.12`, replicated in sign on
MaNGA, with the explicit statement that it is exactly degenerate with a
surface-brightness-dependent `Upsilon_K` of -0.395 dex/dex (a factor 4.1 across
the sample) or an `h_z/h_R` trend a factor 5.2 steeper than Bershady. Do **not**
promote as evidence for a modified force law — the same analysis that establishes
the trend disfavours every local law tested.

---

## 8. Programme failure modes — checked explicitly

* **Shared-denominator artefacts.** The subject of item 2. Structure confirmed
  (coefficient -0.994 on the abscissa itself), magnitude measured by injection
  with the real covariance rather than assumed: -0.012 to -0.018. The MaNGA
  cross-check has its own, larger, floor and it is reported rather than hidden.
* **Monotone-invariant statistics.** `dS/dtheta != 0` verified numerically for
  every headline: the injection recovers injected slopes from -0.60 to +0.30 with
  bias `<= 0.012`; the AIC ranking moves with the nuisance draw; the sensitivity
  table prints `d log B_z/d log x` for all ten inputs and none is zero.
* **Refitting on the held-out set.** RAR and AQUAL `a0` are used **frozen** at the
  SPARC-fitted values. The one model given a free `a0` is reported separately and
  loses by more, not less.
* **Silent extraction failures.** Row and column counts asserted on ingest:
  DiskMass 30 joined / 28 retained, MaNGA 1,671 rows / 240 galaxies (assertion
  raises on mismatch).
* **Test bugs that look like solver bugs.** Two found, both of that class: the
  `sigma_z^2 >= 1e-30` floor, and `h < 0` grid points in the potential models
  where `-1/b` is finite but negative and `log10` of it is silently `nan`. Both
  are now rejected rather than propagated.
* **`|Phi_b|` boundary rule.** Declared in code before any fit, three variants,
  one primary; the ranking is unchanged under all three.
* **No aperture shortcut smuggled in.** The injection reuses the reference
  aperture operator for speed; the gate `max |d slope| = 4.4e-4` over six paired
  trials against exact recomputation is recorded in `injection_results.json`.

---

## 9. What could not be established

* **Whether the trend is gravity or calibration.** Section 4 shows the two are
  the same model to 5e-15 dex on these observables. Separating them needs
  `Upsilon_K` measured to 8% *absolutely*, or `h_z` measured rather than
  inferred — and Run V established that `h_z` needs edge-on while `sigma_z` needs
  face-on, so no amount of extra data from this route fixes it.
* **An independent amplitude.** MaNGA has no scale height, so it checks the sign
  of the trend and never `B_z` itself.
* **The tensor laws through the full 2-D solve.** This lane used the
  semi-analytic `B_z = 1/mu_z`, gated by the original at 0.6%. Enough for the
  ranking (the anisotropic tensor is numerically identical to Newton in `B_z`),
  not for a 1% amplitude claim.
* **Power against a radially *varying* global law.** `M_global` was tested only
  as a radially constant factor. A law global in `Sigma_0` that also varies with
  `R/h_R` was not enumerated, and two numbers per galaxy could not constrain it.
* **Which specific systematic it is.** `Upsilon_K(Sigma_0)`, an `h_z/h_R` trend
  and a genuine flat modification are all the same model here. Section 2e bounds
  each; none is excluded.
* **A calibrated null for the *tensor* and *potential* models.** `model_null.py`
  calibrates Newton, RAR and M_global only. The `dAIC = 16.8` against
  potential depth and 58 against the isotropic tensor are uncalibrated, and by
  the lesson of statistic 2 above, uncalibrated `dAIC` values should not be read
  as evidence strengths.
* **A cross-validated version of the comparison.** `M_global`'s `p` is fitted on
  the same 28 galaxies it is scored on; AIC/BIC penalise that with a parameter
  count, not with a held-out test. With 28 objects a frozen split was not
  attempted.

---

## Files

| file | contents |
|---|---|
| `REPORT.md` | this document |
| `bz_formula.md` | item 1: the exact formula, sensitivities, the `sigma_z^2` floor |
| `vertical_audit.json` | consolidated machine-readable results |
| `vaudit_core.py` | pipeline re-instantiation + reproduction selftest |
| `inject_recover.py` | item 2: Newtonian injection-recovery, five covariance scenarios |
| `slope_stats.py` | items 2b and 3: EIV fit of `s`, error-bar forensics |
| `model_compare.py` | items 4 and 5: model comparison, pipeline predictions |
| `degeneracy_check.py` | the `M_global` = tilted-`Upsilon_K` identity |
| `model_null.py` | calibrated `dAIC` nulls under Newton-truth and RAR-truth |
| `manga_check.py` | independent sign check on resolved MaNGA profiles |
| `bz_sensitivity.py` | the measured logarithmic sensitivity table |
| `build_json.py`, `write_report.py` | assemble the deliverables |
| `*.json` | per-script results |
| `injection_run.log`, `model_compare.log` | run transcripts |

Reproduce: `python vaudit_core.py` (2.5 s, the gate), then `inject_recover.py`
(360 s), `slope_stats.py` (12 s), `model_compare.py` (~15 min),
`model_null.py` (~35 min), `degeneracy_check.py` (3 s), `manga_check.py` (3 s),
`build_json.py`, `write_report.py`.
