# The exact formula for B_z, extracted from the code

Source: `work/gravity-cluster-audit-2026-09/adyn/adyn_run.py` (script body) and
`adyn_model.py`. Reproduced and differentiated by
`vertical-audit/vaudit_core.py` + `bz_sensitivity.py`, which recover the
published slope `-0.346  68% [-0.416, -0.276]` and raw sd `0.1287` to four
decimals. Numbers below are measured, not quoted.

---

## 1. The definition, in one line

```
B_z(galaxy)  =  [ sigma_LOS_0(observed) / sigma_LOS_0(model, Newton) ]^2
```

`adyn_run.py:797`  `lbo = 2 * np.log10(sob / aN)`

* **Numerator** `sob` — DiskMass VI table 6 `sigma_LOS_0`, the amplitude of
  *their* exponential fit to the measured line-of-sight stellar dispersion
  profile, plus a draw from its tabulated error `e_sigma_LOS_0`.
  **One measured quantity.**
* **Denominator** `aN` — *not* a closed form. It is the amplitude of an
  exponential fitted, over the same radial window, to a forward-modelled
  `sigma_LOS(R)` produced by the Newtonian chain below. **Everything else in
  the problem lives here.**

## 2. The denominator, term by term (`adyn_run.py:380 newton_chain`)

```
Sigma_*0   = Upsilon_K * Sigma_L0                                (a)
Sigma_L0   = 10^(0.4 (M_K,sun + 21.572 - mu0_K,i))   Lsun/pc^2   (b)   <== ALSO THE X-AXIS
Sigma_g0   = f_gas * Sigma_*0 / f_hg^2                           (c)
Sigma_*(R) = Sigma_*0 exp(-R/h_R),  Sigma_g(R) = Sigma_g0 exp(-R/(f_hg h_R))
g_R(R)     = pi G Sigma_*0 x [I0K0 - I1K1](x/2) T(x; 2h_z/h_R) + gas term
V_c^2      = R g_R
sigma_z^2(R) = max( 2 pi G h_z [Sigma_*(R) A_ss(k) + Sigma_g(R) A_sg(k,f_hzg)]
                    - L_s(k) h_z^2 (1/R) dV_c^2/dR ,  1e-30 )     (d)
sigma_LOS(R) = sigma_z sqrt( cos^2 i + 0.5 sin^2 i (1+beta^2)/alpha^2 ) * APC(R)
aN, hN     = exponential fit of sigma_LOS over  lo <= R/h_R <= hi
```

with `A_ss = k/2` exactly, `k` the van der Kruit vertical-profile constant
(`k = (2/n)(sqrt(pi)/2) Gamma(1/n)/Gamma(1/n+1/2)`; 2 for sech^2, pi/2 for sech,
1 for exponential), and `T` the finite-thickness reduction of the radial force.

### Which is which

| quantity | status | where it enters |
|---|---|---|
| `sigma_LOS_0`, `e_sigma_LOS_0` | **measured** (DiskMass VI t.6) | numerator only |
| `mu0_K,i`, `e_mu0_K,i` | **measured** (DiskMass VI t.1) | **denominator (b) AND the x-axis** |
| `h_R`, `e_h_R` | **measured** | denominator: `T`, `dV_c^2/dR`, the R grid, the window |
| `i` (`i_TF`) | **measured**, but Tully-Fisher-derived | projection |
| `D`, `e_D` | **measured** | `h_R` in metres, and `h_z` via Bershady |
| `h_z`, `e_h_z` | **INFERRED** from `h_R` via Bershady+2010b, exponent 0.643 | (d) |
| `Upsilon_K` | prior, 0.60 dex-normal 0.15 common + 0.06 per-galaxy + 0.15 dex/mag (B-K) | (a) |
| `k` | prior, U(1.5, 2.0) | (d) via `A_ss`, `A_sg`, `L_s` |
| `alpha = sigma_z/sigma_R` | prior, 0.60 +- 0.12 | projection |
| `f_gas` | prior, 0.25 dex-normal 0.20 common + 0.15 per-galaxy — **not tabulated** | (c) |
| `f_hg`, `f_hzg` | priors, U(1.5,3.0), U(0.3,0.8) | (c), (d) |
| `lo`, `hi` | priors, U(0.2,0.5), U(1.5,2.5) | the fit window |

## 3. Which quantity appears in more than one place

**`Sigma_L0` — and only `Sigma_L0`.** It is the denominator's leading factor
*and* the abscissa of the headline regression:

```
adyn_model.py:269   g.SigmaL0 = 10.0 ** (0.4 * (MSUN_K + 21.572 - g.mu0K))
adyn_run.py:361     SigL0 = np.array([x.SigmaL0 for x in GAL])[:, None]     -> newton_chain
adyn_run.py:996     lSig  = np.log10(np.array([x.SigmaL0 for x in GAL]))    -> the x-axis
```

`h_R` also appears twice (in `h_z` through the Bershady relation, and in the
radial grid) but is not on the abscissa. `Upsilon_K`, `k` and `alpha` appear once
each, in the denominator only.

## 4. The measured logarithmic sensitivities

The prose in `adyn/REPORT.md` gives
`log B_z = 2 log sigma_z - log Upsilon_K - log h_z - log k + const`, i.e. unit
coefficients. That is the closed-form idealisation, not the pipeline. Measured by
finite difference **through the real chain** (`bz_sensitivity.py`, 0.01 dex step,
median over 28 galaxies):

| input | `d log10 B_z / d log10 x` | galaxy-to-galaxy sd | prose claims |
|---|---:|---:|---:|
| `sigma_LOS_0` (observed) | **+2.0000** | 0.0000 | +2 |
| **`Sigma_L0`** | **-0.9943** (-1.0000 at frozen aperture) | 0.0000 | (absent) |
| `Upsilon_K` | -1.0000 | 0.0000 | -1 |
| `h_z` (h_R held) | **-0.6667** | 0.0464 | -1 |
| `k` | **-0.7965** | 0.0400 | -1 |
| `h_R` (h_z, Sigma_L0 held) | -0.3048 | 0.0561 | (absent) |
| `alpha` | +0.7559 | 0.3600 | (absent) |
| `f_gas` | -0.0774 | 0.0019 | (absent) |
| inclination | -0.4402 | — | (absent) |
| distance | -0.3239 | — | (absent) |

Window shifts move `log B_z` by at most 0.009 dex.

Three of these matter for reading the previous report:

1. **`h_z` carries 0.667, not 1.** The leakage term `-L_s h_z^2 (1/R) dV_c^2/dR`
   and the thickness factor `T(2h_z/h_R)` both push back. So the h_z systematic
   propagates at 2/3 strength, and — importantly — the report's claim that the
   Bershady slope "would have to be wrong by 1.99" is computed with a coefficient
   of 1. Its own numerical scan in the same block (slope -0.366 at d=-0.30,
   -0.316 at d=+0.30) implies a response of **+0.083 per unit `d`**, so removing
   -0.340 needs **d = 4.1**, not 1.99. The conclusion is unchanged and in fact
   strengthened; the quoted figure is wrong by a factor of two.
2. **`k` carries 0.797, not 1**, for the same reason.
3. **`Sigma_L0` carries -0.994.** The abscissa is `+1 x log Sigma_L0`. This is
   the shared-denominator structure the audit exists to test — see `REPORT.md`
   item 2.

## 5. The abscissa, exactly

```
log10 Sigma_0  ==  log10 Sigma_L0  =  0.4 (M_K,sun + 21.572 - mu0_K,i)
```

K-band **luminosity** surface density in `Lsun/pc^2`, **disk only**,
**inclination-corrected**, `M_K,sun = 3.28`. It is *not* the baryonic surface
density: `Sigma_b = Upsilon_K Sigma_L0 (1 + gas)`, and `Upsilon_K` is a prior,
not a measurement. It is a **surface brightness**, hence distance-independent —
so the distance error, which is the largest per-galaxy error in the catalogue,
enters the ordinate only and cannot generate the covariance.

* range 1.544 dex (a factor of 35), variance 0.11283 dex^2
* median error 0.0280 dex (`0.4 x e_mu0_K,i`, median `e_mu0_K,i = 0.070` mag)
* so `var(eps_x)/var(x) = 0.0124` from the tabulated error alone, 0.0180 with
  the `mu0`-`h_R` decomposition degeneracy included.

## 6. A defect found in the denominator

`newton_chain` floors `sigma_z^2` at `1e-30` (line d above). When a nuisance draw
puts `h_z` far above the tabulated value for a small, thick galaxy (UGC 6918,
`h_z = 0.21`, `h_R = 1.2` kpc; UGC 1862, `0.24`/`1.4`), the leakage term
`L_s h_z^2 (1/R) dV_c^2/dR` exceeds the gravity term and `sigma_z^2` is clipped.
The exponential fit then returns a collapsed amplitude and that galaxy's
`log B_z` diverges — realised values up to **+41 dex**.

* a floored cell appears **inside the fit window** in **11 of 1600** draws,
  affecting a median of 1 galaxy;
* those draws are exactly the ones with `|slope| > 1` (6 of 1600);
* they are the sole reason the published error bar is `+- 0.173` — see
  `REPORT.md` item 3.

The floor should be replaced by a rejection (the draw is unphysical: it makes the
disk thicker than it is wide) or by an explicit `h_z < h_R/2` prior bound.
