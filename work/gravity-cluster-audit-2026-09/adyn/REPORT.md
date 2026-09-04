# A_dyn from DiskMass absolute dispersion amplitudes

## Headline

**The amplitude cannot distinguish A_dyn = 1 from A_dyn > 1, and cannot
distinguish B_z = 1 from B_z = 1.5.** Every law tested sits within 2 sigma of the
observed amplitude. The measurement is systematics-limited by a factor of 8.4,
and the limiting terms are common-mode, so more galaxies do not help.

One differential statistic *does* have power, and it points the other way from
the scale lengths in the same data. That internal tension is reported, not
resolved.

## 1. The previous run's statistic was provably blind

Holding the baryon model fixed, multiplying K_z by a constant B_0, and re-fitting
the exponential exactly as DiskMass did (`run.log` step 0f):

| B_z (constant) | fitted sigma_z,0 (km/s) | fitted h_sigma_z (arcsec) |
|---:|---:|---:|
| 0.25 | 25.87 | 31.64452 |
| 1.00 | 51.73 | 31.64452 |
| 8.00 | 146.32 | 31.64452 |

Amplitude spans **0.753 dex**; scale length moves **1.6e-15 dex**. A statistic
invariant to its own parameter across a factor of 32 cannot test it. The prior
claim "the vertical field is near-Newtonian" did not follow.

## 2. What was built

Forward chain: photometry -> Sigma_*(R), Sigma_gas(R), rho(R,z) -> K_z(R,z) ->
sigma_z(R) -> sigma_LOS(R) at tabulated i and ellipsoid shape -> 2.7" fibre
convolved with 1.5" PSF -> exponential fit over the same window -> compared
against **both** sigma_LOS,0 and h_sigma_LOS.

**The vertical force is not approximated.** Integrating the field equation in z
from the midplane (K_z(0) = 0) is exact:

    Newton   K_z = 2 pi G Sigma(<z) - (z/R) dV_c^2/dR
    RAR      K_z = nu(|g_N|/a0) K_z^N                     (QUMOND, leading order)
    AQUAL    (K_z - K_z^N) sqrt(g_R^2 + K_z^2) = K_z^N a0  (bisection)
    tensor   K_z = K_z^N / mu_z

The leakage term is identical for every law, so it cancels from B_z at leading
order. sigma_z then follows from sigma_z^2 rho_0 = Integral rho K_z dz, exact for
a z-isothermal tracer.

**The AQUAL fixed point had to be replaced.** The obvious iteration
K <- K^N (1 + a0/g) has contraction factor -> 1 in the deep-MOND limit and
silently fails to converge; it produced B_z posteriors reaching 0.036. Bisection
on the bracketed monotone root fixed it, after which RAR and AQUAL agree as they
must.

**The brief's free parameter k, made explicit -- and it is a factor of two.**
DiskMass write Sigma_dyn = sigma_z^2/(pi G k h_z) and adopt k = 1.5. k is fixed
by the *shape* of the vertical profile at fixed exponential scale height. For the
van der Kruit family rho ~ sech^(2/n)(n z / 2 h_z), the z-Jeans equation gives

    k = Int_0^inf sech^(2/n)(n u / 2) du = (2/n)(sqrt(pi)/2) Gamma(1/n)/Gamma(1/n + 1/2)

so k = 2 (sech^2, self-consistent isothermal), pi/2 = 1.5708 (sech, DiskMass's
"1.5"), 1 (exponential). **k enters sigma_z^2 linearly and spans a factor of
two.** Feeding the tabulated h_z into a sech^2(z/h_z) layer -- the naive reading
-- is a factor-2 error in B_z. It is drawn from a prior, never fixed.

**Gates, all PASS:**

| gate | result |
|---|---|
| solver vs exact Freeman disk | 1.4146e-2 |
| k(n) numeric vs Gamma closed form | 1.09e-6 |
| closed-form vs numerical Jeans, gas + leakage, 6 gal x 3 k | 2.16e-3 |
| semi-analytic K_z vs full 2-D solve | 0.4% median, 0.6% worst |
| semi-analytic g_R vs full 2-D solve | 1.0% median, 3.7% worst |
| vectorised chain vs per-galaxy object chain | 1.28e-13 |
| tensor B_z: semi-analytic 1/mu_z vs 2-D solved | 1.8824 vs 1.8928 (0.6%) |

Extrapolation audit: 0.00% on every axis of every interpolation.

## 3. Data, cuts, structural caveats

30 galaxies joined on UGC. Cuts declared before residuals: e_sigma/sigma <= 0.30,
B/D <= 0.35 -> **28 retained**.

- h_R(arcsec) D/206.265 vs h_R(kpc): median 0.9970, max 3.5%. Same quantity.
- r(log h_R, log h_z) = 0.99927. Unlike in a scale-length ratio, h_z does **not**
  cancel from the amplitude. Marginalised with per-galaxy and common-mode widths.
- **Endogeneity**: V_c sin i / sin(i_TF) = V_flat(TF) to **0.4%** for all 28. The
  deprojected circular speed *is* the Tully-Fisher prediction from M_K. So no law
  is fitted to DiskMass rotation (SPARC only); the rotation-curve *shape* is real
  and is used; the Upsilon-free A_dyn is flagged TF-conditional.
- Two independent routes to the disk light agree to +0.013 +- 0.038 dex;
  inverting the published (sigma_LOS,0, sigma_z,0, i) triples recovers
  alpha_eff = 0.561 [0.462, 0.605], consistent with DiskMass's adopted ~0.6.
- Aperture/PSF: median correction 1.0040.
- V_bar^2/V_c^2 at 2.2 h_R = 0.506 [0.286, 0.683], recovering submaximal disks.

## 4. Laws fitted to rotation curves only, then frozen

SPARC, 123 galaxies, frozen 60/20/20 split, Upsilon_3.6 = 0.5/0.7 declared:

| law | free | fitted a0 | RMS train | RMS blind |
|---|---:|---|---:|---:|
| RAR, nu = 1/(1-e^-sqrt x) | 1 | 1.084e-10 | 0.1641 | 0.1485 |
| AQUAL simple | 1 | 1.059e-10 | 0.1647 | 0.1474 |
| Newton | 0 | -- | 0.5215 | 0.5273 |

chi^2/dof (measurement errors only): amplitude 27.6 / 77.6 / 79.4 / 28.9 / 25.7;
scale length 10.5 / 20.2 / 20.0 / 11.1 / 132.9 (Newton / RAR / AQUAL / aniso /
iso). Not near 1 for any law -- the residual is nuisance-dominated. **No lnL, AIC
or BIC is quoted anywhere.**

## 5. The B_z posterior

| | B_z | log10 width |
|---|---|---:|
| **full budget** | **0.715  68% [0.468, 1.079]  95% [0.301, 1.670]** | 0.192 dex |
| statistical only | 0.811  68% [0.768, 0.853] | 0.023 dex |

Systematic floor 0.191 dex = **8.4x the statistical part**. A degeneracy, not a
noise problem.

| nuisance | sd(log10 B_z) |
|---|---:|
| Upsilon_K zero point (0.15 dex) | 0.154 |
| alpha = sigma_z/sigma_R | 0.078 |
| h_z relation zero point | 0.072 |
| k, vertical profile [1.5, 2.0] | 0.033 |
| f_gas median | 0.029 |
| gas radial scale length | 0.026 |
| (B-K) colour slope | 0.024 |
| fit window | 0.023 |
| measurement errors only | 0.015 |

**B_z predicted by each frozen law:** Newton 1.000, RAR 1.547 [1.401, 1.706],
AQUAL 1.548 [1.416, 1.691], tensor aniso 1.023, tensor iso 1.003.

| law | log10 B_z(law) | log10 B_z(obs) | diff | sigma | verdict |
|---|---:|---:|---:|---:|---|
| Newton | 0.000 | -0.146 | 0.146 | 0.76 | consistent |
| RAR | 0.189 | -0.146 | 0.335 | 1.70 | consistent |
| AQUAL simple | 0.190 | -0.146 | 0.335 | 1.71 | consistent |
| tensor aniso | 0.010 | -0.146 | 0.156 | 0.81 | consistent |
| tensor iso | 0.001 | -0.146 | 0.147 | 0.76 | consistent |

## 6. A_dyn = B_R / B_z

**(a) Predicted by each law:**

| law | B_R @1h_R | B_z @1h_R | A_dyn @1h_R | B_R @2.2 | B_z @2.2 | A_dyn @2.2 |
|---|---:|---:|---:|---:|---:|---:|
| Newton | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 |
| RAR | 1.715 | 1.565 | 1.096 | 1.915 | 1.843 | 1.040 |
| AQUAL simple | 1.736 | 1.569 | 1.106 | 1.929 | 1.837 | 1.050 |
| tensor aniso | 1.119 | 1.037 | 1.080 | 1.438 | 1.045 | **1.377** |
| tensor iso | 1.104 | 1.322 | 0.834 | 1.607 | 1.893 | 0.849 |

RAR/AQUAL sit slightly above 1 because nu and mu take |grad Phi|, which grows
with |z| as K_z -> 2 pi G Sigma, diluting the vertical boost over the layer. Only
the anisotropic tensor is far from 1 -- that is what mu_z = 1 means.

**(b) Required by the data given each law's B_R, at 2.2 h_R:** Newton 1.361
[0.888, 2.137], RAR 2.853 [2.032, 4.019], AQUAL 2.874, aniso 2.032, iso 2.309.

**(c) Direct, Upsilon-free.** B_R and B_z both carry 1/Upsilon, so the ratio does
not -- verified, not asserted: scanning Upsilon_K over 16x moves B_R by 1.20 dex
and A_dyn by **0.00 dex**, while the same code shows A_dyn responding properly to
h_z (2.24 -> 4.67), k (2.15 -> 4.42) and alpha (3.85 -> 2.84). Result:
**A_dyn = 3.75 [2.76, 5.15]**. **TF-conditional -- not the headline.**

## 7. The one live signal, and the tension it creates

A common-mode error in Upsilon_K, h_z or k moves the intercept, not the slope.
The sample spans 1.54 dex (factor 35) in central K surface density.

**d log10 B_z / d log10 Sigma_0:** observed **-0.346 +- 0.173** (2.0 sigma);
galaxy bootstrap -0.349 [-0.561, -0.071], p(slope >= 0) = **0.0095**.
Predictions: Newton 0.000, RAR -0.291, AQUAL -0.264, aniso -0.020, iso -0.055.

Audits: label-control shuffle null p = 0.000-0.008; errors inflated x1.35 by the
chi^2/dof = 1 factor before any significance; partial slopes holding B-K / mu0_K
/ h_z-h_R give 1.234 / 1.037 / 1.122 (raw 1.180); **inclination is uncorrelated
with Sigma_0 (r = -0.057)**, so an alpha error cannot fake it; an uncorrected
20 km/s instrumental floor gives -0.260; a gas fraction varying by a factor 6
with Sigma_0 gives -0.304; the Bershady h_R-h_z relation would have to be wrong
in **slope by 1.99** (published 0.643) to remove it.

**But the same data contradict this radially.** Observed h_sigma_LOS = 28.65"
against Newton 30.80, RAR 35.20, AQUAL 34.96, aniso 31.34, iso 48.16;
chi^2/dof 10.5 / 20.2 / 20.0 / 11.1 / 132.9. The *within-galaxy* radial
dependence of B_z that RAR and AQUAL require is **not** seen. So the
between-galaxy Sigma_0 trend and the within-galaxy radial trend disagree. The
economical reading is that the Sigma_0 trend is a Sigma_0-correlated systematic
not yet identified, and **it is not promoted to evidence for MOND.** It does
decisively reject the isotropic tensor (chi^2/dof ~ 133).

## 8. What the data can and cannot distinguish

**Cannot:** any pair of these laws on amplitude. Width on log10 B_z = 0.192 dex;
largest law-to-Newton separation 0.190 dex = **0.99 sigma**. Predicted A_dyn
spans 0.85-1.38 against a B_z precision of a factor 1.56 -- A_dyn = 1 and
A_dyn > 1 are not separable.

**Can:** reject the isotropic tensor from the scale length; and resolve the
surface-density dependence of B_z at 2.0 sigma / p = 0.0095, with the caveat
above.

## 9. Precision needed for a decisive test

log B_z = 2 log sigma_z - log Upsilon_K - log h_z - log k + const. For 3 sigma
separation of RAR from Newton the total budget must fall below **0.063 dex**, so
each dominant term must reach about **0.032 dex (8%)**:

| term | today | needed | factor |
|---|---:|---:|---:|
| Upsilon_K zero point | 0.154 | 0.032 | 4.9x |
| alpha = sigma_z/sigma_R | 0.078 | 0.032 | 2.5x |
| h_z relation zero point | 0.072 | 0.032 | 2.3x |
| k, vertical profile shape | 0.033 | 0.032 | 1.1x |

Concretely: Upsilon_K to 8% **absolute** (an IMF zero point, not a relative
calibration); h_z **measured** not inferred; the tracer population mix resolved
so k is pinned (DiskMass's sigma_z is a mixture of a thick old and a thin young
component, which do not share a scale height); alpha measured rather than
adopted; and an independent inclination so the rotation side stops depending on
inverted Tully-Fisher. **None of this is a matter of more galaxies.**

The differential route needs no zero points: sd(s) = 0.42 for RAR now -> N ~ 220
for sd(s) = 0.15, 495 for 0.10. For the tensor laws the predicted B_z barely
varies galaxy to galaxy (range 0.027 dex), so no sample size works -- a wider
*range in surface density* is needed, not more galaxies at the same one.

## 10. What is not claimed

- Not that B_z < 1. The posterior is centred below 1 but includes 1 comfortably,
  and the central value is the known DiskMass low-M/L result seen from the other
  side: fixing Upsilon_K = 0.6 forces a sub-Newtonian vertical field exactly as
  fixing B_z = 1 forces Upsilon_K ~ 0.43. The amplitude cannot say which.
- Not that MOND is excluded (1.70 sigma) or confirmed (section 7's tension).
- Not that the Upsilon-free A_dyn = 3.75 is a clean measurement -- its numerator
  is a Tully-Fisher prediction.
- Not that h_sigma_z is useless: it rejects the isotropic tensor decisively. It
  is blind only to the *amplitude*.

**Reproduction:** `python adyn_run.py` (full, 1240 s) or `--fast`. Needs
`axisym.py` + `data.py` from `gravitylab/` and the four DiskMass TSVs from
`acquire/`; `thickness_table.npz` rebuilds automatically (~30 s).
