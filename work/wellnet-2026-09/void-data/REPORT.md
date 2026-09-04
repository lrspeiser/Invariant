# void-data lane — path-integral data assembly and leverage assessment

## Bottom line

**The data exists and the test can be asked. The naive version of the test is
badly broken, and the size of the breakage is measured.**

The single most important number: **a naive regression of redshift on raw `I_q`
has a null expectation of 0.27-0.40 x c1 at 30-38 sigma.** The effect it would
"find" is roughly the size of the effect anyone would hope to find. See section 5.

## 1. Acquired (22 files, all with manifests; `MANIFEST.json`)

**DESIVAST v1.0** (DESI DR1 VAC), 8 FITS, 1.27 GB — **all 8 verified byte-exact
against DESI's own published `sha256sum` manifest**:

| algorithm | NGC | SGC | total | geometry released |
|---|---|---|---|---|
| VoidFinder | 3241 | 524 | **3765** | union of 101,863 spheres — exact |
| V2/VIDE | 1258 | 220 | **1478** | triangulated boundary, 1257 of 1478 |
| V2/REVOLVER | 1692 | 300 | **1992** | triangulated boundary, complete |
| V2/ZOBOV | 2950 | 569 | **3519** | ellipsoid axes only (no TRIANGLE HDU) |

Plus `GALZONE`: 558,745 (NGC) + 84,938 (SGC) in-survey galaxies with comoving
positions — the sample the voids were built from, so field and voids share a
sample.

**Corrected premise.** The brief expected several *independently constructed*
catalogues. VIDE, REVOLVER and ZOBOV are three *prunings of the same ZOBOV
watershed zones* — all three files carry an identical `ZONEVOID` (2950 zones) and
`GALZONE` table. There are **two** genuinely independent algorithms, not four.
VIDE vs REVOLVER correlate at r = 0.92; VoidFinder vs REVOLVER at 0.74.

**Real incompleteness found in the released product:** the VIDE `TRIANGLE` table
is missing 199 of 1258 NGC and 22 of 220 SGC voids (setdiff verified both
directions). REVOLVER is complete. VIDE `I_q` is a lower bound.

**Ingest validated against the publication:** interior counts from
`EDGE_AREA/TOT_AREA <= 0.1` give VIDE 297 (published 295), REVOLVER 389
(published 420); VoidFinder `EDGE==0` = 1489, matching the DESI doc page exactly.

**Also acquired:** SDSS DR7 VAST VoidFinder (Zenodo 11043278, 1163 voids /
39,735 holes, 7000 deg^2, z<0.1125) — genuinely independent, non-DESI;
Pantheon+SH0ES fresh from GitHub (1701x47 + STAT+SYS + STATONLY), byte-identical
to the prior-lane freeze; Cosmicflows-4 table4 (38,053 groups); DESI DR1 LSS
BGS_BRIGHT clustering + n(z); 6 megamasers and GW170817 hand-entered with
citations.

## 2. Pantheon+ covariance trap — confirmed and quantified

Real, in the official release; this copy is byte-identical to upstream.

- `sqrt(diag(STAT+SYS))` is **not** `MU_SH0ES_ERR_DIAG`: median ratio **0.713**;
  **1700 of 1701 rows** differ by >1%; range 0.118-1.008.
- Worst where it matters for low-z work: **0.564** for z<0.01, **0.449** for the
  77 Cepheid calibrators.
- For non-calibrators it is a tight multiplicative constant:
  `col/sqrt(diag) = 1.4013`, 16-84th pct 1.363-1.417.
- Quadrature excess `sqrt(col^2 - diag^2)` = median **0.148 mag**, about **5x**
  the tabulated `m_b_corr_err_VPEC` — so it is **not** the peculiar-velocity
  term. Cause documented, not diagnosed.
- `max|C - C^T| = 3.0e-8` (8-dp text rounding), symmetrised. Off-diagonal carries
  19.6% of summed covariance. Min eigenvalue 8.03e-4, condition number 3144.

## 3. Fiducial frame and validation

Flat LCDM, **Omega_m = 0.315, h = 1, comoving Mpc/h** — exactly DESIVAST's own
header values. `D_C(0.24) = 677.4038194385` vs header `677.4038194061` →
agreement to **3.2e-8 Mpc/h**. RA/Dec recovered from catalogue `X,Y,Z` to
5.7e-14 deg; `||XYZ|| - R` exactly 0.

Ray tracer validated three ways: ray through a VoidFinder maximal centre lands
inside **200/200**; V2 triangulated surfaces **watertight — odd-parity count 0 on
all 5,631 sight lines**; and an independently reconstructed density field puts
VoidFinder void centres at **delta median -0.66, 99.6% negative** versus
**-0.22, 62.6%** for random in-survey cells.

Density field: 4 Mpc/h grid, 5 Mpc/h Gaussian, CuPy FFT,
`lap phi = (3/2) Omega_m H0^2 delta`. Two stated consequences: the radial
monopole of delta is zero by construction (all surviving structure is transverse
— which is what the test needs), and delta = 0 outside the mask, so **`I_g` is a
lower bound and the least trustworthy integral; `c4` should not be fitted on this
data.**

## 4. Leverage — the dynamic range at fixed distance

Isolated with a direction scramble: `DI_q = I_q - <I_q>(r)`, where `<I_q>(r)` is
the footprint-averaged void path length (240 random in-mask directions per cap
per algorithm).

**DESI arm, n = 4,389** (4,320 CF4 + 69 SNe; cuts `path_covered_frac >= 0.5`,
`r_end >= 100` Mpc/h, declared in code before residuals):

| algorithm | sd(DI_q) Mpc/h | p5-p95 | corr(I_q,D) | VIF | corr(DI_q,D) |
|---|---|---|---|---|---|
| VoidFinder | **35.1** | 117.9 | 0.632 | 1.67 | +0.031 |
| VIDE | 33.3 | 108.1 | 0.467 | 1.28 | -0.011 |
| REVOLVER | **43.7** | 144.0 | 0.538 | 1.41 | +0.074 |
| ZOBOV (ellipsoid) | 31.8 | 101.5 | 0.744 | 2.24 | +0.168 |

**SDSS arm, n = 20,683:** sd(DI_q) = **27.2** Mpc/h, p5-p95 = 89.6,
corr(I_q,D) = 0.752, VIF = 2.30, **corr(DI_q,D) = 0.003**.

**Dynamic range at fixed distance is 0.68 to 3.46 x the mean**, by bin. In three
of four DESI bins the 5th percentile is literally **zero** — many sight lines
cross no catalogued void while others cross 124-145 Mpc/h. Matched pairs
(|dD| < 20, |dI_q| > 100 Mpc/h): **122,306** VoidFinder, **270,407** REVOLVER,
**385,104** SDSS. Example: two CF4 groups at 122.1 and 124.2 Mpc/h with
`I_q` = 102.5 vs 0.0.

**Collinearity verdict: `c1` and `c2` ARE separable** (VIF 1.28-2.30, far below
the VIF > 10 danger zone).

**But the six-term law is only separable on watershed geometry.** VIFs over
`[D, I_q, I_T, I_g, I_q^2, I_q I_T]`: VoidFinder gives 2.61 / **10.18** /
**12.13** / 3.06 / **16.52** / **18.23**, condition number **212**; REVOLVER
gives 2.45 / 4.34 / 3.97 / 3.16 / 3.47 / 3.92, condition number 25.4.
`corr(I_q, I_T)` = **-0.754** (VoidFinder) vs **+0.051** (REVOLVER). Physical
cause: inside a uniform sphere the potential is quadratic, so `T_ij ~ delta_ij`
is isotropic and `T_ij k^i k^j` loses all direction dependence — the tidal term
collapses onto a density-weighted copy of `I_q`. **Fit `c3` and `c6` only on
watershed voids; with sphere-based voids any fitted value is a parameterisation
artefact.**

## 5. The shared-denominator artefact — the decisive result

`I_q` is built from a ray truncated at `D_C(z)`, so it knows the *true* distance;
the regressor `D` is the *noisy* independent distance (6% Pantheon+, 23.5% CF4).
The regression uses `I_q` to repair `D`'s noise. Simulating the exact null —
truth is `ln(1+z) = c1 D_true`, **no path term**, real per-source errors, 2000
draws:

| estimator | null mean c2 | significance | in units of c1 |
|---|---|---|---|
| raw `I_q`, VoidFinder | 1.32e-4 | **+38.0 sigma** | **0.397** |
| raw `I_q`, REVOLVER | 9.00e-5 | **+29.6 sigma** | **0.270** |
| transverse `DI_q`, VoidFinder | 2.21e-6 | +0.46 sigma | 0.0066 |
| transverse `DI_q`, REVOLVER | 1.15e-5 | +2.95 sigma | 0.034 |

The scramble decomposition reduces the bias 60x (VoidFinder) and 24x (REVOLVER)
but **does not zero it for REVOLVER — the null must be simulated per algorithm
and subtracted, never assumed zero.**

## 6. Power

Noise model: redshift is precise, distance is the noisy axis,
`sigma_eff^2 = (sigma_v/c)^2 + (c1 sigma_D)^2`, sigma_v = 300 km/s. Median
sigma_eff = 0.0165 in ln(1+z). **No redshift residual was regressed on `I_q`
anywhere — power uses only the design matrix and noise model, so nothing is
unblinded.**

| sample | n | 3 sigma min detectable c2/c1 | equivalent dz over p5-p95 |
|---|---|---|---|
| SDSS arm | 20,683 | **2.8%** | 253 km/s |
| DESI REVOLVER | 4,389 | **4.3%** | 632 km/s |
| DESI VoidFinder | 4,389 | 5.4% | 632 km/s |
| Pantheon+ only | 73 | 17.2% | — |

**Two objects at the same distance, one behind ~90-140 Mpc/h more void, must
differ in redshift by > 250-650 km/s to be seen at 3 sigma.** Driven by CF4
numbers, not Pantheon+ precision.

**The systematic floor is worse.** Two independently built VoidFinder catalogues
on the **same 2,141 sight lines**: mean `I_q` 52.2 (DESIVAST) vs 112.1 (SDSS),
factor **2.15**; raw r = 0.462; **transverse-residual r = 0.153** — and the
transverse residual is what the decisive test uses. That inflates the realistic
threshold to tens of percent.

**Cause traced.** DESI void fraction rises 0.104 -> 0.164 -> 0.275 -> 0.290
across r = 100-300 Mpc/h; SDSS is flat at 0.502 -> 0.570. The deficit is
identical inside and outside the overlap (0.229 vs 0.230), so it is not a
localised mask hole — it is **footprint size**: DESI DR1 BGS covers 0.745 sr, so
at r = 125 Mpc/h the wedge is only ~108 Mpc/h across and a >= 10 Mpc/h void
cannot be inscribed without hitting the edge. SDSS has 2.13 sr and ~180 Mpc/h.
**DESIVAST is edge-limited at low redshift. Use SDSS VAST for z < 0.11 and
DESIVAST only for 0.11 < z < 0.24. Do not average them.** DR2 should relieve it.

## 7. Circularity — stated without softening

It enters in **four** places: (1) void positions are mapped from (RA,Dec,z) via
`r = D_C(z; Omega_m = 0.315)` — the catalogue is a redshift-space product in
Cartesian clothes; (2) the volume-limited sample definition (`MAGLIM = -20`) uses
a cosmology-dependent luminosity distance, so *which galaxies exist* depends on
it; (3) voids are found in redshift space, RSD-stretched along the line of sight,
uncorrected; (4) source endpoints use the same law.

Size, relative to fiducial at z = 0.24: linear `cz/H0` **+6.2% (+42.1 Mpc/h)**,
Milne -4.8% (-32.5), EdS **-9.7% (-66.0)**. That is 1.5-4 void radii — larger
than a void. **But the shift is shared by voids and sources** (both placed by
redshift), so ordering along the ray survives and only the differential stretch
(~5-10% across 0 < z < 0.24) matters.

The half that does **not** cancel is the endpoint. Measured directly by
recomputing every `I_q` with the ray truncated at the source's own independent
distance x a single global h (fitted **h = 0.743**, close to CF4 and SH0ES):
median `|dI_q|` = **6.20 Mpc/h** = **0.177 x sd(DI_q)** — an 18% perturbation on
the leverage variable.

**A genuine no-expansion analysis cannot reuse this catalogue as-is** — it would
have to rerun VoidFinder/V2 under its own distance law, which also changes the
sample definition. Reuse costs ~18% on the leverage variable and 5-10% on the
radial metric: tolerable for a feasibility/power study, not for a claimed
detection.

## 8. Confounds and failure-mode checklist

**Source environment:** `corr(I_q, delta_at_source)` = 0.175-0.217; projecting it
out retains **95-98%** of the transverse leverage. **Attenuation bias:** because
`DI_q` is orthogonal to `D` by construction, OLS attenuation of `c1` does not
leak into `c2`.

Checked explicitly: **shared-denominator artefacts** — CHECKED and **FOUND**,
simulated with the real error covariance (section 5). **Monotone-invariant
statistics** — CHECKED: sd(DI_q) = 35.00, 34.99, 34.55, 32.70, 25.96 Mpc/h for
hole-radius cuts 0/5/8/10/12, range/median 0.262, so dS/dtheta != 0. **Refitting
on held-out data** — N/A: **no fit of the redshift law was performed**; cuts
declared in code before residuals. **Silent extraction failures** — CHECKED:
publisher SHA-256 verified on all 12 DESI files, row/column counts asserted,
identifiers echoed; one genuine incompleteness found (VIDE triangles). **Test
bugs masquerading as solver bugs** — CHECKED via three ray-tracer validations.
**Sealed holdouts** — KiDS and wide binaries never loaded, referenced or looked
at.

## 9. What could NOT be established

- **Whether the law is true** — no fit was run; that was the assignment.
- **`I_g` usefully** — needs mass outside the survey; lower bound only.
- **A siren or megamaser arm** — n = 0 in DESI (nearest 0.75 deg, rest 4-20 deg
  away), n = 3 in SDSS. NGC 6323 (r = 76.8, `I_q` = 8.1) vs NGC 5765b (r = 82.4,
  `I_q` = 51.0) is the right shape of measurement with n = 2 and 9-21% distance
  errors.
- **Whether the r = 0.153 cross-pipeline disagreement is fully explained by
  footprint size** — evidence points there strongly, but VoidFinder was not
  rebuilt on a footprint-matched sample. Until someone does, the systematic floor
  is tens of percent, not 2.8-4.3%.
- **The cause of the Pantheon+ 1.40 ratio** — documented, not diagnosed.

## 10. Files written

`path_integrals.csv` (5,631 x 33, DESI arm) · `path_integrals_analysed.csv`
(4,389 x 47) · `path_integrals_sdss.csv` (25,123 x 20) · `results.json` ·
`results_sdss.json` · `confounds.json` · `robustness.json` · `build_meta.json` ·
`MANIFEST.json` (22 files, all SHA-256) · `raw/desivast/` (1.2 GB, verified) ·
`raw/desi_lss/` (442 MB, verified) · `raw/pantheonplus/` (63 MB) · `manifests/`
· 11 modules in `code/`.
