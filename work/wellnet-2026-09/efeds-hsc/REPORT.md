# Potential depth against RAW weak-lensing shear

## 1. Is raw shear obtainable? Yes — but not from HSC

The brief's premise was that Chiu+2022's HSC data would be reachable. It is not,
and the correction is the most important finding of this lane.

| probe | result |
|---|---|
| `hsc-release.mtk.nao.ac.jp/archive/filetree/` and four sibling routes | **HTTP 401** everywhere |
| VizieR `J/A+A/661/A11` (Chiu+2022) | one table, `tablec1` = **M500 masses only** |
| VizieR `J/ApJ/890/148` (Umetsu+2020 XXL) | properties + **NFW-fitted WL masses only** |
| e-prints 2107.05652 / 2207.12429 / 2109.07836 | `.tex`/`.bib`/`.pdf`, no data |
| `github.com/inonchiu/hsc_shear_selected_clusters` | peak catalogue + MCMC chains |

Chiu's data-availability line is "shared upon a reasonable request". The HSC
catalogue needs an account, which was not created. Their masses are exactly what
hard-constraint 2 forbids as an observable, so substituting them was not an
option.

**What exists instead: DECADE.** The DECADE metacalibration shape catalogue
released in DELVE DR3, served unauthenticated by the NOIRLab Astro Data Lab TAP
endpoint, covers the eFEDS field completely:

    POST https://datalab.noirlab.edu/tap/sync
    SELECT COUNT(*) FROM delve_dr3.decade_shear
    WHERE ra BETWEEN 126 AND 146 AND dec BETWEEN -3 AND 6   ->  14,498,544

It carries `mcal_g_{1,2}_noshear` **plus the 1p/1m/2p/2m sheared copies**, so the
response matrix is recovered rather than assumed; `mcal_w_noshear` weights;
`mcal_sel_noshear` tomographic bins; per-source `dnf_z`. About 6.8 selected
galaxies per square arcmin. **This is the observable the brief asked for.**

Two further items recovered from vector PDFs by exact content-stream coordinates,
**not digitisation**: Chiu+2022's stacked HSC profile (10 bins, total S/N 29.0;
axis log-linearity residual 5.4e-10 dex, recovered bin spacing 0.124303 against
the paper's declared 0.124304), and the eFEDS x HSC footprint (25,689 plotted
sources, RA/Dec scales agreeing to 3e-15 pt/deg). 328 of 542 eFEDS systems land
on covered cells, 60.5%, against Chiu's 313/434 = 72.1%.

**A provenance trap worth propagating.** DataCite and Zenodo return records
titled *"HSC Y3 Shape Catalog — GAMA09H Full Field"* (`10.5281/zenodo.15482851`,
`.15441596`, `.15450535`) describing precisely the eFEDS-overlapping field with
e1/e2/RA/Dec columns. **They must not be used.** The creator list includes an LLM
simulation assistant, and the records' own correction notice admits the depositor
is "not the creator or sole rights holder" and that one is "an algorithmically
scaled or pipeline-derived artifact". They are indistinguishable from the real
thing by title alone.

**Still not obtainable:** per-cluster HSC shear profiles for eFEDS (never
released); XXL-HSC as a frozen-transfer target (Umetsu's Table 3 gives (c200,
M200) constraints, not profiles, and XXL-N is at RA ~ 36 with no eFEDS overlap);
a fresh X-ray sample with resolved baryon profiles in the DECADE footprint
(eRASS1 has count rates and luminosities but **no Vikhlinin density parameters**).
CFHTLenS W2 via CADC does reach into the eFEDS box (~33 systems) and is an
available independent-instrument cross-check not run here.

## 2. Design — how much leverage does eFEDS have?

Gate first: **M_gas,500 reproduced, median mine/published 1.0022, scatter 0.0440
dex, n = 267.**

Within-class spread of x_Phi at matched g_b, five boundary rules declared in
advance (primary `fixed10Mpc`):

    fixed10Mpc 0.220 dex   fixed5Mpc 0.153   fixed3Mpc 0.115
    2xR500     0.144       10xrs     0.232
    reference: SPARC alone 0.309 | Run Z 0.185 | six-rung ladder 0.766

Collinearity against every competitor the brief names:

    quadratic in (log g_b, log r)   R^2 0.9661   residual 0.142 dex
    log M_b 0.646 | log R500 0.209 | f_gas 0.203 | z 0.138 | log T 0.039
    ALL competitors                 R^2 0.9706   residual 0.132 dex

Restricted to the radii where shear is actually measured, for the 248 training
systems: **R^2 = 0.9863, residual 0.087 dex.** So about 98.6% of potential depth
in this sample IS a function of acceleration and radius.

The Run Z identity survives on the difference: corr(resid x_Phi, resid log S |
g_b, g_b^2, r) = **+1.0000**, and corr(log S, log|dln n_e/dln r|) = **-0.9450**.
On the X-ray side potential depth is still exactly the shape factor. **What
changed is the observable.**

## 3. The measurement, and its nulls

542 systems queried, 536 with a profile, **496 pass the declared cuts (>= 4 bins
with >= 50 background sources), 3365 (system, bin) points**, z = 0.017-0.855.

**The sign convention was measured, not asserted.** All four axis-sign
combinations over the 40 most gas-massive systems (33,775 background sources):

    phi = atan2(d_dec, +d_ra cos dec)  ->  <g_t> = -0.00213   WRONG
    phi = atan2(d_dec, -d_ra cos dec)  ->  <g_t> = +0.01082   RIGHT

The DECADE/DES ellipticity basis has its first axis pointing **West**. The
+0.0108 amplitude also matches Chiu's HSC stacked profile in the same field
(0.0117 at 0.73 Mpc), an independent check on the whole chain. The first pass
used the wrong convention and produced a null signal.

| null test | result |
|---|---|
| tangential, inverse-variance mean, 3365 points | **+0.00134 +- 0.00014 = +9.6 sigma** |
| cross (B-mode) | -0.00004 +- 0.00014 = **-0.3 sigma** |
| random-point null, 246 positions | -0.00032 +- 0.00021 = -1.5 sigma; signal 4.2x residual, **PASS** |
| stacked profile S/N | 10.9 (HSC published 29.0) |
| member contamination, inner-3 / outer-5 density | **1.236, FLAGGED** |
| responsiveness d(beta-hat)/d(beta_inj) | **0.9984**, spread 1.000, **PASS** |

Lensing numerics gates: SIS 2.6e-4; Plummer 2.7e-3 / 9.5e-4; NFW against Wright &
Brainerd 5.7e-4; truncation error **not flat** (moves 2.0e-2 over r_t 25-200
Mpc). Monotone M_dyn and rho_dyn >= 0: 0 failures of 328. That gate caught a real
bug — the first Abel projection was missing the cosh Jacobian and sat at exactly
2/pi of truth *independently of every grid parameter*, the flat-error-curve
signature from the checklist.

## 4. The test

Blind split declared before residuals: **248 TRAIN / 248 HELD OUT**, alternating
eFEDS-name rank. Inside one class a class indicator has no content beyond a free
amplitude, so M1 is the strongest possible null.

| model | k | chi2 | dBIC | best fit |
|---|---|---|---|---|
| M3 + gamma log r | 2 | 1850.30 | **0.00** | gamma = -0.60 |
| M0 RAR only | 0 | 1870.72 | +5.54 | |
| M1 + free amplitude (**CLASS STEP**) | 1 | 1870.22 | +12.48 | A = -0.025 |
| M + f_gas | 2 | 1867.34 | +17.04 | -0.150 |
| M + redshift | 2 | 1867.42 | +17.12 | -0.100 |
| M4 + free a0 | 2 | 1867.79 | +17.49 | a0 x 0.05 |
| M + log T | 2 | 1868.62 | +18.32 | -0.050 |
| M + log M_b | 2 | 1868.83 | +18.53 | -0.050 |
| M2b + beta x_Phi, orthogonalised | 2 | 1869.96 | +19.66 | +0.80 |
| **M2 + beta x_Phi (HYPOTHESIS)** | 2 | 1870.22 | **+19.92** | **beta = 0.000** |

**Potential depth is last of ten on BIC and improves chi2 over the class step by
exactly 0.00.**

Frozen transfer, held-out half, 1656 points, every shape parameter frozen:

    M1 class step   1719.23   chi2/dof 1.0382    (baseline)
    M2 beta x_Phi   1719.23   +0.00
    M3 gamma log r  1719.01   +0.22    <- the training winner does NOT transfer
    M4 free a0      1714.26   +4.97

The question in its literal form — at matched g_b, does the lensing residual vary
with dPhi_b? Slope within g_b quartiles: -0.284 +- 0.740, +0.078 +- 0.673,
+0.155 +- 0.471, -0.294 +- 0.355. All consistent with zero.

Sensitivity: on the per-cluster raw shear, beta is completely stable across all
five boundary rules (+0.1, +0.1, +0.1, -0.1, -0.1), all four radial ranges, and
f_star = 0 / 0.15 / 0.30. Grid resolution 0.2; everything in the cell containing
zero.

## 5. The shared-quantity null fired, and it matters

Monte Carlo with the ACTUAL published errors on every density parameter, shear
redrawn independently:

    null expectation of beta-hat under H0: beta = 0
        -0.0666 +- 0.0101  (sd 0.0786, n = 60)  =  -6.6 sigma_MC from zero

**Noise in the X-ray density fit alone drives the naive estimator to -0.067.** A
naive analysis would have reported a significant NEGATIVE potential-depth effect
that is pure X-ray fit noise. Fifth artefact of this family in the programme, and
the check that caught it is the same one every time.

Referring the estimate to its own null:

    beta_raw (linearised, real data)   +0.0053
    E[beta_hat | H0]                   -0.0666
    beta = beta_raw - E[...]           +0.0719 +- 0.0836
                                       0.86 sigma from ZERO
                                       1.20 sigma from Run R's +0.17188

sigma(beta) = 0.084 from the H0 scatter and 0.084 from injection recovery agree.

**The test does not decide.** It is consistent with zero AND with Run R's
transferred value, and Run R's own systematic floor is beta_spurious ~ 0.096,
larger than this measurement's sigma. The Fisher forecast for the private HSC
per-cluster profiles gives sigma(beta) = 0.075 with the class step free — **the
experiment is systematics-limited, not statistics-limited, on either survey.**

## 6. Why stacked profiles cannot do this

The one public HSC stacked profile, 328 covered systems, aperture cap calibrated
on the published error bars only:

    M0 RAR only            204.32          mean g_pred/g_obs = 0.6157
    M1 + free amplitude    154.56   A = +0.180 dex
    M2 + beta x_Phi         31.82   beta = +2.000 (GRID EDGE)  dBIC +27.35
    M3 + gamma log r         4.48   gamma = -1.000             dBIC   0.00
    M4 + free a0           137.58

and across boundary rules on the same data: **beta = +2.0 (edge), +2.0 (edge),
+1.25, +0.50, +2.0 (edge)**.

That is the Run Z warning realised on a real lensing observable: with only
radial-shape information the **boundary rule determines the answer**, beta runs
to the grid edge, and a bare radius tilt beats the hypothesis by dBIC 27. The
per-cluster data, which add cross-system leverage, are stable across the
identical five rules. **A stacked profile cannot test potential depth;
per-cluster raw shear can.**

Amplitude calibration gate: pushing a single-mass NFW population through the
pipeline reproduces Chiu's own best-fit stacked model at log M500 = 14.05, against
their catalogue median 14.011 (+0.04 dex) but their shape-noise-weighted mean
13.74 (+0.31 dex). The HSC-side forward-model amplitude systematic is dominated
by stacking-weight ambiguity at ~0.3 dex.

## 7. Failure modes, each checked

Shared-quantity artefacts: simulated with the actual error covariance, **fired at
-6.6 sigma_MC**, estimate quoted against its own null. The construction
expressions were written out — x_Phi depends only on (n0^2, r_s, alpha, beta_V,
eps, z); g_t only on galaxy shapes, weights and photo-z. **They share no input**,
unlike Run Z where g_obs WAS the density log-slope. Monotone-invariance:
d(beta-hat)/d(beta_inj) = 0.9984, spread 1.000. Refitting on held-out: not done;
only the amplitude, which the null grants every model, was refitted, and the
training winner M3 did not transfer. Silent extraction: counts asserted after
every ingest, identifiers echoed; a probe of `J/A+A/689/A298` returned a
completely different catalogue (RMS YSO survey), reproducing the known trap. Test
bugs: the missing Abel cosh Jacobian, caught by the flat-vs-resolution signature
and pinned by the SIS closed form at exactly 2/pi. Non-monotone M(r): the
projection is cut at 0.8 r_ref for EVERY model including beta = 0; gate then
passes 328/328. Boundary rule: five declared in advance, primary declared first,
all reported. Reduced shear carried explicitly, g_+ = gamma/(1-kappa) with
per-bin measured <beta> and <beta^2>; kappa reaches 0.24 in the inner bin.

## 8. What could NOT be established

**The absolute amplitude.** DECADE fits A = -0.025 dex (the RAR needs no boost);
the HSC stack in the same field needs +0.18. The 0.2-0.4 dex gap is photo-z
dilution — DNF point estimates with a dz = 0.2 margin admit foreground galaxies,
where Chiu uses a 98% P(z) criterion. Only the differential is interpretable, and
the class step absorbs the amplitude by construction.

Member contamination is **flagged, not corrected** (24% inner-bin excess). It is
radius-dependent, so it can mimic a tilt — one reason the M3 tilt should not be
believed, and it did not transfer.

No frozen transfer onto a genuinely fresh sample: XXL shear is not public in
profile form and eRASS1 has no density profiles, so the transfer here is onto a
declared held-out half of eFEDS. And the mapping of a modified non-relativistic
g(r) into a deflection assumes no gravitational slip — stated, not derived.

## 9. Bottom line

Raw shear IS obtainable for eFEDS — from DECADE, not HSC — and the within-class,
same-survey, raw-shear test has now been run for the first time: 496 systems,
3365 points, a 9.6 sigma tangential detection with a clean B-mode and a passing
random-point null.

**Potential depth adds nothing.** Against the class-step null it improves chi2 by
0.00, ranks last of ten on BIC, gains nothing on frozen transfer, and its literal
differential slope is consistent with zero in every acceleration quartile.
Null-calibrated, **beta = +0.072 +- 0.084** — 0.9 sigma from zero, 1.2 sigma from
Run R's +0.17188. The measurement does not exclude the hypothesis; it fails to
find it, at a precision just below the hypothesis's own systematic floor.

The most transferable results are methodological: the shear route DOES break the
Run Z identity, and the shared-quantity null now fires on X-ray fit noise alone
at -0.067, so any future estimate must be quoted against its own null; **stacked
shear profiles are structurally incapable of testing potential depth**, with beta
running from +0.5 to the grid edge across defensible boundary rules on the stack
while staying pinned at zero on the per-cluster data; and 98.6% of x_Phi in this
sample is a function of (g_bar, r), so the residual 0.087 dex is all the leverage
that exists.
