# Run E -- SLACS joint strong-lensing + stellar-dynamics test

**Question.** Does *one* gravitational potential, sourced by *one* stellar mass, simultaneously reproduce what the photons require (the Einstein radius) and what the stars require (the aperture velocity dispersion), for the same lens?

**Short answer.** Yes -- for all three laws, and that is not a success for any of them. The photon-side and star-side mass demands agree to within the measurement floor under Newton, the RAR and simple-mu AQUAL alike, so the Upsilon_lens vs Upsilon_dyn test does **not** discriminate between them. What *all three* fail is the absolute test: taken as stars-only theories, every one needs 1.25-2.5x more mass than the photometric stellar mass provides. The RAR and AQUAL close only 15-45% of that gap, because at SLACS Einstein radii the sample sits at g_N/a0 = 4.4 -- too deep in the Newtonian regime for a MOND-like boost to matter.

---

## 1. Sample and cut

**Cut, declared before any residual was computed** (it is the module docstring of `runE_slacs.py`, written before the analysis was run):

1. object is in the released exploration split (45 rows in `exploration-responses.tsv`);
2. tabulated quality flag `Good == "Yes"`;
3. finite `sigma`, `e_sigma`, `bSIE`;
4. finite `Re` and `b/a` in Bolton+2008 Table 4;
5. all four Grillo+2009 IMF masses present.

No cut on any computed quantity, on any residual, on redshift, on theta_E/R_e, or on sigma.

| | |
|---|---|
| rows in exploration file | 45 |
| rejected (all three by `Good != Yes`; all three also lack sigma) | 3 -- J0008-0004, J0903+4116, J1100+5329 |
| **sample analysed** | **N = 42** |

The 12 reserved-confirmation lenses were **not touched**. Their response values are not on disk and were not requested. Nothing in this run consumes the holdout.

---

## 2. What was computed

**Stellar mass model.** A **Hernquist sphere** normalised to the Grillo total stellar mass, scale radius a = R_e/1.8153 (Hernquist 1990), R_e the Bolton Table 4 effective radius used as tabulated. Chosen because it is the standard analytic de Vaucouleurs surrogate and gives closed forms for M(<r), rho(r) and Sigma(R), so every step is checkable. A **Jaffe sphere** (a = R_e/0.7447, inner slope r^-2 instead of r^-1) is carried as a shape systematic. All **four IMF variants** are carried throughout; none is picked silently.

**Laws.** (a) `g = g_N`; (b) RAR `g = g_N / (1 - exp(-sqrt(g_N/a0)))`; (c) simple-mu AQUAL `g^2 - g_N g - g_N a0 = 0`, solved as `g = (g_N + sqrt(g_N^2 + 4 g_N a0))/2`. a0 = 1.2e-10 m/s^2, fixed, never fitted. Only Upsilon (one global number) and beta (one global number) are free.

**Lensing.** M_dyn(r) = g(r) r^2/G, projected to a cylinder with the exact identity

> M_2D(R) = int_0^(pi/2) M_3D(R / sin phi) sin phi dphi

(derived by swapping the shell/cylinder integration order); theta_E is where the mean convergence equals 1, with Sigma_cr = c^2 D_S/(4 pi G D_L D_LS), flat LCDM, H0 = 70, Om = 0.3.

**Dynamics.** Spherical Jeans, nu sigma_r^2(r) = r^(-2beta) int_r^inf s^(2beta) nu(s) g(s) ds, projected with the Binney-Mamon kernel and luminosity-weighted over a circular aperture of radius 1.5 arcsec (SDSS 3-arcsec fibre). beta = 0 baseline, beta = +/-0.2 as systematics.

### Stated assumption, not hidden

For the RAR and AQUAL the photons are taken to deflect on **the same effective potential the stars feel** -- the *no-slip* case realised by TeVeS-like relativistic completions. This is the assumption that makes the test meaningful rather than trivial: if photons instead felt only the baryonic potential, both laws would fail the lensing side by their full boost factor by construction. Under no-slip the discriminant below is literally a measurement of the gravitational slip.

---

## 3. Validation

Self-tests run at the head of every execution and abort on failure. The load-bearing ones:

| check | result |
|---|---|
| angular diameter distances vs `astropy.FlatLambdaCDM` | agree to 4e-15 |
| M_2D identity vs point mass, SIS, and quadrature of the analytic Hernquist Sigma | exact to 1e-7 or better |
| theta_E end-to-end for an SIS vs textbook theta_E = 4 pi (sigma/c)^2 D_LS/D_S | 5e-15 |
| RAR/AQUAL deep-MOND and Newtonian limits; AQUAL root satisfies its own quadratic | pass |
| Jeans quadrature and radial-grid convergence, on the quantity actually used | < 1.2e-6 |
| aperture denominator equals the independently computed projected stellar mass | 5e-10 |

**External benchmark.** An SIS whose theta_E matches each observed b_SIE predicts the observed sigma with a median offset of **+0.021 dex (5.0%)** and 0.039 dex (9%) scatter. That is the classic SLACS near-isothermality result (the ~5% offset is expected because the SDSS fibre sigma used here is *not* aperture-corrected), and it independently validates the cosmology, Sigma_cr and lensing chain.

---

## 4. Where this sample lives -- and why that limits it

| quantity | median | 16-84% | range |
|---|---|---|---|
| g_N/a0 at the observed theta_E | 4.44 | 3.35 - 6.56 | 2.35 - 11.72 |
| theta_E / R_e | 0.53 | 0.43 - 0.73 | 0.21 - 0.92 |
| R_aperture / R_e | 0.71 | 0.47 - 1.03 | 0.31 - 1.60 |
| g/g_N at theta_E, RAR | 1.139 | -- | 1.034 - 1.275 |
| g/g_N at theta_E, AQUAL | 1.190 | -- | 1.079 - 1.322 |

Two things follow immediately, and they frame everything below.

1. **The two probes sample nearly the same radius** (theta_E = 0.53 R_e, fibre = 0.71 R_e). That is good for the test -- a profile-shape error largely cancels between the two sides -- but it means the test is mostly about the *relativistic sector* (do photons and stars see the same potential?) rather than about the radial run of the law.
2. **The sample is deep in the Newtonian regime.** The most a MOND-like law can supply here is ~0.06-0.08 dex. Keep that number in mind: the missing mass is 0.10-0.41 dex.

---

## 5. THE DISCRIMINANT -- does a lens need the same Upsilon for photons as for stars?

For each lens and each law, solve twice for the *absolute* mass demanded: `M_lens` such that the predicted theta_E equals b_SIE, and `M_dyn` such that the predicted aperture sigma equals the observed sigma. **Their ratio is independent of the IMF and of the catalogue stellar mass** -- each side is an absolute demand, so the stellar-population model cancels exactly. This is the cleanest available form of the requested Upsilon_lens vs Upsilon_dyn comparison.

### log10(M_required-by-LENSING / M_required-by-DYNAMICS), N = 42

| law | beta | median | median 95% CI (bootstrap) | sd | MAD |
|---|---|---|---|---|---|
| NEWTON | +0.0 | **-0.0252** | [-0.0632, -0.0046] | 0.0810 | 0.0902 |
| NEWTON | +0.2 | -0.0113 | [-0.0486, +0.0156] | 0.0814 | 0.0904 |
| NEWTON | -0.2 | -0.0413 | [-0.0761, -0.0144] | 0.0811 | 0.0872 |
| RAR | +0.0 | **-0.0677** | [-0.0944, -0.0394] | 0.0832 | 0.0879 |
| RAR | +0.2 | -0.0498 | [-0.0799, -0.0145] | 0.0846 | 0.0858 |
| RAR | -0.2 | -0.0846 | [-0.1119, -0.0521] | 0.0825 | 0.0872 |
| AQUAL | +0.0 | **-0.0693** | [-0.0986, -0.0418] | 0.0839 | 0.0894 |
| AQUAL | +0.2 | -0.0510 | [-0.0820, -0.0159] | 0.0853 | 0.0849 |
| AQUAL | -0.2 | -0.0871 | [-0.1093, -0.0547] | 0.0832 | 0.0869 |
| *SIS_REF (control, not a law)* | 0.0 | *-0.0361* | *[-0.0636, -0.0100]* | *0.0796* | *0.0847* |

Equivalently in Upsilon terms -- one global scale factor per probe, per sample:

| law | IMF | Upsilon_lens / Upsilon_IMF | Upsilon_dyn / Upsilon_IMF | log10 ratio |
|---|---|---|---|---|
| NEWTON | Salpeter/BC03 | 1.495 | 1.570 | -0.021 |
| NEWTON | Chabrier/BC03 | 2.549 | 2.591 | -0.007 |
| RAR | Salpeter/BC03 | 1.307 | 1.494 | -0.058 |
| RAR | Chabrier/BC03 | 2.225 | 2.425 | -0.037 |
| AQUAL | Salpeter/BC03 | 1.247 | 1.432 | -0.060 |
| AQUAL | Chabrier/BC03 | 2.125 | 2.326 | -0.039 |

**The answer to the question as posed: no law needs a different Upsilon for photons than for stars.** The largest offset anywhere is 0.072 dex (18%), and the two controls below show that even that is below the floor of this measurement.

### Control 1 -- the systematic floor on the offset

Median discriminant under configuration changes that are *not* gravity (beta = 0 throughout):

| configuration | NEWTON | RAR | AQUAL | RAR-NEWT | AQUAL-NEWT |
|---|---|---|---|---|---|
| baseline (Hernquist, R_e as tabulated, 1.5 arcsec) | -0.0252 | -0.0677 | -0.0693 | -0.0364 | -0.0379 |
| R_e circularised (x sqrt(b/a)) | -0.0224 | -0.0633 | -0.0654 | -0.0383 | -0.0410 |
| **Jaffe profile instead of Hernquist** | +0.0705 | +0.0328 | +0.0267 | -0.0396 | -0.0456 |
| aperture radius 1.0 arcsec | -0.0068 | -0.0522 | -0.0550 | -0.0425 | -0.0445 |
| aperture radius 2.0 arcsec | -0.0439 | -0.0841 | -0.0855 | -0.0314 | -0.0321 |

Merely swapping the stellar profile's inner slope from r^-1 to r^-2 moves the Newtonian offset by **0.096 dex**, and the full span across these variants is **0.114 dex**. The zero-point of the discriminant is therefore known no better than ~0.1 dex, which is larger than every law's offset.

### Control 2 -- an independently known-good mass model gives the same answer

`SIS_REF` is a singular isothermal sphere with one free normalisation, pushed through the *identical* lensing and Jeans code. SLACS lenses are known to be very nearly isothermal, so this is close to the right total mass profile. It returns median -0.0361 dex with sd **0.0796** -- statistically indistinguishable from the stars-only Hernquist result (-0.0252, sd 0.0810).

**Consequence: the ~0.08 dex per-lens scatter is not a property of any gravity law.** It is the noise floor of this comparison -- set by the SDSS sigma errors, the absence of a seeing convolution, the circular-vs-SIE Einstein radius convention, and the spherical approximation. A law cannot be convicted on scatter that the correct mass model also produces.

### The one thing that *is* attributable to the law

The **within-lens differential** is far more robust than any absolute offset, because the profile, aperture and R_e systematics cancel inside each lens:

| | median | 95% CI | sd | range across all systematics configs |
|---|---|---|---|---|
| RAR - NEWTON | **-0.0364** | [-0.0395, -0.0329] | 0.0106 | -0.0314 to -0.0425 |
| AQUAL - NEWTON | **-0.0379** | [-0.0427, -0.0350] | 0.0110 | -0.0321 to -0.0456 |

So the MOND-like laws make the photon side demand ~8% *less* mass than the star side, relative to Newton, robustly. The mechanism is clear: the projected lensing integral samples larger radii than the fibre-weighted Jeans integral, so the boost helps lensing more. The direction disfavours RAR and AQUAL -- but 0.036 dex is under a third of the 0.114 dex zero-point systematic, so **it is not a rejection. It is a statement that the MOND-like laws do not improve joint consistency and mildly degrade it.**

### Control 3 -- is the residual structured?

Spearman rho of the discriminant against structural variables, each with a Monte-Carlo null giving the correlation that sigma and theta_E measurement noise alone would manufacture (2000 realisations):

| variable | stars-only NEWTON: rho (p) | beyond noise null? | isothermal SIS_REF: rho (p) | beyond null? |
|---|---|---|---|---|
| thetaE_over_Rap | +0.459 (0.002) | **YES** | +0.456 (0.002) | **YES** |
| thetaE_over_Re | +0.244 (0.119) | no | -0.007 (0.964) | no |
| sigma_obs | -0.341 (0.027) | no | -0.306 (0.048) | no |
| axis_ratio | -0.310 (0.046) | **YES** | -0.375 (0.014) | **YES** |
| z_lens | +0.031 (0.845) | no | -0.013 (0.936) | no |
| Re_kpc | +0.115 (0.468) | no | +0.238 (0.129) | no |

Two trends survive the noise null: with theta_E/R_aperture, and with axis ratio. **Both are reproduced at the same amplitude by the isothermal control**, so neither is a gravity-law signature. They are what they look like: a fixed 1.5-arcsec fibre compared against a varying Einstein radius with no seeing model, and a spherical model applied to visibly flattened galaxies (median b/a = 0.78). The sigma correlation is inside the noise null and should not be interpreted.

### Error model -- deliberately not used for inference

| | |
|---|---|
| d log M / d log sigma | 2.000 |
| median fractional error on sigma | 0.0624 -> 0.0542 dex |
| d log M / d log theta_E | 1.392 |
| **assumed** fractional error on b_SIE (none is tabulated) | 0.030 -> 0.0181 dex |
| expected scatter from measurement error alone | 0.0571 dex |
| observed scatter (NEWTON, beta = 0) | 0.0810 dex |
| **implied chi2/dof** | **2.01** |

chi2/dof is 2.0, not 1. **The error model is not calibrated, so no chi2, likelihood, AIC or BIC is quoted as evidence anywhere in this run.** Effect sizes and dex scatters only. The implied unmodelled scatter is 0.057 dex, and Control 2 shows it is present for the correct mass model too.

---

## 6. THE ABSOLUTE TEST -- and here every law fails

log10(M_required / M_catalogue-stellar), median over the 42 lenses:

| law | IMF | lensing | sd | dynamics | sd |
|---|---|---|---|---|---|
| NEWTON | Salpeter/BC03 | +0.175 | 0.112 | +0.196 | 0.130 |
| NEWTON | Salpeter/M05 | +0.202 | 0.145 | +0.235 | 0.164 |
| NEWTON | Chabrier/BC03 | +0.406 | 0.110 | +0.414 | 0.125 |
| NEWTON | Kroupa/M05 | +0.391 | 0.139 | +0.406 | 0.162 |
| RAR | Salpeter/BC03 | +0.116 | 0.115 | +0.174 | 0.135 |
| RAR | Salpeter/M05 | +0.136 | 0.142 | +0.207 | 0.166 |
| RAR | Chabrier/BC03 | +0.347 | 0.112 | +0.385 | 0.130 |
| RAR | Kroupa/M05 | +0.322 | 0.136 | +0.376 | 0.164 |
| AQUAL | Salpeter/BC03 | +0.096 | 0.115 | +0.156 | 0.136 |
| AQUAL | Salpeter/M05 | +0.115 | 0.142 | +0.188 | 0.166 |
| AQUAL | Chabrier/BC03 | +0.327 | 0.112 | +0.367 | 0.131 |
| AQUAL | Kroupa/M05 | +0.302 | 0.136 | +0.358 | 0.164 |

- **Newton, stars only, needs Upsilon = 1.49 x Salpeter or 2.55 x Chabrier.** In dark-matter language that is a 33% (Salpeter) to 61% (Chabrier) non-stellar mass fraction inside theta_E -- which is what the published SLACS f_DM(< R_e/2) says. The pipeline reproduces the known answer.
- **The RAR closes 33% of the Salpeter gap and 15% of the Chabrier gap; AQUAL closes 45% and 19%.** Both still need 1.25-1.31 x Salpeter after the boost.
- This is not close. The available boost is a quarter to a fifth of the deficit, for the structural reason given in section 4: g_N/a0 = 4.4.

**How well are the stellar masses known?** Median Grillo asymmetric errors are +0.09/-0.12 dex (Salpeter/BC03) to +0.12/-0.22 dex (Kroupa/M05); catalogue round-off to 1e10 Msun contributes a median 0.005-0.009 dex (worst case 0.067 dex). So a +0.41 dex Chabrier deficit is far outside any plausible stellar-population error. A +0.17 dex Salpeter deficit is roughly 1.5x the random per-lens SPS error and would require a coherent SPS zero-point error of that size -- large, but not unimaginable. **The Chabrier/Kroupa rejection is decisive; the Salpeter rejection is strong but rests on the SPS zero-point.** This affects only the absolute test; the discriminant of section 5 is untouched by it.

A blunter statement of the same thing: with Chabrier masses and Newtonian gravity the predicted Einstein radii are 0.05-0.83 arcsec (median 0.44) against observed 0.69-1.78 arcsec. The stars-only model does not merely mis-normalise the lens -- it largely fails to produce a strong lens at the observed radius at all.

---

## 7. Verdict

| law | joint-consistency test (Upsilon_lens vs Upsilon_dyn) | absolute test (stars only) |
|---|---|---|
| (a) Newton / GR | **PASS.** -0.025 dex [-0.063, -0.005], well inside the 0.114 dex systematic floor. | **FAIL.** Needs 1.49 x Salpeter, 2.55 x Chabrier. |
| (b) RAR | **PASS**, but 0.036 dex worse than Newton (robust within-lens differential). | **FAIL.** Needs 1.31 x Salpeter, 2.23 x Chabrier. Closes 15-33% of the gap. |
| (c) AQUAL | **PASS**, but 0.038 dex worse than Newton (robust within-lens differential). | **FAIL.** Needs 1.25 x Salpeter, 2.12 x Chabrier. Closes 19-45% of the gap. |

Read as a measurement of gravitational slip at ~0.5 R_e, this run gives **log10(M_lensing / M_dynamical) = -0.025 +/- 0.029 (stat) +/- 0.06 (sys)** for a stars-only Newtonian source, and -0.036 +/- 0.027 (stat) for an isothermal source: **consistent with no slip**, which is a clean if unexciting null.

---

## 8. What would have to be true to reject each law

**(a) Newton/GR, stars only.** *Already rejected*, on the absolute test -- conditional on the Grillo photometric masses being right to better than 0.17 dex and on Upsilon not exceeding Salpeter. To rescue it you would need either a uniform +0.17 dex error in the SLACS stellar masses, or an IMF 49% heavier than Salpeter at sigma ~ 250 km/s. Note that this rejects *stars-only* Newton; it says nothing against GR + a dark halo, which is not tested here and which the numbers in section 6 in fact reproduce. *On the joint test* it would be rejected if the median discriminant exceeded the profile-shape systematic, i.e. |median| > ~0.11 dex. Observed: 0.025 dex. Not rejected.

**(b) RAR and (c) AQUAL.** *Rejected on the absolute test on the same terms as Newton, and for the same reason*: the boost is too small by a factor of three to five. To survive, you would need the SLACS stellar masses to be underestimated by 0.10-0.12 dex **and** a Salpeter IMF, simultaneously. *On the joint test*, they would be rejected if their -0.068/-0.069 dex offset could be shown to exceed the systematic floor. Concretely, **the falsifying experiment is to pin the zero-point below ~0.02 dex**, which needs three things this run could not do with the data on disk: resolved Sersic (not de Vaucouleurs) light profiles, a seeing-convolved fibre model, and elliptical rather than spherical lens models. With those in hand the observed -0.068 dex would be a >3-sigma rejection of RAR/AQUAL joint consistency. Until then it is a hint with the right sign, nothing more. Separately, both laws would fail *catastrophically* -- by their full boost on the lensing side alone -- if the no-slip assumption of section 2 were dropped. This run cannot test that; it assumes it.

**What would rescue any MOND-like law here?** Only a sample at lower acceleration. At g_N/a0 = 4.4 the maximum available enhancement is ~32% in g. **SLACS strong lenses are structurally incapable of being a decisive MOND test**, and that is the most useful general conclusion of this run.

---

## 9. Limitations, stated plainly

- **Spherical throughout.** Median b/a = 0.78, and the discriminant correlates with b/a beyond the noise null (in the isothermal control too). This is a real modelling error of order the effect being measured.
- **No seeing convolution** on the SDSS fibre; handled only by varying the aperture radius, which moves the Newtonian offset by 0.037 dex between 1.0 and 2.0 arcsec.
- **b_SIE carries no tabulated error.** 3% is assumed and flagged as an assumption; it is not a measurement, and it enters the error budget only, never a likelihood.
- **b_SIE is an SIE intermediate-axis radius** compared against a circularised theta_E.
- **No external convergence** kappa_ext is modelled.
- **R_e used as tabulated**, with the circularised variant carried as a systematic; it moves the result by 0.003 dex.
- **12 reserved-confirmation lenses untouched.** Every number here is exploration-split only.

---

## 10. Files

| file | contents |
|---|---|
| `runE_slacs.py` | the complete analysis; runs start to finish from a fresh process; self-tests abort on failure; regenerates this report |
| `runE_results.json` | every per-lens number -- distances, Sigma_cr, required masses per law, theta_E and sigma predictions per law x IMF x beta, all residuals, full summary block |
| `runE_tables.md` | verbatim console output including the full per-lens table |
| `REPORT.md` | this file |
