# Stage 5 -- the ten benchmark alternate universes

Generated 2026-09-04T20:11:13Z; wall clock 850 s. Every number below is rendered programmatically from `results/*.json`.

The charter's requirement is not injection recovery. It is to determine which fundamentally different universes are **observationally indistinguishable** on this corpus, and to name the observation that would separate them. That is the primary deliverable; the seven Stage 5 questions follow from it.

## 0. What this lane opened

`builtins.open`, `io.open` and `numpy.load/loadtxt/genfromtxt/fromfile` were patched for the whole run. Every read was recorded; a read outside the lane root raises, and a path matching a sealed token raises **before** the read can happen.

* non-library read paths: **0**, all inside `c:/users/henry/documents/codex/2026-08-21/invariant-main-integration/work/wellnet-2026-09`
* foreign reads: **0**
* any path matching a KiDS / wide-binary / real-survey token: **NO**
* sealed tokens guarded: 14 (kids, kids-1000, kids1000, kv450, kids_dr4, ...)

This lane used no real observational data. The noise and systematics amplitudes are **declared synthetic values**, chosen to be representative of current wide-field, X-ray and IFU practice. No survey characterisation file was opened, and KiDS was deliberately excluded even as a source of a published noise model, because it is a sealed holdout for this programme.

| declared quantity | value |
| --- | --- |
| `wl_shape_noise_per_component` | 0.26 |
| `wl_source_density_arcmin2` | 20.0 |
| `wl_multiplicative_bias_sigma` | 0.02 |
| `wl_additive_bias_sigma` | 0.0005 |
| `wl_photoz_outlier_fraction` | 0.05 |
| `ifu_velocity_error_kms_at_1Re` | 8.0 |
| `ifu_psf_fwhm_arcsec` | 1.5 |
| `member_velocity_error_kms` | 30.0 |
| `xray_kT_frac_error` | 0.06 |
| `sn_peak_mag_scatter` | 0.12 |
| `sn_duration_frac_error` | 0.04 |
| `distance_frac_error` | 0.1 |
| `inclination_error_deg` | 3.0 |
| `ml_dex_scatter` | 0.11 |

## 1. The ten universes and what each emits

One **corpus** = 30 disk galaxies + 12 clusters + 200 supernovae, drawn from a shared scene library of 45 galaxies and 18 clusters (seed 20260904). The library is identical for every universe, so a pairwise separation can never come from the scene prior; nuisances and each universe's own constants are redrawn for every corpus, so a universe is a family and not a point.

Detector-level products, identical instrument for all ten:

* **galaxies** -- a PSF-convolved, flux-weighted, aperture-integrated line-of-sight velocity field on a spaxel grid with per-spaxel errors; a surface-brightness map; a vertical stellar dispersion at 1 and 2 R_d; photometric masses with M/L scatter and a radial M/L gradient; inclination, position angle and distance each with their error (the distance error propagates into both the angular-to-physical scale and the photometric mass, as it does in reality)
* **clusters** -- a per-source weak-lensing catalogue (position, e1, e2, weight, photometric source redshift with a mean bias and an outlier population) at a declared source density, carrying a multiplicative shear-calibration bias and a spatially coherent additive PSF residual; individual member sky positions and redshifts with membership probabilities; X-ray annulus **photon counts** and measured temperatures with a radially increasing non-thermal pressure fraction; SZ y in annuli; multiple-image positions and time delays wherever the lens is supercritical; and the surrounding-structure catalogue that defines the observable external axis
* **cosmology** -- supernova redshifts, peak magnitudes and light-curve **durations**

Nothing in a corpus is a mass. The gas temperature is *predicted* by each universe's own hydrostatic equilibrium and then observed with noise; the member velocity dispersion is *predicted* by a spherical Jeans solution in that universe's potential and then sampled one galaxy at a time. The analysis builds its own rotation curves from the velocity fields, its own lensing masses from raw shear, its own hydrostatic masses from counts and temperatures, and its own dynamical masses from individual member redshifts.

| universe | generative law | what makes it different |
| --- | --- | --- |
| **U01** baryons + Newton | standard gravity, baryons only | no dark matter, no modification; one common potential for matter and light |
| **U02** collisionless dark matter | standard gravity with collisionless dark matter | triaxial collisionless NFW halo with a random orientation, offset from the gas in disturbed systems |
| **U03** MOND/AQUAL scalar | MOND/AQUAL-like scalar universe | one global a0; no slip |
| **U04** environment scalar | scalar environment-dependent universe | a0 -> a0 (1 + kappa (dPhi/Phi0)^s), dPhi a gauge-safe potential difference |
| **U05** tensor vacuum, external axis | tensor, direction-dependent vacuum universe | l=2 potential from div[(I + A f(r) Q) grad Phi] = 4 pi G rho, Q locked to the EXTERNAL axis |
| **U06** reciprocal well network | reciprocal nonlocal well-network universe | reciprocal pair kernel Q_ab = 1 + B (S_a S_b/S0^2)^(q/2) at a universal coherence length |
| **U07** gravitational memory | universe with gravitational memory | response amplitude relaxes as exp(-t_merge/tau) toward the instantaneous value |
| **U08** photons/matter couple differently | photons and matter couple differently | g_light = nu^(1+zeta) g_N while g_matter = nu g_N |
| **U09** geometric path redshift | geometric path-redshift universe | redshift AND light-curve duration accrue on the low-density path |
| **U10** systematics only | realistic astrophysical and observational systematics only | standard gravity, baryons only, every systematic at 3x nominal |

**What each universe actually looks like**, as the blind pipeline measures it (medians over each arm's pool). This is the check that a mock universe looks like a universe before any separation result derived from it is believed:

| observable | U01 | U02 | U03 | U04 | U05 | U06 | U07 | U08 | U09 | U10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log10(g_obs/g_bar) at g_bar ~ 1e-11.5 m/s^2 (galaxies) | 0.03 | 0.62 | 0.61 | 0.62 | 0.61 | 0.62 | 0.65 | 0.61 | 0.61 | 0.19 |
| median outer log-slope of the rotation curve | -0.20 | 0.08 | 0.07 | 0.07 | 0.04 | 0.07 | 0.07 | 0.07 | 0.07 | -0.16 |
| log10[(g_z/g_z,bar) / (g_R/g_R,bar)] -- vertical vs radial | 0.01 | -0.29 | 0.00 | 0.00 | 0.01 | 0.00 | 0.00 | 0.00 | 0.00 | 0.02 |
| log10(dSigma_obs/dSigma_bar), 0.45-0.9 R500 (clusters, raw shear) | -0.00 | 0.86 | 0.48 | 0.55 | 0.45 | 0.68 | 0.53 | 0.53 | 0.48 | 0.06 |
| log10(M_hydrostatic/M_bar), 0.4-1.5 R500 | -0.01 | 0.74 | 0.59 | 0.65 | 0.60 | 0.62 | 0.63 | 0.60 | 0.59 | -0.06 |
| log10(M_dyn/M_bar), 0.5-2 R500 (member redshifts) | -0.17 | 0.56 | 0.51 | 0.55 | 0.51 | 0.53 | 0.55 | 0.51 | 0.51 | -0.19 |
| fraction of clusters producing multiple images | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| d log(light-curve duration) / d log(1+z_obs) | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

U1 shows no mass discrepancy and a declining outer rotation curve; U3 shows the RAR with flat curves; U2 shows the same RAR but a vertical-support deficit, a larger lensing signal, a hydrostatic mass that agrees with lensing, and it is the only universe here that produces strong-lensing arcs. U10 looks like U1 with the systematics turned up. The suite is behaving.

**Global gravity parameters only.** Each universe carries universal constants -- a0, kappa, A, B, the coherence length, tau, zeta, eps -- drawn once per corpus from a prior, so a universe is a family and not a point. Nothing is fitted per galaxy or per cluster. Distance, inclination, position angle, M/L and its radial gradient, shear calibration, photo-z bias, miscentring, velocity anisotropy and non-thermal pressure are per-object NUISANCES: they are drawn by the instrument and re-estimated by the analysis, never promoted to physics.

**The analysis freezes across channels.** It fits a flexible seven-knot scalar response nu-hat(g_bar) on a declared half of the galaxies, freezes it, evaluates it out of fold on the other half, and then freezes it again to predict the CLUSTERS. Every cluster residual quoted below is therefore a frozen cross-channel prediction, which is what makes it hard for a directional or network detector to win merely because the scalar interpolating function was imperfect.

U3 is the **base**. U4-U9 are one-knob deformations that return exactly U3 when the knob is zero, which is what makes "at what amplitude does the effect become observable" a single well-posed number per universe. U1, U2 and U10 are structurally different worlds. Fiducial knobs: `U04 kappa = 0.6`, `U05 A = 0.5`, `U06 B = 0.06`, `U07 Mamp = 0.2`, `U08 zeta = 0.1`, `U09 eps = 0.03`.

## 2. The test is sized before anything is interpreted

**Statistic.** max over {whole-corpus shrinkage-LDA discriminant, each single channel}; the look-elsewhere cost of the max is inside the critical value. A single discriminant over all 60 features is diluted by the many features that carry nothing for a given pair, so an analyst would look at the channels too. Taking the max and calibrating *the max* is the honest version of that.

A-vs-A separations -- the **same** universe, different seeds, independent nuisance draws -- over 240 tests spanning all 12 arms (20 replicates each):

* null z_max: median **1.96**, max **3.57**
* single-pair critical value (95th percentile): **2.77**, realised rate at it 0.050 [0.029, 0.085] (n=240)
* **family-wise critical value for 45 simultaneous pairs: 3.57**

Which test wins under H0 -- i.e. where the look-elsewhere cost comes from:

`clu_net` 11%, `gal_aniso` 10%, `gal_env` 10%, `clu_xray` 9%, `sn` 8%, `clu_dyn` 8%, `clu_mem` 8%, `clu_wl` 8%, `gal_rc` 6%, `clu_ep` 6%, `clu_quad` 5%, `full` 5%, `gal_vert` 5%, `clu_sl` 2%

Per-test nulls (each channel test sized separately, because section 4 quotes them):

| test | null z (95th) | null z median | null z max |
| --- | --- | --- | --- |
| `clu_mem` -- cluster residual vs disturbance proxies | 2.09 | 0.74 | 3.57 |
| `clu_xray` -- cluster X-ray hydrostatic | 2.04 | 0.68 | 3.32 |
| `gal_aniso` -- galaxy velocity-field m=3 harmonic | 2.04 | 0.78 | 3.34 |
| `sn` -- supernova Hubble residual and durations | 2.01 | 0.69 | 2.52 |
| `gal_env` -- galaxy residual vs environment | 1.99 | 0.71 | 3.14 |
| `gal_rc` -- galaxy rotation curves / RAR | 1.97 | 0.69 | 2.86 |
| `clu_wl` -- cluster weak-lensing profile | 1.94 | 0.66 | 2.44 |
| `clu_net` -- cluster shear residual vs member well network | 1.94 | 0.80 | 3.43 |
| `clu_quad` -- cluster shear quadrupole | 1.92 | 0.75 | 2.54 |
| `clu_dyn` -- cluster member dynamics | 1.90 | 0.71 | 2.50 |
| `clu_ep` -- lensing vs dynamics vs hydrostatic at matched radii | 1.90 | 0.62 | 3.06 |
| `full` -- whole corpus | 1.77 | 0.66 | 2.70 |
| `gal_vert` -- galaxy vertical vs radial support | 1.76 | 0.70 | 2.38 |
| `clu_sl` -- strong lensing | 1.28 | 0.00 | 2.72 |

**The rate above is 0.05 by construction** -- the critical value IS the 95th percentile of that sample. The charter asks for a third, untouched set. Two splits of 480 A-vs-A tests across 12 arms:

| split | nominal alpha | critical value from calibration | realised rate on the UNTOUCHED half |
| --- | --- | --- | --- |
| by replicate (same universes, independent draws) | 0.05 | 2.90 | 0.050 [0.029, 0.085] (n=240) |
| by ARM (critical value transferred to universes it was never calibrated on) | 0.05 | 2.86 | 0.062 [0.038, 0.101] (n=240) |
| by replicate | 0.01 | 3.06 | 0.033 [0.017, 0.064] (n=240) |
| by ARM | 0.01 | 3.33 | 0.008 [0.002, 0.030] (n=240) |

At a nominal 0.05 the test is correctly sized on untouched nulls (0.050 by replicate, 0.062 by arm -- the harder transfer test). **In the tail it is not:** at a nominal 0.01 the by-replicate split realises 0.033, more than three times nominal. Every verdict in this report is therefore taken at the family-wise 0.05 critical value measured here, never at a nominal tail probability read off a distribution.

Heaviest-tailed null arm: **gravitational memory**, whose own A-vs-A z95 is 3.21 against a median of 1.89 (ratio 1.70). Per-arm nulls differ, which is why the critical value is pooled across all arms rather than taken from any one.

**A bug this sizing caught.** The first implementation ranked discriminant scores with `argsort(argsort(.))`, which assigns *sequential* ranks and does not handle ties. Whenever a channel's features were degenerate -- the strong-lensing channel is identically zero in every universe that produces no arcs -- every score tied, the second group was handed the top ranks, and the AUC came out at exactly 1.0. That produced an apparent z = 4.8 between two universes that are identical in that channel *by construction*. Mid-ranks fix it, and a degenerate test now returns z = 0 with a `degenerate` flag rather than a manufactured z from a zero-variance null.

## 3. The observational equivalence-class map

Robust standardisation, Ledoit-Wolf-shrunk LDA fitted on 120 calibration corpora per universe and scored on 120 **disjoint audit** corpora (the pool of 480 draws per arm is split into quarters so the A-vs-A sizing and the A-vs-B tests run at exactly the same n), p-value from permuting the audit labels. Separated means z_max >= 3.57.

| z_max | U02 | U03 | U04 | U05 | U06 | U07 | U08 | U09 | U10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **U01** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U02** |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U03** |  |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U04** |  |  |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U05** |  |  |  |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U06** |  |  |  |  |  | **8.5** | **8.5** | **8.5** | **8.5** |
| **U07** |  |  |  |  |  |  | **8.5** | **8.5** | **8.5** |
| **U08** |  |  |  |  |  |  |  | **8.5** | **8.5** |
| **U09** |  |  |  |  |  |  |  |  | **8.5** |

Bold = separated. Italic = **not** separated at the family-wise level: the same observational equivalence class on this corpus. z is capped at 8.5, the resolution of a permutation null at this sample size; a capped entry means "separated with certainty by one corpus", not a measured significance.

**Every one of the 45 pairs separates at the fiducial knob settings**, every one at the capped z_max of 8.5. The whole-corpus AUC alone runs from 0.793 to 1.000 -- AUC is the probability that a SINGLE corpus from one universe scores above a single corpus from the other, so an AUC of 1.000 means one survey of this size tells them apart with certainty, and where the whole-corpus AUC is lower it is a single channel that saturates instead. The equivalence classes at fiducial amplitude are therefore all singletons.

That is a result, not a failure: it says the fiducial amplitudes chosen a priori for U4-U9 sit **far above** what this corpus can already see. The scientifically live map is the one at the amplitudes where each effect is only just visible -- section 3.2. The fiducial map's value is that it verifies the suite can in principle tell every pair apart, including the two hardest structural pairs: U2 dark matter vs U3 MOND, and U10 systematics-only vs everything.

**Equivalence classes on this corpus:**

* { **U01** baryons + Newton }
* { **U02** collisionless dark matter }
* { **U03** MOND/AQUAL scalar }
* { **U04** environment scalar }
* { **U05** tensor vacuum, external axis }
* { **U06** reciprocal well network }
* { **U07** gravitational memory }
* { **U08** photons/matter couple differently }
* { **U09** geometric path redshift }
* { **U10** systematics only }

**Which test does the separating,** and what observation that test corresponds to:

| pair | z_max | winning test | its null z95 | the observation it uses |
| --- | --- | --- | --- | --- |
| U01 vs U02 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U03 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U04 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U05 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U06 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U01 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U03 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U04 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U05 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U06 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U02 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U04 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U05 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U06 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U08 | 8.5 | `clu_wl` (cluster weak-lensing profile) | 1.94 | deeper wide-field shear around the same clusters |
| U03 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U03 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U05 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U06 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U04 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U05 vs U06 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U05 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U05 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U05 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U05 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U06 vs U07 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U06 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U06 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U06 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U07 vs U08 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U07 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U07 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U08 vs U09 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U08 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |
| U09 vs U10 | 8.5 | `full` (whole corpus) | 1.77 | the full corpus jointly |

**Channel attribution is a validation of the harness, not just a summary.** The top channels for the diagnostic pairs land exactly where the physics says they should:

| pair | top channels by excess over their own null |
| --- | --- |
| U02 vs U03 | `clu_sl` 8.5, `gal_vert` 8.5, `full` 8.5, `clu_ep` 8.5 |
| U03 vs U04 | `full` 8.5, `clu_dyn` 8.5, `clu_wl` 8.5, `clu_xray` 8.5 |
| U03 vs U05 | `full` 8.5, `clu_quad` 8.5, `gal_rc` 8.5, `gal_aniso` 8.5 |
| U03 vs U06 | `full` 8.5, `clu_ep` 8.5, `clu_wl` 8.5, `clu_xray` 8.5 |
| U03 vs U07 | `full` 8.5, `gal_rc` 8.5, `clu_xray` 8.5, `clu_dyn` 8.3 |
| U03 vs U08 | `clu_wl` 8.5, `full` 8.0, `clu_ep` 6.2, `clu_mem` 3.0 |
| U03 vs U09 | `full` 8.5, `sn` 8.5, `clu_xray` 3.5, `clu_quad` 0.9 |

U3 vs U9 is the cleanest check: those two universes are identical in every gravity channel by construction and differ only in the redshift branch, and the supernova channel is the only one that fires. Nothing leaks between channels.

### 3.2 The map at THRESHOLD amplitude -- where the question is live

Each deformation's knob is reset to the amplitude at which its own E3 scan just reaches the family-wise critical value against the base U3. Those amplitudes come from the scans, not from a hand choice:

| universe | knob | fiducial | threshold amplitude | how it was set |
| --- | --- | --- | --- | --- |
| U04 environment scalar | `kappa` | 0.6 | 0.0689 | amplitude at which the scan reaches the family-wise critical value against U3 |
| U05 tensor vacuum, external axis | `A` | 0.5 | 0.0200 | amplitude at which the scan reaches the family-wise critical value against U3 |
| U06 reciprocal well network | `B` | 0.06 | 0.0072 | amplitude at which the scan reaches the family-wise critical value against U3 |
| U07 gravitational memory | `Mamp` | 0.2 | 0.0285 | amplitude at which the scan reaches the family-wise critical value against U3 |
| U08 photons/matter couple differently | `zeta` | 0.1 | 0.0467 | amplitude at which the scan reaches the family-wise critical value against U3 |
| U09 geometric path redshift | `eps` | 0.03 | 0.0081 | amplitude at which the scan reaches the family-wise critical value against U3 |

U2 and U10 have no amplitude knob and are carried along unchanged. The question is then the one the charter actually asks: at a common, just-detectable observable amplitude, are two fundamentally different modifications distinguishable from **each other**?

#### THRESHOLD set (each knob at its threshold amplitude)

Sized on its own A-vs-A nulls: null z_max median 1.89, single-pair critical 2.71, family-wise over 36 pairs **3.55**.

| z_max | U02 | U10 | U04 | U05 | U06 | U07 | U08 | U09 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **U03** | **8.5** | **8.5** | _2.3_ | _3.4_ | **4.7** | _3.3_ | **4.6** | _3.2_ |
| **U02** |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U10** |  |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U04** |  |  |  | _3.1_ | _1.8_ | _1.6_ | _2.3_ | **4.5** |
| **U05** |  |  |  |  | _3.3_ | **3.7** | _3.5_ | **4.3** |
| **U06** |  |  |  |  |  | _2.2_ | _1.6_ | **5.2** |
| **U07** |  |  |  |  |  |  | _3.3_ | **3.8** |
| **U08** |  |  |  |  |  |  |  | **4.9** |

**Equivalence classes:**

* { **U03** MOND/AQUAL scalar, **U04** environment scalar, **U05** tensor vacuum, external axis, **U06** reciprocal well network, **U07** gravitational memory, **U08** photons/matter couple differently, **U09** geometric path redshift }
* { **U02** collisionless dark matter }
* { **U10** systematics only }

#### HALF set (each knob at HALF its threshold, so none is separable from U3)

Sized on its own A-vs-A nulls: null z_max median 1.92, single-pair critical 2.87, family-wise over 36 pairs **4.13**.

| z_max | U02 | U10 | U04 | U05 | U06 | U07 | U08 | U09 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **U03** | **8.5** | **8.5** | _2.7_ | _2.4_ | _1.5_ | _3.0_ | _3.0_ | _2.9_ |
| **U02** |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U10** |  |  | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** | **8.5** |
| **U04** |  |  |  | _1.7_ | _2.0_ | _1.6_ | _1.0_ | _2.4_ |
| **U05** |  |  |  |  | _1.6_ | _2.3_ | _2.6_ | _1.9_ |
| **U06** |  |  |  |  |  | _1.0_ | _1.3_ | _3.1_ |
| **U07** |  |  |  |  |  |  | _1.6_ | _1.5_ |
| **U08** |  |  |  |  |  |  |  | _2.6_ |

**Equivalence classes:**

* { **U03** MOND/AQUAL scalar, **U04** environment scalar, **U05** tensor vacuum, external axis, **U06** reciprocal well network, **U07** gravitational memory, **U08** photons/matter couple differently, **U09** geometric path redshift }
* { **U02** collisionless dark matter }
* { **U10** systematics only }

## 4. For every indistinguishable pair, the missing observation

Every pair below is one the corpus **cannot** separate. For each, the table names the single most informative channel and then reports, from direct simulation, which of three concrete improvements actually buys the separation.

**Read the channel column with care.** For a pair that does not clear the family-wise threshold, the *identity* of the best channel is itself subject to the look-elsewhere effect across 13 channels: it is the least uninformative channel on this realisation, not a physical attribution. The channel column is reliable only for the SEPARATED pairs in section 3, where the winning channel clears the family-wise critical value. What is reliable here is the oracle column, because each oracle is a fresh end-to-end simulation of a different survey, not a re-reading of the same one.

| pair | amplitude set | z_max | AUC | most informative channel | what separates them |
| --- | --- | --- | --- | --- | --- |
| **U03** MOND/AQUAL scalar vs **U04** environment scalar | THRESHOLD amplitude set | 2.30 | 0.537 | `clu_xray` | **none of the three -- see below** |
| **U03** MOND/AQUAL scalar vs **U05** tensor vacuum, external axis | THRESHOLD amplitude set | 3.38 | 0.559 | `gal_aniso` | noise x0.25, systematics x0.25 |
| **U03** MOND/AQUAL scalar vs **U07** gravitational memory | THRESHOLD amplitude set | 3.31 | 0.549 | `clu_xray` | systematics x0.25 |
| **U03** MOND/AQUAL scalar vs **U09** geometric path redshift | THRESHOLD amplitude set | 3.18 | 0.575 | `sn` | noise x0.25, systematics x0.25, survey x1.5 |
| **U04** environment scalar vs **U05** tensor vacuum, external axis | THRESHOLD amplitude set | 3.11 | 0.616 | `clu_dyn` | systematics x0.25 |
| **U04** environment scalar vs **U06** reciprocal well network | THRESHOLD amplitude set | 1.85 | 0.519 | `clu_quad` (*) | noise x0.25, systematics x0.25 |
| **U04** environment scalar vs **U07** gravitational memory | THRESHOLD amplitude set | 1.58 | 0.560 | `clu_wl` (*) | systematics x0.25 |
| **U04** environment scalar vs **U08** photons/matter couple differently | THRESHOLD amplitude set | 2.27 | 0.584 | `clu_xray` (*) | noise x0.25 |
| **U05** tensor vacuum, external axis vs **U06** reciprocal well network | THRESHOLD amplitude set | 3.25 | 0.622 | `gal_aniso` | noise x0.25, systematics x0.25, survey x1.5 |
| **U05** tensor vacuum, external axis vs **U08** photons/matter couple differently | THRESHOLD amplitude set | 3.46 | 0.594 | `clu_wl` | noise x0.25, systematics x0.25 |
| **U06** reciprocal well network vs **U07** gravitational memory | THRESHOLD amplitude set | 2.19 | 0.548 | `clu_wl` | noise x0.25, systematics x0.25, survey x1.5 |
| **U06** reciprocal well network vs **U08** photons/matter couple differently | THRESHOLD amplitude set | 1.60 | 0.506 | `clu_net` (*) | noise x0.25 |
| **U07** gravitational memory vs **U08** photons/matter couple differently | THRESHOLD amplitude set | 3.27 | 0.624 | `clu_wl` | noise x0.25, systematics x0.25, survey x1.5 |
| **U03** MOND/AQUAL scalar vs **U04** environment scalar | HALF amplitude set | 2.71 | 0.539 | `clu_net` | **none of the three -- see below** |
| **U03** MOND/AQUAL scalar vs **U05** tensor vacuum, external axis | HALF amplitude set | 2.38 | 0.510 | `clu_wl` | systematics x0.25 |
| **U03** MOND/AQUAL scalar vs **U06** reciprocal well network | HALF amplitude set | 1.51 | 0.482 | `clu_ep` (*) | noise x0.25 |
| **U03** MOND/AQUAL scalar vs **U07** gravitational memory | HALF amplitude set | 3.00 | 0.517 | `clu_net` | **none of the three -- see below** |
| **U03** MOND/AQUAL scalar vs **U08** photons/matter couple differently | HALF amplitude set | 3.04 | 0.548 | `clu_wl` | **none of the three -- see below** |
| **U03** MOND/AQUAL scalar vs **U09** geometric path redshift | HALF amplitude set | 2.87 | 0.518 | `clu_net` | noise x0.25, systematics x0.25 |
| **U04** environment scalar vs **U05** tensor vacuum, external axis | HALF amplitude set | 1.67 | 0.562 | `clu_dyn` (*) | **none of the three -- see below** |
| **U04** environment scalar vs **U06** reciprocal well network | HALF amplitude set | 2.04 | 0.512 | `clu_dyn` | noise x0.25 |
| **U04** environment scalar vs **U07** gravitational memory | HALF amplitude set | 1.59 | 0.540 | `clu_wl` (*) | **none of the three -- see below** |
| **U04** environment scalar vs **U08** photons/matter couple differently | HALF amplitude set | 1.01 | 0.537 | `gal_aniso` (*) | noise x0.25 |
| **U04** environment scalar vs **U09** geometric path redshift | HALF amplitude set | 2.44 | 0.562 | `sn` | noise x0.25, systematics x0.25 |
| **U05** tensor vacuum, external axis vs **U06** reciprocal well network | HALF amplitude set | 1.62 | 0.561 | `clu_xray` (*) | noise x0.25 |
| **U05** tensor vacuum, external axis vs **U07** gravitational memory | HALF amplitude set | 2.30 | 0.569 | `clu_xray` | **none of the three -- see below** |
| **U05** tensor vacuum, external axis vs **U08** photons/matter couple differently | HALF amplitude set | 2.63 | 0.530 | `clu_dyn` | noise x0.25 |
| **U05** tensor vacuum, external axis vs **U09** geometric path redshift | HALF amplitude set | 1.86 | 0.531 | `sn` (*) | noise x0.25, systematics x0.25 |
| **U06** reciprocal well network vs **U07** gravitational memory | HALF amplitude set | 1.05 | 0.495 | `clu_mem` (*) | noise x0.25 |
| **U06** reciprocal well network vs **U08** photons/matter couple differently | HALF amplitude set | 1.29 | 0.490 | `clu_dyn` (*) | **none of the three -- see below** |
| **U06** reciprocal well network vs **U09** geometric path redshift | HALF amplitude set | 3.10 | 0.557 | `sn` | noise x0.25, systematics x0.25 |
| **U07** gravitational memory vs **U08** photons/matter couple differently | HALF amplitude set | 1.63 | 0.557 | `clu_wl` (*) | **none of the three -- see below** |
| **U07** gravitational memory vs **U09** geometric path redshift | HALF amplitude set | 1.52 | 0.530 | `clu_xray` (*) | noise x0.25, systematics x0.25 |
| **U08** photons/matter couple differently vs **U09** geometric path redshift | HALF amplitude set | 2.62 | 0.598 | `gal_vert` | noise x0.25, systematics x0.25 |

(*) the channel does not clear its own sized null, so it is the least uninformative channel rather than a detection; for those pairs no channel in this corpus carries usable information about the difference.

The three improvements, each simulated end to end rather than extrapolated:

* **noise x0.25** -- 16x the effective weak-lensing source density, and 4x better spectroscopic, temperature and light-curve precision, at unchanged systematics
* **systematics x0.25** -- shear calibration, photo-z, M/L and its radial gradient, inclination, miscentring and non-thermal pressure all controlled 4x better, at unchanged statistical noise
* **survey x1.5** -- 45 galaxies and 18 clusters per corpus instead of 30 and 12

### The 9 pairs that none of the three improvements separates

#### U03 MOND/AQUAL scalar  vs  U04 environment scalar  (THRESHOLD amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 2.30, whole-corpus AUC = 0.537, family-wise threshold 3.55), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_xray` (cluster X-ray hydrostatic) at z = 2.30 against its own sized null of 2.04 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be resolved temperature profiles to larger radius with controlled non-thermal pressure; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 1.45 | `clu_net` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.36 | `clu_quad` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.67 | `clu_ep` |
| 5x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U04.** On the full corpus the knob `kappa` reaches the family-wise threshold at 0.06220, z = 3 at 0.04961 and z = 5 at 0.08413. A null below that amplitude carries no information.

#### U03 MOND/AQUAL scalar  vs  U04 environment scalar  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 2.71, whole-corpus AUC = 0.539, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_net` (cluster shear residual vs member well network) at z = 2.71 against its own sized null of 1.94 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be shear measured at the positions of individual member galaxies, with a member catalogue complete enough to build the well network; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 1.33 | `clu_ep` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.19 | `clu_quad` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.31 | `clu_ep` |
| 3x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U04.** On the full corpus the knob `kappa` reaches the family-wise threshold at 0.06220, z = 3 at 0.04961 and z = 5 at 0.08413. A null below that amplitude carries no information.

#### U03 MOND/AQUAL scalar  vs  U07 gravitational memory  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 3.00, whole-corpus AUC = 0.517, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_net` (cluster shear residual vs member well network) at z = 3.00 against its own sized null of 1.94 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be shear measured at the positions of individual member galaxies, with a member catalogue complete enough to build the well network; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 1.47 | `sn` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.36 | `gal_env` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.18 | `sn` |
| 3x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U07.** On the full corpus the knob `Mamp` reaches the family-wise threshold at 0.03487, z = 3 at 0.02863 and z = 5 at 0.05000. A null below that amplitude carries no information.

#### U03 MOND/AQUAL scalar  vs  U08 photons/matter couple differently  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 3.04, whole-corpus AUC = 0.548, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_wl` (cluster weak-lensing profile) at z = 3.04 against its own sized null of 1.94 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be deeper wide-field shear around the same clusters; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 3.02 | `clu_ep` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.63 | `sn` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.24 | `gal_rc` |
| 3x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U08.** On the full corpus the knob `zeta` reaches the family-wise threshold at 0.03430, z = 3 at 0.03030 and z = 5 at 0.04842. A null below that amplitude carries no information.

#### U04 environment scalar  vs  U05 tensor vacuum, external axis  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 1.67, whole-corpus AUC = 0.562, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_dyn` (cluster member dynamics) at z = 1.65 against its own sized null of 1.90 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be more member redshifts per cluster, and to larger clustercentric radius; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 2.37 | `clu_net` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 3.81 | `gal_aniso` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.03 | `clu_mem` |
| 9x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U04.** On the full corpus the knob `kappa` reaches the family-wise threshold at 0.06220, z = 3 at 0.04961 and z = 5 at 0.08413. A null below that amplitude carries no information.

**Power statement for U05.** On the full corpus the knob `A` reaches the family-wise threshold at 0.01893, z = 3 at 0.01207 and z = 5 at 0.03786. A null below that amplitude carries no information.

#### U04 environment scalar  vs  U07 gravitational memory  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 1.59, whole-corpus AUC = 0.540, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_wl` (cluster weak-lensing profile) at z = 1.59 against its own sized null of 1.94 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be deeper wide-field shear around the same clusters; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 2.09 | `clu_mem` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.25 | `gal_env` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.20 | `clu_wl` |
| 10x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U04.** On the full corpus the knob `kappa` reaches the family-wise threshold at 0.06220, z = 3 at 0.04961 and z = 5 at 0.08413. A null below that amplitude carries no information.

**Power statement for U07.** On the full corpus the knob `Mamp` reaches the family-wise threshold at 0.03487, z = 3 at 0.02863 and z = 5 at 0.05000. A null below that amplitude carries no information.

#### U05 tensor vacuum, external axis  vs  U07 gravitational memory  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 2.30, whole-corpus AUC = 0.569, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_xray` (cluster X-ray hydrostatic) at z = 2.30 against its own sized null of 2.04 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be resolved temperature profiles to larger radius with controlled non-thermal pressure; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 2.53 | `clu_xray` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 3.43 | `gal_aniso` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.23 | `clu_xray` |
| 5x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U05.** On the full corpus the knob `A` reaches the family-wise threshold at 0.01893, z = 3 at 0.01207 and z = 5 at 0.03786. A null below that amplitude carries no information.

**Power statement for U07.** On the full corpus the knob `Mamp` reaches the family-wise threshold at 0.03487, z = 3 at 0.02863 and z = 5 at 0.05000. A null below that amplitude carries no information.

#### U06 reciprocal well network  vs  U08 photons/matter couple differently  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 1.29, whole-corpus AUC = 0.490, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_dyn` (cluster member dynamics) at z = 1.29 against its own sized null of 1.90 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be more member redshifts per cluster, and to larger clustercentric radius; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 3.45 | `clu_wl` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.18 | `sn` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.54 | `clu_quad` |
| 15x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U06.** On the full corpus the knob `B` reaches the family-wise threshold at 0.00791, z = 3 at 0.00726 and z = 5 at 0.00954. A null below that amplitude carries no information.

**Power statement for U08.** On the full corpus the knob `zeta` reaches the family-wise threshold at 0.03430, z = 3 at 0.03030 and z = 5 at 0.04842. A null below that amplitude carries no information.

#### U07 gravitational memory  vs  U08 photons/matter couple differently  (HALF amplitude set)

> These theories belong to the same observational equivalence class on this corpus (z_max = 1.63, whole-corpus AUC = 0.557, family-wise threshold 4.13), and none of a 16x deeper survey, 4x better systematics control, or a 1.5x larger sample separates them.

The least uninformative channel is `clu_wl` (cluster weak-lensing profile) at z = 1.63 against its own sized null of 1.94 -- suggestive at best, and subject to the look-elsewhere effect across 13 channels. On that reading the missing observation would be deeper wide-field shear around the same clusters; the oracle table below is the harder evidence.

| improvement | measured z_max | winning test |
| --- | --- | --- |
| noise x0.25: 16x the effective source density and 4x the spectroscopic / temperature precision | 3.26 | `full` |
| systematics x0.25: shear calibration, photo-z, M/L, inclination, miscentring and non-thermal pressure all 4x better controlled | 2.85 | `clu_net` |
| survey x1.5: 45 galaxies and 18 clusters per corpus | 2.77 | `clu_xray` |
| 9x the corpus at unchanged precision (sqrt-N extrapolation, not simulated) | 5.0 by construction | -- |

**Power statement for U07.** On the full corpus the knob `Mamp` reaches the family-wise threshold at 0.03487, z = 3 at 0.02863 and z = 5 at 0.05000. A null below that amplitude carries no information.

**Power statement for U08.** On the full corpus the knob `zeta` reaches the family-wise threshold at 0.03430, z = 3 at 0.03030 and z = 5 at 0.04842. A null below that amplitude carries no information.


## 5. At what amplitude does each effect become observable

Each row is the knob scan of one deformation of U3, tested against U3 itself with the same statistic and the same calibration/audit discipline.

| universe | knob | fiducial | z at fiducial | z reaches 3.6 at | z=5 at | responsive? |
| --- | --- | --- | --- | --- | --- | --- |
| **U04** environment scalar | `kappa` | 0.6 | 8.50 | 0.0622 | 0.0841 | yes (dz/dlog10(amp) = 8.3 +/- 1.2 on 5 unsaturated points) |
| **U05** tensor vacuum, external axis | `A` | 0.5 | 8.50 | 0.0189 | 0.0379 | yes (dz/dlog10(amp) = 4.1 +/- 0.9 on 5 unsaturated points) |
| **U06** reciprocal well network | `B` | 0.06 | 8.50 | 0.0079 | 0.0095 | yes (dz/dlog10(amp) = 11.1 +/- 2.3 on 4 unsaturated points) |
| **U07** gravitational memory | `Mamp` | 0.2 | 8.50 | 0.0349 | 0.0500 | yes (dz/dlog10(amp) = 6.7 +/- 0.7 on 6 unsaturated points) |
| **U08** photons/matter couple differently | `zeta` | 0.1 | 8.42 | 0.0343 | 0.0484 | yes (dz/dlog10(amp) = 10.5 +/- 2.7 on 5 unsaturated points) |
| **U09** geometric path redshift | `eps` | 0.03 | 8.50 | 0.0082 | 0.0116 | yes (dz/dlog10(amp) = 8.9 +/- 2.9 on 4 unsaturated points) |

Full scans -- calibrated z_max against U3, and the test that won:

* **U04** `kappa`: 0 -> 2.2 (`gal_env`), 0.0375 -> 2.5 (`clu_xray`), 0.0630672 -> 3.6 (`clu_xray`), 0.106066 -> 6.5 (`clu_xray`), 0.15 -> 6.4 (`clu_xray`), 0.178381 -> 8.3 (`clu_xray`), 0.3 -> 8.5 (`clu_xray`), 0.6 -> 8.5 (`full`), 1.2 -> 8.5 (`full`), 2.4 -> 8.5 (`full`)
* **U05** `A`: 0 -> 1.3 (`clu_net`), 0.00625 -> 2.2 (`gal_vert`), 0.0125 -> 3.1 (`gal_aniso`), 0.025 -> 4.0 (`gal_aniso`), 0.025 -> 3.6 (`gal_aniso`), 0.05 -> 6.3 (`gal_aniso`), 0.1 -> 8.5 (`full`), 0.25 -> 8.5 (`full`), 0.5 -> 8.5 (`full`), 1 -> 8.5 (`full`), 2 -> 8.5 (`full`)
* **U06** `B`: 0 -> 2.5 (`clu_xray`), 0.00375 -> 1.6 (`gal_env`), 0.00630672 -> 2.2 (`clu_quad`), 0.0106066 -> 5.9 (`clu_wl`), 0.015 -> 8.0 (`clu_wl`), 0.0178381 -> 8.5 (`clu_wl`), 0.03 -> 8.5 (`full`), 0.06 -> 8.5 (`full`), 0.12 -> 8.5 (`full`), 0.24 -> 8.5 (`full`)
* **U07** `Mamp`: 0 -> 2.5 (`clu_net`), 0.0125 -> 1.8 (`clu_quad`), 0.025 -> 2.7 (`clu_xray`), 0.05 -> 4.9 (`clu_xray`), 0.05 -> 5.9 (`clu_xray`), 0.1 -> 7.7 (`clu_xray`), 0.1 -> 7.3 (`clu_xray`), 0.2 -> 8.5 (`full`), 0.4 -> 8.5 (`full`), 0.8 -> 8.5 (`full`)
* **U08** `zeta`: 0 -> 3.0 (`clu_xray`), 0.025 -> 2.2 (`clu_xray`), 0.0353553 -> 3.7 (`clu_wl`), 0.05 -> 5.2 (`clu_wl`), 0.05 -> 3.8 (`clu_wl`), 0.0707107 -> 7.5 (`clu_wl`), 0.1 -> 8.4 (`clu_wl`), 0.2 -> 8.5 (`full`), 0.4 -> 8.5 (`full`)
* **U09** `eps`: 0 -> 1.8 (`clu_ep`), 0.0075 -> 3.3 (`sn`), 0.0106066 -> 4.5 (`sn`), 0.015 -> 6.7 (`sn`), 0.015 -> 5.2 (`sn`), 0.0212132 -> 8.5 (`sn`), 0.03 -> 8.5 (`full`), 0.06 -> 8.5 (`full`), 0.12 -> 8.5 (`full`)

**Responsiveness of each named detector to its own knob**, d(detector)/d(knob) with its standard error. Where this is consistent with zero the detector is blind to that physics and has set no upper limit. Note that a scan z of 8.5 is the permutation-null CAP, so the sensitivity above is fitted only on the unsaturated points and against log amplitude:

| universe | knob | detector | d(detector)/d(knob) | responsive? |
| --- | --- | --- | --- | --- |
| U04 | `kappa` | `env` **(primary)** | 0.0033 +/- 0.0003 | yes |
| U04 | `kappa` | `gal_aniso` | 0.0011 +/- 0.0004 | yes |
| U04 | `kappa` | `memory` | -0.0179 +/- 0.0073 | yes |
| U04 | `kappa` | `ep_slip` | 0.0240 +/- 0.0022 | yes |
| U05 | `A` | `gal_aniso` **(primary)** | 0.4199 +/- 0.0110 | yes |
| U05 | `A` | `aniso_ext` | 0.0116 +/- 0.0000 | yes |
| U05 | `A` | `network` | -0.0010 +/- 0.0000 | yes |
| U05 | `A` | `memory` | -0.0777 +/- 0.0110 | yes |
| U05 | `A` | `ep_slip` | -0.0797 +/- 0.0013 | yes |
| U05 | `A` | `env` | -0.0529 +/- 0.0040 | yes |
| U06 | `B` | `network` **(primary)** | -0.0008 +/- 0.0005 | **NO -- no limit set** |
| U06 | `B` | `memory` | -0.9033 +/- 0.1009 | yes |
| U06 | `B` | `ep_slip` | 1.7018 +/- 0.1127 | yes |
| U06 | `B` | `env` | -0.0104 +/- 0.0046 | yes |
| U07 | `Mamp` | `memory` **(primary)** | 0.5221 +/- 0.0248 | yes |
| U07 | `Mamp` | `gal_aniso` | -0.0035 +/- 0.0010 | yes |
| U07 | `Mamp` | `ep_slip` | 0.0249 +/- 0.0032 | yes |
| U07 | `Mamp` | `env` | 0.0155 +/- 0.0025 | yes |
| U08 | `zeta` | `ep_slip` **(primary)** | 0.4980 +/- 0.0085 | yes |
| U08 | `zeta` | `memory` | 0.3040 +/- 0.0476 | yes |
| U09 | `eps` | `path` **(primary)** | -2.8879 +/- 0.0275 | yes |

**Translated into observables**, because a null from a detector with no power below the predicted amplitude says nothing. median over 10 library clusters of the maximum over 0.2-2.0 R500; g_matter moves stars and gas, g_light deflects photons, the quadrupole is the l=2 fraction of the lensing potential, and the redshift excess is the path term at void fraction 0.6. A null from a detector with no power below the predicted amplitude says nothing.:

| universe | knob | value | max d(g_matter) | max d(g_light) | max l=2 fraction | max d(1+z) |
| --- | --- | --- | --- | --- | --- | --- |
| U04 environment scalar | `kappa` | 0.6 (fiducial) | 0.1654 | 0.1654 | n/a | n/a |
| U04 environment scalar | `kappa` | 0.06307 (threshold) | 0.0189 | 0.0189 | n/a | n/a |
| U05 tensor vacuum, external axis | `A` | 0.5 (fiducial) | 0.0000 | 0.0000 | 0.08273 | n/a |
| U05 tensor vacuum, external axis | `A` | 0.025 (threshold) | 0.0000 | 0.0000 | 0.00414 | n/a |
| U06 reciprocal well network | `B` | 0.06 (fiducial) | 0.2254 | 0.2254 | n/a | n/a |
| U06 reciprocal well network | `B` | 0.006307 (threshold) | 0.0237 | 0.0237 | n/a | n/a |
| U07 gravitational memory | `Mamp` | 0.2 (fiducial) | 0.0567 | 0.0567 | n/a | n/a |
| U07 gravitational memory | `Mamp` | 0.025 (threshold) | 0.0071 | 0.0071 | n/a | n/a |
| U08 photons/matter couple differently | `zeta` | 0.1 (fiducial) | 0.0000 | 0.1816 | n/a | n/a |
| U08 photons/matter couple differently | `zeta` | 0.05 (threshold) | 0.0000 | 0.0870 | n/a | n/a |
| U09 geometric path redshift | `eps` | 0.03 (fiducial) | 0.0000 | 0.0000 | n/a | 0.01389 |
| U09 geometric path redshift | `eps` | 0.0075 (threshold) | 0.0000 | 0.0000 | n/a | 0.00347 |

This is the row that makes a null meaningful. The tensor universe puts **nothing** in the monopole -- its entire signal is the l=2 fraction of the lensing potential -- and at its detection threshold that fraction is a few parts in a thousand. The slip universe changes only the light potential. The path universe does not touch gravity at all.

## 6. The seven Stage 5 identifiability questions

### Q1 -- can the system recover an injected scalar law?

**Yes.** Recovering a0 from the galaxy channel alone -- rotation curves rebuilt from the velocity fields, baryons from the observed photometry -- gives d(log a0_hat)/d(log a0_true) = **0.925 +/- 0.049** (n = 480), bias -0.004 dex, scatter 0.043 dex.

Run on the dark-matter universe the same estimator returns a0_hat offset by 0.067 dex from the value it was never given, with scatter 0.080 dex. **A CDM universe also yields a well-defined acceleration scale.** That is the radial acceleration relation, and recovering it is not evidence for modified gravity. On the baryons-only universe the estimator returns log a0_hat = -10.60, pinned at the edge of the search grid because there is no scale to find.

### Q2 -- can it distinguish scalar misspecification from genuine anisotropy?

Null family: 7 qualitatively different scalar families including surface-density-gated, potential-depth-gated and an unbounded smooth random response, PLUS the environment-scalar universe. Not an off-grid member of a search bank -- three of the seven families are not functions of g_N/a0 at all.

| detector | critical value | audit FP | power on U5 | rate on U2 CDM | rate on U10 systematics |
| --- | --- | --- | --- | --- | --- |
| galaxy velocity-field m=3 harmonic, projected on the external axis | 0.0388 | 0.069 [0.053, 0.090] (n=720) | **1.000 [0.992, 1.000] (n=480)** | 0.065 [0.046, 0.090] (n=480) | 0.069 [0.049, 0.095] (n=480) |
| cluster shear quadrupole, projected on the external axis | 0.1434 | 0.026 [0.017, 0.041] (n=720) | **0.000 [0.000, 0.008] (n=480)** | 0.000 [0.000, 0.008] (n=480) | 0.000 [0.000, 0.008] (n=480) |

**The answer is yes, but only in one channel.** The galaxy velocity-field m=3 harmonic reaches power 1.000 on the tensor universe at a false-positive rate of 0.069 on the untouched scalar-null audit; the cluster shear quadrupole reaches power 0.000 -- it is blind. A 0.5-amplitude tensor puts only an l=2 fraction of 0.0827 into the lensing potential, which is far below the shape noise of a single cluster's source catalogue. Directional gravity in this suite is a GALAXY measurement, not a cluster-lensing one.

* galaxy m=3 null: 95th percentile sits at 3.0x its own median -- heavy-tailed, so a nominal 0.05 read off a Gaussian would be wrong. The critical values above are measured empirical quantiles.
* cluster quadrupole null: 95th percentile sits at 83.6x its own median -- heavy-tailed, so a nominal 0.05 read off a Gaussian would be wrong. The critical values above are measured empirical quantiles.

### Q3 -- can it recover an external axis?

Each galaxy has its **own** external axis, so there is no global direction to stack; the recoverable statement is per object. `axis_hat = PA + (1/2) arg(c3 + i s3)`.

| universe | median per-galaxy axis error | concentration R | aligned projection |
| --- | --- | --- | --- |
| MOND/AQUAL scalar | 46.1 deg | 0.206 | 0.0025 |
| tensor vacuum, external axis | 11.9 deg | 0.840 | 0.2235 |
| collisionless dark matter | 44.9 deg | 0.200 | 0.0038 |
| systematics only | 44.1 deg | 0.205 | 0.0040 |
| H0_scalar_null | 44.9 deg | 0.196 | 0.0044 |

Misaligned control -- the same statistic with the assumed axis rotated by 45 degrees: MOND/AQUAL scalar 0.0049, tensor vacuum, external axis 0.0047.

the concentration R is invariant under a global rotation of the assumed axis; the PROJECTION is not, and it is the projection that collapses when the axis is misspecified.

Axis recovery as a function of the tensor amplitude:

| A | median axis error | concentration R | aligned projection | 45-degree projection |
| --- | --- | --- | --- | --- |
| 0.0 | 44.6 deg | 0.218 | 0.0045 | 0.0033 |
| 0.1 | 30.5 deg | 0.373 | 0.0453 | 0.0033 |
| 0.25 | 18.2 deg | 0.665 | 0.1118 | 0.0035 |
| 0.5 | 11.8 deg | 0.833 | 0.2243 | 0.0052 |
| 1.0 | 9.5 deg | 0.892 | 0.4846 | 0.0171 |
| 2.0 | 8.9 deg | 0.904 | 0.8188 | 0.0053 |

* responsiveness of the aligned projection to A: 0.4147 +/- 0.0186 -- responsive.
* responsiveness of the concentration R to A: 0.2902 +/- 0.1259 -- responsive.
* responsiveness of the 45-degree projection to A: 0.0026 +/- 0.0033 -- **consistent with zero; this statistic sets no limit on A.**
* responsiveness of the median axis error to A: -13.4744 +/- 6.6007 -- responsive.

The 45-degree control is the point: the aligned projection tracks A, the misaligned projection does not. A misspecified axis is a null detector, and its null result carries no information about the amplitude.

### Q4 -- can it distinguish network dependence from source ellipticity?

Detector: slope of the shear residual on the member-derived well-strength map, MINUS the same slope for an angle-scrambled member catalogue (mass and clustercentric radius preserved exactly).

* critical value 0.0125; audit FP 0.035 [0.025, 0.049] (n=960)
* rate on U10, which carries strong baryonic ellipticity and 3x systematics: 0.000 [0.000, 0.008] (n=480)
* rate on U2, a triaxial collisionless halo at a random orientation: 0.008 [0.003, 0.021] (n=480)
* **power on U6 at the fiducial coupling: 0.000 [0.000, 0.008] (n=480)**
* responsiveness d(detector)/dB = -0.0012 +/- 0.0005

**The network detector is blind.** Over the entire scanned range of B its value moves by 0.00028, which is 0.023 of its own critical value 0.0125. Its power on U6 is 0.000 even at the fiducial coupling, where the same universe is separated from U3 at the capped z of 8.5 by the `clu_wl` and `clu_xray` channels. **The well network is detectable only through its monopole -- the extra potential it adds to the radial profile -- and not through the lumpy, member-locked azimuthal signature it was designed to leave.** That is the substantive physical result of Q4, and it is not a null: the network IS detected, just not as a network. Correspondingly, the near-zero rate on U2 and U10 is not a demonstration of specificity, because a detector with no power cannot demonstrate anything.

### Q5 -- can it detect a path effect after survey systematics?

Detector: slope of the supernova Hubble residual on the path void fraction.

* critical value 0.1783; audit FP 0.037 [0.026, 0.054] (n=720)
* rate on U10 systematics-only: 0.131 [0.104, 0.164] (n=480)
* **power on U9 at the fiducial amplitude: 0.025 [0.014, 0.043] (n=480)**
* responsiveness d(detector)/d(eps) = -2.891 +/- 0.037 -- responsive.

**A single hand-picked statistic is far weaker than the calibrated channel.** At the fiducial eps the raw slope detector has power 0.025, while the same supernova channel, tested as a multivariate discriminant against its own sized null, separates U9 from U3 at z = 8.5. The difference is entirely methodological: the raw slope's null is heavy-tailed at 200 supernovae, so its 95th percentile sits at 0.178 while the mean injected slope is smaller than that. Any programme that reports a null from a single chosen statistic without showing its power curve is reporting the statistic, not the physics.

Note also the rate on the systematics-only universe: 0.131 [0.104, 0.164] (n=480) against a nominal 0.05. Realistic survey systematics alone fake a path effect at more than twice the nominal rate, which is why the path branch needs its own systematics-only null and cannot borrow the gravity branch's.

Time dilation: the slope of log(light-curve duration) on log(1+z_obs) is 0.998 in MOND/AQUAL scalar, 1.000 in geometric path redshift. The geometric path mechanism stretches durations by exactly the factor by which it stretches frequencies, so it is **not** excluded by the supernova time-dilation constraint. A non-time-stretching (tired-light) variant is excluded a priori and was not simulated.

### Q6 -- does it falsely detect new gravity in a standard dark-matter universe?

The critical control. Critical values are set on the first half of the calibration arms (baryons + Newton, MOND/AQUAL scalar, systematics only, H0_scalar_null) and applied to the untouched second half and to U2.

| detector | critical value | FP on calibration audit | **FP on U2 CDM** | FP on U2 with 3x systematics |
| --- | --- | --- | --- | --- |
| `aniso_ext` | 0.0088 | 0.054 [0.042, 0.070] (n=960) | **0.294 [0.255, 0.336] (n=480)** | 0.331 [0.291, 0.375] (n=480) |
| `aniso_ext_minus_bar` | 0.0213 | 0.051 [0.039, 0.067] (n=960) | **0.479 [0.435, 0.524] (n=480)** | 0.527 [0.482, 0.571] (n=480) |
| `gal_aniso` | 0.0393 | 0.055 [0.042, 0.072] (n=960) | **0.065 [0.046, 0.090] (n=480)** | 0.163 [0.132, 0.198] (n=480) |
| `network` | 0.0062 | 0.048 [0.036, 0.063] (n=960) | **0.233 [0.198, 0.273] (n=480)** | 0.265 [0.227, 0.306] (n=480) |
| `memory` | 1.5340 | 0.044 [0.033, 0.059] (n=960) | **0.004 [0.001, 0.015] (n=480)** | 0.008 [0.003, 0.021] (n=480) |
| `ep_slip` | 0.6047 | 0.044 [0.033, 0.059] (n=960) | **0.000 [0.000, 0.008] (n=480)** | 0.000 [0.000, 0.008] (n=480) |
| `path` | 0.1403 | 0.061 [0.048, 0.078] (n=960) | **0.002 [0.000, 0.012] (n=480)** | 0.225 [0.190, 0.264] (n=480) |
| `env` | 0.1491 | 0.054 [0.042, 0.070] (n=960) | **0.065 [0.046, 0.090] (n=480)** | 0.181 [0.149, 0.218] (n=480) |

**Family-wise, any of the 8 detectors firing at its own nominal 0.05:** 0.226 [0.201, 0.254] (n=960) on the calibration audit, **0.648 [0.604, 0.689] (n=480) on the dark-matter universe**, 0.785 [0.747, 0.820] (n=480) on dark matter with 3x systematics.

8 detectors at a nominal 0.05 each; the family-wise rate is the programme-level multiplicity the brief warns about.

### Q7 -- at what amplitude does each effect become observable?

Thresholds are read from the refined scans of section 5 (the same numbers), which sample the informative band rather than only the a-priori grid. A scan z of 8.5 is the permutation-null cap, so sensitivity is fitted on the unsaturated points only.

| universe | knob | fiducial | z=3 at | family-wise threshold | z=5 at | fiducial / threshold | responsive? |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **U04** environment scalar | `kappa` | 0.6 | 0.04961 | **0.06220** | 0.08413 | 9.6x | yes |
| **U05** tensor vacuum, external axis | `A` | 0.5 | 0.01207 | **0.01893** | 0.03786 | 26.4x | yes |
| **U06** reciprocal well network | `B` | 0.06 | 0.00726 | **0.00791** | 0.00954 | 7.6x | yes |
| **U07** gravitational memory | `Mamp` | 0.2 | 0.02863 | **0.03487** | 0.05000 | 5.7x | yes |
| **U08** photons/matter couple differently | `zeta` | 0.1 | 0.03030 | **0.03430** | 0.04842 | 2.9x | yes |
| **U09** geometric path redshift | `eps` | 0.03 | 0.00605 | **0.00821** | 0.01156 | 3.7x | yes |

The last column is the headline: every fiducial amplitude chosen a priori sits 10x, 26x, 3x, 4x, 6x, 8x above the amplitude at which this corpus can already see the effect.

## 7. Admissibility gates on the generative laws themselves

* **Coarse-graining.** Representing every cluster member as one object and then as ten subcomponents changes the well-strength field by at most 0.00144 (threshold 0.02): **PASS**. The network law cannot depend on how the cataloguer deblended the image.
* **Reciprocity.** The pair forces of the network kernel are equal and opposite to 0.000e+00 of the largest force: **PASS**. The kernel is the gradient of a symmetric pair energy, so this checks the implementation, not a hope.
* **Potential gauge.** The operational depth variable under the declared primary boundary rule (3 R500) and an alternative (1.5 R500) differ by a mean offset of 0.290 dex (spread 0.017 dex) with a rank correlation of 0.9959. The zero point is convention; the ordering is not.

## 8. What this does NOT establish

* Every z is the separation achievable with **one corpus of the stated size**. A pair marked indistinguishable is indistinguishable *at that survey size and precision*, not in principle. The section 4 oracles say which of the three axes -- statistical noise, systematics control, or sample size -- actually buys the separation.
* The tensor universe is solved to **first order in A** via the l=2 Green's function for spherical scenes, and by the leading-order local relation g_i = K_ij dPhi/dx_j for disks. At the largest amplitudes in the scan the first-order treatment is marginal and the quoted thresholds there are approximate.
* AQUAL/QUMOND curl corrections for a thin disk are not solved; the algebraic relation is used. That approximation is shared by every universe built on the base, so it cancels in each pairwise comparison against the base, but not for the U1 / U2 / U10 comparisons.
* The tensor response saturates at large radius, so K tends to a constant there. A constant K is a coordinate stretch, exactly degenerate with source ellipticity, inclination and line-of-sight depth. All the tensor information in this suite therefore comes from the **spatial variation** of K inside the observed range. That is the correct physical statement, and it is why the cluster quadrupole channel is so much weaker than the galaxy channel.
* The well-network coherence length is fixed at a declared 150 kpc with no prior width. Because that is far larger than a galaxy, the network term is nearly uniform across a disk and exerts almost no force there: in this suite U6 is a **cluster-only** phenomenon by construction, and nothing here constrains a network law with a galaxy-scale coherence length.
* The analysis converts the stacked tangential shear to a baryonic reference with a singular-isothermal factor (dSigma_bar = 0.5 Sigma_bar). That is a constant applied identically to every universe, so it shifts all `clu_wl` features by the same amount and cannot affect any pairwise comparison, but the absolute value of log(dSigma_obs/dSigma_bar) is not a calibrated mass ratio. The same applies to `ep_ld`: only its CHANGES between universes carry information.
* The memory universe is given an observable age proxy at a declared precision -- 0.25 dex on the galaxy time-since-merger and a monotone but noisy X-ray centroid shift on clusters. Real morphological age proxies are worse than that, so the U7 detection thresholds quoted here are OPTIMISTIC and scale directly with the proxy precision.
* U9 differs from U3 only in the redshift branch: its galaxy and cluster channels are identical by construction. U3-vs-U9 is therefore a pure test of whether the supernova channel alone can see a path effect, which is exactly the question that pair is meant to answer.
* The angular scramble inside the network detector seeds its generator from Python's string `hash()`, which is salted per process. The control is therefore statistically valid and unbiased -- a random angular scramble is a random angular scramble -- but it is not bit-reproducible across runs unless PYTHONHASHSEED is pinned. Every other random draw in the suite is seeded explicitly.
* The equivalence classes are properties of **this** corpus definition. Adding a channel the suite does not emit -- resolved polar-ring kinematics, pulsar timing, a cluster with both a measured external axis and deep IFU coverage of its members -- can only split classes further, never merge them.
* No real observation has been scored. Nothing here is evidence for or against any gravity law. It is a statement about what this corpus could and could not tell apart.

---

Results: `results/E0_sizing.json`, `E1_equivalence_map.json`, `E2_channel_separation.json`, `E3_amplitude_scans.json`, `E4_questions.json`, `E5_gates.json`, `E6_missing_observations.json`, `E7_observable_amplitudes.json`, `E8_fingerprints.json`, `E9_equivalence_at_threshold.json`, `E10_sizing_audit.json`, `channel_map.json`, `run_manifest.json`. Code: `physics.py` (the ten laws), `baryons.py` and `scenes.py` (the resolved scenes), `corpus.py` (the instrument forward model), `analysis.py` (the blind pipeline), `stats.py` (calibrated testing), `generate.py` (parallel draw), `run_stage5.py`, `run_finescan.py`, `run_equiv_amplitude.py`, `run_fillband.py`, `run_amplitudes.py`, `run_sizing_audit.py`, `fingerprint.py` (the experiments), `run_all.sh` (the chain), `provenance.py` (the file-access ledger), `render_report.py` (this document).
