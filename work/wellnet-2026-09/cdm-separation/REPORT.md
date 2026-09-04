# Run BH — can any statistic separate the surviving families from cold dark matter?

Lane `work/wellnet-2026-09/cdm-separation/`.  Registry: `BH-cdm-separation`, VALID, depends on `identifiability_gate` v1 and `holdout_seal` v2.

Run BF left one question open.  Its detectors fire on a dark-matter universe at a family-wise rate of **0.648 [0.604, 0.689]**, and it drew the consequence that the two surviving theory families must be tested against a CDM null rather than against each other.  This lane asks whether any statistic can do that, and at what amplitude and sample size.

**Provenance.** Purely synthetic. The parent process and every worker run under a patched `open`/`io.open`/`numpy.load` that raises on a read outside the lane root, on any KiDS or wide-binary token, and on any confirmation-reserve token (SPT, X-GAP, CLoGS, Gaia, MUSE/Granata). Foreign reads: **0**. Real-observation token matched: **False**. The guard is exercised as a test, not asserted: T8 below opens a KiDS path, a CLoGS path and a path outside the lane and requires all three to raise.

**Tests.** 16/16 pass.  They found a real sign error in this lane's own independent forward model — see §6.1.

## 1  Sizing first, on an untouched null half

Run BF's own audit found its nominal 0.01 realising 0.033, so every verdict below is taken at a MEASURED rate.  Critical values come from a calibration half; every rate is measured on a disjoint audit half with different seeds, 1000 corpora per arm per half.

**null = scalar / Newtonian universes (for a NEW-GRAVITY detector)**

| statistic | null mean | null sd | realised FPR, nominal 0.05 two-sided | realised FPR, nominal 0.05 one-sided | realised FPR, nominal 0.01 two-sided |
|---|---|---|---|---|---|
| `S_ext` | +0.012 | 1.026 | 0.052 [0.045, 0.061] | 0.045 [0.038, 0.053] | 0.009 [0.006, 0.013] |
| `G_ext` | +0.006 | 1.586 | 0.050 [0.043, 0.058] | 0.049 [0.042, 0.057] | 0.011 [0.008, 0.015] |
| `S_45` | -0.027 | 1.023 | 0.046 [0.039, 0.054] | 0.044 [0.037, 0.052] | 0.011 [0.008, 0.015] |
| `S_bar` | +0.004 | 1.009 | 0.051 [0.043, 0.059] | 0.051 [0.043, 0.059] | 0.008 [0.006, 0.012] |
| `S_diff` | +0.002 | 1.017 | 0.050 [0.043, 0.059] | 0.046 [0.039, 0.054] | 0.009 [0.006, 0.013] |
| `S_morph` | +0.027 | 0.531 | 0.046 [0.039, 0.054] | 0.045 [0.038, 0.053] | 0.010 [0.007, 0.015] |
| `S_shape` | +0.131 | 0.495 | 0.050 [0.043, 0.058] | 0.047 [0.040, 0.055] | 0.010 [0.007, 0.014] |
| `S_ext_raw` | +0.000 | 0.002 | 0.043 [0.036, 0.051] | 0.040 [0.034, 0.048] | 0.010 [0.007, 0.015] |
| `S_bar_raw` | +0.000 | 0.002 | 0.048 [0.041, 0.056] | 0.047 [0.040, 0.055] | 0.010 [0.007, 0.014] |
| `S_diff_raw` | +0.000 | 0.003 | 0.045 [0.038, 0.053] | 0.044 [0.037, 0.052] | 0.008 [0.006, 0.012] |
| `S_45_raw` | -0.000 | 0.002 | 0.054 [0.047, 0.063] | 0.046 [0.039, 0.054] | 0.007 [0.005, 0.011] |

**null = Run BF's own calibration family, which includes the systematics-only universe**

| statistic | null mean | null sd | realised FPR, nominal 0.05 two-sided | realised FPR, nominal 0.05 one-sided | realised FPR, nominal 0.01 two-sided |
|---|---|---|---|---|---|
| `S_ext` | -0.817 | 1.880 | 0.038 [0.033, 0.045] | 0.044 [0.039, 0.051] | 0.010 [0.007, 0.013] |
| `G_ext` | +0.032 | 1.618 | 0.046 [0.040, 0.053] | 0.046 [0.040, 0.053] | 0.008 [0.006, 0.012] |
| `S_45` | -0.160 | 1.215 | 0.046 [0.040, 0.053] | 0.043 [0.038, 0.050] | 0.010 [0.007, 0.014] |
| `S_bar` | +1.899 | 3.472 | 0.048 [0.042, 0.055] | 0.048 [0.042, 0.055] | 0.009 [0.006, 0.012] |
| `S_diff` | -1.540 | 2.904 | 0.045 [0.039, 0.052] | 0.043 [0.037, 0.050] | 0.007 [0.005, 0.011] |
| `S_morph` | +0.792 | 1.548 | 0.052 [0.045, 0.059] | 0.052 [0.045, 0.059] | 0.009 [0.006, 0.012] |
| `S_shape` | +1.210 | 1.969 | 0.050 [0.043, 0.057] | 0.050 [0.043, 0.057] | 0.009 [0.006, 0.012] |
| `S_ext_raw` | -0.001 | 0.003 | 0.040 [0.034, 0.047] | 0.041 [0.035, 0.047] | 0.009 [0.006, 0.012] |
| `S_bar_raw` | +0.003 | 0.005 | 0.050 [0.044, 0.057] | 0.050 [0.044, 0.057] | 0.009 [0.006, 0.012] |
| `S_diff_raw` | -0.004 | 0.007 | 0.042 [0.036, 0.048] | 0.041 [0.035, 0.047] | 0.008 [0.006, 0.011] |
| `S_45_raw` | -0.000 | 0.002 | 0.043 [0.037, 0.050] | 0.046 [0.040, 0.053] | 0.009 [0.006, 0.012] |

**null = the surviving modified-gravity universes (for a CDM discriminator)**

| statistic | null mean | null sd | realised FPR, nominal 0.05 two-sided | realised FPR, nominal 0.05 one-sided | realised FPR, nominal 0.01 two-sided |
|---|---|---|---|---|---|
| `S_ext` | +0.263 | 1.217 | 0.055 [0.051, 0.061] | 0.051 [0.047, 0.057] | 0.012 [0.010, 0.015] |
| `G_ext` | +2.589 | 6.772 | 0.049 [0.044, 0.054] | 0.049 [0.044, 0.054] | 0.009 [0.007, 0.012] |
| `S_45` | -0.001 | 1.018 | 0.048 [0.044, 0.053] | 0.046 [0.042, 0.051] | 0.013 [0.010, 0.015] |
| `S_bar` | -0.102 | 1.040 | 0.051 [0.047, 0.056] | 0.049 [0.044, 0.054] | 0.010 [0.008, 0.012] |
| `S_diff` | +0.213 | 1.152 | 0.054 [0.049, 0.059] | 0.054 [0.049, 0.059] | 0.014 [0.012, 0.017] |
| `S_morph` | +0.009 | 0.534 | 0.049 [0.044, 0.053] | 0.049 [0.044, 0.054] | 0.009 [0.007, 0.011] |
| `S_shape` | +0.110 | 0.505 | 0.051 [0.046, 0.056] | 0.051 [0.046, 0.056] | 0.011 [0.009, 0.014] |
| `S_ext_raw` | +0.000 | 0.002 | 0.057 [0.052, 0.062] | 0.053 [0.049, 0.059] | 0.013 [0.011, 0.016] |
| `S_bar_raw` | -0.000 | 0.002 | 0.050 [0.046, 0.055] | 0.049 [0.044, 0.054] | 0.009 [0.007, 0.012] |
| `S_diff_raw` | +0.001 | 0.003 | 0.047 [0.043, 0.052] | 0.049 [0.044, 0.053] | 0.012 [0.010, 0.015] |
| `S_45_raw` | -0.000 | 0.002 | 0.050 [0.045, 0.055] | 0.046 [0.042, 0.051] | 0.010 [0.008, 0.013] |

## 2  Job 1 — the mechanism of the confusion

Measured with an estimator that shares nothing with Run BF's: monopole, m=2 and m=4 are fitted simultaneously per radial bin in BOTH the tangential and the cross ellipticity, with a covariance, so the quadrupole power can be noise-debiased and the phase carries an error.

### 2.1  Amplitude and phase

| universe | median quadrupole amplitude | median per-cluster SNR | concentration about the BARYON axis | about the EXTERNAL axis | median phase error vs baryon axis | vs external axis |
|---|---|---|---|---|---|---|
| U03_mond | 0.0062 | 1.19 | 0.318 | 0.311 | 44.7 deg | 45.4 deg |
| H0_scalar_null | 0.0065 | 1.18 | 0.305 | 0.304 | 45.6 deg | 45.4 deg |
| U10_systematics | 0.0128 | 2.22 | 0.773 | 0.384 | 12.9 deg | 60.3 deg |
| U02_cdm | 0.0184 | 3.40 | 0.673 | 0.362 | 20.1 deg | 56.9 deg |
| U05_thresh | 0.0062 | 1.20 | 0.318 | 0.305 | 44.2 deg | 46.1 deg |
| U05_fid | 0.0068 | 1.31 | 0.311 | 0.430 | 52.2 deg | 29.0 deg |
| U05_A2 | 0.0131 | 2.54 | 0.400 | 0.813 | 62.6 deg | 12.3 deg |
| U06_fid | 0.0063 | 1.23 | 0.302 | 0.307 | 45.8 deg | 44.8 deg |

A 12-cluster corpus of random phases gives a concentration of about 0.32; that is the null level, not zero.

### 2.2  M1 — the radial profile of the quadrupole

Debiased quadrupole power per radial bin (0.20-0.55, 0.55-1.10, 1.10-2.20 R500), and its shape normalised to sum to one.

| universe | mean Q^2 per bin | normalised shape | studentised power per bin |
|---|---|---|---|
| U03_mond | 6.46e-05, 3.83e-05, -1.91e-06 | +0.640, +0.379, -0.019 | -0.10, -0.13, -0.16 |
| H0_scalar_null | 1.15e-04, 9.87e-06, -1.80e-06 | +0.934, +0.080, -0.015 | -0.10, -0.15, -0.16 |
| U10_systematics | 1.08e-02, 4.76e-04, 2.10e-05 | +0.956, +0.042, +0.002 | +1.83, +0.48, +0.01 |
| U02_cdm | 8.63e-02, 1.88e-03, 5.62e-05 | +0.978, +0.021, +0.001 | +3.42, +1.30, +0.23 |
| U05_thresh | 9.37e-05, 8.50e-06, -2.61e-07 | +0.919, +0.083, -0.003 | -0.10, -0.15, -0.15 |
| U05_fid | 8.58e-05, 8.92e-06, 1.11e-05 | +0.811, +0.084, +0.105 | -0.09, -0.13, -0.08 |
| U05_A2 | 1.79e-04, 2.30e-04, 1.33e-04 | +0.331, +0.424, +0.245 | -0.05, +0.27, +0.63 |
| U06_fid | 7.33e-05, 8.59e-06, 2.24e-06 | +0.871, +0.102, +0.027 | -0.10, -0.14, -0.14 |

**The two mechanisms have opposite radial gradients.** The collisionless halo puts 98% of its quadrupole power inside 0.55 R500 and its studentised power falls outward; the tensor's rises outward.

### 2.3  M2 — dependence on baryonic morphology

Slope of the studentised quadrupole power on the OBSERVED baryon ellipticity, pooled over every cluster of every corpus in the arm.  Slopes, not correlations.

| universe | slope | s.e. | t | n clusters |
|---|---|---|---|---|
| U03_mond | +0.03 | 0.07 | +0.51 | 4800 |
| H0_scalar_null | -0.01 | 0.07 | -0.16 | 4800 |
| U10_systematics | +7.57 | 0.21 | +36.13 | 4800 |
| U02_cdm | +8.84 | 0.35 | +25.18 | 4800 |
| U05_thresh | -0.04 | 0.07 | -0.62 | 4800 |
| U05_fid | -0.03 | 0.06 | -0.53 | 4800 |
| U05_A2 | +0.25 | 0.07 | +3.56 | 4800 |
| U06_fid | +0.02 | 0.06 | +0.37 | 4800 |

A collisionless halo's quadrupole grows with the visible ellipticity because its own shape is set by the same tidal history; a tensor response is sourced by the field, not by the shape, and its slope is consistent with zero at the fiducial amplitude — **no upper limit is set on a tensor's morphology dependence by this statistic.**

### 2.4  M4 — the matter sector

An m=2 modulation of the member velocity dispersion, projected on each axis.  This is the check of whether the quadrupole is present in the matter sector as well as in the light sector.

| universe | projection on the baryon axis | on the external axis | per-cluster error | clusters |
|---|---|---|---|---|
| U02_cdm | +0.0036 +- 0.0051 | -0.0080 +- 0.0051 | 0.191 | 1440 |
| U05_fid | +0.0008 +- 0.0050 | +0.0052 +- 0.0050 | 0.191 | 1440 |
| U03_mond | +0.0076 +- 0.0049 | -0.0049 +- 0.0051 | 0.192 | 1440 |

**Every arm is consistent with zero.**  That is a property of the generator, not of the physics: Run BF's `emit_cluster` applies both the halo ellipticity and the tensor quadrupole to the LENSING map only, and solves the member Jeans equation in the radial field alone.  The bound above says the matter-sector quadrupole is below about 0.01 in fractional amplitude in both universes, so **the joint matter/light behaviour cannot separate a triaxial halo from a tensor response in this corpus, and this lane sets no limit on it.**  In a generator where both mechanisms wrote into the dynamics, they would still write into it the same way: both are metric quadrupoles.  The matter/light axis separates either of them from a SLIP, not from each other.

### 2.5  M5 — coarse-graining and commutation

A triaxial collisionless halo is a SOURCE with a shape; an external-axis tensor is a LAW.  `AzimuthalAverage` keeps every source's radius and randomises its angles, so it destroys a source's own axis and leaves an imposed one untouched.  Shell P2 quadrupole of the radial field at 500, 1000, 1500 kpc:

| law | axis | before | after azimuthal average | surviving fraction | after spherical average | surviving fraction |
|---|---|---|---|---|---|---|
| `newton_on_triaxial_source` | about_external_axis | +0.0228, +0.0049, -0.0099 | -0.0052, -0.0004, -0.0028 | 0.227 | +0.0000, +0.0000, +0.0000 | 0.004 |
| `newton_on_triaxial_source` | about_source_axis | -0.0505, -0.0132, +0.0081 | -0.0020, -0.0001, -0.0017 | 0.040 | -0.0001, -0.0001, -0.0001 | 0.004 |
| `external_axis_tensor_A0.5` | about_external_axis | +0.0888, +0.0697, +0.0537 | +0.0595, +0.0660, +0.0632 | 0.946 | +0.0667, +0.0667, +0.0667 | 0.957 |
| `external_axis_tensor_A0.5` | about_source_axis | -0.0661, -0.0317, -0.0113 | -0.0344, -0.0335, -0.0349 | 1.058 | -0.0334, -0.0334, -0.0334 | 1.052 |
| `source_aligned_tensor_A0.5` | about_external_axis | +0.0034, -0.0171, -0.0330 | +0.0264, +0.0323, +0.0296 | 1.885 | -0.0213, -0.0213, -0.0213 | 1.241 |
| `source_aligned_tensor_A0.5` | about_source_axis | -0.0105, +0.0316, +0.0553 | -0.0288, -0.0276, -0.0293 | 0.873 | +0.0183, +0.0183, +0.0183 | 0.578 |

**The separation is exact in principle.**  The source quadrupole loses 96% of itself under azimuthal averaging and 99.6% under spherical averaging; the law quadrupole keeps 0.95 and 0.96 of itself.  The spherically averaged value +0.0667 is the analytic A*(2/3)*<P2^2> = 0.0667 for A = 0.5, which is a check on the module rather than a result.  But no observer can azimuthally average a real cluster: the operational proxy for this operation is exactly the PHASE of the quadrupole relative to the source's own axis, which is §2.6.

### 2.6  Where Run BF's 0.648 comes from — a factorial decomposition

Four detectors that read the SAME quadrupole, differing only in whether the projections are studentised and whether the test keeps the sign.  Rates on the dark-matter universe, critical values calibrated on Run BF's own null family:

| detector | form | rate on U02 (CDM), two-sided | one-sided upper | one-sided lower | mean on U02 | sd on U02 | sd on the scalar null |
|---|---|---|---|---|---|---|---|
| `S_ext_raw` | unstudentised, external axis (Run BF's `aniso_ext`) | 0.315 [0.287, 0.344] | 0.016 [0.010, 0.026] | 0.315 [0.287, 0.344] | -0.005 | 0.004 | 0.002 |
| `S_ext` | studentised, external axis | 0.349 [0.320, 0.379] | 0.026 [0.018, 0.038] | 0.349 [0.320, 0.379] | -3.792 | 2.818 | 1.027 |
| `S_diff_raw` | unstudentised difference (`aniso_ext_minus_bar`) | 0.456 [0.425, 0.487] | 0.000 [0.000, 0.004] | 0.456 [0.425, 0.487] | -0.017 | 0.006 | 0.003 |
| `S_diff` | studentised difference | 0.488 [0.457, 0.519] | 0.000 [0.000, 0.004] | 0.488 [0.457, 0.519] | -7.347 | 2.712 | 1.011 |
| `S_45_raw` | unstudentised, axis rotated 45 deg | 0.339 [0.310, 0.369] | 0.143 [0.123, 0.166] | 0.278 [0.251, 0.307] | -0.001 | 0.004 | 0.002 |
| `S_45` | studentised, axis rotated 45 deg | 0.435 [0.405, 0.466] | 0.222 [0.197, 0.249] | 0.298 [0.270, 0.327] | -0.541 | 3.001 | 1.019 |

**The mechanism is a variance inflation that a two-sided test converts into a false positive, plus a sign that a two-sided test throws away.**

1. A triaxial halo puts a LARGE quadrupole into the shear (3.0x the scalar null's, at SNR 3.4 per cluster) whose phase is unrelated to the external axis.  Projected on that axis it has mean ~0 but a standard deviation inflated from 1.02 to 3.00 — a factor 2.9.  A detector calibrated on scalar universes has no such width, so |S| exceeds its critical value often.  The misspecified-axis control `S_45` fires on CDM at 0.435 [0.405, 0.466] while its responsiveness to the tensor amplitude is -0.011 +- 0.047 (t = -0.24) — a detector that cannot see the signal at all still fires on dark matter a third of the time.  That is the variance term, isolated.

2. The external-minus-baryon contrast is not symmetric: a halo is baryon-aligned, so the contrast has a large NEGATIVE mean (-7.35), while a tensor gives a positive one.  Run BF's `aniso_ext_minus_bar` tests |S|, so the two land on the same side of the threshold.  Splitting the tail recovers everything: the same statistic fires on CDM at 0.488 [0.457, 0.519] two-sided and 0.000 [0.000, 0.004] in the upper tail alone.

### 2.7  An accidental axis alignment inside Run BF's shared library

Run BF draws every corpus from ONE library of 18 clusters, on purpose, so that a separation cannot come from the scene prior.  The consequence for a DIRECTIONAL statistic was not checked: those 18 (baryon axis, external axis) pairs are fixed, so whatever correlation they happen to have is present in every corpus and never averages out.

| quantity | value |
|---|---|
| mean cos 2(pa_bar - axis_ext) over the library | **-0.3685** |
| expected s.d. of that mean if the axes were independent | 0.167 |
| significance of the accidental alignment | -2.21 sigma |

A baryon-aligned quadrupole of studentised size S therefore projects onto the external axis with mean S x -0.369 in every corpus.  That predicts a mean `S_ext` on the dark-matter arm of about -3.36; the measured value is -3.79.  **Part of the external-axis detector's false-positive rate on dark matter is an accident of an 18-object library, not physics.**  The independent forward model, which redraws both axes for every cluster, gives a halo mean of +0.02 +- 0.07 -- consistent with zero, with only the VARIANCE inflated (sd 1.55 against 1.05 on the empty arm).  Both are reported; the verdicts use the stricter one.

## 3  Job 2 — the candidate statistics

Six candidates, each one number per corpus, each SIGNED, each studentised by its own propagated error.  None reuses a Run BF detector.

* `S_ext` — new gravity: a quadrupole locked to the external axis
* `G_ext` — new gravity: galaxy m=3 locked to the external axis
* `S_45` — misspecified-axis control (must be a null detector)
* `S_bar` — dark matter: a quadrupole locked to the baryon major axis
* `S_diff` — signed contrast, external minus baryon axis
* `S_morph` — dark matter: quadrupole power rises with baryon ellipticity
* `S_shape` — radial shape contrast of the quadrupole power

### 3.1  The number that matters: the rate on the dark-matter universe

Critical values from the scalar/Newtonian calibration half; rates on the untouched audit half.

| statistic | test | U03_mond | U10_systematics | U02_cdm | U02_cdm_3xsys | U05_thresh | U05_fid | U05_A1 | U06_fid | U09_fid |
|---|---|---|---|---|---|---|---|---|---|---|
| `S_ext` | two-sided | 0.050 | 0.781 | 0.758 | 0.731 | 0.045 | 0.490 | 0.966 | 0.053 | 0.041 |
| `S_ext` | one-sided upper | 0.044 | 0.001 | 0.021 | 0.025 | 0.055 | 0.587 | 0.984 | 0.044 | 0.046 |
| `G_ext` | two-sided | 0.061 | 0.077 | 0.078 | 0.111 | 0.079 | 1.000 | 1.000 | 0.059 | 0.052 |
| `G_ext` | one-sided upper | 0.060 | 0.071 | 0.069 | 0.087 | 0.117 | 1.000 | 1.000 | 0.067 | 0.065 |
| `S_45` | two-sided | 0.042 | 0.239 | 0.524 | 0.495 | 0.049 | 0.057 | 0.078 | 0.042 | 0.050 |
| `S_45` | one-sided upper | 0.045 | 0.076 | 0.230 | 0.212 | 0.049 | 0.051 | 0.069 | 0.048 | 0.046 |

**Family-wise, over the two new-gravity detectors** (`S_ext`, `G_ext`), calibrated on Run BF's own null family — the number directly comparable with Run BF's 0.648:

| universe | family-wise two-sided | family-wise one-sided upper |
|---|---|---|
| U03_mond | 0.053 [0.041, 0.069] | 0.108 [0.090, 0.129] |
| U10_systematics | 0.204 [0.180, 0.230] | 0.063 [0.050, 0.080] |
| U02_cdm | 0.384 [0.354, 0.415] | 0.087 [0.071, 0.106] |
| U02_cdm_3xsys | 0.380 [0.350, 0.410] | 0.109 [0.091, 0.130] |
| U05_thresh | 0.063 [0.050, 0.080] | 0.176 [0.154, 0.201] |
| U05_fid | 1.000 [0.996, 1.000] | 1.000 [0.996, 1.000] |
| U05_A1 | 1.000 [0.996, 1.000] | 1.000 [0.996, 1.000] |
| U06_fid | 0.049 [0.037, 0.064] | 0.113 [0.095, 0.134] |
| U09_fid | 0.041 [0.030, 0.055] | 0.109 [0.091, 0.130] |
| U01_newton | 0.031 [0.022, 0.044] | 0.100 [0.083, 0.120] |
| H0_scalar_null | 0.041 [0.030, 0.055] | 0.085 [0.069, 0.104] |

**The joint procedure** — declare new gravity only if an external-axis statistic fires AND the baryon-axis statistic does not (the CDM veto):

| universe | fires | veto rate | fires with no veto |
|---|---|---|---|
| U03_mond | 0.053 [0.041, 0.069] | 0.059 [0.046, 0.075] | **0.049 [0.037, 0.064]** |
| U10_systematics | 0.204 [0.180, 0.230] | 1.000 [0.996, 1.000] | **0.000 [0.000, 0.004]** |
| U02_cdm | 0.384 [0.354, 0.415] | 0.998 [0.993, 0.999] | **0.002 [0.001, 0.007]** |
| U02_cdm_3xsys | 0.380 [0.350, 0.410] | 0.998 [0.993, 0.999] | **0.000 [0.000, 0.004]** |
| U05_thresh | 0.063 [0.050, 0.080] | 0.053 [0.041, 0.069] | **0.059 [0.046, 0.075]** |
| U05_fid | 1.000 [0.996, 1.000] | 0.011 [0.006, 0.020] | **0.989 [0.980, 0.994]** |
| U05_A1 | 1.000 [0.996, 1.000] | 0.002 [0.001, 0.007] | **0.998 [0.993, 0.999]** |
| U06_fid | 0.049 [0.037, 0.064] | 0.053 [0.041, 0.069] | **0.045 [0.034, 0.060]** |
| U09_fid | 0.041 [0.030, 0.055] | 0.048 [0.036, 0.063] | **0.041 [0.030, 0.055]** |

### 3.2  The CDM discriminators, sized against the modified-gravity null

| statistic | test | U03_mond | U10_systematics | U02_cdm | U02_cdm_3xsys | U05_thresh | U05_fid | U05_A1 | U06_fid | U09_fid |
|---|---|---|---|---|---|---|---|---|---|---|
| `S_bar` | one-sided upper | 0.059 | 1.000 | 0.998 | 0.998 | 0.053 | 0.011 | 0.002 | 0.053 | 0.048 |
| `S_bar` | two-sided | 0.045 | 1.000 | 0.997 | 0.997 | 0.044 | 0.128 | 0.357 | 0.039 | 0.032 |
| `S_diff` | one-sided upper | 0.017 | 0.000 | 0.000 | 0.000 | 0.019 | 0.307 | 0.813 | 0.015 | 0.018 |
| `S_diff` | two-sided | 0.027 | 0.995 | 0.965 | 0.966 | 0.030 | 0.253 | 0.765 | 0.019 | 0.022 |
| `S_morph` | one-sided upper | 0.046 | 0.964 | 0.879 | 0.897 | 0.039 | 0.046 | 0.051 | 0.046 | 0.042 |
| `S_morph` | two-sided | 0.040 | 0.947 | 0.893 | 0.900 | 0.047 | 0.048 | 0.054 | 0.046 | 0.048 |
| `S_shape` | one-sided upper | 0.055 | 1.000 | 1.000 | 1.000 | 0.045 | 0.031 | 0.003 | 0.056 | 0.054 |
| `S_shape` | two-sided | 0.054 | 1.000 | 1.000 | 1.000 | 0.043 | 0.054 | 0.165 | 0.053 | 0.048 |

### 3.3  Sample size for 3 sigma, and responsiveness

Every statistic is a studentised sum over clusters divided by sqrt(N), so it grows as sqrt(N).  Measured at N = 3, 6, 12, 18 clusters and extrapolated.

| statistic | per-cluster coefficient k (S = k sqrt(N)) | null sd | clusters for 3 sigma on CDM | d(S)/d(A_tensor) | d(S)/d(B_wellnet) | d(S)/d(eps_path) |
|---|---|---|---|---|---|---|
| `S_bar` | +2.615 | 1.040 | 1.4 | -1.60 +- 0.07 | -0.57 +- 1.11 (consistent with zero) | +0.34 +- 0.85 (consistent with zero) |
| `S_diff` | -2.113 | 1.152 | 2.7 | +3.20 +- 0.08 | +0.48 +- 0.84 (consistent with zero) | -0.92 +- 1.09 (consistent with zero) |
| `S_morph` | +0.976 | 0.534 | 2.7 | +0.04 +- 0.05 (consistent with zero) | +0.40 +- 0.14 | -0.56 +- 0.30 (consistent with zero) |
| `S_shape` | +2.221 | 0.505 | 0.5 | -0.57 +- 0.07 | +0.59 +- 0.33 (consistent with zero) | +0.44 +- 0.24 (consistent with zero) |
| `S_ext` | -1.116 | 1.026 | 7.6 | +3.94 +- 0.07 | +0.29 +- 0.41 (consistent with zero) | -1.21 +- 1.10 (consistent with zero) |
| `S_45` | -0.124 | 1.023 | 612.3 | -0.01 +- 0.05 (consistent with zero) | -0.05 +- 0.50 (consistent with zero) | -0.10 +- 0.56 (consistent with zero) |

Galaxy channel: `d(G_ext)/dA = +41.04 +- 0.29`, misspecified-axis control `d(G_45)/dA = +0.904 +- 0.129`.

Tensor amplitude scan (Run BF's generator), mean of each statistic:

| A | `S_ext` | `G_ext` | `S_bar` | `S_diff` | `S_45` |
|---|---|---|---|---|---|
| 0 | +0.06 | -0.05 | +0.01 | -0.01 | -0.05 |
| 0.0125 | -0.01 | +0.56 | -0.02 | -0.01 | -0.07 |
| 0.025 | +0.16 | +0.78 | -0.09 | +0.13 | -0.06 |
| 0.05 | +0.08 | +1.89 | -0.04 | +0.06 | -0.03 |
| 0.1 | +0.45 | +4.02 | -0.18 | +0.37 | +0.05 |
| 0.25 | +1.01 | +9.76 | -0.50 | +0.87 | -0.03 |
| 0.5 | +1.99 | +20.02 | -0.91 | +1.67 | +0.00 |
| 1 | +3.93 | +41.40 | -1.57 | +3.15 | -0.06 |

Reciprocal / path family, B scan — mean of each statistic:

| B | `S_ext` | `G_ext` | `S_bar` | `S_diff` |
|---|---|---|---|---|
| 0 | -0.02 | +0.09 | +0.03 | -0.04 |
| 0.0072 | -0.05 | +0.05 | +0.01 | -0.04 |
| 0.015 | +0.03 | -0.06 | -0.24 | +0.16 |
| 0.03 | -0.03 | -0.04 | +0.02 | -0.02 |
| 0.06 | -0.05 | +0.06 | -0.04 | +0.00 |
| 0.12 | +0.03 | -0.17 | -0.11 | +0.07 |

Reciprocal / path family, eps scan — mean of each statistic:

| eps | `S_ext` | `G_ext` | `S_bar` | `S_diff` |
|---|---|---|---|---|
| 0 | +0.12 | +0.24 | -0.07 | +0.13 |
| 0.008 | -0.07 | +0.11 | +0.07 | -0.06 |
| 0.015 | +0.23 | +0.02 | -0.13 | +0.22 |
| 0.03 | +0.03 | +0.10 | -0.01 | +0.02 |
| 0.06 | -0.03 | -0.01 | +0.07 | -0.02 |
| 0.12 | -0.04 | +0.06 | -0.02 | -0.00 |

## 4  Stage 4 certificates

3 issued, 12 refused.  Seven checks, all required; typed identifiers so no logic depends on a readable name.  Each candidate is certified at more than one amplitude on purpose: a statistic certified at one and refused at another has named the amplitude at which the answer changes.

* **C2** — measured control lever: the other mechanism's arm
* **C4_noise_sd_tensor** — scalar-null sd on the untouched audit half
* **C4_noise_sd_cdm** — the empty-universe arm of the independent model
* **C7** — response pattern across the statistic set, not an amplitude sequence (a correlation saturates)
* **C2_cluster_caveat** — the control effect for S_ext is the dark-matter arm's mean on Run BF's generator, -3.79.  That mean is LIBRARY SPECIFIC: the shared 18-cluster scene library happens to have mean cos 2(pa_bar - axis_ext) = -0.369, so a baryon-aligned quadrupole projects onto the external axis with a fixed non-zero coefficient in every corpus.  In the independent forward model, which redraws both axes per cluster, the halo's effect on S_ext is consistent with zero and only its VARIANCE is inflated.  The stricter of the two is used here.

| candidate | statistic | amplitude | issued | failed checks |
|---|---|---|---|---|
| `CAND.TENSOR.CLUSTER_QUAD.AT_BF_THRESHOLD` | `S_ext` | 0.0200293 | **REFUSED** | C2_not_a_restatement, C4_powered |
| `CAND.TENSOR.CLUSTER_QUAD.AT_A0.1` | `S_ext` | 0.1 | **REFUSED** | C2_not_a_restatement, C4_powered |
| `CAND.TENSOR.CLUSTER_QUAD.AT_FIDUCIAL` | `S_ext` | 0.5 | **REFUSED** | C2_not_a_restatement, C4_powered |
| `CAND.TENSOR.CLUSTER_QUAD.AT_A1.0` | `S_ext` | 1 | ISSUED | - |
| `CAND.TENSOR.GALAXY_M3.AT_BF_THRESHOLD` | `G_ext` | 0.0200293 | **REFUSED** | C4_powered |
| `CAND.TENSOR.GALAXY_M3.AT_A0.1` | `G_ext` | 0.1 | **REFUSED** | C4_powered |
| `CAND.TENSOR.GALAXY_M3.AT_FIDUCIAL` | `G_ext` | 0.5 | ISSUED | - |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_E0.30` | `S_bar` | 0.3 | **REFUSED** | C4_powered |
| `CAND.CDM.BARYON_AXIS_QUAD.AT_E0.45` | `S_bar` | 0.45 | ISSUED | - |
| `CAND.CDM.SIGNED_CONTRAST.AT_E0.30` | `S_diff` | 0.3 | **REFUSED** | C4_powered |
| `CAND.CDM.SIGNED_CONTRAST.AT_E0.45` | `S_diff` | 0.45 | **REFUSED** | C4_powered |
| `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.30` | `S_morph` | 0.3 | **REFUSED** | C4_powered, C6_out_of_grammar |
| `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.45` | `S_morph` | 0.45 | **REFUSED** | C4_powered, C6_out_of_grammar |
| `CAND.CDM.RADIAL_SHAPE.AT_E0.30` | `S_shape` | 0.3 | **REFUSED** | C4_powered |
| `CAND.CDM.RADIAL_SHAPE.AT_E0.45` | `S_shape` | 0.45 | **REFUSED** | C4_powered |

* `CAND.TENSOR.CLUSTER_QUAD.AT_BF_THRESHOLD` failed `C2_not_a_restatement`: a dark-matter universe (its OWN halo ellipticity) reproduces 48.206x the effect that the tensor at this amplitude produces (3.800 against 0.079)
* `CAND.TENSOR.CLUSTER_QUAD.AT_BF_THRESHOLD` failed `C4_powered`: the theory predicts 0.0200293; through a pipeline of responsiveness 3.936 that is 0.08 sigma
* `CAND.TENSOR.CLUSTER_QUAD.AT_A0.1` failed `C2_not_a_restatement`: a dark-matter universe (its OWN halo ellipticity) reproduces 9.655x the effect that the tensor at this amplitude produces (3.800 against 0.394)
* `CAND.TENSOR.CLUSTER_QUAD.AT_A0.1` failed `C4_powered`: the theory predicts 0.1; through a pipeline of responsiveness 3.936 that is 0.38 sigma
* `CAND.TENSOR.CLUSTER_QUAD.AT_FIDUCIAL` failed `C2_not_a_restatement`: a dark-matter universe (its OWN halo ellipticity) reproduces 1.931x the effect that the tensor at this amplitude produces (3.800 against 1.968)
* `CAND.TENSOR.CLUSTER_QUAD.AT_FIDUCIAL` failed `C4_powered`: the theory predicts 0.5; through a pipeline of responsiveness 3.936 that is 1.92 sigma
* `CAND.TENSOR.GALAXY_M3.AT_BF_THRESHOLD` failed `C4_powered`: the theory predicts 0.0200293; through a pipeline of responsiveness 41.319 that is 0.52 sigma
* `CAND.TENSOR.GALAXY_M3.AT_A0.1` failed `C4_powered`: the theory predicts 0.1; through a pipeline of responsiveness 41.319 that is 2.61 sigma
* `CAND.CDM.BARYON_AXIS_QUAD.AT_E0.30` failed `C4_powered`: the theory predicts 0.3; through a pipeline of responsiveness 9.084 that is 2.73 sigma
* `CAND.CDM.SIGNED_CONTRAST.AT_E0.30` failed `C4_powered`: the theory predicts 0.3; through a pipeline of responsiveness 5.837 that is 1.67 sigma
* `CAND.CDM.SIGNED_CONTRAST.AT_E0.45` failed `C4_powered`: the theory predicts 0.45; through a pipeline of responsiveness 5.837 that is 2.51 sigma
* `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.30` failed `C4_powered`: the theory predicts 0.3; through a pipeline of responsiveness 0.011 that is 0.01 sigma
* `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.30` failed `C6_out_of_grammar`: recovers 11% of an out-of-grammar injection
* `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.45` failed `C4_powered`: the theory predicts 0.45; through a pipeline of responsiveness 0.011 that is 0.01 sigma
* `CAND.CDM.MORPHOLOGY_SLOPE.AT_E0.45` failed `C6_out_of_grammar`: recovers 11% of an out-of-grammar injection
* `CAND.CDM.RADIAL_SHAPE.AT_E0.30` failed `C4_powered`: the theory predicts 0.3; through a pipeline of responsiveness 0.032 that is 0.02 sigma
* `CAND.CDM.RADIAL_SHAPE.AT_E0.45` failed `C4_powered`: the theory predicts 0.45; through a pipeline of responsiveness 0.032 that is 0.03 sigma

## 5  The inverse-crime control, and the axis the answer turns on

`forward.py` is a second, independently written forward model: analytic NFW convergence and mean convergence in closed form, an m=2 convergence profile propagated through the exact 2-D l=2 Green's function, analytic shear components, a different source sampling law and a nuisance model written here rather than imported.  It shares no basis, discretisation, solver or nuisance code with Run BF's 64x64x31 Cartesian projection.

### 5.1  The same statistics in the independent model

| arm | `S_bar` | `S_ext` | `S_diff` | `S_morph` | `S_shape` | `S_45` |
|---|---|---|---|---|---|---|
| none | -0.01 +- 1.00 | -0.07 +- 1.05 | -0.06 +- 1.05 | -0.01 +- 0.55 | +0.04 +- 0.50 | +0.01 +- 1.00 |
| halo | +4.14 +- 1.27 | +0.02 +- 1.55 | -2.64 +- 1.43 | -0.05 +- 0.63 | +0.03 +- 0.59 | +0.03 +- 1.58 |
| halo_physical | +2.39 +- 1.16 | +0.03 +- 1.31 | -1.48 +- 1.24 | +0.38 +- 0.59 | +0.05 +- 0.55 | +0.01 +- 1.28 |
| tensor | +0.04 +- 1.32 | +3.46 +- 1.09 | +2.17 +- 1.11 | -0.02 +- 0.55 | -0.14 +- 0.54 | +0.10 +- 1.07 |
| tensor_strong | +0.02 +- 1.94 | +6.92 +- 1.25 | +4.40 +- 1.38 | +0.00 +- 0.62 | -0.53 +- 0.62 | +0.04 +- 1.26 |
| both | +4.10 +- 1.45 | +3.47 +- 1.58 | -0.40 +- 1.34 | -0.02 +- 0.70 | -0.03 +- 0.60 | +0.05 +- 1.68 |

The `both` arm — a halo AND a tensor in the same universe — gives `S_bar` = +4.10 and `S_ext` = +3.47 while their difference `S_diff` = -0.40 cancels.  **Use the two projections separately; the difference statistic is blind to a universe that contains both.**

### 5.2  THE ALIGNMENT SCAN — where the answer changes

Run BF's generator gives a collisionless halo a projected major axis of `pa_baryon + N(0, 22 deg)` and NO knowledge of the surrounding structure.  Both halves of that are modelling choices.  Cluster haloes in N-body simulations align with the filament they sit in; this scan varies the halo/baryon misalignment scatter `mis`, the fraction `f_lss` of the halo's alignment carried by the EXTERNAL axis, and the halo quadrupole amplitude `e`.

| mis (deg) | f_lss | e | power of the CDM discriminator `S_bar` | false-positive rate of `S_ext`, two-sided | one-sided upper |
|---|---|---|---|---|---|
| 0 | 0 | 0.45 | 0.996 [0.986, 0.999] | 0.102 [0.078, 0.132] | 0.114 [0.089, 0.145] |
| 15 | 0 | 0.45 | 0.982 [0.966, 0.991] | 0.138 [0.111, 0.171] | 0.102 [0.078, 0.132] |
| 30 | 0 | 0.45 | 0.698 [0.656, 0.737] | 0.110 [0.085, 0.140] | 0.078 [0.058, 0.105] |
| 45 | 0 | 0.45 | 0.316 [0.277, 0.358] | 0.110 [0.085, 0.140] | 0.120 [0.094, 0.151] |
| 60 | 0 | 0.45 | 0.132 [0.105, 0.164] | 0.100 [0.077, 0.129] | 0.082 [0.061, 0.109] |
| 90 | 0 | 0.45 | 0.066 [0.047, 0.091] | 0.102 [0.078, 0.132] | 0.076 [0.056, 0.103] |
| 22 | 0.25 | 0.45 | 0.916 [0.888, 0.937] | 0.126 [0.100, 0.158] | 0.178 [0.147, 0.214] |
| 22 | 0.5 | 0.45 | 0.560 [0.516, 0.603] | 0.482 [0.439, 0.526] | 0.608 [0.565, 0.650] |
| 22 | 0.75 | 0.45 | 0.168 [0.138, 0.203] | 0.830 [0.795, 0.860] | 0.918 [0.891, 0.939] |
| 22 | 1 | 0.45 | 0.068 [0.049, 0.094] | 0.850 [0.816, 0.879] | 0.920 [0.893, 0.941] |
| 22 | 0 | 0.1 | 0.064 [0.046, 0.089] | 0.014 [0.007, 0.029] | 0.030 [0.018, 0.049] |
| 22 | 0 | 0.2 | 0.284 [0.246, 0.325] | 0.034 [0.021, 0.054] | 0.032 [0.020, 0.051] |
| 22 | 0 | 0.3 | 0.658 [0.615, 0.698] | 0.038 [0.024, 0.059] | 0.042 [0.028, 0.063] |
| 22 | 0 | 0.6 | 0.992 [0.980, 0.997] | 0.164 [0.134, 0.199] | 0.132 [0.105, 0.164] |
| 22 | 0 | 0.8 | 1.000 [0.992, 1.000] | 0.286 [0.248, 0.327] | 0.188 [0.156, 0.225] |
| 0 | 0.25 | 0.45 | 0.998 [0.989, 1.000] | 0.110 [0.085, 0.140] | 0.178 [0.147, 0.214] |
| 0 | 0.5 | 0.45 | 0.788 [0.750, 0.822] | 0.722 [0.681, 0.759] | 0.812 [0.775, 0.844] |
| 0 | 0.75 | 0.45 | 0.184 [0.152, 0.220] | 0.986 [0.971, 0.993] | 0.996 [0.986, 0.999] |
| 30 | 0.25 | 0.45 | 0.706 [0.665, 0.744] | 0.136 [0.109, 0.169] | 0.170 [0.140, 0.205] |
| 30 | 0.5 | 0.45 | 0.406 [0.364, 0.450] | 0.336 [0.296, 0.379] | 0.442 [0.399, 0.486] |
| 30 | 0.75 | 0.45 | 0.128 [0.102, 0.160] | 0.638 [0.595, 0.679] | 0.746 [0.706, 0.782] |
| 60 | 0.25 | 0.45 | 0.150 [0.121, 0.184] | 0.126 [0.100, 0.158] | 0.126 [0.100, 0.158] |
| 60 | 0.5 | 0.45 | 0.096 [0.073, 0.125] | 0.094 [0.071, 0.123] | 0.126 [0.100, 0.158] |
| 60 | 0.75 | 0.45 | 0.080 [0.059, 0.107] | 0.124 [0.098, 0.156] | 0.150 [0.121, 0.184] |

### 5.3  The galaxy channel with a triaxial halo

Every CDM galaxy in Run BF's generator gets a SPHERICAL halo, so its galaxy m=3 detector — the one channel where the tensor is detectable at a small amplitude — has nothing to fire on.  This is the missing arm.

Null: `G_ext` = -0.09 +- 1.60; critical value 3.03 two-sided.

| arm | rate, two-sided | rate, one-sided upper | mean |
|---|---|---|---|
| `tensor_q0.02` | 0.070 [0.051, 0.096] | 0.120 [0.094, 0.151] | +0.70 |
| `tensor_q0.05` | 0.166 [0.136, 0.201] | 0.280 [0.242, 0.321] | +1.58 |
| `tensor_q0.1` | 0.532 [0.488, 0.575] | 0.646 [0.603, 0.687] | +3.15 |
| `tensor_q0.2` | 0.978 [0.961, 0.988] | 0.988 [0.974, 0.994] | +6.28 |
| `halo_q0.05_mis25_flss0` | 0.056 [0.039, 0.080] | 0.060 [0.042, 0.084] | +0.07 |
| `halo_q0.05_mis45_flss0` | 0.078 [0.058, 0.105] | 0.076 [0.056, 0.103] | -0.02 |
| `halo_q0.05_mis25_flss0.5` | 0.076 [0.056, 0.103] | 0.122 [0.096, 0.154] | +0.74 |
| `halo_q0.05_mis25_flss1` | 0.112 [0.087, 0.143] | 0.176 [0.145, 0.212] | +1.13 |
| `halo_q0.05_mis90_flss0` | 0.056 [0.039, 0.080] | 0.076 [0.056, 0.103] | +0.04 |
| `halo_q0.1_mis25_flss0` | 0.064 [0.046, 0.089] | 0.052 [0.036, 0.075] | -0.05 |
| `halo_q0.1_mis45_flss0` | 0.060 [0.042, 0.084] | 0.064 [0.046, 0.089] | -0.12 |
| `halo_q0.1_mis25_flss0.5` | 0.144 [0.116, 0.177] | 0.210 [0.177, 0.248] | +1.36 |
| `halo_q0.1_mis25_flss1` | 0.280 [0.242, 0.321] | 0.408 [0.366, 0.452] | +2.17 |
| `halo_q0.1_mis90_flss0` | 0.084 [0.063, 0.112] | 0.086 [0.064, 0.114] | +0.08 |
| `halo_q0.2_mis25_flss0` | 0.088 [0.066, 0.116] | 0.074 [0.054, 0.100] | -0.02 |
| `halo_q0.2_mis45_flss0` | 0.100 [0.077, 0.129] | 0.100 [0.077, 0.129] | +0.02 |
| `halo_q0.2_mis25_flss0.5` | 0.440 [0.397, 0.484] | 0.542 [0.498, 0.585] | +2.70 |
| `halo_q0.2_mis25_flss1` | 0.762 [0.723, 0.797] | 0.844 [0.810, 0.873] | +4.26 |
| `halo_q0.2_mis90_flss0` | 0.110 [0.085, 0.140] | 0.094 [0.071, 0.123] | +0.11 |

### 5.4  Out-of-grammar injections (Stage 4 C6)

A log-Gaussian ring in the quadrupole, a radial family neither the generator nor the halo model contains.

| injected on | amplitude | recovery rate | mean statistic |
|---|---|---|---|
| external axis (`S_ext`) | 0.15 | 1.000 [0.992, 1.000] | +22.09 |
| external axis (`S_ext`) | 0.3 | 1.000 [0.992, 1.000] | +44.26 |
| external axis (`S_ext`) | 0.6 | 1.000 [0.992, 1.000] | +89.80 |
| baryon axis (`S_bar`) | 0 | 0.003 [0.000, 0.014] | +0.00 |
| baryon axis (`S_bar`) | 0.05 | 1.000 [0.990, 1.000] | +7.36 |
| baryon axis (`S_bar`) | 0.1 | 1.000 [0.990, 1.000] | +14.75 |
| baryon axis (`S_bar`) | 0.2 | 1.000 [0.990, 1.000] | +29.96 |
| baryon axis (`S_bar`) | 0.4 | 1.000 [0.990, 1.000] | +59.71 |

Per statistic, at injected ring amplitude 0.1 (two-sided recovery, with the one-sided rate beside it):

| statistic | mean | recovery, two-sided | recovery, one-sided upper |
|---|---|---|---|
| `S_bar` | +14.75 | 1.000 [0.990, 1.000] | 1.000 [0.990, 1.000] |
| `S_diff` | -9.44 | 0.943 [0.915, 0.961] | 0.000 [0.000, 0.010] |
| `S_morph` | -0.01 | 0.107 [0.081, 0.142] | 0.085 [0.061, 0.116] |
| `S_shape` | -5.09 | 1.000 [0.990, 1.000] | 0.000 [0.000, 0.010] |

`S_morph` does not recover it, and that is the correct answer: an out-of-grammar quadrupole of FIXED amplitude carries no correlation with the baryon ellipticity, which is the only thing `S_morph` reads.  `S_diff` and `S_shape` recover it only two-sided, because the ring's radial profile reverses their sign — so neither can be used to say WHICH family produced a quadrupole once the radial family is unknown.

### 5.5  Responsiveness in the independent model

| statistic | d(S)/d(halo ellipticity) | d(S)/d(tensor amplitude) |
|---|---|---|
| `S_bar` | +9.084 +- 0.130 (t = +70.1) | -0.150 +- 0.082 (t = -1.8) — **consistent with zero, no upper limit set** |
| `S_ext` | -0.075 +- 0.099 (t = -0.8) — **consistent with zero, no upper limit set** | +14.016 +- 0.081 (t = +173.9) |
| `S_diff` | -5.837 +- 0.164 (t = -35.6) | +9.056 +- 0.043 (t = +208.9) |
| `S_morph` | -0.011 +- 0.028 (t = -0.4) — **consistent with zero, no upper limit set** | -0.054 +- 0.044 (t = -1.2) — **consistent with zero, no upper limit set** |
| `S_shape` | -0.032 +- 0.046 (t = -0.7) — **consistent with zero, no upper limit set** | -1.493 +- 0.099 (t = -15.1) |
| `S_45` | +0.067 +- 0.064 (t = +1.0) — **consistent with zero, no upper limit set** | -0.016 +- 0.187 (t = -0.1) — **consistent with zero, no upper limit set** |

## 6  Job 3 — the honest answer

### 6.1  A bug this lane's own tests found

The first version of `forward.f_halo` returned `+0.5 e R kappa0'` for an elliptical-NFW convergence, which puts the MINOR axis where the major axis belongs.  Because `kappa0' < 0` that flipped the sign of every halo statistic: the independent model reported `S_bar = -2.4` for the same physical universe in which Run BF's generator gives `+10.6`.  A single implementation would have reported a confident, wrong-signed result and the alignment scan would have been mirrored.  Test T6 now runs both forward models on every commit and requires them to agree in SIGN.

| test | result | detail |
|---|---|---|
| T1 phase recovered on an injected quadrupole | PASS | median phase error 1.96 deg |
| T1 amplitude recovered within 10% | PASS | amplitude bias +0.000 |
| T2 debiased quadrupole power is null-centred | PASS | mean/sem = +0.99 (undebiased would be +16.8) |
| T3 covariance is calibrated (E[chi2] = 2) | PASS | mean chi2 = 1.88 |
| T4 estimator is rotationally equivariant | PASS | phase rotated by 66.00 deg, expected 66.00 |
| T5 analytic NFW Sigmabar matches direct integration | PASS | max relative error 2.50e-03 |
| T5 m=2 Green's function solves its own ODE | PASS | max relative residual 8.26e-07 |
| T6 both forward models agree in SIGN on S_bar | PASS | BF generator +9.85, independent model +4.50 |
| T6 both forward models agree in SIGN on S_ext | PASS | BF generator +3.72, independent model +5.66 |
| T7 galaxy m=3 phase recovered | PASS | median |psi_hat - psi_obs| = 12.6 deg at q=0.6 (null 45; floor ~10 from the 12 deg axis error) |
| T7 galaxy m=3 amplitude is responsive | PASS | d(aligned projection)/d(q_amp) = 0.561, expected ~0.5 |
| T8 a SEALED path raises before it can be read | PASS | KiDS token guarded |
| T8 a CONFIRMATION-RESERVE path raises | PASS | CLoGS token guarded |
| T8 a foreign read outside the lane raises | PASS | lane-root guard active |
| T9 label-scrambled arm returns S_ext to the null | PASS | real +5.49, scrambled +0.48 +- 0.22 |
| T10 a 45-degree misspecified axis does not respond | PASS | d S_45/dA = +0.12 vs d S_ext/dA = +14.44 |

### 6.2  The answer: yes, and the confusion was a detector defect

A statistic separates.  The procedure is:

> Estimate the complex m=2 quadrupole of the shear field per cluster, jointly with the monopole and m=4 and in BOTH the tangential and the cross component, with its covariance.  Project it, SIGNED and studentised, on (a) the independently measured external axis and (b) the baryon major axis.  Declare directional gravity only if (a) fires and (b) does not.

| quantity | value |
|---|---|
| false-positive rate on the DARK MATTER universe | **0.002 [0.001, 0.007]** |
| the same with 3x systematics | 0.000 [0.000, 0.004] |
| the same on the systematics-only universe U10 | 0.000 [0.000, 0.004] |
| false-positive rate on the scalar null U03 | 0.049 [0.037, 0.064] |
| power on the tensor universe at its fiducial amplitude A = 0.5 | **0.989 [0.980, 0.994]** |
| power at A = 1.0 | 0.998 [0.993, 0.999] |
| power at Run BF's detectable amplitude A = 0.0200 | 0.059 [0.046, 0.075] |
| Run BF's own family-wise rate on the dark-matter universe | 0.648 [0.604, 0.689] |

**0.648 goes to 0.002 [0.001, 0.007]** at power 0.989 against the tensor at its fiducial amplitude.  Nothing about the physics changed.  What changed is that the test keeps the SIGN of the projection and carries a veto on the one configuration a collisionless halo is bound to produce.

The reason Run BF's rate was so high is measured in section 2.6 and is not a modelling subtlety: `aniso_ext_minus_bar` is a TWO-SIDED test of an asymmetric quantity.  A halo drives it strongly NEGATIVE and a tensor strongly POSITIVE, and |S| puts them on the same side of the threshold.  An independent reimplementation reproduces the rate: `S_diff_raw` fires on CDM at 0.456 [0.425, 0.487] against Run BF's 0.479 for the same statistic, and at 0.000 [0.000, 0.004] once the tail is split.

### 6.3  The amplitude and the sample size at which the answer changes

| question | answer |
|---|---|
| amplitude at which the GALAXY m=3 channel reaches 3 sigma against CDM | A = 0.115 (responsiveness +41.3 +- 0.3 per unit A, null sd 1.59) |
| amplitude at which the CLUSTER quadrupole reaches 3 sigma | A = 0.78 (responsiveness +3.94 +- 0.07, null sd 1.03) |
| Run BF's detectable amplitude for the same universe | A = 0.0200 |
| power of the joint procedure there | 0.059 [0.046, 0.075] |
| clusters for a 3 sigma CDM detection with `S_bar` | 1.4 (S = 2.62 sqrt(N), null sd 1.04; the sqrt(N) law is measured at N = 3, 6, 12, 18) |
| galaxies used for the m=3 channel | 30 per corpus, fixed |

**Below A about 0.1 nothing separates.**  At Run BF's own detectable amplitude A = 0.0200 the joint procedure has power 0.059, indistinguishable from its size.  The gap is a factor of about 6 in amplitude and it is not closed by more clusters: the cluster quadrupole needs A = 0.78 and the galaxy channel, which is the sensitive one, does not scale with the number of clusters at all.

**The galaxy channel scales with the number of GALAXIES, not clusters, and the law is measured before it is extrapolated.**  `G_ext = k sqrt(N_gal)`, fitted over the range the shared scene library allows:

| arm | k | max fractional deviation from the sqrt(N) law over N_gal = 10, 20, 30, 45 | null sd |
|---|---|---|---|
| U05_fid | +3.6540 | 0.008 | 1.89 |
| U05_A0.1 | +0.7201 | 0.025 | 1.60 |
| U03_mond | -0.0255 | n/a (k consistent with zero) | 1.72 |
| U02_cdm | -0.0199 | n/a (k consistent with zero) | 1.78 |

| tensor amplitude A | mean `G_ext` at 30 galaxies | k = G_ext/sqrt(30) | galaxies for 3 sigma against CDM |
|---|---|---|---|
| 0 | -0.05 | -0.010 | not reached |
| 0.0125 | +0.56 | +0.102 | 3,111 |
| 0.025 | +0.78 | +0.143 | 1,578 |
| 0.05 | +1.89 | +0.344 | 271 |
| 0.1 | +4.02 | +0.734 | 60 |
| 0.25 | +9.76 | +1.782 | 10 |
| 0.5 | +20.02 | +3.655 | 2 |
| 1 | +41.40 | +7.559 | 1 |

**At Run BF's own detectable amplitude A = 0.0200 the galaxy m=3 channel needs about 2,012 galaxies** with resolved velocity fields AND an independently measured external axis each, against the 30 a corpus contains.  That is an extrapolation of a factor 45 beyond the measured range, quoted only because the sqrt(N) law itself was measured to 0.8% over N_gal = 10 to 45.

**The separation has a second threshold, and it is not an amplitude.**  It is the alignment of the collisionless halo with the large-scale structure -- the quantity Run BF's generator sets to exactly zero.

| halo/baryon misalignment | alignment taken from the external axis | halo ellipticity | detector fires | veto fires | **joint false-positive rate on CDM** |
|---|---|---|---|---|---|
| 22 deg | 0 | 0.45 | 0.094 [0.071, 0.123] | 0.904 [0.875, 0.927] | **0.000 [0.000, 0.008]** |
| 22 deg | 0.125 | 0.45 | 0.116 [0.091, 0.147] | 0.914 [0.886, 0.936] | **0.006 [0.002, 0.017]** |
| 22 deg | 0.25 | 0.45 | 0.122 [0.096, 0.154] | 0.900 [0.871, 0.923] | **0.012 [0.006, 0.026]** |
| 22 deg | 0.375 | 0.45 | 0.198 [0.165, 0.235] | 0.862 [0.829, 0.889] | **0.036 [0.023, 0.056]** |
| 22 deg | 0.5 | 0.45 | 0.484 [0.440, 0.528] | 0.560 [0.516, 0.603] | **0.226 [0.192, 0.265]** |
| 22 deg | 0.625 | 0.45 | 0.770 [0.731, 0.805] | 0.246 [0.210, 0.286] | **0.592 [0.548, 0.634]** |
| 22 deg | 0.75 | 0.45 | 0.848 [0.814, 0.877] | 0.160 [0.130, 0.195] | **0.706 [0.665, 0.744]** |
| 22 deg | 1 | 0.45 | 0.880 [0.849, 0.906] | 0.082 [0.061, 0.109] | **0.808 [0.771, 0.840]** |
| 0 deg | 0 | 0.45 | 0.122 [0.096, 0.154] | 0.998 [0.989, 1.000] | **0.000 [0.000, 0.008]** |
| 15 deg | 0 | 0.45 | 0.100 [0.077, 0.129] | 0.972 [0.954, 0.983] | **0.006 [0.002, 0.017]** |
| 30 deg | 0 | 0.45 | 0.134 [0.107, 0.167] | 0.718 [0.677, 0.756] | **0.032 [0.020, 0.051]** |
| 45 deg | 0 | 0.45 | 0.104 [0.080, 0.134] | 0.294 [0.256, 0.335] | **0.070 [0.051, 0.096]** |
| 60 deg | 0 | 0.45 | 0.108 [0.084, 0.138] | 0.116 [0.091, 0.147] | **0.094 [0.071, 0.123]** |
| 90 deg | 0 | 0.45 | 0.088 [0.066, 0.116] | 0.068 [0.049, 0.094] | **0.078 [0.058, 0.105]** |
| 22 deg | 0 | 0.15 | 0.016 [0.008, 0.031] | 0.168 [0.138, 0.203] | **0.014 [0.007, 0.029]** |
| 22 deg | 0 | 0.3 | 0.044 [0.029, 0.066] | 0.630 [0.587, 0.671] | **0.006 [0.002, 0.017]** |
| 22 deg | 0 | 0.6 | 0.198 [0.165, 0.235] | 0.978 [0.961, 0.988] | **0.002 [0.000, 0.011]** |
| 22 deg | 0 | 0.8 | 0.306 [0.267, 0.348] | 0.998 [0.989, 1.000] | **0.000 [0.000, 0.008]** |
| _none_ | - | - | 0.016 [0.008, 0.031] | 0.004 [0.001, 0.014] | **0.016 [0.008, 0.031]** |
| _tensor_A0.25_ | - | - | 0.800 [0.763, 0.833] | 0.032 [0.020, 0.051] | **0.774 [0.735, 0.808]** |
| _tensor_A0.5_ | - | - | 1.000 [0.992, 1.000] | 0.102 [0.078, 0.132] | **0.898 [0.868, 0.922]** |

**The joint false-positive rate on a dark-matter universe crosses 0.05 at f_lss = 0.38** -- once about 38% of the halo's projected alignment is inherited from the surrounding structure rather than from the baryons, the procedure is no better than Run BF's.  Beyond that point the halo IS the tensor signature, exactly as Run BF said, and no amount of data helps: the two universes then predict the same quadrupole with the same phase.

### 6.4  What does NOT separate, and where no limit is set

* **The reciprocal well-network family (U06) and the path-redshift family (U09) produce no directional signature at all.**  Across the whole scanned range of their knobs every directional statistic is flat: `d(S_ext)/dB = +0.29 +- 0.41 (t = +0.72)`, `d(S_ext)/d(eps) = -1.21 +- 1.10 (t = -1.10)`.  Both are consistent with zero, so **no upper limit on B or on eps is set by any statistic in this lane.**  The one nominally significant slope, `d(G_ext)/dB = -1.52 +- 0.74 (t = -2.04)`, has the wrong sign for any mechanism and sits inside the multiplicity of a 6-point scan over 7 statistics; it is treated as consistent with zero.  For these two families the separation question is not answered negatively -- it is not posed, because they leave no directional observable to test.

* **The matter/light joint behaviour sets no limit** (section 2.4): in this corpus neither universe writes its quadrupole into the member dynamics, and the measured matter-sector amplitude is consistent with zero at the 0.01 level in both.  That axis separates either mechanism from a SLIP, not from each other.

* **A misspecified axis remains a null detector**, reproducing Run BF's result from an independent estimator: `d(S_45)/dA = -0.011 +- 0.047 (t = -0.24)`, consistent with zero -- **no upper limit is set by a misaligned axis** -- while the same statistic still fires on CDM at 0.435 [0.405, 0.466] two-sided.  A detector that cannot see the signal at all still finds dark matter two fifths of the time.

* **`S_bar` cannot tell dark matter from systematics.**  It fires on U10 at 1.000 [0.996, 1.000] and on U02 at 0.998 [0.993, 0.999].  That is the correct behaviour for a VETO -- both are things that are not new gravity -- but it means a positive `S_bar` is not evidence for dark matter, only against a purely external-axis response.

### 6.5  Limits declared by this lane

1. The separation is a statement about the ALIGNMENT PRIOR, not about gravity.  It works because a collisionless halo inherits its shape from the same material whose ellipticity is measured, while the external axis is measured independently.  Section 6.3 gives the alignment fraction at which it fails.
2. Run BF's generator gives every CDM galaxy a SPHERICAL halo, so its galaxy m=3 channel was never tested against a triaxial galaxy halo.  Section 5.3 supplies the missing arm in an independent model: the channel survives a disc-aligned halo and fails against a tidally aligned one.
3. Both forward models put the quadrupole in the lensing sector only (section 2.4).
4. The independent forward model's halo quadrupole is weaker than the generator's at the same nominal ellipticity, so its absolute rates are conservative; read the SHAPE of the alignment curves across, not the absolute values.
5. No real observational data were opened and no confirmation-reserve product was touched.  Nothing here is evidence about the real Universe; it is evidence about what a detector of this kind can and cannot conclude.

