# Lensing closure: two metric potentials, one slip, in the disciplined order

## 0. Formalism, and where the identification comes from

    ds^2 = -(1 + 2 Psi/c^2) c^2 dt^2 + a^2 (1 - 2 Phi/c^2) dx^2
    slow matter  d^2x/dt^2 = -grad Psi      <- the modified Poisson equation gives THIS
    light        deflection ~ grad(Phi+Psi) <- and says nothing about THIS
    slip eta = Phi/Psi,   lensing response Sigma_s = (Phi+Psi)/(2 Psi) = (1+eta)/2

The slip enters in 3-D, `M_len(r) = Sigma_s(r) M_dyn(r)`, and is deprojected and
re-projected by the same Abel integral the shear pipeline uses, so
`gamma_t = DeltaSigma/Sigma_cr` stays exact for a radially varying slip. Reduced
shear carried in full with per-bin measured `<beta>`, `<beta^2>`.

**Identifiability, verified before anything was fitted.** A constant slip applied
to the 3-D lensing mass and the same constant applied to the projected mass give
chi2 agreeing to **0.0e+00 relative**. Within lensing alone, slip and lens mass
are *exactly* degenerate — no shear profile, no image configuration and no time
delay can separate them. **Slip is identifiable only because the dynamics law is
frozen first.** That is the entire source of identification, not a stylistic
preference about ordering. Nonlinearity kept rather than linearised:
`d ln g_+/d ln Sigma_s = 1.0058` mean, kappa reaching 0.040 (RAR) to 0.189.

## 1. THE FREE-CLOSURE CONTROL — what the discipline protects against

Five closures of increasing freedom, applied in 3-D and re-projected, on the
declared TRAIN half (248 systems, 1709 points):

| dynamics law | C0 (no slip) | C1 (1 par) | C2 (2) | C3 (3) | C4 (248, per cluster) |
|---|---|---|---|---|---|
| Newton (baryons only) | 1981.9 | 1877.6 | 1872.9 | 1872.9 | **1656.1** |
| CONTROL: Newton x (r/Mpc)^-1 | 1980.5 | 1880.8 | 1872.9 | 1872.9 | **1685.1** |
| RAR | 1869.7 | 1869.7 | 1850.3 | 1850.3 | 1650.2 |
| AQUAL | 1869.7 | 1869.7 | 1850.3 | 1850.3 | 1650.2 |
| tidal-gated scalar A=7.5 | 2305.0 | 1871.0 | 1850.3 | 1850.3 | 1652.1 |

Reference: **RAR under no slip, chi2 = 1869.7.** Fraction of the Newton-to-RAR
gap that a free closure recovers:

| law + closure | par | chi2 | vs RAR+no-slip | gap recovered |
|---|---|---|---|---|
| newton + C1 | 1 | 1877.6 | +7.9 | **93.0%** |
| newton + C2 | 2 | 1872.9 | +3.2 | **97.1%** |
| newton + C4 | 248 | 1656.1 | **-213.6** | 290% |
| wrongshape + C1 | 1 | 1880.8 | +11.1 | 90.0% |
| wrongshape + C2 | 2 | 1872.9 | +3.2 | 97.1% |
| wrongshape + C4 | 248 | 1685.1 | **-184.6** | 267% |

**(i)** One closure parameter buys 93% of the gap, two buy 97%. Newton with two
closure parameters is Delta chi2 = 3.2 from the RAR with none, on 1709 points —
not distinguishable. **(ii)** An unrestricted per-cluster closure takes an
obviously wrong law **214 chi2 PAST** the correct one. **(iii)** Newton + C2 and
the wrong-shape control + C2 land on *exactly* the same chi2, because
`(r/Mpc)^s x (r/Mpc)^-1 = (r/Mpc)^(s-1)` — **a radial closure and a radial
modification of the force law are algebraically identical.**

What the free closure learned (Newton at C2) against the RAR's dynamical boost:

| r [Mpc] | 0.3 | 0.5 | **1.0** | 2.0 | 3.0 |
|---|---|---|---|---|---|
| fitted Sigma_s(r) | 16.65 | 12.90 | **9.12** | 6.45 | 5.27 |
| RAR boost B(r) | 6.65 | 7.16 | **9.31** | 15.74 | 22.61 |

At 1 Mpc, where the shear carries most of its weight, the fitted "closure"
reproduces the RAR's dynamical boost **to 2%**; away from there they diverge by
factors of 2-4 and the data do not care.

## 2. Provenance audit — and one headline constant FAILS

| law | frozen constants | verdict |
|---|---|---|
| RAR | a0 = 1.0844e-10, SPARC train rotation curves | dynamics only, **passes** |
| AQUAL | a0 = 1.0580375e-10, SPARC train | dynamics only, **passes** |
| tidal scalar | a0 = 1.0023e-10 SPARC; T0 = 1e-33, m = 2 declared grid; **A = 7.5** from the X-COP flat target (nu/nu_RAR = 2.53, A2029), a hydrostatic X-ray constraint | hydrostatic gas is slow matter, so dynamics: **passes** |
| tidal scalar, headline | identical but **A = 16.0**, selected against the lane-12 radial requirement, interpolated from **published lensing MASS profiles** | those masses assume Sigma_s = 1: **FAILS** |

**This is a finding, not a technicality.** Scoring A = 16 against raw shear under
no slip and calling the agreement a success would be circular — the amplitude was
set by the answer. It is the "never fit the law and the closure simultaneously"
rule caught one level upstream: the closure was not fitted here, it was
*inherited*. **A = 7.5 is taken as primary** and A = 16 carried beside it,
labelled.

### 2b. LEVERAGE — eFEDS cannot test the tidal gate at all

Gate reach measured on the 3365 *measured* points before anything was scored:

    W = 1/(1+(|T|/T0)^2)   min 0.7440, 1st pct 0.9338, median 0.9998
                           frac W < 0.99 = 0.120, frac W < 0.90 = 0.0045
    log10(B_tidal/B_RAR)   median +0.4390 dex (factor 2.748)
                           sd 0.0085 dex, full range 0.0887 dex

**The gate is saturated.** |T| sits far below T0 across the whole measured range,
so the tidal law degenerates to AQUAL with `a0 -> a0(1+A)` — a *constant*
rescaling of the RAR by sqrt(1+A) = 2.915, varying by 0.0085 dex over the entire
data set. **A universal slip absorbs a constant exactly**, so here the tidal law
and the RAR are the same hypothesis up to a closure.

The programme's sharpest falsifiable claim — a boost that RISES outward where the
lensing-derived shape falls — is **not testable on eFEDS weak lensing**, and no
number of extra groups would change it. `|T| ~ T0` needs ~1e14 Msun inside a few
hundred kpc: cluster cores. Quantified by embedding both laws in one family
`g = Sigma_s g_RAR 10^{lambda Delta}` and profiling Sigma_s out:
**lambda_hat = -7.50 [-8.50, -6.00], sigma(lambda) = 1.25**, against a
lambda = 0-vs-1 separation of **0.80 sigma**. No power. (lambda_hat being ~6
sigma from BOTH hypotheses is the signature of a template acting as a
mass/radius proxy rather than testing the gate.)

## 3. STEP 3 — raw shear under NO SLIP, nothing fitted

All 496 systems, 3365 points, Sigma_s = 1 exactly, zero free parameters.

| law | chi2 | chi2/N | mean pull | Sigma_s it would need | eta |
|---|---|---|---|---|---|
| Newton | 3803.2 | 1.1302 | +0.2170 | 8.322 | +15.64 |
| **RAR** | **3588.4** | **1.0664** | -0.0036 | **0.981** | **+0.96** |
| **AQUAL** | **3588.2** | **1.0663** | -0.0009 | **0.992** | **+0.98** |
| tidal A=7.5 | 4472.9 | 1.3292 | -0.4276 | 0.358 | **-0.28** |
| tidal A=16 | 5998.2 | 1.7825 | -0.7069 | 0.253 | **-0.49** |
| *g_pred = 0* | *3865.0* | *1.1486* | | | |

**Both tidal variants fit the raw shear worse than predicting no lensing signal
at all.** The RAR and AQUAL, with a0 frozen on galaxy rotation curves, land on
the observed cluster shear needing a lensing response of 0.98 and 0.99 — **no
slip, to within 2%**. Both tidal variants require **eta < 0**: light bending the
wrong way relative to matter.

**Shear calibration is small, and this corrects the efeds-hsc lane.** Stacking
the *same* clusters against Chiu+2022's HSC profile:

| subsample | n | median M_gas,500 | DECADE/HSC |
|---|---|---|---|
| all | 496 | 6.47e12 | 0.273 (-0.564 dex) |
| top 50% by M_gas,500 | 248 | 1.21e13 | **0.914 (-0.039 dex)** |
| top 20% | 100 | 2.01e13 | **0.875 (-0.058 dex)** |

The 0.56 dex offset is **sample composition, not shear calibration**;
mass-matched, DECADE is good to ~0.05 dex. The efeds-hsc lane's inferred 0.2-0.4
dex photo-z dilution came from comparing *fitted amplitudes* across two model
setups rather than comparing the data.

## 4. Structured or noise-like? (TRAIN only)

| law | best constant Sigma_s | shape chi2 (10 bins) | log-r slope | cross-system excess |
|---|---|---|---|---|
| Newton | 8.511 | **13.9** | -0.101 +- 0.085 (-1.2 s) | 1.17x |
| RAR | 1.000 | 19.8 | **-0.311 +- 0.085 (-3.7 s)** | 1.11x |
| AQUAL | 1.000 | 19.8 | -0.309 +- 0.085 | 1.11x |
| tidal A=7.5 | 0.372 | 20.8 | -0.323 +- 0.085 | 1.11x |
| tidal A=16 | 0.263 | 21.0 | -0.324 +- 0.085 | 1.11x |

Noise reference on the same data: B-mode `<g_x> = -0.000033 +- 0.000199`, per-bin
chi2 = 9.3 on 10 bins. So the failure is **mildly structured in radius and
nothing else** (20 against a floor of 9). The slope is negative — models
over-predict at large radius. Member contamination mimics a *positive* slope, so
it works against this signal rather than explaining it. Note: **Newton with a
large constant slip has LESS radial structure (13.9) than the RAR with none
(19.8)** — the observed shear's radial shape is closer to the baryon shape than
to the RAR-boosted shape.

## 5. STEP 4 — ONE universal slip, TRAIN only

| law | Sigma_s (Delta chi2 = 1) | eta | null-corrected Sigma_s |
|---|---|---|---|
| Newton | 8.511 [7.762, 9.120] | +16.02 | 9.04 - 18.45 |
| **RAR** | **1.000 [0.933, 1.072]** | **+1.000 [+0.867, +1.143]** | **1.06 - 2.17** |
| AQUAL | 1.000 [0.933, 1.072] | +1.000 | 1.06 - 2.17 |
| tidal A=7.5 | 0.372 [0.339, 0.398] | -0.257 | 0.39 - 0.81 |
| tidal A=16 | 0.263 [0.240, 0.282] | -0.474 | 0.28 - 0.57 |

Nothing at a grid edge. **Responsiveness gate:** injected
[-0.40, -0.20, 0.00, +0.20, +0.40, +0.60] recovered
[-0.34, -0.22, +0.01, +0.22, +0.42, +0.59]; slope **0.9686**, spread **0.930 dex
over 1.00 dex injected**. PASS.

**Shared-quantity null with the ACTUAL published errors,** bracketed over three
error scalings because the published errors are marginal and the Vikhlinin
parameters are strongly covariant:

| error scale | E[est \| H0] | sd | bias [dex] | sigma_MC |
|---|---|---|---|---|
| 0.25 | -0.026 | 0.058 | -0.0260 | -2.5 |
| 0.50 | -0.125 | 0.070 | -0.1253 | -9.8 |
| 1.00 | -0.336 | 0.108 | **-0.3360** | **-17.0** |

**The null fires hard.** X-ray density-fit noise ALONE drags a fitted Sigma_s
down by up to 0.34 dex (factor 2.2) at 17 sigma_MC, scaling as the variance.
Sixth artefact of this family, and larger than efeds-hsc's -0.0666 because
Sigma_s is a pure amplitude. **Every Sigma_s must be read against this null, not
against 1.** Its factor-2 width is the dominant uncertainty of the lane, and it
is a property of the published X-ray catalogue, not the shear.

## 6-7. Frozen transfer to the held-out half, once

248 systems, 1656 points, everything frozen at TRAIN values.

| law | Sigma_s frozen | chi2 | chi2/N | (forbidden refit) | d log10 |
|---|---|---|---|---|---|
| Newton | 8.511 | 1714.7 | 1.0354 | 8.128 | -0.020 |
| RAR | 1.000 | 1718.7 | 1.0379 | 0.977 | -0.010 |
| AQUAL | 1.000 | 1718.6 | 1.0378 | 0.977 | -0.010 |
| tidal A=7.5 | 0.372 | 1720.7 | 1.0390 | 0.355 | -0.020 |
| tidal A=16 | 0.263 | 1720.9 | 1.0392 | 0.257 | -0.010 |

**The slip transfers cleanly.** A forbidden refit would have bought 0.06-0.46
chi2; shown and discarded.

**The radial structure does NOT transfer:** the RAR's -3.7 sigma training slope
reappears at **-1.1 sigma** where it was not sought (-0.095 +- 0.087), and
Newton's flips sign to +0.109. Combined -0.206 +- 0.061. **Reported as not
established.** The held-out half is the same survey, instrument, photo-z code and
X-ray catalogue — it controls overfitting, not systematics.

Sensitivity: f_star 0/0.15/0.30, r_trunc 10/20/40 Mpc, a0 x0.9/1.0/1.1 all move
log10 Sigma_s by < 0.07 dex.

---

# SN REFSDAL — the Fermat depth as a closure probe

Used last, and treated as a **joint lens-potential-and-time-delay test, not a
mass-model-free discriminator.**

**R0.** Measured, no mass model: `Delta t(SX-S1) = 376.02 d, 16-84th
370.50-381.65, 1.48%`. Flat LCDM H0 = 70, Om = 0.3 declared: D_l = 1312.8 Mpc,
D_s = 1745.1, D_ls = 931.7, **D_dt = 3791.5 Mpc**, Sigma_cr = 2.372e9 Msun/kpc^2.
Hence **required `Delta phi(SX-S1) = 3.5419 +- 0.0525 arcsec^2`** — as far as the
measurement reaches alone. Images relative to the Shipley+2018 spec-confirmed
BCG: S1 10.58", S2 10.44", S3 11.70", S4 12.46", and **SX at 7.95"**, inside them
and ~50 deg away in position angle.

**R1 — the mass-sheet transform, measured not asserted.** Image positions found
by bisecting the *signed* lens equation independently at each lambda:

| lam | n images | max image shift ["] | Delta phi ["^2] | ratio to lam=1 |
|---|---|---|---|---|
| 0.50 | 3 | 1.4e-14 | 16.492575 | 0.500000 |
| 1.00 | 3 | 0 | 32.985150 | 1.000000 |
| 2.00 | 3 | 4.4e-16 | 65.970300 | 2.000000 |

**Image positions cannot see this closure change at all; delays see it
linearly.** A uniform slip is the rescaling half without the compensating sheet,
so it does move the Einstein radius — hence two independent handles.

**R2 — baryons.** ACCEPT deprojected n_e (41 shells, 0-1.301 Mpc) plus 132
Molino+2017 CLASH members (cuts declared), fitted by two Hernquist components to
the measured projected cumulative M* (rms 0.051 dex over 5-468 kpc). At 50-100
kpc the *baryonic* acceleration is only ~0.1 a0 while the lensing mass needed to
make the images is ~3e13 Msun inside 64 kpc — almost two orders of magnitude
more.

**R3/R4 — two independent handles.**

| law | kappa_bar(theta_E) | **Sigma_s from images** | **Sigma_s from delay** | ratio | beta rms ["] |
|---|---|---|---|---|---|
| Newton | 0.063 | 15.822 | 13.907 | 0.879 | 0.403 |
| RAR | 0.217 | 4.615 | 4.113 | **0.891** | 0.466 |
| AQUAL | 0.215 | 4.642 | 4.136 | 0.891 | 0.482 |
| tidal A=7.5 | 0.408 | 2.449 | 2.117 | 0.864 | 0.608 |
| tidal A=16 | 0.545 | 1.834 | 1.574 | 0.858 | 0.422 |

The delay column is a proper joint solve: at each Sigma_s the lens is rebuilt and
the source re-solved from all five images. **The two estimators agree to 11-14%
for every law** — an internal check, not proof the monopole is right.

**R5.** One closure parameter is *enough* to bring every law, including
unmodified Newton, onto 376.02 d. **A single delay can never test a gravity
law**; it can only measure the closure, and only with the law frozen and the lens
model right.

**R6 — budget, in the recovered slip** (baseline RAR 4.113): centre moved 5.6"
-> 3.552 (-13.6%); baryons truncated at 1.30 Mpc -> +3.4%; M* x0.5 -> +1.8%;
M* x2 -> -3.4%; M* x10 -> -22.6%; **M_gas x0.5 -> +43.5%; M_gas x2 -> -32.7%**.
Closing the factor ~4 would take ~8x the ACCEPT gas inside 80 kpc or 100x the
catalogued stars. **The strong-lensing deficit is not a baryon bookkeeping
error.** The dominant error is not in this table: MACS J1149 is a merger whose
images are not collinear with its centre (beta rms 0.40-0.61").

**R7 — DOES ONE UNIVERSAL SLIP SERVE BOTH REGIMES?**

| law | Sigma_s WL (raw) | WL null-corrected | SL (delay) | SL (images) | **SL/WL** |
|---|---|---|---|---|---|
| Newton | 8.511 | 9.04 - 18.45 | 13.907 | 15.822 | **0.8 - 1.5** |
| RAR | 1.000 | 1.06 - 2.17 | 4.113 | 4.615 | **1.9 - 3.9** |
| AQUAL | 1.000 | 1.06 - 2.17 | 4.136 | 4.642 | 1.9 - 3.9 |
| tidal A=7.5 | 0.372 | 0.39 - 0.81 | 2.117 | 2.449 | **2.6 - 5.4** |
| tidal A=16 | 0.263 | 0.28 - 0.57 | 1.574 | 1.834 | 2.8 - 5.6 |

**Newton is the only one of the five for which a single universal Sigma_s serves
both regimes** — its bracket contains 1. Every MOND-like law needs 2-6x more
lensing response in the cluster core than in the group outskirts. **And the tidal
gate makes it worse:** it multiplies the RAR by 2.75 in the eFEDS groups where
the shear already agreed, but by only 1.88 at MACS J1149's 50-80 kpc where |T| is
large and the gate is partly off. Its sign is backwards for the cluster problem.
*Confound, stated:* the two regimes differ in radius AND in host mass.

## Failure modes on the standing checklist

Shared-quantity artefacts: simulated at three error scalings, **fired at -0.336
dex, 17 sigma_MC**; every Sigma_s quoted against it; construction expressions
written out and share no input, so it is a bias not a spurious correlation.
Monotone-invariance: d(Sigma_hat)/d(Sigma_inj) = 0.9686, spread 0.930 dex
printed. Refitting on held-out: frozen, forbidden refit displayed and discarded.
Silent extraction: counts asserted and identifiers echoed (496/3365; 542/542
Bahar; 41 ACCEPT shells; 34 Treu images; 5 Refsdal images; 103 Molino columns).
M_gas,500 gate median 0.9994, scatter 0.0469 dex, n = 414.

**Two test bugs caught:** (i) D_dt converted to metres twice, giving
`Delta phi = 0.0000` — caught because the number was impossible, not by a test;
(ii) the MST demo solved the lens equation only for theta > 0 and found one image
where a critical lens must give three, the counter-image living at negative
signed radius.

Boundary-rule dependence: not applicable — no potential-depth variable is used,
and the tidal invariant is a Hessian needing no boundary convention, which is a
genuine advantage of it. Detector power measured: sigma(lambda) = 1.25 against a
separation of 1.

## What could NOT be established

1. **The absolute slip to better than a factor of two.** The limiter is not the
   shear (~0.05 dex mass-matched) but the errors-in-variables bias from the
   published Vikhlinin parameters, bracketed 0.03-0.34 dex because their
   covariance is not published. **Publish that covariance and the chain measures
   the slip to ~10%.**
2. **The tidal gate's radial structure at all, from weak lensing** — saturation,
   not a statement about the law.
3. **A percent-level Refsdal constraint** — a circular monopole on a merging
   cluster gives beta rms 0.4-0.6".
4. **A genuinely fresh transfer sample** — the held-out half is the same survey.
5. **Whether the -0.2 dex/decade residual radial slope is real** — 3.7 sigma
   where found, 1.1 sigma where not.
