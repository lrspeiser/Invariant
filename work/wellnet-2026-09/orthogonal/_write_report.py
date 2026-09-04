"""Emit the lane REPORT.md required by work/wellnet-2026-09/BRIEF.md."""
import os

TEXT = r"""# Is gravity directional? One object, tracers in two directions, everything else held fixed

Lane `work/wellnet-2026-09/orthogonal/`, 2026-09-04.
Code: `orbit_model.py` (forward model), `adyn_same_object.py` (ingest, fit,
measurement, nulls, error budget, external systems), `warp_pool.py` (pooled
warped discs). Machine-readable: `orthogonal_results.json`.

---

## 0. Predicted versus measured A_dyn, and the power statement

    A_dyn(x) = [ g_R(x)/g_R,N(x) ] / [ |g_z(x)|/|g_z,N(x)| ]   -- at ONE point

Predictions were computed and written to `orthogonal_results.json` **before**
any stream was compared to any model. Reference point: R = 11.28 kpc,
|z| = 3.59 kpc, fixed by the median geometry of the gated stream sample.

| law (frozen) | A_dyn ALGEBRAIC | A_dyn PREDICTED, field completion | at 100 kpc | at (8.12, 1.1) kpc | A_dyn MEASURED |
|---|---|---|---|---|---|
| Newton (control) | 1 exactly | 1.0011 | 1.0015 | 1.0027 | 0.994 +/- 0.125 |
| RAR, a0 = 1.0844e-10 | **1 exactly** | 1.0062 (QUMOND) | 1.0019 | 1.0488 | 1.333 +/- 0.108 |
| AQUAL, same a0 | **1 exactly** | 1.0014 | 1.0028 | 1.0120 | 1.243 +/- 0.108 |
| tidal-gated scalar A=+16, T0=1e-33, m=2 | **1 exactly** | 1.0026 | **1.0437** | 1.0130 | 1.247 +/- 0.114 |
| well-network tensor_S, phi-gate, A=-94.66 | not scalar | 1.0110 | 1.0032 | 1.0259 | 1.347 +/- 0.116 |

Newton's 1.0011 is the **numerical floor** of the whole chain: for Newton the
ratio is 1 analytically, so 0.0011 is what the grid, the solver and the
interpolation contribute. Every prediction above must be read against it.

**POWER STATEMENT.** End-to-end simulations through the entire chain
(rotation-curve noise with its actual covariance, baryon refit, field solve,
progenitor orbit, stripping, projection, Lambda scan, argmin) give a null
distribution for the estimator with mean **1.0004** and sd **0.0408** on a
calibration set and an audit set that are statistically indistinguishable
(KS p = 0.256, t p = 0.240). Injections recover a known Lambda with a
shrinkage of 0.65-0.81 and give detection significances of
z = -1.9 (Lambda_true = 0.90), +2.9 (1.15), +6.3 (1.40), -5.0 (0.75).
Interpolating: **3 sigma requires |A_dyn - 1| ~ 0.16, 5 sigma requires
|A_dyn - 1| ~ 0.27.**

The five candidates' predictions span **1.0011 to 1.0110** at the reference
point -- a total spread of **0.010**. That is **16 times below the 3-sigma
detection threshold** of this measurement and 4 times below its statistical
error alone. **The test as constructed has no power to separate the
candidates, and the shortfall is more than an order of magnitude.** Section 5
shows the real error is worse still: the forward model is mis-specified by a
factor of about 22 in chi-square per observation, so the measured column above
is a model-dependent offset, not a measurement of gravity's directionality.

---

## 1. Why the algebraic prediction is exactly 1, and why that matters

**THEOREM (scalar blindness of A_dyn).** For any law of the form
`g(x) = F(scalar invariants at x) grad Phi_N(x)`, both components carry the
same factor F(x), so A_dyn(x) = 1 identically -- for every x, every F, every
invariant, including invariants built from directional objects and then
reduced to a scalar, such as |T|.

This covers three of the four candidates named in the brief exactly as the
tournament froze them: Newton, the algebraic RAR, and the tidal-gated scalar
`a0 -> a0(1 + A W(|T|/T0))`. **A_dyn is blind to all of them by construction.**
Only a genuinely tensorial field equation, or the CURL FIELD that a nonlinear
field equation carries for a non-spherical source, can move it.

So the whole test rests on the curl field, and the curl field only exists if
the field equation is actually solved. Both completions were therefore solved
here rather than assumed:

* **QUMOND**: `div grad Phi = div[ nu(|grad Phi_N|/a0_eff) grad Phi_N ]`,
  linear in Phi, one solve, source built on the FACE fluxes the finite-volume
  discretisation conserves.
* **AQUAL**: `div[ mu(|grad Phi|/a0_eff) K grad Phi ] = 4 pi G rho`, damped
  Picard, 13-14 iterations to `max|dPsi|/max|Psi| < 3e-7`, each a
  Jacobi-preconditioned warm-started CG to relative residual 1e-10.

**The algebraic laws are not conservative, and the amount is now measured.**
`max |curl g| x 10 kpc / |g|` over R in [6, 80] kpc, |z| in [1, 60] kpc, on
the analytic field:

| law, ALGEBRAIC form | max | median |
|---|---|---|
| Newton | 3.8e-5 | 1.8e-6 |
| RAR | 0.048 | 4.4e-4 |
| AQUAL (algebraic approximation) | 0.049 | 3.7e-4 |
| tidal-gated scalar | **1.08** | 4.8e-4 |
| well-network tensor_S | 0.137 | **0.054** |

Newton returns the machine-precision zero it must, so the estimator is clean.
The tidal-gated scalar's pointwise form has a curl of **108 per cent of |g|
per 10 kpc** somewhere in the stream volume, and the well-network tensor
violates `curl g = 0` by 5.4 per cent of |g| per 10 kpc in the MEDIAN across
that volume. This is the addendum's reciprocity/action gate, measured inside a
disc galaxy rather than in a two-body test: **neither candidate's pointwise
form can come from an action, and no carrier is declared.** Their field
completions, solved here, are conservative by construction; the price is that
the completion is a different theory from the frozen algebraic law, with a
different rotation curve (median completion/algebraic ratio 0.9966 to 1.0006,
re-fitted for).

---

## 2. The procedure, in the order the brief specifies

**1. Fit using ONLY the disc rotation curve.** Eilers 2019, 38 points,
5.27-24.82 kpc, sha256 `7c6d36f7...`, row and column counts asserted on
ingest. Free parameters: three baryon masses (bulge; disc, with thick tied at
0.25; gas). Shape parameters are measured baryonic inputs, not parameters:
Hernquist a = 0.5 kpc; Miyamoto-Nagai (a, b) = (3.0, 0.28), (4.4, 0.9),
(7.0, 0.085) kpc. Declared 5 km/s systematic floor on v_c. The masses are
calibrated against the SOLVED field's midplane curve, not the algebraic
approximation to it, by a two-pass iteration.

| law | M_baryon (Msun) | RMS solved vs data (km/s) | chi2 | grid mass gate |
|---|---|---|---|---|
| Newton | **2.148e+11** | 6.68 | 31.7 | 0.9789 |
| RAR (QUMOND) | 8.703e+10 | 8.53 | 22.2 | 0.9905 |
| AQUAL | 8.531e+10 | 8.32 | 20.4 | 0.9899 |
| tidal scalar | 8.870e+10 | 8.17 | 19.2 | 0.9895 |
| well-network tensor | 8.720e+10 | 8.22 | 19.5 | 0.9895 |

Newton needs 2.1e11 Msun of baryons -- 2.5x the MOND-family value and about
3x the observed Milky Way baryonic mass. That is the frozen Newton model and
it is what its stream predictions are made with.

**2. FREEZE.** No gravity constant and no baryon mass is touched again. The
streams never enter any fit.

**3. Predict.** Table in section 0.

**4. Forward-model the tracer in full phase space.** This is not optional and
it is not shortcut anywhere. For each stream and each candidate: progenitor
orbit integrated back from the anchor; Fardal et al. 2015 particle spray at
each of 80 release times, offsets scaled by the tidal radius computed from the
CANDIDATE field's own `|g| r^2 / G` (not from Newtonian mass); every particle
integrated forward to the present in the same field; the resulting phase-space
distribution projected onto exactly the observables the catalogue contains.
**No stream track is ever converted into g_R and g_z points anywhere in this
lane.**

**The measurement family.** With the in-plane leg frozen, A_dyn is measured by
deforming the potential along the one direction that leaves that leg exactly
invariant:

    Phi_L(R,z) = Phi(R,0) + (1/L) [ Phi(R,z) - Phi(R,0) ]

so that g_R(R,0) is unchanged EXACTLY for every L, g_z is scaled by 1/L
everywhere, and the force stays a gradient (conservative, passes the
reciprocity gate). A_dyn(L) is then computed numerically, never assumed: it
runs from 0.35 to 2.16 across the declared grid L in [0.55, 1.80].

---

## 3. Data, and the defects

`galstreams` v1.2.1, 217 summary rows x 47 columns, both asserted on ingest.
A physical-plausibility gate is applied to EVERY kinematic ingest in this
lane, and the rule is that **the flag governs; data may only downgrade a
track, never promote it.**

* **68+ unit-distance tracks.** 85 tracks have a distance column identically
  1.000 kpc to within float round-trip noise (max |D - 1| = 5.3e-15). An
  exact `== 1.0` test finds NONE of them; the gate uses a 1e-6 tolerance.
  GD-1's `ibata2024` track is among them and is correctly demoted.
* **Superluminal and unbound velocities.** 34 tracks fail the velocity gate,
  including `Hydrus.ibata2024` at 9,561,412 km/s = 32 c.
* **Sentinels and flag-clear-but-populated columns**, `Pal5.pricewhelan2019`
  Vrad = 999.0 among them.
* **InfoFlags digits take the values 0, 1 AND 2.** Testing `== "1"` silently
  drops `M68-Fjorm.palau2019` (flags 1210) and three others. The decode must
  be `> 0`. This one is not in the brief's list.
* **A fifth defect family, not in the brief.** Two tracks pass every listed
  check and still fail on physics: `M68-Fjorm.ibata2021` has distances down to
  **1.8e-4 kpc** (0.18 pc from the Sun) while advertising InfoFlags 1111, and
  `M5.ibata2021` has a radial-velocity track spanning -641 to **+838** km/s.
  Three more (`NGC3201`, `Sylgr`, and `M5`) exceed a 900 km/s galactocentric
  speed. All are demoted.

Resulting sample: **68 usable 3-D tracks (60 distinct streams), 29 usable 6-D
tracks (26 distinct streams)**, against the summary file's own 69 and 33. The
four-track difference is entirely the fifth defect family above; the gate only
ever removes. 15 usable-3D tracks lie within 10 degrees of polar and 14 reach
past R_gc = 25 kpc, the outer edge of the rotation curve.

**Measured sample, declared by geometry before any residual was examined:**
all 26 six-dimensional streams, plus every 3-D-only stream with
R_gc,max > 25 kpc or |inclination - 90 deg| <= 10 deg -- 12 more (Cetus,
Cetus-New, Cetus-Palca, Gaia-2, Gaia-6, Jet, M5, M68-Fjorm, NGC1261, NGC288,
NGC5466, Sagittarius). **38 streams.** For a 3-D-only track the anchor
velocity is not observed, so it is re-searched inside every candidate
potential at every Lambda by a vectorised two-stage 768-trial orbit search; it
is a property of that stream, never a gravity parameter.

---

## 4. Solver, integrator and frame gates

| gate | result | requirement |
|---|---|---|
| Newtonian solve vs analytic force, R >= 15 kpc | 0.00 to 0.03 per cent | recover Newton |
| Newtonian solve vs analytic, R = 8 kpc, z = 1 kpc | 0.2 per cent | -- |
| grid mass on the box vs analytic total | 0.979-0.990 | remainder is mass outside 300 kpc |
| cell-averaged density (12x12 Gauss-Legendre) | **required** | point sampling loses **27 per cent** of the disc mass at dz = 0.75 kpc against b = 0.28 kpc and drives the solved rotation curve to 0.73 of the analytic one |
| AQUAL Picard convergence | 13-14 iterations to 3e-7 | -- |
| CG relative residual | <= 1.0e-10 | -- |
| fast frame vs astropy | 1.3e-12 deg, 5.0e-8 mas/yr | -- |
| leapfrog energy drift, 3.9 Gyr, refine = 2 | 0.3-0.8 per cent (MOND), 4.8 per cent (Newton) | carried as a systematic |
| box size 200 -> 300 kpc | 0.1 per cent at r = 112 kpc | -- |

The mass-gate item is worth repeating because it silently corrupts everything
downstream: with point-sampled density the grid loses a quarter of the disc,
every multiplier is then evaluated for a galaxy 27 per cent too light, and
nothing in the output looks wrong.

---

## 5. What the streams actually say, and why the declared statistic is void

**Responsiveness gate (required).** dS/dtheta != 0 is verified numerically and
the spread is printed. Over the declared grid, A_dyn runs 0.35 to 2.16
(dA_dyn/dLambda = 1.25 to 1.45), and the summed chi-square spans 1.2e5 to
2.6e5. Neither is flat; the statistic is not monotone-invariant.

**The declared statistic fails its own goodness-of-fit gate.** At its best
Lambda the summed chi-square per observation is:

| law | chi2 / n_obs (n_obs = 8448) |
|---|---|
| Newton | 28.9 |
| RAR | 21.9 |
| AQUAL | 22.9 |
| tidal scalar | 21.9 |
| well-network tensor | 22.0 |

A value of 22 means the forward model misses the data by about 4.7 sigma per
point on average. **The summed-chi-square estimator is therefore not
admissible**: its argmin sits at the grid edge (Lambda = 1.80) for four of the
five laws and returns A_dyn about 2.0, which is a mis-specification artefact,
not a measurement. It is reported here only so that the failure is on record.

**The direct measurement of the mis-specification.** Each stream determines
Lambda to a formal Delta-chi2 = 1 width of **+/-0.0001 to +/-0.0003**, and the
38 streams disagree with one another by **+/-0.42** -- an inconsistency of a
factor **1400 to 3400**. Between 24 and 34 per cent of the per-stream argmins
are pinned at a grid edge.

**The robust statistic** (median of the per-stream argmin, standard error
1.2533 sd / sqrt(n)) is the column reported in section 0:

| law | median Lambda | sd across streams | A_dyn | +/- (stat) |
|---|---|---|---|---|
| Newton | 0.995 | 0.424 | 0.994 | 0.125 |
| RAR | 1.261 | 0.425 | 1.333 | 0.108 |
| AQUAL | 1.191 | 0.421 | 1.243 | 0.108 |
| tidal scalar | 1.191 | 0.438 | 1.247 | 0.114 |
| well-network tensor | 1.261 | 0.445 | 1.347 | 0.116 |

This is a post-hoc change of estimator and is flagged as such. It is motivated
by a Lambda-independent goodness-of-fit failure, not by the residuals, but its
own calibration was NOT established by simulation and the quoted errors assume
the per-stream estimates are independent and uncensored, which the 24-34 per
cent edge fraction shows they are not.

**Reading the numbers honestly.** Newton, whose A_dyn is 1 by construction, is
recovered at 0.994 +/- 0.125 -- the estimator is unbiased where the answer is
known. The four MOND-family laws all land at 1.24-1.35, an offset of 0.25-0.35
from Newton. That offset is 25 to 35 times larger than any predicted
difference between the laws and is a statement about which frozen potential
the 38 streams prefer, given a forward model that is demonstrably wrong at
chi2/n = 22. **It is not evidence of directional gravity.**

**Error budget** (8-stream subset, RAR, 12 variants, `orthogonal_results.json`
key `systematics`; baseline A_dyn = 0.9239):

| term | shift in A_dyn |
|---|---|
| rotation-curve systematic floor 5 -> 10 km/s | -0.061 |
| interpolation grid refine 2 -> 1 | -0.054 |
| anchor draws 12 -> 24 | -0.037 |
| stream error model / 1.5 | +0.015 |
| disc scale length -30 per cent | -0.012 |
| rotation-curve floor 5 -> 2 km/s | +0.010 |
| disc scale length +30 per cent | -0.005 |
| disc thickness x2 | -0.004 |
| bulge scale x2 | +0.003 |
| stream error model x1.5 | +0.002 |
| gas scale length x1.5 | 0.000 |
| **half-range** | **0.038** |
| **RSS** | **0.093** |

The two largest terms are not astrophysical. The anchor-draw term shows the
profile minimum over a finite random draw set has not converged at 24 draws;
the grid-refinement term shows the forward model's interpolation is not yet
negligible. Both are fixable with compute, and neither is close to the 0.010
that would be needed.

**Shared-quantity null, simulated with the actual error covariance (required
check).** The Eilers bins are not independent; the declared covariance is
tabulated errors + nearest-neighbour rho = 0.5 + a fully correlated 5 km/s
systematic, and that last term is the shared quantity. Repeating the entire
chain with the rotation curve FROZEN instead of re-drawn:

| null variant | sd of Lambda-hat |
|---|---|
| full chain, RC re-drawn and baryons refit, 6 streams | **0.0408** |
| RC frozen, only stream noise varies, 6 streams | **0.0178** |
| full chain, 12 streams instead of 6 | **0.0683** |

**81 per cent of the null variance comes from the shared in-plane fit**, not
from the stream data. g_R and g_z share the baryon model, and forming the
ratio does NOT cancel that sharing, because the MOND response is nonlinear in
|g_N|. And the error does not fall when the sample is doubled -- it rises.
**The A_dyn floor for this method is set by the in-plane leg's own
uncertainty, and more off-plane tracers do not lower it.** That is the single
most useful number this lane produces: it says where to spend effort next.

**Calibration hygiene.** Three disjoint simulation sets were used, as Stage 0
requires. The empirical 2.5/97.5 percentiles of the 12-realisation calibration
set put **58 per cent of the audit set outside its own 95 per cent interval**
-- an artefact of estimating tail percentiles from 12 draws, not a real
calibration failure: the Gaussian interval from the same set puts 16.7 per
cent (2 of 12) outside, and the two sets are consistent (KS p = 0.256).
**The 5 per cent false-positive rate is NOT certified here.** Doing so needs of
order 200 realisations per set; 12 is enough for the sd, not for the critical
value.

---

## 6. The other three systems

### NGC 4651, the Umbrella Galaxy -- the cleanest control anywhere, and it has no power

45 Keck/DEIMOS tracers pass the velocity gate: 30 disc, 15 halo, from the same
instrument and the same calibration, so those systematics cancel between the
legs exactly. The in-plane leg is a single deprojected amplitude
(215 +/- 10 km/s at i = 53 deg), not a curve. The umbrella is shell debris
from a radial merger, so it was forward-modelled the same way as a Milky Way
stream, with the merger orbit (apocentre, angular momentum, three viewing
angles), the satellite mass and the merger age as six nuisance parameters
profiled over 256 draws at every Lambda.

| law | Delta-chi2 range over Lambda in [0.6, 1.7] | Lambda preferred |
|---|---|---|
| Newton | 18.8 | 0.60 (grid edge) |
| RAR | 14.3 | 0.84 |
| AQUAL | 39.6 | 1.42 |
| tidal scalar | 16.5 | 1.70 (grid edge) |
| well-network tensor | 18.3 | 0.99 |

The five laws span the entire grid, 0.60 to 1.70, on the same 15 tracers.
**The spread across laws is the systematic**: with only projected positions
and line-of-sight velocities, and no distance along the line of sight, the six
merger nuisances absorb Lambda completely. The apparently tight intervals are
profile minima over a fixed random draw set and are not confidence intervals.
**NGC 4651 constrains A_dyn not at all**, and the reason is the missing
observable (line-of-sight distance), not the sample size.

### M31 -- the only external galaxy with both legs, same verdict

98 of 100 Chemin 2009 HI tilted rings pass the gate (in-plane RMS 22-23 km/s
for the MOND family, 52.4 km/s for Newton) against 115 Chapman 2008 stream
A-D stars (4 A, 12 B, 64 C, 35 D), all with heliocentric radial velocities.

**A silent extraction failure was caught here and is worth recording.** The
VizieR table gives RA and Dec sexagesimally with SPACE separators; a
colon-based parser returns NaN for every row and the entire M31 stream sample
vanishes with no error and no exception -- the first pass of this lane
reported `n_stream_stars = 0`. The row count is now asserted (115) and the
parser accepts both separators.

| law | Delta-chi2 range | Lambda preferred |
|---|---|---|
| Newton | 131.4 | 1.49 |
| RAR | 109.7 | 0.73 |
| AQUAL | 187.3 | 1.16 |
| tidal scalar | 105.7 | 0.74 |
| well-network tensor | 178.0 | 1.06 |

Again a factor-two spread across laws on identical data. **M31 does not
constrain A_dyn either.** The geometry the brief identifies is real -- streams
A-D at a median 83.2 degrees from the disc major axis is exactly what the test
wants -- but line-of-sight velocity alone, with no distance and no proper
motion, leaves the merger orbit free.

### Pooled warped discs -- a 3.8-sigma orientation effect that A_dyn cannot produce

15 galaxies from Verheijen & Sancisi 2001 with i(R), PA(R) and V(R) all
tabulated, 80 warped rings across 13 usable galaxies. `g_bar` for a warped
source is direction-dependent, so it is NOT taken from the axisymmetric
solver: it is integrated exactly over the tabulated tilted-ring geometry as a
sum of circular wires, with per-annulus softening
`eps^2 = (0.5 da)^2 + h_z^2`. Gate: the wire sum reproduces the exact Freeman
disc to 5.1 per cent at 2 kpc and 0.3-1.3 per cent at 5-20 kpc, converging
(n_src 80 -> 160 moves it by less than 0.5 per cent); the residual is the
finite thickness, and it is absorbed by the per-galaxy intercept.

The ring condition is exact, not a small-angle expansion: for a ring of radius
r tilted by psi, a point at ring azimuth phi sits at
`R = r sqrt(cos^2 phi + sin^2 phi cos^2 psi)`, `z = r sin phi sin psi`, and
the inward force along the ring radius is `g_R (R/r) + g_z (z/r)`, averaged
over phi. At psi = 0 it reduces to `V^2 = r g_R(r)` exactly.

**Run W's degeneracy is confirmed and it is worse than reported.** The
Spearman correlation between ring orientation and radius WITHIN a galaxy has a
median of **+0.939** across these 15 galaxies (range +0.674 to +0.992),
against Run W's +0.904 on NGC 2685. Pooling drops it to +0.707, and the
estimator additionally carries an explicit `b ln R` term so that a radial
trend cannot be read as a direction effect.

Two distinct regressions were run, because they are not the same question --
and putting the `sin^2 psi` column into the Lambda scan is a design bug that
this lane made and then fixed: the free coefficient absorbs exactly the
psi-dependence Lambda produces, and chi2(Lambda) comes out flat by
construction (measured: Delta-chi2 range 0.07 with the column in).

| law | c (sin^2 psi coefficient) at Lambda = 1 | significance | b ln R | Delta-chi2 from adding psi | chi2/dof | Delta-chi2 range over Lambda in [0.5, 2] |
|---|---|---|---|---|---|---|
| Newton | +1.018 +/- 0.218 | 4.7 sigma | +0.342 | 21.8 | 1.37 | 4.50 |
| RAR | +0.839 +/- 0.218 | 3.8 sigma | -0.060 | 14.8 | 1.14 | 0.67 (edge) |
| AQUAL | +0.836 +/- 0.218 | 3.8 sigma | -0.056 | 14.7 | 1.13 | 0.31 (edge) |

Two things follow.

1. **There is a real, pooled, 3.8-sigma residual that tracks ring
   orientation**, with the per-galaxy intercept and the common radial trend
   already removed, and chi2/dof of 1.13 says the errors are about right.
   Warped rings rotate faster than the frozen law predicts by a factor
   exp(0.84 sin^2 psi) -- 16 per cent at psi = 25 degrees.
2. **A_dyn cannot be what it is.** Over the whole deformation range Lambda in
   [0.5, 2.0] the chi-square moves by 0.67 (RAR) and 0.31 (AQUAL), and the
   preferred Lambda sits at the grid edge. **The pooled warp lane has no power
   on A_dyn**: warps live at large radius where the vertical frequency
   approaches the spherical value, and most of the 80 rings have psi below 10
   degrees, so sin^2 psi < 0.03.

The most likely origin of `c` is not gravity. A tilted-ring fit derives
`V_rot = V_los / sin i`, so any systematic underestimate of the inclination of
a warped ring inflates V exactly as observed; warped outer rings are also not
guaranteed to be in circular equilibrium. The effect is recorded because it is
measurable, pooled, and survives the Run W protection -- not because it is a
gravity result. `b ln R` is +0.34 for Newton and -0.06 for the MOND family, an
independent confirmation that the frozen Newton model fails radially where the
MOND family does not.

### Polar rings

Not re-searched, per instruction. Run W established that no polar-ring galaxy
anywhere has a numerically tabulated rotation curve in both planes across 59
arXiv sources and 8 VizieR tables. NGC 4650A was not used, so its 58 per cent
polar-stellar-disc mass discrepancy does not enter anything here.

---

## 7. Failure modes from the brief: what was checked

* **Streams are not force samples.** Never treated as such. Every tracer in
  this lane -- Milky Way streams, the NGC 4651 umbrella, the M31 streams -- is
  generated as a progenitor orbit plus a stripping history plus a phase-space
  distribution under each candidate potential, and compared only in the
  observables the catalogue holds. No g_R or g_z point is ever extracted from
  a track.
* **Shared-quantity null with the actual error covariance.** Done, section 5:
  81 per cent of the null variance is the shared in-plane fit, and the naive
  expectation that the ratio cancels it is wrong because the response is
  nonlinear.
* **Responsiveness gate with the spread printed.** Done, section 5: A_dyn
  0.35-2.16, dA_dyn/dLambda 1.25-1.45, chi-square spread 1.2e5-2.6e5.
* **Physical-plausibility gate on every velocity.** Done, and it found a fifth
  defect family the brief did not list (section 3), plus a sixth problem in
  the InfoFlags decode.
* **Refitting on the held-out set.** Structurally impossible here: gravity
  constants are frozen from the tournament, baryon masses are calibrated on
  the rotation curve alone, and the streams touch only Lambda.
* **Silent extraction failures.** Row and column counts asserted on every
  ingest. One was caught: the M31 sexagesimal parse (section 6).
* **Test bugs that look like solver bugs.** Two were found and fixed here.
  Point-sampled density loses 27 per cent of the disc mass with no visible
  symptom; and a fixed wire softening makes the warped-disc integration
  diverge from Freeman as the annulus count RISES, because the softening has
  to scale with the annulus width.
* **Monotone-invariant statistics.** Checked; neither headline statistic is
  flat in its own parameter.
* **Programme-level multiplicity.** This lane reports a null. Nothing here
  needs a look-elsewhere correction, and nothing here should be read as
  supporting any positive result elsewhere.

---

## 8. What could NOT be established

* **Whether gravity is directional at the level any candidate predicts.** The
  predictions span 0.010 in A_dyn; the measurement floor is 0.11 (robust
  statistical) with a 0.038-0.093 systematic budget. The gap is a factor of 10
  to 30.
* **A calibrated false-positive rate.** 12 realisations per simulation set
  give the null sd but not the critical value; about 200 are needed.
* **Any constraint at all from NGC 4651, M31, or the pooled warps.** All three
  return law-dependent answers spanning the full search grid, which is the
  signature of no information rather than of a measurement.
* **The origin of the 3.8-sigma warp orientation residual.** It is real in the
  data as reduced; it is not reproducible by A_dyn; the inclination-bias
  explanation is plausible but untested here.

---

## 9. What this lane says to do next

The binding constraint is **not** the number of streams -- doubling the sample
raised the error. It is:

1. **The in-plane leg.** 81 per cent of the null variance enters through the
   rotation curve's covariance propagating into the frozen baryon model.
   A_dyn cannot be measured better than the in-plane leg is known, and the
   ratio does not cancel it because the response is nonlinear. A better v_c(R)
   with a published covariance matrix, or a baryon model with the disc scale
   length externally fixed, is worth more than any number of new tracers.
2. **The forward model.** chi2/n = 22 and a factor 1400-3400 inconsistency
   between streams says that a Fardal spray from a track-midpoint anchor with
   a uniform 2-4 Gyr stripping history is not good enough. Identified
   progenitors, a stripping rate tied to the orbit, and a treatment of the
   perturbers (the LMC above all) are prerequisites, not refinements.
3. **The observable that is missing everywhere else.** NGC 4651 and M31 fail
   for the same reason: no line-of-sight distance to the debris. Anything that
   supplies it -- tip-of-the-red-giant-branch distances to individual shells --
   converts both from zero-power to real tests.
"""

if __name__ == "__main__":
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPORT.md")
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(TEXT)
    print("wrote", p, len(TEXT), "chars")
