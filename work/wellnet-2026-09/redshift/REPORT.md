# redshift lane: does redshift depend on the geometry of the photon path?

## Bottom line

**The time-dilation constraint kills the mechanism in its natural form, so
everything that follows is a bounded feasibility study.** DES-SN5YR measures the
time-dilation exponent as `b = 1.003 +/- 0.011`; any path-redshift mechanism that
drains photon energy without acting on the metric predicts `b = 0`, which is
**90 sigma away**, and survives only as a sub-1.9% contamination of `ln(1+z)`
(`c2/c1 < 3.4%` at 2 sigma). The single surviving class, a genuinely geometric
path stretch, is untouched by that test but capped near **0.3 to 0.4% in c2/c1**
by the smoothness of the CMB across the local void structure. Both external
bounds are tighter than this dataset's honest sensitivity of **7 to 10%**. The
fit was run regardless and finds nothing: on the primary arm (SDSS VAST, n =
20,683) `c2/c1 = -1.17% +/- 1.45% (stat) +/- 1.40% (sys)`, **-0.58 sigma**, with
no term of the six-term law exceeding 1.8 sigma against its simulated null.

## 0. Independence, stated before anything else

**This is a logically independent hypothesis.** Nothing in the galaxy or cluster
gravity lanes is evidence for it. Nothing established here bears on that work in
either direction. The two branches must not borrow credit from each other, and
this lane shares no data, no fit, no calibration and no model-selection step
with them. KiDS and the wide binaries were never loaded, referenced or looked at.

---

## 1. THE TIME-DILATION VERDICT, which is the answer

**The mechanism is dead in its natural form, so the fit that follows is a
bounded feasibility study, not a discovery test.**

DES-SN5YR measures the time-dilation exponent to **1.1%**, and any path-redshift
mechanism that drains photon energy without acting on the metric predicts a value
**90 sigma away** from it. Such a mechanism survives only as a contamination of at
most **1.9% of ln(1+z)** (2 sigma), i.e. **c2/c1 < 3.4%**, comparable to the best
*statistical* reach of the entire void dataset and about an order of magnitude
below its realistic systematic floor. The one class the time-dilation test cannot
touch, a genuinely geometric path stretch, is separately capped near **0.3 to
0.4% in c2/c1** by the smoothness of the CMB across the local void structure.
**Both external bounds are tighter than anything this dataset can deliver.** The
fit was run anyway and is reported in section 2 with its null subtracted and its
power stated.

### 1.1 What was acquired

| file | source | SHA-256 (first 16) | bytes |
|---|---|---|---|
| `raw/arxiv_2406.05050_abs.html` | `https://export.arxiv.org/abs/2406.05050` | `d63138f16a1cc00f` | 54,919 |
| `raw/arxiv_api_2406.05050.xml` | arXiv API, `id_list=2406.05050` | `07c4d0e2a6fb19ae` | 11,798 |
| `raw/arxiv_2306.04053_abs.html` | `https://export.arxiv.org/abs/2306.04053` | `0bcd14cdaedc0568` | 41,910 |

Full manifests with retrieval timestamp and the exact query are in
`manifests/*.manifest.json`. Against the silent-extraction failure mode the
downloaded pages were asserted to contain the identifiers `2406.05050`, `1504`,
`time dilation`, `1.003`, `0.005` and `0.010` before any number was used; all six
echoed back. Only abstract pages and the API record were retrieved; no per-bin or
per-band table was obtained, which is a stated limit in section 5.

### 1.2 The measurement and its precision

**White R. M. T. et al. 2024**, *The Dark Energy Survey Supernova Program: slow
supernovae show cosmological time dilation out to z ~ 1*, MNRAS **533**, 3365,
arXiv:2406.05050, doi:10.1093/mnras/stae2008.

* 1504 DES type-Ia supernovae, 0.1 < z < 1.2
* parameterisation `dt_obs = dt_em (1+z)^b`
* **b = 1.003 +/- 0.005 (stat) +/- 0.010 (sys)**, total **sigma_b = 0.0112**

The systematic dominates, so the constraint is only as good as the DES treatment
of intrinsic light-curve-width evolution and of selection on stretch. What makes
`b` the right quantity is that it is the ratio of the observed **duration**
stretch to the observed **wavelength** stretch of the same objects: no distance,
no H0 and no cosmology enters it.

Corroboration from a population with entirely different systematics:
**Lewis G. F. and Brewer B. J. 2023**, Nature Astronomy **7**, 1265,
arXiv:2306.04053. 190 quasars to z = 4 modelled as damped random walks give
**n = 1.28 +/- 0.29**, excluding n = 0 at 4.4 sigma.

### 1.3 The decomposition that turns b into a bound on c2

Split the observed redshift factor multiplicatively and let the path term stretch
durations with efficiency eta:

```
(1+z) = (1+z_exp)(1+z_path),   dt_obs/dt_em = (1+z_exp)(1+z_path)^eta
  =>   b = 1 - f (1 - eta),    f = ln(1+z_path) / ln(1+z)
```

* **eta = 1**: the mechanism is geometric, acting on the null-geodesic affine
  structure, so frequency and time stretch together. Then `b = 1` for *any* `f`
  and the time-dilation test has **zero power** by construction.
* **eta = 0**: the mechanism drains photon energy but leaves the arrival-time
  spacing of successive photons alone. Then `b = 1 - f`.

The measurement therefore constrains **only the product f(1-eta) = 1 - b**:

| confidence | upper bound on f(1-eta) |
|---|---|
| 95% one-sided | 0.0154 |
| 2 sigma | 0.0194 |
| 3 sigma | 0.0305 |

Over a long sight line the void path length is the volume filling fraction times
the distance, `I_q -> F_v D`, so `f = (c2/c1) F_v / (1 + (c2/c1) F_v)`. With the
measured path-averaged void fractions (0.495 DESIVAST VoidFinder, 0.523 REVOLVER,
0.588 SDSS VAST):

| confidence | c2/c1 upper bound, eta = 0 |
|---|---|
| 95% one-sided | 2.7 to 3.2% |
| 2 sigma | 3.4 to 4.0% |
| 3 sigma | 5.4 to 6.4% |

### 1.4 Verdict on every candidate mechanism, reached before any fit

| mechanism | eta | verdict |
|---|---|---|
| **M1** photon energy drain / tired light: drag, secular loss to a medium or a background field, photon decay | 0 | **EXCLUDED** as the origin of cosmological redshift at 90 sigma. Survives only at f < 1.9%, i.e. c2/c1 < 3.4% (2 sigma). |
| **M2** scattering off a medium (Compton, Raman, plasma) | 0 | **EXCLUDED** on the same grounds, and independently by image sharpness: every known scattering channel randomises direction. |
| **M3** frequency-dependent energy loss | 0 | **EXCLUDED** on the same grounds, and additionally predicts band-dependent light-curve widths and distorted spectra. |
| **M4** geometric path stretch: the void contributes to the conformal factor along the null geodesic | 1 | **NOT constrained by time dilation.** `b = 1` identically. The only class the redshift regression can address. Capped near 0.3 to 0.4% in c2/c1 by the CMB gate below. |
| **M5** mixed or partially geometric | free | Constrained only in the product f(1-eta) < 0.019 (2 sigma). eta -> 1 escapes entirely. |

Two loopholes, stated rather than hidden:

1. **A mechanism confined to low redshift** evades the DES lever arm, because DES
   fits `b` globally over 0.1 < z < 1.2 and a term acting only below z ~ 0.25 is
   diluted in that fit. This loophole is closed by the CMB gate below, which is
   *entirely* a statement about local foreground structure.
2. **eta strictly between 0 and 1** is bounded only in the product with f.

### 1.5 A second gate that does reach eta = 1: CMB smoothness

Derived here, not taken from a publication, and flagged as such throughout. Any
achromatic path-dependent redshift also redshifts CMB photons, so a sky-to-sky
variation of the foreground void path length produces a temperature anisotropy

```
dT/T = -d ln(1+z) = -c2 dI_q
```

Using only the mapped volume is conservative: the unmapped remainder of the
14 Gpc path adds variance rather than cancelling it. Taking the whole observed
CMB rms (110 uK, dT/T = 4.0e-5) as an upper bound, with the measured transverse
spread of the void path length:

| catalogue | sd(dI_q) Mpc/h | implied bound on abs(c2)/c1 |
|---|---|---|
| DESIVAST VoidFinder | 35.1 | **0.35%** |
| DESIVAST REVOLVER | 43.7 | **0.28%** |
| SDSS VAST VoidFinder | 27.2 | **0.44%** |

Assumptions, all recorded in `timedilation.json`: the mechanism is achromatic and
acts on CMB photons as on supernova photons; the sky-to-sky spread of foreground
void path length is at least the spread measured inside the surveyed volume; no
cancellation against another term of the law; and the entire observed anisotropy
is generously attributed to the mechanism. The sharper version of this test
already exists in the literature as void-ISW stacking, which finds a few
micro-kelvin consistent with the general-relativistic integrated Sachs-Wolfe
prediction; it would tighten the bound by roughly the ratio of the CMB rms to the
stacked signal. **This gate applies to all eta, including eta = 1, and is the
binding constraint on the only surviving mechanism class.**

---

## 2. THE FIT

### 2.1 What was declared before any residual was looked at

* **Cuts**, inherited unchanged from the void-data lane which declared them in
  code before residuals: `path_covered_frac >= 0.5`, `r_end >= 100 Mpc/h`.
* **Regressor**: the transverse residual `dI_q = I_q - <I_q>(r)`, never raw
  `I_q`. Void-data finding 1 measured raw `I_q`'s null expectation at 27 to 40%
  of c1 at 30 to 38 sigma.
* **Tidal terms** `c3`, `c6` fitted only on watershed (REVOLVER) geometry.
* **Arms never averaged**: SDSS VAST below z = 0.11, DESIVAST above it.
* **Blind split**: frozen 50/50 by object, seed 20260904.
* **Normalisation**: every ratio is quoted against the *fiducial* c1 = H0/c =
  3.3356e-4 per Mpc/h, not against the fitted c1. The fitted c1 is attenuated to
  0.68 to 0.81 of fiducial by the errors-in-variables regression of a precise
  redshift on a noisy distance; it is not a measurement of H0 and dividing by it
  would inflate every ratio by 1/attenuation.

### 2.2 The arms, and a structural obstruction

| arm | n | geometry | note |
|---|---|---|---|
| **SDSS VAST VoidFinder, z < 0.11** | **20,683** | spheres | **PRIMARY**. Footprint 2.13 sr. No tidal terms possible (finding 3). |
| DESIVAST REVOLVER, all z | 4,389 | watershed | edge-limited below z = 0.11 (finding 4) |
| DESIVAST VoidFinder, all z | 4,389 | spheres | edge-limited |
| DESIVAST REVOLVER, z > 0.11 | **46** | watershed | the only footprint-safe DESI subset |
| DESIVAST density field | 4,389 | none | void-finder-free: regressor is the underdensity path integral of the reconstructed field |

**The structural obstruction.** Finding 3 forces the tidal terms onto watershed
geometry, which exists only in DESIVAST. Finding 4 forces DESIVAST above
z = 0.11, where **n = 46**, all of them Pantheon+ supernovae. The two findings
cannot be satisfied simultaneously with the data on disk, so **c3 and c6 have no
footprint-safe determination at all**. They are reported below from the
edge-limited full DESI arm and must be read as such.

### 2.3 The null model, and what each component contributes

Every component below is injected under the hypothesis `c2 = 0` and its
contribution to the estimator measured. Values are the null mean of the fitted
`c2`, in per cent of the fiducial `c1`.

| component | what it is | SDSS | REVOLVER | VoidFinder | density field |
|---|---|---|---|---|---|
| **N0** | shared-distance artefact: `I_q` knows the true distance, `D` is the noisy one | -1.17 | +3.95 | +0.79 | -0.16 |
| **N1** | reconstructed linear-theory peculiar velocities | +0.38 | +4.76 | +2.51 | +3.48 |
| **N2** | lensing magnification and demagnification along the sight line | -1.13 | +4.16 | +0.88 | -0.52 |
| **N3** | host-galaxy zero-point drift with the host's own environment | -1.29 | +3.85 | +0.65 | -0.02 |
| **N4** | calibration drift across the sky (dipole plus quadrupole in mu) | -1.18 | +4.25 | +0.91 | +0.27 |
| **N5** | survey selection, inhomogeneous Malmquist | -2.90 | +8.28 | +6.02 | +9.27 |
| **N6a** | covariance from using redshift to build the void catalogue | -0.81 | +4.42 | +1.10 | +2.40 |
| **FULL** | all of the above together | **-1.09** | **+9.07** | **+7.59** | **+14.28** |

Each column is the null with N0 always on plus the one named component, so the
increment over N0 is that component's contribution.

Notes on the individual components.

* **N0** reproduces the void-data lane's independent implementation to within
  0.6 percentage points (they measured +0.66% VoidFinder, +3.44% REVOLVER; this
  code gets +0.79% and +3.95% from a different construction of the regressor).
  That agreement is the cross-check that the null is implemented correctly.
* **N2 is negligible and now quantified.** The void-correlated convergence has
  sd(kappa) = 1.4e-4, so `d ln D = -kappa` is a 0.014% distance perturbation,
  three orders of magnitude below the 23.5% distance errors. The much larger
  total lensing dispersion quoted for supernovae (about 0.055 z mag) comes from
  small-scale structure that the 5 Mpc/h-smoothed field deliberately excludes,
  and which is uncorrelated with the void catalogue: it is noise, not bias.
* **N1 is real but modest**: the reconstructed radial velocity has sd 257 km/s
  and correlates with `dI_q` at +0.168, contributing +0.8 to +3.6 points.
* **N5 is the largest and the least certain.** It required two corrections that
  are stated rather than tuned. The first pass used the raw logarithmic density
  slope measured over a fixed 30 Mpc/h window; over so short a lever arm
  `d ln(1+delta)/d ln r` becomes a ratio of two small numbers and reached +/-8,
  which drove a spurious 10 to 30% null bias. The window is now the distance
  error itself, which is what the Malmquist bias actually samples, giving
  +0.30 in-void against +3.71 outside. The slope is also demeaned, because a
  constant rescales every distance alike and is absorbed by `c1`, and its
  residual amplitude is drawn uniformly in [0, 1] because CF4 and Pantheon+
  already apply their own bias corrections of unknown residual size. Keeping the
  first-pass treatment instead is recorded as `N5raw` in the JSON and moves the
  density-field arm's null from +9.3% to +24.5%. **That 15-point swing from one
  modelling choice is larger than any statistical error in this analysis.**

### 2.4 The result

All values in per cent of the fiducial `c1`. `R` is the exact response of the
fitted coefficient to an injected true `c2`, computed from the linearity of
weighted least squares rather than by a separate injection run.

| arm | n | R | raw c2 | full null | **estimate** | stat | sys | sigma (stat / full) | 95% limit |
|---|---|---|---|---|---|---|---|---|---|
| **SDSS VAST, z<0.11** | 20,683 | 0.997 | -2.26 | -1.09 | **-1.17** | 1.45 | 1.40 | -0.82 / **-0.58** | **< 5.11** |
| DESIVAST REVOLVER | 4,389 | 0.997 | +2.59 | +9.07 | -6.50 | 2.79 | 2.10 | -2.39 / -1.86 | < 13.36 |
| DESIVAST VoidFinder | 4,389 | 0.999 | -2.12 | +7.59 | -9.73 | 2.75 | 2.31 | -3.60 / -2.71 | < 16.76 |
| DESIVAST REVOLVER, z>0.11 | 46 | 1.000 | -1.26 | +0.68 | -1.94 | 2.43 | 1.85 | -0.85 / -0.64 | < 7.93 |
| DESIVAST density field | 4,389 | 0.975 | -3.02 | +14.28 | -17.74 | 5.52 | 2.30 | -3.29 / -2.97 | < 29.46 |

**The primary arm gives `c2/c1 = -1.17% +/- 1.45% (stat) +/- 1.40% (sys)`,
which is -0.58 sigma. There is no path-geometry term in the redshifts at the
precision this dataset can reach.**

**The arm-to-arm dispersion is the real error bar.** Five estimators of the same
coefficient, sharing the same sources and differing only in how the path integral
is built, give estimates spanning **-17.7% to -1.2%** with an sd of **6.74%**,
against a median quoted sigma of **3.50%** -- a dispersion 1.93 times the quoted
error. But the **raw** coefficients before null subtraction span only **-3.02% to
+2.59%**, sd **2.21%**. The data agree; the corrections do not. That locates the
disagreement squarely in the nuisance model, and it means the honest systematic
on `c2/c1` is of order **7 to 10%**, not the 2.8 to 4.3% statistical floor the
void-data lane quoted, and not the 1.4% systematic the Monte Carlo alone
suggests. This is the same conclusion void-data finding 4 reached from the
cross-pipeline transverse-residual correlation, which this lane reproduces at
**r = 0.149** on 2,141 common sight lines (implied relative systematic x6.6).

### 2.5 Power

Injection recovery on the primary arm, at a two-sided 3-sigma threshold
calibrated on the null itself. Recovery is exact by construction, so the
statistic carries no Monte Carlo noise.

| injected c2/c1 | recovered | S (stat null) | power (stat) | S (full null) | power (full) |
|---|---|---|---|---|---|
| 0.0% | +0.000% | 0.00 | 0.00 | 0.00 | 0.00 |
| 0.5% | +0.500% | 0.39 | 0.00 | 0.25 | 0.01 |
| 1.0% | +1.000% | 0.78 | 0.01 | 0.51 | 0.01 |
| 2.0% | +2.000% | 1.55 | 0.10 | 1.01 | 0.01 |
| 5.0% | +5.000% | 3.88 | 0.79 | 2.53 | 0.34 |
| 10.0% | +10.000% | 7.75 | 1.00 | 5.06 | 0.98 |

**Minimum detectable `c2/c1` at 3 sigma: 3.9% on statistics alone, 5.9% with the
declared systematics, and about 20 to 30% once the arm-to-arm dispersion is taken
as the error.** The monotone-invariance check passes with a computed derivative
`dS/dtheta` of **77.51** sigma per unit `c2/c1` (statistical null) and **50.57**
(full null), constant across the tested range by construction, with the statistic
spanning 0 to 7.75 sigma -- so the headline statistic demonstrably moves with the
parameter it is supposed to measure, which is the check a rank statistic once
failed silently across three decades.

The false-positive rate on the **untouched audit simulations**, at the critical
value fixed on the disjoint calibration set, is 0.053 to 0.075 across the five
arms against a nominal 0.05. The calibration is honest.

### 2.6 The decisive comparison, done by differencing

Regression is not the only way to ask the question, and the differenced version
is more transparent. Take pairs of sources matched to `|dD| < 20 Mpc/h` in the
independent distance with `|dI_q| > 60 Mpc/h`, and fit the slope of
`d ln(1+z)` on `dI_q`. This removes `c1` to first order.

| arm | pairs | slope / c1 | null | null-subtracted | sigma |
|---|---|---|---|---|---|
| SDSS VoidFinder | 20,547 | -1.85% | -1.91% | **+0.07%** | **+0.62** |
| DESIVAST REVOLVER | 4,062 | +6.95% | +7.19% | -0.24% | -1.86 |

Note that the differenced estimator has a null of its own: -1.91% and +7.19%.
Even after removing `c1` by differencing, matching on a *noisy* distance still
manufactures a path signal from nothing. That null must be simulated, not
assumed.

Why matching on the measured distance does not match the true distance, made
concrete on the primary arm: take pairs agreeing to better than **1 Mpc/h** in
the inferred distance. Their redshifts still scatter by **4,299 km/s** rms,
because a 23.5% distance error at 200 Mpc/h is about 50 Mpc/h and no amount of
matching in the measured variable fixes that. Against this, a `c2/c1 = 5%` path
effect across the rms `dI_q` of 38.5 Mpc/h would produce **193 km/s**. The
per-pair signal-to-noise is 0.045, which is exactly why the test needs 20,000
pairs and why its floor is a few per cent rather than a few tenths.

Two representative matched pairs, for illustration only:

```
PGC1973480  D = 209.33  dI_q = +80.25  z = 0.08043
PGC1100636  D = 209.44  dI_q = -131.41 z = 0.09637   ->  d(cz) = -4778 km/s
PGC158874   D = 276.02  dI_q = -123.46 z = 0.09670
PGC1995885  D = 276.28  dI_q = +79.03  z = 0.09567   ->  d(cz) =  +309 km/s
```

Opposite signs of `d(cz)` for the same sign of `dI_q`, with magnitudes 25 and 2
times the predicted 5% signal. Individual pairs carry no information.

### 2.7 The six-term law on watershed geometry

Fitted on DESIVAST REVOLVER, n = 4,389, with the path terms radially detrended.
Design conditioning is good, exactly as void-data finding 3 predicted for
watershed geometry: VIFs are `D` 1.01, `dI_q` 3.55, `dI_T` 4.08, `dI_g` 1.43,
`dI_q^2` 3.27, `dI_q*I_T` 3.99.

| term | coefficient | analytic sigma | naive sigma | null mean | null sd | **sigma vs null** |
|---|---|---|---|---|---|---|
| `c2` dI_q | +5.032e-05 | 8.30e-06 | **+6.06** | +1.166e-04 | 3.74e-05 | **-1.77** |
| `c3` dI_T | +6.700e+01 | 1.06e+03 | +0.06 | -2.033e+03 | 1.24e+03 | **+1.70** |
| `c4` dI_g | -4.376e+01 | 1.28e+01 | **-3.42** | -9.010e+01 | 4.28e+01 | **+1.08** |
| `c5` dI_q^2 | -2.032e-07 | 3.76e-08 | **-5.40** | -3.817e-07 | 1.16e-07 | **+1.54** |
| `c6` dI_q*I_T | +1.443e+01 | 1.28e+01 | +1.13 | -5.269e+00 | 2.25e+01 | **+0.88** |

**Nothing survives.** The instructive column is "naive sigma": read against the
ordinary analytic error bar, this table announces a 6.1 sigma detection of `c2`,
a 5.4 sigma detection of `c5` and a 3.4 sigma detection of `c4`. Against the
simulated null the same three numbers are 1.8, 1.5 and 1.1 sigma. That gap is the
whole content of the shared-denominator failure mode, reproduced here in a
setting where the answer is known to be zero.

`c4` is reported but **must not be interpreted**: the reconstructed density field
is zero outside the survey mask, so `I_g` is a lower bound and the least
trustworthy of the integrals. The void-data lane said not to fit it; it is fitted
here only to show that it too collapses against its null.

### 2.8 Blind protection

Frozen 50/50 split by object, seed 20260904, declared in code before residuals.
n_train = 10,281, n_holdout = 10,402. Coefficients fitted on train, **frozen**,
holdout touched exactly once.

* `c2/c1` on train: **-1.66%**
* frozen-coefficient transfer to the holdout: `delta chi2 = +3.48` on 10,402
  points, i.e. nothing
* the same holdout **refitted** would give `c2/c1 = -4.69%`

The refit value is printed only to expose the size of the mistake the programme
has been bitten by before: refitting on the held-out set would have moved the
answer by a factor of 2.8 in this case.

### 2.9 What the regressor actually measures -- a result that changes the reading

A check that had not been run before. Correlating the transverse void path
length against the reconstructed density along the same sight lines:

| catalogue | corr(dI_q, mean delta along LOS) | corr(dI_q, underdensity path integral) |
|---|---|---|
| DESIVAST VoidFinder | +0.069 | +0.130 |
| DESIVAST REVOLVER | **+0.319** | **-0.190** |

**A catalogued void path length is not an underdensity path length.** For the
sphere-based VoidFinder the two are nearly uncorrelated. For the REVOLVER
watershed the transverse residual correlates *positively* with the mean
line-of-sight **density**, and *negatively* with the genuine underdensity path
integral, because the watershed tiles the entire survey volume into zones and
then prunes, rather than selecting empty regions. Any `c2` fitted on a catalogue
`I_q` is therefore a coefficient of catalogue membership, not of emptiness. This
is a second, independent reason -- beyond the footprint-size argument of finding
4 -- why the two pipelines cannot be averaged, and it is why the void-finder-free
density-field arm was added.

---

## 3. CIRCULARITY, stated without softening

The void catalogue is a redshift-space product in Cartesian clothing, and this
analysis reuses it. Circularity enters in **four** places:

1. **Void positions** come from `r = D_C(z; Omega_m = 0.315)`. Every void
   coordinate in the catalogue is a redshift mapped through the very law under
   test.
2. **The sample definition** is volume-limited using `MAGLIM = -20`, which needs
   a cosmology-dependent luminosity distance. **Which galaxies exist in the
   sample depends on the law under test.**
3. **Voids are found in redshift space** and are RSD-stretched along the line of
   sight, uncorrected.
4. **The source endpoints** are placed by the same law.

Size of the effect, from `../void-data/robustness.json`. Relative to the
fiducial at z = 0.24, the alternatives displace the radial coordinate by
**+42.1 Mpc/h (+6.2%)** for linear `cz/H0`, **-32.5 (-4.8%)** for Milne, and
**-66.0 (-9.7%)** for Einstein-de Sitter. That is 1.5 to 4 void radii, larger
than a void. But the shift is **shared** by voids and sources, since both are
placed by redshift, so ordering along the ray survives and only the differential
stretch of 5 to 10% across 0 < z < 0.24 matters.

The half that does **not** cancel is the endpoint, and it was measured directly:
recomputing every `I_q` with the ray truncated at the source's own independent
distance times a single fitted global `h = 0.7431` gives a median `|dI_q|` of
**6.20 Mpc/h = 0.177 sd(dI_q)** -- an **18% perturbation on the leverage
variable**.

This lane adds a fifth item, quantified as N6a: **the source's own peculiar
velocity moves both its redshift and the truncation point of its own ray.** To
first order `dI_q = 1_void(endpoint) x v/H0` while `d ln(1+z) = v/c`, a
covariance between regressor and response that exists by construction. The
endpoint sits inside a catalogued void for 27% of SDSS sight lines, 18% of
DESIVAST VoidFinder and 49% of REVOLVER. Its contribution to the null is +0.3 to
+2.6 points of `c1` -- small, but it is not zero and it had not previously been
written down.

**Verdict, unsoftened. A genuine no-expansion analysis cannot reuse this
catalogue as it stands.** It would have to rerun VoidFinder and V2 under its own
distance law, which also changes the sample definition and therefore which
galaxies exist. Reuse costs about 18% on the leverage variable and 5 to 10% on
the radial metric. That is tolerable for a feasibility and power study. It is not
tolerable for a claimed detection, and no detection is claimed.

---

## 4. Failure-mode checklist, each checked explicitly

* **Shared-denominator artefacts -- CHECKED and FOUND, twice.** Once in the
  inherited form (raw `I_q` has a null of 27 to 40% of `c1` at 30 to 38 sigma)
  and once in a new form here: the six-term table reads 6.1, 5.4 and 3.4 sigma
  against analytic errors and 1.8, 1.5 and 1.1 sigma against the simulated null.
  Every number in this report is quoted against a simulated null, never against
  an analytic error bar.
* **Monotone-invariant statistics -- CHECKED.** `dS/dtheta` = 77.51 (statistical
  null) and 50.57 (full null) sigma per unit `c2/c1`, computed exactly; the
  statistic spans 0 to 7.75 sigma over the tested range.
* **Refitting on the held-out set -- CHECKED and AVOIDED.** Fitted on train,
  frozen, holdout touched once. The frozen transfer gives `delta chi2 = +3.48`;
  the incorrect refit would have given a coefficient 2.8 times larger.
* **Silent extraction failures -- CHECKED.** Every downloaded page was asserted
  to echo back its identifiers before use; row counts asserted (DESI 4,389, SDSS
  25,123) and source names matched element-by-element between the path-integral
  tables and the nuisance tables.
* **Test bugs that look like solver bugs -- CHECKED and FOUND ONE.** The lensing
  convergence was validated against the reconstructed density it integrates
  (corr(kappa, mean delta along LOS) = +0.77, exactly as it must be). The
  Malmquist log-slope was **wrong on the first pass** -- measured over a 30 Mpc/h
  window it reached +/-8 and drove a spurious 10 to 30% null bias. Diagnosis:
  too short a lever arm in `ln r`, not a physical gradient. Both the first-pass
  and corrected treatments are reported.
* **Detector calibration -- CHECKED on three disjoint simulation sets.**
  Calibration sets the critical value, an untouched audit set verifies the false
  positive rate (0.053 to 0.075 against nominal 0.05), and injection measures
  power.
* **Programme-level multiplicity.** This lane reports a null and an upper limit,
  so multiplicity does not inflate anything. Had a positive appeared, it would
  have needed calibration against the whole programme's adaptivity, not this
  lane's alone.
* **Sealed holdouts -- KiDS and the wide binaries were never loaded, referenced
  or looked at.**

---

## 5. What could NOT be established

* **Whether a geometric (eta = 1) path term exists below the external bounds.**
  The CMB gate allows up to 0.3 to 0.4% in `c2/c1`; this dataset's honest
  sensitivity is 7 to 10%. There is a factor of 20 to 30 between them and no way
  to close it with the data on disk.
* **`c3` and `c6` have no footprint-safe determination at all.** Finding 3 puts
  the tidal terms on watershed geometry, finding 4 puts watershed geometry above
  z = 0.11, and there n = 46. The values quoted in section 2.7 come from the
  edge-limited sample and should not be propagated.
* **`c4`** cannot be interpreted, because `I_g` is a lower bound outside the
  survey mask.
* **The absolute size of the inhomogeneous Malmquist residual in CF4.** It is the
  largest single term in the null and its amplitude was modelled as unknown in
  [0, 1] rather than measured. Pinning it down would require CF4's own selection
  function, which was not acquired.
* **Whether the cross-pipeline disagreement (r = 0.149) is entirely footprint
  size.** The evidence points there strongly, but VoidFinder was not rebuilt on a
  footprint-matched sample, so the systematic floor remains empirical.
* **A siren or megamaser arm.** n = 0 in the DESI footprint, n = 3 in SDSS. The
  right shape of measurement exists (NGC 6323 at r = 76.8 with `I_q` = 8.1
  against NGC 5765b at r = 82.4 with `I_q` = 51.0) at n = 2 and 9 to 21%
  distance errors.
* **Per-band and per-redshift-bin time-dilation values.** Only the global
  `b = 1.003 +/- 0.005 +/- 0.010` was acquired, from the abstract page and the
  API record. A redshift-resolved `b(z)` would close the low-redshift loophole in
  section 1.4 directly instead of via the CMB gate.
* **The CMB gate as a real measurement.** It is an order-of-magnitude bound from
  the total anisotropy, not a cross-correlation. Cross-correlating the void
  path-length map against a CMB temperature map would sharpen it by roughly the
  ratio of the CMB rms to the stacked void-ISW signal, and that measurement is
  already routine in the literature.

---

## 6. What would actually move this

In descending order of return per unit effort.

1. **Cross-correlate the void path-length map with Planck.** Days of work, and it
   turns the 0.3 to 0.4% order-of-magnitude gate into a real measurement roughly
   an order of magnitude tighter. It is the only step here that reaches the one
   surviving mechanism class.
2. **A redshift-resolved `b(z)` from DES.** Closes the low-z loophole without any
   new modelling.
3. **Rerun VoidFinder and V2 under the alternative distance law**, with the
   sample definition recomputed. Until this is done no result from this data can
   be called a test of a no-expansion cosmology rather than a consistency check
   inside LCDM.
4. **DESI DR2.** The footprint-size argument predicts the edge limitation
   relaxes; that is a falsifiable prediction of the diagnosis in finding 4.
5. Distances better than 23.5%. Everything here is limited by CF4's error, not by
   the redshifts and not by the void catalogues.

---

## 7. Files

| file | what |
|---|---|
| `timedilation.py` | the time-dilation gate: acquisition, the (f, eta) decomposition, the mechanism verdicts, the CMB gate |
| `timedilation.json` | machine-readable output of the above |
| `redshift_test.py` | nuisance construction, the five arms, the null model, the fits, matched pairs, blind split, injection, systematic floor |
| `redshift_results.json` | machine-readable output of the above |
| `nuisance_desi_v2.csv` | per-source linear-theory peculiar velocity, lensing convergence, Malmquist log-slope, endpoint-in-void indicators (DESI, 4,389 rows) |
| `nuisance_sdss.csv` | endpoint-in-void indicator and two-phase convergence (SDSS, 25,123 rows) |
| `raw/`, `manifests/` | three retrieved documents with SHA-256, byte size, HTTP status, timestamp and the exact query |
| `run_timedilation.log`, `run_redshift_test.log` | full console output of both runs |

Inputs consumed read-only from `../void-data/`: `path_integrals_analysed.csv`
(4,389 x 47), `path_integrals_sdss.csv` (25,123 x 20), `results.json`,
`robustness.json`, `raw/desivast/` and the eleven modules in `code/`. Nothing in
the void-data lane was modified.
