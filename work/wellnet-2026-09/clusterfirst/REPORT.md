# Run BT — gravity from where the baryons actually are

Lane `work/wellnet-2026-09/clusterfirst/`.

## What was run

Eleven clusters carry all three channels at once, which is what makes this
possible and why it has waited:

- **ACCEPT** deprojected, calibrated electron density `n_e(r)` from Chandra
- deep **Chandra** event lists (Run BR)
- public per-source **DECADE** weak-lensing shear, from the OPEN half of the holdout

One of them is 1E0657-56, the Bullet Cluster.

The baryons are turned into a gravitational field with no fitted halo, no mass
model and no scaling relation:

```
rho_gas(r)        = 1.15 m_p n_e(r)
M_gas(<r)         = ∫ 4π r² rho_gas dr
g_bar(r)          = G M_gas(<r) / r²
Sigma_gas(R)      = Abel projection of rho_gas
DeltaSigma_bar(R) = mean Sigma inside R − Sigma at R
```

and `DeltaSigma_bar` is compared directly against `DeltaSigma_obs` from the
shear. The comparison happens **in projection**, where lensing actually
measures, so no new deprojection assumption enters — the only spherical
assumption is the one ACCEPT already made, and re-projecting it is
self-consistent with that.

## The result

| law | median predicted / observed | scatter |
|---|---|---|
| Newton (baryons only) | **0.148** | 0.69 dex |
| RAR (McGaugh+2016) | **0.456** | 0.65 dex |
| MOND simple ν | 0.459 | 0.65 dex |
| MOND standard ν | 0.393 | 0.65 dex |

Read directly: **the observed lensing is about 7× what the observed baryons
produce under Newtonian gravity, and about 2.2× what the RAR predicts.**

That is the textbook cluster discrepancy, and it is not new — MOND's residual
factor of roughly two in clusters has been known since the 1990s. What is new
here is only that it reproduces with a lensing dataset that has never been used
for this: DECADE metacalibration shapes, an independent survey and an
independent reduction pipeline from anything in the existing literature on these
objects.

Stars are missing from `g_bar` — hot gas is 85–90% of cluster baryons and no
stellar profile exists for these eleven. That biases `g_bar` low by 10–15%,
which makes the discrepancy slightly *worse* than quoted. The bias is
conservative with respect to the headline.

## Why this configuration cannot measure it precisely

The honest limitation, and it is severe.

**The calibrated gas and the lensing barely overlap in radius.** ACCEPT profiles
stop between 0.34 and 1.18 Mpc; the shear profile runs from 0.29 to 4.4 Mpc. So
of ten lensing bins per cluster, **one to three** fall inside the measured gas.
Two clusters have **none**.

| cluster | gas measured to | lensing bins inside | obs / baryon |
|---|---|---|---|
| 1eRASS J065829.9−555637 (Bullet) | 1.18 Mpc | 3 | 1.2 |
| 1eRASS J134730.8−114510 | 0.99 Mpc | 2 | 9.7 |
| 1eRASS J052042.5−132848 | 1.00 Mpc | 2 | 13.0 |
| 1eRASS J063846.8−535831 | 0.49 Mpc | 2 | 4.2 |
| 1eRASS J043900.7+071603 | 0.77 Mpc | 2 | 11.8 |
| 1eRASS J123625.2+163246 | 0.42 Mpc | 1 | 20.5 |
| 1eRASS J102339.7+041108 | 0.38 Mpc | 1 | 9.1 |
| 1eRASS J111320.3+173542 | 0.34 Mpc | 1 | 1.9 |
| 1eRASS J140102.0+025240 | 0.59 Mpc | 1 | 1.5 |
| 1eRASS J045410.6−030056 | 0.77 Mpc | **0** | — |
| 1eRASS J044309.7+021017 | 0.35 Mpc | **0** | — |

**About fifteen usable data points in total**, and the per-cluster ratios span a
factor of seventeen. That is the 0.65 dex scatter, and it is dominated by shape
noise in single lensing bins at the smallest radii, where the shear
signal-to-noise is at its worst. The Bullet's 1.2 is not a measurement that its
lensing matches its baryons; it is one of the noisiest entries in the table,
with the fewest background sources of any cluster in the set.

## The extrapolation that was refused

A first version of this ran without a radial cut, letting the gas density
continue as a fitted power law out to the lensing radii. The mass supplied by
that extrapolation was **3× to 140× larger than anything ACCEPT measured** —
2887%, 1043%, 1726%, 13813% of the measured mass, cluster by cluster.

The prediction would then have been a statement about the fit, not about the
baryons, and it would have looked entirely reasonable: the un-cut run gave
Newton at 0.145 and the RAR at 0.765, with a *tighter* 0.48 dex scatter than the
honest version. Better-looking numbers, produced by extrapolating a density
profile a decade in radius past its data.

`temperature_support` v2 already forbids exactly this — fail closed, never clamp
silently, always print the extrapolated fraction. The rule was written for
temperature and applies unchanged to density. Every bin outside the measured
range is now dropped, and the surviving count is reported per cluster above.

## What would make this a measurement

In order of leverage:

1. **Gas profiles that reach the lensing radii.** ACCEPT is an inner-region
   product. X-COP, CHEX-MATE or a dedicated re-analysis of the Chandra events
   already in `clusterxray/raw` would extend `n_e(r)` outward and turn one bin
   per cluster into six.
2. **More clusters with both channels.** Eleven is what the current archive
   overlap allows; the number grows with any new public shear release.
3. **Strong lensing at small radii**, which measures where the gas actually is
   rather than where the weak lensing is best.

None of those is compute. All three are acquisition.

## Provenance

`eRASS1` `M500`, `R500` and `Mgas500` are **excluded throughout**: they come
from eROCOP, which is weak-lensing calibrated, and the aperture is circular even
where the measurement is not — the identity that killed "organised by r/R500"
(`failures/artefacts/08-r500-is-r.md`). Only fixed-physical-aperture X-ray
quantities and the deprojected profile are admissible.

Sealed half never queried: 304 sealed, 0 tokens spent.

---

# Run BU — what the factor of two is a function of

`extend_gas.py` turned the ACCEPT profiles into beta-model densities reaching
0.94–4.6 Mpc, anchored to ACCEPT where the two overlap (0.04–0.19 dex agreement,
21–59 anchor bins each). One cluster was **rejected**: the optimiser found a
corner with `Rc = 0 kpc` and `n0 = 1.4e5 cm⁻³`, five orders of magnitude above
any cluster core — a fit that reproduces the surface brightness and is
physically impossible. Ten survive.

That took the test from 15 usable points to **65 rows over 10 clusters**, and
the residual sharpened:

| | median predicted/observed | scatter |
|---|---|---|
| plain RAR, Run BT (15 pts) | 0.456 | 0.65 dex |
| plain RAR, extended gas (55 pts) | **0.620** | **0.52 dex** |

## The families, and the discriminator

Each family multiplies the RAR prediction and carries one free parameter. The
question is not whether it can fix the median — anything with an amplitude can —
but whether it **reduces the scatter**, which is what a real dependence does.

| family | best p | median | scatter | verdict |
|---|---|---|---|---|
| plain RAR | — | 0.620 | **0.52** | the residual |
| slip (constant) | 1.59 | 0.985 | 0.52 | rescales only |
| a0 to f*a0 | x2.98 | 1.006 | 0.52 | rescales only |
| density^p | 0.30 | 0.976 | 0.53 | rescales only |
| temperature^p | 2.40 | 0.924 | 0.54 | rescales only |
| radius^p | 0.60 | 0.883 | 0.57 | no gain beyond null |
| redshift (1+z)^p | 2.10 | 0.995 | 0.53 | rescales only |
| gas mass^p | 0.20 | 1.037 | 0.55 | rescales only |

**Not one family reduces the scatter below 0.52 dex.** Every one drives the
median to within a few percent of unity and leaves the spread exactly where it
was, or makes it worse. Each was refitted 300 times with its variable permuted
across clusters; none beats its own null.

## What that means

**The cluster residual is an amplitude, not a dependence.**

Whatever is missing does not track the local gas density, the temperature, the
position in the cluster, the redshift, or how much gas there is. Those were the
axes this programme has spent eighteen months exploring, and each can absorb the
factor into its normalisation — which is exactly what a variable that carries no
information looks like when you give it a free exponent.

Two of the seven are honest about being amplitudes by construction: a constant
slip of 1.59, and an acceleration scale about 3x larger in clusters than in
galaxies. Both fit perfectly and neither explains anything, because a single
number fitted to a single number always will.

That is consistent with a roughly universal missing-mass fraction — which is
what dark matter with a near-constant baryon fraction looks like — and it is
inconsistent with any of the environmental modifications tested here.

## The limit that now binds

Scatter, 0.52 dex, a factor of 3.3. With 55 points a one-parameter family can
only move the mean, and a dependence would have to be strong to show through
that spread. The scatter is dominated by shape noise: these clusters carry
6,974–25,199 background sources each, split across ten radial bins.

The vignetting model is the other systematic. Without CIAO exposure maps an
analytic ACIS approximation is used, and it biases the outer surface brightness,
hence beta, hence the outer density. It is stated in `extend_gas.py` rather than
hidden, and it is the first thing to fix if this is pursued.

---

# Run BV — RETRACTION: the factor of two was substantially my own estimator

Runs BT and BU reported that the RAR under-predicts cluster lensing by 1.6–2.2×.
**That number was wrong, and the error was in the estimator, not the data.**

## Two compounding mistakes

**1. Selection on the noisy quantity.** Both runs filtered `ds_obs > 0` before
taking the ratio. Ten of the 65 rows have a negative observed lensing signal —
mean −0.53σ, i.e. ordinary downward noise fluctuations on a positive quantity
measured at low signal-to-noise. Dropping them removes *only* downward
fluctuations and keeps every upward one, which inflates the apparent observed
lensing and therefore inflates the apparent missing gravity.

**2. Median of a ratio with a noisy denominator.** With 45 of 55 points carrying
more than 50% fractional error, the median of `pred/obs` is not an estimate of
the ratio of the truths.

| estimator | residual (pred/obs) |
|---|---|
| median, with `ds_obs > 0` filter (Runs BT, BU) | **0.620** |
| inverse-variance weighted mean, no filter | **0.914 ± 0.142** |

Stated the other way: **observed / RAR-predicted = 1.09 ± 0.14.**

## What Run BU's conclusion was actually worth

Nothing. Decomposing its 0.523 dex scatter:

| | |
|---|---|
| observed scatter | 0.523 dex |
| median per-point measurement error | 0.365 dex |
| points above 50% fractional error | 45 of 55 |
| points below 20% | **0** |
| **intrinsic scatter** | **0.000 dex** |

The spread was entirely measurement error. No modification could have reduced
it, whatever the physics. "No family reduces the scatter" was guaranteed before
any physics entered — a **vacuous control**, in the same class as Run AY's
"baryon-only control was VACUOUS, not passed".

## The stacked test, which does have power

Binning 65 points into four bins per variable drops the per-bin error to
0.2–0.5 and lets the residual be asked whether it *moves*:

| binned by | residual across bins | verdict |
|---|---|---|
| gas density | 0.6, 1.0, 1.9, 1.5 | flat |
| temperature kT | 0.5, 1.3, 1.6, 1.8 | flat, p=0.76 |
| radius | 1.6, 1.7, 1.4, 0.6 | flat |
| redshift | 1.1, 1.6, 0.9, 0.4 | flat, p=0.68 |
| enclosed gas mass | 1.4, 2.6, 1.0, 0.8 | flat |
| baryonic acceleration g_bar | 1.7, 1.2, 0.5, 1.5 | flat |

No variable moves the residual. But the per-bin errors are 20–50%, so this
excludes only a *strong* dependence, not a weak one.

## What this does and does not say

It does **not** say the RAR works on clusters. The literature result — MOND
missing clusters by roughly two — rests on far better data than this, and the
discrepancy is strongest in the core and the far outskirts, neither of which
this radial range covers well.

It says **this dataset cannot see a factor of two**, and that the factor of two
it appeared to see was an artefact of the estimator.

One systematic remains uncontrolled and it acts in the relevant direction: the
outer slope β of the extended gas model is set by the surface brightness at
large off-axis angle, which is exactly where the analytic ACIS vignetting
approximation is least reliable. Too shallow a β overestimates the outer gas
mass, inflates `g_bar`, and shrinks the residual. Fixing that needs CIAO
exposure maps, and until it is fixed no claim should be made in either
direction.

## The correction that generalises

**Never filter on the quantity whose noise you are trying to average.** Selecting
`ds_obs > 0` looks like removing unphysical values; a negative lensing signal is
not unphysical, it is a measurement of a small positive quantity by an
instrument with noise. The right move is to keep every point, weight by inverse
variance, and work in linear space where the estimator is unbiased.

---

# Run BW — is the residual one number? and a second broken null

## The null that was wrong

`stacked.py` reported every variable "flat", three at **p = 1.00** — the real
binning more consistent than all 2000 permutations. That is not a result, it is
a broken null. It permuted **points** across clusters. Real bins hold several
correlated points from one cluster; permuted bins mix clusters and scatter more,
inflating the null χ² and making real data look falsely flat. The null must
permute **cluster labels**, keeping each cluster's points together.

## The question Milgrom's method actually asks

Not "what modification closes the gap". MOND was derived by requiring one change
to produce flat rotation curves **and** Tully-Fisher simultaneously: if
`a = √(a_N a₀)` at low acceleration then `v⁴ = G M a₀` falls out, with one
universal constant and **zero free parameters per galaxy**. Dark halos permit
both regularities and predict neither, needing three numbers per galaxy that the
data turned out not to need.

The decisive property was universality. So for clusters:

> **Is the residual one number across clusters?**

## Per-cluster residuals

| cluster | observed / RAR-predicted | z | bins |
|---|---|---|---|
| 1eRASS J052042.5−132848 | 1.87 ± 0.51 | 0.320 | 7 |
| 1eRASS J063846.8−535831 | 1.79 ± 0.39 | 0.226 | 8 |
| 1eRASS J045410.6−030056 | 1.75 ± 0.86 | 0.540 | 7 |
| 1eRASS J140102.0+025240 | 1.58 ± 0.29 | 0.253 | 7 |
| 1eRASS J111320.3+173542 | 1.35 ± 0.47 | 0.171 | 7 |
| 1eRASS J043900.7+071603 | 1.03 ± 0.82 | 0.254 | 6 |
| 1eRASS J123625.2+163246 | 0.99 ± 1.18 | 0.069 | 4 |
| 1eRASS J065829.9−555637 | 0.53 ± 0.33 | 0.297 | 6 |
| 1eRASS J044309.7+021017 | 0.52 ± 0.73 | 0.200 | 6 |
| 1eRASS J134730.8−114510 | 0.04 ± 0.35 | 0.450 | 7 |

Grand mean **1.09**.

## And then the error budget, which is the point

| error budget | χ² / 9 | p(one universal value) |
|---|---|---|
| shear shape noise only | 22.0 | **0.009** |
| + the gas model's own anchor scatter | 17.4 | 0.043 |
| + anchor scatter and a 15% vignetting term | 15.2 | **0.086** |

The first row is what a careless write-up would have reported: a 2.6σ detection
that the cluster residual is **not** universal, which would have been a genuine
result and the first crack in the amplitude picture.

It does not survive. **None of the gas-model uncertainty was in those error
bars.** `ds_err` is pure shear shape noise; the prediction also depends on a
β-model fit whose normalisation agrees with ACCEPT only to 0.04–0.19 dex — that
is a 10–56% uncertainty on `g_bar`, comparable to or larger than the shear
errors. Adding it, and a conservative 15% for the vignetting that sets the outer
slope, takes p to 0.086.

**Verdict: the residual is consistent with a single universal value. The
apparent cluster-to-cluster variation is not established, and most of it is my
own gas model.**

## One control that did pass

Correlating the residual against the number of background sources per cluster —
a pure noise proxy carrying no physics — gives r = +0.07, p = 0.89. If the
variation were driven by clusters with poor lensing, that correlation would be
there. It is not. So the variation, such as it is, is not simply the noisiest
clusters scattering furthest.

No physical variable tracks the residual either: redshift gives r = −0.52 at
p = 0.155, and with ten clusters a correlation needs |r| > 0.63 to clear p<0.05.
Temperature could not be tested at all — only four of the ten have a measured kT.

## What now binds, precisely

The vignetting model. It is the largest term in the error budget above, it sets
the outer slope β, β sets the enclosed gas mass, and the gas mass is the entire
prediction. Everything downstream — the universality test, any search for a
cluster-specific variable — is limited by it before it is limited by the data.

Fixing it means CIAO exposure maps, which is an install and a re-reduction, not
an analysis. Until then this dataset cannot distinguish a universal amplitude
from a varying one, and no cluster-unique thermodynamic variable can be tested
against a residual whose error is dominated by the model that produced it.
