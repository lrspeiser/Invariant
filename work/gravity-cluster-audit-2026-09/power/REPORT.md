# Is the X-COP radial pressure test underpowered, or is there no signal?

**Injection-recovery simulation. Seed 20260903, fully reproducible.**

Run `python power_analysis.py` (~150 s). Full numbers in `power_results.json`,
full console transcript in `run.log`. Reproducibility verified: two independent
fresh-process runs produce byte-identical transcripts.

---

## The answer, in one table

Detection rate at an **exact 5% false-positive rate**, 400 synthetic
realisations per kappa, 3000 derangements per realisation, kappa0 = 1.36e5.

| kappa / kappa0 | kappa | A (rank) | A_idx (rank, c70) | B (within-cluster) | **C (normalisation)** |
|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.050 | 0.051 | 0.050 | **0.050** |
| 0.5 | 6.8e4 | 0.083 | 0.077 | 0.060 | **0.370** |
| **1** | **1.36e5** | **0.107** | **0.113** | **0.055** | **0.477** |
| 2 | 2.72e5 | 0.105 | 0.100 | 0.068 | **0.550** |
| 4 | 5.44e5 | 0.102 | 0.095 | 0.050 | **0.588** |

**At kappa = kappa0 the test detects an injected signal of exactly the claimed
size 48% of the time**, and only through one of the three statistics. The other
two never rise meaningfully above the 5% false-positive floor, at any kappa.

That is neither the "<30%" that would make "underpowered" a clean verdict nor
the ">80%" that would make the observed null real evidence against the model.

### Verdict (one line)

**The observed null is NOT informative: with 12 clusters the test finds a
genuine kappa0 signal only 48% of the time, so seeing no detection is close to
a coin flip and cannot discriminate for or against the pressure model.**

Two riders, both of which cut against the original framing:

1. c71's blanket claim that "the test has little power" is **right for
   statistics A and B and wrong for C**. A and B are structurally blind: they
   sit near 5% even when kappa is a *million* times kappa0. C is genuinely
   half-powered.
2. The real data's C result was **p = 0.069, not a null**. An injected kappa0
   signal produces a p-value that small 61% of the time; no signal produces one
   only 8.6% of the time. The observed value is the *typical* outcome under the
   model, not evidence against it.

---

## 1. The pipeline reproduces the published test

| statistic | this run | published | null (this run) | published null | p (this run) | p (published) |
|---|---:|---:|---:|---:|---:|---:|
| A  median within-cluster rank rho | +0.6426 | +0.643 | +0.6476 | +0.625 | 0.548 | 0.36 |
| B  within-cluster sd log10 ratio | 0.1689 | 0.1689 | 0.1667 | 0.1670 | 0.712 | 0.71 |
| C  sd of per-cluster median ratio | 0.0683 | 0.0683 | 0.0844 | 0.0846 | 0.069 | 0.064 |

B and C reproduce to the last printed digit. A's *true* value reproduces
exactly, but its null does not, because **c70 and c71 built the shuffled
temperature profile differently**: c70 interpolated on bin index, c71 on
r/R500. Re-running A with c70's index-space interpolation gives null +0.6246
and p = 0.326, i.e. the published +0.625 / 0.36. Both variants are carried
through the whole analysis as `A` and `A_idx`; they behave identically.

## 2. What the test has to work with (no simulation involved)

A derangement swaps cluster i's predicted profile for cluster j's. Everything
the test can use is the difference between those two predictions, and it splits
into two channels:

| kappa / kappa0 | SHAPE sd (dex) | / 0.1689 noise | NORM sd (dex) | / 0.0683 noise |
|---:|---:|---:|---:|---:|
| 0.0625 | 0.00510 | 0.030 | 0.01103 | 0.162 |
| 0.25 | 0.01258 | 0.074 | 0.02722 | 0.399 |
| 1 | 0.02040 | 0.121 | 0.04349 | 0.637 |
| 4 | 0.02446 | 0.145 | 0.05133 | 0.752 |
| 1e4 | 0.02626 | 0.155 | 0.05466 | 0.801 |

- **SHAPE** -- how the prediction runs with radius inside one cluster. Only A and
  B see it. At kappa0 it is 12% of the within-cluster noise, and that noise is
  strongly correlated along the profile (lag-1 = +0.90), so it barely averages
  down. This is why A and B are blind.
- **NORM** -- the level of the prediction, cluster by cluster. Only C sees it.
  At kappa0 it is 64% of the cluster-to-cluster noise, and a derangement
  scrambles it across all twelve clusters at once. This is why C works.

**Both channels saturate.** As kappa grows, `pred -> sqrt(kappa*3kT/mu m_p c^2)`,
so `log10 pred -> 0.5 log10 kT + const` and both the shape and the spread of
levels stop depending on kappa at all. The ceiling is only 1.29x (SHAPE) and
1.26x (NORM) what kappa0 already delivers. That is a fact about the data, not
about the simulation, and it is why the power curve flattens instead of climbing.

## 3. Noise calibration -- the evidence

Synthetic excess profiles are

```
log10 exc_syn(r) = log10 sqrt(1 + kappa_inj * 3kT_i(r)/(mu m_p c^2))
                 + a_i                      per-cluster normalisation offset
                 + LAM * s_i * N_i(r)       within-cluster noise
```

`N_i` is generated **non-parametrically**: it is *another cluster's real
residual profile*, randomly signed and rescaled. Its amplitude, its correlation
along the profile, its non-Gaussian tails and the measurement error embedded in
it are therefore exactly those of the data. That leaves only two free
parameters, and both are **solved**, not assumed:

- `LAM = 1.056`, solved so the median within-cluster scatter matches 0.1689 dex.
- `sigma_offset = 0.0495 dex`, solved so the per-cluster median scatter matches
  0.0683 dex. **It came out positive**, i.e. the within-cluster noise alone does
  not already exhaust the observed cluster-to-cluster spread. Had it hit zero,
  the synthetic data would have been noisier than reality and every power number
  biased low.

### Two targets fitted, seven quantities checked

| quantity | REAL | SYNTHETIC (mean, 5-95%) | |
|---|---:|---|---|
| B median within-cluster sd | 0.1689 | 0.1690 [0.1524, 0.1871] | **target** |
| C sd of per-cluster medians | 0.0683 | 0.0648 [0.0409, 0.0916] | **target** |
| residual autocorrelation, lag 1 | +0.9001 | +0.8967 [+0.8776, +0.9132] | not fitted |
| lag 2 | +0.7301 | +0.7254 [+0.6736, +0.7708] | not fitted |
| lag 4 | +0.4127 | +0.4145 [+0.3171, +0.5035] | not fitted |
| lag 8 | +0.1679 | +0.2074 [+0.1144, +0.3075] | not fitted |
| spread of per-cluster sd across the 12 | 0.0670 | 0.0716 [0.0557, 0.0869] | not fitted |
| mean radial tilt | -0.1245 | -0.1245 [-0.1464, -0.1016] | not fitted |
| A median within-cluster rank rho | +0.6426 | +0.6369 [+0.5478, +0.7150] | not fitted |

Five independent quantities that were never fitted -- the whole autocorrelation
function out to lag 8, the heterogeneity of the per-cluster scatters, the radial
tilt, and statistic A itself -- come out right. The synthetic data are a
faithful stand-in for the real ones.

### Three facts about the noise that drive everything

- **It is intrinsic, not photon noise.** The per-point measurement term implied
  by `eT_X/T_X` is 0.0162 dex -- 10% of the total 0.1689 dex scatter.
- **It is correlated.** lag-1 = +0.900, lag-2 = +0.730, lag-4 = +0.413. The
  residuals are smooth curves, not point scatter, so they barely average down
  over the ~49 radial bins per cluster.
- **It carries a common radial tilt** of -0.1245 dex (cluster-to-cluster sd only
  0.0270). All twelve are negative: this is the known X-ray radial-shape bias
  from outward-rising non-thermal support -- a systematic, and modelled as one.

## 4. False-positive check at kappa = 0 -- the published pipeline FAILED it

2000 realisations with no temperature signal injected:

| statistic | nominal FPR | 95% CI for 0.05 | critical p | realised rate |
|---|---:|---:|---:|---:|
| A | **0.0095** | [0.040, 0.060] | 0.1579 | 0.050 |
| A_idx | **0.0205** | [0.040, 0.060] | 0.0953 | 0.050 |
| B | 0.0470 | [0.040, 0.060] | 0.0549 | 0.050 |
| C | **0.0655** | [0.040, 0.060] | 0.0363 | 0.050 |

Largest deviation 0.0405 = **8.3 sigma**. The p-values are miscalibrated.

**Cause, established from the data rather than guessed.** The true pairing
evaluates cluster i's temperature *at its own nodes*; a deranged pairing
*resamples* cluster j's piecewise-linear profile onto those nodes, which smooths
it and clamps it wherever j does not reach as far as i. The two arms are not
exchangeable even with no signal present. Measured directly: the resampled
profile's steps decrease 57.3% of the time versus 55.2% for the true profile
(it is smoother), and the spread of its per-cluster levels is 17% smaller. A is
therefore dragged conservative and **C anti-conservative** -- the statistic that
carries the whole result was running at 6.6%, not 5%.

**Fix.** Every power number in this report uses the empirically size-corrected
threshold in the `critical p` column, which is the 5% quantile of the kappa = 0
distribution for that statistic. Each test then has an exact 5% false-positive
rate by construction (right-hand column). Without this correction C's apparent
power at kappa0 would have read 0.552 instead of 0.477.

## 5. The smallest detectable kappa: there is none

| kappa / kappa0 | kappa | A | A_idx | B | C | C, matched-kappa |
|---:|---:|---:|---:|---:|---:|---:|
| 0.0625 | 8.5e3 | 0.048 | 0.072 | 0.048 | 0.083 | 0.120 |
| 0.125 | 1.7e4 | 0.077 | 0.085 | 0.072 | 0.152 | 0.158 |
| 0.25 | 3.4e4 | 0.090 | 0.072 | 0.068 | 0.265 | 0.250 |
| 0.5 | 6.8e4 | 0.083 | 0.077 | 0.060 | 0.370 | 0.375 |
| 1 | 1.36e5 | 0.107 | 0.113 | 0.055 | 0.477 | 0.455 |
| 2 | 2.72e5 | 0.105 | 0.100 | 0.068 | 0.550 | 0.562 |
| 4 | 5.44e5 | 0.102 | 0.095 | 0.050 | 0.588 | 0.565 |
| 8 | 1.09e6 | 0.115 | 0.117 | 0.080 | 0.615 | 0.585 |
| 16 | 2.18e6 | 0.098 | 0.105 | 0.072 | 0.613 | 0.575 |
| 64 | 8.7e6 | 0.113 | 0.117 | 0.043 | 0.627 | 0.618 |
| 1024 | 1.39e8 | 0.107 | 0.107 | 0.083 | 0.610 | 0.630 |
| 1e6 | 1.36e11 | 0.150 | 0.142 | 0.062 | 0.620 | 0.613 |

**No kappa reaches 80% power.** The highest detection rate anywhere on a scan
spanning seven decades is 0.627 (0.630 for a more generous pipeline that is told
the injected kappa). This is the saturation of section 2: past a few times
kappa0 the prediction becomes `0.5 log10 kT + const` and stops changing, so
extra coupling buys nothing. **With 12 clusters this test cannot reach 80% power
at any coupling strength whatsoever.** That is a stronger and more useful
statement than a threshold value would have been.

### What sample size would be needed

Power at kappa0 versus number of clusters, treating the 12 observed temperature
profile shapes as the population and resampling with independent noise:

| clusters | 12 | 24 | 36 | 48 | 72 | 96 | 192 | 384 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C | 0.365 | 0.765 | **0.930** | 0.960 | 0.995 | 0.995 | 1.000 | 1.000 |
| A | 0.100 | 0.080 | 0.250 | 0.170 | 0.370 | 0.210 | 0.370 | 0.530 |
| B | 0.170 | 0.060 | 0.085 | 0.070 | 0.065 | 0.080 | 0.145 | 0.265 |

**About 30 clusters would give 80% power at kappa0** (24 gives 77%, 36 gives
93%). This is mildly pessimistic: exact replication lets a derangement pair a
cluster with an identical twin, which a real sample would not do. So roughly
25-30 X-COP-quality clusters would settle the question that 12 cannot -- a very
reachable target, since CHEX-MATE alone has about 120.

Note that A and B do not converge usefully even at 384 clusters. The shape
channel is not merely small-sample-limited; it is limited by the fact that all
cluster temperature profiles decline outward in much the same way.

## 6. Robustness

**Symmetrised pipeline** (every pairing, including the true one, routed
native -> common grid -> cluster radii, all clusters cut to the common radial
range). This attacks the exchangeability failure structurally rather than
empirically:

| statistic | nominal FPR | size-corrected power at kappa0 | main pipeline |
|---|---:|---:|---:|
| A | 0.0060 | 0.103 | 0.107 |
| B | 0.0440 | 0.090 | 0.055 |
| C | 0.0130 | 0.508 | 0.477 |

Symmetrising moves the nominal rates toward 0.05 without reaching them, and
costs 16% of the radial points, so it is a cross-check rather than a
replacement -- but its size-corrected power agrees with the main pipeline.

**Noise-model sensitivity**, each variant fully re-calibrated to the same two
targets:

| variant | lag-1 | solved sigma_offset | power C | power B | power A |
|---|---:|---:|---:|---:|---:|
| resid (primary) | 0.896 | 0.0524 | 0.453 | 0.088 | 0.103 |
| resid, no sign flip | 0.892 | 0.0582 | 0.528 | 0.035 | 0.085 |
| parametric poly + Matern | 0.816 | 0.0526 | 0.538 | 0.115 | 0.078 |
| poly, correlation length / 2 | 0.762 | 0.0588 | 0.503 | 0.080 | 0.080 |
| poly, correlation length x 2 | 0.865 | 0.0396 | 0.503 | 0.148 | 0.093 |
| white noise | 0.726 | 0.0607 | 0.478 | 0.100 | 0.053 |

C's power is **insensitive to the noise correlation** -- every variant lands
between 0.45 and 0.54, white noise included. That is the calibration working,
not luck: C depends on the *total* cluster-to-cluster scatter of the median
ratio, which the second target pins at 0.0683 dex whatever the correlation.
Shortening the correlation lets more within-cluster noise average out of the
median, and the solved offset simply grows to compensate (0.0396 -> 0.0607 dex).
**The headline number does not rest on getting the noise model right.**

B does depend on the correlation (0.035 to 0.148) but never leaves the
neighbourhood of the 5% floor in any variant.

## 7. Where the real data actually sit

ADVANTAGE = (null - true) for B and C, (true - null) for A. Positive means the
true temperature pairing beats a shuffled one, which is what the model predicts.

| stat | observed | synthetic, kappa = 0 | synthetic, kappa = kappa0 |
|---|---:|---|---|
| A | -0.0051 | -0.0301 [-0.1201, +0.0515] | -0.0079 [-0.0829, +0.0661] |
| A_idx | +0.0180 | -0.0199 [-0.1186, +0.0740] | +0.0077 [-0.0755, +0.0863] |
| B | -0.0023 | -0.0009 [-0.0082, +0.0070] | +0.0003 [-0.0072, +0.0091] |
| **C** | **+0.0162** | **+0.0002 [-0.0156, +0.0182]** | **+0.0173 [-0.0017, +0.0318]** |

The observed C advantage of +0.0162 sits essentially on the kappa0 mean
(+0.0173) and near the top of the kappa = 0 range. Equivalently:

| stat | real p | P(p <= real, given kappa = 0) | P(p <= real, given kappa0) |
|---|---:|---:|---:|
| A | 0.548 | 0.338 | 0.495 |
| A_idx | 0.326 | 0.204 | 0.380 |
| B | 0.712 | 0.566 | 0.660 |
| **C** | **0.069** | **0.086** | **0.610** |

So on the one statistic with any power, **the data look about 7x more like an
injected kappa0 signal than like no signal**. That ratio of tail probabilities
is not a Bayes factor and should not be quoted as one, but the direction is
unambiguous, and it is the opposite of what "the radial test found nothing"
implies.

## 8. Caveats, stated plainly

1. **The noise calibration assumes the real residual is noise.** If the pressure
   model is wrong, the residual also contains model error, so the noise is
   over-estimated and these power figures are *under*-estimates. The bias runs
   toward "underpowered", i.e. it is conservative for the verdict given.
2. **Statistic C is not really a radial test.** It depends only on each
   cluster's *median* excess and *median* predicted excess, so it re-tests the
   cluster-level temperature/excess correlation already reported in
   `p01_rigorous.py` (rho = +0.61, p = 0.037, n = 12) in a different guise. The
   genuinely radial statistics -- A and B, which use the run of the profile
   within a cluster -- are exactly the ones with no power. **The
   zero-free-parameter radial test adds essentially no information beyond the
   n = 12 cluster-level correlation already in hand.** That is the most
   important structural finding here.
3. **The temperature-profile population is the observed 12.** The sample-size
   extrapolation assumes future clusters resemble these; exact replication also
   makes the derangement null slightly too easy, so N = 36 is an upper bound.
4. **A "detection" here means the derangement test fires**, not that kappa is
   recovered accurately. Nothing in this report constrains kappa itself.

## 9. What should be said about the original result

- Not "the radial test refutes the pressure model" -- it has 48% power on its
  one working statistic and would have missed a true signal about half the time.
- Not "the radial test supports the pressure model" -- p = 0.069 on a single
  statistic, with two others flat, is not a detection.
- The accurate statement: **twelve clusters are not enough**, the radial
  formulation adds nothing beyond the cluster-level correlation already tested,
  and roughly 30 X-COP-quality clusters would settle it.
