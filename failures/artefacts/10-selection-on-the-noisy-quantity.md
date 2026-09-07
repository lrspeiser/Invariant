# 10. Filtering out negative measurements manufactured a factor of two

**Where:** Runs BT and BU, retracted by Run BV, 2026-09-07

## What was claimed

The radial acceleration relation under-predicts cluster weak lensing by a factor of 1.6 to 2.2, measured with baryons computed from the observed gas.

|  |  |
|---|---|
| the number that looked real | **pred/obs = 0.620, i.e. a factor 1.6 of missing gravity** |
| the same number under control | **0.914 +- 0.142 once the estimator is fixed -- consistent with 1** |

## Why it happened

Two compounding mistakes, both in the estimator. First, the analysis filtered `ds_obs > 0` before taking the ratio. Ten of 65 rows have a negative observed lensing signal, mean -0.53 sigma -- ordinary downward fluctuations on a small positive quantity measured at low signal-to-noise. Dropping them removes ONLY downward fluctuations and keeps every upward one, inflating the apparent lensing and therefore the apparent missing gravity. Second, it took the MEDIAN of a ratio whose denominator carried more than 50% error on 45 of 55 points, which is not an estimate of the ratio of the truths.

## What caught it

Keep every point, weight by inverse variance, and work in linear space where the estimator is unbiased. Then check what the before-and-after difference was.

## The lesson

Never filter on the quantity whose noise you are averaging. A negative lensing signal is not an unphysical value to be cleaned away -- it is a measurement. Removing it is a selection on noise, and it biases in the direction that looks like a discovery. The companion failure is that the follow-up search which found 'no environmental dependence' was VACUOUS: the intrinsic scatter was 0.000 dex, all 0.523 was measurement error, so no modification could have reduced it whatever the physics.

**Cost of the control:** One line: drop the filter, change median to inverse-variance mean.
