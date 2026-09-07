# 05. Published fit noise with a preferred sign

**Where:** the potential-depth lane

## What was claimed

A significant negative potential-depth effect.

|  |  |
|---|---|
| the number that looked real | **-6.6 sigma** |
| the same number under control | **-0.0666 +- 0.0101 with the true effect set to zero** |

## Why it happened

Noise in a PUBLISHED X-ray density fit, propagated through the estimator, did not average out. It had a preferred direction, and the estimator converted it into a significant negative value.

## What caught it

Set the true effect to zero, redraw the catalogue inputs at their published uncertainties, and push them through the actual estimator.

## The lesson

The bias belonged to the catalogue that was combined in, not to the measurement being reported. Inherited uncertainty is still your problem.

**Cost of the control:** A Monte Carlo over published error bars.
