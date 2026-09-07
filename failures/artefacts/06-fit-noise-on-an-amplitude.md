# 06. The same noise biased an amplitude low by a factor of 2.2

**Where:** the slip lane

## What was claimed

A measured amplitude Sigma_s below unity.

|  |  |
|---|---|
| the number that looked real | **factor 2.2 low, at 17 sigma** |
| the same number under control | **-0.026 dex at error scale 0.25, -0.125 at 0.50, -0.336 at 1.00 -- monotonic in the input error** |

## Why it happened

Identical mechanism to case 05, but applied to a pure amplitude with no free parameter able to absorb it. The largest artefact in the set.

## What caught it

Scale the input errors up and down and watch the estimator move. A real amplitude does not track the error scale of its inputs.

## The lesson

Every such amplitude must be read against this null rather than against 1. Its factor-two width became the lane's dominant uncertainty.

**Cost of the control:** Three Monte Carlo runs at different error scales.
