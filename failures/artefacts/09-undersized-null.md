# 09. A permutation null too small to show its own width

**Where:** Run BS, 2026-09-07

## What was claimed

The asymmetry of a cluster's gas predicts the asymmetry of its gravitational lensing.

|  |  |
|---|---|
| the number that looked real | **+2.67 sigma, p = 0.000, at 40 permutations** |
| the same number under control | **+1.64 sigma, p = 0.055, at 200 permutations on identical data** |

## Why it happened

The null's standard deviation was 0.0095 at 40 draws and 0.0144 at 200. Forty draws could not resolve the null's own width, so the measured gain of +0.0146 was compared against a distribution that looked far tighter than it is. Nothing about the data changed between the two runs.

## What caught it

Increase the permutation count until the null's standard deviation stops moving. Plot sd against draw count.

## The lesson

An undersized permutation null is not a conservative control. It inflates significance, and it does so silently. This one was found while writing up the other eight.

**Cost of the control:** 160 more permutations -- about four minutes.
