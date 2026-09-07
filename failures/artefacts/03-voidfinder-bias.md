# 03. 38 sigma from data containing no effect

**Where:** Run Q

## What was claimed

A redshift path-length effect in void catalogues.

|  |  |
|---|---|
| the number that looked real | **30-38 sigma** |
| the same number under control | **the same significance on data constructed with no path effect at all** |

## Why it happened

The void-finding algorithm imprinted its own bias on the path-length statistic, and the bias differed between algorithms -- VoidFinder and REVOLVER gave different amounts of it.

## What caught it

Simulate the null separately for each algorithm and subtract it. A transverse decomposition orthogonal to distance cut the bias 60x for VoidFinder and 24x for REVOLVER, and still did not reach zero.

## The lesson

The null must be simulated per algorithm and subtracted, never assumed to be zero.

**Cost of the control:** One synthetic catalogue per void-finder.
