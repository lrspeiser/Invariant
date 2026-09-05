# Precise omitted-potential calculation: retained result

The repaired source passes its registered source-grid checks, and the active correction passes its independent derivative checks.
This is a numerical milestone for the existing source model, not a new
astronomical result or a validated gravity law.

## Fixed source and arithmetic changes

The physical source, all 4,715 locations per thickness, the 60--80 kpc join
and numerical thresholds are unchanged. The canonical calculation uses
50 decimal digits for cancellation and the low-k Bessel band below 8 kpc^-1.
The stored source coefficients and surface transforms are retained; the
same-order radial Gauss rule is refined accurately. Nine cases per thickness
include eight separate variations, including 35 digits and a wider accurate
band below 16 kpc^-1. All derivatives remain derivatives of one potential.

| Thickness | Force refinement | Hessian refinement | Third refinement | Density identity | Density-gradient identity |
| --- | --- | --- | --- | --- | --- |
| primary | 4.40691e-09 | 1.83615e-08 | 4.96445e-06 | 7.84681e-08 | 4.97557e-05 |
| height_half | 3.86675e-09 | 1.47309e-07 | 4.35801e-05 | 6.27186e-07 | 0.000780684 |

The registered source-grid targets are 0.0001 for force, 0.002 for Hessian and
density, 0.01 for the third tensor and density gradient, and 0.000001 for
potential overlap in GM/r units. Other quantities use their inherited
field/source scales; they are not fractional errors in vanishing density.

## Independent potential derivatives

| Verification | Thickness | Fine gradient error | Fine Hessian error | Fine third-tensor error |
| --- | --- | --- | --- | --- |
| source-tail-verification-001 | primary | 5.0125e-08 | 3.61502e-06 | 0.000743872 |
| source-tail-verification-001 | height_half | 1.00244e-07 | 1.44566e-05 | 0.0059498 |
| source-tail-verification-003 | primary | 6.29024e-11 | 4.37895e-10 | 3.34678e-07 |
| source-tail-verification-003 | height_half | 1.01626e-10 | 8.18932e-10 | 4.76575e-07 |

The fine-step target remains 0.0001. This verifier differentiates the newly
added active correction (1-w) delta psi, normalized by full-field scales.
Inherited Hankel and exterior derivatives have separate earlier controls;
this is not a new finite-difference audit of the entire matched potential.
The verifier checks every original point
at both 0.001 and 0.0005 kpc steps, including both one-sided radial interface
stencils, axis parity and the exact active join weight. It loads checked
execution snapshots. Only unused stencil offsets away from interfaces are
omitted from the expensive evaluation grid; no comparison is removed.

The original failed source join, serialized partial tail file, double-precision
derivative failure and initial Gauss-weight unit-control failure remain in
their original evidence directories. The first precise derivative verifier
also stopped on an inactive-row indexing error; its failure and executed
script are retained as source-tail-verification-002. A regression control and
full stencil-index preflight now precede the expensive calculation. No failure
is relabeled as a success.
The two new precision controls and the focused implementation suite passed
at their recorded execution hashes. The verifier supports both legacy and precise execution snapshots; the new
numerical verification checks the precise correction.

## Remaining work

Next, build and validate a fast representation derived from one C3 potential, then evaluate the full action flux and its separate Poisson solve.
Successful sampled checks do not establish uniform continuum error bounds or
production interpolation accuracy. Galaxy predictions must then be tested
with the same global constants used for clusters and the Solar System.

The user-proposed overlap and elastic directions remain open under their
recorded conservation, mass-scaling and known-family constraints. Complete
light coupling, dynamical stability, source uncertainty, direct outer-star
observations and independent confirmation remain requirements of the full
discovery goal. No new observational score or physical exclusion is added.

## Result hashes

- `source-tail-002`: `804c0a38ff29bc8e8ccfe42e8ab385e7133e6878260e527c9b14d1d78bc2461e`
- `source-tail-verification-001`: `f63f2fb795d4b9f12a13e947331ad071ff9c3b9294353d11526474e8b936d49f`
- `source-tail-003`: `363524c53eab9bfca2d0efe01bfaf80f876660e4f9b236557f844c9f23763a6f`
- `source-tail-verification-003`: `8bbaf18dcb90073b5446fb1a011866ef56780105d3a447dd86db8b2431baa768`
- `source-tail-controls-003`: `443c3d03ea80b9165ef9db97a87c9113b6bd105a1a8f987c29c88d2bb9492bfe`
- `source-tail-verification-002/failure.json`: `b9df3b86a70028651518f6f2d6b484e1512f94360b9bfee68d749a88825f0ad7`
