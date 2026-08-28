# Gravity roadmap Item 10: WALLABY baryonic boundaries

## Decision

`INCONCLUSIVE_ITEM10_WALLABY_QUALITY`

The frozen projected-H I boundary grammar is not promoted. On the usable held-out data its
selected terms are directionally worse than a flexible local H I-profile baseline, but the
source fails the preregistered sample-size and retention gates. This is not a rejection of
all boundary, shell, interface, finite-domain, or field-focusing gravity theories.

## Frozen question

Before opening WALLABY predictor profiles or rotation responses, Item 10 froze the question:

> Does a universal term derived from an observed H I edge, shell, interface, or finite
> baryonic domain add held-out predictive information about resolved rotation speed beyond
> flexible local H I-profile variables?

The scientific freeze is commit `d1f3ea0a303427077a07f6017abd4d0e87b23f0a`.
It fixed the source, predictor/response boundary, exclusions, 75/25 split rule, five-fold
nested evaluation, quality gates, baseline, pseudorandom seed, and formula grammar. The
target-blind sample and all candidates were committed at
`a7989ad42079813d7798d81671b83fd7bb6dd99e` before the response request.

## Formula search

The seeded PCG64 generator created exactly `131,072` cells across twelve declared families:

- edge and signed-edge radial basis kernels;
- shell cusps, image-radius, finite-domain, and Robin-like boundary terms;
- dual-edge annuli and edge-tension terms;
- flux-capture mismatch;
- reflection-interference, edge-resonance, and log-periodic boundary terms.

Threshold, scale, power, phase, and one of five target-blind baryonic modulations varied by
cell. Candidate polarity was omitted because the fitted universal scalar coefficient makes
a sign-flipped cell algebraically equivalent. Every family is labeled
`KNOWN_FAMILY_COMBINATION`, `COMBINATION`, or `UNRESOLVED`; no historical novelty is claimed.
No formula or parameter range was generated after a response was opened.

For each outer fold, the other four folds were rotated through inner validation. A fixed
ten-variable local H I ridge model was trained first, and each boundary cell could fit only
one universal scalar coefficient to training residuals. No object-specific gravity
coefficient, identity feature, dark mass, dynamical mass, stellar response, or observed
speed entered a predictor.

## Real-data source and quality

The public WALLABY Pilot Survey DR2 kinematic table exposed `303` catalogue rows. Only `129`
had at least eight usable projected H I surface-density radii. Target-blind exclusions then
removed eleven PROBES-I coordinate overlaps, 37 profiles without a bracketed
`1 M_sun pc^-2` edge, and the one identity accidentally exposed during the endpoint audit.
The frozen release-row sample contained 66 exploration and 19 reserved rows.

WALLABY `name` is not a unique model key across kinematic releases. The first exact-name
response request therefore triggered the one-row-per-name assertion and wrote no artifact.
The repair used no response values: every name appearing more than once anywhere in the
303-row predictor catalogue was excluded without replacement. This retained 38 unambiguous
exploration names and eleven unambiguous, unqueried confirmation names.

Two release rows originally labeled reserved confirmation may have been transmitted during
the failed broad-name request because their physical names also had exploration releases.
They were never written to the final response artifact or used in extraction, selection, or
scoring. The receipt conservatively records `confirmation_opened=true`, the affected rows as
two, and the admission gate as failed. This means the attempt is not a clean confirmation
experiment.

Of the 38 final exploration galaxies, twenty pass all frozen rotation-quality rules and
contribute 275 radial points. Sixteen fail the minimum-point rule, two fail inclination, and
one fails the published quality flag (one galaxy has two reasons). Retention is `52.63%`.
The preregistered requirements were at least 140 passing galaxies and at least 60% retention,
so a formal positive or negative promotion decision is impossible.

## Result on the valid diagnostic subset

The RTX 5090 evaluated `720,896,000` candidate-point scoring combinations in `3.80` seconds
using CuPy 13.5.1. The maximum CPU/GPU component difference across the frozen cross-check was
`3.33e-16`.

| Model | Equal-galaxy held-out MSE | Held-out R2 |
|---|---:|---:|
| Flexible local H I baseline | 0.0361634 | 0.3916 |
| Nested selected boundary term | 0.0367760 | 0.3813 |

The boundary result is `1.69%` worse than the local baseline. Its mean per-galaxy MSE gain is
negative, with paired sign-flip `p=0.583`. It loses in at least one half of edge radius, edge
sharpness, and profile mass; both profile-mass halves are negative. Only five of fourteen
admission gates pass.

The five outer folds choose different cells and the fitted coefficients change sign. That
instability, plus negative held-out gain, provides no evidence for one universal scalar
boundary correction in the tested representation.

## Failure-space record

Record this region as `NONPROMOTED_LOW_QUALITY_PROJECTED_HI_BOUNDARY_REGION`:

- the exact twelve scalar radial boundary families and seeded parameter ranges in the Item 10
  manifest;
- projected circular H I mass proxies with one universal additive log-speed coefficient;
- the fixed local H I ridge baseline and exact nested selector used here;
- algebraic renamings, sign flips, or rescalings that add no new information.

Do not retune these cells on the 20 opened valid galaxies. A retry requires a materially
larger independent source, a complete baryonic profile, a vector/tensor field, an
action-derived boundary condition, a causal-history mechanism, or another response type.

## What remains open

- Complete stellar, gas, plasma, and three-dimensional baryonic boundaries were not tested.
- The experiment used scalar radial features, not a field that redirects flux or predicts
  lensing.
- It did not derive a covariant action, conservation law, stability condition, or classical
  limit.
- It does not establish or rule out an alternative to general relativity, dark matter, or a
  historically new formula.

## Next real test

Advance to Item 11, external baryonic field. Before opening a fresh response, freeze
neighbor-density, nearest-baryon, tidal-tensor, filament, and large-scale-boundary variables
from an independent environment catalogue. Require whole-object held-out prediction and
separate a true environmental field from survey, distance, group-membership, and population
labels. Do not open the eleven clean WALLABY confirmation names.

## Replay evidence

- result file SHA-256:
  `6e711d42fa092c3f426352449ddc046cce041bc19f3972b6cf65b16d3f0918df`
- result content SHA-256:
  `d75b7bfa9927acf18cfdafe0c2169b9bd454b8765e6607355f2fd4855b77dcd7`
- response-source SHA-256:
  `918e514d4f8158c37a5713fed72746501aa26fa6e292911b6f3fd8ab5a33f240`
- replay command:
  `python -m sigma_theory_compiler.gravity_item10_wallaby_boundaries check`
