# MOND observation atlas — executed milestone

The atlas goal is active and **not complete**. We built and ran a local catalog,
a fixed radial comparison, and numerical foundations for the 3D cube pipeline.
There are **zero newly validated full-field galaxy cube predictions** in this milestone.

## What is available

| Product | Actually processed |
|---|---:|
| Catalog identity groups | 13,530 |
| Possible duplicate pairs awaiting review | 95 |
| Identity groups without a coordinate | 58 |
| Local MaNGA population records | 10,071 |
| PROBES-I catalog rows | 3,163 |
| SPARC radial models | 175 galaxies / 3,391 radii |
| Legacy WALLABY records | 303, including failures and repeat releases |
| Verified image/cube assets | 137 / 137 files |
| Verified raw asset bytes | 5,079,675,498 |
| Galaxies with local HI cubes | 12 |
| Completed total-3D-mass / full-cube gravity validations | 0 |

These source counts overlap. The 13,530 identity groups are **not a certified count
of distinct physical galaxies**. Exact recognized names merge; positional proximity
only creates a review entry. MaNGA is a prior custom selection, not the survey's
official high-quality sample. PROBES supplies extended rotation curves and matched
photometry for a subset; this ingestion verifies its catalog, not every profile.
[SPARC](https://astroweb.case.edu/SPARC/), [PROBES-I](https://arxiv.org/abs/2209.09912).

The MaNGA records preserve 3,033 published integrated-HI detections, 3,280 upper
limits and 3,758 unknown/unavailable entries. A missing measurement is never
replaced with zero gas. Halo and environment-acceleration proxy columns were
excluded from the ingestion allowlist. Stellar-population masses remain model-dependent.

## The first measured pattern

We held the gravity formula fixed: a0 = 1.2e-10 m/s², disk M/L = 0.5 and bulge
M/L = 0.7, with the simple MOND interpolation function and no dark-halo term.
The published signed gas-force convention is preserved. Every input archive row
was checked against the stored SPARC decimal strings. No galaxy gravity parameters
were fitted.

Quality 1–2, inclination 30–80 degrees, at least five valid radii, and finite
positive radius/speed/error/inward total force leave **126 galaxies
and 2,694 radii**. All excluded rows and galaxies remain in the output.
We did not remove low-speed points for having large relative errors.

**MOND's radial approximation has smaller fractional speed error in
107 of those 126 galaxies.**
The median galaxy's RMS fractional speed error falls from
42.0% for baryon-only Newtonian gravity
to 15.6% for this MOND approximation.
The more outlier-sensitive galaxy-weighted RMS is
45.3% versus
36.5%.
Slow inner measurements can dominate fractional errors; neither statistic is a
calibrated likelihood or a claim that all galaxies fit well.

In plain terms, the usual low-acceleration correction helps many galaxies. We
then asked whether a galaxy's gas fraction, brightness concentration, size or
broad type helps predict the error left over, after accounting for the range
of baryonic acceleration sampled.

| Added descriptor | Change in held-galaxy residual MSE; positive is better | Four-test adjusted permutation p |
|---|---:|---:|
| gas mass fraction | -0.48% | 1.00 |
| effective stellar surface brightness | +3.71% | 0.06 |
| disk scale length | -0.25% | 1.00 |
| hubble type | -1.10% | 1.00 |

Gas fraction, size and broad galaxy type did not help this test. Surface
brightness gave a small 3.7% improvement, but its galaxy-bootstrap interval includes
no improvement and its adjusted permutation p is 0.06. That is a lead to examine,
not evidence that density causes a MOND failure. These are exploratory tests on
previously used galaxies. Whole-galaxy cross-validation prevents pixel-level
leakage; it does not create a pristine holdout, remove shared group effects, or
establish transfer to a different survey. The gas fraction is an HI-plus-uniform-
stellar-M/L proxy, not a measured total baryon fraction.

The 27-corner sensitivity envelope varies stellar M/L by ±20%, distance by its
reported error, and inclination by its reported error. It is **not a probability
interval**. Inclination changes velocity projection only in this radial stage;
the full source geometry must be rebuilt in the later 3D stage.

## Numerical work that is ready

The new NumPy engine solves Newtonian and QUMOND potentials on a 3D grid using
explicit boundary potentials and a face-centered nonlinear flux. It passed eight
analytic/symmetry/convergence gates. At the finest Plummer-sphere resolution,
force RMS error is 0.346%
for Newtonian gravity and 0.251%
for QUMOND, on the stated radial test region. This implements the two-Poisson-
equation formulation in [Milgrom (2010)](https://arxiv.org/abs/0911.5464).

The cube building blocks integrate all physical depth layers, apply finite
spectral channels and a spatial beam, and score supplied channel/spatial
covariances. Tests verify that two layers on one sightline remain two velocity
components and that the separable covariance score agrees with a dense matrix
calculation. A spectral cube's velocity axis is never interpreted as depth.

The 17-test offline suite passed. These numerical tests do **not** validate an
astronomical source model, real selection mask, separability of real noise, or
exterior-field boundary. The new implementation has no AQUAL solver yet.

## Why the full atlas is still unfinished

Only NGC2903 passed the prior strict stellar-registration validation. Eleven seed
registrations need repair or stronger independent checks. Even NGC2903 had only
27 of 242 tested positions with the required joint stellar/HI/CO coverage.
The earlier analysis identified possible multiple sightline intersections in
NGC2841, NGC2903, NGC3521 and NGC7331. The new depth integrator can represent them,
but has not yet been connected to validated observational 3D ensembles.

Every seed's readiness row keeps these gaps explicit. Total matter coverage,
depth priors, external baryonic fields, gas support/streaming, a validated mask,
and real channel-plus-spatial covariance remain required before gravity scoring.
The radial approximation cannot substitute for this work.

This session can write research files and run the bundled CPU Python. The
existing CUDA environment cannot launch; direct shell downloads are denied;
Git cannot write the linked worktree metadata outside the writable workspace.
**Nothing from this milestone has been committed or pushed to main.** Raw
observations remain in their existing private directories.

## Files and replay

- [Browse all 175 radial comparisons](atlas.html)
- [Pilot readiness](pilot-readiness.csv)
- [Remaining work and access status](execution-status.json)
- [Acquisition queue](acquisition-queue.json)
- [Catalog summary](../mond-atlas-catalog-004/summary.json)
- [Catalog identities](../mond-atlas-catalog-004/galaxies.csv)
- [Identity review queue](../mond-atlas-catalog-004/identity-review.csv)
- [Radial predictions](../mond-atlas-radial-002/radial-predictions.csv)
- [Galaxy residuals](../mond-atlas-radial-002/galaxy-residuals.csv)
- [Pattern results](../mond-atlas-radial-002/patterns.json)
- [Full-field numerical checks](../mond-atlas-numerics-001/validation.json)

From the repository root, use Python with NumPy. Each run refuses to overwrite
an existing directory. The recorded execution used Python 3.12.14 / NumPy 2.3.5.

```text
python tests/test_mond_atlas_offline.py
python scripts/build_mond_atlas_catalog.py --output work/gravity-first-principles/mond-atlas-catalog-replay
python scripts/run_mond_atlas_radial.py --output work/gravity-first-principles/mond-atlas-radial-replay
python scripts/mond_atlas_fields.py --output work/gravity-first-principles/mond-atlas-numerics-replay/validation.json
```
