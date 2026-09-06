# MOND atlas — first real-source full-field experiment

**One galaxy now has executed three-dimensional Newtonian and full QUMOND
fields: NGC2903. These are conditional source reconstructions, not a validated
3D mass posterior or a completed cube likelihood. The atlas goal remains open.**

## What this experiment found

1. **The mass arrangement changes the direction of the pull much more than the
   mean rotation prediction.** Keeping the mapped bar and asymmetry produces a
   sideways component about 15.5% of the mean inward force at 2 kpc, 7.5% at
   5 kpc and 5.1% at 10 kpc. Averaging the same mass into circular annuli removes
   most of it. This is a force prediction from a conditional map, not an observed
   streaming detection. It is not evidence for a new gravity law.
2. **Mass conversion dominates this particular sensitivity experiment.** At
   10 kpc, the low/high stellar-and-CO conversion cases give MOND force-equivalent
   speeds of approximately 174–227 km/s; the nominal value is about 202 km/s.
   Doubling the assumed stellar and gas heights lowers the nominal 10 kpc speed
   by about 2.8 km/s. These ranges are illustrative assumptions, not confidence
   limits. The highest conversion is not selected as the preferred model.
3. **The photometric decomposition matters.** In the publisher's ICA maps,
   dust-associated light contributes 27.8%
   of combined light on the valid pixels within 20 kpc, rising to
   36.2% between 2 and 5 kpc.
   Counting all that light as old stars would change the source. This does not
   establish that catalog masses or stellar ages are wrong. ICA uncertainty,
   aperture and sky calibration must be checked before comparing catalogs.

| Radius (kpc) | Mapped sideways/inward force | Circular-map numerical remainder | Change in mean MOND force-speed after circular averaging |
|---|---:|---:|---:|
| 2 | 15.50% | 1.45% | +0.275% |
| 5 | 7.53% | 0.78% | +0.016% |
| 10 | 5.07% | 0.30% | +0.053% |
| 15 | 1.62% | 0.26% | -0.006% |

The structure comparison uses the same 0.25 kpc lateral grid and preserves
each component's total mass. Circular averaging also changes within-annulus
detail over 0.5 kpc. The residual sideways force of the circular case measures
discretization and boundary effects; it is not a physical current.

## Actual comparison with published motion

We also evaluated all six predeclared source cases against the 15 published
SPARC rotation points inside the predeclared 2–15 kpc model range. All 34
published points remain in the table; 19 outside that range have no prediction.
No gravity or mass-conversion parameters were fitted to these velocities.

| Conditional source case | Newtonian fractional RMS amplitude error | Full QUMOND fractional RMS amplitude error |
|---|---:|---:|
| nominal | 28.64% | 11.14% |
| uncovered_zero | 29.88% | 12.30% |
| low_conversion | 42.90% | 24.30% |
| high_conversion | 15.39% | 5.40% |
| thicker | 31.38% | 13.83% |
| axisymmetrized | 28.81% | 11.36% |

The nominal finer-grid comparison is **28.31% Newtonian versus
10.62% QUMOND**. This descriptive comparison improves
with QUMOND for every listed case, but it does not establish an acceptable
likelihood or select a correct mass model.

The SPARC curve assumes distance 6.6 Mpc and inclination 66 degrees; the independent
photometric source protocol uses 9.058 Mpc and 61.748 degrees. Published radii are
scaled by distance, and both sides are compared as projected rotation amplitudes
V sin(i). These are **not raw line-of-sight velocities**. Published annuli cannot
be exactly reconstructed at the changed geometry from this table. Bars, pressure,
warps, beam response and correlated errors are not modeled by this comparison;
its RMS is not a chi-square significance or a clean held-out prediction.

## Source and numerical evidence

The source builder reads cleaned stellar light, every nonzero ICA mask label,
THINGS HI moment zero and HERACLES signed CO plus its error map. It reads no
target rotation speed or dynamical mass. Native map blurring remains. The CO
publisher's integration window partly uses HI velocity information, so tracer
products are not fully independent of the kinematic observations.

The nominal tapered source contains approximately 41.92
billion solar masses in stars, 4.76 billion in atomic
gas plus helium, and 3.53 billion in molecular gas plus helium.
No dark-halo mass is added. Missing area is either left unfilled conditionally or
filled from observed annular means. Neither choice measures the missing matter.
Finite-cell coverage is estimated by pixel-center area assignment; native pixel
and beam sampling must be included in later source convergence.

Signed flux is averaged before nonnegative projection. The negative CO cells
removed by that projection amount to **6.9% of the signed measured CO integral**.
This is a visible noise-bias risk, not a correction known to be valid. CO errors
are retained as fully correlated within-cell bounds; no full source measurement
likelihood or complete missing-phase budget has been established. The source is
linearly tapered between 18 and 20 kpc; exterior mass is not inferred as zero.

Combined baryons are lifted with explicit exponential vertical profiles. The
solver applies the QUMOND nonlinear step to the combined Newtonian vector field
and solves the second Poisson equation, following the
[full-field formulation](https://arxiv.org/abs/0911.5464). It does not sum separately
boosted components. Stellar cleaning and masks follow the
[S4G product definitions](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).

Eleven full field runs were executed: six source cases and five numerical
controls. The nominal 0.5 kpc grid passed mean radial-force checks but failed the
stricter full-vector comparison. This failure is retained. Refining from 0.25
to 0.125 kpc reduces the vector difference to **0.579%
Newtonian and 0.510% QUMOND**, with the worst
individual ring below 0.9%. Vertical refinement changes the vector by under
0.3%; increasing the box from 24 to 32 kpc half-width changes QUMOND by about
0.20% aggregate, with the worst ring below 0.8%.

These controls establish numerical stability of this conditional interior
calculation. They do not identify the galaxy's real external field. Newtonian
boundaries include monopole, dipole and quadrupole; MOND uses the isolated
spherical monopole boundary. Missing nonspherical exterior terms are only
tested through the reported box change. No AQUAL control is completed here.

## Work remaining and replay

The next decisive test is whether a source-supported bar/streaming model predicts
the actual channel cube better than an ordinary rotating warped disk, using a
validated signal mask, instrument response and noise covariance. Stellar-map
calibration, mass/depth uncertainties and external fields must enter that test.
This one conditional galaxy does not satisfy the target of 10–20 validated
development pilots or the later 100–300 resolved sample. The 13,525 catalog
identity groups are not 13,525 resolved 3D models.

Raw sources remain private. This milestone is **local only**: the linked Git
metadata is outside the writable workspace, so fetch/commit/push are unavailable.
The CUDA environment and new shell downloads also remain unavailable.

- [Conditional motion predictions and excluded radii](conditional-motion-comparison.csv)
- [Conditional motion scores](conditional-motion-scores.csv)
- [Same-grid structure comparison](structure-comparison.csv)
- [Stellar/dust aperture check](stellar-dust-check.json)
- [Source audit and assumptions](../mond-atlas-source-001/source-audit.json)
- [Initial field convergence](../mond-atlas-field-001/field-audit.json)
- [Stricter vector convergence](../mond-atlas-field-002/vector-audit.json)
- [Verification](verification.json), [test log](validation.log), [outstanding work](execution-status.json)

Run with Python/NumPy from the repository root, choosing unused output folders:

```text
python scripts/build_mond_atlas_ngc2903_source.py --output work/gravity-first-principles/source-replay --private work/private/source-replay
python scripts/run_mond_atlas_ngc2903_fields.py --source work/gravity-first-principles/source-replay --output work/gravity-first-principles/field-replay --private work/private/field-replay --convergence
python scripts/check_mond_atlas_field_pattern.py --source work/gravity-first-principles/source-replay --previous work/gravity-first-principles/field-replay --output work/gravity-first-principles/vector-replay --private work/private/vector-replay
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```
