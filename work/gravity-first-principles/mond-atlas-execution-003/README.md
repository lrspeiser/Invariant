# MOND atlas — source and noise readiness update

**The atlas is still in progress. This milestone clears two preliminary checks
for ten of the twelve pilot galaxies; it does not yet deliver a full 3D gravity prediction.**

## A substantial correction to the previous assessment

The earlier registration test treated every Gaia star inside the rectangular
image boundary as a usable reference. The mosaics contain blank corners and
other unexposed areas. **283 of the 816 tested Gaia positions lacked finite image
support.** The test assigned these positions large mismatches, making good
images look misregistered.

We kept the previous calibration/validation assignments, the previously selected
plain-TAN coordinates, and the original accuracy thresholds. We required a
finite 7×7 image patch at each Gaia position; measured zero flux remains valid.
No translation, rotation, or gravity parameter was fitted.

**Eleven images now pass the same strict thresholds; only NGC4214 lacks enough
supported stars.** This corrects the earlier claim that eleven registrations
needed repair. The old star-distance computations were reproduced to a maximum
difference of 1.54e-10 arcsecond, cross-checking
the new restricted FITS/TAN implementation against the prior Astropy/SciPy outputs.
Independent vector-projection, FITS scaling/blanking, and roundtrip tests also pass.

This is a development repair of a faulty test. It is not new independent sky
data, a complete exposure-weight calibration, or validation of the cleaned
stellar maps. P1/IRAC flux coordinates must still transfer correctly to P5.
NGC2903 has an earlier validated P1/P5 reconstruction; that transfer has not been
established for the other four P5 galaxies. P5 masks still exclude every nonzero
label, including negative labels, as the [publisher specifies](https://irsa.ipac.caltech.edu/data/SPITZER/S4G/docs/P5_README.html).

## We also fitted spatial noise correlations to real cubes

Channel-only noise models miss the fact that neighboring pixels share noise
after beam smoothing. We fitted a channel covariance and a spatial correlation
model using background regions 550–680 arcseconds from each galaxy center.
Calibration and validation use separate guarded blocks; validation is subsampled
to avoid a nearly singular covariance from adjacent, heavily smoothed pixels.
The minimum calibration-to-validation separation is recorded for each galaxy.

The first checkerboard attempt starved some sky quadrants of validation samples.
Its outputs are retained in noise run 001. Run 002 assigns blocks using coordinates
alone so every quadrant is represented, with the same noise thresholds. All
means, variances and correlation parameters are then refitted from calibration
pixels only. A test that multiplies held-region noise by four leaves all fitted
covariances unchanged and correctly fails the validation gate.

In the balanced run, **eleven galaxies pass the preliminary background moment
checks**. NGC3198 retains channel correlation 0.176, above the declared 0.15
threshold. NGC2841 failed this channel test in the earlier unbalanced split but
passes the balanced split: the failure identity is split-dependent, which must
be investigated rather than declared an intrinsic property of either galaxy.
Passing these broad gates does not establish exact Gaussian/separable noise,
the mask within the emitting galaxy, or a calibrated chi-square significance.

| Galaxy | Supported validation stars | Median Gaia offset (arcsec) | Stellar flux coordinates | Held joint-noise mean square | Channel lag-1 | Noise diagnostic |
|---|---:|---:|---|---:|---:|---|
| DDO154 | 4 | 0.442 | pass | 1.085 | +0.043 | pass |
| IC2574 | 22 | 0.341 | pass | 1.191 | +0.049 | pass |
| NGC2841 | 15 | 0.257 | pass | 0.940 | -0.068 | pass |
| NGC2903 | 30 | 0.141 | pass | 0.951 | -0.072 | pass |
| NGC2976 | 20 | 0.304 | pass | 1.096 | +0.032 | pass |
| NGC3198 | 9 | 0.257 | pass | 1.202 | +0.176 | fails channel correlation |
| NGC3521 | 13 | 0.235 | pass | 1.006 | -0.004 | pass |
| NGC4214 | 3 | 0.401 | too few stars | 0.971 | -0.019 | pass |
| NGC5055 | 20 | 0.234 | pass | 1.073 | -0.019 | pass |
| NGC6946 | 54 | 0.338 | pass | 0.979 | -0.020 | pass |
| NGC7331 | 39 | 0.305 | pass | 0.867 | -0.084 | pass |
| UGC04305 | 33 | 0.329 | pass | 0.968 | -0.047 | pass |

“Mean square” would be near one under a correctly normalized noise model. It is
a diagnostic here, not a gravity fit. The overlap clearing both preliminary
checks is **10 galaxies**. No stellar mass, rotation speed, gas descriptor or
gravity formula was adjusted to obtain those passes.

## Five real duplicates resolved

Five pairs of MaNGA identities share the same positive NSA catalog ID, the same
IAU source name, and sub-arcsecond coordinates within the same input release.
We now group each pair as one object, with exact source-row evidence. This brings
the working grouping from 13,530 to **13,525 object groups**, with **90 proximity
pairs still unresolved**. This remains an uncertified count of distinct physical
galaxies. Original observations are preserved, and future holdouts must use the
group identity so repeated observations cannot leak across train/test splits.

## Next required work

The critical remaining work is total-matter mapping and the actual full-field
motion prediction: cleaned-map transfer, conversion uncertainties, masked regions,
shared stellar/atomic/molecular coverage, allowed spatial-depth structures,
exterior fields, gas support/streaming and instrument response. The local 3D and
cube engines are foundations for that work. The catalog and radial findings from
the [previous milestone](../mond-atlas-execution-002/README.md) remain available;
its registration-readiness statement is superseded by this audit.

The existing CUDA environment, shell downloads and linked-worktree Git writes
remain unavailable under this session's permissions. This work ran on bundled
CPU Python/NumPy. **This milestone has not been committed or pushed to main.**

## Evidence and replay

- [Updated pilot readiness](pilot-readiness.csv)
- [Astrometry source hashes](astrometry-source-assets.csv)
- [All 816 star positions and footprint decisions](../mond-atlas-astrometry-001/stars.csv)
- [Astrometry summary](../mond-atlas-astrometry-001/summary.json)
- [Balanced real-cube noise summary](../mond-atlas-noise-002/summary.json)
- [Object grouping overlay](../mond-atlas-identity-001/object-groups.csv)
- [Exact duplicate evidence](../mond-atlas-identity-001/merge-evidence.json)
- [Verification](verification.json) and [28-test log](validation.log)
- [Full outstanding work](execution-status.json)

From the repository root with Python and NumPy; choose new output directories:

```text
python scripts/run_mond_atlas_astrometry.py --output work/gravity-first-principles/mond-atlas-astrometry-replay
python scripts/resolve_mond_atlas_identities.py --catalog work/gravity-first-principles/mond-atlas-catalog-004 --output work/gravity-first-principles/mond-atlas-identity-replay
python scripts/run_mond_atlas_noise.py --output work/gravity-first-principles/mond-atlas-noise-replay --private work/private/mond-atlas-noise-replay
python tests/test_mond_atlas_astrometry.py
python tests/test_mond_atlas_noise.py
```
