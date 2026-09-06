# MOND atlas: recover the native spectral processing

**The original continuum-subtraction choices can be traced directly for nine
galaxies. Eight retain 29 candidate continuum channels in the released cubes.**
The selected native planes now reproduce our earlier smoothed cube planes
exactly, including their old background offsets. This closes a concrete part
of the instrument-processing chain; it does not validate a galaxy gravity fit.

The preceding report showed why a fixed sky annulus was not reliable as pure
background. The new check uses the observing pipeline's historical channel
choices instead of choosing apparently quiet channels from our current residuals.
These are historical continuum-fit candidates, not certified emission-free data.

## What was recovered

The FITS HISTORY records contain the channel weights used by UVLIN, its polynomial
order, and the IMAGR channel range and averaging settings. The
[AIPS imaging documentation](https://www.aips.nrao.edu/CookHTML/CookBookse50.html)
defines those range and increment controls. Direct mapping is accepted here only
for one unambiguous UVLIN/IMAGR chain, matching dataset identifiers, no channel
averaging, the expected output channel count, and complete single-IF weights.

DDO154's output interval contains none of its recorded continuum-fit channels.
That does not prove that every output channel contains HI. NGC2841 and NGC3521
were assembled from separate spectral cubes. NGC7331 combines histories with
different continuum fits. Those mappings remain unresolved rather than being
assigned the last header's fit across the whole cube.

The [THINGS measurement paper](https://arxiv.org/abs/0810.2125) distinguishes its
standard cubes for noise-dependent selection from rescaled, blanked products for
flux measurements. It describes continuum removal using selected channels and
emission detection after spatial smoothing. These processing distinctions matter
when building both baryonic source maps and motion likelihoods.

## A spatial comparison on the retained candidate channels

The native robust scale is Gaussian-normalized median absolute deviation (MAD),
measured inside 300 arcsec and at 550–680 arcsec from the FITS reference pixel in
the image projection plane. Native pixels are sampled every fourth position.
Each row below gives the median across that galaxy's available candidate channels.
The coarse comparison uses the same channel identities after the original
additional smoothing and 8×8 block average.

| Galaxy | Stored channels | Retained historical candidates | Native center/outer MAD | Smoothed center/outer MAD |
|---|---:|---:|---:|---:|
| DDO154 | 57 | 0 | — | — |
| IC2574 | 83 | 2 | 1.006 | 1.122 |
| NGC2841 | 132 | unresolved | — | — |
| NGC2903 | 87 | 6 | 1.030 | 1.162 |
| NGC2976 | 42 | 2 | 0.979 | 0.972 |
| NGC3198 | 72 | 4 | 1.028 | 1.120 |
| NGC3521 | 109 | unresolved | — | — |
| NGC4214 | 102 | 8 | 1.011 | 1.135 |
| NGC5055 | 87 | 3 | 1.025 | 1.039 |
| NGC6946 | 115 | 2 | 0.973 | 1.101 |
| NGC7331 | 116 | unresolved | — | — |
| UGC04305 | 54 | 2 | 0.969 | 1.044 |

Native medians span 0.969–1.030. Several smoothed medians are around 1.10–1.16.
This is a descriptive spatial-scale pattern, not a significance calculation.
Smoothing reduces the number of independent measurements and can expose weak
extended emission or correlated instrumental structure. The small candidate
channel counts, same-observation dependence and correlated spatial samples do
not support assigning the difference to any one cause yet.

No numerical pass threshold was selected for these scale ratios. The next test
must simulate finite-sample stationary noise through the same processing and
measure recovery of injected signals before treating these differences as
evidence of nonstationarity or emission. Agreement in these candidates does not
validate every other channel or the full spatial/channel covariance.

## The preprocessing check is exact on its stated scope

Twenty-nine candidate planes were read from the hashed native cubes, filtered
with the recorded extra Gaussian using zero extension and float32 intermediate
planes, block-averaged, and given the original per-channel offset subtraction.
All 29 reproduce the cached planes with maximum absolute difference **zero**.
Independent direct-convolution, impulse-flux, centroid and block-flux tests check
the new implementation. This numerical equality is specific to these planes;
it is not a claim that the old physical beam approximation was complete.

The first replay stopped after eight successful planes because its declared
spectral contract only accepted FELO-HEL headers. The repair adds the native
radio-velocity and frequency formats already present in the inputs, using only
their monotonic index ordering. The [spectral FITS reference](https://arxiv.org/abs/astro-ph/0507293)
distinguishes those coordinate types. The original incomplete run and its
[failure receipt](../mond-atlas-preprocessing-replay-001/execution-failure.json)
are retained. No smoothing threshold, selected channel or replay tolerance changed.

The old extra circular smoothing does not make a native elliptical beam exactly
circular. Intensity units remain Jy per native beam. A restoring beam is also
not a complete description of interferometer noise. Those limitations remain.

## Continuum removal introduces covariance beyond neighboring channels

[AIPS continuum subtraction](https://www.aips.nrao.edu/CookHTML/CookBookse49.html)
fits the real and imaginary visibility spectra and subtracts the fitted baseline.
We implemented a conditional linear-algebra control: for a supplied spectral
covariance C and weighted polynomial-removal operator A, the residual covariance
is A C Aᵀ. This includes covariance from the uncertain subtracted baseline and
its cross terms with any channels that helped estimate it.

For example, subtracting the mean of m independent calibration channels from
different independent output channels gives covariance I + 11ᵀ/m. Subtracting
that mean from the calibration channels themselves instead gives I − 11ᵀ/m.
Those exact limits show why calibration channels and other channels do not have
identical noise after baseline removal. A short-lag-only covariance can miss
this structure even when the input noise had no long-range correlations.

The nine directly mapped histories were each propagated under independent and
three-tap smoothed unit-variance input-noise controls: **18 conditional cases**.
Polynomial annihilation, a separate weighted least-squares calculation, the exact
constant-fit limits and 60,000 simulated correlated draws verify the algebra.
The actual visibility weights, flags and nonlinear imaging were not replayed;
these cases are not measured covariance models or new galaxy motion scores.

## Atlas status and reproducibility

Current scale remains 13,525 identity groups (not certified distinct galaxies),
175 radial baseline galaxies, 126 passing its descriptive cuts, 12 resolved seed
galaxies, 22 source-image fits and 29 conditional field runs for one galaxy.
**Zero full-field galaxy cube likelihoods are admitted.** The full goal remains
active. The previous fixed-exclusion background result remains two galaxies
passing all splits; the new diagnostic does not promote other objects to that gate.

The atlas suite passes **78 tests**. This report rehashes all 432 files
of the preceding publication manifest before updating the handoff. All new
packages declared SOURCE_BLOCKED before implementation and did no new galaxy
motion fitting. Raw cubes and large fields remain outside Git. The header-text
extracts, source hashes, exact settings, per-channel diagnostics and conditional
covariance controls are retained in the linked stage directories.

```text
python scripts/run_mond_atlas_native_spectral.py --output <new-native-directory>
python scripts/run_mond_atlas_preprocessing_replay_v2.py --output <new-replay-directory>
python -m unittest discover -s tests -p "test_mond_atlas*.py" -v
```

Use new output paths. Configuration inputs deliberately bind the original stages;
replaying one stage does not silently redirect downstream inputs. Inspect
[pilot readiness](pilot-readiness.csv), [execution status](execution-status.json),
[verification](verification.json) and the [publication manifest](publication-manifest.json).

Publication remains local. The previous GitHub blob write required approval,
which the session policy does not permit. Local linked Git metadata is also
outside the writable root. No alternative write was attempted or claimed.
