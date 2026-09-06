# NGC2976: executed conditional native selection and recovery

One bounded CPU milestone completed on 2026-09-06. **SOURCE_BLOCKED remains in
force: zero admitted observational cube likelihoods and zero new gravity or
motion scores.** Existing products, common modules, handoffs and Git metadata
were not edited. This galaxy and its earlier noise diagnostics were already
development-exposed; this is not a new galaxy holdout or retrospective
preregistration.

The immutable result is [run-001/summary.json](run-001/summary.json).
All 9 new unit tests and all declared analytic/numerical controls passed.
The run executed 864 actual-background injection trials at 12 fixed positions,
2,304 Gaussian-surrogate injection trials, 96 Gaussian noise cubes and 72
noiseless controls. It retained all three spectral branches and all 24 cases
per branch. No threshold or branch was selected using these outcomes.

## Source and product identity

[Walter et al. 2008](https://arxiv.org/html/0810.2125), sections III.5–III.6,
assign noise-based selection to standard, unrescaled cubes. Flux-rescaled cubes
are blanked with a master mask and supply the moment maps. Their detection
description uses a 30-arcsec convolution and three consecutive channels above
twice the noise. Primary-beam correction is applied to integrated intensity.
The observing table lists NGC2976 in mode 4AC, with 1.56 MHz, 64 nominal channels
and approximately 5.2 km/s spacing. These entries do not certify the exact
online spectral kernel. The
[contemporary NRAO instrument description](https://library.nrao.edu/public/pubs/obsstat/VLAOS_0302.pdf)
documents separate normal and Hanning correlator options.

The [public data page](https://things.www3.mpia.de/Data.html) contains 266
distinct links. Its NGC2976 entries provide natural/robust cubes and moment
maps; none links a mask, master, rescaled or blanked product. Directory listing
returned HTTP 403. This is a bounded archive audit, not proof that no other
public copy exists. No publisher mask was obtained.

The needed standard natural cube was already cached. Its full local SHA-256
was revalidated, its FITS structure checked with Astropy, and its primary
42 × 1024 × 1024 array found entirely finite, with no zero pixels in the source
identity audit. It includes 42 AIPS CLEAN-component table extensions. No FITS
CHECKSUM/DATASUM validation is claimed because those keywords are absent.
An HTTP 206 request for the first 40,320 bytes of the linked public cube exactly
matched the cached padded header hash and reported the same 191,615,040-byte
total length. This does **not** rehash the full remote array.

| Input | SHA-256 |
|---|---|
| [NGC2976 standard cube](https://things.www3.mpia.de/Data_files/NGC_2976_NA_CUBE_THINGS.FITS) | `e8ce711a354fcf9c76e7ba9afd9e55fa97df6169863d5473140f20f9c8250169` |
| [NGC2976 natural MOM0](https://things.www3.mpia.de/Data_files/NGC_2976_NA_MOM0_THINGS.FITS) | `b4f8b1cdc68b1b5724c9930385c760cc9d2fa1dbac7c994e28508467ccf76878` |
| Cached/public padded cube header | `6f39d945fb79751569d564d13d618c4f1a3c32ec289dfb2eb24114c046851a06` |

[source-evidence.json](source-evidence.json) records URLs, HTTP results,
successful payload sizes and hashes of the downloaded archive page, primary
paper HTML, NRAO PDF and header range. Successful new payloads total 11,087,146
bytes, below the 2,000,000,000-byte cap. Raw source material and derived pixel
arrays stay under `work/private/mond-atlas-native-selection-001/`, outside Git.

## Declared operator and scope

The FITS history provides a 7.407 × 6.42384 arcsec restoring beam at PA 71.79
degrees, a signed (north, east) pixel step of approximately (+1.5, −1.5) arcsec,
and stored radio-heliocentric velocity spacing −5.152666992 km/s. The direct
history maps stored channels 0–41 onto zero-based parent channels 11–52;
first-order continuum fitting used parent channels 5–11 and 52–56. Only parent
11 and 52 remain in the delivered cube. Historical selection is not an
independent certification that these planes contain no line emission.

The new operator convolves native pixels with a unit-sum elliptical Gaussian
whose covariance is circular 30-arcsec covariance minus restoring-beam
covariance. Beam PA uses north through east and the signs of the FITS axes.
Five-sigma finite support gives a 42-pixel image margin. Image edges are zero
extended and that entire margin is excluded from the selection mask. Working
units remain Jy per native restoring beam; sigma uses the same units. The
equivalent conversion to Jy per target beam cancels in the threshold ratio.

MOM0-positive spatial support is expanded by 120 arcsec and excluded from a
550–680 arcsec annulus about the FITS reference pixel. Fixed western pixels
calibrate each channel's median and Gaussian-scaled MAD; the eastern half is
held for descriptive validation, with a 90-arcsec east/west guard. There are
101,299 calibration and 100,339 validation pixels. These are correlated pixel
counts, not independent samples. MOM0 screening uses the same observation,
so this split is not independent selection or proof of absent gas.

After subtracting each calibration median, the operator selects the union of
all runs of at least three consecutive stored channels strictly above twice
that channel's MAD scale. No spectral wrap, extra dilation, hand editing or
coarse spatial rebinning is applied. Its exact sigma estimator and run-union
convention are declared choices; they are not a demonstrated replay of AIPS
master-mask creation.

Synthetic source profiles integrate intrinsic Gaussians over spectral cells,
then apply independent, full-Hanning, or alternate-channel Hanning response
matrices. The latter uses half-spacing input cells. The image-domain
continuum surrogate propagates both source and covariance with the
history-bound operator A. Noise covariance is A H Hᵀ Aᵀ. Restored source
emission uses the measured elliptical beam; Gaussian-surrogate noise uses a
white field filtered by that beam. That filter yields a noise autocovariance,
not a measured dirty beam. Actual-background trials retain the observed
noise, foregrounds and any undetected emission in their native patches.

The assumed top-hat spectral cells are not a recovered correlator passband.
Flags, visibility weights, dirty sidelobes, CLEAN nonlinearity, missing
spacings, primary-beam correction and residual scaling are not reconstructed.
The exact native covariance and spectral response remain unresolved.

## Executed results

Native median MAD is 0.35427 mJy/native beam in the west and 0.35373 in the
east. After smoothing, these values are 0.062186 and 0.061724 mJy/native beam.
The native lag-one correlations are 0.24664 and 0.24798; detector correlations
are 0.25464 and 0.27567. These are descriptive background moments.

The detector MAD is channel-dependent: stored channel 21, at −2.15267 km/s,
reaches 0.116238 mJy/native beam, 1.87 times the across-channel median. Channel
20 has 0.107254 mJy/native beam. No line-free certification or attribution to
foreground emission versus instrumental effects follows from these values.
This variation is retained in the empirical thresholds.

The reconstructed mask selects 762,743 voxels in 95,097 spatial pixels. It
overlaps 77,710 of 95,513 MOM0-positive pixels (81.36%). Conversely, 18.28% of
its selected spatial support lies outside positive MOM0 support. This compares
two projected supports only. It is neither a three-dimensional mask match nor
a false-detection classification. The selected validation voxel fraction is
0.08526%; the validation region is not certified pure noise.

Twelve injection centers were selected geometrically from 66 eligible lattice
points before the new selection pixel values were opened. Each 81 × 81 native
pixel patch avoids expanded MOM0 support, calibration pixels and convolution
edges. Several patches overlap. The 864 empirical trials are consequently
neither 864 independent backgrounds nor a population completeness estimate.

The injection grid spans intrinsic spatial FWHM 6 and 30 arcsec, spectral
FWHM 1, 3 and 6 stored channels, subchannel phase 0 and 0.5, and detector peak
amplitude 2 and 5 times the *across-channel median western MAD*. This last
quantity is not five times the local channel sigma. For example, a nominal
five-times-median peak in channel 20 is only about 2.90 times that channel's
MAD. The center was fixed at stored channel 20 before the new outcomes.

For the 30-arcsec intrinsic source, phase zero, and five-times-median amplitude:

| Spectral response | Intrinsic width (channels) | Peak selected at fixed positions | Mean known native source flux retained |
|---|---:|---:|---:|
| Independent boxcar | 1 | 0/12 | 0.163% |
| Independent boxcar | 3 | 4/12 | 15.474% |
| Independent boxcar | 6 | 6/12 | 21.268% |
| Full Hanning | 1 | 3/12 | 10.878% |
| Full Hanning | 3 | 5/12 | 16.304% |
| Full Hanning | 6 | 6/12 | 22.004% |
| Decimated Hanning | 1 | 0/12 | 0.307% |
| Decimated Hanning | 3 | 4/12 | 15.605% |
| Decimated Hanning | 6 | 6/12 | 21.373% |

All cases are in [case-summary.csv](run-001/case-summary.csv), with individual
trials retained. Known positive pre-continuum flux, signed post-continuum
flux, selected noisy flux, and paired selected-flux change from the uninjected
background are distinct metrics. For the first row above, the mean selected
noisy flux is 13.56% of the known injected flux, while the paired difference is
0.472%. Neither equals the 0.163% true-source retention. Flux integrals refer
to the finite injection patch and modeled restored emission; they are not
observed galaxy mass corrections.

The separate stationary Gaussian controls normalize each branch to the same
mean detector variance, propagate the complete conditional continuum
covariance, and use their own known per-channel marginal sigma. They do not
copy the empirical channel-scale spikes.

| Gaussian spectral branch | Mean noise voxel fraction selected | Monte Carlo standard error across 32 cubes |
|---|---:|---:|
| Independent parent channels | 0.01170% | 0.00468 percentage points |
| Full Hanning | 0.38711% | 0.05342 percentage points |
| Decimated Hanning | 0.03924% | 0.01298 percentage points |

Mean realized detector RMS/target scale is 1.0023, 1.0010 and 0.9988,
respectively. These are finite conditional simulations, not calibrated
observational false-alarm probabilities. The independence branch still
contains nonlocal covariance from continuum subtraction, explaining why it
must not be equated with the earlier independent-output-channel experiment.

## Verification, files and reproduction

[controls.json](run-001/controls.json) records direct convolution error
9.02e−17, separately implemented least-squares error 8.88e−16, polynomial
annihilation error 2.50e−16, beam covariance relative errors below 7.35e−6,
and 5-to-6-sigma kernel L1 changes below 1.07e−6. Independent spectral
quadrature error is 8.88e−16. The compact Gaussian integral agrees under 1×,
2× and 4× pixel refinement. Three 40,000-draw covariance checks have relative
errors 0.02889, 0.02052 and 0.02773, all below the frozen 0.04 tolerance.
Unit tests independently exercise direct convolution, reflection, flux
conservation and units, threshold equality, missing values, mask support,
run adjacency without wrapping, and analytic Hanning covariance lags.

The 18 run-result files in the run manifest and all 20 prospective input
bindings were rehashed after execution without mismatches. The package
manifest additionally binds this report and source evidence. No failed
scientific gate was discarded, no frozen test changed, and no new observed
gravity response was opened. Repository-wide integration tests and publication
remain with the coordinating task.

New implementation files, relative to the Invariant repository root:

- `scripts/mond_atlas_native_selection.py`
- `scripts/run_mond_atlas_native_selection.py`
- `configs/mond_atlas_native_selection_v1.json`
- `tests/test_mond_atlas_native_selection.py`

Public reports and tables are confined to
`work/gravity-first-principles/mond-atlas-native-selection-001/`.
Raw/private new files are confined to
`work/private/mond-atlas-native-selection-001/`.
See [package-manifest.json](package-manifest.json) for the exact file inventory.

Executed from
`C:\Users\henry\Documents\Codex\2026-09-04\pu-2\work\Invariant`:

```powershell
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests -p test_mond_atlas_native_selection.py -v
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B scripts/run_mond_atlas_native_selection.py --output work/gravity-first-principles/mond-atlas-native-selection-001/run-001 --private-output work/private/mond-atlas-native-selection-001/run-001
```

Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1 and the existing Astropy environment
were used with two BLAS threads and no GPU. Reproduction requires the bound
source cache and acquisition evidence. Supply new output and private-output
directories: the runner refuses existing directories. Do not overwrite run-001.

Next admission work requires a validated publisher channel mask or a separately
validated selection alternative, native response/passband and covariance
constraints, independent line-free/background support, and a source flux
likelihood including residual scaling and primary-beam effects. This milestone
supplies an executed conditional step toward those requirements.
