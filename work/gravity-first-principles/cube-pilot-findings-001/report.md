# Motion and gas structure: first conditional cube pilot

The extra motion components explain some previously unexplained spectra, but they do not supply a consistent explanation across all 12 galaxies. After accounting for those components, the covered-gas comparison selected **no additional gas-structure correction in all eight eligible galaxies**. This does not establish a gravity law or reject small corrections, other formulas or a low-density coherence mechanism.

## What was actually tested

The 5090 fitted the intensity in every velocity channel of a coarsened HI cube, comparing five models: regular circular rotation; rotation plus an outer orientation warp; rotation plus radial streaming; rotation plus a broad lagging spectral component; and all components together. Sixty fits were run; 57 converged. The source brightness was supplied from the same observations. These are conditional predictions of withheld spatial spectra, **not independent observing data and not a full physical 3D disk reconstruction**.

Rotation is allowed five radial coefficients. Warps can change outer position angle by 15 degrees and inclination by 8 degrees. Streaming has two radial coefficients bounded by 30% of the initial speed scale. The asymmetric component has a flux fraction up to 0.4, lag up to 0.6, and width 1.7 times the main profile. These bounded phenomenological choices are not an exhaustive inventory of conventional gas dynamics. Their source is frozen in the registered runner.

The scoring mask is geometric: an ellipse between deprojected radii 48 and 450 arcsec, with fixed alternating 192-arcsec spatial blocks and approximately 120-arcsec guard gaps. All channels are retained, so a surprising measured speed cannot cause a position to be selected. This validates response-independent selection for this aperture; it does not recover the survey's original detection mask or make this a representative sample of all gas.

Channel-noise covariance uses channel-dependent variances and six tapered correlation lags, estimated in a separate outer sky annulus. Disjoint background pixels passed the predeclared broad variance/correlation gate in all 12 objects. Their median whitened variances range from 0.52 to 0.99; NGC5055 and NGC6946 are near the lower gate, showing important background nonstationarity. Spatial covariance is not fully modeled, so the losses are comparative scores, **not calibrated chi-square statistics or significance levels**.

## Where the motion models helped

Of the 11 converged combined fits, seven improved withheld spectra and four worsened them. The largest apparent gains occur in DDO154, NGC3198 and IC2574. These are reductions in a channel-intensity prediction error, not percentages of missing gravity explained.

| Galaxy | Combined withheld loss | Change from rotation | Combined converged |
|---|---:|---:|---|
| DDO154 | 23.08 | +73.8% | yes |
| IC2574 | 11.14 | +54.8% | yes |
| NGC2841 | 2.71 | -13.7% | yes |
| NGC2903 | 2.99 | +14.2% | yes |
| NGC2976 | 2.26 | -8.6% | no |
| NGC3198 | 18.85 | +62.0% | yes |
| NGC3521 | 27.59 | +29.4% | yes |
| NGC4214 | 9.88 | +20.4% | yes |
| NGC5055 | 3.03 | -37.5% | yes |
| NGC6946 | 1.82 | -52.3% | yes |
| NGC7331 | 4.96 | -4.7% | yes |
| UGC04305 | 3.27 | +7.5% | yes |

For DDO154 and IC2574, streaming alone predicts withheld spectra better than the combined model. That is a warning about extra flexibility and geometry/motion degeneracy. It does not prove real radial flows. NGC3198 benefits strongly from the combined model. NGC5055 and NGC6946 worsen with extra components. Many final losses remain far above the nominal whitened-noise floor, so substantial model deficiencies remain.

NGC2976's combined fit reached its iteration limit; NGC6946's asymmetric fit and UGC04305's warp fit encountered optimizer failures. These three galaxies were excluded from the stricter gas-term comparison. NGC7331 additionally had no positions meeting the broad gas-coverage rule. Failed fits are retained rather than interpreted as physical exclusions.

## The corrected gas test

The first pilot gas diagnostic incorrectly allowed blanked MOM0 zeros to enter the surroundings descriptor. Those scores are **inadmissible for the gas hypothesis** and retained only for audit. The corrected experiment normalizes smoothing by measured support and requires at least 98% support at both 48- and 96-arcsec widths. It describes the detected-emission domain, not voids. A failed stellar WCS coverage attempt was also retained; the successful replacement marks nonconvergent map coordinates uncovered.

The tested term is `v_rotation -> v_rotation * (1 + beta*C_HI)`. `C_HI` is a bounded broad/local HI contrast with its radial mean removed. That focuses this test on differences around a ring; the flexible baseline rotation curve can already absorb a purely radial enhancement. Candidate beta values run from -0.30 to +0.30 in steps of 0.05. Other galaxies' training spectra select beta, and the target galaxy's separate test spectra score it. All baseline nuisance parameters are frozen.

All eight eligible galaxies selected beta=0. Thus this particular descriptor supplied no transferable addition at the tested grid spacing. The result does not exclude smaller coefficients, improvements after joint refitting, a different measured source descriptor, or a genuinely derived field law. The target galaxy still supplies its baseline kinematic calibration, so this is not prediction of an entirely unseen galaxy without local fitting. All galaxies have been exposed during earlier development.

## Total matter remains unresolved

53 stellar and CO assets were acquired with source URLs, file sizes and hashes. Stellar light, molecular gas and atomic gas must all enter a mass model; neither HI intensity nor its holes is total volume density. The table is a footprint audit at selected sky positions, not a matched-beam mass reconstruction. CO nondetections use the supplied interpolated error map only as a screening indicator; correlated uncertainties have not been propagated into a significance claim.

| Galaxy | Stellar coverage | CO with positive error | Covered CO nondetection positions |
|---|---:|---:|---:|
| DDO154 | 63% | 15% | 46 |
| IC2574 | 100% | 58% | 230 |
| NGC2841 | 99% | 34% | 45 |
| NGC2903 | 100% | 79% | 181 |
| NGC2976 | 62% | 20% | 72 |
| NGC3198 | 87% | 35% | 68 |
| NGC3521 | 100% | 36% | 78 |
| NGC4214 | 26% | 17% | 91 |
| NGC5055 | 100% | 76% | 150 |
| NGC6946 | 99% | 100% | 366 |
| NGC7331 | 100% | 60% | 103 |
| UGC04305 | 93% | 16% | 71 |

CO coverage ranges from about 15% to 100% of these positions. Covered nondetections remain upper-limit information, not zero molecular mass. Stellar foreground masks, mass-to-light uncertainty, the CO-to-H2 conversion, beam matching and line-of-sight depth still require validation. We have no calibrated total 3D density map from this pilot.

## What would make the next result convincing

1. Replace the coarse projected approximation with a validated tilted-ring cube forward model, including native beam velocity mixing, disk thickness and the actual spectral response. Current additional smoothing is about 48 arcsec and native source-beam effects remain approximate.
2. Test nuisance-model adequacy and identifiability with mismatched simulations, not only injections generated by the fitting model. Four matching-model controls recovered withheld synthetic spectra near the noise floor, but that establishes numerical behavior rather than unique physical causes.
3. Build a common-resolution stellar + HI + molecular mass model with propagated conversion and nondetection uncertainties. A freely fitted rotation curve is useful for separating motion, but a constrained mass model is necessary to test gravity.
4. Lock the observable descriptor and compare predictions on new galaxies or independently acquired data. Preserve the geometric mask and evaluate both spectra and well-defined motion diagnostics without tuning to the test set.

Primary methodological references: [3D-Barolo cube fitting](https://arxiv.org/abs/1505.07834), [THINGS processing and kinematics](https://arxiv.org/abs/0810.2125), and [asymmetric HI profiles in NGC3521](https://arxiv.org/abs/1312.2399). Asset-level observational URLs and hashes are in the acquisition receipts.

## Reproduction and evidence

Use Python 3.13 with NumPy, SciPy, Astropy and PyTorch 2.7.1+cu128. The private CUDA environment is `work/private/torch-cuda-env`; raw data and prepared cubes are deliberately excluded from Git. Immutable directories retain registrations, exact runner copies, numerical controls, every galaxy fit, audit results and failures. The old pilot gas scores must not be promoted. The current reusable pilot script removes that legacy scoring; use the separate coverage audit.

The active first-principles gravity goal remains open. No new law is admitted by these results.
