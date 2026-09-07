# Execution033: alternative-source spectral transport and uncertainty

The broader gravity goal remains active. This milestone completes alternative-map spectral transport and investigates numerical/noise uncertainties. It does not score the observed source-region spectra or identify a preferred gravity law.

## What changed

Six positive HI emission packets represent common30arcsec smoothing, missing-region zero fill and annular fill at two planar resolutions. Their HI mass and flux agree with their source integrals to floating-point precision. Sixteen projected caches include the two stellar heights where available, two planar resolutions and two vertical quadratures, preserving all source flux.

All21600 alternative-source spectral convergence comparisons pass across144 physical cases. Maximum profile difference is0.4393%; maximum centroid difference is0.01656km/s. At fixed nuisance parameters, median spectral differences from the original source are22.74%/25.24% for thin/thick common30 smoothing,0.0689% for missingzero and0.0127% for missingannular. These combine changed emitted brightness, gravity and pressure; they are not isolated gravity effects or observed residual improvements.

Common30 introduces central emission below the force table's0.05kpc radius as well as outer emission beyond6.025kpc. The central contribution dominates the unsupported flux reaching the selected apertures. The conservative unknown-emission bound reaches0.04728mJy/beam at the maximum permitted amplitude. It is retained rather than assigned an invented speed. All16 caches and288 envelope cases were independently replayed. These source alternatives are conditional reconstructions, not unique3D maps or a native-beam deconvolution.

## Why the old centroid failures occurred

The42 retained signed-centroid failures are reproduced. Continuum subtraction creates positive and negative spectral values; their net integral can nearly cancel. All60 affected source-resolution profiles in the diagnostic have signed centroids outside the stored velocity window. A prospective check of all192 original cases gives zero failures across14400 comparisons for the positive precontinuum centroid, with maximum difference0.01807km/s. The signed spectral profile differences also remain small. The positive centroid describes the captured window, which can contain only81% of the line flux, not the entire emitted line. Original gates and failure records remain unchanged; this provides evidence for a physically meaningful future diagnostic, not a retroactive pass label.

## Pressure-estimator comparison

The alternative pressure code uses a sampled Gaussian and finite differences, while the earlier baseline uses the analytical Gaussian derivative integral. Recomputing both on identical raw alternative profiles exactly reproduces the exported derivatives. Their aperture-region pressure-acceleration relativeL2 differences are0.00254–0.00475%, much smaller than the source-change spectra. At radius0.05kpc, the missing-fill raw gas column is very small and the absolute difference reaches69.86(km/s)^2/kpc; this remains recorded. Independent source/interpolation/integration replay verifies the comparison. No estimator or physical parameter was retuned.

## Noise sensitivity

The noise-transfer sweep refitted all48 noisy injection trials with29 western leave-one-core-out covariance/mean variants:1392 fits, all identifiable. Maximum systemic/line-width shifts were0.01996/0.02081km/s; maximum amplitude shift was0.002338 (0.234 percentage points of unit normalization). Median amplitude shifts were0.038–0.041 percentage points. Held discrepancy q/channel shifted by at most0.02471. Dense-solve quadratic replay agrees to1.12e-15. These correlated sensitivity alternatives are not independent experiments or posterior uncertainty estimates, and do not test arbitrary source-region noise mismatch.

Independent replay also verifies all1392 fits and4176 start records, prediction hashes, phase order, parameter shifts and aggregate extrema; maximum dense-solve score discrepancy is6.67e-16.

## Outstanding work

Complete admission of a declared observed-spectrum comparison with the full source alternatives, material conversions, both spins, instrumental branches and explicit unknown-emission intervals. Numerical checks must cover fitted nuisance values, and the noise covariance remains a working marginal approximation rather than a validated whole-cube likelihood. Distinct refraction, current and time/memory mechanisms still need observed tests and theoretical/transfer checks. Lensing, clusters and Solar-System validation are not complete.

All2233 preceding manifest entries were verified unchanged before this work. New source/cached raw arrays remain private; scripts, derived diagnostics and their limitations are published.
