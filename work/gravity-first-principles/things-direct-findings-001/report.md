# Direct gas-to-motion test: first multi-galaxy result

The new direct gas measurements do not reproduce the earlier gas-force-proxy lead reliably. We now have substantially better observational inputs, including unblanked spectral cubes, and a tested way to compare real projected gas structure with measured motions. This particular gas-surroundings predictor is not ready to become a coherence formula.

## Data acquired and checked

All 12 galaxies in the pre-existing source selection now have natural and robust integrated HI brightness, velocity and velocity-dispersion products: 72 FITS maps. All 12 natural-weighting spectral cubes were also acquired, totaling 4.52 GB of cube files plus about 0.31 GB of maps. The exact download URLs, file hashes, dimensions, units, beams and blanking records are retained locally. Large FITS inputs remain outside Git; acquisition manifests and analysis evidence are versioned.

The current publisher index omits IC2574. Its exact historical official map URLs remain accessible and were used with their provenance recorded. Initial index-validation failures were retained before any analysis; subsequent successful acquisition contains all 72 maps.

The standard cubes contain the noise outside detected emission. The published survey distinguishes these from flux-rescaled products used for moment maps. We use the standard cubes to audit noise and detection support, and retain the official moment maps for brightness and motion measurements. Standard-cube flux is not silently substituted for rescaled flux. [THINGS measurement paper](https://arxiv.org/html/0810.2125v1)

## What we can trust more now

Measured per-channel noise ranges from 0.344 to 0.943 mJy/beam. Across the twelve cubes, measured noise is 0.92-1.10 times the published survey value. This is a useful check that these are usable noise-bearing cubes, not zero-filled images masquerading as measurements. It does not validate every individual velocity estimate.

We reconstructed a detection mask using the documented recipe: smooth to 30 arcsec FWHM and require emission above twice the measured smoothed noise in three consecutive channels. The resulting sky support overlaps the official support imperfectly. Intersection-over-union ranges from about 0.50 to 0.88; NGC7331 has the largest mismatch. The reconstruction is a diagnostic, not a replacement for the original channel mask or calibrated upper limits.

Neighboring channel noise is correlated, with measured lag-one correlations roughly 0.20-0.40 in clipped background samples. Channels and neighboring sky pixels must not be treated as independent measurements. The released velocity-dispersion map measures gas line width, not uncertainty in its mean velocity. Consequently this experiment does not report noise-calibrated chi-square or discovery significance.

The cubes have two sky coordinates plus a velocity axis. They do not provide a unique spatial depth coordinate or a complete 3D mass distribution.

## Exactly what was predicted

At matched locations on opposite sides of a galaxy, take half the absolute difference in the measured line-of-sight velocities. This removes the systemic recession velocity without fitting a target-galaxy offset or choosing its approaching side. It measures the antisymmetric component of gas motion. Warps, radial motions, asymmetric line profiles and other noncircular flows can affect it; it is not automatically a circular speed or gravitational acceleration.

Both observations and the projected RAR reference are intensity-weighted and smoothed to a common 20 arcsec Gaussian width. The source-only baseline uses published SPARC stellar/gas force components; no SPARC observed rotation speeds are used as predictor values. For five objects, geometry comes from photometry with an assumed intrinsic thickness ratio. Other published metadata include kinematically inferred inclinations and position angles, so those predictions are conditional on previously inferred geometry.

Added gas features come directly from HI brightness: broad/local contrasts at Gaussian widths 40/20 and 80/20 arcsec, plus brightness asymmetry between opposite locations. These are overlapping projected averages rather than physical shell masses or counts of voids. Local comparator inputs include local HI brightness, source-model quantities, radius, projection angle and aperture coverage at all scales. A complete 12-object radial surface-brightness supplement is unavailable in the development package, so the comparator uses a labeled stellar force-component scale rather than inventing missing brightness measurements.

## Sample and validation

Quality cuts were frozen before scoring: both sides must have sufficient detected-emission support, adequate source-profile radial coverage, a projected reference speed above 10 km/s, and be away from the projected minor axis. Samples are taken on a fixed grid and must remain usable across both processing products and all seven source/geometry scenarios.

Ten galaxies retain 868 opposite-side pairs. NGC2976 has only three usable pairs and NGC7331 none under the common-coverage rule; both remain in the acquisition/noise audit but fail the predeclared minimum of ten pairs for prediction. They were not removed for poor fit scores. The admitted sample is DDO154, IC2574, NGC2841, NGC2903, NGC3198, NGC3521, NGC4214, NGC5055, NGC6946 and UGC04305.

Three algorithms were tested: ridge regression, shallow boosted trees and GPU random-feature kernel regression. Each holds out one entire galaxy in turn. Three inner galaxy folds select regularization or tree complexity from fixed grids. Seven scenarios cover nominal/lighter/heavier stars, inclination +/-5 degrees and position angle +/-5 degrees. These are sensitivity brackets, not probability distributions. With two processing versions and two input groups, the experiment contains 84 model runs and 42 added-gas comparisons.

The same 868 locations are reused for all comparisons. Pixels, pairs, processing versions and scenario variations are not independent new galaxies. All are project development tests, not pristine confirmation data.

## Main result

Adding measured gas context improves fractional squared error over the otherwise identical local model in **12/42** comparisons. Under nominal source and geometry assumptions, it improves only **one of six** algorithm/processing comparisons.

| Algorithm | Natural weighting: extra gain | Robust weighting: extra gain |
|---|---:|---:|
| Linear ridge | -6.24% | -4.04% |
| Boosted trees | -1.30% | -0.94% |
| GPU kernel model | +11.23% | -1.89% |

Positive means lower error after adding gas surroundings. The GPU natural-weighting improvement is relative to a poorly performing local kernel model: even the improved version remains worse than the projected RAR reference. Its improvement does not survive switching to the robust-weighting product. In all six nominal added-gas models, squared km/s error is worse than the projected RAR reference.

This is a more direct test than the previous 139-galaxy gas-force-descriptor experiment. The sample, motion target and spatial descriptors differ, so it does not mathematically refute that older statistical finding. It does prevent treating the old finding as established evidence that measured diffuse surroundings strengthen gravity.

The score CSV includes 5,000 paired whole-galaxy bootstrap resamples of fixed outer predictions. They do not repeat model selection or earlier project choices and should not be treated as full uncertainty intervals. With only ten galaxies, a favorable isolated score is weak evidence.

## Verification

Synthetic controls verify constant preservation, systemic-velocity cancellation, linear velocity-field convolution and WCS round trips. Every model predicts each admitted galaxy only from the other galaxies and uses the identical admitted location set. All predicted pair speeds are positive. Deliberately changing SPARC observed velocities and uncertainties leaves the source projection fields exactly unchanged.

Initial pipeline attempts failed before scoring on incomplete brightness-supplement coverage and a string-valued distance field. The completed version uses a labeled stellar source-component descriptor and explicitly parses distances as numbers. These failures and their frozen source snapshots are retained. The successful run is things-direct-patterns-003.

## What this changes for the coherence idea

There is no reliable rule here of the form "the gas surroundings are more diffuse, therefore this region has extra gravitational pull." The data support a useful projected gas measurement, but that measurement alone has not delivered a transferable motion correction.

Before adding formula terms, the most informative next analysis is to distinguish regular rotational motion from warps, asymmetric profiles and streaming. A full cube model can test whether apparently unusual speeds are predicted by those effects. It should use channel noise covariance and a validated selection mask, then check whether a gas-structure term improves independent motion predictions beyond that model. Total stellar and molecular matter remains relevant; HI alone is not total density.

This campaign establishes a usable source archive and reports a negative result for one directly measured gas-context predictor. It does not reject all coherence mechanisms, validate 3D void structure, or produce a new gravity law.

## Evidence

- Map acquisition: things-observable-acquisition-003/receipt.json
- Cube acquisition: things-cube-acquisition-001/receipt.json
- Noise and support reconstruction: things-cube-noise-audit-001/result.json
- Registration, source audit, folds and predictions: things-direct-patterns-003/
- Numerical checks and target-poison check: things-direct-findings-001/verification.json
- Survey methodology: https://arxiv.org/html/0810.2125v1
- Source mass models: https://arxiv.org/abs/1606.09251
