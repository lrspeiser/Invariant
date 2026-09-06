# Spatial-band channel model: aggregate calibration hides unresolved modes

Completed the single frozen follow-up that relaxes spatial/channel separability. Each background core is represented in a complete orthonormal 2D cosine basis, with a separate42-channel covariance for DC, low, middle and high spatial-frequency bands. The model includes1,10,47 and518 spatial modes in those bands, respectively. It assumes independent spatial modes and one covariance within each band; neither assumption is established by the results.

Western-only three-fold selection chooses channel diagonal shrinkage **DC1.0, low0.6, middle0.3, high0.1**. All matrices and choices were frozen before eastern array access. Same29western/27eastern historical-development NGC2976 cores; no emission, motion or gravity targets.

**Aggregate joint q/N=0.9932 passes**, and all six aperture q/N values also pass the fixed[0.8,1.2] range. But predeclared subdivisions expose severe cancellation, so this is **not an admitted joint likelihood**:

| Spatial band | Whole band q/N | Lower-frequency half | Upper-frequency half |
|---|---:|---:|---:|
| DC | 1.166 | — | — |
| Low | 0.988 | 1.113 | 0.862 |
| Middle | 0.997 | **1.291** | **0.691** |
| High | 0.993 | **1.960** | **0.00237** |

The high band assigns nearly the same covariance to modes with very different actual fluctuation power. Its262 lower-frequency modes are underpredicted in aggregate while its256 upper-frequency modes are strongly overpredicted. Averaging both yields an apparently satisfactory score. Channel-eigenvalue quartiles mostly behave better within each band, but the lowest DC channel quartile has q/N1.566. All26 frozen diagnostic groups and their counts are retained.

This locates a concrete failure of broad spatial-frequency pooling. It does not prove which instrumental mechanism created the pattern, or that channel nonseparability alone caused the earlier failures. Cross-mode covariance remains unmodeled. No frequency threshold, band or shrinkage was changed after evaluation; no further adaptive branch was run.

| Aperture side | q/N | Eastern/predicted trace |
|---|---:|---:|
| 1 | 1.011 | 1.042 |
| 2 | 1.184 | 1.221 |
| 4 | 1.152 | 1.189 |
| 8 | 1.120 | 1.138 |
| 12 | 1.180 | 1.186 |
| 24 | 1.166 | 1.031 |

Passing q also does not guarantee every aperture's total fluctuation power: the2-pixel trace remains about22% above prediction. Joint mean log density is0.0968 per voxel, worse than the stationary model's0.9297 despite the improved aggregate q. These metrics constrain different aspects of the covariance; neither can substitute for the failed subdivisions.

Three pre-access checks passed: Parseval and the manual spatial basis, explicit small full covariance/inverse/log determinant, exact aperture projection and rank-deficient covariance regularization. Independent review reconstructs the cosine transform directly from its trigonometric definition, refits selected covariances by moments, replays all27 selected joint scores and all six aperture projections using explicit inverses, verifies every diagnostic group and reproduces the western selection. Maximum covariance difference1.43e-14; all score differences below7.4e-15.

All matrices are compact4x42² per candidate. Every western candidate score, all five evaluated compositions, eastern core scores, aperture scores and diagnostic failures are retained. Both regions were historically exposed; this is development validation, not fresh confirmation. No new raw bytes or source-region mask admission. A calibrated model still needs spatial-mode power variation and cross-mode behavior addressed under a separately frozen test, plus source-region and cross-core validation before gravity or motion scoring.
