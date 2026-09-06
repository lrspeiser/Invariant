# Joint within-core covariance: numerical success, calibration failure

We now have an explicit positive-definite joint model for all **24x24 spatial pixels and 42 channels inside each background core**: covariance K spatial tensor C channel. It includes cross-pixel correlations rather than only separate aperture error bars. The source/emission likelihood remains blocked because its full joint calibration fails.

The same 29 western NGC2976 cores train the model; three geometry-defined western folds select spatial shrinkage. All models and selection were frozen before opening the 27 eastern cores in each run. These data have historical development exposure. No emission pixels, observed velocities, masses or gravitational responses were used.

Three analytic controls passed: explicit small Kronecker inverse/log determinant, aperture projection A K A-transpose, singular-sample regularization and amplitude scaling. The channel covariance uses fixed 30% diagonal shrinkage from the earlier western selection. The four spatial shrinkages are 0.1,0.3,0.6,1.0. Western marginal log density selects **0.1**, the least-shrunk declared candidate. This boundary selection does not establish an optimum beyond the tested range.

| Spatial shrinkage | Full joint eastern q/N | Mean log density per voxel |
|---|---:|---:|
| **0.1, western-selected** | **0.480** | **0.572** |
| 0.3 | 0.426 | 0.258 |
| 0.6 | 0.461 | 0.027 |
| 1.0, spatial diagonal | 1.026 | -0.369 |

The selected full joint score fails the frozen [0.8,1.2] descriptive calibration range. The spatially diagonal model passes this one metric, but discards the spatial correlations and scores worse under the western selection criterion; it cannot be chosen after looking at eastern results. Joint q/N near one alone is not sufficient to pick a useful covariance.

The selected model's six derived aperture q/N values are **1.026,1.107,1.143,1.147,1.160,1.174** for sides 1,2,4,8,12,24; all pass the same descriptive range. Their eastern/predicted trace ratios rise from **1.048 to 1.260**. Thus passing aperture quadratic scores does not imply accurate total variance at every scale, and neither implies a calibrated full joint distribution. The earlier independently fitted aperture model predicted traces more closely.

The important finding is the disagreement between marginal and full joint calibration. Many weighted combinations of pixels/channels are assigned too much variance in aggregate, even while large aperture total power is underpredicted. A single overall rescaling cannot fix both. The selected spatial matrix has effective rank about **79 of 576** (trace-squared over squared-eigenvalue sum) and condition number about 251. These describe strong correlation, not independent samples or a diagnosed physical noise mechanism. Diagonal regularization, covariance separability, unmodeled means and regional changes remain possible causes.

Independent replay reconstructs selected K and C directly from all western values, uses matrix inverses/trace identities for the joint score and direct block sums for all aperture variances. Maximum discrepancy is **1.23e-15**. It never constructs a dense 24192x24192 matrix. This verifies the implementation; it does not repair the failed calibration.

Run001 completed fitting/scoring but failed final JSON serialization of a NumPy Boolean. The exact failed script, failure receipt, scores and model hashes are retained. Run002 changes only Boolean conversion and output directory. Independent verification confirms C and every western/eastern score CSV are byte-identical between runs. Selected K is public (7.10 MB); other generated K matrices and the failed-run duplicate were relocated to private storage with exact original/current paths and hashes in matrix-relocations.json. All scalar candidate scores remain public. No new raw data were downloaded.

Next needed: a better structured joint spatial covariance whose fine-scale modes and aperture marginals can both transfer, preferably constrained by beam/noise properties rather than arbitrary independent spatial entries. Cross-core covariance and source-region mask/noise validation are still absent. This model is a concrete within-core step, not an admitted likelihood for a whole observed galaxy or a gravity discovery.
