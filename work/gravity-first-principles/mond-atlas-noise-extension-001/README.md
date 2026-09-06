# Western-selected background covariance resolves the large-aperture marginal failure

**All six aperture sizes now pass the previously fixed descriptive q/N range [0.8,1.2].** The new covariance choices were selected only with three geometry-defined folds of the 29 western background cores, then frozen before opening the 27 eastern cores in this run. Both sides have historical development exposure. This is useful real-background validation, not new independent confirmation or a gravity result.

| Aperture side | Prior full covariance q/N | Western-selected q/N | Eastern / predicted trace |
|---|---:|---:|---:|
| 1 pixel | 1.012 | 0.983 | 1.006 |
| 2 pixels | 1.020 | 1.015 | 1.007 |
| 4 pixels | 1.055 | 1.025 | 1.015 |
| 8 pixels | 1.163 | 1.030 | 1.003 |
| 12 pixels | 1.405 | 1.049 | 1.010 |
| 24 pixels | 4.224 | 1.135 | 1.031 |

The channel covariance at each aperture is a 42x42 matrix. A full 24x24-core average leaves only 29 western spectral samples, too few to estimate its detailed orientation reliably without strong assumptions. The winning model for sides 2 through 24 borrows the channel correlation shape from single-pixel western data, while independently learning the variance of each channel at the chosen aperture size. It selects zero weight on the aperture's noisy full correlation matrix. Side 1 selects 30% diagonal shrinkage instead of the previous 10%.

This preserves measured spatial aggregation effects; it does not divide pixel variance by area. At side 24, covariance condition number falls from 124.17 to 17.20, q/N falls from 4.224 to 1.135 and held marginal log density improves by 1.196 per channel. At side 12 the log-density gain is 0.111 per channel. Side 1 slightly loses held log density (-0.00026 per channel), which is retained: training-only selection need not win every eastern metric. No eastern retuning occurred.

The scientific inference is restricted: sharing channel shape while allowing aperture-specific amplitudes transfers substantially better than a flexible independently estimated channel matrix at large scales. This is consistent with finite-sample covariance error contributing to the old failure. It does not isolate every cause or prove stationary instrument noise. Background means, foreground contamination and region dependence remain possible.

## What was tested and saved

Fifteen predeclared covariance candidates per side: four full-matrix shrinkages, six band-tapered candidates, and five mixtures toward the single-pixel correlation shape. Every western held score, ranking, fitted matrix, eastern score for all candidates and every scale is retained. Selected matrices were saved and hashed before eastern array access. Three independent pre-access tests passed, including loop aggregation, inverse/log-determinant scoring, planted diagonal preference, singular-sample positive definiteness, amplitude scaling and invalid-input rejection.

The independent verifier reconstructs all six selected matrices using explicit loop aggregation and separately written second moments; maximum matrix difference 1.14e-15. It independently replays all 2430 eastern core scores with matrix inverse/log determinant; maximum difference 1.87e-14. Replaying selections from saved western held scores produces zero mismatches. This verifies computations and recorded selection; it does not refit every western candidate independently.

No new raw data were downloaded (0 bytes). Only saved background packets and inherited support masks were opened; no galaxy emission pixels, motion targets, source masses or gravity fields were scored. Original pixel ordering is preserved by filtering the original hash-verified geometry CSV in the extraction order.

## Mask eligibility and the next blocker

The geometry audit confirms all 56 cores have the declared 24x24 dimensions and no overlapping boxes. Every core lies entirely inside its inherited regional support; western/eastern support masks are disjoint. This checks index eligibility and support membership, **not** whether support pixels are certified free of emission or representative of the galaxy's emitting region.

The result still supplies separate marginal covariances for each aperture. It does not define cross-covariances between apertures, adjacent spatial positions, regions or overlapping tiles. Therefore it is not yet an admitted joint spatial-channel cube likelihood. The next useful step is a positive-definite joint spatial/channel model that reproduces these measured aperture marginals, followed by source-region mask/noise validation and transfer to another observation. These steps are required before a motion model or ML system can treat an entire cube's residuals as calibrated evidence.

Reproduce: `python scripts/mond_atlas_noise_extension.py` (writes fresh run001 only; do not overwrite the existing run), and `python work/gravity-first-principles/mond-atlas-noise-extension-001/verify.py` (fresh independent-review output only). Primary observational paper: [Walter et al. THINGS](https://arxiv.org/abs/0810.2125). The source, protocol, software and exact hashes are bound in run001/pre-access-bindings.json; the private source packet remains outside Git.
