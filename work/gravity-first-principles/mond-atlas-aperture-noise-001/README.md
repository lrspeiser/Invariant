# Predicting background uncertainty across spatial scales

We can predict an important measurement behavior using real NGC2976 background
data: adjacent pixels share fluctuations, so averaging them does not reduce
uncertainty as much as an independent-pixel calculation predicts.

Fit on the 29 western cores, test on 27 eastern cores, with unchanged mean and
six predeclared square sizes. Preserve all 42 channels. The independent-pixel
baseline divides single-pixel covariance by pixel count; the empirical model
estimates a separate aperture covariance from western averages, with fixed 10%
diagonal shrinkage. Both sides have historical development exposure.

| Patch side | Actual / independent-pixel total variance | Actual / learned total variance | Learned channel q/N |
|---|---:|---:|---:|
| 1 pixel (1.5 arcsec) | 1.006 | 1.006 | 1.012 |
| 2 pixels (3 arcsec) | 3.321 | 1.007 | 1.020 |
| 4 pixels (6 arcsec) | 8.132 | 1.015 | 1.055 |
| 8 pixels (12 arcsec) | 14.723 | 1.003 | 1.163 |
| 12 pixels (18 arcsec) | 18.629 | 1.010 | 1.405 |
| 24 pixels (36 arcsec) | 25.509 | 1.031 | 4.224 |

Here "total variance" means the trace of the residual second moment across
channels, about the western fitted mean. It retains any residual offset or
foreground contamination and is not certified pure instrument variance.
At 4x4 pixels the independent-pixel model understates this variance by 8.13x,
equivalent to 2.85x in the corresponding root-mean-square fluctuation.

The western model predicts eastern total fluctuation power within 3.1% at every
tested size. That positive transfer does NOT certify the full channel covariance:
q/N also tests its orientation and relative channel fluctuations. Values near
one indicate agreement in this descriptive diagnostic. The frozen [.8,1.2]
range passes for sides 1,2,4,8 but fails at 12 and 24. Only 29 full-core western
averages are available at side 24 for 42 channels; estimating a flexible covariance
there is poorly constrained. This is a plausible contributor, not an isolated
diagnosis. Finite-sample precision bias, spatial dependence, mean uncertainty and
nonstationarity remain possible. We did not retune shrinkage on eastern results.

Practical implication: training ML on spatial averages with independent-pixel
error bars can reward noise patterns. We now have a reproducible way to measure
that failure and a limited correction that transfers on smaller apertures.
These tests do not demonstrate a density-dependent gravity law or correct
uncertainties inside the galaxy's emitting regions.

Seventeen tests passed before values were read in this run. Independent explicit
loop aggregation and inverse/log-determinant scoring replay all six fitted
covariances and 324 per-core score rows within 6.4e-14. Every scale and failure
is retained. No iid pixel error bars, significance claim, gravity score or joint
cube likelihood is supplied. Nested apertures are not independent experiments.

Reproduce with:

```powershell
python scripts/run_mond_atlas_aperture_noise.py --output work/gravity-first-principles/mond-atlas-aperture-noise-001/run-NEW
python work/gravity-first-principles/mond-atlas-aperture-noise-001/verify_run.py
```

The verifier intentionally audits immutable run-001. Inputs, source URLs,
protocol and source/code hashes are recorded in the config and run bindings.
The next useful extension is training-only selection of less flexible covariance
models and transfer to another observation, with source-region validation kept
separate. Overall gravity discovery remains unfinished.
