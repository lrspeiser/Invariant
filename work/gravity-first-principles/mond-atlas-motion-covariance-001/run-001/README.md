# Executed correlated-channel motion benchmark

**THEORY_BENCHMARK_ONLY.** No observed source/covariance or native selection admission.
Run `run-001`: 30 statistical controls and 25 prior forward-law controls passed before study generation.

Three synthetic cases, four independent noise realizations per case, two fresh
replicates per realization, two overlapping folds and four fitting methods.
Results below average folds within each realization, then report the mean over
four realizations. Full SD/ranges, paired differences, every fit/start, parameter
errors and forecast distributions are retained in summary.json and fold receipts.

| Case | Method | Noiseless signal q/N | Fresh q/N | Same conditional q/N | Prediction passes / fits | Parameter passes / fits |
|---|---|---:|---:|---:|---:|---:|
| amplitude_zero | circular_correct | 0.00100 | 1.01302 | 0.99406 | 8/8 | 8/8 |
| amplitude_zero | expanded_correct | 0.00203 | 1.01376 | 0.99461 | 8/8 | 8/8 |
| amplitude_zero | circular_diagonal | 0.00149 | 1.01388 | 1.00128 | 8/8 | 8/8 |
| amplitude_zero | expanded_diagonal | 0.00293 | 1.01533 | 1.00169 | 8/8 | 8/8 |
| radial_only | circular_correct | 0.15069 | 1.13767 | 1.04842 | 8/8 | 0/8 |
| radial_only | expanded_correct | 0.00259 | 0.99161 | 1.01350 | 8/8 | 8/8 |
| radial_only | circular_diagonal | 0.15567 | 1.14287 | 1.12697 | 8/8 | 0/8 |
| radial_only | expanded_diagonal | 0.00363 | 0.99299 | 0.98952 | 8/8 | 8/8 |
| combined | circular_correct | 0.50157 | 1.51740 | 1.08390 | 0/8 | 0/8 |
| combined | expanded_correct | 0.00233 | 1.01319 | 0.99629 | 8/8 | 8/8 |
| combined | circular_diagonal | 0.51233 | 1.52841 | 1.49421 | 0/8 | 0/8 |
| combined | expanded_diagonal | 0.00435 | 1.01516 | 0.98883 | 8/8 | 8/8 |

The q/N columns are for held-out channels at training pixels. Signal and fresh
errors use the same true marginal covariance for every method. Conditional q/N
uses each method's assumed Schur covariance, so compare log densities including
log determinants in the full receipt when comparing those distributions.
Prediction pass requires all three held-out subsets, not this column alone.

![Signal and fresh noise](signal-and-fresh-noise.png)

![Noise interpolation](noise-interpolation-control.png)

The same-noise correction conditions only on training residuals. Its application
to fresh noise is explicitly a negative control: independent fresh realizations
have zero covariance with the old training noise. Noiseless signal errors and
fresh-data errors prevent that interpolation benefit being counted as motion recovery.

All fitted-mean distributions are plug-in diagnostics. The noise covariance
identity is exact for fixed parameters, but nonlinear fitted-parameter uncertainty
is not integrated. Four realizations do not establish coverage or significance.
All source/instrument parameters outside the frozen fit list are known. Pressure
support, force balance, dynamics, source uncertainty, observed spatial/channel
covariance and response-selected gas masks remain unvalidated or missing.

From the Invariant root:

```powershell
python -B scripts/run_mond_atlas_motion_covariance.py
python -B -m unittest discover -s tests -p test_mond_atlas_motion_covariance.py -v
```

Every run creates new assigned directories; existing receipts and arrays are immutable.
