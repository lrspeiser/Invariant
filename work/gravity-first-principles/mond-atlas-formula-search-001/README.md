# First adaptive sparse-formula experiment

An executable formula-search loop now runs on the RTX 5090. It constructs short
empirical corrections to the existing radial MOND residual predictor. This is
greedy sparse regression with nested validation, not reinforcement learning and
not a new gravitational field equation. The 126 galaxies are previously exposed
development data. No new 3D field or lensing likelihood was used.

The frozen library has 30 expressions involving stellar surface brightness,
disk size, morphology and a gas-fraction proxy, including squares and products.
Four baseline variables describe acceleration, acceleration spread, quality and
inclination. A path may add zero to three expressions. Every transformation,
coefficient and path choice is fit without the outer test galaxies. Inner folds
choose complexity and ridge penalty. No unrestricted generated code is executed.

## Results

| Whole-galaxy split seed | Baseline RMSE, dex | Adaptive RMSE, dex | MSE change |
|---|---:|---:|---:|
| 9060601 | 0.07531 | 0.07666 | 3.62% worse |
| 9060602 | 0.07714 | 0.07888 | 4.55% worse |
| 9060603 | 0.07716 | 0.07751 | 0.90% worse |

Seven of fifteen outer training partitions selected no additional term.
Brightness times size was selected in eight partitions and size times gas
fraction in seven. Repeated selection is not validation: the adaptive predictor
was worse on excluded galaxies in all three splits. All eight structure-shuffle
diagnostics performed at least as well as the actual first-split structure
search. The reference fraction is 1, not a calibrated discovery p-value.

Mean paired MSE improvement was -0.0001766 dex²; a conditional galaxy bootstrap
interval was [-0.0005459, 0.0002008] dex² and includes zero. This interval reuses
fitted predictions; it does not propagate training uncertainty or account for
physical associations between galaxies. It neither proves absence of a physical
relationship nor supports advancing one of these expressions as a gravity law.

## What was built and checked

- Reusable training-only expression construction, forward selection, nested
  complexity choice, exact formula serialization and standalone prediction replay.
- Five tests passed before observational sample values were opened: independent
  ridge/planted recovery, outer-label isolation, preprocessing isolation,
  constants/invalid inputs, and explicit product-library checks.
- A planted nonlinear interaction reduced RMSE to 20.6% of the baseline error.
  CPU/GPU and independent ridge predictions agree within 2.3e-16.
- All 30 saved real-data formulas and 80 shuffled-data formulas replayed.
  Repeating the complete first-seed selection on CPU matched the GPU selections
  and predictions to 2e-16. This is backend verification, not an independent
  scientific confirmation.
- GPU fitting took 46.85 seconds; the retained CuPy pool was 61,440 bytes (not
  total device use). Small matrices need not be faster on GPU.

Run with one CPU numerical thread:

```powershell
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python scripts/run_mond_atlas_formula_search.py --output work/gravity-first-principles/mond-atlas-formula-search-001/run-NEW --backend cuda
python work/gravity-first-principles/mond-atlas-formula-search-001/verify_run.py
```

`run-001` is immutable. Its bindings identify exact config, code and input bytes.
The verifier deliberately replays run-001. Serialized coefficients apply to
the stored training-centered and scaled expressions, not directly to raw units.
The sample and original [SPARC paper](https://arxiv.org/abs/1606.09251) provenance
are inherited from the published radial and pattern-learning packages.

Next priority is better independent inputs and physically grouped validation,
plus resolving the known source/noise/numerical gates. Expanding the expression
library on these same galaxies alone would not establish a new discovery.
No inferred dark mass is a training label; the empirical outcome also does not
settle the existence or absence of dark matter.
