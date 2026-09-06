# Source-closure and correlated-noise controls executed

The NGC2976 source-height study and the correlated-motion-noise task are complete
as bounded development experiments. **179 integration tests pass.** Both packages
retain failed, null and successful comparisons. No observed gravity law or
complete 3D ordinary-matter source is admitted; the larger goal remains active.

## Findings that change the next step

1. **A wrong motion model can look adequate when it predicts nearby noise.**
   In the combined synthetic injection, circular-only fitting has conditional
   q/N=1.084 but fresh-noise q/N=1.517. Adding the injected warp/streaming/emission
   terms gives fresh q/N=1.013. Noiseless signal error falls from 0.502 to 0.00233.
   Independent noise realizations distinguish actual signal recovery from
   interpolation of correlated residuals.
2. **Extra motion freedom has a cost when those effects are absent.** In the
   zero-amplitude control, expanded noiseless error rises from 0.000998 to
   0.002034. All outcomes are kept. Four independent realizations per case are
   a small conditional study, not population significance or unique recovery.
3. **The current source-image RMS flag cannot reject the CO height models.**
   NGC2976's negative CO measurements alone imply a 9.23% lower bound on the
   RMS of any nonnegative prediction, already above the 5% descriptive flag.
   The thin-source fit has 9.85% RMS. Noise/background validation must precede
   a physical interpretation of this mismatch.

The complete [motion covariance report](../mond-atlas-motion-covariance-001/README.md)
contains 96 fitted models, 192 starts, 30 statistical and 25 imported mechanics
controls, with 12 training-noise and 24 fresh-noise cubes. Correct training
marginals, conditional Schur covariances, independent-pixel forecasts and fixed
missing-cell masks are explicitly distinguished. The child verified exact replay
of all 36 noise draws and 96 predictions. The parent rehashed the report and all
16 private design/truth/noise packets and reran its 18-test suite.

![Signal versus fresh noise](../mond-atlas-motion-covariance-001/run-001/signal-and-fresh-noise.png)

This is THEORY_BENCHMARK_ONLY. Noise has known AR(1) channel correlation and
independent spatial pixels after the instrument; the mask is fixed independently
of emission. Parameters of the mean are fitted but their uncertainty is not
marginalized in these forecast diagnostics. Native gas masks, observed covariance,
pressure support and source/instrument uncertainty remain unresolved.

The [NGC2976 source-projection report](../mond-atlas-ngc2976-projection-001/README.md)
contains twelve converged fits across three tracers and four height alternatives.
The old common-basis operator and its five independent controls are unchanged.
Ten fits cross the 5% descriptive mismatch flag; all remain in the output. HI
thin/0.1-kpc RMS is 0.13%/1.50%; stellar thin RMS is 5.23%, requiring source
resolution/beam/calibration checks. None of these coverage-weighted same-source
errors is a calibrated height likelihood. Height zero is an analytic sheet limit,
not a volumetric source to pass to an ordinary finite-height field loader.

## Verification and publication

verification.json records 179 tests with zero failures/errors/skips, all 768
prior execution-015 manifest entries, 59 covariance package entries, sixteen
private covariance packets and twelve private source-projection packets checked.
There are 876 unique integrity-checked files. Source projection adds twelve fits,
bringing the accumulated source-image fit count from 22 to 34, without adding
any gravity field or observed cube likelihood. The conditional field count stays
29 for one galaxy; two galaxies now have conditional source-grid/projection work.

The existing regression subset includes historical SPARC integrity reads; these
do not enter either new experiment or its selection. The child covariance task
itself opened no observations. Prior mutable handoff/task bytes are archived
before updating them. The new public manifests exclude all raw/synthetic arrays.

Publication base: `0d3aace45309ffa2792c2d78953a624b3fb5c4d5`, the NGC2976 source
milestone already on main. After exact staged-byte and whitespace checks, the
coordinator publishes an ordinary fast-forward update. All four separately
created app tasks are idle with their assigned increments complete.

Next work is source-resolution/beam/noise calibration, pressure support, and
an observationally justified motion covariance/selection model. Extend eligible
galaxies and group/survey holdouts while propagating source alternatives. Use
fresh or independently held-out measurements to judge structural corrections;
a gain in correlated-noise prediction is not a gain in gravity prediction.
