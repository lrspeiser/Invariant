# Training/evaluation scorer controls pass

The three-nuisance optimizer and separate frozen-prediction evaluator are now
implemented. Manufactured spectra recover systemic velocity, intrinsic width
and emission multiplier within1.78e-15 absolute. Predictions on five separate
synthetic evaluation apertures agree with the independently expressed truth
within7.33e-16 relativeL2. Direct dense-inverse and triangular-covariance scores
agree within2.09e-17. All eight declared checks pass.

Every optimization start, bound contact and training status is retained. A model
whose line-width parameter has no effect fails the rank gate. Evaluation accepts
numeric frozen predictions; changing evaluation data changes only its residual
score, not those predictions. Nonfinite spectra are rejected. A nonzero supplied
western mean is accounted for once, separately from the model systemic velocity.

These are manufactured optimizer tests. No real source-region spectra were
read and no gravity law was scored. The actual model callback still needs the
converged source-force/pressure/emission and native-instrument products, all
specified source and pressure alternatives, plus a driver binding their hashes
and enforcing training-first file access. The functions are not a substitute
for that observational admission and end-to-end validation.

The working covariance supplies descriptive individual-aperture losses only.
No joint-aperture likelihood significance, independent-discovery claim or
post-evaluation selection is implemented. A new source or instrument hypothesis
must get its own explicitly developmental protocol rather than reuse evaluation
residuals under a fresh-holdout label.
