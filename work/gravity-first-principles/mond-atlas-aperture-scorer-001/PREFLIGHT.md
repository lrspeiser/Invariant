# Restricted spectrum scorer: training and evaluation separation

Implement the optimizer prescribed by first-score protocolv2. This package
tests the scorer with manufactured spectra; it does not admit or read real
source-region responses. The model callback supplies42 native-channel values
for requested aperture indices and three parameters: systemic velocity,
intrinsic tracer dispersion and global emission multiplier. All material,
gravity, pressure, geometry and instrument choices remain outside the fit.

Use bounds[-30,30],[3,20],[.5,2], three starts[0,6,1],[0,10,1],[0,16,1],
ftol/xtol/gtol1e-8 and max_nfev300. Retain every start including failure and
bound contact. Choose the converged minimum training mean patch quadratic
form/42. The working covariance and western mean are explicit immutable
inputs; never subtract a mean fitted to evaluation data. Check rank3 and scaled
singular ratio>1e-6 before exporting any evaluation predictions. The rank check
uses a separately computed central finite-difference Jacobian with declared
parameter scales equal to bound widths; respect bounds with one-sided steps.

Freeze all requested evaluation predictions after fitting training data, before
the evaluator is given observations. The evaluator accepts a frozen numeric
prediction array, never an optimizer or callback. It computes individual
quadratic forms/42, arithmetic mean and raw RMSE. No likelihood significance,
joint-aperture independence, or post-evaluation model selection is implemented.

Manufactured controls: exact three-parameter Gaussian spectral recovery in
non-diagonal channel covariance, held-prediction recovery, rank-deficient model
rejection, global western-mean accounting, dense inverse versus triangular
score agreement<1e-11, unchanged frozen predictions under altered evaluation
data, invalid/nonfinite arrays rejected. Fit parameters must recover within
1e-5 absolute and held model relativeL2<1e-6. These are optimizer/control checks,
not observational gravity evidence or substitutes for the actual native renderer.
