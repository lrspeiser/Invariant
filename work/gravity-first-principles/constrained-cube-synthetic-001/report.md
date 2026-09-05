# Constrained cube numerical and mismatch controls

Rotation is sign-preserving with a zero-speed center and finer inner rings; dispersion varies radially. Several fixed starting guesses are selected exclusively on training loss. Numerical nested-model equality and finite-difference gradient controls pass.

Four independent NumPy generators use analytic rotation and varying dispersion at twice the fitted resolution. They include a warp, radial streaming, and a vertically layered lagging disk with Hanning smoothing. The stream case improves from approximately 7.72 to 1.05 in withheld whitened loss when streaming is modeled. The warped case exposes residual model mismatch and an optimizer issue; its full-model loss is about 1.15. A thick lagging case is almost indistinguishable from a thin rotating model at the chosen beam/noise level. These are explicit identifiability limits, not a certificate that real warps or thickness are uniquely recovered.

Real-data results are a separate milestone. No gravitational formula is selected here.
