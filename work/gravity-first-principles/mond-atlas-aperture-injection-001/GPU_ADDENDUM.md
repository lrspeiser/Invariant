# Exact spectral algebra on the RTX5090

Before noise fitting, compare a float64CUDA implementation against the original
actual-source CPU callback on every spectral branch at the truth and all eight
corners of systemic[-30,30], sigma[3,20], multiplier[.5,2]. The beam-weighted
source fluxes and source velocities are identical; Gaussian bins are evaluated
individually. Sum emitters before the shared linear H and continuum A matrices,
which is an algebraic reassociation, not averaging velocities or changing physics.
Use stable positive/negative normal-CDF tails. Require maximum absolute spectral
error<1e-10mJy/beam and relativeL2<1e-10 on every comparison. Bind both scripts,
actual cache, projection/instrument operators and hardware/library identity.

Only after those gates pass may the GPU callback run the frozen three-start
injection fits. Preserve every failure. Fitted parameters remain the existing
three bounded nuisances. GPU acceleration neither admits source quadrature nor
constitutes machine learning or an observational gravity test by itself.
