# Force and pressure now feed the emission renderer

The new adapter joins the existing steady surface-column balance to thin-ring
emission with explicit independent flux and tracer-width inputs. It preserves
negative rotation-squared failures, rejects radial/vertical flow, and retains
finite spectral-band and spatial-field losses. No observed response was read.

Twenty of twenty-two frozen checks pass. The two failed checks are the strict
1e-11 relative comparisons against a separately implemented Gaussian-CDF,
Cartesian-rotation, direct-deposition and direct-convolution cube. Their errors
are 4.46e-8 and 4.65e-8. Inspection identifies the inherited cube primitive's
documented approximate normal CDF (absolute error below 8e-8); it is not an exact
special-function evaluation. The preflight's wording "exact Gaussian channel
integral" overstates this implementation's numerical precision. That wording
and the failed gates are retained rather than silently changed after execution.
The separate precision-diagnosis receipt isolates this cause: substituting
SciPy's normal CDF in an isolated diagnostic process reduces both errors below
4.62e-16. The on-disk primitive, original tests and failed gates are unchanged.
This verifies the cause; it is not a production precision upgrade.

Independent Cartesian geometry and line-of-sight velocities agree within
2.85e-14, supported and cold analytic speeds within 1.43e-14, and doubling
angular resolution changes the cube by 0.0162%, below the frozen 1% gate.
Changing only tracer width changes the spectrum but leaves pressure balance
unchanged. Impossible equilibrium, nonzero flows and reversed radii are rejected.
The cropped example explicitly loses 0.522708 of 0.848247 intrinsic flux outside
the spectral band and another 0.048867 spatially after band selection. Those
losses are retained, not normalized away.

This is an integration component with a retained precision failure, not an
observational gravity score. The caller must still provide the actual gas-column
density-weighted force and pressure closure; the new Newton field's two sampled
heights do not supply that integral. Axisymmetry and thin emission are conditional
assumptions. Native frequency channels, continuum processing, elliptical beam,
aperture covariance and observational scoring remain separate required joins.
