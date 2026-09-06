# NGC2976 common-basis vertical projection

SOURCE_BLOCKED, source-image diagnostics only. No new gravity/motion response.
Before implementation this config selects all three tracer components and heights
0, 0.1, 0.2, 0.4 kpc, with no selection of a best height. Reuse the unchanged
independently checked nodal_projection_matrix / fit_nodes implementation. Its
analytic finite-cell coefficients have direct line-of-sight integration, thin
limit, flux/centroid, adjoint and omitted-data controls in test_mond_atlas_nodal.py.
Run those five controls before opening the registered source packet.

For a flat disk rho=Sigma(X,Y) exp(-abs(z)/h)/(2h), the sky image expressed in
stretched minor coordinates is Sigma convolved along Y with a Laplace kernel
of scale h tan(i). Both the continuum nodal basis and finite image-cell integrals
are retained. h=0 is a thin sheet. Fit only nonnegative coefficients within the
fixed support; weights use measured coverage and are not inverse noise variance.
The same-source image mismatch diagnoses compatibility with this restricted
family; it cannot by itself measure a true height or validate a 3D gravity law.
All source/beam/missing-matter and covariance limits of generic-source-001 remain.
