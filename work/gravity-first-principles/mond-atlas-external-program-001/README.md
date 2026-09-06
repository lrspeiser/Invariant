# External-field response: relative motion, not a uniform bulk push

Sixteen conditional NGC2976 solves apply a unit external boundary acceleration
along the major disk axis or the disk normal. Both neighborhood-density scales,
0.25 and 0.5 kpc, were kept fixed. Source densities remain the existing stellar,
HI and CO reconstructions. No external environment or galaxy motion was measured
or fitted in this experiment.

The major-axis calculations pass both mesh and enlarged-box gates. The center's
acceleration along that axis is about 0.687 or 0.668 per unit applied field.
Subtracting the center's acceleration leaves RMS differential accelerations of
about 0.131 and 0.115 per unit applied field across the declared sample points.
These are conditional linear response coefficients, not observed accelerations.
They scale with an independently supplied external field; its amplitude is
unknown here. The remaining response is directional, not a universal inward halo.

The normal-direction results remain **numerically incomplete**. Their mesh
refinement passes, but their enlarged-box comparisons fail: raw-vector RMS
changes are 5.51% and 7.01%, while some center-relative height groups change by
roughly 19–29%. These fields must not be used for precise directional comparisons
or observational scoring. All failed points and comparisons remain saved.

An independent analytic sphere control explains why the frame matters. For
epsilon_inside=1 and epsilon_outside=0.2, the uniform field inside is 3/7 of the
applied field. Every internal point receives the same acceleration, so subtracting
the center removes it completely. Subtracting the far-field value instead would
misidentify a bulk acceleration difference as internal gravity. NGC2976's
conditional nonspherical density permits a spatially varying response, but the
same distinction is essential.

At fixed density this PDE is linear. The external homogeneous solution adds to
the isolated solution and does not change its internal Green function. This is
not a test of MOND's nonlinear external-field effect, a dynamical equilibrium,
backreaction, causal transport, or a relativistic light law. Center subtraction
is explicitly a chosen relative reference, not a proven mass-weighted freely
falling center-of-mass frame.

Verification includes the smooth-slab flux solution, uniform-field and polarity
controls, 135 independent sphere-interface checks, an independently assembled
anisotropic sparse Dirichlet operator, all source hashes and exact replay of
the 16 raw/relative refinement and boundary comparisons over 6,160 saved vectors.
The homogeneous zero-RHS residual is normalized by boundary forcing, as frozen
in PREFLIGHT.md; a relative-to-zero source diagnostic would be undefined.

Evidence: run001/summary.json, run001/vectors.csv and analytic-review/field-review.md.
Reproduce using scripts/mond_atlas_external_program.py in a fresh output location.
Further work must resolve the normal-direction domain sensitivity and constrain
the actual external field independently before interpreting galaxy motions.
