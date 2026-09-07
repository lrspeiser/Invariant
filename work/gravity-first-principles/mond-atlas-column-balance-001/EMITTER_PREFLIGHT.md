# Extend the balance check to actual emitting positions

Before evaluation, use repaired fine f4 mass-centroid emitters and the new
run004 .05..6.025 column-force table. Evaluate the same84 already fixed material,
height, force-law and pressure choices at every distinct emitting radius.
Interpolate within the table only; no extrapolation. Pressure and derivative
come from the same source-only Gaussian estimator; the raw HI denominator is
interpolated from the same force table. Preserve any negative rotation squared.

If any emitter has negative rotation squared, do not render an invented speed
or call the whole steady model valid. Report the affected mass/flux and its
positive native-aperture beam-weighted flux. This measures whether an assumed
steady pressure closure is usable across actual emitting support, beyond the
previously evaluated radial interval. It does not compare with measured speeds.

For diagnostic importance only, bound any unknown spectrum from such emitters
using integrated positive flux times1000*maxrow sum(abs(A H)/bin_width), for
each of the three instrument alternatives. This follows because a normalized
nonnegative line has bin probability at most1. It holds for arbitrary velocity
and width, and includes signed continuum subtraction. No missing signal is set
to zero or assigned a velocity. The bound is conservative and is not a model
repair, new admission threshold, or observation-based exclusion.

Group vertical nodes only when their planar radius and azimuth are identical;
their native beam-weighted flux sums exactly. Preserve original source arrays
and source mass; use sampled native pixels. No source-region spectra or
observational score, no retuned pressure normalization, no new formula fit.
