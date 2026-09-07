# Added gravity and stellar mass produce similar predicted speed shifts

Using the actual NGC2976 source maps and the converged HI-column-weighted force
table, all84 declared steady-balance profiles have nonnegative rotation squared
over0.75–2.5kpc. This covers two stellar-height source alternatives, two force
laws, seven material choices and three pressure normalizations. It does not
prove a full gas-flow equilibrium or validate the emitting volume outside this
radius interval. No observed spectra or stellar speeds were fitted.

The fixed logarithmic response raises baseline predicted orbital speeds by a
median7.41–8.01km/s across the height/pressure choices. Raising the predefined
stellar mass-to-light ratio from0.6 to0.8, using ordinary Newton gravity, raises
them by a median9.18–9.81km/s. Their radial change vectors have a signed cosine
of0.983–0.988; this is an uncentered shape-alignment diagnostic, not a statistical
correlation or discovery probability. The two predefined total predictions
still differ by2.05–2.11km/s RMS. Neither amplitude was fitted to imitate the
other, and neither was fitted to observations.

This demonstrates a practical ambiguity in this source-conditioned experiment:
the proposed extra-force signal and a predefined change in stellar mass produce
similar rotation changes. It does not establish that the actual stellar mass is
wrong, that either mass-to-light value is allowed by independent population
constraints, or that the logarithmic mechanism explains galaxy motions.
Independent material estimates and the complete spatial/spectral prediction
remain necessary to distinguish these alternatives.

Raising the pressure reference from5 to15km/s lowers median predicted speed by
4.31–4.34km/s under Newton and3.86–3.88km/s with the log response. Some radii
instead rise by up to0.80km/s because the assumed pressure gradient there points
inward. The effect depends on the gradient, not merely the amount of gas or
its line width. These reference values normalize Pi=s_reference² Sigma_smooth;
the denominator is rawHI density, so local effective pressure speed varies.
No measured line width is reclassified as supporting pressure.

All separate stellar, atomic-plus-helium and CO force components are retained.
HI mass factors scale gravity and the pressure numerator/denominator together;
the latter cancel in the pressure acceleration. Negative rotation squared would
invalidate a whole evaluated profile rather than be clipped; none occurs in
this restricted evaluated interval. Values elsewhere remain untested here.

The later emitter-bound calculation scales intrinsic HI flux together with its
HI source factor, an explicit common-normalization sensitivity. A change in
mass conversion alone need not change measured line flux; that distinct
interpretation must be separated before fitting observations. No opacity or
flux-calibration error was inferred here.

Run001's admission join mistakenly required the superseded coarse radial
interpolation checks as well as the refined ones, and failed before calculating
balance. Run002 explicitly uses run002 force-grid/box/vertical/angular checks
and run003 radial checks. Those all pass within the declared radius interval;
their original full-support failures remain unchanged. Source table hashes,
source-only pressure construction and the revised pressure-closure addendum
are bound. The independent pressure implementation was already tested in
mond-atlas-pressure-support-001; its old tests are not counted again here.

## Full emitting support exposes a pressure-closure failure

A subsequent separately frozen calculation uses the repaired902,5923D emitting
nodes (18,804 planar groups,18,732 distinct radii) and the new force table through
6.025kpc. **None of the84 choices supplies a steady circular solution over every
emitting position.** Each fails in the faint outer material; the smallest affected
radius across alternatives is5.288kpc. Invalid intrinsic flux fractions range
0.000776% to1.505%. No negative rotation squared was clipped and no velocity was
assigned to those nodes. The earlier84/84 result applies only to0.75–2.5kpc.

The prescribed Gaussian-smoothed pressure gradient divided by very small raw
HI column creates strong outward support near the source boundary. This
diagnoses the declared closure/source combination, not gravity in general or
a measured gas instability. Adding the log force reduces some invalid regions
but does not make any tested global steady profile valid. Outer gas can require
a different pressure closure, time dependence or noncircular motion; this run
does not choose or calibrate a replacement from observed velocities.

The problematic material projects almost entirely away from the selected
apertures. A conservative velocity-independent bound uses positive weighted
flux times the absolute-row-sum norm of the full spectral/continuum operator
divided by bin width. Across all84 models and three instrument alternatives,
the largest bound is5.525e-6mJy/beam at emission multiplier1, or1.105e-5 at the
allowed maximum multiplier2. It bounds any normalized nonnegative line profile,
not only the assumed Gaussian. Western aperture noise forecasts are at least
0.102mJy/beam per channel. This can support an explicit prediction-error envelope
for the restricted aperture calculation; it is not a repair of the global
steady model or permission to label missing emission as known zero.

Actual source quadrature refinement, complete source sensitivities, spin-sign
branches, assembled native spectra and training/evaluation access controls
still must be completed. No observed spectral comparison follows merely from
this small bound, and no new observation was accessed in calculating it.
