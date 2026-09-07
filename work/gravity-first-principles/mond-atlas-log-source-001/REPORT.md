# The logarithmic response preserves more source asymmetry

The fixed logarithmic extra force was integrated over all four conditional
NGC2976 source alternatives, at all three quadrature resolutions and all 72
positions. The RTX 5090 calculation completed in 67.5 seconds and produced
10,368 component/total field records, including ordinary Newton and Newton plus
the extra response. No galaxy speeds, lensing measurements or halo parameters
were fitted or scored.

The usable pattern is that this response remembers the source's uneven shape
more strongly than the previously computed finite NFW-like response. Both become
rounder farther from the source. At fixed radius, we rotate each vector into its
local cylindrical frame and measure its RMS variation around the 12-azimuth
mean, divided by vector RMS. In the plane:

| Radius | Logarithmic extra, across four source alternatives | Previous finite extra |
|---:|---:|---:|
| 1 kpc | 19.0–29.3% | 10.1–16.2% |
| 3 kpc | 5.7–9.9% | 2.7–4.9% |
| 6 kpc | 1.5–4.0% | 1.0–2.0% |

At height 0.4 kpc and radius 1 kpc, logarithmic azimuth variation falls to
12.8–23.7%, versus 8.1–13.9% for the finite response. These are conditional
predicted field shapes, not measured anomalies. The 12-rotation source average
is a mass-preserving counterfactual; linear superposition and rotational
covariance make it equivalent to the cylindrical mean used here. One actual
rotated source integration and manufactured covariance controls also passed.

The logarithmic extra is about 6.6–6.8 times the finite extra in plane at 1 kpc,
3.4–3.5 times at 3 kpc, and about 2.0 times at 6 kpc for these fixed parameters.
That amplitude comparison is not a fit or a fair equal-budget normalization:
eta=1 has different meanings for the two kernels. Its directional variation,
normalized by its own field RMS, avoids a simple overall-amplitude explanation.

All sixteen logarithmic-extra component/total comparisons pass the frozen
middle-to-fine gates of 1% vector RMS and 3% maximum individual-point change.
Total-field RMS changes are 0.036%, 0.347%, 0.043% and 0.343% for the f1/h0.1,
f1/h0.4, f4/h0.1 and f4/h0.4 cases, respectively. Maximum total point changes
are 0.094%, 1.138%, 0.144% and 1.131%. Paired planar/vertical refinement is a
numerical diagnostic, not proof that either error is separately negligible.

The ordinary Newton calculation exposes a numerical limitation: all sixteen
Newton comparisons and all sixteen Newton-plus-log comparisons fail at least
one frozen gate. The Newton total RMS changes span 1.11–6.37%, with maximum
point changes 4.07–20.24%. Concentrating each quadrature cell's mass at its
center is much less forgiving for the unsoftened inverse-square singularity.
These are integration failures, not failures of Newtonian physics. The same-source
Newton fields and ratios remain saved as unconverged diagnostics; they cannot
support a validated percentage enhancement or an observational comparison.
No smoothing was added to Newton to conceal this failure. Resolving that
denominator needs a separate frozen near-source integration improvement.

The implemented static law is
Phi_extra = (G eta dm/L) ln[s/(s+L)], s=sqrt(r^2+b^2),
g_extra = -G eta dm d/[s^2(s+L)], with eta=1, L=4 kpc and b=0.05 kpc.
Here G is in kpc (km/s)^2 per solar mass, potential in (km/s)^2 and acceleration
in (km/s)^2/kpc. At b much smaller than r much smaller than L, its potential
is approximately logarithmic and force approximately inverse radius. At large
r it returns to inverse square with extra/Newton point-source ratio eta.
It has a finite softened center and nonnegative spherical effective enclosed
mass derivative. This is a static conservative pair potential, mathematically
compatible with more than one proposed interpretation. It does not validate
oscillator history, reflection, consumption, relativistic lensing or a causal
field model.

Source/code hashes were bound before force evaluation. Manufactured gradient,
translation, rotation, mass-split, reciprocity, potential-integral and limiting
checks passed. Actual-source CPU/GPU and rotation checks passed. A separately
implemented CPU force/potential calculation using independent Laguerre nodes
reproduced six selected finest-grid outputs within 6.46e-14 relative; its
planar mass builder was the previously audited shared builder. All component
sums were independently replayed, with absolute roundoff below 9.1e-13.
The inherited S4G/THINGS/HERACLES maps are conditional tracer reconstructions,
not measured unique 3D matter. Fixed mass conversions, uncertain depth and
exterior, unmatched beams, and the CO reconstruction floor remain. Changing
stellar height also changes its planar inverse reconstruction, so source-case
differences cannot be attributed to thickness alone. Admission remains
SOURCE_BLOCKED for observed-response scoring.

Reproduction: scripts/mond_atlas_log_source.py (fresh output directory required),
scripts/mond_atlas_log_source_review.py and scripts/mond_atlas_log_source_independent.py.
All successes and failures are in run001/review.json; no prior experiment was changed.
