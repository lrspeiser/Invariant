# Actual HI-weighted column forces are now available

We integrated the full conditional stellar, atomic and molecular source maps
to produce the radial force needed by the pressure adapter. This is not a
midplane force or an interpolation of the earlier72point experiment. The table
uses R=.05..6kpc, spacing.00625,953radii, and retains each component separately
for the primary f4/stellar-height.1 model and the f4/height.4 sensitivity.
No observed spectrum was read or fitted.

The table is run003/column-force-table.csv, with7,624 rows. Force units are
(km/s)^2/kpc; positive gbar is inward. Newton, logarithmic extra and their sum
are separate columns. HI columns are in solar masses/pc² excluding helium;
the gravity component includes its original helium correction. The table also
retains HI-weighted radial-force variation and tangential RMS. Component forces
can be scaled linearly for the already-prescribed one-at-a-time mass-factor
sensitivities. They are not fitted strengths.

The exact weighting is integral rho_HI*(-g dot e_R) dphi dz divided by integral
rho_HI dphi dz. The actual angular HI map remains inside both integrals, so
clumpy azimuths carry their own weight. For separable exponential vertical
profiles, source and HI heights can be integrated first without approximation
to the vertical model. The difference-height distribution yields the Newton
multiplier W(k)=[1+k hs ht/(hs+ht)]/[(1+khs)(1+kht)], ht=.2kpc. The same
distribution is integrated numerically for the fixed softened logarithmic
kernel with eta1,L4kpc,b.05kpc. This analytic reordering is why the column
calculation resolves the earlier thin-source point-force difficulty.

Newton uses the exact original bilinear tent coefficients and analytic vertical
average. Logarithmic forces use a padded real-space column kernel convolution
with those same coefficients. The source occupies±8kpc and evaluation lies
within6kpc, so Cartesian source-target separations are less than14kpc, inside
the16kpc halfwidth. Separate32→48kpc box, grid and vertical-order comparisons
are retained; padding is not a model of additional gravitating image galaxies.
Newton's periodic-image effect is tested separately with64→96kpc boxes.

In the aperture-radius interval .75..2.5kpc, all56 component/total box, grid,
vertical and angular comparisons pass. Worst RMS is0.2147%, worst point0.8245%.
All32 subsequent radial interpolation comparisons pass there too: worst RMS
0.1408%, worst point1.8797%, below the frozen1%/3% thresholds. This is useful
numerical progress; it does not by itself close the entire emitting-domain
requirement, because beam convolution and vertical projection import emission
from beyond the aperture-radius interval.

Full-support failures are retained. In run002, individual HI and CO Newton
components fail a pointwise grid check outside the aperture interval. Coarse
radial interpolation also fails near the small central HI hole. The finer
run003 retains full-range radial interpolation failures: at R=.06875kpc the
primary total Newton difference is24.73 and log difference13.76 field units;
the thick-star case has104.25 and7.21 respectively. These cannot be erased by
reporting only the successful aperture interval. Their relevance to delivered
spectra must be bounded using actual emission and beam weights or resolved by
additional numerical work before full emitting-domain admission.

Run001 stopped before actual force evaluation because the optional R=.025kpc
extension has exactly zero HI weight. Its weighted force is undefined. The
requested R=.05..6 support is positive. The failed check and original bindings
remain; a pre-force addendum froze the requested support for run002. Run002
took20.0seconds. A separate frozen numerical refinement to.00625spacing took
12.4seconds in run003. No source or physical law was changed. Do not extrapolate
the radial table into zero-HI rings or past6kpc. The source-side emitter audit
is separately examining small integrated-cell centroid tails beyond this table;
an exact-mass centroid or source-respecting quadrature is needed rather than an
invented force at a zero-density sample.

For orientation, the fixed log extra/Newton column ratio at R=1,2,3,6kpc is
0.211,0.260,0.382,0.551 in the primary model, and0.219,0.278,0.406,0.553 in the
thick-star sensitivity. These are source-conditioned predictions, not measured
missing-gravity fractions or evidence for a reflection/history mechanism. The
full-support qualification above applies to their numerical use.

Controls precede source force evaluation: adaptive difference-height integrals,
exact normalization and Newton multiplier, log vertical quadrature including
the center, Gaussian/Hankel Newton column forces, compact-source direct log
integration, and symmetry. Actual source/code hashes and mass closure are
recorded. A separate RTX5090 direct sum over fine actual stellar-source cells
reproduces three log-column vectors within0.00543%. That direct check changes
the planar integration method; its vertical quadrature is shared but was
independently checked against adaptive integrals before fields. Saved component
means sum to the total at roundoff. The full public package remains below20MB.

The inherited mass conversions, missing-map assumptions, embedded source beams,
assumed vertical structure and same-observation selection dependence remain.
The pressure closure is a separate restricted steady/axisymmetric model and
must use its prescribed HI stress gradient and raw HI denominator. This force
table does not establish vertical equilibrium or a complete vector motion law.
Admission remains conditional and blocked for observed-response scoring until
the remaining emitting-support and adapter requirements are satisfied.
