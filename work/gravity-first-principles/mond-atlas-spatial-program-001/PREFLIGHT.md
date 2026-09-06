# Conditional real-source distributed response

Frozen before source-force evaluation. SOURCE_BLOCKED for observed motion or
lensing scoring. Numerical tests and conditional source predictions only.

Use the four NGC2976 source cases and exact packet hashes in the copied
source-bindings.json. These are bilinear planar inversions of S4G stellar light,
THINGS HI, and HERACLES CO with assumed normalized exponential heights, not
measured 3D matter. Their inherited provenance is in generic-source-001 and
source-resolution-001. Primary references: https://arxiv.org/abs/1410.0009,
https://arxiv.org/abs/0810.2125, https://arxiv.org/abs/0905.4742.
Unmatched beams, fixed mass conversions, uncertain depth and exterior, and the
CO nonnegative reconstruction floor prevent observed-response admission.

Apply a single-generation spherical secondary kernel about EVERY source element:
q(s)=eta/[4*pi*L^3*x*(1+x)^2] for x=s/L<C, zero otherwise.
Its acceleration is -G*eta*dm*m(min(x,C))*d/s^3, m(x)=ln(1+x)-x/(1+x).
Its potential inside is -G*eta*dm*[ln(1+s/L)/s-1/(L*(1+C))]; outside it is
-G*eta*dm*m(C)/s. Use eta=1, L=4 kpc, C=10, G=4.30091727003628e-6
kpc (km/s)^2/Msun, without fitting any observed speed or halo. This fixed
normalization measures geometry, not successful explanation of missing gravity.
The finite integrated secondary-source weight eta*m(C) is not an energy budget.

Compute 72 locations: R=1,3,6 kpc, 12 equally spaced azimuths, z=0,0.4 kpc.
Three quadratures: planar cells 0.125/0.0625/0.03125 kpc and vertical
Gauss-Laguerre orders 12/24/48, paired respectively. Exact bilinear cell masses
are concentrated at cell centers only for numerical integration. Vertical
quadrature covers both exponential tails without a finite z boundary. Preserve
each component's force and potential. No source-mass renormalization.

Require source mass agreement 1e-10, potential gradient 1e-6, rotation/translation/
reciprocity 1e-10, and CPU/GPU agreement 1e-10 in independent controls before
real-source runs. Fine-versus-middle force RMS must be <1%, each point <3%;
retain all failures. Paired refinement alone does not isolate planar vs vertical
errors. These are numerical gates, not statistical confidence intervals.

For each R,z group, the mean cylindrical components equal the response of the
discrete 12-rotation average of the source at the same azimuth. This mass- and
radial-profile-preserving counterfactual is not a second observed galaxy and
not a fully continuous axisymmetric source. Report force deviations and
nonradial fractions, plus changes across the conditional source reconstructions.
Changing stellar height also changes its inverse planar reconstruction; it is
not a pure thickness effect. Do not interpret numerical source differences as
an observed gravitational anomaly or evidence for the kernel mechanism.
