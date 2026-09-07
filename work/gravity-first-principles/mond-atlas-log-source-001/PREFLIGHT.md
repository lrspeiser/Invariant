# Frozen conditional logarithmic source response

SOURCE_BLOCKED for observed response scoring. Reuse exactly the four NGC2976
conditional source alternatives from spatial-program-001, whose S4G, THINGS and
HERACLES source papers/packets and limitations remain inherited. No target motion,
halo fit or actual local-G calibration is used. Read admission policy before run.

Fix G=4.30091727003628e-6 kpc (km/s)^2/Msun, eta=1, L=4 kpc, b=0.05 kpc.
For displacement d from each source element, s=sqrt(|d|^2+b^2),
Phi_extra=G eta dm/L ln[s/(s+L)],
g_extra=-G eta dm d/[s^2(s+L)]. This is a conservative static pair potential.
Its large-distance extra/Newton ratio tends to eta; it does not produce a
permanent flat rotation curve at infinity. A logarithmic intermediate potential
does not validate an oscillator history, reflection or gravity-consumption story.
Compare true unsoftened same-source Newton and previously frozen finite NFW-like
extra vectors; report extra and Newton+extra separately. Do not equate amplitudes
of different kernels merely because both use eta=1.

Freeze original 72 positions R=1,3,6 kpc, 12 azimuths, z=0,0.4 kpc and paired
planar/vertical quadratures (0.125,12),(0.0625,24),(0.03125,48). Exact bilinear
cell masses and normalized two-sided exponential vertical quadrature. Source
hashes/code/this preflight/old finite fields bind before source force evaluation.
Require manufactured potential-gradient relative error <1e-6, translation,
rotation, co-located mass split and reciprocity <1e-10; independent CPU/GPU
agreement <1e-10. Check coincident softened extra is finite, outer inverse-square
limit, nonnegative spherical effective density and analytical potential integral.

Retain original convergence gates: middle-to-fine vector RMS <1%, every point
<3%, separately for both kernels and total Newton+extra, each component and sum.
Retain all failures, especially Newton close-source quadrature failures; do not
claim ratios validated if the Newton denominator fails. Paired refinement cannot
isolate planar and vertical errors. Rotate the source relative to sampled points
using discrete rotational covariance and verify one actual rotated source sum.
No physical parameter changes from results. GPU preferred, CPU independent
selected actual-source points; maximum 300 seconds field calculation and <20MB
public outputs. Failures/partial execution retained if resource cap reached.
