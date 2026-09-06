# Fixed-image source-basis refinement preflight

Disposition: SOURCE_BLOCKED. Source-image diagnostics only, never observed
motion or lensing scoring. Frozen before this implementation is written.

Reuse the exact registered NGC2976 packet and its public source provenance:
S4G P5 cleaned stellar flux (Querejeta et al., arXiv:1410.0009), THINGS HI
(Walter et al., arXiv:0810.2125), and HERACLES CO (Leroy et al., arXiv:0905.4742).
These papers document projected measurements, not uniquely known depth.
The inherited masks, beam differences, missing phases, geometry and calibration
limitations remain. No new observational downloads or response reads.

Replace the square projection matrix with a rectangular integral operator:
A_ij = (1/Delta) integral over observed cell i of [hat_h(x-node_j) *
Laplace_b(x)] dx; b = height tan(inclination), with an exact thin limit.
Hat_h peaks at 1 and integrates to h. Thus A is dimensionless and the
observed cell average retains the source surface-density units. Two axes
form A_left Sigma A_right^T. Nonnegative source values imply nonnegative
predictions. Neither column sums nor Euclidean normalization remain those
of a square operator: column flux scales as h/Delta.

Use analytic antiderivatives with a stable exponential far-tail expression.
Independently integrate the tent against the Laplace CDF with scipy.quad,
and check against the published prior square operator, nested bilinear
prolongation, thin limit, symmetry, length-unit scaling, finite-aperture
flux loss, adjoint/gradient, a separate constrained least-squares solve,
and CPU/GPU agreement. All gates must pass before opening the real packet.
Any failure is retained; no threshold is retuned to source residuals.

Fixed observed cells and weights; latent factors 1, 2, 4; heights 0, 0.1,
0.2, 0.4 kpc for all three tracers. Same regularization convention; zero
initialization. All fits, including nonconvergence, retained. The old
5% descriptive mismatch flag remains descriptive. Report the unavoidable
nonnegative floor from signed data, prediction changes and stationary
optimality diagnostics. Do not treat a small same-image residual as an
independent prediction or an observed depth. Raw latent arrays stay private.
