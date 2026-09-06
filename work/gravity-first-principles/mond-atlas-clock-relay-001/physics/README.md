# Clock rates, exchange, and the proposed relay

Disposition: **THEORY_BENCHMARK_ONLY**. Six target-free checks passed in `test_clock_mechanics.py`; zero observational arrays were opened. This branch establishes consistent bookkeeping and testable distinctions, not an observed source of energy. Parent radial fits, if admitted independently, test static force prescriptions only.

## What the clock connection does establish

For a static metric with no time-space cross terms,

\[
ds^2=-N(\mathbf x)^2c^2dt^2+h_{ij}dx^idx^j,
\qquad d\tau=Ndt.
\]

The local acceleration required to hover is outward, \(\mathbf a_{\rm hover}=c^2\operatorname{grad}_h\ln N\). The opposite is the local gravitational acceleration of an initially stationary freely falling test body relative to the static frame. In weak fields with approximately Euclidean spatial geometry,

\[
\Phi=c^2\ln N,\qquad \mathbf g=-\nabla\Phi=-c^2\nabla\ln N.
\]

Thus every conservative weak-field acceleration prescription has a clock-like potential representation. This mapping alone adds no force and supplies no energy. If N already contains the baryonic potential, adding its gradient a second time double-counts that potential. A candidate independent correction must instead declare \(\ln N=(\Phi_b+\Phi_\chi)/c^2\), with only \(-\nabla\Phi_\chi\) added to ordinary gravity. A clock-like lapse inferred from motions is not a separately measured clock rate. Exact strong-field coordinate acceleration is not simply the Euclidean gradient expression. [Carroll's GR notes](https://arxiv.org/abs/gr-qc/9712019)

Clock-rate differences are measured: Chou and colleagues compared optical clocks at small height differences. That experiment tests rates, not a reservoir that continuously supplies additional galaxy attraction. The journal is Science; the NIST publication metadata incorrectly labels it Nature. [Original paper](https://doi.org/10.1126/science.1192720), [author institution record](https://www.nist.gov/publications/relativity-and-optical-clocks)

For a photon moving in a stationary spacetime, the conserved Killing energy obeys \(E_\infty=N E_{\rm local}\), given the chosen normalization. Different local observers therefore report different photon energies. This is compatible with conservation; it is not evidence that static time dilation emits power. A varying field or actual interaction may exchange energy, but its source and recipient must be included.

## An explicit way to explore the hypothesis

Time is a coordinate/proper-duration variable in established GR, not an independent energy density. We can nevertheless propose a new physical field that changes clock rates and exchanges energy. For example, in natural units and signature (-,+,+,+),

\[
S_\chi=\int d^4x\sqrt{-g}\left[-\tfrac12 g^{\mu\nu}\partial_\mu\chi\partial_\nu\chi-V(\chi)\right],
\quad S_m=S_m[A(\chi)^2g_{\mu\nu},\psi_m].
\]

This is an illustrative scalar coupling, not a completed theory or the equation fitted by the parent. Its canonical local field energy includes \(\dot\chi^2/2+|\nabla\chi|^2/2+V\). A nonnegative potential and positive kinetic sign avoid a negative canonical energy at this elementary level. They do not alone prove stability of coupled matter/gravity, causal consistency, screening, or observational viability. A static field may have stored gradient/potential energy without ongoing generation.

Write exchange explicitly:

\[
\nabla_\mu T_m^{\mu\nu}=Q^\nu,\qquad
\nabla_\mu T_\chi^{\mu\nu}=-Q^\nu.
\]

Then total covariant stress-energy conservation follows by addition. In local energy bookkeeping, \(\partial_tu_m+\nabla\cdot\mathbf S_m=Q\) and \(\partial_tu_\chi+\nabla\cdot\mathbf S_\chi=-Q\). Specifying Q is essential; naming time as the source does not determine it. In curved evolving spacetime a global conserved energy additionally needs appropriate symmetry/boundary conditions. The test script checks a two-reservoir exchange, not a relativistic solution. Bekenstein provides a primary example showing how much additional structure a relativistic modified-gravity theory requires; our candidate is not TeVeS. [Bekenstein 2004, corrected version](https://arxiv.org/abs/astro-ph/0403694v6)

## Checked radial candidate proposed by the parent

Let d=Rd>0, B=GM/Psi0>0, beta>=0, and Psi0 have units of velocity squared:

\[
\Phi_\chi(r)=-\beta\Psi_0\ln\left[1+\frac{GM}{\Psi_0(r+d)}\right],
\quad
g_\chi(r)=\frac{\beta GM}{(r+d)(r+d+B)}.
\]

Acceleration is \(-g_\chi\hat{\mathbf r}\). The potential tends to zero at infinity; its derivative has the required positive sign. Choosing \(\Psi_0=\lambda a_0d\) is dimensionally consistent, with dimensionless lambda. A source-derived M and d let this be tested without inserting fitted halo parameters. It remains a spherical empirical extra term added to the parent baryonic radial prediction, not an observed 3D reconstruction or a unique field dynamics.

The effective enclosed source is \(M_{\rm eff}=\beta M r^2/[(r+d)(r+d+B)]\). Its derivative is

\[
\frac{dM_{\rm eff}}{dr}=
\frac{\beta M r[(2d+B)r+2d(d+B)]}{(r+d)^2(r+d+B)^2}>0.
\]

Therefore the spherical effective density \((4\pi r^2)^{-1}dM_{\rm eff}/dr\) is positive and total effective source approaches beta M. These are useful shape and budget properties, not proof of material particles or conserved time energy. The central force approaches a finite nonzero value: effective density has a 1/r cusp and force direction at the origin is undefined. Far out the force returns to r^-2. An intermediate range d<<r<<B has g approximately beta Psi0/r and an approximately flat circular-speed contribution. That range only exists if B is appreciably larger than d.

The integral of squared gradient over volume converges at both boundaries (r^2 g^2 behaves as r^2 near zero and r^-2 at infinity). This only establishes finiteness for a specified quadratic gradient-energy term; its normalization, potential energy and exchange coupling remain to be supplied. This formula has not been derived from the illustrative canonical action above.

## What real galaxy data can distinguish

SPARC provides stellar light, modeled gas/stellar radial force contributions and observed rotation curves. Conditional on distance, inclination, mass-to-light ratio and circular-motion assumptions, it can compare source-based static radial predictions on held-out galaxies. [SPARC measurement paper](https://arxiv.org/abs/1606.09251)

It does not directly provide galaxy clock comparisons, time-dependent field responses, full three-dimensional mass maps or both metric potentials required for lensing. Rotation curves constrain roughly r dPhi/dr in the disk plane; an additive potential constant is invisible. They cannot determine whether an identical static force came from memory, a static kernel or another mechanism.

For example, tau u_dot+u=S has equilibrium u=S for every positive tau. A damped field mode u_ddot+gamma u_dot+omega^2 u=S has equilibrium S/omega^2 independent of damping. The six tests verify these degeneracies and stable roots for the declared positive coefficients; they do not estimate a time constant. Do not fit tau to a single static curve and label it measured age/storage. Delays require temporal information or a specified non-equilibrium source history with independent constraints.

Promising: test the finite radial potential on real curves and require improvement across galaxies without fitted halo inputs. Challenging: the clock interpretation is indistinguishable from a potential reparameterization until it predicts independent clock, lensing, temporal or environmental measurements. An energy-transfer claim additionally needs explicit dynamics and balance; a better rotation-curve fit cannot supply that missing derivation.
