# Joined tensor potential: source and derivative checks

The fine interpolated source now passes the registered sampled force,
Hessian, third-tensor, density and density-gradient checks after its 60--80 kpc
exterior join, for both retained source thicknesses. Independent derivative
checks pass with stencils appropriate to C3 cell boundaries. **246 focused
tests and lint pass.** This does not yet admit the complete source for new
astronomical predictions: actual-source quadrature refinement and the separate
full-action Poisson solve remain pending.

## One potential, including the join

`MatchedTensorPotential` removes a constant while interpolating, restores that
same constant before joining, and derives all joined fields using the existing
full product rule. Beyond 80 kpc it evaluates only the admitted exterior
potential. No density or derivative is replaced by a desired physical value.
An exact quadratic control checks the gauge, join, symmetry and exterior-only
dispatch beyond the interpolation domain.

The audit retains all 1,364 previous source probe pairs per thickness. It adds
198 fixed probes per thickness at the center, symmetry axis, midplane, source
interfaces and near/beyond the join. Some probe sets overlap. The coarse
representations still fail; the following results concern the fine grids.

| Source | Density error, retained probes | Density-gradient error, retained probes | Density-gradient error, additional probes |
|---|---:|---:|---:|
| Primary | 0.000100667881 | 0.000466401686 | 0.000315611474 |
| Half thickness | 0.000128032699 | 0.00681675229 | 0.00659516315 |

Density errors use max(|4 pi G rho|, Hessian norm). Density-gradient errors
use max(|4 pi G grad rho|, Hessian norm/(radius + minimum source height)).
The fixed targets remain 0.002 and 0.01. These compare the potential's own
trace and trace gradient with the unchanged physical source; they are not
new observations or uniform continuum bounds.

## Failed audit retained, checker repaired

`matched-tensor-001` passed its sampled source identities but failed the
independent derivative diagnostic. Two effects were separated:

1. At the exact center, weighting and dividing large potential samples before
   summing introduced a spurious residual in a derivative that symmetry makes
   zero. The checker now subtracts paired samples first. One-sided differences
   subtract their common base before weighting.
2. At R=36 kpc, a central difference crosses a C3 interpolation boundary. A
   C3 potential has continuous third derivatives but need not have continuous
   fourth derivatives, so this finite-difference estimate converges only
   linearly there. Halving the step approximately halves the observed error.
   This is not evidence of a discontinuous third derivative.

The revised audit retains central results and evaluates **both** left and
right fourth-order one-sided stencils at positive radial tensor boundaries,
using central stencils elsewhere. A separate piecewise-quartic control
demonstrates the expected central-stencil failure and accurate one-sided
derivatives. No error tolerance, physical source or interpolated field was
changed to repair the checker. The original failed verdict remains preserved.
Central stencils that cross the cutoff still exceed the target at the chosen
finite step; they are not relabeled as passing.

| Source | Scalar-to-gradient error | Gradient-to-meridional-Hessian error | Hessian-to-third-tensor error |
|---|---:|---:|---:|
| Primary | 6.47652055e-10 | 5.57289669e-09 | 1.61520809e-05 |
| Half thickness | 7.3363249e-10 | 5.56846099e-09 | 1.49086756e-05 |

These are the maxima across both qualified stencils at h=0.0005 kpc; h=0.001
results are also retained. The unchanged scaled target is 1e-4. The test covers
the meridional Hessian and all six nonzero independent third components via
Hessian differences. It does not independently finite-difference every
three-dimensional Hessian entry from the scalar potential. The azimuthal
geometry also has existing analytic controls and source-identity checks.

## Next required work

Independently vary the actual-source radial/wavenumber/vertical quadratures
for the 16 mixed partials while retaining the interpolator, both thicknesses
and every probe. Then evaluate and refine the separate full-action Poisson
solve; a variational flux is not itself the physical acceleration. Only after
numerical qualification can the global parameter cards be rescored against
galaxy data and compared across clusters and the Solar System.

Light coupling, dynamical stability, direct outer-star observations, a full
Solar System fit and independent confirmation remain open. This checkpoint
adds zero observational scores, validated laws or physical exclusions.

## Evidence

- `matched-tensor-001`: `e6d4eb580cfd3eea9d4dd949288488da0d8ac8c786c556d0c4151a800e33b3dc` (49 verified input snapshots)
- `matched-tensor-002`: `a8dbc3d8a86914d63912fa7841b159aebc54158603abad2ed36c4a5543e9d760` (49 verified input snapshots)
- `tensor-controls-003`: `14b6df20ff8c12dbd52fddb3581d7599c7e1b519d8b291bcd3caf3d2af4a4edb` (124 verified input snapshots)
