# Lane 6: full-3D nonspherical fields — response-blind findings

## What was actually executed

The packet rebuilt all 225 frozen source reconstructions for NGC 2903, NGC 3351,
and NGC 3627 from the 21 acquired S4G, THINGS, and PHANGS source files. Every
source is labeled `MODEL_LIFTED_2P5D_SOURCE_IN_FULL_3D_FIELD_SOLVER`; none is an
observed three-dimensional mass distribution.

Eight mechanisms received the same 17-by-17-by-17 density grid and the same
fixed-radius observable projection: Newton, a source-matched NFW geometry control,
AQUAL, QUMOND, the published DiskMass median Refracted Gravity law, the frozen
GP01 elliptic representative, the published Mashhoon/Rahvar q0 nonlocal force, and
the new GQNS geometry-conditioned nonlocal source operator. The projection contains
mean radial force, azimuthal RMS, m=1 through m=4 force harmonics, and the vertical
force one grid cell off the midplane at 5, 10, and 15 kpc.

No velocity, lensing, or other scientific response was opened, no parameter was fit,
and no score was calculated.

## The new executable branch

GQNS is defined by

`rho_D = A_Q K_L * rho_b`,

or equivalently

`(lap - L^-2) rho_D = -A_Q L^-2 rho_b`,

followed by `lap Phi = 4 pi (rho_b + rho_D)` in the normalized solver units.
`A_Q` is the bounded anisotropy of the full three-dimensional baryonic second-moment
tensor, and `L` is the baryonic RMS radius. Both are computed from the source; there
is no fitted gravitational parameter.

This construction makes one sharp prediction: the modification is exactly absent for
a spherical source but turns on for a flattened, barred, spiral, or otherwise
anisotropic source. It is an instantaneous, noncovariant phenomenological operator,
not yet a complete theory of gravity.

The nine target-free tests all pass:

- published q0 point-force identity: exact at the recorded precision;
- independent Helmholtz manufactured solution: 0.1806% solution error and
  7.63e-15 residual;
- spherical GQNS activation: 3.13e-16;
- bar activation: `A_Q=0.7517`, with a 30.61% potential-level distinction;
- spiral activation: `A_Q=0.4817`, with a 26.38% distinction;
- thin-versus-thick ordering: `0.4820 > 0.1904`;
- symmetric saddle central field: 5.49e-15;
- uniform external-field superposition error: 2.75e-15;
- rotation-covariance error: 6.43e-16.

## Frozen real-source predictions

Across the 216 primary Cartesian source-systematic cells, GQNS predicts a radial
force relative to Newton of:

| Radius | Minimum | Median | Maximum |
|---|---:|---:|---:|
| 5 kpc | 1.0690 | 1.0889 | 1.0954 |
| 10 kpc | 1.1320 | 1.1728 | 1.1864 |
| 15 kpc | 1.2143 | 1.2756 | 1.2937 |

These are predictions from source geometry, not evidence of agreement with observed
motion. The apparent relation between `A_Q` and force enhancement across the 216 rows
must not be assigned a 216-object significance: those rows are correlated source
variants of only three galaxies.

The primary-cell values are:

| Object | `A_Q` | GQNS/Newton at 5 kpc | 10 kpc | 15 kpc | Numerical status |
|---|---:|---:|---:|---:|---|
| NGC 2903 | 0.5033 | 1.0741 | 1.1414 | 1.2297 | retained resolution counterexample |
| NGC 3351 | 0.5065 | 1.0921 | 1.1749 | 1.2783 | source-screen pass |
| NGC 3627 | 0.5145 | 1.0953 | 1.1863 | 1.2935 | source-screen pass |

NGC 2903 is not hidden or removed: its 17-node Newton profile differs from the bound
25-node bridge by 20.96%, just beyond the frozen 20% gate. AQUAL differs by 2.68% and
QUMOND by 16.64%. Thus the packet has 224 passing cells and one retained predecessor
counterexample.

## Nearest known structures and the exact distinction

- Mashhoon/Rahvar q0 nonlocal gravity also uses a positive spatial convolution
  `rho_D=q*rho_b`, but its amplitude and range are universal fitted constants
  (`alpha_0=10.94`, `mu_0=0.059 kpc^-1`). GQNS derives activation and range from
  source geometry and shuts off for a sphere.
- GP01 elliptic smooths a local-acceleration gain and changes the gravitational
  permittivity. GQNS instead smooths the baryonic source and uses a global
  quadrupole invariant.
- AQUAL and QUMOND respond nonlinearly to acceleration and an external field. GQNS
  exactly superposes a uniform external field in this static operator.
- Refracted Gravity uses a local density-dependent permittivity. GQNS is a global
  source-geometry coupling.

The repository search found these close families, but no exact zero-fit
baryonic-quadrupole-determined reciprocal source kernel with spherical shutoff.
That makes GQNS a potentially distinct synthesis, not established historical novelty.

## Source readiness and next falsifier

All eight static mechanisms are executable on the 225 model-lifted source cells.
Measured 3D sources and observed external tidal fields remain source-blocked. GP01,
Refracted Gravity, and GQNS also lack the multisector closure needed to infer photon
propagation or lensing from their matter fields.

The most direct falsifier is a response-frozen matched pair: one round unbarred disk
and one strongly barred or two-arm disk with comparable baryonic mass and Newtonian
acceleration, identical S4G+H I+CO source construction, and an independent resolved
two-dimensional velocity field. GQNS predicts a geometry-linked change in force
harmonics; the published isotropic q0 law does not switch off with roundness.

The existing three galaxies can be used for a development-only score of the frozen
predictions. A publication-strength claim needs an unopened matched round/barred pair
and a higher-resolution replay, especially for NGC 2903.
