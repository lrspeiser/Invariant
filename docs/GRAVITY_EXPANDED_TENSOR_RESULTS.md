# Expanded source interpolation and action-flux diagnostics

The refined numerical representation passes its sampled force, Hessian and
third-derivative targets over a 0..80 kpc rectangle for both retained source
thicknesses. This is an unjoined source interpolation test. It does not yet
admit the complete source provider, solve the modified field equation, or
validate a gravity law. **243 focused tests and lint pass.**

## A failed grid led to a concrete refinement

The first expanded primary grid used 85 x 12 coarse and 169 x 23 fine nodes.
At R=36.5 kpc just outside the source cutoff, the fine grid had scaled Hessian
error 0.00348152 and third-tensor error 0.0508213, above the fixed 0.002 and
0.01 targets. The coarse grid failed all three field targets. This remains a
numerical failure in `tensor-source-002`; it does not falsify a gravity family.

The next registration adds coarse radial nodes every 0.25 kpc between 30 and
40 kpc and bisects every coarse cell for the fine grid. No physical mass
profile, quadrature, gravity parameter or error tolerance was changed.
All earlier probe coordinates are retained, together with quarter-cell probes
of the new grid. Some retained probes now lie on grid nodes; the new quarter-cell
probes remain off-grid on both resolutions. Both thicknesses use the same
coordinates: **1364 probes each**, with
111 x 12 coarse and 221 x 23 fine nodes. The coarse grids still fail; only
the fine grids pass the sampled force/H/T requirements.

| Fine-grid source | Largest scaled force error | Hessian error | Third-tensor error |
|---|---:|---:|---:|
| Primary | 3.1954811e-06 | 7.70627622e-05 | 0.000312393843 |
| Half thickness | 3.17234813e-06 | 0.000127778957 | 0.0029724223 |
| Fixed target | 1e-4 | 0.002 | 0.01 |

Force/Hessian errors use reference norms; the third-tensor denominator is
max(reference third norm, H/(spherical radius + minimum source height)).
The primary worst third error is at R=32.8125, z=1 kpc; the half-thickness worst
is at R=77, z=0.025 kpc. These are interpolation discrepancies against direct
evaluations of the retained quadrature, not a uniform error bound or an
independent qualification of that quadrature at every new point.

## Propagation through the 54 unchanged actions

We evaluated the full variational flux using the gradient, complete
three-dimensional Hessian, gradient of its norm and gradient of its trace.
The parameter cards are byte-verified against the retained Solar-System audit.
Six new controls check cylindrical-to-polar covariance against the existing
action implementation for all three shapes and zero/nonzero lengths.

Every card remains within the separately registered 1% flux diagnostic
tolerance. On the refined fine grids the worst full-flux discrepancy is
3.13577863e-06
for the primary source and
3.11294818e-06
for half thickness. Errors are normalized by the reference **full flux**,
with a floor of 1e-10 times the median Newtonian gradient norm.
This does not establish relative accuracy of a small anomalous term.

Crucially, all 54 cards also passed this flux tolerance on the earlier
source grid that failed the derivative requirements. Therefore a flux pass
cannot replace source qualification. No flux has been substituted for the
physical acceleration: the separate Poisson solve remains required.

## Remaining gates

- Restore the original potential gauge before joining to the exterior:
  saved interpolated potentials have a recorded constant subtracted.
- Validate the complete 60--80 kpc join, source-density identities, symmetry
  axes, midplane and interfaces with independent derivative controls.
- Refine actual-source quadratures of all 16 mixed partials separately from
  interpolation. Retain both thicknesses and all failing probes.
- Then run and refine the separate full-action Poisson solve before any
  astronomical rescoring or widening of the global length parameter.
- Cluster matter/light, precision Solar System fits, stability, a complete
  light sector, direct outer-star observations and independent confirmation
  remain open. There are no new observational scores or physical exclusions.

The original expanded run's configuration retained legacy uniform-spacing
labels even though its explicit nonuniform coordinate arrays controlled the
execution. Subsequent runs remove those misleading labels. The first run's
exact code, arrays, outputs and failed verdict are preserved unchanged.

## Reproducible evidence

- `tensor-source-002`: `34044d0e0d49ccc6e7da8a27573d18833e9dcb20968e85df7b71139a8682021c` (44 verified input snapshots)
- `tensor-source-003`: `41f956064f275472550a03a3a663b792a7171b9d25f68aa08b8e331f6a17d9f0` (44 verified input snapshots)
- `tensor-source-004`: `44965a3659efa9dc8f2129452736a2be7b63c00f1804d080e7a076608ac72e62` (44 verified input snapshots)
- `tensor-flux-001`: `292f573b2c492ce56be23b658309a233a8d6d3e6aed7f16c83210075ec11372b` (41 verified input snapshots)
- `tensor-flux-002`: `297f148126b07d9f6670566bb1635156ca85b8434f5181632de71996655dc224` (41 verified input snapshots)
- `tensor-flux-003`: `6c85812bd5825b4422db077c7719ecdfaca43596269fadd583b44ed2229f4b89` (41 verified input snapshots)
- `tensor-controls-002`: `eef82ce2c648a0a669ff1447e9f4a9c93bc114d3cfca0f728b5e43b6931eb45e` (121 verified input snapshots)
