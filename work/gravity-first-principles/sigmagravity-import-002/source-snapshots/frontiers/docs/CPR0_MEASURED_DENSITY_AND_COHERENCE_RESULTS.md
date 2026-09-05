# CPR0 measured-density and measured-coherence results

## Decision

The present coherence-partitioned Sigma/Refracted-Gravity interpolation does
not advance.  This is now a measured-data result, not a consequence of using
mean enclosed density or of omitting the BCG stellar component.

The strongest surviving empirical observation is narrower: local baryonic
density predicts part of the missing-acceleration trend within one class of
systems.  It does not supply a universal response across galaxy dynamics and
cluster lensing.  The tested MaNGA coherence proxy, beam-corrected
`Lambda_Re`, slightly worsens rather than improves the density-only model.

The failed object is specifically the spherical/algebraic response

\[
g=g_{\rm bar}/\epsilon(\rho_b)
\]

and its CPR0 interpolation with the declared mapping

\[
w(C)=3C^2-2C^3,\qquad
\epsilon_{\rm mix}=w+(1-w)\epsilon_{\rm RG},\qquad
\nu_{\rm src}=1+wB_0h(g_{\rm bar}).
\]

This does not prove that every nonspherical Refracted-Gravity field solution
or every future coherence variable fails.  It does show that another parameter
sweep of this local logistic law is not the next scientific step.

## What was added

The final bridge uses three independent observational products:

- 44 quality-selected MaNGA BCG dynamical points from Tian et al. (2024),
  joined to DynPop's beam-corrected `Lambda_Re`;
- 11,164 radial shells for 242 objects from the ACCEPT Chandra archive
  (Cavagnolo et al. 2009); and
- 84 baryonic and lensing-derived acceleration points in 20 CLASH clusters
  plus their HST BCG masses and sizes from Tian et al. (2020).

The copied ACCEPT snapshot is hash-locked at
`bb762dff0bfff9bb4956cb085cbf51918b3d22040c34623bb9cb09f190f3138d`.
Eighteen CLASH systems have a declared ACCEPT identifier match.  Interpolation
is log density versus log annular-midpoint radius and never extrapolates.

The primary cluster test uses 52 measured points at 100--600 kpc across those
18 systems.  Its density range is
`log10(rho/[g cm^-3]) = -27.214 to -25.392`.  This reduces the unmeasured gap
to the BCG density range from 2.210 dex in the first joint endpoint test to
0.421 dex.  The shared fit can therefore no longer succeed merely by assigning
one constant plateau to galaxies and another to clusters.

The added stellar test deprojects every observed CLASH BCG as the Hernquist
profile used by Tian et al.,

\[
\rho_\star(r)=\frac{M_\star a}{2\pi r(r+a)^3},\qquad a=0.551R_e.
\]

It includes all 20 central CLASH points, producing 72 total cluster points.
Stars provide a median 73.4% of the local central density but only 0.53% beyond
100 kpc.  Thus it is a direct test of the proposed bulge/geometry concern,
rather than an arbitrary global rescaling.

Primary sources: [ACCEPT](https://arxiv.org/abs/0902.1802),
[CLASH accelerations and BCGs](https://arxiv.org/abs/2001.08340),
[MaNGA BCG dynamics](https://arxiv.org/abs/2402.12016), and
[the 100-cluster weak-lensing catalog](https://arxiv.org/abs/1912.04414).

## Complete result sequence

All cross-validation keeps complete galaxies or clusters in one fold.
Reported cluster RMSE first averages squared radial residuals within each
cluster, so a system with more radial coverage does not dominate.

| Test | Sample | Key held-out result | Disposition |
|---|---:|---|---|
| Measured BCG coherence | 44 BCGs | density-only RG 0.08685 dex; CPR0 0.08865 dex | `Lambda_Re` mapping rejected |
| Direct weak-lensing endpoint | 14 clusters | RG 0.05225 dex, but residual-density correlation -0.598 | low-density normalization only; failed gate |
| First shared endpoint | 44 BCGs + 14 clusters | RG 0.10640 equal-domain dex; CPR0 0.10730; density gap 2.210 dex | apparent two-plateau fit, not evidence |
| NFW radial bridge | 44 BCGs + 70 cluster radii | RG cluster 0.15190 dex, radial slope +0.144; CPR0 slightly worse | radial shape failed |
| ACCEPT x CLASH, 100--600 kpc | 18 clusters, 52 points | RG 0.13000 dex vs constant 0.13685; locked transfer 0.14585 | direct local-density law failed |
| ACCEPT gas + observed BCG stars | 20 clusters, 72 points | RG 0.17764 dex; central 0.22773; locked transfer 0.18220 | missing BCG density did not rescue it |
| Final shared gas+stars test | 44 BCGs + 20 clusters | RG: BCG 0.09344, cluster 0.15565, equal-domain 0.12837; CPR0 equal-domain 0.12767 | neither model passed |

No protocol passed all of its frozen advance gates.

## The decisive ACCEPT x CLASH tests

### Outer cluster regime

| Model | Equal-system RMSE (dex) | Mean residual (dex) | Radial slope (dex/dex) |
|---|---:|---:|---:|
| Constant epsilon, cluster CV | 0.13685 | -0.0021 | -0.0447 |
| Density-only RG, cluster CV | **0.13000** | -0.0096 | +0.1492 |
| Earlier joint RG parameters, no refit | 0.14585 | -0.0820 | +0.0505 |
| Published elliptical RG, no refit | 0.12240 | +0.0373 | +0.1446 |
| Shared BCG+cluster RG, grouped CV | 0.12611 cluster / 0.09139 BCG | +0.0029 cluster | +0.0953 |
| Shared BCG+cluster CPR0, grouped CV | 0.12649 cluster / 0.09308 BCG | +0.0032 cluster | +0.0880 |

The cluster-only density law improves the constant control by only 0.00685 dex,
well below the frozen 0.02-dex requirement, and remains above the 0.12-dex
absolute gate.  Its fitted transition density is around -25.2 dex, whereas the
shared galaxy+cluster fit needs about -23.75 dex.  This is a physical transfer
tension, not optimizer instability.

The previously locked shared parameters underpredict the new lensing target by
0.082 dex on average, a factor of about 0.83.  Multiplying every ACCEPT gas
density by 0.6--1.4 changes its RMSE only from 0.1395 to 0.1527 dex; no allowed
density calibration reaches the 0.12 gate.  The published elliptical parameter
set nearly reaches the error gate, but its +0.145 radial residual slope fails
the shape test.

### Central BCG plus cluster regime

Adding the observed BCG density removes the aggregate residual-density
correlation and radial trend, but not the system-to-system predictive failure:

| Region/model | Held-out RMSE (dex) |
|---|---:|
| Central points, RG | 0.22773 |
| Outer points under the same cluster fit, RG | 0.14057 |
| All 72 cluster points, RG | 0.17764 |
| All 72 cluster points, constant epsilon | 0.20289 |
| Shared RG: cluster / BCG / equal domain | 0.15565 / 0.09344 / 0.12837 |
| Shared CPR0: cluster / BCG / equal domain | 0.15462 / 0.09322 / 0.12767 |

The shared CPR0 improvement over density-only RG is only 0.00070 dex, versus
the required 0.01 dex.  Across a 3x3 bracket of gas density (0.6, 1.0, 1.4)
and BCG stellar mass (0.8, 1.0, 1.2), the locked prior RG RMSE stays between
0.1711 and 0.1935 dex.  The conclusion is insensitive to those normalizations.

## What the tests prove and do not prove

They establish that:

1. The declared measured-coherence mapping is not supported by held-out BCG
   dynamics.
2. A universal local logistic permittivity does not predict both BCG dynamics
   and CLASH lensing at the frozen error level.
3. Omitting central BCG stellar density was not the cause.
4. The earlier good one-radius cluster score was mainly a low-density
   normalization result; radial and intermediate-density data remove that
   apparent success.
5. Published galaxy, elliptical, and individual-cluster RG parameters do not
   provide a satisfactory universal transfer on these constructions.

They do not establish that:

- the full nonspherical PDE has been tested in disks;
- a relativistic CPR0 metric predicts the CLASH photon signal, because no such
  metric has been derived;
- every possible independently measured coherence or geometry variable fails;
  or
- CLASH's spherical NFW deprojection and diagonal public errors are the final
  cluster likelihood.

## Most promising next direction

Do not add another free exponent to `epsilon(rho)`.  The data already span
nearly three density decades in clusters and bridge to the BCG range.  A new
formula must introduce a physically measured second invariant and predict a
different observable, not just refit the same acceleration table.

The best next empirical discriminator is a same-system two-potential test:

1. obtain a public radial lensing likelihood with covariance;
2. obtain galaxy-member or BCG kinematics over overlapping radii; and
3. predict both with one baryonic model, thereby measuring whether the two
   metric potentials have the required gravitational slip.

Highest-value additions are the per-cluster CLASH surface-density profiles and
full covariance from the Umetsu et al. joint strong/weak-lensing analysis,
spatially resolved CLASH-VLT member/BCG kinematics, and measured satellite plus
intracluster-light density profiles.  These address the actual remaining
limitations.  Another X-ray density catalog covering the same radii would be a
calibration cross-check, not a new theory test.

If a second state variable is developed, its value and mapping must be frozen
from data independent of the gravity residual.  Candidate measurements include
velocity anisotropy, X-ray centroid shift/power ratios, and three-dimensional
shape.  `Lambda_Re` must remain a negative control because its predeclared CPR0
mapping failed.

## Reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/run_cpr0_manga_bcg_coherence.py
python scripts/run_cpr0_cluster_lensing_density.py
python scripts/run_cpr0_joint_bcg_lensing.py
python scripts/run_cpr0_radial_lensing_bridge.py
python scripts/run_cpr0_accept_clash_bridge.py
python scripts/run_cpr0_accept_clash_bcg_stellar.py
python -m pytest tests/test_accept_profiles.py tests/test_host_profiles.py `
  tests/test_sigma_refracted.py tests/test_cpr0_radial_lensing_bridge.py -q
```

Machine-readable outputs are in `results/cpr0_*`; the two final protocols are
`configs/cpr0_accept_clash_bridge_protocol.json` and
`configs/cpr0_accept_clash_bcg_stellar_protocol.json`.
