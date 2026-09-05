# Gravity source and derivative audit

The inherited order-80 disk representation is unsuitable as a verified derivative
source for extending the length-action scan. An independent Newtonian midplane
integral passes the registered refinement and density-consistency targets. It
finds small radial-force errors alongside much larger Hessian errors. No gravity
family is rejected and no new observational score is added by this audit.

## Angular source resolution

The two fixed sources are the same positive regular-core surface profiles and
sech-squared vertical lifts used previously: nominal and half-height. We project
only even Legendre modes, as required by their exact assumed reflection symmetry.
Projection uses 2,048/4,096 Gauss nodes in the positive hemisphere; separate
evaluation uses 4,096/8,192. The twelve registered spherical radii cover 0.25–35 kpc.
Reported errors are the maximum across these shells, not a volume-integrated
negative-mass fraction and not an observational residual.

| Conditional source | Order | Density L1 fraction | Gradient L1 fraction | Negative density fraction | Within all targets |
|---|---:|---:|---:|---:|---|
| Nominal disk | 80 | 2.14189 | 1.88457 | 0.654045 | no |
| Nominal disk | 640 | 0.0672871 | 0.272372 | 0.0186651 | no |
| Nominal disk | 1280 | 0.000423307 | 0.00332824 | 9.48902e-05 | yes |
| Nominal disk | 2560 | 8.57773e-09 | 1.33271e-07 | 1.4261e-09 | yes |
| Half-height disk | 80 | 2.72028 | 1.70284 | 0.75488 | no |
| Half-height disk | 640 | 0.69489 | 1.56097 | 0.236422 | no |
| Half-height disk | 1280 | 0.0758934 | 0.32409 | 0.0228953 | no |
| Half-height disk | 2560 | 0.000478697 | 0.00400428 | 0.000122155 | yes |

The first sampled order meeting every target is 1,280 for the nominal disk and
2,560 for the half-height disk. Density and gradient L1 targets are 1% and 5%;
negative projected density must integrate to less than 0.5% of the physical
hemispheric density, and quadrature changes must remain below 0.001 in these
dimensionless metrics. These finite targets neither prove pointwise positivity
nor validate a full gravitational field. At order 80, the worst density shell is
35 kpc for both sources. The small integration changes identify angular
truncation as the dominant measured error.

Initial run 001 failed a synthetic gradient tolerance: 2.35e-12 versus 1e-12.
Small floating-point projections of a constant into analytically absent high
modes are amplified by differentiation at the smallest test radius. The retained
correction uses 1e-11 for that synthetic projection test and separately checks
the exact supplied polynomial coefficients to 1e-14. No source-audit target was
loosened, and run 001 remains intact.

## Independent Newtonian midplane reference

The reference integrates the source in cylindrical coordinates. The general
separable-source Green representation follows
[Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
It does not use a spherical multipole expansion. Potential, force, Hessian and
third derivatives are differentiated from the same finite-wavenumber integral.
Its trace is compared with physical density; physical density is never substituted
into a truncated Hessian to force that comparison to pass.

There are 24 retained jets: two heights, three cutoffs (100, 200, 400 per kpc),
two radial source rules (64, 128 nodes per interval), and two wavenumber rules
(16, 32 nodes per 0.5/kpc interval). All measured surface-profile knots, the core
join and the outer taper boundaries are integration edges. Radial transforms
are shared only after checking that the two sources differ solely in height.
All component fields are summed before constructing nonlinear tensor invariants.

Both sources pass all three separate refinements. Maximum changes across them:

- Radial force: 6.956328e-07 of the reference force.
- Hessian: 4.460921e-07 of its tensor norm.
- Radial Hessian derivative: 0.0004270604 of the registered scale.
- Physical density consistency: 4.685713e-08 fractional error.
- Physical density-gradient consistency: 0.000431009 of the registered scale.

Third-derivative comparisons use max(norm(dH/dR), norm(H)/R); density-gradient
consistency uses max(abs(4*pi*G*density_R), norm(H)/R). This retains zero-crossing
probes. These are engineering accuracy criteria for the fixed assumed source,
not uncertainty bounds on the source inferred from observations.

## What the earlier force gate missed

Maximum discrepancies over registered midplane probes at or within 20 kpc:

| Source | Earlier grid | Radial force | Hessian norm | Gradient of H:H |
|---|---|---:|---:|---:|
| Nominal disk | inherited_coarse | 0.3842% | 64.61% | 80.89% |
| Nominal disk | inherited_fine | 0.1377% | 47.41% | 60.23% |
| Half-height disk | inherited_coarse | 0.4501% | 81.24% | 94.69% |
| Half-height disk | inherited_fine | 0.2656% | 70.37% | 87.01% |

The last column is scaled by max(abs(gradient(H:H)), (H:H)/R). Across all twelve
probes, the fine nominal/thinner Hessian discrepancies reach 63.0%/80.4%; fine
force discrepancies reach 2.18%/3.10% at 35 kpc. A nearly converged radial force
therefore does not establish the derivative accuracy required by this action.
The earlier conditional galaxy scores remain recorded, but cannot promote a
physical card. The result strengthens the existing numerical limitation; it is
not evidence against all length-dependent laws.

## Verification and next step

Eight new synthetic tests cover finite-polynomial projection, unclipped negative
reconstruction, adaptive sech-squared integrals, exact Gaussian Hankel transforms,
source partition, a three-dimensional spherical Gaussian potential and its
derivatives, retained finite-cutoff density error, and invalid inputs. All 209
focused tests and the updated workflow's lint command pass locally. The separate
verifier checked 60 exact input snapshots,
48 source coefficient integrals,
and 15 radial-transform spot integrals with adaptive quadrature. Maximum normalized
disagreements are 1.03207e-09 and
6.16155e-16, respectively. Normalizations
and individual values are retained; these checks do not certify unsampled space.

Next, extend the independent integral to off-plane Newtonian derivatives and
validate their source identity and convergence on the full domain feeding the
nonlinear field equation. Then validate that equation's separate Poisson solve.
A midplane action flux alone is not the physical modified disk acceleration.
Only after those checks should a new, globally fixed length grid be registered
and evaluated again in the local, cluster and galaxy regimes.

Source uncertainties, nonaxisymmetric structure, external environment, direct
outer-star observables, photon dynamics, stability and untouched confirmation
remain open. New response scores, physical rejections and quality-verified
cross-regime candidates added here are all zero. The discovery goal stays active.

## Evidence

- Angular result: `044182ba05ee79786112a3cb745c2393816ff209dc4b1c00c492f729adbf9c2c`.
- Midplane result: `24c594de9e07033e1becd8042d50d1d111969f0823201cbeb4cc64329ce70741`.
- Adaptive verification: `ff8a19d57496043dc708da1f3a258c7c0a886116099f4200c292ff6c7f3ae4c1`.

Run directories are under `work/gravity-first-principles/`. Each has its executed
configuration, inputs or verifier snapshot, raw diagnostics and receipt. The
standalone JSON summary and PNG/SVG figure accompany this report.
