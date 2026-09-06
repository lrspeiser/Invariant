# Static weak-field light-projection benchmark

Disposition: **THEORY_BENCHMARK_ONLY**, declared in `preimplementation-declaration.json` before the implementation files existed. The earlier `mond-atlas-lensing-pilot-001` files and receipts remain frozen. This work admits **zero observational likelihoods** and establishes no relativistic sigma/MOND/refracted-gravity closure.

The standalone operator accepts two scalar potential callables, numerically differentiates their sum transversely, integrates along a straight line of sight, and applies a separately supplied angular-distance geometry. It also evaluates the angular lens map and its numerical two-dimensional Jacobian. It does not solve a gravitational field equation or infer distances.

## Equations, conventions, and primary sources

The local metric has signature `(-,+,+,+)`:

```text
ds^2 = -(1 + 2 Phi/c^2)c^2 dt^2
       + (1 - 2 Psi/c^2)(dx^2 + dy^2 + dell^2)
alpha_hat(xi) = (1/c^2) integral grad_perp(Phi + Psi)(xi,ell) dell
beta(theta)   = theta - (D_ls/D_s) alpha_hat(D_l theta)
A_ij          = partial beta_i / partial theta_j
mu_signed     = 1/det(A)
mu_flux       = abs(mu_signed)
```

Here `x,y` are physical transverse Cartesian coordinates, `ell` is physical longitudinal distance, `xi=D_l theta`, and all angles are radians. Potentials have units `m^2/s^2`, gradients `m/s^2`, distances meters, and mass kilograms. Integrating the gradient over length and dividing by `c^2` gives a dimensionless angle. For `Phi=-GM/r`, `alpha_hat` points outward from the mass; the actual propagation-direction change is `-alpha_hat`. These signs define the convention rather than relying on ambiguous verbal descriptions of a bending vector.

The principal fixtures explicitly impose **Psi=Phi as a benchmark assumption**. Manufactured ratios `Psi=eta Phi`, with eta fixed to 0, 1, 2, and -1 before execution, test missing factors, sign changes, and cancellation. They are not fitted or selected closures. In particular, equal and opposite contributions to the two potentials cancel from their lensing sum.

Primary bindings:

| Primary source | Equations used | Translation into this benchmark |
| --- | --- | --- |
| [Lewis & Challinor, 2006](https://arxiv.org/abs/astro-ph/0601594) | 2.1, 2.3, 2.6, 2.11–2.13 | Scalar metric perturbations, Weyl potential and Born null-geodesic displacement. Their `(+---), c=1` notation maps as `Psi_N=Phi/c^2`, `Phi_N=-Psi/c^2`; their Weyl potential is `(Phi+Psi)/(2c^2)`. Their source-minus-image displacement has the opposite sign to the reduced deflection here. |
| [Narayan & Bartelmann, version 2](https://arxiv.org/pdf/astro-ph/9606001v2) | 4–7, 9–14, 17, 20–21, 23–28 | Potential-gradient integral; point deflection; projected density and enclosed mass; separately supplied angular-diameter distances; point images and magnifications. Signed magnification is evaluated from the determinant form; total flux uses absolute magnifications. |
| [Werner & Evans, 2006](https://arxiv.org/abs/astro-ph/0602368) | 3–8 | Plummer density, projected density, critical surface density, lens potential, lens map and Jacobian. |

The primary PDF byte receipts, retrieval times, URLs, versions and SHA-256 hashes are in the frozen config and private reference cache. The PDFs were used for theory equations; no new observational table or image was inspected. Preparation failures are retained separately in `preparation-failures.json`.

For a component of mass M, softening a and transverse offset R, independently integrated equal-potential references are:

```text
rho_Plummer(r)   = 3 M a^2 / [4 pi (r^2+a^2)^(5/2)]
Sigma_Plummer(R) = M a^2 / [pi (R^2+a^2)^2]
M_2D(<R)        = M R^2/(R^2+a^2)
alpha_hat       = (4 G M/c^2) R_vector/(R^2+a^2)
```

For a point mass take `a=0`. At finite endpoints `u_lo,u_hi` relative to the component's longitudinal center, the exact general-eta integral is `(1+eta)GM/c^2 * R_vector/(R^2+a^2) * [u/sqrt(R^2+a^2+u^2)]_lo^hi`. The numerical operator does not insert this formula or a tail correction.

The independent asymmetric reference uses the two-dimensional surface-density integral from Narayan & Bartelmann Eq. 10, with polar coordinates centered on the evaluation ray. Its radius/area factor cancels the integrable kernel singularity. Gauss-Legendre radial integration and an angular trapezoid integrate the sum of three displaced Plummer surface densities. This route calls neither the potential-gradient code nor the line-of-sight integral. It is checked at `(256,512)` and `(512,1024)` radial/angular orders.

For a point source with `u=beta/theta_E`, the reference images are `(u +/- sqrt(u^2+4))/2` in units of theta_E. The signed references are `1/2 +/- (u^2+2)/(2u sqrt(u^2+4))`. The operator locates both roots numerically and obtains magnifications from the full numerical Jacobian, including the negative-parity image.

## Independently supplied geometry

Before implementation, Astropy 7.1.1 `FlatLambdaCDM(H0=70, Om0=1, Tcmb0=0)` supplied a synthetic fixture at `z_l=0.3, z_s=1.0`:

| Distance | Mpc |
| --- | ---: |
| D_l | 810.0456831502036 |
| D_s | 1254.3882571373822 |
| D_ls | 727.85856308975 |

The benchmark validates these supplied numbers using the separate Einstein–de Sitter closed form `chi(z)=2c/H0*(1-1/sqrt(1+z))`. The operator receives only the three distances. It never reads redshifts, runs a cosmology calculator, or assumes `D_ls=D_s-D_l`. Using that incorrect subtraction would change this fixture's reduced deflection by 38.95%, which the negative control detects. General angular-diameter distances need not be monotone; validation therefore checks finite positive D_l/D_s and nonnegative D_ls without imposing an artificial ordering.

## Numerical scope and results

The derivative uses a fourth-order centered stencil with default step `0.0001 kpc`. Gauss-Legendre quadrature uses `ell=scale*tan(t)`, scale `1 kpc`, and default order 256. Infinite bounds mean an ideal mathematical domain. Finite support is passed explicitly as a half-depth and optional longitudinal origin. Checks reject registered singular point rays, nonfinite potential values, invalid geometry/numerics, and stencil samples where either individual potential exceeds `0.001 c^2`. These sampled checks do not prove global weak-field, thin-lens or Born validity.

`run-001` passed **203 required checks and 23 unit tests**. It retains all **251 check records**, including **48 sweep diagnostics**, of which **27 miss the fixed finest-resolution target**. The misses are 12 finite-tail, five derivative-step, and ten quadrature-order diagnostics. There are zero required failures. No threshold was changed after seeing a result. The exact failed IDs and numeric values are in `run-001/checks.json` and `run-001/summary.json`.

| Check family | Largest observed error or convergence result |
| --- | ---: |
| Point/Plummer radial deflection, 48 rays | 3.47e-11 relative |
| Independent asymmetric surface-density reference | 3.97e-9 relative |
| Exact finite-domain point/Plummer integral | 1.41e-12 relative |
| Projected Plummer density | 2.74e-16 relative |
| Numerically enclosed projected mass | 1.20e-11 relative |
| Point image position | 5.61e-12 in theta_E units |
| Point signed magnification | 5.85e-7 relative |
| Measured derivative convergence orders | 3.985–4.037 |
| Finite-tail asymptotic orders | approximately 2.000 |

Additional gates cover reflection, arbitrary transverse rotation, three-dimensional translation with a translated integration window, mass and length scaling, explicit unit conversion, constant potential offsets, superposition, lens-Jacobian symmetry, zero mass, a regular Plummer central ray, zero distance efficiency, and magnification parity/total flux. The fixture Einstein angle is 0.763790564 arcsec; this is a synthetic geometry result, not a galaxy measurement.

Finite-tail convergence is deliberately separate from finite-integral accuracy. At the declared `(1,0) kpc` ray and half-depth 512 kpc, omitted-tail relative errors are `1.91e-6` for the point fixture and `3.13e-6` for Plummer. This does **not** assert the same tail bound for arbitrary larger impact parameters or real numerical-field boundaries. `benchmark-design.json` fixes these sweep locations and the additional tail-order gate before the first execution.

## Reproduction and exact receipts

Run from the Invariant repository root in PowerShell:

```powershell
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B -m unittest discover -s tests -p test_mond_atlas_light_projection.py -v
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B scripts/run_mond_atlas_light_projection.py --run-id run-003
& 'C:\Users\henry\AppData\Local\Programs\Python\Python313\python.exe' -B work/gravity-first-principles/mond-atlas-light-projection-001/verify_receipts.py
```

Use a previously unused `run-NNN` name: the runner refuses to overwrite receipts. The supplied environment is Python 3.13.5, NumPy 2.2.6 and SciPy 1.16.1. Astropy was used only to supply the frozen distance fixture; it is not an operator or benchmark dependency. This is CPU-only and has no network calls. The three primary theory PDFs must be present at the exact private-cache paths and hashes recorded in the config; they total 2,605,347 bytes. No observational cache is accessed.

Each run retains its exact code/config/test snapshot, unit-test output, every check, environment/version receipt, primary-source hashes, and a SHA-256 manifest. The package `delivery-manifest.json` binds the declaration, design, all code, both run receipts, documentation, and private theory references. `replay-verification.json` records the numerical/check replay comparison; all 251 check records are byte-identical between runs 001 and 002. Changing a threshold requires a new declared benchmark version, not editing this config after execution.

## Admission boundary and missing prerequisites

This closes only a bounded synthetic propagation test. The following remain missing before any observational lensing likelihood or candidate-theory comparison is admissible:

1. A derived, healthy relativistic candidate with explicit matter/photon coupling and both metric potentials, their gauge/normalization, field equations, stress contributions and boundary conditions. Equal potentials must not be transferred from this benchmark into sigma/MOND/refracted-gravity claims without derivation. Existing symbolic conformal-cancellation results remain relevant.
2. Actual candidate-specific three-dimensional potential solutions, independently verified spatial interpolation/derivatives, source normalization and environments, exterior/tail bounds, and checks that the weak/static/Born/single-plane approximations apply. This callable-analytic benchmark does not certify finite-grid fields, multiple planes, post-Born effects, time-dependent metrics or vector/tensor perturbations.
3. Independently sourced lens/source distances and cosmology for the intended sample, with their uncertainty and consistency with the candidate geometry; no fitting or silent reuse of local lens observations to set the geometry.
4. Complete source closure for the exact imaging likelihood: calibrated instrument and spatial/chromatic PSF, pixel response and integration, WCS/registration, masks/sky, flux calibration and a validated correlated-noise model. Prior synthetic correlated-noise tests do not calibrate the actual HST observations.
5. A specified source-brightness reconstruction and nuisance/marginalization procedure with independent operator/reference tests, plus a frozen likelihood, reserve/exposure policy and scoring gates. This benchmark does not reconstruct a source or evaluate image residuals.
6. Independent or correctly conditioned stellar-mass/dynamics inputs and a joint treatment of shared response information. The earlier finding that some SPS priors condition on velocity dispersion is unchanged; this benchmark neither resolves that dependence nor adds independent evidence.

No observational table/image reads, sample exposure changes, observational fitting, PSF/noise admission, common-module edits, handoff edits, or Git mutations were performed for this continuation. Publication remains with the coordinator.
