# Off-plane Newtonian reference for the gravity search

Both fixed galaxy-source variants meet every registered off-plane numerical
target. The implementation now evaluates the potential, both force components,
the full axisymmetric Hessian, all six independent nonzero Cartesian third
derivatives, and the gradients of its trace and tensor norm. This removes a
specific derivative-implementation obstacle on the sampled domain. It does not
establish a successful gravity law or validate the full isolated nonlinear solve.

## Test scope and result

The 273 registered locations combine 13 cylindrical radii (0–35 kpc) with 21
heights (−32 to +32 kpc). They include the axis, midplane and reflection pairs.
The sources are the same nominal and half-height NGC3198 assumptions as the prior
audit. Fourteen field configurations retain 3,822
evaluations. The radial transforms are exact byte-checked predecessor artifacts;
no velocity, new gravity parameter, raw source spectrum or reserved observation
was opened. These are numerical checks of a conditional source.

The reference uses 128 nodes per radial source interval, 32 nodes per 0.5/kpc
wavenumber interval, a cutoff of 400/kpc, and 2,400 vertical spline intervals
over 0–24 scale heights. Five separate refinements change radial quadrature,
wavenumber quadrature, cutoff, vertical interpolation and the height at which an
infinite exponential tail is attached. Cutoff 100/kpc is a retained stress case.

| Source | Changed setting | Role | Force error | Hessian error | Third tensor error |
|---|---|---|---:|---:|---:|
| primary | radial_coarse | refinement | 4.70697e-09 | 9.78679e-07 | 2.01218e-05 |
| primary | wavenumber_coarse | refinement | 4.74479e-08 | 3.8844e-07 | 1.08003e-06 |
| primary | cutoff_200 | refinement | 4.31207e-07 | 2.04132e-06 | 0.000773136 |
| primary | cutoff_100 | stress | 3.87717e-06 | 9.39981e-06 | 0.00232584 |
| primary | vertical_coarse | refinement | 4.35496e-09 | 8.15233e-09 | 4.96445e-06 |
| primary | tail_extent | refinement | 8.57786e-15 | 8.23769e-15 | 8.99552e-15 |
| height_half | radial_coarse | refinement | 7.60213e-09 | 8.28502e-07 | 2.43525e-05 |
| height_half | wavenumber_coarse | refinement | 4.82489e-08 | 4.04546e-07 | 1.09202e-06 |
| height_half | cutoff_200 | refinement | 6.95633e-07 | 1.79328e-06 | 0.000527324 |
| height_half | cutoff_100 | stress | 6.2322e-06 | 8.20774e-06 | 0.00158045 |
| height_half | vertical_coarse | refinement | 3.86675e-09 | 6.78279e-09 | 5.4108e-07 |
| height_half | tail_extent | refinement | 5.79702e-15 | 1.30278e-14 | 2.9064e-14 |

The force, Hessian and third-tensor targets are 1e-4, 0.002 and 0.01. Every
mandatory refinement passes for both sources. The largest third-tensor change is
0.00077314, driven by the cutoff refinement; the 100/kpc stress case reaches
0.00232584. The largest reference density-identity discrepancy is 5.2470e-7.
The largest density-gradient discrepancy is 0.00077610, at R=0.25 kpc,
z=−0.025 kpc for the nominal source. The corresponding half-height maximum is
0.00053288 at R=0.25, z=0. These are errors in representing the assumed source,
not residuals against an astronomical measurement.

Force errors use the reference vector norm with a tiny fixed characteristic
floor at its zero. Tensor errors use the full Cartesian Frobenius norms. Third
derivatives use max(norm(T), norm(H)/(spherical radius + minimum source height)).
Density identity errors use max(abs(4*pi*G*rho),norm(H)); gradient errors use
max(norm(4*pi*G*grad rho), norm(H)/(spherical radius + minimum height)). Thus
points with vanishing physical density far from the plane remain in the checks.
The precise normalizations and all per-point values are in the frozen record.

Reflection errors stay below 1.172e-15. At the midplane, the new calculation
agrees with the preceding independent reference within 3.281e-11 on the stated
force, Hessian and radial-derivative scales.

## One source and one potential for all derivatives

The disk Green representation follows
[Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
The implementation extends the previous midplane specialization by calculating
the height-dependent Green convolution and its derivatives from one explicit
vertical source. A cubic spline approximates the normalized sech-squared lift
on its positive half; an exponential matched in value and first derivative
continues to infinity. Exact reflection supplies its other half. The complete
source is normalized to unit mass before use.

Directly subtracting k-squared times the potential kernel from a large local
source term can lose precision in high derivatives. Instead, exact exponential
moments integrate each polynomial source derivative. The third derivative includes
the weak contribution from the small second-derivative jump at the spline/tail
join. The code therefore differentiates one potential throughout, including the
tail. It does not replace a failed numerical trace by the physical density.

The fine vertical source differs from the physical lift by at most 4.168e-10
of peak density and 1.251e-7 of peak density per scale height for its first
derivative on the retained check grid. The independently supplied physical density
and gradient are still used to test the final three-dimensional Poisson identity.

## Verification

Six new synthetic tests cover exact exponential moments, equal decay rates,
adaptive direct Green integrals, the splice contribution, reflection, normalized
mass, the exact sech-squared midplane limit, full Cartesian Gaussian tensors,
source partition, distance homology and invalid inputs. The Gaussian tensor
control is derived from enclosed mass rather than the cylindrical formulas.
All 215 focused tests and the updated workflow's lint command pass locally.

The separate verifier loads the executed package from
37 byte-checked snapshots under an isolated
module name. It evaluates fourth-order finite differences of the Hessian, its
trace and its squared norm at all 273 locations for both sources and two step
sizes, 0.001 and 0.0005 kpc. Through the axis it uses a signed Cartesian extension
with the correct even/odd tensor parity. No axis points are dropped.
The finest differences agree within 3.445547e-08 on the registered
scales. This verifies derivative consistency; it does not independently establish
the true astrophysical mass distribution or an unsampled global error bound.

## Next work toward the universal law

Use the validated integral as a reference while establishing an accurate source
provider over the full isolated domain, including the outer boundary and any
interpolation between points. Then propagate source errors through the full
action flux and its separate Poisson solve. A local action flux is not itself
the physical modified disk acceleration.

Only after that validation should a wider length grid be registered and carried
through Solar System, cluster and galaxy tests with the same constants.
The earlier 54-card comparisons remain conditional; no source or theory is
discarded on the basis of this audit. Source uncertainty, outer-star observables,
photon coupling, dynamics and stability, and untouched confirmation remain open.
The full discovery goal remains active. Added observational scores, physical
rejections and validated universal laws are all zero.

## Evidence

- Off-plane result: `1df5c6356c2f50d079fd29ef826d46cfc86daee3fff506bb51de66ce1b540b52`.
- Derivative verification: `58c6cfe5f68217dade78c3b47160dfe41b7d783156e1e2b80bb1aa42e471e695`.

Full evidence is in `work/gravity-first-principles/hankel-offplane-001/` and
`work/gravity-first-principles/hankel-offplane-verification-001/`. The JSON summary and exportable PNG/SVG
figure accompany this report. The figure colors errors below 1e-12 at that floor;
the underlying numerical values are retained unchanged.
