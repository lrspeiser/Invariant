# Finite positive sources retain the short-wavelength response

Used a Plummer density multiplied by 1+c*cos(k*r), with contrasts c=0.01 and 0.1. Density is positive everywhere and total mass lies between (1-c)M and (1+c)M. Unlike the preceding planar test, the source has finite mass and no uniform external field.

Integrated its enclosed mass, evaluated analytic density derivatives, and used the full spherical action variation. Regularity at the origin fixes the otherwise possible extra C/r² force term. The perturbation transfer compares the physical-force change with the Newtonian-force change across a radial window; it is not an algebraic replacement for the action.

All 36 completed cases have the expected sign: 12 longer-wavelength cases positive, 24 shorter-wavelength cases negative. This holds for all three action shapes, two background sizes and both contrasts. The minimum sampled total inward force is 1.108 in the calculation's units: negative perturbation transfer does not imply that the entire source repels matter.

Doubling radial samples from 256 to 512 changes transfer coefficients by at most 0.026% under the stated normalization. This supports the sampled sign result, not a high-precision universal coefficient or pointwise error bound. Weighted-quadrature error estimates for enclosed mass are retained and are not rigorous bounds.

The first attempted background-size set produced a probe interval crossing the origin for one shape. That invalid run is preserved as `finite-positive-response-001`. The completed successor uses larger backgrounds for all shapes and retains the same contrasts and relative wave numbers; no failing gravity case was removed.

These results establish that the sign-changing response can occur in finite, positive matter distributions. They do not establish a dynamical instability, observational inconsistency, or a physical exclusion. The next theoretical question is whether the resulting scale-dependent matter response has acceptable dynamics and a justified domain of validity.

Evidence: `finite-positive-response-002`, with source definition, frozen modules, all 36 coefficients at both resolutions, total-force minima and numerical integration diagnostics. No observational data were fitted or scored.
