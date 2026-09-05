# Matched source and omitted potential tail: retained results

The leading omitted short-scale potential fixes the registered source-grid
errors, but its current floating-point evaluation fails independent derivative
verification. It remains provisional and is not admitted for new gravity-law
scores. A focused precision diagnostic identifies a practical repair direction.

No physical source, gravity constants, observational responses or reserved
data changed. These are numerical results, not new astronomical validation.

## Source-grid comparison

The new 60--80 kpc join retains all prior coordinates and adds dense sampling
across the disk taper and transition. Each source thickness has 115 radial by
41 vertical coordinates: 4,715 locations. The finite-cutoff audit compares five
one-factor refinements. Tail completion repeats those comparisons and adds a
log-source radial-quadrature refinement.

| Run | Thickness | Largest third-tensor refinement | Density identity error | Density-gradient identity error | All registered source-grid targets met |
| --- | --- | --- | --- | --- | --- |
| matched-source-001 | primary | 0.00717144 | 5.24698e-07 | 0.168944 | False |
| matched-source-001 | height_half | 0.014359 | 7.8345e-07 | 0.315079 | False |
| source-tail-002 | primary | 5.62729e-06 | 9.40783e-08 | 6.03718e-05 | True |
| source-tail-002 | height_half | 8.83251e-05 | 7.52049e-07 | 0.000946244 | True |

The fixed source-grid tolerances are 0.0001 for force, 0.002 for Hessian and
density, 0.01 for third tensor and density gradient, and 0.000001 for the
near/exterior potential difference in GM/r units. Tensor discrepancies use
the inherited full-field scales; density errors use the physical source with
Hessian-based floors. These are not uniform fractional errors in density,
which vanishes at many locations.

The uncompleted source's worst density-gradient errors occur at the fixed
cosine taper endpoint R=36 kpc, z=0. The radial density is C1 there, so a
finite-wavenumber integral converges slowly in its higher spatial derivatives.
The added term is derived from the omitted potential integral. The density
trace and its gradient are still computed from that potential, never imposed.
The exact physical source and all numerical targets are unchanged.

The first tail run ended with a JSON serialization error for numpy.longdouble.
Its failure record and partially written field file are retained. The second
run changes only serialization and completes. The partial file is not a valid
completed scientific result.

## Independent derivative failure

The snapshot-based verifier differentiates the active correction (1-w) delta
psi at every original location, including the axis, radial source interfaces,
both disk sides and the exterior join. It uses 0.001 and 0.0005 kpc steps;
both one-sided stencils are tested at source interfaces. The fine-step target
was fixed at 0.0001 on the full-field force/Hessian/third-tensor scales.

| Thickness | Direction / stencil | Gradient error | Hessian error | Third-tensor error |
| --- | --- | --- | --- | --- |
| primary | R / central | 5.0125e-08 | 3.61502e-06 | 0.000743872 |
| primary | R / right | 3.70556e-08 | 9.01925e-07 | 8.15878e-06 |
| primary | R / left | 2.53984e-08 | 5.51655e-07 | 4.93407e-06 |
| primary | z / central | 1.72874e-14 | 1.60979e-17 | 5.7324e-15 |
| height_half | R / central | 1.00244e-07 | 1.44566e-05 | 0.0059498 |
| height_half | R / right | 7.38601e-08 | 3.33879e-06 | 4.64356e-05 |
| height_half | R / left | 5.06247e-08 | 2.04214e-06 | 2.80433e-05 |
| height_half | z / central | 3.45749e-14 | 2.06014e-15 | 1.44423e-12 |

The largest fine-step discrepancy occurs in a radial derivative at R=66.5
kpc, z=0 and exceeds the registered target. The nominal-source maximum rises
from 0.000343685 to 0.000743872 as the step is halved; the half-height maximum
rises from 0.00274889 to 0.00594980. The source-grid pass therefore does not
qualify the provider for production. All failures remain in the evidence tree.

## Focused precision diagnostic

The logarithmic expression subtracts large terms to obtain a tiny omitted
potential. This Windows runtime provides 52 stored mantissa bits for both
float64 and numpy.longdouble. The latter does not add arithmetic precision.
At the already exposed R=66.5 kpc point, the diagnostic reuses the same source,
mass and transform inputs. It evaluates cancellation arithmetic and low-k
Bessel functions with 50 digits, with a compensated sum for higher k.

| Radial component | Ordinary finite derivative minus analytic derivative | Higher-precision difference, accurate Bessel below k=8 |
| --- | --- | --- |
| stellar_fixed | 0.00509262 | 1.35859e-09 |
| hi | 0.010175 | -3.36395e-09 |
| co | 0.000121685 | 2.71511e-10 |

These are absolute radial A_K derivative differences, not the normalized
three-dimensional tensor errors in the preceding table. This single-point
diagnostic identifies cancellation as a major contributor. It is not a new
full-grid verification or a production repair.

## Next work and scientific limits

Implement a stable evaluation of the same scalar tail potential and preserve
its derivatives consistently. Repeat every source, refinement and derivative
gate without loosening targets or changing the source. Then validate a fast
representation derived from one C3 potential, the full nonlinear action flux
and a separate Poisson solve. Only after those checks can the same global
gravity constants be tested again against galaxies, clusters and local data.

The current cluster/local tension, conditional galaxy comparisons, incomplete
light coupling and untested stability remain unchanged. Environment-dependent
response and matter-current mechanisms remain research directions, not
validated explanations. The discovery goal remains active.

## Retained result hashes

- `matched-source-001`: `ee10f07adc9bff3950fce090a114ebf3d7c490caf2965040ae78593273b1c2c8`
- `source-tail-002`: `804c0a38ff29bc8e8ccfe42e8ab385e7133e6878260e527c9b14d1d78bc2461e`
- `source-tail-verification-001`: `f63f2fb795d4b9f12a13e947331ad071ff9c3b9294353d11526474e8b936d49f`
- `tail-precision-diagnostic-001`: `d334a77cf33c3beaf55036e321421c1783036b28ca3ebb664bc7dc09904d62c8`
- `source-tail-001/failure.json`: `1f33a7ce234746c394544222f0acf76c74d5f6ebc7eca6cefdf45497eb9458bf`
- `source-tail-001/partial-fields`: `914949d46f847cfeb2c1e18635e3943849d6793f3e8846408c94bfc593a8b904`
