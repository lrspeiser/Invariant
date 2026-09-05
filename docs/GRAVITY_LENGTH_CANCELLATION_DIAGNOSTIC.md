# A numerical repair for subtracting very small effects

A synthetic high-precision check identifies a promising way to compute the
small length-dependent flux change without subtracting two nearly equal large
fluxes. The production action and running galaxy calculations remain unchanged.

For the existing action, the relevant coefficient is P_x - 1 = K(x+h) + x K'(x+h).
At fixed x, its difference from zero length is exactly

    delta P_x = integral from 0 to h of [K'(x+t) + x K''(x+t)] dt.

The prototype evaluates that integral directly and includes the unchanged full
second-gradient reaction term. This rearranges the same formula; it adds no
physical parameter or new force. A later Poisson solve could operate on the
flux difference itself, avoiding a second subtraction of large final fields.

The test uses a smooth polynomial potential at two positions, all three action
shapes and seven dimensionless lengths, for 42 cases. Independent 80-digit
evaluation differentiates the closed-form kernel. Both 16- and 32-point
quadratures are retained.

- Worst prototype relative vector error: 9.10e-15.
- Worst direct-subtraction relative vector error: 1.414 (about 141 percent).
- All prototype cases pass the pre-execution 1e-9 relative target.

The test covers nonzero gradients at these synthetic points, with a0=1. It does
not establish behavior at a saddle, the origin, arbitrary h/x, or the full
galaxy source. It also does not repair angular-resolution error or prove an
observationally measurable effect. Those checks remain necessary before
adopting the prototype in a production field calculation.

Evidence: `length-cancellation-001/result.json`, containing all 42 comparisons,
both quadrature orders, and frozen implementation/source inputs. No physical
exclusion or new observational score follows from this diagnostic.
