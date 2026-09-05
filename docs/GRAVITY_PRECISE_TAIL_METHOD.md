# Stable evaluation of the same omitted potential

This numerical successor addresses the failed independent derivative check in
`source-tail-verification-001`. It changes neither the physical source model
nor the gravity law. The overlapping-range and elastic audit remains a
separate research direction.

The earlier expression for the radial omitted potential subtracts quantities
of order the source mass to recover a very small residual. Windows
numpy.longdouble has 52 stored mantissa bits on this runtime. A diagnostic at
R=66.5 kpc showed that cancellation, rather than an actual nonconservative
force, was a major source of disagreement between a finite derivative of the
potential and its analytic derivative.

## Arithmetic and integration changes

For M=integral s Sigma(s) ds and the inherited logarithmic source integral L(R),
evaluate the same expression as

    A_K(R) = M [log(2/K)-gamma+sum_j w_j/k_j]
             -L(R)-sum_j w_j S_j J0(k_j R)/k_j.

Its first derivative is evaluated consistently as

    A_K'(R) = -m(R)/R+sum_j w_j S_j J1(k_j R).

The cancellation arithmetic, low-wavenumber Bessel values and their sums use
cloned mpmath contexts. The canonical setting is 50 decimal digits and an
accurate Bessel band below 8 kpc^-1. Remaining high-wavenumber terms use
compensated summation. The original finite-k surface transforms are retained
exactly. An additional 35-digit case and a band extended to 16 kpc^-1 test
these arithmetic choices separately.

The radial source uses the same stored PCHIP coefficients, central coefficient,
and physical cosine taper. The polynomial and taper are evaluated at higher
precision, as are partial mass and logarithmic mass integrals. The analytic
central series avoids a logarithmic integration endpoint. Neither density nor
its gradient is substituted into the resulting three-dimensional field.

An initial unit control found that merely embedding double-precision Gauss
nodes and weights into high-precision arithmetic retained a mass error of
0.00003236 on a source with mass about 3.15e10. The absolute target was
0.00001. That failure and its execution source are retained in
`precise-tail-preflight-001`. The same Legendre rule is now refined by Newton
iteration in the selected precision, including its weights; the quadrature
order and test threshold are unchanged. Both new independent unit controls
then passed alongside the original two tail controls.

If the accurately evaluated first derivative changes by delta p, the existing
radial Poisson identities require

    delta(A'/R) = delta p/R,
    delta A'' = -delta p/R,
    delta A''' = 2 delta p/R^2,
    delta[d(A'/R)/dR] = -2 delta p/R^2.

These changes are propagated together. The existing regular even series
continues to supply the axis limits. The analytic vertical factors and full
potential-join product rules are unchanged.

## Verification scope

`gravity_source_tail_audit_v2.json` registers nine configurations per thickness:
the reference and eight separate variations. Every one of the original 4,715
locations per thickness, both source identities, and the 60--80 kpc potential
overlap remain required. The source-grid tolerances are unchanged.

The independent verifier differentiates the newly added active correction,
normalized by the full-field scales. Inherited Hankel and exterior derivatives
have separate earlier controls. It retains both step sizes, every original point and
both one-sided stencils at radial interfaces. To avoid unnecessary expensive
Bessel evaluations it forms the union of actually required stencil coordinates;
third and fourth offsets away from interfaces are unused by the declared
central stencil and need not be calculated. No comparison or test point is
removed. Its separate fine-step derivative target remains 0.0001.

Passing a source-grid audit would not automatically pass this derivative
verification. Neither numerical check supplies a production interpolant, the
full nonlinear action solution, a photon law, or new astronomical validation.
Full run outcomes must be read from their terminal result and receipt files.
