# Derivative-consistent galaxy field calculation

The length action requires a Hessian and gradients of its norm and trace.
Differentiating the existing C1 potential interpolation three times does not
provide those quantities for a controlled continuous source. The new implementation
instead evaluates the Green solution of one interpolated source and analytically
derives all required derivatives of that same solution.

## One radial source and its exact Green solution

Let `t=log(r)` and expand the Poisson source `4*pi*G*rho` in Legendre polynomials,
with coefficient `S_l(t)`. The plane-focused angular quadrature projects the
positive joint gas-plus-stellar source. Each coefficient is interpolated with a
C2 cubic spline in `t`. All powers of radius remain analytic.

Define

\[
A_l(t)=\int_{t_{\min}}^t S_l(u)e^{-(l+3)(t-u)}\,du,
\qquad
B_l(t)=\int_t^{t_{\max}}S_l(u)e^{-(l-2)(u-t)}\,du.
\]

Then the potential coefficient and its derivatives are

\[
F_l=-\frac{r^2(A_l+B_l)}{2l+1},\qquad
F_{l,t}=\frac{r^2[(l+1)A_l-lB_l]}{2l+1},
\]

\[
F_{l,tt}=r^2 S_l-F_{l,t}+l(l+1)F_l,
\]

\[
F_{l,ttt}=r^2(2S_l+S_{l,t})-F_{l,tt}+l(l+1)F_{l,t}.
\]

Partial integrals within each cubic segment use exact exponential moments.
Positive decay uses the incomplete gamma function; zero decay uses polynomial
moments; the two negative outer decay rates use the confluent hypergeometric
expression. Independent adaptive quadrature verifies all three cases.

The ODE expressions above are derivatives of these exact partial Green integrals.
They are not source substitutions into derivatives of a different potential
interpolant. The finite inner and outer shell bounds remain numerical approximations
to the isolated source; their impact must be checked by moving both boundaries.

An initial implementation interpolated `r^2*S_l` instead. Its third-derivative
test exposed a false central density gradient: differentiating its interpolation
error and dividing by radius amplified a small error near a constant-density
core. Interpolating `S_l` and retaining radius factors analytically removed this
artifact. This was an implementation control failure, not a gravity rejection.

## Full three-dimensional tensor variation

Set `mu=cos(theta)`, `s=sin(theta)`. For the reconstructed potential, write
`B=psi_t`, `C=psi_tt`, `E=psi_mu`, `F=psi_tmu`, and `H=psi_mumu`.
The orthonormal spherical Hessian components are

\[
H_{rr}=(C-B)/r^2,\quad H_{r\theta}=-s(F-E)/r^2,
\]

\[
H_{\theta\theta}=(B+s^2H-\mu E)/r^2,\quad
H_{\phi\phi}=(B-\mu E)/r^2.
\]

The azimuthal component is required even though the source is axisymmetric.
`H:H=Hrr^2+2*Hrt^2+Htt^2+Hpp^2` and `tr(H)=Hrr+Htt+Hpp` are scalars.
Their radial and angular derivatives use the exact third-order jets and
Legendre derivative recurrences. Basis-rotation terms cancel in those scalar
invariants. The angular formulas are written without division by `sin(theta)`,
so the symmetry axis is included.

The existing full Cartesian action-flux function receives these components in
the orthonormal basis. The isolated flux Green solver then solves its divergence.
No algebraic disk acceleration shortcut or density-gradient omission occurs.
The finite angular expansion can nevertheless ring: a mathematically consistent
potential is not proof of accurate pointwise reconstruction of the physical source.
Those errors are recorded independently of target-force refinement.

## Regular central source and fixed physical length

Below the first measured source radius `R0`, the new surface density is

\[
\Sigma(R)=\Sigma(R_0)\exp[c(R^2-R_0^2)],\qquad
c=\frac{\Sigma'(R_0)}{2R_0\Sigma(R_0)}.
\]

It is positive, has zero radial derivative on the axis, and joins the inherited
PCHIP value and derivative. Measured source knots, vertical sech-squared lifts
and the outer cosine taper are unchanged. The interior is an unmeasured source
assumption. Its mass change is checked separately before gravity predictions.

Under the inherited distance homology, `r -> d*r`, mass scales as `d^2`, and
the Newtonian acceleration at corresponding angles is unchanged. The Hessian
scales as `1/d`; its spatial derivative as `1/d^2`. Keeping physical `ell` fixed
therefore requires evaluating the new field at `ell/d` in nominal coordinates.
Circular speed is then `sqrt(d*r_nominal*g)`. Multiplying a nominal nonzero-length
velocity curve by `sqrt(d)` alone would silently change the physical theory.

## Controls and remaining limits

Eleven new tests cover independent Green quadrature, radial-jet continuity,
small-core stability, third derivatives of manufactured Cartesian potentials
(including odd multipoles and both symmetry axes), the spherical action response,
source partition, dimensional homology, the regular core join, and explicit
invalid-input errors. The combined focused suite has 201 passing tests.

The fixed galaxy campaign keeps every local/cluster card and the same source,
distance and inclination alternatives. Refinement must cover all registered
target radii and distances. Numerical failures withhold the entire card's
velocity scores; nonpositive circular branches are separately retained.
Force refinement is only a numerical diagnostic. Source angular ringing,
shared gas-source/velocity uncertainties, warp and noncircular motion, missing
outer stellar data, and the isolated boundary prevent full empirical validation.
