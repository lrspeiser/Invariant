# Exterior source representation and a C3 potential join

This supplies a Newtonian source field for the length-dependent action. It is
not another gravity law or an observational result. The source maps, radial
interpolation, central continuation, taper and two vertical thickness choices
are inherited from the preceding source audits.

## Compact exterior source

For each positive separable disk, restrict the physical vertical lift to
`|z| <= U h`, without renormalizing it. The missing mass fraction is exactly
`2 exp(-2U)/(1+exp(-2U))`. This defines a bounded numerical source, whose support
radius is rounded outward. A small missing mass fraction alone does not bound
the error in derivatives of the physical field.

Let `M_l = integral rho(s) |s|^l P_l(cos theta_s) d^3s`. Axisymmetry gives

`psi(r,mu) = -G sum_l M_l P_l(mu)/r^(l+1)`.

The positive source inside radius s obeys `|M_l| <= M s^l`. Reflection removes
all odd modes. With a common scale b, the stored even moments are computed from
the solid harmonic identity

`M_l/b^l = 2 pi sum_j (-1)^j binom(l,2j) binom(2j,j)/4^j A_j B_(l/2-j)`,

where `A_j = integral R Sigma(R) (R/b)^(2j) dR` and
`B_n = integral f(z) (z/b)^(2n) dz`. Each radial source interval is integrated
separately. Vertical integration uses positive quadrature weights. The saved
integrals are also recombined independently at 80-digit precision; that checks
floating-point summation, not the accuracy of the underlying quadrature.

## Uniform ideal series bounds

Repeated Jacobi differentiation and its endpoint inequality give
`max_|mu|<=1 |P_l^(j)(mu)| <= (l+j)!/[2^j j! (l-j)!]`, with a zero derivative
when j>l. These are consequences of the [DLMF derivative identity](https://dlmf.nist.gov/18.9.E15)
and [Jacobi bound](https://dlmf.nist.gov/18.14.E1), specialized to Legendre
polynomials. In particular the first three derivative bounds are
`l(l+1)/2`, `(l-1)l(l+1)(l+2)/8`, and
`(l-2)(l-1)l(l+1)(l+2)(l+3)/48`.

For one mode put `t=l+3`. Taking absolute values of the complete orthonormal
spherical tensor components gives the following bounds in units
`G |M_l|/r^(l+1+n)`:

| Derivative order n | Component sum bound | Conservative bound |
| --- | --- | --- |
| 0 | 1 | 1 |
| 1 | t+t^2/2 | t^2 |
| 2 | t^4/8+t^3+2t^2+2t | t^4 |
| 3 | t^6/48+3t^5/8+9t^4/4+7t^3+6t^2 | t^6 |

Tensor component sums include multiplicities: two off-diagonal Hessian entries,
and third-tensor multiplicities `1,3,3,1,3,3` for
`rrr,rrtheta,rthetatheta,thetathetatheta,rphiphi,thetaphihi`. The Frobenius norm
is bounded by these absolute component sums and is invariant under rotation.
Each bound in the last column holds for t>=3: divide by the proposed power,
then bound the remaining decreasing terms by their values at t=3.

Thus the omitted modes beyond L have monopole-scaled derivative bound
`sum_(l>L) (l+3)^(2n) (s/r)^l`, n=0..3. Including odd modes is conservative.
The implementation sums this infinite polynomial-geometric series as an exact
rational expression for an outward-rounded floating radius ratio, and rounds
the answer upward. This bounds ideal series truncation only. Source truncation,
quadrature, finite-precision moments and field evaluation remain separate checks.
The first registered exterior admission radius was 80 kpc. After the first
join failed, a separate order-128 experiment registered admission from 60 kpc
before computing its moments and direct comparisons, keeping the same tolerances.

## Independent spatial comparison

The second method sums the Cartesian potential kernel `-G/d` and its first
three analytic derivatives over positive source quadrature weights. Radial
intervals use Gauss quadrature, azimuth a periodic trapezoidal rule, and
`v=2|z|/h` uses Gauss-Laguerre quadrature on both infinite vertical halves. Three
one-factor refinements test radial, vertical and angular resolution. No Hankel
or multipole kernel is used in this comparison.

It is a discrete-source quadrature comparison, not a uniform theorem about
every physical tail. On the axis the physical untruncated density is nonzero,
although extraordinarily small at the registered exterior probes; derivatives
of a continuum Green integral there require their distributional prescription.
The quadrature comparison does not independently prove a bound for that
continuum contribution. We retain this distinction from the rigorous compact
source series bound.

## Joining potentials

For a fixed 80--120 kpc transition let `u=(r-80)/40` and
`w=35u^4-84u^5+70u^6-20u^7`, with w=0 below and w=1 above the interval.
Its first three derivatives vanish at both endpoints. For near and far
potentials with the same zero at infinity, define

`psi = psi_near + w (psi_far-psi_near)`.

All derivatives are of that single potential. Writing `D=psi_far-psi_near`,

`psi_i = psi_near,i + w D_i + w_i D`,

`psi_ij = psi_near,ij + w D_ij + w_i D_j + w_j D_i + w_ij D`,

`psi_ijk = psi_near,ijk + w D_ijk + w_i D_jk + w_j D_ik + w_k D_ij
           + w_ij D_k + w_ik D_j + w_jk D_i + w_ijk D`.

For a radial w, `u_i=x_i/r`, its Cartesian derivatives are

`w_i = w' u_i`,

`w_ij = (w''-w'/r) u_i u_j + (w'/r) delta_ij`,

`w_ijk = (w'''-3w''/r+3w'/r^2) u_i u_j u_k
          + (w''/r-w'/r^2)(delta_ij u_k+delta_ik u_j+delta_jk u_i)`.

The Hessian trace and its gradient are calculated by contraction, never replaced
with the known source. Those identities test any artificial density introduced
by the join. A test with a constant potential mismatch explicitly verifies that
the spurious force and density terms remain present; independently blending
forces would conceal this error. Symbolic differentiation of a nonspherical
potential checks all Cartesian components through third order, including the
axis and both signs of height.

The finite grid audit cannot certify a production interpolation or the full
modified action solve. Those remain subsequent requirements, regardless of
whether this join meets its numerical targets.
