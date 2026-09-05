# A matched Newtonian source and its omitted short-scale potential

This is numerical source work for the existing length-dependent action. It
introduces no gravity parameter and changes no physical mass model. The
inherited radial interpolation, regular central continuation, 34--36 kpc
cosine taper, physical sech-squared vertical source and two thickness choices
remain fixed.

## One matched scalar potential

The new matched provider uses the near potential below 60 kpc, the independently
checked order-128 exterior potential above 80 kpc, and the existing C3 septic
potential join between them. All Cartesian derivatives include the full
product rule. Beyond 80 kpc only the exterior provider enters the assembled
field; an unused near-provider error is retained as a diagnostic rather than
misidentified as an assembled-field error.

The reference audit retains the previous coordinates and adds points across
the taper, transition and thin vertical layers: 115 radial by 41 vertical
coordinates, or 4,715 locations for each thickness. It checks the actual
assembled potential and derivatives at every location.

## Leading high-wavenumber contribution

Write one separable source as rho(R,z)=Sigma(R) f(z), with integral f dz=1,
and its surface Hankel transform as S(k)=integral R Sigma(R) J0(kR) dR.
The isolated Newtonian potential is

    psi = -2 pi G integral_0^infinity S(k) J0(kR) Z(k,z) dk,
    Z(k,z) = integral f(z') exp(-k |z-z'|) dz'.

Expanding the smooth vertical source about z gives

    Z(k,z) = 2 f(z)/k + 2 f''(z)/k^3 + ... .

Consequently the leading potential omitted at cutoff K is

    delta psi = -4 pi G f(z) A_K(R),
    A_K(R) = integral_K^infinity S(k) J0(kR)/k dk.

This is an asymptotic completion, not an assertion that all further terms
vanish. The Laplacian and its gradient are still computed from the resulting
potential and compared with the physical source. They are never replaced by
the known density or its gradient.

## Logarithmic Green representation

Let M=integral s Sigma(s) ds, m(R)=integral_0^R s Sigma(s) ds, and

    L(R) = log(R) m(R) + integral_R^infinity s Sigma(s) log(s) ds.

The radial two-dimensional logarithmic Green function gives the convergent
subtracted representation

    A_K = [log(2/K)-gamma] M - L(R)
          - integral_0^K [S(k) J0(kR)-M]/k dk.

Every logarithm uses the same reference length (kpc in the implementation);
the reference cancels in this expression. At the axis, L(0) is the finite
radial logarithmic source integral. Outside the radial source, L=M log(R).
Differentiation, including L'=m/R, gives

    A' = -m/R + integral_0^K S(k) J1(kR) dk,
    B = Sigma - integral_0^K k S(k) J0(kR) dk,
    B' = Sigma' + integral_0^K k^2 S(k) J1(kR) dk,
    A'' = -B - A'/R,
    A''' = -B' - (A''-A'/R)/R.

The appearances of Sigma in these formulas are derivatives of the logarithmic
potential, not an override of the three-dimensional Poisson check. Near the
axis an even series supplies A'/R and its derivative, avoiding division of
nearly cancelled quantities. Radial quadrature is split at every source
interval and taper edge. The central quadratic-log density admits analytic
convergent series for both mass and logarithmic mass integrals, avoiding a
logarithmic quadrature endpoint.

For f=sech^2(z/h)/(2h), with t=tanh(z/h),

    f'=-2 t f/h,
    f''=(6 t^2-2) f/h^2,
    f'''=(16 t-24 t^3) f/h^3.

All radial and vertical products of delta psi are evaluated through third
order. The C1 radial source produces a C3 logarithmic potential; the physical
vertical source is smooth. Higher regularity at radial interfaces is not
assumed. Synthetic Gaussian controls compare with an independent integral of
the omitted tail, including its exponential-integral axis value, and
differentiate the correction potential independently.

## Accuracy and current admission limit

The completed source passes the registered grid identities and six separate
refinement comparisons, but the independent derivative verification fails.
The active correction is (1-w) delta psi, where w is the exact exterior join
weight. Fourth-order differences check potential to gradient, gradient to
Hessian, and Hessian to third tensor at all 4,715 original locations for both
thicknesses. At radial source interfaces both one-sided stencils are checked;
at the axis the verifier uses Cartesian parity. Two step sizes are retained.

The failed radial third-derivative comparison worsens when the step is halved.
On this Windows runtime, numpy.longdouble has the same 52 stored mantissa bits
as float64. The subtracted expression for A_K therefore loses precision even
when numpy.longdouble is requested. Good density identities and refinement
agreement did not establish a sufficiently differentiable numerical potential.

A separate diagnostic at the exposed R=66.5 kpc failure point embeds the
unchanged stored quadrature inputs in 50-digit arithmetic, evaluates low-k
Bessel functions accurately, and uses compensated summation for the remainder.
For R outside the source it evaluates the algebraically equivalent form

    A_K = M [log(2/K)-gamma-log(R)+sum_j w_j/k_j]
          - sum_j w_j S_j J0(k_j R)/k_j.

Accurate Bessel values below k=8 kpc^-1 reduce the derivative discrepancies
by several orders of magnitude at that point. This isolates a cancellation
problem; it neither repairs the production provider nor establishes accuracy
between sampled points. Source quadrature and omitted higher asymptotic terms
remain distinct error sources. A generally stable implementation must still
pass the unchanged full-grid source, refinement and derivative checks.

The full nonlinear action flux, separate Poisson solve, production
interpolation and new astronomical predictions remain pending.
