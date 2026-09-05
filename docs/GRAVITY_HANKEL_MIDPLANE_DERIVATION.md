# Independent Newtonian midplane derivatives

This reference covers a reflection-symmetric axisymmetric source
rho(R,z)=sum_j Sigma_j(R) f_j(z), with each f_j normalized. Units are consistent
throughout: kpc, solar masses and (km/s)^2 for the conditional galaxy audit.
It is not a new gravitational law or a modified-field solver.

The standard separated disk Green representation is given in
[Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
The following derivatives and specialization were worked out for this audit.
Define

```
S_j(k) = integral_0^infinity R' Sigma_j(R') J0(k R') dR'
Z_j(k,z) = integral_-infinity^infinity f_j(z') exp(-k abs(z-z')) dz'
psi(R,z) = -2 pi G integral_0^infinity J0(kR) sum_j S_j(k) Z_j(k,z) dk
```

At the midplane, Z_z=0 and the distributional derivative of the exponential
gives Z_zz=k^2 Z-2k f_j(0). This contact term is essential. Put
A(k)=sum_j S_j Z_j(k,0), B(k)=sum_j S_j [2 f_j(0)-k Z_j(k,0)]. Then, using
the physical orthonormal cylindrical tensor components:

```
g_R = psi_R =  2 pi G integral k J1(kR) A(k) dk
H_RR        =  2 pi G integral k^2 J1'(kR) A(k) dk
H_phiphi    =  g_R/R
H_zz        =  2 pi G integral k J0(kR) B(k) dk
H_Rz        =  0
d_R H_RR    =  2 pi G integral k^3 J1''(kR) A(k) dk
d_R H_phiphi=  (H_RR-H_phiphi)/R
d_R H_zz    = -2 pi G integral k^2 J1(kR) B(k) dk
J1'         =  (J0-J2)/2
J1''        =  (J3-3 J1)/4
d_R(H:H)    =  2 sum_i H_ii d_R H_ii
d_R lap psi =  sum_i d_R H_ii
```

At R=0, take the regular limits H_RR=H_phiphi, g_R=0 and all displayed radial
third derivatives zero. No finite differences of potential values are used.
At z=0 the source symmetry makes the vertical gradients of H:H and lap psi zero.

For f_j(z)=sech^2(z/h_j)/(2h_j), the dimensionless a=k h_j transform is

```
I(a) = integral_0^infinity sech^2(u) exp(-a u) du
     = 1 - (a/2) [digamma(1+a/4) - digamma(1/2+a/4)]
Z_j(k,0)=I(k h_j), f_j(0)=1/(2h_j)
I(0)=1
I(a) ~ (1 - 2/a^2 + 16/a^4 - 272/a^6 + 7936/a^8
          - 353792/a^10 + 22368256/a^12)/a
```

The digamma expression follows by integrating sech-squared by parts and summing
the alternating reciprocal series. For a>=32 the displayed asymptotic series
avoids cancellation. Adaptive direct integral controls span a=0 through 10^6,
including the switch. This is a numerical representation of a fixed source lift.

Every displayed jet is evaluated with the same finite-k quadrature. Consequently
its trace reconstructs the finite-k source. Replacing H_zz or d_R lap psi by a
physical-density identity while retaining other truncated derivatives would mix
different potentials and obscure the numerical error; this implementation does
not do that. Instead both identities are measured independently against the
positive physical density and its analytic first derivative.

The independent spherical-Gaussian control has radial transform
S=M exp(-a^2 k^2/2)/(2pi), vertical transform erfcx(a k/sqrt(2)), and a closed
enclosed mass. That mass gives the Cartesian Hessian and radial derivatives
without the disk derivation. It checks potential, force, all three diagonal
Hessian entries, all displayed third derivatives, the trace and H:H gradient.
An intentionally tiny cutoff must retain a large density mismatch, demonstrating
that source consistency has not been forced by substitution.

The registered empirical-source result meets its midplane criteria. Off-plane
derivatives, joint angular integration of the action flux, and the additional
Poisson solve producing a physical modified acceleration still require independent
validation. No source-posterior or dynamical/metric health claim follows from this
Newtonian reference.
