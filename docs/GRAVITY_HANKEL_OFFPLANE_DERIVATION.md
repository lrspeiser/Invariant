# Off-plane derivatives of the isolated disk Green integral

This extends the midplane reference in `GRAVITY_HANKEL_MIDPLANE_DERIVATION.md`.
It is a Newtonian source/derivative provider, not a new action or a complete
modified-gravity solution. The separated disk potential is the standard Green
representation in [Bovy, section 7.3.4](https://galaxiesbook.org/chapters/II-01.-Gravitation-in-Galactic-Disks_3-Gravitational-potentials-from-disk-density-distributions.html).
The derivative formulas and stable implementation below were derived for this
audit.

## Vertical source and stable derivatives

For a component with physical height h, set u=z/h, a=k h. Let f(u) approximate
sech-squared(u)/2, with unit integral on the entire real line. On 0<=u<=U it is
a cubic spline with f'(0)=0 and the physical sech-squared slope at U. Beyond U,
f(U+s)=f(U) exp(-lambda s), lambda=-f'(U)/f(U)>0. Reflect f evenly about zero.
Normalize the entire source including both infinite tails. This defines a C1
source; density and its first derivative agree at the splice. Its approximation
to the physical lift is measured separately.

Define Z(a,u)=integral f(v) exp(-a abs(u-v)) dv. Physical derivatives satisfy
Z_z=h^-1 Z_u, Z_zz=h^-2 Z_uu, Z_zzz=h^-3 Z_uuu. The familiar identities

```
Z_uu  = a^2 Z - 2 a f(u)
Z_uuu = a^2 Z_u - 2 a f'(u)
```

can lose accuracy by cancellation at large a. Instead move derivatives onto the
source by integration by parts. For n=0,1,2, convolve its ordinary piecewise
n-th derivative directly with the exponential. For n=3, the weak derivative
also contains the second-derivative splice jump:

```
J = lambda^2 f(U) - f''(U-)
Z_uuu = integral f'''_regular(v) exp(-a abs(u-v)) dv
        + J [exp(-a abs(u-U)) - exp(-a abs(u+U))].
```

The signs at +/-U follow from even f''. At zero, f'(0)=0 and f'' is even and
continuous, so no additional weak point term is present. Z and Z_uu are even;
Z_u and Z_uuu are odd. This method differentiates the exact Green integral of
the declared approximate source. It does not insert the physical source into
selected components of an otherwise different numerical potential.

Each cubic segment uses exact moments

```
M_j(a,d) = integral_0^d t^j exp(-a t) dt
         = j! gammainc(j+1,a*d) / a^(j+1),  a>0
M_j(0,d) = d^(j+1)/(j+1),                  j=0,...,3.
```

Forward and reverse recurrences propagate the inner and outer convolutions.
Their decay factors are exp(-a*d)<=1. Arbitrary query heights use partial
moments in their containing segment, not an interpolation of potential values.
The exponential tail integral uses a difference quotient evaluated with expm1;
its a=lambda limit is d exp(-a*d). Only the necessary query values and moment
caches are stored. The implementation also supports queries beyond U.

## Cylindrical field derivatives

Let A_n(k,z)=sum_j S_j(k) partial_z^n Z_j(k,z), and C=2 pi G. Every integral
below uses the same finite k grid and weights. The physical orthonormal basis is
(R,z,phi). The tensor-product field grid supports the axis by analytic limits.

```
psi    = -C integral J0(kR) A_0 dk
p_R    =  C integral k J1(kR) A_0 dk
p_z    = -C integral J0(kR) A_1 dk
H_RR   =  C integral k^2 J1'(kR) A_0 dk
H_Rz   =  C integral k J1(kR) A_1 dk
H_zz   = -C integral J0(kR) A_2 dk
H_pp   =  p_R/R
T_RRR  =  C integral k^3 J1''(kR) A_0 dk
T_RRz  =  C integral k^2 J1'(kR) A_1 dk
T_Rzz  =  C integral k J1(kR) A_2 dk
T_zzz  = -C integral J0(kR) A_3 dk
T_Rpp  = (H_RR-H_pp)/R
T_zpp  = H_Rz/R
```

The six displayed T entries represent the symmetric Cartesian third derivative;
components with an odd number of azimuthal indices vanish in this basis. At R=0,
H_pp=H_RR, T_Rpp=0 and T_zpp=T_RRz. The last limit matters above the plane:
setting every azimuthal third derivative to zero at the axis would be incorrect.

```
H:H = H_RR^2 + 2 H_Rz^2 + H_zz^2 + H_pp^2
T:T = T_RRR^2 + 3 T_RRz^2 + 3 T_Rzz^2 + T_zzz^2
      + 3 T_Rpp^2 + 3 T_zpp^2
partial_R(H:H) = 2 [H_RR T_RRR + 2 H_Rz T_RRz + H_zz T_Rzz + H_pp T_Rpp]
partial_z(H:H) = 2 [H_RR T_RRz + 2 H_Rz T_Rzz + H_zz T_zzz + H_pp T_zpp]
partial_R lap psi = T_RRR + T_Rzz + T_Rpp
partial_z lap psi = T_RRz + T_zzz + T_zpp
```

These expressions sum all component potentials before taking tensor invariants.
The source split test verifies invariance under subdividing one identical source.
The distance test uses R,z,h -> D(R,z,h), mass -> D^2 mass, k -> k/D:
psi -> D psi, p -> p, H -> H/D, T -> T/D^2, and grad(H:H) -> grad(H:H)/D^3.
Any later physical length in an action must remain fixed under this source-distance
change; it is not a freely rescaled fitting parameter.

## What remains unproved

The off-plane audit establishes finite-grid source identities, refinement and
derivative consistency. It does not prove an error bound between all grid points,
accuracy at arbitrarily large radius, a production interpolant, a general
nonaxisymmetric source, or convergence of the additional Poisson problem driven
by the nonlinear action flux. These remain required before a wider physical
length scan can support the three-regime gravity objective.
