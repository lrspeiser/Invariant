# Independent external-boundary analytic reference

All135 interface cases pass. Maximum potential mismatch1.56e-15, normal
epsilon-weighted flux mismatch4.45e-16 and tangential-field mismatch1.71e-15.
Independent finite-difference potential gradients agree within3.00e-9; the
largest sampled Laplacian residual is2.23e-8. Linearity in applied strength and
the uniform-medium limit pass. This validates the analytic reference, not the
parent's separate grid solver or an astronomical theory.

## Derivation

For constant epsilon in each region, regularity at the center and a uniform
applied field at infinity permit the dipole-sector ansatz

`Phi_in=-A E r cos(theta)`;
`Phi_out=-E r cos(theta)+B E a^3 cos(theta)/r^2`.

Continuity at r=a gives A+B=1. Normal flux continuity gives
epsilon_in*A=epsilon_out*(1+2B). Solving this two-equation system independently
returns A=3epsilon_out/(epsilon_in+2epsilon_out) and
B=(epsilon_in-epsilon_out)/(epsilon_in+2epsilon_out). For1 and0.2, A=3/7 andB=4/7.
These are the dielectric-sphere interface conditions, used here solely as an
equation benchmark. [University of Texas boundary-value derivation](https://farside.ph.utexas.edu/teaching/jk1/lectures/node42.html)

The acceleration is

`g_in = A E zhat`;
`g_out = E zhat+B E a^3[3z*r_vector/r^5-zhat/r^3]`.

This gives a uniform suppressed internal applied field and a dipolar distortion
outside. It does not give an extra spherically inward halo. Along the outer polar
axis the distortion is parallel to the applied field; on the equator it is
antiparallel. Normal g itself jumps at the interface; epsilon*g_normal, not
g_normal, is continuous when the equation has no interface source.

## The free-fall subtraction matters

To predict motion relative to a galaxy center or reference body, use
`delta_g_relative(x)=delta_g(x)-delta_g(x_reference)`.
Subtracting only the nominal far-field E is a different operation. For this
sphere every interior point and its center acquire the same A E acceleration.
After center-frame subtraction, the external perturbation is exactly zero
throughout the interior. Reporting the internal field deficit `(A-1)E` as an
extra internal attraction would be incorrect.

For E=1, at the external point(0,0,2a), g_z=8/7 and the center-relative change
is5/7. At(2a,0,0), g_z=13/14 and the center-relative change is1/2. Both examples
illustrate a direction-dependent response, not attraction toward the sphere.

These are kinematic reference-frame subtractions. A prescribed material epsilon
field plus an elliptic equation does not itself specify the center's dynamical
equation, matter backreaction, or a covariant equivalence-principle completion.
Using the center field as a proxy for its acceleration is an additional matter
coupling assumption that should remain explicit.

## What the actual density-dependent solver can establish

For a frozen density and epsilon(rho),
`L_epsilon Phi = div(epsilon gradPhi)` is a linear operator. An external-boundary
perturbation therefore solves `L_epsilon deltaPhi=0`, and adds to the existing
source solution. Nonuniform epsilon can distort that applied field. Scaling or
reversing the boundary perturbation must scale or reverse its response, apart
from numerical error. A free-fall-subtracted differential response may survive
in a nonspherical, spatially varying medium even though this spherical interior
control gives zero.

This is a useful physical discriminator: test a spatially varying boundary
response, not an arbitrary extra radial force. But it is not MOND's nonlinear
external-field effect, and no environment-dependent change in the internal
source Green function occurs while epsilon remains fixed. If density or epsilon
is allowed to evolve, that is a different coupled problem.

Density-dependent gravitational permittivity appears in published refracted
gravity work, but the present boundary test neither validates that proposal nor
imports a causal, relativistic or photon law. [Matsakos and Diaferio 2016](https://arxiv.org/abs/1603.04943)

No astronomical source accelerations or observed response values were calculated
in this branch. The parent NGC2976 calculation requires its own mesh/interface,
domain, constant-epsilon, superposition and center-subtraction checks.
