# Multi-field external-boundary derivation

These notes preceded `external_multifield.py`. Its analytic controls and frozen
216-row conditional quadrupole scan are now complete; see
`GRAVITY_MULTIFIELD_EXTERNAL_RESULTS.md`. The derivation and boundary cautions
remain relevant. The observed NGC3198 transfer is now recorded in
`GRAVITY_NGC3198_MULTIFIELD_RESULTS.md`. Relativistic completion and observed
nonspherical cluster transfer remain outstanding.

Use a0=GM=1 and p=grad(psi)=rhat/r^2-eta_N*zhat for the idealized point mass
in a constant Newtonian background. Define

    x=p.p; q=grad(chi); y=q.q; z=2*p.q
    s=lambda/(1+x)^P; w=beta/(1+x)^2
    F=Q(x)-(y-s*z+s*s*x)-w*(x*y-z*z/4)
    A=I+w*(x*I-p*p^T).

The auxiliary equation is div(A*q)=div(s*p). A*p=p and the perpendicular
eigenvalues are 1+w*x. Since w*x<=beta/4, the existing beta<=2 grammar permits
a useful contraction bound for a Poisson-preconditioned fixed-point approach;
actual discretization and boundary convergence still need testing.

A conditional collinear Galactic background uses q_inf=s_ext*p_inf. Set
q=q_inf+grad(u), giving

    lap(u)=div[J], J=s*p-q_inf-(A-I)*(q_inf+grad(u)).

At fixed p/background, chi is linear in lambda and the physical auxiliary
correction is quadratic. With D=x*y-z*z/4, its extra flux is

    DeltaJ=(F_x-Q')*p+F_z*q
    F_x-Q'=s'*(z-2*s*x)-s*s+2*w*D/(1+x)-w*y
    F_z=s+w*z/2.

It vanishes when q=s*p exactly, including the spherical collinear solution.
The auxiliary sector cannot therefore repair a spherical cluster-pressure
prediction merely by changing lambda or beta.

For the dimensionless axisymmetric quadrupole convention used by the current
scalar module, a full-space flux representation is

    Q2=(9/2)*integral dr/r^2 integral dmu
          [J_r*P2(mu)+J_theta*mu*sqrt(1-mu^2)].

Integration over a finite radial shell adds boundary terms unless their
vanishing is demonstrated. Independently verify sign, normalization, units
and surface terms against the scalar exact-integral implementation.

One proposed flux Green solver avoids taking numerical divergence first.
In log radius t, define

    J_l=(2*l+1)/2*integral J_r*P_l dmu
    K_l=(2*l+1)/2*integral J_theta*sqrt(1-mu^2)*P_l' dmu
    S_l=(2*J_l+d_t J_l+K_l)/r.

After integration by parts with appropriate boundary terms, decaying inner
and outer integrals have integrands r*(l*J_l-K_l) and
r*((l+1)*J_l+K_l), respectively. The proposed potential coefficients are
(I-O)/(2*l+1), with logarithmic derivative
r*J_l-[(l+1)*I+l*O]/(2*l+1). Independent manufactured-source checks now pass
in the accompanying implementation. Zero-extending flux outside a
finite shell represents surface sources; this is not automatically the desired
infinite-domain solution.

Completed first checks: manufactured potentials, beta=0 analytic auxiliary
source, lambda sign/scaling and zero-coupling limits, constant-background and
boundary refinement, independent flux/source quadrupole integrals, and known
scalar normalization. The historical Cassini summary has now been evaluated
conditionally; no raw measurements or independent confirmation were opened.
See the action framework in
[Milgrom, Tripotential MOND](https://arxiv.org/abs/2305.19986).
