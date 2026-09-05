# Higher-derivative continuation: derive the force before selecting lengths

The first analytic audit passes 13 checks. It derives the spherical force from
the action in two ways and exposes a sign change that an algebraic screening
prescription would miss. No physical length has been chosen and no observations
have been scored for this successor.

The published GQUMOND construction allows an action to depend on derivatives of
the Newtonian potential and introduces length-sensitive response without using
an object label. It remains a nonrelativistic effective framework, not a unique
microscopic derivation. The framework and length-screening recipe are prior art.
[Milgrom (2023), equations 6, 7 and 16](https://arxiv.org/html/2305.01589v2)

## Proposed action and exact variation

Use the same saturated scalar Q_m and consider a separately registered action

    x = |grad psi|^2/a0^2
    h = ell^2 sum_ij(psi_ij^2)/a0^2
    P(x,h) = x + x K_m(x+h)
    K_m(u) = [Q_m(u)-u]/u.

This combines the published x/(x+h) recipe with the existing bounded scalar
kernel. It leaves all old action cards unchanged. The three shapes remain
m=1/2, 1 and 2; epsilon is a declared regularizer whose convergence must still
be tested. For epsilon>0 the quotient at u=0 has a removable singularity:

    K_m(0) = epsilon^(-1/2)
             * (1+epsilon^(2m))^(-1-3/(4m)).

At h=0, P equals Q_m exactly. The global constants would be a0, ell and the
registered shape, with no source-dependent retuning. Symmetry permits this
ansatz; it does not select it uniquely or derive these constants.

Varying the Newtonian auxiliary potential gives the physical Poisson equation

    laplacian Phi = div J
    J_i = P_x psi_i - ell^2 partial_j(P_h psi_ij).

The second term is required. P_x grad psi alone is not the force law, and a
position-dependent effective a0 inserted into an algebraic acceleration formula
does not implement this action. Newtonian psi still obeys laplacian psi=4 pi G rho.

For spherical psi, let g=psi'(r). Then

    h = ell^2 [g'^2 + 2(g/r)^2]/a0^2
    g_Phi = P_x g - ell^2 [(P_h g')' + 2 P_h(g'-g/r)/r].

Here zero anomalous central integration mass is imposed. A free 1/r^2 term
would change the central mass and is not silently fitted. Independent variation
of the reduced radial action r^2 a0^2 P/2 reproduces this tensor-divergence
formula identically; the numerical audit stores the symbolic check.

## Two informative point-source limits

For g=GM/r^2, the original unbounded, zero-epsilon extension has

    delta g = sqrt(GM a0) [102 ell^4 + 40 ell^2 r^2 + 3 r^4]
              / [3 sqrt(r) (6 ell^2+r^2)^(9/4)].

It reduces to sqrt(GM a0)/r at ell=0. When ell is much greater than r, it is
approximately [17*6^(3/4)/108] sqrt(GM a0/(ell r)). Screening changes its
radial behavior but does not automatically establish a precision local pass.
This resolves an analytic obligation of the old length-sensitive cards; their
dimensionful length and local observable evaluation remain unassigned.

For the bounded successor, Q_m(u)-u tends to a positive constant C at high u.
The resulting leading action excess is C x/(x+h), which varies in space.
Its point-source fractional force is

    delta g/g = -2 C ell^2 a0^2 r^6 (30 ell^2+r^2)
                / [(GM)^2 (6 ell^2+r^2)^3].

This leading term points outward for positive r, ell, GM, a0 and C. It does
not determine the sign of the complete finite-u solution. It also demonstrates
why the scalar high-acceleration tail cannot simply be inherited after adding
length dependence. A bounded action excess need not be a force-free constant.

## Next implementation and admission conditions

Implement the full finite-epsilon spherical and point-plus-constant-external-
field fluxes. Validate derivative normalization, ell=0 recovery, tensor versus
radial variation, manufactured nonspherical potentials, and surface terms.
Audit local precession and external quadrupole with one physical ell before
using the observed galaxy or cluster source. Numerical length is expressed in
the same units as coordinates; the Sun's MOND radius and the galaxy's kpc
coordinates cannot share an unconverted length parameter.

An axisymmetric implementation must include the azimuthal Hessian component:
H:H = H_RR^2 + 2 H_Rz^2 + H_zz^2 + (psi_R/R)^2. The existing 2-by-2 meridional
Hessian is insufficient for this invariant. Its tensor divergence also requires
the cylindrical connection terms. Smooth source and outer-boundary refinement
must precede any score; source-density derivatives can amplify reconstruction
artifacts. Total force, torque, arbitrary source-partition invariance, source
separation and regularizer convergence remain implementation obligations.

The λ=10 multifield cases remain a separate unresolved numerical task. Their
negative or unknown results do not force selection of this new action. The
Sigma stress/baroclinic route, direct outer-star data, nonspherical cluster
observables, derived photon coupling, dynamical health, and independent
cross-regime validation remain open. A static action alone supplies none of
those missing checks.

Reproduce the 13 analytic checks with
`scripts/audit_gravity_length_screening.py --output <unused-directory>`.
Exact inputs, symbolic expressions and the hash receipt are retained in
`work/gravity-first-principles/length-screening-analytic-001/`.
