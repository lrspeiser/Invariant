# Screened nonlocal boundary completion attempt v0

## Claim boundary

This document constructs a testable descendant of the Item 59 phenomenology. It is not
a completed relativistic theory, an accepted action, a fit to new data, or evidence
against dark matter. The static weak-field action below is explicit and variational.
The relativistic scalar-vector completion is an architecture whose exact reduction,
constraint algebra, stability, and causal source map remain to be derived.

The goal is to connect four useful limits without inserting a galaxy/cluster label:

1. general relativity or Newtonian gravity in high-acceleration/local systems;
2. a MOND/AQUAL-like modified-Poisson limit when the auxiliary response becomes local;
3. a nonlocal/refracted-gravity limit when the auxiliary response has finite range;
4. a relativistic two-potential description in which dynamics and lensing are both
   derived rather than assumed equal.

## Why direct element-dependent gravity is not used

Giving photons, hydrogen, and heavy elements separate gravitational charges would
generically violate universal free fall, create composition-dependent accelerations,
and allow energy-momentum exchange that is not controlled by one physical metric.
Those effects are already tightly constrained and are not needed to make lensing differ
from massive-particle dynamics.

The viable distinction is geometric. In a weak relativistic metric there are two
potentials:

    d s_tilde squared
      = -(1 + 2 Phi/c squared) c squared dt squared
        + (1 - 2 Psi/c squared) d x squared.

Slow massive matter responds primarily to Phi. Light deflection responds to
(Phi + Psi)/2. Extra scalar, vector, or tensor fields can contribute differently to
Phi and Psi while every material species and every photon still follows the same
physical metric. This is the route used here.

## Step 1: response-blind dimensionless variables

Let U be the ordinary baryonic potential in the static weak-field predictor:

    Laplacian U = 4 pi G rho_b.

Define the following dimensionless quantities before opening a target response:

    y = magnitude(gradient U) / a0

    C = G M_b(<r) / (r c squared)

    b = magnitude[d squared ln M_b(<r) / d(ln r) squared]

    theta = P_b / (rho_b c squared)

y measures acceleration. C is radial baryonic compactness. b detects a change in the
cumulative baryonic profile and therefore acts as a boundary/profile-state measure.
theta is the source pressure-to-rest-energy ratio; it can distinguish hot cluster gas
from cold disk matter without referring to a test particle's chemical composition.

The radial definitions are appropriate to the current spherical cluster experiment.
An arbitrary-geometry completion must replace r and M_b(<r) with scalar field
invariants or geometrically defined averaging operators. That replacement is open.

## Step 2: a dimensionless two-factor transition

Construct an environmental state

    zeta
      = (C/C_star)^p
        (1 + b/b_star)^q
        (1 + theta/theta_star)^s.

Every ratio is dimensionless. C_star, b_star, theta_star, and all exponents must be
universal and frozen before response access.

Define

    S_high(y) = 1 / [1 + (y/y_screen)^n]

    S_env(zeta) = zeta^m / (1 + zeta^m)

    T(y,zeta) = S_high(y) S_env(zeta).

This creates two independent protections:

- if y is much greater than y_screen, T approaches zero and local GR is restored;
- if the baryonic environment gives zeta much less than one, T approaches zero even
  when acceleration is weak;
- only low-acceleration systems with sufficiently large compactness/boundary/thermal
  state activate the extra response.

This is a hypothesis, not a measured transition. Groups are the decisive intermediate
test because a hidden galaxy/cluster classifier will usually fail to vary smoothly
through the group regime.

## Step 3: local auxiliary-field action with an effective nonlocal limit

Introduce a dimensionless auxiliary response X and the Item 59 occupancy source

    Q(y) = y / (y + y0).

For a static baryonic source, consider the weak-field action

    S_WF = integral dt d cubed x {
      - epsilon(X,T) magnitude(gradient Phi)^2 / (8 pi G)
      - rho_b Phi
      - Lambda_X squared/2 [
          ell squared magnitude(gradient X)^2 + (X - Q)^2
        ]
    },

with

    epsilon(X,T) = exp[-2 alpha T X].

epsilon is positive for every finite field value, avoiding a sign flip in the
gravitational gradient term. Varying Phi gives

    divergence[epsilon(X,T) gradient Phi] = 4 pi G rho_b.       (1)

Varying X gives

    (1 - ell squared Laplacian) X
      = Q
        - epsilon_X magnitude(gradient Phi)^2
          / (8 pi G Lambda_X squared).                         (2)

In the stiff-response or weak-backreaction limit,

    X approximately equals (1 - ell squared Laplacian)^(-1) Q. (3)

Equation (3) is a finite-range spatial average: the inverse Helmholtz operator has a
Green function that integrates Q over neighboring locations. The fundamental action is
local in X, while eliminating X produces an effectively nonlocal gravity equation.

When epsilon varies slowly in spherical symmetry,

    g approximately equals g_bar / epsilon
      approximately equals g_bar [1 + 2 alpha T X + ...].      (4)

Equation (4) reproduces the multiplicative g_bar times kernel structure of Item 59 to
first order. It does not reproduce the separate a0 K_sym term. A second field or an
AQUAL-like nonlinear kinetic term could generate that channel, but adding it now would
hide whether the simpler conservative descendant is sufficient.

## Step 4: connections to GR, MOND, refracted gravity, and nonlocal gravity

### GR/Newton limit

If alpha = 0, T = 0, y is much greater than y_screen, or zeta is much less than one,
then epsilon approaches one and equation (1) becomes ordinary Poisson gravity.

### MOND/AQUAL-like limit

If ell approaches zero, X becomes a local function of Q. Equation (1) becomes

    divergence[mu(local baryonic state) gradient Phi] = 4 pi G rho_b,

which has the modified-Poisson structure used by AQUAL-like MOND theories. This does
not make the proposed epsilon the MOND interpolation function; it shows that MOND is a
contained structural limit against which equivalence must be tested.

### Refracted-gravity limit

If ell approaches zero but epsilon retains density, compactness, or boundary-state
dependence through T, equation (1) is a gravitational-permittivity equation. Spatial
gradients of epsilon redirect as well as amplify the field, matching the qualitative
mechanism of refracted gravity.

### Nonlocal-gravity limit

For finite ell, eliminating X inserts the Green-function average (3) into epsilon.
Gravity at one point then depends on a bounded neighborhood of the baryonic occupancy,
matching the qualitative mechanism of kernel-based nonlocal gravity.

The point of the construction is not to rename one of these known theories. It gives an
explicit bridge showing which limits are rewrites and where a materially distinct
transition or source coupling would have to appear.

## Step 5: dynamics and lensing from two potentials

Write the extra dynamical potential generated by equations (1)-(2) as phi_X:

    Phi = U + phi_X

    Psi = U + gamma_X phi_X.

Then

    extra response for slow matter = gradient phi_X

    extra response for light
      = p_gamma gradient phi_X,

    p_gamma = (1 + gamma_X)/2.

Three preregisterable branches span the minimal physically interpretable cases:

| Branch | gamma_X | p_gamma | Meaning |
|---|---:|---:|---|
| scalar/conformal control | -1 | 0 | extra massive-matter force, no extra lensing |
| transition-linked mixed mode | -1 + 2 T^r | T^r | photon response activates with the same state |
| metric/tensor control | 1 | 1 | extra dynamics and extra lensing are equal |

These are field-polarization hypotheses, not separate photon and heavy-element
gravitational charges. The direct-lensing experiment must freeze all three branches
before looking at arcs, shear, magnification, or time delays.

## Step 6: relativistic scalar-vector architecture

A TeVeS-inspired universal physical metric can realize a nontrivial relation between
Phi and Psi without composition-dependent matter coupling:

    g_tilde_mu_nu
      = exp[-2 A(X,T)] (g_mu_nu + u_mu u_nu)
        - exp[2 A(X,T)] u_mu u_nu,

    A(X,T) = alpha T X,

where u_mu is a unit timelike field. All matter and electromagnetism couple to
g_tilde_mu_nu.

A candidate covariant architecture is

    S_rel = integral d fourth x sqrt(-g) {
      M_Pl squared R/2
      - M_X squared/2 [
          ell squared k_X^mu_nu nabla_mu X nabla_nu X + (X - Q_b)^2
        ]
      - K F_mu_nu F^mu_nu/4
      + lambda_u (u_mu u^mu + 1)/2
    }
    + S_m[g_tilde, matter],

where

    h^mu_nu = g^mu_nu + u^mu u^nu

    k_X^mu_nu = h^mu_nu - c_X^(-2) u^mu u^nu

h projects onto spatial slices, k_X adds a finite-speed time derivative, and F_mu_nu
is the curl of u_mu. The static limit of k_X is the spatial operator used in equation
(2); a causal theory must select a retarded solution and keep 0 < c_X <= c on its
declared background domain.

This architecture is intentionally not registered as a completed action. Q_b still
needs a covariant, causal construction from baryonic stress-energy whose static limit
is Q(y). The preferred-frame/vector sector needs a full constraint and wave-speed
analysis. The static reduction to equations (1)-(2) has not been proved. Those are
formal blockers, not details to assume away.

## Required analytic checks before data

1. epsilon remains positive and finite on the declared field domain.
2. T approaches zero in the Solar System and other high-acceleration systems.
3. all transition inputs and thresholds are dimensionless and universal.
4. linear perturbations contain no ghost or negative-gradient mode.
5. characteristic speeds are causal on the declared background domain.
6. the Hamiltonian/constraint count is stable away from explicitly cataloged singular
   strata.
7. matter stress-energy is conserved with respect to the physical metric.
8. gravitational waves recover the observed high-frequency propagation limit.
9. the scalar/vector response does not create forbidden preferred-frame or fifth-force
   effects.
10. the high-acceleration post-Newtonian parameters reduce to the GR values.

## Frozen empirical program

### Gate A: transition without target labels

Generate candidate T functions from predictor-only baryonic profiles. Reject any
candidate that reconstructs the survey or object-class label better than it predicts a
continuous physical invariant. Freeze candidates and thresholds.

### Gate B: galaxy groups

Use a new group sample spanning disk to cluster compactness and thermal state. This is
the hardest test of whether T is physical rather than a disguised class switch.

### Gate C: unchanged X-COP replay

Check whether the screened descendant preserves the narrow pressure/temperature result
without refitting the four confirmation clusters. This is a development replay, not
new confirmation.

### Gate D: independent cluster observables

Use a non-X-COP X-ray/SZ release or an independently implemented reduction. Predict
pressure and temperature from gas and stellar baryons with covariance and nonthermal
pressure explicitly marginalized.

### Gate E: direct lensing

Freeze the three p_gamma branches and evaluate direct positions, parities, shear,
magnification, and time delays. Do not use an NFW-derived total-mass profile as the
target.

### Gate F: cross-scale and local gravity

Transfer unchanged to SPARC-like and LITTLE-THINGS-like galaxies, wide binaries, Solar
System/post-Newtonian limits, and cluster lensing. One empirical mismatch is retained;
broad replicated failures and hard analytic limit violations block only the tested
representation.

## What would count as a real advance

The next meaningful result is not a lower cluster fit error. It is one universal,
dimensionless transition and one shared field system that simultaneously:

- retains a forward-observable cluster advantage;
- improves materially over the RAR on independent galaxies or explains why it turns
  off there without using labels;
- predicts lensing with a frozen photon-response branch;
- restores GR analytically at high acceleration;
- passes action-level health checks.

Until then the construction is a disciplined mechanism generator, not a theory.
