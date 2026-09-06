# Partition repair was implemented and retested

Status: THEORY_BENCHMARK_ONLY. This version changes the faulty reduced-mass
coefficient, preserves the previous results, and reruns the dynamics. No
observational target was fitted and no parameter was retuned after these results.

The corrected extra kinetic term is

`T_extra = (epsilon/2) sum_(i<j) [m_i*m_j/Mref] exp(-r_ij^2/(2L^2)) |v_i-v_j|^2`,

with epsilon=0.5, Mref=1 and L=1 in the original dimensionless units. Mref is
fixed externally; it is not changed when particles or cells are subdivided.
This produces the same positive graph-Laplacian contribution to inertia, now
bilinear in the source masses.

## The targeted defect is fixed

Before integration we subdivided every particle in an asymmetric four-particle
system into coincident co-moving children, first equal halves, then0.3/0.7 parts.

- Extra kinetic energy is unchanged: split/original ratio1.000000.
- Aggregate canonical momentum and individual child accelerations agree within
  2.23e-16 absolute.
- The finite-difference Euler-Lagrange residual is6.80e-11.
- Positive inertia, energy derivative, rotation and uniform-boost controls pass.
- The original reduced-mass law remains as a negative control: equal subdivision
  doubles its extra kinetic energy and changes aggregate acceleration by0.134 in
  the manufactured example.

Coincident split particles introduce a constant softened Newtonian self-potential
inside each parent. That arbitrary reference is explicitly subtracted when
comparing total energy; the corrected energy agreement is4.45e-15. Its self-force
is zero. No self-force or self-energy mismatch was hidden in the kinetic claim.

## What the new orbit integrations found

Twenty integrations repeated Newton and the corrected law at both resolutions
for nested co-rotation, nested counter-rotation, clustered sources and flyby,
plus the short clustered diagnostic. Orbit duration40, flyby12, short diagnostic2;
same initial positions and velocities, softening and numerical tolerances.

| Configuration | Newton maximum central distance | Corrected maximum central distance |
|---|---:|---:|
| Nested co-rotation, T40 | 2.008 | 178.549 |
| Nested counter-rotation, T40 | 2.000 | 177.158 |
| Clustered, T2 (both converged) | 1.548 | 5.822 |

The corrected clustered system reaches184.313 atT40, but the Newton clustered
long trajectory remains unconverged. We therefore do not present that pair as a
validated long-time trajectory comparison. The corrected system's convergence
largely accompanies dispersal, not a stable galaxy-like configuration.

This dramatic change has an explicit energy explanation. Replacing pair reduced
mass with m_i*m_j/Mref increases the central-mass couplings at the fixed nominal
epsilon. The nested initial total energy changes from Newton's-0.757 to+0.929;
the clustered initial energy changes from-1.012 to+1.957. Identical positions and
velocities are not identical modified energy or equilibrium. The kinetic reservoir
can convert into ordinary motion as coupling decreases with separation. A
conservative, partition-consistent law can therefore disperse a configuration
that was approximately circular under Newtonian gravity.

The earlier reduced-mass experiment distinguished co-/counter-rotation strongly
in its outer excursions (about4.87 versus2.06). At this repaired normalization,
both disperse strongly; that particular distinction becomes small compared with
the common expansion. The subdivision repair has not preserved an apparent
useful orbital behavior automatically.

For the flyby, closest separation changes from Newton's0.318 to0.217 under the
corrected law; both trajectories converge. This is a concrete motion-dependent
deflection in the toy model, not a prediction for an observed stellar passage.

All corrected integrations pass the frozen energy/momentum/angular momentum
threshold1e-5 and trajectory threshold1e-3. The largest corrected fine energy
drift is2.85e-9; final-position resolution differences are at most5.03e-8. Small
numerical error does not imply physically suitable orbits.

## What remains to fix

The co-moving source-partition defect is repaired. A useful next theory version
still needs a physically defined normalization/strength and equilibria consistent
with observed systems. Changing strength would be a new declared experiment,
not a retroactive way to relabel this dispersal result a success.

These checks are necessary but do not establish continuum convergence for
extended cells. Finite cell size, smoothing dependence and unresolved velocity
dispersion require separate refinement tests; co-located identical velocities
are only one controlled limit. A distribution of velocities is a physical input,
not bookkeeping. The model also lacks relativistic propagation and a photon
law. Independent source kinematics and a separate prediction target remain
required before applying it to real cubes or stellar data.

The practical finding is precise: **mass-bilinear coupling fixes artificial
dependence on particle count, but this tested coupling is too disruptive for
the original nearly circular configurations.** That is a concrete repair and
a concrete new constraint, not a rejection of every motion-dependent model.
