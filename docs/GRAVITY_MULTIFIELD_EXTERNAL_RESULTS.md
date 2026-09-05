# Bounded multi-field gravity: external-quadrupole results

The previously unsupported external auxiliary field is now implemented for the
36 bounded TRIMOND cards. Across 216 fixed candidate/background combinations,
**72 are inside and 144 outside the declared historical Cassini quadrupole
screen**. All declared numerical controls pass. Every computed auxiliary
contribution is positive: it increases the quadrupole and does not repair the
higher-a0 scalar cases that already exceeded this screen.

This is a conditional leading-quadrupole calculation, not full Solar System
validation, a new ephemeris fit, a covariant theory, or a gravity discovery.
The lower-a0 galaxy and cluster development tensions remain unresolved.

## Equation and scope

The tested action function is

\[
F=Q(x)-(y-sz+s^2x)-w(xy-z^2/4),\quad
s=\lambda/(1+x)^p,\quad w=\beta/(1+x)^2,
\]

where x, y and z are the squared Newtonian gradient, squared auxiliary gradient
and twice their scalar product, divided by a0 squared. Q is the frozen
bounded-excess scalar action. The field equations follow the known
nonrelativistic tripotential framework; this interpolation and coupling remain
ansatz choices.
[Milgrom, *Tripotential MOND theories*](https://arxiv.org/abs/2305.19986).

In GM=a0=1 units, `psi=-1/r-eta_N*z`. With p_vec=grad(psi) and
q_vec=grad(chi), the auxiliary equation is

\[
\nabla\cdot(A\mathbf q)=\nabla\cdot(s\mathbf p),\qquad
A=I+w(xI-\mathbf p\mathbf p^T).
\]

A has eigenvalues 1 along p_vec and 1+wx perpendicular to it. The frozen
beta<=2 range has 0<=wx<=1/2. A Poisson-preconditioned fixed-point solver
constructs the field. Static ellipticity does not prove relativistic stability,
causality or a healthy dynamical Hamiltonian.

The background assumes `q_infinity=s(eta_N^2)*p_infinity`, with the existing
scalar mapping from physical to Newtonian Galactic acceleration. This is a
collinear boundary assumption, not a reconstructed Galaxy. The physical flux
comes from the complete action variation, including its reaction terms.

At fixed background and beta/p, the auxiliary field is proportional to lambda
and its physical contribution to lambda squared. One unit-coupling solution
therefore evaluates both frozen coupling choices without a refit.

## Controls and scan

`gravity_multifield_external_v1.json` retains three Q shapes, lambda=0.25 or
0.75, beta=0, 0.5 or 2, and p=1 or 2. The three a0 values and two backgrounds
are unchanged from the scalar predecessor. The grid sequence is
513/64/l24, 1025/128/l48 and 1153/160/l64 radial/angular/order resolution;
the last also expands the radial domain from 0.0001–100 to 0.00005–200.

The flux Green solver integrates radial and angular Legendre coefficients.
It distinguishes the quadrupole of zero-extended flux from the volume source
after removing boundary sheets. An analytic control shows why: omitting its
inner surface term changes Q2 from -3 to -1.8. Actual candidate surface terms
are small and converge.

| Check | Observed discrepancy |
|---|---:|
| Analytic smooth potential-gradient recovery | 2.94e-10 relative |
| Manufactured quadrupole | 3.75e-9 absolute dimensionless Q2 |
| Beta=0 field versus independent analytic source integral | 1.01e-7 relative |
| Signed auxiliary and quadratic physical coupling scaling | zero at recorded precision |
| Scalar Q2 versus independent reference integral, worst scanned case | 1.28e-6 absolute dimensionless Q2 |
| Auxiliary Q2 resolution change, worst case | 3.15e-6 absolute dimensionless Q2 |
| Auxiliary Q2 domain change, worst case | 3.34e-8 absolute dimensionless Q2 |
| Finest finite-domain surface contribution, worst case | 1.69e-8 absolute dimensionless Q2 |

There are 108 unit-coupling source cases and 324 candidate auxiliary solves over
the three grids, converging in 2–30 iterations. Recorded update thresholds are
1e-9. A separate test recomputes the equation residual of the returned beta=2
solution; symbolic action-variation, zero-coupling and spherical zero-correction
tests also pass. The focused suite has **156 passing tests**, including ten new
tests. Lint passes. These are numerical convergence checks, not certified error
bounds. No scientific payload was downloaded or opened.

The preflight is retained in `multifield-external-001`; the complete scan is
`multifield-external-002`. Each has exact input snapshots and hash checks.

## Conditional local-gravity result

The historical summary is Q2=(3 +/- 3)e-27 s^-2. We retain the earlier declared
two-sigma screen, [-3,9]e-27 s^-2, as a development diagnostic. The two Galactic
background values are scenarios, not independent measurements or a current
confidence interval. Frozen predecessor records retain measurement provenance.
[Hees et al., Cassini test](https://arxiv.org/abs/1402.6950),
[Hees et al., galaxy/Solar-System comparison](https://arxiv.org/abs/1510.01369).

Ranges cover both backgrounds and twelve auxiliary-parameter choices at each
fixed scalar shape/a0. They are not fitted uncertainty intervals.

| a0 (m/s²) | Q shape | Total Q2, in 1e-27 s^-2 | Inside screen |
|---|---:|---:|---:|
| 5e-11 | 0.5 | 1.802–2.575 | 24/24 |
| 5e-11 | 1 | 0.991–1.944 | 24/24 |
| 5e-11 | 2 | 0.233–0.868 | 24/24 |
| 1.2e-10 | 0.5 | 12.374–17.933 | 0/24 |
| 1.2e-10 | 1 | 17.238–27.600 | 0/24 |
| 1.2e-10 | 2 | 13.252–30.257 | 0/24 |
| 2e-10 | 0.5 | 29.853–41.155 | 0/24 |
| 2e-10 | 1 | 44.209–69.220 | 0/24 |
| 2e-10 | 2 | 37.812–70.235 | 0/24 |

No row is a full Solar System pass, and no family is pruned by these
historical-summary comparisons. Finite-radius corrections, higher multipoles,
planets, a Galactic source solution and relativistic motion remain unimplemented.

## What this changes

A separate **post-scan algebraic diagnostic** uses the quadratic coupling
identity to find intervals compatible with both assumed backgrounds. It does
not fit or select a constant. For the 36 higher-a0 shape/beta/p groups the
interval is empty: their scalar value already exceeds the screen and the
calculated coefficient of lambda squared is positive. Increasing real lambda
cannot lower those predictions for these fixed kernels and backgrounds.
The sign is supported by converged numerical coefficients; it is not a proof
covering arbitrary beta, p, backgrounds or other actions.

For the 18 lower-a0 groups, the absolute coupling interval starts at zero and
ends at 3.19–3.71 for p=1 or 11.41–13.33 for p=2, depending on shape/beta.
These approximate endpoints have no statistical confidence attached. They
provide a finite conditional range for future observed-source development;
they do not show that stronger couplings yield acceptable galaxies or clusters.

Next, transfer the solver to the already exposed observed galaxy source, using
one coupling across all source and geometry scenarios, and test whether its
nonspherical effect repairs the low-a0 underprediction within these bounds.
Exact spherical cluster pressure cannot be repaired by this auxiliary sector:
its correction vanishes on that branch. The cluster question therefore needs
observed nonspherical structure or a physically different source/action, rather
than a cluster-specific coupling. Sigma's thermodynamic-source route remains
open pending its documented measurement-model repair.

Complete scan SHA-256:
`3995286cc99cbc1bd1079758ebd418eab357e0459c61097b0583ffba08611e55`.
The discovery goal remains active.
