# Bounded-action successors: local repair, resolved interaction, confinement failure

## Outcome

The shared spherical Solar System failure can be removed by changing the action,
while retaining its low-acceleration scaling and nontrivial three-potential
interaction. This work constructed 39 versioned cards across three bounded-action
shapes, checked their first variations, performed 234 initial and 54 refined joint
field solves, and tested stronger coupling through an exact amplitude identity.

It has **not** found a new first-principles gravity law. The bounded transition is
an explicit ansatz, and these are known QUMOND/TRIMOND frameworks. There is no new
galaxy fit, isolated cluster prediction, derived photon law, global stability
proof or full Solar System pass. The old cards and their negative results remain
unchanged.

## New action and what was assumed

Let u=x+epsilon^2 and choose m in {1/2,1,2}:

\[
S_m(u)=\frac{u^{3/4}}{(1+u^m)^{3/(4m)}},\qquad
Q_m(x)=x+\frac43[S_m(x+\epsilon^2)-S_m(\epsilon^2)].
\]

The assumptions are MOND-type low-gradient homogeneity and a bounded excess action
at high gradients. They motivate this sparse construction but do not uniquely
derive it from microscopic physics. The shape m is a global model choice, not an
object-specific adjustable parameter. The epsilon=1e-4 regularizer is numerical.

Direct variation gives

\[
Q'_m(x)-1=u^{-1/4}(1+u^m)^{-1-3/(4m)}.
\]

For zero epsilon, the excess action approaches `(4/3)x^(3/4)` at small x and
`4/3` at large x. With y=g_N/a0, the force correction scales as y^(-1/2) at small y
and y^[-2(1+m)] at large y. The three high-acceleration powers are therefore 3, 4
and 6, compared with the predecessor's problematic 1/2. Finite epsilon changes
the extreme origin; it must not be mistaken for an exact MOND limit there.

The action is inserted into the complete three-potential expression

\[
F=Q_m(x)-(y-sz+s^2x)-\beta\frac{xy-z^2/4}{(1+x)^2},\qquad
s=\lambda/(1+x)^p.
\]

Here the F expression uses the original invariant notation y=|grad chi|^2/a0^2,
z=2 grad psi dot grad chi/a0^2, rather than the force ratio y used in the preceding
paragraph. The code keeps these concepts separate. Its coupled field equations
come from first variations, not an acceleration curve substituted after solving.
The collinear regular branch has zero auxiliary flux and physical flux Q'_m grad
psi, as verified symbolically. The finite three-dimensional action variation and
the physical source equation also agree numerically.

Framework references: [Tripotential MOND theories](https://arxiv.org/abs/2305.19986)
and [Generalizations of QUMOND](https://arxiv.org/abs/2305.01589). The specific
bounded ansatz has not received a comprehensive novelty review; no literature
novelty is claimed.

## Local monopole screen

All 54 combinations of three kernels, six planetary orbits and three scenario
a0 values lie inside the same published precession intervals used in the
[predecessor audit](GRAVITY_LOCAL_LIMIT_AUDIT_2026_09_05.md). This is a development
screen, not a new likelihood or independent confirmation. The largest predicted
absolute anomaly in this sweep is 3.82e-7 mas/century.

At a0=1.2e-10 m/s^2, Saturn's spherical anomalies are:

| Action shape m | Extra precession, mas/century |
|---|---:|
| 1/2 | -8.253e-8 |
| 1 | -2.038e-13 |
| 2 | -1.058e-24 |

These statements apply to the regular spherical branch. The subsequent
[scalar external-field audit](GRAVITY_EXTERNAL_QUADRUPOLE_2026_09_05.md) finds
12/18 scalar scenarios outside a historical Cassini-summary screen. Multi-field
external effects, solar/planetary multipoles, time delays and light deflection
remain uncomputed. A rapid local transition does not by itself eliminate the MOND
external-field quadrupole; see [Milgrom, 2009](https://arxiv.org/abs/0906.4817).

## Joint source experiment and resolution

The fixed synthetic scene contains an anisotropic central component, diffuse gas
and two separately named members. Periodic wrapped Gaussian sources are normalized
to declared masses, and a homogeneous background is explicitly subtracted. The
three mass multipliers 0.2, 2 and 20 are scene interventions, not a mapping onto
three astronomical regimes. Source lengths, boundary conditions and units remain
synthetic. No astronomical response was opened or scored.

The initial grammar crosses m={1/2,1,2}, lambda={0.25,0.75}, beta={0,0.5,2}, and
p={1,2}, retaining one scalar control for each m. It performs 39 solves at each
of three source scales on 17^3 and 33^3 meshes, plus six legacy scalar controls.

The strongest change in member-relative RMS acceleration is approximately
**-0.1052%**. The original mesh comparison was insufficient: member RMS values
shifted by as much as 1.08% between grids. A follow-up retained beta=2 and
lambda=0.75, both p values and all three m shapes/scales on 33^3 and 65^3 meshes.
Its 54 successor solves plus six controls reduced the maximum member RMS grid
change to 0.0590%. For the strongest differential effect itself, the estimate
changed by just 0.0513% of the effect between 33^3 and 65^3. This supports that
specific small interaction as numerically resolved; it is not a continuum or
isolated-boundary certificate for all observables.

At the original couplings, member-relative RMS accelerations mostly decrease.
This is no demonstration that the family supplies missing cluster gravity.

## Exact coupling scaling and a misleading strong-coupling result

At fixed source, beta and p, the auxiliary operator is independent of lambda and
its right-hand side is linear in lambda. On the unique zero-mode-fixed solution,
chi=lambda*chi_1. The excess physical source is quadratic in lambda, hence

\[
\mathbf a(\lambda)=\mathbf a(0)+\lambda^2\Delta\mathbf a(1).
\]

The unit test compares lambda=0.25 and 0.75; a separate 30-point amplitude scan
uses lambda={0,0.75,3,10,30}, two p values and the three source scales at m=1,
beta=2. Twelve direct solves at lambda=10 and 30 agree with the extrapolated
vector field to a maximum relative error of 4.44e-12. No coupling was fitted to
data or chosen as a winning physical constant.

The scan also measures signed confinement,

\[
V=-\langle(\mathbf r-\mathbf r_{COM})\cdot
(\mathbf a-\mathbf a_{COM})\rangle_\rho.
\]

Positive V is an inward instantaneous virial contribution. It is not an
equilibrium velocity-dispersion prediction. Both positions and accelerations are
mass-centered, and tests reject a sign reversal while preserving invariance to
uniform acceleration.

For source multiplier 2, p=2 and lambda=30, member-a and member-b RMS accelerations
increase by approximately 208% and 251%. Yet their signed confinement changes by
**-179% and -204%**: both contributions become outward. Treating the larger RMS
as stronger gravitational binding would have created a false cluster success.
This particular synthetic case fails the inward-confinement diagnostic. The
calculation does not falsify every coupling, source scene or TRIMOND theory.

## Reproduction and retained evidence

Run from the repository root; each output directory must be new:

```text
python scripts/run_gravity_saturated_actions.py --output <new-directory>
python scripts/run_gravity_saturated_actions.py --config configs/gravity_saturated_actions_refinement_v1.json --output <new-directory>
python scripts/run_gravity_amplitude_transfer.py --output <new-directory>
python -m pytest tests/test_gravity_extensions.py tests/test_gravity_local_limits.py tests/test_gravity_saturated_actions.py -q
```

All 77 focused tests and lint pass. The workflow includes the new tests. Full
repository and remote CI results are not claimed.

| Evidence directory under work/gravity-first-principles | Result SHA-256 |
|---|---|
| saturated-actions-001 | c9a651a342a790a4d055511b32c099d7f428a4c8aaced22b54a51cde68b1a532 |
| saturated-refinement-001 | 30df6ae42ea3c01f9157d307fded0d1327febe8dda951a5fda3948770c9ff04f |
| amplitude-transfer-002 | 7a9a3981618dcb1b961d2a2b9408cb8b7b7eae4dd315de93886b020b39bd6e8c |

The earlier amplitude-transfer-001, which recorded RMS but not signed confinement,
is retained. Source hashes identify each run's implementation. Commit `e19700e9`
contains the initial successor; `be1cf971` contains its refined and amplitude-only
checkpoint. Later changes add signed confinement and configuration path handling.

## Next decision

Do not choose a winner from these periodic synthetic runs. The subsequent scalar
external-field audit narrows the declared a0 scenarios; now build isolated
galaxy/member/cluster predictions with the same constants and signed diagnostics,
and a separate external-field solution for the multi-field actions.
Use the quadratic identity to constrain coupling jointly, rather than interpreting
bulk-force enhancement without checking internal binding. The unresolved
length-sensitive branch requires its own higher-derivative spherical solver.

The full goal still requires root-observation transfer across galaxies, clusters
and local tests; a common derived matter/light geometry; physical stability; and
an untouched successful prediction. It remains active and unachieved.
