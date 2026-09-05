# Compatibility-first gravity extensions

## Purpose and integration

This is an opt-in successor to the galaxy-improvement-only search, based on main
commit `693cc6a90a4a4a5b19a33d1dea4d5dc31e337e76`. It preserves the historical
`configs/gravity_g4_first_principles_mechanism_search.json`, all prior receipts,
and existing observation-access policies. It does not silently reinterpret old
experiments under a new admission threshold.

The package is `src/invariant_gravity_extensions/`, included by the existing
setuptools package discovery. Its imports do not load the historical bench or
scientific data. It supplies array APIs for future campaigns and a runnable
synthetic campaign now. It is not yet wired into the old campaign automatically.

## Run from the repository root

The following commands work in PowerShell or a POSIX shell. The workflow does
not require a GPU, scientific-data downloads, Git LFS hydration, or paid APIs.

```text
python -m pip install -e ".[dev]"
python -m pytest tests/test_gravity_extensions.py -q
python scripts/run_gravity_extensions.py catalog --output tmp/gravity-extensions-catalog-001
python scripts/run_gravity_extensions.py demo --output tmp/gravity-extensions-demo-001
python scripts/run_gravity_extensions.py demo --assume-no-slip --output tmp/gravity-extensions-light-001
```

After installation, `python -m invariant_gravity_extensions` accepts the same
arguments. Every output directory must be new; increment its suffix to repeat.
Configuration is in `configs/gravity_extension_discovery_v1.json`. Copy that
config to a new version before changing the grammar, seed, margin, or grid.

Default demo: 15 action cards, 8 generated scenes, and 120 source/candidate field
solves. The opt-in light run also makes 120 synthetic Born-slab projections.
Each run writes `started.json`, `result.json`, and `receipt.json`. A failed run
retains `failure.json` without reporting a physical rejection. Code and config
hashes are bound before execution and checked afterward. Changed dependencies
quarantine the result. Completed outputs are never overwritten.

## Implemented capabilities

### 1. Compatibility rather than mandatory improvement

`policy.assess_compatibility()` accepts already-produced log10 observations,
baseline predictions, candidate predictions and stable whole-object IDs. It
weights objects equally, resamples entire objects, and estimates a one-sided
bound on candidate-minus-baseline RMS. Identical predictions survive even when
they do not improve the baseline.

The result is `COMPATIBLE`, `INCOMPATIBLE`, or `INDETERMINATE`. `next_stage()` keeps
observational equivalence and unsupported scorers separate from failure.
Compatibility is necessary, never discovery evidence or confirmation access.

The default 0.015-dex non-inferiority margin is an explicit engineering setting,
not a measurement or universal physical threshold. Set it before an experiment.
The percentile bootstrap is approximate: it does not replace a calibrated
likelihood, source covariance, or correction for the full adaptive search.

The runner is synthetic-only. The array API accepts development inputs under a
separately authorized campaign, but neither role strings nor this policy bypass
any repository seals. Confirmation and validation roles are refused here.

### 2. Action-first multi-potential and derivative-sensitive families

`actions.ActionSpec` produces symbolic action functions, their derivatives,
stable hashes and explicit scope cards. The catalogue includes:

* a regularized analytic QUMOND control;
* a quadratic-auxiliary TRIMOND cross-gradient subclass;
* a length-sensitive generalized-QUMOND subclass.

Notation:

```text
x = |grad psi|^2/a0^2
 y = |grad chi|^2/a0^2
 z = 2 grad psi . grad chi/a0^2
 h = ell^2 sum_ij(psi_ij^2)/a0^2
 Q(x) = x + (4/3) x/(x + epsilon^2)^(1/4)
```

Q is a MOND-limit analytic control, NOT the empirical SPARC RAR. At zero
regularizer and positive x its deep power is 3/4 and its derivative tends to one
at high x. The corresponding AQUAL deep action has power 3/2 in the squared
physical gradient, or power three in its magnitude. Finite epsilon regularizes
the origin. Its physical effects must be checked against epsilon and grid
refinement before interpreting any real-source prediction.

The TRIMOND subclass defines `s=mixing/(1+x)^power` and

```text
F = Q - (y - s*z + s^2*x) - beta*(x*y - z^2/4)/(1+x)^2.
```

The added terms and their first variations vanish on the collinear branch
`grad chi=s grad psi`. This can preserve a one-dimensional symmetric solution;
it does not guarantee identical disk rotation curves. In nonsymmetric sources,
`s grad psi` need not be a gradient, so the auxiliary equation is actually solved.
For nonnegative beta the fixed-psi auxiliary operator is positive definite after
sign reversal and zero-mode fixing. This is not a relativistic health theorem.

The generalized-QUMOND subclass uses

```text
P(x,h) = x + (4/3) x/(x + h + epsilon^2)^(1/4)
lap(phi) = div(P_x grad psi) - ell^2 partial_i partial_j(P_h psi_ij).
```

The Hessian double divergence is required by variation. A negative-control test
demonstrates that dropping it changes the solution. Setting ell=0 recovers the
same QUMOND control. Zero-mixing TRIMOND also reduces to that control and is
removed as a duplicate when generating the default catalogue.

Symbolic certificates test first variations, the collinear auxiliary and physical
flux identities, and fractional-power limits. Action values alone are insufficient.
These are known-family adapters and illustrative sparse subclasses; no historical
novelty, empirical support or universal stability is claimed.

### 3. Joint fields before averaging

`fields.solve_fields()` solves the complete source jointly. It does not add
independently computed modified-gravity fields of separate galaxies.
`joint_density()` retains component names and masses while assembling the common
source. Partitioning one component into identical mass pieces leaves that source
unchanged. `member_relative_acceleration()` subtracts the mass-weighted
centre-of-mass acceleration, rather than mistaking uniform cluster acceleration
for additional stellar confinement.

The implemented domain is an odd periodic grid with an explicitly declared
zero-mean density contrast. The mock campaign subtracts and records a homogeneous
background. Poisson itself refuses an unbalanced source rather than silently
subtracting one. The operators are discrete adjoints, permitting variational tests.
A final physical acceleration is a gradient even when an auxiliary flux is not.

**This is not an isolated cluster or an adaptive nested-grid solver.** Member
components currently share the periodic mesh. Open boundaries, nested resolution,
real-source convergence and independent deprojection remain to be implemented.

### 4. Time-domain controls rather than snapshot proxies

`dynamics.evolve_auxiliary()` evolves
`q_tt+c_q^2(-lap+m^2)q=J` with explicit q and qdot initial data. Each source value is
held over the following time interval, and the terminal source is unused. Exact
spectral oscillator updates avoid a CFL instability for this control. They do not
prove continuum causal-cone or relativistic consistency.

`dynamics.InertiaMemory` integrates the conservative worldline control

```text
L/m = v^2/2 + mu*qdot^2/2 - coupling*v.qdot - mu*omega^2*q^2/2 - Phi(x).
```

It requires `mu>coupling^2`. Eliminating q creates frequency/history dependence;
its initial state is explicit. It is an auxiliary-oscillator implementation test,
not a complete MOND modified-inertia law. Tests cover the Newtonian zero-coupling
limit and conservation of the complete trajectory energy.

### 5. An honest matter/light boundary

`observables.assumed_metric()` refuses to infer photon propagation from a static
nonrelativistic action. An optional `assumed_no_slip` closure sets the two weak
metric potentials equal and labels `derived_from_action=false` in provenance.
`born_lensing()` then projects deflection, convergence and shear through one
synthetic periodic slab with an explicitly supplied distance factor.

This is not reduced shear, a real cluster likelihood, strong-image modeling,
time delays, or a derived covariant completion. Unsupported sectors raise an
explicit error. The existing scalar-tensor proof pack remains unchanged and is
not silently connected through an invented metric.

### 6. Covariance-aware experiment ranking

`policy.rank_experiments()` whitens predictions using the full supplied covariance,
projects away the whitened nuisance-Jacobian span, and ranks experiments by model
separation. It reports both an average utility and the least separated model pair,
so an average cannot hide an equivalence class. This is synthetic experimental
design, not a calibrated detection statistic or a physical intervention claim.

## Validation performed during implementation

The focused suite passed all 46 tests locally under Python 3.13.5, NumPy 2.3.5,
SciPy 1.17.0 and SymPy 1.14.0. It includes analytic Poisson and lensing controls,
discrete integration by parts, finite-difference action variations, zero-amplitude
limits, axis permutations, source partition, correct COM subtraction, temporal
support, worldline energy, explicit unsupported sectors, deterministic runs,
append-only output, and mid-run dependency-change quarantine.

The default 120-solve synthetic run also completed with the explicit no-slip
closure. Maximum physical Poisson relative residual was about 1.23e-14; maximum
auxiliary Euler relative residual was about 7.84e-10. The worldline control's
relative energy drift was about 2.01e-10. These are numerical controls, not
astronomical predictions. The local result SHA-256 was
`7af4fcd661c1baaa7d385c2538ced525a6c9b864da2d074db6941fd613a0adbf`.
Exact run bytes include dependency versions and source hashes; different numerical
versions can legitimately change them. Check tolerances and provenance, not merely
whether a different environment returns the same hash.

The demo deliberately generates truth with its own baseline adapter. It is an
end-to-end smoke/regression test, NOT an independent recovery/power certification.
The implementation and tests were authored together. The full repository suite
was not run, and local ruff was unavailable. A scoped GitHub Actions workflow adds
Linux/Windows and Python 3.11/3.12 tests plus lint and the synthetic demo, without
hydrating scientific LFS data. Its outcome must be checked separately.

## Remaining work (not implemented by this successor)

1. Isolated/open nested cluster/member domains and real-source convergence.
2. Authorized SPARC/RAR and root photometry/IFU/shear adapters with full covariance.
3. Derivation from covariant scalar/vector-tensor actions to an on-shell physical
   metric, followed by joint matter/photon likelihoods.
4. Generalized vector-tensor evolution, cosmology, strong fields and precision-local
   tests.
5. Independent alternate-universe generators and full-search false-positive/power
   calibration.

Use the new action-card hashes and array APIs when creating a successor campaign.
Do not edit frozen predecessor hashes, spend a sealed data product, treat missing
adapters as failed physics, or promote a synthetic compatibility result.

## Primary theory references

* Milgrom, TRIMOND: https://arxiv.org/abs/2305.19986, equations 2, 3, 7-9.
* Milgrom, Generalizations of QUMOND: https://arxiv.org/abs/2305.01589,
  equations 6-7 and the length-dependent construction in 14-16.

The research objective remains a new testable interaction, not a new spelling of
an existing law. This release supplies executable candidate families and controls
for that search without claiming that the final unified theory has been produced.
