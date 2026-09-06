# Execution 018: force recovery and light propagation controls

Two separately executed tasks now supply checked mechanics for the next stage
of the gravity-pattern system. Both remain THEORY_BENCHMARK_ONLY. There are no
new observed galaxy fits, admitted cube/lensing likelihoods, or derived gravity
formulas in this increment.

## Pressure: identical speeds can hide different force

The synthetic harmonic example has supplied force amplitude 625. Omitting
pressure recovers 400 and predicts the same noiseless speed curve: a 36% force
underestimate. Independent fresh speeds cannot distinguish these exactly
reparameterized curves. An independently known pressure profile restores the
force. In the varying-pressure example, omitting pressure also changes curve
shape; average fresh-noise q/N is 3.666 without the pressure term and 1.037
with it. Four noise draws per case are a diagnostic, not population evidence.

Declining pressure supplies part of the support against inward gravity, so
rotation requires less speed. This demonstrates why force labels based solely
on v^2/R can be wrong unless the tracer's other forces are accounted for. It
does not show that pressure explains a galaxy's excess inferred gravity. The
existing catalogs' corrections must be audited rather than assumed absent.

The [pressure report](../mond-atlas-pressure-support-001/README.md) includes
42 numerical controls, 21 tests, 24 noisy fits, six noiseless fits and an
impossible equilibrium retained with negative rotation-squared values. The
pressureless null is unchanged by the known-pressure model. Surface-column
pressure is distinct from local volume pressure; a flaring 3D manufactured
control verifies the required density-weighted force. The study itself is a
restricted radial fluid model, with no observed pressure, self-gravity solver,
energy closure, nonzero mean transport or new cube likelihood.

## Lensing: a separate prediction requires both metric potentials

The [light-propagation report](../mond-atlas-light-projection-001/README.md)
implements a static weak-field, straight-path, single-plane integral of the
transverse gradient of Phi + Psi. It then maps angular source/image positions
using three separately supplied angular-diameter distances and computes a
numerical Jacobian and signed magnifications. Equal potentials are explicitly
an analytic benchmark assumption. No sigma/MOND/refracted-gravity relativistic
closure has been derived or admitted by running these tests.

Both synthetic runs pass 203 required checks and 23 tests. Point/Plummer
potential integration, a separately implemented projected-density integral,
asymmetric arrangements, geometry factors, image roots, magnification parity,
units, symmetry, bounds and convergence are checked. All 251 records are retained,
including 27 coarse sweep cases that miss the fixed finest target. The arbitrary
finite-field/observational boundary problem is still open. Three cached primary
PDFs are theoretical references; no observed table, image or velocity data are
used by these runs. The earlier SLACS ingest and exposure disclosures remain.

## Parent integration review

44 relevant tests pass, with no failures, errors or skips. The parent rehashes
79 delivery files, including 23 private pressure packets and six private theory
reference/receipt files. Every pressure noise array (24), fitted prediction (30),
truth packet and failed signed equilibrium replays exactly. All 244 mathematical
light-check records and derived image/magnification results replay exactly;
seven remaining per-run checks cover metadata, source hashes and test status.
No mathematical check, threshold, candidate or response was changed during review.
The pressure plot was also visually inspected. Detailed records are in
[parent-001/verification.json](parent-001/verification.json).

All 928 previous manifest entries matched before the mutable handoff and task
plan were archived and updated. Both bounded tasks are confirmed idle/completed.
The existing baryonic-source task now works on NGC3198; the native-selection task
now tests empirical background covariance transfer for NGC2976. Their unfinished
files are excluded from this publication. Their exact handles are in the handoff.

The active goal continues with source/beam/noise validation, more eligible
systems, uncertain-depth source/force models and independent group/survey tests.
These controls are components of the system; they are not a complete observed
joint likelihood. Raw arrays stay outside Git. Ordinary publication is to main,
with exact staged bytes and remote equality checked separately.
