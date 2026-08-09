# Reference-recovery and accuracy ladder

The compiler has two separate accuracy questions.

## 1. Is the compiler mechanically correct?

The `validate` command checks the finite static grammar, dimensional controls, deterministic
enumeration, screening, state dependence, convexity, a hand-derived Euler-Lagrange equation, and
exact Newtonian recovery. The `formal-controls` command separately checks the covariant field
contract and known formal-physics answers.

Both suites must pass before a search registry is accepted. A known-answer pass validates only the
scope stated by that control; it does not validate a candidate theory.

## 2. Is a candidate physically accurate?

That requires a staged reference ladder:

| Stage | Golden control | Candidate requirement | Current status |
|---|---|---|---|
| Static weak field | Newtonian `H = D^2/2` | Recover Newton at high acceleration | Implemented |
| Formal backend controls | EH nonlinear variation/identities/ADM, scalar, Proca, Einstein-Aether, degenerate scalar-tensor and pathology controls | Reproduce each known answer in its declared scope | Implemented (99/99; see the generated formal ledger for exact per-control scope) |
| Homogeneous background | Canonical massless-scalar stiff FLRW | Derive the background equations, preserve the constraint, and uniformly retain tensor/scalar health | Interval known-answer implemented; generated candidates require their own run configurations |
| Covariant candidate dynamics | Einstein-Hilbert GR | Nonlinear variation, healthy constraints, two tensor modes, metric cone | Einstein-Hilbert known answer implemented; generated-candidate promotion remains fail-closed |
| Solar weak field | GR PPN solution | Cassini, Mercury, light bending, Shapiro delay | Formally gated Einstein-Hilbert reference implemented; generated-candidate weak-field/PPN adapter remains missing |
| Galaxy | Measured baryonic prediction | Distance-free angular-radius and Doppler-velocity ratios first; no invisible-halo targets | Sealed |
| Cluster | Same frozen action | Radial strength and transfer | Sealed |
| Strong lensing | Same physical metric | Raw image topology | Sealed |
| Generalization | Same frozen constants | Independent holdouts | Sealed |

Solar-System tests are necessary but not sufficient. Many screened alternatives can be made
arbitrarily close to GR locally. A theory becomes interesting only if the **same action** passes
Solar controls and then predicts low-acceleration galaxy, cluster, and lensing behavior without a
domain switch.

All observational stages are governed by
[`OBSERVATIONAL_EVIDENCE_POLICY.md`](OBSERVATIONAL_EVIDENCE_POLICY.md): redshift is admitted as a
measured wavelength ratio, not automatically as a distance, and unobserved components cannot be
target labels or rescues.

The machine-readable benchmark list is
[`../configs/reference_benchmarks.json`](../configs/reference_benchmarks.json). Field definitions and
the fail-closed boundary of the formal controls are in [`FORMAL_BACKEND.md`](FORMAL_BACKEND.md). In
particular, the legacy baryonic `z` is not an allowed covariant action atom under universal minimal
matter coupling.
