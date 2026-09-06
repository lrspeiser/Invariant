# Independent pre-run code review

Reviewed `scripts/run_mond_atlas_clock_relay.py`, `scripts/mond_atlas_clock_relay.py`, `tests/test_mond_atlas_clock_relay.py`, the frozen configuration and PREFLIGHT. No shared files changed and no response scores recomputed during this review. Executed the manufactured unittest file: seven tests passed.

## Selection and data flow

The loader opens only the 139 historical registered archive member bodies. Whole-archive hashing reads bytes for identity, not numerical response access; the receipt's `reserved_member_bodies_opened=0` describes parsing/member-body access, not a claim that no raw bytes from the archive were hashed. The source archive/metadata/history hashes are checked before the manufactured tests and response evaluation.

Only radius, photometric/HI metadata, published baryonic force templates and disk surface brightness enter predictions. Vobs and its uncertainty are eligibility/response columns. There is no imported halo target. Quality/inclination, finite-value and positive-velocity cuts reproduce the declared selection. Requiring positive baryonic force across all mass factors is declared in advance.

Candidate loss arrays include all galaxies, but `loss_select` slices exclusively to the training galaxy mask. Fold construction uses identity hashes and fixed seeds, not outcomes. Each reported galaxy is predicted by a candidate selected without that galaxy's score. The all-family selector likewise selects on training losses. No held-label selection leakage was found. The sample was historically exposed, so these are internal prediction estimates, not an independent discovery test. The unit leakage test checks the selection primitive; the actual runner masks also appear correct on inspection.

## Formula checks

Acceleration units are (km/s)²/kpc; A0 converts correctly from 1.2e-10 m/s². Stellar M/L factors multiply squared component velocities, and signed gas forces are retained. The simple algebraic MOND control satisfies g²/(g+a0)=gb. It is one interpolation prescription, not a numerical AQUAL/QUMOND disk solution.

The finite-p2, finite-p3 and clock potential gradients match their implemented accelerations. Their additional effective enclosed source increases to a finite value. The kernel truncates its effective source at 10L, preserving continuous acceleration with a derivative discontinuity at the cutoff. This is a declared analytic approximation, not a converged spatial convolution. The clock potential has an explicit zero-at-infinity convention and a photometric softening scale; its formula does not measure time transfer or establish energy conservation.

The mass proxy assigns 0.5 M/L to all luminosity, while the force template uses 0.7 for bulge light. Configuration declares this simplification, so it is not an implementation mismatch; bulge-heavy results require caution. The surface relay uses stellar disk surface brightness only, as declared, not total density, gas opacity or an intervening column.

## Reporting gaps requiring a supplement

1. `source_only_parameters: true` in `summary.json` is ambiguous and, read literally, wrong. Formula parameters such as mf, eta, beta and length factor are selected using training velocities. Only predictor inputs are source-only. Explicitly state this in a correction/supplement and user-facing report; do not claim prediction from baryons without learned population parameters.
2. Frozen configuration promises every held prediction and reporting parameter bounds. Runner saves selected radial predictions only for the overall selector, although all other predictions can be reconstructed from saved fold choices and source inputs. Save independently reconstructed family predictions and selected boundary frequencies in an append-only review supplement. This does not require new fitting or modifying the frozen run.
3. Bootstrap intervals resample fixed per-galaxy differences after cross-validation. They do not refit overlapping training sets or include model-search, nuisance, survey or radial covariance uncertainty. Treat them as descriptive variability in this exposed sample, not discovery significance or a fully calibrated sampling interval.
4. `clock_potential` is a static radial shape. Even an improvement does not distinguish a time-energy reservoir from other force laws with similar radial shape. Its beta=0 baseline is embedded, and other candidate families similarly include duplicated zero-strength Newton cases; first-grid tie breaking is deterministic and disclosed.

## Robustness notes

The absorption expression computes exp before log; sufficiently extreme surface brightness could underflow. The runner checks all candidate predictions and fails before scoring if nonfinite, so it does not silently discard a failed candidate. Manufactured fixtures do not span arbitrary high brightness. Should a failure occur, preserve the run and use an explicitly versioned algebraically equivalent log-domain implementation, not outcome-selected exclusions.

The source hash inventory itself is not included in runner bindings, although it contains the expected hashes. For full provenance, include source-audit artifacts in the publication manifest/review bindings. No change in numerical conclusions follows from this bookkeeping gap.

Conclusion: no blocking formula-unit, cohort or train/test selection error was found in this read-only review. Statistical and physical interpretation remains restricted as above; reporting gaps should be corrected before presenting the milestone as complete.
