# Galaxy observable-to-observable discovery protocol

Implementation status: the promotion pipeline now includes a sealed candidate-specific evaluator,
but its prediction-bundle registry and observational source registry are both empty. All 70 current
production candidates are blocked before data access because none has passed candidate-specific
formal health or supplied an exact direct-observable prediction bundle. No galaxy score or
scientific decision has been produced.

The first candidate-data target is galaxy kinematics, but the dataset remains sealed until the
exact candidate action passes every formal gate. The frozen machine-readable contract is
`configs/galaxy_observable_protocol.json`; its latest audit is
`runs/observation-protocol/galaxy-observable-audit.json`.

## Discovery question

The engine will ask whether one covariant, universally coupled, low-parameter law can predict
held-out spectral-line shifts from audited light, gas-emission, angular-geometry, and calibration
measurements. It will not ask for a replacement halo profile and will not receive halo mass,
concentration, radius, abundance-matching labels, or a mass discrepancy computed from the target
rotation curve.

The preferred forward problem is distance-free: calibrated angular profiles and line measurements
in, reproducible line-centroid ratios out. A physical distance is admissible only through a separate
frozen non-redshift distance protocol. Brightness-to-stellar-mass and line-emission-to-gas-mass
transformations remain explicit calibration nuisances with covariance; they are never silently
relabelled as direct observations or tuned independently to each galaxy's kinematics.

## Leakage and scoring controls

- Splits occur by whole galaxy, never by radius row from the same galaxy.
- Test-galaxy identities, measurements, summaries, and residuals remain target-blind until the
  action hash, formula hash, universal constants, calibration hierarchy, likelihood, complexity
  penalty, and stopping rule are frozen.
- The score combines a covariance-aware held-out likelihood with a preregistered symbolic
  description-length penalty and a universality requirement.
- There are zero object-specific gravitational parameters. Galaxy identifiers, per-galaxy
  acceleration scales, halo fits, target-derived discrepancies, and redshift-derived environment
  labels are forbidden inputs.
- Newtonian/weak-field GR with the same audited baryonic inputs is a required baseline. Empirical
  observable relations may be benchmarks, but neither is target truth manufactured from an
  invisible component.

## Independent lensing falsification

Lensing is not used to generate, tune, rank, or stop formulas. Only after the kinematic formula is
frozen may the same covariant action, universal constants, matter coupling, and calibration policy
be tested against calibrated image pixels, relative arc/multiple-image positions, parity/topology,
and audited time delays. Dark-matter maps, halo fits, GR-derived convergence maps treated as truth,
and lensing-only parameters are prohibited.

This sequencing makes a joint success meaningful: one law predicts both baryonic-input-to-kinematic
and baryonic-input-to-lensing observables without receiving an inferred invisible-mass label.

## Current status

The protocol audit passes, but `observational_dataset_opened=false` and
`formula_search_authorized=false`. To inspect the frozen contract without opening data:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler galaxy-protocol-audit `
  --protocol configs/galaxy_observable_protocol.json `
  --policy configs/observational_evidence_policy.json `
  --output runs/observation-protocol/galaxy-observable-audit.json
```

An observational run still requires an eligible exact action hash, an independently audited raw
dataset manifest, a committed whole-galaxy split, and completion of the Hamiltonian/principal-symbol
gates. Dark-matter truth/rescue labels, redshift-derived distances, and supernova distance moduli
remain excluded.
