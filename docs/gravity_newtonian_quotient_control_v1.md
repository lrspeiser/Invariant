# Matched Newtonian quotient control V1

## Purpose

This control asks whether the canonical quotient-aware nuisance sampler can recover a
known Newtonian-generated synthetic signal when its priors, proposal kernels,
diagnostics, orbit moves, Sobol starts, and call accounting are held exactly equal to
the candidate sampler. It is a software and inference control. It does not use real
cluster targets and cannot support or reject the candidate gravity physics.

The 80-row predictor packet is target-blind. It contains only a row index,
dimensionless radius, dimensionless Newtonian baryonic acceleration, dimensionless
uncertainty, and a ten-coordinate nuisance basis. It contains no object, survey,
class, split, outcome, inferred-total-mass, holdout, confirmation, or independent-row
labels. Synthetic targets are generated only in memory after packet validation and
are never persisted as rows.

## Frozen design

- Exact candidate mechanics: 17 independent primitive priors, four independent
  scrambled-Sobol production replicates, correlated Gaussian active proposals with
  whole-proposal out-of-bounds rejection, three exact orbit moves, separate
  adaptation/settling/retained phases, and rank-normalized split diagnostics.
- Exact production maximum: 1,575,104 forward evaluations. The bounded smoke used
  948 of its 2,136-call ceiling.
- Predeclared descriptive recovery gates: at least 8 of 10 quotient truths in their
  marginal 95% intervals and maximum absolute posterior-median z score of 3.0.
- Identifiability boundary: the injected design has rank 10. Seven primitive
  directions remain null, so the 17 primitive labels are not separately identified.
- Atomic, no-clobber JSON and NPZ publication with concurrent-creator controls.

## Current result and gate state

The bounded smoke passed orbit and proposal mechanics. Its short chains intentionally
did not pass all production convergence gates, so it is not scientific adjudication.
All ten injected truths happened to lie inside the smoke's marginal 95% intervals,
but a single injected data set does not measure a false-selection rate.

The frozen SBC receipt is present but has status
`bounded_synthetic_sbc_failed_result_retained`. The Newtonian production path rejects
that receipt before loading the control contract or predictor packet. External
production authorization is also false. Therefore the production result is absent,
the control is not complete, and publication readiness is unchanged.

A future passing SBC must be append-only and versioned. Because V1 is a retained
failure, it must not be rewritten into a pass. A later Newtonian-control contract
must explicitly bind the audited passing SBC version and a separate external approval
before its production command can run.

## Replay

```text
python -m pytest -q tests/test_gravity_cluster_nuisance_quotient_newtonian_control.py
python -m ruff check src/sigma_theory_compiler/gravity_cluster_nuisance_quotient_newtonian_control.py tests/test_gravity_cluster_nuisance_quotient_newtonian_control.py
python -m sigma_theory_compiler.gravity_cluster_nuisance_quotient_newtonian_control check --config configs/gravity_cluster_nuisance_quotient_newtonian_control_v1.json --expected-config-sha256 e8ac73bf8ee6f31b2fea7c00e5ff63554361088648d1ff11b85b0259d6468b8b --implementation-receipt runs/gravity/publication-readiness/nuisance-quotient-newtonian-control-implementation-v1-final.json
```

Do not run the `run` command unless a newly audited contract binds a passing SBC
receipt, an exact external approval has been promoted, and the explicit execution
sentinel is supplied.
