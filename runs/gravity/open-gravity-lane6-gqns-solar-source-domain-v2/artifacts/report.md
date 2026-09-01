# Corrected Lane 6 GQNS Solar source-domain stress test v2

## Result

**PASS_SOURCE_ONLY_STRESS_TEST__OBSERVATIONAL_EXCLUSION_BLOCKED**. The v1 source-only calculation, its large frozen-trajectory accelerations, nonlinear decomposition, and every source-boundary failure are preserved. The v1 observational exclusion decision and its 21632.16367487 INPOP ratio are withdrawn.

The valid analytic statement is narrower: for fixed positive `A_Q` and finite `L`, the point-source enhancement `A_Q*[1-(1+r/L)exp(-r/L)]` is strictly increasing, so one exact common inverse-square constant cannot match unequal positive radii. The true minimum and maximum pairwise enhancement spreads for the four reported radii are `9.306666548525e-12` and `4.425030444719e-01`.

## Preserved source-only stress result

For D05 the median frozen moments remain `A_Q=0.648299955028` and `L=0.337966001994 au`. The maximum D05 Neptune common-projection radial stress amplitude remains `4.732035803877e-06 m/s^2`; it is a source-only diagnostic, not an observed or postfit residual and is not divided by the INPOP constant-acceleration threshold.

Only D05-D06-D07 stability under the declared Moon and asteroid refinements is claimed. Host-only, inner-boundary, and remote-source changes remain explicit localization failures.

## Independent controls

- Maximum isolated self-force norm: `0.000e+00 m/s^2`.
- Maximum independent target-minus-Sun dark-force component disagreement: `4.337e-19 m/s^2`.
- Maximum reported projection-scale disagreement: `4.979e-14`.
- The projection checks enforce their normal equations but confer no ephemeris-fit authority.

## Matched refit gate

The exact DE440 source identity, integration interval, body inventory, D05-D07 variants, DOP853 solver tolerances, variational fit, common nuisance set, and response gate are frozen in `matched-ephemeris-refit-preflight.json`. Execution remains blocked until the official ephemeris and small-body sources, solver implementation, DE440 replay gates, response manifest, weights, priors, thresholds, and injection fixtures are SHA-256 sealed. No ephemeris binary, observational response, row, or residual value was opened by this builder.

## Claim ceiling

This package is a deterministic analytic and source-only stress test plus an execution-blocked refit preflight. It makes no observational exclusion, preference, DE440 postfit, or INPOP postfit claim.
