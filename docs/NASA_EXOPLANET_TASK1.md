# NASA Exoplanet Task 1: Measured-Law Calibration

Task 1 is the first gate in the real-problem roadmap. It asks Invariant to recover a
three-column multiplicative relation from a real NASA Exoplanet Archive snapshot while the
discovery engine sees only anonymous positive columns and reported uncertainties.

## Claim boundary first

This is a calibration on real catalog values, not a new physical result. Exoplanet catalog
semi-major axes and stellar masses can be inferred, model-dependent, or algebraically coupled to
period. A recovered relation therefore demonstrates that the discovery instrument can find the
structure present in the catalog; it does not independently confirm the underlying law.

The target is classical and already occurs elsewhere in this repository. The operational
blindness claim is narrower: the executable generic generator and its serialized input contain no
physical names, target vector, target formula, host names, or holdout rows.

## Architecture

The task has four boundaries:

1. `nasa_exoplanet_task1.py` retrieves and validates an external CSV snapshot, filters rows with
   missing or very uncertain values, groups by host, and creates a chronological host-disjoint
   split.
2. `anonymous_monomial_discovery.py` receives only `x0`, `x1`, `x2` values and uncertainties. It
   evaluates primitive integer exponent vectors and has no scientific-domain vocabulary.
3. Old, new, 32 uniformly random, and unit-rescaled candidates are all frozen before the target
   interpretation or holdout is opened.
4. The evaluator compares the frozen candidates with the hidden structural target and scores the
   holdout without refitting the relation's constant.

Each old/new/random run receives exactly 256 candidate evaluations. The new lane uses a generic
Occam ordering over two- and three-column primitive monomials. The old lane is limited to pairwise
relations. Random lanes shuffle the same complete candidate pool and use a recorded seed.

## Exploratory pilot

Source snapshot:

- NASA Planetary Systems (`ps`) default parameter rows;
- 2,899 source rows, 2,531 eligible rows, and 1,911 eligible hosts;
- 2,020 training rows from 1,528 hosts;
- 511 holdout rows from 383 disjoint hosts;
- snapshot SHA-256 `a35e75d1fd3bca2480896b29f1516815a86468ccacbba7a851e629863d53accf`.

Measured result:

- new best candidate: `x0^2*x1^-3*x2 = constant`;
- old best candidate: `x0^2*x1^-3 = constant`;
- new median holdout response log error: `0.0022723372554187549` (about 0.23%);
- holdout rows within the propagated one-sigma interval: `0.92954990215264188`;
- holdout rows within the propagated two-sigma interval: `0.97847358121330719`;
- 1 of 32 uniformly random lanes recovered a candidate at least as good as the new lane;
- the exponent vector was unchanged after independently rescaling all three column units;
- the new relation beat the old lane, a constant predictor, the best univariate log-linear
  predictor, unconstrained multivariate log-linear regression, and an unconstrained quadratic
  log-predictor.

All numerical checks passed. The receipt decision is nevertheless `BLOCKED`, with status
`EXPLORATORY_PASS_NOT_GATE_ELIGIBLE`, because aggregate provisional-holdout performance was
inspected while the first implementation and thresholds were designed. It cannot unlock Task 2.

## Frozen confirmation

The confirmation configuration was written before any row values from its lane were retrieved. It
selected non-default literature parameter sets published from 2020 onward. Before freeze, only these
availability counts were requested:

- 1,028 rows with the three primary values present;
- 841 rows with all six uncertainty bounds present;
- 508 distinct hosts before eligibility filtering.

No row values, candidate scores, or holdout performance from this alternate-reference lane were
opened during protocol design. The implementation was committed at `587a966`, and the authorization
bound that commit and every executable/configuration hash before the CLI retrieved the values.

A confirmation run is allowed only after:

1. the generator, orchestrator, test, target, configuration, and CLI registration are committed;
2. an authorization object binds their file hashes, the Git commit, the source query, and the
   no-prior-value-access declaration;
3. the CLI retrieves the values itself after the authorization time—`--raw-csv` is forbidden;
4. the same frozen performance requirements are applied without repair.

### Confirmation result: honest REJECT

The first confirmation did not pass:

- 1,028 external source rows;
- 805 eligible rows across 429 hosts;
- 648 training rows from 343 hosts;
- 157 holdout rows from 86 disjoint hosts;
- exact recovered structure: `x0^2*x1^-3*x2 = constant`;
- median holdout response log error: `0.0037307073835366467`;
- new lane better than the old pairwise lane and every frozen fitted baseline;
- 1/32 uniformly random lanes matched the new winner;
- reported-error coverage: 89.81% within one sigma and 91.72% within two sigma.

The frozen requirements were 90% and 95%, so the receipt is `REJECT / PERFORMANCE_GATE_FAILED`.
The miss is narrow at one sigma and material at two sigma. No threshold, exclusion, candidate,
split, or uncertainty was changed after opening the data.

This is precisely why the gates exist: the formula can be structurally right and predict much
better than alternatives while the stated uncertainty model remains inadequate. Task 2 stays
locked.

### Next untouched attempt

The rejected 2020-and-later lane is now debug material. A later Task-1 version may use it to develop
asymmetric log-interval propagation and an explicitly calibrated systematic-error model, but it
must face a different untouched lane.

Only availability counts—not values—have been requested for the proposed next lane: non-default
parameter sets published from 2015 through 2019 with all six uncertainty bounds present. It has
2,445 candidate rows and 1,445 distinct hosts before eligibility filtering. Its values may not be
retrieved until the revised uncertainty semantics, thresholds, code, tests, and query are committed
and authorized.

## Version 2 confirmation: PASS

Version 2 was committed at `d669c3f` before any values from the 2015–2019 lane were retrieved. Its
authorization binds the commit, query, target, v1 shared machinery, v2 evaluator, tests, and every
budget. It preserves v1's rejection and preregisters statistically calibrated empirical coverage
floors:

- at least 60% inside reported one-sigma propagation, compared with the 68.27% Gaussian reference;
- at least 90% inside reported two-sigma propagation, compared with the 95.45% Gaussian reference;
- p90 standardized residual no larger than 2.0;
- all original structure, baseline, old/new, random-budget, unit-stability, and chronology gates.

The untouched run retrieved 2,445 rows, excluded 19 over the relative-uncertainty ceiling, and
retained:

- 1,940 training rows from 1,151 hosts;
- 486 holdout rows from 288 disjoint hosts;
- exact winner `x0^2*x1^-3*x2 = constant`;
- old-lane winner `x0^2*x1^-3 = constant`;
- median holdout response log error `0.00035672743753956127` (about 0.0357%);
- empirical one-/two-sigma coverage 98.15%/99.18%;
- p90 standardized residual `0.15204545676513562`;
- 1/32 uniformly random lanes matching the new winner;
- lower median holdout error than the old lane and every frozen fitted baseline.

All nine checks pass. `confirmation-v2.json` records `PASS / GATE_PASS`, and Task 1 is complete.

The interpretation remains deliberately narrow. This proves that the frozen discovery instrument
can recover and predict a known three-column structure in real external catalog values under this
protocol. It does not prove that the method is historically creative, that the law was discovered
from independent direct measurements, or that Invariant is generally better than random search.
Task 2—the fresh mathematical falsification trial—is now unlocked.

## Commands

Exploratory receipt replay:

```powershell
sigma-nasa-exoplanet-task1 validate --root .
```

Freeze the confirmation after committing the bound implementation files:

```powershell
sigma-nasa-exoplanet-task1 authorization-template `
  --root . `
  --config configs/nasa_exoplanet_task1_confirmation.json `
  --output work/private/nasa-exoplanet-task1-confirmation-authorization.json
```

Retrieve and run the frozen confirmation:

```powershell
sigma-nasa-exoplanet-task1 run `
  --root . `
  --config configs/nasa_exoplanet_task1_confirmation.json `
  --authorization work/private/nasa-exoplanet-task1-confirmation-authorization.json
```

Replay the passing version 2 confirmation:

```powershell
python -m sigma_theory_compiler.nasa_exoplanet_task1_v2 validate `
  --root . `
  --config configs/nasa_exoplanet_task1_confirmation_v2.json
```

The confirmation receipt is immutable. A failure makes that source lane debug material; it may not
be repaired and relabeled blind.
