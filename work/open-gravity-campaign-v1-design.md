# Open Gravity Campaign v1 — implementation contract (working design)

This is a target-blind implementation note, not an authority or result artifact.

## One-campaign state machine

1. Load and validate the committed registry, final TWELL successor, GP01 foundation,
   X-COP source preflight, final prior-art contract, static-radial adapter, and source
   availability v2 package by exact file and content hashes.
2. Build the complete typed-card set and the immutable `FROZEN_UNRUN` manifest.
3. Validate the manifest with the committed registry validator and exact live cards.
4. Atomically create the single genesis terminal ledger at the registry-frozen path.
   The ledger reserves campaign ordinal 1 permanently; its existence forbids a second
   response-scored campaign. The reservation is not a scientific result.
5. Only after the manifest and reservation are durable may the executor open the
   frozen development responses. It may never open confirmation, independent, group,
   or lensing responses and may make no network, model, paid, or tuning calls.
6. Run the pilot and full-development stages without exposing partial rankings or
   altering formulas. A pilot failure is retained; it does not edit the full set.
7. Atomically publish per-object dashboards, counterexample ledger, multiplicity
   ledger, closure matrix, raw result, deterministic adjudication, and one terminal
   success/failure artifact. The campaign is terminal even with zero survivors.

## Candidate scope

- Register all 400 TWELL concepts and every frozen parameter cell.
- Response-score only static radial cells whose source and observable closure is exact.
  Dynamic/retarded/memory cells remain `SOURCE_BLOCKED` or `THEORY_ONLY`.
- Register GP01-L (`n=1,2,4`), the spherical quasi-static GP01 elliptic grid, and the
  T1/T2/telegraph/action branches with their honest blocked/quarantined dispositions.
- Register baseline and strongest available executable comparators as rival/control
  cards. Named but unavailable AQUAL-EFE, Penner, Refracted Gravity, EMOND, MOG, and
  published nonlocal solvers remain source/solver blocked and may not be impersonated.
- Populate all five discovery lanes. A lane without a valid response closure receives
  theory-only/source-blocked rows, never a fabricated score.

## Frozen development data

- SPARC: exact admitted 139-object/2,720-row ledger. Pilot 28, full-development 111
  disjoint objects. Source inputs are baryonic components and declared metadata;
  velocity observations and uncertainties are responses.
- X-COP: A1644, A1795, A2142, A2255, A2319, A3266, A85, ZW1215 only. Pilot A85/A3266;
  full-development the remaining six. Density and available stellar profiles are
  causes; pressure and temperature are responses with shared X-ray ancestry.
- Three missing stellar profiles share one global nuisance rule; no per-object gravity
  parameter is fitted.

## Observable closures

- SPARC matter closure: `g_eff=A_g*g_b`, `v_pred=sqrt(max(A_g*v_bar^2,0))`.
- X-COP matter closure: use adapter-derived source acceleration, integrate the frozen
  spherical hydrostatic pressure-gradient equation inward under each global nuisance
  case, and derive temperature from pressure/density. Candidate state never uses
  pressure or temperature.
- Matter-only candidates make no light, redshift, capture, GW, quantum, or cosmology
  claim. Those rows are explicit `NO_CLAIM`, `RECOVER_CONTROL`, `THEORY_ONLY`, or
  `SOURCE_BLOCKED` in the closure matrix.

## Nuisances, controls, and metrics

- Freeze a finite SPARC stellar mass-to-light nuisance set and a finite X-COP global
  source/systematics set before response access. No per-object tuning.
- Controls include identity, radial-factor reversal, baryon-only, empirical RAR,
  available halo ceilings, the previous cross-scale law, and the extended-source clock.
- Score equal-weight object-level standardized residual losses. Preserve per-radius
  failures, equal-object galaxy aggregation, and equal-cluster pressure/temperature
  aggregation.
- A scientific win requires more than a 2% loss improvement beyond an indifference
  band, broad object support (at least 84/139 galaxies and 6/8 clusters), both domains,
  leave-one-object-out stability, and every frozen worst-case/subgroup ceiling.
- Because the legacy search ledger is incomplete, no global nominal discovery p-value
  is valid. Charge the known legacy floor in the sequential budget and cap this
  campaign at `DEVELOPMENT_SIGNAL`.

## Required output

- 139 galaxy dashboards and 8 cluster dashboards.
- Candidate, comparator, nuisance, transformation, object, observable, metric, and
  complete multiplicity ledgers.
- Append-only failures/counterexamples with worst object and radius.
- Matched-environment analysis marked `SOURCE_BLOCKED` unless an admitted independent
  environment map exists; shuffles are not fabricated.
- Matter/light/redshift/capture/GW/quantum/Solar/cosmology closure matrix.
- Lay result naming what improved, what failed, and the next decisive missing test.

