# TWELL-400 next source-shaped replay slice

Purpose: extend the historical-formula replay only where an audited source packet
actually provides the declared driver, geometry and time support.  Do not coerce
an incompatible packet into the static X-COP radial ABI and do not substitute a
response value for a source driver.

## Frozen predecessors

- Final-v3 TWELL card stream: 400 cards / 1,184 parameter cells.
- Source-shaped X-COP replay v1: independent numerical replay PASS, mechanical
  format-gate BLOCK retained at
  `work/audits/open-gravity-twell-400-source-shaped-rebind-replay-v1-independent-audit-blocked-babbeb72.json`.
- Audited GW/time-series v3: 680 source-anchored scenarios, canonical frequency
  representation, time as a derived unscored view, audit receipt raw
  `64ca8078109321977b3694b62e936cf1040b6f686adc20f8a82078e2cffee60c`.
- Audited Solar/planetary v2-r1: 8 targets x 13 epochs with source positions and
  GM values; no ephemeris response or residual.
- Hydro/DMO source-shaped matrix: use only after its distinct audit passes.

## Required compatibility decision

For every one of the 84 A15–A18 temporal cards, decide independently for each
audited time-bearing packet:

1. Is every declared `Dxx` input derivable from response-blind primitive source
   fields with a dimensionally correct, hash-bound program?
2. Does the packet provide ordered cadence, initial state and boundary support?
3. Does the formula's frame match the packet frame or have a frozen covariant
   transform?
4. Can the formula emit its registered observable without using hidden truth or
   scoring-only response arrays?

Only `yes` on all four permits execution. Otherwise retain one of:
`INCOMPATIBLE_FEATURE_SET`, `SOURCE_BLOCKED_MISSING_DRIVER`,
`SOURCE_BLOCKED_MISSING_TIME_HISTORY`, or `SOURCE_BLOCKED_FRAME_TRANSFORM`.

The GW/time-series audit PASS alone must not be treated as a driver adapter. Its
source waveform is not automatically baryonic acceleration, potential, radius,
density, cooling, environment or another TWELL `Dxx` quantity.

## Plausible response-blind derivations to investigate

- Solar: `D01_ACC` from source GM/positions; `D03_RAD` from heliocentric radius;
  `D07_TIDE` from the source Hessian; possibly `D12_ENV` only if the existing card
  definition and a frozen program agree exactly.
- Hydro/DMO: history/compression drivers only when the matrix exposes the exact
  source quantity, not its hidden truth label or recovery response.
- GW: admit only drivers explicitly present in the typed source packet. Do not
  reinterpret strain, likelihood, injected branch identity or noise realization
  as a source driver.

## Required artifacts

- 400 x source-release compatibility ledger with reason codes.
- All 1,184 parameter cells retained as executed, numerical-invalid, source-
  blocked or incompatible; never omit a cell.
- Per-object/source execution packet and deterministic replay for every admitted
  cell.
- Unit, frame, translation/rotation, cadence, initial-state, null-limit,
  convergence and no-response-access gates.
- Append-only predecessor preservation and a distinct independent audit.

Claim ceiling: `SYNTHETIC_DIRECTIONAL_SIGNAL`. No result is empirical support or
rejection, and finite source-prediction equality is not formula identity.
