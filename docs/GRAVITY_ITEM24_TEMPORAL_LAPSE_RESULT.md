# Item 24: different behavior of time result

## Decision

Item 24 is a **scoped reject on both formal tracks**. The tested temporal-metric families do not
advance as a universal gravity theory and do not yet supply a standalone phenomenon/publication
lead.

This does not make age/history, clock response, compactness, or temporal resonance less viable at
the start. Each received exactly 65,536 raw cells and the same causal, positivity, local-limit,
held-out, counterexample, and provenance rules. The result changes only the status of the precise
representations and observables tested here.

In plain language: the search repeatedly chose an oscillating clock response, but the exact
oscillation changed substantially from fold to fold and predicted new galaxies and lens delays
worse than either calibrated general relativity or ordinary data-driven controls. The stable
family choice is therefore useful evidence about how this search tries and fails, not evidence for
a physical clock resonance.

## One rule for motion and light

The frozen weak-field metric is

`ds^2 = -(1+2 Phi_t/c^2)c^2 dt^2 + (1-2 Psi/c^2) dx^2`,

with `Phi_t=Phi_N+deltaPhi` and `Psi=Phi_N+s deltaPhi`. If `H` is the radial-force response of
`deltaPhi` relative to the Newtonian baryon potential, then the same metric fixes

- slow motion: `mu_D=1+H`;
- leading photon/Shapiro response: `mu_T=1+(1+s)H/2`.

Every response is normalized to its one-AU value. The four equal-capacity mechanisms were:

1. **screened static lapse** — a known acceleration-running rewrite/control;
2. **compactness-dependent clock** — an environment-dependent clock-family extension;
3. **causal settling memory** — a positive retarded factor
   `1-exp[-t_age/(k tau_dyn)]`, with no future support;
4. **screened temporal resonance** — a low-acceleration envelope times a periodic function of
   `log(t_age/tau_dyn)`.

Here `tau_dyn=sqrt(r^3/(G M_b))`; cosmic age supplies one available history clock but receives no
extra search capacity or prior weight. Positive motion and photon responses, a causal memory
kernel, local motion deviation below `1e-5`, local photon deviation below `2.3e-5`, and luminal
photon propagation were imposed. This is a weak-field closure, not an exact covariant completion.
Relevant theory context includes work on [PPN gamma-like parameters and gravitational
slip](https://arxiv.org/abs/2106.12542), [strong-lens time delays as a slip
test](https://doi.org/10.1103/PhysRevD.100.043031), and [modified-gravity time
delay](https://arxiv.org/abs/0710.0477).

## Fresh motion and photon data

The two empirical lanes have equal weight despite different object counts.

### Galaxy motion

The predictor-only audit began with the public ALFALFA catalog:

- 31,502 catalog rows inspected;
- 8,494 passed frozen HI, distance, uncertainty, and velocity cuts after excluding 135 predecessor
  names/AGCs and another 415 predecessor-coordinate matches;
- 384 HMAC-selected objects received target-blind Legacy DR9 imaging queries;
- 244 passed the frozen photometric, shape, contamination, and inclination-proxy checks;
- 160 were frozen across four HI-mass strata: 128 exploration and 32 unqueried confirmation.

Only after this freeze were the 128 exploration `W50` values queried. All were complete. Motion is
the inclination-corrected global HI width compared with a baryon-only speed from independent HI
mass, z-band light, and Tractor geometry. This is a global-linewidth test, not a resolved rotation
curve.

### Photon travel time

The lens lane uses the [COSMOGRAIL XIX raw light-curve
release](https://doi.org/10.1051/0004-6361/201937351) and static SQLS redshifts, image separation,
image flux ratio, and lens light. Five identities whose raw values had been previewed during the
source audit were excluded. A coordinate join found 12 safe `lcab` systems; four were sealed and
eight assigned as exactly two exploration lenses per fold.

The frozen code estimated each A/B delay directly from the raw magnitudes with an interpolation,
telescope-offset, slow-microlensing-slope, robust-loss, and seasonal-bootstrap procedure. It did
not query, encode, or compare a published delay answer. Seven of eight exploration systems passed
the frozen quality rules; SDSS J1320+1644 failed with only 41 rows and a one-day best lag.

Several admitted best delays are close to the `±180 day` search boundary. That makes aliasing a
material limitation even though they pass the frozen bootstrap and zero-lag gates. No precision
lens or paper claim can rest on this estimator alone.

## Search and correction audit

- 262,144 frozen raw cells, exactly 65,536 per mechanism;
- 186,285 pass generic causality, positivity, and Solar-System gates;
- 16 more are removed because their photon response is nonfinite on at least one frozen predictor
  row, leaving 186,269 scored cells;
- 199 full selection-aware null replays;
- 15,087,789,000 held-out training-residual evaluations including nulls;
- NVIDIA GeForce RTX 5090 through CuPy;
- 3.51 seconds for the response matrix and predictor-domain correction, and 5.65 seconds for
  selection plus null replays;
- maximum CPU/GPU log-response disagreement `6.66e-16`;
- the frozen causal-memory injection is recovered as causal memory in all four folds;
- the known-GR control selects no material varying temporal response;
- zero paid model calls and zero API spend.

The first post-response replay produced `NaN` scores because the generic pre-response positivity
grid did not cover 16 candidates' behavior on the frozen lens predictors. That provisional output
is invalid and is not science. The correction removes only candidates nonfinite on a frozen
predictor row before any target scoring. It enforces the already-frozen positivity rule, adds no
candidate, changes no formula/parameter/gate/role/target, and is explicitly bound as a separate
implementation-correction commit in the machine receipt. All controls and all 199 null trials were
then rerun from scratch.

## Held-out result

The primary loss gives galaxy motion and photon delay one half of the score each.

| Predictor | Balanced held-out MSE | Candidate improvement |
|---|---:|---:|
| Shared temporal candidate | 0.941750 | — |
| Calibrated baryon-only GR | 0.523115 | **-80.03%** |
| Flexible ordinary nuisance models | 0.504430 | **-86.70%** |
| Separate best temporal formula per lane | 0.847242 | **-11.15%** |

Both channels reject the shared rule:

| Channel | Objects | Candidate vs calibrated GR | Candidate vs flexible model |
|---|---:|---:|---:|
| Galaxy motion | 128 | **-3.20%** | **-52.81%** |
| Photon delay | 7 | **-207.44%** | **-112.99%** |

The selection-aware raw null-tail fraction is `0.995`; because the candidate loses to the strong
baseline, the guarded value is `1.0`. Eighty of 135 valid objects are individual counterexamples
relative to the flexible baseline. The low-acceleration galaxy half is only `0.17%` better than
calibrated GR and every other galaxy half regresses; every galaxy half loses to the flexible
model. A few three- or four-lens halves improve, but the full photon channel fails badly and the
subsets are too small and internally inconsistent to be a phenomenon lead.

## What the stable resonance selection means

All four folds select the potentially new screened-temporal-resonance niche. That family-level
agreement is retained, but its parameter instability is decisive:

- amplitudes range from `0.3` to `10`;
- two folds suppress while two enhance;
- log frequencies range from `1` to `8`;
- phases span `pi/4` through `3pi/2`;
- transition accelerations and powers also change.

This is consistent with a flexible oscillatory family following small-sample aliases rather than
recovering one universal clock. It also loses to selecting separate temporal formulas in each
lane. The exact four cells and their counterexamples remain in the immutable receipt so a future
dataset can recognize or reject the same failure pattern without retuning it.

## Two-track disposition and next action

- **Universal-gravity track:** do not advance this exact Item 24 representation.
- **Phenomenon/publication track:** no lead yet. Retain the resonance-family selection and the raw
  independent-delay workflow as replication hypotheses, but do not call either a discovery.
- **Age/history:** the causal-settling niche remains open beyond this one exponential kernel. Its
  equal-capacity test was valid, its injection control passed, and it simply was not selected by
  these held-out observables.
- **Confirmations:** keep all 32 galaxies and four lenses sealed.
- **Numbered roadmap:** advance to Item 25, gravity coupled to time, with a fresh response and a
  covariant coupling that predicts both motion and light. Do not retune Item 24 on these 135 opened
  objects.

The machine-readable receipt is `runs/gravity/roadmap/item-24-temporal-lapse-v1.json`.
