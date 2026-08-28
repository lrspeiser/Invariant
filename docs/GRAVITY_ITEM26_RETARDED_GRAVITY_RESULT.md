# Gravity roadmap Item 26: retarded gravity

## Decision

**Formally inconclusive because the frozen quality floor was missed; strongly negative diagnostic direction; item complete.**

The source supplied usable, two-sided curves at all three frozen radii for 28 exploration galaxies versus the preregistered minimum of 50. That shortfall permanently prevents promotion. The unchanged diagnostic search also found no material evidence for the tested finite-propagation closures, so there is no phenomenon/publication lead to carry forward from this representation.

This closes only the specific Hα-activity-times-path-delay proxy. It does not reject retarded field equations using measured historical mass maps, tensor/velocity source terms, environmental propagation, or a different direct observable.

## Equal-capacity theory boundary

Exactly 65,536 raw candidates were assigned to each of four peer mechanisms:

1. luminal single-path retardation, a known retarded-potential control;
2. Lorentz near-zone compensation, representing the known aberration-cancellation class;
3. a size-dependent transition from `c` to a subluminal carrier floor;
4. causal multi-path echoes, a potentially distinct synthesis.

The search began with 262,144 cells. Response-independent causality, positivity, Solar-local, and finite-domain gates left 166,531:

- luminal single path: 39,881;
- Lorentz-compensated: 65,219;
- finite carrier: 30,894;
- multi-path echo: 30,537.

Of these, 141,412 are eligible under the frozen universal propagation-speed rule. Subluminal cells remain in the search so their empirical behavior can be characterized, but they cannot be promoted as universal gravity after GW170817/GRB 170817A. No superluminal or advanced-support cell was admitted.

Every family recovered its exact frozen synthetic injection in all five folds. An important pre-response audit correction made the finite-carrier injection genuinely non-equivalent: a constant slower speed at a single radius is exactly degenerate with amplitude, so the frozen family instead has a size-dependent transition from `c` to a slower floor. This was corrected, tested, committed, and rebound before any rotation response was queried.

## Real-data boundary

The experiment used the Hα kinematic survey of Herschel Reference Survey galaxies. Before response access it froze stellar mass, optical size, distance, ellipticity, inclination, H I deficiency, environment, Hα luminosity, and a specific-growth proxy.

The identity audit examined 27 predecessor sample manifests, 3,488 normalized names, and 3,738 coordinates. It removed 19 name overlaps and four additional coordinate overlaps. Predictor-quality rules left exactly 80 fresh galaxies:

- exploration: 64;
- sealed confirmation: 16;
- confirmation response queries: zero.

The response acquisition queried each exploration HRS identity separately. Fifteen exact identities had no rows in the published rotation table. Among the remaining curves, the strict requirement that both approaching and receding sides bracket `0.8`, `1.3`, and `1.8` effective radii left 28 valid galaxies:

- no two-sided table response: 15;
- failed first at `0.8 reff`: four;
- failed first at `1.3 reff`: 12;
- failed first at `1.8 reff`: five;
- valid at all three radii: 28.

The frozen minimum was 50. It was not lowered.

## Response-access incident and corrections

The first concurrent exploration acquisition aborted when an exact frozen HRS query contained no table rows. No response artifact was written and no confirmation was queried, although some permitted exploration bodies may have been downloaded transiently before the worker pool stopped.

Two disclosed implementation corrections followed without changing a response value, formula, parameter, role, threshold, or confirmation boundary:

1. treat a headerless empty exact exploration query as an explicit missing-response receipt and continue the unchanged identity list;
2. when the unchanged 50-galaxy floor is missed, preserve `INCONCLUSIVE_QUALITY` while allowing the frozen search to run diagnostically on at least 20 valid galaxies.

Both corrections are bound in the machine receipt. The diagnostic cannot promote either formal track.

## Tested response

For each radius, the two sides were interpolated independently and averaged. The causal response was

```text
log(V) = log(V_instantaneous) + 0.5 log(mu_R)

log(mu_R) proportional to
specific_Halpha_growth * nonnegative_path_delay * frozen_radial_window.
```

This is a deliberately scoped weak-field proxy. Integrated Hα luminosity traces recent star formation; it is not a measured historical baryonic mass map. The instantaneous baseline used the stellar exponential-disk velocity proxy, while the flexible ordinary model also used size, activity, ellipticity, inclination, H I deficiency, distance, concentration, and environment.

## Diagnostic results

At the primary `1.3 reff` radius:

| Model | OOF MSE | Change versus selected causal response |
|---|---:|---:|
| selected causal response | 0.0225267 | — |
| instantaneous baryonic | 0.0225244 | causal response is **0.010% worse** |
| flexible ordinary nuisance | 0.0240931 | causal response is 6.50% better |

The flexible model was itself weaker than the simple instantaneous baseline, so beating it does not supply positive evidence. The full 99-trial selection-aware replay gave `p=0.38`; the strongest null improvement exceeded the observed value.

The unchanged radial replays were also negligible:

- `0.8 reff`: 0.0045% better than instantaneous;
- `1.8 reff`: 0.0308% worse than instantaneous.

Broad halves did not hold:

- high mass: 0.0099% better than instantaneous;
- low mass: 0.0405% worse;
- high Hα activity: 0.0106% better;
- low Hα activity: 0.0386% worse.

The selected response improved a separate two-sided-asymmetry intercept by 2.46%, but this underpowered replay was not selection-significant and cannot overcome the primary, radial, slice, or quality failures. Twelve of 28 galaxies were individual counterexamples versus the flexible model.

Four folds selected the same suppressing multi-path echo cell; one selected a subluminal finite-carrier cell. The echo has amplitude `300`, radial power `4`, transition `0.2 reff`, path multiplier `300`, and echo weight `0.03`, but its actual predicted correction on the frozen support is effectively zero. The finite-carrier fold has a `0.001c` floor and is ineligible for universal promotion. Family direction therefore supplies neither a theory nor a paper lead.

## Controls, compute, and cost

- all four synthetic niche recoveries: pass in five of five folds;
- instantaneous false-positive control: pass;
- CPU/GPU check: exact agreement;
- GPU: NVIDIA GeForce RTX 5090 through CuPy;
- admissible candidates: 166,531;
- training residual evaluations: 1,958,407,500;
- full selection-aware null trials: 99;
- measured search wall time: 84.00 seconds;
- paid model calls: zero;
- paid API spend: `$0.00`.

The immutable machine receipt is `runs/gravity/roadmap/item-26-retarded-gravity-v1.json`. Exact predictor, exclusion, role, candidate, response, incident, correction, and compute receipts are under `runs/gravity/roadmap/item-26-retarded-gravity-v1-source/`.

## Exact next actions

1. Preserve the tested activity-delay and echo families as scoped failure-space knowledge; do not retune the 28 opened responses or query the 16 confirmations.
2. Do not promote the tiny echo/asymmetry pattern to a paper-track lead. A future retarded-gravity attempt needs a new source with measured time-dependent or spatially resolved baryonic structure and enough complete two-sided curves.
3. Advance the numbered roadmap to Item 27, gravitational memory, which tests a causal fading state rather than Item 26's finite path delay.
