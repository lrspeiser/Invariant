# Gravity roadmap Item 13: relaxation and mergers

## Decision

`REJECT_ITEM13_MANGA_RELAXATION_AND_MERGERS_EXPLORATION`

The frozen visible-disturbance grammar does not improve held-out stellar-dispersion prediction
beyond structure plus the Item 12 age family. The disturbance term makes the primary error 1.67%
worse, with paired `p=0.683`, and fails every preregistered improvement/stratum gate. The same
fresh test independently replicates the frozen Item 12 age family on disjoint galaxy identities:
it improves over structure alone by 23.06% and persists after disturbance control by 22.69%, both
with paired `p=0.001`.

This is a scoped rejection of the tested visible tidal/CAS families and a same-survey,
disjoint-identity replication of an association. It is not evidence that mergers never matter,
not cross-survey confirmation, not a causal age result, and not evidence for modified gravity.

## Frozen test

Before decoding any morphology row or reading a fresh velocity response, commit
`3c1dd44221307fd09ba0b6629105e4b556478fd9` froze:

- the official 10,126-row SDSS DR17 MaNGA visual-morphology FITS file and both of its hashes;
- the exact Item 12 predictor/config/synthesis bindings and five-cell age consolidation;
- all prior-identity and coordinate exclusions;
- a 400-galaxy design balanced by visual tidal state and stellar-mass half;
- twelve provenance-labeled disturbance/relaxation families and 262,144 seeded cells;
- fixed normalizations, nested five-fold selection, baselines, strata, and admission gates;
- a strict prohibition on confirmation access, post-response formulas, and paid model calls.

The response-free morphology join produced 5,582 valid predictors. After exclusions, 4,630
independent galaxies remained. The exact 300 exploration and 100 reserved-confirmation galaxies,
plus every candidate cell, were committed at
`a805770fb697ae11b7027ff06496d2ae5a7de4f7` before any selected response was requested.

## What was tested

The baseline controls stellar mass, size, surface-density proxy, axis ratio, Sérsic index, color,
redshift, surface brightness, signal quality, morphology type, bar strength, edge-on state, and
concentration. A second baseline adds the frozen Item 12 spectral-clock-consensus × stellar-
surface-density component without retuning its cells.

The target-blind creative grammar then searches transformations and interactions of:

- visible tidal debris and the catalog's TType=11 merger-or-unclassified state;
- CAS asymmetry, clumpiness, their errors, and their coherence;
- tidal/asymmetry, bar/asymmetry, and prior-age interactions;
- disturbance consensus, relaxation deficit, morphological hysteresis, and log-periodic forms.

Every cell is labeled as a known catalog indicator, known transform, known-family combination,
combination, or unresolved. There are 66,121 `UNRESOLVED` cells. The raw count is explicitly a
screening count, not 262,144 independent physical laws; equivalent binary transforms and numeric
duplicates do not count as separate discoveries.

## Quality and held-out result

Four exact queries returned all 300 exploration responses and no confirmation responses. Under
the frozen rules, 243 galaxies pass quality and 57 fail, an 81% retention rate above both floors.

| Model | Held-out MSE | Held-out R2 |
|---|---:|---:|
| Structural baseline | 0.00852248 | 0.8196 |
| Structure + frozen Item 12 age family | 0.00655756 | 0.8612 |
| Structure + selected disturbance, no age | 0.00862377 | 0.8175 |
| Structure + frozen age + selected disturbance | 0.00666688 | 0.8589 |

Relative to the age baseline, selected disturbance worsens MSE by 1.67%. Its paired mean gain is
negative with `p=0.683`. Both visual-tidal strata regress; the lower-mass and lower-age halves
regress; only the higher-mass and higher-age halves show small positive gains. The five outer
folds select three different families—log-periodic relaxation, relaxation deficit, and a
tidal/asymmetry interaction—with one fitted coefficient reversing sign. Six of sixteen gates
fail, all belonging to the disturbance-improvement hypothesis.

Applied without reselection to stellar velocity span, the inherited cells improve MSE by only
0.49%. That secondary diagnostic was not an admission gate and is not promoted.

The RTX 5090 evaluated `1,274,019,840` candidate-galaxy validation combinations in 3.97 seconds
with CuPy 13.5.1. The maximum CPU/GPU component difference was `4.16e-16`.

## What the age replication means

The exact unweighted consolidation of Item 12's five selected clock cells reduces MSE by 23.06%
against structure alone on these disjoint galaxies (`p=0.001`). Compared with the model that has
disturbance but no age, adding age reduces MSE by 22.69% (`p=0.001`). Visible merger morphology
therefore does not explain away the association in this test.

Record `SPECTRAL_CLOCK_CONSENSUS_TIMES_STELLAR_SURFACE_DENSITY` as
`REPLICATED_ON_DISJOINT_IDENTITIES_PENDING_CROSS_SOURCE_CONFIRMATION`, origin `COMBINATION`.
The cells were fixed, but their scalar coefficient was fitted only inside each training fold.
Do not retune on these 243 responses or open the 100 sealed confirmations.

## Counterexamples and boundaries

- Visible debris and CAS structure can miss an old merger after its imaging signatures fade.
- TType=11 includes merger or unclassified objects; it is not a complete merger-stage label.
- Morphology and dynamics are from the same SDSS MaNGA ecosystem despite disjoint identities.
- Integrated spectral indices remain age proxies and can carry stellar-population systematics.
- The response is integrated stellar dispersion, not a resolved rotation curve.
- This test says nothing direct about galaxy clusters, lensing, or general relativity.
- Zero confirmation responses were opened, zero formulas were generated after response access,
  and zero paid model calls were made.

## Next real test

Advance to Item 14, resonance and coherence, on a fresh response. Freeze orbital, pattern-speed,
mode-coupling, phase-locking, and long-lived coherence observables with dimensional controls before
response access. Carry the exact Item 12 age consolidation only as a fixed comparator.

## Replay evidence

- result file SHA-256:
  `de52b4d84fd93925ef8f42c6625e33d9ae530b46584a9561d35a4ef77438310a`
- result content SHA-256:
  `d92883fec791fd8a2ef4514584676a3d36c1ee53d1691e6f92c7a7334f6d9f0b`
- response-source SHA-256:
  `f9cb6fd3843e3e92726c3f79f141ff9155fe37a3d9ce8e9c37a925a527e2910b`
- replay command:
  `python -m sigma_theory_compiler.gravity_item13_manga_relaxation_mergers check`
