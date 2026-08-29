# Gravity roadmap Item 45: universal interaction variables

## Outcome

Item 45 is complete as a **retrospective, non-promoted lead**. A response-blind generator created interactions among geometry, baryonic acceleration-density, radial mass gradient, cosmic time, horizon occupancy, and weak-field compactness. The same geometry-by-density gated recipe was selected in all five grouped outer folds.

The candidate improved the balanced galaxy-lens/cluster-lens loss by 16.87% over Item 44, but it did not improve both populations: it was 19.84% better on CLASH clusters and 12.84% worse on S4TM galaxy lenses. Its paired sign-flip result was `p=0.1003`, above the frozen `0.05` gate. The formal decision is therefore:

`NONPROMOTED_ITEM45_UNIVERSAL_INTERACTION_RESULT_RETAINED`

This is interesting evidence for a cross-scale interaction to test again. It is not confirmation, a dark-matter-free explanation, an alternative to general relativity, a covariant theory, or proof of historical novelty.

## Frozen search

- Scientific-freeze commit: `2a93978b0fc480aec18bae2586547ddbd824035c`
- Raw interaction candidates: 262,144
- Admitted candidates: 203,648
- Equal admitted capacity: 50,912 in each of four niches
- Response-blind interaction recipes: 64, all behaviorally distinct on the development predictors
- Equal-capacity unary/main-effect controls: 262,144
- Real development data: 28 S4TM galaxy lenses and 84 radial points from 20 CLASH galaxy clusters
- Reserved Item 43 confirmations opened: 0 of 7
- Post-response candidates added: 0
- Paid model calls: 0
- RTX 5090 candidate-point-fold evaluations: 229,511,296
- CPU/GPU selected-loss difference: `5.55e-16`

The four equal search niches were pair products, signed contrasts, gated resonances, and triple closures. Pair products and contrasts are labeled as known algebraic feature engineering. Gated and triple terms are labeled only as potentially new observational syntheses; the label explicitly does not claim historical novelty.

## Leakage control

The feature builder received only predictor fields for its calculations. Its receipt contains no observed response, target, or uncertainty values. An executable test replaces every observed response and uncertainty in the source document and confirms that the complete synthesized feature artifact remains unchanged.

The six raw coordinates were frozen as:

1. geometry: `log10(R/R_b)`;
2. density: `log10(g_bar/a0)` (an acceleration/surface-density proxy, not measured 3D density);
3. gradient: `d ln M_bar(<R)/d ln R - 1`;
4. time: `log10[t0/t(z)]`;
5. environment: `log10(R/R_H)` (horizon occupancy, not a local-density measurement);
6. field: `log10(r_s/R)`.

Each was mapped using a fixed physical soft scale, without response-fitted centering or scaling. S4TM gradients use the analytic projected de Vaucouleurs profile; CLASH gradients use response-blind finite differences of the published baryonic profiles.

## Selected formula

Let

`x_g = log10(R/R_b) / [1 + |log10(R/R_b)|]`

and

`x_d = log10(u) / [3 + |log10(u)|]`, where `u=g_bar/a0`.

The selected interaction and gate are

`I = x_g tanh(2 x_d)`

`H = 0.5 + 0.5 tanh(2 I)`.

The full-data descriptive candidate is

`nu = 1 + 6 u^-0.6 / (1 + u/100) * (0.05 + 0.95 H)`.

This means, in plain language, that the apparent gravity enhancement depends not only on local baryonic acceleration but also on where the measured radius sits relative to the system's baryonic size. Density controls how strongly that geometric position is allowed to matter. It is a compact phenomenological rule, not a derived physical cause.

All five outer folds selected the same `geometry*tanh(2*density)` recipe. Three selected the exact full-data parameters; the other two selected nearby `p` or `u_t` values. That structural stability is encouraging but remains retrospective.

## Comparative results

| Model | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Item 45 universal interaction | 0.76148 | 0.18782 | 1.33514 |
| Item 44 scale hierarchy | 0.91598 | 0.16644 | 1.66552 |
| Matched scale-free search | 1.00812 | 0.18094 | 1.83531 |
| Matched unary/main-effect search | 1.07025 | 0.16780 | 1.97269 |
| Ordinary ridge | 1.87782 | 0.26047 | 3.49517 |
| MOND/RAR | 20.99692 | 0.78454 | 41.20930 |
| Baryonic Newton | 67.65046 | 0.91086 | 134.39006 |

Relative to Item 44, Item 45 improved the balanced score by 16.87%. Relative to the matched unary and scale-free controls, it improved by 28.85% and 24.47%, respectively. The population split is important: this gain came from clusters, while the galaxy-lens score became worse.

The fixed candidate beat the fixed Item 44 control under all four frozen global baryonic-mass alternatives:

- S4TM `-0.25 dex`: 4.73% better;
- S4TM `+0.25 dex`: 7.95% better;
- CLASH `-0.10 dex`: 14.57% better;
- CLASH `+0.10 dex`: 26.32% better.

These shifts test only one narrow systematic. They do not repair the lack of fresh data or make the model-independent lens likelihood available.

## Counterexamples and decision discipline

Relative to the strongest nominal control, 19 of 48 objects were raw counterexamples: 15 S4TM lenses and 4 CLASH clusters. Eleven remained counterexamples under all four global mass-scale shifts: 8 S4TM and 3 CLASH.

None was used as a veto. A single mismatch may reflect measurement error, an incorrect baryonic model, hidden variables, an incomplete interaction, or a genuinely wrong law. Counterexample count alone is also non-terminal. Every mismatch is retained for diagnosis, and this finite sample prunes neither the selected formula nor any larger interaction family.

The aggregate gain remained positive in every leave-one-object audit and after the frozen robust trim. That says the overall gain is not caused by one exceptional object. It does not override the failed S4TM population gate or the paired `p` gate.

## What was learned

The useful result is not “we found gravity.” It is that an automated, response-blind interaction grammar found a stable cross-term that a matched unary search did not reproduce. The term suggests a concrete question: does the relationship between local acceleration and position within the baryonic structure help bridge galaxy and cluster lensing?

The strongest negative result is equally specific: this version does not yet bridge both populations. Its cluster improvement trades against the S4TM galaxy-lens score. Item 46 should preserve this candidate and all mismatches while deriving admissible dimensionless groups more systematically, rather than adding post-hoc variations to Item 45.

## Reproducibility

- Aggregate content SHA-256: `2e3088f07d39a9f403d6145f7fb5e3d6771efa175038f61e09e2b9060303ff01`
- Aggregate file SHA-256: `db243c2d12992c938b04da5578e5aefa800b9770f09f29907f51f3598af47c6b`
- Response-blind feature file SHA-256: `dd5e88ab02a302b3dc43fc8c894c1c25fbda18cb00d3d4ad2d7ea3fd0b102f92`

Replay with:

`python -m sigma_theory_compiler.gravity_item45_universal_interactions replay`
