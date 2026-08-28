# Gravity roadmap Item 25: time-varying gravitational strength

## Decision

**Universal-gravity promotion rejected; phenomenon/publication promotion rejected; stable oscillatory hint retained for unchanged independent replication.**

This wording matters. The experiment did not establish changing gravity, and it did not clear the frozen evidence threshold for a paper-track lead. It nevertheless found a coherent pattern worth testing again: every outer fold independently selected the exact same oscillatory cosmic-history cell, and that fixed family improved all four preregistered redshift and baryonic-mass halves. Under the equal-viability policy, this branch is preserved rather than pruned.

## What was tested

The response-blind generator gave exactly 65,536 raw cells to each of four peer mechanisms:

1. smooth cosmic power-law evolution, a known varying-`G` control;
2. bounded cosmic transition, a known-family extension;
3. local maturity/settling, the sole age/history-adjacent niche;
4. oscillatory cosmic history, a potentially distinct synthesis with known scalar-field precedents.

The four niches began with 262,144 cells total. Present-day `dot G/G`, lunar/pulsar-scale normalization, recombination, BBN, positivity, and historical-domain checks were applied without a galaxy velocity. They left 83,577 admissible cells:

- smooth power: 16,824;
- bounded transition: 21,822;
- local maturity/settling: 25,748;
- oscillatory history: 19,183.

Unequal surviving counts are consequences of the same independent physical gates, not unequal initial search opportunity. All four niches recovered their own frozen synthetic injections in all five folds.

## Real-data boundary

The test used the published KMOS3D baryonic Tully-Fisher table from Übler et al. (2017), with spectroscopic redshift, stellar mass, baryonic mass, and intrinsic dispersion frozen before response access. It then queried maximum pressure-support-corrected circular velocity separately for each allowed exploration identity.

- predictor-valid objects: 135;
- exploration objects: 110, with 22 in each of five folds;
- sealed confirmations: 25;
- exploration objects passing the frozen response-quality rules: 110;
- confirmation velocities queried: zero.

The target is a high-redshift baryonic Tully-Fisher residual, not a direct measurement of Newton's constant. Size evolution, gas scaling, stellar-population mass calibration, pressure support, galaxy selection, and ordinary structural evolution remain major alternative explanations.

## Selected relation

All five folds selected admissible cell `68147`, from the oscillatory cosmic-history niche:

```text
mu_G(z) = G_eff(z)/G0
        = exp[0.05 * z/(z + 0.03) * sin(4 * ln(1 + z))]

delta log10(Vcirc) = 0.5 * log10(mu_G)
```

The unused parameters stored with the randomized cell do not enter this niche's formula. Representative values are:

| Epoch | `mu_G` |
|---|---:|
| `z=0` | 1.0000 |
| `z=0.6` | 1.0464 |
| `z=0.9` | 1.0266 |
| `z=1.5` | 0.9758 |
| `z=2.3` | 0.9519 |
| `z=2.6` | 0.9557 |
| recombination, `z=1090` | 1.0146 |
| BBN proxy, `z=10^9` | 1.0479 |

Its present derivative is `|dot G/G| = 4.77e-16 yr^-1` under the frozen numerical definition. Passing these scalar cutoffs is not a full CMB, BBN, stellar-evolution, equivalence-principle, or covariant-theory calculation. The BBN value is close to the admitted upper edge.

## Held-out results

| Comparison | OOF MSE | Candidate improvement |
|---|---:|---:|
| selected oscillatory cell | 0.0079976 | — |
| calibrated baryonic relation | 0.0081903 | **2.35%** |
| flexible ordinary evolution model | 0.0095066 | **15.87%** |

The flexible model was itself worse than the simpler calibrated baryonic relation, so the 15.87% comparison does not establish superiority over the strongest ordinary explanation. The primary improvement was only 2.35%, below the frozen 5% gate.

The selected cell improved over the calibrated baryonic relation in every broad half:

- low redshift: 2.68%;
- high redshift: 2.06%;
- low baryonic mass: 1.12%;
- high baryonic mass: 3.84%.

It was worse than the flexible model for 53 of 110 individual galaxies. A complete selection-aware replay under 99 redshift-stratified residual permutations gave `p=0.08`; the frozen gate was `p<=0.05`. The largest null improvement, 2.93%, exceeded the observed 2.35%.

## Interpretation under the two-track policy

### Universal-gravity track

Rejected for this scope. The relation missed the minimum improvement and selection-aware significance gates. This dataset cannot turn a Tully-Fisher residual into a measurement of universal `G(z)`, and the formula has no covariant completion here.

### Phenomenon/publication track

Not promoted yet. Exact five-fold stability and improvement in every broad half make this a **positive replication hypothesis**, but `p=0.08` does not meet the preregistered paper-lead threshold. The finding may become scientifically useful without solving gravity if the unchanged equation predicts an independent, cross-source velocity or lensing observable. Only that fresh replication can promote it to a paper-track lead.

Age/history was treated equally: the local maturity/settling niche had the largest post-physics candidate count and passed its injection control, but the real data selected another branch. This is evidence from this test, not a reduction in age's starting viability in future distinct representations.

## Reproducibility and cost

- GPU: NVIDIA GeForce RTX 5090 through CuPy;
- admissible candidates: 83,577;
- training residual evaluations: 3,861,268,950;
- full selection-aware null trials: 99;
- full synthetic family searches: four;
- full constant-`G` control searches: one;
- measured search wall time: 47.70 seconds;
- CPU/GPU maximum checked difference: zero;
- paid model calls: zero;
- paid API spend: `$0.00`.

The immutable machine receipt is `runs/gravity/roadmap/item-25-time-varying-g-v1.json`; all exact source, role, candidate, response, and compute receipts are under `runs/gravity/roadmap/item-25-time-varying-g-v1-source/`.

## Exact next actions

1. Preserve the exact equation without retuning and preregister a cross-source, preferably resolved-kinematics replication. Treat it as a phenomenon test, not a changing-gravity claim.
2. If that replication passes, test the unchanged relation on a genuinely independent light-propagation or lensing observable and run a dedicated prior-art/equivalence audit before any novelty language.
3. Continue the equal-priority numbered gravity roadmap with Item 26, retarded gravity. Do not open the 25 Item 25 confirmation velocities without separate authorization.
