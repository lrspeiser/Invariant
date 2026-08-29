# Gravity roadmap Item 55: causal-variable diagnostics result

## Outcome

**CAUSALITY NOT ESTABLISHED; PREDICTIVE SIGNAL AND PROXY RISK BOTH RETAINED.**

The Item 45 geometry-density interaction carries real predictive information beyond a
simple galaxy-versus-cluster label, and its prediction relies on both geometry and density.
However, the present galaxy and cluster populations are too separable in feature space to
show that the interaction is a universal cause rather than a more sophisticated dataset
proxy.

No formula or variable is pruned by this result.

## What was tested

Using the same 28 S4TM exploration galaxy lenses and 20 CLASH clusters, Invariant ran:

- a cross-fitted population-label-only response control;
- within-population median ablations of each of the six Item 45 primitive axes;
- leave-one-object-out galaxy/cluster classification from those axes;
- cross-population common-support matching in standardized six-dimensional space; and
- two out-of-distribution searches: train on galaxies and test clusters, then reverse.

These are observational diagnostics. None is a physical intervention or fresh
confirmation.

## Label control and ablations

| Predictor | Balanced loss |
|---|---:|
| Item 45 interaction | **0.76148** |
| Population label only | 2.29160 |
| Baryonic Newton | 67.65046 |

Item 45 improved on the label-only control by **66.77%**, with paired sign-flip
`p = 0.00025`. That is evidence that the formula uses more than just the words “galaxy”
and “cluster.”

Replacing one feature at a time by its training-fold median within each population changed
Item 45's loss by:

| Ablated axis | Relative loss change |
|---|---:|
| Geometry | **+38.11%** |
| Density | **+11.71%** |
| Gradient | 0.00% |
| Time | 0.00% |
| Environment | 0.00% |
| Field | 0.00% |

This is consistent with the selected recipe, which directly uses geometry and density.
It demonstrates predictive reliance, not causal effect: median replacement can create
unphysical combinations and does not manipulate nature.

## Proxy and common-support audit

At the object level, the galaxy and cluster ranges had these interval-overlap fractions:

| Axis | Cross-population overlap |
|---|---:|
| Geometry | 1.000 |
| Density | **0.000** |
| Gradient | **0.000** |
| Time | 0.435 |
| Environment | **0.000** |
| Field | 0.917 |

A leave-one-object-out nearest-centroid classifier identified galaxy versus cluster from
the six primitives with **100% accuracy**. No S4TM object had a nearest CLASH object within
the preregistered standardized-distance caliper of 1.0.

In lay terms, the datasets occupy visibly different neighborhoods. Even though geometry
overlaps, density—the other winning ingredient—perfectly separates the observed ranges.
That makes it impossible here to cleanly compare a galaxy and cluster that have like-for-like
geometry and density while differing only in the outcome.

## Cross-population transfer

| Training population | Test population | Test loss | Newton loss | Improvement vs Newton |
|---|---|---:|---:|---:|
| S4TM | CLASH | 18.7094 | 134.3901 | 86.08% |
| CLASH | S4TM | 0.18142 | 0.91086 | 80.08% |

Both one-population searches transfer better than baryonic Newton, which is encouraging.
But the galaxy-trained cluster loss of 18.71 is far worse than Item 45's joint cluster loss
of 1.335. Beating a very weak Newton-only baseline is not evidence of universal transfer.

## Interpretation

The honest conclusion is a split verdict:

- **Still interesting:** geometry and density carry within-population predictive signal,
  outperform a label-only shortcut, and transfer in the right broad direction.
- **Not established:** the data do not separate causal physics from population proxying,
  because density and several other constructed inputs have little or no galaxy/cluster
  common support.

The required next evidence is an intermediate-scale dataset—groups, massive ellipticals,
low-mass clusters, or galaxies spanning cluster-like surface acceleration—with overlapping
geometry and density. Predictions must be frozen before opening those outcomes. Natural
experiments or simulations that perturb baryonic geometry at fixed mass would also be much
more causally informative.

No alternative to general relativity, removal of dark matter, or historical novelty is
claimed.

## Reproduction

- `runs/gravity/roadmap/item-55-causal-variable-tests-v1.json`
- `runs/gravity/roadmap/item-55-causal-variable-tests-v1-source/causal-diagnostic-result.json`

```powershell
python -m sigma_theory_compiler.gravity_item55_causal_variable_tests replay
```

The next task is Item 56: a radial disk-galaxy gate using held-out SPARC exploration
measurements, where the system must predict rotation curves from baryons and universal
fields rather than infer one coefficient per object.
