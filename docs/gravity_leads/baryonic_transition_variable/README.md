# Baryonic transition variable

## Status

Item 58 found an interesting cluster meta-law signal but did not pass its full gate.
This is the most relevant clue for turning the Item 59 cluster success and galaxy
failure into one conditional law.

Primary record:

- [Item 58 cluster-coefficient result](../../GRAVITY_ITEM58_CLUSTER_COEFFICIENT_GATE_RESULT.md)

## Core idea

Different objects may not require unrelated formulas. A single law may have a response
strength generated from measurable baryonic state:

    object properties -> dimensionless transition -> effective response strength

The allowed predictors described baryonic radial-profile shape, acceleration span,
inner/outer mass balance, projected compactness, X-ray concentration and asymmetry,
and predeclared boundary/shape interactions. Galaxy or cluster identity was forbidden.

## What was learned

- Twenty exposed CLASH clusters and 84 radial points were evaluated with five
  whole-cluster outer folds and fully nested model selection.
- Baryonic profile plus compactness/boundary features reduced coefficient MSE by
  35.49% relative to a training-only constant.
- The held-out R2 was 0.1893 and the full 1,999-label-shuffle result was p = 0.007.
- Removing any one cluster and trimming both tails preserved the nominal improvement.
- Compactness-boundary synthesis was the strongest individual family and was selected
  in three of five folds.
- X-ray morphology alone was not predictive; most signal came from baryonic radial
  structure and its declared interactions with morphology.
- Seven nominal cluster counterexamples were retained; only A611 and MACS1206 remained
  counterexamples under both one-sigma target shifts.
- A fixed beta near two still fit the underlying acceleration diagnostic better.
- The result reversed on the upper one-sigma target envelope.
- The coefficient target came from a spherical, NFW-based CLASH diagnostic rather than
  direct image, shear, magnification, or time-delay likelihoods.

## Relationship to known work

The general notion of environmental or state-dependent activation overlaps screening
mechanisms, MOND external-field effects, refracted gravity, and baryon/halo assembly
relations. The particular compactness-boundary feature synthesis is not established as
historically new and is not causal.

Relevant context:

- Refracted gravity: https://arxiv.org/abs/1603.04943
- Vainshtein observational test: https://arxiv.org/abs/1201.1508
- X-COP clusters in MOND: https://arxiv.org/abs/2405.08557

## First-principles gap

This is a regression over state descriptors, not a dynamical field or action. A
fundamental version must explain why its transition variable is dimensionless and
universal, why it multiplies a particular action term, and how it evolves causally.

## Suggested next steps

1. Build dimensionless candidates from acceleration, potential compactness, boundary
   slope/contrast, pressure-to-rest-energy ratio, and nonlocal occupancy.
2. Require two separate factors: a high-acceleration GR screen and an environmental
   activation factor.
3. Freeze thresholds and exponents on predictor-only data.
4. Test on galaxy groups, which lie between disks and clusters.
5. Reject any transition that merely reconstructs an object label.
6. Propagate complete measurement covariance and baryonic-mass uncertainty.
7. Promote only a variable that predicts direct observables under unchanged parameters.

