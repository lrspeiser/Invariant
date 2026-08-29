# Item 58 cluster-coefficient gate result

## Decision

`ITEM58_CLUSTER_COEFFICIENT_GATE_NOT_PASSED_VARIABLE_FAMILIES_RETAINED`

Measured baryonic profile geometry and X-ray morphology contain an interesting held-out
signal about the cluster-scale `beta_eff` diagnostic, but the frozen full gate did not
pass. The nested predictor reduced coefficient mean squared error by 35.49% relative to a
training-only constant, obtained held-out `R2=0.1893`, and exceeded the full nested result
in only 13 of 1,999 shuffled-label trials (`p=0.007`).

That is evidence worth following up, not a gravity result. Fixed `beta=2` still fit the
underlying 84 acceleration points better, and the predictor's advantage reversed when all
target accelerations were shifted to the upper one-sigma envelope. No feature family,
formula family, cluster, or physical mechanism is pruned.

## Question and blind boundary

Item 58 asked whether the different best-fit cluster coefficients of the unchanged
cross-scale diagnostic could be predicted from measured baryonic geometry or state:

```text
g = g_bar + beta_eff * (g_bar*chi + g_dagger*psi)
```

The design was frozen in commit `aa84165f` before evaluation. It used all 20 already
exposed CLASH clusters and 84 radial points, but opened no new response rows and performed
no direct lensing likelihood evaluation. The target was the Item 1 coefficient oracle,
which was replayed from the already exposed spherical NFW-lensing posterior acceleration
diagnostic.

The model builder could use only:

- baryonic acceleration-profile slopes, local-dimension summaries, acceleration and
  radius spans, and an outer-to-inner equivalent baryonic-mass ratio;
- projected X-ray axis ratio, concentration, centroid shift, and third-power ratio;
- four declared products combining compactness, boundary, and profile features.

Cluster identity, `g_tot`, lensing or inferred total mass, temperature, hydrostatic mass,
and the per-cluster coefficient were forbidden inputs. A target-free feature table was
written before response evaluation. Five whole-cluster outer folds and four inner folds
selected among five ridge-model families and five penalties without seeing each held-out
cluster.

## Numerical result

| Coefficient predictor | OOF MSE | OOF MAE | OOF R2 |
|---|---:|---:|---:|
| Nested frozen selector | 0.13561 | 0.30556 | 0.18930 |
| Training-only constant | 0.21021 | 0.36600 | -0.25666 |
| Fixed `beta=2` | 0.16740 | 0.31900 | -0.00072 |

The relative MSE improvement over the out-of-fold constant was 35.49%, above the frozen
10% floor. Removing any one cluster or trimming the two largest advantages and two largest
disadvantages did not reverse the aggregate improvement.

The same coefficient predictions were also mapped back to the radial acceleration data:

| Predictor | Chi-square over 84 points | Chi-square per point |
|---|---:|---:|
| Nested frozen selector | 149.887 | 1.784 |
| Training-only constant | 156.105 | 1.858 |
| Fixed `beta=2` | 144.151 | 1.716 |

Thus the learned variation beat the fold-specific constant but did not beat the stronger
fixed-two comparator on the underlying observable diagnostic.

The full 1,999-trial nested permutation control gave:

- observed coefficient-MSE improvement: 35.49%;
- null mean improvement: -34.74%;
- null 95th percentile: 9.02%;
- exact finite-trial `p=0.007`.

This says the nominal association is unlikely to be a routine outcome of this frozen
small-sample search procedure. It does not establish that the predictors cause the
coefficient shift.

## Which feature combinations carried the signal

The compactness-boundary synthesis was the strongest single predeclared family, with
`MSE=0.12508` and `R2=0.25228`. It was selected in three of five outer folds. The simpler
baryonic-profile family was selected in the other two folds. All selected ridge penalties
were 10.0.

The X-ray-only family was not predictive (`R2=-0.27158`). This makes the result more
specific than a generic cluster-disturbance correlation: most of the nominal signal came
from baryonic profile structure and its predeclared interactions with morphology. It is
still only a model-development clue because the target itself is model-dependent.

## Uncertainty and counterexamples

The lower and upper one-sigma target envelopes gave opposite conclusions:

| Target sensitivity | Improvement over OOF constant |
|---|---:|
| `log(g_tot) - sigma` | +40.13% |
| `log(g_tot) + sigma` | -25.94% |

Seven clusters had a larger nominal coefficient error under the nested predictor than the
constant: A209, A611, MACS0416, MACS1206, MACS1931, RXJ1347, and RXJ2129. Only A611 and
MACS1206 remained counterexamples under both one-sigma envelope shifts.

Those two repeated mismatches still do not eliminate the formula. The global empirical
policy classifies the result as `QUALITY_LIMITED_EVIDENCE_RETAINED`: the response is a
model-dependent diagnostic, covariance is unavailable, and no unchanged independent
replication has yet been run. A single counterexample is never terminal, counterexample
count alone is never terminal, and finite samples cannot prune a family.

## Lay interpretation

In plain terms, the clusters do not all appear to want exactly the same strength for this
particular extra-gravity-shaped term. Some of that variation can be guessed in advance
from how the ordinary matter is distributed and, to a lesser extent, from the X-ray shape.
The guess survived held-out clusters and an unusually strict random-label test.

But the practical improvement is not yet robust to plausible movement of the inferred
target, and simply using a coefficient near two still describes the radial acceleration
points better. The result is therefore a promising variable-discovery lead, not evidence
that dark matter is unnecessary or that a new gravity law has been found.

## Next test

Item 59 should move from the derived CLASH coefficient target to X-COP forward observables.
The baryonic compactness, boundary, and profile families should remain intact and be
tested unchanged or through predeclared descendants against held-out X-ray and SZ radii
and held-out clusters. Pressure, member-baryon, calibration, and covariance nuisances must
be explicit. Direct CLASH lensing remains reserved for Item 60.

## Reproduction

```powershell
python -m sigma_theory_compiler.gravity_item58_cluster_coefficient_gate replay
python -m pytest tests/test_gravity_item58_cluster_coefficient_gate.py -q
```

The replay regenerates and byte-compares the preflight manifest, target-free feature
table, evaluation, and aggregate result. Paid model calls: zero. GPU use: none.
