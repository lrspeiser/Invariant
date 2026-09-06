# Sparse empirical formula search, v1

Frozen before implementation and before this search reads response values.
Use the exact published 126-row pattern-learning sample, retaining all rows.
This is a bounded adaptive search over 30 declared expressions, not RL or
a new gravity operator. No failed or unvalidated full-field prediction enters.
Config specifies all features, complexity limits, penalties and folds.

Fit preprocessing, candidate selection and coefficients inside each training
partition. Select path length and ridge penalty with inner validation only.
Report every outer result including zero-term choices and negative gains.
Compare against independently tuned baseline, with paired galaxy bootstrap
conditional on fitted predictions; repeat the entire search on eight joint
structure shuffles. These are historically exposed development observations.

Before opening the sample: test held-out-label perturbation invariance,
training-only transforms, independent sklearn ridge prediction, planted
nonlinear recovery, CPU/GPU agreement, constant-feature behavior and malformed
inputs. GPU pool is capped at 1 GiB, CPU numerical threads at one.
Preserve exact code/config/input hashes and all formulas with transformation
constants so predictions can be replayed independently. A successful run
does not admit a gravity law or establish a causal correlation.
