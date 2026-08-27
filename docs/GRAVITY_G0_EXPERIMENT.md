# G0 SPARC experiment freeze

**Decision:** `PASS_G0_EXPERIMENT_FROZEN`

**Receipt:** `runs/gravity/g0-experiment/receipt-v1.json`

**Contract:** `configs/gravity_g0_experiment.json`

G0 freezes the experiment that decides which local, baryon-input formulas may enter the G1
galaxy atlas. It is a mechanics and calibration result, not a discovered gravity law.

## What was evaluated

- The inherited whole-galaxy split contains 140 exploration and 35 confirmation galaxies.
  One exploration object fails the preregistered baryonic admission rule, leaving 139 galaxies
  and 2,720 radius-ordered rows for G0/G1.
- Every galaxy uses up to five contiguous radial holdouts. Every row is held out exactly once.
- No confirmation galaxy is passed to a baseline or candidate evaluator. The full published
  asset is nevertheless present on this machine, so this is a code-enforced holdout—not a
  claim that its bytes are cryptographically inaccessible.
- The baryonic model is frozen at disk mass-to-light ratio 0.5 and bulge ratio 0.7. No
  per-galaxy baryonic recalibration is allowed.
- The score is conditional on SPARC's published random velocity uncertainty. It is not a full
  likelihood because the source does not provide systematic inclination covariance here.

## Baseline controls

| Comparator | Held-out chi-square | Two-sigma coverage | Role |
|---|---:|---:|---|
| Newtonian baryons | 1,697,326.40 | 13.20% | Required baryon-only baseline |
| Empirical RAR | 130,714.69 | 40.74% | Frozen empirical comparator |
| Wrong high-acceleration boost | 1,622,323.19 | 11.73% | Deliberately false control |
| NFW-shaped halo | 28,018.81 | 68.27% | Training-only flexible ceiling |

The expected orderings all passed. NFW parameters are fitted only on each fold's training
radii and never become a formula input, target, feature, rescue value, or statement of truth.

## Real evaluator benchmark

| Measurement | Result |
|---|---:|
| Device | NVIDIA GeForce RTX 5090 |
| Canonical candidates | 1,000,000 |
| SPARC rows per candidate | 2,720 |
| Candidate-point evaluations | 2,720,000,000 |
| GPU elapsed wall time | 0.957 s |
| Candidates per second | 1,044,840 |
| Candidate-point evaluations per second | 2.842 billion |
| GPU memory-pool peak increment | 606,820,352 bytes |
| Domain-finite candidate fraction | 0.1053% |
| FP64 CPU replay rate | 4,790 candidates/s |
| FP64 CPU/GPU mismatches | 0 / 4,096 |

The rate applies to this pointwise rational interpolation grammar. It cannot be silently
extrapolated to nonlocal kernels, nuisance integration, covariant forward models, or other
grammars; each requires its own measured benchmark.

## What G0 authorizes

G1 may now run the frozen three-arm pilot: structured/Occam, deterministic pseudorandom, and
creativity-guided proposals. Each arm receives at least 10 million distinct canonical
candidates for each of 12 preregistered pilot galaxies. LLM origin labels do not prune ideas;
only typed invalidity, frozen empirical failure, proven equivalence, or exhaustion of a
declared finite cell can close a branch.

G1 has not passed until all 139 admitted exploration galaxies have at least one formula that
clears every radial fold. A compute ceiling without full coverage is a recorded failure, not
permission to proceed to G2.
