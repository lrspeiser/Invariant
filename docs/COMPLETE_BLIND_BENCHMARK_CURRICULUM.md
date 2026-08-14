# Complete blind benchmark curriculum

This receipt is the current authority for the 200-slot blind benchmark curriculum. It explicitly
supersedes the earlier 5/200 readiness receipt, which remains preserved as historical evidence but
is no longer current authority.

## What is complete

The closed registry is the exact Cartesian product of two cohorts, five domains, two artifact
types, five levels, and two ordinals: 200 unique slots. The result records 200 registered, 200
ready, zero missing, and zero invalid. Partition metrics independently show complete coverage for
cohort, domain, artifact type, and level.

Every slot has three exact layers:

1. Public integer constraints and a target commitment are frozen.
2. A candidate is generated and self-sealed before the target-rule fixture can be read.
3. After one atomic target unseal, the slot receives either a symbolic coefficient-equality proof
   for all integer inputs or a concrete exact counterexample with a nonzero integer residual.

The 100 theorem-rediscovery slots have exact proofs. The 100 counterexample-construction slots use
generated false-neighbor candidates and have exact counterexamples. Both are readiness successes:
the outcome is scientifically explicit rather than coerced into PASS.

## Blind chronology and thresholds

The target fixture is inaccessible during Phase A; an instrumented probe is denied and exposes
zero bytes. All 200 candidate seals, all public-constraint hashes, the target-batch commitment, and
all eight completion thresholds are bound into the Phase A root. Only then is the target-rule file
read once. Candidate generation and tuning after the unseal are zero.

Thresholds require 200 registered, 200 ready, 100 historical ready, 100 synthetic ready, 100
proofs, 100 counterexamples, zero missing, and zero invalid. The receipt evaluates every frozen
threshold against observed counts and records all eight as passing.

## Historical honesty

Historical targets carry an explicit classical-source title, date, locator, and an adaptation
label. These are generated parameterized benchmark adaptations, not claims that every generated
formula appeared verbatim in the classical source. Historical counterexample candidates are
explicitly labeled `generated_false_neighbor_control_not_historical_conjecture`; they are never
presented as historical conjectures. Synthetic targets are also explicitly labeled synthetic.

## Reproduce

From the repository root:

```text
python -m sigma_theory_compiler.complete_blind_benchmark_curriculum
python -m sigma_theory_compiler.complete_blind_benchmark_curriculum --validate-checked
```

The immutable current receipt is
`runs/math/complete-blind-benchmark-curriculum/readiness.json`. Validation recomputes all 200
candidate, target, proof/counterexample, slot, partition, threshold, supersession, and top-level
seals. A resealed tamper fails exact replay.

## Claim boundary

Completion means the declared bounded curriculum has complete, replayable evidence. The slots use
five exact one-parameter template families. This does not establish general formula discovery,
historical novelty, coverage of all mathematical structures, or scientific truth outside the
frozen curriculum.
