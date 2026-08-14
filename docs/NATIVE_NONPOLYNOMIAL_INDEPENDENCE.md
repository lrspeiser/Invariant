# Native non-polynomial generator independence

This lane is a bounded prospective test of whether formula recovery depends on one generic
solver or one construction family. It deliberately does not call Formula Discovery Job v1/v2,
a generic exact linear solver, SymPy, or a Bayesian generator.

## Frozen tournament

Before any target record is opened, the checked config freezes:

- two public, genuinely non-polynomial structured worlds;
- three generator family identities and their algorithm descriptors;
- every public exact rational constraint;
- a seed and hard work budget for every family;
- each hidden target commitment; and
- the requirement that generation and tuning stop after target unsealing.

The worlds are a linear-fractional rational function and a shifted exponential. The native
families are:

1. **Reciprocal elimination**, which derives the three linear-fractional parameters from three
   exact public values by a closed algebraic elimination.
2. **Difference ratio**, which derives the base, scale, and offset of a shifted exponential from
   consecutive first differences.
3. **Bounded structural enumeration**, which independently searches a seed-ordered finite
   integer parameter space and stops at a hard work cap.

The first two families emit a public constant baseline outside their declared class. That makes
failure observable rather than silently skipping an incompatible world.

## Chronology and evidence

All six generator outcomes are produced from public rows and sealed in the Phase A receipt. Each
candidate records empty target reads, zero Formula Discovery Job delegations, and zero generic
exact-linear-solver and Bayesian-generator calls. An instrumented pre-unseal probe is denied with
zero target bytes exposed and bound into Phase A. Only then is the two-record target fixture read
once and opened against the preregistered commitments.

Post-unseal evaluation is exact. Matching structured normal forms receive parameter-equality
identity certificates with zero rational residuals. Nonmatching candidates receive a concrete
hidden-holdout point, exact candidate and target values, and an exact nonzero residual. No
candidate is generated or tuned after the unseal.

The leave-one-family-out table removes each family in turn. A tournament PASS requires every
declared benchmark class to retain at least one exact PASS for every removal, and every world to
have been independently recovered by at least two families before removal.

## Reproduce and validate

From the repository root:

```text
python -m sigma_theory_compiler.native_nonpolynomial_independence_tournament
python -m sigma_theory_compiler.native_nonpolynomial_independence_tournament --validate-checked
```

The output is an immutable, canonical JSON receipt at
`runs/math/native-nonpolynomial-independence/receipt.json`. Validation recomputes the complete
tournament from the live bound config, target fixture, implementation, tests, and this document;
a resealed candidate, certificate, counterexample, chronology, claim, or ablation tamper fails
exact replay.

## Claim boundary

A PASS shows bounded prospective construction-family independence on these two synthetic
classes. It does not show universal formula discovery, scientific truth, novelty, or performance
outside the frozen parameter ranges. Adding trigonometric, implicit, multivariate, noisy, or
real-world observational classes remains separate breadth work.
