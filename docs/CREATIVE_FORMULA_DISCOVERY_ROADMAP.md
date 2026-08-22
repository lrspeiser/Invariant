# Creative formula and proof discovery roadmap

## Current measured boundary

- Two externally sourced known-formula controls are rediscovered with zero sealed-holdout loss.
- Two externally sourced bounded-unknown sequence controls do not have a zero-loss candidate.
- Claude completed four blind proposer calls and four post-proposal critic calls. Its output is never
  verifier authority.
- Every declared creativity family has an equal-budget random control and a leave-one-family-out
  ablation.
- Exact arithmetic, CAS, SMT, and interval checks pass for the known controls. The formula-specific
  Lean source is checked by CI, not inferred locally.
- Two evaluator implementations agree on one machine. A second physical host remains mandatory.
- No novel formula, proof, theorem, scientific law, or open-problem result is claimed.
- The famous-open-problem gate is closed: zero of three required independent level-5 successes.

## Definition of a creative success

A run is not a success merely because it generated many candidates or found a low training error.
For a bounded level-5 benchmark it must have all of the following:

1. zero exact training loss;
2. zero exact sealed-holdout loss;
3. at least half of the creativity families beating their budget-matched random controls;
4. at least three distinct behavior signatures;
5. at least two distinct proof-mechanism signatures;
6. byte-agreement between independent exact evaluators;
7. no target or source read before the proposal root is sealed.

This is evidence of bounded rediscovery, not evidence of mathematical novelty.

## Work queue

### 1. Make the typed language genuinely broader

- Add first-class expression types for rational functions, piecewise laws, recurrences, generating
  functions, finite sums/products, modular arithmetic, graphs, tensors, differential operators,
  variational functionals, and asymptotic expansions.
- Carry domains, units, dimensions, tensor ranks, index symmetries, derivative order, singular sets,
  boundary conditions, and admissible rewrites in the type checker.
- Separate representation from meaning so algebraically equivalent polynomial, recurrence,
  combinatorial, and generating-function forms can share a behavior object while retaining distinct
  proof mechanisms.
- Admit a new type only with positive controls, mutation controls, resource limits, a serializer,
  and at least two evaluators.

### 2. Search ideas and proof plans as separate spaces

- Search formula candidates and proof plans independently, then join them by applicability.
- Let proof plans include induction, invariant preservation, descent, extremal arguments,
  contradiction, bijection, generating-function coefficient extraction, dimensional reduction,
  variational derivation, CAS normalization, SMT countermodel search, interval enclosure, and Lean.
- Rank plans by falsification power, premise count, proof debt, and estimated cost—not by persuasive
  prose.
- Preserve failed plan/candidate pairs so counterexamples steer later proposals without changing the
  verifier contract.

### 3. Expand externally authored sealed benchmarks

- Add benchmark packs maintained outside the generator repository and signed by a distinct
  principal.
- Include known-answer calibration problems, formula-free next-value holdouts, proof-repair tasks,
  cross-representation tasks, noisy scientific datasets, and deliberately unidentifiable datasets.
- Freeze public training material and a target commitment; reveal holdout rows and references only
  after proposal and critique roots are sealed.
- Rotate benchmark packs and publish failures to reduce adaptation to a static test set.

### 4. Keep Claude useful but non-authoritative

- Use structured proposer calls for analogies, invariants, representation changes, candidate
  expressions, falsifiers, and proof plans.
- Use structured critic calls only after train evidence is frozen, supplying exact residuals and
  candidate IDs so steering is auditable.
- Quarantine unsupported syntax and record proposed/admitted/non-executable counts separately.
- Enforce per-call, per-campaign, and per-open-problem call/token ceilings before making a request.
- Never let a model response pass an exact, formal, novelty, scientific, or release gate.

### 5. Strengthen controls and ablations

- Match random controls on candidate count, grammar depth, coefficient range, runtime, and verifier
  budget—not candidate count alone.
- Add shuffled-label, corrupted-unit, target-leak, wrong-recurrence, wrong-boundary, and false-proof
  controls.
- Report family leave-one-out, proof-route leave-one-out, Claude-off, counterexample-off, and
  representation-normalization-off ablations.
- Require confidence intervals across benchmark resamples before concluding that a family adds
  value.

### 6. Build a real dataset-to-invariant pipeline

- Normalize declared units and propagate unit uncertainty.
- Compute dimension matrices, dimensionless null-space groups, invariant coordinates, residual
  channels, finite-difference channels, and out-of-distribution splits.
- Accept causal language only when intervention data or a defensible identification design exists;
  observational rows must never be relabelled as interventions.
- Add symmetry-group actions and equivariance tests rather than relying only on coordinate names.
- Model measurement noise, missingness, censoring, dependence, and dataset shift before fitting.
- Include synthetic identifiability controls where the correct result is “underdetermined.”

### 7. Layer independent verification

- Stage 1: exact rational/integer replay and explicit counterexamples.
- Stage 2: CAS canonical identity or residual nonzero witness.
- Stage 3: SMT proof or countermodel inside the declared theory.
- Stage 4: interval enclosures over the declared continuous domain.
- Stage 5: formula-specific Lean theorem with a closed premise manifest and negative mutation.
- Block serious claims unless every required stage passes; “backend unavailable” is a block, not a
  pass.

### 8. Make prior-art screening claim-specific

- Search repository theorems, Lean/mathlib names, OEIS, zbMATH, MathSciNet where licensed,
  Crossref, arXiv, Semantic Scholar, and domain-specific literature indexes.
- Search equivalent notation, recurrence/closed-form duals, transforms, specializations, and proof
  mechanisms—not only literal formula text.
- Bind queries, result identifiers, dates, and source hashes into the receipt.
- Require a named human reviewer to inspect the nearest matches before any novelty language is
  released.
- Treat “no automated match” as “not cleared,” never as “novel.”

### 9. Reproduce independently

- Run at least two implementations with no shared symbolic evaluation path.
- Run on at least two physical hosts and preferably two operating systems.
- Compare canonical predictions, counterexamples, proof objects, verifier versions, dependency
  closures, and receipt hashes.
- Add mutation tests that must fail on every host.
- Promote host evidence only after the CI artifacts are downloaded, hash-checked, and bound into a
  new receipt; a workflow definition alone is not reproduction evidence.

### 10. Measure two kinds of novelty

- Behavior novelty: predictions, residual topology, invariance/equivariance behavior, singularity
  structure, asymptotics, and counterexample sets.
- Proof-mechanism novelty: premise graph, lemma decomposition, invariant used, induction variable,
  transform, and verifier route.
- Deduplicate algebraic restatements before counting either kind.
- Do not infer literature novelty from internal diversity.

### 11. Escalate to famous open problems only after calibration

- Require at least three independent level-5 successes under the definition above.
- Preregister the exact problem statement, allowed premises, grammar, verifier stack, and resource
  budget.
- Limit each problem to four Claude calls and 32,000 total tokens unless a new protocol is reviewed.
- Publish a sealed failure receipt whether the run finds nothing, times out, or produces an invalid
  proof.
- Treat every candidate as unproved until independently reproduced, prior-art reviewed, and
  kernel-checked.

## Next executable milestones

1. Let CI compile `ExternalKnownFormulaControls.lean` and reproduce the deterministic receipt on
   Windows and Linux.
2. Download and bind those host artifacts into a two-machine reproduction receipt.
3. Add a fifth external benchmark pack with a new typed representation, not another polynomial.
4. Improve matched controls to equalize grammar complexity and wall-clock budget.
5. Add one externally supplied intervention dataset and one deliberately non-identifiable dataset.
6. Continue bounded level-5 runs until three genuine successes are recorded; do not spend the open-
   problem budget before then.
