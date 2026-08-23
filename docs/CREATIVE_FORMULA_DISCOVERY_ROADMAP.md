# Creative formula and proof discovery roadmap

## Current measured boundary

- Two externally sourced known-formula controls are rediscovered with zero sealed-holdout loss.
- Two externally sourced bounded-unknown sequence controls do not have a zero-loss candidate.
- The core app loaded the machine credential without persisting it and completed four blind Claude
  proposer calls plus four post-proposal critic calls on `claude-opus-4-6`.
- Those calls produced 21 schema-admitted ideas: 13 model-self-assessed known rewrites and 8
  uncertain. All 21 remain active; none were discarded for their label or verifier status.
- Independent expansion produced 126 proof-plan branches and 64 cross-idea recombination branches.
  This is one treatment specimen, not evidence that the system is more creative.
- Every declared creativity family has an equal-budget random control and a leave-one-family-out
  ablation.
- Exact arithmetic, CAS, SMT, and interval checks pass for the known controls. The formula-specific
  Lean source is checked by CI, not inferred locally.
- Downloaded artifacts bind four distinct GitHub-hosted ephemeral VM runner IDs across Windows and
  Linux, with two evaluator implementations per host and a separate Lean kernel artifact. This is
  multi-host VM reproduction, not a claim about distinct bare-metal machines.
- No novel formula, proof, theorem, scientific law, or open-problem result is claimed.
- The famous-open-problem gate is closed: zero of three required independent level-5 successes.
- A live rotating pack now binds 24 distinct OEIS b-file response hashes, with four tasks each for
  recurrences, generating functions, finite sums, finite products, modular objects, and
  representation bridges. Generation packets omit source IDs and eight-term holdouts. The external
  HTTPS origins are hash-bound but not cryptographically signed, so the pack is usable for the
  creativity tournament and explicitly ineligible for a level-5 count.
- A live claim-specific screen queried OEIS, Crossref, arXiv, Semantic Scholar, and the repository
  for one retained `uncertain` LLM idea. OEIS matched all 18 displayed terms to A005132; Semantic
  Scholar's HTTP 429 is preserved as unavailable evidence. This calibrates the model label at the
  behavior level only. Formula/construction and proof-mechanism provenance remain blocked on a
  named human review.

## Definition of creative progress

Creativity is measured before mathematical success. The primary experimental measure is blinded,
human-reviewed, useful distinct behavior branches per 10,000 tokens under matched resources. A
branch is useful when independent reviewers rate its coherence, nontriviality, and follow-up value;
it need not already pass a theorem gate. We also count cross-domain recombinations, representation
coverage, distinct proof mechanisms, and initially failed or blocked ideas later used as productive
parents. Raw idea volume and the model's own novelty label do not count as wins.

The preregistered paired protocol is
[`configs/creativity_ablation_protocol.json`](../configs/creativity_ablation_protocol.json). It
requires at least 24 rotating paired tasks, frozen baseline and treatment commits, blinded output
order, two named reviewers, identical model/token/time/grammar/verifier budgets, at least a 20%
primary improvement, a one-sided sign-test threshold of 0.05, and typed-usability noninferiority.
Component knockouts isolate lineage labels, non-pruning, expanded grammar, and independent
proof/recombination search.

## Definition of a bounded mathematical success

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

### 4. Use Claude throughout creativity while keeping it non-authoritative

- Make structured roles available for proposing, analogy scouting, representation invention,
  dataset explanation, proof strategy, recombination, and critique.
- Use structured critic calls only after train evidence is frozen, supplying exact residuals and
  candidate IDs so steering is auditable.
- Preserve unsupported but well-typed suggestions in the lineage archive and branch toward new
  executors; inability to execute today changes claim readiness, not retention.
- Require every suggestion to self-label as a likely known rewrite, cross-domain synthesis,
  proposed new construction, or uncertain, with named analogues and source domains. Treat that as
  fallible lineage metadata, never a novelty judgment.
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

1. Run the 24-task paired old-vs-new creativity tournament and its four component knockouts.
2. Obtain two named blinded reviewers and record disagreements before unblinding system identity.
3. Extend the live rotating external pack from recurrence, generating-function, sum/product,
   modular, and representation-bridge tasks to separately maintained tensor and variational packs;
   obtain a detached signature from a distinct pack principal before level-5 use.
4. Replace synthetic intervention/noise/shift/identifiability calibrations with externally supplied
   datasets while retaining the synthetic negative controls.
5. Complete named human review of the bound A005132 behavior match and the nearest formula and
   proof-mechanism results; then run the same automated screen for every tournament survivor.
6. Continue bounded level-5 runs until three independent successes exist; the open-problem gate
   remains closed meanwhile, but creative branches continue accumulating.
