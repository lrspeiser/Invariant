# Counterexample-driven formula search

Invariant should treat negative knowledge as a first-class search asset. Instead of storing only
the best equation or proof found so far, it should accumulate replayable certificates describing
exactly which candidates, finite regions, or proved families cannot satisfy the target.

This does **not** mean that a finite program can usually characterize the literal complement of
all correct equations. The operational objective is narrower and defensible: monotonically grow a
sound, explicitly scoped description of the excluded region, then spend generation and compute on
the remainder. The remainder is *unexplored under current certificates*; it is not automatically
correct, useful, or historically novel.

## Search loop

1. Canonicalize each candidate and collapse exact algebraic or structural equivalences.
2. Query the exclusion ledger before spending expensive verification compute.
3. Apply cheap exact filters: type, units, dimensions, symmetries, domains, boundary values, and
   already proved family predicates.
4. Use CPU/GPU batches to search aggressively for counterexamples, adversarial regimes, holdout
   failures, and dominated parameter regions.
5. Minimize each failure to the smallest witness or assumption set that still fails.
6. Attempt to generalize the witness. An LLM may propose the generalization, but it cannot certify
   or activate it.
7. Admit an exclusion only after its declared scope is verified by exact execution, a formal proof,
   or an independent external evaluator.
8. Recompute the survivor frontier and condition the next diverse generation round on what remains.

The loop is monotone for verified knowledge: adding a certificate may shrink the frontier, but a
heuristic warning never removes a candidate. Retraction is explicit if a verifier or scope was
wrong; silent pruning is forbidden.

## Certificate levels

| Level | What it can exclude | Required evidence |
| --- | --- | --- |
| Exact instance | One canonical candidate | Rejecting verifier receipt bound to its SHA-256 |
| Finite enumerated region | A completely enumerated and canonicalized bounded region | Replayable enumeration plus exact rejection of every member |
| Proved parametric family | Every candidate satisfying an explicit decidable scope predicate | Formal or independently checkable proof of the quantified exclusion |
| Heuristic failure | Nothing; it only prioritizes tests | Observation, analogy, LLM suggestion, or incomplete pattern |

Similarity, embeddings, shared vocabulary, or an LLM's confidence are never sufficient scope
predicates for pruning. Missing features fail open: a candidate survives until its membership in an
excluded family is established.

## Primary knowledge objects

The database should preserve:

- canonical candidate/equivalence-class identity and provenance;
- exclusion certificate, scope predicate, generalization level, and status;
- smallest counterexample or failure witness;
- verifier name, version, code/data hashes, decision, and replay command;
- the proof or enumerated region that justifies any family-level generalization;
- supersession/retraction links when a certificate changes;
- survivor-frontier snapshots and the certificates that shaped each generation prompt;
- non-pruning heuristic warnings separately from verified exclusions;
- prior-art and equivalence results separately from mathematical correctness.

This separation matters. “Cannot satisfy the target,” “is equivalent to a known expression,” and
“has not appeared in the literature” are different claims with different evidence.

## Failure languages by domain

For formula and physics search, useful certified families include dimensional inconsistency, unit
dependence, violated symmetry or conservation law, undefined domain, failed exact special case,
wrong asymptotic limit, instability, non-identifiability, algebraic equivalence to an already
rejected class, sealed-holdout failure, and domination by a simpler comparator.

For proof and constructive mathematics, they include a minimal countermodel, invalid inference,
unmet quantifier, exact witness failure, exhaustive bounded region, unsatisfied proof obligation,
and a proved construction family that always supplies a witness for the opposite claim.

The strongest reusable result is not merely “candidate 847 failed,” but “every candidate with
properties P and Q fails because witness W exists.” That promotion from instance to family must be
verified independently of the proposer.

## LLM and creativity policy

LLMs are used aggressively for creative expansion. They may:

- propose new mechanisms, representations, transformations, counterexamples, or proof plans;
- combine ideas from distant domains;
- explain why a family may fail and suggest the smallest generalization;
- label a proposal as a known rewrite, a new combination of known mechanisms, a proposed new
  construction, or uncertain;
- target sparsely explored frontier regions.

Those origin labels are provenance, not a novelty verdict. Canonical equivalence checks, searches
of known work, and independent reviewers decide whether a correct result is structurally or
historically new. In particular, an LLM suggestion marked “likely known” remains searchable unless
there is a verified equivalence certificate; this prevents premature tree pruning.

## Role of the RTX 5090

GPU compute is most useful between generation and certification:

- batch-evaluate coefficient grids, grammar expansions, and parameterized constructions;
- adversarially sample boundary, singular, asymptotic, and regime-transition cases;
- estimate which candidate families occupy large failing volumes;
- search for small counterexamples in parallel and minimize witnesses;
- train a failure predictor used only to order exact tests;
- measure frontier coverage and structural diversity.

GPU output is a discovery instrument, not a proof. Every activated certificate must end in exact
arithmetic, symbolic verification, a proof kernel, or a sealed external score appropriate to its
claim.

## Measuring whether this is more creative

Compare the new loop with the old failure-first system and matched random search at equal generation
and verification budgets. Freeze the targets, prompts, code hashes, and metrics before opening
answers. Report:

- independently valid survivor discoveries per unit compute;
- verified exclusion volume added per unit compute;
- time to first decisive valid result;
- structural diversity among surviving equivalence classes;
- fraction of compute avoided by replayed certificates;
- family-level certificates discovered, not just failed instances;
- false-prune challenge rate, which must be zero for activated certificates;
- correct candidates the heuristic model warned about but correctly did not prune;
- old/new/random ablations and fresh-target replication.

More exclusions alone are not creativity. A system that excludes everything is useless. The aim is
to remove certified dead regions while preserving and diversifying the viable frontier.

## Task 2 concrete ledger

The available-now MathOverflow trial produced 36 sealed proposals. None supplied a valid
counterexample. The first ledger therefore records:

- 36 exact-instance exclusions, so identical failed submissions are not retried as decisive
  solutions;
- five exact generator-closure failures with executable integer-counting witnesses;
- a proved exclusion for every family with a universal element;
- a proved exclusion for the complete nonempty powerset family;
- a proved exclusion for all `(n-1)`-subsets plus the full set;
- one high-symmetry warning that remains explicitly heuristic and cannot prune;
- a frontier that requires strict residual excess for every element and prioritizes, without
  requiring, near-regular odd-degree candidates close to the boundary.

All 36 historical submissions are excluded. That is an honest record of a failed trial, not a claim
that the full mathematical search space has been exhausted. The architecture must now be tested on
a new untouched target; the opened MathOverflow problem is training and debugging material only.

As an over-pruning control, the canonical classifier excludes the six-set `(n-1)`-subsets-plus-full
family through its proved parametric certificate, while the independently verified 30-set accepted
counterexample survives every active certificate and passes the exact target verifier. Feature
predicates are accepted only with the declared `frankl_finite_family_features_v1` canonicalizer ID.
