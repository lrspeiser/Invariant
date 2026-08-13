# Sigma Core and Math Pack v1

## Purpose

Invariant is evolving from a gravity-first compiler into a domain-independent discovery system.
Gravity remains a first-class domain pack. The shared core operates on **candidate artifacts** and
does not decide what validity means for a particular field.

The governing rule is unchanged:

> A proposal may advance only when every required verifier returns exact, candidate-bound evidence.
> Missing, unsupported, timed-out, or untrusted evidence is a block, never a pass.

## Sigma Core

The core pipeline is:

```text
generate
  -> type check
  -> canonicalize
  -> cheap reject
  -> counterexample search
  -> exact verification
  -> prior-art comparison
  -> domain-specific admission
```

A candidate artifact may be a formula, identity, conjecture, theorem, proof, algorithm,
construction, or physical action. Every artifact carries:

- a closed-world schema and type;
- canonical content and identity hashes;
- generator and premise lineage;
- the exact domain-pack version;
- required gate names and ordered gate receipts;
- forbidden dependency and data-access declarations;
- promotion state derived from receipts rather than asserted by the generator.

Domain packs provide typed generation grammars, canonicalization rules, counterexample domains,
exact verifiers, prior-art adapters, and an admission contract. A domain pack cannot redefine core
receipt semantics or translate `blocked` into `pass`.

## Math Pack v1

The first mathematical primitives cover:

- types: integer, rational, real, complex, sequence, set, matrix, polynomial, function, and graph;
- formulas: equations, inequalities, and recurrences;
- exact algebraic canonicalization;
- deterministic exact, boundary, random, and adversarial counterexample search;
- symbolic identity verification;
- induction certificates for bounded benchmark families;
- a future Lean adapter whose accepted proof dependency closure must be contained in the benchmark's
  allowed-premise manifest.

`math_pack.py` connects those primitives to Sigma Core. Its closed promotion path is `typed`,
`canonicalized`, `counterexample_screened`, `exactly_verified`, and `prior_art_checked`. A bounded
counterexample search may pass its screening stage but cannot satisfy exact verification. The first
exact verifier accepts rational-function equations that SymPy reduces identically to zero; other
formula and proof classes remain blocked until an explicit verifier is registered.

`math_benchmark.py` provides the first closed Mathematical Knowledge Graph and blind-holdout
manifest. It rejects cyclic or missing dependencies, requires the target and every downstream node
to remain forbidden before unseal, expands the dependency closure of every proof receipt, and
places all truth and provenance gates ahead of simplicity in a lexicographic score.

`math_proof.py` emits independently replayable exact certificates for rational identities and
first-order induction. Identity certificates bind the raw and canonical statements and a cleared
numerator witness on the regular denominator domain. Induction certificates independently verify a
registered base case and the symbolic successor after recurrence and induction-hypothesis
substitution. Higher-order induction, denominator nonvanishing, limits, and analytic branch claims
remain unsupported and therefore blocked.

`math_proof_strategy.py` separates heuristic strategy proposals from deterministic execution.
Exact algebra and first-order induction have registered executors; contradiction, substitution,
factorization, generating functions, bijection, invariant, symmetry, extremal arguments, descent,
and change of variables are typed proposal families that remain blocked until executors exist. LLM
proposals require a bound source receipt and carry no authority to promote themselves.

`math_lean_adapter.py` provides a bounded, no-shell Lean execution boundary and an allowed-premise
manifest. Missing Lean is an explicit block. Tagged dependency output is rejected if it reaches the
target, an equivalent theorem, any forbidden namespace, or anything outside the declared closure.
This is adapter readiness: no Lean or Mathlib corpus is downloaded, and the dependency audit is only
as trustworthy as the registered Lean-side emitter and source harness.

## Candidate-generator portfolio

Seven search strategies now operate directly on Sigma Core candidate artifacts:

- `evolutionary_candidate_generator.py` performs deterministic, bounded mutation and crossover.
  Every child binds its exact parent artifact references, canonical content hashes remove duplicate
  offspring, callback failures become typed records, and selection never authorizes promotion.
- `bayesian_candidate_generator.py` updates exact rational prior mass with complete, hash-bound
  likelihood batches and draws deterministic bounded proposals. Posterior mass is proposal priority
  only; it is not truth, evidence acquisition, a gate pass, or promotion authority.
- `egraph_candidate_generator.py` performs deterministic equality saturation using a closed registry
  of 14 exact rational-algebra rewrites. Its equivalence claim extends only to those rules and
  congruence. Unknown or caller-supplied rewrites are rejected.
- `grammar_candidate_generator.py` re-expresses the registered bounded grammar as typed Sigma Core
  artifacts, with exact work accounting and a replayable manifest.
- `symbolic_candidate_generator.py` enumerates exact rational coefficient grids over caller-built,
  bounded SymPy templates and rejects string parsing, floats, unsafe functions, and unknown symbols.
- `cross_domain_candidate_generator.py` transfers structural summaries through three closed
  templates across distinct source packs into an explicit target pack. Transfer never establishes
  that source semantics hold in the target domain.
- `llm_candidate_generator.py` accepts only a caller-injected, provider-neutral callback. It
  reserves exact integer micro-dollar and token exposure before each call, persists neither prompts,
  raw responses, nor credential values, and quarantines every deduplicated Sigma Core candidate
  behind the ordinary downstream gates. The module itself performs no network access.

`candidate_generator_portfolio.py` seals this capability boundary. Evolutionary, Bayesian,
e-graph, grammar, symbolic, cross-domain, and LLM generation are implemented and
candidate-artifact native. This completes the requested generator mechanism set, not the scientific
evaluation program: every proposal still has to traverse preregistered domain gates and held-out
benchmarks. Generator registration proves neither scientific truth nor novelty.

## Knowledge, ranking, and readable research records

`candidate_knowledge_graph.py` stores exact Sigma Core artifacts beside typed definition, axiom,
lemma, theorem, conjecture, proof, derivation, equivalence, and dependency relations. It computes
deterministic dependency closures and holdout cuts. A lookup can report only `present_in_this_corpus`
or `absent_from_this_corpus`; corpus absence is never novelty.

`candidate_pareto_explanations.py` enforces hard gates before soft objectives. Only candidates with
every required gate receipt at `PASS` enter exact rational Pareto fronts. Blocked, rejected, and
errored candidates receive no front, however attractive their simplicity or fit metrics may be.
Every explanation binds the contributing gate, check, evidence, and metric-receipt hashes and grants
no promotion authority.

`candidate_evaluation_ladder.py` is the Sigma-Core-native orchestration contract. It binds every
domain-pack stage to exactly one consecutive admission gate and one nondecreasing phase: cheap,
symbolic, formal, then observational. Execution stops on the first block, rejection, or error.
Observational code is never called unless every earlier stage and gate, including a formal phase,
has passed. Results retain exact stage/gate receipts, phase and status counts, a terminal outcome,
all skipped steps, and an explicit observation-opening flag; the runner itself never promotes.

`research_notebook.py` renders sealed receipts as deterministic Markdown and Jupyter notebooks. The
first pair presents the anonymous natural-sum rediscovery as a complete induction proof and one
quartic scalar-tensor candidate as a formal local survivor. Every cell cites immutable receipt paths;
the claim ledger distinguishes `proved`, `certified_local`, `blocked`, and `scope_limit`. The
notebooks are explicitly modern historical-style reconstructions and derived views, never authentic
historical manuscripts, private model reasoning, or replacement proof kernels.

Computational agreement is never proof. A million successful evaluations can advance a candidate
to a proof attempt, but cannot produce a `proved` receipt.

## Blind rediscovery protocol

Each benchmark has a preregistered manifest with four disjoint closures:

1. **Generation closure** — definitions, axioms, examples, grammar, and generator code visible before
   candidate selection.
2. **Verification closure** — proof rules and exact backends allowed to verify a selected candidate.
3. **Forbidden closure** — the target, equivalent formulations, target-derived lemmas, its historical
   proof, answer-bearing imports, network access, and undeclared files.
4. **Post-unseal closure** — historical theorem and equivalence references used only after a candidate
   and proof receipt have been sealed.

Pre-unseal file reads are deny-by-default and recorded. A successful benchmark must bind the allowed
premise root, source root, complete enumeration or search receipt, candidate root, counterexample
receipt, proof receipt, dependency-closure root, and post-unseal equivalence receipt.

The first implemented control withholds the closed form for the sequence defined by
`S(n) = sum(k, k=1..n)`. The generator receives only the definition, exact examples, and a bounded
integer arithmetic grammar. A correct result must independently propose a closed form, survive
adversarial integer evaluation, and carry an induction proof whose base and step identities are
checked exactly before the historical theorem is unsealed. Its bounded search enumerates 46,656 raw
coefficient triples and 12,167 canonical classes. One candidate survives the public examples and all
59 preregistered counterexample points; the induction base and successor identity are then proved
before the withheld fixture is read. This demonstrates the bounded protocol, not novelty or
unbounded polynomial discovery, and its file-read guard is process-local rather than an OS sandbox.

Required negative controls include:

- a formula that matches only the provided examples;
- a formula with a hidden boundary failure;
- a candidate that reads or imports the withheld theorem;
- a proof with an undeclared lemma;
- a correctly typed but unproved conjecture;
- a semantically valid result with forged or re-sealed promotion fields.

## Synthetic mathematics

Historical rediscovery is complemented by mathematical worlds generated after the model-training
cutoff. A synthetic-world generator seals:

- fresh symbols and operations;
- finite or algebraic axioms;
- a complete derivation graph generated by a trusted reference engine;
- a split between visible ancestor theorems and hidden holdouts.

The discovery engine sees only the visible subgraph. Scoring uses exact equivalence and proof
verification against hidden ground truth after sealing. This controls for memorization more strongly
than historical examples alone.

The first implemented world is an anonymous order-seven finite algebra. Its 24-term grammar is
evaluated on all 2,401 assignments per term (57,624 evaluations), producing 12 nontrivial theorem
classes. Eleven classes are visible and one complete two-formulation class is withheld. The engine
reconstructs and exhaustively verifies the single target candidate before unseal, then matches the
withheld class post-seal. This is one bounded synthetic world—not evidence of general equational
completeness—and its next gate is replication across independently generated worlds and an external
proof kernel that cannot see the holdout.

A second, structurally different world uses a connected anonymous nine-vertex, sixteen-edge graph.
All 255 nontrivial cuts modulo complement are checked through 4,080 edge-incidence evaluations and
partitioned into 11 exact cut-size theorem classes. One entire 40-member class is hidden; one class
candidate is rediscovered and every member is exhaustively proved before unseal. Two successful
worlds are evidence that the mechanism crosses representations, but still fall far short of the
100-world cohort and do not establish general theorem discovery.

## Scoring and promotion

Scores are lexicographic and cannot compensate for a failed hard gate:

1. well typed;
2. nontrivial under the registered equivalence relation;
3. survives counterexample search;
4. exactly agrees on the declared computational domain;
5. formally proved;
6. independent of forbidden premises;
7. simple under preregistered objectives;
8. absent from the unsealed comparison corpus.

Prior-art absence means only `not found in the registered corpus`; it never establishes novelty.

The principal benchmark metric is:

```text
withheld results independently rediscovered and proved
------------------------------------------------------
eligible preregistered holdouts
```

Secondary metrics report proof rate, counterexample kill rate, forbidden-dependency rejection,
search cost, proof cost, equivalence-class coverage, and results by artifact type. Formula,
conjecture, theorem, algorithm, and construction results are reported separately.

## Curriculum

The planned historical suite progresses through arithmetic and geometric sequences, sums of powers,
recurrences, binomial and generating-function identities, trigonometric and calculus identities,
number theory, combinatorial constructions, and historically nontrivial machine-checkable theorems.
The first release target is 100 historical holdouts and 100 synthetic holdouts.

That target is now preregistered as a closed 200-slot curriculum spanning five domains, two artifact
types, five levels, and two ordinals per cell in each historical/synthetic cohort. Only the two
implemented controls are ready; 198 slots are explicitly missing. The aggregate decision remains
`not_ready_missing_or_invalid_benchmarks`, so registration is not misreported as benchmark success.

No benchmark-wide success is claimed until every holdout has an immutable manifest, an isolated
generation closure, deterministic replay, negative controls, and independently validated proof and
dependency receipts.
