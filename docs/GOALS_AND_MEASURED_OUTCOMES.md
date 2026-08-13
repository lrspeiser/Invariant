# Invariant goals and measured outcomes

This is the release-facing goal registry for the Invariant formula and proof discovery
system, current through **2026-08-13**. It distinguishes bounded engineering success
from mathematical, scientific, and operational completion. Counts below come from
checked receipts, tests, and the terminal workflow linked in the release section;
absence from a corpus is never treated as novelty, and a bounded proof is never
promoted into a broader scientific claim.

Overall status: **strong alpha; not yet scientifically or operationally complete**.

## A. Discovery platform

| # | Goal | Current measured outcome | Status / completion target |
|---:|---|---|---|
| 1 | Canonical candidate and provenance core | Eight artifact kinds, closed schemas, deterministic hashes, fail-closed stages/gates, and a hash-chained promotion ledger. | Bounded goal achieved. |
| 2 | Mathematical expression and proof layer | Exact expression IR, rational canonicalization, bounded counterexample search, rational-identity certificates, and induction certificates. | Bounded goal achieved. |
| 3 | Generator portfolio | Grammar, symbolic, evolutionary, Bayesian, e-graph, LLM, and cross-domain adapters exist: 7/7 families. | Implemented; scientific independence remains partial. |
| 4 | Common evaluation ladder | Typed PASS/BLOCK/REJECT/ERROR outcomes, hard gates, exact metrics, Pareto fronts, replay, and explanations. | Bounded goal achieved. |
| 5 | Domain-neutral Formula Discovery Job | Caller-supplied exact constraints now flow through rank classification, candidate construction, independent validation/counterexample, optional induction proof, Sigma Core provenance, deterministic reporting, orchestration, Pareto ranking, CLI exit codes, and immutable JSON/Markdown publication. | Product milestone achieved for bounded integer/rational linear-basis problems; broader nonlinear search remains open. |
| 6 | Knowledge graph and corpus boundary | Equation corpus: 18 artifacts, 31 nodes, 36 edges. Prior-art audit: 181 records; four alpha survivors absent from that snapshot. | Bounded implementation achieved; corpus breadth incomplete and absence is not novelty. |

## B. Discovery experiments

| # | Goal | Current measured outcome | Status / interpretation |
|---:|---|---|---|
| 7 | Basic benchmark curriculum | 5/200 registered slots ready; 195 explicitly missing. | Early coverage only. |
| 8 | Cross-generator alpha | Seven candidates: 4 PASS, 1 BLOCK, 1 REJECT, 1 ERROR; four entered exact Pareto ranking. | End-to-end pipeline exercised. |
| 9 | Prospective blind tournament | Three unseen worlds x seven families = 21 candidates; two worlds passed and one rejected. Both passes depended on Bayesian proposals. | Genuine bounded success; weak portfolio robustness. |
| 10 | Robustness and ablation | 8/8 CPU replay seeds exact; 168 evaluations and 336 gate comparisons stable. Removing Bayesian flipped both successful worlds; removing any other family changed none. | Reproducible but Bayesian-dependent. |
| 11 | Non-Bayesian recovery | Retrospective repair generated 48 candidates and recovered both frozen worlds. A prospective repaired tournament generated 72 candidates: 12 PASS, 60 REJECT, all three worlds recovered. | Breadth improved; worlds remain bounded and simple. |
| 12 | Blind semantic formula guessing | Three structured worlds x seven families: 0 PASS, 21 REJECT, 21 exact counterexamples. | Honest failure: hidden formulas were not learnable from the public input. |
| 13 | Constraint-conditioned semantic recovery | Quartic polynomial, partial fraction, and recurrence worlds: 21/21 candidates passed with exact certificates. Controls: 2 BLOCK, 1 REJECT. | Major bounded success. |
| 14 | Native independent discovery | A preregistered integer-polynomial world produced three non-Bayesian candidates before one target unseal. A native Newton forward-difference constructor recovered `[11,-3,0,2,-1]` and passed exact coefficient identity; fixed grammar and e-graph baselines were exactly rejected. Counts: 1 PASS, 2 REJECT, one identity certificate, two counterexamples, zero generic-solver calls, and zero floating-point operations. | Bounded native-construction goal achieved for one consecutive-grid polynomial world. Multi-world, non-polynomial, and independently isolated target trials remain open. |

## C. Independent proof checking

| # | Goal | Current measured outcome | Status |
|---:|---|---|---|
| 15 | Real external proof kernel | Portable Lean 4.33 execution, path-free checked receipts, pinned toolchain metadata, and closed dependency manifests. | Achieved. |
| 16 | Known-formula rediscovery proof | The natural-number sum was blindly rediscovered and proved by induction in Lean. | PASS. |
| 17 | Bounded number-theory proof | For nonzero elements of `Fin 11`, the recovered exponent-10 identity was checked by Lean. | PASS; bounded finite theorem. |
| 18 | Generated-formula proof | Recovered recurrence `u(n)=2n^3+2n^2+n+7` was proved by Lean induction; a false base case was rejected. | PASS. |
| 19 | Proof-strategy breadth | Quartic and partial-fraction identities are independently replayed with exact integer coefficient arithmetic; the quartic convolution is checked by Lean and a `-30` to `-29` false control is rejected. Formula Job translation executes generated polynomial and recurrence sources in real Lean. | Bounded breadth achieved; general rational-domain Lean translation and additional proof strategies remain open. |

## D. Gravity-theory science

| # | Goal | Current measured outcome | Status / missing work |
|---:|---|---|---|
| 20 | Known-answer formal controls | 118/118 portable formal controls pass. | Achieved as controls. |
| 21 | Local candidate health | All 12 quartic candidates pass local ADM rank/nullity, Dirac three-DOF counting, positive reduced quadratic Hamiltonian, and bounded local hyperbolicity checks. | Strong local result, not nonlinear/global closure. |
| 22 | Nonlinear second-variation coverage | 242 of 257,499 ordered D2F entries per candidate are registered; 257,257 remain. | Far from complete. |
| 23 | H6/H7 commutator closure | Finite hierarchy remains 12 BLOCK, 0 PASS. Representative slices cancel some terms, but the full high-atom identity and variable Bony remainder remain unclosed. | Major scientific blocker. |
| 24 | Nonlinear energy and lifespan | No closed global H7 inequality, quantitative lifespan, or dynamically preserved tube bootstrap. | Major scientific blocker. |
| 25 | Fourth-jet/full-direction closure | Thirteen local direction certificates exist. The prior BLOCK receipt remains historical evidence. The checkpointable materializer sealed 3/3 exact 55x55 P55 axis matrices with 144 total sparse entries and zero residuals in 3,025 linearity and 3,025 sphere minimal-polynomial checks. The recurrence-emitter chain advanced 3/304 to 6/304; candidate normalization and the sphere reducer reached 19/304. The committed axes then construct the coordinate-free order-zero pencil `P(n)=n1 P1+n2 P2+n3 P3` and register all 15 polarization packets, reaching 34/304. It still emits 0/117,180 coefficient rows and admits zero phase-two solves. | The coordinate-free recurrence remains BLOCKED on 270 packets. P55 orders 1–4 need 60 serialized state-jet derivative matrices erased by flat substitution; K55, TC2, and lower-Sylvester families also remain. Full direction-sphere D4, nonlinear H7, PDE closure, and lifespan remain open. |
| 26 | Universal matter PDE coupling | The minimally coupled scalar and irrotational `P(X)=kappa X^2` sector pass their four bounded gates; Maxwell now has an arbitrary-background Hilbert-stress identity with four zero residual components and a 48-monomial wrong-sign negative. A combined three-sector gate passes four interfaces: one physical metric/action split, additive on-shell stress conservation, common-time principal compatibility, and internal matter constraint-source closure. The six-component matter principal polynomial is `(|k|^2-omega^2)^5(|k|^2-3 omega^2)` with five light blocks, one acoustic block, and positive kinetic weights `[1,1,1,1,1,3]`. | The three-sector matter interface is achieved. Candidate-specific gravity coupling remains a typed BLOCK requiring six exact registrations: selected gravity action/Euler hash, total-stress insertion, full coupled principal matrix, common-time symmetrizer bounds, sourced gravity-constraint propagation, and a negative control. External Maxwell current/boundary terms, vortical matter, universal all-matter, global evolution, and H7 remain open. |

## E. Observational validation

| # | Goal | Current measured outcome | Status |
|---:|---|---|---|
| 27 | Solar one-shot trial | Two candidates BLOCKED; zero primary reads, held-out reads, or real evaluations. Each lacks source-branch instantiation, split commitment, selected primary roots, and opening authorization: eight missing commitments total. | Correctly fail-closed. |
| 28 | Galaxy validation | Generic evaluator: 70/70 BLOCKED. Reviewed registration has 11 entries with seven missing; only analytic/synthetic controls are admitted. | No real held-out pass. |
| 29 | Lensing validation | Schema controls pass; authorized real packets opened: 0; scientific result: one BLOCK. | Awaiting authorized data. |
| 30 | Cluster validation | Schema controls pass; authorized real packets opened: 0; scientific result: one BLOCK; rank writes: 0. | Awaiting authorized data. |
| 31 | One genuine observation trial | No candidate has completed a sealed, authorized, no-refit direct-observation evaluation. | Highest external scientific gap. |

## F. Operations and acceleration

| # | Goal | Current measured outcome | Status |
|---:|---|---|---|
| 32 | Persistent scheduler | Historical CPU campaigns processed 7,864,320 formulas. A current isolated rehearsal admitted three CPU tasks, deliberately expired one attempt-one lease, recovered it once with zero recovery failures, completed all three tasks with attempts `[2,1,1]`, and sealed checkpoint sequence 1. | Current scratch mechanics demonstrated without production-state access. |
| 33 | Current continuous operation | Formal batch 0003 advanced the cumulative cursor to prefix 13: 226 checked, 224 new, two reconciled, 226 REJECT, zero PASS/BLOCK/promotion, and 11,023 pending. | A bounded batch ran successfully; sustained operation is not established. |
| 34 | GPU acceleration | 163 candidates; 87,509,958,656 measured formula evaluations; 5,341,184 GPU/CPU comparisons and 5,216 exact checks with zero violations. | Strong synthetic acceleration control. |
| 35 | LLM safety and spending | Adapter is secret-safe and quarantined. Alpha runs made zero network calls. Historical aggregate: 51 calls and about $14.60 spent under the $500 cap. | Safety controls achieved; discovery value unproven. |
| 36 | Dashboard and recovery | Dashboard, immutable checkpoints, leases, and replay exist. Current samples admitted at 5.7%, 4.6%, and 5.1% CPU with at least 73,443 MiB available RAM, and the scratch recovery passed. Production SQLite/WAL/SHM was not opened, so current live process/lease freshness remains unestablished. | Current resource and scratch evidence; production freshness still open. |

## G. Reproducibility and release

| # | Goal | Current measured outcome | Status |
|---:|---|---|---|
| 37 | Focused alpha CI | The terminal clean-clone workflow exercised the expanded Formula Job/CLI/Lean translation, matter-control, P55 registration, native discovery, and proof-breadth surfaces without a failing focused shard. | PASS at release head `bb76e8e`; retain as a required merge check. |
| 38 | Full workflow CI | [Run 31746356515](https://github.com/lrspeiser/Invariant/actions/runs/31746356515) completed at head `bb76e8ec60325d72db69caf6d65d70ae4337f6d3`: 39/39 successful jobs, zero failures, zero cancellations, and zero timeouts. | Achieved for the current release head; any subsequent change must earn a new terminal 39/39 receipt. |
| 39 | Human-readable evidence | Deterministic Markdown/Jupyter reports cover successful derivations, exact blockers, prospective tournaments, semantic 0/21 vs 21/21 contrast, identity breadth, and a public CLI PASS/REJECT walkthrough. | Strong auditability. |
| 40 | Repository migration | Development, evidence, CI, and [draft PR #1](https://github.com/lrspeiser/Invariant/pull/1) live in `lrspeiser/Invariant`. At head `bb76e8e`, the PR is synchronized, mergeable, and reports 39/39 successful checks. | Migration achieved; remaining release step is human review and intentionally leaving draft state. |
| 41 | Release hygiene | Exact-path commits are clean. The materialized Windows worktree reports 1,935 line-ending/stat modifications, while both tracked and staged semantic diffs are empty. Live runtime/SQLite state remains outside the release boundary. | Use clean-clone CI and exact-path staging; do not bulk-stage the materialized worktree. |

## Current priority order

1. Preserve the 39/39 release baseline on every integrated change; never trade a
   scientific increment for an unverified release head.
2. Register the remaining 270 coordinate-free recurrence
   Taylor/operator/normal-form packets, emit the 117,180 exact rows, and resume the
   full-sphere D4/H7 chain.
3. Close the global H7 energy, tube-bootstrap, and quantitative-lifespan chain, or
   produce a decisive exact obstruction that terminates the candidate family honestly.
4. Supply the six candidate-specific gravity registrations required by the combined
   matter interface, close Maxwell external-current/boundary terms, and add a separate
   vortical-fluid model if universal matter remains the target.
5. Complete one authorized, sealed, no-refit Solar observation trial. Until the four
   opening commitments exist for each candidate, the correct result remains BLOCK
   with zero target reads.
6. Extend exact D2F coverage beyond 242/257,499 entries per candidate, with every new
   block carrying replayable leaf/root provenance.
7. Run and archive a fresh sustained scheduler/recovery campaign with current lease,
   checkpoint, resource-admission, and terminal queue evidence.
8. Extend native construction beyond one consecutive-grid polynomial world and test
   non-polynomial discovery without delegating the mathematical solve to the generic
   exact solver.
9. Expand the 5/200 benchmark curriculum and the 181-record prior-art snapshot using
   independent sources and externally checked proofs.
10. Resolve review feedback, reproduce the release from a clean checkout, and convert
    draft PR #1 into an intentionally reviewed merge-ready release.

The system is credible as an exact, auditable formula-recovery and proof pipeline with
documented success and failure modes. It is not yet a comprehensive theorem-discovery
system, a nonlinear gravity theorem, a universal matter result, or a validated empirical
discovery product.
