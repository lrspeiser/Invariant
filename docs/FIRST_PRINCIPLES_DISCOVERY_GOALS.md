# First-principles discovery goals

The forward-looking spec for the Invariant engine, written 2026-08-18, reordered 2026-08-19.

**The primary goal is to find a creative new law that fits the data.** Everything else in this
document is subordinate to that, including every guard in it. The engine currently has roughly
132 gate, audit and control modules against 26 search and generator modules, and not one novel
result. That ratio is the problem, and a spec that leads with its guards makes it worse.

So read the tiers in this order: **Tier 2 (derivation), Tier 6 (construction), Tier 7
(reconciliation) are the work.** Tier 1 is the substrate they run on. Tiers 0 and 3 are what
make a result checkable *once you have one* -- they are not a gate on attempting anything, and
no work should be scheduled on them that is not unblocking a search.

One guard is worth keeping cheaply and continuously: a held-out split. Fit on most objects, keep
a few back. Not to protect an announcement -- so that when something fits you can tell within a
day whether it means anything. The sealed-trial opening protocol (I5, U3) is publication
infrastructure and is explicitly OFF the critical path until there is a candidate worth
spending it on.

`GOALS_AND_MEASURED_OUTCOMES.md` is retrospective: it records what has been measured.
This document is prospective: it states what the engine **claims it can do**, the
**test that would confirm it**, and — the part that matters — the **observation that
would prove the claim false**. A goal with no falsifier is not a goal here; it is
marketing, and it does not get an entry.

Every goal declares its **lane**: which substrate does the work.

| Lane | Substrate | Good for | Must never do |
|---|---|---|---|
| **CPU** | exact rational / prime-field arithmetic | certificates, canonicalization, proof translation | approximate anything |
| **GPU** | RTX 5090, 32 GB | screening 10^9 to 10^12 candidates, one-sided filtering | produce an accepted result alone |
| **LLM** | Claude via CLI, spend-capped | proposing *programs* and *kinds of thing to look for*; diagnosing stalls | see a target, or adjudicate its own output |
| **Kernel** | Lean 4.33 | independent proof of a finished claim | be trusted without a false control |

The division of labour is the thesis: **GPU proposes at scale, CPU disproves exactly,
Kernel certifies, LLM redirects when the search stalls.** No lane may close a loop by
itself. A result accepted by only one lane is an artifact, not a discovery.

---

## Tier 0 — Discipline invariants

These are not achievements and they are not a gate on attempting anything. They are what
makes a result checkable once you have one. Keep them cheap and continuous; never schedule a
sprint on them. If an invariant is costing more than the search it protects, it is
mis-scoped -- fix the scope, not the search.

| # | Invariant | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **I1** | **Blindness.** The answer is unreachable from the proposal path. | Sealed answer table off the prompt path; vocabulary guard refuses a prompt containing declared forbidden terms; deliberate leak attempts rejected, not sanitized. | Any accepted result whose prompt bytes contained a forbidden term, or any proposer able to read the target table. | LLM | **NOT enforced.** `vocabulary_violations` matches `[a-z]+` runs only, so word-channel; a numeric sealed constant reaches prompts unflagged |
| **I2** | **One-sided sampling error.** Sampling may only ever *overstate* the solution space, never understate it. | Sampled nullity must be at least the count of explicitly exhibited members; uniqueness claimed only when the sandwich closes on held-out samples. | A uniqueness claim where exhibited members are fewer than sampled nullity — sampling manufactured the result. | CPU | **Enforced.** Repaired: identically-vanishing is now measured on the full jet and unioned over bank+holdout, and `published_dimension` refuses an untight sandwich rather than publishing an open bound. All 5 receipt cells now publish with `sandwich_tight: true` |
| **I3** | **Exactness.** No floating point in any accepted certificate. | Rational or prime-field arithmetic throughout; every derivation replayed under a second prime. | Any accepted certificate whose second-prime replay differs, or any float on a certificate path. | CPU | Enforced |
| **I4** | **Platform-invariant replay.** A receipt verifies identically everywhere. | The same receipt hash from a Windows and a Linux checkout of one commit. | A receipt that passes on one OS and fails on another. | CPU | **Achieved only via out-of-band normalization.** 137 of 262 pins are CRLF-sealed; CI goes green because `scripts/materialize_hash_bound_worktree.py` (invoked 7x) rewrites the worktree until hashes match. `_canonical_lf_sha` already exists in 23 modules — the right primitive, inconsistently applied against raw-byte `_file_sha` |
| **I5** | **Seal before open.** Commitments precede any contact with real data. | Consume-once opening packet; preflight is target-free; zero reads recorded before authorization. | Any target read whose timestamp precedes its sealed commitment. | CPU | Enforced; 8 external commitments outstanding |
| **I6** | **Negatives are deliverables.** A refutation carries the same provenance weight as a confirmation. | Negative receipts have identical schema, hashing, and publication as positive ones. | A negative result that was run but not sealed and published. | all | Enforced (SPARC 0/12 INFEASIBLE) |
| **I7** | **Absence is not novelty.** Not finding prior art never implies originality. | Every novelty-adjacent claim cites the corpus snapshot and its size, and states the bound. | Any novelty claim resting on corpus absence alone. | CPU | Enforced (181-record snapshot) |

---

## Tier 1 — Brute-force substrate

The claim: **the search space can be traversed at a scale where exhaustion replaces
intuition.** This is what the 5090 is for.

| # | Goal | Test | Falsifier | Lane | Scale target |
|---|---|---|---|---|---|
| **S1** | **Trillion-scale expression enumeration.** Enumerate compositional expression space with a calibrated chance-match gate, so "we found a match" is separable from "a space this large contains matches by luck". | Run the enumeration; report matches *and* the calibrated expected number of chance matches at that space size. A result counts only if observed greatly exceeds chance. | An enumeration reporting matches without its chance-match rate; or observed matches within noise of chance. | GPU + CPU | **Achieved:** 1.911e12 enumerated, 3.822e12 evaluations, 531.9 s on the 5090 |
| **S2** | **GPU screening with zero divergence.** Every GPU verdict reproducible on CPU. | Sample GPU verdicts, recompute exactly on CPU, compare. Zero mismatches required. | One unexplained GPU/CPU disagreement. | GPU + CPU | 87.5B measured, 0 mismatches |
| **S3** | **Provable coverage, not sampled coverage.** For a declared space, prove nothing was skipped. | Emit a coverage certificate: enumeration cardinality, traversal cardinality, and a bijection or counted partition. | A coverage claim backed by a spot check rather than a counting argument. | CPU | every enumerated space |
| **S4** | **Canonical deduplication at scale.** Count distinct-up-to-symmetry, not distinct-as-written. | Canonicalize before counting; verify the canonicalizer is idempotent and symmetry-complete on a control set. | Two expressions equal under declared symmetry receiving different canonical forms. | CPU + GPU | 10^9+ |
| **S5** | **Sustained unattended operation.** The loop runs for days without a human. | A 72-hour continuous campaign with lease recovery, checkpointing, and a terminal receipt. | Any run requiring manual intervention to resume. | CPU + GPU | 72h (6h achieved) |

---

## Tier 2 — Derivation from first principles

The claim: **given a framework and a set of requirements, the engine derives what those
requirements force, rather than searching for a formula that fits data.** This is the
difference between curve-fitting and physics.

| # | Goal | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **D1** | **Constraints to field equations.** Declare geometry, tensor rank, symmetry, conservation, and a limit; derive the admissible family by exhaustion. | Enumerate contraction patterns, collapse by declared symmetries, report family dimension, exhibit explicit members on held-out jets. Rediscover a known answer as control. | The derived family excludes a known-admissible tensor, or includes one refuted by the declared constraints. | CPU | **Achieved** — Einstein tensor, dim 2, Lambda unforced |
| **D2** | **Equations of motion to Lagrangian.** Invert the variational map. | Given EOM, derive a Lagrangian whose variation reproduces them exactly; reject EOM admitting none (Helmholtz conditions). | A returned Lagrangian whose variation does not reproduce the input, or a false negative on a known-variational system. | CPU | **Achieved.** Control battery exists and is in CI: 78 Helmholtz conditions, 8 systems, 4 VARIATIONAL / 3 NOT_VARIATIONAL / 1 out-of-class, 65/65 tests pass |
| **D3** | **New equation *kinds*, not new instances.** The engine extends its own grammar. | Declare a grammar, statement, tactic, or gate as data; the engine admits it under the same verification discipline as mathematics and searches the new space with no new hand-written module. | Any new capability still requiring a hand-written module and a manual topology pin. | LLM + CPU | **Not built.** Note `LANGUAGE_PROPOSAL.md` is NOT in the repo or in git history — it is untracked in the `Invariant-release-final` worktree, so the earlier citation was wrong. 122 of 125 gaps open |
| **D4** | **Symmetry- and dimension-forced derivation.** Derive the form of a law from invariance plus dimensional analysis alone. | Declare the symmetry group and dimensions; enumerate invariants; report the forced functional form and its free parameters. | A derivation admitting a form that violates the declared invariance. | CPU | Not built |
| **D5** | **Domain transfer without modification.** The same derivation engine works in a second domain with only declarations changed. | Run D1's machinery on a non-gravitational system (elasticity, fluid, gauge) with zero code changes. | Any code change required to change domain. | CPU | **Achieved.** The framework is now *measured* off the generator's arrays (metric signature, curvature vanishing, index structure, rank, pair symmetry) and a mismatch raises `FrameworkMismatch`. One engine, two declarations: the curvature declaration returns `G_mn` and `g_mn`; the strain declaration returns the Lamé basis and the Navier-Cauchy operator. Config-only, zero code changes |

---

## Tier 3 — Verification

The claim: **nothing is believed because the engine produced it.**

| # | Goal | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **V1** | **Independent kernel proof.** A finished claim is proved by a checker sharing no code with the generator. | Translate to Lean 4.33; the kernel accepts. Pinned toolchain, closed dependency manifest, path-free receipt. | A claim the engine accepts and the kernel rejects, or a Lean receipt with an unpinned toolchain. | Kernel | Achieved for bounded classes |
| **V2** | **False controls must fail.** Every proof battery includes deliberately wrong inputs. | For each positive case, a minimally perturbed negative (one token, one coefficient) must be rejected. | A false control that passes. | Kernel + CPU | 118/118 formal controls; 3/3 single-token rejections |
| **V3** | **Adversarial self-refutation.** The engine tries to destroy its own result before publishing. | For each surviving candidate, run a refutation pass with an independent objective; survival is evidence only afterwards. | A published result never subjected to a refutation attempt. | LLM + CPU | Partial |
| **V4** | **Tamper detection is live.** Rewriting a receipt to match reality must fail closed. | Perturb a sealed artifact; the gate rejects rather than re-seals. | A gate that can be made green by editing its pin. | CPU | Enforced — and this is exactly what constrains the I4 fix |

---

## Tier 4 — Unsolved targets

The claim: **this machinery can be pointed at problems nobody has solved.** Each target
carries a *decision criterion* — the condition under which we stop, in either direction.
A target with no stopping condition is a treadmill.

| # | Target | Decision criterion (stop condition) | Falsifier of progress | Lane | Current |
|---|---|---|---|---|---|
| **U1** | **Global H7 / commutator closure** for the 12 quartic candidates | Either a checked global energy plus bootstrap plus positive lifespan, **or** a completion-grade exact obstruction covering every admitted closure strategy. | A claimed closure whose energy estimate fails a control, or an obstruction leaving a strategy unaddressed. | CPU | 12 BLOCK / 0 PASS |
| **U2** | **The 150 remaining recurrence packets** (45 K55, 45 TC2, 60 Sylvester) | All 150 registered and 117,180 coefficient rows emitted, or an exact proof the lift cannot close at `subset_2` order 3. | Any packet marked registered without its exact replay. | CPU + GPU | 154/304 |
| **U3** | **Sealed no-refit observation trial** | One authorized, sealed, no-refit real-data evaluation completed — PASS or REJECT, both publishable. | The trial being run after any constant was tuned on the same data. | CPU | 8 commitments outstanding; **0 data opened** |
| **U4** | **The 32 continued fractions** (DG1) | Each adjudicated: known identity, new identity with proof, or refuted. | An "interesting" verdict with no proof and no refutation. | CPU + Kernel | Open |
| **U5** | **Autonomous target selection.** The engine chooses what to work on next. | Given the 122-gap ledger, the engine ranks and selects; its choice must beat a random baseline on subsequent yield. | Selection indistinguishable from random. | LLM + CPU | Ledger exists; selection manual |

---

## Tier 5 — LLM course-correction

The claim: **LLMs are useful exactly where enumeration is not — proposing new *kinds* of
structure and diagnosing why a search stalled — and their contribution must be measurable
by ablation, not asserted.**

| # | Goal | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **L1** | **LLM writes programs, not answers.** The proposer returns source that is executed and scored, never a claimed result. | Sandboxed execution with kernel-enforced memory and wall limits; hostile-program suite fully contained. | A proposer output accepted without execution, or a hostile program escaping the sandbox. | LLM | Achieved 2026-08-18 (job-object memory cap) |
| **L2** | **Contribution is ablatable.** Removing the LLM must measurably degrade yield, or it is decoration. | Leave-one-family-out over the generator portfolio; report the delta. | An LLM lane whose removal changes nothing — then delete it rather than fund it. | LLM + CPU | Bayesian-dependent; LLM lane unmeasured |
| **L3** | **Stall diagnosis.** When a search plateaus, the LLM proposes the next *kind* of thing to try, not the next instance. | On a stalled campaign, the proposal must open a space the previous grammar could not express. | A proposal expressible in the existing grammar — that is enumeration, and the GPU should do it. | LLM | Not built |
| **L4** | **Spend discipline.** Cost is bounded, ledgered, attributable. | Hard cap, per-call charge, ledger written before the call, cap enforced fail-closed. | Any call not reflected in the ledger, or a ledger lagging actual spend. | LLM | Enforced; ledger drift found and fixed 2026-08-18 |

---

## Tier 6 — Construction and yield

**Why this tier exists.** Tiers 0 and 3 are guards, and Tier 1 is a filter. A system built
only from those converges on the safest possible output, which is nothing at all. The
measured record shows this drift already: the repository carries roughly 132 gate, audit,
and control modules against roughly 26 search, generator, and synthesis modules — about
5:1 toward refutation — and every PASS in the retrospective ledger is a rediscovery of a
known result or a bounded synthetic world. Not one is a novel discovery. That is an honest
record, and it is also a warning: refutation is cheap and decidable, construction is
expensive and open-ended, so an unbalanced system will spend its compute on the cheap side
and call the result rigour.

This tier is the counterweight. Its goals are generative, and they are falsifiable in the
same way the guards are.

| # | Goal | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **C1** | **Reachability certificates.** Before a search runs, prove the target class is *inside* the declared space. A null result must distinguish "no such object exists" from "our grammar could not express it". | For a declared space and a declared target class, emit a certificate: either an explicit embedding of the class into the space, or a proof that the class is outside it. Control: plant a known answer in the space and confirm the certificate finds it reachable. | Any exhaustive search reported as a negative result without a reachability certificate for what it was looking for. | CPU | **Built** (one space). 879,652,953 valid 9-token programs collapse to 701,726 distinct exact rationals; per-target certificates in ~2 ms; 29/40 random rationals REACHABLE with constructive derivations. **No search consumes it yet**, so `main`'s negatives still carry no certificate |
| **C2** | **Yield is measured.** Discovery rate is a tracked metric with a floor, not an emergent accident. | Track sealed novel positives per unit of GPU-hour and per unit of LLM-dollar, per campaign. Publish the number whether or not it is zero. | A reporting period with zero new sealed positives *and* no reachability certificate explaining the null — that combination means the engine cannot tell whether it is working. | all | Not tracked |
| **C3** | **The engine proposes the statement.** Conjecture generation, not just conjecture checking. | The engine emits candidate statements with their own verification obligations, and at least some are proved without a human having named the target. | Every proved statement traceable to a human-supplied target. Then the engine is a checker, not a discoverer. | LLM + CPU | **Built.** 210 statements emitted with no human-named target, 179 adjudicated, 173 PROVED / 6 REFUTED; verdicts independently re-derived by direct summation to n=3999 with zero discrepancies. Mathematics is elementary by construction (degrees ≤ 5) |
| **C4** | **Partial credit and gradient.** A near-miss must score differently from noise. | Every gate that returns BLOCK also returns a distance: which obligations were met, which failed, and by how much. Rank by it. | A candidate meeting all but one obligation scoring identically to one meeting none — that destroys the signal a search needs to steer. | CPU | **Satisfied for one gate.** The uniform 12-way BLOCK on global H7 becomes a 4-tier strict order, each step carrying two separating rationals checkable in integer arithmetic (tightest gap 6.42e-14). Every other BLOCK gate is unchanged |
| **C5** | **Symmetric resourcing of blockers.** Every BLOCK gets construction effort, not only sharper characterisation. | For each blocked target, resource an explicit attempt to *build* the missing primitive, with at least the compute spent on describing the obstruction. Track both numbers. | A blocker whose entire work product over a reporting period is a more precise account of why it is blocked. Characterising an obstruction is refutation wearing a lab coat. | mixed | U1 has 12 BLOCK and a growing obstruction literature |
| **C6** | **The search learns.** A 10^12 sweep must be steered by what survived the last one. | Compare the proposal distribution at the start and end of a campaign; survivors must shift it measurably. | Enumeration whose distribution is identical at both ends — that is a lottery with extra steps, not a search. | GPU + LLM | FunSearch loop exists but is not wired to the hard targets |
| **C7** | **Exploration budget.** A declared fraction of compute goes to low-probability, high-payoff regions. | Reserve and report a fixed share of each campaign for candidates the current scoring would reject; measure what it returns. | An exploration budget of zero, or one that is silently reallocated to exploitation when results are thin. | GPU | Not declared |

**The governing ratio.** Report refutation-compute against construction-compute every period.
The number is currently unmeasured and the module count suggests it is badly skewed. A
verification tier that costs more than the search it verifies is not rigour; it is a system
protecting itself from the possibility of being interesting.

## Tier 7 - Per-object decomposition and reconciliation

**The idea.** Fit every galaxy and every cluster *independently*, with its own free
parameters, and keep the whole population of local solutions rather than one global fit.
Then run a second pass that searches for a single universal law able to reproduce that
entire population with no per-object freedom left.

**Why it is worth doing.** A global fit that fails tells you only that it failed. A
population of per-object fits tells you *which parameter refused to stay constant, and by
how much* - and that is a measurement, not a null. If a per-object parameter is constant
within its uncertainty, the universal law is already in hand. If it varies, the variation
is either tracked by something measurable, which is the discovery of a new dependence, or
it is irreducible, which refutes the family far more sharply than a global INFEASIBLE.

**Refit freely.** Per-object knobs are the point of pass 1; a population of per-object fits
is the measurement. The only bookkeeping required is a held-out split declared before pass 1 --
fit on most objects, keep a few back -- so a reconciled law can be checked cheaply against
objects it never saw. That is one line of record-keeping, not a ceremony. Formal sealed-trial
machinery applies only if and when a reconciled law is good enough to publish, and must never
be a reason to delay running pass 1.

| # | Goal | Test | Falsifier | Lane | Current |
|---|---|---|---|---|---|
| **R1** | **Per-object decomposition.** Every object is fitted independently and the whole population of solutions is kept, not just the best. | Fit each galaxy/cluster with its own parameters over the declared law space; seal the population as an exploratory receipt with the exploration/confirmation split recorded. | Any per-object fit that touched a confirmation-set object, or a population reported without its split. | GPU + CPU | **Built.** `per_object_law_decomposition.py`: 14 declared law spaces x 4 exploration galaxies = 56 independent single-object fits, whole population kept and none ranked. The split is a salted hash of galaxy *names* (confirmation NGC7331, DDO154), sealed with its own digest before pass 1, and every fitting entry point raises `ConfirmationSetTouched` rather than declining. Caveat recorded in the receipt: the withheld galaxies are not virgin, the predecessor exploratory receipt already scanned them |
| **R2** | **Parameter-variation diagnosis.** Report which per-object parameters are constant within uncertainty and which are not. | For each parameter, an exact interval per object; test whether a single value lies in every interval. A constant parameter means the universal law already exists. | Declaring a parameter constant when no single value lies in all intervals, or variable when one does. | CPU | **Built.** Exact rational interval per object per parameter, and an exhibited witness or an exhibited disjoint pair with its exact gap. Measured: the Newtonian per-galaxy baryonic rescale is **VARIES** (price of universality 1.1045, NGC2403 vs NGC3198 disjoint by an exact rational), every one of the 12 screened families is **CONSTANT** (price exactly 1). The deliberately wrong law is CONSTANT too at k_pop = 165.7, which is why the price is published as a diagnosis and never as a score |
| **R3** | **Channel discovery.** Ask whether the variation is tracked by a measured but *unlabelled* channel. | Give the reconciliation search extra unlabelled channels plus an MDL cost for using one. A channel earns its place only by beating every law that ignores it. | A channel declared explanatory without paying the MDL cost, or a named channel - naming it before it is measured leaks the concept. | GPU + CPU | Not built |
| **R4** | **Reconciliation search.** Find one law that reproduces the entire population with zero per-object freedom. | Search the law space for a candidate whose per-object residuals are inside the intervals from R2, with no free parameter left per object. Report the best candidate AND the exact obstruction if none exists. | A reconciled law that quietly retains a per-object parameter, or a claimed reconciliation whose residuals are not checked against R2's intervals. | GPU + CPU | Not built |
| **R5** | **Held-out confirmation.** The reconciled law is tried once, on objects that took no part in R1-R4. | A sealed no-refit evaluation on the confirmation set, PASS or REJECT, both publishable under I6. | Any tuning of the reconciled law after the confirmation set is opened. That spends the trial and voids the result. | CPU | Not built |

**How this answers a question the global fit cannot.** The sealed SPARC run returned
INFEASIBLE for all 12 candidates *and* for Newtonian baryons alone. That is two very
different failures reported as one word. Under Tier 7 they separate: baryons-only would
show a per-object parameter that varies wildly and tracks nothing, while a screened-gravity
family would show a parameter that is nearly constant with a structured residual. The
second is a lead. The first is a dead end. Today they are indistinguishable.

## How the tiers compose

A discovery counts only when it has cleared every tier it touches:

```
    LLM proposes a KIND            (D3, L3)
              |
    GPU enumerates INSTANCES       (S1, S2)     10^9 - 10^12
              |
    CPU disproves EXACTLY          (I3, S3, V2)
              |
    survivors -> DERIVATION        (D1, D2, D4)
              |
    Kernel PROVES                  (V1)
              |
    adversarial REFUTATION         (V3)
              |
    sealed CONFRONTATION           (I5, U3)
```

The right-hand column can only remove candidates. That is what makes surviving it mean
something -- no lane can rescue a candidate the previous one killed. But a funnel with no
pump outputs nothing, and reports that as success.

So the loop must close backwards as well:

```
    reachability certificate  (C1) --> is the answer even IN this space?
              |                             |
              | yes: search is meaningful   | no: WIDEN THE SPACE first (D3, D4, C3)
              v                             v
      filters run                    the null result was about the grammar,
      (S1-S4, V1-V4)                 not about nature -- and says nothing
              |
              v
    survivors + near-misses (C4) --> steer the next sweep (C6)
              |
    zero survivors + valid reachability certificate
              = a REAL negative, and publishable (I6)
    zero survivors + no certificate
              = an uninformative null, and must not be reported as a result
```

That distinction is the whole point of C1. Without it, "we searched 10^12 expressions and
found nothing" is indistinguishable from "we searched the wrong 10^12 expressions", and the
engine cannot tell which of those it just did.

## What would make this document wrong

Stated plainly, so it can be checked:

1. If a result clears every tier and is later shown false, the tier that should have
   caught it was a formality, not a filter. Find it, fix it, or delete it.
2. If the LLM lane is ablated and yield is unchanged (L2), Tier 5 is decoration.
3. If GPU screening ever disagrees with CPU verification (S2), the brute-force thesis
   collapses: scale would be producing artifacts, not candidates.
4. If a receipt verifies on one platform and not another (I4), every prior receipt's
   provenance is weaker than claimed. **This is currently true and unfixed.**
5. If the engine runs a full reporting period with zero new sealed positives and cannot
   produce a reachability certificate explaining the null (C1, C2), then it has become a
   very expensive way of declining to answer, and the balance in Tier 6 is wrong.
