# Independent Discovery Trial (IDT)

This document defines what must be true before Invariant may claim it **independently discovered**
a proof rather than **recovered** one it was given enough information to determine. It is written
against the measured evidence in `GOALS_AND_MEASURED_OUTCOMES.md` and inherits every existing
claim-boundary rule: absence from a corpus is never novelty, a bounded proof is never promoted to a
broader scientific claim, and a negative result is a valid, publishable trial outcome.

## 1. The measurement problem

Every discovery experiment run so far is real, blinded, and honestly reported. None of them
answers the question in this document's title. Here is why, experiment by experiment.

| Experiment | Measured result | Why it does not establish independent discovery |
|---|---|---|
| Blind semantic formula guessing | 0 PASS / 21 REJECT | Honest failure. Establishes the system cannot infer hidden structure from public rows alone. |
| Constraint-conditioned semantic recovery | 21 PASS / 0 REJECT | The public constraints made the answer unique. The report states it directly: recovery "belongs to the generic solver plus the public constraints, not to seven independent discoveries." |
| Complete blind benchmark curriculum | 200/200 registered and ready | Targets are self-generated from five exact one-parameter template families. They are in-basis by construction. |
| Native non-polynomial independence | 4 PASS / 2 REJECT | Genuine construction-family independence, but both worlds were authored inside this project. |
| Prospective blind tournament | 2 of 3 worlds passed | Both passes depended on Bayesian proposals; the ablation flipped both when Bayesian was removed. |
| Lean kernel proofs | 3 PASS, false controls rejected | The kernel is real and pinned. The theorems are elementary: `2*sum(n) = n*(n+1)`, a `Fin 11` exponent identity, and a cubic recurrence closed form. |

The common defect is structural, not procedural. **Every target so far was authored inside the
system's own grammar.** That guarantees the target lies in the declared basis, and an in-basis
target reduces discovery to an exact linear solve. The blinding chronology is already rigorous —
denied pre-unseal probes, zero target bytes exposed, one atomic unseal, zero post-unseal tuning.
Chronology is not the gap. **Target provenance and basis membership are the gap.**

## 2. Four conditions for a valid trial

A trial result is admissible only if all four hold simultaneously.

1. **External target authority.** The target is authored outside the discovery system by an
   identity with no read access to the declared grammar, basis, or `src/`.
2. **Out-of-basis certification.** Before unseal, an exact certificate proves the target is not
   expressible in the declared basis. Without this the trial re-tests the linear solver.
3. **Blinded chronology.** Preserved unchanged from current practice.
4. **Exposure control.** The proposal path must not have literature access to the target, or the
   target must postdate the proposer's knowledge cutoff.

Condition 4 needs emphasis. LLM and Bayesian proposers carry literature priors. "Discovered without
knowing it already exists" is only meaningful under one of two declared modes:

- **Mode N (no-exposure).** The proposal path excludes LLM and literature-derived priors entirely,
  exactly as the existing non-Bayesian lane does.
- **Mode P (post-cutoff).** LLM proposals are permitted, but every target is verified to postdate
  the proposer's knowledge cutoff by publication date.

Modes are reported separately and never pooled. A Mode P pass with an unverified cutoff is a
recall result, not a discovery result.

## 3. Sequencing warning

**Do not build the trial harness before the capability gaps in section 4 are closed.**

Conditions 1 and 2 without capability B1 guarantee a rigorous, well-documented zero. That
experiment has already been run: blind semantic guessing returned 0/21 for exactly this reason.
Repeating its shape at higher ceremony produces no new information.

The minimum viable trial is **A1 + A2 + B1 + B2**. B3 through B7 deepen the result but are not
required for a first honest measurement.

## 4. Open analysis areas — what we currently cannot discover

These are capability gaps, not tuning problems. Each one makes an entire class of target
unreachable by construction.

| ID | Gap | Status | What it blocks |
|---|---|---|---|
| B1 | **Basis synthesis** | **BUILT** — `basis_synthesis.py`, 30 tests. Declared 21-entry ladder, Occam-ordered, minimality certified against every simpler entry, holdout-confirmation required. | Was: everything out-of-basis. Now reaches polynomial, geometric, alternating, harmonic, factorial, binomial, and reciprocal families without a caller-declared basis. |
| B2 | **Nonlinear coefficient search** | **BUILT** — `nonlinear_coefficient_search.py`, 22 tests. Exact elimination over a declared model set; irrational/complex roots discarded, never rounded. | Was: any parameter inside an exponent or denominator. Now reaches geometric, shifted-geometric, linear-fractional, reciprocal-affine, and bounded-exponent power laws. |
| B3 | **Conjecture generation** | **BUILT** — `conjecture_generation.py`, 17 tests. Seven statement kinds, proposed from a prefix and confronted with a held-out suffix, ranked on a Pareto front with no scalar score. | Was: the system could only answer, never ask. Now proposes closed forms, recurrences, divisibility, congruence, sign, monotonicity, and partial-sum statements from raw data with no declared target. |
| B4 | **Mathlib and proof search** | **NOT BUILT — deliberately deferred.** Lean is not installed on this machine (no `elan`/`lake`/`lean`, zero packages in `lake-manifest.json`, no mathlib cache). Adding mathlib is a multi-hour build on a CI that is currently red and already hit a 30-minute shard timeout. | Nearly all mainstream mathematics, and all of B6 beyond `Nat`. See sequencing note below. |
| B5 | **Lemma decomposition** | **BUILT** — `lemma_decomposition.py`, 22 tests. Generates the base-case / successor-identity / induction-skeleton split that was previously hand-written, with obligations that fail independently. | Was: every helper lemma was authored by a person, capping proofs at what the author already understood. |
| B6 | **Quantified / infinite-domain reasoning** | **PARTIAL** — `quantified_inequality_proofs.py`, 18 tests. Universally quantified polynomial inequalities over `Nat` via exact forward differences. Reals, limits, and analysis remain blocked on B4. | Was: equalities only, over `Nat` induction or finite `Fin 11`. Now covers inequalities over an infinite domain — enough to prove B3's monotonicity and sign conjectures, which previously had no proof path at all. |
| B7 | **Structural counterexample repair** | **BUILT** — `structural_repair.py`, 18 tests. Basis union and declared domain restriction, under a budget where `rows - parameters >= min_confirmations` across the whole composition. | Was: a wrong structure was simply REJECT. Now recovers cross-family structure such as `n^2 + 2^n` and reports restricted-domain theorems with the restriction always attached. |

### Sequencing note on B4

B4 is the one primary-track item deliberately left unbuilt, and the reason is a schedule
conflict rather than a design one. Mathlib would be a large new dependency on a CI that
is presently failing seven system7 lanes and has already timed out one shard. The
generated Lean in B5 and B6 therefore restricts itself to the Std-only tactic vocabulary
that this repository's existing kernel-verified proofs already use (`simp only [...]`,
`omega`, `rfl`), so nothing emitted today depends on mathlib. Land B4 only after CI is
green, and expect the mathlib build to need its own cached CI stage.

Two further areas are blocked for reasons outside the math stack:

| ID | Gap | Current state |
|---|---|---|
| B8 | Nonlinear well-posedness for the gravity candidates | 12 BLOCK / 0 PASS on H6/H7 closure; zero global H7 proofs, zero closed bootstraps, zero positive lifespans. D2F coverage 5,324 of 257,499 entries per candidate. |
| B9 | Any physics claim requiring data | Zero authorized observations opened, ever. Eight external opening commitments outstanding. |

## 5. Goal ladder

### Tier A — make the trial measurable

- **A1. External target authority.** Bind an authoring identity with zero read access to the
  grammar. Acceptable sources: a fixed public corpus snapshot with a hard date cutoff; targets
  authored directly by the operator and sealed; or a separate agent isolated from `src/`. Emit
  `external_target_authority.json` recording source, cutoff, authoring identity, and the existing
  denied-probe chronology. **Fail-closed:** if the authoring identity can read the declared basis,
  the trial is void.
- **A2. Out-of-basis certification.** Emit an exact rank certificate proving the target is not in
  the span of the declared basis, sealed before unseal. This is the single change that converts
  "recovery" into "discovery."
- **A3. Exposure-control modes.** Implement Mode N and Mode P as declared, separately reported
  lanes with their own receipts.
- **A4. Negative-result contract.** 0/N is a completed trial with full ceremony, not a failed run.

### Tier B — close the capability gaps

Build order was **B1 → B2 → B7 → B3**, with **B4 → B5 → B6** as a parallel proof-side track.
B1, B2, B7, B3, B5 are built and B6 is partial; B4 is deferred per the sequencing note above.
**129 tests, Ruff clean.**

The stack composes end to end. Given only the raw integers `7, 12, 33, 82, 171, 312, …`
with no declared basis, no declared target, and no declared statement:

* **B3** proposes five statements across four theorem kinds and all five survive the
  held-out suffix — the closed form `7 + n + 2n^2 + 2n^3`, the third-order recurrence
  `a(n) = 3a(n-1) - 3a(n-2) + a(n-3)`, a partial-sum closed form, `a(n) < a(n+1)`, and
  `a(n) > 0`. Four are Pareto-non-dominated.
* **B5** splits the closed-form proof into `base_case`, `successor_identity`, and
  `main_induction`, and emits Lean whose skeleton matches the proof this repository
  previously kernel-verified with a hand-written helper.
* **B6** proves the monotonicity conjecture over all of `Nat` via the exact forward
  difference `6n^2 + 10n + 5`.

Every one of those receipts carries `kernel_verified: false`. Layer-one exact checking is
not kernel verification, and the two are never merged.

### Case study: probing the engine with an open problem (Collatz)

The engine was pointed at the total stopping time `sigma(n)` of the Collatz map — a
genuinely open problem since 1937.  **No progress on the conjecture itself is claimed or
was expected.**  The point was to observe the failure precisely, and the first run failed
completely: B1 BLOCK, B7 BLOCK, and B3 proposed *nothing* — zero conjectures, zero
refutations — while two elementary true statements sat unseen in the same data.  That
silence exposed two structural blind spots, both general rather than Collatz-specific:

| Gap | What was missing | Fix | Discovery it enabled |
|---|---|---|---|
| A | Sub-domains were dense-only (`n >= c`, parity) with no relabeling; structure living in a transformed index was invisible. | Sparse geometric and arithmetic-progression restrictions with explicit reindexing maps in `structural_repair.py`; a reindexed result always declares its index variable. | `sigma(2^k) = k` on `n = 2^k`, holdout-confirmed. |
| B | No statement kind could relate `a(n)` to `a(c*n)` — closed forms bind `a(n)` to `n`, recurrences to fixed lags. | New `index_scaling_relation` statement kind (`a(cn) = alpha*a(n) + beta`) in `conjecture_generation.py`, with vacuous-point discipline: rows whose scaled partner is absent contribute no support. | `sigma(2n) = sigma(n) + 1`, 26 holdout confirmations. |

The second discovery then closed the full loop: its honest conditional form — *if* `n`
reaches 1 in `k` steps *then* `2n` reaches 1 in `k + 1` steps — is provable without any
termination assumption, and `formal/lean/CollatzHalvingRelation.lean` states reachability
inductively and proves it with Std-only tactics (pending CI kernel verification, since
Lean is not installed locally; wiring the file into a CI lane is follow-up work).
The claim boundary holds throughout: both discoveries presuppose the stopping times
exist; neither says anything about termination, which is the open part.  The case-study
gates live in `tests/test_collatz_case_study.py`, including a test that no statement the
engine emits can even express a termination claim.

### Billion-scale gravity gate: the GPU phenomenology screen

The production funnel's measured failure mode was structural: ~10^9 static actions
screened cheaply, then straight into minutes-per-candidate symbolic formal gates —
**70/70 formal rejections with zero rotation curves ever computed**.  The missing tier
is a physics-informative gate at GPU speed, and
`gpu_baryonic_interpolation_screen.py` is that tier, run to completion on this machine:

| Measured | Value |
|---|---|
| Family | `nu(y) = (P(u)/Q(u))^beta`, `u = y^(-1/2)`, coefficients `{-3..3}`, `beta in {1/3,1/2,1,2}` — ordinal-indexed, zero per-galaxy freedom |
| Candidates processed | **1,129,900,996** (the complete declared family) |
| Wall time / throughput | **18.5 s** on the RTX 5090 — **61.1M candidates/s** |
| fp64 survivors | 4,478,916 (0.396%) |
| Pareto front (simplicity, Solar-System convergence, flatness) | 64, **all 64 re-confirmed at 50-digit mpmath** |
| CPU/GPU decision cross-check | 4,096 samples, **0 disagreements** |
| Receipt | `runs/gpu-baryonic-screen/billion-v1.json` (sealed, config-hash-bound) |

Screening criteria on frozen Freeman-disk controls (three disks spanning a 256x
baryonic mass range): definedness/positivity, Newtonian recovery at `y = 10^4, 10^6`,
monotone `g_obs(g_bar)`, flat outer rotation curves, and measured baryonic
Tully-Fisher slope within 0.30 of 4.  The physics controls behave exactly as the
dark-matter problem says they must: **`nu = 1` (baryons under pure Newton) fails flat
curves**; known interpolating laws (`1+u`, `sqrt(1+u^2)`, `cbrt(1+u^3)`) pass with
measured slopes 3.89-4.00; overboost (`1+u^2`) and every `beta = 2` shape fail.  The
`beta = 2` family is enumerated *because* it cannot work — the tests discriminate, the
grammar does not smuggle the answer in.

Notable front structure: the single-term known law `1+u` is the simplicity anchor;
two-term laws such as `(1+3u^5)/(1+u^4)` reach Newtonian error ~10^-8 at `y = 10^4`
(orders of magnitude faster Solar-System convergence than `1+u`) while keeping ~2%
outer-curve flatness.  A constant rescaling of the deep-limit coefficient is degenerate
with the value of `a0` and is not treated as a distinct discovery.

Boundary, unchanged: survivors are **search priorities for the sealed
covariant/formal/observational ladder, not validated theories**.  No observational
data was opened, no invisible mass appears anywhere in the grammar, and the receipt's
claims block records both.  The remaining funnel bottleneck is now explicitly the
formal tier's per-candidate cost, which is where the next feature investment belongs.

### Tier C — run the trials

- **C1. IDT-1.** External, out-of-basis, exposure-controlled targets in a domain where B1+B2
  suffice. Report PASS / REJECT / BLOCK with exact certificates and Lean proofs where the theorem
  admits one. A low pass rate is the expected and acceptable first result.
- **C2. IDT-2.** Add B3: the system proposes the statement, not only the proof.
- **C3. IDT-3.** Add B4/B5: a mathlib-scale theorem closed by decomposed lemmas.
- **C4. Physics IDT.** Gated on B8 and B9. Kept strictly separate — a math result never implies a
  physics result.

### Tier D — the novelty boundary

- **D1.** Expand the 181-record prior-art snapshot with independently sourced corpora. The current
  corpus is far too small to support any absence-based statement.
- **D2.** Keep two terms permanently distinct: **independent rediscovery** (the result exists; the
  system was blinded from it) and **novelty** (absent from a large corpus, plus expert review).
  IDT measures the first. It does not measure the second, and no IDT receipt may imply it.

## 6. Claim boundary

Completing this ladder would establish bounded independent rediscovery under declared exposure
control, on externally authored out-of-basis targets, with exact post-hoc verification. It would
not establish novelty, scientific significance, coverage of mathematics, or any physics result.
Those remain separate questions governed by their existing gates.
