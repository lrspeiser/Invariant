# Continuous discovery roadmap

Target state: the system runs unattended, continuously pulling targets from a curated
queue of unsolved math problems and open physics problems, generating candidates at
GPU scale, filtering through exact and formal gates, proving what is provable in a
real kernel, and sealing honest receipts — including negative ones.  A famous open
problem being solved is never promisable; a pipeline that produces kernel-verified
new-to-corpus theorems, exact counterexamples, named capability gaps, and sealed
physics verdicts on every epoch is.

Grounding (measured, this repository):

- 1,129,900,996 baryonic laws screened in 18.5 s at 61.1M/s on the RTX 5090; 64-law
  Pareto front, all mpmath-confirmed (`runs/gpu-baryonic-screen/billion-v1.json`).
- Conjecture → decomposition → Nat-inequality proof loop closes end to end (B1-B7,
  162 tests), but only when a human launches each stage.
- Formal physics tier runs at minutes per candidate; the last production epoch died
  70/70 at that tier with zero rotation curves computed (now fixed by the screen, but
  the throughput mismatch — 4.5M prioritized survivors vs minutes each — remains).
- Zero real observations have ever been opened; eight external commitments outstanding.
- Lean runs in CI only; mathlib is not a dependency; CI has 7 red lanes mid-repair.

Every task below lists its **Output** — the artifact that makes it done.  An output is
always a sealed receipt, a merged capability with tests, or a named decision; never a
vibe.  A negative result (family rejected, conjecture refuted, gate failed) is a valid
output everywhere.

---

## Track A — Continuous operation (the "continuously" part)

**A1. Discovery scheduler service.**  Wrap the conjecture → recovery → repair → proof
pipeline in the existing durable campaign engine (leases, checkpoints, crash recovery,
`engine-start/status/stop/resume`), so discovery is a service, not a session.
*Output:* `discovery-engine-*` CLI; a 24-hour unattended soak receipt showing N
problems attempted, decisions per stage, zero manual steps, clean resume after a
forced kill.

**A2. Problem intake queue.**  A hash-bound registry of open targets: OEIS sequences
with no known closed form, bounded open conjectures (Erdős–Straus instances, aliquot
trajectories, prime-gap statements), and physics candidate families.  Each entry
carries provenance, why it is believed open, and what counts as progress.
*Output:* `configs/problem_queue_v1.json` + registry validator test; every queue entry
cites a source; "believed open" is a documented claim, never an inference.

**A3. Epoch triage into the dashboard.**  Surviving conjectures, proved theorems,
refutations, and physics survivors flow into the existing unified dashboard with
lineage links and Pareto axes.
*Output:* dashboard rows for discovery outputs; a test asserting no scalar
truth/probability score appears anywhere in the rendering.

**A4. Budget and watchdog.**  Per-epoch wall/disk/GPU-hour/LLM-dollar caps through the
existing atomic budget ledger; a tripped cap ends the epoch cleanly.
*Output:* epoch receipt recording enforced caps, plus one control run that
deliberately trips each cap and recovers.

---

## Track B — Math: from raw data to kernel-proved theorems, unattended

**M1. Mathlib integration (B4, currently deferred).**  Pinned mathlib in CI with a
cached build stage; Std-only emission stays the local default.
*Output:* one generated theorem whose proof requires a mathlib lemma, kernel-verified
in CI; toolchain + mathlib commit hashes pinned in the receipt.  Precondition: CI
green (D1).

**M2. Bounded proof search.**  Replace hand-templated proof shapes with bounded
tactic search (Std first, mathlib after M1) driven by failure packets: when a proof
attempt fails, the failing goal is recorded and becomes a lemma-synthesis target.
*Output:* receipt showing ≥K generated statements proved with zero hand-written
tactic scripts, plus a failure taxonomy for every unproved statement.

**M3. Statement-kind expansion by adversarial probing.**  Repeat the Collatz method on
new open-problem data: aliquot sequences, Erdős–Straus, prime gaps, continued-fraction
expansions.  Each probe either yields surviving conjectures or names the exact missing
statement kind (bivariate relations, valuation identities, asymptotic statements,
inequality chains between sequences), which becomes a build item.
*Output:* one case-study receipt per problem in the style of
`tests/test_collatz_case_study.py`; every engine silence converted into a named
capability gap or a discovery.

**M4. Conjecture-to-proof auto-wiring.**  B3 survivors route automatically: polynomial
closed forms → B5 decomposition; monotonicity/sign → B6; recurrences → the recurrence
prover; everything else → a typed `missing_prover:<kind>` block.
*Output:* a closed-loop receipt — raw data in, kernel-verified Lean out — with zero
human steps, and counts of auto-proved vs blocked per statement kind.

**M5. Independent targets and out-of-basis certification (IDT Tier A).**  External
target authority (dated OEIS snapshot, operator-sealed targets), pre-unseal
out-of-basis rank certificates, Mode N / Mode P exposure lanes.
*Output:* the IDT-1 campaign receipt on externally authored targets with honest
pass/reject counts; a low pass rate is the expected first result.

**M6. Prior-art corpus at meaningful scale.**  Grow the 181-record snapshot (OEIS
import under its license, arXiv/INSPIRE metadata via the source policy) so
"not found in corpus" carries weight.
*Output:* equation-universe import receipts; every discovery dossier carries a
prior-art screen against the enlarged corpus; absence still never claims novelty.

**M7. GPU counterexample search.**  For existence/universality conjectures in the
queue, exhaustive CUDA search over declared ranges (10^11-10^13 candidates per epoch)
with exact CPU verification of any witness.
*Output:* per-conjecture receipt: exact searched range, zero-or-witness, witness
re-verified in exact arithmetic; "no counterexample below N" stated with N explicit.

---

## Track C — Physics: from screen survivors to a no-dark-matter verdict

Order matters here: lensing and clusters come **before** heavy formal investment, so
families that die on known hard cases die cheaply.

**P1. Lensing gate at GPU speed.**  Extend the screen with synthetic lensing controls
(point, disk, and gas-dominated profiles): the same frozen law that flattens curves
must produce the observed deflection excess with zero per-object mass.  A
nonrelativistic law needs a declared lensing prescription (e.g., lensing traces the
same effective potential); the prescription is part of the candidate, not a free pass.
*Output:* screen v2 receipt with a lensing criterion column over the full family;
survivors-of-both counted; families that flatten curves but fail lensing named.

**P2. Cluster gate.**  Synthetic hydrostatic cluster controls (declared gas
density/temperature profiles).  This is where MOND-like laws historically fail; the
gate exists to find that out per family at screen speed.
*Output:* cluster criterion column; per-family verdict including, if true, the sealed
negative "no candidate in this grammar passes clusters," with exact margins — that
negative would itself be a scientific deliverable of the grammar.

**P3. External-field-capable grammar (screen v2).**  The current grammar is pointwise
`nu(y)`.  Add candidates with environment dependence (external-field-effect analogs,
gradient-direction dependence, nonlocal kernels with declared finite parameterization).
The EFE is the distinguishing testable signature separating modified-law families from
particle dark matter on synthetic controls.
*Output:* grammar v2 ordinal space with EFE test columns; measured pass/fail structure
of EFE vs non-EFE families.

**P4. Covariant completion adapter.**  Map each Pareto survivor family to typed
covariant actions in the existing IR (k-essence / Aether / Horndeski lanes), derive
the quasistatic weak-field limit, and check it reproduces the family's `nu`.
*Output:* per-family: either a hash-bound typed action with a derived matching
weak-field limit, or a named obstruction; this is the bridge from the screen into the
existing covariant-first machinery.

**P5. Family-level formal certificates (the throughput fix).**  The formal tier at
minutes/candidate cannot absorb 4.5M survivors.  Generalize the quartic-campaign
pattern: prove ADM/hyperbolicity/energy conditions once per family, parameterized by
coefficients, with per-candidate membership reduced to interval checks.
*Output:* measured formal throughput lifted from minutes/candidate to one family
certificate covering ≥10^4 candidates, with certified coefficient-interval domains.

**P6. Solar-System precision gate.**  PPN gamma/beta per covariant completion against
Cassini bounds; the screen's `y = 10^6` probe is necessary but not sufficient.
*Output:* PPN column in the promotion registry with cited bounds; rejections carry
margins.

**P7. Interval-arithmetic hardening.**  Survivors near any threshold get an
outward-rounded interval pass so fp64 luck can never decide a verdict.
*Output:* interval receipt over a threshold-straddling sample; zero misclassifications
by construction.

**P8. Open one real observation (the eight commitments).**  Source registration,
split commitment, primary roots, authorization — operator decisions, not code.  Then
one no-refit trial of the best surviving family on real rotation-curve data under the
frozen protocol.
*Output:* the sealed authority packet plus one executed PASS/REJECT/BLOCK verdict on
real data — the first observation the project has ever opened.  A REJECT is a valid
and publishable output.

---

## Track D — Substrate

**D1. CI green.**  The 7 red system7 byte-authority lanes fixed (Codex mid-flight),
new discovery/screen suites wired into shards, and a fresh all-green clean-clone run.
*Output:* a new terminal 39/39-style receipt including the new shards.  This gates M1
and D2.

**D2. Land the release.**  PR #1 out of draft and merged; the 193-commit divergence
closed.
*Output:* merged `main` with the release receipt; every later epoch builds from it.

**D3. Actually spend the LLM budget.**  $14.60 of $500 used to date, yet proposal
quality is the system's weakest link.  LLM-proposed grammar extensions and
failed-proof-packet analyses, quarantined, with lineage — never marking gates.
*Output:* N reviewed proposal packets per epoch, each adopted-with-tests or rejected
with a recorded reason; adoption rate tracked on the dashboard.

**D4. Two-machine scale-out.**  The durable two-host campaign passed its 6-hour
logical test; run it on two physical machines with the GPU box as the screen worker.
*Output:* a physical two-host epoch receipt with the same recovery guarantees.

---

## Sequencing

1. **D1 → D2** unblock everything (CI green, release landed).
2. **A1 + A2** make the system continuous; **M4** closes the math loop inside it.
3. **P1 + P2** immediately, at screen speed — cheap physics falsification first.
4. **M1 → M2** and **P4 → P5** are the two deep-capability climbs; they run in
   parallel tracks.
5. **M5** turns math output into defensible independent-discovery claims.
6. **P8** is the single highest-stakes step and is deliberately last: it is consumed
   once per candidate family, and only after lensing, clusters, formal, and PPN gates
   have filtered the family to its best survivor.

## Claim discipline (unchanged, binding on every task)

Corpus absence is never novelty.  Survival is never proof.  A restricted domain is
never a global claim.  Kernel verification happens only in the kernel.  No invisible
mass as target or rescue.  Sealed data opens once, no refit.  Negative receipts are
deliverables.
