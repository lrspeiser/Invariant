# Erdős–Straus creative shadow experiment

Date: 2026-08-24

Campaign: `erdos-straus-creative-shadow-2026-08-24-001`

Receipt: `71bbdb5014738e6198b28d99649481e0d092fb59025d06702674784e84c9c27c`

## Result in one paragraph

Invariant successfully ran an LLM-to-typed-search-to-GPU-to-exact-check pipeline on a bounded
shadow of the Erdős–Straus conjecture. Claude Opus 4.6 returned 12 ideas: 11 compiled into the
experiment's search DSL and one malformed idea was retained for repair. The system expanded 117
direct parameter pairs into 1,051 local mutations, tested 296 pairs outside the old fixed schedule,
and recovered 173 of the old schedule's 1,261 hard-tail cases at `n <= 100,000,000`. All 173 were
independently rechecked with a separate CPU integer identity. This is operationally useful, but it
is **not evidence of a new formula or proof**. A stringent degree-preserving random control matched
or beat the LLM pairing in 24 of 32 trials. The defensible result is that the LLM-selected offset
*scales* were useful; the exact LLM pairings were not superior to carefully matched random
recombination.

## Scientific boundary

The conjecture asks whether every integer `n >= 2` has positive integers `x <= y <= z` such that

\[
\frac{4}{n}=\frac{1}{x}+\frac{1}{y}+\frac{1}{z}.
\]

It remains open. Salez reported verification through `10^17` in 2014, and a 2025 preprint reports
verification through `10^18`; therefore this experiment's `10^8` sweep is not a new computational
bound. See [Salez (2014)](https://arxiv.org/abs/1406.6307),
[Mihnea–Bogdan (2025)](https://arxiv.org/abs/2509.00128), and the broader solution-counting
literature in [Elsholtz–Tao](https://arxiv.org/abs/1107.1010).

The project's famous-open-problem gate remained closed: 0 of the required 3 independent level-5
process successes have been obtained. This was a finite mechanism experiment, not an authorized
solution attempt. No proof, novelty, or serious claim was released.

## What ran

The exact finite pipeline was:

1. Four elementary parametric identities handled the easy residue classes symbolically.
2. The remaining `n = 1 (mod 12)` class used `x = floor(n/4) + 1 + dx`, `a = 4x-n`, and `b = nx`.
3. For `y = ceil(b/a) + t`, the GPU tested the exact divisor condition `d = ay-b > 0` and `d | by`.
4. Every hit reconstructed `z = by/d`; the independent CPU check verified
   `n(yz+xz+xy) = 4xyz` with Python big integers.
5. Any unresolved value went to the pre-existing complete CPU divisor search over factors of
   `b^2`.

This avoids brute-forcing all triples `(x,y,z)`. It uses algebra to turn a huge three-dimensional
space into exact vectorized `(dx,t)` screens, and reserves expensive exhaustive work for a small
tail.

## Exact accounting

| Quantity | Result |
|---|---:|
| LLM ideas returned | 12 |
| Executable ideas | 11 |
| Non-executable ideas retained | 1 |
| Direct parameter pairs | 117 |
| Mutated parameter pairs | 1,051 |
| Creative pairs outside old schedule | 296 |
| Per-idea calibration lane tests | 7,486,679 |
| Old baseline GPU lane tests | 104,839,060 |
| Creative-tail lane tests | 344,279 |
| Fixed-lane creative reference tests | 373,256 |
| Fixed-lane random-control tests | 35,832,576 |
| Total exact modular lane tests | 148,875,850 |
| Denominators finitely covered, `2 <= n <= 10^8` | 99,999,999 |
| Unsolvable values found in that finite range | 0 |

The full finite sweep ended `NO_UNSOLVABLE_N_IN_RANGE`. This says nothing about `n > 10^8` and
does not decide the conjecture.

## How the RTX 5090 helped

The machine exposed an NVIDIA GeForce RTX 5090 with 32 GB VRAM and CUDA 12.9. CuPy and Numba could
use compute capability 12.0; the installed PyTorch and JAX builds were CPU-only, so the campaign
used CuPy.

On the identical `n <= 10^7` control:

| Backend | Time | Throughput |
|---|---:|---:|
| NumPy CPU | 44.002 s | 227,261 denominators/s |
| RTX 5090 / CuPy | 1.924 s | 5,197,021 denominators/s |

The measured speedup was **22.87×**, with identical lane counts, decisions, and unresolved sets.
The final `n <= 10^8` GPU sweep took 12.033 seconds at 8,310,205 denominators/s.

For the hard class alone, the old GPU schedule resolved 8,332,072 of 8,333,333 values and left
1,261 for exhaustive CPU completion. The creative mutations resolved 173 of those 1,261 cases,
reducing that fallback tail by **13.72%** to 1,088.

## What the LLM proposed

The LLM self-assessed 4 ideas as `known_rewrite`, 6 as `cross_domain_synthesis`, and 2 as
`proposed_new_construction`. These labels were stored but never used to prune ideas and do not
establish origin or novelty.

| Idea | Basis | Self-assessed origin | Calibration hard cases resolved | Status |
|---|---|---|---:|---|
| ES-RC-001 | residue cover | known rewrite | 33,404 | executed |
| ES-CF-002 | continued fraction | cross-domain synthesis | 35,933 | executed |
| ES-DG-003 | descent graph | cross-domain synthesis | 49,304 | executed |
| ES-DP-004 | divisor pair | known rewrite | 38,017 | executed |
| ES-LT-005 | lattice transform | cross-domain synthesis | 26,166 | executed |
| ES-PA-006 | polynomial ansatz | known rewrite | 69,815 | executed |
| ES-GO-007 | greedy offset | known rewrite | 67,788 | executed |
| ES-MS-008 | modular sieve | cross-domain synthesis | 31,944 | executed |
| ES-DC-009 | residue cover | proposed new construction | 21,640 | executed |
| ES-HL-010 | lattice transform | cross-domain synthesis | 20,810 | executed |
| ES-GF-011 | generating function | cross-domain synthesis | — | retained for repair |
| ES-TP-012 | polynomial ansatz | proposed new construction | 25,264 | executed |

The resolved counts overlap and are not additive. This historical campaign used `ESDSL1`, which
compiles all admitted conceptual bases into the same Cartesian offset-search kernel. Consequently,
labels such as “continued fraction” or “lattice transform” describe the proposal rationale, but only
the numeric offsets and moduli were machine-executed. This remains a limitation of the historical
result, not a semantic proof that those mechanisms worked.

## Versioned ESDSL2 basis semantics

`ESDSL2` is now implemented as a separate, strict compiler version. It does not reinterpret any
`ESDSL1` expression or change the 173 historical recoveries. Instead, every basis has its own field
schema and executable schedule operator:

| Basis | ESDSL2 operator | Positive-control pairs |
|---|---|---:|
| continued fraction | ordered convergents | 4 |
| descent graph | bounded graph reachability | 6 |
| divisor pair | exact factor pairs | 6 |
| greedy offset | diagonal-cost budget selection | 5 |
| lattice transform | integer affine-lattice image | 4 |
| modular sieve | linear-congruence-filtered product | 8 |
| polynomial ansatz | integer polynomial parameter map | 4 |
| residue cover | complementary residue lifts | 6 |

The eight positive controls produce eight distinct sealed schedules. Each has a structural control
with the same basis, grammar field count, direct-pair count, and exact verifier-lane budget, but a
changed typed parameter and a different schedule. Degenerate matrices, empty congruence slices,
out-of-range polynomial images, noncanonical fields, duplicate fields, and oversized schedules fail
closed. Every positive schedule also produces witnesses that pass the existing independent exact
integer checker on a bounded control range.

The runtime receipt embeds and recomputes this compiler contract. Its hardcoded compatibility
vector requires the legacy expression
`ESDSL1|basis=lattice_transform|x=0,2,65|t=0,7|m=24,120` to retain its old parse shape and exact
Cartesian order. A newly preregistered campaign can set `proposal_dsl_version` to `ESDSL2`, causing
the core Claude prompt to expose all eight strict forms; this exhausted historical campaign remains
explicitly pinned to `ESDSL1`. These are compiler controls, not evidence that any basis is novel,
causal, or better at the conjecture.

## A concrete example

ES-TP-012 proposed the recipe

```text
ESDSL1|basis=polynomial_ansatz|x=3,9,27,81,243|t=1,4,13|m=9,36,108
```

and self-labeled it `proposed_new_construction`, citing a possible 3-adic synthesis. The deterministic
mutation operator expanded `(dx,t) = (81,13)` to the nearby pair `(83,11)`. That pair found a
witness for a value the old fixed GPU schedule had left to CPU fallback:

```text
n = 398161
x = 99624
y = 118407150
z = 1240566393036600
```

The independent exact check reduced both sides of the cross-multiplied identity to the same integer:

```text
4*x*y*z                  = 58535846929895654223858240000
n*(y*z + x*z + x*y)      = 58535846929895654223858240000
```

This demonstrates the working idea-to-mutation-to-GPU-to-independent-check chain. It does **not**
validate the proposal's 3-adic rationale, show that the witness is new, or prove a residue family.

The complete set of 173 creative-tail witnesses was CPU-verified and sealed as
`9bf44e7e0b8eb6a81d8c2119757d4bc9f494492c7f4eea4a40e51d1eaec0177b`.

## Which ideas were linked to the recoveries?

The refreshed receipt now records the first exact successful pair in the fixed canonical search
order for every one of the 173 recoveries. Each record binds the denominator and exact witness to
the winning pair, every direct parent pair, mutation delta, idea ID, basis label, role, and LLM
origin assessment. Recomputing any winner from its recorded pair reproduces its exact witness.

| Idea | Declared basis | Origin assessment | Linked recoveries |
|---|---|---|---:|
| ES-TP-012 | polynomial ansatz | proposed new construction | 70 |
| ES-MS-008 | modular sieve | cross-domain synthesis | 51 |
| ES-CF-002 | continued fraction | cross-domain synthesis | 33 |
| ES-DC-009 | residue cover | proposed new construction | 18 |
| ES-HL-010 | lattice transform | cross-domain synthesis | 1 |

Twenty-seven recovered denominators had more than one direct mutation parent, but those overlaps
were within the same idea and basis; none crossed idea or basis boundaries. The five linked counts
therefore sum to 173. This is deterministic **first-success lineage**, not causal credit or evidence
that the declared conceptual rationale is correct. In particular, the historical ESDSL1 evidence
still executes every basis through the same numeric offset kernel. The attribution can now support
a future preregistered ESDSL2 comparison without retroactively reinterpreting that run.

For `n = 398161`, the record is exact: ES-TP-012 direct pair `(81,13)`, mutation `(+2,-2)`,
winning pair `(83,11)`, and the witness printed above.

## Did it beat random search?

Three increasingly strict 32-trial controls each used 296 pairs and the same exact evaluator:

| Control | Median recovered | Range | Trials at least 173 | One-sided randomization p |
|---|---:|---:|---:|---:|
| Uniform over the full declared offset domain | 38 | 25–53 | 0/32 | 0.0303 |
| Same LLM-selected offset supports | 158 | 145–170 | 0/32 | 0.0303 |
| Exact `x`/`t` marginal frequencies; pairings rewired | 174 | 168–180 | 24/32 | 0.7576 |

The strengthened comparison disables early stopping for both the creative reference and every
control schedule. Each executes exactly `296 * 1261 = 373,256` pair-by-tail lanes on the same RTX
5090 evaluator with the same fail-closed 30-second ceiling. The creative reference took 0.182917
seconds; across all 96 controls, observed times ranged from 0.089636 to 0.244319 seconds. These are
reported outcomes, not a claim that wall times are equal. All runs passed the common ceiling, and
the fixed-lane result agrees exactly with the historical first-success witnesses. The three control
distributions and p-values are unchanged.

The first two comparisons show that the LLM portfolio concentrated computation in useful regions.
The third comparison is the fairest test of its precise combinations and does not support an
advantage: random rewiring with the same marginal frequencies was slightly better on average.

Therefore this run is evidence of **computational complementarity**, not evidence that the new
system is generally “more creative” than the old one. A stronger creativity claim requires
pre-registered controls across multiple unseen problems and independent machines.

## Provider and failure accounting

Four provider message calls completed across the experiment attempts. The first transport attempt
was rejected before any message completed because its fixed-slot JSON grammar was too large. A
second attempt completed three creative calls, then the per-candidate critic grammar was rejected;
because partial output was intentionally in-memory only, those three responses were discarded and
their exact token usage is not recoverable. They are still counted against the four-call problem
budget.

The one retained provider call used 1,416 input and 5,093 output tokens (6,509 total), returned the
12 ideas above, and stored no credential. The final run used deterministic admission/repair routing
instead of claiming that an LLM critic ran.

This exposed a real reliability gap. It is now fixed prospectively in `run-live`: every scheduled
message dispatch is hash-chained and fsynced before transport, and the complete response is fsynced
before parsing or contract validation. Resume replays a completed response only when the newly
generated request has the exact original body hash, and every dispatch counts against the fixed
call budget even if the response is malformed or the process stops. The private journal stores
header names but no header values, credential, or private path in a public receipt.

Two offline provider-boundary rehearsals verify the recovery behavior without spending another
call. One stops the process after a valid response is journaled but before it is parsed, then
recovers the idea with zero network requests. The other runs one proposer and two one-candidate
critic batches, deliberately malforms the last critic response, retains both candidate lineages,
and resumes all three paid slots with zero requests. The historical three discarded responses were
never journaled and therefore cannot be reconstructed; the refreshed live receipt preserves that
failure accounting instead of pretending otherwise. LLM critic batching remains disabled in this
historical campaign because its four-call budget is already exhausted.

## What should happen next

1. Exercise the journaled critic batches in a newly preregistered, separately budgeted campaign;
   do not reopen this exhausted four-call campaign.
2. Run the next campaign with `proposal_dsl_version=ESDSL2`; compare each emitted basis schedule
   with its pair-count-, grammar-field-, and verifier-budget-matched structural control.
3. Pre-register train/tail splits and all controls before generation, then repeat on rotating unseen
   problems and a second machine.
4. If a candidate suggests an actual residue-class identity, move it through exact arithmetic,
   CAS, SMT, interval checks where applicable, Lean, automated prior art, and named human review.
5. Keep the famous-open-problem gate closed until the required three independent level-5 process
   successes exist.

## Bottom line

The system worked as an auditable research instrument: it generated diverse suggestions, labeled
their possible origins, retained a malformed branch, compiled 11 ideas, expanded them, used the
5090 to screen a large exact space, independently verified every promoted hit, and tested the result
against progressively stronger controls. It produced a useful engineering improvement—173 fewer
CPU fallbacks—but no new theorem, formula, verification bound, or demonstrated superiority of the
LLM's exact idea combinations.
