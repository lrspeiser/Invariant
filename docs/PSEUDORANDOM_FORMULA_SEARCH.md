# Pseudorandom formula search

Invariant now supports deterministic, random-looking traversal of an ordinal formula grammar
without replacement. It is intended to complement Occam ordering, LLM proposals, and
counterexample-guided exclusions—not replace them.

## Why a permutation instead of repeated random draws

Independent random draws waste compute on duplicates and cannot prove that a finite grammar was
exhausted. Materializing and shuffling a trillion ordinals would require prohibitive memory. The
new scheduler uses an eight-round keyed Feistel permutation over a power-of-two domain and cycle
walks back into the declared ordinal range.

For any declared size up to `2^64`, it provides:

- one-to-one coverage with no replacement;
- deterministic reproduction from a public seed;
- constant-memory random access by schedule position;
- exact restart from any position;
- complete coverage if all positions are visited;
- a clear claim boundary: it is a search scheduler, not a cryptographic primitive.

For trillion-scale GPU execution, Invariant permutes large chunks rather than asking Python to
schedule every formula. Each selected chunk contains contiguous ordinals that a CUDA kernel can
decode efficiently. A `2,127,732,389,840`-formula grammar divided into blocks of ten million has
`212,774` restartable work units, including one final partial block.

## Formula mapping

The permutation operates on ordinals, not text. Each problem owns a typed decoder:

- a polynomial decoder maps base-11 digits to four coefficients in `[-5,5]`;
- an exact linear-relation decoder selects one of 112 canonical primitive coefficient vectors;
- a monomial decoder selects one of 6,152 canonical primitive exponent vectors;
- a large production grammar can map ordinals to basis terms, signs, coefficients, transforms,
  and parameter cells.

Canonical decoding avoids syntactic duplicates. The exclusion ledger can reject certified ordinal
families or chunks before an expensive evaluator runs.

## Benchmark results

The sealed benchmark traversed all candidates for three already-known calibration problems in
pseudorandom order:

| Problem | Candidate space | Winning candidate | Winner visit position |
| --- | ---: | --- | ---: |
| Exact planted polynomial | 14,641 | `5 - 2x + 3x^2` | 3,952 |
| Real Archimedes force readings | 112 | `[1,1,-1,-1]` force balance | 87 |
| Anonymous NASA catalog columns | 6,152 | `x0^2*x1^-3*x2 = constant` | 1,208 |

All three winners were recovered after complete traversal. The Archimedes winner retained a
training mean absolute residual of `1/25 N` and a held-out residual of `1/50 N`. The NASA winner was
selected from 2,020 anonymous training rows and scored on 511 host-disjoint held-out rows.

The logical benchmark evaluated 20,905 formulas. It also sampled 10,000 distinct random-access
ordinals and 1,000 distinct GPU chunks from the 2.13-trillion space. It did **not** evaluate two
trillion formulas, and the receipt says so explicitly.

## What this test establishes

It establishes that the scheduler is collision-free on fully enumerated controls, reproducible,
restartable, capable of addressing a trillion-scale grammar, and correctly connected to three
formula decoders and evaluators.

It does not establish that pseudorandom order is more efficient or more creative. On structured
simple laws, Occam ordering can find the answer earlier. Random traversal is valuable as an
orthogonal exploration lane because it reaches structurally distant regions early and gives every
candidate a defined visit position. Production comparisons must allocate equal budgets to:

- Occam/complexity order;
- pseudorandom permutation;
- exclusion-guided frontier order;
- LLM-generated structural expansions.

The current benchmark problems and answers were already exposed, so every success is a known
rediscovery. No new formula or historical novelty is claimed.
