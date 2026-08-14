# Exact active formula identifiability

The active-identifiability boundary prevents the formula finder from guessing when two or more
registered formulas agree on all current public data. It works over a bounded, caller-declared
finite formula family and exact integer/rational arithmetic.

## Contract

An initial problem declares:

- a closed exact hypothesis family;
- current public point/value observations;
- an ordered legal query space and a hard query-evaluation budget; and
- a SHA-256 commitment to the target hypothesis and a secret nonce.

The initial run removes hypotheses contradicted by public data. When at least two remain, it seals
their common observation signature as an ambiguity witness and returns `BLOCK`. It evaluates only
the budgeted prefix of legal queries. Each defined query partitions the survivors by their exact
predicted answer. The selected query minimizes the largest remaining partition, then maximizes
the number of partitions, then preserves caller order.

The target commitment is opaque during this selection. An answer later opens the commitment and
binds the problem, initial result, query receipt, exact answer, target hypothesis, and nonce. Resume
returns `PASS` only if the answer leaves exactly the opened target. The resulting Sigma Core
candidate and proof bind every public observation, the active answer, the registered hypothesis,
and all predecessor hashes.

## Public use

Start from the checked example:

```powershell
$env:PYTHONPATH = "src"
python -m sigma_theory_compiler.active_formula_identifiability_cli start `
  --problem examples/active-identifiability/ambiguous-square.json `
  --result initial.json `
  --report initial.md
```

This example must return `BLOCK` and propose `n=2`: both `n` and `n**2` match the public `n=0`
observation, while their query predictions are exactly 2 and 4. Its preregistered target opening
is hypothesis `formula.square` with nonce `public-example-nonce-001`.

Use `build_query_answer(...)` to construct the sealed answer from that opening, then run the
`resume` command with the problem, initial result, and answer. Both stages have read-only
`validate-start` and `validate-resume` commands. Result and Markdown paths must be new; the CLI
never overwrites existing output.

## Decisions and stop conditions

- `PASS`: public data already identify one hypothesis, or a provenance-bound answer reduces the
  sealed ambiguity class to the opened target alone.
- `REJECT`: public observations match no registered hypothesis, or an answer contradicts the
  surviving family/target.
- `BLOCK`: multiple hypotheses remain. A separating query may be present. The typed reasons
  distinguish a proposed query, budget exhaustion, and no identifiable legal query.
- Integrity/schema errors: malformed, unsafe, duplicated, unsealed, or provenance-inconsistent
  values fail the closed API or replay validator.

The 13 focused tests cover the preregistered target-isolated PASS, initial exact ambiguity BLOCK,
illegal/repeated/uninformative query rejection, answer and candidate tamper rejection,
deterministic initial/resumed replay, budget stop, terminal no-query BLOCK, strict problem
negatives, immutable publication, Markdown replay, and public CLI exit codes.

## Claim boundary

The system proves identifiability only inside the declared finite family. It does not prove that
the selected expression is unique among all mathematical expressions, scientifically true,
novel, or globally valid. It currently performs one active-query round; multi-round decision
trees and automatic experiment execution remain separate build-outs.
