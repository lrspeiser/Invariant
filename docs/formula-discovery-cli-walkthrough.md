# Formula Discovery CLI walkthrough

This walkthrough exercises the public Formula Discovery boundary with two caller-owned JSON
problems. Both problems make the same four exact observations available to the solver. The first
provides two consistent heldout observations; the second deliberately changes one heldout value.
No formula for either case is encoded in production source.

The examples are:

- `examples/formula-discovery/pass-exact-polynomial.json`
- `examples/formula-discovery/reject-heldout-counterexample.json`

## The JSON input

Each file is one closed `sigma-formula-discovery-problem-1.0` object. Its important fields are:

- `solver`: the caller declares the finite basis `1, x, x**2, x**3`.
- `constraints`: four exact rational point/value rows used for synthesis.
- `validation`: separate exact point/value rows that are not used during synthesis.
- `limits`: per-job caps no larger than the system caps.
- `proof`: `none` here; recurrence jobs may instead request the supported induction certificate.

Every rational is written as a reduced integer numerator and positive denominator. Floating-point
JSON numbers, duplicate keys, unknown schema fields, unsupported solvers, and exceeded budgets fail
closed.

## Run the PASS example

From the repository root in PowerShell:

```powershell
$env:PYTHONPATH = "src"
python -m sigma_theory_compiler.formula_discovery_cli run `
  --problem examples/formula-discovery/pass-exact-polynomial.json `
  --result work/walkthrough-pass-result.json `
  --report work/walkthrough-pass-report.md
$LASTEXITCODE
```

The expected exit code is `0` (`PASS`). The exact solver uniquely recovers
`x**3 - 2*x + 5`, and both disjoint validation rows agree. Run the read-only replay boundary:

```powershell
python -m sigma_theory_compiler.formula_discovery_cli validate `
  --problem examples/formula-discovery/pass-exact-polynomial.json `
  --result work/walkthrough-pass-result.json `
  --report work/walkthrough-pass-report.md
```

Validation recomputes the discovery result, Sigma candidate, provenance, evaluation outcomes,
metric receipts, Pareto result, top-level seal, and Markdown text. It does not rewrite either file.

## Run the REJECT example

```powershell
python -m sigma_theory_compiler.formula_discovery_cli run `
  --problem examples/formula-discovery/reject-heldout-counterexample.json `
  --result work/walkthrough-reject-result.json `
  --report work/walkthrough-reject-report.md
$LASTEXITCODE
```

The expected exit code is `10` (`REJECT`). Synthesis still recovers the same unique expression, but
the second heldout row claims that the value at `x = 4` is `62`. Exact evaluation returns `61`, so
the sealed counterexample records residual `-1`. This means the candidate is incompatible with the
caller's heldout evidence. It does not mean that the synthesis engine crashed, nor does it authorize
changing the formula after seeing the heldout value.

The candidate remains in the audit trail. Its `hard_structure` gate passes, its `hard_validation`
gate rejects, and it is excluded from exact Pareto ranking. A favorable expression-size metric can
never compensate for a failed hard gate.

## Outputs and provenance

`--result` is a sealed JSON envelope containing:

1. the complete Formula Discovery Job receipt;
2. the Sigma Core candidate and its original discovery-candidate provenance input;
3. evaluation-ladder stages and hard-gate outcomes;
4. exact metric receipts and Pareto explanations when a candidate exists; and
5. conservative claim boundaries: no truth, novelty, promotion, or law claim.

`--report` is a deterministic Markdown rendering of the same evidence. The CLI stages and flushes
both outputs in their destination directories, publishes only to new paths, never overwrites an
existing path, and rolls back a partially published pair. Re-running `run` with the same output
paths is therefore an operational error; use `validate` for replay.

## Exit codes

| Code | Decision | Meaning |
|---:|---|---|
| `0` | PASS | Candidate passed exact heldout validation and every registered hard gate. |
| `10` | REJECT | Exact evidence falsified the candidate or its constraints were inconsistent. |
| `20` | BLOCK | The valid problem could not uniquely produce a candidate, such as deficient rank. |
| `30` | SCHEMA_ERROR | Input JSON or its closed problem schema was invalid or unsupported. |
| `40` | OPERATIONAL_ERROR | Immutable publication or result/report replay failed. |

These are bounded mathematical outcomes. PASS does not establish a scientific law or novelty, and
REJECT applies to the submitted problem and evidence rather than to every possible formula family.
