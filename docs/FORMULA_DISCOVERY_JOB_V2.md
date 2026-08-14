# Formula Discovery Job v2

Formula Discovery Job v2 extends the public, immutable Formula Discovery boundary to four
bounded exact problem classes. It does not claim to exhaust an unbounded formula space.

## Public contract

Problems use the closed schema `sigma-formula-discovery-problem-2.0`. Results use
`sigma-formula-discovery-result-2.0`, and the public file/CLI envelope uses
`sigma-formula-discovery-public-result-2.0`. V1 problems and results remain replay-compatible.

Every v2 job declares:

- one to six integer or rational variables;
- zero or more exact nonzero domain premises;
- one registered solver adapter;
- exact public synthesis constraints and disjoint exact holdout rows;
- a proof contract; and
- caller limits no larger than the fixed system caps.

The parser accepts only integers, registered symbols, arithmetic operators, division, and
bounded integer powers. It does not evaluate Python, function calls, floating-point values,
network data, databases, or external processes.

Four directly runnable problems are under `examples/formula-discovery-v2/`. For example:

```powershell
$env:PYTHONPATH = "src"
python -m sigma_theory_compiler.formula_discovery_cli run `
  --problem examples/formula-discovery-v2/rational-domain.json `
  --result result.json `
  --report report.md
```

Both output paths must be new. The corresponding `validate` command performs read-only exact
replay of the problem, sealed JSON, nested orchestration, Pareto evidence, and Markdown report.

## Measured class coverage

| Class | Exact synthesis | Independent acceptance condition | Measured control |
|---|---|---|---|
| Multivariate polynomial | Exact rank classification over a caller basis | Disjoint multivariate rational holdout rows | Recovered `x**2 + 2*x*y + 3*y + 1` |
| Rational function with domain | Linearized numerator/denominator solve with one fixed denominator coefficient | Holdout equality plus denominator nonzero at every checked point | Recovered `(x + 1)/(x - 1)` and generated premise `x - 1 != 0` |
| Nonlinear algebraic parameterization | Exhaustive exact finite parameter grid with optional implicit parameter equations | Exactly one distinct expression survives constraints and holdout | Recovered `4*x + 3` from `a**2*x+b` under `a**2-4=0`; both `a=-2` and `a=2` collapse to one expression |
| Higher-order recurrence | Exact coefficient identity and initial-value solve for order 2 through the declared cap | Held-out sequence values and sealed recurrence residual/initial checks | Recovered `n**2 + 1` from an order-two recurrence |

The focused v2 suite contains 16 tests. The adjacent Formula Job suite contains 56 tests and
also exercises v1 jobs, orchestration, public CLI publication/replay, the walkthrough, and the
existing Lean translation boundary. Controls include exact counterexamples, undefined rational
holdouts, inconsistent and underdetermined grids, unsafe expressions, budgets, immutable output,
deterministic replay, and resealed tamper rejection.

## Decision semantics

- `PASS`: one candidate was constructed, all independent holdout rows passed exactly, and every
  requested v2 proof certificate closed.
- `REJECT`: exact constraints are inconsistent, a bounded parameter grid has no match, a
  candidate is undefined on a holdout, or an exact holdout counterexample exists.
- `BLOCK`: the problem is malformed/unsupported/over budget or exact public constraints leave
  multiple candidates.
- `SCHEMA_ERROR`: the public CLI received a job that did not cross a registered problem schema.

Only candidates passing both orchestration hard gates enter the exact Pareto layer. A PASS is not
a scientific-law, novelty, promotion, or unbounded-completeness claim.

## Exact remaining blockers

V2 is a coherent bounded architecture, not the final universal formula finder:

1. Nonlinear algebraic synthesis is exhaustive only over a caller-declared finite rational grid;
   general polynomial-system solving, positive-dimensional solution varieties, and branch/domain
   decomposition remain unsupported.
2. The bounded Proof/Lean v2 acceptance suite now generates and checks one rational identity with an
   explicit nonzero-denominator premise and one genuine order-two recurrence theorem in real Lean
   4.33, with single-coefficient false controls rejected. General translation of arbitrary v2
   rational candidates and higher-order closed forms remains open; see
   [`FORMULA_DISCOVERY_PROOF_V2.md`](FORMULA_DISCOVERY_PROOF_V2.md).
3. Recurrence coefficients and forcing may be rational expressions whose cleared residual is
   polynomial in one integer index. Multisequence, nonlinear, variable-order, and partial
   recurrences remain unsupported.
4. The candidate object must be an explicit expression. Pure implicit relations, inequalities,
   quantified theorem statements, and multivalued algebraic functions need a theorem/candidate IR
   beyond the Formula artifact used here.
5. V2 classifies insufficient information as BLOCK but does not yet choose or execute the next
   information-gaining experiment.

Those limits are surfaced as typed BLOCKs or documented scope; none is silently treated as zero,
uniqueness, proof, or novelty.
