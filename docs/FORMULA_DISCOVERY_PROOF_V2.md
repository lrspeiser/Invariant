# Formula Discovery Proof/Lean v2

Proof/Lean v2 is a six-execution acceptance suite for the public Formula Discovery product. It
generates three theorem sources from closed, hash-bound case specifications, executes them with the
registered portable Lean 4.33 kernel, and then executes one minimally false source for each case.
An overall `PASS` is emitted only when all three theorems check and all three false controls return a
nonzero Lean exit code.

The checked Windows receipt is
[`runs/math/formula-discovery-proof-v2/receipt.json`](../runs/math/formula-discovery-proof-v2/receipt.json),
sealed by content SHA-256 `df4702073046170e01a535b271c924c3ebc0cdc295ce7b29c6022a82d915a2f7`.

## The three kernel cases

| Case | Generated theorem | Proof strategy | False control |
|---|---|---|---|
| Rational regular-domain identity | `((x + 3) * (x - 2)) / (x - 2) = x + 3` under the explicit premise `x - 2 ≠ 0` | Rational cancellation using only the registered `Rat.mul_div_cancel` premise | Change the final coefficient `3` to `4` |
| Higher-order recurrence | The order-two sequence with bases `u 0 = 2`, `u 1 = 5`, and `u(n+2) = u(n+1) + u(n)` satisfies its generated recurrence for every `n` | Definitional reduction of a genuine two-prior-term recurrence | Change the coefficient of `u n` from `1` to `2` |
| Quantified non-identity | `∀ n : Nat, 3*n + 2 ≠ 3*n + 3` | Strict-order contradiction discharged by registered Presburger arithmetic (`omega`) | Change the right offset `3` to `2` |

For every pair, whitespace tokenization of the positive and negative sources has the same length and
exactly one differing token. The target name, import, proof script, dependency audit, and execution
environment remain identical. This makes the negative result evidence about the false coefficient,
not an incidental compiler or configuration difference.

Generated sources are rejected before execution if they contain `sorry`, `axiom`, `admit`,
`Classical.choice`, or `False.elim`. Each positive source reports a closed allowed-premise manifest,
and the adapter rejects any dependency outside it.

## Public CLI: PASS and REJECT in one receipt

Run all six executions and immutably publish the sealed result:

```console
sigma-formula-discovery proof-v2-run --result proof-v2.json
```

The command exits `0` and prints an overall `PASS` only after the three positive checks. Inside the
same receipt, each `false_control.outcome` is `REJECT`, with
`decision = block_lean_process_failure`, `attempted = true`, `timed_out = false`, and
`nonzero_exit_code = true`.

Validate the stored receipt and repeat all six real-kernel executions:

```console
sigma-formula-discovery proof-v2-validate --result proof-v2.json
```

Use `--lean PATH` with either command to select a registered Lean 4.33 executable explicitly. The
path is used only to launch the owned child process and is never persisted.

## Python API

```python
from pathlib import Path

from sigma_theory_compiler.formula_discovery_proof_v2 import (
    build_proof_v2_receipt,
    validate_live_proof_v2_receipt,
    validate_proof_v2_receipt,
    write_proof_v2_receipt,
)

receipt = build_proof_v2_receipt()
validate_proof_v2_receipt(receipt)       # sealed, path-free, no Lean process required
validate_live_proof_v2_receipt(receipt)  # repeats all six Lean executions
write_proof_v2_receipt(receipt, Path("proof-v2.json"))
```

Publication uses exclusive creation and refuses to overwrite an existing output. Static validation
reconstructs every generated source and premise manifest, checks nested seals and the pinned
toolchain identity, and rejects resealed semantic or host-path tampering. Live validation adds exact
replay of the three successful checks and three rejections. Path-sensitive compiler diagnostics from
expected failures are deliberately not persisted; the sealed negative receipt retains the source,
kernel identity, attempted/timed-out flags, and decisive nonzero-exit fact.

## Claim boundary

These are three bounded theorem shapes, not a general proof synthesizer. They establish that the
product can generate and independently kernel-check a rational identity with a real domain premise,
a higher-order recurrence theorem, and a quantified non-identity via a distinct proof strategy. They
do not prove novelty, scientific truth, arbitrary rational-expression translation, or arbitrary
higher-order recurrence closed forms.
