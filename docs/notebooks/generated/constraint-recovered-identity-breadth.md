# Two recovered identities: arithmetic replay and an independent Lean check

> This notebook is generated from a fail-closed checked receipt. It documents two
> bounded synthetic recoveries; it does not establish general discovery or novelty.

## Evidence binding

The source receipt is `runs/math/constraint-recovered-identity-breadth-lean-bridge/receipt.json` with content SHA-256 `9dd811f41264f98dd733b57d7feea580699990bf00c5208fa8adc0e5a913cec4` and canonical file SHA-256 `b5077b15dfff126c6db4e87b86d3a9cf65d15475098f43ac8b99ec5fc7d4b03a`. Its terminal decision is `pass_two_recovered_identities_replayed_and_quartic_checked_by_real_lean_kernel`.

The receipt binds two recovered worlds, 14 lineage candidates, two exact symbolic certificates, two integer replays, one successful kernel theorem, and one rejected false control.

## 1. Quartic identity

The recovered expanded expression is **x^4 + 2x^3 - x - 30**. The independent factorized form is **(x - 2)(x + 3)(x^2 + x + 5)**.

First, (x - 2)(x + 3) = x^2 + x - 6. Multiplying its coefficient vector `[-6, 1, 1]` by `[5, 1, 1]` gives:

- constant: (-6)(5) = -30
- x: (-6)(1) + (1)(5) = -1
- x^2: (-6)(1) + (1)(1) + (1)(5) = 0
- x^3: (1)(1) + (1)(1) = 2
- x^4: (1)(1) = 1

Thus the constant-first vector is `[-30, -1, 0, 2, 1]`, exactly the recovered coefficient vector. This replay used integer additions and multiplications only: zero floating-point operations.

## 2. Partial-fraction identity

Start from **3/(x + 2) - 2/(x + 5) + 5/(x + 7)** over the common denominator **(x + 2)(x + 5)(x + 7)**. Its three numerator contributions are:

- 3(x + 5)(x + 7) = 3x^2 + 36x + 105
- -2(x + 2)(x + 7) = -2x^2 - 18x - 28
- 5(x + 2)(x + 5) = 5x^2 + 35x + 50

Adding them yields **6x^2 + 53x + 127**. Expanding the common denominator yields **x^3 + 14x^2 + 59x + 70**. Therefore the result is **(6x^2 + 53x + 127)/(x^3 + 14x^2 + 59x + 70)**.

The exact integer replay produced numerator coefficients `[127, 53, 6]` and denominator coefficients `[70, 59, 14, 1]`. Equality is asserted only on the regular domain; the excluded points are `x = -7, -5, -2`.

## 3. Independent Lean kernel check

Lean 4.33.0 checked `Invariant.constraintRecoveredQuarticIdentity` with exit code 0. The theorem executes the same constant-first `List Int` convolution and closes by kernel reduction with `rfl`. The dependency audit closed over `Invariant.recoveredPolyAdd, Invariant.recoveredPolyMul, Invariant.recoveredPolyScale`; no `sorry` or user axiom was admitted.

This Lean theorem checks the quartic coefficient identity independently of the recovery campaign's SymPy certificate. The partial-fraction identity is independently replayed by the bridge's closed integer arithmetic, but is not claimed here as a second Lean theorem.

## 4. Deliberate failure

The negative control changed the constant coefficient from `-30` to `-29`. Lean returned `block_lean_process_failure` with a nonzero exit code, and the result was rejected before receipt promotion. This demonstrates that the bridge records a failed proof rather than silently accepting or rewriting it.

## Boundary of the result

These are two synthetic identities recovered inside a preregistered exact grammar. The evidence establishes exact replay for both and a real Lean check for the quartic. It does not establish general formula discovery, mathematical novelty, scientific truth, physics truth, or promotion eligibility.

Report content SHA-256: `1be3f5355747cd749ccd14e1632aef73cb1a3446e9395075ba9e16b8836b4773`.
