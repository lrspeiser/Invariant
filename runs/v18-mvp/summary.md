# Sigma Theory Compiler run

Exhaustive over the exact atom/operator/cost grammar below; not exhaustive over field theories or mathematics.

## Outcome

- Enumerated signed candidates: 234
- Rejected before covariant work: 216
- Static-sector survivors requiring a covariant lift: 18
- Fully validated theories: 0

> A requires_covariant_lift row is only a static-sector survivor. It is not a healthy relativistic theory and is not authorized for observational fitting.

## Static-sector survivors

| ID | Complexity | Coupling | Correction |
|---|---:|---:|---|
| `STC-0aaf33300b9a` | 4 | +0.1 | `q*(q + 2)/(q + 1)` |
| `STC-20786bbd4407` | 4 | +0.1 | `q*x/(x + 1)` |
| `STC-2d18d6a8e5eb` | 2 | +0.1 | `sqrt(q + 1) - 1` |
| `STC-5c53d4becebb` | 4 | +0.1 | `sqrt(q + z + 1) - 1` |
| `STC-601d4ac50ce0` | 4 | +0.1 | `q*(sqrt(x + 1) - 1)` |
| `STC-68db61523d40` | 1 | +0.1 | `q` |
| `STC-7e2211a62212` | 4 | +0.1 | `q + sqrt(q + 1) - 1` |
| `STC-805deaf32ddd` | 3 | +0.1 | `2*q` |
| `STC-8e9115fd2e64` | 4 | +0.1 | `sqrt(2*q + 1) - 1` |
| `STC-9622cafac0ea` | 4 | +0.1 | `(q*z + q + z)/(z + 1)` |
| `STC-c8bb402e3a49` | 4 | +0.1 | `(q + 1)**(1/8) - 1` |
| `STC-ca32cbf6d8d1` | 4 | +0.1 | `sqrt(q + x + 1) - 1` |
| `STC-d06bce18e314` | 3 | +0.1 | `(q + 1)**(1/4) - 1` |
| `STC-d8fd3439c08d` | 4 | +0.1 | `(q*x + q + x)/(x + 1)` |
| `STC-e3ae4abb58ac` | 4 | +0.1 | `q + sqrt(x + 1) - 1` |
| `STC-e6057fe81765` | 3 | +0.1 | `q + z` |
| `STC-f493ca1ce3da` | 4 | +0.1 | `z + sqrt(q + 1) - 1` |
| `STC-fc4ee7a56d79` | 4 | +0.1 | `q + sqrt(z + 1) - 1` |

## Gate semantics

- `pass`: this implementation produced positive evidence for the named bounded check.
- `reject`: the candidate is dead in this grammar; later gates stay closed.
- `deferred`: the compiler has not performed the check and makes no health claim.

The JSON registry contains the derived radial Euler–Lagrange equation and evidence for every gate of every candidate.
