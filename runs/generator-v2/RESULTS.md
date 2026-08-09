# Generator v2 milestone results

## Complete declared traversal

The frozen 50-basis, one-to-six-term grammar was traversed in full on 2026-08-06.

| Quantity | Result |
|---|---:|
| Declared actions | 1,088,651,720 |
| Processed actions | 1,088,651,720 |
| Rejected: flux only | 728 |
| Rejected: high-field limit | 729,109,786 |
| Rejected: no gradient sector | 1,472,776 |
| Rejected: negative elasticity control | 286,045 |
| Rejected: sampled static convexity | 340,241,945 |
| Sampled-static survivors | 17,540,440 |
| Authoritative elapsed time | 94.1 seconds |
| Sustained throughput | 11.80 million actions/second |
| Fixed blocks | 16,612 |
| Manifest size | 4.06 MB |
| Observational data opened | No |

Complete commitment root:

```text
8c56d27cbf9c3ec28328a3244f745424148c6f8516ddd7c5922f6ad4c8b77a93
```

The authoritative machine-readable record with action-level asymptotic cancellation, hash-random samples, and rejection witnesses is [`billion-authoritative.json`](billion-authoritative.json). Its independent audit is [`billion-authoritative-crosscheck.json`](billion-authoritative-crosscheck.json).

## Accuracy and determinism evidence

- Rust and independent Python implementations both compute exactly `1,088,651,720` actions.
- One-thread and eight-thread million-action runs produced identical block roots and gate counts.
- Independently compiled Windows GNU and Ubuntu 24.04 WSL2 binaries produced identical million-action basis hashes, survivor samples, gate counts, and commitment roots.
- The `10^6`, `10^8`, and complete `1.08865172×10^9` milestones all executed rather than being extrapolated.
- The complete manifest’s config hash, basis hash, block root, gate accounting, survivor count, and observation-closure flag pass the independent Python audit.
- All 32 deterministic hash-random survivor samples agree across Rust and Python on ordinal decoding, term IDs, signs, expressions, stable IDs, cheap gates, and the sampled Hessian verdict. Their ordinals span `22,688,083` through `965,566,111`.
- The authoritative manifest contains one reproducible lowest-hash witness for each of the five rejection families.
- The high-field gate was audited and corrected before finalization so that opposite-signed terms with equal leading powers can cancel at the action level. A dedicated positive control proves the cancellation survives while its uncancelled partner is rejected.
- A checkpointed million-action run reused all 16 completed blocks on replay, computed zero actions, and reproduced the identical root and gate counts.

## Interpretation

The run proves that the compiler can exhaustively enumerate and apply its declared pre-covariant gates to more than one billion sparse actions on this machine.

It does **not** prove that 17,540,440 healthy gravity theories remain. Those rows have only survived dimensionless-by-construction, spatial-state, action-level high-field, gradient-sector, negative-elasticity, and sampled radial-convexity checks. They still require stronger global tensor gates and covariant action construction before exact variation, GR/Solar reference recovery, or any measured observational test is authorized.
