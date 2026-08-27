# G1 exploration-atlas PASS

G1 reached `PASS_G1_ATLAS_UNION_139_OF_139` on 2026-08-27.

## Result

- The v1 production search evaluated 13.9 billion formula-galaxy candidates and covered
  138/139 admitted SPARC exploration galaxies.
- Repair v2 evaluated another 100 million RAR-base single-feature pairs on NGC2955 and
  retained no survivor. That finite family remains recorded as an exclusion.
- Repair v3 exhaustively evaluated all 6,081,328 pairs from a 3,488-component target-blind
  interaction grammar. GPU screening found 25 slack survivors; CPU-FP64 replay admitted 24.
- The cumulative G1 total is **14,006,081,328 candidate-galaxy trials**.
- Union coverage is **139/139 exploration galaxies** and confirmation-evaluator access is
  still zero.
- The sealed PASS receipt is `runs/gravity/g1-atlas/repair-v3.json`, content seal
  `3e56e19f8c4c116c533fd986d5f37358105b0d51aad2854b13c1d4bf7a8c549a`.

The exhaustive interaction screen ran at about 4.56 million candidate formulas per second
and completed its GPU portion in about 1.33 seconds on the NVIDIA GeForce RTX 5090. CPU FP64,
not the GPU, made all 24 admission decisions.

## Example survivor

The leading retained NGC2955 formula is

```text
V_pred^2 = V_RAR^2 + r*(A*Phi_1 + B*Phi_2)

Phi_1 = T_3(z_log(r/r_disk_peak)) * T_6(z_mass_proxy_fraction)
Phi_2 = T_3(z_disk_fraction)       * T_6(z_mass_proxy_fraction)
```

`T_n` is a Chebyshev polynomial. Each `z` maps a baryonic feature to `[-1,1]` from that
galaxy's baryonic inputs only. `A` and `B` are signed acceleration coefficients fitted on
the training radii of each fold. The formula has aggregate held-out chi-square
`5.226584559062e+01`, two-sigma coverage `0.8333`, and passes every frozen aggregate check.
Its five contiguous-fold chi-squares are approximately `17.655`, `14.676`, `8.464`, `0.504`,
and `10.966`; every fold passes every baseline and coverage obligation.

## Creativity and lineage

This formula is **not** labeled a historical novelty. Its base is a known empirical
RAR/MOND-family relation, Chebyshev polynomials and tensor-product interactions are known,
and the pipeline's label is `new_combination_of_known_ideas`. G2 must still test algebraic and
behavioral equivalence and prior-family membership.

The interaction grammar was designed after inspecting the exploration-only NGC2955 failure,
and a passing member was observed in a diagnostic before the sealed exhaustive replay. The
v3 result is therefore a reproducible exploration-atlas completion, not an independent
prediction. This disclosure is machine-checked in the config and receipt.

## Claim boundary

G1 proves that the current system can build a compressed, cross-validated local diagnostic
formula for every admitted exploration galaxy while retaining failed families. It does not
prove one universal gravity law, remove two local per-galaxy coefficients, predict a new
galaxy, establish a no-dark-matter theory, or provide an alternative to general relativity.
It authorizes G2 equivalence collapse only. The 35 confirmation galaxies remain sealed.
