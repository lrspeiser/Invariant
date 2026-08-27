# G1 pilot: 12/12 PASS

**Decision:** `PASS_G1_PILOT_UNION_12_OF_12`

**Receipt:** `runs/gravity/g1-pilot/receipt-v3.json`

The counterexample-driven G1 pilot now covers all 12 target-blind selected SPARC exploration
galaxies. The checked union comprises v2 formulas for ten galaxies and v3 baryonic-structure
repairs for UGC11820 and UGC11455.

## Measured search

- Cumulative candidate–galaxy trials: **450,000,000**.
- V2 full pilot: 360 million trials, 10/12 covered.
- V3 targeted repair: 60 million new trials, 2/2 repaired.
- V1 retained counterexample: 30 million UGC06787 trials, 0 survivors.
- Confirmation evaluator accesses: **0**.

UGC11820 produced CPU-admitted formulas in structured, pseudorandom, and creativity-guided
arms. UGC11455 produced eight structured and ten pseudorandom CPU admissions among retained
replays; its creativity-guided branch remains a visible failure rather than being silently
discarded.

## Example local formula class

V3 formulas have the typed shell

```text
V_pred^2 = V_bar^2 + r * (A*phi_1(feature_1) + B*phi_2(feature_2))
```

`A` and `B` are the only fitted local acceleration coefficients and are estimated separately
on each fold's training radii. Feature choices include dimensionless baryonic acceleration,
radius normalized by the peak of the published disk contribution, gas/disk/bulge fractions,
baryonic log slope, a baryonic mass proxy, and gas-to-disk ratio. Every such feature is a
deterministic function of allowed baryonic inputs and reads neither `V_obs` nor its error.

The discrete feature, center, width, and shape address is charged in description length.
Provider origin labels are retained but do not decide survival and do not establish novelty.

## Claim boundary

This is a successful search-system test, not a new theory of gravity. Each galaxy may select a
different formula and carries two local gravitational coefficients. The result therefore
demonstrates that the engine can generate, falsify, repair, and retain predictive local
relations under contiguous radial holdout. It does not replace dark matter, modify GR, predict
an unseen galaxy, or establish originality.

The PASS authorizes the next G1 step: freeze the union grammar and search all 139 admitted
exploration galaxies. G1 itself remains incomplete until coverage is 139/139.
