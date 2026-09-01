# TWELL-400 source-shaped rebind/replay v1 independent audit brief

Perform a strict read-only independent audit of `open-gravity-twell-400-source-shaped-rebind-replay-v1`. Do not edit the subject config, module, test, receipt, or artifacts. Do not reuse the subject builder as the independent recomputation path.

## Exact subject hashes

- config: `12df390acedd7afb650b101b714758b5bd8e1ccf2f5f36da7a1779040533688a`
- module: `babbeb727660623bca3816a3ad58d87c319774657ed09f18f00e89c76b2e9e18`
- test: `fdf5ea8ff098ccc3dec5731396501f9374149ae54cb4ba83d484d863a42646cb`
- receipt raw: `bcaf6df58e39a9dc52329e8afd78546c905cba20271858bb68e0e3b4e19bdab4`
- receipt content: `14369de6eafbe07f3918249af4704f0405ffac7fd86e21f4aa5e7b7a67a6140f`

Artifact hashes:

- compatibility ledger: `b1abf2353db5be673ff46f51dc3c346b1979405e7e8496e08390f8abde961816`
- parameter-cell disposition ledger: `641fa0e182389cb27c3832adf32ef610d5181a7048228cd7e521194a2e2a67a1`
- execution bindings: `9741698a813b0ac04cdca84be1145005c6a985b6cd6ff2e3a30e2a9f6d6af020`
- source projection ledger: `94c6103ea1dc095f75f9968d1ddbac1dcee8d78eca521119f5a95d4fef1018f8`
- source projections NPZ: `5bd7f07c8d36dbd0cda9a27e5e701f6bdd3fa623c23d9aa9d93cb3d61b9a7bd9`
- unique executions: `0ebf1a8a825514bb42af6153deb9df1a531ec3075d93878286353e54d04e4567`
- predictions NPZ: `f6d1a6345e422c0c7b4dbb4081b1eab37affb016866d8fad29018e621f17d941`
- replay ledger: `0b54a50fbce357c0081040f950d079171b58282fdd5f5da685f2bfa37371d363`
- equivalence ties: `d374bf4cc405b8764c5cf340495449d496c19b3f514b558bdfec56cb4b5ac309`
- invariance gates: `c1ff6e709ddfad8bdb92ea23c9d876bdbbc9aa9b287254d6d5b8c859508d1f1c`

## Required independent recomputation

1. Verify exact final-v3 400-card order, card SHA, formula SHA, and all 1,184 unique parameter cells directly from the card stream.
2. Recompute the disjoint gap without importing the subject classifier: 126 provisional static cards/370 cells, 84 A15-A18 temporal cards/253 cells, and 190 missing-driver cards/561 cells.
3. Recompute the 400 x 5 compatibility matrix: 2,000 rows total; 110 executable, 290 source-blocked, and 1,600 incompatible. Confirm X-COP alone has 110 executable plus 16 D13, 84 temporal, and 190 missing-driver blocks.
4. Verify the exact independently audited PHANGS, X-COP, Solar, lens, void, static-adapter, final-v3, and generic-runner hashes in the config. Confirm both void Hubble-rate features remain `km s^-1 Mpc^-1`.
5. Independently load only the 72 `source__*` X-COP NPZ members, verify all formula-feature hashes/units/axes/frame/geometry/time, and confirm no response, candidate, variance, or truth member is loaded.
6. Independently reconstruct at least one complete object across all 324 parameter cells using the declared Cartesian-radius shell-mean total-baryon projection and the frozen static adapter. Prefer recomputing all eight objects. Verify deterministic byte replay, prediction hashes, shapes/units, and the inherited 257-vs-129 convergence decision.
7. Confirm exact unique execution counts: 2,592 attempted = 2,554 completed + 38 `NUMERICAL_INVALID`. Exact invalid formulas: `TW2-A11-D03` 16, `TW2-A11-D06` 16, `TW2-A11-D04` 5, `TW2-A11-D07` 1. Confirm replay fan-out: 62,208 = 61,296 completed + 912 invalid.
8. Verify every replay row binds card/formula/adapter/source/scenario/result/prediction/metric hashes plus truth/noise/seed/nuisance identity; every metric is source-only numerical health with `scientific_score=false`.
9. Verify 122 finite-source prediction tie groups are not promoted to formula identity; the 110 executable formula equivalence families are singleton exact families.
10. Exercise mutation rejection, temporal static-replication rejection, D13 rejection, resource ceilings, receipt/artifact self-seals, focused tests, Ruff, and deterministic `check`.

If and only if all gates pass, create one unique append-only self-hashed audit receipt under `runs/gravity/` and report its raw/content hashes. If blocked, preserve work evidence only and do not create a PASS receipt.
