# G4 photometric construction repair

Date: **2026-08-27**

## Decision

`BLOCK_G4_PHOTOMETRIC_CONSTRUCTION`

The repair imported the two published SPARC surface-brightness columns that the original
Invariant source asset had intentionally dropped. The exploration-only supplement contains
139 galaxies and 2,720 rows, with zero confirmation galaxies and rows. Its builder cross-checks
the original six published fields and distance header for every retained galaxy before saving
only `SBdisk` and `SBbul`.

The sealed result is
`runs/gravity/g4/universal-galaxy-law-construction-v2-photometric.json`.

## Search and best formula

Six photometric features were added to the six physically screened G4-v1 features. Every
linear term and commutative degree-two product formed a 90-term grammar. With 19 shrinkages,
the complete run evaluated **1,710 formula cells, 237,690 candidate-galaxy evaluations, and
4,651,200 candidate-point evaluations**. Every formula still had only two universal correction
constants and zero galaxy-local gravitational constants.

The best valid formula was

```text
V^2 = V_RAR^2 + r g_dagger [C0 + C1 f_bulge d(log(1+SBtotal))/d(log r)]
C0 = -0.01119952096575
C1 = -0.04784694015743
```

Its whole-galaxy cross-validated chi-square is **124,807.946**, a **4.52%** improvement over
RAR. The construction is labeled `COMBINATION`: it mixes known baryonic mass decomposition,
surface-brightness gradients, and the known RAR base. That label is not a novelty claim.

## Failed obligations

- NFW-plus-slack ceiling: candidate **124,807.946** versus limit **33,458.807**, an excess of
  **91,349.139**.
- Calibration: the second bulge-fraction quartile regressed **21.54%** against RAR and the
  second gas-fraction quartile regressed **20.70%**, beyond the frozen 10% limit.

Published surface brightness is real missing input, but it closes only 335.303 chi-square units
of the G4-v1 ceiling gap and makes two population strata materially worse. This finite family
is therefore excluded. It does not justify a confirmation run or a gravity-theory claim.
