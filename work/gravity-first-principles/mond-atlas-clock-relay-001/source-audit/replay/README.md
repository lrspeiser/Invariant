# Independent replay and reporting corrections

An independently written NumPy loader and formula evaluator reproduced the frozen run without importing its numerical helpers. It evaluated all 713 candidates, checked every saved training loss and all 180 selections, reconstructed 79,632 held radial predictions, and compared every galaxy/family score. It opened only the same 139 historical member bodies; the declared selection retained 102 galaxies and 2212 radii. Zero reserved member bodies were opened. The original run was not edited or refitted.

Maximum differences: training loss 8.89e-16; chosen candidate's excess over independently recomputed minimum 2.78e-17; galaxy log-speed MSE 6.94e-17; galaxy speed MSE 1.28e-11 (km/s)²; saved selected radial prediction/residual 4.45e-16 dex. All pass the declared replay tolerances. Exact file bindings and figures are in `receipt.json`.

## Corrections to interpretation of original run fields

- Original `source_only_parameters: true` is misleading. Predictor inputs are source-only. Parameters were selected using observed training velocities, not derived from ordinary matter alone.
- Original fixed-MOND row reports 19 galaxies improved against itself. These are floating-point averaging differences, not improvements. Its meaningful comparison to itself is zero gain and zero improved galaxies. Do not report 19 as a scientific result.
- All family held predictions, promised in the frozen protocol but only saved for the overall selector by the runner, are now in `all-family-held-predictions.csv`.

## Boundary results

Each family has 15 selections across three seeds and five folds. Frequencies describe a small discrete candidate grid, not fitted confidence intervals.

| Family | Boundary behavior |
|---|---|
| Adjusted MOND and overall selector | All 15 select stellar mass factor 1.2 (upper edge) and acceleration factor 0.5 (lower edge). |
| Absorption | All 15 select zero opacity, reducing to the adjusted Newton baseline. |
| Clock potential | All 15 select stellar mass factor 1.2 and clock factor 0.1, both grid edges. |
| Surface relay | Stellar mass factor is at the upper edge in all 15; response strength/scale are interior. |
| Point kernel | Mass factor, response strength and scale are interior. |
| Finite p2 | Length is at its upper edge in 5/15; strength is at its upper edge in 1/15; stellar mass factor is on an edge in all 15. |
| Finite p3 | Stellar mass factor is at its lower edge in all 15; response strength/scale are interior. |
| Finite mixture | All 15 select pure p3 (mixture weight 1), so the mixture adds no selected predictive benefit. |

No edge was moved after seeing the results. A future expanded grid would be a separately registered follow-up. The clock model's lower-bound scale indicates unresolved parameter preference, not evidence that a measured clock reservoir exists. Static velocity data do not identify energy exchange or memory delay.

All bootstrap intervals remain descriptive for this historically exposed sample and fixed cross-validation predictions; they do not propagate refitting, nuisance, sample selection or survey dependence.
