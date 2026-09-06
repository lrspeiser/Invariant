# Acceleration-derived length repair: completed exploratory result

This is one explicitly post-hoc adjustment after seeing the original run001 summary. It reuses the same historically exposed SPARC development galaxies, not a fresh confirmation set. The frozen addendum, three passing pre-response tests and complete code/source hashes are retained. Same source definitions, eligibility, grids, three five-fold galaxy splits and equal-galaxy mean squared log-speed loss. No halo parameters, per-galaxy optimized parameters, new observations or grid expansion.

The change replaces disk-size-based response lengths with rM=sqrt(G Mproxy/a0). Clock inner softening remains Rd, while its potential scale becomes lambda sqrt(G Mproxy a0). This offers a mass-derived speed scale: in the clock's intermediate flat regime, v_extra^4=beta^2 lambda^2 G Mproxy a0. The regime is not guaranteed to exist for every galaxy or parameter choice.

| Family | Original RMSE dex | Repaired RMSE dex | MSE improvement over original | MSE worse than adjusted MOND |
|---|---:|---:|---:|---:|
| Clock potential | 0.15295 | 0.13741 | 19.29% | 83.47% |
| Point kernel | 0.12365 | 0.11636 | 11.43% | 31.58% |
| Finite p2 | 0.13841 | 0.14781 | -14.05% | 112.30% |
| Finite p3 | 0.11976 | 0.12326 | -5.93% | 47.63% |
| Finite mixture | 0.11976 | 0.11650 | 5.38% | 31.88% |
| Fixed MOND control | 0.10667 | 0.10667 | numerical zero | 10.56% |
| Adjusted MOND control | 0.10144 | 0.10144 | numerical zero | 0% |

All quantities are errors on whole galaxies excluded from each parameter-selection fold, averaged over three repeated splits of the same sample. Post-hoc family development remains exposed to this sample; these splits do not erase that exposure. RMSE in dex is a logarithmic speed error, not a percentage speed error. Improvement percentages compare MSE, not RMSE.

Mass-derived scaling helps the clock and kernel, but does not close their predictive gap to the adjusted MOND control. The mixture changed from the original pure p3 preference to q=0.5 in all 15 selections, suggesting that the transition shape matters along with length scaling. This is a development clue, not a discovery of a physical mechanism.

Boundary behavior is important: kernel and mixture selected eta=30, length_factor=8, mf=1.2 in all 15 folds. The clock selected the lowest clock_factor=0.1 and beta=10 in all folds; mf=1 in 11 and 0.8 in four. Finite p3 selected eta=30, length_factor=4, mf=1.2 in all folds. These limited-grid comparisons do not locate unconstrained optima. No grid was expanded after seeing these scores. The adjusted MOND control also selected boundary values a0_factor=0.5 and mf=1.2 in all folds.

Artifacts in run001 include every candidate, every fold training loss, every family held radial prediction/residual, per-galaxy errors, selections, parameter boundary flags, fold assignments, source/member receipts, tests and summary. The raw source archive remains outside Git. Three tests executed before response access check finite predictions, zero-strength reduction, an independent clock-potential derivative, independent finite-profile formulas and the intended mass scaling; parent mechanics tests already cover the retained kernel integral and other unchanged formulas.

No result here demonstrates that time produces energy, identifies an energy-transfer rate, measures a storage timescale, or validates a three-dimensional field or lensing prediction. The constructive result is narrower: changing the source-derived scaling can help, and the remaining errors are sufficiently large that a successful physical explanation requires more than renaming the potential as clock energy.
