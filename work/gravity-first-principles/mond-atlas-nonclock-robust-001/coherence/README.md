# Broader continuous coherence and relay tests

Completed post-hoc development analysis on the same 102 eligible SPARC galaxies and 2212 radii. This directly broadens the nonclock branches: continuous global parameters, three deterministic optimizer starts per training fold, a free coherence exponent, and passive versus active reemission. Neither full 3D coherence nor measured gravitational opacity is being tested; Sigma is a local stellar surface-density proxy. No reserved galaxies, fitted halo parameters, time-energy claim or post-score bound expansion.

Two pre-response tests passed. All 270 ordinary-fold optimizer starts converged successfully. A separate frozen extension then tested gas-to-stellar and stellar-to-gas source-domain transfer with another 36 successful starts. Convergence is not a guarantee of the global optimum; all starts and outcomes are retained.

| Family | Whole-galaxy CV RMSE dex | MSE worse than continuously trained MOND |
|---|---:|---:|
| Fixed MOND | 0.10667 | 7.97% |
| Continuously trained MOND | 0.10265 | 0% |
| Newton, fitted global mass factor | 0.20709 | 306.98% |
| Surface coherence, exponent 1 | 0.11522 | 25.97% |
| Surface coherence, free exponent | 0.11409 | 23.53% |
| Passive absorption/reemission | 0.20571 | 301.55% |
| Active absorption/reemission | 0.20627 | 303.75% |

Free exponent improves coherence MSE by 1.94% relative to exponent 1. Its selected parameters averaged mf=1.012, A=4.371, log(Sigma0)=1.661 and n=0.571 over 15 folds; none hit declared parameter boundaries. These are parameter summaries, not a new combined prediction fit. A shallow transition is favored over the originally fixed exponent, but the improvement is small and remains behind the MOND control.

Mean signed log-speed residuals show a useful difference in error shape. Free coherence: inner -0.0114 dex, middle -0.0014, outer +0.0164. Continuously trained MOND: inner +0.0444, middle -0.0019, outer -0.0184. These signed averages can cancel individual errors and must not be read as better total accuracy; coherence still has larger held MSE. Active relay is particularly weak in the outskirts, with outer mean -0.1905 dex. All regions and actual counts are in run001/strata.csv.

Newton and both relay families hit the mass-factor upper bound 2 in all 15 folds. Coherence and trained MOND did not hit parameter boundaries. The relay law enhances high-Sigma locations as more influence interacts, while many outer deficits occur where Sigma is low. Its poorer predictions expose a problem with this specific local law, not a rejection of every possible spatial relay. The passive law has no gain above Newton for fixed mass; allowing its globally fitted mass factor to move does not supply the missing outer response.

## Source-domain transfer

Donor and recipient groups use the frozen nominal proxy 1.33 HI/(1.33 HI+.5 L)>=.5. There are 44 gas-rich and 58 stellar-rich galaxies. Thresholds and bounds stayed fixed; these are post-hoc transfer tests on an exposed sample.

| Family | Train gas-rich, test stellar-rich RMSE dex | Train stellar-rich, test gas-rich RMSE dex |
|---|---:|---:|
| Trained MOND | 0.09359 | 0.11187 |
| Coherence exponent 1 | 0.13021 | 0.14105 |
| Free coherence exponent | 0.13372 | 0.13454 |
| Passive relay | 0.14508 | 0.26851 |
| Active relay | 0.23250 | 0.26851 |

Freeing the coherence exponent helps transfer in one direction and hurts in the other. Both remain behind the trained control. Active reemission transfers particularly poorly from gas-rich training galaxies to stellar-rich recipients. Source-domain robustness therefore does not establish a universal surface-only law.

## Verification and limits

run001 contains every optimizer start, chosen parameters, 46,452 held predictions, per-galaxy errors, signed regional residuals, folds and source bindings. domain001 contains both donor-only fits and 15,484 recipient predictions. The independently written forward evaluator replays every selected prediction and selected training loss; maximum log-speed discrepancy is 8.88e-16, training-MSE discrepancy below 8.4e-17. It shares the source loader and does not independently refit optimizers. Original review output serialization failure (a NumPy boolean) is retained in independent-review/failure.json; independent-review-002 contains the successful corrected receipt. Prediction outputs were unchanged.

This is substantive additional testing of nonclock alternatives, not an exhaustive search. The remaining physical opportunity is to find an independently supported spatial or environmental quantity beyond this local stellar proxy. Neither the small free-exponent improvement nor a static fit identifies energy transfer, memory, a 3D redistribution mechanism or a lensing metric.
