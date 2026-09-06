# Post-hoc repair: reduce the central pull, connect the scale to baryonic mass

The diagnosed problem was too much added pull near the center. Multiplying the original clock acceleration by `r/(r+Rd)` makes the added acceleration vanish linearly at the center while retaining its intermediate behavior and finite effective total source. This is a developmental repair chosen after inspecting run001, not an independent confirmation.

We froze and tested two families, 72 candidates each, on the same 102 galaxies/2212 radii. Four independent manufactured tests passed before response access. Every family uses three repetitions of five whole-galaxy folds and globally trained parameters. All source archive hashes were checked and no reserved member bodies were opened. Every candidate training loss, choice, held prediction and regional bias is saved in `run001/`.

| Formula | Held log-speed RMSE | MSE change from original clock | MSE relative to fixed MOND |
|---|---:|---:|---:|
| Original clock | 0.15295 dex | Baseline | 105.59% worse |
| Cored clock, original scale Psi0=lambda a0 Rd | 0.12131 dex | 37.09% better | 29.34% worse |
| Cored clock, mass scale Psi0=lambda sqrt(GM a0) | 0.10969 dex | 48.56% better | 5.75% worse |
| Fixed simple MOND | 0.10667 dex | 51.36% better | Baseline |
| Adjusted simple MOND | 0.10144 dex | 56.01% better | 9.55% better |

The mass-scaled repair remains 16.92% worse in MSE than adjusted MOND. Its use of sqrt(GM a0) imports a MOND-like mass/acceleration scaling; its improvement is not evidence that clock energy was detected.

## Where the repair worked and where it did not

Across the three repeated splits, the inner signed log-speed bias dropped from +0.16771 dex to +0.04498 dex for the mass-scaled repair. Of 96 galaxies with inner coverage, the count with positive inner mean bias fell from 84 to 58. Inner RMSE improved from 0.25416 to 0.15650 dex. These are region-wise equal-galaxy summaries, not physical percentages of gravity.

Outer predictions remain too slow: the mass-scaled repair's outer mean bias is -0.06508 dex and RMSE 0.10061 dex across 90 galaxies. Fixed MOND's corresponding outer RMSE is 0.07054 dex. Thus the central shape problem is substantially reduced, but the outer response needs better explanation. The original-scale core worsens outer bias to -0.09205 dex, illustrating why fixing the center alone is insufficient.

Both repaired families selected exactly the same tuple in all 15 folds: stellar mass factor1.2, strength10, lambda0.1. Stellar mass is at the upper grid edge and lambda at the lower edge. These remain unresolved preferences, not stable physical parameter determinations. No grid was expanded after inspecting outcomes.

## Mathematical and numerical checks

For `B=GM/Psi0`, the added field is `gextra=beta GM r/[(r+Rd)^2(r+Rd+B)]`. Its normalized effective enclosed source is `H=r^3/[(r+Rd)^2(r+Rd+B)]`, which increases monotonically to1. The potential is defined by `Phi=-integral_r^infinity gextra(s)ds`. Independent improper quadrature and finite differences verified its gradient. A 1/r intermediate acceleration exists only if `Rd << r << B`; it is not guaranteed for every parameter combination.

An independently rearranged NumPy evaluator replayed all 144 candidates, all training losses and all held predictions; it shared the previously audited source loader but did not use the repair's formula function. Maximum training-loss difference was 5.56e-17, chosen-loss excess over independent minimum 5.21e-18, and held prediction difference 4.45e-16 dex. `replay-receipt.json` binds the exact artifacts.

The repaired law is an empirical radial potential. It does not establish a time reservoir, local energy transfer, a covariant conservation law, lensing, or a three-dimensional matter response. A later experiment must be registered separately; these post-hoc improvements must remain distinguished from confirmation.
