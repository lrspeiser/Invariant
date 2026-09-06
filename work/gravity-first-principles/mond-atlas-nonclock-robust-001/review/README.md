# Independent numerical review and limits of robustness

**The completed branches pass numerical replay. They do not establish robust physical mechanisms.** The small average advantages of finite_mix and finite_flat_bridge over continuous MOND are sensitive to individual galaxies and have descriptive bootstrap intervals spanning zero. The bridge overlaps the previously studied generalized cored-clock potential class; it is not an independent theory simply because its label changed.

## What was reviewed and replayed

Read the coherence and return runners, tests, preflights and domain-transfer scripts/addenda. Reran all six manufactured tests: two coherence tests and four return tests pass. The return branch retained a documented pre-response quadrature repair at a known sharp cutoff; force equations and tolerances were unchanged.

An independent algebraic evaluator, sharing only the previously audited SPARC source loader, replayed every saved start's training loss, checked that each chosen fit minimized allowed successful training starts, reproduced held predictions, and recalculated the reported RMSEs. No production prediction function was used. No parameters were refit and no original artifacts changed.

| Branch/evaluation | Starts successful / attempted | Choices | Held radial records | Largest prediction difference |
|---|---:|---:|---:|---:|
| Coherence ordinary galaxy folds | 270 / 270 | 105 | 46,452 | 8.89e-16 dex |
| Coherence source-domain transfer | 36 / 36 | 14 | 15,484 | 8.89e-16 dex |
| Return ordinary galaxy folds | 315 / 315 | 75 | 33,180 | 9.33e-15 dex |
| Return source-domain transfer | 42 / 42 | 10 | 11,060 | 7.55e-15 dex |

Total: 663 optimizer starts, 204 choices, 106,176 held records. These repeated predictions are not independent observations. Every branch uses the same 102 eligible historical galaxies and 2212 radii from 139 opened development members; zero reserved members were opened by these replays.

The training masks are correct on inspection and numerical replay. Coherence optimizers receive training-row slices; return optimizers turn training-galaxy masks into rows before constructing objectives. The domain split is source-defined, not response-defined: 44 nominal gas-rich versus 58 nominal stellar-rich galaxies. No recipient values enter the fitting objective. All three ordinary fold seeds reproduce the identity-hash assignments. Coherence includes separately optimized Newton as a declared nested candidate; return includes explicit zero-amplitude starts.

All successful solver flags mean the specified stopping criteria were met, not that a global minimum or unique parameters were proved. Three starts run along the same diagonal of parameter space. Some coherence active-relay starts end at training losses differing by about1.07%. Return attempt spreads include the deliberately distinct zero-amplitude branch and must not be mistaken for numerical nonconvergence. Cutoff positions outside the measured range may be unidentifiable even when the optimizer reports success.

## Post-hoc paired and influence diagnostics

`DIAGNOSTIC_ADDENDUM.md` froze the diagnostic before computation. We averaged each galaxy's squared log-speed errors across the three splits, subtracted candidate error from that branch's continuous-MOND error, and drew 4000 paired whole-galaxy resamples with seed9062491. No refitting occurs. These are descriptive intervals on an exposed sample, not calibrated significance or a full uncertainty analysis.

| Family | Average MSE gain vs continuous MOND | Descriptive 95% interval for MSE improvement (dex²) | Leave-one-galaxy-out gain range |
|---|---:|---:|---:|
| Finite mixture | +0.498% | [-0.001177, +0.001386] | -2.391% to +2.424% |
| Truncated point kernel | -5.939% | [-0.001751, +0.000574] | -9.834% to -4.117% |
| Finite flat bridge | +4.510% | [-0.001224, +0.002349] | -0.194% to +7.543% |
| Free-exponent surface coherence | -23.526% | [-0.005000, +0.000141] | -35.086% to -18.945% |

All intervals span zero. The bridge's largest positive contributor, UGC07577, contributes103.92% of the net summed improvement. Removing that galaxy reverses the average gain to -0.194%. It accounts for10.23% of the total absolute positive/negative changes, so this statement concerns cancellation in a small net gain, not domination of all absolute errors. Finite_mix is similarly fragile: CamB contributes502.68% of its small net gain. Every galaxy is retained; these diagnostics are not permission to exclude inconvenient systems.

The bootstrap omits optimizer refitting, shared training-set dependence, formula selection, survey correlations, inclination/distance, mass calibration, gas completeness, and radial covariance. Its positive-draw fraction is not a probability that a physical theory is true.

## Domain transfer retains the harder cases

Both domain directions are reported rather than pooled away. For return models, gas-rich to stellar-rich transfer gives RMSE0.09341 dex for bridge versus0.09359 for continuous MOND, while the finite mixture degrades to0.10305. In the reverse direction, bridge gives0.10852 versus0.11187 for continuous MOND; mixture gives0.11205 and point kernel0.11401. These are the same survey's source-proxy groups, not independent survey replication.

Coherence free-exponent transfer is worse than continuous MOND in both directions:0.13372 versus0.09359 dex and0.13454 versus0.11187 dex. Active relay is much worse in gas-to-stellar transfer (0.23250 dex). Those failures must remain in the mechanism discussion; a better ordinary-fold score does not erase them.

## Bridge is a generalized cored-clock shape

Let `d=delta Rd`, `b=C rM`, `rM=sqrt(GM/a0)`. The bridge can be written exactly as

`gextra = eta C GM r / [(r+d)^2(r+b)]`.

The earlier cored-clock form is `beta GM r/[(r+d)^2(r+d+B)]`. Setting `beta=eta C` and `b=d+B` maps the forms when the required parameter relations hold. The new bridge lets inner and outer scales vary more independently; that is a useful shape generalization, not independent evidence for a second physical mechanism. If b<d, the mapping would require negative B and is outside the earlier positive-clock-scale parameterization; both still belong to the same broader rational force class. No time-energy exchange was measured in either labeling.

## Controls still missing: do not call these branches complete mechanism tests

- **Matter uncertainty:** mass-to-light factors vary globally, but per-galaxy population, bulge, molecular gas, distance, inclination and depth uncertainties are not jointly propagated.
- **Measurement model:** no admitted observed full-cube likelihood, spatial/channel covariance, source-region mask validation or warp/streaming/pressure marginalization enters these scores.
- **Geometry:** stellar surface brightness is not total three-dimensional density or intervening opacity; point kernels are not extended-source convolution; scalar radial coherence is not a refracted-gravity PDE.
- **Physics:** active re-emission has no established energy source/budget. Memory, spin/current coupling, causal propagation, conservation/action, relativistic light response, cluster and Solar System transfer remain untested here.
- **Statistical independence:** repeated folds and domain partitions reuse exposed data. No untouched confirmation survey or galaxies were scored. The post-hoc structural choices remain part of the search history.
- **Optimization/identifiability:** successful starts do not prove global optima; profile likelihoods, source-uncertainty stability and broadly sampled starts are absent. Boundary flags and cutoff degeneracies must be retained.

The concrete progress is broader and auditable *radial formula development*, plus a useful diagnosis that small gains are fragile. The numerical review supports publication with these limitations; it does not support claiming that coherence, redistribution or any competing mechanism has been comprehensively tested or established.

Reproduction: run `review/replay.py coherence run001`, `review/replay.py coherence domain001`, `review/replay.py return run001`, and `review/replay.py return run001 transfer` using the repository Python environment. `diagnostics.py` repeats the frozen descriptive diagnostic. Exact hashes and every retained optimizer failure/status are in the corresponding review receipts.
