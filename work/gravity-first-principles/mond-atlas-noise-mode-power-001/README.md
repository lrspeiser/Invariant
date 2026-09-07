# Per-mode noise power transfers; residual limitations remain

Completed the single frozen extension that gives every spatial cosine mode its own positive power before fitting a band-specific channel covariance. No modes were excluded. Same29western/27eastern NGC2976 background cores, same inherited three-fold western selection, historical development exposure retained. No source-region pixels, motion/lensing targets or gravity scores.

Western selection chooses **no neighboring-mode pooling (radius0)** in all four bands. Selected channel shrinkage remains DC1.0, low0.6, middle0.3, high0.1. All candidate powers, matrices and choices were saved before opening eastern data in this run.

The previous high-band cancellation is resolved in this background transfer:

| Diagnostic | Previous broad-band model | Per-mode power model |
|---|---:|---:|
| Joint q/N | 0.9932 | 0.9998 |
| Middle-band lower-frequency half | 1.2906 | 0.9980 |
| Middle-band upper-frequency half | 0.6907 | 0.9827 |
| High-band lower-frequency half | 1.9602 | 0.9997 |
| High-band upper-frequency half | 0.00237 | 1.0014 |

All six aperture q/N scores pass the fixed descriptive[0.8,1.2] range: **1.028,1.023,1.031,1.048,1.053,1.166** for sides1,2,4,8,12,24. Eastern/predicted trace ratios range **1.031–1.067**. Mean joint log density rises from0.0968 to2.0078 per voxel. This is improved numerical/measurement modeling, not new gravitational behavior.

**575 of576 individual spatial modes pass** the same descriptive q range. The retained exception is(ky11,kx9), q/N1.2141. One channel diagnostic remains clearly outside range: the lowest DC channel-eigenvalue quartile is1.566, unchanged from the preceding model. No thresholds or powers were adjusted after seeing these outcomes; descriptive thresholds are not multiple-testing significance claims.

Among1104 adjacent spatial-mode pairs, mean absolute normalized residual product is0.0230 and none exceed0.1. These are limited collapsed-channel adjacency checks. They do not test every cross-mode/channel covariance, nonadjacent pairs, cross-core dependence or the full distribution of residuals, and cannot establish mode independence.

The smallest learned high-band raw power is2.36e-10, versus a largest high-band power0.4046 in squared packet-intensity units. No selected power hits its positive floor and no weak mode is removed. The smallest power is about325,000 times the heuristic marker machine-epsilon64 times the largest global power. This does not establish the instrument's physical noise floor or supply a rigorous floating-point error bound; it indicates that the result did not arise by clipping these modes to our declared floor.

Three pre-access checks passed: pooling positivity/locality/scale, complete-basis dense covariance/Parseval/log determinant and singular channel covariance regularization. Independent verification uses a manually constructed cosine basis, independently reconstructs every selected raw/pool power and channel covariance, checks the western choice, replays27 core scores,576 mode scores,26 diagnostic groups,1104 adjacent products and all aperture projections. Maximum discrepancy is4.98e-13.

Every western candidate fit, power vector, covariance, score and all eastern diagnostics are retained. No new raw bytes were downloaded. This single step materially fixes the broad spatial-power mismatch on the exposed background regions, but **source/emission likelihood admission remains blocked** by the retained DC discrepancy and untested source-region/cross-mode/cross-core behavior. No further adaptive branch was run.
