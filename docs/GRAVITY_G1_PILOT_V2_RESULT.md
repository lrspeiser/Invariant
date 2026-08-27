# G1 pilot v2: two-kernel result

**Decision:** `BLOCK_G1_PILOT_V2_UNCOVERED_OR_INCOMPLETE`

**Receipt:** `runs/gravity/g1-pilot/receipt-v2.json`

G1 v2 evaluated the full 360-million candidate–galaxy allocation: 10 million candidates
in each of three arms for each of 12 target-blind, baryonic-feature-diverse SPARC exploration
galaxies. It covered 10/12 galaxies with zero confirmation evaluator accesses.

| Pilot galaxy | Covered | CPU-admitted formulas among retained replays |
|---|---:|---:|
| UGC06787 | yes | 1,330 |
| D564-8 | yes | 1,332 |
| UGC06973 | yes | 1,536 |
| NGC2403 | yes | 9 |
| UGC06614 | yes | 1,536 |
| F579-V1 | yes | 1,461 |
| UGCA444 | yes | 1,530 |
| NGC0300 | yes | 1,226 |
| NGC2841 | yes | 1,533 |
| UGC11820 | **no** | 0 |
| UGC07232 | yes | 4 |
| UGC11455 | **no** | 0 |

The result establishes that v1's UGC06787 failure was representational: paired kernels with
two signed acceleration coefficients, fitted on training radii only, clear every frozen fold.
It does not establish a universal law. The selected kernel centers, widths, and shapes are
charged in the formula address/description length, while the two fitted acceleration values
remain galaxy-local diagnostics that cannot survive G3.

UGC11820 is dominated by the empirical-RAR fold bound in the v2 failure ledger. UGC11455 is
dominated by a Newtonian-compatible fold. A separate CPU/GPU diagnostic over target-blind
baryonic-structure functions found CPU-FP64 passing pairs for both: `log(y)` with gas fraction
for UGC11820, and baryon-derived normalized radius with gas/disk ratio or baryonic slope for
UGC11455. That diagnostic is not itself an atlas receipt. It justifies v3's next grammar:
include those declared baryonic invariants, retain the same two-constant ceiling, and replay
all 12 galaxies under the unchanged folds and admission thresholds.

G2 remains locked because G1 requires 139/139 exploration galaxies, and even the pilot has not
yet reached 12/12.
