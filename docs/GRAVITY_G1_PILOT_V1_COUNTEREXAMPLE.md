# G1 pilot v1: first-galaxy counterexample

**Decision:** `BLOCK_G1_PILOT_UNCOVERED_OR_INCOMPLETE`

**Receipt:** `runs/gravity/g1-pilot/first-galaxy-counterexample-v1.json`

The first preregistered pilot galaxy, UGC06787, received the complete v1 allocation of
10 million candidates in each of three arms: 30 million candidate–galaxy trials total.
No confirmation galaxy or confirmation summary reached an evaluator. No candidate survived
the GPU slack gate, so no formula was admitted by CPU replay.

| Arm | Evaluated | Invalid domain | First predictive blocker |
|---|---:|---:|---:|
| Structured/Occam | 10,000,000 | 8,488,543 | 1,511,457 fail a Newtonian fold |
| Pseudorandom outer shell | 10,000,000 | 8,931,630 | 1,068,363 fail a Newtonian fold; 7 reach the wrong-law blocker |
| Claude-seeded basis recombination | 10,000,000 | 0 | 10,000,000 fail a Newtonian fold |

The important counterexample is the innermost contiguous radial block. Newtonian baryons
already score about 13.04 chi-square there. The best aggregate v1 candidates improve the
whole curve by orders of magnitude but worsen that inner block. A formula that merely adds a
nonnegative `A*r*h(y)` term, while `h` remains appreciable at high baryonic acceleration,
cannot safely leave this region alone.

This excludes the tested v1 candidate cells, not every formula of the broad verbal family.
The next grammar version should encode high-acceleration recovery directly—for example, an
envelope that forces the added acceleration toward zero as `y` grows—then repeat the same
frozen folds and baselines. Increasing the random count inside the unchanged v1 shells is not
scientifically justified by this result.

Claude's eight ideas and origin labels remain retained. Four were compiled into v1 basis
banks; four were retained but deferred. The provider labels are not novelty judgments and did
not decide survival. Several raw Claude shells were dimensionally invalid as written; the
compiler preserved that fact and used only their dimensionless basis ideas inside the typed
shell.
