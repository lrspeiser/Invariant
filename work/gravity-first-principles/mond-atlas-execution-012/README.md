# Gravity pattern system: tasks created and first CUDA learning run completed

The user-authorized system now has an explicit task graph and three independently
running source/selection/lensing tasks. Compute and initial radial learning have
executed here. See [task plan](../../../docs/GRAVITY_PATTERN_SYSTEM_TASKS.md),
[actual learning findings](../mond-atlas-pattern-findings-001/README.md), and
[execution status](execution-status.json) for exact dependencies and task IDs.

The first CUDA experiment covers 126 development galaxies, eight model/feature
comparisons, three whole-galaxy fold partitions, 3,024 out-of-fold predictions
and 16 structure shuffles. Numerical controls agree with CPU and an independent
library. Combined structure features yield a split-sensitive 2.26% improvement
in nonlinear mean squared error; the conditional uncertainty includes zero.
No stable physical correction or new gravity law has been established.

The previously blocked 535-file atlas milestone has been pushed to main as
34b156ac95e9b03a8fc27a82bb99e3727331a756. Ordinary Git/network access now works.
Byte-preserving Git attributes retain the original hashed protocols and data
receipts; fixed-width FITS header whitespace is deliberately preserved. Raw
observations and large fields remain outside Git. The new publication subset
passes 89 tests. Other concurrent source tasks are not silently included before
review, even when their tests happen to pass in the shared workspace.

Two source tables have been recovered by T2 with reported matching old hashes;
its reconciliation and conversion audit is still executing. T3 is executing a
native-geometry selection pilot, with real covariance limitations retained.
T5 is ingesting three historical exploration lenses and recording missing
instrument/source constraints. These tasks own separate new paths and cannot
change the coordinator's Git index or frozen outputs.

The whole research goal remains active. There are still zero admitted full-field
observed cube likelihoods. Source recovery, 3D uncertainty, motion controls and
survey/group transfer remain necessary before a resolved structure or lensing
claim can be made. This is the first executable learning stage, not completion
of the whole scientific program.
