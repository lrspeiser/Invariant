# Gravity pattern system: source and native-selection tasks reviewed

Two additional executed task milestones have passed coordinator review and the
publication subset now passes 112 tests. The task graph is active; first CUDA
learning and source/selection controls are complete at their bounded scopes.
Lensing ingestion and synthetic motion controls continue independently.

The [baryonic metadata package](../mond-atlas-baryon-recovery-001/README.md)
recovers both original S4G tables with the old hashes and reproduces 65 previous
geometry fields. It has geometry for nine of twelve seeds, including three
reliable and six uncertain orientation flags. Three seeds require independent
geometry sources. It also verifies current fixed M/L=0.6 field provenance and
separates a historical cleaned-color conversion error from those current runs.
Fourteen new tests pass. Absolute calibration, source covariance and 3D depths
remain open; recovered source metadata is not an independently measured mass.

The [native selection package](../mond-atlas-native-selection-001/README.md)
executes 864 known-signal injections into twelve fixed real background patches,
2304 conditional simulated injections, 96 noise cubes and 72 noiseless controls.
Nine new tests pass. The declared mask overlaps 81.36% of positive published
moment-map spatial support. That projected comparison is not a 3D mask match,
false-positive rate or complete recovery of the publisher's selection. The
response brackets and correlated/possibly emitting backgrounds stay explicit.

All 21 native public-package entries and 17 baryon-package entries were rehashed
before integration. The coordinator also rehashed the 22 recovered source
payloads and 18 native-run result files including their private array products.
Private files are not published. Frozen reports retain their historical local
publication wording; the coordinator's Git receipt establishes publication.

The [first GPU experiment](../mond-atlas-pattern-findings-001/README.md) remains
unchanged: 126 galaxies, three split seeds, no stable improvement from these
coarse structure summaries. Newly recovered metadata was not retroactively
inserted into that frozen experiment. The goal remains active, with zero
admitted observed full-field cube likelihoods and no new gravity-law claim.

Read [task plan](../../../docs/GRAVITY_PATTERN_SYSTEM_TASKS.md) and
[execution status](execution-status.json) for active task IDs and dependencies.
