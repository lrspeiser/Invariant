# Cluster force failures depend on source smoothing

This audit examines the nine retained candidates with a one-megaparsec length scale, across eight development clusters. It reuses saved results; no new observations were accessed and no candidate was rescored.

| Registered smoothing width | Failed model–cluster cases / 72 | Sources outside preset fidelity limits |
| --- | --- | --- |
| 0.0025, primary | 44 | None |
| 0.005 | 19 | A85 |
| 0.01 | 0 | A85 |

The primary failures occur in A1795, A2142, A2319, A85 and ZW1215, all of which have supplied stellar profiles. The recorded bad accelerations are finite and nonpositive. These are repeated model calculations, not independent observations.

Increasing smoothing removes the recorded force failures, but the wider A85 profiles exceed the inherited stellar-source fidelity limit. We therefore retain the original nine candidates as unscored in the campaign. Selecting the widest source after seeing the failures would not establish success. Conversely, the strong source sensitivity prevents treating these calculations alone as a robust physical exclusion of the candidate family.

The next discriminating calculation should isolate the terms involving spatial derivatives of the source field, then assess whether the force reversals persist for independently justified smooth matter profiles within source uncertainties. Agreement in enclosed mass alone does not establish the accuracy of its derivatives. Neither pressure agreement nor numerical force positivity establishes lensing, dynamical stability, or Solar System success.

Reproducible evidence: `cluster-width-failure-audit-001/result.json` and its saved `audit.py` in the research worktree. The result records SHA-256 hashes of both original inputs and all 216 model–cluster–width cases.
