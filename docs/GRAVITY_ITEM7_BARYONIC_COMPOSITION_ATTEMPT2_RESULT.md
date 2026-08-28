# Gravity roadmap Item 7: independent baryonic-composition replay

## Decision

`INCONCLUSIVE_ITEM7_COMPOSITION_REPLAY_QUALITY_GATE`

The PHANGS composition lead did **not** replicate on the independent xCOLD GASS plus
GASS/xGASS exploration sample. One of 96 frozen galaxies has a published zero width
uncertainty, so the attempt is formally inconclusive under the all-96 quality rule. On the
95 valid galaxies, however, every preregistered performance diagnostic favors the known
baryonic baseline over the replayed nonlinear phase family.

## Independent real-data problem

The frozen test asks whether the exact nonlinear stellar/atomic/molecular phase-balance
family retained from PHANGS predicts a different kinematic response in a materially larger
sample:

- composition and structure: Saintonge et al. (2017) xCOLD GASS stellar mass, molecular
  mass, optical half-light radius, concentration, and inclination;
- atomic gas and response: the four GASS/xGASS releases' H I mass, signal-to-noise,
  corrected 21-cm width, and published width uncertainty;
- primary response: `log10[W50c/(2 sin i)]`;
- known controls: fixed baryonic Tully-Fisher, fixed Newtonian mass/size, flexible stellar
  or total mass/size/acquisition terms, and raw phase main effects;
- qualifying replay: the already frozen phase entropy, phase boundaries, ratio curvature,
  and composition-by-structure interactions.

The response-blind audit identified 129 complete CO-detected, good-H I, moderately inclined
galaxies. A salted mass-by-molecular-ratio split assigned 96 to exploration and sealed 33
for confirmation before any H I width value was requested. The PHANGS confirmations also
remained sealed.

Primary sources:

- xCOLD GASS: <https://doi.org/10.3847/1538-4365/aa95c1>
- GASS data release 1: <https://doi.org/10.1111/j.1365-2966.2009.16298.x>
- GASS data release 2: <https://doi.org/10.1051/0004-6361/201219261>
- GASS data release 3: <https://doi.org/10.1093/mnras/stt1413>
- xGASS low-mass extension: <https://doi.org/10.1093/mnras/sty089>

## Leakage and quality boundary

Commit `9eb2bf66e5ba6867cca8ffa81f4621c2881717fd` froze the sample, formulas,
folds, null, and gates before response access. Acquisition then made exactly one
ID-constrained width query for each of the 96 exploration galaxies. Stored URLs are
mechanically validated to contain one frozen exploration ID and the exact allowed columns.

There were zero xGASS confirmation, PHANGS confirmation, dark/dynamical-mass, lensing-mass,
or paid-model accesses. `GASS 114010` has a valid quality-1 width of `249 km/s` but a
published uncertainty of `0 km/s`; it was excluded by the frozen uncertainty rule. It was
not replaced, and the rule was not relaxed.

## Results on the 95 quality-passing galaxies

| Evaluation | Held-out MSE (dex squared) | Held-out R2 |
|---|---:|---:|
| Fixed baryonic Tully-Fisher / strongest nonqualifying selector | 0.006603 | 0.671 |
| Replayed qualifying composition selector | 0.008137 | 0.594 |

- The replayed family increases MSE by `23.23%` rather than improving it.
- The unrestricted selector chooses fixed baryonic Tully-Fisher in all five outer folds.
- The qualifying family loses in both mass strata, both molecular-ratio strata, and both
  high- and low-mass GASS release strata.
- Both published width-error envelopes preserve the loss.
- The 499-permutation result is `p=0.882`; the observed MSE change is in the wrong direction.
- Only two of eleven frozen gates pass: positive qualifying `R2` and untouched confirmations.

## Interpretation

This result does not show that baryonic composition is irrelevant to galaxy dynamics. It
does show that the exact global phase-entropy, boundary, curvature, and
composition-by-structure family that looked promising in PHANGS does not generalize to an
independent H I-width problem. The original `NONPROMOTED_POSITIVE_LEAD` is therefore archived
as a failed independent replay, not promoted and not silently deleted.

Future composition work must add materially new information—especially resolved gas and
stellar geometry, ionized/plasma phases, or a prior action-derived coupling—rather than
retuning these functions on either opened response set. Item 8 now tests the distinct field
gradient and curvature variable family.

## Replay evidence

- config SHA-256: `4a15302decc4a2017e7c5d4b341333124c0067b90dbd13163d07bb91acabc8a2`
- sample manifest SHA-256: `fa80a1993bdc9ae18e611036d56ab83c89a944d5461c846dcc03c7c33b45af29`
- source manifest SHA-256: `603e051407f9c7016830f415571f02303f8e2a2033847b2ae6f7bc1a2fdfe5f1`
- feature table SHA-256: `4706213b55d809f59c94350d991ecd9cac958c6bff1355f2eeeaf6924db0dcbd`
- extraction summary SHA-256: `7e542367a930d7addef05cf398611a5616b4178679d5e1308c39e5e9c0a0a6f7`
- result file SHA-256: `a27089b5fc2fc5f7223b1ba67f91da7fade08d20efdf7ee2eccdc496a275b71e`
- result content SHA-256: `12bf6ba7f5ff691daa7f3b28d1e4ca4a3394c2c80eb31afbe48d01f5fbe5b4b8`
- replay command:
  `python -m sigma_theory_compiler.gravity_item7_baryonic_composition_replay check`

