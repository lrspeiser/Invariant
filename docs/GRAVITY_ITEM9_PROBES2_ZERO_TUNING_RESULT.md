# Gravity roadmap Item 9 attempt 2: PROBES-II zero-tuning replay

## Decision

`INCONCLUSIVE_ITEM9_PROBES2_QUALITY`

The preregistered absolute quality floor required 150 galaxies. Of 233 target-blind selected
identities, 136 galaxies with 4,604 rotation points pass every point and galaxy criterion.
The 58.4% retention fraction passes its separate 50% floor, but the absolute count misses by
14. The formal attempt is therefore inconclusive regardless of the diagnostic scores below.

The diagnostic is nevertheless a strong counterexample to promotion. The inherited
interior/exterior formulas modestly improve the fixed stellar-only RAR, but lose badly to a
flexible local baryonic-profile control in every prespecified broad stratum. This is not a
replication of the PROBES-I advantage over the strongest baseline.

## Independent real-data problem

The public PROBES-II release targets lower-mass and low-surface-brightness galaxies. Its
specific project page describes 631 late-type galaxies from 25 source compilations, 356 with
at least three optical bands; the expanded data landing page describes a 716-object release.
The pinned concrete files contain 435 metadata rows, 773 rotation-curve files, and 321
r-band light profiles.

Primary sources:

- PROBES-II data page: <https://mattfrosst.github.io/projects/11_project/>
- article DOI: <https://doi.org/10.1093/mnras/stac1497>

The source was pinned to the files served on 2026-08-28 with last-modified time
`Fri, 20 Feb 2026 22:17:50 GMT`. The metadata SHA-256 is
`a353f428fbd88cdcb798321341537c7f959edbcd195c3908a0718eb2310c19e4`; the profile archive
SHA-256 is `4b8d34e4e206be7773cbf5434c0fc3e02c6488faaf96163c1cdb30776e0adf0b`.

## Frozen response boundary

Commit `036dda3e` froze the hypothesis, exact formulas, ensemble, measurement rules,
exclusions, quality criteria, controls, strata, and 15 gates before the response archive was
downloaded. Commit `5423dfa0` then froze 233 identities and exact profile filenames after
reading only the metadata allowlist and ZIP central directory.

The identity normalization conservatively excluded every name in the PROBES-I attempt-1
sample and every galaxy in the earlier SPARC focusing receipt. The final selected sample has
zero overlap with both predecessors and 43--49 identities per fold. It selected one
rotation/r-profile pair per physical identity with target-blind lexicographic tie-breaking.
All 229 alternate rotation entries remain unopened.

Before sample freeze, the source stage opened zero ZIP payloads, zero rotation rows, and zero
photometry rows. After sample freeze it opened exactly the 233 selected light profiles and
rotation curves. No predecessor confirmation, alternate rotation entry, derived published
mass column, paid model call, or post-response formula was opened or generated.

## Zero-tuning formula replay

The atomic set contains exactly six formulas:

- the five acceleration-occupancy, log-radius-scale-one, `I_in-I_out` cells selected by the
  five independent PROBES-I outer folds;
- the exact earlier SPARC surface-brightness focusing cell.

The primary prediction is the pointwise median log-speed of the five PROBES-I cells. A mean
ensemble and each atomic cell are secondary sensitivity checks. PROBES II performs zero
candidate selection, coefficient fitting, operator choice, threshold choice, ensemble-weight
fitting, or LLM generation. Its only trained model is the frozen-architecture local ridge
competitor, trained and evaluated strictly out of fold.

All formulas remain labelled `COMBINATION`. No historical novelty claim is made.

## Quality result

- selected identities: 233;
- passing identities: 136 (`58.4%`);
- accepted rotation points: 4,604;
- quality failures: 97;
- failure flags: 63 inclination, 40 photometry overlap, 24 insufficient accepted points,
  three radial span, and one insufficient r-profile sampling. Flags may overlap.

The failed identities were not replaced. The 136 valid galaxies remain usable for a disclosed
diagnostic, but they do not retroactively lower the frozen 150-galaxy gate.

## Diagnostic result on the valid set

| Evaluation | Equal-galaxy MSE | R2 |
|---|---:|---:|
| Stellar Newtonian approximation | 0.129812 | -0.239 |
| Fixed stellar-only RAR | 0.048355 | 0.538 |
| OOF flexible local control / strongest baseline | 0.027036 | 0.742 |
| Frozen five-cell median nonlocal ensemble | 0.045038 | 0.570 |
| Frozen five-cell mean nonlocal ensemble | 0.045092 | 0.570 |

- The primary ensemble improves over fixed stellar RAR by `6.86%`.
- All five inherited acceleration-occupancy cells individually improve over fixed stellar
  RAR; their MSE range is `0.044979` to `0.045538`.
- The exact prior SPARC cell improves over fixed stellar RAR by `4.17%`.
- The primary ensemble is `66.59%` worse than the flexible local control.
- Its paired whole-galaxy sign-flip result against the strongest baseline is `p=1.0`.
- It loses to the strongest baseline in both distance halves, both stellar-mass halves, both
  surface-density halves, and both inclination halves.
- Four source families meet the frozen ten-galaxy threshold. The nonlocal ensemble helps one
  and regresses in three, including both selected GASS gas/star response families.
- Six of fifteen gates pass.

The coherent improvement over the fixed RAR says the inherited formulas are not numerically
random. Their large loss to a local profile model says the improvement is not evidence that
the nonlocal term is a universal new gravitational cause. On this lower-mass release, local
baryonic-profile variables explain much more of the response.

## Mechanical repairs

The following repairs are disclosed and do not alter an identity, formula, gate, threshold,
or prediction rule:

- `afefe594` treats only the optional HTTP weak-ETag prefix as transport-equivalent;
- `53653264` accepts missing trailing forbidden metadata fields while retaining only the
  fixed five-column allowlist;
- `c59f42fd` resolves filename prefix collisions using the longest exact normalized identity
  and derives source-family labels from filename citations;
- `a299bf8e`, after response opening, maps the release's explicit `ApparentMag` and
  `VelocityErr` headers to the already-frozen cumulative-magnitude and velocity-error roles.

The extraction was regenerated after the last repair, and the immutable result checker
rebuilds it exactly.

## Interpretation and next boundary

Attempt 1 remains a real, independently observed positive pattern, but it is now demonstrably
dataset- or representation-dependent: one PROBES-I survey failed, and the unchanged formulas
do not beat a strong local model in PROBES II. Retuning amplitudes, relaxing the 150-galaxy
floor, admitting failed inclinations, or opening the 229 alternates to search for a better
answer would invalidate the replay.

The appropriate synthesis is a scoped rejection of these exact stellar-light occupancy
formulas as a promoted universal law. It does not reject all nonlocal gravity, complete
baryonic interior/exterior fields, vector redirection, gas-inclusive kernels, or action-derived
boundary laws. Those materially different mechanisms remain eligible in later roadmap items.

## Replay evidence

- config SHA-256: `7cc174d80db1b88ead80c697353addd68c524c85159c759abc5187443583f4bd`
- candidate manifest SHA-256: `2472177d2f65d23eeafdc4d857484162c95ed8cd8f442089434139a57c26c1ae`
- source manifest SHA-256: `eb1b8bbcbca49378bdc2753b38bbdeae60c5a5562a4d361ca61e2d68146c110e`
- sample manifest SHA-256: `16b84f27c144f114a188f32cdbc9d9ac779c9dc54951199870e246819919e8fb`
- extraction summary SHA-256: `668b178c58b62f02ea217ef3fa55ec6c2724b1597aae296dc6a2dfc794251e05`
- feature table SHA-256: `a56ec7fc77ccd062a2d1039ac1fcf1023f36cdb54e76d92bf4ca1f90dd957251`
- response table SHA-256: `cd4e5cc38d1412a65703998a61bc330a55831007cc404d322c1c03c39e05f3ce`
- result file SHA-256: `09e2896cf78bd9782835ebbc31d833261dd8293ff3ebef967937add41b37de4f`
- result content SHA-256: `5250b829f1704b6a0cc1fc7a6e5497d0feb3ca47ac6816762de40e139c9deda7`
- replay command: `python -m sigma_theory_compiler.gravity_item9_probes2_replay check`
