# Gravity roadmap Item 4: baryonic compactness

## Decision

**`REJECT_ITEM4_TESTED_PROJECTED_BARYONIC_COMPACTNESS_FAMILIES_ADVANCE_ITEM5`**

Item 4 is complete for the tested projected-light compactness families. This is a scoped
rejection: it does not reject every possible compactness-dependent theory. It establishes
that the frozen mass-to-size, projected pair-binding, center-depth, and inner-versus-outer
potential variables do not add a reliable new cause of optical-group velocity dispersion
beyond ordinary mass, size, richness, redshift, and environment controls.

The immutable attempt receipt is
`runs/gravity/roadmap/item-04-baryonic-compactness-v1.json`; the deterministic closure is
`runs/gravity/roadmap/item-04-synthesis-v1.json`.

## What was tested

For member light converted with a fixed mass-to-light ratio, the target-blind derivation
measured a luminosity-weighted projected size `R_rms` and softened pair kernel

`K_pair = sum_(i<j) m_i m_j / sqrt(r_ij^2 + epsilon^2)`,

with `epsilon=max(20 kpc, 0.05 R_rms)`. It then formed the known gravitational radius
`R_g=M^2/K_pair`, pairwise virial scale `sqrt(G K_pair/M)`, center-potential depth,
inner-versus-outer potential contrast, potential dispersion, and a frozen pair/center
interaction.

The `GM/(R c^2)` and `GM/(R^2 g_dagger)` terms were explicitly labeled algebraic
mass-size rewrites. The pairwise virial scale and gravitational radius were labeled known
constructions. Only structure beyond those controls could qualify as a potentially new
synthesis; historical novelty was never claimed.

## Real-data boundary

The test selected 52 entirely unused AXES optical groups before opening member rows: 40
groups with 10–14 members and 12 with at least 15 members. Another 21 groups were frozen
as confirmation and remain unopened. None of the 450 group IDs used by Items 2 or 3 was
reused.

All 52 exploration groups passed the frozen source and representation checks, yielding 912
member rows. Compactness was finalized from member positions and light in a function that
cannot accept member redshift. Only afterward did a separate function recompute Beers
gapper and MAD velocity dispersions. No published group dispersion, `R200`, `M200`, X-ray,
lensing, or halo column was read.

## Held-out result

Whole groups define five folds, stratified by the two richness regimes. Eight formula
families and 43 family/penalty cells were compared. The unrestricted selector chose a
qualifying potential family in only one fold; three folds chose the nonqualifying
mass-size rewrite and one chose the strongest nuisance control.

The unrestricted held-out prediction is real but not uniquely structural:

| Metric | All 52 | Richness 10–14 | Richness 15+ |
|---|---:|---:|---:|
| Held-out `R^2` | 0.456 | 0.435 | 0.362 |
| MSE, log-dispersion | 0.00744 | 0.00566 | 0.01336 |

Restricting selection to qualifying potential families yields overall `R^2=0.489`, but
its improvement over the strongest nuisance baseline is only `0.000023` MSE. It improves
the 10–14-member groups and loses in the 15+ group stratum. A 499-permutation test gives
`p=0.348`, so this tiny incremental gain is not distinguishable from chance under the
frozen test.

The MAD-response robustness control improves overall, but the pairwise virial ratio does
not reduce total dispersion relative to the simple mass-size virial control. Four of eight
gates pass: data quality, positive within-stratum prediction, MAD robustness, and sealed
confirmation. The incremental-cause, permutation, universal selection, and virial-ratio
gates fail.

## Meaning and failure-space boundary

The system did discover a predictive relationship, but the controls show why prediction
alone is insufficient. In this sample, projected baryonic compactness mostly repackages
mass and size. The attempted inner/outer synthesis does not earn evidence as a new cause.

Later search should skip these equivalence regions:

- `log(GM/(R c^2))` and `log(GM/(R^2 g_dagger))` as new laws;
- `sqrt(G K_pair/M)` relabeled as a novel velocity formula;
- `R_g=M^2/K_pair` relabeled as a novel radius formula;
- retuning these same projected-light summaries on the opened 52 responses.

This result does not cover complete three-dimensional baryons, resolved galaxy/cluster
concentration, nonlocal interior/exterior kernels, baryonic boundaries, field curvature,
or compactness derived from a new action. Those belong to later roadmap items.

## Replay

```powershell
python -m sigma_theory_compiler.gravity_item4_baryonic_compactness check-sample
python -m sigma_theory_compiler.gravity_item4_baryonic_compactness_experiment --check
python -m sigma_theory_compiler.gravity_item4_synthesis --check
python -m pytest tests/test_gravity_item4_baryonic_compactness.py tests/test_gravity_item4_synthesis.py -q
```

The run used 499 permutations, zero paid model calls, zero direct-lensing evaluations, and
zero confirmation accesses.
