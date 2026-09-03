# Gravity wells — analysis behind the landing page (September 2026)

The scripts that produced `runs/gravity/wells/*.json` and the numbers quoted in
`docs/GRAVITY_DISCOVERY_PROGRAM.md`. Everything here reads measured data; none
of it assumes dark matter.

`invariant_bench.py` is the shared harness: 4,255 measurements across seven
probes, with KiDS and wide binaries held as sealed blind holdouts.

## What each script does

| script | what it establishes |
| --- | --- |
| `m11_verify_identity.py` | Audits the X-COP cluster identity map. Found the pairing was done by rank of profile end, assuming a constant profile-end/R500 ratio; that ratio runs 1.12–2.16, so **11 of 12 clusters were misassigned**. The bench's `extent` *is* R500 in kpc, so exact matching was available all along. |
| `m12_redo_correct.py` | Redoes the temperature and merger-bias tests on the corrected map. Both results flip. |
| `p01_rigorous.py` | Referee-standard restatement: propagated errors, bootstrap intervals, exact permutation tests (2×10⁵ draws), power analysis, multiple-testing accounting. |
| `p02_systematics.py` | Shows the χ²/dof = 3.69 and the "κ = 10⁵ excluded" claim in `p01` were artefacts of statistical-only error bars. With 15% intrinsic scatter the model fits at χ²/dof = 0.93. **Both `p01` claims are withdrawn.** |
| `p03_audit_confound.py` | Tests the confound checker itself against variables with known answers. Three defects: pure noise passed; ties broken by array position gave a **global constant** corr = +0.948 with the dataset label; and both blind holdouts were being consumed by the check. |
| `p04_reaudit.py` | Re-runs every confound verdict with the defects fixed. |
| `p05_floor.py` | Calibrates the noise floor over 600 pure-noise draws. The bare dataset label sits 18× above it; no physical variable exceeds 4.7×. **The check separates signal from noise but cannot rank real variables or justify a kill on its own.** |
| `v01_extract.py` | Builds the measured potential wells for 10 SPARC galaxies and 12 X-COP clusters, anchored at the outermost measured radius and integrated inward. |
| `v03_lens_and_substructure.py` | Lensing: κ(R) = Σ(R)/Σ_crit for all 13 lenses. Substructure: 60 AXES-SDSS groups with real member positions, stellar mass from luminosity, total mass from member velocity dispersion. |
| `v04_lumpiness.py` | Solves QUMOND on a 192³ grid for a lumpy cluster (gas + 300 members) against the same mass smoothed spherically, answering whether spherical averaging invalidates a nonlinear-law test. |
| `v05_realizations.py` | Repeats that over five draws of the galaxy population to separate systematic bias from shot noise. |

## Results worth carrying forward

- **Both blind holdouts land on the acceleration relation.** They never entered any fit.
- **Newton/GR without dark matter: 0.542 dex. MOND: 0.115. GR + NFW per object: 0.112 — at 274 free parameters for 137 objects.**
- **Inferred halo mass is 94% reproducible** by a synthetic twin knowing only the baryons and the law (corr +0.943). Halo *shape* is neither reproduced nor predicted by any of 15 measured properties.
- **Halo concentration is prior-dominated**: changing the statistical prior moves log c200 by 0.401 dex against a galaxy-to-galaxy spread of 0.542 dex.
- **Seven candidate variables were eliminated** by controls after passing conventional significance testing. Two of them had been written up first.
- **The X-COP temperature correlation is not a discovery**: ρ = +0.615, p = 0.037 at n = 12, falling to ρ = +0.442, p = 0.20 on the ten clusters matched exactly rather than by elimination. Power at ρ = 0.6 is 0.51.
- **Spherical averaging does not rescue MOND on clusters.** Doing the nonlinear solve properly changes the answer by 0.8% in 3D and 2.7% in the innermost projected annulus, and it moves in the direction that makes MOND *worse*. The cluster discrepancy is a factor of two.

## Reproducing

The scripts expect the repository root as their data source and were run under
Python 3.13 with numpy, scipy and astropy. `v04` and `v05` are the expensive
ones: each solves QUMOND on a 160³–192³ grid several times.

Note that `invariant_bench.py` here is the **patched** harness — see
`p03_audit_confound.py` for what was wrong with the previous confound check and
why its verdicts should not be trusted without an independent control.
