# A finite density-response scale passes the declared spatial checks

Both newly declared finite-scale prescriptions passed this bounded numerical refinement and boundary experiment. This is a useful computational lead beyond a radial proxy. It does **not** show that a galaxy has the predicted extra gravity, that the physical scale is correct, or that coherence/time supplies energy.

We changed only the density used to set permittivity:

`rho_s = Gaussian(ell) * rho`,

`epsilon = 0.2 + 0.8 rho_s/(rho_s + 10^7 Msun/kpc^3)`,

`div(epsilon grad Phi) = 4 pi G rho`.

The original, unsmoothed matter density stays on the right-hand side. The two frozen Gaussian standard deviations are ell0.25 and0.5kpc. These are **new nonlocal physical prescriptions**, not numerical fixes of the earlier point-density equation. Neither scale nor any force parameter was fitted. The original point-density failures remain in the parent package.

## Executed results

Each prescription was solved on base, enlarged-box, fine and finer grids using the same NGC2976 conditional stellar/HI/CO source. Eight new solves completed in80.45seconds on one CPU thread, with no new full-field storage. The257³ finer grid and source adapter match the preceding Newton/reference experiment.

| Density scale | Fine-to-finer vector RMS change | Largest height-group change | Own enlarged-box RMS change |
|---|---:|---:|---:|
| 0.25kpc | **3.1045%** | 4.7459% | 1.0594% |
| 0.5kpc | **3.3960%** | 5.1697% | 1.0988% |

Frozen limits were5% overall and8% for each height group. All comparisons pass; no original point-law boundary result was substituted for the new prescriptions' box tests. Maximum physical PDE residual was8.76e-9 against1e-8; all discrete boundary-flux gates passed. Finite-domain Gaussian smoothing loses at most0.00556% of grid mass, below the frozen0.1% check, without renormalization. Grid, interior RHS and excluded boundary-node masses are saved for every run.

Pre-source Gaussian tests verify positivity, interior preservation of a constant density, mass conservation for a centered impulse, and agreement with independently written direct kernel convolution. The separate parent cross-review additionally checks explicit-kernel convolution on actual source grids. Initial uniform/variable-permittivity and independent face-flux solver checks remain bound; the new density prescription does not replace them.

## What the conditional fields do

Against the same-source Newton field at the same257³ grid and384 declared sample points, the0.25kpc prescription has median force magnitude1.557times Newton and median direction change4.27degrees. The0.5kpc prescription gives1.364times and3.23degrees. At height1kpc, the median magnitude ratios are1.923 and1.492. These are distributions over fixed mathematical sample positions, not observational errors, circular speeds or significance estimates.

The important structural behavior is that the solve changes both force strength and direction. It is not equivalent to multiplying each Newtonian vector by a local density factor. Some sample positions in the0.25kpc model have a weaker force magnitude, illustrating redistribution rather than a universal local boost. Individual extrema and direction changes should not be treated as precisely validated numbers merely because an overall vector RMS gate passed.

The fields depend materially on the newly introduced physical scale. That remains a model choice needing independent observational constraints. The improvement over the unsmoothed equation is numerical evidence that these finite-scale prescriptions are more tractable on the tested source/grid family, not proof that unresolved gradients were the sole cause or that nature uses Gaussian coherence.

## Remaining work and preserved scope

No velocities, lensing responses or halo-fit targets entered this branch. The source has assumed depth, conversion, nonnegative reconstruction and missing-phase/beam uncertainties. Passing one joint grid refinement and a coarse box check does not establish an asymptotic error bound, independent axis convergence, environmental robustness or transfer to other galaxies. This nonrelativistic equation still lacks an admitted energy-transfer model or photon metric.

Next observational work requires the existing source/motion/noise admission gates and independent responses. No further smoothing lengths were tested after seeing these outcomes. All failures, including the original point-density field and manufactured Plummer resolution failure, remain available.

Artifacts: `PREFLIGHT.md`, `run001/gaussian-controls.json`, `run001/progress.json`, `run001/summary.json`, and every sampled vector. `conditional-field-effects.json` contains source-matched Newton comparisons. The runner is `scripts/mond_atlas_refraction_program_coherence.py`; previous files and their completion hashes were not changed.
