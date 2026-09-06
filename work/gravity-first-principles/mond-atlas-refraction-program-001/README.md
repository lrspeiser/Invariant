# Actual spatial refraction operator executed; field refinement remains unresolved

We implemented and ran the actual three-dimensional density-dependent equation on conditional NGC2976 stellar, HI and CO sources:

`div[epsilon(rho) grad Phi] = 4 pi G rho`, with `g = -grad Phi`,

`epsilon(rho) = 0.2 + 0.8 rho/(rho + 10^7 Msun/kpc^3)`.

This is a spatial elliptic solve with harmonic face transport, not a radial density multiplier. The density threshold, vacuum response and power were frozen as illustrative values before evaluating fields; none was fitted to velocities or changed to improve numerical results. The operator is the phenomenological refraction equation described by [Matsakos & Diaferio2016](https://arxiv.org/abs/1603.04943) and the [nonrelativistic discussion in Sanna et al.2023](https://www.aanda.org/articles/aa/full_html/2023/06/aa43553-22/aa43553-22.html). Our chosen rational transition is a declared test prescription, not a claim to reproduce published best-fit parameters.

**The equations solve accurately, but the refraction field is not yet sufficiently resolution-independent.** That distinction is the main result. No observed velocities or lensing were scored, and no gravity law was validated.

## Executed source and solver work

The source is the existing hashed f4-stars-h0p4 reconstruction, with stellar M/L0.6, atomic gas including helium, and CO conversion4.35/0.65. The assumed stellar exponential height is0.4kpc; gas heights are0.2kpc. Bilinear planar source integrals and exponential vertical cell integrals are evaluated exactly for the declared basis, without mass renormalization. Total untruncated source mass is approximately2.440 billion solar masses. Its three-dimensional density is conditional on those reconstruction assumptions, not directly measured depth.

Initial execution included Newton and refraction on three domains/grids for that source, plus both laws on the thinner f4-stars-h0p1 source at base resolution: eight solved potentials. Changing the thin-source case changes fitted planar coefficients as well as height, so it is joint source/deprojection sensitivity rather than an isolated height experiment.

All grids use fixed isolated monopole Dirichlet boundaries; refraction's far boundary includes the declared vacuum epsilon0.2. This is an assumed isolated exterior, not a measured surrounding environment. Initial grids used spacings(.25,.25,.125) and(.125,.125,.0625)kpc inside half-widths(8,8,4)kpc; the box check enlarged half-widths to(12,12,6) at base spacing. A separately frozen follow-up used257³ nodes and spacings(.0625,.0625,.03125)kpc in the same original domain.

## Numerical checks and retained failures

Uniform-permittivity quadratic potentials agree at approximately3e-16 relative RMS. A known variable-permittivity quadratic solution improves from2.374e-5 to6.601e-6 when the test grid doubles. Positive symmetric harmonic coefficients, source-mass arithmetic and analytic exponential slab mass checks passed before source evaluation.

Independent face-difference replay of all eight initial saved potentials verified both the PDE residual and the net flux through the discrete boundary. This independently checks the equation implementation while sharing the audited source adapter. It does not prove spatial convergence.

| Comparison | Newton vector RMS change | Refraction vector RMS change | Frozen overall threshold |
|---|---:|---:|---:|
| Initial base to fine | 12.56% — failed | 16.45% — failed | 5% |
| Base to enlarged box | 0.239% — passed | 0.831% — passed | 5% |
| Fine to257³ finer grid | 4.142% — passed | **13.171% — failed** | 5% |

Height-specific refinement threshold is8%. Newton's finer-grid groups pass (maximum6.26%); refraction's changes remain16.23%,14.92%,11.87%,8.62% at heights0,.2,.5,1kpc. All solver residual/flux gates pass, including the final refraction physical residual8.93e-9 against1e-8 and flux balance2.31e-10. A small algebraic residual is therefore not sufficient to establish a resolved physical field.

The finer run first encountered a manufactured Plummer force-sampling failure:65 nodes gave1.195% error against the frozen1% threshold. That failed attempt and exact runner were preserved. Adding129 nodes to the manufactured control reduced error to0.2443% without changing equations or tolerances. The subsequent actual257³ calculation then completed in37.45seconds on one CPU thread. The Plummer reference validates the Newton potential-gradient/interpolation chain against an analytic force, not direct integration of the actual galaxy.

The initial raw strength/direction diagnostics are retained in `run001/summary.json`, but **must not be quoted as converged galaxy predictions**. Density-dependent coefficients remain substantially more resolution-sensitive than the same-source Newton comparison. The next numerical task is to identify and resolve that coefficient/source-interface sensitivity and compare further refinements or an independent discretization; changing the density law to hide the error would not fix the numerical issue.

## Scope, files and resource use

Ten conditional potentials were solved overall; only the initial eight were saved privately (34,414,214 bytes). The finer run saved sampled vectors and diagnostics only, well below the1GB raw-storage allowance. No GPU was required. Initial and refinement protocols, failures, source hashes, every sampled vector and numerical result are retained in this package.

- `scripts/mond_atlas_refraction_program.py`: harmonic-face operator, source integration and initial execution.
- `scripts/mond_atlas_refraction_program_review.py`: independent discrete flux replay.
- `scripts/mond_atlas_refraction_program_finer.py`: bounded numerical refinement and Plummer control.
- `run001/independent-flux-review.json`: independent PDE/flux evidence.
- `finer001/`: preserved failed manufactured control and exact earlier runner.
- `finer002/summary.json`: successful solves with failed refraction refinement explicitly retained.

The only imported source numerical helper is the tracked `scripts/mond_atlas_source_resolution.py`; its hash is bound. Inputs and primary source references are inherited through the bound sibling `mond-atlas-spatial-program-001/source-bindings.json`, with every packet hash checked before use.

Unknown depth, original source fitting floors, beams, missing matter, registration, conversion and exterior uncertainties remain. This nonrelativistic PDE does not supply a photon metric, energy-transfer mechanism, causal dynamics or stability proof. Status remains **conditional mathematical field development; observed-response admission blocked**. It is substantive progress beyond radial proxies, with a specific numerical limitation still to resolve.
