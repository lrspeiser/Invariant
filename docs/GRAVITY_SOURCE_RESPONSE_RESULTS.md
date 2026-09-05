# Numerical source variations through the full gravitational action

All five retained source-representation variations have been propagated through
the fine-grid Poisson solver for both thicknesses, all 54 existing global cards
and all three distances. The 1,620 comparisons below each retain 97 radii.
Every reported force and small-signal change was recomputed from saved arrays;
all frozen input snapshots were verified. Lint and 246 focused tests pass.

| Thickness | Source variation | Passing full-force comparisons | Largest scaled force change | Largest scaled length-signal change |
|---|---|---:|---:|---:|
| primary | radial_coarse | 162/162 | 1.12731116e-12 | 7.87217125e-13 |
| primary | wavenumber_coarse | 162/162 | 8.95113508e-13 | 9.95992758e-13 |
| primary | cutoff_200 | 162/162 | 1.41658801e-10 | 4.28571119e-11 |
| primary | vertical_coarse | 162/162 | 1.12935286e-10 | 4.64278035e-11 |
| primary | tail_extent | 162/162 | 4.24553739e-13 | 4.22541636e-13 |
| height_half | radial_coarse | 162/162 | 3.51633283e-12 | 1.34463065e-12 |
| height_half | wavenumber_coarse | 162/162 | 8.10577932e-13 | 8.59624246e-13 |
| height_half | cutoff_200 | 162/162 | 1.10797953e-09 | 2.01012664e-10 |
| height_half | vertical_coarse | 162/162 | 9.17639645e-11 | 6.2982468e-11 |
| height_half | tail_extent | 162/162 | 7.06537302e-13 | 7.05892209e-13 |

The full-force diagnostic limit was fixed at 0.002 before this run: one tenth
of the inherited 2 percent solver-refinement target. It is not an observational
accuracy threshold. Numerical source representation is not physical source
uncertainty; stellar mass, gas content, thickness assumptions, geometry and
other registered source scenarios remain separate requirements.

## Why small length effects remain open

The saved solver-grid audit shows that a small full-force change can still be
large relative to the nonzero-length minus zero-length signal. Worst ratios
reach about 35 percent for lengths from 0.1 to 10 pc, about 36 percent at 0.01 pc,
and exceed three at 0.001 pc. These ratios compare maxima within each matching
case, potentially at different radii. They are descriptive numerical checks,
not calibrated error bars or detection significance.

The cancellation diagnostics retain 42 synthetic cases and 972 actual-source
sample cases, including 252 exact-zero controls. Fixed quadrature fails 54 of
the source cases; a conditioned combination of integration and direct
subtraction passes all 972. These are compared with 80-digit arithmetic for
nonzero references. The method has not been admitted for the entire source
domain or its branch transition. The production runs above retain
their original implementation and results. Neither a numerical failure nor
an unresolved small effect is converted into a physical-family exclusion.

Next, validate a direct flux-difference implementation over the actual source
domain, including small gradients, and refine angular response relative to the
small signal. Continue physical source/geometry propagation before new
observational rankings. No parameter choices are removed because they are
harder to resolve. Direct outer-star data, cluster dynamics and lensing,
precision Solar System observations, a complete matter/light law, stability
and independent validation remain requirements of the full goal.

This checkpoint adds no observational score or physical exclusion.

## Evidence

- `tensor-source-response-001`: `e8b48f5946906ecf35456bd826169adc599136cf9cc4ac565ee1be9ea6bffd51` (57 verified snapshots)
- `tensor-length-signal-001`: `3a5bce7b1f59f156b15a427a4fb93dd98cb7b2d5b5aa3095d4301fc4034533b1` (3 verified snapshots)
- `tensor-controls-009`: `14ab6833c3505cb3cf7ee48d7effeed8b8f89148c4d6816e81c15e099d959cdc` (133 verified snapshots)
- `length-cancellation-001`: `3769992a2b2f1cad7bb3f1454e54bf4cd948c9b254f63d90951d7529f4e46303` (2 verified snapshots)
- `source-cancellation-001`: `22d1343cb75b21d79bc2e028436e56c8051fd494dbe2dc19c6cd265fd85f1dc7` (45 verified snapshots)
- `source-cancellation-002`: `25a87a8aded4a5bbd7411597f9d5b68f462308c94f0a91528d54df771c4ce320` (45 verified snapshots)
