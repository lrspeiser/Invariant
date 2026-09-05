# Direct calculation of the small length-dependent response

The complete run retains all 54 existing cards, three distances, both
thicknesses and five grids. It solves Poisson's equation for J(ell)-J(0)
directly using the exact logarithmic flux difference. The original subtraction
results remain saved. All 1,296 grid-comparison values were recomputed from
the saved arrays, and frozen inputs were verified. Lint and 278 focused code
tests pass.

The numerical target is a change below 1 percent of the reference signal's
peak, using a fixed full-force normalization. It is not a pointwise relative
bound or an observational significance threshold. Zero-length cards have
separate exact-zero controls and are not counted as resolved nonzero signals.

| Thickness | Refinement pair | Nonzero cases passing | Zero controls passing | Largest change / peak signal |
|---|---|---:|---:|---:|
| primary | angles vs baseline | 135/135 | 27/27 | 0.00646773044 |
| primary | multipoles vs angles | 0/135 | 27/27 | 0.155975815 |
| primary | radial vs multipoles | 45/135 | 27/27 | 0.029379486 |
| primary | boundary vs radial | 135/135 | 27/27 | 0.00854532313 |
| height_half | angles vs baseline | 0/135 | 27/27 | 0.0408310265 |
| height_half | multipoles vs angles | 0/135 | 27/27 | 0.223420127 |
| height_half | radial vs multipoles | 60/135 | 27/27 | 0.0210511104 |
| height_half | boundary vs radial | 135/135 | 27/27 | 0.00666779272 |

Angular sampling changes 320 to 640 Gauss nodes at fixed multipole order 80.
The multipole comparison changes order 80 to 160 at fixed 640 angular nodes.
Radial refinement changes 2,049 to 4,097 radial samples. The boundary grid
doubles the outer radius, halves the inner cutoff and uses 4,609 radial samples.
Every failed comparison remains a numerical failure; no physical model or
unfavorable case is removed because its field is harder to resolve.

## Interpretation and next work

The primary baseline confirms the importance of avoiding cancellation: at
0.001 pc the old subtraction differs by up to 4.1 times the direct signal's
peak. The two smallest direct responses approach quadratic length scaling
within about 7.3e-9 on that fixed grid. This internal consistency does not
establish angular or radial convergence.

The retained primary multipole comparison fails all 135 nonzero cases,
with a maximum change of about 15.6 percent. Increasing angular quadrature
alone passes that primary pair, so multipole truncation requires further
refinement. Select the next resolution using the complete pair table, keep
every existing card and thickness, and preserve the failed coarser results.

The provisional primary baseline's maximum model-to-model circular-speed
change is about 0.00218 km/s relative to the matching zero-length law. It was
calculated from model forces without reading or scoring measured velocities.
It remains provisional because the small-signal grid tests above are required.

The full objective still requires physical source and geometry uncertainty,
all registered source scenarios, direct galaxy-outskirts stellar data, cluster
dynamics and lensing, precision Solar System predictions, conservation,
symmetry, stability, a complete matter/light law and independent validation.
This numerical campaign adds no observational score or physical exclusion.

## Evidence

- `direct-difference-001`: `299f258373aa0afc58511330f4f7dcd6e78a5661a8a487bf18b05df01c945d6c` (51 verified input snapshots)
- `tensor-controls-011`: `2b5994ad11662d1513c8103f90848d7b11d8cc5f16b674ac04d3fdf357dd4b6f` (138 verified input snapshots)
