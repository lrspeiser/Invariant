# Full-action numerical field campaign

The separate Poisson calculation has completed for all 54 existing global
parameter cards, three distance assumptions and both qualified disk thicknesses.
Five numerical grids per thickness give 1,620 field predictions at 97 registered
radii each. All finite family outputs and zero-length distance invariance were
checked after serialization. Lint and 246 focused code tests pass.

The table reports numerical comparisons, not fits to observations. Flux was
rotated into the solver's spherical components, its divergence solved through
the Green-function solver, and the direct Newtonian gradient added afterward.
The physical acceleration is the negative potential gradient; the flux itself
is not substituted for the physical force.

| Thickness | Alternative compared with fine | Passing cases | Largest scaled force change | Limit |
|---|---|---:|---:|---:|
| primary | coarse | 162/162 | 0.00387667064 | 0.02 |
| primary | boundary | 162/162 | 4.6625625e-06 | 0.005 |
| primary | radial_only | 162/162 | 0.00387698216 | 0.02 |
| primary | angular_only | 162/162 | 6.06560692e-05 | 0.02 |
| height_half | coarse | 162/162 | 0.00380963401 | 0.02 |
| height_half | boundary | 162/162 | 5.46773698e-06 | 0.005 |
| height_half | radial_only | 162/162 | 0.00380793674 | 0.02 |
| height_half | angular_only | 162/162 | 6.15840092e-05 | 0.02 |

All force refinements pass: **True**.
All reflection checks pass: **True**.
All radial force branches are inward: **True**.

The inherited coarse/fine force target is 0.02; the boundary target is 0.005.
The boundary comparison doubles the outer limit and halves the inner cutoff.
Radial-only and angular-only grids isolate numerical contributions, while the
fine grid changes both. The solver zero-extends anomalous flux outside the
finite shell. These comparisons do not prove an infinite-domain or uniform
continuum error bound.

## Small effects and remaining requirements

Every nonzero-length result retains its force difference from the matching
zero-length action at the same shape, acceleration scale, distance, source and
grid. Changes of that difference between grids are stored separately from total
force changes. Passing the total-force tolerance does not by itself resolve
these much smaller effects, establish an observational detection or identify a
preferred length. No small length effect is admitted at this checkpoint.

Before new observational rankings, propagate the retained source-representation
variations through this solve and assess angular/radial convergence relative to
the claimed signal, not merely the much larger total force. The other physical
source scenarios and map uncertainty remain pending; the two thicknesses here
do not replace the full registered source/geometry scope. Preserve unfavorable
or unresolved cases. No new observational scores or physical exclusions are
introduced by this numerical campaign.

The full goal still requires a single global law across direct galaxy-outskirts
stellar dynamics, cluster dynamics and lensing, and precision Solar System
observations, with conservation, symmetry, stability, a complete matter/light
prescription, known limits and independent confirmation. The current static
action and numerical checks do not establish those requirements.

## Evidence

- `tensor-poisson-001`: `5c5d19ea954df993f7e4e2104d6495257eb7b7995c0bb9cf56fb12f0ad656306` (47 verified input snapshots)
- `tensor-controls-008`: `0fee3edaffa449558530131b4b02e7ed77e14aff78f1d814ee80396eaf991ac3` (131 verified input snapshots)
