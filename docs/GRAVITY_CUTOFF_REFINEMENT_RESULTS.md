# Cutoff interpolation failure: radial refinement resolves the tested case

Radial mesh refinement resolves the retained cutoff-200 source-identity failure
for both thicknesses over all old probes and 1,280 new off-grid probes.
Vertical refinement alone does not resolve the half-thickness failure.
The integration method, cutoff, physical source, gravity parameters and error
tolerances remain unchanged. **246 focused tests and lint pass.**

This is a successful refinement of one failed numerical case, not admission of
the full source or a new gravity law. The canonical source and all five
quadrature variations still need replay on the revised radial mesh before
the separate full-action Poisson calculation and any astronomical scoring.

## What caused the next experiment

At the exposed R=65, z=0.025 kpc point, a linear decomposition of the cutoff-200
minus cutoff-400 interpolated density-gradient difference identifies the
third-radial-derivative inputs as the largest contributors. The leading
signed vertical contributions, divided by the retained Hessian/radius scale,
are approximately +0.07057, -0.04463 and -0.01999 for input orders (3,0),
(3,1) and (3,2). These terms partially cancel. They are not independent
physical forces or fractions of the total error, and dropping them would
break the prescribed interpolation.

The summed linear contributions reconstruct the measured difference with
a scaled residual of 4.38e-6 from finite-precision evaluation. This is a
single exposed-point attribution, not a uniform diagnosis. It motivated
separate radial and vertical resolution tests; the earlier failed global-gauge
repair remains a negative result.

## Controlled refinement

The baseline uses 221 x 23 nodes. The radial refinement bisects every radial
interval wholly within 40--80 kpc, producing 241 x 23 nodes. The vertical
refinement bisects near-plane intervals within 0--0.2 kpc, producing
221 x 27 nodes; the combined mesh has 241 x 27 nodes.

Every previously stored mixed-derivative sample is retained bit for bit:
81,328 double-precision values per thickness, verified after serialization.
New integration values are used only at added nodes. The tests retain all
1,562 old probe entries per thickness and add 1,280 quarter/three-quarter
off-grid positions, for 2,842 entries per thickness. Overlaps in the inherited
probe sets remain; these are not independent observational measurements.

| Mesh | Nodes | Primary density-gradient error | Half-thickness density-gradient error | Half-thickness all-target verdict |
|---|---|---:|---:|---|
| baseline | 221 x 23 | 0.000466401686 | 0.0131153962 | fail |
| radial_only | 241 x 23 | 0.000466401686 | 0.00229235033 | pass |
| vertical_only | 221 x 27 | 0.000466401686 | 0.0130722305 | fail |
| both | 241 x 27 | 0.000466401686 | 0.00229011767 | pass |

The density-gradient target remains 0.01, using the same physical-source
normalization as the preceding audits. The radial-only half-thickness worst
error is now at R=36.0625, z=0.025 kpc, rather than the original R=65 point.
Thus the result is not just a failed probe becoming an interpolation node;
the new off-grid positions also satisfy the registered requirements.

Radial-only refinement changes the half-thickness field by at most
1.87287e-8 in scaled force, 5.91898e-6 in Hessian and 0.00322813 in third
tensor, all below the unchanged targets. Its density and density-gradient
identities pass for both thicknesses. The combined refinement also passes,
but adds little improvement to the worst density-gradient error. Vertical-only
refinement remains a failed intervention for the half-thickness source.

## Next required gate

Use the radial-only refinement as the next numerical candidate. Rebuild the
canonical cutoff-400 representation and replay radial, wavenumber, cutoff,
vertical-resolution and tail-extent variations on that same mesh, retaining
both thicknesses and every probe, including the new off-grid set. Recheck
independent derivatives at added radial boundaries. Preserve the old failed
grid and failed vertical-only intervention; do not rewrite their verdicts.

Only after source qualification should the separate full-action Poisson
solve be run and refined before astronomical rescoring. Cross-regime matter
and light predictions, precision Solar System fits, stability, direct outer-star
data and independent confirmation remain requirements of the active goal.
This checkpoint adds no observational score, validated law or physical exclusion.

## Evidence

- `tensor-jet-contributions-001`: `a10e4956a97749e314645659156a1fdc82effd2415134e28839f729057d6ddb5` (42 verified snapshots)
- `tensor-cutoff-refinement-001`: `7a6ae289ba7595d6bef1c5f080848ad3ea18592a34b59ecdb4e8f1f4d93166f0` (48 verified snapshots)
- `tensor-controls-006`: `5f1ba6933ee0703116c48393c22dad58e79a2d1c692cbec7cdb4b3a04fb43e08` (129 verified snapshots)
