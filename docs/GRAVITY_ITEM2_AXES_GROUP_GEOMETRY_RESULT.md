# Gravity roadmap Item 2: AXES group geometry result

## Decision

**INCONCLUSIVE_ITEM2_AXES_GROUP_GEOMETRY_QUALITY_GATE**

This fifth Item 2 attempt does not pass.  It adds a real intermediate-scale population
and direct member dynamics, but projected group geometry is not a reliable incremental
cause of the measured response after luminosity, size, richness, redshift, and environment
are controlled.

The immutable receipt is
`runs/gravity/roadmap/item-02-axes-group-geometry-v5.json`:

- file SHA-256: `bff8bc67c27ba27cd1d4c9ac4662920ccc8248a342e9771e7169f71def298f6e`
- content SHA-256: `bb4df4b26bbf211de84837a99f44bbcfe7f6e805c41ce0e5c393265c2975e44d`

## What was frozen

The target-blind sample, geometry grammar, response, baselines, five-fold nested
cross-validation, admission gates, and confirmation prohibition were committed at
`e138203df70cefb6eb22aee7fd6e93fed2e95fa7` before any selected member row was opened.

The source is the public AXES-SDSS catalog `J/A+A/690/A52`:

- catalog DOI: <https://doi.org/10.26093/cds/vizier.36900052>
- article DOI: <https://doi.org/10.1051/0004-6361/202449591>
- CDS schema: <https://cdsarc.cds.unistra.fr/ftp/J/A+A/690/A52/ReadMe>

The catalog contains 8,465 cleaned FoF groups and 77,224 cleaned member rows.  Selection
queried only group ID, cleaned member count, median redshift, `r`-band luminosity, and
large-scale density.  It did not query the published `sigmaGAP`, `sigmaMAD`, `R200c`,
group mass, X-ray tables, lensing mass, or dark-halo parameters.

From 523 eligible groups at `0.003 <= z < 0.06`, the salted catalog-only rule froze:

| Cleaned members | Exploration | Sealed confirmation |
|---|---:|---:|
| 10–14 | 60 | 30 |
| 15–24 | 60 | 30 |
| 25+ | 60 | 30 |
| **Total** | **180** | **90** |

The 180 exploration queries returned 4,744 member rows and 832,343 bytes.  Two additional
queries to exploration group 1549 resolved a source-schema mismatch: the CDS ReadMe labels
coordinates `RAdeg/DEdeg`, while the ASU service exposes `RAJ2000/DEJ2000`.  `GalID` is the
member key; `SpecObjID=0` is a missing identifier and can occur on distinct rows.  These
operational corrections changed no object identity, role, hypothesis, model, or gate.

No confirmation group was queried.  Total confirmation target accesses remain zero.

## Target-blind representation and direct response

Before member redshifts are passed to a separate response function, positions and
luminosities produce:

- global axis ratio, ellipticity, and second- through fourth-order multipoles;
- inner/outer axis-ratio and quadrupole differences, centroid shift, axis twist, and
  outer multipole energy;
- minimum-spanning-tree length and diameter efficiency, projected linearity, and angular
  gap entropy.

The primary response is recomputed from member redshifts rather than copied from the group
table:

`v_i = c (z_i - median(z)) / (1 + median(z))`

`eta_L = sigma_gap^2 R_lum,rms / L_r,total`

The unknown universal luminosity-to-baryonic-mass conversion shifts the intercept.  It
does not by itself supply object-specific response information.  However, because
`eta_L` explicitly contains radius and luminosity, a high score from radius/luminosity
inputs is partly algebraic and is not evidence for a new gravity law.  Direct `sigma_gap`,
half-radius, and MAD-dispersion responses are therefore mandatory controls.

## Frozen quality result

Seventeen of 180 exploration groups fail the preregistered radial representation because
the luminosity-half-radius split leaves fewer than three finite members on one side.  All
failures were retained and no replacement was selected.  The 163 quality-valid groups
still contain 47, 56, and 60 groups in the three respective richness bins.

Because the gate requires all 180 exploration groups to pass, confirmation is forbidden
regardless of the predictive result.

## Held-out result

The nested selector gives:

| Model or selection | Overall held-out `R^2` | MSE |
|---|---:|---:|
| Luminosity + size | 0.634 | 0.03329 |
| Luminosity + size + richness + redshift + environment | **0.655** | **0.03145** |
| Global projected shape | 0.633 | 0.03339 |
| Radial/nonlocal shape | 0.621 | 0.03453 |
| Graph filamentarity | 0.642 | 0.03259 |
| All geometry | 0.629 | 0.03378 |
| All geometry + environment | 0.664 | 0.03059 |
| Nested selection across all families | 0.646 | 0.03222 |

The nested selected response is positive in every richness bin:

- 10–14 members: `R^2=0.622`
- 15–24 members: `R^2=0.733`
- 25+ members: `R^2=0.538`

That is a successful demonstration that the pipeline can recover predictable structure
from a real nonsynthetic data set.  It is not a shape discovery:

- folds 0 and 3 select the nonqualifying richness/environment baseline;
- folds 1, 2, and 4 select all geometry plus environment;
- the nested selected MSE is worse than the fixed richness/environment baseline;
- geometry fails its ablation requirement in at least one richness bin;
- a 199-run richness-stratified permutation test gives `p=0.145`, above the frozen 0.05
  threshold;
- the primary-selected geometry does not improve the nuisance baseline for direct
  `sigma_gap`, half-radius `eta_L`, or MAD-dispersion `eta_L`.

Only four of ten gates pass: positive overall `R^2`, positive `R^2` in every richness bin,
improvement over the weaker luminosity/size baseline, and untouched confirmation.

## Scientific interpretation

This result narrows the space of viable explanations.  Low-order projected multipoles,
radial differences, and member-graph filamentarity are not a sufficient universal cause
of group dynamics.  A geometry-plus-environment family has a small possible incremental
signal, but it is not permutation-significant, is not stable across response definitions,
and is confounded by the published membership algorithm.

The source authors produced membership using an FoF finder followed by the Clean algorithm,
which uses redshifts and mass-model assumptions.  This attempt therefore tests shape within
that released selection; it cannot establish independence from the group finder.  The
member luminosities also omit hot gas, diffuse intragroup light, cold gas, and variable
stellar mass-to-light ratios.

Do not open the 90 confirmation groups and do not retune this geometry grammar on the same
responses.  Combine this counterexample with Item 2 attempts 1–4 in a scoped synthesis,
then advance to Item 3 surface-versus-volume density unless the synthesis exposes an
untested shape dependency that is materially different from all five attempts.

## Replay

The raw exploration cache is local.  The committed source hashes, feature table, extraction
summary, and receipt support exact model replay:

```powershell
python -m sigma_theory_compiler.gravity_item2_axes_group_geometry_experiment --root . --check
python -m pytest tests/test_gravity_item2_axes_group_geometry.py -q
```

The run evaluated eight model/equivalence families, 43 model/ridge cells, 860 primary
inner-fold ridge fits, 80 recorded final/robustness fits, and 199 stratified permutations.
It made zero paid model calls, zero SPARC confirmation accesses, and zero direct-lensing
likelihood evaluations.
