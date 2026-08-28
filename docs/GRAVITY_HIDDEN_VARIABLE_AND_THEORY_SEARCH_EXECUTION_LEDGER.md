# Gravity hidden-variable and theory search execution ledger

This is the mutable status companion to
`docs/GRAVITY_HIDDEN_VARIABLE_AND_THEORY_SEARCH_GOALS.md`. The goals file defines the stable
72-item search and testing contract; this ledger records what has actually been attempted.
Its embedded execution table is a frozen legacy snapshot because the Item 1 receipt binds
that entire file byte for byte. Update this companion ledger, not the goals file, when new
attempts finish.

| Item | Current status | Evidence | Required next move |
|---|---|---|---|
| **1. Effective dimension** | **INCONCLUSIVE** — the specific `beta=1/D_support` rule is rejected by the real cross-scale diagnostic; continuous baryonic mass-profile dimension does not predict the response; a binary disk/cluster label reproduces population means but is not a cause. | `GRAVITY_ITEM1_EFFECTIVE_DIMENSION_RESULT.md`; `runs/gravity/roadmap/item-01-effective-dimension-v1.json` | Continue through Item 2 to measure shape and anisotropy continuously, add intermediate/filamentary geometries, then freeze a rule before independent confirmation. |
| **2. Shape and anisotropy** | **INCONCLUSIVE, FIVE ATTEMPTS; READY FOR SCOPED SYNTHESIS** — attempt 5 adds 180 frozen AXES-SDSS groups across three richness strata, recomputes dynamics from 4,744 member redshifts, and tests projected multipoles, radial shape, and graph filamentarity. Seventeen groups fail frozen radial representation and are not replaced. On 163 valid groups the nested response is predictable (`R^2=0.646`, positive in every richness bin), but luminosity/size/richness/redshift/environment already give `R^2=0.655`; geometry is selected in only three of five folds, its permutation result is `p=0.145`, and it fails every response robustness control. All 90 confirmation groups remain unopened. Attempts 1–4 remain retained counterexamples. | `GRAVITY_ITEM2_SHAPE_ANISOTROPY_RESULT.md`; `GRAVITY_ITEM2_WISE_MULTIPOLES_RESULT.md`; `GRAVITY_ITEM2_STELLAR_MULTIPOLES_RESULT.md`; `GRAVITY_ITEM2_MANGA_NONLOCAL_SHAPE_RESULT.md`; `GRAVITY_ITEM2_AXES_GROUP_GEOMETRY_RESULT.md`; `runs/gravity/roadmap/item-02-shape-anisotropy-v1.json`; `runs/gravity/roadmap/item-02-wise-multipoles-v2.json`; `runs/gravity/roadmap/item-02-stellar-multipoles-v3.json`; `runs/gravity/roadmap/item-02-manga-nonlocal-shape-v4.json`; `runs/gravity/roadmap/item-02-axes-group-geometry-v5.json` | Create an immutable five-attempt Item 2 synthesis. If no projected-shape family survives every population, overlap, robustness, and quality requirement, record a scoped `REJECT` for the tested shape families and advance to Item 3 surface-versus-volume density. Do not open any Item 2 confirmation set or retune a failed representation. |
| **3–72** | **NOT YET EXECUTED under this roadmap ledger** | Existing repository work may supply controls or predecessor evidence but does not count as a roadmap PASS without an item-specific receipt. | Follow the numbered dependencies and real-test rule in the stable roadmap. |

Every row remains subordinate to its immutable JSON receipt. If prose and a receipt differ,
the receipt controls until the discrepancy is reviewed and corrected.
