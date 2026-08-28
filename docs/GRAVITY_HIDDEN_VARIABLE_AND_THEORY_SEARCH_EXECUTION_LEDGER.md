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
| **2. Shape and anisotropy** | **INCONCLUSIVE, FIRST ATTEMPT** — projected axis ratio and aperture concentration improve on a global constant but lose to the binary disk/cluster proxy, have negative within-population coefficient `R^2`, and fail in the shared axis-ratio interval. The attempt lacks intrinsic galaxy thickness/two-dimensional anisotropy and an intermediate/filamentary population. | `GRAVITY_ITEM2_SHAPE_ANISOTROPY_RESULT.md`; `runs/gravity/roadmap/item-02-shape-anisotropy-v1.json` | Stay on Item 2: add target-blind intrinsic or two-dimensional galaxy morphology plus an intermediate/filamentary geometry population, and require prediction within every population before independent confirmation. |
| **3–72** | **NOT YET EXECUTED under this roadmap ledger** | Existing repository work may supply controls or predecessor evidence but does not count as a roadmap PASS without an item-specific receipt. | Follow the numbered dependencies and real-test rule in the stable roadmap. |

Every row remains subordinate to its immutable JSON receipt. If prose and a receipt differ,
the receipt controls until the discrepancy is reviewed and corrected.
