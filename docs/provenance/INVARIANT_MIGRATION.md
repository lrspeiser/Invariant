# Invariant repository migration

Invariant was extracted from `lrspeiser/sigmagravity` on 2026-08-12. The source project lived at
`research/sigma-theory-compiler` on branch `agent/publish-gravity-theory-compiler`.

## Provenance anchors

- Source repository: `https://github.com/lrspeiser/sigmagravity`
- Source anchor commit: `558fee73a25c060f8ae034309f81b63743954012`
- Source anchor tag: `invariant-monorepo-anchor-20260812`
- Filtered import commit: `1cfccb1023f3b2eca4361a2a8ed8e55bbbd78994`
- Filtered import tag: `invariant-history-import-20260812`
- Root-layout migration tag: `invariant-root-layout-20260812`
- History import: 162 commits including the root workflow history
- Old-to-new map: [`monorepo-commit-map.tsv`](monorepo-commit-map.tsv)
- Commit-map SHA-256: `8fbd5f25f35a218290444e556db450fd4a07208bcf295e50db487fa828b06503`

The extraction retained the compiler subtree and its GitHub Actions workflow, moved the project to
the repository root, and rewrote Git commit IDs as required by the path change. File-level
provenance hashes inside the project were not rewritten.

## Data boundary

The committed evidence and historical database blobs are preserved. The working copy of
`runs/campaigns/campaign-v1-live.sqlite` had uncommitted runtime changes at migration time; those
mutable bytes were deliberately not imported. Ignored service queues, leases, checkpoints, stop
requests, caches, local toolchains, and the mixed-third-jet supervisor state were also excluded.
They describe machine/process ownership, not portable scientific evidence.

The repository contains 294 Git LFS paths (293 `.bin` files and one `.tar`). A complete clone must
run `git lfs pull`; the migration is complete only when `git lfs fsck` succeeds.

Two reviewed, isolated post-anchor lanes were copied without claiming unified admission:

- `quartic_fitted_output_connection_action_feature_factorization_gate`
- `continuous_scientific_pipeline_epoch_003_formal_receipt_batch_0002`

Their own sealed artifacts and focused tests define their exact boundaries. They remain separate
from the unified projection until reviewed integration is completed in Invariant.

## Historical host paths

Some immutable receipts record the workstation and WSL paths on which they were produced. They are
preserved byte-for-byte because other artifacts bind their hashes. Portable projections are used
for current validation; historical receipts must be superseded through a new provenance-linked
artifact rather than edited in place.

## Standalone distribution contract

Top-level configs, runs, formal assets, and scripts remain intentionally outside the Python wheel.
They are now distributed with the wheel in the versioned standalone source-release ZIP described
in [`docs/STANDALONE_DISTRIBUTION.md`](../STANDALONE_DISTRIBUTION.md). Its complete manifest binds
every tracked resource and all 294 hydrated Git LFS objects. Consumers install the bundled wheel
normally and use the adjacent extracted tree as the resource root; neither an editable checkout
nor Git metadata is required after extraction.
