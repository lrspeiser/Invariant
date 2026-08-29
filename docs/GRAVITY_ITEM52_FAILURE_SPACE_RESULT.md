# Gravity roadmap Item 52: failure-space database result

## Outcome

**PASS: QUERYABLE FAILURE SPACE OPERATIONAL, WITH NO GLOBAL EMPIRICAL PRUNING.**

Invariant now stores two deliberately different kinds of negative knowledge:

1. a formula region that cannot pass an explicitly named frozen mathematical or physical
   gate; and
2. a region in which no actually tested member beat a named empirical threshold on named
   retrospective data.

The second statement is evidence, not a declaration that the whole family is false. Every
empirically weak region retains its best formula, all object-level counterexamples remain
available, and neither one mismatch nor a mismatch count is terminal.

## Real-data test

The database replayed the complete Item 51 stream:

- 67,108,864 raw scheduled ordinals;
- 5,505,024 physically admitted formulas;
- 3,082,813,440 useful candidate-point-selection evaluations;
- 28 S4TM exploration galaxy lenses;
- 20 CLASH clusters with 84 cluster-radius rows; and
- the frozen Item 45 full-data training loss of `0.6744646154` as the empirical threshold.

The replay used the local RTX 5090 and took 12.27 seconds while collecting region minima,
counts, and representative ordinals.

## Formal failure layer

The Item 49 response envelope has 4,096 outer-parameter cells. Each cell was checked over
256 acceleration probes and five possible program-coordinate probes for:

- finite output;
- multiplier at least 0.05 and at most 100;
- the frozen high-acceleration limit; and
- a nontrivial low-acceleration effect.

Results:

| Formal status | Outer cells | Addressable ordinals in full grammar |
|---|---:|---:|
| Pass frozen uniform probe gate | 336 | 532,886,323,200 |
| Cannot pass frozen uniform probe gate | 3,760 | 5,963,251,712,000 |

This is an exact characterization of the declared envelope gate. It is not a proof that
5.963 trillion conceivable laws of nature are impossible: a different envelope, probe
domain, or theory grammar is outside this certificate.

## Empirical failure layer

The 5,505,024 tested formulas were indexed in five overlapping ways:

- eight binary-operator regions;
- 16 ordered source-item pairs;
- 128 operator-by-source-pair regions;
- 512 operator-by-transform-pair regions; and
- 336 admitted outer-parameter cells.

That creates exactly 1,000 queryable empirical region records. In this run, none contained
a scheduled formula whose full-data training loss beat the frozen Item 45 threshold. The
correct stored statement is therefore:

> No scheduled member of this region passed the named threshold on the named retrospective
> data.

It is not stored as “this family is false.” The regions overlap, unsampled members remain,
the data are imperfect, and no independent confirmation was opened.

For every one of the 1,000 regions the database stores the number tested, its best loss,
the exact best representative ordinal, the margin to the threshold, data-quality and
replication status, and explicit `global_family_pruned=false`.

## Example query

The strongest region representative was ordinal `2510928084750`, with full-data training
loss `1.0267539097`. The same formula appears as the best representative of several
overlapping descriptions:

- weighted-difference operator;
- Item 46 primitive on the left and Item 47 primitive on the right;
- identity left transform and sinusoidal right transform; and
- amplitude 6, exponent 0.2, transition 1000.

That repeated appearance tells us where the best part of the failed search lives without
pretending that the region should be erased. A future generator can exploit, repair, or
deliberately contrast that niche.

Example CLI query:

```powershell
python -m sigma_theory_compiler.gravity_item52_failure_space query `
  --region-type binary_operator_x_ordered_source_item_pair --limit 10
```

## Why this matters

The engine can now learn from failure in a disciplined way. It can avoid blindly repeating
the same tested cells, identify which constraints caused formal exclusion, find the best
surviving representative of an empirically weak region, and distinguish “not successful
here” from “mathematically incompatible with this gate.”

This is the architectural foundation for counterexample-guided search. It does not yet
characterize every equation that cannot be right, establish a new gravity law, eliminate
dark matter, or provide independent confirmation.

## Reproduction

Primary receipts:

- `runs/gravity/roadmap/item-52-failure-space-v1.json`
- `runs/gravity/roadmap/item-52-failure-space-v1-source/failure-space-database.json`
- `runs/gravity/roadmap/item-52-failure-space-v1-source/query-test-result.json`

Fast receipt and query replay:

```powershell
python -m sigma_theory_compiler.gravity_item52_failure_space replay
```

Full GPU reconstruction of all substantive database fields:

```powershell
python -m sigma_theory_compiler.gravity_item52_failure_space replay --full-gpu-rebuild
```

The next task is Item 53: use this map to preserve diverse mechanism niches and to avoid
letting the current best score collapse the search onto one family.
