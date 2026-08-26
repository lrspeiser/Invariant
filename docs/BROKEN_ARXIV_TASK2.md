# Task 2: fresh BrokenArXiv creative falsification

Task 2 is a prospective test, not a replay. The implementation and selector are committed before
an eligible problem exists. The source check reads only the official Hugging Face catalog metadata;
it does not retrieve statements, reference answers, model outputs, or judging material.

The frozen cutoff is BrokenArXiv June 2026. The first raw `MathArena/brokenarxiv-MMYY` release at
or after July 2026 is mandatory, and its catalog `lastModified` time must be later than the complete
harness authorization. Once that release appears, the system hashes every stable problem ID with
the frozen seed, dataset ID, and revision, and stages the minimum-ranked problem. There is no manual
replacement path.

The `fetch-release` command verifies the selected repository SHA, locates its canonical train
Parquet shards, and uses Parquet projection pushdown to materialize only `problem_idx` and
`problem`. In particular, `original_problem`—which can reveal the intended repair—is forbidden from
the materialized table. The revision-pinned file paths, projected columns, row counts, and absence
of forbidden columns are sealed before deterministic selection. Hand-authored release packets are
not an admitted live path.

## Real problem and gate

The real problem will be one plausible-but-false research-mathematics statement from that future
release. Three arms see the same statement and receive the same Claude model, 12 calls, 12 candidate
slots, per-call token cap, wall-clock allowance, and three verifier invocations per candidate:

1. `old_failure_first_llm` replays the frozen historical direct-falsification policy from the
   declared baseline commit;
2. `creativity_first_llm` cycles across proposal, analogy, proof-strategy, representation-invention,
   and recombination roles while preserving uncertain lineages; and
3. `matched_random_falsifier` assigns each slot one of 12 generic falsifier families in a
   deterministic hash-randomized order.

The 36 submissions are HMAC-blinded and sorted by blinded ID. The public evaluation packet contains
no arm identity. The key, arm map, exact role/lens schedule, and call evidence remain in an ignored
private coordinator file until every submission is scored.

Task 2 passes only if the system:

1. says the statement is false as written;
2. supplies an exact counterexample or earns an independent external rejection;
3. identifies the smallest failed assumption;
4. gives a nonvacuous repaired statement;
5. proves the bounded repair or earns independent external acceptance; and
6. either beats both the old and matched-random controls on search cost or supplies a distinct valid
   repair missed by both controls.

Every LLM proposal must self-label its likely lineage as a known rewrite, cross-domain synthesis,
proposed new construction, or uncertain. Those labels preserve ideas; they do not prove novelty.
Non-overlap with a finite reference answer likewise does not establish historical novelty.

## Chronology

1. Commit implementation and configuration.
2. Create and commit the authorization receipt with zero statement/reference reads.
3. Poll official metadata only until the first eligible release exists.
4. Fetch a revision-pinned, column-projected release packet with no reference columns.
5. Stage exactly one deterministically selected statement without reference answers.
6. Run and freeze all three matched arms; the credential is activated transiently and never stored
   in a receipt.
7. Give the blinded submissions to an independent exact verifier, formal kernel, official MathArena
   judge, or named independent reviewers. Human-only acceptance requires two named reviewers.
8. After all 36 scores are sealed, open the private arm map and compute the frozen comparison gate.

The independent evaluation must score every submission and bind the canonical counterexample and
repair-proof graph when it marks a candidate valid. A correct treatment candidate without a lower
search cost or distinct repair is a Task-2 rejection, not a creativity success.

The initial selector-only authorization at commit `16fb1eb` and the subsequent full-trial
authorization at commit `3a0a6b2` are preserved as superseded preflights. Both opened zero problem
rows and zero reference answers. The first lacked the complete runner; the second lacked a
revision-pinned official-dataset projection command. Version 3 supersedes both before any eligible
release exists.

Until step 8 passes, Task 3 remains locked.
