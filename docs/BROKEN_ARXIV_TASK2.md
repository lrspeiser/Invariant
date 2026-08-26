# Task 2: fresh BrokenArXiv creative falsification

Task 2 is a prospective test, not a replay. The implementation and selector are committed before
an eligible problem exists. The source check reads only the official Hugging Face catalog metadata;
it does not retrieve statements, reference answers, model outputs, or judging material.

The frozen cutoff is BrokenArXiv June 2026. The first raw `MathArena/brokenarxiv-MMYY` release at
or after July 2026 is mandatory. Once that release appears, the system hashes every stable problem
ID with the frozen seed, dataset ID, and revision, and stages the minimum-ranked problem. There is
no manual replacement path.

## Real problem and gate

The real problem will be one plausible-but-false research-mathematics statement from that future
release. Both arms see the same statement. The creativity-first arm receives 12 candidate slots;
the matched-random arm receives 12 predeclared falsifier families and the same statement access and
verification allowance.

Task 2 passes only if the system:

1. says the statement is false as written;
2. supplies an exact counterexample or earns an independent external rejection;
3. identifies the smallest failed assumption;
4. gives a nonvacuous repaired statement;
5. proves the bounded repair or earns independent external acceptance; and
6. either beats matched random search on verification cost or supplies a distinct valid repair.

Every LLM proposal must self-label its likely lineage as a known rewrite, cross-domain synthesis,
proposed new construction, or uncertain. Those labels preserve ideas; they do not prove novelty.
Non-overlap with a finite reference answer likewise does not establish historical novelty.

## Chronology

1. Commit implementation and configuration.
2. Create and commit the authorization receipt with zero statement/reference reads.
3. Poll official metadata only until the first eligible release exists.
4. Stage exactly one deterministically selected statement without reference answers.
5. Run and freeze both arms.
6. Open judging/reference material, verify, compare, and publish the receipt.

Until step 6 passes, Task 3 remains locked.
