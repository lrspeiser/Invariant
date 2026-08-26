# Task 2 available-now MathOverflow trial

This is a disclosed operational substitute for the still-unpublished future BrokenArXiv
release. It preserves the substantive Task 2 comparison while avoiding an indefinite calendar
wait. The original prospective BrokenArXiv trial remains authorized as a stronger replication.

## Real target and blindness boundary

The source is MathOverflow, an independently operated public research-mathematics forum. The
eligible pool contains questions tagged `counterexamples`, created on or after 2026-07-01 and no
later than the authorization instant, with an accepted answer and at least one selected domain
tag. The target is the SHA-256-minimum question ID. Manual substitution is forbidden.

Before authorization, the operator queried metadata and saw only IDs, timestamps, answer counts,
answer status, and tags for 100 rows. The first API probe technically materialized its default
`title` field in transient process memory, but no title was printed, persisted, sent to the
candidate generator, or viewed by the operator. No question body or answer body was fetched. This
history is sealed into the config. Because the target is public rather than privately held, the
trial cannot prove model-training exclusion or historical novelty.

The metadata-only API filter omits titles and bodies. After authorization, staging may fetch only
the selected question. The accepted answer remains closed until all 36 submissions are frozen.
Version 1 failed closed because its custom filters omitted the API wrapper's `items` field; it read
zero titles, question bodies, and answer bodies. Version 2 records that receipt and corrects all
three filters before reauthorization.

## Creative comparison

The frozen BrokenArXiv generator supplies the same three matched 12-candidate arms:

- historical failure-first LLM;
- creativity-first LLM;
- matched-random falsifier lenses.

All arms receive the identical statement, model, token ceilings, wall-clock bounds, and verifier
budget. Candidate origin labels remain provenance metadata and never establish novelty.

## Gate

Every blinded submission is scored before arm identities are opened. A Task 2 pass requires a
correct counterexample/rejection plus a non-vacuous repair or direct answer, and requires the
creativity-first arm either to reach a decisive valid result earlier than both controls or to
produce a distinct valid structure missed by both. The accepted MathOverflow answer is independent
reference evidence, not automatic validation of a superficially similar candidate.

Any result is reported as an available-now real-problem comparison. The future BrokenArXiv trial
is still required as a prospective replication.

## Result

The frozen trial ran 36 Claude Opus 4.6 calls: 12 failure-first, 12 creativity-first, and 12
matched-random falsifier proposals. It used 54,415 input tokens and 52,624 output tokens, with an
estimated API cost of $1.5877 at the recorded pricing. Independent scoring occurred before the arm
map was opened.

No arm supplied a valid counterexample. Five promising generator descriptions were closed under
union and checked with exact integer arithmetic; all five had an element whose residual degree was
at most half the family's maximum degree. The accepted-answer reference was separately verified as
a 30-set counterexample with degree 19 and residual degree 10 for every element. The gate decision
is therefore `REJECT / PERFORMANCE_OR_CORRECTNESS_GATE_FAILED`. This opened target is now training
material and cannot be used for a later blind pass.

The post-trial failure-space compiler converts the result into a sealed exclusion ledger. It stores
36 exact-response exclusions, three proved parametric construction-family exclusions, and one
non-pruning heuristic warning. See `docs/COUNTEREXAMPLE_DRIVEN_FORMULA_SEARCH.md` and
`runs/math/mathoverflow-task2/exclusion-ledger-v1.json`. The next Task 2 pass attempt must use a new
untouched target selected only after the improved system is frozen.
