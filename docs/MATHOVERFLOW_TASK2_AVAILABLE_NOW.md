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
