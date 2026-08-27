# G4 LLM creativity campaign result

Date: **2026-08-27**

## Decision

`BLOCK_G4_LLM_CREATIVITY_CAMPAIGN_ENGINEERING_FAILURE`

The sealed receipt is
`runs/gravity/g4/llm-creativity-campaign-failure-v1.json`. This result is an
engineering failure of the proposal channel, not evidence for or against any gravity law.

## What was attempted

Claude Opus 5 received only aggregate exploration failures, allowed variable names and units,
and explicit origin-label rules. It received no observed target rows, residuals, galaxy IDs, or
confirmation data. The contract required each proposal to say whether it appeared to be a known
rewrite, a known-family instance, a combination of known ideas, a proposed new construction, or
uncertain. A label could never prune an idea or establish historical novelty.

Ten provider HTTP requests were made. Six were rejected before inference because Anthropic's
structured-output grammar accepted a narrower JSON-schema subset than the runner initially used.
Four inferences completed:

1. The first response was lost before persistence because its proposal IDs violated the local
   regex.
2. The second was lost because its proposal count violated the exact-count contract.
3. The first retained response used 15,090 output tokens, 14,946 of them thinking tokens, and
   returned twelve empty proposal strings.
4. The final retained response used the 20,000-token cap, 19,781 of them thinking tokens, and
   stopped before writing its first proposal.

The two retained calls have a conservative reconstructed usage ceiling of **$7.3635**. Usage for
the first two completed calls is unavailable, so the receipt reports the predeclared **$20 total
campaign ceiling**, not an invented actual charge.

## Result and repair boundary

There are **zero usable proposals**. No fifth call is authorized. Both retained raw provider
responses are immutable evidence, and the runner now persists raw output before semantic
validation. The campaign therefore contributes an orchestration lesson but no candidate family.
The independent typed nonlocal G4 campaign proceeded from already documented failure geometry,
not from hidden model reasoning or a reconstructed proposal.
