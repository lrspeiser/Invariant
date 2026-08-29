# Gravity roadmap Item 50: LLM creativity result

## Outcome

**OPERATIONAL PASS; COMPARATIVE SCIENTIFIC VALUE NOT DEMONSTRATED.**

The language-model ensemble successfully proposed and independently criticized a broad,
response-blind set of equation structures. The complete candidate set was retained and
tested. It did not beat either a size-matched seeded-random control or the strongest
earlier formula from Item 45. Item 50 therefore demonstrates a working creativity lane,
not a better empirical gravity law.

This result does not prune the 48 proposed mechanisms or any broader equation family.
The global counterexample policy applies: one mismatch is never a terminal veto, counts
alone are never a terminal veto, and the present retrospective data are not independent
confirmation.

## Question tested

Given the same frozen primitive catalog, operators, transforms, and outer parameter grid,
can an ensemble of frontier language models choose structurally diverse combinations that
predict the galaxy-lens and cluster response better than:

1. an equally sized seeded-random set of structures; and
2. the strongest previously retained Item 45 universal-interaction formula?

Lower balanced loss is better. The score gives equal population weight to 28 S4TM galaxy
lenses and 20 CLASH clusters rather than allowing the larger set to dominate.

## Frozen creativity protocol

- Models: `claude-opus-5`, `claude-sonnet-5`, and `claude-opus-4-8`.
- Generation: six successful calls, eight proposal slots per call, 48 retained slots.
- Criticism: three successful cross-model calls, 48 advisory assessments.
- Successful paid calls: nine; total provider attempts: eleven.
- Recoverable use: 187,239 input tokens and 45,235 output tokens.
- Recoverable estimated standard cost: **$1.629109**.
- Two early completed attempts failed before durable response capture. Their content,
  token use, and cost are not recoverable and are disclosed rather than invented.
- The API key was loaded from the local environment. No credential material was written
  into a receipt.

The prompts contained primitive definitions and formal search constraints, but no target
response values, object losses, prior winning ordinal, or sealed confirmation rows.
Critics could classify and repair proposals, but their advice could not delete a proposal.

## Creativity and lineage accounting

The providers labeled the 48 proposal slots as:

| Self-label | Count |
|---|---:|
| Potentially new synthesis | 19 |
| Known-family combination | 18 |
| Uncertain | 9 |
| Algebraic rewrite | 1 |
| Known formula | 1 |

The independent critics reclassified them as 14 potentially new syntheses, 22 known-family
combinations, eight uncertain cases, and four algebraic rewrites. Dimensional status was
16 consistent, 21 repairable, and 11 uncertain. Three critic `retain=false` opinions were
archived as advice and caused no pruning. One malformed proposal slot was preserved as a
non-executable placeholder rather than silently replaced; one malformed critique became an
uncertain, non-vetoing assessment.

The executable LLM lane contained 47 distinct structures, 58 distinct primitives, all
eight binary operators, ten source-item pairings, and 27 cross-item structures. Its matched
random lane contained 48 structures, 89 distinct primitives, all eight operators, ten
source-item pairings, and 28 cross-item structures. No behavioral duplicates were found in
either expanded lane.

## Search and result

Each structure expanded into 336 physically admitted outer-parameter cells:

- LLM: 47 structures and **15,792** outcome-scoring candidates.
- Matched random: 48 structures and **16,128** outcome-scoring candidates.
- Total: **31,920** candidates and **17,875,200 candidate-point-fold evaluations**.
- Compute backend: NVIDIA GeForce RTX 5090 through CuPy/CUDA.
- Selected CPU/GPU score differences were at floating-point roundoff scale.

| Method | Balanced loss | S4TM loss | CLASH loss |
|---|---:|---:|---:|
| Item 45 universal interaction | **0.76148** | **0.18782** | **1.33514** |
| Item 47 operator generator | 0.77891 | 0.16351 | 1.39431 |
| Item 46 dimensionless generator | 0.87009 | 0.24753 | 1.49265 |
| Item 49 pseudorandom program | 1.10118 | 0.17467 | 2.02769 |
| Matched seeded-random Item 50 | 1.59140 | 0.24574 | 2.93706 |
| Ordinary ridge control | 1.87782 | 0.26047 | 3.49517 |
| **LLM ensemble Item 50** | **2.31579** | **0.23825** | **4.39334** |
| Baryonic Newton control | 67.65046 | 0.91086 | 134.39006 |

The best LLM-selected formula was 45.52% worse than its size-matched random control and
204.12% worse than Item 45. The paired sign-flip test gave `p = 0.00015`, but the significant
direction favored Item 45, not the LLM. Leave-one-object-out, trimmed, and all four frozen
baryonic-mass shift checks also failed to reverse that conclusion.

Thirty-four objects were raw counterexamples to the LLM winner relative to Item 45; 15
remained so under the relevant mass-shift check. This is recorded as
`QUALITY_LIMITED_EVIDENCE_RETAINED`, not as a globally pruned formula family.

## What the selected formula means

The winning LLM structure came from a Sonnet 5 proposal labeled `uncertain` by its author
and `potentially_new_synthesis` by an independent critic. It compared:

- a local measure of how quickly enclosed baryonic mass changes with radius; and
- a nonlocal measure of baryonic mass outside the current radius.

It used the larger of the two transformed effects as the active response. In lay terms,
the suggestion was that local structure controls gravity in one regime and exterior
baryons control it in another. The critic correctly flagged the hard switch as a likely
unphysical kink. Empirically, the candidate fit some galaxy lenses but transferred poorly
to clusters.

This is not a claimed new law, a derivation from an action, an alternative to general
relativity, or evidence that dark matter has been eliminated.

## Interpretation

The important positive result is procedural: Invariant can solicit labeled mechanisms,
preserve uncertain and criticized ideas, compile valid suggestions into deterministic
formula programs, compare them against a blind random control, and replay the full
experiment without making another paid call.

The important negative result is scientific: this ensemble and prompt did not concentrate
probability on better-performing structures. Random selection covered more primitives and
predicted the retrospective data better. A future LLM campaign should change the mechanism
generation task materially—for example by asking for new primitives, smooth derived
operators, or action-level constructions—and preregister it, rather than merely resampling
the same prompt.

## Reproduction

The result is recorded in:

- `runs/gravity/roadmap/item-50-llm-creativity-v1.json`
- `runs/gravity/roadmap/item-50-llm-creativity-v1-source/joint-evaluation-result.json`

Replay without paid provider calls:

```powershell
python -m sigma_theory_compiler.gravity_item50_llm_creativity replay
```

The next roadmap task is Item 51: measure RTX 5090 screening throughput and use that
measurement to design honest large-batch searches without claiming unsupported trillions.
