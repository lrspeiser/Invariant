# Typed full-D2F partition and maximal same-direction row extension

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The extension question

The previous theorem sealed 22 bounded row-10 entries per candidate. This gate builds an exact typed partition of the entire ordered $11\times153\times153$ domain, then admits the maximal same-direction extension available from the already registered leaf derivatives.

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## Partition the full ordered domain

| Exact typed block | Entries/candidate | Status |
|---|---:|---|
| Prior row-10 diagonal slots | 22 | registered |
| Selected rows 0–9 diagonal slots | 220 | registered here |
| Registered-direction off-diagonal pairs | 5,082 | cross-direction jets missing |
| Registered $D^1$, derivative direction unregistered | 31,702 | blocked |
| Unregistered $D^1$, derivative direction registered | 31,702 | blocked |
| Both directions unregistered | 188,771 | blocked |

The six blocks are disjoint and exhaustive. Their counts sum exactly to

$$22+220+5{,}082+31{,}702+31{,}702+188{,}771=257{,}499=11\cdot153^2.$$

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## Extend the same-direction rows

Each of the 22 registered direction slots has an exact derivative along itself. Applying those same-direction leaf jets to source rows 0 through 9 adds

$$22\text{ directions}\cdot10\text{ rows}=220$$

sealed entries per candidate. Across 12 candidates the gate adds 2,640 exact records.

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## Update the registered inventory

The 220 new records join the 22 previously sealed row-10 diagonal records:

$$22+220=242$$

registered entries per candidate. Hence the exact remaining inventory is

$$257{,}499-242=257{,}257.$$

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## The next bounded block

The next preregistered extension is the 5,082 ordered off-diagonal pairs among already registered directions. Their primal $D^1$ roots and both direction labels are known, but differentiation along the other direction requires cross-direction leaf jets. Registering those jets would make this whole block replayable without opening the larger unregistered-direction blocks.

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## The fail-closed boundary

A complete typed partition is an accounting theorem, not a complete $D^2F$ tensor. Exactly 257,257 entries per candidate remain blocked, beginning with the 5,082 cross-direction entries. Therefore the full high-atom identity, global $H^7$, nonlinear PDE closure, lifespan, physical no-go, and candidate rejection all remain unavailable. The first blocker is

register_cross_direction_leaf_derivatives_for_the_5082_registered_direction_off_diagonal_entries_per_candidate.

Evidence: `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json`

## Claim ledger

- **proved:** Six exact typed blocks partition all 257,499 ordered D2F entries per candidate.
- **proved:** The same-direction rows 0–9 extension adds 220 entries per candidate, reaching 242 registered.
- **blocked:** The next 5,082 cross-direction entries require registered cross-direction leaf derivatives.
- **scope_limit:** Typed partition and row extension do not establish complete D2F, high-atom closure, a global theorem, or rejection.

## Receipt bindings

- `runs/physics-language/quartic-full-d2f-typed-partition-row-extension-gate/campaign.json` — file `9502843234509a4ddd21631acdfe412d0f17fe3552d7c9cac0daf7fb1475190a`, content `76eff324a16396dfbeee91552220b26dd745b3c22aa5dd6fb9538fffa843bece`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the registered extension uses only same-direction leaf jets for source rows 0 through 9
- 257,257 ordered entries per candidate remain outside the admitted inventory
- complete D2F, high-atom, global H7, nonlinear PDE, lifespan, physical no-go, and rejection remain fail-closed

Notebook content seal: `790c62753ad4c6e7c76b613773fdfee29b16de66077612606a3a7f85d78bfffc`
