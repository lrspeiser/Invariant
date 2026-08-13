# Exact differentiability boundary for the ordered mixed-D2 arithmetic DAG

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The differentiation question

For each of 12 candidates and 22 target coordinate tangents, the desired ordered mixed derivative begins from a registered first-derivative arithmetic root. This audit asks whether those exact $D^1$ DAGs can be differentiated using only the registered operator and leaf data.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Replay the dependency closure

The 20 distinct target $D^1$ roots per candidate have a union closure of exactly 13,983 arithmetic-DAG nodes. Those nodes contain 341 distinct component-input labels from the $A$, $B_1$, $B_2$, and six $C$ families. Every target-root closure contains exactly 132 component-input leaves. This is a dependency result: it says which inputs a derivative must know before any derivative DAG can be emitted.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Close the non-input operator calculus

Let $D$ denote differentiation along one registered coordinate tangent. The five non-input operators obey exact rules:

$$D(c)=0,\qquad D(-x)=-D(x),\qquad D\!\left(\sum_i x_i\right)=\sum_iD(x_i),$$

$$D(xy)=D(x)y+xD(y),$$

$$D(x/y)=\frac{D(x)y-xD(y)}{y^2},$$

with the quotient rule restricted to the registered nonzero-denominator domain. Thus constants, sums, negations, products, and quotients are closed under the operator calculus. Only an exact_component_input leaf needs new data.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Count the leaf-jet obligations

For one candidate there are 20 distinct target roots and 132 required input-leaf jets per root, hence

$$20\cdot132=2{,}640$$

deduplicated candidate-bound obligations. Across 12 candidates this gives

$$12\cdot2{,}640=31{,}680.$$

The unreduced coordinate references total 34,848 because repeated target roots share packets. Deduplication removes those repeats; it does not remove any distinct candidate, tangent, root, and input-label obligation.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Why no ordered mixed-D2 root is emitted

Each input leaf has a registered value label and provenance hash, but none has a registered coordinate-derivative root. Therefore all 31,680 leaf-jet obligations remain unbound. The exact tally is zero registered leaf-derivative roots and zero registered ordered mixed-$D^2$ roots out of 264 targets. Emitting a formal skeleton would not create an arithmetic certificate, so the derivative DAG remains blocked.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Missing data are not zero data

An absent leaf jet is an unknown required input, not the equation $D(x)=0$. Defaulting it to zero would select an unregistered completion and could change the chain-rule result. The audit therefore establishes a differentiability-domain boundary, not a vanishing theorem. Registering the candidate-bound coordinate derivatives of the reachable $A/B/C$ leaves would discharge the present blocker and require a fresh derivative audit.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## The scientific boundary

No candidate is rejected: all 12 downstream admissions remain blocked. Missing leaf jets do not prove a physical no-go, any derivative zero, a corrected second-source jet, complete $D^2F$, the high-atom identity, global $H^7$, nonlinear PDE closure, or lifespan. The first blocker is

register_candidate_bound_coordinate_derivatives_for_the_31680_reachable_A_B_C_component_input_leaf_obligations.

Evidence: `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json`

## Claim ledger

- **proved:** The 20 target D1 roots have a 13,983-node union closure with 341 component-input labels and 132 leaves per root.
- **proved:** Exact chain rules reduce the derivative problem to 31,680 deduplicated candidate-bound input-leaf jet obligations.
- **blocked:** Zero of 264 ordered mixed-D2 roots can be emitted while all required component-input leaf derivatives remain unregistered.
- **scope_limit:** Missing input jets imply neither zero derivatives nor a physical no-go, D2F admission, global theorem, or candidate rejection.

## Receipt bindings

- `runs/physics-language/quartic-ordered-mixed-d2-arithmetic-dag-differentiability-gate/campaign.json` — file `2992571c544846efc96142e2e4a74efe280a7bb025efadb1ff945ab9515bafcc`, content `d8afd9f91c090ad1c07e4bb22257baa8c61c095f8d434e02a27082b5591abb6a`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the operator calculus is closed only on the declared nonzero-denominator domain
- unregistered component-input derivatives are unknown and are never defaulted to zero
- physical no-go, D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, and observational claims remain fail-closed

Notebook content seal: `59452def759f9f08ffa9541f399e0f6ca47a572c72d9deba204771d915709c55`
