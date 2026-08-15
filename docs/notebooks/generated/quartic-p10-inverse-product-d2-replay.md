# Exact inverse-product replay closes all P10 ordered-D2 roots

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The replay question

The preceding gate registered all 7,920 candidate-bound P10 leaf derivatives but stopped before differentiating the complete source expression. This gate asks whether those roots close the exact forward replay of the bound inverse/product $D^1$ arithmetic DAG.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## Forward-mode operator calculus

Associate each primal node $x$ with its tangent $\dot x$. Constants and inputs obey $\dot c=0$ and $\dot x=$ the bound leaf root. Sums and negations are linear, while

$$\dot{(xy)}=\dot x\,y+x\,\dot y,$$

$$\dot{(x/y)}=\frac{\dot x\,y-x\,\dot y}{y^2}.$$

The quotient rule is used only on the registered domain where $c_{11}=(-1)^{11}\det(A)$ is nonzero. Exact zero additions, products, and numerators are simplified without dropping any nonzero term.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## Replay the full dependency traces

Across 12 candidates and five P10 packets per candidate, the replay consumes all 7,920 bound input roots. It visits 811,296 nodes in the primal dependency closures and constructs 786,396 exact derived Merkle nodes. The validator recomputes the full per-primal-node trace rather than trusting a summarized root.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## Seal 60 roots and expand to 84 records

There are five unique P10 directions per candidate, hence

$$12\cdot5=60$$

unique replay roots. Repeated coordinate ordinals for $s_{11}[10]$ and $s_{22}[10]$ expand five roots to seven ordered records per candidate, so

$$12\cdot7=84.$$

Every one of the 84 P10 ordered-$D^2$ records is sealed by an exact canonical Merkle replay root: the P10 subset is now complete.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## A genuine subset success

This gate advances from registered leaf data to derivatives of the full bound inverse/product expression. It registers 84 of 84 arbitrary-background P10 ordered $D^2$ targets, with zero P10 blockers. That is a constructive exact success, not merely a list of remaining obligations.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## The remaining Pother and full-D2 boundary

Pother leaf derivatives are still absent, so its 180 ordered targets remain blocked. The total admitted inventory is therefore 84 of 264 targets, not a complete ordered $D^2F$ tensor. No high-atom identity, global $H^7$, nonlinear PDE closure, lifespan, observation, candidate rejection, or physical no-go follows. All 12 downstream candidates remain blocked. The first blocker is

register_candidate_bound_Pother_A_B_C_leaf_derivatives_then_replay_the_remaining_180_ordered_mixed_D2_roots.

Evidence: `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json`

## Claim ledger

- **proved:** Exact forward replay consumes 7,920 bound P10 leaf roots across 811,296 primal visits and 786,396 derived nodes.
- **proved:** All 60 unique replay roots and all 84 ordered P10 D2 records are exactly sealed.
- **blocked:** The remaining 180 Pother ordered-D2 roots require candidate-bound Pother leaf derivatives and replay.
- **scope_limit:** P10 closure is not complete D2F, a global analytic theorem, candidate rejection, or physical no-go.

## Receipt bindings

- `runs/physics-language/quartic-p10-inverse-product-d2-replay-gate/campaign.json` — file `2a9814a27123099b9e942bde72fa45fe8783e3ddde0743d080b17008dbb9318c`, content `e02949cb28f43851483d2b0b6cb06c6710ac53a16f210150449d85ceb0ec92ba`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the quotient replay applies on the declared nonzero-determinant domain
- Pother leaf derivatives and 180 ordered roots remain unregistered
- complete D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, observation, and physical no-go remain fail-closed

Notebook content seal: `1f28d37ce7b78c41702ab1a53bdf8d26b6008cf0f352a6c920283fac8479a620`
