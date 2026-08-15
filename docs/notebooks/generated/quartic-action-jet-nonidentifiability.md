# Exact action-jet nonidentifiability from four registered values

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The finite-data question

For each of 22 fitted connection coordinates, the receipt registers only four values of a function of $g=G_{4,X}$, at

$$g\in\{-1,-\tfrac{1}{2},\tfrac{1}{2},1\},\qquad f_i(g)=\beta_i g.
$$

Can these four values determine the first and second $g$ derivatives of the underlying extension? No derivative functor or polynomial degree bound below four is among the registered premises.

Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## Derive the null polynomial

A polynomial that vanishes at all four registered values is obtained by taking one factor for each point:

$$\begin{aligned}
p(g)&=(g+1)(g+\tfrac{1}{2})(g-\tfrac{1}{2})(g-1)\\
&=(g^2-1)(g^2-\tfrac{1}{4})\\
&=g^4-\tfrac{5}{4}g^2+\tfrac{1}{4}.
\end{aligned}$$

Exact differentiation gives

$$p'(g)=4g^3-\tfrac{5}{2}g,\qquad p''(g)=12g^2-\tfrac{5}{2}.$$


Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## Evaluate values and jets at all four points

| $g$ | $p(g)$ | $p'(g)$ | $p''(g)$ |
|---:|---:|---:|---:|
| $-1$ | $0$ | $-3/2$ | $19/2$ |
| $-1/2$ | $0$ | $3/4$ | $1/2$ |
| $1/2$ | $0$ | $-3/4$ | $1/2$ |
| $1$ | $0$ | $3/2$ | $19/2$ |

Thus $p$ is invisible to every registered value sample, while both $p'$ and $p''$ are nonzero at every sampled point. The table is exact rational arithmetic, not a numerical fit.

Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## The 22-parameter ambiguity family

For coordinate $i\in\{0,\ldots,21\}$ introduce an independent parameter $\lambda_i$ and set

$$F_i(g)=\beta_i g+\lambda_i p(g).$$

At each registered grid point, $F_i(g)=\beta_i g$ for every $\lambda_i$. But

$$F_i'(g)=\beta_i+\lambda_i p'(g),\qquad F_i''(g)=\lambda_i p''(g),$$

and the table shows that both jets vary nontrivially with $\lambda_i$ at all four points. Because the parameters are coordinate-wise independent, their product gives a 22-parameter family. Equivalently, the four values leave all 88 recorded first-jet samples and all 88 second-jet samples unidentified.

Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## What is proved

The registered four-point value map is not injective on first or second action feature jets within the displayed degree-four extension class. Therefore those finite values alone cannot select the affine extension $\beta_i g$ over the alternatives $\beta_i g+\lambda_i p(g)$. This is an exact identifiability obstruction, and it holds independently in all 22 coordinates.

Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## What remains open

This obstruction is **not** a no-go theorem for a covariant action derivation. A registered local variation rule, derivative samples, a justified degree bound, or corrected second-source jet values could select one extension. In the sealed receipt, zero corrected second-source entries and zero cross-slice $D^2F$ entries are admitted; complete ordered $D^2F$, the high-atom identity, global $H^7$, nonlinear PDE closure, and lifespan all remain open. All 12 downstream candidates remain blocked rather than rejected. The first blocker is

`registered_local_covariant_variation_rule_or_corrected_second_source_jet_values_required_to_select_one_extension_from_the_exact_22_parameter_jet_ambiguity_family`.

Evidence: `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json`

## Claim ledger

- **proved:** Four registered values do not identify the first or second G4_X jets in the displayed degree-four extension class.
- **proved:** The product construction supplies 22 independent ambiguity parameters and leaves 88 first-jet and 88 second-jet samples unidentified.
- **blocked:** No covariant variation functor, corrected second-source jet, cross-slice D2F admission, complete D2F tensor, H7 closure, nonlinear PDE closure, or lifespan is established.
- **scope_limit:** The proved finite-data obstruction neither rejects the 12 candidates nor proves that no covariant action derivation exists.

## Receipt bindings

- `runs/physics-language/quartic-fitted-output-connection-action-jet-nonidentifiability-gate/campaign.json` — file `e0b87eb270d73f1fa7acb1ff31e0f234a545cf80c383fac21ffa0abc390a902b`, content `b73d3bb175cf008f080ac900c0aed7f463f341d8efc8ebd4cdc4a8fbc3b6de21`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the obstruction concerns the registered four-point value data and displayed degree-four null direction
- a registered variation rule, derivative evidence, or corrected second-source jet may remove the ambiguity
- complete covariant D2F, high-atom, global H7, nonlinear PDE, lifespan, and observational claims remain fail-closed

Notebook content seal: `a9789b02a772f14fa5db404014dfae5e099a97b2b750f838a853cd90945f7946`
