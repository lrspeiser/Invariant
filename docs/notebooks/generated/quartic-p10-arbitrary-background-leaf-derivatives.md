# Arbitrary-background P10 leaf derivatives before D2 propagation

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The P10 arbitrary-background question

The flat gate computed all target values only at one reference. This gate instead binds the nonlinear geometric map for the five unique scalar-second-partial P10 directions on every nonsingular background and differentiates the live $A/B/C$ input formulas along them.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## Derive the five scalar-Hessian tangents

For a scalar field,

$$H_{ij}=\nabla_i\nabla_j\phi=\partial_i\partial_j\phi-\Gamma^\rho_{ij}\partial_\rho\phi.$$

Varying the scalar second-partial coordinate $s_{ij}[10]$ while holding the background connection and first derivatives fixed gives

$$\frac{\partial H_{ij}}{\partial s_{ij}[10]}=1.$$

The five registered tangents are $H_{11}$, $H_{12}$, $H_{13}$, $H_{22}$, and $H_{23}$, each with unit coefficient. This identity retains the connection term; it does not assume a flat background.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## Differentiate the live A/B/C leaves

Each direction reaches 132 component-input leaves. Exact differentiation of the unspecialized block formulas gives four nonzero roots and 128 zero roots per direction. Thus one candidate has

$$5\cdot132=660,\qquad 5\cdot4=20\text{ nonzero},\qquad 5\cdot128=640\text{ zero}.$$

Across 12 candidates the exact census is 7,920 roots: 240 nonzero and 7,680 zero.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## The eleven-node exact DAG

All registered leaf derivatives take values in

$$\{-2,-\sqrt2,-1,-\tfrac{\sqrt2}{2},-\tfrac12,0,\tfrac12,\tfrac{\sqrt2}{2},1,\sqrt2,2\}.$$

The receipt stores exactly eleven constant nodes, one per value, and all 7,920 candidate-bound leaf roots point into this DAG. No floating-point value is used.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## Constructive progress without a propagated D2 root

The gate has supplied the formerly missing P10 input derivatives, but a leaf root is not yet the derivative of the full source expression. The bound inverse/product $D^1$ DAG must still be differentiated and replayed with these leaves. Consequently zero of 84 P10 ordered-$D^2$ targets is registered. This is constructive partial progress: the input domain is now closed for P10, while composition remains open.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## The scientific boundary

The Pother nonlinear coordinate map and its 23,760 leaf roots remain unregistered, and zero of all 264 ordered-$D^2$ targets is admitted. Therefore the P10 subset does not establish complete $D^2F$, the high-atom identity, global $H^7$, nonlinear PDE closure, lifespan, observation, candidate rejection, or a physical no-go. All 12 downstream candidates remain blocked. The first blocker is

differentiate_and_replay_the_bound_inverse_product_D1_DAG_using_the_7920_registered_P10_leaf_roots_then_register_Pother_leaf_derivatives.

Evidence: `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json`

## Claim ledger

- **proved:** Five background-independent scalar-Hessian coordinate tangents register all 7,920 reachable P10 input-leaf derivative roots.
- **proved:** The exact leaf census is 240 nonzero and 7,680 zero roots represented by an eleven-node constant DAG.
- **blocked:** Zero of 84 P10 ordered-D2 roots is registered until the inverse/product D1 DAG is differentiated and replayed.
- **scope_limit:** P10 input-leaf progress is not complete D2F, a global PDE theorem, candidate rejection, or a physical no-go.

## Receipt bindings

- `runs/physics-language/quartic-p10-arbitrary-background-leaf-derivative-gate/campaign.json` — file `c74171c48d7fc4f80de8f0c51b2b2700a1ce33de8795c3a999cee7c957b35869`, content `51f76fa7ebc81ab2f570bfe5ad920215420e005687d0c861b24ea6da766c37e0`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the arbitrary-background result registers input-leaf derivatives, not propagated ordered-D2 roots
- Pother coordinate tangents and leaf derivatives remain unregistered
- complete D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, observation, and physical no-go remain fail-closed

Notebook content seal: `35da0e27ce737daf4ec47889d76f0c087e89bd48c7cd2bd8fc5dc7cf00c7f1b5`
