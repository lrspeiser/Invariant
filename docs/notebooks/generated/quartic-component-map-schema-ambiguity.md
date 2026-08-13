# Constructive nonidentifiability of a quartic component-map schema

> **Verdict:** `proved`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## The cross-registry question

The registered generic $G_4$ variation has 24 exact abstract term coefficients, while the fitted output connection has 22 coordinates. The source inventory also contains the target $D^1$ row-atom values. The question is whether those registered values uniquely determine a component projection and the corresponding mixed second jets.

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## Set up the 22-by-24 projection problem

Let $c\in\mathbb K^{24}$ be the generic coefficient vector over $\mathbb K=\mathbb Q(\sqrt2)$, let $M\in\mathbb K^{22\times24}$ be a proposed cross-registry map, and let $\beta\in\mathbb K^{22}$ be the fitted value vector. Registered value agreement imposes

$$Mc=\beta.$$

There are $22\cdot24=528$ entries of $M$. Because $c_0=1\ne0$, each output row contributes one independent scalar equation, and different rows use disjoint unknowns. Therefore the constraint rank is exactly 22 and

$$\dim\{M:Mc=\beta\}=528-22=506.$$

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## Construct two maps with identical registered values

A base witness puts $\beta_j$ in column zero of row $j$ and zeros elsewhere. Since $c_0=1$, it sends $c$ to $\beta$. The alternate witness changes only row zero: it sets $M_{00}=1/2$ and $M_{01}=-1$. Since $c_1=-1/2$,

$$M_{00}c_0+M_{01}c_1=\tfrac12(1)+(-1)(-\tfrac12)=1=\beta_0.$$

Every other row is unchanged. Thus two distinct exact $22\times24$ maps have zero residual against the same registered value vector. This is a constructive failure of schema identification, not an approximate fit.

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## Construct the mixed-second-jet ambiguity

For each typed target coordinate introduce an independent exact parameter $\mu_i$ and retain its registered first derivative:

$$D^1F_i=\beta_i,\qquad D^2_{\mathrm{mixed}}F_i=\mu_i,\qquad i=0,\ldots,21.$$

The receipt registers every target $D^1$ membership but no direction-to-state tangent embedding and no ordered mixed-$D^2F$ root. Hence changing any $\mu_i$ preserves all registered values. The zero vector and the 22 unit vectors $e_0,\ldots,e_{21}$ give 23 explicit, pairwise distinct completions. More generally, the mixed-jet ambiguity has 22 independent parameters.

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## What identical values do not determine

The first construction holds $Mc=\beta$ fixed while changing the projection schema through a 506-dimensional affine family. The second holds all 22 registered $D^1$ values fixed while changing 22 mixed-$D^2$ entries. Together they show constructively that equality of the registered values alone does not select the map or its action jets.

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## The scientific boundary

These witnesses are schema completions, not certified covariant physical maps. Tensor equivariance, the typed generic-term-to-source-component projection, the $P10/Pother$ state-tangent embedding, and the 22 ordered mixed-$D^2F$ roots remain unregistered. Therefore schema nonidentifiability is **not** a physical no-go, a candidate rejection, or an admission of corrected source jets, complete $D^2F$, the high-atom identity, global $H^7$, nonlinear PDE closure, or lifespan. All 12 candidates remain blocked. The first blocker is

register_the_typed_generic_term_to_source_component_projection_P10_Pother_state_tangent_embedding_and_22_ordered_mixed_D2F_roots.

Evidence: `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json`

## Claim ledger

- **proved:** The exact 22-by-24 registered value system has rank 22 and a 506-dimensional affine family of projection completions.
- **proved:** Twenty-two independent mixed-D2 parameters admit at least 23 explicit completions preserving every registered target D1 value.
- **blocked:** The typed cross-registry projection, state-tangent embedding, and 22 ordered mixed-D2F roots remain unregistered.
- **scope_limit:** Schema nonidentifiability is not a physical component-map no-go, D2F admission, global theorem, or candidate rejection.

## Receipt bindings

- `runs/physics-language/quartic-fitted-output-connection-component-map-schema-ambiguity-gate/campaign.json` — file `0256f64acb53f38c0cada5e43a58c974b7f9bebe2529bdf7c3f08e65b9d2563f`, content `3a3da9ecef30e596ae18cb8e76687338a9fe1bf8e7284ee009287420ce5613ec`

## Limits

- the notebook is a derived presentation of one sealed receipt, not an independent proof kernel
- the two projection witnesses satisfy registered values but are not certified covariant maps
- the 23 mixed-D2 witnesses are schema completions, not admitted corrected second-source jets
- physical no-go, D2F, high-atom, global H7, nonlinear PDE, lifespan, rejection, and observational claims remain fail-closed

Notebook content seal: `38cdec046a600d9e3790dc3c6a666d7fb94be80218b0e2c4334e9974d794453f`
