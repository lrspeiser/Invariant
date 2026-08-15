# Local viability analysis of a quartic scalar–tensor candidate

> **Verdict:** `formal_local_survivor`
> **Disclosure:** Historical-style reconstruction generated from sealed machine receipts. It is not an authentic historical document, private model reasoning, or a replacement for the cited receipts.

## Question and candidate

Consider the exact candidate `quartic-symbol-06e267a9215345b6` with

- $G_2=X+(-1)*X^2$,
- $G_3=0$,
- $G_4=1/2+(-1/2)*X$,
- $G_5=0$.

The scientific question is not merely whether this expression is compact, but whether its constrained dynamics, local PDE symbol, and nonlinear energy structure survive exact checks.

Evidence: `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json`

## 1. Exhibit an exact local solution

At the certified FLRW point the lapse, scale, and scalar equation residuals are all exactly zero. The witness uses $X=1/200000000000000000000$ and $A_\star=1/10000000000$. Every recorded jet component lies strictly inside the local hyperbolicity box. This establishes a nonempty on-shell patch, not a global spacetime solution theorem.

Evidence: `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json`, `runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json`

## 2. Count the physical modes

The velocity Hessian in the ordered variables V_star, K11, K22, K33, K12, K13, K23 has rank 6 and nullity 1. Its null direction yields the primary constraint `$p_V_star=0`. The closed Dirac chain records 6 first-class and 2 second-class constraints on an extended phase space of dimension 20:

$$N_{\rm dof}=\frac{20-2(6)-2}{2}=3.
$$

Thus the local constrained system propagates three configuration degrees of freedom.

Evidence: `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json`

## 3. Check local energy and hyperbolicity

The reduced quadratic Hamiltonian has the form

$$H_k=\tfrac12[P^T K^{-1}P+k^2Q^TFQ],
$$

with the recorded kinetic and gradient matrices strictly positive. Independently, the complete 22-by-22 directional symbol is strongly hyperbolic throughout the declared local-jet box, and its symmetrizer lifts to the full 55-state first-order system. For compatible vacuum initial data in a compact subset of that box, the local Cauchy theorem applies for some unspecified $T>0$.

Evidence: `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json`, `runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json`, `runs/physics-language/quartic-nonquasilinear-pde-campaign/campaign.json`

## 4. Locate the unresolved obstruction

The second-derivative source domain contains $153^2=23{,}409$ ordered atom pairs and $11\times153^2=257{,}499$ output entries per candidate. Only 891 entries have admitted corrected values. Therefore 256,608 entries remain unregistered, including 106,920 principal high-atom entries. The first blocker is `candidate_bound_covariant_source_derivatives_and_output_bundle_connection_extension_for_remaining_106920_principal_high_atom_D2F_entries_not_registered`. Consequently the complete high-atom identity, global $H^7$ closure, nonlinear global PDE theorem, and lifespan remain unproved.

Evidence: `runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json`

## Scientific conclusion

This is a **formal local survivor**: it has an exact local on-shell witness, the expected three-mode constraint count, positive reduced quadratic energy, and local strong hyperbolicity. It is not yet an admitted global theory. A mathematician's notebook should preserve both halves of that sentence.

Evidence: `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json`, `runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json`, `runs/physics-language/quartic-nonquasilinear-pde-campaign/campaign.json`, `runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json`

## Claim ledger

- **certified_local:** The selected candidate has a three-mode local constrained Hamiltonian, positive reduced quadratic energy, and a conditional local vacuum Cauchy certificate.
- **blocked:** The complete D2F/high-atom identity, global H7 estimate, nonlinear global PDE closure, and lifespan are not proved.
- **scope_limit:** No observational or universal-matter conclusion follows from these local vacuum certificates.

## Receipt bindings

- `runs/physics-language/quartic-dirac-hamiltonian-campaign/campaign.json` — file `68541766993d0d46f23dd2707c4e5db8bbf00dbdd9c442fc3802c1c2f7d9bb3f`, content `69f6f67237020adab07f741b8de154465fa5d24984d78dfb2541da4567db2a47`
- `runs/physics-language/quartic-symmetrizer-uniform-domain-campaign/campaign.json` — file `ba10d92e31d1d098baf82eceee9b02e6542ee04c36a8eba35a1e9dad2aa76e7d`, content `e9344d537d14ed11a8f4cfb26b954c90985a8f8a86c4cd106b7318647f564d5d`
- `runs/physics-language/quartic-nonquasilinear-pde-campaign/campaign.json` — file `f2bab55c2921557fc69e6daffc08c3095e62930d3280e3f9bf636b3fa5c63ed2`, content `be874df291f2679283a473b9978b95bb88318f3e39a4067994488a49caf6c876`
- `runs/physics-language/quartic-full-d2f-high-atom-coverage-gate/campaign.json` — file `b9ce34960b766a6fe74a36a13190b0f050a1447884d599fad1eebfe189b32590`, content `e7e4e4171aed90d07d68791183c58a696e77b9bed745f1018da2c5ee9438c38a`

## Limits

- the presentation is derived from sealed receipts and is not an independent proof kernel
- the local Cauchy time is existential and has no numerical lower bound
- preservation of the local box under general inhomogeneous evolution is not proved
- matter evolution, boundary estimates, global H7 closure, lifespan, and observations are outside the certified result

Notebook content seal: `a7e8de890b914305526ac6cd0de7e42e19dccd152eec60d5cfd7e3599f4de0ee`
