# Sigma Campaign Engine report

Campaign: `CMP-55b43fa111e201988d9f2922` — state **active**.

This report identifies work-priority candidates only within the declared grammar and completed gates. It does not claim a true or uniquely best gravity theory.

## Durable accounting

- Tasks: `{'deferred': 8, 'queued': 2, 'succeeded': 80}`
- Candidates: `{'active': 7, 'deferred': 7, 'rejected': 6}`
- Hard-gate evidence: `{'pass': 60, 'reject': 6, 'unresolved': 7}`
- Database integrity: `ok`

## Leading work candidates

- `CAND-05b40623bc6c8a61861f7abc` — `+(z)+(q)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-5307aea98ee49298008e5723` — `-(z)+(q)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-7a073bc5088238a235e3085c` — `+(q)+(sqrt(1+(x*z))-1)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-803f0f8140df5d40ababdfac` — `+(q)+(q**2)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-9de08857a7bf87274e618fee` — `+(q)-(x)+(x*z)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-9e3b312e6989353002b88e5b` — `+(q)-(sqrt(1+(x*z))-1)`; front `1`, hard passes `4`, remaining claims `11`.
- `CAND-2bfa308599a6f36cb0aeffe0` — `constrained_vector_flux_v1: F=+(q)-(sqrt(1+(x*z))-1)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-32e678897fd4e38e870605e8` — `constrained_vector_flux_v1: F=+(z)+(q)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-72ecdf8e80cc8320cbf39b02` — `constrained_vector_flux_v1: F=+(q)+(q**2)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-af6d32ab75393fe7e5e5e5af` — `constrained_vector_flux_v1: F=+(q)+(sqrt(1+(x*z))-1)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-b5a780fb757cec8b85e36411` — `constrained_vector_flux_v1: F=-(z)+(q)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-e93ec24ee8154d159f816457` — `constrained_vector_flux_v1: F=+(q)-(x)+(x*z)`; front `1`, hard passes `3`, remaining claims `11`.
- `CAND-dc1b2f46833f5a239a719fcc` — `S = integral sqrt(-g) [M_Pl^2 R/2 + K(X) + lambda(A_mu A^mu + 1)] + S_m[g,psi]`; front `None`, hard passes `3`, remaining claims `11`.

## Failure clusters

- `higher_derivative_degeneracy_declaration` / `gradient_state`: 6
- `higher_derivative_degeneracy_declaration` / `measured_state`: 5
- `higher_derivative_degeneracy_declaration` / `flux`: 3
- `higher_derivative_degeneracy_declaration` / `saturation`: 2

## Interpretation

A candidate is promoted only by completing hard gates. Historical results and LLM proposals can schedule work but cannot rescue a rejection.
