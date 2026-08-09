# Candidate CAND-2bfa308599a6f36cb0aeffe0

Expression: `constrained_vector_flux_v1: F=+(q)-(sqrt(1+(x*z))-1)`

Status: **rejected**; hard-gate state: **rejected**.

## Completed evidence

- `observational_evidence_policy`: **pass** (stage 0)
- `reduced_symbolic_compilation`: **pass** (stage 2)
- `covariant_variation`: **unresolved** (stage 2)
- `universal_minimal_matter_coupling`: **reject** (stage 2)
- `derivative_order_bound`: **pass** (stage 3)
- `constraint_algebra`: **unresolved** (stage 3)

## Remaining claims

- `covariant_action_complete`
- `covariant_variation`
- `kinetic_rank`
- `hamiltonian_boundedness`
- `constraint_algebra`
- `physical_degree_count`
- `characteristic_cones`
- `covariance_identity`
- `gr_limit`
- `solar_system_controls`
- `audited_measurement_holdout`

## Structural ablations

- Remove `1` -> `q - sqrt(x*z + 1)`
- Remove `q` -> `1 - sqrt(x*z + 1)`
- Remove `-sqrt(x*z + 1)` -> `q + 1`

## Equation-universe prior-art screen

- Status: `screened`
- Classification: `known_project_history_exact`
- Novelty claim allowed: `false`
- An unmatched result means only that this finite corpus has no match.

## Family comparisons

- `CAND-9e3b312e6989353002b88e5b`: rejected / rejected
- `CAND-43e21cb5734f7daf47ae2de6`: rejected / rejected

This dossier explains work priority only. It is not a probability of truth.
