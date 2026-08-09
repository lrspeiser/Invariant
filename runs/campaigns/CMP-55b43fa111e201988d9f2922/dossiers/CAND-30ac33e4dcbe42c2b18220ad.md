# Candidate CAND-30ac33e4dcbe42c2b18220ad

Expression: `scalar_second_derivative_v1: F=+(q)+(sqrt(1+(x*z))-1)`

Status: **rejected**; hard-gate state: **rejected**.

## Completed evidence

- `observational_evidence_policy`: **pass** (stage 0)
- `reduced_symbolic_compilation`: **pass** (stage 2)
- `covariant_variation`: **unresolved** (stage 2)
- `universal_minimal_matter_coupling`: **reject** (stage 2)
- `higher_derivative_degeneracy_declaration`: **reject** (stage 3)

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

- Remove `-1` -> `q + sqrt(x*z + 1)`
- Remove `q` -> `sqrt(x*z + 1) - 1`
- Remove `sqrt(x*z + 1)` -> `q - 1`

## Equation-universe prior-art screen

- Status: `screened`
- Classification: `known_project_history_exact`
- Novelty claim allowed: `false`
- An unmatched result means only that this finite corpus has no match.

## Family comparisons

- `CAND-7a073bc5088238a235e3085c`: rejected / rejected
- `CAND-af6d32ab75393fe7e5e5e5af`: rejected / rejected

This dossier explains work priority only. It is not a probability of truth.
