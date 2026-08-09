# Sigma formal-backend controls

- Passed: 5 / 6
- Cadabra 2 available: True

| Control | Status | Verified scope |
|---|---:|---|
| `covariant_field_contract` | pass | Exact schema and policy validation. |
| `einstein_hilbert_linearized_bianchi` | fail | Exact Fourier-space identity around Minkowski; not nonlinear arbitrary-background variation. |
| `canonical_scalar` | pass | Exact quadratic control around Minkowski. |
| `proca_adm_dirac` | pass | Exact flat-background quadratic Hamiltonian control. |
| `einstein_aether_modes` | pass | Known linearized Minkowski formulas; not a derivation of the full nonlinear constraint algebra. |
| `dhost_degenerate_kinetic_block` | pass | Exact reduced ADM scalar kinetic block; not a full covariant DHOST classification. |

## Candidate readiness

The known-answer harness is operational, but arbitrary candidate variation, nonlinear Noether identities, full ADM/Dirac closure, and background-dependent principal symbols remain fail-closed. Observational gates stay sealed.
