# Sigma formal-backend controls

- Passed: 54 / 54
- Cadabra 2 available: True

| Control | Status | Verified scope |
|---|---:|---|
| `covariant_field_contract` | pass | Exact schema and policy validation. |
| `einstein_hilbert_linearized_bianchi` | pass | Exact Fourier-space identity around Minkowski; not nonlinear arbitrary-background variation. |
| `einstein_hilbert_linearized_adm` | pass | Complete nonzero Fourier-mode control around Minkowski; nonlinear ADM remains separate. |
| `nonlinear_adm_hamiltonian_constraint_algebra` | pass | Exact 3D covariant H-H bracket with general q_ij, pi^ij, lapse jets, DeWitt kinetic term, curvature lapse Hessian, and compact-support boundary reduction. |
| `spatial_curvature_density_diffeomorphism_covariance` | pass | Exact 3D D-H curvature-sector identity modulo a compact-support boundary; the lapse-smeared metric derivative is independently derived by Cadabra. |
| `canonical_scalar` | pass | Exact quadratic control around Minkowski. |
| `canonical_scalar_noether_identity` | pass | Exact local tensor identity in Riemann normal coordinates at an arbitrary point. |
| `proca_adm_dirac` | pass | Exact flat-background quadratic Hamiltonian control. |
| `proca_divergence_identity` | pass | Exact local principal-derivative identity and algebraic consequence of the vector equation. |
| `proca_stress_noether_identity` | pass | Exact 4D Minkowski component identity for arbitrary smooth A_mu; curved-background executable reduction remains separate. |
| `proca_curved_background_noether_identity` | pass | Exact FLRW homogeneous-vector and static-spherical radial-vector controls; not every metric/profile. |
| `einstein_aether_modes` | pass | Known linearized Minkowski formulas; not a derivation of the full nonlinear constraint algebra. |
| `einstein_aether_flrw_variation_noether` | pass | Exact lapse-FLRW homogeneous reduction with independent N, a, U, and lambda; not the full inhomogeneous identity. |
| `einstein_aether_adm_kinetic_hessian` | pass | Exact pointwise aligned and rationally tilted unit-aether controls; secondary constraints and Poisson closure remain separate. |
| `einstein_aether_generic_3plus1_legendre` | pass | Exact block decomposition, symbolic aligned determinant, and one inhomogeneous tilted rational nine-velocity Legendre patch; distributed Hamiltonian constraints and H-D/H-H brackets remain separate. |
| `einstein_aether_generic_lapse_shift_constraint_seeds` | pass | Exact lapse-acceleration Legendre cancellation, boundary reduction, and spatial cotangent lift; H-H closure, global rank, nonlinear degree count, and boundedness remain unresolved. |
| `einstein_aether_generic_dh_covariance` | pass | Exact arbitrary-GL(3) tensor-contraction and canonical-density proof of D-H covariance; the normal-deformation H-H bracket, global rank, nonlinear degree count, and boundedness remain unresolved. |
| `einstein_aether_generic_hh_deformation_kinematics` | pass | Exact normal-embedding deformation algebra, Jacobi reduction, inverse-metric structure function, and Aether-specific -chi D_i N Hamilton-flow check; requires the separately executable arbitrary-background Noether control and assumes compact support or a completed boundary generator. Singular coupling strata and Hamiltonian boundedness remain unresolved. |
| `einstein_aether_linearized_physical_energy` | pass | Exact on-shell spin-2/spin-1/spin-0 wave-energy coefficients after linearized gauge and constraint reduction, with two positive-speed negative-energy controls and the restricted hypersurface-orthogonal nonlinear positive-energy domain; generic nonlinear Hamiltonian boundedness remains unresolved. |
| `einstein_aether_restricted_nonlinear_total_energy` | pass | Executable conformal-curvature and boundary-charge reduction to the Schoen-Yau positive-mass theorem with nonnegative matter energy, 0<=c14<=2, and c13<=1. Twisting Aether, nonmaximal data, and out-of-domain couplings remain unresolved rather than rejected; generic nonlinear reduced-Hamiltonian stability is not claimed. |
| `einstein_aether_reduced_five_mode_principal_domain` | pass | Necessary-and-sufficient aligned-Minkowski linearized certificate with exact spin-2/spin-1/spin-0 kinetic and gradient matrices, five negative witnesses, and six singular or strong-coupling strata; arbitrary nonlinear backgrounds, global tilted strata, and observational cone cuts remain outside this proof. |
| `einstein_aether_global_tilt_legendre_strata` | pass | Exact pointwise theorem for every unit-timelike tilt magnitude and orientation by spatial rotational covariance, with a rank-loss witness and a globally subluminal noncharacteristic certificate; arbitrary inhomogeneous-background principal symbols, boundary charges, and nonlinear Hamiltonian boundedness remain outside this proof. |
| `einstein_aether_covariant_arbitrary_background_hyperbolicity` | pass | Executable five-mode covariant effective-cone and exact-boost controls plus the Sarbach-Barausse-Preciado-Lopez frozen-principal theorem: all speeds positive and finite with nonluminal spin-1 and spin-0 sectors. Luminal formulation boundaries remain unresolved rather than rejected; nonlinear Hamiltonian boundedness and generated-action automation are separate. |
| `einstein_aether_coupled_unit_normal` | pass | Exact inverse-kinetic normality in aligned, axis-tilted, and oblique rational unit-timelike patches; spatial Hamiltonian brackets, global coupling-domain regularity, and reduced stability remain separate. |
| `einstein_aether_spatial_diffeomorphism_algebra` | pass | Exact Einstein-Aether D-D momentum-constraint sector; unit/Hamiltonian constraints, higher consistency, and reduced Hamiltonian remain separate. |
| `unit_timelike_vector_dirac_chain` | pass | Exact finite-point four-generation unit-vector control; spatial derivatives and coupling to the full Einstein-Aether metric Hamiltonian remain separate. |
| `regular_holonomic_multiplier_dirac_theorem` | pass | Exact local dimension-independent Dirac theorem on the patch C_,A G^AB C_,B != 0; it does not establish lapse/shift Hamiltonian constraints or H-D/H-H closure for a specific field theory. |
| `maxwell_unit_aether_nonlinear_hamiltonian` | pass | Exact nonlinear c3=-c1, c2=c4=0 subclass after solving the positive unit branch; this is a stability-reject control and not the generic K1..K4 Einstein-Aether Hamiltonian. |
| `dirac_constraint_surface_poisson_rank` | pass | Exact finite-dimensional polynomial negative control for an off-surface structure-function false positive. |
| `dirac_tertiary_constraint_chain` | pass | Exact finite-dimensional velocity-quadratic control with two primary, two secondary, and two higher-generation constraints. |
| `field_theory_smeared_constraint_algebra` | pass | Exact 1+1 local-functional Virasoro/hypersurface-deformation control modulo spatial boundary terms; not the 3+1 Einstein-Aether algebra. |
| `three_spatial_dimensional_smeared_brackets` | pass | Exact three-spatial-dimensional scalar canonical control, with DD equality modulo spatial boundaries; the canonical metric sector and mixed gravity bracket remain separate. |
| `canonical_metric_diffeomorphism_algebra` | pass | Exact 3D canonical-metric D-D algebra via componentwise Lie-generator identities; the curvature-dependent GR H-H bracket remains separate. |
| `canonical_metric_dewitt_kinetic_covariance` | pass | Exact arbitrary-first-jet 3D D-H kinetic-sector identity; the spatial-curvature potential and H-H bracket remain separate. |
| `einstein_aether_inhomogeneous_2d_noether` | pass | Exact arbitrary-jet 2D control through third derivatives; not the outstanding arbitrary-background 4D tensor identity. |
| `einstein_aether_inhomogeneous_4d_numeric_noether` | pass | Source-bound floating-point 4D falsification on three general Lorentzian jets; not an exact symbolic proof. |
| `dhost_degenerate_kinetic_block` | pass | Exact reduced ADM scalar kinetic block; not a full covariant DHOST classification. |
| `principal_symbol_controls` | pass | Exact reduced isotropic quadratic systems on a frozen local background. |
| `anisotropic_principal_symbol_directions` | pass | Finite declared-direction anisotropic falsification; not uniform strong hyperbolicity over the complete direction sphere. |
| `reduced_lagrangian_principal_extraction` | pass | Automatic reduced K/G^{ij}/B^i extraction with mixed omega-k matrix-polynomial characteristics; gauge reduction and arbitrary action/background export remain separate. |
| `uniform_scalar_anisotropy_sphere` | pass | Exact Rayleigh-quotient eigenvalue proof for one reduced scalar mode without time-space mixed terms; multi-field uniformity remains separate. |
| `uniform_multifield_block_certificate` | pass | Exact sufficient certificate for symmetric reduced systems without time-space mixed terms; an inconclusive block test is unresolved because block positivity is stronger than rank-one positivity. |
| `curved_background_principal_controls` | pass | Exact principal two-derivative sectors after reduction; not arbitrary nonminimal candidate extraction. |
| `einstein_aether_arbitrary_background_4d_noether` | pass | Exact abstract-tensor proof in the fixed-covector convention, source-bound to the metric/vector/multiplier Euler variations; arbitrary independent c1..c4 coefficients establish K1..K4 termwise, with corrupted-sign and omitted-connection negative controls. |
| `cadabra_metric_contraction` | pass | Exact tensor-algebra backend smoke control; not action variation. |
| `cadabra_proca_variation` | pass | Exact vector-field variation in flat derivative notation; metric variation is not included. |
| `cadabra_einstein_aether_vector_variation` | pass | Exact vector and multiplier variations; metric variation and nonlinear Hamiltonian analysis remain separate. |
| `cadabra_einstein_aether_metric_variation` | pass | Exact nonlinear abstract metric variation holding u_a fixed; total divergences are discarded after integration by parts. |
| `cadabra_einstein_hilbert_metric_variation` | pass | Exact nonlinear metric variation with the total divergence explicitly tracked; assumes compact support or the matching boundary completion. |
| `cadabra_adm_spatial_curvature_variation` | pass | Exact fully covariant variation of integral N sqrt(q) R^(3), with the first divergence step explicit and the second integration by parts executed by Cadabra. |
| `cadabra_nonlinear_contracted_bianchi` | pass | Exact nonlinear abstract-tensor identity for a Levi-Civita connection. |
| `cadabra_canonical_scalar_metric_variation` | pass | Exact nonlinear matter metric variation; p_a denotes nabla_a phi with its index down. |
| `cadabra_proca_metric_variation` | pass | Exact nonlinear matter metric variation; F_ab is connection independent by antisymmetry. |
| `cadabra_canonical_scalar_variation` | pass | Exact covariant scalar control in flat derivative notation; metric variation is not included. |

## Candidate readiness

The known-answer harness is operational, but arbitrary candidate variation, nonlinear Noether identities, full ADM/Dirac closure, and background-dependent principal symbols remain fail-closed. Observational gates stay sealed.
