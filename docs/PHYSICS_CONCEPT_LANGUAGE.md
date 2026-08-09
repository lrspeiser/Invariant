# Physics concept language

## Purpose

The discovery engine must search constructions, not a flat list of equation strings. A construction
can combine geometry, classical fields, quantum or statistical states, operators, couplings,
boundary conditions, and observables. The language makes those roles explicit and routes each
primitive to reusable proof adapters.

Expression in the language is not scientific admission. An unfamiliar concept can be declared with
a type and semantic contract immediately, but it remains `unresolved_missing_adapters` until every
required verifier exists. This makes distant-domain transfer possible without allowing an analogy to
masquerade as a physical derivation.

## Why coherence is not simply another gravity term

Optical coherence normally describes a state or a correlation function, such as a first-order
two-point correlation. It is not by itself a local classical gravitational action. Applying the
concept to gravity can mean several different, testable constructions:

- a coherent state of a matter field whose locally derived stress tensor sources one universal
  metric;
- coherence or decoherence of propagating gravitational-wave modes;
- a bilocal correlation observable derived from an underlying local covariant theory;
- an explicitly nonlocal effective action, which requires separate causality, initial-value, and
  auxiliary-field controls.

The language requires the generator to choose one of these meanings. It cannot paste an optical
formula into the gravity action and call the result covariant.

## Layers

1. **Typed concept graph.** Fields, states, operators, actions, interactions, and observables are
   distinct node kinds with checked input/output types.
2. **Semantic primitive registry.** Each primitive declares its mathematical meaning and the proof
   capabilities currently implemented for it.
3. **Mutation spaces.** Discrete bases, function families, representations, and construction choices
   are stored as axes. Their Cartesian-product cardinality is calculated without materializing every
   candidate.
4. **Canonical theory IR.** Equivalent index relabelings, algebraic simplifications, integrations by
   parts, known identities, and eventually invertible field redefinitions are used to suppress
   duplicates before expensive work.
5. **Verification funnel.** Type/dimension/covariance checks precede variation and Noether identities;
   ADM/Dirac, Hamiltonian, and principal-symbol gates precede observations.
6. **Counterexample search.** Every claimed domain is attacked by singular-rank, ghost, elliptic,
   superluminal, constraint-nonclosure, and boundary-crossing searches. A counterexample is often much
   cheaper and more decisive than a global proof.
7. **Observation protocol.** Only admitted theories can see direct, uncertainty-bearing data. Dark
   halo fits and prohibited reconstructed targets remain outside the scoring path.

## Scaling rule

We add one verified adapter per mathematical operator family, not one implementation per generated
formula. A family adapter accepts symbolic coefficient functions and returns proof obligations and
certificates. Cheap structural and numerical rejection can run across very large spaces; exact
symbolic work is reserved for unique survivors. GPU kernels are appropriate for batched polynomial
roots, matrix spectra, background grids, and interval prefilters. Exact tensor variation and
constraint proofs remain primarily symbolic and certificate-producing.

## Discovery loop

The intended long-running loop is:

1. select a sharp conjecture, obstruction, or target behavior;
2. retrieve structurally relevant concepts from the equation universe;
3. propose typed cross-domain constructions;
4. canonicalize and novelty-check them;
5. search aggressively for counterexamples on cheap tiers;
6. send survivors through exact covariant and Hamiltonian verification;
7. ask a reasoning model to analyze failure certificates and propose the next construction;
8. retain a complete provenance trail and submit rare survivors for expert review.

This is the physics analogue of importing algebraic-number-theory machinery into a geometry
problem: the creative step may come from a distant concept, but success is determined by a precise
claim and a checkable certificate.

## Current executable scaffold

`physics_language.py` compiles a directed typed concept graph, enforces one universal matter metric,
computes mutation-space cardinality without enumeration, and reports missing verifier adapters. The
coherence example is intentionally expressible but unresolved: the complex-scalar and correlation
nodes lack the full formal adapters. The existing Einstein plus real-scalar composition routes as
formally ready.

Compile the example without enumerating its mutation space:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler physics-compile `
  --program configs/physics_programs/coherence_gravity_example.json `
  --output runs/physics-language/coherence-gravity-ir.json
```

Next implementation packs should be:

1. extend the symbolic `G2(phi,X)`, `G3(phi,X)`, `G4(phi,X)` pack from exact normalized functions,
   `G4_X` completion, exact generic `G2/G3` variation/Noether controls, arbitrary-`G4` scalar
   variation, flat nonlinear-`G4_X` metric/Noether closure, and full curved `G4=F(phi)` closure into
   termwise tensor-backend normalization of the curved nonlinear-`G4_X` metric variation plus the
   candidate-background and arbitrary-inhomogeneous principal adapters; the ADM/Dirac gates,
   homogeneous tensor and constraint-reduced FLRW scalar principal/Hamiltonian adapters, three
   exact curved rational witnesses, and the 345-symbol all-jet theorem already pass;
2. extend the new regular rank-one reduced quadratic-DHOST pack into the full covariant quadratic
   classification, generic secondary closure, Hamiltonian, and principal adapters, then add
   controlled `G5` and cubic classes;
3. generalized vector-tensor operators with unit, gauge, and unconstrained-vector types separated;
4. complex-field state and correlation observables, kept distinct from action operators;
5. selected topological or degenerate curvature combinations, excluding generic higher-derivative
   terms unless their extra modes are explicitly retained and tested.

The first operator-family compiler is now
`src/sigma_theory_compiler/scalar_tensor_pack.py`. It accepts dimensionless functions of
`u=phi/Lambda_phi` and `x=-nabla(phi)^2/(2 Lambda_phi^4)`, derives all first and second `x`
derivatives, and binds the L4 Hessian-difference coefficient to `d(g4)/dx`. The included polynomial
family declares 135 coefficient combinations without enumerating them. Generic `G2` and cubic
Horndeski `G3` variation and Noether identities are exact. The arbitrary `G4(phi,X)` scalar current
passes a complete flat third-jet cancellation audit and curved linear-`X` reduction. Its complete
flat nonlinear-`X` metric tensor and combined Noether identity also pass, while `G4=F(phi)` has full
arbitrary-curved metric/Noether closure. Three exact-rational curved nonlinear-`G4_X` witnesses with
nonzero Weyl curvature/curvature gradients pass the complete metric/scalar identity, and a separate
345-symbol polynomial expansion proves it on the complete local jet. Independent backend derivation
of the metric variation now reaches a complete second-order raw coefficient, while termwise
normalization to the published spelling remains open. The generic ADM primary and conditional
distributed Dirac gates pass. The exact arbitrary-function homogeneous tensor block now emits
`G_T=2(G4-2 X G4_X)`, `F_T=2G4`, its two-polarization characteristic polynomial, and a positive
reduced Hamiltonian on `G_T>0, F_T>0`. The constraint-reduced FLRW scalar adapter also derives
`Sigma`, `Theta`, `G_S`, `F_S`, and `c_S^2`, with exact constraint, integration-by-parts, principal,
and Hamiltonian residuals. Candidate-specific on-shell background sign proofs,
arbitrary-inhomogeneous strong hyperbolicity, and nonlinear global energy remain unresolved. The
artifact also emits the action-derived FLRW energy/pressure system and its exact `2x2`
`(h_tau,x_tau)` evolution matrix. The `flrw-background-certify` adapter now integrates that system
with outward-rounded interval Picard enclosures, uniformly bounds constraint drift and every
`G_T/F_T/Theta/G_S/F_S` health surface, encloses the analytic canonical stiff-FLRW known answer,
and rejects three pathology inputs. Campaign-scale generation of admissible initial-condition
domains and certificates for every survivor remains. This follows
the L2-L4 structure of [generalized Galileons](https://arxiv.org/abs/1105.5723); the reduced DHOST
pack below likewise generates a degeneracy relation instead of treating every higher-derivative
coefficient as independent.

The generic weak-field formulation classifier now canonicalizes away `G3(phi)` boundary-equivalent
terms and applies the exact generalized-harmonic structural condition `canonical G3=0` and
`G4_X=0`. It partitions the sample mutation axes into 3 generalized-harmonic k-essence assignments
and 132 generalized-harmonic-ineligible assignments without promoting the latter. The k-essence branch
has an executable effective-metric, cone, and Hamiltonian theorem; modified-harmonic candidates
still require uniform weak-coupling, positive-symmetrizer, and auxiliary-cone bounds.

The k-essence adapter also derives the exact nonlinear pointwise ADM scalar momentum, Legendre
Jacobian, and Hamiltonian density. For a coefficient-bound homogeneous trajectory the interval
runner requires positive gradient, Legendre, and energy-density margins. This is a local scalar and
trajectory claim, not a gravitational boundary-charge or global positive-energy theorem.

`flrw-background-campaign` turns the finite mutation axes into a bind-many workflow. It consumes a
complete exact formulation partition, independently rebinds each eligible assignment, solves the
background constraint near a declared seed, and emits a separate interval certificate. The current
pack certifies all 3 eligible assignments and leaves all 132 generalized-harmonic-ineligible assignments
unresolved. Those 132 are not a single opaque bucket: exact residuals identify 6 `G3`-only, 42
`G4_X`-only, and 84 combined cases. It does not claim that one seed covers every connected
initial-data domain.

The proof-language partition further separates the 126 `G4_X` assignments into 12 `G4`-only
linear-`X`, 30 `G4`-only nonlinear-`X`, 24 mixed-`G3` linear-`X`, and 60 mixed nonlinear-`X`
subclasses. The dedicated linear-`X` binding adapter covers all 12 simplest cases under explicit
fixed-zero `c11,c02,d01,a01` requirements. Four reuse the canonical-scalar quartic matrix and eight
receive the exact `G2=X+c20 X^2` scalar-block extension. A reusable symmetrizer-domain adapter now
closes their pointwise hyperbolicity gate. It constructs six exact baseline Riesz projectors,
uses the source `H_star^+/-` form on the physical groups, and bounds the complete candidate companion
matrix, action form, time block, and hat quotient over a component cube containing every unit spatial
direction. All 12 pass on a common nonzero `2e-10` normalized local-jet radius. The formulas are not
observationally promoted. A further family adapter now generates an action-specific on-shell FLRW
witness within each box and evaluates the ADM Hessian, primary/secondary Dirac chain, Poisson rank,
three-mode count, and `G_T,F_T,G_S,F_S` reduced Hamiltonian automatically; all 12 pass locally.
It also proves the exact expanding homogeneous ray remains in the box for every finite future
time. A chained energy adapter now proves a finite-horizon, all-wavenumber Sobolev estimate for
the three reduced linearized physical modes and emits a positive admissible initial-energy radius.
This is not promoted to a nonlinear trapping claim: constraint/gauge reconstruction, the complete
22-variable energy, multidimensional nonlinear products, and boundary generators remain separate
language capabilities. The linear scalar constraint layer is now partially generated rather than
hand-waved: it derives bounded lapse and longitudinal-shift operators, isolates the harmless
periodic zero-mode potential kernel, and tightens the admissible energy radius. The next adapter
also generates the linear auxiliary time derivatives: it differentiates the reconstruction IR, substitutes the scalar Euler equation,
and emits exact coefficient-drift and `C2` Sobolev bounds. Nonlinear products, gauge-sector
commutators, the complete first-order state, and a full spacetime-jet trapping certificate remain
outside the pass claim.

The 6 `G3`-only formulas are routed to the dedicated cubic-Horndeski BSSN/CCZ4 weak-field theorem.
Its compiler contract keeps the source conditions explicit rather than inventing a universal
numeric interpretation of `much less than one`. Adaptive background screening certifies 5 over the
declared 0.1-time interval and rejects one missing positive constraint root; the 5 still need
uniform inhomogeneous weak-field derivative and scalar/gauge-cone bounds.
Their certificates retain all 15 homogeneous derivative-ratio bounds and the `sigma=1` cone gap.
The campaign sorts these diagnostics Pareto-style without interpreting the source's qualitative
`much less than one` as a universal number.

The next adapter closes that pointwise gap over explicit arbitrary-local-jet boxes. It binds the
source convention `R+X+G2+G3 box(phi)` to the compiler convention, evaluates the complete cubic
scalar effective metric, and proves a common time covector, positive spatial block, positive
characteristic discriminant, and full-direction slicing-cone separation. All 5 candidates possess
nonzero certified boxes. The remaining issue is dynamical: proving an initial-data region evolves
without leaving its certified box.

The first reduced DHOST family compiler is now
`src/sigma_theory_compiler/dhost_pack.py`. For the two-velocity Hessian `[[a,b],[b,c]]`, it works on
the declared regular `a!=0` branch and generates `c=b^2/a`, the null vector `(-b,a)`, and its primary
momentum constraint before enumerating coefficients. The sample declares 15 combinations and
rejects a corrupted independent `c`. Its embedded known-answer control demonstrates that kinetic
degeneracy can generate a primary-secondary second-class pair and remove the extra scalar for one
potential; that mechanism is not generalized into a full covariant proof. This fail-closed scope is
consistent with the Hamiltonian role of degeneracy described by
[Langlois and Noui](https://arxiv.org/abs/1512.06820).
