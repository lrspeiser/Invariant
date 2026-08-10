# Remaining formal bottlenecks

This file is the execution contract for work that must finish before any generated gravity action
can reach observational fitting. A known-control pass is evidence that the harness recognizes a
known answer in its declared scope; it is not evidence that a new theory is correct.

## Claude access and budget boundary

- The current machine has no `ANTHROPIC_API_KEY` or `CLAUDE_API_KEY` in its process, user, or
  machine environment.
- Claude Code is authenticated through Claude Max OAuth and the campaign has already completed one
  structured, no-tools proposal call through the local `claude` executable.
- The durable campaign ledger reserves the full per-call ceiling before launch. The live limits are
  $500 total, $2 per call, and 250 calls. Unknown-cost failures are charged at the full reservation.
- Claude may propose bounded action grammars and analyze failed proof packets. It may never mark a
  physics gate as passed, change the evidence policy, or rescue a rejected candidate.
- Direct Anthropic API execution requires a separately created Console key supplied through a
  secret environment or secret manager. Keys must never be written to this repository or its run
  artifacts.

## B1 — Arbitrary-candidate covariant variation

**Current evidence:** Einstein–Hilbert, scalar, Proca, and the standard Einstein–Æther `K1..K4`
control actions have bounded Cadabra variations. The named quartic-Horndeski action is now compiled
and action-hash-bound. Its exact unitary-gauge ADM kinetic cancellation, fixed-metric scalar
variation, Palatini metric Euler tensor, and combined metric-scalar Noether identity now pass. The
scalar equation reduces exactly to a second-order Einstein-tensor Hessian equation.

**Missing:** an exporter that maps every admitted action-IR term to tensor declarations, performs
metric and extra-field variation, integrates all derivative variations by parts, and returns Euler
derivatives bound to the input action hash.

**Definition of done:** positive controls reproduce stored exact Euler derivatives; negative
controls fail closed; unsupported terms are `unresolved`; independent reruns produce identical
canonical residuals and artifact hashes.

## B2 — Full 4D off-shell Noether identity

**Implemented:** the exact arbitrary-background fixed-covector identity
`2 nabla^a E^(g)_ab + E_u^a nabla_b u_a - nabla_a(E_u^a u_b) + E_lambda nabla_b lambda = 0`.
Cadabra derives the Euler coefficient by inserting the complete Lie derivatives and integrating by
parts. A second source-bound script proves that the full `K1..K4` plus unit action-density variation
is a covariant divergence. The proof is abstract-index and dimension independent, hence covers all
four free coordinate identities in 4D. Independent symbolic `c1..c4` coefficients establish the
kinetic terms separately as well as in combination. The artifact is bound to the actual
fixed-covector metric, vector, and multiplier variation scripts. A corrupted metric-divergence sign
and omitted `D_ab` connection/index terms are both rejected. Reduced FLRW, exact 2D arbitrary-jet,
and numerical 4D arbitrary-jet controls remain independent corroboration.

**Status:** complete for the standard Einstein–Æther control action. Generated candidate actions
still require B1 export before this identity machinery can be instantiated for their own Euler
artifacts.

## B3 — Nonlinear field-theory ADM/Dirac closure

**Current evidence:** complete linearized Einstein–Hilbert and Proca controls, a reusable exact
finite-mode Dirac engine with maximal-minor partial Legendre transformation, iterative tertiary and
higher-generation consistency, and quotient-ring constraint-surface Poisson ranks, DHOST kinetic
degeneracy, aligned/tilted Einstein–Æther pointwise kinetic Hessians, and exact 1+1 plus
three-spatial-dimensional smeared local-functional bracket controls with spatial boundary reduction. Negative controls prove
that off-surface rank can be nonzero while the correct surface rank vanishes and that a six-constraint
higher-generation chain closes with the correct zero-mode count.
The full six-component canonical spatial metric now has an exact cotangent-lift generator control:
both metric and symmetric momentum-density Lie transformations and their 3D diffeomorphism
commutators vanish componentwise. This completes the momentum-constraint D-D sector, but not the
curvature-dependent Hamiltonian sector by itself.
The complete DeWitt kinetic Hamiltonian density also transforms as a weight-one scalar density on
arbitrary first spatial jets, closing the kinetic D-H sector. The curvature sector now has a
source-hash-bound Cadabra lapse-smeared variation, exact weight-one D-H covariance, and the complete
nonlinear H-H contraction and boundary reduction. Together these reproduce the pure-GR
hypersurface-deformation algebra with its inverse-metric structure function. Wrong DeWitt-trace,
curvature-sign, and density-weight negative controls fail.
The Einstein-Aether spatial metric, covector, normal scalar, and their conjugate momentum densities
now form an exact 3D cotangent lift, closing its complete D-D sector. This result does not promote
the theory because the unit and Hamiltonian constraints and their higher consistency are still
missing.
The bounded regular unit-vector sector now derives a four-generation, rank-four second-class chain
and the correct three vector modes. It also exposed and fixed a collision between physical `u0`
coordinates and internal Dirac multiplier names. Coupling this chain to the complete spatially
dependent Einstein-Aether Hamiltonian remains outstanding. The full pointwise metric-vector kinetic
mixing has now also been contracted with the exact unit-constraint normal in aligned, axis-tilted,
and oblique rational patches. Its inverse-kinetic normality coefficient is nonzero in all three,
while a kinetic-free vector negative control is singular. This rules out an accidental extra
pointwise gauge direction at the declared coupling point, but is not a global coupling-domain
proof.
The unit-chain result is no longer restricted to an identity kinetic toy model: an exact general
Dirac theorem now allows arbitrary coordinate-dependent kinetic mixing and potential terms. Its
four-constraint Poisson determinant is `[C_,A G^AB C_,B]^4`, so the chain is four second class
wherever the inverse-kinetic unit normal is nonzero. Controls 42 and 44 therefore join the actual
mixed Aether kinetic patches to the general multiplier-chain proof. Together with controls 47--49,
they now support both the reduced and unreduced five-mode counts on regular Legendre patches;
singular strata remain unclassified.
Control 45 supplies the first complete nonlinear metric-vector known answer: on the restricted
Maxwell-form unit-Aether coupling surface, the combined GR+Aether D-D/D-H/H-H algebra closes and
the full count is five physical modes. It then rejects that same subclass because an exact
longitudinal periodic sequence drives the reduced energy to minus infinity. This proves the
compiler can separate constraint consistency from Hamiltonian health, but it cannot be extrapolated
to generic independent `c1..c4` couplings.
Control 46 removes another generic bottleneck before the bracket calculation: every standard
`K1..K4` invariant now has an exact inhomogeneous 3+1 block representation, including lapse
acceleration and the metric-dependent transport of the spatial Aether. The positive unit branch
reduces it to nine velocities, exposes the exact aligned coupling singularities, and passes an
inhomogeneous tilted Legendre transform.
The apparent nonlinear lapse dependence is now removed as well. In the transported electric
velocity chart it is an affine velocity shift, and the Legendre transform leaves only a term linear
in `D_i N`; integration by parts makes `N` a bulk multiplier. Control 47 therefore establishes the
four lapse/shift primary constraints and their four secondary Hamiltonian/momentum seeds.
The generic D-H bracket is now executable too. Control 48 proves the complete Legendre density has
weight one under arbitrary infinitesimal spatial `GL(3)` data, termwise in `c1..c4`, and combines
that result with the exact cotangent-lift generator. Control 49 closes the remaining H-H bracket on
every regular positive-unit-branch Legendre patch. Its executable normal-embedding algebra derives
the inverse-metric structure function and verifies the Aether-specific `-chi D_i N` Hamilton-flow
term; the canonical Jacobi identity and exact arbitrary-background Noether proof then close the
bracket into the verified momentum generator. Both canonical charts count five physical modes.
Three negative controls reject missing normal variation, missing lapse gradient, and a constant
structure function.

The generated-action path now has an action-hash-bound `sigma-adm-ir-1.0` compiler. Every term in
the bounded EH/scalar/Proca/Einstein-Aether/Q/X grammar receives its exact registered 3+1 bulk
template, coefficient, boundary contract, and velocity/spatial-jet incidence. For first-derivative
families it also emits verified lapse/shift plus field-specific primary/secondary seeds. For Q it
instead records the generic-tilt `P3^{nn}=A_i A^i` block, exposes `L_n a_perp` and `L_n a_i`, and
explicitly withholds lapse/shift constraint seeds until a higher-jet Dirac reduction exists.
Missing templates or failed known-answer controls return `unresolved`.

The follow-on action-hash- and ADM-hash-bound `sigma-legendre-ir-1.0` compiler now assembles the
actual frozen term coefficients into an exact combined local kinetic Lagrangian and Hessian. It
records the velocity basis, determinant, generic rank/nullity, factorized coupling regularity
conditions, sector blocks, and null-momentum primary constraints. EH, canonical scalar, Proca, and
full `K1..K4` Einstein-Aether controls reproduce their regular generic ranks; an Aether `K2`-only
negative control exposes the expected three vector kinetic primaries. Its current proof scope is
pointwise and aligned/frozen; it does not assert distributed Poisson closure or Hamiltonian health.

The hash chain now continues through `sigma-dirac-ir-1.0`. The compiler converts the metric channel
to `dot(h)_ij=2K_ij`, derives exact canonical momenta and the local kinetic Hamiltonian, preserves
null-momentum primaries, and instantiates distributed closure only from executable controls. Pure
GR, minimally coupled canonical scalar, massive Proca after exact second-class reduction, and
regular positive-branch Einstein-Aether specializations close D-D/D-H/H-H and reproduce 2, 3, 5,
and 5 combined physical modes. The Proca H-H evidence is a direct three-dimensional local-functional
calculation with all eight boundary Euler residuals zero. An Aether `K2`-only singular control stays
`unresolved` with three kinetic primaries rather than inheriting the generic five-mode count.

**Missing:** complete global Dirac rank stratification beyond the exact local aligned factors across
singular Einstein-Aether coupling surfaces, boundary-charge completion, and automatic physical
reduced-Hamiltonian stability on generic nonlinear backgrounds. The parameter-domain compiler now
binds actual action coefficients to reduced Hamiltonian controls and rejects wrong-sign or unproved
domains. The chained `sigma-physical-hamiltonian-ir-1.0` now emits the physical Fourier-mode
coordinate/momentum Hessians, reconstructs the reduced Lagrangian, verifies the Legendre residual,
and includes exact massive-Proca second-class elimination. Einstein-Aether's five-mode quadratic
energy is certified on its open domain, but this does not turn restricted controls into a generic
nonlinear energy theorem. New action terms outside the bounded grammar still require decomposition
and closure adapters.

The first higher-jet discovery terms are now inside the fail-closed grammar. `Q_a_u` is the
projected Aether acceleration-gradient square and `X_a_u` is the normalized acceleration norm.
The exporter maps six dense-priority survivors exactly. Necessary symbol tests reject all six
uncompleted actions: pure-Q polynomials change Legendre rank at `k=0`; the positive mixed
`q+sqrt(1+x)-1` action fixes that rank but has zero Aether gradient energy; its negative partner is
a kinetic ghost. The evaluated completion adds equal actual K1/K4 coefficients. Those terms cancel on
the declared static ansatz while K1 supplies positive transverse/longitudinal gradient energy. Its
finite-velocity nonlinear-X Hessian, Q kinetic operator, and preferred-frame reduced dispersive
relation pass on `0<gamma<=epsilon`.

The complete tilt audit supersedes the earlier real-branch-only result. For every nonzero tilt the
lab-frequency polynomial is quartic, while an exact monotone-bijection theorem gives exactly two
real roots; the remaining two are a nonreal conjugate pair. The static-null Q completion is
therefore rejected by the frozen-coefficient hyperbolicity gate, not left awaiting observations.
The exact auxiliary lift and aligned positive Hamiltonian remain useful negative-control evidence,
but they cannot rescue the generic-slice failure. A preferred-foliation/khronon reformulation would
be a new grammar branch with separate field, constraint, and admissible-Cauchy-surface contracts.

The covariant-first v3 branch adds the globally convex/sublinear family
`F_p(X)=(1+X)^p-1`, with `p=1/2,2/3,3/4`. A constant static-null `K1+K4` completion is now an exact
negative control: its gradient coefficient stays finite while the nonlinear kinetic Hessian
decays, producing unbounded high-`X` characteristic speeds. The derivative-matched replacement
`W_p(X)(K1+K4)`, where `W_p=(1+X)^(p-2)[1+(2p-1)X]`, keeps the static cancellation and passes the
necessary aligned finite-speed cone test on `0<gamma<=epsilon`. The coupled metric-vector Hessian
also passes at `K_ij=0` after declaring the necessary `gamma<1` shear condition. It nevertheless
fails at finite generic curvature: at `gamma=1/2`, `epsilon=1`, `X=1`, and traceless
`K_ij K^ij/a_sigma^2=8`, the exact Schur-reduced longitudinal Hessian is negative for all three
tested exponents. Because it is positive at zero shear, a finite Legendre-rank-changing surface is
unavoidable. The three matched actions are now rejected rather than awaiting Dirac or observation
work. The next design must satisfy generic coupled-Hessian regularity before a distributed closure
adapter is built. The formal multiplicative-completion no-go control further shows that replacing
`W_p` by another positive decaying `W(X)` cannot fix this class: shear regularity would require a
globally concave even radial weight, which cannot remain positive and decay to zero; constant
weights return to the unbounded-speed failure. A different tensor structure or compensating
operator is required.

The named covariant quartic-Horndeski control now closes several previously missing known-answer steps.
For `G4=M_Pl^2/2+alpha X`, the compiler combines the `G4 R` Gauss-Codazzi boundary term with
`G4_X[(box phi)^2-phi_mn phi^mn]`, cancels the scalar normal-Hessian velocity exactly, and derives
the primary local Hessian null direction. A wrong relative coefficient produces a nonzero
three-velocity determinant. Its termwise ADM artifact and generated seven-channel Legendre artifact
now pass: the metric block has generic rank six and the retained cancelled `V_star` channel gives
the single primary `p_V_star=0`. Its scalar Euler, metric Euler, and combined diffeomorphism
identity pass. An action-bound curved-FLRW scalar-clock Dirac chain also closes a generic
second-class `p_N`/secondary lapse pair. The exact 3D metric+lapse cotangent lift and
secondary-density algebra now close; on invertible lapse-Hessian operator patches the constraint
rank gives three physical modes. This is a genuine patchwise distributed Dirac result, while
global invertibility, boundary zero modes, and singular branches remain unresolved. After adding
the positive canonical `G2=X` term, the
flat constant-timelike-gradient three-mode principal block also passes, with exact tensor speed
`(M_Pl^2+alpha A_star^2)/(M_Pl^2-alpha A_star^2)` and a luminal scalar. Curved/varying-gradient
backgrounds and uniform strong hyperbolicity remain unresolved. The reduced quadratic physical
Hamiltonian is positive and Legendre-consistent on that same healthy flat patch; nonlinear
curved-background energy and a global positive-energy theorem remain unresolved.

The linear-`X` control also now has an exact all-timelike-amplitude no-go. Simultaneous tensor
kinetic and gradient positivity requires `A_star^2<M_Pl^2/abs(alpha)`. Every nonzero sign of
`alpha` therefore hits either kinetic rank loss/ghost or cone collapse/gradient instability at a
finite amplitude. The remaining task is not to "prove" a nonexistent global domain; it is to
replace or explicitly restrict the action. The action compiler now requires every background
variable in such a patch to have a covariant definition, mass dimension, unitary-gauge
identification, and local-measurability flag; safe normalized inequalities and the preservation
contract are action-hash-bound. The pointwise certificate passes inside the declared patch, but an
exact contracting closed-FLRW solution crosses its negative-`alpha` boundary with nonzero outward
derivative and enters the gradient-unstable side. Unrestricted nonlinear preservation is therefore
rejected. Remaining scientifically legitimate options are a separately proved restricted solution
class, an explicit EFT stopping boundary, or a nonlinear `G4(X)` completion with a globally healthy
cone.

The fixed-metric scalar portion of the next background tier is exact on an arbitrary local
curvature jet: `P^(mu nu)=g^(mu nu)-2 alpha G^(mu nu)`, including a time-space flux block and exact
healthy, cone-collapse, elliptic, gradient, and metric-superluminal witnesses. This does not close
the bottleneck by itself. The complete local orthonormal-frame 11-by-11 modified-harmonic symbol is
now extracted for independent scalar-gradient, scalar-Hessian, Einstein-tensor, and covector jets.
Its action block is symmetric, has the four exact principal pure-gauge kernel vectors, reduces to
Einstein-scalar at zero coupling, and exactly reproduces both the ADM tensor polynomial and the
fixed-metric scalar cone. Its 22-by-22 generalized first-order pencil reconstructs the second-order
symbol exactly and recovers all Einstein-scalar physical and gauge mode multiplicities. Strong
hyperbolicity still requires a positive symmetrizer and the uniform bounds below. The time block now
has a conditional exact certificate: its zero-coupling determinant is nonzero, its smallest singular
value is `min(1,M_Pl^2/4)`, and an exact sum-of-squares Frobenius correction bound gives a sufficient
invertibility radius. A rank-ten curvature witness inside the old gradient-only inequality proves
that this inequality alone is not enough; uniform curvature, scalar-Hessian, and gradient-component
bounds must be part of any admitted background domain.

The formulation question is no longer implicit. Generalized harmonic gauge is rejected for the
nonzero-`G4_X` action on generic weak backgrounds by the Papallo theorem. Modified harmonic gauge
has a conditional Kovacs--Reall strong-hyperbolicity theorem, and the compiler now verifies a
nonempty exact auxiliary-cone hierarchy. To turn that theorem into an action-specific pass, the
remaining implementation must impose and verify background-jet bounds satisfying the exact
time-block radius, construct an explicitly positive symmetrizer for the 22-by-22 generalized pencil,
and bound the
correction in its induced norm uniformly over the declared background and every spatial covector
direction, and prove all physical characteristics stay separated from both auxiliary
cones. The exact flat minimum squared-speed gap is `19/36`; a conditional Weyl certificate now
supplies the sufficient target `||Delta C||_2<19/72` when the reduced squared-speed operator is
real symmetric, or self-adjoint in an explicitly positive symmetrizer using its induced norm.
Meeting it would retain a gap greater than `19/72`. The correction and its uniform norm are still
missing, so no action-specific weak-coupling threshold is invented in the meantime.

This formulation split now applies to the generated L2--L4 pack rather than only the named
linear-`X` action. Pure-`phi` `G3` is removed by its exact boundary equivalence, after which the
compiler tests canonical `G3=0` and `G4_X=0` as polynomial identities. Three of the current 135
axis assignments enter the generalized-harmonic Einstein-plus-k-essence branch; its arbitrary-`G2`
effective metric, characteristic cone, and reduced Hamiltonian pass, and the interval trajectory
uniformly checks its gradient, nonlinear Legendre, and homogeneous-energy factors. The arbitrary-
`G2` pointwise nonlinear ADM Legendre map is now exact as well; global gravitational energy remains
a separate boundary-generator problem. The other 132 are generalized-harmonic-ineligible and
remain `unresolved`, but no longer share one proof obligation. The 6 `G3`-only assignments use the
dedicated cubic-Horndeski BSSN/CCZ4 weak-field theorem; 5 pass the adaptive local FLRW background
screen and one lacks a positive constraint root near the seed. Those 5 now have sufficient uniform
arbitrary-local-jet principal domains: exact effective-metric, common-time, spatial-positivity,
discriminant, and full-direction BSSN cone-separation bounds all pass on nonzero boxes. What remains
for them is a nonlinear evolution-invariant trapping-region proof. The 126 assignments containing
`G4_X` have the exact proof
subclasses are 12 `G4`-only linear-`X`, 30 `G4`-only nonlinear-`X`, 24 mixed-`G3` linear-`X`, and
60 mixed nonlinear-`X`. Principal extraction and pointwise strong hyperbolicity are no longer
bottlenecks for the 12 simplest cases: all are hash-bound to exact local 11-by-11 symbols and pass
the complete 22-by-22 Riesz/H-star symmetrizer construction on a common `2e-10` normalized local-jet
component box. A `1e-6` box is an exact negative control. The exact expanding homogeneous ray is
now forward-invariant for every finite time. Their next bottlenecks are multidimensional enlargement
and nonlinear inhomogeneous PDE evolution-invariance of the box plus nonlinear global-energy closure.
The intervening reduced linear problem is now closed on a nonzero compact time segment: exact
all-wavenumber Sobolev energies for both tensor modes and the scalar mode have finite amplification
bounds and positive initial-energy radii for all 12 candidates. This does not yet control lapse,
shift, constraint reconstruction, gauge variables, or nonlinear products in the complete
physical-space first-order system. The exact linear scalar constraints now reconstruct and bound the lapse and
physical longitudinal shift in spatial `C1`, including a correct periodic zero-mode/infrared
contract. Auxiliary time-derivative control is now closed at linear order as well: the scalar evolution equation removes
`ddot(zeta)`, and exact `H^4 -> C^2` estimates bound `dot(alpha)` and `dot(B_i)` in spatial `C1`.
The coefficient-composition part of the nonquasilinear bridge now passes: the exact `A/B/C` blocks
are quadratic in the 24 covariant jet variables, their raw derivative tensors vanish above order
two, and the inverse-time-block recurrence supplies uniform companion `C4` envelopes for all 12
candidates. The exact nonlinear state-to-jet map and acceleration-independent Euler remainder are
also generated. The published block lift now supplies an explicitly bounded positive 55-state
symmetrizer and a conditional local vacuum gauge-fixed Cauchy theorem for compatible data in the
box interior. The remaining bridge is quantitative nonlinear constraint/gauge reconstruction,
source and state-to-jet commutator estimates, boundary energy, a computable lifespan, and proof of
long-time bootstrap invariance.

The operator audit has now split that broad bridge into exact, independently testable layers. The
fixed physical low-frequency symmetrization defect has an explicit candidate-specific `L2` bound
with Schur coefficient `4/3`, and it vanishes on correctly rescaled high dyadic shells. The targeted
annular `C6` campaign now supplies `(x,xi)` pairs `(2,4)`, `(0,6)`, `(0,5)`, and `(1,4)`, closing
finite principal anti-Wick composition constants, high-shell principal/time/projection inequalities,
and the entire finite-low principal operator for all 12 candidates. The exact dyadic audit still
proves that a naive global `H7` commutator cannot close from `H6` coefficient fields.

The good-unknown/source work now proves all 3,025 entries of the principal identity
`D_Y E55 J=iP55` and materializes the full first source Jacobian: 1,089 principal plus 594 lower
entries form an exact arithmetic-only `11 x 153` manifest with all 1,683 positions present. All 11
Euler rows have universal acceleration-affine `A/W` tensor DAGs. The common Cartesian
Minkowski/constant-scalar reference is an exact equilibrium with a localized whole-space `L2`
source convention, and a frozen-reference bilinear subfamily has an explicit `H7` coefficient for
one configured atom pair in rows 0--4. Global `D2F`--`D4F` operator envelopes now bypass full
component enumeration for the pointwise Taylor remainder, tube Lipschitz bounds, and the complete
all-direction frozen `D2F` `H7` bilinear estimate. The remaining proof work is the spatially
variable Bony/paracomposition remainder, remote paraproduct commutator, monotone global dyadic
summation, and then the nonlinear bootstrap/lifespan. Exact rational Taylor-jet recurrences now
provide solved-source operator envelopes through order nine for all 12 candidates, so the direct
derivative-order gap in `D_x^7(D2F(Y))` is closed. The remaining conservative route needs a
quantitative vector-valued `153 -> 11` paracomposition theorem, an explicit `H7/H6/H5` regularity
ledger for the coordinate atoms, every variable good-unknown Bony branch, and remote/resonant shell
constants before the global sum can be applied.

The first paracomposition-topology campaign now resolves most of that bookkeeping: 54 coordinate
atoms are in `H7`, 99 second atoms are in `H6`, differentiated second atoms are in `H5`, all 14
nonprincipal seventh-derivative partitions are compatible, the Faà di Bruno multiplicity sum is
876, the principal coefficient-low/state-high branch cancels through the 3,025-entry identity, and
the remote shell-index weight sum is explicit. An `R3`
Schwartz packet with uniformly bounded `H6` coefficient norm has coefficient-high/state-low `H7`
growth proportional to its frequency, so C9 outer smoothness alone cannot repair the lost spatial
derivative. The balanced/resonant branch is now closed by an exact cutoff/Plancherel/Bernstein
estimate, including seven partners, remote low outputs, and explicit support/weight factors. The
remaining blocker is structural, and the first componentwise high-atom `D2F` audit refutes the
currently named cancellation rather than merely leaving it unknown. At the exact flat reference,
the real source blocks give `det(A)=6561/4096` and
`D_H01(D_s01[10] F_0)=-2 a10`; an independent 15,621-node arithmetic `A/W` DAG reproduces it.
The term `T_(D_Y E55) deltaY_j` is coefficient-low/state-high and has zero projection on this
coefficient-high/state-low slice. All 12 candidates therefore have a nonzero residual (`+/-1` or
`+/-2`). The exact required four-entry correction has rank two:
`2 a10[(e0-4e4)e10^T + e10(e7+e9)^T]`. A decisive `2 x 2` minor equals `4 a10^2`, while the
same minor vanishes for every rank-one outer product. This rules out the entire one-scalar,
one-output modified-unknown ansatz class for all 12 candidates. The two displayed channels are now
realized as `v_sharp=v+Q T_(partial_1 v_low) w1[10]_high`; the certified kinematic identity makes
their differentiated principal term cancel the entire four-entry slice, including after
`J_s01=i xi1 I_11`. The actual physical packet now contains 24 TC1-principal and 14 TC2 nonzero
entries. Exact `H7 -> W2,infinity` bounds close TC1's reference principal shell and TC3's reference
shell; the C9 `D2F` bound closes TC5 pointwise with Taylor weight `1/2`; TC4 is exactly zero.
Remaining first blockers are explicit. Unchanged-`K55` separate TC2 absorption is impossible: its
direction-1 packet has rank two, whereas Hermitian absorption would force the `K55`-mapped range
into the one-dimensional high-state line. The canonical missing reciprocal block
`K55^-1 B^dagger K55` completes the pairing algebraically, but is not derived from the state and its
constraint preservation is unknown. TC1's source part needs the `Q`-contracted `partial_1F`
state-to-jet topology, variable TC3
needs the 153-atom tensor `D_Y(P55 E_v Q)`, and TC5 in `H7` needs component
`D2F[J(C_Q),J(C_Q)]` plus `H7` control of `J(C_Q)`. The next gate is to close those terms and extend
the construction to every high atom, or adopt a separately controlled derivative-loss theory.
A conditional `H8` assumption raises the second atoms to `H7`, but the packet
`U_N=N^-7 exp(iNx_1)u_0` has bounded `H7` size and `H8` size growing like `N`; an `E8` energy alone
repeats the loss. `B7`, global `H7`, dyadic application, and lifespan remain false.

The state-to-jet side now has a quantitative coordinate tube. A common `1e-13` bound on 153
normalized coordinate 2-jet atoms implies strict component bounds for `nabla(phi)`,
`nabla nabla(phi)`, and `G^mu nu` inside the existing `2e-10` covariant box, and positive radial
majorants provide derivative envelopes through order four. The next unresolved estimate is no
longer the geometry map itself: it is a termwise majorant for the acceleration-independent Euler
remainder and the nonlinear modified-harmonic gauge/constraint source composed with this tube.

The physical-space principal reduction is no longer implicit. The directional 22-by-22 companion
has been lifted exactly to the genuine 55-variable three-dimensional state `(q_A,v_A,w_iA)`, with
33 definition constraints and 33 independent curl constraints that propagate. The nonlinear
connection/Euler map and the Appendix-A 55-state symmetrizer lift now pass as well. The remaining
state-to-jet bottleneck is quantitative: bound the complete nonlinear constraint/gauge source and
commutators in Sobolev norms, formulate boundary energy where applicable, and prove a solution
cannot leave the certified local-jet box over the requested evolution interval.
Candidate-level local ADM/Dirac and quadratic-energy closure is no longer missing: all 12 have exact on-shell FLRW
witnesses inside the box, rank-six ADM Hessians, strictly nonzero lapse pairings, three physical
modes, and positive `G_T,F_T,G_S,F_S`.
The remaining 114 `G4_X` cases still require mixed or nonlinear principal/symmetrizer adapters.
All 15 source-defined weak-field derivative ratios and the `sigma=1` homogeneous scalar/slicing
cone gap are now interval-bounded for those 5 candidates. Their diagnostic maximum ratios span
`0.0548972` to `0.0954730`; by themselves those diagnostics could not close the source's
qualitative premise or arbitrary-direction requirement. The direct effective-metric proof now does
so as the executable
pointwise verdict. It does not supersede the need to prove that PDE evolution stays inside the box.

**Definition of done:** GR, Proca, and Einstein–Æther reproduce known constraint counts on the
constraint surface; the algorithm rejects off-surface rank shortcuts; each admitted candidate has
a closed algebra or remains `unresolved`; boundedness is assessed only after all second-class
constraints and gauge directions are removed.

## B4 — Gauge-reduced principal symbols and stability

**Current evidence:** scalar, Proca, GR TT, and known Einstein–Æther Minkowski mode controls plus
FLRW and static-spherical reduced metric-cone controls, automatic reduced quadratic extraction of
`K`, `G^{ij}`, and time–space `B^i` blocks, exact finite-direction anisotropic matrix-polynomial
symbols, a uniform complete-direction-sphere scalar proof via Rayleigh-quotient eigenvalues, and an
exact sufficient uniform multi-field certificate based on spatial-field block positivity. Negative
controls include defective mixed characteristics and an oblique-only gradient instability. The
multi-field certificate fails conservatively to `unresolved` because it is sufficient, not
necessary, for positivity on rank-one direction-field products.
The `sigma-stability-ir-1.0` layer now freezes sign and inequality assumptions inside the action
hash and proves the required effective coefficient domain. The subsequent
`sigma-physical-principal-ir-1.0` layer is chained to the action, Dirac, and stability hashes,
checks that the retained basis size equals the constraint-surface physical degree count, and emits
the exact reduced kinetic, gradient, propagation, polynomial, and speed data for bounded EH,
canonical-scalar, massive-Proca, and full K1..K4 Einstein-Aether actions. Exact wrong signs reject;
unproved symbolic signs remain unresolved. This closes the generated physical-symbol step inside
the current bounded adapters, not the derivation of a new gauge reduction for arbitrary future
nonminimal terms or arbitrary candidate-specific nonlinear backgrounds.
Control 50 now adds exact cycle-averaged energy coefficients for all five reduced aligned-Minkowski
Einstein-Aether modes and two positive-speed/negative-energy witnesses. It also records the
restricted hypersurface-orthogonal maximal-slice nonlinear positive-energy theorem. Control 51
combines all five physical polarizations into exact diagonal kinetic and gradient matrices and
proves the necessary-and-sufficient healthy open domain
`1-c13 > 0`, `0 < c14 < 2`, `2c1-c1^2+c3^2 > 0`, and
`c123(2+c13+3c2) > 0`. It classifies the aligned tensor, vector, scalar-trace, spin-1-gradient,
spin-0-amplitude, and spin-0-gradient singular strata and supplies five fail-closed negative
witnesses. The generic nonlinear energy problem, arbitrary-background strong hyperbolicity, and
global tilted-stratum classification remain unresolved within Control 51 alone. Control 52 closes
the latter local question exactly: after solving the unit constraint, the nine-velocity determinant
factorizes as `2 F2^2 F1^2 F0/(1+x)` for every tilt magnitude and orientation. Its zeros are exactly
the characteristic slicings `x=1/(s_s^2-1)` of healthy superluminal sectors; no finite timelike
zeros occur for healthy subluminal or luminal sectors. What remains is the genuinely harder
arbitrary inhomogeneous-background principal symbol and nonlinear energy problem. Control 53 now
closes the former for the known Einstein-Aether action within a precise sufficient formulation:
the Aether-aligned first-order tetrad system is covariantly strongly hyperbolic on arbitrary smooth
vacuum backgrounds when all three physical speeds are positive and finite and its spin-1 and spin-0
speeds are nonluminal. Exact effective-cone, Lorentz-covariance, quasilinear-Hessian, and boundary
controls are executable. Luminal formulation boundaries remain unresolved. Generated principal
reduction now exists for the bounded action adapters, while arbitrary future nonminimal actions
still need a reduction adapter. The generic nonlinear energy problem remains open.

Control 54 sharpens that open problem. It executes the complete conformal-curvature cancellation
and asymptotic boundary-charge identity behind the known nonlinear positive-energy theorem. Thus
the hypersurface-orthogonal maximal-slice sector now has a checked total-energy proof for
nonnegative matter density, `0 <= c14 <= 2`, and `c13 <= 1`. The compiler also records why this does
not finish the generic problem: the canonical bulk Hamiltonian is a sum of constraints, physical
energy is a completed asymptotic boundary generator, and the conformal proof does not cover Aether
twist or arbitrary nonmaximal data. Failed theorem premises remain `unresolved`, not counterexamples.

**Missing:** extraction from generated Euler equations, automatic removal of gauge and constraint
rows/columns, and necessary-and-sufficient uniform strong-hyperbolicity/eigenvector, kinetic,
gradient, and characteristic-cone tests for general multi-field systems with time-space mixing on
each declared background.

**Definition of done:** the same frozen candidate and background solution pass exact or
interval-certified rank, real-characteristic, complete-eigenbasis, ghost, gradient, and speed
checks for every physical polarization. Singular reductions and ambiguous roots are `unresolved`.

The completed one-billion-formula production screen currently leaves 70 exact typed covariant
lifts after the static gate. Formal necessary-condition adapters have decisively rejected all 70:
2 fail the earlier higher-jet checks, 15 have an ADM velocity-Hessian rank jump between zero and
nonzero spatial frequency, 6 have a vanishing quadratic vector principal symbol, 20 have negative
low-frequency kinetic energy plus a finite-wave-number rank loss, and 7 retain three unconstrained
negative Aether kinetic directions. The final 20 have positive aligned kinetic Hessians but their
exact generic-tilt frozen principal polynomial has a nonreal conjugate frequency pair for every
nonzero tilt. None has passed into the Solar or direct-observable galaxy gates. The next discovery
task is therefore not to weaken these gates; it is to change or enlarge the formula grammar and
generate new covariant action families that can survive the same exact ADM/Dirac/principal ladder.
The first rejection-informed grammar-v3 seed manifest does this conservatively: it prevents the old
generic `F(X_a_u,Q_a_u)` actions and `z_b` from re-entering, calibrates against five hash-bound
known-answer controls, and defines seven new typed families. Four are enabled because callable
formal adapters already exist (complete `K1..K4` Aether, convex `G2` k-essence, weak-cell cubic
`G3`, and `X`-independent conformal `G4(phi)`); three cross-domain or partial-adapter families remain
disabled with explicit blockers. Its six concrete seeds are only deterministic queue inputs. The
next execution step is candidate-specific action compilation and formal evaluation of those seeds,
then a scalable enumeration only after the bounded hook reproduces the same lineage and exclusions.

The generated L2--L4 FLRW path now has the first validated background implementation:
`flrw-background-certify` binds a compiled action, uses an outward-rounded interval Picard
enclosure for the action-derived `(h_tau,x_tau)` system, bounds energy-constraint drift, and
uniformly excludes the tensor/scalar health and evolution singular surfaces. Its canonical
massless-scalar stiff-FLRW endpoint contains the analytic solution. This closes the homogeneous
known-answer integration mechanism. The campaign wrapper now enumerates all 135 declared
assignments, uses the exact formulation partition, automatically constraint-roots a shared local
seed, and interval-certifies all 3 generalized-harmonic candidates while retaining the 132
modified-harmonic candidates as unresolved. Exhaustive multi-seed/connected-domain construction
and the arbitrary-inhomogeneous B4 proof remain open. The unresolved proof queue is now exact:
6 assignments have only the canonical-`G3` obstruction, 42 only `G4_X`, and 84 both. Five cubic
cases and all 12 `G4`-only linear-`X` cases now have pointwise local-jet domains. The next bounded
targets are evolution-invariant/larger on-shell domains for those 17 and a principal adapter for
the 24 mixed-`G3` linear-`X` cases or 30 nonlinear-`G4`-only cases.

## B5 — Background solutions, static dictionary, and observations

This stage remains sealed until B1–B4 pass for the same action hash. Then derive rather than assume
the background solution and weak-field dictionary, reproduce Newton/GR and Solar-System controls,
freeze universal constants, and test only audited direct measurements with uncertainties and sealed
holdouts. The evidence policy forbids unobserved-halo target labels and unverified distance
reconstructions from acting as truth or rescue data.

## Execution order

1. Extend the 12 forward-invariant homogeneous Dirac/Hamiltonian rays to inhomogeneous PDE trapping
   domains and a nonlinear gravitational positive-energy/boundary-generator result, including
   singular coupling strata.
2. Build B4 from the Euler and constraint artifacts produced by B1/B3 rather than from handwritten
   mode formulas.
3. Generalize B1 to generated action IR and rerun B2–B4 for every surviving action hash.
4. Unseal B5 only for a candidate with every preceding gate at `pass`.

GPU enumeration remains useful for cheap algebraic and static filters, but these bottlenecks are
mostly symbolic, sparse-linear-algebra, and proof-orchestration work. The RTX 5090 should be used for
batched candidate evaluation and numerical interval stress tests after symbolic reduction, not as a
replacement for the formal gates.
