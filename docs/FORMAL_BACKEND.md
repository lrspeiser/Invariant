# Covariant field contract and formal backend

## Normative theory split

Every theory admitted to the current grammar must have the form

```text
S = S_grav[g_mu_nu, gravitational fields] + S_m[g_mu_nu, psi_m].
```

There is one physical metric, `g_mu_nu`. Every matter species is minimally coupled to that metric
and not directly to an extra scalar or vector. This makes the matter equations imply
`nabla_mu T_m^{mu nu} = 0` on shell and prevents object-specific or lensing-only laws.

The machine-readable, normative declaration is
[`../configs/covariant_field_contract.json`](../configs/covariant_field_contract.json).

## Resolution of the legacy baryonic scalar

If a conserved, timelike, coarse-grained baryon-number current exists, define

```text
n_b = sqrt(-g_mu_nu J_b^mu J_b^nu)
z_b = n_b^2/n_0^2 = -g_mu_nu J_b^mu J_b^nu/n_0^2.
```

This is a dimensionless local scalar. An observer comoving with the current can measure `n_b` as
the local baryon-number density. Its domain and uncertainty must be stated; it is not defined as a
rest density where the current is absent, null, or not meaningfully coarse-grained.

`z_b` is **not** universal matter information. Baryon number selects a matter species and does not
describe photons, leptons, or vacuum. Adding `F(z_b)` to `S_grav` therefore directly couples the
candidate gravity sector to selected matter and violates the universal minimal matter split above.
Likewise, placing a stress tensor obtained from varying `S_m` back inside `S_grav` makes the split
nonminimal and potentially implicit.

Consequently:

- `z_b` is allowed for audited measurement provenance and post-solution diagnostics;
- `z_b` is forbidden as an atom in the present gravitational action generator;
- legacy static candidates using `z` are quarantined until a covariant static-limit derivation
  shows that the apparent state variable emerges from admissible gravitational fields;
- a deliberately nonminimal baryon-coupled theory would require a new grammar and separate
  equivalence-principle, conservation, and matter-stability tests. It cannot silently enter this one.

## Admitted gravitational invariants

The first bounded covariant basis contains an optional scalar and/or unit timelike vector:

```text
X_phi = -(nabla phi)^2/(2 Lambda_phi^4)
H4_phi = [Lambda_phi^4 X_phi R + (box phi)^2
          - (nabla_mu nabla_nu phi)(nabla^mu nabla^nu phi)]/Lambda_phi^6
u_mu u^mu = -1
K1 = L_u^2 nabla_mu u_nu nabla^mu u^nu
K2 = L_u^2 (nabla_mu u^mu)^2
K3 = L_u^2 nabla_mu u_nu nabla^nu u^mu
K4 = L_u^2 (u^mu nabla_mu u_nu)(u^rho nabla_rho u^nu).
X_a = a_mu a^mu/a_sigma^2
Q_a = (L_sigma^2/a_sigma^2) P^{mu rho} P^{nu sigma}
      nabla_mu(a_nu) nabla_rho(a_sigma),  P_mu_nu=g_mu_nu+u_mu u_nu.
```

With signature `(-,+,+,+)`, the standard Einstein–Æther kinetic combination used here is
`-(c1 K1 + c2 K2 + c3 K3 - c4 K4)/2`. In particular, the positive scalar `K4=a_mu a^mu`
has coefficient `+c4/2`; this is the convention compatible with the standard combinations
`c14=c1+c4` and the known mode-speed formulas. The convention is pinned in the field contract and
action grammar so a sign change necessarily changes the canonical action hash.

These definitions are covariant, local, dimensionless, and contain at most one derivative per
extra field except the explicitly second-derivative Horndeski control `H4_phi`. They do not by
themselves show that a static Sigma expression is their limit. That
dictionary must be derived from the candidate field equations and boundary conditions.

`HORNDESKI_L4_LINEAR_X` is control-only. Together with the positive canonical `SCALAR_X` term, it
binds the named covariant action
`G4(X_c)R+G4_X[(box phi)^2-phi_mn phi^mn]`, with
`G4=M_Pl^2/2+alpha X_c` and `X_c=Lambda_phi^4 X_phi`, to a real action IR hash. Its exact
unitary-gauge ADM identity retains the Gauss-Codazzi boundary contribution and cancels every
`V_star K` term, leaving a rank-two local Hessian with null vector `(1,0,0)` in the
`(V_star,K,T)` basis. Multiplying the second-derivative completion by a wrong coefficient makes
the determinant nonzero. Its fixed-metric scalar variation is also executed by Cadabra and
reduces, using the scalar-Hessian commutator and contracted Bianchi identity, to the second-order
equation `G^(mu nu) nabla_mu nabla_nu(phi)=0`, with exact zero fourth-derivative and
curvature-gradient coefficients. Its Palatini metric Euler variation and combined metric-scalar
Noether identity now pass on the same action hash. The generated Legendre IR retains all six metric
velocities plus the cancelled `V_star` channel, proves generic rank six/nullity one, and emits
`p_V_star=0`. A curved-FLRW `phi=t` Dirac control further produces and closes the generic
second-class `p_N`/secondary lapse pair. The full 3D unitary-gauge cotangent lift closes `D-D` and
`D-C_N`; on invertible distributed lapse-Hessian patches the six spatial first-class constraints
and two lapse second-class constraints give `(20-2*6-2)/2=3` physical modes. On the flat
constant-timelike-gradient background, the exact reduced physical principal block now contains
two tensor modes with kinetic/gradient coefficients
`(M_Pl^2-alpha A_star^2)/2` and `(M_Pl^2+alpha A_star^2)/2`, plus one luminal canonical scalar.
Explicit tensor-ghost, tensor-gradient, and omitted-canonical-scalar controls are rejected.
The matching reduced three-mode quadratic Hamiltonian has positive coordinate and momentum
Hessians on the healthy witness and reconstructs the reduced Lagrangian with zero Legendre
residual. Global lapse-operator invertibility, singular/boundary branches, nonlinear energy, and
arbitrary-background strong hyperbolicity remain unresolved in
[`../runs/formal-controls-v1/action-health/quartic_horndeski_control/action-health.json`](../runs/formal-controls-v1/action-health/quartic_horndeski_control/action-health.json).

The same principal block gives an exact global-in-amplitude obstruction for this linear `G4`:
`A_star^2 < M_Pl^2/abs(alpha)` is necessary. For `alpha>0`, the tensor kinetic coefficient vanishes
at the boundary and is negative above it; for `alpha<0`, the tensor gradient coefficient does the
same. Thus the nonzero-`alpha` control is only a bounded-background/EFT control unless a nonlinear
completion changes the large-gradient behavior. No unbounded timelike-gradient domain is claimed.
The bounded patch is now part of canonical action IR rather than a prose assumption:
`A_star_squared=-g^{mu nu} nabla_mu(phi)nabla_nu(phi)` is declared as a nonnegative, locally
measurable background variable, the healthy inequality is normalized and included in the action
hash, and the stability artifact separates `pointwise_status=pass` from the global preservation
status. A changed or omitted domain changes the action hash and cannot reuse
an older health packet. The preservation question is now resolved negatively for unrestricted
evolution: the exact closed-FLRW witness
`M2=1, alpha=-1, k=1, a=4, N=1, dot(a)=-sqrt(6)/6` satisfies the lapse constraint, scale-factor
Euler equation, and differentiated constraint with `ddot(a)=-1/48`, `dot(N)=-sqrt(6)/4`. Its
healthy-boundary derivative is `sqrt(6)/2>0`, while the tensor-gradient derivative is
`-sqrt(6)/4<0`. Thus a contracting homogeneous solution crosses into the unstable region. A
separately justified restricted solution class, EFT stopping boundary, or nonlinear completion is
not excluded by this counterexample.

The arbitrary-curvature fixed-metric scalar principal part is now generated directly from the
second-order Euler equation:
`P^(mu nu)=g^(mu nu)-2 alpha G^(mu nu)`. The local orthonormal adapter accepts independent
`G^(00)=rho`, spatial eigenvalues `p1,p2,p3`, and one flux `G^(01)=j`. It derives the mixed
characteristic discriminant, Schur gradient, determinant, and directional speeds exactly. A
diagonal healthy witness has squared speeds `(3/4,7/8,5/6)`; an oblique healthy witness has exact
x-characteristic speeds `(1,-2/3)`. Rank-zero cone collapse, negative spatial gradient, and a
kinetic-flip positive-definite/elliptic symbol are rejected. A speed-squared-two witness is recorded
as outside the metric light cone but is not made a health rejection without a declared cone policy.
This control is covariant and arbitrary in the local curvature jet, but fixed-metric: completing the
gauge-reduced coupled metric-scalar principal matrix on an inhomogeneous background remains open.

Formulation dependence is now explicit. Papallo's
[general-Horndeski hyperbolicity result](https://arxiv.org/abs/1710.10155) implies that the present
`G5=0`, `G4_X=alpha!=0` action is not strongly hyperbolic on a generic weak-field background in any
generalized harmonic gauge. The later Kovacs--Reall
[modified harmonic formulation](https://arxiv.org/abs/2003.08398) proves strong hyperbolicity for
weakly coupled Horndeski systems when the physical characteristics avoid two auxiliary null cones.
The executable control constructs exact auxiliary inverse metrics `diag(-4,1,1,1)` and
`diag(-9,1,1,1)`, verifies their Lorentzian signatures and common spacelike time surface, and checks
that the known flat tensor/scalar speeds are disjoint with minimum squared-speed gap `19/36`.
Weyl's eigenvalue perturbation bound then supplies the conservative sufficient target
`||Delta C||_2<19/72` for a real-symmetric reduced squared-speed operator, or the induced norm of
an explicitly positive symmetrizer, retaining a gap greater than `19/72`. This is a quantitative
target, not a measured action-specific correction. The complete local orthonormal-frame 11-by-11
gauge-fixed symbol is now extracted from the specialized Papallo action blocks and the
Kovacs--Reall gauge block. It retains independent scalar-gradient, scalar-Hessian,
Einstein-tensor, and covector jets; its action block is symmetric, its four principal pure-gauge
vectors have exact zero residual, its zero-coupling limit is Einstein-scalar modified harmonic,
and its flat tensor and scalar reductions match the independent ADM and scalar-cone controls. Its
exact 22-by-22 generalized first-order pencil reconstructs the second-order symbol with zero
residual and recovers the 3 physical, 4 pure-gauge, and 4 gauge-violating Einstein-scalar modes per
time direction. Its time block has an exact zero-coupling determinant and singular-value floor, and
the exact Frobenius norm of the background-jet correction supplies a sufficient open invertibility
radius. A rank-ten curvature witness proves that the currently declared scalar-gradient-only domain
does not imply that radius. Uniform curvature/Hessian/gradient-component bounds, an explicitly
positive symmetrizer, a uniform induced-norm bound, and uniform cone separation remain required;
therefore the coupled-background principal gate is `unresolved` while the declared flat three-mode
patch remains a scoped pass.

The finite term library is declared in
[`../configs/covariant_action_grammar.json`](../configs/covariant_action_grammar.json). Candidate
actions are compiled to deterministic `sigma-action-ir-1.0` records before a backend sees them. The
compiler rejects unknown terms, baryonic diagnostics, missing fields, duplicate terms, a unit vector
without its norm constraint, excessive derivative order, excessive constants, a non-universal
matter metric, and an asserted rather than derived static dictionary.

The compositional-language lane now also has a normalized symbolic Horndeski L2-L4 family compiler
at `src/sigma_theory_compiler/scalar_tensor_pack.py`. It accepts dimensionless `g2(u,x)`, `g3(u,x)`,
and `g4(u,x)`, derives their first and second derivatives, and binds the L4 Hessian-difference
coefficient to `d(g4)/dx`. The production formal harness proves the arbitrary-`G2` scalar Euler
coefficient, metric stress tensor, and off-shell Noether identity on an arbitrary local
gradient/Hessian jet, including a corrupted metric-pressure-sign negative control. It now also
derives the arbitrary cubic-Horndeski `-G3(phi,X) box(phi)` scalar Euler coefficient and Hilbert
stress tensor. The covariant Hessian commutator reduces its Euler equation to second order, all four
off-shell Noether components vanish, and omitted-braiding-stress and omitted-Ricci-commutator
negative controls fail. The arbitrary-`G4(phi,X)` fixed-metric scalar current from equations
B.8/B.12 of [Kobayashi, Yamaguchi, and Yokoyama](https://arxiv.org/abs/1105.5723) is now
instantiated. Automatic four-dimensional component differentiation proves that all 20 independent
symmetric third-jet coefficients cancel on an arbitrary flat jet, and the curved linear-`X` limit
reproduces the exact Einstein-tensor scalar equation and wrong-completion negative. The complete
equation-B.4 nonlinear-`X` metric tensor is then evaluated on the same arbitrary flat jet: all six
symmetry residuals and all four combined metric-scalar Noether residuals vanish, while deleting the
`G4_XX q_mu q_nu` term fails. The entire `G4=F(phi)` subfamily separately has exact
arbitrary-background metric/scalar variation and combined Noether closure. Curved nonlinear-`G4_X`
metric/Noether closure is additionally checked on three independent exact-rational four-dimensional
normal-frame witnesses generated from second/third metric Taylor coefficients. Each has nonzero
Weyl curvature and curvature gradients; algebraic and differential Bianchi, scalar-Hessian
commutators, metric symmetry, and all four combined Noether components vanish exactly. These are
strong full-formula falsification witnesses. The separate symbolic certificate promotes this
further: 100 arbitrary second-metric Taylor
coefficients, 200 third-metric coefficients, the complete scalar jet, and ten independent local
`G4` Taylor coefficients produce six exact metric-symmetry zeros and four exact combined Noether
zeros after full polynomial expansion. This proves the source-form identity on the complete local
jet. Cadabra now independently varies the generic action as well: determinant, inverse metrics,
`G4`, `G4_X`, the scalar-Hessian connection, and the exact twice-integrated Palatini adjoint all
appear, and no derivative of `h^{ab}` remains. The exact scalar-Hessian commutator decomposition
then splits the third scalar jet into its curvature hook and arbitrary symmetric part. A rank-one
polarization certificate proves the complete symmetric coefficient vanishes; the final metric
Euler coefficient contains curvature and at most second scalar derivatives. Omitting the required
Palatini `G4_X p^c nabla_a H_bc` completion leaves two independent polarization contractions and is
rejected. Termwise normalization of the independently derived coefficient against the published
B.4 spelling is still pending. The generic L2--L4 unitary-gauge ADM kinetic adapter is now exact:
for arbitrary local `G4` and `G4_X`, the complete seven-velocity Hessian has rank six and the sole
null direction `p_V_star=0` on `G4-2 X G4_X != 0`. A wrong completion restores rank seven. Generic
preservation of `p_N` then produces `C_N`; exact D-D/D-C covariance and the two-by-two Poisson
matrix prove a second-class lapse pair and three physical modes wherever the complete
action-specific `Delta_N` operator is invertible. `G2=X, G3=0, G4=constant` proves this regular set
is nonempty, while `Delta_N=0` is rejected. Global operator invertibility, boundary zero modes,
and singular strata remain unresolved. The arbitrary-function homogeneous tensor adapter is now
exact and source-bound: `G_T=2(G4-2 X G4_X)`, `F_T=2G4`, and both physical TT modes have the exact
principal polynomial and a zero-residual Legendre Hamiltonian. Positivity is certified on
`G_T>0, F_T>0`, with ghost, gradient, kinetic-collapse, gradient-collapse, and omitted-completion
negative controls. The generic FLRW scalar perturbation block is now exact as well: lapse and shift
are eliminated from the source action, the mixed term is integrated by parts with zero residual,
and the compiler emits `Sigma`, `Theta`, `G_S`, `F_S`, `c_S^2`, the scalar principal polynomial,
and the reduced Hamiltonian. Health requires `Theta!=0, G_S>0, F_S>0` on a candidate-supplied
on-shell background. Arbitrary-inhomogeneous-background strong hyperbolicity and nonlinear global
energy remain unresolved. The function-family artifact also derives the FLRW energy constraint,
pressure equation, differentiated constraint flow, and exact `2x2` evolution matrix for
`(h_tau,x_tau)`. Its reconstruction residual vanishes identically and its determinant defines the
regular integration patch. `flrw-background-certify` now supplies an outward-rounded interval
Picard certificate. The canonical massless-scalar control takes 40 certified steps, contains its
analytic stiff-FLRW endpoint, keeps the energy-constraint enclosure within tolerance, and uniformly
excludes `G_T=0`, `F_T=0`, `Theta=0`, `G_S=0`, `F_S=0`, `x=0`, and evolution-determinant zero. An
off-constraint start, singular matrix, tensor ghost, and negative k-essence energy are rejected. For
the generalized-harmonic k-essence route the same enclosure also proves positive nonlinear scalar
Legendre and homogeneous-energy margins. Generated candidates require their own coefficient,
initial-domain, and time-span configurations.

The weak-field formulation boundary is now generated for the full L2--L4 function pack. A
`G3(phi)` term is first absorbed into `G2` by the exact boundary equivalence; the remaining
canonical `G3` and `G4_X` are then tested as polynomial identities. The current 135-axis family has
3 generalized-harmonic-eligible Einstein-plus-k-essence assignments and 132 that must use modified
harmonic gauge. For the eligible branch, the effective inverse metric
`P^(mu nu)=G2_X g^(mu nu)-G2_XX nabla^mu(phi)nabla^nu(phi)`, its determinant, characteristic cone,
and reduced Hamiltonian are exact, with ghost, gradient, and two cone-collapse negatives. The
interval background certificate binds the actual coefficients and uniformly checks
`G2_X>0`, `G2_X+2XG2_XX>0`, and `2XG2_X-G2>0` along its enclosed homogeneous trajectory. The
generic nonlinear ADM Legendre theorem derives `p=G2_X v_n`,
`dp/dv_n=G2_X+v_n^2G2_XX`, and `H=G2_X v_n^2-G2`, while explicitly leaving gravitational boundary
charges and global positive energy unresolved. Noneligible assignments remain
`unresolved`, not passed or discarded. The `G3`-only subclass now uses the dedicated
[cubic-Horndeski BSSN/CCZ4 theorem](https://arxiv.org/abs/1904.00963): its exact control records
`m>1/4`, suitable `sigma>1/2`, scalar/slicing-cone separation, and the source's 15 weak-field
`G2/G3` derivative ratios. The 126 assignments containing `G4_X` are partitioned into
12 `G4`-only linear-`X`, 30 `G4`-only nonlinear-`X`, 24 mixed-`G3` linear-`X`, and 60 mixed
nonlinear-`X` cases. A hash-replayed campaign binds all 12 simplest cases to exact local 11-by-11
symbols: 4 use canonical `G2`, while 8 add the exact arbitrary-covector quadratic-kessence
scalar-block correction. A nonzero `a01` (`phi`-dependent `G4`) is an explicit fail-closed negative.
The follow-on campaign implements the Kovacs--Reall six-Riesz-group construction: exact physical
`H_star^+/-` forms, positive 22-by-22 baseline LDL symmetrizer, resolvent/Neumann projector drift,
time-block inversion, exact diffeomorphism kernel, rank-seven hat quotient, and separated auxiliary
cones. All 12 pass on `|normalized local-jet component|<=2e-10` for the complete direction sphere;
`1e-6` rejects. A chained ADM/Dirac/Hamiltonian campaign then solves an exact expanding FLRW local
initial state for each action inside its box, verifies all background equations and constraint
preservation, substitutes the action coefficients into the rank-six Horndeski ADM Hessian and
strictly nonzero lapse pair, obtains the three-mode distributed Dirac count, and proves
`G_T,F_T,G_S,F_S>0` for the reduced quadratic Hamiltonian. Zero clock gradient, an out-of-box
witness, a tensor ghost, and a scalar ghost reject. The other 114 `G4_X` assignments still need new
adapters. Exact rational sign bounds also prove the connected expanding homogeneous ray
`0<A_star^2<=1e-20` remains inside each box for every finite future time with positive
`G_T,F_T,G_S,F_S` and lapse pairing. The 12 cases still need an inhomogeneous PDE trapping region,
larger multidimensional domains, and a nonlinear global positive-energy theorem.

The next campaign closes one strictly intermediate step without weakening that requirement. On the
compact segment where `A_star^2` falls from `1e-20` to `5e-21`, it derives exact rational lower
bounds for `G_T,F_T,G_S,F_S`, bounds their logarithmic drift along the action-derived FLRW flow,
and applies Hamilton's equations to every Fourier wavenumber. The canonical cross terms cancel,
giving `dE_s/dt <= gamma H E_s` for the two tensor modes and `zeta`; an explicit three-torus
Sobolev majorant then supplies a positive initial-energy radius controlling first time/spatial and
second spatial derivatives for a finite proper-time interval. All 12 candidates pass, while a
zero-length background segment and a scalar-gradient ghost reject. This is a linearized
inhomogeneous physical-mode theorem only. The lapse/shift/constraint reconstruction estimate and
the nonlinear 22-variable Moser/bootstrap estimate are still missing, so it does not certify that
the full nonlinear solution remains inside the local-jet box.

The chained reconstruction campaign now closes the spatial auxiliary-variable part of that gap.
Using the exact KYY scalar constraints, it derives
`alpha_k=(G_T/Theta) dot(zeta_k)` and
`beta_k=-(G_T/Theta)zeta_k-a^2(G_S/G_T)dot(zeta_k)/|k|^2`. On the declared
`2*pi` three-torus, nonzero modes have `|k|>=1`; the scalar `beta_0` kernel is harmless because the
physical longitudinal shift is `B_i(0)=i k_i beta_0=0`. Candidate-specific lower bounds on
`|Theta|` and upper bounds on `G_T`, `G_S/G_T`, and `a` turn these formulas into explicit `C1`
operator norms. Tightening the physical initial-energy radius by the square of that norm controls
the lapse, its first spatial derivative, the longitudinal shift, its first spatial derivative,
and the unitary-clock normal-gradient perturbation for all 12 cases. `Theta=0`, an unbounded
infrared inverse Laplacian, and a broken artifact hash reject. Time derivatives of these auxiliaries
and nonlinear constraint products remain the next missing reconstruction estimates.

The time-reconstruction campaign now closes the first of those two items. It differentiates the
exact `alpha_k` and physical `B_i(k)` solutions and eliminates `ddot(zeta)` using the reduced scalar
Euler equation. Exact rational bounds on the logarithmic FLRW drift of `R=G_T/Theta`,
`S=G_S/G_T`, and `G_S`, together with the scalar damping and `F_S/G_S` cone, yield uniform
operators for `dot(alpha)` and `dot(B_i)` in spatial `C1`. Because this requires one more spatial
derivative, the campaign proves and applies an explicit `H^4 -> C^2` Fourier-lattice embedding;
order three is an exact fail-closed negative. All 12 pass with a positive chained initial-energy
radius. Omitting the scalar acceleration equation leaves a nonzero exact residual and rejects.
The remaining bridge is genuinely nonlinear: products and commutators in the constraints and
modified-harmonic gauge sector must be bounded in the complete first-order system.

The first quasilinear/Moser prerequisite is now generated from the complete extracted matrices.
Every entry of the action-derived `A`, `B`, and `C` blocks is polynomial of degree at most two in
the 24 independent covariant background-jet components: four components of `nabla(phi)`, ten of
the symmetric scalar Hessian, and ten of the symmetric Einstein tensor. The campaign differentiates
those matrices exactly, proves every raw third and fourth derivative vanishes, and uses the
candidate symmetrizer certificate to verify the rational ceiling `||A^-1||_2<5`. Repeated
differentiation of `A F=X` then gives explicit uniform companion-matrix derivative envelopes
through order four for all 12 candidates. A false degree-one declaration, a missing binomial
factor, and an `H^3` declaration reject. This closes coefficient-composition regularity only: the
nonlinear state-to-covariant-jet map, nonlinear sources, symmetrizer derivatives, gauge variables,
commuted energy inequality, and bootstrap invariance are still required.

The bounded campaign adapter now consumes that exact partition without recompiling known
noneligible assignments. It rebinds all eligible assignments, solves the FLRW energy constraint
near a declared shared seed, and interval-certifies each resulting trajectory. For the current
135-assignment pack it writes 3/3 eligible certificates and records 132 generalized-harmonic-ineligible
assignments as unresolved. The latter are exactly subclassed into 6 `G3`-only, 42 `G4_X`-only,
and 84 combined-obstruction assignments, with the actual polynomial identity residuals retained.
Five of the six `G3`-only candidates pass an adaptive 0.1-time FLRW interval screen; one has no
positive constraint root near the shared seed. These five now have sufficient arbitrary-local-jet
domains. The exact source-normalized `P'_phi_phi` metric—including its trace-reversed `G3_X^2`
correction—is interval-bounded over independent gradient/Hessian boxes. `P00<0`, the spatial block
is positive, the scalar discriminant is positive, and the `sigma=1` slicing cone is separated for
the complete direction sphere using Gershgorin/norm bounds rather than sampling.
The diagnostic ranking currently puts `c20=1,d10=1` first with maximum ratio `0.0548972` and
minimum squared-cone gap `1.18975`; the source supplies no universal pass threshold for that ratio.
Nested-box bisection finds certified Hessian-component radii from `0.0142922` to `0.0274462` across
the five candidates. This closes their pointwise principal-symbol domain condition, not invariance
of the box under nonlinear PDE evolution or global Hamiltonian energy.
Multi-seed and global admissible-initial-domain construction remains separate.

The reduced DHOST family front end at `src/sigma_theory_compiler/dhost_pack.py` takes a general
two-velocity kinetic Hessian `[[a,b],[b,c]]`, declares the regular `a!=0` branch, and generates
`c=b^2/a` rather than enumerating an independent coefficient. It proves the determinant and null
direction exactly and emits the corresponding primary momentum constraint. The existing
finite-point DHOST Dirac control remains the known-answer primary-secondary mechanism; generic
secondary closure and the full covariant quadratic classification remain unresolved.

Compile a control action with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler action-compile `
  --spec configs/actions/canonical_scalar_control.json `
  --output runs/formal-controls-v1/canonical-scalar-ir.json
```

Derive the static invariant dictionary on that same action hash with:

```powershell
python -m sigma_theory_compiler action-static-dictionary `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/static-lift/einstein-aether-static-dictionary-ir.json

python -m sigma_theory_compiler static-lift-audit `
  --dictionary runs/static-lift/einstein-aether-static-dictionary-ir.json `
  --priority runs/knowledge-base/generated-priority-dense.json `
  --output runs/static-lift/dense-priority-static-lift-audit.json
```

`sigma-static-dictionary-ir-1.0` evaluates the exact tensor contractions on
`ds^2=-N^2dt^2+h_ij dx^i dx^j`, zero shift, `K_ij=0`, and `u^a=n^a`. It proves
`K1/L_u^2=-a^2`, `K2=K3=0`, `K4/L_u^2=+a^2`, and hence the static Aether action coefficient
`M_Pl^2(c1+c4)/2` multiplying `a_i a^i`. It also proves that `Q_a` reduces exactly to the legacy
`q=L_sigma^2 c^4(D_i a_j)(D^i a^j)/a_sigma^2`, verifies static scalar, Proca, EH boundary, and
baryon-current reductions, and keeps `z_b=n_b^2/n_0^2` forbidden in the gravity action.

`AETHER_Q1..Q3` and `AETHER_X_SQRT1P` are now fail-closed generator terms. The covariant exporter
finds six exact formulas in the 124-family priority queue and binds every spec to the priority-file
SHA-256. All six uncompleted actions fail a necessary aligned test. The active
`GF-5df8715b319f54cb` completion adds equal actual `K1` and `K4` coefficients: their static
acceleration contributions cancel, so the discovered `q+sqrt(1+x)-1` shape is unchanged, while
`K1` supplies positive vector gradient energy. `sigma-q-operator-ir-1.0` proves positive
homogeneous and dispersive kinetic symbols, transverse/longitudinal gradient signs, finite-velocity
convexity of `sqrt(1+X_a)-1`, and a complete constant-background tilted-frame root audit. The audit
corrects the earlier branch-only inference: the nonzero-tilt polynomial is quartic with two real
and two nonreal roots, so the current projected-Q candidate is rejected for lack of an open
hyperbolicity cone.
`sigma-q-variation-ir-1.0` derives the fixed-metric vector Euler coefficient and has exact rational
projector/acceleration first-variation residuals. The connection-dependent metric variation,
boundary completion, full Noether identity, and generic-background constraint reduction remain
unresolved.

The covariant-first v3 lane also admits nonlinear `X_a` powers `p=1/2,2/3,3/4` and exact
static-null completions. The formal convexity control proves both acceleration-space Hessian
eigenvalues positive for `X_a>=0` and `1/2<=p<1`, with `F_p(X_a)/X_a -> 0`. The generated
`sigma-x-operator-ir-1.0` then distinguishes two completion classes. Constant `K1+K4`
completions are rejected because their high-background speeds are unbounded. The
derivative-matched class `W_p(X_a)(K1+K4)` passes a necessary aligned frozen vector-sector cone
certificate on `0<gamma<=epsilon`. Its exact coupled metric-vector Hessian at `K_ij=0` also stays
regular under the declared `gamma<1` condition. The generic-shear extension is decisive: an exact
declared-domain point with finite traceless `K_ij` makes the longitudinal Schur complement negative
for every tested exponent, while it is positive at zero shear. The resulting rank-changing surface
rejects the matched actions before the Dirac/principal/Hamiltonian gates. The executable
`static_null_k14_multiplicative_completion_no_go` control then proves that no positive decaying
weight in this multiplicative class can avoid that shear obstruction; a constant weight instead
reproduces the unbounded high-`X` speed failure. See
[`COVARIANT_FIRST_V3_STATUS.md`](COVARIANT_FIRST_V3_STATUS.md).

This operator class has prior art in nonprojectable Hořava/Æther effective theories; the compiler
therefore makes no novelty claim. Relevant primary starting points are
[Blas–Pujolàs–Sibiryakov](https://arxiv.org/abs/0909.3525) and
[Jacobson](https://arxiv.org/abs/1310.5115).

Compile the same frozen action hash into a verified termwise 3+1 representation with its boundary
contract, velocity/spatial-jet channels, nondynamical variables, and primary/secondary constraint
seeds:

```powershell
python -m sigma_theory_compiler action-adm `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-adm-ir.json
```

The resulting `sigma-adm-ir-1.0` covers every term in the current bounded grammar. A `pass` proves
that the registered exact decomposition templates and their known-answer controls were instantiated
for that action hash. For Q terms this is a kinematic decomposition pass only: the recorded normal
higher-jet channels deliberately block inheritance of ordinary lapse/shift constraint seeds.

Compile the actual frozen coefficients into the combined local kinetic Hessian with:

```powershell
python -m sigma_theory_compiler action-legendre `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-legendre-ir.json
```

The resulting `sigma-legendre-ir-1.0` records the exact velocity ordering, Hessian, determinant,
generic rank and nullity, factorized regularity conditions, sector blocks, and any kinetic primary
constraints. It is independently bound to both the action and ADM hashes. The present proof scope
is a local aligned/frozen orthonormal frame for the current one-extra-field grammar. It does not
prove distributed secondary closure, a physical degree count, or bounded reduced Hamiltonian;
those remain separate fail-closed gates.

Compile the same hash chain through the canonical and distributed Dirac layer with:

```powershell
python -m sigma_theory_compiler action-dirac `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-dirac-ir.json
```

The `sigma-dirac-ir-1.0` artifact first converts `K_ij` to the actual canonical channel
`dot(h)_ij=2K_ij` in the frozen unit-lapse frame, derives momenta, solves the maximal regular
Legendre block, and records null-momentum primaries on singular branches. It then instantiates
distributed D-D/D-H/H-H closure only when exact functional/tensor controls cover that action and
coefficient specialization. Current regular EH, canonical-scalar, massive-Proca, and
Einstein-Aether controls produce 2, 3, 5, and 5 combined physical modes. Singular branches remain
`unresolved`, and physical Hamiltonian boundedness is still a separate gate.

Bind the actual coefficients to a frozen parameter domain and then construct the physical
principal matrices with:

```powershell
python -m sigma_theory_compiler action-stability `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-stability-ir.json

python -m sigma_theory_compiler action-principal `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-principal-ir.json

python -m sigma_theory_compiler action-hamiltonian `
  --spec configs/actions/einstein_aether_control.json `
  --output runs/formal-controls-v1/generated-action-adm/einstein-aether-hamiltonian-ir.json
```

`sigma-stability-ir-1.0` proves or disproves every required kinetic, gradient, mass, and
nondegeneracy condition from the parameter domain stored inside the action hash. It rejects an
exact wrong sign, leaves an undeclared symbolic sign unresolved, and binds successful implications
to executable reduced-energy and hyperbolicity controls. `sigma-physical-principal-ir-1.0` then
verifies the Dirac physical degree count, retains no constrained or gauge variables, and emits the
reduced `K`, `G`, propagation matrix, principal polynomial, and characteristic speeds on the same
action/Dirac/stability hash chain. No observational subluminality cut is imposed.

`sigma-physical-hamiltonian-ir-1.0` extends that chain through the physical quadratic Hamiltonian.
It emits coordinate, momentum, and full phase-space Hessians; reconstructs the reduced Lagrangian;
and requires an exact zero Legendre-transform residual. The scalar adapter includes its potential
Hessian. The Proca adapter solves the normal-component second-class pair and retains the positive
longitudinal `(div p)^2/(2 b m_A^2)` contribution. Einstein-Aether receives a positive five-mode
aligned-Minkowski quadratic certificate on its declared domain, while generic twisting-Aether or
nonmaximal nonlinear total-energy positivity remains explicitly `unresolved`.

## Installed backends and controls

The formal harness uses two independent layers:

- SymPy performs exact known-answer algebra, Hessians, reduced constraint matrices, mode formulas,
  and Fourier-space identities.
- Cadabra 2 performs tensor algebra. The bootstrap script extracts Ubuntu's signed package into the
  task workspace, so it does not need administrator access or alter the WSL system installation.

Bootstrap or repair Cadabra with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_cadabra_wsl.ps1
```

Run the full formal control suite with:

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m sigma_theory_compiler formal-controls `
  --contract configs/covariant_field_contract.json `
  --output runs/formal-controls-v1
```

The suite currently verifies:

1. the field contract and a baryon-specific negative control;
2. `k^mu G^(1)_mu_nu = 0` for a general linear metric perturbation;
3. the canonical scalar kinetic rank and metric characteristic cone;
4. Proca's rank-three velocity Hessian, primary/secondary constraints, nonzero constraint bracket,
   closure, three physical degrees of freedom, and positive reduced Hamiltonian, derived by the
   reusable Dirac analyzer; its exact reduced nonlinear three-dimensional smeared H-H bracket also
   closes into the covector momentum constraint modulo a spatial boundary after the normal
   second-class pair is eliminated;
5. the known spin-2, spin-1, and spin-0 Einstein-Aether mode formulas at a declared control point;
6. a rank-one quadratic-DHOST ADM scalar kinetic block, its derived primary/secondary
   second-class pair, one-mode count, positive reduced control Hamiltonian, and a nondegenerate
   extra-mode control;
7. execution of a real Cadabra tensor script with inverse-metric elimination;
8. Cadabra variation of the canonical scalar action, integration by parts, canonicalization, and
   factorization of the Euler-Lagrange variation;
9. scalar and reduced-Proca principal symbols plus ghost, negative-gradient, and superluminal
   negative controls;
10. the scalar Noether identity `nabla^mu T_mu_nu = E_phi nabla_nu(phi)` with zero exact residual;
11. action-IR-generated Proca variation and its divergence/Lorenz-constraint identity;
12. the linearized Einstein-Hilbert ADM Hessian, eight first-class constraints, two TT degrees of
    freedom, positive TT Hamiltonian, and metric cone;
13. Cadabra variation of the full `K1..K4` Einstein-Aether vector sector;
14. independent variation of its multiplier, recovering `u_mu u^mu + 1 = 0`;
15. nonlinear Einstein-Hilbert metric variation with the Palatini boundary divergence retained and
    the Einstein tensor isolated;
16. the exact nonlinear contracted differential Bianchi identity reduced to zero by Cadabra;
17. canonical scalar, reduced Proca, and GR TT principal cones on FLRW, generic static spherical,
    and Schwarzschild-exterior backgrounds;
18. Cadabra metric variation of the canonical scalar sector, deriving its stress tensor from the
    metric dependence rather than inserting a known answer;
19. Cadabra metric variation of the Proca kinetic and mass sectors, again deriving the stress
    coefficients from the action;
20. an independent exact four-dimensional off-shell Proca stress-tensor Noether residual for an
    arbitrary smooth vector field on Minkowski space.
21. exact off-shell Proca stress-Noether residuals on lapse-FLRW and generic static-spherical
    backgrounds with nonzero connection and volume-element terms;
22. the full `K1..K4` Einstein–Æther metric variation while holding the declared covector field
    fixed, including the connection variation and integration by parts;
23. a nonlinear lapse-FLRW Einstein–Æther metric/vector/multiplier Noether identity and its exact
    gauge-null Hessian direction;
24. aligned and tilted unit-Æther pointwise ADM kinetic Hessians, including the expected `c14`
    spatial-vector kinetic coefficient; and
25. a source-hash-bound arbitrary-jet 2D Einstein–Æther diffeomorphism identity for `K1..K4` and
    the unit constraint, each with two exact zero residuals.
26. a source-hash-bound unrestricted 4D coordinate-jet Einstein–Æther identity test for all five
    terms at three independent Lorentzian jets, with residuals below `7.5e-16` and an omitted-metric
    negative control at order `1e-1`; this is numerical falsification evidence, not symbolic proof.
27. an exact Dirac negative control whose raw Poisson matrix has rank two but whose quotient-ring
    reduction on the constraint surface has rank zero, preventing off-surface misclassification.
28. an exact iterative Dirac chain with two primary, two secondary, and two higher-generation
    constraints, full rank six on the constraint surface, zero physical modes, and closed
    multiplier consistency.
29. an exact 1+1 local-functional constraint algebra for smeared Hamiltonian and spatial
    diffeomorphism constraints, including derivative-of-smearing terms and equality modulo spatial
    boundary densities.
30. exact three-spatial-dimensional Hamiltonian–Hamiltonian and spatial-diffeomorphism brackets for
    a canonical field, with pointwise HH closure and DD closure modulo a spatial boundary density.
31. exact rational-direction anisotropic principal symbols with cross-gradient blocks, including a
    negative control that passes every coordinate axis but develops a gradient instability on an
    oblique direction.
32. automatic differentiation of a reduced quadratic Lagrangian into `K`, every anisotropic
    `G^{ij}`, and time–space `B^i` blocks; mixed terms are solved through an exact directional
    quadratic matrix polynomial and companion linearization.
33. a uniform all-directions scalar anisotropy proof using exact Rayleigh-quotient eigenvalue
    extrema, including kinetic-ghost and oblique-gradient negative controls.
34. a uniform all-directions multi-field sufficient certificate that proves kinetic positivity,
    gradient positivity, and metric-cone bounds through exact spatial-field block matrices. Because
    block positivity is stronger than positivity on rank-one direction-field products, failed
    certificates remain `unresolved` instead of rejecting the theory.
35. the exact three-dimensional cotangent lift of spatial diffeomorphisms on all six independent
    canonical metric components and their symmetric momentum density, including componentwise
    generator and Lie-commutator closure. The curvature-dependent GR Hamiltonian bracket is not
    included in this scoped control.
36. arbitrary-first-jet covariance of the full six-component DeWitt kinetic Hamiltonian density,
    proving the kinetic D-H bracket under the canonical metric momentum constraint. The spatial
    curvature potential and H-H bracket remain separate.
37. fully covariant Cadabra variation of the lapse-smeared spatial-curvature action, retaining the
    exact `N G^{ij} + q^{ij} nabla^2 N - nabla^i nabla^j N` coefficient and binding the script hash.
38. the exact nonlinear pure-GR Hamiltonian-Hamiltonian bracket for general symmetric `q_ij`,
    `pi^ij`, and lapse jets, closing into `D[q^{ij}(N partial_j M-M partial_j N)]` modulo a spatial
    boundary. Wrong DeWitt-trace and curvature-sign controls fail.
39. exact weight-one density covariance of `sqrt(q) R^(3)`, closing the curvature D-H sector with
    an omitted-density-weight negative control.
40. the exact three-dimensional Einstein-Aether spatial momentum generator on the metric, spatial
    Aether covector, normal Aether scalar, and all conjugate momentum densities, including
    componentwise cotangent-lift and D-D commutator closure.
41. an exact four-generation unit-timelike-vector Dirac chain deriving the multiplier primary,
    norm secondary, tangency tertiary, and multiplier-fixing quaternary constraints; its surface
    Poisson rank is four and it leaves three vector configuration modes. A multiplier-name collision
    regression prevents false closure when physical fields are named `u0`, `u1`, and so on.
42. exact inverse-kinetic unit-normal controls for the complete pointwise metric-vector kinetic
    mixing in aligned, axis-tilted, and oblique rational unit-timelike patches. All three full
    10-by-10 blocks are regular and `C_,A (H^-1)^AB C_,B` is nonzero; deleting every vector kinetic
    coupling is rejected as a rank-six negative control. This does not replace the outstanding
    spatial H-D/H-H algebra or a global coupling-domain proof.
43. an exact arbitrary-background four-dimensional Einstein–Æther Noether proof in the
    fixed-covector convention. Cadabra derives
    `2 nabla^a E^(g)_ab + E_u^a nabla_b u_a - nabla_a(E_u^a u_b) + E_lambda nabla_b lambda`
    from the complete Euler-form variation and independently proves the `K1..K4` plus unit
    action-density variation is a covariant divergence. Independent `c1..c4` coefficients make the
    proof termwise; a corrupted metric-divergence sign and omitted connection/index terms fail.
44. a dimension-independent regular-holonomic Dirac theorem with arbitrary coordinate-dependent
    inverse kinetic metric, off-diagonal kinetic mixing, and arbitrary potential. The primary,
    holonomic, tangency, and multiplier-fixing chain has Poisson determinant
    `[C_,A G^AB C_,B]^4`; therefore all four constraints are second class wherever the kinetic
    normal is nonzero. Combined with control 42, this applies to the declared exact Aether patches,
    controls 47--49 add the lapse/shift algebra on regular patches; reduced Hamiltonian stability
    remains separate.
45. the complete nonlinear ADM Hamiltonian for the Maxwell-form unit-Aether subclass
    (`c3=-c1`, `c2=c4=0`, up to convention) after solving the positive unit branch. Its exact
    D-D, D-H, and H-H brackets close with
    `D_i^A=p^j F_ij-A_i D_j p^j`, and both canonical counts give five physical modes. A wrong
    kinetic/magnetic normalization fails. The same control constructs an exact periodic
    high-frequency family whose energy tends to minus infinity, so the subclass is rejected on
    Hamiltonian stability despite its consistent algebra. This is one restricted coupling surface,
    not the generic independent-`c1..c4` result.
46. an exact generic `K1..K4` 3+1 decomposition with spatial Aether gradients, lapse acceleration,
    extrinsic curvature, normal-scalar velocity, and transported spatial-vector velocity retained.
    The four block expressions reproduce their direct four-dimensional contractions, the positive
    unit branch enforces acceleration orthogonality, and omitting the `-K_ij A^j` transport term
    fails. The aligned nine-velocity Hessian determinant is
    `-8(M2-c13)^5(2M2+c13+3c2)c14^3`; a tilted inhomogeneous rational patch has rank nine and an
    exact affine Legendre transform. Replacing the spatial-vector velocity by
    `W_i=V_i-K_ij A^j+chi D_i ln(N)` removes every nonlinear lapse-acceleration term before the
    Legendre transform.
47. exact generic lapse/shift constraint seeding. The remaining Hamiltonian acceleration term is
    `-chi p_W^i D_i ln(N)`, which becomes `N D_i(chi p_W^i)` modulo a spatial boundary. Thus lapse
    and shift have four vanishing momenta and seed one Hamiltonian plus three momentum constraints.
    The spatial D-D sector is first class; the Hamiltonian sector, global rank, nonlinear degree
    count, and boundedness remain unresolved pending the generic distributed H-D/H-H calculation.
48. exact generic D-H covariance. In an orthonormal frame at an arbitrary spatial point, every
    component of `A_i`, `W_i`, `D_i chi`, `D_i A_j`, and `K_ij` is retained with an arbitrary
    symbolic infinitesimal `GL(3)` Jacobian. All primitive contractions and each independent
    `K1..K4` invariant transform as scalars; the volume element and canonical momenta supply weight
    one. Therefore the Legendre Hamiltonian obeys
    `{D[M],H[N]}=H[M^i D_i N]` modulo a boundary. Omitting any of the three density weights fails.
49. exact generic H-H normal-deformation closure on every regular positive-unit-branch Legendre
    patch. The executable embedding calculation derives
    `S^i=q^ij(N D_j M-M D_j N)`, verifies the metric flow and the Aether-specific
    `delta_N A_i=N(W_i+K_ij A^j)-chi D_i N` term, and uses the canonical Jacobi identity plus the
    exact arbitrary-background Noether control to obtain `{H[N],H[M]}=D[S]` modulo a spatial
    boundary. Removing the normal-basis variation, the `-chi D_i N` term, or the inverse-metric
    structure function fails. Both the reduced and unreduced Dirac counts give five physical
    modes on regular patches. This follows the derivative-of-smearing organization used in the
    [nonperturbative AeST Hamiltonian analysis](https://arxiv.org/abs/2307.15126), while applying
    the geometric theorem to the independent-`c1..c4` Einstein-Aether action. Singular coupling
    strata, boundary charges, and reduced Hamiltonian boundedness remain separate.
50. exact reduced physical-mode wave energies for aligned Minkowski Einstein-Aether. After the
    linearized constraints and gauge conditions isolate two spin-2, two spin-1, and one spin-0
    modes, the common-positive-factor energy coefficients are
    `1`, `(2c1-c1^2+c3^2)/(1-c13)`, and `c14(2-c14)`. The same control evaluates the exact mode
    speeds and retains spin-1 and spin-0 examples with positive squared speed but negative energy,
    preventing a hyperbolicity-only false pass. It also records the
    [linearized energy derivation](https://arxiv.org/abs/gr-qc/0507059) and the restricted
    [nonlinear positive-energy theorem](https://arxiv.org/abs/1108.1835) for asymptotically flat,
    hypersurface-orthogonal, maximal-slice data with nonnegative matter density,
    `0<=c14<=2`, and `c13<=1`. Generic nonlinear energy positivity is explicitly not claimed.

The scalar action-IR exporter is now executable rather than control-file-only:

```powershell
python -m sigma_theory_compiler action-vary-scalar `
  --spec configs/actions/canonical_scalar_control.json `
  --output runs/formal-controls-v1/generated-scalar-variation
```

The bounded Proca exporter follows the same path:

```powershell
python -m sigma_theory_compiler action-vary-proca `
  --spec configs/actions/proca_control.json `
  --output runs/formal-controls-v1/generated-proca-variation
```

Compile one complete, fail-closed action-health packet with:

```powershell
python -m sigma_theory_compiler action-health `
  --spec configs/actions/canonical_scalar_control.json `
  --output runs/formal-controls-v1/action-health/canonical_scalar_control
```

The packet binds the canonical action hash to every applicable gate and its proof scope, including
the generated ADM, Legendre, Dirac, stability, physical-principal, and physical-Hamiltonian
artifacts. The current
known-answer results are: Einstein-Hilbert pass (2 DoF), minimally coupled scalar pass (3 combined
DoF), and Proca control pass (5 combined DoF). Generic Einstein–Æther now passes the ADM/Dirac gate
with five modes on regular positive-unit-branch patches, but the overall action remains unresolved
because generic nonlinear reduced-Hamiltonian stability is not established. Its full
`K1..K4` field/metric variation, reduced 4D FLRW identity, arbitrary-jet 2D identity, and pointwise
ADM kinetic Hessians, spatial D-D algebra, bounded unit-vector chain, and exact mixed unit-normal
patches do pass; the generic spatially inhomogeneous 3+1 decomposition, local Legendre map,
lapse/shift constraint seeds, and complete D-D/D-H/H-H sectors now also pass, as does its exact
arbitrary-background 4D Noether gate. The restricted
linearized five-mode energy control also passes at the declared rational point, while two exact
speed-only ghost witnesses fail as intended. The restricted
Maxwell-form unit-Aether surface additionally has a complete nonlinear five-mode ADM result and an
exact instability rejection. None of these
control outcomes unseals observations.

Control 51 adds the exact reduced five-mode Einstein-Aether kinetic, gradient, and propagation
matrices. On aligned Minkowski Aether it proves ghost freedom, gradient stability, real
characteristics, and strong hyperbolicity precisely when `1-c13 > 0`, `0 < c14 < 2`,
`2c1-c1^2+c3^2 > 0`, and `c123(2+c13+3c2) > 0`. Five instability witnesses and six
singular/strong-coupling strata are fail-closed. This certificate does not cover arbitrary
nonlinear backgrounds or the global tilted-stratum problem, and it imposes no observational speed
cut.

Control 52 closes the local global-tilt question on the positive unit branch. With
`x=A_i A^i>=0`, the exact nine-velocity determinant is
`2 F2(x)^2 F1(x)^2 F0(x)/(1+x)`, where `Fs=-Ks+(Ns-Ks)x` and
`s_s^2=Ns/Ks`. A healthy superluminal sector therefore loses Legendre rank precisely at
`x=1/(s_s^2-1)`, when the chosen foliation becomes characteristic for that sector. Healthy
subluminal or luminal sectors remain noncharacteristic for every finite unit-timelike tilt. This
distinguishes a bad time slicing from coupling-space strong coupling; it does not prove the
principal symbol on an arbitrary inhomogeneous nonlinear background.

Control 53 supplies that nonlinear-background result for the known Einstein-Aether action in the
Aether-aligned first-order tetrad formulation of Sarbach, Barausse, and Preciado-Lopez. The
executable control constructs the covariant effective inverse metrics
`g_s^ab=g^ab+(1-1/s_s^2)u^a u^b`, verifies their characteristic factors and exact Lorentz-boost
covariance, and confirms that the action Hessian's principal coefficients contain no background
first derivatives. The cited frozen-principal theorem is strongly hyperbolic on arbitrary smooth
vacuum backgrounds when every physical speed is positive and finite and `s1^2 != 1`,
`s0^2 != 1`. Those two nonluminal restrictions belong to this sufficient formulation: boundary
points are `unresolved`, not rejected. This result promotes the Einstein-Aether known-control
principal gate to pass, while generic nonlinear Hamiltonian boundedness still blocks overall
promotion. Primary source: <https://arxiv.org/abs/1902.05130>.

Control 54 makes the restricted nonlinear positive-energy result executable instead of merely
recording it. The hypersurface-orthogonal Hamiltonian constraint and the conformal transformation
`h_tilde_ab=N^c14 h_ab` give
`R_tilde=N^-c14[16*pi*G*rho+c14(1-c14/2)a^2+(1-c13)K_ab K^ab]` on a maximal
slice. The same conformal factor turns the exact Aether boundary energy into
`M_ae=M_ADM[h_tilde]`; both algebraic residuals vanish. Schoen-Yau positivity then applies for
nonnegative matter energy, `0 <= c14 <= 2`, and `c13 <= 1`. This is a physical asymptotic boundary
charge theorem, not positivity of a local Hamiltonian density. Twisting Aether and nonmaximal data
remain unresolved. Primary sources: <https://arxiv.org/abs/1108.1835> and
<https://arxiv.org/abs/gr-qc/0507059>.

## Fail-closed boundary

The controls establish that the harness recognizes several known answers. They do not yet establish
any generated candidate as healthy. These stages remain unresolved and block observations:

- generated arbitrary nonminimal metric and field variation (nonlinear Einstein-Hilbert, bounded
  scalar/Proca field and metric variations, and complete fixed-covector Einstein–Æther `K1..K4`
  vector/multiplier/metric control variation are implemented);
- generic nonlinear reduced-Hamiltonian stability after all constraints and gauge directions are removed
  (action-hash-bound termwise 3+1 decomposition, an exact combined local kinetic Hessian, canonical
  momenta/Hamiltonian, singular primary seeds, distributed D-D/D-H/H-H closure, and physical mode
  counts now cover the regular EH/scalar/massive-Proca/Einstein-Aether paths in the bounded grammar;
  the exact finite-mode Dirac engine
  uses a maximal-minor partial Legendre transform, projects primary consistency onto the left null
  space of the primary Poisson matrix, iterates higher generations, and performs quotient-ring
  Poisson-rank reduction on the constraint surface; exact 1+1 smeared functional-bracket machinery,
  the complete Proca control, and a finite-point quadratic-DHOST primary/secondary chain are
  implemented);
- physical Hamiltonian boundedness after all constraints are imposed;
- automatic extraction for action families outside the bounded EH/scalar/Proca/Aether adapters and
  nonlinear background-specific reduction beyond the current frozen local frames (the bounded
  adapters now emit action-hash-bound gauge-reduced physical matrices and speeds; canonical reduced
  sectors are independently verified on Minkowski, FLRW, and static-spherical backgrounds, and
  reusable anisotropic finite-direction symbols include cross-gradient blocks);
- ghost, gradient, hyperbolicity, and characteristic-cone decisions for those future unsupported
  families and candidate-specific nonlinear backgrounds;
- preservation under nonlinear evolution and admissible boundary conditions of any bounded
  background/EFT domain used to prove pointwise health;
- a derived static dictionary and Solar-System recovery for the same frozen action.

Each future backend result must include the input action hash, backend/version, assumptions,
background, intermediate artifact hashes, exact residuals, and an explicit proof scope. A timeout or
unsupported operation is `unresolved`, never a pass.

The GR/Solar reference runner is now formally gated. It consumes the passing Einstein-Hilbert
`action-health.json`, verifies all twelve required formal gates, reloads the Q-operator,
Q-variation, physical-principal, and Hamiltonian artifacts, and checks their action/content hash
bindings before allowing the five
golden reference checks to report `pass`. Without this certificate those checks are `blocked`.
Run the audit independently with:

```powershell
python -m sigma_theory_compiler observation-audit `
  --health runs/formal-controls-v1/action-health/einstein_hilbert_control/action-health.json `
  --mode known_answer_reference `
  --output runs/gr-reference/formal-eligibility.json
```

Known-answer reference eligibility does not authorize candidate observations. Candidate-data mode
requires a generated candidate with every formal promotion gate passed and no discovery blockers;
even then it only permits an independently audited dataset manifest to be considered. Dark-matter
targets/rescues, supernova distance moduli, and redshift-derived distances remain excluded by the
frozen policy.

The live requirement-by-requirement status is machine-readable in
[`../configs/formal_completion_matrix.json`](../configs/formal_completion_matrix.json). This matrix
deliberately distinguishes an operational control from end-to-end support for generated candidates.
