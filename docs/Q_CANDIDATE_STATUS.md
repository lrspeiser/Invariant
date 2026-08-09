# Projected-acceleration candidate status

## Outcome: rejected by the complete tilt-root audit

The compiler no longer stops at “q has no covariant meaning.” It now defines

```text
P_mu_nu = g_mu_nu + u_mu u_nu
a_mu = u^alpha nabla_alpha u_mu
X_a = a_mu a^mu/a_sigma^2
Q_a = (L_sigma^2/a_sigma^2)
      P^{mu rho} P^{nu sigma} nabla_mu(a_nu) nabla_rho(a_sigma)
```

and proves on the declared static ansatz that `X_a=x` and `Q_a=q` (with `c` restored in the
legacy variables). Baryonic `z` remains a diagnostic-only quantity and is never admitted to the
gravity action.

## Measured queue result

- Dense Pareto queue: 124 families.
- Rejected for forbidden baryonic `z`: 104.
- Clean families containing `q`: 20.
- Exactly expressible in the current polynomial/nonlinear basis: 6.
- Uncompleted actions rejected by necessary formal preflight: 6.
- Covariant completion variants evaluated: 1.
- Variants surviving the complete constant-tilt root audit: 0.

Pure `q` polynomial actions have a spatial-vector velocity Hessian of rank three for `k!=0` but
rank zero at `k=0`; higher powers `Q_a^n`, `n>1`, begin at perturbative order four or higher and
cannot repair the linearized zero mode. The mixed `q+sqrt(1+x)-1` action restores homogeneous
kinetic rank, but without a completion its transverse and longitudinal Aether gradient-energy
coefficients are both zero. The negative nonlinear-x partner is a kinetic ghost.

## Evaluated completion

The active action is the origin-bound family `GF-5df8715b319f54cb` plus a static-null gradient
completion:

```text
S_grav = integral sqrt(-g) [
    (M_Pl^2/2) R
  + epsilon M_Pl^2 a_sigma^2 (Q_a + sqrt(1+X_a)-1)
  - (gamma M_Pl^2/2) (K1_u/L_u^2 + K4_u/L_u^2)
  + lambda_u (u_mu u^mu+1)
]
```

Since `K1_u/L_u^2=-a_i a^i` and `K4_u/L_u^2=+a_i a^i` in the target static sector, the two
completion terms cancel there exactly. The generator formula remains
`q+sqrt(1+x)-1`; this is not a notation-only substitution or an asserted match.

On `M_Pl,L_sigma,a_sigma,epsilon,gamma>0` and `gamma<=epsilon`, the current exact necessary
certificates establish:

- constant homogeneous velocity rank;
- positive nonlinear finite-velocity `sqrt(1+X_a)-1` Hessian eigenvalues;
- positive Q kinetic-operator coefficient;
- positive transverse and longitudinal spatial-gradient energy;
- aligned dispersion `Omega^2 = gamma*kappa^2/[epsilon*(1+2 L_sigma^2 kappa^2)]`;
- low-frequency speed squared `gamma/epsilon <= 1`;
- monotone mapping of the two real rest-frame branches under constant subluminal Lorentz tilt;
- exact fixed-metric vector first variation of `X_a` and `Q_a`, with both rational tensor
  first-variation residuals equal to zero.

The real-branch result was not a complete hyperbolicity test. At every nonzero tilt and `L_sigma>0`,
the lab-frequency polynomial has degree four. The monotone branch theorem proves exactly two real
roots, so the other two form a nonreal conjugate pair. An exact rational Sturm control confirms the
count at an interior parameter point. Consequently there is no open hyperbolicity cone around the
aligned time covector for this Q operator in the present generic unit-vector theory. The candidate
is rejected before observations and does not proceed to metric variation or nonlinear closure.

The health packet now includes `higher-jet-auxiliary-ir.json`. It exactly rewrites the Q sector
with independent `b_mu` and multiplier `r^mu`, so every independent field appears with at most one
derivative. The multiplier equation restores `b_mu=u^alpha nabla_alpha u_mu` and hence the original
Q action. This is the correct input representation for the next Dirac calculation; its equivalence
certificate passes while its constraint-closure status deliberately remains unresolved.
On the aligned frozen-metric quadratic background, the lifted action now also has a complete
finite-mode Dirac control: four second-class constraints leave one physical mode per polarization,
the reduced Hamiltonian is positive for the declared signs, and
`omega^2=G k^2/(K0+K2 k^2)`. This does not cover generic tilt or metric mixing.
Those aligned results do not rescue the action because the full nonzero-tilt polynomial contains
the additional nonreal pair. A future preferred-foliation/khronon grammar would be a different
theory contract and would need its own constraint and Cauchy-surface analysis.

## Reproduce

```powershell
$env:PYTHONPATH='src'

python -m sigma_theory_compiler covariant-export `
  --priority runs/knowledge-base/generated-priority-dense.json `
  --output runs/covariant-export-v1

python -m sigma_theory_compiler action-health `
  --spec configs/actions/generated_gf_5df8715b319f54cb_static_null_completion.json `
  --output runs/generated-candidates/GF-5df8715b319f54cb-static-null-v1/formal-health
```

The latest production formal harness passes 98/98 controls, and 275 tests are collected. The complete
234/234-test suite passed in 995.1 seconds; all subsequently added scoped tests pass. The newest
controls add the arbitrary-`G4(phi,X)` fixed-metric scalar current with
all 20 flat third-jet coefficients canceled, complete flat nonlinear-`X` metric/scalar Noether
closure, its exact curved linear-`X` reduction, and complete arbitrary-background `G4=F(phi)`
metric/scalar Noether closure. Three exact-rational curved witnesses and a 345-symbol all-local-jet
polynomial expansion now prove the complete source-form nonlinear-`G4_X` identity. The independent
Cadabra metric variation also cancels its arbitrary symmetric third scalar jet and rejects the
omitted-Palatini negative control. The arbitrary-`G2/G3` identities remain exact. Scoped Ruff
checks pass.

The generic Horndeski L2--L4 pack now also passes the primary ADM gate on
`G4-2 X G4_X != 0`: all six metric velocities remain regular while `V_star` is the unique null
direction. A wrong Hessian completion restores the seventh kinetic direction. No candidate is
promoted from this alone. On patches where the complete action-specific `Delta_N` operator is
invertible, the generic secondary chain, D-D/D-C covariance, second-class Poisson rank, and
three-mode count now pass. Global operator invertibility, boundary zero modes, and singular strata
are still required. The arbitrary-function homogeneous tensor and constraint-reduced FLRW scalar
sectors now have exact principal polynomials and reduced Hamiltonians on the declared
`G_T,F_T,G_S,F_S` positive patch with `Theta!=0`. The 12 exact linear-`X` quartic candidates now
also have candidate-specific on-shell sign proofs at a local expanding FLRW state inside each
hyperbolicity box, including complete regular three-mode ADM/Dirac counts and positive reduced
quadratic Hamiltonians. Arbitrary-inhomogeneous domain preservation and nonlinear-global energy
are still required. The intervening reduced linear problem now passes: every spatial Fourier mode
of both tensor polarizations and the scalar mode has a coercive finite-horizon Sobolev-energy
estimate on a compact segment of each exact FLRW branch. Because lapse/shift/constraint
reconstruction and nonlinear product estimates are not yet present, this result is not labeled a
full PDE trapping certificate. Spatial linear lapse/shift reconstruction is now present as a
separate hash-bound campaign, including `Theta` and infrared negatives and positive tightened
energy radii. Auxiliary time derivatives now pass at linear order after exact use of the scalar
equation and an `H^4 -> C^2` estimate. The remaining reconstruction scope is nonlinear constraint
products, modified-harmonic gauge variables, quasilinear commutators, and the complete
spacetime-jet map.
The compiled FLRW background system now has a reusable outward-rounded interval integrator. The
canonical massless-scalar control encloses the analytic stiff-FLRW endpoint for all 40 accepted
steps while uniformly bounding the constraint and health margins; off-constraint, singular-matrix,
and tensor-ghost inputs are rejected. Each generated candidate still needs a declared admissible
initial-condition domain and its own certificate.
The weak-field formulation gate now exactly identifies the generalized-harmonic k-essence subset
after absorbing boundary-equivalent `G3(phi)`. Three of 135 current axis assignments take that
route and 132 are generalized-harmonic-ineligible. The former have an exact effective-metric and
Hamiltonian check. The 6 `G3`-only cases use a dedicated cubic BSSN/CCZ4 theorem; 5 pass adaptive
FLRW screening and one lacks a positive constraint root near the seed. All 5 now have nonzero
uniform arbitrary-local-jet scalar principal domains with a common time covector and full-direction
BSSN cone separation; nonlinear evolution-invariance of those boxes remains open. The 126 cases
containing `G4_X` split into 12 `G4`-only linear-`X`, 30 `G4`-only nonlinear-`X`, 24 mixed linear-`X`,
and 60 mixed nonlinear-`X`. All 12 simplest cases now have exact candidate-bound 11-by-11 symbols,
including 8 quadratic-kessence extensions, and pass a complete modified-harmonic 22-by-22
Riesz/H-star symmetrizer certificate on a common nonzero `2e-10` local-jet box. They are not yet
physics survivors: every case also passes a local on-shell ADM/Dirac/quadratic-energy certificate
and its exact expanding homogeneous ray stays inside the box at every finite future time, but
the new all-wavenumber linearized physical-energy tube still falls short of nonlinear
inhomogeneous box preservation/enlargement and nonlinear global-energy closure.
The other 114 still require their corresponding principal/symmetrizer adapters.

## Prior art and observation seal

Acceleration-dependent and acceleration-gradient operators have structural prior art in
nonprojectable Hořava/Einstein-Aether theory. The equation universe now stores that provenance and
the integration-by-parts equivalence between `a_i Delta a^i` and an acceleration-gradient square.
See [A healthy extension of Hořava gravity](https://arxiv.org/abs/0909.3525) and
[Undoing the twist](https://arxiv.org/abs/1310.5115). This project makes no novelty claim from a
finite-corpus mismatch.

No candidate observation has been opened. Dark-matter-derived targets/rescues remain prohibited;
redshift is not treated as distance by default; supernova distance inference remains excluded.
