# Covariant-first v3 candidate status

## Result

The v3 lane starts from a covariant action rather than inventing another static formula. It studies

```text
F_p(X) = (1 + X)^p - 1,       1/2 <= p < 1,
X = a_mu a^mu/a_sigma^2,
```

for a unit timelike Aether field. The current bounded seeds use `p=1/2`, `2/3`, and `3/4`.
All three have an exact globally positive acceleration-space Hessian for `X>=0` and decouple
relative to `X` at high field. This is a useful nonlinear constitutive family, not a complete
gravity theory by itself.

## Why the first completion failed

The simple static-null completion added a constant multiple of `K1+K4`. On the declared static
ansatz, `K1=-X` and `K4=+X`, so this preserves the desired static formula. Away from that ansatz,
however, the nonlinear `F_p` kinetic Hessian decays with increasing `X` while the constant spatial
gradient coefficient does not. The characteristic speeds therefore grow without bound. The
compiler rejects all three constant-completion seeds at the `nonlinear_x_regularity` gate.

## Derivative-matched completion

The new completion is

```text
gamma * W_p(X) * (K1 + K4),
W_p(X) = (1 + X)^(p-2) * [1 + (2p-1)X].
```

It is still exactly zero on the declared static ansatz, but its gradient coefficient now follows
the longitudinal Hessian of `F_p`. In an aligned, frozen-background vector-sector audit, with
`0 < gamma <= epsilon` and `gamma < 1`, the exact speed bounds are

| exponent | maximum squared speed |
|---|---|
| `p=1/2` | `gamma/epsilon` |
| `p=2/3` | `3 gamma/(4 epsilon)` |
| `p=3/4` | `2 gamma/(3 epsilon)` |

Thus each seed has positive finite aligned transverse and longitudinal symbols and no superluminal
mode within its declared parameter domain. The radial nonlinear Legendre transform is also
positive on this restricted sector. The additional `gamma < 1` condition comes from the metric
block: at `K_ij=0`, its five traceless/shear eigenvalues are proportional to `1-gamma W_p(X)`.
Since `0<W_p(X)<=1`, this condition prevents the completion from zeroing or reversing those
kinetic directions. The exact coupled metric-vector Hessian determinant at `K_ij=0` is recorded in
each X-operator artifact.

That restricted success does not survive the next coupled test. At the allowed point
`gamma=1/2`, `epsilon=1`, `X=1`, choose a pure traceless extrinsic-curvature shear with
`R_K=K_ij K^ij/a_sigma^2=8`. After eliminating the regular metric shear block, the exact
Schur-reduced longitudinal Aether Hessian is negative for `p=1/2`, `2/3`, and `3/4`. It is positive
at `R_K=0`, so continuity forces a finite surface where the full Legendre Hessian loses rank.
This is an off-shell, finite-background kinetic obstruction, not an observational mismatch.

## Current formal classification

The action-health packets are:

- `runs/generated-candidates/CV3-X-P1-2-MATCHED/formal-health/action-health.json`
- `runs/generated-candidates/CV3-X-P2-3-MATCHED/formal-health/action-health.json`
- `runs/generated-candidates/CV3-X-P3-4-MATCHED/formal-health/action-health.json`

Each is now `reject`, with:

- exact action compilation: pass;
- exact static dictionary and static-null matching: pass;
- aligned cone audit and coupled `K_ij=0` Hessian: pass as restricted subchecks;
- generic traceless-curvature Legendre witness: reject;
- nonlinear-X regularity gate: reject;
- termwise ADM decomposition: pass;
- later Dirac, principal, Hamiltonian, and observational stages: blocked by the earlier rejection;
- observational promotion: false.

The restricted aligned result remains useful as a regression control, but it cannot rescue the
finite generic-shear failure. None of these six v3 completions is an active candidate.

## Reproduce

```powershell
$env:PYTHONPATH = "$PWD\src"
python -m pytest -q tests/test_x_operator_ir.py tests/test_covariant_first_v3.py

python -m sigma_theory_compiler action-health `
  --spec configs/actions/generated_cv3_x_p2_3_matched.json `
  --output runs/generated-candidates/CV3-X-P2-3-MATCHED/formal-health
```

The standalone CLI reruns the complete formal-control suite. Production packet generation may
instead reuse the already verified `runs/formal-controls-v1/formal-controls.json`; the action,
intermediate artifacts, and each gate remain hash-bound.

The current production verification is 97/97 formal controls and 272 collected tests. The complete
234/234-test suite passed in 995.1 seconds; all subsequently added scoped tests pass. The newest
controls add the arbitrary-`G4(phi,X)` fixed-metric scalar current with all 20
flat third-jet coefficients canceled, complete flat nonlinear-`X` metric/scalar Noether closure, its
exact curved linear-`X` reduction, and complete arbitrary-background `G4=F(phi)` metric/scalar
Noether closure. Three exact-rational curved witnesses and a 345-symbol all-local-jet polynomial
expansion now prove the complete source-form nonlinear-`G4_X` identity. Independent Cadabra metric
variation now also cancels the complete symmetric third scalar jet, retains only curvature and
second derivatives, and rejects omission of the required Palatini third-jet completion. The
arbitrary-`G2/G3` controls remain exact as well.

The same function-family compiler now derives the regular ADM factor `G4-2 X G4_X`. On its nonzero
branch, the arbitrary L2--L4 seven-velocity Hessian has rank six, exactly one primary null direction,
and an explicit wrong-completion rank-seven negative control. On patches with invertible `Delta_N`,
the generic secondary chain, distributed D-D/D-C algebra, lapse-pair Poisson rank, and three-mode
count now pass. Global `Delta_N` invertibility, boundary zero modes, and singular strata remain
fail-closed. The generic homogeneous tensor and constraint-reduced FLRW scalar sectors now pass
their exact principal and reduced-Hamiltonian gates on the declared `G_T,F_T,G_S,F_S` positive
patch with `Theta!=0`. The 12 exact linear-`X` quartic candidates now have candidate-specific local
on-shell FLRW sign proofs inside their strong-hyperbolicity boxes, complete regular ADM/Dirac
three-mode counts, and positive reduced quadratic Hamiltonians. Arbitrary-inhomogeneous domain
preservation and nonlinear global energy remain fail-closed. An additional exact energy campaign
now controls all spatial Fourier wavenumbers of the three linearized physical modes over a nonzero
compact segment of every expanding branch, with explicit Sobolev amplification and initial-energy
radii. It deliberately does not reconstruct lapse/shift/constraint/gauge variables or bound the
nonlinear 22-variable system, so it is not used to open the observation gate. A chained exact
constraint campaign now reconstructs lapse and physical longitudinal shift in spatial `C1` with
finite candidate-specific operator bounds and positive tightened energy radii. Their time
derivatives and nonlinear reconstruction remain fail-closed.
The action-derived FLRW evolution matrix now has an outward-rounded interval integration adapter.
Its canonical massless-scalar stiff-FLRW control takes 40 certified steps, contains the analytic
endpoint, preserves the energy constraint within tolerance, and uniformly excludes every declared
tensor/scalar health boundary; off-constraint, singular-matrix, and ghost controls reject.
The generic weak-field formulation classifier now absorbs pure-`phi` `G3`, partitions the 135-axis
family into 3 generalized-harmonic k-essence assignments and 132 generalized-harmonic-ineligible
assignments, and uniformly checks the k-essence effective cone on the coefficient-bound interval
trajectory. The 6 `G3`-only cases use a dedicated cubic-Horndeski BSSN/CCZ4 conditional theorem;
5 pass the adaptive FLRW screen and one lacks a positive constraint root near the seed. All 5 have
nonzero arbitrary-local-jet principal domains with exact effective-metric and full-direction BSSN
cone certificates; evolution-invariance of those boxes is still unresolved. The 126 cases containing
`G4_X` split exactly into 12 `G4`-only linear-`X`, 30 `G4`-only nonlinear-`X`, 24 mixed linear-`X`,
and 60 mixed nonlinear-`X` cases. All 12 simplest cases are bound to exact local 11-by-11 symbols;
8 use the verified quadratic-kessence scalar-block extension. They now also pass the full
modified-harmonic six-group symmetrizer theorem on a common nonzero `2e-10` normalized local-jet box.
They also pass the chained local on-shell ADM/Dirac/quadratic-Hamiltonian campaign, and exact sign
bounds keep their expanding homogeneous rays inside the box for every finite future time. The other
114 retain the missing adapter, while the 12 require inhomogeneous box preservation/enlargement and
nonlinear global-energy evidence before promotion.

## Next design target

The next useful step is not observational fitting or a blind exponent sweep. A replacement
static-null completion must be designed against the generic coupled metric-vector Hessian from the
start. In particular, its `X`-dependent coefficient must avoid both the constant-completion
high-`X` speed divergence and the matched-completion finite-shear rank change. Only a completion
that survives those necessary tests should receive a full distributed Dirac, principal-symbol,
and nonlinear-Hamiltonian implementation. Newton/GR/Solar and direct-observation gates remain
sealed until that formal chain passes.

The formal control `static_null_k14_multiplicative_completion_no_go` strengthens this from three
examples to the whole positive multiplicative class. If `w(r)=W(r^2)` kept the large-shear Schur
coefficient nonnegative everywhere, it would have to be globally concave. Evenness gives
`w'(0)=0`; a positive nonconstant globally concave function starting there cannot decay to zero
without eventually crossing zero. A constant positive `W` avoids that contradiction but restores
the unbounded high-`X` speed ratio. Therefore a new branch needs a different tensor structure or
additional compensating operators, not another choice of positive `W` alone.
