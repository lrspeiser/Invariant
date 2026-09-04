# The pre-data admissibility compiler

## 0. Data statement

**No observational data of any kind is opened by this lane.** No data-reading
code, no network. The only file read is `../tournament/tournament.json`, a record
of a previous lane's candidate list, used solely for the retrospective. KiDS and
the wide binaries are never loaded, listed or referenced; neither is SPARC nor
any cluster catalogue. Because nothing observational is touched there is no
blind-protection issue — no fit, no split, no held-out set, nothing to leak.
`test_no_observational_data_is_opened` asserts it mechanically by intercepting
`open`: **0 files opened, 0 outside the lane**.

## 1. VALIDATION — 35 tests, 35 passed, 68.3 s

### 1.1 Whole-family verdicts against the recorded answers

| family | recorded answer | compiler | gates fired |
|---|---|---|---|
| `A1_aqual`, `A2_qumond`, `A3_qumond_rar`, `X0_newton` | pass all 15 Stage-1 screens and all 7 Stage-2 geometries | **ADMIT** | — |
| `B1_depth_mond` | fails gauge and reciprocity; third law 0.688 | **REJECT** | 4 (+ gauge FLAG) |
| `C1_wells_pow_p1` | M_dyn moves 14% from 1 row to 10^4; third law 0.564 | **REJECT** | 1, 3, 4 |
| `C2_wells_pow_p05` | p = 0.5 fails selective refinement at slope 0.50 | **REJECT** | 1, 3, 4 |
| `C3_wells_exp_p1` | as C1 | **REJECT** | 3, 4 |
| `C5_wells_pow_p2` | p = 2 fails at slope -1.00 | **REJECT** | 1, 3, 4 |
| `D1/D2/D3_pairs` | no continuum limit; zero effect on an isolated object | **REJECT** | 1, 3, 4 |
| `E1_tidal`, `E2_tidal_strong` | catalogue-artefactual; third law 0.197 | **REJECT** | 3, 4 |

**No disagreement with any recorded verdict.**

### 1.2 Recorded numbers reproduced

| quantity | recorded | compiler |
|---|---|---|
| `Phi_K(x) = Phi_N(K^-1/2 x)` residual | an identity | **4.84e-16** |
| median galaxy depth, `inf` vs `flat_1Mpc` | 0.87 dex | **0.692 dex** |
| spread over all six rules | — | **1.035 dex** (vs 0.90 off/on margin) |
| uniform refinement drift at p = 0.5/1/2 | 0.28013, identical to 5 figures | 0.2784123268692895 / ...6716 / ...86767 — identical to **9 figures across p** |
| selective refinement slope, p = 0.25...2 | 0.7507/0.5007/0.2507/0.00067/-0.4993/-0.9994 | **0.7496/0.4996/0.2496/-0.00045/-0.5004/-1.0004** |
| coherence slope, genuine kernel | **-3.11** | **-3.113** |
| coherence slope, family C p = 1 | **-0.55** | **-0.549** |
| coherence slope, pure row counting | **+0.12** | **+0.124** |
| family D at p = 1/2, lambda_min(K), N = 10 -> 800 | **3.4e-1 -> 8.3e-80** | **3.397e-1 -> 8.301e-80** |
| family C cluster M_dyn, 1 row vs 10^4 | **+14%** | **+15.5%** |
| variational base laws | 0.000 | **0.0 exactly**; independent FD route 1.5-2.1e-3, at the measured floor |
| exponential grammar cannot make a repulsive shell | k_r = exp(.) > 0 | 200,000 random draws: min g = 3.4e-240, **zero repulsive** |

### 1.3 Every recorded third-law violator fails Gate 4

| family | recorded F_net/(GM1M2/d^2) | Jacobian asymmetry | verdict |
|---|---|---|---|
| AQUAL/QUMOND base | 0.000 | 0.0 | **PASS** |
| family C1 | 0.564 | structural | **FAIL** |
| family E1 | 0.197 | 0.361 | **FAIL** |
| family B1 | 0.688 | 0.0647 | **FAIL** |
| scalar_a0 depth | 0.801/0.667/0.591 | 0.0644 | **FAIL** |
| scalar_a0 TIDAL | 0.823 | 0.0132 | **FAIL** |
| tensor_T | 0.872/0.616/0.581 | 0.0096 | **FAIL** |
| tensor_d | 1.699/1.756/1.694 | 0.1537 | **FAIL** |
| iso_K | 16.53/15.57/14.93 | 2.87e-6 | **FAIL** |

**8/8 violators fail; 4/4 variational base laws pass at round-off.**

### 1.4 The gates are two-sided

Gate 4's admissions are the non-trivial half: in QUMOND form the law comes from
an action with Phi_N still solving Poisson **iff K(u)u is a gradient in
u = grad Phi_N**. For `K = phi(|u|)I` and for the field-direction structure
`K = exp(a(|u|)(uu^T - I/3))` — where `Ku = e^(2a/3) u` exactly — it is; for
anything reading Phi_N, the Hessian, rho, a nonlocal ball mass or a row list, it
is not.

Monotone-invariance: dS/dtheta non-zero for all four gates, spreads
**0.065 / 1.270 / 1.750 / 0.050**, all printed. Two independently written
Jacobian routes agree on every verdict. The FD floor is **2.10e-3**, measured on
the Newtonian control rather than assumed, with violators 8-136x above it.

## 2. The gates

**GATE 1 — constant-K degeneracy.** A 400-point triaxial clumpy source, the exact
constant-K solution on 300 field points, against the plain Newtonian potential of
the K^(-1/2)-stretched source with masses m/sqrt(det K). Two different
expressions, agreement 4.84e-16.

What it is degenerate with, quantified: sqrt(det K) on GM is an Upsilon* offset
of **-0.0375 dex** against a measured Upsilon* uncertainty of 0.06 dex; the
eigenvalue ratios give an apparent axis ratio of **0.569**, inside what
inclination, depth and deprojection supply; and an isotropic K has axis ratio
exactly 1, i.e. **no shape signature at all, purely a G rescale**.

Escapes: (a) single-probe residual > 0.040 dex; (b) an independently measured
axis misaligned > 10 deg from radial; (c) joint residual > 0.040 dex. The verdict
uses the unbounded fit (conservative); a bounded residual is reported alongside
so an escape resting on an implausible stretch is visible.

Notable: the nonlocal invariant `qbar` is smoothed on the declared global
L_NL = 300 kpc, so across a galaxy's 10-30 kpc it does not move at all —
`spread(ln k_r) = 0.0` exactly, residual 4.6e-16 dex. **Any qbar-gated response is
a pure conductivity inside a galaxy, degenerate to round-off with the
mass-to-light ratio.**

**GATE 2 — potential gauge.** Six rules over 400 synthetic galaxies:

| rule | median abs Phi_N (m^2 s^-2) |
|---|---|
| `scale_radius` | 4.23e9 |
| `overdensity` | 5.09e9 |
| `saddle` | 8.09e9 |
| `env_volume` | 9.08e9 |
| `inf` (primary) | 9.31e9 |
| `flat_1Mpc` | 4.58e10 |

**Spread 1.035 dex against a 0.90 dex off/on margin.** Gate 2 never eliminates,
it flags. For the tournament's headline depth gate the response spans **2.07 dex
and the on/off verdict itself changes** — it fires under
`saddle`/`env_volume`/`inf`/`flat_1Mpc` and not under
`overdensity`/`scale_radius`.

**GATE 3 — coarse graining, sign convention stated in full.**

    drift(N; L) = max over probe points of ||K_N - K_ref||_inf / ||K_ref||_inf

with K_N on an N-row partition of ONE fixed continuous mass distribution and
K_ref the same response on a 16,384-point quadrature cloud. **N is held fixed and
L — the LAW's own coherence length — is swept.** Negative slope means widening
the law's kernel buys accuracy at fixed catalogue resolution, i.e. a physical
length. Near-zero or positive means the scale is set by the distance to the
nearest row.

**The +1.0 to +1.5 discrepancy with the tournament is settled.** The tournament's
`coarse.py` sweeps the PARTITION's nearest-neighbour spacing — a variable
belonging to the catalogue, not the law. Both computed here on the same controls:

| law | screen-lane convention | tournament successive-step |
|---|---|---|
| genuine smoothing kernel | **-3.113** | **+2.387** |
| family C p = 1 | **-0.549** | **+2.140** |
| pure row counting | **+0.124** | **-4.069** |

**The two conventions order the controls in opposite directions.** Neither lane
was wrong; they measured different functions of different variables, and a
positive value under one cannot be read as a positive value under the other.

Representation convergence for one galaxy as 1 object / 10 subcomponents / N
cells: drift **1.0019 / 0.2984 / 0.0654 / 0.0115 / 0.00106**. A continuum limit
exists, but one row is wrong by 100% and ten rows by 30%. Catalogue
perturbations: detection threshold **0.0927**, mesh 4x **0.0133**, merge
**0.00252**, deblend **0.00134**, ICL transfer **8.4e-13**. Five of six exceed
tolerance, and the ICL exception is informative — S is a normalised average so a
uniform mass transfer divides out, the same cancellation that makes uniform
refinement blind. **It is a cataloguer's geometric choices, not their mass
bookkeeping, that move this family.**

Family E's named repair reproduced: the tidal tensor from the row list FAILS
(drift 1.53/1.63/1.79/1.42, not shrinking); the same law with the tensor from the
smooth density PASSES exactly by construction.

**GATE 4 — reciprocity and action.** Family D's kernel is symmetric to **0.0**
and family D still fails, which is the point (Run Y: reciprocity 4.1e-16 with an
11% momentum leak — **reciprocity is not the third law**). The functional
Jacobian is computed semi-analytically, with the |g_N| channel contributing a
term exactly symmetric by construction so every reported asymmetry is the
response's. Where the instrument returns a value at or below its floor the reason
string says so and the verdict rests on the structural criterion; **the gate has
no power below its floor and that is reported, not hidden.** A declared momentum
carrier downgrades to a loud flag; no candidate in this programme has one. **A
symmetric Jacobian does not prove a relativistic completion exists — it only
fails to reject.**

## 3. THE 3,123-CANDIDATE RETROSPECTIVE

    3,123 candidates compiled in 31-47 s, caches cold, one CPU core.

    REJECTED  3,036 / 3,123  =  97.2%   before any data
      by gates 1 and 3 alone:  1,701  =  54.5%
    FLAGGED convention-dependent (gate 2):  624  =  20.0%
    ADMITTED                                 87  =   2.8%

The 87 are 3 named base laws, 12 whose fitted amplitude is exactly zero (the base
law under another name — the same outcome as the screen lane's 450 survivors all
sitting at s_0 = s_T = 0), and **72 with a live response, ALL of them gn-gated**
(scalar_a0 30, iso_K 23, tensor_d 19): QUMOND with a redefined interpolating
function, variational, partition-independent, gauge-free. **Not one is a
tournament survivor.** The compiler and the tournament agree the grammar contains
nothing new; they disagree only about how much could have been known in advance.

| gate | kills alone | unique kills |
|---|---|---|
| 1 constant-K degeneracy | 149 | **52** |
| 2 potential gauge | 0 (flags 624) | 0 |
| 3 coarse graining | 1,560 | 0 |
| 4 reciprocity and action | 2,984 | **1,335** |

`tensor_T` is the only structure rejected 390/390 by a single gate: its tensor is
built from the Hessian, so no member is variational at any amplitude with any
invariant.

### Do any of the 18 tournament survivors fail a gate? ALL 18. Seven are also flagged.

**Seven `tensor_S|phi` survivors fail Gates 3 and 4 and are gauge-flagged.** All
seven carry **p = 0, which selective refinement forbids**: measured
`d ln(W1/W2)/d ln N = +1.000` against the admissible 0, i.e. maximal
representation dependence. At p = 0 a 4e11 Msun host counts no more than a 1e9
dwarf, so how finely a DIFFERENT object is tabulated sets the field near an
untouched one. Run AH found this from the other side — "restore p = 1 and the
member violation goes 0.007 -> 0.528 dex" — and called the escape a coin flip at
4 of 8 realisations. **Gate 3 says it is not a coin flip; it is inadmissible, and
could have been known before a single member galaxy was drawn.**

**Eleven tidal-gated survivors pass Gates 1, 2 and 3 and fail only Gate 4.** |T|
is a local second derivative with no boundary constant, a functional of the
smooth density with no row list, and it separates a galaxy from a cluster shell
by two orders of magnitude so no coordinate stretch can imitate it. **Run AH's
two structural arguments for preferring the tidal gate are independently
confirmed here, before any data.**

**Gate 4's verdict is a statement about the grammar, not about these eighteen.**
2,984 of 3,123 fail because every response except the gn-gated ones reads a
functional of rho that is not grad Phi_N. **Read it as a to-do list, not an
extermination: a variational completion or a declared momentum carrier is a
prerequisite for the grammar, not an optional extra for the winner.**

### What the compiler would have saved

| stage | entering | cost |
|---|---|---|
| the tournament as run | 3,123 | four channels, SPARC + vertical + cluster solves |
| with the compiler in front | **87** (2.8%) | 31 s |
| with Gate 4 downgraded to a flag | **1,422** (45.5%) | 31 s |

**Gates 1 and 3 alone remove 54.5% on grounds no amount of data could have
overturned, and they remove exactly the seven survivors the tournament spent its
member-galaxy screen deciding.**

## 4. Disagreements chased

**4.1 Gate 1 fires on families C and D where the record names only 3 and 4.** Not
a contradiction. Family C's |S|_2 < 2/3 identically and far from a source every
n_a -> -xhat, so K becomes radially aligned with constant eigenvalues and the
exterior solution is exactly `Psi = -GM/(k_r r)` — which the screen lane itself
derived and added as `radial_far_field`. A constant radial k_r is a rescaling of
G, so **Gate 1's verdict is the bounded-response no-go seen through the
degeneracy lens.** Family D is starker: lambda = exactly 0.0 on an isolated
galaxy. Family C3 escapes Gate 1, so this is not a blanket rejection.

**4.2 Run AH's probe table is internally inconsistent by a factor of ten.** The
stated ratios do not follow from the stated medians: 6.87e-32/3.66e-34 = **188,
not 19**, and 5.54e-31/3.66e-34 = **1513, not 151**. Both annotations are exactly
10x too small, consistent with a shell median of 3.66e**-33**. One of the two
columns carries a decade error. The compiler's own probes give shell 8.01e-34,
field 9.89e-32, member 9.88e-32, so member/shell = **123** against the recorded
annotation of 151 — close. **The qualitative conclusion is unaffected and
independently confirmed:** the tidal invariant orders the member two orders of
magnitude above the shell while potential depth places them within a factor of
**1.02** (9.04e11 against 8.89e11).

**4.3 The compiler's member galaxy is not 8x tidally louder than a field
galaxy** — measured 9.88e-32 against 9.89e-32, a factor 1.001. At 10-30 kpc from
a 5e10 Msun galaxy its own tide dominates whatever the environment. A limitation
of the compiler's probe geometry, not a claim against Run AH, and it does not
touch the member/shell ordering the gates use.

## 5. Throughput — run structurally first

| quantity | measured |
|---|---|
| distinct Gate 3 families in 3,123 settings | **8** |
| distinct Gate 4 structural families | **253** |
| cost of one previously unseen family | **0.075 s** |
| structural pass (gates 2, 3, 4) over all 3,123 | 3.2 s = **983/s** |
| **inheriting a known family's verdict** | **4.2-6.9e6 settings/s** |
| settings surviving the structural gates | **139** |
| Gate 1 on that residue | **1.9 s** |
| the Stage-1 screen it must front | 2.05e6/s |

The family-verdict lookup runs at **2-3x the Stage-1 screen's own throughput**,
and Gate 1 then runs only on the 139 survivors. On a 1e9-setting Stage 1 the
family count does not grow with the setting count: about **3.5 minutes against
the screen's 8**.

## 6. What could NOT be established

- **Whether any admitted candidate fits anything.** This is a necessary-condition
  machine; 72 of its 87 admissions are QUMOND with a redefined nu, and the
  tournament's data screens already killed all of them.
- **Whether a variational completion exists for any rejected family.** Gate 4
  shows the law AS WRITTEN is not the Euler-Lagrange system of an action with
  Phi_N Newtonian. Promoting the gating field to a dynamical one might produce
  one, at the price of changing the law. Nothing here bears on a relativistic
  completion.
- **Gate 4's power below its floor** (1e-9 semi-analytic, 2.10e-3 FD).
- **Full-tensor behaviour.** Gates 1 and 4 use the spherical reduction the
  tournament's own channels score, so a candidate whose only signature is the
  phase of a shear quadrupole would not be seen. Deliberate — the full tensor
  solve costs a PDE per candidate.
- **Which column of Run AH's probe table carries the decade error**, and the
  recorded 8x member/field tidal excess. Both need the tournament lane's own
  geometry, which this lane deliberately does not import.

**Reproduce:** `python test_compiler.py` (35 tests, ~70 s) and
`python retrospective.py` (3,123 candidates, ~31-47 s).
