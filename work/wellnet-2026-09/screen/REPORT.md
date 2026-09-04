# Well-network funnel, Stages 1, 1b and 2

Lane: `work/wellnet-2026-09/screen/`
Owns the cheap front of `1e9 -> 1e5 -> 1e3 -> 1e2 -> 1e1 -> blind test`.

## 0. Headline

Screening the five specified families produced a stronger result than the
funnel design anticipated: **families C, D and E contain no viable candidate at
any parameter value**, and the reason is structural rather than a matter of
tuning. Of 158,406,840 parameter settings swept over the full exponent and
coupling grids of C and D, **450 survive the three decisive screens, and all
450 sit at the single point `s_0 = s_T = 0`, which is Newton with the well
network switched off.** Family D has zero survivors.

Four structural results carry that conclusion. Each has an analytic statement
behind it and each was verified numerically.

**1. Bounded-response no-go.** `S` is a normalised weighted average of
`n n^T - I/3`, so `|S|_2 < 2/3` identically; the sweep measures
`max |lambda(S)| = 0.666666666666` over 3,600 weight settings, saturating the
bound. `That` is normalised by its own norm, so `|That|_2 < 1`. Hence
`K = exp[.]` has eigenvalues in a fixed finite band, and for a spherical source
the exterior solution is exactly `Psi = -GM/(k_r r)`. Measured
`d ln g / d ln r` at `r ~ 1e5 r_s`: **-2.0000** for every C, D and E candidate,
against **-1.00001** for AQUAL, QUMOND and family B. A bounded K can only
rescale G. It cannot produce a flat rotation curve at any radius, for any
exponents, because a flat curve needs a gain growing linearly with r.

**2. Momentum identity.** For `div[K(x) grad Psi] = 4 pi G rho`, integrating
`rho (-grad Psi)` by parts gives exactly

    F_net,i = - oint T_ij n_j dS  -  (1/8 pi G) int (d_i K_jk) d_j Psi d_k Psi

with `T` the generalised Maxwell stress. The surface term is what Newton has;
the volume term exists only because K depends on position, and it *is* the
third-law violation. Measured for C1 on two unequal masses: net force
**0.564** of `G M1 M2 / d^2` against a Newtonian null of **0.0019**, with the
grad-K term supplying **0.555** and the surface term **1.4e-4**. The gap
between the direct force and the identity falls as **h^2.03** under grid
refinement while the force itself converges to a fixed fraction: the violation
is the law's, not the solver's.

**3. The acceleration screening in family C's third weight form does
nothing.** In `w_a = (M_a/M_0)^p / {[1+(g_N/a0)^m][1+(r_a/L)^q]^s}`, `g_N`
carries no index `a`, so it is a factor common to every term of a ratio of
sums over `a` and cancels identically. Measured effect on S: **3.89e-12** at
`eps = 1e-12`, and exactly proportional to `eps` (3.89e-12, 3.89e-6, 3.57e-3,
0.094, 0.285 for eps = 1e-12, 1e-6, 1e-3, 0.1, 1). The intended screening is
degenerate with a rescaling of the regulator. Reading `g_N` at each well
instead does have an effect, of 0.251, and is a different law.

**4. Family D's response tensor collapses exponentially with catalogue
resolution unless p = 1 and q < 3.** `||C|| ~ N^(2-2p)` for q < 3, with a
logarithmic divergence at q = 3: measured local log-slopes at N = 4096 of
**0.0102 / 1.0101 / 0.1667** for (p,q) = (1,1) / (0.5,1) / (1,3) against
predictions 0 / 1 / log-divergent. Since `K = exp[-alpha C]`, at p = 1/2 the
smallest eigenvalue of K falls **3.40e-1 -> 8.30e-80** as N goes 10 -> 800,
i.e. `d ln lambda_min / dN = -0.229` per row. That is not slow convergence;
there is no continuum limit at all.

Family A (AQUAL and QUMOND) passes all fifteen Stage-1 screens and all seven
Stage-2 geometries. Family B fails gauge invariance and reciprocity. The
Newtonian negative control passes everything; two positive controls fire
exactly where they were built to.

## 1. Data statement

**No observational data of any kind was opened by this lane.** There is no
data-reading code in it. KiDS and the wide binaries were not loaded, listed or
referenced. Every number below comes from closed-form constructions and
synthetic geometries. That is what makes this the cheap front of the funnel:
it eliminates candidates before they can touch a measurement.

## 2. What was built

| file | role |
|---|---|
| `screen.py` | Stage 1 and 1b. `run_screen(candidate)` runs fifteen screens on a candidate spec. Also holds the exponent sweeps, the targeted analyses and the sensitivity harness. |
| `stage2.py` | Stage 2. `run_stage2(candidate)` runs the seven synthetic geometries; `rerun(geoms)` redoes one geometry and merges it back. |
| `families.py` | candidate specs A-E, response-tensor constructions, equal-mass clouds, nested partitions. |
| `fieldsolve.py` | solver wrappers, listed below. |
| `dimx.py` | (M, L, T) dimensional algebra with guarded `exp` and `log`. |
| `summarise.py`, `finalise.py` | console tables; post-run repair, added controls, source hashing. |
| `screen_results.json`, `stage2_results.json` | machine-readable results, including SHA-256 of every source file and of the two unmodified gravitylab modules. |

### What was added to the solver

`work/gravitylab/solver.py` and `axisym.py` were **imported and never
modified**; their hashes are recorded in `screen_results.json`.
`fieldsolve.py` adds five things they lack.

1. **`solve_K`** — a driver for a general position-dependent tensor field K(x).
2. **`radial_far_field`** — the Dirichlet shell for a K that becomes *radially
   aligned* rather than constant at infinity. Families C and E do not tend to a
   constant K: far from the source every `n_a` tends to the same `-x_hat`, so
   `S -> x_hat x_hat - I/3` and K keeps a fixed anisotropy locked to the radius
   direction. For such a medium the exact exterior solution is
   `Psi = -GM/(k_r r)` with `k_r = x_hat . K . x_hat`. Using the constant-K
   monopole instead leaves an O(s_T) error over the whole shell.
3. **`solve_aqual`** — damped Picard for the nonlinear AQUAL operator, with a
   Dirichlet shell from the exact spherical MOND solution integrated inward
   from 60 box lengths and an inexact-Newton tolerance schedule on the inner
   linear solve. Converges in 17 Picard steps to a 6.3e-12 outer change.
4. **`solve_qumond`** — two linear solves, with the effective source built by
   `solver.apply_operator` itself, so the divergence is the one the
   discretisation conserves rather than a re-derived centre difference.
5. **`net_force`, `force_at`, `vcirc_axis`, `spherical_g`** — the diagnostics
   the reciprocity, midpoint and rotation-curve screens need, plus the exact
   spherically symmetric reduction used where a box large enough to see
   `r -> infinity` does not exist.

`axisym.freeman_vc` is used unmodified as an independent check on the disk.

**Independent cross-check of the two MOND solvers.** AQUAL (nonlinear Picard)
and QUMOND (two linear solves) share no code path. On the same point mass they
give boosts of 1.4318 / 1.5335 / 1.8240 / 2.2589 and 1.4343 / 1.5351 / 1.8241 /
2.2587 at r = 8.0, 13.1, 21.4, 35.0 kpc: agreement to 0.17% at the worst point
and 0.01% at the outermost.

## 3. Three design decisions the results depend on

**The source density and the well list are separate objects.** The right-hand
side of the field equation is a fixed smooth `rho`. The well list, the rows of
a catalogue, feeds only K. Every coarse-graining test holds `rho` fixed and
varies only the partition, so any difference is caused by the partition and
by nothing else. Discretising the source as well would confound the law's
discreteness with the source's.

**Partitions are nested refinements of one equal-mass cloud.** The reference
mass distribution is a deterministic low-discrepancy cloud of equal-mass points
(inverse-CDF radii, Fibonacci directions, fixed seed). Partitions are cut from
it by repeatedly splitting the heaviest cell at the mass median of its widest
axis, with snapshots at each requested N, so the N = 10^4 partition is a
refinement of the N = 100 partition and not an independent resample. The cloud
has to be equal-mass: on a uniform quadrature grid the greedy split cannot
balance below the mass of a single grid point, so the "refinement" silently
stops refining, which looks exactly like a converged law. Mass conservation is
asserted after every partition; the worst relative error seen was below 1e-15.

**The convergence reference is the cloud itself, never a member of the
series.** Using the finest partition as the reference would put the same
quantity on both sides of the comparison. A reference-free successive-step
series `|K_N - K_prev| / |K_prev|` is reported alongside and is the primary
criterion, because a "continuum" reference that is itself a finite row list
would flatter a law that merely converges to the reference's own row count.

## 4. Stage 1 verdict matrix

P pass, F fail, dot informational.

```
candidate                S1  S2 S2b  S3  S4  S5  S6  S7  S8  S9 S10 S11 S11b S12 S13
A1_aqual_simple           P   P   P   P   P   P   P   P   .   P   P   P   P    P   P   PASS
A2_qumond_simple          P   P   P   P   P   P   P   P   .   P   P   P   P    P   P   PASS
A3_qumond_rar             P   P   P   P   P   P   P   P   .   P   P   P   P    P   P   PASS
B1_depth_mond             P   P   P   P   F   P   P   P   .   P   F   P   P    P   P   FAIL
B2_depth_mond_weak        P   P   P   P   F   P   P   P   .   P   F   P   P    P   P   FAIL
C1_wells_pow_p1           P   P   P   P   P   P   F   P   .   P   F   P   F    P   F   FAIL
C2_wells_pow_p05          P   P   P   P   P   P   F   P   .   P   F   P   F    F   F   FAIL
C3_wells_exp_p1           P   P   P   P   P   P   F   P   .   P   F   P   F    P   F   FAIL
C4_wells_gsupp_p1         P   P   P   P   P   P   F   P   .   P   F   P   F    P   F   FAIL
C5_wells_pow_p2           P   P   P   P   P   P   F   P   .   P   F   P   F    F   F   FAIL
D1_pairs_p1_q1            P   P   P   P   P   P   F   P   .   P   P   P   F    F   F   FAIL
D2_pairs_p05_q1           P   P   P   P   P   F   F   P   .   P   P   F   F    F   F   FAIL
D3_pairs_p1_q3            P   P   P   P   P   F   F   P   .   P   P   P   F    F   F   FAIL
E1_tidal                  P   P   P   P   P   P   F   P   .   P   F   F   F    P   F   FAIL
E2_tidal_strong           P   P   P   P   P   P   F   P   .   P   F   F   F    F   F   FAIL
X0_newton    (control)    P   P   P   P   P   P   P   P   .   P   P   P   P    P   P   PASS
X1_wells_linear           P   P   P   P   P   P   F   P   .   P   F   P   F    P   F   FAIL
X2_count_wells (control)  P   P   P   P   P   P   F   P   .   P   F   F   F    F   F   FAIL
X3_pairs_linear           P   P   P   P   P   F   F   P   .   P   F   P   F    F   F   FAIL
X4_smooth_density(ctrl)   P   P   P   P   P   P   F   P   .   P   F   P   F    P   P   FAIL
```

S8 is informational, not a gate: Newton itself is "bounded", which is the
point of the programme. `X4_smooth_density` is a control built only to exercise
the *other* branch of the coherence classifier; it is the only candidate other
than Newton that passes S13, and it fails S6 and S10 for the same reasons every
inhomogeneous-K law does.

### S1 dimensional consistency (executed, not asserted)

Each candidate's kernel is run twice from the same source: once with numpy and
once with `dimx`, where every intermediate carries an (M, L, T) exponent vector
and `exp` raises the moment a dimensionful argument reaches it. All fifteen
pass. **The negative control fires for all five kinds**: corrupting one
parameter's declared dimension produces, for example,
`DimensionError: cannot add [L^2] and [1]` from `1 + (r/L)^q`, and
`DimensionError: exp received a dimensionful argument [L^2]` from the
exponential weight form. The field-equation balance `div[K grad Phi]` versus
`4 pi G rho` is checked as `[T^-2]` on both sides.

### S2, S2b, S3, S9 covariance and permutation invariance

Maxima over all nineteen candidates:

| screen | worst value | tolerance |
|---|---|---|
| S2 rotation, `K(Rx; Rw)` vs `R K(x; w) R^T`, random SO(3) | 6.16e-15 | 1e-10 |
| S2b 90-degree lattice rotation of the *solved* field | 4.43e-14 | 1e-6 |
| S3 translation, `K(x+t; w+t)` vs `K(x; w)` | 2.60e-15 | 1e-10 |
| S9 permutation of catalogue rows (random, reversal, sort-by-mass) | 7.89e-15 | 1e-12 |

Source-label permutation invariance therefore holds to floating-point
summation order for every family. It is necessary and almost entirely
uninformative: any symmetric sum over rows passes it trivially while remaining
completely catalogue-dependent, which is what the coarse-graining screens are
for. The 90-degree rotation is chosen because it is an exact symmetry of the
cubic lattice, so it tests the discretisation as well as the law; an arbitrary
angle would be dominated by interpolation error.

### S4 potential-gauge invariance

A galaxy placed 1 Mpc from a 1e14 Msun cluster acquires
`dPhi = -4.302e11 m2/s2` with a local tidal acceleration of only
`1.39e-11 m/s2 = 0.12 a0`. Family B reads `|Phi|`, which is not a local
observable:

| candidate | `|dPhi| / Phi_0` | change in predicted v_c |
|---|---|---|
| B1_depth_mond | 43 | **1.173 (117%)** |
| B2_depth_mond_weak | 0.43 | 0.0399 (4.0%) |

Every other family uses only derivatives of Phi and is unaffected.

### S5 positive-definiteness

The `exp[.]` construction is symmetric positive definite by design and is
measured to be so: asymmetry `|K - K^T| / |K|` at most 1.6e-16, eigenvalues in
[0.849, 1.387] for family C. The screen has teeth because it also locates the
critical point of the parameterisations that break:

- S has measured eigenvalues in [-0.3273, +0.6541], at the analytic bounds
  [-1/3, +2/3]. The **linearised** form `K = I + s_T S` therefore loses
  positive-definiteness at `s_T = 3.055` or `s_T = -1.529` (computed, not
  assumed). At the control's `s_T = 2` it is still definite (lambda_min 0.345).
- For pairs, `C_max = 3.418`, so `K = I - alpha C` fails at
  `alpha_crit = 0.2925`. The control `X3_pairs_linear` at `alpha = 3` has
  **lambda_min = -9.255**: indefinite, and caught.
- **D2 and D3 fail the screen on their own account**: lambda_min = 1.08e-9 and
  6.52e-9, below the 1e-8 floor, already at a 100-row partition.
- AQUAL is strictly positive but **not uniformly elliptic**: `mu -> X` as
  `X -> 0`, so the ellipticity constant vanishes with the acceleration
  (measured min 1e-8 over `g_N/a0` in [1e-8, 1e8]). That is a conditioning
  statement, not a well-posedness failure, and is reported separately rather
  than charged as a failure.

### S6 Newtonian high-acceleration limit

For the scalar families, `|g/g_N - 1|` against `g_N/a0`:

| candidate | 1e2 | 1e3 | 1e4 | 1e6 |
|---|---|---|---|---|
| A1 AQUAL (mu simple) | 9.90e-3 | 9.99e-4 | **1.00e-4** | 1.00e-6 |
| A2 QUMOND (nu simple) | 9.90e-3 | 9.99e-4 | **1.00e-4** | 1.00e-6 |
| A3 QUMOND (RAR nu) | 4.54e-5 | 1.8e-14 | **0.0** | 0.0 |
| B1 depth-MOND, `|Phi| = (200 km/s)^2` | 4.77e-2 | 4.98e-3 | **5.00e-4** | 5.00e-6 |
| B2 depth-MOND, deeper Phi_0 | 1.01e-2 | 1.02e-3 | **1.02e-4** | 1.02e-6 |

The simple-mu tail is exactly `a0/g_N`, as it must be. All pass at the stated
tolerance of 1e-3 at `g_N/a0 = 1e4`.

For the tensor families the requirement is `K -> I` where `g_N/a0` is large.
None of them carries any `g_N` dependence at all (family C's third form
cancels; see result 3 above), so the anisotropy at solar-system accelerations
is the same object as the anisotropy in the outskirts. At probe points reaching
`g_N/a0 = 2.11e7`:

| candidate | max `|K - I|` | anisotropy `k_max/k_min - 1` | max `|K - I|` anywhere |
|---|---|---|---|
| C1, C4 | 0.175 | 0.511 | 0.382 |
| C2 | 0.172 | 0.508 | 0.383 |
| C3 | 0.174 | 0.506 | 0.382 |
| C5 | 0.181 | 0.518 | 0.381 |
| D1 | 0.628 | 0.255 | 0.640 |
| D2 | 1.000 | 115.9 | 1.000 |
| D3 | 1.000 | 336.8 | 1.000 |
| E1 | 0.441 | 1.027 | 0.476 |
| E2 | 2.081 | 7.326 | 2.242 |
| X0 Newton | 0.0 | 0.0 | 0.0 |

A 51% anisotropy in the gravitational response at solar-system accelerations
is excluded by any of the classical tests, and the screen reaches that verdict
without touching a measurement.

### S7 asymptotics, and the discontinuity at a catalogue row

`d ln g / d ln r` at `r ~ 1e5 r_s` is **-1.00001** for A and B (flat rotation
curve) and **-2.0000** for every C, D and E candidate and for all controls.
The inner slope is -2.000 everywhere except family E (-1.822 for E1, -1.409
for E2), and no field is non-finite anywhere.

`n_a n_a^T` is *even* in `n_a`, so approaching a catalogue row from `+e` and
`-e` gives the same limit; the genuine discontinuity is between different
*lines* of approach, and it survives `eps -> 0`. Measured with the softening
pushed three decades below the probe offset, as the spread of K over 24
directions at `eps = 5.1e-3 kpc`:

| candidate | 100-row partition | a single isolated row |
|---|---|---|
| C1, C4 | 0.0098 | **0.3885** |
| C2 / C3 / C5 | 0.0110 / 0.0132 / 0.0071 | 0.3885 |
| D1 / D2 / D3 | 3.7e-5 / 6.3e-4 / 3.2e-4 | 0 (one row makes no pair) |
| E1 / E2 | **0.4486** / **0.8238** | 0.4487 / 0.8238 |
| X0, X2, X4 | 0 | 0 |

The response tensor of families C and E has no value *at* a catalogue point,
only a direction-dependent limit. For C the jump scales with that row's
fractional weight, so it is 39% for an isolated object and about 1% when the
row is one of a hundred. For E it does not shrink with the number of rows at
all, because the tidal tensor is a `1/r^3` singularity always dominated by the
nearest row. Only an arbitrary softening length, which is not part of the
specification, makes either finite.

### S8 asymptotic gain (informational)

A flat rotation curve at `v_c` needs a gain `g/g_N = v_c^2 r / (G M)` growing
linearly with r. What each law can supply, for M = 5e10 Msun:

| candidate | gain at 20 kpc | gain at 20 Mpc | needed at 20 Mpc for 200 km/s | unbounded? |
|---|---|---|---|---|
| A1 / A2 | 3.17 | 2625 | 3719 | yes |
| A3 | 3.16 | 2625 | 3719 | yes |
| B1 | 4.31 | 2626 | 3719 | yes |
| C1, C2, C4, C5 | 0.736 | **0.717** | 3719 | no |
| C3 | 0.736 | 1.000 | 3719 | no |
| D1, D2, D3 | 1.000 | **1.000** | 3719 | no |
| E1 / E2 | 1.074 / 1.239 | **1.000** | 3719 | no |
| X0 Newton | 1.000 | 1.000 | 3719 | no |

MOND's apparent "shortfall" of 1.417 is exactly `(200/168.3)^2`: for this
baryonic mass its own Tully-Fisher speed is `(G M a0)^(1/4) = 168 km/s`, so it
delivers a flat curve at its own prediction and the ratio is a statement about
the 200 km/s I asked for, not a failure. Families C, D and E fall short by
factors of 3719 to 5191 at 20 Mpc, and the shortfall grows linearly with radius
without bound. Note that family C with `s_T > 0` supplies a gain **below** one:
it makes gravity 28% weaker than Newton, not stronger.

### S10 reciprocity

Two unequal masses (M1 = 4 M2) at 25 kpc, net force
`|integral rho (-grad Psi) dV|` normalised by `G M1 M2 / d^2`, with the
Newtonian K = I run on the identical grid as the null (0.00204 at n = 40,
falling to 0.00191 at n = 64).

| candidate | law | excess over null | grad-K term | surface term | identity agreement | variational? |
|---|---|---|---|---|---|---|
| X0 Newton | 0.00204 | **0** | - | - | - | yes |
| A1 AQUAL | 0.0232 | 0.0211 | - | - | - | declared |
| A2 / A3 QUMOND | 0.0130 / 0.0135 | 0.0110 / 0.0115 | - | - | - | declared |
| B1 depth-MOND | 0.690 | **0.688** | - | - | - | **no** |
| B2 | 0.0197 | 0.0176 | - | - | - | no |
| C1, C4 | 0.559 | **0.557** | 0.535 | 1.07e-4 | 4.3% | no |
| C2 / C3 / C5 | 0.607 / 0.588 / 0.542 | 0.605 / 0.586 / 0.540 | 0.563 / 0.560 / 0.539 | 0.011 / 0.004 / 0.015 | 5.5 / 4.0 / 3.3% | no |
| D1, D2, D3 (default alpha) | 0.0007 - 0.0018 | **0** (below the null) | 1.5e-3 | 2.0e-3 | - | no |
| E1 / E2 | 0.199 / 0.401 | **0.197 / 0.399** | 0.214 / 0.548 | 0.002 / 0.003 | 8.9 / 37% | no |
| X1 linear wells | 1.378 | 1.376 | 1.254 | 0.006 | 9.4% | no |

Family A's 1-2% excess is numerical: AQUAL and QUMOND are variational, and the
nonlinear or two-stage solve carries more discretisation error than the linear
Newton run on the same grid. Families C and E give 20-60%, and the grad-K
identity accounts for it: the volume term alone supplies 0.535 of the 0.559,
the surface term is 1e-4, and the residual gap falls as `h^2.03`.

Family D's pass at its default `alpha = 0.3` is a statement about coupling
strength, not structure: there `|K - I|` is only a few parts in a thousand.
Sweeping alpha shows the violation appearing exactly as expected:

| alpha | 0.3 | 3 | 30 | 100 | 300 |
|---|---|---|---|---|---|
| net-force excess | 0 | 0.0066 | 0.101 | 0.310 | 0.682 |

**No candidate in families C, D or E declares a momentum carrier.** As
specified, K is an externally prescribed function of the source positions and
is not a dynamical field, so there is nothing to absorb the momentum the
matter loses. The identity gives the exact size of what a carrier would have to
transport: `(1/8 pi G) integral (grad K) : grad Psi grad Psi`.

## 5. Stage 1b: coarse graining

The test the brief asks for, run exactly as specified. One galaxy of mass
M = 5e10 Msun (Plummer, a = 3 kpc). The same smooth `rho` is the source in
every run; only the number of catalogue rows describing it changes.

### 5.1 The deliverable numbers: N = 1 versus N = 10^4

Full 3-D solve on a 48^3 grid, 90 kpc box, `rho` identical in all runs.
`dPhi` is `max|Phi_1 - Phi_Nmax| / max|Phi|` over 2 kpc < r < 0.35 L; `dv_c` is
the largest fractional change in the circular speed over 3-30 kpc.

| candidate | N range | dPhi (max) | dPhi (rms) | **dv_c** |
|---|---|---|---|---|
| C1_wells_pow_p1 | 1 -> 10^4 | 10.06% | 3.50% | **9.11%** |
| C2_wells_pow_p05 | 1 -> 10^4 | 10.06% | 3.50% | **9.11%** |
| C3_wells_exp_p1 | 1 -> 10^4 | 10.11% | 3.53% | **9.12%** |
| C4_wells_gsupp_p1 | 1 -> 10^4 | 10.06% | 3.50% | **9.11%** |
| C5_wells_pow_p2 | 1 -> 10^4 | 10.06% | 3.49% | **9.10%** |
| D1_pairs_p1_q1 | 1 -> 256 | 21.80% | 5.36% | **30.12%** |
| D2_pairs_p05_q1 | 1 -> 10 | 25.88% | 6.41% | **29.10%** |
| D3_pairs_p1_q3 | 1 -> 10 | 44.03% | 12.96% | **43.62%** |
| E1_tidal | 1 -> 10^4 | 9.70% | 2.53% | **10.80%** |
| E2_tidal_strong | 1 -> 10^4 | 37.61% | 13.44% | **36.12%** |
| A1, A2, A3, B1, B2 | - | 0 | 0 | **0** (no row list enters) |
| X0_newton (control) | 1 -> 10^4 | 0 | 0 | **0** |
| X2_count_wells (control) | 1 -> 10^4 | 118.9% | 15.8% | **379.2%** |

For D2 and D3 only N = 1 and N = 10 are solvable: at N >= 100 the response
tensor is degenerate (condition number above 1e9) and the screen refuses the
solve rather than reporting 8,000 useless conjugate-gradient iterations as a
convergence failure. The refusal is recorded per N in the results file.

The C1 series in full, showing that this is a genuine convergence and not a
plateau:

| N | 1 | 10 | 100 | 10^4 |
|---|---|---|---|---|
| dPhi vs N = 10^4 | 10.06% | 1.80% | 0.72% | 0 |
| dv_c vs N = 10^4 | 9.11% | 2.68% | 0.60% | 0 |
| v_c at 3 kpc (km/s) | 127.11 | 143.59 | 140.46 | 139.84 |

A 12 km/s swing in the predicted rotation speed of the same galaxy, produced
by nothing but the number of rows used to tabulate its mass.

### 5.2 Rate of convergence

Drift of K at 200 probe points against the full cloud as reference, with the
reference-free successive-step series alongside. The partition scale is the
mean nearest-neighbour distance between rows.

C1 (`L = 10 kpc`):

| N | 1 | 2 | 4 | 10 | 40 | 100 | 400 | 1000 | 4000 | 10^4 |
|---|---|---|---|---|---|---|---|---|---|---|
| partition scale (kpc) | - | 5.08 | 5.06 | 4.11 | 2.62 | 2.02 | 1.50 | 1.20 | 0.81 | 0.58 |
| drift vs cloud | 0.280 | 0.284 | 0.139 | 0.074 | 0.036 | 0.018 | 0.0086 | 0.0052 | 0.0015 | 0.0005 |
| step vs previous | - | 0.375 | 0.246 | 0.130 | 0.058 | 0.028 | 0.017 | 0.0075 | 0.0051 | 0.0012 |

Fitted `drift ~ N^-beta`: **beta = 0.77**, and on the reference-free steps
**beta_step = 0.64**. The prediction for a one-point quadrature of a continuum
integral is 2/3, since a cell of linear size `l` contributes an error
`O(l^2) = O(N^(-2/3))`. Rates for all five C variants: beta = 0.68 to 0.77,
beta_step = 0.45 to 0.64.

**The mass exponent cancels exactly under uniform equal-mass refinement.** C1
(p = 1), C2 (p = 0.5) and C5 (p = 2) give drift 0.28013 to five significant
figures and identical dPhi and dv_c. This is not a coincidence: with equal-mass
cells the factor `(M/N M_0)^p` is common to numerator and denominator of S and
divides out. Uniform refinement therefore cannot see the exponent at all,
which is why the selective test below is the one with teeth.

`N_safe`, the row count at which the drift first falls below 1e-3 and stays
there: **10^4 for every family-C variant**, 1000 for D1. In operational terms,
family C needs a resolved mass map with of order ten thousand cells per galaxy;
a catalogue row per galaxy is wrong at the 10% level.

### 5.3 Selective refinement: the test with teeth

Two equal 6e10 Msun objects 40 kpc apart. Object 1 is split into N rows;
object 2 is never split. K is read 8 kpc from object 2, where nothing physical
has changed.

The analytic prediction is that the relative weight of the refined object
against the unrefined one moves as `N^(1-p)`, because splitting M into N
pieces changes that object's total weight from `M^p` to `N^(1-p) M^p`.
Measured directly:

| p | 0.25 | 0.5 | 0.75 | 1.0 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| measured slope `d ln(W1/W2) / d ln N` | 0.7507 | 0.5007 | 0.2507 | 0.00067 | -0.4993 | -0.9994 |
| predicted `1 - p` | 0.75 | 0.50 | 0.25 | 0 | -0.50 | -1.00 |

Agreement to about 7e-4 across the whole range. The drift in K itself between
a one-row and a 4096-row description of object 1:

| candidate | p | drift | verdict |
|---|---|---|---|
| C1 | 1 | 2.3% | passes: genuine quadrature convergence |
| C3 | 1 | 2.4% | passes |
| C2 | 0.5 | **31.3%** | fails: weight slope 0.50 |
| C5 | 2 | **6.9%** | fails: weight slope -1.00 |

**Only p = 1 is admissible.** At any other exponent the field near an untouched
object depends on how finely a *different* object happens to be tabulated, and
the dependence has no limit.

### 5.4 Physical coherence length, or catalogue rows?

Convergence alone does not settle this: a law with a genuine coherence length
and a law that merely has a continuum limit both converge. What separates them
is *what sets* the catalogue resolution you need. A law that is a quadrature of
a kernel of width L has error `O((l/L)^2)`, so raising L buys accuracy at fixed
row count. A law whose structure is set by the distance to the nearest row does
not respond to L at all.

Measured `d ln(drift) / d ln L` at fixed N, sweeping L over a factor of 16:

| candidate | swept parameter | slope | drift at N = 100, L = 2 -> 32 kpc |
|---|---|---|---|
| X4_smooth_density (control) | L | **-3.11** | 0.517 -> 3.9e-5 |
| C3_wells_exp_p1 | L | -1.11 | 0.168 -> 0.0146 |
| C1_wells_pow_p1 | L | **-0.55** | 0.0389 -> 0.0139 |
| C5_wells_pow_p2 | L | -0.48 | 0.0561 -> 0.0198 |
| D1_pairs_p1_q1 | sigma_perp | -0.045 | flat |
| X2_count_wells (control) | L | **+0.12** | 0.671 -> 0.999 |
| E1_tidal | none exists | - | 0.477 at N=100, 0.513 at N=1000 |

`X4_smooth_density` is a control I added for exactly this purpose:
`K = exp[s rho_L / rho_0] I` with `rho_L` the density smoothed on a fixed
length L. It has a real physical scale, and a sixteenfold increase in L reduces
the drift by a factor of 13,000. Family C's power-law form reduces it by 2.8.
The row-counting control does not respond at all.

The reason is structural: family C's directional factor `n_a n_a^T - I/3`
turns over when the field point moves by of order its distance to well `a`,
and that distance is set by the partition, not by L. The weight profile has a
scale; the direction does not.

Four-way classification of every candidate, from the drift and step series:

| classification | meaning | candidates |
|---|---|---|
| partition-independent | the row list never enters | X0_newton |
| **coherence-limited** (physical) | drift stops once the partition is finer than L/10 and stays stopped | **X4_smooth_density** |
| convergent-quadrature | has a continuum limit, but at any finite catalogue resolution the answer is set by how finely the mass is tabulated | C1-C5, D1, D3, X1, X3 |
| **catalogue-artefactual** | successive refinements do not shrink, or grow | **D2, E1, E2, X2_count_wells** |

Both controls land where they were designed to, so the classifier is two-sided
and not merely a rejection machine. `X2_count_wells` is decisive: its step
series *grows* from 4e-4 to 6.44 as N goes 2 to 10^4 (beta_step = -1.22) and
its predicted v_c changes by 379%.

Family E deserves a separate note. Its drift sits at 0.39 to 0.56 across four
decades of N and falls by a factor of only 1.26 in total. The tidal tensor of a
set of point masses is a `1/r^3` singularity at every row, so refining the
catalogue never stops mattering. **This is a statement about E sourced from a
row list. If `T` is instead obtained from the smooth density by solving
Poisson, family E's coarse-graining behaviour is exact by construction**, and
that distinction is the single most important thing to know about the family.

## 6. Stage 2: synthetic geometries

Every geometry is also run with K = I on the identical grid; that Newtonian
null is reported next to every number and is what separates a property of the
law from a property of the discretisation.

| candidate | G1 point | G2 two-body | G3 disk | G4 sphere | G5 disk+external | G6/G7 cluster | verdict |
|---|---|---|---|---|---|---|---|
| X0_newton | P | P | P | P | P | P | **PASS** |
| X2_count_wells | P | P | P | P | P | P (see note) | PASS |
| A1_aqual_simple | P | P | P | P | P | P | **PASS** |
| A2_qumond_simple | P | P | P | P | P | P | **PASS** |
| B1_depth_mond | P | P | P | P | P | P | **PASS** |
| C1_wells_pow_p1 | P | P | P | P | P | **F** | FAIL |
| C2_wells_pow_p05 | P | P | P | P | P | **F** | FAIL |
| C4_wells_gsupp_p1 | P | P | P | P | P | **F** | FAIL |
| D1_pairs_p1_q1 | P | P | P | P | P | P (see note) | PASS |
| D2_pairs_p05_q1 | P | P | P | P | P | P (see note) | PASS |
| E1_tidal | P | P | P | P | P | **F** | FAIL |

No candidate was rejected as singular, unstable or non-convergent in Stage 2.
K condition numbers stayed at 1.0 (D), 1.65 (C) and 1.84 (E1), far below the
1e6 gate; every conjugate-gradient solve reached its tolerance; the
grid-refinement excess over the Newtonian null was 0 for all but E1 (0.012)
and X2 (4e-5), against a 5% gate. The only Stage-2 rejections are
**representation-dependence**.

### G2, two equal masses: is there a spurious midpoint force?

**No. Not for any family.** `|g(midpoint)| / (G M / (d/2)^2)` sits at 3.5e-15
to 1.5e-12 for every candidate at every separation, and the excess over the
Newtonian null is at most 1.4e-12 against a 1e-3 tolerance. The reflection
symmetry of the two-equal-mass configuration is respected exactly, because
`n n^T` and the pair tube are both even under the reflection. A law can respect
the midpoint exactly and still misbehave off centre, so the largest on-axis
departure from Newton anywhere inside the pair is reported too, in the same
units:

| candidate | d = 8 kpc | 16 | 32 | 64 |
|---|---|---|---|---|
| A1 AQUAL | 0.134 | 0.802 | 2.74 | 7.11 |
| A2 QUMOND | 0.139 | 0.813 | 2.79 | 7.22 |
| B1 depth-MOND | 0.836 | 3.85 | 8.15 | 13.85 |
| C1 / C2 / C4 | 0.210 | 0.814 | 1.25 | 1.30 |
| D1 | 0.988 | 0.128 | **5.2e-6** | **4.2e-15** |
| D2 | 0.268 | 0.032 | 1.3e-6 | 4.2e-15 |
| E1 | 0.501 | 1.43 | 1.66 | 0.710 |
| X0 Newton | 0 | 0 | 0 | 0 |

A and B keep growing with separation, as a MOND-like law must. Family C
saturates, the bounded-K signature again. **Family D switches off entirely
beyond about 32 kpc**: its Gaussian tube weights give the pair channel a hard
range limit, so at wide separations it produces literally no deviation from
Newton, to 4e-15.

### G1, G3, G4: point mass, exponential disk, uniform sphere

Boost `v_c(law) / v_c(Newton)` on the identical grid:

| candidate | point mass, 8 -> 35 kpc | exp disk, 4 -> 30 kpc | uniform sphere, 2 -> 30 kpc |
|---|---|---|---|
| A1 AQUAL | 1.432 -> 2.259 | 1.309 -> 2.078 | 1.542 -> 2.111 |
| A2 QUMOND | 1.434 -> 2.259 | 1.311 -> 2.079 | 1.535 -> 2.110 |
| B1 depth-MOND | 1.804 -> 2.532 | 1.706 -> 2.355 | 2.098 -> 2.399 |
| C1 / C2 / C4 | **0.851 -> 0.847** | 0.960 -> 0.877 | 0.963 -> 0.854 |
| D1 | **1.000 flat** | 1.103 -> 0.999 | 1.196 -> 1.000 |
| D2 | 1.000 flat | 2.403 -> 0.991 | 5.770 -> 1.003 |
| E1 | 1.190 -> 1.006 | 1.046 -> 1.015 | 1.075 -> 1.009 |
| X0 Newton | 1.000 | 1.000 | 1.000 |

Family C makes gravity 15% *weaker* than Newton and essentially flat in radius.
Family D gives exactly 1.000 on a point mass, because a single row makes no
pair and `C = 0` identically: the law has no effect whatsoever on an isolated
object, of any mass, at any radius.

Two informational validations: the Newtonian null differs from the exact
razor-thin Freeman disk by 6.54% at R > 3 R_d (the model disk has finite
thickness h_z = 1.5 kpc, which lowers v_c on its own) and from the exact
uniform sphere by 6.58% (staircase edge on a Cartesian mesh). These are
properties of the null and are identical across candidates, which is why the
law is judged on the boost against that same-grid null, where geometry and
grid errors cancel to first order.

### G5, disk plus an external mass

A 5e11 Msun mass at 70 kpc from a disk, giving `g_ext = 0.12 a0`. Newton
itself shifts the inner rotation curve (the external mass is inside the box and
drags the whole potential), so the reported quantity is the *excess* over that
null:

| candidate | excess shift in inner v_c |
|---|---|
| A1 / A2 (MOND external field effect) | 7.9% / 7.9% |
| B1 | 1.8% |
| C1, C4 | 6.2% |
| C2 (p = 0.5) | **25.8%** |
| D1 / D2 | 0.05% / 2.5% |
| E1 | -2.6% |
| X0 Newton | **0** |

Family A's 7.9% is the genuine MOND external field effect and is expected.
Family C2's 25.8% is not an external field effect: it is the `N^(1-p)` weight
asymmetry appearing again, because the external mass enters the row list as a
single row of very large mass while the disk enters as 256 rows.

### G6/G7, a cluster progressively subdivided

The same 1e14 Msun cluster (Plummer, a = 400 kpc), source density identical in
every run, described by N rows:

| candidate | dv_c between N = 1 and N = max | implied M_dyn at 1 Mpc, N = 1 -> max | verdict |
|---|---|---|---|
| X0_newton | **0** | 8.568e13 -> 8.568e13 | pass |
| A1 / A2 | **0** | 3.46e14 unchanged | pass |
| B1 | **0** | 1.913e15 unchanged | pass |
| C1, C4 | **12.23%** | **6.163e13 -> 7.040e13** | **fail** |
| C2 | **12.24%** | 6.163e13 -> 7.034e13 | **fail** |
| E1 | **4.51%** | 8.575e13 -> 8.571e13 | **fail** |
| D1, D2 | 3.1e-15 | unchanged | pass (see note) |
| X2_count_wells | 2.6e-5 | unchanged | pass (see note) |

**Family C's inferred dynamical mass for one and the same cluster moves by 14%
depending only on whether the cluster is written down as one row or ten
thousand.** That is the rejection: representation-dependent.

Two honest caveats on the "pass" entries. Family D's tube weight
`exp[-(d/L)^s]` with L = 10 kpc is negligible at the 400 kpc separations of a
cluster, and the row-counting control's 8 kpc counting radius is 50 times below
the cluster scale, so **this geometry does not exercise either of them**; both
are exercised, and both fail, in the Stage-1b galaxy-scale test. Family D is
also capped at 256 rows here because its cost is `O(P N^2)` with no locality.

## 7. Funnel throughput

Screening fifteen hand-written specs is a demonstration, not a funnel. Most of
the decisive Stage-1 conditions turn out to be closed form once the eigenvalues
of S (or C) are known at three probe sets, and those eigenvalues do not depend
on the couplings at all:

    K = exp[s_0 I + s_T S]  =>  lambda_i(K) = exp(s_0 + s_T lambda_i(S))

so `|K - I|_2 = max_i |exp(s_0 + s_T lambda_i) - 1|` is monotone in lambda and
attained at `lambda_min` or `lambda_max`. **Six numbers per weight setting
decide the entire coupling plane exactly.** That factorisation is what makes
the front cheap.

| | family C | family D |
|---|---|---|
| weight-parameter settings (p, q, s, L, shape / sigmas) | 3,600 | 3,240 |
| coupling grid (s_0, s_T) or alpha | 201 x 201 | 4,001 |
| **total settings screened** | **145,443,600** | **12,963,240** |
| pass S6 Newtonian limit | 3,600 | 1,581,342 |
| pass S10 reciprocity | 723,600 | 141,387 |
| pass S12 / S11 coarse graining | 18,180,450 | 972,243 |
| **pass all three** | **450** | **0** |
| wall clock | 28.9 s | 48.3 s |
| rate | 5.03e6 settings/s | 2.68e5 settings/s |

Overall: **158,406,840 settings in 77.2 s = 2.05 million per second**, so the
brief's 1e9 front stage is about eight minutes of wall clock.

The counts are exactly interpretable, which is the point of doing it in closed
form rather than by sampling:

- S6 passes 3,600 = exactly one coupling point per weight setting, namely the
  grid point `s_0 = s_T = 0`.
- S10 passes 723,600 = 3,600 x 201 = the whole line `s_T = 0`, any `s_0`. That
  is precisely the analytic statement: momentum is conserved if and only if K
  is homogeneous.
- S12 passes exactly one eighth of the grid: the `p = 1` slice.
- All three together leave 450 = the 450 weight settings with p = 1, each
  contributing the single point `s_0 = s_T = 0`.

So the 450 "survivors" are Newton, counted 450 times. At the sampled coupling
resolution the surviving fraction is 3.09e-6; the continuum measure of the
surviving set is about 3e-8 and shrinks to zero as the Newtonian-limit
tolerance is tightened. Family D has no survivors at any resolution because
`p = 1 and q < 3` and `alpha` small enough to satisfy S6 and S10 are mutually
exclusive on this grid.

The sweep also confirms the analytic bound directly:
`max |lambda(S)| = 0.666666666666` over all 3,600 weight settings, against the
bound `|S|_2 < 2/3` that follows from `|n n^T - I/3|_2 = 2/3`.

## 8. Per-family verdict, and what would fix it

A candidate is not dead because it fails somewhere. For each family: what it
passes, what it fails and by how much, and what modification would repair it.

### A. AQUAL and QUMOND

**Passes everything.** All fifteen Stage-1 screens, all seven Stage-2
geometries. Newtonian limit `|g/g_N - 1| = 1.00e-4` at `g_N/a0 = 1e4` (exactly
`a0/g_N`), asymptotic slope -1.00001, unbounded gain, exact reciprocity within
the numerical null, no row list anywhere so coarse graining is exact by
construction, midpoint force zero, cluster subdivision exactly zero. The one
caveat is that AQUAL is **not uniformly elliptic** (`mu -> X` as `X -> 0`), so
its operator is arbitrarily ill-conditioned in the deep-MOND limit. That is a
solver-conditioning fact, not a well-posedness failure.

Its role here is as the positive reference: it demonstrates that the screen
battery is passable, and it is the yardstick the well-network families are
measured against.

### B. Potential-depth MOND

**Fails S4 (gauge) and S10 (reciprocity).**

- A 1e14 Msun mass at 1 Mpc exerts a local tidal acceleration of only 0.12 a0
  but shifts the potential by 4.3e11 m2/s2, and B1's predicted rotation speed
  changes by **117%**. The law reads a quantity no local experiment can
  measure.
- Net-force excess **0.688** of the pair force, the largest of any family
  tested: `A_0(|Phi|)` makes `nu` strongly position-dependent and the law is
  not variational.

**What would fix it.** `|Phi|` must be replaced by something built from
derivatives, or by a difference against a declared reference. Options that are
genuinely local: the tidal invariant `sqrt(T:T) x L^2` (dimensionally a
potential), or `|grad Phi|^2 / a0` times a length. Any repair that keeps a bare
`|Phi|` is unusable, because `Phi` is defined only up to a constant and the
constant is set by matter arbitrarily far away. The variant with a much deeper
`Phi_0` (B2) reduces the gauge sensitivity to 4.0% and the reciprocity excess
to 0.018, but only by making the depth term nearly inoperative: it approaches
plain QUMOND, which already passes.

### C. Well-alignment tensor

**Fails S6, S10, S11b, S13, and G6/G7. Passes S12 only at p = 1.**

| failure | measured | by how much |
|---|---|---|
| Newtonian limit | `|K - I| = 0.175`, anisotropy 0.51, at `g_N/a0 = 2.1e7` | 175x the 1e-3 tolerance |
| reciprocity | net force 0.557 above the null | no momentum carrier declared |
| coarse graining, Phi and v_c | dPhi 10.06%, dv_c 9.11%, N = 1 -> 10^4 | 100x the 1e-3 tolerance |
| coherence classification | convergent-quadrature, not coherence-limited | needs N_safe = 10^4 rows per galaxy |
| selective refinement | weight slope `1 - p`, exact to 7e-4 | 31% drift at p = 0.5 |
| cluster subdivision | dv_c 12.2%, M_dyn ratio 0.875 | 122x the 1e-3 tolerance |
| asymptotic gain | 0.717 at 20 Mpc, needs 3719 | short by 5191x, and growing with r |
| discontinuity at a row | 0.389 for an isolated row | K has no value at a catalogue point |

**What would fix each of them.**

1. *Coarse graining*: `p = 1` is mandatory (the sweep shows p = 1 is the only
   admissible eighth of the exponent grid), and the sum over rows must be
   replaced by an explicit integral over the density field,

       S[rho](x) = int rho(y) f(|y-x|) (n n^T - I/3) d3y
                   / (eps M_0 + int rho(y) f(|y-x|) d3y)

   which is exactly the p = 1 continuum limit and is manifestly
   partition-independent. This is a reformulation, not a tuning: the law must
   be written as a functional of the density field and never as a sum over
   catalogue rows. It also removes the direction discontinuity, since the
   `1/r^2`-weighted directional integral against a smooth `rho` is continuous.
2. *Newtonian limit*: the acceleration screening must sit **outside** the
   normalisation, `s_T -> s_T / [1 + (g_N/a0)^m]`, not inside `w_a` where it
   cancels identically (result 3 above). This is a one-line change with a
   completely different physical content.
3. *Asymptotic gain*: `s_0` must be allowed to diverge as `g_N -> 0`, for
   example `s_0 = -ln nu(g_N/a0)`, which reproduces QUMOND as an isotropic
   backbone and leaves S as an anisotropic correction on top. Without an
   unbounded K the family cannot address the phenomenon it was invented for.
4. *Reciprocity*: the alignment field must be made dynamical, with its own
   kinetic term and a coupling to `rho`, so that the momentum matter loses is
   carried by the field; or a momentum carrier must be declared explicitly.
   The identity gives the exact size of what has to be carried.

A family C repaired on all four points is no longer family C: it is
"QUMOND plus an anisotropic correction sourced by a density functional". That
is a well-posed and interesting object, and it is the natural next candidate
class.

### D. Pair-channel tensor

**Fails S6, S11b, S12, S13; D2 and D3 also fail S5.**

- `||C|| ~ N^(2-2p)` (q < 3), log-divergent at q = 3, measured slopes 0.0102,
  1.0101, 0.1667 against predictions 0, 1, log. Only **p = 1 with q < 3** has
  a finite limit.
- At p = 1/2 the response tensor collapses exponentially in the row count:
  `lambda_min(K)` falls from 3.40e-1 to 8.30e-80 as N goes 10 to 800. There is
  no continuum limit, and at N >= 100 the field equation is not solvable to
  tolerance at all.
- Cost is `O(P N^2)` with no locality. A 1e6-row catalogue is
  **499,999,500,000 pairs** per field point. The screen refuses above 4e7 pairs
  and records the refusal rather than silently subsampling.
- On an isolated object the law has **no effect at all** (one row makes no
  pair, so `C = 0` and `K = I` exactly), and beyond about 32 kpc in the
  two-body test its deviation from Newton is 4e-15. Its whole content lives in
  a tube of fixed width around pairs at separations of order L.

**What would fix it.** `p = 1` and `q < 3` are mandatory, and the double sum
must become a double integral over `rho (x) rho`, i.e. an explicitly non-local
two-point functional of the density field. Even then it needs the same
acceleration screening and unbounded backbone as C, plus a declared momentum
carrier, and the cost problem remains unless the kernel is truncated in a way
that also has to be justified physically.

### E. Tidal-tensor gravity

**Fails S6, S10, S11 (reclassified catalogue-artefactual), S11b, S13, G6/G7.**

- `|K - I| = 0.441` with anisotropy 1.03 at `g_N/a0 = 2.1e7`.
- Net force excess 0.197 (E1) and 0.399 (E2) of the pair force.
- dPhi 9.7%, dv_c 10.8% between one row and 10^4 rows; the drift falls by a
  factor of only 1.26 across four decades of N.
- The discontinuity at a row is 0.449 and **does not shrink with the number of
  rows**, because the tidal tensor is a `1/r^3` singularity always dominated by
  the nearest row.
- For a spherical source `That = (I - 3 r_hat r_hat)/sqrt(6)` is constant in
  the vacuum, so E's entire effect outside an isolated spherical body is a
  renormalisation of G. Its measured boost decays to 1.006 by 35 kpc.

**What would fix it.** The single decisive change is to source `T` from the
smooth density by solving Poisson, not from catalogue rows. `T = grad grad Phi`
of a smooth field is a local functional of `rho`, so coarse-graining becomes
exact by construction, the row discontinuity disappears, and S11, S11b, S13 and
G6/G7 all pass automatically. What remains after that repair is S6 (needs the
`f_T` screening outside the normalisation) and S8 (needs an unbounded
backbone), i.e. the same two repairs as family C. **Family E is the closest of
the three to being salvageable, and the entire difference is whether `T` is
read off a catalogue or off a field.**

## 9. Controls, nulls, and the programme's known failure modes

Every failure mode listed in the standing brief was checked explicitly.

- **Sealed holdouts.** No observational file was opened. KiDS and the wide
  binaries were not loaded, listed or referenced.
- **Shared-denominator artefacts.** Checked. The coarse-graining reference is
  the full equal-mass cloud used as its own well list, not a member of the
  refinement series, and the reference-free successive-step series is reported
  alongside and is the primary criterion. Family C's own construction *does*
  put a shared denominator (`eps + sum|w_a|`) into every component of S, and
  that is exactly why the `g_N` factor in weight form 3 cancels; the effect is
  quantified rather than glossed.
- **Monotone-invariant statistics.** Checked. Every headline statistic was
  swept against the parameter it is supposed to measure:

| statistic | swept | range of the statistic | verdict |
|---|---|---|---|
| C anisotropy vs `s_T` | 0.02 to 2 | 0.0131 to 2.678 | responds |
| C selective weight slope vs `p` | 0.25 to 2 | 0.751 to -0.999 | responds |
| C reciprocity vs `s_T` | 0 to 1 | **0** to 0.913 | responds, and is exactly 0 at `s_T = 0` |
| C uniform coarse drift vs `L` | 2 to 100 kpc | 0.2791 to 0.2783 | responds by only 0.27% over a factor of 50 in L, which is itself the finding |
| D anisotropy vs `alpha` | 0.03 to 3 | 0.0997 to 1.000 | responds |
| D reciprocity vs `alpha` | 0.3 to 300 | 0 to 0.682 | responds |
| E anisotropy vs `f_T` | 0.05 to 2 | 0.0386 to 3.642 | responds |

- **Refitting on the held-out set.** Not applicable: nothing is fitted anywhere
  in Stage 1 or Stage 2. Every parameter is a declared global constant.
- **Silent extraction failures.** Row count, array shape, positivity and mass
  conservation are asserted after every partition (worst relative mass error
  below 1e-15). Every solve asserts its residual. Infeasible configurations
  raise and are recorded in an `infeasible` field per N rather than skipped.
- **Test bugs that look like solver bugs.** Open Dirichlet boundaries
  everywhere, never zero flux. For the anisotropic families the shell is the
  exact radially-aligned exterior solution, not the constant-K monopole.
  Box-size convergence **at fixed grid spacing** is 0.39% from L = 60 to
  120 kpc; the 21% first seen at fixed n was resolution, not the boundary.
  Every Stage-2 geometry is run with K = I on the identical grid and the null
  is reported next to the result.

Two bugs of exactly this kind were found and fixed during the work, and both
are worth recording because both initially looked like physics:

1. The well-discontinuity probe compared K at `+e` and `-e`. Since `n n^T` is
   *even* in `n`, that difference is identically zero and the measurement was
   reporting only the softening length; the real discontinuity is between
   different *lines* of approach. Corrected, and it changed the answer from
   5e-5 to 0.389.
2. The two-body midpoint test normalised an acceleration by a force. Every
   number came out around 1e-50 and looked like an exact null. Corrected to
   `G M / (d/2)^2`, the midpoint force is still zero to 1e-12, but the on-axis
   comparison became informative and revealed family D's hard range cutoff.

**Controls.** Four were built, and all four behave as designed:

| control | purpose | outcome |
|---|---|---|
| `X0_newton` (K = I) | negative control: must pass everything | passes all 15 Stage-1 screens and all 7 geometries; every null exactly 0 |
| `X2_count_wells` (K depends on the *number* of rows within L) | positive control for coarse graining | step series **grows** 4e-4 to 6.44, dv_c 379%, classified catalogue-artefactual |
| `X1`, `X3` (linearised `I + s_T S`, `I - alpha C`) | positive control for positive-definiteness | X3 has `lambda_min = -9.255`; the screen computes the critical couplings 3.055 and 0.2925 |
| `X4_smooth_density` | positive control for the *other* branch: a law with a genuine coherence length | classified coherence-limited, `d ln(drift)/d ln L = -3.11` |

Without X4 the coherence test would only ever have returned rejections and
there would be no evidence it can recognise a physical scale when it sees one.

## 10. What I could not establish

- **Whether a repaired candidate works.** Everything above is necessary
  conditions. Whether "QUMOND plus a density-functional anisotropy" fits
  anything is a Stage-3 question and needs data.
- **The momentum carrier.** The identity gives the exact magnitude of the
  momentum that families C, D and E fail to conserve, but I did not construct a
  field that carries it, and I cannot say whether a variational completion
  exists that preserves the alignment structure.
- **Family D beyond 4e7 pairs.** The `||C|| ~ N^(2-2p)` scaling is measured
  over three decades of N (4 to 4096) at probe points and up to 256 rows on a
  full grid. The extrapolation to catalogue scale is analytic, and verified
  over that range, but not directly computed.
- **Hierarchical partitions.** The coarse-graining tests refine one galaxy
  uniformly and two objects selectively. A genuinely hierarchical case (stars
  in a galaxy in a group in a cluster, with several distinct scales
  interacting) was not tested and could in principle behave differently from
  either.
- **Arbitrary-angle covariance of the solved field.** Rotation covariance was
  tested exactly for the construction (6e-15) and at 90 degrees on the lattice
  (4e-14). An arbitrary angle on a Cartesian grid would be dominated by
  interpolation error and could not separate the law from the discretisation.
- **The cluster geometry does not exercise family D or the counting control**,
  because their fixed length parameters (10 kpc and 8 kpc) are far below the
  400 kpc cluster scale. Both are exercised at galaxy scale in Stage 1b, and
  both fail there, but their Stage-2 cluster "pass" should not be read as
  information.
- **The reading of family D's tube.** I measure `d_par` from the pair midpoint
  with `sigma_par` a fixed global length, so the tube does not stretch with the
  pair separation. Another reading is defensible. The coarse-graining
  conclusion does not depend on it, since it follows from the mass scaling of
  `w_ab`, but the Stage-2 range cutoff does.

## 11. Reproduction

```
cd work/wellnet-2026-09/screen
python screen.py        # Stage 1 + 1b -> screen_results.json   (~25 min, RTX 5090)
python stage2.py        # Stage 2      -> stage2_results.json   (~35 min)
python finalise.py      # repairs, added controls, source hashes
python summarise.py     # console tables
```

`screen.run_screen(cand)` and `stage2.run_stage2(cand)` are importable and take
any candidate spec; `families.CANDIDATES` and `screen.CONTROLS` hold the
nineteen used here. `screen.sweep_C` / `sweep_D` run the exponent sweeps.
`screen_results.json` carries the SHA-256 of every source file including
`solver.py` and `axisym.py`, which were imported and not modified. No GPU
fallbacks were needed; every number was computed in float64 on the GPU.
