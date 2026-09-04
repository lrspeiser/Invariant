# The nonlocal kernel: stability of `F - rF'`, the clipping surface, and the repair

Frozen split `seed=20260903`; blind touched once. KiDS and wide binaries never
loaded, read or referenced.

## JOB 1 VERDICT: `F - rF'` IS stable; the 0.156 dex result survives the numerics

**`D = F - rF'` is converged at the previous lane's production settings.**
Against an all-fine reference (grid 5600 vs 1400; `n_D/n_s/n_gl` 96/32/14 vs
32/12/8; `dlnr_max` 0.08 vs 0.35; cubic vs log-linear `q` interpolation; limits
1e-4 -> 3e5 vs 1e-3 -> 3e4 kpc), production `D` differs over the 1028 SPARC train
points beyond 2 R_disk by **0.13% median, 0.97% p95, 5.6% worst point.**

One knob at a time, all median changes < 1%: radial resolution 0.11%,
interpolation rule 0.008%, `n_D` 0.03%, `n_s` 0.07%, `n_gl` 0.05%, `dlnr_max`
0.05%, inner/outer limits 0.08%/0.05%.

On the brief's own configuration — MW-like, smooth `q`, `rho_ref=1e5`, `F1`,
`alpha=3`, reproducing `max dlnF/dlnr = 0.8995` and margin `D/F = 0.0654` — every
quadrature knob moves `D` by at most **0.22%**. The 10:1 cancellation costs about
one decimal digit; float64 had fifteen to spare. **So 0.156 dex is not a
numerical artefact.** It fails for two other reasons.

**Method.** The previous lane obtained every `D` by differencing `F`, which
differences the same cancellation twice. `dcore.py` differentiates under the
integral sign instead: with `J = Int r' rho I dr'`, `I = Int_a^b F(qbar) dD`,

    D = (2 pi/M) Int r' rho [I - r dI/dr] dr'
    dI/dr = F(qbar|_b) - sgn(r-r') F(qbar|_a) + Int F_q(qbar)(dqbar/dr) dD
    dqbar/dr = Int q'(r_s) r(1-s)/r_s ds

The Newtonian limit drops out term by term.

| gate | result |
|---|---|
| `alpha=0`, `D = M(<r)/M_tot` — analytic exponential sphere | 1.3e-9 |
| same on the SPARC equivalent-sphere profile | 2.3e-4 |
| gauge identity `F -> F + c r` leaves `D` unchanged | 3.5e-13 |
| analytic vs the lane's spline route, 1028 points | median 1.0e-3, p95 1.0e-2 |

**Gauge fact for the record.** `F` is defined only up to an additive multiple of
`r` (`F -> F + cr` shifts `Phi` by the constant `-GMc`). **`F` is not an
observable; `D` is.** Any "F is bounded" claim must be converted to a claim about
`D` before it constrains dynamics.

**Amplification** (median dD/dF): `n_D` 135-3300, `n_s` 27-280, interpolation
rule 25-50, radial grid 1.5-8.4, everything else ~1. The two path-quadrature
parameters and the interpolation rule are where the cancellation bites, and are
exactly the ones never varied before. Still small enough in absolute terms.

### The second parameter set does NOT survive, and the failure is not numerical

The lane's "best set that also passes both local tests" — `screen rho_ref=1e6,
L_q=2, F1_poly, alpha=10, p=2`, reported at 0.196 dex — **produces `D <= 0`, i.e.
repulsive gravity, at 260 of 1028 SPARC train points (25.3%) in 36 of 71
galaxies**, and 72 of 309 blind points. Three independent routes agree: analytic
`D`, a batched five-point stencil, and the lane's own `spherical_vcirc_spline`
(identical count). The lane missed it because its SPARC screen tested
`F_eff <= 0` — the potential's sign — not `D <= 0`, the force's. Its 42%
repulsive figure was measured on the six-galaxy model ladder and never applied to
the SPARC configurations. **The 0.196 dex "solar-and-Oort-safe corner" is
withdrawn.**

### Where `D` comes closest to zero

Margin `m = D/F = 1 - dlnF/dlnr`; `m = 0` is reversal.

| | min | 5 pct | 50 pct | 95 pct | location of minimum |
|---|---|---|---|---|---|
| best-overall (`F3_pade a=10 p=1`) | **+0.0046** | +0.108 | +0.311 | +0.630 | r/R_disk 2.32 (2.02-4.69) |
| best-passing-local (`F1_poly a=10 p=2`) | **-0.572** | -0.450 | -0.005 | +0.361 | r/R_disk 3.42 |
| what the DATA require | +0.0505 | +0.165 | +0.437 | +0.714 | r/R_disk 2.14 |

**Reversal appears first at the inner edge of the fitted range, just outside
2 R_disk, in massive high-surface-brightness spirals.** Closest in train: NGC 5371
at 23.1 kpc = 3.11 R_disk, `m = +0.0046`; then NGC 3953, F583-1, UGC 11455,
NGC 5985.

### A numerical trap worth recording

`NK.spherical_vcirc` (per-radius panels) is **unusable for `D`**: median 0.64
fractional error, p95 5.2, and it falsely reports 210 repulsive points where
there are none. Each stencil point gets different panels, so the 1e-4 quadrature
error is *uncorrelated* and then multiplied by 8/(12 dlog) = 333.
`spherical_vcirc_spline` shares one node set via `_global_panels`, errors
correlate, accuracy 1e-3. The lane used the good route; the bad one sits beside
it in the same module.

## THE COMPARISON THAT MATTERS: the kernel underperforms the RAR by a factor two

"0.156 dex against Newton's 0.646" was a potential-space statistic beyond
2 R_disk set beside "the RAR sits at ~0.11 dex", an acceleration-space scatter on
all points. Different quantities, different point sets. Same galaxies, same
points, same nuisances (Ups*_disk = 0.5, Ups*_bulge = 0.7, catalogue distances
and inclinations, no per-galaxy freedom for anybody), same frozen split:

**Acceleration space, rms log10(g_pred/g_obs), R >= 2 R_disk**

| model | train (71 gal, 1028 pts) | validation (23, 296) | **blind (23, 309)** |
|---|---|---|---|
| AQUAL, simple mu | **0.121** | **0.098** | **0.121** |
| RAR (McGaugh+2016) | **0.121** | 0.098 | **0.122** |
| AQUAL, standard mu | 0.126 | 0.104 | 0.133 |
| nonlocal kernel, best-overall | 0.256 | 0.218 | **0.209** |
| Newton | 0.597 | 0.517 | 0.588 |
| kernel, best-passing-local | 0.734 (260 dropped) | 0.780 (86) | 0.666 (72) |

**Potential space** (the previous lane's currency): RAR/AQUAL-simple 0.113 train
/ 0.119-0.120 blind; kernel best-overall **0.156** train (reproduces the previous
lane exactly) / **0.135** blind; Newton 0.690 / 0.661.

**All radial points, acceleration space:** RAR 0.162 / 0.145, AQUAL simple 0.163
/ 0.144, kernel 0.288 / 0.225, Newton 0.527 / 0.539.

Fairness notes, all favouring the kernel where there was a choice: `a0` fitted on
TRAIN then frozen (RAR 1.202e-10, x1.002 of literature) because the kernel got a
396-point TRAIN grid search; AQUAL is exactly algebraic for the equivalent
spherical model so those are exact solutions; the RAR's published 0.11 dex uses
per-galaxy nuisance marginalisation, and held to the kernel's fixed nuisances it
scores 0.121/0.162 dex, which is the number to beat; points where the kernel
predicts `g <= 0` are **dropped**, which flatters it.

**The kernel's blind result, never previously quoted: 0.209 dex acceleration /
0.135 dex potential, against the RAR's 0.122 / 0.120 on identical points.** Blind
beats train for the kernel only because blind is an easier sample — Newton 0.588
vs 0.597 and RAR 0.122 vs 0.121 confirm it.

Monotone-invariance: `S(a0)` over x25 in `a0` runs 0.286, 0.197, 0.136, **0.121**,
0.142, 0.208, 0.327 — spread 0.206 dex, interior optimum. Not degenerate.

## JOB 2: the clipping smoothness audit

**S1 — the clip is C0 and the kink is large.**

| variant | class | max abs [q'] | max abs [q''] | sup abs q'' |
|---|---|---|---|---|
| `hard` | C0 | **1.000** | 0 | 0 (delta) |
| `quad`, w = 0.05 | C1 | 1.0e-8 | 10.0 | 10.0 |
| `quintic`, w = 0.05 | **C2** | 3.0e-16 | 6.0e-7 | 15.0 |
| `softplus`, w = 0.05 | C-inf | 7.1e-9 | 2.1e-7 | 10.0 |

The C2 replacement is `phi(u) = 2w(t^3 - t^4/2)`, `t = (u+w)/(2w)`, the unique
quintic with `g(0)=g'(0)=g''(0)=0`, `g(1)=1/2`, `g'(1)=1`, `g''(1)=0` (the `t^5`
coefficient is exactly zero); C2 and not C3.

**S2 — in the kernel formulation the surface adds NO force, NO flux, NO energy
discontinuity.** Two things had to be separated.

*(a) A quadrature pathology that mimics a jump.* `D` contains
`dqbar/dr = Int q'(r_s) r(1-s)/r_s ds`, and a hard clip makes `q'`
**discontinuous in s**. Gauss-Legendre on a discontinuous integrand converges at
first order:

| n_s | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|
| hard clip, rho_ref = 1e6 | 0.559 | 0.325 | 3.1e-2 | 1.8e-3 | 1.5e-3 | 3.8e-4 |
| C2 clip, w = 0.20 | 0.204 | 2.2e-3 | 5.3e-5 | 1.7e-4 | 1.9e-4 | 4.4e-5 |

**At the production `n_s = 12`, `D` near a clip surface carries a 20-50% error —
a defect of the clip, not the solver.** The C2 clip removes it, and this explains
why `n_s` is one of the two amplifying knobs in Job 1.

*(b) The physical question*, at `n_s = 512` with the window shrunk: the
extrapolated `[g]` falls 5.2e-2 -> 2.3e-2 -> 1.0e-2 as the window halves — a
straight line to zero, which is what a continuous function with a kink does under
a two-sided linear fit. **No shell: `[Phi] = 0`, `[g] = 0`.** What survives is a
step in `dg/dr`, i.e. in the inferred density, at 79-265x the local baryon
density for `p = 1`, and exactly zero for `p = 2`.

**S3 — same conclusion in a field formulation.** For `div[K(q) grad Psi] =
4 pi G rho` with `K = 1 + alpha q^p`: `rho_eff = rho/K - M K'/(4 pi r^2 K^2)`.
`q'` jumps so `rho_eff` jumps; no delta, so no surface layer. At `p = 0.5`
divergent; at `p = 1` the step is 3.75-377x the local baryon density; at `p = 2`
exactly zero. **`p >= 1` is a requirement, not a preference.**

**S5 — C2 and "exactly safe in the solar system" are COMPATIBLE.** `quad` and
`quintic` round the corner with **compact support**. With `rho_s(Sun) = 4e7` and
`rho_ref = 1e6`, `u(Sun) = -0.975`, so any `w <= 0.975` leaves **`q(Sun) = 0`
exactly** and `eps(1 AU) = 0` identically. **It is ANALYTICITY that costs the
exact zero, not smoothness** — `softplus` leaves `q(Sun) = w^2/(4|u_Sun|)`, which
forces `rho_ref` down.

**S4 — but the C2 replacement does NOT repair the physics.** Across 40
delta-type configurations no variant is free of repulsive points anywhere in the
grid; smoothing moves the rms by < 0.03 dex and the repulsive fraction by < 1
percentage point. The delta state runs from `q = 0` to `q = 1` over a **factor 2
in density**; rounding the corners of a step that narrow changes nothing.

## JOB 3(a): unbounded `F` is necessary, not sufficient; the family is not repairable

**The theorem, sharpened.** `D = -r^2 (F/r)'`, so `F/r = Int_r^inf D/r'^2 dr'`.
Flat needs `D -> (v_f^2/GM) r`, and that integral then diverges logarithmically.
**The correct condition is `|F|/r -> infinity`, not `F -> infinity`.**

| form | min `D` | dlnD/dlnr | dlnv/dlnr | verdict |
|---|---|---|---|---|
| `1 + 3r^0.5` | +2.5 | +0.497 | -0.252 | unbounded, still declining |
| `1 + 3r^0.9` | +1.3 | +0.899 | -0.050 | unbounded, still declining |
| `1 + 3r` | +1 | 0.000 | -0.500 | **pure gauge, `D = 1` exactly** |
| `1 + 3r^1.1` | **-7535** | +1.100 | +0.050 | `F/r -> inf` but gravity reverses |
| `1 + 3r ln(1e4/r)` | +4 | +1.000 | **0.000** | unique exactly-flat form |
| `F = 4` (bounded) | +4 | 0.000 | -0.500 | Keplerian |

**The search: 6720 settings.** Seven separation-dependent atoms x `a` x `p` x `n`
x `L` x `rho_ref` x `q`-type on a six-galaxy ladder.

| requirement | passing |
|---|---|
| R1 `D > 0` from 1 AU to 1 Mpc | 2139 / 6720 |
| R2 `dlnD/dlnr = 1 +- 0.2` over 2-20 R_disk | 102 / 6720 |
| R3 `abs(D(1 AU) - 1) < 1e-11` | 4570 / 6720 |
| R4 `F_local in [1.10, 1.70]` | 325 / 6720 |
| R5 BTFR slope in [3.5, 4.2] | 71 / 6720 |
| **R1 and R2** | **0 / 6720** |

**Pareto: the best flatness reachable with `D > 0` everywhere is
`dlnD/dlnr = 0.915` (flat needs 1.000), and the best `min D` among flat settings
is `-0.973` (needs > 0). The two frontiers do not meet.**

**Why: the gate subtracts from the force.** For `F = 1 + a G(qbar(r)) H(r)`,

    D = 1 + a G (H - r H') - a r G'(qbar) qbar'(r) H

and the last term is **strictly negative** whenever the modification grows in
voids (`G' > 0`) and paths get more void-like outwards (`qbar' > 0`). Raising `a`
to lengthen the flat part raises the negative term by the same factor:

| a | ungated (p=0) min D | gated p=1 | gated p=2 |
|---|---|---|---|
| 0.3 | +1.027 | +0.818 | +0.913 |
| 1 | +1.091 | +0.393 | +0.711 |
| 3 | +1.273 | **-0.823** | +0.133 |
| 10 | +1.909 | **-5.075** | **-1.890** |

**Why ungated is not an escape.** The exactly-flat ungated form has `D = 1 + cr`
with a **universal** `c`, so the fractional departure from Newton is `cr` at every
radius. An ISL bound of 1e-11 at 1 AU caps `c <= 2.06e-3 /kpc`, hence
`D - 1 <= 0.021` at 10 kpc, where SPARC train demands 1.90 / 4.64 / 9.55
(5/50/95 pct). **Short by a factor 225.** At a looser 1e-8 bound the cap becomes
`D - 1 <= 20.6` and this leg does not exclude — so it is bound-dependent and said
so. MOND evades the same problem with an *acceleration* gate whose solar
suppression is `1/x = 2e-8` (simple mu) or `1/(2x^2) = 2e-16` (standard mu).

**The second, independent no-go: linearity fixes the BTFR slope at 2.** `Phi[rho]`
is a **linear** functional of rho whenever `F` does not depend on rho, giving
`v_f^2 ~ M`, i.e. BTFR slope **2**. Verified: universal `c` in `D = cr` gives
2.000; `c ~ M^(-1/2)` (MOND's `c = sqrt(a0/GM)`) gives 4.000. The only
nonlinearity is `q`, and a **bounded** `q` saturates in the outskirts — exactly
where the flat part lives — restoring linearity there. **That is why the lane
measured 2.88 on and 2.07 off: 2 and 4 bracket it and 2.88 is the transition, not
a solution.** A repaired, unbounded `F` inherits this unchanged, because the
repair is to `F`'s r-dependence, not to the kernel's linearity in rho.

**Job 3(a) verdict: no.** Gated forms cannot be flat and attractive at once
(0 of 6720; Pareto gap 0.915 vs 1.000). Ungated forms can, but are excluded by
the solar system at 1e-11 (factor 225) or, if that is relaxed, by the BTFR at
slope 2 against 3.85.

## JOB 3(b): the nonlocal atom in the tensor grammar

`K = exp[f_nl(qbar) I + f_T(qbar) That]`. The path average is what makes the atom
nonlocal: a local `q` saturates the moment rho drops below `rho_ref`, while the
path average approaches its ceiling only as `1 - r_ref/r`. That `1/r` tail is the
entire content of the atom. Family E of the existing screen is the same
expression with **constant** `f0, fT`.

Exterior monopole: `T = (GM/r^3) diag(-2,1,1)` is already traceless, radial
eigenvalue `exp(f_nl - 2 f_T/sqrt6)`, and `g = GM(<r)/(k_r r^2)` exactly. Thin
slab: vertical eigenvalue `exp(f_nl + 2 f_T/sqrt6)`, `Sigma_dyn/Sigma_bar = 1/k_z`.
**The sign of the `f_T` term flips between the two — that is the decoupling knob.**

**One structural gain, for free.** `k_r = exp(...) > 0` identically, so `g > 0`
identically. **The exponential tensor grammar cannot produce a repulsive shell.**
The scalar kernel produced them at 23-48% of SPARC train points.

**Stage 1, 9600 settings:**

| metric | passing |
|---|---|
| M1 outer rms log-slope of `v_c` < 0.05 (Newton 0.190) | 272 / 9600 |
| M2 asymptotic `dln g/dln r > -1.30` | 2285 / 9600 |
| M3 BTFR slope in [3.5, 4.2] | **0 / 9600** |
| M4 `F_local = 1/k_z in [1.10, 1.70]` | 1132 / 9600 |
| M5 solar anisotropy `abs(k_r/k_t - 1) < 1e-10` | 5280 / 9600 |
| M1 and M2 | 9 / 9600 |
| **all five** | **0 / 9600** |

**The first empty cumulative cut is the BTFR**, exactly as the linearity no-go
predicts. Best BTFR anywhere 3.066; among flat settings 2.50-2.92, median 2.77;
observed 3.85 +- 0.09.

| | best asymptotic dln g/dln r | best outer rms slope |
|---|---|---|
| `f_nl` unbounded | +13.7 (runaway) | 0.0218 |
| `f_nl` bounded | +1.60 (runaway) | 0.0036 |
| **nonlocal `qbar`** | +13.7 | **0.0036** |
| **local `q` (control)** | **-2.0000 exactly, always** | 0.0121 |

**The local-`q` control never leaves Kepler — `dln g/dln r = -2.0000` for all
4800 of its settings. The path average does, but only into a runaway.**
Nonlocality is the only thing that can change the asymptotics at all.

**Stage 2 — full 3-D solves, and a bug in this lane's own first version.**
`families.tidal_hat` normalises by `sqrt(eps_T^2 + |T0|^2)` with `eps_T` in s^-2.
The first run used `eps_T = 1e-30`, which is **190x larger** than the actual
`|T0| = 5.3e-33 s^-2` at 50 kpc from a 6e10 Msun galaxy; that suppressed `That`
by ~200 and made "the anisotropy does no independent work" an artefact of the
regulariser. Redone with `eps_T` screened:

| `eps_T` | 10-20 kpc | 20-40 kpc | 40-70 kpc |
|---|---|---|---|
| **a0/(10 kpc) = 3.9e-31 — the existing screen's own default** | -0.297 | -0.051 | **-0.008** |
| 1e-33 | -0.794 | -0.812 | -0.776 |
| 1e-37 (unregularised) | -0.794 | -0.814 | **-0.816** |

**Under the existing screen's own regulariser the anisotropy is switched off
throughout galaxy outskirts** — a factor 97 suppression at 40-70 kpc.

| `eps_T` | variant | mean radial boost | vertical boost at z=3kpc | outer dlnv/dlnr |
|---|---|---|---|---|
| 3.9e-31 | f_T on (c=0.3) | 2.471 | 0.970 | -0.068 |
| 3.9e-31 | f_T = 0 control | 2.447 | 0.961 | -0.064 |
| **1e-37** | **f_T on (c=0.3)** | **2.731** | **0.958** | **-0.022** |
| **1e-37** | **f_T = 0 control** | **2.447** | **0.961** | **-0.064** |

1. **The directional term does do independent work — a 12% effect.** Turning
   `f_T` on raises the radial boost 2.447 -> 2.731 (+11.6%) while leaving the
   vertical boost at 0.958 vs 0.961 (-0.3%), and flattens the outer curve from
   -0.064 to -0.022. The brief's hypothesis is confirmed in direction.
2. **But the decoupling is overwhelmingly the GATE, not the direction.** Even at
   `f_T = 0` the radial boost is 2.45 against a vertical boost of 0.96. The
   vertical force near the midplane is sourced by dense material where
   `qbar ~ 0` and `K = I`, so an isotropic density-gated `K` already avoids the
   excessive vertical boost. `f_T` adds about a tenth on top.

**Stage 3 — on SPARC**, acceleration space, same points and split:

| setting | train rms | blind rms | repulsive points |
|---|---|---|---|
| `f_nl=expo a=0.3 p=2`, `f_T=poly c=0.3 m=1`, delta 1e6, nonlocal | **0.448** | **0.443** | **0** |
| `f_nl=poly a=2 p=2`, `f_T=zero` | 0.479 | 0.475 | 0 |
| — RAR, same points | 0.121 | 0.122 | — |
| — scalar nonlocal kernel, same points | 0.256 | 0.209 | — |
| — Newton, same points | 0.597 | 0.588 | — |

**The tensor atom is worse than the scalar kernel on SPARC and much worse than
the RAR**, though it is the only one with zero repulsive points. One caveat in
its favour: the 3-D disk solve gives a radial boost 50-67% larger than the
spherical proxy, so the equivalent-spherical number is a *lower bound*.

**Job 3(b) verdict.** Worth keeping in the grammar — it cannot produce repulsive
shells, and its directional term does genuine if small independent work once the
tidal regulariser is set correctly. Not a solution: 0 of 9600 reach the BTFR
(best 3.07 vs 3.85) and it loses to the RAR on SPARC by 3.7x, for the same
linearity reason as 3(a), which the tensor form does not escape.

## Failure modes from the brief, each checked

Shared-denominator artefacts: checked — `R` cancels exactly in
`log10(g_pred/g_obs)`; `g_obs` from `V_obs` and `g_bar` from `V_gas, V_disk,
V_bul` are independent measurements. Monotone-invariance: checked and printed
twice, both non-degenerate with interior optima. Refitting on the held-out set:
not done — parameters frozen verbatim from the previous lane's TRAIN selection,
`a0` fitted on TRAIN and frozen, blind evaluated once. Silent extraction
failures: counts asserted and echoed (123 galaxies, 71/23/23 usable,
1028/296/309 points); two degenerate points dropped identically from every model,
where an earlier version clamped them to 1e-30 and that alone moved AQUAL's
all-points rms from 0.163 to 0.526 dex.

**Test bugs found, four, all in the test.** (i) `eps_T = 1e-30 s^-2` was 190x the
actual tidal magnitude and silently switched off the anisotropy — this one
changed a conclusion. (ii) The first S1 measured smoothness by fitting *across*
the rounded corner, so a C2 function showed a spurious 3e-2 "jump". (iii) The
first S2 fitted quadratics over a window 30x wider than the feature and reported
a 4% jump in `g` that was fit error plus the first-order quadrature error of a
discontinuous integrand. (iv) The RAR's small-argument branch returned `a0` where
the correct limit is `sqrt(g_N a0) -> 0`.

## What could NOT be established

* Whether the tensor atom's SPARC number improves with real disk geometry. The
  3-D boost is 50-67% larger than the spherical proxy, so 0.448 dex is a lower
  bound; needs an axisymmetric solve per galaxy.
* A box-size sweep for the anisotropic 3-D solves. `shell_spread` 0.6-0.7 means
  the shell is not isotropic; residual boundary sensitivity unmeasured.
* The Oort number for the tensor atom at the observed height (`h = 2.5 kpc` does
  not resolve `|z| = 1.1 kpc`).
* **Whether the screen lane's "bounded anisotropy does no independent work"
  conclusion survives a smaller `eps_T`.** The default suppresses `That` by 97x
  in galaxy outskirts, and with it corrected the anisotropy does 12% of
  independent work in this atom. Families C/D/E were NOT re-run at a corrected
  `eps_T`. **This is the single most actionable follow-up.**
* The solar-system anisotropy bound for a `K` theory, converted to a rigorous
  ephemeris bound.
* Whether the R1/R2 Pareto gap closes outside the grammar searched. The
  *mechanism* is a theorem for any `F = 1 + a G(qbar) H(r)` with `G' > 0` and
  `qbar' > 0`; it does not cover forms where `qbar` decreases outwards or `G` is
  non-monotone, neither of which was searched or is physically motivated.
