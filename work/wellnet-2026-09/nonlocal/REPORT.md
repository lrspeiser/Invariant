# Symmetric nonlocal kernel: implementation, screen and verdict

Lane `work/wellnet-2026-09/nonlocal/`.
Code: `nonlocal_kernel.py`, `models.py`, `screen_nonlocal.py`.
Results: `nonlocal_results.json`, full console transcript in `screen_log.txt`.
Whole screen runs in 584 s on the RTX 5090 (CuPy 13.5.1).

The law screened here is

```
Phi(x)     = -G Int [ rho_b(x') / |x - x'| ] F[ qbar(x,x'), Tbar(x,x') ] d^3x'
qbar(x,x') = Int_0^1 q[ (1-s) x + s x' ] ds
Tbar(x,x') = Int_0^D q T_ij k^i k^j dl / a0
```

with the four families `F1 = 1 + a q^p`, `F2 = exp(a q^p)`,
`F3 = 1 + a q^p/(1 + b q^p)`, `F4 = 1 + a q^p + b Tbar`, and global constants
only. All four are implemented; all four are screened.

---

## VERDICT ON THE FLAT-ROTATION-CURVE QUESTION (brief item 3)

**Asymptotically flat rotation curves are impossible for the entire family.
That is a theorem, not an observation. But the reason the brief gives for
expecting that answer is wrong, and over the radial range where rotation
curves are actually measured the family works far better than the brief
anticipates.** The verdict has three parts and the reader needs all three.

### (1) Asymptotic flatness: impossible, proven, and verified

For a source seen from outside, `Phi = -G M F(r)/r` is exact, so

```
v_c^2 = G M ( F/r - dF/dr ) = (G M F / r) ( 1 - dlnF/dlnr ).
```

Every member has `qbar` in `[0,1)` and a convergent `Tbar`, so `F` is bounded
above: `sup F` is `1+a`, `e^a`, `1 + a/(1+b)`. Therefore
`r v_c^2 -> G M sup F` and the curve is *exactly* Keplerian at large radius
with `G` renormalised by a constant. Measured at 30 Mpc from a Milky-Way-like
source:

| family | alpha | sup F | r v^2 / G M at 30 Mpc | dlnv/dlnr |
|---|---|---|---|---|
| F1 | 3 | 4.000 | 3.9998 | -0.5000 |
| F2 | 1 | 2.7183 | 2.7181 | -0.5000 |
| F3 | 5, b=1 | 3.500 | 3.4999 | -0.5000 |

No member of this family, at any parameter value, has an asymptotically flat
rotation curve.

### (2) Over the measured range it CAN flatten a curve, and the brief's reason for expecting otherwise does not hold

The brief's argument is: "the surrounding q is roughly constant on galaxy
scales, so F is roughly constant, so Phi is a rescaled Newtonian potential,
so `v^2 ~ 1/r`". **That argument drops the `-G M F'` term, and that term is
not small.** In a Milky-Way-like model with the smooth q and `alpha = 3` the
measured `dlnF/dlnr` reaches **0.899**, where the naive argument assumes 0.
Since `v_c^2 = (G M F/r)(1 - dlnF/dlnr)`, an F rising with logarithmic slope
approaching 1 flattens the curve completely. Numerically:

* across a six-galaxy ladder spanning `M_b` from 1.5e9 to 2.1e11 and `r_d`
  from 1 to 6 kpc, one global parameter set brings the RMS outer log-slope of
  `v_c` from the Newtonian **0.190** down to **0.056** — a factor 3.4, with
  slopes between +0.08 and -0.03 across the whole ladder;
* forward-modelled against **71 SPARC train galaxies** with one global
  parameter set and no per-galaxy freedom, the family reproduces the
  modification factor the observed curves demand to **0.156 dex rms**, against
  **0.646 dex** for Newton.

This is a real effect, not a rounding of one. Section 4 gives the theorems and
the numbers.

### (3) It fails on four other things instead, each quantified

| failure | measured | what it should be |
|---|---|---|
| repulsive shells (`v_c^2 < 0` somewhere) | 42% of the solar-system-allowed grid; 70% for the clipped delta q | 0% |
| galaxy-to-galaxy scatter no global parameter can absorb | 0.160 dex | the RAR sits at ~0.11 dex |
| baryonic Tully-Fisher | slope 2.88, scatter 0.229 dex, size residual +0.49 | 3.85 +/- 0.09, 0.10 dex, 0 |
| local dynamics (Oort limit) | best-fitting sets give `F_local` = 2.2 to 5.3 | 1.36 +/- 0.15 |
| **the nonlocal signature itself** | **0 of 108 parameter points keep it** | at least 1 |

The last row is the deepest, and it is arithmetic rather than numerical. The
family's entire reason for existing is that *what lies between two masses*
changes their coupling. That signature needs `rho_ref` near the cosmic mean
baryon density, 6.2 Msun/kpc^3, so that a void and a filament differ in `q` at
all. Bending a rotation curve needs `rho_ref` near 1e5-1e6 Msun/kpc^3. Those
are four to five decades apart, there is one global density scale, and the
brief forbids a per-object one. **The family can be a theory of rotation
curves or a theory of intervening structure. It cannot be both.**

A narrow corner does survive the two local constraints: 47 of 396 global sets
pass both the inverse-square-law bound and the Oort window, the best being the
screened q with `rho_ref = 1e6`, `L_q = 2 kpc`, `F1`, `alpha = 10`, `p = 2`,
which costs 0.04 dex of rotation-curve accuracy (0.196 instead of 0.156 dex)
and sits at `F_local = 1.666`, at the very edge of the allowed window. That
corner has no nonlocal signature left.

### What minimal modification would be needed

Boundedness of `F` is the whole obstruction, and it follows from `qbar` being
an *average* of a bounded field. Two escapes, both of which leave this family:

* **Give `F` explicit separation dependence**, `F(qbar, |x-x'|)`, growing
  without bound as the separation grows in low-q regions. That is what the
  nonlocal `f(Box^-1 R)` literature does. It is a different theory, and it
  re-opens the solar-system question, because the kernel then changes shape
  rather than amplitude.
* **Replace the path average by an extensive path functional** whose value
  grows with path length. The `Tbar` term in F4 is of that type, and it is
  implemented and screened here, but the tidal integrand falls as `l^-3` and
  converges, so F4 is bounded too and inherits Theorem 1 — it behaves like F1
  with a small correction at every parameter value tested. An integrand
  falling no faster than `l^-1` would be needed, which is a long-range force
  under another name.

Within the family as stated, the only lever that helps is making `q` vary on
the rotation-curve scale, and that is what the screened definition with
`L_q` of a few kpc does. It is also what breaks the solar neighbourhood.

---

## The q definitions, and how much depends on the choice

Both definitions the brief names are implemented, plus a third as a control.

| tag | definition | note |
|---|---|---|
| `delta` | `q = -delta/(1+delta)` clipped to `[0,1)` | the algebra is `q = rho_ref/rho_s - 1`, so `q = 0` wherever `rho_s >= rho_ref` and `q -> 1` once `rho_s <= rho_ref/2`. It is a **near-step function of density**. |
| `screen` | `(1 - L^2 lap) q = S(rho, g)`, `S` = the programme's Q3 source | solved, not approximated: by the exact Green function in spherical symmetry, by a banded solve on a line for the solar-neighbourhood check. |
| `smooth` | `q = 1/(1 + (rho_s/rho_ref)^m)` | the programme's Q1; smooth everywhere so it has analytic gradients. Needed for the momentum residual, where a clipped q would hide the effect. |

**Adopted as primary: `screen`.** It is the only one of the three that both
avoids manufacturing repulsive shells (15% of its parameter grid versus 70%
for `delta`) and survives the inverse-square-law bound, and it is the
definition this programme already uses elsewhere.

**The choice is not cosmetic, and the dependence is systematic rather than
random:**

| q definition | repulsive fraction of the grid | inverse-square law | Oort limit |
|---|---|---|---|
| `delta` | 503/720 = **70%** | **exactly safe**, `q(Sun) = 0` identically | fails low: predicts `F_local = 1`, no local dark matter |
| `smooth` | 142/344 = **41%** | needs `rho_ref < 3.0e5` at `alpha = p = 1` | passes in a narrow band |
| `screen` | 103/708 = **15%** | passes easily, `eps ~ 5e-13` | **fails high**, `F_local = 2.2-5.3` |

That trade-off is structural. The sharp q that is trivially safe in the solar
system is the one whose switch drives `dlnF/dlnr` past 1 and removes circular
orbits. The screening length that makes q vary smoothly across a rotation
curve is the same length that imports the outer void state into the solar
neighbourhood. **No q definition is good at both ends**, and the ranking of
parameter sets changes materially with the choice, so the primary must be
declared, which it is.

---

## 1. Newtonian limit (brief item 1)

`PASS`, exactly.

* At `alpha = 0` the computed potential is **bit-identical** to Newton,
  `max |dPhi/Phi| = 0.0`, for all four families.
* Against the closed-form potential of an exponential sphere, all four
  families at `alpha = 0` agree to `1.2e-13` relative; the circular speed from
  the spline gradient to `1.6e-6`.
* The deviation vanishes exactly linearly in `alpha`: 5.000000e-2,
  5.000000e-3, 5.000000e-4, 5.000000e-5, 5.000000e-7 at `alpha` = 1e-1, 1e-2,
  1e-3, 1e-4, 1e-6, with `q = 1/2` everywhere.
* The 3-D direct sum at `alpha = 0` matches a plain Newtonian pair sum to
  `2.2e-16`.

---

## 2. Reciprocity, and the momentum it does not conserve

**Reciprocity holds to round-off.** `qbar(x,x')` equals `qbar(x',x)` to
`4.1e-16` over 300 random pairs in a deliberately asymmetric four-blob q field.
The path average is manifestly invariant under `s -> 1-s`; the numerics confirm
it rather than assume it.

**But a symmetric kernel does not conserve momentum.** This is the sharpest
structural result in the lane.

With the test-particle force `f = -m grad Phi` and q held fixed, an isolated
two-body system feels a net force

```
f_1 + f_2 = G m1 m2 F'(qbar) <grad q>_path / D
<grad q>_path . n  =  [ q(x2) - q(x1) ] / D    exactly, by the fundamental
                                               theorem of calculus along the
                                               segment
```

so its magnitude along the separation is `F'(qbar) * [q2 - q1]` in units of
the Newtonian force. Measured with q sourced self-consistently by the two
bodies themselves (`alpha = 1`, `p = 1`, separation 100 kpc, smoothing 20 kpc):

| m2/m1 | qbar | q2 - q1 | \|f1+f2\| / f_Newton | F'(qbar)(q2-q1) |
|---|---|---|---|---|
| 1 | 0.00471 | 0 | 0.000000 | 0.000000 |
| 0.3 | 0.00886 | +2.92e-3 | 2.9238e-3 | 2.9238e-3 |
| 0.1 | 0.01667 | +1.12e-2 | 1.1183e-2 | 1.1183e-2 |
| 0.01 | 0.06689 | +1.11e-1 | 1.1053e-1 | 1.1053e-1 |

The identity holds to `4.9e-12`. Equal masses give **exactly zero** by
reflection symmetry. Every unequal-mass pair leaks, because the heavier body
digs a deeper density well and therefore sits at lower q. At mass ratio 100
the leak is **11% of the pair's own binding force**, directed towards the
more void-like member.

Symmetry of the kernel buys reciprocity of the pair term. Only *translation
invariance* buys momentum conservation, and `F(qbar)` is not translation
invariant while q is held fixed. **An explicit momentum carrier must therefore
be declared.** q has to be a dynamical field with its own stress-energy and the
leaked momentum has to be the momentum it carries. Until that field equation is
written down the law is a force prescription, not a closed theory.

**Deliberately broken reciprocity, as the brief asks.** Weighting the path
average by `w(s) = 1 + gamma (s - 1/2)`, applied from whichever endpoint is
named first, gives `W(x,x') != W(x',x)` and adds a second, independent leak
`G m1 m2 [F(qbar_ij) - F(qbar_ji)] n / D^2`. On an isolated three-body system
the exact split is

| gamma | total / f_N | gradient term | asymmetry term | split residual |
|---|---|---|---|---|
| 0 (reciprocal) | 0.01482 | 0.01482 | 0 | 5.7e-15 |
| 0.5 | 0.01795 | 0.01583 | 0.00215 | 3.1e-15 |
| 1.0 | 0.02121 | 0.01693 | 0.00430 | 7.4e-15 |
| 2.0 | 0.02791 | 0.01931 | 0.00860 | 3.4e-15 |

Both leaks scale exactly linearly in `alpha` and vanish to `8e-17` at
`alpha = 0`. Note what the `gamma = 0` row says: **reciprocity is not the thing
that protects momentum.** Breaking it adds a second violation on top of one
that was already there.

One trap is recorded in the code because the first version of the test fell
into it: applying `w(1-s)` on the reverse path silently *restores* reciprocity,
and the asymmetry term then measured exactly zero for every `gamma`. A
deliberately non-reciprocal kernel must apply the same `w(s)` measured from
whichever endpoint is named first.

---

## 3. Solar-system safety (brief item 2)

The relevant channel is a violation of the inverse-square law. The anomalous
force is `G m M F'(qbar) grad_1 qbar / D` — a **fixed-direction** term falling
as `1/D` rather than `1/D^2` — so

```
eps(D) = |anomalous| / |Newtonian| = |F'(qbar)| |grad q| D / (2 F)
```

and the *inner* solar system is the binding case. The exact path average agrees
with this linearisation to 0.25% at 1, 10 and 30 AU.

**The clipped delta form is exactly safe.** `q(Sun) = 0` identically for any
`rho_ref` below the local smoothed density `4.0e7 Msun/kpc^3`, so `F = 1` and
`eps = 0` — not small, zero, to all orders, with no expansion and no residual.

**The smooth form**, with a Milky Way disk (`Sigma(R0) = 45 Msun/pc^2`,
`h_R = 2.6 kpc`, `h_z = 0.30 kpc`, midplane `rho(R0) = 7.6e7 Msun/kpc^3`,
consistent with the Bland-Hawthorn & Gerhard 2016 local baryon budget):

| rho_ref | q(Sun) | q(25 kpc) | eps(1 AU), a = p = 1 | a_anom(1 AU) |
|---|---|---|---|---|
| 6.2 (cosmic mean) | 1.5e-7 | 1.0e-4 | 2.1e-16 | 1.2e-18 m/s^2 |
| 1e3 | 2.5e-5 | 1.6e-2 | 3.3e-14 | 2.0e-16 m/s^2 |
| 1e4 | 2.5e-4 | 1.4e-1 | 3.3e-13 | 2.0e-15 m/s^2 |
| 1e5 | 2.5e-3 | 6.2e-1 | 3.3e-12 | 2.0e-14 m/s^2 |
| 1e6 | 2.4e-2 | 9.4e-1 | 3.2e-11 | 1.9e-13 m/s^2 |
| 1e7 | 2.0e-1 | 9.9e-1 | 2.1e-10 | 1.3e-12 m/s^2 |

Against a bound of `1e-11` on this channel, `alpha = p = 1` requires
`q(Sun) < 7.4e-3`, i.e. `rho_ref < 3.0e5 Msun/kpc^3`. The midplane density
contrast `rho(8.2 kpc)/rho(25 kpc) = 663` then still delivers `q = 1` at
25 kpc. **So the solar-system bound and a galaxy-scale q of order unity are
compatible** — with about one order of magnitude of headroom, not more.
`p = 2` buys three further decades, because `q_Sun^2` is tiny while
`q(25 kpc)^2` is not.

**The screened form passes the inverse-square-law bound easily**, `eps` about
`3e-13` to `6e-13` even at `q(Sun) = 0.40`, because `L_q = 10 kpc` flattens
`|grad q|` to `7.7e-5 kpc^-1`. This is worth stating plainly: the screening
length does not merely fail to hurt the solar system, it actively protects it.

**And then the Oort limit is what bites.** With no dark matter the local
dynamical surface density must be the local baryonic one times `F` evaluated on
the short paths that set the vertical force.
`Sigma_dyn(|z| < 1.1 kpc) = 68 +/- 4 Msun/pc^2` against
`Sigma_baryon = 47-54` (Bovy & Rix 2013; McKee, Parravano & Hollenbach 2015,
quoted from the literature and not refitted here), so `F_local` must sit near
**1.36**, certainly inside `[1.1, 1.7]`. The global parameter sets that best
reproduce SPARC give `F_local` between **2.2 and 5.3**, a factor 1.6 to 3.9 too
large. 47 of 396 sets do fit inside the window, at a cost of 0.04 dex in
rotation-curve accuracy, the best sitting at `F_local = 1.666` — the very edge.

`F_local` is computed from each family's own `F`, not from `1 + alpha q^p` for
every member; using the F1 form for the exponential and Pade members
misreports it by up to a factor two, which is exactly the difference between
passing and failing.

`q(Sun)` itself rests on two 1-D reductions that bracket it: a vertical slab
gives 0.40 and an independent spherical Milky Way model gives 0.62. They err in
opposite directions (the slab under-counts the inner disk, the spherical model
under-counts the local midplane density by a factor 15) and both land far
outside the window at large alpha, so the conclusion does not depend on which
is used. A full axisymmetric screened solve would pin the number inside that
bracket; it was not done, and it is the main numerical loose end in this lane.

---

## 4. Rotation curves in detail

### 4.1 Three theorems, each verified numerically

**Theorem 1 (asymptotic Keplerianity).** Bounded `F` implies
`r v_c^2 -> G M sup F`. Verified to four significant figures at 30 Mpc with
`dlnv/dlnr = -0.5000`; table in the verdict above.

**Theorem 2 (the repulsion threshold).** `v_c^2 < 0` whenever
`dlnF/dlnr > 1`. **Flat rotation sits exactly on the boundary of a repulsive
regime**: to flatten a curve, `F` must rise with logarithmic slope tending to
1, and any overshoot removes circular orbits entirely. A q that switches
sharply always overshoots, which is why the clipped delta form — a near-step
function of density by construction — produces repulsive shells in 70% of its
parameter grid against 15% for the screened form.

**Theorem 3 (the flat window).** The only `F` making `v_c` exactly flat is
`F = C r ln(r_*/r)` with `C = v_f^2/(G M)`; substituting it gives
`F/r - F' = C` identically, verified to `8.9e-16`. Two bounds follow:

| alpha | sup F | widest exactly-flat r2/r1 | with F(r1)=1, F'(r1)=0 |
|---|---|---|---|
| 0.3 | 1.30 | 4.35 | <= 1.30 |
| 1 | 2.00 | 11.55 | <= 2.00 |
| 3 | 4.00 | 36.26 | <= 4.00 |
| 10 | 11.00 | 144.7 | <= 11.00 |
| 30 | 31.00 | 522.0 | <= 31.00 |

The second column is the unconstrained optimum, with the apex of `F` set on the
ceiling. The third is the physically relevant one — an inner rotation curve
fitted by baryons alone demands `F ~ 1` and `F' ~ 0` where the flat part joins
on — and its proof is two lines: `v_f^2 = G M F(r1)/r1` and
`v_f^2 <= G M F(r2)/r2` give `r2/r1 <= F(r2)/F(r1) <= sup F`. **A decade of
exactly flat curve needs `alpha >= 9` under the physical constraint, or
`alpha ~ 1` without it.**

### 4.2 The global-parameter screen on a model ladder

1772 solar-system-allowed configurations across three q definitions, five
`rho_ref` values, four families, four `alpha` and three `p`, each evaluated on
a six-galaxy ladder over 2-20 disk scale lengths.

* **748 of 1772 (42.2%) produce a repulsive shell** somewhere on the ladder,
  broken down by q definition in the table in section "The q definitions".
* Best RMS outer log-slope of `v_c` across the ladder: **0.056** against the
  Newtonian control's **0.190**. The best sets are all the screened q with
  `L_q = 2 kpc` and `rho_ref` between 1e5 and 1e6, with `alpha` of 3 to 10.
* Individual slopes for the best set, from dwarf LSB to massive spiral:
  +0.08, -0.09, -0.03, -0.03, -0.02, -0.02, against the Newtonian
  +0.00, -0.19, -0.12, -0.21, -0.23, -0.26.

**Monotone-invariant-statistic check, as the brief requires.** The headline
statistic must move with the parameter it is supposed to measure. It does, and
non-monotonically:

| alpha | 0.1 | 0.3 | 1 | 3 | 10 | 30 |
|---|---|---|---|---|---|---|
| RMS slope | 0.179 | 0.160 | 0.113 | 0.056 | 0.098 | 0.172 |

Spread 0.123 over three decades of `alpha`, with a genuine interior optimum at
`alpha ~ 3`. The statistic is not degenerate in the parameter.

### 4.3 What the SPARC train curves demand

For a source seen from outside, `F_req(r) = -r Phi_req(r)/(G M_b)` with
`Phi_req(r) = -Int_r^inf v_obs^2 dln r'`, taking the tail beyond the last
measured point Keplerian (which Theorem 1 forces anyway). 75 train galaxies,
no fitting of any kind, a pure forward inversion:

* `max F_req` over the sample **17.53**, so one global `alpha` must be at least
  **16.5**; letting each curve stay flat for another factor of two in radius
  raises that to 28.7.
* `F_req` percentiles 5/25/50/75/95/100: 2.04 / 3.85 / 6.19 / 8.27 / 11.00 /
  17.53.
* `max dlnF/dlnr` percentiles 5/50/95/100: 0.302 / 0.554 / 0.827 / **0.939**.
  The repulsion threshold of Theorem 2 is 1.000. The data sit below it by
  construction, since `v_obs^2 > 0` forces it, but the *margin* is the
  fine-tuning: the best-observed galaxies demand an F rising at 94% of the
  slope at which gravity would reverse.
* No galaxy requires `F < 1` anywhere beyond `2 R_disk`, so the sign of `alpha`
  is not in question.
* Spread of `log10 F_req,max`: 0.243 dex, correlated with `log V_flat` at
  `-0.336`. Low-mass galaxies need the larger modification, which is the right
  sign for a void-state kernel.

### 4.4 Forward test with one global parameter set

Each train galaxy gets an **equivalent spherical mass distribution** defined by
`M(<R) = R V_bar^2 / G` at its own tabulated radii, with
`V_bar^2 = |V_gas| V_gas + 0.5 V_disk^2 + 0.7 V_bul^2`. That makes the model's
Newtonian curve identical to the tabulated one by construction, so the
baryon-geometry error is *removed* rather than estimated: the control measures
**0.0132 dex rms**. The q field is then computed with global parameters only
and `F_eff(r) = -r Phi/(G M_tot)` is compared with `F_req(r)` beyond
`2 R_disk`. 71 usable galaxies, 396 global parameter sets, no per-galaxy
freedom.

| | rms | bias | galaxy scatter | F_local | ISL | Oort |
|---|---|---|---|---|---|---|
| best overall (screen 1e6, L_q=2, F3, a=10, p=1) | **0.156** | -0.039 | 0.160 | 3.05 | ok | fails |
| best also passing both local tests (screen 1e6, L_q=2, F1, a=10, p=2) | 0.196 | -0.103 | 0.146 | 1.67 | ok | ok |
| Newton, `F = 1` everywhere | 0.646 | -0.614 | 0.197 | 1.00 | ok | fails low |

**The family improves on Newton by 0.49 dex in rms.** What it leaves is
0.156 dex — a factor 1.43 typical error in the modification factor — of which
**0.160 dex is galaxy-to-galaxy scatter that no global parameter can absorb**.
The radial acceleration relation sits at about 0.11 dex, so this family lands
close to, but not inside, the accuracy of the relation Run J could not beat.

The first version of this test used an exponential-sphere baryon model built
from `R_disk` and total masses. Its control was **0.324 dex rms, twice the
residual being measured**, so that version was discarded rather than reported.
The price of the replacement is stated: the equivalent spherical *density* is
not the true 3-D density — a disk's midplane density is several times higher at
the same radius — so the q field built from it is biased towards larger q.

### 4.5 The baryonic Tully-Fisher relation

**Shared-denominator check performed.** The BTFR residual is regressed against
`log r_d`, deliberately *not* against the central surface density
`Sigma_0 = M/(2 pi r_d^2)`, because `Sigma_0` carries the same `M` that sits on
the abscissa and would manufacture a correlation from nothing — the failure
mode that retracted `rho_p = -0.304`. `r_d` is an independent input.

| case | slope | scatter | d(residual)/d(log r_d) |
|---|---|---|---|
| Newton, alpha = 0 | 2.07 | 0.311 dex | +0.74 |
| screen, F1, a=3, p=0.5 | 2.88 | 0.229 dex | +0.49 |
| screen, F4 (tidal), a=3, p=0.5 | 2.88 | 0.230 dex | +0.49 |
| screen, F3, a=10, p=0.5 | 2.83 | 0.241 dex | +0.51 |
| **observed SPARC** | **3.85 +/- 0.09** | **0.10 dex** | **0** |

The family moves every number in the right direction and reaches none of them.
It recovers about half the slope deficit, a quarter of the excess scatter, and
a third of the spurious size dependence. A residual size dependence of +0.49
means a galaxy twice as large at fixed mass sits 0.15 dex off the relation,
which the observed relation does not allow.

---

## 5. What only a nonlocal kernel predicts (brief item 4)

The signature is that two sources at the same separation couple differently
when the material between them differs. **No point-local law can produce it** —
not MOND, not the RAR, not `f(R)`, not any `mu(g)` or `mu(rho)` — because those
depend only on fields evaluated at one point. Testable configurations, sized at
`rho_ref = rho_bar_b` (the only choice for which intergalactic environments
differ in q at all), with the assumed contrast stated:

| configuration | q: dense -> void | coupling change (a = 0.3 / 1 / 3) | relative velocity |
|---|---|---|---|
| cluster pair, 10 Mpc, filament (30x mean) vs void (0.2x mean) | 0.032 -> 0.833 | +24% / +78% / +219% | +11% / +33% / +79% |
| galaxy pair, 1 Mpc, filament (10x) vs void (0.2x) | 0.091 -> 0.833 | +22% / +68% / +175% | +10% / +30% / +66% |
| lensing sightline, 2 Mpc, wall (5x) vs void (0.2x) | 0.167 -> 0.833 | +19% / +57% / +133% | +9% / +25% / +53% |

The best measurements are: the mean pairwise infall velocity of cluster pairs
split on the density of the connecting segment; the azimuthal dependence of
stacked tangential shear at fixed projected radius on the transverse galaxy
density; and the relative velocity dispersion of galaxy pairs matched in
separation and mass but differing in intervening density. All three need a wide
survey with a void/filament catalogue. **KiDS and wide binaries are sealed
holdouts for this programme and are excluded from this list. Neither was
loaded, read or used anywhere in this lane.**

**The tension that makes this moot.** Four requirements were imposed at once —
the inverse-square-law bound at 1 AU, the Oort window, `M_dyn/M_b >= 2` at
25 kpc in a Milky-Way-like galaxy, and at least a 5% void-versus-filament
contrast in the coupling of a matched pair. **0 of 108 parameter points satisfy
all four.** The obstruction is that `rho_ref` must be near 6.2 Msun/kpc^3 for
intergalactic environments to differ in q, and near 1e5-1e6 for q to rise
across a rotation curve.

**The cluster excess: half right.** With `rho_ref = 1e5` and `alpha = 1`, a
beta-model cluster (`M_gas = 1.2e14`, `r_c = 200 kpc`, `beta = 0.65`,
`R500 = 1073 kpc` and `M500 = 3.4e14` from `M_gas/f_gas` with `f_gas = 0.13`)
gives

| r/R500 | 0.21 | 0.36 | 0.62 | 1.06 | 1.81 | 3.09 |
|---|---|---|---|---|---|---|
| M_dyn/M_b | 0.945 | 1.249 | 1.559 | 1.780 | 1.902 | 1.959 |

That **is** an excess organised by `r/R500`, rising monotonically through
`R500`, which is the shape the programme's own cluster audit reports. But it
saturates at `1 + alpha = 2` where clusters need about 6; inside `0.25 R500`
the `-G M F'` term drives `M_dyn/M_b` *below* 1, which is the wrong sign; and
the same `rho_ref` switches the modification on **inside** low-surface-
brightness galaxies. The turn-on radius across the ladder runs from 0.08 disk
scale lengths (large LSB) to 5.8 (Milky Way), so an LSB gets a near-constant
rescaling of `G`, degenerate with its stellar mass-to-light ratio, while an HSB
gets a genuine shape change. **That is exactly backwards**: LSBs are the
galaxies with the largest discrepancies and the most shape to explain.

---

## 6. Computational cost and the accelerations (brief item 5)

The double integral is `O(N_f N_s n_s)` trilinear interpolations; evaluated at
every cell of an `n^3` grid that is `O(n^6 n_s)` — 1.1e12 pair-samples at
`n = 64`, 7.0e13 at `n = 128`.

Measured on the RTX 5090 with CuPy 13.5.1: **4.5e8 pair-samples/s** against
**3.8e6/s** on the CPU, a factor **119**, at 64-bit precision throughout.
Extrapolated all-pairs cost 2468 s at `n = 64` and 1.6e5 s (44 hours) at
`n = 128`. **A brute-force 3-D solve at production resolution is not feasible**,
which is why two accelerations are implemented.

**Acceleration 1, the spherical reduction, which is exact.** Writing the
separation `D = |x - x'|` as the inner integration variable turns
`Int dmu / D` into `Int dD / (r r')`, removing the `1/|x-x'|` singularity
analytically:

```
Phi(r) = -(2 pi G / r) Int r' rho(r') [ Int_{|r-r'|}^{r+r'} F dD ] dr'
```

With `F = 1` the bracket is `2 min(r,r')` and the exact Newtonian result drops
out identically. Cost falls from `O(n^6 n_s)` to `O(N_f N_r' N_D n_s) ~ 1e7`
and **no approximation is made at all** — the only error is quadrature,
`2.2e-6` at production settings. Every galaxy and cluster number in this report
uses it. A composite Gauss-Legendre rule in `ln r'` with a panel edge at
`r' = r` keeps the `|r-r'|` kink on a node; sharing one such node set across
all field radii is what makes GPU batching possible and gives a further factor
120 over the serial version.

**Acceleration 2, separable midpoint surrogate plus FFT, for genuinely 3-D
configurations.** Replacing the path average by `qbar ~ [q(x) + q(x')]/2` makes
`F` a function of two scalars; an SVD to rank `R` turns the potential into `R`
ordinary `1/r` convolutions, `O(R n^3 log n)`, about 15 ms per rank on a 24^3
grid. Ranks needed for a `1e-10` singular-value tail:

| family | p = 0.5 | p = 1 | p = 2 |
|---|---|---|---|
| F1 | 12 | **2** | 3 |
| F2 | 12 | **1** | 7 |
| F3 | 12 | 5 | 7 |

`F1` with `p = 1` is exactly rank 2 because `1 + alpha(u+v)/2` separates, and
`F2` with `p = 1` is exactly rank 1 because `exp(alpha(u+v)/2)` factorises.
Fractional `p` needs the full 12 because `q^0.5` has a square-root branch at
the origin.

**The accuracy price is the surrogate, not the truncation.** Against the exact
path average the midpoint form is wrong by **4.3% rms and 61% at worst**; at
rank 2 the low-rank FFT reproduces the midpoint direct sum to `2.7e-8`. That is
why the surrogate is used only for the cost benchmark and never for a physics
number in this report.

A diagnostic trap is recorded in the code. Comparing the FFT accelerator
against a *Plummer*-softened direct sum showed a `1.5e-2` "floor" that had
nothing to do with the acceleration. With `D = max(|x-x'|, h/2)` on both sides
— the kernel the FFT actually tabulates — they agree to `8.9e-16`.

---

## 7. Numerical gates

Screened against the gates the local solver passes, where they apply.

| gate | result | local solver, for comparison |
|---|---|---|
| Newtonian recovery, spherical, all four families | `1.2e-13` | 2.33e-4 |
| Newtonian recovery, 3-D direct sum vs pair sum | `2.2e-16` | - |
| resolution convergence at production settings | `2.2e-6` | 3.63e-4 |
| domain-size convergence, `r_hi` 300 kpc -> 30 Mpc | `3.8e-10` (0.00000004%) | 0.089% |
| **source-label permutation invariance** | `2.2e-16` over 8000 permuted cells | - |
| CPU vs GPU direct sum | `2.2e-16` | - |
| reciprocity `qbar(x,x') = qbar(x',x)` | `4.1e-16` | - |
| exactness at `alpha = 0` | `0.0` | - |

The domain-size gate is six orders of magnitude tighter than the local
solver's, because the kernel has no boundary condition to get wrong: the
potential is an explicit integral, not the solution of a boundary-value
problem. That is the one clear numerical advantage of the nonlocal
formulation, and it is worth keeping whatever happens to the physics.

---

## 8. Failure modes checked explicitly

The brief lists five failure modes this programme has already been bitten by.
Each was checked, and here is what the check found.

* **Shared-denominator artefacts.** Checked. The BTFR residual is regressed
  against `log r_d`, not against `Sigma_0 = M/(2 pi r_d^2)`, precisely because
  `Sigma_0` shares `M` with the abscissa. The `F_req` inversion puts `v_obs` on
  one side and `M_b` (from luminosity and HI mass) on the other, independent
  measurements with no shared quantity. `F_eff/F_req` shares nothing between
  numerator and denominator: `F_eff` comes from the model potential, `F_req`
  from the observed velocities, and the one quantity they have in common,
  `M_tot`, is deliberately the *same* number in both so that it cancels
  exactly rather than correlating.
* **Monotone-invariant statistics.** Checked and printed. The headline
  statistic moves with `alpha` over three decades with spread 0.123 and a
  genuine interior optimum; the momentum leak is exactly linear in `alpha` over
  four decades with the correct limit at `alpha = 0`.
* **Refitting on the held-out set.** Not applicable and not risked: nothing in
  this lane is fitted. The only observational contact is a forward inversion
  and a forward comparison on the SPARC **train** split. Validation and blind
  were never loaded. KiDS and wide binaries were never touched.
* **Silent extraction failures.** The SPARC ingest reports its own attrition
  and every count is echoed: 175 curves with data, 123 after the frozen cuts,
  75 train, 71 usable with at least three points beyond `2 R_disk`, median
  `R_last/R_disk = 6.9`. No web retrieval was performed in this lane, so no
  manifests were required.
* **Test bugs that look like solver bugs.** Four were found, and in all four
  the test was wrong rather than the solver. (i) A `w(1-s)` reverse weighting
  silently restored reciprocity, so the asymmetry term measured exactly zero
  for every `gamma`. (ii) A cumulative cluster mass truncated at `rmax` while
  `rho(r)` kept going produced `M_dyn/M_b < 1` at all radii, which is
  impossible for `alpha > 0`. (iii) The Plummer-versus-plateau softening
  mismatch in section 6. (iv) `solar_check` evaluated `1 + alpha q^p` for every
  family, misreporting `F_local` by up to a factor two for the exponential and
  Pade members — the difference between passing and failing the Oort window.

---

## 9. What I could not establish

* **A full axisymmetric screened solve for `q(Sun)`.** The Oort result rests on
  two 1-D reductions bracketing `q(Sun)` at 0.40-0.62. Both ends fail at large
  alpha, so the conclusion is safe, but the number is not pinned and the
  surviving 47-set corner sits close enough to the boundary that it matters.
  This is the main loose end.
* **Disk geometry in the model ladder.** The ladder uses spherical baryon
  models. The theorems are geometry-independent, and the SPARC forward test
  removes the geometry error by construction, but a disk rotation curve
  computed with the full kernel was not produced. The axisymmetric machinery is
  implemented and validated to `3e-4`; only the driver is missing.
* **Whether the 0.156 dex residual can be reduced by a better q.** Only three q
  definitions and a coarse `(rho_ref, m, L_q)` grid were screened. The residual
  is dominated by galaxy-to-galaxy scatter, which suggests a structural limit
  rather than a grid limit, but that was not proven.
* **The `Tbar` tidal term never became the binding ingredient.** F4 is
  implemented, screened and appears among the best sets, but its integrand
  falls as `l^-3` and converges, so it inherits Theorem 1 and behaves like F1
  with a small correction at every parameter value tested. Whether some other
  tidal invariant evades the boundedness theorem was not explored.
* **A dynamical q with its own stress-energy.** The momentum leak is measured
  and its exact form derived, but the field equation that would carry the
  leaked momentum was not written down. Without it the law is a force
  prescription rather than a theory, and no cosmological or lensing prediction
  from it should be taken at face value.
* **Nothing here is a blind test.** Every number is from the train split or
  from model galaxies. The family is not in a state to spend a blind
  evaluation on.
