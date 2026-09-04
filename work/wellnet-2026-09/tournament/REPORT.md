# The joint tournament: four channels at once, and the screen that decides it

2026-09-04. Windows 11, Python 3.13.5, CuPy 13.5.1 on an RTX 5090. Method
appendix: `METHODS.md`. Nothing was rebuilt — the cluster, the closed-form
symmetric-3x3 exponential and the GPU field machinery come from `../tensor/`;
the momentum identity and field solvers from `../screen/`; the vertical forward
chain from `../../gravity-cluster-audit-2026-09/adyn/`; SPARC from
`../../gravitylab/data.py`. All imported unmodified; SHA-256 in `tournament.json`.

## Correction (Run AQ): the `rar` k-scaling, and the numbers below

**Everything in sections 0-8 below is from the pre-correction run.** It is
preserved verbatim as `tournament_prefix_k15.json`; `tournament.json` and
`gates.json` in this directory are the corrected re-run. Read the deltas here
first.

`tw_core.mond_invert`'s `rar` branch applied `nu` to `F/(k**1.5 * a0)`, i.e.
p = 3/2 in the family `nu(F/(k^p a0)) F/k`. Within that family p is fixed, not
free: k = 1 reproduces plain RAR for every p and the Newtonian limit gives
g ~ F/k for every p, but deep MOND gives **g ~ sqrt(F a0) k^(p/2 - 1)**, so
matching the AQUAL branch's k^(-3/4) requires **p = 1/2 uniquely**. At p = 3/2
the `rar` branch ran at k^(-1/4) — a response three times weaker in the
exponent than its AQUAL twin — while the comment beside it claimed k^(-3/4).

The error is **exactly zero at k = 1**, so it never touched `scalar_a0` (where
`k_radial_pointwise` returns 1), the Newtonian limit, or any `aqual`/`newton`
candidate. Verified as a control: all **1758** such candidates re-scored
bit-identically (0 moved). It bit only the **1365** `rar` candidates with
k != 1, at 0.15 dex per e-fold of k. Fixed to p = 1/2, which preserves the
k = 1 identity exactly. Regression test: `test_mond_invert.py`.

**The amplitude absorbed it.** Because A is fitted on the cluster channel, a
weaker response was compensated by a larger |A|, so the `rar` half of the
tournament was fitted on a different footing from the `aqual` half:

| candidate | A before | A after | aqual twin |
|---|---|---|---|
| `rar\|tensor_S[plaw_p0q1s2_L300]\|phi\|pow\|m1\|I3e+12` | -40.0 | **-26.0** | -25.5 |
| `rar\|tensor_S[plaw_p0q1s2_L300]\|phi\|pow\|m2\|I3e+12` | -156.1 | **-94.66** | -94.66 |

Over all 1365 exposed twin pairs the median `rar`/`aqual` fitted-A ratio moves
from **1.331 to 1.000** — the two halves are now on a common footing, which is
the point of the fix.

**Survivors: 18 -> 26** (1 lost, 9 gained; all 10 `rar`). The list becomes
symmetric — 13 `aqual` and 13 `rar`, each an exact twin pair — where before it
was 13 `aqual` against 5 `rar`. The corrected funnel:

| screen | kills alone | unique kills | sequential |
|---|---|---|---|
| H7 asymptotic slope in [-1.25, -0.75] | 846 | 0 | 3123 -> 2277 |
| H1 cluster reach B(1 Mpc) >= 1.5 | 627 | 14 | 2277 -> 1769 |
| H4 radial RMS <= 0.30 dex | 2721 | 0 | 1769 -> 374 |
| H5 vertical amplitude in [0.301, 1.670] | 2299 | 0 | 374 -> 286 |
| H6 vertical shape chi2/dof <= 40 | 972 | 0 | 286 -> 285 |
| H2 field galaxy <= 0.040 dex | 2852 | 14 | 285 -> 134 |
| **H3 member galaxy <= 0.040 dex** | 2843 | **108** | 134 -> **26** |

**The qualitative findings survive.** H3 is still the only screen with
substantial unique kills, and still decides the tournament. Every survivor is
still gated on potential depth or the tidal invariant, with **no acceleration
gate anywhere**; the tidal count goes from 13 of 18 to **16 of 26**. The
parsimony pick is unchanged (`aqual|tensor_S[plaw_p0q1s2_L300]|phi|pow|m2|I3e+12`,
J = 1.553, k = 4) — it is `aqual`, so the bug never touched it. Section 8's
verdict is unchanged and if anything strengthened: the list is soft, it grew by
44%, and nothing should be promoted.

**Inherited by the axis/2-D lane.** `axis-2d/` was run before this correction
and its `amplitudes.json` records the pre-fix 18-survivor list with the pre-fix
amplitudes (e.g. A = -40.0 above). That lane consumes recorded amplitudes as-is
and states the conventions it inherited; its nulls are stated against a
predicted amplitude that is ~54% too large for the `rar` members. It has
deliberately not been re-run here.

## 0. The short answer

3,123 candidates scored simultaneously on all four channels. **18 survive all
seven screens.** The member-galaxy screen — the constraint the brief said nobody
had written down — is the only screen with substantial unique kills (**105**,
against 14, 14, 1, 0, 0, 0). It is where the tournament is decided.

**1. The honest expectation is confirmed on three channels, refuted on the
fourth — then reinstated by a different gate.** At a matched potential-depth gate
the scalar competitor equals or beats every tensor on radial rotation, on both
vertical channels and on cluster shape, and loses only the member screen:

| structure (gate: phi, sat, m=2, Phi0=1e12, AQUAL base) | radial (dex) | B_z | h (arcsec) | cluster (dex) | B(1Mpc) | field | **member** | J |
|---|---|---|---|---|---|---|---|---|
| **`scalar_a0`** | **0.165** | 1.62 | 33.81 | **0.091** | 2.62 | 0.0015 | **0.437** | **1.411** |
| `iso_K` | 0.189 | 2.18 | 31.10 | 0.018 | 2.35 | 0.0051 | 7.233 | 1.554 |
| `tensor_d` | 0.177 | 1.68 | 32.08 | 0.022 | 2.39 | 0.0035 | 1.357 | 1.322 |
| `tensor_T` | 0.177 | 1.38 | 34.77 | 0.032 | 2.41 | 0.0056 | 4.110 | 1.316 |
| `tensor_S` p = 0 | 0.173 | 1.44 | 36.23 | 0.203 | 2.74 | 0.0025 | **0.0071** | 1.747 |
| `tensor_S` literal p = 1 | 0.178 | 1.41 | 36.55 | 0.208 | 2.75 | 0.0034 | 0.528 | 1.781 |

Tolerance on both galaxy columns is 0.040 dex, the RAR's intrinsic scatter.

**2. The anisotropy's advantage is bought inside the weight family, not by
anisotropy.** `S` is a NORMALISED direction average, so whether the host galaxy
or the crowd of 300 members dominates is set by the mass exponent `p` and by
self-exclusion. At `p = 0` a 4e11 Msun host counts no more than a 1e9 dwarf, the
crowd wins, and `S` inside the member points along the CLUSTER radius. Restore
the brief's literal formula and the member violation goes 0.007 -> 0.528 dex,
back to the scalar's. Minimum member violation by weight family: p=0 / no
exclusion **0.007**; p=1 with exclusion 0.048-0.056; p=1 literal 0.057-0.253.

**3. Even at its best that escape is a coin flip.** Amplitude frozen after one
fit, then eight independent draws of the 300 members:

| structure | member violation, mean +- sd | realisations inside 0.040 |
|---|---|---|
| `scalar_a0` phi sat m=2 | 0.418 +- 0.011 | **0 / 8** |
| `iso_K` phi sat m=2 | 7.018 +- 0.385 | 0 / 8 |
| `tensor_d` phi sat m=2 | 1.387 +- 0.226 | 0 / 8 |
| `tensor_T` phi sat m=2 | 4.697 +- 0.470 | 0 / 8 |
| **`tensor_S` p=0, phi sat m=2** | **0.042 +- 0.028** | **4 / 8** |
| `tensor_S` p=0, phi sat m=4 | 0.107 +- 0.065 | 1 / 8 |
| `tensor_S` literal p=1 | 0.486 +- 0.216 | 0 / 8 |

This reproduces and extends the tensor lane's 0.031 +- 0.023 dex on five draws:
**marginal, not comfortable.** The scalar is not marginal — it misses by a factor
of ten with 0.011 dex scatter.

**4. A SCALAR DOES SURVIVE, gated on the tidal invariant rather than potential
depth.** This was not anticipated and it is the strongest result in the lane.

| probe | median abs T (s^-2) | median abs Phi_N (m^2/s^2) |
|---|---|---|
| cluster shells 300-1414 kpc | 3.66e-34 | 5.9-10.6e11 |
| isolated field galaxy at 10-30 kpc | 6.87e-32 (19x) | 1.13e10 |
| **cluster member galaxy at 10-30 kpc** | **5.54e-31 (151x)** | **1.09e12 (deepest)** |

**Potential depth orders the member galaxy ABOVE the cluster shell, so every
depth gate fires hardest exactly where it must not. The tidal invariant orders it
the other way round by 151x.** An inverse-tidal gate therefore switches off
inside galaxies automatically, with no anisotropy and no weight-family choice:

    aqual | scalar_a0 | tidal | inv m=2 | T0 = 1e-33
    a0 = 1.002e-10,  A = +16.0,  radial 0.168 dex,  B_z 1.515,
    h 34.82 arcsec (chi2/dof 19.4),  cluster lane-12 RMS 0.271 dex,
    cluster flat-target RMS 0.118 dex,  B = [1.56, 2.12, 3.12, 3.48],
    field 0.0070,  member 0.0001 dex,  asymptotic slope -1.00002,  J 1.914, k = 4

It costs cluster shape: its B(r) RISES outward where the lane-12 lensing shape
falls, which is a sharp falsifiable prediction rather than a fitted degree of
freedom.

**5. Momentum is violated by everything, including both scalars.**

| structure | abs(F_net)/(GM1M2/d^2), n = 28/36/44 | grad-K term |
|---|---|---|
| AQUAL / QUMOND base alone | 0.000 (variational) | — |
| `scalar_a0` potential-depth | 0.801 / 0.667 / 0.591 | — |
| `scalar_a0` **tidal**, tidal-matched pair | **0.823** | — |
| `tensor_T` | 0.872 / 0.616 / 0.581 | 0.61-1.15 |
| `tensor_S` (A = -5) | 0.245 | — |
| `tensor_d` | 1.699 / 1.756 / 1.694 | 3.13-3.59 |
| `iso_K` | 16.53 / 15.57 / 14.93 | 16.6-19.8 |

**The third-law violation is a property of "the response depends on position",
not of anisotropy.** No candidate has a declared carrier or a variational
completion.

## 1. The boundedness theorem, sharpened again

Of 130 measured combinations, 100 leave the asymptotic slope at the base law's
value because W -> 0 or W -> const, and a constant only renormalises G. The 30
whose response grows outward change it — to -0.5, a RISING curve.

**Making f unbounded does not help, and the reason is not the form of f.** Every
invariant in the grammar DECAYS outward — g_N ~ r^-2, |Phi_N| ~ r^-1, rho = 0
outside, |T| ~ r^-3, qbar -> const — so an unbounded f is evaluated on a
vanishing argument. Only an invariant that GROWS outward could change the
asymptotics, and there is none; forcing one overshoots to a rising curve. H7
kills 959 candidates on this alone.

## 2. The funnel

| screen | kills alone | **unique kills** | sequential |
|---|---|---|---|
| H7 asymptotic slope in [-1.25, -0.75] | 959 | 1 | 3123 -> 2164 |
| H1 cluster reach B(1 Mpc) >= 1.5 | 634 | 14 | 2164 -> 1637 |
| H4 radial RMS <= 0.30 dex | 2728 | 0 | 1637 -> 359 |
| H5 vertical amplitude in [0.301, 1.670] | 2291 | 0 | 359 -> 266 |
| H6 vertical shape chi2/dof <= 40 | 900 | 0 | 266 -> 265 |
| H2 field galaxy <= 0.040 dex | 2848 | 14 | 265 -> 123 |
| **H3 member galaxy <= 0.040 dex** | 2850 | **105** | 123 -> **18** |

H4/H5/H6 kill thousands but nothing uniquely — Run L's "the amplitude constrains
rather than discriminates", quantified, and it extends to the radial channel too.

**Every single-channel winner fails the tournament**: the radial winner fails H3;
the two vertical winners fail H2, H3, H4 and H5; the cluster winner fails H2 and
H3; and Newton, which wins the member screen outright at 0.0000, fails H1, H4
and H7.

## 3. Survivors

Every survivor is gated on either potential depth (all `tensor_S`) or the tidal
invariant (everything else). **Not one uses an acceleration gate**, reproducing
the tensor lane's finding from a wider grammar — and **13 of the 18 use the tidal
invariant**, which is new.

Reference rows, response genuinely off: BASE_newton radial 0.5215, h 30.80,
chi2 10.48; BASE_rar 0.1641, 35.21, 20.23; BASE_aqual 0.1647, 34.91, 19.61.

The best scalar potential-depth row for comparison —
`aqual|scalar_a0|phi|sat|m4|1e12`, radial 0.165, cluster **0.029 dex** (the best
cluster fit of any candidate reaching the amplitude), field 0.000, **member
0.507** — is eliminated by H3 alone.

## 4. Model selection

Best J = 1.553, bootstrap SE(J) = 0.131 over 400 resamples OF OBJECTS. Three
candidates lie within 1 SE, all at k = 4, so the one-standard-error parsimony
rule returns the same pick as the bare argmin. It did not have to break a tie,
but it matters that it could have: **the tidal scalar at J = 1.914 is 2.8 SE
behind and is not selected by J — it is selected by the member screen**, which is
the point of a joint tournament.

## 5. What each channel can decide

| channel | constrains | cannot do |
|---|---|---|
| radial rotation | the base law and a0 | cannot see the gate — every surviving gate is off at galaxy scales by construction |
| vertical amplitude | almost nothing by design | 0.192 dex width, 0 unique kills |
| vertical radial shape | the base law's radial run | same blindness to the gate |
| cluster amplitude + shape | the gate amplitude A | in-sample for A |
| **field + member galaxy screens** | **the gate, with ZERO free parameters** | the only out-of-sample test of a gate in the tournament |

Among the 58 candidates passing both galaxy screens, h spans 33.8-37.4 arcsec.
**Every one is 5-9 arcsec above the observed 28.65 and worse than Newton's
30.80. The vertical shape channel prefers Newton over every law in the
tournament, base laws included.** No gated law relieves Run L's tension.

## 6. Reproduction of the lanes reused

Newton 0.52155 (Run L 0.5215), h 30.7994 (30.80), chi2 10.480 (10.5);
RAR 0.164137 (0.1641), 35.2049 (35.20), 20.231 (20.2); AQUAL 0.164674 (0.1647),
34.9088 (34.96), 19.609 (20.0). Fitted a0: RAR 1.084e-10, AQUAL 1.058e-10, both
matching Run L. Split hash `e5f74522`.

Cluster channel against the tensor lane's published survivor, through an
independently written code path: B at 300/500/1000/1414 kpc published 2.51, 3.22,
2.57, 1.98 against **2.509, 3.224, 2.576, 1.979**; RMS 0.099 against **0.0986**;
member 0.015 against **0.0151**. Agreement to 0.2%.

**The 0.11 dex bar.** That figure is the RAR's scatter with per-galaxy nuisances
marginalised. Under this tournament's protocol, which gives a law no per-object
freedom at all, **the RAR's own value is 0.164 dex**, and that is the bar.

## 7. Robustness, and one number that changed

Cluster grid n = 48/64/96 gives B(1 Mpc) 2.725/2.736/2.712 and member
0.0071/0.0071/0.0070 — under 1%.

**The shell average** is harmonic throughout. The bracket between harmonic and
arithmetic at each survivor's fitted amplitude has median **0.017 dex** and max
**0.039 dex**; over all 3,123 candidates the median is 0.173 and the max 12.7 dex.
The choice of average is decisive across the grammar and merely small for the
survivors.

**Full 3-D verification, and this should temper the survivor list.** The cluster
channel is a spherical surrogate calibrated at |A_T| <= 8; the survivors sit at
|A| = 25-102. Four direct nonlinear 3-D solves:

| case | surrogate B | full 3-D B | max discrepancy |
|---|---|---|---|
| tensor lane's reference (A_T = -24.7) | 2.509, 3.224, 2.576, 1.979 | 2.854, 3.327, 2.589, 1.982 | 13.7% |
| survivor 1 (A = -94.7) | 2.292, 2.810, 2.682, 2.532 | 2.501, 2.873, 2.696, 2.533 | 9.1% |
| survivor 4 (A = -25.0) | 1.918, 2.385, 2.739, 2.906 | 2.877, 3.765, **4.667, 5.107** | **75.7%** |
| best scalar depth gate | 3.563, 3.262, 2.468, 1.941 | 4.793, 4.205, 2.916, 2.174 | 34.5% |

K's condition number is only 8-55, so all converge in 7-8 Picard steps. The
surrogate always under-predicts, but by up to 76% — far outside its 20.4%
calibration. **The cluster RMS and the ranking among the 18 are good to tens of
per cent, not better**, and survivor 4's true 3-D profile RISES outward.

**|Phi_N| is defined by its boundary rule.** Two of four rules are global and
admissible, and they differ by **0.87 dex in the median galaxy potential depth**
(1.42e10 for `inf`, 1.06e11 for `flat`). A depth gate is a function of exactly
that number, and the margin between off and on is only 0.9 dex. **The whole
potential-depth mechanism rests on a quantity uncertain by nearly a decade.**
The tidal-gated survivors do not have this problem: |T| is a local second
derivative with no boundary constant.

**Coarse graining.** The four smooth-field structures show exactly zero drift —
a property of the construction, not a passed test. `tensor_S` drifts by up to
0.17 in S_rr at N <= 40, converging above N ~ 1e3. **The synthetic cluster's 300
members put it at S_rr ~ 0.139-0.147, about 30% off the converged value**, so the
well-network response is catalogue-resolution dependent at that level.

**One correction after the first full run.** `ch_cluster.base_tensor` returned
W = 1 rather than 0 when `form == "off"`, so the three BASE_ rows were given a
fitted amplitude the radial channel never saw, and reported member violations of
0.38-0.45 dex for laws with no environmental response at all. Fixed; the 18
survivors are unchanged.

## 8. Failure modes checked

Shared-denominator: cluster B is a ratio of two solves of the SAME source, so the
artefact has no surface; the vertical comparison does have one and it is stated
rather than hidden (the Newtonian term cancels exactly from the difference).
Monotone-invariance verified with the spread printed for every headline
statistic: cluster RMS 0.447, member 12.0, h 242.7, B_z 8.4e41, radial 299.4,
asymptotic slope 2.0. a0 fitted on SPARC train only; validation and blind never
loaded; **KiDS and wide binaries never loaded, listed or referenced.**

**Four test bugs caught.** (i) The momentum null must be the same base law with
the response off, not Newton, or AQUAL's own 1-2% discretisation residual is
charged to the gate. (ii) The momentum test must be run WHERE the response is on
— on a galaxy-scale pair every phi-gated candidate returns a null because
|Phi_N| ~ 1e10 is four orders below the gate. (iii) The field-galaxy probe must
not see the cluster's gas: the first version gave the isolated galaxy
|Phi_N| = 1.22e12 instead of 1.13e10, silently switching the gate on inside the
control. (iv) The coarse-graining cloud drew N identical directions from one
re-seeded RNG, which would have made every drift zero.

## 9. What could not be established

One synthetic A2029 with a statistical member population; the member screen is
binding and its eight-realisation scatter is the difference between passing and
failing. The 0.040 dex member tolerance is imported from the RAR's intrinsic
scatter and needs a real fundamental-plane analysis. The member violation is
computed from the same spherical reduction as everything else and **cannot be
verified by a 3-D solve at 94 kpc resolution by any lane**. The cluster ranking
among survivors is good to tens of per cent only. The external field effect is
neglected in the member probe. The vertical channels can never discriminate a
gate. Momentum: every candidate violates it and none has a carrier.

## 10. Verdict

**Do not promote anything.** 18 of 3,123 survive, and the list is soft — the
cluster surrogate is out of its calibrated range at their amplitudes, and one
direct 3-D check moves a survivor by 76% and reverses its slope.

What is robust, because the effects are one to two orders of magnitude larger
than these uncertainties:

* The member-galaxy screen is the discriminating constraint of the entire
  programme — 105 unique kills against 14, 14, 1, 0, 0, 0.
* At a matched potential-depth gate the scalar equals or beats every tensor on
  three of four channels and loses only that screen, by a factor of sixty; the
  tensor's escape is a `p = 0` choice inside the weight family, vanishes under
  the brief's literal formula, and holds in only 4 of 8 member realisations.
* **A scalar law gated on the tidal invariant passes all seven screens with a
  member violation of 0.0001 dex and no anisotropy at all**, because a galaxy's
  internal tidal field exceeds a cluster shell's by 151x — the opposite ordering
  to potential depth. No tensor, no weight family, no boundary-rule ambiguity in
  |Phi_N|. It pays with a cluster profile that RISES outward where the
  lensing-derived shape falls: a clean falsifiable prediction.
* The boundedness theorem's prescribed repair is unavailable in this grammar, and
  the obstruction is the asymptotic DECAY of every available invariant, not the
  boundedness of f.
* Momentum non-closure is a property of position-dependent response, not of
  anisotropy.

**Recommended next measurements.** (1) The internal dynamics of cluster member
galaxies — the tournament says it is the only screen with independent power, and
it separates the two surviving gate families by three orders of magnitude in the
invariant they respond to. (2) The radial run of the cluster excess, with the
tidal gate's RISING prediction as the alternative hypothesis, since that is where
the two families disagree most sharply. (3) A variational completion or a
declared momentum carrier for any candidate that gets that far.
