"""Emit the lane's REPORT.md.  Numbers are quoted from formation_results.json;
this file exists only so the report is version-controlled alongside the code
that produced it."""
import io
import os

BODY = """# Formation and stability: the cheap linear gate

2026-09-04. Windows 11, Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1, CPU only.
Lane: `work/wellnet-2026-09/formation/`. Code: `linear_response.py`,
`growth.py`, `test_solver.py`. Results: `formation_results.json`,
`solver_gates.json`.

**No observational data is loaded in this lane at all.** Nothing is fitted, no
split is consumed, no holdout is touched, and there is therefore no
blind-protection issue to manage - stated rather than omitted. KiDS and wide
binaries are not loaded, not listed and not referenced; neither is SPARC nor any
cluster catalogue. Every constant is frozen from `../tournament/tournament.json`
and `../tournament/focus.json` and is quoted with its source row in
`formation_results.json["constants"]`. `../tensor/wellnet.py`,
`../tournament/tw_core.py` and `../../gravitylab/solver.py` are imported
unmodified; their SHA-256s in `formation_results.json` match the hashes the
tournament recorded.

---

## 0. Does each candidate admit a sensible homogeneous background at all?

**Three of the six do, two do not, and the one the brief flagged as the
interesting one does - for the opposite reason to the one anticipated.**

**The tidal-gated scalar does not diverge on a homogeneous background: it
SATURATES.** The brief expected `f ~ (T/T0)^-m` to blow up where `|T| = 0`. The
form the tournament actually froze is `inv`, and `tw_core.W_of` implements it as
`W = 1/(1 + I^m)`, which `tw_core.W_sup` reports as BOUNDED by 1. On an exactly
homogeneous background the Hessian of `Phi_N` is `(4 pi G rhobar/3) delta_ij`,
whose traceless part is identically zero, so `I = 0`, `W = 1` - the gate's
MAXIMUM - and `a0 -> a0 (1 + A) = 17 a0 = 1.7039e-9 m/s^2`. The homogeneous
state exists, is an ordinary solution, and has the gate switched fully ON.

Two further facts decide what that means.

* **The gate's linear response is identically zero.** `|T|` is a norm of a
  quantity that vanishes in the background, so `I` is of order `|delta|`; with
  the frozen even exponent `m = 2`, `1 - W = I^2/(1+I^2)` and the measured
  log-log slope of `1 - W` against `delta` is **1.9999998** against an analytic
  2. The gate first acts at SECOND order. **At linear order the tidal-gated
  candidate is exactly AQUAL with `a0 -> 17 a0`, and the gate - the entire
  mechanism that won it the tournament - contributes nothing.**
* **But `I` is not small, so that expansion is not the relevant one.** The
  gate's argument is `|T|/T0` with `|T| ~ 4 pi G rhobar delta` and
  `rhobar ~ a^-3`. At recombination `delta = 1e-5` gives `|T| = 2.89e-33` and
  `I = 2.89`: the gate is a strongly nonlinear function of `delta` at
  `delta = 1e-5`. Its expansion parameter is `I`, not `delta`, and the two are
  unrelated in size. That is what makes the candidate's cosmology time dependent
  rather than trivial: `a0_eff` runs from **3.28 a0** at `z = 1000` to
  **17.0 a0** today, purely because `rhobar` falls.

**The two depth-gated well-network tensors do NOT admit a homogeneous
background, for three independent reasons.**

1. **`|Phi_N|` on a homogeneous background is pure gauge.** Run AH measured the
   two admissible galaxy rules differing by 0.87 dex against a 0.9 dex off/on
   margin. In cosmology it is not 0.87 dex, it is the ENTIRE RANGE of the gate.
   Under the Jeans swindle the field variable is the peculiar potential, whose
   largest value anywhere in this calculation is `2.41e11 m^2/s^2` (1000 Mpc,
   today), giving `W = 0.055`, and whose smallest is `8.5e4` (1 Mpc, at
   recombination), giving `W = 7.3e-15`. Under any rule that references an
   external potential - half `c^2` for a Hubble patch, or the potential of the
   mass inside the horizon - the argument is `2.2e15` to `4.5e16` and
   `W = 1.000000`. Both are defensible and they disagree about whether the
   mechanism exists.
2. **`S` is a normalised direction average over a DISCRETE catalogue, and a
   homogeneous background has no catalogue.** In the continuum numerator and
   denominator both vanish and `S = 0/0`, regulated only by `wellnet`'s `eps_w`.
   Measured with `wellnet.S_tensor` imported unmodified, a homogeneous universe
   represented by N wells gives a Frobenius norm of 0.8165 at N = 1 (exactly
   `sqrt(2/3)`), 0.0865 at N = 100, 0.0094 at N = 1e4 and 0.0030 at N = 1e5 - a
   fitted shot-noise slope of **-0.489** against the expected -0.5, and a factor
   **269** between the extremes. The response of the homogeneous state is set by
   how finely the cataloguer chose to chop it up.
3. **The frozen weight kernel is not normalisable.** Both surviving well
   settings are `plaw` with `q = 1, s = 2`, so `w ~ r^-2` and `Int w r^2 dr`
   grows with the catalogue radius - measured slope **+1.083** over
   `rmax/L >= 20`. `S` is normalised by that divergent sum, so on an unbounded
   homogeneous background `S = 0` IDENTICALLY, and on a finite one the response
   falls as `1/rmax` (measured slope **-1.098**).

**AQUAL, QUMOND and Newton admit the homogeneous state in the ordinary way**,
by the Jeans swindle, exactly as in Newtonian cosmology; nothing in them depends
on the value of the potential.

**And one fact that applies to every candidate except Newton: the homogeneous
state is a solution but not a UNIQUE one.** In the deep-MOND limit the growth
source is proportional to `delta^(1/2)`, which is not Lipschitz at `delta = 0`.
Both `delta == 0` and `delta = (C^2/144) t^4` satisfy the same equation with the
same initial data (zero contrast, zero velocity); the quartic branch was verified
to solve it to a relative residual of **5.4e-9**. So "linearise about the
homogeneous state and read off an eigenvalue" is NOT AN AVAILABLE OPERATION for
any candidate whose base law is AQUAL or QUMOND. What this lane computes instead
is the standard quasi-linear growth equation, which is the right object, and it
says so rather than calling it a linearisation. Newton is Lipschitz and its
trivial solution is unique.

---

## 1. Stability and growth

Primary background: standard Friedmann expansion with `Omega_m = 0.315`,
`Omega_L = 0.685`, `h = 0.674`, and ONLY BARYONS (`Omega_b = 0.0493`) sourcing
perturbations - the assumption every published MOND-cosmology calculation makes,
because a modified Poisson equation does not determine the background and no
candidate here has a relativistic completion. Two other backgrounds are carried
and the spread is in the JSON. Perturbations start at `z = 1000` with
`delta_b = 1e-5`, so reaching `delta = 1` requires an amplification of 1e5.

Each cell is `delta(z=0)` / the scale factor at which `delta` first reaches 1
(`--` means never). Pancake geometry, primary background.

| comoving scale | baryon mass inside half a wavelength | tidal scalar | S, p=0 | S, literal p=1 | AQUAL | QUMOND | Newton |
|---|---|---|---|---|---|---|---|
| 0.3 Mpc | 8.8e7 Msun | 672.6 / 0.057 | 71.9 / 0.100 | 74.5 / 0.098 | 74.5 / 0.098 | 76.4 / 0.097 | 1.18e-4 / -- |
| 1 Mpc | 3.3e9 Msun | 270.8 / 0.073 | 21.8 / 0.176 | 22.6 / 0.173 | 22.6 / 0.173 | 23.2 / 0.171 | 1.18e-4 / -- |
| 3 Mpc | 8.8e10 Msun | 105.2 / 0.097 | 7.37 / 0.302 | 7.64 / 0.296 | 7.64 / 0.296 | 7.82 / 0.293 | 1.18e-4 / -- |
| 10 Mpc | 3.3e12 Msun | 34.41 / 0.149 | 2.26 / 0.568 | 2.34 / 0.556 | 2.34 / 0.556 | 2.40 / 0.548 | 1.18e-4 / -- |
| 30 Mpc | 8.8e13 Msun | 11.94 / 0.240 | 0.775 / -- | 0.802 / -- | 0.802 / -- | 0.822 / -- | 1.18e-4 / -- |
| 100 Mpc | 3.3e15 Msun | 3.685 / 0.433 | 0.243 / -- | 0.252 / -- | 0.252 / -- | 0.258 / -- | 1.18e-4 / -- |
| 1000 Mpc | 3.3e18 Msun | 0.394 / -- | 0.0283 / -- | 0.0293 / -- | 0.0293 / -- | 0.0299 / -- | 1.18e-4 / -- |

`f = dln delta/dln a` at `z = 0`, 10 Mpc: tidal 1.177, all four others 1.143,
Newton 0.153.

Reading the table:

* **Newton without cold dark matter fails by three to four orders of
  magnitude**, at every scale and in every background - amplification 11.8 in
  the primary background, and 4.98 to 486 across all three, against the 1e5
  required. It is scale free because a Newtonian source multiplier is exactly 1
  everywhere. The control behaves as it must.
* **Every MOND-family candidate is fast enough, by a wide margin**, with
  amplifications from 509 (1000 Mpc in the EdS-expansion case) to 2.26e8.
* **The depth-gated tensors reproduce AQUAL to every printed digit.**
  `depth_S_p1_literal` is IDENTICAL to AQUAL because it shares AQUAL's fitted
  `a0`; `depth_S_p0` differs only through the 3.6% difference in its own fitted
  `a0`. The tensor contributes nothing: `S = 0` on a homogeneous continuum so
  `K = exp(A W S) = I` at zeroth order, and `delta S` is first order in `delta`
  multiplying a first-order gradient, hence second order in the source. **The
  entire well-network mechanism is invisible to a linear cosmological gate.**
* **The tidal gate is worth a factor 9.0 to 14.9 in present-day amplitude** -
  purely through `a0_eff`, not through any gate physics, since the gate has no
  linear response. It brings `delta = 1` forward by a factor 1.7 to 3.7 in scale
  factor across the mass scales where both reach it.
* **The integrator was validated against a number this lane did not compute.**
  Newton with a full matter source in LCDM gives `D(1)/a` = 0.7878126 against the
  closed form 0.7878126, relative error **2.0e-8**, and `f = 0.527` against
  `Omega_m^0.55 = 0.530`.

### The deep-MOND attractor, and what it erases

In an EdS expansion with source fraction `f = Omega_src/Omega_m`, matching powers
gives an exact self-similar solution `delta ~ a^2` on which `y = g_N/a0` is
CONSTANT:

| geometry | `Q_*` | `y_*` analytic | `y_*` measured | ratio |
|---|---|---|---|---|
| slab (pancake) | `10/(3f)` | `(3f/20)^2 = 5.4686e-5` | 5.4591e-5 | **0.9983** |
| cylinder (filament) | `40/(9f)` | `(9f/40)^2 = 1.2304e-4` | - | - |
| sphere | `4/f` | `(f/4)^2 = 1.5191e-4` | - | - |

The measured growth exponent is **2.016** against the analytic 2. **The exponent
is independent of `k`, of `a0`, of `f` and of the geometry**, so the attractor is
an amplitude attractor: a 1e4 range in the initial contrast is compressed to a
factor **1.57** by `z = 0`. On the attractor the mode amplitude is proportional
to `k` - measured `dln delta/dln k = 0.933` against an analytic 1.

**On the deep-MOND attractor the present-day amplitude of a mode is set by `k`
and `a0`, not by the primordial spectrum.** That is a single-mode scaling, not a
derived power spectrum - the equation is nonlinear and does couple modes - and it
is labelled as one.

---

## 2. The other five questions

### Does any mode grow without bound?

Three senses, three answers.

* **No finite-time blow-up.** `delta'' ~ delta^(1/2)` is SUBlinear, so the
  solution is the power law `delta ~ t^4`, not a singularity.
* **The growth RATE is bounded and nearly scale free**: `f` at `z = 0` spans
  1.734 to 2.011 over four decades of `k`, the attractor value being 2.
* **The AMPLITUDE and the response are unbounded in the ultraviolet.** The source
  multiplier diverges as `k^(1/2)` (measured log-log slope **0.499**), the mode
  amplitude grows in proportion to `k` (measured 0.955), and NOTHING IN THE
  GRAVITY LAW SUPPLIES A SMALLEST SCALE. Only baryonic pressure would, and this
  lane does not include it.

### Is the response finite at both wavelength limits?

| candidate | `Q` as `k -> 0` | `Q` as `k -> infinity` |
|---|---|---|
| Newton | 1 exactly | 1 exactly |
| tidal scalar, both tensors, AQUAL, QUMOND | **-> 1, Newtonian** | **divergent, `Q ~ k^0.499`** |

**Infrared: finite and Newtonian for every candidate**, because `g_N ~ 1/k` grows
without bound as `k -> 0`, so a MOND law is LEAST modified on the largest scales.
That is the opposite of the naive expectation and it is what makes the 1000 Mpc
rows of the growth table behave.

**Ultraviolet: divergent for everything but Newton**, at exactly the `k^(1/2)`
rate the deep-MOND scaling predicts.

**The well-network tensor is the one response in the programme that is band
limited.** Its exact linear response to a plane wave is

    delta S_ij = - delta * Jfac(k) * (zhat_i zhat_j - delta_ij/3)
    Jfac(k)    = Int w r^2 j_2(k r) dr / Int w r^2 dr

with `Jfac ~ k^2` in the infrared (measured slope **2.000**) and `~ k^-2` in the
ultraviolet (measured **-2.041**), peaking at `k L = 0.23`. That analytic form
was checked against `wellnet.S_tensor` called unmodified on a rejection-sampled
catalogue: agreement at **0.5, 1.6, 0.9 and 1.3 sigma** at `kL = 0.5, 1, 2, 4`.
But `Jfac` is proportional to `1/rmax` because the frozen kernel is not
normalisable, so its amplitude is a property of the catalogue, not of the field
equation.

### Does a preferred cosmic axis appear spontaneously?

**Locally yes, globally no - and for the well-network tensors the local axis is
set by the cataloguer.**

The linearised response about a background field is
`K_ij = pre (delta_ij + L nhat_i nhat_j)`, giving

    AQUAL   Q(theta) = 1 / [ mu(x) (1 + L_mu cos^2 theta) ]
    QUMOND  Q(theta) = nu(y) (1 + L_nu cos^2 theta)

Both give `nu` at 90 degrees and `nu/2` at 0 in the deep-MOND limit, so **modes
whose wavevector is perpendicular to the local acceleration grow fastest, by a
factor approaching exactly 2.** Measured at a cosmological configuration:
**1.9603** (AQUAL), **1.9610** (QUMOND), **1.9773** (tidal-gated, closer to 2
because its boosted `a0` puts it deeper into MOND). The two variational
references AGREE AT 0 AND 90 DEGREES AND DIFFER BY UP TO 8.2% IN BETWEEN, so
"the MOND anisotropy" is not a single number.

Cross-checked against a full nonlinear periodic solve with a uniform background
field imposed as `psi = -g.x + phi`:

| `g/a0` | numeric ratio | analytic `1 + L_mu` |
|---|---|---|
| 0.3 | 1.7659 | 1.7692 |
| 1.0 | 1.4979 | 1.5000 |
| 3.0 | 1.2489 | 1.2500 |

Worst relative error on the individual amplitudes **0.33%**.

**No global axis appears in the base laws.** The axis is the local field
direction, and a statistically isotropic field has direction average `I/3`, so
the ensemble-averaged response is exactly isotropic; the Monte-Carlo ensemble
quadrupole falls with a fitted slope of **-0.557** against the -0.5 of a random
walk.

**The two depth-gated tensors DO produce a spontaneous axis, and it is an
artefact.** A homogeneous universe represented by N discrete wells has `S` equal
to shot noise with a random direction, and `K = exp(A W S)` converts that into a
growth anisotropy:

| N wells | S Frobenius norm | anisotropy, `depth_S_p0` (A = -25) | `depth_S_p1_literal` (A = -31) |
|---|---|---|---|
| 100 | 0.0821 | **12.4** | **22.6** |
| 1 000 | 0.0275 | 2.32 | 2.84 |
| 10 000 | 0.0092 | 1.33 | 1.42 |
| 100 000 | 0.0030 | 1.095 | 1.119 |

The axis is real, order unity at any catalogue resolution a simulation would
actually use, randomly oriented, and set entirely by a number the field equation
does not specify. This is the coarse-graining objection - "a real coherence scale
must be universal and appear in the field equation; it cannot be set by the
cataloguer" - appearing in a cosmological setting, where it is sharper than in
the cluster setting because the true continuum answer is identically zero.

### Does the theory overproduce filaments or pancakes?

**No. Every MOND-family candidate does the opposite, and the tidal gate does it
hardest.**

For a symmetric collapse in `n` dimensions the source multiplier is exactly
`Q_n = nu (1 + L_nu/n)`, giving `nu/2`, `3nu/4`, `5nu/6` for pancake, filament
and sphere in the deep-MOND limit. In Newton all three are exactly 1.

| candidate | sphere/pancake at fixed g_N | filament/pancake | max sphere/pancake at fixed delta |
|---|---|---|---|
| Newton | 1.0000 | 1.0000 | 1.0000 |
| AQUAL | 1.6537 | 1.4903 | 2.1703 (z = 0) |
| QUMOND | 1.6538 | 1.4903 | 2.1704 (z = 0) |
| both depth-gated tensors | 1.6537 | 1.4903 | 2.1703 (z = 0) |
| **tidal-gated scalar** | 1.6543 | 1.4907 | **4.8964 (z = 999)** |

Analytic deep-MOND limits: 5/3 = 1.6667 and 3/2 = 1.5000 at fixed `g_N`;
`2 n^(1/4) (1 + L/n)/(1 + L)` = 2.1935 at fixed total contrast.

The tidal gate doubles the effect because the traceless tidal norm of an n-mode
isotropic superposition is `c_n * 4 pi G rhobar delta` with `c_1 = sqrt(2/3)`,
`c_2 = 1/sqrt(6)`, `c_3 = 0`: an isotropic configuration has NO tidal field, so
the gate is fully on there (`a0_eff = 17 a0`) and only partly on for a pancake
(3.29 `a0` at recombination). **The candidate preferentially boosts gravity
exactly where the collapse is isotropic and switches off where it is planar** -
anti-Zel'dovich, and a sharp qualitative prediction for any simulation that
follows.

### Does the response become statistically isotropic where local growth is directional?

**Yes, and the rate is measured.** The local response carries a quadrupole whose
Frobenius size relative to the isotropic part is **0.594** (AQUAL), **0.478**
(QUMOND) and **0.602** (tidal-gated) - large, not a perturbation. Its ensemble
mean is exactly zero for an isotropic distribution of field directions, the
Monte-Carlo residual falling as N to the power -0.557. The isotropic part of the
ensemble-averaged tensor equals the SPHERICAL geometry factor `1 + L/3` exactly,
in both variational references and to all reported digits.

One caveat with a number on it, because this programme has been bitten by it
before: **the average of a response is not the response of the average.** The
arithmetic mean of AQUAL's `K` inverts to an effective 19.08 where QUMOND's
arithmetic mean gives 21.33 - an **11%** bracket on the same physical quantity,
of the same character as the arithmetic-versus-harmonic shell-average problem
that once produced `A_T = -12.8` where the truth was -4.7. For the tidal gate the
corresponding bracket, between the tournament's own deep-MOND-calibrated rule
and the arithmetic mean over the exact chi_5 distribution of `|T|`, is only
**0.0128 dex** - so here, unlike in the cluster channel, the choice of average
does not matter.

---

## 3. Momentum, in a periodic box

No candidate conserves momentum and none has a declared carrier. In a
cosmological setting the statement has a clean form: for a potential force the
net peculiar momentum of a periodic box is

    Int rho grad psi  =  (1/4 pi G) Int (dPsi/da0) grad a0

which vanishes identically when `a0` is constant and does not when the gate makes
it a function of position. Measured, with the null taken as THE SAME BASE LAW
WITH THE RESPONSE SWITCHED OFF - never Newton, which is the mistake the
tournament had to unlearn:

| quantity | value |
|---|---|
| epoch of maximum violation | `a = 0.03`, `z = 32.3`, where `4 pi G rhobar delta ~ T0` |
| tidal-gated net force per unit mass, relative to its own RMS field, 8 realisations | **1.52e-4 +- 2.47e-4** |
| AQUAL null, same grid | **1.65e-6** |
| ratio | **92** |
| base-null convergence with grid spacing | `h^3.25` |
| tidal-violation convergence with grid spacing | **`h^0.02`, flat** |
| implied spurious bulk velocity per Hubble time, 40 Mpc box | 0.33 +- 0.54 km/s |

**The discriminator is the resolution scan.** The base null falls from 5.4e-6 at
`n = 24` to 2.6e-7 at `n = 64` while the gated violation stays at 4.0e-5 to
within 2%. It is the law, not the grid - the same test, and the same conclusion,
as the tournament's `h^2.03`.

**Two honest qualifications.** The violation exists ONLY IN THE EPOCH WHERE THE
GATE IS SWITCHING: at `a = 0.01` the gate is off everywhere and the excess over
the null is 0.97; at `a = 0.15` it is on everywhere and the excess is 1.15. And
0.33 km/s is small; because the net force is a random-phase sum it should fall as
the square root of the box volume, so this is a stochastic effect and not a
systematic bulk flow. What it is not is zero, and what it is not is numerical.

---

## 4. Verification, and the failure modes on the checklist

Six solver gates, all passing (`test_solver.py`, `solver_gates.json`):

| gate | result |
|---|---|
| G1 constant-K plane wave | order **1.99**, error 8.0e-4 at n = 64 |
| G2 anisotropic constant K, three orientations | max error 4.7e-3 |
| G3 operator symmetry, without which CG is not applicable | relative asymmetry **2.7e-15** |
| G4 discrete flux conservation | 3.5e-19 |
| G5 exact 1-D AQUAL vs full 3-D nonlinear solve | 1.4e-3 away from the cusps, order 1.9 |
| G6 momentum null for a variational law | AQUAL 2.0e-5, Newton 1.4e-18 |

**Four test bugs caught, all of the kind the brief warns about.**

1. **A Picard iteration started from `psi = 0` converges to the wrong fixed
   point.** For AQUAL `mu(0) = 0`, so `A` vanishes identically, the linear solve
   returns `psi = 0`, the Picard change is zero and the loop declares success.
   The iteration is now seeded with the Newtonian solution.
2. **The face-averaged cross-term flux is not self-adjoint** - measured at 1.9e-2
   relative asymmetry, which conjugate gradients are not entitled to. Replaced by
   `D_i(A_ij D_j psi)` with centred `D_i`, self-adjoint because the periodic
   centred difference is anti-self-adjoint and `A_ij = A_ji`. This is a
   periodic-box issue, not a criticism of `gravitylab/solver.py`, whose
   open-boundary form is the correct one there.
3. **A sign error in the background-field source term hid behind a passing
   test.** With `div(mu g zhat)` entering with the wrong sign the k-perpendicular
   case still agreed with the analytic answer to 0.14% - because the derivative
   of `mu` along the field vanishes there - while the k-parallel case was wrong
   by up to a factor 6.6 and the anisotropy ratio came out BELOW 1. Half a test
   passing is how a sign error survives.
4. **A flat error curve versus resolution meant a modelling mismatch, not a
   discretisation error.** The 1-D AQUAL comparison converged at order 0.53 in
   the max norm. That is not a bug: the peculiar acceleration behaves as
   `sqrt(|S| a0)` and so has a SQUARE-ROOT CUSP at every node of the source, so
   the deep-MOND field is continuous but not differentiable there and no
   discretisation can be second order in the max norm. Away from the cusps the
   order is 1.9. Worth recording as physics: the deep-MOND field is not smooth.

**Shared-denominator artefacts.** Checked, and largely absent by construction:
the growth ratios compare two solves of the same source, and the anisotropy ratio
is a ratio of two amplitudes at the same `k` and the same `delta`, so the common
factor cancels exactly rather than appearing on both axes. The one place the
artefact could surface is the geometry comparison, where `Q_n` and `|T|` are both
functions of `delta` - so both columns are reported, `Q` at fixed `g_N` (which
holds the gate at its one-dimensional value and isolates the purely geometric
factor) and `Q` at fixed `delta`, and the difference between them IS the shared
dependence.

**Monotone-invariant statistics.** Every headline statistic was pushed through
`linear_response.responsiveness`, which verifies that the derivative with respect
to the parameter is non-zero over the tested range and prints the spread. All
seven are responsive:

| statistic | parameter and range | distinct values | spread | relative spread |
|---|---|---|---|---|
| `delta(z=0)` | `a0`, one decade either side | 9/9 | 22.35 | 9.55 |
| `delta(z=0)` | gate amplitude `A`, 0 to 32 | 9/9 | 61.88 | 1.80 |
| `delta(z=0)` | `T0`, three decades either side | 9/9 | 31.73 | 0.92 |
| `delta(z=0)` | comoving scale, 1 to 1000 Mpc | 9/9 | 22.57 | 29.6 |
| anisotropy ratio | `g_N/a0`, eight decades | 12/13 | 0.990 | 0.702 |
| `a0` boost | tidal RMS, six decades | 13/13 | 16.00 | 1.66 |
| well form factor | `k L`, four decades | 13/13 | 0.2081 | 43.5 |

The anisotropy ratio is the only one with a flat stretch, and it is flat exactly
where it should be: the Newtonian limit, where the ratio is 1 by construction.

**Refitting on a held-out set** cannot arise - nothing is fitted here. **Silent
extraction failures**: every frozen constant is quoted with its source row and
echoed back in the JSON, and the three imported modules' SHA-256s match the
tournament's recorded hashes. **Non-monotonic profiles and clipped outer
slopes**: not applicable, no deprojection is performed.

Two identities were verified numerically rather than assumed. The traceless tidal
norm of a statistically isotropic Gaussian density field has mean square exactly
two thirds of `(4 pi G rhobar sigma)^2` - Monte Carlo gives 0.6689 against
0.66667, a relative error of **3.4e-3** - which is what makes the chi_5 gate
average well defined. And the analytic well-network form factor was reproduced by
`wellnet.S_tensor` at four wavenumbers, within 1.6 sigma everywhere.

**Ellipticity.** Every candidate's response tensor is positive definite over
eight decades of its argument, so no candidate is ill-posed and none has a
catastrophic short-wavelength instability. But the two tensors are close to it:
at a unit-norm `S`, `K = exp(A S)` has condition number **4.0e8**
(`depth_S_p0`, A = -25) and **4.6e10** (`depth_S_p1_literal`, A = -31). The
tournament recorded the same thing, and it is a statement about how extreme the
fitted amplitudes are.

---

## 5. What a linear gate cannot decide

This lane is a screen, not a cosmology. It decides less than its numbers might
suggest, and the boundary should be explicit.

1. **It cannot decide the background expansion.** A modified Poisson equation
   determines the peculiar potential for slow matter; it says nothing about
   `H(a)`. The background here is assumed, and the assumption matters: the
   present-day amplitude at 10 Mpc moves by a factor **130** across the three
   backgrounds carried (0.364, 2.34, 47.4 for AQUAL). Fixing it needs the
   covariant action nobody has written for any of these candidates.
2. **It cannot decide the initial conditions.** `delta_b = 1e-5` at `z = 1000` is
   an input. In a baryon-only universe the pre-recombination physics - acoustic
   driving, Silk damping, the baryon-photon sound speed - is exactly what sets
   it, and none of that is here. The attractor partially rescues this (a 1e4
   range compresses to 1.57) but only for modes that reach it.
3. **It cannot decide the power spectrum.** The attractor scaling is a
   single-mode result. The equations are nonlinear at every amplitude, they do
   couple modes, and the harmonic generation is real: a pure cosine source does
   not stay a cosine, because the deep-MOND response develops square-root cusps
   at the source nodes.
4. **It cannot decide the CMB, the acoustic peaks, or lensing.** Those need both
   metric potentials, a photon sector and a lensing closure. A dynamics law fixes
   neither the slip nor what photons do; that is a separate lane and the brief
   says so.
5. **It cannot decide whether the momentum violation matters.** It measures the
   violation and shows it is physics rather than grid (`h^0.02` against
   `h^3.25`), but turning 0.33 km/s per Hubble time in a 40 Mpc box into a
   statement about real bulk flows needs the volume scaling of a random-phase sum
   over the real spectrum, which needs item 3.
6. **It cannot decide anything about the depth-gated tensors' cosmology at all**,
   because they do not have one until somebody states an operational boundary
   rule for `|Phi_N|` and a universal coherence scale for `S`, and the two
   admissible families of rule disagree about whether the mechanism exists.
7. **It cannot validate itself past `delta ~ 1`.** In the primary background the
   tidal-gated candidate reaches `delta = 1` at 100 Mpc by `a = 0.43`, so the
   quasi-linear treatment has expired on those scales well before `z = 0` and the
   `z = 0` entries in its row are extrapolations of a calculation that has left
   its regime. The same is true of AQUAL and QUMOND below about 30 Mpc.
8. **For the tensors, the neglected second-order term is not small.**
   `K = exp(A W delta S)` has exponent `|A| Jfac(k) delta`, which reaches 1 at
   `delta = 0.21` for a 6 Mpc catalogue radius at a 10 Mpc mode and at
   `delta = 111` for a 3000 Mpc one. **How big the term this lane throws away is
   depends on the catalogue radius, not on the field equation** - a factor 500
   across the range tabulated.
9. **The tidal gate's argument is set by a smoothing scale the theory does not
   supply.** `|T|` at a point is fixed by the TOTAL variance of the density
   field, and on the attractor that variance is ultraviolet dominated. Taking
   this candidate's own predicted spectrum, the boost `a0_eff/a0` runs 17.0 to
   11.2 as the smoothing goes from 300 Mpc to 0.1 Mpc - and linear theory is only
   self-consistent (variance below 1) at the 300 Mpc end.

---

## 6. Verdict

**Nothing here promotes or eliminates a candidate on its own, but three results
should change what the next lane spends compute on.**

* **The mechanism that survived the tournament does no work in linear structure
  formation.** The tidal gate's linear response is identically zero - measured
  slope 2.0000 for `1 - W` against `delta` - and the well-network tensor's is
  zero twice over, once because `S = 0` on a homogeneous continuum and again
  because its frozen kernel is not normalisable. What survives of the tidal-gated
  candidate at linear order is **AQUAL with `a0` multiplied by a time-dependent
  factor running from 3.28 at recombination to 17.0 today**. A cosmological
  simulation of it would, to linear order, be a simulation of MOND with a running
  `a0`, and should be costed as such.
* **The homogeneous state is not a state you can linearise about.** For every
  candidate except Newton the growth source is non-Lipschitz at `delta = 0`, so
  the trivial solution is not isolated. Any claim that the law helped matter
  self-organise must be made about the quasi-linear or nonlinear problem; the
  linear-eigenvalue version of the question has no answer.
* **The one axis that appears spontaneously is an artefact.** The base laws'
  anisotropy is local, order unity (quadrupole over isotropic 0.48 to 0.60),
  aligned with the local field, and exactly isotropic on ensemble average. The
  depth-gated tensors add a GLOBAL random axis of amplitude 12.4 at 100 wells and
  1.10 at 1e5 wells, set by catalogue resolution and vanishing in the continuum.
  **A simulation that discretises the well network will measure a cosmic
  anisotropy that is a property of its own particle count.**

**Recommended before any cosmological simulation.** (1) A boundary rule for
`|Phi_N|` and a universal coherence scale for `S`, declared in advance - without
them the two tensor candidates have no cosmology to simulate. (2) A declared
momentum carrier or a variational completion; the violation is now measured to be
grid independent at `h^0.02` and 92 times its own null. (3) A decision about the
ultraviolet: the response diverges as `k^(1/2)` with no cutoff in the gravity
law, so a simulation's smallest resolved scale will set both the smallest
structures and, through `|T|`, the value of the tidal gate everywhere else.
"""

if __name__ == "__main__":
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "REPORT.md")
    with io.open(p, "w", encoding="utf-8", newline="\n") as f:
        f.write(BODY)
    print("wrote", p, len(BODY), "chars")
