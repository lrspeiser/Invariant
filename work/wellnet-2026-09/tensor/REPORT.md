# Anisotropic response tensors: implementation, gates, and how large an effect they can make

Lane: `work/wellnet-2026-09/tensor/`  -  2026-09-03
Windows 11, Python 3.13.5, NumPy 2.2.6, SciPy 1.16.1, CuPy 13.5.1 on an
RTX 5090 (compute capability 120, 32 GB). Every tensor evaluation and every
field solve here ran on the GPU in float64.

---

## 0. The short answer

**The well-network tensor can reach the cluster amplitude. The pair-channel
tensor can reach the amplitude too, but only with a radial profile that is
wrong by a large factor, and it is killed outright by a constraint the brief
did not name: galaxies that live *inside* the cluster.**

`B` is defined as `|g|(tensor) / |g|(K = I)` for the same source and the same
mu, so `B` is exactly the extra factor on top of what MOND already gives.
Synthetic A2029: the real X-COP baryon profile plus 300 members.

| requirement, applied in sequence | well network | pair channels |
|---|---|---|
| points scanned | 1920 | 288 |
| reach B = 2.0 at 1 Mpc | 1541 | 140 |
| ... and leave a **field** galaxy inside 0.04 dex | 760 | 140 |
| ... and leave a **cluster member** galaxy inside 0.04 dex | 563 | **0** |
| ... and a flat profile, 0.7 < B(1414)/B(300) < 1.4 | 291 | 0 |
| ... and the whole B(r) profile inside 1.6 - 2.5 | **127** | 0 |
| best RMS against lane 12's measured radial requirement | **0.033 dex** | 0.077 dex |
| ... points within 0.10 dex of it that also keep both galaxies | **1** | **0** |

Two structural results matter more than the counts.

> **Every viable point uses a potential-depth gate. Not one uses an
> acceleration gate.** A cluster at 0.3-1.4 Mpc and a galaxy outskirt sit at the
> *same* `g_N/a0` - that is what the RAR says - so no function of `g_N/a0` can
> separate them. Only `|Phi_N|` can, and it has to be steep: 126 of the 127
> survivors need gate exponent m >= 2 at `Phi_0 = 1e12` m^2/s^2, i.e. a switch
> that turns on at velocity dispersions near 1000 km/s.

> A cluster member galaxy sits at `|Phi_N| = 1.09e12` m^2/s^2, **deeper than the
> cluster's own 1 Mpc shell at 7.22e11**, and has all 44,850 pairs threading it.
> Anything that switches on with potential depth or with pair density switches
> on hardest *inside* cluster galaxies - exactly where the fundamental plane
> says nothing is happening. This is the binding constraint for both tensors
> and it is where tensor 2 dies.

---

## 1. What was built

| file | what it is |
|---|---|
| `wellnet.py` | Tensor 1. Closed-form symmetric-3x3 spectral algebra, three weight families, `S^ij`, `K = exp[s_0 I + s_T S]`, the environmental gates, and an exact continuum quadrature used by the isotropy gate. |
| `channels.py` | Tensor 2. Pair construction, `w_ab`, three `d_par` conventions, a CuPy `RawKernel` for `C^ij(x)`, a NumPy reference, `K = exp[+-alpha C]`. |
| `field.py` | Nonlinear solver: backend-generic copy of `gravitylab/solver.py`'s stencil, Jacobi-preconditioned CG, the mu(X) Picard iteration, the exact 1-D spherical reduction and the Dirichlet shell built from it. |
| `cluster.py` | Synthetic A2029 (real X-COP baryons + 300 statistical members) and the field-galaxy control. |
| `run.py` | Analytic Newtonian potential of the model source, k(r) profiles, full solves, boosts. |
| `test_gates_wellnet.py` | Seven mandatory gates plus five auxiliary checks -> `gates.json`. |
| `mechanism.py` | The parameter map -> `mechanism_map.json`. |
| `calib.py` | Which shell average of k reproduces the 3-D solve -> `calibration.json`. |
| `seeds.py` | Realisation-to-realisation scatter -> `seed_robustness.json`. |
| `summarise.py` | `mechanism_map.json` -> the tables here (`summary.txt`). |
| `a2029_baryons.npz` + `.manifest.json` | Cached X-COP A2029 baryonic mass profile with provenance. |

`work/gravitylab/solver.py` was **not modified**. Gate A2 checks that this
lane's operator, face fluxes and Jacobi diagonal reproduce it *bit-identically*
(max absolute difference 0.000e+00 on random inputs), and that the CuPy path
reproduces the NumPy path bit-identically too.

---

## 2. The field equation and how it is solved

    div_i [ mu(X) K^ij(x) grad_j Phi ] = 4 pi G rho_b
    X = sqrt( grad_i Phi K^ij grad_j Phi ) / a0,      g = -grad Phi

* **Discretisation** identical to `gravitylab/solver.py`: face fluxes with
  explicit forward differences on the normal component and face-averaged
  centred derivatives on the transverse components, never `np.roll`.
* **Linear solver:** Jacobi-preconditioned CG. The diagonal of the FV operator
  is exactly minus the sum of the six face conductivities over h^2 - the cross
  terms use centred derivatives and carry no self-coefficient at interior cells
  - verified against `A e_ijk` on random cells to 3.6e-15.
* **Nonlinearity:** lagged-diffusivity Picard on `A = mu(X) K`, under-relaxation
  0.75. Typically 7-12 outer iterations to `max|dPhi|/max|Phi| < 1e-5`.
* **mu:** `X/(1+X)` throughout; `mu = 1` for the Newtonian gate;
  `X/sqrt(1+X^2)` also implemented. A floor `mu >= 1e-4` keeps the operator
  non-singular: `mu(0) = 0` exactly and `grad Phi` vanishes at the centre of a
  symmetric source, so without the floor the operator has an all-zero row and
  CG stalls. That was a real failure here, not a hypothetical one - the first
  cluster solves returned NaN from exactly that.
* **Boundary condition:** open Dirichlet, but *not* Newtonian. At 1.4 Mpc an
  A2029-like cluster sits at `g_b = 0.074 a0`, deep in the MOND regime where
  Phi grows logarithmically. The shell values come from the spherical reduction
  of the same equation using the model's own k(r),

      mu( sqrt(k) |Phi'| / a0 ) k |Phi'| = G M(<r) / r^2,

  which for a statistically spherical well distribution is *exact*: spherical
  symmetry forces `K = a(r) I + b(r)(rhat rhat^T - I/3)`, whose radial
  eigenvalue is that k. Only an additive constant in Phi is arbitrary, and
  constants are annihilated by the operator.

  A bug worth recording: the k profile was first binned on 400 radial bins over
  a grid whose cells are 94 kpc wide, so inner bins were empty, `k = 0`
  propagated into `1/sqrt(k)`, and the whole field came back NaN. Empty bins
  are now filled by nearest-value interpolation.

---

## 3. Tensor 1 - well alignment

    n_a(x)  = (x_a - x)/|x_a - x|
    S^ij(x) = [ sum_a w_a (n_a^i n_a^j - delta^ij/3) ] / [ eps + sum_a |w_a| ]
    K(x)    = exp[ s_0(x) I + s_T(x) S(x) ]

    plaw     w_a = (M_a/M_0)^p [1 + (r_a/L)^q]^-s
    expo     w_a = (M_a/M_0)^p exp[-(r_a/L)^q]
    gscreen  w_a = (M_a/M_0)^p / { [1+(g_N/a0)^m] [1+(r_a/L)^q]^s }

**Two things the brief leaves open, stated rather than buried.**

1. **What s_0 and s_T are.** `Phi_0` appears in the global list and nowhere else
   in the construction, so the reading taken here is

       s_0(x) = A_0 g(x),   s_T(x) = A_T g(x)
       gate "none": g = 1
       gate "phi" : g = u^m/(1+u^m),  u = |Phi_N(x)|/Phi_0
       gate "gn"  : g = 1/(1 + (|g_N(x)|/a0)^m)

   with Phi_N and g_N the *Newtonian* potential and field of the baryons, so K
   is a functional of the source alone and adds no second nonlinearity. The
   exponent m is reused from the third weight family; with m = 1 the phi gate
   saturates at a contrast of about `|Phi_cluster|/|Phi_galaxy| = 60`, which
   turns out not to be enough. **All of the cluster-versus-galaxy
   discrimination lives in this choice.** S itself cannot discriminate, because
   it is normalised and therefore scale-free: doubling every mass, or doubling
   the number of wells at fixed geometry, does not change S at all.

2. **Whose `g_N` appears in the third family.** Both readings are implemented:
   `gn_mode="pair"` (`g_N = G M_a / r_a^2`, the well's own field, a genuine
   per-well reweighting - the default), and `gn_mode="local"` (the total
   baryonic field at the point, which is a *common* factor on every weight and
   therefore cancels out of S exactly except through the eps regulariser, i.e.
   it acts as an on/off switch for the whole tensor and not as a reweighting).

**Self-inclusion.** The formula has no self-exclusion, so by default every well
counts everywhere. That has a consequence which dominates the galaxy answer: a
point 20 kpc from a galaxy centre has that galaxy as its nearest well, and with
any steeply falling w_a that one well dominates the sum and drives |S| to near
its maximum. Measured: at `A_T = -4.7` with no gate - the amplitude that gives
B = 2 in the cluster - an isolated 5e10 Msun disc is boosted by **1.169 dex, a
factor 14.7**, at 10 kpc. `exclude_nearest=True` (drop the single nearest well)
is implemented as the alternative and both are mapped; 47 of the 127 final
survivors use the brief's literal, self-inclusive formula.

**Matrix exponential.** Not a generic `expm`. Eigenvalues from the closed-form
trigonometric solution of the characteristic polynomial; the exponential
assembled by Sylvester / Newton divided differences,

    exp(M) = f[l0] I + f[l0,l1](M - l0 I) + f[l0,l1,l2](M - l0 I)(M - l1 I)

so no eigenvectors are needed and the whole thing vectorises over a 128^3 grid.
The first two divided differences use `expm1` and are stable at any eigenvalue
spacing; the second switches to its confluent limit only when all three
eigenvalues coincide to 1e-4 of the spectral scale, where the matrix factor it
multiplies is itself of order 1e-8. Gate A1 checks it against
`scipy.linalg.expm` on 1,000 matrices including exactly-degenerate and
near-degenerate spectra (gaps 0, 1e-12, 1e-8, 1e-5, 1e-3): **worst relative
error 2.4e-12**.

---

## 4. Tensor 2 - pair channels

    e_ab    = (x_b - x_a)/|x_b - x_a|
    W_ab(x) = exp[-d_perp^2/(2 sigma_perp^2)] exp[-d_par^2/(2 sigma_par^2)]
    C^ij(x) = sum_{a<b} w_ab W_ab(x) e_ab^i e_ab^j
    w_ab    = (M_a M_b/M_0^2)^p (d_ab/L)^-q exp[-(d_ab/L)^s]
    K(x)    = exp[ sign * alpha * C(x) ]

**d_par, defined precisely.** With no d_par factor the tube is an infinite
cylinder and every pair contributes everywhere - 44,850 pairs would paint the
whole box. Three conventions, all implemented and named:

* `mode="clip"` - **the default, used for every number in this report.** With
  `t = (x - m_ab) . e_ab` measured from the pair midpoint,
  `d_par = max(0, |t| - d_ab/2)`. The tube is a capsule: flat along the segment
  that actually joins the two wells, Gaussian beyond each end. This is the
  object the model describes - a channel *between* two wells - and it is
  scale-correct: a pair 2 Mpc apart gets a 2 Mpc channel, not a
  sigma_par-sized blob.
* `mode="mid"` - `d_par = |t|`, Gaussian from the midpoint. A widely separated
  pair then contributes nothing at either of its own wells unless
  `sigma_par > d_ab/2`, which makes the tensor depend on sigma_par far more
  strongly than on the geometry. Scanned for comparison; it never does better.
* `mode="line"` - `d_par = 0`, the failure mode named above, implemented only
  so it can be exhibited.

**The sign convention, settled by measurement (gate A4).** C is positive
semi-definite, so `e^T C e > 0` along a channel. The intuition "a poorer
conductor needs a larger gradient to carry the same flux, so `exp[-alpha C]`
with `alpha > 0` strengthens gravity along the channel" is **wrong**. For a
uniform `K = diag(k_par, 1, 1)` the exact monopole
`Psi = -GM/(sqrt(det K) sqrt(r^T K^-1 r))` gives
`|g|_along / |g|_across = sqrt(k_par/k_perp)` at equal radius, which the solver
reproduces to 1.5% at k_par = 0.4 and 2.5. Lowering the conductivity along an
axis therefore *weakens* |g| along it: flux is diverted into the
better-conducting transverse directions faster than the extra resistance raises
the gradient. On the two-well tube, each run normalised by the same source at
K = I:

| K | along / across, relative to K = I |
|---|---|
| `exp[-alpha C]`, alpha > 0 (the brief's sign) | **0.9914** - weaker along the channel |
| `exp[+alpha C]` | **1.0072** - stronger along the channel |

> **`sign = +1`, i.e. `K = exp[+alpha C]`, is the sign that makes the response
> STRONGER ALONG the lines joining wells. `sign = -1`, the brief's
> `exp[-alpha C]`, makes it stronger TRANSVERSE to them.** Both are implemented
> and both are scanned.

The naive argument is right only where flux cannot be diverted - a radially
aligned K in spherical symmetry, where |g| goes as `k_rad^-1` (Newtonian) to
`k_rad^-3/4` (deep MOND). That is the regime the *shell-averaged* boost lives
in, which is why `sign = -1` is nevertheless the only sign that raises the
shell-averaged cluster field: `exp[-alpha C] <= I` in every direction, so radial
conduction drops wherever C has support. `sign = +1` gives B <= 1 at every one
of the 144 channel parameter settings - it can only *weaken* cluster gravity.
Both facts are true and they are not in conflict.

**GPU and the cutoff.** `C^ij` at every grid point is a CuPy `RawKernel`, one
thread per cell, looping the pair list in registers with a bounding-sphere
early-out. 44,850 pairs x 128^3 cells = 9.4e10 tube evaluations in **2.0 s**
(0.32 s at 64^3). No spatial hash was needed. Cutoff cost measured, not assumed,
against a no-cutoff reference:

| cutoff | max error / peak of C | median per-point relative error |
|---|---|---|
| 3 sigma | 8.3e-3 | 1.3e-1 |
| 4 sigma | 2.5e-4 | 3.0e-3 |
| 5 sigma | 2.5e-6 | 3.2e-5 |
| **6 sigma (default)** | **9.8e-9** | **1.5e-7** |

CuPy kernel vs NumPy reference with the cutoff disabled: **2.1e-15**.

---

## 5. Validation gates

`python test_gates_wellnet.py` -> `gates.json`. **13 of 13 pass.**

| # | gate | measured | criterion |
|---|---|---|---|
| 1 | K symmetric positive definite everywhere, whole search range | well network min eigenvalue **1.83e-2**, max condition number 4.03e2, over 108 weight settings x 5 amplitudes x 110,592 cells; pair channels min eigenvalue **5.93e-7**, condition 5.17e1; numpy eigvalsh cross-check on 4,000 cells 2.29e-2 with asymmetry **exactly 0**; eigenvalues of S confined to [-1/3, +2/3] as the traceless normalised form requires (measured -0.3333 to +0.6666) | min eig > 0 |
| 2 | flux conservation on the FACE fluxes | worst closed surface **3.29e-14**, four nested surfaces, both tensors | < 1e-5 |
| 3 | curl(g) | **4.82e-17**, flat in resolution | round-off |
| 4 | Newtonian recovery, K -> I and mu -> 1 | **2.30e-4** at order **1.99**; constant anisotropic K against the exact monopole **3.57e-4** at order **1.99** | 2nd order |
| 5 | isotropy | see below | S -> 0 |
| 6a | grid-resolution convergence | 64 -> 96 change **1.47%**; self-convergence order 1.25 / 2.05 at 500 / 1000 kpc; the curve is not flat | < 2% |
| 6b | domain-size stability | 4 Mpc -> 8 Mpc at fixed h: **0.357%** | < 2% |
| 7 | source-label permutation invariance | S 8.5e-16, C 1.3e-14, full nonlinear field magnitude **3.3e-14** | < 1e-12 |
| A1 | closed-form expm vs scipy.linalg.expm | worst **2.4e-12** over 1,000 matrices including degenerate spectra; analytic eigenvalues 2.1e-9 | < 1e-9 |
| A2 | operator vs gravitylab/solver.py | **bit-identical, 0.0**; GPU vs CPU **0.0**; Jacobi diagonal 3.6e-15 | identical |
| A3 | GPU channel kernel and cutoff cost | GPU vs CPU **2.1e-15**; 6-sigma cutoff costs 9.8e-9 | measured |
| A4 | sign convention | analytic anchor reproduced to **1.5%**; direction measured, not asserted | consistent |
| A5 | 1-D surrogate vs full 3-D | worst **5.8%** with the harmonic mean, 15.2% with the arithmetic mean | stated |

### Gate 5, the most diagnostic test

Read literally, "a spherically symmetric well distribution must give S -> 0" is
**true at the centre and false everywhere else**, and the difference matters.
Both were measured.

*(a) Exact symmetry.* 1,440 wells built as the full octahedral orbit (48 images)
of 30 random seeds, field point at the centre. For all three weight families:

    |S| at the centre       4.2e-17 .. 5.9e-17   (round-off)
    |K - exp(s_0) I|_max    4.4e-16 .. 8.9e-16   (isotropic rescaling, no axis)

*(b) Off centre.* 2e6 wells uniform in a 1.5 Mpc ball, probes on the x axis,
against a 2-D Gauss-Legendre quadrature of the exact continuum integral for the
same distribution (wellnet.S_rr_continuum):

| r (kpc) | S_rr Monte-Carlo | S_rr exact quadrature | difference |
|---|---|---|---|
| 0 | -0.000432 | +0.000000 | 4.3e-4 |
| 100 | -0.000133 | +0.000016 | 1.5e-4 |
| 300 | +0.000460 | +0.000159 | 3.0e-4 |
| 600 | +0.001638 | +0.000844 | 7.9e-4 |
| 900 | +0.003610 | +0.003320 | 2.9e-4 |
| 1200 | +0.017115 | +0.016479 | 6.4e-4 |
| 2500 | +0.515068 | +0.515045 | 2.3e-5 |

Monte-Carlo noise is 1/sqrt(N) = 7.1e-4; the largest discrepancy is 1.1 sigma.
S_rr is genuinely non-zero off centre and must be, because the direction field
n_a(x) seen from an off-centre point is not isotropic. What symmetry requires is
that S keep the form lambda(r) (rhat rhat^T - I/3), and the largest residual
from that form, over the probes where |S| rises above the Monte-Carlo floor, is
**0.044**.

An earlier version of this gate compared the form residual to |S| at every
radius and "failed" at 0.99. But at small radius |S| **is** the Monte-Carlo
noise, so the test was measuring nothing. Another test bug that looks like a
solver bug.

---

## 6. The synthetic cluster

* **Smooth component:** the real X-COP A2029 baryonic mass profile,
  M_b(<r) = g_b r^2 / G from the bench's *baryonic* acceleration column, which
  is an observable-derived baryon profile (X-ray gas plus stars), not a
  dark-matter-inferred mass. 38 rows, 125 to 1644 kpc, identifier echoed back as
  extent = 1414 kpc (= R500). M_b(<1644 kpc) = 1.52e14 Msun,
  M_b(<R500) = 1.44e14 Msun. numpy.maximum.accumulate enforces monotone M(r).
  Cached with a SHA-256 manifest.
* **Outward continuation:** the fitted outer power law times a Gaussian taper of
  800 kpc scale. A bare rho ~ r^-2 continued to the box corner would add
  1.2e14 Msun of invented gas, more than the measured M_b(<R500). With the
  taper M_tot = 2.17e14 Msun.
* **Members:** 300 galaxies holding 15% of the baryons, masses resampled from
  the AXES luminosity function of the 140 brightest group members measured
  earlier in this programme, positions drawn from the gas mass profile,
  isotropic, cloud-in-cell deposited. **A statistical population, not the actual
  A2029 catalogue** - deliberately the same prescription the earlier
  QUMOND-lumpiness calculation used, so the two are comparable. 44,850 pairs.
* **Regime:** g_b/a0 = 0.183, 0.149, 0.098, 0.074 at 300, 500, 1000, 1414 kpc,
  deep MOND everywhere of interest. Baseline K = I solution
  |g|/a0 = 0.521, 0.462, 0.364, 0.312.
* **Controls:** a field galaxy (5e10 Msun exponential disc, 24 neighbours within
  5 Mpc) and a *member* galaxy (the most massive member near 500 kpc: 4.0e11
  Msun sitting at 384 kpc from the cluster centre).

Newtonian potentials are computed analytically as (spherical gas profile) plus
(members as softened point masses) rather than from the grid Poisson solve,
because the gate has to be evaluated 10 kpc from a galaxy centre as well as
1 Mpc from a cluster centre and a 94 kpc cell cannot represent the first.

    |Phi_N|   field galaxy at 20 kpc    1.13e10 m^2/s^2
              cluster 1 Mpc shell       7.22e11
              member galaxy at 20 kpc   1.09e12    <- the deepest of the three

---

## 7. How the map is computed, and why the choice of average is physics

Each parameter point gives k(r) = <rhat^T K rhat> on the four shells, pushed
through the exact spherical reduction. Two exact simplifications make a
2,208-point scan cheap: k is only needed on the shells (12,000 cells instead of
262,144), and A_0 factors out of the exponential.

**Which shell average is a real question.** Once |A_T| is large, k varies by
orders of magnitude across a shell and the candidate averages disagree badly.
calib.py measures three of them against the full nonlinear 3-D solve at
A_T = -1 to -8 (calibration.json):

| A_T | 3-D B(1000 kpc) | arithmetic mean of k | harmonic mean of k | cell-wise mean of B(k) |
|---|---|---|---|---|
| -1 | 1.150 | 1.140 | 1.145 | 1.145 |
| -2 | 1.331 | 1.287 | 1.310 | 1.307 |
| -3 | 1.546 | 1.438 | 1.494 | 1.488 |
| -4.5 | 1.941 | 1.659 | 1.802 | 1.788 |
| -6 | 2.432 | 1.855 | 2.134 | 2.108 |
| -8 | 3.230 | 2.043 | 2.573 | 2.523 |

Worst departure over all radii and amplitudes: **arithmetic 46.9%, harmonic
20.4%, cell-wise 22.4%.** The map uses the **harmonic** mean; both are stored so
the bracket is visible.

The arithmetic mean is not merely less accurate, it is qualitatively wrong: as
A_T goes to minus infinity the shell cells with S_rr < 0 blow up and drag the
mean to infinity, so the predicted boost turns over and appears to **saturate at
B = 2.1**. A first version of this map reported exactly that, and reported that
A_T = -12.8 was needed for B = 2. Both were artefacts of the average. The
correct answer is that the ungated tensor reaches B = 2 at 1 Mpc at
**A_T = -4.7**, with no saturation.

The surrogate always **under**-predicts, so the amplitudes the map reports are
upper limits on what is needed and the galaxy damage it reports is an
over-estimate. Selected points are re-solved in full 3-D anyway (section 9).

---

## 8. The mechanism map

`mechanism_map.json`: 1,920 well-network rows (3 families x p x q x s x L x
self-exclusion x 8 gates, each scanned over 141 amplitudes) and 288 pair-channel
rows (2 d_par modes x sigma_perp x sigma_par x q x p x L x 2 signs, 110
amplitudes). Full tables in `summary.txt`.

### 8.1 Amplitude: both tensors reach B = 2 easily, and it is not shot noise

This is the first thing that separates the result from the earlier
QUMOND-on-lumpiness calculation, which found a shell-averaged difference of
**0.4%** and a projected-deflection difference of about 1% (2.7% at 150 kpc),
because a smoother member population makes that effect vanish. Here:

* the well-network S is **normalised by the total weight**, hence scale-free: it
  does not vanish as the population is smoothed, and its eigenvalues are bounded
  in [-1/3, 2/3] by geometry alone. One global A_T then buys any amplitude.
  Ungated, A_T = -4.7 gives B = 2 at 1 Mpc; A_T = -8 gives 3.2.
* the pair-channel C is **not** normalised, so its magnitude scales with the
  number of pairs times the weights: tr C reaches 3.1e3 in the cluster against
  1.7e-3 at the field galaxy, a contrast of **1.8 million**. Any alpha at all
  then buys an enormous cluster-versus-field-galaxy contrast for free.

Measured directly on the configuration where QUMOND gave 0.4%: the *lumpiness*
contribution to the well-network boost is **0.2 to 0.3%** (B_full / B_smooth =
0.997 to 0.998, where B_smooth replaces S_rr by its shell mean before
exponentiating). **The boost is smooth geometry, not shot noise** - the opposite
of the earlier lumpiness finding, and confirmed by the realisation scatter in
section 9 (5 to 11% across five independent draws of the 300 members, against an
effect of a factor 2).

### 8.2 Amplitude with the galaxy-scale limit attached

| gate | rows | reach B=2 | field galaxy ok | + member galaxy ok |
|---|---|---|---|---|
| none | 240 | 196 | 2 | 1 |
| gn, m=1 | 240 | 196 | 1 | 1 |
| gn, m=2 | 240 | 196 | 1 | 1 |
| phi, Phi_0=1e11, m=1 | 240 | 196 | 29 | 1 |
| phi, Phi_0=1e12, m=1 | 240 | 196 | 166 | 156 |
| phi, Phi_0=1e12, m=2 | 240 | 196 | **196** | 160 |
| phi, Phi_0=1e12, m=4 | 240 | 196 | **196** | 112 |
| phi, Phi_0=3e12, m=2 | 240 | 169 | 169 | 131 |

Ungated, the tensor is a catastrophe at galaxy scale: the same A_T = -4.7 that
gives a factor 2 in the cluster gives **1.169 dex, a factor 14.7**, inside an
isolated 5e10 Msun disc at 10 kpc, because the galaxy's own well dominates S
there. The acceleration gate does essentially nothing, and that is not a
numerical accident but the RAR itself: a cluster at 1 Mpc and a galaxy at 20 kpc
both sit near g_N of order 0.1 to 1 a0. Only the potential-depth gate separates
them.

For the pair channels the field galaxy is trivially safe (best violation
1.3e-10 dex, because it has almost no pairs) but the **member** galaxy is not:
the best member violation over all 288 rows is **0.687 dex, a factor 4.9**. The
same pair density that makes the cluster work threads straight through its
galaxies.

### 8.3 Radial shape, where the two tensors part company

* **Against a flat target** (B about 2 across 300 to 1414 kpc, matching X-COP's
  nu/nu_RAR = 2.53 for A2029): the well-network survivors are excellent, e.g.
  B = 1.62, 1.99, 2.00, 1.65 at 300/500/1000/1414 kpc, shape 1.02. The pair
  channels are hopeless: the flattest of 288 rows is
  B = 3.40, 2.12, 2.00, 1.09, shape 0.32, because C follows the *square* of the
  member density and is far more centrally concentrated than the mass.
  **127 well-network points survive everything; 0 channel points do.**

* **Against the programme's own measured radial requirement** (lane 12: the
  cluster a0 enhancement from lensing alone, 21.95x at 0.073 R500 falling as
  (r/R500)^-1.354 to 1.19x at 1.5 R500, reproduced by three samples sharing no
  clusters and no pipeline). In deep MOND g = sqrt(g_N a0), so the required
  *field* boost is B = sqrt(A):

      required B at 300, 500, 1000, 1414 kpc  =  3.86, 3.33, 2.40, 1.76

  a steeply **declining** profile, which changes the comparison:

  | | best RMS vs lane 12 | rows within 0.10 dex | ... and both galaxies ok |
  |---|---|---|---|
  | well network | **0.033 dex** | 117 | **1** |
  | pair channels | 0.077 dex | 3 | **0** |

  The well network still wins on RMS, but the points that match the shape need
  A_T between -25 and -31, and at that amplitude the *member* galaxy is boosted
  by 0.13 to 0.37 dex. Exactly one scanned point threads both needles - plaw,
  p=0, q=1, s=2, L=300 kpc, no self-exclusion, phi gate Phi_0=1e12 m=4,
  A_T = -24.7 - giving B = 2.51, 3.22, 2.57, 1.98 at RMS 0.099 dex with a member
  violation of 0.015 dex, and it sits on the boundary of both cuts. The next
  best have member violations of 0.044 and 0.051 dex against a 0.040 tolerance.

  **Provenance caveat:** the lane-12 numbers derive from published lensing MASS
  profiles, which this programme's rules admit only for debugging and
  comparison. Nothing here is fitted to them; they are used solely as the shape
  a candidate would have to reproduce, and the flat B = 2 target is reported
  independently.

### 8.4 Is the pair-channel effect actually about channels?

Mostly not. Replacing C by (tr C / 3) I, discarding all directional information
and keeping only pair density, retains most of the boost:

| sigma_perp / sigma_par (kpc) | B(1000) full | B(1000) trace only | ratio | median anisotropy/isotropy of C |
|---|---|---|---|---|
| 400 / 600 | 2.000 | 1.631 | 1.23 | 0.44 |
| 50 / 600 | 2.000 | 1.153 | 1.74 | 0.87 |
| 50 / 150 | 2.000 | 1.100 | 1.82 | 0.84 |

For wide tubes the shell-averaged effect is about 80% an isotropic
there-are-many-pairs-here conductivity change, an environment-dependent
rescaling of a0 dressed as a tensor. Only for narrow tubes does the directional
structure carry most of it. Either way the member-galaxy constraint kills it.

The well-network tensor has the opposite property, worth stating: with A_0 = 0
the exponent is **traceless**, so det K = 1 exactly everywhere and *all* of its
effect is genuine anisotropy. It changes no volume element; it only redirects
flux.

---

## 9. Robustness

**Resolution.** Two independent checks. The PDE gate (6a) gives a 1.47% change
from n = 64 to n = 96 at fixed box, self-convergence order 1.25 to 2.05 - not
flat, which is what it has to be, since a flat error curve versus resolution
would mean a modelling mismatch rather than a discretisation error. Separately,
k(r) itself was recomputed at n = 32, 48, 64, 96, 128 at the amplitude the map
actually uses, with no PDE in the way (resolution_check in
mechanism_map.json):

    plaw p=1 q=2 s=1.5 L=300 kpc, A_T = -4.7
      n= 32  B = 1.588 1.670 1.853 2.464
      n= 64  B = 1.552 1.685 1.845 2.383
      n=128  B = 1.537 1.695 1.837 2.373        (n=64 -> 128: 0.4 to 1.0%)

This mattered: with the *arithmetic* mean the same quantity moved by 10 to 17%
between n = 32 and n = 64, because it was dominated by a handful of near-member
cells. The harmonic mean is converged.

**Domain.** 4 Mpc -> 8 Mpc at fixed cell size moves the shell-averaged field by
**0.357%**.

**Realisation scatter** (seed_robustness.json, five independent draws of the
300 members):

| case | B mean (300/500/1000/1414 kpc) | B sd | max sd/mean | member violation |
|---|---|---|---|---|
| lane-12 survivor, A_T = -24.7 | 2.538 3.116 2.582 1.982 | 0.281 0.335 0.110 0.044 | 11.1% | 0.031 +- 0.023 dex |
| flat-target survivor, A_T = -14.9 | 1.596 1.960 2.002 1.651 | 0.078 0.098 0.048 0.017 | 5.0% | 0.034 +- 0.011 dex |
| ungated reference, A_T = -4.7 | 1.430 1.564 1.812 2.289 | 0.089 0.115 0.119 0.128 | 7.4% | 0.342 +- 0.085 dex |

The cluster boost is stable at the 5 to 11% level while the effect itself is a
factor 2 to 3, so it is a property of the model and not of one draw. **But the
member-galaxy violation of the surviving points is 0.03 dex against a 0.04 dex
tolerance with a realisation scatter of 0.01 to 0.02 dex, so it crosses the
tolerance in some realisations.** The survivors are marginal, not comfortable.

**Full 3-D verification** (headline_3d in mechanism_map.json). Selected points
re-solved on the full 64^3 grid with the complete nonlinear operator:

| candidate | 3-D B(300,500,1000,1414) | map B | projected deflection ratio at 150/300/500/800/1100/1400 kpc |
|---|---|---|---|
| well network, flat target | 1.623 1.973 1.990 1.645 | 1.623 1.992 2.000 1.647 | 1.61 1.78 1.92 1.85 1.66 1.47 |
| well network, flat target | 1.930 2.208 1.995 1.657 | 1.884 2.210 2.000 1.658 | 1.79 1.96 2.04 1.88 1.67 1.49 |
| well network, lane-12 shape | 3.465 3.980 2.645 1.896 | 3.705 3.825 2.357 1.831 | 2.70 3.06 3.26 2.62 1.96 1.69 |
| pair channels, lane-12 shape | 4.390 3.553 1.943 1.373 | 4.472 3.612 1.965 1.381 | 3.51 3.13 2.52 1.82 1.43 1.22 |

Agreement 1 to 12%, always in the direction the calibration predicted. **The
lensing observable moves too** - projected deflection is boosted by 1.5x to
3.5x - so this is not a 3-D-only effect that projection washes out, in contrast
with the earlier lumpiness result where a 0.4% 3-D difference became under 1%
projected.

One pair-channel candidate could not be verified: the row with the smallest
field-galaxy violation reaches B = 2 at 1 Mpc only at an alpha that also gives
B(300 kpc) of order 1e6, and the Picard iteration on that tensor diverges. That
is not a solver complaint; it is a statement about how extreme the tensor has to
be there.

---

## 10. Failure modes checked explicitly, as the brief requires

* **Shared-denominator artefacts.** No correlation statistic is reported in this
  lane, so the artefact cannot arise. B is a ratio of two independent solves of
  the *same* source, not a ratio of two noisy measurements of overlapping
  quantities, and the cluster, field-galaxy and member-galaxy probes share no
  quantity on both axes.
* **Monotone-invariant statistics.** dS/dtheta checked numerically for the
  headline statistic: the spread of B(1000 kpc) over the scanned amplitude range
  has median **11.4**, minimum **0.010**, and **zero rows with zero spread**,
  for both tensors.
* **Refitting on held-out sets.** Nothing in this lane is fitted to data at all;
  the map is a forward calculation. **KiDS and wide binaries were never loaded
  and never looked at.**
* **Silent extraction failures.** Row and column counts asserted after ingest:
  38 A2029 profile rows, identifier extent = 1414 kpc echoed back, 300 members,
  44,850 pairs, 110,592 to 262,144 grid cells per gate; SHA-256 manifest
  written.
* **Test bugs that look like solver bugs.** Four caught here: flux measured on
  the face fluxes rather than a centre-differenced gradient (3.3e-14 instead of
  1e-2); the analytic gate uses a source spherical in u = sqrt(r^T K^-1 r) and
  not in r; the isotropy gate's form residual is meaningless where |S| is below
  the Monte-Carlo floor; and the resolution curve is checked to be *not* flat.
* **Non-monotonic M(r).** numpy.maximum.accumulate on the enclosed-mass profile;
  the outer continuation is tapered rather than clipped.
* **No data that presupposes dark matter.** The only observational input to the
  solve is the X-COP *baryonic* acceleration column. The lane-12 comparison
  target does derive from published lensing mass profiles, is flagged as such,
  is used as a comparison shape only, and nothing is fitted to it.

---

## 11. What could not be established

1. **Whether the surviving corner survives a real cluster.** Everything rests on
   one synthetic configuration with a statistical member population. The
   realisation scatter is measured (5 to 11% on the boost) but A2029's actual
   member catalogue is not on disk, and the member-galaxy constraint - the
   binding one - depends on where the members actually are.
2. **A quantitative member-galaxy limit.** The 0.04 dex tolerance is imported
   from the RAR's intrinsic scatter. Turning "cluster early types lie on the
   same fundamental plane as field ones" into a number would need an actual
   fundamental-plane analysis, which this lane did not do. The constraint is
   therefore qualitative in strength even though the model's prediction for it
   is quantitative.
3. **The external-field effect.** The member-galaxy boost is computed as the
   response to the member's own enclosed mass at the local k, neglecting the
   MOND external field effect from the cluster. That is an approximation whose
   sign is not obvious; it indicates the size of the problem rather than
   predicting the observable.
4. **Whether a finer gate grid opens the surviving region.** Only three Phi_0
   values and three exponents m were scanned. The survivors cluster at
   Phi_0 = 1e12, m = 4, i.e. at an edge of the scanned gate grid, so the region
   may be larger (or the boundary sharper) than 127 of 1920 suggests.
5. **The extreme-amplitude regime.** At |A_T| > 20 with no gate the tensor's
   condition number reaches about 1e5 and Jacobi-preconditioned CG stops
   converging (residual stalls near 1e-2 after 4,000 iterations). Those points
   are reported from the calibrated surrogate only. A multigrid or line-relaxed
   preconditioner would be needed to verify them directly.
6. **Anything about groups, voids, or time dependence.** Out of scope here.

---

## 12. Verdict

**Can either tensor produce the factor-of-two cluster gap? Yes, both can, and
easily. The amplitude was never the difficulty.** The well-network tensor
reaches B = 2 at 1 Mpc at A_T = -4.7 ungated, and the effect is smooth and
geometric: the lumpiness contribution is 0.2 to 0.3%, against the 0.4% that was
the *entire* effect in the earlier QUMOND-on-lumpiness calculation. Normalising
S by the total weight is what buys this - it makes the tensor scale-free, so it
does not die as the member population is smoothed, and a single universal
constant then sets the amplitude. The pair-channel tensor reaches any amplitude
at all because C is unnormalised and grows with the pair count.

**Can either do it without destroying the galaxy-scale limit? The pair-channel
tensor: no, at every one of 288 scanned points.** It is trivially safe for
isolated field galaxies (they have no pairs) and impossible for cluster members
(the best case boosts a member's internal gravity by a factor 4.9), and its
radial profile is far too centrally concentrated in every case because C follows
the square of the member density.

**The well-network tensor: yes, but only in a specific corner, and only with a
steep potential-depth gate that is doing all of the work.** 127 of 1920 points
reach a flat B in 1.6 to 2.5 with both galaxy probes intact; all of them use a
phi gate, 126 of them with m >= 2 at Phi_0 = 1e12 m^2/s^2. Held instead to the
programme's own *measured* radial run of the cluster excess, exactly one scanned
point survives, and it sits on the boundary of both cuts with a
realisation-to-realisation scatter that crosses the tolerance.

The honest summary is that **the anisotropy is not the active ingredient**. What
makes the model work is an environment-dependent switch on the depth of the
local Newtonian potential; the traceless tensor then converts that switch into a
change in the radial conductivity. A scalar a0 -> a0 f(|Phi_N|/Phi_0) would
reproduce the amplitude and, to within the accuracy of this map, the radial
shape. The tensor's distinctive content is (i) that it changes no volume element
(det K = 1 when A_0 = 0), (ii) that it predicts anisotropic light bending around
individual members, and (iii) that the same construction predicts a large boost
*inside* cluster galaxies, which is the sharpest way to kill it and is the one
that nearly does.

**Recommendation to the programme.** Do not eliminate the well-network tensor:
it beats plain MOND on clusters by a factor 2 in a region of parameter space
that survives every numerical gate, and it is not shot noise. Do record that its
discriminating power comes entirely from a potential-depth screen and belongs
with the potential-depth lane rather than being credited to the anisotropy. And
treat "does the internal dynamics of cluster member galaxies change?" as the
decisive next measurement: it is the constraint that separates the surviving
corner from the rest of the parameter space, it is the one constraint the brief
did not anticipate, and it is a real observation that has already been made by
somebody.
