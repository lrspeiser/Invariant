# Method appendix -- definitions, freeze order, and every declared constant

This is the long form of REPORT.md sections 2-5.  Everything here was fixed
before any residual was examined, and nothing here was changed afterwards.

## 1. The grammar

    K = exp[ f_0(I) I + f_T(I) That + f_d(I) dhat dhat^T + f_nl(I) B_nl ]

A candidate is a tuple

    (base MOND law, invariant I, response form f, exponent m, invariant scale
     I_0, tensor structure, amplitude A)

with exactly TWO fitted global constants -- `a0` and `A` -- and no per-object
parameter of any kind anywhere.

### 1.1 The five invariants, all of them genuine fields

An invariant here must be a function of position that needs no object-specific
centre, radius or catalogue row, because the same definition has to be
evaluated in a disk at 20 kpc, in a cluster shell at 1 Mpc, and inside a
galaxy that is itself inside that cluster.

| name | definition | SI units | behaviour outward |
|---|---|---|---|
| `gn` | `\|g_N\|/a0` | 1 | falls as r^-2 |
| `phi` | `\|Phi_N\|/Phi_0` | 1 | falls as r^-1 |
| `rhobar` | `div g_N/(4 pi G)`, i.e. the Poisson source itself | kg m^-3 | zero outside |
| `tidal` | `\|traceless Hessian of Phi_N\|_F` | s^-2 | falls as r^-3 |
| `qbar` | `M_b(within L_NL of x)/(M_b + M_0)`, bounded in [0,1) | 1 | -> const |

`L_NL = 300 kpc` and `M_0 = 1e12 Msun` are declared globals of the nonlocal
invariant.  `qbar` is carried specifically as the boundedness theorem's
control: it is bounded by construction, so by the theorem it can only
renormalise G, and the tournament exhibits that rather than assuming it.

`rhobar` is defined as the Poisson source and not as a mean enclosed density,
because a mean enclosed density needs a centre and is therefore an
object-relative quantity dressed as a field.

### 1.2 The five response forms

| form | W(I) | sup W | can it change the asymptotics? |
|---|---|---|---|
| `off` | 0 | 0 | no -- the null |
| `sat` | I^m/(1+I^m) | 1 | bounded |
| `inv` | 1/(1+I^m) | 1 | bounded |
| `pow` | I^m, m = 0.5, 1, 2 and **-1, -2** | infinity | unbounded |
| `log` | ln(1+I^m) | infinity | unbounded |

Negative exponents are in the grid on purpose: they are the only way inside
this grammar to make W GROW outward, which is what the boundedness theorem's
prescribed repair actually requires.  Their asymptotics are measured, not
assumed (`asym_table.json`).

A declared numerical ceiling `W <= 1e6` prevents overflow when an invariant
legitimately reaches 1e-40; every clip is counted and reported, and no headline
number rests on a clipped value.

### 1.3 The five structures

| structure | K | what it is |
|---|---|---|
| `scalar_a0` | none; `a0 -> a0 (1 + A W)` | **the brief's explicit competitor** |
| `iso_K` | `exp(-A W) I` | isotropic conductivity, det K != 1 |
| `tensor_S` | `exp(A W S)` | the well-network alignment tensor, traceless |
| `tensor_d` | `exp(A W (dhat dhat^T - I/3))`, dhat = ghat_N | field-direction, traceless |
| `tensor_T` | `exp(A W That)`, That = traceless Hessian / its spectral norm | tidal, traceless |

`tensor_S` uses the tensor lane's own weight families and settings unchanged;
`focus.py` adds the brief's LITERAL reading (mass weighting p = 1, no
self-exclusion) beside them.

For the traceless structures, `dhat dhat^T - I/3` is a rank-one projector minus
a constant, so the exponential is exact in closed form,

    e^T K e = e^{-a/3} + (e^{2a/3} - e^{-a/3}) (e . dhat)^2,   a = A W

which is what the vertical channel uses; the cluster channel uses the tensor
lane's validated closed-form symmetric-3x3 exponential (worst relative error
2.4e-12 against scipy.linalg.expm over 1,000 matrices including degenerate
spectra).

## 2. Freeze order

1. `a0` is fitted on the **SPARC TRAIN split only**, frozen split `e5f74522`,
   at declared `Upsilon_3.6 = 0.5` (disk) and `0.7` (bulge), catalogue distances
   and inclinations, no per-galaxy freedom.
2. `A` is fitted on the **cluster channel** by minimising the RMS in dex
   against the lane-12 measured radial requirement.
3. Where a gate fires in galaxies (`max |A W| > 1e-3` on the SPARC train
   points) the two fits are ALTERNATED to convergence and the number of
   iterations is recorded per candidate.
4. The two vertical channels are then evaluated with no free parameter left.

Channels 1 and 4 are in-sample.  Channels 2 and 3 are out-of-sample.

The SPARC validation and blind splits, KiDS and the wide binaries are never
loaded.  There is no code path in this lane that reads them.

## 3. |Phi_N|, and the rule that defines it

`|Phi_b|` is defined only up to a constant, and Run Z showed the residual at
fixed `(g_bar, r)` is EXACTLY the shape factor, so the boundary rule DEFINES
the variable rather than conditioning it.  Four rules are implemented; the
primary is declared in advance.

| rule | definition | global? |
|---|---|---|
| **`inf` (PRIMARY)** | Phi -> 0 at infinity, baryons continued outside the last measured point as a point mass | yes |
| `flat` | outer continuation as a flat rotation curve truncated at a UNIVERSAL R_trunc = 1 Mpc | yes |
| `last` | referenced to the object's own last measured radius | **no** |
| `half` | `G M_b/max(r, R_eff)` | **no** |

`last` and `half` name an object-specific reference and therefore violate the
programme's global-parameter rule; they are carried only to show how far the
variable moves.  `inf` is also the convention the cluster channel uses (the
tensor lane's tapered outer continuation with Phi -> 0 at infinity), so the two
channels use one convention rather than two.

## 4. The four channels

### Channel 1, radial rotation
RMS of `log10(g_pred/g_obs)` over every retained radial point of every TRAIN
galaxy.  Declared cut before residuals: a point whose net baryonic `v^2` is
negative has no meaningful `g_bar` and is dropped -- the identical cut Run L
uses, so the two lanes' numbers are directly comparable.  Measured here:
**Newton 0.5215, RAR 0.1641, AQUAL 0.1647 dex**, reproducing Run L's
0.5215 / 0.1641 / 0.1647 exactly.  The 0.11 dex the brief names is the RAR's
scatter with per-galaxy nuisances marginalised; under THIS protocol, which
allows a law no per-object freedom at all, the RAR's own value is 0.164 dex and
that is the honest bar.  Both are quoted.

### Channel 2, vertical amplitude -- A CONSTRAINT, NOT A DISCRIMINATOR
`B_z = 0.715`, 95% `[0.301, 1.670]`, width 0.192 dex, systematic floor 8.4x the
statistical part and dominated by common-mode terms.  The largest
law-to-Newton separation any law produces is 0.190 dex = 0.99 sigma.  Scored as
a PASS/FAIL against the 95% interval with the z-score reported; the joint
ranking is also reported with this channel removed, so it is visible that it
decides nothing on its own.

`B_z(law)` is defined exactly as Run L defines it: the square of the ratio of
the FITTED exponential amplitudes, law over Newton, averaged in the log over
galaxies.

### Channel 3, vertical radial shape -- THIS ONE DISCRIMINATES
`h_sigma_LOS`, observed 28.65 arcsec.  It is blind to a constant vertical boost
(multiplying K_z by 8 moves it by 1.6e-15 dex) and therefore sees only the
RADIAL RUN of the boost -- the part the amplitude systematics cannot fake.
Reproduced here: **Newton 30.80 arcsec at chi2/dof 10.48, RAR 35.20 at 20.23,
AQUAL 34.91 at 19.61**, against Run L's 30.80 / 35.20 / 34.96 and 10.5 / 20.2 /
20.0.  The 0.05 arcsec AQUAL difference is this lane's use of the algebraic
vertical reduction for both bases where Run L bisects the exact AQUAL equation.

### Channel 4, cluster amplitude and shape
Synthetic A2029 from the tensor lane: the real X-COP baryonic mass profile plus
300 statistical members, 44,850 pairs.  `B = |g|(response)/|g|(K = I)` on
shells at 300, 500, 1000 and 1414 kpc.  TWO targets, both reported:

* **lane-12** (primary for fitting): the programme's own measured radial run of
  the cluster excess, `B_req = 3.86, 3.33, 2.40, 1.76`.  PROVENANCE CAVEAT:
  this derives from published lensing MASS profiles, which the programme's
  rules admit only for comparison.  Nothing is fitted to the masses themselves;
  the profile is used as the shape a candidate must reproduce.
* **flat B = 2** (independent): X-COP's `nu/nu_RAR = 2.53` for A2029, which does
  not come from a lensing mass profile.  The best-fitting amplitude for this
  target and its own galaxy violations are reported separately.

## 5. The shell average is physics

`k` varies by orders of magnitude across a shell once `|A|` is large.  The
tensor lane measured three candidate averages against six full nonlinear 3-D
solves: worst departure **arithmetic 46.9%, cell-wise 22.4%, harmonic 20.4%**,
and the arithmetic mean is qualitatively wrong -- it turns over, fakes a
saturation at B = 2.1, and is how `A_T = -12.8` got reported where the truth is
-4.7.  The **harmonic** mean is used throughout; the arithmetic one is computed
at every candidate's fitted amplitude and the bracket
`|log10 B_harm - log10 B_arith|` is stored per candidate.

For the scalar competitor there is no conductivity to average, so the same
calibrated rule is TRANSLATED rather than dropped: in deep MOND the two
parametrisations are related exactly by `k_eq = (1 + A W)^(-2/3)`, so the
harmonic mean of `k_eq` corresponds to

    (1 + A W)_eff = < (1 + A W)^(2/3) >^(3/2)

with `< 1 + A W >` stored beside it.  The cluster sits at `g_b/a0 = 0.07-0.18`,
deep enough for that translation to hold to a few per cent; the galaxy probes,
which are NOT deep MOND, use the exact mu inversion instead.

## 6. The seven hard screens

| screen | criterion | declared source of the number |
|---|---|---|
| H1 cluster reach | `B(1 Mpc) >= 1.5` | the measured factor-2 cluster gap |
| H2 field galaxy | `\|log10 B\| <= 0.040 dex` at 10/20/30 kpc | the RAR's intrinsic scatter |
| H3 **member galaxy** | `\|log10 B\| <= 0.040 dex` at 10/20/30 kpc | same; the constraint nobody wrote down |
| H4 radial | RMS `<= 0.30` dex on SPARC train | between the RAR's 0.16 and Newton's 0.52 |
| H5 vertical amplitude | inside `[0.301, 1.670]` | Run L's 95% interval |
| H6 vertical shape | `chi2/dof <= 40` | the isotropic tensor was rejected at 133 |
| H7 asymptotic | `d ln g/d ln r` in `[-1.25, -0.75]` | a flat rotation curve |

Both the SEQUENTIAL funnel and the MARGINAL power of each screen (how many it
kills on its own, and how many it kills UNIQUELY) are reported, because a
sequential funnel credits whichever screen runs first with everything the
screens share.

## 7. The momentum screen

    F_net,i = - oint T_ij n_j dS  -  (1/8 pi G) int (d_i K_jk) d_j Psi d_k Psi

measured with the screen lane's `fieldsolve.py` and `screen._identity_force`,
both imported unmodified, on two UNEQUAL masses.

Two departures from the screen lane's configuration, both necessary:

1. **The null is the same base law with the response switched off**, not
   Newton.  AQUAL and QUMOND are variational, so their exact net force is zero,
   but the two-solve discretisation leaves 1-2% of `G M1 M2/d^2` (the screen
   lane measures 0.021 for AQUAL, 0.011 for QUMOND, and passes them on declared
   variational grounds).  Measuring a gated law against the Newtonian null
   would credit the gate with the base law's discretisation residual.
2. **The configuration is CLUSTER-scale**, `M1 = 8e13`, `M2 = 2e13 Msun` at
   500 kpc, because a galaxy-scale pair sits at `|Phi_N| ~ 1e10 m^2/s^2`, four
   orders below a `Phi_0 = 1e12` gate, and would measure nothing but the
   discretisation floor.  The galaxy-scale pair is kept as the secondary, where
   a null IS the correct answer.

## 8. Model selection

A bare argmin over a joint score mis-identifies nested families, because a
richer model can only tie or win: the controls lane measured 4 of 5 injected
families recovered by argmin against 5 of 5 by a one-standard-error parsimony
rule.  The rule is used here.  `SE(J)` is estimated by bootstrap over OBJECTS
-- SPARC train galaxies, DiskMass galaxies, cluster shells, resampled with
replacement -- not over points.  Among candidates within 1 SE of the best `J`,
the one with the fewest free global constants is the pick; ties inside that set
are broken by `J`.

`n_params` counts: `a0` always; `+1` for the amplitude `A`; `+1` for the
exponent `m`; `+1` for the invariant scale `I_0`.  A base law with no response
therefore has k = 1 and a fully specified gated law has k = 4.
