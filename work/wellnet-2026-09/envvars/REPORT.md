# Run AJ — four environmental variables, one sample, one set of folds

Lane `work/wellnet-2026-09/envvars/`.

| file | what it does |
|---|---|
| `envvars.py` | builds the four variables, the collinearity measurement and the coarse-graining gate |
| `fixedeffects.py` | both estimators, the simulated nulls (with a `worker` mode that computes one Monte-Carlo slice), the responsiveness gate, the frozen split |
| `fragility.py` | how many objects each beta is actually made of |
| `refresh.py` | merges Monte-Carlo slices into the results JSON |
| `report.py`, `tables.py` | render this file; every number below comes through them, none is transcribed |
| `envvars_build.json`, `envvars_results.json`, `envvars_fragility.json` | machine-readable results |
| `envvars_table.npz` | every variable on every system's radial grid |
| `envvars_build.log`, `run.log`, `fragility.log`, `refresh.log` | the runs |
| `null_part_*.json`, `inj_*.json`, `wnull_*.log`, `winj_*.log` | the raw Monte-Carlo slices and their worker logs |

Nothing was re-acquired: the eFEDS Vikhlinin density fits, the DECADE
per-cluster shear profiles and their manifests are reused unmodified
from `lead01/` and `efeds-hsc/`.  KiDS and wide binaries were not
loaded, opened or referenced.

---

## 0.  The headline

Four physically distinct environmental variables were built on the SAME
496 systems and the SAME 3365 (system, radial-bin) shear points, and each was
fitted twice: once with a free amplitude per object (WITHIN-OBJECT, the
design the brief asks for) and once with a single global amplitude
(WITHIN-CLASS).  Every estimate is quoted against its own simulated
null, and the nulls are not the same for the four variables.

**The two structural results are worth more than any of the fits.**

1. **The vector sum and the directionless sum are separated by 2.41 dex on this catalogue, and the
   separation is not a detail of the sample — it is forced.**  The
   vector sum over ALL mass is exactly `g_bar` by Newton's shell
   theorem, so a V2 that includes the object's own mass is identically
   the acceleration already carried by `f(g_bar, r)` and the test is
   vacuous.  V2 is therefore external-only *by theorem*, and the
   external field of a real catalogue is 5.53e-05 of a0 and 8.52e-03 of the local g_bar.  The
   directionless sum has no shell theorem: its self term survives at
   1.420 of g_bar — near it, but not equal
   to it, and with a different radial shape.  **Variables 2 and 3 differ
   by whether opposing wells cancel, and the cancellation costs 2.41 dex.**

2. **The within-object design and the environmental content are in
   direct competition, and on this sample they are mutually
   exclusive.**  The variables with real radial variation inside an
   object (V1 1.24 dex, V3 1.97 dex, V4a 2.43 dex) are 98.4%, 96.8% and 97.2% explained
   by a quadratic in (log g_bar, log r) at the shear-measured radii.
   The variables that are genuinely orthogonal to (g_bar, r) (V2, R^2 = 0.119; V4d, R^2 = 0.084) vary by 0.00326 and 0.02120 dex
   inside an object.  There is no variable on this sample that is both
   environmental and radially resolved.

Against that background, of the 28 null-calibrated estimates (8 variables x raw/residualised x 2 estimators), 4 exceed 3 sigma against their own null and the largest is |z| = 5.97 (-- acceleration tilt, raw, within-class).

| rank | variable | parameterisation | estimator | beta | its null | z |
|---|---|---|---|---|---|---|
| 1 | -- acceleration tilt | raw | within-class | +0.1166 | -0.3350 +- 0.0756 | +5.97 |
| 2 | V1 potential depth | raw | within-class | +0.0316 | -0.3057 +- 0.0684 | +4.93 |
| 3 | V3 directionless W | raw | within-class | +0.1485 | -0.2545 +- 0.0897 | +4.49 |
| 4 | V4a tidal magnitude | raw | within-class | +0.2405 | -0.3549 +- 0.1610 | +3.70 |
| 5 | V3 directionless W | perp | within-class | +0.0915 | -0.2608 +- 0.1309 | +2.69 |
| 6 | V4d external tidal | raw | within-object | -2.0000 (edge) | -0.1943 +- 0.7388 | +2.44 |
| 7 | V4b tidal shape | raw | within-class | +0.1453 | -0.1253 +- 0.1245 | +2.17 |
| 8 | V4d external tidal | perp | within-object | -1.8178 | -0.0046 +- 0.9233 | +1.96 |

`(edge)` marks a profile minimum pinned at the edge of the [-2, +2] beta grid; those rows are bounds, not estimates, and their z is not interpretable.

On raw training chi2 the largest within-class improvement of any variable is -- radius tilt at dchi2 = 19.22 -- a bare radius tilt, which contains no environment at all, reproducing Run AI's M3 result from inside a different estimator.

**And the control decides it.**  Ranking the within-class estimates by z against their own nulls:

| variable | z vs its own null | environmental? |
|---|---|---|
| -- acceleration tilt | +5.97 | **no** |
| V1 potential depth | +4.93 | yes |
| V3 directionless W | +4.49 | yes |
| V4a tidal magnitude | +3.70 | yes |
| V4b tidal shape | +2.17 | yes |
| V4d external tidal | +0.38 | yes |
| V2 vector g_ext | +0.35 | yes |
| -- radius tilt | -1.44 | **no** |

The largest z belongs to -- acceleration tilt at +5.97 -- a quantity with NO environmental content at all, above the best environmental variable (V1 potential depth, +4.93).  Every variable that depends on the X-ray density fit sits well above a null whose mean is strongly negative, and the ordering does not favour environment.  Nothing here is evidence for a second environmental variable; it is evidence that the null for any density-fit-derived radial regressor is displaced, which is exactly the artefact family the brief warned about, now seen for the sixth time.

---

## 1.  What was measured, and on what

| item | value |
|---|---|
| eFEDS Vikhlinin density fits (Bahar+2022 table 1) | 542 rows x 19 cols, asserted |
| eFEDS properties (table 2) | 542 rows x 40 cols, asserted |
| DECADE per-cluster shear profiles | 536 systems, 4228 rows with finite g_t, asserted |
| matched sample | 496 systems, 3365 (system, bin) points |
| M_gas,500 reproduction gate | n = 414, median 0.9994 of Bahar+2022, 0.0469 dex scatter, PASS |
| well network | 542 catalogued concentrations, median M_b(<R500) 6.87e+12 Msun |
| nearest-neighbour separation | median 32.7 Mpc comoving, 10th pct 9.4, min 0.66 |
| declared split | 248 train (1709 points) / 248 held out (1656 points); sorted by eFEDS ID, even=train, odd=held out; identical to Run AI |
| fitted global amplitude | 10^A = 0.9294 (exact nonlinear 0.8660) |
| max convergence kappa in the sample | 0.0396 |
| chi2, per-object amplitudes vs one global amplitude | 3060.6 vs 3589.3 on 3365 points |
| per-object amplitudes that come out negative | 30.4% |

The last two rows set the scale of what a within-object estimator can
do here.  495 free per-object amplitudes buy 528.7 in chi2, i.e.
1.07 per parameter — exactly what pure noise buys.  The per-object
weak-lensing signal-to-noise is far below one; the whole detection is
9.6 sigma across 496 systems.  Object-level amplitudes therefore carry
no information, which is precisely why they are safe to profile out,
and equally why the within-object estimator is the noisier of the two.

### The four variables as built

| | definition | self term | declared scale |
|---|---|---|---|
| V1 | `DeltaPhi_b(r) = Phi_b(r_ref) - Phi_b(r)`, an operational DIFFERENCE | yes | Phi_0 = 1e12 m^2/s^2, primary rule `fixed10Mpc`, four alternatives |
| V2 | `g_ext = sum_a G M_a d_a / \|d_a\|^3`, opposing wells CANCEL | excluded **by theorem** (see below) | a0 = 1.2e-10 m/s^2 |
| V3 | `W = sum_a G M_a / (d_a^2 + eps^2)`, opposing wells DO NOT cancel | included, as the exact continuum angular integral | eps = 50 kpc primary, 20 and 200 kpc sensitivity |
| V4 | `T_ij = d_i d_j Phi_b`; magnitude, shape and eigenvectors kept separately | included, analytic | T_0 = 1e-33 s^-2 |

V2 has to be external-only.  `sum_a G M_a d_a/\|d_a\|^3` over ALL mass
is `grad Phi` — that is Newton's theorem, not an approximation — so
including the object's own mass makes V2 identically `g_bar`, which
`f(g_bar, r)` already carries.  V3 has no such theorem, so its self term
is a new function of radius and is kept.

V4 on a spherical system collapses, analytically, to

```
    lam_r = (g/r)(2q - 2),   lam_t = (g/r)(1 - q),   q = rho / <rho>
    lam_r / lam_t = -2  IDENTICALLY
    |T~|  = sqrt(6) (g/r) |1 - rho/<rho>|
```

so the tidal SHAPE of a spherical object carries exactly one bit (the
sign of `rho/<rho> - 1`), the principal eigenvector is radial by
construction, and the only content beyond (g_bar, r) is the local
density contrast `rho/<rho>`.  The eigenvector information is therefore
degenerate until an external tide breaks it, and the external tide here
is 2.85e-04 of the internal one.  That is why
V4b is reported as `rho/<rho>` and V4d as the external tidal magnitude
separately, rather than as a fabricated 'shape scalar'.

---

## 2.  THE FOUR-VARIABLE COMPARISON

`beta` is in dex of `log g` per ONE STANDARD DEVIATION of the variable,
so the four are directly comparable.  `lev` is the fraction of the
variable's Fisher information for beta that survives the amplitude
projection: `lev = 0` means beta is not identified at all.  `z` is
`(beta - E[beta|H0]) / sd(beta|H0)` against that variable's OWN
simulated null.

### 2a.  Raw variables, fitted on TRAIN

| variable | WO beta | WO null E[b\|H0] | WO z | WO lev | WC beta | WC null E[b\|H0] | WC z | WC lev |
|---|---|---|---|---|---|---|---|---|
| V1 potential depth | +0.2540 | -0.0708 +- 0.1639 | +0.31 | 0.149 | +0.0316 | -0.3057 +- 0.0108 | +4.93 | 0.729 |
| V2 vector g_ext | -0.1730 | -0.0217 +- 0.1582 | -0.15 | 0.178 | +0.0368 | +0.0001 +- 0.0165 | +0.35 | 0.913 |
| V3 directionless W | +0.1290 | -0.0339 +- 0.0846 | +0.30 | 0.529 | +0.1485 | -0.2545 +- 0.0142 | +4.49 | 1.000 |
| V4a tidal magnitude | +0.1750 | -0.1282 +- 0.0865 | +0.55 | 0.750 | +0.2405 | -0.3549 +- 0.0255 | +3.70 | 0.990 |
| V4b tidal shape | +0.0794 | -0.0007 +- 0.0521 | +0.24 | 0.413 | +0.1453 | -0.1253 +- 0.0197 | +2.17 | 0.625 |
| V4d external tidal | -2.0000 (edge) | -0.1943 +- 0.1168 | -2.44 | 0.169 | +0.0379 | -0.0019 +- 0.0165 | +0.38 | 0.900 |
| -- radius tilt | -0.1563 | +0.0135 +- 0.0235 | -1.14 | 0.285 | -0.3283 | -0.1249 +- 0.0223 | -1.44 | 0.304 |
| -- acceleration tilt | +0.0876 | -0.2272 +- 0.1177 | +0.42 | 0.474 | +0.1166 | -0.3350 +- 0.0120 | +5.97 | 0.883 |

### 2b.  Residualised variables — the quadratic in (log g_bar, log r) projected out

This is the flexible-scalar-nuisance version of the same question: it
asks whether the variable carries anything an arbitrary smooth function
of acceleration and radius does not.

| variable | WO beta | WO null E[b\|H0] | WO z | WO lev | WC beta | WC null E[b\|H0] | WC z | WC lev |
|---|---|---|---|---|---|---|---|---|
| V1 potential depth | -0.3210 | +0.1890 +- 0.0579 | -1.39 | 0.241 | -0.0141 | -0.1262 +- 0.0202 | +0.88 | 0.739 |
| V2 vector g_ext | -2.0000 (edge) | +0.0250 +- 0.1736 | -1.84 | 0.183 | +0.0401 | +0.0037 +- 0.0167 | +0.34 | 0.938 |
| V3 directionless W | -0.4388 | -0.0490 +- 0.0898 | -0.69 | 0.276 | +0.0915 | -0.2608 +- 0.0207 | +2.69 | 0.913 |
| V4a tidal magnitude | +0.0478 | -0.0425 +- 0.0880 | +0.16 | 0.869 | +0.0001 | +0.2298 +- 0.0323 | -1.13 | 0.999 |
| V4b tidal shape | -0.0779 | +0.0155 +- 0.0603 | -0.24 | 0.415 | +0.0262 | -0.0625 +- 0.0132 | +1.07 | 1.000 |
| V4d external tidal | -1.8178 | -0.0046 +- 0.1460 | -1.96 | 0.170 | +0.0430 | +0.0017 +- 0.0168 | +0.39 | 0.926 |

### 2c.  The nulls themselves, and the power they imply

40 realisations.  Each redraws `n0^2`, `rs`, `epsilon`, `beta`,
`alpha` for all 542 catalogued systems from the published errors, plus
an assumed `sigma_z = 0.005(1+z)`, rebuilds the well network and every
variable from the redraw, and regenerates the shear independently around
the TRUE model.  The same redraw enters the baseline model and the
variable — that is the shared-quantity channel that gave Run AI
`E[beta|H0] = -0.0666 +- 0.0101`, a -6.6 sigma artefact from X-ray fit
noise alone.

| variable | WO E[beta\|H0] | WO sd | WO 95% detectable \|beta\| | WC E[beta\|H0] | WC sd | WC 95% detectable \|beta\| |
|---|---|---|---|---|---|---|
| V1 potential depth | -0.0708 +- 0.1639 | 1.0365 | 2.032 | -0.3057 +- 0.0108 | 0.0684 | 0.134 |
| V2 vector g_ext | -0.0217 +- 0.1582 | 1.0003 | 1.961 | +0.0001 +- 0.0165 | 0.1041 | 0.204 |
| V3 directionless W | -0.0339 +- 0.0846 | 0.5348 | 1.048 | -0.2545 +- 0.0142 | 0.0897 | 0.176 |
| V4a tidal magnitude | -0.1282 +- 0.0865 | 0.5473 | 1.073 | -0.3549 +- 0.0255 | 0.1610 | 0.316 |
| V4b tidal shape | -0.0007 +- 0.0521 | 0.3292 | 0.645 | -0.1253 +- 0.0197 | 0.1245 | 0.244 |
| V4d external tidal | -0.1943 +- 0.1168 | 0.7388 | 1.448 | -0.0019 +- 0.0165 | 0.1044 | 0.205 |
| -- radius tilt | +0.0135 +- 0.0235 | 0.1486 | 0.291 | -0.1249 +- 0.0223 | 0.1411 | 0.277 |
| -- acceleration tilt | -0.2272 +- 0.1177 | 0.7446 | 1.459 | -0.3350 +- 0.0120 | 0.0756 | 0.148 |

The nulls are NOT the same for the four variables, which is the
whole reason the brief insists on one null per variable.  Note also
that the WITHIN-OBJECT null saturates the [-2, +2] beta grid in up
to 48% of realisations, so its quoted sd is a LOWER
bound on the true spread and every within-object z above is an
upper bound on significance.  The within-class nulls never reach
the grid edge.

Two things to read off.  First, the null mean is large and NEGATIVE for
every variable built from the X-ray density fit, and it is not small:
for potential depth within-class it is -0.306 +- 0.011, i.e. 28 sigma_MC from zero, driven by fit noise alone.  That is the same
artefact Run AI measured at -0.0666 +- 0.0101, now seen in a different
estimator and a different parameterisation.  Second, the Fisher error
bars in section 2a are badly optimistic wherever the variable depends on
the density fit: for V2 within-object the Fisher sigma is 0.033 while the null sd is 1.000, a factor of 30.
An analysis that quoted the Fisher error would have reported a
multi-sigma within-object detection of the external field that is
entirely propagated X-ray fit noise.

---

## 3.  COARSE-GRAINING, the gate V2 and V3 had to pass

Uniform refinement first, to confirm it has no teeth.  Splitting every
catalogue row into four equal pieces at the SAME position and
re-evaluating gives, for mass exponents p = 0.5, 1 and 2, a scatter
across probe points of

```
    p = 0.5    scatter 6.154e-17 dex
    p = 1.0    scatter 5.173e-17 dex
    p = 2.0    scatter 1.167e-16 dex
```

i.e. machine zero for every exponent: a p != 1 sum is rescaled by a
global constant that any fitted amplitude absorbs, and the p = 1 sums
used here are bit-identical.  The brief's warning is confirmed exactly.

The test with teeth represents the SAME continuous mass as N catalogue
rows on a spherical mesh, each cell carrying the exact enclosed mass at
its own centre of mass, and compares against the continuum: the exact
angular integral for W, Newton's `G M(<r)/r^2` for the vector sum.
Drift is the RMS over 48 probe points spanning 0.3-4 Mpc.  The
selective variant refines only the mass inside 1 Mpc and leaves the
outside as a single row.

| system | N rows | drift W (dex) | drift \|g\| (dex) | drift W selective | drift \|g\| selective |
|---|---|---|---|---|---|
| 084129.0+002645 | 1 | 1.05877 | 1.69421 | 0.13461 | 0.34470 |
| 084129.0+002645 | 10 | 0.43156 | 0.68782 | 0.22877 | 0.33068 |
| 084129.0+002645 | 64 | 0.19309 | 0.39524 | 0.04454 | 0.04102 |
| 084129.0+002645 | 512 | 0.07050 | 0.18206 | 0.01234 | 0.02951 |
| 084129.0+002645 | 4096 | 0.03005 | 0.08077 | 0.00668 | 0.00922 |
| 084129.0+002645 | 32768 | 0.00799 | 0.03179 | 0.00166 | 0.00145 |
| 090129.1-013853 | 1 | 0.79882 | 1.27324 | 0.13496 | 0.41997 |
| 090129.1-013853 | 10 | 0.58326 | 0.85284 | 0.27108 | 0.41422 |
| 090129.1-013853 | 64 | 0.18075 | 0.36128 | 0.05803 | 0.11415 |
| 090129.1-013853 | 512 | 0.06488 | 0.16239 | 0.01450 | 0.06016 |
| 090129.1-013853 | 4096 | 0.03396 | 0.05892 | 0.00773 | 0.01622 |
| 090129.1-013853 | 32768 | 0.00756 | 0.02555 | 0.00194 | 0.00235 |
| 092031.8+040621 | 1 | 0.09081 | 0.25693 | 0.06986 | 0.23472 |
| 092031.8+040621 | 10 | 0.27511 | 0.33959 | 0.25361 | 0.31620 |
| 092031.8+040621 | 64 | 0.07291 | 0.17269 | 0.04715 | 0.08019 |
| 092031.8+040621 | 512 | 0.01469 | 0.08717 | 0.01234 | 0.03922 |
| 092031.8+040621 | 4096 | 0.01090 | 0.01698 | 0.00544 | 0.00877 |
| 092031.8+040621 | 32768 | 0.00199 | 0.01690 | 0.00092 | 0.00101 |

| system | beta_N W | beta_N \|g\| | beta_N W selective | beta_N \|g\| selective |
|---|---|---|---|---|
| 084129.0+002645 | 0.4636 | 0.3757 | 0.4670 | 0.5309 |
| 090129.1-013853 | 0.4545 | 0.3926 | 0.4585 | 0.5018 |
| 092031.8+040621 | 0.4183 | 0.3215 | 0.4750 | 0.5349 |

`beta_N = -dln(drift)/dln(N)`, the convention already used by
`work/wellnet-2026-09/screen/screen.py`.

And the refinement that actually matters for V2 and V4d, which are
external-only: split each of the 60 nearest catalogued neighbours into K
pieces spread over a generous 1.5 Mpc extent and re-evaluate.

| system | K pieces per external well | drift W_ext (dex) | drift \|g_ext\| (dex) |
|---|---|---|---|
| 084129.0+002645 | 1 | 0.000000 | 0.000000 |
| 084129.0+002645 | 8 | 0.005746 | 0.013219 |
| 084129.0+002645 | 64 | 0.001933 | 0.004369 |
| 084129.0+002645 | 512 | 0.000210 | 0.001338 |
| 090129.1-013853 | 1 | 0.000000 | 0.000000 |
| 090129.1-013853 | 8 | 0.001262 | 0.004437 |
| 090129.1-013853 | 64 | 0.000503 | 0.000250 |
| 090129.1-013853 | 512 | 0.000333 | 0.000248 |
| 092031.8+040621 | 1 | 0.000000 | 0.000000 |
| 092031.8+040621 | 8 | 0.016857 | 0.033839 |
| 092031.8+040621 | 64 | 0.002548 | 0.003286 |
| 092031.8+040621 | 512 | 0.000902 | 0.003436 |

### The verdict

**V2 and V4d PASS.**  Every neighbour is 10-100 Mpc away and every probe
sits inside a 4 Mpc sphere, so a neighbour is point-like to the probe by
a wide margin: resolving each one into 8 to 512 components moves
`|g_ext|` by at most 0.034 dex and settles at 0.003 dex, against a
between-object spread of 0.61 dex.  The external variables are catalogue-invariant to the precision
that matters.

**V3 FAILS as a catalogue quantity and survives only as a continuum
functional.**  Represented the way the catalogue actually represents an
object — one row, all the mass, at the centroid — W is wrong by 0.09 to 1.06 dex, against a between-object spread of 0.66 dex.  The drift falls only
as `N^-0.42 ... N^-0.46` with **no plateau**, which places W in the
`convergent-quadrature` class of `screen.py`'s taxonomy, not the
`coherence-limited` class: there is a continuum limit, but no physical
scale emerges, so at any finite catalogue resolution the answer is set
by how finely the mass happens to be tabulated.  Extrapolating the
measured slope, reaching 0.01 dex needs of order 195-23331 rows per object.

This lane therefore evaluates V3's self term as the exact continuum
angular integral of the fitted density profile, which is well defined —
but that is only possible because eFEDS publishes a resolved density fit
for the object at the centre of each field.  For every OTHER well in the
network only a catalogue row exists, and for those the same construction
would be uncontrolled at the 1 dex level if they were ever close enough
to matter.  **A directionless inverse-square well strength is not a
quantity a catalogue can deliver; it is a quantity a mass map can
deliver.**

Note also the asymmetry that makes the gate decisive for V3 and not for
V2: the vector sum's continuum limit is exactly `G M(<r)/r^2`, a
quantity nobody would ever compute by summing rows, whereas W's
continuum limit is a genuinely new functional with no closed form.

---

## 4.  Responsiveness — every statistic was checked for blindness

The programme has caught five monotone-blind statistics.  Two levels of
check were run.

**The constructions.**  Each variable must move when the quantity it
claims to measure moves.

| construction knob | response | spread |
|---|---|---|
| V1 vs the five boundary rules | mean value spans 0.0664 dex | per-system spread sd 0.1836 dex |
| V3 vs the smoothing scale 20 -> 200 kpc | 0.0544 dex | per-system 0.0197 dex |
| V2 vs a global rescale of the well masses | 1.0000 dex per dex (exact, by construction) | 1.8e-15 |
| V4a vs the density slope alpha | 0.1980 per unit alpha | 0.1499 |
| V4b vs the density beta | -0.1813 per unit beta | 0.1266 |

Every construction is responsive; none is blind.  Note that V1's
boundary-rule spread is small ONLY because each rule is evaluated inside
its own `0.8 r_ref`; the per-system spread is 0.184 dex, and Run
AH.6's 0.87 dex figure compares two GLOBAL prescriptions over a whole
population, which is a different and larger quantity.

**The estimator.**  `d(beta-hat)/d(beta_injected)`, with the spread.

| variable | estimator | beta-hat at injected 0 | at injected 0.30 | slope | slope error |
|---|---|---|---|---|---|
| V1 potential depth | within_object | +0.0037 | -0.0229 | -0.0889 | 1.0570 |
| V1 potential depth | within_class | -0.3031 | -0.2828 | +0.0677 | 0.1063 |
| V2 vector g_ext | within_object | -0.0383 | +0.0691 | +0.3581 | 1.0385 |
| V2 vector g_ext | within_class | +0.0198 | +0.1561 | +0.4544 | 0.0795 |
| V3 directionless W | within_object | -0.2119 | -0.0222 | +0.6322 | 0.6561 |
| V3 directionless W | within_class | -0.2375 | -0.1890 | +0.1617 | 0.1658 |
| V4a tidal magnitude | within_object | -0.1142 | +0.0243 | +0.4616 | 0.5861 |
| V4a tidal magnitude | within_class | -0.3034 | -0.3772 | -0.2461 | 0.2678 |
| V4b tidal shape | within_object | -0.0034 | +0.3537 | +1.1904 | 0.5195 |
| V4b tidal shape | within_class | -0.1499 | -0.0430 | +0.3563 | 0.1345 |

---

## 4b.  How many objects is each beta actually made of?

A variable that is nearly constant inside most objects can still
show Fisher information for beta if a handful of objects happen to
have a close neighbour.  Objects are ranked by their contribution
to the Fisher information for beta and the fit is repeated with the
top contributors removed.

| variable | estimator | beta, all 248 | -1% | -5% | -10% | top 1% share of Fisher info | top 5% share |
|---|---|---|---|---|---|---|---|
| V1 potential depth raw | within-object | +0.2540 | +0.2547 | +0.2554 | +0.0768 | 0.054 | 0.209 |
| V1 potential depth raw | within-class | +0.0316 | +0.0636 | +0.1099 | +0.1008 | 0.152 | 0.384 |
| V1 potential depth perp | within-object | -0.3210 | -0.3112 | -0.3299 | -0.3357 | 0.128 | 0.322 |
| V1 potential depth perp | within-class | -0.0141 | -0.1419 | -0.2651 | -0.3269 | 0.224 | 0.567 |
| V2 vector g_ext raw | within-object | -0.1730 | -0.1737 | -0.1732 | +1.7966 | 0.686 | 0.994 |
| V2 vector g_ext raw | within-class | +0.0368 | +0.0488 | +0.1205 | +0.1169 | 0.487 | 0.808 |
| V2 vector g_ext perp | within-object | -2.0000 | -2.0000 | -2.0000 | -2.0000 | 0.643 | 0.992 |
| V2 vector g_ext perp | within-class | +0.0401 | +0.0496 | +0.1223 | +0.0964 | 0.434 | 0.795 |
| V3 directionless W raw | within-object | +0.1290 | +0.1319 | +0.1324 | +0.1117 | 0.052 | 0.209 |
| V3 directionless W raw | within-class | +0.1485 | +0.1386 | +0.2281 | +0.2685 | 0.076 | 0.266 |
| V3 directionless W perp | within-object | -0.4388 | -0.0040 | -0.4048 | -0.3827 | 0.463 | 0.814 |
| V3 directionless W perp | within-class | +0.0915 | +0.0716 | +0.1502 | +0.2079 | 0.269 | 0.674 |
| V4a tidal magnitude raw | within-object | +0.1750 | +0.1460 | +0.1951 | +0.1303 | 0.090 | 0.241 |
| V4a tidal magnitude raw | within-class | +0.2405 | +0.2825 | +0.3458 | +0.2865 | 0.140 | 0.288 |
| V4a tidal magnitude perp | within-object | +0.0478 | +0.0376 | +0.0809 | +0.2681 | 1.000 | 1.000 |
| V4a tidal magnitude perp | within-class | +0.0001 | +0.0094 | +0.0213 | +0.0547 | 0.998 | 0.999 |
| V4b tidal shape raw | within-object | +0.0794 | +0.0702 | +0.0773 | +0.0775 | 0.102 | 0.360 |
| V4b tidal shape raw | within-class | +0.1453 | +0.1393 | +0.1777 | +0.2330 | 0.079 | 0.333 |
| V4b tidal shape perp | within-object | -0.0779 | -0.1036 | -0.0978 | -0.1127 | 0.256 | 0.585 |
| V4b tidal shape perp | within-class | +0.0262 | +0.0062 | +0.0663 | +0.1297 | 0.189 | 0.545 |
| V4d external tidal raw | within-object | -2.0000 | -2.0000 | -2.0000 | -2.0000 | 0.693 | 0.990 |
| V4d external tidal raw | within-class | +0.0379 | +0.0533 | +0.1087 | +0.1430 | 0.481 | 0.787 |
| V4d external tidal perp | within-object | -1.8178 | -1.8199 | -1.8357 | -0.7846 | 0.654 | 0.987 |
| V4d external tidal perp | within-class | +0.0430 | +0.0568 | +0.1134 | +0.1194 | 0.430 | 0.767 |
| -- radius tilt raw | within-object | -0.1563 | -0.1573 | -0.0828 | -0.1384 | 0.055 | 0.240 |
| -- radius tilt raw | within-class | -0.3283 | -0.3287 | -0.2753 | -0.3379 | 0.053 | 0.232 |
| -- acceleration tilt raw | within-object | +0.0876 | +0.2435 | +0.0951 | +0.0564 | 0.046 | 0.184 |
| -- acceleration tilt raw | within-class | +0.1166 | +0.1370 | +0.2047 | +0.1814 | 0.070 | 0.229 |

This separates the table cleanly.  For the internally-sourced
variables (V1, V3, V4a, V4b, raw) the top 1% of objects carry 5-10% of the
information and beta barely moves when they are dropped.  For the
EXTERNAL variables the top 1% -- two objects out of 248 -- carry 69% (V2) and 69% (V4d) of it, and
the top 5% carry over 99%.  Dropping 10% of objects moves V2's
within-object beta from -0.173 to +1.797.  The
apparent within-object leverage on an external field is a
measurement of two clusters that happen to have a catalogued
neighbour within a few Mpc -- which is also where the point-mass
treatment of that neighbour is least defensible.  Note too that
V4a's RESIDUALISED information is 100% in the top 1%, which is why its
Fisher error bar in section 2b is absurdly small.

**The slopes are well below one, and that is the power statement.**
An injected effect is defined on the TRUE density profile; the
analyst measures it through the PUBLISHED one, which differs by the
published error.  The resulting attenuation is real, not a bug, and
it means the naive `1.96 sd(beta|H0)` in section 2c understates the
true amplitude this design can exclude by `1/slope`:

| variable | estimator | slope | 95% detectable beta-hat | implied 95% detectable TRUE beta |
|---|---|---|---|---|
| V1 potential depth | within-object | -0.089 +- 1.057 | 2.032 | not bounded: the slope is consistent with zero |
| V1 potential depth | within-class | +0.068 +- 0.106 | 0.134 | not bounded: the slope is consistent with zero |
| V2 vector g_ext | within-object | +0.358 +- 1.038 | 1.961 | not bounded: the slope is consistent with zero |
| V2 vector g_ext | within-class | +0.454 +- 0.080 | 0.204 | 0.45 |
| V3 directionless W | within-object | +0.632 +- 0.656 | 1.048 | not bounded: the slope is consistent with zero |
| V3 directionless W | within-class | +0.162 +- 0.166 | 0.176 | not bounded: the slope is consistent with zero |
| V4a tidal magnitude | within-object | +0.462 +- 0.586 | 1.073 | not bounded: the slope is consistent with zero |
| V4a tidal magnitude | within-class | -0.246 +- 0.268 | 0.316 | not bounded: the slope is consistent with zero |
| V4b tidal shape | within-object | +1.190 +- 0.519 | 0.645 | 0.54 |
| V4b tidal shape | within-class | +0.356 +- 0.134 | 0.244 | 0.68 |

Read the last column literally.  Where the slope is consistent with
zero at this Monte-Carlo size, the lane has NOT set an upper limit
on that variable; it has only failed to find it.  Where the slope is
resolved, the excludable true amplitude is several tenths of a dex
per standard deviation of the variable -- far above the effect the
cross-class step would need.

---

## 5.  Frozen transfer to the held-out half, touched once

beta was fitted on TRAIN and FROZEN.  The per-object intercepts are
object-specific nuisance parameters and are refitted on the held-out
objects — without them the within-object model is not defined on a new
object at all; beta, the hypothesis, is never refitted.

**raw variables.**

| variable | WO dchi2 | WO dBIC | WC dchi2 | WC dBIC |
|---|---|---|---|---|
| V1 potential depth | +1.494 | +5.92 | +1.870 | +5.54 |
| V2 vector g_ext | +2.502 | +4.91 | -0.581 | +7.99 |
| V3 directionless W | +2.776 | +4.64 | +8.283 | -0.87 |
| V4a tidal magnitude | -2.079 | +9.49 | +6.555 | +0.86 |
| V4b tidal shape | +6.012 | +1.40 | +4.613 | +2.80 |
| V4d external tidal | -16.019 | +23.43 | -0.640 | +8.05 |
| -- radius tilt | +2.093 | +5.32 | +3.369 | +4.04 |
| -- acceleration tilt | +0.729 | +6.68 | +6.966 | +0.45 |

**perp variables.**

| variable | WO dchi2 | WO dBIC | WC dchi2 | WC dBIC |
|---|---|---|---|---|
| V1 potential depth | +4.149 | +3.26 | +0.233 | +7.18 |
| V2 vector g_ext | -27.581 | +34.99 | -1.035 | +8.45 |
| V3 directionless W | -13.835 | +21.25 | -1.689 | +9.10 |
| V4a tidal magnitude | -5.769 | +13.18 | -0.006 | +7.42 |
| V4b tidal shape | -9.567 | +16.98 | -0.424 | +7.84 |
| V4d external tidal | -12.926 | +20.34 | -1.128 | +8.54 |
| -- radius tilt | +2.093 | +5.32 | +3.369 | +4.04 |
| -- acceleration tilt | +0.729 | +6.68 | +6.966 | +0.45 |

Positive `dchi2` means the frozen model fits the held-out half better
than the same model with beta = 0.  A negative `dBIC` is the only case
in which adding the variable is preferred on the held-out data.

Of the 32 frozen held-out evaluations, 1 reaches a negative dBIC:
  * `x3_raw` within-class: dBIC = -0.87

On the Jeffreys scale a |dBIC| below 2 is "not worth more than a
bare mention", and the environment-free acceleration tilt sits at dBIC = +0.45 in the
same column.  Nothing here transfers at a level that would survive the
32-fold multiplicity of this lane.

---

## 6.  Sensitivity

| setting | within-object beta | within-class beta |
|---|---|---|
| V1_fixed10Mpc | +0.2540 | +0.0316 |
| V1_fixed5Mpc | +0.3644 | +0.1133 |
| V1_fixed3Mpc | +0.0484 | +0.0560 |
| V1_2xR500 | -0.0327 | +0.0073 |
| V1_10xrs | +0.2306 | +0.2323 |
| V3_eps20kpc | +0.1324 | +0.1530 |
| V3_eps50kpc | +0.1290 | +0.1485 |
| V3_eps200kpc | +0.1169 | +0.1297 |

**Exact versus linearised.**  The headline fits linearise the
forward model in beta to second order around beta = 0, using the
central first and second differences of the FULL nonlinear model at
+-0.25.  The same linearisation is used inside the null, so the
null-calibrated z is self-consistent.  Against a full nonlinear grid:

| variable | estimator | exact beta | linearised beta |
|---|---|---|---|
| V1 potential depth | within_object | +0.1962 | +0.2540 |
| V1 potential depth | within_class | +0.0402 | +0.0316 |
| V3 directionless W | within_object | +0.1587 | +0.1290 |
| V3 directionless W | within_class | +0.1642 | +0.1485 |
| V4a tidal magnitude | within_object | +0.1777 | +0.1750 |
| V4a tidal magnitude | within_class | +0.2254 | +0.2405 |

---

## 7.  What the data can actually support

**Within-class, no environmental variable separates itself from the two
environment-free controls.**  On training chi2 the largest within-class
improvement of anything tested is a bare radius tilt (dchi2 = 19.2).  Against
its own simulated null the largest z belongs to the bare ACCELERATION
tilt (+5.97), above every environmental variable.  On the frozen held-out
half the best environmental result is dBIC = -0.87, with the
acceleration tilt at +0.45 beside it.  That reproduces Run AI's finding — where
`M3 + gamma log r` won on training chi2 and potential depth came last of
ten on BIC — from a different estimator and a different
parameterisation, and it extends it to all four variables.

**Within-object, the design is real but the variables are not.**  The
per-object intercept does exactly what the brief wants: it removes the
class label, the mass, the redshift and every selection effect, so no
Simpson's paradox of Run AD's kind can occur.  What it cannot remove is
that the only variables with radial structure inside an object are
71-98% reconstructible from (log g_bar, log r).  A within-object
estimator on such a variable is measuring a radius tilt with an
environmental label on it.

**The two genuinely environmental variables have no within-object
leverage, and that is a theorem, not a data limitation.**  An external
field is constant across a small object to leading order; its first
radial derivative IS the external tidal tensor, which is 2.9e-04 of the
internal one here.  So V2 can only ever be tested BETWEEN objects, which
is the estimator the brief was trying to escape.  Any future attempt to
test an external-field variable within an object needs objects embedded
in a field that varies on the object's own scale — cluster member
galaxies inside their host, not clusters inside a 140 deg^2 survey.

**What this lane cannot establish.**

* Nothing about V2 or V4d at amplitudes that matter.  The external field
  of the eFEDS catalogue is 4-5 dex below a0.  Even a large beta on a
  variable that small is not the cross-class effect; testing it here is
  testing an ORDERING, not an amplitude.
* The line-of-sight geometry of the well network.  Bahar+2022 publishes
  no redshift error.  Under an assumed `sigma_z = 0.005(1+z)` the
  implied radial distance error is 29 Mpc against a median neighbour separation of 33 Mpc, so the
  3-D network is only marginally resolved along the line of sight.  The
  null includes that jitter; the point estimates do not correct for it.
* Whether the catalogue is a fair census of the mass field.  It is an
  X-ray flux-limited list of 542 concentrations in 140 deg^2.  Field
  galaxies and low-mass groups are missing entirely.
* Anything about V3 at catalogue resolution — see the coarse-graining
  verdict.

---

## 8.  The programme's failure-mode checklist, explicitly

| failure mode | what was done |
|---|---|
| Shared-denominator / shared-quantity artefacts | A separate Monte
Carlo null per variable per estimator, redrawing every published density
parameter plus an assumed redshift error, with the shear regenerated
independently. Every estimate is quoted against its own null, and the
nulls differ between variables. |
| Monotone-invariant statistics | Both levels checked: each
construction against its own knobs (section 4, all responsive), and the
estimator against an injected signal (`d(beta-hat)/d(beta_inj)` with the
spread printed). |
| Refitting on the held-out set | beta fitted on TRAIN, frozen, held-out
set scored once. Per-object intercepts are declared nuisance parameters
and are refitted, which is stated rather than hidden. |
| Silent extraction failures | Row and column counts asserted on every
ingest (542 x 19, 542 x 40, 5411 x 13, 496 systems, 3365 points); the
catalogue identifier `J/A+A/661/A7` echoed; the M_gas,500 reproduction
gate re-run. |
| Test bugs that look like solver bugs | The reduced-shear
linearisation was checked against the full nonlinear amplitude profile
and against `kappa_max`; the exact-versus-linearised beta comparison is
reported rather than assumed. |
| Coarse-graining invariance | Section 3, including the demonstration
that uniform refinement is toothless for every mass exponent. |
| Potential-gauge invariance | V1 is an operational DIFFERENCE with five
prespecified rules, one declared primary, each evaluated inside its own
`0.8 r_ref`. |
| Weak lensing measures reduced shear, not mass | The observable is
per-cluster, per-bin metacalibration-corrected `g_t`. No mass
catalogue, no NFW fit, no convergence map enters anywhere. |
| Programme-level multiplicity | This lane ran up to 32 fits (8
variables x 2 estimators x raw/residualised, minus the residualised
competitors, which are the nuisance basis itself). Nothing here is a
discovery
claim, so no look-elsewhere correction is quoted; if any of these were
promoted, the correction would be a factor of about 32. |

A note on what is NOT protected: the eFEDS + DECADE sample has now been
used by Run AI and by this lane.  It is validation data, not a fresh
sample.  KiDS and wide binaries were not loaded, looked at, or
referenced.

---

## 9.  One correction to the brief's premise

The brief says variables 1 and 4 order the key probe oppositely, citing
Run AH's cluster member galaxy at `|T| = 5.54e-31` against a cluster
shell's `3.66e-34`.  That is right, and this lane can now say WHY in
closed form rather than as an empirical curiosity:

```
    |T~| = sqrt(6) (g/r) |1 - rho/<rho>|
```

For a spherical baryonic source the tidal magnitude is `g/r` times a
bounded shape factor.  A member galaxy at 20 kpc has `g/r` larger than a
cluster shell at 700 kpc by roughly `(g_gal/g_cl)(r_cl/r_gal)`, which is
two to three orders of magnitude, while `|Phi|` adds the host's
contribution and so orders them the other way.  The 151x is therefore
not a property of the tidal invariant as an environmental variable — it
is `g/r` at two very different radii.  On the eFEDS sample the same
statement reads: V4a is 97.2% a function of (log g_bar, log r) at the
shear-measured radii.  **The tidal gate's advantage over the potential
gate in Run AH is an advantage of `g/r` over `Phi`, and `g/r` is not
environmental.**  That does not make the tidal gate wrong — Run AH's
member screen still separates the families by three orders of magnitude
— but it does mean the tidal gate should be described as a
LOCAL-KINEMATIC gate, not an environmental one.

