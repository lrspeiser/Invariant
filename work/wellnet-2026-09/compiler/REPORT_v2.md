# The pre-data admissibility compiler, v2: four corrections

Supersedes the interpretation in `REPORT.md` (Run AM). **`REPORT.md` is
unchanged and still stands for what it measured**; this document
re-partitions the same measurements, corrects one published claim that
was false as stated, adds an external control suite, and adds one basis
element the searched grammar never contained.

Four corrections, all from an external review:

1. the "97.2% rejected" figure conflated four verdicts that are not
   scientifically equivalent;
2. **"a field with curl cannot come from an action" is FALSE as
   stated** and was published in Run AR;
3. "35 of 35 tests agree with previous programme verdicts" is
   regression testing, not validation;
4. the grammar could not express an external tidal axis, so the 2-D
   shear phase channel was aimed at a hypothesis it did not contain.

## 0. Data statement

**No observational data of any kind is opened by this lane.** The only
file read is `../tournament/tournament.json`, a record of a previous
lane's own candidate list. KiDS and the wide binaries are never loaded,
listed or referenced; neither is SPARC nor any cluster catalogue.
`test_no_observational_data_is_opened` asserts it mechanically by
intercepting `open` -- and the interception now covers every code path
added here (the disc geometry and the curl module, the external control
suite, the u-space test, the external-axis element): **0
files opened, 0 outside the lane**.

Two sets of *constants* are quoted verbatim from Run AR's report so its
curl table can be reproduced like-for-like: the four component masses of
its Milky Way caricature, calibrated separately for each law, and its
frozen `a0` values. They are floats in `compiler.py`. Quoting a number
from a previous lane's report is not a data read and does not trip the
interception, and none of the conclusions below depends on them --
the identity is verified on the lane's own field either way.

---

## 1. FIX 1 -- the rejection taxonomy

`REPORT.md` reported `3,036 / 3,123 = 97.2% rejected`. That number sums
bins whose scientific content differs, and the sum is the least
informative thing about it. Re-partitioned:

### 1.1 The corrected headline

```
    mathematically_inconsistent              390    12.5%
    representation_convention_dependent     1482    47.5%
    physically_incomplete_as_written        1008    32.3%
    not_decidable_on_this_bench              100     3.2%
    non_identifiable_on_this_bench            52     1.7%
    admissible                                91     2.9%
```

That is: **12.5% mathematically inconsistent**, **47.5% representation convention dependent**, **32.3% physically incomplete as written**, **3.2% not decidable on this bench**, **1.7% non identifiable on this bench**, **2.9% admissible**.

The old single figure is retained for continuity -- **3032 of 3123 = 97.1%** are rejected -- but it is no longer
the headline, because the rejections are not the same kind of thing.

### 1.2 What each bin means and what repairs it

| bin | what it says | repair |
|---|---|---|
| `mathematically_inconsistent` | ill-posed PDE, indefinite kinetic operator, a violated DECLARED symmetry, or mesh-dependence with no physical scale | **none named.** Dead inside the declared class and outside it |
| `representation_convention_dependent` | the prediction depends on the arbitrary additive zero of Phi, or on a cataloguer's partition. Kills the FORMULA AS WRITTEN, not every theory of its kind | declare the convention (a boundary rule, an environmental scalar, a partition-independent functional). The physics content does not change |
| `physically_incomplete_as_written` | no action for the law AS WRITTEN in the declared class; a variational completion may exist | promote the gating field to a dynamical one. **The theory changes** -- this is how AQUAL supplies an action for MOND |
| `not_decidable_on_this_bench` | the solver cannot reach its declared tolerance at THIS amplitude. A property of the setting and of float64, not of the law | a smaller amplitude, or a better-conditioned solve |
| `non_identifiable_on_this_bench` | internally consistent; no experiment on this bench can identify it | **a different experiment, not a different theory** |
| `outside_declared_model_class` | GATE 4 does not adjudicate this model class | none needed: the gate has no jurisdiction |
| `admissible` | passes every gate that applies | -- |

Severity order, used to pick ONE primary bin per candidate and declared
in the source as `TAXONOMY_SEVERITY`, is by **what it takes to repair**:

```
    mathematically_inconsistent > representation_convention_dependent > physically_incomplete_as_written > not_decidable_on_this_bench > non_identifiable_on_this_bench
```

`representation_convention_dependent` ranks above
`physically_incomplete_as_written` deliberately: a formula that depends
on an arbitrary constant has no determinate content, so the variational
question does not even arise for it. Every candidate's FULL (non-exclusive)
defect list is recorded alongside the primary bin, so the partition can
be re-cut without re-running the compiler.

### 1.3 Per-gate contribution to each bin

| gate | mathematically inconsistent | representation convention dependent | physically incomplete as written | not decidable on this bench | non identifiable on this bench |
|---|---|---|---|---|---|
| gate1_constant_K | -- | -- | -- | -- | **150** |
| gate2_potential_gauge | -- | **106** | -- | -- | -- |
| gate3_coarse_graining | **390** | **1170** | -- | -- | -- |
| gate4_scalar_potential_integrability | **390** | **1482** | **1008** | **100** | -- |

Cells count **defect instances**, so a candidate flagged by two gates
appears in both rows; the §1.1 counts are the primary-bin partition and
sum to 3123.

Read the two together. **One gate does almost all of the work, and it is
not the work the old headline implied.** GATE 4 supplies 1482 representation-dependent defects and 1008 incomplete-as-written ones against only 390 mathematically inconsistent -- so the great majority of what Run AM counted as "rejected before any data" is a **named repair**, not a refutation. GATE 3 flags the same 390 candidates as GATE 4 does in that bin: they are the well-network settings whose response has no continuum limit at all, and they are the only ones this bench can call dead. GATE 1's 150 are a different claim again -- consistent theories this bench cannot see.

### 1.4 Defect census (non-exclusive: a candidate may carry several)

| defect | n |
|---|---|
| `functional_derivative_undefined_without_a_partition` | 1560 |
| `coarse_graining_dependent` | 1560 |
| `not_variational_as_written` | 1004 |
| `response_reads_an_undetermined_additive_constant` | 312 |
| `degenerate_with_a_coordinate_stretch` | 150 |
| `potential_zero_point_changes_the_verdict` | 106 |
| `unsolvable_at_this_amplitude` | 100 |
| `K_of_u_is_not_a_gradient` | 2 |
| `jacobian_asymmetric` | 2 |

### 1.5 The correction that mattered most inside FIX 1

The first cut of this taxonomy put **2,075 candidates (66%) in
`mathematically_inconsistent`**, on the strength of GATE 4's numerical
health check -- `cond(K) > 1e8` across the probes. That was wrong twice
over, and finding it is the reason the taxonomy is worth having:

* a badly conditioned but uniformly elliptic operator is **not an
  ill-posed one**. `cond(K) > 1e8` is the point beyond which a float64
  conjugate-gradient solve cannot reach a 1e-11 residual. That is a fact
  about the solver and the fitted amplitude, not about the law;
* GATE 4's control flow returns on the health check **first**, which let
  a solver limitation mask the structural defect underneath. Of the
  2,075, **1,975 carried a structural defect as well** and only
  **100 were
  conditioning alone**.

The taxonomy therefore consults the structural findings -- which the
gate records whether or not it returned on them -- **before** the
conditioning one, and gives conditioning its own honest bin,
`not_decidable_on_this_bench`. The mathematically-inconsistent bin fell
from 66% to **12.5%**.

---

## 2. FIX 2 -- the curl claim was false, and here is the exact result

### 2.1 The published error

Run AR measured `max|curl g| x 10 kpc / |g|` and the programme record
(AR.3, and the master record) then said that **a field with curl cannot
come from an action.** That is false. The Lorentz force has non-zero
curl and follows from

```
    L = (1/2) m v^2 + q A.v - q phi
```

because it is velocity-dependent and carries a vector potential;
gravitomagnetism is the exact gravitational analogue and is a limit of
general relativity.

### 2.2 The exact result that replaces it

For an **algebraic vector prescription** built on a curl-free Newtonian
field,

```
    g_alg = nu(|g_N|) g_N ,        curl g_N = 0

    curl g_alg = curl(nu g_N) = (grad nu) x g_N + nu (curl g_N)
               = (grad nu) x g_N
               = nu'(|g_N|) ( grad|g_N| ) x g_N
```

which vanishes identically **iff `grad|g_N|` is parallel to `g_N`**, i.e.
iff the level surfaces of `|g_N|` are the field's own -- the spherical
case. In a nonspherical system it is generically non-zero, and its size
is set by how fast `nu` is turning.

So a non-zero curl shows that **the ALGEBRAIC VECTOR PRESCRIPTION is not
the gradient of a single static scalar potential.** It does **not** show
that MOND has no action. AQUAL was constructed precisely to supply one,
and its field-equation form is curl-free by construction.

### 2.3 Verified on the lane's own field, to round-off

Both sides computed independently with **exact derivatives** (complex
step, which has no subtractive cancellation, so the residual is round-off
and not a differencing artefact), on a declared closed-form disc:

| row | identity residual, max rel | `curl g_N` control | exact `max q` | at Run AR's h = 0.05 kpc | Run AR recorded | rel |
|---|---|---|---|---|---|---|
| `newton` | n/a (both sides at round-off) | 7.67e-16 | 7.67344e-16 | 3.81836e-05 | 3.81836e-05 | 0 |
| `rar` | **2.52e-14** | 6.68e-16 | 0.0484537 | 0.0482477 | 0.0482477 | 0 |
| `aqual` | **2.02e-14** | 4.80e-16 | 0.048958 | 0.0487499 | 0.0487499 | 5.43e-13 |
| `tidal_scalar` | **1.51e-14** | 6.65e-16 | 1.07847 | 1.08253 | 1.08253 | 2.05e-14 |

Reading that table:

* **the identity holds to 2.52e-14** relative, which is round-off;
* the Newtonian control returns 7.67e-16 in the continuum and 3.82e-05 at Run AR's own
  step -- the estimator is clean and its finite-difference floor is
  measured, so every number above that floor is the law's own;
* **all four of Run AR's analytic rows are reproduced**, by an
  independent implementation, to 5.43e-13 relative or
  better. Each row uses Run AR's own per-law mass calibration, since it
  fitted the baryons separately for every law;
* **the RAR's 0.048 is a PREDICTION of the identity, not an anomaly.**
  Its continuum value is 0.0484537; Run AR's
  0.0482477 is that number seen through a
  central difference at h = 0.05 kpc. The FD residual against the
  identity converges at order
  2.01 / 2.00 / 2.00 / 2.00 / 2.00 in h, i.e. second order, which is what "the finite difference is
  approximating the identity" means;
* **the AQUAL row is the decisive one.** AQUAL is the theory that was
  built to give MOND an action, and its ALGEBRAIC form still carries a
  curl of 0.049. If a non-zero
  curl meant "no action", this row alone would refute the claim;
* the tidal-gated row generalises the identity. With
  `a0 -> a0[1 + A W(|T|)]` the multiplier `F` depends on **two** fields,
  `grad F` picks up a tidal term, and `curl(F g_N) = (grad F) x g_N`
  still holds exactly -- residual 1.51e-14.
  Run AR's 1.08 is that.

### 2.4 The other side of the identity: why this bench was blind to it

In a spherical system `grad|g_N| || g_N`, so `(grad nu) x g_N == 0` and
the algebraic prescription **is** a gradient. Measured on the compiler's
own spherical probe:

```
    max relative antisymmetry, rar      2.00e-10
    max relative antisymmetry, aqual    1.95e-10
```

**Every spherical channel in this programme -- including this compiler's
own radial Jacobian -- is therefore blind to the obstruction the curl
measures.** That is a measured property of the bench, not an assumption,
and it is why GATE 4 needed a second, non-spherical channel (§2.6).

### 2.5 The gate is renamed, with its scope declared

* **was**: `gate4_reciprocity_action`
* **is**: `gate4_scalar_potential_integrability`
* **title**: *scalar-potential integrability under the declared static, velocity-independent model class*

**In scope.** Laws written as a STATIC, VELOCITY-INDEPENDENT prescription for the acceleration of a test particle, in which the whole content is a single scalar potential (equivalently, a conductivity tensor K acting on grad Phi_N with Phi_N still Newtonian). For these, and ONLY these, the gate's criterion is exact: the law comes from an action with Phi_N solving Poisson if and only if K(u)u is a gradient in u = grad Phi_N.

**NOT in scope. The gate returns no verdict on any of these and labels
them instead:**

* **`velocity_dependent`** -- a velocity-dependent force. The Lorentz force is the standard example: it has non-zero curl in the position-space sense and comes from L = (1/2) m v^2 + q A.v - q phi. Nothing measured by this gate bears on such a law.
* **`vector_potential_gravitomagnetic`** -- a gravitomagnetic / vector-potential sector. g = -grad Phi - dA/dt + v x (curl A) is derived from an action and has non-zero curl by design. This gate's criterion -- 'is the static field the gradient of one scalar?' -- is simply not the right question for it.
* **`extra_propagating_field`** -- a theory with extra PROPAGATING degrees of freedom (a dynamical scalar, vector or tensor beyond Phi). The action exists but its Euler-Lagrange system is not a single scalar equation for Phi, so the QUMOND-form integrability criterion does not decide it.
* **`relativistic_completion`** -- a relativistic completion. This bench is a weak-field bench and has no jurisdiction over one.

The old key `gate4_reciprocity_action` is kept as a **deprecated
alias** in every result dict, pointing at the same tuple, so committed
readers of `REPORT.md` and Run AM do not break. It is not a member of
`GATES`, so it cannot double-count.

### 2.6 A new, non-spherical channel inside GATE 4

The gate's declared criterion -- *the QUMOND-form law comes from an
action with `Phi_N` still solving Poisson iff `K(u)u` is a gradient in
`u = grad Phi_N`* -- was previously tested only through a **spherical**
radial Jacobian, which §2.4 shows is blind to any obstruction whose only
signature is a direction. `u_space_integrability` now tests it
**directly**, on a 3-D cloud of `u` vectors, by measuring the
antisymmetry of `dM_i/du_j` for `M(u) = K(u)u`. Floor measured on laws
that are gradients exactly, not assumed:

```
    A1_aqual               2.24e-10
    A2_qumond              2.24e-10
    A3_qumond_rar          2.02e-10
    X0_newton              0
    tensor_d_gn_gated      2.98e-10
    declared floor         1.00e-07
```

This is what decides the external-axis element in §4, and the spherical
Jacobian could not have.

---

## 3. FIX 3 -- external positive controls

`REPORT.md`'s "35 of 35 agree with previous programme verdicts" is
regression testing: it risks validating the compiler against the
conclusions that shaped it. The suite below has answers fixed **outside**
this programme, by textbook field theory.

**12 of 12 agree.**

| control | required | got | bin | why the answer is known independently |
|---|---|---|---|---|
| `XC1_newton_poisson` | **ADMIT** | ADMIT | `admissible` | the canonical variational field theory of gravity in the weak field. |
| `XC2_aqual` | **ADMIT** | ADMIT | `admissible` | Bekenstein & Milgrom 1984 constructed AQUAL PRECISELY to give MOND an action. |
| `XC3_qumond` | **ADMIT** | ADMIT | `admissible` | QUMOND is a bi-potential Lagrangian theory (Milgrom 2010); K(u)u is a gradient in u = grad Phi_N by construction for nu(\|u\|)u. |
| `XC4_yukawa_from_action` | **ADMIT** | ADMIT | `admissible` | a Yukawa scalar is the textbook example of a variational modification: its two-point kernel is a function of \|x - y\| alone, so it is reciprocal and its functional Jacobian is symmetric exactly. |
| `XC5_symmetric_nonlocal_action` | **ADMIT** | ADMIT | `admissible` | a nonlocal action with a SYMMETRIC kernel is variational and reciprocal by construction. |
| `XC6_scalar_tensor_weak_field` | **ADMIT** | ADMIT | `admissible` | the weak-field limit of a scalar-tensor theory is a Newtonian term plus a Yukawa term with coupling alpha = 1/(3 + 2 omega); omega = -1 is the low-energy string / dilaton value, so alpha = 1 exactly. |
| `XC7_vector_potential_nonzero_curl` | **OUTSIDE-CLASS** | OUTSIDE-CLASS | `outside_declared_model_class` | THE SHARPEST TEST. |
| `XC8_non_reciprocal_catalogue_force` | **REJECT** | REJECT | `mathematically_inconsistent` | Newton's third law is violated by construction: the bracket is not symmetric under x <-> x'. |
| `XC9_coarse_graining_well_count` | **REJECT** | REJECT | `mathematically_inconsistent` | a law whose value depends on how finely a mass distribution happens to be tabulated has no continuum limit and therefore no physical content. |
| `XC10_indefinite_kinetic_energy` | **REJECT** | REJECT | `mathematically_inconsistent` | a kinetic term that is not sign-definite carries a ghost: the energy is unbounded below and the elliptic operator changes type. |
| `XCS_yukawa_subthreshold` | **REJECT** | REJECT | `non_identifiable_on_this_bench` | the CONTRAST control. |
| `XCS2_fR_scalar_tensor_subthreshold` | **REJECT** | REJECT | `non_identifiable_on_this_bench` | the second CONTRAST control, and a real limitation this suite exposes. |

### 3.1 The sharpest test: the vector-potential force

A gravitomagnetic vector-potential force has **non-zero curl** and a
**perfectly valid action**. It is exactly the case the published claim
would have mishandled. If the compiler rejected it, the gate would still
be mis-scoped.

```
    verdict   OUTSIDE-CLASS
    failed    []
    label     ['action-based but OUTSIDE the scalar-potential class']
    bin       outside_declared_model_class
```

**Labelled, not rejected**, and `_failed` is empty -- gates 1, 2 and 3
still apply to it and it passes them. GATE 4's own reason string says:

> OUT OF DECLARED SCOPE. This gate adjudicates the 'static_scalar_potential' class only; this candidate declares 'vector_potential_gravitomagnetic', which is a gravitomagnetic / vector-potential sector. g = -grad Phi - dA/dt + v x (curl A) is derived from an action and has non-zero curl by design. This gate's criterion -- 'is the static field the gradient of one scalar?' -- is simply not the right question for it. LABELLED 'action-based but OUTSIDE the scalar-potential class'. NOT REJECTED: a non-zero curl is perfectly consistent with an action, and the published claim that it is not was wrong. This gate returns NO VERDICT on this candidate; gates 1, 2 and 3 still apply to it.

### 3.2 The two sub-threshold contrast rows, and a real limitation

GATE 1 is a statement about **identifiability**, so it necessarily
depends on a law's amplitude and range. An external control suite has to
name parameter values, and the honest test is not "does a Yukawa
admit" but **"does the same theory class move between ADMIT and
`non_identifiable_on_this_bench` -- and never into an inconsistency bin
-- as its parameters cross the threshold"**. The threshold is measured
rather than assumed, so the parameter choice is a reported measurement:

| Yukawa alpha | ranges (kpc) that escape GATE 1 |
|---|---|
| 0.1 | *none* |
| 0.3333 | *none* |
| 1 | 5, 10 |
| 3 | 3, 5, 10, 20, 100, 1000 |

Tolerance 0.04 dex; probe span 10-30 kpc.

Two consequences, both reported rather than tidied away:

* a long-range weak Yukawa (`alpha = 0.05`, range 3 Mpc) is a constant
  rescaling of `G` over every probe and lands in
  **`non_identifiable_on_this_bench`** -- correct, and *not* an
  inconsistency claim;
* **f(R) gravity fixes `alpha = 1/3`, and at that amplitude NO choice of
  range makes the deviation exceed GATE 1's 0.040 dex on this bench's
  three probes** -- a two-parameter coordinate stretch absorbs it to
  0.019 dex. That is a limitation of the *probe geometry*, it is binned
  as non-identifiable rather than rejected on principle, and the taxonomy
  exists precisely to keep the two apart. The scalar-tensor ADMIT row
  therefore uses Brans-Dicke `omega = -1` (the low-energy string
  dilaton), for which `alpha = 1/(3+2w) = 1` exactly.

---

## 4. FIX 4 -- the external tidal axis the grammar never had

Run AO established that **not one of the 3,123 candidates carries an
external tidal axis** (network 1,560 / source 780 / isotropic 783 /
**EXTERNAL 0**), so the built and calibrated 2-D shear phase channel was
pointed at a hypothesis the grammar could not express. The basis element

```
    K = exp[ f0 I + f_E e_ext e_ext^T ]
```

is added and run through the gates. `e_ext` is ONE declared global
direction fixed by the environment and **not derived from any probe's
own source** -- that is what external provenance means operationally,
and it is why Run AO measured external-axis power as not collapsing when
the source rounds.

| element | verdict | bin | failed |
|---|---|---|---|
| `F1_ext_axis_const` | **ADMIT** | `admissible` | -- |
| `F2_ext_axis_gn_gated` | **REJECT** | `physically_incomplete_as_written` | gate4_scalar_potential_integrability |
| `F3_ext_axis_tidal_gated` | **REJECT** | `physically_incomplete_as_written` | gate4_scalar_potential_integrability |

**The split between them is the whole content, and it is derived, not
fitted.**

**Constant couplings -> ADMISSIBLE.** `K` is then a constant symmetric
positive-definite tensor and `div[K grad Psi] = 4 pi G rho` is exactly
the Euler-Lagrange equation of
`L = -(1/8 pi G)(grad Psi)^T K (grad Psi) - rho Psi`. Variational by
construction; u-space antisymmetry 0.
It escapes GATE 1 on all three escapes, including **(b), the
independently measured axis** -- the external axis is misaligned with
the probes' radial direction by
60 deg,
far above the declared 10 deg. This is the one axis provenance for which
escape (b) is available at all, and it is the reason an external-axis
tensor is not degenerate with source ellipticity the way a source-axis
one is.

**Gated couplings -> `physically_incomplete_as_written`.** Writing
`(Ku)_i = a(|u|) u_i + b(|u|)(e.u) e_i`, the antisymmetric part of
`dM_i/du_j` is

```
    (e.u) b'(|u|) [ uhat_j e_i - uhat_i e_j ]
```

which vanishes only where `e || uhat`. So a gated external-axis tensor is
**not** a gradient in `u`: measured antisymmetry
0.0494
against a floor of 1.00e-07. This is the same obstruction
as the curl identity, seen in `u`-space instead of position space, and
**the spherical radial Jacobian could not have found it** (§2.4).

**NO OBSERVATIONAL CLAIM IS ATTACHED TO ANY OF THIS, AND NONE CAN BE.**
Run AO's 95% exclusion for an external-axis tensor sits at an ellipticity
of 2.11, above the geometric maximum of 1: the present sample cannot
exclude physically allowed amplitudes. This is a **grammar completeness
fix**, not evidence.

The declared radial reduction `k_r = exp(A W lambda)` with
`lambda = (e.rhat)^2 - 1/3` is the same approximation the bench already
makes for `tensor_d` and `tensor_T`; the exact projector eigenvalue is
`e^f0 [1 + (e^f_E - 1)(e.rhat)^2]` and the two differ by at most
14.2%
at the amplitude used. Reported, not assumed away.

---

## 5. Verdict invariance: none of this changed a verdict

A rename, a new scope, a new gate channel and a re-partition are all
chances to change an answer by accident. The committed pre-REPORT_v2
compiler was checked out and run against this same `tournament.json`:

```
    rejected_total             baseline   3032   now   3032
    admitted_total             baseline     91   now     91
    n_tournament_survivors     baseline     26   now     26
    rejected_without_gate4     baseline   1702   now   1702
    kills alone, gate1_constant_K          150      150
    kills alone, gate2_potential_gauge       0        0
    kills alone, gate3_coarse_graining    1560     1560
    kills alone, gate4_scalar_potential   2980     2980
```

**IDENTICAL.**
`retrospective.py` asserts it on every run. REPORT_v2 changes how
rejections are *described*, not which candidates are rejected.

One number moved for a reason that has nothing to do with this work:
`tournament.json` was itself re-run after Run AM (26 survivors now, 18
then), so `3,036 / 97.2%` in `REPORT.md` reads
`3032 / 97.1%` here. That is
the tournament's change, not the compiler's.

---

## 6. Scoping: generating candidates FROM admissible actions

The reviewer's structural suggestion is to invert the generator -- start
from

```
  L = -(1/8 pi G)(grad Phi)^T K(q, I) (grad Phi) - (Z(q)/2)|grad q|^2
      - V(q) - rho Phi
  K = exp[ f0(I) I + f_T(I) That + f_E(I) e_ext e_ext^T ]
```

vary automatically, and emit the field equations, so symmetry,
reciprocity and scalar-potential integrability hold **by construction**.
Scored against the compiler's own measured defect census:

```
    PREVENTED                     4128  of 4796 defect instances
    SURVIVES                       518  of 4796 defect instances
    SURVIVES, AND MATTERS MORE     150  of 4796 defect instances
    prevented by construction    86.1%
```

### 6.1 What survives the inversion

| gate | fate |
|---|---|
| `gate1_constant_K` | SURVIVES INTACT, and becomes the binding gate. Degeneracy with a coordinate stretch is a property of K, not of its provenance. |
| `gate2_potential_gauge` | SURVIVES INTACT. An action may still be written with \|Phi\| in V(q) or in the coupling. |
| `gate3_coarse_graining` | BECOMES VACUOUS for this grammar, because the grammar has no row-list atom. It must be KEPT for any future grammar that adds one -- the cost of keeping it is zero and the cost of losing it is family C. |
| `gate4_scalar_potential_integrability` | ITS VERDICT BECOMES VACUOUS; ITS MEASUREMENTS DO NOT. Integrability holds by construction, so the pass/fail is uninformative. But the u-space antisymmetry, the curl identity and the reciprocity measurement are still the only things that catch a BUG in the variation code -- an automatic differentiator that emits the wrong field equation produces an asymmetric Jacobian, and nothing else on this bench would notice. Demote to a self-test of the generator, do not delete. |

### 6.2 Per-defect

| defect | fate | why |
|---|---|---|
| `indefinite_kinetic_operator` | **PARTLY** | K = exp[...] with a symmetric exponent is positive definite identically, so the Phi sector is safe. Z(q) >= 0 and the sign of V''(q) are NOT automatic and still need checking: a wrong sign there is a ghost in the q sector. |
| `no_bounded_solution` | **PARTLY** | ellipticity is automatic; boundedness of the SOLUTION on a given source is not, and neither is the existence of a stable vacuum for V(q). |
| `K_of_u_is_not_a_gradient` (2) | **PREVENTED** | varying a scalar functional of grad Phi gives a gradient in grad Phi identically. |
| `coarse_graining_dependent` (1560) | **PREVENTED** | same reason: no row-list atom exists to be re-tabulated. |
| `functional_derivative_undefined_without_a_partition` (1560) | **PREVENTED** | the Lagrangian above has no row-list atom: K reads q, I, That and e_ext, all fields. A catalogue partition cannot enter a term that is not written. |
| `jacobian_asymmetric` (2) | **PREVENTED** | the second variation of an action is a Hessian and a Hessian is symmetric. |
| `not_variational_as_written` (1004) | **PREVENTED** | the field equations ARE the Euler-Lagrange system, so 'no action produces this law' is not expressible. |
| `violates_declared_reciprocity` | **PREVENTED** | a two-point kernel obtained by inverting an elliptic operator derived from an action is symmetric identically. |
| `potential_zero_point_changes_the_verdict` (106) | **SURVIVES** | as above: this is a property of what the author writes, not of how the field equations are generated. |
| `response_reads_an_undetermined_additive_constant` (312) | **SURVIVES** | nothing stops an author writing V(q) or the coupling with \|Phi\| in it. GATE 2 is untouched by the inversion and is needed exactly as it is. |
| `unsolvable_at_this_amplitude` (100) | **SURVIVES** | the conditioning of K at a given amplitude is a numerical property of the setting; the generator does not bound f0, f_T, f_E. |
| `degenerate_with_a_coordinate_stretch` (150) | **SURVIVES, AND MATTERS MORE** | a constant K is degenerate with x -> K^(-1/2) x whatever action produced it. Generating only admissible actions raises the fraction of the search that reaches GATE 1, so GATE 1 becomes the binding constraint rather than a residual one. |

### 6.3 Cost, and the recommendation

* **symbolic variation** -- the largest single item. Varying a functional of (Phi, grad Phi, q, grad q) with K a matrix exponential of invariants needs either a symbolic layer or forward-mode AD through the matrix exponential. The compiler already has the pieces for the second: `_K_of_u` is the map that would be differentiated, and `u_space_integrability` already differentiates it numerically.
* **new solver** -- the emitted system is TWO coupled equations (Phi and q), not one algebraic prescription. Every channel in the tournament scores an algebraic reduction, so either the channels change or each candidate needs a coupled solve -- which is the PDE-per-candidate cost the pre-data compiler was built to avoid.
* **reusable** -- Plummer/probe geometry, the gauge rule population, every GATE 1 and GATE 3 routine, the curl module, the u-space test, the whole taxonomy, and the external control suite all carry over unchanged.
* **rewritten** -- the candidate grammar (`Candidate`, `radial_eigen`, `predict_g`, `probe_lambda`) and GATE 4's control flow.

**WORTH DOING, but not as a replacement for the gates. The inversion removes the two bins that dominate the taxonomy and leaves the two that do not: what it cannot do is tell you whether the law it generated is MEASURABLE, which is GATE 1's job and the one that survives every reformulation. Build the generator, keep GATES 1 and 2 as they are, keep GATE 3 dormant against future grammars, and demote GATE 4 to a self-test of the variation code.**

Put precisely against §1.1: the inversion prevents the ROW-LIST half of
`representation_convention_dependent` and the whole of
`physically_incomplete_as_written`. It does **not** touch the GAUGE half
of `representation_convention_dependent` (312 + 106 defects), it does not touch
`non_identifiable_on_this_bench` (150), and it
does not touch `not_decidable_on_this_bench` (100). A generator
cannot tell you whether the law it generated is **measurable**.

---

## 7. What still could NOT be established

Everything in `REPORT.md` section 6 still stands, plus:

* **Whether any bin is right about a particular candidate's future.**
  `physically_incomplete_as_written` says a variational completion *may*
  exist, not that it does. Finding one for the tidal gate is a piece of
  work, not a formality.
* **The curl channel is reported, never verdict-bearing.** AQUAL and the
  RAR both carry a non-zero algebraic curl and both must ADMIT, so a
  curl measurement cannot be allowed to reject. It is a scope statement.
* **The u-space test only applies where `K` is a function of `u`.** For
  a response reading `Phi_N`, the Hessian, `rho`, a ball mass or a row
  list, `K` is not a function of `u` at all and the structural argument
  decides -- the same one-sidedness the radial Jacobian always had.
* **GATE 1 cannot see f(R) gravity on this probe geometry** at any
  range, and by extension cannot see any modification whose amplitude is
  below ~0.04 dex after a two-parameter stretch. Measured in §3.2.
* **`not_decidable_on_this_bench` is a real gap, not a bin of
  convenience.** 100
  settings are rejected because a float64 CG solve cannot reach its
  tolerance at their fitted amplitude. A better-conditioned solve would
  have to re-decide them.
* **The external control suite is 12 rows.** It covers the theory classes
  the reviewer named and two contrast rows; it is not a proof of general
  correctness.

---

## 8. Reproduce

```
    python test_compiler.py     # 48 tests, 48 passed, 0 failed, 75 s
    python retrospective.py     # 3123 candidates, 48 s, caches cold
    python write_report_v2.py   # regenerates this file from the JSON
```

`REPORT.md` is not touched. Every number above is read from
`compiler_results.json` or `retrospective.json` by `write_report_v2.py`;
none is typed in.

New tests added by this work, all passing:

* `test_curl_identity_holds_and_predicts_the_run_AR_value`
* `test_curl_identity_holds_on_every_row_run_AR_measured`
* `test_curl_vanishes_in_spherical_symmetry`
* `test_gate4_scope_is_declared_and_names_what_it_excludes`
* `test_gate4_legacy_key_is_still_readable`
* `test_u_space_gradient_floor_is_measured_on_laws_that_are_gradients`
* `test_vector_potential_force_is_LABELLED_not_rejected`
* `test_external_positive_controls_all_agree`
* `test_gate1_identifiability_threshold_is_measured_not_tuned`
* `test_taxonomy_partitions_every_rejection_into_exactly_one_bin`
* `test_taxonomy_severity_order_is_declared_and_total`
* `test_external_axis_element_lands_where_the_derivation_says`
* `test_external_axis_reduction_is_reported_against_the_exact_projector`

