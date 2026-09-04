# Control harness — `work/wellnet-2026-09/controls/`

**Lane:** controls. **Date:** 2026-09-03. **Status:** all nine controls built;
all eleven worked examples pass.

    controls.py               the nine controls, 2,892 lines, numpy + scipy, CPU only
    test_controls.py          eleven worked examples, 804 lines; script-first, pytest-collectable
    controls_validation.json  every number in this report, machine-readable
    full_run.log              the console transcript of the run that produced it
    REPORT.md                 this file

Run it with

    python test_controls.py            # full, a few minutes
    python test_controls.py --fast     # smaller simulation counts, ~1 minute

This lane was built **before** the discoveries it polices, on purpose. Run J
established that the search machinery manufactures gain on its own: a
physics-free twin with the radial structure permuted away inside each galaxy
recovered +3.1% of the training improvement at K = 8 and +2.1% at k = 3, while
the real winners came out about 4% **worse** than the RAR on blind galaxies. A
number produced without a control of that kind is not a measurement, and this
module is the smallest set of controls that makes the difference checkable
rather than asserted.

Two design rules run through all nine:

1. **Every claimed invariant is verified numerically, in code, on every
   realisation.** A control returns a `ControlRealisation` carrying
   `invariants` (things it says it preserves, each with the measured residual)
   and `destroyed` (things it says it removes, with both numbers).
   `.check(tol)` raises if any invariant moved. Nothing is asserted in prose
   that is not also asserted in code.
2. **Where a mistake can be made structurally impossible, it is** — control 9 —
   and where it cannot, the limit is stated plainly rather than dressed up.

---

## Summary

| # | Control | What it caught in the worked example |
|---|---------|--------------------------------------|
| 1 | `residual_null` | Separates a real +6.06% tensor gain from the −0.14% its own null produces; the residual's correlation with angle falls +0.378 → +0.004 while every shell mean is bit-identical |
| 2 | `position_scramble` | The mock cluster's well network sits **+3.5σ** above the scrambled ensemble with the radial mass profile changed by **exactly 0** |
| 3 | `mass_scramble` | Removes most of the mass–radius segregation (−0.409 → −0.124) at bit-identical geometry, and moves W by a factor 2.3 |
| 4 | `smoothed_source` | Reproduces the closed-form ensemble mean of control 2 to **0.24%**; exposed a **cubic-lattice ℓ = 4 floor** that would have read as residual source anisotropy |
| 5 | `synthetic_universe` + `run_discovery` | 5/5 injected families recovered; **tensor false-positive rate in scalar data: 28.7% naive → 5.2% ± 0.9% calibrated** |
| 6 | `assert_parameter_responsive` | 2 of 5 headline statistics are **bit-identical across three decades of κ**, spread exactly 0.000000 |
| 7 | `check_exchangeability` | Catches an extra smoothing pass (error), a different convolution kernel of the same shape (warning, fatal under `strict`), and randomness inside the pipeline |
| 8 | `shared_denominator_report` | Reproduces the retracted ρ_p = −0.3042 from the raw LoCuSS tables, its +0.957 error correlation, its −0.113 null expectation and p = 0.525 against its own null |
| 9 | `SplitData` / `FrozenModel` | −0.73% frozen versus **+4.07%** re-solved on blind: a 4.80-point swing that turns a refutation into a discovery |

---

## 1. Residual nulls

`residual_null(source, seed, block=...)` and `residual_null_batch(...)`.
Accepts an `ObjectPointSource` (rotation-curve shaped) or a `FieldSource` (a
resolved three-dimensional map), so one control covers both.

**What it does.** Residuals are permuted inside blocks. Three block structures
matter:

* `block="object"` — Run J's `perm_g`, byte for byte with
  `standardise=False`: every object's mean offset survives (distance,
  inclination and M/L errors are real and are not new physics) and only the
  within-object dependence is destroyed.
* `block="object+shell"` — the **field generalisation**: the radial profile of
  the residual is held exactly and only the angular structure is destroyed.
  This is the correct null for a well-network claim, which is a claim about
  geometry at fixed radial profile.
* an explicit block array, when the (system, radius) structure is known.

`standardise=True` (default) permutes σ-standardised residuals and rescales, so
a heteroscedastic field does not acquire a fake noise map, then subtracts the
block mean again so the object offset is preserved *exactly* rather than
approximately.

**Verified invariants** (3,000 points, 600 blocks of 5):

| invariant | residual |
|---|---|
| mean residual in every block | 4.2 × 10⁻¹⁷ |
| σ map, at the same positions | 0 (bit-identical) |
| model, coordinates, point count | 0 |
| within-block residual multiset (`standardise=False`) | 0 |

**What it caught.** On a universe generated with a genuine anisotropic law:

* the residual's within-block correlation with the *radius* stays at 4 × 10⁻¹⁸
  (it is constant inside a block, by construction) while its correlation with
  the *angle* falls from **+0.3780 to +0.0038** — the control removes the
  angular signal and nothing else;
* adding the tensor atoms improves the blind fit by **+6.06%** on the real data
  and by **−0.14%** on its own residual null. Without the null, a +6% blind
  improvement reads as "a tensor effect"; the null is what turns it into a
  measurement.

On the resolved field, the angular structure of the residual falls from 0.0680
to 0.0091 (**86.6% removed**) with every shell mean preserved to 3 × 10⁻¹⁶.

**Performance.** `residual_null_batch` is 160× faster than a loop of single
realisations (6.46 s → 0.04 s for 99 draws on 3,000 points), because the block
permutation is a `lexsort` and the block means are `bincount`s. This is not
cosmetics: a control loop that costs more than the search it polices is a
control loop that gets shortened.

---

## 2. Position-scrambled clusters

`position_scramble(cluster, seed, footprint=None)`.

Every member keeps its **mass** and its **clustercentric radius** exactly; only
its angular position is randomised. The radial mass profile — which is what
every spherically-averaged cluster measurement actually constrains — is
therefore held bit-identical, so any statistic that only sees M(<r) *cannot*
move. Whatever does move is geometry. That is the point of the control for the
well-network hypothesis.

**Verified invariants** (200-member mock cluster with injected mass
segregation and a filamentary axis):

| invariant | residual |
|---|---|
| member radii, relative | 3.3 × 10⁻¹⁶ |
| member masses | 0 |
| radial mass profile, 12 log shells | **exactly 0** |
| cluster membership | 0 |

**What it caught.** The network energy W = Σ_{i<j} m_i m_j / |r_i − r_j| — the
simplest statistic that depends on geometry at fixed radial profile — is
5.2575 × 10²⁶ observed against 4.7629 ± 0.141 × 10²⁶ over 300 scrambles, i.e.
**+3.5σ**, p = 0.0133. The radial mass profile changed by 0.000 M⊙ over the
same 300 scrambles. A well-network claim that survives this control is a claim
about geometry; one that does not is a claim about the radial profile wearing a
disguise.

**Guards built in.** 3-D positions flagged `projected` are **refused**: an
isotropic scramble of a projected separation changes the observed R_proj, so
the control would be destroying the observable rather than the geometry. A
`footprint` callable rejection-samples directions inside a survey mask, and
members for which no accepted direction was found keep their original position
— that count is reported, not hidden, because those members are a residue of
the real signal left inside the control.

---

## 3. Mass-scrambled clusters

`mass_scramble(cluster, seed)`. Geometry preserved exactly; masses permuted
among positions, within each cluster.

Everything attached to the **mass** travels with it — the mass error, and any
per-member array listed in `extra["mass_attached"]` — while everything attached
to the **position** stays put. Getting that wrong is how a control quietly
stops being a control.

**Verified invariants:** positions bit-identical (fingerprint match,
max |Δ| = 0); member radii 0; per-cluster mass multiset 0; per-cluster total
mass 1.4 × 10⁻¹⁶ relative.

**What it caught.** The mass–radius correlation (mass segregation, injected at
d log m / d log r = −0.55) goes from **−0.409 to −0.124**, and W drops from
5.258 × 10²⁶ to 2.325 × 10²⁶ — a much larger move than control 2 produces,
because breaking segregation also breaks the massive-members-at-small-radius
pairing that dominates W. Controls 2 and 3 remove different things: control 2
keeps mass segregation and destroys geometry, control 3 keeps geometry and
destroys segregation. Running only one of them leaves the other explanation
standing.

---

## 4. Smoothed source

`smoothed_source(source)` — no randomness. Dispatches on type:

* `FieldSource` → a `FieldSource` whose ρ is the shell mean. Because the grid
  is regular, the shell mean is the mass-conserving average, so the mass inside
  every shell — hence M(<r) at every shell edge — is preserved **exactly**.
* `ClusterSource` → a `SmoothedCluster` with M(<r) and ρ(r), the members' mass
  redistributed uniformly over their own shells.

**Verified invariants** (field / cluster): total mass 2.7 × 10⁻¹⁶ / 1.4 × 10⁻¹⁶
relative; M(<r) at every shell edge 6.3 × 10⁻¹³ / 4.1 × 10⁻¹⁶ relative;
**M(<r) monotone** residual exactly 0 in both — checked because non-monotonic
M(r) is on the programme's standing failure list, and "cannot happen" is what
every such bug says; monopole unchanged.

**Control 4 is the ensemble mean of control 2, and that is proved rather than
asserted.** For independent isotropic directions at fixed radii,
⟨1/|r_i − r_j|⟩ = 1/max(r_i, r_j) exactly — only the ℓ = 0 term of the
multipole expansion survives the angular average — so

    E[W | position scramble]  =  sum_{i<j} m_i m_j / r_>

with no simulation at all. `meanfield_network_energy` returns that number:
4.7745 × 10²⁶ in closed form against 4.7629 × 10²⁶ from 300 position scrambles,
a **relative difference of 2.4 × 10⁻³**. Controls 2 and 4 are the same null,
one sampled and one in closed form, and the identity is the cross-check that
both are implemented correctly. For 2-D projected positions the angular average
is an elliptic integral rather than 1/r_>, and the function returns NaN rather
than a wrong number.

**What it caught — a test bug that looks like a source signal.** The ℓ = 4
multipole of the smoothed field does *not* go to zero: it goes to 0.0887,
against the source's 0.0973. The smoothed field is purely radial by
construction, so 0.0887 **is the cubic lattice's own ℓ = 4 anisotropy**. Read
naively, 91% of the source's apparent hexadecapole is the grid. ℓ = 2 has no
cubic term and does vanish exactly (0.0740 → 0). The fix is a lattice-free
statistic — `FieldSource.shell_anisotropy`, the RMS fractional departure of ρ
from its own shell mean, which is exactly zero for any purely radial field on
any grid. On the worked example it goes 0.1759 → 4.9 × 10⁻¹⁴. **Use
`shell_anisotropy` for the resolved-versus-averaged test; use `multipole` only
with the lattice floor subtracted, which the realisation now reports as
`lattice_l4_floor`.**

---

## 5. Synthetic known-law universes

`synthetic_universe(law, seed)` generates mock observations under five laws;
`run_discovery(u)` runs the full pipeline on each; `known_law_suite()` and
`tensor_false_positive_rate()` are the batch drivers.

**The five generators.** Systems are Hernquist baryon spheroids with
*independent* mass and scale, so "the shape of the source at fixed enclosed
mass" is a genuine second direction and a nonlocal law is identifiable; each
system carries an orientation axis and every point is sampled at a known angle
to it, so an anisotropic law is identifiable. Noise is heteroscedastic per
point (0.08–0.14 dex) plus a per-system offset (0.06 dex), which is what
distance / inclination / M-to-L errors look like and is exactly what control 1
must preserve. Default sampling: 60 systems × 10 radii × 5 angles = 3,000
points, split 40/20 by whole system.

| law | form | free global constants |
|---|---|---|
| `newton` | g = g_N | 0 |
| `mond` | g = ν(g_N/a₀) g_N | 1 (a₀) |
| `gr_dm` | g = G[M_b(<r) + M_NFW(<r)]/r², M₂₀₀ = A (M_b/10¹⁰)^b, c = 10 | 2 |
| `tensor` | g = ν(x) g_N [1 + ε_T P₂(cos θ)], axis-aligned quadrupole | 2 |
| `nonlocal` | g = ν(x_eff) g_eff with M_eff(<r) = ∫ M_b(<s) K(s−r; L) ds | 2 |

No parameter is ever fitted per object; the constants are global, per the
standing brief.

### 5a. Recovery of the injected family — 5 / 5 correct

Every family gets its global constants fitted on the training systems, plus an
intercept and one amplitude coefficient, is then **frozen**, and is scored once
on the blind systems through the control-9 guard.

| injected | parsimonious pick (blind dex) | bare argmin | margin over the runner-up |
|---|---|---|---|
| newton | **newton** 0.12180 | `nonlocal` 0.12145 — **wrong** | all five tied inside 1 SE (0.0027 dex) |
| mond | **mond** 0.12825 | `mond` | `nonlocal` tied at 0.12825, `tensor` at 0.12831 |
| gr_dm | **gr_dm** 0.12665 | `gr_dm` | 0.0092 dex over `mond` |
| tensor | **tensor** 0.12265 | `tensor` | 0.0053 dex over `mond` |
| nonlocal | **nonlocal** 0.11986 | `nonlocal` | 0.0315 dex over `mond` |

**A finding the harness produced about itself.** A bare argmin over blind RMS
gets `newton` **wrong**, and for a structural reason: the tensor family nests
MOND (ε_T → 0) and the nonlocal family nests MOND (L → 0), so a richer family
can only tie or beat the one that generated the data, by a margin that is pure
noise — 0.00035 dex in the newton case, against a one-standard-error band of
0.0027 dex. On a Newtonian universe all five families are statistically tied
and the argmin picks whichever noise favoured. Selection therefore uses a
**one-standard-error parsimony rule**: among the families tied with the best
(within RMS/√(2 n_blind)), take the one with the fewest free global constants.
With that rule 5/5 are correct; with the bare argmin, 4/5. The argmin is
reported alongside as `recovered_argmin`, so the disagreement is visible rather
than smoothed over.

This matters beyond the mock. Any future comparison of a candidate law against
the RAR is a comparison of nested families, and "the richer model had the lower
out-of-sample error" is not by itself evidence for the richer model.

### 5b. It must not invent tensor or nonlocal effects

The detection rule is: adding the tensor atoms improves the **blind** fit,
calibrated against a null that destroys exactly the structure being claimed and
nothing else. For tensor that null is control 1 with (system, radius) blocks —
the radial profile of the residual is held bit-identical and only the
assignment of residual to *angle* is permuted, which is an exact permutation
argument under H₀. For nonlocal, the system-level shape parameter is permuted
across systems.

Three decision rules of increasing discipline are scored on the same
simulations, because the gap between them is the number that matters.

**600 scalar universes** (200 each under `newton`, `mond` and `gr_dm`), 199
nulls per universe:

| decision rule for "a tensor effect is present" | false-positive rate |
|---|---|
| **naive** — the tensor atoms improved the blind fit at all | **28.7%** |
| **threshold** — they improved it by more than 1% | 0.0% |
| **calibrated** — p ≤ 0.05 against the matched permutation null | **5.2% ± 0.9%** |

Per generating law: `newton` 6.0%, `mond` 5.0%, `gr_dm` 4.5% (n = 200 each).
The median tensor "gain" on scalar data is **−0.06%** and the 95th percentile is
**+0.08%**, which is why the 1%-threshold rule never fires and is useless at
this sampling: it is two orders of magnitude above the noise it is meant to
exclude, and it would also miss any real effect below 1%.

**The calibrated test is correctly sized, not merely small.** The 600
calibrated p-values are uniform on [0, 1]: Kolmogorov–Smirnov D = 0.043,
p = 0.205. A test that never fires would also show a low false-positive rate;
uniformity is what distinguishes a calibrated test from a dead one.

**So: the credibility number for any future tensor claim in this programme is
5.2%, and only if the claim is made with the calibrated rule.** Made with the
naive rule — "the tensor atoms improved the out-of-sample fit" — it is 28.7%,
i.e. better than one in four physics-free universes produces one.

**A negative result on the nonlocal side, stated plainly.** The same procedure
applied to the nonlocal detector, on 120 of the same scalar universes with 99
nulls each, gives **45.0% naive and 11.7% calibrated**. That calibrated rate is
2.3σ above the nominal 5% and is **not acceptable**: the covariate-permutation
null used for the nonlocal test (permute the system-level shape parameter
across systems) is evidently not exact, most likely because the shape parameter
is not exchangeable across systems at fixed enclosed mass in these mocks. Until
that null is fixed or replaced, a nonlocal claim must clear roughly **p ≤ 0.02**
to buy a true 5% rate, and this report does not certify the nonlocal detector
at nominal size. The tensor detector, which is the one the brief asked for, is
certified.

**One universe per law, both requirements checked on each:**

| injected | tensor detected? | p_T | nonlocal detected? | p_N |
|---|---|---|---|---|
| newton | no | 0.625 | no | 0.290 |
| mond | no | 0.335 | no | 0.310 |
| gr_dm | no | 0.070 | no | 0.370 |
| tensor | **YES** | 0.005 | no | 0.580 |
| nonlocal | no | 0.750 | **YES** | 0.010 |

All five requirements pass: every injected family is recovered; no tensor
effect is reported in scalar data; no nonlocal effect is reported in scalar
data; the tensor effect is found in tensor data; the nonlocal effect is found
in nonlocal data.

**How to read the false-positive rate.** It is a property of *this test
procedure at this sampling* (60 systems × 10 radii × 5 angles, 0.08–0.14 dex
point noise, 0.06 dex per-system offsets, B nulls per universe), not a
universal constant. Any real tensor claim must re-run
`tensor_false_positive_rate` with the real sampling and quote **that** number.
What transfers is the shape of the result: the uncalibrated rule fires on
roughly a third of physics-free universes, and the calibrated rule fires at its
nominal size.

---

## 6. Parameter-sensitivity tests

`assert_parameter_responsive(stat, thetas, name=...)` — a hard raise, not a
diagnostic. `responsiveness_suite(dict_of_stats, thetas)` runs a whole family
and raises **once** at the end naming every failure, so a report cannot quietly
contain one blind statistic among ten good ones.

**Worked example: the real trap, on the real data.** On the 40 LoCuSS
clusters the model quantity in the deep-MOND limit is Y_pred = κ · t · f_gas.
κ multiplies it, so every rank statistic of Y_pred is *exactly* invariant in κ.
Over three decades, κ ∈ [10³, 10⁶], seven settings:

| statistic | distinct values / 7 | spread | verdict |
|---|---|---|---|
| `spearman(Y_pred, kT)` | **1** | **0.000000** | BLIND |
| `partial_spearman(Y_pred, kT \| M_WL)` | **1** | **0.000000** | BLIND |
| `median Y_pred` | 7 | 34.51 | pass |
| `mean ln(E_obs/E_pred)` | 7 | 2.347 | pass |
| `chi²-like mean square ln residual` | 7 | 3.541 | pass |

Two of five headline statistics are bit-identical across three decades of the
coupling they are supposed to measure — reproducing exactly what Run K
recorded. `responsiveness_suite` raises, naming both. A clean null from either
of those two carries no information at all: it is a theorem about ranks, not a
fact about gravity.

---

## 7. Exchangeability tests

`check_exchangeability(pipeline, arm_true, arm_control)` runs the pipeline once
on each arm with the numpy and scipy entry points **patched**, and diffs the
two operation traces. This is instrumentation, not a convention: a pipeline
that quietly smooths one arm and not the other shows up whether or not it says
so.

Traced classes, matching the brief's list: interpolation (`np.interp`,
`ndimage.map_coordinates` / `zoom` / `shift` / `rotate`); smoothing
(`np.convolve`, `np.correlate`, `ndimage.gaussian_filter*`, `uniform_filter`,
`median_filter`, `ndimage.convolve` / `correlate`); masking (`np.where`,
`clip`, `compress`, `nan_to_num`, `isfinite`, `isnan`, `extract`, `putmask`);
sampling and binning (`histogram*`, `digitize`, `searchsorted`, `percentile`,
`quantile`, `bincount`, `average`); aperture (declared with `trace_note` or
`@traced`); and randomness (`np.random.*`).

Severity:

* **ERROR** — a different op, a different order, a different op count, a
  different array shape or dtype, a different exact-valued argument, or **any**
  randomness inside the pipeline. A random pipeline makes the arms
  incomparable by construction: the control must be realised *before* the
  pipeline, not inside it.
* **WARN** — a different count of non-finite values, a data-derived float
  argument that differs, or a **small array (≤ 64 elements) that differs in
  value**. `strict=True` promotes warnings to errors.

**What it caught** on a five-stage pipeline (aperture → mask → quantile
binning → convolution → interpolation), 10 traced ops per arm:

| injected bug | result |
|---|---|
| none | 0 errors, 1 warning — the small binned *profile* differs, which is the data and is correct |
| the true arm smoothed twice | **2 errors**, raises: `op-sequence numpy.convolve vs numpy.interp`, `op-count 11 vs 10` |
| the control arm convolved with a different 3-element kernel | 0 errors, **5 warnings** including a value-level flag on `arg1` of `numpy.convolve`; raises under `strict=True` |
| `np.random.normal` called inside the pipeline | raises immediately |

The small-array value fingerprint is what makes the third case visible at all:
a kernel of the same shape and dtype is structurally identical, and without a
value hash the check would pass. It is a warning rather than an error because a
small array can equally be a data profile — the trace names the op and the
argument index so the analyst can tell which, and `strict=True` is there when
no adaptive step is intended.

**What it cannot see** (stated plainly, not papered over): `from numpy import
interp` binds the original at import time and is invisible to the patch — call
through the module; ndarray *methods* (`a.mean()`, `a.clip()`) and compiled
inner loops are not traceable this way; and aperture or selection steps written
by hand must be declared with `trace_note` or wrapped with `@traced`, which is
the one cooperative part of the design. A pipeline written entirely in ndarray
methods would trace as empty and pass vacuously, so the report always includes
the op count: a trace of zero or near-zero ops means "not instrumented", not
"clean".

---

## 8. Shared-denominator detector

`shared_denominator_report(inputs, exprs, estimator, ...)`, plus `eiv_fit`,
`structural_null` and `validate_eiv`.

Given the measured inputs with their uncertainties and the **construction
expressions** for each series, it: parses the expressions' ASTs to find inputs
common to more than one series (`M_WL` and `M_WL_err` are different names, and
`log(M)` does not contain `M` by accident); propagates the published errors
through the *actual* expressions by Monte Carlo to measure the induced error
correlation; and computes the **null expectation of the naive estimator**
instead of assuming zero.

**Worked example: the LoCuSS retraction, rebuilt from the raw tables.** The
ingest asserts 41 rows in each table and name-set equality between them, drops
Abell2697 by the stated criterion (missing L_K) and asserts the retained count
is 40 — the silent-extraction check the standing brief requires. The forward
chain is Run K's exactly: r₅₀₀ from the lensing mass, M_star = 0.73 L_K,
g_N,b = G M_b / r₅₀₀², E_obs = g_WL / ν(x_b) g_N,b. **M_WL enters twice** —
once as the numerator gravity and once through r₅₀₀, which is derived from it.

| quantity | measured here | Run K record |
|---|---|---|
| E_obs median (range) | 1.617 (1.22–2.34) | 1.62 (1.22–2.34) |
| induced error correlation ln E_obs vs ln M_WL | **+0.957** | +0.96 |
| naive partial Spearman ρ_p(E, kT \| M_WL) | **−0.3042** | −0.304 (retracted) |
| naive OLS partial slope | **−0.1550** | −0.155 |
| its null expectation | **−0.1132** | −0.12 |
| p of the observed against its own null | **0.525** | 0.563 |
| EIV slope d ln E / d ln kT | **−0.1634** | −0.166 |
| EIV mass-slope attenuation | **0.687** | 0.66 |

Every documented number is reproduced from the catalogue by generic machinery
that was given the construction expressions, not the answer.

**The finding that matters, and it is new.** *Which null you choose decides the
answer, and the obvious null is the wrong one.* Three nulls, same statistic,
same data:

| null | mean | p of the observed |
|---|---|---|
| permute the carrier (kT) across clusters | −0.004 | **0.023** — "significant" |
| decorrelate the carrier's latent from everything | +0.002 | 0.034 — "significant" |
| **structural**: fit the EIV model, set β(kT) = 0, keep everything else | **−0.113** | **0.525** — nothing |

Permuting kT destroys not only the E–kT link under test but also the **real**
mass–temperature relation, and that relation is the mechanism that biases the
naive partial estimator: partialling out a noisy ln M_WL whose error correlates
at +0.96 with ln E_obs's error over-corrects, and the over-correction is
proportional to r(M, T). Remove r(M, T) from the null and the artefact goes
with it — leaving the artefact in the data looking like a detection at
p = 0.023.

The structural null keeps the latent M–T correlation, the real dependence of E
on M, and the real off-diagonal error covariance, and removes only the link
under test. It is a parametric bootstrap from the fitted errors-in-variables
model. **A permutation null is not automatically the conservative choice.**
That is now a standing caution for this programme, on the same footing as the
shared-denominator rule itself.

**The errors-in-variables estimator**, `eiv_fit(Y, C)`, takes a *per-object*
error covariance C_i measured by pushing the published errors through the
construction expressions, and does not assume it diagonal — when one measured
input enters two series, the off-diagonal is the whole point. Exact Gaussian
marginal likelihood, MLE by L-BFGS-B with a Nelder-Mead polish, bootstrap
warm-started from the full-data solution (a cold start at n = 40 wanders off
and produces intervals ten units wide, which is optimiser noise, not
uncertainty).

**Validated as unbiased by simulation across the whole range**, at the actual
pathology (error correlation +0.96, n = 40, 300 simulations per point):

| true β | naive mean (bias) | EIV mean (bias) | MC SE |
|---|---|---|---|
| −0.600 | −0.283 (**+0.317**) | −0.610 (−0.010) | 0.009 |
| −0.300 | −0.045 (**+0.255**) | −0.304 (−0.004) | 0.008 |
| +0.000 | +0.192 (**+0.192**) | −0.021 (−0.021) | 0.006 |
| +0.300 | +0.435 (**+0.135**) | +0.290 (−0.010) | 0.006 |
| +0.600 | +0.673 (**+0.073**) | +0.591 (−0.009) | 0.005 |

The naive estimator's bias reaches 0.32 and, crucially, **is not zero at
β = 0**: it is +0.19 there, which is the whole retraction in one number. The
EIV estimator's worst bias is 0.021, a factor 15 smaller. That residual is a
genuine finite-sample bias of an MLE at n = 40 — with 300 simulations it is
3.2 MC standard errors, i.e. statistically detectable — so "unbiased" here
means **unbiased to better than 0.05 in absolute terms**, not exactly zero, and
the criterion in code is a statement about size rather than a p-value. An
estimator validated only at β = 0 is not validated; this one was checked across
the range.

---

## 9. Frozen-coefficient enforcement

`SplitData` and `FrozenModel`. The design goal is that the wrong thing
**cannot be expressed**:

* the held-out rows of the design matrix and of the target are captured in a
  **closure** at construction and are never bound to the object, so there is no
  `splits.blind_y` to hand to a solver;
* the only thing that crosses into the held-out set is a `FrozenModel`, and
  `evaluate` accepts nothing else;
* a `FrozenModel` can only be produced by `fit`, which slices the *train* rows
  and has no code path reaching any other row;
* `evaluate` returns scalars only — never residuals, predictions or targets —
  so an outer loop cannot reconstruct the held-out target by differencing;
* coefficients are returned as a read-only array and sealed with an HMAC over
  the coefficient bytes and the train and design fingerprints, re-checked on
  every use;
* a model whose train fingerprint does not match cannot be evaluated, so a
  model fitted elsewhere cannot be smuggled in;
* the held-out set is touch-counted, and `max_touches > 1` requires a written
  reason that is recorded in the audit log.

**What it caught.** A deliberately small search (20,000 random k = 6 subsets of
a 17-atom bank, on 512 training points from 24 systems; MOND-generated data
with no tensor and no nonlocal content), scored against an RAR-plus-global-
offset baseline:

| | train | blind |
|---|---|---|
| baseline (RAR + offset) | 0.12155 | 0.11981 dex |
| candidate, coefficients **frozen** | 0.12070 (**+0.70%**) | 0.12069 (**−0.73%**) |
| candidate, coefficients **re-solved on blind** | — | 0.11493 (**+4.07%**) |

The bug is worth **+4.80 percentage points and it flips the sign**: a law that
is 0.73% *worse* than the RAR out of sample is reported as 4.07% better. Run J
saw the same thing at +2.17% re-solved against −3.73% frozen, a 5.90-point
swing. The anti-pattern is implemented once, as `_unguarded_refit_on_holdout`,
clearly named, called by no control, and present only so the size of what the
guard prevents can be measured.

Every route to the wrong answer was attempted and blocked:

| attempt | result |
|---|---|
| `splits.blind`, `splits.y` | `SealedHoldoutError` |
| second touch of the blind split | `SealedHoldoutError` |
| `evaluate` a raw coefficient array | `SealedHoldoutError` |
| `evaluate` a model fitted on other data | `SealedHoldoutError` |
| `max_touches=5` with no written reason | `ValueError` |
| mutate `model.coef` in place | read-only array |
| overwrite `_coef` via `object.__setattr__` | `FrozenSealError` (HMAC) |

**What this cannot do.** Python has no private state:
`splits.evaluate.__closure__[i].cell_contents` still reaches the held-out
arrays for anyone determined to get at them. The guard stops the *mistake*, not
a deliberate forgery. That distinction is stated here rather than dressed up.

---

## Failure modes from the standing brief — each checked explicitly

* **Shared-denominator artefacts.** Control 8. Reproduced from raw data and
  extended: the choice of null decides the answer, and the natural permutation
  null is the wrong one here (p = 0.023 against p = 0.525).
* **Monotone-invariant statistics.** Control 6. Two of five statistics on the
  real LoCuSS sample are bit-identical across three decades of κ, spread
  exactly 0.000000. The check raises.
* **Refitting on the held-out set.** Control 9. Measured at +4.80 percentage
  points and a sign flip on the worked example, and made structurally
  inexpressible through the API.
* **Silent extraction failures.** `load_locuss` asserts 41 rows in each table,
  asserts name-set equality between them, names the single excluded cluster
  (Abell2697) and the criterion that excluded it, and asserts the retained
  count is 40. Every control also asserts its own point, voxel or member
  counts, and every realisation carries a content fingerprint of the arrays it
  claims not to have touched.
* **Test bugs that look like solver bugs.** Found one: the cubic-lattice ℓ = 4
  floor in control 4, which would read as residual source anisotropy after
  angular averaging. Replaced with a lattice-free statistic, with the floor
  reported alongside.
* **Non-monotonic M(r).** Checked as an invariant in control 4 on both the
  field and the cluster path; residual exactly 0 by construction, and verified
  anyway.
* **Sealed holdouts.** KiDS and the wide binaries were not loaded, not opened,
  and are not referenced anywhere in this lane. The only real data used are the
  LoCuSS tables already in `work/gravity-cluster-audit-2026-09/acquire/`
  (Mulroy 2019 observables and Subaru weak-lensing masses), read read-only. The
  SPARC-derived code in `work/gravitylab` was read but not re-run, and no new
  files were written outside this lane directory.
* **Global gravity parameters only.** Every law in control 5 gets universal
  constants; no parameter is fitted per object anywhere in the module.
* **No data that presupposes dark matter.** The `gr_dm` universe is a *mock
  generator* with a declared NFW halo, used to test whether the pipeline can
  tell a halo law from a MOND law. No published mass map, NFW-fitted mass or
  GR-derived convergence map is treated as an observation anywhere.

---

## What I could not build, and why

Stated plainly, as the brief requires.

1. **No control was demonstrated on a real resolved 3-D mass map or a real
   cluster member catalogue, because neither is in the repository yet.** The
   sibling lane directories (`cluster-data/`, `env-data/`, `potential-depth/`,
   `void-data/`) were empty scaffolding when this lane started; by the time it
   finished `cluster-data/bcg/` held BCG and ICL photometry, which is not a
   member catalogue with a position and a mass per member and is not a
   resolved map. Controls 1 (field path), 2, 3 and 4 are therefore demonstrated
   on mock sources with *injected* structure of known amplitude — the right way
   to demonstrate a control, but not the same as running it on the real
   catalogue. The interfaces (`FieldSource`, `ClusterSource`) are the ones the
   real data will need: positions, masses, uncertainties, a cluster index, and
   nothing that assumes the mock. **First job for whoever brings the real
   catalogue: run `position_scramble(...).check()` on it, because that call
   will fail loudly if the radial profile is not held exactly, and a silent
   pass is the only evidence that the control is doing what it says.**

2. **The MOND/AQUAL universe is the algebraic relation g = ν(g_N/a₀) g_N,
   which is exact for AQUAL only in spherical symmetry.** The mock sources are
   spherical, so it is exact *there*, but a genuine aspherical AQUAL test needs
   the finite-volume solver in `work/gravitylab/solver.py`. Wiring that solver
   into the mock generator is the obvious next step and was out of scope here.

3. **The tensor and nonlocal generators are the leading terms of their
   families, not full field solutions.** The tensor law is an axis-aligned
   quadrupolar modulation of |g| — the leading anisotropic term of a K-tensor
   law — and the nonlocal law is a Gaussian-kernel-smoothed enclosed mass at a
   fixed physical length, not a full nonlocal Green function. They are
   sufficient for the question asked ("can the search invent a direction or a
   length scale that is not there?"), because the *atoms* that would detect
   them are the atoms a real search would use, but a claim about a specific
   tensor theory needs that theory's own forward solve.

4. **The exchangeability tracer is not complete coverage** — see the end of
   section 7.

5. **The frozen-coefficient guard is mistake-proof, not tamper-proof** — see
   the end of section 9.

6. **Control 2 on projected cluster data is weaker than on 3-D data.** A
   sky-plane angular scramble cannot destroy line-of-sight structure, and the
   closed-form mean-field identity does not hold in 2-D (the angular average of
   1/|r_i − r_j| is an elliptic integral). The function returns NaN there
   rather than a wrong number, but the consequence is that a projected
   well-network claim has a weaker null available than a 3-D one, and that
   should be said in any such claim.

7. **No control for selection-function-induced structure.** The programme's
   earlier "label control" — running the identical search on data containing
   only survey structure and no physics — is a different animal from the nine
   here and needs the real survey selection function to build. It is not in
   this module. Neither is a control for train/blind covariate shift beyond the
   split-by-object rule.

8. **The false-positive rates are procedure-specific, not universal.** They are
   measured at one mock sampling and one noise model. Re-measure with the real
   sampling before quoting them against a real claim.

---

## How a future lane uses this

```python
import controls as C

# 1. any claim about geometry at fixed radial profile
ctl = C.position_scramble(cluster, seed)
ctl.check()                      # raises if the radial profile moved at all
null = [C.position_scramble(cluster, s).data for s in range(200)]

# 2. any claim that adds atoms and improves an out-of-sample fit
Yn, rec = C.residual_null_batch(source, seed, 199, block=blocks)
rec.check()

# 3. every held-out evaluation, without exception
sd  = C.SplitData(design, y, split, atoms=names)
mdl = sd.fit()                   # train only; no other code path exists
sd.evaluate(mdl)                 # one touch, scalars out

# 4. every headline number, before it is written down
C.assert_parameter_responsive(stat, thetas, name="...")   # raises

# 5. every correlation between constructed quantities
C.shared_denominator_report(inputs, exprs, estimator,
                            null_carrier="...", series_order=[...],
                            carrier_series="...")

# 6. before any permutation null is trusted
C.check_exchangeability(pipeline, real_source, control_source)
```

The controls are cheap — the whole validation suite, including 600 simulated
universes for the false-positive rate, runs in 288 s on one CPU core — so there
is no performance argument for skipping them.
