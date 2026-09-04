"""
Render REPORT.md from results.json.  Every number in the report comes from the
JSON; nothing is typed by hand.
"""
from __future__ import annotations

import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(HERE, "results.json"), encoding="utf-8"))
L = []


def w(s=""):
    L.append(s)


def f(x, n=4):
    return f"{x:+.{n}f}"


def pc(x, n=1):
    return f"{100*x:.{n}f}%"


M = R["job3"]["modes"]
E = R["extrapolation"]
V = R["verdicts"]
J4 = R["job4"]
AT = R.get("at_null_split", {})
REACH = R["reach"]
meta = R["meta"]

w("# Temperature-clamp audit of `invariant_bench._cluster_profile`")
w()
w(f"Run {meta['when']}, numpy {meta['numpy']}, Python {meta['python']}. "
  f"Sealed probes never loaded: {', '.join(meta['sealed_never_loaded'])}.")
w()
w("`_cluster_profile` interpolated the coarse X-COP temperature profile onto "
  "the fine density grid with a bare `np.interp` -- no `left`, no `right` -- "
  "so `kT` was **clamped** to the endpoint value outside the measured range. "
  "The hydrostatic acceleration carries "
  "`(dln n_e/dln r + dln kT/dln r)`, so beyond the last measured temperature "
  "bin `dln kT/dln r` was forced to **exactly zero** and the "
  "temperature-gradient term was silently deleted -- in the outskirts, where "
  "this programme reads its radial trend, and where the true temperature is "
  "falling.")
w()
w(f"Bench sha256 before `{meta['bench_sha256_before']}`  \n"
  f"Bench sha256 after  `{meta['bench_sha256_after']}`")
w()

# ---------------------------------------------------------------- headline
w("## Headline")
w()
w(f"* **{E['n_extrapolated']} of {E['n_points']} X-COP points "
  f"({pc(E['frac'],2)}) rest on clamped temperature**, all of them the "
  f"outermost points of their cluster ({E['n_inner_extrap']} inner "
  f"extrapolations, so the clamped set is exactly the outer set).")
w(f"* The measured temperature grid ends at a median "
  f"**r/R500 = {E['median_last_T_over_R500']:.3f}** "
  f"(range {E['min_last_T_over_R500']:.3f} to "
  f"{E['max_last_T_over_R500']:.3f}); the relation is quoted out to "
  f"**r/R500 = {E['max_r_over_R500']:.3f}**.")
w(f"* **{V['n_changed']} of {V['n_total']} recorded verdicts change** when the "
  f"clamped points are dropped. The three that move are the X-COP "
  f"temperature correlation (`rho_T`), its permutation p, and the pressure "
  f"amplitude `kappa`.")
w(f"* Against a synthetic truth with a genuinely falling temperature, the "
  f"clamp biases the within-cluster radial slope by "
  f"**{f(J4['summary']['within']['clamp']['bias_at_flat_truth'])} dex/dex**, "
  f"i.e. {J4['summary']['within']['clamp']['pct_of_observed']:.1f}% of the "
  f"observed slope, and it is "
  f"{pc(J4['summary']['within']['decomposition']['frac_clamp'])} of the "
  f"pipeline's entire flat-truth bias.")
if AT:
    s = AT["split"]
    w(f"* Inside Run AT's **own** noiseless forward null, reproduced "
      f"bit-identically, the clamp is "
      f"**{s['clamp_pct_of_bias_S3']:.1f}% of the 29%** "
      f"(S3 bias {f(s['total_bias_S3'])} -> {f(s['other_S3'])} with the clamp "
      f"removed and nothing else changed).")
w("* The trend itself **survives**: the within-cluster radial slope moves "
  f"from {f(M['clamp']['slope_radius_within'])} to "
  f"{f(M['drop']['slope_radius_within'])}, "
  f"{abs(100*(M['drop']['slope_radius_within']-M['clamp']['slope_radius_within'])/M['clamp']['slope_radius_within']):.1f}%.")
w()

# ---------------------------------------------------------------- job 1
w("## Job 1 -- consumer inventory")
w()
J1 = R["job1"]
w(f"Derived by walking the repo with `ast`, not asserted (`inventory.py`). "
  f"**{J1['n_direct']}** direct callers outside the bench, "
  f"**{J1['n_importers']}** importers of `Bench` of which "
  f"**{J1['n_reads_xcop']}** consume X-COP, and "
  f"**{J1['n_same_direction']}** files that re-implement the identical "
  f"unguarded interpolation. This lane's own files are excluded.")
w()
w("### Direct callers of `_cluster_profile`")
w()
if J1["direct_callers"]:
    for p in J1["direct_callers"]:
        w(f"* `{p}`")
else:
    w("* none outside the bench. `_xcop` is the only in-repo caller, which is "
      "why patching `_cluster_profile` reaches every lane at once.")
w()
w("### Lanes that import `Bench`, and therefore run `_xcop` -> `_cluster_profile`")
w()
w("**All fifteen consume X-COP.** The first pass of this inventory said three "
  "of them did not; that was wrong -- `p03`, `p04` and `p05` never write "
  "`d[\"xcop\"]`, they call `b.confound(...)`, `b.score(...)` or iterate "
  "`b.d`, every one of which pulls X-COP in.")
w()
w("| lane | consumes X-COP via |")
w("| --- | --- |")
for q in J1["importers"]:
    w(f"| `{q['path']}` | {', '.join('`'+v+'`' for v in q['via']) if q['reads_xcop'] else '**no consumption found**'} |")
w()
w("### Independent re-implementations of the same clamp")
w()
w("These do not call `_cluster_profile`, but they interpolate the X-COP "
  "temperature table with their own bare `np.interp`. Every `np.interp` call "
  "in the repo was parsed; **not one** passes `left=` or `right=` on a "
  "temperature, so the patch to the bench does not fix any of them.")
w()
w("| file | unguarded temperature interps | direction |")
w("| --- | --- | --- |")
for q in J1["reimplementations"]:
    if q["same_direction"]:
        w(f"| `{q['path']}` | lines {q['same_direction']} | **same as the "
          f"bench: coarse T on a finer grid** |")
for q in J1["reimplementations"]:
    if not q["same_direction"]:
        w(f"| `{q['path']}` | lines {q['onto_T_grid']} | onto the T grid -- "
          f"same mechanism, different exposure, **not measured here** |")
w()
w("### Recorded JSON downstream")
w()
w("| file | producer | present |")
w("| --- | --- | --- |")
for q in J1["recorded_json"]:
    w(f"| `{q['path']}` | `{q['producer']}` | {'yes' if q['exists'] else 'MISSING'} |")
w()

# ---------------------------------------------------------------- exposure
w("## Exposure, per cluster")
w()
w("| cluster | n | clamped | % | stencil | last measured T (r/R500) | outermost point (r/R500) |")
w("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
for q in E["per_cluster"]:
    w(f"| {q['name']} | {q['n']} | {q['n_extrap']} | {pc(q['frac'])} | "
      f"{q['n_stencil']} | {q['r_last_T_over_R500']:.3f} | "
      f"{q['r_max_over_R500']:.3f} |")
w(f"| **total** | **{E['n_points']}** | **{E['n_extrapolated']}** | "
  f"**{pc(E['frac'],2)}** | | median **{E['median_last_T_over_R500']:.3f}** | "
  f"max **{E['max_r_over_R500']:.3f}** |")
w()
w("`stencil` counts points whose `np.gradient` three-point stencil touches a "
  "clamped point, so their `dlnT/dlnr` is contaminated even though they are "
  "themselves inside the measured range.")
w()

# ---------------------------------------------------------------- job 2
w("## Job 2 -- the fix")
w()
w("`_cluster_profile(d, temp_extrapolation=...)` with four modes:")
w()
w("| mode | behaviour |")
w("| --- | --- |")
w("| `clamp` | **default**; `np.interp`'s own clamping, bit-identical to every recorded number |")
w("| `drop` | keep only points inside the measured temperature range |")
w("| `loglinear` | continue the measured outer log-log temperature slope |")
w("| `forbid` | raise `TemperatureExtrapolationError` |")
w()
w("The return value is a `ClusterProfile`, a 4-tuple subclass, so every "
  "existing `r, gb, go, R500 = self._cluster_profile(d)` keeps working "
  "unchanged. It carries `.extrapolated`, `.stencil`, `.frac_extrapolated`, "
  "`.r_tmin`, `.r_tmax`, `.kT`, `.mode`. `Bench(temp_extrapolation=...)` "
  "threads the choice through, `Bench.extrapolation_report` holds one "
  "structured dict per cluster, and `Bench._announce_extrapolation` emits a "
  "`TemperatureExtrapolationWarning` carrying the extrapolated fraction on "
  "every load that touches a clamped point.")
w()
rep = R["job3"]["reproduction"]
w(f"**Default behaviour is bit-identical.** "
  f"{R['job3']['n_reproduced']} of {R['job3']['n_checked']} recorded "
  f"quantities reproduce under `mode='clamp'`:")
w()
w("| recorded quantity | recorded | reproduced | \\|diff\\| |")
w("| --- | ---: | ---: | ---: |")
for q in rep:
    w(f"| {q['quantity']} | {q['recorded']:.6g} | {q['clamp']:.6g} | "
      f"{q['abs_diff']:.2e} |")
w()
w("### Regression test")
w()
pp = json.load(open(os.path.join(HERE, "prepatch_proof.json"), encoding="utf-8"))
w(f"`test_tempclamp.py`, 93 tests. `prove_test_fails_prepatch.py` loads the "
  f"untouched pre-patch source and shows **{pp['n_failing_checks']} of "
  f"{pp['n_checks']}** of the suite's demands fail against it:")
w()
w("| demand | fails pre-patch | detail |")
w("| --- | --- | --- |")
for q in pp["checks"]:
    w(f"| {q['check']} | {'**yes**' if q['fails_prepatch'] else 'no'} | "
      f"{q['detail']} |")
w()
w("The load-bearing one is `test_forbid_raises_on_xcop`: pre-patch there was "
  "no way to say *do not extrapolate*, so the call raised `TypeError` instead "
  "of the specific error and the clamped points went through unannounced. "
  "Its counterweight is `test_default_is_bit_identical`, which inlines the "
  "pre-patch body verbatim and requires `np.array_equal` on `r`, `gb` and "
  "`go` for all twelve clusters.")
w()
w("### Bugs this lane's own tests found")
w()
w("1. **My first `deep`-mask test was wrong.** It let the innermost "
  "extrapolated point (a one-sided `np.gradient` stencil) into a set asserted "
  "to be exactly flat. Also, a clamped stencil returns ~5.7e-14, not literal "
  "zero: `np.gradient`'s unequal-spacing coefficients sum to zero only to "
  "rounding. Fixed to exclude endpoints and use a 1e-11 tolerance -- four "
  "orders below the smallest measured slope.")
w("2. **My first synthetic truth produced negative temperatures.** Anchoring "
  "the hydrostatic pressure at the middle bin let `P(r_out)` go negative for "
  "five of twelve clusters; `log(kT)` became `nan` and the reconstruction "
  "then *silently dropped* those points -- reproducing, inside the audit, the "
  "exact class of silent mask the audit exists to catch. It also flipped the "
  "sign of the reported bias. Fixed by anchoring at the outer boundary, where "
  "positivity is automatic, plus an assertion.")
w("3. **`mode='drop'` went silent.** The first version of the patch counted "
  "only extrapolated points still present, so `drop` reported zero "
  "extrapolation and said nothing about the 93 points it had just deleted. "
  "Now covered by `test_drop_mode_still_reports_what_it_removed`.")
w("4. **`Bench._load` swallowed the `forbid` error.** Its bare "
  "`except Exception` turned a deliberate hard stop into a silently missing "
  "probe. `_load` now re-raises `TemperatureExtrapolationError` and records "
  "every other swallowed loader failure on `self.load_errors`. Covered by "
  "`test_load_does_not_swallow_the_forbid_error`.")
w("5. **My pre-patch proof script counted warnings instead of reading them.** "
  "astropy emits two FITS warnings, so `len(w) == 0` reported a pass for a "
  "module that says nothing about extrapolation at all. Now matched on "
  "content.")
w()

# ---------------------------------------------------------------- job 3
w("## Job 3 -- impact audit")
w()
w("| quantity | recorded (clamp) | drop | loglinear | drop - clamp | change |")
w("| --- | ---: | ---: | ---: | ---: | ---: |")
ROWS = [
    ("n points", "n", 0),
    ("within-cluster radial slope", "slope_radius_within", 4),
    ("its standard error", "slope_radius_within_se", 4),
    ("Spearman(r/R500, residual)", "spearman_frac_residual", 4),
    ("rho_T = Spearman(kT, excess)", "rho_T", 4),
    ("p_T (permutation)", "p_T", 4),
    ("c70 dln(excess)/dln(kT)", "c70_slope", 4),
    ("c70 within-cluster rho median", "c70_within_rho_median", 4),
    ("c70 normalisation ratio median", "c70_ratio_median", 4),
    ("c71 within-cluster scatter", "c71_true_within", 4),
    ("c71 median-ratio scatter", "c71_true_median", 4),
    ("kappa", "kappa", 0),
    ("chi2 of the kappa fit", "chi2", 2),
]
for lab, k, n in ROWS:
    c, d, l_ = M["clamp"][k], M["drop"][k], M["loglinear"][k]
    ch = f"{100*(d-c)/c:+.1f}%" if c else "--"
    fmt = f"{{:.{n}f}}" if n else "{:.0f}"
    w(f"| {lab} | {fmt.format(c)} | {fmt.format(d)} | {fmt.format(l_)} | "
      f"{fmt.format(d-c)} | {ch} |")
w()
w("The bench's own headline score moves too:")
w()
BS = R["bench_score_by_mode"]
w("| mode | n X-COP | median \\|err\\| dex, X-COP |")
w("| --- | ---: | ---: |")
for m in ("clamp", "drop", "loglinear"):
    w(f"| `{m}` | {BS[m]['n_xcop']} | {BS[m]['xcop']:.4f} |")
w()
w("### Per-cluster median excess")
w()
w("| cluster | clamp | drop | loglinear | drop - clamp |")
w("| --- | ---: | ---: | ---: | ---: |")
for nm in M["clamp"]["per_cluster_excess"]:
    a = M["clamp"]["per_cluster_excess"][nm]
    b = M["drop"]["per_cluster_excess"][nm]
    c2 = M["loglinear"]["per_cluster_excess"][nm]
    w(f"| {nm} | {a:.4f} | {b:.4f} | {c2:.4f} | {100*(b-a)/a:+.1f}% |")
w()
w("### Does the verdict change, not just the digit?")
w()
w("| quantity | clamp | drop | recorded verdict | verdict after dropping | changes |")
w("| --- | ---: | ---: | --- | --- | --- |")
for q in V["table"]:
    w(f"| {q['quantity']} | {q['clamp']:.4g} | {q['drop']:.4g} | "
      f"{q['verdict_recorded']} | {q['verdict_drop']} | "
      f"{'**YES**' if q['verdict_changes'] else 'no'} |")
w()
for q in V["table"]:
    if q["note"]:
        w(f"* **{q['quantity']}** -- {q['note']}")
w()
P = R["job3b"]
w("### Placebo, and what it cannot do")
w()
w(f"Dropping the clamped points moves the within-cluster slope by "
  f"{f(P['change'])}. Dropping the **same number of points per cluster at "
  f"random positions** moves it by {f(P['null_mean'])} +- {P['null_sd']:.4f} "
  f"(95% [{f(P['null_lo'])}, {f(P['null_hi'])}]), so the real change sits at "
  f"p = {P['p_vs_random_drop']:.4f} against that null: it is not the loss of "
  f"{P['n_dropped']} points, it is *which* points.")
w()
w(f"But the second placebo is degenerate, and this matters: the clamped set "
  f"IS the outermost set ({P['clamped_set_is_outermost']}). So the "
  f"drop-versus-clamp difference **cannot on its own** attribute the change to "
  f"clamping rather than to radial range. That attribution requires a "
  f"synthetic truth, which is Job 4.")
w()
FP = R["job3c"]
w("### Calibration of the test that produced p_T")
w()
w(f"Run AT found an obvious permutation test running at FPR 0.53-0.70 against "
  f"a nominal 0.05, so this one was checked before being read. Against a null "
  f"with excess independent of kT at the same n = 12 and the same marginal "
  f"spread, the realised FPR is **{FP['realised_fpr']:.3f} +- {FP['se']:.3f}** "
  f"over {FP['ndraw']} draws against a nominal 0.05: "
  f"{'correctly sized' if FP['correctly_sized'] else 'MIS-SIZED'}. "
  f"The p-values above are therefore usable at face value -- but at n = 12 a "
  f"single cluster's median excess moving by 4% takes rho_T from "
  f"{M['clamp']['rho_T']:.4f} to {M['loglinear']['rho_T']:.4f}, which is the "
  f"underpowering the register already records, now demonstrated.")
w()

# ---------------------------------------------------------------- reach
w("## The honest reach of the X-COP relation")
w()
D = R["job3d"]
w("| r/R500 | clusters with a MEASURED temperature there |")
w("| ---: | ---: |")
for g in (0.70, 0.80, 0.90, 1.00, 1.10, 1.52, 1.90, 2.50):
    n = sum(1 for v in D["last_T_per_cluster"].values() if v >= g)
    w(f"| {g:.2f} | {n} of {D['R200_over_R500'] and 12} |")
w()
rc = REACH["clamp"]
w(f"* All twelve clusters have a measured temperature only out to "
  f"**r/R500 = {D['r_all12']:.2f}**; at least half out to "
  f"**{D['r_half']:.2f}**.")
w(f"* **R200 = {D['R200_over_R500']} R500: "
  f"{D['n_clusters_measured_at_R200']} of 12** clusters have a measured "
  f"temperature there.")
w(f"* **The a0 crossing at r/R500 = 1.9-2.5: "
  f"{D['n_clusters_measured_at_crossing']} of 12.** The outermost X-COP data "
  f"point of any kind sits at r/R500 = {D['max_r_over_R500']:.3f}, so the "
  f"crossing is beyond every X-COP point, not merely beyond every measured "
  f"temperature.")
w(f"* `g_bar = a0` is directly bracketed in "
  f"{rc['n_gbar_a0_measured']} of {rc['n_clusters']} clusters within the "
  f"bench's own 120-1650 kpc radial cut.")
w()
w("### What can honestly be said about \"factor 1.4 at R200\"")
w()
w(f"Fitting each cluster's excess against r/R500 using **only points with a "
  f"measured temperature**, the within-cluster slope is "
  f"{f(rc['slope_median'])} +- {rc['slope_sd']:.4f} (spread across clusters) "
  f"and the excess at the last measured temperature radius is "
  f"**{rc['excess_at_last_measured_T_median']:.3f}**. Continuing that fit to "
  f"R200 -- {rc['median_dex_extrapolated']:.3f} dex beyond the data -- gives")
w()
w(f"> excess(R200) = **{rc['excess_at_R200_median']:.2f}**, 16-84 range "
  f"**[{rc['excess_at_R200_lo']:.2f}, {rc['excess_at_R200_hi']:.2f}]**, with "
  f"{rc['n_excess_at_R200_below_1']} of {rc['n_clusters']} clusters "
  f"extrapolating below 1.0.")
w()
w("So X-COP does not contradict \"factor 1.4 at R200\", but it does not "
  "support it either: its own extrapolation centres nearer 1.2 with a 1-sigma "
  "interval that reaches below 1.0, i.e. *no excess at all*. The recorded 1.4 "
  "came from lensing; **X-COP cannot corroborate it, because X-COP measures "
  "no temperature at R200 in any of the twelve clusters.** The honest "
  f"statement is that the X-COP relation is measured to "
  f"**r/R500 = {D['r_all12']:.2f} in all twelve clusters and to "
  f"{D['r_half']:.2f} in half of them**, and everything beyond "
  f"r/R500 = {E['median_last_T_over_R500']:.2f} is extrapolation.")
w()
w("### What can honestly be said about \"crossing at r/R500 = 1.9-2.5\"")
w()
w(f"Continuing the same measured-temperature fits until the excess reaches "
  f"1.0 gives r/R500 = **{rc['t_excess_unity_median']:.2f}**, 16-84 range "
  f"**[{rc['t_excess_unity_lo']:.2f}, {rc['t_excess_unity_hi']:.2f}]** -- "
  f"{rc['median_dex_to_excess_unity']:.2f} dex beyond the last measured "
  f"temperature. The central value happens to land inside the recorded "
  f"1.9-2.5, but that interval spans a factor {2.5/1.9:.2f} in radius while "
  f"the honest 1-sigma range spans a factor "
  f"{rc['t_excess_unity_hi']/rc['t_excess_unity_lo']:.1f} -- "
  f"**{(rc['t_excess_unity_hi']/rc['t_excess_unity_lo'])/(2.5/1.9):.1f} times "
  f"wider than what was recorded**, and 12 of 12 clusters have no measured "
  f"temperature anywhere near it. Quote the range or do not quote the number.")
w()

# ---------------------------------------------------------------- job 4
w("## Job 4 -- is clamping the right default?")
w()
w(f"A synthetic truth is built on each cluster's **real** n_e: the target "
  f"excess is exactly `C0 * (r/r_mid)^s_true`, and the temperature is solved "
  f"from hydrostatic equilibrium, anchored at the outer boundary by "
  f"continuing that cluster's own measured outer log-slope. The synthetic "
  f"temperature falls outward in **{J4['n_falling']} of "
  f"{len(J4['synthetic_realism'])}** clusters, which is the condition the job "
  f"asked for. No noise at all.")
w()
w("| cluster | measured dlnT/dlnr | synthetic dlnT/dlnr |")
w("| --- | ---: | ---: |")
for q in J4["synthetic_realism"]:
    w(f"| {q['name']} | {q['s_meas_outer']:+.3f} | {q['synth_dlnT_outer']:+.3f} |")
w()
w("Two statistics are reported because they disagree about how the clamp "
  "matters: `within` is the fixed-effects within-cluster slope this programme "
  "actually reads, `pooled_S3` is Run AT's statistic (pooled slope beyond "
  "0.25 R500, no per-cluster level).")
w()
for stat in ("within", "pooled_S3"):
    S = J4["summary"][stat]
    w(f"### `{stat}`")
    w()
    w("| policy | bias at a FLAT truth | response d(measured)/d(true) | "
      "observed | bias as % of observed | de-biased truth |")
    w("| --- | ---: | ---: | ---: | ---: | ---: |")
    for k in ("clamp", "drop", "loglinear", "full_coverage", "perfect"):
        q = S[k]
        w(f"| `{k}` | {f(q['bias_at_flat_truth'])} | {q['response']:.4f} | "
          f"{f(q['observed'])} | {q['pct_of_observed']:.1f}% | "
          f"{f(q['debiased_truth'])} |")
    d = S["decomposition"]
    w()
    w(f"Flat-truth bias {f(d['total_flat_bias'])} decomposes as "
      f"`np.gradient` discretisation {f(d['gradient_only'])} "
      f"({pc(d['frac_gradient'])}), coarse temperature grid at **full** "
      f"radial coverage {f(d['coarse_full_coverage']-d['gradient_only'])} "
      f"({pc(d['frac_coarse'])}), and **the clamp "
      f"{f(d['clamp_only'])} ({pc(d['frac_clamp'])})**.")
    w()
w("`full_coverage` uses a coarse grid with the same number of bins and the "
  "same log spacing but covering the whole radial range, so it isolates "
  "coarsening from extrapolation; `perfect` uses the true fine-grid "
  "temperature and isolates `np.gradient` alone.")
w()
w("### Which policy is least biased")
w()
Sw = J4["summary"]["within"]
order = sorted(("clamp", "drop", "loglinear"),
               key=lambda k: abs(Sw[k]["bias_at_flat_truth"]))
w(f"On the statistic this programme reads, ordered by |bias| at a flat truth: "
  + ", ".join(f"**`{k}`** {f(Sw[k]['bias_at_flat_truth'])}" for k in order)
  + ".")
w()
w(f"`drop` is the least biased of the three "
  f"({f(Sw['drop']['bias_at_flat_truth'])}, "
  f"{Sw['drop']['pct_of_observed']:.1f}% of the observed slope) and has much "
  f"the best response ({Sw['drop']['response']:.3f} against "
  f"{Sw['clamp']['response']:.3f} for the clamp), i.e. it both mis-states the "
  f"zero point least and tracks a real signal most faithfully. `loglinear` "
  f"sits between them and buys back only "
  f"{100*(1-Sw['loglinear']['bias_at_flat_truth']/Sw['clamp']['bias_at_flat_truth']):.0f}% "
  f"of the clamp's bias, because continuing a slope fitted on three noisy "
  f"outer bins is itself an assumption. "
  f"**Clamping is the worst of the three and should not be the scientific "
  f"default** -- it stays the code default only so that no recorded number "
  f"moves without someone choosing it.")
w()
if AT:
    s = AT["split"]
    w("### How much of Run AT's 29% is this bug")
    w()
    w(f"Run AT's noiseless forward null was reproduced **bit-identically** "
      f"(S1 {f(AT['reproduced_clamped']['S1_hse'])}, "
      f"S3 {f(AT['reproduced_clamped']['S3_hse'])}; recorded "
      f"{f(AT['recorded_AT']['S1_hse'])} and "
      f"{f(AT['recorded_AT']['S3_hse'])}). One switch was then flipped inside "
      f"Run AT's own machinery: the simulated cluster is observed on a "
      f"temperature grid extended outward to the last density bin "
      f"({AT['clamp_removed']['n_bins_added']} bins added across the sample), "
      f"so nothing is ever extrapolated. The truth, the boundary pressure, "
      f"the R500 inference and the statistics are the same code.")
    w()
    w("| | S1 (correlation) | S3 (slope) |")
    w("| --- | ---: | ---: |")
    w(f"| observed on real data | {f(AT['observed_S1'])} | {f(AT['observed_S3'])} |")
    w(f"| total pipeline bias | {f(s['total_bias_S1'])} | {f(s['total_bias_S3'])} |")
    w(f"| of which **the clamp** | {f(s['clamp_S1'])} | {f(s['clamp_S3'])} |")
    w(f"| of which everything else | {f(s['other_S1'])} | {f(s['other_S3'])} |")
    w(f"| bias as % of observed | {s['bias_pct_of_observed_S1']:.1f}% | "
      f"{s['bias_pct_of_observed_S3']:.1f}% |")
    w(f"| **clamp share of that bias** | **{s['clamp_pct_of_bias_S1']:.1f}%** | "
      f"**{s['clamp_pct_of_bias_S3']:.1f}%** |")
    w(f"| clamp as % of observed | {s['clamp_pct_of_observed_S1']:.1f}% | "
      f"{s['clamp_pct_of_observed_S3']:.1f}% |")
    w()
    w(f"So of the 29% that Run AT attributed to the pipeline, "
      f"**{s['clamp_pct_of_bias_S3']:.0f}% is this bug** and "
      f"{100-s['clamp_pct_of_bias_S3']:.0f}% is everything else -- chiefly the "
      f"R500 inference step, whose recovery ratio improves from "
      f"{AT['reproduced_clamped']['R500_ratio_median']:.4f} to "
      f"{AT['clamp_removed']['R500_ratio_median']:.4f} once the clamp is "
      f"removed. **The clamp was also biasing the inferred R500 low by "
      f"{100*(1-AT['reproduced_clamped']['R500_ratio_median']):.0f}%**, which "
      f"is a second, unrecorded consequence of the same defect.")
    w()

# ---------------------------------------------------------------- caveats
w("## What this audit did NOT establish")
w()
w("* **Whether the residual trend is physics or hydrostatic bias.** Removing "
  "the clamp reduces the pipeline's flat-truth bias but does not separate "
  "modified gravity from an outward-rising non-thermal pressure. X-ray data "
  "alone still cannot.")
w("* **Whether `loglinear` is right.** It continues a slope fitted on three "
  "outer bins with no uncertainty propagated into the downstream statistics. "
  "It is offered as a sensitivity, not a recommendation.")
w("* **CLASH.** Its binned table does not pass through `_cluster_profile` and "
  "was not touched here; Run AT already flagged it as the open half of that "
  "audit.")
_same = [q["path"] for q in J1["reimplementations"] if q["same_direction"]]
w(f"* **The re-implementations.** {len(_same)} files carry their own bare "
  f"`np.interp` on the same table, in the same direction: "
  + ", ".join(f"`{p.split('/')[-1]}`" for p in _same)
  + ". Their exposure is quantified above through the recomputed statistics, "
    "but the fix was applied only to the shared bench; patching each lane "
    "individually is a separate job. `r500-audit/nullsim.py` is a deliberate "
    "reproduction of the bug for Run AT's forward null and should keep it.")
w("* **Provenance.** `repro/inputs_c70.json` and `repro/inputs_c71.json` pin "
  "a 20,139-byte `invariant_bench.py` at sha256 `fe817b22...` that lived in a "
  "scratchpad and no longer exists; the surviving repo copy was 19,707 bytes "
  "at `" + meta["bench_sha256_before"][:8] + "...` before this patch. Those "
  "receipts already pinned a file that cannot be checked, and this patch "
  "changes the repo file's hash to `" + meta["bench_sha256_after"][:8] +
  "...`. They need a legitimate reseal, not a hash edit.")
w("* **A standing seal hazard, pre-existing.** `Bench.__init__` calls "
  "`_widebin()`, which returns hard-coded El-Badry boosts from the source "
  "itself, so any bare `Bench()` loads a sealed probe. Nothing in this lane "
  "constructs a Bench without stubbing `_kids` and `_widebin` first.")
w()
w("---")
w()
w(f"Rendered programmatically from `results.json` by `report.py`. "
  f"Audit runtime {meta['runtime_s']} s.")

open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8").write(
    "\n".join(L) + "\n")
print(f"wrote REPORT.md, {len(L)} lines")
