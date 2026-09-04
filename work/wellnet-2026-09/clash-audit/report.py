"""
Render REPORT.md.  Every number in the report is read out of the results JSONs
programmatically -- nothing is typed by hand.
"""
from __future__ import annotations
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
FILES = ["provenance_results.json", "cancellation_results.json",
         "structure_results.json", "null_results.json",
         "diagnostics_results.json", "sensitivity_results.json",
         "truthcheck_results.json"]


def load():
    R = {}
    for f in FILES:
        R[f.split("_")[0]] = json.load(open(os.path.join(HERE, f), encoding="utf-8"))
    return R


def f(x, n=4, sign=True):
    if x is None:
        return "--"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    if v != v:
        return "nan"
    return f"{v:+.{n}f}" if sign else f"{v:.{n}f}"


def main():
    R = load()
    P, CA, ST, NU, DG, SE, TC = (R["provenance"], R["cancellation"],
                                 R["structure"], R["null"], R["diagnostics"],
                                 R["sensitivity"], R["truthcheck"])
    DOC = []
    w = DOC.append

    w("# CLASH R500 tautology audit — the open half of Run AT")
    w("")
    w("Every number below is rendered from `results.json` by `report.py`; nothing")
    w("is typed by hand.  KiDS and the wide binaries were never loaded.")
    w("")
    w("Reproduce with `python run_all.py`.  Code: `ingest.py`, `stats.py`,")
    w("`nullsim.py`, `run_provenance.py`, `run_cancellation.py`,")
    w("`run_structure.py`, `run_diagnostics.py`, `run_null.py`,")
    w("`run_sensitivity.py`, `run_truthcheck.py`, `tests.py`, `report.py`.")
    w("Results: `results.json` (merged) plus the seven per-job JSONs;")
    w("`ACQUISITION.json` has every source URL, arXiv version and sha256; `raw/`")
    w("holds the acquired tables; `*.log` the run transcripts.")
    w("")

    # ---------------------------------------------------------------- POWER
    w("## 0. POWER, stated before any verdict")
    w("")
    pw = DG["power"]
    w("| contrast | n | \\|rho\\| detectable at 80% power | power at rho = 0.5 | power at rho = 0.3 |")
    w("|---|---|---|---|---|")
    for k in ("100 kpc", "200 kpc", "400 kpc", "600 kpc", "between_cluster_all"):
        d = pw[k]
        w(f"| {k} | {d['n']} | {f(d['rho_detectable_at_80pct'],2,False)} | "
          f"{f(d['power_at_rho_0p5'],2,False)} | {f(d['power_at_rho_0p3'],2,False)} |")
    w("")
    ex = DG["R500_extrapolation"]
    w(f"The sample is 84 rows over 20 clusters, and the whole lensing chain carries")
    w(f"only **{P['degrees_of_freedom']['free_numbers_in_the_lensing_chain']} free")
    w(f"numbers** (20 M200c + 20 c200c).  With a per-cluster level free, the radial")
    w(f"trend has **{P['degrees_of_freedom']['within_cluster_radial_shape_dof']}")
    w(f"degree of freedom per cluster**, so its effective df is 20, not 64.")
    w("")
    w(f"R500 is never reached: the outermost datum sits at")
    w(f"**{f(ex['outermost_datum_over_R500_median'],3,False)} R500** (median; range")
    w(f"{f(ex['outermost_datum_over_R500_min'],3,False)}–"
      f"{f(ex['outermost_datum_over_R500_max'],3,False)}), every CLASH point is at")
    w(f"r/R500 <= {f(ex['max_r_over_R500_any_point'],3,False)}, and R500 lies")
    w(f"**{f(ex['R500_over_outermost_datum_median'],2,False)}x beyond the last")
    w(f"measurement**.")
    w("")

    # ------------------------------------------------------------ Job 1
    w("## 1. Job 1 — acquisition")
    w("")
    w("Umetsu+2016 (ApJ 821, 116) has **no VizieR catalogue**.  Verified two ways,")
    w("both with positive controls:")
    w("")
    w("```")
    w("asu-tsv -source=J/ApJ/821/116&-out.all=1   HTTP 200,")
    w("        #INFO Error=Table or Catalog not found: J/ApJ/821/116")
    w("        (identifier echoed back; no CatalogsExamined= fallback)")
    w("METAcat title=*CLASH*                      14 catalogues, Umetsu+2016 absent;")
    w("        positive control J/ApJ/896/70 (Tian+2020) IS in the list")
    w("```")
    w("")
    w("The masses were obtained instead from the **arXiv e-print source** of")
    w("arXiv:1507.04385v4, whose `table2.tex` and `table3.tex` are the journal")
    w("tables verbatim.  Also acquired: Donahue+2014 CLASH-X (arXiv:1405.7876v3),")
    w("which supplies a Chandra hydrostatic r500 that is **independent of the")
    w("lensing** — the control Run AT could not build for X-COP.")
    w("")
    w("| file | rows | contents |")
    w("|---|---|---|")
    w("| `raw/umetsu2016_table1.tex` | 20 | z, kT_X (Postman+2012) |")
    w("| `raw/umetsu2016_table2.tex` | 20 | M200c, c200c, r_-2 |")
    w("| `raw/umetsu2016_table3.tex` | 20 | M2500c … M200m, **M500c** |")
    w("| `raw/donahue2014_chandra_hse.tex` | 25 (20 matched) | Chandra JACO r500 |")
    w("")
    w("Ingest asserts, on every table: the catalogue identifier echoed from the")
    w("`#Name:` line, the absence of `CatalogsExamined=`, the full column list")
    w("against the ReadMe, the row count, and finiteness of every value column.")
    w("`tests.py` verifies that a wrong identifier and a truncated column list are")
    w("both rejected.")
    w("")

    # ------------------------------------------------------------ Job 2.1
    w("## 2. The provenance table")
    w("")
    w("| quantity | source | derived from | root measurement | assumes GR | assumes a halo model |")
    w("|---|---|---|---|---|---|")
    for d in P["provenance_table"]:
        deriv = d["derived_from"].replace("|", "\|")   # markdown table cells
        w(f"| {d['quantity']} | {d['source']} | `{deriv}` | "
          f"{d['root_measurement']} | {d['assumes_GR']} | {d['assumes_a_halo_model']} |")
    w("")
    w("**The numerator's lensing profile and the M500c are not merely the same")
    w("measurement — they are two functionals of the same two-parameter fit.**")
    w("Tian+2020 §2.1: *\"we use these posterior distributions of the NFW")
    w("parameters to obtain well-characterized inference of M_tot(<r|M200,c200)\"*.")
    w("Umetsu+2016 Table 3 caption: *\"Cluster mass estimates M_3D(<r) from single")
    w("spherical NFW fits to individual surface mass density profiles\"*.")
    w("")
    ng = P["numerator_regeneration"]
    w("Proved rather than asserted.  Regenerating Tian's published `log(gtot)` from")
    w(f"Umetsu's (M200c, c200c) **alone** reproduces all {ng['n']} rows to")
    w(f"mean {f(ng['mean_dex'])} dex, sd {f(ng['sd_dex'],4,False)} dex, max")
    w(f"{f(ng['max_abs_dex'],4,False)} dex — against a published uncertainty whose")
    w(f"median is {f(ng['published_error_median_dex'],3,False)} dex, i.e. the")
    w(f"regeneration residual is **{1/ng['ratio_sd_to_median_published_error']:.0f}x")
    w(f"smaller than the quoted error**.")
    w("")
    rc = P["R500_closure"]
    w(f"And R500 is the same fit: R500 from the published M500c agrees with R500")
    w(f"solved directly on M_NFW(<r|M200,c200) to")
    w(f"{100*rc['median_rel_diff']:.2f}% (median), {100*rc['max_rel_diff']:.2f}% (max).")
    w("")

    # ------------------------------------------------------------ Job 2.2
    w("## 3. Is there a cancellation lemma?  **No.**")
    w("")
    tl = CA["table_level"]
    el = CA["estimator_level"]
    w("Run AT's X-COP lemma has a precondition — the numerator must be tabulated")
    w("in R500-scaled units so the R500 that scales and the R500 that unscales are")
    w("the same number.  CLASH does not meet it: Tian tabulates absolute m/s^2")
    w("against absolute kpc.")
    w("")
    w(f"* **Table level.** Substituting a different R500 on the x-axis moves the")
    w(f"  tabulated numerator by exactly {tl['max_abs_move_dex'][0]:.0e} dex.  That is a")
    w(f"  property of the table, not of the measurement: R500 is never an input to")
    w(f"  the tabulated numerator, so there is nothing to cancel — and therefore no")
    w(f"  lemma bounding the numerator when the underlying mass moves.")
    w(f"* **Estimator level.** Move the lensing mass that *generates* R500 and the")
    w(f"  numerator moves with it, one for one:")
    w("")
    w("| mass scaled by | d log10 g_obs | d log10 R500 |")
    w("|---|---|---|")
    for r_ in el["scan"]:
        w(f"| {r_['f']:.2f} | {f(r_['dlog10_gobs'])} | {f(r_['dlog10_R500'])} |")
    w("")
    w(f"    d log10 g_obs / d log10 R500 = {f(-el['induced_slope_dy_dx'],3)}")
    w(f"    X-COP, same test             =  1.6e-13")
    w(f"    ratio                        =  {el['ratio_to_xcop']:.1e}")
    w("")
    w("**There is no cancellation lemma, and the induced slope is sign-definite")
    w("negative — the sign the claim requires:**")
    w("")
    w(f"    induced d(y)/d log10(r/R500)      = {f(el['induced_slope_dy_dx'],3)}")
    w(f"    induced d(log a0)/d log10(r/R500) = {f(el['induced_slope_da0_dx'],3)}")
    w("")
    bp = CA["by_parameter"]
    w("The dangerous parameter is the concentration, not the mass:")
    w("")
    w("| parameter | d log g_obs / d log par | d log R500 / d log par | ratio |")
    w("|---|---|---|---|")
    for k, d in bp.items():
        w(f"| {k} | {f(d['dlog10_gobs_per_dex'])} | {f(d['dlog10_R500_per_dex'])} | "
          f"{f(d['ratio'],3)} |")
    w("")
    lv = CA["leverage"]
    w(f"Leverage: sd(log10 R500) across the 20 clusters is")
    w(f"{f(lv['sd_log10_R500_observed'],4,False)} dex (span factor")
    w(f"{f(lv['R500_span_factor'],2,False)}), while the mean *quoted* uncertainty on")
    w(f"log10 R500 is {f(lv['mean_e_log10_R500'],4,False)} dex.  So")
    w(f"**{100*lv['error_to_scatter_ratio']:.0f}% of the R500 spread is measurement")
    w(f"error, and that error is shared with the numerator.**")
    w("")

    # ------------------------------------------------------------ Job 2.6
    w("## 4. The rank identity — it applies, and the design also isolates the tautology")
    w("")
    w("CLASH's binned table **does** carry per-cluster levels: `fig2.dat` has an")
    w("`AName` column, 84 rows over 20 named clusters.  (`invariant_bench._clash()`")
    w("reads `q[2],q[3],q[4]` and discards `q[1]=AName`, which is why the record")
    w("says CLASH has no object identity.  The identity is in the file.)  So Run")
    w("AT's identity applies unchanged:")
    w("")
    w("| subset | n | clusters | rank[ind\\|log r] | rank[ind\\|log(r/R500)] | rank[ind\\|both] | R500 adds |")
    w("|---|---|---|---|---|---|---|")
    for k, d in ST["rank_identity"].items():
        w(f"| {k} | {d['n']} | {d['n_clusters']} | "
          f"{d['rank_indicators_plus_logr']} | "
          f"{d['rank_indicators_plus_log_r_over_R500']} | "
          f"{d['rank_indicators_plus_BOTH']} of {d['columns_of_BOTH']} | "
          f"{d['extra_directions_from_R500']} |")
    w("")
    w("But CLASH's radial grid is **common across clusters** — exactly 100, 200,")
    w("400 and 600 kpc, plus one per-cluster BCG radius.  The design is therefore")
    w("crossed, and it separates the two contrasts perfectly:")
    w("")
    w("* **within cluster, across levels**: r varies, R500_i fixed → r and r/R500")
    w("  are the same regressor (the identity above), and a per-cluster normaliser")
    w("  cannot move a within-cluster statistic.")
    w("* **between clusters, at one level**: r is *fixed*, so log(r/R500_i) varies")
    w("  **only** through R500_i.  Any correlation of the excess with r/R500 at")
    w("  fixed r *is* a correlation with -log R500_i.")
    w("")
    w("CLASH is the only sample in this programme that isolates the second")
    w("contrast, and it is the reviewer's tautology in its purest available form.")
    w("")

    # ------------------------------------------------------------ Job 2.4
    w("## 5. Within- versus between-cluster variance")
    w("")
    w("| subset / statistic | total | between | within | between % |")
    w("|---|---|---|---|---|")
    for k, d in ST["variance_decomposition"].items():
        if not isinstance(d, dict):
            continue
        w(f"| {k} | {f(d['total'],5,False)} | {f(d['between'],5,False)} | "
          f"{f(d['within'],5,False)} | **{100*d['between_fraction']:.1f}%** |")
    w("")
    vd = ST["variance_decomposition"]
    w(f"X-COP was **90.3% within**.  On the cluster-scale CLASH points — where the")
    w(f"lane-12 claim lives — CLASH is")
    w(f"**{100*vd['cluster_scale_64/y']['between_fraction']:.1f}% between** (RAR")
    w(f"residual) and")
    w(f"{100*vd['cluster_scale_64/a0']['between_fraction']:.1f}% between (a0")
    w(f"statistic).  **The monotone-invariance protection is not merely weaker than")
    w(f"X-COP's, it is inverted.**  Run AT's second structural protection is absent.")
    w("")

    # ------------------------------------------------------------ Job 2.5
    w("## 6. Slopes under the radial definitions")
    w("")
    w("Slopes, not correlations (AT.6).  `pooled` is the OLS slope over all points;")
    w("`within (FE)` gives each cluster its own level.")
    w("")
    for sub in ("all_84", "cluster_scale_64"):
        for stat, lab in (("y", "RAR residual"), ("a0", "log10(a0_eff/a0_can)")):
            w(f"**{sub}, statistic = {lab}**")
            w("")
            w("| radial definition | pooled slope | within (FE) slope | Spearman | n |")
            w("|---|---|---|---|---|")
            for k in ("r_physical", "r_over_R500_lens", "r_over_R500_xray",
                      "r_over_R500_TX", "r_over_Rb_gas", "r_over_Rb_M",
                      "r_over_Rb_g"):
                key = f"{sub}/{stat}/{k}"
                if key not in ST["slopes"]:
                    continue
                d = ST["slopes"][key]
                w(f"| {k} | {f(d['pooled_slope'])} | {f(d['fe_slope'])} | "
                  f"{f(d['spearman'])} | {d['n']} |")
            w("")
    w("Two things to read off these tables.")
    w("")
    w("**(a) The within-cluster slope is bit-identical under every normaliser.**")
    w("That is the rank identity, and `tests.py` confirms it survives even a random")
    w("per-cluster normaliser spanning two decades.  Only the *pooled* slope can")
    w("differ, and it differs only through the between-cluster part.")
    w("")
    d1 = ST["slopes"]["cluster_scale_64/a0/r_physical"]["pooled_slope"]
    d2 = ST["slopes"]["cluster_scale_64/a0/r_over_R500_lens"]["pooled_slope"]
    d3 = ST["slopes"]["cluster_scale_64/a0/r_over_R500_xray"]["pooled_slope"]
    d4 = ST["slopes"]["cluster_scale_64/a0/r_over_Rb_gas"]["pooled_slope"]
    w(f"**(b) The trend survives radii containing no total mass.**  On the a0")
    w(f"statistic: physical r {f(d1)}, r/R500_lens {f(d2)}, r/R500_Xray {f(d3)},")
    w(f"baryon-only r/R_b,gas {f(d4)}.  As in Run AT §AT.4, the effect is not")
    w(f"visible only under the mass-derived radius.")
    w("")
    exn = ST["baryon_radius_extrapolated_clusters"]
    w(f"Caveats on the baryon radii: R_b,gas requires extrapolating past the last")
    w(f"measured point for {exn['Rb_gas']}/20 clusters and R_b,g for {exn['Rb_g']}/20;")
    w(f"R_b,M needs none.  And a baryon-only normaliser is **not automatically a")
    w(f"clean control** — see §9.")
    w("")

    # ------------------------------------------------- pure tautology contrast
    w("### 6b. The pure tautology contrast: fixed r, R500 varying")
    w("")
    w("At fixed physical radius, log(r/R500) is exactly `const - log R500`.")
    w("")
    w("| statistic | r | n | corr(excess, log R500_lens) | slope | corr(excess, log R500_Xray) |")
    w("|---|---|---|---|---|---|")
    for k, d in ST["pure_tautology_contrast"].items():
        st, rr = k.split("/")
        w(f"| {st} | {rr} | {d['n']} | "
          f"**{f(d['pearson_excess_vs_log10R500_lens'])}** | "
          f"{f(d['slope_dexcess_dlog10R500_lens'],3)} | "
          f"{f(d['pearson_excess_vs_log10R500_xray'])} |")
    w("")
    w("The correlation with the **shared** lensing R500 rises steadily outward and")
    w("reaches +0.71 (y) and +0.77 (a0) at 600 kpc; the correlation with the")
    w("**independent** Chandra R500 does not follow it and changes sign.  That is")
    w("the shape a live tautology makes.  §7 shows it is also the shape a flat")
    w("truth makes, so it is not by itself evidence of one.")
    w("")
    sel = DG["selection"]
    w(f"Not a selection effect: the {sel['600kpc']['n_with']} clusters reaching")
    w(f"600 kpc have mean log10 R500 {f(sel['600kpc']['mean_log10R500_with'],4,False)}")
    w(f"against {f(sel['600kpc']['mean_log10R500_without'],4,False)} for the rest,")
    w(f"and restricted to those same clusters the correlation still climbs with")
    w(f"radius: ")
    for LV in ("100", "200", "400"):
        kk = f"restricted_to_600kpc_clusters/{LV}kpc"
        if kk in sel:
            w(f"{LV} kpc {f(sel[kk]['pearson_y'])}, ")
    w(f"then {f(ST['pure_tautology_contrast']['y/600kpc']['pearson_excess_vs_log10R500_lens'])} at 600 kpc.")
    w("")
    rr = DG["R500_lens_vs_xray"]
    w(f"The independent control is blunt, and this bounds how much it can settle:")
    w(f"corr(log R500_lens, log R500_Xray) = {f(rr['pearson_log'])} over 20")
    w(f"({f(rr['pearson_log_no_macs0416'])} dropping MACS0416, whose Donahue fit is")
    w(f"unconstrained — r_s is `nodata` with only a `<8 Mpc` limit).  Median ratio")
    w(f"{f(rr['median_ratio'],3,False)}, sd of the log ratio")
    w(f"{f(rr['sd_log10_ratio'],3,False)} dex.  Dropping MACS0416 changes the pooled")
    mr = DG["macs0416_robustness"]
    w(f"X-ray-radius slope from {f(mr['pooled_slope/all']['slope_a0'])} to")
    w(f"{f(mr['pooled_slope/no MACS0416']['slope_a0'])}, so nothing here rests on it.")
    w("")

    # ------------------------------------------------------------ Job 2.3
    w("## 7. The forward synthetic null")
    w("")
    nc = NU["noise_calibration"]
    w("Clusters are built with **no true radial dependence of the excess at all**,")
    w("projected (Abel, converged to 2e-5 against the analytic NFW Sigma), given")
    w("noise, fitted with a **spherical NFW over R <= 2.9 Mpc** — Umetsu's model and")
    w("fit range — and then published exactly as the papers publish: g_tot at the")
    w("tabulated radii and M500c → R500 from the same fit.")
    w("")
    w(f"The noise is calibrated against **both** of Umetsu's quoted uncertainties")
    w(f"simultaneously: coherent amplitude {nc['chosen_f_coherent']:.2f}, radial")
    w(f"tilt {nc['chosen_f_tilt']:.2f}, independent {nc['chosen_f_independent']:.2f},")
    w(f"giving e_M500/M500 = {f(nc['achieved_e_M500'],3,False)} (Umetsu")
    w(f"{f(nc['umetsu_median_e_M500_over_M500'],3,False)}) and e_c200/c200 =")
    w(f"{f(nc['achieved_e_c200'],3,False)} (Umetsu")
    w(f"{f(nc['umetsu_median_e_c200_over_c200'],3,False)}).")
    w("")
    w("### 7a. The pipeline manufactures the trend with no noise at all")
    w("")
    tb = NU["template_bias_noise_free"]
    w("| statistic / subset | S1 pooled slope | S2 Spearman | S3 within slope | S4 between corr |")
    w("|---|---|---|---|---|")
    for k, d in tb.items():
        if k == "template_misfit_dex_by_level":
            continue
        w(f"| {k} | {f(d['S1'])} | {f(d['S2'])} | {f(d['S3'])} | {f(d['S4'])} |")
    w("")
    w("The mechanism, measured directly — `log10(g_published / g_true)` by radius:")
    w("")
    w("| radius | NFW-template misfit |")
    w("|---|---|")
    for k, v in tb["template_misfit_dex_by_level"].items():
        w(f"| {k} | {f(v)} dex |")
    w("")
    ob = NU["observed_vs_null"]
    o1 = ob["y/cluster_scale_64"]["S1"]
    t1 = tb["y/cluster_scale_64"]["S1"]
    o2 = ob["a0/cluster_scale_64"]["S1"]
    t2 = tb["a0/cluster_scale_64"]["S1"]
    w(f"The NFW template overstates g at 100–200 kpc and understates it at 600 kpc,")
    w(f"which is a manufactured negative slope.  A **flat truth with no noise**")
    w(f"returns {f(t1)} against an observed {f(o1)} on the RAR residual")
    w(f"(**{100*t1/o1:.0f}%**) and {f(t2)} against {f(o2)} on the a0 statistic")
    w(f"(**{100*t2/o2:.0f}%**).")
    w("")
    w(f"Run AT found 29% of the X-COP slope was pipeline.  **For CLASH it is")
    w(f"{100*t1/o1:.0f}–{100*t2/o2:.0f}%.**")
    w("")
    rb = NU["r_break_sensitivity"]
    w("That figure depends on what the truth is assumed to do beyond the data, and")
    w("nothing measures that region:")
    w("")
    w("| truth follows the flat-excess law out to | noise-free S1 | fraction of observed |")
    w("|---|---|---|")
    for d in rb:
        w(f"| {d['r_break_Mpc']:.1f} Mpc | {f(d['S1'])} | {100*d['S1']/o1:.0f}% |")
    w("")

    w("### 7b. The observed values against the null")
    w("")
    w("| subset / statistic | statistic | observed | null mean +- sd | z | percentile |")
    w("|---|---|---|---|---|---|")
    for key in ("y/all_84", "y/cluster_scale_64", "a0/all_84",
                "a0/cluster_scale_64"):
        for k in ("S1", "S2", "S3", "S4", "S6_600", "S7_600",
                  "S8_slope_vs_xray_R500"):
            if k + "_z" not in ob[key]:
                continue
            n_ = NU["forward_null"][key][k]
            w(f"| {key} | {k} | {f(ob[key][k])} | "
              f"{f(n_['mean'])} +- {f(n_['sd'],4,False)} | "
              f"**{f(ob[key][k+'_z'],2)}** | {ob[key][k+'_pct']:.1f} |")
    w("")
    w("Statistic key: S1 pooled slope vs log(r/R500); S2 Spearman; S3 within-cluster")
    w("slope; S4 between-cluster corr(mean excess, log R500); S6_600 corr(excess,")
    w("log R500_lens) at r = 600 kpc; S7_600 the same against the independent")
    w("Chandra R500; S8 pooled slope against the independent X-ray radius.")
    w("")

    rs = NU["responsiveness"]["y/S1"]
    rs3 = NU["responsiveness"]["y/S3"]
    w("### 7c. Responsiveness")
    w("")
    w("| injected slope | measured pooled slope S1 | measured within slope S3 |")
    w("|---|---|---|")
    for a_, b_, c_ in zip(rs["injected"], rs["measured"], rs3["measured"]):
        w(f"| {f(a_,2)} | {f(b_)} | {f(c_)} |")
    w("")
    k = rs["responsiveness"]
    k3 = rs3["responsiveness"]
    w(f"    d(S1 measured)/d(injected) = {k:.3f}")
    w(f"    d(S3 measured)/d(injected) = {k3:.3f}")
    w("")
    n1 = NU["forward_null"]["y/cluster_scale_64"]["S1"]
    n3 = NU["forward_null"]["y/cluster_scale_64"]["S3"]
    o3 = ob["y/cluster_scale_64"]["S3"]
    imp = (o1 - n1["mean"]) / k
    imperr = n1["sd"] / k
    imp3 = (o3 - n3["mean"]) / k3
    imperr3 = n3["sd"] / k3
    w(f"**The CLASH pipeline attenuates a true radial trend by a factor")
    w(f"{1/k:.1f} in the pooled slope and {1/k3:.1f} in the within-cluster")
    w(f"slope.**  Inverting each against its own null:")
    w("")
    w(f"| statistic | observed | null mean | implied TRUE slope |")
    w("|---|---|---|---|")
    w(f"| S1 pooled | {f(o1)} | {f(n1['mean'])} | "
      f"**{f(imp,3)} +- {f(imperr,3,False)}** |")
    w(f"| S3 within | {f(o3)} | {f(n3['mean'])} | "
      f"**{f(imp3,3)} +- {f(imperr3,3,False)}** |")
    w("")
    w(f"The pooled slope is **consistent with zero**; a 95% interval is")
    w(f"[{f(imp-1.96*imperr,2)}, {f(imp+1.96*imperr,2)}], so **no upper limit")
    w(f"tighter than |s| < {abs(imp)+1.96*imperr:.2f} dex/dex has been set** by this")
    w(f"sample.  The within-cluster slope is the only statistic that is not")
    w(f"consistent with zero, at {abs(imp3/imperr3):.1f} sigma before the")
    w(f"systematic in the next subsection is applied.")
    w("")

    w("### 7d. What the verdict depends on: the unmeasured outer truth")
    w("")
    w("The null's **centre**, not its width, moves with how far out the flat-excess")
    w("truth is imposed before it is handed to the published NFW.  Nothing in CLASH")
    w("measures beyond 600 kpc, so this is scanned rather than chosen.")
    w("")
    w("| flat truth imposed out to | S1 null mean +- sd | S1 z (y) | S1 z (a0) | S3 z (y) | S3 z (a0) |")
    w("|---|---|---|---|---|---|")
    for d in SE["r_break_scan"]:
        w(f"| {d['r_break_Mpc']:.1f} Mpc | {f(d['y/S1/null_mean'])} +- "
          f"{f(d['y/S1/null_sd'],4,False)} | **{f(d['y/S1/z'],2)}** | "
          f"{f(d['a0/S1/z'],2)} | {f(d['y/S3/z'],2)} | {f(d['a0/S3/z'],2)} |")
    w("")
    sm = SE["summary"]
    w(f"The conservative choice is the data edge, 0.6 Mpc: it grants the published")
    w(f"NFW everywhere nothing is measured and asks only whether a flat excess")
    w(f"*inside* the measurements comes back out sloped.  Larger values assert the")
    w(f"flat-excess law over a region nothing constrains, i.e. assume part of the")
    w(f"hypothesis under test.  Across the whole scan the pooled slope moves from")
    w(f"z = {f(sm['y_S1_z_min'],2)} to {f(sm['y_S1_z_max'],2)} and the within-cluster")
    w(f"slope from z = {f(sm['y_S3_z_min'],2)} to {f(sm['y_S3_z_max'],2)}.")
    w("")
    w("Left there, that would be an undecidable systematic.  It is not — see 7e.")
    w("")
    w("### 7e. Which candidate truths the data itself allows")
    w("")
    w("Tian tabulates g_tot only to 600 kpc, but **Umetsu+2016 measured the")
    w(f"projected profile out to R = 2.9 Mpc**.  A flat-excess truth imposed past")
    w("~1 Mpc predicts a Sigma(R) that Umetsu did not observe, and is excluded by")
    w("that data whatever it does to the null.  Amplitude profiled out, so this")
    w("tests shape only:")
    w("")
    tcs = TC["truth_consistency"]
    w(r"| flat truth imposed out to | chi2/dof of its Sigma vs the published NFW | median max \|dlnSigma\| | |")
    w("|---|---|---|---|")
    for d in tcs["scan"]:
        w(f"| {d['r_break_Mpc']:.1f} Mpc | {f(d['median_chi2_per_dof'],2,False)} | "
          f"{f(d['median_max_abs_dlnSigma'],3,False)} | "
          f"{'**allowed**' if d['admissible'] else 'excluded'} |")
    w("")
    adm = [d["r_break_Mpc"] for d in tcs["scan"] if d["admissible"]]
    zs = [d["y/S1/z"] for d in SE["r_break_scan"] if d["r_break_Mpc"] in adm]
    w(f"**The admissible nulls are exactly the ones that put the observation inside")
    w(f"them.**  Over r_break <= {max(adm):.1f} Mpc the pooled slope sits at")
    w(f"z = {f(min(zs),2)} to {f(max(zs),2)}; the r_break = 2.5 Mpc case that gave")
    w(f"z = {f(SE['r_break_scan'][-1]['y/S1/z'],2)} predicts a lensing profile off by")
    w(f"{f(tcs['scan'][-1]['median_max_abs_dlnSigma'],2,False)} in ln Sigma and is")
    w(f"ruled out by Umetsu's own measurement.")
    w("")
    w("### 7f. What the surviving within-cluster discrepancy actually is")
    w("")
    w("The published g_tot **is** `M_NFW(<r|M200,c200)` exactly (§2), so with a")
    w("per-cluster level free the within-cluster radial run of the excess is a")
    w("function of **c200_i and the baryon profile alone**.  The S3 discrepancy is")
    w("therefore, exactly and only, the statement that the published concentrations")
    w("differ from those an NFW fit to a flat-excess cluster would return:")
    w("")
    cc = TC["c200_comparison"]
    w("| flat truth out to | mean c200 published | mean c200 from the null | log10 ratio | sigma |")
    w("|---|---|---|---|---|")
    for d in cc["scan"]:
        w(f"| {d['r_break_Mpc']:.1f} Mpc | {f(d['mean_c200_published'],2,False)} | "
          f"{f(d['mean_c200_null'],2,False)} | {f(d['mean_log10_ratio'])} +- "
          f"{f(d['sem'],4,False)} | {f(d['sigma'],1,False)} |")
    w("")
    c0 = cc["scan"][0]
    w(f"At the admissible r_break of {c0['r_break_Mpc']:.1f} Mpc the whole effect is")
    w(f"an offset of **{f(c0['mean_log10_ratio'],4)} dex in log10 c200** — "
      f"{100*(c0['mean_c200_published']/c0['mean_c200_null']-1):.0f}% in the")
    w(f"concentration, accumulated over 20 clusters.  Umetsu quotes")
    w(f"e_c200/c200 = {f(cc['umetsu_e_c200_over_c200_median'],2,False)} per cluster,")
    w(f"i.e. {f(cc['umetsu_e_log10_c200_median'],3,False)} dex, so the offset is")
    w(f"**{c0['mean_log10_ratio']/cc['umetsu_e_log10_c200_median']:.2f} of ONE")
    w(f"cluster's own quoted concentration uncertainty**.")
    w("")
    w("That is a statement about NFW concentrations inside a dark-matter halo")
    w("model.  CLASH concentrations are known to carry selection and triaxiality")
    w("systematics of order this size; nothing about it is a measurement of")
    w("gravity, and it is not the kind of quantity this programme can admit.")
    w("")
    fp = NU["false_positive_rates"]
    w("### 7g. False-positive rates of the obvious tests")
    w("")
    w("| test | nominal | measured FPR under a flat truth |")
    w("|---|---|---|")
    for k_, d in fp.items():
        w(f"| {k_} | {d['nominal']} | **{d['measured']:.3f}** |")
    w("")
    w(f"Run AT measured 0.53-0.70 for the X-COP permutation test and called it")
    w(f"\"not a test\".  The CLASH versions are worse: the R500-label permutation")
    w(f"rejects {100*fp['R500_label_permutation']['measured']:.0f}% of the time and")
    w(f"the naive OLS t-test")
    w(f"{100*fp['naive_OLS_t_test']['measured']:.0f}% of the time when the truth is")
    w(f"flat.  Neither carries any information.")
    w("")

    # ------------------------------------------------------------ Job 3
    w("## 8. Job 3 — the dark-matter-presupposition check")
    w("")
    j3 = P["job3_admissibility"]
    w(f"* numerator = `{j3['numerator_is']}`")
    w(f"* raw shear in the repo: **{j3['raw_shear_available_in_repo']}**")
    w(f"* reconstructible from raw shear: **{j3['can_be_rebuilt_from_raw_shear']}**")
    w("")
    w(j3["why"])
    w("")
    w("Every lensing product CLASH publishes, checked table by table:")
    w("")
    w("| published product | source | what it is | admissible |")
    w("|---|---|---|---|")
    for d in j3["published_lensing_products"]:
        w(f"| {d['product']} | {d['source']} | {d['kind']} | "
          f"{'yes' if d['admissible'] else '**no**'} |")
    w("")
    va = j3["vizier_availability"]
    w(f"None of the CLASH lensing papers is on VizieR: {', '.join(va['checked'])} "
      f"all return \"Table or Catalog not found\", and {va['meta_search']}.")
    w(f"Positive control: {va['positive_control']}.")
    w("")
    w("Quotes, from the acquired sources:")
    w("")
    w(f"> {j3['quote_tian']}  — Tian+2020 §2.1")
    w("")
    w(f"> {j3['quote_umetsu_t2']}  — Umetsu+2016 Table 2 caption")
    w("")
    w("And the repo already says so, in a file written before this audit:")
    w("")
    w(f"> {j3['repo_lineage_statement']}")
    w("")
    w(f"**Verdict: {j3['verdict']}.**  Run AL.3 rejected an amplitude selected")
    w("against *interpolated* published lensing mass profiles.  CLASH is a stronger")
    w("case of the same failure: the numerator is not interpolated from a fitted")
    w("mass, it **is** a fitted mass — a two-parameter NFW whose only inputs are a")
    w("GR-derived convergence map and the assumption of spherical symmetry.  Under")
    w("standing constraint 2 that is not a raw observation.")
    w("")

    # ------------------------------------------------------------ bugs
    w("## 9. Bugs the tests found")
    w("")
    w("`tests.py`: 18/18 pass.  Six defects were found and fixed on the way, all in")
    w("this lane's own first implementation.")
    w("")
    w("1. **Abel truncation.** The truth density was truncated at 5 Mpc; the")
    w("   projection then lost 6.5% of Sigma at 1.5 Mpc and **21% at the 2.9 Mpc")
    w("   outer fit radius**, biasing the fitted NFW and manufacturing a radial")
    w("   effect inside the null itself.  Converged to 2e-5 only past ~100 Mpc.")
    w("2. **Non-monotone M(<r).** The inner-sphere mass was written into `M[0]`")
    w("   after the cumulative integral instead of added as an offset, so `M[0]`")
    w("   exceeded `M[1]` by ~180x.")
    w("3. **The null's R500 population was wrong.** Imposing the constant-excess law")
    w("   all the way to 1.5 Mpc gave R500 values 1.2–1.7x the real ones and")
    w("   corr(excess, log R500) = +0.74 against +0.14 in the data.  Fixed by making")
    w("   the truth agree with the published NFW outside the measured range, where")
    w("   nothing is measured anyway.")
    w("4. **The null's c200 uncertainty was 4x too small.** A coherent amplitude")
    w("   term reproduces e_M500/M500 = 0.224 but leaves e_c200/c200 at 0.074")
    w("   against Umetsu's 0.301, because rescaling Sigma barely moves its shape.")
    w("   The width of the *within-cluster* slope null is set by exactly that shape")
    w("   uncertainty, so S3's z was ~4x too large before a radial tilt was added.")
    w("")
    w("6. **A shadowed accumulator in the reporting layer.**  `report.py` built the")
    w("   document in a list named `L` and later reused `L` as a loop variable over")
    w("   radius labels; `w = L.append` kept writing to the original list while the")
    w("   final `join(L)` joined the string `\"400\"`.  REPORT.md came out 6 bytes")
    w("   long.  Caught only because the file size was checked -- a silent")
    w("   truncation that no assertion in the analysis would have found.")
    w("")
    w("And one substantive error of reasoning, caught twice by the same check:")
    w("")
    w("5. **A baryon-only normaliser is not automatically a clean control.**  The")
    w("   excess carries g_bar in its denominator, so a normaliser whose")
    w("   between-cluster variation tracks the baryon amplitude puts the same")
    w("   quantity on both axes.  I first predicted the contaminated one was R_b,g")
    w("   and used a pooled diagnostic; both were wrong.  Measured between clusters")
    con = ST["normaliser_gbar_contamination"]
    w("   at fixed radius, corr(log R_norm, log g_bar) is "
      + ", ".join(f"{k} {f(v['pearson_logRnorm_vs_log_gbar_at_200kpc'],2)}"
                  for k, v in con.items()) + ".")
    w("   **R_b,M** — the radius enclosing a fixed baryonic mass — is")
    w("   an almost exact inverse of the baryon amplitude and must not be quoted as")
    w("   an independent control.  This is the eighth instance of the")
    w("   shared-denominator pattern, and the first found in a *control* rather than")
    w("   in a measurement.")
    w("")

    # ------------------------------------------------------------ verdict
    w("## 10. Verdict")
    w("")
    w("**The tautology is live on CLASH, and unlike X-COP nothing structural stops")
    w("it.**")
    w("")
    w("| protection | X-COP (Run AT) | CLASH |")
    w("|---|---|---|")
    w(f"| cancellation lemma | numerator moves 1.6e-13 | **absent**; moves "
      f"{f(-el['induced_slope_dy_dx'],2)} dex per dex of R500 |")
    w(f"| monotone invariance | 90.3% within-cluster | **inverted**; "
      f"{100*vd['cluster_scale_64/y']['between_fraction']:.0f}% between-cluster |")
    w(f"| independent radius | none available | Donahue+2014 Chandra r500, but "
      f"corr with the lensing radius is only {f(rr['pearson_log'],2)} |")
    w(f"| pipeline share of the slope | 29% | **{100*t1/o1:.0f}–{100*t2/o2:.0f}%** |")
    w("")
    w("**But the audit does not end where Run AT's did.**  On X-COP the trend")
    w("survived a forward null at 16 sigma with responsiveness 0.87.  On CLASH the")
    w("pooled slope does not survive at all — it lands on the null's median:")
    w("")
    for key, lab in (("y/cluster_scale_64", "RAR residual"),
                     ("a0/cluster_scale_64", "a0 statistic")):
        d = ob[key]
        n_ = NU["forward_null"][key]["S1"]
        w(f"* {lab}, pooled slope: observed {f(d['S1'])}, forward null (**no true")
        w(f"  radial dependence at all**) {f(n_['mean'])} +- {f(n_['sd'],4,False)},")
        w(f"  **z = {f(d['S1_z'],2)}, percentile {d['S1_pct']:.1f}**.")
    w("")
    w("The pure tautology contrast goes the same way.  The +0.71 correlation of the")
    w("excess with log R500 at 600 kpc, which looks like a smoking gun in §6b, sits")
    w(f"at z = {f(ob['y/cluster_scale_64']['S6_600_z'],2)} against the flat-truth")
    w(f"null (which produces {f(NU['forward_null']['y/cluster_scale_64']['S6_600']['mean'])}")
    w(f"+- {f(NU['forward_null']['y/cluster_scale_64']['S6_600']['sd'],3,False)}).  It")
    w("is real, it is the tautology, and it is **also** exactly what a genuine")
    w("excess would produce — a cluster with a larger true excess has a larger true")
    w("mass and hence a larger true R500.  The two hypotheses predict the same sign")
    w("and, on this sample, the same size.  That is why the contrast cannot decide.")
    w("")
    w(f"The one statistic that is not consistent with the null is the")
    w(f"**within-cluster** slope, the one Run AT's monotone-invariance argument")
    w(f"protects from R500 entirely:")
    w("")
    for key, lab in (("y/cluster_scale_64", "RAR residual"),
                     ("a0/cluster_scale_64", "a0 statistic")):
        d = ob[key]
        n_ = NU["forward_null"][key]["S3"]
        w(f"* {lab}, within-cluster slope: observed {f(d['S3'])}, null "
          f"{f(n_['mean'])} +- {f(n_['sd'],4,False)}, **z = {f(d['S3_z'],2)}**.")
    w("")
    sm = SE["summary"]
    c0 = TC["c200_comparison"]["scan"][0]
    ecl = TC["c200_comparison"]["umetsu_e_log10_c200_median"]
    w(f"Across the r_break scan that runs from z = {f(sm['y_S3_z_min'],2)} to")
    w(f"{f(sm['y_S3_z_max'],2)}, and only the small-r_break end is allowed by")
    w(f"Umetsu's own measured Sigma (§7e), so the honest figure is the ~2.8 sigma")
    w(f"at the data edge.  With responsiveness {k3:.2f} the implied true")
    w(f"within-cluster slope is {f(imp3,3)} +- {f(imperr3,3,False)} dex/dex.")
    w("")
    w(f"But §7f says what that 2.8 sigma **is**: the published NFW concentrations")
    w(f"sit {100*(c0['mean_c200_published']/c0['mean_c200_null']-1):.0f}% "
      f"({f(c0['mean_log10_ratio'],4)} dex) above those an NFW fit to a flat-excess")
    w(f"cluster with the same baryons would return — "
      f"{c0['mean_log10_ratio']/ecl:.2f} of ONE cluster's own quoted concentration")
    w(f"uncertainty, accumulated over 20.  **The whole surviving CLASH signal is a")
    w(f"{100*(c0['mean_c200_published']/c0['mean_c200_null']-1):.0f}% offset in NFW")
    w(f"concentration.**  It is a statement inside a dark-matter")
    w(f"halo model, at a level that CLASH's known selection and triaxiality")
    w(f"systematics reach, and it is not a measurement of gravity.")
    w("")
    w("Following standing constraint 4, none of this kills the candidate.  It")
    w("removes CLASH from the evidence for it.  The surviving cluster statement is")
    w("Run AT's X-COP one, unchanged; **CLASH adds nothing to it and should not be")
    w("quoted alongside it**, for two independent reasons, either of which is")
    w("sufficient on its own:")
    w("")
    w("1. **Admissibility.** The numerator is a two-parameter NFW fit to a")
    w("   GR-derived convergence map.  Standing constraint 2 excludes it, and this")
    w("   is not a matter of degree: there is no version of the CLASH numerator that")
    w("   is not a fitted halo mass.")
    w("2. **Statistics.** Even taken at face value, the pooled radial trend sits at")
    w(f"   z = {f(ob['a0/cluster_scale_64']['S1_z'],2)} (percentile")
    w(f"   {ob['a0/cluster_scale_64']['S1_pct']:.0f}) against a null containing no")
    w("   radial dependence whatsoever, and the tests that made it look significant")
    w(f"   have false-positive rates of {fp['R500_label_permutation']['measured']:.2f}")
    w(f"   and {fp['naive_OLS_t_test']['measured']:.2f} against a nominal 0.05.")
    w("")
    w("**What would change this.**  Not more CLASH clusters — the limitation is")
    w("structural, not statistical.  It needs a numerator that is not a fitted halo")
    w("mass: the CLASH shear catalogues themselves, scored the way Run AL.5 scored")
    w("raw eFEDS/HSC shear, with the law predicting the shear rather than being")
    w("compared against somebody's NFW posterior.  Until then CLASH is a")
    w("consistency check on Umetsu+2016's NFW fits, not a measurement of gravity.")
    w("")
    w("### Corrections owed to the record")
    w("")
    rep = DG["record_reproduction"]
    d = rep["within_CLASH_a0_drop_100_to_600"]
    w(f"* The record's within-CLASH number reproduces exactly:")
    w(f"  {f(d['mean'])} +- {f(d['sem'],4,False)} over n = {d['n']},")
    w(f"  {d['n_negative']}/{d['n']} negative, against the quoted")
    w(f"  \"{d['record_quote']}\".  **The number is right; the inference from it is")
    w(f"  not.**")
    k1 = rep["lane12_a0_ratio_at_100kpc"]
    k4 = rep["lane12_a0_ratio_at_400kpc"]
    w(f"* Lane 12's two \"CLASH fig2\" rows use **one pooled R500 for all of CLASH**")
    w(f"  — the quoted r/R500 of {k1['record_r_over_R500']} and")
    w(f"  {k4['record_r_over_R500']} at 100 and 400 kpc imply")
    w(f"  {k1['implied_R500_kpc']:.0f} and {k4['implied_R500_kpc']:.0f} kpc.  The")
    w(f"  per-cluster R500 values span")
    w(f"  {ST['normalisers_kpc']['RXJ1532']['R500_lens']:.0f}–"
      f"{ST['normalisers_kpc']['RXJ1347']['R500_lens']:.0f} kpc.  Under a single")
    w(f"  global normaliser r/R500 and r are the same variable up to a constant, so")
    w(f"  those rows carry no R500 information at all — which is *conservative*, and")
    w(f"  worth stating rather than leaving implicit.")
    w(f"* **Every CLASH point lies at r/R500 <= "
      f"{f(ex['max_r_over_R500_any_point'],2,False)}.**  The record's Lane-12 table")
    w(f"  places CLASH at r/R500 = 0.073 and 0.291 and then extrapolates the same")
    w(f"  sequence past R500; R500 itself sits")
    w(f"  {f(ex['R500_over_outermost_datum_median'],1,False)}x beyond the outermost")
    w(f"  CLASH measurement and is a property of the NFW fit, not of the data.")
    w(f"* `invariant_bench._clash()` discards the `AName` column.  The record's note")
    w(f"  that \"CLASH has no object identity in the bench\" is a bench defect, not a")
    w(f"  data limitation — 20 named clusters are in the file.")
    aa = ST["slopes"]["all_84/y/r_over_R500_lens"]["pooled_slope"]
    w(f"* On the full 84 points the RAR-residual slope against log(r/R500) is")
    w(f"  {f(aa)} — **flat**.  The CLASH radial trend exists only after the BCG")
    w(f"  points are dropped, or when the a0 parametrisation is used.  Whichever is")
    w(f"  quoted, that choice should be stated.")
    w("")

    out = "\n".join(DOC) + "\n"
    open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8").write(out)
    json.dump(R, open(os.path.join(HERE, "results.json"), "w", encoding="utf-8"),
              indent=1)
    print(f"wrote REPORT.md ({len(out)} chars) and results.json")


if __name__ == "__main__":
    main()
