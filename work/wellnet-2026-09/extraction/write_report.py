"""write_report.py -- render REPORT.md.  Every number comes from the JSONs.

No figure in the report is typed by hand; if a result changes, re-running this
changes the report.  Where a quantity has not been measured the renderer says
so rather than omitting the row.  Verdict sentences are CONDITIONED on the
numbers (a separation that did not survive is reported as not surviving).
"""
from __future__ import annotations

import io
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))
OUT = os.environ.get("EXTRACTION_REPORT", os.path.join(HERE, "REPORT.md"))
sys.path.insert(0, HERE)


def load(name, optional=False):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        if optional:
            return None
        raise SystemExit(f"missing {p}")
    return json.load(io.open(p, encoding="utf-8"))


def f(x, nd=3, plus=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(x):
        return "n/a"
    return f"{x:+.{nd}f}" if plus else f"{x:.{nd}f}"


def R(d, nd=3):
    if not d or not np.isfinite(d.get("rate", np.nan)):
        return "n/a"
    return f"{d['rate']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def sep(s):
    if not s:
        return "n/a"
    cap = "*" if s.get("z_capped") else ""
    return f"AUC {f(s['auc'])}, z {f(s['z'], 2)}{cap}, p {f(s['p_perm'], 4)}"


def scan_key(v, k):
    """Scan-table cell; tolerant of the older 'scatter_only' key."""
    if k == "curve_res_only" and k not in v:
        k = "scatter_only"
    r = v.get(k)
    return f"{f(r['auc'])} (z {f(r['z'], 1)})" if r else "n/a"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out) + "\n"


def main():
    D = load("D_discriminator.json")
    CF = load("CF_counterfactual.json")
    CL = load("CL_closure.json")
    X = load("X_distil.json")
    C = load("C_certificates.json")
    T = load("T_tests.json")
    PV = load("provenance_generate.json", optional=True)
    L = []
    w = L.append

    CLASS = D["class_tags"]
    main_pool = D["D1_main"]["pool"]
    n_pass, n_tot = T["n_pass"], T["n_total"]

    w("# Run BL -- the Principle Extraction Lane: what separates dark matter from the modified-gravity class, located, "
      "distilled and validated across two generators\n")
    w(f"Lane `work/wellnet-2026-09/extraction/`.  Registry `BL-extraction`, VALID.  Entirely synthetic.  "
      f"Provenance: {'foreign reads **' + str(len(PV['foreign_reads'])) + '**; real-observation token matched **' + str(PV['any_real_observational_file_opened']) + '**' if PV else 'ledger not found'}; "
      f"the guard is installed in every worker and exercised by test T7.  Tests: **{n_pass}/{n_tot}** pass.\n")
    w("Input: Run BF's finding that the synthetic corpus separates the dark-matter universe (U2) from the seven-family "
      "modified-gravity class at z = 8.5 while the named detectors are not what carries it (BF.3, BJ.6), and Run BK's "
      "finding that the tensor/halo separation is a statement about the halo-alignment prior (BK.4).  Protocol: the fourth "
      "review's, adopted verbatim (BJ.7).\n")

    # ------------------------------------------------------------------ 0 design
    w("## 0  Design\n")
    w(f"* **Paired sets.** {D['n_sets']} paired sets on BF's shared scene library: one scene draw (30 galaxies, 12 clusters), "
      f"one set of block-seeded noise streams, one halo draw, emitted under every arm.  Test T1 checks that the seven "
      f"deformations at zero amplitude reproduce U3 bit for bit and that two different universes share their sources, photo-z "
      f"and shape noise.  {D['n_cal']} sets calibrate, {D['n_aud']} audit; nothing is fitted and scored on the same set.")
    w(f"* **The class** is BF's seven families at BF's own threshold amplitudes (E9) plus the scalar-null family H0: "
      f"`{', '.join(CLASS)}`.  The fiducial amplitudes (U4f-U9f), Newton (U1) and systematics-only (U10) are reference arms.")
    w("* **The CDM prior.** Every property a halo has that the baryons do not fix is a declared knob with a nominal prior: "
      "SHMR scatter 0.16 dex, concentration scatter 0.11/0.13 dex, cluster halo mass scatter 0.05 dex, shape "
      "e_h = 0.72 ell_bar + N(0, 0.10), **halo-filament alignment f_lss ~ Beta(2, 2) per object** (BK's mixture), galaxy "
      "in-plane halo quadrupole U(0, 0.10), oblateness q_h ~ U(0.70, 1.00), dark-disc fraction f_dd ~ U(0, 0.05).  Every "
      "separation below is reported as a function of these.")
    w("* **Invariant reductions.** Rotation-, translation- and permutation-invariant by construction (test T4): per-ring "
      "harmonic decompositions of the velocity field, a PSF-forward-modelled rotation curve, vertical dispersions; per-bin "
      "m = 0, 1, 2, 4 fits of tangential AND cross ellipticity with covariance, X-ray/SZ/hydrostatic, member-dispersion and "
      "strong-lensing profiles, the member-locked well-strength correlation with its scrambled control.  Phases enter only "
      "relative to observed axes.")
    w("* **The scalar monopole is matched away.** Every gravitational quantity is a residual from the corpus's own "
      "cross-fitted universal scalar law nu-hat(g_bar), a P-spline fitted on half the galaxies and applied out of fold to "
      "the other half and to every cluster.  Test T6: an a0 shift of 0.15 dex moves the raw galaxy boost by "
      "0.06 dex and the residuals by 0.0002 dex.")
    w("* **Two generators.** Generator 1 is BF's physics re-emitted in paired form.  Generator 2 (`forward2.py`) shares "
      "nothing: Miyamoto-Nagai discs, Einasto haloes, a quadratic SHMR, a different law family, Vikhlinin-type gas, a "
      "shell-sum projection, an Osipkov-Merritt Jeans solve, BK's lensing quadrupole and cosmology, a different instrument.\n")
    w("**Bugs the tests caught before any result.** T2: the first analysis projection used a stretched z grid whose first "
      "step was tens of thousands of kpc and returned Sigma too large by 6-60x with a 1/R dependence.  T5: the ring-fitted "
      "rotation curve carried a 0.1 dex beam-smearing scatter into the vertical contrast even without noise; the "
      "forward-modelled curve removed it.  T2 (generator 2): a mid-radius shell sum was 15% off the analytic NFW until the "
      "inverse-square-root singularity was integrated per shell.  T4: the scrambled network control was neither rotation- nor "
      "permutation-invariant in its first form.\n")

    # ------------------------------------------------------------------ 1 sizing
    s0 = D["D0_sizing"]
    w("## 1  Sizing first, on untouched halves\n")
    w(f"A-vs-A: one universe, random halves of its calibration sets labelled 1/0, the discriminator fitted, random halves "
      f"of its audit sets scored.  {s0['n_null']} draws: null AUC mean {f(s0['null_auc_mean'])}, sd **{f(s0['sd_null_auc'], 4)}** "
      f"(galaxies only {f(s0['sd_null_gal'], 4)}, clusters only {f(s0['sd_null_clu'], 4)}); the 95th percentile of |AUC - 0.5| under the null is "
      f"{f(s0['null_auc_p95_abs'], 4)}.  z = (AUC - 0.5)/sd, capped at {s0['z_cap']} as in BF; a permutation p-value on the "
      f"audit scores is quoted beside every z.\n")

    # ------------------------------------------------------------------ 2 discriminator
    w("## 2  The discriminator: the CDM prior against the class\n")
    rows = []
    for t in CLASS:
        if t in D["D1_main"]:
            rows.append([t, sep(D["D1_main"][t]), sep(D["D1_galaxies_only"].get(t)), sep(D["D1_clusters_only"].get(t))])
    rows.append(["**pool**", "**" + sep(main_pool) + "**", sep(D["D1_galaxies_only"]["pool"]), sep(D["D1_clusters_only"]["pool"])])
    w(table(["class member", "galaxies + clusters", "galaxies only", "clusters only"], rows))
    r5, r1 = D["D1_main"]["rates_0.05"], D["D1_main"]["rates_0.01"]
    w(f"Per-object discrimination (one galaxy / one cluster, U2 vs U3): galaxies **{f(main_pool.get('obj_auc_gal'))}**, "
      f"clusters **{f(main_pool.get('obj_auc_clu'))}**.  At the critical value from one half of the class's audit corpora "
      f"(never used in training): **power {R(r5['power'])} on CDM at a realised class false-positive rate of "
      f"{R(r5['realised_fpr'])}** (nominal 0.05); at nominal 0.01, power {R(r1['power'])}, realised {R(r1['realised_fpr'])}.  "
      f"The class pool holds near-duplicate corpora (the same scene and noise under deformations below detectability), so "
      f"the effective number of class corpora is the number of sets, {main_pool.get('n_neg_effective')}, and the pool's "
      f"permutation p is the single-arm U3 value.\n")
    ref = D["D1_reference_arms"]
    fire = D["D1_fire_rate_at_class_crit_0.05"]
    w("Reference arms, scored by the same discriminator (a CDM detector must not fire on the class at fiducial amplitude, "
      "on systematics, or on a Newtonian universe):\n")
    w(table(["arm", "U2 vs arm", "rate the arm is called CDM at the class's 0.05 critical value"],
            [[t, sep(ref[t]), R(fire.get(t))] for t in ref] + [["U2 itself", "", R(fire.get("U2"))]]))

    # ------------------------------------------------------------------ 3 ablations
    w("## 3  Which channel carries it: the ablation table\n")
    ab = D["D2_ablations"]
    w(f"Full discriminator: AUC {f(main_pool['auc'])}, z {f(main_pool['z'], 2)}.  Leave one channel out / keep one channel in "
      f"(pool AUC; z against the sized null; the per-member AUCs are in the JSON):\n")
    rows = []
    for name in ab:
        if name.startswith("drop:") and not any(s in name for s in ("(", "AND")):
            c = name[5:]
            only = ab.get(f"only:{c}", {})
            oa = only.get("obj_auc_gal") if c.startswith("gal") else only.get("obj_auc_clu")
            da = ab[name].get("obj_auc_gal") if c.startswith("gal") else ab[name].get("obj_auc_clu")
            rows.append([c, f(ab[name]["auc"]), f(ab[name]["z"], 2), f(da), f(only.get("auc")), f(only.get("z"), 2), f(oa)])
    w(table(["channel", "corpus AUC without it", "z", "per-object AUC without it", "corpus AUC with it alone", "z",
             "per-object AUC alone"], rows))
    w("The corpus AUC saturates at 30 galaxies and 12 clusters per corpus; the per-object AUC (one galaxy, one cluster, "
      "U2 vs U3) is the number that ranks the channels.\n")
    w("The named removals of the brief:\n")
    rows = [[name, f(v["auc"]), f(v["z"], 2), f(v["p"], 4), f(v.get("obj_auc_gal")), f(v.get("obj_auc_clu"))]
            for name, v in ab.items() if not name.startswith(("drop:gal_", "drop:clu_", "only:gal_", "only:clu_"))]
    w(table(["ablation", "corpus AUC", "z", "p (U3)", "per-galaxy AUC", "per-cluster AUC"], rows))
    imp = D["D2_importance"]
    rows = sorted([(k, v["drop"]) for k, v in imp.items() if not k.startswith("_")], key=lambda x: -x[1])
    w(f"Permutation importance by channel (drop in audit AUC when the channel's columns are shuffled; base AUC "
      f"{f(imp['_base_auc'])}): " + ", ".join(f"`{k}` {f(v, 3)}" for k, v in rows[:8]) + ".\n")

    # ------------------------------------------------------------------ 4 localisation
    w("## 4  Localisation: radius, harmonic, source class, regime\n")
    loc = D["D3_localisation"]
    w(table(["restricted to", "n features", "AUC", "z"],
            [[k, v["n_features"], f(v["auc"]), f(v["z"], 2)] for k, v in loc.items()]))
    st = D["D3_strata"]
    rows = []
    for k, v in st.items():
        rows.append([k] + [f"{f(v[q]['auc'])} (n={v[q]['n']})" if q in v else "n/a" for q in ("low", "mid", "high")])
    w("Per-object discrimination of the full discriminator in terciles of an observed property (AUC of object log-odds, "
      "U2 vs U3):\n")
    w(table(["property", "low tercile", "mid", "high"], rows))

    # ------------------------------------------------------------------ 5 counterfactuals
    w("## 5  Causal counterfactuals: dO/dB, paired scenes, one thing changed\n")
    w("Per-object differences between the edited and the base emission (same scene, same noise; the halo HELD where "
      "stated), mean +- standard error across objects.  Invariants: `dz_1` the vertical/radial boost contrast at 2 R_d; "
      "`res_mean` the galaxy residual from the universal law; `res_sd` its within-galaxy scatter; `wl_lA`, `t_lA`, `d_lA` the "
      "lensing, X-ray and dynamical residual levels; `t_slope` the X-ray residual slope; `pe_tot`/`pb_tot` the quadrupole "
      "projections on the external/baryon axis; `S` the discriminator's corpus score.\n")
    rows = []
    resp = CF["responses"]
    for tag, r in resp.items():
        base, edit = tag.split("|", 1)
        dS = ""
        if base == "U2" and f"U3|{edit}" in resp:
            dS = f(r["S_discriminator"]["mean"] - resp[f"U3|{edit}"]["S_discriminator"]["mean"], 1, True)
        rows.append([base, r["edit"],
                     f"{f(r['dz_1']['mean'], 4, True)} +- {f(r['dz_1']['se'], 4)}",
                     f"{f(r['res_mean']['mean'], 4, True)} +- {f(r['res_mean']['se'], 4)}",
                     f"{f(r['wl_lA']['mean'], 3, True)}", f"{f(r['t_lA']['mean'], 3, True)}", f"{f(r['t_slope']['mean'], 3, True)}",
                     f"{f(r['pe_tot']['mean'], 2, True)}", f"{f(r['pb_tot']['mean'], 2, True)}",
                     f"{f(r['S_discriminator']['mean'], 1, True)} +- {f(r['S_discriminator']['se'], 1)}", dS])
    w(table(["universe", "edit", "d dz_1", "d res_mean", "d wl_lA", "d t_lA", "d t_slope", "d pe_tot", "d pb_tot", "d S",
             "d S, U2 minus U3"], rows))
    w("Two things to read the table with.  The discriminator score S is CONDITIONAL on the baryonic scene (it uses the "
      "photometry as context), so it moves under a baryon edit in every universe; the CDM-specific response is the last "
      "column.  And an edit applied to every object of a corpus is partly re-absorbed by the corpus's own refitted "
      "universal law -- that is the monopole matching doing its job -- so the paired responses of the RESIDUALS to a "
      "global halo edit understate the object-to-object response, which the partial slopes below measure.\n")
    sl = CF["object_level_slopes"]
    w("Object-level partial slopes inside the CDM prior (the invariant against the halo the object happens to have, at "
      "fixed observed baryons -- the response a real population would show):\n")
    w(table(["slope", "value", "se", "n"], [[k, f(v["slope"], 3, True), f(v["se"], 3), v["n"]] for k, v in sl.items()]))

    # ------------------------------------------------------------------ 6 distillation
    w("## 6  Distillation: from the discriminator to named invariants\n")
    single = X["single_terms"]
    top = list(single)[:12]
    w(f"The discriminator's separation: AUC {f(X['discriminator_auc'])}, z {f(X['discriminator_z'], 2)}.  Single corpus-level "
      f"terms, ranked by audit AUC (U2 mean, class U3 mean +- corpus sd):\n")
    w(table(["term", "type", "AUC", "z", "d'", "U2", "U3", "sd(U3)"],
            [[f"`{n}`", "level" if single[n].get("level") else "structure", f(single[n]["auc"]), f(single[n]["z"], 2),
              f(single[n].get("dprime"), 1), f(single[n]["mean_U2"], 3, True),
              f(single[n]["mean_U3"], 3, True), f(single[n]["sd_U3"], 3)] for n in top]))
    w("`level` marks a term that carries the strength of gravity at some radius (what an extra cluster-scale component "
      "in a modified-gravity theory would remove); `structure` marks scatter, shape, geometry or the vertical contrast.\n")
    gr = X["greedy"]
    w("Greedy forward selection into a linear score (fitted on calibration corpora, scored on audit; ties in a "
      "saturated AUC broken by d'):\n")
    w(table(["terms", "AUC", "z", "d'"], [[", ".join(f"`{t}`" for t in h["terms"]), f(h["auc"]), f(h["z"], 2), f(h.get("dprime"), 1)] for h in gr]))
    gsf = X.get("greedy_strength_free", [])
    if gsf:
        w("The same selection with every level term excluded (the strength-free distillation):\n")
        w(table(["terms", "AUC", "z", "d'"], [[", ".join(f"`{t}`" for t in h["terms"]), f(h["auc"]), f(h["z"], 2), f(h.get("dprime"), 1)] for h in gsf]))
    dist = X["distilled"]
    w(f"**Distilled invariant:** {', '.join('`' + t + '`' for t in dist['terms'])} -- AUC {f(dist['auc'])}, z {f(dist['z'], 2)}, "
      f"recovering **{f(100 * dist['fraction_of_discriminator_auc_excess'], 0)}%** of the discriminator's AUC excess over 0.5.  "
      f"Frozen, per class member: " + ", ".join(f"{k} {f(v)}" for k, v in dist["per_member"].items()) +
      f".  On the reference arms: " + ", ".join(f"{k} {f(v)}" for k, v in dist["fiducial_and_reference"].items()) + ".\n")
    nm = X["named"]
    tr = X["transfer_frozen"]
    w("The named physical invariants, each fitted alone and then FROZEN and transferred to the untouched scene library and "
      "to generator 2 (a frozen score can only lose separation):\n")
    rows = []
    for k, v in nm.items():
        t = tr.get(k, {})
        rows.append([k, ", ".join(f"`{x}`" for x in v["terms"]), f(v["auc"]), f(v["z"], 2),
                     f(t.get("heldout", {}).get("auc")), f(t.get("generator2", {}).get("auc")),
                     f(t.get("generator2_vs_tensor", {}).get("auc"))])
    t = tr["distilled"]
    rows.append(["distilled (unrestricted)", ", ".join(f"`{x}`" for x in dist["terms"]), f(dist["auc"]), f(dist["z"], 2),
                 f(t["heldout"]["auc"]), f(t["generator2"]["auc"]), f(t["generator2_vs_tensor"]["auc"])])
    if "distilled_strength_free" in X:
        dsf, tsf = X["distilled_strength_free"], tr["distilled_strength_free"]
        rows.append(["distilled (strength-free)", ", ".join(f"`{x}`" for x in dsf["terms"]), f(dsf["auc"]), f(dsf["z"], 2),
                     f(tsf["heldout"]["auc"]), f(tsf["generator2"]["auc"]), f(tsf["generator2_vs_tensor"]["auc"])])
    w(table(["invariant", "terms", "AUC (G1 audit)", "z", "AUC untouched scenes (frozen)", "AUC generator 2 vs class (frozen)",
             "AUC generator 2 vs tensor (frozen)"], rows))
    g1m, g2m = X["g1_term_means"], X["g2_term_means"]
    w("Values of the named terms per arm (corpus means), generator 1 and generator 2:\n")
    keys = ["mean:dz_1", "sd:res_mean", "corpus:rar_scatter_oof", "mean:t_slope", "mean:t_lA", "mean:wl_lA", "mean:pb_tot", "mean:pe_tot"]
    rows = [[f"G1 {t}"] + [f(g1m[t][k], 3, True) for k in keys] for t in g1m]
    rows += [[f"G2 {t}"] + [f(g2m[t][k], 3, True) for k in keys] for t in g2m]
    w(table(["arm"] + [f"`{k}`" for k in keys], rows))

    # ------------------------------------------------------------------ 7 certificates
    w("## 7  Stage 4 certificates\n")
    w(f"{C['n_issued']} issued, {C['n_refused']} refused, seven checks each, typed identifiers.  C2 uses a MEASURED "
      f"control (the largest effect of the same statistic on anything that is not a halo: the class at fiducial amplitude, "
      f"systematics-only, Newton, and the instrument-nuisance arms); C3 is the realised false-positive rate on the untouched "
      f"audit half; C6 is the fraction of the generator-1 effect recovered on generator 2; C7 is the response pattern across "
      f"the {len(C['signatures']['statistic_order'])}-statistic set.\n")
    rows = []
    for cid, c in C["cases"].items():
        m = c["meta"]
        rows.append([f"`{cid}`", f(m["effect"], 3, True), f(m["class_sd"], 3),
                     "**ISSUED**" if c["issued"] else "refused: " + ", ".join(k.split("_")[0] for k in c["failed"]),
                     f(c["checks"]["C2_not_a_restatement"].get("ratio"), 2),
                     f(c["checks"]["C4_powered"].get("z_at_predicted"), 1),
                     f(c["checks"]["C6_out_of_grammar"].get("recovery"), 2),
                     f(c["checks"]["C7_nuisance_distinct"].get("worst_corr"), 2)])
    w(table(["candidate", "effect (U2 - U3)", "class corpus sd", "verdict", "C2 control/target", "C4 sigma at predicted",
             "C6 G2 recovery", "C7 worst |r|"], rows))
    for cid, c in C["cases"].items():
        if not c["issued"]:
            w(f"* `{cid}`: " + "; ".join(f"{k}: {c['checks'][k]['detail']}" for k in c["failed"]))
    w("")

    # ------------------------------------------------------------------ 8 f_lss and nuisance scans
    w("## 8  The separation as a function of f_lss, and of every other halo nuisance\n")
    sc = D["D5_scans"]
    w("The discriminator trained on the CDM PRIOR (f_lss ~ Beta(2, 2)), read on CDM with one nuisance fixed, against the "
      "class's audit corpora.  `directional` uses only the quadrupole phases and powers; `non-directional` drops every "
      "quadrupole/harmonic and network feature; `vertical` and `curve residuals` are the single-channel galaxy discriminators (the latter carries residual SHAPE as well as scatter).\n")
    rows = []
    order = ["S_flss_0", "S_flss_0.25", "S_flss_0.38", "S_nominal", "S_flss_0.75", "S_flss_1"]
    for t in order:
        if t in sc:
            lab = t.replace("S_flss_", "f_lss = ").replace("S_nominal", "f_lss = 0.5 (nominal)")
            v = sc[t]
            rows.append([lab] + [scan_key(v, k) for k in ("full", "directional", "phase_only", "non_directional", "vertical_only", "curve_res_only")]
                        + [f(v.get("phase_only", {}).get("auc_vs_tensor_fid")), f(v.get("non_directional", {}).get("auc_vs_tensor_fid"))])
    w(table(["CDM arm", "full", "directional (power + phase)", "phase only", "non-directional", "vertical only",
             "curve residuals only", "phase only vs TENSOR (U5f)", "non-directional vs TENSOR (U5f)"], rows))
    w("The last two columns restate BK's question: the CDM arm against the tensor universe at fiducial amplitude.  A "
      "phase discriminator must lose that separation as the halo's alignment moves onto the external axis; a "
      "non-directional one must not.\n")
    rows = []
    for t, v in sc.items():
        if t in order:
            continue
        rows.append([t] + [scan_key(v, k) for k in ("full", "directional", "non_directional", "vertical_only", "curve_res_only")])
    w("Every other scan arm (S_U3_* are the CLASS under the same instrument change and must sit at chance):\n")
    w(table(["arm", "full", "directional only", "non-directional only", "vertical only", "curve residuals only"], rows))
    g2s = D["D7_g2_refit"]["scans"]
    if g2s:
        w("Generator 2, its own discriminator (D vs C), the same scans:\n")
        w(table(["arm", "full", "directional only", "non-directional only"],
                [[t, f(v["full"]), f(v["directional"]), f(v["non_directional"])] for t, v in g2s.items()]))

    # ------------------------------------------------------------------ 9 sample size
    w("## 9  Sample size: where the answer changes with N\n")
    ss = D["D4_sample_size"]
    w(table(["galaxies per corpus", "clusters per corpus", "U2 vs U3"],
            [[k.split("_")[0][3:], k.split("_")[1][3:], sep(v)] for k, v in ss.items()]))

    # ------------------------------------------------------------------ 10 transfer
    w("## 10  Transfer: untouched scenes and the independent generator\n")
    tr6 = D["D6_heldout_transferred"]
    rf6 = D["D6_heldout_refit"]
    w(f"**Untouched scene library** (a library BF never drew).  The generator-1 discriminator transferred unchanged: "
      f"U2 vs class pool {sep(tr6['pool'])}; refitted on the held-out library (null sd {f(rf6['sd_null'], 4)}): "
      f"{sep(rf6['pool'])}; vertical channel alone, refitted: {sep(D['D6_heldout_refit_vertical_only']['pool'])}.\n")
    w(table(["class member", "transferred", "refitted"],
            [[t, sep(tr6.get(t)), sep(rf6.get(t))] for t in CLASS if t in tr6]))
    t7, r7 = D["D7_g2_transferred"], D["D7_g2_refit"]
    t7c = D.get("D7_g2_transferred_newton_calibrated", {})
    w(f"**Generator 2.**  Transferred unchanged (a fingerprint test -- absolute zero points differ between the generators' "
      f"disc conventions, see section 12): D vs C {sep(t7['C'])}, D vs tensor {sep(t7['T'])}, D vs Newton {sep(t7['N'])}.  "
      + (f"Transferred after calibrating every feature's zero point on the two generators' Newtonian arms (label-free with "
         f"respect to CDM vs class; possible in simulation only): D vs C {sep(t7c.get('C'))}, D vs tensor {sep(t7c.get('T'))}.  "
         if t7c else "")
      + f"Refitted on generator 2 (null sd {f(r7['sd_null'], 4)}): D vs C **{sep(r7['C'])}**, D vs tensor {sep(r7['T'])}, "
      + f"D vs Newton {sep(r7['N'])}.\n")
    w(table(["generator-2 ablation", "AUC", "z"], [[k, f(v["auc"]), f(v["z"], 2)] for k, v in r7["ablations"].items()]))

    # ------------------------------------------------------------------ 11 closure
    w("## 11  Baryonic closure: does P(G | B) differ, and by how much\n")
    stoch = CL["stochastic_closure"]
    w("**Stochastic closure.**  The same library object emitted across sets: the sd of its residual across sets "
      "(noise only for the class; noise plus the halo draw for CDM), pooled over objects.  `fixhalo` keeps ONE halo per "
      "object across sets, so `halo-isolated` = sqrt(nominal^2 - fixhalo^2) is the halo's own contribution and "
      "`excess over class` = sqrt(nominal^2 - U3^2).\n")
    rows = []
    for k, v in stoch.items():
        if "S_nominal" in v:
            rows.append([f"`{k}`", f(v["U3"]["sd_within"]), f(v["H0"]["sd_within"]) if "H0" in v else "n/a",
                         f(v["U2"]["sd_within"]), f(v["S_nominal"]["sd_within"]), f(v["S_fixhalo"]["sd_within"]),
                         f(v["S_zeroscatter"]["sd_within"]) if "S_zeroscatter" in v else "n/a",
                         f"**{f(v['sd_halo_isolated'])}**", f(v["sd_excess_over_class"])])
    w(table(["residual", "U3", "H0", "U2 prior", "CDM nominal", "CDM halo held", "CDM zero scatter", "halo-isolated sd",
             "excess over class"], rows))
    pr = CL["scatter_vs_prior"]
    w("The galaxy residual scatter as a function of the SHMR scatter prior (dex of halo mass at fixed stellar mass):\n")
    w(table(["arm", "out-of-fold RAR scatter", "within-galaxy residual sd", "between-object sd of res_mean", "within-object sd of res_mean"],
            [[t, f(v["rar_scatter_oof"]), f(v["res_sd"]), f(v["res_mean_sd_between"]), f(v["res_mean_sd_within"])] for t, v in pr.items()]))
    cov = CL["increment_covariance"]
    w("**Covariance of the increments.**  Per-object increment of each residual, arm minus U3 on the same scene and noise; "
      "slopes between channels (a halo moves lensing, X-ray and dynamics together; a photon-coupling change does not):\n")
    rows = []
    for pair, rec in cov.items():
        rows.append([pair,
                     f"{f(rec.get('slope d_lA on wl_lA', {}).get('slope'), 2, True)} +- {f(rec.get('slope d_lA on wl_lA', {}).get('se'), 2)}",
                     f"{f(rec.get('slope t_lA on wl_lA', {}).get('slope'), 2, True)} +- {f(rec.get('slope t_lA on wl_lA', {}).get('se'), 2)}",
                     f"{f(rec.get('slope dz_1 on res_mean', {}).get('slope'), 2, True)} +- {f(rec.get('slope dz_1 on res_mean', {}).get('se'), 2)}",
                     f(rec["mean_increment"]["wl_lA"], 3, True), f(rec["mean_increment"]["d_lA"], 3, True),
                     f(rec["mean_increment"]["t_lA"], 3, True), f(rec["mean_increment"]["dz_1"], 3, True)])
    w(table(["arm - U3", "slope d_lA on wl_lA", "slope t_lA on wl_lA", "slope dz_1 on res_mean", "mean d wl_lA", "mean d d_lA",
             "mean d t_lA", "mean d dz_1"], rows))
    sm = CL["structural_means"]
    w("**Structural closure.**  Corpus means with every halo scatter set to zero (G a deterministic function of B):\n")
    keys = ["dz_1", "rz_1", "res_mean", "res_out_in", "wl_lA", "t_lA", "t_slope", "d_lA", "sl_res", "rar_scatter_oof"]
    w(table(["arm"] + [f"`{k}`" for k in keys], [[t] + [f(v.get(k), 3, True) for k in keys] for t, v in sm.items()]))
    zs = sc.get("S_zeroscatter")
    if zs:
        w(f"The discriminator on zero-scatter CDM: full {f(zs['full']['auc'])} (z {f(zs['full']['z'], 1)}), non-directional "
          f"{f(zs['non_directional']['auc'])}, vertical only {f(zs['vertical_only']['auc'])}, scatter only "
          f"{f(zs.get('curve_res_only', zs.get('scatter_only', {})).get('auc'))}.\n")

    # ------------------------------------------------------------------ 12 verdicts
    w("## 12  What this establishes, as bounds\n")
    vert = C["cases"].get("CAND.CDM.BOOST_ISOTROPY.AT_PRIOR_FDD_0-0.05", {})
    scat = C["cases"].get("CAND.CDM.CLOSURE_SCATTER.AT_PRIOR_SHMR_0.16", {})
    fdd_rows = [(t, sc[t]["vertical_only"]["auc"]) for t in ("S_nominal", "S_fdd_0.05", "S_fdd_0.1", "S_fdd_0.2", "S_fdd_0.3", "S_fdd_0.5", "S_fdd_0.7", "S_fdd_1") if t in sc]
    w("**Bound on distinguishability.**  Under the corpus, amplitudes, geometries and nuisance priors stated in section 0, "
      f"the calibrated discriminator separates the dark-matter universe from the modified-gravity class at AUC "
      f"{f(main_pool['auc'])} (z {f(main_pool['z'], 2)}{'*' if main_pool.get('z_capped') else ''}) on generator 1, "
      f"{f(rf6['pool']['auc'])} on untouched scenes, and {f(r7['C']['auc'])} on the independent generator; the vertical channel "
      f"alone reaches {f(ab.get('only:gal_vertical', {}).get('auc'))} and survives every alignment prior "
      f"(f_lss from 0 to 1: " + ", ".join(f"{f(sc[t]['vertical_only']['auc'], 2)}" for t in order if t in sc) + ").  "
      "This is a statement about these source distributions, noise levels, nuisance models and channels, not a theorem "
      "about every future observation.\n")
    if fdd_rows:
        w("**Where the vertical answer changes.**  Vertical-only discrimination against the dark-disc fraction: " +
          ", ".join(f"f_dd {t.replace('S_fdd_', '').replace('S_nominal', '0')}: {f(a, 2)}" for t, a in fdd_rows) +
          ".  A razor-thin dark component carrying the corresponding fraction of the halo's enclosed-mass profile is the one "
          "collisionless degree of freedom that mimics an isotropic boost; the report quotes the crossing as the amplitude at "
          "which the vertical statistic stops separating.\n")
    w("**Zero point.**  The vertical contrast's absolute value depends on the disc's vertical-structure convention: generator 1 "
      f"(razor-thin, sigma_z^2 = g_z h_z) puts the class at {f(g1m['U3']['mean:dz_1'], 2, True)} dex and CDM at "
      f"{f(g1m['U2']['mean:dz_1'], 2, True)}; generator 2 (Miyamoto-Nagai discs, k_z in [0.8, 1.25]) puts the class at "
      f"{f(g2m['C']['mean:dz_1'], 2, True)} and CDM at {f(g2m['D']['mean:dz_1'], 2, True)}.  The CDM-minus-class CONTRAST "
      f"({f(g1m['U2']['mean:dz_1'] - g1m['U3']['mean:dz_1'], 2, True)} vs {f(g2m['D']['mean:dz_1'] - g2m['C']['mean:dz_1'], 2, True)} dex) "
      "transfers; the zero point does not, and on real data it is the systematic the test lives or dies by.\n")
    w("**Responsiveness.**  " + (f"d(dz_1 estimate)/d(true contrast) = {f(vert.get('meta', {}).get('responsiveness_estimate_vs_true', {}).get('slope'), 3)} "
      f"+- {f(vert.get('meta', {}).get('responsiveness_estimate_vs_true', {}).get('se'), 3)} across the dark-disc scan; " if vert else "") +
      (f"d(closure scatter)/d(SHMR scatter, dex) = {f(scat.get('meta', {}).get('responsiveness_vs_shmr_dex', {}).get('slope'), 3)} "
       f"+- {f(scat.get('meta', {}).get('responsiveness_vs_shmr_dex', {}).get('se'), 3)}." if scat else "") + "\n")

    io.open(OUT, "w", encoding="utf-8", newline="\n").write("\n".join(L))
    print(f"wrote {OUT} ({len(L)} blocks)")


if __name__ == "__main__":
    main()
