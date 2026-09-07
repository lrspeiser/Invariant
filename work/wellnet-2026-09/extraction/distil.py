"""distil.py -- from the discriminator to 1-3 sparse, interpretable invariants.

Protocol item 6.  The discriminator locates the information; this module
asks how much of it a SMALL number of named corpus-level statistics carry.

Library of candidate invariants (each one number per corpus):
  * the corpus mean of every per-object residual feature ('mean:<f>')
  * the corpus standard deviation across objects of the level residuals
    ('sd:<f>') -- the object-to-object freedom of gravity at fixed baryons
  * a small symbolic grammar over pairs of library terms: a - b, a + b,
    a * b, a / b (guarded)

Search: single-term ranking by audit AUC; then greedy forward selection of up
to three terms into a linear score fitted on calibration corpora (a logistic
fit), scored on audit corpora; then the same on the untouched library and on
generator 2 with the terms FROZEN.  Every AUC is quoted with the calibrated
z of run_discriminate.  The fraction of the discriminator's separation
recovered is the headline of the distillation.
"""
from __future__ import annotations

import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import arms as AR                        # noqa: E402
import discriminator as D                # noqa: E402
import gen1 as W                         # noqa: E402
import invariants as IV                  # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run
CLASS = AR.CLASS_TAGS

MEAN_G = ["dz_0", "dz_1", "rz_0", "rz_1", "vz_1", "vr_1", "res_mean", "res_slope", "res_sd", "res_out_in",
          "res_1", "res_3", "res_5", "res_7", "p2_5", "p3_5"]
MEAN_C = ["wl_lA", "wl_slope", "t_lA", "t_slope", "h_lA", "h_slope", "y_lA", "d_lA", "d_slope", "sl_res",
          "sl_obs", "n_img", "kb_in", "q2_in", "q2_out", "q2_grad", "net_excess", "ep_ld", "ep_lt", "ep_dt",
          "wres_2", "wres_5", "tres_2", "tres_8", "hres_2", "hres_8", "dres_0", "dres_2"]
SD_G = ["res_mean", "dz_1", "rz_1", "res_out_in"]
SD_C = ["wl_lA", "t_lA", "h_lA", "d_lA", "wl_slope", "t_slope"]
PH_C = ["pb_tot", "pe_tot", "p45_tot", "pe_minus_pb"]
PH_G = ["g3_ext", "g3_45", "g3_disc"]


def corpus_library(stack, tag):
    """One row per corpus of the arm: the candidate invariants."""
    g, c = stack[tag]["gal"], stack[tag]["clu"]
    gp, cp = IV.galaxy_phase_features(g), IV.cluster_phase_features(c)
    sets = np.unique(g["set"])
    rows = {}
    for s in sets:
        mg, mc = g["set"] == s, c["set"] == s
        r = {}
        for k in MEAN_G:
            r[f"mean:{k}"] = float(np.nanmean(g[k][mg]))
        for k in PH_G:
            r[f"mean:{k}"] = float(np.nanmean(gp[k][mg]))
        for k in SD_G:
            r[f"sd:{k}"] = float(np.nanstd(g[k][mg]))
        for k in MEAN_C:
            r[f"mean:{k}"] = float(np.nanmean(c[k][mc]))
        for k in PH_C:
            r[f"mean:{k}"] = float(np.nanmean(cp[k][mc]))
        for k in SD_C:
            r[f"sd:{k}"] = float(np.nanstd(c[k][mc]))
        r["corpus:rar_scatter_oof"] = float(stack[tag]["corpus"]["rar_scatter_oof"][stack[tag]["corpus"]["set"] == s][0])
        rows[int(s)] = r
    return rows


def to_matrix(rows, names):
    ids = sorted(rows)
    X = np.array([[rows[i].get(n, np.nan) for n in names] for i in ids], float)
    return X, ids


def fit_logistic(X, y, l2=1e-2, iters=300):
    """Small ridge logistic regression by Newton steps (no sklearn dependence
    for a 1-3 term model; standardised inputs)."""
    mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-12
    Z = np.nan_to_num((X - mu) / sd)
    A = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        grad = A.T @ (p - y) + l2 * np.r_[0, w[1:]]
        Hm = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.diag(np.r_[0, np.ones(A.shape[1] - 1)])
        step = np.linalg.solve(Hm + 1e-9 * np.eye(len(w)), grad)
        w -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return dict(w=w, mu=mu, sd=sd)


def apply_logistic(model, X):
    Z = np.nan_to_num((X - model["mu"]) / model["sd"])
    return np.column_stack([np.ones(len(Z)), Z]) @ model["w"]


def evaluate(names, lib_pos, lib_neg_list, cal, aud, sd_null):
    """Fit on cal corpora (pos = 1, every neg arm = 0), AUC on audit."""
    Xp, ip = to_matrix(lib_pos, names)
    Xn, inn = [], []
    for L in lib_neg_list:
        X, ids = to_matrix(L, names)
        Xn.append(X)
        inn += ids
    Xn = np.vstack(Xn)
    mp = np.isin(ip, list(cal))
    mn = np.isin(inn, list(cal))
    model = fit_logistic(np.vstack([Xp[mp], Xn[mn]]), np.r_[np.ones(mp.sum()), np.zeros(mn.sum())])
    ap = np.isin(ip, list(aud))
    an = np.isin(inn, list(aud))
    sp, sn = apply_logistic(model, Xp[ap]), apply_logistic(model, Xn[an])
    a = D.auc(sn, sp)
    # d': the standardised separation of the score, which does not saturate
    # when the AUC does -- the tie-break for ranking and selection
    dprime = float((np.mean(sp) - np.mean(sn)) / np.sqrt(0.5 * (np.var(sp) + np.var(sn)) + 1e-30))
    return dict(auc=a, z=float(np.clip(abs(a - 0.5) / max(sd_null, 1e-9), 0, D.Z_CAP)), model=model,
                n_pos=int(ap.sum()), n_neg=int(an.sum()), dprime=dprime)


def _rank_key(r):
    return (round(r["auc"], 3), r.get("dprime", 0.0))


LEVEL_PAT = ("wres_", "hres_", "tres_", "dres_", "_lA", "kb_in", "sl_obs", "sl_res", "n_img", "mean:vz_",
             "mean:vr_", "mean:ly_", "mean:rz_", "mean:res_mean", "mean:res_1", "mean:res_3", "mean:res_5",
             "mean:res_7", "rar_scatter")


def is_level(name):
    """A term that carries the STRENGTH of gravity at some radius (the monopole
    excess a cluster missing-mass component would remove) rather than its
    structure, scatter or geometry."""
    return any(p in name for p in LEVEL_PAT)


def frozen_eval(model, names, lib_pos, lib_neg_list, sd_null):
    Xp, _ = to_matrix(lib_pos, names)
    Xn = np.vstack([to_matrix(L, names)[0] for L in lib_neg_list])
    sp, sn = apply_logistic(model, Xp), apply_logistic(model, Xn)
    a = D.auc(sn, sp)
    return dict(auc=a, z=float(np.clip(abs(a - 0.5) / max(sd_null, 1e-9), 0, D.Z_CAP)),
                mean_pos=float(np.mean(sp)), mean_neg=float(np.mean(sn)))


def main():
    st = W.load_stack(os.path.join(RES, "G1_main.npz"))
    Dj = json.load(open(os.path.join(RES, "D_discriminator.json")))
    sd_null = Dj["D0_sizing"]["sd_null_auc"]
    sets = np.unique(st["U2"]["corpus"]["set"])
    cal, aud = D.split_sets(sets, 0.5, seed=0)
    libs = {t: corpus_library(st, t) for t in ["U2"] + CLASS + ["U5f", "U8f", "U10", "U1"]}
    names = sorted(next(iter(libs["U2"].values())).keys())
    negs = [libs[t] for t in CLASS]
    out = {"sd_null": sd_null, "discriminator_auc": Dj["D1_main"]["pool"]["auc"],
           "discriminator_z": Dj["D1_main"]["pool"]["z"]}

    # single terms
    single = {}
    for n in names:
        r = evaluate([n], libs["U2"], negs, cal, aud, sd_null)
        single[n] = dict(auc=r["auc"], z=r["z"], dprime=r["dprime"], level=is_level(n),
                         sign=float(np.sign(r["model"]["w"][1])),
                         mean_U2=float(np.nanmean(to_matrix(libs["U2"], [n])[0])),
                         mean_U3=float(np.nanmean(to_matrix(libs["U3"], [n])[0])),
                         sd_U3=float(np.nanstd(to_matrix(libs["U3"], [n])[0])))
    ranked = sorted(single, key=lambda n: (-round(abs(single[n]["auc"] - 0.5), 3), -abs(single[n]["dprime"])))
    out["single_terms"] = {n: single[n] for n in ranked}
    print("top single terms:")
    for n in ranked[:12]:
        print(f"   {n:<28} auc {single[n]['auc']:.3f}  z {single[n]['z']:.2f}  U2 {single[n]['mean_U2']:+.3f}  U3 {single[n]['mean_U3']:+.3f} +- {single[n]['sd_U3']:.3f}")

    # pair grammar over the top 12 single terms
    top = ranked[:12]
    pairs = {}
    for a, b in itertools.combinations(top, 2):
        for op, fn in (("-", lambda x, y: x - y), ("+", lambda x, y: x + y), ("*", lambda x, y: x * y)):
            name = f"({a} {op} {b})"
            lp = {s: {name: fn(r[a], r[b])} for s, r in libs["U2"].items()}
            ln = [{s: {name: fn(r[a], r[b])} for s, r in L.items()} for L in negs]
            r = evaluate([name], lp, ln, cal, aud, sd_null)
            pairs[name] = dict(auc=r["auc"], z=r["z"])
    prank = sorted(pairs, key=lambda n: -abs(pairs[n]["auc"] - 0.5))
    out["pair_terms"] = {n: pairs[n] for n in prank[:20]}

    # greedy forward selection, up to 3 terms, on the standardised linear score;
    # ties in a saturated AUC are broken by d'.  Run twice: unrestricted, and
    # STRENGTH-FREE (no level term), which is the distillation the brief's
    # monopole-matching rule asks for.
    def greedy(pool_names, label):
        chosen, hist = [], []
        for step in range(3):
            best = None
            for n in pool_names:
                if n in chosen:
                    continue
                r = evaluate(chosen + [n], libs["U2"], negs, cal, aud, sd_null)
                if best is None or _rank_key(r) > _rank_key(best[1]):
                    best = (n, r)
            chosen.append(best[0])
            hist.append(dict(terms=list(chosen), auc=best[1]["auc"], z=best[1]["z"], dprime=best[1]["dprime"],
                             weights=best[1]["model"]["w"].tolist()))
            print(f"   {label} step {step + 1}: {chosen} auc {best[1]['auc']:.3f} z {best[1]['z']:.2f} d' {best[1]['dprime']:.2f}")
        return chosen, hist

    chosen, hist = greedy(ranked[:25], "unrestricted")
    out["greedy"] = hist
    chosen_sf, hist_sf = greedy([n for n in ranked if not is_level(n)][:25], "strength-free")
    out["greedy_strength_free"] = hist_sf
    final_sf = evaluate(chosen_sf, libs["U2"], negs, cal, aud, sd_null)
    out["distilled_strength_free"] = dict(terms=chosen_sf, auc=final_sf["auc"], z=final_sf["z"], dprime=final_sf["dprime"],
                                          weights=final_sf["model"]["w"].tolist(),
                                          per_member={t: frozen_eval(final_sf["model"], chosen_sf, libs["U2"], [libs[t]], sd_null)["auc"] for t in CLASS},
                                          fiducial_and_reference={t: frozen_eval(final_sf["model"], chosen_sf, libs["U2"], [libs[t]], sd_null)["auc"]
                                                                  for t in ("U5f", "U8f", "U10", "U1")})
    final = evaluate(chosen, libs["U2"], negs, cal, aud, sd_null)
    out["distilled"] = dict(terms=chosen, auc=final["auc"], z=final["z"], weights=final["model"]["w"].tolist(),
                            fraction_of_discriminator_auc_excess=float((final["auc"] - 0.5) / max(Dj["D1_main"]["pool"]["auc"] - 0.5, 1e-9)),
                            per_member={t: frozen_eval(final["model"], chosen, libs["U2"], [libs[t]], sd_null)["auc"] for t in CLASS},
                            fiducial_and_reference={t: frozen_eval(final["model"], chosen, libs["U2"], [libs[t]], sd_null)["auc"]
                                                    for t in ("U5f", "U8f", "U10", "U1")})
    # the three NAMED physical invariants, evaluated on their own and frozen
    named = {"boost_isotropy": ["mean:dz_1"],
             "closure_scatter": ["sd:res_mean"],
             "cluster_profile_freedom": ["mean:t_slope", "sd:t_lA"],
             "cluster_level_excess": ["mean:t_lA"],
             "directional_phase": ["mean:pb_tot", "mean:pe_tot"]}
    nm = {}
    for k, terms in named.items():
        r = evaluate(terms, libs["U2"], negs, cal, aud, sd_null)
        nm[k] = dict(terms=terms, auc=r["auc"], z=r["z"], weights=r["model"]["w"].tolist(), model=r["model"])
    out["named"] = {k: {kk: vv for kk, vv in v.items() if kk != "model"} for k, v in nm.items()}

    # frozen transfer: untouched scenes and generator 2
    ho = W.load_stack(os.path.join(RES, "G1_heldout.npz"))
    g2 = W.load_stack(os.path.join(RES, "G2_main.npz"))
    lho = {t: corpus_library(ho, t) for t in ["U2"] + [t for t in CLASS if t in ho]}
    lg2 = {t: corpus_library(g2, t) for t in ("C", "D", "T", "N")}
    sd_ho = Dj["D6_heldout_refit"]["sd_null"]
    sd_g2 = Dj["D7_g2_refit"]["sd_null"]
    tr = {"distilled": dict(heldout=frozen_eval(final["model"], chosen, lho["U2"], [lho[t] for t in CLASS if t in lho], sd_ho),
                            generator2=frozen_eval(final["model"], chosen, lg2["D"], [lg2["C"]], sd_g2),
                            generator2_vs_tensor=frozen_eval(final["model"], chosen, lg2["D"], [lg2["T"]], sd_g2)),
          "distilled_strength_free": dict(heldout=frozen_eval(final_sf["model"], chosen_sf, lho["U2"], [lho[t] for t in CLASS if t in lho], sd_ho),
                                          generator2=frozen_eval(final_sf["model"], chosen_sf, lg2["D"], [lg2["C"]], sd_g2),
                                          generator2_vs_tensor=frozen_eval(final_sf["model"], chosen_sf, lg2["D"], [lg2["T"]], sd_g2))}
    for k, v in nm.items():
        tr[k] = dict(heldout=frozen_eval(v["model"], v["terms"], lho["U2"], [lho[t] for t in CLASS if t in lho], sd_ho),
                     generator2=frozen_eval(v["model"], v["terms"], lg2["D"], [lg2["C"]], sd_g2),
                     generator2_vs_tensor=frozen_eval(v["model"], v["terms"], lg2["D"], [lg2["T"]], sd_g2))
    out["transfer_frozen"] = tr
    # generator-2 values of the named terms
    out["g2_term_means"] = {t: {n: float(np.nanmean(to_matrix(lg2[t], [n])[0])) for n in
                                ["mean:dz_1", "sd:res_mean", "mean:t_slope", "mean:t_lA", "mean:pb_tot", "mean:pe_tot",
                                 "mean:wl_lA", "mean:res_sd", "corpus:rar_scatter_oof"]} for t in ("C", "D", "T", "N")}
    out["g1_term_means"] = {t: {n: float(np.nanmean(to_matrix(libs[t], [n])[0])) for n in
                                ["mean:dz_1", "sd:res_mean", "mean:t_slope", "mean:t_lA", "mean:pb_tot", "mean:pe_tot",
                                 "mean:wl_lA", "mean:res_sd", "corpus:rar_scatter_oof"]} for t in ("U2", "U3", "H0", "U5f", "U8f", "U10", "U1")}
    with open(os.path.join(RES, "X_distil.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote X_distil.json")


if __name__ == "__main__":
    main()
