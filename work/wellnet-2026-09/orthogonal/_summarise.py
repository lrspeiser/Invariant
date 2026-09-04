"""Merge every stage into orthogonal_results.json and print the report tables."""
import json, os, glob
import numpy as np

R = json.load(open("orthogonal_results.json"))
for f, key in (("_side_measure.json", "measure"),
               ("_side_secondary2.json", "secondary"),
               ("_side_sys.json", "systematics"),
               ("_extras.json", None)):
    if not os.path.exists(f):
        print("MISSING", f); continue
    d = json.load(open(f))
    R.update(d) if key is None else R.update({key: d[key] if key in d else d})
if os.path.exists("_side_secondary.json"):
    R["nulls"] = json.load(open("_side_secondary.json"))["nulls"]
if os.path.exists("_side_null2.json"):
    R["nulls"]["decomposition"] = json.load(open("_side_null2.json"))
if os.path.exists("_warps_out.json"):
    R["warps"] = json.load(open("_warps_out.json"))

# ---- null calibration, pooled
n = R.get("nulls", {})
if n:
    cal = np.array([r["lam_hat"] for r in n["cal"]])
    aud = np.array([r["lam_hat"] for r in n["audit"]])
    pool = np.concatenate([cal, aud])
    from scipy import stats
    lo, hi = cal.mean() - 1.96 * cal.std(ddof=1), cal.mean() + 1.96 * cal.std(ddof=1)
    n["pooled"] = dict(mean=float(pool.mean()), sd=float(pool.std(ddof=1)),
                       n=len(pool),
                       cal_mean=float(cal.mean()), cal_sd=float(cal.std(ddof=1)),
                       audit_mean=float(aud.mean()), audit_sd=float(aud.std(ddof=1)),
                       ks_p=float(stats.ks_2samp(cal, aud).pvalue),
                       t_p=float(stats.ttest_ind(cal, aud).pvalue),
                       gaussian_cal95=[float(lo), float(hi)],
                       audit_outside_gaussian95=float(np.mean((aud < lo) | (aud > hi))),
                       audit_outside_empirical95=n["audit_outside_cal_95"])
    pw = {}
    for k, v in n["injections"].items():
        lh = np.array([r["lam_hat"] for r in v])
        pw[k] = dict(mean=float(lh.mean()), sd=float(lh.std(ddof=1)),
                     bias=float(lh.mean() - float(k)),
                     shrink=float((lh.mean() - 1) / (float(k) - 1)),
                     z=float((lh.mean() - pool.mean()) / pool.std(ddof=1)))
    n["power"] = pw
    if "decomposition" in n:
        dd = {k: float(np.std([r["lam_hat"] for r in v], ddof=1))
              for k, v in n["decomposition"].items()}
        sh = n["pooled"]["sd"] ** 2 - dd.get("no_refit_6", 0) ** 2
        n["shared_quantity"] = dict(
            sd_with_rc_refit=n["pooled"]["sd"],
            sd_rc_frozen=dd.get("no_refit_6"),
            sd_12_streams=dd.get("refit_12"),
            shared_component=float(np.sqrt(max(sh, 0.0))),
            shared_fraction_of_variance=float(max(sh, 0.0) / n["pooled"]["sd"] ** 2))

# ---- measured A_dyn per law
if "measure" in R:
    tab = {}
    for law, m in R["measure"].items():
        L = np.array(sorted(float(x) for x in m["chi2_total"]))
        c = np.array([m["chi2_total"][str(x)] for x in L])
        A = np.array([m["A_dyn_of_lambda"][str(x)] for x in L])
        i = int(np.argmin(c))
        # parabolic sub-grid argmin
        if 0 < i < len(L) - 1:
            x0, x1, x2 = L[i - 1:i + 2]; y0, y1, y2 = c[i - 1:i + 2]
            den = (x0 - x1) * (x0 - x2) * (x1 - x2)
            a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / den
            b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / den
            lam = float(np.clip(-b / (2 * a), L[0], L[-1])) if a > 0 else float(L[i])
        else:
            lam = float(L[i])
        sd = n.get("pooled", {}).get("sd", np.nan)
        shr = np.median([v["shrink"] for v in n.get("power", {}).values()]) \
            if n.get("power") else 1.0
        lam_corr = 1.0 + (lam - 1.0) / shr
        tab[law] = dict(lam_argmin=lam, at_grid_edge=bool(i in (0, len(L) - 1)),
                        A_dyn_measured=float(np.interp(lam, L, A)),
                        A_dyn_shrinkcorr=float(np.interp(np.clip(lam_corr, L[0], L[-1]), L, A)),
                        A_dyn_at_lam1=float(np.interp(1.0, L, A)),
                        sigma_stat=float(sd),
                        dchi2_range=float(c.max() - c.min()),
                        A_dyn_responsiveness=[float(A.min()), float(A.max())],
                        n_streams=len(m["per_stream"]),
                        chi2=dict(zip([str(x) for x in L], c.tolist())))
    R["measured_table"] = tab
json.dump(R, open("orthogonal_results.json", "w"), indent=1)
print(json.dumps({k: R.get(k) for k in ("measured_table",)}, indent=1)[:4000])
if n:
    print("\nNULL:", json.dumps({k: n.get(k) for k in ("pooled", "power", "shared_quantity")}, indent=1))
