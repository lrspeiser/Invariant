"""run_closure.py -- the baryonic-closure test: does P(G | B) differ?

The candidate principle (BJ.6): under a universal law gravity has very low
residual freedom once the baryonic scene is given; a CDM universe carries a
halo whose mass, concentration, shape and orientation the baryons do not fix.
Tested three ways, on residuals from the corpus's own cross-fitted universal
law so that strength is matched away:

  stochastic closure   the same library object emitted many times: its residual
                       varies with the noise only (class), or with the noise
                       AND the halo draw (CDM).  The halo part is isolated by
                       the arm in which each object keeps ONE halo across sets.
  structural closure   CDM with every halo scatter set to zero: G is then a
                       deterministic function of B -- but not a LOCAL function
                       of g_bar, so residual shape survives.
  covariance           the per-object INCREMENT of each residual, CDM minus the
                       class on the same scene and noise: its covariance across
                       channels is the halo's joint imprint on lensing, X-ray
                       and dynamics.  Under the photon-slip universe the lensing
                       increment decouples from the dynamical one.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import gen1 as W                         # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run

GAL_RES = ["res_mean", "res_out_in", "res_slope", "rz_1", "dz_1", "res_6"]
CLU_RES = ["wl_lA", "t_lA", "h_lA", "d_lA", "wl_slope", "t_slope", "sl_res"]


def within_object_sd(stack, tag, grp, key):
    """Variance of a residual ACROSS SETS at fixed library object, averaged over
    objects (pooled within-scene variance), and the between-object variance."""
    d = stack[tag][grp]
    v, sc = np.asarray(d[key], float), np.asarray(d["scene"])
    out_w, out_n, means = [], [], []
    for s in np.unique(sc):
        x = v[sc == s]
        x = x[np.isfinite(x)]
        if len(x) >= 5:
            out_w.append(np.var(x, ddof=1))
            out_n.append(len(x))
            means.append(np.mean(x))
    if not out_w:
        return dict(sd_within=np.nan, sd_between=np.nan, n_obj=0)
    sw = float(np.sqrt(np.average(out_w, weights=out_n)))
    return dict(sd_within=sw, sd_between=float(np.std(means)), n_obj=len(out_w),
                mean=float(np.mean(means)))


def main():
    st = W.load_stack(os.path.join(RES, "G1_main.npz"))
    sc = W.load_stack(os.path.join(RES, "G1_scans.npz"))
    if os.path.exists(os.path.join(RES, "G1_scans_extra.npz")):
        sc.update(W.load_stack(os.path.join(RES, "G1_scans_extra.npz")))
    out = {}

    # ---------------- stochastic closure ----------------------------------
    stoch = {}
    for grp, keys in (("gal", GAL_RES), ("clu", CLU_RES)):
        for k in keys:
            row = {}
            for tag, src in (("U3", st), ("H0", st), ("U2", st), ("U5f", st), ("U8f", st),
                             ("S_nominal", sc), ("S_fixhalo", sc), ("S_zeroscatter", sc),
                             ("S_shmr_0", sc), ("S_shmr_2", sc), ("S_conc_0", sc), ("S_conc_2", sc)):
                if tag in src:
                    row[tag] = within_object_sd(src, tag, grp, k)
            # the halo's own contribution: nominal minus halo-held, in quadrature
            if "S_nominal" in row and "S_fixhalo" in row:
                a, b = row["S_nominal"]["sd_within"], row["S_fixhalo"]["sd_within"]
                row["sd_halo_isolated"] = float(np.sqrt(max(a * a - b * b, 0.0)))
                c = row["U3"]["sd_within"]
                row["sd_excess_over_class"] = float(np.sqrt(max(a * a - c * c, 0.0)))
            stoch[f"{grp}:{k}"] = row
    out["stochastic_closure"] = stoch

    # ---------------- covariance of increments --------------------------
    cov = {}
    for pair in (("U2", "U3"), ("U8f", "U3"), ("U5f", "U3"), ("U6f", "U3"), ("U7f", "U3"), ("U4f", "U3")):
        a, b = pair
        if a not in st or b not in st:
            continue
        rec = {}
        C = {k: np.asarray(st[a]["clu"][k], float) - np.asarray(st[b]["clu"][k], float) for k in ("wl_lA", "t_lA", "h_lA", "d_lA")}
        Gd = {k: np.asarray(st[a]["gal"][k], float) - np.asarray(st[b]["gal"][k], float) for k in ("res_mean", "dz_1", "rz_1")}
        for (k1, k2) in (("wl_lA", "d_lA"), ("wl_lA", "t_lA"), ("d_lA", "t_lA"), ("wl_lA", "h_lA")):
            m = np.isfinite(C[k1]) & np.isfinite(C[k2])
            if m.sum() > 30:
                rec[f"slope {k2} on {k1}"] = _slope(C[k1][m], C[k2][m])
                rec[f"corr {k1},{k2}"] = float(np.corrcoef(C[k1][m], C[k2][m])[0, 1])
        for (k1, k2) in (("res_mean", "dz_1"), ("res_mean", "rz_1")):
            m = np.isfinite(Gd[k1]) & np.isfinite(Gd[k2])
            if m.sum() > 30:
                rec[f"slope {k2} on {k1}"] = _slope(Gd[k1][m], Gd[k2][m])
        rec["mean_increment"] = {k: float(np.nanmean(v)) for k, v in {**C, **Gd}.items()}
        rec["sd_increment"] = {k: float(np.nanstd(v)) for k, v in {**C, **Gd}.items()}
        cov[f"{a}-{b}"] = rec
    out["increment_covariance"] = cov

    # ---------------- structural closure ------------------------------------
    # done by the discriminator on S_zeroscatter (D5); here the plain means
    struct = {}
    for tag, src in (("U3", st), ("U2", st), ("S_nominal", sc), ("S_zeroscatter", sc), ("S_fixhalo", sc)):
        if tag not in src:
            continue
        struct[tag] = {k: float(np.nanmean(src[tag]["gal"][k])) for k in GAL_RES}
        struct[tag].update({k: float(np.nanmean(src[tag]["clu"][k])) for k in CLU_RES})
        struct[tag]["rar_scatter_oof"] = float(np.nanmean(src[tag]["corpus"]["rar_scatter_oof"]))
    out["structural_means"] = struct

    # ---------------- the residual scatter as a function of the halo scatter prior
    prior = {}
    for tag in ("S_shmr_0", "S_shmr_0.5", "S_nominal", "S_shmr_1.5", "S_shmr_2"):
        if tag in sc:
            prior[tag] = dict(rar_scatter_oof=float(np.nanmean(sc[tag]["corpus"]["rar_scatter_oof"])),
                              res_sd=float(np.nanmean(sc[tag]["gal"]["res_sd"])),
                              res_mean_sd_between=within_object_sd(sc, tag, "gal", "res_mean")["sd_between"],
                              res_mean_sd_within=within_object_sd(sc, tag, "gal", "res_mean")["sd_within"])
    prior["U3"] = dict(rar_scatter_oof=float(np.nanmean(st["U3"]["corpus"]["rar_scatter_oof"])),
                       res_sd=float(np.nanmean(st["U3"]["gal"]["res_sd"])),
                       res_mean_sd_between=within_object_sd(st, "U3", "gal", "res_mean")["sd_between"],
                       res_mean_sd_within=within_object_sd(st, "U3", "gal", "res_mean")["sd_within"])
    out["scatter_vs_prior"] = prior

    with open(os.path.join(RES, "CL_closure.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    for k, v in stoch.items():
        if "S_nominal" in v:
            print(f"{k:<18} within: U3 {v['U3']['sd_within']:.4f}  U2 {v['U2']['sd_within']:.4f}  "
                  f"fixhalo {v['S_fixhalo']['sd_within']:.4f}  halo-isolated {v['sd_halo_isolated']:.4f}")
    print("wrote CL_closure.json")


def _slope(x, y):
    if len(x) < 4 or np.var(x) <= 1e-30 * max(np.mean(x * x), 1e-300):
        return dict(slope=np.nan, se=np.nan, n=int(len(x)), note="regressor has no variance (the arm does not move this residual)")
    A = np.stack([np.ones_like(x), x], 1)
    b, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ b
    s2 = float(r @ r) / max(len(x) - 2, 1)
    se = float(np.sqrt(s2 * np.linalg.inv(A.T @ A)[1, 1]))
    return dict(slope=float(b[1]), se=se, n=int(len(x)))


if __name__ == "__main__":
    main()
