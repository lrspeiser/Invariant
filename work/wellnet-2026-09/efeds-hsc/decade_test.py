"""THE TEST: potential depth against RAW tangential shear, within one class.

The observable is the per-cluster, per-bin, metacalibration-response-corrected
tangential reduced shear measured by this lane from the DECADE per-source
catalogue (see acquire_decade.py).  It is not a mass, not an NFW fit, and not a
convergence map.  Its construction expression is

    g_t(theta) = [ sum_s w_s e_+,s ] / [ R_bar sum_s w_s ]

which contains galaxy shapes, weights and photometric redshifts and NOTHING
from the X-ray density fit.  That is the whole point: Run Z's hydrostatic
g_obs WAS the density log-slope, so the test variable and the observable were
the same quantity.  Here they cannot be.

MODELS, all pushed through the identical forward model
    M0  RAR
    M1  RAR x 10^A                          free amplitude == the CLASS STEP,
                                            which inside one class has no other
                                            content.  PRIMARY NULL.
    M2  RAR x 10^(A + beta x_Phi)           THE HYPOTHESIS
    M3  RAR x 10^A x (r/Mpc)^gamma          radius tilt      (competitor)
    M4  RAR(a0 -> f a0) x 10^A              free a0          (competitor)
    M5  RAR x 10^(A + b log M_b)            mass step        (competitor)
    M6  RAR x 10^(A + b log T)              temperature      (competitor)
    M7  RAR x 10^(A + b z)                  redshift         (competitor)
    M8  RAR x 10^(A + b f_gas)              gas fraction     (competitor)

BLIND PROTECTION, DECLARED BEFORE ANY RESIDUAL WAS EXAMINED
    the systems are sorted by eFEDS name and split alternately; even ranks are
    TRAIN, odd ranks are HELD OUT.  Every parameter is fitted on TRAIN, frozen,
    and the held-out set is scored once.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

import pipeline as P
import efeds_hsc as E

HERE = os.path.dirname(os.path.abspath(__file__))
MPC, MSUN, G = P.MPC, P.MSUN, P.G
PRIMARY_RULE = "fixed10Mpc"
RULES = ["fixed10Mpc", "fixed5Mpc", "fixed3Mpc", "2xR500", "10xrs"]
RES = {}


def hdr(s):
    print("\n" + "=" * 78 + f"\n{s}\n" + "=" * 78)


# ------------------------------------------------------------------- ingest
def load_profiles(path=None):
    path = path or os.path.join(HERE, "decade_efeds_shear_profiles.tsv")
    d = {}
    for ln in open(path, encoding="utf-8"):
        if ln.startswith("#") or ln.startswith("id\t"):
            continue
        p = ln.rstrip("\n").split("\t")
        if int(p[2]) < 0:
            continue
        v = [float(x) for x in p[3:]]
        if not np.isfinite(v[3]):                      # gt
            continue
        d.setdefault(p[0], []).append(dict(
            bin=int(p[2]), R=v[0] * MPC, theta=v[1], n=int(v[2]),
            gt=v[3], gx=v[4], err=v[5], R11=v[6], R22=v[7],
            beta=v[8], beta2=v[9]))
    return d


class Obs:
    """The measured data, matched to the systems that have an X-ray fit."""

    def __init__(self, recs, prof, min_bins=4, min_n=50):
        self.sys, self.rows = [], []
        for rc in recs:
            b = [x for x in prof.get(rc["id"], [])
                 if x["n"] >= min_n and x["err"] > 0 and x["beta"] > 0.02]
            if len(b) < min_bins:
                continue
            b.sort(key=lambda x: x["bin"])
            self.sys.append(rc)
            self.rows.append(b)
        self.R = [np.array([x["R"] for x in b]) for b in self.rows]
        self.gt = [np.array([x["gt"] for x in b]) for b in self.rows]
        self.gx = [np.array([x["gx"] for x in b]) for b in self.rows]
        self.er = [np.array([x["err"] for x in b]) for b in self.rows]
        self.bt = [np.array([x["beta"] for x in b]) for b in self.rows]
        self.b2 = [np.array([x["beta2"] for x in b]) for b in self.rows]
        self.scinv = [(4.0 * math.pi * G / P.CLIGHT ** 2)
                      * P.d_ang(rc["z"]) * bt
                      for rc, bt in zip(self.sys, self.bt)]

    def __len__(self):
        return len(self.sys)


# --------------------------------------------------------------- forward model
def shapes(systems, obs, idx, **kw):
    """Sigma and DeltaSigma at the measured radii, for one shape model."""
    out = []
    for k in idx:
        s = systems[k]
        g = s.g_pred(**kw)
        rt = s.trunc_for(kw.get("rule", PRIMARY_RULE), 20.0)
        S, dS = s.sigma_profile(g, obs.R[k], rt)
        out.append((S, dS))
    return out


def gplus(S, dS, scinv, b2, bt, c=1.0):
    """Reduced shear for a mass profile scaled by the constant c.  Exact."""
    kap = c * S * scinv
    gam = c * dS * scinv
    corr = 1.0 + kap * (b2 / np.maximum(bt, 1e-9) ** 2 - 1.0)
    return gam / np.maximum(1.0 - kap, 1e-3) * corr


def chi2(obs, idx, sh, logc):
    """chi^2 with a per-system multiplicative constant c_k = 10^logc[k]."""
    tot = 0.0
    for j, k in enumerate(idx):
        S, dS = sh[j]
        p = gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k], 10.0 ** logc[j])
        tot += float(np.sum(((p - obs.gt[k]) / obs.er[k]) ** 2))
    return tot


def fit_amp(obs, idx, sh, extra=None, grid=np.linspace(-1.0, 2.0, 121)):
    """Profile a single global amplitude, optionally plus a per-system term."""
    off = np.zeros(len(idx)) if extra is None else np.asarray(extra)
    c = [chi2(obs, idx, sh, off + a) for a in grid]
    i = int(np.argmin(c))
    return float(c[i]), float(grid[i])


# --------------------------------------------------------------------- driver
def main():
    hdr("DECADE x eFEDS -- potential depth against RAW tangential shear")
    recs, cuts = E.load_efeds()
    prof = load_profiles()
    obs = Obs(recs, prof)
    systems = [P.System(rc) for rc in obs.sys]
    n_pt = sum(len(r) for r in obs.rows)
    print(f"\n   {len(prof)} systems have a DECADE profile; {len(obs)} pass "
          f"the declared\n   quality cuts (>= 4 bins with >= 50 background "
          f"sources), {n_pt} points")
    z = np.array([s.z for s in systems])
    print(f"   redshift {z.min():.3f} - {z.max():.3f}, median {np.median(z):.3f}")

    # ---- NULL TESTS ON THE OBSERVABLE ITSELF -----------------------------
    hdr("N.  Null tests on the measured shear, before any model is fitted")
    ivw = [1.0 / e ** 2 for e in obs.er]
    W = np.concatenate(ivw)
    T = np.concatenate(obs.gt)
    X = np.concatenate(obs.gx)
    gt_m = float(np.sum(W * T) / np.sum(W))
    gx_m = float(np.sum(W * X) / np.sum(W))
    se = float(1.0 / math.sqrt(np.sum(W)))
    print(f"\n   N1 inverse-variance mean over all {len(T)} points")
    print(f"      tangential  g_t = {gt_m:+.5f} +- {se:.5f}   "
          f"({gt_m / se:+.1f} sigma)   <- must be POSITIVE, this validates the")
    print(f"                                            sign convention "
          f"empirically")
    print(f"      cross       g_x = {gx_m:+.5f} +- {se:.5f}   "
          f"({gx_m / se:+.1f} sigma)   <- must be consistent with ZERO")
    ok_sign = gt_m / se > 3
    ok_cross = abs(gx_m / se) < 3
    print(f"      -> sign convention {'VALIDATED' if ok_sign else 'FAILS'}; "
          f"B-mode {'clean' if ok_cross else 'CONTAMINATED'}")
    RES["N_nulls"] = {"gt_mean": gt_m, "gx_mean": gx_m, "sigma": se,
                      "gt_snr": gt_m / se, "gx_snr": gx_m / se,
                      "sign_validated": bool(ok_sign),
                      "bmode_clean": bool(ok_cross), "n_points": int(len(T))}
    if not ok_sign:
        raise SystemExit("no tangential signal detected; nothing to test")

    # per-bin stack, for the record
    print("\n   N2 stacked profile by radial bin")
    Rall = np.concatenate(obs.R) / MPC
    edges = np.geomspace(0.2, 3.5, 11) / P.H_LITTLE
    print(f"      {'R [Mpc]':>10s} {'n_pts':>6s} {'g_t':>10s} {'+-':>9s} "
          f"{'g_x':>10s} {'S/N':>6s}")
    stack = []
    for i in range(10):
        m = (Rall >= edges[i]) & (Rall < edges[i + 1])
        if m.sum() < 3:
            continue
        w = W[m]
        t = float(np.sum(w * T[m]) / np.sum(w))
        x = float(np.sum(w * X[m]) / np.sum(w))
        s = float(1.0 / math.sqrt(np.sum(w)))
        stack.append(dict(R=float(np.mean(Rall[m])), n=int(m.sum()), gt=t,
                          gx=x, err=s))
        print(f"      {np.mean(Rall[m]):10.3f} {m.sum():6d} {t:+10.5f} "
              f"{s:9.5f} {x:+10.5f} {t / s:6.2f}")
    tot = math.sqrt(sum((r["gt"] / r["err"]) ** 2 for r in stack))
    print(f"      total S/N of the DECADE stack {tot:.1f} "
          f"(HSC published stack: 29.0)")
    RES["N2_stack"] = stack
    RES["N2_total_SN"] = tot

    # N3 cluster-member contamination: an excess of "background" sources
    # towards the centre would dilute g_t and mimic a radial tilt
    print("\n   N3 background-source surface density versus radius "
          "(member-contamination test)")
    dens = np.zeros(10)
    area = np.zeros(10)
    for k in range(len(obs)):
        for r_, x in zip(obs.rows[k], obs.rows[k]):
            i = x["bin"]
            th = x["theta"]
            if not np.isfinite(th) or th <= 0:
                continue
            dens[i] += x["n"]
            area[i] += 2 * math.pi * th ** 2 * (math.log(3.5 / 0.2) / 10.0)
    sd = np.where(area > 0, dens / np.maximum(area, 1e-9), np.nan)
    ref = np.nanmedian(sd[5:])
    print(f"      {'bin':>4s} {'n_src':>8s} {'sigma [/arcmin^2]':>18s} "
          f"{'ratio to outer':>15s}")
    for i in range(10):
        if np.isfinite(sd[i]):
            print(f"      {i:4d} {int(dens[i]):8d} {sd[i]:18.3f} "
                  f"{sd[i] / ref:15.3f}")
    inner = float(np.nanmean(sd[:3] / ref))
    print(f"      inner three bins / outer five = {inner:.3f}  ->  "
          f"{'no member excess' if inner < 1.15 else 'CONTAMINATION'}")
    RES["N3_source_density"] = {"per_bin": [float(x) for x in sd],
                                "inner_over_outer": inner}

    # ---- the declared blind split ---------------------------------------
    order = np.argsort([s.id for s in systems])
    train = np.array(sorted(order[0::2]))
    held = np.array(sorted(order[1::2]))
    print(f"\n   declared split: {len(train)} TRAIN / {len(held)} HELD OUT, "
          f"by alternating eFEDS name rank")

    # ---- model comparison on TRAIN --------------------------------------
    hdr("M.  Model comparison on the training half, identical pipeline")
    logMb = np.array([math.log10(np.interp(s.R500, s.r, s.M_b) / MSUN)
                      for s in systems])
    logT = np.array([math.log10(max(s.T, 1e-3)) for s in systems])
    zz = np.array([s.z for s in systems])
    fgas = np.array([np.interp(s.R500, s.r, s.M_gas)
                     / (500.0 * (3.0 * (P.L.H0 * P.L.E(s.z)) ** 2
                                 / (8 * math.pi * G))
                        * 4.0 / 3.0 * math.pi * s.R500 ** 3)
                     for s in systems])
    covars = {"log M_b": logMb, "log T": logT, "redshift": zz,
              "f_gas": fgas}
    for k in covars:
        covars[k] = (covars[k] - covars[k].mean()) / covars[k].std()

    def ndof(idx):
        return sum(len(obs.rows[k]) for k in idx)

    results = {}
    sh0 = shapes(systems, obs, train)
    c0 = chi2(obs, train, sh0, np.zeros(len(train)))
    c1, a1 = fit_amp(obs, train, sh0)
    results["M0  RAR only"] = (0, c0, {})
    results["M1  + free amplitude (CLASS STEP)"] = (1, c1, {"A": a1})

    grid_b = np.linspace(-1.5, 2.0, 36)
    prof_b = []
    for b in grid_b:
        sh = shapes(systems, obs, train, beta_pd=b)
        prof_b.append(fit_amp(obs, train, sh)[0])
    prof_b = np.array(prof_b)
    ib = int(np.argmin(prof_b))
    beta_hat = float(grid_b[ib])
    results["M2  + beta x_Phi (HYPOTHESIS)"] = (2, float(prof_b[ib]),
                                                {"beta": beta_hat})

    # M2b: the part of potential depth that a function of (g_bar, r) cannot
    # reproduce.  The quadratic is fitted on the SHEAR-MEASURED radii of the
    # training half only, then frozen.
    lg, lr, xp = [], [], []
    for k in train:
        s = systems[k]
        lgk = np.log10(np.maximum(np.interp(obs.R[k], s.r, s.g_b), 1e-30))
        lrk = np.log10(obs.R[k] / MPC)
        d, _ = s.dphi(PRIMARY_RULE)
        lg.append(lgk)
        lr.append(lrk)
        xp.append(np.log10(np.interp(obs.R[k], s.r, d) / P.PHI0))
    lg, lr, xp = map(np.concatenate, (lg, lr, xp))
    Xd = np.column_stack([np.ones_like(lg), lg, lg ** 2, lr, lr ** 2, lg * lr])
    coef, *_ = np.linalg.lstsq(Xd, xp, rcond=None)
    r2 = 1 - np.var(xp - Xd @ coef) / np.var(xp)
    print(f"\n   x_Phi on a quadratic in (log g_b, log r), fitted on TRAIN: "
          f"R^2 = {r2:.4f},\n   residual {np.std(xp - Xd @ coef):.3f} dex.  "
          f"M2b uses only that residual.")
    for s in systems:
        s.set_xperp(coef)
    grid_p = np.linspace(-4.0, 4.0, 41)
    prof_p = np.array([fit_amp(obs, train,
                               shapes(systems, obs, train, beta_perp=b))[0]
                       for b in grid_p])
    ip = int(np.argmin(prof_p))
    results["M2b + beta x_Phi PERPENDICULAR to (g_b, r)"] = (
        2, float(prof_p[ip]), {"beta_perp": float(grid_p[ip])})
    RES["M2b_xphi_perp"] = {"quadratic_R2": float(r2),
                            "residual_dex": float(np.std(xp - Xd @ coef)),
                            "beta_perp": float(grid_p[ip]),
                            "grid": [float(x) for x in grid_p],
                            "chi2": [float(x) for x in prof_p]}

    grid_t = np.linspace(-1.5, 1.5, 31)
    prof_t = np.array([fit_amp(obs, train,
                               shapes(systems, obs, train, tilt=t))[0]
                       for t in grid_t])
    it = int(np.argmin(prof_t))
    results["M3  + gamma log r (competitor)"] = (2, float(prof_t[it]),
                                                 {"gamma": float(grid_t[it])})

    grid_a = np.geomspace(0.05, 100.0, 31)
    prof_a = np.array([fit_amp(obs, train,
                               shapes(systems, obs, train, a0_scale=x))[0]
                       for x in grid_a])
    ia = int(np.argmin(prof_a))
    results["M4  + free a0 (competitor)"] = (2, float(prof_a[ia]),
                                             {"a0_scale": float(grid_a[ia])})

    grid_c = np.linspace(-1.0, 1.0, 41)
    for name, v in covars.items():
        pr = np.array([fit_amp(obs, train, sh0, extra=c * v[train])[0]
                       for c in grid_c])
        i = int(np.argmin(pr))
        results[f"M   + {name} (competitor)"] = (2, float(pr[i]),
                                                 {"slope": float(grid_c[i])})

    nd = ndof(train)
    print(f"\n   {len(train)} systems, {nd} shear points")
    print(f"   {'model':40s} {'k':>2s} {'chi2':>9s} {'BIC':>9s} {'dBIC':>8s} "
          f"best")
    bics = {k: v[1] + v[0] * math.log(nd) for k, v in results.items()}
    bmin = min(bics.values())
    for k, (kk, c, par) in sorted(results.items(),
                                  key=lambda t: bics[t[0]]):
        ps = ", ".join(f"{a}={b:+.3f}" for a, b in par.items())
        print(f"   {k:40s} {kk:2d} {c:9.2f} {bics[k]:9.2f} "
              f"{bics[k] - bmin:+8.2f} {ps}")
    RES["M_model_comparison"] = {
        "n_train_systems": int(len(train)), "n_points": int(nd),
        "models": {k: {"k": v[0], "chi2": v[1], "bic": bics[k],
                       "dbic": bics[k] - bmin, "params": v[2]}
                   for k, v in results.items()},
        "beta_profile_grid": [float(x) for x in grid_b],
        "beta_profile_chi2": [float(x) for x in prof_b]}

    ok = grid_b[prof_b <= prof_b.min() + 1.0]
    ci = ([float(ok.min()), float(ok.max())] if ok.size else None)
    edge = ci is not None and (ci[0] <= grid_b[0] + 1e-9
                               or ci[1] >= grid_b[-1] - 1e-9)
    print(f"\n   beta = {beta_hat:+.3f}, 68% "
          + ("UNCONSTRAINED" if ci is None else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]")
          + ("  (grid edge)" if edge else ""))
    dchi = results["M1  + free amplitude (CLASS STEP)"][1] \
        - results["M2  + beta x_Phi (HYPOTHESIS)"][1]
    print(f"   improvement over the class-step null: dchi2 = {dchi:.2f} "
          f"for 1 parameter  ({math.sqrt(max(dchi, 0)):.2f} sigma)")
    RES["M_beta"] = {"beta": beta_hat, "ci68": ci, "hits_edge": bool(edge),
                     "dchi2_vs_class_step": dchi}

    # ---- FROZEN TRANSFER -------------------------------------------------
    hdr("H.  FROZEN transfer onto the held-out half, evaluated once")
    print("\n   Every shape parameter is frozen at its TRAIN value.  Only the\n"
          "   single amplitude is re-fitted on the held-out half, because an\n"
          "   amplitude is what the class-step null already grants every "
          "model.")
    rows = []
    for name, kw in [("M1  class step", {}),
                     ("M2  beta x_Phi", dict(beta_pd=beta_hat)),
                     ("M3  gamma log r",
                      dict(tilt=results["M3  + gamma log r (competitor)"][2]
                           ["gamma"])),
                     ("M4  free a0",
                      dict(a0_scale=results["M4  + free a0 (competitor)"][2]
                           ["a0_scale"]))]:
        sh = shapes(systems, obs, held, **kw)
        c, a = fit_amp(obs, held, sh)
        rows.append((name, c, a))
    ndh = ndof(held)
    base = rows[0][1]
    print(f"\n   {len(held)} systems, {ndh} points")
    print(f"   {'model':22s} {'chi2':>9s} {'chi2/dof':>9s} {'dchi2 vs M1':>12s}")
    for name, c, a in rows:
        print(f"   {name:22s} {c:9.2f} {c / ndh:9.4f} {base - c:+12.2f}")
    RES["H_frozen_transfer"] = {
        "n_held_systems": int(len(held)), "n_points": int(ndh),
        "rows": [{"model": n, "chi2": c, "chi2_per_dof": c / ndh,
                  "dchi2_vs_class_step": base - c, "amplitude": a}
                 for n, c, a in rows]}

    # ---- the question in its literal form --------------------------------
    section_Q(systems, obs)
    # ---- sensitivity to the boundary rule --------------------------------
    section_S(systems, obs, train, grid_b)
    # ---- shared-quantity null with the real error covariance --------------
    section_D(systems, obs, train, obs.sys)
    # ---- responsiveness ---------------------------------------------------
    section_R(systems, obs, train, grid_b)
    # ---- random-point null ------------------------------------------------
    section_RND()

    with open(os.path.join(HERE, "decade_results.json"), "w",
              encoding="utf-8") as f:
        json.dump(RES, f, indent=1)
    print("\n   wrote decade_results.json")


# ------------------------------------------------------------------ sections
def section_Q(systems, obs):
    hdr("Q.  The question in its literal form: at matched g_b, does the "
        "lensing\n    residual vary with DeltaPhi_b?")
    idx = np.arange(len(obs))
    sh = shapes(systems, obs, idx)
    xs, ys, ws, gb, ids = [], [], [], [], []
    for j, k in enumerate(idx):
        S, dS = sh[j]
        p = gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k])
        s = systems[k]
        g_b = np.interp(obs.R[k], s.r, s.g_b)
        d, _ = s.dphi(PRIMARY_RULE)
        xphi = np.log10(np.interp(obs.R[k], s.r, d) / P.PHI0)
        m = (p > 0) & np.isfinite(xphi)
        ys.append((obs.gt[k] / p)[m])                 # ratio, not a log
        ws.append((p / obs.er[k])[m] ** 2)            # weight on the ratio
        xs.append(xphi[m])
        gb.append(np.log10(g_b[m]))
        ids += [k] * int(m.sum())
    y = np.concatenate(ys)
    w = np.concatenate(ws)
    x = np.concatenate(xs)
    g = np.concatenate(gb)
    ids = np.array(ids)
    print(f"\n   {len(y)} points.  y = g_t(observed) / g_t(RAR predicted); "
          f"a value\n   above 1 means baryons plus the RAR UNDERPREDICT the "
          f"shear.")
    print(f"   weighted mean y = "
          f"{np.sum(w * y) / np.sum(w):.3f} "
          f"+- {1 / math.sqrt(np.sum(w)):.3f}")
    bins = np.quantile(g, np.linspace(0, 1, 5))
    print(f"\n   {'log g_b bin':>22s} {'n':>6s} {'<x_Phi>':>9s} "
          f"{'sd(x_Phi)':>10s} {'slope dy/dx_Phi':>16s}")
    out = []
    for i in range(4):
        m = (g >= bins[i]) & (g <= bins[i + 1])
        if m.sum() < 30:
            continue
        xx, yy, ww = x[m], y[m], w[m]
        A_ = np.vstack([np.ones_like(xx), xx - xx.mean()]).T
        Aw = A_ * ww[:, None]
        cof = np.linalg.solve(A_.T @ Aw, Aw.T @ yy)
        # system-level bootstrap for the slope
        u = np.unique(ids[m])
        bs = []
        rng = np.random.default_rng(11)
        for _ in range(300):
            pick = rng.choice(u, len(u), replace=True)
            sel = np.concatenate([np.where(ids[m] == q)[0] for q in pick])
            a2 = np.vstack([np.ones(len(sel)), xx[sel] - xx.mean()]).T
            aw = a2 * ww[sel][:, None]
            try:
                bs.append(np.linalg.solve(a2.T @ aw, aw.T @ yy[sel])[1])
            except np.linalg.LinAlgError:
                pass
        sd = float(np.std(bs)) if bs else float("nan")
        print(f"   [{bins[i]:+.2f},{bins[i + 1]:+.2f}] {m.sum():6d} "
              f"{xx.mean():9.3f} {xx.std():10.3f} "
              f"{cof[1]:+10.3f} +- {sd:.3f}")
        out.append({"logg_lo": float(bins[i]), "logg_hi": float(bins[i + 1]),
                    "n": int(m.sum()), "mean_xPhi": float(xx.mean()),
                    "sd_xPhi": float(xx.std()), "slope": float(cof[1]),
                    "slope_sd": sd})
    RES["Q_literal"] = {"weighted_mean_ratio":
                        float(np.sum(w * y) / np.sum(w)),
                        "sem": float(1 / math.sqrt(np.sum(w))),
                        "bins": out}


def section_S(systems, obs, train, grid_b):
    hdr("S.  Sensitivity: boundary rule, radial range, stellar baryons")
    gb = grid_b[::2]
    out = {}
    print("\n   S1 boundary rule.  Run Z: the rule DEFINES the variable.")
    for rule in RULES:
        pr = np.array([fit_amp(obs, train,
                               shapes(systems, obs, train, beta_pd=b,
                                      rule=rule))[0] for b in gb])
        i = int(np.argmin(pr))
        ok = gb[pr <= pr.min() + 1.0]
        ci = [float(ok.min()), float(ok.max())] if ok.size else None
        out.setdefault("boundary_rule", {})[rule] = {
            "beta": float(gb[i]), "ci68": ci, "chi2_min": float(pr.min())}
        print(f"       {rule:12s} beta = {gb[i]:+.3f}  68% "
              + ("UNCONSTRAINED" if ci is None
                 else f"[{ci[0]:+.3f}, {ci[1]:+.3f}]"))

    # radial range: beyond ~1.5 Mpc the correlated two-halo term contributes
    # real shear that a one-halo baryon model does not carry, and it is a
    # smooth outward-rising term -- exactly the shape a radius tilt or a
    # potential-depth factor can absorb.
    print("\n   S2 radial range (the two-halo term this model does not carry\n"
          "      lives at large R, so a cut there is the honest check)")
    shs = [shapes(systems, obs, train, beta_pd=b) for b in gb]
    for lab, rmax in (("all radii", 1e9), ("R < 2.0 Mpc", 2.0),
                      ("R < 1.5 Mpc", 1.5), ("R > 1.5 Mpc", -1.5)):
        msk = {}
        for k in train:
            msk[k] = ((obs.R[k] / MPC < rmax) if rmax > 0
                      else (obs.R[k] / MPC > -rmax))
        pr = []
        for s in shs:
            best = None
            for a in np.linspace(-1.0, 2.0, 61):
                c = 0.0
                for j, k in enumerate(train):
                    S, dS = s[j]
                    m = msk[k]
                    if not np.any(m):
                        continue
                    p = gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k],
                              10 ** a)
                    c += float(np.sum(((p - obs.gt[k]) / obs.er[k])[m] ** 2))
                best = c if best is None else min(best, c)
            pr.append(best)
        pr = np.array(pr)
        i = int(np.argmin(pr))
        npts = sum(int(msk[k].sum()) for k in train)
        out.setdefault("radial_range", {})[lab] = {"beta": float(gb[i]),
                                                   "n_points": npts}
        print(f"       {lab:14s} beta = {gb[i]:+.3f}   ({npts} points)")

    print("\n   S3 stellar baryons (the eFEDS tables carry no stellar mass, "
          "so the\n      primary M_b is gas only; these are uniform rescalings)")
    for fs in (0.0, 0.15, 0.30):
        alt = [P.System(rc, f_star=fs) for rc in obs.sys]
        pr = np.array([fit_amp(obs, train,
                               shapes(alt, obs, train, beta_pd=b))[0]
                       for b in gb])
        i = int(np.argmin(pr))
        out.setdefault("f_star", {})[str(fs)] = {"beta": float(gb[i])}
        print(f"       f_star = {fs:4.2f}   beta = {gb[i]:+.3f}")
    RES["S_sensitivity"] = out


def section_D(systems, obs, train, recs, n_mc=60):
    hdr("D.  Shared-quantity null, with the ACTUAL error covariance")
    print("""
   The construction expressions share no input quantity:

       x_Phi(r)   = f(n0^2, r_s, alpha, beta_V, epsilon, z)   X-ray fit only
       g_t(theta) = [sum_s w_s e_+,s] / [R_bar sum_s w_s]     shapes only

   Run Z's hydrostatic g_obs WAS the density log-slope, so its estimator was
   guaranteed to find something.  This Monte Carlo confirms the estimator is
   centred here: the X-ray parameters are perturbed within their published
   errors, moving g_b and x_Phi coherently, while the shear is redrawn from
   its own shape noise.""")
    db = 0.15
    rng = np.random.default_rng(20260904)
    sh0 = shapes(systems, obs, train)
    _, a0 = fit_amp(obs, train, sh0)

    def lin_beta(shA, shB, ys):
        num = den = 0.0
        for j, k in enumerate(train):
            SA, dA = shA[j]
            SB, dB = shB[j]
            pA = gplus(SA, dA, obs.scinv[k], obs.b2[k], obs.bt[k], 10 ** a0)
            pB = gplus(SB, dB, obs.scinv[k], obs.b2[k], obs.bt[k], 10 ** a0)
            d = (pB - pA) / db
            iv = 1.0 / obs.er[k] ** 2
            num += float(np.sum(d * (ys[j] - pA) * iv))
            den += float(np.sum(d * d * iv))
        return num / den

    shD = shapes(systems, obs, train, beta_pd=db)
    p_true = []
    for j, k in enumerate(train):
        S, dS = sh0[j]
        p_true.append(gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k],
                            10 ** a0))
    lin_on_data = lin_beta(sh0, shD, [obs.gt[k] for k in train])
    print(f"   linearised estimator on the real data: beta = "
          f"{lin_on_data:+.4f}")
    bh = []
    for it in range(n_mc):
        pert = []
        for k in train:
            rc = recs[k]
            pert.append(P.System(dict(
                rc, rs=max(rc["rs"] + rng.normal(0, rc["e_rs"]),
                           0.05 * rc["rs"]),
                n0sq=max(rc["n0sq"] + rng.normal(0, rc["e_n0sq"]),
                         1e-3 * rc["n0sq"]),
                eps=rc["eps"] + rng.normal(0, rc["e_eps"]),
                beta=max(rc["beta"] + rng.normal(0, rc["e_beta"]), 0.34),
                alpha=rc["alpha"] + rng.normal(0, rc["e_alpha"]))))
        loc = list(range(len(train)))
        shA = [pert[j].sigma_profile(
            pert[j].g_pred(), obs.R[k],
            pert[j].trunc_for(PRIMARY_RULE, 20.0))
            for j, k in zip(loc, train)]
        shB = [pert[j].sigma_profile(
            pert[j].g_pred(beta_pd=db), obs.R[k],
            pert[j].trunc_for(PRIMARY_RULE, 20.0))
            for j, k in zip(loc, train)]
        ys = [p + rng.normal(0, obs.er[k])
              for p, k in zip(p_true, train)]
        bh.append(lin_beta(shA, shB, ys))
        if it % 15 == 0:
            print(f"      ... {it + 1}/{n_mc}  running mean "
                  f"{np.mean(bh):+.4f}")
    bh = np.array(bh)
    se = float(bh.std() / math.sqrt(len(bh)))
    print(f"\n   null expectation of beta-hat = {bh.mean():+.4f} +- {se:.4f} "
          f"(sd {bh.std():.4f}, n = {len(bh)})")
    print(f"   {bh.mean() / se:+.2f} sigma_MC from zero -> "
          f"{'no shared-quantity bias' if abs(bh.mean() / se) < 3 else 'BIAS'}")
    RES["D_shared_quantity_null"] = {
        "n_mc": int(n_mc), "mean": float(bh.mean()), "sem": se,
        "sd": float(bh.std()), "z": float(bh.mean() / se),
        "p2.5": float(np.percentile(bh, 2.5)),
        "p97.5": float(np.percentile(bh, 97.5)),
        "linear_estimator_on_data": float(lin_on_data)}


def section_R(systems, obs, train, grid_b, n_real=25):
    hdr("R.  Responsiveness: d beta-hat / d beta_injected must not be zero")
    gb = grid_b[::2]
    shs = [shapes(systems, obs, train, beta_pd=b) for b in gb]
    amps = [fit_amp(obs, train, s)[1] for s in shs]
    rng = np.random.default_rng(3)
    rec = []
    for bt in (-0.5, -0.25, 0.0, 0.25, 0.5):
        sh = shapes(systems, obs, train, beta_pd=bt)
        _, a = fit_amp(obs, train, sh)
        base = []
        for j, k in enumerate(train):
            S, dS = sh[j]
            base.append(gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k],
                              10 ** a))
        hats = []
        for _ in range(n_real):
            ys = [p + rng.normal(0, obs.er[k]) for p, k in zip(base, train)]
            best, bb = None, None
            for gi, s in enumerate(shs):
                c = 0.0
                for j, k in enumerate(train):
                    S, dS = s[j]
                    p = gplus(S, dS, obs.scinv[k], obs.b2[k], obs.bt[k],
                              10 ** amps[gi])
                    c += float(np.sum(((p - ys[j]) / obs.er[k]) ** 2))
                if best is None or c < best:
                    best, bb = c, gb[gi]
            hats.append(bb)
        rec.append((bt, float(np.mean(hats)), float(np.std(hats))))
        print(f"       beta_inj {bt:+.2f} -> beta_hat {rec[-1][1]:+.4f} "
              f"+- {rec[-1][2]:.4f}")
    x = np.array([r[0] for r in rec])
    y = np.array([r[1] for r in rec])
    sl = float(np.polyfit(x, y, 1)[0])
    print(f"\n   d beta-hat/d beta_inj = {sl:.4f}, spread "
          f"{y.max() - y.min():.4f}  -> "
          f"{'PASS' if abs(sl) > 0.5 else 'FAIL, statistic is degenerate'}")
    print(f"   per-realisation sd {np.mean([r[2] for r in rec]):.4f} is the "
          f"achievable sigma(beta)")
    RES["R_responsiveness"] = {
        "slope": sl, "spread": float(y.max() - y.min()),
        "sigma_beta": float(np.mean([r[2] for r in rec])),
        "points": [{"inj": a, "hat": b, "sd": c} for a, b, c in rec]}


def section_RND():
    hdr("X.  Random-point null on the measurement itself")
    p = os.path.join(HERE, "decade_random_shear_profiles.tsv")
    if not os.path.exists(p):
        print("\n   decade_random_shear_profiles.tsv not present -- run "
              "acquire_decade_random.py")
        RES["X_random_null"] = None
        return
    prof = load_profiles(p)
    T, X_, W = [], [], []
    for v in prof.values():
        for x in v:
            if x["n"] >= 50 and x["err"] > 0 and np.isfinite(x["gt"]):
                T.append(x["gt"])
                X_.append(x["gx"])
                W.append(1.0 / x["err"] ** 2)
    T, X_, W = np.array(T), np.array(X_), np.array(W)
    m = float(np.sum(W * T) / np.sum(W))
    mx = float(np.sum(W * X_) / np.sum(W))
    se = float(1 / math.sqrt(np.sum(W)))
    print(f"\n   {len(prof)} random positions, {len(T)} points")
    print(f"   tangential {m:+.5f} +- {se:.5f}  ({m / se:+.2f} sigma)")
    print(f"   cross      {mx:+.5f} +- {se:.5f}  ({mx / se:+.2f} sigma)")
    real = RES.get("N_nulls", {}).get("gt_mean")
    if real:
        print(f"   cluster signal {real:+.5f} is {real / max(abs(m), se):.1f}x "
              f"the random-point residual")
    print(f"   -> {'PASS' if abs(m / se) < 3 else 'FAIL: spurious signal'}")
    RES["X_random_null"] = {"n_positions": len(prof), "n_points": int(len(T)),
                            "gt": m, "gx": mx, "sigma": se,
                            "gt_sigma": m / se, "gx_sigma": mx / se,
                            "passed": bool(abs(m / se) < 3)}


if __name__ == "__main__":
    main()
