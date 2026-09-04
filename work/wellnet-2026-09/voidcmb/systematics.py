"""Supplementary systematic splits and the linearity check.

Every split is a SUBSET of the certified analysis, re-nulled inside its own
geometry, so each row carries its own null-calibrated significance rather than
borrowing the headline's.  Splits are declared here as a list before any of them
is run; none was chosen after seeing a value.

    python systematics.py
"""
from __future__ import annotations

import copy
import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify_voidcmb as CV                                   # noqa: E402
import estimator as E                                          # noqa: E402
import planckio as P                                           # noqa: E402


def subset(ctx, keep, label):
    c = copy.copy(ctx)
    c.pix = ctx.pix[keep]
    c.G = ctx.G[keep]
    c.n = int(keep.sum())
    c.map_t = {k: v[keep] for k, v in ctx.map_t.items()}
    c.map_t["dI_q"] = c.map_t["I_q"] - c.map_t["I_q"].mean()
    c.guard = P.BlindGuard(c.pix, max_overlap=0.05)
    c.guard.disarm(f"systematic split: {label}")
    return c


def main():
    t0 = time.time()
    ctx = CV.Context()
    ctx.guard.disarm("systematics: certificate already issued")
    b = np.degrees(np.arcsin(np.clip(ctx.G[:, 2], -1, 1)))
    l = np.degrees(np.arctan2(ctx.G[:, 1], ctx.G[:, 0])) % 360.0
    dust = ctx.sky_full["dust"][ctx.pix]
    edge = ctx.map_t["edge_deg"]

    splits = {
        "b_gt_30": b > 30.0,
        "b_gt_45": b > 45.0,
        "b_below_median": b <= np.median(b),
        "b_above_median": b > np.median(b),
        "l_below_median": l <= np.median(l),
        "l_above_median": l > np.median(l),
        "interior_edge_gt_15deg": edge > 15.0,
        "near_edge_le_15deg": edge <= 15.0,
        "dust_quiet_half": dust <= np.median(dust),
        "dust_loud_half": dust > np.median(dust),
        "fully_unmasked_only": ctx.frac[ctx.pix] >= 0.999,
    }
    rows = {}
    for name, keep in splits.items():
        if keep.sum() < 800:
            rows[name] = dict(skipped="fewer than 800 pixels", n=int(keep.sum()))
            continue
        c = subset(ctx, keep, name)
        bank, tried, dets = CV.rotation_bank(c, 600, seed=101)
        null = np.array([c.evaluate(R, E.HEADLINE)["c2c1"] for R in bank])
        true = c.evaluate(None, E.HEADLINE, what="split")
        mu, sd = float(null.mean()), float(null.std(ddof=1))
        rows[name] = dict(n=int(keep.sum()), c2_over_c1=float(true["c2c1"]),
                          null_mean=mu, null_sd=sd,
                          z=float((true["c2c1"] - mu) / sd),
                          limit_95=float(abs(true["c2c1"]) + 1.96 * sd),
                          n_rotations=len(bank))
        print(f"  {name:24s} n={keep.sum():5d}  c2/c1 = {true['c2c1']:+.5f} "
              f"({rows[name]['z']:+.2f} sigma)", flush=True)

    # ---- linearity: mean T in deciles of dI_q, monopole+dipole removed
    cols = dict(ctx.map_t)
    cols.update(ctx.sky_cols(ctx.pix, ctx.G))
    X0 = E.design(cols, [], ctx.G)                 # constant + dipole only
    T = ctx.T_full[ctx.pix]
    Tres = T - X0 @ np.linalg.lstsq(X0, T, rcond=None)[0]
    q = np.quantile(ctx.map_t["dI_q"], np.linspace(0, 1, 11))
    dec = []
    for i in range(10):
        m = (ctx.map_t["dI_q"] >= q[i]) & (ctx.map_t["dI_q"] <= q[i + 1])
        dec.append(dict(bin=i + 1, dI_q_mean=float(ctx.map_t["dI_q"][m].mean()),
                        T_mean_uK=float(Tres[m].mean()), n=int(m.sum())))
    xs = np.array([d["dI_q_mean"] for d in dec])
    ys = np.array([d["T_mean_uK"] for d in dec])
    lin = np.polyfit(xs, ys, 1)
    quad = np.polyfit(xs, ys, 2)
    doc = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               splits=rows, deciles=dec,
               decile_linear_slope_uK_per_mpch=float(lin[0]),
               decile_quadratic_curvature=float(quad[0]),
               seconds=time.time() - t0)
    p = os.path.join(HERE, "systematics.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, default=float))
    print(f"\nwrote {p}  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
