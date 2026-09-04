"""Shared machinery for lead01-ablation.

Reuses the ladder built in work/wellnet-2026-09/potential-depth/ verbatim and
reproduces, as a GATE, the exact system-level table that produced the published
numbers (252 systems, transfer 0.1066 / 0.0954 dex, beta 0.17188 / 0.16866).
Nothing downstream is trusted unless that gate passes.

The response variable, the window, the system aggregation and the free quadratic
in log g_bar are all taken unchanged from potential-depth/code/analyse.py sec 8.
"""
from __future__ import annotations

import csv
import hashlib
import math
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
PD = os.path.abspath(os.path.join(LANE, "..", "potential-depth"))
LADDER = os.path.join(PD, "potential_depth_ladder.csv")
DATA = os.path.join(PD, "data")

A0 = 1.2e-10
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
MSUN = 1.98892e30
G = 6.67430e-11

# published reference values -- the gate
REF = dict(n_systems_window=252,
           transfer_M1=0.10663923247024337,
           transfer_M3=0.09536049731045002,
           transfer_M0=0.2917146880783851,
           transfer_M2=0.12520136775322022,
           beta_train_rungs1to4=0.17188370232387992,
           beta_all=0.1686638467066086)


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_ladder(path=LADDER):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    assert len(rows) == 4150, f"ladder row count {len(rows)} != 4150"
    assert len(rows[0]) == 20, f"ladder column count {len(rows[0])} != 20"
    d = {}
    for k in ("r_kpc", "Mb_Msun", "g_bar", "g_obs", "nu_obs", "abs_Phi_b",
              "S_shape", "e_lg_gbar", "e_lg_gobs", "sys_lg_Mb"):
        d[k] = np.array([float(r[k]) for r in rows])
    d["system"] = np.array([r["system"] for r in rows])
    d["cls"] = np.array([r["class"] for r in rows])
    d["rank"] = np.array([int(r["class_rank"]) for r in rows])
    d["source"] = np.array([r["source"] for r in rows])
    d["probe"] = np.array([r["probe"] for r in rows])
    d["phi_method"] = np.array([r["phi_method"] for r in rows])
    d["lg"] = np.log10(d["g_bar"])
    d["lr"] = np.log10(d["r_kpc"])
    d["lp"] = np.log10(d["abs_Phi_b"])
    d["lnu"] = np.log10(d["nu_obs"])
    assert len(set(d["system"])) == 317
    return d


def nu_rar(g_bar):
    """The RAR itself, McGaugh+2016 interpolating function."""
    return 1.0 / (1.0 - np.exp(-np.sqrt(g_bar / A0)))


def window_mask(d):
    """The matched-acceleration window, copied from analyse.py sec 7."""
    lo = max(d["lg"][d["rank"] >= 5].min(), -11.6)
    hi = min(d["lg"][d["rank"] >= 5].max(), -10.4)
    return (d["lg"] >= lo) & (d["lg"] <= hi), lo, hi


def system_table(d, lp_override=None):
    """One row per system inside the window: median of each quantity.

    lp_override lets a different potential variable (a different boundary rule)
    be substituted for log|Phi_b| with everything else held fixed.
    """
    win, lo, hi = window_mask(d)
    dev = np.log10(d["nu_obs"] / nu_rar(d["g_bar"]))
    lp = d["lp"] if lp_override is None else lp_override
    ok = win & np.isfinite(lp)
    rows = {}
    for i in np.where(ok)[0]:
        rows.setdefault(d["system"][i], []).append(i)
    names = sorted(rows)
    t = dict(
        name=np.array(names),
        lg=np.array([np.median(d["lg"][rows[s]]) for s in names]),
        lp=np.array([np.median(lp[rows[s]]) for s in names]),
        lr=np.array([np.median(d["lr"][rows[s]]) for s in names]),
        dev=np.array([np.median(dev[rows[s]]) for s in names]),
        rank=np.array([d["rank"][rows[s][0]] for s in names]),
        npt=np.array([len(rows[s]) for s in names]),
        e_dev=np.array([float(np.median(d["e_lg_gobs"][rows[s]])
                              / math.sqrt(len(rows[s]))) for s in names]),
        src=np.array([d["source"][rows[s][0]] for s in names]),
    )
    t["window"] = (lo, hi)
    return t


def design(t, which):
    """Model design matrices.  Every model carries a free quadratic in log g_bar
    so the RAR itself is never the thing being tested."""
    n = len(t["lg"])
    base = np.column_stack([np.ones(n), t["lg"], t["lg"] ** 2])
    if which == "M0":
        return base
    if which == "M1":
        return np.column_stack([base, t["lp"]])
    if which == "M2":
        return np.column_stack([base, t["lr"]])
    if which == "M3":
        return np.column_stack([base, (t["rank"] > 1).astype(float)])
    if which == "M4":
        ks = sorted(set(t["rank"].tolist()))[1:]
        return np.column_stack([base]
                               + [(t["rank"] == k).astype(float) for k in ks])
    raise KeyError(which)


MODELS = ("M0", "M1", "M2", "M3", "M4")
LABEL = {"M0": "RAR only (free quadratic in log g_bar)",
         "M1": "+ beta log|Phi_b|            (potential depth)",
         "M2": "+ gamma log r                (radius)",
         "M3": "+ delta [is it not a galaxy] (class step)",
         "M4": "+ full class dummies"}


def fit_freeze_eval(t, train, test, which):
    """Fit on train ONLY, FREEZE, evaluate once on test.  Returns
    (coefficients, per-object residual on test, rms on test)."""
    A = design(t, which)
    c, *_ = np.linalg.lstsq(A[train], t["dev"][train], rcond=None)
    res = t["dev"][test] - A[test] @ c
    return c, res, float(np.sqrt(np.mean(res ** 2)))


def bic(t, which, mask=None):
    A = design(t, which)
    y = t["dev"]
    if mask is not None:
        A, y = A[mask], y[mask]
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    r = y - A @ c
    n, k = len(y), A.shape[1]
    return n * math.log(max(float(np.mean(r ** 2)), 1e-300)) + k * math.log(n), \
        float(np.sqrt(np.mean(r ** 2))), c


def rank_deficient(A, tol=1e-9):
    """True if the design has a numerically dependent column -- which is how a
    class step stops being estimable when the training set does not straddle
    the class boundary."""
    s = np.linalg.svd(A, compute_uv=False)
    return bool(s[-1] / s[0] < tol), float(s[-1] / s[0])


def gate(t, verbose=True):
    """Reproduce the published numbers before doing anything new."""
    n = len(t["lg"])
    tr = t["rank"] <= 4
    te = t["rank"] >= 5
    out = dict(n_systems_window=n)
    for m in ("M0", "M1", "M2", "M3"):
        _, _, rms = fit_freeze_eval(t, tr, te, m)
        out["transfer_" + m] = rms
    c1, *_ = np.linalg.lstsq(design(t, "M1")[tr], t["dev"][tr], rcond=None)
    ca, *_ = np.linalg.lstsq(design(t, "M1"), t["dev"], rcond=None)
    out["beta_train_rungs1to4"] = float(c1[3])
    out["beta_all"] = float(ca[3])
    bad = []
    for k, v in REF.items():
        got = out[k]
        if isinstance(v, int):
            ok = got == v
        else:
            ok = abs(got - v) < 1e-9
        if verbose:
            print(f"   GATE {k:<24} published {v!r:>22}  reproduced {got!r:>22}"
                  f"   {'OK' if ok else '*** MISMATCH ***'}")
        if not ok:
            bad.append(k)
    if bad:
        raise SystemExit(f"gate failed on {bad}")
    return out
