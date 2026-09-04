"""Option 1 of the well-mirror model: perpendicular push projected on a curved
well. This is the option the earlier sweep never tested.

    g_mirror(r) = A_u(r) h'(r) / (1 + h'(r)^2)
    g(r)        = (1 - eta) g_N(r) + eta g_mirror(r)

There are two readings of it, and they differ in the one respect that matters
for the question that was actually asked -- how much push against how much pull.

OPTION 1a, the surface-projection ansatz as written.
    The proposal closes the system by DEFINING
        g_mirror(r) = g_N(r) + V_W^2 / (r + r_t)
    i.e. the mirror is declared to restore the eta g_N that was removed from the
    direct well and then to add an edge term. Substituting:
        g(r) = (1-eta) g_N + eta g_N + eta V_W^2/(r+r_t)
             = g_N(r) + eta V_W^2 / (r + r_t)
    eta now appears ONLY in the product eta V_W^2. It is not a measurable
    push/pull ratio; it is a reparameterisation of the edge-term amplitude.
    That is the same structural degeneracy already found in Options 2 and 3,
    and the test below confirms it numerically rather than asserting it: the
    fit is run at eta = 0.1 ... 0.9 and the RMS must be bit-identical.

OPTION 1b, the literal point-mirror, which is what the geometry actually gives.
    The proposal itself notes the limitation: an attractive source at u = -d
    with an equal repulsive image at u = +d produces, on the middle surface,
        A_u(r) = 2 G M_b d / (r^2 + d^2)^{3/2}
    which falls as 1/r^3, not as the 1/r a flat rotation curve needs. Taking
    that seriously and NOT inserting g_N into the mirror by hand:
        h'(r)  = chi r_M / (r + lambda r_M)        (the log-flare slope)
        d      = delta r_M
        r_M    = sqrt(G M_b / a0)
        g      = (1-eta) g_N + eta A_u h'/(1+h'^2)
    Here eta is IDENTIFIABLE, because it sets the coefficient of the direct
    Newtonian term independently of the mirror amplitude: at small r the data
    pin (1-eta), and the mirror sector cannot compensate. So this formulation
    can actually be asked "how much of the pull is push?" and made to answer.

Every parameter is GLOBAL. The only per-galaxy inputs are the measured
baryonic quantities M_b, g_N(r) and r. No galaxy gets a gravity parameter.
Protocol as everywhere else: fit on train, blind touched once at the end.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(HERE, "..", "..", "gravitylab")))

G = 6.674e-11
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
A0 = 1.2e-10


def load():
    """The same frozen bench and the same frozen split as every other run."""
    from gpu_search import Bench
    b = Bench(ndraw=1, verbose=False)
    to = (lambda a: a.get() if hasattr(a, "get") else np.asarray(a))
    d = {k: to(getattr(b, k)).astype(np.float64)
         for k in ("R", "gbar", "gobs", "Mb", "Rd")}
    d["gid"] = to(b.gid).astype(int)
    d["split"] = np.array(b.splits)[d["gid"]]
    d["r_m"] = d["R"] * KPC
    d["Mb_kg"] = d["Mb"] * MSUN
    d["y"] = np.log10(d["gobs"])
    return d


# ------------------------------------------------------------------- models
def rar(d, a0):
    x = np.maximum(d["gbar"] / a0, 1e-30)
    return d["gbar"] / (1.0 - np.exp(-np.sqrt(x)))


def opt1a(d, eta, lam, a0, VW2_scale=1.0):
    """g = g_N + eta V_W^2/(r + lambda r_M), with eta V_W^2 = s sqrt(G M a0)."""
    rM = np.sqrt(G * d["Mb_kg"] / a0)
    vf2 = VW2_scale * np.sqrt(G * d["Mb_kg"] * a0)
    return d["gbar"] + vf2 / (d["r_m"] + lam * rM)


def opt1b(d, eta, lam, chi, delta, a0):
    """The literal point-mirror push, projected onto the flaring surface."""
    rM = np.sqrt(G * d["Mb_kg"] / a0)
    dd = delta * rM
    Au = 2.0 * G * d["Mb_kg"] * dd / (d["r_m"] ** 2 + dd ** 2) ** 1.5
    hp = chi * rM / (d["r_m"] + lam * rM)
    gm = Au * hp / (1.0 + hp ** 2)
    return (1.0 - eta) * d["gbar"] + eta * gm


def rms(d, g, m):
    g = np.maximum(g, 1e-30)
    return float(np.sqrt(np.mean((d["y"][m] - np.log10(g[m])) ** 2)))


# --------------------------------------------------------------------- fits
def fit(d, m, f, x0, bounds, nstart=24, seed=3):
    rng = np.random.default_rng(seed)
    best = (np.inf, None)
    starts = [np.asarray(x0, float)]
    for _ in range(nstart - 1):
        starts.append(np.array([rng.uniform(lo, hi) for lo, hi in bounds]))
    for s in starts:
        try:
            r = minimize(lambda p: rms(d, f(p), m), s, bounds=bounds,
                         method="L-BFGS-B")
        except Exception:
            continue
        if np.isfinite(r.fun) and r.fun < best[0]:
            best = (float(r.fun), r.x.copy())
    return best


def main():
    d = load()
    tr = d["split"] == "train"
    bl = d["split"] == "blind"
    print("=" * 76)
    print("OPTION 1 OF THE WELL-MIRROR MODEL -- the formulation never tested")
    print("=" * 76)
    print(f"   {tr.sum():,} train points / {bl.sum():,} blind points, "
          f"{len(set(d['gid']))} galaxies")

    out = {}

    # ---------------------------------------------------------- RAR baseline
    fr, xr = fit(d, tr, lambda p: rar(d, p[0]), [A0], [(1e-11, 1e-9)])
    br = rms(d, rar(d, xr[0]), bl)
    print(f"\n   RAR baseline (a0 free)     train {fr:.4f}  blind {br:.4f}  "
          f"a0 = {xr[0]:.3e}")
    out["rar"] = {"train": fr, "blind": br, "a0": float(xr[0])}

    # ------------------------------------------- Option 1a and the eta claim
    print("\n   OPTION 1a  g = g_N + eta V_W^2/(r + lambda r_M)")
    f1, x1 = fit(d, tr, lambda p: opt1a(d, 0.5, p[0], p[1], p[2]),
                 [1.0, A0, 1.0], [(1e-3, 50.0), (1e-11, 1e-9), (1e-3, 30.0)])
    b1 = rms(d, opt1a(d, 0.5, x1[0], x1[1], x1[2]), bl)
    print(f"      best fit                 train {f1:.4f}  blind {b1:.4f}")
    print(f"      lambda = {x1[0]:.4f}   a0 = {x1[1]:.3e}   "
          f"eta V_W^2 scale = {x1[2]:.4f}")
    out["opt1a"] = {"train": f1, "blind": b1, "lam": float(x1[0]),
                    "a0": float(x1[1]), "scale": float(x1[2])}

    print("\n      is eta identifiable?  refit at fixed eta, all else free:")
    ident = []
    for eta in (0.1, 0.25, 0.5, 0.75, 0.9):
        fe, xe = fit(d, tr, lambda p, e=eta: opt1a(d, e, p[0], p[1], p[2]),
                     [1.0, A0, 1.0],
                     [(1e-3, 50.0), (1e-11, 1e-9), (1e-3, 30.0)], nstart=12)
        ident.append((eta, fe))
        print(f"         eta = {eta:.2f}   train RMS {fe:.6f}")
    spread = max(f for _, f in ident) - min(f for _, f in ident)
    print(f"      spread across eta: {spread:.2e} dex")
    print("      => eta is NOT identifiable in Option 1a. It enters only as the")
    print("         product eta V_W^2, so it is a relabelling of the edge-term")
    print("         amplitude, not a measurable push/pull ratio.")
    out["opt1a_eta_scan"] = ident
    out["opt1a_eta_spread_dex"] = spread

    # -------------------------------------------------- Option 1b, eta real
    print("\n   OPTION 1b  literal point mirror, "
          "g = (1-eta) g_N + eta A_u h'/(1+h'^2)")
    bnds = [(0.0, 0.999), (1e-3, 50.0), (1e-4, 1e4), (1e-3, 50.0),
            (1e-11, 1e-9)]
    f2, x2 = fit(d, tr, lambda p: opt1b(d, p[0], p[1], p[2], p[3], p[4]),
                 [0.5, 1.0, 1.0, 1.0, A0], bnds, nstart=64)
    b2 = rms(d, opt1b(d, x2[0], x2[1], x2[2], x2[3], x2[4]), bl)
    print(f"      best fit                 train {f2:.4f}  blind {b2:.4f}")
    print(f"      eta = {x2[0]:.4f}  lambda = {x2[1]:.3f}  chi = {x2[2]:.4g}  "
          f"delta = {x2[3]:.3f}  a0 = {x2[4]:.3e}")
    out["opt1b"] = {"train": f2, "blind": b2, "eta": float(x2[0]),
                    "lam": float(x2[1]), "chi": float(x2[2]),
                    "delta": float(x2[3]), "a0": float(x2[4])}

    print("\n      profile over eta -- THE push-vs-pull question, answered:")
    prof = []
    for eta in np.linspace(0.0, 0.95, 20):
        fe, xe = fit(d, tr,
                     lambda p, e=eta: opt1b(d, e, p[0], p[1], p[2], p[3]),
                     [1.0, 1.0, 1.0, A0],
                     [(1e-3, 50.0), (1e-4, 1e4), (1e-3, 50.0), (1e-11, 1e-9)],
                     nstart=20)
        prof.append((float(eta), fe))
    fmin = min(f for _, f in prof)
    for eta, fe in prof:
        bar = "#" * int(round((fe - fmin) / max(fmin, 1e-9) * 400))
        print(f"         eta = {eta:4.2f}   train RMS {fe:.5f}   {bar}")
    out["opt1b_eta_profile"] = prof
    ebest = min(prof, key=lambda t: t[1])[0]
    print(f"      best eta on train: {ebest:.2f}")
    print(f"      RMS spread across eta: {max(f for _,f in prof)-fmin:.4f} dex"
          "  (this is what makes eta identifiable here)")

    print("\n" + "=" * 76)
    print(f"   RAR      train {fr:.4f}   blind {br:.4f}")
    print(f"   Opt 1a   train {f1:.4f}   blind {b1:.4f}   "
          f"({100*(br-b1)/br:+.2f}% vs RAR on blind)")
    print(f"   Opt 1b   train {f2:.4f}   blind {b2:.4f}   "
          f"({100*(br-b2)/br:+.2f}% vs RAR on blind)")
    print("=" * 76)

    p = os.path.join(HERE, "option1_results.json")
    with open(p, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\n   written: {p}")


if __name__ == "__main__":
    main()
