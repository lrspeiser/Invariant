"""
RE-AUDIT: every confound verdict, recomputed with the three defects fixed.

Defects found in p03:
  A. The rule fires when |r_vy| and |r_ly| are within 0.08. It therefore cannot
     tell "beats the label" from "far worse than the label", and pure random
     noise is reported as carrying information beyond the label.
  B. argsort(argsort(x)) breaks ties by array position. The bench concatenates
     probe by probe, so a GLOBAL CONSTANT scores corr = +0.948 with the dataset
     label. That number is manufactured entirely by concatenation order.
  C. The probe filter excludes only role == "bound", which is solar alone.
     Both blind holdouts -- kids and widebin -- were being consumed by the
     check, so holdout data influenced variable selection.

The corrected check fixes all three and replaces the distance rule with the
question actually being asked: does V explain the RAR residual BEYOND what the
dataset label already explains? That is a partial correlation, and it separates
three cases the old rule collapsed into two.

     LABEL        r_vy substantial but partial ~ 0  -> the signal IS the label
     CARRIES NIL  r_vy ~ 0 and partial ~ 0          -> the variable is empty
     INFORMATIVE  partial substantially non-zero    -> real, beyond the label
"""
import math
import numpy as np
from invariant_bench import Bench, KPC, MSUN

BAR = "=" * 78
A0 = 1.2e-10
b = Bench(verbose=False)
rng = np.random.default_rng(11)
HOLDOUT = {k for k, v in b.PROBE_KIND.items() if v[2] in ("holdout", "bound")}
print(f"excluded from the check: {sorted(HOLDOUT)}")
print(f"used: {sorted(set(b.d) - HOLDOUT)}")


def rank(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float); r[o] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def corr(u, w):
    u = u - u.mean(); w = w - w.mean()
    dn = math.sqrt((u @ u) * (w @ w))
    return float(u @ w / dn) if dn > 0 else 0.0


def check(getter, leak=False, nperm=4000):
    vals, labs, nus, xs = [], [], [], []
    for k, d in b.d.items():
        if not leak and k in HOLDOUT:
            continue
        if leak and b.PROBE_KIND[k][2] == "bound":
            continue
        try:
            v = np.asarray(getter(d), float)
        except Exception:
            return None
        if np.ndim(v) == 0:
            v = np.full(len(d), float(v))
        if len(v) != len(d):
            return None
        vals.append(v); nus.append(d.nu); xs.append(d.x)
        labs.append(np.full(len(d),
                    1.0 if b.PROBE_KIND[k][1] == "spheroid" else 0.0))
    V, LB = np.concatenate(vals), np.concatenate(labs)
    NU, XX = np.concatenate(nus), np.concatenate(xs)
    m = np.isfinite(V) & np.isfinite(NU) & (NU > 0) & (XX > 0)
    V, LB, NU, XX = V[m], LB[m], NU[m], XX[m]
    if len(V) < 50 or np.ptp(V) == 0:
        return dict(n=len(V), r_vl=0.0, r_vy=0.0, r_ly=0.0, partial=0.0,
                    p=1.0, verdict="CARRIES NIL")
    y = np.log10(NU) - np.log10(1 / (1 - np.exp(-np.sqrt(XX))))
    rv, rl, ry = rank(V), rank(LB), rank(y)
    r_vl, r_vy, r_ly = corr(rv, rl), corr(rv, ry), corr(rl, ry)
    den = math.sqrt(max(1e-12, (1 - r_vl ** 2) * (1 - r_ly ** 2)))
    part = (r_vy - r_vl * r_ly) / den
    # permutation null: shuffle V WITHIN each label block, so the label
    # structure is preserved exactly and only V's within-block ordering moves
    idx0, idx1 = np.where(LB == 0)[0], np.where(LB == 1)[0]
    null = np.empty(nperm)
    for i in range(nperm):
        vp = rv.copy()
        vp[idx0] = rng.permutation(vp[idx0])
        vp[idx1] = rng.permutation(vp[idx1])
        a_, b_ = corr(vp, rl), corr(vp, ry)
        null[i] = (b_ - a_ * r_ly) / math.sqrt(
            max(1e-12, (1 - a_ ** 2) * (1 - r_ly ** 2)))
    p = float(np.mean(np.abs(null) >= abs(part)))
    if p > 0.05:
        vd = "LABEL" if abs(r_vy) > 0.15 else "CARRIES NIL"
    else:
        vd = "INFORMATIVE"
    return dict(n=len(V), r_vl=r_vl, r_vy=r_vy, r_ly=r_ly, partial=part,
                p=p, verdict=vd)


def sph(d):
    return np.asarray(d.sphericity, float) * np.ones(len(d))


VARS = [
    ("sphericity", sph),
    ("radius r", lambda d: d.r),
    ("-radius", lambda d: -d.r),
    ("exp(-r/31.6kpc)", lambda d: np.exp(-d.r / (31.6 * KPC))),
    ("compact = r/extent", lambda d: d.r / (np.asarray(d.extent, float)
                                            * np.ones(len(d)))),
    ("a_N", lambda d: d.x * A0),
    ("log10 enclosed baryonic mass", lambda d: np.log10(d.M / MSUN)),
    ("log10 enclosed density", lambda d: np.log10(np.abs(d.rho) + 1e-40)),
    ("baryonic potential phi", lambda d: d.phi),
    ("g_bar", lambda d: d.gb),
    ("t = r / r_a0", lambda d: d.r / np.sqrt(np.maximum(d.M * 6.674e-11 / A0, 1e-9))),
    ("PURE NOISE (control)", lambda d: rng.normal(size=len(d))),
    ("GLOBAL CONSTANT (control)", lambda d: np.full(len(d), 3.0)),
]

print("\n" + BAR + "\nCORRECTED VERDICTS (holdouts excluded, ties averaged)\n" + BAR)
print(f"   {'variable':<32}{'r_vy':>8}{'partial':>9}{'p':>8}   verdict")
print("   " + "-" * 72)
RES = {}
for nm, g in VARS:
    r = check(g)
    if r is None:
        print(f"   {nm:<32}   (not computable on all probes)")
        continue
    RES[nm] = r
    print(f"   {nm:<32}{r['r_vy']:>+8.3f}{r['partial']:>+9.3f}{r['p']:>8.4f}"
          f"   {r['verdict']}")
print("   " + "-" * 72)

print("\n" + BAR + "\nDid the holdout leak change any verdict?\n" + BAR)
print(f"   {'variable':<32}{'clean':>14}{'with leak':>14}   same?")
print("   " + "-" * 68)
flips = 0
for nm, g in VARS:
    a, c = check(g, leak=False, nperm=2000), check(g, leak=True, nperm=2000)
    if a is None or c is None:
        continue
    same = a["verdict"] == c["verdict"]
    flips += (not same)
    print(f"   {nm:<32}{a['verdict']:>14}{c['verdict']:>14}   "
          f"{'yes' if same else 'NO -- FLIPPED'}")
print("   " + "-" * 68)
print(f"   {flips} verdict(s) changed when the blind holdouts were included.")
