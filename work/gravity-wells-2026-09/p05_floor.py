"""
CALIBRATE THE NOISE FLOOR.

The corrected check still issues verdicts off a p-value, and at n ~ 4000 that
is worthless: pure random noise scored partial = -0.031 at p = 0.043 and was
reported INFORMATIVE. Significance is not the question. Effect size is.

So: run the check on many independent noise draws, measure the distribution of
|partial| that pure noise actually produces, and compare every real variable to
it. A variable that cannot clear the noise floor by a clear margin should not
receive a verdict at all -- the check should say so rather than guess.
"""
import math
import numpy as np
from invariant_bench import Bench, KPC, MSUN

BAR = "=" * 78
A0 = 1.2e-10
b = Bench(verbose=False)
rng = np.random.default_rng(99)
USE = [k for k, v in b.PROBE_KIND.items() if v[2] not in ("bound", "holdout")]

vals_nu, vals_x, vals_lb, lens = [], [], [], {}
for k in USE:
    d = b.d[k]
    vals_nu.append(d.nu); vals_x.append(d.x)
    vals_lb.append(np.full(len(d), 1.0 if b.PROBE_KIND[k][1] == "spheroid" else 0.0))
    lens[k] = len(d)
NU, XX, LB0 = (np.concatenate(vals_nu), np.concatenate(vals_x),
               np.concatenate(vals_lb))
Y0 = np.log10(NU) - np.log10(1 / (1 - np.exp(-np.sqrt(XX))))
FIN = np.isfinite(NU) & (NU > 0) & (XX > 0) & np.isfinite(Y0)


def rk(a):
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float); r[o] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def co(u, w):
    u = u - u.mean(); w = w - w.mean()
    dn = math.sqrt((u @ u) * (w @ w))
    return float(u @ w / dn) if dn > 0 else 0.0


def partial_of(V, mask):
    v, lb, y = V[mask], LB0[mask], Y0[mask]
    rv, rl, ry = rk(v), rk(lb), rk(y)
    a_, c_, d_ = co(rv, rl), co(rv, ry), co(rl, ry)
    return (c_ - a_ * d_) / math.sqrt(max(1e-12, (1 - a_ ** 2) * (1 - d_ ** 2)))


print(BAR + "\n1. What |partial| does PURE NOISE produce at this n?\n" + BAR)
draws = np.array([abs(partial_of(rng.normal(size=len(LB0)), FIN))
                  for _ in range(600)])
f50, f95, f99 = (float(np.percentile(draws, q)) for q in (50, 95, 99))
print(f"   600 independent noise draws, n = {int(FIN.sum())}")
print(f"      median |partial| = {f50:.4f}")
print(f"      95th percentile  = {f95:.4f}   <- the noise floor")
print(f"      99th percentile  = {f99:.4f}")
print(f"      maximum observed = {draws.max():.4f}")

print("\n" + BAR + "\n2. Every real variable against that floor\n" + BAR)


def get(fn):
    out = []
    for k in USE:
        d = b.d[k]
        v = np.asarray(fn(d), float)
        if np.ndim(v) == 0:
            v = np.full(len(d), float(v))
        out.append(v)
    return np.concatenate(out)


VARS = [
    ("sphericity", lambda d: np.asarray(d.sphericity, float) * np.ones(len(d))),
    ("radius r", lambda d: d.r),
    ("compact = r/extent",
     lambda d: d.r / (np.asarray(d.extent, float) * np.ones(len(d)))),
    ("a_N", lambda d: d.x * A0),
    ("log10 enclosed baryonic mass", lambda d: np.log10(d.M / MSUN)),
    ("log10 enclosed density", lambda d: np.log10(np.abs(d.rho) + 1e-40)),
    ("baryonic potential phi", lambda d: d.phi),
]
lb_r = rk(LB0[FIN]); y_r = rk(Y0[FIN])
r_ly = co(lb_r, y_r)
print(f"   the bare dataset label alone reaches |r| = {abs(r_ly):.3f}")
print(f"   the noise floor (95th pct) is           {f95:.3f}")
print(f"\n   {'variable':<32}{'|partial|':>11}{'x floor':>9}{'% of label':>12}"
      f"   status")
print("   " + "-" * 76)
for nm, fn in VARS:
    V = get(fn)
    m = FIN & np.isfinite(V)
    if np.ptp(V[m]) == 0:
        continue
    pa = abs(partial_of(V, m))
    ratio = pa / f95
    pct = 100 * pa / abs(r_ly)
    st = ("clears floor" if ratio > 3 else
          "INDETERMINATE" if ratio > 1 else "at noise")
    print(f"   {nm:<32}{pa:>11.3f}{ratio:>9.1f}{pct:>11.0f}%   {st}")
print("   " + "-" * 76)

print("\n" + BAR + "\n3. What this means for the protocol\n" + BAR)
print(f"""   Every real variable lands between {min(abs(partial_of(get(f), FIN & np.isfinite(get(f)))) for _, f in VARS):.3f} and """
      f"""{max(abs(partial_of(get(f), FIN & np.isfinite(get(f)))) for _, f in VARS):.3f}.
   The noise floor is {f95:.3f}. The dataset label alone is {abs(r_ly):.3f}.

   So the entire dynamic range available to this check -- floor to best real
   variable -- is a factor of {max(abs(partial_of(get(f), FIN & np.isfinite(get(f)))) for _, f in VARS)/f95:.1f}, while the label sits {abs(r_ly)/f95:.0f}x above the floor.

   The check can separate a real variable from noise. It CANNOT rank real
   variables against each other, and it cannot support a kill decision on its
   own, because everything physical is compressed into a narrow band well
   below the label. Kills need an independent control.""")
