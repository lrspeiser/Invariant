"""
AUDIT OF THE CONFOUND PROTOCOL ITSELF.

Every verdict this programme issued -- seven kills and a handful of passes --
came out of Bench.confound. Before deciding whether to retest anything, the
check has to be tested on variables whose correct verdict is known in advance.

Two suspected defects:

  A. The rule convicts when |r_vy| and |r_ly| are within 0.08 of EACH OTHER.
     A variable carrying almost NO information (|r_vy| ~ 0) is far from the
     label's |r_ly|, so it passes -- and is reported as "carries information
     beyond the dataset label". That is a false reassurance, not a kill.

  B. Ranks come from argsort(argsort(x)), which breaks ties by POSITION in the
     array. The bench concatenates probe by probe, so a variable that is
     constant within each probe gets ranks that climb monotonically inside each
     block -- manufacturing correlation out of concatenation order.

Both are tested below against a corrected implementation.
"""
import math
import numpy as np
from invariant_bench import Bench

BAR = "=" * 78
b = Bench(verbose=False)
rng = np.random.default_rng(4)


def assemble(getter):
    vals, labs, nus, xs = [], [], [], []
    for k, d in b.d.items():
        if b.PROBE_KIND[k][2] == "bound":
            continue
        v = np.asarray(getter(d), float)
        if np.ndim(v) == 0:
            v = np.full(len(d), float(v))
        vals.append(v); nus.append(d.nu); xs.append(d.x)
        labs.append(np.full(len(d),
                    1.0 if b.PROBE_KIND[k][1] == "spheroid" else 0.0))
    V, LB = np.concatenate(vals), np.concatenate(labs)
    NU, XX = np.concatenate(nus), np.concatenate(xs)
    m = np.isfinite(V) & np.isfinite(NU) & (NU > 0) & (XX > 0)
    return V[m], LB[m], NU[m], XX[m]


def rank_pos(a):                      # what the bench does: ties by position
    return np.argsort(np.argsort(a)).astype(float)


def rank_avg(a):                      # correct: ties share the average rank
    o = np.argsort(a, kind="mergesort")
    r = np.empty(len(a), float)
    r[o] = np.arange(len(a), dtype=float)
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    s = np.zeros(len(cnt)); np.add.at(s, inv, r)
    return (s / cnt)[inv]


def corr(u, w):
    u = u - u.mean(); w = w - w.mean()
    d = math.sqrt((u @ u) * (w @ w))
    return float(u @ w / d) if d > 0 else 0.0


def verdicts(getter):
    V, LB, NU, XX = assemble(getter)
    y = np.log10(NU) - np.log10(1 / (1 - np.exp(-np.sqrt(XX))))
    out = {}
    for tag, rk in (("bench", rank_pos), ("fixed", rank_avg)):
        rv, rl, ry = rk(V), rk(LB), rk(y)
        r_vl, r_vy, r_ly = corr(rv, rl), corr(rv, ry), corr(rl, ry)
        # bench rule
        bench_kill = abs(r_vl) > 0.8 or abs(abs(r_vy) - abs(r_ly)) < 0.08
        # principled rule: does V explain y BEYOND LB? partial correlation.
        den = math.sqrt(max(1e-12, (1 - r_vl ** 2) * (1 - r_ly ** 2)))
        part = (r_vy - r_vl * r_ly) / den
        out[tag] = dict(r_vl=r_vl, r_vy=r_vy, r_ly=r_ly, partial=part,
                        bench_kill=bench_kill, weak=abs(r_vy) < abs(r_ly))
    return out


print(BAR + "\nTEST BATTERY: variables whose correct verdict is known\n" + BAR)
CASES = [
    ("pure noise (correct: carries NOTHING)",
     lambda d: rng.normal(size=len(d))),
    ("the label itself (correct: LABEL)",
     lambda d: np.full(len(d), 1.0 if b.PROBE_KIND[d.name][1] == "spheroid"
                       else 0.0) if hasattr(d, "name") else np.zeros(len(d))),
    ("constant within each probe (correct: LABEL)",
     lambda d: np.full(len(d), float(abs(hash(str(d.probe)) % 97)))
     if hasattr(d, "probe") else np.zeros(len(d))),
    ("radius (known LABEL)", lambda d: d.r),
    ("-radius (same information)", lambda d: -d.r),
    ("exp(-r/31.6kpc) (passed before the sign fix)",
     lambda d: np.exp(-d.r / (31.6 * 3.0856775814913673e19))),
    ("a_N (previously PASSED)", lambda d: d.x * 1.2e-10),
    ("noise x 0.01 + tiny real signal",
     lambda d: 0.01 * rng.normal(size=len(d)) + np.log10(d.x + 1e-30)),
]
print(f"   {'variable':<44}{'bench':>9}{'|r_vy|':>9}{'|r_ly|':>9}{'partial':>10}")
print("   " + "-" * 81)
for nm, g in CASES:
    try:
        r = verdicts(g)
    except Exception as e:
        print(f"   {nm:<44}  skipped ({type(e).__name__})")
        continue
    bb = r["bench"]
    tag = "KILL" if bb["bench_kill"] else "PASS"
    print(f"   {nm:<44}{tag:>9}{abs(bb['r_vy']):>9.3f}{abs(bb['r_ly']):>9.3f}"
          f"{bb['partial']:>10.3f}")
print("   " + "-" * 81)

print("\n" + BAR + "\nDEFECT A: does a variable carrying NOTHING pass?\n" + BAR)
r = verdicts(lambda d: rng.normal(size=len(d)))["bench"]
print(f"   pure random noise:")
print(f"      corr(noise, RAR residual) = {r['r_vy']:+.4f}   (it is noise)")
print(f"      corr(label, RAR residual) = {r['r_ly']:+.4f}")
print(f"      |difference| = {abs(abs(r['r_vy'])-abs(r['r_ly'])):.4f}  "
      f"-> {'KILL' if r['bench_kill'] else 'PASS'}")
print(f"      partial corr (noise | label) = {r['partial']:+.4f}")
print("\n   The bench reports this as 'carries information beyond the dataset")
print("   label'. It carries nothing. The rule cannot tell 'better than the")
print("   label' from 'far worse than the label' -- it only measures distance.")

print("\n" + BAR + "\nDEFECT B: do position-broken ties manufacture correlation?\n" + BAR)
V, LB, NU, XX = assemble(lambda d: np.full(len(d), 3.0))   # global constant
y = np.log10(NU) - np.log10(1 / (1 - np.exp(-np.sqrt(XX))))
print(f"   a GLOBAL CONSTANT (identical value for every point, n = {len(V)}):")
print(f"      corr with label, ties-by-position = "
      f"{corr(rank_pos(V), rank_pos(LB)):+.4f}")
print(f"      corr with label, ties-averaged    = "
      f"{corr(rank_avg(V), rank_avg(LB)):+.4f}")
print(f"      corr with residual, by-position   = "
      f"{corr(rank_pos(V), rank_pos(y)):+.4f}")
print(f"      corr with residual, ties-averaged = "
      f"{corr(rank_avg(V), rank_avg(y)):+.4f}")
print("\n   A constant contains zero information by construction. Any nonzero")
print("   number in the by-position row is manufactured by concatenation order.")
