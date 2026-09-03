"""
TWO TESTS THE CRITIQUE DEMANDS, both runnable on data in hand.

TEST 1 -- ENDOGENEITY. The objection: temperature sits on BOTH sides of the
correlation. The hydrostatic acceleration is

    g_HSE(r) = -(kT/(mu m_p r)) [ dln n_e/dln r + dln T/dln r ]

so g_obs is proportional to T at fixed logarithmic slopes, while g_bar depends
on n_e and not on T. Taking logs of the excess,

    ln excess = ln T + ln S - ln( nu_RAR * g_bar * r ) + const

the +ln T term is there by construction. A correlation with kT is therefore
partly guaranteed, and the question is HOW MUCH.

This is decidable because the two hypotheses predict DIFFERENT SLOPES:

    pure algebraic channel        d ln excess / d ln T  ->  1
    pressure model at kappa       d ln excess / d ln T  =  (1/2) x/(1+x)
                                  with x = kappa*3kT/(mu m_p c^2) ~ 3.6
                                  giving ~ 0.39

So measure the slope. Near 1 means the signal is bookkeeping; near 0.4 means it
is not, and the difference is large enough to see with twelve points.

TEST 2 -- THE RADIAL PROFILE, zero free parameters. Every cluster test so far
used ONE number per cluster, the median excess. But the pressure model makes a
prediction at every radius, because T varies with radius inside each cluster:

    excess(r) = sqrt( 1 + kappa * 3kT(r) / (mu m_p c^2) )

With kappa already fixed there is nothing left to tune. If the model is right
it must track the radial run of the excess WITHIN each cluster, not merely its
cluster-to-cluster average. That is a far stronger test and it has never been
run.
"""
import json
import math
import os
import re

import numpy as np
from astropy.io import fits

from invariant_bench import Bench, KPC

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
A0, OM, OL = 1.2e-10, 0.3, 0.7
KAPPA = 1.36e5
BAR = "=" * 78
rng = np.random.default_rng(4242)


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    d = math.sqrt(float(ra @ ra) * float(rb @ rb))
    return float(ra @ rb / d) if d > 0 else float("nan")


# ------------------------------------------------------------------- ingest
tex = open(SCR + "xcop_T/XCOP_thermo.tex", encoding="utf-8").read()
i = tex.index("Basic properties of the X-COP sample")
TAB = {}
for line in tex[i:i + 4000].split("\n"):
    cells = [c.strip() for c in line.split("&")]
    if len(cells) < 6 or line.strip().startswith("%"):
        continue
    nm = cells[0].strip()
    if not re.match(r"^[A-Za-z]+\d", nm):
        continue

    def val(s):
        m = re.search(r"([\d.]+)", s.replace("$", ""))
        return float(m.group(1)) if m else float("nan")
    z, M500, R500 = val(cells[1]), val(cells[3]), val(cells[4])
    Ez = math.sqrt(OM * (1 + z) ** 3 + OL)
    TAB[nm] = dict(z=z, M500=M500, R500=R500,
                   T500=8.85 * (M500 * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3))

b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
NAME = {}
for v in sorted(np.unique(ext)):
    hit = [k for k, t in TAB.items() if abs(t["R500"] - v) < 3.0]
    if len(hit) == 1:
        NAME[v] = hit[0]
left_v = [v for v in sorted(np.unique(ext)) if v not in NAME]
left_n = [k for k in TAB if k not in NAME.values()]
for v in left_v:
    k = min(left_n, key=lambda q: abs(TAB[q]["R500"] - v))
    NAME[v] = k
    left_n.remove(k)

CL = []
for v in sorted(np.unique(ext)):
    nm = NAME[v]
    t = TAB[nm]
    m = np.abs(ext - v) < 1e-9
    o = np.argsort(xc.r[m])
    r_kpc = (xc.r[m] / KPC)[o]
    gobs = xc.go[m][o]
    gbar = xc.gb[m][o]
    exc_r = (xc.nu[m] / nu_rar(xc.x[m]))[o]
    with fits.open(os.path.join(XR, nm, f"{nm}_temperature.fits")) as h:
        d = h[1].data
        rw = np.asarray(d["RW_X"], float)
        tx = np.asarray(d["T_X"], float)
    ok = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
    kT_r = np.interp(r_kpc / t["R500"], rw[ok], tx[ok]) * t["T500"]
    CL.append(dict(name=nm, r=r_kpc, kT=kT_r, exc=exc_r, gobs=gobs, gbar=gbar,
                   kT_med=float(np.median(kT_r)),
                   exc_med=float(np.median(exc_r)),
                   gbar_med=float(np.median(gbar)),
                   gobs_med=float(np.median(gobs)),
                   elim=nm in ("A644", "A2319")))

head("TEST 1  --  is the temperature correlation algebraic?")
kT = np.array([c["kT_med"] for c in CL])
ex = np.array([c["exc_med"] for c in CL])
gb = np.array([c["gbar_med"] for c in CL])
go = np.array([c["gobs_med"] for c in CL])

print("   First, where the shared dependence would enter:\n")
print(f"   rho(kT, g_obs)   = {spearman(kT, go):+.3f}   "
      f"<- g_HSE carries T directly")
print(f"   rho(kT, g_bar)   = {spearman(kT, gb):+.3f}   "
      f"<- g_bar carries n_e, not T")
print(f"   rho(kT, excess)  = {spearman(kT, ex):+.3f}")

lt, le = np.log(kT), np.log(ex)
A = np.vstack([lt, np.ones_like(lt)]).T
slope, icept = np.linalg.lstsq(A, le, rcond=None)[0]
res = le - (slope * lt + icept)
se = float(np.sqrt(np.sum(res ** 2) / (len(lt) - 2)
                   / np.sum((lt - lt.mean()) ** 2)))
x_typ = KAPPA * 3 * (5.0 * KEV) / (MU * MP * C ** 2)
pred_pressure = 0.5 * x_typ / (1.0 + x_typ)
print(f"""
   Now the slope, which the two explanations disagree about:

      d ln(excess) / d ln(kT)  measured   = {slope:+.3f} +- {se:.3f}

      pure algebraic channel   predicts   = +1.000
      pressure model, kappa = {KAPPA:.2e}    = {pred_pressure:+.3f}
        (x = kappa*3kT/(mu m_p c^2) = {x_typ:.2f} at 5 keV)""")
d_alg = abs(slope - 1.0) / se
d_prs = abs(slope - pred_pressure) / se
print(f"""
      distance from algebraic  : {d_alg:.1f} sigma
      distance from pressure   : {d_prs:.1f} sigma""")

head("TEST 2  --  the radial profile, with kappa already fixed")
print("""   excess(r) = sqrt( 1 + kappa * 3kT(r)/(mu m_p c^2) ),  no free parameters.
   Within each cluster T varies with radius, so the model predicts how the
   excess RUNS, not just its average. Tested per cluster below.\n""")
print(f"   {'cluster':<10}{'n':>4}{'kT range keV':>16}{'exc obs range':>16}"
      f"{'exc pred range':>17}{'rho(r)':>9}{'med ratio':>11}")
print("   " + "-" * 83)
rows = []
for c in CL:
    pred = np.sqrt(1.0 + KAPPA * 3 * (c["kT"] * KEV) / (MU * MP * C ** 2))
    rho_r = spearman(c["exc"], pred)
    ratio = float(np.median(c["exc"] / pred))
    rows.append((c["name"], rho_r, ratio, c["elim"]))
    print(f"   {c['name']:<10}{len(c['r']):>4}"
          f"{f'{c[chr(107)+chr(84)].min():.1f}-{c[chr(107)+chr(84)].max():.1f}':>16}"
          f"{f'{c[chr(101)+chr(120)+chr(99)].min():.2f}-{c[chr(101)+chr(120)+chr(99)].max():.2f}':>16}"
          f"{f'{pred.min():.2f}-{pred.max():.2f}':>17}"
          f"{rho_r:>+9.3f}{ratio:>11.3f}")
print("   " + "-" * 83)
rr = np.array([r[1] for r in rows])
ra = np.array([r[2] for r in rows])
print(f"   median within-cluster rho(observed, predicted) : {np.median(rr):+.3f}")
print(f"   clusters with rho > 0                          : "
      f"{int((rr > 0).sum())} of {len(rr)}")
print(f"   normalisation ratio, median across clusters    : {np.median(ra):.3f}")
print(f"   scatter of that ratio                          : {np.std(ra):.3f}")

head("A cheaper null: does the same test pass on a shuffled temperature map?")
null = []
for _ in range(2000):
    perm = rng.permutation(len(CL))
    vals = []
    for i2, c in enumerate(CL):
        other = CL[perm[i2]]
        kt = np.interp(np.linspace(0, 1, len(c["r"])),
                       np.linspace(0, 1, len(other["kT"])), other["kT"])
        p = np.sqrt(1.0 + KAPPA * 3 * (kt * KEV) / (MU * MP * C ** 2))
        vals.append(spearman(c["exc"], p))
    null.append(np.median(vals))
null = np.array(null)
obs = float(np.median(rr))
print(f"   observed median within-cluster rho : {obs:+.3f}")
print(f"   shuffled-temperature null          : {np.median(null):+.3f} "
      f"[{np.percentile(null,2.5):+.3f}, {np.percentile(null,97.5):+.3f}]")
print(f"   fraction of null >= observed       : {float(np.mean(null >= obs)):.4f}")

json.dump(dict(slope=float(slope), slope_err=se,
               pred_pressure=float(pred_pressure),
               within_rho_median=float(np.median(rr)),
               ratio_median=float(np.median(ra)),
               null_frac=float(np.mean(null >= obs))),
          open(SCR + "c70_endogeneity.json", "w", encoding="utf-8"), indent=1)
print("\n   wrote c70_endogeneity.json")
