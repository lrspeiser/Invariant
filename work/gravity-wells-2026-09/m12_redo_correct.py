"""
BOTH X-COP TESTS REDONE with correct cluster identities.

m11 found that the rank pairing used in h12 (merger bias) and m10 (temperature)
misassigned 11 of 12 clusters. It assumed every density profile extends to the
same multiple of R500; the ratio actually runs from 1.12 to 2.16.

The correct route needs no inference: the bench's `extent` IS R500 in kpc, and
10 of the 12 values match Ghirardini Table 1 to within 2 kpc. The remaining two
extents (1250, 1368) and the two unassigned names (A644 R500=1230,
A2319 R500=1346) pair by elimination and by proximity, so the map is complete
and forced.

Both results are recomputed below. Whatever they say now supersedes what was
reported before.
"""
import csv
import json
import math
import os
import numpy as np
from astropy.io import fits
from invariant_bench import Bench, KPC

BAR = "=" * 78
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
A0, OM, OL = 1.2e-10, 0.3, 0.7
rng = np.random.default_rng(101)


def head(x):
    print("\n" + BAR + "\n" + x + "\n" + BAR)


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


ID = {float(k): v for k, v in
      json.load(open(SCR + "xcop_identity.json", encoding="utf-8")).items()}
# the two forced by elimination
TAB_EXTRA = {1250.0: dict(name="A644", z=0.0704, M500=5.66, R500=1230),
             1368.0: dict(name="A2319", z=0.0557, M500=7.31, R500=1346)}
for k, v in TAB_EXTRA.items():
    Ez = math.sqrt(OM * (1 + v["z"]) ** 3 + OL)
    v["T500"] = 8.85 * (v["M500"] * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3)
    ID[k] = v

head("1. The corrected identity map")
b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc))
uq = sorted(np.unique(ext) / KPC)
ROWS = []
for v in uq:
    key = min(ID, key=lambda k: abs(k - v))
    if abs(key - v) > 3:
        print(f"   extent {v:.1f}: NO IDENTITY")
        continue
    info = ID[key]
    m = np.abs(ext / KPC - v) < 1e-6
    exc = float(np.median(xc.nu[m] / nu_rar(xc.x[m])))
    f = os.path.join(XR, info["name"], f"{info['name']}_temperature.fits")
    kT = np.nan
    tsc = np.nan
    if os.path.exists(f):
        with fits.open(f) as h:
            d = h[1].data
            rw = np.asarray(d["RW_X"], float)
            tx = np.asarray(d["T_X"], float)
        ok = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
        rr = (xc.r[m] / KPC) / info["R500"]
        tsc = float(np.median(np.interp(rr, rw[ok], tx[ok])))
        kT = tsc * info["T500"]
    ROWS.append(dict(name=info["name"], exc=exc, kT=kT, tsc=tsc,
                     T500=info["T500"], M500=info["M500"], n=int(m.sum())))
print(f"   {'cluster':<10}{'R500':>7}{'M500':>8}{'T500':>8}{'T/T500':>9}"
      f"{'kT keV':>9}{'excess':>9}{'n':>5}")
print("   " + "-" * 66)
for r in sorted(ROWS, key=lambda q: q["kT"]):
    print(f"   {r['name']:<10}{'':>7}{r['M500']:>8.2f}{r['T500']:>8.2f}"
          f"{r['tsc']:>9.3f}{r['kT']:>9.2f}{r['exc']:>9.2f}{r['n']:>5}")
print("   " + "-" * 66)


def spear(a, c, nperm=200000):
    a, c = np.asarray(a, float), np.asarray(c, float)
    m = np.isfinite(a) & np.isfinite(c)
    a, c = a[m], c[m]
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(c)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    r0 = float((ra @ rb) / math.sqrt((ra @ ra) * (rb @ rb)))
    nl = np.array([abs(float(np.corrcoef(
        np.argsort(np.argsort(rng.permutation(a))), rb)[0, 1]))
        for _ in range(nperm)])
    return r0, float(np.mean(nl >= abs(r0))), len(a)


def partial(a, c, z):
    a, c, z = (np.asarray(v, float) for v in (a, c, z))
    m = np.isfinite(a) & np.isfinite(c) & np.isfinite(z)
    R = [np.argsort(np.argsort(v[m])).astype(float) for v in (a, c, z)]
    R = [(v - v.mean()) / (v.std() or 1) for v in R]
    n = len(R[0])
    rac, raz, rcz = (float(R[0] @ R[1] / n), float(R[0] @ R[2] / n),
                     float(R[1] @ R[2] / n))
    return (rac - raz * rcz) / math.sqrt(max(1e-12, (1 - raz ** 2) * (1 - rcz ** 2)))


head("2. THE TEMPERATURE TEST, redone")
kT = np.array([r["kT"] for r in ROWS])
ts = np.array([r["tsc"] for r in ROWS])
lM = np.log10(np.array([r["M500"] for r in ROWS]))
ex = np.array([r["exc"] for r in ROWS])
r1, p1, n1 = spear(kT, ex)
r2, p2, _ = spear(ts, ex)
r3, p3, _ = spear(lM, ex)
print(f"   (a) absolute kT vs excess      rho = {r1:+.3f}  p = {p1:.4f}  n = {n1}")
print(f"   (b) same, mass partialled out  rho = {partial(kT, ex, lM):+.3f}")
print(f"   (c) measured T/T500 vs excess  rho = {r2:+.3f}  p = {p2:.4f}")
print(f"   control: log M500 vs excess    rho = {r3:+.3f}  p = {p3:.4f}")
print(f"   confounding: kT vs log M500    rho = {spear(kT, lM, 2000)[0]:+.3f}")
kap = 1e5
pred = np.sqrt(1 + kap * 3 * (kT * KEV) / (MU * MP * C ** 2))
rr, pp, _ = spear(pred, ex)
print(f"\n   size test, kappa fixed at 1e5  rho = {rr:+.3f}  p = {pp:.4f}")
print(f"   predicted spread {np.nanmax(pred)/np.nanmin(pred):.2f}x   "
      f"observed spread {ex.max()/ex.min():.2f}x")
print(f"   median |log10(obs/pred)| = {np.nanmedian(np.abs(np.log10(ex/pred))):.4f} dex")

head("3. THE MERGER-BIAS TEST, redone")
D = {}
for r in csv.DictReader(open(SCR + "halo07_environment/"
                             "clusters_dynamical_state.tsv", encoding="utf-8"),
                        delimiter="\t"):
    k = r["cluster_name"].strip().upper().replace(" ", "").replace("_", "")
    if k.startswith("ABELL"):
        k = "A" + k[5:]
    D.setdefault(k, r)


def num(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else np.nan
    except (TypeError, ValueError):
        return np.nan


M = []
for r in ROWS:
    q = D.get(r["name"].upper()) or next(
        (v for k, v in D.items() if k.startswith(r["name"].upper())), None)
    if q is None:
        continue
    dp = num(q.get("disturbance_pct", ""))
    rp = num(q.get("relaxation_param", ""))
    if np.isfinite(dp):
        M.append(dict(name=r["name"], exc=r["exc"], dp=dp, delta=rp,
                      state=q.get("dyn_state_class", "").strip(),
                      cc=q.get("cool_core_flag", "").strip()))
print(f"   {len(M)} clusters with a disturbance index")
print(f"\n   {'cluster':<10}{'excess':>9}{'disturb %':>12}{'Yuan delta':>13}"
      f"{'state':>15}{'CC':>6}")
print("   " + "-" * 66)
for t in sorted(M, key=lambda q: q["dp"]):
    print(f"   {t['name']:<10}{t['exc']:>9.2f}{t['dp']:>12.1f}"
          f"{t['delta']:>13.2f}{t['state'][:13]:>15}{t['cc'][:4]:>6}")
print("   " + "-" * 66)
if len(M) >= 6:
    x = np.array([t["dp"] for t in M])
    y = np.array([t["exc"] for t in M])
    r0, p0, n0 = spear(x, y)
    med = np.median(x)
    lo, hi = y[x <= med], y[x > med]
    print(f"\n   Spearman(disturbance, excess) = {r0:+.3f}   p = {p0:.4f}"
          f"   n = {n0}")
    print(f"   relaxed half   median excess {np.median(lo):.2f}  (n={len(lo)})")
    print(f"   disturbed half median excess {np.median(hi):.2f}  (n={len(hi)})")
    print(f"   ratio disturbed/relaxed = {np.median(hi)/np.median(lo):.3f}")
    print("   Merger bias predicts a ratio ABOVE 1.")
    cc = [t["exc"] for t in M if t["cc"].upper().startswith("CC")]
    nc = [t["exc"] for t in M if t["cc"].upper().startswith("NCC")]
    if cc and nc:
        print(f"\n   cool-core     n={len(cc)}  median {np.median(cc):.2f}")
        print(f"   non-cool-core n={len(nc)}  median {np.median(nc):.2f}")

head("VERDICT")
print(f"""   Both results are superseded. With CORRECT identities:

   TEMPERATURE   absolute kT vs excess  {r1:+.3f} (p = {p1:.4f})
                 mass removed           {partial(kT, ex, lM):+.3f}
                 measured T/T500        {r2:+.3f} (p = {p2:.4f})
                 size test, kappa fixed {rr:+.3f} (p = {pp:.4f})

   MERGER BIAS   disturbance vs excess  {r0:+.3f} (p = {p0:.4f})
                 disturbed/relaxed      {np.median(hi)/np.median(lo):.3f}

   The previously reported numbers -- temperature -0.084, merger -0.309 with a
   ratio of 0.981 -- were computed on a scrambled name map and are withdrawn.""")
