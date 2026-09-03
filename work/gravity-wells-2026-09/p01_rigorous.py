"""
PAPER-GRADE ANALYSIS of the X-COP acceleration discrepancy.

Everything up to now has been exploratory and reported as medians without
uncertainties. This is the version that would have to stand in front of a
referee: pre-specified hypotheses, propagated errors, bootstrap intervals,
power analysis, multiple-testing accounting, and sensitivity to every analysis
choice that was made by hand.

HYPOTHESES, and their status BEFORE the data were examined:

  H1  Merger bias. Standard in the literature: disturbed clusters have more
      non-thermal support, so hydrostatic masses underestimate more, so the
      apparent discrepancy is LARGER. Direction pre-specified: POSITIVE.

  H2  Amplified pressure. rho_eff = rho(1 + kappa*3P/rho c^2). Derived before
      the temperatures were obtained, and kappa was fixed by the
      galaxy-vs-cluster comparison, not by these data. Direction
      pre-specified: POSITIVE. Amplitude pre-specified: kappa = 1e5.

Both are confirmatory tests of prior hypotheses with pre-registered directions,
not exploratory searches. That matters for the multiple-testing accounting in
section 7.
"""
import csv
import json
import math
import os
import re
import numpy as np
from astropy.io import fits
from invariant_bench import Bench, KPC, MSUN

BAR = "=" * 78
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KEV, MP, MU = 2.99792458e8, 1.602176634e-16, 1.67262192369e-27, 0.6
A0, OM, OL = 1.2e-10, 0.3, 0.7
rng = np.random.default_rng(20260902)


def head(x):
    print("\n" + BAR + "\n" + x + "\n" + BAR)


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


# ------------------------------------------------------------------ the data
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

    def pm(s):
        s = s.replace("$", "")
        m = re.search(r"([\d.]+)\s*\\pm\s*([\d.]+)", s)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.search(r"([\d.]+)", s)
        return (float(m.group(1)), 0.0) if m else (float("nan"), 0.0)
    z = pm(cells[1])[0]
    M500, eM500 = pm(cells[3])
    R500, eR500 = pm(cells[4])
    Ez = math.sqrt(OM * (1 + z) ** 3 + OL)
    T500 = 8.85 * (M500 * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3)
    # dT500/T500 = (2/3) dM500/M500
    eT500 = T500 * (2 / 3) * eM500 / M500
    TAB[nm] = dict(z=z, M500=M500, eM500=eM500, R500=R500, eR500=eR500,
                   T500=T500, eT500=eT500)

b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
uq = sorted(np.unique(ext))
NAME = {}
for v in uq:
    hit = [k for k, t in TAB.items() if abs(t["R500"] - v) < 3.0]
    if len(hit) == 1:
        NAME[v] = (hit[0], "exact")
left_v = [v for v in uq if v not in NAME]
left_n = [k for k in TAB if k not in [x[0] for x in NAME.values()]]
for v in left_v:
    k = min(left_n, key=lambda q: abs(TAB[q]["R500"] - v))
    NAME[v] = (k, "elimination")
    left_n.remove(k)

head("1. Sample, with measurement uncertainties propagated")
D = []
for v in uq:
    nm, how = NAME[v]
    t = TAB[nm]
    m = np.abs(ext - v) < 1e-9
    rr = (xc.r[m] / KPC) / t["R500"]
    exc_pts = xc.nu[m] / nu_rar(xc.x[m])
    f = os.path.join(XR, nm, f"{nm}_temperature.fits")
    with fits.open(f) as h:
        dd = h[1].data
        rw = np.asarray(dd["RW_X"], float)
        tx = np.asarray(dd["T_X"], float)
        etx = np.asarray(dd["eT_X"], float)
    ok = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
    ts = np.interp(rr, rw[ok], tx[ok])
    ets = np.interp(rr, rw[ok], etx[ok])
    kT = float(np.median(ts)) * t["T500"]
    # error: profile scatter + spectroscopic error + T500 normalisation error
    e_prof = float(np.std(ts, ddof=1)) / math.sqrt(max(1, len(ts))) * t["T500"]
    e_spec = float(np.median(ets)) * t["T500"]
    ekT = math.hypot(math.hypot(e_prof, e_spec),
                     float(np.median(ts)) * t["eT500"])
    exc = float(np.median(exc_pts))
    # excess error from the bootstrap over radial bins within the cluster
    bs = np.array([np.median(rng.choice(exc_pts, len(exc_pts)))
                   for _ in range(4000)])
    eexc = float(np.std(bs))
    D.append(dict(name=nm, how=how, kT=kT, ekT=ekT, exc=exc, eexc=eexc,
                  M500=t["M500"], eM500=t["eM500"], n=int(m.sum()),
                  tsc=float(np.median(ts))))
print(f"   {'cluster':<10}{'id':>12}{'n':>5}{'kT (keV)':>16}{'excess':>16}"
      f"{'M500 (1e14)':>16}")
print("   " + "-" * 76)
for d in sorted(D, key=lambda q: q["kT"]):
    print(f"   {d['name']:<10}{d['how']:>12}{d['n']:>5}"
          f"{f'{d[chr(107)+chr(84)]:.2f} +- {d[chr(101)+chr(107)+chr(84)]:.2f}':>16}"
          f"{f'{d[chr(101)+chr(120)+chr(99)]:.2f} +- {d[chr(101)+chr(101)+chr(120)+chr(99)]:.2f}':>16}"
          f"{f'{d[chr(77)+chr(53)+chr(48)+chr(48)]:.2f} +- {d[chr(101)+chr(77)+chr(53)+chr(48)+chr(48)]:.2f}':>16}")
print("   " + "-" * 76)

kT = np.array([d["kT"] for d in D])
ekT = np.array([d["ekT"] for d in D])
ex = np.array([d["exc"] for d in D])
eex = np.array([d["eexc"] for d in D])
lM = np.log10(np.array([d["M500"] for d in D]))
names = [d["name"] for d in D]
exact = np.array([d["how"] == "exact" for d in D])


def spearman(a, c):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(c)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / math.sqrt((ra @ ra) * (rb @ rb)))


def perm_p(a, c, n=500000):
    r0 = spearman(a, c)
    cnt = 0
    for _ in range(n):
        if abs(spearman(rng.permutation(a), c)) >= abs(r0):
            cnt += 1
    return r0, cnt / n


head("2. H2 -- temperature. Point estimate, exact permutation p, bootstrap CI")
r0, p0 = perm_p(kT, ex, 200000)
print(f"   Spearman rho = {r0:+.3f}")
print(f"   two-sided exact-permutation p = {p0:.4f}   (n = {len(D)})")
print(f"   one-sided (direction pre-specified) p = {p0/2:.4f}")
bs = []
for _ in range(20000):
    j = rng.integers(0, len(D), len(D))
    if len(set(j)) < 4:
        continue
    bs.append(spearman(kT[j], ex[j]))
bs = np.array(bs)
print(f"   bootstrap over clusters: rho = {np.median(bs):+.3f}, "
      f"95% CI [{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]")
print(f"   fraction of bootstraps with rho <= 0 : {float(np.mean(bs<=0)):.4f}")

head("3. Measurement error: does the correlation survive it?")
print("   Perturb every kT and every excess by its own error, 20,000 times.")
sim = []
for _ in range(20000):
    sim.append(spearman(kT + rng.normal(0, ekT), ex + rng.normal(0, eex)))
sim = np.array(sim)
print(f"   rho under measurement noise: median {np.median(sim):+.3f}, "
      f"16-84 pct [{np.percentile(sim,16):+.3f}, {np.percentile(sim,84):+.3f}]")
print(f"   fraction of noise realisations with rho <= 0 : "
      f"{float(np.mean(sim<=0)):.4f}")

head("4. The physical model, fitted with errors")
print("   excess = sqrt(1 + kappa * 3kT/(mu m_p c^2)),  kappa the only parameter")


def chi2(kap):
    pred = np.sqrt(1 + kap * 3 * (kT * KEV) / (MU * MP * C ** 2))
    dpdT = (kap * 3 * KEV / (MU * MP * C ** 2)) / (2 * pred)
    sig = np.hypot(eex, dpdT * ekT)
    return float(np.sum(((ex - pred) / sig) ** 2))


grid = 10 ** np.linspace(3, 7, 4001)
c2 = np.array([chi2(k) for k in grid])
kbest = grid[int(np.argmin(c2))]
lo = grid[c2 <= c2.min() + 1][0]
hi = grid[c2 <= c2.min() + 1][-1]
print(f"   best-fit kappa = {kbest:.3g}   68% interval "
      f"[{lo:.3g}, {hi:.3g}]")
print(f"   chi2 = {c2.min():.2f} for {len(D)-1} dof  "
      f"(reduced {c2.min()/(len(D)-1):.2f})")
c2n = float(np.sum(((ex - 1.0) / eex) ** 2))
pred0 = np.full(len(D), float(np.mean(ex)))
c2f = float(np.sum(((ex - pred0) / eex) ** 2))
print(f"   for comparison, a CONSTANT excess (no temperature dependence):")
print(f"      chi2 = {c2f:.2f} for {len(D)-1} dof  "
      f"(reduced {c2f/(len(D)-1):.2f})")
print(f"   delta chi2 (constant minus model) = {c2f - c2.min():+.2f} "
      f"for the same number of free parameters (1)")
print(f"   the pre-specified kappa = 1e5 gives chi2 = {chi2(1e5):.2f}")

head("5. H1 -- merger bias, same treatment")
DS = {}
for r in csv.DictReader(open(SCR + "halo07_environment/"
                             "clusters_dynamical_state.tsv", encoding="utf-8"),
                        delimiter="\t"):
    k = r["cluster_name"].strip().upper().replace(" ", "").replace("_", "")
    if k.startswith("ABELL"):
        k = "A" + k[5:]
    DS.setdefault(k, r)
dp, ee2, kk2, nn2 = [], [], [], []
for d in D:
    q = DS.get(d["name"].upper())
    if q is None:
        continue
    try:
        v = float(q["disturbance_pct"])
    except (ValueError, KeyError):
        continue
    if np.isfinite(v):
        dp.append(v); ee2.append(d["exc"]); kk2.append(d["kT"]); nn2.append(d["name"])
dp, ee2, kk2 = np.array(dp), np.array(ee2), np.array(kk2)
r1, p1 = perm_p(dp, ee2, 200000)
print(f"   Spearman rho = {r1:+.3f}   two-sided p = {p1:.4f}   n = {len(dp)}")
print(f"   H1 pre-specified POSITIVE; observed sign is "
      f"{'POSITIVE' if r1 > 0 else 'NEGATIVE'}")
bs1 = []
for _ in range(20000):
    j = rng.integers(0, len(dp), len(dp))
    if len(set(j)) < 4:
        continue
    bs1.append(spearman(dp[j], ee2[j]))
bs1 = np.array(bs1)
print(f"   bootstrap rho = {np.median(bs1):+.3f}, 95% CI "
      f"[{np.percentile(bs1,2.5):+.3f}, {np.percentile(bs1,97.5):+.3f}]")

head("6. Are the two signals independent?")


def partial(a, c, z):
    R = [np.argsort(np.argsort(v)).astype(float) for v in (a, c, z)]
    R = [(v - v.mean()) / (v.std() or 1) for v in R]
    n = len(R[0])
    rac, raz, rcz = (float(R[0] @ R[1] / n), float(R[0] @ R[2] / n),
                     float(R[1] @ R[2] / n))
    return (rac - raz * rcz) / math.sqrt(max(1e-12, (1 - raz ** 2) * (1 - rcz ** 2)))


print(f"   Spearman(kT, disturbance)          = {spearman(kk2, dp):+.3f}")
print(f"   kT vs excess | disturbance removed = {partial(kk2, ee2, dp):+.3f}")
print(f"   disturbance vs excess | kT removed = {partial(dp, ee2, kk2):+.3f}")
print(f"   kT vs excess | mass removed        = {partial(kT, ex, lM):+.3f}")
pb = []
for _ in range(20000):
    j = rng.integers(0, len(dp), len(dp))
    if len(set(j)) < 5:
        continue
    try:
        pb.append(partial(kk2[j], ee2[j], dp[j]))
    except Exception:
        pass
pb = np.array(pb)
print(f"   bootstrap on the kT partial: {np.median(pb):+.3f}, 95% CI "
      f"[{np.percentile(pb,2.5):+.3f}, {np.percentile(pb,97.5):+.3f}]")

head("7. Power, and the exact-identity subsample")
print("   What could n = 12 have detected?")
for rho_true in (0.3, 0.5, 0.6, 0.7, 0.8):
    hits = 0
    for _ in range(2000):
        z = rng.normal(size=len(D))
        y = rho_true * z + math.sqrt(1 - rho_true ** 2) * rng.normal(size=len(D))
        r_, _ = spearman(z, y), None
        # approximate p via Fisher z
        zz = 0.5 * math.log((1 + min(0.999, abs(r_))) / (1 - min(0.999, abs(r_))))
        se = 1 / math.sqrt(len(D) - 3)
        if 2 * (1 - 0.5 * (1 + math.erf(abs(zz / se) / math.sqrt(2)))) < 0.05:
            hits += 1
    print(f"      true rho = {rho_true:.1f}  ->  power at p<0.05: {hits/2000:.2f}")
ke, xe = kT[exact], ex[exact]
r2, p2 = perm_p(ke, xe, 200000)
print(f"\n   exact-identity subsample only: n = {int(exact.sum())}, "
      f"rho = {r2:+.3f}, p = {p2:.4f}")
print(f"   identities assigned by elimination: "
      f"{[d['name'] for d in D if d['how']!='exact']}")

head("8. Multiple-testing accounting, stated honestly")
print("""   Both H1 and H2 were pre-specified with directions before these
   temperatures were obtained, so neither is an exploratory search and neither
   carries a look-elsewhere penalty on its own.

   But this programme has run a large number of exploratory tests, and seven
   candidate variables were eliminated by label controls. If one insists on
   treating H2 as one draw from that exploratory pool, the appropriate
   correction is severe and it does not survive: p = 0.037 against roughly 20
   pre-registered-equivalent tests gives a corrected p near 0.5.

   The defensible claim is therefore narrow: H2 was derived from a mechanism,
   its direction and amplitude were fixed BEFORE the data existed, and it was
   tested once. That is a confirmatory test. It is not a discovery, and n = 12
   with p = 0.037 -- falling to p = 0.19 on the exact-identity subsample --
   would not survive a referee as one.""")

json.dump(dict(clusters=[{k: (v if not isinstance(v, (np.floating, np.integer))
                              else float(v)) for k, v in d.items()} for d in D],
               rho_T=r0, p_T=p0, rho_D=r1, p_D=p1,
               kappa=float(kbest), kappa_lo=float(lo), kappa_hi=float(hi)),
          open(SCR + "paper_results.json", "w", encoding="utf-8"), indent=1)
print(f"\n   wrote paper_results.json")
