"""
THE DECISIVE TEST: absolute X-COP temperatures, and does the excess follow them?

Acquired: Ghirardini et al. 2019, A&A 621, A41 (arXiv:1805.00042), LaTeX source.
Table 1 gives M_500 and redshift for all twelve X-COP clusters, and Eq. 6 gives
the normalisation the release's scaled profiles use:

    T_500 = 8.85 keV * (M_500 / 1e15 Msun)^(2/3) * E(z)^(2/3) * (mu/0.6)

The on-disk profiles store (T/T_500)_X, so absolute temperature per radius is

    kT(r) = (T/T_500)_X(r) * T_500(M_500, z)

CAVEAT STATED UP FRONT, because it decides how the answer must be read:
T_500 depends on M_500^(2/3). So cluster-to-cluster variation in absolute
temperature is PART mass and PART the measured profile shape. That is exactly
the confound that has killed every previous candidate, so the test is run three
ways:

    (a) against absolute kT               -- partly mass
    (b) against kT with mass partialled out
    (c) against the MEASURED scaled shape (T/T_500), which contains no mass at all

Only (b) and (c) can distinguish the pressure mechanism from mass in disguise.
"""
import math
import os
import re
import numpy as np
from astropy.io import fits

BAR = "=" * 78
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
      "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/")
C, KB, MP, MU = 2.99792458e8, 1.380649e-23, 1.67262192369e-27, 0.6
KEV, KPC, A0, MSUN = 1.602176634e-16, 3.0856775814913673e19, 1.2e-10, 1.98892e30
OM, OL = 0.3, 0.7
rng = np.random.default_rng(31)


def head(x):
    print("\n" + BAR + "\n" + x + "\n" + BAR)


def nu_rar(x):
    return 1.0 / (1.0 - np.exp(-np.sqrt(np.maximum(x, 1e-300))))


head("1. Parsing Table 1 of Ghirardini et al. 2019 from the LaTeX source")
tex = open(SCR + "xcop_T/XCOP_thermo.tex", encoding="utf-8").read()
i = tex.index("Basic properties of the X-COP sample")
blk = tex[i:i + 4000]
SAMP = {}
for line in blk.split("\n"):
    if "&" not in line or line.strip().startswith("%"):
        continue
    m = re.match(r"\s*([A-Za-z0-9]+)\s*&\s*([\d.]+)\s*&\s*([\d.]+)\s*&\s*"
                 r"\$([\d.]+)\s*\\pm\s*([\d.]+)\$\s*&\s*\$(\d+)", line)
    if m:
        nm, z, sn, m500, em500, r500 = m.groups()
        SAMP[nm] = dict(z=float(z), M500=float(m500), eM500=float(em500),
                        R500=float(r500))
print(f"   parsed {len(SAMP)} clusters")
print(f"\n   {'cluster':<10}{'z':>9}{'M500 (1e14)':>14}{'R500 kpc':>11}"
      f"{'T500 keV':>11}")
print("   " + "-" * 56)
for nm, v in sorted(SAMP.items()):
    Ez = math.sqrt(OM * (1 + v["z"]) ** 3 + OL)
    v["T500"] = 8.85 * (v["M500"] * 1e14 / 1e15) ** (2 / 3) * Ez ** (2 / 3)
    print(f"   {nm:<10}{v['z']:>9.4f}{v['M500']:>14.2f}{v['R500']:>11}"
          f"{v['T500']:>11.2f}")
print("   " + "-" * 56)
print("   T500 from Eq. 6 of the paper; these are the standard X-COP")
print("   temperatures and land where the literature puts them (4-9 keV).")

head("2. Absolute temperature profiles")
PROF = {}
for nm in SAMP:
    f = os.path.join(XR, nm, f"{nm}_temperature.fits")
    if not os.path.exists(f):
        continue
    with fits.open(f) as h:
        d = h[1].data
        rw = np.asarray(d["RW_X"], float)
        tx = np.asarray(d["T_X"], float)
    m = np.isfinite(rw) & np.isfinite(tx) & (tx > 0)
    PROF[nm] = dict(rw=rw[m], tscaled=tx[m], kT=tx[m] * SAMP[nm]["T500"])
print(f"   profiles read: {len(PROF)}")
print(f"\n   {'cluster':<10}{'n':>5}{'T/T500 range':>20}{'kT keV range':>20}"
      f"{'median kT':>12}")
print("   " + "-" * 68)
for nm, p in sorted(PROF.items()):
    print(f"   {nm:<10}{len(p['kT']):>5}"
          f"{f'{p[chr(116)+chr(115)+chr(99)+chr(97)+chr(108)+chr(101)+chr(100)].min():.2f} - {p[chr(116)+chr(115)+chr(99)+chr(97)+chr(108)+chr(101)+chr(100)].max():.2f}':>20}"
          f"{f'{p[chr(107)+chr(84)].min():.2f} - {p[chr(107)+chr(84)].max():.2f}':>20}"
          f"{np.median(p['kT']):>12.2f}")
print("   " + "-" * 68)

head("3. Matching to the measured excess")
from invariant_bench import Bench
b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc))
uq = sorted(np.unique(ext))
RM = {}
for nm in SAMP:
    f = os.path.join(XR, nm, f"{nm}_density_L1.fits")
    if os.path.exists(f):
        with fits.open(f) as h:
            RM[nm] = float(np.nanmax(np.asarray(h[1].data["R_OUT"], float)))
order = [k for k, _ in sorted(RM.items(), key=lambda t: t[1])]
ROWS = []
for j, v in enumerate(uq):
    if j >= len(order):
        break
    nm = order[j]
    if nm not in PROF:
        continue
    m = ext == v
    r_kpc = xc.r[m] / KPC
    exc = xc.nu[m] / nu_rar(xc.x[m])
    r500 = SAMP[nm]["R500"]
    # temperature at each bench radius, from the measured profile
    kT_at = np.interp(r_kpc / r500, PROF[nm]["rw"], PROF[nm]["kT"])
    ts_at = np.interp(r_kpc / r500, PROF[nm]["rw"], PROF[nm]["tscaled"])
    ROWS.append(dict(name=nm, exc=float(np.median(exc)),
                     kT=float(np.median(kT_at)),
                     tscaled=float(np.median(ts_at)),
                     T500=SAMP[nm]["T500"], M500=SAMP[nm]["M500"],
                     n=int(m.sum())))
print(f"   {len(ROWS)} clusters matched")
print(f"\n   {'cluster':<10}{'median kT':>11}{'T500':>9}{'T/T500':>9}"
      f"{'M500':>9}{'excess':>10}")
print("   " + "-" * 60)
for r in sorted(ROWS, key=lambda q: q["kT"]):
    print(f"   {r['name']:<10}{r['kT']:>11.2f}{r['T500']:>9.2f}"
          f"{r['tscaled']:>9.3f}{r['M500']:>9.2f}{r['exc']:>10.2f}")
print("   " + "-" * 60)

head("4. THE TEST, three ways")
kT = np.array([r["kT"] for r in ROWS])
ts = np.array([r["tscaled"] for r in ROWS])
lM = np.log10(np.array([r["M500"] for r in ROWS]))
ex = np.array([r["exc"] for r in ROWS])


def spear(a, c, nperm=200000):
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(c)).astype(float)
    ra -= ra.mean(); rb -= rb.mean()
    r0 = float((ra @ rb) / math.sqrt((ra @ ra) * (rb @ rb)))
    nl = np.array([abs(float(np.corrcoef(
        np.argsort(np.argsort(rng.permutation(a))), rb)[0, 1]))
        for _ in range(nperm)])
    return r0, float(np.mean(nl >= abs(r0)))


def partial(a, c, z):
    R = [np.argsort(np.argsort(v)).astype(float) for v in (a, c, z)]
    R = [(v - v.mean()) / (v.std() or 1) for v in R]
    n = len(R[0])
    rac, raz, rcz = (float(R[0] @ R[1] / n), float(R[0] @ R[2] / n),
                     float(R[1] @ R[2] / n))
    return (rac - raz * rcz) / math.sqrt(max(1e-12, (1 - raz ** 2) * (1 - rcz ** 2)))


print("   The pressure model requires a POSITIVE correlation of excess with kT.")
r1, p1 = spear(kT, ex)
r2, p2 = spear(ts, ex)
r3, p3 = spear(lM, ex)
print(f"\n   (a) absolute kT vs excess          rho = {r1:+.3f}   p = {p1:.4f}")
print(f"   (b) same, mass partialled out      rho = {partial(kT, ex, lM):+.3f}")
print(f"   (c) MEASURED T/T500 vs excess      rho = {r2:+.3f}   p = {p2:.4f}")
print(f"\n   control: log M500 vs excess        rho = {r3:+.3f}   p = {p3:.4f}")
print(f"   confounding: kT vs log M500        rho = {spear(kT, lM, 2000)[0]:+.3f}")
print("\n   (c) is the cleanest: T/T500 is the measured profile shape and")
print("   contains no mass information at all.")

head("5. Does the SIZE match, with kappa already fixed?")
kap = 1e5
pred = np.sqrt(1 + kap * 3 * (kT * KEV) / (MU * MP * C ** 2))
print(f"   kappa = 1e5 was fixed by the galaxy-cluster comparison. No freedom.")
print(f"\n   {'cluster':<10}{'kT':>8}{'predicted':>12}{'observed':>11}"
      f"{'ratio':>9}")
print("   " + "-" * 52)
for r, p_ in sorted(zip(ROWS, pred), key=lambda t: t[0]["kT"]):
    print(f"   {r['name']:<10}{r['kT']:>8.2f}{p_:>12.2f}{r['exc']:>11.2f}"
          f"{r['exc']/p_:>9.2f}")
print("   " + "-" * 52)
rr, pp = spear(pred, ex)
print(f"\n   Spearman(predicted, observed) = {rr:+.3f}   p = {pp:.4f}")
print(f"   predicted spread {pred.max()/pred.min():.2f}x   "
      f"observed spread {ex.max()/ex.min():.2f}x")
print(f"   median |log10(observed/predicted)| = "
      f"{np.median(np.abs(np.log10(ex/pred))):.4f} dex")

head("VERDICT")
pb = partial(kT, ex, lM)
print(f"""   Twelve X-COP clusters, absolute temperatures from the published
   normalisation, one instrument, one pipeline, all of them clusters -- so no
   dataset label can act.

      absolute kT vs excess                {r1:+.3f}  (p = {p1:.4f})
      with mass removed                    {pb:+.3f}
      measured T/T500 vs excess            {r2:+.3f}  (p = {p2:.4f})
      size test, kappa fixed               {rr:+.3f}  (p = {pp:.4f})

   The pressure model requires all of these to be POSITIVE. This is the test
   it could fail, run on the data it needed, with no parameter left free.""")
