"""
Extract real potential wells for a spread of galaxies and clusters.

The height field in the visualisation is the NEWTONIAN POTENTIAL Phi(r), which
is the honest choice for the weak field: orbits follow g = -grad Phi, and light
deflection in GR without slip is governed by the same Phi. So one surface
answers both questions the user asked -- how stars move and how light bends.

Two surfaces per object:

    Phi_bar   built by integrating the acceleration the VISIBLE matter makes
    Phi_obs   built by integrating the acceleration the MOTION actually shows

Both are anchored Phi(r_max) = 0 and integrated inward, so only measured data
enters -- no extrapolation, no assumption about what lies outside the last
data point. Depths are therefore relative to the outermost measured radius,
which is stated in the tool.
"""
import json
import math
import os
import numpy as np
from astropy.io import fits

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
XR = ROOT + "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/"
KPC = 3.0856775814913673e19
KMS = 1e3
MSUN = 1.98892e30
G = 6.674e-11
C = 2.99792458e8
BAR = "=" * 78


def well(r_m, g):
    """Phi(r) = -int_r^rmax g dr', anchored to zero at the outermost radius."""
    o = np.argsort(r_m)
    r, gg = r_m[o], g[o]
    seg = 0.5 * (gg[1:] + gg[:-1]) * np.diff(r)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    return o, (cum - cum[-1])            # <= 0, deepest at small r


OBJ = []

print(BAR + "\nSPARC galaxies\n" + BAR)
cfg = json.load(open(ROOT + "configs/sparc_rotation_curves_full_v1.json",
                     encoding="utf-8"))
gals = []
for gal in cfg["galaxies"]:
    rows = [[float(q) for q in w] for w in gal["rows"]]
    R, VO, VB = [], [], []
    for q in rows:
        r0, vo, ev = q[0], q[1], q[2]
        cg = q[3] * abs(q[3]); cd = 0.5 * q[4] ** 2; cb = 0.7 * q[5] ** 2
        v2 = cg + cd + cb
        if r0 <= 0 or vo <= 0 or ev <= 0 or v2 <= 0:
            continue
        R.append(r0); VO.append(vo); VB.append(math.sqrt(v2))
    if len(R) < 8:
        continue
    R, VO, VB = np.array(R), np.array(VO), np.array(VB)
    gals.append(dict(name=gal.get("name", gal.get("galaxy", "?")),
                     R=R, VO=VO, VB=VB,
                     ratio=float(np.median(VO ** 2 / VB ** 2)),
                     vflat=float(np.median(VO[-3:]))))
print(f"   {len(gals)} galaxies with >= 8 usable points")
gals.sort(key=lambda q: q["ratio"])
# span the discrepancy range: weakest, median, strongest, plus big rotators
pick = [gals[0], gals[len(gals) // 6], gals[len(gals) // 3],
        gals[len(gals) // 2], gals[2 * len(gals) // 3],
        gals[5 * len(gals) // 6], gals[-1]]
big = sorted(gals, key=lambda q: -q["vflat"])[:3]
for gsel in big:
    if gsel["name"] not in [p["name"] for p in pick]:
        pick.append(gsel)
print(f"\n   {'galaxy':<14}{'n':>4}{'r_max kpc':>11}{'v_flat':>9}"
      f"{'median g_obs/g_bar':>20}")
print("   " + "-" * 58)
for gsel in pick:
    R, VO, VB = gsel["R"], gsel["VO"], gsel["VB"]
    rm = R * KPC
    go = (VO * KMS) ** 2 / rm
    gb = (VB * KMS) ** 2 / rm
    o, po = well(rm, go)
    _, pb = well(rm, gb)
    print(f"   {gsel['name']:<14}{len(R):>4}{R.max():>11.1f}"
          f"{gsel['vflat']:>9.0f}{gsel['ratio']:>20.2f}")
    OBJ.append(dict(
        name=gsel["name"], kind="galaxy",
        r=[round(float(x), 4) for x in R[o]],
        v_obs=[round(float(x), 2) for x in VO[o]],
        v_bar=[round(float(x), 2) for x in VB[o]],
        g_obs=[float(f"{x:.6g}") for x in go[o]],
        g_bar=[float(f"{x:.6g}") for x in gb[o]],
        phi_obs=[float(f"{x/1e6:.6g}") for x in po],
        phi_bar=[float(f"{x/1e6:.6g}") for x in pb],
        ratio=round(gsel["ratio"], 3), vflat=round(gsel["vflat"], 1)))
print("   " + "-" * 58)

print("\n" + BAR + "\nX-COP clusters\n" + BAR)
ID = {float(k): v for k, v in
      json.load(open(SCR + "xcop_identity.json", encoding="utf-8")).items()}
EXTRA = {1250.0: dict(name="A644", M500=5.66, R500=1230),
         1368.0: dict(name="A2319", M500=7.31, R500=1346)}
ID.update(EXTRA)
from invariant_bench import Bench
b = Bench(verbose=False)
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
print(f"   {'cluster':<10}{'n':>4}{'r kpc':>18}{'R500':>8}"
      f"{'median g_obs/g_bar':>20}")
print("   " + "-" * 62)
for v in sorted(np.unique(ext)):
    key = min(ID, key=lambda k: abs(k - v))
    if abs(key - v) > 3:
        continue
    info = ID[key]
    m = np.abs(ext - v) < 1e-9
    rm = xc.r[m]
    go, gb = xc.go[m], xc.gb[m]
    o, po = well(rm, go)
    _, pb = well(rm, gb)
    rk = rm[o] / KPC
    print(f"   {info['name']:<10}{int(m.sum()):>4}"
          f"{f'{rk.min():.0f} - {rk.max():.0f}':>18}{info['R500']:>8.0f}"
          f"{float(np.median(go/gb)):>20.2f}")
    OBJ.append(dict(
        name=info["name"], kind="cluster",
        r=[round(float(x), 2) for x in rk],
        v_obs=[round(float(math.sqrt(max(g, 0) * r)) / 1e3, 1)
               for g, r in zip(go[o], rm[o])],
        v_bar=[round(float(math.sqrt(max(g, 0) * r)) / 1e3, 1)
               for g, r in zip(gb[o], rm[o])],
        g_obs=[float(f"{x:.6g}") for x in go[o]],
        g_bar=[float(f"{x:.6g}") for x in gb[o]],
        phi_obs=[float(f"{x/1e6:.6g}") for x in po],
        phi_bar=[float(f"{x/1e6:.6g}") for x in pb],
        ratio=round(float(np.median(go / gb)), 3),
        R500=float(info["R500"]), M500=float(info["M500"])))
print("   " + "-" * 62)

json.dump(OBJ, open(SCR + "wells.json", "w", encoding="utf-8"))
sz = os.path.getsize(SCR + "wells.json") / 1024
print(f"\n   wrote wells.json  ({len(OBJ)} objects, {sz:.0f} KB)")
print(f"   galaxies: {sum(1 for o in OBJ if o['kind']=='galaxy')}   "
      f"clusters: {sum(1 for o in OBJ if o['kind']=='cluster')}")
