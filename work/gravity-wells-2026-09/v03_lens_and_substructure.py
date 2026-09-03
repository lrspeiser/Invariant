"""
TWO ADDITIONS, both computed from real data before anything is drawn.

(1) LENSING. The tool currently draws one bent ray, which is not what a
    telescope sees. What it sees is a distorted sky, and that needs the
    PROJECTED surface density:

        kappa(R) = Sigma(R)/Sigma_crit,  Sigma_crit = c^2 D_S/(4 pi G D_L D_LS)

    deflection alpha(theta) = 2 <kappa>(<theta) theta, lens equation
    beta = theta - alpha, Einstein radius where <kappa> = 1.

    A NUMERICAL TRAP, hit on the first attempt: Sigma needs rho, rho needs
    dM/dr, and the hydrostatic M(r) is NOT monotonic -- A2319 has 21 of 54
    points with negative slope and swings between -8.7 and +8.8. Differencing
    it directly gave that one cluster kappa_0 = 13.4 against ~0.2 for every
    other, and a fake 121-arcsec Einstein radius. Fixed by imposing the
    physical constraint (enclosed mass cannot decrease), fitting a smooth
    log-log form, and differentiating that.

(2) SUBSTRUCTURE. A cluster is NOT one smooth well -- it is hundreds of
    galaxies each with its own, inside a common larger one. The X-COP profiles
    cannot show that: being hydrostatic, they are spherically averaged by
    construction. So member catalogues are used instead -- AXES-SDSS groups
    (Damsted+ 2024), with per-member sky position, redshift and r-band
    luminosity. Stellar mass from luminosity, total mass from the member
    velocity dispersion. Both observational; neither assumes dark matter.
"""
import glob
import json
import math
import os
import numpy as np

SCR = ("C:/Users/henry/AppData/Local/Temp/claude/C--Users-henry-dev/"
       "a2309145-5e60-4815-97f2-bb0c877edc0d/scratchpad/")
INV = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
AX = INV + "work/item2-axes-groups-v5-raw-acquisition/"
G, C = 6.674e-11, 2.99792458e8
KPC = 3.0856775814913673e19
MSUN = 1.98892e30
H0 = 70e3 / (1000 * KPC)
OM, OL = 0.3, 0.7
UPS_R = 2.5
BAR = "=" * 78


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def Dc(z, n=3000):
    zz = np.linspace(0, z, n)
    return (C / H0) * np.trapezoid(1 / np.sqrt(OM * (1 + zz) ** 3 + OL), zz)


def DA(z):
    return Dc(z) / (1 + z)


def DA12(z1, z2):
    return (Dc(z2) - Dc(z1)) / (1 + z2)


def density_from_mass(r, M3, deg=4):
    """rho(r) from an enclosed-mass profile, robustly.

    Enclosed mass is physically non-decreasing, so impose that first; then fit
    a low-order polynomial in log-log and differentiate the FIT, never the
    noisy samples. Slope is clipped to [0.02, 3.5]: below 0 means negative
    density, above ~3 cannot hold over a decade.
    """
    Mm = np.maximum.accumulate(np.maximum(M3, 1e-30))
    x, y = np.log(r), np.log(Mm)
    c = np.polyfit(x, y, deg)
    sl = np.clip(np.polyval(np.polyder(c), x), 0.02, 3.5)
    Msm = np.exp(np.polyval(c, x))
    rho = Msm * sl / (4 * math.pi * r ** 3)
    return rho, sl, Msm


def outer_slope(r, rho):
    """Log-slope of rho over the outer third, forced to be physical.

    The tail must fall faster than r^-2 or the enclosed mass diverges and the
    line-of-sight integral never converges. Taking the slope from the global
    polynomial let it clip at +3.5, i.e. density RISING outward -- which gave
    A85 a flat kappa = 5.4 across two decades and RXC1825 a flat 1.5.
    """
    k = r > r[-1] * 0.35
    if k.sum() < 4:
        k = np.arange(len(r)) >= len(r) - 4
    pl = np.polyfit(np.log(r[k]), np.log(np.maximum(rho[k], 1e-300)), 1)[0]
    return float(np.clip(pl, -5.0, -2.0))


def project(r, rho, p_out, Rg_kpc):
    """Sigma(R) by line-of-sight integration, power-law continued outside."""
    Sig = np.empty(len(Rg_kpc))
    for i, Rk in enumerate(Rg_kpc):
        R = Rk * KPC
        zz = np.logspace(-3, math.log10(60 * r[-1] / R), 600) * R
        rr = np.hypot(R, zz)
        rh = np.interp(rr, r, rho)
        out = rr > r[-1]
        if out.any():
            rh[out] = rho[-1] * (rr[out] / r[-1]) ** p_out
        rh[rr < r[0]] = rho[0]
        Sig[i] = 2 * np.trapezoid(rh, zz)
    return Sig


def lens_profile(r, g, Scr, dl):
    """Return R grid (kpc), kappa, mean-kappa-inside, and Einstein radius."""
    M3 = g * r ** 2 / G
    rho, sl, _ = density_from_mass(r, M3)
    Rg = np.logspace(math.log10(r[0] / KPC * 1.05),
                     math.log10(r[-1] / KPC * 0.95), 100)
    Sig = project(r, rho, outer_slope(r, rho), Rg)
    Rm = Rg * KPC
    ring = 2 * math.pi * Rm * Sig
    Mp = np.concatenate([[0.0], np.cumsum(0.5 * (ring[1:] + ring[:-1]) * np.diff(Rm))])
    Mp += Sig[0] * math.pi * Rm[0] ** 2            # inner disc
    kbar = Mp / (math.pi * Rm ** 2 * Scr)
    thE = np.nan
    for i in range(len(kbar) - 1):
        if (kbar[i] - 1) * (kbar[i + 1] - 1) < 0:
            f = (1 - kbar[i]) / (kbar[i + 1] - kbar[i])
            thE = (Rm[i] + f * (Rm[i + 1] - Rm[i])) / dl * 206265
            break
    return Rg, Sig / Scr, kbar, thE


head("1. LENSING -- what a telescope would actually see")
from invariant_bench import Bench
b = Bench(verbose=False)
ZS = 2.0
print(f"   source plane fixed at z_s = {ZS}\n")

TARGETS = []
xc = b.d["xcop"]
ext = np.asarray(xc.extent, float) * np.ones(len(xc)) / KPC
ID = {float(k): v for k, v in
      json.load(open(SCR + "xcop_identity.json", encoding="utf-8")).items()}
ID[1250.0] = dict(name="A644", z=0.0704, M500=5.66, R500=1230)
ID[1368.0] = dict(name="A2319", z=0.0557, M500=7.31, R500=1346)
for v in sorted(np.unique(ext)):
    key = min(ID, key=lambda k: abs(k - v))
    if abs(key - v) > 3:
        continue
    info = ID[key]
    m = np.abs(ext - v) < 1e-9
    o = np.argsort(xc.r[m])
    TARGETS.append((info["name"], float(info["z"]),
                    xc.r[m][o], xc.go[m][o], xc.gb[m][o]))
# CLASH: the stacked lensing-selected profile. Those clusters are at
# z ~ 0.19-0.89; 0.35 is representative and is stated as an assumption.
cl = b.d["clash"]
o = np.argsort(cl.r)
TARGETS.append(("CLASH stack", 0.35, cl.r[o], cl.go[o], cl.gb[o]))

print(f"   {'lens':<13}{'z_l':>7}{'kap_max':>9}{'thE req':>10}{'thE vis':>10}"
      f"{'strong?':>9}")
print("   " + "-" * 60)
LENS = {}
for name, zl, r, go, gb in TARGETS:
    dl, ds, dls = DA(zl), DA(ZS), DA12(zl, ZS)
    Scr = C ** 2 * ds / (4 * math.pi * G * dl * dls)
    Rg, ko, kbo, teo = lens_profile(r, go, Scr, dl)
    _, kb, kbb, teb = lens_profile(r, gb, Scr, dl)
    strong = "YES" if np.isfinite(teo) else "no"
    print(f"   {name:<13}{zl:>7.3f}{ko.max():>9.2f}"
          f"{(f'{teo:.1f}\"' if np.isfinite(teo) else '--'):>10}"
          f"{(f'{teb:.1f}\"' if np.isfinite(teb) else '--'):>10}{strong:>9}")
    LENS[name] = dict(z=zl, dl_kpc=float(dl / KPC), Scr=float(Scr),
                      thE_obs=(float(teo) if np.isfinite(teo) else None),
                      thE_bar=(float(teb) if np.isfinite(teb) else None),
                      R=[round(float(x), 1) for x in Rg],
                      kap_obs=[float(f"{x:.5g}") for x in ko],
                      kap_bar=[float(f"{x:.5g}") for x in kb],
                      kbar_obs=[float(f"{x:.5g}") for x in kbo],
                      kbar_bar=[float(f"{x:.5g}") for x in kbb])
print("   " + "-" * 60)
print("""   kappa is the surface density in units of the critical value. Above 1
   the lens makes multiple images and full arcs; below it, only a subtle
   tangential stretch of every background galaxy.

   THE X-COP CLUSTERS ARE NOT STRONG LENSES, and that is not a defect in the
   data -- it is their redshift. Sigma_crit grows as the lens gets nearer, so
   a cluster at z ~ 0.06 needs far more mass to reach kappa = 1 than the same
   cluster at z ~ 0.4. Real Einstein-ring clusters sit at z ~ 0.2-0.6, which
   is what the CLASH row is for.""")

head("2. SUBSTRUCTURE -- is a cluster one well or many?")
META = {}
for line in open(INV + "work/item2-axes-groups-v5-audit/group-metadata-only.tsv",
                 encoding="utf-8"):
    q = line.rstrip("\n").split("\t")
    if len(q) >= 4 and q[0].strip().isdigit():
        try:
            META[int(q[0])] = dict(N=int(q[1]), z=float(q[2]))
        except ValueError:
            continue
files = sorted(glob.glob(AX + "members-*.tsv"))
print(f"   {len(files)} AXES-SDSS member catalogues, {len(META)} with metadata")

GROUPS = []
for f in files:
    gid = int(os.path.basename(f).split("-")[1].split(".")[0])
    ra, de, zs, lr = [], [], [], []
    for line in open(f, encoding="utf-8"):
        q = line.split("\t")
        if len(q) < 7 or not q[0].strip().isdigit():
            continue
        try:
            ra.append(float(q[3])); de.append(float(q[4]))
            zs.append(float(q[5])); lr.append(float(q[6]))
        except ValueError:
            continue
    if len(ra) < 25:
        continue
    ra, de, zs, lr = (np.array(x) for x in (ra, de, zs, lr))
    zc = float(np.median(zs))
    vp = C / 1e3 * (zs - zc) / (1 + zc)
    for _ in range(5):
        s = np.std(vp, ddof=1)
        k = np.abs(vp - np.mean(vp)) < 3 * s
        if k.all():
            break
        ra, de, zs, lr, vp = ra[k], de[k], zs[k], lr[k], vp[k]
    if len(ra) < 20:
        continue
    sig = float(np.std(vp, ddof=1))
    da = DA(zc) / KPC
    rac, dec = float(np.median(ra)), float(np.median(de))
    x = (ra - rac) * math.cos(math.radians(dec)) * math.pi / 180 * da
    y = (de - dec) * math.pi / 180 * da
    Rrms = float(np.sqrt(np.mean(x ** 2 + y ** 2)))
    Mstar = float(np.sum(lr) * 1e10 * UPS_R)
    Mdyn = float(3 * (sig * 1e3) ** 2 * (Rrms * KPC) / G / MSUN)
    GROUPS.append(dict(gid=gid, n=len(ra), z=zc, sigma=sig, Rrms=Rrms,
                       Mstar=Mstar, Mdyn=Mdyn, ratio=Mdyn / Mstar,
                       x=x, y=y, lr=lr))
GROUPS.sort(key=lambda q: -q["n"])
print(f"   {len(GROUPS)} groups with >= 20 clean members\n")
print(f"   {'group':>7}{'N':>5}{'z':>8}{'sigma':>7}{'R_rms':>7}{'log M*':>9}"
      f"{'log M_dyn':>11}{'ratio':>8}{'top gal':>9}")
print("   " + "-" * 71)
for g in GROUPS[:12]:
    print(f"   {g['gid']:>7}{g['n']:>5}{g['z']:>8.4f}{g['sigma']:>7.0f}"
          f"{g['Rrms']:>7.0f}{math.log10(g['Mstar']):>9.2f}"
          f"{math.log10(g['Mdyn']):>11.2f}{g['ratio']:>8.0f}"
          f"{100*g['lr'].max()/g['lr'].sum():>8.0f}%")
print("   " + "-" * 71)
print("   sigma km/s, R_rms kpc; 'top gal' = brightest member's share of stars.")

head("3. So how much of the well do the visible galaxies make?")
rr = np.array([g["ratio"] for g in GROUPS])
fb = np.array([g["lr"].max() / g["lr"].sum() for g in GROUPS])
print(f"   groups analysed                {len(GROUPS)}")
print(f"   median M_dyn / M_star          {np.median(rr):.0f}")
print(f"   16th-84th percentile           {np.percentile(rr,16):.0f} - "
      f"{np.percentile(rr,84):.0f}")
print(f"   visible galaxies are           {100/np.median(rr):.2f}% of the mass")
print(f"   brightest member, of stars     {100*np.median(fb):.0f}%")
print(f"   brightest member, of the total {100*np.median(fb)/np.median(rr):.3f}%")
print(f"""
   The answer is YES, a cluster genuinely has many wells -- every one of those
   galaxies has its own. But they are pinpricks. All the galaxies together are
   about {100/np.median(rr):.1f}% of the mass that has to be there, and the single deepest
   galaxy well is a few hundredths of one percent of the total.

   That is the point worth drawing: the lumps are real and they are visible,
   and the big smooth well they sit in is neither.""")

OUT = []
for g in GROUPS[:6]:
    keep = np.argsort(-g["lr"])[:140]
    OUT.append(dict(gid=g["gid"], n=int(g["n"]), z=round(g["z"], 5),
                    sigma=round(g["sigma"], 1), Rrms=round(g["Rrms"], 1),
                    Mstar=float(f"{g['Mstar']:.4g}"),
                    Mdyn=float(f"{g['Mdyn']:.4g}"), ratio=round(g["ratio"], 1),
                    mx=[round(float(v), 1) for v in g["x"][keep]],
                    my=[round(float(v), 1) for v in g["y"][keep]],
                    mL=[float(f"{v:.4g}") for v in g["lr"][keep]]))
json.dump(dict(lens=LENS, groups=OUT, ups_r=UPS_R, zs=ZS),
          open(SCR + "lens_sub.json", "w", encoding="utf-8"))
print(f"\n   wrote lens_sub.json "
      f"({os.path.getsize(SCR+'lens_sub.json')/1024:.0f} KB, "
      f"{len(LENS)} lenses, {len(OUT)} groups)")
