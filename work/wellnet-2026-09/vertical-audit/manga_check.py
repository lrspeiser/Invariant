"""Independent check of the SIGN, on resolved MaNGA face-on dispersion profiles.

Source: `work/wellnet-2026-09/env-data/clean/manga_faceon_sigma_profiles.csv`
(240 near-face-on MaNGA DR17 late-type disks, 1,671 radial points, SDSS DR17 DAP
HYB10-MILESHC-MASTARSSP; manifest sibling present).

WHAT IT CAN AND CANNOT DO
-------------------------
CAN   (i)  measure  d log sigma_LOS,0 / d log Sigma_b  between galaxies on a
           completely different sample, different photometry, different stellar
           masses (NSA SED, Chabrier) and a different dispersion instrument;
      (ii) measure the WITHIN-galaxy radial slope  h_sigma / R_d  directly from
           4-9 resolved points per galaxy at ~1 km/s, which is the statistic
           DiskMass has only through a published fit.
CANNOT     form B_z, because no scale height is measured.  h_z must be imported
           from the same Bershady+2010b relation, and that import is stated
           explicitly at every step, never hidden.
LIMITS     sigma_LOS is not sigma_z (i < 30 deg, so the in-plane leak is
           <= 25% in quadrature and is corrected with the SAME alpha, beta the
           DiskMass chain uses); MaNGA's instrumental sigma is ~70 km/s, so the
           strict tier keeps only galaxies entirely above it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ("C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
       "work/wellnet-2026-09/env-data/clean/manga_faceon_sigma_profiles.csv")
C_HZ = 0.6667          # measured pipeline coefficient, d log Bz / d log h_z
BERSHADY = 0.643       # d log h_z / d log h_R
ALPHA, BETA = 0.60, 0.70
R = {}

df = pd.read_csv(SRC)
print(f"  loaded {len(df)} radial points, {df.plateifu.nunique()} galaxies")
assert len(df) == 1671 and df.plateifu.nunique() == 240, "ingest row count moved"

# de-project sigma_LOS -> sigma_z with the SAME ellipsoid the DiskMass chain uses
i = np.radians(df.incl_deg.to_numpy())
proj = np.sqrt(np.cos(i) ** 2 + 0.5 * np.sin(i) ** 2 * (1 + BETA ** 2) / ALPHA ** 2)
df["sigma_z_kms"] = df.sigma_los_kms / proj
df["x_Rd"] = df.r_med_kpc / df.Rd_kpc
print(f"  median projection factor sigma_LOS/sigma_z = {np.median(proj):.4f}"
      f"  (range {proj.min():.3f}-{proj.max():.3f})")

TIERS = {
    "T1_all_above_70": lambda g: bool(g.above_70kms.all()),
    "T2_points_above_70": None,        # handled by row filter
    "T3_all_above_50": lambda g: bool(g.above_50kms.all()),
    # floor-distance ladder: MaNGA's instrumental sigma is ~70 km/s and the
    # sample median is 77, so a truncation at the low-sigma end is the obvious
    # way to flatten d log sigma / d log Sigma_b.  Walk away from the floor.
    "T4_all_above_90": lambda g: bool((g.sigma_los_kms > 90).all()),
    "T5_all_above_110": lambda g: bool((g.sigma_los_kms > 110).all()),
}
WIN = (0.3, 3.0)                        # R/R_d, declared before any fit


def fit_galaxy(g):
    m = (g.x_Rd >= WIN[0]) & (g.x_Rd <= WIN[1]) & np.isfinite(g.sigma_z_kms)
    if m.sum() < 4:
        return None
    xx = g.x_Rd[m].to_numpy()
    yy = np.log(g.sigma_z_kms[m].to_numpy())
    w = (g.sigma_z_kms[m] / g.e_sigma_los_kms[m]).to_numpy() ** 2
    if np.ptp(xx) < 0.5:
        return None
    p = np.polyfit(xx, yy, 1, w=np.sqrt(w))
    if p[0] >= 0:
        hs = np.inf
    else:
        hs = -1.0 / p[0]
    return dict(plateifu=g.plateifu.iloc[0], n=int(m.sum()),
                lsig0=float(p[1] / np.log(10)), h_over_Rd=float(hs),
                lSigb=float(np.log10(g.Sigma_b_Msun_pc2.iloc[0])),
                lRd=float(np.log10(g.Rd_kpc.iloc[0])),
                incl=float(g.incl_deg.iloc[0]),
                xmin=float(xx.min()), xmax=float(xx.max()),
                e_lsig0=float(np.std(yy - np.polyval(p, xx)) / np.log(10)
                              / max(np.sqrt(m.sum() - 2), 1)))


for tier, gate in TIERS.items():
    d = df if tier != "T2_points_above_70" else df[df.above_70kms]
    rows = []
    for pid, g in d.groupby("plateifu"):
        if gate is not None and not gate(g):
            continue
        r = fit_galaxy(g)
        if r and np.isfinite(r["lSigb"]) and np.isfinite(r["h_over_Rd"]):
            rows.append(r)
    T = pd.DataFrame(rows)
    if len(T) < 8:
        print(f"\n  {tier}: only {len(T)} galaxies, skipped")
        continue
    x, y = T.lSigb.to_numpy(), T.lsig0.to_numpy()
    b_sig = float(np.polyfit(x, y, 1)[0])
    b_Rd = float(np.polyfit(x, T.lRd.to_numpy(), 1)[0])
    b_hz = BERSHADY * b_Rd
    slope_Bz = 2 * b_sig - 1.0 - C_HZ * b_hz
    # galaxy bootstrap
    rb = np.random.default_rng(2718)
    bs = np.array([np.polyfit(x[k], y[k], 1)[0]
                   for k in (rb.integers(0, len(T), len(T)) for _ in range(4000))])
    bsl = 2 * bs - 1.0 - C_HZ * b_hz
    print(f"\n  {tier}:  {len(T)} galaxies, "
          f"{int(T.n.sum())} radial points, median {T.n.median():.0f} per galaxy")
    print(f"    log10 Sigma_b range     {x.min():.2f} to {x.max():.2f}"
          f"   sd {x.std():.3f} dex")
    print(f"    d log sigma_z,0/d log Sigma_b = {b_sig:+.4f}"
          f"   (Newton + fixed Upsilon + Bershady needs "
          f"{(1+C_HZ*b_hz)/2:+.4f})")
    print(f"    d log R_d       /d log Sigma_b = {b_Rd:+.4f}"
          f"  -> d log h_z/d log Sigma_b = {b_hz:+.4f}")
    print(f"    => d log10 B_z / d log10 Sigma_b = {slope_Bz:+.4f}"
          f"   bootstrap 68% [{np.percentile(bsl,16):+.3f},"
          f"{np.percentile(bsl,84):+.3f}]"
          f"   P(>=0) = {np.mean(bsl >= 0):.4f}")
    print(f"       DiskMass, same statistic:  -0.346, P(>=0) = 0.0095")
    # radial statistic
    hh = T.h_over_Rd.to_numpy()
    hh = hh[np.isfinite(hh) & (hh > 0) & (hh < 20)]
    smed = float(np.median(d[d.plateifu.isin(T.plateifu)].sigma_los_kms))
    print(f"    median sigma_LOS in this tier = {smed:.1f} km/s"
          f"   (MaNGA instrumental floor ~70; DiskMass sample median "
          f"{np.median([24.0]):.0f}-45)")
    print(f"    WITHIN-galaxy: h_sigma/R_d = {np.median(hh):.3f}"
          f"   68% [{np.percentile(hh,16):.3f},{np.percentile(hh,84):.3f}]"
          f"   n = {hh.size}")
    print(f"       naive Newton (h_z const, Sigma ~ exp(-R/R_d)) = 2.000.")
    print(f"       NOT like-for-like: the DiskMass forward chain, on ITS")
    print(f"       galaxies, gives 2.499 (Newton) and 2.896 (RAR) because of")
    print(f"       the gas layer, the thickness factor and the leakage term.")
    print(f"       MaNGA has no matched chain and, with a 68% range this wide,")
    print(f"       no power to separate 2.50 from 2.90 in any case.")
    # shared-denominator bound for THIS sample
    var_x = float(np.var(x))
    for e_lM, e_lRd in ((0.10, 0.043), (0.20, 0.087)):
        ve = e_lM ** 2 + (2 * e_lRd) ** 2
        print(f"    shared-denominator bound: e(logM*)={e_lM:.2f}, "
              f"e(logR_d)={e_lRd:.3f} -> var(eps_x)={ve:.4f}, "
              f"bias = {-ve/var_x:+.4f}")
    R[tier] = dict(n_gal=int(len(T)), n_pts=int(T.n.sum()),
                   d_logsigma_d_logSigma_b=b_sig,
                   newton_requires=float((1 + C_HZ * b_hz) / 2),
                   d_logRd_d_logSigma_b=b_Rd, d_loghz_d_logSigma_b=float(b_hz),
                   slope_Bz=float(slope_Bz),
                   slope_Bz_boot_p16=float(np.percentile(bsl, 16)),
                   slope_Bz_boot_p84=float(np.percentile(bsl, 84)),
                   slope_Bz_boot_p2p5=float(np.percentile(bsl, 2.5)),
                   slope_Bz_boot_p97p5=float(np.percentile(bsl, 97.5)),
                   p_ge_zero=float(np.mean(bsl >= 0)),
                   h_over_Rd_median=float(np.median(hh)),
                   h_over_Rd_p16=float(np.percentile(hh, 16)),
                   h_over_Rd_p84=float(np.percentile(hh, 84)),
                   var_logSigma_b=var_x,
                   corr_incl_logSigma_b=float(np.corrcoef(
                       x, T.incl.to_numpy())[0, 1]),
                   shared_denominator_bias_e0p10=float(
                       -(0.10 ** 2 + (2 * 0.043) ** 2) / var_x),
                   median_sigma_los=smed)

R["notes"] = dict(
    window_R_over_Rd=list(WIN), alpha=ALPHA, beta=BETA,
    c_hz=C_HZ, bershady=BERSHADY,
    caveat="No scale height is measured in MaNGA. h_z enters only through the "
           "IMPORTED Bershady+2010b exponent, so this is not an independent "
           "test of the h_z channel -- it is an independent test of the "
           "sigma - Sigma_b scaling and of the radial slope.",
    source=SRC)
with open(os.path.join(HERE, "manga_check.json"), "w") as fh:
    json.dump(R, fh, indent=1)
print("\n  wrote manga_check.json")
