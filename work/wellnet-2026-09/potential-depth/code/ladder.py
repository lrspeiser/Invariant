"""Build the potential-depth ladder: (g_bar, |Phi_b|, g_obs, class, provenance).

CONVENTIONS, stated once and used everywhere
--------------------------------------------
Baryonic acceleration      g_bar(r) = G M_b(<r) / r^2,  M_b = stars + gas.
Baryonic potential depth   Phi_b(r) = -[ Int_r^Rmax g_bar dr' + g_bar(Rmax)*Rmax ]
                           i.e. all baryonic mass is inside Rmax and the field
                           falls off as a point mass outside it.  Rmax is the
                           OUTERMOST RADIUS AT WHICH THAT SYSTEM'S BARYONS ARE
                           MEASURED.  This matches work/wellnet-2026-09/phi_rank.py
                           exactly so the two are directly comparable.
Shape factor               S(r) = |Phi_b(r)| / (g_bar(r) * r).

  THEOREM (used repeatedly below).  M_b(<r') >= M_b(<r) for r' >= r, hence
      |Phi_b(r)| = Int_r^inf G M_b(<r')/r'^2 dr' >= G M_b(<r)/r = g_bar(r) * r,
  so S(r) >= 1 ALWAYS, with equality iff no baryonic mass lies outside r.
  Therefore  log|Phi_b| = log g_bar + log r + log S,  log S >= 0.
  Every system for which only one radius is measured gets S = 1, which is a
  strict LOWER BOUND on |Phi_b|, not a guess.

Boost                      nu_obs = g_obs / g_bar.

CLASS LADDER (the 6 rungs the brief asks for)
  1 field_galaxy      SPARC rotation curves
  2 small_group       SDSS optical groups, 10-14 members
  3 poor_group        X-ray groups, kT < 1.0 keV   (+ optical groups 15+ members)
  4 rich_group        X-ray groups, 1.0 <= kT < 2.0 keV
  5 low_mass_cluster  X-ray, 2.0 <= kT < 4.0 keV
  6 massive_cluster   X-ray, kT >= 4.0 keV  (X-COP resolved + Gonzalez)
"""
from __future__ import annotations

import csv
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
DATA = os.path.join(LANE, "data")
REPO = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration"
sys.path.insert(0, os.path.join(REPO, "work", "gravitylab"))

G = 6.67430e-11
MSUN = 1.98892e30
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
A0 = 1.2e-10
KMS = 1e3

# cosmologies as published, used only for rho_c(z) in the overdensity radii
COSMO = {"sun2009": dict(H0=73.0, Om=0.24, Ol=0.76),
         "lovisari2015": dict(H0=70.0, Om=0.27, Ol=0.73),
         "gonzalez2013": dict(H0=70.0, Om=0.3, Ol=0.7)}


def rho_c(z, H0, Om, Ol):
    """Critical density in kg/m^3 at redshift z."""
    H0s = H0 * 1e3 / MPC
    Ez2 = Om * (1 + z) ** 3 + Ol
    return 3.0 * (H0s ** 2) * Ez2 / (8.0 * math.pi * G)


def m_delta(r_m, delta, z, cos):
    """Spherical-overdensity mass for a measured radius r_delta."""
    return delta * rho_c(z, **COSMO[cos]) * (4.0 / 3.0) * math.pi * r_m ** 3


# --------------------------------------------------------------------------
# Phi from a tabulated, resolved g_bar(r)  (SPARC, X-COP)
# --------------------------------------------------------------------------
def phi_resolved(r_m, g):
    """Trapezoid inward-cumulative Int_r^Rmax g dr' plus the point-mass tail."""
    o = np.argsort(r_m)
    r, gg = np.asarray(r_m)[o], np.asarray(g)[o]
    seg = 0.5 * (gg[1:] + gg[:-1]) * np.diff(r)
    inner = np.concatenate([[0.0], np.cumsum(seg)])
    inner = inner[-1] - inner
    outer = gg[-1] * r[-1]
    out = -(inner + outer)
    inv = np.empty_like(out)
    inv[o] = out
    return inv


# --------------------------------------------------------------------------
# Phi from two measured radii with a power-law M_b in between (X-ray groups)
# --------------------------------------------------------------------------
def phi_two_radius(r_in, M_in, r_out, M_out):
    """|Phi_b| at r_in and r_out for M_b(r) = M_out (r/r_out)^s on [r_in,r_out],
    point-mass tail beyond r_out.  Returns (|Phi_in|, |Phi_out|, s)."""
    x = r_in / r_out
    s = math.log(M_out / M_in) / math.log(r_out / r_in)
    base = G * M_out / r_out                      # = |Phi(r_out)|, the tail
    if abs(s - 1.0) < 1e-9:
        integral = base * math.log(1.0 / x)
    else:
        integral = base * (1.0 - x ** (s - 1.0)) / (s - 1.0)
    return base + integral, base, s


# --------------------------------------------------------------------------
# Stellar mass calibration from Gonzalez+2013 -- the only MEASURED stellar
# masses in the ladder.  log10(M*/M_gas) as a linear function of log10 M_gas.
# --------------------------------------------------------------------------
def stellar_calibration(verbose=True):
    rows = list(csv.DictReader(open(os.path.join(DATA,
                "gonzalez2013_baryons.tsv"), encoding="utf-8"), delimiter="\t"))
    x, y = [], []
    for r in rows:
        if not r["Mstar3d500_1e13"]:
            continue
        mg = float(r["Mgas500_1e13"]) * 1e13
        ms = float(r["Mstar3d500_1e13"]) * 1e13
        x.append(math.log10(mg))
        y.append(math.log10(ms / mg))
    x, y = np.array(x), np.array(y)
    A = np.column_stack([np.ones_like(x), x])
    c, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ c
    if verbose:
        print(f"  stellar calibration on Gonzalez+2013 (n={len(x)}, "
              f"log10 Mgas {x.min():.2f}-{x.max():.2f}):")
        print(f"    log10(M*/Mgas) = {c[0]:+.4f} {c[1]:+.4f} log10(Mgas/Msun)")
        print(f"    rms residual {res.std():.4f} dex, "
              f"corr {np.corrcoef(x, y)[0,1]:+.3f}")
    return c, float(res.std()), (x.min(), x.max())


def mstar_of_mgas(Mgas, c):
    """M* from M_gas via the Gonzalez calibration, capped at M* = M_gas."""
    lg = np.log10(np.asarray(Mgas, float))
    ratio = 10.0 ** (c[0] + c[1] * lg)
    ratio = np.minimum(ratio, 1.0)
    return np.asarray(Mgas, float) * ratio


# --------------------------------------------------------------------------
def kT_class(kT):
    if kT < 1.0:
        return 3, "poor_group"
    if kT < 2.0:
        return 4, "rich_group"
    if kT < 4.0:
        return 5, "low_mass_cluster"
    return 6, "massive_cluster"


ROW = ("system class class_rank source probe r_kpc Mb_Msun g_bar g_obs nu_obs "
       "abs_Phi_b S_shape e_lg_gbar e_lg_gobs sys_lg_Mb tier phi_method "
       "baryon_method gobs_method assumes").split()


def build(stellar=True, verbose=True):
    cal, cal_rms, cal_rng = stellar_calibration(verbose)
    out = []

    def add(**kw):
        kw.setdefault("tier", 1)
        kw["nu_obs"] = kw["g_obs"] / kw["g_bar"]
        kw["S_shape"] = kw["abs_Phi_b"] / (kw["g_bar"] * kw["r_kpc"] * KPC)
        out.append({k: kw.get(k, "") for k in ROW})

    # ---------------- 1. SPARC + 6. X-COP, resolved profiles --------------
    rp = list(csv.DictReader(open(os.path.join(DATA,
              "resolved_profiles_sparc_xcop.csv"), encoding="utf-8")))
    bysys = {}
    for r in rp:
        bysys.setdefault((r["system"], r["probe"]), []).append(r)
    for (name, probe), rs in bysys.items():
        r_kpc = np.array([float(r["r_kpc"]) for r in rs])
        gb = np.array([float(r["g_bar"]) for r in rs])
        go = np.array([float(r["g_obs"]) for r in rs])
        eb = np.array([float(r["err_boost"]) if r["err_boost"] else np.nan
                       for r in rs])
        if len(r_kpc) < 3:
            continue
        ph = np.abs(phi_resolved(r_kpc * KPC, gb))
        if probe == "rotation_curve":
            rank, cls, src = 1, "field_galaxy", "SPARC"
            bm = ("3.6um starlight at Upsilon*=0.5 disk / 0.7 bulge "
                  "plus 21cm HI x1.33 for helium")
            gm = "circular velocity, V_obs^2/r, inclination-corrected"
            e_gb, sys_mb = 0.11, 0.11
        else:
            rank, cls, src = 6, "massive_cluster", "X-COP"
            bm = ("deprojected n_e (XMM) -> M_gas; measured stellar profiles "
                  "(Ghizzardi+2020) for 7 of 12, 10% of M_gas for the other 5")
            gm = "hydrostatic equation on n_e(r) and T(r), XMM + Planck SZ"
            e_gb, sys_mb = 0.06, 0.10
        for i in range(len(r_kpc)):
            nu = go[i] / gb[i]
            e_go = (abs(eb[i] / nu) / math.log(10)
                    if np.isfinite(eb[i]) and nu > 0 else 0.05)
            add(system=name, **{"class": cls}, class_rank=rank, source=src,
                probe=probe, r_kpc=r_kpc[i], Mb_Msun=gb[i] * (r_kpc[i] * KPC) ** 2
                / G / MSUN, g_bar=gb[i], g_obs=go[i], abs_Phi_b=ph[i],
                e_lg_gbar=e_gb, e_lg_gobs=min(e_go, 1.0), sys_lg_Mb=sys_mb,
                phi_method="resolved trapezoid + point-mass tail at Rmax",
                baryon_method=bm, gobs_method=gm, assumes="none")

    # ---------------- 3/4/5. Sun+2009 Chandra groups ----------------------
    for r in csv.DictReader(open(os.path.join(DATA, "sun2009_groups.tsv"),
                                 encoding="utf-8"), delimiter="\t"):
        z = float(r["z"])
        r2500 = float(r["r2500_kpc"]) * KPC
        M2500 = float(r["M2500_Msun"]) * MSUN
        Mg2500 = float(r["Mgas2500_Msun"]) * MSUN
        kT = float(r["T2500_keV"])
        rank, cls = kT_class(kT)
        has500 = bool(r["r500_kpc"] and r["Mgas500_Msun"]
                      and "S" not in (r["r500_flag"] or ""))
        pairs = [(r2500, M2500, Mg2500, "r2500")]
        if has500:
            pairs.append((float(r["r500_kpc"]) * KPC,
                          float(r["M500_from_r500_Msun"]) * MSUN,
                          float(r["Mgas500_Msun"]) * MSUN, "r500"))
        Mb = [mg + (mstar_of_mgas(mg / MSUN, cal) * MSUN if stellar else 0.0)
              for (_, _, mg, _) in pairs]
        if len(pairs) == 2:
            phis = phi_two_radius(pairs[0][0], Mb[0], pairs[1][0], Mb[1])
            phi = [G * phis[0] * 0 + phis[0], phis[1]]
            pm = ("two measured radii, M_b power-law index "
                  f"s={phis[2]:.2f} between them, point-mass tail beyond r500")
        else:
            phi = [G * Mb[0] / pairs[0][0]]
            pm = "single measured radius: |Phi| = g_bar*r, a strict LOWER BOUND"
        for k, (rr, Mt, mg, ap) in enumerate(pairs):
            gb = G * Mb[k] / rr ** 2
            go = G * Mt / rr ** 2
            add(system="Sun2009_" + r["name"].replace(" ", ""),
                **{"class": cls}, class_rank=rank, source="Sun+2009",
                probe="hydrostatic", r_kpc=rr / KPC, Mb_Msun=Mb[k] / MSUN,
                g_bar=gb, g_obs=go, abs_Phi_b=phi[k],
                e_lg_gbar=0.08, e_lg_gobs=0.10, sys_lg_Mb=0.12,
                phi_method=pm,
                baryon_method=("M_gas = f_gas x M_hydrostatic (Chandra); stars "
                               "from the Gonzalez+2013 M*/M_gas calibration"
                               if stellar else "M_gas only, no stellar term"),
                gobs_method=("X-ray hydrostatic mass at the measured "
                             f"overdensity radius {ap}"),
                assumes="hydrostatic equilibrium; spherical symmetry")

    # ---------------- 3/4/5. Lovisari+2015 XMM groups ---------------------
    for r in csv.DictReader(open(os.path.join(DATA, "lovisari2015_groups.tsv"),
                                 encoding="utf-8"), delimiter="\t"):
        kT = float(r["kT_keV"])
        rank, cls = kT_class(kT)
        pairs = [(float(r["R2500_kpc"]) * KPC, float(r["M2500_Msun"]) * MSUN,
                  float(r["Mgas2500_Msun"]) * MSUN, "R2500",
                  float(r["M2500_err_Msun"]) / float(r["M2500_Msun"])),
                 (float(r["R500_kpc"]) * KPC, float(r["M500_Msun"]) * MSUN,
                  float(r["Mgas500_Msun"]) * MSUN, "R500",
                  float(r["M500_err_Msun"]) / float(r["M500_Msun"]))]
        Mb = [mg + (mstar_of_mgas(mg / MSUN, cal) * MSUN if stellar else 0.0)
              for (_, _, mg, _, _) in pairs]
        p_in, p_out, s = phi_two_radius(pairs[0][0], Mb[0], pairs[1][0], Mb[1])
        for k, ((rr, Mt, mg, ap, ferr), ph) in enumerate(
                zip(pairs, [p_in, p_out])):
            gb = G * Mb[k] / rr ** 2
            add(system="Lovisari2015_" + r["name"], **{"class": cls},
                class_rank=rank, source="Lovisari+2015", probe="hydrostatic",
                r_kpc=rr / KPC, Mb_Msun=Mb[k] / MSUN, g_bar=gb,
                g_obs=G * Mt / rr ** 2, abs_Phi_b=ph,
                e_lg_gbar=0.08, e_lg_gobs=ferr / math.log(10),
                sys_lg_Mb=0.12,
                phi_method=("two measured radii, M_b power-law index "
                            f"s={s:.2f} between them, point-mass tail beyond R500"),
                baryon_method=("M_gas tabulated (XMM); stars from the "
                               "Gonzalez+2013 M*/M_gas calibration"
                               if stellar else "M_gas only, no stellar term"),
                gobs_method=("X-ray hydrostatic mass at the measured "
                             f"overdensity radius {ap}"),
                assumes="hydrostatic equilibrium; spherical symmetry")

    # ---------------- 5/6. Gonzalez+2013 ----------------------------------
    for r in csv.DictReader(open(os.path.join(DATA, "gonzalez2013_baryons.tsv"),
                                 encoding="utf-8"), delimiter="\t"):
        kT = float(r["kT_keV"])
        rank, cls = kT_class(kT)
        rr = float(r["r500_Mpc"]) * MPC
        Mt = float(r["M500_1e14"]) * 1e14 * MSUN
        mg = float(r["Mgas500_1e13"]) * 1e13 * MSUN
        if r["Mstar3d500_1e13"]:
            ms = float(r["Mstar3d500_1e13"]) * 1e13 * MSUN
            bm = ("M_gas (XMM) + MEASURED deprojected stellar mass "
                  "(BCG + ICL + satellites, Gonzalez+2007 photometry)")
        else:
            ms = mstar_of_mgas(mg / MSUN, cal) * MSUN if stellar else 0.0
            bm = ("M_gas (XMM) + stars from this lane's own Gonzalez "
                  "calibration (no photometry for this object)")
        Mb = mg + (ms if stellar else 0.0)
        gb = G * Mb / rr ** 2
        add(system="Gonzalez2013_" + r["name"].replace(" ", ""),
            **{"class": cls}, class_rank=rank, source="Gonzalez+2013",
            probe="hydrostatic", r_kpc=rr / KPC, Mb_Msun=Mb / MSUN, g_bar=gb,
            g_obs=G * Mt / rr ** 2, abs_Phi_b=G * Mb / rr,
            e_lg_gbar=0.05, e_lg_gobs=0.05, sys_lg_Mb=0.10,
            phi_method="single measured radius: |Phi| = g_bar*r, a strict LOWER BOUND",
            baryon_method=bm,
            gobs_method="X-ray hydrostatic M500 at the measured r500",
            assumes="hydrostatic equilibrium; spherical symmetry")

    # ---------------- 2/3. SDSS optical groups (tier 2) -------------------
    ETA = 2.0    # isotropic isothermal sphere: M(<r) = 2 sigma^2 r / G
    for r in csv.DictReader(open(os.path.join(DATA,
             "optical_groups_features.tsv"), encoding="utf-8"), delimiter="\t"):
        n = int(r["members"])
        rank, cls = (2, "small_group") if n < 15 else (3, "poor_group")
        rr = float(r["r_rms_kpc"]) * KPC
        Ms = float(r["total_mass_msun"]) * MSUN
        sig = float(r["sigma_gap_km_s"]) * KMS
        gb = G * Ms / rr ** 2
        add(system="SDSSgrp_" + r["group"], **{"class": cls}, class_rank=rank,
            source="J/A+A/690/A52 groups", probe="velocity_dispersion",
            r_kpc=rr / KPC, Mb_Msun=Ms / MSUN, g_bar=gb,
            g_obs=ETA * sig ** 2 / rr, abs_Phi_b=G * Ms / rr,
            e_lg_gbar=0.15, e_lg_gobs=2.0 / math.log(10) / math.sqrt(2 * (n - 1)),
            sys_lg_Mb=0.30, tier=2,
            phi_method="single measured radius: |Phi| = g_bar*r, a strict LOWER BOUND",
            baryon_method=("STELLAR MASS ONLY from SDSS r-band luminosity. "
                           "Hot gas is NOT included and is the dominant baryon "
                           "reservoir at this scale, so g_bar is a LOWER BOUND "
                           "and nu_obs an UPPER BOUND. tier 2 for this reason."),
            gobs_method=(f"g_obs = eta sigma^2 / r_rms with eta = {ETA} "
                         "(isotropic isothermal sphere, 1-D gapper dispersion "
                         f"from {n} members)"),
            assumes="virial equilibrium; velocity isotropy; eta=2")

    return out, cal, cal_rms, cal_rng


def write(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=ROW)
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    rows, cal, rms, rng = build()
    write(rows, os.path.join(LANE, "potential_depth_ladder.csv"))
    import collections
    c = collections.Counter((r["class_rank"], r["class"]) for r in rows)
    print(f"\n  {len(rows)} rows, "
          f"{len(set(r['system'] for r in rows))} systems")
    for k in sorted(c):
        sub = [r for r in rows if r["class_rank"] == k[0]]
        lg = np.log10([r["g_bar"] for r in sub])
        lp = np.log10([r["abs_Phi_b"] for r in sub])
        print(f"    {k[0]} {k[1]:<18} n={c[k]:5d}  "
              f"nsys={len(set(r['system'] for r in sub)):4d}  "
              f"log g_bar {lg.min():+7.2f}..{lg.max():+7.2f}  "
              f"log|Phi| {lp.min():6.2f}..{lp.max():6.2f}")
