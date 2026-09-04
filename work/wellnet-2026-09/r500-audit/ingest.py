"""
R500 TAUTOLOGY AUDIT -- ingest layer.

Loads exactly the data the lane-12 claim rests on, with every assertion the
`astro-data-acquisition-traps` memory pattern demands: row counts, column
counts, catalogue identifier echoed, and a finiteness check on every row of a
value column.

PROVENANCE, stated up front because it is the deliverable:

  X-COP profiles   runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/
                   <NAME>_density_L1.fits   HDU1 DENSITY   R_IN,R_OUT [kpc], NE [cm-3]
                                            HDU1 header    R500 [kpc], M500 [1e14 Msun]
                                            *** header comment: "Hydrostatic-equilibrium R500" ***
                   <NAME>_temperature.fits  HDU1 XRAY      RW_X [R/R500], T_X [T/T500]
                                            *** BOTH AXES OF THE TEMPERATURE PROFILE ARE
                                                SCALED BY THE HYDROSTATIC M500/R500 ***
                   <NAME>_mstar.fits        HDU2 MSTAR_SMOOTHED  RADIUS [kpc], MSTAR [Msun]

  Herbonnet+2020   work/gravity/item15-attempt2-research/herbonnet-tex/masses.tex
                   Table tbl:masses, split over TWO table* environments (rows 1-59,
                   60-100).  Column R500_ap [Mpc] = deprojected-aperture WEAK LENSING
                   R500.  Independent of the X-ray hydrostatic mass.

KiDS and wide binaries are NOT loaded anywhere in this lane.
"""
from __future__ import annotations
import glob
import math
import os
import re

import numpy as np
from astropy.io import fits

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
XCOP_DIR = ROOT + "runs/gravity/roadmap/item-59-xcop-forward-observable-gate-v1-source/raw/"
HERB_TEX = ROOT + "work/gravity/item15-attempt2-research/herbonnet-tex/masses.tex"

G = 6.67430e-11
MSUN = 1.98892e30
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
A0 = 1.2e-10
MP = 1.67262192e-27
MU = 0.6
MU_E = 1.14
H0 = 70.0 * 1e3 / 3.0856775814913673e22
RHOC0 = 3 * H0 ** 2 / (8 * math.pi * G)
OM, OL = 0.3, 0.7

# the bench's radial cut, reproduced verbatim
R_MIN_KPC, R_MAX_KPC = 120.0, 1650.0

# X-COP is the 12-cluster sample; the record's within-X-COP correlation used
# whichever of the 12 the bench could load.
XCOP_EXPECTED = ["A1644", "A1795", "A2029", "A2142", "A2255", "A2319",
                 "A3158", "A3266", "A644", "A85", "RXC1825", "ZW1215"]


def rho_c(z):
    """critical density at redshift z, flat LCDM, h70."""
    return RHOC0 * (OM * (1 + z) ** 3 + OL)


def nu_rar(x):
    """the bench's interpolating function, nu = g_obs/g_bar predicted from x=g_bar/a0."""
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


# --------------------------------------------------------------------------
#  X-COP
# --------------------------------------------------------------------------
def _finite_check(name, arr, what):
    arr = np.asarray(arr, float)
    n_bad = int((~np.isfinite(arr)).sum())
    assert n_bad == 0, f"[{name}] {what}: {n_bad} of {arr.size} values non-finite"
    return arr


def load_xcop_cluster(cname, verbose=False):
    """Load ONE X-COP cluster exactly as invariant_bench._cluster_profile does,
    but keeping every intermediate so the audit can perturb any of them.

    Returns a dict, or None if the cluster lacks a required file.
    """
    d = os.path.join(XCOP_DIR, cname)
    fd = glob.glob(os.path.join(d, "*density*.fits"))
    ft = glob.glob(os.path.join(d, "*temperature*.fits"))
    if not fd or not ft:
        return None

    hd = fits.open(fd[0])
    ht = fits.open(ft[0])
    H = hd[1].header

    # --- identifier echo, the VizieR-trap guard ---------------------------
    echoed = str(H["CLUSTER"]).strip()
    assert echoed == cname, f"catalogue identifier mismatch: dir={cname} header={echoed}"
    echoed_t = str(ht[1].header["CLUSTER"]).strip()
    assert echoed_t == cname, f"temperature file identifier mismatch: {echoed_t}"

    # --- column-name assertions (the -out.all truncation guard) -----------
    dcols = [c.name for c in hd[1].columns]
    tcols = [c.name for c in ht[1].columns]
    assert dcols[:3] == ["R_IN", "R_OUT", "NE"], f"density columns changed: {dcols}"
    assert tcols[:3] == ["RW_X", "T_X", "eT_X"], f"temperature columns changed: {tcols}"
    dunits = [c.unit for c in hd[1].columns]
    tunits = [c.unit for c in ht[1].columns]
    assert dunits[0] == "kpc" and dunits[2] == "cm-3", f"density units changed: {dunits}"
    # THE PROVENANCE ASSERTION THIS WHOLE AUDIT TURNS ON:
    assert tunits[0] == "R/R500" and tunits[1] == "T/T500", (
        f"temperature profile is no longer stored in R500/T500 units: {tunits}")

    M500 = float(H["M500"]) * 1e14 * MSUN
    R500 = float(H["R500"]) * KPC
    eM500 = float(H["ERR_M500"]) * 1e14 * MSUN
    eR500 = float(H["ERR_R500"]) * KPC
    z = float(H["REDSHIFT"])

    da = hd[1].data
    n_rows_d = len(da)
    r_in = _finite_check(cname, da["R_IN"].astype(np.float64), "R_IN")
    r_out = _finite_check(cname, da["R_OUT"].astype(np.float64), "R_OUT")
    ne_raw = _finite_check(cname, da["NE"].astype(np.float64), "NE")
    ne_lo = da["NE_LOW"].astype(np.float64)
    ne_hi = da["NE_HIGH"].astype(np.float64)
    assert len(r_in) == len(ne_raw) == n_rows_d
    assert np.all(r_out > r_in), f"[{cname}] non-monotonic radial bins"
    assert np.all(ne_raw > 0), f"[{cname}] non-positive electron density"

    td = ht[1].data
    n_rows_t = len(td)
    rw_x = _finite_check(cname, td["RW_X"].astype(np.float64), "RW_X")
    t_x = _finite_check(cname, td["T_X"].astype(np.float64), "T_X")
    et_x = td["eT_X"].astype(np.float64)
    assert np.all(np.diff(rw_x) > 0), f"[{cname}] temperature radii not increasing"
    assert np.all(t_x > 0), f"[{cname}] non-positive scaled temperature"

    # stellar mass, physical kpc grid (HDU 2 = MSTAR_SMOOTHED)
    fs = glob.glob(os.path.join(d, "*mstar*.fits"))
    if fs:
        hs = fits.open(fs[0])
        assert str(hs[1].header["CLUSTER"]).strip() == cname
        scols = [c.name for c in hs[2].columns]
        sunits = [c.unit for c in hs[2].columns]
        assert scols[:2] == ["RADIUS", "MSTAR"], f"mstar columns changed: {scols}"
        assert sunits[0] == "kpc", f"mstar HDU2 radius no longer in kpc: {sunits}"
        mst_r = _finite_check(cname, hs[2].data["RADIUS"].astype(np.float64), "MSTAR RADIUS")
        mst_m = _finite_check(cname, hs[2].data["MSTAR"].astype(np.float64), "MSTAR")
        hs.close()
        has_mstar = True
    else:
        mst_r, mst_m, has_mstar = None, None, False

    hd.close()
    ht.close()

    out = dict(
        name=cname, z=z,
        M500_hse=M500, R500_hse=R500, eM500_hse=eM500, eR500_hse=eR500,
        r_in_kpc=r_in, r_out_kpc=r_out, ne_cm3=ne_raw,
        ne_lo=ne_lo, ne_hi=ne_hi,
        rw_x=rw_x, t_x=t_x, et_x=et_x,
        mstar_r_kpc=mst_r, mstar_m=mst_m, has_mstar=has_mstar,
        n_rows_density=n_rows_d, n_rows_temperature=n_rows_t,
    )
    if verbose:
        print(f"   [{cname}] z={z:.4f} R500={R500/KPC:7.1f} kpc "
              f"(+-{eR500/KPC:.1f})  M500={M500/1e14/MSUN:6.3f}e14 "
              f"(+-{eM500/1e14/MSUN:.3f})  ne rows={n_rows_d}  T rows={n_rows_t}  "
              f"mstar={'yes' if has_mstar else 'NO -> 10% of Mgas'}")
    return out


def build_profile(c, R500=None, M500=None):
    """Reproduce invariant_bench._cluster_profile with R500/M500 injectable.

    This is the FUNCTION UNDER AUDIT.  Note the three places R500 enters:
      (1) kT500 = G M500 mu m_p / (2 R500)   -- sets the temperature SCALE
      (2) rw_x * R500                        -- sets where the shape is sampled
      (3) (downstream) the r/R500 axis itself
    while g_bar depends on R500 not at all.
    """
    if R500 is None:
        R500 = c["R500_hse"]
    if M500 is None:
        M500 = c["M500_hse"]
    kT500 = G * M500 * MU * MP / (2 * R500)

    r = 0.5 * (c["r_in_kpc"] + c["r_out_kpc"]) * KPC
    ne = c["ne_cm3"] * 1e6
    kT = np.interp(r, c["rw_x"] * R500, c["t_x"] * kT500)
    lr = np.log(r)
    go = -(kT / (MU * MP)) * (np.gradient(np.log(ne), lr)
                              + np.gradient(np.log(kT), lr)) / r
    rho = MU_E * ne * MP
    Mg = (4 / 3 * np.pi * r[0] ** 3 * rho[0]
          + np.concatenate([[0.], np.cumsum(4 * np.pi * rho[:-1] * r[:-1] ** 2 * np.diff(r))]))
    if c["has_mstar"]:
        Mst = np.interp(r, c["mstar_r_kpc"] * KPC, c["mstar_m"] * MSUN)
    else:
        Mst = Mg * 0.10
    Mb = Mg + Mst
    gb = G * Mb / r ** 2
    return dict(r=r, gb=gb, go=go, Mgas=Mg, Mstar=Mst, Mb=Mb, kT=kT, ne=ne,
                kT500=kT500, R500=R500, M500=M500)


def xcop_points(clusters, R500_map=None, mask=True):
    """Flatten to the bench's point list, applying the bench's radial cut."""
    rows = []
    for c in clusters:
        R5 = None if R500_map is None else R500_map[c["name"]]
        p = build_profile(c, R500=R5)
        m = np.ones(len(p["r"]), bool)
        if mask:
            m = ((p["r"] > R_MIN_KPC * KPC) & (p["r"] < R_MAX_KPC * KPC)
                 & (p["go"] > 0) & (p["gb"] > 0))
        for k in np.where(m)[0]:
            rows.append(dict(name=c["name"], z=c["z"], r=p["r"][k],
                             gb=p["gb"][k], go=p["go"][k],
                             Mgas=p["Mgas"][k], Mstar=p["Mstar"][k],
                             Mb=p["Mb"][k],
                             R500_hse=p["R500"], M500_hse=p["M500"]))
    return rows


def rar_residual(gb, go):
    """the bench's confound residual: log10 nu_obs - log10 nu_RAR(x)."""
    x = np.asarray(gb, float) / A0
    nu = np.asarray(go, float) / np.asarray(gb, float)
    return np.log10(nu) - np.log10(nu_rar(x))


# --------------------------------------------------------------------------
#  Herbonnet+2020 weak-lensing R500 (independent mass)
# --------------------------------------------------------------------------
_ROW = re.compile(
    r"^\s*(\d{1,3})\s*&\s*([A-Za-z0-9+\-.]+)\s*&\s*([\d.]+)\s*&"
    r"\s*\$\s*([\d.]+)\s*\\pm\s*([\d.]+)\s*\$\s*&"
    r"\s*\$\s*([\d.]+)\s*\\pm\s*([\d.]+)\s*\$\s*&"
    r"\s*\$\s*([\d.]+)_\{?-([\d.]+)\}?\^\{?\+([\d.]+)\}?\s*\$\s*&")


def load_herbonnet(verbose=False):
    """Parse tbl:masses. The table is split over two table* environments -- the
    known silent-truncation trap. We assert 100 rows and ordinals 1..100.
    """
    txt = open(HERB_TEX, encoding="utf-8", errors="replace").read()
    rows = {}
    seen = []
    for line in txt.splitlines():
        m = _ROW.match(line)
        if not m:
            continue
        nr = int(m.group(1))
        seen.append(nr)
        rows[nr] = dict(
            nr=nr, cluster=m.group(2), z=float(m.group(3)),
            M200_nfw=float(m.group(4)), eM200_nfw=float(m.group(5)),
            M500_nfw=float(m.group(6)), eM500_nfw=float(m.group(7)),
            R500_ap_Mpc=float(m.group(8)),
            eR500_lo=float(m.group(9)), eR500_hi=float(m.group(10)),
        )
    assert len(rows) == 100, (
        f"Herbonnet tbl:masses: got {len(rows)} rows, expected 100 "
        f"(the two-table* truncation trap). ordinals present: "
        f"{sorted(rows)[:5]}..{sorted(rows)[-5:]}")
    assert sorted(rows) == list(range(1, 101)), "ordinals are not exactly 1..100"
    assert len(seen) == 100, f"duplicate ordinal rows: {len(seen)}"
    for k, v in rows.items():
        assert np.isfinite(v["R500_ap_Mpc"]) and v["R500_ap_Mpc"] > 0, f"row {k} bad R500"
        assert np.isfinite(v["M500_nfw"]), f"row {k} bad M500"
    if verbose:
        print(f"   Herbonnet+2020 tbl:masses  100/100 rows, ordinals 1..100 verified; "
              f"R500_ap range {min(v['R500_ap_Mpc'] for v in rows.values()):.2f}"
              f"-{max(v['R500_ap_Mpc'] for v in rows.values()):.2f} Mpc")
    return rows


# X-COP name -> Herbonnet 'cluster' string.  Only exact same-object matches.
XCOP_TO_HERBONNET = {
    "A85": "A85", "A644": "A644", "A1644": "A1644", "A1795": "A1795",
    "A2029": "A2029", "A2142": "A2142", "A2255": "A2255", "A2319": "A2319",
    "A3158": "A3158", "A3266": "A3266",
}


def herbonnet_for_xcop(verbose=False):
    H = load_herbonnet(verbose=verbose)
    by_name = {}
    for v in H.values():
        by_name.setdefault(v["cluster"], []).append(v)
    out = {}
    for xn, hn in XCOP_TO_HERBONNET.items():
        if hn in by_name:
            cand = by_name[hn]
            assert len(cand) == 1, f"ambiguous Herbonnet match for {xn}: {len(cand)}"
            out[xn] = cand[0]
    if verbose:
        print(f"   X-COP x Herbonnet WL overlap: {len(out)} clusters -> "
              f"{sorted(out)}")
    return out


# --------------------------------------------------------------------------
#  BARYON-ONLY radii  (the definitions that cannot be tautological with the
#  total mass, because no total mass enters them)
# --------------------------------------------------------------------------
FB_COSMIC = 0.156          # global constant, Planck Omega_b/Omega_m; NOT per object
NE_THRESHOLD = 1.0e-4      # cm^-3, global constant


def baryonic_radii(c):
    """Two radii built from the gas alone.

    R_b,gas :  mean enclosed GAS density = 500 * rho_c(z) * f_b_cosmic.
               Exactly the R500 definition with M_tot replaced by M_gas/f_b.
               Uses no total mass, no NFW fit, no lensing.
    R_b,ne  :  radius at which n_e crosses a fixed threshold. Uses only the
               deprojected X-ray surface brightness; not even an integral.
    """
    p = build_profile(c)
    r, Mg = p["r"], p["Mgas"]
    target = (4 / 3) * np.pi * 500 * rho_c(c["z"]) * FB_COSMIC * r ** 3
    f = Mg - target                      # positive inside, negative outside
    R_gas = np.nan
    s = np.sign(f)
    idx = np.where(np.diff(s) < 0)[0]
    if len(idx):
        i = idx[-1]
        # log-linear interpolation of the crossing
        lr = np.log(r[i:i + 2])
        lf = f[i:i + 2]
        R_gas = float(np.exp(lr[0] + (lr[1] - lr[0]) * (0 - lf[0]) / (lf[1] - lf[0])))

    ne = c["ne_cm3"]
    rr = 0.5 * (c["r_in_kpc"] + c["r_out_kpc"]) * KPC
    R_ne = np.nan
    j = np.where(ne < NE_THRESHOLD)[0]
    if len(j) and j[0] > 0:
        k = j[0] - 1
        lne = np.log(ne[k:k + 2])
        lrr = np.log(rr[k:k + 2])
        R_ne = float(np.exp(lrr[0] + (lrr[1] - lrr[0])
                            * (np.log(NE_THRESHOLD) - lne[0]) / (lne[1] - lne[0])))
    return R_gas, R_ne


def load_all(verbose=True):
    if verbose:
        print("INGEST -- X-COP hydrostatic profiles")
    cl = []
    for nm in sorted(os.listdir(XCOP_DIR)):
        if not os.path.isdir(os.path.join(XCOP_DIR, nm)):
            continue
        c = load_xcop_cluster(nm, verbose=verbose)
        if c is not None:
            cl.append(c)
    got = sorted(x["name"] for x in cl)
    assert got == sorted(XCOP_EXPECTED), (
        f"X-COP sample changed: got {got}, expected {sorted(XCOP_EXPECTED)}")
    if verbose:
        print(f"   {len(cl)}/12 X-COP clusters loaded, identifiers verified")
        print("INGEST -- Herbonnet+2020 weak-lensing masses")
    return cl


if __name__ == "__main__":
    cl = load_all(verbose=True)
    herb = herbonnet_for_xcop(verbose=True)
    pts = xcop_points(cl)
    print(f"\n   {len(pts)} X-COP points after the bench cut "
          f"({R_MIN_KPC:.0f}-{R_MAX_KPC:.0f} kpc, go>0, gb>0)")
    r = np.array([p["r"] for p in pts])
    gb = np.array([p["gb"] for p in pts])
    go = np.array([p["go"] for p in pts])
    R5 = np.array([p["R500_hse"] for p in pts])
    y = rar_residual(gb, go)
    assert np.all(np.isfinite(y)), "non-finite residual"
    print(f"   r/R500 range {(r/R5).min():.3f} - {(r/R5).max():.3f}")
    print(f"   r range      {(r/KPC).min():.1f} - {(r/KPC).max():.1f} kpc")
    print(f"   residual     {y.min():+.3f} to {y.max():+.3f} dex")
    print("\n   baryon-only radii:")
    for c in cl:
        Rg, Rn = baryonic_radii(c)
        print(f"      {c['name']:<8} R500_hse={c['R500_hse']/KPC:7.1f}  "
              f"R_b,gas={Rg/KPC if np.isfinite(Rg) else float('nan'):7.1f}  "
              f"R_b,ne={Rn/KPC if np.isfinite(Rn) else float('nan'):7.1f} kpc")
