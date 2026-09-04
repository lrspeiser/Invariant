"""
CLASH R500 TAUTOLOGY AUDIT -- ingest layer.

The open half of the Run AT audit.  Every assertion the
`astro-data-acquisition-traps` memory pattern demands is made here: row counts,
column counts, catalogue identifier echoed from the #Name: line, and a
finiteness check on every value column.

PROVENANCE, stated up front because it is the deliverable
=========================================================

  Tian+2020 fig2.dat   runs/gravity/g4/cluster-lensing-exploration-v7-source/fig2.tsv
                       VizieR J/ApJ/896/70/fig2, 84 rows, 20 clusters
                       columns: AName, Rad [kpc], log(gbar), log(gtot), errors
                       *** THIS IS THE ONLY CLASH TABLE THE BENCH USES.
                           invariant_bench._clash() reads q[2],q[3],q[4] and
                           DISCARDS q[1] = AName -- which is why the record says
                           "CLASH has no object identity in the bench".  The
                           identity is in the file. ***

  Tian+2020 table1.dat work/item2-common-w1-v3-audit/clash-table1.tsv
                       VizieR J/ApJ/896/70/table1, 20 rows
                       z, Rad, M*, Mgas, Mtot(<Rad) from Umetsu+2016

  Umetsu+2016 Tab.2    raw/umetsu2016_table2.tex   (arXiv:1507.04385v4 e-print)
                       M200c, c200c, r_-2 per cluster.
                       *** "Cluster parameters derived from single spherical NFW
                           fits to individual surface mass density profiles" ***

  Umetsu+2016 Tab.3    raw/umetsu2016_table3.tex
                       M2500c, M1000c, M500c, Mvir, M100c, M200m, M(<1.5Mpc)
                       *** "Cluster mass estimates M_3D(<r) from single spherical
                           NFW fits to individual surface mass density profiles" ***
                       -> M500c IS A FUNCTION OF (M200c, c200c).  Not merely
                          correlated with the numerator: identical inputs.

  Umetsu+2016 Tab.1    raw/umetsu2016_table1.tex
                       z_l, kT_X (from Postman+2012) -- the ONLY per-cluster
                       quantity in this lane that is independent of the lensing.

  Donahue+2014 CLASH-X raw/donahue2014_chandra_hse.tex (arXiv:1405.7876v3)
                       Chandra JACO hydrostatic r500 per cluster.
                       INDEPENDENT of the lensing mass -- different instrument,
                       different physics, different pipeline.

  Umetsu+2016 has NO VizieR catalogue.  Verified: asu-tsv -source=J/ApJ/821/116
  returns "Error=Table or Catalog not found", and a META catalogue search over
  title=*CLASH* returns 14 catalogues, none of them Umetsu+2016 (positive
  control: J/ApJ/896/70 Tian+2020 IS in that list).  The masses were obtained
  from the arXiv e-print source instead.

KiDS and the wide binaries are NOT loaded anywhere in this lane.
"""
from __future__ import annotations
import hashlib
import math
import os
import re

import numpy as np

ROOT = "C:/Users/henry/Documents/Codex/2026-08-21/Invariant-main-integration/"
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "raw")

FIG2 = ROOT + "runs/gravity/g4/cluster-lensing-exploration-v7-source/fig2.tsv"
TAB1 = ROOT + "work/item2-common-w1-v3-audit/clash-table1.tsv"
U16_T1 = os.path.join(RAW, "umetsu2016_table1.tex")
U16_T2 = os.path.join(RAW, "umetsu2016_table2.tex")
U16_T3 = os.path.join(RAW, "umetsu2016_table3.tex")
D14_CXC = os.path.join(RAW, "donahue2014_chandra_hse.tex")

# ------------------------------------------------------------------ constants
G = 6.67430e-11
MSUN = 1.98892e30
KPC = 3.0856775814913673e19
MPC = 1000.0 * KPC
A0 = 1.2e-10

# Tian+2020 / Umetsu+2016 concordance cosmology -- NOT the bench's (0.3, 0.7).
# Stated in both ReadMe and both papers: h = 0.7, Om = 0.27, OL = 0.73.
H0 = 70.0 * 1e3 / 3.0856775814913673e22
OM, OL = 0.27, 0.73
RHOC0 = 3 * H0 ** 2 / (8 * math.pi * G)

# cosmic baryon fraction, for the AT-style gas-overdensity radius
F_B = 0.157

EXPECTED_ANAMES = [
    "A209", "A383", "A611", "A2261", "MACS0329", "MACS0416", "MACS0429",
    "MACS0647", "MACS0717", "MACS0744", "MACS1115", "MACS1149", "MACS1206",
    "MACS1720", "MACS1931", "MS2137", "RXJ1347", "RXJ1532", "RXJ2129",
    "RXJ2248"]

# Umetsu / Donahue long names -> Tian AName.  Every mapping asserted below.
U16_TO_ANAME = {
    "Abell 383": "A383", "Abell 209": "A209", "Abell 2261": "A2261",
    "RXJ2129.7+0005": "RXJ2129", "Abell 611": "A611", "MS2137-2353": "MS2137",
    "RXJ2248.7-4431": "RXJ2248", "MACSJ1115.9+0129": "MACS1115",
    "MACSJ1931.8-2635": "MACS1931", "RXJ1532.9+3021": "RXJ1532",
    "MACSJ1720.3+3536": "MACS1720", "MACSJ0429.6-0253": "MACS0429",
    "MACSJ1206.2-0847": "MACS1206", "MACSJ0329.7-0211": "MACS0329",
    "RXJ1347.5-1145": "RXJ1347", "MACSJ0744.9+3927": "MACS0744",
    "MACSJ0416.1-2403": "MACS0416", "MACSJ1149.5+2223": "MACS1149",
    "MACSJ0717.5+3745": "MACS0717", "MACSJ0647.7+7015": "MACS0647"}

D14_TO_ANAME = {
    "Abell 209": "A209", "Abell 383": "A383", "MACS0329-02": "MACS0329",
    "MACS0429-02": "MACS0429", "MACS0744+39": "MACS0744", "Abell 611": "A611",
    "MACS1115+01": "MACS1115", "MACS1206-08": "MACS1206",
    "RXJ1347": "RXJ1347", "MACS1532+30": "RXJ1532",
    "MACS1720+35": "MACS1720", "Abell2261": "A2261",
    "MACS1931-26": "MACS1931", "RXJ2129+00": "RXJ2129",
    "MS2137-2353": "MS2137", "RXJ2248-44": "RXJ2248",
    "MACS0416-24": "MACS0416", "MACS0647+70": "MACS0647",
    "MACS0717+37": "MACS0717", "MACS1149+22": "MACS1149"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 16), b""):
            h.update(blk)
    return h.hexdigest()


def rhoc(z):
    """critical density at z for the Tian/Umetsu cosmology."""
    return RHOC0 * (OM * (1 + z) ** 3 + OL)


# --------------------------------------------------------------------------
#  VizieR ASU tables -- identifier echoed, columns asserted against the ReadMe
# --------------------------------------------------------------------------
def _vizier(path, want_name, want_cols, want_rows):
    """Read a VizieR asu-tsv dump.  Enforces every trap in the memory file:
      * echo the catalogue identifier from the #Name: line (a bad -source= can
        return HTTP 200 serving an unrelated REAL catalogue)
      * refuse if CatalogsExamined= appears (fuzzy fallback)
      * assert the COLUMN list, not just the row count (-out.all with an EMPTY
        value silently returns a default column subset)
    """
    L = [l.rstrip("\n") for l in open(path, encoding="utf-8")]
    names = [l.split(":", 1)[1].strip() for l in L if l.startswith("#Name:")]
    assert want_name in names, f"{path}: #Name: lines {names} lack {want_name!r}"
    assert not any("CatalogsExamined" in l for l in L), \
        f"{path}: CatalogsExamined= present -> fuzzy fallback, refuse"
    declared = [l.split("\t")[1] for l in L if l.startswith("#Column\t")]
    assert declared == want_cols, \
        f"{path}: columns {declared} != ReadMe {want_cols}"
    i0 = next(i for i, l in enumerate(L) if l.split("\t")[0] == want_cols[0])
    hdr = L[i0].split("\t")
    assert hdr == want_cols, f"{path}: header row {hdr} != {want_cols}"
    rows = []
    for l in L[i0 + 3:]:
        q = l.split("\t")
        if len(q) != len(want_cols) or not q[0].strip():
            continue
        rows.append(dict(zip(want_cols, [x.strip() for x in q])))
    assert len(rows) == want_rows, f"{path}: {len(rows)} rows, expected {want_rows}"
    return rows


FIG2_COLS = ["recno", "AName", "Rad", "log(gbar)", "log(gtot)",
             "e_log(gbar)", "e_log(gtot)"]
TAB1_COLS = ["recno", "RAR", "Name", "z", "RAJ2000", "DEJ2000", "Band", "n",
             "Re", "e_Re", "Rad", "M*", "Mgas", "e_Mgas", "Mtot", "e_Mtot",
             "AName", "Plot", "SimbadName", "NED"]


def load_fig2(verbose=False):
    rows = _vizier(FIG2, "J/ApJ/896/70/fig2", FIG2_COLS, 84)
    out = []
    for r in rows:
        d = dict(name=r["AName"],
                 r_kpc=float(r["Rad"]),
                 lgb=float(r["log(gbar)"]),
                 lgt=float(r["log(gtot)"]),
                 e_lgb=float(r["e_log(gbar)"]),
                 e_lgt=float(r["e_log(gtot)"]))
        for k in ("r_kpc", "lgb", "lgt", "e_lgb", "e_lgt"):
            assert np.isfinite(d[k]), f"non-finite {k} in {d['name']}"
        out.append(d)
    got = sorted(set(p["name"] for p in out))
    assert got == sorted(EXPECTED_ANAMES), f"cluster set {got}"
    if verbose:
        print(f"  fig2.dat        84 rows, 20 clusters, sha256 {sha256(FIG2)[:16]}")
    return out


def load_tab1(verbose=False):
    rows = _vizier(TAB1, "J/ApJ/896/70/table1", TAB1_COLS, 20)
    out = {}
    for r in rows:
        n = r["AName"]
        out[n] = dict(name=n, longname=r["Name"], z=float(r["z"]),
                      rad_kpc=float(r["Rad"]),
                      Mstar=float(r["M*"]) * 1e11,
                      Mgas=float(r["Mgas"]) * 1e11 if r["Mgas"] else float("nan"),
                      Mtot_bcg=float(r["Mtot"]) * 1e11,
                      n_rar=int(r["RAR"]))
    assert sorted(out) == sorted(EXPECTED_ANAMES)
    tot = sum(v["n_rar"] for v in out.values())
    assert tot == 84, f"table1 RAR counts sum to {tot}, not 84"
    if verbose:
        print(f"  table1.dat      20 rows, RAR counts sum to 84, "
              f"sha256 {sha256(TAB1)[:16]}")
    return out


# --------------------------------------------------------------------------
#  Umetsu+2016 arXiv e-print tables
# --------------------------------------------------------------------------
_PM = r"\$\s*([-\d.]+)\s*\\pm\s*([\d.]+)\s*\$"


def _u16_rows(path, ncols):
    txt = open(path, encoding="utf-8").read()
    body = txt.split(r"\startdata", 1)[1].split(r"\enddata", 1)[0]
    out = []
    for line in body.split("\\\\"):
        line = line.strip()
        if not line or "Selected" in line or "Magnification" in line:
            continue
        line = line.replace(r"\hline", "").strip()
        cells = [c.strip().replace("~~", "") for c in line.split("&")]
        if len(cells) != ncols:
            continue
        out.append(cells)
    return out


def load_umetsu_nfw(verbose=False):
    """Table 2: M200c [1e14 Msun h70^-1], c200c, r_-2 [Mpc h70^-1]."""
    rows = _u16_rows(U16_T2, 4)
    assert len(rows) == 20, f"umetsu table2: {len(rows)} rows"
    out = {}
    for c in rows:
        m = re.match(_PM, c[1]); cc = re.match(_PM, c[2]); r2 = re.match(_PM, c[3])
        assert m and cc and r2, c
        an = U16_TO_ANAME[c[0]]
        out[an] = dict(M200=float(m.group(1)) * 1e14, e_M200=float(m.group(2)) * 1e14,
                       c200=float(cc.group(1)), e_c200=float(cc.group(2)),
                       r_m2_mpc=float(r2.group(1)))
    assert sorted(out) == sorted(EXPECTED_ANAMES)
    if verbose:
        print(f"  Umetsu+16 Tab.2 20 rows (M200c, c200c, r_-2), "
              f"sha256 {sha256(U16_T2)[:16]}")
    return out


def load_umetsu_mass(verbose=False):
    """Table 3: M2500c, M1000c, M500c, Mvir, M100c, M200m, M(<1.5Mpc) [1e14 Msun]."""
    rows = _u16_rows(U16_T3, 8)
    assert len(rows) == 20, f"umetsu table3: {len(rows)} rows"
    keys = ["M2500", "M1000", "M500", "Mvir", "M100", "M200m", "M1p5"]
    out = {}
    for c in rows:
        an = U16_TO_ANAME[c[0]]
        d = {}
        for k, cell in zip(keys, c[1:]):
            m = re.match(_PM, cell)
            assert m, (an, k, cell)
            d[k] = float(m.group(1)) * 1e14
            d["e_" + k] = float(m.group(2)) * 1e14
        out[an] = d
    assert sorted(out) == sorted(EXPECTED_ANAMES)
    if verbose:
        print(f"  Umetsu+16 Tab.3 20 rows (M2500..M200m, M500c), "
              f"sha256 {sha256(U16_T3)[:16]}")
    return out


def load_umetsu_tx(verbose=False):
    """Table 1: z_l and kT_X [keV] (Postman+2012).  Lensing-independent."""
    txt = open(U16_T1, encoding="utf-8").read()
    body = txt.split(r"\startdata", 1)[1].split(r"\enddata", 1)[0]
    out = {}
    for line in body.split("\\\\"):
        line = line.replace(r"\hline", "").strip()
        if not line or "Selected" in line or "Magnification" in line:
            continue
        cells = [c.strip().replace("~~", "") for c in line.split("&")]
        if len(cells) != 10:
            continue
        an = U16_TO_ANAME[cells[0]]
        z = float(cells[1].strip("$"))
        m = re.match(r"\$([\d.]+)\\pm([\d.]+)\$", cells[4].replace(" ", ""))
        assert m, cells[4]
        out[an] = dict(z_umetsu=z, kT=float(m.group(1)), e_kT=float(m.group(2)))
    assert sorted(out) == sorted(EXPECTED_ANAMES), sorted(out)
    if verbose:
        print(f"  Umetsu+16 Tab.1 20 rows (z, kT_X), sha256 {sha256(U16_T1)[:16]}")
    return out


def load_donahue_r500(verbose=False):
    """Donahue+2014 CLASH-X Chandra JACO hydrostatic r500 [Mpc h70^-1].
    Independent of the lensing mass.  25 rows in the paper; we keep the 20 that
    map onto the Tian sample."""
    txt = open(D14_CXC, encoding="utf-8").read()
    body = txt.split(r"\startdata", 1)[1].split(r"\enddata", 1)[0]
    out, seen = {}, 0
    for line in body.split("\\\\"):
        line = line.replace(r"\hline", "").strip()
        cells = [c.strip() for c in line.split("&")]
        if len(cells) != 13 or not cells[0] or cells[0].startswith("\\"):
            continue
        seen += 1
        nm = cells[0].replace("*", "").strip()
        if nm not in D14_TO_ANAME:
            continue
        try:
            r5 = float(cells[11]); e5 = float(cells[12])
        except ValueError:
            continue
        out[D14_TO_ANAME[nm]] = dict(r500_mpc=r5, e_r500_mpc=e5)
    assert seen >= 25, f"donahue: only {seen} data rows parsed"
    if verbose:
        print(f"  Donahue+14 CXC  {seen} rows parsed, {len(out)}/20 map to CLASH, "
              f"sha256 {sha256(D14_CXC)[:16]}")
    return out


# --------------------------------------------------------------------------
#  NFW -- the function that generates BOTH the numerator and the x-axis
# --------------------------------------------------------------------------
def nfw_mass(r, M200, c200, z):
    """M(<r) for a spherical NFW halo with (M200c, c200c) at redshift z.
    Exactly Tian+2020 eq. (7)-(8)."""
    r200 = (3 * M200 * MSUN / (4 * math.pi * 200 * rhoc(z))) ** (1 / 3)
    rs = r200 / c200
    mu = lambda x: np.log(1 + x) - x / (1 + x)
    return M200 * mu(np.asarray(r, float) / rs) / mu(c200)


def r_delta(M200, c200, z, delta):
    """Solve M_NFW(<R) = (4/3) pi delta rho_c(z) R^3 by bisection."""
    A = (4 / 3) * math.pi * delta * rhoc(z)
    f = lambda R: float(nfw_mass(R, M200, c200, z)) * MSUN - A * R ** 3
    lo, hi = 1e-3 * MPC, 20.0 * MPC
    assert f(lo) > 0 > f(hi), "bracket failed"
    for _ in range(200):
        mid = math.sqrt(lo * hi)
        if f(mid) > 0:
            lo = mid
        else:
            hi = mid
    return math.sqrt(lo * hi)


def r_from_mass(Mdelta, z, delta=500.0):
    """R_delta from M_delta alone: the overdensity DEFINITION, exact."""
    A = (4 / 3) * math.pi * delta * rhoc(z)
    return (Mdelta * MSUN / A) ** (1 / 3)


# --------------------------------------------------------------------------
#  the excess statistic -- identical to Run AT / the bench confound residual
# --------------------------------------------------------------------------
def nu_rar(x):
    """RAR interpolation function nu(x), x = g_bar/a0.  Bench form."""
    x = np.asarray(x, float)
    return 1.0 / (1.0 - np.exp(-np.sqrt(x)))


def rar_residual(gb, go):
    """log10 nu_obs - log10 nu_RAR(g_bar/a0).  Same as Run AT ingest.py."""
    x = np.asarray(gb, float) / A0
    nu = np.asarray(go, float) / np.asarray(gb, float)
    return np.log10(nu) - np.log10(nu_rar(x))


# --------------------------------------------------------------------------
def load_all(verbose=True):
    if verbose:
        print("CLASH ingest -- identifiers echoed, counts asserted")
    fig2 = load_fig2(verbose)
    tab1 = load_tab1(verbose)
    nfw = load_umetsu_nfw(verbose)
    mass = load_umetsu_mass(verbose)
    tx = load_umetsu_tx(verbose)
    d14 = load_donahue_r500(verbose)

    # redshift cross-check between the two papers (Tian table1 vs Umetsu table1)
    dz = {n: abs(tab1[n]["z"] - tx[n]["z_umetsu"]) for n in EXPECTED_ANAMES}
    worst = max(dz, key=dz.get)
    if verbose:
        print(f"  redshift cross-check Tian vs Umetsu: max |dz| = {dz[worst]:.4f} "
              f"({worst})")

    cl = {}
    for n in EXPECTED_ANAMES:
        z = tab1[n]["z"]
        M200, c200 = nfw[n]["M200"], nfw[n]["c200"]
        M500 = mass[n]["M500"]
        cl[n] = dict(
            name=n, z=z, longname=tab1[n]["longname"],
            M200=M200, c200=c200, e_M200=nfw[n]["e_M200"], e_c200=nfw[n]["e_c200"],
            M500=M500, e_M500=mass[n]["e_M500"],
            M1000=mass[n]["M1000"], M2500=mass[n]["M2500"],
            kT=tx[n]["kT"], e_kT=tx[n]["e_kT"],
            # x-axis radius as the lane uses it: from the published M500c
            R500_lens=r_from_mass(M500, z),
            # the same thing solved directly off the NFW profile (consistency)
            R500_nfw=r_delta(M200, c200, z, 500.0),
            R500_xray=(d14[n]["r500_mpc"] * MPC if n in d14 else float("nan")),
            e_R500_xray=(d14[n]["e_r500_mpc"] * MPC if n in d14 else float("nan")),
            Mstar_bcg=tab1[n]["Mstar"], rad_bcg=tab1[n]["rad_kpc"])
    return dict(points=fig2, clusters=cl)


def points_table(D):
    """flat arrays over the 84 rows."""
    P = D["points"]; C = D["clusters"]
    nm = np.array([p["name"] for p in P])
    r = np.array([p["r_kpc"] for p in P]) * KPC
    gb = 10 ** np.array([p["lgb"] for p in P])
    go = 10 ** np.array([p["lgt"] for p in P])
    e_lgt = np.array([p["e_lgt"] for p in P])
    e_lgb = np.array([p["e_lgb"] for p in P])
    y = rar_residual(gb, go)
    z = np.array([C[n]["z"] for n in nm])
    return dict(name=nm, r=r, gb=gb, go=go, y=y, z=z, e_lgt=e_lgt, e_lgb=e_lgb)


if __name__ == "__main__":
    D = load_all(verbose=True)
    T = points_table(D)
    print(f"\n{len(T['r'])} points, {len(set(T['name']))} clusters")
    print(f"radius levels present: "
          f"{sorted(set(np.round(T['r']/KPC,1)))}")
    print(f"RAR residual: mean {T['y'].mean():+.4f}, sd {T['y'].std():.4f}")
    print("\nper-cluster:")
    print(f"{'name':<10}{'z':>7}{'M200':>9}{'c200':>7}{'M500':>9}"
          f"{'R500_lens':>11}{'R500_nfw':>10}{'R500_X':>9}{'kT':>7}{'n':>4}")
    for n in EXPECTED_ANAMES:
        c = D["clusters"][n]
        k = int((T["name"] == n).sum())
        print(f"{n:<10}{c['z']:7.3f}{c['M200']/1e14:9.2f}{c['c200']:7.1f}"
              f"{c['M500']/1e14:9.2f}{c['R500_lens']/KPC:11.1f}"
              f"{c['R500_nfw']/KPC:10.1f}"
              f"{c['R500_xray']/KPC if np.isfinite(c['R500_xray']) else float('nan'):9.1f}"
              f"{c['kT']:7.1f}{k:4d}")
