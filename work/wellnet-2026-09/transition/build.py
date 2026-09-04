"""PART 1 -- ingest the three surveys into ONE common representation.

Every object becomes a `Cluster`: a radial grid, M_gas(<r), M_star(<r),
g_b(r), a redshift, and -- computed identically for all three surveys under
the FROZEN law -- a dynamical R500 and the gas mass inside it.

Nothing here fits anything.  Counts are asserted and identifiers echoed on
every ingest, per the programme checklist ("silent extraction failures").
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(ROOT))
for p in ("efeds-hsc", "lead01", "closure"):
    q = os.path.join(ROOT, p)
    if q not in sys.path:
        sys.path.insert(0, q)

import pipeline as P                                            # noqa: E402
import lead01 as L                                              # noqa: E402
import efeds_hsc as E                                           # noqa: E402
import decade_test as DT                                        # noqa: E402
import decl                                                     # noqa: E402

MPC, KPC, MSUN, G = P.MPC, P.KPC, P.MSUN, P.G
ARCSEC = P.ARCSEC
CLUSTER_DATA = os.path.join(ROOT, "cluster-data")
AUDIT = os.path.join(REPO, "work", "gravity-cluster-audit-2026-09", "acquire")

R_GRID = np.geomspace(5e-3, 30.0, 480) * MPC       # proper metres


# ----------------------------------------------------------------- utilities
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def rho_c(z):
    """Critical density at z, SI, for the declared flat LCDM (Om=0.3, h=0.7)."""
    Hz = L.H0 * L.E(z)
    return 3.0 * Hz ** 2 / (8.0 * math.pi * G)


def g_rar(g_b, a0=decl.A0_RAR):
    x = np.sqrt(np.maximum(g_b, 1e-30) / a0)
    return g_b / (1.0 - np.exp(-x))


def cumtrap(y, x):
    return np.concatenate([[0.0], np.cumsum(0.5 * (y[1:] + y[:-1]) * np.diff(x))])


def hernquist_M(r, M, a):
    return M * r ** 2 / (r + a) ** 2


# ------------------------------------------------------------------- Cluster
class Cluster:
    """One object, in the common representation.

    r        radial grid, metres
    M_gas    cumulative gas mass, kg
    M_star   cumulative stellar mass, kg
    g_b      baryonic Newtonian field, m/s^2
    R500     DYNAMICAL R500 under the frozen law (metres)
    M_gas500 gas mass inside R500 (kg)
    """

    def __init__(self, cid, survey, z, r, M_gas, M_star, kT_keV=np.nan,
                 extra=None):
        self.id, self.survey, self.z = cid, survey, float(z)
        self.r = r
        self.M_gas = M_gas
        self.M_star = M_star
        self.M_b = M_gas + M_star
        self.g_b = G * self.M_b / r ** 2
        self.kT = float(kT_keV)
        self.extra = extra or {}
        self._solve_r500()

    def _solve_r500(self):
        """R500 where M_dyn(R) = 500 rho_c(z) (4pi/3) R^3, under the frozen law.

        Uses the BARYONS and the frozen law only -- no lensing mass, no NFW,
        no hydrostatic mass.  Identical in all three surveys.
        """
        g = g_rar(self.g_b)
        M_dyn = g * self.r ** 2 / G
        target = 500.0 * rho_c(self.z) * (4.0 * math.pi / 3.0) * self.r ** 3
        f = M_dyn - target
        s = np.where((f[:-1] > 0) & (f[1:] <= 0))[0]
        if s.size == 0:
            self.R500 = float("nan")
            self.M_gas500 = float("nan")
            return
        i = int(s[0])
        t = f[i] / (f[i] - f[i + 1])
        self.R500 = float(self.r[i] + t * (self.r[i + 1] - self.r[i]))
        self.M_gas500 = float(np.interp(self.R500, self.r, self.M_gas))
        self.M_b500 = float(np.interp(self.R500, self.r, self.M_b))

    def M_dyn(self, r):
        g = g_rar(np.interp(r, self.r, self.g_b))
        return g * r ** 2 / G

    def g_b_at(self, r):
        return np.interp(r, self.r, self.g_b)


# ================================================================== 1a. eFEDS
def build_efeds(verbose=True):
    recs, cuts = E.load_efeds()
    prof = DT.load_profiles()
    obs = DT.Obs(recs, prof)
    syss = [P.System(rc) for rc in obs.sys]
    n_pt = sum(len(r) for r in obs.rows)
    assert len(obs) == 496 and n_pt == 3365, (len(obs), n_pt)
    clusters = []
    for s in syss:
        # eFEDS carries no stellar catalogue.  f_star = 0 is the closure lane's
        # primary; its sensitivity was measured there (< 0.07 dex for
        # f_star = 0 / 0.15 / 0.30) and is re-checked in PART 6 here.
        c = Cluster(s.id, "efeds", s.z, s.r, s.M_gas, np.zeros_like(s.M_gas),
                    kT_keV=s.T, extra=dict(R500_cat=s.R500,
                                           Mgas500_pub=s.Mgas500_pub))
        clusters.append(c)
    if verbose:
        print(f"   eFEDS   ingested {len(clusters)} systems / {n_pt} shear "
              f"points  (assert 496 / 3365)  cuts={cuts}")
        print(f"           identifiers echoed: {clusters[0].id}, "
              f"{clusters[1].id}, ... {clusters[-1].id}")
    return clusters, obs, syss, cuts


# ================================================================= ACCEPT gas
_ACCEPT_MASTER = None


def accept_master():
    """The whole ACCEPT deprojected-profile table, keyed by cluster name.

    One source file for every ACCEPT cluster used in this lane -- the LoCuSS
    overlap and the strong-lens targets alike -- so the two ingests cannot
    diverge.
    """
    global _ACCEPT_MASTER
    if _ACCEPT_MASTER is None:
        d = {}
        path = os.path.join(CLUSTER_DATA, "gas", "accept_all_profiles.dat")
        nrow = 0
        with open(path) as f:
            for line in f:
                if line.startswith("#"):
                    continue
                p = line.split()
                if len(p) < 14:
                    continue
                nrow += 1
                d.setdefault(p[0], []).append(
                    (float(p[1]), float(p[2]), float(p[3]), float(p[4]),
                     float(p[12])))
        assert len(d) >= 240, len(d)
        _ACCEPT_MASTER = (d, nrow)
    return _ACCEPT_MASTER


def read_accept(name):
    """ACCEPT deprojected n_e shells for one cluster, sorted OUTWARD."""
    d, _ = accept_master()
    rows = sorted(d[name])
    rm = np.array([0.5 * (a + b) for a, b, _, _, _ in rows]) * MPC
    ne = np.array([c for _, _, c, _, _ in rows])
    nee = np.array([e for _, _, _, e, _ in rows])
    tx = np.array([t for _, _, _, _, t in rows])
    return rm, ne, nee, tx, len(rows)


def gas_from_accept(name, rgrid=R_GRID, slope_lo=-4.5, slope_hi=-1.2):
    """M_gas(<r) on rgrid from an ACCEPT deprojected n_e profile.

    Declared extrapolation: log-log power law from the outer 5 shells for
    r > r_max (slope clipped to [slope_lo, slope_hi]); inner 3 shells for
    r < r_min (slope clipped to [-2, 0]).  ACCEPT profiles for these clusters
    reach 0.9-1.4 Mpc, which is near R500, so the outward extrapolation is
    load-bearing only for R500 itself and is bracketed in PART 6.
    """
    rm, ne, nee, tx, nrow = read_accept(name)
    lr, ln = np.log(rm), np.log(np.maximum(ne, 1e-12))
    k = min(5, len(rm))
    so = np.polyfit(lr[-k:], ln[-k:], 1)[0]
    so = float(np.clip(so, slope_lo, slope_hi))
    k2 = min(3, len(rm))
    si = float(np.clip(np.polyfit(lr[:k2], ln[:k2], 1)[0], -2.0, 0.0))
    lg = np.log(rgrid)
    out = np.interp(lg, lr, ln)
    out = np.where(lg > lr[-1], ln[-1] + so * (lg - lr[-1]), out)
    out = np.where(lg < lr[0], ln[0] + si * (lg - lr[0]), out)
    ne_g = np.exp(out) * 1e6                      # cm^-3 -> m^-3
    rho = L.MU_E * L.M_P * ne_g
    Mg = cumtrap(4.0 * math.pi * rgrid ** 2 * rho, rgrid)
    return Mg, dict(n_shells=nrow, r_min_Mpc=float(rm[0] / MPC),
                    r_max_Mpc=float(rm[-1] / MPC), outer_slope=so,
                    inner_slope=si,
                    Tx_outer_keV=float(np.median(tx[len(tx) // 2:])))


# ================================================================= 1b. LoCuSS
def build_locuss(verbose=True):
    """LoCuSS: Mulroy+2019 sample x observables, gas profile from ACCEPT.

    S_i = M_WL,i / M_dyn(r500_WL,i).

    CONSTRAINT-2 LABEL, stated plainly: M_WL is Okabe & Smith (2016)'s
    NFW-FITTED M_500 from Subaru tangential shear.  It is a parametric lens
    model, not raw shear.  The weak-lensing availability audit in
    cluster-data/ establishes that NO public per-source shear catalogue exists
    for any LoCuSS cluster, so there is no raw-shear route to this sample.
    The point is carried, LABELLED, and its sensitivity to that assumption is
    reported rather than hidden.
    """
    def rd(path):
        rows = []
        with open(path, encoding="utf-8") as f:
            hdr = f.readline().rstrip("\n").split("\t")
            for line in f:
                p = line.rstrip("\n").split("\t")
                if len(p) != len(hdr):
                    continue
                rows.append(dict(zip(hdr, p)))
        return hdr, rows

    h1, samp = rd(os.path.join(AUDIT, "mulroy2019_sample.tsv"))
    h2, obsv = rd(os.path.join(AUDIT, "mulroy2019_observables.tsv"))
    assert len(samp) == 41 and len(obsv) == 41, (len(samp), len(obsv))
    assert len(h1) == 13 and len(h2) == 28, (len(h1), len(h2))
    ob = {r["Name"]: r for r in obsv}

    import re

    def norm(s):
        s = s.upper().replace("_", "").replace(" ", "")
        s = s.replace("ABELL", "A").replace("RXCJ", "RXJ").replace("ZWCL", "ZW")
        return re.sub(r"^A0*", "A", s)

    master, _ = accept_master()
    amap = {norm(n): n for n in master}

    cuts = dict(ingested=len(samp), has_accept=0, has_LK=0, finite_MWL=0,
                r500_solved=0)
    clusters, points = [], []
    for r1 in samp:
        nm = r1["Name"]
        an = amap.get(norm(nm))
        if an is None or an not in master:
            continue
        cuts["has_accept"] += 1
        r2 = ob.get(nm, {})
        try:
            LK = float(r2.get("L_K_tot", ""))
        except ValueError:
            continue
        cuts["has_LK"] += 1
        try:
            M_WL = float(r1["M_WL"]) * 1e14 * MSUN
            eM_p = float(r1["M_WL_ep"]) * 1e14 * MSUN
            eM_m = float(r1["M_WL_em"]) * 1e14 * MSUN
            z = float(r1["z"])
        except ValueError:
            continue
        if not (M_WL > 0):
            continue
        cuts["finite_MWL"] += 1

        Mg, ginfo = gas_from_accept(an)

        # stars: M* = 0.73 L_K (Run K step 2), Hernquist a = 100 kpc DECLARED
        Mstar_tot = 0.73 * LK * 1e12 * MSUN
        Ms = hernquist_M(R_GRID, Mstar_tot, 100.0 * KPC)

        try:
            kT = float(r2.get("kT_X_ce", "nan"))
        except ValueError:
            kT = ginfo["Tx_outer_keV"]
        c = Cluster(nm, "locuss", z, R_GRID, Mg, Ms, kT_keV=kT,
                    extra=dict(accept_name=an, M_WL=M_WL,
                               eM_WL=0.5 * (eM_p + eM_m), L_K=LK,
                               Mgas_pub=r2.get("M_gas", ""), gas=ginfo))
        if not np.isfinite(c.R500):
            continue
        cuts["r500_solved"] += 1
        # aperture radius: r500 from the LENSING mass, which is where M_WL is
        # quoted.  M_500 = 500 rho_c (4pi/3) r500^3.
        r500_wl = (3.0 * M_WL / (4.0 * math.pi * 500.0 * rho_c(z))) ** (1.0 / 3.0)
        c.extra["r500_WL"] = r500_wl
        clusters.append(c)
        points.append(dict(cid=nm, r=r500_wl, M_WL=M_WL,
                           frac_err=0.5 * (eM_p + eM_m) / M_WL))
    if verbose:
        print(f"   LoCuSS  ingested 41 clusters / 41 observable rows "
              f"(assert 41/41, 13 and 28 columns) -> {len(clusters)} usable")
        print(f"           cuts={cuts}")
        print(f"           identifiers echoed: "
              f"{', '.join(c.id for c in clusters[:3])} ... {clusters[-1].id}")
    return clusters, points, cuts


# ====================================================== 1c. strong-lens cores
# The five Hubble Frontier Field clusters that have BOTH a public
# multiple-image catalogue with spectroscopic redshifts AND an ACCEPT
# deprojected gas profile.  MACS J0416 has images but no ACCEPT profile and is
# therefore excluded, with the reason recorded.
SL_TARGETS = [
    dict(key="A2744", accept="ABELL_2744",
         images="A2744_multiple_images_Bergamini2023_GLASS-JWST.tsv",
         bcg="Abell 2744", z=0.308),
    dict(key="A370", accept="ABELL_0370",
         images="A370_multiple_images_Lagattuta2019.tsv",
         bcg="Abell 370", z=0.375),
    dict(key="AS1063", accept="ABELL_1063S",
         images="AS1063_multiple_images_Caminha2016.tsv",
         bcg="Abell S1063", z=0.348),
    dict(key="MACSJ0717", accept="MACS_J0717.5+3745",
         images="MACSJ0717_multiple_images_Limousin2016.tsv",
         bcg="MACSJ0717.5+3745", z=0.545),
    dict(key="MACSJ1149", accept="MACS_J1149.5+2223",
         images="MACSJ1149_multiple_images_Treu2016.tsv",
         bcg="MACSJ1149.5+2223", z=0.543),
]
SL_EXCLUDED = {"MACSJ0416": "multiple-image catalogue present "
                            "(Caminha+2017, Bergamini+2023) but no ACCEPT "
                            "deprojected n_e profile, so no baryon model"}

# DECLARED universal cluster stellar template: the two-Hernquist fit the
# Refsdal lane made to MACS J1149's measured projected cumulative M* (rms
# 0.051 dex over 5-468 kpc), scaled linearly by M_gas,500.  Bracketed x0.5,
# x2 in PART 6; the Refsdal lane measured that bracket as -3.4% / +1.8% on
# the recovered response, so it is not load-bearing.
STAR_TEMPLATE = [(8.1411769485937e10, 0.5), (1.3826721180738979e13, 1200.0)]
STAR_TEMPLATE_MGAS500 = None      # filled from MACS J1149 at build time


def read_bcgs():
    path = os.path.join(CLUSTER_DATA, "bcg",
                        "shipley2018_hff_BCG_brightest_per_cluster.tsv")
    rows = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == len(hdr):
                rows.append(dict(zip(hdr, p)))
    assert len(rows) == 6, len(rows)
    return {r["Cluster"]: r for r in rows}


def read_images(fn):
    path = os.path.join(CLUSTER_DATA, "stronglensing", fn)
    rows = []
    with open(path, encoding="utf-8") as f:
        hdr = f.readline().rstrip("\n").split("\t")
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) == len(hdr):
                rows.append(dict(zip(hdr, p)))
    return rows


def read_mcxc():
    """MCXC (Piffaretti+2011) R500, the EXTERNAL catalogue aperture for the
    strong-lens cores.  R500 is used here as an aperture LABEL, not as a mass
    measurement; it is matched by sky position with the separation printed."""
    path = os.path.join(CLUSTER_DATA, "gas", "mcxc_raw.tsv")
    hdr, rows = None, []
    for line in open(path, encoding="utf-8", errors="replace"):
        if line.startswith("recno\t"):
            hdr = line.rstrip("\n").split("\t")
            continue
        if hdr is None or line.startswith("#") or not line.strip():
            continue
        p = line.rstrip("\n").split("\t")
        if len(p) == len(hdr):
            rows.append(dict(zip(hdr, p)))
    assert len(rows) > 1700, len(rows)

    def hms(s):
        a = s.split()
        return (float(a[0]) + float(a[1]) / 60 + float(a[2]) / 3600) * 15.0

    def dms(s):
        a = s.split()
        sg = -1.0 if a[0].strip().startswith("-") else 1.0
        return sg * (abs(float(a[0])) + float(a[1]) / 60 + float(a[2]) / 3600)

    out = []
    for r in rows:
        try:
            out.append((hms(r["RAJ2000"]), dms(r["DEJ2000"]),
                        float(r["R500"]), float(r["M500"]), r["MCXC"]))
        except (ValueError, IndexError):
            continue
    return out, len(rows)


MCXC_MAX_SEP_ARCMIN = 3.0        # declared match radius


# A multiply-imaged system tells you about the CLUSTER's critical curve only
# if it is lensed by the cluster monopole rather than by a member galaxy or a
# substructure.  The largest cluster Einstein radii known are ~55 arcsec, so a
# system whose mean image radius is far beyond that is not a cluster-scale
# critical-curve tracer.  THETA_MAX is the declared outlier cut; None keeps
# everything, and both are reported.
def build_sl(verbose=True, star_mult=1.0, theta_max=None):
    bcgs = read_bcgs()
    mcxc, n_mcxc = read_mcxc()
    out, cuts = [], {}
    # normalisation reference for the stellar template
    ref_Mg, _ = gas_from_accept("MACS_J1149.5+2223")
    ref_star = hernquist_M(R_GRID, STAR_TEMPLATE[0][0] * MSUN,
                           STAR_TEMPLATE[0][1] * KPC) \
        + hernquist_M(R_GRID, STAR_TEMPLATE[1][0] * MSUN,
                      STAR_TEMPLATE[1][1] * KPC)
    ref = Cluster("ref", "sl", 0.543, R_GRID, ref_Mg, ref_star)
    ref_Mgas500 = ref.M_gas500

    for t in SL_TARGETS:
        Mg, ginfo = gas_from_accept(t["accept"])
        c0 = Cluster(t["key"], "sl", t["z"], R_GRID, Mg, np.zeros_like(Mg))
        scale = star_mult * c0.M_gas500 / ref_Mgas500
        Ms = scale * ref_star
        c = Cluster(t["key"], "sl", t["z"], R_GRID, Mg, Ms,
                    kT_keV=ginfo["Tx_outer_keV"],
                    extra=dict(gas=ginfo, star_scale=float(scale),
                               accept=t["accept"]))
        b = bcgs[t["bcg"]]
        ra0, de0 = float(b["RAdeg_J2000"]), float(b["DEdeg_J2000"])
        imgs = read_images(t["images"])
        n_all = len(imgs)
        DA = float(P.d_ang(t["z"]))
        by = {}
        for r in imgs:
            if r.get("z_flag", "").strip() != "spec":
                continue
            try:
                zs = float(r["z"])
                ra, de = float(r["ra_deg"]), float(r["dec_deg"])
            except ValueError:
                continue
            if not (zs > t["z"] + 0.1):
                continue
            dra = (ra - ra0) * math.cos(math.radians(de0))
            dde = de - de0
            th = math.hypot(dra, dde) * 3600.0        # arcsec
            by.setdefault(r["system_id"], []).append((th, zs))
        systems = []
        for sid, v in sorted(by.items()):
            if len(v) < 2:                            # need >= 2 images
                continue
            th = np.array([a for a, _ in v])
            zs = float(np.median([b for _, b in v]))
            if theta_max is not None and th.mean() > theta_max:
                continue
            systems.append(dict(system_id=sid, n_img=len(v),
                                theta_as=float(th.mean()),
                                theta_sd=float(th.std(ddof=1)),
                                z_s=zs))
        best = min(mcxc, key=lambda m: math.hypot(
            (m[0] - ra0) * math.cos(math.radians(de0)), m[1] - de0))
        sep = math.hypot((best[0] - ra0) * math.cos(math.radians(de0)),
                         best[1] - de0) * 60.0
        if sep <= MCXC_MAX_SEP_ARCMIN:
            c.extra["R500_cat"] = best[2] * MPC
            c.extra["mcxc"] = dict(name=best[4], sep_arcmin=sep,
                                   M500_1e14=best[3])
        else:
            c.extra["R500_cat"] = None
            c.extra["mcxc"] = dict(nearest=best[4], sep_arcmin=sep)
        cuts[t["key"]] = dict(images_in_file=n_all,
                              spec_images=sum(len(v) for v in by.values()),
                              systems_with_2plus=len(systems),
                              mcxc=c.extra["mcxc"])
        c.extra["DA"] = DA
        c.extra["bcg"] = (ra0, de0)
        out.append((c, systems))
        if verbose:
            rc = c.extra["R500_cat"]
            print(f"   SL      {t['key']:10s} accept={t['accept']:20s} "
                  f"images={n_all:3d} spec-systems={len(systems):3d} "
                  f"gas shells={ginfo['n_shells']:2d} "
                  f"r=[{ginfo['r_min_Mpc']:.3f},{ginfo['r_max_Mpc']:.3f}] Mpc "
                  f"R500_dyn={c.R500/MPC:.3f} "
                  f"R500_MCXC={'%.3f' % (rc / MPC) if rc else 'ABSENT'} "
                  f"(sep {c.extra['mcxc']['sep_arcmin']:.2f}')")
    if verbose:
        print(f"   SL      MCXC rows read: {n_mcxc}; match radius "
              f"{MCXC_MAX_SEP_ARCMIN}'")
        print(f"   SL      excluded: {SL_EXCLUDED}")
    return out, cuts


# ------------------------------------------------------------------ manifest
def input_manifest():
    files = [
        os.path.join(ROOT, "efeds-hsc", "decade_efeds_shear_profiles.tsv"),
        os.path.join(ROOT, "lead01", "efeds_bahar2022_table1_density.tsv"),
        os.path.join(ROOT, "lead01", "efeds_bahar2022_table2.tsv"),
        os.path.join(AUDIT, "mulroy2019_sample.tsv"),
        os.path.join(AUDIT, "mulroy2019_observables.tsv"),
        os.path.join(CLUSTER_DATA, "gas", "accept_all_profiles.dat"),
        os.path.join(CLUSTER_DATA, "bcg",
                     "shipley2018_hff_BCG_brightest_per_cluster.tsv"),
    ]
    for t in SL_TARGETS:
        files.append(os.path.join(CLUSTER_DATA, "stronglensing", t["images"]))
    return {os.path.basename(f): sha(f) for f in files if os.path.exists(f)}


if __name__ == "__main__":
    print("declaration sha256:", sha(os.path.join(HERE, "decl.py")))
    ef, obs, syss, c1 = build_efeds()
    lo, lop, c2 = build_locuss()
    sl, c3 = build_sl()
    print(json.dumps(input_manifest(), indent=1))
