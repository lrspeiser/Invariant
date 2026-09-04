"""TEST 1 stage 1 -- build the MaNGA galaxy table with environment attached.

Joins, in order:
  DRPall (MaNGA DR17, v3_1_1)          -> targeting, redshift, NSA photometry
  DAPall HYB10-MILESHC-MASTARSSP       -> resolved stellar + ionised-gas kinematics summary
  MaNGA PyMorph DR17 (g, r, i)         -> Sersic and Ser+Exp (bulge+disk) photometry
  MaNGA Deep-Learning morphology DR17  -> T-Type, P_LTG, P_edge, P_bar
  MaNGA visual morphology 2.0.1        -> visual T-Type, edge-on and bar flags
  HI-MaNGA v2_0_1                      -> M_HI (or its upper limit)
  GEMA 2.0.2                           -> tidal strength Q, local density, LSS tidal tensor
  Tempel+2014 (SDSS DR10 FoF groups)   -> group membership, sigma_v (OBSERVED), Rvir
  Tempel+2017 (SDSS DR12 FoF groups)   -> independent membership, R200, M200
  MCXC (Piffaretti+2011)               -> X-ray confirmation of the host, L500 (OBSERVED)

Everything derived is computed in ONE place (derive()) so the provenance of each
derived column is auditable.  Nothing here uses a dark-matter-dependent quantity
as an observation; the columns that ARE dark-matter dependent (M200, MNFW, M500)
are carried through with a `_rank_only` suffix so they cannot be used by accident.
"""
import hashlib
import json
import os
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vizier_tsv import read_vizier_tsv, num

LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
RAW = os.path.join(LANE, "raw")
CLEAN = os.path.join(LANE, "clean")
MANGA = os.path.join(RAW, "manga")
GROUPS = os.path.join(RAW, "groups")

# Cosmology used for every distance in this lane.  Tempel+2014/2017 used
# H0=100h with h=1 for the tabulated Mpc/h quantities; see rescale() below.
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
H_TEMPEL = 1.0     # Tempel distances are in Mpc/h with h=1.0 (their eq. sect. 2)
G_MSUN_KPC_KMS = 4.30091e-6   # G in kpc (km/s)^2 / Msun

LOG = []


def say(msg):
    print(msg, flush=True)
    LOG.append(msg)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def native(a):
    """FITS arrays are big-endian; pandas/numpy ops need native byte order."""
    a = np.asarray(a)
    if a.dtype.kind in "SU":
        return a
    if a.dtype.byteorder not in ("=", "|"):
        return a.astype(a.dtype.newbyteorder("="))
    return a


def strip_bytes(a):
    """FITS string columns -> stripped python str."""
    return np.array([s.strip() if isinstance(s, str) else s.decode().strip() for s in a])


# ----------------------------------------------------------------- ingest
def load_drpall():
    p = os.path.join(MANGA, "drpall-v3_1_1.fits")
    with fits.open(p) as h:
        d = h[1].data
        n_all = len(d)
        keep = ["plateifu", "mangaid", "objra", "objdec", "z", "nsa_z", "nsa_zdist",
                "nsa_elpetro_mass", "nsa_sersic_mass", "nsa_elpetro_th50_r",
                "nsa_sersic_th50", "nsa_sersic_n", "nsa_sersic_ba", "nsa_sersic_phi",
                "nsa_elpetro_ba", "nsa_elpetro_phi", "nsa_iauname",
                "mngtarg1", "mngtarg2", "mngtarg3", "drp3qual", "srvymode",
                "ifudesignsize", "bluesn2", "redsn2", "ebvgal", "nsa_nsaid"]
        out = {}
        for c in keep:
            col = d[c]
            out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
        # NSA elpetro absolute magnitudes: FNugriz -> index 3,4,5 = g,r,i
        am = native(d["nsa_elpetro_absmag"])
        for i, b in zip((3, 4, 5), ("g", "r", "i")):
            out["nsa_absmag_" + b] = am[:, i]
    df = pd.DataFrame(out)
    say("DRPall MANGA rows read: %d (expect 11273)" % n_all)
    assert n_all == 11273, "DRPall row count changed: %d" % n_all
    return df


def load_dapall():
    p = os.path.join(MANGA, "dapall-v3_1_1-3.1.0.fits")
    with fits.open(p) as h:
        hdu = None
        for x in h[1:]:
            if x.header.get("EXTNAME", "").startswith("HYB10-MILESHC-MASTARSSP"):
                hdu = x
        assert hdu is not None, "HYB10-MILESHC-MASTARSSP extension not found in DAPall"
        d = hdu.data
        keep = ["PLATEIFU", "DAPQUAL", "DAPDONE", "SNR_MED", "RCOV90", "BIN_RMAX",
                "STELLAR_SIGMA_1RE", "STELLAR_VEL_LO", "STELLAR_VEL_HI",
                "STELLAR_VEL_LO_CLIP", "STELLAR_VEL_HI_CLIP",
                "HA_GSIGMA_1RE", "HA_GVEL_LO", "HA_GVEL_HI",
                "HA_GVEL_LO_CLIP", "HA_GVEL_HI_CLIP", "SFR_TOT", "SB_1RE"]
        out = {}
        for c in keep:
            col = d[c]
            if col.ndim > 1:
                col = col[:, 0]
            out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
        out["SNR_MED_r"] = native(d["SNR_MED"])[:, 1]
        # H-alpha total flux: emission-line channel 23 = Ha-6564 in DR17 DAP
        out["HA_GFLUX_TOT"] = native(d["EMLINE_GFLUX_TOT"])[:, 23]
        n = len(d)
    df = pd.DataFrame(out).rename(columns={"PLATEIFU": "plateifu"})
    df = df.drop(columns=["SNR_MED"])
    say("DAPall HYB10 rows read: %d (expect 10782)" % n)
    assert n == 10782, "DAPall row count changed: %d" % n
    return df


def load_pymorph():
    p = os.path.join(MANGA, "manga-pymorph-DR17.fits")
    bands = {1: "g", 2: "r", 3: "i"}     # verified against the SDSS data model
    frames = []
    with fits.open(p) as h:
        for hdu_i, b in bands.items():
            d = h[hdu_i].data
            cols = ["PLATEIFU", "FLAG_FIT", "FLAG_FAILED_S", "FLAG_FAILED_SE",
                    "M_S", "A_HL_S", "N_S", "BA_S", "PA_S",
                    "M_SE_DISK", "A_HL_SE_DISK", "N_SE_DISK", "BA_SE_DISK",
                    "PA_SE_DISK", "M_SE_BULGE", "A_HL_SE_BULGE", "BT_SE",
                    "DUPL_N", "DUPL_ID"]
            out = {}
            for c in cols:
                col = d[c]
                out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
            f = pd.DataFrame(out).rename(columns={"PLATEIFU": "plateifu"})
            f = f.rename(columns={c: "pym_%s_%s" % (b, c) for c in f.columns
                                  if c != "plateifu"})
            if b != "r":
                f = f[["plateifu", "pym_%s_M_S" % b, "pym_%s_M_SE_DISK" % b]]
            frames.append(f)
            n = len(d)
    df = frames[0]
    for f in frames[1:]:
        df = df.merge(f, on="plateifu", how="outer")
    say("PyMorph rows read: %d per band (expect 10293); merged %d" % (n, len(df)))
    assert n == 10293
    # PyMorph uses -999 as its null
    for c in df.columns:
        if c == "plateifu":
            continue
        df[c] = pd.to_numeric(df[c], errors="coerce").replace(-999.0, np.nan)
    return df


def load_dlmorph():
    p = os.path.join(MANGA, "manga-morphology-dl-DR17.fits")
    with fits.open(p) as h:
        d = h[1].data
        cols = ["PLATEIFU", "T-Type", "P_LTG", "P_S0", "P_edge", "P_bar",
                "Visual_Class", "Visual_Flag"]
        out = {}
        for c in cols:
            col = d[c]
            out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
        n = len(d)
    df = pd.DataFrame(out).rename(columns={"PLATEIFU": "plateifu", "T-Type": "dl_TType",
                                           "P_LTG": "dl_P_LTG", "P_S0": "dl_P_S0",
                                           "P_edge": "dl_P_edge", "P_bar": "dl_P_bar",
                                           "Visual_Class": "dl_VisualClass",
                                           "Visual_Flag": "dl_VisualFlag"})
    say("Deep-learning morphology rows: %d (expect 10293)" % n)
    assert n == 10293
    return df


def load_vmorph():
    p = os.path.join(MANGA, "manga_visual_morpho-2.0.1.fits")
    with fits.open(p) as h:
        d = h[1].data
        cols = ["plateifu", "Type", "TType", "Unsure", "Bars", "Edge_on", "Tidal"]
        out = {}
        for c in cols:
            col = d[c]
            out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
        n = len(d)
    df = pd.DataFrame(out).rename(columns={"Type": "vm_Type", "TType": "vm_TType",
                                           "Unsure": "vm_Unsure", "Bars": "vm_Bars",
                                           "Edge_on": "vm_EdgeOn", "Tidal": "vm_Tidal"})
    say("Visual morphology rows: %d (expect 10126)" % n)
    assert n == 10126
    return df


def load_himanga():
    p = os.path.join(MANGA, "mangaHIall.fits")
    with fits.open(p) as h:
        d = h[1].data
        cols = ["PLATEIFU", "LOGMHI", "LOGHILIM200KMS", "SNR", "FHI", "EFHI",
                "WP50", "WF50", "VHI", "SESSION", "conflag", "conf_prob"]
        out = {}
        for c in cols:
            col = d[c]
            out[c] = strip_bytes(col) if col.dtype.kind in "SU" else native(col)
        n = len(d)
    df = pd.DataFrame(out).rename(columns={"PLATEIFU": "plateifu"})
    df = df.rename(columns={c: "hi_" + c for c in df.columns if c != "plateifu"})
    say("HI-MaNGA rows: %d (expect 6632)" % n)
    assert n == 6632
    # HI-MaNGA carries one row per OBSERVING SESSION, so a galaxy can appear
    # several times.  Keep the highest-S/N row per galaxy; among non-detections
    # that is also the deepest one.
    n0 = len(df)
    df["_snr"] = pd.to_numeric(df["hi_SNR"], errors="coerce").fillna(-99.0)
    df = df.sort_values(["plateifu", "_snr"], ascending=[True, False])
    df = df.drop_duplicates("plateifu", keep="first").drop(columns=["_snr"])
    say("  HI-MaNGA deduplicated by plateifu (one row per session): %d -> %d"
        % (n0, len(df)))
    return df


def load_gema():
    p = os.path.join(MANGA, "GEMA_2.0.2.fits")
    with fits.open(p) as h:
        def tab(name, cols, pref):
            for x in h[1:]:
                if x.header.get("EXTNAME") == name:
                    d = x.data
                    out = {"mangaid": strip_bytes(d["mangaid"])}
                    for c in cols:
                        out[pref + c] = native(d[c])
                    return pd.DataFrame(out), len(d)
            raise AssertionError("GEMA extension %s missing" % name)
        g1, n1 = tab("DR17_param_groups",
                     ["MG", "GroupSize", "nneigh", "Q_nn", "dnn", "Q_group"], "gema_grp_")
        g2, n2 = tab("DR17_param_LSS", ["mh", "den1", "t1", "t2", "t3"], "gema_lss_")
        g3, n3 = tab("DR17_param_overdensity", ["overdensity", "local_density"], "gema_")
    say("GEMA groups=%d LSS=%d overdensity=%d (expect 9670/10086/3287)" % (n1, n2, n3))
    assert (n1, n2, n3) == (9670, 10086, 3287)
    df = g1.merge(g2, on="mangaid", how="outer").merge(g3, on="mangaid", how="outer")
    return df


# ------------------------------------------------------------- environment
def rescale_tempel(v):
    """Tempel tabulates comoving distances/radii in Mpc (their h=1 convention,
    i.e. Mpc/h with h=1).  Converting to our H0=70 cosmology multiplies lengths
    by 1/0.7.  Recorded explicitly so it is auditable rather than implicit."""
    return v / 0.7


def crossmatch_tempel(gal):
    """Attach Tempel+2014 (sigma_v) and Tempel+2017 (R200) group membership."""
    t14g = read_vizier_tsv(os.path.join(GROUPS, "tempel2014_galaxies.tsv"),
                           expect_cols=["GalID", "GroupID", "Ngal", "RAJ2000",
                                        "DEJ2000", "zobs"], expect_min_rows=588000)
    t14gr = read_vizier_tsv(os.path.join(GROUPS, "tempel2014_groups.tsv"),
                            expect_cols=["GroupID", "Ngal", "sig.v", "Rvir", "Rmax",
                                         "MNFW", "Lrgroup"], expect_min_rows=82000)
    t17g = read_vizier_tsv(os.path.join(GROUPS, "tempel2017_table1_galaxies.tsv"),
                           expect_cols=["GalID", "GroupID", "RAJ2000", "DEJ2000",
                                        "zobs"], expect_min_rows=584000)
    t17gr = read_vizier_tsv(os.path.join(GROUPS, "tempel2017_table2_groups.tsv"),
                            expect_cols=["GroupID", "Ngal", "M200", "R200"],
                            expect_min_rows=88000)
    say("Tempel2014 galaxies=%d groups=%d | Tempel2017 galaxies=%d groups=%d"
        % (len(t14g), len(t14gr), len(t17g), len(t17gr)))
    assert len(t14g) == 588193, "Tempel2014 galaxy count %d != 588193 stated in paper" % len(t14g)
    assert len(t17g) == 584449, "Tempel2017 galaxy count %d != 584449 stated in paper" % len(t17g)

    mc = SkyCoord(gal["objra"].to_numpy() * u.deg, gal["objdec"].to_numpy() * u.deg)

    for tag, tg, tgr in (("t14", t14g, t14gr), ("t17", t17g, t17gr)):
        tc = SkyCoord(num(tg, "RAJ2000").to_numpy() * u.deg,
                      num(tg, "DEJ2000").to_numpy() * u.deg)
        idx, d2d, _ = mc.match_to_catalog_sky(tc)
        sep = d2d.arcsec
        dz = np.abs(gal["z"].to_numpy() - num(tg, "zobs").to_numpy()[idx])
        good = (sep < 3.0) & (dz < 0.002)
        say("  %s: matched %d / %d MaNGA galaxies within 3\" and |dz|<0.002"
            % (tag, int(good.sum()), len(gal)))

        gid = np.where(good, num(tg, "GroupID").to_numpy()[idx], np.nan)
        gal[tag + "_GroupID"] = gid
        gal[tag + "_sep_arcsec"] = np.where(good, sep, np.nan)
        gal[tag + "_Ngal"] = np.where(good, num(tg, "Ngal").to_numpy()[idx], np.nan)
        if "Rank" in tg.columns:
            gal[tag + "_Rank"] = np.where(good, num(tg, "Rank").to_numpy()[idx], np.nan)

        # group properties
        grp = tgr.copy()
        grp["GroupID"] = num(grp, "GroupID")
        gmap = grp.set_index("GroupID")
        have = np.isfinite(gid) & (gid > 0)
        sel = pd.Series(gid).map(lambda x: x if (np.isfinite(x) and x > 0) else np.nan)
        j = gmap.reindex(sel.to_numpy())

        gal[tag + "_grp_RA"] = num(j.reset_index(drop=True), "RAJ2000").to_numpy()
        gal[tag + "_grp_DE"] = num(j.reset_index(drop=True), "DEJ2000").to_numpy()
        gal[tag + "_grp_Dc_Mpc"] = rescale_tempel(
            num(j.reset_index(drop=True), "Dist.c").to_numpy())
        gal[tag + "_grp_Rmax_Mpc"] = rescale_tempel(
            num(j.reset_index(drop=True), "Rmax").to_numpy())
        if tag == "t14":
            gal["t14_grp_sigma_v"] = num(j.reset_index(drop=True), "sig.v").to_numpy()
            gal["t14_grp_Rvir_Mpc"] = rescale_tempel(
                num(j.reset_index(drop=True), "Rvir").to_numpy())
            gal["t14_grp_MNFW_rank_only"] = num(j.reset_index(drop=True), "MNFW").to_numpy()
            gal["t14_grp_Lr_1e10Lsun"] = num(j.reset_index(drop=True), "Lrgroup").to_numpy()
            gal["t14_grp_zcmb"] = num(j.reset_index(drop=True), "zcmb").to_numpy()
        else:
            gal["t17_grp_R200_Mpc"] = rescale_tempel(
                num(j.reset_index(drop=True), "R200").to_numpy())
            gal["t17_grp_M200_rank_only"] = num(j.reset_index(drop=True), "M200").to_numpy()

        # projected clustercentric separation, in physical kpc at the group distance
        gc = SkyCoord(gal[tag + "_grp_RA"].to_numpy() * u.deg,
                      gal[tag + "_grp_DE"].to_numpy() * u.deg)
        theta = mc.separation(gc).radian
        # Tempel's Dist.c is comoving; convert to angular-diameter for a projected
        # physical separation.  z of the group from its comoving distance is not
        # tabulated in t17, so use the galaxy redshift, which agrees to <1%.
        zg = gal["z"].to_numpy()
        dA = np.asarray(COSMO.angular_diameter_distance(np.clip(zg, 1e-5, None)).to(u.kpc))
        gal[tag + "_Rproj_kpc"] = theta * dA
        gal[tag + "_PA_to_host_deg"] = mc.position_angle(gc).deg
    return gal


def crossmatch_mcxc(gal):
    """Flag hosts that are X-ray detected clusters.  L500 is an OBSERVABLE;
    M500 and R500 in MCXC come from an L-M scaling relation and are rank-only."""
    m = read_vizier_tsv(os.path.join(GROUPS, "mcxc_piffaretti2011.tsv"),
                        expect_cols=["MCXC", "RAJ2000", "DEJ2000", "z", "L500",
                                     "M500", "R500"], expect_min_rows=1700)
    say("MCXC clusters: %d (expect 1743)" % len(m))
    assert len(m) == 1743, "MCXC row count %d != 1743" % len(m)
    mc_c = SkyCoord(m["RAJ2000"].to_numpy(), m["DEJ2000"].to_numpy(),
                    unit=(u.hourangle, u.deg))
    # match on the HOST GROUP centre, not the galaxy
    for tag in ("t14", "t17"):
        ra = gal[tag + "_grp_RA"].to_numpy()
        de = gal[tag + "_grp_DE"].to_numpy()
        ok = np.isfinite(ra) & np.isfinite(de)
        idx = np.full(len(gal), -1)
        sep = np.full(len(gal), np.nan)
        if ok.sum():
            gc = SkyCoord(ra[ok] * u.deg, de[ok] * u.deg)
            i, d2d, _ = gc.match_to_catalog_sky(mc_c)
            idx[ok] = i
            sep[ok] = d2d.arcmin
        zc = num(m, "z").to_numpy()
        dz = np.abs(gal["z"].to_numpy() - np.where(idx >= 0, zc[idx], np.nan))
        hit = (sep < 10.0) & (dz < 0.01)      # 10 arcmin, dz 0.01
        gal[tag + "_mcxc_name"] = np.where(hit, m["MCXC"].to_numpy()[idx], "")
        gal[tag + "_mcxc_sep_arcmin"] = np.where(hit, sep, np.nan)
        gal[tag + "_mcxc_L500_1e44"] = np.where(hit, num(m, "L500").to_numpy()[idx], np.nan)
        gal[tag + "_mcxc_M500_rank_only"] = np.where(hit, num(m, "M500").to_numpy()[idx], np.nan)
        gal[tag + "_mcxc_R500_rank_only"] = np.where(hit, num(m, "R500").to_numpy()[idx], np.nan)
        say("  %s host groups matched to an MCXC X-ray cluster: %d galaxies"
            % (tag, int(hit.sum())))

    # GALAXY-CENTRED X-ray flag.  The group-centre match above has false
    # negatives on exactly the richest systems: Coma's Tempel luminosity centre
    # sits 12.4 arcmin from the X-ray peak, outside the 10 arcmin window, so
    # Coma -- 306 MaNGA galaxies -- was silently missed.  Flag instead on the
    # galaxy itself lying inside a projected 2 Mpc of an X-ray peak at matching
    # redshift, which is the physically meaningful statement "this galaxy sits
    # in an X-ray emitting intracluster medium".
    gc = SkyCoord(gal["objra"].to_numpy() * u.deg, gal["objdec"].to_numpy() * u.deg)
    idx, d2d, _ = gc.match_to_catalog_sky(mc_c)
    zc = num(m, "z").to_numpy()[idx]
    dA = np.asarray(COSMO.angular_diameter_distance(
        np.clip(gal["z"].to_numpy(), 1e-5, None)).to(u.Mpc))
    rproj = d2d.radian * dA
    hit = (np.abs(gal["z"].to_numpy() - zc) < 0.01) & (rproj < 2.0)
    gal["xray_name"] = np.where(hit, m["MCXC"].to_numpy()[idx], "")
    gal["xray_oname"] = np.where(hit, m["OName"].to_numpy()[idx], "")
    gal["xray_Rproj_Mpc"] = np.where(hit, rproj, np.nan)
    gal["xray_dz"] = np.where(hit, gal["z"].to_numpy() - zc, np.nan)
    gal["xray_L500_1e44"] = np.where(hit, num(m, "L500").to_numpy()[idx], np.nan)
    gal["xray_M500_rank_only"] = np.where(hit, num(m, "M500").to_numpy()[idx], np.nan)
    gal["xray_R500_rank_only"] = np.where(hit, num(m, "R500").to_numpy()[idx], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        gal["xray_R_over_R500_rank_only"] = (
            gal["xray_Rproj_Mpc"] / gal["xray_R500_rank_only"])
    say("  galaxy-centred X-ray flag (within 2 Mpc projected of an MCXC peak, "
        "|dz|<0.01): %d galaxies across %d distinct clusters"
        % (int(hit.sum()), len(set(m["MCXC"].to_numpy()[idx][hit]))))
    return gal


# ----------------------------------------------------------------- derive
def freeman_disk_gbar(M_disk, R_d, R):
    """Circular acceleration of a razor-thin exponential disk (Freeman 1970).
    M_disk in Msun, R_d and R in kpc -> g in (km/s)^2 / kpc."""
    from scipy.special import iv, kv
    y = R / (2.0 * R_d)
    Sig0 = M_disk / (2.0 * np.pi * R_d ** 2)
    with np.errstate(over="ignore", invalid="ignore"):
        br = iv(0, y) * kv(0, y) - iv(1, y) * kv(1, y)
    v2 = 4.0 * np.pi * G_MSUN_KPC_KMS * Sig0 * R_d * y ** 2 * br
    return v2 / R


def hernquist_gbar(M_b, R_half, R):
    """Spherical Hernquist bulge, a = R_half / (1 + sqrt(2))."""
    a = R_half / (1.0 + np.sqrt(2.0))
    return G_MSUN_KPC_KMS * M_b / (R + a) ** 2


def derive(df):
    z = df["z"].to_numpy()
    df["DL_Mpc"] = np.asarray(COSMO.luminosity_distance(np.clip(z, 1e-5, None)).value)
    df["DA_Mpc"] = np.asarray(COSMO.angular_diameter_distance(np.clip(z, 1e-5, None)).value)
    kpc_per_arcsec = df["DA_Mpc"].to_numpy() * 1e3 * (np.pi / 180.0 / 3600.0)
    df["kpc_per_arcsec"] = kpc_per_arcsec

    # ---- structural: prefer the Ser+Exp DISK when PyMorph says SerExp is OK
    flag = df["pym_r_FLAG_FIT"].to_numpy()
    failed_se = df["pym_r_FLAG_FAILED_SE"].to_numpy()
    use_se = np.isin(flag, [0, 2]) & (failed_se == 0) & \
        np.isfinite(df["pym_r_A_HL_SE_DISK"].to_numpy()) & \
        (df["pym_r_A_HL_SE_DISK"].to_numpy() > 0)
    df["struct_source"] = np.where(use_se, "pymorph_SerExp_disk", "pymorph_Sersic")

    a_hl = np.where(use_se, df["pym_r_A_HL_SE_DISK"], df["pym_r_A_HL_S"])
    nser = np.where(use_se, df["pym_r_N_SE_DISK"], df["pym_r_N_S"])
    ba = np.where(use_se, df["pym_r_BA_SE_DISK"], df["pym_r_BA_S"])
    pa = np.where(use_se, df["pym_r_PA_SE_DISK"], df["pym_r_PA_S"])

    df["Rhl_disk_arcsec"] = a_hl
    df["Rhl_disk_kpc"] = a_hl * kpc_per_arcsec
    # exponential disk: R_d = R_half / 1.678
    df["Rd_kpc"] = df["Rhl_disk_kpc"] / 1.678
    df["nser_disk"] = nser
    df["ba_disk"] = ba
    df["pa_disk_deg"] = pa

    # inclination from the axis ratio, with an intrinsic thickness q0=0.20
    q0 = 0.20
    q = np.clip(ba, q0 + 1e-4, 1.0)
    cosi2 = (q ** 2 - q0 ** 2) / (1.0 - q0 ** 2)
    df["incl_deg"] = np.degrees(np.arccos(np.sqrt(np.clip(cosi2, 0.0, 1.0))))

    # ---- masses
    # NSA CAVEAT.  The SDSS data model states that NSA_ELPETRO_MASS and
    # NSA_ELPETRO_ABSMAG are computed with (Om=0.3, OL=0.7, h=1), i.e. masses in
    # h^-2 Msun and absolute magnitudes on the h=1 distance scale.  This lane
    # works at H0=70, so both must be rescaled or every baryonic mass is 0.31 dex
    # too low.  This was caught empirically: the 103 galaxies observed by both
    # MaNGA and SAMI showed a -0.308 dex offset in log M_star, and
    # 2 log10(h) = 2 log10(0.7) = -0.3098.  See code/crosscal_manga_sami.py.
    H_NSA = 0.7                       # the h of THIS lane's cosmology
    DLOG_M_H = -2.0 * np.log10(H_NSA)          # +0.3098 dex
    DMAG_H = 5.0 * np.log10(H_NSA)             # -0.7745 mag
    df["logMstar_nsa_h1_raw"] = np.log10(np.where(df["nsa_elpetro_mass"] > 0,
                                                  df["nsa_elpetro_mass"], np.nan))
    df["logMstar_nsa"] = df["logMstar_nsa_h1_raw"] + DLOG_M_H
    df["nsa_h_rescale_dex"] = DLOG_M_H
    # independent colour-based stellar mass, Taylor+2011 eq.8 (Chabrier IMF):
    #   log M*/Msun = 1.15 + 0.70 (g-i) - 0.4 M_i
    # colour is h-independent; the absolute magnitude is not.
    gi = df["nsa_absmag_g"].to_numpy() - df["nsa_absmag_i"].to_numpy()
    Mi = df["nsa_absmag_i"].to_numpy() + DMAG_H
    df["gi_rest"] = gi
    df["logMstar_taylor"] = 1.15 + 0.70 * gi - 0.4 * Mi

    logMHI = df["hi_LOGMHI"].to_numpy()
    snr = df["hi_SNR"].to_numpy()
    det = np.isfinite(logMHI) & (logMHI > 0) & np.isfinite(snr) & (snr >= 5.0)
    df["hi_detected"] = det
    lim = df["hi_LOGHILIM200KMS"].to_numpy()
    df["logMHI_use"] = np.where(det, logMHI, np.nan)
    df["logMHI_limit"] = np.where(~det & np.isfinite(lim) & (lim > 0), lim, np.nan)

    Mstar = 10.0 ** df["logMstar_nsa"].to_numpy()
    MHI = 10.0 ** df["logMHI_use"].to_numpy()
    Mgas = 1.33 * MHI                      # helium correction; no H2 available
    df["Mgas_Msun"] = Mgas
    df["Mb_Msun"] = Mstar + np.where(np.isfinite(Mgas), Mgas, 0.0)
    df["logMb"] = np.log10(df["Mb_Msun"])
    df["f_gas"] = np.where(np.isfinite(Mgas), Mgas / df["Mb_Msun"], np.nan)

    # ---- surface density and a fiducial baryonic acceleration
    Rd = df["Rd_kpc"].to_numpy()
    df["Sigma_b_Msun_pc2"] = df["Mb_Msun"].to_numpy() / (2.0 * np.pi * (Rd * 1e3) ** 2)
    bt = np.clip(df["pym_r_BT_SE"].to_numpy(), 0.0, 1.0)
    bt = np.where(use_se & np.isfinite(bt), bt, 0.0)
    Mdisk = df["Mb_Msun"].to_numpy() * (1.0 - bt)
    Mbulge = df["Mb_Msun"].to_numpy() * bt
    Rb = df["pym_r_A_HL_SE_BULGE"].to_numpy() * kpc_per_arcsec
    Rb = np.where(np.isfinite(Rb) & (Rb > 0), Rb, 0.3 * Rd)
    for label, R in (("2p2Rd", 2.2 * Rd), ("1Rd", 1.0 * Rd), ("4Rd", 4.0 * Rd)):
        with np.errstate(invalid="ignore", divide="ignore"):
            g = freeman_disk_gbar(Mdisk, Rd, R)
            g = g + np.where(bt > 0, hernquist_gbar(Mbulge, Rb, R), 0.0)
        # (km/s)^2/kpc -> m/s^2 : 1 (km/s)^2/kpc = 1e6 / 3.0857e19 m/s^2
        df["gbar_%s_ms2" % label] = g * 1e6 / 3.0856775814913673e19
        df["log_gbar_%s" % label] = np.log10(df["gbar_%s_ms2" % label])

    # ---- kinematic amplitude proxies straight from the DAP summary
    for tag, lo, hi in (("stel", "STELLAR_VEL_LO_CLIP", "STELLAR_VEL_HI_CLIP"),
                        ("ha", "HA_GVEL_LO_CLIP", "HA_GVEL_HI_CLIP")):
        amp = 0.5 * (df[hi].to_numpy() - df[lo].to_numpy())
        df["vamp_%s_kms" % tag] = amp
        with np.errstate(invalid="ignore", divide="ignore"):
            df["vamp_%s_deproj_kms" % tag] = amp / np.sin(np.radians(df["incl_deg"]))

    # ---- geometry: angle between disk normal and direction to host centre
    for tag in ("t14", "t17"):
        pah = df[tag + "_PA_to_host_deg"].to_numpy()
        dphi = np.radians(pah - df["pa_disk_deg"].to_numpy())
        inc = np.radians(df["incl_deg"].to_numpy())
        # sky-plane angle between the major axis and the host direction, [0,90]
        th = np.degrees(np.arccos(np.clip(np.abs(np.cos(dphi)), 0, 1)))
        df[tag + "_theta_sky_deg"] = th
        # angle between the disk normal and the host direction, assuming the
        # host offset is purely transverse; folded to [0,90] because the near
        # side of the disk is unknown.
        cpsi = np.abs(np.sin(inc) * np.sin(dphi))
        df[tag + "_psi_norm_host_deg"] = np.degrees(np.arccos(np.clip(cpsi, 0, 1)))

    # ---- environment scalars.  sigma_v is OBSERVED; everything derived from it
    # under a virial assumption is flagged.
    sv = df["t14_grp_sigma_v"].to_numpy()
    df["Phi_proxy_sigma2"] = sv ** 2                       # (km/s)^2, observable
    df["log_Phi_proxy"] = np.log10(np.where(sv > 0, sv ** 2, np.nan))
    rv = df["t14_grp_Rvir_Mpc"].to_numpy() * 1e3
    df["R_over_Rvir_t14"] = df["t14_Rproj_kpc"].to_numpy() / rv
    r200 = df["t17_grp_R200_Mpc"].to_numpy() * 1e3
    df["R_over_R200_t17"] = df["t17_Rproj_kpc"].to_numpy() / r200
    # external field strength: |g_ext| ~ sigma_v^2 / R_proj  (order of magnitude,
    # observable up to the virial-tracer assumption; NOT a mass)
    with np.errstate(invalid="ignore", divide="ignore"):
        gext = sv ** 2 / df["t14_Rproj_kpc"].to_numpy()
        df["gext_proxy_ms2"] = gext * 1e6 / 3.0856775814913673e19
        df["log_gext_proxy"] = np.log10(df["gext_proxy_ms2"])
    return df


# ------------------------------------------------------------------- main
def main():
    os.makedirs(CLEAN, exist_ok=True)
    drp = load_drpall()

    # sample cuts, declared here BEFORE any residual or kinematic selection
    ok = (drp["srvymode"] == "MaNGA dither") & (drp["z"] > 0.001) & (drp["z"] < 0.2)
    # DRP3QUAL CRITICAL bit is 2**30
    ok &= (drp["drp3qual"].to_numpy().astype(np.int64) & (1 << 30)) == 0
    ok &= drp["mangaid"] != "0"
    say("after DRP quality/mode/z cuts: %d rows" % int(ok.sum()))
    drp = drp[ok].copy()

    # deduplicate repeat observations of the same galaxy: keep the largest IFU,
    # then the highest total S/N^2
    drp["_sn"] = drp["bluesn2"] + drp["redsn2"]
    drp = drp.sort_values(["mangaid", "ifudesignsize", "_sn"],
                          ascending=[True, False, False])
    n_before = len(drp)
    drp = drp.drop_duplicates("mangaid", keep="first").drop(columns=["_sn"])
    say("deduplicated by mangaid: %d -> %d unique galaxies" % (n_before, len(drp)))

    df = drp
    for loader, key in ((load_dapall, "plateifu"), (load_pymorph, "plateifu"),
                        (load_dlmorph, "plateifu"), (load_vmorph, "plateifu"),
                        (load_himanga, "plateifu"), (load_gema, "mangaid")):
        t = loader()
        assert not t[key].duplicated().any(), \
            "%s has duplicate %s keys - would multiply rows" % (loader.__name__, key)
        n0 = len(df)
        df = df.merge(t, on=key, how="left")
        assert len(df) == n0, "join on %s changed row count %d -> %d" % (key, n0, len(df))
    say("after all MaNGA-side joins: %d rows, %d columns" % (df.shape[0], df.shape[1]))

    df = crossmatch_tempel(df)
    df = crossmatch_mcxc(df)
    df = derive(df)

    out = os.path.join(CLEAN, "manga_env_master.csv")
    df.to_csv(out, index=False)
    say("WROTE %s : %d rows x %d cols" % (out, df.shape[0], df.shape[1]))

    # ---- manifest
    inputs = {}
    for p in [os.path.join(MANGA, f) for f in os.listdir(MANGA) if f.endswith(".fits")] + \
             [os.path.join(GROUPS, f) for f in os.listdir(GROUPS) if f.endswith(".tsv")]:
        inputs[os.path.basename(p)] = {"sha256": sha256(p), "bytes": os.path.getsize(p)}
    man = {
        "file": "manga_env_master.csv",
        "produced_by": "env-data/code/build_manga_env.py",
        "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256(out),
        "bytes": os.path.getsize(out),
        "row_count": int(df.shape[0]),
        "column_count": int(df.shape[1]),
        "columns": list(df.columns),
        "inputs": inputs,
        "cosmology": "FlatLambdaCDM H0=70 km/s/Mpc, Om0=0.3; Tempel lengths divided by 0.7",
        "dark_matter_dependent_columns": [c for c in df.columns if "rank_only" in c],
        "build_log": LOG,
    }
    with open(out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    say("WROTE manifest")


if __name__ == "__main__":
    main()
