"""
Build the independently-distanced source table.

Sources
-------
  pantheon+  : Pantheon+SH0ES (Scolnic+2022 / Brout+2022), full 1701 light
               curves, with the STAT+SYS covariance loaded properly.
  cf4        : Cosmicflows-4 group distances, CDS J/ApJ/944/94 table4
               (Tully+2023), 38053 groups.
  megamaser  : Megamaser Cosmology Project geometric distances.
  siren      : GW170817 standard siren.

Every source carries: name, survey, ra, dec, z_helio, z_cmb, D (independent
comoving distance, Mpc/h), sigma_D, and a flag for which distance indicator.
"""
from __future__ import annotations

import gzip
import io
import os

import numpy as np
import pandas as pd

from common import (comoving_distance, luminosity_to_comoving, C_KMS,
                    z_of_comoving)

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
REPO = os.path.abspath(os.path.join(LANE, "..", "..", ".."))
PRIV = os.path.join(REPO, "work", "private")

PANTHEON_DIR = os.path.join(LANE, "raw", "pantheonplus")
CF4_DIR = os.path.join(PRIV, "open-gravity-void-source-v2")


# --------------------------------------------------------------------------
def load_pantheon():
    dat = os.path.join(PANTHEON_DIR, "Pantheon+SH0ES.dat")
    df = pd.read_csv(dat, sep=r"\s+")
    assert len(df) == 1701, f"Pantheon+ row count {len(df)} != 1701"
    assert df.shape[1] == 47, f"Pantheon+ col count {df.shape[1]} != 47"
    return df


def load_pantheon_cov(name="Pantheon+SH0ES_STAT+SYS.cov"):
    path = os.path.join(PANTHEON_DIR, name)
    with open(path) as fh:
        n = int(fh.readline().split()[0])
        vals = np.fromstring(fh.read(), sep=" ")
    assert vals.size == n * n, f"cov has {vals.size} values, expected {n*n}"
    C = vals.reshape(n, n)
    asym = np.abs(C - C.T).max()
    C = 0.5 * (C + C.T)
    return C, n, asym


def pantheon_sources():
    """
    One row per *unique* SN (Pantheon+ has repeat light curves of the same SN
    from different surveys).  We keep the light curve with the smallest
    covariance-diagonal error as the representative, and record the multiplicity.
    """
    df = load_pantheon()
    C, n, asym = load_pantheon_cov()
    sd = np.sqrt(np.diag(C))
    df = df.copy()
    df["_sqrt_diag_cov"] = sd
    df["_row"] = np.arange(n)
    # unique SN by CID
    df = df.sort_values("_sqrt_diag_cov")
    grp = df.groupby("CID", sort=False)
    rep = grp.head(1).copy()
    mult = grp.size().rename("n_lightcurves")
    rep = rep.join(mult, on="CID")

    z_cmb = rep["zCMB"].to_numpy(float)
    z_hd = rep["zHD"].to_numpy(float)
    mu = rep["MU_SH0ES"].to_numpy(float)
    # sigma_mu from the covariance diagonal (correct), NOT the _DIAG column
    sig_mu = rep["_sqrt_diag_cov"].to_numpy(float)
    dl = 10.0 ** (mu / 5.0 - 5.0)                # Mpc, luminosity distance
    # h=1 lengths: MU_SH0ES is on the SH0ES absolute scale, so Mpc -> Mpc/h
    # requires h.  We keep Mpc/h with h = H0_fid/100 handled at use time; here
    # we store the *observed* luminosity distance in Mpc and convert with the
    # declared h in path_integrals.py.
    out = pd.DataFrame({
        "name": rep["CID"].astype(str).to_numpy(),
        "survey": "PantheonPlus",
        "indicator": "SNIa",
        "ra": rep["RA"].to_numpy(float),
        "dec": rep["DEC"].to_numpy(float),
        "z_helio": rep["zHEL"].to_numpy(float),
        "z_cmb": z_cmb,
        "z_hd": z_hd,
        "mu": mu,
        "sigma_mu": sig_mu,
        "sigma_mu_column": rep["MU_SH0ES_ERR_DIAG"].to_numpy(float),
        "dl_mpc": dl,
        "is_calibrator": rep["IS_CALIBRATOR"].to_numpy(int),
        "n_lightcurves": rep["n_lightcurves"].to_numpy(int),
        "pantheon_row": rep["_row"].to_numpy(int),
    })
    return out.sort_values("name").reset_index(drop=True), C, asym


# --------------------------------------------------------------------------
CF4_COLSPECS = [
    (0, 7, "PGC1", int), (8, 14, "DMzp", float), (15, 20, "e_DMzp", float),
    (21, 26, "Dist", float), (27, 32, "Vh", float), (33, 38, "Vls", float),
    (39, 44, "V3k", float), (45, 50, "fV3k", float), (51, 57, "Vpds", float),
    (58, 63, "Vpwf", float), (64, 69, "Vpec", float), (70, 75, "Hi", float),
    (76, 82, "logHi", float), (83, 91, "RAdeg", float), (92, 100, "DEdeg", float),
    (101, 109, "GLON", float), (110, 118, "GLAT", float),
    (119, 127, "SGL", float), (128, 136, "SGB", float),
]


def cf4_sources():
    path = os.path.join(CF4_DIR, "cf4-table4.dat.gz")
    with gzip.open(path, "rt") as fh:
        text = fh.read()
    lines = [ln for ln in text.split("\n") if ln.strip()]
    assert len(lines) == 38053, f"CF4 table4 rows {len(lines)} != 38053"
    rec = {}
    for a, b, nm, tp in CF4_COLSPECS:
        vals = []
        for ln in lines:
            s = ln[a:b].strip()
            vals.append(np.nan if s == "" else float(s))
        rec[nm] = np.array(vals)
    df = pd.DataFrame(rec)
    ok = np.isfinite(df["DMzp"]) & np.isfinite(df["Dist"]) & (df["Dist"] > 0)
    df = df[ok].reset_index(drop=True)
    out = pd.DataFrame({
        "name": ["PGC" + str(int(p)) for p in df["PGC1"]],
        "survey": "Cosmicflows-4",
        "indicator": "CF4group",
        "ra": df["RAdeg"].to_numpy(float),
        "dec": df["DEdeg"].to_numpy(float),
        "z_helio": df["Vh"].to_numpy(float) / C_KMS,
        "z_cmb": df["V3k"].to_numpy(float) / C_KMS,
        "z_hd": df["V3k"].to_numpy(float) / C_KMS,
        "mu": df["DMzp"].to_numpy(float),
        "sigma_mu": df["e_DMzp"].to_numpy(float),
        "sigma_mu_column": df["e_DMzp"].to_numpy(float),
        "dl_mpc": df["Dist"].to_numpy(float),
        "is_calibrator": 0,
        "n_lightcurves": 1,
        "pantheon_row": -1,
    })
    return out


# --------------------------------------------------------------------------
# Megamaser Cosmology Project geometric distances.
# Values transcribed from the published tables; see manifests/hand_entered.json
# for the per-object citation.  These are *angular diameter* distances from the
# maser disc modelling.
MEGAMASER = [
    # name, ra_deg, dec_deg, z_helio, D_Mpc, sigma_D_Mpc, reference
    ("NGC4258",      184.739583,  47.303889, 0.001494,   7.576, 0.234,
     "Reid, Pesce & Riess 2019 ApJ 886 L27"),
    ("UGC3789",      109.836667,  59.354444, 0.010696,  51.5,   4.5,
     "Reid et al. 2013 ApJ 767 154 / Pesce et al. 2020 ApJ 891 L1"),
    ("NGC6264",      254.087083,  27.837778, 0.033763, 144.0,  19.0,
     "Kuo et al. 2013 ApJ 767 155 / Pesce et al. 2020"),
    ("NGC6323",      257.925833,  43.548611, 0.025773, 107.0,  23.0,
     "Kuo et al. 2015 ApJ 800 26 / Pesce et al. 2020"),
    ("NGC5765b",     222.827917,   5.113889, 0.027653, 126.3,  11.6,
     "Gao et al. 2016 ApJ 817 128 / Pesce et al. 2020"),
    ("CGCG074-064",  349.649583,   9.549722, 0.024234,  87.6,   7.9,
     "Pesce et al. 2020 ApJ 891 L1"),
    ("NGC4258_dup",  184.739583,  47.303889, 0.001494,   7.576, 0.234, "duplicate guard"),
]


def megamaser_sources():
    rows = [r for r in MEGAMASER if not r[0].endswith("_dup")]
    d = np.array([r[4] for r in rows], float)
    sd = np.array([r[5] for r in rows], float)
    z = np.array([r[3] for r in rows], float)
    # maser distances are angular-diameter distances; convert to luminosity
    dl = d * (1.0 + z) ** 2
    return pd.DataFrame({
        "name": [r[0] for r in rows],
        "survey": "MegamaserCosmologyProject",
        "indicator": "H2Omaser",
        "ra": [r[1] for r in rows],
        "dec": [r[2] for r in rows],
        "z_helio": z,
        "z_cmb": z,          # replaced by CMB correction in path_integrals
        "z_hd": z,
        "mu": 5.0 * np.log10(dl) + 25.0,
        "sigma_mu": 5.0 / np.log(10) * (sd / d),
        "sigma_mu_column": 5.0 / np.log(10) * (sd / d),
        "dl_mpc": dl,
        "is_calibrator": 0,
        "n_lightcurves": 1,
        "pantheon_row": -1,
    })


# --------------------------------------------------------------------------
SIRENS = [
    # GW170817 / NGC 4993.  Luminosity distance from the LVC standard-siren
    # measurement (Abbott et al. 2017 Nature 551 85), asymmetric error
    # symmetrised for this table.
    ("GW170817", 197.450375, -23.381500, 0.009783, 43.8, 6.9,
     "Abbott et al. 2017 Nature 551 85 (host NGC 4993)"),
]


def siren_sources():
    dl = np.array([r[4] for r in SIRENS], float)
    sd = np.array([r[5] for r in SIRENS], float)
    return pd.DataFrame({
        "name": [r[0] for r in SIRENS],
        "survey": "GW-standard-siren",
        "indicator": "siren",
        "ra": [r[1] for r in SIRENS],
        "dec": [r[2] for r in SIRENS],
        "z_helio": [r[3] for r in SIRENS],
        "z_cmb": [r[3] for r in SIRENS],
        "z_hd": [r[3] for r in SIRENS],
        "mu": 5.0 * np.log10(dl) + 25.0,
        "sigma_mu": 5.0 / np.log(10) * (sd / dl),
        "sigma_mu_column": 5.0 / np.log(10) * (sd / dl),
        "dl_mpc": dl,
        "is_calibrator": 0,
        "n_lightcurves": 1,
        "pantheon_row": -1,
    })


def all_sources():
    pan, cov, asym = pantheon_sources()
    parts = [pan, cf4_sources(), megamaser_sources(), siren_sources()]
    df = pd.concat(parts, ignore_index=True)
    return df, cov, asym


if __name__ == "__main__":
    df, cov, asym = all_sources()
    print(df.groupby("survey").size())
    print("total", len(df))
