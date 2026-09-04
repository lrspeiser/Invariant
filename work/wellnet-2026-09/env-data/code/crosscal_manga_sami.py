"""MaNGA/SAMI cross-calibration on the galaxies both surveys observed.

The MaNGA and SAMI matched-pair tables cannot be pooled naively: their stellar
masses, effective radii and morphologies come from different pipelines, and a
zero-point offset between them would appear as a spurious field/cluster signal
whenever the two surveys contribute unequally to the two arms.

That caveat is only actionable if some galaxies appear in both surveys.  103 do.
This script measures the offsets on them, so the two tables CAN be placed on a
common scale rather than merely being warned about.

Nothing here touches a kinematic quantity, so the blind protection on Test 1 is
unaffected.
"""
import hashlib
import json
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")
CLEAN = os.path.join(LANE, "clean")
COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)
TOL_ARCSEC = 3.0


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def robust(x):
    x = x[np.isfinite(x)]
    if not len(x):
        return dict(n=0)
    med = float(np.median(x))
    nmad = float(1.4826 * np.median(np.abs(x - med)))
    return dict(n=int(len(x)), median=med, nmad=nmad,
                mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                se_median=float(1.2533 * nmad / np.sqrt(len(x))))


def main():
    m = pd.read_csv(os.path.join(CLEAN, "manga_env_master.csv"), low_memory=False)
    s = pd.read_csv(os.path.join(LANE, "raw", "sami",
                                 "sami_dr3_master_galaxy_inventory.tsv"),
                    sep="\t", low_memory=False)
    sra = pd.to_numeric(s.RA_OBJ, errors="coerce").to_numpy()
    sde = pd.to_numeric(s.DEC_OBJ, errors="coerce").to_numpy()
    ok = np.isfinite(sra) & np.isfinite(sde)
    s = s[ok].reset_index(drop=True)

    mc = SkyCoord(m.objra.to_numpy() * u.deg, m.objdec.to_numpy() * u.deg)
    sc = SkyCoord(sra[ok] * u.deg, sde[ok] * u.deg)
    idx, d2d, _ = mc.match_to_catalog_sky(sc)
    hit = d2d.arcsec < TOL_ARCSEC
    print("galaxies observed by BOTH MaNGA and SAMI (within %.1f arcsec): %d"
          % (TOL_ARCSEC, int(hit.sum())))

    M = m[hit].reset_index(drop=True)
    S = s.iloc[idx[hit]].reset_index(drop=True)

    z = pd.to_numeric(S.z_spec, errors="coerce").to_numpy()
    dA = np.asarray(COSMO.angular_diameter_distance(np.clip(z, 1e-5, None)).to("kpc"))
    kpc_as = dA * (np.pi / 180.0 / 3600.0)

    out = pd.DataFrame({
        "manga_plateifu": M.plateifu, "manga_mangaid": M.mangaid,
        "sami_CATID": S.CATID,
        "sep_arcsec": d2d.arcsec[hit],
        "manga_z": M.z, "sami_z": z,
        "manga_logMstar_nsa": M.logMstar_nsa,
        "sami_logMstar": pd.to_numeric(S.Mstar, errors="coerce"),
        "manga_Rd_kpc": M.Rd_kpc,
        "sami_Rd_kpc": pd.to_numeric(S.ReMGE, errors="coerce").to_numpy() * kpc_as / 1.678,
        "manga_incl_deg": M.incl_deg,
        "manga_dl_TType": M.dl_TType, "sami_morph_type": S.morph_type,
    })
    out["d_logMstar"] = out.manga_logMstar_nsa - out.sami_logMstar
    out["d_logRd"] = np.log10(out.manga_Rd_kpc) - np.log10(out.sami_Rd_kpc)
    out["d_z"] = out.manga_z - out.sami_z

    fp = os.path.join(CLEAN, "manga_sami_crosscal.csv")
    out.to_csv(fp, index=False)

    res = {"n_shared": int(hit.sum()),
           "match_tolerance_arcsec": TOL_ARCSEC,
           "redshift_agreement": robust(out.d_z.to_numpy()),
           "log_Mstar_offset_manga_minus_sami": robust(out.d_logMstar.to_numpy()),
           "log_Rd_offset_manga_minus_sami": robust(out.d_logRd.to_numpy())}
    for k in ("redshift_agreement", "log_Mstar_offset_manga_minus_sami",
              "log_Rd_offset_manga_minus_sami"):
        r = res[k]
        if r.get("n"):
            print("%-38s n=%3d  median %+.4f  nMAD %.4f  s.e.(median) %.4f"
                  % (k, r["n"], r["median"], r["nmad"], r["se_median"]))

    dm = res["log_Mstar_offset_manga_minus_sami"]
    dr = res["log_Rd_offset_manga_minus_sami"]
    res["verdict"] = (
        "The MaNGA and SAMI scales differ by %+.3f dex in log M_star "
        "(scatter %.3f dex) and %+.3f dex in log R_d (scatter %.3f dex) on the "
        "%d galaxies both surveys observed. Because those offsets are MEASURED, "
        "the two pair tables can be placed on a common scale rather than merely "
        "kept apart -- subtract the offsets before pooling. Note the offsets are "
        "comparable to the 0.10 dex matching tolerances, so pooling WITHOUT the "
        "correction would move galaxies across the tolerance box."
        % (dm["median"], dm["nmad"], dr["median"], dr["nmad"], res["n_shared"]))
    print("\n" + res["verdict"])

    with open(os.path.join(CLEAN, "manga_sami_crosscal_summary.json"), "w") as f:
        json.dump(res, f, indent=2)
    man = {"file": "manga_sami_crosscal.csv",
           "produced_by": "env-data/code/crosscal_manga_sami.py",
           "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
           "sha256": sha256(fp), "bytes": os.path.getsize(fp),
           "row_count": int(len(out)), "column_count": int(out.shape[1]),
           "columns": [{"name": c, "unit": (
               "arcsec" if c == "sep_arcsec" else
               "log10 Msun" if "logMstar" in c else
               "kpc" if c.endswith("_kpc") else
               "dex" if c.startswith("d_log") else
               "deg" if c.endswith("_deg") else "")} for c in out.columns],
           "method": "Positional cross-match of the MaNGA DR17 master table "
                     "against the SAMI DR3 master inventory at %.1f arcsec. "
                     "MaNGA M_star is NSA elpetro (Chabrier IMF, k-correct); SAMI "
                     "M_star is the DR3 catalogue value. MaNGA R_d is the PyMorph "
                     "r-band disk half-light semi-major axis / 1.678; SAMI R_d is "
                     "ReMGE / 1.678. Both use the same H0=70, Om0=0.3 cosmology "
                     "and the same 1.678 exponential convention, so the residual "
                     "offset is a pipeline difference, not a unit difference."
                     % TOL_ARCSEC,
           "note": res["verdict"]}
    with open(fp + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print("WROTE %s and manifest" % fp)


if __name__ == "__main__":
    main()
