"""
member-dynamics lane -- resolved STELLAR kinematics from MaNGA DR17 DAP MAPS.

Extracts, for every galaxy appearing in any env-data matched-pair tier, the
aperture second velocity moment inside 1 Re

    sigma_e_tot^2  =  sum_b F_b (V_b^2 + sigma_b^2) / sum_b F_b        (R <= 1 Re)

together with its two parts (sigma_e, (V/sigma)_e), a set of environmental-
contamination diagnostics that require no disk model, and the gas velocity
field measured through the identical aperture so that gas-vs-star disagreement
can be used as a FLAG.

DECLARED BEFORE ANY RESIDUAL WAS INSPECTED.  Nothing in this file reads the
field/cluster label; the label is joined on afterwards in analyse.py.

Definitions (fixed here, not tuned):
  * bins       : unique stellar-continuum Voronoi bins, BINID channel 1, >= 0.
  * good bin   : STELLAR_VEL_MASK == 0, STELLAR_SIGMA_MASK == 0, both IVAR > 0,
                 and STELLAR_SIGMA > STELLAR_SIGMACORR (astrophysical sigma real).
  * sigma_astro: sqrt(SIGMA^2 - SIGMACORR[0]^2).
  * weight F   : BIN_MFLUX (g-band-ish mean flux of the bin), clipped >= 0.
  * V          : STELLAR_VEL minus the flux-weighted mean inside 1 Re
                 (removes any systemic-velocity zero-point error).
  * aperture   : SPX_ELLCOO channel 1 (R/Re) <= 1.0, Re from the DAP header REFF
                 (NSA elliptical Petrosian half-light radius, arcsec).
  * A_kin      : point-reflection asymmetry of the stellar velocity field about
                 the IFU centre, median_b |V(r) + V(-r)| / (2 sigma_e_tot).
                 Model-free: a regular rotator is antisymmetric under r -> -r.
  * PA_kin     : kinematic position angle of the stellar velocity field from a
                 global chi^2 fit of V = A cos(theta - PA) sin-projected, i.e.
                 the standard bisymmetric estimator on a 1-deg grid.

Outputs clean/manga_internal_kin.csv (one row per plateifu).
"""
from __future__ import annotations

import os
import sys
import json
import warnings
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from astropy.io import fits

warnings.filterwarnings("ignore")

ENV = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\env-data"
LANE = r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration\work\wellnet-2026-09\member-dynamics"
MAPS = os.path.join(ENV, "raw", "manga", "maps")

HA_CH = 23  # EMLINE index of Ha-6564 in the DR17 DAP 35-line stack (0-based)


def _kin_pa(x, y, v, w):
    """Global bisymmetric kinematic PA on a 1-degree grid.

    Model  V = A(R) cos(theta - PA); we use a single global amplitude per
    radial ring is overkill, so use the standard cheap estimator: for each
    trial PA, project onto cos(theta-PA) and take the weighted correlation.
    Returns PA in degrees measured from +y (North) towards +x (East),
    matching the MaNGA on-sky convention used for photometric PA.
    """
    if x.size < 8:
        return np.nan
    r = np.hypot(x, y)
    ok = r > 0
    if ok.sum() < 8:
        return np.nan
    x, y, v, w, r = x[ok], y[ok], v[ok], w[ok], r[ok]
    # theta measured from +y towards +x  (i.e. from North towards East)
    th = np.arctan2(x, y)
    pas = np.deg2rad(np.arange(0.0, 360.0, 1.0))
    # weighted least squares amplitude for each trial PA
    c = np.cos(th[None, :] - pas[:, None])
    num = (w * v)[None, :] * c
    den = w[None, :] * c * c
    amp = num.sum(1) / np.maximum(den.sum(1), 1e-30)
    resid = (w[None, :] * (v[None, :] - amp[:, None] * c) ** 2).sum(1)
    i = int(np.argmin(resid))
    pa = np.rad2deg(pas[i])
    if amp[i] < 0:
        pa = (pa + 180.0) % 360.0
    return pa % 360.0


def _aperture_moments(F, V, S, inap):
    w = F[inap]
    v = V[inap]
    s = S[inap]
    W = w.sum()
    if W <= 0:
        return dict(n=0)
    vsys = float((w * v).sum() / W)
    v = v - vsys
    sig2 = float((w * s ** 2).sum() / W)
    vel2 = float((w * v ** 2).sum() / W)
    return dict(n=int(inap.sum()), W=float(W), vsys=vsys, sig2=sig2, vel2=vel2)


def process(plateifu):
    path = os.path.join(MAPS, f"manga-{plateifu}-MAPS-HYB10-MILESHC-MASTARSSP.fits.gz")
    out = {"plateifu": plateifu, "ok": 0, "why": ""}
    if not os.path.exists(path):
        out["why"] = "missing_file"
        return out
    try:
        with fits.open(path, memmap=False) as f:
            h0 = f[0].header
            out["mangaid"] = h0.get("MANGAID")
            out["dapqual"] = int(h0.get("DAPQUAL", -1))
            out["drp3qual"] = int(h0.get("DRP3QUAL", -1))
            out["reff_arcsec"] = float(h0.get("REFF", np.nan))
            out["ecoo_ell"] = float(h0.get("ECOOELL", np.nan))
            out["ecoo_pa"] = float(h0.get("ECOOPA", np.nan))

            ell = f["SPX_ELLCOO"].data          # 0 R", 1 R/Re, 2 R kpc/h, 3 azim
            bell = f["BIN_LWELLCOO"].data       # same, luminosity-weighted per bin
            binid = f["BINID"].data             # 0 binned spec, 1 stellar continua
            sky = f["SPX_SKYCOO"].data          # 0 = +x on sky (arcsec, E), 1 = +y (N)
            bsky = f["BIN_LWSKYCOO"].data
            bflux = f["BIN_MFLUX"].data
            barea = f["BIN_AREA"].data
            bsnr = f["BIN_SNR"].data
            sv = f["STELLAR_VEL"].data
            svm = f["STELLAR_VEL_MASK"].data
            svi = f["STELLAR_VEL_IVAR"].data
            ss = f["STELLAR_SIGMA"].data
            ssm = f["STELLAR_SIGMA_MASK"].data
            ssi = f["STELLAR_SIGMA_IVAR"].data
            sc = f["STELLAR_SIGMACORR"].data[0]
            gv = f["EMLINE_GVEL"].data[HA_CH]
            gvm = f["EMLINE_GVEL_MASK"].data[HA_CH]
            gf = f["EMLINE_GFLUX"].data[HA_CH]
            ganr = f["EMLINE_GANR"].data[HA_CH]

        sbin = binid[1]
        rre = ell[1]
        # ---- unique stellar bins -------------------------------------------
        good = (sbin >= 0) & (svm == 0) & (ssm == 0) & (svi > 0) & (ssi > 0)
        good &= np.isfinite(sv) & np.isfinite(ss) & np.isfinite(sc)
        good &= ss > sc  # astrophysical sigma must be resolved at all
        if good.sum() < 10:
            out["why"] = "too_few_good_spaxels"
            return out
        ids, first = np.unique(sbin[good], return_index=True)
        gy, gx = np.where(good)
        gy, gx = gy[first], gx[first]

        Vb = sv[gy, gx].astype(float)
        Sb = np.sqrt(np.maximum(ss[gy, gx].astype(float) ** 2 - sc[gy, gx].astype(float) ** 2, 0.0))
        # luminosity weight of a Voronoi bin = mean flux x bin area, not mean flux
        Fb = np.maximum(bflux[gy, gx].astype(float), 0.0) * np.maximum(barea[gy, gx].astype(float), 0.0)
        Rb = bell[1][gy, gx].astype(float)
        Xb = bsky[0][gy, gx].astype(float)
        Yb = bsky[1][gy, gx].astype(float)
        SNRb = bsnr[gy, gx].astype(float)
        eV = 1.0 / np.sqrt(svi[gy, gx].astype(float))
        eS = 1.0 / np.sqrt(ssi[gy, gx].astype(float))

        # coverage: how much of the 1 Re ellipse has usable stellar kinematics
        inap_all = rre <= 1.0
        cov = float((good & inap_all).sum()) / max(int(inap_all.sum()), 1)
        out["frac_good_1Re"] = cov
        out["n_spx_1Re"] = int(inap_all.sum())

        # ---- several apertures ---------------------------------------------
        # Rkpc_b: DAP BIN_LWELLCOO channel 2 is R in h^-1 kpc (DAP uses h=1);
        # physical kpc at H0 = 70 is that divided by h = 0.7.
        Rkpc_b = bell[2][gy, gx].astype(float) / 0.7
        apertures = {
            "1Re": Rb <= 1.0,          # PRIMARY, matches the SAMI definition
            "0p5Re": Rb <= 0.5,
            "3kpc": Rkpc_b <= 3.0,     # fixed physical: immune to any Re mismatch
            "5kpc": Rkpc_b <= 5.0,
        }
        inap = apertures["1Re"]
        if inap.sum() < 5:  # tiny IFU / large Re
            out["why"] = "few_bins_in_1Re"
            out["n_bins_1Re"] = int(inap.sum())
            return out

        for name, sel in apertures.items():
            if sel.sum() < 5:
                out[f"sigma_e_tot_{name}"] = np.nan
                out[f"n_bins_{name}"] = int(sel.sum())
                continue
            mm = _aperture_moments(Fb, Vb, Sb, sel)
            out[f"sigma_e_tot_{name}"] = float(np.sqrt(mm["sig2"] + mm["vel2"]))
            out[f"sigma_e_{name}"] = float(np.sqrt(mm["sig2"]))
            out[f"v_e_{name}"] = float(np.sqrt(mm["vel2"]))
            out[f"n_bins_{name}"] = mm["n"]
        out["Rmax_kpc"] = float(np.nanmax(Rkpc_b))
        out["Rmax_Re"] = float(np.nanmax(Rb))

        m = _aperture_moments(Fb, Vb, Sb, inap)
        out["n_bins_1Re"] = m["n"]
        sig2, vel2 = m["sig2"], m["vel2"]
        out["sigma_e"] = float(np.sqrt(sig2))
        out["v_e"] = float(np.sqrt(vel2))
        out["sigma_e_tot"] = float(np.sqrt(sig2 + vel2))
        out["vsigma_e"] = float(np.sqrt(vel2 / sig2)) if sig2 > 0 else np.nan
        out["vsys_offset"] = m["vsys"]
        # local logarithmic slope of the aperture second moment, so the
        # sensitivity to an aperture (Re) mismatch can be priced directly
        if np.isfinite(out.get("sigma_e_tot_0p5Re", np.nan)) and out["sigma_e_tot"] > 0:
            out["dlogS_dlogAp"] = float(
                (np.log10(out["sigma_e_tot"]) - np.log10(out["sigma_e_tot_0p5Re"]))
                / np.log10(2.0))
        else:
            out["dlogS_dlogAp"] = np.nan

        # formal error on sigma_e_tot from the DAP inverse variances
        w = Fb[inap] / Fb[inap].sum() if Fb[inap].sum() > 0 else np.full(inap.sum(), 1.0 / inap.sum())
        vv = Vb[inap] - m["vsys"]
        ss_ = Sb[inap]
        d2 = ((2 * w * vv) ** 2 * eV[inap] ** 2 + (2 * w * ss_) ** 2 * eS[inap] ** 2).sum()
        out["e_sigma_e_tot"] = float(np.sqrt(d2) / (2 * out["sigma_e_tot"])) if out["sigma_e_tot"] > 0 else np.nan
        out["med_sigma_astro"] = float(np.median(ss_))
        out["med_sigmacorr"] = float(np.median(sc[gy, gx][inap]))
        out["frac_sigma_gt50"] = float(np.mean(ss_ > 50.0))
        out["med_snr_1Re"] = float(np.median(SNRb[inap]))

        # ---- model-free point-reflection asymmetry -------------------------
        xa, ya, va, wa = Xb[inap], Yb[inap], vv, w
        # nearest opposite-side partner within 1 arcsec of the reflected point
        d = np.hypot(xa[:, None] + xa[None, :], ya[:, None] + ya[None, :])
        j = np.argmin(d, axis=1)
        dm = d[np.arange(len(j)), j]
        okp = dm < 1.5
        if okp.sum() >= 6:
            asym = np.abs(va[okp] + va[j[okp]]) / (2.0 * out["sigma_e_tot"])
            out["A_kin"] = float(np.median(asym))
            out["n_asym_pairs"] = int(okp.sum())
        else:
            out["A_kin"] = np.nan
            out["n_asym_pairs"] = int(okp.sum())

        # ---- kinematic PA of the stars, and of the gas ---------------------
        out["pa_kin_star"] = _kin_pa(xa, ya, va, wa)

        ggood = (gvm == 0) & np.isfinite(gv) & (ganr > 3.0) & (gf > 0)
        ginap = ggood & inap_all
        out["n_gas_1Re"] = int(ginap.sum())
        if ginap.sum() >= 10:
            gy2, gx2 = np.where(ginap)
            gvv = gv[gy2, gx2].astype(float)
            gww = np.maximum(gf[gy2, gx2].astype(float), 0.0)
            gxx = sky[0][gy2, gx2].astype(float)
            gyy = sky[1][gy2, gx2].astype(float)
            gvv = gvv - (gww * gvv).sum() / max(gww.sum(), 1e-30)
            out["pa_kin_gas"] = _kin_pa(gxx, gyy, gvv, gww)
            out["v_e_gas"] = float(np.sqrt((gww * gvv ** 2).sum() / max(gww.sum(), 1e-30)))
        else:
            out["pa_kin_gas"] = np.nan
            out["v_e_gas"] = np.nan

        out["ok"] = 1
        return out
    except Exception as e:  # noqa: BLE001
        out["why"] = f"exc:{type(e).__name__}:{e}"
        return out


def main():
    mp = pd.read_csv(os.path.join(ENV, "clean", "matched_pairs.csv"))
    gals = sorted(set(mp["cl_plateifu"]) | set(mp["fi_plateifu"]))
    if os.environ.get("ALLMAPS"):
        # every MAPS cube on disk, so the MaNGA/SAMI cross-calibration and the
        # field-arm calibration relation are not limited to the paired galaxies
        gals = sorted({f.split("manga-")[1].split("-MAPS")[0]
                       for f in os.listdir(MAPS) if f.endswith(".fits.gz")})
    print(f"{len(gals)} unique galaxies across all tiers", flush=True)
    rows = []
    nw = int(os.environ.get("NW", "6"))
    if nw <= 1:  # serial fallback: the machine's commit limit is shared
        for i, g in enumerate(gals):
            rows.append(process(g))
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(gals)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=nw) as ex:
            for i, r in enumerate(ex.map(process, gals, chunksize=4)):
                rows.append(r)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1}/{len(gals)}", flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(LANE, "clean"), exist_ok=True)
    out = os.path.join(LANE, "clean", "manga_internal_kin_all.csv" if os.environ.get("ALLMAPS") else "manga_internal_kin.csv")
    df.to_csv(out, index=False)
    print("wrote", out, df.shape)
    print("ok:", int(df["ok"].sum()), "of", len(df))
    print(df.loc[df["ok"] == 0, "why"].value_counts())
    if df["ok"].sum():
        d = df[df["ok"] == 1]
        for c in ["sigma_e", "v_e", "sigma_e_tot", "vsigma_e", "A_kin", "frac_good_1Re", "n_bins_1Re"]:
            print(f"  {c:16s} p10={np.nanpercentile(d[c],10):9.3f} p50={np.nanpercentile(d[c],50):9.3f} p90={np.nanpercentile(d[c],90):9.3f}")


if __name__ == "__main__":
    main()
