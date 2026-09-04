"""TEST 2(e) -- measure the gas-versus-stellar kinematic misalignment directly.

A galaxy whose ionised gas rotates in a plane strongly misaligned from its
stellar disk is the IFU analogue of a polar-ring galaxy: one baryonic system
supplying rotation tracers in two different planes.  At 90 degrees it IS a
polar-gas system.  This is Test 2(e), and unlike the literature-catalogue route
it yields a MEASURED angle per galaxy rather than a citation.

Method.  For each of the DAP MAPS already held by this lane, fit a plane
V = a + b*x + c*y to the unmasked velocity field inside 1.5 R_eff, separately
for STELLAR_VEL and for EMLINE_GVEL (H-alpha, channel 23).  The kinematic
position angle is atan2(b, c) measured from the +y (north) axis toward +x.  The
misalignment is the difference folded onto [0, 180] deg, where 0 means co-rotating,
180 means counter-rotating and 90 means orthogonal (polar) gas.

This linear-gradient estimator is deliberately crude: it recovers the global
rotation axis and is robust to noise, but it cannot see a kinematically
decoupled core or a warp, and it degrades for a disturbed field.  It is used
here as a SCREEN.  Anything it flags needs a proper kinematic-PA fit
(Krajnovic et al. 2006) on the same cube before being called a detection.
"""
import hashlib
import json
import os
import warnings
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from astropy.io import fits

warnings.filterwarnings("ignore")

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")
CLEAN = os.path.join(LANE, "clean")
MAPS = os.path.join(LANE, "raw", "manga", "maps")
DAPTYPE = "HYB10-MILESHC-MASTARSSP"
HA_CHANNEL = 23          # EMLINE_GVEL channel for Ha-6564 in the DR17 DAP
RMAX_RE = 1.5
MIN_SPX = 30


def plane_pa(V, ivar, mask, rr, weight=None):
    """Fit V = a + b x + c y inside RMAX_RE and return (PA_deg, amplitude, n)."""
    ny, nx = V.shape
    y, x = np.mgrid[0:ny, 0:nx].astype(float)
    x -= (nx - 1) / 2.0
    y -= (ny - 1) / 2.0
    g = (mask == 0) & (ivar > 0) & np.isfinite(V) & (rr <= RMAX_RE)
    if g.sum() < MIN_SPX:
        return np.nan, np.nan, int(g.sum())
    w = ivar[g] if weight is None else weight[g]
    A = np.vstack([np.ones(g.sum()), x[g], y[g]]).T
    W = w / w.sum()
    ATA = (A * W[:, None]).T @ A
    ATy = (A * W[:, None]).T @ V[g]
    try:
        coef = np.linalg.solve(ATA, ATy)
    except np.linalg.LinAlgError:
        return np.nan, np.nan, int(g.sum())
    b, c = coef[1], coef[2]
    pa = np.degrees(np.arctan2(b, c)) % 360.0
    amp = float(np.hypot(b, c))
    return pa, amp, int(g.sum())


def one(plateifu):
    p = os.path.join(MAPS, "manga-%s-MAPS-%s.fits.gz" % (plateifu, DAPTYPE))
    if not os.path.exists(p):
        return None
    with fits.open(p) as h:
        sv = h["STELLAR_VEL"].data.astype(float)
        svi = h["STELLAR_VEL_IVAR"].data.astype(float)
        svm = h["STELLAR_VEL_MASK"].data.astype(int)
        gv = h["EMLINE_GVEL"].data[HA_CHANNEL].astype(float)
        gvi = h["EMLINE_GVEL_IVAR"].data[HA_CHANNEL].astype(float)
        gvm = h["EMLINE_GVEL_MASK"].data[HA_CHANNEL].astype(int)
        gf = h["EMLINE_GFLUX"].data[HA_CHANNEL].astype(float)
        rr = h["SPX_ELLCOO"].data[1].astype(float)
        chan = h["EMLINE_GVEL"].header.get("C%02d" % (HA_CHANNEL + 1), "")
    pa_s, amp_s, n_s = plane_pa(sv, svi, svm, rr)
    pa_g, amp_g, n_g = plane_pa(gv, gvi, gvm, rr, weight=np.where(gf > 0, gf, 0))
    if not (np.isfinite(pa_s) and np.isfinite(pa_g)):
        return None
    d = abs(pa_s - pa_g) % 360.0
    if d > 180.0:
        d = 360.0 - d
    return dict(plateifu=plateifu, pa_stellar_deg=pa_s, pa_gas_deg=pa_g,
                misalign_deg=d, grad_stellar=amp_s, grad_gas=amp_g,
                n_spx_stellar=n_s, n_spx_gas=n_g, ha_channel_label=chan)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    files = [f for f in os.listdir(MAPS) if f.endswith(".fits.gz")]
    pifs = sorted(f.replace("manga-", "").replace("-MAPS-%s.fits.gz" % DAPTYPE, "")
                  for f in files)
    rows = [r for r in (one(p) for p in pifs) if r]
    df = pd.DataFrame(rows)

    master = pd.read_csv(os.path.join(CLEAN, "manga_env_master.csv"), low_memory=False)
    keep = ["plateifu", "mangaid", "objra", "objdec", "z", "incl_deg", "pa_disk_deg",
            "logMstar_nsa", "Rd_kpc", "dl_TType", "dl_P_LTG", "SNR_MED_r",
            "t14_grp_sigma_v", "xray_name", "hi_detected", "logMHI_use"]
    df = df.merge(master[keep], on="plateifu", how="left")

    # confirm the H-alpha channel is what we think it is
    lbl = df.ha_channel_label.dropna().unique()
    assert len(lbl) and all("Ha" in str(x) for x in lbl), \
        "EMLINE_GVEL channel %d is not H-alpha: %s" % (HA_CHANNEL, lbl)

    strong = df[(df.grad_stellar > 0.5) & (df.grad_gas > 0.5)]
    print("cubes measured                       : %d of %d" % (len(df), len(pifs)))
    print("H-alpha channel label confirmed      : %s" % lbl[0])
    print("both components rotating (grad>0.5)  : %d" % len(strong))
    for lo, hi, nm in ((0, 30, "aligned      "), (30, 60, "mild         "),
                       (60, 120, "ORTHOGONAL   "), (120, 150, "strong        "),
                       (150, 180, "COUNTER-ROT  ")):
        m = strong.misalign_deg.between(lo, hi)
        print("   %s %3d-%3d deg : %4d  (%.1f%%)"
              % (nm, lo, hi, int(m.sum()), 100 * m.mean()))
    out = os.path.join(CLEAN, "manga_gas_star_misalignment.csv")
    df.to_csv(out, index=False)

    pol = strong[strong.misalign_deg.between(70, 110)].sort_values("grad_gas",
                                                                   ascending=False)
    cr = strong[strong.misalign_deg > 150].sort_values("grad_gas", ascending=False)
    print("\nnear-polar gas candidates (70-110 deg), top 10 by gas gradient:")
    for _, r in pol.head(10).iterrows():
        print("   %-12s misalign %5.1f deg  i=%4.1f  logM*=%.2f  TType=%.1f"
              % (r.plateifu, r.misalign_deg, r.incl_deg, r.logMstar_nsa, r.dl_TType))
    print("\ncounter-rotating candidates (>150 deg), top 10 by gas gradient:")
    for _, r in cr.head(10).iterrows():
        print("   %-12s misalign %5.1f deg  i=%4.1f  logM*=%.2f  TType=%.1f"
              % (r.plateifu, r.misalign_deg, r.incl_deg, r.logMstar_nsa, r.dl_TType))

    man = {
        "file": os.path.basename(out),
        "produced_by": "env-data/code/measure_gas_star_misalignment.py",
        "produced_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": sha256(out), "bytes": os.path.getsize(out),
        "row_count": int(len(df)), "column_count": int(df.shape[1]),
        "columns": [
            {"name": "plateifu", "unit": "MaNGA plate-IFU"},
            {"name": "pa_stellar_deg", "unit": "deg, kinematic PA of the stellar "
                                               "velocity field, from +y toward +x"},
            {"name": "pa_gas_deg", "unit": "deg, same for the H-alpha velocity field"},
            {"name": "misalign_deg", "unit": "deg, |PA_stellar - PA_gas| folded to "
                                             "[0,180]; 0 co-rotating, 90 orthogonal "
                                             "(polar gas), 180 counter-rotating"},
            {"name": "grad_stellar", "unit": "km/s per spaxel, fitted gradient amplitude"},
            {"name": "grad_gas", "unit": "km/s per spaxel"},
            {"name": "n_spx_stellar", "unit": "spaxels used"},
            {"name": "n_spx_gas", "unit": "spaxels used"},
            {"name": "ha_channel_label", "unit": "DAP channel label, asserted to be H-alpha"},
        ],
        "source": "SDSS DR17 MaNGA DAP MAPS, DAPTYPE %s, held in raw/manga/maps/" % DAPTYPE,
        "sample": "Every cube this lane holds: the union of the matched-pair galaxies "
                  "and the near-face-on dispersion sample. This is NOT a "
                  "misalignment-selected sample, so the fractions below are "
                  "incidence rates within those two selections, not within MaNGA.",
        "method": "Weighted least-squares plane fit V = a + b x + c y inside "
                  "1.5 R_eff on the unmasked spaxels; PA = atan2(b, c). Stellar fit "
                  "weighted by the DAP inverse variance, gas fit weighted by H-alpha "
                  "flux. Minimum %d spaxels." % MIN_SPX,
        "caveats": [
            "This is a SCREEN, not a measurement of record. A linear gradient "
            "recovers the global rotation axis only; it cannot see a kinematically "
            "decoupled core, a warp, or a counter-rotating inner disk, and it "
            "degrades on a disturbed field.",
            "Anything flagged here requires a proper kinematic position-angle fit "
            "(Krajnovic et al. 2006) on the same cube before it counts as a "
            "detection.",
            "A near-face-on galaxy has an ill-defined PA in both components, so "
            "small gradients give unstable angles. Filter on grad_stellar and "
            "grad_gas before using misalign_deg.",
        ],
    }
    with open(out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print("\nWROTE %s and manifest" % out)


if __name__ == "__main__":
    main()
