"""End-to-end physical sanity check of the derived baryonic quantities.

A units bug of the kind this lane already hit once (the NSA h=1 convention,
worth 0.31 dex) does not raise an exception -- it produces plausible-looking
numbers.  The defence is to check the derived quantities against values that
are known a priori for disk galaxies, and to check that the baryons-only
circular velocity sits BELOW the observed one by about the factor the radial
acceleration relation predicts at the measured g_bar.

This is not a test of gravity.  It compares the lane's own baryonic bookkeeping
against textbook ranges and prints the boost factor; it fits nothing and it
does not touch the field-versus-cluster contrast.
"""
import os

import numpy as np
import pandas as pd

LANE = (r"C:\Users\henry\Documents\Codex\2026-08-21\Invariant-main-integration"
        r"\work\wellnet-2026-09\env-data")
A0 = 1.2e-10
KPC_M = 3.0856775814913673e19

EXPECT = [
    ("logMstar_nsa", "log10 M_star [Msun]", 8.5, 11.5),
    ("Rd_kpc", "R_d [kpc]", 0.5, 12.0),
    ("Sigma_b_Msun_pc2", "Sigma_b [Msun/pc^2]", 20.0, 3000.0),
    ("f_gas", "f_gas", 0.02, 0.95),
]


def main():
    d = pd.read_csv(os.path.join(LANE, "clean", "manga_env_master.csv"),
                    low_memory=False)
    s = d[(d.dl_TType > 0) & (d.dl_P_LTG > 0.5)
          & d.incl_deg.between(25, 75)]
    print("late-type disks, 25 < i < 75 deg: n = %d\n" % len(s))
    bad = []
    for c, lab, lo, hi in EXPECT:
        v = s[c].dropna().to_numpy()
        p = np.percentile(v, [10, 50, 90])
        ok = (p[0] > lo) and (p[2] < hi)
        print("   %-24s p10/50/90 = %9.3f %9.3f %9.3f   %s"
              % (lab, p[0], p[1], p[2], "ok" if ok else "OUT OF RANGE"))
        if not ok:
            bad.append(lab)

    g = s.gbar_2p2Rd_ms2.dropna().to_numpy() / A0
    p = np.percentile(g, [10, 50, 90])
    print("   %-24s p10/50/90 = %9.3f %9.3f %9.3f   %s"
          % ("g_bar(2.2 R_d) / a0", p[0], p[1], p[2],
             "ok" if 0.05 < p[1] < 5 else "OUT OF RANGE"))

    vb = np.sqrt((s.gbar_2p2Rd_ms2 * 2.2 * s.Rd_kpc * KPC_M).dropna()) / 1e3
    vo = s.vamp_ha_deproj_kms
    vo = vo[(vo > 10) & (vo < 500)].dropna()
    print("   %-24s p10/50/90 = %9.1f %9.1f %9.1f   (baryons only)"
          % ("V_bar(2.2 R_d) [km/s]", *np.percentile(vb, [10, 50, 90])))
    print("   %-24s p10/50/90 = %9.1f %9.1f %9.1f   (observed proxy)"
          % ("V_obs [km/s]", *np.percentile(vo, [10, 50, 90])))
    boost = np.median(vo) / np.median(vb)
    # RAR prediction for the boost at the median g_bar
    x = np.median(g) * A0
    gobs = x / (1.0 - np.exp(-np.sqrt(x / A0)))
    print("\n   median V_obs / V_bar          = %.2f" % boost)
    print("   RAR prediction at g_bar/a0=%.2f = %.2f" % (np.median(g),
                                                         np.sqrt(gobs / x)))
    print("\n   Taylor-vs-NSA stellar mass offset (M/L prescription only, "
          "h now consistent): %+.3f dex"
          % np.nanmedian(s.logMstar_taylor - s.logMstar_nsa))
    print("\nVERDICT: %s" % ("all derived quantities inside their expected ranges"
                             if not bad else "OUT OF RANGE: %s" % bad))


if __name__ == "__main__":
    main()
