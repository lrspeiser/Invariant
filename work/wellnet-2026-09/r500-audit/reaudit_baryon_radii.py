"""Run AY -- re-audit of Run AT's baryon-only radial controls, demanded by AX.5.

Run AT reassured on the R500 tautology partly by showing the radial slope is
unchanged under two "baryon-only" normalisers:

    r / R_b,gas   -0.4875        r / R_b,ne   -0.4841

against physical r at -0.4996.  Run AX.5 then found -- in its OWN control, which
is why it generalises -- that a baryon-only radius is NOT automatically
independent, because `g_bar` already sits in the excess's denominator.  Its worst
normaliser measured -0.99 against the baryon amplitude.

Reading AT's definitions (`ingest.py:baryonic_radii`), the concern is concrete:

    R_b,gas  is the radius where the mean enclosed GAS density hits
             500 rho_c f_b.  It is an AMPLITUDE-dependent radius, and the
             amplitude is M_gas -- exactly what g_bar is built from.
    R_b,ne   is where n_e crosses a fixed threshold.  Amplitude enters only
             through where a profile crosses a constant, so more weakly.

This measures the coupling directly, by perturbing the baryon amplitude and
watching R_b and the excess co-move.  X-COP is already VALIDATION data (spent),
so no seal is touched.

    python reaudit_baryon_radii.py
"""
import io
import json
import os

import numpy as np

import ingest

HERE = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(20260904)
N_DRAW = 400
AMP_SIGMA_DEX = 0.05          # a representative gas-mass normalisation error


def one_cluster(c, amp_dex):
    """Return (log R_b,gas, log R_b,ne, log median excess) at a gas rescaling."""
    cc = dict(c)
    cc["ne_cm3"] = c["ne_cm3"] * 10.0 ** amp_dex      # n_e scales the amplitude
    p = ingest.build_profile(cc)
    Rg, Rn = ingest.baryonic_radii(cc)
    gb, go = p["gb"], p["go"]
    m = np.isfinite(gb) & np.isfinite(go) & (gb > 0) & (go > 0)
    exc = np.median(np.log10(go[m] / gb[m])) if m.sum() else np.nan
    return (np.log10(Rg) if np.isfinite(Rg) else np.nan,
            np.log10(Rn) if np.isfinite(Rn) else np.nan,
            exc)


def main():
    cl_list = ingest.load_all(verbose=False)
    cl = {c["name"]: c for c in cl_list}
    names = sorted(cl)
    print(f"{len(names)} X-COP clusters loaded (VALIDATION data; no seal touched)")
    print()
    print("THE DIAGNOSTIC: perturb the baryon amplitude, watch the normaliser and")
    print("the excess co-move.  A clean normaliser has slope ~0 against the")
    print("amplitude; AX.5's worst case measured -0.99.")
    print()

    out = {}
    for key, idx in (("R_b,gas", 0), ("R_b,ne", 1)):
        sl_R, sl_E, corrs = [], [], []
        for nm in names:
            c = cl[nm]
            amps = np.linspace(-2 * AMP_SIGMA_DEX, 2 * AMP_SIGMA_DEX, 9)
            R, E = [], []
            for a in amps:
                r = one_cluster(c, a)
                R.append(r[idx])
                E.append(r[2])
            R, E = np.array(R), np.array(E)
            m = np.isfinite(R) & np.isfinite(E)
            if m.sum() < 4:
                continue
            # d log R_b / d (amplitude dex) and d log excess / d (amplitude dex)
            sR = np.polyfit(amps[m], R[m], 1)[0]
            sE = np.polyfit(amps[m], E[m], 1)[0]
            sl_R.append(sR)
            sl_E.append(sE)
            if np.std(R[m]) > 0 and np.std(E[m]) > 0:
                corrs.append(float(np.corrcoef(R[m], E[m])[0, 1]))
        sl_R, sl_E, corrs = map(np.array, (sl_R, sl_E, corrs))
        out[key] = dict(
            n=len(sl_R),
            dlogR_damp_median=float(np.median(sl_R)),
            dlogE_damp_median=float(np.median(sl_E)),
            corr_R_excess_median=float(np.median(corrs)) if len(corrs) else None,
            corr_R_excess_worst=float(corrs[np.argmax(np.abs(corrs))])
            if len(corrs) else None)
        print(f"  {key}")
        print(f"    d log R_b / d amp   median {np.median(sl_R):+.4f}")
        print(f"    d log E   / d amp   median {np.median(sl_E):+.4f}")
        print(f"    corr(R_b, excess) under amplitude noise:"
              f"  median {np.median(corrs):+.4f}   "
              f"worst {corrs[np.argmax(np.abs(corrs))]:+.4f}")
        print()

    print("AX.5's threshold for concern was |corr| near 0.99.")
    for k, v in out.items():
        w = abs(v["corr_R_excess_worst"] or 0)
        verdict = ("CONTAMINATED like AX.5's R_b,M" if w > 0.9 else
                   "coupled but not degenerate" if w > 0.5 else
                   "clean on this diagnostic")
        print(f"  {k:<9} worst |corr| = {w:.4f}  ->  {verdict}")

    p = os.path.join(HERE, "reaudit_baryon_radii.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(
        dict(n_clusters=len(names), amp_sigma_dex=AMP_SIGMA_DEX,
             note=("AX.5: a baryon-only radius is not automatically independent, "
                   "because g_bar already sits in the excess's denominator."),
             results=out), indent=1))
    print(f"\nwrote {p}")


if __name__ == "__main__":
    main()


def within_cluster_test():
    """The decisive question AX.5 does NOT settle by itself.

    A gas-amplitude error shifts R_b (a constant per cluster) and shifts the
    excess level (a constant per cluster).  Both are per-cluster CONSTANTS, and
    by Run AT's own rank identity a per-cluster constant cannot move a
    within-cluster slope.  So does the contamination actually reach the
    statistic AT reported?
    """
    import numpy as np
    import ingest
    cl = ingest.load_all(verbose=False)
    amps = np.array([-0.10, -0.05, 0.0, +0.05, +0.10])
    print()
    print("=" * 74)
    print("DOES THE CONTAMINATION REACH THE WITHIN-CLUSTER SLOPE?")
    print("=" * 74)
    for label, use_Rb in (("r / R_b,gas", True), ("physical r", False)):
        slopes = []
        for a in amps:
            per = []
            for c in cl:
                cc = dict(c)
                cc["ne_cm3"] = c["ne_cm3"] * 10.0 ** a
                p = ingest.build_profile(cc)
                Rg, _ = ingest.baryonic_radii(cc)
                r, gb, go = p["r"], p["gb"], p["go"]
                m = np.isfinite(gb) & np.isfinite(go) & (gb > 0) & (go > 0)
                if m.sum() < 5 or not np.isfinite(Rg):
                    continue
                x = np.log10(r[m] / Rg) if use_Rb else np.log10(r[m])
                y = np.log10(go[m] / gb[m])
                per.append(np.polyfit(x, y, 1)[0])
            slopes.append(np.median(per))
        slopes = np.array(slopes)
        drift = slopes.max() - slopes.min()
        print(f"  {label:<14} slope at amp = {list(amps)}")
        print(f"  {'':14} {np.array2string(slopes, precision=4)}")
        print(f"  {'':14} full drift over +-0.10 dex of gas amplitude: {drift:.4f}")
    print()
    print("  A per-cluster constant cannot move a within-cluster slope --")
    print("  Run AT's own rank identity.  The measurement above is the check.")


if __name__ == "__main__":
    within_cluster_test()
