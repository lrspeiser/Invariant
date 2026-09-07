"""injection.py -- does the estimator invent a radius dependence that is not there?

WHAT IS BEING CHECKED. Fitting

    (required boost) - (RAR boost)  =  A + B log10(g_bar/a0) + C log10(r)

on the real data gives C = -11.98 +- 3.03, four sigma, permutation p = 0.011.
The RAR says g_obs is a function of g_bar ALONE, so C must be zero. Before that
is read as a violation, one thing has to be excluded: radius appears in the
CONSTRUCTION of both sides. g_bar = G M(<R) / R^2 carries an explicit 1/R^2, and
Delta_Sigma_bar is an integral over the same gas model at the same R. A ratio
built from those, regressed against R, is the shared-denominator setup that has
produced eight artefacts in this programme.

THE TEST. Replace the observed lensing with a signal generated from a KNOWN law,
using the real gas models, the real radii and the real error bars, then run the
identical estimator. Whatever the estimator returns on RAR-generated data is what
it manufactures from geometry alone.

    injected law          what C should come back as
    ------------------    --------------------------
    exact RAR             0 -- any departure is manufactured
    Newtonian             0 -- also a function of g_bar alone
    RAR x (r/Mpc)^-0.5    negative, and recovered at the injected size

The third case is the calibration: an estimator that cannot see a dependence that
IS there is no use either, so it has to pass both directions.

    python injection.py
"""
from __future__ import annotations

import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WELLNET = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WELLNET, "clustershear"))
sys.path.insert(0, HERE)
sys.path.insert(0, WELLNET)

import cosmo as C                                               # noqa: E402
import firstprinciples as FP                                    # noqa: E402
import modifications as MOD                                     # noqa: E402

OUT = os.path.join(HERE, "injection.json")
N_REAL = 400
SEED = 20260907


def wls(y, lg, lr, w):
    """Weighted least squares of y on [1, log g_bar, log r]. Returns coef, se."""
    X = np.column_stack([np.ones_like(lg), lg, lr])
    XtW = X.T * w
    cov = np.linalg.inv(XtW @ X)
    return cov @ (XtW @ y), np.sqrt(np.diag(cov))


def main():
    from holdout import loader                                  # noqa: E402
    loader.verify()

    os.environ.pop("R_WINDOW_MPC", None)
    rows = MOD.build_rows()
    loader.assert_not_sealed(sorted({r["cluster"] for r in rows}), "injection test")

    R = np.array([r["R"] for r in rows]) / C.MPC
    gb = np.array([r["g_bar"] for r in rows])
    dsb = np.array([r["ds_bar"] for r in rows])
    dso = np.array([r["ds_obs"] for r in rows])
    dse = np.array([r["ds_err"] for r in rows])

    keep = np.isfinite(dsb) & (dsb > 0) & (dse > 0) & np.isfinite(dso)
    R, gb, dsb, dso, dse = R[keep], gb[keep], dsb[keep], dso[keep], dse[keep]

    rar = MOD.boost_rar(gb)
    lg, lr = np.log10(gb / FP.A0), np.log10(R)
    err = dse / dsb
    w = 1.0 / err ** 2

    # the real measurement
    c_real, s_real = wls(dso / dsb - rar, lg, lr, w)

    LAWS = [("exact RAR", rar, 0.0),
            ("Newtonian (boost = 1)", np.ones_like(rar), 0.0),
            ("RAR x (r/Mpc)^-0.5", rar * R ** -0.5, None),
            ("RAR x (r/Mpc)^-1.0", rar * R ** -1.0, None)]

    rng = np.random.default_rng(SEED)
    print("what does the estimator return when the answer is KNOWN?")
    print("")
    print("  %-24s %-20s %-20s %s"
          % ("injected law", "C (log r coef)", "expected", "verdict"))

    res = []
    for label, boost, expect in LAWS:
        truth = dsb * boost                                     # noiseless signal
        cs = []
        for _ in range(N_REAL):
            fake = truth + rng.normal(0.0, dse)                 # real error bars
            c, _ = wls(fake / dsb - rar, lg, lr, w)
            cs.append(c[2])
        cs = np.array(cs)
        mean, sd = float(cs.mean()), float(cs.std())
        if expect is None:
            # what the injected law actually implies for C, measured the same way
            c_t, _ = wls(truth / dsb - rar, lg, lr, w)
            expect = float(c_t[2])
            verdict = "recovered" if abs(mean - expect) < 2 * sd / math.sqrt(N_REAL) \
                else "BIASED"
        else:
            verdict = ("clean" if abs(mean) < 2 * sd / math.sqrt(N_REAL)
                       else "MANUFACTURES %+.2f" % mean)
        res.append(dict(law=label, C_mean=mean, C_sd=sd, C_expected=expect,
                        verdict=verdict))
        print("  %-24s %+7.2f +- %-11.2f %+-20.2f %s"
              % (label, mean, sd, expect, verdict))

    manufactured = res[0]["C_mean"]
    corrected = c_real[2] - manufactured
    print("")
    print("  REAL DATA                %+7.2f +- %-11.2f" % (c_real[2], s_real[2]))
    print("  minus what RAR-generated data manufactures (%+.2f)" % manufactured)
    print("  CORRECTED radius coefficient  %+.2f +- %.2f   (%+.1f sigma)"
          % (corrected, s_real[2], corrected / s_real[2]))
    print("")
    if abs(manufactured) > 0.5 * abs(c_real[2]):
        print("  -> a large share of the signal is manufactured by the estimator.")
    elif abs(corrected / s_real[2]) >= 3:
        print("  -> the radius dependence SURVIVES. It is not a construction artefact:")
        print("     RAR-generated data with the same radii, gas models and error")
        print("     bars returns C = %+.2f, and the real data returns %+.2f."
              % (manufactured, c_real[2]))
    else:
        print("  -> does not survive at 3 sigma once the manufactured part is out.")

    out = dict(lane="clusterfirst", stage="injection-null",
               gas_file=os.environ.get("GAS_FILE", "gas_extended.json"),
               n_points=int(len(R)), n_realisations=N_REAL,
               C_real=float(c_real[2]), C_real_err=float(s_real[2]),
               C_manufactured=manufactured, C_corrected=float(corrected),
               sigma_corrected=float(corrected / s_real[2]), laws=res,
               note=("observations replaced by a known law on the real gas "
                     "models, radii and error bars; whatever comes back on "
                     "RAR-generated data is manufactured by geometry"),
               sealed_untouched=len(loader.sealed_names()))
    io.open(OUT, "w", newline="\n", encoding="utf-8").write(
        json.dumps(out, indent=1) + "\n")
    print("")
    print("wrote injection.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
