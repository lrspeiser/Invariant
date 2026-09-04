"""The ISW separation, done three ways, with the contamination SIZED not assumed.

The integrated Sachs-Wolfe effect is a real physical signal with the same sign
structure as the hypothesis: voids are cold spots in both.  So it is treated as
a nuisance to be separated, and the separation is reported three ways:

  A  free-amplitude marginalisation      (the declared headline, M2)
  B  LCDM-normalised template subtraction, amplitude FIXED by theory, not fitted
  C  no ISW term at all                  (M1)

and, the number that actually decides whether ISW matters here, the BIAS the
LCDM ISW would induce in an ISW-free fit, compared with the null width.

    python isw_separation.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import certify_voidcmb as CV                                   # noqa: E402
import estimator as E                                          # noqa: E402
from measure import ISW_UK_PER_JPHI, OMEGA_M, DELTA_VOID, F_GROWTH   # noqa: E402


def main():
    t0 = time.time()
    ctx = CV.Context()
    ctx.guard.disarm("ISW separation; certificate already issued")
    bank, tried, dets = CV.rotation_bank(ctx, 1500, seed=7)

    J = ctx.map_t["I_phi_in"]
    dI = ctx.map_t["dI_q"]

    # ---- A: free-amplitude marginalisation, and the ISW coefficient's own null
    ev = [ctx.evaluate(R, "M2_isw_marginalised") for R in bank]
    isw_null = np.array([e["coef"][5] for e in ev])
    c2_null_A = np.array([e["c2c1"] for e in ev])
    true = ctx.evaluate(None, "M2_isw_marginalised", what="ISW")
    isw_hat = float(true["coef"][5])
    z_isw = (isw_hat - isw_null.mean()) / isw_null.std(ddof=1)

    # ---- B: subtract the LCDM ISW at its PREDICTED amplitude, then fit M1
    T_sub_full = ctx.T_full.copy()
    predicted_isw_uK = ISW_UK_PER_JPHI * (J - J.mean())
    pix, G = ctx.placement(None)
    T_true_sub = ctx.T_full[pix] - predicted_isw_uK
    evB = ctx.evaluate(None, "M1_no_isw", T_override=T_true_sub, what="ISW-B")
    # its null: the same subtraction travels with the void map, so it is applied
    # to every rotated placement too
    nullB = []
    for R in bank:
        p, g = ctx.placement(R)
        nullB.append(ctx.evaluate(R, "M1_no_isw",
                                  T_override=ctx.T_full[p] - predicted_isw_uK)["c2c1"])
    nullB = np.array(nullB)

    # ---- C: no ISW term
    evC = [ctx.evaluate(R, "M1_no_isw") for R in bank]
    c2_null_C = np.array([e["c2c1"] for e in evC])
    trueC = ctx.evaluate(None, "M1_no_isw", what="ISW-C")

    # ---- the decisive number: how much does the LCDM ISW bias an ISW-free fit?
    X = E.design({**ctx.map_t, **ctx.sky_cols(ctx.pix, ctx.G)}, ["dI_q"], ctx.G)
    w = E.weights_for_beta(X)
    bias_beta = float(w @ predicted_isw_uK)
    bias_c2c1 = bias_beta * E.UK_PER_MPCH_TO_C2C1
    sd = float(c2_null_C.std(ddof=1))

    doc = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lcdm_inputs=dict(Omega_m=OMEGA_M, delta_void=DELTA_VOID,
                         f_growth=F_GROWTH, uK_per_Jphi=ISW_UK_PER_JPHI),
        A_free_marginalisation=dict(
            c2_over_c1=float(true["c2c1"]), null_sd=float(c2_null_A.std(ddof=1)),
            z=float((true["c2c1"] - c2_null_A.mean()) / c2_null_A.std(ddof=1)),
            isw_coef=isw_hat, isw_null_sd=float(isw_null.std(ddof=1)),
            isw_z=float(z_isw),
            isw_over_lcdm=float(isw_hat / ISW_UK_PER_JPHI),
            isw_rms_uK=float(abs(isw_hat) * J.std()),
            lcdm_isw_rms_uK=float(abs(ISW_UK_PER_JPHI) * J.std())),
        B_lcdm_fixed_subtraction=dict(
            c2_over_c1=float(evB["c2c1"]), null_sd=float(nullB.std(ddof=1)),
            z=float((evB["c2c1"] - nullB.mean()) / nullB.std(ddof=1))),
        C_no_isw_term=dict(
            c2_over_c1=float(trueC["c2c1"]), null_sd=sd,
            z=float((trueC["c2c1"] - c2_null_C.mean()) / sd)),
        lcdm_isw_bias_on_c2_over_c1=float(bias_c2c1),
        lcdm_isw_bias_in_sigma=float(abs(bias_c2c1) / sd),
        error_inflation_A_over_C=float(c2_null_A.std(ddof=1) / sd),
        corr_isw_template_with_dIq=float(np.corrcoef(J, dI)[0, 1]),
        n_rotations=len(bank), seconds=time.time() - t0)
    p = os.path.join(HERE, "isw_separation.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, default=float))
    print(json.dumps(doc, indent=1, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
