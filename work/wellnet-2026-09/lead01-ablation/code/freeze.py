"""Item 4, step 1 of 2: FREEZE the models on the existing ladder and SEAL them.

The existing held-out clusters have had four models ranked on them, so they are
validation data now, not a holdout.  For the fresh test they are therefore used
for TRAINING -- all 252 ladder systems -- and the fresh sample is the only
holdout.  This script writes the frozen coefficients and the scoring protocol
to a sealed JSON with a timestamp.  It NEVER reads the fresh sample.  Nothing
about the fresh sample's masses, radii or temperatures has been looked at when
this runs.
"""
from __future__ import annotations

import datetime
import hashlib
import json
import os

import numpy as np

import common as C
from boundary import RULES, build_profiles, compute


def main():
    d = C.load_ladder()
    profs, _ = build_profiles(d)
    seal = dict(
        sealed_utc=datetime.datetime.now(datetime.UTC).isoformat()
        .replace("+00:00", "Z"),
        prereg_sha256=json.load(open(os.path.join(C.LANE, "prereg_seal.json")))
        ["prereg_sha256"],
        ladder_sha256=C.sha256(C.LADDER),
        training_set="ALL 252 ladder systems inside the matched-acceleration "
                     "window, rungs 1-6.  The 52 clusters are included because "
                     "four models have already been ranked on them, so they no "
                     "longer function as a holdout.",
        response="log10( nu_obs / nu_RAR ),  nu_RAR = 1/(1-exp(-sqrt(g_bar/a0)))"
                 ", a0 = 1.2e-10 m/s^2",
        design="every model carries a free quadratic in log10 g_bar",
        window_log_gbar=[float(x) for x in C.system_table(d)["window"]],
    )

    models = {}
    for rule in ("BARY", "TAIL"):
        lp, _, _ = compute(d, profs, rule)
        t = C.system_table(d, lp_override=lp)
        A = C.design(t, "M1")
        c, *_ = np.linalg.lstsq(A, t["dev"], rcond=None)
        models[f"M1_{rule}"] = dict(
            terms=["1", "log10 g_bar", "(log10 g_bar)^2",
                   f"log10 |DeltaPhi_b| under rule {rule}"],
            coef=[float(x) for x in c], n_train=len(t["lg"]),
            train_rms=float(np.std(t["dev"] - A @ c)))
    t = C.system_table(d)
    for m in ("M0", "M3"):
        A = C.design(t, m)
        c, *_ = np.linalg.lstsq(A, t["dev"], rcond=None)
        terms = ["1", "log10 g_bar", "(log10 g_bar)^2"]
        if m == "M3":
            terms.append("step: 1 if the object is NOT a galaxy, else 0")
        models[m] = dict(terms=terms, coef=[float(x) for x in c],
                         n_train=len(t["lg"]),
                         train_rms=float(np.std(t["dev"] - A @ c)))
    seal["frozen_models"] = models
    seal["primary_model"] = "M1_BARY"
    seal["primary_null"] = "M3"

    seal["fresh_sample"] = dict(
        identity="Babyk, McNamara, Nulsen, Hogan, Vantyghem, Russell, Pulido, "
                 "Edge 2018, ApJ 857, 32 (2018ApJ...857...32B, arXiv:1803.00020)"
                 " -- 94 early-type galaxies with Chandra hydrostatic masses at "
                 "5 r_e; K-band photometry from 2MASS for the stellar mass.",
        why_fresh="No lane of this programme has used Babyk+2018.  It has never "
                  "appeared in the record (Runs A-AB) and is not in the "
                  "cluster-audit acquisition set.  It was acquired by a process "
                  "that computed no residual, no boost and no acceleration "
                  "ratio.",
        why_decisive="These objects ARE galaxies, so the class-step null "
                     "predicts NO offset for them, while the potential-depth "
                     "model predicts an offset set by their potential depth, "
                     "which sits between spirals and groups.  That is the one "
                     "configuration where the two models disagree.",
        cuts_declared_in_advance=[
            "finite r_5re, kT, M_gas(<5re), M_tot,hydro(<5re) and a matched "
            "2MASS K_s total magnitude",
            "M_tot_err_stat / M_tot <= 0.5",
            "PRIMARY subset: BCG_flag and cD_flag both blank -- these are "
            "unambiguously individual galaxies.  BCG/cD objects sit at the "
            "centre of a group or cluster halo, so their class and their "
            "hydrostatic mass are both contaminated; they are reported "
            "SEPARATELY as a secondary, never pooled with the primary.",
            "PRIMARY subset additionally restricted to objects whose log10 "
            "g_bar falls inside the fitted window; objects outside it are "
            "reported separately as an acknowledged extrapolation.",
        ],
        baryon_model=dict(
            gas="Babyk's own isothermal beta-model, M_gas(<r) from "
                "rho(r) = rho_0 [1+(r/r_c)^2]^(-3 beta/2), integrated.  GATE: "
                "must reproduce the tabulated M_gas(<5 r_e).",
            stars="Hernquist profile with scale a = r_e / 1.8153 and total "
                  "M_* = Upsilon_K L_K, Upsilon_K = 0.75 Msun/Lsun_K DECLARED "
                  "IN ADVANCE, M_K,sun = 3.27 (Willmer 2018).  Upsilon_K in "
                  "[0.6, 1.0] is carried as a systematic.",
            note="No dark matter enters either component.  The total mass is "
                 "Babyk's hydrostatic mass, not an NFW fit."),
        class_assignment="galaxy (step = 0) under the class-step null, because "
                         "they are individual galaxies.  The alternative "
                         "reading -- 'is it an X-ray hot-gas system' (step = 1) "
                         "-- is reported alongside, because that ambiguity is "
                         "exactly the weakness of a model whose only variable "
                         "is a label.",
        evaluation="ONE pass.  Frozen coefficients, no refitting of any kind.",
    )

    p = os.path.join(C.LANE, "fresh_seal.json")
    json.dump(seal, open(p, "w"), indent=2)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    seal["self_sha256"] = h
    json.dump(seal, open(p, "w"), indent=2)
    print(f"SEALED {seal['sealed_utc']}")
    print(f"   {p}")
    print(f"   sha256 of the seal before self-stamping: {h}")
    for k, v in models.items():
        print(f"   {k:<10} coef = "
              + ", ".join(f"{x:+.6f}" for x in v["coef"])
              + f"   (train rms {v['train_rms']:.4f} dex on {v['n_train']} "
                f"systems)")


if __name__ == "__main__":
    main()
