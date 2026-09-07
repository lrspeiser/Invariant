"""certify.py -- the Stage 4 sensitivity certificate for the bridge statistics.

Three statistics, each at both rho_* settings, get a STABLE TYPED IDENTIFIER
(the prospective validation's rule: no logic may depend on a human-readable
name) and the seven checks of stage4/certificate.py:

    BRIDGE.AMF.<RHO>.001   the matched-filter amplitude (units of the eps=0.03
                           prediction)         -- the detection statistic
    BRIDGE.Q.<RHO>.001     the core-minus-wing contrast in the between region
                           -- the model-free compensation statistic
    BRIDGE.DQ.<RHO>.001    between minus behind -- the statistic an m = 2 halo
                           pattern cannot move

  C1 responsive       stat over eps in {0.003, 0.03, 0.3}
  C2 not restatement  a pure normalisation nuisance (the endpoint masses
                      rescaled by -30..+30%, run through the same pipeline)
                      against the effect span
  C3 exchangeable     a FRESH untouched null half (300 draws, seed 909):
                      realised FPR at nominal 0.05 and the null mean
  C4 powered          at the PREDICTED effect: the member-safe ceiling
                      eps_safe(5%) from members.json, for a DECLARED sample of
                      N_PAIRS = 100 pairs under BF's detector
  C5 support          the nearest-endpoint distance the statistic reads
                      against the survey coverage, under TWO declared
                      coverages: BF's per-cluster 2.4 R500 and a contiguous
                      wide field (the estimator's own 1.5 D x 1.4 D)
  C6 out-of-grammar   a compensated bridge of a shape NOT in the P family
                      (Gaussian core, Gaussian wings, zero net) injected and
                      recovered through the estimator
  C7 nuisance-distinct the studentised response pattern of the statistic
                      vector against: the best positive-mass CDM mimic, the
                      QUMOND scalar null, the tensor action, a positive
                      filament, miscentring by 100 kpc, and BF's coherent PSF
                      residual

    python certify.py
"""
from __future__ import annotations

import json
import math
import os
import time
from typing import Dict

import numpy as np

import guard
import scene as S
import lensing as L
import estimator as E
import build_cache as BC
import certificate as CERT                          # stage4/certificate.py

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
KPC, MPC, MSUN = S.KPC, S.MPC, S.MSUN
EPS_REF = 0.03
N_PAIRS_DECLARED = 100
N_NULL_FRESH = int(os.environ.get("N_NULL_CERT", "300"))
R500_M = 1.0 * MPC                     # declared: R500 of a 3e14 Msun cluster
COVERAGE = {"BF_per_cluster_2.4R500": (0.09, 2.4),
            "contiguous_wide_field": (0.0, 6.0)}
RHO_TAGS = {"1e-24": "RHOSTAR_1E24", "rho_mean": "RHOSTAR_MEAN"}
STATS = {"A_mf": "AMF", "Q": "Q", "DQ": "DQ"}


def read_range(stat: str, geom: E.Geometry, D: float) -> tuple:
    """Nearest-endpoint distance the statistic reads, in R500."""
    half = 0.5 * D
    if stat == "DQ":
        far = geom.R_ex + geom.L_between      # the behind regions' far edge
        return (geom.R_ex / R500_M, far / R500_M)
    # the wing corner on the midplane is the between-region point farthest
    # from the nearer endpoint
    far = math.sqrt(half ** 2 + geom.w_wing ** 2)
    return (geom.R_ex / R500_M, far / R500_M)


def studentised(vec: Dict[str, float], sd: Dict[str, float]) -> np.ndarray:
    keys = ("S_core", "S_wing", "S_bcore", "S_bwing", "Q", "N_kappa_m2", "DQ")
    v = np.array([vec[k] / max(sd[k], 1e-300) for k in keys])
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


def gaussian_bridge_kappa(sky: L.Sky, geom: E.Geometry, amp: float,
                          s_core: float = 200.0 * KPC, s_wing: float = 600.0 * KPC):
    """A compensated feature NOT in the P family: kappa(y) = amp [e^{-y^2/2s1^2}
    - (s1/s2) e^{-y^2/2s2^2}] (zero net across y), uniform along the axis in
    the between region with soft ends."""
    X1, X2 = sky.mesh()
    prof = (np.exp(-0.5 * (X2 / s_core) ** 2)
            - (s_core / s_wing) * np.exp(-0.5 * (X2 / s_wing) ** 2))
    x0, x1 = geom.xA + geom.R_ex, geom.xB - geom.R_ex
    taper = 0.5 * (np.tanh((X1 - x0) / (150 * KPC)) - np.tanh((X1 - x1) / (150 * KPC)))
    return -amp * prof * taper


def main():
    guard.arm()
    t0 = time.perf_counter()
    bridge = json.load(open(os.path.join(RES, "bridge.json"), encoding="utf-8"))
    detect = json.load(open(os.path.join(RES, "detect.json"), encoding="utf-8"))
    attack = json.load(open(os.path.join(RES, "cdm_attack.json"), encoding="utf-8"))
    members = json.load(open(os.path.join(RES, "members.json"), encoding="utf-8"))
    pf = BC.load("fid")
    sc = pf.scene
    D = sc.D
    sky = L.default_sky(D)
    sv = L.Survey()
    scr = sv.sigma_crit_eff
    geom = E.Geometry(xA=-0.5 * D, xB=0.5 * D)
    template = np.load(os.path.join(RES, "template_kres.npy"))
    iso: Dict = {}
    sd = {k: detect["null"][k]["test_sd"] for k in E.STAT_KEYS}

    # ---------------------------------------------- shared: the endpoints
    mapsP = E.law_kappa_maps("P", pf, sky, scr, eps=EPS_REF, rho_star=S.RHO_STAR_FID,
                             iso_cache=iso)
    g1e, g2e = mapsP["g1_ends"], mapsP["g2_ends"]
    ends_stats = E.bridge_statistics(g1e, g2e, sky, geom, scr, template=template)

    # ---------------------------------------------- C2: the normalisation nuisance
    nuis_rows = {}
    for f in (0.7, 0.85, 1.0, 1.15, 1.3):
        r = E.bridge_statistics(f * g1e + (mapsP["g1"] - g1e), f * g2e + (mapsP["g2"] - g2e),
                                sky, geom, scr, template=template)
        nuis_rows[f] = {k: float(r[k]) for k in E.STAT_KEYS}

    # ---------------------------------------------- C3: a fresh untouched null half
    rng = np.random.default_rng(909)
    null = []
    for _ in range(N_NULL_FRESH):
        nz = L.noise_maps(sky, sv, rng)
        null.append(E.stat_vector(E.bridge_statistics(g1e + nz["e1"], g2e + nz["e2"], sky,
                                                      geom, scr, template=template)))
    null = np.array(null)

    # ---------------------------------------------- C6: out-of-grammar injection
    amp = 3.0e-3
    kinj = gaussian_bridge_kappa(sky, geom, amp)
    gi1, gi2 = L.shear_from_kappa(kinj, sky.pix, pad=3)
    rinj = E.bridge_statistics(g1e + gi1, g2e + gi2, sky, geom, scr, template=template)
    masks = E.region_masks(sky, geom)
    Q_true = float(kinj[masks["core_between"]].mean() - kinj[masks["wing_between"]].mean())
    DQ_true = float((kinj[masks["core_between"]].mean() - kinj[masks["core_behind"]].mean())
                    - (kinj[masks["wing_between"]].mean() - kinj[masks["wing_behind"]].mean()))
    A_true = float((kinj[masks["net_between"]] * template[masks["net_between"]]).sum()
                   / (template[masks["net_between"]] ** 2).sum())
    recovery = dict(Q=(rinj["Q"] - ends_stats["Q"]) / Q_true,
                    DQ=(rinj["DQ"] - ends_stats["DQ"]) / DQ_true,
                    A_mf=(rinj["A_mf"] - ends_stats["A_mf"]) / A_true if A_true != 0 else float("nan"),
                    injected=dict(Q=Q_true, DQ=DQ_true, A_mf_projection=A_true, amp=amp,
                                  shape="Gaussian core 200 kpc minus Gaussian wing 600 kpc, "
                                        "zero net, tapered along the axis"))

    # ---------------------------------------------- C7: nuisance patterns
    def strips_to_vec(d):
        return dict(S_core=d["S_core"], S_wing=d["S_wing"], S_bcore=d["S_bcore"],
                    S_bwing=d.get("S_bwing", 0.0), Q=d["Q"], N_kappa_m2=d.get("N_kappa", d.get("N_kappa_m2", 0.0)),
                    DQ=d["DQ"])
    # miscentring by 100 kpc along the axis (both centres shifted toward B)
    gmis = E.Geometry(xA=geom.xA + 100 * KPC, xB=geom.xB + 100 * KPC)
    rmis = E.bridge_statistics(g1e, g2e, sky, gmis, scr, template=template)
    # BF's coherent PSF residual, systematics only (no shape noise), one draw
    rng2 = np.random.default_rng(77)
    nz = L.noise_maps(sky, sv, rng2, systematics=True)
    # isolate the systematics part: the shape noise is drawn FIRST in
    # noise_maps, so the same seed without systematics reproduces it exactly
    rng3 = np.random.default_rng(77)
    sh = L.noise_maps(sky, sv, rng3, systematics=False)
    rpsf = E.bridge_statistics(g1e + (nz["e1"] - sh["e1"]), g2e + (nz["e2"] - sh["e2"]), sky,
                               geom, scr, template=template)
    comp = bridge["competitors"]
    mimic = attack["fits"]["incl0|between+behind|any_orientation"]["strips_mimic"]
    nuisances = {
        "CDM positive-mass best mimic": studentised(strips_to_vec(mimic), sd),
        "QUMOND scalar null (two-body)": studentised(comp["qumond"]["strip_stats"], sd),
        "tensor action f_E = 0.3": studentised(comp["tensor|fE=0.3"]["strip_stats"], sd),
        "positive filament 1e-25": studentised(comp["cdm|rho_f_dm=1e-25"]["strip_stats"], sd),
        "miscentring 100 kpc": studentised({k: float(rmis[k]) - float(ends_stats[k])
                                            for k in E.STAT_KEYS}, sd),
        "coherent PSF residual (BF)": studentised({k: float(rpsf[k]) - float(ends_stats[k])
                                                   for k in E.STAT_KEYS}, sd),
    }

    certificates = {}
    for rtag, rid in RHO_TAGS.items():
        rs = S.RHO_STAR_LADDER[rtag]
        ladder = {e: bridge["P_ladder"][f"{rtag}|eps={e}"]["strip_stats"] for e in S.EPS_LADDER}
        sig_vec = studentised(ladder[EPS_REF], sd)
        eps_pred = members["rows"][rtag]["eps_safe"]["tol_0.05"]
        for stat, sid in STATS.items():
            i = E.STAT_KEYS.index(stat)
            cid = f"BRIDGE.{sid}.{rid}.001"
            eff = np.array(S.EPS_LADDER)
            vals = np.array([ladder[e][stat] for e in S.EPS_LADDER])
            R = float(detect["responsiveness"][rtag][stat]["d_stat_d_eps"])
            checks = {}
            checks["C1_responsive"] = CERT.c1_responsive(
                lambda e, v=vals, ef=eff: float(np.interp(e, ef, v)), eff, tol=3.0 * sd[stat] / 30.0)
            checks["C2_not_a_restatement"] = CERT.c2_not_a_restatement(
                lambda e, v=vals, ef=eff: float(np.interp(e, ef, v)), eff,
                lambda f, s_=stat: nuis_rows[f][s_], list(nuis_rows.keys()))
            checks["C3_exchangeable"] = CERT.c3_exchangeable(
                float(ladder[EPS_REF][stat]), null[:, i])
            checks["C4_powered"] = CERT.c4_powered(
                responsiveness=abs(R), predicted_effect=eps_pred,
                noise_sd=sd[stat] / math.sqrt(N_PAIRS_DECLARED))
            checks["C4_powered"]["declared"] = dict(N_pairs=N_PAIRS_DECLARED,
                                                    eps_predicted="eps_safe at 5% member tolerance")
            rr = read_range(stat, geom, D)
            c5 = {cov: CERT.c5_support(rr, rng_) for cov, rng_ in COVERAGE.items()}
            checks["C5_support"] = dict(c5["contiguous_wide_field"],
                                        by_coverage=c5,
                                        note="issued under the contiguous wide-field coverage; "
                                             "the BF per-cluster coverage verdict is beside it")
            checks["C6_out_of_grammar"] = CERT.c6_out_of_grammar(
                float(abs(recovery["Q"])) if stat != "DQ" else float(abs(recovery["DQ"])))
            checks["C6_out_of_grammar"]["recovery_by_statistic"] = {
                k: float(v) for k, v in recovery.items() if k != "injected"}
            checks["C7_nuisance_distinct"] = CERT.c7_nuisance_distinct(sig_vec, nuisances)
            issued = CERT.certify(cid, checks, verbose=True)
            certificates[cid] = dict(statistic=stat, rho_star=rs, rho_tag=rtag, issued=issued,
                                     checks=checks,
                                     C5_under_BF_per_cluster_coverage=c5["BF_per_cluster_2.4R500"]["passed"],
                                     z_at_predicted_100_pairs=checks["C4_powered"]["z_at_predicted"])
    n_issued = sum(1 for c in certificates.values() if c["issued"])
    out = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               lane="work/wellnet-2026-09/bridge", stage="4 (identifiability)",
               n_certificates=len(certificates), n_issued=n_issued,
               certificates=certificates,
               declared=dict(N_pairs=N_PAIRS_DECLARED, coverage=COVERAGE, R500_Mpc=R500_M / MPC,
                             fresh_null_draws=N_NULL_FRESH, eps_ref=EPS_REF),
               out_of_grammar_injection=recovery,
               nuisance_patterns={k: v.tolist() for k, v in nuisances.items()},
               normalisation_nuisance_rows=nuis_rows,
               provenance=guard.summary(), wall_seconds=time.perf_counter() - t0)
    with open(os.path.join(RES, "certificate_bridge.json"), "w", encoding="utf-8",
              newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"\n{n_issued}/{len(certificates)} certificates issued; wrote certificate_bridge.json "
          f"in {out['wall_seconds']:.0f}s")


if __name__ == "__main__":
    main()
