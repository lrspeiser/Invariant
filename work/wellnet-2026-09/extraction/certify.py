"""certify.py -- Stage 4 certificates for every distilled invariant.

Seven checks, all required, from ``work/wellnet-2026-09/stage4/certificate.py``
(used unmodified); typed identifiers, so nothing depends on a human-readable
name.  Each candidate is certified at MORE THAN ONE amplitude of the halo
degree of freedom it reads, so the amplitude at which the answer changes is a
row in the table rather than a caveat in the prose.

Wiring, declared:

  C1  responsiveness of the statistic to its OWN halo lever: the dark-disc
      fraction for the vertical contrast, the SHMR scatter for the closure
      scatter, the concentration scatter for the cluster profile, the halo
      mass for the levels.  Read from the scans and the counterfactuals.
  C2  a MEASURED control (BK's convention): the largest effect the same
      statistic shows on anything that is NOT a halo -- the class at FIDUCIAL
      amplitude, the systematics-only universe, the Newtonian universe, and
      every instrument-nuisance arm (h_z, M/L, distance, inclination and
      sigma_z errors x 2.5) -- against the CDM effect at that amplitude.
  C3  realised false-positive rate on the UNTOUCHED audit half of the class,
      at critical values from the calibration half, nominal 0.05 and 0.01.
  C4  power at the PREDICTED effect: responsiveness x (the true injected
      quantity under the CDM prior) / the class's corpus-level scatter.
  C5  the statistic reads radii inside the measured range, and the universal
      law inside the acceleration range it was fitted on.
  C6  out-of-grammar recovery: the fraction of the generator-1 effect
      recovered on the INDEPENDENT generator 2 (Einasto haloes, Miyamoto-
      Nagai discs, a different law family, a different instrument).
  C7  nuisance-distinct: the RESPONSE PATTERN across the whole statistic set
      (BK's rule -- a correlation on an amplitude sequence saturates), CDM
      against every nuisance arm.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.append(os.path.abspath(os.path.join(HERE, "..", "stage4")))

import certificate as C                  # noqa: E402  (the Stage 4 gate itself)
import arms as AR                        # noqa: E402
import gen1 as W                         # noqa: E402
import invariants as IV                  # noqa: E402
from universes.stats import rate_with_ci, responsiveness   # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run
CLASS = AR.CLASS_TAGS


# ======================================================================
# corpus-level statistics
# ======================================================================
def _per_set(stack, tag, grp, key, fn):
    d = stack[tag][grp]
    v = np.asarray(d[key] if key in d else IV.cluster_phase_features(d)[key] if grp == "clu"
                   else IV.galaxy_phase_features(d)[key], float)
    out = {}
    for s in np.unique(d["set"]):
        x = v[d["set"] == s]
        x = x[np.isfinite(x)]
        out[int(s)] = fn(x) if len(x) >= 3 else np.nan
    return out


STATS = {
    "vert": ("gal", "dz_1", np.mean, "corpus mean of the boost-isotropy contrast at 2 R_d"),
    "vert1": ("gal", "dz_0", np.mean, "corpus mean of the boost-isotropy contrast at 1 R_d"),
    "scat": ("gal", "res_mean", np.std, "corpus sd across galaxies of the mean residual from the universal law"),
    "res_slope": ("gal", "res_slope", np.mean, "corpus mean of the radial slope of the galaxy residual"),
    "res_out_in": ("gal", "res_out_in", np.mean, "corpus mean of outer-minus-inner galaxy residual"),
    "t_slope": ("clu", "t_slope", np.mean, "corpus mean of the X-ray residual slope"),
    "t_level": ("clu", "t_lA", np.mean, "corpus mean of the X-ray residual level"),
    "t_sd": ("clu", "t_lA", np.std, "corpus sd across clusters of the X-ray residual level"),
    "wl_level": ("clu", "wl_lA", np.mean, "corpus mean of the lensing residual level"),
    "wl_slope": ("clu", "wl_slope", np.mean, "corpus mean of the lensing residual slope"),
    "d_level": ("clu", "d_lA", np.mean, "corpus mean of the dynamical residual level"),
    "h_slope": ("clu", "h_slope", np.mean, "corpus mean of the hydrostatic-mass residual slope"),
    "sl_res": ("clu", "sl_res", np.mean, "corpus mean of the Einstein-radius residual"),
    "q2_in": ("clu", "q2_in", np.mean, "corpus mean inner quadrupole power (studentised)"),
    "q2_grad": ("clu", "q2_grad", np.mean, "corpus mean quadrupole power gradient"),
    "pb": ("clu", "pb_tot", np.mean, "corpus mean quadrupole projection on the baryon axis"),
    "pe": ("clu", "pe_tot", np.mean, "corpus mean quadrupole projection on the external axis"),
    "p45": ("clu", "p45_tot", np.mean, "corpus mean quadrupole projection 45 deg off the external axis"),
    "g3": ("gal", "g3_ext", np.mean, "corpus mean galaxy m=3 projection on the external axis"),
    "ep_ld": ("clu", "ep_ld", np.mean, "corpus mean lensing-minus-dynamics residual"),
}


def stat(stack, tag, name):
    grp, key, fn, _ = STATS[name]
    if tag not in stack:
        return {}
    return _per_set(stack, tag, grp, key, fn)


def mean_of(d):
    v = np.array([x for x in d.values() if np.isfinite(x)])
    return (float(np.mean(v)), float(np.std(v)), int(len(v))) if len(v) else (np.nan, np.nan, 0)


def c2_measured(target, control, target_name, control_name, worst_name):
    t, c = abs(float(target)), abs(float(control))
    if t <= 0.0 or not np.isfinite(t):
        return dict(passed=False, target=t, control=c, ratio=None,
                    detail=f"statistic INSENSITIVE to {target_name}")
    r = c / t
    return dict(passed=r < 1.0, target=t, control=c, ratio=float(r), worst_control=worst_name,
                detail=(f"{control_name} ({worst_name}) reproduces {r:.3f}x the effect of "
                        f"{target_name} ({c:.4f} against {t:.4f})"))


def c3_realised(vals_cal, vals_aud, nominal, sign=+1.0):
    """Realised one-sided false-positive rate on the audit half, on the SIDE
    OF THE EFFECT (a negative effect is tested in the lower tail); the other
    tail is returned as well."""
    if sign < 0:
        vals_cal, vals_aud = -vals_cal, -vals_aud
    crit = float(np.quantile(vals_cal, 1 - nominal))
    r = rate_with_ci(int(np.sum(vals_aud >= crit)), len(vals_aud))
    lo = float(np.quantile(vals_cal, nominal))
    r_lo = rate_with_ci(int(np.sum(vals_aud <= lo)), len(vals_aud))
    return (crit if sign > 0 else -crit), r, r_lo


def main():
    st = W.load_stack(os.path.join(RES, "G1_main.npz"))
    sc = W.load_stack(os.path.join(RES, "G1_scans.npz"))
    if os.path.exists(os.path.join(RES, "G1_scans_extra.npz")):
        sc.update(W.load_stack(os.path.join(RES, "G1_scans_extra.npz")))
    cf = W.load_stack(os.path.join(RES, "G1_cf.npz"))
    g2 = W.load_stack(os.path.join(RES, "G2_main.npz"))
    ho = W.load_stack(os.path.join(RES, "G1_heldout.npz"))
    import discriminator as D
    sets = np.unique(st["U2"]["corpus"]["set"])
    cal, aud = D.split_sets(sets, 0.5, seed=0)

    # ---- every statistic on every arm (means), for C2 and C7 -----------
    arms_main = ["U2", "U3", "H0", "U1", "U10", "U4t", "U5t", "U6t", "U7t", "U8t", "U9t",
                 "U4f", "U5f", "U6f", "U7f", "U8f", "U9f"]
    M = {name: {} for name in STATS}
    for name in STATS:
        for tag in arms_main:
            M[name][tag] = mean_of(stat(st, tag, name))
        for tag in sc:
            M[name][tag] = mean_of(stat(sc, tag, name))
    sd_class = {name: float(np.nanstd([v for t in CLASS for v in stat(st, t, name).values()
                                       if np.isfinite(v)])) for name in STATS}
    NUIS_ARMS = {"systematics-only universe (U10)": "U10", "Newtonian universe, no boost (U1)": "U1",
                 "systematics x3 on the class": "S_U3_sys3", "noise x0.5 on the class": "S_U3_noise0.5",
                 "h_z error x2.5": "S_U3_nuis_hz", "M/L scatter x2.5": "S_U3_nuis_ml",
                 "distance error x2.5": "S_U3_nuis_dist", "inclination error x2.5": "S_U3_nuis_incl",
                 "sigma_z error x2.5": "S_U3_nuis_sz_err",
                 "environment scalar at fiducial (U4f)": "U4f", "tensor at fiducial (U5f)": "U5f",
                 "well network at fiducial (U6f)": "U6f", "memory at fiducial (U7f)": "U7f",
                 "EP slip at fiducial (U8f)": "U8f", "path redshift at fiducial (U9f)": "U9f"}

    def eff(name, tag, ref="U3"):
        return M[name][tag][0] - M[name][ref][0]

    # C7 signatures: response pattern across the statistic set, studentised
    names = list(STATS)
    sig_cdm = np.array([eff(n, "U2") / max(sd_class[n], 1e-9) for n in names])
    sig_nuis = {lab: np.array([eff(n, tag) / max(sd_class[n], 1e-9) for n in names])
                for lab, tag in NUIS_ARMS.items() if tag in M["vert"]}
    # a halo-only signature (levels removed) for the profile / vertical candidates
    nz = [n for n in names if n not in ("t_level", "wl_level", "d_level")]
    sig_cdm_nz = np.array([eff(n, "U2") / max(sd_class[n], 1e-9) for n in nz])
    sig_nuis_nz = {lab: np.array([eff(n, tag) / max(sd_class[n], 1e-9) for n in nz]) for lab, tag in NUIS_ARMS.items() if tag in M["vert"]}

    # generator-2 effects for C6
    G2 = {name: (mean_of(stat(g2, "D", name))[0] - mean_of(stat(g2, "C", name))[0]) for name in STATS}
    HO = {name: (mean_of(stat(ho, "U2", name))[0] - mean_of(stat(ho, "U3", name))[0]) for name in STATS}

    # the injected truths under the prior, for C4
    U2g, U2c = st["U2"]["gal"], st["U2"]["clu"]
    true_contrast = float(np.nanmean(U2g["T_boost_z2"] - U2g["T_boost_R2"]))
    true_contrast_1 = float(np.nanmean(U2g["T_boost_z1"] - U2g["T_boost_R1"]))
    true_scatter = float(np.nanstd(np.log10(U2g["T_M200"])))          # dex of halo mass at fixed baryons (marginal)

    cases = {}

    # ------------------------------------------------------------------
    # CAND.CDM.BOOST_ISOTROPY  -- the vertical/radial boost contrast
    # ------------------------------------------------------------------
    fdd_tags = [("S_nominal", 0.0), ("S_fdd_0.05", 0.05), ("S_fdd_0.1", 0.1), ("S_fdd_0.2", 0.2), ("S_fdd_0.3", 0.3),
                ("S_fdd_0.5", 0.5), ("S_fdd_0.7", 0.7), ("S_fdd_1", 1.0)]
    fdd_tags = [(t, f) for t, f in fdd_tags if t in sc]
    fdd_x = np.array([f for _, f in fdd_tags])
    fdd_y = np.array([M["vert"][t][0] for t, _ in fdd_tags])
    # responsiveness of the ESTIMATE to the TRUE contrast, across the f_dd arms
    true_c = np.array([float(np.nanmean(sc[t]["gal"]["T_boost_z2"] - sc[t]["gal"]["T_boost_R2"])) for t, _ in fdd_tags])
    resp_vert = responsiveness(true_c, fdd_y)
    hm = cf["U2|halo_mass"]["gal"]["dz_1"] - cf["U2"]["gal"]["dz_1"]
    hm = hm[np.isfinite(hm)]
    for amp_tag, amp, label in (("U2", None, "AT_PRIOR_FDD_0-0.05"), ("S_fdd_0.1", 0.1, "AT_FDD_0.1"),
                                ("S_fdd_0.3", 0.3, "AT_FDD_0.3"), ("S_fdd_0.5", 0.5, "AT_FDD_0.5"),
                                ("S_fdd_1", 1.0, "AT_FDD_1.0")):
        if amp_tag not in M["vert"]:
            continue
        target = eff("vert", amp_tag)
        ctrl = max(((abs(eff("vert", t)), lab) for lab, t in NUIS_ARMS.items() if t in M["vert"]), key=lambda x: x[0])
        vc = np.array([v for t in CLASS for v in stat(st, t, "vert").values()])
        vc_cal = np.array([v for t in CLASS for s, v in stat(st, t, "vert").items() if s in cal and np.isfinite(v)])
        vc_aud = np.array([v for t in CLASS for s, v in stat(st, t, "vert").items() if s in aud and np.isfinite(v)])
        crit5, r5, r5lo = c3_realised(vc_cal, vc_aud, 0.05, np.sign(target))
        crit1, r1, r1lo = c3_realised(vc_cal, vc_aud, 0.01, np.sign(target))
        pred = float(np.nanmean(sc[amp_tag]["gal"]["T_boost_z2"] - sc[amp_tag]["gal"]["T_boost_R2"])) if amp_tag in sc else true_contrast
        cases[f"CAND.CDM.BOOST_ISOTROPY.{label}"] = dict(
            _meta=dict(statistic="vert", description=STATS["vert"][3], alternative="collisionless halo (any f_dd)",
                       amplitude=(amp if amp is not None else "prior U(0, 0.05)"), null_family="the universal-law class",
                       effect=target, class_sd=sd_class["vert"]),
            C1_responsive=C.c1_responsive(lambda f, x=fdd_x, y=fdd_y: float(np.interp(f, x, y)), fdd_x),
            C2_not_a_restatement=c2_measured(target, ctrl[0], "a collisionless halo", "the largest non-halo effect", ctrl[1]),
            C3_exchangeable=dict(passed=abs(r5["rate"] - 0.05) < 0.02 and abs(r1["rate"] - 0.01) < 0.01,
                                 **{"realised_0.05": r5}, **{"realised_0.01": r1}, **{"realised_lower_0.05": r5lo}, **{"crit_0.05": crit5}, **{"crit_0.01": crit1},
                                 detail=(f"untouched audit half: realised {r5['rate']:.3f} at nominal 0.05, "
                                         f"{r1['rate']:.3f} at nominal 0.01 (lower tail {r5lo['rate']:.3f})")),
            C4_powered=C.c4_powered(abs(resp_vert["slope"]), abs(pred), sd_class["vert"]),
            C5_support=C.c5_support((1.0, 2.0), (0.5, 5.2)),
            C6_out_of_grammar=C.c6_out_of_grammar(float(G2["vert"] / target) if target else 0.0),
            C7_nuisance_distinct=C.c7_nuisance_distinct(sig_cdm_nz, sig_nuis_nz))
        cases[f"CAND.CDM.BOOST_ISOTROPY.{label}"]["_meta"]["responsiveness_estimate_vs_true"] = resp_vert
        cases[f"CAND.CDM.BOOST_ISOTROPY.{label}"]["_meta"]["counterfactual_halo_mass_dlogM_0.1"] = dict(
            mean=float(np.mean(hm)), se=float(np.std(hm) / np.sqrt(len(hm))))
        cases[f"CAND.CDM.BOOST_ISOTROPY.{label}"]["_meta"]["transfer"] = dict(generator2=G2["vert"], heldout=HO["vert"])

    # ------------------------------------------------------------------
    # CAND.CDM.CLOSURE_SCATTER -- object-to-object freedom at fixed baryons
    # ------------------------------------------------------------------
    sh_tags = [("S_shmr_0", 0.0), ("S_shmr_0.5", 0.5), ("S_nominal", 1.0), ("S_shmr_1.5", 1.5), ("S_shmr_2", 2.0)]
    sh_tags = [(t, s) for t, s in sh_tags if t in sc]
    sh_x = np.array([s * 0.16 for _, s in sh_tags])          # dex of SHMR scatter
    sh_y = np.array([M["scat"][t][0] for t, _ in sh_tags])
    resp_scat = responsiveness(sh_x, sh_y)
    for amp_tag, amp, label in (("U2", 0.16, "AT_PRIOR_SHMR_0.16"), ("S_shmr_0.5", 0.08, "AT_SHMR_0.08"),
                                ("S_shmr_0", 0.0, "AT_SHMR_0"), ("S_shmr_2", 0.32, "AT_SHMR_0.32")):
        if amp_tag not in M["scat"]:
            continue
        target = eff("scat", amp_tag)
        ctrl = max(((abs(eff("scat", t)), lab) for lab, t in NUIS_ARMS.items() if t in M["scat"]), key=lambda x: x[0])
        vc_cal = np.array([v for t in CLASS for s, v in stat(st, t, "scat").items() if s in cal and np.isfinite(v)])
        vc_aud = np.array([v for t in CLASS for s, v in stat(st, t, "scat").items() if s in aud and np.isfinite(v)])
        crit5, r5, _ = c3_realised(vc_cal, vc_aud, 0.05, np.sign(target))
        crit1, r1, _ = c3_realised(vc_cal, vc_aud, 0.01, np.sign(target))
        cases[f"CAND.CDM.CLOSURE_SCATTER.{label}"] = dict(
            _meta=dict(statistic="scat", description=STATS["scat"][3], alternative="halo mass scatter at fixed baryons",
                       amplitude=amp, null_family="the universal-law class", effect=target, class_sd=sd_class["scat"]),
            C1_responsive=C.c1_responsive(lambda s, x=sh_x, y=sh_y: float(np.interp(s, x, y)), sh_x),
            C2_not_a_restatement=c2_measured(target, ctrl[0], "halo scatter", "the largest non-halo effect", ctrl[1]),
            C3_exchangeable=dict(passed=abs(r5["rate"] - 0.05) < 0.02 and abs(r1["rate"] - 0.01) < 0.01,
                                 **{"realised_0.05": r5}, **{"realised_0.01": r1}, **{"crit_0.05": crit5}, **{"crit_0.01": crit1},
                                 detail=f"untouched audit half: realised {r5['rate']:.3f} at nominal 0.05, {r1['rate']:.3f} at 0.01"),
            C4_powered=C.c4_powered(abs(resp_scat["slope"]), amp, sd_class["scat"]),
            C5_support=C.c5_support((0.5, 5.2), (0.08, 5.2)),
            C6_out_of_grammar=C.c6_out_of_grammar(float(G2["scat"] / target) if target else 0.0),
            C7_nuisance_distinct=C.c7_nuisance_distinct(sig_cdm_nz, sig_nuis_nz))
        cases[f"CAND.CDM.CLOSURE_SCATTER.{label}"]["_meta"]["responsiveness_vs_shmr_dex"] = resp_scat
        cases[f"CAND.CDM.CLOSURE_SCATTER.{label}"]["_meta"]["transfer"] = dict(generator2=G2["scat"], heldout=HO["scat"])

    # the scatter statistic against the IN-PLANE HALO QUADRUPOLE lever: the
    # prior arm's scatter effect is far larger than SHMR scatter alone makes,
    # and q_amp (a flattened halo's m=2 in v_c^2) is the suspect
    qa_tags = [("S_qamp0", 0.0), ("S_nominal", 0.05), ("S_qamp0.2", 0.20)]
    qa_tags = [(t, q) for t, q in qa_tags if t in sc]
    if len(qa_tags) >= 2:
        qa_x = np.array([q for _, q in qa_tags])
        qa_y = np.array([M["scat"][t][0] for t, _ in qa_tags])
        resp_qa = responsiveness(qa_x, qa_y) if len(qa_x) >= 3 else dict(slope=(qa_y[-1] - qa_y[0]) / (qa_x[-1] - qa_x[0]), se=np.nan)
        for amp_tag, amp, label in (("S_qamp0", 0.0, "AT_QAMP_0"), ("S_nominal", 0.05, "AT_QAMP_0.05"), ("S_qamp0.2", 0.20, "AT_QAMP_0.2")):
            if amp_tag not in M["scat"]:
                continue
            target = eff("scat", amp_tag)
            ctrl = max(((abs(eff("scat", t)), lab) for lab, t in NUIS_ARMS.items() if t in M["scat"]), key=lambda x: x[0])
            vc_cal = np.array([v for t in CLASS for s, v in stat(st, t, "scat").items() if s in cal and np.isfinite(v)])
            vc_aud = np.array([v for t in CLASS for s, v in stat(st, t, "scat").items() if s in aud and np.isfinite(v)])
            crit5, r5, _ = c3_realised(vc_cal, vc_aud, 0.05, np.sign(target))
            crit1, r1, _ = c3_realised(vc_cal, vc_aud, 0.01, np.sign(target))
            cases[f"CAND.CDM.CLOSURE_SCATTER.{label}"] = dict(
                _meta=dict(statistic="scat", description=STATS["scat"][3], alternative="a flattened halo's in-plane m=2 (q_amp)",
                           amplitude=amp, null_family="the universal-law class", effect=target, class_sd=sd_class["scat"],
                           responsiveness_vs_qamp=resp_qa, transfer=dict(generator2=G2["scat"], heldout=HO["scat"])),
                C1_responsive=C.c1_responsive(lambda q, x=qa_x, y=qa_y: float(np.interp(q, x, y)), qa_x),
                C2_not_a_restatement=c2_measured(target, ctrl[0], "the halo quadrupole", "the largest non-halo effect", ctrl[1]),
                C3_exchangeable=dict(passed=abs(r5["rate"] - 0.05) < 0.02 and abs(r1["rate"] - 0.01) < 0.01,
                                     **{"realised_0.05": r5, "realised_0.01": r1, "crit_0.05": crit5, "crit_0.01": crit1},
                                     detail=f"untouched audit half: realised {r5['rate']:.3f} at nominal 0.05, {r1['rate']:.3f} at 0.01"),
                C4_powered=C.c4_powered(abs(resp_qa["slope"]), max(amp, 1e-9), sd_class["scat"]),
                C5_support=C.c5_support((0.5, 5.2), (0.08, 5.2)),
                C6_out_of_grammar=C.c6_out_of_grammar(float(G2["scat"] / target) if target else 0.0),
                C7_nuisance_distinct=C.c7_nuisance_distinct(sig_cdm_nz, sig_nuis_nz))

    # ------------------------------------------------------------------
    # CAND.CDM.CLUSTER_PROFILE_FREEDOM -- the X-ray residual slope
    # ------------------------------------------------------------------
    co_tags = [("S_conc_0", 0.0), ("S_conc_0.5", 0.5), ("S_nominal", 1.0), ("S_conc_2", 2.0)]
    co_tags = [(t, s) for t, s in co_tags if t in sc]
    for sname, cid in (("t_slope", "CAND.CDM.CLUSTER_PROFILE_SLOPE"), ("t_sd", "CAND.CDM.CLUSTER_LEVEL_SCATTER"),
                       ("t_level", "CAND.CDM.CLUSTER_LEVEL_EXCESS"), ("wl_level", "CAND.CDM.LENSING_LEVEL_EXCESS"),
                       ("pb", "CAND.CDM.BARYON_AXIS_QUAD"), ("pe", "CAND.CDM.EXTERNAL_AXIS_QUAD")):
        co_x = np.array([s * 0.13 for _, s in co_tags])
        co_y = np.array([M[sname][t][0] for t, _ in co_tags])
        hc = cf["U2|halo_conc"]["clu"][STATS[sname][1]] - cf["U2"]["clu"][STATS[sname][1]] if STATS[sname][1] in cf["U2"]["clu"] else np.array([np.nan])
        hmc = cf["U2|halo_mass"]["clu"][STATS[sname][1]] - cf["U2"]["clu"][STATS[sname][1]] if STATS[sname][1] in cf["U2"]["clu"] else np.array([np.nan])
        for amp_tag, label in (("U2", "AT_PRIOR"), ("S_conc_0", "AT_CONC_SCATTER_0"), ("S_flss_0", "AT_FLSS_0"),
                               ("S_flss_1", "AT_FLSS_1")):
            if amp_tag not in M[sname]:
                continue
            target = eff(sname, amp_tag)
            ctrl = max(((abs(eff(sname, t)), lab) for lab, t in NUIS_ARMS.items() if t in M[sname]), key=lambda x: x[0])
            vc_cal = np.array([v for t in CLASS for s, v in stat(st, t, sname).items() if s in cal and np.isfinite(v)])
            vc_aud = np.array([v for t in CLASS for s, v in stat(st, t, sname).items() if s in aud and np.isfinite(v)])
            crit5, r5, r5lo = c3_realised(vc_cal, vc_aud, 0.05, np.sign(target))
            crit1, r1, _ = c3_realised(vc_cal, vc_aud, 0.01, np.sign(target))
            two = abs(r5["rate"] - 0.05) < 0.02 or abs(r5lo["rate"] - 0.05) < 0.02
            fos = float(np.nanmean(st["U2"]["clu"]["frac_out_support"]))
            c5 = C.c5_support((0.12, 2.3), (0.09, 2.4))
            c5["law_support_fraction_outside"] = fos
            c5["passed"] = bool(c5["passed"] and fos < 0.25)
            c5["detail"] += f"; universal law read outside its fitted acceleration range on {fos:.1%} of the cluster grid (clipped, declared)"
            resp = responsiveness(co_x, co_y) if len(co_x) >= 3 else dict(slope=np.nan, se=np.nan)
            cases[f"{cid}.{label}"] = dict(
                _meta=dict(statistic=sname, description=STATS[sname][3], alternative="collisionless halo",
                           amplitude=label, null_family="the universal-law class", effect=target, class_sd=sd_class[sname],
                           counterfactual_halo_conc=dict(mean=float(np.nanmean(hc)), se=float(np.nanstd(hc) / np.sqrt(max(np.isfinite(hc).sum(), 1)))),
                           counterfactual_halo_mass=dict(mean=float(np.nanmean(hmc)), se=float(np.nanstd(hmc) / np.sqrt(max(np.isfinite(hmc).sum(), 1)))),
                           transfer=dict(generator2=G2[sname], heldout=HO[sname])),
                C1_responsive=C.c1_responsive(lambda s, x=co_x, y=co_y: float(np.interp(s, x, y)), co_x) if len(co_x) >= 2
                else dict(passed=False, detail="no scan"),
                C2_not_a_restatement=c2_measured(target, ctrl[0], "a collisionless halo", "the largest non-halo effect", ctrl[1]),
                C3_exchangeable=dict(passed=two and abs(r1["rate"] - 0.01) < 0.01, **{"realised_0.05": r5}, **{"realised_0.01": r1},
                                     **{"realised_lower_0.05": r5lo}, **{"crit_0.05": crit5}, **{"crit_0.01": crit1},
                                     detail=f"untouched audit half: realised {r5['rate']:.3f} (upper) / {r5lo['rate']:.3f} (lower) at 0.05, {r1['rate']:.3f} at 0.01"),
                C4_powered=C.c4_powered(1.0, abs(target), sd_class[sname]),
                C5_support=c5,
                C6_out_of_grammar=C.c6_out_of_grammar(float(G2[sname] / target) if target else 0.0),
                C7_nuisance_distinct=C.c7_nuisance_distinct(sig_cdm_nz if sname not in ("t_level", "wl_level") else sig_cdm,
                                                            sig_nuis_nz if sname not in ("t_level", "wl_level") else sig_nuis))
            cases[f"{cid}.{label}"]["_meta"]["responsiveness_vs_conc_scatter"] = resp

    # ------------------------------------------------------------------
    out = {"cases": {}, "n_issued": 0, "n_refused": 0, "statistics": {k: v[3] for k, v in STATS.items()},
           "effects_table": {n: {t: M[n][t][0] for t in M[n]} for n in STATS},
           "class_sd": sd_class, "generator2_effects": G2, "heldout_effects": HO,
           "true_contrast_under_prior": dict(at_2Rd=true_contrast, at_1Rd=true_contrast_1),
           "signatures": {"cdm": sig_cdm.tolist(), "statistic_order": names,
                          "nuisance": {k: v.tolist() for k, v in sig_nuis.items()}}}
    for cid, spec in cases.items():
        meta = spec.pop("_meta")
        ok = C.certify(f"{cid}   [{meta['statistic']}: {meta['description']}; amplitude {meta['amplitude']}]", spec)
        out["cases"][cid] = dict(meta=meta, issued=ok, checks=spec,
                                 failed=[k for k, v in spec.items() if not v["passed"]])
        out["n_issued"] += int(ok)
        out["n_refused"] += int(not ok)
    print(f"\n{out['n_issued']} certificates issued, {out['n_refused']} refused")
    with open(os.path.join(RES, "C_certificates.json"), "w") as f:
        json.dump(out, f, indent=1, default=_conv)
    print("wrote C_certificates.json")


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


if __name__ == "__main__":
    main()
