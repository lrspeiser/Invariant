"""run_discriminate.py -- locate the information (protocol items 3, 4, 7, 8).

D0  size the test: A-vs-A null of the corpus-level AUC on untouched halves
D1  the discriminator: CDM (prior) vs the class at threshold; realised FPR
    and power at measured critical values; galaxies-only and clusters-only
D2  ablations: leave-one-channel-out, keep-one-channel-in, the named removals
    of the brief (shear phase, shear monopole, member identities, gas,
    internal galaxy dynamics, strong-lens timing, randomised alignment), the
    mean-profile-matched variant; permutation importance by channel
D3  localisation: per-object discrimination in strata of radius, harmonic,
    source class, cluster geometry, environment and acceleration regime
D4  sample size: z against the number of galaxies and clusters per corpus
D5  nuisance scans: the discriminator trained on the CDM prior, read on CDM
    with one nuisance moved -- f_lss first; directional-only and
    non-directional-only discriminators side by side
D6  untouched scenes: the discriminator transferred to the held-out library,
    and refitted there
D7  the independent generator: transferred and refitted
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import arms as AR                        # noqa: E402
import discriminator as D                # noqa: E402
import gen1 as W                         # noqa: E402
import invariants as IV                  # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run
CLASS = AR.CLASS_TAGS
GAL_CH = sorted(set(IV.galaxy_channel(k) for k in IV.galaxy_feature_names() + IV.GAL_PHASE))
CLU_CH = sorted(set(IV.cluster_channel(k) for k in IV.cluster_feature_names() + IV.CLU_PHASE))
DIRECTIONAL = {"clu_shear_quad_phase", "gal_m3_phase"}
QUAD_ALL = DIRECTIONAL | {"clu_shear_quad_power", "gal_harmonic_power"}


def halves(sets_use):
    """Two disjoint halves of a set collection, by sorted position."""
    u = sorted(sets_use)
    return set(u[0::2]), set(u[1::2])


def sep_all(test, stack, aud, sd, rates=False):
    """Separation of U2 (audit) from every class member and from the pool.

    The class members at threshold are NEAR-DUPLICATE corpora (same scene,
    same noise, a deformation below detectability), so the pooled AUC is an
    estimate over n_sets effective corpora, not 8 x n_sets: the pool's
    permutation p is taken from the single-arm U3 comparison and the
    effective n is recorded.  Per-object AUCs (one galaxy, one cluster) are
    reported beside the corpus AUC because the corpus AUC saturates.

    Detection rates use a NESTED split of the audit sets: the critical value
    comes from one half of the class's audit corpora (never seen in training),
    the realised rate and the power from the other half.  The first version
    took the critical value on the calibration corpora the classifier had been
    fitted on and realised a class false-positive rate of 0.48 at nominal
    0.05 -- the training-set critical value error, caught on the scratch pool.
    """
    sp, obj2 = test.score(stack, "U2", aud, per_object=True)
    out = {}
    pool = {}
    for t in CLASS:
        if t not in stack:
            continue
        sn, objn = test.score(stack, t, aud, per_object=True)
        out[t] = D.separation(sp, sn, sd)
        if t == "U3":
            for grp in obj2:
                out[t][f"obj_auc_{grp}"] = D.auc(objn[grp][0], obj2[grp][0])
        for s, v in sn.items():
            pool[(t, s)] = v
    out["pool"] = D.separation(sp, pool, sd)
    out["pool"]["p_perm"] = out["U3"]["p_perm"] if "U3" in out else out["pool"]["p_perm"]
    out["pool"]["n_neg_effective"] = len(sp)
    for grp in obj2:
        out["pool"][f"obj_auc_{grp}"] = out["U3"].get(f"obj_auc_{grp}")
    if rates:
        a_crit, a_rate = halves(aud)
        crit_pool, rate_pool = {}, {}
        for t in CLASS:
            if t in stack:
                for s, v in test.score(stack, t, a_crit).items():
                    crit_pool[(t, s)] = v
                for s, v in test.score(stack, t, a_rate).items():
                    rate_pool[(t, s)] = v
        sp_rate = {s: v for s, v in sp.items() if s in a_rate}
        out["rates_0.05"] = D.rate_at(sp_rate, crit_pool, rate_pool, 0.05)
        out["rates_0.01"] = D.rate_at(sp_rate, crit_pool, rate_pool, 0.01)
        out["rates_note"] = ("critical value from one half of the class's AUDIT corpora, rates on the other half; "
                             "the class pool holds near-duplicate corpora, so the effective n is the number of sets")
    return out


def main():
    t0 = time.time()
    st = W.load_stack(os.path.join(RES, "G1_main.npz"))
    sets = np.unique(st["U2"]["corpus"]["set"])
    cal, aud = D.split_sets(sets, 0.5, seed=0)
    out = {"n_sets": int(len(sets)), "n_cal": len(cal), "n_aud": len(aud),
           "class_tags": CLASS, "channels": {"gal": GAL_CH, "clu": CLU_CH}}

    # ------------------------------------------------------------ D0 sizing
    print("[D0] sizing", flush=True)
    sd, draws = D.null_auc_sd(st, ("U3", "H0", "U2"), cal, aud, n_rep=8)
    sd_g, _ = D.null_auc_sd(st, ("U3", "U2"), cal, aud, n_rep=6, groups=("gal",))
    sd_c, _ = D.null_auc_sd(st, ("U3", "U2"), cal, aud, n_rep=6, groups=("clu",))
    out["D0_sizing"] = dict(sd_null_auc=sd, sd_null_gal=sd_g, sd_null_clu=sd_c,
                            null_auc_mean=float(np.mean(draws)), n_null=len(draws),
                            null_auc_p95_abs=float(np.quantile(np.abs(draws - 0.5), 0.95)),
                            z_cap=D.Z_CAP,
                            note=("A-vs-A: one universe, random halves of the calibration sets "
                                  "labelled 1/0, fitted, scored on random halves of the audit sets"))
    print(f"   sd_null(auc) = {sd:.4f} (gal {sd_g:.4f}, clu {sd_c:.4f})  {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ D1 the discriminator
    print("[D1] the discriminator", flush=True)
    main_test = D.Test(st, ["U2"], CLASS, cal)
    out["D1_main"] = sep_all(main_test, st, aud, sd, rates=True)
    gal_test = D.Test(st, ["U2"], CLASS, cal, groups=("gal",))
    clu_test = D.Test(st, ["U2"], CLASS, cal, groups=("clu",))
    out["D1_galaxies_only"] = sep_all(gal_test, st, aud, sd_g, rates=True)
    out["D1_clusters_only"] = sep_all(clu_test, st, aud, sd_c, rates=True)
    # reference arms: Newton, systematics-only, and the class at FIDUCIAL amplitude
    sp = main_test.score(st, "U2", aud)
    ref = {}
    for t in ("U1", "U10", "U4f", "U5f", "U6f", "U7f", "U8f", "U9f"):
        if t in st:
            ref[t] = D.separation(sp, main_test.score(st, t, aud), sd)
    out["D1_reference_arms"] = ref
    # does the discriminator fire on the class at fiducial amplitude, on
    # systematics, on Newton?  Critical value from a class audit half never
    # used in training; rates on the other half.
    a_crit, a_rate = halves(aud)
    crit_pool = []
    for t in CLASS:
        crit_pool += list(main_test.score(st, t, a_crit).values())
    crit = float(np.quantile(np.array(crit_pool), 0.95))
    from universes.stats import rate_with_ci
    fire = {}
    for t in ("U1", "U10", "U4f", "U5f", "U6f", "U7f", "U8f", "U9f", "U2", "U3", "H0"):
        if t in st:
            v = np.array(list(main_test.score(st, t, a_rate).values()))
            fire[t] = rate_with_ci(int(np.sum(v >= crit)), len(v))
    out["D1_fire_rate_at_class_crit_0.05"] = fire
    print(f"   U2 vs pool: auc {out['D1_main']['pool']['auc']:.3f} z {out['D1_main']['pool']['z']:.2f}; "
          f"gal-only {out['D1_galaxies_only']['pool']['auc']:.3f}; clu-only {out['D1_clusters_only']['pool']['auc']:.3f}  "
          f"{time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ D2 ablations
    print("[D2] ablations", flush=True)
    abl = {}

    def run_abl(name, opts, groups=("gal", "clu")):
        t = D.Test(st, ["U2"], CLASS, cal, opts=opts, groups=groups)
        r = sep_all(t, st, aud, sd)
        abl[name] = dict(auc=r["pool"]["auc"], z=r["pool"]["z"], p=r["pool"]["p_perm"],
                         obj_auc_gal=r["pool"].get("obj_auc_gal"), obj_auc_clu=r["pool"].get("obj_auc_clu"),
                         per_member={k: v["auc"] for k, v in r.items() if k in CLASS})
        print(f"   {name:<40} auc {r['pool']['auc']:.3f}  z {r['pool']['z']:.2f}  "
              f"obj gal {r['pool'].get('obj_auc_gal', float('nan')):.3f} clu {r['pool'].get('obj_auc_clu', float('nan')):.3f}", flush=True)

    for c in GAL_CH + CLU_CH:
        run_abl(f"drop:{c}", dict(drop={c}))
    for c in GAL_CH + CLU_CH:
        run_abl(f"only:{c}", dict(keep={c}))
    run_abl("drop:shear angular phase (all phases)", dict(drop=DIRECTIONAL))
    run_abl("drop:all quadrupole/harmonic content", dict(drop=QUAD_ALL))
    run_abl("drop:shear monopole", dict(drop={"clu_shear_mono_raw", "clu_shear_mono_res"}))
    run_abl("drop:member identities (network)", dict(drop={"clu_network"}))
    run_abl("drop:gas (X-ray, SZ)", dict(drop={"clu_gas"}))
    run_abl("drop:internal galaxy dynamics",
            dict(drop={"gal_curve_raw", "gal_curve_res", "gal_harmonic_power", "gal_vertical", "gal_m3_phase"}))
    run_abl("drop:strong-lens timing", dict(drop={"clu_strong_lens"}))
    run_abl("drop:vertical AND scatter (curve residuals)", dict(drop={"gal_vertical", "gal_curve_res"}))
    run_abl("randomise:baryon-field alignment",
            dict(randomise_alignment=np.random.default_rng(1)))
    run_abl("only:non-directional", dict(drop=QUAD_ALL | {"clu_network"}))
    run_abl("only:directional", dict(keep=DIRECTIONAL | {"clu_shear_quad_power"}))
    # the mean-profile-matched variant: every level feature centred PER ARM on
    # its own calibration mean (label-dependent: a diagnostic, not a test)
    st_m = _matched_stack(st, cal)
    tm = D.Test(st_m, ["U2"], CLASS, cal)
    r = sep_all(tm, st_m, aud, sd)
    abl["matched:mean profiles removed per arm"] = dict(auc=r["pool"]["auc"], z=r["pool"]["z"],
                                                        p=r["pool"]["p_perm"],
                                                        per_member={k: v["auc"] for k, v in r.items() if k in CLASS})
    tm2 = D.Test(st_m, ["U2"], CLASS, cal, opts=dict(drop={"gal_vertical"}))
    r2 = sep_all(tm2, st_m, aud, sd)
    abl["matched:mean profiles removed AND no vertical"] = dict(auc=r2["pool"]["auc"], z=r2["pool"]["z"],
                                                                p=r2["pool"]["p_perm"])
    out["D2_ablations"] = abl
    out["D2_importance"] = main_test.importance(st, "U2", "U3", aud, n_rep=2)
    print(f"   ablations done {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ D3 localisation
    print("[D3] localisation", flush=True)
    loc = {}
    # by radius: galaxy inner vs outer rings; cluster inner/mid/outer bins
    inner_g = [f"res_{k}" for k in range(4)] + [f"ly_{k}" for k in range(4)] + ["vz_0", "vr_0", "rz_0", "rr_0", "dz_0"]
    outer_g = [f"res_{k}" for k in range(4, 8)] + [f"ly_{k}" for k in range(4, 8)] + ["vz_1", "vr_1", "rz_1", "rr_1", "dz_1"]
    for name, keep_names in (("galaxy inner rings (0.5-2.5 Rd)", inner_g), ("galaxy outer rings (2.5-5.2 Rd)", outer_g)):
        loc[name] = _named_test(st, cal, aud, sd, "gal", keep_names)
    for name, ks in (("cluster inner bins (0.12-0.4 R500)", range(0, 3)),
                     ("cluster mid bins (0.4-0.9 R500)", range(3, 6)),
                     ("cluster outer bins (0.9-2.3 R500)", range(6, 9))):
        names = [f"{p}_{k}" for k in ks for p in ("wres", "mono", "q2", "q4")]
        loc[name] = _named_test(st, cal, aud, sd, "clu", names)
    for name, ks in (("cluster X-ray inner (<0.4 R500)", range(0, 6)), ("cluster X-ray outer (>0.4 R500)", range(6, 10))):
        loc[name] = _named_test(st, cal, aud, sd, "clu", [f"tres_{k}" for k in ks] + [f"hres_{k}" for k in ks])
    # by harmonic
    loc["m=0 only (monopole profiles, both classes)"] = _named_test(
        st, cal, aud, sd, None, [f"wres_{k}" for k in range(9)] + [f"mono_{k}" for k in range(9)] + ["wl_lA", "wl_slope"],
        [f"res_{k}" for k in range(8)] + [f"lv_{k}" for k in range(8)] + ["res_mean", "res_slope"])
    loc["m=2 only (quadrupole power + phase)"] = _named_test(
        st, cal, aud, sd, None, [f"q2_{k}" for k in range(9)] + ["q2_in", "q2_mid", "q2_out", "q2_grad"] + IV.CLU_PHASE,
        [f"p2_{k}" for k in range(8)])
    loc["m=3 only (galaxy)"] = _named_test(st, cal, aud, sd, "gal", [f"p3_{k}" for k in range(8)] + IV.GAL_PHASE)
    loc["vertical only (galaxy dz, rz)"] = _named_test(st, cal, aud, sd, "gal", ["dz_0", "dz_1", "rz_0", "rz_1", "vz_0", "vz_1"])
    loc["vertical contrast only (dz_1)"] = _named_test(st, cal, aud, sd, "gal", ["dz_1"])
    loc["galaxy residual scatter only (res_sd, res_slope, res_out_in)"] = _named_test(
        st, cal, aud, sd, "gal", ["res_sd", "res_slope", "res_out_in"])
    loc["cluster profile shape only (wl_slope, t_slope, h_slope, d_slope)"] = _named_test(
        st, cal, aud, sd, "clu", ["wl_slope", "t_slope", "h_slope", "y_slope", "d_slope"])
    loc["cluster levels only (wl_lA, t_lA, h_lA, d_lA)"] = _named_test(
        st, cal, aud, sd, "clu", ["wl_lA", "t_lA", "h_lA", "y_lA", "d_lA"])
    loc["matter-photon covariance only (ep_*)"] = _named_test(st, cal, aud, sd, "clu", ["ep_ld", "ep_lt", "ep_dt"])
    loc["strong lensing only"] = _named_test(st, cal, aud, sd, "clu", ["thE_obs", "thE_pred", "sl_res", "sl_obs", "sl_pred", "n_img", "ldelay", "kb_in"])
    out["D3_localisation"] = loc
    # strata: per-object AUC of the main discriminator
    strata = {}
    _, objU2 = main_test.score(st, "U2", aud, per_object=True)
    _, objU3 = main_test.score(st, "U3", aud, per_object=True)
    for grp, key, label in (("gal", "lgb_out", "galaxy outer acceleration log g_bar [SI]"),
                            ("gal", "lSext", "galaxy environment log S_ext"),
                            ("gal", "lMd", "galaxy stellar mass"),
                            ("gal", "incl", "galaxy inclination"),
                            ("clu", "ell", "cluster baryon ellipticity"),
                            ("clu", "lMgas", "cluster gas mass"),
                            ("clu", "wcen", "cluster centroid shift (disturbance)"),
                            ("clu", "zc", "cluster redshift")):
        strata[label] = _strata(st, grp, key, objU2[grp], objU3[grp], aud)
    out["D3_strata"] = strata
    print(f"   localisation done {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ D4 sample size
    print("[D4] sample size", flush=True)
    ss = {}
    rng = np.random.default_rng(3)
    for ng, nc in ((1, 0), (3, 0), (10, 0), (30, 0), (0, 1), (0, 3), (0, 12), (3, 1), (10, 3), (30, 12)):
        sp = main_test.score(st, "U2", aud, n_sub={"gal": ng, "clu": nc}, rng=rng)
        sn = main_test.score(st, "U3", aud, n_sub={"gal": ng, "clu": nc}, rng=rng)
        # objects are sub-sampled by zeroing the excluded ones' contribution
        ss[f"gal{ng}_clu{nc}"] = D.separation(sp, sn, sd)
    out["D4_sample_size"] = ss

    # ------------------------------------------------------------ D5 nuisance scans
    print("[D5] nuisance scans", flush=True)
    sc = W.load_stack(os.path.join(RES, "G1_scans.npz"))
    if os.path.exists(os.path.join(RES, "G1_scans_extra.npz")):
        sc.update(W.load_stack(os.path.join(RES, "G1_scans_extra.npz")))
    allscan = set()
    for tag in sc:
        allscan |= set(np.unique(sc[tag]["corpus"]["set"]).tolist())
    dir_test = D.Test(st, ["U2"], CLASS, cal, opts=dict(keep=DIRECTIONAL | {"clu_shear_quad_power"}))
    phase_test = D.Test(st, ["U2"], CLASS, cal, opts=dict(keep=DIRECTIONAL))
    nd_test = D.Test(st, ["U2"], CLASS, cal, opts=dict(drop=QUAD_ALL | {"clu_network"}))
    vert_only = D.Test(st, ["U2"], CLASS, cal, opts=dict(keep={"gal_vertical"}), groups=("gal",))
    scat_only = D.Test(st, ["U2"], CLASS, cal, opts=dict(keep={"gal_curve_res"}), groups=("gal",))
    variants = (("full", main_test, sd), ("directional", dir_test, sd), ("phase_only", phase_test, sd),
                ("non_directional", nd_test, sd), ("vertical_only", vert_only, sd_g), ("curve_res_only", scat_only, sd_g))
    sn_U3 = {name: test.score(st, "U3", aud) for name, test, _ in variants}
    # BK's question restated: CDM against the TENSOR universe at fiducial
    # amplitude, as the halo's alignment moves onto the external axis
    sn_U5 = {name: test.score(st, "U5f", aud) for name, test, _ in variants}
    scans = {}
    for tag in sc:
        if not tag.startswith("S_"):
            continue
        r, r5 = {}, {}
        for name, test, sdx in variants:
            s_arm = test.score(sc, tag, allscan)
            r[name] = D.separation(s_arm, sn_U3[name], sdx)
            r5[name] = D.separation(s_arm, sn_U5[name], sdx)
        scans[tag] = {k: dict(auc=v["auc"], z=v["z"], auc_vs_tensor_fid=r5[k]["auc"], z_vs_tensor_fid=r5[k]["z"])
                      for k, v in r.items()}
        print(f"   {tag:<22} full {r['full']['auc']:.3f} dir {r['directional']['auc']:.3f} phase {r['phase_only']['auc']:.3f} "
              f"nondir {r['non_directional']['auc']:.3f} vert {r['vertical_only']['auc']:.3f} curve {r['curve_res_only']['auc']:.3f} "
              f"| vs U5f: phase {r5['phase_only']['auc']:.3f} nondir {r5['non_directional']['auc']:.3f}", flush=True)
    out["D5_scans"] = scans
    # the scans' class-side arms (U3 with sys x3 etc.) must NOT fire
    out["D5_scan_note"] = ("each scan arm is scored against the main pool's U3 audit corpora; "
                           "S_U3_* arms are the class under the same instrument change and must sit at chance")

    # ------------------------------------------------------------ D6 held-out scenes
    print("[D6] held-out library", flush=True)
    ho = W.load_stack(os.path.join(RES, "G1_heldout.npz"))
    ho_sets = np.unique(ho["U2"]["corpus"]["set"])
    ho_cal, ho_aud = D.split_sets(ho_sets, 0.5, seed=1)
    allho = set(ho_sets.tolist())
    # transferred (fixed classifier): null sd from two random halves of the class
    sd_fixed = _fixed_null_sd(main_test, ho, CLASS, allho)
    tr = {}
    spho = main_test.score(ho, "U2", allho)
    pool = {}
    for t in CLASS:
        if t in ho:
            snho = main_test.score(ho, t, allho)
            tr[t] = D.separation(spho, snho, sd_fixed)
            for s, v in snho.items():
                pool[(t, s)] = v
    tr["pool"] = D.separation(spho, pool, sd_fixed)
    for t in ("U1", "U10", "U5f", "U8f"):
        if t in ho:
            tr[t] = D.separation(spho, main_test.score(ho, t, allho), sd_fixed)
    out["D6_heldout_transferred"] = tr
    sd_ho, _ = D.null_auc_sd(ho, ("U3", "U2"), ho_cal, ho_aud, n_rep=6)
    ho_test = D.Test(ho, ["U2"], [t for t in CLASS if t in ho], ho_cal)
    out["D6_heldout_refit"] = sep_all(ho_test, ho, ho_aud, sd_ho, rates=True)
    out["D6_heldout_refit"]["sd_null"] = sd_ho
    out["D6_heldout_refit_vertical_only"] = sep_all(
        D.Test(ho, ["U2"], [t for t in CLASS if t in ho], ho_cal, opts=dict(keep={"gal_vertical"}), groups=("gal",)),
        ho, ho_aud, sd_ho)
    print(f"   transferred pool auc {tr['pool']['auc']:.3f}; refit pool auc {out['D6_heldout_refit']['pool']['auc']:.3f}  {time.time()-t0:.0f}s", flush=True)

    # ------------------------------------------------------------ D7 generator 2
    print("[D7] generator 2", flush=True)
    g2 = W.load_stack(os.path.join(RES, "G2_main.npz"))
    g2_sets = np.unique(g2["C"]["corpus"]["set"])
    g2_cal, g2_aud = D.split_sets(g2_sets, 0.5, seed=2)
    allg2 = set(g2_sets.tolist())
    sd_fixed2 = _fixed_null_sd(main_test, g2, ["C"], allg2)
    t2 = {}
    spg = main_test.score(g2, "D", allg2)
    for t in ("C", "N", "T"):
        t2[t] = D.separation(spg, main_test.score(g2, t, allg2), sd_fixed2)
    out["D7_g2_transferred"] = t2
    # NEWTON-CALIBRATED transfer: each generator has a Newtonian baryons-only
    # arm (U1, N).  The difference of their feature means is the two
    # generators' convention difference (disc vertical structure, gas model,
    # instrument), label-free with respect to CDM vs class; subtracting it
    # from generator 2 aligns the zero points before the fixed discriminator
    # is applied.  Possible in simulation only -- on real data this is the
    # systematic the test lives or dies by.
    g2c = _newton_calibrated(g2, st, "N", "U1")
    t2c = {}
    spgc = main_test.score(g2c, "D", allg2)
    for t in ("C", "N", "T"):
        t2c[t] = D.separation(spgc, main_test.score(g2c, t, allg2), sd_fixed2)
    out["D7_g2_transferred_newton_calibrated"] = t2c
    print(f"   G2 transferred, Newton-calibrated: D vs C auc {t2c['C']['auc']:.3f}", flush=True)
    sd_g2, _ = D.null_auc_sd(g2, ("C", "D"), g2_cal, g2_aud, n_rep=6)
    g2_test = D.Test(g2, ["D"], ["C"], g2_cal)
    spg2 = g2_test.score(g2, "D", g2_aud)
    rf = {"C": D.separation(spg2, g2_test.score(g2, "C", g2_aud), sd_g2),
          "T": D.separation(spg2, g2_test.score(g2, "T", g2_aud), sd_g2),
          "N": D.separation(spg2, g2_test.score(g2, "N", g2_aud), sd_g2),
          "sd_null": sd_g2}
    # ablations on generator 2
    ab2 = {}
    for name, opts in (("only:gal_vertical", dict(keep={"gal_vertical"})),
                       ("only:gal_curve_res", dict(keep={"gal_curve_res"})),
                       ("drop:gal_vertical", dict(drop={"gal_vertical"})),
                       ("drop:all quadrupole/harmonic content", dict(drop=QUAD_ALL)),
                       ("only:directional", dict(keep=DIRECTIONAL | {"clu_shear_quad_power"})),
                       ("only:clu_gas", dict(keep={"clu_gas"})),
                       ("only:clu_shear_mono_res", dict(keep={"clu_shear_mono_res"})),
                       ("only:clu_strong_lens", dict(keep={"clu_strong_lens"})),
                       ("only:clu_dynamics", dict(keep={"clu_dynamics"}))):
        tt = D.Test(g2, ["D"], ["C"], g2_cal, opts=opts)
        r = D.separation(tt.score(g2, "D", g2_aud), tt.score(g2, "C", g2_aud), sd_g2)
        ab2[name] = dict(auc=r["auc"], z=r["z"])
        print(f"   G2 {name:<40} auc {r['auc']:.3f}  z {r['z']:.2f}", flush=True)
    rf["ablations"] = ab2
    # generator-2 scans (f_lss, f_dd, zero scatter) with the G2-trained discriminator
    sc2 = {}
    snC = g2_test.score(g2, "C", g2_aud)
    dir2 = D.Test(g2, ["D"], ["C"], g2_cal, opts=dict(keep=DIRECTIONAL | {"clu_shear_quad_power"}))
    nd2 = D.Test(g2, ["D"], ["C"], g2_cal, opts=dict(drop=QUAD_ALL | {"clu_network"}))
    snC_dir, snC_nd = dir2.score(g2, "C", g2_aud), nd2.score(g2, "C", g2_aud)
    for tag in g2:
        if tag.startswith("D_"):
            sc2[tag] = dict(full=D.separation(g2_test.score(g2, tag, g2_aud), snC, sd_g2)["auc"],
                            directional=D.separation(dir2.score(g2, tag, g2_aud), snC_dir, sd_g2)["auc"],
                            non_directional=D.separation(nd2.score(g2, tag, g2_aud), snC_nd, sd_g2)["auc"])
    rf["scans"] = sc2
    out["D7_g2_refit"] = rf
    print(f"   G2 transferred D-vs-C auc {t2['C']['auc']:.3f}; refit {rf['C']['auc']:.3f}  {time.time()-t0:.0f}s", flush=True)

    out["elapsed_s"] = time.time() - t0
    with open(os.path.join(RES, "D_discriminator.json"), "w") as f:
        json.dump(out, f, indent=1, default=_conv)
    print(f"wrote D_discriminator.json ({time.time()-t0:.0f}s)")


def _conv(o):
    if isinstance(o, (np.floating, np.integer)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def _named_test(st, cal, aud, sd, grp, names_a, names_b=None):
    """A discriminator restricted to NAMED features (one or both object classes)."""
    class _Keep:
        pass
    if grp == "gal":
        groups, keep = ("gal",), {"gal": names_a}
    elif grp == "clu":
        groups, keep = ("clu",), {"clu": names_a}
    else:
        groups, keep = ("gal", "clu"), {"clu": names_a, "gal": names_b}
    test = _NamedTest(st, ["U2"], CLASS, cal, keep, groups)
    sp = test.score(st, "U2", aud)
    pool = {}
    per = {}
    for t in CLASS:
        sn = test.score(st, t, aud)
        per[t] = D.auc(np.array(list(sn.values())), np.array(list(sp.values())))
        for s, v in sn.items():
            pool[(t, s)] = v
    r = D.separation(sp, pool, sd)
    return dict(auc=r["auc"], z=r["z"], p=r["p_perm"], per_member=per, n_features=sum(len(v) for v in keep.values()))


class _NamedTest(D.Test):
    def __init__(self, stack, pos, neg, cal, keep_names, groups):
        self.opts = {}
        self.groups = groups
        self.keep_names = keep_names
        self.clf, self.names = {}, {}
        for grp in groups:
            X1, names, s1, _ = D._concat(stack, pos, grp, {})
            X0, _, s0, _ = D._concat(stack, neg, grp, {})
            cols = [j for j, k in enumerate(names) if k in keep_names[grp]]
            if not cols:
                continue
            self.clf[grp] = D.fit(X1[D.rows_in(s1, cal)][:, cols], X0[D.rows_in(s0, cal)][:, cols])
            self.names[grp] = [names[j] for j in cols]
            self.cols = getattr(self, "cols", {})
            self.cols[grp] = cols

    def score(self, stack, tag, sets_use, n_sub=None, rng=None, per_object=False):
        total, obj = {}, {}
        for grp in self.groups:
            if grp not in self.clf:
                continue
            X, names, sets, _ = D.feature_table(stack, tag, grp)
            m = D.rows_in(sets, sets_use)
            llr = D.logodds(self.clf[grp], X[m][:, self.cols[grp]])
            if per_object:
                obj[grp] = (llr, sets[m])
            for s, v in D.corpus_scores(llr, sets[m]).items():
                total[s] = total.get(s, 0.0) + v
        return (total, obj) if per_object else total


def _strata(st, grp, key, o2, o3, aud):
    """Per-object AUC of the discriminator in terciles of an observed property."""
    llr2, s2 = o2
    llr3, s3 = o3
    v2 = st["U2"][grp][key][D.rows_in(st["U2"][grp]["set"], aud)]
    v3 = st["U3"][grp][key][D.rows_in(st["U3"][grp]["set"], aud)]
    allv = np.concatenate([v2, v3])
    q = np.nanquantile(allv, [1 / 3, 2 / 3])
    out = {}
    for name, lo, hi in (("low", -np.inf, q[0]), ("mid", q[0], q[1]), ("high", q[1], np.inf)):
        m2 = (v2 >= lo) & (v2 < hi)
        m3 = (v3 >= lo) & (v3 < hi)
        if m2.sum() > 20 and m3.sum() > 20:
            out[name] = dict(auc=D.auc(llr3[m3], llr2[m2]), n=int(m2.sum() + m3.sum()),
                             range=[float(lo), float(hi)])
    return out


def _matched_stack(st, cal):
    """Every level feature centred per ARM on that arm's calibration mean."""
    import copy
    out = {}
    for tag in st:
        rec = {grp: {k: np.array(v, copy=True) for k, v in st[tag][grp].items()} for grp in st[tag]}
        for grp in ("gal", "clu"):
            m = D.rows_in(rec[grp]["set"], cal)
            for k in rec[grp]:
                if k.startswith(D.LEVEL_PREFIXES):
                    mu = np.nanmean(rec[grp][k][m])
                    if np.isfinite(mu):
                        rec[grp][k] = rec[grp][k] - mu
        out[tag] = rec
    return out


def _newton_calibrated(g2, st, tag2, tag1):
    """Generator-2 stack with every feature's zero point shifted by
    mean(generator 1, Newtonian arm) - mean(generator 2, Newtonian arm)."""
    out = {}
    shifts = {}
    for grp in ("gal", "clu"):
        shifts[grp] = {}
        for k in g2[tag2][grp]:
            if k.startswith("T_") or k in ("set", "scene") or k not in st[tag1][grp]:
                continue
            if k in ("pa_bar", "ax_ext", "psi_obs") or k.startswith(("z_re", "z_im", "z_c", "m3_")):
                continue
            a, b = np.nanmean(st[tag1][grp][k]), np.nanmean(g2[tag2][grp][k])
            if np.isfinite(a) and np.isfinite(b):
                shifts[grp][k] = a - b
    for tag in g2:
        rec = {grp: {k: np.array(v, copy=True) for k, v in g2[tag][grp].items()} for grp in g2[tag]}
        for grp in ("gal", "clu"):
            for k, sft in shifts[grp].items():
                rec[grp][k] = rec[grp][k] + sft
        out[tag] = rec
    return out


def _fixed_null_sd(test, stack, tags, sets_use, n_rep=30, seed=0):
    """Null sd of the AUC for a FIXED discriminator scoring two random halves
    of same-distribution corpora."""
    rng = np.random.default_rng(seed)
    scores = []
    for t in tags:
        if t in stack:
            scores += list(test.score(stack, t, sets_use).values())
    scores = np.array(scores)
    draws = []
    for _ in range(n_rep):
        p = rng.permutation(len(scores))
        h = len(scores) // 2
        draws.append(D.auc(scores[p[:h]], scores[p[h:2 * h]]))
    return float(np.std(draws))


if __name__ == "__main__":
    main()
