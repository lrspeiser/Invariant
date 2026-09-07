"""run_counterfactual.py -- the causal responses dO/dB (protocol item 5).

Paired scenes, one thing changed, everything else held.  For each edit and
each arm the PER-OBJECT difference of every invariant between the edited and
the base emission is formed (same scene, same halo where held, same noise),
and its mean and standard error across objects are the response.  Reported
as d(invariant)/d(edit) with sign, per universe, for the invariants the
distillation names and for the corpus-level discriminator score.

The object of interest is the RESPONSE, not the value: under a universal law
no edit that preserves the baryonic scene changes any residual; under CDM
the halo can be moved with the baryons fixed and the residuals follow it.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))

import arms as AR                        # noqa: E402
import discriminator as D                # noqa: E402
import gen1 as W                         # noqa: E402

RES = os.environ.get("EXTRACTION_RES", os.path.join(HERE, "results"))   # override for a scratch run

EDITS = {
    "bar_mass": ("all baryonic masses x 10^0.10, halo HELD", 0.10, "dex"),
    "bar_size": ("baryonic scale lengths x 10^0.06, halo HELD", 0.06, "dex"),
    "bar_hz": ("disc scale height x 10^0.10, halo HELD", 0.10, "dex"),
    "halo_mass": ("halo mass x 10^0.10, baryons HELD", 0.10, "dex"),
    "halo_conc": ("halo concentration x 10^0.10, baryons HELD", 0.10, "dex"),
    "halo_shape": ("halo ellipticity + 0.10, baryons HELD", 0.10, "1"),
    "halo_axis": ("halo axis rotated 45 deg, baryons HELD", 45.0, "deg"),
    "ext_axis": ("external tidal axis rotated 45 deg, local source HELD (halo held)", 45.0, "deg"),
    "ext_axis_follow": ("external axis rotated 45 deg, halo REDRAWN under the f_lss prior", 45.0, "deg"),
    "scramble": ("member positions scrambled, every radial profile preserved", 1.0, "1"),
    "history": ("formation history: t_merge x 3, present density field preserved", np.log10(3.0), "dex"),
    "path": ("photon path: void fraction -> 1 - void, endpoints preserved", 1.0, "1"),
    "bar_mass_follow": ("baryonic masses x 10^0.10, halo REDRAWN (follows the SHMR)", 0.10, "dex"),
}

GAL_INV = ["dz_1", "dz_0", "rz_1", "res_mean", "res_sd", "res_slope", "res_out_in", "vr_1", "vz_1",
           "ly_6", "res_6", "p2_5"]
CLU_INV = ["wl_lA", "wl_slope", "t_lA", "t_slope", "h_lA", "h_slope", "d_lA", "d_slope", "sl_res",
           "sl_obs", "q2_in", "q2_out", "q2_grad", "net_excess", "ep_ld", "ep_lt"]
PHASE_INV = ["pb_tot", "pe_tot", "p45_tot", "pe_minus_pb"]
GPHASE_INV = ["g3_ext", "g3_45", "g3_disc"]


def resp(a, b):
    d = np.asarray(a, float) - np.asarray(b, float)
    d = d[np.isfinite(d)]
    if len(d) < 10:
        return dict(mean=np.nan, se=np.nan, n=int(len(d)))
    return dict(mean=float(np.mean(d)), se=float(np.std(d) / np.sqrt(len(d))), n=int(len(d)))


def main():
    st = W.load_stack(os.path.join(RES, "G1_cf.npz"))
    import invariants as IV
    out = {"edits": {k: v[0] for k, v in EDITS.items()}, "responses": {}}
    # the discriminator trained on the main pool, for dS/dB of the corpus score
    main_st = W.load_stack(os.path.join(RES, "G1_main.npz"))
    sets = np.unique(main_st["U2"]["corpus"]["set"])
    cal, _ = D.split_sets(sets, 0.5, seed=0)
    test = D.Test(main_st, ["U2"], AR.CLASS_TAGS, cal)
    cf_sets = set(np.unique(st["U2"]["corpus"]["set"]).tolist())
    base_scores = {t: test.score(st, t, cf_sets) for t in ("U2", "U3", "U5f", "U6f", "U7f", "U8f", "U9f") if t in st}

    for tag in st:
        if "|" not in tag:
            continue
        base, edit = tag.split("|", 1)
        if base not in st:
            continue
        A, B = st[tag], st[base]
        # objects must be paired: same set ids and scene ids in the same order
        for grp in ("gal", "clu"):
            assert np.array_equal(A[grp]["set"], B[grp]["set"]), (tag, grp)
        r = {"edit": EDITS[edit][0], "delta": EDITS[edit][1], "unit": EDITS[edit][2]}
        for k in GAL_INV:
            r[k] = resp(A["gal"][k], B["gal"][k])
        ga, gb = IV.galaxy_phase_features(A["gal"]), IV.galaxy_phase_features(B["gal"])
        for k in GPHASE_INV:
            r[k] = resp(ga[k], gb[k])
        for k in CLU_INV:
            r[k] = resp(A["clu"][k], B["clu"][k])
        pa, pb = IV.cluster_phase_features(A["clu"]), IV.cluster_phase_features(B["clu"])
        for k in PHASE_INV:
            r[k] = resp(pa[k], pb[k])
        # discriminator score response per corpus
        sA = test.score(st, tag, cf_sets)
        sB = base_scores.get(base) or test.score(st, base, cf_sets)
        common = sorted(set(sA) & set(sB))
        r["S_discriminator"] = resp([sA[s] for s in common], [sB[s] for s in common])
        # truth: what the edit did to the true boosts (a check that the edit acted)
        if "T_boost_R2" in A["gal"]:
            r["T_boost_R2"] = resp(A["gal"]["T_boost_R2"], B["gal"]["T_boost_R2"])
            r["T_boost_z2"] = resp(A["gal"]["T_boost_z2"], B["gal"]["T_boost_z2"])
        if "T_boost_05" in A["clu"]:
            r["T_boost_05"] = resp(A["clu"]["T_boost_05"], B["clu"]["T_boost_05"])
        out["responses"][tag] = r
        print(f"{tag:<22} dz_1 {r['dz_1']['mean']:+.4f}+-{r['dz_1']['se']:.4f}  res_mean {r['res_mean']['mean']:+.4f}  "
              f"wl_lA {r['wl_lA']['mean']:+.4f}  t_lA {r['t_lA']['mean']:+.4f}  pe_tot {r['pe_tot']['mean']:+.3f}  "
              f"S {r['S_discriminator']['mean']:+.2f}+-{r['S_discriminator']['se']:.2f}", flush=True)

    # per-object halo truths vs residuals inside the CDM arm: the object-level
    # response of each invariant to the halo the object happens to have
    U2 = st["U2"]
    slopes = {}
    for k in ("dz_1", "rz_1", "res_mean", "res_out_in"):
        slopes[f"gal:{k} vs log M200 at fixed baryons"] = _partial_slope(
            U2["gal"][k], np.log10(U2["gal"]["T_M200"]), [U2["gal"]["lMd"], U2["gal"]["lRd"]])
        slopes[f"gal:{k} vs log c at fixed baryons"] = _partial_slope(
            U2["gal"][k], np.log10(U2["gal"]["T_c"]), [U2["gal"]["lMd"], U2["gal"]["lRd"]])
    for k in ("wl_lA", "t_lA", "h_lA", "d_lA", "wl_slope", "t_slope"):
        slopes[f"clu:{k} vs log M200 at fixed baryons"] = _partial_slope(
            U2["clu"][k], np.log10(U2["clu"]["T_M200"]), [U2["clu"]["lMgas"], U2["clu"]["lMstar"]])
        slopes[f"clu:{k} vs log c at fixed baryons"] = _partial_slope(
            U2["clu"][k], np.log10(U2["clu"]["T_c"]), [U2["clu"]["lMgas"], U2["clu"]["lMstar"]])
    out["object_level_slopes"] = slopes
    with open(os.path.join(RES, "CF_counterfactual.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("wrote CF_counterfactual.json")


def _partial_slope(y, x, controls):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    C = np.stack([np.asarray(c, float) for c in controls], 1)
    m = np.isfinite(y) & np.isfinite(x) & np.all(np.isfinite(C), 1)
    if m.sum() < 30:
        return dict(slope=np.nan, se=np.nan, n=int(m.sum()))
    A = np.column_stack([np.ones(m.sum()), x[m], C[m]])
    beta, res, rank, _ = np.linalg.lstsq(A, y[m], rcond=None)
    r = y[m] - A @ beta
    s2 = float(r @ r) / max(m.sum() - A.shape[1], 1)
    cov = s2 * np.linalg.inv(A.T @ A)
    return dict(slope=float(beta[1]), se=float(np.sqrt(cov[1, 1])), n=int(m.sum()))


if __name__ == "__main__":
    main()
