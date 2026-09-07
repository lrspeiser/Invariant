"""worker.py -- one paired set -> per-arm feature tables.

A JOB is (seed, arms, options).  The worker draws the scene ONCE from the
library named in the options, draws the universe constants ONCE, draws the
dark-matter haloes ONCE, and then emits every arm on that same scene with the
same noise streams.  Each arm's corpus is reduced by ``invariants.analyse_corpus``
(the universal law is cross-fitted PER ARM: the analyst never knows the label).

Nothing but feature tables leaves the worker; the corpora are far too large
to move.  Truth columns (prefixed ``T_``) travel with the features for the
response analysis and are masked from every discriminator.
"""
from __future__ import annotations

import copy
import os
import pickle
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.abspath(os.path.join(HERE, ".."))
for p in (LANE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import halo as H                                     # noqa: E402
import invariants as IV                              # noqa: E402
import paired as P                                   # noqa: E402
from universes import generate as gn                 # noqa: E402
from universes import scenes as sc                   # noqa: E402

N_GAL, N_CLU = 30, 12
RES = os.path.join(HERE, "results")
HELDOUT_SEED = 31415926
_LIBS = {}


def get_lib(which="train"):
    """'train' = BF's shared library; 'heldout' = a library BF never drew;
    'scrambled' = the training library with every member cloud angle-scrambled
    (radial profiles preserved exactly), geometry rebuilt."""
    if which in _LIBS:
        return _LIBS[which]
    if which == "train":
        lib = gn.get_lib()
    elif which == "heldout":
        fn = os.path.join(RES, f"scene_library_heldout_{HELDOUT_SEED}.pkl")
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                lib = pickle.load(f)
        else:
            lib = sc.build_library(seed=HELDOUT_SEED, n_gal=gn.N_GAL_LIB, n_clu=gn.N_CLU_LIB)
            os.makedirs(RES, exist_ok=True)
            with open(fn + ".tmp", "wb") as f:
                pickle.dump(lib, f, protocol=4)
            os.replace(fn + ".tmp", fn)
    elif which == "scrambled":
        fn = os.path.join(RES, "scene_library_scrambled.pkl")
        if os.path.exists(fn):
            with open(fn, "rb") as f:
                lib = pickle.load(f)
        else:
            base = get_lib("train")
            geoms = [sc.build_geom(g.clu.smoothed()) for g in base.geoms]
            lib = sc.SceneLibrary(galaxies=base.galaxies, geoms=geoms, seed=base.seed)
            with open(fn + ".tmp", "wb") as f:
                pickle.dump(lib, f, protocol=4)
            os.replace(fn + ".tmp", fn)
    else:
        raise ValueError(which)
    _LIBS[which] = lib
    return lib


def _pack(A, truths_g, truths_c):
    gal = {k: np.array([F.get(k, np.nan) for F in A["gal"]], float) for k in IV.galaxy_feature_names()}
    clu = {k: np.array([F.get(k, np.nan) for F in A["clu"]], float) for k in IV.cluster_feature_names()}
    for k in truths_g:
        gal["T_" + k] = np.asarray(truths_g[k], float)
    for k in truths_c:
        clu["T_" + k] = np.asarray(truths_c[k], float)
    return {"gal": gal, "clu": clu, "corpus": A["corpus"]}


def run_set(job):
    seed, arms, opt = job
    lib = get_lib(opt.get("lib", "train"))
    n_gal, n_clu = opt.get("n_gal", N_GAL), opt.get("n_clu", N_CLU)
    gi, ci = P.draw_scene_indices(seed, len(lib.galaxies), len(lib.geoms), n_gal, n_clu)
    gals = [lib.galaxies[i] for i in gi]
    geoms = [lib.geoms[i] for i in ci]
    out = {}
    for arm in arms:
        u = P.make_universe(arm["uid"], arm.get("knob"), arm.get("sys", 1.0),
                            arm.get("noise", 1.0), seed)
        cf = arm.get("cf") or {}
        gals_a = [P.perturb_galaxy(g, cf["scene"]) for g in gals] if cf.get("scene") else gals
        if cf.get("scramble"):
            slib = get_lib("scrambled")
            geoms_a = [slib.geoms[i] for i in ci]
        elif cf.get("scene"):
            geoms_a = [P.geom_with_scene(gm, P.perturb_cluster_scene(gm.clu, cf["scene"]))
                       for gm in geoms]
        else:
            geoms_a = geoms
        dm_g, dm_c = [None] * n_gal, [None] * n_clu
        if arm["uid"] == "U02_cdm":
            k = H.knobs(**(arm.get("halo") or {}))
            hold = cf.get("hold_halo", True)
            src_g = gals if hold else gals_a
            src_c = [gm.clu for gm in (geoms if hold else geoms_a)]
            dm_g, dm_c = P.draw_halos(seed, src_g, src_c, u.params, k)
            if cf.get("halo"):
                dm_g = [H.rescale_halo(d, **cf["halo"]) for d in dm_g]
                dm_c = [H.rescale_halo(d, **cf["halo"]) for d in dm_c]
        gal_d, tg = [], {"M200": [], "c": [], "f_lss": [], "q_amp": [], "q_h": [], "f_dd": [],
                         "boost_z1": [], "boost_R1": [], "boost_z2": [], "boost_R2": []}
        for g, d in zip(gals_a, dm_g):
            gd = P.emit_galaxy(u, g, seed, dm=d, nuis=arm.get("nuis"))
            gal_d.append(gd)
            T = gd["_truth"]
            for key in ("M200", "c", "f_lss", "q_amp", "q_h", "f_dd"):
                tg[key].append(d[key] if d is not None else np.nan)
            tg["boost_z1"].append(np.log10(T["gz_true"][0] / T["gzN"][0]))
            tg["boost_R1"].append(np.log10(T["gR_true"][0] / T["gN"][0]))
            tg["boost_z2"].append(np.log10(T["gz_true"][1] / T["gzN"][1]))
            tg["boost_R2"].append(np.log10(T["gR_true"][1] / T["gN"][1]))
        clu_d, tc = [], {"M200": [], "c": [], "f_lss": [], "ell": [], "pa": [],
                         "boost_05": [], "boost_15": [], "Mbar500": []}
        for gm, d in zip(geoms_a, dm_c):
            cd = P.emit_cluster(u, gm, seed, dm=d)
            clu_d.append(cd)
            T = cd["_truth"]
            for key in ("M200", "c", "f_lss", "ell", "pa"):
                tc[key].append(d[key] if d is not None else np.nan)
            R5 = gm.clu.R500
            gN = gm.clu.gN(T["rg"])
            tc["boost_05"].append(np.log10(np.interp(0.5 * R5, T["rg"], T["g_m"] / gN)))
            tc["boost_15"].append(np.log10(np.interp(1.5 * R5, T["rg"], T["g_m"] / gN)))
            tc["Mbar500"].append(np.log10(T["Mbar500"]))
        sn = P.emit_sn(u, seed)
        A = IV.analyse_corpus(gal_d, clu_d, sn, split_seed=seed % 7919)
        if A is None:
            continue
        # keep truths aligned with the surviving objects (reduce_* may drop some)
        names_g = [F.get("_name") for F in A["gal"]]
        rec = _pack(A, tg, tc)
        # align by count: reduce_galaxy drops none in practice; guard anyway
        if len(A["gal"]) != n_gal or len(A["clu"]) != n_clu:
            rec["gal"] = {k: v[:len(A["gal"])] if k.startswith("T_") else v for k, v in rec["gal"].items()}
            rec["clu"] = {k: v[:len(A["clu"])] if k.startswith("T_") else v for k, v in rec["clu"].items()}
        rec["corpus"]["a0"] = float(np.log10(u.params.get("a0", np.nan)))
        rec["corpus"]["scene_gal"] = gi.tolist()
        rec["corpus"]["scene_clu"] = ci.tolist()
        out[arm["tag"]] = rec
    return seed, out


# ----------------------------------------------------------------- the pool
_POOL = None


def _init(lib_names):
    import guard
    try:
        guard.start()
    except Exception:                                          # noqa: BLE001
        pass
    for n in lib_names:
        get_lib(n)


def get_pool(nproc=None, libs=("train",)):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        nproc = nproc or max(1, min(20, (os.cpu_count() or 4) - 4))
        _POOL = mp.get_context("spawn").Pool(nproc, initializer=_init, initargs=(tuple(libs),))
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def run_sets(seeds, arms, opt, serial=False, chunk=2):
    jobs = [(int(s), arms, opt) for s in seeds]
    if serial or len(jobs) < 4:
        return [run_set(j) for j in jobs]
    return get_pool(libs=(opt.get("lib", "train"),)).map(run_set, jobs, chunksize=chunk)


def stack(results, arms):
    """List of (seed, {tag: rec}) -> {tag: {'gal': {k: array}, 'clu': {...},
    'corpus': {k: array}, 'set': array of seeds per object}}."""
    tags = [a["tag"] for a in arms]
    out = {}
    for tag in tags:
        recs = [(s, r[tag]) for s, r in results if tag in r]
        if not recs:
            continue
        gal = {k: np.concatenate([r["gal"][k] for _, r in recs]) for k in recs[0][1]["gal"]}
        clu = {k: np.concatenate([r["clu"][k] for _, r in recs]) for k in recs[0][1]["clu"]}
        gal["set"] = np.concatenate([[s] * len(r["gal"]["lMd"]) for s, r in recs])
        clu["set"] = np.concatenate([[s] * len(r["clu"]["lMgas"]) for s, r in recs])
        gal["scene"] = np.concatenate([r["corpus"]["scene_gal"][:len(r["gal"]["lMd"])] for _, r in recs])
        clu["scene"] = np.concatenate([r["corpus"]["scene_clu"][:len(r["clu"]["lMgas"])] for _, r in recs])
        ck = [k for k in recs[0][1]["corpus"] if not k.startswith("scene_")]
        corpus = {k: np.array([r["corpus"].get(k, np.nan) for _, r in recs], float) for k in ck}
        corpus["set"] = np.array([s for s, _ in recs])
        out[tag] = {"gal": gal, "clu": clu, "corpus": corpus}
    return out


def save_stack(path, st):
    flat = {}
    for tag, d in st.items():
        for grp in ("gal", "clu", "corpus"):
            for k, v in d[grp].items():
                flat[f"{tag}|{grp}|{k}"] = np.asarray(v)
    np.savez_compressed(path, **flat)


def load_stack(path):
    z = np.load(path, allow_pickle=False)
    out = {}
    for key in z.files:
        # tags may themselves contain "|" (counterfactual arms: "U2|bar_mass"),
        # feature names never do: parse from the right
        tag, grp, k = key.rsplit("|", 2)
        out.setdefault(tag, {"gal": {}, "clu": {}, "corpus": {}})[grp][k] = z[key]
    return out
