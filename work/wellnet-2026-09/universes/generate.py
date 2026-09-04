"""generate.py -- parallel corpus generation and blind analysis.

One worker process = one copy of the shared scene library (built from a fixed
seed, so every worker holds the IDENTICAL library) + the instrument + the blind
analysis.  A job is (arm_spec, seed) and returns only the feature vector, the
detector values and the auxiliary arrays -- never the corpus itself, which is
far too large to move between processes.
"""
from __future__ import annotations

import os

import numpy as np

from . import analysis as an
from . import corpus as cp
from . import physics as ph
from . import scenes as sc

LIB_SEED = 20260904
N_GAL_LIB, N_CLU_LIB = 45, 18
N_GAL, N_CLU, N_SN = 30, 12, 200

_LIB = None
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "results", f"scene_library_{LIB_SEED}_{N_GAL_LIB}_{N_CLU_LIB}.pkl")


def get_lib():
    """The shared scene library, built once and cached on disk.

    Every worker holds the IDENTICAL library, so a pairwise separation can
    never come from the scene prior.
    """
    global _LIB
    if _LIB is not None:
        return _LIB
    import pickle
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            _LIB = pickle.load(f)
        return _LIB
    _LIB = sc.build_library(seed=LIB_SEED, n_gal=N_GAL_LIB, n_clu=N_CLU_LIB)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    tmp = CACHE + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(_LIB, f, protocol=4)
    os.replace(tmp, CACHE)
    return _LIB


def make_universe(arm, rng):
    """arm = (uid, knob, sys_scale, noise_scale)."""
    uid, knob, sys_scale, noise_scale = arm
    if uid == "H0_scalar_null":
        u = ph.draw_scalar_null_universe(rng, sys_scale=sys_scale)
    else:
        u = ph.draw_universe(uid, rng, knob=knob, sys_scale=sys_scale)
    u.noise_scale = float(noise_scale)
    return u


def one(job):
    arm, seed = job
    lib = get_lib()
    rng = np.random.default_rng(seed)
    u = make_universe(arm, rng)
    C = cp.draw_corpus(u, lib, rng, n_gal=N_GAL, n_clu=N_CLU, n_sn=N_SN)
    A = an.analyse(C, split_seed=int(seed % 7919))
    if A is None:
        return None
    ax0 = an.estimate_axis(A["aux"], 0.0)
    ax45 = an.estimate_axis(A["aux"], 45.0)
    return {
        "features": A["features"], "detectors": A["detectors"],
        "a0_hat": an.estimate_a0(A["aux"]["lg"], A["aux"]["ly"]),
        "a0_true": float(np.log10(u.params["a0"] / 3.0856775814913673e13)),
        "axis_err": ax0["median_err_deg"], "axis_R": ax0["concentration_R"],
        "axis_proj": ax0["aligned_projection"],
        "axis_proj45": ax45["aligned_projection"],
        "knob": arm[1], "arm": arm[0], "seed": int(seed),
        "n_gal": A["aux"]["n_gal"], "n_clu": A["aux"]["n_clu"],
    }


_POOL = None


def get_pool(nproc=None):
    """One persistent worker pool for the whole run.

    Re-creating a pool per batch would rebuild/reload the scene library dozens
    of times; the library is the expensive object here.
    """
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        nproc = nproc or max(1, min(20, (os.cpu_count() or 4) - 4))
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(nproc, initializer=get_lib)
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close(); _POOL.join(); _POOL = None


def run_batch(jobs, chunk=3, serial=False):
    if serial or len(jobs) < 8:
        return [r for r in (one(j) for j in jobs) if r is not None]
    out = get_pool().map(one, jobs, chunksize=chunk)
    return [r for r in out if r is not None]


FEATURE_ORDER = None


def to_matrix(recs, keys=None):
    global FEATURE_ORDER
    if keys is None:
        if FEATURE_ORDER is None:
            FEATURE_ORDER = sorted(recs[0]["features"])
        keys = FEATURE_ORDER
    X = np.array([[r["features"].get(k, 0.0) for k in keys] for r in recs], float)
    return np.nan_to_num(X, nan=0.0, posinf=30.0, neginf=-30.0), list(keys)
