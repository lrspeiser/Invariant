"""worker.py -- one corpus -> one row of candidate statistics.

Uses Run BF's GENERATOR (universes/) unchanged -- the point of this lane is to
score BF's own universes -- but none of Run BF's DETECTORS.  The inference side
is ``estimators.py`` + ``pipeline.py``, which share no basis, no discretisation
and no nuisance model with ``universes/analysis.py``.

A separate, independently implemented forward model lives in ``forward.py`` and
is the lane's inverse-crime control.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.abspath(os.path.join(HERE, ".."))
for p in (LANE, HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

import estimators as E          # noqa: E402
import pipeline as PL           # noqa: E402
from universes import corpus as cp      # noqa: E402
from universes import generate as gn    # noqa: E402
from universes import physics as ph     # noqa: E402

N_GAL, N_CLU, N_SN = 30, 12, 200


def make_universe(arm, rng):
    uid, knob, sys_scale, noise_scale = arm
    if uid == "H0_scalar_null":
        u = ph.draw_scalar_null_universe(rng, sys_scale=sys_scale)
    else:
        u = ph.draw_universe(uid, rng, knob=knob, sys_scale=sys_scale)
    u.noise_scale = float(noise_scale)
    return u


def galaxy_stats(C):
    """Signed, studentised galaxy m=3 statistics."""
    proj, sd, p45, sd45, amps = [], [], [], [], []
    for gd in C.gal:
        r = E.galaxy_m3(gd)
        if r is None:
            continue
        W, Cw = r
        psi = np.deg2rad(gd["axis_ext_obs"] - gd["pa_obs"])
        for off, P, S in ((0.0, proj, sd), (np.pi / 4, p45, sd45)):
            u = np.array([np.cos(2 * psi + 2 * off), np.sin(2 * psi + 2 * off)])
            P.append(float(u @ np.array([W.real, W.imag])))
            S.append(float(np.sqrt(max(u @ Cw @ u, 1e-30))))
        amps.append(abs(W))
    if len(proj) < 6:
        return {"G_ext": 0.0, "G_45": 0.0, "_n_gal": len(proj), "_g_amp": 0.0}
    v = np.array(proj) / np.array(sd)
    v45 = np.array(p45) / np.array(sd45)
    return {"G_ext": float(np.sum(v) / np.sqrt(len(v))),
            "G_45": float(np.sum(v45) / np.sqrt(len(v45))),
            "_n_gal": len(proj), "_g_amp": float(np.median(amps))}


def one(job, want_diag=False):
    # job = (arm, seed) or (arm, seed, n_clu)
    arm, seed = job[0], job[1]
    n_clu = job[2] if len(job) > 2 else N_CLU
    lib = gn.get_lib()
    rng = np.random.default_rng(seed)
    u = make_universe(arm, rng)
    C = cp.draw_corpus(u, lib, rng, n_gal=N_GAL, n_clu=n_clu, n_sn=N_SN)
    rows = PL.cluster_rows(C, ph.sigma_crit)
    S = PL.statistics(rows)
    if S is None:
        return None
    S.update(galaxy_stats(C))
    S["_arm"] = arm[0]
    S["_knob"] = arm[1]
    S["_seed"] = int(seed)
    S["_n_clu_req"] = int(n_clu)
    if want_diag:
        S["_diag"] = PL.diagnostics(rows)
    return S


def one_diag(job):
    return one(job, want_diag=True)


_POOL = None


def _init():
    """Worker start-up: install the provenance guard, THEN load the library.

    Installing it in the worker is what makes the assertion real -- Run BF's
    ledger covered the parent process only, so a read inside a worker would
    not have raised.  Here any read outside the lane root raises in the worker
    and the batch fails loudly.
    """
    import guard
    try:
        guard.start()
    except Exception:                                          # noqa: BLE001
        pass
    gn.get_lib()


def get_pool(nproc=None):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        nproc = nproc or max(1, min(20, (os.cpu_count() or 4) - 4))
        ctx = mp.get_context("spawn")
        _POOL = ctx.Pool(nproc, initializer=_init)
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def run_batch(jobs, diag=False, serial=False, chunk=4):
    fn = one_diag if diag else one
    if serial or len(jobs) < 8:
        return [r for r in (fn(j) for j in jobs) if r is not None]
    return [r for r in get_pool().map(fn, jobs, chunksize=chunk) if r is not None]
