"""fwd_worker.py -- one corpus from the INDEPENDENT forward model.

Job spec is a plain dict so the whole configuration -- halo ellipticity,
halo/baryon misalignment, halo/external-axis alignment fraction, tensor
amplitude, radial family, systematics scale -- travels to the worker.
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import estimators as E        # noqa: E402
import forward as F           # noqa: E402
import pipeline as PL         # noqa: E402

_POOL = None


def one(job):
    cfg, seed = job
    rng = np.random.default_rng(seed)
    kw = {k: cfg[k] for k in ("e_halo", "A_tensor", "mis_deg", "f_lss", "ring",
                              "sys_scale") if k in cfg}
    cl = F.corpus(cfg["kind"], rng, n_clu=cfg.get("n_clu", 12), **kw)
    S = PL.statistics(PL.cluster_rows(cl, F.sigma_crit))
    if S is None:
        return None
    if cfg.get("galaxy"):
        S.update(galaxy(cfg, rng))
    S["_cfg"] = cfg.get("tag", "")
    return S


def galaxy(cfg, rng, n_gal=30):
    proj, sd, p45, s45 = [], [], [], []
    for _ in range(n_gal):
        gd = F.emit_galaxy(rng, cfg.get("gal_kind", "none"),
                           q_amp=cfg.get("q_amp", 0.0),
                           mis_deg=cfg.get("gal_mis_deg", 25.0),
                           f_lss=cfg.get("gal_f_lss", 0.0))
        r = E.galaxy_m3(gd)
        if r is None:
            continue
        W, Cw = r
        psi = np.deg2rad(gd["axis_ext_obs"] - gd["pa_obs"])
        for off, P, S_ in ((0.0, proj, sd), (np.pi / 4, p45, s45)):
            u = np.array([np.cos(2 * psi + 2 * off), np.sin(2 * psi + 2 * off)])
            P.append(float(u @ np.array([W.real, W.imag])))
            S_.append(float(np.sqrt(max(u @ Cw @ u, 1e-30))))
    if len(proj) < 6:
        return {"G_ext": 0.0, "G_45": 0.0}
    v = np.array(proj) / np.array(sd)
    v45 = np.array(p45) / np.array(s45)
    return {"G_ext": float(np.sum(v) / np.sqrt(len(v))),
            "G_45": float(np.sum(v45) / np.sqrt(len(v45)))}


def get_pool(nproc=None):
    global _POOL
    if _POOL is None:
        import multiprocessing as mp
        nproc = nproc or max(1, min(20, (os.cpu_count() or 4) - 4))
        _POOL = mp.get_context("spawn").Pool(nproc)
    return _POOL


def close_pool():
    global _POOL
    if _POOL is not None:
        _POOL.close()
        _POOL.join()
        _POOL = None


def run(cfg, n, seed0, serial=False):
    jobs = [(cfg, seed0 + i) for i in range(n)]
    if serial or n < 8:
        return [r for r in (one(j) for j in jobs) if r is not None]
    return [r for r in get_pool().map(one, jobs, chunksize=4) if r is not None]
