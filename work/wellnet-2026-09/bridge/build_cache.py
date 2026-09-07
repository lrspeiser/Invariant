"""build_cache.py -- the pairwise column-product grids, computed once per
scene SHAPE and cached (results/cache_<tag>.npz).  Every mass / filament
ladder is an exact bilinear rescaling of these.

    python build_cache.py            # all scenes
    python build_cache.py fid        # one tag
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import guard
import scene as S

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")

#: tag -> (scene kwargs, grid kwargs, n_dir).  rho_f is a REFERENCE value
#: for the filament scenes (the ladder rescales it); the vacuum scenes carry
#: the separation ladder.
SCENES = {
    "fid": (dict(), dict(x_half=1.5, R_max_frac=1.0), 4000),
    "fid_fil": (dict(rho_f=1.0e-25), dict(x_half=1.0, R_max_frac=0.75), 2000),
    "D2": (dict(D=2.0 * S.MPC), dict(x_half=1.5, R_max_frac=1.0), 2000),
    "D3": (dict(D=3.0 * S.MPC), dict(x_half=1.5, R_max_frac=1.0), 2000),
    "D6": (dict(D=6.0 * S.MPC), dict(x_half=1.25, R_max_frac=0.75), 2000),
    "fid_wide": (dict(a=800.0 * S.KPC), dict(x_half=1.5, R_max_frac=1.0), 2000),
}


def cache_path(tag: str) -> str:
    return os.path.join(RES, f"cache_{tag}.npz")


def build(tag: str, force: bool = False) -> "S.PathFields":
    skw, gkw, n_dir = SCENES[tag]
    sc = S.TwoBody(**skw)
    grid = S.default_grid(sc.D, **gkw)
    p = cache_path(tag)
    if os.path.exists(p) and not force:
        return load(tag)
    t0 = time.perf_counter()
    pf = S.PathFields.build(sc, grid, n_dir)
    np.savez(p, xs=grid.xs, Rs=grid.Rs, n_dir=n_dir,
             MA=sc.MA, MB=sc.MB, a=sc.a, D=sc.D, rho_f=sc.rho_f, R_f=sc.R_f,
             seconds=time.perf_counter() - t0,
             **{"P_" + k: v for k, v in pf.Pgh.items()},
             **{"rho_" + k: v for k, v in pf.rho_g.items()})
    print(f"{tag}: grid {grid.shape}, n_dir {n_dir}, "
          f"{time.perf_counter() - t0:.0f}s -> {p}")
    return pf


def load(tag: str) -> "S.PathFields":
    d = np.load(cache_path(tag))
    sc = S.TwoBody(MA=float(d["MA"]), MB=float(d["MB"]), a=float(d["a"]),
                   D=float(d["D"]), rho_f=float(d["rho_f"]), R_f=float(d["R_f"]))
    grid = S.AxiGrid(d["xs"], d["Rs"])
    Pgh = {k: d["P_" + k] for k in S.PAIRS}
    rho_g = {g: d["rho_" + g] for g in "ABF"}
    return S.PathFields(sc, grid, int(d["n_dir"]), Pgh, rho_g,
                        seconds=float(d["seconds"]))


if __name__ == "__main__":
    guard.arm()
    os.makedirs(RES, exist_ok=True)
    tags = sys.argv[1:] or list(SCENES)
    for t in tags:
        build(t, force="--force" in sys.argv)
    print("provenance:", guard.summary()["assertion"])
