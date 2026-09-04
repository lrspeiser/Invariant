"""run_forward.py -- the INVERSE-CRIME CONTROL and the axis of the answer.

F1  the same statistics scored against a forward model that shares no basis,
    discretisation, solver or nuisance code with Run BF's generator
F2  sizing on an untouched audit half of that model
F3  THE ALIGNMENT SCAN.  Run BF's generator gives a collisionless halo a
    projected major axis of "baryon axis + N(0, 22 deg)" and NO knowledge of
    the surrounding structure.  Both halves of that are choices.  Real haloes
    align with the filament they sit in.  This scan varies
        mis_deg  the halo/baryon misalignment scatter
        f_lss    the fraction of the halo's alignment carried by the EXTERNAL
                 axis instead of the baryon axis
        e_halo   the halo quadrupole amplitude
    and reports, at each point, the power of the CDM discriminator and the
    false-positive rate of the new-gravity detector.  This is where the answer
    changes.
F4  C6: an OUT-OF-GRAMMAR quadrupole (a log-Gaussian ring, from neither the
    halo family nor the tensor family) -- is it recovered?
F5  responsiveness d(estimate)/d(injected) for both amplitudes
F6  the GALAXY channel with a TRIAXIAL halo, which Run BF's generator does not
    have: every CDM galaxy there gets a spherical halo, so its galaxy m=3
    detector has nothing to fire on.
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

import fwd_worker as FW                                     # noqa: E402
import guard                                                # noqa: E402
from universes.stats import rate_with_ci, responsiveness    # noqa: E402

RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)
N_F = int(os.environ.get("N_F", 400))

BASE_E = 0.45          # halo convergence ellipticity at the scan's base point
BASE_MIS = 22.0        # Run BF's generator value
BASE_A = 0.25          # tensor amplitude at the scan's base point

STATS = ("S_bar", "S_ext", "S_diff", "S_morph", "S_shape", "S_45")


def col(pool, k):
    return np.array([r.get(k, 0.0) for r in pool], float)


def crit(pool, k, alpha=0.05):
    v = col(pool, k)
    return dict(two=float(np.quantile(np.abs(v), 1 - alpha)),
                up=float(np.quantile(v, 1 - alpha)),
                lo=float(np.quantile(-v, 1 - alpha)))


def rates(pool, k, c):
    v = col(pool, k)
    return dict(two_sided=rate_with_ci(int(np.sum(np.abs(v) >= c["two"])), len(v)),
                upper=rate_with_ci(int(np.sum(v >= c["up"])), len(v)),
                lower=rate_with_ci(int(np.sum(-v >= c["lo"])), len(v)),
                mean=float(np.mean(v)), sd=float(np.std(v)))


def main():
    guard.start()
    t0 = time.time()
    out = {"n_per_arm": N_F, "base": dict(e_halo=BASE_E, mis_deg=BASE_MIS,
                                          A_tensor=BASE_A)}

    # -------------------------------------------------- F1/F2 arms and sizing
    arms = {
        "none": dict(kind="none"),
        "halo": dict(kind="halo", e_halo=BASE_E, mis_deg=BASE_MIS, f_lss=0.0),
        "halo_physical": dict(kind="halo", mis_deg=BASE_MIS, f_lss=0.0),
        "tensor": dict(kind="tensor", A_tensor=BASE_A),
        "tensor_strong": dict(kind="tensor", A_tensor=0.5),
        "both": dict(kind="both", e_halo=BASE_E, mis_deg=BASE_MIS, f_lss=0.0,
                     A_tensor=BASE_A),
    }
    pools = {}
    for i, (name, cfg) in enumerate(arms.items()):
        for half, off in (("cal", 0), ("aud", 250_000)):
            pools[(name, half)] = FW.run(cfg, N_F, 7_700_000 + 30_011 * i + off)
        print(f"  {name:<15} done {time.time() - t0:6.1f}s", flush=True)

    # a NEW-GRAVITY detector is calibrated on everything that is not new
    # gravity: the empty universe and the dark-matter universe.
    cal_ng = pools[("none", "cal")] + pools[("halo", "cal")]
    aud_ng = pools[("none", "aud")] + pools[("halo", "aud")]
    # a CDM detector is calibrated on the modified-gravity universes
    cal_cdm = pools[("none", "cal")] + pools[("tensor", "cal")] \
        + pools[("tensor_strong", "cal")]
    aud_cdm = pools[("none", "aud")] + pools[("tensor", "aud")] \
        + pools[("tensor_strong", "aud")]
    C_ng = {s: crit(cal_ng, s) for s in STATS}
    C_cdm = {s: crit(cal_cdm, s) for s in STATS}
    out["F2_sizing"] = {
        "new_gravity_null": {s: dict(crit=C_ng[s], realised=rates(aud_ng, s, C_ng[s]))
                             for s in STATS},
        "cdm_null": {s: dict(crit=C_cdm[s], realised=rates(aud_cdm, s, C_cdm[s]))
                     for s in STATS}}
    out["F1_arms"] = {name: {s: dict(mean=float(np.mean(col(pools[(name, "aud")], s))),
                                     sd=float(np.std(col(pools[(name, "aud")], s))))
                             for s in STATS}
                      for name in arms}
    out["F1_rates"] = {name: {s: dict(vs_newgrav_null=rates(pools[(name, "aud")], s, C_ng[s]),
                                      vs_cdm_null=rates(pools[(name, "aud")], s, C_cdm[s]))
                              for s in ("S_bar", "S_ext", "S_diff")}
                       for name in arms}

    # ------------------------------------------------------ F3 alignment scan
    print("F3 alignment scan", flush=True)
    scan = {}
    grid = []
    for mis in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        grid.append(dict(mis_deg=mis, f_lss=0.0, e_halo=BASE_E))
    for fl in (0.25, 0.5, 0.75, 1.0):
        grid.append(dict(mis_deg=BASE_MIS, f_lss=fl, e_halo=BASE_E))
    for e in (0.10, 0.20, 0.30, 0.60, 0.80):
        grid.append(dict(mis_deg=BASE_MIS, f_lss=0.0, e_halo=e))
    for mis in (0.0, 30.0, 60.0):
        for fl in (0.25, 0.5, 0.75):
            grid.append(dict(mis_deg=mis, f_lss=fl, e_halo=BASE_E))
    for i, g in enumerate(grid):
        cfg = dict(kind="halo", **g)
        pool = FW.run(cfg, N_F, 8_800_000 + 40_009 * i)
        key = f"mis{g['mis_deg']:g}_flss{g['f_lss']:g}_e{g['e_halo']:g}"
        scan[key] = dict(
            config=g,
            cdm_detector_power=rates(pool, "S_bar", C_cdm["S_bar"]),
            newgrav_fp_S_ext=rates(pool, "S_ext", C_ng["S_ext"]),
            newgrav_fp_S_diff=rates(pool, "S_diff", C_ng["S_diff"]),
            S_morph=rates(pool, "S_morph", C_cdm["S_morph"]))
    out["F3_alignment_scan"] = scan
    print(f"  {len(grid)} grid points, {time.time() - t0:.0f}s", flush=True)

    # ----------------------------------------------------- F4 out-of-grammar
    print("F4 out-of-grammar injection", flush=True)
    og = {}
    for A in (0.15, 0.3, 0.6):
        pool = FW.run(dict(kind="tensor", A_tensor=A, ring=True),
                      N_F, 9_900_000 + int(1000 * A))
        og[str(A)] = dict(S_ext=rates(pool, "S_ext", C_ng["S_ext"]),
                          S_shape=rates(pool, "S_shape", C_ng["S_shape"]))
    out["F4_out_of_grammar_ring"] = og

    # --------------------------------------------------- F5 responsiveness
    print("F5 responsiveness", flush=True)
    es = [0.0, 0.1, 0.2, 0.3, 0.45, 0.6, 0.8]
    As = [0.0, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75]
    e_scan, a_scan = {}, {}
    for i, e in enumerate(es):
        p = FW.run(dict(kind="halo", e_halo=e, mis_deg=BASE_MIS, f_lss=0.0),
                   250, 10_100_000 + 3001 * i)
        e_scan[str(e)] = {s: float(np.mean(col(p, s))) for s in STATS}
    for i, A in enumerate(As):
        p = FW.run(dict(kind="tensor", A_tensor=A), 250, 10_200_000 + 3001 * i)
        a_scan[str(A)] = {s: float(np.mean(col(p, s))) for s in STATS}
    out["F5_e_scan"] = e_scan
    out["F5_A_scan"] = a_scan
    out["F5_responsiveness_vs_e"] = {
        s: responsiveness(es, [e_scan[str(e)][s] for e in es]) for s in STATS}
    out["F5_responsiveness_vs_A"] = {
        s: responsiveness(As, [a_scan[str(A)][s] for A in As]) for s in STATS}

    # ---------------------------------------------------- F6 galaxy channel
    print("F6 galaxy channel with a triaxial halo", flush=True)
    gal = {}
    gcfg = dict(kind="none", n_clu=3, galaxy=True)
    base = {**gcfg, "gal_kind": "none", "q_amp": 0.0}
    pool0 = FW.run(base, N_F, 11_100_000)
    cg = crit(pool0, "G_ext")
    cg45 = crit(pool0, "G_45")
    gal["_null"] = dict(crit=cg, crit_45=cg45,
                        mean=float(np.mean(col(pool0, "G_ext"))),
                        sd=float(np.std(col(pool0, "G_ext"))))
    for q in (0.02, 0.05, 0.10, 0.20):
        p = FW.run({**gcfg, "gal_kind": "tensor", "q_amp": q},
                   N_F, 11_200_000 + int(10000 * q))
        gal[f"tensor_q{q:g}"] = rates(p, "G_ext", cg)
    for q in (0.05, 0.10, 0.20):
        for mis, fl in ((25.0, 0.0), (45.0, 0.0), (25.0, 0.5), (25.0, 1.0),
                        (90.0, 0.0)):
            p = FW.run({**gcfg, "gal_kind": "halo", "q_amp": q,
                        "gal_mis_deg": mis, "gal_f_lss": fl},
                       N_F, 11_300_000 + int(1e5 * q) + int(97 * mis) + int(7 * fl))
            gal[f"halo_q{q:g}_mis{mis:g}_flss{fl:g}"] = rates(p, "G_ext", cg)
    out["F6_galaxy"] = gal

    FW.close_pool()
    out["provenance"] = guard.stop()
    out["elapsed_s"] = time.time() - t0
    p = os.path.join(RES, "F_forward.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"wrote {p}  ({out['elapsed_s']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
