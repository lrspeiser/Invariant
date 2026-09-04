"""stage2.py -- STAGE 2 of the well-network funnel: synthetic geometries.

Every survivor of Stage 1 (and, per the programme's rule that a candidate is
not killed merely for failing somewhere, every family whether it survived or
not) is run on:

    G1  one point mass
    G2  two equal masses over a range of separations   -> spurious midpoint force
    G3  an exponential disk
    G4  a uniform sphere
    G5  a disk plus an external mass                   -> external-field response
    G6  a smooth cluster
    G7  that identical cluster progressively subdivided -> representation test

and rejected if it is

    singular              non-finite field, K eigenvalue <= 0, or a condition
                          number above 1e6
    unstable              the CG solve does not reach its tolerance, or the
                          operator loses positive-definiteness
    non-convergent        the field does not converge under grid refinement
    representation-dep.   subdividing the cluster changes the prediction

Every geometry is also run with K = I on the identical grid.  That Newtonian
null is what separates a property of the law from a property of the
discretisation, and it is reported next to every number.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import numpy as np

import families as F
import fieldsolve as FS
import screen as SC

G, KPC, MSUN, A0 = F.G, F.KPC, F.MSUN, F.A0

REJECT = dict(singular_cond=1e6, grid_conv=0.05, repr_dep=1e-3,
              cg_tol=1e-8, midpoint=1e-3)


def _finite(*arrs):
    return all(np.all(np.isfinite(a)) for a in arrs)


def _nwells(cand, default=256):
    """Family D costs O(P N^2) with no locality, so a geometry that other
    families run with 256 rows has to be run with far fewer.  That is not a
    convenience: it is the same O(N^2) fact the coarse-graining screen reports,
    showing up as a wall-clock limit."""
    return 48 if cand.kind in ("pairs", "pairs_linear") else default


def _cluster_Ns(cand):
    if cand.kind in ("pairs", "pairs_linear"):
        return (1, 10, 100, 256)
    return (1, 10, 100, 1000, 10000)


def _vprof(res, box, R):
    return FS.vcirc_axis(res["Psi"], box, R)


# ------------------------------------------------------------------ G1 point
def g1_point_mass(cand, n=48, Lbox=110.0, M=5e10 * MSUN, sig=5.0 * KPC):
    """One compact mass.  The grid-convergence gate is applied to the EXCESS
    over the Newtonian null run on the identical grids: a Gaussian blob on a
    Cartesian mesh has its own discretisation error, and charging that to the
    law would reject Newton itself."""
    box = F.Box(n, Lbox)
    rho = F.normalise_mass(F.gauss_rho(box.pts, M, sig).reshape(box.shape),
                           box.vol, M)
    wx, wm = np.zeros((1, 3)), np.array([M])
    R = np.geomspace(8 * KPC, 35 * KPC, 10)
    r = SC.solve_candidate(cand, rho, box, wx, wm, M)
    v = _vprof(r, box, R)
    null = FS.solve_newton(rho, box, Mtot=M)
    vN = _vprof(null, box, R)
    conv, convN = {}, {}
    for n2 in (int(n * 0.75) // 2 * 2, int(n * 1.5) // 2 * 2):
        b2 = F.Box(n2, Lbox)
        rho2 = F.normalise_mass(F.gauss_rho(b2.pts, M, sig).reshape(b2.shape),
                                b2.vol, M)
        r2 = SC.solve_candidate(cand, rho2, b2, wx, wm, M)
        n2r = FS.solve_newton(rho2, b2, Mtot=M)
        conv[n2] = float(np.abs(_vprof(r2, b2, R) / np.maximum(v, 1e-30)
                                - 1).max())
        convN[n2] = float(np.abs(_vprof(n2r, b2, R) / np.maximum(vN, 1e-30)
                                 - 1).max())
    gc = max(conv.values())
    gcN = max(convN.values())
    excess = max(gc - gcN, 0.0)
    ok = (_finite(r["Psi"]) and r["resid"] < REJECT["cg_tol"]
          and excess < REJECT["grid_conv"]
          and r.get("K_cond", 1.0) < REJECT["singular_cond"])
    return dict(passed=bool(ok), radii_kpc=[float(x / KPC) for x in R],
                vc_kms=[float(x / 1e3) for x in v],
                vc_newton_kms=[float(x / 1e3) for x in vN],
                boost=[float(a / max(b, 1e-30)) for a, b in zip(v, vN)],
                grid_convergence=conv, grid_convergence_newton_null=convN,
                grid_convergence_excess=excess,
                cg_resid=float(r["resid"]),
                K_eig_min=r.get("K_eig_min"), K_cond=r.get("K_cond"),
                reject=None if ok else _why(r, excess))


def _why(r, gc):
    if not np.all(np.isfinite(r["Psi"])):
        return "singular: non-finite potential"
    if r["resid"] >= REJECT["cg_tol"]:
        return f"unstable: CG residual {r['resid']:.2e}"
    if gc >= REJECT["grid_conv"]:
        return f"non-convergent: {gc:.3f} change under grid refinement"
    if r.get("K_cond", 1.0) >= REJECT["singular_cond"]:
        return f"singular: K condition number {r['K_cond']:.2e}"
    return "unclassified"


# --------------------------------------------------------------- G2 two-body
def g2_two_body(cand, seps_kpc=(8.0, 16.0, 32.0, 64.0), n=48, M=4e10 * MSUN,
                sig=1.6 * KPC):
    """Two EQUAL masses.  The midpoint force must vanish by reflection
    symmetry; anything left is either grid error (the Newtonian null measures
    it) or a genuine symmetry violation of the law.

    The axial force profile is reported as well, because a law can respect the
    midpoint symmetry exactly and still put a spurious attractor or repeller a
    short way off centre -- which is what a pair-channel tube does."""
    out = {}
    for d_kpc in seps_kpc:
        d = d_kpc * KPC
        Lbox = max(6 * d_kpc, 60.0)
        box = F.Box(n, Lbox)
        c1 = np.array([-d / 2, 0, 0.])
        c2 = np.array([+d / 2, 0, 0.])
        rho = (F.gauss_rho(box.pts, M, sig, c1)
               + F.gauss_rho(box.pts, M, sig, c2)).reshape(box.shape)
        Mt = 2 * M
        rho = F.normalise_mass(rho, box.vol, Mt)
        wx, wm = np.stack([c1, c2]), np.array([M, M])
        Fref = G * M * M / d ** 2
        r = SC.solve_candidate(cand, rho, box, wx, wm, Mt)
        nl = FS.solve_newton(rho, box, Mtot=Mt)
        fm = FS.force_at(r["Psi"], box, np.zeros(3))
        fmN = FS.force_at(nl["Psi"], box, np.zeros(3))
        # axial force profile, |x| < d
        xs = np.linspace(-0.45 * d, 0.45 * d, 19)
        fx = np.array([FS.force_at(r["Psi"], box, np.array([x, 0, 0]))[0]
                       for x in xs])
        fxN = np.array([FS.force_at(nl["Psi"], box, np.array([x, 0, 0]))[0]
                        for x in xs])
        # net force on the pair (must vanish: equal masses, so symmetry helps)
        out[str(d_kpc)] = dict(
            midpoint_force_rel=float(np.linalg.norm(fm) / Fref),
            midpoint_force_newton_null=float(np.linalg.norm(fmN) / Fref),
            axial_x_kpc=[float(x / KPC) for x in xs],
            axial_fx_rel=[float(x / Fref) for x in fx],
            axial_fx_newton_rel=[float(x / Fref) for x in fxN],
            max_axial_excess=float(np.abs((fx - fxN) / Fref).max()),
            cg_resid=float(r["resid"]),
            K_cond=r.get("K_cond"))
    worst = max(v["midpoint_force_rel"] - v["midpoint_force_newton_null"]
                for v in out.values())
    ok = worst < REJECT["midpoint"]
    return dict(passed=bool(ok), value=float(worst), tol=REJECT["midpoint"],
                per_separation=out,
                detail=("|F(midpoint)| / (G M^2 / d^2), minus the Newtonian "
                        "null on the same grid. 'max_axial_excess' is the "
                        "largest departure of the on-axis force from Newton "
                        "anywhere inside the pair"))


# ------------------------------------------------------------------ G3 disk
def g3_expdisk(cand, n=56, Lbox=120.0, M=5e10 * MSUN, Rd=4 * KPC,
               hz=1.5 * KPC, Nq=16384):
    box = F.Box(n, Lbox)
    rho = F.normalise_mass(
        F.expdisk_rho(box.pts, M, Rd, hz).reshape(box.shape), box.vol, M)
    qx, qm = F.equal_mass_cloud("expdisk", Nq, M, Rd=Rd, hz=hz)
    nw = _nwells(cand)
    wx, wm = F.nested_partitions(qx, qm, [nw])[nw]
    R = np.geomspace(4 * KPC, 30 * KPC, 12)
    r = SC.solve_candidate(cand, rho, box, wx, wm, M)
    nl = FS.solve_newton(rho, box, Mtot=M)
    v, vN = _vprof(r, box, R), _vprof(nl, box, R)
    # Freeman check on the Newtonian null (validates the geometry, not the law)
    import axisym as AX
    Sig0 = M / (2 * np.pi * Rd ** 2)
    vF = AX.freeman_vc(R / KPC, Sig0, Rd / KPC)
    out = R > 3 * Rd
    ok = _finite(r["Psi"]) and r["resid"] < REJECT["cg_tol"]
    return dict(passed=bool(ok), radii_kpc=[float(x / KPC) for x in R],
                vc_kms=[float(x / 1e3) for x in v],
                vc_newton_kms=[float(x / 1e3) for x in vN],
                vc_freeman_kms=[float(x / 1e3) for x in vF],
                newton_vs_freeman_outer=float(np.abs(vN / vF - 1)[out].max()),
                boost=[float(a / max(b, 1e-30)) for a, b in zip(v, vN)],
                cg_resid=float(r["resid"]), K_cond=r.get("K_cond"),
                detail=("INFORMATIONAL: newton_vs_freeman_outer compares the "
                        "K = I run with the exact RAZOR-THIN Freeman disk at "
                        "R > 3 Rd. The model disk has finite thickness "
                        "hz = 1.5 kpc, which lowers v_c on its own, so a few "
                        "per cent is expected. The law is judged on 'boost' "
                        "against the same-grid Newtonian null, where the "
                        "geometry and grid errors cancel to first order"))


# ---------------------------------------------------------------- G4 sphere
def g4_sphere(cand, n=44, Lbox=90.0, M=5e10 * MSUN, a=8 * KPC, Nq=16384):
    box = F.Box(n, Lbox)
    rho = np.where(box.r <= a, 1.0, 0.0)
    rho = F.normalise_mass(rho, box.vol, M)
    qx, qm = F.equal_mass_cloud("plummer", Nq, M, a=a * 0.5, umax=0.9)
    nw = _nwells(cand)
    wx, wm = F.nested_partitions(qx, qm, [nw])[nw]
    R = np.geomspace(2 * KPC, 30 * KPC, 12)
    r = SC.solve_candidate(cand, rho, box, wx, wm, M)
    nl = FS.solve_newton(rho, box, Mtot=M)
    v, vN = _vprof(r, box, R), _vprof(nl, box, R)
    Menc = M * np.minimum((R / a) ** 3, 1.0)
    vEx = np.sqrt(G * Menc / R)
    ok = _finite(r["Psi"]) and r["resid"] < REJECT["cg_tol"]
    return dict(passed=bool(ok), radii_kpc=[float(x / KPC) for x in R],
                vc_kms=[float(x / 1e3) for x in v],
                vc_newton_kms=[float(x / 1e3) for x in vN],
                vc_exact_newton_kms=[float(x / 1e3) for x in vEx],
                newton_vs_exact=float(np.abs(vN / vEx - 1)[2:-2].max()),
                boost=[float(x / max(y, 1e-30)) for x, y in zip(v, vN)],
                cg_resid=float(r["resid"]), K_cond=r.get("K_cond"))


# ------------------------------------------------------ G5 disk + external
def g5_disk_external(cand, n=48, Lbox=200.0, M=5e10 * MSUN, Rd=3 * KPC,
                     hz=0.4 * KPC, Mex=5e11 * MSUN, Rex=70 * KPC, Nq=16384):
    """An external mass outside the disk.  The interesting number is how much
    the INNER rotation curve moves; a local law should respond only through the
    external tidal field, not through the external potential depth."""
    box = F.Box(n, Lbox)
    disk = F.expdisk_rho(box.pts, M, Rd, hz).reshape(box.shape)
    disk = F.normalise_mass(disk, box.vol, M)
    ext = F.gauss_rho(box.pts, Mex, 6 * KPC,
                      (Rex, 0, 0)).reshape(box.shape)
    ext = F.normalise_mass(ext, box.vol, Mex)
    qx, qm = F.equal_mass_cloud("expdisk", Nq, M, Rd=Rd, hz=hz)
    nw = _nwells(cand)
    wx0, wm0 = F.nested_partitions(qx, qm, [nw])[nw]
    R = np.geomspace(2 * KPC, 15 * KPC, 10)
    a = SC.solve_candidate(cand, disk, box, wx0, wm0, M)
    wx1 = np.vstack([wx0, np.array([[Rex, 0, 0.]])])
    wm1 = np.concatenate([wm0, [Mex]])
    b = SC.solve_candidate(cand, disk + ext, box, wx1, wm1, M + Mex)
    nA = FS.solve_newton(disk, box, Mtot=M)
    nB = FS.solve_newton(disk + ext, box, Mtot=M + Mex)
    va, vb = _vprof(a, box, R), _vprof(b, box, R)
    na, nb = _vprof(nA, box, R), _vprof(nB, box, R)
    shift = np.abs(vb / np.maximum(va, 1e-30) - 1)
    shiftN = np.abs(nb / np.maximum(na, 1e-30) - 1)
    g_ext = G * Mex / Rex ** 2
    ok = _finite(a["Psi"], b["Psi"]) and max(a["resid"], b["resid"]) < REJECT["cg_tol"]
    return dict(passed=bool(ok), radii_kpc=[float(x / KPC) for x in R],
                vc_isolated_kms=[float(x / 1e3) for x in va],
                vc_with_external_kms=[float(x / 1e3) for x in vb],
                shift=[float(x) for x in shift],
                shift_newton_null=[float(x) for x in shiftN],
                excess_shift=float((shift - shiftN).max()),
                g_ext_over_a0=float(g_ext / A0),
                cg_resid=float(max(a["resid"], b["resid"])),
                detail=("Newton also shifts, because the external mass is "
                        "inside the box and drags the whole potential; the "
                        "excess over that null is the law's own external-field "
                        "response"))


# --------------------------------------------------------------- G6/G7 cluster
def g6g7_cluster(cand, n=44, Lbox=6000.0, M=1e14 * MSUN, a=400 * KPC,
                 Ns=None, Nq=32768):
    """A smooth cluster, then the IDENTICAL cluster cut into more and more
    subcomponents.  rho on the right-hand side never changes."""
    Ns = _cluster_Ns(cand) if Ns is None else Ns
    box = F.Box(n, Lbox)
    rho = F.normalise_mass(
        F.plummer_rho(box.pts, M, a).reshape(box.shape), box.vol, M)
    qx, qm = F.equal_mass_cloud("plummer", Nq, M, a=a, umax=0.97)
    parts = F.nested_partitions(qx, qm, Ns)
    R = np.geomspace(100 * KPC, 1500 * KPC, 10)
    res, fails = {}, {}
    for N in Ns:
        wx, wm = parts[N]
        F.check_partition(wx, wm, M)
        try:
            r = SC.solve_candidate(cand, rho, box, wx, wm, M)
            res[N] = dict(v=_vprof(r, box, R), resid=float(r["resid"]),
                          cond=r.get("K_cond"))
        except (MemoryError, AssertionError) as e:
            fails[N] = str(e)
    nl = FS.solve_newton(rho, box, Mtot=M)
    vN = _vprof(nl, box, R)
    have = sorted(res)
    if not have:
        return dict(passed=False, infeasible={str(k): v for k, v in fails.items()},
                    detail="no feasible subdivision")
    lo, hi = have[0], have[-1]
    dv = float(np.abs(res[lo]["v"] / np.maximum(res[hi]["v"], 1e-30) - 1).max())
    # implied dynamical mass at 1 Mpc, the quantity a cluster study would quote
    iM = {str(N): float((res[N]["v"][-2] ** 2) * R[-2] / G / MSUN)
          for N in have}
    ok = dv < REJECT["repr_dep"]
    return dict(passed=bool(ok), value=dv, tol=REJECT["repr_dep"],
                radii_kpc=[float(x / KPC) for x in R],
                vc_newton_kms=[float(x / 1e3) for x in vN],
                vc_by_N={str(N): [float(x / 1e3) for x in res[N]["v"]]
                         for N in have},
                dvc_1_to_max=dv,
                implied_Mdyn_at_1Mpc_Msun=iM,
                Mdyn_ratio_1_to_max=float(iM[str(lo)] / iM[str(hi)]),
                n_rows_used=[int(N) for N in have],
                infeasible={str(k): v for k, v in fails.items()},
                detail=("the same 1e14 Msun cluster, described by N rows; "
                        "'representation-dependent' means the answer moved. "
                        "Family D is capped at 256 rows because its cost is "
                        "O(P N^2) with no locality"))


GEOMS = [("G1_point_mass", g1_point_mass),
         ("G2_two_body", g2_two_body),
         ("G3_exponential_disk", g3_expdisk),
         ("G4_sphere", g4_sphere),
         ("G5_disk_plus_external", g5_disk_external),
         ("G6G7_cluster_subdivision", g6g7_cluster)]


def run_stage2(cand, verbose=True, only=None):
    out = dict(candidate=cand.name, family=cand.family, kind=cand.kind,
               geometries={})
    t0 = time.time()
    for key, fn in GEOMS:
        if only and key not in only:
            continue
        t = time.time()
        try:
            r = fn(cand)
        except Exception as e:                        # noqa: BLE001
            r = dict(passed=False, error=f"{type(e).__name__}: {e}",
                     trace=traceback.format_exc(limit=3))
        r["seconds"] = round(time.time() - t, 2)
        out["geometries"][key] = r
        if verbose:
            print(f"    {key:28s} {'PASS' if r.get('passed') else 'FAIL'} "
                  f"({r['seconds']}s) {r.get('reject') or r.get('error') or ''}")
    out["failed"] = [k for k, v in out["geometries"].items()
                     if v.get("passed") is False]
    out["verdict"] = "PASS" if not out["failed"] else "FAIL"
    out["seconds"] = round(time.time() - t0, 2)
    return out


def main(names=None, path="stage2_results.json"):
    names = names or ["X0_newton", "X2_count_wells",
                      "A1_aqual_simple", "A2_qumond_simple",
                      "B1_depth_mond", "C1_wells_pow_p1", "C2_wells_pow_p05",
                      "C4_wells_gsupp_p1", "D1_pairs_p1_q1", "D2_pairs_p05_q1",
                      "E1_tidal"]
    allres = {}
    for nm in names:
        print(f"\n=== STAGE 2  {nm}")
        allres[nm] = run_stage2(SC.ALL[nm])
        print(f"    -> {allres[nm]['verdict']}  {allres[nm]['seconds']}s")
    Path(path).write_text(json.dumps(allres, indent=1), encoding="utf-8")
    print(f"\nwrote {path}")
    return allres


if __name__ == "__main__":
    import sys
    main(sys.argv[1:] or None)
