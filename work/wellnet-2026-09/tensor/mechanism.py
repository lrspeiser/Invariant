"""The mechanism map: how big an effect can each tensor produce, as a
function of its globals, and does any point in the allowed space reach the
cluster amplitude without breaking galaxies.

WHAT IS BEING MAPPED
    B(r) = |g| with the tensor  /  |g| with K = I,  same source, same mu.
K = I is plain AQUAL/MOND, so B is exactly the extra factor the tensor supplies
on top of what MOND already gives.  The cluster gap this programme measured is
nu/nu_RAR = 2.5 for A2029 and a median 0.196 dex (1.57x) over 133 clusters, so
the target band is B = 1.6 to 2.5 with 2.0 as the headline.

THREE PROBES, because a model has to pass all three at once:
  cluster        shell-averaged B at 300, 500, 1000, 1414 kpc of the synthetic
                 A2029.  Required 1.6 - 2.5 and roughly flat, since the
                 measured X-COP excess varies by ~20% across that range.
  field galaxy   B at 10, 20, 30 kpc around an isolated 5e10 Msun disc with a
                 realistic field neighbourhood.  Required |log10 B| < 0.04
                 dex, the RAR's intrinsic scatter.
  member galaxy  B at 10, 20, 30 kpc around the most massive member near
                 500 kpc inside the same cluster.  Cluster early types lie on
                 the same fundamental plane as field ones, so a large B here
                 is excluded too -- and every potential-depth gate makes this
                 the WORST case, not the best, because a galaxy inside a
                 cluster sits at the bottom of the cluster's potential.

HOW B IS COMPUTED
    k(r) = < rhat^T K(x) rhat >  over a shell (cluster) or a sphere of
    directions (galaxies), then the exact spherical reduction
    mu(sqrt(k)|Phi'|/a0) k |Phi'| = G M(<r)/r^2.
Gate A5 measures this surrogate against the full nonlinear 3-D solve, and
verify_headline() re-runs the selected points in 3-D and also reports the
projected deflection, which is what lensing actually measures.

Two exact simplifications make the scan cheap enough to cover the globals:
  * k is only needed on the four shells, so the tensor is evaluated on those
    12k cells rather than all 262k.  No approximation.
  * A_0 factors out: exp(A_0 g I + A_T g S) = exp(A_0 g) exp(A_T g S).
"""
from __future__ import annotations

import gc
import json
import math
import time

import numpy as np

import channels as CH
import cluster as CL
import field as F
import run as RUN
import wellnet as W
from wellnet import G, A0, KPC, MSUN

XP = W.get_xp(True)

TARGET = 2.0
BAND = (1.6, 2.5)
GAL_TOL_DEX = 0.04
RADII = [300.0, 500.0, 1000.0, 1414.0]
GAL_RADII = [10.0, 20.0, 30.0]
RPROJ = [150.0, 300.0, 500.0, 800.0, 1100.0, 1400.0]
R500_KPC = 1414.0

# The programme's own measured requirement, lane 12: the cluster a0
# enhancement measured from lensing alone, on an r/R500 axis, reproduced by
# three samples sharing no clusters and no pipeline.
#     r/R500   a0/a0_canonical
#     0.073    21.95      CLASH fig2 (Umetsu+16 SL+WL+magnification)
#     0.291    13.30      CLASH fig2
#     0.698     5.90      CLASH x Umetsu+16 M1000c
#     1.000     3.09      mean of CLASH 3.67, XXL 2.71, X-COP x WL 2.88
#     1.505     1.19      mean of XXL M200 1.00, X-COP x WL M200 1.38
# PROVENANCE CAVEAT: these come from published lensing MASS profiles, which
# the programme's own rules allow only for debugging and comparison.  Nothing
# here is fitted to them; they are used solely as the shape a candidate would
# have to reproduce, and the headline B = 2 target is kept independently.
# In the deep-MOND regime g = sqrt(g_N a0), so an a0 enhancement A means a
# field boost B = sqrt(A); clusters sit at g_b = 0.07 - 0.18 a0, deep enough
# for that to hold to a few per cent.
LANE12_X = np.array([0.073, 0.291, 0.698, 1.000, 1.505])
LANE12_A = np.array([21.95, 13.30, 5.90, 3.09, 1.19])


def B_required(radii_kpc=RADII):
    x = np.asarray(radii_kpc) / R500_KPC
    logA = np.interp(np.log10(x), np.log10(LANE12_X), np.log10(LANE12_A))
    return np.sqrt(10.0 ** logA)


BREQ = B_required()


def _free():
    gc.collect()
    if XP is not np:
        XP.get_default_memory_pool().free_all_blocks()


# --------------------------------------------------------------- geometry
def fib_sphere(m):
    i = np.arange(m) + 0.5
    ph = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th),
                     np.cos(ph)], 1)


def contexts(n=64, Lbox=6000 * KPC, seed=20260903, ndir=1500):
    """cluster / field-galaxy / member-galaxy probes."""
    c = CL.build(n=n, Lbox=Lbox, seed=seed)
    pts = RUN.points_of(c, XP)
    R = XP.asarray(c["R"]).ravel()
    PhiN, gN = RUN.newton_analytic(pts, c, xp=XP)
    Rs = XP.maximum(R, 1e-30)
    rhat = XP.stack([XP.asarray(c["X"]).ravel() / Rs,
                     XP.asarray(c["Y"]).ravel() / Rs,
                     XP.asarray(c["Z"]).ravel() / Rs], axis=1)
    shells = [XP.abs(R - rk * KPC) < c["dx"] for rk in RADII]
    sel = shells[0].copy()
    for sh in shells[1:]:
        sel = sel | sh
    idx = XP.where(sel)[0]
    sub_sh = [XP.asarray(W.asnumpy(sh)[W.asnumpy(idx)]) for sh in shells]
    clu = dict(c=c, pts=pts, R=R, rhat=rhat, PhiN=PhiN, gN=gN, shells=shells,
               wx=XP.asarray(c["pos"]), wm=XP.asarray(c["Mg"]),
               r_prof=c["r_prof"], M_prof=c["M_prof"], radii=RADII,
               sub_idx=idx, sub_pts=pts[idx], sub_rhat=rhat[idx],
               sub_PhiN=PhiN[idx], sub_gN=gN[idx], sub_shells=sub_sh,
               n_shell_cells=[int(m.sum()) for m in sub_sh])

    dirs = fib_sphere(ndir)

    def sphere_probe(centre, wx, wm, Mgal, Rd, use_cluster_phi):
        P = np.concatenate([centre[None, :] + dirs * rk * KPC
                            for rk in GAL_RADII])
        rr = np.concatenate([np.full(ndir, rk * KPC) for rk in GAL_RADII])
        if use_cluster_phi:
            Ph, gg = RUN.newton_analytic(XP.asarray(P), c, xp=XP)
            Ph, gg = W.asnumpy(Ph), W.asnumpy(gg)
        else:
            d = P[:, None, :] - wx[None, :, :]
            dist = np.maximum(np.sqrt((d * d).sum(-1)), 1.0 * KPC)
            Ph = -(G * wm[None, :] / dist).sum(1)
            gg = np.sqrt((((-(G * wm[None, :, None] * d
                              / dist[:, :, None] ** 3)).sum(1)) ** 2).sum(-1))
        x = rr / Rd
        Menc = Mgal * (1.0 - (1.0 + x) * np.exp(-x))
        masks = [XP.asarray(np.arange(len(rr)) // ndir == j)
                 for j in range(len(GAL_RADII))]
        return dict(pts=XP.asarray(P), rhat=XP.asarray((P - centre)
                                                       / rr[:, None]),
                    r=XP.asarray(rr), Menc=XP.asarray(Menc),
                    PhiN=XP.asarray(Ph), gN=XP.asarray(gg),
                    wx=XP.asarray(wx), wm=XP.asarray(wm), masks=masks,
                    radii=GAL_RADII, ndir=ndir, Mgal=Mgal)

    gx, gm, Mgal, Rd = CL.field_galaxy()
    fld = sphere_probe(np.zeros(3), gx, gm, Mgal, Rd, False)

    rmem = np.sqrt((c["pos"] ** 2).sum(1))
    big = np.argsort(c["Mg"])[-10:]
    pick = big[np.argmin(np.abs(rmem[big] - 500 * KPC))]
    mem = sphere_probe(c["pos"][pick], c["pos"], c["Mg"],
                       float(c["Mg"][pick]), 5.0 * KPC, True)
    mem["r_from_centre_kpc"] = float(rmem[pick] / KPC)
    return clu, fld, mem


# --------------------------------------------------------- batched k(amp)
def k_means(base, rhat, amps, masks, chunk=24):
    """Shell means of k = rhat^T exp(amp * base) rhat, for every amplitude.

    base (P,6), rhat (P,3), amps (A,), masks list of (P,) boolean.
    Returns (k_harm, k_arith), each numpy (A, nmask).  Batched over
    amplitudes, so the whole scan is a handful of kernel launches.

    WHICH MEAN.  k varies by orders of magnitude across a shell once |A_T| is
    large, and the two obvious averages then disagree badly.  calib.py
    measures all three candidates against the full 3-D nonlinear solve at
    A_T = -1 to -6; the harmonic mean wins at every radius and amplitude, the
    arithmetic mean is worst and is not even monotonic in A_T (as A_T -> -inf
    the cells with S_rr < 0 blow up and drag <k> to infinity, so the boost it
    predicts turns over and saturates near 2.1 -- an artefact of the average,
    not of the physics).  The harmonic mean is what the map uses; both are
    stored so the bracket is visible.
    """
    A = len(amps)
    out_h = np.empty((A, len(masks)))
    out_a = np.empty((A, len(masks)))
    amps_d = XP.asarray(np.asarray(amps, float))
    for a0 in range(0, A, chunk):
        a1 = min(A, a0 + chunk)
        M = amps_d[a0:a1, None, None] * base[None, :, :]
        k = W.sym3_quad(W.sym3_expm(M, XP), rhat[None, :, :], XP)   # (a,P)
        ki = 1.0 / XP.maximum(k, 1e-300)
        for j, m in enumerate(masks):
            out_a[a0:a1, j] = W.asnumpy(k[:, m].mean(axis=1))
            out_h[a0:a1, j] = 1.0 / W.asnumpy(ki[:, m].mean(axis=1))
        del M, k, ki
    return out_h, out_a


def boost_from_k(r, Menc, k, mu):
    Fq = G * Menc / r ** 2
    return mu.invert(Fq, k, XP) / mu.invert(Fq, XP.ones_like(k), XP)


def _boost_vec(r_scalar, M_scalar, kcol, mu):
    r = XP.full(kcol.shape, r_scalar)
    M = XP.full(kcol.shape, M_scalar)
    return W.asnumpy(boost_from_k(r, M, kcol, mu))


def cluster_B(clu, kmat, mu):
    B = np.empty_like(kmat)
    for j, rk in enumerate(RADII):
        i = min(int(np.searchsorted(clu["r_prof"], rk * KPC)),
                len(clu["r_prof"]) - 1)
        B[:, j] = _boost_vec(float(clu["r_prof"][i]), float(clu["M_prof"][i]),
                             XP.asarray(kmat[:, j]), mu)
    return B


def sphere_B(g, kmat, mu):
    B = np.empty_like(kmat)
    for j in range(len(g["radii"])):
        sl = int(j * g["ndir"])
        B[:, j] = _boost_vec(float(g["r"][sl]), float(g["Menc"][sl]),
                             XP.asarray(kmat[:, j]), mu)
    return B


def crossing(amps, B, target):
    amps = np.asarray(amps, float)
    for i in range(len(amps) - 1):
        lo, hi = B[i], B[i + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            t = (target - lo) / (hi - lo)
            return float(amps[i] + t * (amps[i + 1] - amps[i])), i, float(t)
    return None, None, None


def assemble(rec, amps, kcl, Bcl, Bfl, Bmem, target=TARGET, kcl_alt=None):
    # clip: the pair-channel tensor has no normalisation, so at large alpha
    # exp() simply overflows.  1e6 is far outside anything observable and
    # keeps the JSON finite.
    Bcl = np.clip(np.nan_to_num(Bcl, nan=0.0, posinf=1e6), 0.0, 1e6)
    Bfl = np.clip(np.nan_to_num(Bfl, nan=1.0, posinf=1e6), 1e-6, 1e6)
    Bmem = np.clip(np.nan_to_num(Bmem, nan=1.0, posinf=1e6), 1e-6, 1e6)
    cr, i, t = crossing(amps, Bcl[:, 2], target)
    rec["B_max"] = float(Bcl[:, 2].max())
    rec["B_min"] = float(Bcl[:, 2].min())
    rec["dB_spread"] = float(Bcl[:, 2].max() - Bcl[:, 2].min())

    # separately: the amplitude that best reproduces the lane-12 radial
    # requirement, and how well it does
    rms = np.sqrt(np.mean((np.log10(np.maximum(Bcl, 1e-12))
                           - np.log10(BREQ)[None, :]) ** 2, axis=1))
    j = int(np.argmin(rms))
    rec["amp_for_shape"] = float(amps[j])
    rec["rms_dex_vs_lane12"] = float(rms[j])
    rec["B_cl_at_shape"] = list(Bcl[j])
    rec["field_dex_at_shape"] = float(np.max(np.abs(np.log10(Bfl[j]))))
    rec["member_dex_at_shape"] = float(np.max(np.abs(np.log10(Bmem[j]))))
    if cr is None:
        rec["reaches_target"] = False
        rec["feasible"] = False
        rec["feasible_incl_members"] = False
        return rec, None
    mix = lambda M: list((1 - t) * M[i] + t * M[i + 1])
    bc, bf, bm, kc = mix(Bcl), mix(Bfl), mix(Bmem), mix(kcl)
    if kcl_alt is not None:
        rec["k_cl_arith_at_target"] = mix(kcl_alt)
    fdex = float(np.max(np.abs(np.log10(bf))))
    mdex = float(np.max(np.abs(np.log10(bm))))
    rec.update(reaches_target=True, amp_for_target=cr, B_cl_at_target=bc,
               k_cl_at_target=kc, B_fld_at_target=bf, B_mem_at_target=bm,
               field_dex=fdex, member_dex=mdex, shape=float(bc[3] / bc[0]),
               feasible=bool(fdex < GAL_TOL_DEX),
               feasible_incl_members=bool(fdex < GAL_TOL_DEX
                                          and mdex < GAL_TOL_DEX))
    return rec, cr


# ------------------------------------------------------------------ main
GATES = [("none", dict(kind="none")),
         ("phi_1e11_m1", dict(kind="phi", Phi_0=1e11, m=1.0)),
         ("phi_1e12_m1", dict(kind="phi", Phi_0=1e12, m=1.0)),
         ("phi_1e12_m2", dict(kind="phi", Phi_0=1e12, m=2.0)),
         ("phi_1e12_m4", dict(kind="phi", Phi_0=1e12, m=4.0)),
         ("phi_3e12_m2", dict(kind="phi", Phi_0=3e12, m=2.0)),
         ("gn_m1", dict(kind="gn", m=1.0)),
         ("gn_m2", dict(kind="gn", m=2.0))]


def main(n=64, quick=False):
    t0 = time.time()
    mu = F.Mu("simple")
    clu, fld, mem = contexts(n=n)
    print(f"cluster grid {n}^3, {clu['pts'].shape[0]} cells, "
          f"{len(clu['wm'])} members; shell cells {clu['n_shell_cells']}")
    print(f"field galaxy  M = {fld['Mgal']/MSUN:.2e} Msun, |Phi_N| at 20 kpc "
          f"= {float(XP.abs(fld['PhiN'][fld['masks'][1]]).mean()):.3e}")
    print(f"member galaxy M = {mem['Mgal']/MSUN:.2e} Msun at "
          f"{mem['r_from_centre_kpc']:.0f} kpc, |Phi_N| at 20 kpc = "
          f"{float(XP.abs(mem['PhiN'][mem['masks'][1]]).mean()):.3e}")
    print(f"cluster |Phi_N| on the 1 Mpc shell = "
          f"{float(XP.abs(clu['sub_PhiN'][clu['sub_shells'][2]]).mean()):.3e}")

    amps_T = np.concatenate([-np.geomspace(80, 0.05, 70), [0.0],
                             np.geomspace(0.05, 80, 70)])
    fams, ps, qs, ss, Ls = (["plaw", "expo", "gscreen"], [0.0, 1.0],
                            [1.0, 2.0, 4.0], [0.5, 2.0],
                            [100.0, 300.0, 1000.0, 3000.0])
    excl, gates = [False, True], GATES
    if quick:
        fams, ps, qs, ss, Ls = ["plaw"], [1.0], [2.0], [1.5], [300.0]
        gates = GATES[:4]

    wn, nshape = [], 0
    for fam in fams:
        for p in ps:
            for q in qs:
                for s in (ss if fam != "expo" else [ss[0]]):
                    for Lk in Ls:
                        for ex in excl:
                            kw = dict(family=fam, p=p, q=q, s=s, m=1.0,
                                      L=Lk * KPC, M_0=1e11 * MSUN,
                                      exclude_nearest=ex)
                            Sc = W.S_tensor(clu["sub_pts"], clu["wx"],
                                            clu["wm"], xp=XP,
                                            gN_local=clu["sub_gN"], **kw)
                            Sf = W.S_tensor(fld["pts"], fld["wx"], fld["wm"],
                                            xp=XP, gN_local=fld["gN"], **kw)
                            Sm = W.S_tensor(mem["pts"], mem["wx"], mem["wm"],
                                            xp=XP, gN_local=mem["gN"], **kw)
                            nshape += 1
                            for gname, gkw in gates:
                                gc, gf, gm_ = (
                                    W.gate_field(gkw["kind"], PhiN=z[0],
                                                 gN=z[1],
                                                 Phi_0=gkw.get("Phi_0", 1e12),
                                                 m=gkw.get("m", 1.0), xp=XP)
                                    for z in ((clu["sub_PhiN"], clu["sub_gN"]),
                                              (fld["PhiN"], fld["gN"]),
                                              (mem["PhiN"], mem["gN"])))
                                bc = (gc[:, None] * Sc if not np.isscalar(gc)
                                      else gc * Sc)
                                bf = (gf[:, None] * Sf if not np.isscalar(gf)
                                      else gf * Sf)
                                bm = (gm_[:, None] * Sm if not np.isscalar(gm_)
                                      else gm_ * Sm)
                                kcl, kcl_a = k_means(bc, clu["sub_rhat"],
                                                     amps_T,
                                                     clu["sub_shells"])
                                kfl, _ = k_means(bf, fld["rhat"], amps_T,
                                                 fld["masks"])
                                kme, _ = k_means(bm, mem["rhat"], amps_T,
                                                 mem["masks"])
                                rec, cr = assemble(dict(
                                    tensor="wellnet", family=fam, p=p, q=q,
                                    s=s, L_kpc=Lk, exclude_nearest=ex,
                                    gate=gname, gate_par=gkw, A_0=0.0),
                                    amps_T, kcl, cluster_B(clu, kcl, mu),
                                    sphere_B(fld, kfl, mu),
                                    sphere_B(mem, kme, mu), kcl_alt=kcl_a)
                                if cr is not None:
                                    # smooth (Jensen) decomposition: the boost
                                    # the same S would give if it were replaced
                                    # by its shell mean, i.e. with the
                                    # lumpiness around individual members
                                    # averaged away before exponentiating
                                    Srr = W.sym3_quad(bc, clu["sub_rhat"], XP)
                                    ks = [float(XP.exp((cr * Srr)[sh].mean()))
                                          for sh in clu["sub_shells"]]
                                    rec["k_cl_smooth"] = ks
                                    rec["B_cl_smooth"] = [
                                        float(_boost_vec(
                                            float(clu["r_prof"][min(int(
                                                np.searchsorted(
                                                    clu["r_prof"],
                                                    RADII[j] * KPC)),
                                                len(clu["r_prof"]) - 1)]),
                                            float(clu["M_prof"][min(int(
                                                np.searchsorted(
                                                    clu["r_prof"],
                                                    RADII[j] * KPC)),
                                                len(clu["r_prof"]) - 1)]),
                                            XP.asarray(np.array([ks[j]])),
                                            mu)[0])
                                        for j in range(len(RADII))]
                                wn.append(rec)
                            del Sc, Sf, Sm
                            _free()
                        print(f"  wellnet {nshape:3d}  {fam:<8} p={p} q={q} "
                              f"s={s} L={Lk:6.0f}  "
                              f"Bmax={max(r['B_max'] for r in wn[-len(gates):]):9.3f}"
                              f"  feas="
                              f"{sum(1 for r in wn[-2*len(gates):] if r['feasible'])}"
                              f"  {time.time()-t0:6.0f}s")

    # ------------------------------------------------------------ channels
    amps_A = np.geomspace(1e-8, 3e-1, 110)
    ch, nsh = [], 0
    sperps, spars = [50.0, 150.0, 400.0], [150.0, 600.0]
    qs_c, ps_c, modes, Ls_c = [0.0, 1.0, 2.0], [0.0, 1.0], ["clip", "mid"], \
        [1000.0, 3000.0]
    if quick:
        sperps, spars, qs_c, ps_c, modes, Ls_c = ([150.0], [600.0], [1.0],
                                                  [1.0], ["clip"], [1000.0])
    for mode in modes:
        for sp in sperps:
            for sl in spars:
                for q in qs_c:
                    for p in ps_c:
                        for Lk in Ls_c:
                            pk = dict(p=p, q=q, s=2.0, L=Lk * KPC,
                                      M_0=1e11 * MSUN)
                            ck = dict(sigma_perp=sp * KPC, sigma_par=sl * KPC,
                                      mode=mode, n_sigma=6.0)
                            Cc = CH.C_tensor(clu["sub_pts"], CH.build_pairs(
                                clu["wx"], clu["wm"], xp=XP, **pk), xp=XP, **ck)
                            Cf = CH.C_tensor(fld["pts"], CH.build_pairs(
                                fld["wx"], fld["wm"], xp=XP, **pk), xp=XP, **ck)
                            Cm = CH.C_tensor(mem["pts"], CH.build_pairs(
                                mem["wx"], mem["wm"], xp=XP, **pk), xp=XP, **ck)
                            nsh += 1
                            tr = Cc[:, :3].sum(1)
                            z = XP.zeros_like(tr)
                            an = XP.sqrt(XP.sum(
                                (Cc - XP.stack([tr / 3, tr / 3, tr / 3,
                                                z, z, z], 1)) ** 2
                                * XP.asarray([1., 1., 1., 2., 2., 2.]), axis=1))
                            aoi = float(XP.median(an / XP.maximum(
                                XP.abs(tr) / math.sqrt(3.), 1e-300)))
                            for sign in (-1, +1):
                                kcl, kcl_a = k_means(sign * Cc,
                                                     clu["sub_rhat"], amps_A,
                                                     clu["sub_shells"])
                                kfl, _ = k_means(sign * Cf, fld["rhat"],
                                                 amps_A, fld["masks"])
                                kme, _ = k_means(sign * Cm, mem["rhat"],
                                                 amps_A, mem["masks"])
                                rec, cr = assemble(dict(
                                    tensor="channels", mode=mode,
                                    sigma_perp_kpc=sp, sigma_par_kpc=sl, q=q,
                                    p=p, s=2.0, L_kpc=Lk, sign=sign,
                                    trC_cluster_max=float(tr.max()),
                                    trC_field_max=float(Cf[:, :3].sum(1).max()),
                                    trC_member_max=float(Cm[:, :3].sum(1).max()),
                                    aniso_over_iso_median=aoi),
                                    amps_A, kcl, cluster_B(clu, kcl, mu),
                                    sphere_B(fld, kfl, mu),
                                    sphere_B(mem, kme, mu), kcl_alt=kcl_a)
                                if cr is not None:
                                    ki = XP.exp(sign * cr * tr / 3.0)
                                    kk = np.array([[float(ki[sh].mean())
                                                    for sh in
                                                    clu["sub_shells"]]])
                                    rec["B_cl_isotropic_part"] = list(
                                        cluster_B(clu, kk, mu)[0])
                                ch.append(rec)
                            del Cc, Cf, Cm
                            _free()
                            print(f"  channel {nsh:3d}  {mode} sp={sp:5.0f} "
                                  f"sl={sl:5.0f} q={q} p={p} L={Lk:5.0f}  "
                                  f"Bmax={max(r['B_max'] for r in ch[-2:]):10.3f}"
                                  f"  feas={sum(1 for r in ch[-2:] if r['feasible'])}"
                                  f"  {time.time()-t0:6.0f}s")

    fw = [r for r in wn if r.get("feasible")]
    fc = [r for r in ch if r.get("feasible")]
    out = dict(
        meta=dict(grid=n, Lbox_kpc=6000.0, n_members=int(len(clu["wm"])),
                  n_pairs=int(len(clu["wm"]) * (len(clu["wm"]) - 1) // 2),
                  radii_kpc=RADII, galaxy_radii_kpc=GAL_RADII, target=TARGET,
                  band=list(BAND), gal_tol_dex=GAL_TOL_DEX, mu="simple",
                  shell_cells=clu["n_shell_cells"],
                  member_r_kpc=mem["r_from_centre_kpc"],
                  member_mass_Msun=mem["Mgal"] / MSUN,
                  field_mass_Msun=fld["Mgal"] / MSUN,
                  PhiN_field_20kpc=float(XP.abs(
                      fld["PhiN"][fld["masks"][1]]).mean()),
                  PhiN_member_20kpc=float(XP.abs(
                      mem["PhiN"][mem["masks"][1]]).mean()),
                  PhiN_cluster_1Mpc=float(XP.abs(
                      clu["sub_PhiN"][clu["sub_shells"][2]]).mean()),
                  seconds=time.time() - t0,
                  definition="B = |g|(tensor)/|g|(K=I), same source, same mu"),
        lane12=dict(x_over_R500=list(LANE12_X), a0_enhancement=list(LANE12_A),
                    B_required_at_RADII=list(BREQ), R500_kpc=R500_KPC,
                    note="published lensing mass profiles; comparison target "
                         "only, nothing is fitted to them"),
        wellnet=wn, channels=ch,
        summary=dict(
            wellnet_rows=len(wn), channel_rows=len(ch),
            wellnet_reach_target=int(sum(1 for r in wn if r["reaches_target"])),
            channel_reach_target=int(sum(1 for r in ch if r["reaches_target"])),
            wellnet_feasible=len(fw), channel_feasible=len(fc),
            wellnet_feasible_incl_members=int(sum(
                1 for r in wn if r.get("feasible_incl_members"))),
            channel_feasible_incl_members=int(sum(
                1 for r in ch if r.get("feasible_incl_members"))),
            wellnet_B_max=max(r["B_max"] for r in wn),
            channel_B_max=max(r["B_max"] for r in ch),
            best_wellnet_field_dex=min(
                [r["field_dex"] for r in wn if r["reaches_target"]],
                default=None),
            best_channel_field_dex=min(
                [r["field_dex"] for r in ch if r["reaches_target"]],
                default=None),
            best_wellnet_member_dex=min(
                [r["member_dex"] for r in wn if r["reaches_target"]],
                default=None),
            best_channel_member_dex=min(
                [r["member_dex"] for r in ch if r["reaches_target"]],
                default=None),
            best_wellnet_rms_dex=min(r["rms_dex_vs_lane12"] for r in wn),
            best_channel_rms_dex=min(r["rms_dex_vs_lane12"] for r in ch),
            wellnet_shape_ok=int(sum(
                1 for r in wn if r["rms_dex_vs_lane12"] < 0.1
                and r["field_dex_at_shape"] < GAL_TOL_DEX
                and r["member_dex_at_shape"] < GAL_TOL_DEX)),
            channel_shape_ok=int(sum(
                1 for r in ch if r["rms_dex_vs_lane12"] < 0.1
                and r["field_dex_at_shape"] < GAL_TOL_DEX
                and r["member_dex_at_shape"] < GAL_TOL_DEX))))
    with open("mechanism_map.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwritten mechanism_map.json  {len(wn)} well-network rows, "
          f"{len(ch)} channel rows, {time.time()-t0:.0f} s")
    for nm, rr, ff in (("well-network", wn, fw), ("pair channels", ch, fc)):
        print(f"  {nm}: {sum(1 for r in rr if r['reaches_target'])} of "
              f"{len(rr)} reach B=2 at 1 Mpc; {len(ff)} of those keep the "
              f"FIELD galaxy inside {GAL_TOL_DEX} dex; "
              f"{sum(1 for r in rr if r.get('feasible_incl_members'))} also "
              f"keep the MEMBER galaxy inside it")
    return out


# ------------------------------------------------ robustness and 3-D checks
def resolution_check(shapes, gates, amps, ns=(32, 48, 64, 96, 128)):
    """Is the shell-averaged k(r) converged in grid resolution?

    The well-network boost is driven by cells with large |S|, and those sit
    close to individual members.  If the answer depends on how finely the
    members are resolved then it is not a prediction, it is a mesh artefact.
    Measured directly on k and B, with no PDE solve in the way.
    """
    mu = F.Mu("simple")
    out = []
    for kw, (gname, gkw), amp in zip(shapes, gates, amps):
        row = dict(shape=dict(kw, L_kpc=kw["L"] / KPC), gate=gname, A_T=amp,
                   n=[], k=[], B=[])
        row["shape"].pop("L", None)
        row["shape"].pop("M_0", None)
        print(f"    shape {row['shape']}  gate={gname} A_T={amp}")
        for n in ns:
            clu, _, _ = contexts(n=n)
            Sc = W.S_tensor(clu["sub_pts"], clu["wx"], clu["wm"], xp=XP,
                            gN_local=clu["sub_gN"], **kw)
            gc = W.gate_field(gkw["kind"], PhiN=clu["sub_PhiN"],
                              gN=clu["sub_gN"], Phi_0=gkw.get("Phi_0", 1e12),
                              m=gkw.get("m", 1.0), xp=XP)
            base = gc[:, None] * Sc if not np.isscalar(gc) else gc * Sc
            kmat, _ = k_means(base, clu["sub_rhat"], [amp],
                              clu["sub_shells"])
            B = cluster_B(clu, kmat, mu)
            row["n"].append(n)
            row["k"].append(list(kmat[0]))
            row["B"].append(list(B[0]))
            print(f"      n={n:4d}  k={' '.join(f'{v:.4f}' for v in kmat[0])}"
                  f"   B={' '.join(f'{v:.3f}' for v in B[0])}")
            del clu, Sc, base
            _free()
        out.append(row)
    return out


def verify_headline(cands, n=64):
    """Re-solve the selected points in full 3-D and compare with the map."""
    mu = F.Mu("simple")
    clu, fld, mem = contexts(n=n)
    c = clu["c"]
    KI = RUN.identity_K(c["rho"].shape, XP)
    Psi0, _ = RUN.solve_with_K(c, KI, mu, xp=XP, outer=60)
    g0, gv0 = F.gradient_mag(Psi0, c["dx"], XP)
    R = XP.asarray(c["R"])
    zsel = XP.abs(XP.asarray(c["ax"])) < 2000 * KPC
    R2 = XP.sqrt(XP.asarray(c["X"])[:, :, 0] ** 2
                 + XP.asarray(c["Y"])[:, :, 0] ** 2)
    alp0 = F.projected_deflection(gv0[0], gv0[1], c["dx"], zsel, XP)
    out = []
    for cd in cands:
        amp = cd["amp_for_target"]
        if cd["tensor"] == "wellnet":
            kw = dict(family=cd["family"], p=cd["p"], q=cd["q"], s=cd["s"],
                      m=1.0, L=cd["L_kpc"] * KPC, M_0=1e11 * MSUN,
                      exclude_nearest=cd["exclude_nearest"])
            Sc = W.S_tensor(clu["pts"], clu["wx"], clu["wm"], xp=XP,
                            gN_local=clu["gN"], **kw)
            gkw = cd["gate_par"]
            gc = W.gate_field(gkw["kind"], PhiN=clu["PhiN"], gN=clu["gN"],
                              Phi_0=gkw.get("Phi_0", 1e12),
                              m=gkw.get("m", 1.0), xp=XP)
            base = gc[:, None] * Sc if not np.isscalar(gc) else gc * Sc
            K = W.sym3_expm(amp * base, XP)
        else:
            pk = dict(p=cd["p"], q=cd["q"], s=cd["s"], L=cd["L_kpc"] * KPC,
                      M_0=1e11 * MSUN)
            ck = dict(sigma_perp=cd["sigma_perp_kpc"] * KPC,
                      sigma_par=cd["sigma_par_kpc"] * KPC, mode=cd["mode"],
                      n_sigma=6.0)
            Cc = CH.C_tensor(clu["pts"], CH.build_pairs(
                clu["wx"], clu["wm"], xp=XP, **pk), xp=XP, **ck)
            K = W.sym3_expm(cd["sign"] * amp * Cc, XP)
        Kf = XP.moveaxis(K.reshape(c["n"], c["n"], c["n"], 6), -1, 0)
        Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=40,
                                     tol_inner=1e-8, tol_outer=1e-5)
        gm, gv = F.gradient_mag(Psi, c["dx"], XP)
        b3 = [float(gm[XP.abs(R - rk * KPC) < c["dx"]].mean()
                    / g0[XP.abs(R - rk * KPC) < c["dx"]].mean())
              for rk in RADII]
        alp = F.projected_deflection(gv[0], gv[1], c["dx"], zsel, XP)
        bproj = [float(alp[XP.abs(R2 - rk * KPC) < c["dx"]].mean()
                       / alp0[XP.abs(R2 - rk * KPC) < c["dx"]].mean())
                 for rk in RPROJ]
        rec = dict(cand={k: v for k, v in cd.items()
                         if k in ("tensor", "family", "p", "q", "s", "L_kpc",
                                  "exclude_nearest", "gate", "mode", "sign",
                                  "sigma_perp_kpc", "sigma_par_kpc",
                                  "amp_for_target")},
                   B_3d=b3, B_map=cd["B_cl_at_target"],
                   ratio=[a / b for a, b in zip(b3, cd["B_cl_at_target"])],
                   R_proj_kpc=RPROJ, B_deflection=bproj,
                   outer=info["outer"], cg_iters=info["cg_iters"],
                   cg_residual=info["cg_rel"], dPhi_last=info["dPhi"])
        out.append(rec)
        print(f"    {cd['tensor']:<9} 3D B = "
              f"{' '.join(f'{v:.3f}' for v in b3)}   map B = "
              f"{' '.join(f'{v:.3f}' for v in cd['B_cl_at_target'])}   "
              f"[outer {info['outer']}, CG {info['cg_iters']}, "
              f"resid {info['cg_rel']:.1e}, dPhi {info['dPhi']:.1e}]")
        print(f"              projected deflection ratio = "
              f"{' '.join(f'{v:.3f}' for v in bproj)}")
        del K, Kf, Psi, gm, gv
        _free()
    return out


if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv
    out = main(n=64, quick=quick)

    print("\n--- resolution sensitivity of the well-network boost ---")
    shp = [dict(family="plaw", p=1.0, q=2.0, s=1.5, m=1.0, L=300 * KPC,
                M_0=1e11 * MSUN, exclude_nearest=False),
           dict(family="plaw", p=1.0, q=2.0, s=1.5, m=1.0, L=300 * KPC,
                M_0=1e11 * MSUN, exclude_nearest=True),
           dict(family="expo", p=1.0, q=2.0, s=0.5, m=1.0, L=1000 * KPC,
                M_0=1e11 * MSUN, exclude_nearest=False)]
    gts = [GATES[0], GATES[0], GATES[0]]
    amp = [-4.7, -4.7, -4.7]      # the amplitude that reaches B = 2 at 1 Mpc
    ns = (32, 48, 64) if quick else (32, 48, 64, 96, 128)
    res = resolution_check(shp, gts, amp, ns=ns)

    print("\n--- full 3-D verification of the selected points ---")
    cands = []
    for grp in ("wellnet", "channels"):
        rr = [r for r in out[grp] if r["reaches_target"]]
        rr.sort(key=lambda r: r["field_dex"])
        cands += rr[:2]
        rs = sorted(out[grp], key=lambda r: r["rms_dex_vs_lane12"])
        for r in rs[:2]:
            r = dict(r)
            r["amp_for_target"] = r["amp_for_shape"]
            r["B_cl_at_target"] = r["B_cl_at_shape"]
            r["selected_for"] = "lane12 shape"
            cands.append(r)
    ver = verify_headline(cands, n=64)

    out["resolution_check"] = res
    out["headline_3d"] = ver
    with open("mechanism_map.json", "w") as f:
        json.dump(out, f, indent=1)
    print("mechanism_map.json updated with resolution_check and headline_3d")
