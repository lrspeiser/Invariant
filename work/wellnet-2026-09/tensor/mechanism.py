"""The mechanism map: how big an effect can each tensor produce, as a
function of its globals, and does any point in the allowed space reach the
cluster amplitude without breaking galaxies.

WHAT IS BEING MAPPED
    B(r) = |g| with the tensor  /  |g| with K = I,  same source, same mu.
K = I is plain AQUAL/MOND, so B is exactly the extra factor the tensor supplies
on top of what MOND already gives.  The cluster gap this programme measured is
nu/nu_RAR = 2.5 for A2029 and a median 0.196 dex (1.57x) over 133 clusters, so
the target band is B = 1.6 to 2.5 with 2.0 as the headline.

THREE PROBES, because the model has to pass all three at once:
  cluster        shell-averaged B at 300, 500, 1000, 1414 kpc of the synthetic
                 A2029; required 1.6 - 2.5, ideally flat, since the measured
                 excess is flat to ~20% across X-COP's radial range.
  field galaxy   B at 10, 20, 30 kpc around an isolated 5e10 Msun disc with a
                 realistic field neighbourhood; required |log10 B| < 0.04 dex,
                 the RAR's intrinsic scatter.
  member galaxy  B at 10, 20, 30 kpc around a 300 kpc-scale member sitting at
                 ~500 kpc inside the cluster.  Cluster ellipticals lie on the
                 same fundamental plane as field ones, so a large B here is
                 also excluded; it is reported because every potential-depth
                 gate makes this the WORST case, not the best.

HOW B IS COMPUTED
    k(r) = < rhat^T K(x) rhat >  over a shell (cluster) or a sphere of
    directions (galaxies), then the exact spherical reduction
    mu(sqrt(k)|Phi'|/a0) k |Phi'| = G M(<r)/r^2.
Gate A5 measures this surrogate against the full nonlinear 3-D solve, and
verify_headline() re-runs the selected points in 3-D.

A_0 factors out exactly: exp(A_0 g I + A_T g S) = exp(A_0 g) exp(A_T g S), so
the scan is over A_T with A_0 as an explicit multiplicative offset.
"""
from __future__ import annotations

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


# --------------------------------------------------------------- geometry
def fib_sphere(m):
    i = np.arange(m) + 0.5
    ph = np.arccos(1 - 2 * i / m)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.stack([np.sin(ph) * np.cos(th), np.sin(ph) * np.sin(th),
                     np.cos(ph)], 1)


def _probe(points, centre, Menc, wx, wm, PhiN, gN, radii, ndir):
    rel = points - centre
    rr = np.sqrt((rel ** 2).sum(1))
    return dict(pts=XP.asarray(points), rhat=XP.asarray(rel / rr[:, None]),
                r=XP.asarray(rr), Menc=XP.asarray(Menc),
                PhiN=XP.asarray(PhiN), gN=XP.asarray(gN),
                wx=XP.asarray(wx), wm=XP.asarray(wm),
                radii=list(radii), ndir=ndir)


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
    # The amplitude scan only ever needs k on the four shells, so the tensor
    # is evaluated on those cells alone: 1.2e4 cells instead of 2.6e5, a 20x
    # saving with no approximation whatsoever.  The full grid is kept for the
    # 3-D verification.
    sel = shells[0].copy()
    for sh in shells[1:]:
        sel = sel | sh
    idx = XP.where(sel)[0]
    sub_shell = [XP.asarray(W.asnumpy(sh)[W.asnumpy(idx)]) for sh in shells]
    clu = dict(c=c, pts=pts, R=R, rhat=rhat, PhiN=PhiN, gN=gN, shells=shells,
               wx=XP.asarray(c["pos"]), wm=XP.asarray(c["Mg"]),
               r_prof=c["r_prof"], M_prof=c["M_prof"], ndir=None,
               radii=RADII, sub_idx=idx, sub_pts=pts[idx],
               sub_rhat=rhat[idx], sub_PhiN=PhiN[idx], sub_gN=gN[idx],
               sub_shells=sub_shell)

    dirs = fib_sphere(ndir)
    # ---- field galaxy
    gx, gm, Mgal, Rd = CL.field_galaxy()
    P, rr = [], []
    for rk in GAL_RADII:
        P.append(dirs * rk * KPC)
        rr.append(np.full(ndir, rk * KPC))
    P = np.concatenate(P)
    rr = np.concatenate(rr)
    d = P[:, None, :] - gx[None, :, :]
    dist = np.maximum(np.sqrt((d * d).sum(-1)), 1.0 * KPC)
    Phi = -(G * gm[None, :] / dist).sum(1)
    gv = -(G * gm[None, :, None] * d / dist[:, :, None] ** 3).sum(1)
    x = rr / Rd
    Menc = Mgal * (1.0 - (1.0 + x) * np.exp(-x))
    fld = _probe(P, np.zeros(3), Menc, gx, gm, Phi,
                 np.sqrt((gv ** 2).sum(-1)), GAL_RADII, ndir)
    fld["Mgal"] = Mgal

    # ---- member galaxy inside the cluster, nearest to 500 kpc among the
    #      ten most massive members
    rmem = np.sqrt((c["pos"] ** 2).sum(1))
    big = np.argsort(c["Mg"])[-10:]
    pick = big[np.argmin(np.abs(rmem[big] - 500 * KPC))]
    centre = c["pos"][pick]
    Mmem = float(c["Mg"][pick])
    Rd_m = 5.0 * KPC
    P2 = np.concatenate([centre[None, :] + dirs * rk * KPC
                         for rk in GAL_RADII])
    rr2 = np.concatenate([np.full(ndir, rk * KPC) for rk in GAL_RADII])
    Phi2, g2 = RUN.newton_analytic(XP.asarray(P2), c, xp=XP)
    x = rr2 / Rd_m
    Menc2 = Mmem * (1.0 - (1.0 + x) * np.exp(-x))
    mem = _probe(P2, centre, Menc2, c["pos"], c["Mg"], W.asnumpy(Phi2),
                 W.asnumpy(g2), GAL_RADII, ndir)
    mem["Mgal"] = Mmem
    mem["r_from_centre_kpc"] = float(np.sqrt((centre ** 2).sum()) / KPC)
    return clu, fld, mem


# ------------------------------------------------------------- the boost
def boost_from_k(r, Menc, k, mu):
    Fq = G * Menc / r ** 2
    return mu.invert(Fq, k, XP) / mu.invert(Fq, XP.ones_like(k), XP)


def boost_at(ctx, rk, kbar, mu):
    i = min(int(np.searchsorted(ctx["r_prof"], rk * KPC)),
            len(ctx["r_prof"]) - 1)
    return float(boost_from_k(XP.asarray(np.array([ctx["r_prof"][i]])),
                              XP.asarray(np.array([ctx["M_prof"][i]])),
                              XP.asarray(np.array([kbar])), mu)[0])


def cluster_boost(ctx, k_cells, mu, sub=True):
    out = []
    masks = ctx["sub_shells"] if sub else ctx["shells"]
    for j, rk in enumerate(RADII):
        kbar = float(k_cells[masks[j]].mean())
        out.append((kbar, boost_at(ctx, rk, kbar, mu)))
    return out


def sphere_boost(g, k_cells, mu):
    out = []
    nd = g["ndir"]
    for j in range(len(g["radii"])):
        sl = slice(j * nd, (j + 1) * nd)
        kbar = float(k_cells[sl].mean())
        out.append((kbar, float(boost_from_k(
            g["r"][sl][:1], g["Menc"][sl][:1],
            XP.asarray(np.array([kbar])), mu)[0])))
    return out


def krad_wellnet(S, rhat, gate, A_T, A_0=0.0):
    M = (A_T * gate)[:, None] * S if not np.isscalar(gate) else A_T * gate * S
    k = W.sym3_quad(W.sym3_expm(M, XP), rhat, XP)
    if np.isscalar(gate):
        return k * math.exp(A_0 * gate)
    return k * XP.exp(A_0 * gate)


def krad_channels(C, rhat, alpha, sign, isotropic_only=False):
    if isotropic_only:
        tr = C[:, :3].sum(1) / 3.0
        return XP.exp(sign * alpha * tr)
    return W.sym3_quad(W.sym3_expm(sign * alpha * C, XP), rhat, XP)


def scan(kfun, amps, clu, fld, mem, mu, target=TARGET):
    rows = []
    for a in amps:
        cb = cluster_boost(clu, kfun(a, "cluster"), mu)
        fb = sphere_boost(fld, kfun(a, "field"), mu)
        mb = sphere_boost(mem, kfun(a, "member"), mu)
        rows.append(dict(amp=float(a), k_cl=[x[0] for x in cb],
                         B_cl=[x[1] for x in cb],
                         B_fld=[x[1] for x in fb],
                         B_mem=[x[1] for x in mb]))
    B = np.array([r["B_cl"][2] for r in rows])
    amps = np.asarray(amps, float)
    cross = None
    for i in range(len(amps) - 1):
        lo, hi = B[i], B[i + 1]
        if (lo - target) * (hi - target) <= 0 and lo != hi:
            t = (target - lo) / (hi - lo)
            cross = float(amps[i] + t * (amps[i + 1] - amps[i]))
            break
    return rows, cross, float(B.max()), float(B.min())


def interp_row(rows, amp):
    a = np.array([r["amp"] for r in rows])
    j = int(np.clip(np.searchsorted(a, amp) - 1, 0, len(a) - 2))
    t = (amp - a[j]) / (a[j + 1] - a[j])

    def mix(key):
        return [(1 - t) * u + t * v
                for u, v in zip(rows[j][key], rows[j + 1][key])]
    return dict(amp=float(amp), B_cl=mix("B_cl"), k_cl=mix("k_cl"),
                B_fld=mix("B_fld"), B_mem=mix("B_mem"))


def finish(rec, rows, cross, bmax, bmin):
    rec.update(B_max=bmax, B_min=bmin)
    if cross is None:
        rec["feasible"] = False
        rec["reaches_target"] = False
    else:
        at = interp_row(rows, cross)
        fdex = float(np.max(np.abs(np.log10(at["B_fld"]))))
        mdex = float(np.max(np.abs(np.log10(at["B_mem"]))))
        rec.update(reaches_target=True, B_cl_at_target=at["B_cl"],
                   k_cl_at_target=at["k_cl"], B_fld_at_target=at["B_fld"],
                   B_mem_at_target=at["B_mem"], field_dex=fdex,
                   member_dex=mdex,
                   shape=float(at["B_cl"][3] / at["B_cl"][0]),
                   feasible=bool(fdex < GAL_TOL_DEX),
                   feasible_incl_members=bool(fdex < GAL_TOL_DEX
                                              and mdex < GAL_TOL_DEX))
    B = np.array([r["B_cl"][2] for r in rows])
    rec["dB_spread"] = float(B.max() - B.min())
    return rec


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
          f"{len(clu['wm'])} members")
    print(f"field galaxy  M = {fld['Mgal']/MSUN:.2e} Msun, "
          f"|Phi_N| at 20 kpc = {float(XP.abs(fld['PhiN'][1500:3000]).mean()):.3e}")
    print(f"member galaxy M = {mem['Mgal']/MSUN:.2e} Msun at "
          f"{mem['r_from_centre_kpc']:.0f} kpc, |Phi_N| at 20 kpc = "
          f"{float(XP.abs(mem['PhiN'][1500:3000]).mean()):.3e}")
    icl = clu["shells"][2]
    print(f"cluster |Phi_N| on the 1 Mpc shell = "
          f"{float(XP.abs(clu['PhiN'][icl]).mean()):.3e}")

    amps_T = np.concatenate([-np.geomspace(80, 0.05, 70), [0.0],
                             np.geomspace(0.05, 80, 70)])
    fams, ps, qs, ss, Ls = (["plaw", "expo", "gscreen"], [0.0, 1.0],
                            [1.0, 2.0, 4.0], [0.5, 2.0],
                            [100.0, 300.0, 1000.0, 3000.0])
    excl = [False, True]
    gates = GATES
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

                                def kf(a, which, Sc=Sc, Sf=Sf, Sm=Sm, gc=gc,
                                       gf=gf, gm_=gm_):
                                    if which == "cluster":
                                        return krad_wellnet(Sc,
                                                            clu["sub_rhat"],
                                                            gc, a)
                                    if which == "field":
                                        return krad_wellnet(Sf, fld["rhat"],
                                                            gf, a)
                                    return krad_wellnet(Sm, mem["rhat"], gm_, a)

                                rows, cr, bx, bn = scan(kf, amps_T, clu, fld,
                                                        mem, mu)
                                rec = finish(dict(
                                    tensor="wellnet", family=fam, p=p, q=q,
                                    s=s, L_kpc=Lk, exclude_nearest=ex,
                                    gate=gname, gate_par=gkw, A_0=0.0,
                                    A_T_for_target=cr), rows, cr, bx, bn)
                                if cr is not None:
                                    # smooth (Jensen) decomposition: the boost
                                    # S would give if it were replaced by its
                                    # shell mean, i.e. with the lumpiness
                                    # around individual members removed
                                    Srr = W.sym3_quad(Sc, clu["sub_rhat"],
                                                      XP)
                                    e = cr * gc * Srr
                                    ksm = [float(XP.exp(e[sh].mean()))
                                           for sh in clu["sub_shells"]]
                                    rec["k_cl_smooth"] = ksm
                                    rec["B_cl_smooth"] = [
                                        boost_at(clu, RADII[j], ksm[j], mu)
                                        for j in range(len(RADII))]
                                wn.append(rec)
                            print(f"  wellnet {nshape:3d}  {fam:<8} p={p} q={q}"
                                  f" s={s} L={Lk:6.0f} excl={int(ex)}  "
                                  f"Bmax={max(r['B_max'] for r in wn[-len(gates):]):8.3f}"
                                  f"  feas={sum(1 for r in wn[-len(gates):] if r['feasible'])}"
                                  f"  {time.time()-t0:6.0f}s")

    # ------------------------------------------------------------ channels
    amps_A = np.geomspace(1e-8, 3e-1, 110)
    ch, nsh = [], 0
    sperps, spars = [50.0, 150.0, 400.0], [150.0, 600.0]
    qs_c, ps_c, modes = [0.0, 1.0, 2.0], [0.0, 1.0], ["clip", "mid"]
    Ls_c = [1000.0, 3000.0]
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
                            aniso = XP.sqrt(XP.sum(
                                (Cc - XP.stack([tr / 3, tr / 3, tr / 3,
                                                z, z, z], 1)) ** 2
                                * XP.asarray([1., 1., 1., 2., 2., 2.]), axis=1))
                            aoi = float(XP.median(
                                aniso / XP.maximum(XP.abs(tr) / math.sqrt(3.),
                                                   1e-300)))
                            for sign in (-1, +1):
                                def kf(a, which, Cc=Cc, Cf=Cf, Cm=Cm,
                                       sign=sign):
                                    z = {"cluster": (Cc, clu["sub_rhat"]),
                                         "field": (Cf, fld["rhat"]),
                                         "member": (Cm, mem["rhat"])}[which]
                                    return krad_channels(z[0], z[1], a, sign)

                                rows, cr, bx, bn = scan(kf, amps_A, clu, fld,
                                                        mem, mu)
                                rec = finish(dict(
                                    tensor="channels", mode=mode,
                                    sigma_perp_kpc=sp, sigma_par_kpc=sl,
                                    q=q, p=p, s=2.0, L_kpc=Lk, sign=sign,
                                    alpha_for_target=cr,
                                    trC_cluster_max=float(tr.max()),
                                    trC_field_max=float(Cf[:, :3].sum(1).max()),
                                    trC_member_max=float(Cm[:, :3].sum(1).max()),
                                    aniso_over_iso_median=aoi),
                                    rows, cr, bx, bn)
                                # how much of the boost is the trace alone
                                if cr is not None:
                                    kiso = krad_channels(Cc, clu["sub_rhat"],
                                                         cr, sign,
                                                         isotropic_only=True)
                                    rec["B_cl_isotropic_part"] = [
                                        b for _, b in
                                        cluster_boost(clu, kiso, mu)]
                                ch.append(rec)
                            print(f"  channel {nsh:3d}  {mode} sp={sp:5.0f} "
                                  f"sl={sl:5.0f} q={q} p={p} L={Lk:5.0f}  "
                                  f"Bmax={max(r['B_max'] for r in ch[-2:]):9.3f}"
                                  f"  feas={sum(1 for r in ch[-2:] if r['feasible'])}"
                                  f"  {time.time()-t0:6.0f}s")

    fw = [r for r in wn if r.get("feasible")]
    fc = [r for r in ch if r.get("feasible")]
    out = dict(
        meta=dict(grid=n, Lbox_kpc=6000.0, n_members=int(len(clu["wm"])),
                  radii_kpc=RADII, galaxy_radii_kpc=GAL_RADII, target=TARGET,
                  band=list(BAND), gal_tol_dex=GAL_TOL_DEX, mu="simple",
                  member_r_kpc=mem["r_from_centre_kpc"],
                  member_mass_Msun=mem["Mgal"] / MSUN,
                  field_mass_Msun=fld["Mgal"] / MSUN,
                  seconds=time.time() - t0,
                  definition="B = |g|(tensor)/|g|(K=I), same source, same mu"),
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
                default=None)))
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
    This measures it directly on k and B, with no PDE solve in the way.
    """
    mu = F.Mu("simple")
    out = []
    for kw, gname, gkw, amp in zip(shapes, [g[0] for g in gates],
                                   [g[1] for g in gates], amps):
        row = dict(shape=dict(kw, L_kpc=kw["L"] / KPC), gate=gname, A_T=amp,
                   n=[], k=[], B=[])
        for n in ns:
            clu, _, _ = contexts(n=n)
            Sc = W.S_tensor(clu["sub_pts"], clu["wx"], clu["wm"], xp=XP,
                            gN_local=clu["sub_gN"], **kw)
            gc = W.gate_field(gkw["kind"], PhiN=clu["sub_PhiN"],
                              gN=clu["sub_gN"],
                              Phi_0=gkw.get("Phi_0", 1e12),
                              m=gkw.get("m", 1.0), xp=XP)
            k = krad_wellnet(Sc, clu["sub_rhat"], gc, amp)
            cb = cluster_boost(clu, k, mu)
            row["n"].append(n)
            row["k"].append([c[0] for c in cb])
            row["B"].append([c[1] for c in cb])
            print(f"    res {n:4d}  k={' '.join(f'{c[0]:.4f}' for c in cb)}"
                  f"   B={' '.join(f'{c[1]:.3f}' for c in cb)}")
            del clu, Sc, k
            _free()
        out.append(row)
    return out


def _free():
    import gc
    gc.collect()
    if XP is not np:
        XP.get_default_memory_pool().free_all_blocks()


def verify_headline(cands, n=64):
    """Re-solve the selected points in full 3-D and compare with the map."""
    mu = F.Mu("simple")
    clu, fld, mem = contexts(n=n)
    c = clu["c"]
    KI = RUN.identity_K(c["rho"].shape, XP)
    Psi0, _ = RUN.solve_with_K(c, KI, mu, xp=XP, outer=60)
    g0, gv0 = F.gradient_mag(Psi0, c["dx"], XP)
    R = XP.asarray(c["R"])
    ax = XP.asarray(c["ax"])
    zsel = XP.abs(ax) < 2000 * KPC
    R2 = XP.sqrt(XP.asarray(c["X"])[:, :, 0] ** 2
                 + XP.asarray(c["Y"])[:, :, 0] ** 2)
    alp0 = F.projected_deflection(gv0[0], gv0[1], c["dx"], zsel, XP)
    RPROJ = [150.0, 300.0, 500.0, 800.0, 1100.0, 1400.0]
    out = []
    for cd in cands:
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
            amp = cd["A_T_for_target"]
            M = (amp * gc)[:, None] * Sc if not np.isscalar(gc) else amp * gc * Sc
            K = W.sym3_expm(M, XP)
        else:
            pk = dict(p=cd["p"], q=cd["q"], s=cd["s"], L=cd["L_kpc"] * KPC,
                      M_0=1e11 * MSUN)
            ck = dict(sigma_perp=cd["sigma_perp_kpc"] * KPC,
                      sigma_par=cd["sigma_par_kpc"] * KPC, mode=cd["mode"],
                      n_sigma=6.0)
            Cc = CH.C_tensor(clu["pts"], CH.build_pairs(
                clu["wx"], clu["wm"], xp=XP, **pk), xp=XP, **ck)
            K = W.sym3_expm(cd["sign"] * cd["alpha_for_target"] * Cc, XP)
        Kf = XP.moveaxis(K.reshape(c["n"], c["n"], c["n"], 6), -1, 0)
        Psi, info = RUN.solve_with_K(c, Kf, mu, xp=XP, outer=80)
        gm, gv = F.gradient_mag(Psi, c["dx"], XP)
        b3 = []
        for rk in RADII:
            sel = XP.abs(R - rk * KPC) < c["dx"]
            b3.append(float(gm[sel].mean() / g0[sel].mean()))
        alp = F.projected_deflection(gv[0], gv[1], c["dx"], zsel, XP)
        bproj = []
        for rk in RPROJ:
            sel = XP.abs(R2 - rk * KPC) < c["dx"]
            bproj.append(float(alp[sel].mean() / alp0[sel].mean()))
        rec = dict(cand={k: v for k, v in cd.items()
                         if k in ("tensor", "family", "p", "q", "s", "L_kpc",
                                  "exclude_nearest", "gate", "mode", "sign",
                                  "sigma_perp_kpc", "sigma_par_kpc",
                                  "A_T_for_target", "alpha_for_target")},
                   B_3d=b3, B_map=cd["B_cl_at_target"],
                   ratio=[a / b for a, b in zip(b3, cd["B_cl_at_target"])],
                   R_proj_kpc=RPROJ, B_deflection=bproj,
                   outer=info["outer"])
        out.append(rec)
        print(f"    {cd['tensor']:<9} 3D B = "
              f"{' '.join(f'{v:.3f}' for v in b3)}   map B = "
              f"{' '.join(f'{v:.3f}' for v in cd['B_cl_at_target'])}")
        del K, Kf, Psi
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
    amp = [-12.8, -12.8, -12.8]
    ns = (32, 48, 64) if quick else (32, 48, 64, 96, 128)
    res = resolution_check(shp, gts, amp, ns=ns)

    print("\n--- full 3-D verification of the selected points ---")
    cands = []
    for grp in ("wellnet", "channels"):
        rr = [r for r in out[grp] if r["reaches_target"]]
        rr.sort(key=lambda r: r["field_dex"])
        cands += rr[:3]
    ver = verify_headline(cands, n=64)

    out["resolution_check"] = res
    out["headline_3d"] = ver
    with open("mechanism_map.json", "w") as f:
        json.dump(out, f, indent=1)
    print("mechanism_map.json updated with resolution_check and headline_3d")
