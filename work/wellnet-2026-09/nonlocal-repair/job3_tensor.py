"""JOB 3(b) -- the nonlocal atom inside the tensor grammar.

        K = exp[ f_nl(qbar) I + f_T(qbar) That ]

qbar is the kernel's OWN path average, qbar(x) = Int_0^1 q((1-s)x + s x_c) ds,
which is what makes this atom nonlocal: a local q saturates the moment the
density drops below rho_ref, whereas the path average approaches its ceiling
only as 1 - r_ref/r.  That 1/r tail is the entire content of the atom.

WHY THIS IS THE RIGHT PLACE TO PUT IT.  The Stage-1 screen found families
C/D/E fail the bounded-response no-go: a bounded anisotropy can only rescale
G.  Family E is literally K = exp[f0 I + fT That] with CONSTANT f0, fT.  The
new atom differs in exactly two ways, and both are the point:
  * f_nl is a FUNCTION of a nonlocal scalar, and is allowed to be UNBOUNDED
    as qbar -> 1, which is the repair the boundedness theorem names;
  * f_T is a function of the same scalar, so the atom is DIRECTIONAL, and the
    radial and vertical responses are exp(-f_nl + 2 f_T/sqrt6) and
    exp(-f_nl - 2 f_T/sqrt6): f_T moves them in OPPOSITE directions.  That is
    the mechanism by which a directional version could keep the radial
    behaviour without the excessive vertical boost.

THE TWO GEOMETRIES, worked out once.
  Exterior monopole:  T = (GM/r^3) diag(-2,1,1) is already traceless, norm
  sqrt6 GM/r^3, so That = diag(-2,1,1)/sqrt6 and the RADIAL eigenvalue of K
  is exp(f_nl - 2 f_T/sqrt6).  Only that eigenvalue enters the radial flux,
  so in spherical symmetry k_r Psi' r^2 = G M(<r) integrates exactly and
        g(r) = G M(<r) / (k_r(r) r^2).
  Thin-slab midplane:  T ~ diag(0,0,4 pi G rho), traceless part
  4 pi G rho diag(-1,-1,2)/3, so That = diag(-1,-1,2)/sqrt6 and the VERTICAL
  eigenvalue is exp(f_nl + 2 f_T/sqrt6).  The vertical flux gives
  k_z g_z = 2 pi G Sigma, so Sigma_dyn/Sigma_bar = 1/k_z.
  The sign of the f_T term therefore FLIPS between the two, which is the
  decoupling knob.

ONE STRUCTURAL GAIN, FOR FREE.  k_r = exp(...) > 0 identically, so
g = G M/(k_r r^2) > 0 identically: the exponential tensor grammar CANNOT
produce a repulsive shell.  The scalar kernel produced them at 23-48% of
SPARC train points.  That failure mode is removed by construction.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
import time

import numpy as np

import common as C
import dcore as DC

SCREEN = ("C:/Users/henry/Documents/Codex/2026-08-21/"
          "Invariant-main-integration/work/wellnet-2026-09/screen")
if SCREEN not in sys.path:
    sys.path.append(SCREEN)

RES = {}
T0 = time.time()
S6 = math.sqrt(6.0)
AU = C.AU_KPC
R0_SUN = 8.2

#: Oort window and the solar anisotropy bound.  The anisotropy bound is the
#: binding solar-system constraint for a K theory, NOT the inverse-square
#: law: a CONSTANT K in the solar system merely renormalises G M_sun, which
#: is unobservable, but an anisotropic K makes the force non-central by a
#: fixed fraction |k_r/k_t - 1|, which planetary ephemerides bound.  1e-10 is
#: adopted as the primary; 1e-8 and 1e-6 are also counted, because turning
#: this into a rigorous ephemeris bound is a calculation this lane did not do.
OORT = (1.10, 1.70)
ANISO_BOUND = 1.0e-10


def say(*a):
    print(*a, flush=True)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


# ==========================================================================
#  The nonlocal scalar
# ==========================================================================
LADDER = [("dwarf_LSB", 1.5e9, 1.0), ("dwarf_HSB", 3.0e9, 1.2),
          ("LSB_large", 1.1e10, 6.0), ("spiral_mid", 2.0e10, 2.5),
          ("MW_like", 6.0e10, 3.0), ("massive", 2.1e11, 5.0)]


def galaxy_qbar(M, rd, rho_ref, qkind, nonlocal_=True, nr=1200):
    """(r, M(<r), qbar) for an exponential-sphere galaxy.

    `nonlocal_` False replaces the path average by the LOCAL q, which is the
    control that isolates what nonlocality buys.
    """
    rs = rd
    rho0 = M / (8.0 * math.pi * rs ** 3)
    r = np.geomspace(1e-8, 3.0e3, nr)
    x = r / rs
    rho = rho0 * np.exp(-x) + C.NK.RHO_BAR_B
    Menc = M * (1.0 - np.exp(-x) * (1.0 + x + 0.5 * x ** 2))
    if qkind == "delta":
        q = np.clip(rho_ref / rho - 1.0, 0.0, 1.0 - 1e-15)
    elif qkind == "smooth":
        q = 1.0 / (1.0 + rho / rho_ref)
    else:
        raise KeyError(qkind)
    if nonlocal_:
        cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1])
                                               * np.diff(r))])
        qb = np.clip(cum / r, 0.0, 1.0 - 1e-15)
    else:
        qb = q
    return r, Menc, qb


# ==========================================================================
#  f_nl and f_T, all vanishing at qbar = 0 so a fully screened region is
#  EXACTLY Newtonian and exactly isotropic.
# ==========================================================================
def f_nl(kind, qb, a, p):
    if kind == "poly":                       # bounded
        return -a * qb ** p
    if kind == "log":                        # unbounded
        return a * np.log(np.maximum(1.0 - qb, 1e-300))
    if kind == "ratio":                      # unbounded
        return -a * qb ** p / np.maximum(1.0 - qb, 1e-300)
    if kind == "expo":                       # bounded on [0,1)
        return -a * (np.exp(p * qb) - 1.0)
    raise KeyError(kind)


def f_T(kind, qb, c, m):
    if kind == "zero":
        return np.zeros_like(qb)
    if kind == "poly":
        return c * qb ** m
    if kind == "log":
        return -c * np.log(np.maximum(1.0 - qb, 1e-300))
    if kind == "peak":
        return c * qb ** m * (1.0 - qb)
    raise KeyError(kind)


FNL_UNBOUNDED = {"log", "ratio"}


#: exp() of the exponents is clipped at +/-40 (a factor 2e17 in K).  The
#: clip is a NUMERICAL guard, and it firing is itself a physical statement:
#: an unbounded f_nl that reaches -40 has driven k_r to 2e-18 and the force
#: to 2e17 times Newton, which is not a candidate.  `exponent_clipped` is
#: recorded for every setting so nothing is hidden behind the guard.
EXP_CLIP = 40.0


def kr_kt_kz(fn, ft):
    """Radial (exterior monopole), transverse, and vertical (slab) eigenvalues."""
    er = np.clip(fn - 2.0 * ft / S6, -EXP_CLIP, EXP_CLIP)
    et = np.clip(fn + ft / S6, -EXP_CLIP, EXP_CLIP)
    ez = np.clip(fn + 2.0 * ft / S6, -EXP_CLIP, EXP_CLIP)
    return np.exp(er), np.exp(et), np.exp(ez)


# ==========================================================================
def stage1():
    head("STAGE 1  Spherical semi-analytic screen of "
         "K = exp[f_nl(qbar) I + f_T(qbar) That]")
    say("In spherical symmetry the field equation integrates exactly, so the")
    say("screen is closed-form: g(r) = G M(<r) / (k_r(r) r^2) with")
    say("k_r = exp(f_nl - 2 f_T/sqrt6).  No solver is needed until Stage 2.")
    prof = {}
    for (name, M, rd), rr_, qk, nl in itertools.product(
            LADDER, (1e5, 1e6), ("delta", "smooth"), (True, False)):
        prof[(name, rr_, qk, nl)] = galaxy_qbar(M, rd, rr_, qk, nl)

    grid = list(itertools.product(
        ("poly", "log", "ratio", "expo"),          # f_nl form
        ("zero", "poly", "log", "peak"),           # f_T form
        (0.3, 1.0, 2.0, 3.0, 5.0),                 # a
        (0.5, 1.0, 2.0),                           # p
        (0.0, 0.3, 1.0, 3.0),                      # c
        (1.0, 2.0),                                # m
        (1e5, 1e6),                                # rho_ref
        ("delta", "smooth"),                       # q definition
        (True, False)))                            # nonlocal path average
    say(f"settings screened: {len(grid)}")

    rows = []
    for fnk, ftk, a, p, c, m, rr_, qk, nl in grid:
        if ftk == "zero" and c != 0.0:
            continue
        if ftk != "zero" and c == 0.0:
            continue
        sl, vf, ok = [], [], True
        for name, M, rd in LADDER:
            r, Menc, qb = prof[(name, rr_, qk, nl)]
            fn = f_nl(fnk, qb, a, p)
            ft = f_T(ftk, qb, c, m)
            kr, _, _ = kr_kt_kz(fn, ft)
            g = C.G * Menc / (kr * r ** 2)
            v2 = r * g
            sel = (r >= 2 * rd) & (r <= 20 * rd)
            lv = 0.5 * np.log(np.maximum(v2, 1e-300))
            sl.append(float(np.mean(np.gradient(lv, np.log(r))[sel])))
            vf.append(float(np.sqrt(np.interp(math.log(15 * rd), np.log(r),
                                              v2))))
        #  asymptotic slope of g, at 100-300 R_d of the MW-like model
        r, Menc, qb = prof[("MW_like", rr_, qk, nl)]
        fn = f_nl(fnk, qb, a, p); ft = f_T(ftk, qb, c, m)
        kr, kt, kz = kr_kt_kz(fn, ft)
        g = C.G * Menc / (kr * r ** 2)
        far = (r >= 300.0) & (r <= 3000.0)
        gslope = float(np.mean(np.gradient(np.log(g), np.log(r))[far]))
        #  solar neighbourhood: qbar at R0, then the slab eigenvalues
        qs = float(np.interp(math.log(R0_SUN), np.log(r), qb))
        fns = f_nl(fnk, np.array([qs]), a, p)[0]
        fts = f_T(ftk, np.array([qs]), c, m)[0]
        F_local = float(math.exp(-fns - 2.0 * fts / S6))
        aniso = float(abs(math.exp(-3.0 * fts / S6) - 1.0))
        Mb = np.array([mm for _, mm, _ in LADDER])
        vfa = np.array(vf)
        slope = float(np.polyfit(np.log10(vfa), np.log10(Mb), 1)[0]) \
            if np.all(vfa > 0) else float("nan")
        clipped = bool(np.max(np.abs(f_nl(fnk, prof[("MW_like", rr_, qk, nl)][2], a, p))) >= EXP_CLIP)
        rows.append(dict(
            exponent_clipped=clipped,
            f_nl=fnk, f_T=ftk, unbounded=fnk in FNL_UNBOUNDED, a=a, p=p, c=c,
            m=m, rho_ref=rr_, qdef=qk, nonlocal_qbar=nl,
            rms_outer_slope=float(np.sqrt(np.mean(np.array(sl) ** 2))),
            mean_outer_slope=float(np.mean(sl)),
            g_slope_far=gslope, btfr_slope=slope,
            qbar_sun=qs, F_local=F_local, solar_anisotropy=aniso,
            M1_flat=bool(np.sqrt(np.mean(np.array(sl) ** 2)) < 0.05),
            M2_asymptotic=bool(gslope > -1.30),
            M3_btfr=bool(np.isfinite(slope) and 3.5 <= slope <= 4.2),
            M4_oort=bool(OORT[0] <= F_local <= OORT[1]),
            M5_solar=bool(aniso < ANISO_BOUND)))
    RES["stage1"] = dict(n_settings=len(rows), rows=rows)
    say(f"evaluated: {len(rows)}")

    #  Newtonian control
    r, Menc, qb = prof[("MW_like", 1e6, "delta", True)]
    say("")
    keys = ["M1_flat", "M2_asymptotic", "M3_btfr", "M4_oort", "M5_solar"]
    say("   metric                                    passing / total")
    for k in keys:
        say(f"   {k:<40s} {sum(r_[k] for r_ in rows):6d} / {len(rows)}")
    say("")
    for k in range(1, len(keys) + 1):
        n_ok = sum(all(r_[q] for q in keys[:k]) for r_ in rows)
        say(f"   cumulative M1..M{k:<2d}                          "
            f"{n_ok:6d} / {len(rows)}")

    #  what unboundedness buys, isolated
    say("")
    for tag, sub in (("f_nl UNBOUNDED", [r_ for r_ in rows if r_["unbounded"]]),
                     ("f_nl bounded",
                      [r_ for r_ in rows if not r_["unbounded"]])):
        gs = np.array([r_["g_slope_far"] for r_ in sub])
        m1 = np.array([r_["rms_outer_slope"] for r_ in sub])
        say(f"   {tag:<16s}: asymptotic dln g/dln r  best "
            f"{gs.max():+.4f}  median {np.median(gs):+.4f}   "
            f"(Kepler -2, flat -1);  best outer rms slope {m1.min():.4f}")
    say("")
    for tag, sub in (("NONLOCAL qbar (path average)",
                      [r_ for r_ in rows if r_["nonlocal_qbar"]]),
                     ("LOCAL q (control)",
                      [r_ for r_ in rows if not r_["nonlocal_qbar"]])):
        gs = np.array([r_["g_slope_far"] for r_ in sub])
        m1 = np.array([r_["rms_outer_slope"] for r_ in sub])
        say(f"   {tag:<30s}: best asymptotic dln g/dln r {gs.max():+.4f}, "
            f"best outer rms slope {m1.min():.4f}")
    say("   Newtonian control: outer rms slope 0.190, asymptotic dln g/dln r "
        "= -2.")

    #  can f_T decouple radial from vertical?
    say("")
    say("   DOES THE ANISOTROPY DO INDEPENDENT WORK?  Compare the best "
        "achievable")
    say("   flatness at fixed Oort compliance, with f_T on and off.")
    dec = {}
    for tag, sub in (("f_T = 0 (isotropic)",
                      [r_ for r_ in rows if r_["f_T"] == "zero"]),
                     ("f_T != 0 (directional)",
                      [r_ for r_ in rows if r_["f_T"] != "zero"])):
        oo = [r_ for r_ in sub if r_["M4_oort"]]
        best = min(oo, key=lambda r_: r_["rms_outer_slope"]) if oo else None
        dec[tag] = dict(n_oort=len(oo),
                        best_rms_outer_slope=(best["rms_outer_slope"]
                                              if best else None),
                        best=best)
        say(f"   {tag:<24s}: {len(oo):5d} settings inside the Oort window; "
            f"best outer rms slope among them "
            + (f"{best['rms_outer_slope']:.4f}" if best else "none"))
    RES["stage1_decoupling"] = dec

    surv = [r_ for r_ in rows if all(r_[k] for k in keys)]
    RES["stage1"]["n_survivors"] = len(surv)
    say("")
    if surv:
        surv.sort(key=lambda r_: r_["rms_outer_slope"])
        say(f"   {len(surv)} settings pass all five.  Best ten:")
        for r_ in surv[:10]:
            say(f"      f_nl={r_['f_nl']:<6s} f_T={r_['f_T']:<5s} a={r_['a']:<4g}"
                f" p={r_['p']:<4g} c={r_['c']:<4g} m={r_['m']:<4g} "
                f"rho_ref={r_['rho_ref']:<7g} q={r_['qdef']:<7s} "
                f"nl={int(r_['nonlocal_qbar'])}  slope "
                f"{r_['rms_outer_slope']:.4f}  g_inf "
                f"{r_['g_slope_far']:+.3f}  BTFR {r_['btfr_slope']:.2f}  "
                f"F_loc {r_['F_local']:.3f}")
    else:
        say("   NO setting passes all five.  Where they die:")
        for k in range(1, len(keys) + 1):
            n_ok = sum(all(r_[q] for q in keys[:k]) for r_ in rows)
            if n_ok == 0:
                say(f"      the first empty cumulative cut is M1..M{k} "
                    f"({keys[k - 1]})")
                break
    #  best-effort survivors for Stage 2: pass M1 and M5, rank by BTFR gap
    cand = [r_ for r_ in rows if r_["M1_flat"] and r_["M5_solar"]]
    cand.sort(key=lambda r_: (abs(r_["btfr_slope"] - 3.85)
                              if np.isfinite(r_["btfr_slope"]) else 9e9))
    RES["stage1"]["stage2_candidates"] = cand[:6]
    say(f"\n   {len(cand)} settings are flat AND solar-safe; the six with the "
        f"best BTFR go to Stage 2.")
    for r_ in cand[:6]:
        say(f"      f_nl={r_['f_nl']:<6s} f_T={r_['f_T']:<5s} a={r_['a']:<4g} "
            f"p={r_['p']:<4g} c={r_['c']:<4g} m={r_['m']:<4g} "
            f"rho_ref={r_['rho_ref']:<7g} q={r_['qdef']:<7s} "
            f"nl={int(r_['nonlocal_qbar'])}  slope "
            f"{r_['rms_outer_slope']:.4f}  BTFR {r_['btfr_slope']:.2f}  "
            f"F_loc {r_['F_local']:.3f}  aniso {r_['solar_anisotropy']:.1e}")
    return rows, cand[:6]


# ==========================================================================
def stage3_sparc(cands):
    head("STAGE 3  The candidates on SPARC, in acceleration space, against "
         "the RAR")
    say("Same galaxies, same points, same nuisances and the same frozen "
        "split as")
    say("the honest comparison.  The kernel atom is applied through the "
        "spherical")
    say("field equation g = G M(<r) / (k_r r^2), which is EXACT for the "
        "equivalent")
    say("spherical baryon model.")
    train = C.sparc("train")
    blind = C.sparc("blind")
    out = []
    for cd in cands:
        row = dict(setting={k: cd[k] for k in
                            ("f_nl", "f_T", "a", "p", "c", "m", "rho_ref",
                             "qdef", "nonlocal_qbar")})
        for split, gals in (("train", train), ("blind", blind)):
            res, nneg, ntot = [], 0, 0
            for g in gals:
                prof = C.build_profile(g)
                r, rho, Mr, rfun, Mfun, Mtot = prof
                R, F_req, D_req, g_obs = C.required(g, Mtot)
                rho_f = rho + C.NK.RHO_BAR_B
                if cd["qdef"] == "delta":
                    q = np.clip(cd["rho_ref"] / rho_f - 1.0, 0.0, 1 - 1e-15)
                else:
                    q = 1.0 / (1.0 + rho_f / cd["rho_ref"])
                if cd["nonlocal_qbar"]:
                    cum = np.concatenate([[0.0], np.cumsum(
                        0.5 * (q[1:] + q[:-1]) * np.diff(r))])
                    qb = np.clip(cum / r, 0.0, 1 - 1e-15)
                else:
                    qb = q
                qbR = np.interp(np.log(R), np.log(r), qb)
                fn = f_nl(cd["f_nl"], qbR, cd["a"], cd["p"])
                ft = f_T(cd["f_T"], qbR, cd["c"], cd["m"])
                kr, _, _ = kr_kt_kz(fn, ft)
                gp = C.G * Mfun(R) / (kr * R ** 2)
                ntot += len(R)
                bad = gp <= 0
                nneg += int(bad.sum())
                if (~bad).any():
                    res.append(np.log10(gp[~bad] / g_obs[~bad]))
            allr = np.concatenate(res)
            per = np.array([np.mean(x) for x in res])
            row[split] = dict(rms_dex=float(np.sqrt(np.mean(allr ** 2))),
                              bias_dex=float(np.mean(allr)),
                              galaxy_scatter_dex=float(np.std(per)),
                              n_points=int(len(allr)),
                              n_nonpositive=nneg, n_total=ntot)
        out.append(row)
        s = row["setting"]
        say(f"   f_nl={s['f_nl']:<6s} f_T={s['f_T']:<5s} a={s['a']:<4g} "
            f"p={s['p']:<4g} c={s['c']:<4g} rho_ref={s['rho_ref']:<7g} "
            f"q={s['qdef']:<7s} nl={int(s['nonlocal_qbar'])} : "
            f"train {row['train']['rms_dex']:.3f} dex, blind "
            f"{row['blind']['rms_dex']:.3f} dex, repulsive points "
            f"{row['train']['n_nonpositive']}")
    RES["stage3_sparc"] = out
    say("")
    say("   For reference, from model_comparison.json on the SAME points:")
    say("      RAR            train 0.121  blind 0.122 dex")
    say("      AQUAL simple   train 0.121  blind 0.121 dex")
    say("      scalar kernel  train 0.256  blind 0.209 dex")
    say("      Newton         train 0.597  blind 0.588 dex")


# ==========================================================================
def stage2_full3d(cands, n=64, L=160.0):
    head("STAGE 2  Full 3-D anisotropic solves on a DISK")
    say("Two things are being checked: whether the spherical proxy's BOOST "
        "survives")
    say("real disk geometry, and whether f_T decouples the radial from the "
        "vertical")
    say("response.  Everything is reported as a BOOST relative to the "
        "Newtonian")
    say("solve on the SAME density, because the raw speeds differ between a "
        "disk")
    say("and the exponential sphere of the Stage-1 proxy by 30-40% from "
        "geometry")
    say("alone and that difference has nothing to do with K.")
    import families as FA
    import fieldsolve as FS
    from scipy.ndimage import map_coordinates
    KPC, MSUN = FA.KPC, FA.MSUN
    box = FA.Box(n, L)
    Mg, Rd, hz = 6.0e10, 3.0, 1.0
    rho = FA.expdisk_rho(box.pts, Mg * MSUN, Rd * KPC,
                         hz * KPC).reshape(box.shape)
    rho = FA.normalise_mass(rho, box.vol, Mg * MSUN)
    say(f"   grid {n}^3, box {L} kpc, h = {box.h / KPC:.2f} kpc; "
        f"exponential disk M = {Mg:.2g} Msun, R_d = {Rd} kpc, h_z = {hz} kpc")
    say("   NOTE the resolution limit, stated rather than hidden: h = "
        f"{box.h / KPC:.2f} kpc does")
    say("   NOT resolve the |z| = 1.1 kpc Oort column, so the vertical force "
        "is")
    say("   reported at z = 3 and 6 kpc and the Oort number stays with the "
        "slab")
    say("   proxy of Stage 1.")
    rN = FS.solve_newton(rho, box, tol=1e-10, maxiter=4000)
    say(f"   Newtonian reference: {rN['iters']} iters, resid "
        f"{rN['resid']:.2e}")
    That = FA.tidal_hat(rN["Psi"], box.h, dict(eps_T=1e-30))
    #  is That really diag(-2,1,1)/sqrt6 in the outskirts?
    rr = box.r.ravel()
    sel = (rr > 40 * KPC) & (rr < 70 * KPC)
    nh = (np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
          / np.maximum(rr, 1e-30)[:, None])
    trad = np.einsum("pi,pij,pj->p", nh[sel], That[sel], nh[sel])
    say(f"   orientation check: n.That.n over 40-70 kpc has mean "
        f"{trad.mean():+.4f}, sd {trad.std():.4f}")
    say(f"      (the exterior monopole value is -2/sqrt6 = "
        f"{-2 / S6:+.4f}; the Stage-1 proxy assumes it)")

    ns = 24
    sarr = (np.arange(ns) + 0.5) / ns
    P = np.stack([box.X.ravel(), box.Y.ravel(), box.Z.ravel()], 1)
    Rq = np.array([8.0, 16.0, 24.0, 32.0, 48.0]) * KPC
    vN = FS.vcirc_axis(rN["Psi"], box, Rq)
    gzN = [abs(FS.force_at(rN["Psi"], box, np.array(
        [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]

    out = dict(orientation=dict(n_That_n_mean=float(trad.mean()),
                                n_That_n_sd=float(trad.std()),
                                monopole_value=float(-2 / S6)),
               grid=dict(n=n, L_kpc=L, h_kpc=float(box.h / KPC),
                         M=Mg, Rd=Rd, hz=hz),
               runs=[])
    seen = set()
    for cd in cands:
        key = (cd["f_nl"], cd["f_T"], cd["a"], cd["p"], cd["c"], cd["m"],
               cd["rho_ref"], cd["qdef"], cd["nonlocal_qbar"])
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > 3:
            break
        rho_ref = cd["rho_ref"]
        rho_ms = rho.ravel() / MSUN * KPC ** 3 + C.NK.RHO_BAR_B
        if cd["qdef"] == "delta":
            q = np.clip(rho_ref / rho_ms - 1.0, 0.0, 1 - 1e-15)
        else:
            q = 1.0 / (1.0 + rho_ms / rho_ref)
        if cd["nonlocal_qbar"]:
            acc = np.zeros(P.shape[0])
            qg = q.reshape(box.shape)
            for sv in sarr:
                idx = ((P * (1.0 - sv) / box.h) + (n - 1) / 2.0).T
                acc += map_coordinates(qg, idx, order=1, mode="nearest")
            qb = acc / ns
            del acc, qg
        else:
            qb = q
        fn = f_nl(cd["f_nl"], qb, cd["a"], cd["p"])
        for ctag, cval in ((f"f_T={cd['f_T']} c={cd['c']:g}", cd["c"]),
                           ("f_T = 0 CONTROL", 0.0)):
            ft = (f_T(cd["f_T"], qb, cval, cd["m"]) if cval else
                  np.zeros_like(qb))
            eye = np.eye(3)[None]
            M = (np.clip(fn, -EXP_CLIP, EXP_CLIP)[:, None, None] * eye
                 + ft[:, None, None] * That)
            try:
                K = FA._sym_expm(M, "K = exp[f_nl I + f_T That]")
                rK = FS.solve_K(rho, K, box, tol=1e-10, maxiter=6000,
                                Mtot=float(rho.sum() * box.vol))
            except Exception as e:                       # noqa: BLE001
                say(f"   {ctag}: SOLVE FAILED {type(e).__name__}: {e}")
                continue
            vK = FS.vcirc_axis(rK["Psi"], box, Rq)
            gzK = [abs(FS.force_at(rK["Psi"], box, np.array(
                [R0_SUN * KPC, 0.0, z * KPC]))[2]) for z in (3.0, 6.0)]
            slK = float(np.mean(np.gradient(np.log(np.maximum(vK, 1e-30)),
                                            np.log(Rq))))
            slN = float(np.mean(np.gradient(np.log(np.maximum(vN, 1e-30)),
                                            np.log(Rq))))
            #  the Stage-1 spherical proxy's BOOST at the same radii
            rr_, Menc, qbs = galaxy_qbar(Mg, Rd, rho_ref, cd["qdef"],
                                         cd["nonlocal_qbar"])
            krr = kr_kt_kz(f_nl(cd["f_nl"], qbs, cd["a"], cd["p"]),
                           (f_T(cd["f_T"], qbs, cval, cd["m"]) if cval
                            else np.zeros_like(qbs)))[0]
            bproxy = np.interp(np.log(Rq / KPC), np.log(rr_), 1.0 / krr)
            b3d = (vK / vN) ** 2
            row = dict(setting={k: cd[k] for k in
                                ("f_nl", "f_T", "a", "p", "c", "m",
                                 "rho_ref", "qdef", "nonlocal_qbar")},
                       variant=ctag, c_used=float(cval),
                       R_kpc=(Rq / KPC).tolist(),
                       radial_boost_3d=b3d.tolist(),
                       radial_boost_proxy=bproxy.tolist(),
                       proxy_boost_rel_err=float(np.max(np.abs(
                           b3d / np.maximum(bproxy, 1e-30) - 1.0))),
                       vertical_boost_z3=float(gzK[0] / gzN[0]),
                       vertical_boost_z6=float(gzK[1] / gzN[1]),
                       outer_logslope_3d=slK,
                       outer_logslope_newton=slN,
                       shell_spread=float(rK["shell_spread"]),
                       iters=int(rK["iters"]), resid=float(rK["resid"]))
            out["runs"].append(row)
            s_ = row["setting"]
            say(f"   f_nl={s_['f_nl']} a={s_['a']:g} p={s_['p']:g} "
                f"q={s_['qdef']} nl={int(s_['nonlocal_qbar'])}  |  {ctag}")
            say("      radial boost 3-D   : " + " ".join(f"{x:6.3f}"
                                                         for x in b3d))
            say("      radial boost proxy : " + " ".join(f"{x:6.3f}"
                                                         for x in bproxy))
            say(f"      VERTICAL boost g_z/g_z,N: {row['vertical_boost_z3']:.3f}"
                f" at z=3 kpc, {row['vertical_boost_z6']:.3f} at z=6 kpc")
            say(f"      outer dlnv/dlnr {slK:+.3f} (Newton {slN:+.3f}); "
                f"proxy boost error {row['proxy_boost_rel_err'] * 100:.0f}%")
            del K, rK
        del qb, fn
    RES["stage2_full3d"] = out
    say("")
    say("   READ THE CONTROL ROWS.  If the directional run and its f_T = 0 "
        "control")
    say("   have the SAME ratio of vertical to radial boost, the anisotropy "
        "is only")
    say("   rescaling G and the Stage-1 no-go stands.  If they differ, f_T "
        "is doing")
    say("   independent work and the atom is a genuine new direction.")
    for r_ in out["runs"]:
        rb = float(np.mean(r_["radial_boost_3d"]))
        say(f"      {r_['variant']:<22s} radial {rb:6.3f}  vertical(z=3) "
            f"{r_['vertical_boost_z3']:6.3f}  ratio "
            f"{r_['vertical_boost_z3'] / rb:6.3f}")


# ==========================================================================
def main():
    rows, cands = stage1()
    if cands:
        stage3_sparc(cands)
    say("\nStage 2 (full 3-D solves) runs separately: job3_stage2.py")
    RES["runtime_s"] = time.time() - T0
    with open("tensor_atom_screen.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    say(f"\nwrote tensor_atom_screen.json  ({time.time() - T0:.1f} s)")


if __name__ == "__main__":
    main()
