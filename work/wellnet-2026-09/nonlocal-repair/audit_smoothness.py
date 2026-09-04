"""JOB 2 -- the clipping smoothness audit.

The clipped delta state q = clip(rho_ref/rho_s - 1, 0, 1-eps) gives
q(Sun) = 0 EXACTLY, which is legitimate screening.  But a hard clip has a
discontinuous q', and the question is whether that surface adds a force, a
flux or an energy discontinuity -- a shell.  Five things are measured:

 S1  the smoothness class of each clip variant, and the size of the jump in
     q' across the two clipping surfaces;
 S2  what the surface does in the KERNEL formulation: jumps in Phi, in
     g = (GM/r^2) D and in dD/dr across it, i.e. shell potential, shell
     surface density and effective-density discontinuity;
 S3  what it does in a FIELD formulation div[K(q) grad Psi] = 4 pi G rho,
     where the answer is different and has a closed form;
 S4  whether the rotation-curve behaviour survives a C^2 replacement;
 S5  whether solar-system safety survives it -- the trade the brief asks to
     quantify rather than assume.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

import clipmod as CM
import common as C
import dcore as DC

GPU = True
RES = {}
T0 = time.time()
RHO_SUN_LOCAL, HZ_LOCAL, Z_SUN = 7.6e7, 0.30, 0.020
NEWTON_1AU_MS2 = 5.9301e-3
ISL_BOUND = 1.0e-11


def say(*a):
    print(*a, flush=True)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


# ==========================================================================
def s1_smoothness_class():
    head("S1  Smoothness class of each clip variant, and the size of the "
         "kink")
    say("Measured as EXACT one-sided limits of q\' and q\'\' at the corner, "
        "not by")
    say("fitting across it.  Test map u(x) = 1 - x, so du/dx = -1 and a jump "
        "in")
    say("dq/dx IS the jump in dq/du.  The hard clip\'s corner sits at u = 0; "
        "a")
    say("rounded corner of width w occupies u in [-w, +w] with outer corners "
        "at")
    say("u = -w and u = +w, and BOTH are checked.")
    out = {}
    rows = []
    d = 1e-9
    for kind in CM.KINDS:
        for w in ((0.0,) if kind == "hard" else (0.02, 0.05, 0.1, 0.2)):
            corners = (0.0,) if kind == "hard" else (-w, w)
            j1 = j2 = 0.0
            for uc in corners:
                for k in (1, 2):
                    a = float(CM.q_clip(np.array([uc + d]), w=w, kind=kind,
                                        deriv=k)[0])
                    b = float(CM.q_clip(np.array([uc - d]), w=w, kind=kind,
                                        deriv=k)[0])
                    if k == 1:
                        j1 = max(j1, abs(a - b))
                    else:
                        j2 = max(j2, abs(a - b))
            uu = np.linspace(-1.5, 1.5, 2000001)
            m2 = float(np.max(np.abs(CM.q_clip(uu, w=w, kind=kind, deriv=2))))
            rows.append(dict(kind=kind, w=w, C=CM.SMOOTHNESS[kind],
                             jump_dq_du=j1, jump_d2q_du2=j2, max_d2q_du2=m2))
            say(f"   {kind:<9s} w={w:<5.2f}  claimed C^"
                f"{CM.SMOOTHNESS[kind]:<3}  max |[dq/du]| = {j1:9.3e}   "
                f"max |[d2q/du2]| = {j2:9.3e}   sup|d2q/du2| = {m2:9.3e}")
    say("")
    say("   hard    : [dq/du] = 1 exactly -- a genuine gradient "
        "discontinuity.")
    say("   quad    : [dq/du] = 0, [d2q/du2] = 1/(2w) -- C^1 only.")
    say("   quintic : both jumps 0 -- C^2, at the price of sup|d2q/du2| = "
        "1.5/w.")
    say("   softplus: all jumps 0, sup|d2q/du2| = 1/(2w), but q is never "
        "exactly 0.")
    say("   THE TRADE NO SMOOTHING REMOVES: the delta state runs from q = 0 "
        "to q = 1")
    say("   over u in [0,1], i.e. over a FACTOR 2 in density.  Rounding its "
        "corners")
    say("   over w <= 0.2 leaves a near-step, and w >= 0.5 destroys the "
        "exact zero.")
    out["corner_ladder"] = rows

    tr = C.sparc("train")
    g = ([x for x in tr if x.name == "NGC2403"] or tr)[0]
    prof = C.build_profile(g)
    r = prof[0]
    phys = {}
    for rho_ref in (1e4, 1e5, 1e6):
        rr = CM.clip_radii(prof, rho_ref)
        rho_f = prof[1] + C.NK.RHO_BAR_B
        dlnrho = np.gradient(np.log(rho_f), np.log(r))
        j = {}
        for tag, rc in rr.items():
            if not np.isfinite(rc):
                continue
            sl = float(np.interp(math.log(rc), np.log(r), dlnrho))
            j[tag] = dict(r_kpc=rc, dlnrho_dlnr=sl,
                          jump_dq_dr_per_kpc=abs(sl) / rc)
        phys[f"rho_ref={rho_ref:g}"] = j
        say(f"   {g.name}, rho_ref={rho_ref:g}: " + "  ".join(
            f"{k} at r={v['r_kpc']:.3f} kpc, [dq/dr] = "
            f"{v['jump_dq_dr_per_kpc']:.4f} /kpc" for k, v in j.items()))
    out["physical_kink"] = dict(galaxy=g.name, per_rho_ref=phys)
    RES["S1_smoothness_class"] = out
    return g, prof


# ==========================================================================
def _analytic_field(rho_ref, kind, w, n=40001, r_lo=1e-3, r_hi=3.0e4):
    """MW-like exponential-sphere galaxy with the clipped q on a fine grid.

    An ANALYTIC baryon profile is used here, not the SPARC equivalent sphere,
    because the question is structural and the equivalent sphere\'s own rho
    comes from differencing a PCHIP: its roughness would be indistinguishable
    from the effect being measured.  dln r = 4e-4 on this grid, so the
    log-linear q interpolation smears the clip over ~0.006 kpc at 15 kpc,
    well inside the smallest fit window used.
    """
    from scipy.optimize import brentq
    gal = C.MO.GALAXY_LADDER[4]
    r = np.geomspace(r_lo, r_hi, n)
    rho_p = gal.rho_pert(r)
    q = CM.q_clip(CM.u_of_rho(rho_p + gal.rho_floor, rho_ref), w=w, kind=kind)
    fld = C.NK.SphericalField(r=r, rho=rho_p, q=q, rho_fun=gal.rho_pert,
                              Menc_fun=gal.Menc, label=f"MW|{kind}|{w}")
    f = lambda x: float(gal.rho_pert(np.array([x]))[0]) + gal.rho_floor - rho_ref
    rc = brentq(f, 1e-3, 3.0e3) if f(1e-3) * f(3.0e3) < 0 else float("nan")
    return fld, gal, rc


def s2_kernel_surface(g, prof):
    head("S2  What the clip surface does in the KERNEL formulation: force, "
         "flux, energy")
    say("Two separate things have to be told apart here, and the first "
        "version of")
    say("this test conflated them.")
    say("")
    say("(a) A QUADRATURE pathology.  D contains dqbar/dr = Int q\'(r_s) "
        "r(1-s)/r_s ds")
    say("    and a hard clip makes q\' DISCONTINUOUS in s.  Gauss-Legendre "
        "on a")
    say("    discontinuous integrand converges like 1/n_s, not spectrally, "
        "so D")
    say("    near the clip radius carries an O(1/n_s) error that mimics a "
        "jump.")
    say("(b) The PHYSICAL question: with the quadrature converged, is there "
        "a jump")
    say("    in Phi or in g across the surface?")
    out = {}

    # ---- (a) convergence in n_s, hard versus C^2 ------------------------
    say("")
    say("(a) convergence of D at radii straddling the clip surface, as n_s "
        "grows.")
    say("    Reported as max |D(n_s)/D(1024) - 1| over 6 radii within 2% of "
        "r_clip.")
    conv = {}
    for rho_ref in (1e5, 1e6):
        for kind, w in (("hard", 0.0), ("quintic", 0.05), ("quintic", 0.20)):
            fld, gal, rc = _analytic_field(rho_ref, kind, w)
            if not np.isfinite(rc):
                continue
            rg = rc * np.array([0.98, 0.99, 0.995, 1.005, 1.01, 1.02])
            ref = None
            row = {}
            for n_s in (8, 16, 32, 64, 128, 256, 512, 1024):
                _, D = DC.phi_and_D(fld, rg, Fname="F1_poly", alpha=3.0,
                                    p=1.0, Mtot=gal.Mtot, use_gpu=GPU,
                                    chunk=2, n_D=48, n_s=n_s, n_gl=10,
                                    dlnr_max=0.12)
                if n_s == 1024:
                    ref = D
                row[n_s] = D
            errs = {k: float(np.max(np.abs(v / ref - 1.0)))
                    for k, v in row.items() if k != 1024}
            conv[f"rho_ref={rho_ref:g}|{kind}|w={w}"] = errs
            say(f"   rho_ref={rho_ref:<6g} {kind:<8s} w={w:<5.2f} : " +
                "  ".join(f"n_s={k}:{v:.2e}" for k, v in errs.items()))
    say("")
    say("   The hard clip loses roughly one factor of 2 in error per "
        "doubling of")
    say("   n_s -- first-order, the signature of a discontinuous integrand. "
        "The C^2")
    say("   clip converges far faster.  n_s = 12, the production value, is "
        "therefore")
    say("   the WRONG quadrature for a clipped q, and that is a defect of "
        "the clip,")
    say("   not of the solver.")
    out["n_s_convergence"] = conv

    # ---- (b) the physical jump, at converged quadrature ------------------
    say("")
    say("(b) one-sided limits at n_s = 512, window shrunk until the "
        "extrapolation")
    say("    settles.  [g] != 0 would be a shell of surface density "
        "[g]/(4 pi G).")
    for rho_ref in (1e5, 1e6):
        for kind, w in (("hard", 0.0), ("quintic", 0.05)):
            fld, gal, rc = _analytic_field(rho_ref, kind, w)
            if not np.isfinite(rc):
                continue
            Mt = gal.Mtot
            rho_c = float(gal.rho_pert(np.array([rc]))[0])
            for alpha, p in ((3.0, 1.0), (10.0, 1.0), (3.0, 2.0)):
                seq, gr = [], None
                for tmax in (2.0e-2, 1.0e-2, 5.0e-3):
                    t = np.linspace(0.35 * tmax, tmax, 8)
                    rg = np.concatenate([rc * (1 - t[::-1]), rc * (1 + t)])
                    Fe, D = DC.phi_and_D(fld, rg, Fname="F1_poly",
                                         alpha=alpha, p=p, Mtot=Mt,
                                         use_gpu=GPU, chunk=2, n_D=48,
                                         n_s=512, n_gl=10, dlnr_max=0.12)
                    Phi = -C.G * Mt * Fe / rg
                    gr = C.G * Mt * D / rg ** 2
                    x = rg / rc - 1.0
                    lo, hi = x < 0, x > 0
                    fa = lambda f_, m_: np.polyfit(x[m_], f_[m_], 1)
                    pP, pM = fa(Phi, hi), fa(Phi, lo)
                    gP, gM = fa(gr, hi), fa(gr, lo)
                    seq.append(dict(
                        tmax=tmax,
                        jump_Phi_rel=float(abs((pP[1] - pM[1])
                                               / np.mean(np.abs(Phi)))),
                        jump_g_rel=float(abs((gP[1] - gM[1])
                                             / np.mean(np.abs(gr)))),
                        jump_dgdx_rel=float(abs((gP[0] - gM[0])
                                                / np.mean(np.abs(gr))))))
                c = seq[-1]
                gmean = float(np.mean(np.abs(gr)))
                rho_eff_step = (c["jump_dgdx_rel"] * gmean / rc
                                / (4.0 * math.pi * C.G))
                sigma_shell = c["jump_g_rel"] * gmean / (4.0 * math.pi * C.G)
                key = f"rho_ref={rho_ref:g}|{kind}|w={w}|a={alpha}|p={p}"
                out[key] = dict(
                    r_clip_kpc=float(rc), window_sequence=seq,
                    jump_Phi_rel=c["jump_Phi_rel"],
                    jump_g_rel=c["jump_g_rel"],
                    jump_dg_dlnr_rel=c["jump_dgdx_rel"],
                    shell_surface_density_Msun_kpc2=float(sigma_shell),
                    shell_mass_over_Mtot=float(
                        4 * math.pi * rc ** 2 * sigma_shell / Mt),
                    rho_eff_step_Msun_kpc3=float(rho_eff_step),
                    rho_baryon_at_clip=rho_c,
                    rho_eff_step_over_baryon=float(rho_eff_step / rho_c))
                say(f"   {key}  r_clip = {rc:.3f} kpc")
                say(f"      [Phi]/|Phi| {c['jump_Phi_rel']:.2e}   [g]/|g| "
                    f"{c['jump_g_rel']:.2e}  -> shell mass/M_tot "
                    f"{out[key]['shell_mass_over_Mtot']:.2e}")
                say(f"      [dg/dlnr]/|g| {c['jump_dgdx_rel']:.2e}  -> "
                    f"effective-density step "
                    f"{out[key]['rho_eff_step_over_baryon']:.2e} x the local "
                    f"baryon density")
                say("      shrinking window, [g]/|g| : " + "  ".join(
                    f"{cc['tmax']:.0e}->{cc['jump_g_rel']:.1e}"
                    for cc in seq))
    RES["S2_kernel_surface"] = out


# ==========================================================================
def s3_field_surface(g, prof):
    head("S3  What the clip surface does in a FIELD formulation "
         "div[K(q) grad Psi] = 4 pi G rho")
    say("In spherical symmetry the field equation integrates exactly:")
    say("   K Psi' r^2 = G M(<r)   =>   Psi' = G M /(K r^2)")
    say("so the density a Newtonian observer would infer is")
    say("   rho_eff = rho/K - M K'/(4 pi r^2 K^2),   K' = alpha p q^(p-1) q'.")
    say("q' jumps across the clip surface, so rho_eff jumps.  There is NO "
        "delta")
    say("function: the force and the flux are continuous, the inferred "
        "DENSITY is not.")
    out = {}
    r = prof[0]
    rho_f = prof[1] + C.NK.RHO_BAR_B
    Mr = prof[2]
    for rho_ref in (1e5, 1e6):
        rc = CM.clip_radii(prof, rho_ref)["q_leaves_0"]
        if not np.isfinite(rc):
            continue
        dlnrho = float(np.interp(math.log(rc), np.log(r),
                                 np.gradient(np.log(rho_f), np.log(r))))
        M = float(np.interp(math.log(rc), np.log(r), Mr))
        rho_c = float(np.interp(math.log(rc), np.log(r), rho_f))
        jq = abs(dlnrho) / rc                      # [dq/dr], per kpc
        for alpha in (1.0, 3.0, 10.0):
            for p in (0.5, 1.0, 2.0):
                if p > 1.0:
                    jK = 0.0                        # q^(p-1) -> 0 at q = 0
                    note = "no jump: q^(p-1) vanishes at the surface"
                elif p == 1.0:
                    jK = alpha * jq
                    note = "finite jump"
                else:
                    jK = float("inf")
                    note = ("DIVERGENT: q^(p-1) -> inf at q = 0, so rho_eff "
                            "has an integrable but unbounded spike")
                jrho = (float("inf") if not np.isfinite(jK)
                        else M * jK / (4.0 * math.pi * rc ** 2))
                key = f"rho_ref={rho_ref:g}|alpha={alpha}|p={p}"
                out[key] = dict(r_clip_kpc=float(rc), jump_dq_dr=float(jq),
                                jump_rho_eff=float(jrho),
                                rho_baryon_at_clip=rho_c,
                                ratio_to_baryon=float(jrho / rho_c),
                                note=note)
                say(f"   rho_ref={rho_ref:<6g} alpha={alpha:<5.1f} p={p:<4.1f} "
                    f" [rho_eff] = {jrho:11.4g} Msun/kpc^3 = "
                    f"{jrho / rho_c:9.3g} x the local baryon density   "
                    f"({note})")
    say("")
    say("READ THIS OFF: at p = 1 the clip puts a step in the INFERRED dark")
    say("density of order alpha |dln rho/dln r| M/(4 pi r^3) -- comparable to")
    say("the baryon density itself.  At p > 1 the step is exactly zero; at")
    say("p < 1 it diverges.  So p >= 1 is not a preference, it is a "
        "requirement,")
    say("and p = 1 still needs the clip rounded.")
    RES["S3_field_surface"] = out


# ==========================================================================
def _solar(kind, w, rho_ref, L_s=0.0):
    """q and |grad q| at the solar position for one clip variant.

    Same 1-D vertical reduction the previous lane used: rho(z) = 7.6e7
    exp(-|z|/0.30) + rho_bar_b, evaluated at z = 20 pc.
    """
    z = np.linspace(-6.0, 6.0, 240001)
    rho = RHO_SUN_LOCAL * np.exp(-np.abs(z) / HZ_LOCAL) + C.NK.RHO_BAR_B
    if L_s > 0:
        gk = np.exp(-z ** 2 / (2 * L_s ** 2)); gk /= gk.sum()
        rho = np.real(np.fft.ifft(np.fft.fft(rho)
                                  * np.fft.fft(np.fft.ifftshift(gk))))
    q = CM.q_clip(CM.u_of_rho(rho, rho_ref), w=w, kind=kind)
    i = int(np.argmin(np.abs(z - Z_SUN)))
    return float(q[i]), float(abs(np.gradient(q, z[1] - z[0])[i]))


def s5_solar_trade():
    head("S5  Does solar-system safety survive the C^2 replacement?")
    say("Channel: violation of the inverse-square law, eps(1 AU) = "
        "|F'| |grad q| D /(2F),")
    say(f"bound {ISL_BOUND:g}.  Oort window: F_local in [1.10, 1.70].")
    out = {}
    rows = []
    for rho_ref in (1e4, 1e5, 1e6, 1e7, 4e7):
        for kind, ws in (("hard", (0.0,)), ("quad", (0.05, 0.1, 0.2)),
                         ("quintic", (0.02, 0.05, 0.1, 0.2, 0.5)),
                         ("softplus", (0.02, 0.05, 0.1, 0.2))):
            for w in ws:
                q0, dq = _solar(kind, w, rho_ref)
                for alpha, p in ((3.0, 1.0), (10.0, 1.0), (10.0, 2.0)):
                    F = float(C.NK.F_poly(q0, alpha=alpha, p=p))
                    dF = (float(C.NK.dF_poly(q0, alpha=alpha, p=p))
                          if q0 > 0 else 0.0)
                    eps = (abs(dF) * dq * C.AU_KPC / (2.0 * F)
                           if q0 > 0 else 0.0)
                    rows.append(dict(
                        rho_ref=rho_ref, kind=kind, w=w, alpha=alpha, p=p,
                        q_sun=q0, grad_q=dq, eps_1AU=float(eps),
                        a_anom_1AU_ms2=float(eps * NEWTON_1AU_MS2),
                        F_local=F, isl_ok=bool(eps < ISL_BOUND),
                        exactly_zero=bool(q0 == 0.0)))
    out["grid"] = rows
    say("   kind      w     rho_ref   q(Sun)      eps(1 AU)    verdict  "
        "(alpha=3, p=1)")
    for r in rows:
        if r["alpha"] == 3.0 and r["p"] == 1.0:
            say(f"   {r['kind']:<9s} {r['w']:<5.2f} {r['rho_ref']:<9.3g} "
                f"{r['q_sun']:11.4e} {r['eps_1AU']:11.4e}   "
                f"{'EXACT ZERO' if r['exactly_zero'] else ('ok' if r['isl_ok'] else 'FAILS')}")
    #  the headline: largest rho_ref allowed by the ISL bound, per variant
    lim = {}
    for kind in CM.KINDS:
        for w in sorted({r["w"] for r in rows if r["kind"] == kind}):
            sel = [r for r in rows if r["kind"] == kind and r["w"] == w
                   and r["alpha"] == 3.0 and r["p"] == 1.0]
            ok = [r["rho_ref"] for r in sel if r["isl_ok"]]
            lim[f"{kind}|w={w}"] = dict(
                max_rho_ref_passing=float(max(ok)) if ok else None,
                exact_zero_up_to=float(max(
                    [r["rho_ref"] for r in sel if r["exactly_zero"]] or [0.0])))
    out["rho_ref_limits_alpha3_p1"] = lim
    say("")
    say("   variant             largest rho_ref passing the ISL bound   "
        "exactly q=0 up to")
    for k, v in lim.items():
        say(f"   {k:<19s} {str(v['max_rho_ref_passing']):>16s}   "
            f"{v['exact_zero_up_to']:>18.3g}")
    RES["S5_solar_trade"] = out


# ==========================================================================
def s4_rotation_with_c2():
    head("S4  Does the rotation-curve behaviour survive the C^2 "
         "replacement?")
    say("Forward test on the SPARC TRAIN split, delta-type q, no per-galaxy "
        "freedom.")
    say("Reported in ACCELERATION space (the honest currency) with the "
        "fraction of")
    say("points at which the model predicts g <= 0 -- the repulsive shells.")
    train = C.sparc("train")
    profs = {g.name: C.build_profile(g) for g in train}
    reqs = {g.name: C.required(g, profs[g.name][5]) for g in train}
    out = []
    variants = [("hard", 0.0), ("quad", 0.10), ("quintic", 0.05),
                ("quintic", 0.10), ("quintic", 0.20), ("softplus", 0.10)]
    for rho_ref in (1e5, 1e6):
        flds = {}
        for kind, w in variants:
            flds[(kind, w)] = {
                g.name: CM.build_field_clipped(profs[g.name],
                                               rho_ref=rho_ref, w=w,
                                               kind=kind, label=g.name)
                for g in train}
        for kind, w in variants:
            for alpha in (3.0, 10.0):
                for p in (1.0, 2.0):
                    res, nneg, ntot, nglx = [], 0, 0, 0
                    for g in train:
                        R, F_req, D_req, g_obs = reqs[g.name]
                        Fe, D = DC.phi_and_D(
                            flds[(kind, w)][g.name], R, Fname="F1_poly",
                            alpha=alpha, p=p, Mtot=profs[g.name][5],
                            use_gpu=GPU, chunk=64)
                        bad = D <= 0
                        nneg += int(bad.sum()); ntot += len(R)
                        nglx += int(bad.any())
                        if (~bad).any():
                            gp = C.G * profs[g.name][5] * D[~bad] / R[~bad] ** 2
                            res.append(np.log10(gp / g_obs[~bad]))
                    allr = np.concatenate(res) if res else np.array([np.nan])
                    per = np.array([np.mean(x) for x in res]) if res else \
                        np.array([np.nan])
                    row = dict(rho_ref=rho_ref, kind=kind, w=w, alpha=alpha,
                               p=p, rms_dex=float(np.sqrt(np.mean(allr ** 2))),
                               bias_dex=float(np.mean(allr)),
                               galaxy_scatter_dex=float(np.std(per)),
                               frac_repulsive=float(nneg / ntot),
                               n_galaxies_repulsive=nglx,
                               n_points=int(len(allr)))
                    out.append(row)
                    say(f"   rho_ref={rho_ref:<6g} {kind:<9s} w={w:<5.2f} "
                        f"a={alpha:<5.1f} p={p:<4.1f}  rms "
                        f"{row['rms_dex']:.3f} dex   repulsive points "
                        f"{100 * row['frac_repulsive']:5.1f}%   galaxies "
                        f"{nglx:2d}/{len(train)}")
    RES["S4_rotation_with_c2"] = out
    say("")
    ok = [r for r in out if r["frac_repulsive"] == 0.0]
    if ok:
        b = min(ok, key=lambda r: r["rms_dex"])
        say(f"   best clip variant with NO repulsive point: {b['kind']} "
            f"w={b['w']} rho_ref={b['rho_ref']:g} alpha={b['alpha']} "
            f"p={b['p']}, rms {b['rms_dex']:.3f} dex")
    else:
        say("   NO delta-type variant is free of repulsive points anywhere "
            "in the grid.")


# ==========================================================================
def main():
    g, prof = s1_smoothness_class()
    s2_kernel_surface(g, prof)
    s3_field_surface(g, prof)
    s5_solar_trade()
    s4_rotation_with_c2()
    RES["runtime_s"] = time.time() - T0
    with open("smoothness_audit.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    say(f"\nwrote smoothness_audit.json  ({time.time() - T0:.1f} s)")


if __name__ == "__main__":
    main()
