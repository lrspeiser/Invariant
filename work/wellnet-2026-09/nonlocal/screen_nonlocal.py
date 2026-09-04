"""Screen for the symmetric nonlocal kernel family.

Runs, in the order the lane brief specifies:

  G1  Newtonian limit as alpha -> 0
  G2  reciprocity, and the momentum residual a reciprocal kernel still leaves
  G3  solar-system safety, quantified against the inverse-square-law channel
  G4  the flat-rotation-curve question -- theory first, then numerics
  G5  numerical gates: resolution, domain size, source-label permutation
  G6  cost, and the accuracy price of the accelerations
  G7  the configurations where a nonlocal kernel says something no local law
      does, with predicted effect sizes

Writes `nonlocal_results.json`.  Nothing here fits anything to data: the only
observational contact is a forward inversion on the SPARC TRAIN split, and the
validation and blind splits are never loaded.  KiDS and wide binaries are not
touched anywhere in this lane.
"""
from __future__ import annotations

import json
import math
import os
import platform
import re
import sys
import time
from datetime import datetime, timezone

import numpy as np

import nonlocal_kernel as NK
import models as MO

HERE = os.path.dirname(os.path.abspath(__file__))
GLAB = os.path.abspath(os.path.join(HERE, "..", "..", "gravitylab"))
BAR = "=" * 78
RES: dict = {}

try:
    import cupy as _cp
    _cp.zeros(1)
    GPU = True
except Exception:                                   # pragma: no cover
    GPU = False


def head(t):
    print("\n" + BAR + "\n" + t + "\n" + BAR)


def say(*a):
    print("   " + " ".join(str(x) for x in a))


# ==========================================================================
#  Solar-neighbourhood check, shared by G4b, G4e and G7.
#
#  The steep direction at the Sun is the vertical one, so the reduction is a
#  1-D profile rho(z) = rho_sun exp(-|z|/h_z) + rho_bar with h_z = 0.30 kpc
#  and rho_sun = 7.6e7 Msun/kpc^3 (Bland-Hawthorn & Gerhard 2016 local baryon
#  budget).  The Sun sits ~20 pc above the plane, which is where the gradient
#  is evaluated.  For the screened definition the operator (1 - L_q^2 d^2/dz^2)
#  is inverted on that line, which is what makes L_q relax the bound: it
#  smooths the gradient that drives the anomaly.
#
#  Anomalous force / Newtonian force at separation D:
#      eps(D) = alpha p qbar^(p-1) |grad q| D / (2 F)
#  a fixed-direction term falling as 1/D, so the inner solar system is the
#  binding case.
RHO_SUN_LOCAL = 7.6e7
HZ_LOCAL = 0.30
Z_SUN = 0.020
NEWTON_1AU_MS2 = 5.9301e-3


_SOLAR_CACHE: dict = {}


def _solar_profile(qdef, kw):
    """q and |grad q| at the solar position, cached on (qdef, kw)."""
    key = (qdef, tuple(sorted(kw.items())))
    if key in _SOLAR_CACHE:
        return _SOLAR_CACHE[key]
    z = np.linspace(-6.0, 6.0, 24001)
    rho = RHO_SUN_LOCAL * np.exp(-np.abs(z) / HZ_LOCAL) + NK.RHO_BAR_B
    L_s = kw.get("L_s", 0.0)
    if L_s > 0:
        # FFT convolution: np.convolve on a 24k grid is O(N^2) and dominated
        # the whole screen the first time this was run.
        g = np.exp(-z ** 2 / (2 * L_s ** 2))
        g /= g.sum()
        rho = np.real(np.fft.ifft(np.fft.fft(rho)
                                  * np.fft.fft(np.fft.ifftshift(g))))
    rho_ref = kw["rho_ref"]
    m = kw.get("m", 1.0)
    if qdef == "delta":
        q = NK.q_from_delta(rho, rho_ref)
    elif qdef == "smooth":
        q = NK.q_from_smooth(rho, rho_ref, m=m)
    elif qdef == "screen":
        gN = NK.G * 8.0e10 / 8.2 ** 2
        S = NK.q_source_Q3(rho, gN, rho_ref, m=m)
        q = np.clip(NK.screen_1d(z, S, kw.get("L_q", 0.0)), 0.0, 1.0 - 1e-12)
    else:
        q = np.zeros_like(z)
    i = int(np.argmin(np.abs(z - Z_SUN)))
    out = (float(q[i]), float(abs(np.gradient(q, z[1] - z[0])[i])))
    _SOLAR_CACHE[key] = out
    return out


def solar_check(qdef, kw, alpha, p, bound=1e-11, Fname="F1_poly", beta=0.0):
    """Inverse-square-law and Oort status of one global parameter set.

    F and F' come from the ACTUAL family.  Using 1 + alpha q^p for the
    exponential and Pade members misreports F_local by up to a factor two,
    which is exactly the difference between passing and failing the Oort
    window.

    THE OORT LIMIT.  With no dark matter the local dynamical surface density
    must be the local baryonic one times F evaluated on the short paths that
    set the vertical force.  Sigma_dyn(|z| < 1.1 kpc) = 68 +/- 4 Msun/pc^2
    against Sigma_baryon = 47-54 (Bovy & Rix 2013; McKee, Parravano &
    Hollenbach 2015, quoted from the literature and not refitted here), so
    F_local must sit near 1.3 and certainly inside about [1.1, 1.7].
    """
    q0, dq = _solar_profile(qdef, kw)
    Ffun, dFfun, _ = NK.FAMILIES[Fname]
    F = float(Ffun(q0, 0.0, alpha=alpha, beta=beta, p=p))
    dF = float(dFfun(q0, 0.0, alpha=alpha, beta=beta, p=p)) if q0 > 0 else 0.0
    eps = abs(dF) * dq * NK.AU_KPC / (2.0 * F) if q0 > 0 else 0.0
    return dict(q_sun=q0, grad_q_per_kpc=dq, eps_1AU=float(eps),
                a_anom_1AU_ms2=float(eps * NEWTON_1AU_MS2),
                F_local=float(F),
                passes=bool(eps < bound),
                passes_oort=bool(1.10 <= F <= 1.70))


# ==========================================================================
def g1_newtonian_limit():
    head("G1  Newtonian limit: alpha -> 0 must reproduce Newton exactly")
    out = {}
    M, rd = 5e10, 3.0
    rho0 = M / (8 * math.pi * rd ** 3)
    rfun = lambda x: rho0 * np.exp(-np.asarray(x, float) / rd)
    Mfun = lambda x: MO.exp_sphere_M(x, M, rd)
    r = np.geomspace(1e-4, 1e5, 1500)
    fld = NK.SphericalField(r=r, rho=rfun(r), q=np.full_like(r, 0.5),
                            rho_fun=rfun, Menc_fun=Mfun)
    re = np.geomspace(0.3, 300.0, 12)
    x = re / rd
    Phi_ex = (-NK.G * Mfun(re) / re
              - 4 * math.pi * NK.G * rho0 * rd ** 2 * np.exp(-x) * (1 + x))
    v2_ex = NK.G * Mfun(re) / re
    for fam in NK.FAMILIES:
        phi = NK.spherical_potential_batch(fld, re, Fname=fam, alpha=0.0,
                                           beta=0.0, p=1.0)
        out[f"{fam}_phi_relerr"] = float(np.max(np.abs(phi / Phi_ex - 1)))
    v2, _, _ = NK.spherical_vcirc_spline(fld, re, alpha=0.0, npts=260, pad=0.6)
    out["vcirc_relerr"] = float(np.max(np.abs(v2 / v2_ex - 1)))
    say(f"analytic potential, all four families : "
        f"max rel err {max(out[k] for k in out if k.endswith('relerr')):.3e}")
    for fam in NK.FAMILIES:
        say(f"   {fam:<10s} {out[f'{fam}_phi_relerr']:.3e}")
    say(f"circular speed from spline gradient   : {out['vcirc_relerr']:.3e}")

    # deviation must vanish LINEARLY in alpha (and to round-off at alpha = 0)
    fld.q = np.full_like(r, 0.5)
    base = NK.spherical_potential_batch(fld, re, alpha=0.0)
    ladder = {}
    for a in [1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 0.0]:
        ph = NK.spherical_potential_batch(fld, re, alpha=a, p=1.0)
        ladder[f"{a:g}"] = float(np.max(np.abs(ph / base - 1.0)))
    out["alpha_ladder"] = ladder
    say("fractional deviation vs alpha (q = 1/2 everywhere, F1):")
    for k, v in ladder.items():
        say(f"   alpha = {k:<8s} max|dPhi/Phi| = {v:.6e}")
    out["exact_at_alpha0"] = ladder["0"]
    ok = ladder["0"] < 1e-15 and out["vcirc_relerr"] < 1e-4
    out["pass"] = bool(ok)
    say(f"[{'PASS' if ok else 'FAIL'}] alpha = 0 reproduces Newton to "
        f"{ladder['0']:.1e} (round-off)")
    RES["G1_newtonian_limit"] = out


# ==========================================================================
def g2_reciprocity_and_momentum():
    head("G2  Reciprocity, and the momentum residual it does NOT buy")
    out = {}
    rng = np.random.default_rng(20260903)
    pos = rng.uniform(-200, 200, (4, 3))
    mass = 10 ** rng.uniform(9, 11.5, 4)
    cl = NK.GaussianCloud(pos=pos, mass=mass, L=35.0, rho_amb=NK.RHO_BAR_B)

    worst = 0.0
    for _ in range(300):
        a = rng.uniform(-300, 300, 3)
        b = rng.uniform(-300, 300, 3)
        q1, *_ = NK.path_qbar(cl, a, b, n_s=48, rho_ref=1e3)
        q2, *_ = NK.path_qbar(cl, b, a, n_s=48, rho_ref=1e3)
        worst = max(worst, abs(q1 - q2) / max(abs(q1), 1e-300))
    out["qbar_symmetry_max_relerr"] = float(worst)
    say(f"qbar(x,x') vs qbar(x',x), 300 random asymmetric pairs : "
        f"{worst:.3e}")

    # -- the residual force on an ISOLATED system whose q it sources itself
    p2 = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    leak = []
    for m2 in [1e11, 3e10, 1e10, 1e9]:
        mm = np.array([1e11, m2])
        c2 = NK.GaussianCloud(pos=p2, mass=mm, L=20.0, rho_amb=NK.RHO_BAR_B)
        f1, f2, d = NK.pair_forces(c2, p2[0], p2[1], mm[0], mm[1],
                                   alpha=1.0, p=1.0, rho_ref=1e3)
        net = np.linalg.norm(f1 + f2)
        pred = abs(d["dF"] * (d["q2"] - d["q1"]))
        leak.append(dict(m2_over_m1=float(m2 / 1e11), qbar=d["qbar"],
                         dq=float(d["q2"] - d["q1"]),
                         net_over_fN=float(net / d["f_newton"]),
                         identity=float(pred)))
    out["two_body_leak"] = leak
    say("isolated pair, q sourced by the pair itself, alpha = 1, p = 1:")
    say("   m2/m1     qbar      q2-q1      |f1+f2|/f_N   F'(qbar)*(q2-q1)")
    for L in leak:
        say(f"   {L['m2_over_m1']:<9.3g} {L['qbar']:.5f}  {L['dq']:+.3e}  "
            f"{L['net_over_fN']:.6e}   {L['identity']:.6e}")
    out["identity_max_relerr"] = float(max(
        abs(L["net_over_fN"] - L["identity"]) / max(L["identity"], 1e-300)
        for L in leak if L["identity"] > 0))
    say(f"   identity |f1+f2| = G m1 m2 F'(qbar) [q2-q1] / D^2 holds to "
        f"{out['identity_max_relerr']:.2e}")

    # equal masses: exact null by reflection symmetry
    ce = NK.GaussianCloud(pos=p2, mass=np.array([1e11, 1e11]), L=20.0,
                          rho_amb=NK.RHO_BAR_B)
    f1, f2, d = NK.pair_forces(ce, p2[0], p2[1], 1e11, 1e11, alpha=1.0,
                               p=1.0, rho_ref=1e3)
    out["equal_mass_leak"] = float(np.linalg.norm(f1 + f2) / d["f_newton"])
    say(f"equal-mass pair (reflection-symmetric q)  : leak/f_N = "
        f"{out['equal_mass_leak']:.2e}")

    # three-body, reciprocal vs deliberately non-reciprocal path weighting
    pos3 = np.array([[0.0, 0.0, 0.0], [150.0, 0.0, 0.0], [40.0, 90.0, 0.0]])
    m3 = np.array([2e11, 5e10, 8e9])
    c3 = NK.GaussianCloud(pos=pos3, mass=m3, L=25.0, rho_amb=NK.RHO_BAR_B)
    fN12 = NK.G * m3[0] * m3[1] / 150.0 ** 2
    rows = []
    for gam in [0.0, 0.5, 1.0, 2.0]:
        w = None if gam == 0 else (lambda s, g=gam: 1.0 + g * (s - 0.5))
        f, net, dec = NK.nbody_forces(c3, pos3, m3, alpha=1.0, p=1.0,
                                      rho_ref=1e3, weight=w)
        rows.append(dict(gamma=gam,
                         net_over_fN=float(np.linalg.norm(net) / fN12),
                         gradient_term=float(np.linalg.norm(dec["gradient"])
                                             / fN12),
                         asymmetry_term=float(np.linalg.norm(dec["asymmetry"])
                                              / fN12),
                         split_residual=float(np.max(np.abs(
                             net - dec["gradient"] - dec["asymmetry"]))
                             / np.linalg.norm(net))))
    out["three_body"] = rows
    say("isolated 3-body, w(s) = 1 + gamma (s - 1/2)  (gamma = 0 reciprocal):")
    say("   gamma   |sum f|/f_N   gradient   asymmetry   split resid")
    for r_ in rows:
        say(f"   {r_['gamma']:<7.1f} {r_['net_over_fN']:.5f}      "
            f"{r_['gradient_term']:.5f}    {r_['asymmetry_term']:.5f}     "
            f"{r_['split_residual']:.1e}")

    # alpha scaling of the leak
    sc = {}
    for a in [1.0, 1e-1, 1e-2, 1e-3, 0.0]:
        f, net, _ = NK.nbody_forces(c3, pos3, m3, alpha=a, p=1.0, rho_ref=1e3)
        sc[f"{a:g}"] = float(np.linalg.norm(net) / fN12)
    out["leak_vs_alpha"] = sc
    say("leak scales linearly in alpha and vanishes at alpha = 0: "
        + ", ".join(f"{k}:{v:.3e}" for k, v in sc.items()))
    out["pass_reciprocity"] = bool(worst < 1e-12)
    out["momentum_conserved"] = bool(max(L["net_over_fN"] for L in leak) < 1e-12)
    RES["G2_reciprocity_momentum"] = out


# ==========================================================================
def _mw_disk_grid(L_s, nR=420, nz=421, Rmax=60.0, zmax=20.0):
    """Milky Way exponential disk, smoothed, on an (R,z) grid.

    Sigma(R0 = 8.2 kpc) = 45 Msun/pc^2, h_R = 2.6 kpc, h_z = 0.30 kpc, giving
    a midplane density at the Sun of 7.5e7 Msun/kpc^3 = 0.075 Msun/pc^3,
    consistent with the Bland-Hawthorn & Gerhard 2016 local baryon budget.
    """
    R0, hR, hz = 8.2, 2.6, 0.30
    Sig0 = 45.0e6 * math.exp(R0 / hR)          # Msun/kpc^2
    R = np.linspace(1e-3, Rmax, nR)
    z = np.linspace(-zmax, zmax, nz)
    RR, ZZ = np.meshgrid(R, z, indexing="ij")
    rho = (Sig0 / (2 * hz)) * np.exp(-RR / hR) * np.exp(-np.abs(ZZ) / hz)
    rho = rho + NK.RHO_BAR_B
    rs = NK.smooth_axisym(R, z, rho, L_s) if L_s > 0 else rho
    return R, z, rs, (R0, hR, hz)


def g3_solar_system():
    head("G3  Solar-system safety, and what it costs the galaxy application")
    out = {}
    rows = []
    for L_s in [0.0, 0.05, 0.3, 1.0]:
        R, z, rs, (R0, hR, hz) = _mw_disk_grid(L_s)
        iR = int(np.argmin(np.abs(R - R0)))
        iz = int(np.argmin(np.abs(z)))
        rho_sun = float(rs[iR, iz])
        # steepest logarithmic gradient of rho at the Sun (vertical direction)
        dz = z[1] - z[0]
        dlnrho_dz = float(abs(np.gradient(np.log(rs[iR, :]), dz)[iz + 1]))
        dR_ = R[1] - R[0]
        dlnrho_dR = float(abs(np.gradient(np.log(rs[:, iz]), dR_)[iR]))
        glog = max(dlnrho_dz, dlnrho_dR)
        # density at 25 kpc in the midplane, for the efficacy comparison
        i25 = int(np.argmin(np.abs(R - 25.0)))
        rho_25 = float(rs[i25, iz])
        rows.append(dict(L_s_kpc=L_s, rho_sun=rho_sun,
                         dln_rho_max_per_kpc=glog, rho_25kpc=rho_25,
                         contrast_8_to_25=rho_sun / rho_25))
        say(f"L_s = {L_s:<5.2f} kpc : rho(Sun) = {rho_sun:.3e}, "
            f"|dln rho| = {glog:.3f}/kpc, rho(25 kpc) = {rho_25:.3e}, "
            f"contrast = {rho_sun / rho_25:.1f}")
    out["local_density"] = rows

    base = rows[2]              # L_s = 0.3 kpc, the fiducial smoothing
    NEWTON_1AU = 5.9301e-3      # m/s^2, GM_sun / (1 AU)^2

    # THE CLIPPED delta FORM IS EXACTLY SAFE, and it is worth being explicit
    # about why: q = rho_ref/rho_s - 1 clipped at 0 is IDENTICALLY ZERO wherever
    # rho_s > rho_ref.  The Sun sits at rho_s ~ 4e7 Msun/kpc^3, so for any
    # rho_ref below that -- and rho_ref must be far below it or the whole
    # Galaxy would count as a void -- F = 1 to all orders.  No expansion, no
    # small parameter, no residual.
    out["delta_form_eps_1AU"] = 0.0
    out["delta_form_q_sun"] = float(NK.q_from_delta(base["rho_sun"], 1e6))
    say("")
    say("clipped delta form, q = rho_ref/rho_s - 1 clipped to [0,1):")
    say(f"   q(Sun) = {out['delta_form_q_sun']:.1f} exactly for every "
        f"rho_ref < rho_s(Sun) = {base['rho_sun']:.2e};  F = 1 identically, "
        f"eps = 0 exactly.")

    # The SMOOTH form q = 1/(1 + rho/rho_ref) has no clip and so has a small
    # but nonzero q at the Sun.  The anomalous force is
    #   G m M F'(qbar) grad_1 qbar / D,   grad_1 qbar -> grad q / 2,
    # a FIXED-DIRECTION force falling as 1/D, so
    #   eps(D) = |anomalous| / |Newton| = (alpha p / 2) q^p |dln q| D.
    say("")
    say("smooth form q = 1/(1 + rho_s/rho_ref)  (programme Q1, m = 1):")
    say("   rho_ref     alpha  p    q(Sun)     q(25kpc)   eps(1AU)    "
        "a_anom(1AU) m/s^2")
    tab = []
    for rho_ref in [NK.RHO_BAR_B, 1e3, 1e4, 1e5, 1e6, 1e7]:
        q_sun = float(NK.q_from_smooth(base["rho_sun"], rho_ref))
        q25 = float(NK.q_from_smooth(base["rho_25kpc"], rho_ref))
        # |dln q| = |dln rho| * (1 - q)  for the smooth form
        dlnq = base["dln_rho_max_per_kpc"] * (1.0 - q_sun)
        for alpha, p in [(1.0, 1.0), (3.0, 1.0), (1.0, 2.0), (10.0, 1.0)]:
            eps_1au = (alpha * p / 2.0) * q_sun ** p * dlnq * NK.AU_KPC
            tab.append(dict(rho_ref=float(rho_ref), alpha=alpha, p=p,
                            q_sun=q_sun, q_25kpc=q25,
                            eps_1AU=float(eps_1au),
                            eps_10AU=float(eps_1au * 10.0),
                            a_anom_1AU_ms2=float(eps_1au * NEWTON_1AU),
                            passes_1e11=bool(eps_1au < 1e-11)))
    out["isl_violation_smooth"] = tab
    for t in tab:
        say(f"   {t['rho_ref']:<10.3g} {t['alpha']:<5.1f} {t['p']:<4.1f} "
            f"{t['q_sun']:.3e}  {t['q_25kpc']:.3e}  {t['eps_1AU']:.3e}  "
            f"{t['a_anom_1AU_ms2']:.3e}  "
            f"{'ok' if t['passes_1e11'] else 'FAILS 1e-11'}")

    # ceiling on q at the Sun implied by the bound, and whether the galaxy
    # application survives it
    glog = base["dln_rho_max_per_kpc"]
    q_max = 1e-11 / (0.5 * glog * NK.AU_KPC)
    out["q_sun_ceiling_alpha1_p1"] = float(q_max)
    out["rho_ref_ceiling_alpha1_p1"] = float(q_max * base["rho_sun"])
    contrast = base["rho_sun"] / base["rho_25kpc"]
    out["density_contrast_8_to_25"] = float(contrast)
    out["q_at_25kpc_at_ceiling"] = float(min(q_max * contrast, 1.0))
    say("")
    say(f"bound eps(1 AU) < 1e-11, alpha = p = 1 : q(Sun) < {q_max:.3e}, "
        f"i.e. rho_ref < {q_max * base['rho_sun']:.3e} Msun/kpc^3")
    say(f"   midplane contrast rho(8.2)/rho(25) = {contrast:.1f} then gives "
        f"q(25 kpc) = {min(q_max * contrast, 1.0):.3f}")
    say("   the solar-system bound and a galaxy-scale q of order unity are "
        "compatible for the smooth form,")
    say("   with about one order of magnitude of headroom; the clipped delta "
        "form has unlimited headroom.")

    # exact path average vs the linearisation, smooth form, vertical direction
    hz = 0.30
    rho_ref = 1e5
    rho_sun = base["rho_sun"]
    rho_of_l = lambda l: rho_sun * np.exp(-np.abs(l) / hz)
    chk = []
    for Dau in [1.0, 10.0, 30.0]:
        D = Dau * NK.AU_KPC
        ss, ws = NK.gauss_legendre(64)
        s = 0.5 * (ss + 1.0)
        w = 0.5 * ws
        qs = NK.q_from_smooth(rho_of_l(s * D), rho_ref)
        qbar = float(np.sum(w * qs))
        q0 = float(NK.q_from_smooth(rho_sun, rho_ref))
        # exact grad_1 qbar along the path direction
        dq = qs * (1.0 - qs) / hz
        g1 = float(np.sum(w * (1.0 - s) * dq))
        eps_exact = float(1.0 * 1.0 * g1 * D / (1.0 + 1.0 * qbar))
        eps_lin = float(0.5 * q0 * (1.0 - q0) / hz * D)
        chk.append(dict(D_AU=Dau, qbar=qbar, q_sun=q0,
                        eps_exact=eps_exact, eps_linearised=eps_lin,
                        ratio=eps_exact / eps_lin))
    out["path_average_check"] = chk
    say("")
    say("exact path average vs the linearisation (smooth form, rho_ref = 1e5, "
        "alpha = p = 1,")
    say("unsmoothed h_z = 0.30 kpc, i.e. the worst-case gradient rather than "
        "the L_s = 0.3 kpc fiducial):")
    for c in chk:
        say(f"   D = {c['D_AU']:>4.0f} AU : qbar = {c['qbar']:.8e}, "
            f"eps_exact = {c['eps_exact']:.4e}, "
            f"eps_lin = {c['eps_linearised']:.4e}, ratio = {c['ratio']:.4f}")
    RES["G3_solar_system"] = out


# ==========================================================================
def g4_rotation_curves():
    head("G4  Flat rotation curves: theory first, then numbers")
    out = {}

    # ---- Theorem 1: F bounded => asymptotically Keplerian ----------------
    say("THEOREM 1.  For a point source Phi = -G M F(r)/r exactly, so")
    say("   v_c^2 = G M (F/r - F') = (G M F / r) (1 - dlnF/dlnr).")
    say("   Every family here has qbar in [0,1) and Tbar finite, so F is")
    say("   bounded: sup F = 1+alpha (F1), e^alpha (F2), 1+alpha/(1+beta) (F3).")
    say("   Hence r v_c^2 -> G M F_inf: the curve is exactly Keplerian at")
    say("   large r, with G renormalised by F_inf.  Asymptotically flat is")
    say("   impossible for the whole family.  Numerically:")
    gal = MO.GALAXY_LADDER[4]
    fld = MO.build_field(gal, "smooth", rho_ref=1e5, m=1.0)
    rr = np.geomspace(10.0, 3.0e4, 40)
    t1 = []
    for fam, alpha, beta in [("F1_poly", 3.0, 0.0), ("F2_exp", 1.0, 0.0),
                             ("F3_pade", 5.0, 1.0)]:
        v2, _, _ = NK.spherical_vcirc_spline(fld, rr, Fname=fam, alpha=alpha,
                                             beta=beta, p=1.0, use_gpu=GPU)
        Feff = rr * v2 / (NK.G * gal.Mtot)
        sup = NK.F_sup(fam, alpha, beta, 1.0)
        t1.append(dict(family=fam, alpha=alpha, beta=beta, F_sup=float(sup),
                       r_v2_over_GM_at_3e4kpc=float(Feff[-1]),
                       ratio_to_sup=float(Feff[-1] / sup),
                       outer_logslope_v=float(np.gradient(
                           np.log(np.sqrt(np.abs(v2))), np.log(rr))[-1])))
        say(f"   {fam:<9s} alpha={alpha:<4.1f} sup F={sup:6.3f}   "
            f"r v^2/GM at 30 Mpc = {Feff[-1]:.4f} "
            f"({Feff[-1] / sup:.4f} of sup),  dlnv/dlnr = "
            f"{t1[-1]['outer_logslope_v']:+.4f}  (Kepler = -0.5)")
    out["theorem1_asymptotic"] = t1

    # ---- the naive argument, and why it is wrong ------------------------
    say("")
    say("THE NAIVE ARGUMENT IS WRONG.  'q is roughly constant on galaxy")
    say("   scales, so F is roughly constant, so Phi is a rescaled Newtonian")
    say("   potential and v^2 ~ 1/r' drops the -G M F' term.  That term is")
    say("   not small: v^2 = (G M F/r)(1 - dlnF/dlnr), so an F rising with")
    say("   logarithmic slope approaching 1 flattens the curve completely.")
    say("   Size of the dropped term, MW-like model, smooth q, alpha = 3:")
    rr2 = np.geomspace(3.0, 60.0, 16)
    Feff = NK.spherical_F_effective(fld, rr2, Fname="F1_poly", alpha=3.0,
                                    p=1.0)
    s_eff = np.gradient(np.log(Feff), np.log(rr2))
    out["dropped_term"] = dict(r_kpc=rr2.tolist(), F_eff=Feff.tolist(),
                               dlnF_dlnr=s_eff.tolist(),
                               max_dlnF_dlnr=float(np.max(s_eff)))
    say("   r/kpc    " + " ".join(f"{x:6.1f}" for x in rr2[::3]))
    say("   F_eff    " + " ".join(f"{x:6.3f}" for x in Feff[::3]))
    say("   dlnF/dlnr" + " ".join(f"{x:6.3f}" for x in s_eff[::3]))
    say(f"   max dlnF/dlnr = {np.max(s_eff):.3f}; the naive argument assumes 0.")

    # ---- Theorem 2: the repulsion threshold -----------------------------
    say("")
    say("THEOREM 2.  v_c^2 < 0 whenever dlnF/dlnr > 1.  Flat rotation sits on")
    say("   the boundary of a REPULSIVE regime: to flatten a curve F must")
    say("   rise with logarithmic slope tending to 1, and any overshoot")
    say("   removes circular orbits entirely.  A q that switches sharply --")
    say("   which the clipped delta form does by construction, since")
    say("   q = rho_ref/rho_s - 1 is a near-step function of density -- always")
    say("   overshoots.")

    # ---- Theorem 3: the flat window ------------------------------------
    say("")
    say("THEOREM 3.  The only F making v_c exactly flat is F = C r ln(r_*/r),")
    say("   C = v_f^2/(G M).  Two bounds on the radial stretch it can cover:")
    say("   (a) unconstrained, apex of F on the ceiling  : r2/r1 from the two")
    say("       roots of C r ln(r_*/r) = 1;")
    say("   (b) with F(r1) = 1 and F'(r1) = 0, which is what an inner")
    say("       rotation curve fitted by baryons alone demands, then")
    say("       v_f^2 = G M F(r1)/r1 <= G M F(r2)/r2 gives r2/r1 <= sup F.")
    t3 = []
    for alpha in [0.3, 1.0, 3.0, 10.0, 30.0]:
        wide, mono = NK.flat_window(alpha, "F1_poly")
        # direct verification that F = C r ln(r_*/r) is exactly flat
        Fmax = 1.0 + alpha
        r_star = math.e * Fmax          # with C = 1 and G M = 1
        rv = np.geomspace(1e-3 * r_star, 0.999 * r_star, 400)
        Fv = rv * np.log(r_star / rv)
        v2 = Fv / rv - (np.log(r_star / rv) - 1.0)     # = C = 1 identically
        t3.append(dict(alpha=alpha, window_unconstrained=float(wide),
                       window_monotone_from_apex=float(mono),
                       window_constrained_bound=float(Fmax),
                       flat_identity_maxdev=float(np.max(np.abs(v2 - 1.0)))))
        say(f"   alpha = {alpha:<5.1f} sup F = {Fmax:<6.2f} "
            f"r2/r1 (unconstrained) = {wide:8.2f}   "
            f"r2/r1 (F'(r1)=0) <= {Fmax:6.2f}   "
            f"|v^2 - const| = {t3[-1]['flat_identity_maxdev']:.1e}")
    out["theorem3_flat_window"] = t3
    say("   A decade of exactly flat curve therefore needs alpha >= 9 under")
    say("   the physical constraint, or alpha ~ 1 without it.")
    RES["G4_rotation_curves"] = out


# ==========================================================================
def _outer_metrics(fld, gal, Fname, alpha, beta, p, rlo=2.0, rhi=20.0):
    """Rotation-curve shape over [rlo, rhi] disk scale lengths."""
    rr = np.geomspace(0.5 * gal.rd, rhi * gal.rd, 26)
    v2, _, _ = NK.spherical_vcirc_spline(fld, rr, Fname=Fname, alpha=alpha,
                                         beta=beta, p=p, use_gpu=GPU)
    if np.min(v2) <= 0:
        return dict(repulsive=True, slope=float("nan"), vmax=float("nan"),
                    r_vmax=float("nan"))
    v = np.sqrt(v2)
    sl = np.gradient(np.log(v), np.log(rr))
    sel = rr >= rlo * gal.rd
    i = int(np.argmax(v))
    return dict(repulsive=False, slope=float(np.mean(sl[sel])),
                vmax=float(v[i]), r_vmax=float(rr[i]))


def g4b_global_parameter_screen():
    head("G4b  Can ONE global parameter set flatten the whole ladder?")
    out = {}
    # solar-system filter, from the shared 1-D solar-neighbourhood check
    def ss_ok(qdef, kw, alpha, p, fam="F1_poly", beta=0.0):
        return solar_check(qdef, kw, alpha, p, Fname=fam,
                           beta=beta)["passes"]

    grid = []
    for qdef, extra in [("delta", [dict(L_s=0.0), dict(L_s=0.3), dict(L_s=2.0)]),
                        ("smooth", [dict(m=0.25), dict(m=0.5), dict(m=1.0),
                                    dict(m=2.0)]),
                        ("screen", [dict(L_q=2.0), dict(L_q=10.0),
                                    dict(L_q=50.0)])]:
        for ex in extra:
            for rho_ref in [1e3, 1e4, 1e5, 3e5, 1e6]:
                grid.append((qdef, dict(rho_ref=rho_ref, **ex)))
    fams = [("F1_poly", 0.0), ("F2_exp", 0.0), ("F3_pade", 1.0),
            ("F4_tidal", 0.3)]
    alphas = [0.3, 1.0, 3.0, 10.0]
    ps = [0.5, 1.0, 2.0]

    t0 = time.time()
    rows = []
    fields = {}
    for gi, gal in enumerate(MO.GALAXY_LADDER):
        for qdef, kw in grid:
            fields[(gi, qdef, tuple(sorted(kw.items())))] = \
                MO.build_field(gal, qdef, **kw)
    say(f"built {len(fields)} q fields")
    for qdef, kw in grid:
        for fam, beta in fams:
            for alpha in alphas:
                for p in ps:
                    if not ss_ok(qdef, kw, alpha, p, fam, beta):
                        continue
                    met = []
                    for gi, gal in enumerate(MO.GALAXY_LADDER):
                        f = fields[(gi, qdef, tuple(sorted(kw.items())))]
                        met.append(_outer_metrics(f, gal, fam, alpha, beta, p))
                    rep = any(m["repulsive"] for m in met)
                    sl = np.array([m["slope"] for m in met])
                    rows.append(dict(qdef=qdef, **{k: float(v) for k, v
                                                   in kw.items()},
                                     family=fam, alpha=alpha, beta=beta, p=p,
                                     repulsive=bool(rep),
                                     rms_slope=float("nan") if rep
                                     else float(np.sqrt(np.mean(sl ** 2))),
                                     slopes=[None if rep else float(x)
                                             for x in sl],
                                     vmax=[m["vmax"] for m in met]))
    dt = time.time() - t0
    say(f"{len(rows)} solar-system-allowed configurations screened in "
        f"{dt:.1f} s ({dt / max(len(rows), 1) * 1000:.0f} ms each)")
    nrep = sum(1 for r in rows if r["repulsive"])
    say(f"configurations with a REPULSIVE shell (v_c^2 < 0 somewhere): "
        f"{nrep} of {len(rows)}  ({100 * nrep / max(len(rows), 1):.1f}%)")
    out["n_configs"] = len(rows)
    out["n_repulsive"] = nrep
    byq = {}
    for qd in ("delta", "smooth", "screen"):
        sub = [r for r in rows if r["qdef"] == qd]
        if sub:
            byq[qd] = dict(n=len(sub),
                           n_repulsive=sum(1 for r in sub if r["repulsive"]),
                           frac=float(np.mean([r["repulsive"] for r in sub])))
            say(f"   {qd:<7s}: {byq[qd]['n_repulsive']:>4d} / "
                f"{byq[qd]['n']:<4d} repulsive "
                f"({100 * byq[qd]['frac']:.0f}%)")
    out["repulsive_by_qdef"] = byq
    say("   the clipped delta form switches almost like a step, so it drives")
    say("   dlnF/dlnr past 1 and manufactures repulsive shells; the smooth and")
    say("   screened forms avoid that but reacquire a solar-system constraint.")
    good = [r for r in rows if not r["repulsive"]]
    good.sort(key=lambda r: r["rms_slope"])
    out["best_10"] = good[:10]
    say("")
    say("best 10 by RMS outer log-slope of v_c across the six-galaxy ladder")
    say("(0 = flat, -0.5 = Keplerian; the Newtonian control is the last row):")
    say("   rms   qdef    rho_ref  extra          family    alpha p    slopes")
    for r in good[:10]:
        ex = {k: v for k, v in r.items() if k in ("m", "L_s", "L_q")}
        say(f"   {r['rms_slope']:.3f} {r['qdef']:<7s} {r['rho_ref']:<8.0e} "
            f"{str(ex):<14s} {r['family']:<9s} {r['alpha']:<5.1f} "
            f"{r['p']:<4.1f}"
            + " ".join(f"{x:+.2f}" for x in r["slopes"]))
    ctrl = []
    for gal in MO.GALAXY_LADDER:
        f = MO.build_field(gal, "zero")
        ctrl.append(_outer_metrics(f, gal, "F1_poly", 0.0, 0.0, 1.0))
    out["newton_control"] = [c["slope"] for c in ctrl]
    out["newton_rms"] = float(np.sqrt(np.mean(
        np.array([c["slope"] for c in ctrl]) ** 2)))
    say(f"   {out['newton_rms']:.3f} NEWTON  (alpha = 0)                      "
        f"                    "
        + " ".join(f"{c['slope']:+.2f}" for c in ctrl))
    say("   ladder order: dwarf_LSB dwarf_HSB LSB_large spiral_mid MW_like "
        "massive")

    # monotone-invariance check: does the headline statistic move with alpha?
    say("")
    say("MONOTONE-INVARIANT-STATISTIC CHECK.  dS/dalpha must not be zero.")
    ref = good[0]
    kwr = {k: v for k, v in ref.items() if k in ("rho_ref", "m", "L_s", "L_q")}
    spread = []
    for a in [0.1, 0.3, 1.0, 3.0, 10.0, 30.0]:
        met = []
        for gal in MO.GALAXY_LADDER:
            f = MO.build_field(gal, ref["qdef"], **kwr)
            met.append(_outer_metrics(f, gal, ref["family"], a, ref["beta"],
                                      ref["p"]))
        sl = np.array([m["slope"] for m in met])
        spread.append(dict(alpha=a, rms_slope=None if any(
            m["repulsive"] for m in met) else float(np.sqrt(np.mean(sl ** 2)))))
    out["alpha_sensitivity"] = spread
    say("   alpha : " + " ".join(f"{s['alpha']:>7.2f}" for s in spread))
    say("   S     : " + " ".join(
        ("  repul" if s["rms_slope"] is None else f"{s['rms_slope']:7.3f}")
        for s in spread))
    vals = [s["rms_slope"] for s in spread if s["rms_slope"] is not None]
    out["S_spread"] = float(max(vals) - min(vals))
    say(f"   spread of S over three decades of alpha = {out['S_spread']:.3f} "
        f"-- the statistic is not degenerate.")
    RES["G4b_global_screen"] = out
    return good


# ==========================================================================
def g4c_btfr(good):
    head("G4c  The baryonic Tully-Fisher relation the family predicts")
    out = {}
    say("SHARED-DENOMINATOR CHECK.  The BTFR residual is regressed against")
    say("   log r_d, NOT against the central surface density Sigma_0 =")
    say("   M/(2 pi r_d^2), because Sigma_0 carries the same M that sits on")
    say("   the abscissa and would manufacture a correlation from nothing.")
    say("   r_d is an independent input.")
    Mb = np.array([g.Mtot for g in MO.GALAXY_LADDER])
    rd = np.array([g.rd for g in MO.GALAXY_LADDER])
    rows = []
    cases = [("newton (alpha = 0)", "zero", {}, "F1_poly", 0.0, 0.0, 1.0)]
    for r in good[:3]:
        kwr = {k: v for k, v in r.items()
               if k in ("rho_ref", "m", "L_s", "L_q")}
        cases.append((f"{r['qdef']}|{r['family']}|a={r['alpha']}|p={r['p']}",
                      r["qdef"], kwr, r["family"], r["alpha"], r["beta"],
                      r["p"]))
    for lab, qdef, kwr, fam, alpha, beta, p in cases:
        vmax = []
        for gal in MO.GALAXY_LADDER:
            f = MO.build_field(gal, qdef, **kwr)
            vmax.append(_outer_metrics(f, gal, fam, alpha, beta, p)["vmax"])
        vmax = np.array(vmax)
        x = np.log10(vmax)
        y = np.log10(Mb)
        A = np.vstack([x, np.ones_like(x)]).T
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        res = y - A @ coef
        B = np.vstack([np.log10(rd), np.ones_like(x)]).T
        c2, *_ = np.linalg.lstsq(B, res, rcond=None)
        rows.append(dict(case=lab, slope=float(coef[0]),
                         scatter_dex=float(np.std(res, ddof=2)),
                         resid_vs_logrd_slope=float(c2[0]),
                         vmax=vmax.tolist()))
        say(f"   {lab:<40s} slope = {coef[0]:5.2f}  scatter = "
            f"{np.std(res, ddof=2):.3f} dex  d(resid)/dlog r_d = "
            f"{c2[0]:+.2f}")
    out["cases"] = rows
    out["observed"] = dict(slope=3.85, slope_err=0.09, scatter_dex=0.10,
                           resid_vs_size="consistent with zero",
                           source="Lelli, McGaugh & Schombert 2016 SPARC "
                                  "BTFR, quoted from the literature and not "
                                  "refitted here")
    say("")
    say("   observed SPARC BTFR: slope 3.85 +/- 0.09, scatter 0.10 dex, and no")
    say("   residual dependence on galaxy size or surface brightness.")
    RES["G4c_btfr"] = out


# ==========================================================================
def g4d_sparc_required_F():
    head("G4d  What F(r) the SPARC TRAIN rotation curves actually demand")
    out = {}
    sys.path.insert(0, GLAB)
    import data as SP
    gals = SP.ingest(verbose=False)
    SP.stratified_split(gals, verbose=False)
    train = [g for g in gals if g.split == "train"]
    say(f"SPARC galaxies after the frozen cuts : {len(gals)}")
    say(f"TRAIN split used here                : {len(train)}")
    say("validation and blind splits are NOT loaded, NOT read and NOT used.")
    say("Nothing is fitted: this is a forward inversion of each curve.")
    say("")
    say("For a source seen from outside, Phi = -G M_b F(r)/r is EXACT, so")
    say("   F(r) = -r Phi_req(r) / (G M_b),  "
        "Phi_req(r) = -Int_r^inf v_obs^2 dln r'.")
    say("The tail beyond the last measured point is taken Keplerian, which is")
    say("what this family forces anyway (Theorem 1); assuming instead that the")
    say("curve stays flat for another factor of two in radius raises F_req by")
    say("a further r ln 2 v_f^2 / (G M_b) and is reported as a sensitivity.")

    rows = []
    for g in train:
        R = g.R0
        Vb2 = np.abs(g.Vgas) * g.Vgas + 0.5 * g.Vdisk ** 2 + 0.7 * g.Vbul ** 2
        v2 = g.Vobs0 ** 2
        Mb = g.Mb
        if Mb <= 0 or len(R) < 5:
            continue
        lnr = np.log(R)
        # Phi(r) = -[ Int_r^rlast v^2 dlnr + v_last^2 ]  (Keplerian tail)
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (v2[1:] + v2[:-1]) * np.diff(lnr))])
        Phi = -((cum[-1] - cum) + v2[-1])
        F = -R * Phi / (NK.G * Mb)
        Phi2 = -((cum[-1] - cum) + v2[-1] * (1.0 + math.log(2.0)))
        F2 = -R * Phi2 / (NK.G * Mb)
        s = np.gradient(np.log(np.maximum(F, 1e-30)), lnr)
        far = R >= 2.0 * max(g.Rdisk, 1e-3)
        if far.sum() < 3:
            far = R >= np.median(R)
        rows.append(dict(name=g.name, Mb=float(Mb), Rdisk=float(g.Rdisk),
                         Rlast=float(R[-1]), Rlast_over_Rd=float(
                             R[-1] / max(g.Rdisk, 1e-6)),
                         F_min_far=float(np.min(F[far])),
                         F_max=float(np.max(F)),
                         F_max_flat_tail=float(np.max(F2)),
                         s_max_far=float(np.max(s[far])),
                         Vflat=float(g.Vflat), fgas=float(g.fgas)))
    out["n_used"] = len(rows)
    Fmax = np.array([r["F_max"] for r in rows])
    Fmax2 = np.array([r["F_max_flat_tail"] for r in rows])
    smax = np.array([r["s_max_far"] for r in rows])
    Fminfar = np.array([r["F_min_far"] for r in rows])
    out["alpha_required_global"] = float(np.max(Fmax) - 1.0)
    out["alpha_required_flat_tail"] = float(np.max(Fmax2) - 1.0)
    out["F_max_percentiles"] = {str(q): float(np.percentile(Fmax, q))
                                for q in (5, 25, 50, 75, 95, 100)}
    out["s_max_percentiles"] = {str(q): float(np.percentile(smax, q))
                                for q in (5, 50, 95, 100)}
    out["frac_F_below_1_in_far_region"] = float(np.mean(Fminfar < 1.0))
    say("")
    say(f"galaxies inverted                    : {len(rows)}")
    say(f"max F over the sample                : {np.max(Fmax):.2f}  "
        f"=> one GLOBAL alpha must be at least {np.max(Fmax) - 1:.2f}")
    say(f"   with a flat tail for another x2   : {np.max(Fmax2) - 1:.2f}")
    say("F_max percentiles  5/25/50/75/95/100  : "
        + " / ".join(f"{np.percentile(Fmax, q):.2f}"
                     for q in (5, 25, 50, 75, 95, 100)))
    say("max dlnF/dlnr percentiles 5/50/95/100 : "
        + " / ".join(f"{np.percentile(smax, q):.3f}" for q in (5, 50, 95, 100)))
    say("   (dlnF/dlnr = 1 is the repulsion threshold of Theorem 2; the data")
    say("    sit below it by construction, but the margin is the fine-tuning)")
    say(f"fraction of galaxies needing F < 1 anywhere beyond 2 R_d : "
        f"{np.mean(Fminfar < 1.0) * 100:.1f}%")
    say("   F < 1 cannot be produced by any member with alpha > 0, since")
    say("   F = 1 + alpha qbar^p >= 1; it would require alpha < 0, which")
    say("   makes gravity weaker in voids, the wrong sign for the rest.")

    # the spread in F_max at fixed velocity: the family needs one alpha but
    # the data demand a range
    v = np.array([r["Vflat"] for r in rows])
    out["Fmax_vs_Vflat"] = dict(
        logFmax_mean=float(np.mean(np.log10(Fmax))),
        logFmax_sd=float(np.std(np.log10(Fmax))),
        corr_logFmax_logVflat=float(np.corrcoef(np.log10(Fmax),
                                                np.log10(v))[0, 1]))
    say(f"spread of log10 F_max across the sample : "
        f"{np.std(np.log10(Fmax)):.3f} dex")
    say(f"corr(log F_max, log V_flat)             : "
        f"{np.corrcoef(np.log10(Fmax), np.log10(v))[0, 1]:+.3f}")
    out["per_galaxy"] = rows
    RES["G4d_sparc_required_F"] = out


# ==========================================================================
def _sparc_train():
    sys.path.insert(0, GLAB)
    import data as SP
    gals = SP.ingest(verbose=False)
    SP.stratified_split(gals, verbose=False)
    return [g for g in gals if g.split == "train"]


def _required_F(g):
    """F_req(r) demanded by one observed curve, and the radii it applies at."""
    R = g.R0
    v2 = g.Vobs0 ** 2
    lnr = np.log(R)
    cum = np.concatenate([[0.0], np.cumsum(
        0.5 * (v2[1:] + v2[:-1]) * np.diff(lnr))])
    Phi = -((cum[-1] - cum) + v2[-1])
    return R, -R * Phi / (NK.G * g.Mb)


def g4e_sparc_forward():
    head("G4e  Forward test: one global parameter set against SPARC TRAIN")
    out = {}
    say("Each TRAIN galaxy gets an EQUIVALENT SPHERICAL mass distribution")
    say("defined by M(<R) = R V_bar^2 / G at its own tabulated radii, with")
    say("V_bar^2 = |V_gas| V_gas + 0.5 V_disk^2 + 0.7 V_bul^2.  That makes the")
    say("model's NEWTONIAN curve identical to the tabulated one by")
    say("construction, so the baryon-geometry error is removed rather than")
    say("estimated.  An exponential-sphere model built from R_disk and total")
    say("masses instead -- the first version of this test -- is wrong by")
    say("0.32 dex rms in v_bar^2, which is twice the residual being measured,")
    say("and that version is not reportable.")
    say("The price is stated: the equivalent spherical DENSITY is not the true")
    say("3-D density, so the q field built from it is biased towards larger q.")
    say("")
    say("The q field is then computed with GLOBAL parameters only and the")
    say("kernel's F_eff(r) = -r Phi/(G M_tot) is compared with the F_req(r)")
    say("the observed curve demands, at the measured radii beyond 2 R_disk.")
    say("Statistic: rms of log10(F_eff / F_req).  No per-galaxy freedom.")
    train = _sparc_train()
    usable = [g for g in train
              if g.Mb > 0 and g.Rdisk > 0 and np.sum(g.R0 >= 2 * g.Rdisk) >= 3]
    say(f"TRAIN galaxies {len(train)}, usable with >=3 points beyond 2 R_d : "
        f"{len(usable)}")
    rl = np.array([g.R0[-1] / g.Rdisk for g in usable])
    say(f"R_last / R_disk : median {np.median(rl):.1f}, "
        f"5-95 pct {np.percentile(rl, 5):.1f}-{np.percentile(rl, 95):.1f}")

    prof = {}
    req = {}
    for g in usable:
        Vb2 = (np.abs(g.Vgas) * g.Vgas + 0.5 * g.Vdisk ** 2
               + 0.7 * g.Vbul ** 2)
        r, rho, Mr, rfun, Mfun, Mtot = MO.sparc_equivalent_sphere(
            g.R0, Vb2, g.Mb, max(g.Rdisk, 0.2))
        prof[g.name] = (r, rho, Mr, rfun, Mfun, Mtot)
        # required F, normalised by the SAME total mass the model carries
        v2 = g.Vobs0 ** 2
        lnr = np.log(g.R0)
        cum = np.concatenate([[0.0], np.cumsum(
            0.5 * (v2[1:] + v2[:-1]) * np.diff(lnr))])
        Phi = -((cum[-1] - cum) + v2[-1])
        F = -g.R0 * Phi / (NK.G * Mtot)
        m = g.R0 >= 2 * g.Rdisk
        req[g.name] = (g.R0[m], F[m])

    # control: the model's own Newtonian curve against the tabulated one
    ctl = []
    for g in usable:
        r, rho, Mr, rfun, Mfun, Mtot = prof[g.name]
        m = g.R0 >= 2 * g.Rdisk
        Vb2 = (np.abs(g.Vgas) * g.Vgas + 0.5 * g.Vdisk ** 2
               + 0.7 * g.Vbul ** 2)[m]
        mod = NK.G * Mfun(g.R0[m]) / g.R0[m]
        ok = Vb2 > 0
        if ok.sum():
            ctl.append(np.log10(mod[ok] / Vb2[ok]))
    allb = np.concatenate(ctl)
    out["baryon_model_control"] = dict(
        rms_dex=float(np.sqrt(np.mean(allb ** 2))),
        bias_dex=float(np.mean(allb)),
        note="rms log10 of (model v_bar^2)/(SPARC v_bar^2) beyond 2 R_disk; "
             "exact by construction for the equivalent spherical model")
    say(f"BARYON-MODEL CONTROL: model v_bar^2 vs tabulated v_bar^2 = "
        f"{out['baryon_model_control']['rms_dex']:.4f} dex rms "
        f"(bias {out['baryon_model_control']['bias_dex']:+.4f}) -- exact by "
        f"construction.")

    configs = [
        ("delta  rho_ref=1e4 L_s=0", "delta", dict(rho_ref=1e4, L_s=0.0)),
        ("delta  rho_ref=1e5 L_s=0", "delta", dict(rho_ref=1e5, L_s=0.0)),
        ("delta  rho_ref=1e6 L_s=0", "delta", dict(rho_ref=1e6, L_s=0.0)),
        ("smooth rho_ref=1e4 m=1", "smooth", dict(rho_ref=1e4, m=1.0)),
        ("smooth rho_ref=1e5 m=1", "smooth", dict(rho_ref=1e5, m=1.0)),
        ("smooth rho_ref=1e5 m=2", "smooth", dict(rho_ref=1e5, m=2.0)),
        ("smooth rho_ref=1e6 m=0.5", "smooth", dict(rho_ref=1e6, m=0.5)),
        ("screen rho_ref=1e5 L_q=2", "screen", dict(rho_ref=1e5, L_q=2.0)),
        ("screen rho_ref=1e5 L_q=10", "screen", dict(rho_ref=1e5, L_q=10.0)),
        ("screen rho_ref=1e6 L_q=2", "screen", dict(rho_ref=1e6, L_q=2.0)),
        ("screen rho_ref=1e6 L_q=10", "screen", dict(rho_ref=1e6, L_q=10.0)),
    ]
    rows = []
    t0 = time.time()
    for lab, qdef, kw in configs:
        fields = {}
        for g in usable:
            r, rho, Mr, rfun, Mfun, Mtot = prof[g.name]
            fields[g.name] = MO.build_field_from_profile(
                r, rho, Mr, rfun, Mfun, qdef=qdef, label=g.name, **kw)
        for fam, beta in [("F1_poly", 0.0), ("F2_exp", 0.0),
                          ("F3_pade", 1.0)]:
            for alpha in [1.0, 3.0, 10.0, 30.0]:
                for p in [0.5, 1.0, 2.0]:
                    resid, bad = [], 0
                    for g in usable:
                        R, Fr = req[g.name]
                        Fe = NK.spherical_F_effective(
                            fields[g.name], R, Fname=fam, alpha=alpha,
                            beta=beta, p=p, use_gpu=GPU)
                        if np.any(Fe <= 0):
                            bad += 1
                            continue
                        resid.append(np.log10(Fe / Fr))
                    if not resid:
                        continue
                    allr = np.concatenate(resid)
                    per = np.array([np.mean(x) for x in resid])
                    sc = solar_check(qdef, kw, alpha, p, Fname=fam, beta=beta)
                    rows.append(dict(
                        config=lab, qdef=qdef, family=fam, alpha=alpha,
                        beta=beta, p=p, n_bad=bad,
                        rms_dex=float(np.sqrt(np.mean(allr ** 2))),
                        bias_dex=float(np.mean(allr)),
                        galaxy_scatter_dex=float(np.std(per)),
                        n_points=int(len(allr)),
                        q_sun=sc["q_sun"], eps_1AU=sc["eps_1AU"],
                        F_local=sc["F_local"], isl_ok=sc["passes"],
                        oort_ok=sc["passes_oort"]))
    say(f"{len(rows)} global parameter sets evaluated on {len(usable)} "
        f"galaxies in {time.time() - t0:.1f} s")
    rows.sort(key=lambda r_: r_["rms_dex"])
    out["all"] = rows
    out["best_15"] = rows[:15]
    say("")
    say("best 15 by rms log10(F_eff / F_req).  0.00 dex would be a perfect")
    say("global reproduction of every SPARC train rotation curve.")
    say("   rms   bias  gal-sc  config                     family    alpha p "
        "  F_local  ISL Oort")
    for r_ in rows[:15]:
        say(f"   {r_['rms_dex']:.3f} {r_['bias_dex']:+.3f} "
            f"{r_['galaxy_scatter_dex']:.3f}   {r_['config']:<26s} "
            f"{r_['family']:<9s} {r_['alpha']:<5.1f} {r_['p']:.1f} "
            f"{r_['F_local']:8.3f}  {'ok ' if r_['isl_ok'] else 'BAD'} "
            f"{'ok' if r_['oort_ok'] else 'BAD'}")
    ctrl = [np.log10(1.0 / req[g.name][1]) for g in usable]
    allc = np.concatenate(ctrl)
    out["newton_control"] = dict(
        rms_dex=float(np.sqrt(np.mean(allc ** 2))),
        bias_dex=float(np.mean(allc)),
        galaxy_scatter_dex=float(np.std([np.mean(x) for x in ctrl])))
    say(f"   {out['newton_control']['rms_dex']:.3f} "
        f"{out['newton_control']['bias_dex']:+.3f} "
        f"{out['newton_control']['galaxy_scatter_dex']:.3f}   "
        f"NEWTON, F = 1 everywhere")
    best = rows[0]
    surv = [r_ for r_ in rows if r_["isl_ok"] and r_["oort_ok"]]
    out["improvement_dex"] = float(out["newton_control"]["rms_dex"]
                                   - best["rms_dex"])
    out["n_galaxies"] = len(usable)
    out["n_passing_local"] = len(surv)
    out["best_passing_local"] = surv[0] if surv else None
    say("")
    say(f"best global set improves on Newton by "
        f"{out['improvement_dex']:.3f} dex in rms, leaving "
        f"{best['rms_dex']:.3f} dex of residual --")
    say(f"a factor {10 ** best['rms_dex']:.2f} typical error in F, with "
        f"{best['galaxy_scatter_dex']:.3f} dex of galaxy-to-galaxy scatter")
    say("that no global parameter can absorb (the RAR sits at about 0.11 dex).")
    if surv:
        b = surv[0]
        say(f"best set that ALSO passes the inverse-square-law bound and the")
        say(f"Oort window: {b['config']} {b['family']} alpha={b['alpha']} "
            f"p={b['p']}, rms {b['rms_dex']:.3f} dex, F_local "
            f"{b['F_local']:.3f}")
    else:
        say("NO global set passes the inverse-square-law bound and the Oort "
            "window simultaneously.")
    RES["G4e_sparc_forward"] = out


# ==========================================================================
def g5_numerical_gates():
    head("G5  Numerical gates: resolution, domain size, label permutation")
    out = {}
    gal = MO.GALAXY_LADDER[4]
    fld = MO.build_field(gal, "smooth", rho_ref=1e5, m=1.0)
    rr = np.geomspace(1.0, 60.0, 12)
    ref = NK.spherical_potential_batch(fld, rr, Fname="F1_poly", alpha=3.0,
                                       p=1.0, n_D=96, n_s=48, n_gl=12,
                                       dlnr_max=0.10)

    say("resolution convergence, referenced to n_D=96 n_s=48 n_gl=12 "
        "dlnr_max=0.10:")
    conv = []
    for nD, ns, ngl, dl in [(8, 4, 4, 0.7), (16, 8, 6, 0.5), (32, 12, 8, 0.35),
                            (48, 24, 8, 0.20), (64, 32, 10, 0.15)]:
        phi = NK.spherical_potential_batch(fld, rr, Fname="F1_poly", alpha=3.0,
                                           p=1.0, n_D=nD, n_s=ns, n_gl=ngl,
                                           dlnr_max=dl)
        e = float(np.max(np.abs(phi / ref - 1)))
        conv.append(dict(n_D=nD, n_s=ns, n_gl=ngl, dlnr_max=dl, rel_err=e))
        say(f"   n_D={nD:<3d} n_s={ns:<3d} n_gl={ngl:<3d} dlnr={dl:<5.2f} "
            f"max rel err = {e:.3e}")
    out["resolution"] = conv
    out["production_setting_rel_err"] = conv[2]["rel_err"]

    say("")
    say("domain-size convergence (outer radius of the source integral):")
    dom = []
    base = None
    for rhi in [3e2, 1e3, 3e3, 1e4, 3e4]:
        phi = NK.spherical_potential_batch(fld, rr, Fname="F1_poly", alpha=3.0,
                                           p=1.0, r_hi=rhi)
        if base is None:
            base = phi
        dom.append(dict(r_hi_kpc=rhi,
                        rel_change=float(np.max(np.abs(phi / base - 1)))))
        say(f"   r_hi = {rhi:>8.0f} kpc   max |dPhi/Phi| vs r_hi = 300 kpc : "
            f"{dom[-1]['rel_change']:.3e}")
    out["domain"] = dom
    out["domain_converged_pct"] = float(
        abs(dom[-1]["rel_change"] - dom[-2]["rel_change"]) * 100)

    say("")
    say("SOURCE-LABEL PERMUTATION INVARIANCE (3-D direct sum).")
    rng = np.random.default_rng(11)
    n = 20
    h, ax, X, Y, Z = NK.grid3d(n, 400.0)
    rho = 1e10 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * 60.0 ** 2))
    qg = NK.q_from_smooth(rho + NK.RHO_BAR_B, 1e4)
    Ps = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    Ms = rho.ravel() * h ** 3
    Pf = rng.uniform(-150, 150, (24, 3))
    a = NK.direct_potential_3d(Pf, Ps, Ms, qg, ax, alpha=2.0, p=1.0, n_s=12,
                               soft=0.5 * h, use_gpu=GPU)
    perm = rng.permutation(len(Ms))
    b = NK.direct_potential_3d(Pf, Ps[perm], Ms[perm], qg, ax, alpha=2.0,
                               p=1.0, n_s=12, soft=0.5 * h, use_gpu=GPU)
    out["label_permutation_max_relerr"] = float(np.max(np.abs(b / a - 1)))
    say(f"   permuting all {len(Ms)} source cells changes Phi by "
        f"{out['label_permutation_max_relerr']:.3e}")

    if GPU:
        c = NK.direct_potential_3d(Pf, Ps, Ms, qg, ax, alpha=2.0, p=1.0,
                                   n_s=12, soft=0.5 * h, use_gpu=False)
        out["cpu_gpu_max_relerr"] = float(np.max(np.abs(c / a - 1)))
        say(f"   CPU vs GPU direct sum agree to "
            f"{out['cpu_gpu_max_relerr']:.3e}")

    # 3-D Newtonian recovery of the same direct sum
    nn = NK.direct_potential_3d(Pf, Ps, Ms, qg * 0, ax, alpha=0.0, p=1.0,
                                n_s=12, soft=0.5 * h, use_gpu=GPU)
    Dm = np.sqrt(np.sum((Ps[None, :, :] - Pf[:, None, :]) ** 2, axis=2)
                 + (0.5 * h) ** 2)
    nn_ref = -NK.G * np.sum(Ms[None, :] / Dm, axis=1)
    out["direct3d_newton_relerr"] = float(np.max(np.abs(nn / nn_ref - 1)))
    say(f"   3-D direct sum at alpha = 0 vs a plain Newtonian sum : "
        f"{out['direct3d_newton_relerr']:.3e}")
    RES["G5_numerical_gates"] = out


# ==========================================================================
def g6_cost_and_acceleration():
    head("G6  Cost of the double integral, and the price of accelerating it")
    out = {}
    say("Cost model.  N_f field points, N_s source cells, n_s path samples:")
    say("   direct  O(N_f N_s n_s) trilinear interpolations")
    say("   grid of n^3 cells evaluated at every cell -> O(n^6 n_s)")
    say("   n = 64 : 6.9e10 pair-samples;  n = 128 : 4.4e12.")
    say("The spherical and axisymmetric reductions used for the rotation-curve")
    say("work collapse this to O(N_f N_r' N_D n_s) ~ 1e7, which is why the")
    say("parameter screen is possible at all.  For genuinely 3-D configurations")
    say("two accelerations are implemented and priced below.")

    rng = np.random.default_rng(5)
    timings = []
    for n in [10, 12, 16, 20, 24]:
        h, ax, X, Y, Z = NK.grid3d(n, 400.0)
        rho = 1e10 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * 60.0 ** 2))
        qg = NK.q_from_smooth(rho + NK.RHO_BAR_B, 1e4)
        Ps = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
        Ms = rho.ravel() * h ** 3
        # the CPU sum is O(n^6 n_s); above n = 16 it is minutes, so it is
        # timed only where that is quick and the rate is what matters
        devs = [False, True] if (GPU and n <= 16) else ([True] if GPU
                                                        else [False])
        for gpu in devs:
            t = time.time()
            NK.direct_potential_3d(Ps, Ps, Ms, qg, ax, alpha=2.0, p=1.0,
                                   n_s=16, soft=0.5 * h, use_gpu=gpu,
                                   chunk=2048 if gpu else 512)
            dt = time.time() - t
            timings.append(dict(n=n, device="gpu" if gpu else "cpu",
                                pairs=float(n ** 6), seconds=dt,
                                pair_samples_per_s=float(n ** 6 * 16 / dt)))
            say(f"   n = {n:<3d} ({n ** 3:>6d} cells) {'GPU' if gpu else 'CPU'} "
                f": {dt:8.3f} s  -> {n ** 6 * 16 / dt:.3e} pair-samples/s")
    out["direct_timings"] = timings
    fast = max((t for t in timings), key=lambda t: t["pair_samples_per_s"])
    out["best_rate"] = fast
    for n in (64, 128):
        say(f"   extrapolated all-pairs cost at n = {n}: "
            f"{n ** 6 * 16 / fast['pair_samples_per_s']:.1f} s on the "
            f"{fast['device'].upper()}")
    out["extrapolated_n64_s"] = float(64 ** 6 * 16
                                      / fast["pair_samples_per_s"])
    out["extrapolated_n128_s"] = float(128 ** 6 * 16
                                       / fast["pair_samples_per_s"])

    # ---- accuracy price of the separable midpoint surrogate + SVD --------
    say("")
    say("ACCELERATION 1: replace the path average by the separable midpoint")
    say("   surrogate qbar ~ [q(x) + q(x')]/2, expand F(u,v) by SVD to rank R,")
    say("   and do each rank-1 term as one FFT convolution with 1/r.")
    say("   Cost drops from O(n^6 n_s) to O(R n^3 log n).")
    n = 24
    h, ax, X, Y, Z = NK.grid3d(n, 400.0)
    rho = 1e10 * np.exp(-(X ** 2 + Y ** 2 + Z ** 2) / (2 * 60.0 ** 2))
    qg = NK.q_from_smooth(rho + NK.RHO_BAR_B, 1e4)
    Ps = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    Ms = rho.ravel() * h ** 3
    exact = NK.direct_potential_3d(Ps, Ps, Ms, qg, ax, alpha=2.0, p=1.0,
                                   n_s=24, soft=0.5 * h, use_gpu=GPU,
                                   soft_mode="plateau")
    mid = NK.direct_potential_3d(Ps, Ps, Ms, qg, ax, alpha=2.0, p=1.0,
                                 n_s=24, soft=0.5 * h, use_gpu=GPU,
                                 qbar_mode="midpoint", soft_mode="plateau")
    surro = float(np.max(np.abs(mid / exact - 1)))
    out["midpoint_surrogate_max_relerr"] = surro
    out["midpoint_surrogate_rms_relerr"] = float(
        np.sqrt(np.mean((mid / exact - 1) ** 2)))
    say(f"   midpoint surrogate vs the exact path average : "
        f"max {surro:.3e}, rms {out['midpoint_surrogate_rms_relerr']:.3e}")
    ranks = []
    for R in [1, 2, 3, 4, 6, 8]:
        t = time.time()
        phi_lr, svd_rel = NK.lowrank_potential_3d(rho, qg, h, alpha=2.0, p=1.0,
                                                  rank=R)
        dt = time.time() - t
        e = float(np.max(np.abs(phi_lr.ravel() / mid - 1)))
        ranks.append(dict(rank=R, svd_tail=svd_rel, vs_midpoint_max_relerr=e,
                          seconds=dt))
        say(f"   rank {R:<2d}: SVD tail {svd_rel:.2e}, vs midpoint direct sum "
            f"{e:.3e}, {dt:.3f} s")
    out["lowrank"] = ranks
    say("   The direct sum here uses D = max(|x-x'|, h/2), exactly the kernel")
    say("   the FFT tabulates; with the Plummer convention instead the two")
    say("   disagree at 1.5e-2 for a reason that has nothing to do with the")
    say("   acceleration.  F1 with p = 1 is EXACTLY rank 2 in the midpoint")
    say("   surrogate, since 1 + alpha (u+v)/2 separates; other members need")
    say("   more, so the spectrum is tabulated:")
    spec = []
    for fam in ("F1_poly", "F2_exp", "F3_pade"):
        for pp in (0.5, 1.0, 2.0):
            _, _, S_, _ = NK.lowrank_factors(fam, 2.0, 1.0, pp, 12)
            S_ = S_ / S_[0]
            k = int(np.argmax(S_ < 1e-10)) if np.any(S_ < 1e-10) else len(S_)
            spec.append(dict(family=fam, p=pp, rank_for_1e10=k,
                             sv=[float(x) for x in S_[:8]]))
            say(f"      {fam:<9s} p = {pp:<4.1f} rank for 1e-10 = {k:<3d} "
                f"singular values " + " ".join(f"{x:.1e}" for x in S_[:5]))
    out["svd_spectra"] = spec
    say("")
    say("ACCELERATION 2 (used for every galaxy number in this report): the")
    say("   spherical reduction, which is EXACT -- the separation D is the")
    say("   inner integration variable, so the 1/|x-x'| singularity is removed")
    say("   analytically and no approximation is made at all.  Its accuracy is")
    say("   the quadrature accuracy reported in G5, ~1e-6 at production")
    say("   settings.")
    RES["G6_cost"] = out


# ==========================================================================
def g7_distinctive_predictions():
    head("G7  What only a nonlocal kernel predicts, and how big it is")
    out = {}
    say("The signature is that two sources at the SAME separation couple")
    say("differently when the material BETWEEN them differs.  No point-local")
    say("law -- MOND, the RAR, f(R), any mu(g) or mu(rho) -- can produce it,")
    say("because those depend only on fields evaluated at one point.")
    say("")
    say("But the signature and the galaxy application both draw on the SAME")
    say("single global density scale rho_ref, and they want opposite values.")
    say("This gate measures that tension rather than asserting it.")

    # ------------------------------------------------------------------
    #  three requirements, one global scale
    # ------------------------------------------------------------------
    rho_env = {"void_0.2mean": 0.2 * NK.RHO_BAR_B,
               "mean": NK.RHO_BAR_B,
               "filament_30mean": 30.0 * NK.RHO_BAR_B}
    mw = MO.GALAXY_LADDER[4]
    rows = []
    for qdef, kws in [("delta", [dict(L_s=0.0)]),
                      ("smooth", [dict(m=0.5), dict(m=1.0), dict(m=2.0)]),
                      ("screen", [dict(L_q=2.0), dict(L_q=10.0)])]:
        for kwx in kws:
            for rho_ref in [NK.RHO_BAR_B, 6.2e1, 6.2e2, 1e4, 1e5, 1e6]:
                kw = dict(rho_ref=rho_ref, **kwx)
                fld = MO.build_field(mw, qdef, **kw)
                for alpha, pp in [(1.0, 1.0), (3.0, 1.0), (10.0, 2.0)]:
                    sc = solar_check(qdef, kw, alpha, pp)
                    # efficacy = the mass discrepancy a Newtonian observer
                    # would infer at 25 kpc, M_dyn/M_b(<25 kpc).  Newton gives
                    # exactly 1, so this isolates the kernel from the fraction
                    # of the baryons that happen to lie inside 25 kpc.
                    rq = np.array([25.0])
                    v2k, _, _ = NK.spherical_vcirc_spline(
                        fld, rq, Fname="F1_poly", alpha=alpha, p=pp,
                        use_gpu=GPU)
                    F25 = float(rq[0] * v2k[0] / (NK.G * mw.Menc(rq)[0]))
                    qe = {}
                    for lab, rr in rho_env.items():
                        if qdef == "delta":
                            qe[lab] = float(NK.q_from_delta(rr, rho_ref))
                        elif qdef == "smooth":
                            qe[lab] = float(NK.q_from_smooth(
                                rr, rho_ref, m=kw.get("m", 1.0)))
                        else:
                            qe[lab] = float(np.clip(NK.q_source_Q3(
                                rr, 0.0, rho_ref), 0.0, 1.0 - 1e-12))
                    Fv = 1.0 + alpha * qe["void_0.2mean"] ** pp
                    Ff = 1.0 + alpha * qe["filament_30mean"] ** pp
                    contrast = Fv / Ff - 1.0
                    rows.append(dict(
                        qdef=qdef, rho_ref=float(rho_ref),
                        extra={k: float(v) for k, v in kwx.items()},
                        alpha=alpha, p=pp,
                        q_sun=sc["q_sun"], eps_1AU=sc["eps_1AU"],
                        F_local=sc["F_local"], isl_ok=sc["passes"],
                        oort_ok=sc["passes_oort"],
                        Mdyn_over_Mb_25kpc=F25, galaxy_ok=bool(F25 >= 2.0),
                        env_contrast=float(contrast),
                        signature_ok=bool(abs(contrast) >= 0.05)))
    out["three_way"] = rows
    def line(r_):
        allok = (r_["isl_ok"] and r_["oort_ok"] and r_["galaxy_ok"]
                 and r_["signature_ok"])
        say(f"   {r_['qdef']:<7s} {r_['rho_ref']:<8.2g} "
            f"{str(r_['extra']):<12s} {r_['alpha']:<5.1f} {r_['p']:.0f}   "
            f"{'ok ' if r_['isl_ok'] else 'BAD'}  "
            f"{'ok  ' if r_['oort_ok'] else 'BAD '}  "
            f"{r_['F_local']:7.2f}  {r_['Mdyn_over_Mb_25kpc']:9.3f}  "
            f"{r_['env_contrast']:+11.4f}   {'YES' if allok else '-'}")

    say("")
    say("   best on the GALAXY criterion (largest M_dyn/M_b at 25 kpc):")
    say("   qdef    rho_ref  extra        alpha p   ISL  Oort  F_local  "
        "Mdyn/Mb   void/fil     all4")
    for r_ in sorted(rows, key=lambda z: -z["Mdyn_over_Mb_25kpc"])[:8]:
        line(r_)
    say("   best on the NONLOCAL-SIGNATURE criterion (void/filament "
        "contrast):")
    for r_ in sorted(rows, key=lambda z: -z["env_contrast"])[:8]:
        line(r_)
    say("   best on the LOCAL criteria (inverse-square law and Oort):")
    for r_ in sorted(rows, key=lambda z: (not z["isl_ok"], not z["oort_ok"],
                                          -z["Mdyn_over_Mb_25kpc"]))[:8]:
        line(r_)
    for r_ in rows:
        if (r_["isl_ok"] and r_["oort_ok"] and r_["galaxy_ok"]
                and r_["signature_ok"]):
            line(r_)
    n_all = sum(1 for r_ in rows if r_["isl_ok"] and r_["oort_ok"]
                and r_["galaxy_ok"] and r_["signature_ok"])
    out["n_satisfying_all_three"] = n_all
    out["n_tested"] = len(rows)
    say("")
    say(f"   {n_all} of {len(rows)} parameter points satisfy all four at once:")
    say("   inverse-square law at 1 AU, the Oort local-dynamics window,")
    say("   M_dyn/M_b >= 2 at 25 kpc in a Milky-Way-like galaxy, and a")
    say("   >= 5 per cent void-versus-filament contrast in the coupling")
    say("   of a matched pair.")
    say("   The obstruction is arithmetic: rho_ref must sit near the cosmic")
    say(f"   mean {NK.RHO_BAR_B:.1f} Msun/kpc^3 for intergalactic environments to")
    say("   differ in q at all, but near 1e5-1e6 for q to rise across a")
    say("   rotation curve.  One global scale cannot be in both places.")

    # ------------------------------------------------------------------
    #  the one thing it does get right: a cluster-only excess in r/R500
    # ------------------------------------------------------------------
    say("")
    say("CLUSTER-ONLY EXCESS.  qbar(r) is the path average from the centre")
    say("out to r, so the modification switches on where a real fraction of")
    say("that segment lies below rho_ref.  With ONE global rho_ref that")
    say("happens beyond the last measured point of a galaxy rotation curve")
    say("and inside the weak-lensing range of a cluster.")
    rho_ref = 1e5
    tr = []
    for gal in MO.GALAXY_LADDER:
        fl = MO.build_field(gal, "smooth", rho_ref=rho_ref, m=1.0)
        rr = np.geomspace(0.5, 3000.0, 180)
        Fe = NK.spherical_F_effective(fl, rr, Fname="F1_poly", alpha=1.0,
                                      p=1.0, use_gpu=GPU)
        F0 = NK.spherical_F_effective(fl, rr, Fname="F1_poly", alpha=0.0,
                                      p=1.0, use_gpu=GPU)
        boost = Fe / F0
        i = int(np.argmax(boost > 1.1)) if np.any(boost > 1.1) else -1
        tr.append(dict(obj=gal.name, kind="galaxy", Mb=gal.Mtot,
                       scale_kpc=gal.rd,
                       r_turnon_kpc=float(rr[i]) if i >= 0 else None,
                       r_turnon_over_scale=float(rr[i] / gal.rd)
                       if i >= 0 else None))
    clu = MO.Cluster()
    rg = np.geomspace(1.0, 3.0e4, 1600)
    fclu = NK.SphericalField(r=rg, rho=clu.rho_pert(rg),
                             q=NK.q_from_smooth(clu.rho_full(rg), rho_ref),
                             rho_fun=clu.rho_pert, Menc_fun=clu.Menc)
    rc = np.geomspace(50.0, 6000.0, 90)
    v2c, _, _ = NK.spherical_vcirc_spline(fclu, rc, Fname="F1_poly", alpha=1.0,
                                          p=1.0, use_gpu=GPU)
    Mdyn = rc * v2c / NK.G
    ratio = Mdyn / clu.Menc(rc)
    # R500 from the OBSERVABLE route: M500 = M_gas(<R500) / f_gas with
    # f_gas = 0.13, then M500 = 500 rho_crit (4 pi/3) R500^3.  Defining it
    # instead from the model's own dynamical mass gives 130 kpc, because a
    # baryons-only cluster with alpha = 1 is nowhere near massive enough to
    # reach an overdensity of 500 -- which is itself the point.
    f_gas = 0.13
    lhs = clu.Menc(rc) / f_gas
    rhs = 500.0 * NK.RHO_CRIT * (4 * math.pi / 3) * rc ** 3
    j = int(np.argmin(np.abs(np.log(lhs / rhs))))
    r500 = float(rc[j])
    v2b, _, _ = NK.spherical_vcirc_spline(fclu, rc, Fname="F1_poly",
                                          alpha=0.0, p=1.0, use_gpu=GPU)
    boost_c = (rc * v2c) / (rc * v2b)
    i = int(np.argmax(boost_c > 1.1)) if np.any(boost_c > 1.1) else 0
    tr.append(dict(obj="cluster", kind="cluster", Mb=clu.Mtot, scale_kpc=r500,
                   r_turnon_kpc=float(rc[i]),
                   r_turnon_over_scale=float(rc[i] / r500)))
    out["turn_on"] = tr
    out["cluster_R500_kpc"] = r500
    out["cluster_M500_from_fgas"] = float(lhs[j])
    out["cluster_boost_profile"] = [
        dict(r_over_R500=float(x), boost=float(y))
        for x, y in zip(rc / r500, boost_c) if 0.1 <= x <= 4.0]
    for r_ in tr:
        say(f"   {r_['obj']:<11s} {r_['kind']:<8s} M_b = {r_['Mb']:.2e} "
            f"scale = {r_['scale_kpc']:7.1f} kpc   turn-on at "
            + (f"{r_['r_turnon_kpc']:8.1f} kpc = "
               f"{r_['r_turnon_over_scale']:6.2f} x scale"
               if r_["r_turnon_kpc"] else "never inside 3 Mpc"))
    say(f"   cluster R500 from M_gas / f_gas = {r500:.0f} kpc, "
        f"M500 = {lhs[j]:.2e} Msun")
    prof = [dict(r_over_R500=float(x), Mdyn_over_Mb=float(y),
                 boost_over_newton=float(z))
            for x, y, z in zip(rc / r500, ratio, boost_c) if 0.2 <= x <= 4.0]
    out["cluster_excess_profile"] = prof
    say("   cluster M_dyn/M_baryon vs r/R500 (alpha = 1, rho_ref = 1e5):")
    for pr in prof[::10]:
        say(f"      r/R500 = {pr['r_over_R500']:5.2f}   "
            f"M_dyn/M_b = {pr['Mdyn_over_Mb']:6.3f}   "
            f"boost over Newton = {pr['boost_over_newton']:6.3f}")
    say("   Read this carefully, because it is half right.  The cluster excess")
    say("   IS organised by r/R500 and rises monotonically through R500, which")
    say("   is the shape the programme's own cluster audit reports.  But")
    say("   (i) it saturates at 1 + alpha = 2 where clusters need about 6,")
    say("   (ii) inside 0.25 R500 the -G M F' term drives M_dyn/M_b BELOW 1,")
    say("   which is the wrong sign, and (iii) the same rho_ref switches the")
    say("   modification on INSIDE low-surface-brightness galaxies -- the")
    say("   turn-on radius ranges from 0.08 to 5.8 disk scale lengths across")
    say("   the ladder above -- so an LSB gets a near-constant rescaling of G,")
    say("   degenerate with its stellar mass-to-light ratio, while an HSB gets")
    say("   a genuine shape change.  That is exactly backwards: LSBs are the")
    say("   galaxies with the largest discrepancies and the most shape to")
    say("   explain.")

    # ------------------------------------------------------------------
    #  observables, with sizes, at the rho_ref where the signature exists
    # ------------------------------------------------------------------
    say("")
    say("OBSERVABLE CONFIGURATIONS, sized at rho_ref = rho_bar_b (the only")
    say("choice for which intergalactic environments differ in q at all):")
    obs = []
    for lab, d_mpc, rho_a, rho_b in [
            ("cluster pair 10 Mpc: filament vs void", 10.0,
             30.0 * NK.RHO_BAR_B, 0.2 * NK.RHO_BAR_B),
            ("galaxy pair 1 Mpc: filament vs void", 1.0,
             10.0 * NK.RHO_BAR_B, 0.2 * NK.RHO_BAR_B),
            ("lensing sightline 2 Mpc: wall vs void", 2.0,
             5.0 * NK.RHO_BAR_B, 0.2 * NK.RHO_BAR_B)]:
        qa = float(NK.q_from_smooth(rho_a, NK.RHO_BAR_B))
        qb = float(NK.q_from_smooth(rho_b, NK.RHO_BAR_B))
        row = dict(case=lab, separation_Mpc=d_mpc, q_dense=qa, q_void=qb)
        for a_ in (0.3, 1.0, 3.0):
            row[f"dcoupling_alpha{a_}"] = float(
                (1 + a_ * qb) / (1 + a_ * qa) - 1.0)
            row[f"dvelocity_alpha{a_}"] = float(
                math.sqrt((1 + a_ * qb) / (1 + a_ * qa)) - 1.0)
        obs.append(row)
        say(f"   {lab:<40s} q: {qa:.3f} -> {qb:.3f}")
        say(f"      coupling changes by "
            + " / ".join(f"{100 * row[f'dcoupling_alpha{a_}']:+.1f}%"
                         for a_ in (0.3, 1.0, 3.0))
            + "  (alpha = 0.3 / 1 / 3)")
        say(f"      relative velocity by "
            + " / ".join(f"{100 * row[f'dvelocity_alpha{a_}']:+.1f}%"
                         for a_ in (0.3, 1.0, 3.0)))
    out["observables"] = obs
    say("")
    say("   Best measurements: the mean pairwise infall velocity of cluster")
    say("   pairs split on the density of the connecting segment; the")
    say("   azimuthal dependence of stacked tangential shear at fixed")
    say("   projected radius on the transverse galaxy density; and the")
    say("   relative velocity dispersion of galaxy pairs matched in")
    say("   separation and mass but differing in intervening density.  All")
    say("   three need a wide survey with a void/filament catalogue.  KiDS")
    say("   and wide binaries are sealed holdouts for this programme and are")
    say("   excluded from this list.")
    RES["G7_predictions"] = out


# ==========================================================================
def main():
    t0 = time.time()
    RES["meta"] = dict(
        generated_utc=datetime.now(timezone.utc).isoformat(),
        python=sys.version.split()[0], platform=platform.platform(),
        numpy=np.__version__, gpu=GPU,
        G_kpc_kms2_per_Msun=NK.G, a0_kms2_per_kpc=NK.A0,
        rho_bar_baryon_Msun_per_kpc3=NK.RHO_BAR_B,
        lane="work/wellnet-2026-09/nonlocal",
        sealed_holdouts_untouched=["KiDS", "wide binaries"],
        sparc_splits_used=["train"])
    g1_newtonian_limit()
    g2_reciprocity_and_momentum()
    g3_solar_system()
    g4_rotation_curves()
    good = g4b_global_parameter_screen()
    g4c_btfr(good)
    g4d_sparc_required_F()
    g4e_sparc_forward()
    g5_numerical_gates()
    g6_cost_and_acceleration()
    g7_distinctive_predictions()
    RES["meta"]["runtime_s"] = time.time() - t0
    path = os.path.join(HERE, "nonlocal_results.json")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(RES, fh, indent=1, sort_keys=True, default=float)
    head(f"wrote {path}  ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    main()
