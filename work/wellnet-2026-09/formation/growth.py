"""FORMATION LANE, part 2 -- evolve the perturbation, and answer the brief's
seven questions.

    is the homogeneous state STABLE at all?
    does any mode grow without bound?
    does a PREFERRED COSMIC AXIS appear spontaneously?
    is growth fast enough WITHOUT cold dark matter?
    does the theory overproduce filaments or pancakes?
    is the response finite at BOTH wavelength limits?
    does the response become statistically isotropic where growth is local and
        directional?

WHAT THE GROWTH CALCULATION ASSUMES ABOUT MOMENTUM.  No candidate conserves
momentum and none has a declared carrier.  The fluid equations used here --
continuity plus Euler with g = -grad psi -- are valid for a POTENTIAL force,
which every candidate here has, so the growth equation itself is consistent.
What fails is the vanishing of the total peculiar momentum of a periodic box:
that integral is exactly zero for AQUAL, QUMOND and Newton and non-zero for
every gated candidate, because the gate makes the response an explicit function
of position.  A box with a net acceleration acquires a bulk velocity that no
observer can transform away, so the violation is not a bookkeeping artefact of
the periodic boundary but a prediction.  It is MEASURED here (momentum_in_box)
rather than assumed, against the same base law with the response switched off,
which is the null the tournament had to learn to use.

The equation integrated is the standard quasi-linear one,

    delta'' + (2 + dlnH/dlna) delta' = (4 pi G rho_src Q / H^2) delta

with Q the candidate's source multiplier relative to Newton, exact for a
symmetric collapse in n dimensions.  In log-amplitude variables u = ln delta
this is smooth even where Q ~ delta^(-1/2).
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from scipy.integrate import solve_ivp

import linear_response as LR
from linear_response import (BACKGROUNDS, G, MPC, PRIMARY_BACKGROUND,
                             RHO_CRIT0, TIDAL_C_GEOM, TIDAL_C_RMS,
                             CANDIDATES, LNU, NU, hubble, rho_src)

Z_START = 1000.0
A_START = 1.0 / (1.0 + Z_START)
DELTA_START = 1.0e-5      # baryon contrast at recombination, order (dT/T)
REQUIRED_AMPLIFICATION = 1.0 / DELTA_START


# --------------------------------------------------------------- background
def dlnH_dlna(a, bg):
    om, ol, _ = BACKGROUNDS[bg]
    ok = 1.0 - om - ol
    E2 = om / a ** 3 + ol + ok / a ** 2
    return 0.5 * (-3.0 * om / a ** 3 - 2.0 * ok / a ** 2) / E2


# ------------------------------------------------------------------ the gate
def gate_factor(cand, a, delta, k, n_geom, bg, branch="primary",
                averaging="deep_mond_calibrated"):
    """R = a0_eff/a0 for the candidate, at scale factor a and contrast delta.

    tidal_scalar : |T| = c_n * 4 pi G rhobar delta with c_n = sqrt(2/3), 1/sqrt6,
        0 for pancake/filament/sphere -- the same coefficients that make the RMS
        traceless tidal norm of a statistically isotropic Gaussian field equal
        sqrt(2/3) 4 pi G rhobar sigma.  The gate is averaged over the chi_5
        distribution of |T| with the tournament's own deep-MOND-calibrated rule.
    depth-gated tensors : K = exp(A W S).  On a homogeneous continuum S = 0, so
        R = 1 under every boundary rule; the two branches differ only in W and W
        multiplies zero.  The tensor's first non-vanishing effect on a single
        mode is O(delta^2).  Returned as 1.0 with a flag.
    """
    if cand["form"] == "off":
        return 1.0, {}
    if cand["inv"] == "tidal":
        c = TIDAL_C_GEOM[n_geom]
        rb = rho_src(a, bg)
        Trms = c * 4.0 * np.pi * G * rb * abs(delta)
        if Trms <= 0.0:
            return 1.0 + cand["A"], dict(absT=0.0, W=1.0, saturated=True)
        av = LR.tidal_gate_average(cand["A"], cand["I0"], cand["m"], Trms)
        return av[averaging], dict(absT=Trms, **av)
    # depth-gated tensor: S vanishes on a homogeneous continuum
    return 1.0, dict(note="S = 0 on a homogeneous continuum; O(delta^2) only")


def Q_of(cand, a, delta, k, n_geom, bg, branch="primary"):
    """Source multiplier relative to Newton, and the diagnostics behind it."""
    base = cand["base"]
    if base == "newton":
        return 1.0, dict(y=np.inf, nu=1.0, R=1.0)
    R, gd = gate_factor(cand, a, delta, k, n_geom, bg, branch=branch)
    a0e = cand["a0"] * R
    rb = rho_src(a, bg)
    # |g_N| of the mode: 4 pi G rhobar delta / k_phys, with the sqrt(n)
    # reduction of an n-mode isotropic superposition at fixed total delta.
    gN = 4.0 * np.pi * G * rb * abs(delta) * a / (k * math.sqrt(n_geom))
    y = gN / a0e
    nu = float(NU[base](y))
    L = float(LNU[base](y))
    return nu * (1.0 + L / n_geom), dict(y=float(y), nu=nu, Lnu=L, R=float(R),
                                         a0_eff=float(a0e), **gd)


# ---------------------------------------------------------------- the solver
def integrate(name, k, n_geom=1, bg=PRIMARY_BACKGROUND, a_i=A_START,
              d_i=DELTA_START, a_f=1.0, npts=400, d_cap=1e4):
    """Integrate the quasi-linear growth equation in u = ln delta."""
    cand = CANDIDATES[name]

    def rhs(x, s):
        a = math.exp(x)
        u, up = s
        d = math.exp(min(u, math.log(d_cap)))
        Q, _ = Q_of(cand, a, d, k, n_geom, bg)
        H2 = hubble(a, bg) ** 2
        src = 4.0 * np.pi * G * rho_src(a, bg) * Q / H2
        return [up, src - up * up - (2.0 + dlnH_dlna(a, bg)) * up]

    def stop(x, s):
        return s[0] - math.log(d_cap)
    stop.terminal, stop.direction = True, 1

    xs = np.linspace(math.log(a_i), math.log(a_f), npts)
    sol = solve_ivp(rhs, (xs[0], xs[-1]), [math.log(d_i), 1.0], t_eval=xs,
                    method="LSODA", rtol=1e-9, atol=1e-11, events=stop)
    a = np.exp(sol.t)
    d = np.exp(sol.y[0])
    return dict(a=a, delta=d, dlnd_dlna=sol.y[1], hit_cap=bool(sol.status == 1),
                a_cap=float(np.exp(sol.t_events[0][0]))
                if sol.t_events[0].size else None)


# ============================================================================
# THE SEVEN QUESTIONS
# ============================================================================
def q_stability_and_uniqueness():
    """Is the homogeneous state stable -- and is it even a UNIQUE solution?

    In the deep-MOND limit the source is proportional to delta^(1/2), which is
    NOT Lipschitz at delta = 0.  delta == 0 and delta = (C^2/144) t^4 satisfy
    the same equation with the same initial data delta(0) = delta'(0) = 0.  The
    homogeneous state is therefore a solution but not an isolated one, and
    "linearise about it and read an eigenvalue" is not an available operation.
    Newton is Lipschitz and its trivial solution is unique.
    """
    out = {}
    C = 1.0
    t = np.linspace(0.0, 4.0, 20001)
    exact = C ** 2 / 144.0 * t ** 4

    def rhs(tt, s):
        return [s[1], C * math.sqrt(max(s[0], 0.0))]
    sol = solve_ivp(rhs, (0, 4), [0.0, 0.0], t_eval=t, rtol=1e-12, atol=1e-16)
    out["deep_mond"] = dict(
        exponent_analytic=4.0, coefficient_analytic=C ** 2 / 144.0,
        trivial_branch_max=float(np.max(np.abs(sol.y[0]))),
        second_branch_at_t4=float(exact[-1]),
        both_satisfy_same_ic=True,
        lipschitz=False,
        note="the ODE solver stays on delta = 0 because it is A solution; the "
             "quartic branch is an equally valid one with identical initial "
             "data, which is what non-uniqueness means")
    # verify the quartic branch really solves it
    resid = np.gradient(np.gradient(exact, t), t)[10:-10] - \
        C * np.sqrt(exact)[10:-10]
    out["deep_mond"]["quartic_branch_residual_rel"] = float(
        np.max(np.abs(resid)) / np.max(C * np.sqrt(exact)))
    out["newton"] = dict(lipschitz=True, source_power=1.0,
                         note="delta'' = C delta is linear, the trivial "
                              "solution is unique, and the instability is the "
                              "ordinary Jeans one")
    # ---- does any mode grow WITHOUT BOUND?  three separate senses ----
    kk = np.geomspace(2 * np.pi / (1000.0 * MPC), 2 * np.pi / (0.1 * MPC), 13)
    fs, des = [], []
    for k in kk:
        r = integrate("aqual", k, bg="eds_baryon_src")
        fs.append(float(r["dlnd_dlna"][-1]))
        des.append(float(r["delta"][-1]))
    out["unbounded"] = dict(
        finite_time_blowup=False,
        finite_time_note="delta'' ~ delta^(1/2) is SUBlinear, so the solution "
                         "is delta ~ t^4 and there is no finite-time "
                         "singularity; the amplitude is unbounded but the "
                         "growth is a power law, not a blow-up",
        growth_rate_vs_k=dict(k_invMpc=(kk * MPC).tolist(),
                              f_at_z0=fs, delta_at_z0=des),
        f_max=float(max(fs)), f_min=float(min(fs)),
        f_is_k_independent=bool((max(fs) - min(fs)) < 0.25),
        amplitude_dlnDelta_dlnk=float(np.polyfit(np.log(kk),
                                                 np.log(des), 1)[0]),
        uv_cutoff_exists=False,
        uv_note="the RATE is bounded (the attractor exponent is 2 for every k) "
                "but the AMPLITUDE grows in proportion to k with no cutoff, and "
                "the source multiplier Q diverges as k^(1/2).  Nothing in the "
                "gravity law supplies a smallest scale; only baryonic pressure "
                "would, and this lane does not include it.")
    return out


def q_ellipticity(nsamp=41):
    """Is the field equation elliptic for every candidate?  A negative
    eigenvalue of the response tensor would make the problem ill-posed and the
    homogeneous state catastrophically unstable at short wavelength."""
    rows = {}
    ys = np.geomspace(1e-6, 1e6, nsamp)
    for nm in LR.ORDER:
        c = CANDIDATES[nm]
        ev = []
        for y in ys:
            _, e, _ = LR.response_tensor(c["base"], y * c["a0"], c["a0"])
            ev.append(e)
        ev = np.array(ev)
        rows[nm] = dict(min_eig=float(ev.min()), max_eig=float(ev.max()),
                        elliptic=bool(ev.min() > 0))
    # the tensor structures: K = exp(A W S) is SPD for any real symmetric
    # argument, so ellipticity is structural; verified numerically anyway
    S = np.array([0.4, -0.15, -0.25, 0.2, -0.1, 0.05])
    for nm in ("depth_S_p0", "depth_S_p1_literal"):
        A = CANDIDATES[nm]["A"]
        K = LR.WN.sym3_expm((A * S)[None, :])
        M = LR.WN.sym3_to_full(K)[0]
        rows[nm]["tensor_min_eig"] = float(np.linalg.eigvalsh(M).min())
        rows[nm]["tensor_cond"] = float(np.linalg.cond(M))
        rows[nm]["elliptic"] = bool(rows[nm]["tensor_min_eig"] > 0
                                    and rows[nm]["elliptic"])
    return rows


def q_wavelength_limits(bg=PRIMARY_BACKGROUND, a=A_START, d=DELTA_START):
    """Is the response finite at BOTH wavelength limits?"""
    # The limits are asymptotic statements, so the scan runs well past any
    # physical scale: down to 1e8 Mpc (far outside the horizon -- this is a
    # limit of the equation, not a claim about a real mode) and up to 10 kpc.
    kk = np.geomspace(2 * np.pi / (1e8 * MPC), 2 * np.pi / (0.01 * MPC), 121)
    rows = {}
    for nm in LR.ORDER:
        c = CANDIDATES[nm]
        Q = np.array([Q_of(c, a, d, k, 1, bg)[0] for k in kk])
        sl_uv = float(np.polyfit(np.log(kk[-12:]), np.log(Q[-12:]), 1)[0])
        sl_ir = float(np.polyfit(np.log(kk[:12]), np.log(Q[:12]), 1)[0])
        rows[nm] = dict(k_min_invMpc=float(kk[0] * MPC),
                        k_max_invMpc=float(kk[-1] * MPC),
                        Q_at_k_min=float(Q[0]), Q_at_k_max=float(Q[-1]),
                        Q_at_10Mpc=float(np.interp(
                            2 * np.pi / (10.0 * MPC), kk, Q)),
                        uv_loglog_slope=sl_uv, ir_loglog_slope=sl_ir,
                        finite_ir=bool(np.isfinite(Q[0])
                                       and abs(Q[0] - 1.0) < 0.02),
                        finite_uv=bool(sl_uv < 0.05),
                        uv_divergence="Q ~ k^%.3f" % sl_uv,
                        spread_dex=float(np.log10(Q.max() / max(Q.min(), 1e-300))))
    # the well-network form factor is the one response that is band limited
    kL = np.geomspace(1e-3, 1e3, 400)
    for nm in ("depth_S_p0", "depth_S_p1_literal"):
        w = CANDIDATES[nm]["well"]
        J = LR.S_form_factor(kL / w["L"], family=w["family"], p=w["p"],
                             q=w["q"], s=w["s"], L=w["L"])
        rows[nm]["S_formfactor"] = dict(
            kL=kL[::40].tolist(), J=J[::40].tolist(),
            J_max=float(np.max(np.abs(J))),
            kL_at_max=float(kL[np.argmax(np.abs(J))]),
            J_at_kL_1e_3=float(J[0]), J_at_kL_1e3=float(J[-1]),
            ir_slope=float(np.polyfit(np.log(kL[:40]),
                                      np.log(np.abs(J[:40])), 1)[0]),
            uv_slope=float(np.polyfit(np.log(kL[-40:]),
                                      np.log(np.abs(J[-40:])), 1)[0]),
            band_limited=bool(abs(J[0]) < 0.1 * np.max(np.abs(J))
                              and abs(J[-1]) < 0.1 * np.max(np.abs(J))),
            normalisability=LR.S_kernel_normalisability(
                q=w["q"], s=w["s"], L=w["L"]))
    return rows


def q_anisotropy_and_axis(bg=PRIMARY_BACKGROUND):
    """Does a preferred cosmic axis appear spontaneously?"""
    out = {}
    a, d, k = A_START, DELTA_START, 2 * np.pi / (10.0 * MPC)
    ct = np.linspace(0.0, 1.0, 41)
    for nm in ("aqual", "qumond", "newton", "tidal_scalar"):
        c = CANDIDATES[nm]
        R, _ = gate_factor(c, a, d, k, 1, bg)
        a0e = c["a0"] * R
        gN = 4.0 * np.pi * G * rho_src(a, bg) * d * a / k
        if c["base"] == "aqual":
            gmag = float(LR.nu_simple(gN / a0e)) * gN     # AQUAL needs |g|
        else:
            gmag = gN
        Q = LR.mode_multiplier(c["base"], gmag, a0e, ct)
        out[nm] = dict(a0_eff=float(a0e),
                       Q_kperp=float(Q[0]), Q_kpar=float(Q[-1]),
                       anisotropy_ratio=float(Q[0] / Q[-1]),
                       deep_mond_limit=2.0,
                       fastest_growing_orientation="k perpendicular to g",
                       isotropy=LR.ensemble_isotropy(c["base"], gmag, a0e))
    # AQUAL vs QUMOND at intermediate angles -- they agree at 0 and 90 deg only
    ca = CANDIDATES["aqual"]
    gN = 4.0 * np.pi * G * rho_src(a, bg) * d * a / k
    ga = float(LR.nu_simple(gN / ca["a0"])) * gN
    Qa = LR.mode_multiplier("aqual", ga, ca["a0"], ct)
    Qq = LR.mode_multiplier("qumond", gN, CANDIDATES["qumond"]["a0"], ct)
    out["aqual_vs_qumond"] = dict(
        costheta=ct[::8].tolist(),
        rel_diff=(np.abs(Qa / Qa[0] - Qq / Qq[0])).tolist()[::8],
        max_rel_diff=float(np.max(np.abs(Qa / Qa[0] - Qq / Qq[0]))),
        note="normalised to their common k-perpendicular value; the two "
             "variational references agree at 0 and 90 degrees and differ in "
             "between, so 'the MOND anisotropy' is not a single number")
    # the catalogue-set axis of the well-network tensor
    rows = []
    for N in (100, 1000, 10000, 100000):
        r = [x for x in LR.tensor_S_background_defined(
            nwell_list=(N,), nseed=16)["rows"] if x["N"] == N][0]
        s = r["S_norm_rms"]
        lam = s * math.sqrt(1.5)          # |S|_F = |lam| sqrt(2/3)
        for nm in ("depth_S_p0", "depth_S_p1_literal"):
            A = CANDIDATES[nm]["A"]
            rows.append(dict(candidate=nm, N=N, S_norm=s,
                             growth_anisotropy=float(math.exp(abs(A * lam)))))
    out["catalogue_axis"] = dict(
        rows=rows,
        note="a homogeneous universe represented by N discrete wells has "
             "S = shot noise with a RANDOM axis; K = exp(A W S) turns that into "
             "a growth anisotropy of the size tabulated.  The axis is real, "
             "order unity and set entirely by the catalogue resolution, which "
             "the field equation does not specify.")
    return out


def q_geometry(bg=PRIMARY_BACKGROUND):
    """Does the theory overproduce filaments or pancakes?

    Two comparisons, because they answer different questions:
      fixed_gN  the classic symmetric-collapse ratio Q_n/Q_1 = (1+L/n)/(1+L)
      fixed_delta  the same at fixed TOTAL contrast and fixed k, which also
                carries the sqrt(n) reduction of |g_N| for an n-mode
                superposition and the geometry dependence of |T|
    """
    k = 2 * np.pi / (10.0 * MPC)
    out = dict(analytic_deep_mond=dict(
        Q_ratio_fixed_gN={1: 1.0, 2: 1.5, 3: 5.0 / 3.0},
        Q_ratio_fixed_delta={1: 1.0, 2: 2 * 2 ** 0.25 * 0.75,
                             3: 2 * 3 ** 0.25 * (5.0 / 6.0)},
        newton_ratio={1: 1.0, 2: 1.0, 3: 1.0},
        note="Q_n = nu (1 + L_nu/n) at fixed g_N; the fixed-delta column also "
             "carries the sqrt(n) reduction of |g_N| for an n-mode isotropic "
             "superposition at fixed total contrast.  In Newton all three are "
             "equal, so any departure from 1 is the theory PREFERRING one "
             "collapse geometry over another."))
    for nm in LR.ORDER:
        c = CANDIDATES[nm]
        # follow the candidate's own history rather than one arbitrary point
        hist = integrate(nm, k, n_geom=1, bg=bg)
        eps = []
        for a_ev in (1.0e-3, 1.0e-2, 1.0e-1, 1.0):
            d_ev = float(np.interp(a_ev, hist["a"], hist["delta"]))
            fixed_gN, fixed_delta, diag = {}, {}, {}
            y0 = None
            for n in (1, 2, 3):
                Q, dg = Q_of(c, a_ev, d_ev, k, n, bg)
                fixed_delta[n] = float(Q)
                diag[n] = {kk: float(dg[kk]) for kk in ("y", "nu", "R")
                           if kk in dg}
                if c["base"] == "newton":
                    fixed_gN[n] = 1.0
                    continue
                if y0 is None:
                    y0 = dg["y"]
                nu = float(NU[c["base"]](y0))
                L = float(LNU[c["base"]](y0))
                # at fixed g_N the gate is held at its n = 1 value as well, so
                # the column isolates the PURELY GEOMETRIC factor (1 + L/n)
                fixed_gN[n] = float(nu * (1.0 + L / n))
            eps.append(dict(
                a=a_ev, z=1.0 / a_ev - 1.0, delta=d_ev,
                Q_fixed_gN=fixed_gN, Q_fixed_delta=fixed_delta,
                diagnostics=diag,
                sphere_over_pancake_fixed_gN=float(fixed_gN[3] / fixed_gN[1]),
                sphere_over_pancake_fixed_delta=float(fixed_delta[3]
                                                      / fixed_delta[1]),
                filament_over_pancake_fixed_gN=float(fixed_gN[2]
                                                     / fixed_gN[1]),
                filament_over_pancake_fixed_delta=float(fixed_delta[2]
                                                        / fixed_delta[1])))
        r = max(eps, key=lambda e: e["sphere_over_pancake_fixed_delta"])
        out[nm] = dict(
            epochs=eps,
            sphere_over_pancake_fixed_gN=eps[-1]["sphere_over_pancake_fixed_gN"],
            filament_over_pancake_fixed_gN=eps[-1][
                "filament_over_pancake_fixed_gN"],
            max_sphere_over_pancake_fixed_delta=r[
                "sphere_over_pancake_fixed_delta"],
            max_at_z=r["z"],
            overproduces_pancakes=bool(
                eps[-1]["Q_fixed_gN"][1] > eps[-1]["Q_fixed_gN"][3]))
    return out


def q_growth_table(bgs=("lcdm_bg_baryon_src", "eds_baryon_src",
                        "baryon_only_flat"),
                   scales_mpc=(0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 1000.0),
                   n_geom=1):
    """Is growth fast enough WITHOUT cold dark matter?

    The comoving wavelength is quoted with the BARYON mass inside a sphere of
    radius lambda/2, so the reader can see which astrophysical object each row
    is about without a halo-mass function or any assumption about dark matter.
    """
    rows = []
    for bg in bgs:
        for Lm in scales_mpc:
            k = 2 * np.pi / (Lm * MPC)
            for nm in LR.ORDER:
                r = integrate(nm, k, n_geom=n_geom, bg=bg)
                d_end = float(r["delta"][-1])
                rows.append(dict(
                    candidate=nm, background=bg, scale_Mpc=Lm,
                    delta_start=DELTA_START, delta_end=d_end,
                    amplification=d_end / DELTA_START,
                    a_end=float(r["a"][-1]), z_end=float(1 / r["a"][-1] - 1),
                    hit_cap=r["hit_cap"], a_cap=r["a_cap"],
                    a_at_delta1=_a_at(r, 1.0),
                    dlnd_dlna_end=float(r["dlnd_dlna"][-1]),
                    dlnd_dlna_mid=float(np.interp(
                        0.1, r["a"], r["dlnd_dlna"])),
                    baryon_mass_Msun=float(
                        (4 * np.pi / 3) * (0.5 * Lm * MPC) ** 3
                        * LR.RHO_B0 / 1.98892e30),
                    fast_enough=bool(_a_at(r, 1.0) is not None)))
    return rows


def _a_at(r, target):
    d, a = r["delta"], r["a"]
    if d[-1] < target:
        return None
    return float(np.interp(target, d, a))


def q_attractor(bg="eds_baryon_src", Lm=10.0, n_geom=1):
    """The deep-MOND attractor: delta ~ a^2 with y = g_N/a0 fixed, reached from
    any initial amplitude.  Analytic values cross-checked numerically.

    EdS expansion with a source fraction f = Omega_src/Omega_m: matching powers
    in delta'' + (4/3t) delta' = (3/2) f H^2 Q delta gives Q_* = 10/(3f), so
        slab      Q = nu/2   -> nu = 20/(3f) -> y_* = (3f/20)^2
        cylinder  Q = 3nu/4  -> nu = 40/(9f) -> y_* = (9f/40)^2
        sphere    Q = 5nu/6  -> nu = 4/f     -> y_* = (f/4)^2
    (f = 1 recovers 0.0225, 0.0506, 0.0625.)  The exponent 2 is INDEPENDENT of
    k, of a0, of f and of the geometry, so the attractor erases the initial
    spectrum: delta_k on it is proportional to k, i.e. P(k) ~ k^2 whatever the
    primordial spectrum was.  That is a single-mode statement -- the equation
    is nonlinear and does couple modes -- so it is a scaling, not a derived
    power spectrum.
    """
    k = 2 * np.pi / (Lm * MPC)
    om, ol, osrc = BACKGROUNDS[bg]
    f = osrc / om
    ana = {1: (3 * f / 20.0) ** 2, 2: (9 * f / 40.0) ** 2, 3: (f / 4.0) ** 2}
    out = dict(analytic_y_star=ana, analytic_exponent=2.0,
               source_fraction=f, background=bg)
    runs = {}
    for d0 in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        r = integrate("aqual", k, n_geom=n_geom, bg=bg, d_i=d0, a_f=1.0)
        a, d = r["a"], r["delta"]
        m = (a > 0.3)
        slope = float(np.polyfit(np.log(a[m]), np.log(d[m]), 1)[0])
        rb = rho_src(a, bg)
        y = 4 * np.pi * G * rb * d * a / (k * CANDIDATES["aqual"]["a0"])
        runs[f"{d0:g}"] = dict(exponent_late=slope,
                               y_late=float(y[-1]),
                               y_over_analytic=float(y[-1] / ana[n_geom]),
                               delta_end=float(d[-1]))
    out["runs"] = runs
    out["y_star_measured_over_analytic"] = float(
        np.median([v["y_over_analytic"] for v in runs.values()]))
    vals = [v["delta_end"] for v in runs.values()]
    out["delta_end_spread_over_1e4_in_ic"] = float(max(vals) / min(vals))
    out["y_late_spread"] = float(max(v["y_late"] for v in runs.values())
                                 / min(v["y_late"] for v in runs.values()))
    # P(k) shape on the attractor
    ks = np.geomspace(2 * np.pi / (1000.0 * MPC), 2 * np.pi / (3.0 * MPC), 9)
    de = []
    for kk in ks:
        r = integrate("aqual", kk, n_geom=n_geom, bg=bg, d_i=1e-5)
        de.append(float(r["delta"][-1]))
    out["attractor_spectrum"] = dict(
        scale_Mpc=(2 * np.pi / ks / MPC).tolist(), delta_today=de,
        dlnDelta_dlnk=float(np.polyfit(np.log(ks), np.log(de), 1)[0]),
        expected=1.0)
    return out


def q_tidal_gate_history(bg=PRIMARY_BACKGROUND, Lm=10.0, n_geom=1):
    """The tidal gate's cosmological history and its UV sensitivity."""
    c = CANDIDATES["tidal_scalar"]
    k = 2 * np.pi / (Lm * MPC)
    r = integrate("tidal_scalar", k, n_geom=n_geom, bg=bg)
    hist = []
    for a, d in zip(r["a"][::40], r["delta"][::40]):
        R, gd = gate_factor(c, a, d, k, n_geom, bg)
        hist.append(dict(a=float(a), z=float(1 / a - 1), delta=float(d),
                         absT=float(gd.get("absT", 0.0)),
                         I=float(gd.get("absT", 0.0) / c["I0"]),
                         R_calibrated=float(R),
                         R_arithmetic=float(gd.get("arithmetic", R)),
                         R_of_mean=float(gd.get("response_of_mean", R)),
                         a0_eff=float(c["a0"] * R)))
    # UV SENSITIVITY.  |T| at a point is set by the TOTAL variance of the
    # density field, not by the one mode being evolved, and on the deep-MOND
    # attractor delta_k grows in proportion to k, so the variance is
    # ultraviolet-dominated.  The gate's argument is therefore fixed by the
    # smallest scale retained -- a smoothing scale the field equation does not
    # supply.  The spectrum is normalised to THIS candidate's own delta at
    # 1000 Mpc rather than to an arbitrary unit, so the numbers below are the
    # candidate's own prediction.
    piv = 1000.0
    d_piv = float(integrate("tidal_scalar", 2 * np.pi / (piv * MPC),
                            n_geom=n_geom, bg=bg)["delta"][-1])
    uv = []
    for Rs in (300.0, 100.0, 30.0, 10.0, 3.0, 1.0, 0.3, 0.1):
        sig = LR.sigma_from_cutoff(d_piv, 2 * np.pi / (piv * MPC),
                                   2 * np.pi / (Rs * MPC), slope=1.0)
        Trms = TIDAL_C_RMS * 4 * np.pi * G * rho_src(1.0, bg) * sig
        av = LR.tidal_gate_average(c["A"], c["I0"], c["m"], Trms)
        uv.append(dict(smoothing_Mpc=Rs, sigma_delta=float(sig),
                       absT=float(Trms), R=float(av["primary"]),
                       linear_theory_valid=bool(sig < 1.0)))
    return dict(history=hist, uv_pivot_scale_Mpc=piv, uv_pivot_delta=d_piv,
                a0_eff_range=[float(min(h["a0_eff"] for h in hist)),
                              float(max(h["a0_eff"] for h in hist))],
                boost_range=[float(min(h["R_calibrated"] for h in hist)),
                             float(max(h["R_calibrated"] for h in hist))],
                averaging_bracket_dex=float(max(
                    abs(math.log10(h["R_arithmetic"] / h["R_calibrated"]))
                    for h in hist)),
                uv_sensitivity=uv,
                uv_R_spread=float(max(u["R"] for u in uv)
                                  / min(u["R"] for u in uv)))


def q_depth_gate_cosmological(bg=PRIMARY_BACKGROUND):
    """What the potential-depth gate's argument actually is in cosmology.

    Under the Jeans swindle the field variable is the PECULIAR potential, whose
    plane-wave amplitude is 4 pi G rhobar delta a^2/k^2.  Under a rule that
    references the potential to something outside the perturbation -- a Hubble
    patch, the mass inside the horizon -- it is ~c^2-scale and the gate is
    saturated.  The two admissible families therefore differ not by 0.87 dex,
    as Run AH measured for galaxies, but by the ENTIRE RANGE of the gate.
    """
    rows = []
    for Lm in (1.0, 10.0, 100.0, 1000.0):
        k = 2 * np.pi / (Lm * MPC)
        for nm in ("depth_S_p0", "depth_S_p1_literal"):
            c = CANDIDATES[nm]
            r = integrate(nm, k, bg=bg)
            for a_ev in (A_START, 1.0):
                d = float(np.interp(a_ev, r["a"], r["delta"]))
                phi = 4 * np.pi * G * rho_src(a_ev, bg) * d * a_ev ** 2 / k ** 2
                rows.append(dict(candidate=nm, scale_Mpc=Lm, a=a_ev,
                                 delta=d, abs_phi_peculiar=float(phi),
                                 Phi_0=c["I0"],
                                 ratio=float(phi / c["I0"]),
                                 W_jeans=float(LR.W_of(c["form"],
                                                       phi / c["I0"],
                                                       c["m"]))))
    cc = 2.99792458e8
    ref = {"hubble_patch_c2_over_2": 0.5 * cc ** 2,
           "horizon_mass_potential":
               float(G * (4 * np.pi / 3) * rho_src(1.0, bg) * (cc / LR.hubble(1.0, bg)) ** 2)}
    return dict(peculiar_rows=rows,
                max_W_jeans=float(max(r["W_jeans"] for r in rows)),
                external_reference_rules=ref,
                W_under_external_rules={k: float(LR.W_of("sat", v / 1e12, 2.0))
                                        for k, v in ref.items()},
                gate_range_spanned=[0.0, 1.0],
                verdict="under the Jeans-swindle rule the gate is off by 3 to 9 "
                        "orders of magnitude in its argument; under any rule "
                        "that references an external potential it is saturated "
                        "on.  Both are defensible and they disagree about "
                        "whether the mechanism exists at all.")


def q_tensor_second_order(bg=PRIMARY_BACKGROUND):
    """How big is the term the linear gate throws away for the tensors?

    K = exp(A W S) with delta S = -delta Jfac(k) (zhat zhat - I/3), so the
    exponent is |A| Jfac delta and the neglected effect reaches order unity at
    delta ~ 1/(|A| Jfac).  Jfac falls as 1/rmax, so HOW BIG the neglected term
    is depends on the catalogue radius, not on the field equation.
    """
    rows = []
    for nm in ("depth_S_p0", "depth_S_p1_literal"):
        c = CANDIDATES[nm]
        w = c["well"]
        for Lm in (1.0, 10.0, 100.0):
            k = 2 * np.pi / (Lm * MPC)
            for rmax_over_L in (20.0, 100.0, 1000.0, 10000.0):
                J = float(LR.S_form_factor(np.array([k]), family=w["family"],
                                           p=w["p"], q=w["q"], s=w["s"],
                                           L=w["L"],
                                           rmax_over_L=rmax_over_L)[0])
                amp = abs(c["A"]) * abs(J)
                rows.append(dict(
                    candidate=nm, scale_Mpc=Lm,
                    catalogue_radius_Mpc=rmax_over_L * w["L"] / MPC,
                    Jfac=J, A_times_J=float(amp),
                    delta_at_which_exponent_is_1=float(1.0 / amp)
                    if amp > 0 else None))
    return dict(rows=rows,
                note="the tensor contributes NOTHING at first order in delta; "
                     "this is the size of the first term it does contribute, "
                     "and it is a function of the catalogue radius")


# ============================================================================
# MOMENTUM IN A PERIODIC BOX
# ============================================================================
def tidal_norm_field(box, rho):
    """|traceless Hessian of Phi_N|_F on the box, by FFT (exact, periodic)."""
    n, h = box.n, box.h
    kx = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0
    rk = np.fft.fftn(rho - rho.mean())
    pk = -4 * np.pi * G * rk / K2
    pk[0, 0, 0] = 0.0
    comp = {}
    for (i, j), (Ki, Kj) in zip([(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)],
                                [(KX, KX), (KY, KY), (KZ, KZ), (KX, KY),
                                 (KX, KZ), (KY, KZ)]):
        comp[(i, j)] = np.real(np.fft.ifftn(-Ki * Kj * pk))
    tr = comp[(0, 0)] + comp[(1, 1)] + comp[(2, 2)]
    d = [comp[(i, i)] - tr / 3.0 for i in range(3)]
    return np.sqrt(d[0] ** 2 + d[1] ** 2 + d[2] ** 2
                   + 2 * (comp[(0, 1)] ** 2 + comp[(0, 2)] ** 2
                          + comp[(1, 2)] ** 2))


def momentum_in_box_scan(bg=PRIMARY_BACKGROUND):
    """Where the violation lives, how big it is, and whether it is physics.

    THREE THINGS THIS HAS TO GET RIGHT, all of which the tournament had to
    learn the hard way:
      * the NULL is the same base law with the response switched off, never
        Newton, or AQUAL's own discretisation residual is charged to the gate;
      * the test must be run WHERE THE GATE IS IN TRANSITION.  A gate that is
        saturated everywhere has no gradient and returns a null by
        construction; here that means an epoch with 4 pi G rhobar delta ~ T_0;
      * physics does not vanish under refinement.  The base null falls as h^p;
        if the gated excess does not, it is the law and not the grid.
    """
    epochs = [0.01, 0.02, 0.03, 0.05, 0.08, 0.15]
    scan = [dict(a=a, z=1.0 / a - 1.0, **{
        k: v for k, v in _mom_one(n=40, a=a, bg=bg).items()
        if k in ("aqual", "qumond", "newton", "tidal_scalar")})
        for a in epochs]
    peak = max(scan, key=lambda r: r["tidal_scalar"]["excess_over_base_null"])
    a_pk = peak["a"]
    res, seeds = [], []
    for n in (24, 32, 40, 48, 64):
        r = _mom_one(n=n, a=a_pk, bg=bg)
        res.append(dict(n=n, h_Mpc=40.0 / n,
                        base_null=r["aqual"]["rel_to_own_field"],
                        tidal=r["tidal_scalar"]["rel_to_own_field"],
                        excess=r["tidal_scalar"]["excess_over_base_null"]))
    for sd in range(8):
        r = _mom_one(n=40, a=a_pk, bg=bg, seed=20260904 + 101 * sd)
        seeds.append(dict(seed=20260904 + 101 * sd,
                          base_null=r["aqual"]["rel_to_own_field"],
                          tidal=r["tidal_scalar"]["rel_to_own_field"],
                          vbulk=r["tidal_scalar"][
                              "bulk_velocity_km_s_per_Hubble_time"]))
    lh = np.log([r["h_Mpc"] for r in res])
    return dict(
        epoch_scan=scan, peak_a=a_pk, peak_z=1.0 / a_pk - 1.0,
        resolution=res,
        base_null_h_slope=float(np.polyfit(
            lh, np.log([r["base_null"] for r in res]), 1)[0]),
        tidal_h_slope=float(np.polyfit(
            lh, np.log([r["tidal"] for r in res]), 1)[0]),
        seed_ensemble=seeds,
        tidal_mean=float(np.mean([s["tidal"] for s in seeds])),
        tidal_sd=float(np.std([s["tidal"] for s in seeds])),
        base_null_mean=float(np.mean([s["base_null"] for s in seeds])),
        vbulk_mean_km_s=float(np.mean([s["vbulk"] for s in seeds])),
        vbulk_sd_km_s=float(np.std([s["vbulk"] for s in seeds])))


def _mom_one(n=48, Lbox=40.0 * MPC, seed=20260904, nmode=6,
             amp=0.30, a=0.10, bg=PRIMARY_BACKGROUND):
    """Net peculiar acceleration of a periodic box, per candidate."""
    rng = np.random.default_rng(seed)
    box = LR.PeriodicBox(n, Lbox)
    rb = rho_src(a, bg)
    d = np.zeros(box.X.shape)
    for _ in range(nmode):
        kv = 2 * np.pi / Lbox * rng.integers(-2, 3, 3).astype(float)
        if not np.any(kv):
            kv = np.array([2 * np.pi / Lbox, 0.0, 0.0])
        d += (amp / math.sqrt(nmode)) * np.cos(
            kv[0] * box.X + kv[1] * box.Y + kv[2] * box.Z
            + rng.random() * 2 * np.pi)
    rho = rb * (1.0 + d)
    absT = tidal_norm_field(box, rho)
    out = {}
    scale = None
    for nm in ("newton", "aqual", "qumond", "tidal_scalar"):
        c = CANDIDATES[nm]
        if c["base"] == "newton":
            psi, _, _ = box.solve_linear(rho, box.iso_tensor(1.0))
        elif c["base"] == "qumond":
            psi = _qumond_solve(box, rho, c["a0"])
        else:
            if c["inv"] == "tidal":
                W = LR.W_of(c["form"], absT / c["I0"], c["m"])
                a0f = c["a0"] * (1.0 + c["A"] * W)
            else:
                a0f = np.full(rho.shape, c["a0"])
            psi, _, _ = box.solve_nonlinear(
                rho, lambda p, a0f=a0f: LR.aqual_tensor_field(box, p, a0f))
        F, acc = box.net_force(rho, psi)
        gx, gy, gz = box.grad(psi)
        s = float(np.sqrt(np.mean(gx ** 2 + gy ** 2 + gz ** 2)))
        if nm == "aqual":
            scale = s
        out[nm] = dict(net_accel=acc.tolist(),
                       net_accel_mag=float(np.linalg.norm(acc)),
                       rms_field=s,
                       rel_to_own_field=float(np.linalg.norm(acc) / max(s, 1e-300)))
    base = out["aqual"]["rel_to_own_field"]
    out["tidal_scalar"]["excess_over_base_null"] = float(
        out["tidal_scalar"]["rel_to_own_field"] / max(base, 1e-300))
    # spurious bulk velocity accumulated over a Hubble time
    Ht = 1.0 / hubble(a, bg)
    for nm in out:
        if isinstance(out[nm], dict) and "net_accel_mag" in out[nm]:
            out[nm]["bulk_velocity_km_s_per_Hubble_time"] = float(
                out[nm]["net_accel_mag"] * Ht / 1000.0)
    out["_setup"] = dict(n=n, Lbox_Mpc=Lbox / MPC, a=a, amp=amp,
                         nmode=nmode, seed=seed,
                         rms_absT=float(np.sqrt(np.mean(absT ** 2))),
                         T0=CANDIDATES["tidal_scalar"]["I0"],
                         null="aqual with the response off, same grid")
    return out


def _qumond_solve(box, rho, a0):
    """QUMOND: lap psi = div[nu(|grad Phi_N|/a0) grad Phi_N], by FFT."""
    n, h = box.n, box.h
    kx = 2 * np.pi * np.fft.fftfreq(n, d=h)
    KX, KY, KZ = np.meshgrid(kx, kx, kx, indexing="ij")
    K2 = KX ** 2 + KY ** 2 + KZ ** 2
    K2[0, 0, 0] = 1.0
    rk = np.fft.fftn(rho - rho.mean())
    pk = -4 * np.pi * G * rk / K2
    pk[0, 0, 0] = 0.0
    gx = np.real(np.fft.ifftn(1j * KX * pk))
    gy = np.real(np.fft.ifftn(1j * KY * pk))
    gz = np.real(np.fft.ifftn(1j * KZ * pk))
    gm = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    nu = LR.nu_rar(gm / a0)
    div = np.real(np.fft.ifftn(1j * KX * np.fft.fftn(nu * gx)
                               + 1j * KY * np.fft.fftn(nu * gy)
                               + 1j * KZ * np.fft.fftn(nu * gz)))
    dk = np.fft.fftn(div - div.mean())
    psik = -dk / K2
    psik[0, 0, 0] = 0.0
    return np.real(np.fft.ifftn(psik))


# ============================================================================
# ANALYTIC / NUMERICAL CROSS-CHECKS
# ============================================================================
def q_integrator_validation():
    """Newton with a FULL matter source must reproduce the standard linear
    growth factor.  Nothing in this lane is believed until the integrator
    reproduces a number that was not computed by this lane."""
    BACKGROUNDS["_validate_lcdm"] = (LR.OMEGA_M_LCDM, 1 - LR.OMEGA_M_LCDM,
                                     LR.OMEGA_M_LCDM)
    r = integrate("newton", 2 * np.pi / (10.0 * MPC), bg="_validate_lcdm",
                  a_i=1e-4, d_i=1e-4, a_f=1.0)
    D1_over_a = float(r["delta"][-1] / r["a"][-1] / (r["delta"][0] / r["a"][0]))
    # closed form: D(a) proportional to H(a) Int da/(a H)^3, evaluated here
    om = LR.OMEGA_M_LCDM
    aa = np.geomspace(1e-6, 1.0, 200000)
    E = np.sqrt(om / aa ** 3 + 1 - om)
    I = np.array([np.trapezoid((1.0 / (aa[:i + 1] * E[:i + 1]) ** 3), aa[:i + 1])
                  for i in (len(aa) - 1,)])
    D1 = float(2.5 * om * E[-1] * I[0])
    del BACKGROUNDS["_validate_lcdm"]
    return dict(numeric_D1_over_a=D1_over_a, closed_form_D1=D1,
                rel_err=float(abs(D1_over_a - D1) / D1),
                f_growth_rate_today=float(r["dlnd_dlna"][-1]),
                f_closed_form=float(om ** 0.55),
                pass_=bool(abs(D1_over_a - D1) / D1 < 3e-3))


def q_anisotropy_numerical():
    """The 2:1 anisotropy, from a FULL NONLINEAR SOLVE, against the analytic
    linear-response tensor.  Two routes to the same number, as the brief asks.

    A uniform background field g is imposed by writing psi = -g.x + phi with
    phi periodic; the equation for phi is
        div[ mu(|-g + grad phi|/a0) grad phi ] = 4 pi G d rho + div[ mu g ]
    and a small plane-wave d rho is added with k parallel or perpendicular to
    g.  The measured amplitude of phi at that mode is compared with
        phi_k = -4 pi G d rho_k / (k.K.k),  K = mu (I + L_mu ghat ghat)
    """
    L = 60.0 * MPC
    n = 48
    box = LR.PeriodicBox(n, L)
    a0 = CANDIDATES["aqual"]["a0"]
    rows = []
    for gfac in (0.3, 1.0, 3.0):
        g = gfac * a0
        gvec = np.array([0.0, 0.0, g])
        mu_b = float(LR.mu_simple(g / a0))
        Lm = float(LR.Lmu_simple(g / a0))
        for tag, kvec in (("k_par_g", [0, 0, 1]), ("k_perp_g", [1, 0, 0])):
            kv = 2 * np.pi / L * np.array(kvec, float)
            ph = kv[0] * box.X + kv[1] * box.Y + kv[2] * box.Z
            drho = 1.0e-32 * np.cos(ph)          # deliberately tiny: linear
            phi = np.zeros(box.X.shape)
            for _ in range(40):
                gx, gy, gz = box.grad(phi)
                gm = np.sqrt(gx ** 2 + (gy) ** 2 + (gz - g) ** 2)
                mu = LR.mu_simple(gm / a0)
                A = (mu, mu.copy(), mu.copy(), np.zeros_like(mu),
                     np.zeros_like(mu), np.zeros_like(mu))
                # div[mu grad phi] - g d_z mu = 4 pi G drho, so the source
                # carries +div(mu g zhat).  The sign matters: with it reversed
                # the k-parallel case is wrong by up to a factor 6.6 while the
                # k-perpendicular case -- where d_z mu vanishes -- still agrees
                # with the analytic answer to 0.14%, which is exactly how a
                # sign error hides.
                fz = 0.5 * (mu + np.roll(mu, -1, 2)) * g
                extra = (fz - np.roll(fz, 1, 2)) / box.h
                b = drho + extra / (4 * np.pi * G)
                new, _, _ = box.solve_linear(b, A, x0=phi)
                conv = np.max(np.abs(new - phi)) < 1e-12 * max(
                    np.max(np.abs(new)), 1e-300)
                phi = 0.5 * (phi + new)
                if conv:
                    phi = new
                    break
            amp = 2.0 * float(np.mean(phi * np.cos(ph)))
            kKk = float(mu_b * (np.dot(kv, kv)
                                + Lm * np.dot(kv, gvec / g) ** 2))
            ana = -4 * np.pi * G * 1.0e-32 / kKk
            rows.append(dict(g_over_a0=gfac, orientation=tag,
                             phi_amp_numeric=amp, phi_amp_analytic=ana,
                             rel_err=float(abs(amp - ana) / abs(ana))))
    out = dict(rows=rows)
    for gfac in (0.3, 1.0, 3.0):
        p = [r for r in rows if r["g_over_a0"] == gfac]
        num = [r for r in p if r["orientation"] == "k_perp_g"][0]
        den = [r for r in p if r["orientation"] == "k_par_g"][0]
        out[f"anisotropy_g_over_a0_{gfac:g}"] = dict(
            numeric=float(num["phi_amp_numeric"] / den["phi_amp_numeric"]),
            analytic=float(num["phi_amp_analytic"] / den["phi_amp_analytic"]),
            analytic_1_plus_Lmu=float(1.0 + LR.Lmu_simple(gfac)))
    out["max_rel_err"] = float(max(r["rel_err"] for r in rows))
    out["pass_"] = bool(out["max_rel_err"] < 0.02)
    return out


def q_gate_derivative():
    """The tidal gate's linear response is IDENTICALLY ZERO -- measured.

    W = 1/(1 + (|T|/T_0)^m) with m = 2 and |T| proportional to |delta|, so
    dW/d delta -> 0 and d^2 W/d delta^2 -> a finite non-zero value.  Both are
    computed by finite differences over six decades of delta, and the fitted
    log-log slope of (1 - W) against delta is reported: 2 means second order.
    """
    c = CANDIDATES["tidal_scalar"]
    a, bg = A_START, PRIMARY_BACKGROUND
    rb = rho_src(a, bg)
    ds = np.geomspace(1e-14, 1e-8, 25)
    I = TIDAL_C_RMS * 4 * np.pi * G * rb * ds / c["I0"]
    # 1 - W = I^m/(1+I^m) computed directly: forming it as 1 - 1/(1+I^m)
    # underflows to exactly zero below I ~ 1e-8 and returns a NaN slope.
    oneminusW = I ** c["m"] / (1.0 + I ** c["m"])
    slope = float(np.polyfit(np.log(ds), np.log(oneminusW), 1)[0])
    h = 1e-12
    Ih = TIDAL_C_RMS * 4 * np.pi * G * rb * h / c["I0"]
    dWdd = -(Ih ** 2 / (1.0 + Ih ** 2)) / h
    return dict(loglog_slope_of_one_minus_W=slope, expected=2.0,
                one_minus_W_at_delta=dict(zip([float(x) for x in ds[::6]],
                                              [float(x) for x in
                                               oneminusW[::6]])),
                first_derivative_at_delta_1e_12=float(dWdd),
                order_of_first_effect=2,
                note="the gate is second order in delta, so the LINEAR response "
                     "of the tidal-gated candidate is exactly AQUAL with "
                     "a0 -> a0(1+A); the gate itself contributes nothing at "
                     "first order, at any epoch, for any k")


# ============================================================================
# RESPONSIVENESS -- every headline statistic
# ============================================================================
def q_responsiveness(bg=PRIMARY_BACKGROUND):
    out = {}
    k10 = 2 * np.pi / (10.0 * MPC)

    out["growth_vs_a0"] = LR.responsiveness(
        lambda t: integrate("aqual", k10, bg=bg)["delta"][-1]
        if _seta("aqual", t) else np.nan,
        np.geomspace(1.058e-11, 1.058e-9, 9), "delta(z=0) vs a0, one decade "
        "either side of the fitted value")
    _seta("aqual", 1.0580375e-10)

    out["growth_vs_A"] = LR.responsiveness(
        lambda t: integrate("tidal_scalar", k10, bg=bg)["delta"][-1]
        if _setA("tidal_scalar", t) else np.nan,
        np.linspace(0.0, 32.0, 9), "delta(z=0) vs the tidal gate amplitude A")
    _setA("tidal_scalar", 16.0)

    out["growth_vs_T0"] = LR.responsiveness(
        lambda t: integrate("tidal_scalar", k10, bg=bg)["delta"][-1]
        if _setI("tidal_scalar", t) else np.nan,
        np.geomspace(1e-36, 1e-30, 9), "delta(z=0) vs the tidal scale T_0, "
        "three decades either side")
    _setI("tidal_scalar", 1.0e-33)

    out["growth_vs_k"] = LR.responsiveness(
        lambda t: integrate("aqual", 2 * np.pi / (t * MPC), bg=bg)["delta"][-1],
        np.geomspace(1.0, 1000.0, 9), "delta(z=0) vs comoving scale")

    out["anisotropy_vs_y"] = LR.responsiveness(
        lambda t: LR.mode_multiplier("qumond", t * 1.06e-10, 1.06e-10, 0.0)
        / LR.mode_multiplier("qumond", t * 1.06e-10, 1.06e-10, 1.0),
        np.geomspace(1e-4, 1e4, 13), "anisotropy ratio Q_perp/Q_par vs g_N/a0")

    out["gate_vs_absT"] = LR.responsiveness(
        lambda t: LR.tidal_gate_average(16.0, 1e-33, 2.0, t)["primary"],
        np.geomspace(1e-36, 1e-30, 13), "a0 boost vs |T| RMS, six decades")

    out["Sformfactor_vs_kL"] = LR.responsiveness(
        lambda t: float(LR.S_form_factor(np.array([t / (300.0 * LR.KPC)]),
                                         p=0.0, q=1.0, s=2.0,
                                         L=300.0 * LR.KPC)[0]),
        np.geomspace(1e-2, 1e2, 13), "well-network form factor vs k L")
    return out


def _seta(nm, v):
    CANDIDATES[nm]["a0"] = float(v)
    return True


def _setA(nm, v):
    CANDIDATES[nm]["A"] = float(v)
    return True


def _setI(nm, v):
    CANDIDATES[nm]["I0"] = float(v)
    return True


# ============================================================================
if __name__ == "__main__":
    t0 = time.time()
    R = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
         "lane": "work/wellnet-2026-09/formation",
         "no_observational_data": True,
         "blind_note": "No observational data of any kind is loaded in this lane, "
                       "so there is no blind-protection issue: nothing is fitted, "
                       "no split is consumed, no holdout is touched.  KiDS and "
                       "wide binaries are not loaded, listed or referenced; "
                       "neither is SPARC nor any cluster catalogue.  Every "
                       "constant is frozen from ../tournament/tournament.json.",
         "constants": {nm: {k: v for k, v in c.items() if k != "well"}
                       | ({"well": {k: (v / LR.KPC if k == "L" else v)
                                    for k, v in c["well"].items()}}
                          if "well" in c else {})
                       for nm, c in CANDIDATES.items()},
         "cosmology": dict(H0_km_s_Mpc=67.4, Omega_b=LR.OMEGA_B,
                           Omega_m_lcdm=LR.OMEGA_M_LCDM,
                           z_start=Z_START, delta_start=DELTA_START,
                           primary_background=PRIMARY_BACKGROUND,
                           backgrounds=BACKGROUNDS)}

    steps = [
        ("homogeneous_background",
         lambda: {nm: LR.homogeneous_state(nm) for nm in LR.ORDER}),
        ("tensor_S_on_homogeneous", LR.tensor_S_background_defined),
        ("tidal_chi5_identity", LR.tidal_chi5_check),
        ("stability_and_uniqueness", q_stability_and_uniqueness),
        ("ellipticity", q_ellipticity),
        ("wavelength_limits", q_wavelength_limits),
        ("anisotropy_and_axis", q_anisotropy_and_axis),
        ("geometry", q_geometry),
        ("attractor", q_attractor),
        ("tidal_gate_history", q_tidal_gate_history),
        ("depth_gate_cosmological", q_depth_gate_cosmological),
        ("tensor_second_order", q_tensor_second_order),
        ("growth_table", q_growth_table),
        ("momentum_in_box", momentum_in_box_scan),
        ("S_formfactor_measured",
         lambda: [LR.S_form_factor_measured(kl, p=0.0, q=1.0, s=2.0)
                  for kl in (0.5, 1.0, 2.0, 4.0)]),
        ("integrator_validation", q_integrator_validation),
        ("anisotropy_numerical", q_anisotropy_numerical),
        ("gate_derivative", q_gate_derivative),
        ("responsiveness", q_responsiveness),
    ]
    for name, fn in steps:
        t = time.time()
        try:
            R[name] = fn()
            print(f"{name:28s} ok   {time.time()-t:6.1f}s")
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            R[name] = {"error": repr(e)}
            print(f"{name:28s} FAIL {time.time()-t:6.1f}s  {e}")
    R["seconds"] = round(time.time() - t0, 1)
    import hashlib
    R["source_sha256"] = {
        f: hashlib.sha256(open(os.path.join(os.path.dirname(
            os.path.abspath(__file__)), f), "rb").read()).hexdigest()
        for f in ("linear_response.py", "growth.py", "test_solver.py")}
    for rel in ("../tensor/wellnet.py", "../tournament/tw_core.py",
                "../../gravitylab/solver.py"):
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), rel)
        R["source_sha256"][rel] = hashlib.sha256(
            open(p, "rb").read()).hexdigest()

    def _j(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.bool_,)):
            return bool(o)
        return str(o)
    json.dump(R, open("formation_results.json", "w"), indent=1, default=_j)
    print("wrote formation_results.json", R["seconds"], "s")
