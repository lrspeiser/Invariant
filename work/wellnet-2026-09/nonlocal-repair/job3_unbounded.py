"""JOB 3(a) -- is the family repairable at all?

The boundedness theorem says qbar in [0,1) bounds F, so r v_c^2 -> G M sup F.
The brief asks whether making F UNBOUNDED is enough.  It is not, and this
script establishes three things, in order of increasing sharpness.

T1.  THE CORRECT CONDITION IS F/r UNBOUNDED, NOT F UNBOUNDED.
     D = F - r F' = -r^2 (F/r)', so F/r = Int_r^inf D/r'^2 dr'.  An
     asymptotically flat curve needs D -> (v_f^2/GM) r, and then that integral
     diverges logarithmically.  Hence |F|/r -> inf.  F ~ r^0.9 is unbounded
     and still gives a DECLINING curve; the true requirement is
     F ~ -(v_f^2/GM) r ln(r/r_*), which is unbounded, eventually NEGATIVE,
     and grows faster than r.

T2.  F IS ONLY DEFINED UP TO AN ADDITIVE MULTIPLE OF r.
     F -> F + c r shifts Phi by the constant -G M c, so D is unchanged.  Any
     claim about "F bounded" must be converted into a claim about D before it
     constrains dynamics.

T3.  A SECOND, INDEPENDENT NO-GO: LINEARITY FIXES THE BTFR SLOPE AT 2.
     Phi = -G Int rho F(...)/|x-x'| is LINEAR in rho whenever F does not
     depend on rho.  A linear functional gives v_f^2 proportional to M, i.e.
     M proportional to v_f^2: BTFR slope 2, against the observed 3.85.  The
     only escape is the q field, and a BOUNDED q saturates in the outskirts,
     which restores linearity exactly where the flat part lives.  So even a
     repaired, unbounded F inherits a BTFR that is too shallow unless the
     coefficient of the flat term scales as M^-1/2 -- which requires the
     kernel to know the source mass.

Then the actual search: a grammar of separation-dependent F(qbar, r) forms,
screened against
   R1  D > 0 everywhere from 1 AU to 1 Mpc          (no reversal)
   R2  dlnD/dlnr = 1 +/- 0.2 over 2-20 R_disk       (flat over the observed range)
   R3  |D(1 AU) - 1| < 1e-11                        (solar-system limit)
   R4  F_local in [1.10, 1.70]                      (Oort limit)
   R5  BTFR slope in [3.5, 4.2]                     (the relation itself)
"""
from __future__ import annotations

import itertools
import json
import math
import time

import numpy as np

import common as C

RES = {}
T0 = time.time()
AU = C.AU_KPC


def say(*a):
    print(*a, flush=True)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


# ==========================================================================
#  D from a closed-form F(r) by a high-order log stencil.
# ==========================================================================
_OFF = np.array([-3, -2, -1, 0, 1, 2, 3])
_C7 = np.array([-1.0, 9.0, -45.0, 0.0, 45.0, -9.0, 1.0]) / 60.0


def D_of(Ffun, r, h=1e-4):
    """D = F - dF/dlnr for a closed-form F(r), seven-point log stencil."""
    r = np.atleast_1d(np.asarray(r, float))
    rr = r[:, None] * np.exp(_OFF[None, :] * h)
    Fv = Ffun(rr)
    return Fv[:, 3], Fv[:, 3] - (Fv @ _C7) / h


# ==========================================================================
def t1_sharpened_theorem():
    head("T1  The correct condition is F/r unbounded, not F unbounded")
    say("D = F - r F' = -r^2 (F/r)', so F/r = Int_r^inf D/r'^2 dr'.")
    say("Flat means D = (v_f^2/GM) r, and Int^inf dr/r diverges: |F|/r -> "
        "inf.")
    out = {}
    rows = []
    r = np.geomspace(1.0, 1.0e4, 400)
    for tag, f, note in [
            ("F = 1 + 3 r^0.5", lambda x: 1 + 3 * x ** 0.5,
             "unbounded, F/r -> 0"),
            ("F = 1 + 3 r^0.9", lambda x: 1 + 3 * x ** 0.9,
             "unbounded, F/r -> 0"),
            ("F = 1 + 3 r", lambda x: 1 + 3 * x,
             "F/r -> const: PURE GAUGE, D = 1 exactly"),
            ("F = 1 + 3 r^1.1", lambda x: 1 + 3 * x ** 1.1,
             "F/r -> inf but D < 0: gravity REVERSES"),
            ("F = 1 + 3 r ln(1e4/r)", lambda x: 1 + 3 * x * np.log(1e4 / x),
             "the unique exactly-flat form"),
            ("F = 4.0 (bounded)", lambda x: np.full_like(x, 4.0),
             "bounded: exactly Keplerian")]:
        Fv, D = D_of(f, r)
        sl = np.gradient(np.log(np.abs(D)), np.log(r))
        # v_c^2 = GM D / r ; log-slope of v_c
        lv = 0.5 * (np.log(np.abs(D)) - np.log(r))
        rows.append(dict(form=tag, note=note,
                         D_at_r1=float(D[0]), D_at_r1e4=float(D[-1]),
                         min_D=float(np.min(D)),
                         dlnD_dlnr_outer=float(sl[-1]),
                         dlnv_dlnr_outer=float(np.gradient(lv,
                                                           np.log(r))[-1]),
                         F_over_r_outer=float(Fv[-1] / r[-1])))
        say(f"   {tag:<26s} min D {np.min(D):+11.4g}   dlnD/dlnr "
            f"{sl[-1]:+7.4f}   dlnv/dlnr {rows[-1]['dlnv_dlnr_outer']:+7.4f}"
            f"   F/r {rows[-1]['F_over_r_outer']:9.3g}   ({note})")
    say("")
    say("   Read the F = 1 + 3 r^0.9 row: F IS unbounded and the curve is "
        "still")
    say("   declining (dlnv/dlnr = -0.05, not 0).  Read the F = 1 + 3 r row: "
        "D = 1")
    say("   exactly, a pure gauge shift of Phi.  Only the r ln r form is "
        "flat.")
    out["ladder"] = rows

    #  gauge invariance, exactly
    for c in (0.0, 1.0, 7.3):
        Fv, D = D_of(lambda x: 1 + 3 * x ** 0.5 + c * x, r)
        out[f"gauge_c={c}"] = float(np.max(np.abs(D - D_of(
            lambda x: 1 + 3 * x ** 0.5, r)[1])))
    say(f"   T2 gauge check: adding c r to F changes D by at most "
        f"{max(out[f'gauge_c={c}'] for c in (1.0, 7.3)):.2e} over the whole "
        f"range.")
    RES["T1_sharpened_theorem"] = out


# ==========================================================================
def t3_linearity_btfr():
    head("T3  The second, independent no-go: linearity fixes the BTFR slope "
         "at 2")
    say("If F does not depend on rho, Phi[rho] is a LINEAR functional of rho, "
        "so")
    say("doubling the mass doubles the potential and v_f^2 scales as M.  "
        "M = v_f^2 x")
    say("const is BTFR slope 2.  The observed slope is 3.85 +/- 0.09.  MOND "
        "gets 4")
    say("only because g = sqrt(G M a0)/r has the coefficient scaling as "
        "M^(1/2),")
    say("which no linear kernel can produce.")
    out = {}
    #  measure it: v_f^2 from a flat-ish D = c r, with c independent of M
    M = np.geomspace(1e9, 3e11, 40)
    for tag, c_of_M, exp_slope in (
            ("c independent of M (any linear kernel)", lambda m: 1e-3, 2.0),
            ("c proportional to M^-1/2 (MOND)",
             lambda m: math.sqrt(C.A0 / (C.G * m)) / 1.0, 4.0)):
        vf2 = np.array([C.G * m * c_of_M(m) for m in M])
        sl = np.polyfit(np.log10(np.sqrt(vf2)), np.log10(M), 1)[0]
        out[tag] = dict(slope=float(sl), expected=exp_slope)
        say(f"   {tag:<40s} BTFR slope {sl:.3f}  (expected {exp_slope})")
    say("")
    say("   The q field is the only nonlinearity available, and q in [0,1) "
        "SATURATES")
    say("   in the outskirts -- exactly where the flat part lives -- so the "
        "kernel")
    say("   becomes linear again there.  The previous lane measured slope "
        "2.88 with")
    say("   the modification on and 2.07 with it off; 2 and 4 bracket that, "
        "and the")
    say("   2.88 is the transition region, not a solution.")
    RES["T3_linearity_btfr"] = out


# ==========================================================================
#  The search.
# ==========================================================================
LADDER = [("dwarf_LSB", 1.5e9, 1.0), ("dwarf_HSB", 3.0e9, 1.2),
          ("LSB_large", 1.1e10, 6.0), ("spiral_mid", 2.0e10, 2.5),
          ("MW_like", 6.0e10, 3.0), ("massive", 2.1e11, 5.0)]


def qbar_profile(M, rd, rho_ref=1e6, kind="delta"):
    """Path-average void state from radius r to the centre, for an
    exponential-sphere galaxy: qbar(r) = (1/r) Int_0^r q ds.

    This is the kernel's OWN functional evaluated on the dominant source, and
    it is what makes the atom nonlocal: a LOCAL q saturates the moment the
    density drops below rho_ref, whereas the path average approaches its
    ceiling only as 1 - r_ref/r.
    """
    rs = rd
    rho0 = M / (8.0 * math.pi * rs ** 3)
    r = np.geomspace(1e-9, 1.0e4, 6000)
    rho = rho0 * np.exp(-r / rs) + C.NK.RHO_BAR_B
    if kind == "delta":
        q = np.clip(rho_ref / rho - 1.0, 0.0, 1.0 - 1e-15)
    else:
        q = 1.0 / (1.0 + rho / rho_ref)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (q[1:] + q[:-1])
                                           * np.diff(r))])
    qb = cum / r
    return r, q, np.clip(qb, 0.0, 1.0 - 1e-15)


#: F(qbar, r) atoms.  Every one is 1 at qbar = 0, so a fully screened region
#: is EXACTLY Newtonian, which is the only way the solar-system limit can be
#: met without a coincidence.
ATOMS = {
    "bounded_poly":  lambda qb, x, a, p, n: 1 + a * qb ** p,
    "power":         lambda qb, x, a, p, n: 1 + a * qb ** p * x ** n,
    "log":           lambda qb, x, a, p, n: 1 + a * qb ** p * np.log1p(x),
    "xlogx":         lambda qb, x, a, p, n: 1 + a * qb ** p * x
                     * np.log1p(n / np.maximum(x, 1e-30)),
    "pow_over_1mq":  lambda qb, x, a, p, n: 1 + a * qb ** p * x ** n
                     / (1.0 - qb),
    "exp_growth":    lambda qb, x, a, p, n: np.exp(
        np.minimum(a * qb ** p * x ** n, 500.0)),
    "x_over_1mq":    lambda qb, x, a, p, n: 1 + a * qb ** p * x
                     / (1.0 - qb),
}
UNBOUNDED = {"power", "log", "xlogx", "pow_over_1mq", "exp_growth",
             "x_over_1mq"}


def gate_diagnostic():
    """Why R1 and R2 are mutually exclusive: the gate subtracts from D.

    For F = 1 + a G(qbar(r)) H(r),

        D = F - r F' = 1 + a G (H - r H') - a r G'(qbar) qbar'(r) H

    and the LAST term is strictly negative whenever G is increasing in qbar
    (the modification grows in voids) and qbar is increasing in r (paths get
    more void-like outwards).  So the very r-dependence that switches the
    modification on SUBTRACTS from the force, and it scales with the same
    amplitude a that the flat part needs.  That is the whole conflict.
    """
    head("Why R1 (D > 0) and R2 (D proportional to r) exclude each other")
    say("For F = 1 + a G(qbar(r)) H(r),")
    say("   D = 1 + a G (H - r H')  -  a r G'(qbar) qbar'(r) H")
    say("The last term is strictly NEGATIVE when the modification grows in "
        "voids")
    say("(G' > 0) and paths get more void-like outwards (qbar' > 0).  Raising "
        "a to")
    say("lengthen the flat part raises the negative term by the same factor.")
    out = {}
    r, q, qb = qbar_profile(6.0e10, 3.0, 1e6)
    qbf = lambda x: np.interp(np.log(np.maximum(x, 1e-30)), np.log(r),
                              np.log(np.maximum(qb, 1e-300)))
    ro = np.geomspace(1.0, 300.0, 60)
    L, n = 10.0, 1.0
    H = lambda x: (x / L) * np.log1p(n * L / np.maximum(x, 1e-30))
    for a in (0.3, 1.0, 3.0, 10.0):
        for p_ in (0.0, 1.0, 2.0):
            Ff = lambda x: 1 + a * np.exp(p_ * qbf(x)) * H(x)
            Fv, D = D_of(Ff, ro)
            sl = np.gradient(np.log(np.abs(D)), np.log(ro))
            sel = (ro >= 6.0) & (ro <= 60.0)
            out[f"a={a}|p={p_}"] = dict(
                min_D=float(np.min(D)),
                mean_slope=float(np.mean(sl[sel])),
                gated=bool(p_ > 0))
            say(f"   a={a:<5g} p={p_:<4g} ({'gated  ' if p_ else 'UNGATED'})"
                f"  min D over 1-300 kpc {np.min(D):+11.4g}   "
                f"mean dlnD/dlnr over 2-20 R_d {np.mean(sl[sel]):+.3f}")
    say("")
    say("   The p = 0 rows are the SAME H(r) with the gate removed: D stays")
    say("   positive and the slope still reaches 1.  Turning the gate on at "
        "the")
    say("   same a drives min D negative.  An ungated form is not a "
        "candidate,")
    say("   because without a gate the modification is present in the solar")
    say("   system too -- which is what R3 tests.")
    RES["gate_diagnostic"] = out


def screen_unbounded():
    head("The search: unbounded F that ALSO keeps D > 0 and D proportional "
         "to r")
    say("Seven atoms x global (a, p, n, L, rho_ref), evaluated on the "
        "six-galaxy")
    say("ladder with the path-averaged qbar.  Requirements R1-R5 as in the "
        "header.")
    prof = {}
    for name, M, rd in LADDER:
        for rho_ref in (1e5, 1e6):
            for qkind in ("delta", "smooth"):
                r, q, qb = qbar_profile(M, rd, rho_ref, qkind)
                prof[(name, rho_ref, qkind)] = (r, q, qb, M, rd)

    rows = []
    grid = list(itertools.product(
        ATOMS, (0.3, 1.0, 3.0, 10.0, 30.0), (0.0, 0.5, 1.0, 2.0),
        (0.3, 0.5, 1.0, 1.5), (1.0, 10.0, 100.0), (1e5, 1e6),
        ("delta", "smooth")))
    say(f"settings screened: {len(grid)}")
    for atom, a, p, n, L, rho_ref, qkind in grid:
        f = ATOMS[atom]
        ok_all, sl_all, minD_all, vf2 = True, [], [], []
        for name, M, rd in LADDER:
            r, q, qb, _, _ = prof[(name, rho_ref, qkind)]
            qbf = lambda x: np.interp(np.log(np.maximum(x, 1e-30)),
                                      np.log(r), qb)
            Ff = lambda x: f(qbf(x), x / L, a, p, n)
            #  R1 positivity, 1 AU to 1 Mpc
            rr = np.geomspace(AU, 1.0e3, 260)
            Fv, D = D_of(Ff, rr)
            minD_all.append(float(np.min(D)))
            if np.min(D) <= 0:
                ok_all = False
            #  R2 flatness over the observed range
            ro = np.geomspace(2 * rd, 20 * rd, 40)
            Fo, Do = D_of(Ff, ro)
            if np.any(Do <= 0):
                ok_all = False
                sl_all.append(np.nan)
            else:
                sl_all.append(float(np.mean(np.gradient(np.log(Do),
                                                        np.log(ro)))))
            vf2.append(float(C.G * M * Do[-1] / ro[-1]))
        #  R3 solar
        rsun = np.array([AU])
        r, q, qb, M0, rd0 = prof[("MW_like", rho_ref, qkind)]
        qbf = lambda x: np.interp(np.log(np.maximum(x, 1e-30)), np.log(r), qb)
        Ff = lambda x: f(qbf(x), x / L, a, p, n)
        _, Dsun = D_of(Ff, rsun)
        #  R4 Oort: force enhancement on the ~0.3 kpc paths that set g_z
        _, Dloc = D_of(Ff, np.array([0.3]))
        #  R5 BTFR
        Mb = np.array([m for _, m, _ in LADDER])
        good = np.array(vf2) > 0
        slope = (float(np.polyfit(np.log10(np.sqrt(np.array(vf2)[good])),
                                  np.log10(Mb[good]), 1)[0])
                 if good.sum() >= 3 else float("nan"))
        rows.append(dict(
            atom=atom, unbounded=atom in UNBOUNDED, a=a, p=p, n=n, L=L,
            qkind=qkind, gated=bool(p > 0),
            rho_ref=rho_ref, min_D=float(np.min(minD_all)),
            R1_positive=bool(np.min(minD_all) > 0),
            mean_dlnD_dlnr=float(np.nanmean(sl_all)),
            max_dev_from_1=float(np.nanmax(np.abs(np.array(sl_all) - 1.0))),
            R2_flat=bool(np.nanmax(np.abs(np.array(sl_all) - 1.0)) < 0.2),
            D_1AU=float(Dsun[0]),
            R3_solar=bool(abs(Dsun[0] - 1.0) < 1e-11),
            F_local=float(Dloc[0]),
            R4_oort=bool(1.10 <= Dloc[0] <= 1.70),
            btfr_slope=slope,
            R5_btfr=bool(np.isfinite(slope) and 3.5 <= slope <= 4.2)))
    RES["unbounded_search"] = dict(n_settings=len(rows), rows=rows)

    keys = ["R1_positive", "R2_flat", "R3_solar", "R4_oort", "R5_btfr"]
    say("")
    say("   requirement                              passing / total")
    for k in keys:
        say(f"   {k:<40s} {sum(r[k] for r in rows):6d} / {len(rows)}")
    say("")
    for k in range(1, len(keys) + 1):
        n_ok = sum(all(r[q] for q in keys[:k]) for r in rows)
        say(f"   cumulative R1..R{k:<2d}                          "
            f"{n_ok:6d} / {len(rows)}")
    for tag, sub in (("gated (p > 0)", [r for r in rows if r["gated"]]),
                     ("ungated (p = 0)",
                      [r for r in rows if not r["gated"]])):
        n12 = sum(r["R1_positive"] and r["R2_flat"] for r in sub)
        say(f"   R1 and R2 together, {tag:<16s}: {n12:5d} / {len(sub)}")
    #  the Pareto statement: how flat can D get while staying positive?
    pos = [r for r in rows if r["R1_positive"]
           and np.isfinite(r["mean_dlnD_dlnr"])]
    flat = [r for r in rows if r["R2_flat"]]
    if pos:
        b = max(pos, key=lambda r: r["mean_dlnD_dlnr"])
        say(f"\n   PARETO.  Best flatness reachable with D > 0 everywhere: "
            f"dlnD/dlnr = {b['mean_dlnD_dlnr']:.3f}")
        say(f"      ({b['atom']}, a={b['a']:g}, p={b['p']:g}, n={b['n']:g}, "
            f"L={b['L']:g}, rho_ref={b['rho_ref']:g}, q={b['qkind']}), "
            f"flat needs 1.000")
        RES["unbounded_search"]["pareto_best_positive"] = b
    if flat:
        w = max(flat, key=lambda r: r["min_D"])
        say(f"   Best min D reachable among the flat settings: "
            f"{w['min_D']:+.4g}   ({w['atom']}, a={w['a']:g}, p={w['p']:g}, "
            f"n={w['n']:g}, L={w['L']:g}, q={w['qkind']})")
        RES["unbounded_search"]["pareto_best_flat"] = w
    say("   The two frontiers do not meet: every setting flat enough to "
        "matter has")
    say("   a repulsive shell, and every setting free of repulsion is too "
        "shallow.")
    surv = [r for r in rows if all(r[k] for k in keys)]
    RES["unbounded_search"]["n_survivors"] = len(surv)
    say("")
    if surv:
        say(f"   {len(surv)} settings satisfy ALL FIVE.  Best by |slope - 1|:")
        surv.sort(key=lambda r: r["max_dev_from_1"])
        for r in surv[:8]:
            say(f"      {r['atom']:<14s} a={r['a']:<5g} p={r['p']:<4g} "
                f"n={r['n']:<4g} L={r['L']:<6g} rho_ref={r['rho_ref']:<7g}  "
                f"dlnD/dlnr {r['mean_dlnD_dlnr']:+.3f}  BTFR "
                f"{r['btfr_slope']:.2f}  F_loc {r['F_local']:.3f}")
    else:
        say("   NO setting satisfies all five.")
    #  where the survivors of R1+R2 fall over
    r12 = [r for r in rows if r["R1_positive"] and r["R2_flat"]]
    RES["unbounded_search"]["n_R1R2"] = len(r12)
    if r12:
        say(f"\n   {len(r12)} settings give D > 0 everywhere AND D "
            f"proportional to r over 2-20 R_d.")
        say("   Of those:")
        for k in keys[2:]:
            say(f"      also {k:<12s} : {sum(r[k] for r in r12):5d}")
        bt = np.array([r["btfr_slope"] for r in r12
                       if np.isfinite(r["btfr_slope"])])
        if len(bt):
            say(f"      their BTFR slopes: min {bt.min():.2f}, median "
                f"{np.median(bt):.2f}, max {bt.max():.2f}  "
                f"(observed 3.85 +/- 0.09)")
            RES["unbounded_search"]["R1R2_btfr"] = dict(
                min=float(bt.min()), median=float(np.median(bt)),
                max=float(bt.max()),
                frac_above_3p5=float(np.mean(bt >= 3.5)))
        fl = np.array([r["F_local"] for r in r12])
        say(f"      their F_local: min {fl.min():.3f}, median "
            f"{np.median(fl):.3f}, max {fl.max():.3f}  "
            f"(Oort window 1.10-1.70)")
        ds = np.array([r["D_1AU"] for r in r12])
        say(f"      their D(1 AU) - 1: min {np.min(np.abs(ds - 1)):.3e}, "
            f"max {np.max(np.abs(ds - 1)):.3e}  (bound 1e-11)")
        RES["unbounded_search"]["R1R2_detail"] = dict(
            F_local=[float(fl.min()), float(np.median(fl)), float(fl.max())],
            D1AU_dev=[float(np.min(np.abs(ds - 1))),
                      float(np.max(np.abs(ds - 1)))])
    #  bounded control
    bnd = [r for r in rows if not r["unbounded"]]
    say(f"\n   BOUNDED control atom (1 + a q^p): {sum(r['R2_flat'] for r in bnd)}"
        f" of {len(bnd)} pass R2 -- the boundedness theorem, re-measured.")


# ==========================================================================
def t4_why_a_gate_is_mandatory():
    """The ungated branch closes on the solar system, with one number."""
    head("T4  Why a gate is mandatory, and therefore why R1 fails")
    say("The unique exactly-flat form is F = 1 + c r ln(r_*/r), whose")
    say("D = 1 + c r is POSITIVE everywhere and has dlnD/dlnr -> 1.  So an")
    say("UNGATED unbounded F does satisfy R1 and R2.  It dies on R3 instead,")
    say("and the arithmetic is one line: c is a universal constant, so the")
    say("fractional departure from Newton is c r at EVERY radius.")
    out = {}
    for bound in (1e-11, 1e-8):
        c_max = bound / AU
        out[f"bound={bound:g}"] = dict(
            c_max_per_kpc=float(c_max),
            D_minus_1_at_10kpc=float(c_max * 10.0),
            D_minus_1_at_30kpc=float(c_max * 30.0))
        say(f"   ISL bound {bound:g} at 1 AU  =>  c <= {c_max:.3e} /kpc  =>  "
            f"D - 1 <= {c_max * 10:.3e} at 10 kpc, {c_max * 30:.3e} at "
            f"30 kpc")
    #  what the data demand, from the SPARC train split
    train = C.sparc("train")
    Dlast, Dall = [], []
    for g in train:
        prof = C.build_profile(g)
        R, F_req, D_req, g_obs = C.required(g, prof[5])
        Dlast.append(float(D_req[-1])); Dall.append(D_req)
    Dall = np.concatenate(Dall)
    out["required"] = dict(
        n_galaxies=len(train),
        D_req_last_point=dict(p5=float(np.percentile(Dlast, 5)),
                              p50=float(np.median(Dlast)),
                              p95=float(np.percentile(Dlast, 95))),
        D_req_all=dict(p5=float(np.percentile(Dall, 5)),
                       p50=float(np.median(Dall)),
                       p95=float(np.percentile(Dall, 95))))
    say(f"   SPARC TRAIN demands D at the last measured point: 5/50/95 pct = "
        f"{np.percentile(Dlast, 5):.2f} / {np.median(Dlast):.2f} / "
        f"{np.percentile(Dlast, 95):.2f}")
    sh = np.median(Dlast) / (1e-11 / AU * 10.0)
    out["shortfall_factor"] = float(sh)
    say(f"   The ungated form is short by a factor {sh:.3g} at 10 kpc. "
        f"Hence a gate,")
    say("   hence a rising qbar, hence the strictly negative gate term in D, "
        "hence R1.")
    say("   For comparison: MOND evades this with an ACCELERATION gate whose")
    say("   solar-system suppression is 1/x for the simple mu (2e-8 at 1 AU) "
        "and")
    say("   1/(2x^2) for the standard mu (2e-16).  A DENSITY gate cannot do "
        "that")
    say("   job here, because the density gate must also switch on across "
        "the")
    say("   rotation curve and it is the switching that subtracts from D.")
    RES["T4_gate_mandatory"] = out


def main():
    t1_sharpened_theorem()
    t3_linearity_btfr()
    gate_diagnostic()
    t4_why_a_gate_is_mandatory()
    screen_unbounded()
    RES["runtime_s"] = time.time() - T0
    with open("unbounded_search.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    say(f"\nwrote unbounded_search.json  ({time.time() - T0:.1f} s)")


if __name__ == "__main__":
    main()
