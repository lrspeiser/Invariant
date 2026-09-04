"""JOB 1 -- the F - r F' stability audit.

The previous lane established every rotation-curve number from F, and F alone.
The force is not F, it is

    g(r) = (G M / r^2) D(r) ,    D = F - r F'

and at the measured dlnF/dlnr = 0.899 that is D = 0.101 F: a ten-to-one
cancellation.  This script perturbs every numerical knob, one at a time and
then jointly, and reports the fractional change in D (not in F).

Knobs audited
  A  radial resolution of the source/q grid            n_grid
  B  interpolation rule for q                          loglin/lin/pchip/cubic/akima
  C  smoothing scale                                   L_s (numerical), L_q (physical)
  D  kernel-resolution parameters                      n_D, n_s, n_gl, dlnr_max,
                                                       and the DERIVATIVE ROUTE
                                                       (analytic vs log stencil)
  E  source-boundary handling                          r_lo, r_hi, tail scale,
                                                       cosmic floor in q
  F  all of the above jointly

Everything is measured on the SPARC TRAIN split only, at the tabulated radii
beyond 2 R_disk, with the frozen parameter sets the previous lane selected.
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

import common as C
import dcore as DC

GPU = True
RES = {}
T0 = time.time()


def say(*a):
    print(*a, flush=True)


def head(t):
    say("\n" + "=" * 78)
    say(t)
    say("=" * 78)


# --------------------------------------------------------------------------
_PROF = {}
_FLD = {}


def prof_of(g, pkey):
    k = (g.name,) + pkey
    if k not in _PROF:
        n, r_lo, r_hi, tm = pkey
        _PROF[k] = C.build_profile(g, n=n, r_lo=r_lo, r_hi=r_hi, tail_mult=tm)
    return _PROF[k]


def fld_of(g, pkey, qkey):
    k = (g.name,) + pkey + qkey
    if k not in _FLD:
        rho_ref, L_q, L_s, floor = qkey
        _FLD[k] = C.build_field(prof_of(g, pkey), qdef="screen",
                                rho_ref=rho_ref, L_q=L_q, L_s=L_s,
                                rho_floor=floor, label=g.name)
    return _FLD[k]


PKEY0 = (1400, 1e-3, 3.0e4, 1.0)
QKEY0 = (1e6, 2.0, 0.0, C.NK.RHO_BAR_B)
QUAD0 = dict(n_D=32, n_s=12, n_gl=8, dlnr_max=0.35)
RULE0 = "loglin"


def run_config(gals, kern, pkey=PKEY0, qkey=QKEY0, quad=None, rule=RULE0,
               r_lo=None, r_hi=None, stencil=None):
    """D and F_eff at the measured radii for every galaxy.  Returns dict
    name -> (R, F_eff, D)."""
    quad = dict(QUAD0 if quad is None else quad)
    out = {}
    for g in gals:
        prof = prof_of(g, pkey)
        fld = fld_of(g, pkey, qkey)
        R, Fr, Dr, go = C.required(g, prof[5])
        kw = dict(Fname=kern["Fname"], alpha=kern["alpha"],
                  beta=kern["beta"], p=kern["p"], Mtot=prof[5])
        if stencil is None:
            qp = DC.QProfile(fld.r, fld.q, rule)
            Fe, D = DC.phi_and_D(fld, R, qprof=qp, use_gpu=GPU, chunk=64,
                                 r_lo=r_lo, r_hi=r_hi, **quad, **kw)
        else:
            Fe, D = DC.D_stencil(fld, R, dlog=stencil, use_gpu=GPU, chunk=64,
                                 r_lo=r_lo, r_hi=r_hi, **quad, **kw)
        out[g.name] = (R, Fe, D)
    return out


def compare(ref, alt, floor_frac=0.02):
    """Fractional change in D and in F between two runs.

    `floor_frac`: points where |D_ref| < floor_frac * F_ref are excluded from
    the RELATIVE statistic (D crosses zero there and a ratio is meaningless);
    they are covered instead by dD/F, which is well behaved everywhere.  Both
    are reported, with the count of excluded points.
    """
    rd, rf, ad, af, n_ex, n_tot = [], [], [], [], 0, 0
    for k in ref:
        R, F0, D0 = ref[k]
        R1, F1, D1 = alt[k]
        n_tot += len(R)
        ok = np.abs(D0) >= floor_frac * np.abs(F0)
        n_ex += int((~ok).sum())
        if ok.any():
            rd.append(np.abs(D1[ok] / D0[ok] - 1.0))
        rf.append(np.abs(F1 / F0 - 1.0))
        ad.append(np.abs(D1 - D0) / np.abs(F0))
        af.append(np.abs(F1 - F0) / np.abs(F0))
    rd = np.concatenate(rd) if rd else np.array([np.nan])
    rf = np.concatenate(rf)
    ad = np.concatenate(ad)
    return dict(
        dD_rel_med=float(np.median(rd)), dD_rel_p95=float(np.percentile(rd, 95)),
        dD_rel_max=float(np.max(rd)),
        dD_over_F_med=float(np.median(ad)), dD_over_F_max=float(np.max(ad)),
        dF_rel_med=float(np.median(rf)), dF_rel_max=float(np.max(rf)),
        n_excluded=n_ex, n_points=n_tot,
        amplification=float(np.median(rd) / max(np.median(rf), 1e-300)))


def line(tag, c):
    say(f"   {tag:<38s} dD/D med {c['dD_rel_med']:9.3e}  p95 "
        f"{c['dD_rel_p95']:9.3e}  max {c['dD_rel_max']:9.3e}   |  dF/F med "
        f"{c['dF_rel_med']:9.3e}   amp x{c['amplification']:8.1f}")


# ==========================================================================
def main():
    head("JOB 1  --  stability of D = F - r F' (NOT of F)")
    train = C.sparc("train")
    say(f"SPARC TRAIN galaxies usable (>=3 points beyond 2 R_disk): "
        f"{len(train)}")
    npts = sum(int(np.sum(g.R0 >= 2 * g.Rdisk)) for g in train)
    say(f"radial points entering every statistic                  : {npts}")
    RES["sample"] = dict(n_galaxies=len(train), n_points=npts,
                         split="train", cut="R >= 2 R_disk",
                         upsilon_disk=C.UPS_DISK, upsilon_bulge=C.UPS_BULGE)

    # ---------------------------------------------------------------- G0
    head("G0  Exactness gates on the analytic D route")
    g0 = {}
    prof = prof_of(train[0], PKEY0)
    fld = fld_of(train[0], PKEY0, QKEY0)
    R = train[0].R0[train[0].R0 >= 2 * train[0].Rdisk]
    Fe, D = DC.phi_and_D(fld, R, Fname="F1_poly", alpha=0.0, p=1.0,
                         Mtot=prof[5], use_gpu=GPU, **QUAD0)
    ex = fld.Menc_at(R) / prof[5]
    g0["newton_D_rel_err"] = float(np.max(np.abs(D / ex - 1)))
    say(f"alpha = 0 : D must equal M(<r)/M_tot exactly.  max rel err "
        f"{g0['newton_D_rel_err']:.3e}")
    #  gauge identity: F -> F + c r leaves D unchanged.  Verified on the
    #  closed form rather than the solver, since it is an algebraic identity.
    c = 3.7
    rr = np.geomspace(1, 100, 50)
    Ff = 1 + 2 * rr ** 0.5
    Dd = Ff - rr * np.gradient(Ff, rr)
    Ff2 = Ff + c * rr
    Dd2 = Ff2 - rr * np.gradient(Ff2, rr)
    g0["gauge_D_invariance"] = float(np.max(np.abs(Dd2 - Dd)))
    say(f"gauge check  F -> F + {c} r leaves D unchanged to "
        f"{g0['gauge_D_invariance']:.2e}   (so F is not an observable, D is)")
    RES["G0_gates"] = g0

    for kern in (C.KERNEL_BEST, C.KERNEL_LOCAL):
        tag = kern["tag"]
        head(f"AUDIT for the frozen set '{tag}':  {kern['Fname']} "
             f"alpha={kern['alpha']} beta={kern['beta']} p={kern['p']}, "
             f"screen rho_ref=1e6 L_q=2")
        blk = {}
        base = run_config(train, kern)
        Dall = np.concatenate([v[2] for v in base.values()])
        Fall = np.concatenate([v[1] for v in base.values()])
        say(f"baseline: D median {np.median(Dall):.4f}, "
            f"D/F median {np.median(Dall / Fall):.4f}, "
            f"fraction of points with D <= 0 : "
            f"{np.mean(Dall <= 0):.4f}")
        blk["baseline"] = dict(D_median=float(np.median(Dall)),
                               DoverF_median=float(np.median(Dall / Fall)),
                               frac_D_nonpositive=float(np.mean(Dall <= 0)))

        # ---- A radial resolution
        say("\nA  RADIAL RESOLUTION of the source/q grid  (baseline n = 1400)")
        blk["A_radial_resolution"] = {}
        for n in (350, 700, 2800, 5600):
            alt = run_config(train, kern, pkey=(n, 1e-3, 3.0e4, 1.0))
            c_ = compare(base, alt)
            blk["A_radial_resolution"][f"n={n}"] = c_
            line(f"n_grid {n}  vs 1400", c_)

        # ---- B interpolation rule
        say("\nB  INTERPOLATION RULE for q  (baseline: log-linear, the rule "
            "the existing kernel uses)")
        blk["B_interpolation"] = {}
        for rule in ("lin", "pchip", "cubic", "akima"):
            alt = run_config(train, kern, rule=rule)
            c_ = compare(base, alt)
            blk["B_interpolation"][rule] = c_
            line(f"rule {rule}  vs loglin", c_)

        # ---- C smoothing scales
        say("\nC  SMOOTHING SCALES.  L_s is a NUMERICAL pre-smoothing of rho "
            "(baseline 0);")
        say("   L_q is the PHYSICAL screening length of the q equation "
            "(baseline 2 kpc).")
        blk["C_smoothing"] = {}
        for L_s in (0.05, 0.1, 0.2, 0.5):
            alt = run_config(train, kern, qkey=(1e6, 2.0, L_s,
                                                C.NK.RHO_BAR_B))
            c_ = compare(base, alt)
            blk["C_smoothing"][f"L_s={L_s}"] = c_
            line(f"L_s {L_s} kpc  vs 0 (numerical)", c_)
        for L_q in (1.0, 1.5, 3.0, 4.0):
            alt = run_config(train, kern, qkey=(1e6, L_q, 0.0,
                                                C.NK.RHO_BAR_B))
            c_ = compare(base, alt)
            blk["C_smoothing"][f"L_q={L_q}"] = c_
            line(f"L_q {L_q} kpc  vs 2 (PHYSICAL)", c_)

        # ---- D kernel resolution + derivative route
        say("\nD  KERNEL-RESOLUTION parameters  (baseline n_D=32 n_s=12 "
            "n_gl=8 dlnr_max=0.35)")
        blk["D_kernel_resolution"] = {}
        for k_, vals in (("n_D", (8, 16, 64, 128)), ("n_s", (4, 8, 24, 48)),
                         ("n_gl", (4, 6, 12, 16)),
                         ("dlnr_max", (0.7, 0.5, 0.2, 0.1))):
            for v in vals:
                q = dict(QUAD0); q[k_] = v
                alt = run_config(train, kern, quad=q)
                c_ = compare(base, alt)
                blk["D_kernel_resolution"][f"{k_}={v}"] = c_
                line(f"{k_} {v}  vs {QUAD0[k_]}", c_)

        say("\nD'  DERIVATIVE ROUTE: the five-point log stencil the previous "
            "lane used, against the analytic D.")
        blk["D_derivative_route"] = {}
        for dlog in (1e-1, 3e-2, 1e-2, 2e-3, 1e-3, 3e-4, 1e-4):
            alt = run_config(train, kern, stencil=dlog)
            c_ = compare(base, alt)
            blk["D_derivative_route"][f"dlog={dlog:g}"] = c_
            line(f"stencil dlog {dlog:g}  vs analytic", c_)

        # ---- E boundaries
        say("\nE  SOURCE-BOUNDARY handling")
        blk["E_boundaries"] = {}
        for rl in (1e-2, 1e-4):
            alt = run_config(train, kern, pkey=(1400, rl, 3.0e4, 1.0))
            c_ = compare(base, alt)
            blk["E_boundaries"][f"r_lo={rl:g}"] = c_
            line(f"inner limit r_lo {rl:g}  vs 1e-3", c_)
        for rh in (3.0e3, 3.0e5):
            alt = run_config(train, kern, pkey=(1400, 1e-3, rh, 1.0))
            c_ = compare(base, alt)
            blk["E_boundaries"][f"r_hi={rh:g}"] = c_
            line(f"outer limit r_hi {rh:g}  vs 3e4", c_)
        for tm in (0.5, 2.0):
            alt = run_config(train, kern, pkey=(1400, 1e-3, 3.0e4, tm))
            c_ = compare(base, alt)
            blk["E_boundaries"][f"tail_mult={tm}"] = c_
            line(f"outer mass-tail scale x{tm}  vs x1", c_)
        #  truncating the POTENTIAL integral inside the grid
        for rh in (3.0e2, 3.0e3):
            alt = run_config(train, kern, r_hi=rh)
            c_ = compare(base, alt)
            blk["E_boundaries"][f"integrate_to={rh:g}"] = c_
            line(f"truncate the r' integral at {rh:g} kpc", c_)
        #  the cosmic floor inside q
        alt = run_config(train, kern, qkey=(1e6, 2.0, 0.0, 0.0))
        c_ = compare(base, alt)
        blk["E_boundaries"]["no_cosmic_floor_in_q"] = c_
        line("drop the cosmic floor from q", c_)

        # ---- F joint
        say("\nF  JOINT perturbation: every numerical knob moved to a much "
            "finer setting at once.")
        fine = run_config(train, kern, pkey=(5600, 1e-4, 3.0e5, 1.0),
                          quad=dict(C.FINE_QUAD), rule="cubic")
        c_ = compare(base, fine)
        blk["F_joint_fine"] = c_
        line("baseline vs ALL-FINE reference", c_)
        coarse = run_config(train, kern, pkey=(350, 1e-2, 3.0e3, 1.0),
                            quad=dict(n_D=8, n_s=4, n_gl=4, dlnr_max=0.7),
                            rule="lin")
        c_ = compare(fine, coarse)
        blk["F_joint_coarse_vs_fine"] = c_
        line("ALL-COARSE vs ALL-FINE reference", c_)

        # ---- convergence ladder against the fine reference
        say("\nF'  CONVERGENCE LADDER, each level measured against the "
            "ALL-FINE reference.  The 1% line is the brief's target.")
        blk["F_convergence_ladder"] = {}
        ladder = [
            ("L0 coarse", dict(pkey=(350, 1e-2, 3.0e3, 1.0),
                               quad=dict(n_D=8, n_s=4, n_gl=4, dlnr_max=0.7),
                               rule="lin")),
            ("L1 production (previous lane)", dict()),
            ("L2", dict(pkey=(2800, 1e-3, 3.0e4, 1.0),
                        quad=dict(n_D=48, n_s=16, n_gl=10, dlnr_max=0.25),
                        rule="loglin")),
            ("L2p pchip", dict(pkey=(2800, 1e-3, 3.0e4, 1.0),
                               quad=dict(n_D=48, n_s=16, n_gl=10,
                                         dlnr_max=0.25), rule="pchip")),
            ("L3", dict(pkey=(2800, 1e-4, 1.0e5, 1.0),
                        quad=dict(n_D=64, n_s=24, n_gl=12, dlnr_max=0.15),
                        rule="cubic")),
        ]
        for tag_, kw in ladder:
            alt = run_config(train, kern, **kw)
            c_ = compare(fine, alt)
            blk["F_convergence_ladder"][tag_] = c_
            line(f"{tag_}", c_)

        RES[f"audit_{tag}"] = blk

    # ------------------------------------------------------------------
    head("WHERE D COMES CLOSEST TO ZERO ACROSS SPARC")
    say("Sign reversal happens when D <= 0, i.e. dlnF/dlnr >= 1.  The natural")
    say("margin is m = D/F = 1 - dlnF/dlnr; m = 0 is the reversal surface.")
    zz = {}
    for kern in (C.KERNEL_BEST, C.KERNEL_LOCAL):
        base = run_config(train, kern)
        rows = []
        for g in train:
            R, F_, D_ = base[g.name]
            mrg = D_ / F_
            i = int(np.argmin(mrg))
            rows.append(dict(name=g.name, margin_min=float(mrg[i]),
                             r_kpc=float(R[i]), r_over_Rd=float(R[i] / g.Rdisk),
                             Vflat=float(g.Vflat), Mb=float(g.Mb),
                             D_min=float(D_[i]), F_at=float(F_[i]),
                             any_negative=bool(np.any(D_ <= 0))))
        rows.sort(key=lambda x: x["margin_min"])
        mm = np.array([r["margin_min"] for r in rows])
        rr = np.array([r["r_over_Rd"] for r in rows])
        neg = [r for r in rows if r["any_negative"]]
        say(f"\n{kern['tag']}: galaxies with D <= 0 somewhere beyond 2 R_d : "
            f"{len(neg)} / {len(rows)}")
        say(f"   margin m = D/F, percentiles 0/5/25/50/95 : "
            f"{np.min(mm):+.3f} / {np.percentile(mm, 5):+.3f} / "
            f"{np.percentile(mm, 25):+.3f} / {np.median(mm):+.3f} / "
            f"{np.percentile(mm, 95):+.3f}")
        say(f"   the minimum sits at r/R_disk : median {np.median(rr):.2f}, "
            f"5-95 pct {np.percentile(rr, 5):.2f}-{np.percentile(rr, 95):.2f}")
        say("   five closest to reversal:")
        for r_ in rows[:5]:
            say(f"      {r_['name']:<12s} m = {r_['margin_min']:+.4f} at "
                f"r = {r_['r_kpc']:6.2f} kpc = {r_['r_over_Rd']:5.2f} R_d, "
                f"V_flat {r_['Vflat']:5.1f}")
        zz[kern["tag"]] = dict(
            n_negative=len(neg), n_total=len(rows),
            margin_pct=dict(p0=float(np.min(mm)),
                            p5=float(np.percentile(mm, 5)),
                            p25=float(np.percentile(mm, 25)),
                            p50=float(np.median(mm)),
                            p95=float(np.percentile(mm, 95))),
            r_over_Rd_of_min=dict(p5=float(np.percentile(rr, 5)),
                                  p50=float(np.median(rr)),
                                  p95=float(np.percentile(rr, 95))),
            per_galaxy=rows)

    # what the DATA themselves demand
    say("\nWhat the DATA demand, for comparison: F_req from the observed "
        "curves.")
    rows = []
    for g in train:
        prof = prof_of(g, PKEY0)
        R, Fr, Dr, go = C.required(g, prof[5])
        mrg = Dr / Fr
        i = int(np.argmin(mrg))
        rows.append(dict(name=g.name, margin_min=float(mrg[i]),
                         r_kpc=float(R[i]), r_over_Rd=float(R[i] / g.Rdisk)))
    mm = np.array([r["margin_min"] for r in rows])
    rr = np.array([r["r_over_Rd"] for r in rows])
    say(f"   required margin 1 - dlnF_req/dlnr, percentiles 0/5/50/95 : "
        f"{np.min(mm):.4f} / {np.percentile(mm, 5):.4f} / "
        f"{np.median(mm):.4f} / {np.percentile(mm, 95):.4f}")
    say(f"   attained at r/R_disk : median {np.median(rr):.2f}")
    rows.sort(key=lambda x: x["margin_min"])
    for r_ in rows[:5]:
        say(f"      {r_['name']:<12s} m = {r_['margin_min']:.4f} at "
            f"r = {r_['r_kpc']:6.2f} kpc = {r_['r_over_Rd']:5.2f} R_d")
    zz["data_required"] = dict(
        margin_pct=dict(p0=float(np.min(mm)), p5=float(np.percentile(mm, 5)),
                        p50=float(np.median(mm)),
                        p95=float(np.percentile(mm, 95))),
        r_over_Rd_of_min_median=float(np.median(rr)),
        closest=rows[:10])
    RES["closest_to_reversal"] = zz

    RES["runtime_s"] = time.time() - T0
    with open("stability_audit.json", "w", encoding="utf-8") as fh:
        json.dump(RES, fh, indent=1, default=float)
    say(f"\nwrote stability_audit.json   ({time.time() - T0:.1f} s)")


if __name__ == "__main__":
    main()
