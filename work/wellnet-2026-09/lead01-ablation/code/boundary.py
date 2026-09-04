"""Item 3.  An operational definition of potential depth.

Newtonian potential is defined only up to an additive constant, so |Phi_b| is
not a physical quantity until the calculation fixes where Phi = 0.  Replace it
by a potential DIFFERENCE with a prespecified reference rule,

    DeltaPhi_b(r; r_ref) = Int_r^r_ref g_b(s) ds

and repeat the whole analysis under four defensible rules, one declared PRIMARY
in advance (see PREREGISTRATION.md, sealed 2026-09-04T12:05:06Z).

    BARY  PRIMARY   r_ref = 10 * r_half,b   (fixed multiple of a baryonic radius)
    PHYS            r_ref = 2000 kpc        (fixed physical radius)
    OVER            r_ref = r_200b, mean enclosed BARYONIC density = 200 rho_c
    TAIL            r_ref -> infinity with a point-mass tail  (existing convention)

Each system's baryonic profile is reconstructed from the ladder itself, in the
three forms the ladder actually used, and the reconstruction is GATED by
requiring that the TAIL rule reproduce the published |Phi_b| column.
"""
from __future__ import annotations

import json
import math
import os
import re

import numpy as np

import common as C
from ablation import paired_bootstrap

G = C.G
KPC = C.KPC
MSUN = C.MSUN
RHO_C0 = 9.2039e-27          # kg/m^3, h = 0.7, at z = 0 (see note in main)
R_PHYS_M = 2000.0 * KPC
K_BARY = 10.0
DELTA_OVER = 200.0


class Profile:
    """Baryonic profile of one system, in the form the ladder built it."""

    def __init__(self, kind, **kw):
        self.kind = kind
        self.__dict__.update(kw)

    # ---- enclosed baryonic mass -----------------------------------------
    def M(self, s):
        s = np.atleast_1d(np.asarray(s, float))
        if self.kind == "resolved":
            out = np.interp(np.log(s), np.log(self.r), np.log(self.Mgrid))
            out = np.exp(out)
            out[s > self.r[-1]] = self.Mgrid[-1]
            # inside the first point, hold the innermost power law
            in_ = s < self.r[0]
            if in_.any():
                k = math.log(self.Mgrid[1] / self.Mgrid[0]) / \
                    math.log(self.r[1] / self.r[0])
                out[in_] = self.Mgrid[0] * (s[in_] / self.r[0]) ** k
            return out
        m = np.where(s <= self.r_out,
                     self.M_tot * (s / self.r_out) ** self.k, self.M_tot)
        return m

    # ---- Int_a^b g_b ds, a < b (b may be inf) ---------------------------
    def integral(self, a, b):
        if b == a:
            return 0.0
        if b < a:
            return -self.integral(b, a)
        if self.kind == "resolved":
            r, g = self.r, self.g
            tot = 0.0
            hi = min(b, r[-1])
            if hi > a:
                # trapezoid on the tabulated grid, endpoints interpolated
                lo = max(a, r[0])
                if hi > lo:
                    xs = np.concatenate([[lo], r[(r > lo) & (r < hi)], [hi]])
                    gs = np.interp(xs, r, g)
                    tot += float(np.trapezoid(gs, xs))
                if a < r[0]:
                    # inside the first point: power-law M, same as M()
                    k = math.log(self.Mgrid[1] / self.Mgrid[0]) / \
                        math.log(self.r[1] / self.r[0])
                    hh = min(b, r[0])
                    tot += _plaw_int(a, hh, self.Mgrid[0], r[0], k)
            if b > r[-1]:
                lo = max(a, r[-1])
                tot += G * self.Mgrid[-1] * (1.0 / lo -
                                             (0.0 if math.isinf(b) else 1.0 / b))
            return tot
        tot = 0.0
        hi = min(b, self.r_out)
        if hi > a:
            tot += _plaw_int(a, hi, self.M_tot, self.r_out, self.k)
        if b > self.r_out:
            lo = max(a, self.r_out)
            tot += G * self.M_tot * (1.0 / lo -
                                     (0.0 if math.isinf(b) else 1.0 / b))
        return tot

    # ---- reference radii -------------------------------------------------
    def r_half(self):
        if self.kind == "resolved":
            target = 0.5 * self.Mgrid[-1]
            j = int(np.argmax(self.Mgrid >= target))
            if self.Mgrid[j] < target:
                return self.r[-1]
            if j == 0:
                return self.r[0]
            lo, hi = j - 1, j
            f = (math.log(target) - math.log(self.Mgrid[lo])) / \
                (math.log(self.Mgrid[hi]) - math.log(self.Mgrid[lo]))
            return math.exp(math.log(self.r[lo]) + f *
                            (math.log(self.r[hi]) - math.log(self.r[lo])))
        return self.r_out * 0.5 ** (1.0 / self.k)

    def r_over(self, delta=DELTA_OVER, rho_c=RHO_C0):
        """Radius where the mean enclosed BARYONIC density = delta * rho_c.
        No dark matter enters.  Solved on a fine log grid, then bisected."""
        rr = np.geomspace(self.rmin_grid, 3e5 * KPC, 4000)
        rho = 3.0 * self.M(rr) / (4.0 * math.pi * rr ** 3)
        tgt = delta * rho_c
        below = np.where(rho < tgt)[0]
        if len(below) == 0:
            return rr[-1]
        j = int(below[0])
        if j == 0:
            return rr[0]
        a, b = rr[j - 1], rr[j]
        for _ in range(80):
            mid = math.sqrt(a * b)
            if 3.0 * float(self.M(mid)[0]) / (4 * math.pi * mid ** 3) > tgt:
                a = mid
            else:
                b = mid
        return math.sqrt(a * b)


def _plaw_int(a, b, M_out, r_out, k):
    """Int_a^b G M_out (s/r_out)^k / s^2 ds  for a<b<=r_out."""
    if abs(k - 1.0) < 1e-9:
        return G * M_out / r_out * math.log(b / a)
    c = G * M_out * r_out ** (-k) / (k - 1.0)
    return c * (b ** (k - 1.0) - a ** (k - 1.0))


def build_profiles(d):
    """Reconstruct one Profile per system from the ladder, in the exact three
    forms ladder.py used.  Returns (profiles, per-system row index, notes)."""
    bysys = {}
    for i in range(len(d["system"])):
        bysys.setdefault(d["system"][i], []).append(i)
    profs, notes = {}, dict(resolved=0, two_radius=0, single_radius=0)
    ks = []
    for s, ix in bysys.items():
        pm = d["phi_method"][ix[0]]
        if pm.startswith("two measured radii"):
            o = sorted(ix, key=lambda j: d["r_kpc"][j])
            r_in, r_out = (d["r_kpc"][o[0]] * KPC, d["r_kpc"][o[1]] * KPC)
            M_in, M_out = (d["Mb_Msun"][o[0]] * MSUN, d["Mb_Msun"][o[1]] * MSUN)
            k = math.log(M_out / M_in) / math.log(r_out / r_in)
            k_str = float(re.search(r"s=([0-9.]+)", pm).group(1))
            assert abs(k - k_str) < 0.006, (s, k, k_str)
            ks.append(k)
    k_default = float(np.median(ks))
    for s, ix in bysys.items():
        pm = d["phi_method"][ix[0]]
        if pm.startswith("resolved"):
            o = sorted(ix, key=lambda j: d["r_kpc"][j])
            r = np.array([d["r_kpc"][j] for j in o]) * KPC
            g = np.array([d["g_bar"][j] for j in o])
            M = g * r ** 2 / G
            profs[s] = Profile("resolved", r=r, g=g, Mgrid=M,
                               rmin_grid=r[0] * 1e-3, rows=o)
            notes["resolved"] += 1
        elif pm.startswith("two measured radii"):
            o = sorted(ix, key=lambda j: d["r_kpc"][j])
            r_in, r_out = (d["r_kpc"][o[0]] * KPC, d["r_kpc"][o[1]] * KPC)
            M_in, M_out = (d["Mb_Msun"][o[0]] * MSUN, d["Mb_Msun"][o[1]] * MSUN)
            k = math.log(M_out / M_in) / math.log(r_out / r_in)
            profs[s] = Profile("power", r_out=r_out, M_tot=M_out, k=k,
                               rmin_grid=r_in * 1e-3, rows=o)
            notes["two_radius"] += 1
        else:
            j = ix[0]
            r_out = d["r_kpc"][j] * KPC
            profs[s] = Profile("power", r_out=r_out,
                               M_tot=d["Mb_Msun"][j] * MSUN, k=k_default,
                               rmin_grid=r_out * 1e-3, rows=[j],
                               assumed_index=True)
            notes["single_radius"] += 1
    notes["default_power_law_index_for_single_radius_systems"] = k_default
    return profs, notes


RULES = ("BARY", "PHYS", "OVER", "TAIL")
RULE_DESC = {
    "BARY": "PRIMARY.  r_ref = 10 x r_half,b (radius enclosing half the "
            "baryonic mass inside the outermost measured radius)",
    "PHYS": "r_ref = 2000 kpc for every system",
    "OVER": "r_ref = r_200b, mean enclosed BARYONIC density = 200 rho_c(0)",
    "TAIL": "r_ref -> infinity with a point-mass tail beyond the last measured "
            "radius (the existing convention)",
}


def r_ref_of(p, rule):
    if rule == "BARY":
        return K_BARY * p.r_half()
    if rule == "PHYS":
        return R_PHYS_M
    if rule == "OVER":
        return p.r_over()
    if rule == "TAIL":
        return math.inf
    raise KeyError(rule)


def compute(d, profs, rule, floor_frac=0.02):
    """log10|DeltaPhi_b(r; r_ref)| for every row, plus bookkeeping."""
    n = len(d["system"])
    out = np.full(n, np.nan)
    rref = np.full(n, np.nan)
    n_inside = 0          # rows where r_ref <= r  (sign flip)
    n_dropped = 0         # rows where |log(r_ref/r)| is too small to be usable
    for s, p in profs.items():
        rr = r_ref_of(p, rule)
        for j in p.rows:
            r = d["r_kpc"][j] * KPC
            rref[j] = rr / KPC
            if not math.isinf(rr):
                if abs(rr - r) < floor_frac * r:
                    n_dropped += 1
                    continue
                if rr < r:
                    n_inside += 1
            v = abs(p.integral(min(r, rr), max(r, rr)))
            out[j] = math.log10(v) if v > 0 else np.nan
    return out, rref, dict(n_rows_r_ref_inside_r=n_inside,
                           n_rows_dropped_degenerate=n_dropped,
                           n_rows_usable=int(np.isfinite(out).sum()))


def main():
    print("=" * 78)
    print("ITEM 3   AN OPERATIONAL DEFINITION OF POTENTIAL DEPTH")
    print("=" * 78)
    d = C.load_ladder()
    profs, notes = build_profiles(d)
    print(f"\nreconstructed {len(profs)} system profiles: "
          f"{notes['resolved']} resolved, {notes['two_radius']} two-radius, "
          f"{notes['single_radius']} single-radius")
    print(f"single-radius systems use the median two-radius power-law index "
          f"k = {notes['default_power_law_index_for_single_radius_systems']:.4f}"
          f" (a DECLARED assumption; those rows are lower bounds either way)")
    print(f"max radius anywhere in the ladder: {d['r_kpc'].max():.1f} kpc "
          f"(the fixed-physical rule uses 2000 kpc, so it is outside every row)")

    res = dict(prereg_sha256=json.load(
        open(os.path.join(C.LANE, "prereg_seal.json")))["prereg_sha256"],
        primary_rule="BARY", rule_descriptions=RULE_DESC,
        reconstruction=notes,
        cosmology_note="rho_c evaluated at z = 0 for every system.  The ladder "
                       "spans z ~ 0 to 0.13; E(z)^2 changes rho_c by at most "
                       "13% there, hence r_200b by at most 4%, which is far "
                       "below the differences between the four rules.")

    # ---- GATE: the TAIL rule must reproduce the published |Phi_b| --------
    lp_tail, _, bk = compute(d, profs, "TAIL")
    err = lp_tail - d["lp"]
    ok = np.isfinite(err)
    print(f"\nGATE  TAIL rule vs the published log|Phi_b| column: "
          f"max |diff| = {np.nanmax(np.abs(err)):.3e} dex over "
          f"{int(ok.sum())} rows")
    assert np.nanmax(np.abs(err)) < 1e-9, "profile reconstruction does not " \
        "reproduce the published potential"
    res["reconstruction_gate_max_abs_diff_dex"] = float(np.nanmax(np.abs(err)))
    print("      -> the reconstruction is the ladder's own convention, exactly.")

    # ---- the four rules ---------------------------------------------------
    t_ref = C.system_table(d)
    tr_ref = t_ref["rank"] <= 4
    te_ref = t_ref["rank"] >= 5
    per_rule = {}
    print(f"\n{'rule':<6} {'n rows':>7} {'inside':>7} {'drop':>5} "
          f"{'nsys':>5} {'beta':>9} {'q':>8} {'transfer M1':>12} "
          f"{'transfer M3':>12} {'dRMS':>9}")
    lps = {}
    for rule in RULES:
        lp, rref, bk = compute(d, profs, rule)
        lps[rule] = lp
        t = C.system_table(d, lp_override=lp)
        tr = t["rank"] <= 4
        te = t["rank"] >= 5
        c1, r1, rms1 = C.fit_freeze_eval(t, tr, te, "M1")
        c3, r3, rms3 = C.fit_freeze_eval(t, tr, te, "M3")
        _, _, rms0 = C.fit_freeze_eval(t, tr, te, "M0")
        ca, *_ = np.linalg.lstsq(C.design(t, "M1"), t["dev"], rcond=None)
        # collinearity and the Run Z identity, under this rule
        lgv, lrv = t["lg"], t["lr"]
        A = np.column_stack([np.ones(len(lgv)), lgv, lgv ** 2, lrv])
        cc, *_ = np.linalg.lstsq(A, t["lp"], rcond=None)
        resid_lp = t["lp"] - A @ cc
        # shape factor under this rule: S = |DeltaPhi| / (g_bar r)
        lS = t["lp"] - t["lg"] - (t["lr"] + math.log10(KPC))
        cS, *_ = np.linalg.lstsq(A, lS, rcond=None)
        resid_S = lS - A @ cS
        corr_id = float(np.corrcoef(resid_lp, resid_S)[0, 1])
        # partial corr(lp, lr | lg)
        B = np.column_stack([np.ones(len(lgv)), lgv, lgv ** 2])
        e1 = t["lp"] - B @ np.linalg.lstsq(B, t["lp"], rcond=None)[0]
        e2 = lrv - B @ np.linalg.lstsq(B, lrv, rcond=None)[0]
        pcorr = float(np.corrcoef(e1, e2)[0, 1])
        pb = paired_bootstrap(t, tr, te, "M1", "M3", nb=4000)
        rec = dict(rule=rule, description=RULE_DESC[rule], **bk,
                   n_systems=len(t["lg"]),
                   beta_train_rungs1to4=float(c1[3]), q_implied=float(2 * c1[3]),
                   beta_all=float(ca[3]),
                   transfer_rms_M0=rms0, transfer_rms_M1=rms1,
                   transfer_rms_M3=rms3,
                   delta_rms_M1_minus_M3=float(rms1 - rms3),
                   paired_ci95=pb["ci95"], p_M1_better=pb["p_A_better"],
                   n_objects_M1_better=pb["n_objects_A_better"],
                   lp_sd=float(t["lp"].std()),
                   lp_range=[float(t["lp"].min()), float(t["lp"].max())],
                   partial_corr_lp_lr_given_lg=pcorr,
                   corr_residual_lp_with_residual_logS=corr_id,
                   median_r_ref_kpc=float(np.nanmedian(rref)) if rule != "TAIL"
                   else None,
                   r_ref_kpc_by_rank={
                       str(k): (float(np.nanmedian(rref[d["rank"] == k]))
                                if rule != "TAIL" else None)
                       for k in sorted(set(d["rank"].tolist()))})
        per_rule[rule] = rec
        print(f"{rule:<6} {bk['n_rows_usable']:7d} "
              f"{bk['n_rows_r_ref_inside_r']:7d} "
              f"{bk['n_rows_dropped_degenerate']:5d} {len(t['lg']):5d} "
              f"{c1[3]:+9.4f} {2 * c1[3]:+8.4f} {rms1:12.4f} {rms3:12.4f} "
              f"{rms1 - rms3:+9.5f}")
    res["rules"] = per_rule

    bs = [per_rule[r]["beta_train_rungs1to4"] for r in RULES]
    ts = [per_rule[r]["transfer_rms_M1"] for r in RULES]
    ds = [per_rule[r]["delta_rms_M1_minus_M3"] for r in RULES]
    res["spread_across_rules"] = dict(
        beta_min=min(bs), beta_max=max(bs), beta_range=max(bs) - min(bs),
        beta_relative_range=float((max(bs) - min(bs)) / abs(np.mean(bs))),
        q_min=2 * min(bs), q_max=2 * max(bs),
        transfer_rms_M1_min=min(ts), transfer_rms_M1_max=max(ts),
        transfer_rms_M1_range=max(ts) - min(ts),
        delta_rms_min=min(ds), delta_rms_max=max(ds),
        published_beta_shift_when_refit_on_everything=abs(
            C.REF["beta_train_rungs1to4"] - C.REF["beta_all"]),
        published_beta_shift_percent=100 * abs(
            C.REF["beta_train_rungs1to4"] - C.REF["beta_all"])
        / C.REF["beta_train_rungs1to4"])
    sp = res["spread_across_rules"]
    print(f"\nSPREAD ACROSS THE FOUR RULES")
    print(f"   beta         {sp['beta_min']:+.4f} ... {sp['beta_max']:+.4f}"
          f"   range {sp['beta_range']:.4f} = "
          f"{sp['beta_relative_range']:.1%} of the mean")
    print(f"   implied q    {sp['q_min']:+.4f} ... {sp['q_max']:+.4f}")
    print(f"   transfer M1  {sp['transfer_rms_M1_min']:.4f} ... "
          f"{sp['transfer_rms_M1_max']:.4f}   range "
          f"{sp['transfer_rms_M1_range']:.4f} dex")
    print(f"   dRMS(M1-M3)  {sp['delta_rms_min']:+.5f} ... "
          f"{sp['delta_rms_max']:+.5f}")
    print(f"   for scale, the published train/all refit moved beta by "
          f"{sp['published_beta_shift_percent']:.2f}% "
          f"({sp['published_beta_shift_when_refit_on_everything']:.5f})")

    print(f"\n   Run Z identity check -- corr(residual log|DeltaPhi|, residual "
          f"log S) after controlling log g_bar, its square and log r:")
    for r in RULES:
        print(f"      {r:<6} {per_rule[r]['corr_residual_lp_with_residual_logS']:+.6f}"
              f"    partial corr(lp, lr | lg) = "
              f"{per_rule[r]['partial_corr_lp_lr_given_lg']:+.4f}")

    print(f"\n   median r_ref by rung (kpc):")
    hdr = "      " + "rung".ljust(6) + "".join(f"{r:>12}" for r in RULES)
    print(hdr)
    for k in sorted(set(d["rank"].tolist())):
        row = "      " + str(k).ljust(6)
        for r in RULES:
            v = per_rule[r]["r_ref_kpc_by_rank"][str(k)]
            row += f"{'inf':>12}" if v is None else f"{v:12.1f}"
        print(row)

    # ---- the ablation, repeated under the PRIMARY rule --------------------
    print("\n   THE ABLATION REPEATED UNDER THE PRIMARY RULE (BARY):")
    t = C.system_table(d, lp_override=lps["BARY"])
    gal = t["rank"] == 1
    grp = (t["rank"] >= 2) & (t["rank"] <= 4)
    clu = t["rank"] >= 5
    abl = {}
    print(f"      {'arm':<18} {'model':<4} {'rms':>8} {'beta':>9}")
    for aname, trm in (("galaxies only", gal), ("groups only", grp),
                       ("galaxies+groups", gal | grp)):
        for m in ("M0", "M1", "M3"):
            c, r_, rms = C.fit_freeze_eval(t, trm, clu, m)
            abl[f"{aname}|{m}"] = dict(rms=rms, coef=[float(x) for x in c],
                                       bias=float(r_.mean()),
                                       scatter=float(r_.std()))
            b = f"{c[3]:+9.4f}" if len(c) > 3 else "        -"
            print(f"      {aname:<18} {m:<4} {rms:8.4f} {b}")
    res["ablation_under_primary_rule"] = abl

    out = os.path.join(C.LANE, "boundary_sensitivity.json")
    json.dump(res, open(out, "w"), indent=2)
    print(f"\nwrote {out}")
    return res, lps


if __name__ == "__main__":
    main()
