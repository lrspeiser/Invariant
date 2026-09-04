"""THE JOINT TOURNAMENT.

Every candidate is scored on all four channels at once, with the same two
global constants, and must survive the structural screens.  No candidate is
promoted for winning one channel.

    python tournament.py            full run  -> tournament.json, gates.json
    python tournament.py --quick    a tenth of the grid, same structure

FREEZE ORDER, declared before any residual is looked at.
    1. a0 is fitted on the SPARC TRAIN split only (frozen split e5f74522).
    2. The amplitude A is fitted on the cluster channel.  Galaxies cannot see
       it -- for every gate that survives the galaxy screens the response is
       off at galaxy depths by construction -- so this is not a second bite at
       channel 1.  Where a gate DOES fire in galaxies the two fits are
       alternated to convergence, and that is recorded per candidate.
    3. The two VERTICAL channels are then pure predictions with no free
       parameter left.  Channels 1 and 4 are in-sample; channels 2 and 3 are
       out-of-sample.  Said plainly rather than buried.
    4. The SPARC blind and validation splits, KiDS and the wide binaries are
       never loaded.  There is no code path in this lane that reads them.

WHAT ELIMINATES WHAT is reported as a funnel, screen by screen, with the count
that falls at each stage and the quantified reason.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ch_cluster as CC                                          # noqa: E402
import ch_radial as CR                                           # noqa: E402
import ch_vertical as CV                                         # noqa: E402
import screens as SS                                             # noqa: E402
import tw_core as TC                                             # noqa: E402
from tw_core import A0, KPC, MSUN, Candidate, W_of, n_params      # noqa: E402

QUICK = "--quick" in sys.argv
T0 = time.time()

# ===================================================================== grid
BASES = ("rar", "aqual")
INV_SCALES = {
    "gn": (0.1, 1.0, 10.0),
    "phi": (1.0e11, 1.0e12, 3.0e12),
    "rhobar": (1.0e-25, 1.0e-24, 1.0e-23),
    "tidal": (1.0e-33, 1.0e-32, 1.0e-31),
    "qbar": (0.3, 0.6, 0.9),
}
FORM_M = (("sat", 1.0), ("sat", 2.0), ("sat", 4.0),
          ("pow", 0.5), ("pow", 1.0), ("pow", 2.0),
          ("pow", -1.0), ("pow", -2.0),
          ("log", 1.0), ("log", 2.0),
          ("inv", 1.0), ("inv", 2.0), ("inv", 4.0))
STRUCTS = ("scalar_a0", "iso_K", "tensor_d", "tensor_T")
WELLS_MAIN = CC.WELL_SETTINGS
#: Hybrid amplitude grid: dense and LINEAR where the tensor amplitudes live
#: (the tensor lane scans -40..0 on 141 linear points), with log tails to
#: +-1000 so a structure needing a very large or very small amplitude is not
#: silently pinned at a grid edge.  Landing on an edge is recorded per
#: candidate (`at_amp_grid_edge`) rather than hidden.
AMPS = np.unique(np.concatenate([
    -np.logspace(3, 1.79, 40), np.linspace(-60.0, 60.0, 241),
    np.logspace(1.79, 3, 40)]))

# ================================================== declared tolerances
TOL = dict(
    radial_dex=0.11,        # the RAR bar named in the brief
    cluster_dex=0.10,       # the tensor lane's own "within 0.10 dex" criterion
    vert_amp_dex=0.192,     # Run L's measured posterior width on log10 B_z
    galaxy_dex=0.040,       # RAR intrinsic scatter: field AND member galaxies
    vert_shape_chi2=40.0,   # iso tensor was rejected at 133; Newton 10.5
    radial_max_dex=0.30,    # a law worse than this is not a rotation-curve law
    cluster_min_B=1.5,      # must address the factor-2 cluster gap at all
    asym_slope=(-1.25, -0.75),
)
BZ_95 = CV.BZ_95
H_OBS = 28.65


def sha_files():
    out = {}
    for f in sorted(os.listdir(HERE)):
        if f.endswith(".py"):
            out[f] = hashlib.sha256(
                open(os.path.join(HERE, f), "rb").read()).hexdigest()
    for p in ("../tensor/wellnet.py", "../tensor/mechanism.py",
              "../tensor/cluster.py", "../screen/fieldsolve.py",
              "../screen/screen.py",
              "../../gravitylab/solver.py", "../../gravitylab/axisym.py",
              "../../gravitylab/data.py",
              "../../gravity-cluster-audit-2026-09/adyn/adyn_model.py"):
        q = os.path.normpath(os.path.join(HERE, p))
        if os.path.exists(q):
            out[p] = hashlib.sha256(open(q, "rb").read()).hexdigest()
    return out


def enumerate_candidates():
    cands = []
    for base in BASES:
        cands.append(Candidate(f"BASE_{base}", base=base, inv="one",
                               form="off", struct="scalar_a0"))
    cands.append(Candidate("BASE_newton", base="newton", inv="one",
                           form="off", struct="scalar_a0"))
    for base in BASES:
        for st in STRUCTS:
            for inv, scales in INV_SCALES.items():
                for I0 in scales:
                    for form, m in FORM_M:
                        cands.append(Candidate(
                            f"{base}|{st}|{inv}|{form}|m{m:g}|I{I0:g}",
                            base=base, inv=inv, form=form, m=m, I0=I0,
                            struct=st))
        for ws in WELLS_MAIN:
            for inv, scales in INV_SCALES.items():
                for I0 in scales:
                    for form, m in FORM_M:
                        cands.append(Candidate(
                            f"{base}|tensor_S[{ws['tag']}]|{inv}|{form}"
                            f"|m{m:g}|I{I0:g}",
                            base=base, inv=inv, form=form, m=m, I0=I0,
                            struct="tensor_S", extra=dict(well=ws)))
    if QUICK:
        cands = cands[:3] + cands[3::10]
    return cands


# ================================================================= scoring
def joint(rec):
    """Standardised residual per channel, and the joint score.

    Each channel is expressed in units of its OWN declared tolerance, so the
    joint number is dimensionless and no channel is silently up-weighted by
    having a smaller natural scale.
    """
    z1 = rec["radial_rms_dex"] / TOL["radial_dex"]
    z2 = abs(np.log10(max(rec["vert_Bz"], 1e-12)) - np.log10(CV.BZ_OBS)) \
        / TOL["vert_amp_dex"]
    z3 = abs(rec["vert_h_as"] - H_OBS) / rec["_h_sigma"]
    z4 = rec["cluster_rms_dex"] / TOL["cluster_dex"]
    z = np.array([z1, z2, z3, z4])
    return dict(z_radial=z1, z_vert_amp=z2, z_vert_shape=z3, z_cluster=z4,
                J=float(np.sqrt(np.mean(z ** 2))),
                J_no_amp=float(np.sqrt(np.mean(z[[0, 2, 3]] ** 2))),
                z_worst=float(z.max()))


def score_one(cand, RD, INV, VB, CB, amps=AMPS, n_alt=3):
    """Fit, freeze, and score all four channels for one candidate."""
    rec = dict(name=cand.name, base=cand.base, struct=cand.struct,
               inv=cand.inv, form=cand.form, m=cand.m, I0=cand.I0,
               n_params=n_params(cand))
    if cand.struct == "tensor_S":
        rec["well"] = cand.extra["well"]["tag"]
    # ---- a0 from SPARC train, with the response switched off to start
    cand.A = 0.0
    a0_prev, _ = CR.fit_a0(cand, RD, INV)
    gate_fires, n_iter = False, 0
    for it in range(n_alt):
        n_iter = it + 1
        cl = CC.evaluate(CB, cand, amps, target="lane12")
        # does the response actually do anything in a galaxy?
        Wg = W_cand_train(cand, INV, RD)
        gate_fires = bool(float(np.max(np.abs(cand.A * Wg))) > 1e-3)
        if not gate_fires:
            break
        a0, _ = CR.fit_a0(cand, RD, INV)
        if abs(a0 - a0_prev) / max(a0, 1e-30) < 1e-3:
            break
        a0_prev = a0
    rms, res = CR.score(cand, RD, INV)
    vt = VB.predict(cand)
    asym = SS.asymptotic(cand)
    rec.update(a0=cand.a0, A=cand.A, gate_fires_in_galaxies=gate_fires,
               n_alt_iters=n_iter, at_amp_grid_edge=cl["at_amp_grid_edge"],
               gate_W_max_galaxy=float(np.max(W_cand_train(cand, INV, RD))),
               radial_rms_dex=rms,
               cluster_rms_dex=cl["rms_dex_lane12"],
               cluster_rms_flat_dex=cl["rms_dex_flat"],
               cluster_A_flat=cl["A_flat"],
               cluster_rms_at_A_flat=cl["rms_dex_at_A_flat"],
               cluster_B_flat=cl["B_cluster_flat"],
               field_dex_flat=cl["field_dex_flat"],
               member_dex_flat=cl["member_dex_flat"],
               cluster_B=cl["B_cluster"], cluster_B_arith=cl["B_cluster_arith"],
               cluster_B_1Mpc=cl["B_1Mpc"], cluster_shape=cl["shape"],
               harm_vs_arith_dex=cl["harm_vs_arith_dex"],
               field_dex=cl["field_dex"], member_dex=cl["member_dex"],
               vert_Bz=vt["Bz_law"], vert_h_as=vt["h_median_as"],
               vert_h_chi2dof=vt["h_chi2dof"], vert_amp_chi2dof=vt["amp_chi2dof"],
               vert_A_dyn=vt["A_dyn_2p2"], vert_BR=vt["BR_2p2"],
               asym_slope=asym["slope_total"],
               asym_slope_response=asym["slope_response"],
               W_sup=("inf" if not np.isfinite(asym["W_sup"])
                      else asym["W_sup"]),
               _h_sigma=VB.h_sigma, _resid=res, _h_as=vt["h_as"],
               _B=cl["B_cluster"])
    rec.update(joint(rec))
    rec["screens"] = hard_screens(rec)
    rec["failed"] = [k for k, v in rec["screens"].items() if not v["pass"]]
    rec["survives"] = not rec["failed"]
    return rec


def W_cand_train(cand, INV, RD):
    if cand.form == "off" or cand.inv == "one":
        return np.zeros(1)
    m = RD["is_train"]
    return W_of(cand.form, INV[cand.inv][m] / cand.I0, cand.m)


def hard_screens(rec):
    def S(ok, val, tol, why):
        return dict(**{"pass": bool(ok)}, value=val, tol=tol, why=why)
    return {
        "H1_cluster_reach": S(rec["cluster_B_1Mpc"] >= TOL["cluster_min_B"],
                              rec["cluster_B_1Mpc"], TOL["cluster_min_B"],
                              "B at 1 Mpc must reach the measured cluster gap"),
        "H2_field_galaxy": S(rec["field_dex"] <= TOL["galaxy_dex"],
                             rec["field_dex"], TOL["galaxy_dex"],
                             "isolated galaxy must stay on the RAR"),
        "H3_member_galaxy": S(rec["member_dex"] <= TOL["galaxy_dex"],
                              rec["member_dex"], TOL["galaxy_dex"],
                              "a CLUSTER MEMBER sits deeper than the 1 Mpc "
                              "shell; a depth-gated law fires hardest there"),
        "H4_radial": S(rec["radial_rms_dex"] <= TOL["radial_max_dex"],
                       rec["radial_rms_dex"], TOL["radial_max_dex"],
                       "SPARC train RMS in dex"),
        "H5_vert_amplitude": S(BZ_95[0] <= rec["vert_Bz"] <= BZ_95[1],
                               rec["vert_Bz"], list(BZ_95),
                               "inside Run L's 95% interval on B_z; a "
                               "CONSTRAINT, not a discriminator"),
        "H6_vert_shape": S(rec["vert_h_chi2dof"] <= TOL["vert_shape_chi2"],
                           rec["vert_h_chi2dof"], TOL["vert_shape_chi2"],
                           "h_sigma_LOS chi2/dof; this is the channel that "
                           "rejected the isotropic tensor at 133"),
        "H7_asymptotic": S(TOL["asym_slope"][0] <= rec["asym_slope"]
                           <= TOL["asym_slope"][1], rec["asym_slope"],
                           list(TOL["asym_slope"]),
                           "d ln g/d ln r must be -1, a flat rotation curve"),
    }


# ===================================================================== main
def main():
    print("=" * 78)
    print("JOINT TOURNAMENT -- four channels at once")
    print("=" * 78)
    print("\n[1/5] benches")
    RD = CR.build(verbose=True)
    INV = CR.invariants(RD, "inf")
    VB = CV.VerticalBench()
    VB.h_sigma = float(1.2533 * np.std(VB.OBS_H, ddof=1) / np.sqrt(VB.NG))
    print(f"   DiskMass galaxies          : {VB.NG}")
    print(f"   observed h_sigma_LOS median: {np.median(VB.OBS_H):.2f} arcsec"
          f"   sd {np.std(VB.OBS_H, ddof=1):.2f}"
          f"   sigma(median) {VB.h_sigma:.3f}")
    CB = CC.ClusterBench(n=64)
    print(f"   cluster shell cells        : "
          f"{[int(m.sum()) for m in CB.probes['cluster']['masks']]}")
    print(f"   lane-12 required B(r)      : {np.round(CC.BREQ, 3)}")

    cands = enumerate_candidates()
    print(f"\n[2/5] scoring {len(cands)} candidates")
    ckpt = os.path.join(HERE, "_checkpoint.json")
    done = {}
    if os.path.exists(ckpt) and "--fresh" not in sys.argv:
        done = {r["name"]: r for r in json.load(open(ckpt))}
        print(f"   resuming: {len(done)} candidates already scored")
    recs = []
    t = time.time()
    for i, c in enumerate(cands):
        if c.name in done:
            recs.append(done[c.name])
            continue
        try:
            recs.append(score_one(c, RD, INV, VB, CB))
        except Exception as e:                       # noqa: BLE001
            recs.append(dict(name=c.name, error=repr(e), survives=False,
                             screens={}, J=float("inf")))
        if (i + 1) % 100 == 0:
            print(f"   {i+1}/{len(cands)}  {time.time()-t:.0f}s", flush=True)
            if CC.XP is not np:
                CC.XP.get_default_memory_pool().free_all_blocks()
            with open(ckpt + ".tmp", "w", newline="\n") as fh:
                json.dump([strip(r) for r in recs], fh, default=float)
            os.replace(ckpt + ".tmp", ckpt)
    with open(ckpt, "w", newline="\n") as fh:
        json.dump([strip(r) for r in recs], fh, default=float)
    print(f"   done in {time.time()-t:.0f}s; "
          f"{TC.N_CLIPPED[0]} response evaluations hit the W ceiling")

    ok = [r for r in recs if "error" not in r]
    print(f"\n[3/5] funnel")
    funnel = {"candidates": len(recs), "evaluated": len(ok),
              "errors": len(recs) - len(ok)}
    alive = list(ok)
    order = ("H7_asymptotic", "H1_cluster_reach", "H4_radial",
             "H5_vert_amplitude", "H6_vert_shape", "H2_field_galaxy",
             "H3_member_galaxy")
    #: MARGINAL power: how many candidates each screen kills ON ITS OWN, i.e.
    #: independently of the order.  A sequential funnel credits whichever
    #: screen happens to run first with everything the screens share.
    print(f"   {'screen':<20}{'kills alone':>12}{'unique kills':>14}"
          f"{'sequential':>12}")
    for key in order:
        alone = [r for r in ok if not r["screens"][key]["pass"]]
        uniq = [r for r in alone
                if all(r["screens"][k]["pass"] for k in order if k != key)]
        funnel[key] = dict(kills_alone=len(alone), unique_kills=len(uniq))
        print(f"   {key:<20}{len(alone):>12}{len(uniq):>14}", end="")
        before = len(alive)
        alive = [r for r in alive if r["screens"][key]["pass"]]
        funnel[key]["sequential"] = dict(before=before, after=len(alive),
                                         eliminated=before - len(alive))
        print(f"{before:>7} ->{len(alive):>4}")
    funnel["survivors"] = len(alive)

    # THE POINT OF A JOINT TOURNAMENT: the winner of each channel taken alone,
    # so it is visible that single-channel winners are not the joint winner.
    per_ch = {}
    for key, lab, lo in (("radial_rms_dex", "radial rotation", True),
                         ("z_vert_amp", "vertical amplitude", True),
                         ("z_vert_shape", "vertical radial shape", True),
                         ("cluster_rms_dex", "cluster amplitude+shape", True),
                         ("member_dex", "member-galaxy screen", True)):
        b = min(ok, key=lambda r: r[key])
        per_ch[key] = dict(label=lab, winner=b["name"], value=b[key],
                           J=b["J"], survives=b["survives"],
                           failed=b["failed"])
        print(f"   best on {lab:<24} {b['name'][:44]:<44} "
              f"{b[key]:.4f}  survives={b['survives']}")
    funnel["per_channel_winner"] = per_ch

    print(f"\n[4/5] ranking and the one-standard-error parsimony rule")
    ok.sort(key=lambda r: r["J"])
    ranked = sorted(alive if alive else ok, key=lambda r: r["J"])
    sel = parsimony(ranked, VB, RD)
    sel["ranked_over"] = "survivors" if alive else "ALL (nothing survived)"

    print(f"\n[5/5] full-cost screens on the shortlist")
    short = ranked[:8] + [r for r in ok[:12] if r not in ranked[:8]]
    short = short[:16]
    for r in short:
        c = rebuild(r)
        try:
            r["momentum"] = SS.momentum(c, n=32)
        except Exception as e:                        # noqa: BLE001
            r["momentum"] = dict(error=repr(e))
        r["coarse_grain"] = SS.coarse_grain(c)
        print(f"   {r['name'][:58]:<58} F_net/Fref "
              f"{r['momentum'].get('excess', float('nan')):.3f}")

    extra = extras(RD, INV, VB, CB, ok)
    out = dict(
        generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        lane="work/wellnet-2026-09/tournament", quick=QUICK,
        tolerances=TOL, n_candidates=len(recs), funnel=funnel,
        selection=sel, source_hashes=sha_files(),
        seconds=time.time() - T0,
        records=[strip(r) for r in ok])
    with open(os.path.join(HERE, "tournament.json"), "w", newline="\n") as fh:
        json.dump(out, fh, indent=1, default=float)
    with open(os.path.join(HERE, "gates.json"), "w", newline="\n") as fh:
        json.dump(extra, fh, indent=1, default=float)
    print(f"\nwrote tournament.json ({len(ok)} records) and gates.json "
          f"in {time.time()-T0:.0f}s")
    return out, extra, ok, ranked


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def rebuild(r):
    c = Candidate(r["name"], base=r["base"], inv=r["inv"], form=r["form"],
                  m=r["m"], I0=r["I0"], struct=r["struct"], A=r["A"],
                  a0=r["a0"])
    if r["struct"] == "tensor_S":
        c.extra = dict(well=[w for w in CC.WELL_SETTINGS
                             if w["tag"] == r["well"]][0])
    return c


def parsimony(ranked, VB, RD, nboot=400, seed=20260904):
    """The one-standard-error rule, on a bootstrap over OBJECTS.

    A bare argmin over a joint score mis-identifies nested families, because a
    richer model can only tie or win: the controls lane measured 4 of 5 correct
    with argmin against 5 of 5 with this rule.  The standard error is estimated
    by resampling the SPARC train galaxies, the DiskMass galaxies and the
    cluster shells with replacement -- objects, not points.
    """
    if not ranked:
        return dict(note="no survivor to select from")
    rng = np.random.default_rng(seed)
    best = ranked[0]
    if "_resid" not in best:            # restored from a checkpoint
        c = rebuild(best)
        INV = CR.invariants(RD, "inf")
        best["_resid"] = CR.score(c, RD, INV)[1]
        best["_h_as"] = VB.predict(c)["h_as"]
        best["_B"] = best["cluster_B"]
        best["_h_sigma"] = VB.h_sigma
    gal = RD["gal"][RD["is_train"]]
    ug = np.unique(gal)
    Js = []
    for _ in range(nboot):
        gsel = rng.choice(ug, size=len(ug), replace=True)
        mask = np.concatenate([np.where(gal == g)[0] for g in gsel])
        z1 = np.sqrt(np.mean(best["_resid"][mask] ** 2)) / TOL["radial_dex"]
        dsel = rng.integers(0, VB.NG, VB.NG)
        hmed = float(np.median(np.asarray(best["_h_as"])[dsel]))
        z3 = abs(hmed - H_OBS) / best["_h_sigma"]
        ssel = rng.integers(0, len(best["_B"]), len(best["_B"]))
        z4 = float(np.sqrt(np.mean(
            (np.log10(np.maximum(np.asarray(best["_B"])[ssel], 1e-12))
             - np.log10(CC.BREQ[ssel])) ** 2))) / TOL["cluster_dex"]
        z2 = best["z_vert_amp"]
        Js.append(np.sqrt(np.mean(np.array([z1, z2, z3, z4]) ** 2)))
    se = float(np.std(Js, ddof=1))
    within = [r for r in ranked if r["J"] <= best["J"] + se]
    kmin = min(r["n_params"] for r in within)
    simplest = [r for r in within if r["n_params"] == kmin]
    simplest.sort(key=lambda r: r["J"])
    print(f"   best J = {best['J']:.3f} ({best['name'][:50]})")
    print(f"   bootstrap SE(J) = {se:.3f} over {nboot} object resamples")
    print(f"   within 1 SE: {len(within)}; fewest parameters there: k = {kmin}")
    print(f"   PARSIMONY PICK: {simplest[0]['name'][:60]}  J = "
          f"{simplest[0]['J']:.3f}  k = {kmin}")
    return dict(best_argmin=best["name"], best_J=best["J"], se_J=se,
                n_within_1se=len(within), k_min=kmin,
                parsimony_pick=simplest[0]["name"],
                parsimony_pick_J=simplest[0]["J"],
                within_1se=[r["name"] for r in within][:60])


def extras(RD, INV, VB, CB, ok):
    """gates.json: reproductions, sensitivities and the responsiveness checks."""
    g = {}
    # --- reproduction of the lanes this one reuses
    rep = {}
    for nm, base, a0 in (("newton", "newton", 1e-10), ("rar", "rar", 1.084e-10),
                         ("aqual", "aqual", 1.058e-10)):
        c = Candidate(nm, base=base, a0=a0)
        p = VB.predict(c)
        rms, _ = CR.score(c, RD, INV)
        rep[nm] = dict(sparc_train_rms_dex=rms, h_as=p["h_median_as"],
                       h_chi2dof=p["h_chi2dof"], Bz=p["Bz_law"],
                       Bz_1hR=p["Bz_law_1hR"], BR_2p2=p["BR_2p2"],
                       A_dyn=p["A_dyn_2p2"])
    g["reproduction_of_prior_lanes"] = dict(
        measured=rep,
        run_L_published=dict(
            h_as=dict(newton=30.80, rar=35.20, aqual=34.96),
            h_chi2dof=dict(newton=10.5, rar=20.2, aqual=20.0),
            Bz_pointwise_1hR=dict(rar=1.565), Bz_pointwise_2p2hR=dict(rar=1.843),
            BR_2p2=dict(rar=1.915, aqual=1.929)),
        adyn_published_sparc=dict(rar=0.1641, aqual=0.1647, newton=0.5215),
        note="the SPARC RMS and the vertical h reproduce Run L exactly; the "
             "AQUAL h differs by 0.05 arcsec because this lane uses the "
             "algebraic vertical reduction for both bases where Run L "
             "bisects the exact AQUAL equation")
    # --- |Phi_N| boundary rule sensitivity
    rule = {}
    for r in CR.PHI_RULES:
        I2 = CR.invariants(RD, r)
        c = Candidate("s", base="aqual", a0=1.058e-10, inv="phi", form="sat",
                      m=4.0, I0=1e12, struct="scalar_a0", A=30.0)
        rms, _ = CR.score(c, RD, I2)
        rule[r] = dict(sparc_rms_dex=rms,
                       median_phi=float(np.median(I2["phi"][RD["is_train"]])),
                       p95_phi=float(np.percentile(I2["phi"][RD["is_train"]],
                                                   95)))
    g["phi_boundary_rule"] = dict(
        rules=rule, primary="inf",
        note="'inf' and 'flat' are GLOBAL prescriptions and are admissible; "
             "'last' and 'half' reference an object's own radius and "
             "therefore violate the global-parameter rule -- carried only to "
             "show how far the variable moves")
    # --- responsiveness, dS/dtheta != 0
    resp = {}
    for key in ("cluster_rms_dex", "member_dex", "vert_h_as", "vert_Bz",
                "radial_rms_dex", "asym_slope"):
        resp[key] = SS.responsive([r[key] for r in ok if key in r], key)
    g["responsiveness"] = resp
    # --- realisation scatter of the member screen
    g["member_realisation_scatter"] = member_scatter(ok)
    # --- averaging bracket
    hv = [r["harm_vs_arith_dex"] for r in ok if "harm_vs_arith_dex" in r]
    g["shell_average_bracket_dex"] = dict(
        median=float(np.median(hv)), p95=float(np.percentile(hv, 95)),
        max=float(np.max(hv)),
        note="|log10 B_harmonic - log10 B_arithmetic| at each candidate's "
             "fitted amplitude; the harmonic mean is the one calibrated "
             "against six full 3-D solves (worst 20.4% vs 46.9%)")
    g["W_ceiling_hits"] = TC.N_CLIPPED[0]
    return g


def member_scatter(ok, seeds=(20260903, 11, 23, 37, 51)):
    """Re-draw the 300 cluster members and re-measure the member violation."""
    top = [r for r in ok if r.get("survives")][:3]
    if not top:
        top = sorted([r for r in ok if "member_dex" in r],
                     key=lambda r: r["member_dex"])[:3]
    out = []
    for r in top:
        c = rebuild(r)
        vals = []
        for sd in seeds:
            B = CC.ClusterBench(n=64, seed=sd)
            e = CC.evaluate(B, c, AMPS, target="lane12")
            vals.append(e["member_dex"])
            del B
        out.append(dict(name=r["name"], seeds=list(seeds),
                        member_dex=[float(v) for v in vals],
                        mean=float(np.mean(vals)), sd=float(np.std(vals, ddof=1)),
                        tol=TOL["galaxy_dex"]))
    return out


if __name__ == "__main__":
    main()
