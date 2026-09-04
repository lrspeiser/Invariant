"""JOB 0 -- the Stage 4 sensitivity certificate, run BEFORE the measurement exists.

Candidate  : the GEOMETRIC half of the path-redshift class (Run AK): an
             achromatic path redshift d ln(1+z) = c2 I_q, which supernova time
             dilation cannot touch because it predicts b = 1 identically.
Statistic  : beta = dT/d(dI_q) on the Planck temperature map over the SDSS
             VoidFinder footprint, reported as c2/c1 = beta * (-1.09989e-3).
Prediction : Run AK's own CMB-smoothness gate, |c2|/c1 < 0.28-0.44%.  C4 is
             powered AT THAT amplitude, not at a convenient one.

THE BLIND GUARD IS ARMED THROUGHOUT.  Every temperature this file reads comes
from a sky placement whose overlap with the true footprint is <= 5%; the
identity placement raises PermissionError.  So the certificate can size its own
test against the real sky -- foregrounds, noise, mask and all -- and still
cannot see the answer.  The guard is disarmed only in measure.py, once.

    python certify_voidcmb.py
"""
from __future__ import annotations

import io
import json
import os
import sys
import time

import numpy as np
import astropy.units as u
from astropy_healpix import HEALPix

HERE = os.path.dirname(os.path.abspath(__file__))
LANE = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(LANE, "stage4"))

import certificate as CERT                                    # noqa: E402
import estimator as E                                         # noqa: E402
import planckio as P                                          # noqa: E402

RNG = np.random.default_rng(20260904)
PATHMAP = "pathmap_ns64_er5.npz"
CL_FILE = os.path.join(
    HERE, "raw",
    "COM_PowerSpect_CMB-base-plikHM-TTTEEE-lowl-lowE-lensing-minimum-theory_R3.01.txt")

# the amplitude the theory predicts, from Run AK 1.5 -- NOT a convenient one
AK_BOUND_LO, AK_BOUND_HI = 0.0028, 0.0044
BETA_PER_C2C1 = 1.0 / E.UK_PER_MPCH_TO_C2C1          # -909.13 uK/(Mpc/h) per unit


# --------------------------------------------------------------------- context
class Context:
    def __init__(self, pathmap=PATHMAP, nside=64, map_key="smica_nosz_pr3"):
        d = np.load(os.path.join(HERE, pathmap))
        self.hp = HEALPix(nside=nside, order="nested")
        self.nside = nside
        _, self.frac = P.load_mask(nside)
        keep = self.frac[d["pix"]] >= 0.9
        self.pix = d["pix"][keep]
        self.n = len(self.pix)
        self.G = E.galactic_vectors(d["l_gal"][keep], d["b_gal"][keep])
        # MAP-attached templates: properties of the void catalogue / footprint
        self.map_t = {k: d[k][keep] for k in
                      ("I_q", "I_phi_in", "I_phi_k3", "I_R", "I_sat", "N_v",
                       "I_q_near", "I_q_far", "edge_deg", "I_q_nonedge")}
        self.map_t["dI_q"] = self.map_t["I_q"] - self.map_t["I_q"].mean()
        # SKY-attached templates: properties of the sky, so they follow rotations
        self.sky_full = {}
        for k in ("dust", "sync", "freefree", "ame", "co"):
            v, _ = P.load_foreground(k, nside)
            pos = v[np.isfinite(v) & (v > 0)]
            floor = np.percentile(pos, 1) if pos.size else 1e-6
            self.sky_full[k] = np.log10(np.maximum(np.nan_to_num(v, nan=floor), floor))
        self.T_full = P.load_map(map_key, nside)["T"]
        self.map_key = map_key
        self.guard = P.BlindGuard(self.pix, max_overlap=0.05)
        self.n_footprint_raw = int(len(d["pix"]))

    # ---- a placement is the identity or a rotation matrix
    def placement(self, R):
        if R is None:
            pix, G = self.pix, self.G
        else:
            Gr = self.G @ R.T
            lr, br = E.vectors_to_lb(Gr)
            pix = np.asarray(self.hp.lonlat_to_healpix(lr * u.deg, br * u.deg))
            G = Gr
        return pix, G

    def sky_cols(self, pix, G):
        b = np.degrees(np.arcsin(np.clip(G[:, 2], -1, 1)))
        out = {"csc_b": 1.0 / np.maximum(np.abs(np.sin(np.radians(b))), 1e-3)}
        for k, v in self.sky_full.items():
            out[k] = v[pix]
        return out

    def evaluate(self, R, model, inject=None, T_override=None, use_pixels=None,
                 what="null"):
        """beta and c2/c1 for one placement. inject: (template_name, amplitude uK/(Mpc/h))."""
        pix, G = self.placement(R)
        self.guard.check(pix, what)
        ok = self.frac[pix] >= 0.9
        pix_u, G_u = pix[ok], G[ok]
        T = (self.T_full[pix_u] if T_override is None else T_override[ok])
        cols = {}
        cols.update({k: v[ok] for k, v in self.map_t.items()})
        cols.update(self.sky_cols(pix_u, G_u))
        if inject is not None:
            name, amp = inject
            t = cols[name]
            T = T + amp * (t - t.mean())
        X = E.design(cols, E.MODELS[model], G_u)
        beta, coef, sd = E.fit_beta(X, T)
        return dict(beta=beta, c2c1=beta * E.UK_PER_MPCH_TO_C2C1, sd_ols=sd,
                    coef=coef, n=len(pix_u), unmasked_frac=float(ok.mean()))


def rotation_bank(ctx, n_want=2000, min_unmasked=0.95, seed=7):
    """Random rotations AND reflections of the footprint; strict alternation so
    exactly half the bank has det = -1.  (The development version called
    random_rotations(rng, 1) in a loop, whose internal index was always 0, so it
    produced no reflections at all -- fixed here, and the two halves are reported
    separately as a parity null.)"""
    rng = np.random.default_rng(seed)
    tset = set(ctx.pix.tolist())
    bank, dets, tried = [], [], 0
    while len(bank) < n_want and tried < 200 * n_want:
        R = E.random_rotation(rng, reflect=(len(bank) % 2 == 1))
        tried += 1
        pix, G = ctx.placement(R)
        f = np.mean(ctx.frac[pix] >= 0.9)
        if f < min_unmasked:
            continue
        if len(tset.intersection(pix.tolist())) / len(pix) > 0.05:
            continue
        bank.append(R)
        dets.append(float(np.linalg.det(R)))
    return bank, tried, np.array(dets)


# ------------------------------------------------------------- the seven checks
def main():
    t0 = time.time()
    print("building context (no temperature at the true footprint is read) ...", flush=True)
    ctx = Context()
    print(f"  analysis pixels {ctx.n} of {ctx.n_footprint_raw} footprint pixels "
          f"(Planck common mask >= 0.9)", flush=True)

    print("building the rotation bank ...", flush=True)
    bank, tried, dets = rotation_bank(ctx, 2000)
    print(f"  {len(bank)} admissible rotations from {tried} draws "
          f"({time.time()-t0:.0f}s)", flush=True)

    # -------- null distribution of the statistic, per model
    null = {}
    for m in E.MODELS:
        ev = [ctx.evaluate(R, m) for R in bank]
        vals = np.array([e["c2c1"] for e in ev])
        ns = np.array([e["n"] for e in ev])
        null[m] = vals
        if m == E.HEADLINE:
            n_used_null = ns
        print(f"  null {m:22s} mean {vals.mean():+.5f} sd {vals.std(ddof=1):.5f} "
              f"(n_used {ns.mean():.0f} vs true {ctx.n})", flush=True)

    hm = E.HEADLINE
    sd_null = float(null[hm].std(ddof=1))
    mean_null = float(null[hm].mean())

    # -------- C1 responsive: does the statistic move when c2 moves?
    amps = np.array([-0.006, -0.003, 0.0, 0.003, 0.006])
    R0 = bank[0]
    def stat_at(c2c1):
        return ctx.evaluate(R0, hm, inject=("dI_q", BETA_PER_C2C1 * c2c1),
                            what="C1")["c2c1"]
    c1_res = CERT.c1_responsive(stat_at, amps, tol=1e-4)
    resp_in_grammar = c1_res["slope"]
    # responsiveness averaged over placements, which is the honest number
    resp_bank = []
    for R in bank[:60]:
        a = ctx.evaluate(R, hm, inject=("dI_q", BETA_PER_C2C1 * 0.004), what="C1")["c2c1"]
        b = ctx.evaluate(R, hm, inject=("dI_q", -BETA_PER_C2C1 * 0.004), what="C1")["c2c1"]
        resp_bank.append((a - b) / 0.008)
    resp_bank = np.array(resp_bank)
    c1_res["responsiveness_mean"] = float(resp_bank.mean())
    c1_res["responsiveness_sd"] = float(resp_bank.std(ddof=1))
    # the physical responsiveness: the template is the pixel-CENTRE value while
    # the real sky delivers the pixel AVERAGE.  Measured against the nside-128 map.
    a64 = np.load(os.path.join(HERE, PATHMAP))
    b128 = np.load(os.path.join(HERE, "pathmap_ns128_er5.npz"))
    par = b128["pix"] >> 2
    cnt = np.bincount(par, minlength=int(a64["pix"].max()) + 1)
    sm = np.bincount(par, weights=b128["I_q"], minlength=int(a64["pix"].max()) + 1)
    full = cnt[a64["pix"]] == 4
    xc = a64["I_q"][full] - a64["I_q"][full].mean()
    ya = sm[a64["pix"]][full] / 4.0
    ya = ya - ya.mean()
    resp_pix = float(np.dot(xc, ya) / np.dot(xc, xc))
    c1_res["responsiveness_pixelisation"] = resp_pix
    c1_res["responsiveness_total"] = float(resp_bank.mean() * resp_pix)
    c1_res["detail"] += (f"; d(estimate)/d(injected) = {resp_bank.mean():.4f} "
                         f"+- {resp_bank.std(ddof=1):.4f} over 60 placements, "
                         f"x {resp_pix:.4f} for pixelisation = {resp_bank.mean()*resp_pix:.4f}")

    # -------- C2: is it a restatement of a fitted normalisation?
    # control levers, at amplitudes far beyond their real uncertainties
    def lever_monopole(dT_uK):
        T = ctx.T_full.copy()
        pix, _ = ctx.placement(R0)
        return ctx.evaluate(R0, hm, T_override=(ctx.T_full[pix] + dT_uK), what="C2")["c2c1"]

    def lever_gain(g):
        pix, _ = ctx.placement(R0)
        return ctx.evaluate(R0, hm, T_override=ctx.T_full[pix] * (1.0 + g), what="C2")["c2c1"]

    mono = np.array([lever_monopole(v) for v in (-200.0, -100.0, 0.0, 100.0, 200.0)])
    gain = np.array([lever_gain(v) for v in (-1e-3, -5e-4, 0.0, 5e-4, 1e-3)])
    resp_mono = float(np.polyfit([-200, -100, 0, 100, 200], mono, 1)[0]) * 100.0  # per 100 uK
    resp_gain = float(np.polyfit([-1e-3, -5e-4, 0, 5e-4, 1e-3], gain, 1)[0]) * 1e-3
    ctrl = max(abs(resp_mono), abs(resp_gain))
    tgt = abs(resp_in_grammar) * 0.004
    c2_res = dict(passed=ctrl < tgt,
                  target_response_at_predicted=float(tgt),
                  monopole_lever_100uK=resp_mono, gain_lever_0p1pct=resp_gain,
                  ratio=float(ctrl / tgt) if tgt > 0 else None,
                  detail=(f"a 100 uK monopole moves c2/c1 by {resp_mono:+.2e} and a "
                          f"0.1% gain error by {resp_gain:+.2e}; the predicted signal "
                          f"moves it by {tgt:.2e} -- ratio {ctrl/tgt:.2e}"))

    # -------- C3: exchangeability, and the REALISED false-positive rate
    # (a) leave-one-out over rotations: score each rotation against the others
    v = null[hm]
    p_loo = np.array([(np.sum(np.abs(np.delete(v, i)) >= abs(v[i])) + 1) /
                      (len(v)) for i in range(len(v))])
    fpr_loo = float(np.mean(p_loo <= 0.05))
    # (b) an INDEPENDENT null: Gaussian CMB realisations on the true footprint
    #     directions, calibrated to the real map's variance in rotated patches
    print("  simulating Gaussian CMB nulls ...", flush=True)
    cl = E.cl_theory(CL_FILE, lmax=1200)
    var_real = np.mean([np.var(ctx.T_full[ctx.placement(R)[0]]) for R in bank[:200]])
    lo, hi = 0.2, 3.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        bl = E.beam_pixwin(1200, mid)
        c0 = float(np.sum((2 * np.arange(1201) + 1) / (4 * np.pi) * cl * bl ** 2))
        if c0 > var_real:
            lo = mid
        else:
            hi = mid
    fwhm_eff = 0.5 * (lo + hi)
    bl = E.beam_pixwin(1200, fwhm_eff)
    C = E.pixel_cov(ctx.G, cl, bl)
    C[np.diag_indices_from(C)] += 1e-6 * np.trace(C) / len(C)
    L = np.linalg.cholesky(C)
    del C
    n_sim = 600
    cols = {k: v2 for k, v2 in ctx.map_t.items()}
    cols.update(ctx.sky_cols(ctx.pix, ctx.G))
    X_true = E.design(cols, E.MODELS[hm], ctx.G)
    w_true = E.weights_for_beta(X_true)
    sims = L @ RNG.normal(size=(len(L), n_sim))
    sim_beta = (w_true @ sims) * E.UK_PER_MPCH_TO_C2C1
    sd_sim = float(sim_beta.std(ddof=1))
    # (c) the analytic OLS error, on a rotated placement
    sd_ols = float(np.mean([ctx.evaluate(R, hm)["sd_ols"] for R in bank[:150]])
                   * abs(E.UK_PER_MPCH_TO_C2C1))
    # (d) FPR of the independent simulations scored against the ROTATION null
    p_sim = np.array([(np.sum(np.abs(v) >= abs(s)) + 1) / (len(v) + 1) for s in sim_beta])
    fpr_sim = float(np.mean(p_sim <= 0.05))
    bias_z = mean_null / (sd_null / np.sqrt(len(v)))
    refl = dets < 0
    c3_res = dict(
        null_sd_proper=float(null[hm][~refl].std(ddof=1)),
        null_sd_reflected=float(null[hm][refl].std(ddof=1)),
        null_mean_proper=float(null[hm][~refl].mean()),
        null_mean_reflected=float(null[hm][refl].mean()),
        passed=(abs(fpr_loo - 0.05) < 0.05 and abs(fpr_sim - 0.05) < 0.05
                and abs(mean_null) < 0.5 * sd_null and 0.5 < sd_sim / sd_null < 2.0),
        realised_fpr_rotation_loo=fpr_loo, realised_fpr_simulation=fpr_sim,
        null_mean=mean_null, null_sd=sd_null, sd_simulation=sd_sim,
        sd_analytic_ols=sd_ols, sd_sim_over_sd_null=float(sd_sim / sd_null),
        sd_ols_over_sd_null=float(sd_ols / sd_null), n_rotations=len(v),
        null_mean_z=float(bias_z), fwhm_eff_deg=float(fwhm_eff),
        detail=(f"rotation null {mean_null:+.5f} +- {sd_null:.5f}; realised FPR "
                f"{fpr_loo:.3f} (leave-one-out) and {fpr_sim:.3f} (independent "
                f"Gaussian skies) against nominal 0.05; sd(sim)/sd(null) = "
                f"{sd_sim/sd_null:.2f}, sd(analytic OLS)/sd(null) = {sd_ols/sd_null:.2f}"))

    # -------- C4: power AT THE PREDICTED amplitude (AK's own bound)
    c4_lo = CERT.c4_powered(abs(resp_bank.mean()), AK_BOUND_LO, sd_null)
    c4_hi = CERT.c4_powered(abs(resp_bank.mean()), AK_BOUND_HI, sd_null)
    c4_res = dict(passed=bool(c4_lo["passed"]), z_at_0p28pct=c4_lo["z_at_predicted"],
                  z_at_0p44pct=c4_hi["z_at_predicted"],
                  responsiveness=float(resp_bank.mean()), sd_null=sd_null,
                  min_detectable_3sigma=float(3 * sd_null / abs(resp_bank.mean())),
                  detail=(f"AK's gate predicts |c2/c1| < {AK_BOUND_LO:.2%}-{AK_BOUND_HI:.2%}; "
                          f"at {AK_BOUND_LO:.2%} this pipeline gives "
                          f"{c4_lo['z_at_predicted']:.1f} sigma, at {AK_BOUND_HI:.2%} "
                          f"{c4_hi['z_at_predicted']:.1f} sigma; 3-sigma floor "
                          f"{3*sd_null/abs(resp_bank.mean()):.2%}"))

    # -------- C5: support
    meta = json.loads(io.open(os.path.join(HERE, PATHMAP.replace(".npz", ".json")),
                              encoding="utf-8").read())
    c5_chi = CERT.c5_support((0.0, meta["chi_max_mpch"]), (0.0, meta["chi_max_mpch"]))
    c5_z = CERT.c5_support((0.0, meta["z_max"]), (0.0, 0.1125))
    sky_ok = ctx.n / ctx.n_footprint_raw
    c5_res = dict(passed=bool(c5_chi["passed"] and c5_z["passed"] and sky_ok > 0.9),
                  chi=c5_chi, z=c5_z, sky_fraction_of_footprint_used=float(sky_ok),
                  detail=(f"reads chi in [0, {meta['chi_max_mpch']:.1f}] Mpc/h and z <= "
                          f"{meta['z_max']}, exactly the catalogue's own limits; "
                          f"{sky_ok:.1%} of the eroded footprint survives the Planck mask"))

    # -------- C6: an injection from OUTSIDE the inference grammar
    # grammar = 'temperature is linear in the union-of-spheres path length'.
    # Each g is standardised to dI_q's spread, so an exactly matching injection
    # would return 1.00 and the shortfall is honest template mismatch.
    c6 = {}
    for gname in ("I_R", "I_sat", "N_v", "I_phi_in", "I_q_far"):
        g = ctx.map_t[gname]
        scale = ctx.map_t["dI_q"].std() / g.std()
        rec = []
        for R in bank[:80]:
            base = ctx.evaluate(R, hm, what="C6")["c2c1"]
            inj = ctx.evaluate(R, hm, inject=(gname, BETA_PER_C2C1 * 0.004 * scale),
                               what="C6")["c2c1"]
            rec.append((inj - base) / 0.004)
        c6[gname] = float(np.mean(rec))
    worst = min(abs(x) for x in c6.values())
    best_oog = max(abs(v2) for k, v2 in c6.items() if k in ("I_R", "I_sat", "N_v"))
    c6_res = dict(passed=bool(best_oog >= 0.5), recovery=c6, best_out_of_grammar=best_oog,
                  detail=(f"out-of-grammar injections recover " +
                          ", ".join(f"{k} {v2:.2f}" for k, v2 in c6.items()) +
                          f"; best {best_oog:.2f}"))

    # -------- C7: can a common systematic reproduce the signature?
    sky = ctx.sky_cols(ctx.pix, ctx.G)
    nuis = dict(sky)
    nuis["edge_deg"] = ctx.map_t["edge_deg"]
    nuis["ISW_phi_in"] = ctx.map_t["I_phi_in"]
    nuis["ISW_phi_k3"] = ctx.map_t["I_phi_k3"]
    nuis["mask_fraction"] = ctx.frac[ctx.pix]
    nuis["dipole_x"] = ctx.G[:, 0]
    nuis["dipole_y"] = ctx.G[:, 1]
    nuis["dipole_z"] = ctx.G[:, 2]
    c7_res = CERT.c7_nuisance_distinct(ctx.map_t["dI_q"], nuis, tol=0.9)
    c7_res["all_correlations"] = {
        k: float(np.corrcoef(ctx.map_t["dI_q"], np.asarray(v2, float))[0, 1])
        for k, v2 in nuis.items()}

    results = {"C1_responsive": c1_res, "C2_not_a_restatement": c2_res,
               "C3_exchangeable": c3_res, "C4_powered": c4_res,
               "C5_support": c5_res, "C6_out_of_grammar": c6_res,
               "C7_nuisance_distinct": c7_res}
    issued = CERT.certify(
        "voidcmb: geometric path redshift x Planck TT  [candidate/statistic pair]",
        results)

    doc = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               run_id="BI-voidcmb", lane="work/wellnet-2026-09/voidcmb",
               stage="4 (identifiability)", statistic="beta = dT/d(dI_q)",
               headline_model=hm, models={k: v2 for k, v2 in E.MODELS.items()},
               pathmap=PATHMAP, map_key=ctx.map_key,
               n_analysis_pixels=int(ctx.n), n_rotations=len(bank),
               blind_guard=dict(armed=ctx.guard.armed, checks=ctx.guard.n_checked,
                                refusals=ctx.guard.n_refused,
                                max_overlap=ctx.guard.max_overlap),
               ak_bound=[AK_BOUND_LO, AK_BOUND_HI],
               null_by_model={k: dict(mean=float(v2.mean()), sd=float(v2.std(ddof=1)))
                              for k, v2 in null.items()},
               checks=results, issued=bool(issued),
               opened_true_footprint_temperature=False,
               seconds=time.time() - t0)
    p = os.path.join(HERE, "certificate_voidcmb.json")
    io.open(p, "w", encoding="utf-8", newline="\n").write(json.dumps(doc, indent=1, default=float))
    np.savez_compressed(os.path.join(HERE, "null_bank.npz"),
                        **{k: v2 for k, v2 in null.items()}, sim=sim_beta)
    print(f"\nwrote {p}   ({time.time()-t0:.0f}s)")
    return 0 if issued else 2


if __name__ == "__main__":
    raise SystemExit(main())
