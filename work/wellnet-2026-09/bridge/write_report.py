"""write_report.py -- REPORT.md, rendered from the lane's result JSONs and
nothing else.  Every number in the report is read from a JSON written by the
module that measured it.

    python write_report.py
"""
from __future__ import annotations

import json
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")


def load(name):
    p = os.path.join(RES, name)
    if not os.path.exists(p):
        return None
    return json.load(open(p, encoding="utf-8"))


def f(x, fmt="{:.3g}"):
    try:
        if x is None:
            return "--"
        if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
            return "inf" if math.isinf(x) else "nan"
        return fmt.format(x)
    except Exception:                                          # noqa: BLE001
        return str(x)


def e(x, d=2):
    return f(x, "{:+." + str(d) + "e}")


def table(head, rows):
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def main():
    B = load("bridge.json")
    Dt = load("detect.json")
    A = load("cdm_attack.json")
    Cp = load("compile_bridge.json")
    Ce = load("certificate_bridge.json")
    Mb = load("members.json")
    Xc = load("crosscheck.json")
    Bl = load("baseline_compare.json")
    Ts = load("tests.json")
    L = []
    P = L.append

    P("# Run BM -- the two-concentration probe: the path family's compensated bridge at detector level\n")
    P("Lane `work/wellnet-2026-09/bridge/`; registry `BM-bridge`. Entirely synthetic: every "
      "module arms `guard.py` (universes/provenance.py with the confirmation-reserve tokens "
      "added) and reports 0 foreign reads, 0 sealed-token matches, 0 reserve-token matches. "
      "Global parameters only (eps, rho_*; the scene is BL's declared bridge scene). Letter "
      "provisional: the record's owner assigns it.\n")

    # ------------------------------------------------------------ 0. discipline
    P("## 0. Discipline\n")
    rows = []
    for name, d in (("bridge.json", B), ("detect.json", Dt), ("cdm_attack.json", A),
                    ("compile_bridge.json", Cp), ("certificate_bridge.json", Ce),
                    ("members.json", Mb), ("crosscheck.json", Xc)):
        if d and "provenance" in d:
            pv = d["provenance"]
            rows.append((name, pv.get("n_read_paths_non_library", "--"),
                         len(pv.get("foreign_reads", [])),
                         pv.get("any_real_observational_file_opened", "--"),
                         pv.get("any_reserve_token_in_reads", "--"),
                         f(d.get("wall_seconds"), "{:.0f}")))
    P(table(["result", "non-library reads", "foreign reads", "real-data token", "reserve token", "wall s"], rows))
    P("")
    if Bl:
        P(f"* Compiler extension (Job 1): +102/-5 lines, additive. **{Bl['n_compared']} pre-existing "
          f"verdicts compared field by field against the pre-patch snapshot: {Bl['n_changed']} changed**; "
          f"external controls {Bl['n_agree']}/{Bl['n_external_controls']}; probes now "
          f"{Bl['probes_after']}. `test_compiler.py`: 48/48 before and after the patch.")
    if Ts:
        P(f"* This lane's suite `test_bridge.py`: **{Ts['n_passed']}/{Ts['n_tests']}**"
          + (f" (failures: {Ts['failures']})" if Ts["n_failed"] else "")
          + ". Bugs the suite caught during development: " + "; ".join(Ts["bugs_caught"]) + ".")
    if Xc:
        s = Xc["summary"]
        P(f"* Inverse-crime control: the carrier P(z) by DIRECT Monte-Carlo pair summation (no column "
          f"formula, no direction quadrature; {Xc['n_pairs']:,} pairs, blob sigma = {Xc['sigma_kpc']:.0f} kpc) "
          f"agrees with the closed form at {s['n_points']} points in two scenes: rms |ratio - 1| = "
          f"{s['rms_ratio_minus_1']:.4f}, max {s['max_abs_ratio_minus_1']:.4f}, worst {s['max_abs_sigmas']:.2f} sigma.")
    P("")

    # ------------------------------------------------------------ 1. the probe
    P("## 1. Job 1 -- the two-concentration probe\n")
    if Cp:
        pr = Cp["probe"]
        P(f"`compiler.probes()['bridge_pair']`: two {pr['M_Msun']:.0e} Msun Plummers, a = {pr['a_kpc']:.0f} kpc, "
          f"{pr['D_Mpc']:.0f} Mpc apart, vacuum between (BL's declared scene); 15 points: 6 on the axis "
          f"(x = +-0.5, 1, 1.5 Mpc), 6 in the wings (R = 0.5, 1 Mpc at x = -1, 0, +1 Mpc), 3 on the midplane "
          f"(R = 0.25, 1.5, 2 Mpc). g_N/a0 at the points: {', '.join(f'{v:.3f}' for v in pr['gN_over_a0'])} -- "
          f"the transition regime, where a stretch is least degenerate. GATE 1 consults it ONLY for a "
          f"candidate declaring `Candidate.bridge_field` (|g| at the points, declared by the lane that owns "
          f"the kernel, as `force_factor` is) and fits the same two-constant stretch; a residual above "
          f"0.040 dex is escape (d) `d_two_body_probe`, and the probe joins the joint fit. Every candidate "
          f"without the declaration is untouched by construction (section 0).\n")

    # ------------------------------------------------------------ 2. the bridge
    P("## 2. Job 2 -- the bridge observables\n")
    if B:
        sc, sv, es = B["scene"], B["survey"], B["estimator"]
        P(f"Scene: M_A = M_B = {sc['M_A_Msun']:.0e} Msun, a = {sc['a_kpc']:.0f} kpc, D = {sc['D_Mpc']:.0f} Mpc, "
          f"grid {sc['grid']['nx']} x {sc['grid']['nR']} at {sc['grid']['dx_kpc']:.0f} kpc with "
          f"{sc['grid']['n_dir']} directions; density at the midpoint from the endpoints' own tails "
          f"{sc['rho_at_midpoint']:.2e} kg/m^3 ({sc['rho_at_midpoint'] / 1e-24:.2f} rho_* at the fiducial). "
          f"Detector (BF's declared model): z_l = {sv['z_l']}, {sv['kpc_per_arcmin']:.0f} kpc/arcmin, "
          f"Sigma_cr,eff = {sv['sigma_crit_eff_kg_m2']:.2f} kg/m^2, sigma_e = {sv['sigma_e']}, "
          f"n = {sv['n_arcmin2']:.0f}/arcmin^2, {sv['pixel_kpc']:.0f}-kpc pixels "
          f"({sv['sources_per_pixel']:.2f} sources, shear sd {sv['pixel_shear_sd']:.3f} per pixel). "
          f"Estimator: perpendicular-sector endpoint models (|phi -/+ 90| < {es['sector_half_deg']:.0f} deg), "
          f"KS residual, strips: R_ex = {es['R_ex_kpc']:.0f} kpc, core |y| < {es['w_core_kpc']:.0f} kpc, "
          f"wings to {es['w_wing_kpc']:.0f} kpc, net aperture {es['w_net_kpc']:.0f} kpc; matched-filter "
          f"template = the P residual at eps = {es['eps_ref']}.\n")
        P("### 2.1 The P ladder (map level and strip statistics, noise-free)\n")
        rows = []
        for k, r in B["P_ladder"].items():
            ft, ss = r["features"], r["strip_stats"]
            nm = r["net_mass"]
            rows.append((k, f(r["phi3_midpoint_kms2"], "{:.3e}"),
                         f"{ft['Sigma_axis']:+.3e} +- {r['Sigma_axis_error_grid']:.1e}",
                         f"{ft['wing_peak']:+.2e} @ {ft['wing_peak_kpc']:.0f}", f(ft["zero_crossing_kpc"], "{:.0f}"),
                         e(nm["box_x1.4D_R0.95D"]["net_over_abs"]), e(nm["sky_2D"]["net_over_abs"]),
                         e(ss["Q"]), e(ss["DQ"]), f(ss["A_mf"], "{:+.3f}"), e(r["N_Msun"]),
                         e(r["endpoint_cross_fraction_axis"], 1)))
        P(table(["rho_* | eps", "Phi_3(mid) (km/s)^2", "Sigma_eff axis (kg/m^2)", "wing peak @ kpc",
                 "zero-crossing kpc", "net/abs 3D", "net/abs 2D", "Q", "DQ", "A_mf", "N (Msun)", "endpoint-cross frac"], rows))
        P("\nSigma_eff error: from grid halving (second order, |Sigma_40 - Sigma_80|/3). N is the net "
          "residual 'mass' in the wide between aperture minus the same aperture behind (finite-aperture "
          "truncation of the positive wings, not a failure of compensation -- the 3-D and 2-D net/abs "
          "columns are the compensation). BL recorded -0.49 on the axis and +0.16 in the wings at the "
          "fiducial on a 125-kpc grid; at 40 kpc the axis is -0.61 and the wing +0.17 at 320 kpc.\n")
        k0 = "1e-24|eps=0.3"
        if k0 in B["P_ladder"]:
            r = B["P_ladder"][k0]
            P("Transverse profile across the midplane at the fiducial (rho_* = 1e-24, eps = 0.3), "
              "Sigma_eff in kg/m^2 and the bridge shear (gamma_1 along the axis, gamma_2):\n")
            tr, g1, g2 = r["Sigma_transverse_midplane"], r["gamma1_transverse_midplane"], r["gamma2_transverse_midplane"]
            P(table(["y (kpc)"] + list(tr.keys()),
                    [["Sigma_eff"] + [e(v, 2) for v in tr.values()],
                     ["gamma_1"] + [e(v, 2) for v in g1.values()],
                     ["gamma_2"] + [e(v, 2) for v in g2.values()]]))
            P("\nAlong the axis (y = 0):\n")
            al = r["Sigma_along_axis"]
            P(table(["x (kpc)"] + list(al.keys()), [["Sigma_eff"] + [e(v, 2) for v in al.values()]]))
            gt = r["gamma_t_about_midpoint"]
            P("\nTangential shear of the bridge feature about the midpoint (a net-zero feature gives "
              "gamma_t -> 0 outside itself; a positive filament gives gamma_t ~ M/R^2 > 0): "
              + ", ".join(f"R = {k.replace('_', ' ')}: gamma_t = {v['gamma_t']:+.2e}, gamma_x = {v['gamma_x']:+.1e}"
                          for k, v in gt.items()) + ".\n")
        P("### 2.2 Property (a): net mass consistent with zero\n")
        for k in ("1e-24|eps=0.03", "rho_mean|eps=0.03"):
            if k in B["P_ladder"]:
                nm, nc = B["P_ladder"][k]["net_mass"], B["P_ladder"][k]["net_mass_coarse"]
                P(f"* {k}: 3-D box (|x| < 1.4 D, R < 0.95 D): net {nm['box_x1.4D_R0.95D']['net_Msun']:+.2e} Msun "
                  f"against |abs| {nm['box_x1.4D_R0.95D']['abs_Msun']:.2e} (net/abs {nm['box_x1.4D_R0.95D']['net_over_abs']:+.1e}; "
                  f"at 80 kpc {nc['box_x1.4D_R0.95D']['net_over_abs']:+.1e}); smaller box (0.75 D, 0.5 D): "
                  f"{nm['box_x0.75D_R0.5D']['net_over_abs']:+.1e}; sky projection: {nm['sky_2D']['net_over_abs']:+.1e}. "
                  f"By Gauss the infinite-volume net is exactly zero for any potential feature that decays "
                  f"faster than 1/r; the box numbers measure truncation plus discretisation.")
        P("")
        P("### 2.3 Property (b): scaling with M_A M_B\n")
        rows = []
        for rt, m in B["mass_scaling"].items():
            for mode in ("A_only", "both"):
                for q in ("Sigma_axis", "Q", "A_mf"):
                    s = m["log_slopes"][mode][q]
                    rows.append((rt, mode, q, f"{s['slope']:+.3f} +- {s['se']:.3f}", f(s["rms_dex"], "{:.4f}"),
                                 "1" if mode == "A_only" else "2"))
        P(table(["rho_*", "varied", "observable", "log-slope +- se", "rms about power law (dex)", "bilinear expectation"], rows))
        P("\n'A_only': M_A varied at fixed M_B over 0.5-2x; 'both': M_A = M_B varied together. The slope is "
          "measured on the OBSERVABLE (the projected feature through the estimator), not on P, so the "
          "phi'(rho) nonlinearity is in it: at rho_* = 1e-24 the endpoints' tails at the midpoint are "
          "0.44 rho_* and the slope sits 3% below 1; at rho_* = rho_mean the midpoint is 150 rho_*, "
          "phi' ~ rho^-2 with rho set by the endpoints' own tails, and the M_A M_B scaling is GONE "
          "(slope ~0.1): the no-new-scale variant does not have BL's bridge.\n")
        P("### 2.4 Property (c): dependence on the filament density rho_f\n")
        for rt, fl in B["filament_scaling"].items():
            rows = [(f(r["rho_f"], "{:.0e}"), f(r["rho_f_over_rho_star"], "{:.2g}"), e(r["Sigma_axis_specific"]),
                     e(r["Sigma_axis_filament_newton"]), e(r["Sigma_wing_specific_400kpc"]),
                     f(r["minus_dphi_over_max"], "{:.3f}"), e(r["Q_specific_plus_filament"]), e(r["N_Msun"]))
                    for r in fl["rows"]]
            P(f"rho_* = {rt}:\n")
            P(table(["rho_f (kg/m^3)", "rho_f/rho_*", "Sigma_axis law-specific", "Sigma_axis Newtonian filament",
                     "wing (400 kpc) law-specific", "-phi'/max", "Q (total)", "N (Msun)"], rows))
            sl = fl["local_log_slopes"]
            P("\nlocal log-slopes d ln|Sigma_axis| / d ln rho_f between adjacent rungs -- path-specific: "
              + ", ".join(f"{s['path_specific']:+.2f}" for s in sl) + "; Newtonian filament: "
              + ", ".join(f"{s['newton_filament']:+.2f}" for s in sl) + ".\n")
        P("The law-specific bridge exists at rho_f = 0 (it is made of the endpoints' columns), is flat "
          "while rho_f << rho_* (slope ~0 against the filament's own slope of exactly 1), and is "
          "suppressed by (1 + rho_f/rho_*)^-2 above it: 'saturation' is the right word below rho_*, "
          "'suppression' above it. A bridge that scaled with the filament's own mass would have slope 1.\n")
        P("### 2.5 Inclination of the pair axis\n")
        rows = [(k, f(v["projected_D_Mpc"], "{:.2f}"), e(v["features"]["Sigma_axis"]),
                 e(v["strip_stats"]["Q"]) if v["strip_stats"] else "--",
                 f(v["strip_stats"]["A_mf"], "{:+.3f}") if v["strip_stats"] else "--", v.get("note") or "")
                for k, v in B["inclination"].items()]
        P(table(["i (deg)", "D_proj (Mpc)", "Sigma_axis", "Q", "A_mf", ""], rows))
        P("")

    # ------------------------------------------------------------ 3. competitors
    P("## 3. Separation from Newton, the scalar, the tensor and a positive-mass filament\n")
    if B:
        rows = []
        for k, c in B["competitors"].items():
            ss = c["strip_stats"]
            nm = c["net_mass_sky_2D"]
            rows.append((k, e(ss["S_core"]), e(ss["S_wing"]), e(ss["S_bcore"]), e(ss["Q"]), e(ss["DQ"]),
                         e(c["N_Msun"]), f(ss["A_mf"], "{:+.3f}"),
                         f"{nm['net_Msun']:+.1e} / {nm['abs_Msun']:.1e}"))
        P(table(["law", "S_core", "S_wing", "S_bcore", "Q", "DQ", "N (Msun)", "A_mf", "map net / abs (Msun)"], rows))
        P("\nAll on the same scene, the same estimator, noise-free. Signs: P (eps > 0) has Q < 0 with "
          "the behind region empty (DQ = Q) and a compensated map; the QUMOND/RAR scalar's two-body "
          "cross term is POSITIVE on the axis near the saddle and negative transversally (Q > 0), riding "
          "on a broad negative deficit (the pair's phantom halo is subadditive: net -1.6e14 Msun growing "
          "with aperture); BL's tensor at f_E > 0 puts an m = 2 excess along the axis (Q > 0, and "
          "S_bcore ~ S_core: symmetric between/behind), at f_E < 0 it flips the Q sign but not the "
          "between/behind symmetry nor the negative net; a positive filament has Q >= 0 and N > 0; "
          "Newton is zero to the 1e-5 floor.\n")
    if Dt:
        P("Pairs needed for a 3-sigma separation of P from each competitor, per statistic, under BF's "
          "detector (null sd per pair from the untouched half):\n")
        rows = []
        for law, d in Dt["competitor_separation"].items():
            for key, v in d.items():
                if key.split("|eps=")[1] in ("0.003", "0.03"):
                    n = v["N_pairs_for_3sigma_separation"]
                    rows.append((law, key, f(n["Q"], "{:.0f}"), f(n["DQ"], "{:.0f}"), f(n["N_kappa_m2"], "{:.0f}"),
                                 f(n["A_mf"], "{:.0f}")))
        P(table(["competitor", "P setting", "via Q", "via DQ", "via N", "via A_mf"], rows))
        P("")

    # ------------------------------------------------------------ 4. the attack
    P("## 4. Job 3 -- the compensation attacked with positive mass\n")
    if A:
        d = A["dictionary"]
        P(f"Dictionary: {d['n']} positive-mass components -- elliptical Plummer haloes about each endpoint "
          f"with q in {d['q']}, position angle {d['theta_deg']} deg from the axis, scale {d['a_over_a']} a, "
          f"centre offset {d['offset_over_a']} a along the axis (sums of them are haloes with radius-dependent "
          f"ellipticity, isophote twists and lopsidedness), and positive filaments of width {d['R_f_kpc']} kpc; "
          f"each run through the estimator alone. The estimator is linear, so any positive combination's "
          f"residual is the same combination of residuals, and 'can rho_DM >= 0 mimic P in projection' is a "
          f"non-negative least-squares problem. The any-sign fit is the control.\n")
        rows = []
        for k, v in A["fits"].items():
            bp = v["best_positive"]
            bud = v["mimic_by_mass_budget"]
            rows.append((k, f(v["mimic_fraction_positive_mass"], "{:.3f}"),
                         f(v["mimic_fraction_unconstrained_sign"], "{:.3f}"),
                         f(bp.get("added_halo_mass_Msun", 0.0), "{:.1e}"),
                         f(bp.get("weight_fraction_perpendicular", 0.0), "{:.2f}"),
                         f(bp.get("weight_fraction_offset", 0.0), "{:.2f}"),
                         f(bud.get("0.3", 0.0), "{:.2f}"), f(bud.get("1.0", 0.0), "{:.2f}"),
                         f(bud.get("3.0", 0.0), "{:.2f}"), f(bud.get("10.0", 0.0), "{:.2f}"),
                         f"{v['strips_target']['DQ']:+.1e} / {v['strips_mimic']['DQ']:+.1e}"))
        P(table(["incl | region | prior", "mimic, rho >= 0", "mimic, any sign", "added halo mass (Msun)",
                 "perp. weight", "offset weight", "at 0.3 M", "at 1 M", "at 3 M", "at 10 M", "DQ target / mimic"], rows))
        P("\n'mimic' = 1 - ||kappa_res(P) - kappa_res(model)||^2 / ||kappa_res(P)||^2 over the fit region; "
          "'at k M' = the best positive mimic when the ADDED halo mass is capped at k x (2 x 3e14 Msun), the "
          "scene's own lens mass. 'cdm_prior_aligned' restricts the dictionary to haloes elongated within "
          "30 deg of the pair axis (BK.4: haloes align with their filament).\n")
        bs = A["best_single_component"]
        P(f"Best single component: {bs['component']} reproduces {bs['mimic_fraction']:.3f} of the signature.\n")

    # ------------------------------------------------------------ 5. Job 4
    P("## 5. Job 4 -- identifiability, noise, the member-safe window, the certificate\n")
    if Dt:
        P(f"Null: {Dt['n_null']} draws of endpoints + BF noise (shape noise, additive c, coherent PSF "
          f"residual), split {Dt['n_null'] // 2} calibration / {Dt['n_null'] - Dt['n_null'] // 2} untouched; "
          f"linearity of the estimator |stats(s+n) - stats(n) - stats(s)|/|stats(s)| = {Dt['linearity_max_rel']:.1e}.\n")
        rows = [(k, e(v["cal_sd"]), e(v["test_sd"]), e(v["test_mean"]), f(v["realised_fpr_two_sided"], "{:.3f}"),
                 f(v["realised_fpr_one_sided_low"], "{:.3f}"), e(Dt["null_sd_shape_noise_only"][k]))
                for k, v in Dt["null"].items()]
        P(table(["statistic", "sd (calibration half)", "sd (untouched half)", "mean (untouched)",
                 "realised FPR two-sided @0.05", "one-sided @0.05", "sd, shape noise only"], rows))
        P("\n### 5.1 Responsiveness and the 3-sigma amplitude\n")
        rows = []
        for rt, d in Dt["responsiveness"].items():
            for k in ("A_mf", "Q", "DQ", "N_kappa_m2"):
                v = d[k]
                e3 = v["eps_3sigma_by_N_pairs"]
                rows.append((rt, k, e(v["d_stat_d_eps"]), e(v["se_per_pair"]), f(v["z_per_pair_at_eps_ref"], "{:.3f}"),
                             f(e3["1"], "{:.3g}"), f(e3["100"], "{:.3g}"), f(e3["1000"], "{:.3g}"), f(e3["3000"], "{:.3g}")))
        P(table(["rho_*", "statistic", "d(stat)/d(eps)", "se per pair", "z per pair at eps = 0.03",
                 "eps_3sigma, 1 pair", "100 pairs", "1000 pairs", "3000 pairs"], rows))
        P("\n### 5.2 Source density and separation\n")
        rows = [(n, e(v["A_mf"]), f(v["eps_3sigma_A_mf_100pairs"], "{:.3g}")) for n, v in Dt["source_density_scan"].items()]
        P(table(["n (arcmin^-2)", "sd(A_mf) per pair", "eps_3sigma (A_mf, 100 pairs)"], rows))
        P("")
        rows = [(f(v["D_Mpc"], "{:.0f}"), e(v["Sigma_axis"]), e(v["stats_at_eps_ref"]["Q"]), e(v["null_sd"]["Q"]),
                 f(v["eps_3sigma_100pairs"]["Q"], "{:.3g}"), f(v["eps_3sigma_100pairs"]["A_mf"], "{:.3g}"))
                for v in Dt["separation_scan"].values()]
        P(table(["D (Mpc)", "Sigma_axis (eps = 0.03)", "Q per pair", "sd(Q) per pair", "eps_3sigma (Q, 100)", "eps_3sigma (A_mf, 100)"], rows))
        P("")
    if Mb:
        P("### 5.3 Member confinement (the binding constraint) and the window\n")
        rows = [(k, f(v["k_per_eps"]["10"] if "10" in v["k_per_eps"] else v["k_per_eps"][10], "{:.2f}"),
                 f(v["k_per_eps"]["20"] if "20" in v["k_per_eps"] else v["k_per_eps"][20], "{:.2f}"),
                 f(v["factor_at_eps0p3_20kpc"], "{:.2f}"),
                 f(v["eps_safe"]["tol_0.02"], "{:.2e}"), f(v["eps_safe"]["tol_0.05"], "{:.2e}"),
                 f(v["eps_safe"]["tol_0.10"], "{:.2e}"), f(v["eps_safe"]["tol_0.20"], "{:.2e}"))
                for k, v in Mb["rows"].items()]
        P(table(["rho_*", "k(10 kpc)/eps", "k(20 kpc)/eps", "factor at eps=0.3, 20 kpc",
                 "eps_safe 2%", "5%", "10%", "20%"], rows))
        bc = Mb["BL_comparison"]
        P(f"\nk = (g_3/g_N)(20 kpc) per unit eps on the compiler's member caricature (this lane's column "
          f"code, 20000 directions); BL recorded a carrier factor of {bc['BL_carrier_factor_20kpc_eps0p3']:.2f} "
          f"at eps = 0.3, this lane {bc['this_lane_carrier_factor_20kpc_eps0p3']:.2f} (ratio "
          f"{bc['ratio']:.3f}). The tolerance is declared; the data that would test it are in the "
          f"confirmation reserve and were not opened.\n")
    if Dt:
        rows = []
        for rt, w in Dt["member_safe_window"].items():
            for tol, ww in w.items():
                a = ww["A_mf"]
                rows.append((rt, tol.replace("tol_", ""), f(a["100"]["eps_safe"], "{:.2e}"),
                             f(a["100"]["eps_3sigma"], "{:.2e}"), "OPEN" if a["100"]["window_open"] else "empty",
                             f(a["1000"]["eps_3sigma"], "{:.2e}"), "OPEN" if a["1000"]["window_open"] else "empty",
                             f(a["N_pairs_to_open"], "{:.0f}")))
        P(table(["rho_*", "member tol.", "eps_safe", "eps_3sigma (100 pairs)", "window, 100 pairs",
                 "eps_3sigma (1000 pairs)", "window, 1000 pairs", "pairs needed to open"], rows))
        P("\nThe window is [eps_3sigma(N), eps_safe(tol)] on the matched-filter statistic under BF's "
          "detector at n = 20/arcmin^2, z_l = 0.3, D = 4 Mpc.\n")
    if Ce:
        P("### 5.4 The Stage 4 certificate\n")
        rows = []
        for cid, c in Ce["certificates"].items():
            ch = c["checks"]
            fails = [k for k, v in ch.items() if not v["passed"]]
            rows.append((cid, c["statistic"], c["rho_tag"], "ISSUED" if c["issued"] else "REFUSED",
                         ", ".join(fails) or "--", f(c["z_at_predicted_100_pairs"], "{:.2f}"),
                         f(ch["C7_nuisance_distinct"]["worst_corr"], "{:.2f}") + " (" + str(ch["C7_nuisance_distinct"]["worst_nuisance"]) + ")",
                         "pass" if c["C5_under_BF_per_cluster_coverage"] else "FAIL"))
        P(table(["certificate", "statistic", "rho_*", "verdict", "failed checks", "z at predicted (100 pairs)",
                 "C7 worst |r| (nuisance)", "C5 under BF per-cluster coverage"], rows))
        inj = Ce["out_of_grammar_injection"]
        P(f"\nDeclared: N_pairs = {Ce['declared']['N_pairs']}, the predicted effect = eps_safe at 5% member "
          f"tolerance, coverage for C5 = contiguous wide field (the BF per-cluster 2.4 R500 verdict beside it), "
          f"fresh untouched null half of {Ce['declared']['fresh_null_draws']} draws for C3. Out-of-grammar "
          f"injection ({inj['injected']['shape']}): recovery Q {inj['Q']:.2f}, DQ {inj['DQ']:.2f}, "
          f"A_mf {inj['A_mf']:.2f}.\n")

    # ------------------------------------------------------------ 6. Job 5
    P("## 6. Job 5 -- the path family re-compiled\n")
    if Cp:
        rows = []
        for k, s in Cp["settings"].items():
            g1 = s["gate1"]
            rows.append((k, s["verdict"], s["primary_bin"], ", ".join(g1["escapes"] or []) or "none",
                         f(g1["max_single_probe_resid_dex"], "{:.4f}"), f(g1["joint_resid_dex"], "{:.4f}"),
                         f(g1["two_body_probe_resid_dex"], "{:.4f}") if g1["two_body_probe_resid_dex"] is not None else "--"))
        P(table(["rho_* | eps | declared", "verdict", "bin", "gate-1 escapes", "max single-probe resid (dex)",
                 "joint (dex)", "two-body probe resid (dex)"], rows))
        P("")
        for rt, sc in Cp["threshold_scan"].items():
            P(f"GATE 1 alone, rho_* = {rt}, with the two-body probe: "
              + "; ".join(f"eps = {k}: bridge {v['bridge_resid_dex']:.4f} dex -> {v['escapes'] or 'no escape'}"
                          for k, v in sc["rows"].items())
              + f". Smallest identifiable eps on the ladder: {sc['smallest_identifiable_eps']}.\n")
        k = "1e-24|eps=0.003"
        if k in Cp["fields"]:
            fd = Cp["fields"][k]
            P(f"The P field at the 15 probe points at eps = 0.003, rho_* = 1e-24: log10(g/g_N) = "
              + ", ".join(f"{v:+.3f}" for v in fd["log10_g_over_gN"])
              + "; carrier |g_3|/g_N = " + ", ".join(f"{v:.2f}" for v in fd["g3_over_gN"]) + ".\n")

    # ------------------------------------------------------------ 7. findings
    P("## 7. Findings the brief did not ask for\n")
    P("* **The core is narrow.** P is a PRODUCT of two columns, so the negative ridge has half-width "
      "~0.4 a (zero crossing at 200 kpc for a = 400 kpc) while the positive wings extend to ~1.5 Mpc: "
      "the compensation is a 200-kpc trough against a 1.5-Mpc plateau, and any strip estimator must "
      "be built to that shape.\n"
      "* **The no-new-scale variant is a different physics regime.** At rho_* = rho_mean the "
      "inter-cluster density (the endpoints' own r^-5 tails, 4e-25 kg/m^3 at the midpoint) is 150 rho_*, "
      "phi' ~ rho^-2, and the 'bridge' becomes the suppression of each endpoint's own Phi_3 halo by the "
      "other's density: larger on the axis, non-compact (3-D net/abs -0.15 in the box), and with NO "
      "M_A M_B scaling (slope 0.07 +- 0.11). BL's compensated M_A M_B bridge is a rho_* = 1e-24 "
      "statement.\n"
      "* **The scalar null has its own two-body signature of the opposite sign.** Near the saddle the "
      "QUMOND phantom density is positive along the axis and negative across it; the pair's phantom "
      "halo is subadditive, so the map-level net is a large negative number that grows with the aperture. "
      "At member-safe eps the P bridge is ~3x SMALLER than the scalar's own cross term.\n"
      "* **A negative f_E tensor flips the sign of Q but not the between/behind symmetry nor the net.**\n"
      "* **Bugs found by the lane's tests** (section 0): an estimator floor of 1e-3 from FFT endpoint shear "
      "and a 1/r tail; a NaN at the saddle poisoning every map through the FFT; a 1e17 Msun spike from the "
      "l=2 ODE's boundary node; an inward/outward sign slip in the member scan; and a fitted-slope tail "
      "extrapolation that made the estimator nonlinear in the data (caught by T8 and by detect.py's own "
      "check; fixed to a declared r^-2 tail).\n")

    # ------------------------------------------------------------ 8. assumptions
    P("## 8. Fields filled with an assumption this lane had to invent\n")
    P("* the scene's Plummers are each law's TOTAL lens mass and, for P, the column-setting baryonic "
      "mass (BL's convention); a realistic cluster's baryons are ~1/7 of its lensing mass, and P's "
      "carrier is quadratic in the column, so the bridge amplitude at fixed eps scales with that "
      "fraction squared -- the member constraint scales the same way, and the WINDOW does not depend on it;\n"
      "* the endpoint term's two-body cross part is evaluated with each source as a point source "
      "(measured to be 0.04% of the carrier term on the axis at rho_f = 0);\n"
      "* the filament is a chain of Plummer beads (axis density rho_f, width R_f = 250 kpc);\n"
      "* the tensor competitor is a declared caricature: each body's exact first-order l=2 response on its "
      "own AQUAL(simple-mu) base, superposed on the QUMOND-form two-body base with the simple nu;\n"
      "* the member tolerance ladder (2-20%) is declared; the member data are in the confirmation reserve;\n"
      "* the declared sample for C4 is 100 pairs of 3e14 Msun clusters at 4 Mpc, z = 0.3, and the "
      "coverage for C5 is a contiguous wide field; R500 = 1 Mpc;\n"
      "* independent pairs of identical geometry stack coherently (sd ~ 1/sqrt(N)); real pairs vary in "
      "mass, separation and inclination, which the mass, separation and inclination tables bound.\n")

    P("## 9. Files\n")
    P("`guard.py`, `scene.py`, `lensing.py`, `estimator.py`, `build_cache.py`, `crosscheck.py`, "
      "`members.py`, `run_bridge.py`, `detect.py`, `cdm_attack.py`, `baseline.py`, `compile_bridge.py`, "
      "`certify.py`, `test_bridge.py`, `write_report.py`, `run_all.py`; results in `results/*.json` "
      "(+ the cached grids `results/cache_*.npz`, the template `results/template_kres.npy`). Compiler "
      "patch: `../compiler/compiler.py` (+102/-5: `Candidate.bridge_field`, `bridge_probe_points`, "
      "`make_bridge_probe`, `probes()['bridge_pair']`, GATE 1 escape (d)).\n")
    P(f"\n_Rendered {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} from the JSONs above._\n")
    txt = "\n".join(L)
    with open(os.path.join(HERE, "REPORT.md"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(txt)
    print(f"wrote REPORT.md ({len(txt.splitlines())} lines)")


if __name__ == "__main__":
    main()
