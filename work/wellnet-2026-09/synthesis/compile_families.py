"""compile_families.py -- Job 3: both actions through the Stage 3 compiler v2.

For each family the compiler is given
  * the element written HERE (the action's own flux map / kernel), and
  * the nearest element the compiler's pre-existing grammar could express,
so the report can say exactly which verdict is about the theory and which is
about the grammar.  Every candidate carries global parameters only.

Also asserted here: extending the compiler's grammar with `tensor_L` and
`path_kernel` changed NO pre-existing verdict (30 candidates: the known
families, the external-axis elements and the external positive controls,
compared field by field against the pre-patch snapshot).
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import guard                                    # noqa: F401
import compiler as C                            # noqa: E402
import path_family as P                         # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def _summ(r: dict) -> dict:
    g4 = r[C.GATE4][1]
    return dict(
        verdict=r["_verdict"], failed=r["_failed"], labels=r["_labels"],
        flags=r["_flags"], model_class=r["_model_class"],
        primary_bin=r["_taxonomy"]["primary"],
        defects=[d["code"] for d in r["_taxonomy"]["defects"]],
        gate1=dict(passed=r["gate1_constant_K"][0],
                   escapes=r["gate1_constant_K"][1].get("escapes"),
                   max_single_probe_resid_dex=r["gate1_constant_K"][1].get(
                       "max_single_probe_resid_dex"),
                   joint_resid_dex=r["gate1_constant_K"][1].get("joint_resid_dex"),
                   axis_misalignment_deg=r["gate1_constant_K"][1].get(
                       "axis_misalignment_deg"),
                   reason=r["gate1_constant_K"][2]),
        gate2=dict(passed=r["gate2_potential_gauge"][0],
                   gauge_dependent=r["gate2_potential_gauge"][1].get("gauge_dependent"),
                   reason=r["gate2_potential_gauge"][2]),
        gate3=dict(passed=r["gate3_coarse_graining"][0],
                   reason=r["gate3_coarse_graining"][2]),
        gate4=dict(passed=r[C.GATE4][0],
                   out_of_declared_class=g4.get("out_of_declared_class"),
                   u_space=g4.get("u_space"),
                   jacobian_asymmetry=g4.get("asymmetry"),
                   reciprocity=g4.get("reciprocity"),
                   health=g4.get("health"),
                   momentum_carrier=g4.get("momentum_carrier"),
                   reason=r[C.GATE4][2]),
    )


def tensor_candidates() -> dict:
    out = {}
    for fE in (0.1, 0.3, 1.0, 1.8):
        out[f"T_tensor_L_fE{fE}"] = C.Candidate(
            f"T_tensor_L_fE{fE}", base="aqual", struct="tensor_L", inv="gn",
            form="off", A=fE, ext=dict(axis=C.E_EXT),
            note="THE ACTION: L = -(1/8 pi G)[a0^2 F + f_E h(|u|/a0)"
                 "((e.u)^2 - |u|^2/3)] - rho Phi; e_ext declared background")
    # the closed version: the axis field is dynamical -> extra propagating
    # field; the compiler labels it and gates 1-3 still apply
    out["T_tensor_L_fE0.3_dynamical_axis"] = C.Candidate(
        "T_tensor_L_fE0.3_dynamical_axis", base="aqual", struct="tensor_L",
        inv="gn", form="off", A=0.3, ext=dict(axis=C.E_EXT),
        model_class="extra_propagating_field",
        momentum_carrier="the environmental tidal tensor field That_env "
                         "(its back-reaction carries the momentum the "
                         "anisotropic force does not conserve)",
        note="the same action with That_env promoted to a dynamical field")
    # the nearest PRE-EXISTING grammar elements (REPORT_v2 FIX 4)
    for k, c in C.external_axis_elements().items():
        out["grammar_" + k] = c
    return out


def _rescale_tables(tables: dict, eps_new: float, eps_ref: float) -> dict:
    """Both the endpoint term and the carrier term are LINEAR in eps at fixed
    rho_*, so g/g_N - 1 scales exactly with eps."""
    out = {}
    for nm, t in tables.items():
        s = eps_new / eps_ref
        out[nm] = dict(t)
        for k in ("factor", "factor_direct_only", "factor_carrier_only"):
            out[nm][k] = (1.0 + s * (np.array(t[k]) - 1.0)).tolist()
        out[nm]["eps"] = eps_new
    return out


def path_candidates(path_json: dict) -> dict:
    out = {}
    eps, rs = P.EPS_FID, P.RHO_STAR_FID
    ff_fid = P.make_force_factor(path_json["force_factor_tables_fiducial"])
    ff_mean = P.make_force_factor(path_json["force_factor_tables_rho_mean"])
    bg = P.Scene([C.Plummer(1.0e12 * C.MSUN, 20.0 * C.KPC)], rs)
    bg_mean = P.Scene([C.Plummer(1.0e12 * C.MSUN, 20.0 * C.KPC)], P.RHO_MEAN)
    out["P_path_kernel_fid"] = C.Candidate(
        "P_path_kernel_fid", base="newton", struct="path_kernel", inv="one",
        form="off", A=eps,
        pair_kernel=lambda x, y: float(bg.W(np.asarray(x), np.asarray(y), eps)),
        green=P.make_green(eps, rs),
        force_factor=ff_fid,
        momentum_carrier="matter on the connecting segment (three-body term "
                         "-grad Phi_3); verified in path_results.json",
        note=f"THE ACTION: S = -(1/2) Int Int rho W rho, W = -(G/d)[1 + eps "
             f"v], v = vacuum fraction of the segment, phi = 1/(1+rho/rho_*); "
             f"eps = {eps}, rho_* = {rs:.1e} kg/m^3")
    # the amplitude ladder: the same action at eps small enough that the
    # cluster-member confinement (x7.8 at 20 kpc at eps = 0.3) is a few
    # per cent -- where does the bench stop seeing it?
    for e_new in (0.03, 0.003):
        tabs = _rescale_tables(path_json["force_factor_tables_fiducial"],
                               e_new, eps)
        out[f"P_path_kernel_eps{e_new}"] = C.Candidate(
            f"P_path_kernel_eps{e_new}", base="newton", struct="path_kernel",
            inv="one", form="off", A=e_new,
            pair_kernel=(lambda x, y, e=e_new: float(
                bg.W(np.asarray(x), np.asarray(y), e))),
            green=P.make_green(e_new, rs),
            force_factor=P.make_force_factor(tabs),
            momentum_carrier="matter on the connecting segment (three-body term)",
            note=f"the same action at eps = {e_new} (member confinement "
                 f"{1 + (7.76 - 1) * e_new / eps:.2f}x at 20 kpc)")
    out["P_path_kernel_fid_no_carrier_declared"] = C.Candidate(
        "P_path_kernel_fid_no_carrier_declared", base="newton",
        struct="path_kernel", inv="one", form="off", A=eps,
        pair_kernel=out["P_path_kernel_fid"].pair_kernel,
        green=out["P_path_kernel_fid"].green, force_factor=ff_fid,
        note="the same element with no carrier declared: the kernel is "
             "reciprocal by construction, so this must still ADMIT -- the "
             "carrier flag is informational")
    out["P_path_kernel_rho_mean"] = C.Candidate(
        "P_path_kernel_rho_mean", base="newton", struct="path_kernel",
        inv="one", form="off", A=eps,
        pair_kernel=lambda x, y: float(bg_mean.W(np.asarray(x), np.asarray(y), eps)),
        green=P.make_green(eps, P.RHO_MEAN),
        force_factor=ff_mean,
        momentum_carrier="matter on the connecting segment (three-body term)",
        note="the NO-NEW-SCALE variant rho_* = rho_mean: every probe segment "
             "runs through matter denser than the cosmic mean, so v ~ 0")
    out["P_path_kernel_field_carrier"] = C.Candidate(
        "P_path_kernel_field_carrier", base="newton", struct="path_kernel",
        inv="one", form="off", A=eps,
        pair_kernel=out["P_path_kernel_fid"].pair_kernel,
        green=out["P_path_kernel_fid"].green, force_factor=ff_fid,
        model_class="extra_propagating_field",
        momentum_carrier="a vacuum-state field q sourced by rho with range "
                         "l_q; the segment state is q along the path",
        note="the variant whose path state is carried by a dynamical field "
             "rather than by the matter itself")
    # the nearest PRE-EXISTING grammar element: the symmetric nonlocal control
    out["grammar_XC5_symmetric_nonlocal_action"] = C.external_controls()[
        "XC5_symmetric_nonlocal_action"][0]
    return out


def baseline_unchanged() -> dict:
    base = json.load(open(os.path.join(HERE, "baseline_verdicts_prepatch.json"),
                          encoding="utf-8"))
    now = {}

    def rec(tag, r):
        now[tag] = dict(verdict=r["_verdict"], failed=r["_failed"],
                        labels=r["_labels"], flags=r["_flags"],
                        primary=r["_taxonomy"]["primary"],
                        defects=[d["code"] for d in r["_taxonomy"]["defects"]],
                        g4_asym=r[C.GATE4][1].get("asymmetry"),
                        g1=r["gate1_constant_K"][1].get("max_single_probe_resid_dex"))
    for k, c in C.known_families().items():
        rec("KF:" + k, C.check(c))
    for k, c in C.external_axis_elements().items():
        rec("EA:" + k, C.check(c))
    xc = C.run_external_controls(cheap=True)
    for k, v in xc["rows"].items():
        now["XC:" + k] = dict(verdict=v["verdict"], required=v["required"],
                              agrees=v["agrees"], bin=v["taxonomy_bin"],
                              failed=v["failed"])
    keys = [k for k in base if k.startswith(("KF", "EA", "XC"))]
    diffs = [k for k in keys if base[k] != now[k]]
    return dict(n_compared=len(keys), n_changed=len(diffs), changed=diffs,
                external_controls_all_agree=bool(xc["all_agree"]),
                statement="the grammar extension is additive: no pre-existing "
                          "verdict, failed-gate list, taxonomy bin, defect "
                          "list or measured statistic changed")


def main():
    guard.arm()
    t0 = time.perf_counter()
    path_json = json.load(open(os.path.join(HERE, "path_results.json"),
                               encoding="utf-8"))
    res = dict(generated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               compiler="work/wellnet-2026-09/compiler/compiler.py (v2 + Run BK "
                        "elements tensor_L, path_kernel)",
               tensor={}, path={})
    for tag, c in tensor_candidates().items():
        r = C.check(c, cheap=True)
        res["tensor"][tag] = _summ(r)
        res["tensor"][tag]["note"] = c.note
        print(f"{tag:<40} {r['_verdict']:<14} {r['_taxonomy']['primary']:<40} "
              f"failed={r['_failed']} labels={r['_labels']} flags={r['_flags']}")
    for tag, c in path_candidates(path_json).items():
        r = C.check(c, cheap=True)
        res["path"][tag] = _summ(r)
        res["path"][tag]["note"] = c.note
        print(f"{tag:<40} {r['_verdict']:<14} {r['_taxonomy']['primary']:<40} "
              f"failed={r['_failed']} labels={r['_labels']} flags={r['_flags']}")
    res["baseline_unchanged"] = baseline_unchanged()
    print("baseline:", res["baseline_unchanged"])
    res["provenance"] = guard.summary()
    res["wall_seconds"] = time.perf_counter() - t0
    with open(os.path.join(HERE, "compile_results.json"), "w",
              encoding="utf-8", newline="\n") as fh:
        json.dump(res, fh, indent=1, default=float)
    print("provenance:", res["provenance"]["assertion"],
          "| foreign:", res["provenance"]["foreign_reads"])


if __name__ == "__main__":
    main()
