"""run_scene.py -- the Stage 1 driver.  Writes `scene_results.json`.

Every number in `REPORT.md` is rendered from this file by `write_report.py`;
nothing is typed in by hand.

    python run_scene.py            -> scene_results.json

NO OBSERVATIONAL DATA IS OPENED.  The scenes are synthetic and the inventory is
metadata about what exists, not the data itself.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import metadata as M                                            # noqa: E402
import registry as REG                                          # noqa: E402
import schema as S                                              # noqa: E402
import ensemble as E                                            # noqa: E402
import commutation as C                                         # noqa: E402
import bridge as B                                              # noqa: E402
import inventory as INV                                         # noqa: E402

KPC, MPC = C.KPC, C.MPC
SEED = 20260904


# ------------------------------------------------------------------ scene
def demo_scene(reg, n_members=120, seed=SEED):
    """A small probabilistic cluster scene exercising every node and edge
    type the charter lists.  Synthetic; nothing is read from a catalogue."""
    mem, phase, z_true = E.synthetic_cluster(n_members, seed)
    b = E.SceneEnsembleBuilder("demo_cluster", reg, phase, mem, seed=seed)
    g = b.build()
    rng = np.random.default_rng(seed)

    # -- one node of every remaining charter type, so the schema is exercised
    g.add_node(S.Node("bcg", "central_galaxy", {
        "x": S.Fixed(0.0), "y": S.Fixed(0.0),
        "z": S.Fixed(0.0, "the centre defines the origin"),
        "m_star": S.Fixed(2.0e12 * E.MSUN),
        "r_e": S.Fixed(300.0 * KPC), "sersic_n": S.Fixed(5.55),
    }, source="resolved surface photometry"))
    g.add_node(S.Node("icl", "intracluster_light", {
        "m_icl": S.Uncertain(lambda r, n: 10 ** r.normal(12.6, 0.25, n)
                             * E.MSUN, label="ICL mass posterior"),
    }, source="deep surface photometry below a stated mu cut"))
    g.add_node(S.Node("bh", "black_hole", {
        "m_bh": S.Uncertain(lambda r, n: 10 ** r.normal(10.0, 0.4, n)
                            * E.MSUN, label="M-sigma posterior"),
    }, source="M-sigma scaling"))
    for i in range(6):
        th = 2 * math.pi * i / 6
        g.add_node(S.Node(f"gas{i}", "gas_cell", {
            "x": S.Fixed(400 * KPC * math.cos(th)),
            "y": S.Fixed(400 * KPC * math.sin(th)), "z": S.Fixed(0.0),
            "n_e": S.Uncertain(lambda r, n: 10 ** r.normal(-3.0, 0.12, n),
                               label="deprojected n_e"),
            "T_x": S.Uncertain(lambda r, n: r.normal(7.5e7, 6e6, n),
                               label="spectral T"),
        }, source="X-ray surface-brightness deprojection"))
    for i in range(4):
        g.add_node(S.Node(f"src{i}", "background_source", {
            "x": S.Fixed(rng.uniform(-1, 1) * MPC),
            "y": S.Fixed(rng.uniform(-1, 1) * MPC),
            "e1": S.Uncertain(lambda r, n: r.normal(0.0, 0.28, n),
                              label="measured shape + shape noise"),
            "e2": S.Uncertain(lambda r, n: r.normal(0.0, 0.28, n),
                              label="measured shape + shape noise"),
        }, source="per-source shape catalogue"))
    g.add_node(S.Node("obs", "observer", {"distance": S.Fixed(935.6 * MPC)},
                      source="assumed distance-redshift relation"))
    g.add_node(S.Node("hst", "instrument",
                      {"psf_fwhm": S.Fixed(0.4 * KPC),
                       "shear_m": S.Uncertain(
                           lambda r, n: r.normal(0.0, 0.01, n),
                           label="multiplicative shear calibration")},
                      source="instrument model"))
    g.add_node(S.Node("void1", "void", {"rho_env": S.Fixed(1e-27)},
                      source="environment reconstruction"))
    g.add_node(S.Node("fil1", "filament",
                      {"rho_env": S.Fixed(4e-26),
                       "ext_axis": S.Fixed((0.0, 0.0, 1.0))},
                      source="environment reconstruction"))
    g.add_node(S.Node("sad1", "saddle", {"phi_depth_saddle": S.Fixed(0.0)},
                      source="the boundary rule's own locating surface"))
    g.add_node(S.Node("bnd1", "boundary", {"r_3d": S.Fixed(3.0 * MPC)},
                      source="declared scene volume"))
    g.add_node(S.Node("star1", "star_population",
                      {"sigma_star": S.Uncertain(
                          lambda r, n: r.normal(220e3, 15e3, n),
                          label="pPXF aperture dispersion"),
                       "x": S.Fixed(50 * KPC), "y": S.Fixed(0.0),
                       "z": S.Fixed(0.0)},
                      source="IFU pPXF fit"))
    g.add_node(S.Node("sub1", "compact_substructure",
                      {"mass": S.Fixed(5e12 * E.MSUN), "x": S.Fixed(700 * KPC),
                       "y": S.Fixed(200 * KPC), "z": S.Fixed(0.0)},
                      source="member overdensity"))
    g.add_node(S.Node("latent1", "latent_field_cell",
                      {"vacuum_order": S.Fixed(0.0),
                       "vacuum_axis": S.Fixed((0.0, 0.0, 1.0)),
                       "field_memory": S.Fixed(0.0)},
                      source="generated by a candidate universe"))
    # a DM-contaminated product, present so the guard can catch it
    g.add_node(S.Node("kappa_map", "latent_field_cell",
                      {"kappa": S.Fixed(0.3)},
                      source="parametric lens model with one halo per member",
                      presupposes_dm=True,
                      dm_reason="a lens model that assigns a dark-matter clump "
                                "to each cluster galaxy by construction is "
                                "circular for any does-lensing-follow-light "
                                "test"))

    ids = [m.mid for m in mem][:8]
    for i, a in enumerate(ids):
        g.add_edge(S.Edge(f"sep{i}", "spatial_separation", a, "bcg"))
        g.add_edge(S.Edge(f"rel{i}", "relative_velocity", a, "bcg"))
        g.add_edge(S.Edge(f"mem{i}", "membership", a, "bcg",
                          {"p_member": S.Fixed(0.95)}))
        g.add_edge(S.Edge(f"tid{i}", "tidal_pair", a, "sub1"))
        g.add_edge(S.Edge(f"orb{i}", "orbital", a, "bcg"))
    for i in range(4):
        g.add_edge(S.Edge(f"lp{i}", "light_path", f"src{i}", "obs",
                          {"path_density": S.Fixed(2e-2),
                           "path_void_fraction": S.Fixed(0.4)}))
        g.add_edge(S.Edge(f"cov{i}", "shared_covariance", f"src{i}", "hst"))
        g.add_edge(S.Edge(f"cau{i}", "causal_retarded", f"src{i}", "bcg"))
    g.add_edge(S.Edge("ss0", "source_source", "fil1", "bcg"))
    g.add_edge(S.Edge("if0", "image_family", "src0", "src1",
                      {"time_delay": S.Fixed(376.02 * 86400.0)}))
    return g, b, z_true


# ------------------------------------------------------------------ blocks
def block_schema(reg, g):
    audit = M.audit_contract(reg)
    return {
        "node_types": list(S.NODE_TYPES),
        "edge_types": list(S.EDGE_TYPES),
        "field_types": list(S.FIELD_TYPES),
        "charter_node_bullets": list(S.CHARTER_NODE_BULLETS),
        "n_node_types": len(S.NODE_TYPES), "n_edge_types": len(S.EDGE_TYPES),
        "n_field_types": len(S.FIELD_TYPES),
        "registry_size": len(reg),
        "contract_audit": audit,
        "contract_items": len(M.CHARTER_ITEMS),
        "ontology_coverage": {str(k): v for k, v in
                              REG.coverage_by_section(reg).items()},
        "gauge_unsafe": reg.gauge_unsafe(),
        "non_commuting": reg.non_commuting(),
        "n_non_commuting": len(reg.non_commuting()),
        "catalogue_dependent": reg.catalogue_dependent(),
        "not_independently_measurable": reg.not_independently_measurable(),
        "n_not_independently_measurable":
            len(reg.not_independently_measurable()),
        "acausal": reg.acausal(),
        "exact_identities": [{"target": t, "inputs": list(i), "relation": r}
                             for t, i, r in REG.EXACT_IDENTITIES],
        "demo_scene": g.to_json(),
        "demo_scene_uncertain_attrs": len(g.uncertain_attrs()),
    }


def block_ensemble(g, b, z_true, n_draws=64):
    t0 = time.perf_counter()
    ens = g.ensemble(n_draws, SEED, "SceneEnsembleBuilder")
    wall = time.perf_counter() - t0
    diag = b.diagnostics(ens)
    cov = E.coverage_test(400, SEED)

    # is the ensemble a posterior or a disguised point estimate?
    ids = [m.mid for m in b.members]
    Z = np.array([[d.node_attrs[i]["z"] for i in ids] for d in ens.draws])

    # E[f(scene)] vs f(E[scene]) -- the root-data rule at ensemble level
    R = np.array([m.R for m in b.members])

    def f_r3(d):
        z = np.array([d.node_attrs[i]["z"] for i in ids])
        return float(np.mean(np.sqrt(R ** 2 + z ** 2)))
    m_of_f, sd_of_f = ens.expectation(f_r3)
    f_of_m = float(np.mean(np.sqrt(R ** 2 + Z.mean(axis=0) ** 2)))

    # how much does the VELOCITY term actually add over the density prior?
    s = E.DepthSampler(b.phase)
    prior_only = E.ClusterPhaseSpace(
        b.phase.profile,
        E.DispersionProfile(b.phase.dispersion.sigma0 * 1e6,
                            b.phase.dispersion.rs, b.phase.dispersion.beta),
        b.phase.morphology, b.phase.selection)
    w_full = np.mean([np.diff(s.credible_interval(m.R, m.v_los, m.is_early,
                                                  b.phase, 0.68))[0]
                      for m in b.members])
    w_prior = np.mean([np.diff(s.credible_interval(m.R, m.v_los, m.is_early,
                                                   prior_only, 0.68))[0]
                       for m in b.members])
    # ESS with the morphology term reweighted vs folded into the proposal
    ess = {}
    for flag in (False, True):
        bb = E.SceneEnsembleBuilder("demo_cluster", b.registry, b.phase,
                                    b.members, seed=SEED,
                                    exact_morphology=flag)
        ee = bb.build().ensemble(n_draws, SEED)
        ess["exact_proposal" if flag else "importance_reweighted"] =             bb.diagnostics(ee)["ess"]

    return {
        "n_draws": n_draws, "wall_s": wall,
        "ess_contrast": ess,
        "draws_per_s": n_draws / wall if wall else math.inf,
        "diagnostics": diag,
        "coverage_test": cov,
        "commute_check": {
            "E_of_f_r3d_Mpc": m_of_f / MPC, "sd_Mpc": sd_of_f / MPC,
            "f_of_E_r3d_Mpc": f_of_m / MPC,
            "difference_Mpc": (m_of_f - f_of_m) / MPC,
            "difference_pct": 100.0 * (m_of_f - f_of_m) / m_of_f,
            "note": "E[f(scene)] against f(E[scene]) for the mean 3-D radius. "
                    "Non-zero because sqrt is concave: collapsing the ensemble "
                    "to its mean scene UNDERSTATES every mean 3-D radius."},
        "velocity_information": {
            "posterior_width_68_Mpc": w_full / MPC,
            "density_prior_only_width_68_Mpc": w_prior / MPC,
            "width_ratio": w_full / w_prior,
            "note": "how much the line-of-sight VELOCITY narrows the depth "
                    "beyond the radial number-density prior alone"},
    }


def block_commutation():
    out = {}
    sf = C.flattened_cluster_scene(300, SEED, q_z=0.55)
    C.attach_history(sf)
    ss = C.synthetic_cluster_scene(300, SEED)
    r = 1000.0 * KPC
    out["scene"] = {"n_sources": sf.n(),
                    "total_mass_Msun": sf.total_mass() / C.MSUN,
                    "n_galaxies": 300, "flattening_q_z": 0.55,
                    "probe_radius_kpc": r / KPC}

    # -- the null control, against the analytic reference
    nl = []
    for rr in (300 * KPC, 1000 * KPC, 2000 * KPC):
        ex = C.analytic_spherical_avg_g(ss, rr)
        v = C.shell_radial_g(C.Newtonian(), ss, rr, 256, 8)
        nl.append({"radius_kpc": rr / KPC, "analytic": ex, "quadrature": v,
                   "rel_err": v / ex - 1.0})
    out["null_control"] = {
        "rows": nl,
        "max_abs_rel_err": max(abs(x["rel_err"]) for x in nl),
        "note": "Newtonian gravity is linear in the source, so the shell "
                "average of the resolved field EQUALS the field of the "
                "spherically averaged source exactly. The residual is the "
                "gate's quadrature floor and nothing else."}

    # -- quadrature convergence, the evidence for BUG 3
    conv = []
    for nd, nr in ((256, 1), (256, 8), (768, 24)):
        e = max(abs(C.shell_radial_g(C.Newtonian(), ss, rr, nd, nr)
                    / C.analytic_spherical_avg_g(ss, rr) - 1.0)
                for rr in (300 * KPC, 1000 * KPC, 2000 * KPC))
        conv.append({"n_dir": nd, "n_rot": nr, "max_abs_rel_err": e})
    out["quadrature_convergence"] = conv

    # -- the erasure matrix
    pl = C.PathLaw(0.30).calibrate(sf, r)
    cases = [
        ("nonlinearity", C.QuasiLinearMOND(), C.SphericalAverage(24), ss,
         "shell_radial_g", 1),
        ("depth fabrication", C.QuasiLinearMOND(), C.LOSCollapse(), ss,
         "shell_radial_g", 1),
        ("directional, SOURCE axis", C.SourceAlignedTensor(0.30),
         C.AzimuthalAverage(), sf, "shell_quadrupole", 8),
        ("directional, EXTERNAL axis", C.ExternalAxisTensor(0.30),
         C.AzimuthalAverage(), sf, "shell_quadrupole", 8),
        ("network", C.WellNetwork(0.30, L=300 * KPC),
         C.GaussianSmooth(300 * KPC), sf, "shell_radial_g", 1),
        ("network vs merge", C.WellNetwork(0.30, L=300 * KPC),
         C.CatalogueMerge(150 * KPC), sf, "shell_radial_g", 1),
        ("path", pl, C.SphericalAverage(24), sf, "shell_dispersion", 1),
        ("memory", C.MemoryLaw(0.60), C.PresentOnly(), sf, "shell_radial_g", 1),
    ]
    rows = []
    for label, law, op, scene, obs, nd in cases:
        t0 = time.perf_counter()
        e = C.erasure(law, op, scene, r, observable=obs,
                      n_dir=(128 if obs == "shell_dispersion" else 256),
                      n_rot=(4 if obs == "shell_dispersion" else 8),
                      n_op_draws=nd)
        v = C.erasure_verdict(e, target_precision=0.01)
        v["label"] = label
        v["wall_s"] = time.perf_counter() - t0
        rows.append(v)
    out["erasure"] = rows

    # the charter's A2029-like number is quoted as a single ~0.4%; it is in
    # fact strongly radius dependent, so report the profile
    scan = []
    for rr in (200 * KPC, 300 * KPC, 500 * KPC, 700 * KPC, 1000 * KPC,
               1500 * KPC, 2000 * KPC):
        e = C.erasure_verdict(
            C.erasure(C.QuasiLinearMOND(), C.SphericalAverage(24), ss, rr),
            target_precision=0.01)
        scan.append({"radius_kpc": rr / KPC,
                     "shift": e["observable_shift"],
                     "verdict": e["verdict"]})
    out["qumond_radius_scan"] = scan
    out["n_refuse"] = sum(r["verdict"] == "REFUSE" for r in rows)
    out["n_allow"] = sum(r["verdict"] == "ALLOW" for r in rows)
    return out


def block_bridge(reg):
    quantities = sorted({q for qs in INV.LAYER_QUANTITIES.values() for q in qs}
                        | set(INV.UNIVERSAL_QUANTITIES))
    inv_index = INV.availability_index([q for q in quantities if q in reg])
    cands = [
        {"name": "newton", "reads": ["M_enc", "r_3d", "G"],
         "nonlinear_of": []},
        {"name": "rar_qumond", "reads": ["g_N", "a0"], "nonlinear_of": ["g_N"]},
        {"name": "rar_with_radius", "reads": ["g_N", "M_enc", "r_3d", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "potential_depth", "reads": ["g_N", "phi_depth_saddle", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "external_axis_tensor",
         "reads": ["g_N", "ext_axis", "alignment_angle", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "source_axis_tensor",
         "reads": ["g_N", "position_angle", "axis_ratio_q", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "well_network",
         "reads": ["g_N", "n_wells", "graph_degree", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "path_law",
         "reads": ["g_N", "path_density", "path_void_fraction", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "memory_law",
         "reads": ["g_N", "field_memory", "t_since_merger", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "matter_light_slip",
         "reads": ["g_N", "phi_slip", "time_delay", "e1", "e2", "a0"],
         "nonlinear_of": ["g_N"]},
        {"name": "turbulent_pressure_source",
         "reads": ["g_N", "sigma_turb", "P_e", "a0"], "nonlinear_of": ["g_N"]},
        {"name": "reads_kappa_as_data", "reads": ["kappa", "r_3d", "a0"],
         "nonlinear_of": []},
        {"name": "raw_temperature_scale", "reads": ["g_N", "T_x"],
         "nonlinear_of": ["T_x"]},
    ]
    res = B.prescreen_many(cands, reg, inv_index)
    tax = {}
    for r in res:
        tax[r["taxonomy"]] = tax.get(r["taxonomy"], 0) + 1
    return {"n_candidates": len(res), "taxonomy": tax, "results": res,
            "n_quantities_indexed": len(inv_index)}


def block_inventory():
    return {
        "clusters": list(INV.CLUSTERS),
        "layers": dict(INV.LAYERS),
        "matrix": INV.MATRIX,
        "layer_counts": INV.layer_counts(),
        "cluster_scores": INV.cluster_scores(),
        "gold_verdict": INV.gold_cluster_verdict(),
        "dm_contaminated": INV.dm_contaminated_products(),
        "acquisition_notes": INV.ACQUISITION_NOTES,
        "status_vocabulary": {
            "RAW_MACHINE_READABLE": "raw observable, tabulated, downloadable",
            "RAW_ARXIV_LATEX_ONLY": "raw, but only inside a paper's LaTeX "
                                    "source -- invisible to a catalogue search",
            "RAW_PIXELS_ONLY": "raw, but no tabulated product exists",
            "DERIVED_UNDER_THEORY": "presupposes a gravity or mass model",
            "PARTIAL": "exists but materially weaker than the charter asks",
            "ABSENT": "confirmed absent; `searched` records what was tried"},
    }


def main():
    t0 = time.perf_counter()
    reg = REG.build_registry()
    g, b, z_true = demo_scene(reg)
    out = {
        "lane": "work/wellnet-2026-09/scene -- Stage 1, the probabilistic "
                "four-dimensional gravitational scene graph",
        "seed": SEED,
        "schema": block_schema(reg, g),
    }
    print("schema block done")
    out["ensemble"] = block_ensemble(g, b, z_true)
    print("ensemble block done")
    out["commutation"] = block_commutation()
    print("commutation block done")
    out["bridge"] = block_bridge(reg)
    print("bridge block done")
    out["inventory"] = block_inventory()
    out["registry"] = reg.to_json()
    tp = os.path.join(HERE, "test_results.json")
    if os.path.exists(tp):
        with open(tp, encoding="utf-8") as fh:
            out["tests"] = json.load(fh)
    out["wall_s"] = time.perf_counter() - t0
    p = os.path.join(HERE, "scene_results.json")
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {p} in {out['wall_s']:.1f}s")
    return out


if __name__ == "__main__":
    main()
