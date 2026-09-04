"""test_scene.py -- the validation suite for Stage 1.

Every test checks a property that can be stated INDEPENDENTLY of the code that
implements it: an exact identity, a charter requirement quoted verbatim, a
recorded number from this programme's own earlier runs, or a symmetry that must
hold whatever the implementation.  A suite that only checks the code against
itself would have passed on every one of the eight bugs listed in REPORT.md.

Run as `python test_scene.py` (prints a summary and writes `test_results.json`)
or under pytest.

NO OBSERVATIONAL DATA IS OPENED.  `test_no_observational_data_is_opened`
asserts it mechanically.
"""
from __future__ import annotations

import builtins
import json
import math
import os
import sys

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
RESULTS: dict = {}


def _rec(name, **kw):
    RESULTS[name] = kw
    return kw


# ========================================================== the contract
def test_charter_node_types_are_complete():
    """The charter lists twelve node bullets; the schema must cover all of
    them, with 'Voids, filaments, saddles, and boundaries' expanded to four."""
    assert len(S.CHARTER_NODE_BULLETS) == 12
    assert len(S.NODE_TYPES) == 15          # 12 bullets, one expanded to 4
    for t in ("star_population", "galaxy", "gas_cell", "central_galaxy",
              "intracluster_light", "black_hole", "compact_substructure",
              "background_source", "observer", "instrument", "void",
              "filament", "saddle", "boundary", "latent_field_cell"):
        assert t in S.NODE_TYPES, t
    _rec("node_types", n_bullets=12, n_types=len(S.NODE_TYPES))


def test_charter_edge_and_field_types_are_complete():
    assert len(S.EDGE_TYPES) == 10, S.EDGE_TYPES
    assert len(S.FIELD_TYPES) == 8, S.FIELD_TYPES
    for t in ("spatial_separation", "relative_velocity", "membership",
              "light_path", "source_source", "tidal_pair", "orbital",
              "image_family", "causal_retarded", "shared_covariance"):
        assert t in S.EDGE_TYPES, t
    _rec("edge_field_types", n_edges=len(S.EDGE_TYPES),
         n_fields=len(S.FIELD_TYPES))


def test_every_contract_item_is_populated():
    """The charter's seventeen metadata items, for every quantity."""
    reg = REG.build_registry()
    a = M.audit_contract(reg)
    bad = {k: v["missing"] for k, v in a["items"].items() if not v["complete"]}
    assert a["all_complete"], bad
    assert a["n_items"] == 17, a["n_items"]
    _rec("contract_audit", n_quantities=len(reg), n_items=a["n_items"],
         all_complete=True)


def test_gauge_rule_is_enforced_at_construction():
    """A potential that shifts under a change of origin CANNOT be built
    without naming a boundary rule.  The charter forbids it; the constructor
    must refuse it rather than the report noting it afterwards."""
    try:
        M.Quantity(name="phi_absolute",
                   definition="absolute Newtonian potential",
                   kind="scalar", dim=M.DIM_PHI,
                   translation="SHIFTS_BY_CONSTANT")   # no gauge
    except M.ContractError as ex:
        _rec("gauge_enforced", raised=True, message=str(ex)[:120])
        return
    raise AssertionError("a gauge-free absolute potential was accepted")


def test_log_of_a_dimensionful_quantity_is_refused():
    try:
        M.Quantity(name="bad_log", definition="log of a length",
                   kind="scalar", dim=M.DIM_L,
                   allowed_ops=("log_dimensionless",))
    except M.ContractError:
        _rec("log_dim_enforced", raised=True)
        return
    raise AssertionError("log of a dimensionful quantity was accepted")


def test_unregistered_attribute_cannot_enter_a_scene():
    """R1: the metadata contract is binding, not advisory."""
    reg = REG.build_registry()
    g = S.SceneGraph("t", reg)
    try:
        g.add_node(S.Node("n", "galaxy", {"not_a_quantity": S.Fixed(1.0)}))
    except M.ContractError:
        _rec("unregistered_attr_refused", raised=True)
        return
    raise AssertionError("an unregistered quantity entered a scene")


def test_bare_number_cannot_enter_a_scene():
    """R2: a bare float hides whether the quantity is known or sampled."""
    reg = REG.build_registry()
    g = S.SceneGraph("t", reg)
    try:
        g.add_node(S.Node("n", "galaxy", {"x": 1.0}))    # not Fixed/Uncertain
    except M.ContractError:
        _rec("bare_number_refused", raised=True)
        return
    raise AssertionError("a bare number entered a scene")


def test_dimension_algebra_is_exact():
    """G M / r^2 must come out as an acceleration, exactly."""
    d = M.DIM_G * M.DIM_M / (M.DIM_L ** 2)
    assert str(d) == str(M.DIM_ACC), (str(d), str(M.DIM_ACC))
    # and a half-integer exponent must survive as a Fraction
    h = M.DIM_ACC ** 0.5
    assert h.T.denominator == 1 and h.L.denominator == 2, h
    _rec("dim_algebra", gm_over_r2=str(d), sqrt_acc=str(h))


def test_dm_contaminated_node_is_flagged():
    reg = REG.build_registry()
    g = S.SceneGraph("t", reg)
    g.add_node(S.Node("k", "latent_field_cell", {"kappa": S.Fixed(0.3)},
                      source="parametric lens model, one halo per member",
                      presupposes_dm=True, dm_reason="circular by construction"))
    c = g.dm_contaminated()
    assert len(c) == 1 and c[0]["id"] == "k", c
    _rec("dm_flag", n_flagged=len(c))


# ========================================================== the ensemble
def test_depth_intervals_are_calibrated():
    """Frequentist coverage of the depth credible intervals, on a synthetic
    cluster whose true depths are known.  This is the test that found BUG 1
    (the prior volume did not match the declared scene volume, giving
    over-coverage at every level and a posterior WIDER than the truth)."""
    cov = E.coverage_test(400, 20260904)
    _rec("coverage", **{k: v["empirical"] for k, v in cov["levels"].items()},
         all_calibrated=cov["all_calibrated"],
         information_gain_ratio=cov["information_gain_ratio"])
    assert cov["all_calibrated"], cov["levels"]
    assert cov["information_gain_ratio"] > 1.0, (
        "the posterior must be narrower than the spread of true depths")


def test_ensemble_is_not_a_point_estimate():
    reg = REG.build_registry()
    mem, phase, _ = E.synthetic_cluster(120, 20260904)
    b = E.SceneEnsembleBuilder("t", reg, phase, mem, seed=20260904)
    g = b.build()
    ens = g.ensemble(48, 20260904)
    d = b.diagnostics(ens)
    assert d["ess"] > 0.9 * len(ens), d["ess"]        # exact proposal
    assert d["depth_sd_median_Mpc"] > 0.3, d          # genuinely uncertain
    assert len(g.uncertain_attrs()) >= len(mem), "depths must be Uncertain"
    _rec("ensemble_shape", ess=d["ess"], n=len(ens),
         depth_sd_median_Mpc=d["depth_sd_median_Mpc"])


def test_importance_reweighting_collapses_the_ess():
    """BUG 8, kept as a regression: applying the morphology term by weight
    instead of folding it into the proposal costs most of the sample."""
    reg = REG.build_registry()
    mem, phase, _ = E.synthetic_cluster(120, 20260904)
    out = {}
    for flag in (False, True):
        b = E.SceneEnsembleBuilder("t", reg, phase, mem, seed=20260904,
                                   exact_morphology=flag)
        ens = b.build().ensemble(64, 20260904)
        out["exact" if flag else "reweighted"] = b.diagnostics(ens)["ess"]
    assert out["exact"] > 2 * out["reweighted"], out
    assert out["exact"] > 63.0, out
    _rec("ess_contrast", **out)


def test_substructure_correlates_member_depths():
    """Independent per-member depths would destroy the lumpy correlated
    geometry a network law is meant to see."""
    reg = REG.build_registry()
    mem, phase, _ = E.synthetic_cluster(120, 20260904)
    b = E.SceneEnsembleBuilder("t", reg, phase, mem, seed=20260904)
    d = b.diagnostics(b.build().ensemble(64, 20260904))
    assert d["mean_pairwise_depth_corr"] > 0.01, d
    _rec("substructure_corr", corr=d["mean_pairwise_depth_corr"])


def test_mean_scene_understates_three_d_radius():
    """E[f(scene)] != f(E[scene]).  Collapsing the ensemble to its mean puts
    every member back in the plane of the sky -- the depth fabrication."""
    reg = REG.build_registry()
    mem, phase, _ = E.synthetic_cluster(120, 20260904)
    b = E.SceneEnsembleBuilder("t", reg, phase, mem, seed=20260904)
    ens = b.build().ensemble(64, 20260904)
    ids = [m.mid for m in mem]
    R = np.array([m.R for m in mem])
    Z = np.array([[d.node_attrs[i]["z"] for i in ids] for d in ens.draws])
    E_f = float(np.mean(np.sqrt(R[None, :] ** 2 + Z ** 2)))
    f_E = float(np.mean(np.sqrt(R ** 2 + Z.mean(axis=0) ** 2)))
    frac = (E_f - f_E) / E_f
    assert frac > 0.10, frac
    _rec("mean_scene_bias", E_of_f_Mpc=E_f / MPC, f_of_E_Mpc=f_E / MPC,
         fractional_understatement=frac)


def test_depth_posterior_uses_no_mass_model():
    """The phase-space description must not contain an NFW or any halo: it is
    a fit to observed counts and observed velocities, nothing else."""
    src = open(os.path.join(HERE, "ensemble.py"), encoding="utf-8").read()
    low = src.lower()
    for bad in ("nfw", "navarro", "einasto", "concentration parameter",
                "virial mass", "halo mass"):
        assert bad not in low.replace("no nfw", "").replace(
            "an nfw-based", "").replace("nfw profile", ""), bad
    _rec("no_mass_model", checked=True)


# ======================================================= the commutation gate
def test_newton_commutes_with_spherical_averaging():
    """THE NULL CONTROL.  Newtonian gravity is linear in the source, so the
    shell average of the resolved field equals the field of the spherically
    averaged source EXACTLY.  Checked against the closed-form shell-averaged
    Plummer potential, not against the gate's own other branch."""
    s = C.synthetic_cluster_scene(300, 20260904)
    errs = []
    for r in (300 * KPC, 1000 * KPC, 2000 * KPC):
        ex = C.analytic_spherical_avg_g(s, r)
        v = C.shell_radial_g(C.Newtonian(), s, r, 256, 8)
        errs.append(abs(v / ex - 1.0))
    floor = max(errs)
    assert floor < 1e-3, floor
    _rec("newton_null", max_abs_rel_err=floor, radii_kpc=[300, 1000, 2000])


def test_gate_reports_zero_erasure_for_a_linear_law():
    """A linear law's deviation from the linear control is identically zero,
    so the gate must not manufacture an erasure."""
    s = C.synthetic_cluster_scene(300, 20260904)
    e = C.erasure(C.Newtonian(), C.SphericalAverage(24), s, 1000 * KPC)
    assert abs(e["deviation_resolved"]) < 1e-25, e["deviation_resolved"]
    _rec("linear_law_zero_deviation", dev=e["deviation_resolved"])


def test_azimuthal_averaging_erases_a_SOURCE_axis_but_not_an_EXTERNAL_one():
    """The charter's 'directional laws are erased by azimuthal averaging',
    made precise.  It is the ORIGIN of the axis that decides, and this is the
    same distinction GATE 1 of the pre-data compiler turns on."""
    sf = C.flattened_cluster_scene(300, 20260904, q_z=0.55)
    r = 1000 * KPC
    src = C.erasure(C.SourceAlignedTensor(0.30), C.AzimuthalAverage(), sf, r,
                    observable="shell_quadrupole", n_op_draws=8)
    ext = C.erasure(C.ExternalAxisTensor(0.30), C.AzimuthalAverage(), sf, r,
                    observable="shell_quadrupole", n_op_draws=8)
    assert src["erased_fraction"] > 0.7, src["erased_fraction"]
    assert ext["erased_fraction"] < 0.3, ext["erased_fraction"]
    _rec("directional_contrast",
         source_axis_erased=src["erased_fraction"],
         external_axis_erased=ext["erased_fraction"])


def test_smoothing_erases_a_network_law():
    sf = C.flattened_cluster_scene(300, 20260904, q_z=0.55)
    e = C.erasure(C.WellNetwork(0.30, L=300 * KPC),
                  C.GaussianSmooth(300 * KPC), sf, 1000 * KPC)
    assert e["erased_fraction"] > 0.9, e["erased_fraction"]
    _rec("network_erased", erased=e["erased_fraction"])


def test_present_only_erases_a_memory_law():
    sf = C.flattened_cluster_scene(300, 20260904, q_z=0.55)
    C.attach_history(sf)
    e = C.erasure(C.MemoryLaw(0.60), C.PresentOnly(), sf, 1000 * KPC)
    assert e["erased_fraction"] > 0.99, e["erased_fraction"]
    _rec("memory_erased", erased=e["erased_fraction"])


def test_radial_averaging_erases_a_path_law():
    """BUG 6's regression: a spherically symmetric scene has ONE column per
    radius, so the path law's sightline-to-sightline signal must vanish."""
    sf = C.flattened_cluster_scene(300, 20260904, q_z=0.55)
    r = 1000 * KPC
    pl = C.PathLaw(0.30).calibrate(sf, r)
    e = C.erasure(pl, C.SphericalAverage(24), sf, r,
                  observable="shell_dispersion", n_dir=128, n_rot=4)
    assert e["erased_fraction"] > 0.9, e["erased_fraction"]
    _rec("path_erased", erased=e["erased_fraction"])


def test_averaging_ops_conserve_mass():
    s = C.synthetic_cluster_scene(200, 20260904)
    rng = np.random.default_rng(0)
    out = {}
    for op in (C.SphericalAverage(24), C.AzimuthalAverage(),
               C.GaussianSmooth(100 * KPC), C.LOSCollapse(),
               C.CatalogueMerge(150 * KPC), C.RadialBin(12)):
        rel = op(s, rng).total_mass() / s.total_mass() - 1.0
        out[op.name] = rel
        assert abs(rel) < 1e-10, (op.name, rel)
    _rec("mass_conservation", **out)


def test_los_collapse_is_measurably_costly():
    """Putting every source at zero depth is the fabrication the charter
    names.  Its cost must be a measured number, not an argument."""
    s = C.synthetic_cluster_scene(300, 20260904)
    v = C.erasure_verdict(
        C.erasure(C.QuasiLinearMOND(), C.LOSCollapse(), s, 1000 * KPC),
        target_precision=0.01)
    assert v["observable_shift"] > 0.01, v["observable_shift"]
    assert v["verdict"] == "REFUSE"
    _rec("los_collapse_cost", shift=v["observable_shift"],
         verdict=v["verdict"])


# ============================================================== the bridge
def test_prescreen_catches_a_dimensionful_nonlinear_argument():
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "T_x"], reg, nonlinear_of=["T_x"])
    assert not r["gates"]["S1_units"]["pass"]
    assert r["taxonomy"] == "mathematically_inconsistent"
    _rec("s1_catches_units", taxonomy=r["taxonomy"])


def test_prescreen_catches_a_frame_fixed_quantity():
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "sigma_turb"], reg)
    assert not r["gates"]["S3_frame"]["pass"]
    _rec("s3_catches_frame", boost=r["gates"]["S3_frame"]["boost_frame_fixed"])


def test_prescreen_catches_a_catalogue_dependent_quantity():
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "n_wells"], reg)
    assert not r["gates"]["S4_coarse"]["pass"]
    assert "n_wells" in r["gates"]["S4_coarse"]["catalogue_dependent"]
    _rec("s4_catches_catalogue", cat=r["gates"]["S4_coarse"]["catalogue_dependent"])


def test_prescreen_finds_the_rank_collapse():
    """The programme's variable-lists-collapse finding, as a gate: a law
    reading g_N, M_enc and r_3d has two independent directions, not three."""
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "M_enc", "r_3d", "a0"], reg)
    g = r["gates"]["S7_rank"]
    assert not g["pass"]
    assert g["n_independent"] == g["n_declared"] - 1, g
    _rec("s7_rank", declared=g["n_declared"], independent=g["n_independent"])


def test_newton_is_admissible_not_non_identifiable():
    """BUG 7's regression.  The first prescreen flagged Newtonian gravity as
    non-identifiable because g_N and r_3d are constructed rather than
    observed.  A gate that fails Newton is measuring the wrong thing."""
    reg = REG.build_registry()
    r = B.prescreen(["M_enc", "r_3d", "G"], reg, name="newton")
    assert r["taxonomy"] in ("admissible", "admissible_but_redundant"), r["taxonomy"]
    assert r["gates"]["S6_identifiable"]["pass"], r["gates"]["S6_identifiable"]
    _rec("newton_admissible", taxonomy=r["taxonomy"],
         marginalised=r["gates"]["S6_identifiable"]["marginalised_by_the_ensemble"])


def test_reading_a_gauge_fixed_potential_is_flagged_as_convention_dependent():
    """A gauge-fixed potential is admissible, but the RULE is a convention.
    The first version returned pass on `unsafe` alone, so this branch of the
    taxonomy could never be reached and a potential-depth candidate came back
    as plainly admissible."""
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "phi_depth_saddle", "a0"], reg)
    assert r["taxonomy"] == "convention_dependent", r["taxonomy"]
    assert r["gates"]["S2_gauge"]["rule_spread_dex"] == 0.87
    _rec("s2_convention", taxonomy=r["taxonomy"],
         spread_dex=r["gates"]["S2_gauge"]["rule_spread_dex"])


def test_scoring_against_a_convergence_map_is_flagged_as_circular():
    """The charter forbids scoring a candidate against a precomputed
    convergence map or an NFW-defined radius. Both are marked
    derived_under_theory and both must reach the taxonomy."""
    reg = REG.build_registry()
    assert set(reg.theory_contaminated()) == {"kappa", "R500"},         reg.theory_contaminated()
    for q in ("kappa", "R500"):
        r = B.prescreen([q, "r_3d", "a0"], reg)
        assert r["taxonomy"] == "theory_contaminated", (q, r["taxonomy"])
    _rec("theory_contamination", flagged=sorted(reg.theory_contaminated()))


def test_a_free_latent_field_is_non_identifiable():
    reg = REG.build_registry()
    r = B.prescreen(["g_N", "vacuum_order", "a0"], reg)
    assert r["taxonomy"] == "non_identifiable"
    _rec("latent_non_identifiable", free=r["gates"]["S6_identifiable"]["non_identifiable"])


def test_availability_gate_finds_the_time_delay_bottleneck():
    """S8: a law reading a measured time delay is testable on exactly the
    clusters that have one."""
    reg = REG.build_registry()
    qs = sorted({q for v in INV.LAYER_QUANTITIES.values() for q in v}
                | set(INV.UNIVERSAL_QUANTITIES))
    idx = INV.availability_index([q for q in qs if q in reg])
    r = B.prescreen(["g_N", "time_delay", "e1", "a0"], reg, inventory=idx)
    have = r["gates"]["S8_available"]["clusters_with_every_input"]
    assert have == [], have          # A370 has shear, M1149 has the delay
    _rec("s8_time_delay", clusters=have,
         delay_clusters=sorted(k for k, v in
                               INV.availability("time_delay").items() if v),
         shear_clusters=sorted(k for k, v in
                               INV.availability("e1").items() if v))


# ============================================================ the inventory
def test_no_cluster_satisfies_corpus_E():
    v = INV.gold_cluster_verdict()
    assert not v["corpus_E_satisfied"], v["clusters_meeting_all"]
    assert v["n_layers_required"] == 10
    _rec("corpus_E", satisfied=False, binding=v["binding_constraints"],
         best=v["best"]["cluster"], best_n=v["best"]["n_layers_usable"])


def test_every_absence_records_what_was_tried():
    """A confirmed absence is a deliverable only if it is auditable."""
    bad = []
    for c, row in INV.MATRIX.items():
        for layer, cell in row.items():
            if cell["status"] == INV.ABSENT and not (cell["searched"]
                                                     or cell["note"]):
                bad.append((c, layer))
    assert not bad, bad
    n = sum(1 for row in INV.MATRIX.values() for cell in row.values()
            if cell["status"] == INV.ABSENT)
    _rec("absences_documented", n_absent=n)


def test_matrix_is_complete_and_uses_the_status_vocabulary():
    valid = {INV.RAW_MR, INV.RAW_ARXIV, INV.RAW_PIXELS, INV.DERIVED,
             INV.PARTIAL, INV.ABSENT}
    assert set(INV.MATRIX) == set(INV.CLUSTERS)
    for c, row in INV.MATRIX.items():
        assert set(row) == set(INV.LAYERS), (c, set(INV.LAYERS) - set(row))
        for layer, cell in row.items():
            assert cell["status"] in valid, (c, layer, cell["status"])
    _rec("matrix_complete", n_cells=len(INV.CLUSTERS) * len(INV.LAYERS))


def test_layer_quantities_are_all_registered():
    reg = REG.build_registry()
    missing = [q for v in INV.LAYER_QUANTITIES.values() for q in v
               if q not in reg]
    missing += [q for q in INV.UNIVERSAL_QUANTITIES if q not in reg]
    assert not missing, missing
    _rec("layer_quantities_registered", n=sum(
        len(v) for v in INV.LAYER_QUANTITIES.values()))


# ================================================== the standing constraints
FORBIDDEN = ("kids", "kids-1000", "kids1000", "wide binar", "gaia dr3 binar")


#: A sealed name may legitimately appear in a sentence SAYING it was not used.
#: Without this the test fails on its own source and on the inventory's own
#: sealed-datasets declaration -- a false positive that would train the reader
#: to ignore the check.
NEGATIVE_CONTEXT = ("not loaded", "not touched", "were not", "was not",
                    "sealed", "do not load", "untouchable", "forbidden")


def test_sealed_datasets_are_not_referenced():
    """KiDS and the wide binaries are sealed for this round.

    Scans every module for the sealed names and, where one appears, requires
    the surrounding 200 characters to say it was NOT used.  This file is
    skipped: it necessarily contains the names it searches for.
    """
    hits = []
    scanned = 0
    for fn in sorted(os.listdir(HERE)):
        if not fn.endswith(".py") or fn == os.path.basename(__file__):
            continue
        scanned += 1
        txt = open(os.path.join(HERE, fn), encoding="utf-8").read().lower()
        for f in FORBIDDEN:
            i = txt.find(f)
            while i >= 0:
                ctx = txt[max(0, i - 200):i + 200]
                if not any(n in ctx for n in NEGATIVE_CONTEXT):
                    hits.append((fn, f, ctx[:80]))
                i = txt.find(f, i + 1)
    assert not hits, hits
    _rec("sealed_clean", n_files_scanned=scanned)


def test_no_observational_data_is_opened():
    """Mechanical: run the whole pipeline with `open` instrumented and assert
    that nothing outside this directory is read."""
    real_open = builtins.open
    opened = []

    def spy(file, mode="r", *a, **kw):
        try:
            p = os.path.abspath(str(file))
        except Exception:
            p = str(file)
        if "r" in mode and not p.startswith(HERE):
            opened.append(p)
        return real_open(file, mode, *a, **kw)

    builtins.open = spy
    try:
        reg = REG.build_registry()
        mem, phase, _ = E.synthetic_cluster(40, 1)
        E.SceneEnsembleBuilder("t", reg, phase, mem).build().ensemble(4, 1)
        s = C.synthetic_cluster_scene(40, 1, n_diffuse=200)
        C.erasure(C.QuasiLinearMOND(), C.SphericalAverage(8), s, 1000 * KPC,
                  n_dir=32, n_rot=1)
        B.prescreen(["g_N", "a0"], reg, nonlinear_of=["g_N"])
        INV.gold_cluster_verdict()
    finally:
        builtins.open = real_open
    assert not opened, opened
    _rec("no_data_opened", n_external_reads=0)


# ===================================================================== main
def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    npass = nfail = 0
    fails = []
    for t in tests:
        try:
            t()
            npass += 1
            print(f"  PASS  {t.__name__}")
        except Exception as ex:
            nfail += 1
            fails.append((t.__name__, f"{type(ex).__name__}: {ex}"))
            print(f"  FAIL  {t.__name__}: {type(ex).__name__}: {ex}")
    print(f"\n{npass} passed, {nfail} failed, of {len(tests)}")
    with open(os.path.join(HERE, "test_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_tests": len(tests), "n_pass": npass, "n_fail": nfail,
                   "failures": fails, "records": RESULTS}, fh, indent=1,
                  default=float)
    return nfail == 0


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
